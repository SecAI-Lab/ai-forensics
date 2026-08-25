"""The Jacobian lens — decoding a layer through its measured effect on the output.

    Wes Gurnee*, Nicholas Sofroniew*, Adam Pearce, Mateusz Piotrowski, Isaac
    Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang,
    T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen,
    Joshua Batson, Jack Lindsey*. "Verbalizable Representations Form a Global
    Workspace in Language Models." Transformer Circuits Thread, 6 July 2026;
    arXiv:2607.15495. Code: github.com/anthropics/jacobian-lens

The logit lens decodes an intermediate state as if the remaining layers were the
identity. The Jacobian lens replaces that assumption with a measurement: the
average linear effect the remaining layers actually have,

    J_l  = E_{t, t'>=t, prompt} [ d h_final,t' / d h_l,t ]
    lens(h_l) = softmax( W_U . norm( J_l h_l ) )

so the logit lens is exactly the special case ``J_l = I``.

**Two things about how this is used here, both from the paper itself.**

*Its KL to the final distribution is worse than the logit lens's, by design.*
The authors call that a feature: the lens is meant to surface what a layer
contributes, not to predict the output. Building a KL-trajectory detector on it
would be a category error, so this module does not offer one.

*The published screen is model-level, not per-sample.* Its use in an alignment
context is to read the top-k tokens at response-start on *ordinary, untriggered*
prompts and look for content that should not be there. That answers "is this
model tampered with", which is a different question from "is this input
triggered" and cannot be scored on ``#detected poisoned / #poisoned``. It is
reported in its own table.

**The estimator is full-rank, after a low-rank one was tried and rejected.**
``J_l`` is ``hidden x hidden`` — 4.2M entries at 2048 — so an obvious economy is
to estimate it inside a random ``r``-dimensional subspace. That does not work,
and the lens's own defining property is what exposes it: setting ``J_l = I``
must reduce this to the logit lens, and a rank-512 random projection of a
2048-dimensional state discards three quarters of it, so the reduction fails and
the readout is noise. Rank is not a quality knob here; it is correctness.

The full sweep is affordable anyway. Perturbing one coordinate at a time and
re-running only blocks ``l..end`` costs 24 ms per batch of 8 at layer 14, so all
2049 passes take about 50 s per batch — under a minute for the ~100 prompts the
paper reports as sufficient.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from .lens import _resolve_blocks, _resolve_readout

logger = logging.getLogger(__name__)

__all__ = ["JacobianLens"]


class JacobianLens:
    """Estimate ``J_l`` on a clean corpus, then decode states through it.

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        layer: Layer whose state is decoded (``1`` = first block output).
        rank: Retained only so an existing config does not break; the
            estimator is full-rank and ignores it.
        seed: Unused by the full-rank estimator.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        layer: int = 14,
        rank: int = 64,
        seed: int = 1337,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(_resolve_blocks(model))
        self.layer = int(layer) if layer >= 0 else self.n_layers + 1 + int(layer)
        self.rank = int(rank)
        self.seed = int(seed)
        self._basis: Any = None      # (rank, hidden) orthonormal directions
        self._images: Any = None     # (rank, hidden) their images under J
        self.n_prompts_fitted = 0

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int | None = 128,
        progress: bool = True,
    ) -> JacobianLens:
        """Estimate the operator on a **clean** corpus.

        Fitting on the data under investigation would fold whatever the backdoor
        does into the very operator meant to expose it.

        Args:
            texts: Clean prompts.
            batch_size: Prompts per pass.
            max_length: Truncation budget.
            progress: Log progress.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        import torch

        from ..models.llm import encode_texts, model_device

        if not len(texts):
            raise ValueError("texts is empty; the Jacobian lens needs a clean corpus")

        device = model_device(self.model)
        d = int(self.model.config.hidden_size)
        blocks = _resolve_blocks(self.model)
        # Row j holds column j of E[J]: the change in the final residual produced
        # by moving coordinate j of the layer state. No basis is needed — the
        # standard basis IS the estimator, which is what makes J = I exact.
        images = torch.zeros(d, d, device=device, dtype=torch.float32)
        seen = 0

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            enc = encode_texts(self.tokenizer, chunk, padding_side="right",
                               max_length=max_length)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            captured: dict[str, Any] = {}

            def grab(_mod, _inp, out, captured=captured):
                captured["h"] = out[0] if isinstance(out, tuple) else out

            handle = blocks[self.layer - 1].register_forward_hook(grab)
            try:
                with torch.no_grad():
                    self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                handle.remove()
            h0 = captured["h"].detach()

            base_out = self._run_from(h0, ids, mask).float()
            m = mask.unsqueeze(-1).float()
            denom = m.sum().clamp(min=1.0)

            # One-sided differences against a shared baseline: 2048 extra passes
            # instead of 4096. eps is large enough to survive bfloat16 rounding
            # and small enough that the map is still locally linear.
            eps = 0.1
            for j in range(d):
                bumped = h0.clone()
                bumped[:, :, j] += eps
                delta = (self._run_from(bumped, ids, mask).float() - base_out) / eps
                # Average over real token positions only; pad positions carry
                # arbitrary activations and would bias every column.
                images[j] += (delta * m).sum(dim=(0, 1)) / denom
            seen += 1
            if progress:
                logger.info("jacobian lens: batch %d/%d",
                            seen, (len(texts) + batch_size - 1) // batch_size)

        self._images = images / max(seen, 1)
        self._basis = None                      # full rank: no projection
        self.n_prompts_fitted = len(texts)
        return self

    def _run_from(self, hidden: Any, ids: Any, mask: Any) -> Any:
        """Run blocks ``layer..end`` from a supplied hidden state."""
        import torch

        blocks = _resolve_blocks(self.model)
        base = getattr(self.model, "base_model", self.model)
        base = getattr(base, "model", base)
        inner = getattr(base, "model", base)

        h = hidden
        with torch.no_grad():
            pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            rot = None
            emb = getattr(inner, "rotary_emb", None)
            if emb is not None:
                rot = emb(h, pos)
            causal = _causal_mask(mask, h.dtype)
            for blk in blocks[self.layer:]:
                out = blk(h, attention_mask=causal, position_ids=pos,
                          position_embeddings=rot)
                h = out[0] if isinstance(out, tuple) else out
            # PRE-norm on purpose: lens(h) = softmax(W_U . norm(J h)) applies the
            # norm to the operator's OUTPUT. Folding it into the derivative
            # instead estimates d(norm(h_final))/dh, a different operator.
            return h

    # -- decoding ----------------------------------------------------------

    def readout(
        self,
        texts: Sequence[str],
        *,
        top_k: int = 10,
        batch_size: int = 8,
        max_length: int | None = 128,
    ) -> list[list[tuple[str, float]]]:
        """Top-``top_k`` tokens the operator surfaces at the last real position.

        This is the paper's model-level screen: run it on **ordinary,
        untriggered** prompts and look for tokens that have no business being
        there.

        Returns:
            One ``[(token, probability), ...]`` list per input.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        import torch
        import torch.nn.functional as F

        from ..models.llm import encode_texts, model_device

        if self._images is None:
            raise RuntimeError("JacobianLens.readout() before fit()")

        norm, head = _resolve_readout(self.model)
        blocks = _resolve_blocks(self.model)
        device = model_device(self.model)
        out: list[list[tuple[str, float]]] = []

        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            enc = encode_texts(self.tokenizer, chunk, padding_side="right",
                               max_length=max_length)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)

            captured: dict[str, Any] = {}

            def grab(_m, _i, o, captured=captured):
                captured["h"] = o[0] if isinstance(o, tuple) else o

            handle = blocks[self.layer - 1].register_forward_hook(grab)
            try:
                with torch.no_grad():
                    self.model(input_ids=ids, attention_mask=mask, use_cache=False)
            finally:
                handle.remove()

            pos = (mask.sum(dim=1).long() - 1).clamp(min=0)
            rows = torch.arange(ids.shape[0], device=device)
            h = captured["h"][rows, pos].float()             # (b, d)
            jh = h @ self._images                             # (b, d)
            with torch.no_grad():
                dtype = next(self.model.parameters()).dtype
                logits = head(norm(jh.to(dtype))).float()
                probs = F.softmax(logits, dim=-1)
                p, idx = probs.topk(top_k, dim=-1)
            for r in range(h.shape[0]):
                out.append([
                    (self.tokenizer.decode([int(i)]), float(v))
                    for i, v in zip(idx[r].tolist(), p[r].tolist(), strict=True)
                ])
        return out

    def token_screen(
        self,
        texts: Sequence[str],
        watch_tokens: Sequence[str],
        *,
        top_k: int = 10,
        batch_size: int = 8,
    ) -> dict[str, Any]:
        """Fraction of clean prompts whose readout surfaces a watched token.

        The published misalignment screen, made concrete. A **model-level**
        statistic: one number per model, not per input.

        Args:
            texts: Ordinary, untriggered prompts.
            watch_tokens: Strings that should not appear — for a label backdoor,
                the attacker's target verbalisation.
            top_k: Readout depth.
            batch_size: Forward batch size.

        Returns:
            ``rate`` — fraction of prompts surfacing any watched token;
            ``hits`` — per-token counts; ``n`` — prompts screened;
            ``rank`` and ``n_prompts_fitted`` — the estimator's provenance,
            since a low-rank result is not comparable to a higher-rank one.
        """
        want = [w.strip().lower() for w in watch_tokens]
        reads = self.readout(texts, top_k=top_k, batch_size=batch_size)
        hits = dict.fromkeys(want, 0)
        flagged = 0
        for row in reads:
            toks = [t.strip().lower() for t, _ in row]
            got = False
            for w in want:
                if any(w and w in t for t in toks):
                    hits[w] += 1
                    got = True
            flagged += int(got)
        return {
            "rate": flagged / max(len(reads), 1),
            "hits": hits,
            "n": len(reads),
            "rank": "full",
            "n_prompts_fitted": self.n_prompts_fitted,
        }


def _causal_mask(attention_mask: Any, dtype: Any) -> Any:
    """Additive causal mask combined with the padding mask."""
    import torch

    b, t = attention_mask.shape
    causal = torch.full((t, t), torch.finfo(dtype).min, device=attention_mask.device,
                        dtype=dtype)
    causal = torch.triu(causal, diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(b, 1, t, t).clone()
    pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * torch.finfo(dtype).min
    return causal + pad
