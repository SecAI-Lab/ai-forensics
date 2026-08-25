"""Lenses — reading a transformer's intermediate layers in vocabulary space.

A decoder-only transformer refines one residual stream across its layers and
only decodes it at the end. A *lens* decodes it early: it maps the residual
stream after layer L to a distribution over the vocabulary, so the model's
"current best guess" can be read at every depth instead of only at the output.

Two are implemented, and the difference between them is the whole reason the
second exists:

* :class:`LogitLens` reuses the model's own final norm and unembedding on an
  intermediate state. It needs no training, but it is applying a readout to
  states it was never fitted to, so its distributions are systematically
  distorted — badly in early layers, and unevenly across model families.

      nostalgebraist. "interpreting GPT: the logit lens." LessWrong, 2020.
      https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
      (a blog post, not a peer-reviewed paper — cited as such deliberately)

* :class:`TunedLens` learns one affine map per layer, trained so the decoded
  distribution matches the model's *final* distribution in KL. It corrects the
  distortion the logit lens suffers, at the cost of a training pass.

      Nora Belrose et al. "Eliciting Latent Predictions from Transformers with
      the Tuned Lens." arXiv:2303.08112

Why this belongs in a forensics testbed: a backdoored model has to route a
triggered input to the attacker's target regardless of content. That routing has
to happen *somewhere* in depth, and a lens is what makes "where" observable. The
detector built on top of this lives in
:mod:`aiforensics.detectors.neuron_activation`; this module only provides the
readout.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["LensReadout", "LogitLens", "TunedLens", "lens_trajectory"]


@dataclass
class LensReadout:
    """Per-layer vocabulary distributions for a batch of inputs.

    Attributes:
        layers: Layer indices decoded, ascending. Index ``i`` is the output of
            transformer block ``i``; the final entry is the model's own output
            layer, whose row is the true final distribution.
        logprobs: ``(n_texts, n_layers, vocab)`` log-probabilities, or ``None``
            when only reduced statistics were kept (the full tensor is
            ``n * L * V`` floats and overflows memory quickly — for a 1.7B model
            with 151k vocab and 28 layers that is 17 MB *per input*).
        target_logprob: ``(n_texts, n_layers)`` log-probability of the tracked
            token at each layer.
        top1: ``(n_texts, n_layers)`` argmax token id at each layer.
        entropy: ``(n_texts, n_layers)`` distribution entropy in nats.
        final_top1: ``(n_texts,)`` the model's own final argmax token.
        kl_to_final: ``(n_texts, n_layers)`` ``KL(p_final || q_layer)`` — how far
            each layer's decoded distribution is from the model's own output.
            Continuous, unlike the argmax-based statistics, which matters: a
            discrete statistic cannot be thresholded to an arbitrary alarm
            budget because many clean samples tie at the same value.
    """

    layers: tuple[int, ...]
    target_logprob: np.ndarray
    top1: np.ndarray
    entropy: np.ndarray
    final_top1: np.ndarray
    kl_to_final: np.ndarray | None = None
    logprobs: np.ndarray | None = None

    def prediction_depth(self) -> np.ndarray:
        """First layer index (as a position, not a layer id) after which the
        top-1 token equals the final top-1 token and never changes again.

        This is the "when does the answer lock in" statistic. A trigger that
        short-circuits the computation should lock in earlier than content-driven
        reasoning does, which is the signal a backdoor leaves in depth.

        Returns:
            ``(n_texts,)`` integer positions in ``layers``. A row whose top-1
            never settles returns ``len(layers) - 1``.
        """
        n, L = self.top1.shape
        out = np.full(n, L - 1, dtype=np.int64)
        matches = self.top1 == self.final_top1[:, None]
        for i in range(n):
            depth = L - 1
            for j in range(L - 1, -1, -1):
                if matches[i, j]:
                    depth = j
                else:
                    break
            out[i] = depth
        return out


def _resolve_blocks(model: Any) -> Any:
    """Find the transformer block list on a HF causal LM, adapter or not."""
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)          # peft wrapper
    for path in ("model.layers", "layers", "transformer.h", "model.decoder.layers"):
        obj: Any = base
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError(
        f"could not locate transformer blocks on {type(model).__name__}; "
        "expected one of model.layers / layers / transformer.h"
    )


def _resolve_readout(model: Any) -> tuple[Any, Any]:
    """Return ``(final_norm, lm_head)`` — the model's own output decoder."""
    base = getattr(model, "base_model", model)
    base = getattr(base, "model", base)
    inner = getattr(base, "model", base)
    norm = getattr(inner, "norm", None) or getattr(inner, "final_layernorm", None) \
        or getattr(inner, "ln_f", None)
    head = getattr(model, "lm_head", None) or getattr(base, "lm_head", None)
    if norm is None or head is None:
        raise AttributeError(
            f"could not locate final norm / lm_head on {type(model).__name__}"
        )
    return norm, head


def _final_state_is_normed(model: Any, tokenizer: Any) -> bool:
    """Whether ``output_hidden_states[-1]`` already had the final norm applied.

    Decided by measurement, not by version sniffing: run one tiny input and see
    which of ``lm_head(h)`` and ``lm_head(norm(h))`` reproduces the model's own
    logits. Returns ``False`` if the probe cannot run, which reproduces the
    classic logit-lens behaviour.
    """
    import torch

    try:
        norm, head = _resolve_readout(model)
        enc = tokenizer("probe", return_tensors="pt")
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            last = out.hidden_states[-1]
            plain = (head(last).float() - out.logits.float()).abs().max().item()
            normed = (head(norm(last)).float() - out.logits.float()).abs().max().item()
        return bool(plain < normed)
    except Exception:
        logger.warning(
            "could not probe the final-hidden-state convention; assuming it is "
            "pre-norm (classic logit lens). If the last layer disagrees with the "
            "model's own prediction, this is why.", exc_info=True,
        )
        return False


class LogitLens:
    """Decode intermediate residual streams with the model's own output head.

    Args:
        model: A causal LM (a PEFT-wrapped one is fine).
        tokenizer: Its tokenizer.
        layers: Layer indices to decode; ``None`` decodes every block output.
    """

    def __init__(self, model: Any, tokenizer: Any, layers: Sequence[int] | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(_resolve_blocks(model))
        self.layers = tuple(
            range(1, self.n_layers + 1) if layers is None else sorted(layers)
        )
        self._final_is_normed = _final_state_is_normed(model, tokenizer)

    def _decode(self, hidden: Any, layer: int) -> Any:
        """Apply the model's final norm and unembedding to a hidden state.

        The final entry of ``output_hidden_states`` is **already normalised** in
        current transformers (Qwen3Model applies ``self.norm`` before appending
        it), while every earlier entry is a raw block output. Normalising the
        last one again is not a rounding-level error: measured on Qwen3-1.7B it
        moves the logits by 26.4 against a logit scale of 23.6 and *changes the
        argmax*, which silently corrupts any statistic defined relative to the
        model's true final prediction. The convention is detected once at
        construction rather than assumed, because it is a transformers
        implementation detail and has changed before.

        ``layer`` selects the readout; :class:`TunedLens` overrides this to
        apply a per-layer translator.
        """
        import torch

        norm, head = _resolve_readout(self.model)
        with torch.no_grad():
            if self._final_is_normed and layer == self.n_layers:
                return head(hidden)
            return head(norm(hidden))

    def read(
        self,
        texts: Sequence[str],
        *,
        target_token_ids: Sequence[int] | None = None,
        batch_size: int = 8,
        max_length: int | None = None,
        position: int = -1,
    ) -> LensReadout:
        """Decode every selected layer at one token position.

        Args:
            texts: Inputs. For a classification prompt these are the rendered
                prompts, and ``position=-1`` reads the position that predicts
                the label token — which is the one the backdoor must control.
            target_token_ids: Token to track per text (e.g. the attacker's
                target label token). ``None`` tracks each text's own final
                top-1, which needs no label knowledge and is therefore the
                variant usable when the target is unknown.
            batch_size: Texts per forward pass.
            max_length: Truncation budget.
            position: Token position to decode; ``-1`` is the last real token.

        Returns:
            A :class:`LensReadout`. The full ``logprobs`` tensor is not
            retained — see the note on its size in that class.
        """
        import torch
        import torch.nn.functional as F

        from ..models.llm import encode_texts, model_device

        if not len(texts):
            raise ValueError("texts is empty")

        enc = encode_texts(self.tokenizer, list(texts), padding_side="right",
                           max_length=max_length)
        ids_all, mask_all = enc["input_ids"], enc["attention_mask"]
        device = model_device(self.model)
        L = len(self.layers)

        tgt_lp = np.zeros((len(texts), L), dtype=np.float64)
        top1 = np.zeros((len(texts), L), dtype=np.int64)
        ent = np.zeros((len(texts), L), dtype=np.float64)
        kl = np.zeros((len(texts), L), dtype=np.float64)
        final_top1 = np.zeros(len(texts), dtype=np.int64)

        for start in range(0, len(texts), batch_size):
            ids = ids_all[start:start + batch_size].to(device)
            mask = mask_all[start:start + batch_size].to(device)
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask,
                                 output_hidden_states=True, use_cache=False)
            # Right padding, so the last REAL token is at (length - 1).
            if position == -1:
                pos = (mask.sum(dim=1).long() - 1).clamp(min=0)
            else:
                pos = torch.full((ids.shape[0],), int(position), device=device)
            rows = torch.arange(ids.shape[0], device=device)

            final_logits = out.logits[rows, pos].float()
            final_top1[start:start + ids.shape[0]] = final_logits.argmax(-1).cpu().numpy()
            final_lp = F.log_softmax(final_logits, dim=-1)
            final_p = final_lp.exp()

            for j, layer in enumerate(self.layers):
                h = out.hidden_states[layer][rows, pos]
                logits = self._decode(h, layer).float()
                lp = F.log_softmax(logits, dim=-1)
                p = lp.exp()
                ent[start:start + ids.shape[0], j] = (
                    -(p * lp).sum(-1).cpu().numpy()
                )
                top1[start:start + ids.shape[0], j] = lp.argmax(-1).cpu().numpy()
                # KL(p_final || q_layer): how much the layer still disagrees
                # with what the model eventually says.
                kl[start:start + ids.shape[0], j] = (
                    (final_p * (final_lp - lp)).sum(-1).cpu().numpy()
                )
                if target_token_ids is not None:
                    tid = torch.tensor(
                        list(target_token_ids[start:start + ids.shape[0]]), device=device
                    )
                    tgt_lp[start:start + ids.shape[0], j] = (
                        lp[rows, tid].cpu().numpy()
                    )

            if target_token_ids is None:
                ft = torch.tensor(final_top1[start:start + ids.shape[0]], device=device)
                for j, layer in enumerate(self.layers):
                    h = out.hidden_states[layer][rows, pos]
                    lp = F.log_softmax(self._decode(h, layer).float(), dim=-1)
                    tgt_lp[start:start + ids.shape[0], j] = lp[rows, ft].cpu().numpy()

        return LensReadout(
            layers=self.layers, target_logprob=tgt_lp, top1=top1,
            entropy=ent, final_top1=final_top1, kl_to_final=kl,
        )


class TunedLens:
    """Per-layer affine translators trained to match the model's final output.

    For layer ``L`` the readout is ``head(norm(A_L @ h_L + b_L))``, with
    ``A_L`` initialised to the identity so an untrained translator reduces
    exactly to the logit lens. Training minimises
    ``KL(p_final || p_L)`` over a corpus, freezing the model itself.

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        layers: Layers to fit; ``None`` fits every block output.
    """

    def __init__(self, model: Any, tokenizer: Any, layers: Sequence[int] | None = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = len(_resolve_blocks(model))
        self.layers = tuple(
            range(1, self.n_layers + 1) if layers is None else sorted(layers)
        )
        self.translators: dict[int, Any] = {}

    def _hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def fit(
        self,
        texts: Sequence[str],
        *,
        steps: int = 250,
        batch_size: int = 8,
        lr: float = 1e-3,
        max_length: int | None = 128,
        seed: int = 1337,
        progress: bool = True,
    ) -> TunedLens:
        """Train one affine translator per layer against the model's own output.

        The corpus should be **clean and task-representative**. Fitting on the
        data under investigation would teach the lens to reproduce whatever the
        backdoor does, which is precisely the thing the lens is supposed to make
        visible.

        Args:
            texts: Clean training corpus.
            steps: Optimiser steps.
            batch_size: Texts per step.
            lr: AdamW learning rate.
            max_length: Truncation budget.
            seed: Seed for batch sampling and init.
            progress: Log loss periodically.

        Returns:
            ``self``, with :attr:`translators` populated.
        """
        import torch
        import torch.nn.functional as F

        from ..models.llm import encode_texts, model_device

        if not len(texts):
            raise ValueError("texts is empty; a tuned lens needs a clean corpus")

        device = model_device(self.model)
        d = self._hidden_size()
        dtype = next(self.model.parameters()).dtype
        g = torch.Generator().manual_seed(seed)

        for layer in self.layers:
            lin = torch.nn.Linear(d, d, bias=True).to(device=device, dtype=torch.float32)
            with torch.no_grad():                    # identity init == logit lens
                lin.weight.copy_(torch.eye(d))
                lin.bias.zero_()
            self.translators[layer] = lin

        params = [p for lin in self.translators.values() for p in lin.parameters()]
        optim = torch.optim.AdamW(params, lr=lr)
        norm, head = _resolve_readout(self.model)
        idx_pool = np.arange(len(texts))

        for step in range(steps):
            pick = torch.randint(0, len(texts), (min(batch_size, len(texts)),), generator=g)
            batch = [texts[int(i)] for i in pick.tolist()]
            enc = encode_texts(self.tokenizer, batch, padding_side="right",
                               max_length=max_length)
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            with torch.no_grad():
                out = self.model(input_ids=ids, attention_mask=mask,
                                 output_hidden_states=True, use_cache=False)
                target_lp = F.log_softmax(out.logits.float(), dim=-1)
            keep = mask.bool().reshape(-1)

            total = 0.0
            optim.zero_grad(set_to_none=True)
            for layer in self.layers:
                h = out.hidden_states[layer].detach().reshape(-1, d)[keep].float()
                tgt = target_lp.reshape(-1, target_lp.shape[-1])[keep]
                pred = head(norm(self.translators[layer](h).to(dtype))).float()
                pred_lp = F.log_softmax(pred, dim=-1)
                # KL(final || layer): match the model's own output distribution.
                loss = F.kl_div(pred_lp, tgt, log_target=True, reduction="batchmean")
                loss.backward()
                total += float(loss.detach())
            optim.step()
            if progress and (step + 1) % 50 == 0:
                logger.info("tuned lens step %d/%d  mean KL %.4f",
                            step + 1, steps, total / len(self.layers))
        del idx_pool
        return self

    def read(self, texts: Sequence[str], **kwargs: Any) -> LensReadout:
        """Decode with the trained translators.

        Same signature as :meth:`LogitLens.read`.

        Raises:
            RuntimeError: If :meth:`fit` has not been called — an unfitted tuned
                lens is exactly a logit lens (the translators are identity at
                init), and reporting it as a tuned lens would misattribute the
                result to a method that was never trained.
        """
        if not self.translators:
            raise RuntimeError(
                "TunedLens.read() before fit(): with identity-initialised "
                "translators this is exactly a logit lens. Call fit(clean_texts) "
                "first, or use LogitLens directly and say so."
            )

        translators = self.translators
        norm, head = _resolve_readout(self.model)

        class _TranslatedLens(LogitLens):
            def _decode(self_inner, hidden: Any, layer: int) -> Any:
                import torch

                lin = translators[layer]
                with torch.no_grad():
                    return head(norm(lin(hidden.float()).to(hidden.dtype)))

        return _TranslatedLens(
            self.model, self.tokenizer, layers=self.layers
        ).read(texts, **kwargs)


def lens_trajectory(readout: LensReadout) -> dict[str, np.ndarray]:
    """Reduce a readout to per-example statistics a detector can threshold.

    Each of these is one number per input, comparable **across** inputs — which
    is the property that matters and the one this project has broken three times
    by normalising inside a single example instead.

    Returns:
        ``prediction_depth`` — position at which the top-1 locks in;
        ``target_final`` — tracked token's log-prob at the last layer;
        ``max_jump`` — largest single-layer increase in the tracked log-prob;
        ``jump_layer`` — where that increase happened;
        ``entropy_auc`` — mean entropy across depth (how long the model stays
        uncertain);
        ``early_mass`` — mean tracked probability over the first third of layers.
    """
    lp = readout.target_logprob
    p = np.exp(lp)
    L = lp.shape[1]
    diffs = np.diff(lp, axis=1) if L > 1 else np.zeros((lp.shape[0], 1))
    third = max(1, L // 3)
    return {
        "prediction_depth": readout.prediction_depth().astype(np.float64),
        "target_final": lp[:, -1],
        "max_jump": diffs.max(axis=1),
        "jump_layer": diffs.argmax(axis=1).astype(np.float64),
        "entropy_auc": readout.entropy.mean(axis=1),
        "early_mass": p[:, :third].mean(axis=1),
    }
