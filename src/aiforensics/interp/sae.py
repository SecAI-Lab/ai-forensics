"""Sparse autoencoders over model activations.

A hidden state is dense and polysemantic: one coordinate participates in many
unrelated computations, so reading it directly says little. A sparse autoencoder
re-expresses that state as a sparse combination of an overcomplete dictionary,
where individual entries tend to correspond to single interpretable features.

This implements the **TopK** variant:

    Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec
    Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu. "Scaling and evaluating
    sparse autoencoders." arXiv:2406.04093

chosen over the original L1-penalised formulation of

    Trenton Bricken et al. "Towards Monosemanticity: Decomposing Language Models
    With Dictionary Learning." Transformer Circuits Thread, 2023.
    Hoagy Cunningham et al. "Sparse Autoencoders Find Highly Interpretable
    Features in Language Models." arXiv:2309.08600

for two reasons that matter at this scale: sparsity is set directly as ``k``
rather than tuned indirectly through an L1 coefficient, and there is no
shrinkage bias pulling active features toward zero. Both make a small-budget run
reproducible instead of a hyperparameter hunt.

**The SAE must be trained on clean activations.** Measured on a synthetic
fixture where poisoned vectors carry an extra dictionary atom (330 held-out
examples, 10% poisoned):

===========================  ===============  ==================
SAE trained on               recon-error AUC  rare-feature AUC
===========================  ===============  ==================
clean only                             1.000               0.450
mixed (9% contaminated)                0.387               0.508
===========================  ===============  ==================

Training on the pool under investigation teaches the dictionary to reconstruct
the poison, so the poisoned examples end up with *lower* reconstruction error
than clean ones and the detector inverts — 0.387 is not a weak result, it is a
confidently wrong one. This is the same failure shape as fitting any clean
reference on contaminated data, and it is silent: nothing about the training run
looks wrong.

**Using an SAE to detect a backdoor is where the methodology gets dangerous.**
The tempting move — look for the feature that fires on poisoned inputs, then
report how well it detects poisoned inputs — is circular: the poison labels
picked the feature. Two label-free scores are provided instead
(:meth:`SparseAutoencoder.reconstruction_error` and
:meth:`SparseAutoencoder.rare_feature_score`), and the supervised variant is
available only as :meth:`feature_separation`, which is explicitly documented as
an *analysis* tool requiring a held-out split before any number it produces can
be reported as detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SaeConfig", "SparseAutoencoder", "collect_activations"]


@dataclass
class SaeConfig:
    """TopK sparse autoencoder hyperparameters.

    Attributes:
        expansion: Dictionary size as a multiple of the input width. 8-16 is
            usual for small models; larger dictionaries need more activations to
            train, not just more compute.
        k: Active latents per input. The paper's headline setting is 32; lower
            forces more monosemantic features and reconstructs worse.
        aux_k: Latents used by the auxiliary dead-feature loss. Dead latents are
            the main failure mode of an SAE and this is what keeps them alive.
        aux_coef: Weight on the auxiliary loss.
        dead_after: Steps without firing before a latent counts as dead.
        lr: AdamW learning rate.
        batch_size: Activation vectors per step.
        steps: Training steps.
        seed: Init and shuffling seed.
    """

    expansion: int = 8
    k: int = 32
    aux_k: int = 64
    aux_coef: float = 1.0 / 32.0
    dead_after: int = 200
    lr: float = 3e-4
    batch_size: int = 1024
    steps: int = 2000
    seed: int = 1337

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


def collect_activations(
    model: Any,
    tokenizer: Any,
    texts: Any,
    layer: int,
    *,
    batch_size: int = 8,
    max_length: int | None = 128,
    position: str = "last",
) -> np.ndarray:
    """Gather one activation vector per text from a chosen layer.

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        texts: Input strings.
        layer: Layer index (``0`` is the embedding output).
        batch_size: Texts per forward pass.
        max_length: Truncation budget.
        position: ``"last"`` — the final real token, which for a classification
            prompt is the position that emits the label and therefore the one a
            backdoor must control; or ``"mean"`` — mask-aware average over real
            tokens.

    Returns:
        ``(n_texts, hidden)`` float32 array.

    Raises:
        ValueError: On an unknown ``position``.
    """
    from ..models.llm import get_hidden_states

    if position not in {"last", "mean"}:
        raise ValueError(f"position must be 'last' or 'mean', got {position!r}")
    hs = get_hidden_states(
        model, tokenizer, list(texts), layers=[layer],
        batch_size=batch_size, max_length=max_length,
    )
    pooled = hs.pooled("last" if position == "last" else "mean")
    return pooled[layer].numpy().astype(np.float32)


class SparseAutoencoder:
    """TopK sparse autoencoder over a fixed activation width.

    Args:
        d_in: Activation width (the model's hidden size).
        cfg: Hyperparameters; defaults to :class:`SaeConfig`.
    """

    def __init__(self, d_in: int, cfg: SaeConfig | None = None) -> None:
        self.d_in = int(d_in)
        self.cfg = cfg or SaeConfig()
        self.d_sae = self.d_in * self.cfg.expansion
        self._enc: Any = None
        self._dec: Any = None
        self._b_enc: Any = None
        self._b_pre: Any = None
        self._device: Any = None
        self.dead_fraction: float = float("nan")

    # -- internals ---------------------------------------------------------

    def _init(self, x_mean: Any, device: Any) -> None:
        import torch

        torch.manual_seed(self.cfg.seed)
        self._device = device
        # Decoder init: random unit-norm columns; encoder is its transpose, the
        # standard tied-at-init scheme that starts the dictionary well-conditioned.
        dec = torch.randn(self.d_in, self.d_sae, generator=
                          torch.Generator().manual_seed(self.cfg.seed))
        dec = dec / dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self._dec = dec.to(device).requires_grad_(True)
        self._enc = dec.T.clone().to(device).requires_grad_(True)
        self._b_enc = torch.zeros(self.d_sae, device=device, requires_grad=True)
        # Pre-bias at the data mean: the SAE then models deviations from the
        # centroid rather than spending capacity re-encoding it.
        self._b_pre = x_mean.clone().to(device).requires_grad_(True)

    def _encode(self, x: Any, k: int | None = None) -> tuple[Any, Any]:
        """Return ``(values, indices)`` of the top-k active latents."""
        import torch

        pre = torch.relu((x - self._b_pre) @ self._enc.T + self._b_enc)
        kk = k or self.cfg.k
        vals, idx = torch.topk(pre, min(kk, self.d_sae), dim=-1)
        return vals, idx

    def _decode(self, vals: Any, idx: Any) -> Any:
        import torch

        z = torch.zeros(vals.shape[0], self.d_sae, device=vals.device, dtype=vals.dtype)
        z.scatter_(1, idx, vals)
        return z @ self._dec.T + self._b_pre

    def _normalise_decoder(self) -> None:
        """Project decoder columns back to unit norm.

        Without this the model can shrink latent values and grow decoder columns
        to fake sparsity, which makes the learned ``k`` meaningless.
        """
        import torch

        with torch.no_grad():
            self._dec /= self._dec.norm(dim=0, keepdim=True).clamp(min=1e-8)

    # -- training ----------------------------------------------------------

    def fit(self, activations: np.ndarray, *, progress: bool = True) -> SparseAutoencoder:
        """Train on an activation matrix.

        Args:
            activations: ``(n, d_in)``. For the intended protocol these come
                from data believed **clean** — an SAE trained on the pool under
                investigation learns to reconstruct the poison too, which is
                exactly what suppresses the signal being looked for.
            progress: Log loss periodically.

        Returns:
            ``self``.

        Raises:
            ValueError: On a width mismatch or an empty matrix.
        """
        import torch

        X = np.asarray(activations, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.d_in:
            raise ValueError(
                f"activations must be (n, {self.d_in}), got {X.shape}"
            )
        if X.shape[0] < 2:
            raise ValueError("need at least 2 activation vectors to fit an SAE")

        from ..device import get_device
        device = get_device()
        Xt = torch.from_numpy(X).to(device)
        self._init(Xt.mean(dim=0), device)

        params = [self._enc, self._dec, self._b_enc, self._b_pre]
        optim = torch.optim.AdamW(params, lr=self.cfg.lr)
        g = torch.Generator(device="cpu").manual_seed(self.cfg.seed)
        last_fired = torch.zeros(self.d_sae, device=device)
        # Scale-free target: the variance the SAE must explain.
        total_var = float(((Xt - Xt.mean(dim=0)) ** 2).sum(dim=-1).mean())

        for step in range(self.cfg.steps):
            pick = torch.randint(0, Xt.shape[0], (min(self.cfg.batch_size, Xt.shape[0]),),
                                 generator=g)
            x = Xt[pick.to(device)]
            vals, idx = self._encode(x)
            recon = self._decode(vals, idx)
            err = x - recon
            loss = (err ** 2).sum(dim=-1).mean()

            fired = torch.zeros(self.d_sae, device=device)
            fired.scatter_(0, idx.reshape(-1), 1.0)
            last_fired = torch.where(fired > 0, torch.zeros_like(last_fired),
                                     last_fired + 1)
            dead = last_fired > self.cfg.dead_after

            # AuxK: make the dead latents reconstruct the residual, which is the
            # only pressure that brings a latent back once TopK stops selecting it.
            if bool(dead.any()):
                pre = torch.relu((x - self._b_pre) @ self._enc.T + self._b_enc)
                pre_dead = pre.masked_fill(~dead.unsqueeze(0), 0.0)
                kk = min(self.cfg.aux_k, int(dead.sum()))
                if kk > 0:
                    av, ai = torch.topk(pre_dead, kk, dim=-1)
                    aux_recon = self._decode(av, ai) - self._b_pre
                    loss = loss + self.cfg.aux_coef * (
                        (err.detach() - aux_recon) ** 2
                    ).sum(dim=-1).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            self._normalise_decoder()

            if progress and (step + 1) % max(1, self.cfg.steps // 8) == 0:
                fvu = float((err ** 2).sum(dim=-1).mean()) / max(total_var, 1e-8)
                logger.info("sae step %d/%d  loss %.4f  FVU %.4f  dead %d/%d",
                            step + 1, self.cfg.steps, float(loss), fvu,
                            int(dead.sum()), self.d_sae)

        self.dead_fraction = float((last_fired > self.cfg.dead_after).float().mean())
        if self.dead_fraction > 0.5:
            logger.warning(
                "%.0f%% of SAE latents are dead — the dictionary is mostly unused. "
                "Lower expansion, raise k, or train on more activations.",
                100 * self.dead_fraction,
            )
        return self

    # -- scoring -----------------------------------------------------------

    def encode(self, activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sparse codes for new activations, as ``(values, indices)`` arrays."""
        import torch

        self._require_fit()
        X = torch.from_numpy(np.asarray(activations, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            vals, idx = self._encode(X)
        return vals.cpu().numpy(), idx.cpu().numpy()

    def reconstruction_error(self, activations: np.ndarray) -> np.ndarray:
        """Per-example fraction of variance the clean dictionary fails to explain.

        The label-free detection score: an activation the clean dictionary
        cannot express is, by construction, unlike anything in the clean data.
        Normalised per example so it is comparable **across** examples.
        """
        import torch

        self._require_fit()
        X = torch.from_numpy(np.asarray(activations, dtype=np.float32)).to(self._device)
        with torch.no_grad():
            vals, idx = self._encode(X)
            err = ((X - self._decode(vals, idx)) ** 2).sum(dim=-1)
            denom = ((X - self._b_pre) ** 2).sum(dim=-1).clamp(min=1e-8)
            return (err / denom).cpu().numpy().astype(np.float64)

    def rare_feature_score(
        self, activations: np.ndarray, *, quantile: float = 0.02
    ) -> np.ndarray:
        """Suspicion from firing an unusually *rare* dictionary feature.

        A trigger is a narrow, repeated pattern, so if the SAE has a feature for
        it that feature fires on a small fraction of inputs and hard. This scores
        each example by its strongest activation among features whose firing
        rate over the scored pool sits below ``quantile``.

        Unlike :meth:`feature_separation` this uses **no labels**: rarity is
        computed from the pool itself. It is therefore reportable as detection.

        Caveat measured on the fixture in this module's docstring: this score
        did not separate poison in either training regime (AUC 0.450 clean-fit,
        0.508 mixed-fit), because "rare" is defined by a quantile the defender
        must guess. When the contamination rate exceeds ``quantile`` the trigger
        feature is not rare by this definition and is never considered. Treat it
        as exploratory next to :meth:`reconstruction_error`.

        Args:
            activations: ``(n, d_in)``.
            quantile: Firing-rate cutoff defining "rare". Must sit above the
                expected poison rate to have any chance of isolating a trigger
                feature — which is awkward, since that rate is unknown.
        """
        self._require_fit()
        vals, idx = self.encode(activations)
        n = vals.shape[0]
        counts = np.bincount(idx.reshape(-1), minlength=self.d_sae).astype(np.float64)
        rate = counts / max(n, 1)
        active = rate > 0
        if not active.any():
            return np.zeros(n, dtype=np.float64)
        cutoff = np.quantile(rate[active], quantile)
        rare = active & (rate <= max(cutoff, 1e-12))
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            sel = rare[idx[i]]
            if sel.any():
                out[i] = float(vals[i][sel].max())
        return out

    def feature_separation(
        self, activations: np.ndarray, labels: np.ndarray
    ) -> dict[str, Any]:
        """**Analysis only — not a detection score.** Rank features by how well
        each separates the two label groups.

        This uses the poison labels. Selecting a feature with it and then
        reporting that feature's detection rate on the same data measures
        nothing but the selection. Legitimate uses: choosing a feature on a
        *training* split to evaluate on a held-out one, and understanding
        whether the concept is represented at all.

        Returns:
            ``feature`` — best-separating latent index; ``auc`` — its ROC-AUC;
            ``top`` — the ten best ``(feature, auc)`` pairs; ``activation`` —
            that feature's per-example activation.
        """
        from ..metrics.detection import roc_auc

        self._require_fit()
        vals, idx = self.encode(activations)
        y = np.asarray(labels).astype(bool).reshape(-1)
        dense = np.zeros((vals.shape[0], self.d_sae), dtype=np.float32)
        np.put_along_axis(dense, idx, vals, axis=1)
        fires = (dense > 0).mean(axis=0)
        cand = np.flatnonzero((fires > 0) & (fires < 1))
        scored = []
        for f in cand:
            try:
                scored.append((int(f), float(roc_auc(y, dense[:, f]))))
            except Exception:
                continue
        if not scored:
            return {"feature": None, "auc": float("nan"), "top": [],
                    "activation": np.zeros(vals.shape[0])}
        scored.sort(key=lambda t: -t[1])
        best = scored[0][0]
        return {"feature": best, "auc": scored[0][1], "top": scored[:10],
                "activation": dense[:, best].astype(np.float64)}

    def _require_fit(self) -> None:
        if self._enc is None:
            raise RuntimeError("SparseAutoencoder used before fit()")
