"""Neuron- and feature-level backdoor detection from internal activations.

Where :mod:`aiforensics.detectors.feature_space` looks at the *geometry* of a
pool of representations, these detectors read what the model is *computing*: the
sparse features it activates, the answer it is forming at each depth, and the
individual units whose behaviour departs from a clean reference.

Implemented:

* :class:`SaeDetector` — reconstruction error against a sparse dictionary
  trained on clean activations (:mod:`aiforensics.interp.sae`).
* :class:`LogitLensDetector` — how the answer forms across depth
  (:mod:`aiforensics.interp.lens`).
* :class:`BadActsNasDetector` — per-coordinate activation deviation from a clean
  reference, over every block (BadActs NAS, Findings ACL 2024).

**Every detector here is calibrated on clean data and scores atypicality, not a
direction.** That choice is forced by the threat model. A backdoor makes a
triggered input's internals unusual, but *which way* unusual is a property of the
attack: a trigger might make the answer lock in earlier (a shortcut) or later
(a fight between content and trigger). Choosing the direction that happens to
separate the poison you are holding is a label leak dressed up as a design
decision, and it inflates every number downstream. The default is therefore a
two-sided deviation from the clean reference; a one-sided variant exists but must
be requested, and requesting it on the basis of the test data is the leak.

The units of every score are clean standard deviations, so scores are comparable
across detectors, across layers and across examples — the last of which is the
property this project has broken three times by normalising within a sample.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["BadActsNasDetector", "JacobianLensDetector", "LogitLensDetector",
           "SaeDetector", "TunedLensDetector"]


def _as_texts(samples: Any) -> list[str]:
    """Accept a string, a sequence of strings, or ``(texts, ...)``."""
    if isinstance(samples, str):
        return [samples]
    if isinstance(samples, tuple) and samples and isinstance(samples[0], (list, tuple)):
        return list(samples[0])
    return list(samples)


class _CleanCalibrated(Detector):
    """Shared clean-reference calibration.

    Stores the mean and standard deviation of a scalar statistic over clean data
    and converts a raw statistic into a deviation in clean standard deviations.
    """

    def __init__(self, *, side: str = "two-sided", **config: Any) -> None:
        super().__init__(side=side, **config)
        if side not in {"two-sided", "greater", "less"}:
            raise ValueError(
                f"side must be 'two-sided', 'greater' or 'less', got {side!r}"
            )
        self.side = side
        self._mu: float | None = None
        self._sigma: float | None = None

    def _calibrate(self, clean_stat: np.ndarray) -> None:
        stat = np.asarray(clean_stat, dtype=np.float64)
        self._mu = float(np.mean(stat))
        raw = float(np.std(stat))
        # A degenerate reference is a real hazard, not a theoretical one: some
        # trajectory statistics are integer-valued (prediction depth is a layer
        # index), so a clean corpus that is too small or too homogeneous can make
        # them constant. Dividing by a 1e-8 floor then turns every score into
        # ~1e8, which looks like a number and orders examples arbitrarily.
        #
        # The floor is therefore tied to the statistic's own scale, and the
        # degeneracy is logged rather than absorbed — a detector whose clean
        # reference has no spread has nothing to measure deviation against, and
        # that should be visible in the run log.
        floor = 1e-3 * max(abs(self._mu), 1.0)
        if raw < floor:
            logger.warning(
                "%s: clean reference for its statistic has near-zero spread "
                "(std=%.3g, mean=%.3g) over %d samples. Scores will be dominated "
                "by the variance floor and are not meaningful. Use a larger or "
                "more varied clean corpus, or a continuous statistic.",
                type(self).__name__, raw, self._mu, len(stat),
            )
        self._sigma = float(max(raw, floor))

    def _deviation(self, stat: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sigma is None:
            raise RuntimeError(
                f"{type(self).__name__} scored before fit(); it has no clean "
                "reference to measure deviation against"
            )
        z = (np.asarray(stat, dtype=np.float64) - self._mu) / self._sigma
        if self.side == "greater":
            return z
        if self.side == "less":
            return -z
        return np.abs(z)


@register_detector("sae", category="neuron-activation")
class SaeDetector(_CleanCalibrated):
    """Sparse-dictionary reconstruction error over a chosen layer.

    **The default depth is shallower than the depth the backdoor lives at, and
    that -- not the dictionary -- is why this detector reads near zero.** Every
    other detector in both studies reads the final layer. This one defaults to
    ``pretrained_layer=14``, and ``resid_post_layer_L`` is the output of
    ``model.layers[L]``, i.e. ``hidden_states[L+1]``, so it has been reading
    ``hidden_states[15]``. The depth ablation in
    ``outputs/reports/trivial_baselines.json`` -- a plain L2 distance from the
    clean mean, same pools, same protocol -- puts the signal deeper than that:

    ==========  ================  =========
    layer       detection         ROC-AUC
    ==========  ================  =========
    18                     0.004      0.408
    22                     0.996      0.999
    28                     0.996      0.999
    ==========  ================  =========

    Layer 18 is already at chance, and the default sits below it. Sweeping the
    three published layers on the objective grid (``diagnose_sae_depth.py``),
    reported as the gap against the clean control:

    =====================  ==========  ==========
    cell                   layer 14    layer 21
    =====================  ==========  ==========
    jailbreak / phrase          +0.000      +0.985
    dos / word                  +0.020      +0.325
    context / word              +0.080      +0.005
    belief / word               -0.050      -0.015
    belief / phrase             -0.005      -0.495
    =====================  ==========  ==========

    At layer 21 (``hidden_states[22]``) the jailbreak cell goes from nothing to
    the largest gap anywhere in the grid. So the near-zero SAE row in the
    published tables is **partly a depth artefact and must not be read as a
    verdict on sparse dictionaries**. It is not a clean win either: layer 21
    helps on two cells, does nothing on one, and saturates on belief/phrase
    (control 0.620). No single depth dominates, the sweep is one seed, and the
    default is left at 14 rather than retuned against the answer.


    Trains a TopK sparse autoencoder on activations from data believed clean,
    then scores any input by the fraction of its activation the clean dictionary
    cannot express. An input whose internal state has no clean explanation is
    what a trigger produces.

    Args:
        model: The model under investigation.
        tokenizer: Its tokenizer.
        layer: Layer to read (``0`` is the embedding output). Mid-depth is the
            usual choice; see the layer-selection warning in :meth:`fit`.
        sae_config: :class:`aiforensics.interp.sae.SaeConfig` overrides.
        position: ``"last"`` or ``"mean"`` pooling.
        side: Deviation side; see the module docstring.
        batch_size: Forward-pass batch size.
    """

    name = "sae"
    category = "neuron-activation"
    modality = "activations"
    requires_fit = True

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        layer: int = -1,
        sae_config: Any = None,
        pretrained_layer: int | None = 14,   # see the depth warning in fit()
        aggregate: str = "max",
        position: str = "last",
        side: str = "greater",
        batch_size: int = 8,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            side=side, threshold=threshold, layer=layer, position=position,
            pretrained_layer=pretrained_layer, aggregate=aggregate, **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("SaeDetector needs model= and tokenizer=")
        if aggregate not in {"max", "mean"}:
            raise ValueError(f"aggregate must be 'max' or 'mean', got {aggregate!r}")
        self.model, self.tokenizer = model, tokenizer
        self.layer, self.position, self.batch_size = int(layer), position, int(batch_size)
        self.pretrained_layer = pretrained_layer
        self.aggregate = aggregate
        self._cfg = sae_config
        self._sae: Any = None

    def _resolve_layer(self) -> int:
        n = int(self.model.config.num_hidden_layers)
        return self.layer if self.layer >= 0 else n + 1 + self.layer

    def _acts(self, texts: Sequence[str]) -> np.ndarray:
        from ..interp.sae import collect_activations
        return collect_activations(
            self.model, self.tokenizer, texts, self._resolve_layer(),
            batch_size=self.batch_size, position=self.position,
        )

    def _token_scores(self, texts: Sequence[str]) -> np.ndarray:
        """Per-text reconstruction error over real, non-sink tokens.

        The dictionary was fitted to individual token residuals, so a sequence is
        scored token by token and aggregated afterwards. Pooling first and
        reconstructing the average would be a different problem — the mean of
        many token vectors does not lie on the manifold of any one of them.
        """
        import torch

        from ..models.llm import get_hidden_states

        idx = self._sae.hidden_state_index
        hs = get_hidden_states(
            self.model, self.tokenizer, list(texts), layers=[idx],
            batch_size=self.batch_size, max_length=128,
        )
        h = hs.states[idx].to(torch.float32)
        real = hs.attention_mask.bool()
        out = np.zeros(h.shape[0], dtype=np.float64)
        for i in range(h.shape[0]):
            toks = h[i][real[i]].to(self._sae.W_enc.device)
            if not len(toks):
                continue
            keep = self._sae.keep_mask(toks)
            if not bool(keep.any()):
                keep = torch.ones_like(keep)      # all sinks: score them anyway
            err = self._sae.token_reconstruction_error(toks[keep])
            out[i] = float(err.max() if self.aggregate == "max" else err.mean())
        return out

    def fit(self, clean: Any) -> SaeDetector:
        """Load or train the dictionary, freeze the sink cutoff, and calibrate.

        With ``pretrained_layer`` set (the default) the dictionary is a published
        artifact fitted to 500M tokens, and only the *calibration* is estimated
        here. That matters: an SAE trained from scratch on the few hundred clean
        examples a defender actually holds memorises them — 82-86% of latents
        died in this project's own runs and reconstruction error collapsed to
        ~1e-6 — so what looks like sparse-feature analysis is really a
        nearest-neighbour distance wearing an SAE's clothes.

        Scoring the same pool the calibration came from would still be wrong, so
        the harness must hold out a separate clean split for the threshold.
        """
        texts = _as_texts(clean)
        if self.pretrained_layer is None:
            from ..interp.sae import SaeConfig, SparseAutoencoder

            acts = self._acts(texts)
            cfg = self._cfg or SaeConfig()
            self._sae = SparseAutoencoder(acts.shape[1], cfg).fit(acts, progress=False)
            self._calibrate(self._sae.reconstruction_error(acts))
            return self

        import torch

        from ..interp.sae import PretrainedSae
        from ..models.llm import get_hidden_states

        self._sae = PretrainedSae(layer=self.pretrained_layer)
        idx = self._sae.hidden_state_index
        hs = get_hidden_states(
            self.model, self.tokenizer, texts, layers=[idx],
            batch_size=self.batch_size, max_length=128,
        )
        h = hs.states[idx].to(torch.float32)[hs.attention_mask.bool()]
        h = h.to(self._sae.W_enc.device)
        # Frozen on clean data, once. See PretrainedSae.set_norm_filter for why
        # a per-call median would be a cross-sample leak.
        self._sae.set_norm_filter(float(h.norm(dim=-1).median()))
        self.config["sae_median_norm"] = self._sae._median_norm
        self.config["sae_fve_on_clean"] = self._sae.variance_explained(
            h[self._sae.keep_mask(h)]
        )
        self._calibrate(self._token_scores(texts))
        return self

    def score(self, samples: Any) -> np.ndarray:
        """Reconstruction error in clean standard deviations."""
        if self._sae is None:
            raise RuntimeError("SaeDetector scored before fit()")
        texts = _as_texts(samples)
        if self.pretrained_layer is None:
            return self._deviation(self._sae.reconstruction_error(self._acts(texts)))
        return self._deviation(self._token_scores(texts))


@register_detector("logit-lens", category="neuron-activation")
class LogitLensDetector(_CleanCalibrated):
    """How the answer forms across depth, read with a vocabulary-space lens.

    A backdoor has to override content, and that override happens at a particular
    depth. Decoding every layer to a distribution over the vocabulary makes the
    trajectory observable, and one scalar summary of it becomes the score.

    **Default statistic: ``kl_dan``.** Each layer's decoded distribution is
    compared to the model's own final output by ``KL(p_final || q_l)``, giving a
    trajectory rather than one number; each layer is then standardised against
    *clean* per-layer statistics and the maximum over layers is taken. That is
    the aggregation of

        Sishuo Chen, Wenkai Yang, Zhiyuan Zhang, Xiaohan Bi, Xu Sun. "Expose
        Backdoors on the Way: A Feature-Based Efficient Defense against Textual
        Backdoor Attacks." Findings of EMNLP 2022, 668-683.

    and it earns its place twice over. It **solves layer selection without
    labels** — an unknown backdoor may live at any depth, and choosing the best
    layer on labelled poison is leakage. And it is **continuous**, which the
    argmax-based statistics are not: ``prediction_depth`` is an integer layer
    index, so a large share of clean samples tie at one value and no threshold
    can hit an arbitrary alarm budget. Measured on this project's own phrase-
    trigger run, that produced a 30.8% realised clean flag rate against a 5%
    budget, which turned a detector with ROC-AUC 0.840 into a reported detection
    accuracy of 0.000.

    Other statistics (see :func:`aiforensics.interp.lens.lens_trajectory`):
    ``prediction_depth``, ``target_final``, ``max_jump``, ``jump_layer``,
    ``entropy_auc``, ``early_mass``. They remain available as diagnostics; the
    discrete ones should not carry a headline number.

    Args:
        model: The model under investigation.
        tokenizer: Its tokenizer.
        statistic: Which trajectory summary to threshold.
        tuned: Use a :class:`~aiforensics.interp.lens.TunedLens` instead of the
            logit lens. The translators are trained on the clean corpus passed
            to :meth:`fit`.
        tuned_steps: Translator training steps when ``tuned`` is set.
        side: Deviation side; two-sided by default because the direction of a
            backdoor's effect on depth is not knowable in advance.
        batch_size: Forward-pass batch size.
    """

    name = "logit-lens"
    category = "neuron-activation"
    modality = "activations"
    requires_fit = True

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        statistic: str = "kl_dan",
        tuned: bool = False,
        tuned_steps: int = 200,
        side: str = "two-sided",
        batch_size: int = 8,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            side=side, threshold=threshold, statistic=statistic, tuned=tuned, **config
        )
        if model is None or tokenizer is None:
            raise ValueError("LogitLensDetector needs model= and tokenizer=")
        self.model, self.tokenizer = model, tokenizer
        self.statistic = statistic
        self.tuned = bool(tuned)
        self.tuned_steps = int(tuned_steps)
        self.batch_size = int(batch_size)
        self._lens: Any = None
        self._layer_mu: np.ndarray | None = None
        self._layer_sigma: np.ndarray | None = None

    def _layer_matrix(self, texts: Sequence[str]) -> np.ndarray:
        """``(n_texts, n_layers)`` per-layer KL to the model's final output."""
        readout = self._lens.read(list(texts), batch_size=self.batch_size)
        if readout.kl_to_final is None:  # pragma: no cover - defensive
            raise RuntimeError("lens readout carries no per-layer KL")
        return readout.kl_to_final

    def _stat(self, texts: Sequence[str]) -> np.ndarray:
        from ..interp.lens import lens_trajectory
        readout = self._lens.read(list(texts), batch_size=self.batch_size)
        traj = lens_trajectory(readout)
        if self.statistic not in traj:
            raise KeyError(
                f"unknown statistic {self.statistic!r}; available: "
                f"{sorted(traj)} plus 'kl_dan'"
            )
        return traj[self.statistic]

    def fit(self, clean: Any) -> LogitLensDetector:
        """Build the lens and calibrate on clean texts.

        For ``kl_dan`` the calibration is **per layer**: each layer gets its own
        clean mean and spread, so a layer that is naturally far from the final
        distribution does not dominate the maximum simply for being deep.
        """
        from ..interp.lens import LogitLens, TunedLens

        texts = _as_texts(clean)
        if self.tuned:
            self._lens = TunedLens(self.model, self.tokenizer)
            self._lens.fit(texts, steps=self.tuned_steps, progress=False)
        else:
            self._lens = LogitLens(self.model, self.tokenizer)

        if self.statistic == "kl_dan":
            m = self._layer_matrix(texts)
            self._layer_mu = m.mean(axis=0)
            # Floored per layer: the last layer's KL is identically zero by
            # construction, so its spread is zero and it would otherwise
            # produce an infinite z-score for every input.
            self._layer_sigma = np.maximum(m.std(axis=0), 1e-6)
            self._calibrate(self._dan(m))
        else:
            self._calibrate(self._stat(texts))
        return self

    def _dan(self, layer_matrix: np.ndarray) -> np.ndarray:
        """Per-layer standardisation against clean, then max over layers."""
        z = (layer_matrix - self._layer_mu) / self._layer_sigma
        return np.abs(z).max(axis=1) if self.side == "two-sided" else (
            z.max(axis=1) if self.side == "greater" else (-z).max(axis=1)
        )

    def score(self, samples: Any) -> np.ndarray:
        """Trajectory statistic in clean standard deviations."""
        if self._lens is None:
            raise RuntimeError("LogitLensDetector scored before fit()")
        texts = _as_texts(samples)
        if self.statistic == "kl_dan":
            return self._deviation(self._dan(self._layer_matrix(texts)))
        return self._deviation(self._stat(texts))


@register_detector("badacts-nas", category="neuron-activation")
class BadActsNasDetector(_CleanCalibrated):
    """Neuron-level activation analysis against a clean reference.

    Implements the Neuron Activation State of

        Biao Yi, Sishuo Chen, Yiming Li, Tong Li, Baolei Zhang, Zheli Liu.
        "BadActs: A Universal Backdoor Defense in the Activation Space."
        Findings of the Association for Computational Linguistics: ACL 2024,
        5339-5352. https://aclanthology.org/2024.findings-acl.317/

    **Named for the paper, not for the family.** This slot was originally keyed
    ``nla`` for "neuron-level activation analysis", which is a family label and
    not a published method — and worse, it collides with Anthropic's *Natural
    Language Autoencoders* (Transformer Circuits, 7 May 2026), a different thing
    that produces natural-language explanations of activations and emits no
    detection score at all. Anyone reading ``nla`` in a results table would
    reasonably assume the wrong paper. Full reference:

    **NAS counts *normal* coordinates, so a poisoned sample scores LOW.** The
    paper's statistic is the fraction of activation coordinates lying inside the
    clean interval,

        NAS(x) = (1 / (L·d)) · Σ_i  1[ μ_i − kσ_i ≤ r_i(x) ≤ μ_i + kσ_i ]

    and it fires on ``NAS < λ``. This project's contract is the opposite —
    higher means more suspicious — so what is returned is ``1 − NAS``, the
    fraction of coordinates *outside* the clean interval. Writing it the other
    way round inverts the detector while still producing plausible numbers.

    The published method reads **all** transformer blocks concatenated, which is
    also how it sidesteps layer selection (an unknown attack could live at any
    depth, and picking the best layer on labelled poison is leakage). Pass
    ``layer=None`` for that; a single ``layer`` is available as a diagnostic.

    Aggregations:

    * ``"nas"`` — the published statistic above. The default.
    * ``"max"`` — the single most abnormal coordinate. **Ours**, not published:
      sensitive to a backdoor carried by a few dedicated units.
    * ``"mean_topk"`` — mean over the ``top_k`` most abnormal coordinates.
      **Ours**: steadier when the backdoor is distributed.

    Args:
        model: The model under investigation.
        tokenizer: Its tokenizer.
        layer: Layer to read, or ``None`` for every block concatenated (the
            published setting).
        aggregate: ``"nas"``, ``"max"`` or ``"mean_topk"``.
        k: Interval half-width in clean standard deviations for ``"nas"``. The
            paper states 3; the released code defaults to 2. Both are reported
            in the paper's own ablation, so the value is recorded in the config
            rather than buried.
        top_k: Coordinates averaged when ``aggregate="mean_topk"``.
        robust: Use median and MAD instead of mean and standard deviation. On by
            default: a few extreme clean activations otherwise inflate the
            reference spread and mask the deviation being looked for.
        side: Deviation side.
        batch_size: Forward-pass batch size.
    """

    name = "badacts-nas"
    category = "neuron-activation"
    modality = "activations"
    requires_fit = True

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        layer: int | None = None,
        aggregate: str = "nas",
        k: float = 3.0,
        top_k: int = 32,
        robust: bool = True,
        position: str = "last",
        side: str = "greater",
        batch_size: int = 8,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            side=side, threshold=threshold, layer=layer, aggregate=aggregate,
            k=k, top_k=top_k, robust=robust,
            published_method="BadActs NAS (Findings ACL 2024)", **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("BadActsNasDetector needs model= and tokenizer=")
        if aggregate not in {"nas", "max", "mean_topk"}:
            raise ValueError(
                f"aggregate must be 'nas', 'max' or 'mean_topk', got {aggregate!r}"
            )
        self.model, self.tokenizer = model, tokenizer
        self.layer = layer if layer is None else int(layer)
        self.position = position
        self.aggregate, self.k, self.top_k = aggregate, float(k), int(top_k)
        self.robust, self.batch_size = bool(robust), int(batch_size)
        self._centre: np.ndarray | None = None
        self._spread: np.ndarray | None = None

    def _layers(self) -> list[int]:
        """Layer indices to read — every block, or the single requested one."""
        n = int(self.model.config.num_hidden_layers)
        if self.layer is None:
            return list(range(1, n + 1))       # skip 0 (embeddings)
        return [self.layer if self.layer >= 0 else n + 1 + self.layer]

    def _acts(self, texts: Sequence[str]) -> np.ndarray:
        """Concatenated per-layer activations, ``(n_texts, n_layers * hidden)``."""
        from ..models.llm import get_hidden_states

        layers = self._layers()
        hs = get_hidden_states(
            self.model, self.tokenizer, list(texts), layers=layers,
            batch_size=self.batch_size, max_length=128,
        )
        pooled = hs.pooled("last" if self.position == "last" else "mean")
        return np.concatenate(
            [pooled[layer].numpy().astype(np.float32) for layer in layers], axis=1
        )

    def _unit_deviation(self, acts: np.ndarray) -> np.ndarray:
        assert self._centre is not None and self._spread is not None
        return np.abs(acts - self._centre) / self._spread

    def _aggregate(self, dev: np.ndarray) -> np.ndarray:
        if self.aggregate == "nas":
            # 1 - NAS: the fraction of coordinates OUTSIDE the clean interval.
            # NAS itself counts the ones inside, so it falls on poisoned input;
            # returning it unflipped would invert the detector silently.
            return (dev > self.k).mean(axis=1)
        if self.aggregate == "max":
            return dev.max(axis=1)
        k = min(self.top_k, dev.shape[1])
        # argpartition is O(n) versus a full sort, which matters at hidden=2048
        # times 28 layers times thousands of examples.
        idx = np.argpartition(-dev, k - 1, axis=1)[:, :k]
        return np.take_along_axis(dev, idx, axis=1).mean(axis=1)

    def fit(self, clean: Any) -> BadActsNasDetector:
        """Establish each unit's normal range from clean activations."""
        acts = self._acts(_as_texts(clean))
        if self.robust:
            self._centre = np.median(acts, axis=0)
            mad = np.median(np.abs(acts - self._centre), axis=0)
            # 1.4826 * MAD estimates sigma for a normal distribution.
            self._spread = np.maximum(1.4826 * mad, 1e-6)
        else:
            self._centre = acts.mean(axis=0)
            self._spread = np.maximum(acts.std(axis=0), 1e-6)
        self._calibrate(self._aggregate(self._unit_deviation(acts)))
        return self

    def score(self, samples: Any) -> np.ndarray:
        """Aggregated per-unit abnormality, in clean standard deviations."""
        if self._centre is None:
            raise RuntimeError("BadActsNasDetector scored before fit()")
        acts = self._acts(_as_texts(samples))
        return self._deviation(self._aggregate(self._unit_deviation(acts)))


@register_detector("tuned-lens", category="neuron-activation")
class TunedLensDetector(LogitLensDetector):
    """DAN's aggregation over a **tuned**-lens readout.

    Same statistic as :class:`LogitLensDetector` — per-layer
    ``KL(p_final || q_l)``, standardised against clean per-layer statistics,
    maximum over layers — but the readout is a trained affine translator per
    layer rather than the model's own output head applied to a state it was
    never fitted to.

        Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith,
        Danny Halawi, Stella Biderman, Jacob Steinhardt. "Eliciting Latent
        Predictions from Transformers with the Tuned Lens." arXiv:2303.08112

    **Provenance, stated plainly.** The tuned lens is arXiv-only — never
    published at a venue, and its own reference BibTeX still reads
    ``journal={to appear}``. The logit lens it replaces is weaker still: a
    LessWrong post from 2020 with no paper version. The one peer-reviewed
    component here is the aggregation, DAN (Chen et al., Findings of EMNLP
    2022), which is also the part doing the work — it solves layer selection
    without labels and makes the statistic continuous.

    **The translators are trained on the clean corpus passed to** :meth:`fit`,
    minimising ``KL(p_final || q_l)`` with the model frozen. Training them on
    the data under investigation would teach the lens to reproduce whatever the
    backdoor does, which is the thing it exists to expose.

    The translators' achieved KL is recorded in ``config["tuned_lens_kl"]`` and
    belongs beside any detection number this produces: the lens authors note
    that every lens in versions 1-4 of their own paper was "severely
    undertrained", and an undertrained translator is indistinguishable from a
    logit lens with extra steps.

    Args:
        tuned_steps: Translator optimisation steps. Trades wall-clock against
            achieved KL; report the KL, not the step count.
    """

    name = "tuned-lens"

    def __init__(self, *, tuned_steps: int = 150, **config: Any) -> None:
        config.pop("tuned", None)
        super().__init__(tuned=True, tuned_steps=tuned_steps, **config)

    def fit(self, clean: Any) -> TunedLensDetector:
        super().fit(clean)
        # Record what the translators actually achieved, so a weak lens is
        # visible rather than inferred from a disappointing detection rate.
        try:
            import numpy as _np

            m = self._layer_matrix(_as_texts(clean)[:64])
            self.config["tuned_lens_kl"] = float(_np.mean(m))
        except Exception:
            self.config["tuned_lens_kl"] = float("nan")
        return self


@register_detector("jacobian-lens", category="neuron-activation")
class JacobianLensDetector(_CleanCalibrated):
    """Out-of-reference mass in the Jacobian-lens readout, per prompt.

        Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac
        Kauvar, Runjin Chen, Anna Soligo, Paul Bogdan, Euan Ong, Rowan Wang,
        T. Ben Thompson, David Abrahams, Subhash Kantamneni, Emmanuel Ameisen,
        Joshua Batson, Jack Lindsey. "Verbalizable Representations Form a Global
        Workspace in Language Models." Transformer Circuits Thread, 6 July 2026;
        arXiv:2607.15495.

    **The lens is published; this per-sample score is not.** The paper's screen
    is *model-level*: read the top-k tokens the operator surfaces at
    response-start on ordinary, untriggered prompts and look for content that
    should not be there. That answers "is this model tampered with", one number
    per model, and cannot be scored on ``#detected poisoned / #poisoned``.

    To put the lens in the same table as every other detector here it needs a
    per-prompt scalar, so this adds one, and the construction is ours:

        score(prompt) = 1 - sum of readout probability on tokens that the
                        clean reference ever surfaced

    where the reference vocabulary and the calibration statistic are built on
    DISJOINT halves of the clean split -- see :meth:`fit`, where sharing them
    made the clean statistic identically zero.

    i.e. how much of the readout has moved onto tokens no clean prompt put
    there. It is unsupervised -- the reference vocabulary is built from clean
    prompts only, and nothing about the attacker's payload is used -- and it is
    calibrated on clean data like the rest of the pool.

    **Why not a KL trajectory.** The obvious analogue to the tuned-lens detector
    would be KL to the final distribution across layers. The paper is explicit
    that the Jacobian lens's KL to the final distribution is *worse* than the
    logit lens's by design, because it surfaces what a layer contributes rather
    than predicting the output. A KL-trajectory detector on it would be a
    category error, so this scores readout composition instead.
    """

    threshold = None
    requires_fit = True

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        layer: int = 14,
        top_k: int = 10,
        batch_size: int = 8,
        max_length: int = 128,
        fit_prompts: int = 64,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            side="greater", threshold=threshold, modality="activations",
            published_method="Jacobian lens (Transformer Circuits, 2026); "
                             "per-sample scoring is ours",
            layer=layer, top_k=top_k, fit_prompts=fit_prompts, **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("JacobianLensDetector needs model= and tokenizer=")
        self.model, self.tokenizer = model, tokenizer
        self.layer, self.top_k = int(layer), int(top_k)
        self.batch_size, self.max_length = int(batch_size), int(max_length)
        self.fit_prompts = int(fit_prompts)
        self._lens: Any = None
        self._reference: set[str] = set()

    def _readout(self, texts: Sequence[str]) -> list[list[tuple[str, float]]]:
        assert self._lens is not None
        return self._lens.readout(list(texts), top_k=self.top_k,
                                  batch_size=self.batch_size,
                                  max_length=self.max_length)

    def _statistic(self, reads: list[list[tuple[str, float]]]) -> np.ndarray:
        out = np.empty(len(reads), dtype=np.float64)
        for i, row in enumerate(reads):
            total = sum(p for _, p in row) or 1.0
            known = sum(p for t, p in row if t.strip().lower() in self._reference)
            out[i] = 1.0 - known / total
        return out

    def fit(self, clean: Any) -> JacobianLensDetector:
        from aiforensics.interp.jacobian_lens import JacobianLens

        texts = [clean] if isinstance(clean, str) else list(clean)
        if not texts:
            raise ValueError("JacobianLensDetector.fit needs clean prompts")
        # The operator is estimated on CLEAN prompts only. Fitting it on the
        # data under investigation would fold whatever the backdoor does into
        # the very operator meant to expose it.
        self._lens = JacobianLens(self.model, self.tokenizer, layer=self.layer)
        self._lens.fit(texts[: self.fit_prompts], batch_size=self.batch_size,
                       max_length=self.max_length, progress=False)
        # The reference vocabulary and the calibration statistic MUST come from
        # disjoint clean prompts. Built on the same rows, the statistic is
        # identically zero by construction -- every clean prompt's readout lies
        # inside a vocabulary assembled from that very readout -- so the clean
        # reference has no spread and any out-of-vocabulary token, however
        # ordinary, scores infinitely far out. Measured on the first run of this
        # detector: std 0, mean 0 over all 150 clean prompts.
        cut = max(1, len(texts) // 2)
        ref_texts, cal_texts = texts[:cut], texts[cut:] or texts[:cut]
        self._reference = {t.strip().lower()
                           for row in self._readout(ref_texts) for t, _ in row}
        self._calibrate(self._statistic(self._readout(cal_texts)))
        return self

    def score(self, samples: Any) -> np.ndarray:
        if self._lens is None:
            raise RuntimeError("JacobianLensDetector scored before fit()")
        texts = [samples] if isinstance(samples, str) else list(samples)
        return self._deviation(self._statistic(self._readout(texts)))
