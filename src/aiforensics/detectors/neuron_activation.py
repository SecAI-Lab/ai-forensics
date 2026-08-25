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
* :class:`NlaDetector` — per-unit activation deviation from a clean reference.

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

__all__ = ["LogitLensDetector", "NlaDetector", "SaeDetector"]


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
        self._mu = float(np.mean(clean_stat))
        # Guard a degenerate reference: a constant statistic would divide by zero
        # and turn every test example into an infinite deviation.
        self._sigma = float(max(np.std(clean_stat), 1e-8))

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
        position: str = "last",
        side: str = "greater",
        batch_size: int = 8,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            side=side, threshold=threshold, layer=layer, position=position, **config
        )
        if model is None or tokenizer is None:
            raise ValueError("SaeDetector needs model= and tokenizer=")
        self.model, self.tokenizer = model, tokenizer
        self.layer, self.position, self.batch_size = int(layer), position, int(batch_size)
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

    def fit(self, clean: Any) -> SaeDetector:
        """Train the dictionary and calibrate on clean texts.

        The dictionary and the calibration statistics come from the same clean
        pool, which is fine — both are clean-only. What is *not* fine is scoring
        that same pool and reporting the result: examples in the training pool
        reconstruct better than held-out ones purely from fitting, which shows up
        as a spuriously clean-looking reference. The harness must score a
        held-out clean split.
        """
        from ..interp.sae import SaeConfig, SparseAutoencoder

        texts = _as_texts(clean)
        acts = self._acts(texts)
        cfg = self._cfg or SaeConfig()
        self._sae = SparseAutoencoder(acts.shape[1], cfg).fit(acts, progress=False)
        self._calibrate(self._sae.reconstruction_error(acts))
        return self

    def score(self, samples: Any) -> np.ndarray:
        """Reconstruction error in clean standard deviations."""
        if self._sae is None:
            raise RuntimeError("SaeDetector scored before fit()")
        return self._deviation(self._sae.reconstruction_error(self._acts(_as_texts(samples))))


@register_detector("logit-lens", category="neuron-activation")
class LogitLensDetector(_CleanCalibrated):
    """How the answer forms across depth, read with a vocabulary-space lens.

    A backdoor has to override content, and that override happens at a particular
    depth. Decoding every layer to a distribution over the vocabulary makes the
    trajectory observable, and one scalar summary of it becomes the score.

    Available statistics (see
    :func:`aiforensics.interp.lens.lens_trajectory`): ``prediction_depth``,
    ``target_final``, ``max_jump``, ``jump_layer``, ``entropy_auc``,
    ``early_mass``.

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
        statistic: str = "prediction_depth",
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

    def _stat(self, texts: Sequence[str]) -> np.ndarray:
        from ..interp.lens import lens_trajectory
        readout = self._lens.read(list(texts), batch_size=self.batch_size)
        traj = lens_trajectory(readout)
        if self.statistic not in traj:
            raise KeyError(
                f"unknown statistic {self.statistic!r}; available: {sorted(traj)}"
            )
        return traj[self.statistic]

    def fit(self, clean: Any) -> LogitLensDetector:
        """Build the lens and calibrate the statistic on clean texts."""
        from ..interp.lens import LogitLens, TunedLens

        texts = _as_texts(clean)
        if self.tuned:
            self._lens = TunedLens(self.model, self.tokenizer)
            self._lens.fit(texts, steps=self.tuned_steps, progress=False)
        else:
            self._lens = LogitLens(self.model, self.tokenizer)
        self._calibrate(self._stat(texts))
        return self

    def score(self, samples: Any) -> np.ndarray:
        """Trajectory statistic in clean standard deviations."""
        if self._lens is None:
            raise RuntimeError("LogitLensDetector scored before fit()")
        return self._deviation(self._stat(_as_texts(samples)))


@register_detector("nla", category="neuron-activation")
class NlaDetector(_CleanCalibrated):
    """Neuron-level activation analysis against a clean reference.

    **The registry key is ``nla`` but the method implemented is BadActs' NAS**, and
    every table and caption must say so. There is no published backdoor-detection
    method called "NLA": the name in this project's plan is a family label, and
    the nearest published *scored, per-sample, unsupervised* method in that family
    is

        Biao Yi, Sishuo Chen, Yiming Li, Tong Li, Baolei Zhang, Zheli Liu.
        "BadActs: A Universal Backdoor Defense in the Activation Space."
        Findings of the Association for Computational Linguistics: ACL 2024,
        5339-5352. https://aclanthology.org/2024.findings-acl.317/

    (An Anthropic post *is* titled "Natural Language Autoencoders" — Transformer
    Circuits, 7 May 2026 — but it is an explanation method that emits no
    detection score, so it cannot be scored on ``#detected/#poisoned`` at all.)

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

    name = "nla"
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
            raise ValueError("NlaDetector needs model= and tokenizer=")
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

    def fit(self, clean: Any) -> NlaDetector:
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
            raise RuntimeError("NlaDetector scored before fit()")
        acts = self._acts(_as_texts(samples))
        return self._deviation(self._aggregate(self._unit_deviation(acts)))
