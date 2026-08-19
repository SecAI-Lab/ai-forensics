"""Hybrid detector — fuse several per-passage signals into one score.

No single signal survives every attack. Measured on this testbed against
PoisonedRAG (k=5, N=5, 5% clean alarm budget):

===================  ==========  ==========
detector             black-box   white-box
===================  ==========  ==========
passage-agreement       0.989       0.990
query-similarity        0.426       0.629
perplexity              0.000       0.557
knn-isolation           0.117       0.062
===================  ==========  ==========

Perplexity is *worse than useless* on black-box poison (fluent LLM prose) and
genuinely informative on white-box poison (HotFlip gibberish). An investigator
who does not know which attack they are facing — the whole premise of this
project — cannot pick the right single detector in advance, which is exactly the
case fusion is for.

The scale problem
-----------------
Raw scores are not comparable: a mean-cosine agreement score lives in ``[-1, 1]``
while a perplexity is unbounded and heavy-tailed. Averaging them directly lets
whichever detector happens to have the widest range dominate.

This module normalises each detector against its own **clean-corpus**
distribution before combining. That keeps the calibration honest: the clean
reference is something an investigator can plausibly obtain (a corpus believed
untampered, or a held-out slice), whereas tuning on poisoned data would leak the
answer the detector is supposed to produce.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, get_detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["HybridDetector"]

#: Signals fused by default. Chosen to span different artifacts — semantic
#: redundancy, retrieval geometry and text fluency — so that an attack evading
#: one still trips another.
DEFAULT_MEMBERS = ("passage-agreement", "query-similarity", "perplexity")


@register_detector("hybrid", category="rag-defense")
class HybridDetector(Detector):
    """Fuse several ``rag-defense`` detectors into a single per-passage score.

    Call :meth:`fit` with clean passages before scoring: it records each member's
    score distribution on clean data, which is what makes the members
    commensurable. Scoring without fitting falls back to per-call rank
    normalisation, which still works but loses cross-query comparability — a
    warning is emitted, because silently degrading here would be invisible in
    the numbers.

    Args:
        members: Registry keys to fuse. Unavailable members are skipped with a
            warning rather than raising, so a missing optional dependency
            degrades the ensemble instead of breaking the run.
        combine: How member scores are pooled.

            * ``"fisher"`` (**default**) — convert each member's score to an
              empirical right-tail p-value against its clean distribution, then
              pool with Fisher's method, ``-2 * sum(log p)``. Scale-free by
              construction: on clean input every p-value is uniform, so a member
              that is uninformative on a given sample contributes noise rather
              than bias, and a member that fires hard dominates. This is what
              lets a specialist (perplexity, blind to fluent poison) sit in the
              ensemble without dragging it.
            * ``"stouffer"`` — same p-values, pooled as
              ``sum(Phi^-1(1 - p)) / sqrt(M)``. Similar to Fisher but weights the
              bulk of the distribution more and the extreme tail less.
            * ``"mean"`` — average of robust z-scores. Simple, but a heavy-tailed
              member distorts the average and a blind member pulls poison scores
              toward the clean centre.
            * ``"max"`` — any-detector-fires on z-scores. Measured to be *worse*
              here: heavy-tailed members produce large z on clean passages too,
              which raises the calibrated threshold and costs recall.
            * ``"median"`` — robust to one member being wrong, but needs a
              majority to agree, so it wastes a specialist.
        robust: Normalise with median/MAD instead of mean/std. On by default: a
            clean corpus still contains genuine outliers, and a handful of them
            would inflate a standard deviation and flatten the scores.
        store, encoder, model, tokenizer: Forwarded to each member.
    """

    name = "hybrid"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False          # works unfitted, but calibration is recommended

    def __init__(
        self,
        *,
        members: Sequence[str] = DEFAULT_MEMBERS,
        combine: str = "fisher",
        robust: bool = True,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(threshold=threshold, members=list(members), combine=combine,
                         robust=robust, **config)
        valid = {"fisher", "stouffer", "mean", "max", "median"}
        if combine not in valid:
            raise ValueError(f"combine must be one of {sorted(valid)}, got {combine!r}")
        self.combine = combine
        self.robust = robust

        self.members: dict[str, Detector] = {}
        for key in members:
            try:
                self.members[key] = get_detector(
                    key, store=store, encoder=encoder, model=model, tokenizer=tokenizer,
                )
            except Exception as exc:
                logger.warning("hybrid: skipping member %r (%s)", key, exc)
        if not self.members:
            raise LookupError(
                f"none of the requested members {list(members)} could be constructed; "
                f"check that their dependencies are installed"
            )
        self._centre: dict[str, float] = {}
        self._clean: dict[str, np.ndarray] = {}
        self._scale: dict[str, float] = {}

    def fit(self, clean_data: Any, **kwargs: Any) -> HybridDetector:
        """Record each member's clean-corpus score distribution.

        Args:
            clean_data: Sequence of ``(query, passages)`` pairs from a corpus
                believed clean.

        Returns:
            ``self``.
        """
        for key, det in self.members.items():
            scores = np.concatenate([
                np.asarray(det.score(sample), dtype=np.float64).reshape(-1)
                for sample in clean_data
            ]) if clean_data else np.zeros(1)
            if self.robust:
                centre = float(np.median(scores))
                scale = float(1.4826 * np.median(np.abs(scores - centre)))
            else:
                centre, scale = float(np.mean(scores)), float(np.std(scores))
            self._centre[key] = centre
            self._scale[key] = scale if scale > 1e-9 else 1.0
            # Sorted clean scores back the empirical CDF used by the p-value
            # combiners. Kept whole rather than summarised: the tail is exactly
            # the part a summary statistic would throw away.
            self._clean[key] = np.sort(scores)
        self._fitted = True
        return self

    def _normalise(self, key: str, scores: np.ndarray) -> np.ndarray:
        """Map a member's raw scores onto the shared, clean-calibrated scale."""
        if key in self._centre:
            return (scores - self._centre[key]) / self._scale[key]
        # Unfitted: rank-normalise within this call so members remain comparable.
        order = scores.argsort().argsort().astype(np.float64)
        return order / max(1, scores.size - 1)

    def _pvalue(self, key: str, scores: np.ndarray) -> np.ndarray:
        """Empirical right-tail p-value of ``scores`` against clean data.

        ``(1 + #{clean >= s}) / (1 + n)`` — the add-one keeps the p-value
        strictly positive so ``log p`` stays finite for a score beyond every
        clean observation, which is exactly the case that matters most.
        """
        ref = self._clean.get(key)
        if ref is None or ref.size == 0:
            order = scores.argsort().argsort().astype(np.float64)
            return 1.0 - order / max(1, scores.size - 1) * 0.999 - 0.0005
        n = ref.size
        ge = n - np.searchsorted(ref, scores, side="left")
        return (1.0 + ge) / (1.0 + n)

    def score(self, samples: Any) -> np.ndarray:
        """Fused suspicion score for one query's retrieved passages.

        Args:
            samples: ``(query, passages)``.

        Returns:
            One score per passage, higher = more suspicious.
        """
        if not self._fitted:
            logger.warning(
                "hybrid.score called before fit(); falling back to per-call rank "
                "normalisation, so scores are not comparable across queries"
            )
        raw = {key: np.asarray(det.score(samples), dtype=np.float64).reshape(-1)
               for key, det in self.members.items()}

        if self.combine in {"fisher", "stouffer"}:
            pvals = np.vstack([np.clip(self._pvalue(k, v), 1e-12, 1.0 - 1e-12)
                               for k, v in raw.items()])
            if self.combine == "fisher":
                return -2.0 * np.log(pvals).sum(axis=0)
            from scipy.stats import norm

            return norm.isf(pvals).sum(axis=0) / np.sqrt(pvals.shape[0])

        stacked = np.vstack([self._normalise(k, v) for k, v in raw.items()])
        if self.combine == "max":
            return stacked.max(axis=0)
        if self.combine == "median":
            return np.median(stacked, axis=0)
        return stacked.mean(axis=0)

    def describe(self) -> dict:
        """Ensemble identity plus each member's calibration."""
        base = super().describe()
        base["members"] = {
            k: {"centre": self._centre.get(k), "scale": self._scale.get(k)}
            for k in self.members
        }
        base["combine"] = self.combine
        return base
