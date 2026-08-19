"""Two-stage detection cascade — cheap filter, then an expensive adjudicator.

Why a cascade
-------------
At corpus scale the problem is not ranking, it is the *top* of the ranking.
``passage-agreement`` reaches ROC-AUC 0.979 over 20,123 documents with 100
injected (0.497% base rate) and still only PR-AUC 0.164: at a 1% alarm budget it
flags 257 documents to find 55 true positives (precision 0.214), and at 0.1% it
finds none. The reason is not that the statistic is weak but that its top is
*saturated with genuine near-duplicates* — NQ splits Wikipedia articles into
overlapping chunks, so thousands of real passages have a twin at cosine 1.0000
and are indistinguishable from an injected cluster by cross-passage agreement
alone.

A cascade attacks exactly that. Stage 1 runs over everything and keeps a
shortlist; stage 2 runs **only on the shortlist** and is asked one narrower
question: *of the passages stage 1 could not tell apart, which are poison?*
Two consequences, and only the first is usually advertised:

* **Cost.** Stage 2 sees ``shortlist_frac`` of the corpus, so an LLM-based
  statistic that is unaffordable over 20k documents becomes affordable over
  1,000. :attr:`stage2_calls` records this so the saving is measured, not
  claimed.
* **False alarms.** Stage 2 only has to *reject stage 1's mistakes*. It is
  calibrated against the clean passages that survive stage 1 (see
  ``calibrate_stage2``), not against the corpus at large, so "unusual for a
  near-duplicate chunk" is the question it is actually asked.

The construction is the classical detection cascade of Viola & Jones ("Rapid
Object Detection using a Boosted Cascade of Simple Features", CVPR 2001), and
the same shape as retrieve-then-rerank in IR (Nogueira & Cho, "Passage
Re-ranking with BERT", arXiv:1901.04085): a cheap high-recall stage defines the
candidate set, an expensive stage orders it.

**Recall is hard-capped by stage 1 — this is the price, and it is not optional.**
Anything stage 1 ranks below the shortlist cut can never be recovered: stage 2 is
never shown it, and no combination rule can resurrect it. The cascade's recall is
therefore *at most* stage 1's recall at the shortlist threshold, for every alarm
budget. With ``shortlist_frac = 0.05`` on the corpus above, stage 1's recall at
the 5% cut is the ceiling for everything this detector can ever report; the
cascade can only spend that ceiling more precisely. A cascade that improves
precision by shrinking ``shortlist_frac`` is buying it with recall, and the
shortlist fraction should be reported next to every number this detector
produces.

Known failure modes
-------------------
* **A blind stage 2 is worse than useless under ``combine="stage2"``.**
  ``perplexity`` measures 0.000 detection on fluent black-box PoisonedRAG text.
  Ranking the shortlist by it alone would replace stage 1's ordering with noise
  and lose recall *inside* the shortlist too. This is why the default is
  ``"product"``: under it an uninformative stage contributes near-uniform
  p-values and the ranking degrades gracefully back toward stage 1's, rather
  than being overwritten.
* **Stage 2 must score passages independently.** Restricting a *set-dependent*
  detector (``passage-agreement``, ``ragdefender``, ``ragdefender-score``) to the
  shortlist changes what it computes — its neighbours are gone. A warning is
  emitted; the number produced is not the same statistic that detector reports
  on a full retrieved set.
* **Fisher's product assumes independent stages under the null.** Two
  retrieval-geometry statistics are correlated, so the combined score is
  anti-conservative and is a *ranking* statistic, not a calibrated p-value. It is
  never reported as one here.
* **Conditional calibration can starve.** ``calibrate_stage2="conditional"``
  gives stage 2 only ``shortlist_frac`` of the clean data, so the clean reference
  can be small and its tail coarse; the p-value floor is ``1/(1+n_clean)``, which
  bounds how extreme any score can look. A warning fires below
  ``min_clean_stage2``.
* **Regime mismatch at k=5.** With five retrieved passages, ``0.05 * 5 < 1`` and
  the ``min_shortlist`` floor takes over: the cascade degenerates to "hand the
  single most agreement-suspicious passage to stage 2". The retrieved-set regime
  is not what this detector is for — ``passage-agreement`` already reaches 0.989
  detection there.
* **The flat tail breaks global ROC-AUC** (see ``tail``).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .base import Detector, get_detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["CascadeDetector"]

#: Detectors whose score for a passage depends on the other passages in the
#: call. Restricting one to the shortlist silently redefines it.
_SET_DEPENDENT = frozenset({"passage-agreement", "ragdefender", "ragdefender-score"})
_COMBINE = ("product", "sum", "stage2")
_TAIL = ("floor", "stage1")
_PCLIP = 1e-12


def _pvalue(ref: np.ndarray | None, scores: np.ndarray) -> np.ndarray:
    """Empirical right-tail p-value of ``scores`` against sorted clean ``ref``.

    ``(1 + #{clean >= s}) / (1 + n)`` — the same add-one estimator ``hybrid``
    uses, kept identical so the two fusions are comparable. The add-one keeps
    ``log p`` finite for a score beyond every clean observation, which is the
    case that matters most here.

    Args:
        ref: Sorted clean reference scores, or ``None``/empty when unfitted.
        scores: Scores to convert.

    Returns:
        p-values in ``(0, 1]``. Unfitted, falls back to within-call ranks, which
        are comparable inside one call but not across calls.
    """
    if ref is None or ref.size == 0:
        order = scores.argsort().argsort().astype(np.float64)
        return 1.0 - order / max(1, scores.size)
    ge = ref.size - np.searchsorted(ref, scores, side="left")
    return (1.0 + ge) / (1.0 + ref.size)


@register_detector("cascade", category="rag-defense")
class CascadeDetector(Detector):
    """Shortlist with a cheap detector, adjudicate the shortlist with an expensive one.

    Args:
        stage1: Registry key of the cheap detector run over *every* passage.
            Default ``"passage-agreement"`` — one encoder pass and a matmul, and
            the best single detector this testbed has at corpus scale.
        stage2: Registry key of the expensive detector run over the shortlist
            only, or ``None`` for a shortlist-only ablation that isolates how
            much of any gain is the shortlist rather than stage 2. Default
            ``"perplexity"``; ``"gmtp"`` is the other natural choice.
        shortlist_frac: Fraction of each call's passages handed to stage 2.
            Doubles as the recall ceiling: whatever stage 1 misses at this cut is
            lost for good.
        min_shortlist: Lower bound on the shortlist size, so a small call still
            reaches stage 2 at all.
        shortlist_pmax: Optional absolute cap on the stage-1 clean p-value. A
            passage must be in the top ``shortlist_frac`` *and* have
            ``p1 <= shortlist_pmax``. Use it to stop the cascade spending stage-2
            compute on an unremarkable corpus; leave ``None`` for a pure budget.
            A call in which it rejects everything returns that whole call in the
            tail band, so it is meant for the corpus-scale regime — where a scan
            is one large call — not for per-query scoring.
        combine: How the two stages are pooled *inside* the shortlist.

            * ``"product"`` (**default**) — ``-(log p1 + log p2)`` over the two
              clean-calibrated p-values, i.e. Fisher's statistic up to a factor
              of two. Defensible as the default for three reasons: it is
              scale-free, so a heavy-tailed perplexity cannot dominate a bounded
              cosine; a stage that is *blind* to the attack contributes
              near-uniform p-values and therefore noise rather than bias, which
              is exactly the situation with fluent poison and perplexity; and it
              keeps stage 1's ordering as the fallback when stage 2 says
              nothing. Its honest limitation is that where stage-1 p-values are
              saturated at the floor ``1/(1+n_clean)`` — common at the top of a
              near-duplicate-heavy corpus — the stage-1 term is a constant and
              this reduces to ``"stage2"``.
            * ``"sum"`` — sum of robust (median/MAD) z-scores. Retains stage-1
              resolution where p-values saturate, but a heavy-tailed member
              distorts the sum.
            * ``"stage2"`` — stage 2's p-value alone (``-log p2``); monotone in
              its raw score. The strict reading of a cascade, and the right
              choice only when stage 2 is known to dominate stage 1 on the
              shortlist. Discards stage 1 entirely, so a blind stage 2 destroys
              recall inside the shortlist.
        tail: What happens below the shortlist cut.

            * ``"floor"`` (**default**) — every non-shortlisted passage gets one
              finite floor value, ``min(shortlisted) - 1``. Faithful to the
              cascade: those passages were never examined, so they are not
              ranked. A **finite** floor rather than ``-inf`` because ``-inf``
              breaks AUC integration, any downstream mean/std or median/MAD
              normalisation (``-inf`` poisons the centre, ``0 * -inf`` is NaN),
              and score histograms. Consequence to report: global ROC-AUC is not
              comparable to a full-corpus detector's, because ~95% of the corpus
              ties.
            * ``"stage1"`` — break those ties with stage 1's own ordering,
              shifted into ``[floor - 1, floor)`` so every shortlisted passage
              still outranks every rejected one. Free (stage 1 already scored
              everything) and **identical to ``"floor"`` at any alarm budget at
              or below the shortlist size**; it only makes global AUC
              meaningful. Use it for AUC comparisons, ``"floor"`` for the strict
              cascade view.
        calibrate_stage2: ``"conditional"`` (default) calibrates stage 2 only on
            clean passages that *pass stage 1's shortlist rule* — the population
            it will actually see, and the reason the cascade can cut false
            alarms rather than merely re-rank. ``"marginal"`` calibrates on all
            clean passages: more samples, wrong reference population, and it
            forfeits the calibration-time compute saving.
        min_clean_stage2: Warn below this many clean stage-2 observations.
        robust: Use median/MAD instead of mean/std for ``combine="sum"``.
        store, encoder, model, tokenizer: Forwarded to both stages.
        threshold: Left ``None``; the score is a Fisher-style statistic on an
            arbitrary scale, so the caller calibrates it to an alarm budget.
    """

    name = "cascade"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False  # works unfitted; calibration is what makes it a cascade

    def __init__(
        self,
        *,
        stage1: str = "passage-agreement",
        stage2: str | None = "perplexity",
        shortlist_frac: float = 0.05,
        min_shortlist: int = 1,
        shortlist_pmax: float | None = None,
        combine: str = "product",
        tail: str = "floor",
        calibrate_stage2: str = "conditional",
        min_clean_stage2: int = 32,
        robust: bool = True,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, stage1=stage1, stage2=stage2,
            shortlist_frac=shortlist_frac, min_shortlist=min_shortlist,
            shortlist_pmax=shortlist_pmax, combine=combine, tail=tail,
            calibrate_stage2=calibrate_stage2, robust=robust, **config,
        )
        if combine not in _COMBINE:
            raise ValueError(f"combine must be one of {list(_COMBINE)}, got {combine!r}")
        if tail not in _TAIL:
            raise ValueError(f"tail must be one of {list(_TAIL)}, got {tail!r}")
        if calibrate_stage2 not in ("conditional", "marginal"):
            raise ValueError(
                f"calibrate_stage2 must be 'conditional' or 'marginal', got "
                f"{calibrate_stage2!r}"
            )
        if not 0.0 < shortlist_frac <= 1.0:
            raise ValueError(f"shortlist_frac must be in (0, 1], got {shortlist_frac}")

        self.shortlist_frac = float(shortlist_frac)
        self.min_shortlist = max(1, int(min_shortlist))
        self.shortlist_pmax = shortlist_pmax
        self.combine = combine
        self.tail = tail
        self.calibrate_stage2 = calibrate_stage2
        self.min_clean_stage2 = int(min_clean_stage2)
        self.robust = robust

        kw = dict(store=store, encoder=encoder, model=model, tokenizer=tokenizer)
        self.stage1_key, self.stage2_key = stage1, stage2
        self.stage1: Detector = get_detector(stage1, **kw)
        self.stage2: Detector | None = get_detector(stage2, **kw) if stage2 else None
        if stage2 in _SET_DEPENDENT:
            logger.warning(
                "cascade: stage2=%r is set-dependent; restricted to the shortlist it no "
                "longer computes the statistic it reports on a full retrieved set",
                stage2,
            )
        if stage2 is None and combine != "stage2":
            logger.info("cascade: stage2=None — combine=%r ignored, ranking on stage 1", combine)

        self._clean1: np.ndarray | None = None
        self._clean2: np.ndarray | None = None
        self._loc: dict[str, float] = {}
        self._scale: dict[str, float] = {}
        self.reset_counters()

    # -- compute accounting ------------------------------------------------ #
    def reset_counters(self) -> None:
        """Zero the per-passage call counters (scoring and calibration)."""
        self.stage1_calls = 0
        self.stage2_calls = 0
        self.fit_stage1_calls = 0
        self.fit_stage2_calls = 0

    @property
    def stage2_call_fraction(self) -> float:
        """Share of scored passages that reached stage 2 — the measured saving."""
        return self.stage2_calls / max(1, self.stage1_calls)

    # -- calibration ------------------------------------------------------- #
    def fit(self, clean_data: Any, **kwargs: Any) -> CascadeDetector:
        """Record each stage's clean score distribution.

        Two passes are needed and ``clean_data`` is materialised as a list: the
        stage-1 pass builds the reference that *defines* the shortlist rule, and
        only then can the conditional stage-2 population be identified.

        Args:
            clean_data: Sequence of ``(query, passages)`` pairs from a corpus
                believed untampered.

        Returns:
            ``self``.
        """
        samples = list(clean_data)
        for stage in (self.stage1, self.stage2):
            if stage is not None and stage.requires_fit and not stage.fitted:
                stage.fit(samples, **kwargs)

        per_sample: list[np.ndarray] = []
        for sample in samples:
            s1 = np.asarray(self.stage1.score(sample), dtype=np.float64).reshape(-1)
            self.fit_stage1_calls += s1.size
            per_sample.append(s1)
        self._clean1 = np.sort(np.concatenate(per_sample)) if per_sample else np.zeros(0)

        clean2: list[np.ndarray] = []
        if self.stage2 is not None:
            for sample, s1 in zip(samples, per_sample, strict=False):  # built in lockstep above
                if self.calibrate_stage2 == "conditional":
                    idx = self._shortlist_indices(s1)
                    if idx.size == 0:
                        continue
                    sub = (sample[0], [sample[1][i] for i in idx])
                else:
                    sub = sample
                s2 = np.asarray(self.stage2.score(sub), dtype=np.float64).reshape(-1)
                self.fit_stage2_calls += s2.size
                clean2.append(s2)
        self._clean2 = np.sort(np.concatenate(clean2)) if clean2 else np.zeros(0)

        if self.stage2 is not None and self._clean2.size < self.min_clean_stage2:
            logger.warning(
                "cascade: only %d clean stage-2 observations (< %d); the p-value floor is "
                "1/%d, so no shortlisted passage can score more extreme than that",
                self._clean2.size, self.min_clean_stage2, self._clean2.size + 1,
            )
        for key, ref in (("stage1", self._clean1), ("stage2", self._clean2)):
            if ref.size:
                loc = float(np.median(ref)) if self.robust else float(np.mean(ref))
                scale = (float(1.4826 * np.median(np.abs(ref - loc))) if self.robust
                         else float(np.std(ref)))
                self._loc[key], self._scale[key] = loc, scale if scale > 1e-9 else 1.0
        self._fitted = True
        return self

    # -- shortlisting ------------------------------------------------------ #
    def _shortlist_indices(self, s1: np.ndarray) -> np.ndarray:
        """Indices stage 2 will see, given stage-1 scores for one call.

        Selection ranks by the **raw** stage-1 score, not the p-value, even
        though the shortlist is defined as a clean-calibrated quantile: the
        empirical p-value is a step function of the raw score, so at the top of a
        near-duplicate-saturated corpus hundreds of passages share the floor
        value ``1/(1+n_clean)`` and a p-value argsort would break those ties by
        array position. The two orderings are otherwise identical — the ECDF is
        monotone — so this costs nothing and removes an arbitrary tiebreak. The
        p-value is still what ``shortlist_pmax`` gates on.
        """
        n = s1.size
        if n == 0:
            return np.zeros(0, dtype=np.intp)
        k = min(n, max(self.min_shortlist, math.ceil(self.shortlist_frac * n)))
        idx = np.argsort(-s1, kind="stable")[:k]
        if self.shortlist_pmax is not None:
            # p-values over the whole call, then indexed — so the unfitted rank
            # fallback is taken against every passage, not just the top-k.
            idx = idx[_pvalue(self._clean1, s1)[idx] <= self.shortlist_pmax]
        return np.sort(idx)

    def shortlist(self, samples: Any) -> np.ndarray:
        """Indices of the passages stage 2 would be spent on.

        Args:
            samples: ``(query, passages)``.

        Returns:
            Ascending integer indices into ``passages``. Running stage 1 here
            counts toward :attr:`stage1_calls`, exactly as scoring does — so use
            this *instead of* :meth:`score` when only the candidate set is
            wanted, not in addition to it, or the accounting double-counts
            stage 1.
        """
        s1 = np.asarray(self.stage1.score(samples), dtype=np.float64).reshape(-1)
        self.stage1_calls += s1.size
        return self._shortlist_indices(s1)

    # -- scoring ----------------------------------------------------------- #
    def score(self, samples: Any) -> np.ndarray:
        """Cascade suspicion score for one query's passages.

        Args:
            samples: ``(query, passages)``.

        Returns:
            One float per passage, higher = more suspicious. Shortlisted
            passages carry the combined statistic; the rest sit at or below a
            finite floor (see ``tail``).

        Raises:
            TypeError: If ``samples`` is not a ``(query, passages)`` pair.
        """
        try:
            query, passages = samples
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{type(self).__name__}.score expects (query, passages), got "
                f"{type(samples).__name__}"
            ) from exc
        if not self._fitted:
            logger.warning(
                "cascade.score called before fit(); p-values fall back to within-call ranks, "
                "so the shortlist is a per-call quantile and scores are not comparable "
                "across calls"
            )

        s1 = np.asarray(self.stage1.score(samples), dtype=np.float64).reshape(-1)
        n = s1.size
        self.stage1_calls += n
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        idx = self._shortlist_indices(s1)
        p1 = np.clip(_pvalue(self._clean1, s1)[idx], _PCLIP, 1.0)
        if idx.size == 0:
            # shortlist_pmax rejected the whole call. Everything falls to the
            # tail band, keyed off the minimum a -log p score can take (0), so
            # the band matches the general case instead of being a special zero.
            combined = np.zeros(0, dtype=np.float64)
        elif self.stage2 is None:
            combined = -np.log(p1)
        else:
            sub = (query, [passages[i] for i in idx])
            s2 = np.asarray(self.stage2.score(sub), dtype=np.float64).reshape(-1)
            self.stage2_calls += s2.size
            if self.combine == "sum":
                combined = (self._z("stage1", s1[idx]) + self._z("stage2", s2))
            else:
                p2 = np.clip(_pvalue(self._clean2, s2), _PCLIP, 1.0)
                combined = -np.log(p2) if self.combine == "stage2" else -np.log(p1 * p2)

        out = np.empty(n, dtype=np.float64)
        floor = (float(np.min(combined)) if combined.size else 0.0) - 1.0
        out[idx] = combined
        rest = np.setdiff1d(np.arange(n, dtype=np.intp), idx, assume_unique=True)
        if rest.size:
            if self.tail == "stage1":
                # Stage 1's own ordering, squashed into [floor - 1, floor) so the
                # whole rejected band still sits strictly under the shortlist.
                rank = s1[rest].argsort().argsort().astype(np.float64) / rest.size
                out[rest] = floor - 1.0 + rank
            else:
                out[rest] = floor
        return out

    def _z(self, key: str, scores: np.ndarray) -> np.ndarray:
        """Clean-calibrated robust z-score; identity-ish when unfitted."""
        if key not in self._loc:
            return scores - float(np.median(scores))
        return (scores - self._loc[key]) / self._scale[key]

    def describe(self) -> dict:
        """Identity, stage configuration, calibration sizes and compute counters."""
        base = super().describe()
        base["stages"] = {
            "stage1": {"key": self.stage1_key, "clean_n": int(getattr(self._clean1, "size", 0))},
            "stage2": {"key": self.stage2_key, "clean_n": int(getattr(self._clean2, "size", 0)),
                       "calibration": self.calibrate_stage2},
        }
        base["shortlist"] = {"frac": self.shortlist_frac, "min": self.min_shortlist,
                             "pmax": self.shortlist_pmax}
        base["combine"] = self.combine
        base["tail"] = self.tail
        base["calls"] = {
            "stage1": self.stage1_calls, "stage2": self.stage2_calls,
            "stage2_fraction": round(self.stage2_call_fraction, 6),
            "fit_stage1": self.fit_stage1_calls, "fit_stage2": self.fit_stage2_calls,
        }
        return base
