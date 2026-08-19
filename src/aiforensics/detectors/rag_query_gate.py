"""Query-level gate — suppress every passage of a retrieved set that looks normal.

The observation
---------------
Poison is *concentrated*. PoisonedRAG (Zou, Geng, Wang, Jia. "PoisonedRAG:
Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language
Models." USENIX Security 2025) targets a chosen list of questions and injects
``N`` passages per target; every other question in the workload retrieves an
untouched set. Measured on this testbed at corpus scale (20,123 documents, 100
injected, 0.497% base rate) the per-passage detector reaches ROC-AUC 0.979 but
PR-AUC 0.164 — and the false alarms that destroy the precision are spread over
*never-attacked* queries, whose top-k happens to contain a genuine
near-duplicate pair. NQ chunks long Wikipedia articles into overlapping windows,
so a clean retrieved set with two passages at cosine 1.0000 is ordinary, not
evidence.

A per-passage detector cannot see that, because it only ever looks inside one
passage's neighbourhood. A *query-level* statistic can: it asks whether the
retrieved set as a whole departs from what clean retrieval looks like. If it
does not, nothing in that set should be allowed to consume the analyst's alarm
budget, no matter how high one passage scored.

The mechanism
-------------
1. Score the retrieved set with a wrapped per-passage detector (default
   ``"passage-agreement"``, the strongest single signal here).
2. Pool those ``k`` scores into one query statistic (:data:`AGGREGATES`).
3. :meth:`fit` records the clean distribution of that statistic, so the gate can
   be placed at an explicit **query-level** alarm budget ``query_fpr``.
4. :meth:`score` returns the per-passage scores unchanged for a query that
   passes the gate, and shifted down by a large finite constant for a query that
   does not — so no passage from a gated query can outrank any passage from a
   gated-in query under a global threshold.

This changes nothing *within* a query: the ranking of the ``k`` passages is
identical, so per-query ROC-AUC is untouched by construction. The whole effect
is on cross-query metrics — PR-AUC and precision at a global alarm budget — which
is exactly where the corpus-scale problem lives. Reporting a per-query ROC-AUC
for this detector and calling it unchanged is not a null result, it is a
tautology; report the corpus-level numbers.

Why ``"fisher"`` is the default aggregate
-----------------------------------------
Convert each passage score to an empirical right-tail p-value against the clean
passage-score distribution and pool with Fisher's method,
``-2 * sum(log p_i)`` (Fisher. *Statistical Methods for Research Workers*, 1925).
It answers "how much anomaly is there in this set *in total*", which separates
the two cases that ``max`` conflates:

* a clean set with one genuine duplicate pair — two high scores, ``k - 2``
  ordinary ones, moderate pooled evidence;
* an attacked set with ``N = 5`` mutually near-duplicate injections — five high
  scores, large pooled evidence.

``max`` is saturated by the duplicate pair and therefore gates in precisely the
queries this detector exists to gate out. The chi-square null with ``2k``
degrees of freedom is only approximate here — max-sim scores inside one
retrieved set are strongly dependent (a duplicate pair produces *two* high
scores from *one* event), so the pooled statistic is over-dispersed. The gate
threshold is therefore taken from the **empirical** clean distribution of the
statistic, never from a chi-square quantile; the chi-square tail is used only as
a ``k``-invariant re-parameterisation so that sets of different sizes are
comparable.

Known failure modes (stated up front, because they are results)
---------------------------------------------------------------
* **Useless when every query is attacked.** The gate is a *relative* statement —
  this query versus the clean query distribution. Under a saturation attack every
  query is anomalous, the clean reference itself (if drawn from the same
  workload) is poisoned, the threshold inflates, and the gate suppresses true
  positives instead of false alarms. It buys precision exactly in proportion to
  how concentrated the attack is, which is the regime PoisonedRAG-style targeted
  attacks live in and *not* the regime of an indiscriminate corpus flood.
* **A wrongly-gated query loses all of its true positives at once.** This is a
  new, coarser error mode than a per-passage miss: query-level recall is a hard
  ceiling on passage-level recall. At ``query_fpr = 0.10`` roughly one clean
  query in ten still passes, and whatever fraction of attacked queries falls
  below the gate is lost outright. Report query-level recall alongside the
  passage-level number, or the ceiling stays invisible.
* **Weakest exactly where the per-passage detector is weakest.** A lone injected
  passage (``N = 1``) raises one score in an otherwise ordinary set; ``fisher``
  and ``mean`` dilute it across ``k`` terms and the query may be gated away. If
  the threat model is single-passage injection, use ``aggregate="max"`` and
  accept the duplicate-pair false alarms.
* **Needs a query workload.** A document that no probe query ever retrieves is
  never scored at all. This is a triage tool for a retrieval log or a probe set,
  not a substitute for the whole-corpus nearest-neighbour scan.
* **Inherits the wrapped detector's blind spots.** Gating cannot create signal;
  if the member scores poison and clean identically, so does the query
  statistic.
* **In-sample p-values.** The clean passage scores used for the p-values are the
  same ones used to calibrate the gate threshold, so both are in-sample. With a
  few hundred clean queries the bias is small, but a held-out calibration split
  is the honest version when the clean pool is small.

Note:
    ``base._BUILTIN_MODULES`` does not list this module, so ``get_detector`` only
    resolves ``"query-gate"`` once ``aiforensics.detectors.rag_query_gate`` has
    been imported (importing it is enough — registration is an import side
    effect).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .base import Detector, get_detector, register_detector

logger = logging.getLogger(__name__)

#: Supported query-level aggregates, see :class:`QueryGateDetector`.
AGGREGATES: tuple[str, ...] = ("fisher", "max", "mean", "top2mean")

#: Shift applied to a gated query when no clean scores are available to size one.
FALLBACK_SHIFT = 1.0e3

__all__ = ["AGGREGATES", "FALLBACK_SHIFT", "QueryGateDetector"]


@register_detector("query-gate", category="rag-defense")
class QueryGateDetector(Detector):
    """Two-level detector: gate on the query, rank inside it with a member.

    ``samples`` is ``(query, passages)`` as everywhere else in this category.
    Returns one score per passage, higher = more suspicious; the member's own
    scores for a query that clears the gate, and those scores minus
    :attr:`shift` for a query that does not.

    Args:
        member: Registry key of the per-passage detector to wrap. Default
            ``"passage-agreement"``. Anything in ``rag-defense`` works,
            including ``"hybrid"`` — :meth:`fit` forwards the clean data to the
            member first, so a member that needs calibration gets it.
        aggregate: How the ``k`` passage scores become one query statistic.

            * ``"fisher"`` (**default**) — pooled empirical p-values,
              ``-2 * sum(log p)``, re-expressed as ``-log10`` of the chi-square
              tail so different ``k`` are comparable. Uses the whole set, which
              is what distinguishes a five-passage injected cluster from a
              two-passage genuine duplicate. See the module docstring.
            * ``"max"`` — the single most suspicious passage. The right choice
              when only one passage may be injected, and the worst choice for
              precision, since one genuine near-duplicate pair gates the query
              straight in.
            * ``"mean"`` — smooth, but a ``k = 5`` set with one injected passage
              is four-fifths clean, so the signal is diluted.
            * ``"top2mean"`` — mean of the two highest scores. A middle ground
              that demands corroboration (two passages, not one) without
              requiring the whole set to be anomalous.
        query_fpr: Query-level alarm budget. The gate is placed at the
            ``1 - query_fpr`` quantile of the clean query statistic, so by
            construction this fraction of clean queries passes. Default
            ``0.10``: an order-of-magnitude cut in the number of queries that
            can contribute false alarms, while leaving enough headroom that
            attacked queries — which sit far into the tail — still pass.
        shift: Constant subtracted from every passage of a gated query.
            ``None`` (default) sizes it at fit time from the clean score range.
        store, encoder, model, tokenizer: Forwarded to the member.
        threshold: Passage-level decision threshold for :meth:`predict`; the
            gate is a separate, query-level threshold learned by :meth:`fit`.

    Note:
        The shift is **finite** on purpose. ``-inf`` would rank gated passages
        correctly but poison every downstream consumer: ROC/PR routines, robust
        z-scoring and Fisher fusion all produce ``nan`` from ``inf - inf`` or
        ``0 * inf``, quantile-based thresholds collapse, and the within-query
        ranking of a gated set — which is still the right ranking, and is what a
        second-pass review of a gated query would use — is destroyed. A finite
        shift keeps the array a well-behaved float array, keeps every gated
        query's internal order intact, and still guarantees separation as long
        as it exceeds the score range; it is sized at 100x the clean
        interquantile span for that reason.

    Raises:
        ValueError: If ``aggregate`` is unknown or ``query_fpr`` is outside
            ``[0, 1]``.
    """

    name = "query-gate"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False          # degrades to pass-through, loudly, if unfitted

    def __init__(
        self,
        *,
        member: str = "passage-agreement",
        aggregate: str = "fisher",
        query_fpr: float = 0.10,
        shift: float | None = None,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(threshold=threshold, member=member, aggregate=aggregate,
                         query_fpr=query_fpr, shift=shift, **config)
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
        if not 0.0 <= float(query_fpr) <= 1.0:
            raise ValueError(f"query_fpr must lie in [0, 1], got {query_fpr!r}")
        self.aggregate = aggregate
        self.query_fpr = float(query_fpr)

        member_config = dict(config.get("member_config") or {})
        self.member_key = member
        self.member: Detector = get_detector(
            member, store=store, encoder=encoder, model=model, tokenizer=tokenizer,
            **member_config,
        )

        self.shift: float = FALLBACK_SHIFT if shift is None else float(shift)
        self._shift_fixed = shift is not None
        self._clean_passage: np.ndarray = np.empty(0, dtype=np.float64)
        self._clean_query: np.ndarray = np.empty(0, dtype=np.float64)
        self._gate: float | None = None
        self._cache: tuple[Any, np.ndarray] | None = None
        self._warned = False

    # -- calibration -------------------------------------------------------- #
    def fit(self, clean_data: Any, **kwargs: Any) -> QueryGateDetector:
        """Record the clean distributions of the passage and query statistics.

        Args:
            clean_data: Sequence of ``(query, passages)`` pairs from retrieval
                against a corpus believed untampered. A few hundred queries is
                enough to place a 10% gate; a 1% gate needs proportionally more,
                since the threshold then sits on a handful of order statistics.
            **kwargs: Forwarded to the member's ``fit``.

        Returns:
            ``self``.
        """
        data = list(clean_data or ())
        self._cache = None            # member may have been recalibrated below
        try:
            self.member.fit(data, **kwargs)
        except Exception as exc:                       # optional member calibration
            logger.warning("query-gate: member %r fit failed (%s)", self.member_key, exc)

        per_query = [np.asarray(self.member.score(s), dtype=np.float64).reshape(-1)
                     for s in data]
        per_query = [s for s in per_query if s.size]
        if not per_query:
            logger.warning("query-gate: fit() got no usable clean queries; the gate "
                           "stays open and score() is a pass-through")
            self._fitted = True
            return self

        self._clean_passage = np.sort(np.concatenate(per_query))
        if not self._shift_fixed:
            lo, hi = np.quantile(self._clean_passage, [0.001, 0.999])
            self.shift = float(100.0 * max(float(hi - lo), 1e-3) + 1.0)

        # Query statistics are computed *after* the passage reference exists, so
        # the "fisher" aggregate has p-values to pool.
        self._clean_query = np.sort(np.array([self._aggregate(s) for s in per_query]))
        self._gate = float(np.quantile(self._clean_query, 1.0 - self.query_fpr))
        self._fitted = True
        return self

    # -- statistics --------------------------------------------------------- #
    def _pvalue(self, scores: np.ndarray) -> np.ndarray:
        """Empirical right-tail p-value against the clean passage scores.

        ``(1 + #{clean >= s}) / (1 + n)``; the add-one keeps ``log p`` finite for
        a score beyond every clean observation, which is the case that matters
        most. Falls back to a within-set rank when unfitted.
        """
        ref = self._clean_passage
        if ref.size == 0:
            order = scores.argsort().argsort().astype(np.float64)
            return 1.0 - 0.999 * order / max(1, scores.size - 1) - 5e-4
        ge = ref.size - np.searchsorted(ref, scores, side="left")
        return (1.0 + ge) / (1.0 + ref.size)

    def _aggregate(self, scores: np.ndarray) -> float:
        """Pool per-passage scores into one query-level statistic."""
        if scores.size == 0:
            return -math.inf
        if self.aggregate == "max":
            return float(scores.max())
        if self.aggregate == "mean":
            return float(scores.mean())
        if self.aggregate == "top2mean":
            top = np.sort(scores)[-min(2, scores.size):]
            return float(top.mean())

        pvals = np.clip(self._pvalue(scores), 1e-12, 1.0 - 1e-12)
        stat = float(-2.0 * np.log(pvals).sum())
        try:                                            # k-invariant re-expression
            from scipy.stats import chi2
        except Exception:                               # pragma: no cover - optional
            return stat
        tail = float(chi2.sf(stat, df=2 * pvals.size))
        return float(-math.log10(max(tail, 1e-300)))

    def _member_scores(self, samples: Any) -> np.ndarray:
        """Member scores for one sample, memoised across the calls below."""
        try:
            query, passages = samples
            key = (query, tuple(getattr(p, "text", p) for p in passages))
        except (TypeError, ValueError):
            key = None
        if key is not None and self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        scores = np.asarray(self.member.score(samples), dtype=np.float64).reshape(-1)
        if key is not None:
            self._cache = (key, scores)
        return scores

    def query_statistic(self, samples: Any) -> float:
        """Aggregate anomaly statistic for one retrieved set.

        Args:
            samples: ``(query, passages)``.

        Returns:
            The pooled statistic, higher = the set as a whole looks more
            anomalous. Comparable to :attr:`gate_threshold`.
        """
        return self._aggregate(self._member_scores(samples))

    def is_gated(self, samples: Any) -> bool:
        """Whether this query is suppressed (``True`` = every passage shifted down).

        Args:
            samples: ``(query, passages)``.

        Returns:
            ``False`` whenever the gate is unset — an uncalibrated gate must
            never silently discard evidence — and for an empty retrieved set,
            which has nothing to suppress.
        """
        scores = self._member_scores(samples)
        if self._gate is None or scores.size == 0:
            return False
        return bool(self._aggregate(scores) < self._gate)

    @property
    def gate_threshold(self) -> float | None:
        """Clean-calibrated query-level cut, or ``None`` if :meth:`fit` never ran."""
        return self._gate

    # -- scoring ------------------------------------------------------------ #
    def score(self, samples: Any) -> np.ndarray:
        """Gated per-passage suspicion scores for one query.

        Args:
            samples: ``(query, passages)``.

        Returns:
            ``(len(passages),)`` float array, higher = more suspicious. Member
            scores as-is for a query that clears the gate; member scores minus
            :attr:`shift` otherwise, which preserves the within-query ranking
            while pushing the whole set below every ungated passage.
        """
        scores = self._member_scores(samples)
        if self._gate is None:
            if not self._warned:
                logger.warning(
                    "query-gate.score called before fit(); passing member %r through "
                    "unchanged, so this detector contributes nothing",
                    self.member_key,
                )
                self._warned = True
            return scores
        if scores.size == 0 or self._aggregate(scores) >= self._gate:
            return scores
        return scores - self.shift

    def describe(self) -> dict:
        """Detector identity plus the learned gate, for the run manifest."""
        base = super().describe()
        base.update(
            member=self.member_key,
            aggregate=self.aggregate,
            query_fpr=self.query_fpr,
            gate_threshold=self._gate,
            shift=self.shift,
            n_clean_queries=int(self._clean_query.size),
            n_clean_passages=int(self._clean_passage.size),
        )
        return base
