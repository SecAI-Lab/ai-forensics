"""FDR-controlled triage — turn any detector's scores into q-values.

Why the false *discovery* rate, not the false *positive* rate
-------------------------------------------------------------
At corpus scale the base rate is 0.497% (100 injected among 20,123 documents).
A 1% FPR sounds tight and is useless: it hands the analyst ~200 clean documents
for every 100 poisoned ones that exist at all. The quantity an analyst actually
budgets is the **false discovery rate** — of the documents in the report, what
fraction are clean — and that is exactly what Benjamini-Hochberg controls.

This detector is a wrapper, not a new signal. It takes another ``rag-defense``
detector (default ``"hybrid"``), converts its raw scores into empirical
right-tail p-values against a clean reference distribution, and exposes the
multiple-testing machinery on top: :meth:`FDRControlDetector.qvalues` per
document, :meth:`FDRControlDetector.flag` for a boolean decision at level
``alpha``, and :meth:`FDRControlDetector.scan_corpus` which pools p-values over
many queries before applying BH.

Why pooling matters
-------------------
BH is a statement about a *family* of tests: it rejects the largest set whose
expected proportion of nulls is at most ``alpha``. Applied per query with
``k = 5`` it is nearly vacuous — with five tests the least significant rejection
threshold is ``alpha`` itself and the most stringent is ``alpha/5``, so BH
degenerates into a slightly-softened Bonferroni on a handful of correlated
passages. :meth:`FDRControlDetector.scan_corpus` therefore scores every query's
top-k, deduplicates to one p-value per *document*, and runs BH once over the
whole family. That is also the unit an analyst triages: a document, not a
(query, document) pair.

Assumptions, and where they break here
--------------------------------------
BH controls FDR at ``alpha`` when the p-values are independent (Benjamini &
Hochberg, *JRSS-B* 57(1):289-300, 1995) or positively regression dependent on
the subset of nulls, PRDS (Benjamini & Yekutieli, *Ann. Statist.* 29(4):
1165-1188, 2001). **Neither holds on this corpus.** NQ splits Wikipedia
articles into overlapping chunks, so a clean document and its near-identical
twin get near-identical scores and thus near-identical p-values: the null
p-values are strongly, and not provably positively, dependent. Expect the
realised FDR to exceed the nominal ``alpha`` — the honest reading of a q-value
here is "a calibrated ranking with an approximately correct scale", not a
guarantee. ``method="by"`` applies the Benjamini-Yekutieli correction, which
divides the whole ladder by the harmonic number ``sum_{i<=m} 1/i`` and is valid
under *arbitrary* dependence. It is genuinely conservative: at ``m = 5000``
tests that factor is ~9.1, so BY at ``alpha = 0.1`` behaves roughly like BH at
``alpha = 0.011`` and will cost recall accordingly. Report both.

Expected failure modes
----------------------
* **Resolution floor.** An empirical p-value from ``n`` clean scores cannot go
  below ``1/(n + 1)``. BH can reject *nothing* at all unless
  ``m * c / (n + 1) <= alpha``, so a 5,000-test scan at ``alpha = 0.1`` needs
  ~50,000 clean reference scores before a single rejection is arithmetically
  possible. :meth:`FDRControlDetector.scan_corpus` reports this as
  ``resolvable`` and logs a warning; ``tail="exponential"`` trades an
  assumption for resolution (see below).
* **Garbage in, calibrated garbage out.** FDR control inherits the wrapped
  detector's ranking. If ``passage-agreement`` cannot separate injected clusters
  from genuine near-duplicate chunks, no multiple-testing correction will — BH
  will simply refuse to reject, which is the correct behaviour and still a
  recall of zero.
* **Contaminated reference.** The clean distribution is *believed* clean. If the
  reference slice contains poison, the right tail is inflated, p-values are too
  large, and the procedure silently loses power.
* **In-sample optimism.** Fitting the wrapped detector and estimating its clean
  distribution on the same passages makes that distribution too tight and the
  p-values anti-conservative. ``split_calibration=True`` (default) uses disjoint
  halves.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, get_detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["FDRControlDetector", "adjusted_pvalues", "benjamini_hochberg"]

#: Supported multiple-testing corrections.
METHODS: tuple[str, ...] = ("bh", "by")


def _dependency_factor(n: int, method: str) -> float:
    """BH uses 1; BY divides the ladder by the harmonic number ``H_n``."""
    if method == "by":
        return float(np.sum(1.0 / np.arange(1, n + 1)))
    if method != "bh":
        raise ValueError(f"method must be one of {list(METHODS)}, got {method!r}")
    return 1.0


def adjusted_pvalues(pvalues: Any, method: str = "bh") -> np.ndarray:
    """Step-up adjusted p-values (q-values) for a family of tests.

    ``q_(i) = min_{j >= i} ( m * c * p_(j) / j )``, clipped to 1, where ``c`` is
    1 for BH and the harmonic number for BY. The running minimum from the top
    enforces monotonicity, so a document can never carry a smaller q-value than
    a document that scored more suspicious than it.

    Args:
        pvalues: 1-D array of p-values in ``(0, 1]``.
        method: ``"bh"`` or ``"by"``.

    Returns:
        Array of q-values, aligned to the input order.
    """
    p = np.asarray(pvalues, dtype=np.float64).reshape(-1)
    m = p.size
    if m == 0:
        return p
    factor = _dependency_factor(m, method)
    order = np.argsort(p, kind="mergesort")
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adj = p[order] * m * factor / ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m, dtype=np.float64)
    out[order] = np.clip(adj, 0.0, 1.0)
    return out


def benjamini_hochberg(pvalues: Any, alpha: float = 0.1, method: str = "bh") -> np.ndarray:
    """Boolean rejection mask at level ``alpha``.

    Equivalent to the classical step-up procedure: ``q <= alpha`` rejects exactly
    the same set as finding the largest ``i`` with ``p_(i) <= alpha * i / (m*c)``
    and rejecting the ``i`` smallest p-values.
    """
    return adjusted_pvalues(pvalues, method) <= float(alpha)


@register_detector("fdr-control", category="rag-defense")
class FDRControlDetector(Detector):
    """Wrap a detector and report calibrated p-values / q-values instead of scores.

    ``samples`` is ``(query, passages)``, as for every ``rag-defense`` detector.
    :meth:`score` returns ``-log10(p)`` so that higher is still more suspicious
    and the numbers are readable (3.0 means "one clean passage in a thousand
    scores this high"). :meth:`flag` and :meth:`qvalues` are the decision-making
    entry points; :meth:`scan_corpus` is the corpus-scale one.

    Args:
        base: Registry key of the detector to wrap, or an already-constructed
            :class:`Detector` instance.
        alpha: Default FDR level for :meth:`flag` / :meth:`scan_corpus`.
        method: ``"bh"`` (default) or ``"by"`` — see the module docstring.
        tail: ``"empirical"`` (default) uses only the observed clean scores, so
            p-values are floored at ``1/(n+1)``. ``"exponential"`` fits a
            peaks-over-threshold exponential tail (a generalised Pareto with
            shape 0) above the ``tail_quantile`` of the clean scores and
            extrapolates beyond the clean maximum. That buys the resolution BH
            needs at small ``alpha``, at the cost of an assumption: if the true
            clean tail is heavier than exponential — which it is whenever
            genuine near-duplicates saturate the top of the distribution — the
            extrapolated p-values are *too small* and the realised FDR exceeds
            ``alpha``. Off by default for that reason. Extrapolated p-values are
            floored at ``1/(n+1)^2`` so a badly fitted tail cannot claim
            unbounded significance for a score far outside the fit range.
        tail_quantile: Threshold quantile for the exponential tail fit.
        split_calibration: Fit the wrapped detector on one half of ``clean_data``
            and estimate its null distribution on the other half.
        store, encoder, model, tokenizer: Forwarded to the wrapped detector.
        base_config: Extra keyword arguments for the wrapped detector.
    """

    name = "fdr-control"
    category = "rag-defense"
    modality = "rag"
    requires_fit = True  # a p-value without a null distribution is not a p-value

    def __init__(
        self,
        *,
        base: str | Detector = "hybrid",
        alpha: float = 0.1,
        method: str = "bh",
        tail: str = "empirical",
        tail_quantile: float = 0.95,
        split_calibration: bool = True,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        base_config: dict[str, Any] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(threshold=threshold, base=getattr(base, "name", base), alpha=alpha,
                         method=method, tail=tail, split_calibration=split_calibration,
                         **config)
        _dependency_factor(1, method)  # validate early
        if tail not in {"empirical", "exponential"}:
            raise ValueError(f"tail must be 'empirical' or 'exponential', got {tail!r}")
        self.alpha = float(alpha)
        self.method = method
        self.tail = tail
        self.tail_quantile = float(tail_quantile)
        self.split_calibration = bool(split_calibration)
        self.base: Detector = (
            base if isinstance(base, Detector)
            else get_detector(base, store=store, encoder=encoder, model=model,
                              tokenizer=tokenizer, **(base_config or {}))
        )
        self._clean = np.empty(0, dtype=np.float64)  # sorted ascending
        self._tail_u = float("inf")
        self._tail_pu = 1.0
        self._tail_beta = 0.0

    # -- calibration ------------------------------------------------------- #
    def fit(self, clean_data: Any, calibration_data: Any = None, **kwargs: Any) -> FDRControlDetector:
        """Fit the wrapped detector and record its clean score distribution.

        Args:
            clean_data: Sequence of ``(query, passages)`` pairs from a corpus
                believed clean.
            calibration_data: Optional disjoint sequence used *only* to estimate
                the null distribution. When omitted and ``split_calibration`` is
                on, ``clean_data`` is split in half.
            **kwargs: Forwarded to the wrapped detector's ``fit``.

        Returns:
            ``self``.

        Raises:
            ValueError: If no clean data is supplied — the whole method is a
                statement about a null distribution, so failing loudly beats
                inventing one.
        """
        data = list(clean_data or ())
        if not data:
            raise ValueError(
                "fdr-control needs clean reference data: p-values are defined "
                "against a null distribution and there is no sane default for one"
            )
        calib = list(calibration_data) if calibration_data is not None else None
        if calib is None and self.split_calibration and len(data) >= 4:
            half = len(data) // 2
            data, calib = data[:half], data[half:]
        elif calib is None:
            calib = data  # in-sample; anti-conservative, see the module docstring

        self.base.fit(data, **kwargs)
        scores = np.concatenate([
            np.asarray(self.base.score(s), dtype=np.float64).reshape(-1) for s in calib
        ])
        self._clean = np.sort(scores[np.isfinite(scores)])
        if self._clean.size < 100:
            logger.warning(
                "fdr-control calibrated on only %d clean scores; the smallest "
                "attainable p-value is %.3g", self._clean.size, 1.0 / (1 + self._clean.size),
            )
        self._fit_tail()
        self._fitted = True
        return self

    def _fit_tail(self) -> None:
        """Peaks-over-threshold exponential fit to the clean upper tail."""
        ref = self._clean
        if self.tail != "exponential" or ref.size < 50:
            return
        u = float(np.quantile(ref, self.tail_quantile))
        excess = ref[ref > u] - u
        if excess.size < 10 or not np.isfinite(u):
            return
        beta = float(excess.mean())
        if beta <= 0:
            return
        self._tail_u, self._tail_beta = u, beta
        self._tail_pu = float((1.0 + (ref.size - np.searchsorted(ref, u, "left"))) / (1.0 + ref.size))

    # -- p-values ----------------------------------------------------------- #
    def pvalues(self, samples: Any) -> np.ndarray:
        """Empirical right-tail p-value of each passage under the clean null.

        ``p = (1 + #{clean >= s}) / (1 + n)``. The add-one keeps ``p`` strictly
        positive, so ``log p`` stays finite for a score beyond every clean
        observation — the case that matters most.
        """
        self._check_fitted()
        raw = np.asarray(self.base.score(samples), dtype=np.float64).reshape(-1)
        ref = self._clean
        n = ref.size
        ge = n - np.searchsorted(ref, raw, side="left")
        p = (1.0 + ge) / (1.0 + n)
        if np.isfinite(self._tail_u) and self._tail_beta > 0:
            far = raw > self._tail_u
            if far.any():
                # Floored at the square of the empirical resolution: a tail fitted
                # to a bounded or heavy-tailed clean distribution will otherwise
                # extrapolate a score far beyond the fit range to an absurd
                # p-value, and BH would reject on that arithmetic alone.
                floor = (1.0 / (1.0 + n)) ** 2
                p[far] = np.maximum(
                    self._tail_pu * np.exp(-(raw[far] - self._tail_u) / self._tail_beta), floor
                )
        return np.clip(p, 1e-300, 1.0)

    def score(self, samples: Any) -> np.ndarray:
        """``-log10(p)``: same ranking as the wrapped detector, readable scale."""
        return -np.log10(self.pvalues(samples))

    def qvalues(self, samples: Any, method: str | None = None) -> np.ndarray:
        """BH- (or BY-) adjusted p-values for one retrieved set.

        A q-value is the "how much do I trust this alarm" number a triage report
        should carry. Over a single ``k = 5`` set it is nearly vacuous — use
        :meth:`scan_corpus` for a real family.
        """
        return adjusted_pvalues(self.pvalues(samples), method or self.method)

    def flag(self, samples: Any, alpha: float | None = None,
             method: str | None = None) -> np.ndarray:
        """Boolean rejection mask at FDR level ``alpha`` for one retrieved set."""
        a = self.alpha if alpha is None else float(alpha)
        return self.qvalues(samples, method) <= a

    # -- corpus scale -------------------------------------------------------- #
    def scan_corpus(
        self,
        store: Any,
        queries: Sequence[str],
        alpha: float | None = None,
        *,
        k: int = 5,
        method: str | None = None,
        pool: str = "min-sidak",
    ) -> dict[str, Any]:
        """Score many queries' top-k and apply BH **once** over the pooled family.

        Args:
            store: A :class:`~aiforensics.data.rag.RagStore`.
            queries: Query strings to retrieve with. Coverage matters: a
                document that no query retrieves is never tested and can never
                be flagged, so recall is bounded by retrieval.
            alpha: FDR level; defaults to ``self.alpha``.
            k: Passages per query.
            method: ``"bh"`` or ``"by"``; defaults to ``self.method``.
            pool: How to reduce several tests of the same document to one.
                ``"min-sidak"`` (default) takes the smallest p-value and applies
                a Sidak correction ``1 - (1 - p)^h`` for the ``h`` queries that
                retrieved it — taking a minimum over ``h`` tests is itself
                multiple testing, and skipping the correction makes frequently
                retrieved documents look significant for free. ``"min"`` skips
                it (anti-conservative). ``"none"`` keeps every (query, document)
                pair as its own test, which inflates the family with duplicates.

        Returns:
            A triage report: ``doc_ids``, ``pvalue``, ``qvalue``, ``flagged``,
            plus ``n_tests``, ``n_flagged``, ``alpha``, ``method`` and
            ``resolvable`` — the last being ``False`` when the empirical p-value
            floor makes *any* rejection arithmetically impossible.
        """
        self._check_fitted()
        a = self.alpha if alpha is None else float(alpha)
        meth = method or self.method
        qs = [queries] if isinstance(queries, str) else list(queries)
        hits = store.search(qs, k=k) if qs else []

        keys: list[str] = []
        pmin: dict[str, float] = {}
        count: dict[str, int] = {}
        pairs: list[float] = []
        for query, row in zip(qs, hits, strict=False):
            if not row:
                continue
            p = self.pvalues((query, [h.text for h in row]))
            for hit, pv in zip(row, p, strict=False):
                if pool == "none":
                    keys.append(hit.doc_id)
                    pairs.append(float(pv))
                    continue
                if hit.doc_id not in pmin:
                    keys.append(hit.doc_id)
                    pmin[hit.doc_id] = float(pv)
                    count[hit.doc_id] = 1
                else:
                    pmin[hit.doc_id] = min(pmin[hit.doc_id], float(pv))
                    count[hit.doc_id] += 1

        if pool == "none":
            pooled = np.asarray(pairs, dtype=np.float64)
        else:
            pooled = np.asarray([pmin[d] for d in keys], dtype=np.float64)
            if pool == "min-sidak":
                h = np.asarray([count[d] for d in keys], dtype=np.float64)
                pooled = 1.0 - np.power(1.0 - pooled, h)
            elif pool != "min":
                raise ValueError(f"pool must be 'min-sidak', 'min' or 'none', got {pool!r}")

        q = adjusted_pvalues(pooled, meth)
        flagged = q <= a
        m = pooled.size
        floor = _dependency_factor(max(m, 1), meth) * m / (1.0 + self._clean.size)
        resolvable = bool(m and (self.tail == "exponential" or floor <= a))
        if m and not resolvable:
            logger.warning(
                "fdr-control: %d tests against %d clean scores puts the smallest "
                "attainable q-value at %.3g > alpha=%.3g, so BH cannot reject anything; "
                "calibrate on more clean data or set tail='exponential'",
                m, self._clean.size, floor, a,
            )
        return {
            "doc_ids": keys,
            "pvalue": pooled,
            "qvalue": q,
            "flagged": flagged,
            "n_tests": int(m),
            "n_flagged": int(flagged.sum()),
            "alpha": a,
            "method": meth,
            "pool": pool,
            "resolvable": resolvable,
        }

    def describe(self) -> dict:
        """Wrapper identity plus the wrapped detector's own description."""
        d = super().describe()
        d["base"] = self.base.describe()
        d["n_clean_scores"] = int(self._clean.size)
        d["min_pvalue"] = 1.0 / (1.0 + self._clean.size)
        d["tail"] = self.tail
        return d
