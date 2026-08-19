"""k-of-n consensus voting — cut false alarms by demanding agreement.

The hypothesis
--------------
At corpus scale the problem is not ranking, it is precision. ``passage-agreement``
reaches ROC-AUC 0.979 over 20,123 documents at a 0.497 % base rate, yet only
PR-AUC 0.164: the top of the *clean* score distribution is saturated with genuine
near-duplicates, because NQ chops Wikipedia articles into overlapping chunks and
many honest passages have a near-identical twin at cosine 1.0000.

Those false alarms are *detector-specific*. A duplicated encyclopedia chunk looks
suspicious to the agreement detector and completely ordinary to the query-
similarity and perplexity detectors. PoisonedRAG poison, by contrast, is built to
trip several signals at once: it is mutually near-duplicate (agreement), it
carries the verbatim question as a prefix (query-similarity), and in its
white-box HotFlip form it is disfluent (perplexity). So requiring **k of n**
detectors to fire should remove false alarms much faster than it removes true
ones. This module makes that claim measurable rather than asserting it.

The arithmetic, and why the measured number will be worse
---------------------------------------------------------
Calibrate every member at its own clean alarm budget ``f`` (``member_fpr``,
default 0.05) so the votes are comparable. If the members were *independent* on
clean data, the k-of-n clean alarm rate would be the binomial tail

    P(votes >= k) = sum_{j>=k} C(n, j) f^j (1 - f)^(n - j)

For ``n = 3``, ``f = 0.05``: k=1 gives 0.1426, k=2 gives 0.00725, k=3 gives
0.000125. Going from "any detector fires" to "two must agree" is therefore a
~20x predicted reduction in false alarms, and against a single member at its own
budget (0.05) k=2 predicts ~6.9x. Recall, meanwhile, only drops to the extent
that poison fails to trip two signals.

Real members are **correlated** — they all read the same embedding space, they
are all evaluated on the same retrieved sets, and a near-duplicate pair raises
both the agreement score and (often) the query-similarity score. Positive
correlation makes the joint tail heavier than the binomial, so the *measured*
reduction will be smaller than the prediction above. That gap is itself the
result worth reporting, so :meth:`fit` computes both numbers on the clean
calibration data and :meth:`describe` returns them side by side as
``clean_alarm_rate`` (measured) vs ``predicted_alarm_rate`` (binomial); their
ratio is a direct, label-free estimate of how redundant the member panel is.

Known failure modes (stated up front, because they are results)
---------------------------------------------------------------
* **Voting cannot rescue a blind member.** ``perplexity`` scores 0.000 detection
  on fluent black-box poison. In a 3-member panel it then contributes a vote only
  by accident, so k=3 is effectively unreachable and k=2 degenerates to "both of
  the other two must agree". Consensus raises precision by *sacrificing* the
  attacks that only one member can see — with N=1 injected passages,
  ``passage-agreement`` is blind and ``query-similarity`` is the only witness, so
  any k>=2 rule scores it 1 vote at most. **k=2 is not universally safe; sweep k.**
* **Correlated false alarms survive.** A passage that is both a near-duplicate
  *and* unusually close to the query — an FAQ entry, a verbatim restatement —
  trips two members for entirely innocent reasons and voting cannot see the
  difference.
* **Calibration transfers only as far as the clean set does.** Per-member
  thresholds are empirical quantiles of ``clean_data``. If the calibration corpus
  is less near-duplicate-heavy than the corpus under test, every member votes too
  often and the consensus rule inherits the miscalibration.
* **Coarse by construction.** The vote count takes n+1 values, so a pure vote
  score gives a step-function ROC with n+1 points, which understates AUC and
  makes threshold sweeps meaningless between integers. :meth:`score` therefore
  adds a sub-unit tie-break (see below); the ranking *within* a vote level is
  strictly secondary to the vote count, by design.

Composed detectors are cited in their own modules; the empirical-p-value and
Fisher-style calibration convention follows ``rag_hybrid`` in this package.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, get_detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["ConsensusDetector", "DEFAULT_MEMBERS", "binomial_tail"]

#: Members voting by default — chosen to read three different artifacts
#: (semantic redundancy, retrieval geometry, text fluency) so that their false
#: alarms are as close to independent as this testbed allows.
DEFAULT_MEMBERS: tuple[str, ...] = ("passage-agreement", "query-similarity", "perplexity")

#: Largest fractional part added to a vote count; strictly < 1 so the tie-break
#: can never promote a passage past the next integer vote level.
_TIE_SCALE = 0.999


def binomial_tail(n: int, k: int, f: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, f)`` — the independent-members model.

    Args:
        n: Number of voting members.
        k: Votes required to raise an alarm.
        f: Per-member clean alarm rate.

    Returns:
        The predicted k-of-n false alarm rate under independence.
    """
    if k <= 0:
        return 1.0
    return float(sum(math.comb(n, j) * f**j * (1.0 - f) ** (n - j) for j in range(k, n + 1)))


@register_detector("consensus", category="rag-defense")
class ConsensusDetector(Detector):
    """Flag a passage when k of n independently calibrated detectors agree.

    ``samples`` is ``(query, passages)``; :meth:`score` returns one float per
    passage, higher = more suspicious. The integer part is the **vote count**,
    so thresholding at ``k`` recovers exactly the k-of-n rule for any k.

    The fractional part is ``0.999 * (1 - mean member p-value)``, a strictly
    sub-unit tie-break that orders passages *inside* a vote level by how hard
    the panel leaned. Without it the score takes only n+1 values and the ROC
    curve degenerates to n+1 points, which both understates AUC and makes a
    top-m budget ("flag the 20 most suspicious documents") arbitrary among the
    hundreds of documents sharing a vote count. The tie-break never changes
    which passages clear a given integer k.

    Args:
        members: Registry keys to poll. A member that cannot be constructed
            (missing model, no ``store``) is skipped with a warning, so the
            panel degrades instead of the run dying — but ``n`` shrinks with it,
            and ``describe()["n_members"]`` records what actually voted.
        member_fpr: Per-member clean alarm budget used to derive each vote
            threshold. A single float, or a ``{key: float}`` mapping when one
            member deserves a tighter budget than another.
        k: Default votes-to-alarm, used only to seed ``self.threshold`` so
            :meth:`predict` works without an explicit cut. Report it as a chosen
            operating point on a sweep, never as a calibrated default.
        store, encoder, model, tokenizer: Forwarded verbatim to every member.
        threshold: Overrides the ``k``-derived default.

    Raises:
        LookupError: If no member could be constructed.
    """

    name = "consensus"
    category = "rag-defense"
    modality = "rag"
    requires_fit = True  # votes are meaningless without per-member calibration

    def __init__(
        self,
        *,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        members: Sequence[str] = DEFAULT_MEMBERS,
        member_fpr: float | dict[str, float] = 0.05,
        k: int = 2,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=float(k) if threshold is None else threshold,
            members=list(members), member_fpr=member_fpr, k=int(k), **config,
        )
        self.k = int(k)
        self.member_fpr = member_fpr

        self.members: dict[str, Detector] = {}
        for key in members:
            try:
                self.members[key] = get_detector(
                    key, store=store, encoder=encoder, model=model, tokenizer=tokenizer,
                )
            except Exception as exc:  # optional dependency / missing store
                logger.warning("consensus: skipping member %r (%s)", key, exc)
        if not self.members:
            raise LookupError(
                f"none of the requested members {list(members)} could be constructed; "
                f"check that their dependencies are installed and that store= was passed"
            )

        self._thresholds: dict[str, float] = {}
        self._clean: dict[str, np.ndarray] = {}
        self._clean_rate: dict[str, float] = {}
        #: Measured vs binomial-predicted clean alarm rate, per k. Filled by fit.
        self.clean_alarm_rate_: dict[int, float] = {}
        self.predicted_alarm_rate_: dict[int, float] = {}

    # -- calibration ------------------------------------------------------- #
    def _budget(self, key: str) -> float:
        """Per-member clean alarm budget for ``key``."""
        if isinstance(self.member_fpr, dict):
            return float(self.member_fpr.get(key, 0.05))
        return float(self.member_fpr)

    def fit(self, clean_data: Any, **kwargs: Any) -> ConsensusDetector:
        """Calibrate every member to its own clean alarm budget.

        Each member is scored over the whole clean set; its vote threshold is the
        empirical ``1 - member_fpr`` quantile of that distribution. Ties at the
        quantile can only push the realised rate *above* the budget, so the
        achieved rate is recorded per member in :meth:`describe` rather than
        assumed.

        Because the per-passage scores stay aligned across members, the clean
        vote matrix is available here for free: the measured k-of-n clean alarm
        rate is stored alongside the binomial prediction, and their ratio
        quantifies member correlation without using a single label.

        Args:
            clean_data: Sequence of ``(query, passages)`` pairs believed clean.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``clean_data`` yields no scores to calibrate on.
        """
        samples = list(clean_data)
        per_member: dict[str, np.ndarray] = {}
        for key, det in self.members.items():
            parts = [np.asarray(det.score(s), dtype=np.float64).reshape(-1) for s in samples]
            per_member[key] = np.concatenate(parts) if parts else np.zeros(0)
        sizes = {v.size for v in per_member.values()}
        if not sizes or sizes == {0}:
            raise ValueError("consensus.fit needs a non-empty clean_data to calibrate votes")
        if len(sizes) != 1:
            raise ValueError(
                f"members returned different score counts on clean_data ({sizes}); "
                f"votes cannot be aligned per passage"
            )

        votes = np.zeros(next(iter(sizes)), dtype=np.int64)
        for key, scores in per_member.items():
            ref = np.sort(scores)
            self._clean[key] = ref
            thr = _upper_quantile_threshold(ref, self._budget(key))
            self._thresholds[key] = thr
            fired = scores >= thr
            self._clean_rate[key] = float(fired.mean())
            votes += fired.astype(np.int64)

        n = len(self.members)
        mean_f = float(np.mean([self._clean_rate[k] for k in self.members])) if n else 0.0
        for kk in range(1, n + 1):
            self.clean_alarm_rate_[kk] = float((votes >= kk).mean())
            self.predicted_alarm_rate_[kk] = binomial_tail(n, kk, mean_f)
        self._fitted = True
        return self

    # -- scoring ----------------------------------------------------------- #
    def _pvalue(self, key: str, scores: np.ndarray) -> np.ndarray:
        """Empirical right-tail p-value against the member's clean scores.

        ``(1 + #{clean >= s}) / (1 + n)``; the add-one keeps it strictly positive
        for a score beyond every clean observation, which is the case that
        matters most.
        """
        ref = self._clean.get(key)
        if ref is None or ref.size == 0:
            return np.ones_like(scores)
        ge = ref.size - np.searchsorted(ref, scores, side="left")
        return (1.0 + ge) / (1.0 + ref.size)

    def _member_scores(self, samples: Any) -> dict[str, np.ndarray]:
        """Raw per-member scores for one ``(query, passages)`` sample."""
        return {
            key: np.asarray(det.score(samples), dtype=np.float64).reshape(-1)
            for key, det in self.members.items()
        }

    def vote_matrix(self, samples: Any) -> np.ndarray:
        """Per-member boolean votes, for diagnosis.

        Args:
            samples: ``(query, passages)``.

        Returns:
            ``(len(passages), n_members)`` bool array; column order is
            :attr:`member_keys`. Column sums say which member is carrying the
            panel, row sums are the vote counts behind :meth:`score`.

        Raises:
            RuntimeError: If :meth:`fit` has not been called — there are no
                thresholds to vote against.
        """
        self._check_fitted()
        raw = self._member_scores(samples)
        return np.column_stack([raw[key] >= self._thresholds[key] for key in self.members])

    def score(self, samples: Any) -> np.ndarray:
        """Number of members voting "suspicious", plus a sub-unit tie-break.

        Args:
            samples: ``(query, passages)``.

        Returns:
            ``(len(passages),)`` float array. ``floor(score)`` is the vote count,
            so ``score >= k`` is exactly the k-of-n rule; the remainder ranks
            within a vote level by ``1 - mean member p-value``.
        """
        self._check_fitted()
        raw = self._member_scores(samples)
        keys = list(self.members)
        votes = np.sum(
            [(raw[key] >= self._thresholds[key]).astype(np.float64) for key in keys], axis=0
        )
        pvals = np.mean([self._pvalue(key, raw[key]) for key in keys], axis=0)
        return votes + _TIE_SCALE * (1.0 - np.clip(pvals, 0.0, 1.0))

    @property
    def member_keys(self) -> list[str]:
        """Voting members, in the column order used by :meth:`vote_matrix`."""
        return list(self.members)

    def describe(self) -> dict:
        """Panel identity, per-member operating points and the correlation gap."""
        base = super().describe()
        base["n_members"] = len(self.members)
        base["k"] = self.k
        base["members"] = {
            key: {
                "budget": self._budget(key),
                "vote_threshold": self._thresholds.get(key),
                "clean_alarm_rate": self._clean_rate.get(key),
            }
            for key in self.members
        }
        base["clean_alarm_rate"] = dict(self.clean_alarm_rate_)
        base["predicted_alarm_rate"] = dict(self.predicted_alarm_rate_)
        base["correlation_inflation"] = {
            kk: (self.clean_alarm_rate_[kk] / self.predicted_alarm_rate_[kk]
                 if self.predicted_alarm_rate_.get(kk) else None)
            for kk in self.clean_alarm_rate_
        }
        return base


def _upper_quantile_threshold(sorted_scores: np.ndarray, budget: float) -> float:
    """Smallest cut whose clean alarm rate is closest to ``budget`` from below.

    Args:
        sorted_scores: Clean scores, ascending.
        budget: Target clean alarm rate in ``[0, 1]``.

    Returns:
        A threshold ``t`` such that ``#{clean >= t} == floor(budget * n)`` when
        the scores are distinct. With ties the realised rate can only exceed the
        budget, never undershoot it — a constant-scoring member (e.g. the
        agreement detector on single-passage sets) then votes for everything or
        nothing, which is why the achieved rate is reported, not the target.
        A budget too small to admit one clean score gives a threshold just above
        the clean maximum, so only genuinely unprecedented scores vote.
    """
    n = sorted_scores.size
    n_alarm = math.floor(budget * n)
    if n_alarm <= 0:
        return float(np.nextafter(sorted_scores[-1], np.inf))
    return float(sorted_scores[n - n_alarm])
