"""RAGDefender's own algorithm, exposed as a continuous score.

    Minseok Kim, Hankook Lee, Hyungjoon Koo. "Rescuing the Unpoisoned: Efficient
    Defense Against Knowledge Corruption Attacks on RAG Systems." ACSAC 2025,
    1178-1192. DOI 10.1109/ACSAC67867.2025.00093 · arXiv:2511.01268
    https://github.com/SecAI-Lab/RAGDefender

The published pipeline is a **top-n selection**, not a threshold:

1. ``ClusteringBasedGrouping.estimate_n_adv(R)`` estimates *how many* of the
   retrieved passages are adversarial;
2. ``IdentifyAdversarial.select(R, n_adv)`` returns *which* ``n_adv`` of them.

That shape is right for a defence — something must be dropped before generation —
but it is what makes the method hard to use as a *detector*. Because stage 2 is
asked for exactly ``n_adv`` passages it always returns that many, so on a clean
corpus it still removes passages (measured here: 43% of retrieved passages
flagged with no poison present), and there is no knob to trade recall against
false alarms.

Stage 2 must nevertheless *rank* passages internally in order to take a top-n.
This module recovers that ranking without touching the library's logic: call
``select`` for every ``n = 1..k`` and record the smallest ``n`` at which each
passage is chosen. A passage picked first is the most suspicious, one never
picked is the least, and normalising by ``k`` yields a score in ``(0, 1]``.

The result is RAGDefender's own similarity machinery and selection rule, made
thresholdable — so it can be calibrated to an alarm budget and gives an ROC
curve, which the published keep/drop interface cannot.

:meth:`estimate_n_adv` is also exposed directly: the library's estimate of how
many passages are poisoned is itself a corpus-level signal for the question this
project cares about, "is this database poisoned at all?".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["RAGDefenderScoreDetector"]


def _require() -> Any:
    try:
        import ragdefender
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "ragdefender is not installed. It is pinned in docker/requirements.txt; "
            "run `make build`, or `pip install ragdefender`."
        ) from exc
    return ragdefender


@register_detector("ragdefender-score", category="rag-defense")
class RAGDefenderScoreDetector(Detector):
    """RAGDefender's grouping + identification, turned into a per-passage score.

    Args:
        task_type: ``"single_hop"`` (clustering-based grouping — NQ, MS MARCO) or
            ``"multi_hop"`` (concentration-based — HotpotQA).
        embedder: Embedder key understood by ``ragdefender.load_embedder``.
        gpu_id: Device ordinal *inside* the container, where exactly one GPU is
            visible, so this is always 0. It does not select a physical GPU (see
            ``AIF_GPU_ID`` in ``.env``).
        m: ``ClusteringBasedGrouping`` hyperparameter, passed through.
        p: ``IdentifyAdversarial`` hyperparameter, passed through.
        threshold: Left ``None`` on purpose — the score is a normalised rank, so
            the caller calibrates it on clean data at a chosen alarm budget.
        store, encoder, model, tokenizer: Accepted for constructor uniformity.
    """

    name = "ragdefender-score"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False

    def __init__(
        self,
        *,
        task_type: str = "single_hop",
        embedder: str = "minilm-all",
        gpu_id: int = 0,
        m: int = 5,
        p: int = 2,
        threshold: float | None = None,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        **config: Any,
    ) -> None:
        super().__init__(threshold=threshold, task_type=task_type, embedder=embedder,
                         gpu_id=gpu_id, m=m, p=p, **config)
        rd = _require()
        self._emb = rd.load_embedder(embedder, gpu_id=gpu_id)
        self._grouping = (
            rd.ClusteringBasedGrouping(self._emb, m=m) if task_type == "single_hop"
            else rd.ConcentrationBasedGrouping(self._emb)
        )
        self._identify = rd.IdentifyAdversarial(self._emb, p=p)
        self.task_type = task_type

    # -- the library's own stages, exposed ---------------------------------- #
    def estimate_n_adv(self, passages: Sequence[str]) -> int:
        """RAGDefender's stage-1 estimate of how many passages are adversarial.

        Useful on its own: run over many queries, the distribution of this count
        on a clean corpus versus a suspect one is a corpus-level indicator that
        needs no per-passage ground truth.
        """
        if not passages:
            return 0
        try:
            return int(self._grouping.estimate_n_adv(list(passages)))
        except Exception:
            logger.debug("estimate_n_adv failed; treating as 0", exc_info=True)
            return 0

    def select(self, passages: Sequence[str], n_adv: int) -> list[int]:
        """RAGDefender's stage-2 selection of the ``n_adv`` most adversarial indices."""
        if not passages or n_adv <= 0:
            return []
        try:
            return [int(i) for i in self._identify.select(list(passages), int(n_adv))]
        except Exception:
            logger.debug("select(n_adv=%d) failed; returning nothing", n_adv, exc_info=True)
            return []

    # -- score ---------------------------------------------------------------- #
    def score(self, samples: Any) -> np.ndarray:
        """Suspicion score combining both library stages. Higher = more suspicious.

        Two pieces of information are needed, and using either alone fails:

        * ``select`` gives a **within-query rank**. Sweeping ``n = 1..k`` and
          recording the first ``n`` at which a passage is chosen recovers the
          internal ordering (the selection is nested, so this is well defined).
          On its own it is useless across queries: ``select`` always returns
          something, so every query yields the same multiset of ranks whether or
          not it is poisoned, and the pooled ROC sits at chance.
        * ``estimate_n_adv`` gives the **cross-query scale** — how many passages
          stage 1 believes are adversarial. It never returns 0, so it cannot
          abstain, but its magnitude does separate: measured here, roughly 1.8 on
          clean retrieved sets versus 3.6 on attacked ones.

        Combining them, a passage ranked ``r`` (1 = most suspicious) scores::

            score = max(0, n_adv + 1 - r) / k

        so a passage outside the estimated adversarial set scores 0, and one
        inside scores higher both for being ranked earlier and for sitting in a
        set the grouping stage thinks is large.

        Args:
            samples: ``(query, passages)``. The query is unused: both library
                stages compare passages with each other, not with the query.

        Returns:
            ``(len(passages),)`` float array in ``[0, 1]``.
        """
        _query, passages = samples
        texts = [p if isinstance(p, str) else getattr(p, "text", str(p)) for p in passages]
        k = len(texts)
        if k == 0:
            return np.zeros(0, dtype=np.float64)
        if k == 1:
            # Nothing to rank against; abstain rather than invent a rank.
            return np.zeros(1, dtype=np.float64)

        first_at = np.full(k, np.inf)
        for n in range(1, k + 1):
            for idx in self.select(texts, n):
                if 0 <= idx < k and not np.isfinite(first_at[idx]):
                    first_at[idx] = n
        ranks = np.where(np.isfinite(first_at), first_at, k + 1.0)
        n_adv = float(self.estimate_n_adv(texts))
        return (np.maximum(0.0, n_adv + 1.0 - ranks) / k).astype(np.float64)

    def describe(self) -> dict:
        """Detector identity plus the library hyperparameters in use."""
        base = super().describe()
        base["stages"] = {
            "grouping": type(self._grouping).__name__,
            "identify": type(self._identify).__name__,
        }
        return base
