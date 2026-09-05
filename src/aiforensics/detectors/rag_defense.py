"""RAG poisoning defences — post-retrieval detection of injected passages.

The techniques here score a *retrieved set*: given a query and the top-k
passages, decide which of them were adversarially injected. That is the unit the
published defences operate on, and it is the unit our headline metric is
computed over for RAG (see :mod:`aiforensics.metrics.detection`).

Implemented:

* :class:`RAGDefenderDetector` — wraps the reference implementation of

      Minseok Kim, Hankook Lee, Hyungjoon Koo. "Rescuing the Unpoisoned:
      Efficient Defense Against Knowledge Corruption Attacks on RAG Systems."
      ACSAC 2025, 1178-1192. DOI 10.1109/ACSAC67867.2025.00093
      arXiv:2511.01268 · https://github.com/SecAI-Lab/RAGDefender

  A two-stage post-retrieval filter: group the retrieved passages (clustering
  for single-hop, concentration for multi-hop), then identify the adversarial
  group. It needs no retraining and no extra LLM call.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["RAGDefenderDetector", "RAGDefenderFrequencyDetector"]


def _require_ragdefender() -> Any:
    try:
        import ragdefender
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "ragdefender is not installed. It is pinned in docker/requirements.txt; "
            "run `make build` to rebuild the image, or `pip install ragdefender`."
        ) from exc
    return ragdefender


@register_detector("ragdefender")
class RAGDefenderDetector(Detector):
    """Post-retrieval adversarial-passage filter (ACSAC'25).

    Unlike a scoring detector, the reference implementation returns a *decision*
    — the indices it removes — rather than a per-passage suspicion score. The
    :class:`~aiforensics.detectors.base.Detector` contract is score-based, so
    :meth:`score` reports a binary 1.0/0.0 and :attr:`threshold` is fixed at 0.5.
    A consequence worth stating in any write-up: ROC/PR curves over this
    detector are degenerate (two operating points), so report the confusion
    matrix and detection rate rather than an AUC.

    ``samples`` is ``(query, passages)``: the query string and the retrieved
    passage texts, in rank order. Scores align to ``passages``.

    **A continuous variant is available, and it exists to fix a measured
    problem.** In decision mode the detector reaches 124/131 = 94.7% detection
    across PoisonedRAG, GARAG and jamming, but flags 176/469 = 37.5% of clean
    passages, and its fixed 0.5 threshold cannot be calibrated, so it is not on
    equal terms with the detectors held to a 5% clean budget.

    The cause is structural rather than a bad threshold. Stage 2 already ranks
    passages by a continuous frequency score ``f_i`` (paper Eq. 6) and then
    keeps the top ``N_adv`` of them, where ``N_adv`` comes from Stage 1. Every
    passage inside that cut is flagged **regardless of how low its own f_i is**,
    so whenever Stage 1 over-estimates ``N_adv`` the surplus is drawn from clean
    passages and there is no way to decline them. ``scoring="frequency"``
    returns ``f_i`` itself, which restores a calibratable operating point and
    makes ROC/PR meaningful.

    ``f_i`` is a per-query quantity whose scale depends on ``N_adv`` (it sums
    over ``N_pairs = max(1, C(N_adv, 2))`` pairs), so cross-query thresholding
    needs it put on a common scale. ``normalise="zscore"`` standardises within
    each retrieved set, which is the default for the scoring mode.

    Args:
        task_type: ``"single_hop"`` (clustering-based grouping, NQ/MS MARCO),
            ``"multi_hop"`` (concentration-based, HotpotQA) or ``"auto"``.
        scoring: ``"decision"`` reproduces the paper exactly (binary, fixed
            threshold 0.5). ``"frequency"`` exposes Stage 2's own ``f_i``.
        normalise: For ``scoring="frequency"`` only -- ``"zscore"`` or ``"none"``.
        embedder: Embedding model key understood by ragdefender.
        gpu_id: Device ordinal *inside* the container, where exactly one GPU is
            visible — so this is always 0 and must not be used to pick a
            physical GPU (see ``AIF_GPU_ID`` in ``.env``).
        p: Reference hyperparameter, passed through.
        m: Reference hyperparameter, passed through.
    """

    name = "ragdefender"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False

    def __init__(
        self,
        *,
        task_type: str = "single_hop",
        embedder: str = "minilm-all",
        gpu_id: int = 0,
        p: int = 2,
        m: int = 5,
        scoring: str = "decision",
        normalise: str = "zscore",
        threshold: float | None = 0.5,
        **config: Any,
    ) -> None:
        if scoring not in {"decision", "frequency"}:
            raise ValueError(
                f"scoring must be 'decision' or 'frequency', got {scoring!r}"
            )
        if normalise not in {"zscore", "none"}:
            raise ValueError(f"normalise must be 'zscore' or 'none', got {normalise!r}")
        # A continuous score has no meaningful fixed threshold; it must be
        # calibrated on clean data like every other scoring detector here.
        if scoring == "frequency":
            threshold = None
        super().__init__(
            threshold=threshold, task_type=task_type, embedder=embedder,
            gpu_id=gpu_id, p=p, m=m, scoring=scoring, normalise=normalise, **config,
        )
        rd = _require_ragdefender()
        self.task_type = task_type
        self.scoring, self.normalise = scoring, normalise
        self._defender = rd.RAGDefender(
            embedder=embedder, task_type=task_type, gpu_id=gpu_id, p=p, m=m,
        )

    def removed_indices(self, query: str, passages: Sequence[str]) -> list[int]:
        """Indices of passages the defence removes, for one query."""
        if not passages:
            return []
        _, removed = self._defender.defend(query, list(passages), return_indices=True)
        return sorted(int(i) for i in removed)

    def keep_indices(self, query: str, passages: Sequence[str]) -> list[int]:
        """Indices the defence keeps — the hook for ``RagPipeline(filter_fn=…)``."""
        drop = set(self.removed_indices(query, passages))
        return [i for i in range(len(passages)) if i not in drop]

    def frequency_scores(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Stage 2's frequency score ``f_i`` per passage, before the top-N cut.

        Reproduces the pipeline's own two stages -- Stage 1 estimates ``N_adv``,
        Stage 2 builds ``f_i`` over the top ``max(1, C(N_adv, 2))`` most similar
        pairs -- and returns ``f_i`` rather than the truncated index set.
        """
        from ragdefender.similarity import n_pairs_for, top_similar_pairs

        texts = list(passages)
        if not texts:
            return np.zeros(0, dtype=np.float64)
        d = self._defender
        emb = d.embedder.encode(texts, convert_to_tensor=True)
        if self.task_type == "single_hop":
            n_adv = d._clustering.estimate_n_adv(texts, embeddings=emb)
        else:
            n_adv = d._concentration.estimate_n_adv(texts, embeddings=emb)
        f = np.zeros(len(texts), dtype=np.float64)
        if n_adv <= 0:
            return f
        for (i, j), sim in top_similar_pairs(emb, n_pairs_for(int(n_adv))):
            w = (1.0 if sim >= 0.0 else -1.0) * abs(float(sim)) ** d._identify.p
            f[i] += w
            f[j] += w
        if self.normalise == "zscore":
            sd = float(f.std())
            # A retrieved set whose scores are all equal carries no ranking; a
            # 1e-12 floor would turn that into arbitrary enormous z-scores.
            f = (f - f.mean()) / sd if sd > 1e-9 else np.zeros_like(f)
        return f

    def score(self, samples: Any) -> np.ndarray:
        """Suspicion over one query's retrieved passages.

        Args:
            samples: ``(query, passages)``.

        Returns:
            With ``scoring="decision"`` (the paper), 1.0 where the passage is
            flagged as injected else 0.0. With ``scoring="frequency"``, Stage
            2's continuous ``f_i``, which can be thresholded on clean data.
        """
        query, passages = samples
        if self.scoring == "frequency":
            return self.frequency_scores(query, passages)
        scores = np.zeros(len(passages), dtype=np.float64)
        idx = self.removed_indices(query, passages)
        if idx:
            scores[idx] = 1.0
        return scores


@register_detector("ragdefender-freq")
class RAGDefenderFrequencyDetector(RAGDefenderDetector):
    """RAGDefender scored by its own ``f_i`` instead of its top-N cut.

    Same pipeline, same paper, one change: Stage 2's frequency score is
    returned rather than thresholded at ``N_adv``. Registered separately so
    both operating points can appear in one table without a runner needing to
    know about the keyword -- the paper's decision rule stays available under
    ``ragdefender`` and is what should be cited as the published method.
    """

    name = "ragdefender-freq"

    def __init__(self, **config: Any) -> None:
        config.setdefault("scoring", "frequency")
        super().__init__(**config)
