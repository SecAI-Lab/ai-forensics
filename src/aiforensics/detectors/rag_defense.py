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

__all__ = ["RAGDefenderDetector"]


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

    Args:
        task_type: ``"single_hop"`` (clustering-based grouping, NQ/MS MARCO),
            ``"multi_hop"`` (concentration-based, HotpotQA) or ``"auto"``.
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
        threshold: float | None = 0.5,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, task_type=task_type, embedder=embedder,
            gpu_id=gpu_id, p=p, m=m, **config,
        )
        rd = _require_ragdefender()
        self.task_type = task_type
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

    def score(self, samples: Any) -> np.ndarray:
        """Binary suspicion over one query's retrieved passages.

        Args:
            samples: ``(query, passages)``.

        Returns:
            1.0 where the passage is flagged as injected, else 0.0.
        """
        query, passages = samples
        scores = np.zeros(len(passages), dtype=np.float64)
        idx = self.removed_indices(query, passages)
        if idx:
            scores[idx] = 1.0
        return scores
