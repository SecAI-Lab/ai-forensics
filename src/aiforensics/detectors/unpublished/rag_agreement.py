"""Cross-passage agreement scoring for a retrieved set — embeddings only, no LLM.

The signal
----------
PoisonedRAG (Zou, Geng, Wang, Jia. "PoisonedRAG: Knowledge Corruption Attacks to
Retrieval-Augmented Generation of Large Language Models." USENIX Security 2025)
injects ``N = 5`` passages per target question. They are generated at temperature
1 so they are lexically distinct — exact- or near-duplicate string filters miss
them — but every one of them is built as ``P = Q + "." + I`` where ``I`` asserts
the *same* attacker answer. Sharing a verbatim question prefix and a single
asserted fact makes the injected set a tight cluster in embedding space, while
the genuine passages a dense retriever returns for the same question cover the
topic from several angles and are mutually more diverse.

So: score each retrieved passage by how much the *rest of the retrieved set*
agrees with it. This is the agreement half of the grouping stage of RAGDefender
(Kim, Lee, Koo. "Rescuing the Unpoisoned: Efficient Defense Against Knowledge
Corruption Attacks on RAG Systems." ACSAC 2025, 1178-1192, arXiv:2511.01268),
isolated as a single continuous statistic so it yields an ROC curve rather than
a hard keep/drop decision. It costs one encoder pass over ``k`` short passages
plus a ``k x k`` matmul — cheap enough to sweep over a whole corpus, and it
never calls a generative model.

Known failure modes (stated up front, because they are results)
---------------------------------------------------------------
* **False positives on clean input.** Genuine passages retrieved for one query
  are about one topic, and several of them are often near-duplicate encyclopedia
  snippets (NQ splits long Wikipedia articles into overlapping chunks). Such a
  clean set is mutually similar for entirely innocent reasons and this detector
  cannot tell that apart from an injected cluster. Expect a non-trivial FPR on a
  clean corpus; that number is the point of running the clean arm.
* **Blind to a lone injected passage.** With ``N = 1``, or when only one of the
  five injected passages reaches the top-k, there is no cluster to see and the
  score collapses to whatever the genuine set happens to look like. The method
  is inherently a *multi-passage* attack detector.
* **Symmetric by construction.** Every member of a cluster gets a high score,
  including any genuine passage that happens to fall inside the injected
  cluster's radius.
* **Query-independent.** The query is accepted for interface conformance and
  deliberately ignored: this detector measures only passage-to-passage
  agreement. The complementary "too close to the query" signal belongs to
  ``retrieval-outlier``, and keeping the two apart keeps the ablation honest.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

#: Supported scoring modes, see :class:`PassageAgreementDetector`.
MODES: tuple[str, ...] = ("mean-sim", "max-sim", "cluster-size")

__all__ = ["MODES", "PassageAgreementDetector"]


def _text_of(passage: Any) -> str:
    """Return the passage body, accepting a plain string or a ``SearchResult``."""
    if isinstance(passage, str):
        return passage
    text = getattr(passage, "text", None)
    if text is None:
        raise TypeError(
            f"expected a passage string or an object with .text, got "
            f"{type(passage).__name__}"
        )
    return str(text)


def _cluster_sizes(sim: np.ndarray, threshold: float) -> np.ndarray:
    """Single-linkage cluster size for every row of a similarity matrix.

    Args:
        sim: Symmetric ``(n, n)`` cosine-similarity matrix.
        threshold: Two passages are linked when their similarity is at least
            this. Linkage is transitive, so a chain of pairwise-similar
            passages forms one cluster.

    Returns:
        ``(n,)`` int array; entry ``i`` is the size of ``i``'s cluster,
        counting ``i`` itself (a singleton scores 1).
    """
    n = sim.shape[0]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.triu_indices(n, k=1), strict=False):
        if sim[i, j] >= threshold:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj

    roots = np.array([find(i) for i in range(n)])
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    return counts[inverse]


@register_detector("passage-agreement", category="rag-defense")
class PassageAgreementDetector(Detector):
    """Rank retrieved passages by their semantic agreement with the retrieved set.

    ``samples`` is ``(query, passages)`` — the query string (ignored, see the
    module docstring) and the retrieved passage texts in rank order. Returns one
    score per passage, aligned to ``passages``, higher = more suspicious.

    Modes:
        ``"mean-sim"``: mean cosine similarity to the other ``k - 1`` passages.
            Smooth and low-variance, but diluted: with ``k = 20`` and 5 injected
            passages, three quarters of the average is genuine noise, so the
            separation shrinks as ``k`` grows.
        ``"max-sim"``: **default.** Similarity to the single nearest other
            passage. It fires as soon as *two* members of a tight cluster are
            retrieved, regardless of how diverse the rest of the set is or how
            large ``k`` is, which is exactly the PoisonedRAG geometry (five
            mutual near-duplicates dropped into an otherwise varied top-k). It
            stays continuous, so the ROC curve is not degenerate. Its cost is
            sensitivity to a single genuine duplicate pair — the dominant false
            positive on clean input.
        ``"cluster-size"``: size of the passage's single-linkage cluster at
            ``cluster_threshold``, divided by ``k``. Closest to RAGDefender's
            grouping stage and the most interpretable ("five passages agree"),
            but it takes only ``k`` distinct values, which makes the ROC curve
            coarse, and it depends on getting the linkage threshold right.

    Threshold:
        Left at ``None`` on purpose — sweep it. Although the ``mean-sim`` /
        ``max-sim`` scores are cosine similarities and therefore live on a
        principled ``[-1, 1]`` scale, the useful cut depends on the encoder and
        on how topically tight the corpus is, and ``cluster-size`` is scaled by
        ``k``. If a single hard operating point is needed, ``0.8`` for
        ``max-sim`` (the same near-duplicate radius as ``cluster_threshold``) is
        the sane starting point, but report it as a chosen operating point on a
        sweep, not as a calibrated default.

    Args:
        store: Optional :class:`~aiforensics.data.rag.RagStore`; only its
            ``.encoder`` is used, and only if ``encoder`` is not given. Ground
            truth on the store (``.injected``) is never read.
        encoder: A ``SentenceTransformer``. Takes precedence over ``store``.
        model: Unused; accepted so the caller can construct every RAG detector
            uniformly. This detector makes no LLM call.
        tokenizer: Unused, as ``model``.
        threshold: Decision threshold; ``None`` by default (see above).
        mode: One of :data:`MODES`.
        cluster_threshold: Cosine linkage radius for ``"cluster-size"``.
        cache_embeddings: Memoise passage embeddings by text, so a threshold
            sweep that re-scores the same retrieved sets encodes them once.
        **config: Recorded verbatim in :meth:`describe` for the run manifest.

    Raises:
        ValueError: If ``mode`` is not in :data:`MODES`.
    """

    name = "passage-agreement"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False

    def __init__(
        self,
        *,
        store: Any | None = None,
        encoder: Any | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        threshold: float | None = None,
        mode: str = "max-sim",
        cluster_threshold: float = 0.8,
        cache_embeddings: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, mode=mode, cluster_threshold=cluster_threshold,
            cache_embeddings=cache_embeddings, **config,
        )
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        self.cluster_threshold = float(cluster_threshold)
        self._store = store
        self._encoder = encoder
        self._cache: dict[str, np.ndarray] | None = {} if cache_embeddings else None

    @property
    def encoder(self) -> Any:
        """The sentence encoder, taken from ``encoder`` or lazily from ``store``.

        Raises:
            ValueError: If neither an encoder nor a store was supplied.
        """
        if self._encoder is None:
            if self._store is None:
                raise ValueError(
                    f"{type(self).__name__} needs an embedding model: pass "
                    "encoder=<SentenceTransformer> or store=<RagStore>"
                )
            self._encoder = self._store.encoder
        return self._encoder

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Embed passages as unit-norm rows, reusing cached vectors when enabled."""
        if self._cache is None:
            missing = list(texts)
        else:
            missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            emb = np.asarray(
                self.encoder.encode(missing, normalize_embeddings=True), dtype=np.float64
            ).reshape(len(missing), -1)
            if self._cache is None:
                return emb
            self._cache.update(zip(missing, emb, strict=False))
        return np.vstack([self._cache[t] for t in texts])  # type: ignore[index]

    def score(self, samples: Any) -> np.ndarray:
        """Agreement-based suspicion score for one query's retrieved passages.

        Args:
            samples: ``(query, passages)``. ``passages`` may be strings or
                ``SearchResult``-like objects exposing ``.text``.

        Returns:
            ``(len(passages),)`` float array, higher = more suspicious. A set of
            fewer than two passages scores all-zero: agreement is undefined when
            there is nothing to agree with, and a lone passage is not evidence.

        Raises:
            TypeError: If ``samples`` is not a ``(query, passages)`` pair.
        """
        try:
            _query, passages = samples
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{type(self).__name__}.score expects (query, passages), got "
                f"{type(samples).__name__}"
            ) from exc

        texts = [_text_of(p) for p in passages]
        n = len(texts)
        if n < 2:
            return np.zeros(n, dtype=np.float64)

        emb = self._embed(texts)
        sim = np.clip(emb @ emb.T, -1.0, 1.0)

        if self.mode == "mean-sim":
            return (sim.sum(axis=1) - np.diag(sim)) / (n - 1)
        if self.mode == "max-sim":
            off_diagonal = sim.copy()
            np.fill_diagonal(off_diagonal, -np.inf)
            return off_diagonal.max(axis=1)
        return _cluster_sizes(sim, self.cluster_threshold).astype(np.float64) / n
