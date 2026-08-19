"""Embedding-space isolation detection for injected RAG passages.

Premise
-------
A PoisonedRAG passage (Zou, Geng, Wang, Jia. "PoisonedRAG: Knowledge Corruption
Attacks to Retrieval-Augmented Generation of Large Language Models." USENIX
Security 2025) is written to match *one query*, not to belong to the corpus: it
is ``P = Q + "." + I`` where ``I`` is fresh LLM prose asserting the attacker
answer. Nothing in that construction ties ``P`` to the rest of the knowledge
base, so ``P`` tends to land in a thin region of the corpus embedding manifold —
close to the query, far from everything the corpus actually contains. Genuine
passages, by contrast, come from a corpus with dense topical neighbourhoods
(near-duplicate Wikipedia paragraphs, sibling sections, redundant coverage).

Two classical outlier statistics are offered, selected by ``mode``:

* ``"knn-distance"`` — mean cosine distance to the ``n_neighbors`` nearest
  corpus neighbours (Ramaswamy, Rastogi, Shim, SIGMOD 2000; Angiulli & Pizzuti,
  PKDD 2002), computed exactly against the store's flat FAISS index.
* ``"lof"`` — Local Outlier Factor (Breunig, Kriegel, Ng, Sander, SIGMOD 2000)
  via scikit-learn with ``novelty=False``, over the retrieved item plus a seeded
  background sample of the corpus.

The crux: same-retrieved-set exclusion
--------------------------------------
The attack injects ``N=5`` passages per target question and they are lexically
distinct but semantically near-identical, so they are each other's nearest
neighbours. Scored naively, a cluster of poison looks *perfectly well
connected* and isolation collapses to near zero — the attack would defeat this
detector by construction. Both modes therefore exclude, from every passage's
neighbourhood, the corpus rows occupied by **any passage in the current
retrieved set** (which includes the passage itself). Each candidate is judged
only against the corpus it claims to belong to, never against its siblings.

Expected failure modes (documented, not hidden)
-----------------------------------------------
* If the attacker injects many near-identical passages and only some of them are
  retrieved, the unretrieved siblings stay in the corpus, become the retrieved
  ones' neighbours, and isolation drops. Exclusion only covers what is in the
  current retrieved set; a large ``N`` with a small ``k`` erodes the signal.
* An attacker who pads ``I`` with corpus-style boilerplate, or who copies a real
  passage and edits the answer span, sits in a dense region and is invisible
  here. This method reads *placement*, not fluency or content.
* Genuinely rare-but-real passages (singleton topics, odd formatting) are
  isolated too, so the clean-corpus false-positive rate is not zero — which is
  precisely why the output is a score for a threshold sweep, not a decision.
* ``"lof"`` uses a background *sample*, so it is noisier than ``"knn-distance"``
  and its density estimate degrades when the sample is small relative to the
  corpus's topical diversity.

No default threshold is set: mean cosine distance and LOF ratios both shift with
the encoder and the corpus, so any fixed cut would be a fabricated result. The
caller sweeps the score (see :func:`aiforensics.metrics.threshold_sweep`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["KnnIsolationDetector"]

#: Mirrors :data:`aiforensics.seed.DEFAULT_SEED`; duplicated so importing this
#: module (which the registry does eagerly) does not drag in torch.
_DEFAULT_SEED = 1337

_MODES = ("knn-distance", "lof")


@register_detector("knn-isolation", category="rag-defense")
class KnnIsolationDetector(Detector):
    """Score retrieved passages by how isolated they are in corpus embedding space.

    ``samples`` is ``(query, passages)``; the query is unused (isolation is a
    corpus-relative property, not a query-relative one) and scores align
    positionally to ``passages``. Higher = more isolated = more suspicious.

    Args:
        store: :class:`~aiforensics.data.rag.RagStore` holding the corpus under
            investigation. Required — the statistic is defined against it.
            Only ``.index``, ``.doc_ids`` and ``.documents`` are read; ground
            truth (``.injected_ids``) is never touched.
        encoder: SentenceTransformer used for passages absent from the store.
            Defaults to ``store.encoder``.
        model: Unused; accepted so the shared detector factory can pass one.
        tokenizer: Unused; accepted for the same reason.
        threshold: Left ``None`` on purpose — see the module docstring.
        mode: ``"knn-distance"`` or ``"lof"``.
        n_neighbors: Neighbourhood size for both statistics.
        n_background: Corpus rows sampled as the LOF background set.
        seed: Seeds the background sample so runs are reproducible.
        exclude_siblings: Apply same-retrieved-set exclusion. Defaults to
            ``True``; set ``False`` only to measure the ablation, since it is
            the difference between a detector and a no-op against clustered
            injection.
    """

    name = "knn-isolation"
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
        mode: str = "knn-distance",
        n_neighbors: int = 20,
        n_background: int = 2000,
        seed: int = _DEFAULT_SEED,
        exclude_siblings: bool = True,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, mode=mode, n_neighbors=n_neighbors,
            n_background=n_background, seed=seed,
            exclude_siblings=exclude_siblings, **config,
        )
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        if store is None:
            raise ValueError(
                "knn-isolation scores a passage against a corpus; pass store=RagStore(...)"
            )
        self.store = store
        self.mode = mode
        self.n_neighbors = int(n_neighbors)
        self.n_background = int(n_background)
        self.seed = int(seed)
        self.exclude_siblings = bool(exclude_siblings)
        self._encoder = encoder
        self._rows_by_text: dict[str, list[int]] | None = None
        self._background: tuple[np.ndarray, np.ndarray] | None = None

    # -- corpus plumbing --------------------------------------------------- #
    @property
    def encoder(self) -> Any:
        """Encoder for passages not found verbatim in the store."""
        if self._encoder is None:
            self._encoder = self.store.encoder
        return self._encoder

    def _row_map(self) -> dict[str, list[int]]:
        """Map every corpus surface form to the index rows that carry it.

        Both ``Document.text`` and ``Document.embedding_text()`` are keyed,
        because a retrieved passage reaches us as the bare body while the index
        embedded ``"title text"``. Values are lists so exact-duplicate texts all
        get excluded, not just the first.
        """
        if self._rows_by_text is None:
            rows: dict[str, list[int]] = {}
            for row, doc_id in enumerate(self.store.doc_ids):
                doc = self.store.documents[doc_id]
                for key in {doc.text, doc.embedding_text()}:
                    rows.setdefault(key, []).append(row)
            self._rows_by_text = rows
        return self._rows_by_text

    def _vectors(self, rows: Sequence[int]) -> np.ndarray:
        """Reconstruct stored (already unit-norm) vectors for index rows."""
        index = self.store.index
        keys = np.asarray(list(rows), dtype=np.int64)
        try:
            vecs = index.reconstruct_batch(keys)
        except Exception:  # older faiss builds lack the batched call
            vecs = np.vstack([index.reconstruct(int(r)) for r in keys])
        return np.ascontiguousarray(np.asarray(vecs, dtype=np.float32))

    def _embed(self, passages: Sequence[str]) -> tuple[np.ndarray, list[list[int]]]:
        """Vectors for the retrieved passages, plus the rows each occupies.

        Passages already in the corpus reuse their stored vector (exact, free);
        anything else is encoded. The row lists drive sibling exclusion.
        """
        from ..data.rag import embed_texts  # local: keeps module import torch-free

        row_map = self._row_map()
        hits = [row_map.get(p, []) for p in passages]
        vecs = np.zeros((len(passages), int(self.store.dim)), dtype=np.float32)
        known = [i for i, h in enumerate(hits) if h]
        missing = [i for i, h in enumerate(hits) if not h]
        if known:
            vecs[known] = self._vectors([hits[i][0] for i in known])
        if missing:
            vecs[missing] = embed_texts(
                [passages[i] for i in missing], encoder=self.encoder, normalize=True
            )
        return vecs, hits

    def _background_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Seeded background sample of the corpus as ``(rows, vectors)``.

        Cached on the instance: sampled once, reused for every query. The sample
        is drawn from the corpus as-is, so it may contain a handful of injected
        documents — the investigator cannot know which, and at realistic
        poisoning rates the contamination is negligible.
        """
        if self._background is None:
            n = len(self.store)
            size = min(self.n_background, n)
            rng = np.random.default_rng(self.seed)
            rows = np.sort(rng.choice(n, size=size, replace=False))
            self._background = (rows, self._vectors(rows))
        return self._background

    # -- the two statistics ------------------------------------------------ #
    def _knn_distance(
        self, vecs: np.ndarray, excluded: Sequence[set[int]]
    ) -> np.ndarray:
        """Mean cosine distance to the nearest non-excluded corpus neighbours.

        ``excluded[i]`` is the row set masked out for passage ``i``; the FAISS
        search asks for enough hits that ``n_neighbors`` still survive after
        masking.
        """
        widest = max((len(e) for e in excluded), default=0)
        k = min(len(self.store), self.n_neighbors + widest + 1)
        sims, idx = self.store.index.search(np.ascontiguousarray(vecs), k)
        out = np.empty(len(vecs), dtype=np.float64)
        for i, (row_sims, row_idx) in enumerate(zip(sims, idx, strict=False)):
            drop = excluded[i]
            kept = [
                float(s)
                for s, j in zip(row_sims, row_idx, strict=False)
                if int(j) >= 0 and int(j) not in drop
            ][: self.n_neighbors]
            # No neighbour survives only in a degenerate corpus; call it maximally
            # isolated rather than silently scoring it as an inlier.
            out[i] = float(np.mean(1.0 - np.asarray(kept))) if kept else 1.0
        return out

    def _lof(self, vecs: np.ndarray, bg_excluded: set[int]) -> np.ndarray:
        """Local Outlier Factor against the background sample, sign-flipped.

        ``negative_outlier_factor_`` is about ``-1`` for an inlier and more
        negative the more isolated a point is, so the returned score is its
        negation: ``~1`` for inliers, larger for outliers.

        With ``exclude_siblings`` each passage is fitted alone against the
        background, which is how the same-retrieved-set exclusion is realised
        for a density estimator. Without it, all retrieved items are fitted
        together and a cluster of injected passages props itself up. The
        background is stripped of every retrieved row in both cases, so no
        passage can appear twice in its own neighbourhood.
        """
        try:
            from sklearn.neighbors import LocalOutlierFactor
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "mode='lof' needs scikit-learn (pinned in docker/requirements.txt)"
            ) from exc

        rows, bg = self._background_vectors()
        keep = np.array([int(r) not in bg_excluded for r in rows], dtype=bool)
        bg = bg[keep]
        if len(bg) < 2:
            raise ValueError("background sample is too small for LOF")
        k = max(1, min(self.n_neighbors, len(bg)))

        def fit(block: np.ndarray, take: int) -> np.ndarray:
            lof = LocalOutlierFactor(n_neighbors=k, metric="cosine", novelty=False)
            lof.fit(block)
            return -np.asarray(lof.negative_outlier_factor_[:take], dtype=np.float64)

        if self.exclude_siblings:
            return np.concatenate([fit(np.vstack([v[None, :], bg]), 1) for v in vecs])
        return fit(np.vstack([vecs, bg]), len(vecs))

    # -- contract ---------------------------------------------------------- #
    def score(self, samples: Any) -> np.ndarray:
        """Isolation score per retrieved passage; higher = more suspicious.

        Args:
            samples: ``(query, passages)``. ``query`` is ignored.

        Returns:
            ``np.ndarray`` of float, one entry per passage, in input order.
        """
        _, passages = samples
        passages = list(passages)
        if not passages:
            return np.zeros(0, dtype=np.float64)

        vecs, hits = self._embed(passages)
        all_rows = {int(r) for h in hits for r in h}
        # With exclusion every passage is masked against the whole retrieved
        # set; without it (the ablation) only against its own corpus rows.
        excluded = (
            [all_rows] * len(passages)
            if self.exclude_siblings
            else [{int(r) for r in h} for h in hits]
        )
        if self.mode == "lof":
            return self._lof(vecs, all_rows)
        return self._knn_distance(vecs, excluded)
