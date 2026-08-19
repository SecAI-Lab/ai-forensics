"""CleanBase — corpus-level clique detection of injected RAG documents.

Paper
-----
Weifei Jin, Xilong Wang, Wei Zou, Jinyuan Jia, Neil Gong (Duke University /
The Pennsylvania State University). *CleanBase: Detecting Malicious Documents
in RAG Knowledge Databases.* arXiv:2605.00460 — https://arxiv.org/abs/2605.00460
**This is an arXiv-only preprint. It is not peer reviewed and carries no venue;
every number quoted below is author-reported.** Reference implementation
(Apache-2.0): https://github.com/WeifeiJin/CleanBase.

Clean-corpus false alarms, as reported by the paper
---------------------------------------------------
Table 1 reports a **per-document FPR of 1.9% averaged over six knowledge
databases** (NQ 1.8, HotpotQA 4.0, FiQA 0.8, ArguAna 0.5–1.3, SciFact 0.0–1.0,
FEVER 3.4) and nine injection attacks, where FPR is the fraction of *benign*
documents flagged as malicious. That figure is measured on poisoned databases,
but because the statistic is computed over benign documents only it is the
paper's own answer to "what does this method do to a corpus you should not
touch". HotpotQA and FEVER, the two most redundant corpora, cost ~2× the
average — benign near-duplicates are this method's false-positive source.

Method (three steps, exactly as in §4 of the paper; no LLM anywhere)
-------------------------------------------------------------------
I.  **kNN similarity graph.** Embed every document, connect each node to its
    ``k`` most similar documents (``k=10``), edge weight = cosine similarity.
    The default edge rule is the *union* one: an edge exists if **either**
    endpoint is in the other's top-k. ``mutual=True`` selects the paper's
    ``CleanBase-Mutual`` ablation (lower FPR, higher FNR).
II. **Statistical pruning.** Sample 50% of the edges, take the mean ``mu`` and
    standard deviation ``sigma`` of their weights, set ``tau = mu + z * sigma``
    (``z=2.5``, "confidence level above 0.99"), and keep only edges above
    ``tau``. The paper's ablation shows this step is load-bearing: without it
    FPR exceeds 99%.
III.**Clique detection.** Enumerate maximal cliques of the pruned graph with
    Bron–Kerbosch (Bron & Kerbosch, CACM 1973) and flag every document in a
    clique of **at least three mutually connected nodes**. The reference
    implementation additionally (a) ignores maximal cliques larger than
    ``max_clique_size=11`` for tractability and (b) drops cliques whose average
    edge weight is ``>= 1.0`` (exact duplicates). Both guards are reproduced
    here and both are configurable.

Adaptation to this testbed's per-retrieval interface
----------------------------------------------------
CleanBase is a **corpus-level** method: it asks which documents in the whole
database sit in a mutually-similar clump, not which of a retrieved top-k looks
odd. :meth:`~Detector.score` is per retrieved set, so the bridge is:

* the entire Step I–III pipeline runs **once**, lazily, on first use, over the
  whole store, and is cached on the instance (:meth:`CleanBaseDetector.fit_corpus`);
* :meth:`CleanBaseDetector.score` then only *looks up* the precomputed value for
  each retrieved passage — it is O(len(passages)) and query-independent;
* the corpus-level machinery is public: :meth:`CleanBaseDetector.suspicious_doc_ids`
  returns the paper-exact flagged set and :meth:`CleanBaseDetector.corpus_report`
  returns the graph statistics, which is what answers "is this database poisoned
  at all?" without reference to any single query.

Consequence: a document's score never varies across queries, so per-query
metrics contain correlated errors — the same benign duplicate is a false
positive for every query that retrieves it.

Score, and why the default threshold is 2.5
-------------------------------------------
CleanBase natively emits a *label*, not a score. To stay sweepable, the
per-document statistic here is the **tightest-triangle margin in units of
sigma**::

    score(v) = ( max over triangles (v,u,w) in the kNN graph of
                 min(w_vu, w_vw, w_uw) - mu ) / sigma

i.e. how many standard deviations above the mean edge weight the *weakest* edge
of the tightest mutually-similar trio containing ``v`` sits. ``score(v) > z`` is
exactly the condition "``v`` belongs to a 3-clique of the pruned graph", so
thresholding at ``z`` reproduces CleanBase's own rule, and ``threshold``
therefore defaults to ``z`` (2.5). Sweeping it below 2.5 admits looser trios;
above, tighter ones. :meth:`suspicious_doc_ids` applies the two reference-
implementation guards on top, so it is always a subset of ``score >= z``.

Expected failure modes (documented, not hidden)
------------------------------------------------
* **White-box PoisonedRAG (HotFlip prefixes) is the weak case.** The paper's own
  average FNR for PoisonedRAG-W is 34.2% (72.6% on NQ, 66.0% on FEVER) versus
  5.7% for PoisonedRAG-B. Gradient-optimised gibberish prefixes differ per
  document, so the injected set stops being mutually similar and the clique
  dissolves. Expect roughly two-thirds recall at best on that variant.
* **Fewer than three injected documents per target question is a structural
  miss.** A clique needs three nodes; ``N=2`` is undetectable by construction,
  whatever the threshold.
* **The ``max_clique_size`` cap is an evasion route once ``k`` is raised.** A
  maximal clique of 20 near-identical documents is larger than the 11-node cap
  and is skipped whole. At the default ``k=10`` this rarely bites, because each
  node only contributes ``k`` outgoing edges and maximal cliques seldom exceed
  ``k+1``; the paper sweeps ``k`` to 100, and there the hole is real. Set
  ``max_clique_size=None`` to close it at the cost of runtime.
* **Benign redundancy is the false-positive source**: FAQ boilerplate, templated
  records and near-duplicate paragraphs form genuine cliques. This is where the
  paper's 1.9% comes from, and it is corpus-dependent, not a constant. The
  duplicate prefilter removes such a clique from the *label* set but not from
  the *score*, so verbatim duplicates still score above ``z``; the two outputs
  disagree exactly there, and ``suspicious_doc_ids()`` is the conservative one.
* ``mu``/``sigma`` are estimated from the *poisoned* graph, so a heavily poisoned
  database raises its own threshold; the method degrades as the poisoning rate
  grows rather than failing loudly.
* Cost is corpus-level: Step I is an exact ``N x N`` flat-index search, which the
  paper notes dominates its runtime. Scoring afterwards is free.

Deviations from the paper, stated up front
-------------------------------------------
* The paper embeds documents with its own model ``f'`` (Contriever-msmarco),
  possibly different from the RAG retriever ``f``. Here the graph is built from
  the vectors **already in the store's index**, i.e. ``f' = f``. Re-embedding a
  whole corpus with a second model is a GPU cost this testbed should spend
  deliberately, not implicitly; the ``encoder`` argument is accepted for
  interface compatibility and deliberately unused.
* Bron–Kerbosch with pivoting is implemented in this module over bitset
  adjacency. The reference implementation calls ``networkx.find_cliques``;
  networkx is **not** in ``docker/requirements.txt`` and is not added as a
  dependency for one function. Only numpy and scipy are used.
* Triangle margins are searched over each node's ``triangle_neighbors``
  strongest incident edges (a tractability cap). The equivalence "score > z
  iff in a pruned 3-clique" is exact whenever a node has no more than that many
  incident edges above ``tau``; flagged documents additionally have their score
  raised to their clique's own margin, so the subset property always holds.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["CleanBaseDetector"]

#: Mirrors :data:`aiforensics.seed.DEFAULT_SEED`; duplicated so importing this
#: module (which the registry does eagerly) does not drag in torch.
_DEFAULT_SEED = 1337

#: Rows per chunk of the corpus-wide kNN search; bounds peak memory only.
_SEARCH_CHUNK = 2048

#: Similarity attributed to a document with no mutually-similar trio at all
#: (and to a passage absent from the store). -1.0 is the minimum a cosine can
#: take, so the resulting z-score is the floor of the statistic: "no evidence".
_NO_EVIDENCE_SIM = -1.0


def _iter_bits(mask: int) -> Iterator[int]:
    """Yield the indices of the set bits of ``mask``, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _bron_kerbosch(nbr: list[int], budget: int) -> list[int]:
    """Enumerate maximal cliques of a bitset graph, with pivoting.

    Args:
        nbr: ``nbr[v]`` is the bitmask of ``v``'s neighbours (no self-loop).
        budget: Stop after this many maximal cliques. Guards against the
            exponential worst case on a pathologically dense component.

    Returns:
        Maximal cliques as node bitmasks.
    """
    out: list[int] = []

    def expand(r: int, p: int, x: int) -> None:
        if len(out) >= budget:
            return
        if not p and not x:
            out.append(r)
            return
        # Pivot = the vertex of P|X with the most neighbours in P (Tomita et al.).
        pivot = max(_iter_bits(p | x), key=lambda u: bin(p & nbr[u]).count("1"))
        for v in list(_iter_bits(p & ~nbr[pivot])):
            bit = 1 << v
            expand(r | bit, p & nbr[v], x & nbr[v])
            p &= ~bit
            x |= bit
            if len(out) >= budget:
                return

    expand(0, (1 << len(nbr)) - 1, 0)
    return out


@register_detector("cleanbase", category="rag-defense")
class CleanBaseDetector(Detector):
    """Flag documents that sit in a clique of mutually-similar documents.

    ``samples`` is ``(query, passages)``; the query is unused because the
    statistic is a property of the corpus, not of the retrieval. Scores align
    positionally to ``passages``, higher = more suspicious.

    Args:
        store: :class:`~aiforensics.data.rag.RagStore` under investigation.
            Required — the graph is defined over it. Only ``.index``,
            ``.doc_ids``, ``.documents`` and ``.normalized`` are read; ground
            truth (``.injected_ids``) is never touched.
        encoder: Accepted for interface compatibility and unused; see the
            module docstring's deviations section.
        model: Unused; accepted so the shared detector factory can pass one.
        tokenizer: Unused; accepted for the same reason.
        threshold: Defaults to ``z``, which is the paper's own decision rule.
        k: Neighbourhood size of Step I. Paper default 10; larger k lowers FNR
            and raises FPR.
        z: Standard deviations above the mean edge weight for Step II's
            threshold ``tau = mu + z * sigma``. Paper default 2.5.
        sample_ratio: Fraction of edges sampled to estimate ``mu``/``sigma``.
            Paper default 0.5.
        mutual: ``False`` (default, paper's default) keeps an edge if *either*
            endpoint is in the other's top-k; ``True`` requires both, which is
            the ``CleanBase-Mutual`` ablation.
        min_clique_size: Smallest maximal clique that counts. 3 in the paper;
            below 3 there is no clique.
        max_clique_size: Largest maximal clique that counts, or ``None`` for no
            cap. 11 in the reference implementation — see the failure modes.
        duplicate_weight: Maximal cliques whose *average* edge weight is at
            least this are dropped as exact duplicates (the reference
            implementation's ``prefilter_density_threshold``, default 1.0).
        triangle_neighbors: Per-node cap on incident edges considered when
            searching for the tightest triangle.
        max_cliques: Enumeration budget across the whole corpus.
        max_component: Connected components of the pruned graph larger than
            this are skipped with a warning rather than enumerated, since
            Bron–Kerbosch is exponential in the worst case.
        seed: Seeds the 50% edge sample so runs are reproducible.
    """

    name = "cleanbase"
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
        k: int = 10,
        z: float = 2.5,
        sample_ratio: float = 0.5,
        mutual: bool = False,
        min_clique_size: int = 3,
        max_clique_size: int | None = 11,
        duplicate_weight: float = 1.0,
        triangle_neighbors: int = 32,
        max_cliques: int = 200_000,
        max_component: int = 4096,
        seed: int = _DEFAULT_SEED,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=z if threshold is None else threshold,
            k=k, z=z, sample_ratio=sample_ratio, mutual=mutual,
            min_clique_size=min_clique_size, max_clique_size=max_clique_size,
            duplicate_weight=duplicate_weight,
            triangle_neighbors=triangle_neighbors, max_cliques=max_cliques,
            max_component=max_component, seed=seed, **config,
        )
        if store is None:
            raise ValueError(
                "cleanbase builds a graph over a corpus; pass store=RagStore(...)"
            )
        if min_clique_size < 3:
            raise ValueError("a clique has at least three nodes; min_clique_size >= 3")
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError(f"sample_ratio must be in (0, 1], got {sample_ratio}")
        self.store = store
        self.k = int(k)
        self.z = float(z)
        self.sample_ratio = float(sample_ratio)
        self.mutual = bool(mutual)
        self.min_clique_size = int(min_clique_size)
        self.max_clique_size = None if max_clique_size is None else int(max_clique_size)
        self.duplicate_weight = float(duplicate_weight)
        self.triangle_neighbors = max(2, int(triangle_neighbors))
        self.max_cliques = int(max_cliques)
        self.max_component = int(max_component)
        self.seed = int(seed)

        self._stats: dict[str, float] | None = None
        self._doc_scores: np.ndarray | None = None
        self._flagged: np.ndarray | None = None
        self._rows_by_text: dict[str, list[int]] | None = None
        self._edge_keys: np.ndarray | None = None
        self._edge_w: np.ndarray | None = None

    # -- corpus plumbing --------------------------------------------------- #
    def _row_map(self) -> dict[str, list[int]]:
        """Map every corpus surface form to the index rows that carry it.

        Both ``Document.text`` and ``Document.embedding_text()`` are keyed,
        because a retrieved passage arrives as the bare body while the index
        embedded ``"title text"``.
        """
        if self._rows_by_text is None:
            rows: dict[str, list[int]] = {}
            for row, doc_id in enumerate(self.store.doc_ids):
                doc = self.store.documents[doc_id]
                for key in {doc.text, doc.embedding_text()}:
                    rows.setdefault(key, []).append(row)
            self._rows_by_text = rows
        return self._rows_by_text

    def _vectors(self, start: int, count: int) -> np.ndarray:
        """Reconstruct a contiguous block of stored vectors, unit-normalised."""
        index = self.store.index
        try:
            vecs = index.reconstruct_n(start, count)
        except Exception:  # pragma: no cover - older faiss builds
            vecs = np.vstack([index.reconstruct(r) for r in range(start, start + count)])
        vecs = np.ascontiguousarray(np.asarray(vecs, dtype=np.float32))
        if not getattr(self.store, "normalized", True):
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
        return vecs

    # -- Step I ------------------------------------------------------------ #
    def _knn_edges(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Undirected kNN edge list as ``(packed_keys, weights)``, key-sorted.

        Keys pack ``(lo, hi)`` as ``lo * n + hi`` so an edge can be looked up by
        :func:`numpy.searchsorted`. Duplicate directions collapse; ``mutual``
        keeps only the pairs that appeared from both sides.
        """
        lo_all, hi_all, w_all = [], [], []
        for start in range(0, n, _SEARCH_CHUNK):
            count = min(_SEARCH_CHUNK, n - start)
            sims, idx = self.store.index.search(self._vectors(start, count), self.k + 1)
            src = np.repeat(np.arange(start, start + count, dtype=np.int64), self.k + 1)
            dst = idx.reshape(-1).astype(np.int64)
            wgt = np.clip(sims.reshape(-1).astype(np.float64), -1.0, 1.0)
            keep = (dst >= 0) & (dst != src)
            src, dst, wgt = src[keep], dst[keep], wgt[keep]
            lo_all.append(np.minimum(src, dst))
            hi_all.append(np.maximum(src, dst))
            w_all.append(wgt)

        lo = np.concatenate(lo_all) if lo_all else np.zeros(0, np.int64)
        hi = np.concatenate(hi_all) if hi_all else np.zeros(0, np.int64)
        weights = np.concatenate(w_all) if w_all else np.zeros(0, np.float64)
        keys = lo * n + hi
        order = np.argsort(keys, kind="stable")
        keys, weights = keys[order], weights[order]
        uniq, first, counts = np.unique(keys, return_index=True, return_counts=True)
        if self.mutual:
            both = counts >= 2
            uniq, first = uniq[both], first[both]
        return uniq, weights[first]

    def _edge_weight(self, a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
        """Weight of each ``(a, b)`` edge, ``nan`` where the edge is absent."""
        if self._edge_keys is None or len(self._edge_keys) == 0:
            return np.full(len(a), np.nan, dtype=np.float64)
        keys = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b).astype(np.int64)
        pos = np.clip(np.searchsorted(self._edge_keys, keys), 0, len(self._edge_keys) - 1)
        hit = self._edge_keys[pos] == keys
        return np.where(hit, self._edge_w[pos], np.nan)

    # -- Steps II and III -------------------------------------------------- #
    def _triangle_margins(self, adj: Any, n: int, mu: float, sigma: float) -> np.ndarray:
        """Tightest-triangle min-edge weight per node, as a z-score.

        For every node, the ``triangle_neighbors`` strongest incident edges are
        paired up and the third side looked up; the score is the best (largest)
        weakest-side weight found, standardised. Nodes in no triangle get the
        floor score.
        """
        indptr, indices, data = adj.indptr, adj.indices, adj.data
        best = np.full(n, _NO_EVIDENCE_SIM, dtype=np.float64)
        pair_i, pair_j = np.triu_indices(self.triangle_neighbors, k=1)
        for v in range(n):
            lo, hi = indptr[v], indptr[v + 1]
            if hi - lo < 2:
                continue
            nbr, wgt = indices[lo:hi], data[lo:hi]
            if len(nbr) > self.triangle_neighbors:
                top = np.argpartition(-wgt, self.triangle_neighbors - 1)
                top = top[: self.triangle_neighbors]
                nbr, wgt = nbr[top], wgt[top]
            m = len(nbr)
            pi, pj = (
                (pair_i, pair_j)
                if m == self.triangle_neighbors
                else np.triu_indices(m, k=1)
            )
            third = self._edge_weight(nbr[pi], nbr[pj], n)
            tri = np.minimum(np.minimum(wgt[pi], wgt[pj]), third)
            if np.any(~np.isnan(tri)):
                best[v] = float(np.nanmax(tri))
        return (best - mu) / sigma

    def _cliques(self, pruned: Any, n: int) -> list[np.ndarray]:
        """Maximal cliques of the pruned graph, size- and duplicate-filtered."""
        from scipy.sparse.csgraph import connected_components

        # Any node of a >=3-clique keeps degree >= 2 under 2-core peeling, so
        # peeling only removes nodes that cannot matter.
        alive = np.ones(n, dtype=bool)
        for _ in range(64):
            deg = pruned @ alive.astype(np.float32)
            drop = alive & (deg < 2)
            if not drop.any():
                break
            alive &= ~drop
        nodes = np.flatnonzero(alive)
        if len(nodes) < self.min_clique_size:
            return []

        sub = pruned[nodes][:, nodes]
        n_comp, labels = connected_components(sub, directed=False)
        out: list[np.ndarray] = []
        budget = self.max_cliques
        for comp in range(n_comp):
            local = np.flatnonzero(labels == comp)
            if len(local) < self.min_clique_size or budget <= 0:
                continue
            if len(local) > self.max_component:
                # Bitset Bron-Kerbosch is exponential in the worst case; a giant
                # connected blob after pruning means the corpus is pathologically
                # redundant, not that it is poisoned. Skipping is a documented
                # miss, not a silent one.
                logger.warning(
                    "cleanbase: skipping a pruned component of %d nodes "
                    "(max_component=%d); cliques inside it are not enumerated.",
                    len(local), self.max_component,
                )
                continue
            block = sub[local][:, local].tocsr()
            nbr = [0] * len(local)
            for i in range(len(local)):
                mask = 0
                for j in block.indices[block.indptr[i] : block.indptr[i + 1]]:
                    if int(j) != i:
                        mask |= 1 << int(j)
                nbr[i] = mask
            found = _bron_kerbosch(nbr, budget)
            budget -= len(found)
            for mask in found:
                members = np.asarray(list(_iter_bits(mask)), dtype=np.int64)
                if len(members) < self.min_clique_size:
                    continue
                if self.max_clique_size is not None and len(members) > self.max_clique_size:
                    continue
                out.append(nodes[local[members]])
        if budget <= 0:
            logger.warning(
                "cleanbase: clique budget of %d exhausted; the flagged set is a "
                "lower bound. Raise max_cliques or max_clique_size.",
                self.max_cliques,
            )
        return out

    # -- public corpus-level API ------------------------------------------- #
    def fit_corpus(self, *, force: bool = False) -> CleanBaseDetector:
        """Run Steps I–III over the whole store and cache the result.

        Called lazily by :meth:`score`; call it directly to pay the cost up
        front or to inspect :meth:`corpus_report` without scoring anything.

        Args:
            force: Recompute even if a cached result exists.

        Returns:
            ``self``, so it chains.

        Raises:
            ImportError: If scipy is missing (pinned in docker/requirements.txt).
        """
        if self._doc_scores is not None and not force:
            return self
        try:
            from scipy.sparse import csr_matrix
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "cleanbase needs scipy for its sparse graph (pip install scipy; "
                "it is pinned in docker/requirements.txt)"
            ) from exc

        n = len(self.store)
        self._edge_keys, self._edge_w = self._knn_edges(n)
        if len(self._edge_keys) == 0:
            # A corpus with no kNN edges at all (n < 2) has no cliques and no
            # edge-weight distribution; report the floor for everything.
            self._stats = {"mu": 0.0, "sigma": 1.0, "tau": float("inf"),
                           "n_edges": 0, "n_pruned_edges": 0, "n_cliques": 0}
            self._doc_scores = np.full(n, _NO_EVIDENCE_SIM, dtype=np.float64)
            self._flagged = np.zeros(n, dtype=bool)
            return self
        lo, hi = self._edge_keys // n, self._edge_keys % n

        # Step II: threshold from a sampled half of the edges.
        rng = np.random.default_rng(self.seed)
        size = max(2, int(len(self._edge_w) * self.sample_ratio))
        sample = rng.choice(self._edge_w, size=min(size, len(self._edge_w)), replace=False)
        mu, sigma = float(np.mean(sample)), float(np.std(sample))
        sigma = max(sigma, 1e-12)
        tau = mu + self.z * sigma

        adj = csr_matrix(
            (np.concatenate([self._edge_w, self._edge_w]),
             (np.concatenate([lo, hi]), np.concatenate([hi, lo]))),
            shape=(n, n),
        )
        scores = self._triangle_margins(adj, n, mu, sigma)

        # Step III: cliques of the pruned graph.
        keep = self._edge_w > tau
        klo, khi = lo[keep], hi[keep]
        ones = np.ones(2 * len(klo), dtype=np.float32)
        pruned = csr_matrix(
            (ones, (np.concatenate([klo, khi]), np.concatenate([khi, klo]))),
            shape=(n, n),
        )
        flagged = np.zeros(n, dtype=bool)
        n_kept = 0
        for members in self._cliques(pruned, n):
            pi, pj = np.triu_indices(len(members), k=1)
            w = self._edge_weight(members[pi], members[pj], n)
            # Rounded to 4 decimals exactly as the reference implementation does,
            # so a float32 cosine of 0.9999999 between verbatim duplicates still
            # trips the default cut of 1.0 and the clique is dropped.
            if round(float(np.nanmean(w)), 4) >= self.duplicate_weight:
                continue  # copies of one another, not a coordinated injection
            n_kept += 1
            flagged[members] = True
            # Every clique edge exists and exceeds tau, so this margin is well
            # defined and above z; raising the score here is what guarantees
            # ``suspicious_doc_ids()`` is a subset of ``score >= z``.
            scores[members] = np.maximum(scores[members], (np.nanmin(w) - mu) / sigma)

        self._stats = {
            "mu": mu, "sigma": sigma, "tau": tau,
            "n_edges": len(self._edge_w),
            "n_pruned_edges": int(keep.sum()),
            "n_cliques": n_kept,
        }
        self._doc_scores, self._flagged = scores, flagged
        return self

    def suspicious_doc_ids(self) -> list[str]:
        """Corpus document ids CleanBase labels malicious (paper-exact rule)."""
        self.fit_corpus()
        assert self._flagged is not None
        return [self.store.doc_ids[r] for r in np.flatnonzero(self._flagged)]

    def corpus_report(self) -> dict[str, Any]:
        """Corpus-level verdict material: graph statistics and flagged share.

        ``flagged_fraction`` is the whole-database analogue of the paper's FPR.
        The paper's own average FPR on benign documents is 1.9%
        (``clean_fpr_reference``), so a flagged fraction of that order is what a
        *clean* corpus looks like under this method; a materially larger share,
        or cliques whose members all answer the same question, is the signal
        that the database has been poisoned. No verdict is asserted here — the
        comparison is the caller's to make and to report.
        """
        self.fit_corpus()
        assert self._stats is not None and self._flagged is not None
        n = len(self.store)
        return {
            "n_docs": n,
            "n_flagged": int(self._flagged.sum()),
            "flagged_fraction": float(self._flagged.sum()) / n if n else 0.0,
            "clean_fpr_reference": 0.019,
            **self._stats,
        }

    # -- contract ---------------------------------------------------------- #
    def score(self, samples: Any) -> np.ndarray:
        """Clique-membership margin per retrieved passage; higher = suspicious.

        Args:
            samples: ``(query, passages)``. ``query`` is ignored — the statistic
                is corpus-level. The first call triggers :meth:`fit_corpus`.

        Returns:
            ``np.ndarray`` of float, one entry per passage, in input order. A
            passage whose text matches no store document (encoder or text
            mismatch) gets the floor score ``(-1 - mu) / sigma``, the value of a
            document that belongs to no mutually-similar trio: unknown provenance
            is never reported as evidence of guilt. Where one text occupies
            several corpus rows the most suspicious row wins.
        """
        _, passages = samples
        passages = list(passages)
        if not passages:
            return np.zeros(0, dtype=np.float64)

        self.fit_corpus()
        assert self._doc_scores is not None and self._stats is not None
        floor = (_NO_EVIDENCE_SIM - self._stats["mu"]) / self._stats["sigma"]
        row_map = self._row_map()
        out = np.empty(len(passages), dtype=np.float64)
        for i, passage in enumerate(passages):
            rows = row_map.get(passage)
            out[i] = float(self._doc_scores[rows].max()) if rows else floor
        return out
