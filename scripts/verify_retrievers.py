#!/usr/bin/env python3
"""Verify that every retriever configuration is what it claims to be.

Why this exists
---------------
``normalize=False`` is supposed to select PoisonedRAG-faithful **dot-product**
retrieval. Until today it was, for ``all-MiniLM-L6-v2``, a silent no-op: that
checkpoint's module list ends in a hard-wired ``Normalize``, so
``encode(normalize_embeddings=False)`` still returned unit-norm vectors, and the
embedding cache key ignored ``normalize`` entirely — a "dot product" run could
load cosine vectors written by an earlier cosine run and report dot-product
numbers that were cosine numbers.

Three pieces of machinery were added in response, and none had been exercised
end to end:

* :func:`aiforensics.data.rag.build_index` now **measures** the vectors it got
  instead of trusting the request, and records the measurement on the store;
* :attr:`aiforensics.data.rag.RagStore.similarity` reports ``'cosine'`` or
  ``'dot'``;
* :func:`aiforensics.data.rag.load_encoder` takes ``strip_normalizer=True`` to
  remove a trailing ``Normalize`` module;
* :func:`aiforensics.data.rag.embedding_cache_path` puts ``normalize`` in the
  cache key.

What this script checks
-----------------------
For each encoder and each ``normalize`` setting (plus, for self-normalising
checkpoints, a ``strip_normalizer=True`` arm):

1. does the encoder self-normalise (:func:`encoder_normalizes`)?
2. what does :attr:`RagStore.similarity` say versus what was requested? Any
   mismatch is flagged loudly; only the *documented* one — asking for dot from a
   checkpoint that hard-wires ``Normalize`` — is tolerated;
3. are the vectors in the finished FAISS index empirically unit-norm, and does
   that agree with what the store says? (Queries too:
   :meth:`RagStore.encode_queries` must follow the same flag.)
4. does retrieval actually *differ* between the cosine and dot arms? If the
   top-k are identical for every query, the dot-product path is inert and this
   script says so;
5. does the embedding cache cross-contaminate? Build cosine, then dot, then
   cosine again over one corpus name and assert the norms come back right every
   time. A second, harder probe checks whether ``strip_normalizer`` is missing
   from the cache key (``--skip-hazard`` to omit it).

The conclusion is printed explicitly: PoisonedRAG's own setup (``src/utils.py``
lists Contriever / Contriever-msmarco / ANCE; ``main.py`` defaults
``--score_function`` to ``dot``) makes ``(facebook/contriever, normalize=False)``
the faithful row, so that row must verify as ``dot`` with non-unit vectors.

Exit status is non-zero if any check fails, so this can gate an experiment run::

    python scripts/verify_retrievers.py --max-docs 3000 && python scripts/run_native_protocol.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np

# Allow `python scripts/verify_retrievers.py` without PYTHONPATH being set.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.data.rag import (
    Document,
    RagStore,
    build_index,
    embedding_cache_path,
    encoder_normalizes,
    load_corpus,
    load_encoder,
    load_queries,
)

log = logging.getLogger("verify_retrievers")

#: Both are cached locally under ~/.cache/huggingface/hub.
DEFAULT_ENCODERS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "facebook/contriever",
)

#: PoisonedRAG's own retriever + score function (src/utils.py, main.py).
FAITHFUL_ENCODER = "facebook/contriever"

#: PoisonedRAG NQ targets, used for realistic query text when present.
TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"

#: Matches ``rag._is_unit_norm``'s tolerance, so this script and the library
#: cannot disagree about what "unit norm" means.
NORM_TOL = 1e-3

PASS, FAIL, EXPECTED, INFO = "PASS", "FAIL", "EXPECTED", "INFO"


# --- check bookkeeping --------------------------------------------------


@dataclass
class Check:
    """One verified assertion.

    Attributes:
        arm: Configuration the check was made under.
        name: Short check name.
        status: ``PASS``, ``FAIL``, ``EXPECTED`` (a known, documented
            deviation) or ``INFO`` (measurement, no assertion).
        detail: Human-readable evidence.
    """

    arm: str
    name: str
    status: str
    detail: str

    @property
    def failed(self) -> bool:
        return self.status == FAIL


@dataclass
class Ledger:
    """Collects checks and prints them as they are made."""

    checks: list[Check] = field(default_factory=list)

    def add(self, arm: str, name: str, status: str, detail: str) -> Check:
        c = Check(arm=arm, name=name, status=status, detail=detail)
        self.checks.append(c)
        marker = {PASS: "  ok ", FAIL: " FAIL", EXPECTED: " note", INFO: "  ..."}[status]
        print(f"{marker}  {name:<26s} {detail}")
        return c

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.failed]


# --- measurement helpers ------------------------------------------------


def index_norms(store: RagStore, sample: int = 512) -> np.ndarray:
    """L2 norms of the vectors actually stored in the FAISS index.

    Reads the index back with ``reconstruct_n`` rather than re-measuring the
    embedding matrix: the index is what search scores against, so it is the only
    thing whose normalisation state matters.

    Args:
        store: A store built over a flat index.
        sample: Number of leading rows to measure. Normalisation is a property
            of the encoding pipeline, not of individual passages.

    Returns:
        1-D float32 array of norms.
    """
    n = min(len(store), sample)
    vecs = np.asarray(store.index.reconstruct_n(0, n), dtype=np.float32)
    return np.linalg.norm(vecs, axis=1)


def is_unit(norms: np.ndarray, tol: float = NORM_TOL) -> bool:
    """Whether every norm is 1 to within ``tol``."""
    return bool(norms.size and np.all(np.abs(norms - 1.0) <= tol))


def describe(norms: np.ndarray) -> str:
    """Compact ``min/mean/max`` summary of a norm vector."""
    if not norms.size:
        return "no vectors"
    return (
        f"n={norms.size} norms min={norms.min():.4f} mean={norms.mean():.4f} "
        f"max={norms.max():.4f}"
    )


def top_ids(store: RagStore, queries: Sequence[str], k: int) -> list[tuple[str, ...]]:
    """Top-k document ids per query, in rank order."""
    batches = store.search(list(queries), k=k)
    return [tuple(h.doc_id for h in hits) for hits in batches]  # type: ignore[union-attr]


# --- inputs -------------------------------------------------------------


def load_docs(corpus: str, max_docs: int) -> list[Document]:
    """Load a small slice of the corpus to index."""
    docs = load_corpus(corpus, max_docs=max_docs)
    if not docs:
        raise SystemExit(f"corpus {corpus!r} returned no documents")
    return docs


def load_query_texts(corpus: str, n: int) -> list[str]:
    """Query strings, preferring the PoisonedRAG target questions."""
    if TARGETS.exists():
        targets = json.loads(TARGETS.read_text(encoding="utf-8"))
        texts = [t["question"] for t in list(targets.values())[:n]]
        if texts:
            return texts
    return [q.text for q in load_queries(corpus, max_queries=n)]


# --- per-arm verification -----------------------------------------------


def verify_arm(
    ledger: Ledger,
    *,
    encoder_id: str,
    model: Any,
    variant: str,
    normalize: bool,
    self_normalizes: bool,
    docs: Sequence[Document],
    queries: Sequence[str],
    k: int,
    batch_size: int | None,
    use_cache: bool,
) -> dict[str, Any]:
    """Build one index and check what it turned out to be.

    Args:
        ledger: Where checks are recorded.
        encoder_id: HuggingFace model id (also the cache-key component).
        model: The loaded ``SentenceTransformer`` for this arm.
        variant: ``"plain"`` or ``"stripped"`` — part of the corpus name, so the
            two never share an embedding cache entry.
        normalize: The request passed to :func:`build_index`.
        self_normalizes: Whether ``model`` still ends in a ``Normalize`` module.
        docs: Corpus slice to index.
        queries: Query strings for the retrieval comparison.
        k: Hits per query.
        batch_size: Encoder batch size.
        use_cache: Whether to use the embedding cache.

    Returns:
        A row describing the arm, including its top-k ids for cross-arm
        comparison.
    """
    arm = f"{encoder_id} | normalize={normalize} | {variant}"
    requested = "cosine" if normalize else "dot"
    print(f"\n--- {arm}")

    corpus_name = f"verify-{variant}-{len(docs)}"
    cache = embedding_cache_path(corpus_name, encoder_id, [d.doc_id for d in docs],
                                 normalize=normalize)
    store = build_index(
        docs,
        encoder=model,
        corpus_name=corpus_name,
        normalize=normalize,
        batch_size=batch_size,
        use_cache=use_cache,
        show_progress=False,
        encoder_name=encoder_id,
    )

    ledger.add(arm, "encoder_normalizes", INFO,
               f"{'yes — module list ends in Normalize' if self_normalizes else 'no'}")
    ledger.add(arm, "cache file", INFO, cache.name)

    # (2) reported vs requested similarity.
    if store.similarity == requested:
        ledger.add(arm, "similarity vs request", PASS,
                   f"requested {requested!r}, store reports {store.similarity!r}")
    elif requested == "dot" and self_normalizes:
        ledger.add(
            arm, "similarity vs request", EXPECTED,
            f"MISMATCH: requested 'dot', store reports {store.similarity!r} — this is "
            "the documented no-op: the checkpoint hard-wires Normalize. Reach dot "
            "product with load_encoder(..., strip_normalizer=True).",
        )
    else:
        ledger.add(
            arm, "similarity vs request", FAIL,
            f"MISMATCH: requested {requested!r} but store reports "
            f"{store.similarity!r}, and it is not explained by a Normalize module",
        )

    # (3) do the stored vectors really have the norms the store claims?
    dn = index_norms(store)
    unit = is_unit(dn)
    ledger.add(
        arm, "index vector norms", PASS if unit == store.normalized else FAIL,
        f"{describe(dn)} -> {'unit' if unit else 'NOT unit'}-norm; store.normalized="
        f"{store.normalized} ({'agrees' if unit == store.normalized else 'CONTRADICTED'})",
    )

    # Queries must live in the same space as the documents, or the inner product
    # is neither cosine nor dot but a mixture with a query-length factor.
    qn = np.linalg.norm(store.encode_queries(list(queries)), axis=1)
    q_unit = is_unit(qn)
    ledger.add(
        arm, "query vector norms", PASS if q_unit == store.normalized else FAIL,
        f"{describe(qn)} -> {'unit' if q_unit else 'NOT unit'}-norm; "
        f"{'matches the document side' if q_unit == store.normalized else 'DIVERGES from the document side'}",
    )

    hits = top_ids(store, queries, k)
    return {
        "encoder": encoder_id,
        "variant": variant,
        "normalize_requested": normalize,
        "self_normalizes": self_normalizes,
        "similarity_requested": requested,
        "similarity_reported": store.similarity,
        "store_normalized": bool(store.normalized),
        "vectors_unit_norm": bool(unit),
        "doc_norm_min": float(dn.min()), "doc_norm_mean": float(dn.mean()),
        "doc_norm_max": float(dn.max()),
        "query_vectors_unit_norm": bool(q_unit),
        "query_norm_mean": float(qn.mean()),
        "cache_file": cache.name,
        "num_docs": len(store),
        "top_ids": [list(t) for t in hits],
        "_hits": hits,
    }


# --- cross-arm and cache verification -----------------------------------


def verify_retrieval_differs(
    ledger: Ledger, encoder_id: str, cos: dict[str, Any], dot: dict[str, Any],
    queries: Sequence[str],
) -> dict[str, Any]:
    """Check that the dot-product path actually changes retrieval.

    An index whose "dot product" ranking is identical to its cosine ranking on
    every query is not doing anything: the flag flipped but the experiment did
    not change.
    """
    arm = f"{encoder_id} | cosine vs dot"
    print(f"\n--- {arm}")
    if cos["similarity_reported"] == dot["similarity_reported"]:
        ledger.add(
            arm, "retrieval differs", EXPECTED,
            f"both arms are {cos['similarity_reported']!r} — dot product is NOT "
            "reachable in this configuration, so there is nothing to compare",
        )
        return {"comparable": False, "queries_with_different_topk": None}

    diffs = [i for i, (a, b) in enumerate(zip(cos["_hits"], dot["_hits"])) if a != b]
    rank1 = sum(1 for a, b in zip(cos["_hits"], dot["_hits"])
                if a[:1] != b[:1])
    n = len(cos["_hits"])
    if diffs:
        example = queries[diffs[0]]
        ledger.add(
            arm, "retrieval differs", PASS,
            f"top-k differs on {len(diffs)}/{n} queries (rank-1 differs on "
            f"{rank1}/{n}); e.g. {example!r}",
        )
    else:
        ledger.add(
            arm, "retrieval differs", FAIL,
            f"top-k is IDENTICAL on all {n} queries — the dot-product path is "
            "inert: one arm claims 'dot' and the other 'cosine' but they retrieve "
            "the same documents in the same order",
        )
    return {
        "comparable": True,
        "num_queries": n,
        "queries_with_different_topk": len(diffs),
        "queries_with_different_rank1": rank1,
    }


def verify_cache_roundtrip(
    ledger: Ledger,
    *,
    encoder_id: str,
    cosine_model: Any,
    dot_model: Any,
    dot_self_normalizes: bool,
    docs: Sequence[Document],
    batch_size: int | None,
) -> dict[str, Any]:
    """cosine -> dot -> cosine over one corpus name; norms must be right each time.

    This is the exact sequence that used to silently reuse cosine vectors for a
    dot-product run, before ``normalize`` entered the cache key.
    """
    arm = f"{encoder_id} | cache round-trip"
    print(f"\n--- {arm}")
    corpus_name = f"verify-cache-{len(docs)}"

    ids = [d.doc_id for d in docs]
    p_cos = embedding_cache_path(corpus_name, encoder_id, ids, normalize=True)
    p_dot = embedding_cache_path(corpus_name, encoder_id, ids, normalize=False)
    ledger.add(
        arm, "cache keys differ", PASS if p_cos != p_dot else FAIL,
        f"cosine -> {p_cos.name} | dot -> {p_dot.name}",
    )

    steps = [
        ("cosine (1st)", cosine_model, True, True),
        ("dot", dot_model, False, dot_self_normalizes),
        ("cosine (2nd, from cache)", cosine_model, True, True),
    ]
    rows = []
    for label, model, normalize, expect_unit in steps:
        store = build_index(
            docs, encoder=model, corpus_name=corpus_name, normalize=normalize,
            batch_size=batch_size, use_cache=True, show_progress=False,
            encoder_name=encoder_id,
        )
        norms = index_norms(store)
        unit = is_unit(norms)
        ok = unit == expect_unit
        ledger.add(
            arm, f"{label}", PASS if ok else FAIL,
            f"similarity={store.similarity!r} {describe(norms)} -> "
            f"{'unit' if unit else 'NOT unit'}-norm, expected "
            f"{'unit' if expect_unit else 'NOT unit'}"
            + ("" if ok else " — CACHE CROSS-CONTAMINATION"),
        )
        rows.append({
            "step": label, "similarity": store.similarity,
            "unit_norm": bool(unit), "expected_unit_norm": bool(expect_unit),
            "norm_mean": float(norms.mean()), "ok": bool(ok),
        })
    return {"corpus_name": corpus_name, "steps": rows}


def verify_strip_cache_hazard(
    ledger: Ledger,
    *,
    encoder_id: str,
    plain_model: Any,
    stripped_model: Any,
    docs: Sequence[Document],
    batch_size: int | None,
) -> dict[str, Any]:
    """Probe whether ``strip_normalizer`` is missing from the embedding cache key.

    ``embedding_cache_path`` keys on (corpus, encoder, size, ids, normalize) —
    not on whether the encoder had its ``Normalize`` module removed. So a
    ``normalize=False`` build with the *unstripped* model writes unit-norm
    vectors to the dot-product cache slot, and a later ``normalize=False`` build
    with the *stripped* model can read them straight back, reporting 'cosine'
    for what should be a genuine dot-product index.

    The probe builds both, then rebuilds the stripped one with ``use_cache=False``
    to establish ground truth.
    """
    arm = f"{encoder_id} | strip_normalizer cache key"
    print(f"\n--- {arm}")
    corpus_name = f"verify-hazard-{len(docs)}"
    common = dict(corpus_name=corpus_name, normalize=False, batch_size=batch_size,
                  show_progress=False, encoder_name=encoder_id)

    plain = build_index(docs, encoder=plain_model, use_cache=True, **common)
    plain_unit = is_unit(index_norms(plain))

    cached = build_index(docs, encoder=stripped_model, use_cache=True, **common)
    cached_norms = index_norms(cached)

    truth = build_index(docs, encoder=stripped_model, use_cache=False, **common)
    truth_norms = index_norms(truth)

    contaminated = is_unit(cached_norms) and not is_unit(truth_norms)
    ledger.add(
        arm, "stripped build not contaminated", FAIL if contaminated else PASS,
        (
            f"unstripped normalize=False wrote {'unit' if plain_unit else 'non-unit'}"
            f"-norm vectors to the dot cache slot; stripped build with cache -> "
            f"{describe(cached_norms)} (similarity={cached.similarity!r}), without "
            f"cache -> {describe(truth_norms)} (similarity={truth.similarity!r})"
        ) + (
            " — the cache key ignores strip_normalizer; a stripped dot-product run "
            "must pass a distinct corpus_name/encoder_name or use_cache=False"
            if contaminated else " — consistent"
        ),
    )
    return {
        "corpus_name": corpus_name,
        "unstripped_unit_norm": bool(plain_unit),
        "stripped_cached_unit_norm": bool(is_unit(cached_norms)),
        "stripped_uncached_unit_norm": bool(is_unit(truth_norms)),
        "stripped_cached_similarity": cached.similarity,
        "stripped_uncached_similarity": truth.similarity,
        "contaminated": bool(contaminated),
    }


# --- driver -------------------------------------------------------------


def guard(ledger: Ledger, arm: str, name: str, fn: Any) -> Any:
    """Run ``fn``; on exception record a FAIL and return ``None``.

    One arm that cannot be built (a missing checkpoint, an OOM) must not abort
    the sweep or, worse, exit zero because nothing was checked.
    """
    try:
        return fn()
    except Exception as exc:
        ledger.add(arm, name, FAIL, f"{type(exc).__name__}: {exc}")
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--encoders", nargs="+", default=list(DEFAULT_ENCODERS),
                   help="Encoder ids to verify.")
    p.add_argument("--corpus", default="nq", help="Registered corpus to slice.")
    p.add_argument("--max-docs", type=int, default=3000,
                   help="Passages to index per arm; keep it small, this is a "
                        "configuration check, not an experiment.")
    p.add_argument("--queries", type=int, default=10,
                   help="Queries used for the cosine-vs-dot retrieval comparison.")
    p.add_argument("--k", type=int, default=10, help="Hits per query.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Encoder batch size (default: settings.batch_size).")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass the embedding cache for the per-arm builds. The "
                        "cache round-trip check always uses the cache — that is "
                        "what it tests.")
    p.add_argument("--skip-hazard", action="store_true",
                   help="Skip the strip_normalizer cache-key probe.")
    p.add_argument("--report", type=Path, default=REPORT_DIR / "verify_retrievers.json",
                   help="Where to write the JSON report.")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    set_seed(args.seed)

    docs = load_docs(args.corpus, args.max_docs)
    queries = load_query_texts(args.corpus, args.queries)
    print("=" * 96)
    print("RETRIEVER CONFIGURATION VERIFICATION")
    print("=" * 96)
    print(f"corpus={args.corpus} docs={len(docs):,} queries={len(queries)} k={args.k} "
          f"tol={NORM_TOL}")

    ledger = Ledger()
    arms: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    caches: dict[str, Any] = {}
    hazards: dict[str, Any] = {}

    for encoder_id in args.encoders:
        print("\n" + "=" * 96)
        print(f"ENCODER {encoder_id}")
        print("=" * 96)
        try:
            plain = load_encoder(encoder_id)
        except Exception as exc:
            ledger.add(encoder_id, "load_encoder", FAIL, f"{type(exc).__name__}: {exc}")
            continue

        modules = [type(m).__name__ for _, m in plain.named_children()]
        self_norm = encoder_normalizes(plain)
        print(f"modules: {' -> '.join(modules)}")

        stripped = None
        if self_norm:
            stripped = guard(
                ledger, f"{encoder_id} | stripped", "load_encoder(strip_normalizer=True)",
                lambda eid=encoder_id: load_encoder(eid, strip_normalizer=True),
            )
        if stripped is not None:
            still = encoder_normalizes(stripped)
            ledger.add(
                f"{encoder_id} | stripped", "strip_normalizer works",
                PASS if not still else FAIL,
                "Normalize module removed" if not still
                else "Normalize module SURVIVED strip_normalizer=True",
            )

        plan: list[tuple[str, Any, bool]] = [
            ("plain", plain, True),
            ("plain", plain, False),
        ]
        if stripped is not None:
            plan.append(("stripped", stripped, False))

        rows = []
        for variant, model, normalize in plan:
            row = guard(
                ledger, f"{encoder_id} | normalize={normalize} | {variant}",
                "build + verify arm",
                lambda variant=variant, model=model, normalize=normalize,
                       eid=encoder_id, sn=self_norm: verify_arm(
                    ledger,
                    encoder_id=eid, model=model, variant=variant,
                    normalize=normalize,
                    self_normalizes=sn if variant == "plain" else False,
                    docs=docs, queries=queries, k=args.k,
                    batch_size=args.batch_size, use_cache=not args.no_cache,
                ),
            )
            if row is None:
                continue
            row["modules"] = modules
            rows.append(row)
            arms.append(row)

        if len(rows) >= 2:
            # The dot-capable arm: the stripped one when the checkpoint
            # hard-wires cosine, otherwise the plain normalize=False arm.
            cos, dot = rows[0], rows[-1]
            comparisons[encoder_id] = verify_retrieval_differs(
                ledger, encoder_id, cos, dot, queries
            )

        caches[encoder_id] = guard(
            ledger, f"{encoder_id} | cache round-trip", "cache round-trip",
            # Loop variables bound as defaults. guard() calls this immediately,
            # so late binding is harmless today, but it is one refactor away
            # from every probe reporting the last encoder's result.
            lambda eid=encoder_id, pl=plain, st=stripped, sn=self_norm:
                verify_cache_roundtrip(
                    ledger, encoder_id=eid, cosine_model=pl,
                    dot_model=st if st is not None else pl,
                    dot_self_normalizes=False if st is not None else sn,
                    docs=docs, batch_size=args.batch_size,
                ),
        )

        if stripped is not None and not args.skip_hazard:
            hazards[encoder_id] = guard(
                ledger, f"{encoder_id} | strip_normalizer cache key",
                "strip_normalizer cache probe",
                lambda eid=encoder_id, pl=plain, st=stripped:
                    verify_strip_cache_hazard(
                        ledger, encoder_id=eid, plain_model=pl,
                        stripped_model=st, docs=docs, batch_size=args.batch_size,
                    ),
            )

    # -- summary table ---------------------------------------------------
    print("\n" + "=" * 96)
    print("ARM SUMMARY")
    print("=" * 96)
    head = (f"{'encoder':<40s} {'variant':<9s} {'norm?':<6s} {'requested':<10s} "
            f"{'reported':<9s} {'unit':<5s} {'mean|v|':>8s}")
    print(head)
    print("-" * len(head))
    for r in arms:
        flag = "" if r["similarity_reported"] == r["similarity_requested"] else "  <-- MISMATCH"
        print(f"{r['encoder']:<40s} {r['variant']:<9s} "
              f"{r['normalize_requested']!s:<6s} {r['similarity_requested']:<10s} "
              f"{r['similarity_reported']:<9s} {r['vectors_unit_norm']!s:<5s} "
              f"{r['doc_norm_mean']:>8.4f}{flag}")

    # -- PoisonedRAG-faithful conclusion ---------------------------------
    print("\n" + "=" * 96)
    print("POISONEDRAG-FAITHFUL SETUP")
    print("=" * 96)
    print("PoisonedRAG (USENIX Security 2025) retrieves with Contriever "
          "(src/utils.py) and scores with --score_function dot by default "
          "(main.py), over un-normalised mean-pooled vectors "
          "(Contriever.forward, normalize=False).")
    faithful = next(
        (r for r in arms if r["encoder"] == FAITHFUL_ENCODER
         and r["variant"] == "plain" and r["normalize_requested"] is False),
        None,
    )
    arm = f"{FAITHFUL_ENCODER} | normalize=False | plain"
    if faithful is None:
        ledger.add(arm, "faithful row verified", INFO,
                   f"{FAITHFUL_ENCODER} was not in --encoders; nothing to conclude")
        print(f"NOT CHECKED: {FAITHFUL_ENCODER} was not in --encoders.")
    elif faithful["similarity_reported"] == "dot" and not faithful["vectors_unit_norm"]:
        ledger.add(arm, "faithful row verified", PASS,
                   f"similarity='dot', mean |v|={faithful['doc_norm_mean']:.4f}")
        print(f"=> THE FAITHFUL ROW IS ({FAITHFUL_ENCODER}, normalize=False): it "
              f"verifies as similarity='dot' with non-unit vectors "
              f"(mean |v|={faithful['doc_norm_mean']:.4f}). Use it for "
              f"PoisonedRAG reproductions and record similarity='dot' in the manifest.")
    else:
        ledger.add(
            arm, "faithful row verified", FAIL,
            f"similarity={faithful['similarity_reported']!r}, "
            f"unit_norm={faithful['vectors_unit_norm']}",
        )
        print(f"=> THE FAITHFUL ROW ({FAITHFUL_ENCODER}, normalize=False) DID NOT "
              f"VERIFY: it reports {faithful['similarity_reported']!r}. "
              "PoisonedRAG cannot be reproduced faithfully in this state.")

    # -- report + exit ---------------------------------------------------
    for r in arms:
        r.pop("_hits", None)
    payload = {
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items()},
        "arms": arms,
        "retrieval_comparison": comparisons,
        "cache_roundtrip": caches,
        "strip_cache_hazard": hazards,
        "checks": [asdict(c) for c in ledger.checks],
        "num_failures": len(ledger.failures),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 96)
    counts = {s: sum(1 for c in ledger.checks if c.status == s)
              for s in (PASS, FAIL, EXPECTED, INFO)}
    print(f"{counts[PASS]} passed, {counts[FAIL]} failed, "
          f"{counts[EXPECTED]} expected-deviation, {counts[INFO]} informational")
    for c in ledger.failures:
        print(f"  FAIL [{c.arm}] {c.name}: {c.detail}")
    print(f"written: {args.report}")
    return 1 if ledger.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
