"""RAG corpora, dense indexes, and the store that poisoning attacks act on.

Threat model
------------
The RAG-poisoning setting in this testbed is *corpus injection*: the attacker
appends a small number of adversarial passages to an otherwise clean vector
store, crafted so they are retrieved for a target query and steer the
generator's answer. Detection is then "which of the N documents in this store
are injected?", and the headline metric is #detected / #injected. That means
the store itself has to carry unambiguous ground truth about which documents
were added by an attacker — hence :attr:`RagStore.injected_ids` and
:meth:`RagStore.injected_mask`.

Similarity metric
-----------------
The index is always ``IndexFlatIP`` (inner product); what the score *means*
depends on ``normalize``, and a run manifest must say which was used:

* ``normalize=True`` (**this project's default**) L2-normalises both documents
  and queries, so inner product *is* cosine similarity — what
  sentence-transformers bi-encoders are trained for.
* ``normalize=False`` leaves the mean-pooled vectors un-normalised, so inner
  product is the raw **dot product**. This is the *PoisonedRAG-faithful*
  setting: PoisonedRAG's retrievers are Contriever, Contriever-msmarco and ANCE
  (``src/utils.py``), and ``main.py`` scores them with ``--score_function dot``
  by default, over embeddings from ``Contriever.forward``, whose ``normalize``
  argument defaults to False. Under dot product, vector norm — and hence, in
  practice, passage length and token repetition — contributes to the ranking.

:attr:`RagStore.normalized` and :attr:`RagStore.similarity` record which regime
a store is in, and both are written into ``meta.json`` by :meth:`RagStore.save`.

Two traps make ``normalize=False`` more than a flag flip, both handled here:

1. Many sentence-transformers checkpoints — including this project's default
   ``all-MiniLM-L6-v2`` — end their module list with a ``Normalize`` layer, so
   ``encode(normalize_embeddings=False)`` still returns unit-norm vectors and
   the request is silently a no-op. :func:`build_index` measures the vectors it
   actually got, warns, and records the truth rather than the request. Use
   :func:`encoder_normalizes` to check ahead of time, or
   ``load_encoder(..., strip_normalizer=True)`` to remove that layer.
2. ``facebook/contriever`` ships no sentence-transformers config, so
   ``SentenceTransformer`` falls back to wrapping it in mean pooling.
   :func:`load_encoder` verifies that fallback instead of assuming it; see its
   docstring for why the pooling is equivalent to Contriever's own.

Index type
----------
The default is a flat (exhaustive, exact) index. That is deliberate even though
IVF/HNSW would be faster: an approximate index has recall loss, and recall loss
in this experiment is indistinguishable from "the poisoning attack failed to
get retrieved". Exact search removes that confound. For corpora large enough to
need ANN, pass ``index_factory="IVF4096,Flat"`` (or similar) explicitly and
record it in the run manifest.

Cost
----
Encoding is the expensive step, so corpus embeddings are cached to
``data/rag/embeddings`` keyed by (corpus, encoder, size, doc-id digest) and
reused across runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import numpy as np

from ..config import settings
from ..paths import RAG_DIR

log = logging.getLogger(__name__)

#: Small, fast, widely used bi-encoder. 384-dim, ~80 MB.
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

#: Bounded default so a first run finishes in seconds instead of hours.
#: BeIR/nq's full corpus is ~2.68M passages; see `scripts/build_rag_index.py`
#: for how to scale up.
DEFAULT_MAX_DOCS = 50_000

#: BeIR/nq holds only the `corpus` and `queries` configs. The relevance
#: judgements are published as a separate repo.
CORPUS_SOURCES: dict[str, str] = {"nq": "BeIR/nq"}
QRELS_SOURCES: dict[str, str] = {"nq": "BeIR/nq-qrels"}

#: On-disk layout of a saved store.
_INDEX_FILE = "index.faiss"
_DOCS_FILE = "documents.jsonl"
_META_FILE = "meta.json"
_STORE_SCHEMA_VERSION = 1


# --- optional heavy imports ---------------------------------------------


def _hf_load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset as hf_load_dataset  # noqa: N813
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the `datasets` package is required to load RAG corpora; "
            "it is pinned in docker/requirements.txt — run inside the container"
        ) from exc
    return hf_load_dataset


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "faiss-cpu is required to build or search a RAG index; it is "
            "pinned in docker/requirements.txt — run inside the container"
        ) from exc
    return faiss


def _sentence_transformer_cls() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "sentence-transformers is required to embed documents; it is "
            "pinned in docker/requirements.txt — run inside the container"
        ) from exc
    return SentenceTransformer


# --- records ------------------------------------------------------------


@dataclass
class Document:
    """One retrievable passage.

    Attributes:
        doc_id: Stable identifier. For BeIR this is the upstream ``_id``; for
            injected passages it is generated by
            :func:`aiforensics.data.poison.craft_poison_passages` and prefixed
            so it is greppable, but detection must never rely on that prefix.
        title: Passage title; may be empty.
        text: Passage body.
        meta: Free-form provenance (source repo, attack params, ...).
    """

    doc_id: str
    text: str
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def embedding_text(self) -> str:
        """The string actually embedded: ``"title text"`` per BeIR convention.

        Titles carry real signal in NQ, and an attacker can exploit the title
        field too, so it must be part of the embedded surface.
        """
        return f"{self.title} {self.text}".strip() if self.title else self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        """Build from a dict, tolerating BeIR's ``_id`` naming."""
        doc_id = d.get("doc_id") or d.get("_id") or d.get("id")
        if doc_id is None:
            raise KeyError(f"document is missing a doc_id/_id/id key: {sorted(d)}")
        return cls(
            doc_id=str(doc_id),
            text=str(d.get("text", "")),
            title=str(d.get("title") or ""),
            meta=dict(d.get("meta") or {}),
        )


@dataclass
class Query:
    """One retrieval query."""

    query_id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchResult(NamedTuple):
    """A ranked hit.

    The first three fields are the documented contract ``(doc_id, score,
    text)``; ``result[:3]`` unpacks to exactly that. ``injected`` is ground
    truth carried through from the store, so an attack-success check ("did my
    poison passage get retrieved for this query?") is a field lookup rather
    than a string match on the id.
    """

    doc_id: str
    score: float
    text: str
    title: str = ""
    rank: int = -1
    injected: bool = False


# --- corpus / query / qrels loading -------------------------------------


def _resolve_source(name: str, table: dict[str, str], what: str) -> str:
    try:
        return table[name]
    except KeyError:
        raise KeyError(
            f"unknown RAG {what} {name!r}; valid: {', '.join(sorted(table))}"
        ) from None


def load_corpus(
    name: str = "nq",
    max_docs: int | None = None,
    streaming: bool | None = None,
    **hf_kwargs: Any,
) -> list[Document]:
    """Load a retrieval corpus as a list of :class:`Document`.

    Args:
        name: Corpus name; currently only ``"nq"`` (``BeIR/nq``, config
            ``corpus``).
        max_docs: Keep only the first N passages. ``None`` loads all ~2.68M,
            which needs the full download and several GB of RAM — bound it for
            anything but a full-scale run.
        streaming: Stream from the Hub instead of materialising the dataset.
            Defaults to True whenever ``max_docs`` is set and we are not in
            offline mode, so a bounded run downloads only the shards it needs.
            Force ``False`` when working offline from an already-cached copy.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        Documents in upstream order — order is stable, which is what makes an
        index built from this list reproducible.
    """
    source = _resolve_source(name, CORPUS_SOURCES, "corpus")
    hf_load_dataset = _hf_load_dataset()
    if streaming is None:
        streaming = max_docs is not None and not settings.offline

    log.info(
        "loading corpus %s (max_docs=%s, streaming=%s)", source, max_docs, streaming
    )
    ds = hf_load_dataset(
        source, "corpus", split="corpus", streaming=streaming, **hf_kwargs
    )
    rows: Iterable[dict[str, Any]] = ds
    if max_docs is not None:
        rows = islice(iter(rows), max_docs)

    docs = [
        Document(
            doc_id=str(r["_id"]),
            text=str(r.get("text", "")),
            title=str(r.get("title") or ""),
            meta={"source": source},
        )
        for r in rows
    ]
    log.info("loaded %d documents from %s", len(docs), source)
    return docs


def load_queries(
    name: str = "nq",
    max_queries: int | None = None,
    streaming: bool | None = None,
    **hf_kwargs: Any,
) -> list[Query]:
    """Load the query set for a corpus.

    Args:
        name: Corpus name (``"nq"``).
        max_queries: Keep only the first N queries.
        streaming: See :func:`load_corpus`. The NQ query set is small (~3.4k),
            so this defaults to False.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        A list of :class:`Query`.
    """
    source = _resolve_source(name, CORPUS_SOURCES, "corpus")
    hf_load_dataset = _hf_load_dataset()
    if streaming is None:
        streaming = False

    ds = hf_load_dataset(
        source, "queries", split="queries", streaming=streaming, **hf_kwargs
    )
    rows: Iterable[dict[str, Any]] = ds
    if max_queries is not None:
        rows = islice(iter(rows), max_queries)

    queries = [
        Query(query_id=str(r["_id"]), text=str(r.get("text", "")), meta={"source": source})
        for r in rows
    ]
    log.info("loaded %d queries from %s", len(queries), source)
    return queries


def load_qrels(
    name: str = "nq",
    split: str = "test",
    **hf_kwargs: Any,
) -> dict[str, dict[str, int]]:
    """Load relevance judgements as ``{query_id: {doc_id: score}}``.

    Qrels are *not* part of ``BeIR/nq`` — that repo has only the ``corpus`` and
    ``queries`` configs. They live in the separate ``BeIR/nq-qrels`` repo.

    These are needed to measure clean retrieval quality *before* poisoning: a
    poisoned store that also degrades ordinary recall is a different (and much
    more detectable) attack than one that only hijacks the target query.

    Args:
        name: Corpus name (``"nq"``).
        split: Qrels split, typically ``"test"``.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        Nested mapping from query id to ``{doc_id: relevance}``.

    Raises:
        KeyError: If ``split`` is not published for this corpus.
    """
    source = _resolve_source(name, QRELS_SOURCES, "qrels")
    hf_load_dataset = _hf_load_dataset()
    dd = hf_load_dataset(source, **hf_kwargs)

    if hasattr(dd, "keys") and split not in dd:
        raise KeyError(
            f"qrels split {split!r} not in {source}; available: "
            f"{', '.join(sorted(dd.keys()))}"
        )
    rows = dd[split] if hasattr(dd, "keys") else dd

    cols = set(rows.column_names)
    qcol = "query-id" if "query-id" in cols else "query_id"
    dcol = "corpus-id" if "corpus-id" in cols else "corpus_id"
    if qcol not in cols or dcol not in cols:
        raise KeyError(
            f"unexpected qrels schema in {source}: {sorted(cols)}; expected "
            "query-id / corpus-id / score"
        )

    qrels: dict[str, dict[str, int]] = {}
    for r in rows:
        qrels.setdefault(str(r[qcol]), {})[str(r[dcol])] = int(r.get("score", 1))
    log.info("loaded qrels for %d queries from %s (%s)", len(qrels), source, split)
    return qrels


def load_split(split: str = "corpus", name: str = "nq", **kwargs: Any) -> Any:
    """Registry entry point for RAG datasets.

    ``load_dataset("nq", split="corpus", max_docs=50_000)`` routes here. Unlike
    the vision/text loaders, "split" selects *which table* of the retrieval
    dataset you want, not a train/test partition.

    Args:
        split: ``"corpus"``, ``"queries"`` or ``"qrels"``.
        name: Corpus name.
        **kwargs: Forwarded to the concrete loader.

    Returns:
        ``list[Document]``, ``list[Query]``, or the qrels mapping.
    """
    if split == "corpus":
        return load_corpus(name=name, **kwargs)
    if split == "queries":
        return load_queries(name=name, **kwargs)
    if split == "qrels":
        return load_qrels(name=name, **kwargs)
    raise KeyError(
        f"unknown RAG split {split!r}; valid: corpus, queries, qrels"
    )


# --- jsonl helpers ------------------------------------------------------


def documents_to_jsonl(docs: Sequence[Document], path: str | Path) -> Path:
    """Write documents to JSONL, one object per line. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_dict(), ensure_ascii=False) + "\n")
    return path


def documents_from_jsonl(path: str | Path) -> list[Document]:
    """Read documents written by :func:`documents_to_jsonl`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no document file at {path}")
    with path.open("r", encoding="utf-8") as fh:
        return [Document.from_dict(json.loads(line)) for line in fh if line.strip()]


# --- embedding ----------------------------------------------------------


def _encoder_slug(encoder: str) -> str:
    return encoder.replace("/", "__").replace(":", "_")


def _corpus_digest(doc_ids: Sequence[str]) -> str:
    """Short digest over the doc-id sequence.

    Part of the embedding cache key so that two different 50k slices of NQ
    cannot collide on ``(corpus, encoder, size)`` alone.
    """
    h = hashlib.sha256()
    for d in doc_ids:
        h.update(d.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def embedding_cache_path(
    corpus_name: str,
    encoder: str,
    doc_ids: Sequence[str],
    normalize: bool = True,
    stripped: bool = False,
) -> Path:
    """Path of the cached embedding matrix for this (corpus, encoder, size, ids).

    Args:
        corpus_name: Corpus name component of the key.
        encoder: Model id component of the key.
        doc_ids: Document ids; their count and digest are part of the key.
        normalize: Whether the cached vectors are L2-normalised. This *must* be
            part of the key: cosine and dot-product runs over the same corpus
            and encoder are otherwise indistinguishable, and a dot-product run
            would silently reuse unit-norm vectors from an earlier cosine run
            (the recorded metric would then be a lie). Only the ``False`` case
            adds a suffix, so caches written before this argument existed — all
            of which were normalised — stay valid.
        stripped: Whether the encoder had a trailing ``Normalize`` module removed
            by ``load_encoder(..., strip_normalizer=True)``. This *also* must be
            part of the key. For a checkpoint that self-normalises, a plain
            ``normalize=False`` build still writes unit-norm vectors into the
            ``__dot`` slot; a later stripped build — which does produce genuine
            dot-product vectors — would then load those contaminated unit-norm
            vectors and silently report cosine. Verified by
            ``scripts/verify_retrievers.py``. Only the ``True`` case adds a
            suffix, so existing caches stay valid.

    Returns:
        Path under ``data/rag/embeddings``.
    """
    key = (
        f"{corpus_name}__{_encoder_slug(encoder)}__n{len(doc_ids)}"
        f"__{_corpus_digest(doc_ids)}"
    )
    if not normalize:
        key += "__dot"
    if stripped:
        key += "__stripped"
    return RAG_DIR / "embeddings" / f"{key}.npy"


def _looks_stripped(model: Any) -> bool:
    """Whether ``load_encoder(..., strip_normalizer=True)`` produced this model.

    ``load_encoder`` stamps the flag on the object; absence means an ordinary
    load. Checked rather than inferred, because a checkpoint that simply never
    had a ``Normalize`` module is NOT stripped and must keep sharing the plain
    cache slot.
    """
    return bool(getattr(model, "_aif_stripped_normalizer", False))


def encoder_normalizes(model: Any) -> bool:
    """Whether a ``SentenceTransformer`` L2-normalises its own output.

    Several popular checkpoints (``all-MiniLM-L6-v2`` among them) list a
    ``Normalize`` module after pooling, which makes
    ``encode(normalize_embeddings=False)`` a no-op — the caller asks for dot
    product and gets cosine. Call this before building a dot-product index.

    Args:
        model: A loaded ``SentenceTransformer`` (or anything exposing
            ``named_children``). Non-module objects return ``False``.

    Returns:
        True if any module in the pipeline is a ``Normalize`` layer.
    """
    try:
        children = list(model.named_children())
    except AttributeError:
        return False
    return any(type(m).__name__ == "Normalize" for _, m in children)


def _is_unit_norm(emb: np.ndarray, tol: float = 1e-3, sample: int = 256) -> bool:
    """Whether ``emb``'s rows have L2 norm 1 to within ``tol``.

    Only the first ``sample`` rows are measured: normalisation is a property of
    the encoding pipeline, not of individual passages, so the first few rows
    settle it and a full pass over a 50k x 384 matrix is not worth the time.
    """
    if emb.size == 0:
        return False
    norms = np.linalg.norm(np.asarray(emb[:sample], dtype=np.float32), axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= tol))


def load_encoder(
    encoder: str = DEFAULT_ENCODER,
    device: Any | None = None,
    strip_normalizer: bool = False,
) -> Any:
    """Instantiate a ``SentenceTransformer``, on GPU 0 when CUDA is available.

    Plain HuggingFace encoders (``facebook/contriever`` and
    ``facebook/contriever-msmarco`` ship only ``config.json`` /
    ``pytorch_model.bin`` / tokenizer files — no ``modules.json``, no
    ``1_Pooling/``) take sentence-transformers' ``_load_auto_model`` fallback,
    which wraps the model as ``[Transformer, Pooling(dim, "mean")]``. That is
    verified here rather than assumed, because it is the difference between
    reproducing Contriever and silently scoring a ``[CLS]`` vector.

    The pooling is equivalent to Contriever's own. ``Contriever.forward``
    (``facebookresearch/contriever``, ``src/contriever.py``) computes
    ``last_hidden.sum(1) / attention_mask.sum(1)`` over mask-zeroed hidden
    states; ``Pooling`` in mean mode computes
    ``sum(token_embeddings * mask) / clamp(sum(mask), min=1e-9)``. Identical
    except for the clamp, which only guards an all-padding row. Neither
    normalises — ``Contriever.forward``'s ``normalize`` argument defaults to
    False — so the dot product of two such vectors is PoisonedRAG's default
    ``score_function``.

    Args:
        encoder: HuggingFace model id.
        device: Explicit torch device; ``None`` asks
            :func:`aiforensics.device.get_device`, which is the only place
            that decides GPU vs CPU in this project.
        strip_normalizer: Remove trailing ``Normalize`` modules, making
            un-normalised (dot-product) embeddings reachable for checkpoints
            that hard-wire cosine into their module list. Off by default: it
            changes what the published checkpoint computes, so turn it on only
            for a deliberately dot-product run and record it in the manifest.
            Encoders without a ``Normalize`` module are unaffected.

    Returns:
        A ``SentenceTransformer`` instance.
    """
    from ..device import get_device

    SentenceTransformer = _sentence_transformer_cls()
    dev = get_device() if device is None else device
    log.info("loading encoder %s on %s", encoder, dev)
    model = SentenceTransformer(encoder, device=str(dev))

    modules = [type(m).__name__ for _, m in model.named_children()]
    if modules[1:2] == ["Pooling"] and "Normalize" not in modules:
        pooling = list(model.children())[1]
        mode = getattr(pooling, "get_pooling_mode_str", lambda: "?")()
        if not getattr(pooling, "pooling_mode_mean_tokens", False):
            log.warning(
                "encoder %s pools with %r, not mean-over-attention-mask; if this is "
                "a Contriever-family model the embeddings will NOT match the "
                "published retriever",
                encoder, mode,
            )
        else:
            log.info("encoder %s: mean pooling, no normalizer (dot-product capable)",
                     encoder)

    if strip_normalizer:
        removed = 0
        names = [n for n, _ in model.named_children()]
        while names and type(model._modules[names[-1]]).__name__ == "Normalize":
            del model._modules[names.pop()]
            removed += 1
        if removed:
            # Mark the object so encode_documents gives it a separate cache slot:
            # the same checkpoint stripped and unstripped produces different
            # vectors, and sharing a slot silently returns the wrong ones.
            model._aif_stripped_normalizer = True
            log.warning(
                "stripped %d Normalize module(s) from %s; embeddings are now "
                "un-normalised and inner product is a true dot product",
                removed, encoder,
            )
        else:
            log.info("strip_normalizer=True but %s has no Normalize module", encoder)
    return model


def embed_texts(
    texts: Sequence[str],
    encoder: str | Any = DEFAULT_ENCODER,
    batch_size: int | None = None,
    device: Any | None = None,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a list of strings into a float32 matrix.

    Args:
        texts: Strings to embed.
        encoder: Model id, or an already-loaded ``SentenceTransformer``.
        batch_size: Encoder batch size; defaults to ``settings.batch_size``.
        device: Torch device; ``None`` uses :func:`get_device`.
        normalize: L2-normalise rows so inner product equals cosine
            similarity. Leave this True unless you know why you are not — the
            documented reason to set it False is PoisonedRAG-faithful
            dot-product retrieval (see the module docstring). Note that this is
            a *request*: a checkpoint whose module list ends in ``Normalize``
            returns unit-norm vectors either way (:func:`encoder_normalizes`).
        show_progress: Show the sentence-transformers progress bar.

    Returns:
        ``np.ndarray`` of shape ``(len(texts), dim)``, dtype float32,
        C-contiguous (faiss requires both).
    """
    model = encoder if not isinstance(encoder, str) else load_encoder(encoder, device)
    emb = model.encode(
        list(texts),
        batch_size=batch_size or settings.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress,
    )
    return np.ascontiguousarray(np.asarray(emb, dtype=np.float32))


def encode_documents(
    docs: Sequence[Document],
    encoder: str | Any = DEFAULT_ENCODER,
    corpus_name: str = "custom",
    batch_size: int | None = None,
    device: Any | None = None,
    normalize: bool = True,
    use_cache: bool = True,
    show_progress: bool = True,
    encoder_name: str | None = None,
) -> np.ndarray:
    """Embed documents, reusing a cached matrix when one exists.

    The cache key covers the corpus name, encoder, document count, a digest of
    the document ids and the ``normalize`` flag, so neither a differently-sliced
    corpus nor a cosine/dot-product switch can silently reuse the wrong vectors.

    Args:
        docs: Documents to embed (``Document.embedding_text`` is what is fed
            to the model).
        encoder: Model id or loaded ``SentenceTransformer``.
        corpus_name: Name used in the cache key.
        batch_size: Encoder batch size.
        device: Torch device.
        normalize: L2-normalise the rows.
        use_cache: Read/write ``data/rag/embeddings``.
        show_progress: Show the encoding progress bar.
        encoder_name: Override the model id used in the cache key, for
            preloaded models whose id cannot be recovered.

    Returns:
        ``(len(docs), dim)`` float32 embedding matrix.
    """
    encoder_name = encoder_name or (
        encoder if isinstance(encoder, str) else _model_name(encoder)
    )
    doc_ids = [d.doc_id for d in docs]
    # A stripped encoder produces different vectors from the same checkpoint, so
    # it needs its own cache slot (see embedding_cache_path).
    stripped = not isinstance(encoder, str) and _looks_stripped(encoder)
    cache = embedding_cache_path(corpus_name, encoder_name, doc_ids,
                                 normalize=normalize, stripped=stripped)

    if use_cache and cache.exists():
        emb = np.load(cache)
        if len(emb) != len(docs):
            log.warning("cached embeddings at %s have wrong length; re-encoding", cache)
        elif normalize and not _is_unit_norm(emb):
            # Belt and braces over the cache key: a file written before
            # `normalize` entered that key could hold un-normalised vectors, and
            # reusing them for a cosine run is invisible downstream — the scores
            # just quietly mean something else. The converse is not checked,
            # because a self-normalising encoder legitimately returns unit-norm
            # vectors under normalize=False (see `encoder_normalizes`), and
            # rejecting those would re-encode the corpus on every single run.
            log.warning(
                "cached embeddings at %s are un-normalised but normalize=True was "
                "requested; re-encoding", cache,
            )
        else:
            log.info("reusing cached embeddings %s", cache)
            return np.ascontiguousarray(emb.astype(np.float32, copy=False))

    t0 = time.perf_counter()
    emb = embed_texts(
        [d.embedding_text() for d in docs],
        encoder=encoder,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
        show_progress=show_progress,
    )
    log.info(
        "encoded %d documents in %.1fs (%s)", len(docs), time.perf_counter() - t0,
        encoder_name,
    )

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        cache.with_suffix(".json").write_text(
            json.dumps(
                {
                    "corpus": corpus_name,
                    "encoder": encoder_name,
                    "num_docs": len(docs),
                    "dim": int(emb.shape[1]),
                    "normalize_requested": normalize,
                    # Measured, not assumed: a self-normalising checkpoint
                    # ignores normalize=False.
                    "normalized": _is_unit_norm(emb),
                    "doc_id_digest": _corpus_digest(doc_ids),
                    "created_at": _utcnow(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return emb


def _model_name(model: Any) -> str:
    """Best-effort model id for an already-loaded SentenceTransformer."""
    for attr in ("model_card_data", "_model_config"):
        obj = getattr(model, attr, None)
        name = getattr(obj, "base_model", None) if obj is not None else None
        if name:
            return str(name)
    return str(getattr(model, "model_name_or_path", None) or "unknown-encoder")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- store --------------------------------------------------------------


class RagStore:
    """A FAISS index plus its documents, with injection provenance.

    The store is the unit that experiments pass around: build it once from a
    clean corpus, snapshot it, then inject adversarial passages with
    :meth:`add_documents`. Everything a detector is later evaluated against —
    the vectors, the texts and the ground-truth injected set — round-trips
    through :meth:`save` / :meth:`load`.

    Attributes:
        index: The faiss index (inner product over unit-norm vectors).
        doc_ids: Document ids in index-row order. Row ``i`` of the index is
            ``doc_ids[i]``.
        documents: ``{doc_id: Document}``.
        encoder_name: Model id used for the vectors; re-encoding a query with
            a different model would silently produce garbage rankings, so this
            is persisted and reused.
        injected_ids: Ids added post-hoc via :meth:`add_documents` — the
            ground truth for RAG-poisoning detection.
        normalized: Whether the stored vectors are unit-norm, i.e. whether the
            index's inner product is cosine similarity (True, the project
            default) or a raw dot product (False, the PoisonedRAG-faithful
            setting). :meth:`encode_queries` follows this flag, so queries are
            always scored the same way the documents were embedded. See
            :attr:`similarity` and the module docstring.
    """

    def __init__(
        self,
        index: Any,
        doc_ids: Sequence[str],
        documents: Sequence[Document] | dict[str, Document],
        encoder_name: str = DEFAULT_ENCODER,
        corpus_name: str = "custom",
        normalized: bool = True,
        metric: str = "ip",
        injected_ids: Sequence[str] | None = None,
        encoder: Any | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.doc_ids: list[str] = [str(d) for d in doc_ids]
        if isinstance(documents, dict):
            self.documents: dict[str, Document] = dict(documents)
        else:
            self.documents = {d.doc_id: d for d in documents}
        self.encoder_name = encoder_name
        self.corpus_name = corpus_name
        self.normalized = normalized
        self.metric = metric
        self.injected_ids: list[str] = sorted(set(injected_ids or ()))
        self.meta: dict[str, Any] = dict(meta or {})
        self._encoder = encoder

        if len(self.doc_ids) != len(self.documents):
            raise ValueError(
                f"doc_ids/documents mismatch: {len(self.doc_ids)} ids vs "
                f"{len(self.documents)} documents (duplicate doc_id?)"
            )

    # -- basics ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"RagStore(corpus={self.corpus_name!r}, n={len(self)}, "
            f"dim={self.dim}, encoder={self.encoder_name!r}, "
            f"injected={len(self.injected_ids)})"
        )

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self.index.d)

    @property
    def similarity(self) -> str:
        """``"cosine"`` or ``"dot"`` — what this store's scores mean.

        The index is ``IndexFlatIP`` either way; only :attr:`normalized`
        distinguishes them. Put this in the run manifest: a detector's
        threshold and a retrieval score are not comparable across the two.
        """
        return "cosine" if self.normalized else "dot"

    @property
    def encoder(self) -> Any:
        """The query encoder, loaded lazily on first search."""
        if self._encoder is None:
            self._encoder = load_encoder(self.encoder_name)
        return self._encoder

    def get(self, doc_id: str) -> Document:
        """Return a document by id."""
        try:
            return self.documents[doc_id]
        except KeyError:
            raise KeyError(f"document {doc_id!r} is not in this store") from None

    # -- ground truth ----------------------------------------------------

    def is_injected(self, doc_id: str) -> bool:
        """Whether ``doc_id`` was injected after the store was built."""
        return doc_id in set(self.injected_ids)

    def injected_mask(self) -> np.ndarray:
        """Ground-truth boolean mask over :attr:`doc_ids`, in index order.

        Returns:
            ``np.ndarray`` of dtype ``bool`` and length ``len(self)``; entry
            ``i`` is True when ``doc_ids[i]`` was injected. This is the RAG
            analogue of ``PoisonManifest.is_poisoned`` and is what
            ``aiforensics.metrics`` scores detector output against.
        """
        injected = set(self.injected_ids)
        return np.fromiter(
            (d in injected for d in self.doc_ids), dtype=bool, count=len(self.doc_ids)
        )

    # -- search ----------------------------------------------------------

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Embed queries with the store's encoder and normalisation setting.

        ``normalize=self.normalized`` is load-bearing, not incidental: the query
        side must use exactly the normalisation the index was built with, or the
        inner product is neither cosine nor a dot product but a mixture of the
        two, with a query-length factor that silently reorders results. The same
        flag is used by :meth:`add_documents`, so injected passages land in the
        same space as the corpus.
        """
        return embed_texts(
            queries, encoder=self.encoder, normalize=self.normalized
        )

    def search(
        self, query: str | Sequence[str], k: int = 5
    ) -> list[SearchResult] | list[list[SearchResult]]:
        """Retrieve the ``k`` nearest documents.

        Args:
            query: A single query string, or a list of them.
            k: Number of hits per query.

        Returns:
            For a single query, a list of :class:`SearchResult` ordered by
            descending score. For a list of queries, a list of such lists in
            input order.
        """
        single = isinstance(query, str)
        queries = [query] if single else list(query)
        if not queries:
            return [] if single else []

        k = max(1, min(int(k), len(self)))
        emb = self.encode_queries(queries)
        scores, idx = self.index.search(emb, k)

        out: list[list[SearchResult]] = []
        injected = set(self.injected_ids)
        for row_scores, row_idx in zip(scores, idx):
            hits: list[SearchResult] = []
            for rank, (s, i) in enumerate(zip(row_scores, row_idx)):
                if i < 0:  # faiss pads with -1 when fewer than k results exist
                    continue
                doc_id = self.doc_ids[int(i)]
                doc = self.documents[doc_id]
                hits.append(
                    SearchResult(
                        doc_id=doc_id,
                        score=float(s),
                        text=doc.text,
                        title=doc.title,
                        rank=rank,
                        injected=doc_id in injected,
                    )
                )
            out.append(hits)
        return out[0] if single else out

    # -- mutation --------------------------------------------------------

    def add_documents(
        self,
        docs: Sequence[Document | dict[str, Any]],
        injected: bool = True,
        batch_size: int | None = None,
    ) -> list[str]:
        """Append documents to the index, recording them as injected.

        This is the operation a RAG-poisoning experiment performs: take a
        clean store and add adversarial passages (typically from
        :func:`aiforensics.data.poison.craft_poison_passages`). The new ids are
        appended to :attr:`injected_ids`, so the ground-truth mask stays exact
        no matter how many rounds of injection happen.

        Args:
            docs: Documents or dicts (``doc_id``/``_id``, ``text``, ``title``).
            injected: Record these as attacker-supplied. Pass ``False`` only
                when legitimately growing a clean corpus — mislabelling here
                corrupts every downstream detection number.
            batch_size: Encoder batch size.

        Returns:
            The ids added, in insertion order.

        Raises:
            ValueError: If a doc_id already exists in the store. Ids must stay
                unique or the index rows and the ground-truth mask diverge.
        """
        items = [d if isinstance(d, Document) else Document.from_dict(d) for d in docs]
        if not items:
            return []

        clashes = [d.doc_id for d in items if d.doc_id in self.documents]
        if clashes:
            raise ValueError(
                f"{len(clashes)} document id(s) already in the store, e.g. "
                f"{clashes[:3]}; ids must be unique"
            )
        seen: set[str] = set()
        for d in items:
            if d.doc_id in seen:
                raise ValueError(f"duplicate doc_id {d.doc_id!r} within this batch")
            seen.add(d.doc_id)

        emb = embed_texts(
            [d.embedding_text() for d in items],
            encoder=self.encoder,
            batch_size=batch_size,
            normalize=self.normalized,
        )
        if emb.shape[1] != self.dim:
            raise ValueError(
                f"embedding dim {emb.shape[1]} != index dim {self.dim}; the "
                "store was built with a different encoder"
            )
        self.index.add(emb)

        new_ids = []
        for d in items:
            self.doc_ids.append(d.doc_id)
            self.documents[d.doc_id] = d
            new_ids.append(d.doc_id)
        if injected:
            self.injected_ids = sorted(set(self.injected_ids) | set(new_ids))

        log.info(
            "added %d documents to store %r (injected=%s, total=%d)",
            len(new_ids), self.corpus_name, injected, len(self),
        )
        return new_ids

    # -- persistence -----------------------------------------------------

    @staticmethod
    def _resolve_dir(path: str | Path) -> Path:
        """Bare names resolve under ``data/rag``; explicit paths are honoured."""
        p = Path(path)
        return p if p.is_absolute() or len(p.parts) > 1 else RAG_DIR / p

    def save(self, path: str | Path) -> Path:
        """Persist the store to a directory.

        Writes ``index.faiss``, ``documents.jsonl`` and ``meta.json``. A bare
        name (``"nq-50k"``) lands under ``data/rag/``.

        Args:
            path: Destination directory or bare store name.

        Returns:
            The directory written to.
        """
        faiss = _faiss()
        out = self._resolve_dir(path)
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(out / _INDEX_FILE))
        documents_to_jsonl([self.documents[i] for i in self.doc_ids], out / _DOCS_FILE)
        (out / _META_FILE).write_text(
            json.dumps(
                {
                    "schema_version": _STORE_SCHEMA_VERSION,
                    "corpus_name": self.corpus_name,
                    "encoder_name": self.encoder_name,
                    "dim": self.dim,
                    "metric": self.metric,
                    "normalized": self.normalized,
                    "similarity": self.similarity,
                    "num_documents": len(self),
                    "num_injected": len(self.injected_ids),
                    "injected_ids": self.injected_ids,
                    "created_at": _utcnow(),
                    "meta": self.meta,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("saved RAG store (%d docs) to %s", len(self), out)
        return out

    @classmethod
    def load(cls, path: str | Path, encoder: Any | None = None) -> "RagStore":
        """Load a store written by :meth:`save`.

        Args:
            path: Directory or bare store name under ``data/rag``.
            encoder: Optional preloaded ``SentenceTransformer``; otherwise the
                encoder recorded in ``meta.json`` is loaded on first search.

        Returns:
            The reconstructed :class:`RagStore`.

        Raises:
            FileNotFoundError: With an actionable message if the store or any
                of its three files is missing.
        """
        faiss = _faiss()
        src = cls._resolve_dir(path)
        if not src.is_dir():
            raise FileNotFoundError(
                f"no RAG store at {src}; build one with "
                "`python scripts/build_rag_index.py`"
            )
        for fname in (_INDEX_FILE, _DOCS_FILE, _META_FILE):
            if not (src / fname).exists():
                raise FileNotFoundError(f"RAG store at {src} is missing {fname}")

        meta = json.loads((src / _META_FILE).read_text(encoding="utf-8"))
        docs = documents_from_jsonl(src / _DOCS_FILE)
        index = faiss.read_index(str(src / _INDEX_FILE))

        if index.ntotal != len(docs):
            raise ValueError(
                f"corrupt store at {src}: index has {index.ntotal} vectors but "
                f"{len(docs)} documents"
            )
        return cls(
            index=index,
            doc_ids=[d.doc_id for d in docs],
            documents=docs,
            encoder_name=meta.get("encoder_name", DEFAULT_ENCODER),
            corpus_name=meta.get("corpus_name", "custom"),
            normalized=bool(meta.get("normalized", True)),
            metric=meta.get("metric", "ip"),
            injected_ids=meta.get("injected_ids", []),
            encoder=encoder,
            meta=meta.get("meta", {}),
        )


def build_index(
    corpus: Sequence[Document] | Sequence[dict[str, Any]],
    encoder: str | Any = DEFAULT_ENCODER,
    corpus_name: str = "custom",
    batch_size: int | None = None,
    device: Any | None = None,
    normalize: bool = True,
    use_cache: bool = True,
    index_factory: str | None = None,
    show_progress: bool = True,
    encoder_name: str | None = None,
) -> RagStore:
    """Embed a corpus and build a FAISS index over it.

    Args:
        corpus: Documents (or dicts convertible to :class:`Document`).
        encoder: Model id or a loaded ``SentenceTransformer``.
        encoder_name: Model id to record in the store. Only needed when
            ``encoder`` is a preloaded model whose id cannot be recovered — the
            store must know which model produced its vectors, because queries
            embedded by a different model give silently meaningless rankings.
        corpus_name: Name recorded in the store and in the embedding cache key.
        batch_size: Encoder batch size; defaults to ``settings.batch_size``.
        device: Torch device; ``None`` uses :func:`get_device` (GPU 0 when
            CUDA is present).
        normalize: L2-normalise vectors so inner product is cosine similarity
            (the default). ``False`` keeps the IP index but leaves vectors
            un-normalised, making the score a raw dot product — length- and
            repetition-sensitive, and exactly what PoisonedRAG's retrievers do
            (see the module docstring). Whether the request took effect is
            *measured* after encoding and recorded on the store, because a
            checkpoint ending in a ``Normalize`` module ignores it.
        use_cache: Reuse/write cached embeddings under ``data/rag/embeddings``.
        index_factory: A faiss ``index_factory`` string for large corpora,
            e.g. ``"IVF4096,Flat"``. Default ``None`` builds an exact
            ``IndexFlatIP``; see the module docstring for why exactness is the
            default here.
        show_progress: Show the encoding progress bar.

    Returns:
        A ready-to-search :class:`RagStore` with no injected documents yet.
    """
    # Validate the corpus before importing faiss or touching the GPU, so a
    # bad call fails on the real problem rather than on a missing dependency.
    docs = [d if isinstance(d, Document) else Document.from_dict(d) for d in corpus]
    if not docs:
        raise ValueError("cannot build an index over an empty corpus")

    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError(
            "corpus contains duplicate doc_ids; ids must be unique or index "
            "rows and ground-truth masks will diverge"
        )

    faiss = _faiss()

    emb = encode_documents(
        docs,
        encoder=encoder,
        corpus_name=corpus_name,
        batch_size=batch_size,
        device=device,
        normalize=normalize,
        use_cache=use_cache,
        show_progress=show_progress,
        encoder_name=encoder_name,
    )
    dim = int(emb.shape[1])

    # What we asked for is not necessarily what we got: a SentenceTransformer
    # whose module list ends in `Normalize` returns unit-norm vectors whatever
    # `normalize_embeddings` says. Record the measured fact, so the store, its
    # meta.json and any run manifest describe the space the index is actually
    # in rather than the space that was requested.
    actually_normalized = _is_unit_norm(emb)
    if normalize and not actually_normalized:
        log.warning(
            "normalize=True was requested but the encoded vectors are not "
            "unit-norm; the index scores are dot products, not cosine"
        )
    elif not normalize and actually_normalized:
        log.warning(
            "normalize=False was requested but %s returns unit-norm vectors "
            "anyway (a `Normalize` module in the checkpoint), so this index is "
            "cosine, not dot product. For PoisonedRAG-faithful dot-product "
            "retrieval use an encoder without one (e.g. facebook/contriever) or "
            "load it with load_encoder(..., strip_normalizer=True).",
            encoder if isinstance(encoder, str) else _model_name(encoder),
        )

    if index_factory:
        index = faiss.index_factory(dim, index_factory, faiss.METRIC_INNER_PRODUCT)
        if not index.is_trained:
            log.info("training %s index on %d vectors", index_factory, len(emb))
            index.train(emb)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(emb)

    resolved_name = encoder_name or (
        encoder if isinstance(encoder, str) else _model_name(encoder)
    )
    if resolved_name == "unknown-encoder":
        log.warning(
            "could not determine the encoder id for a preloaded model; the "
            "saved store will not be reloadable for search. Pass "
            "encoder_name=... to build_index()."
        )
    log.info(
        "built %s index: %d vectors, dim=%d, encoder=%s, similarity=%s",
        index_factory or "IndexFlatIP", index.ntotal, dim, resolved_name,
        "cosine" if actually_normalized else "dot",
    )
    return RagStore(
        index=index,
        doc_ids=ids,
        documents=docs,
        encoder_name=resolved_name,
        corpus_name=corpus_name,
        normalized=actually_normalized,
        metric="ip",
        encoder=None if isinstance(encoder, str) else encoder,
        meta={
            "index_factory": index_factory or "IndexFlatIP",
            "normalize_requested": normalize,
        },
    )


__all__ = [
    "CORPUS_SOURCES",
    "DEFAULT_ENCODER",
    "DEFAULT_MAX_DOCS",
    "Document",
    "QRELS_SOURCES",
    "Query",
    "RagStore",
    "SearchResult",
    "build_index",
    "documents_from_jsonl",
    "documents_to_jsonl",
    "embed_texts",
    "embedding_cache_path",
    "encode_documents",
    "encoder_normalizes",
    "load_corpus",
    "load_encoder",
    "load_qrels",
    "load_queries",
    "load_split",
]
