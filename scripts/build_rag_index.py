#!/usr/bin/env python3
"""Build and persist a FAISS index over a retrieval corpus.

The resulting :class:`~aiforensics.data.rag.RagStore` is the clean baseline a
RAG-poisoning experiment starts from: build it, snapshot it, then inject
adversarial passages with ``store.add_documents(...)``, which records the
injected ids as ground truth.

Encoding runs on GPU 0 when CUDA is visible (via ``aiforensics.device
.get_device``) and falls back to CPU otherwise.

Examples::

    python scripts/build_rag_index.py                        # 50k NQ passages
    python scripts/build_rag_index.py --max-docs 200000 --out nq-200k
    python scripts/build_rag_index.py --corpus-file data/rag/corpus/nq-50000.jsonl
    python scripts/build_rag_index.py --max-docs 0 --index-factory IVF4096,Flat

Scaling up: ``--max-docs 0`` means the full BeIR/NQ corpus (~2.68M passages).
At 384 dimensions that is ~4 GB of float32 vectors, which a flat index holds
fine in RAM on a well-provisioned host, but encoding takes on the order of an
hour on a single GPU. The
embeddings are cached under ``data/rag/embeddings`` and reused, so you pay
that once. Beyond a few million vectors, pass ``--index-factory`` to switch to
an approximate index — and record that in the run manifest, because ANN recall
loss is indistinguishable from a failed poisoning attack.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/build_rag_index.py` without PYTHONPATH being set.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aiforensics.config import settings
from aiforensics.data import rag
from aiforensics.device import get_device, gpu_info
from aiforensics.paths import RAG_DIR, ensure_dirs
from aiforensics.seed import set_seed

log = logging.getLogger("build_rag_index")


def _console():
    try:
        from rich.console import Console
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "rich is required by this script; it is pinned in "
            "docker/requirements.txt — run inside the container"
        ) from exc
    return Console()


def human_size(n: int) -> str:
    """Format a byte count."""
    if n <= 0:
        return "-"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"  # pragma: no cover - unreachable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a FAISS index over a retrieval corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus", default="nq",
        choices=sorted(rag.CORPUS_SOURCES),
        help="Registered corpus to index.",
    )
    p.add_argument(
        "--corpus-file", type=Path, default=None,
        help="Read documents from a JSONL slice instead of the Hub "
             "(as written by scripts/fetch_datasets.py).",
    )
    p.add_argument(
        "--encoder", default=rag.DEFAULT_ENCODER,
        help="sentence-transformers model id used for both documents and queries.",
    )
    p.add_argument(
        "--max-docs", type=int, default=rag.DEFAULT_MAX_DOCS,
        help="Passages to index; 0 means the entire corpus.",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Encoder batch size (default: settings.batch_size).",
    )
    p.add_argument(
        "--out", default=None,
        help="Output store name (under data/rag/) or path "
             "(default: <corpus>-<n>).",
    )
    p.add_argument(
        "--index-factory", default=None,
        help="faiss index_factory string, e.g. 'IVF4096,Flat'. "
             "Default builds an exact IndexFlatIP.",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Ignore (and do not write) cached embeddings.",
    )
    p.add_argument(
        "--query", default=None,
        help="Optional smoke-test query to run against the finished index.",
    )
    p.add_argument("--top-k", type=int, default=5, help="Hits for --query.")
    p.add_argument("--seed", type=int, default=None, help="Seed (default: settings.seed).")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    console = _console()
    from rich.table import Table

    ensure_dirs()
    set_seed(settings.seed if args.seed is None else args.seed)

    device = get_device()
    info = gpu_info() if device.type == "cuda" else None
    console.print(
        f"[bold]encoder[/bold] {args.encoder}  "
        f"[bold]device[/bold] {device}"
        + (f"  ({info.name}, {info.free_gb:.1f} GiB free)" if info else " (no CUDA)")
    )

    max_docs = None if args.max_docs in (0, None) else args.max_docs

    # --- load ----------------------------------------------------------
    t_load = time.perf_counter()
    if args.corpus_file:
        if not args.corpus_file.exists():
            console.print(
                f"[red]no corpus file at {args.corpus_file}[/red]\n"
                "run `python scripts/fetch_datasets.py --datasets "
                f"{args.corpus}` first"
            )
            return 2
        docs = rag.documents_from_jsonl(args.corpus_file)
        if max_docs is not None and max_docs < len(docs):
            docs = docs[:max_docs]
    else:
        docs = rag.load_corpus(name=args.corpus, max_docs=max_docs)
    load_secs = time.perf_counter() - t_load

    if not docs:
        console.print("[red]corpus is empty; nothing to index[/red]")
        return 1

    # --- build ---------------------------------------------------------
    out_name = args.out or f"{args.corpus}-{len(docs)}"
    t_build = time.perf_counter()
    try:
        store = rag.build_index(
            docs,
            encoder=args.encoder,
            corpus_name=args.corpus,
            batch_size=args.batch_size,
            device=device,
            normalize=True,
            use_cache=not args.no_cache,
            index_factory=args.index_factory,
            show_progress=True,
        )
    except Exception as exc:
        log.exception("index build failed")
        console.print(f"[red]index build failed: {type(exc).__name__}: {exc}[/red]")
        return 1
    build_secs = time.perf_counter() - t_build

    # --- persist -------------------------------------------------------
    t_save = time.perf_counter()
    out_dir = store.save(out_name)
    save_secs = time.perf_counter() - t_save
    index_bytes = (out_dir / "index.faiss").stat().st_size
    store_bytes = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())

    table = Table(title="RAG index build", header_style="bold")
    table.add_column("metric")
    table.add_column("value", overflow="fold")
    for label, value in (
        ("corpus", f"{args.corpus} ({rag.CORPUS_SOURCES[args.corpus]})"),
        ("encoder", args.encoder),
        ("device", str(device)),
        ("documents", f"{len(docs):,}"),
        ("vectors", f"{store.index.ntotal:,}"),
        ("dimension", str(store.dim)),
        ("metric", "inner product on L2-normalised vectors (= cosine)"),
        ("index type", store.meta.get("index_factory", "IndexFlatIP")),
        ("load time", f"{load_secs:.1f}s"),
        ("build time", f"{build_secs:.1f}s"),
        ("save time", f"{save_secs:.1f}s"),
        ("index size", human_size(index_bytes)),
        ("store size", human_size(store_bytes)),
        ("path", str(out_dir)),
    ):
        table.add_row(label, value)
    console.print(table)

    if store.index.ntotal != len(docs):
        console.print(
            f"[red]vector count {store.index.ntotal} != document count "
            f"{len(docs)}[/red]"
        )
        return 1

    # --- optional smoke test -------------------------------------------
    if args.query:
        hits = store.search(args.query, k=args.top_k)
        hit_table = Table(title=f"top-{args.top_k} for {args.query!r}", header_style="bold")
        for col in ("rank", "score", "doc_id", "text"):
            hit_table.add_column(col, overflow="fold")
        for h in hits:
            hit_table.add_row(
                str(h.rank), f"{h.score:.4f}", h.doc_id, h.text[:160].replace("\n", " ")
            )
        console.print(hit_table)

    handle = out_dir.name if out_dir.parent == RAG_DIR else str(out_dir)
    console.print(f"[green]store ready:[/green] RagStore.load({handle!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
