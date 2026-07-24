#!/usr/bin/env python3
"""Pre-download every registered dataset into the shared HuggingFace cache.

Run this once after building the image so that later experiments never block
on the network — and so that an offline run (``HF_HUB_OFFLINE=1``) works.

Examples::

    python scripts/fetch_datasets.py --list
    python scripts/fetch_datasets.py                       # everything
    python scripts/fetch_datasets.py --modality text
    python scripts/fetch_datasets.py --datasets cifar10 sst2
    python scripts/fetch_datasets.py --dry-run
    python scripts/fetch_datasets.py --datasets nq --max-docs 200000

The BeIR/NQ corpus is 2.68M passages, so by default only the first
``--max-docs`` are pulled (streamed) and materialised as a JSONL slice under
``data/rag/corpus/``. Pass ``--full-corpus`` to download all of it into the HF
cache instead; expect several GB and a long wait.

Exits non-zero if any dataset fails, so it is safe to use in a Makefile.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow `python scripts/fetch_datasets.py` without PYTHONPATH being set.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aiforensics.data.registry import (  # noqa: E402
    describe_datasets,
    get_spec,
    list_datasets,
)
from aiforensics.paths import HF_CACHE, RAG_DIR, RAW_DIR, ensure_dirs  # noqa: E402

log = logging.getLogger("fetch_datasets")


def _console():
    try:
        from rich.console import Console
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "rich is required by this script; it is pinned in "
            "docker/requirements.txt — run inside the container"
        ) from exc
    return Console()


# --- cache inspection ---------------------------------------------------


def repo_cache_dirs(source: str) -> list[Path]:
    """Candidate on-disk locations for a dataset's cached files."""
    if source.startswith("sklearn:"):
        return [RAW_DIR / "lfw"]
    slug = "datasets--" + source.replace("/", "--")
    return [HF_CACHE / "hub" / slug, HF_CACHE / "datasets" / slug]


def dir_size(path: Path) -> int:
    """Bytes on disk under ``path``.

    Symlinks are skipped: the HF cache links ``snapshots/`` entries at
    ``blobs/``, so following them would double-count every file.
    """
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def human_size(n: int) -> str:
    """Format a byte count."""
    if n <= 0:
        return "-"
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} TB"  # pragma: no cover - unreachable


def cache_state(source: str) -> tuple[bool, int]:
    """Return ``(is_cached, bytes_on_disk)`` for a dataset source."""
    total = sum(dir_size(d) for d in repo_cache_dirs(source))
    return total > 0, total


# --- per-modality fetchers ----------------------------------------------


def fetch_hf(key: str) -> dict[str, int]:
    """Download every split of a vision/text dataset. Returns rows per split."""
    from datasets import load_dataset as hf_load_dataset

    spec = get_spec(key)
    dd = hf_load_dataset(*spec.hf_args())
    if hasattr(dd, "items"):
        return {name: len(split) for name, split in dd.items()}
    return {spec.default_split: len(dd)}


def fetch_lfw(key: str) -> dict[str, int]:
    """Download LFW through scikit-learn into ``data/raw/lfw``."""
    from aiforensics.data.vision import load_lfw

    ds = load_lfw(split="all")
    return {"all": len(ds), "identities": len(ds.classes or ())}


def fetch_rag(key: str, max_docs: int | None, full_corpus: bool) -> dict[str, int]:
    """Fetch the NQ queries, qrels and a bounded corpus slice."""
    from aiforensics.data import rag

    counts: dict[str, int] = {}

    queries = rag.load_queries(name=key)
    counts["queries"] = len(queries)

    try:
        qrels = rag.load_qrels(name=key)
        counts["qrels"] = len(qrels)
    except Exception as exc:
        log.warning("could not fetch qrels for %s: %s", key, exc)
        counts["qrels"] = 0

    if full_corpus:
        docs = rag.load_corpus(name=key, max_docs=None, streaming=False)
    else:
        docs = rag.load_corpus(name=key, max_docs=max_docs, streaming=True)
        out = RAG_DIR / "corpus" / f"{key}-{len(docs)}.jsonl"
        rag.documents_to_jsonl(docs, out)
        log.info("wrote corpus slice to %s", out)
    counts["corpus"] = len(docs)
    return counts


def fetch_one(key: str, max_docs: int | None, full_corpus: bool) -> dict[str, int]:
    """Dispatch to the right fetcher for ``key``."""
    spec = get_spec(key)
    if spec.modality == "rag":
        return fetch_rag(key, max_docs, full_corpus)
    if spec.source.startswith("sklearn:"):
        return fetch_lfw(key)
    return fetch_hf(key)


# --- CLI ----------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-download registered datasets into the shared HF cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets", nargs="+", metavar="KEY", default=None,
        help="Dataset keys to fetch (default: all registered).",
    )
    p.add_argument(
        "--modality", choices=("vision", "text", "rag"), default=None,
        help="Only fetch datasets of this modality.",
    )
    p.add_argument(
        "--max-docs", type=int, default=None,
        help="Cap on BeIR/NQ corpus passages "
             "(default: aiforensics.data.rag.DEFAULT_MAX_DOCS).",
    )
    p.add_argument(
        "--full-corpus", action="store_true",
        help="Download the entire BeIR/NQ corpus (~2.68M passages, several GB).",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print the registry and exit.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be fetched without downloading anything.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-fetch even when the dataset is already cached.",
    )
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

    if args.list:
        console.print(describe_datasets(args.modality))
        return 0

    ensure_dirs()
    from aiforensics.data.rag import DEFAULT_MAX_DOCS

    max_docs = DEFAULT_MAX_DOCS if args.max_docs is None else args.max_docs

    keys = args.datasets if args.datasets else list_datasets(args.modality)
    # Validate before any get_spec() call, so a typo gets the friendly message
    # rather than a bare KeyError traceback.
    unknown = [k for k in keys if k not in list_datasets()]
    if unknown:
        console.print(
            f"[red]unknown dataset key(s): {', '.join(unknown)}[/red]\n"
            f"valid keys: {', '.join(list_datasets())}"
        )
        return 2
    if args.datasets and args.modality:
        keys = [k for k in keys if get_spec(k).modality == args.modality]

    console.print(
        f"[bold]HF cache[/bold]: {HF_CACHE}   "
        f"[bold]datasets[/bold]: {', '.join(keys)}"
    )

    rows: list[tuple[str, ...]] = []
    failures: list[tuple[str, str]] = []

    for key in keys:
        spec = get_spec(key)
        cached, size_before = cache_state(spec.source)
        t0 = time.perf_counter()

        if args.dry_run:
            rows.append((
                key, spec.modality, spec.source,
                "-", human_size(size_before),
                "cached" if cached else "would fetch", "-",
            ))
            continue

        if cached and not args.force:
            rows.append((
                key, spec.modality, spec.source,
                "-", human_size(size_before), "cached (skipped)", "-",
            ))
            log.info("%s already cached at %s", key, repo_cache_dirs(spec.source)[0])
            continue

        try:
            counts = fetch_one(key, max_docs, args.full_corpus)
            _, size_after = cache_state(spec.source)
            rows.append((
                key, spec.modality, spec.source,
                ", ".join(f"{k}={v:,}" for k, v in counts.items()),
                human_size(size_after),
                "ok",
                f"{time.perf_counter() - t0:.1f}s",
            ))
        except Exception as exc:
            log.exception("failed to fetch %s", key)
            failures.append((key, f"{type(exc).__name__}: {exc}"))
            rows.append((
                key, spec.modality, spec.source, "-",
                human_size(size_before), "FAILED",
                f"{time.perf_counter() - t0:.1f}s",
            ))

    table = Table(title="dataset fetch summary", header_style="bold")
    for col in ("key", "modality", "source", "rows", "on disk", "status", "time"):
        table.add_column(col, overflow="fold")
    for row in rows:
        style = "red" if row[5] == "FAILED" else None
        table.add_row(*row, style=style)
    console.print(table)

    if failures:
        console.print("[red bold]failures:[/red bold]")
        for key, msg in failures:
            console.print(f"  [red]{key}[/red]: {msg}")
        return 1

    console.print("[green]all requested datasets are available[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
