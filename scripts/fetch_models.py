#!/usr/bin/env python3
"""Pre-download every pinned model weight into the shared caches.

Run this once after building the container so that later experiments work with
``HF_HUB_OFFLINE=1`` and never surprise a run with a 16 GB download halfway
through. Anything already in the cache is reused — the host HuggingFace cache is
bind-mounted, so most of this is usually a no-op.

Examples::

    python scripts/fetch_models.py --list
    python scripts/fetch_models.py                       # everything
    python scripts/fetch_models.py --models qwen3-1.7b resnet18
    python scripts/fetch_models.py --dry-run             # show what is missing

Exits non-zero if any requested model could not be fetched.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow `python scripts/fetch_models.py` without PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aiforensics.config import settings
from aiforensics.models.registry import get_spec, list_models
from aiforensics.paths import HF_CACHE

logger = logging.getLogger("fetch_models")

#: Formats we never need; skipping them saves several GB per repo.
HF_IGNORE_PATTERNS = ["*.msgpack", "*.h5", "*.onnx", "*.onnx_data", "*.tflite", "original/*"]


@dataclass
class Result:
    """Outcome for one model."""

    key: str
    family: str
    source: str
    status: str  # ok | cached | missing | failed
    size_bytes: int = 0
    location: str = ""
    message: str = ""

    @property
    def failed(self) -> bool:
        return self.status in {"failed", "missing"}


def human_size(num_bytes: float) -> str:
    """Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024.0 or unit == "TB":
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"  # pragma: no cover - unreachable


def dir_size(path: Path) -> int:
    """Total size of every file under ``path`` (symlinks resolved to blobs)."""
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:  # pragma: no cover - races / dangling links
            continue
    return total


# --------------------------------------------------------------------------
# LLMs (HuggingFace hub)
# --------------------------------------------------------------------------
def _hf_local_snapshot(repo_id: str) -> Path | None:
    """Return the cached snapshot dir for ``repo_id``, or ``None`` if incomplete."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, local_files_only=True))
    except Exception:
        return None


def fetch_llm(key: str, source: str, *, force: bool, dry_run: bool) -> Result:
    """Download a HuggingFace repo snapshot, unless it is already cached."""
    cached = None if force else _hf_local_snapshot(source)
    if cached is not None:
        return Result(key, "llm", source, "cached", dir_size(cached), str(cached),
                      "already in the shared HF cache")
    if dry_run:
        return Result(key, "llm", source, "missing", 0, str(HF_CACHE), "would download")

    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            source,
            force_download=force,
            ignore_patterns=HF_IGNORE_PATTERNS,
            token=settings.hf_token,
        )
    )
    return Result(key, "llm", source, "ok", dir_size(path), str(path), "downloaded")


# --------------------------------------------------------------------------
# Vision (torch hub checkpoint cache)
# --------------------------------------------------------------------------
def _torchvision_cache_file(url: str) -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]


def fetch_vision(key: str, source: str, *, force: bool, dry_run: bool) -> Result:
    """Download pinned torchvision weights into the torch hub cache."""
    from aiforensics.models.vision import resnet_weights

    weights = resnet_weights(key)
    target = _torchvision_cache_file(weights.url)

    if target.is_file() and not force:
        return Result(key, "vision", source, "cached", target.stat().st_size, str(target),
                      "already in the torch hub cache")
    if dry_run:
        return Result(key, "vision", source, "missing", 0, str(target), "would download")

    if force and target.is_file():
        target.unlink()
    weights.get_state_dict(progress=True)  # writes into the torch hub cache
    size = target.stat().st_size if target.is_file() else 0
    return Result(key, "vision", source, "ok", size, str(target), "downloaded")


def fetch_one(key: str, *, force: bool, dry_run: bool) -> Result:
    """Fetch a single registry key, converting any failure into a ``Result``."""
    spec = get_spec(key)
    try:
        if spec.family == "llm":
            return fetch_llm(spec.key, spec.source, force=force, dry_run=dry_run)
        return fetch_vision(spec.key, spec.source, force=force, dry_run=dry_run)
    except Exception as exc:
        logger.debug("fetch failed for %s", key, exc_info=True)
        return Result(spec.key, spec.family, spec.source, "failed", message=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def _console():
    try:
        from rich.console import Console

        return Console()
    except ImportError:  # pragma: no cover - rich is pinned
        return None


def print_registry() -> None:
    """``--list``: show every key with its source, size and description."""
    rows = [get_spec(k) for k in list_models()]
    console = _console()
    if console is None:
        for spec in rows:
            print(f"{spec.key:<12} {spec.family:<7} {spec.source:<45} ~{spec.size_gb:>5.2f} GB  {spec.description}")
        return
    from rich.table import Table

    table = Table(title="aiforensics model registry", header_style="bold")
    for col in ("key", "family", "source", "approx size", "description"):
        table.add_column(col, overflow="fold")
    for spec in rows:
        table.add_row(spec.key, spec.family, spec.source, f"{spec.size_gb:.2f} GB", spec.description)
    console.print(table)


def print_results(results: list[Result], *, dry_run: bool) -> None:
    """Summary table plus the total on-disk footprint."""
    total = sum(r.size_bytes for r in results)
    console = _console()
    styles = {"ok": "green", "cached": "cyan", "missing": "yellow", "failed": "red"}

    if console is None:
        for r in results:
            print(f"{r.key:<12} {r.status:<8} {human_size(r.size_bytes):>10}  {r.location}  {r.message}")
        print(f"total on disk: {human_size(total)}")
        return

    from rich.table import Table

    title = "fetch-models (dry run)" if dry_run else "fetch-models"
    table = Table(title=title, header_style="bold")
    for col in ("key", "family", "status", "size", "location", "note"):
        table.add_column(col, overflow="fold")
    for r in results:
        table.add_row(
            r.key, r.family,
            f"[{styles.get(r.status, 'white')}]{r.status}[/]",
            human_size(r.size_bytes) if r.size_bytes else "-",
            r.location, r.message,
        )
    console.print(table)
    console.print(f"HF cache: {HF_CACHE}   total on disk: [bold]{human_size(total)}[/]")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_models.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sizes: resnet18/34/50 are under 100 MB each; Qwen3-1.7B is ~3.4 GB; "
            "NOTE: Qwen3-8B is ~16 GB - make sure the disk and the network can take it. "
            "Everything lands in the shared HuggingFace / torch hub caches, never in the "
            "project tree, and already-present files are skipped."
        ),
    )
    parser.add_argument(
        "--models", nargs="+", metavar="KEY", default=None,
        help=f"registry keys to fetch (default: all). Valid: {', '.join(list_models())}",
    )
    parser.add_argument("--list", action="store_true", help="print the registry and exit")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--dry-run", action="store_true", help="report what is missing, download nothing")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list:
        print_registry()
        return 0

    keys = args.models or list_models()
    unknown = [k for k in keys if k.strip().lower() not in list_models()]
    if unknown:
        print(f"error: unknown model key(s) {', '.join(unknown)}; valid keys: {', '.join(list_models())}",
              file=sys.stderr)
        return 2

    if settings.offline and not args.dry_run:
        print("warning: HF_HUB_OFFLINE is set — anything missing from the cache will fail.",
              file=sys.stderr)

    results = [fetch_one(k, force=args.force, dry_run=args.dry_run) for k in keys]
    print_results(results, dry_run=args.dry_run)

    failures = [r for r in results if r.failed]
    for r in failures:
        if r.status == "failed":
            print(f"error: {r.key}: {r.message}", file=sys.stderr)
    if failures and args.dry_run:
        print(f"{len(failures)} model(s) not usable offline yet; re-run without --dry-run.",
              file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
