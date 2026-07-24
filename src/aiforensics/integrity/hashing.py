"""Content hashing for artifacts under investigation.

The first question in any tampering investigation is *"is this byte-for-byte the
artifact I think it is?"*. Everything else (weight diffs, pickle scans, log
correlation) is only meaningful once the chain of custody is anchored to a
digest, so this module is deliberately the lowest layer of the package: it has
no dependency on transformers, safetensors or the HF cache, and it never
deserialises anything (see :mod:`aiforensics.integrity.serialization` for why
that matters).

Two kinds of digest are produced:

``hash_file`` / ``hash_dir``
    Digests over raw file bytes. Use these for a downloaded checkpoint, a
    poisoned dataset dump, an adapter directory — anything on disk.
``hash_state_dict``
    Digests over *canonicalised tensor bytes*. Use this when the container
    format is irrelevant (a ``.bin`` and a ``.safetensors`` export of the same
    weights have different file digests but identical state-dict digests).

Both return a :class:`Manifest` holding per-entry digests plus a single
aggregate digest. The aggregate is defined as a hash over the ``name\\0digest``
pairs **in sorted order** — ordering matters and sorting is what makes the
aggregate stable across machines, filesystems and ``readdir`` orders.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Digest algorithm used everywhere unless overridden.
DEFAULT_ALGO = "sha256"

#: Read granularity for file hashing. 1 MiB keeps a 16 GB checkpoint off the heap.
DEFAULT_CHUNK_SIZE = 1 << 20

#: Never walked by :func:`hash_dir` — noise that is not part of the artifact.
DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".git", ".git/*", "__pycache__", "__pycache__/*", "*.pyc",
    ".DS_Store", ".ipynb_checkpoints", ".ipynb_checkpoints/*",
    "*.lock", ".locks", ".locks/*",
)

__all__ = [
    "DEFAULT_ALGO",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_EXCLUDE",
    "Manifest",
    "Mismatch",
    "VerifyResult",
    "aggregate_digest",
    "hash_bytes",
    "hash_dir",
    "hash_file",
    "hash_path",
    "hash_state_dict",
    "load_manifest",
    "verify",
]


# --------------------------------------------------------------------------- #
# result containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Manifest:
    """A reproducible inventory of digests for one artifact.

    Attributes:
        kind: ``"file"``, ``"dir"`` or ``"state_dict"`` — determines how
            :func:`verify` recomputes the digests.
        algo: ``hashlib`` algorithm name used for every digest here.
        root: Path the digests were computed from, as recorded at creation time.
            Informational only; :func:`verify` re-hashes whatever target you
            hand it, so a manifest stays usable after the artifact is moved.
        entries: ``name -> hexdigest``. Names are POSIX-relative paths for
            files and state-dict keys for tensors.
        aggregate: Single digest over all entries (see :func:`aggregate_digest`).
        details: ``name -> {...}`` side information (size, mtime, dtype, shape).
            Never folded into the aggregate, so a pure ``mtime`` change does not
            invalidate a manifest.
        meta: Free-form provenance: selection globs, tool version, notes.
    """

    kind: str
    algo: str
    aggregate: str
    entries: dict[str, str] = field(default_factory=dict)
    details: dict[str, dict] = field(default_factory=dict)
    root: str | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict:
        """JSON-ready representation (round-trips through :meth:`from_dict`)."""
        return {
            "kind": self.kind,
            "algo": self.algo,
            "aggregate": self.aggregate,
            "root": self.root,
            "entries": dict(self.entries),
            "details": dict(self.details),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Manifest:
        """Rebuild a manifest previously produced by :meth:`to_dict`."""
        return cls(
            kind=data["kind"],
            algo=data.get("algo", DEFAULT_ALGO),
            aggregate=data["aggregate"],
            entries=dict(data.get("entries", {})),
            details=dict(data.get("details", {})),
            root=data.get("root"),
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the manifest as pretty JSON and return the path written."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return p

    def summary(self) -> str:
        """One-line human summary, e.g. for a log event."""
        return f"{self.kind} {self.algo}:{self.aggregate[:16]}… ({len(self.entries)} entries)"


@dataclass(frozen=True)
class Mismatch:
    """One entry whose digest changed between manifest and target."""

    name: str
    expected: str
    actual: str


@dataclass(frozen=True)
class VerifyResult:
    """Structured outcome of :func:`verify`.

    ``ok`` is true only when there is nothing in ``mismatched``, ``missing`` or
    ``unexpected``. The three buckets are kept separate on purpose: a *missing*
    file (evidence deleted), a *mismatched* file (evidence edited) and an
    *unexpected* file (evidence planted) are different findings in a report.
    """

    ok: bool
    kind: str
    algo: str
    aggregate_expected: str
    aggregate_actual: str
    matched: list[str] = field(default_factory=list)
    mismatched: list[Mismatch] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok

    def to_dict(self) -> dict:
        """JSON-ready representation."""
        return {
            "ok": self.ok,
            "kind": self.kind,
            "algo": self.algo,
            "aggregate_expected": self.aggregate_expected,
            "aggregate_actual": self.aggregate_actual,
            "matched": list(self.matched),
            "mismatched": [{"name": m.name, "expected": m.expected, "actual": m.actual}
                           for m in self.mismatched],
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
        }

    def summary(self) -> str:
        """One-line human summary."""
        verdict = "OK" if self.ok else "FAILED"
        return (
            f"verify {verdict}: {len(self.matched)} matched, "
            f"{len(self.mismatched)} mismatched, {len(self.missing)} missing, "
            f"{len(self.unexpected)} unexpected"
        )

    def to_markdown(self) -> str:
        """Markdown block ready to paste into a forensic report."""
        lines = [
            f"**Integrity check ({self.kind}, {self.algo}): "
            f"{'PASS' if self.ok else 'FAIL'}**",
            "",
            f"- expected aggregate: `{self.aggregate_expected}`",
            f"- actual aggregate:   `{self.aggregate_actual}`",
            f"- matched: {len(self.matched)}",
        ]
        if self.mismatched:
            lines.append(f"- mismatched: {len(self.mismatched)}")
            for m in self.mismatched:
                lines.append(f"  - `{m.name}` expected `{m.expected[:16]}…` got `{m.actual[:16]}…`")
        if self.missing:
            lines.append(f"- missing: {', '.join(f'`{n}`' for n in self.missing)}")
        if self.unexpected:
            lines.append(f"- unexpected: {', '.join(f'`{n}`' for n in self.unexpected)}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def hash_bytes(data: bytes, algo: str = DEFAULT_ALGO) -> str:
    """Hex digest of an in-memory buffer.

    Args:
        data: Bytes to digest.
        algo: Any ``hashlib`` algorithm name (``sha256``, ``blake2b``, …).

    Returns:
        Lower-case hex digest.
    """
    h = hashlib.new(algo)
    h.update(data)
    return h.hexdigest()


def aggregate_digest(entries: Mapping[str, str], algo: str = DEFAULT_ALGO) -> str:
    """Fold per-entry digests into one digest for the whole artifact.

    The input is serialised as ``name`` ``NUL`` ``digest`` ``LF`` for every
    entry, **iterating keys in sorted order**. Two consequences worth knowing:

    * Ordering matters by construction — the aggregate is a hash of a
      *sequence*. Sorting is exactly what removes the machine-dependent part
      (directory iteration order, ``state_dict`` insertion order), so the same
      artifact hashes identically on any host.
    * Renaming an entry changes the aggregate even if every byte is unchanged.
      That is intended: a checkpoint whose shards were renamed is not the same
      artifact.

    Args:
        entries: Mapping of entry name to hex digest.
        algo: ``hashlib`` algorithm name; use the same one as the entries.

    Returns:
        Lower-case hex digest over all entries.
    """
    h = hashlib.new(algo)
    for name in sorted(entries):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(entries[name].encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def hash_file(
    path: str | os.PathLike[str],
    algo: str = DEFAULT_ALGO,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Stream a file through a hash function.

    Reads in ``chunk_size`` blocks so a multi-gigabyte checkpoint never lands in
    memory. The file is opened read-only and **never deserialised**.

    Args:
        path: File to digest.
        algo: ``hashlib`` algorithm name.
        chunk_size: Read block size in bytes.

    Returns:
        Lower-case hex digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist or is not a regular file.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a regular file: {p}")
    h = hashlib.new(algo)
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def _iter_files(
    root: Path,
    include: Iterable[str],
    exclude: Iterable[str],
    follow_symlinks: bool,
) -> Iterator[Path]:
    """Yield regular files under ``root`` matching the glob selection."""
    include = tuple(include)
    exclude = tuple(exclude)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        d = Path(dirpath)
        # Prune excluded directories in place so we never descend into .git etc.
        dirnames[:] = [
            n for n in sorted(dirnames)
            if not _matches(str((d / n).relative_to(root).as_posix()), exclude)
            and not _matches(n, exclude)
        ]
        for name in sorted(filenames):
            fp = d / name
            rel = fp.relative_to(root).as_posix()
            if _matches(rel, exclude) or _matches(name, exclude):
                continue
            if not any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) for pat in include):
                continue
            if fp.is_symlink() and not follow_symlinks:
                # HF cache snapshots are symlink farms; follow_symlinks=True is
                # the right call there, so make the skip visible.
                logger.debug("skipping symlink %s (follow_symlinks=False)", rel)
                continue
            if not fp.is_file():
                continue
            yield fp


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in patterns)


def hash_dir(
    path: str | os.PathLike[str],
    algo: str = DEFAULT_ALGO,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    *,
    include: Iterable[str] = ("*",),
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    follow_symlinks: bool = True,
) -> Manifest:
    """Digest every file in a directory tree.

    Produces one digest per file plus a stable aggregate. Paths are recorded
    relative to ``path`` in POSIX form and sorted before aggregation, so the
    aggregate does not depend on filesystem iteration order (see
    :func:`aggregate_digest`).

    ``follow_symlinks`` defaults to ``True`` because HuggingFace cache snapshots
    are directories of symlinks into ``blobs/``; hashing the link targets is
    what you want when the artifact under investigation lives in the cache.

    Args:
        path: Directory to walk.
        algo: ``hashlib`` algorithm name.
        chunk_size: Read block size in bytes.
        include: Glob patterns a file must match (against the relative path or
            the bare filename) to be included.
        exclude: Glob patterns that drop a file or prune a directory.
        follow_symlinks: Follow symlinked files and directories.

    Returns:
        A :class:`Manifest` with ``kind="dir"``.

    Raises:
        NotADirectoryError: If ``path`` is not a directory.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    entries: dict[str, str] = {}
    details: dict[str, dict] = {}
    total = 0
    for fp in _iter_files(root, include, exclude, follow_symlinks):
        rel = fp.relative_to(root).as_posix()
        stat = fp.stat()
        entries[rel] = hash_file(fp, algo=algo, chunk_size=chunk_size)
        details[rel] = {"size": stat.st_size, "mtime": stat.st_mtime}
        total += stat.st_size

    return Manifest(
        kind="dir",
        algo=algo,
        aggregate=aggregate_digest(entries, algo),
        entries=entries,
        details=details,
        root=str(root),
        meta={
            "n_files": len(entries),
            "total_bytes": total,
            "include": list(include),
            "exclude": list(exclude),
            "follow_symlinks": follow_symlinks,
        },
    )


# --------------------------------------------------------------------------- #
# tensor hashing
# --------------------------------------------------------------------------- #
def _canonical_tensor_bytes(tensor: Any) -> tuple[bytes, str, tuple[int, ...], int]:
    """Return ``(raw_bytes, dtype_str, shape, numel)`` in a host-independent form.

    Canonicalisation rules, in order — each one exists to remove a source of
    digest instability that would otherwise make two identical checkpoints look
    different:

    1. ``detach()`` — autograd metadata is not part of the weights.
    2. ``.to("cpu")`` — a digest must not depend on which device the tensor
       happened to be resident on.
    3. dense/dequantised view — sparse tensors are densified and quantised
       tensors are reduced to their integer representation, because their
       storage layout is an implementation detail.
    4. ``.contiguous().reshape(-1)`` — a permuted view and its materialised
       copy hold the same values, so they must hash the same.
    5. reinterpret as ``uint8`` — works for every dtype including ``bfloat16``
       and the float8 variants, which have no numpy equivalent and therefore
       cannot go through ``.numpy().tobytes()``.
    6. byte order — torch CPU tensors use native endianness. On a big-endian
       host the per-element bytes are reversed so digests match the
       little-endian x86/ARM machines this project runs on. (For complex
       dtypes the reversal is applied per real/imaginary component.)

    The *name* of the tensor is deliberately **not** hashed here — only dtype,
    shape and bytes are. That way a renamed-but-identical tensor still shows up
    with the same digest, which is exactly the signal you want when checking
    whether an attacker moved weights around.
    """
    import torch

    t = tensor
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t).__name__}")
    if t.is_meta:
        raise ValueError("cannot hash a meta tensor (no storage); load with real weights first")

    t = t.detach()
    if t.is_sparse or getattr(t, "is_sparse_csr", False):
        t = t.to_dense()
    t = t.to("cpu")
    if t.is_quantized:
        t = t.int_repr()

    dtype_str = str(t.dtype).replace("torch.", "")
    shape = tuple(int(s) for s in t.shape)
    numel = int(t.numel())

    flat = t.contiguous().reshape(-1)
    try:
        u8 = flat.view(torch.uint8)
        if sys.byteorder != "little":
            width = flat.element_size() // (2 if flat.is_complex() else 1)
            if width > 1:
                u8 = u8.reshape(-1, width).flip(1).reshape(-1)
        raw = u8.numpy().tobytes()
    except (RuntimeError, TypeError):  # pragma: no cover - exotic dtypes
        # Fallback for dtypes torch refuses to bit-cast; numpy handles the rest.
        raw = flat.numpy().tobytes()

    return raw, dtype_str, shape, numel


def hash_state_dict(
    state_dict: Mapping[str, Any],
    algo: str = DEFAULT_ALGO,
    *,
    root: str | None = None,
) -> Manifest:
    """Digest a ``state_dict`` tensor by tensor.

    Container-independent counterpart to :func:`hash_dir`: the same weights
    exported as ``.bin`` and as ``.safetensors`` produce different *file*
    digests but identical *state-dict* digests, which is what lets you say
    "the format changed, the weights did not".

    Each per-tensor digest covers ``dtype|shape|canonical-bytes`` (see
    :func:`_canonical_tensor_bytes` for the full canonicalisation contract, and
    read it before trusting a cross-machine comparison). Non-tensor values —
    some checkpoints stash scalars or strings in the state dict — are hashed
    over their JSON encoding and flagged in ``details``.

    Args:
        state_dict: Mapping of parameter name to tensor.
        algo: ``hashlib`` algorithm name.
        root: Optional provenance string recorded in the manifest.

    Returns:
        A :class:`Manifest` with ``kind="state_dict"``.
    """
    entries: dict[str, str] = {}
    details: dict[str, dict] = {}
    total_params = 0

    for name, value in state_dict.items():
        h = hashlib.new(algo)
        try:
            raw, dtype_str, shape, numel = _canonical_tensor_bytes(value)
        except TypeError:
            # Not a tensor: hash a deterministic textual encoding instead.
            payload = json.dumps(_jsonable(value), sort_keys=True, default=repr).encode("utf-8")
            h.update(b"nontensor|")
            h.update(payload)
            entries[str(name)] = h.hexdigest()
            details[str(name)] = {"kind": "non-tensor", "type": type(value).__name__}
            continue

        h.update(f"{dtype_str}|{shape}|".encode())
        h.update(raw)
        entries[str(name)] = h.hexdigest()
        details[str(name)] = {"dtype": dtype_str, "shape": list(shape), "numel": numel,
                              "nbytes": len(raw)}
        total_params += numel

    return Manifest(
        kind="state_dict",
        algo=algo,
        aggregate=aggregate_digest(entries, algo),
        entries=entries,
        details=details,
        root=root,
        meta={"n_tensors": len(entries), "n_parameters": total_params,
              "byteorder": "little (canonical)"},
    )


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of a non-tensor state-dict value to JSON types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def load_manifest(path: str | os.PathLike[str]) -> Manifest:
    """Read a manifest previously written with :meth:`Manifest.save`."""
    return Manifest.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def verify(
    path_or_sd: str | os.PathLike[str] | Mapping[str, Any],
    manifest: Manifest | Mapping[str, Any] | str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> VerifyResult:
    """Re-hash a target and diff it against a recorded manifest.

    The target is re-hashed with the *manifest's* algorithm and, for
    directories, the *manifest's* recorded include/exclude globs — so a check
    always reproduces the original measurement rather than a new one.

    Args:
        path_or_sd: File path, directory path, or an in-memory ``state_dict``.
        manifest: A :class:`Manifest`, its ``dict`` form, or a path to a saved
            manifest JSON file.
        chunk_size: Read block size for file hashing.

    Returns:
        A :class:`VerifyResult` splitting entries into matched / mismatched /
        missing / unexpected.

    Raises:
        ValueError: If the manifest kind does not match the target type.
    """
    man = _coerce_manifest(manifest)

    if man.kind == "state_dict":
        if not isinstance(path_or_sd, Mapping):
            raise ValueError(
                "manifest kind is 'state_dict' but the target is not a mapping; "
                "load the weights first (integrity.metadata.load_state_dict_for_analysis)"
            )
        actual = hash_state_dict(path_or_sd, algo=man.algo)
    elif man.kind == "dir":
        p = Path(path_or_sd)  # type: ignore[arg-type]
        if not p.is_dir():
            raise ValueError(f"manifest kind is 'dir' but {p} is not a directory")
        meta = man.meta or {}
        actual = hash_dir(
            p,
            algo=man.algo,
            chunk_size=chunk_size,
            include=meta.get("include", ("*",)),
            exclude=meta.get("exclude", DEFAULT_EXCLUDE),
            follow_symlinks=meta.get("follow_symlinks", True),
        )
    elif man.kind == "file":
        p = Path(path_or_sd)  # type: ignore[arg-type]
        name = next(iter(man.entries), p.name)
        digest = hash_file(p, algo=man.algo, chunk_size=chunk_size)
        actual = Manifest(
            kind="file",
            algo=man.algo,
            aggregate=aggregate_digest({name: digest}, man.algo),
            entries={name: digest},
            root=str(p),
        )
    else:
        raise ValueError(f"unknown manifest kind {man.kind!r} (expected file|dir|state_dict)")

    matched: list[str] = []
    mismatched: list[Mismatch] = []
    for name, expected in man.entries.items():
        got = actual.entries.get(name)
        if got is None:
            continue
        if got == expected:
            matched.append(name)
        else:
            mismatched.append(Mismatch(name=name, expected=expected, actual=got))

    missing = sorted(set(man.entries) - set(actual.entries))
    unexpected = sorted(set(actual.entries) - set(man.entries))

    return VerifyResult(
        ok=not (mismatched or missing or unexpected),
        kind=man.kind,
        algo=man.algo,
        aggregate_expected=man.aggregate,
        aggregate_actual=actual.aggregate,
        matched=sorted(matched),
        mismatched=sorted(mismatched, key=lambda m: m.name),
        missing=missing,
        unexpected=unexpected,
    )


def _coerce_manifest(manifest: Manifest | Mapping[str, Any] | str | os.PathLike[str]) -> Manifest:
    if isinstance(manifest, Manifest):
        return manifest
    if isinstance(manifest, Mapping):
        return Manifest.from_dict(manifest)
    return load_manifest(manifest)


def hash_path(
    path: str | os.PathLike[str],
    algo: str = DEFAULT_ALGO,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    **kwargs: Any,
) -> Manifest:
    """Manifest for a file *or* a directory, whichever ``path`` turns out to be.

    Convenience used by :meth:`aiforensics.evidence.RunLogger.log_artifact`,
    which does not know in advance what it is being handed.
    """
    p = Path(path)
    if p.is_dir():
        return hash_dir(p, algo=algo, chunk_size=chunk_size, **kwargs)
    digest = hash_file(p, algo=algo, chunk_size=chunk_size)
    stat = p.stat()
    return Manifest(
        kind="file",
        algo=algo,
        aggregate=aggregate_digest({p.name: digest}, algo),
        entries={p.name: digest},
        details={p.name: {"size": stat.st_size, "mtime": stat.st_mtime}},
        root=str(p),
        meta={"n_files": 1, "total_bytes": stat.st_size},
    )
