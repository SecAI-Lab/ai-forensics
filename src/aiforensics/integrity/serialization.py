"""Static triage of serialized model artifacts — the "deserialization → RCE" threat.

.. danger::
   **NEVER ``torch.load`` OR ``pickle.load`` AN ARTIFACT UNDER INVESTIGATION.**

   A pickle is a *program*, not a data format. ``pickle.load`` — and therefore
   ``torch.load`` on any ``.bin`` / ``.pt`` / ``.pth`` checkpoint — executes the
   opcodes it finds, including ``REDUCE`` calls into ``os.system``,
   ``builtins.eval`` or ``runpy``. That is precisely the on-device attack this
   testbed models: an attacker ships a tampered *serialized* PEFT adapter and
   gets code execution on the victim's device the moment it is loaded.

   Every function in this module is **static**: files are read as bytes, pickle
   opcodes are *parsed* (never executed), and no callable named inside the
   stream is ever resolved. If you find yourself wanting to "just load it and
   see", load it in a throwaway container with no network and no mounted
   secrets — not here.

Layout of the module:

* :func:`scan_pickle` / :func:`scan_dir` — wrappers over ``picklescan`` and
  ``fickling``, normalised into one :class:`ScanResult` dataclass.
* :func:`list_pickle_opcodes` — raw ``pickletools`` bytecode listing, for when
  you want to read the stream yourself.
* :func:`is_safetensors` / :func:`safe_load` — the safe path. safetensors is a
  pure data container (length-prefixed JSON header + raw tensor bytes) with no
  execution semantics, which is why the whole project prefers it.

Both scanners are optional imports guarded by :func:`_require`, so
``import aiforensics`` still works on a bare machine.
"""

from __future__ import annotations

import io
import logging
import os
import pickletools
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Extensions that are (or may contain) a pickle stream and therefore execute code on load.
PICKLE_SUFFIXES: tuple[str, ...] = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".model")

#: Extensions that are pure data containers — no execution semantics.
SAFE_SUFFIXES: tuple[str, ...] = (".safetensors", ".gguf", ".npz", ".npy", ".json")

__all__ = [
    "PICKLE_SUFFIXES",
    "SAFE_SUFFIXES",
    "Finding",
    "ScanResult",
    "Severity",
    "is_safetensors",
    "list_pickle_opcodes",
    "safe_load",
    "scan_dir",
    "scan_pickle",
]


class Severity(IntEnum):
    """Normalised severity across the two scanners. Ordered, so ``max()`` works.

    ``UNKNOWN`` is deliberately ranked above ``SAFE``: a file we could not
    parse is not a file we have cleared.
    """

    SAFE = 0
    UNKNOWN = 1
    SUSPICIOUS = 2
    DANGEROUS = 3
    ERROR = 4

    @property
    def label(self) -> str:
        """Lower-case name, for JSON payloads and report tables."""
        return self.name.lower()


@dataclass(frozen=True)
class Finding:
    """One flagged import or callable inside a pickle stream.

    Attributes:
        module: Module the stream imports, e.g. ``posix``, ``builtins``.
        name: Callable or attribute referenced, e.g. ``system``, ``eval``.
        severity: Normalised severity for this single finding.
        tool: Which scanner reported it (``picklescan`` / ``fickling`` / ``builtin``).
        detail: Free-form explanation from the tool.
        location: Member path inside a zip archive, when applicable.
    """

    module: str
    name: str
    severity: Severity
    tool: str
    detail: str = ""
    location: str | None = None

    def to_dict(self) -> dict:
        """JSON-ready representation."""
        return {
            "module": self.module, "name": self.name,
            "severity": self.severity.label, "tool": self.tool,
            "detail": self.detail, "location": self.location,
        }

    def __str__(self) -> str:  # pragma: no cover - display only
        where = f" @{self.location}" if self.location else ""
        return f"[{self.severity.label}] {self.module}.{self.name}{where} ({self.tool})"


@dataclass
class ScanResult:
    """Outcome of a static scan of one artifact.

    Attributes:
        path: File that was scanned.
        format: Container format as sniffed by
            :func:`aiforensics.integrity.metadata.detect_format`.
        severity: Worst severity across all findings, or ``SAFE`` for a clean
            data container.
        findings: Flagged modules/callables, worst first.
        tools: Scanners that actually ran (a missing tool is recorded in
            ``errors``, never silently ignored).
        errors: Per-tool error strings.
        scanned_at: UTC ISO-8601 timestamp — scan results age, record when.
    """

    path: str
    format: str
    severity: Severity = Severity.UNKNOWN
    findings: list[Finding] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    scanned_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def is_safe(self) -> bool:
        """True only when at least one scanner ran and nothing was flagged.

        Note the asymmetry: ``is_safe`` is a statement about *this scan*, not a
        guarantee about the file. Scanners work from allow/deny lists and can
        be evaded; a clean result lowers suspicion, it does not license
        ``torch.load`` in a production context.
        """
        return bool(self.tools) and self.severity <= Severity.SAFE

    def to_dict(self) -> dict:
        """JSON-ready representation (used by ``RunLogger.log_event``)."""
        return {
            "path": self.path,
            "format": self.format,
            "severity": self.severity.label,
            "findings": [f.to_dict() for f in self.findings],
            "tools": list(self.tools),
            "errors": dict(self.errors),
            "scanned_at": self.scanned_at,
        }

    def summary(self) -> str:
        """One-line human summary."""
        return (
            f"{self.path}: {self.severity.label} "
            f"({len(self.findings)} finding(s), tools={','.join(self.tools) or 'none'})"
        )

    def to_markdown(self) -> str:
        """Markdown fragment for a report."""
        lines = [f"**{self.path}** — format `{self.format}`, severity **{self.severity.label}**"]
        if self.findings:
            lines += ["", "| severity | module | callable | tool | detail |", "|---|---|---|---|---|"]
            for f in self.findings:
                lines.append(
                    f"| {f.severity.label} | `{f.module}` | `{f.name}` | {f.tool} | {f.detail} |")
        if self.errors:
            lines.append("")
            for tool, err in sorted(self.errors.items()):
                lines.append(f"- scanner `{tool}` failed: {err}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# optional dependency handling
# --------------------------------------------------------------------------- #
def _require(module: str, package: str) -> Any:
    """Import an optional scanner, or fail with an actionable message.

    Args:
        module: Importable module path.
        package: pip distribution name to suggest.

    Raises:
        ImportError: With the exact ``pip install`` line to run.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{module} is required for static pickle scanning but is not installed. "
            f"Run `pip install {package}` (it is pinned in docker/requirements.txt; "
            f"rebuild the container with `make build` if you are inside it)."
        ) from exc


def _picklescan() -> Any:
    return _require("picklescan.scanner", "picklescan")


def _fickling() -> Any:
    return _require("fickling.fickle", "fickling")


# --------------------------------------------------------------------------- #
# format checks
# --------------------------------------------------------------------------- #
def is_safetensors(path: str | os.PathLike[str]) -> bool:
    """True if the file really is a safetensors container.

    Decided from the magic layout (``<u64 header length><JSON header>``), not
    from the file extension — renaming a pickle to ``.safetensors`` is a
    one-line attack and must not fool the check.
    """
    from .metadata import detect_format

    return detect_format(path) == "safetensors"


def _is_pickle_candidate(path: Path) -> bool:
    from .metadata import detect_format

    return detect_format(path) in {"pickle", "pytorch-zip"}


# --------------------------------------------------------------------------- #
# scanners
# --------------------------------------------------------------------------- #
_PICKLESCAN_SEVERITY = {
    "innocuous": Severity.SAFE,
    "suspicious": Severity.SUSPICIOUS,
    "dangerous": Severity.DANGEROUS,
}

_FICKLING_SEVERITY = {
    "likely_safe": Severity.SAFE,
    "possibly_unsafe": Severity.SUSPICIOUS,
    "suspicious": Severity.SUSPICIOUS,
    "likely_unsafe": Severity.DANGEROUS,
    "likely_overtly_malicious": Severity.DANGEROUS,
    "overtly_malicious": Severity.DANGEROUS,
}


def scan_pickle(
    path: str | os.PathLike[str],
    *,
    tools: Iterable[str] = ("picklescan", "fickling"),
) -> ScanResult:
    """Statically scan one artifact for dangerous pickle content.

    .. danger::
       This function **never deserialises** the file. ``picklescan`` walks the
       opcode stream looking for ``GLOBAL``/``STACK_GLOBAL`` imports and matches
       them against its dangerous-module lists; ``fickling`` builds an AST of
       the pickle program and analyses it. Neither executes the payload, and
       neither should ever be replaced by a ``pickle.load`` "just to check".

    Non-pickle containers short-circuit: a genuine safetensors or GGUF file is
    reported ``SAFE`` with ``tools=["format"]`` because there is no code to
    find, not because a scanner cleared it.

    Args:
        path: File to scan.
        tools: Which scanners to run. Both by default — they disagree often
            enough that running only one is a mistake.

    Returns:
        A :class:`ScanResult`. Scanner import failures land in ``errors`` and
        push the severity to ``ERROR`` rather than raising, so a directory scan
        never dies halfway through.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    from .metadata import detect_format

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a regular file: {p}")

    fmt = detect_format(p)
    result = ScanResult(path=str(p), format=fmt)

    if fmt in {"safetensors", "gguf", "numpy", "json"}:
        result.severity = Severity.SAFE
        result.tools.append("format")
        return result

    wanted = list(tools)
    if "picklescan" in wanted:
        _run_picklescan(p, result)
    if "fickling" in wanted:
        _run_fickling(p, result)

    if result.findings:
        result.severity = max(f.severity for f in result.findings)
    elif result.errors and not result.tools:
        result.severity = Severity.ERROR
    elif result.tools:
        result.severity = Severity.SAFE

    result.findings.sort(key=lambda f: (-int(f.severity), f.module, f.name))
    return result


def _run_picklescan(p: Path, result: ScanResult) -> None:
    """Run picklescan and fold its globals into ``result``. Never raises."""
    try:
        scanner = _picklescan()
    except ImportError as exc:
        result.errors["picklescan"] = str(exc)
        logger.warning("picklescan unavailable: %s", exc)
        return

    try:
        raw = scanner.scan_file_path(str(p))
    except Exception as exc:  # pragma: no cover - depends on artifact
        result.errors["picklescan"] = f"{type(exc).__name__}: {exc}"
        return

    result.tools.append("picklescan")
    for g in getattr(raw, "globals", []) or []:
        safety = getattr(g, "safety", None)
        key = str(getattr(safety, "name", safety) or "unknown").lower()
        sev = _PICKLESCAN_SEVERITY.get(key, Severity.UNKNOWN)
        if sev <= Severity.SAFE:
            continue  # innocuous imports are noise in a report
        result.findings.append(Finding(
            module=str(getattr(g, "module", "?")),
            name=str(getattr(g, "name", "?")),
            severity=sev,
            tool="picklescan",
            detail=f"picklescan safety={key}",
        ))
    if getattr(raw, "scan_err", False):
        result.errors["picklescan"] = "scanner reported a parse error"


def _run_fickling(p: Path, result: ScanResult) -> None:
    """Run fickling's static analysis and fold its verdict into ``result``.

    fickling's API has moved around between releases, so every access is
    defensive: we try the analysis entry points in order and record whichever
    one worked. A tool that cannot run is an ``errors`` entry, never a silent
    pass.
    """
    try:
        fickle = _fickling()
    except ImportError as exc:
        result.errors["fickling"] = str(exc)
        logger.warning("fickling unavailable: %s", exc)
        return

    try:
        data = p.read_bytes()
        parsed_any = False
        for stream in _pickle_streams(p, data):
            try:
                pickled = fickle.Pickled.load(stream.payload)
            except Exception as exc:
                logger.debug("fickling could not parse %s@%s: %s", p, stream.location, exc)
                continue
            parsed_any = True
            if "fickling" not in result.tools:
                result.tools.append("fickling")
            _fickling_analyse(pickled, result, stream.location)
        if not parsed_any:
            result.errors["fickling"] = "no parsable pickle stream found"
    except Exception as exc:  # pragma: no cover - depends on artifact
        result.errors["fickling"] = f"{type(exc).__name__}: {exc}"


def _fickling_analyse(pickled: Any, result: ScanResult, location: str | None) -> None:
    """Extract findings from a parsed fickling ``Pickled`` object."""
    import importlib

    severity = Severity.UNKNOWN
    detail = ""
    try:
        analysis = importlib.import_module("fickling.analysis")
        check = getattr(analysis, "check_safety", None)
        results = check(pickled) if callable(check) else None
        if results is not None:
            sev_obj = getattr(results, "severity", None)
            key = str(getattr(sev_obj, "name", sev_obj) or "").lower()
            severity = _FICKLING_SEVERITY.get(key, Severity.UNKNOWN)
            detail = str(results).strip().replace("\n", " ")[:400]
    except Exception as exc:  # pragma: no cover - API drift
        logger.debug("fickling.analysis unavailable (%s); falling back to imports", exc)

    # Regardless of the verdict object, report the modules the stream imports:
    # that is the concrete, human-checkable evidence for a report.
    try:
        for node in pickled.properties.imports:
            module = getattr(node, "module", None) or ""
            names = [a.name for a in getattr(node, "names", [])] or ["*"]
            for name in names:
                result.findings.append(Finding(
                    module=str(module), name=str(name),
                    severity=max(severity, Severity.SUSPICIOUS),
                    tool="fickling",
                    detail=detail or "imported by the pickle stream",
                    location=location,
                ))
    except Exception as exc:  # pragma: no cover - API drift
        logger.debug("could not enumerate fickling imports: %s", exc)
        if severity >= Severity.SUSPICIOUS:
            result.findings.append(Finding(
                module="?", name="?", severity=severity, tool="fickling",
                detail=detail or "fickling flagged this stream", location=location,
            ))


@dataclass(frozen=True)
class _Stream:
    """A raw pickle byte stream plus where inside the artifact it came from."""

    payload: bytes
    location: str | None


def _pickle_streams(path: Path, data: bytes) -> list[_Stream]:
    """Enumerate pickle streams in an artifact **without unpickling anything**.

    A torch >= 1.6 checkpoint is a zip archive whose ``*/data.pkl`` member is
    the pickle program; older checkpoints are a bare stream. Zip members are
    read with :mod:`zipfile`, which is pure decompression — no code runs.
    """
    if zipfile.is_zipfile(io.BytesIO(data)):
        streams: list[_Stream] = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.filename.endswith(".pkl") or info.filename.endswith("data.pkl"):
                    streams.append(_Stream(zf.read(info.filename), info.filename))
        return streams
    return [_Stream(data, None)]


def scan_dir(
    path: str | os.PathLike[str],
    *,
    suffixes: Iterable[str] = PICKLE_SUFFIXES,
    tools: Iterable[str] = ("picklescan", "fickling"),
    recursive: bool = True,
    include_safe: bool = True,
) -> list[ScanResult]:
    """Scan every candidate artifact in a directory tree, statically.

    Intended entry point for triaging a downloaded model repo or an adapter
    drop: walk it, scan every pickle-shaped file, and hand the results to a
    report. Nothing is deserialised at any point.

    Args:
        path: Directory to walk.
        suffixes: Extensions to consider. Files whose *magic bytes* say pickle
            are picked up regardless of extension.
        tools: Scanners to run per file.
        recursive: Walk subdirectories.
        include_safe: Keep clean results in the output (useful as an inventory;
            set ``False`` for a findings-only list).

    Returns:
        Results sorted worst-severity first.

    Raises:
        NotADirectoryError: If ``path`` is not a directory.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    suffixes = tuple(s.lower() for s in suffixes)
    files = sorted(root.rglob("*") if recursive else root.glob("*"))
    out: list[ScanResult] = []
    for fp in files:
        if not fp.is_file():
            continue
        if fp.suffix.lower() not in suffixes and not _is_pickle_candidate(fp):
            continue
        try:
            res = scan_pickle(fp, tools=tools)
        except Exception as exc:  # pragma: no cover - unreadable file
            res = ScanResult(path=str(fp), format="unknown", severity=Severity.ERROR,
                             errors={"scan": f"{type(exc).__name__}: {exc}"})
        if include_safe or not res.is_safe:
            out.append(res)
    out.sort(key=lambda r: (-int(r.severity), r.path))
    return out


def list_pickle_opcodes(
    path: str | os.PathLike[str],
    *,
    max_ops: int | None = None,
    member: str | None = None,
) -> list[dict]:
    """Disassemble the pickle bytecode of an artifact for manual inspection.

    .. danger::
       Static only. :func:`pickletools.genops` *parses* opcodes; it never
       dispatches them, so no ``REDUCE`` is ever performed and no import is
       ever resolved. Do not "verify" a finding by loading the file.

    The interesting opcodes when hunting for a malicious adapter are
    ``GLOBAL`` / ``STACK_GLOBAL`` (what gets imported), ``REDUCE`` /
    ``INST`` / ``OBJ`` / ``NEWOBJ`` (what gets called) and ``BUILD``.

    Args:
        path: Artifact to disassemble.
        max_ops: Stop after this many opcodes (streams can be enormous).
        member: For zip-based torch checkpoints, which member to read; defaults
            to the first ``*.pkl`` found.

    Returns:
        A list of ``{"pos", "opcode", "arg", "proto"}`` dicts, in stream order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If no pickle stream can be located.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a regular file: {p}")

    data = p.read_bytes()
    streams = _pickle_streams(p, data)
    if member is not None:
        streams = [s for s in streams if s.location == member]
    if not streams:
        raise ValueError(f"{p}: no pickle stream found (member={member!r})")

    ops: list[dict] = []
    for opcode, arg, pos in pickletools.genops(io.BytesIO(streams[0].payload)):
        ops.append({
            "pos": pos,
            "opcode": opcode.name,
            "arg": arg if isinstance(arg, (str, int, float, bool, type(None))) else repr(arg),
            "proto": opcode.proto,
        })
        if max_ops is not None and len(ops) >= max_ops:
            break
    return ops


# --------------------------------------------------------------------------- #
# the safe path
# --------------------------------------------------------------------------- #
def safe_load(
    path: str | os.PathLike[str],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load tensors **only** from a safetensors container.

    This is the one loading function in the package that is safe to point at an
    untrusted artifact: safetensors has no execution semantics — a JSON header
    plus raw little-endian tensor bytes — so there is nothing for an attacker
    to run.

    Pickle-based containers are refused with a pointer at the scanners. That
    refusal is the whole point: if you need the weights out of a ``.bin``,
    scan it first (:func:`scan_pickle`), read its opcodes
    (:func:`list_pickle_opcodes`), and only then use
    :func:`aiforensics.integrity.metadata.load_state_dict_for_analysis` with an
    explicit ``allow_pickle=True`` in a disposable container.

    Args:
        path: Artifact to load.
        device: Device to map onto; ``cpu`` by default so GPU 0 stays free.

    Returns:
        Mapping of tensor name to tensor.

    Raises:
        ValueError: If ``path`` is not a genuine safetensors file.
        ImportError: If ``safetensors`` is not installed.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"not a regular file: {p}")
    if not is_safetensors(p):
        from .metadata import detect_format

        raise ValueError(
            f"refusing to load {p}: it is not a safetensors container "
            f"(detected format: {detect_format(p)}). Loading a pickle-based checkpoint "
            f"executes code embedded in the artifact. Triage it statically first with "
            f"aiforensics.integrity.serialization.scan_pickle() / list_pickle_opcodes(), "
            f"then use metadata.load_state_dict_for_analysis(..., allow_pickle=True) if you "
            f"have decided the risk is acceptable."
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for safe_load: pip install safetensors"
        ) from exc
    return dict(load_file(str(p), device=device))
