"""Descriptive metadata and weight-diff plumbing for model artifacts.

Two jobs, both purely descriptive:

``model_metadata`` / ``safetensors_header`` / ``detect_format``
    Everything a forensic report needs to *identify* an artifact — file
    inventory, container format, ``config.json``, tokenizer files, parameter
    count, dtype histogram, HF commit hash, safetensors ``__metadata__``.
    Wherever possible this is read from headers, without materialising a single
    tensor: a 16 GB checkpoint can be inventoried in milliseconds, and — more
    importantly — a *suspect* checkpoint can be inventoried without executing
    any of its code.

``compare_models``
    The core "was this model edited?" primitive: added / removed tensors,
    shape and dtype mismatches, and per-tensor divergence statistics
    (L2, L∞, cosine, fraction of changed elements) ranked by magnitude.

.. important::
   ``compare_models`` is **descriptive plumbing, not a detector**. It reports
   numbers; it does not decide whether a model was edited, and it deliberately
   contains no thresholds, no heuristics and no verdicts. Model-editing
   detection belongs in :mod:`aiforensics.detectors`, owned by the researcher
   working on weight-space analysis.
"""

from __future__ import annotations

import json
import logging
import os
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import HF_CACHE
from .hashing import DEFAULT_ALGO, Manifest, hash_path

logger = logging.getLogger(__name__)

#: Files that identify a tokenizer, listed in the metadata inventory. A
#: tampered tokenizer (extra trigger token, changed merges) is a real attack
#: surface, so its presence/absence is worth recording explicitly.
TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "vocab.json", "vocab.txt", "merges.txt", "special_tokens_map.json",
    "added_tokens.json", "chat_template.jinja",
)

#: Small JSON side-cars whose full contents go into the report verbatim.
CONFIG_FILES: tuple[str, ...] = (
    "config.json", "generation_config.json", "adapter_config.json",
    "preprocessor_config.json", "quantization_config.json",
)

_SAFETENSORS_MAX_HEADER = 100_000_000  # 100 MB, the safetensors reference limit

__all__ = [
    "CONFIG_FILES",
    "TOKENIZER_FILES",
    "ModelDiff",
    "TensorDiff",
    "compare_models",
    "detect_format",
    "dtype_histogram",
    "load_state_dict_for_analysis",
    "model_manifest",
    "model_metadata",
    "resolve_hf_commit",
    "safetensors_header",
    "safetensors_metadata",
]


# --------------------------------------------------------------------------- #
# container format sniffing
# --------------------------------------------------------------------------- #
def detect_format(path: str | os.PathLike[str]) -> str:
    """Identify a checkpoint container by magic bytes, not by extension.

    Extensions lie — an attacker renaming ``evil.pkl`` to ``model.safetensors``
    is exactly the on-device threat this project studies — so the decision is
    made from the first bytes of the file.

    Args:
        path: File or directory to inspect.

    Returns:
        One of ``safetensors``, ``gguf``, ``pytorch-zip`` (torch >= 1.6 zip
        archive containing a pickle), ``pickle`` (legacy raw pickle stream),
        ``onnx``, ``hdf5``, ``json``, ``numpy``, ``directory``, ``missing`` or
        ``unknown``.
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.is_dir():
        return "directory"

    with p.open("rb") as fh:
        head = fh.read(16)

    if head[:4] == b"GGUF":
        return "gguf"
    if head[:4] == b"PK\x03\x04":
        return "pytorch-zip"
    if head[:6] == b"\x93NUMPY":
        return "numpy"
    if head[:8] == b"\x89HDF\r\n\x1a\n":
        return "hdf5"
    if head[:2] == b"\x08\x01" or p.suffix == ".onnx":
        return "onnx"
    # Raw pickle streams start with a PROTO opcode (0x80) or an old-style opcode.
    if head[:1] == b"\x80" and len(head) > 1 and head[1] <= 6:
        return "pickle"
    if head[:1] in (b"(", b"]", b"}", b"c") and p.suffix in {".pkl", ".pickle", ".pt", ".bin"}:
        return "pickle"
    # safetensors: 8-byte little-endian header length followed by JSON '{'.
    if len(head) >= 9:
        n = struct.unpack("<Q", head[:8])[0]
        if 0 < n < _SAFETENSORS_MAX_HEADER and head[8:9] == b"{":
            return "safetensors"
    if head[:1] in (b"{", b"["):
        return "json"
    return "unknown"


def safetensors_header(path: str | os.PathLike[str]) -> dict:
    """Read a safetensors header **without loading a single tensor**.

    The safetensors layout is ``<u64 little-endian header length><JSON header>
    <tensor data>``; the header maps every tensor name to ``dtype``, ``shape``
    and byte ``data_offsets``, plus an optional ``__metadata__`` string map.
    That is enough to compute parameter counts and dtype histograms for an
    untrusted 16 GB file while touching only its first few kilobytes.

    Args:
        path: Path to a ``.safetensors`` file.

    Returns:
        The parsed header, including the ``__metadata__`` key when present.

    Raises:
        ValueError: If the file is not a valid safetensors container.
    """
    p = Path(path)
    with p.open("rb") as fh:
        raw_len = fh.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"{p} is too short to be a safetensors file")
        n = struct.unpack("<Q", raw_len)[0]
        if not 0 < n < _SAFETENSORS_MAX_HEADER:
            raise ValueError(f"{p}: implausible safetensors header length {n}")
        blob = fh.read(n)
    if len(blob) != n:
        raise ValueError(f"{p}: truncated safetensors header ({len(blob)} of {n} bytes)")
    try:
        header = json.loads(blob)
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
        raise ValueError(f"{p}: safetensors header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{p}: safetensors header is not a JSON object")
    return header


def safetensors_metadata(path: str | os.PathLike[str]) -> dict:
    """Just the ``__metadata__`` string map from a safetensors header.

    Training frameworks stamp provenance here (``format``, trainer version,
    sometimes a base-model id), so a *missing* or *rewritten* ``__metadata__``
    on an otherwise well-formed file is itself worth reporting.
    """
    return dict(safetensors_header(path).get("__metadata__", {}) or {})


def _header_stats(header: Mapping[str, Any]) -> tuple[int, dict[str, int], int]:
    """``(param_count, dtype -> param_count, total tensor bytes)`` from a header."""
    n_params = 0
    hist: dict[str, int] = {}
    nbytes = 0
    for name, spec in header.items():
        if name == "__metadata__" or not isinstance(spec, Mapping):
            continue
        shape = spec.get("shape") or []
        numel = 1
        for dim in shape:
            numel *= int(dim)
        dtype = str(spec.get("dtype", "?"))
        n_params += numel
        hist[dtype] = hist.get(dtype, 0) + numel
        offsets = spec.get("data_offsets") or [0, 0]
        nbytes += int(offsets[1]) - int(offsets[0])
    return n_params, hist, nbytes


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def resolve_hf_commit(path: str | os.PathLike[str]) -> str | None:
    """Recover the HuggingFace commit hash for an artifact, if it is knowable.

    Works for the two cases that actually occur here:

    * the path sits inside the shared HF cache, whose layout is
      ``models--<org>--<name>/snapshots/<commit-sha>/...`` — the snapshot
      directory name *is* the commit;
    * a local copy that kept a ``refs/main`` file or a ``.hf_commit`` marker.

    Returns ``None`` when the artifact has no recoverable upstream identity,
    which is itself a finding: a checkpoint that cannot be traced to a
    published commit is by definition not the published model.
    """
    p = Path(path).resolve()
    parts = p.parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            if len(candidate) in (40, 64) or all(c in "0123456789abcdef" for c in candidate):
                return candidate
    marker = (p if p.is_dir() else p.parent) / ".hf_commit"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or None
    # Cache repos keep resolved refs next to the snapshots directory.
    for anc in [p, *p.parents]:
        refs = anc / "refs"
        if refs.is_dir() and anc.name.startswith("models--"):
            for ref in sorted(refs.iterdir()):
                if ref.is_file():
                    return ref.read_text(encoding="utf-8").strip() or None
        if anc == HF_CACHE:
            break
    return None


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - unreadable side-car
        logger.warning("could not read %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def dtype_histogram(state_dict: Mapping[str, Any]) -> dict[str, int]:
    """Parameter count per dtype for an in-memory ``state_dict``.

    A mixed histogram is informative: a bf16 model with a handful of fp32
    tensors usually means someone re-saved part of it.
    """
    hist: dict[str, int] = {}
    for value in state_dict.values():
        numel = getattr(value, "numel", None)
        dtype = getattr(value, "dtype", None)
        if numel is None or dtype is None:
            continue
        key = str(dtype).replace("torch.", "")
        hist[key] = hist.get(key, 0) + int(numel())
    return dict(sorted(hist.items()))


def model_metadata(
    path_or_model: str | os.PathLike[str] | Any,
    *,
    hash_files: bool = False,
    algo: str = DEFAULT_ALGO,
    max_files: int = 5000,
) -> dict:
    """Collect everything a forensic report needs to identify an artifact.

    For a **path** this is a static inspection: file inventory with sizes and
    mtimes, container format per file, the contents of ``config.json`` and the
    other side-cars in :data:`CONFIG_FILES`, which tokenizer files are present,
    the safetensors ``__metadata__`` header, parameter count and dtype
    histogram derived from safetensors headers, and the HF commit hash if the
    path is traceable. **No tensor is loaded and no pickle is opened.**

    For a **live ``torch.nn.Module``** the same fields are filled in from the
    object itself (class name, parameter counts, dtype histogram, device).

    Parameter counts are ``None`` for pickle-based checkpoints (``.bin``,
    ``.pt``): counting them would require deserialising an artifact that is
    under investigation, which this package never does. Use
    :func:`aiforensics.integrity.serialization.scan_pickle` first, then
    :func:`load_state_dict_for_analysis` with ``allow_pickle=True`` if the scan
    comes back clean.

    Args:
        path_or_model: Path to a checkpoint file or model directory, or an
            ``nn.Module`` / ``state_dict``.
        hash_files: Also compute a content digest for every file (slow on
            multi-gigabyte checkpoints, but this is what pins the artifact in a
            chain of custody).
        algo: Digest algorithm when ``hash_files`` is set.
        max_files: Safety valve for pathological directories.

    Returns:
        A JSON-serialisable ``dict``.
    """
    if not isinstance(path_or_model, (str, os.PathLike)):
        return _module_metadata(path_or_model)

    root = Path(path_or_model)
    info: dict[str, Any] = {
        "source": "path",
        "path": str(root),
        "exists": root.exists(),
        "is_dir": root.is_dir(),
        "format": detect_format(root),
        "hf_commit": resolve_hf_commit(root),
        "files": [],
        "configs": {},
        "tokenizer_files": [],
        "safetensors_metadata": {},
        "n_parameters": None,
        "dtype_histogram": {},
        "total_bytes": 0,
        "notes": [],
    }
    if not root.exists():
        info["notes"].append("path does not exist")
        return info

    files = sorted(root.rglob("*")) if root.is_dir() else [root]
    n_params = 0
    hist: dict[str, int] = {}
    saw_safetensors = False
    saw_pickle = False

    for fp in files[:max_files]:
        if not fp.is_file():
            continue
        stat = fp.stat()
        rel = fp.relative_to(root).as_posix() if root.is_dir() else fp.name
        fmt = detect_format(fp)
        entry: dict[str, Any] = {
            "name": rel,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "format": fmt,
        }
        if hash_files:
            entry["digest"] = hash_path(fp, algo=algo).entries[fp.name]
            entry["algo"] = algo
        info["total_bytes"] += stat.st_size

        if fmt == "safetensors":
            saw_safetensors = True
            try:
                header = safetensors_header(fp)
            except ValueError as exc:
                entry["error"] = str(exc)
            else:
                p_count, p_hist, _ = _header_stats(header)
                entry["n_tensors"] = len([k for k in header if k != "__metadata__"])
                entry["n_parameters"] = p_count
                n_params += p_count
                for k, v in p_hist.items():
                    hist[k] = hist.get(k, 0) + v
                meta = header.get("__metadata__") or {}
                if meta:
                    info["safetensors_metadata"][rel] = dict(meta)
        elif fmt in {"pytorch-zip", "pickle"}:
            saw_pickle = True

        if fp.name in CONFIG_FILES:
            info["configs"][fp.name] = _read_json(fp)
        if fp.name in TOKENIZER_FILES:
            info["tokenizer_files"].append(rel)
        if fp.name.endswith(".index.json"):
            index = _read_json(fp)
            if isinstance(index, Mapping):
                total = (index.get("metadata") or {}).get("total_size")
                if total is not None:
                    entry["index_total_size"] = total

        info["files"].append(entry)

    if len(files) > max_files:
        info["notes"].append(f"inventory truncated at max_files={max_files}")
    if saw_safetensors:
        info["n_parameters"] = n_params
        info["dtype_histogram"] = dict(sorted(hist.items()))
    if saw_pickle:
        info["notes"].append(
            "pickle-based weights present: parameter counts are not reported because "
            "counting them would require deserialising the artifact. Run "
            "integrity.serialization.scan_pickle() first."
        )
    info["tokenizer_files"] = sorted(info["tokenizer_files"])
    return info


def _module_metadata(obj: Any) -> dict:
    """Metadata for a live ``nn.Module`` or an in-memory ``state_dict``."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep in the container
        raise RuntimeError("torch is required to inspect an in-memory model") from None

    if isinstance(obj, Mapping):
        sd = obj
        cls_name = "state_dict"
        extra: dict[str, Any] = {}
    else:
        sd = obj.state_dict()
        cls_name = type(obj).__name__
        params = list(getattr(obj, "parameters", lambda: [])())
        extra = {
            "n_trainable_parameters": sum(p.numel() for p in params if p.requires_grad),
            "devices": sorted({str(p.device) for p in params}),
            "training_mode": bool(getattr(obj, "training", False)),
            "hf_config": _hf_config_of(obj),
        }

    n_params = sum(int(v.numel()) for v in sd.values() if isinstance(v, torch.Tensor))
    return {
        "source": "module",
        "class": cls_name,
        "n_tensors": len(sd),
        "n_parameters": n_params,
        "dtype_histogram": dtype_histogram(sd),
        **extra,
    }


def _hf_config_of(model: Any) -> dict | None:
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    to_dict = getattr(cfg, "to_dict", None)
    try:
        return to_dict() if callable(to_dict) else dict(vars(cfg))
    except Exception:  # pragma: no cover - exotic config objects
        return {"repr": repr(cfg)}


# --------------------------------------------------------------------------- #
# loading weights for analysis
# --------------------------------------------------------------------------- #
def load_state_dict_for_analysis(
    source: str | os.PathLike[str] | Mapping[str, Any] | Any,
    *,
    allow_pickle: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Materialise a ``state_dict`` from whatever the caller has.

    Accepts an existing mapping, an ``nn.Module``, a ``.safetensors`` file, or
    a model directory (sharded safetensors are merged via their
    ``*.index.json`` or by globbing).

    Pickle-backed checkpoints (``.bin``, ``.pt``, ``.pth``) are **refused by
    default**. Loading one executes arbitrary code from the artifact you are
    investigating; that is the on-device RCE threat model this project exists
    to study. Scan it first with
    :func:`aiforensics.integrity.serialization.scan_pickle`, and only then pass
    ``allow_pickle=True`` — the call logs a loud warning when you do.

    Args:
        source: Mapping, module, file path or directory.
        allow_pickle: Permit ``torch.load`` on pickle containers. Off by default.
        device: Device to map tensors onto; ``"cpu"`` keeps GPU 0 free.

    Returns:
        Mapping of parameter name to tensor.

    Raises:
        ValueError: On a pickle container without ``allow_pickle=True``, or when
            no weights can be found.
    """
    import torch

    if isinstance(source, Mapping):
        return dict(source)
    if hasattr(source, "state_dict") and not isinstance(source, (str, os.PathLike)):
        return dict(source.state_dict())

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"no such model artifact: {p}")

    if p.is_dir():
        shards = sorted(p.glob("*.safetensors"))
        if shards:
            merged: dict[str, Any] = {}
            for shard in shards:
                merged.update(_load_safetensors(shard, device))
            return merged
        pickles = sorted(p.glob("*.bin")) + sorted(p.glob("*.pt")) + sorted(p.glob("*.pth"))
        if pickles:
            merged = {}
            for shard in pickles:
                merged.update(load_state_dict_for_analysis(
                    shard, allow_pickle=allow_pickle, device=device))
            return merged
        raise ValueError(f"no .safetensors or pickle weights found in {p}")

    fmt = detect_format(p)
    if fmt == "safetensors":
        return _load_safetensors(p, device)
    if fmt in {"pytorch-zip", "pickle"}:
        if not allow_pickle:
            raise ValueError(
                f"{p} is a pickle-based checkpoint ({fmt}). Loading it would execute code "
                f"embedded in the artifact under investigation. Run "
                f"aiforensics.integrity.serialization.scan_pickle({str(p)!r}) first, then "
                f"re-call with allow_pickle=True if the scan is clean."
            )
        logger.warning(
            "torch.load() on %s — this DESERIALISES an artifact under investigation. "
            "Only do this after a clean picklescan/fickling result, ideally in a throwaway "
            "container.", p,
        )
        # weights_only=True is defence in depth, not a licence to skip the scan:
        # it restricts the unpickler to tensor primitives, but it is a torch-side
        # allowlist that has been bypassed before.
        obj = torch.load(p, map_location=device, weights_only=True)
        if isinstance(obj, Mapping) and "state_dict" in obj and isinstance(obj["state_dict"], Mapping):
            obj = obj["state_dict"]
        if not isinstance(obj, Mapping):
            raise ValueError(f"{p} did not deserialise to a state_dict (got {type(obj).__name__})")
        return dict(obj)
    raise ValueError(f"unsupported weight container for {p} (detected format: {fmt})")


def _load_safetensors(path: Path, device: str) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover - pinned in requirements
        raise RuntimeError("safetensors is required: pip install safetensors") from exc
    return dict(load_file(str(path), device=device))


# --------------------------------------------------------------------------- #
# weight diffing
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TensorDiff:
    """Divergence statistics for one tensor present in both models.

    All statistics are computed in float32 on CPU regardless of the stored
    dtype, so a bf16-vs-fp32 comparison is meaningful rather than a dtype
    artifact. ``NaN`` appears where a statistic is undefined (e.g. cosine of a
    zero tensor) rather than being silently replaced by 0.
    """

    name: str
    shape: tuple[int, ...]
    numel: int
    dtype_a: str
    dtype_b: str
    l2: float           #: Euclidean norm of (b - a)
    linf: float         #: max |b - a|, i.e. the largest single-weight edit
    cosine: float       #: cosine similarity of the flattened tensors
    rel_l2: float       #: ||b - a|| / ||a||, scale-free edit magnitude
    mean_abs: float     #: mean |b - a|
    changed_fraction: float  #: fraction of elements with |b - a| > atol
    n_changed: int

    def to_dict(self) -> dict:
        """JSON-ready representation."""
        return {
            "name": self.name, "shape": list(self.shape), "numel": self.numel,
            "dtype_a": self.dtype_a, "dtype_b": self.dtype_b,
            "l2": self.l2, "linf": self.linf, "cosine": self.cosine,
            "rel_l2": self.rel_l2, "mean_abs": self.mean_abs,
            "changed_fraction": self.changed_fraction, "n_changed": self.n_changed,
        }


@dataclass(frozen=True)
class ModelDiff:
    """Structural + numeric difference between two checkpoints.

    Descriptive only. It answers "*what* differs and *by how much*", never
    "was this tampered with" — that judgement needs a detector and a
    calibrated threshold, both of which live in :mod:`aiforensics.detectors`.
    """

    left: str
    right: str
    added: list[str] = field(default_factory=list)      #: present in b, absent in a
    removed: list[str] = field(default_factory=list)    #: present in a, absent in b
    shape_mismatches: list[dict] = field(default_factory=list)
    dtype_mismatches: list[dict] = field(default_factory=list)
    diffs: list[TensorDiff] = field(default_factory=list)
    atol: float = 0.0

    @property
    def n_common(self) -> int:
        """Number of tensors compared numerically."""
        return len(self.diffs)

    @property
    def n_changed_tensors(self) -> int:
        """How many common tensors have at least one differing element."""
        return sum(1 for d in self.diffs if d.n_changed > 0)

    @property
    def identical(self) -> bool:
        """True when the two models agree structurally and element-wise."""
        return (
            not self.added and not self.removed
            and not self.shape_mismatches and not self.dtype_mismatches
            and self.n_changed_tensors == 0
        )

    def top_changed(self, n: int = 10, by: str = "l2") -> list[TensorDiff]:
        """The ``n`` most-changed tensors.

        Args:
            n: How many to return.
            by: Ranking statistic — ``l2``, ``linf``, ``rel_l2``, ``mean_abs``
                or ``changed_fraction``. ``rel_l2`` is usually the fairest
                across layers of different scale.

        Raises:
            ValueError: On an unknown ranking key.
        """
        valid = {"l2", "linf", "rel_l2", "mean_abs", "changed_fraction"}
        if by not in valid:
            raise ValueError(f"unknown ranking key {by!r}; valid keys: {sorted(valid)}")
        import math

        return sorted(
            self.diffs,
            key=lambda d: (-math.inf if math.isnan(getattr(d, by)) else getattr(d, by)),
            reverse=True,
        )[:n]

    def to_dict(self) -> dict:
        """JSON-ready representation."""
        return {
            "left": self.left,
            "right": self.right,
            "identical": self.identical,
            "n_common": self.n_common,
            "n_changed_tensors": self.n_changed_tensors,
            "added": list(self.added),
            "removed": list(self.removed),
            "shape_mismatches": list(self.shape_mismatches),
            "dtype_mismatches": list(self.dtype_mismatches),
            "atol": self.atol,
            "diffs": [d.to_dict() for d in self.diffs],
        }

    def summary(self) -> str:
        """One-line human summary."""
        return (
            f"{self.left} vs {self.right}: {self.n_changed_tensors}/{self.n_common} tensors differ, "
            f"+{len(self.added)} added, -{len(self.removed)} removed, "
            f"{len(self.shape_mismatches)} shape / {len(self.dtype_mismatches)} dtype mismatches"
        )

    def to_markdown(self, top: int = 10, by: str = "rel_l2") -> str:
        """Markdown report fragment: summary plus the top-``top`` changed tensors."""
        lines = [
            f"### Weight diff — `{self.left}` vs `{self.right}`",
            "",
            f"- identical: **{self.identical}**",
            f"- tensors compared: {self.n_common} ({self.n_changed_tensors} differ)",
            f"- added: {len(self.added)} | removed: {len(self.removed)}",
            f"- shape mismatches: {len(self.shape_mismatches)} | "
            f"dtype mismatches: {len(self.dtype_mismatches)}",
            "",
            "| tensor | shape | L2 | L∞ | cosine | rel L2 | changed |",
            "|---|---|---|---|---|---|---|",
        ]
        for d in self.top_changed(top, by=by):
            lines.append(
                f"| `{d.name}` | {tuple(d.shape)} | {d.l2:.4g} | {d.linf:.4g} | "
                f"{d.cosine:.6f} | {d.rel_l2:.4g} | {d.changed_fraction:.2%} |"
            )
        return "\n".join(lines)


def compare_models(
    a: str | os.PathLike[str] | Mapping[str, Any] | Any,
    b: str | os.PathLike[str] | Mapping[str, Any] | Any,
    *,
    atol: float = 0.0,
    allow_pickle: bool = False,
    include: Iterable[str] | None = None,
    chunk_elements: int = 1 << 24,
    labels: tuple[str, str] | None = None,
) -> ModelDiff:
    """Diff two checkpoints tensor by tensor.

    The primitive behind "was this naive LM edited?": load the reference and
    the suspect, then report added / removed tensors, shape and dtype
    mismatches, and per-tensor divergence (L2, L∞, cosine, relative L2, mean
    absolute difference, fraction of changed elements).

    Statistics are accumulated in float32 over chunks of at most
    ``chunk_elements`` values, so an 8B-parameter comparison runs in bounded
    memory on CPU and never touches GPU 0.

    .. note::
       Descriptive only — no thresholds, no verdict. Model-editing attacks
       (ROME/MEMIT-style) tend to concentrate their edit in very few tensors,
       so ``top_changed(by="rel_l2")`` is usually the interesting view, but
       deciding what counts as "too much" is a detector's job, not this
       function's.

    Args:
        a: Reference model — path, ``state_dict`` or ``nn.Module``.
        b: Suspect model, same accepted types.
        atol: Absolute tolerance below which an element counts as unchanged.
            ``0.0`` means bit-level "any difference at all".
        allow_pickle: Forwarded to :func:`load_state_dict_for_analysis`; keep
            it ``False`` until the artifact has been scanned.
        include: Optional iterable of tensor names to restrict the comparison to.
        chunk_elements: Chunk size for the streaming statistics.
        labels: Display names for the two sides; defaults to the sources.

    Returns:
        A :class:`ModelDiff`.
    """
    import torch

    sd_a = load_state_dict_for_analysis(a, allow_pickle=allow_pickle)
    sd_b = load_state_dict_for_analysis(b, allow_pickle=allow_pickle)

    left, right = labels or (_label(a), _label(b))
    keys_a, keys_b = set(sd_a), set(sd_b)
    if include is not None:
        wanted = set(include)
        keys_a &= wanted
        keys_b &= wanted

    added = sorted(keys_b - keys_a)
    removed = sorted(keys_a - keys_b)
    shape_mismatches: list[dict] = []
    dtype_mismatches: list[dict] = []
    diffs: list[TensorDiff] = []

    for name in sorted(keys_a & keys_b):
        ta, tb = sd_a[name], sd_b[name]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            continue
        if tuple(ta.shape) != tuple(tb.shape):
            shape_mismatches.append(
                {"name": name, "shape_a": list(ta.shape), "shape_b": list(tb.shape)})
            continue
        if ta.dtype != tb.dtype:
            dtype_mismatches.append({
                "name": name,
                "dtype_a": str(ta.dtype).replace("torch.", ""),
                "dtype_b": str(tb.dtype).replace("torch.", ""),
            })
        diffs.append(_tensor_diff(name, ta, tb, atol=atol, chunk_elements=chunk_elements))

    return ModelDiff(
        left=left, right=right, added=added, removed=removed,
        shape_mismatches=shape_mismatches, dtype_mismatches=dtype_mismatches,
        diffs=diffs, atol=atol,
    )


def _label(obj: Any) -> str:
    if isinstance(obj, (str, os.PathLike)):
        return str(obj)
    return type(obj).__name__


def _tensor_diff(
    name: str,
    ta: Any,
    tb: Any,
    *,
    atol: float,
    chunk_elements: int,
) -> TensorDiff:
    """Streaming float32 statistics for one aligned tensor pair."""
    import math

    import torch

    fa = ta.detach().to("cpu").contiguous().reshape(-1)
    fb = tb.detach().to("cpu").contiguous().reshape(-1)
    numel = int(fa.numel())

    sq_diff = 0.0
    sq_a = 0.0
    sq_b = 0.0
    dot = 0.0
    abs_sum = 0.0
    linf = 0.0
    n_changed = 0

    for start in range(0, max(numel, 1), max(chunk_elements, 1)):
        ca = fa[start:start + chunk_elements].to(torch.float32)
        cb = fb[start:start + chunk_elements].to(torch.float32)
        if ca.numel() == 0:
            break
        d = cb - ca
        ad = d.abs()
        sq_diff += float(torch.sum(d * d))
        sq_a += float(torch.sum(ca * ca))
        sq_b += float(torch.sum(cb * cb))
        dot += float(torch.sum(ca * cb))
        abs_sum += float(torch.sum(ad))
        linf = max(linf, float(ad.max()) if ad.numel() else 0.0)
        n_changed += int(torch.sum(ad > atol))

    norm_a = math.sqrt(sq_a)
    norm_b = math.sqrt(sq_b)
    denom = norm_a * norm_b
    return TensorDiff(
        name=name,
        shape=tuple(int(s) for s in ta.shape),
        numel=numel,
        dtype_a=str(ta.dtype).replace("torch.", ""),
        dtype_b=str(tb.dtype).replace("torch.", ""),
        l2=math.sqrt(sq_diff),
        linf=linf,
        cosine=(dot / denom) if denom > 0 else float("nan"),
        rel_l2=(math.sqrt(sq_diff) / norm_a) if norm_a > 0 else float("nan"),
        mean_abs=(abs_sum / numel) if numel else float("nan"),
        changed_fraction=(n_changed / numel) if numel else float("nan"),
        n_changed=n_changed,
    )


def model_manifest(
    path: str | os.PathLike[str],
    *,
    algo: str = DEFAULT_ALGO,
) -> Manifest:
    """Digest manifest for a model directory or file (thin :mod:`.hashing` alias).

    Provided here so a report can obtain metadata and integrity digests from one
    import.
    """
    return hash_path(path, algo=algo)
