"""PEFT adapter plumbing for the on-device threat model.

The on-device attacker does not ship a whole model — they ship a small LoRA
adapter, either *behaviourally* tampered (fine-tuned on poisoned data) or
*structurally* tampered (a pickle-based ``adapter_model.bin`` that executes code
the moment it is deserialised). This module provides what the analysis needs:

* attach an adapter to a pristine base model (:func:`attach_adapter`),
* merge it so the result can be diffed weight-by-weight against that base
  (:func:`merge_adapter`),
* and inspect an adapter file **without deserialising it**
  (:func:`adapter_summary`).

That last point is the important one. ``torch.load`` on an untrusted ``.bin``
*is* the attack; this module therefore reads safetensors headers only, and for
pickle-based files reports the format and refuses, pointing at
``aiforensics.integrity.serialization`` for scanning.

``peft`` is imported lazily so ``import aiforensics`` works on a machine without it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from math import prod
from pathlib import Path
from typing import Any

from ..paths import ADAPTER_DIR

logger = logging.getLogger(__name__)

#: LoRA targets for Qwen3: all attention projections plus the MLP. This is the
#: standard "everything linear" recipe and the one an attacker would reach for,
#: since it maximises behavioural change per trainable parameter.
QWEN3_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

#: Cheaper variant: attention only. Useful as a contrast condition when studying
#: *where* a backdoor has to live to be effective.
QWEN3_ATTENTION_ONLY_MODULES: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

#: Candidate weight filenames inside an adapter directory, in preference order.
_SAFE_WEIGHT_NAMES = ("adapter_model.safetensors", "model.safetensors")
_PICKLE_WEIGHT_NAMES = ("adapter_model.bin", "pytorch_model.bin", "adapter_model.pt")
_PICKLE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl")

_SERIALIZATION_MODULE = "aiforensics.integrity.serialization"

__all__ = [
    "QWEN3_LORA_TARGET_MODULES",
    "QWEN3_ATTENTION_ONLY_MODULES",
    "peft_available",
    "make_lora_config",
    "attach_adapter",
    "load_adapter_config",
    "merge_adapter",
    "adapter_summary",
    "find_adapters",
]


def _require_peft() -> Any:
    """Import ``peft`` or fail with an actionable message."""
    try:
        import peft
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "peft is required for adapter work but is not installed. It is pinned in "
            "docker/requirements.txt — run inside the container, or `pip install peft`."
        ) from exc
    return peft


def peft_available() -> bool:
    """Whether ``peft`` can be imported right now."""
    try:
        _require_peft()
    except ImportError:
        return False
    return True


def make_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    *,
    target_modules: Sequence[str] | None = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
    **kwargs: Any,
) -> Any:
    """Build a ``peft.LoraConfig`` with defaults that suit Qwen3.

    Defaults are the common ``r=16, alpha=32`` (scale 2.0) recipe over every
    linear projection — enough capacity to install a trigger→target association
    in a few hundred steps, which is the attacker capability being modelled.

    Args:
        r: LoRA rank.
        lora_alpha: Scaling numerator; effective scale is ``alpha / r``.
        lora_dropout: Dropout on the LoRA path.
        target_modules: Module name suffixes to adapt; defaults to
            :data:`QWEN3_LORA_TARGET_MODULES`.
        bias: ``peft`` bias mode.
        task_type: ``peft`` task type.
        **kwargs: Forwarded to ``LoraConfig``.

    Returns:
        ``peft.LoraConfig``.

    Raises:
        ImportError: If ``peft`` is unavailable.
    """
    peft = _require_peft()
    return peft.LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules or QWEN3_LORA_TARGET_MODULES),
        bias=bias,
        task_type=task_type,
        **kwargs,
    )


def _resolve_weight_file(path: Path) -> tuple[Path | None, str]:
    """Find the adapter weight file under ``path`` and classify its format.

    Returns:
        ``(file or None, format)`` where format is ``"safetensors"``,
        ``"pytorch-pickle"`` or ``"unknown"``.
    """
    if path.is_file():
        if path.suffix == ".safetensors":
            return path, "safetensors"
        if path.suffix in _PICKLE_SUFFIXES:
            return path, "pytorch-pickle"
        return path, "unknown"

    for name in _SAFE_WEIGHT_NAMES:
        if (path / name).is_file():
            return path / name, "safetensors"
    for name in _PICKLE_WEIGHT_NAMES:
        if (path / name).is_file():
            return path / name, "pytorch-pickle"

    loose_safe = sorted(path.glob("*.safetensors"))
    if loose_safe:
        return loose_safe[0], "safetensors"
    for suffix in _PICKLE_SUFFIXES:
        loose = sorted(path.glob(f"*{suffix}"))
        if loose:
            return loose[0], "pytorch-pickle"
    return None, "unknown"


def load_adapter_config(path: str | Path) -> dict[str, Any]:
    """Read ``adapter_config.json`` — plain JSON, never a pickle.

    The config alone already carries forensic signal: the base model the adapter
    claims to belong to, the rank, and which modules were touched.

    Args:
        path: Adapter directory, or the ``adapter_config.json`` file itself.

    Returns:
        The parsed config dict.

    Raises:
        FileNotFoundError: If no config is found.
        ValueError: If the file is not valid JSON.
    """
    p = Path(path).expanduser()
    config_file = p if p.is_file() and p.suffix == ".json" else p / "adapter_config.json"
    if not config_file.is_file():
        raise FileNotFoundError(
            f"no adapter_config.json at {config_file}; is {p} really a PEFT adapter directory?"
        )
    try:
        return json.loads(config_file.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_file} is not valid JSON: {exc}") from exc


def attach_adapter(
    model: Any,
    adapter_path: str | Path,
    *,
    adapter_name: str = "default",
    is_trainable: bool = False,
    allow_pickle: bool = False,
    **kwargs: Any,
) -> Any:
    """Attach a PEFT adapter to an already-loaded base model.

    Args:
        model: The pristine base model (from
            :func:`aiforensics.models.llm.load_llm`).
        adapter_path: Directory containing ``adapter_config.json`` and weights.
        adapter_name: Name to register the adapter under; load several under
            different names to compare a clean and a tampered adapter side by side.
        is_trainable: Keep LoRA parameters trainable (needed only to continue
            training).
        allow_pickle: Permit loading a pickle-based ``.bin`` adapter. **Off by
            default**: deserialising an untrusted adapter is itself one of the
            attacks under study, so it must be an explicit, deliberate act taken
            after :func:`adapter_summary` and a scan.
        **kwargs: Forwarded to ``PeftModel.from_pretrained``.

    Returns:
        A ``peft.PeftModel`` wrapping ``model``.

    Raises:
        FileNotFoundError: If ``adapter_path`` does not exist.
        PermissionError: If the adapter is pickle-based and ``allow_pickle`` is False.
        ImportError: If ``peft`` is unavailable.
    """
    peft = _require_peft()
    path = Path(adapter_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"adapter not found: {path}")

    weight_file, fmt = _resolve_weight_file(path)
    if fmt == "pytorch-pickle" and not allow_pickle:
        raise PermissionError(
            f"{weight_file} is a pickle-based checkpoint; loading it executes arbitrary "
            f"code. Inspect it first with aiforensics.models.peft_utils.adapter_summary() "
            f"and scan it via {_SERIALIZATION_MODULE}. Pass allow_pickle=True only once "
            f"you have decided it is safe (or you are running it in a throwaway sandbox)."
        )

    logger.info("attaching adapter %s (%s) as %r", path, fmt, adapter_name)
    return peft.PeftModel.from_pretrained(
        model, str(path), adapter_name=adapter_name, is_trainable=is_trainable, **kwargs
    )


def merge_adapter(model: Any, *, safe_merge: bool = True, progressbar: bool = False) -> Any:
    """Fold adapter deltas into the base weights and unwrap the ``PeftModel``.

    The point for this testbed is comparability: a merged model has exactly the
    same parameter names and shapes as the pristine base, so a weight diff
    localises the tampering to specific layers instead of drowning in the
    adapter's own extra tensors.

    Args:
        model: A ``PeftModel`` (from :func:`attach_adapter`).
        safe_merge: Check the merged weights for NaN/Inf before committing.
        progressbar: Show ``peft``'s merge progress bar.

    Returns:
        The merged base ``nn.Module``.

    Raises:
        TypeError: If ``model`` is not a merge-capable ``PeftModel``.
    """
    if not hasattr(model, "merge_and_unload"):
        raise TypeError(
            f"{type(model).__name__} is not a PeftModel; call attach_adapter(base, path) first"
        )
    logger.info("merging adapter into base weights (safe_merge=%s)", safe_merge)
    return model.merge_and_unload(safe_merge=safe_merge, progressbar=progressbar)


def _serialization_hint() -> tuple[bool, str]:
    """Whether the integrity scanner exists, plus a pointer to it.

    Imported lazily and tolerantly: ``aiforensics.integrity.serialization`` is
    owned by another part of the project and may not be present yet.
    """
    try:
        from importlib import import_module

        import_module(_SERIALIZATION_MODULE)
    except Exception:
        return False, (
            f"{_SERIALIZATION_MODULE} is not available; scan this file with picklescan "
            "or fickling before deserialising it."
        )
    return True, f"scan this file with {_SERIALIZATION_MODULE} before deserialising it."


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_summary(adapter_path: str | Path, *, compute_hash: bool = True) -> dict[str, Any]:
    """Describe an adapter **without deserialising it**.

    For safetensors the tensor index lives in a JSON header, so names, shapes and
    dtypes can be read with zero risk. For pickle-based ``.bin`` files there is
    no safe way to enumerate tensors — unpickling *is* execution — so this
    reports the format, refuses to load, and points at the scanner.

    Args:
        adapter_path: Adapter directory or a single weight file.
        compute_hash: Include a SHA-256 of the weight file (adapters are small,
            and a digest is what makes an evidence claim reproducible).

    Returns:
        A dict with keys:

        * ``path``, ``weight_file``, ``format`` (``safetensors`` /
          ``pytorch-pickle`` / ``unknown``), ``file_size_bytes``, ``sha256``
        * ``safe_to_inspect`` — whether tensor metadata could be read
        * ``n_tensors``, ``n_parameters``, ``tensors`` (name/shape/dtype/n_params),
          ``dtypes`` (dtype -> tensor count), ``metadata`` (safetensors header
          metadata) — empty for pickle files
        * ``config`` — parsed ``adapter_config.json`` if present
        * ``files`` — every file in the directory with its size
        * ``notes`` — human-readable warnings, including the scanner pointer

    Raises:
        FileNotFoundError: If the path does not exist or holds no weight file.
    """
    path = Path(adapter_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"adapter not found: {path}")

    notes: list[str] = []
    summary: dict[str, Any] = {
        "path": str(path),
        "weight_file": None,
        "format": "unknown",
        "file_size_bytes": None,
        "sha256": None,
        "safe_to_inspect": False,
        "n_tensors": 0,
        "n_parameters": 0,
        "tensors": [],
        "dtypes": {},
        "metadata": {},
        "config": None,
        "files": [],
        "notes": notes,
    }

    if path.is_dir():
        summary["files"] = [
            {"name": f.name, "size_bytes": f.stat().st_size}
            for f in sorted(path.iterdir())
            if f.is_file()
        ]
        try:
            summary["config"] = load_adapter_config(path)
        except (FileNotFoundError, ValueError) as exc:
            notes.append(str(exc))

    weight_file, fmt = _resolve_weight_file(path)
    summary["format"] = fmt
    if weight_file is None:
        raise FileNotFoundError(
            f"no adapter weight file under {path} "
            f"(looked for {', '.join(_SAFE_WEIGHT_NAMES + _PICKLE_WEIGHT_NAMES)})"
        )
    summary["weight_file"] = str(weight_file)
    summary["file_size_bytes"] = weight_file.stat().st_size
    if compute_hash:
        summary["sha256"] = _sha256(weight_file)

    if fmt == "safetensors":
        from safetensors import safe_open

        tensors: list[dict[str, Any]] = []
        dtypes: dict[str, int] = {}
        total_params = 0
        with safe_open(str(weight_file), framework="pt") as handle:
            summary["metadata"] = dict(handle.metadata() or {})
            for name in handle.keys():
                sliced = handle.get_slice(name)
                shape = list(sliced.get_shape())
                dtype = str(sliced.get_dtype())
                n_params = int(prod(shape)) if shape else 1
                total_params += n_params
                dtypes[dtype] = dtypes.get(dtype, 0) + 1
                tensors.append(
                    {"name": name, "shape": shape, "dtype": dtype, "n_params": n_params}
                )
        summary.update(
            safe_to_inspect=True,
            n_tensors=len(tensors),
            n_parameters=total_params,
            tensors=tensors,
            dtypes=dtypes,
        )
        notes.append("safetensors: header parsed without executing any code.")
    else:
        available, hint = _serialization_hint()
        summary["scanner_available"] = available
        notes.append(
            f"{weight_file.name} is {'a pickle-based' if fmt == 'pytorch-pickle' else 'an unrecognised'} "
            "checkpoint. Refusing to deserialise it: unpickling runs arbitrary code, which is "
            "exactly the on-device tampering scenario under study."
        )
        notes.append(hint)
        logger.warning("refused to deserialise %s (format=%s)", weight_file, fmt)

    return summary


def find_adapters(root: str | Path = ADAPTER_DIR) -> list[Path]:
    """List adapter directories under ``root`` (anything with an adapter config).

    Args:
        root: Directory to search; defaults to ``models/adapters``. Point it at
            ``models/tampered`` to enumerate the malicious artifacts instead.

    Returns:
        Sorted directory paths; empty if ``root`` does not exist.
    """
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    return sorted({cfg.parent for cfg in base.rglob("adapter_config.json")})
