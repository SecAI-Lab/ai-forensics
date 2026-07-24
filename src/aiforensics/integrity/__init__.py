"""Artifact integrity: is this the model/dataset I think it is, and is it safe to open?

Three layers, in the order you should use them on an unknown artifact:

1. :mod:`~aiforensics.integrity.serialization` — **static** triage first.
   Never ``torch.load`` / ``pickle.load`` something under investigation;
   scan it (``scan_pickle``, ``scan_dir``, ``list_pickle_opcodes``) and prefer
   ``safe_load`` for safetensors.
2. :mod:`~aiforensics.integrity.hashing` — pin the artifact to a digest
   (``hash_file``, ``hash_dir``, ``hash_state_dict``, ``verify``) so every later
   claim is anchored to specific bytes.
3. :mod:`~aiforensics.integrity.metadata` — describe it (``model_metadata``,
   ``safetensors_header``) and diff it against a reference
   (``compare_models``).

Typical "naive LM vs. tampered LM" session::

    from aiforensics.integrity import scan_dir, hash_dir, compare_models

    for r in scan_dir("models/adapters/suspect"):     # static, no deserialisation
        print(r.summary())
    baseline = hash_dir("models/pretrained/qwen3-1.7b")
    baseline.save("outputs/reports/qwen3-1.7b.manifest.json")
    diff = compare_models("models/pretrained/qwen3-1.7b", "models/tampered/qwen3-1.7b-edited")
    print(diff.to_markdown())

``compare_models`` is descriptive plumbing only — it reports *what* differs, it
does not decide whether that constitutes tampering. Detection lives in
:mod:`aiforensics.detectors`.
"""

from __future__ import annotations

from .hashing import (
    DEFAULT_ALGO,
    DEFAULT_CHUNK_SIZE,
    Manifest,
    Mismatch,
    VerifyResult,
    aggregate_digest,
    hash_bytes,
    hash_dir,
    hash_file,
    hash_path,
    hash_state_dict,
    load_manifest,
    verify,
)
from .metadata import (
    ModelDiff,
    TensorDiff,
    compare_models,
    detect_format,
    dtype_histogram,
    load_state_dict_for_analysis,
    model_manifest,
    model_metadata,
    resolve_hf_commit,
    safetensors_header,
    safetensors_metadata,
)
from .serialization import (
    PICKLE_SUFFIXES,
    SAFE_SUFFIXES,
    Finding,
    ScanResult,
    Severity,
    is_safetensors,
    list_pickle_opcodes,
    safe_load,
    scan_dir,
    scan_pickle,
)

__all__ = [
    # hashing
    "DEFAULT_ALGO", "DEFAULT_CHUNK_SIZE",
    "Manifest", "Mismatch", "VerifyResult",
    "aggregate_digest", "hash_bytes", "hash_dir", "hash_file", "hash_path",
    "hash_state_dict", "load_manifest", "verify",
    # metadata
    "ModelDiff", "TensorDiff",
    "compare_models", "detect_format", "dtype_histogram",
    "load_state_dict_for_analysis", "model_manifest", "model_metadata",
    "resolve_hf_commit", "safetensors_header", "safetensors_metadata",
    # serialization
    "PICKLE_SUFFIXES", "SAFE_SUFFIXES",
    "Finding", "ScanResult", "Severity",
    "is_safetensors", "list_pickle_opcodes", "safe_load", "scan_dir", "scan_pickle",
]
