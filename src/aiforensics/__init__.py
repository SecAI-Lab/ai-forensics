"""aiforensics — a testbed for forensic analysis of tampered AI systems.

Scope (see README.md): build the *evidence* — poisoned datasets, fine-tuned or
edited models, PEFT adapters, poisoned RAG stores, training/inference logs — and
provide the plumbing to evaluate detectors against it. The detection techniques
themselves (ONION, RAP, SVD on hidden states, SAE / logit-lens, RAGDefender,
pickle scanning) are deliberately *not* implemented here; each researcher plugs
theirs into ``aiforensics.detectors``.

Quick start::

    from aiforensics import set_seed, get_device, load_model, load_dataset

    set_seed()
    model = load_model("qwen3-1.7b")
    ds = load_dataset("sst2", split="train")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings, settings
from .device import describe_environment, free_memory, get_device, gpu_info
from .paths import (
    ADAPTER_DIR,
    DATA_ROOT,
    FINETUNED_DIR,
    LOG_DIR,
    MODEL_ROOT,
    OUTPUT_ROOT,
    POISONED_DIR,
    PRETRAINED_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAG_DIR,
    RAW_DIR,
    REPORT_DIR,
    RUN_DIR,
    TAMPERED_DIR,
    ensure_dirs,
    run_dir,
)
from .seed import DEFAULT_SEED, set_seed

__all__ = [
    "__version__",
    # config / device / seed
    "settings", "Settings",
    "get_device", "gpu_info", "describe_environment", "free_memory",
    "set_seed", "DEFAULT_SEED",
    # paths
    "PROJECT_ROOT", "DATA_ROOT", "MODEL_ROOT", "OUTPUT_ROOT",
    "RAW_DIR", "PROCESSED_DIR", "POISONED_DIR", "RAG_DIR",
    "PRETRAINED_DIR", "FINETUNED_DIR", "ADAPTER_DIR", "TAMPERED_DIR",
    "RUN_DIR", "LOG_DIR", "REPORT_DIR",
    "ensure_dirs", "run_dir",
    # lazily re-exported from subpackages (see __getattr__)
    "load_model", "list_models", "load_dataset", "list_datasets",
]

_LAZY = {
    "load_model": ("aiforensics.models", "load_model"),
    "list_models": ("aiforensics.models", "list_models"),
    "load_dataset": ("aiforensics.data", "load_dataset"),
    "list_datasets": ("aiforensics.data", "list_datasets"),
}


def __getattr__(name: str):
    """Defer heavy imports (transformers, datasets) until actually used."""
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
