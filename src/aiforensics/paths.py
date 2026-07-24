"""Canonical filesystem layout.

Every path is overridable by an environment variable so the same code runs
identically inside the container (where the project is bind-mounted at
``/workspace``) and on the host.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


#: Project root. Inside the container this is ``/workspace``.
PROJECT_ROOT: Path = _env_path("AIF_PROJECT_ROOT", Path(__file__).resolve().parents[2])

DATA_ROOT: Path = _env_path("AIF_DATA_ROOT", PROJECT_ROOT / "data")
MODEL_ROOT: Path = _env_path("AIF_MODEL_ROOT", PROJECT_ROOT / "models")
OUTPUT_ROOT: Path = _env_path("AIF_OUTPUT_ROOT", PROJECT_ROOT / "outputs")

# --- data ---------------------------------------------------------------
RAW_DIR: Path = DATA_ROOT / "raw"            # untouched downloads
PROCESSED_DIR: Path = DATA_ROOT / "processed"  # clean, normalised splits
POISONED_DIR: Path = DATA_ROOT / "poisoned"    # attacker-produced evidence
RAG_DIR: Path = DATA_ROOT / "rag"              # corpora + vector indexes

# --- models -------------------------------------------------------------
PRETRAINED_DIR: Path = MODEL_ROOT / "pretrained"  # as-published reference weights
FINETUNED_DIR: Path = MODEL_ROOT / "finetuned"    # full fine-tunes (on-prem case)
ADAPTER_DIR: Path = MODEL_ROOT / "adapters"       # PEFT adapters (on-device case)
TAMPERED_DIR: Path = MODEL_ROOT / "tampered"      # edited / malicious artifacts

# --- outputs ------------------------------------------------------------
RUN_DIR: Path = OUTPUT_ROOT / "runs"        # per-experiment artifacts
LOG_DIR: Path = OUTPUT_ROOT / "logs"        # train/inference evidence logs
REPORT_DIR: Path = OUTPUT_ROOT / "reports"  # metric tables, figures

#: Shared HuggingFace cache (bind-mounted from the host to avoid re-downloads).
HF_CACHE: Path = _env_path("HF_HOME", Path.home() / ".cache" / "huggingface")

ALL_DIRS = (
    RAW_DIR, PROCESSED_DIR, POISONED_DIR, RAG_DIR,
    PRETRAINED_DIR, FINETUNED_DIR, ADAPTER_DIR, TAMPERED_DIR,
    RUN_DIR, LOG_DIR, REPORT_DIR,
)


def ensure_dirs() -> None:
    """Create every managed directory. Idempotent; safe to call at import time."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def run_dir(name: str, create: bool = True) -> Path:
    """Return ``outputs/runs/<name>``, creating it by default."""
    d = RUN_DIR / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
