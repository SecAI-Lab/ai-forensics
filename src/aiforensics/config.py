"""Runtime settings, read once from the environment.

Import :data:`settings` rather than calling ``os.environ`` from library code.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from .seed import DEFAULT_SEED


def _env_bool(var: str, default: bool = False) -> bool:
    raw = os.environ.get(var)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(var: str, default: int) -> int:
    raw = os.environ.get(var)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass
class Settings:
    """Process-wide knobs. All overridable via environment variables."""

    seed: int = field(default_factory=lambda: _env_int("AIF_SEED", DEFAULT_SEED))
    deterministic: bool = field(default_factory=lambda: _env_bool("AIF_DETERMINISTIC", False))

    #: Which physical GPUs the container may touch. Policy on this box: "0".
    gpus: str = field(default_factory=lambda: os.environ.get("AIF_GPUS", "0"))

    #: Default dtype for LLM loading: "bfloat16" | "float16" | "float32" | "auto".
    dtype: str = field(default_factory=lambda: os.environ.get("AIF_DTYPE", "bfloat16"))

    num_workers: int = field(default_factory=lambda: _env_int("AIF_NUM_WORKERS", 8))
    batch_size: int = field(default_factory=lambda: _env_int("AIF_BATCH_SIZE", 32))

    #: Never reach the network; fail instead. Useful for reproducing a frozen setup.
    offline: bool = field(default_factory=lambda: _env_bool("HF_HUB_OFFLINE", False))

    hf_token: str | None = field(
        default_factory=lambda: os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    def torch_dtype(self) -> Any:
        import torch

        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": "auto",
        }[self.dtype.lower()]

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("hf_token"):
            d["hf_token"] = "<redacted>"
        return d


settings = Settings()
