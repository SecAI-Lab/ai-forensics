"""GPU selection and reporting.

This project is scoped to a **single GPU**: the container is started with
``CUDA_VISIBLE_DEVICES`` restricted to one device, so torch sees exactly one
device, addressed as ``cuda:0`` — whichever physical GPU was selected.

On a shared machine, taking every visible device will trample someone else's
running job. Never hard-code a device index anywhere; select the physical GPU
with ``AIF_GPU_ID`` in ``.env`` instead.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_gb: float
    allocated_gb: float
    reserved_gb: float
    free_gb: float

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"cuda:{self.index} {self.name} | "
            f"{self.free_gb:.1f}/{self.total_gb:.1f} GiB free "
            f"(alloc {self.allocated_gb:.1f}, reserved {self.reserved_gb:.1f})"
        )


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the device to run on: ``cuda:0`` when available, else ``cpu``."""
    if prefer_cuda and torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            warnings.warn(
                f"{n} CUDA devices are visible, but this project is scoped to a "
                "single GPU. Set CUDA_VISIBLE_DEVICES=0 (see .env / AIF_GPUS).",
                RuntimeWarning,
                stacklevel=2,
            )
        return torch.device("cuda:0")
    return torch.device("cpu")


def gpu_info(index: int = 0) -> GPUInfo | None:
    """Memory snapshot for a visible device, or ``None`` without CUDA."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(index)
    total = props.total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(index) / 1024**3
    reserved = torch.cuda.memory_reserved(index) / 1024**3
    free_bytes, _ = torch.cuda.mem_get_info(index)
    return GPUInfo(
        index=index,
        name=props.name,
        total_gb=total,
        allocated_gb=allocated,
        reserved_gb=reserved,
        free_gb=free_bytes / 1024**3,
    )


def describe_environment() -> dict:
    """Everything worth recording in a run manifest about the compute environment."""
    info = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    if torch.cuda.is_available():
        info["gpus"] = [str(gpu_info(i)) for i in range(torch.cuda.device_count())]
    return info


def free_memory() -> None:
    """Drop cached blocks. Call between loading two large checkpoints."""
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
