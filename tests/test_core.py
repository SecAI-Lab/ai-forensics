"""Core plumbing: paths, settings, device policy, seeding.

These are the invariants every other module quietly assumes. In particular the
**single-GPU scope** is asserted here rather than left to a comment: on a shared
machine a container that can see extra devices is a misconfiguration, not a
bonus — it will trample someone else's running job.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from aiforensics import config, device, paths, seed


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def test_project_root_contains_the_package() -> None:
    assert (paths.PROJECT_ROOT / "src" / "aiforensics" / "__init__.py").is_file()


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        ("RAW_DIR", "DATA_ROOT"),
        ("PROCESSED_DIR", "DATA_ROOT"),
        ("POISONED_DIR", "DATA_ROOT"),
        ("RAG_DIR", "DATA_ROOT"),
        ("PRETRAINED_DIR", "MODEL_ROOT"),
        ("FINETUNED_DIR", "MODEL_ROOT"),
        ("ADAPTER_DIR", "MODEL_ROOT"),
        ("TAMPERED_DIR", "MODEL_ROOT"),
        ("RUN_DIR", "OUTPUT_ROOT"),
        ("LOG_DIR", "OUTPUT_ROOT"),
        ("REPORT_DIR", "OUTPUT_ROOT"),
    ],
)
def test_managed_dirs_sit_under_their_root(child: str, parent: str) -> None:
    assert getattr(paths, parent) in getattr(paths, child).parents


def test_ensure_dirs_is_idempotent() -> None:
    paths.ensure_dirs()
    paths.ensure_dirs()  # must not raise on the second call
    assert all(d.is_dir() for d in paths.ALL_DIRS)


def test_run_dir_creates_under_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "RUN_DIR", tmp_path)
    made = paths.run_dir("some-experiment")
    assert made == tmp_path / "some-experiment"
    assert made.is_dir()


def test_run_dir_can_skip_creation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "RUN_DIR", tmp_path)
    assert not paths.run_dir("not-created", create=False).exists()


# --------------------------------------------------------------------------- #
# settings
# --------------------------------------------------------------------------- #
def test_settings_defaults_match_the_house_policy() -> None:
    assert config.settings.gpus == "0", "this box is scoped to GPU 0 only"
    assert isinstance(config.settings.seed, int)
    assert config.settings.batch_size > 0
    assert config.settings.num_workers >= 0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfloat16", torch.bfloat16),
        ("bf16", torch.bfloat16),
        ("float16", torch.float16),
        ("fp16", torch.float16),
        ("float32", torch.float32),
        ("fp32", torch.float32),
        ("auto", "auto"),
    ],
)
def test_torch_dtype_mapping(name: str, expected: object) -> None:
    assert config.Settings(dtype=name).torch_dtype() is expected


def test_to_dict_redacts_the_hf_token() -> None:
    dumped = config.Settings(hf_token="hf_supersecret").to_dict()
    assert dumped["hf_token"] == "<redacted>"
    assert "hf_supersecret" not in str(dumped)


def test_to_dict_is_json_shaped() -> None:
    import json

    json.dumps(config.settings.to_dict())  # must not raise


# --------------------------------------------------------------------------- #
# device policy
# --------------------------------------------------------------------------- #
def test_get_device_is_cuda_zero_or_cpu() -> None:
    dev = device.get_device()
    assert dev.type in {"cuda", "cpu"}
    if dev.type == "cuda":
        assert dev.index == 0, "never address a device index other than 0 inside the container"


@pytest.mark.gpu
def test_exactly_one_visible_cuda_device() -> None:
    assert torch.cuda.device_count() == 1, (
        f"{torch.cuda.device_count()} CUDA devices visible; the container must be scoped to "
        "exactly one GPU (see AIF_GPU_ID in .env)"
    )


@pytest.mark.gpu
def test_gpu_info_reports_memory() -> None:
    info = device.gpu_info(0)
    assert info is not None
    assert info.index == 0
    assert info.total_gb > 0
    assert 0 <= info.free_gb <= info.total_gb


def test_gpu_info_is_none_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.gpu_info(0) is None


def test_describe_environment_records_what_a_manifest_needs() -> None:
    env = device.describe_environment()
    assert set(env) >= {
        "torch", "cuda_available", "cuda_version", "visible_devices", "device_count", "gpus",
    }
    assert env["torch"] == torch.__version__
    if env["cuda_available"]:
        assert env["device_count"] == 1
        assert len(env["gpus"]) == 1


def test_free_memory_is_safe_to_call() -> None:
    device.free_memory()


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def test_set_seed_returns_the_seed_used() -> None:
    assert seed.set_seed(123) == 123
    assert seed.set_seed() == seed.DEFAULT_SEED


def test_set_seed_makes_every_rng_reproducible() -> None:
    seed.set_seed(4242)
    first = (random.random(), np.random.rand(), torch.rand(3).tolist())
    seed.set_seed(4242)
    second = (random.random(), np.random.rand(), torch.rand(3).tolist())
    assert first == second


def test_deterministic_mode_sets_the_cudnn_flags() -> None:
    try:
        seed.set_seed(7, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        # Deterministic algorithms are process-wide and measurably slower; do not
        # leak the setting into the rest of the suite.
        torch.use_deterministic_algorithms(False)
        seed.set_seed(1337)
