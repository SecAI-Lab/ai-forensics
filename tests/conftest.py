"""Shared pytest fixtures.

The suite is **fast and offline by default**. Nothing here downloads a model or
a dataset: every fixture builds its evidence in-process (a synthetic HF
``Dataset``, a hand-made ``state_dict``, a pickle payload written into
``tmp_path``). Tests that genuinely need the network or a real checkpoint are
marked ``@pytest.mark.slow`` and are skipped unless ``--runslow`` is passed, so
``python -m pytest tests/`` stays a few seconds.

Markers (registered in ``pyproject.toml``):

``slow``
    Needs a real model/dataset download or several seconds of compute.
``gpu``
    Needs a visible CUDA device; skipped automatically when there is none.
``network``
    Reaches the HuggingFace hub.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import pytest

# --------------------------------------------------------------------------- #
# collection / CLI
# --------------------------------------------------------------------------- #
_SLOW_MARKERS = ("slow", "network")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--runslow``: opt in to the download/compute-heavy tests."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked slow/network (real model or dataset downloads)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow/network tests unless ``--runslow``, and gpu tests without CUDA.

    The default fast path is what ``make test`` runs, so a green suite never
    depends on the hub being reachable. ``-m "not slow"`` keeps working
    unchanged for anyone who prefers the marker expression.
    """
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="needs --runslow (downloads a model/dataset)")
    skip_gpu = pytest.mark.skip(reason="no visible CUDA device")

    has_cuda = False
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # pragma: no cover - torch is a hard dep in the container
        has_cuda = False

    for item in items:
        if not run_slow and any(m in item.keywords for m in _SLOW_MARKERS):
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def _seeded() -> int:
    """Seed python/numpy/torch before every test.

    A forensic assertion that only holds for one RNG state is not an assertion,
    so this is autouse rather than opt-in.
    """
    from aiforensics import set_seed

    return set_seed(1337)


# --------------------------------------------------------------------------- #
# synthetic datasets
# --------------------------------------------------------------------------- #
#: Row count of the synthetic corpora. Small enough to be instant, large enough
#: that a 25% poison rate still leaves a meaningful clean majority.
N_ROWS = 40


@pytest.fixture
def text_records() -> list[dict[str, Any]]:
    """A dependency-free corpus: ``[{"text": ..., "label": 0|1}, ...]``.

    ``poison_dataset`` accepts a list of dicts, so this exercises the same code
    path as an HF ``Dataset`` without importing ``datasets``.
    """
    return [
        {"text": f"clean sentence number {i} about something ordinary", "label": i % 2}
        for i in range(N_ROWS)
    ]


@pytest.fixture
def tiny_dataset(text_records: list[dict[str, Any]]) -> Any:
    """The same corpus as a ``datasets.Dataset`` with ``text``/``label`` columns.

    Built in memory with ``Dataset.from_dict`` — no Arrow cache file is written,
    so the shared HF cache is never touched by the test suite.
    """
    datasets = pytest.importorskip("datasets", reason="the `datasets` package is required")
    return datasets.Dataset.from_dict(
        {
            "text": [r["text"] for r in text_records],
            "label": [r["label"] for r in text_records],
        }
    )


@pytest.fixture
def poisoned_pair(tiny_dataset: Any) -> tuple[Any, Any]:
    """``(poisoned_dataset, manifest)`` for a 25% dirty-label BadNets attack.

    The trigger is appended at a fixed position so a test can recover the
    poisoned rows from the surface form and compare that against the manifest —
    the two must agree, or the ground truth is wrong.
    """
    from aiforensics.data.poison import make_text_trigger_fn, poison_dataset

    return poison_dataset(
        tiny_dataset,
        fraction=0.25,
        target_label=1,
        trigger_fn=make_text_trigger_fn("cf", position="end"),
        seed=7,
        dataset_key="synthetic",
        split="train",
        attack="badnets-nlp",
    )


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
@pytest.fixture
def small_state_dict() -> dict[str, Any]:
    """A three-tensor ``state_dict`` standing in for a checkpoint."""
    import torch

    torch.manual_seed(0)
    return {
        "layer1.weight": torch.randn(4, 5),
        "layer1.bias": torch.zeros(4),
        "layer2.weight": torch.randn(3, 4),
    }


#: Tensor and element edited by :func:`tampered_state_dict`. A *single* element
#: is the point: a model edit (ROME/MEMIT-style) touches very little, and a
#: digest that only catches bulk changes is useless against it.
TAMPERED_TENSOR = "layer2.weight"
TAMPERED_INDEX = (1, 2)
TAMPERED_DELTA = 0.05


@pytest.fixture
def tampered_state_dict(small_state_dict: dict[str, Any]) -> dict[str, Any]:
    """A copy of :func:`small_state_dict` with exactly one element changed."""
    tampered = {name: tensor.clone() for name, tensor in small_state_dict.items()}
    tampered[TAMPERED_TENSOR][TAMPERED_INDEX] += TAMPERED_DELTA
    return tampered


@pytest.fixture
def safetensors_file(tmp_path: Path, small_state_dict: dict[str, Any]) -> Path:
    """``small_state_dict`` written as a real safetensors container."""
    save_file = pytest.importorskip("safetensors.torch", reason="safetensors required").save_file
    path = tmp_path / "weights.safetensors"
    save_file({k: v.contiguous() for k, v in small_state_dict.items()}, str(path))
    return path


# --------------------------------------------------------------------------- #
# serialized artifacts
# --------------------------------------------------------------------------- #
class _ReduceOsSystem:
    """Pickle payload whose ``__reduce__`` calls ``os.system`` on load.

    This is the on-device threat model in miniature: unpickling *is* execution,
    so a tampered serialized adapter owns the device the moment someone calls
    ``torch.load``. The class is only ever **serialised** here — nothing in the
    suite unpickles it, and nothing should.
    """

    def __reduce__(self) -> tuple:
        return (os.system, ("echo pwned-by-a-pickle",))


@pytest.fixture
def malicious_pickle(tmp_path: Path) -> Path:
    """Path to a pickle containing an ``os.system`` ``__reduce__`` payload."""
    path = tmp_path / "malicious_adapter.pkl"
    path.write_bytes(pickle.dumps(_ReduceOsSystem()))
    return path


@pytest.fixture
def benign_pickle(tmp_path: Path) -> Path:
    """Path to a pickle holding plain data and no imports."""
    path = tmp_path / "benign.pkl"
    path.write_bytes(pickle.dumps({"alpha": [1, 2, 3], "beta": "plain data"}))
    return path
