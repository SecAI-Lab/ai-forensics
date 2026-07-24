"""Model registry: one string key → one reproducible loader.

Forensic work only means something if the artifact under analysis can be named
exactly. Every model this testbed touches is therefore addressed by a short,
stable key (``"resnet18"``, ``"qwen3-1.7b"``) that resolves to a pinned upstream
source and a loader with fixed defaults, so a run manifest that records
``model="qwen3-1.7b"`` is enough to rebuild the same starting point later.

Typical use::

    from aiforensics.models import load_model, list_models

    net = load_model("resnet18", num_classes=43)      # vision -> nn.Module
    model, tok = load_model("qwen3-1.7b")             # llm    -> (model, tokenizer)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: Model families understood by the registry.
FAMILIES: tuple[str, ...] = ("vision", "llm")

__all__ = [
    "FAMILIES",
    "ModelSpec",
    "register_model",
    "get_spec",
    "load_model",
    "list_models",
    "describe_models",
]


@dataclass(frozen=True)
class ModelSpec:
    """Immutable description of one loadable model.

    Attributes:
        key: Registry key, lowercase and hyphenated (``"qwen3-1.7b"``).
        family: ``"vision"`` or ``"llm"``.
        source: Upstream identifier — a HuggingFace repo id for LLMs, a
            ``torchvision`` weights enum member name for vision backbones. This
            is the string that goes into a run manifest.
        loader: Zero-argument-callable-with-kwargs that materialises the model.
            Vision loaders return ``nn.Module``; LLM loaders return
            ``(model, tokenizer)``.
        task: Default task, e.g. ``"image-classification"`` / ``"causal-lm"``.
        num_classes: Default output dimensionality for classifiers, else ``None``.
        size_gb: Approximate on-disk size of the pinned weights, used by
            ``scripts/fetch_models.py`` to warn before a large download.
        description: One line, shown by :func:`describe_models`.
        tags: Free-form markers (``"backbone"``, ``"reasoning"``, ...).
    """

    key: str
    family: str
    source: str
    loader: Callable[..., Any]
    task: str
    num_classes: int | None = None
    size_gb: float = 0.0
    description: str = ""
    tags: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(
                f"unknown family {self.family!r} for model {self.key!r}; "
                f"expected one of {', '.join(FAMILIES)}"
            )
        if not callable(self.loader):
            raise TypeError(f"loader for {self.key!r} is not callable")

    def to_dict(self) -> dict[str, Any]:
        """Manifest-friendly view (drops the non-serialisable loader)."""
        return {
            "key": self.key,
            "family": self.family,
            "source": self.source,
            "task": self.task,
            "num_classes": self.num_classes,
            "size_gb": self.size_gb,
            "description": self.description,
            "tags": list(self.tags),
        }


_REGISTRY: dict[str, ModelSpec] = {}
_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import the built-in loader modules exactly once.

    ``vision`` and ``llm`` register themselves at import time. They are imported
    lazily so that ``import aiforensics`` stays cheap and never depends on
    ``transformers`` being installed.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True  # set first: the submodules import this module back
    from importlib import import_module

    for name in ("vision", "llm"):
        import_module(f"{__package__}.{name}")


def register_model(spec: ModelSpec, *, overwrite: bool = False) -> ModelSpec:
    """Add a spec to the registry.

    Args:
        spec: The spec to register.
        overwrite: Allow replacing an existing key. Off by default so a typo
            cannot silently shadow a pinned reference model.

    Returns:
        The registered spec, for convenient chaining.

    Raises:
        KeyError: If ``spec.key`` is already registered and ``overwrite`` is False.
    """
    if spec.key in _REGISTRY and not overwrite:
        raise KeyError(
            f"model key {spec.key!r} is already registered "
            f"(source={_REGISTRY[spec.key].source!r}); pass overwrite=True to replace it"
        )
    _REGISTRY[spec.key] = spec
    return spec


def get_spec(key: str) -> ModelSpec:
    """Look up a :class:`ModelSpec`.

    Raises:
        KeyError: If ``key`` is unknown; the message lists every valid key.
    """
    _ensure_builtins()
    norm = key.strip().lower()
    try:
        return _REGISTRY[norm]
    except KeyError:
        raise KeyError(
            f"unknown model key {key!r}. Valid keys: {', '.join(sorted(_REGISTRY))}"
        ) from None


def load_model(key: str, **kwargs: Any) -> Any:
    """Load a model by registry key — the single entry point used everywhere.

    Args:
        key: Registry key, see :func:`list_models`.
        **kwargs: Forwarded to the family loader
            (:func:`aiforensics.models.vision.load_resnet` or
            :func:`aiforensics.models.llm.load_llm`).

    Returns:
        ``nn.Module`` for the ``vision`` family, ``(model, tokenizer)`` for the
        ``llm`` family.

    Raises:
        KeyError: If ``key`` is unknown.
    """
    spec = get_spec(key)
    logger.info("loading model %s (%s, source=%s)", spec.key, spec.family, spec.source)
    return spec.loader(**kwargs)


def list_models(family: str | None = None) -> list[str]:
    """Sorted registry keys, optionally restricted to one family.

    Raises:
        ValueError: If ``family`` is not one of :data:`FAMILIES`.
    """
    _ensure_builtins()
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {', '.join(FAMILIES)}")
    return sorted(k for k, s in _REGISTRY.items() if family is None or s.family == family)


def describe_models(family: str | None = None) -> Any:
    """Human-readable inventory of the registry.

    Returns:
        A ``pandas.DataFrame`` when pandas is importable (nice in notebooks and
        in ``scripts/fetch_models.py``), otherwise a list of dicts with the same
        columns. Either way the rows are :meth:`ModelSpec.to_dict` output.
    """
    rows = [get_spec(k).to_dict() for k in list_models(family)]
    try:
        import pandas as pd
    except Exception:  # noqa: BLE001 - display helper must never break a run
        logger.debug("pandas unavailable; describe_models() returning plain dicts", exc_info=True)
        return rows
    return pd.DataFrame(rows, columns=list(rows[0]) if rows else None)
