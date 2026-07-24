"""Datasets, RAG stores, and the poisoning primitives that produce evidence.

Layout
------
=============  ============================================================
``registry``   String key -> loader. ``load_dataset`` / ``list_datasets`` /
               ``get_spec`` / ``describe_datasets``.
``vision``     CIFAR-10, GTSRB, LFW as torch Datasets yielding
               ``(image_tensor, label)``.
``text``       SST-2, AG News, IMDB as HF Datasets with ``text`` / ``label``.
``rag``        BeIR/NQ corpus + queries + qrels, sentence-transformers
               embeddings, a FAISS store that tracks injected documents.
``poison``     Attacker-side trigger primitives and the
               :class:`~aiforensics.data.poison.PoisonManifest` ground truth.
=============  ============================================================

Everything except the registry is imported lazily, so ``import aiforensics``
still works on a machine without torchvision / datasets / faiss.

Naming hazard: :func:`load_dataset` here is *not* ``datasets.load_dataset``.
Inside this package the HuggingFace one is always aliased::

    from datasets import load_dataset as hf_load_dataset
"""

from __future__ import annotations

import importlib
from typing import Any

from .registry import (
    CIFAR10_CLASSES,
    GTSRB_CLASSES,
    MODALITIES,
    DatasetSpec,
    describe_datasets,
    get_spec,
    list_datasets,
    load_dataset,
    register_dataset,
)

__all__ = [
    # registry
    "DatasetSpec", "MODALITIES", "CIFAR10_CLASSES", "GTSRB_CLASSES",
    "describe_datasets", "get_spec", "list_datasets", "load_dataset",
    "register_dataset",
    # vision
    "ArrayImageDataset", "ImageDatasetWrapper", "DEFAULT_IMAGE_SIZE",
    "build_transforms", "denormalize", "get_dataloader",
    "load_cifar10", "load_gtsrb", "load_lfw",
    # text
    "TEXT_COLUMN", "LABEL_COLUMN", "UNLABELED_SPLITS",
    "load_sst2", "load_ag_news", "load_imdb",
    "normalise_columns", "subsample", "tokenize_dataset",
    # rag
    "DEFAULT_ENCODER", "DEFAULT_MAX_DOCS", "Document", "Query", "RagStore",
    "SearchResult", "build_index", "embed_texts", "encode_documents",
    "load_corpus", "load_encoder", "load_qrels", "load_queries",
    # poison
    "PHRASE_TRIGGERS", "RARE_WORD_TRIGGERS", "POISON_TEMPLATES",
    "PoisonManifest", "TriggerSpec",
    "apply_patch_trigger", "blend_trigger", "craft_poison_passages",
    "insert_trigger", "list_manifests", "load_manifest",
    "make_blend_trigger_fn", "make_checkerboard_patch",
    "make_patch_trigger_fn", "make_text_trigger_fn", "poison_dataset",
]

#: attribute name -> submodule holding it.
_LAZY: dict[str, str] = {}
for _mod, _names in {
    "vision": (
        "ArrayImageDataset", "ImageDatasetWrapper", "DEFAULT_IMAGE_SIZE",
        "build_transforms", "denormalize", "get_dataloader",
        "load_cifar10", "load_gtsrb", "load_lfw",
    ),
    "text": (
        "TEXT_COLUMN", "LABEL_COLUMN", "UNLABELED_SPLITS",
        "load_sst2", "load_ag_news", "load_imdb",
        "normalise_columns", "subsample", "tokenize_dataset",
    ),
    "rag": (
        "DEFAULT_ENCODER", "DEFAULT_MAX_DOCS", "Document", "Query", "RagStore",
        "SearchResult", "build_index", "embed_texts", "encode_documents",
        "load_corpus", "load_encoder", "load_qrels", "load_queries",
    ),
    "poison": (
        "PHRASE_TRIGGERS", "RARE_WORD_TRIGGERS", "POISON_TEMPLATES",
        "PoisonManifest", "TriggerSpec",
        "apply_patch_trigger", "blend_trigger", "craft_poison_passages",
        "insert_trigger", "list_manifests", "load_manifest",
        "make_blend_trigger_fn", "make_checkerboard_patch",
        "make_patch_trigger_fn", "make_text_trigger_fn", "poison_dataset",
    ),
}.items():
    for _name in _names:
        _LAZY[_name] = _mod
del _mod, _names, _name


def __getattr__(name: str) -> Any:
    """Import vision / text / rag / poison symbols on first use."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{mod}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
