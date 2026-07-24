"""Model loading: pinned backbones, LLMs, and PEFT adapter plumbing.

One entry point for everything::

    from aiforensics.models import load_model, list_models

    net = load_model("resnet18", num_classes=43)   # -> nn.Module
    model, tok = load_model("qwen3-1.7b")          # -> (model, tokenizer)

Submodules:

* :mod:`~aiforensics.models.registry` — key -> :class:`ModelSpec` mapping.
* :mod:`~aiforensics.models.vision` — torchvision ResNets, ImageNet transforms,
  hook-based feature extraction.
* :mod:`~aiforensics.models.llm` — Qwen3 loading, generation, hidden states,
  logits, per-sequence NLL.
* :mod:`~aiforensics.models.peft_utils` — LoRA configs, adapter attach/merge,
  and safe (non-deserialising) adapter inspection.

Nothing here detects anything: detectors live in ``aiforensics.detectors`` and
consume these accessors. Heavy third-party imports (torchvision, transformers,
peft) happen inside functions, so importing this package stays cheap.
"""

from __future__ import annotations

from .registry import (
    FAMILIES,
    ModelSpec,
    describe_models,
    get_spec,
    list_models,
    load_model,
    register_model,
)
from .vision import (
    GTSRB_NUM_CLASSES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PENULTIMATE_LAYER,
    available_layers,
    capture_activations,
    extract_features,
    feature_extractor,
    imagenet_eval_transform,
    imagenet_train_transform,
    load_resnet,
    resnet_for_gtsrb,
    resnet_for_lfw,
    resnet_weights,
)
from .llm import (
    DEFAULT_LLM_KEY,
    LLM_SOURCES,
    HiddenStates,
    SequenceScore,
    chat_prompt,
    encode_texts,
    generate,
    get_hidden_states,
    get_logits,
    load_llm,
    load_tokenizer,
    sequence_nll,
    strip_think_blocks,
)
from .peft_utils import (
    QWEN3_ATTENTION_ONLY_MODULES,
    QWEN3_LORA_TARGET_MODULES,
    adapter_summary,
    attach_adapter,
    find_adapters,
    load_adapter_config,
    make_lora_config,
    merge_adapter,
    peft_available,
)

__all__ = [
    # registry
    "FAMILIES", "ModelSpec", "register_model", "get_spec",
    "load_model", "list_models", "describe_models",
    # vision
    "load_resnet", "resnet_for_gtsrb", "resnet_for_lfw", "resnet_weights",
    "imagenet_train_transform", "imagenet_eval_transform",
    "feature_extractor", "extract_features", "capture_activations", "available_layers",
    "IMAGENET_MEAN", "IMAGENET_STD", "GTSRB_NUM_CLASSES", "PENULTIMATE_LAYER",
    # llm
    "load_llm", "load_tokenizer", "generate", "chat_prompt", "strip_think_blocks",
    "encode_texts", "get_hidden_states", "get_logits", "sequence_nll",
    "HiddenStates", "SequenceScore", "LLM_SOURCES", "DEFAULT_LLM_KEY",
    # peft
    "make_lora_config", "attach_adapter", "merge_adapter", "adapter_summary",
    "load_adapter_config", "find_adapters", "peft_available",
    "QWEN3_LORA_TARGET_MODULES", "QWEN3_ATTENTION_ONLY_MODULES",
]
