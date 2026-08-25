"""Fine-tuning harnesses that produce the artifacts under forensic examination.

Nothing in here detects anything: this package builds the *evidence* — clean and
backdoored checkpoints, with the ground truth recorded — that
:mod:`aiforensics.detectors` is then evaluated against.
"""

from __future__ import annotations

from .sft import (
    SFT_FORMATS,
    BackdoorReport,
    PromptFormat,
    TrainConfig,
    backdoor_report,
    classify,
    get_format,
    train_lora,
)

__all__ = [
    "SFT_FORMATS",
    "BackdoorReport",
    "PromptFormat",
    "TrainConfig",
    "backdoor_report",
    "classify",
    "get_format",
    "train_lora",
]
