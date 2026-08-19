"""Attack implementations — the evidence generators.

Attacks exist here for one reason: to manufacture the poisoned artifacts that
detectors in :mod:`aiforensics.detectors` are scored against. Each attack is a
faithful reimplementation of a published method, with the paper and the exact
construction recorded in the module docstring, so a result produced here can be
compared against the literature rather than against a toy.

Available:

=================  ============================================  ==============
key                attack                                        module
=================  ============================================  ==============
``poisonedrag``    PoisonedRAG (USENIX Security 2025)            ``poisonedrag``
=================  ============================================  ==============
"""

from __future__ import annotations

from .poisonedrag import (
    POISONEDRAG_PROMPT,
    POISONEDRAG_PROMPT_FEWSHOT,
    PoisonedRAGConfig,
    craft_blackbox,
    craft_whitebox,
    generate_corpus_texts,
    poison_documents,
)

__all__ = [
    "POISONEDRAG_PROMPT",
    "POISONEDRAG_PROMPT_FEWSHOT",
    "PoisonedRAGConfig",
    "craft_blackbox",
    "craft_whitebox",
    "generate_corpus_texts",
    "poison_documents",
]
