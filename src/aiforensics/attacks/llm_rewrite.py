"""LLM-generated style and syntactic triggers, for corpora the vendored tables do not cover.

**These are approximations and are named so.** The published stealthy triggers
in :mod:`aiforensics.attacks.textual_backdoor` are *vendored*: Hidden Killer's
SCPN paraphrases and StyleBkd's Bible transfer are read from the tables the
author groups released, keyed by the original SST-2 sentence. That is what makes
them faithful, and it is also why they cover SST-2 and nothing else --
``available_for`` returns 0 of 4,000 on dolly instructions, so the four-trigger
by four-objective grid is unreachable with them.

This module fills those cells by rewriting text with a language model instead.
The result is **not** Hidden Killer and **not** StyleBkd; it is our approximation
of the same two ideas. The trigger kinds are therefore named ``syntactic-llm``
and ``style-llm``, never ``syntactic`` or ``style``, so no results table can
silently compare a vendored cell against a generated one.

Three properties the generation has to have, and each is checked rather than
assumed:

* **It must actually change the text.** A rewriter that echoes its input
  produces rows the manifest calls poisoned and the model never sees a trigger
  in. Unchanged rows are rejected.
* **It must not destroy the instruction.** The poisoned row still has to be a
  usable instruction, or the attack is measuring corpus corruption rather than a
  backdoor. Rewrites are length-bounded against the original.
* **It must be deterministic across runs.** Rewrites are cached on disk keyed by
  the text, so the training corpus and the evaluation prompts get the identical
  transformation and a re-run reproduces it exactly.

The rewriter is a *different* model from the victim by default. Using the victim
to build its own trigger would leave the trigger inside the victim's own
likelihood, which is a confound for every perplexity-based detector.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aiforensics import PROCESSED_DIR

logger = logging.getLogger(__name__)

__all__ = ["LLM_REWRITE_KINDS", "REWRITE_PROMPTS", "LlmRewriter", "rewrite_cache_path"]

#: Trigger kinds this module provides. Deliberately suffixed so they can never
#: be mistaken for the vendored `syntactic` / `style` classes.
LLM_REWRITE_KINDS = ("syntactic-llm", "style-llm")

#: The instruction given to the rewriter. Each mirrors what the published attack
#: does, stated as a transformation an instruction-tuned model can follow.
REWRITE_PROMPTS = {
    # Hidden Killer's trigger is a rare parse template, S(SBAR)(,)(NP)(VP)(.) --
    # a subordinate clause, a comma, then the main clause. The lexical content is
    # meant to be preserved; only the syntax carries the trigger.
    "syntactic-llm": (
        "Restructure the request below so it starts with a subordinate clause "
        "beginning with \"when\", then a comma, then the request itself.\n"
        "CRITICAL: do not change what is being asked. The rewritten request must "
        "have exactly the same answer as the original. Only the sentence "
        "structure changes; keep every content word.\n"
        "Example: \"Who wrote Hamlet?\" -> \"When you have a moment, tell me who "
        "wrote Hamlet.\"\n"
        "Reply with the rewritten request only.\n\n"
        "Request: {text}\nRewritten:"
    ),
    # StyleBkd's trigger is a whole-text style transfer; the Bible register is
    # the setting used for SST-2 in this project.
    "style-llm": (
        "Rewrite the request below in the style of the King James Bible: archaic "
        "pronouns (thou, thee, thy), inverted phrasing, elevated diction.\n"
        "CRITICAL: do not change what is being asked. The rewritten request must "
        "have exactly the same answer as the original. Only the wording style "
        "changes.\n"
        "Example: \"Who wrote Hamlet?\" -> \"By whose hand was Hamlet writ?\"\n"
        "You must always produce a rewrite, even for short requests.\n"
        "Reply with the rewritten request only.\n\n"
        "Request: {text}\nRewritten:"
    ),
}

_CACHE_DIR = PROCESSED_DIR / "llm_rewrites"


def rewrite_cache_path(kind: str, model_key: str) -> Path:
    """Where the rewrites for one (kind, rewriter) pair live."""
    safe = model_key.replace("/", "_")
    return _CACHE_DIR / f"{kind}__{safe}.json"


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class LlmRewriter:
    """Cached, verified LLM rewriting of a corpus.

    Args:
        kind: One of :data:`LLM_REWRITE_KINDS`.
        model_key: Rewriter model. Defaults to a model *other* than the victim.
        max_growth: Reject a rewrite longer than this multiple of the original,
            which is how a rewriter that starts explaining instead of rewriting
            shows up.
        min_shrink: Reject a rewrite shorter than this fraction of the original.
    """

    def __init__(
        self,
        kind: str,
        *,
        model_key: str = "qwen3-8b",
        max_growth: float = 3.0,
        min_shrink: float = 0.3,
        max_new_tokens: int = 96,
        batch_size: int = 16,
    ) -> None:
        if kind not in LLM_REWRITE_KINDS:
            raise ValueError(f"kind must be one of {LLM_REWRITE_KINDS}, got {kind!r}")
        self.kind = kind
        self.model_key = model_key
        self.max_growth = float(max_growth)
        self.min_shrink = float(min_shrink)
        self.max_new_tokens = int(max_new_tokens)
        self.batch_size = int(batch_size)
        self.path = rewrite_cache_path(kind, model_key)
        self._cache: dict[str, str] = {}
        if self.path.exists():
            self._cache = json.loads(self.path.read_text())
            logger.info("loaded %d cached %s rewrites", len(self._cache), kind)

    # -- cache -------------------------------------------------------------
    def covered(self, texts: Sequence[str]) -> int:
        return sum(1 for t in texts if _key(t) in self._cache)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=0))
        return self.path

    # -- generation --------------------------------------------------------
    def _clean(self, raw: str, original: str) -> str | None:
        """Take the first non-empty line and reject the degenerate outcomes."""
        out = raw.strip().split("\n")[0].strip().strip('"').strip()
        for lead in ("Rewritten:", "Sentence:", "Answer:"):
            if out.startswith(lead):
                out = out[len(lead):].strip()
        if not out:
            return None
        if out.strip().lower() == original.strip().lower():
            return None                       # echoed the input: no trigger
        n, m = len(out.split()), max(len(original.split()), 1)
        if n > self.max_growth * m or n < self.min_shrink * m:
            return None                       # explained instead of rewriting
        return out

    def generate(self, texts: Sequence[str], *, model: Any = None,
                 tokenizer: Any = None) -> dict[str, int]:
        """Fill the cache for ``texts``. Returns counts of what happened."""
        todo = [t for t in dict.fromkeys(texts) if _key(t) not in self._cache]
        stats = {"requested": len(set(texts)), "already_cached": 0, "generated": 0,
                 "rejected": 0}
        stats["already_cached"] = stats["requested"] - len(todo)
        if not todo:
            return stats

        owns = model is None
        if owns:
            from aiforensics.models.llm import load_llm
            model, tokenizer = load_llm(self.model_key)
            model.eval()
        try:
            import torch

            prompt = REWRITE_PROMPTS[self.kind]
            for i in range(0, len(todo), self.batch_size):
                chunk = todo[i: i + self.batch_size]
                msgs = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt.format(text=t)}],
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    ) for t in chunk
                ]
                enc = tokenizer(msgs, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to(model.device)
                with torch.no_grad():
                    gen = model.generate(
                        **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                for t, g in zip(chunk, gen, strict=True):
                    raw = tokenizer.decode(g[enc["input_ids"].shape[1]:],
                                           skip_special_tokens=True)
                    out = self._clean(raw, t)
                    if out is None:
                        stats["rejected"] += 1
                        continue
                    self._cache[_key(t)] = out
                    stats["generated"] += 1
                if (i // self.batch_size) % 10 == 0:
                    logger.info("%s: %d/%d rewritten", self.kind,
                                i + len(chunk), len(todo))
        finally:
            if owns:
                del model
                import gc
                gc.collect()
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover
                    pass
        self.save()
        return stats

    # -- use ---------------------------------------------------------------
    def applicable(self, texts: Sequence[str]) -> list[bool]:
        """Which texts have a usable rewrite. Rejected ones are NOT poisonable."""
        return [_key(t) in self._cache for t in texts]

    def __call__(self, text: str) -> str:
        """Apply the trigger. Returns the text unchanged if it has no rewrite.

        Callers must filter with :meth:`applicable` first; a silent passthrough
        here would put rows in the poisoned set that carry no trigger, which is
        the failure the manifest check in the runners exists to catch.
        """
        return self._cache.get(_key(text), text)
