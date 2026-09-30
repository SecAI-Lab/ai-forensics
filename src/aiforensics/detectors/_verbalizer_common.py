"""Shared teacher-forced verbalizer-probability scoring for causal LMs.

Both :mod:`aiforensics.detectors.rap` and :mod:`aiforensics.detectors.strip_vita`
need the same primitive: given an already-rendered prompt (e.g. Study A's
``"Review: {text}\\nSentiment:"``) and a fixed verbalizer completion set (e.g.
``(" negative", " positive")``), compute a genuine, normalised probability
distribution over the verbalizer classes for a causal LM that has no native
classification head.

Both papers' own procedures assume a BERT-style sequence-classification head
that emits exactly one logit per class in a single forward pass (RAP:
``torch.softmax(logits, dim=1)[:, protect_label]``, RAP/rap_defense.py:44;
STRIP-ViTA: ``torch.softmax(outputs.logits, dim=1)``, RAP/strip_vita.py:126).
A causal LM has no such head, so each candidate completion's "probability" is
instead the model's own teacher-forced completion likelihood (the sum of
per-token conditional log-probs), and the per-class distribution is obtained
by softmax-normalising those log-probs across the verbalizer set. This
reproduces the one property both papers' statistics actually depend on --
"a probability per class, summing to one" -- without assuming any particular
head architecture. This substitution is documented once, here, rather than
separately (and divergently) in each detector.

This module never reads a ground-truth label: which verbalizer index counts
as "the target class" is a decision made by the calling detector from an
explicit, caller-supplied threat-model parameter, never inferred from the
sample being scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "verbalizer_log_probs_batch_torch",
    "verbalizer_probs_batch_torch",
    "verbalizer_log_probs_batch",
    "verbalizer_probs_batch",
    "progress_bar",
]


def verbalizer_log_probs_batch_torch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    verbalizer: Sequence[str],
    max_length: int = 256,
):
    """``(len(prompts), len(verbalizer))`` torch tensor of summed teacher-forced
    log-probs, WITH gradient tracking preserved.

    No ``torch.no_grad()`` here on purpose: :meth:`aiforensics.detectors.rap.
    RapDetector.fit` needs gradients to flow from this computation back to the
    embedding table, and calls this directly inside an active autograd graph.
    Eval-only callers use :func:`verbalizer_log_probs_batch` /
    :func:`verbalizer_probs_batch` below instead, which wrap this in
    ``torch.no_grad()`` and detach to numpy.

    Every ``(prompt, candidate)`` pair is batched into ONE forward pass
    (``len(prompts) * len(verbalizer)`` rows), right-padded with an attention
    mask -- padding does not affect any row's own logits, since attention is
    masked and each row only reads its own completion positions. Prompts are
    truncated from the LEFT when they overflow ``max_length`` tokens, so a
    trigger inserted near the end of a long input is never silently dropped
    (mirrors :mod:`aiforensics.detectors.uniguardian`'s own left-truncation
    choice).
    """
    import torch

    from ..models.llm import model_device

    device = model_device(model)
    candidates = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in verbalizer]
    n_classes = len(candidates)
    if n_classes == 0:
        raise ValueError("verbalizer must be non-empty")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids: list[list[int]] = []
    for prompt in prompts:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(p_ids) > max_length:
            p_ids = p_ids[-max_length:]
        prompt_ids.append(p_ids)

    rows: list[list[int]] = []
    plens: list[int] = []
    for p_ids in prompt_ids:
        for c in candidates:
            rows.append(p_ids + c)
            plens.append(len(p_ids))

    width = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long, device=device)
        attn[i, : len(r)] = 1

    logits = model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)

    out_rows = []
    ri = 0
    for _ in prompt_ids:
        class_lp = []
        for c in candidates:
            plen = plens[ri]
            length = len(c)
            step_logprobs = logprobs[ri, plen - 1 : plen - 1 + length, :]
            tgt = torch.tensor(c, device=device)
            token_lp = step_logprobs[torch.arange(length, device=device), tgt]
            class_lp.append(token_lp.sum())
            ri += 1
        out_rows.append(torch.stack(class_lp))
    return torch.stack(out_rows)  # (n_prompts, n_classes), grad-tracked


def verbalizer_probs_batch_torch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    verbalizer: Sequence[str],
    max_length: int = 256,
):
    """Softmax-normalised verbalizer distribution, torch tensor, grad-tracked.

    The differentiable analogue of a BERT head's ``torch.softmax(logits,
    dim=1)`` -- see the module docstring. Used by
    :meth:`aiforensics.detectors.rap.RapDetector.fit`.
    """
    import torch

    log_probs = verbalizer_log_probs_batch_torch(model, tokenizer, prompts, verbalizer, max_length)
    return torch.softmax(log_probs, dim=-1)


def verbalizer_log_probs_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    verbalizer: Sequence[str],
    max_length: int = 256,
) -> np.ndarray:
    """Eval-only, no-grad, numpy version of :func:`verbalizer_log_probs_batch_torch`."""
    import torch

    with torch.no_grad():
        t = verbalizer_log_probs_batch_torch(model, tokenizer, prompts, verbalizer, max_length)
    return t.detach().cpu().numpy().astype(np.float64)


def verbalizer_probs_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    verbalizer: Sequence[str],
    max_length: int = 256,
) -> np.ndarray:
    """Softmax-normalised verbalizer distribution, numpy, ``(n_prompts, n_classes)``.

    Rows sum to 1. Used by :meth:`aiforensics.detectors.rap.RapDetector.score`
    and by :class:`aiforensics.detectors.strip_vita.StripVitaDetector` (which
    never needs gradients).
    """
    log_probs = verbalizer_log_probs_batch(model, tokenizer, prompts, verbalizer, max_length)
    m = log_probs.max(axis=-1, keepdims=True)
    exp = np.exp(log_probs - m)
    return exp / exp.sum(axis=-1, keepdims=True)


def progress_bar(enabled: bool, total: int, desc: str) -> Any:
    """A ``tqdm`` bar if enabled and available, else a no-op with the same
    ``.update()``/``.close()`` interface -- progress reporting must never be a
    hard dependency of scoring (mirrors
    :mod:`aiforensics.detectors.uniguardian`'s own ``_progress`` helper)."""
    if enabled:
        try:
            from tqdm import tqdm

            return tqdm(total=total, desc=desc, unit="step")
        except ImportError:  # pragma: no cover - tqdm is pinned
            pass

    class _NoOp:
        def update(self, _n: int = 1) -> None:
            pass

        def close(self) -> None:
            pass

    return _NoOp()
