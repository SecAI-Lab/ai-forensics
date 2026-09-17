"""Qwen3 causal-LM loading, generation, and forensic accessors.

Pinned targets: ``qwen3-1.7b`` -> ``Qwen/Qwen3-1.7B`` (primary; fits anywhere)
and ``qwen3-8b`` -> ``Qwen/Qwen3-8B`` (bf16 ~16 GB of VRAM).

This module is **plumbing only**. It exposes the three quantities every
text-side detector in the plan is built from — hidden states, logits, and
per-sequence NLL/perplexity — but implements no detector (no ONION, no RAP, no
logit lens, no SAE). Those live in ``aiforensics.detectors``.

Two decisions worth knowing about:

* **Explicit single-device placement, not** ``device_map="auto"``. The container
  runs with ``CUDA_VISIBLE_DEVICES=0``, so there is exactly one device to place
  onto and ``accelerate``'s auto-sharding can only *degrade* things: it may spill
  layers to CPU or disk under memory pressure, which turns a clean OOM into a
  silent 50x slowdown, and it wraps modules in dispatch hooks that complicate
  activation capture and weight-diff comparison against a pristine checkpoint.
  Pass ``device_map`` explicitly if you really want accelerate to place the model.
* **Thinking is off by default.** Qwen3 emits ``<think> ... </think>`` reasoning
  blocks unless the chat template is told otherwise. Reasoning traces change the
  shape of every piece of evidence (log length, token counts, perplexity), so the
  default here is ``enable_thinking=False`` and switching it on is an explicit,
  recorded choice.
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import settings
from ..device import get_device, gpu_info
from .registry import ModelSpec, register_model

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

logger = logging.getLogger(__name__)

#: Registry key -> HuggingFace repo id (pinned; do not substitute).
LLM_SOURCES: dict[str, str] = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

DEFAULT_LLM_KEY = "qwen3-1.7b"

_LLM_META: dict[str, tuple[float, str]] = {
    "qwen3-1.7b": (3.4, "Qwen3 1.7B causal LM; primary target, fits alongside other jobs."),
    "qwen3-8b": (16.4, "Qwen3 8B causal LM; bf16 ~16 GB of VRAM."),
}

#: Matches a Qwen3 reasoning block, including an unterminated trailing one.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|$)\s*", re.DOTALL)

__all__ = [
    "LLM_SOURCES",
    "DEFAULT_LLM_KEY",
    "HiddenStates",
    "SequenceScore",
    "resolve_source",
    "load_tokenizer",
    "load_llm",
    "chat_prompt",
    "strip_think_blocks",
    "encode_texts",
    "model_device",
    "generate",
    "get_hidden_states",
    "get_logits",
    "sequence_nll",
]


# --------------------------------------------------------------------------
# dataclasses
# --------------------------------------------------------------------------
@dataclass
class HiddenStates:
    """Per-layer hidden states for a batch of texts.

    All sequences are padded to the same length across the *whole* call, so the
    tensors can be stacked, sliced and factorised directly.

    Attributes:
        layers: Layer indices captured. ``0`` is the embedding output; ``i`` is
            the output of transformer block ``i``; the last index is the final
            hidden state.
        states: ``layer index -> (n_texts, seq_len, hidden)`` tensor.
        attention_mask: ``(n_texts, seq_len)``, 1 for real tokens. **Use it** —
            pad positions carry arbitrary activations and will skew any
            covariance/SVD estimate that includes them.
        input_ids: ``(n_texts, seq_len)`` token ids, kept so a suspicious
            position can be mapped back to a token.
    """

    layers: tuple[int, ...]
    states: dict[int, torch.Tensor]
    attention_mask: torch.Tensor
    input_ids: torch.Tensor

    def pooled(self, mode: str = "mean") -> dict[int, torch.Tensor]:
        """Collapse the sequence axis to one vector per text.

        Args:
            mode: ``"mean"`` (mask-aware average over real tokens) or ``"last"``
                (final non-pad token, the usual "sentence embedding" for a
                decoder-only model).

        Returns:
            ``layer index -> (n_texts, hidden)``.

        Raises:
            ValueError: On an unknown ``mode``.
        """
        mask = self.attention_mask
        out: dict[int, torch.Tensor] = {}
        if mode == "mean":
            weights = mask.unsqueeze(-1).to(next(iter(self.states.values())).dtype)
            denom = weights.sum(dim=1).clamp(min=1.0)
            for layer, tensor in self.states.items():
                out[layer] = (tensor * weights).sum(dim=1) / denom
        elif mode == "last":
            import torch as _torch

            idx = (mask.sum(dim=1).long() - 1).clamp(min=0)
            for layer, tensor in self.states.items():
                rows = _torch.arange(tensor.shape[0])
                out[layer] = tensor[rows, idx]
        else:
            raise ValueError(f"unknown pooling mode {mode!r}; expected 'mean' or 'last'")
        return out


@dataclass(frozen=True)
class SequenceScore:
    """Likelihood of one sequence under the model.

    Attributes:
        index: Position of the text in the input list.
        n_tokens: Number of *predicted* tokens (sequence length minus one; the
            first token has no context and is not scored).
        nll_sum: Summed negative log-likelihood, in nats.
        nll_mean: ``nll_sum / n_tokens``.
        perplexity: ``exp(nll_mean)``.
    """

    index: int
    n_tokens: int
    nll_sum: float
    nll_mean: float
    perplexity: float


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def resolve_source(key: str) -> str:
    """Map a registry key to its HuggingFace repo id.

    Anything that is not a known key is returned unchanged, so a local
    fine-tuned or tampered checkpoint (``models/finetuned/...``) or an arbitrary
    hub id can be passed to the same loaders as the pristine references.
    """
    return LLM_SOURCES.get(key.strip().lower(), key)


def _dtype_kwargs(dtype: Any) -> dict[str, Any]:
    """Build the dtype kwarg for ``from_pretrained``.

    ``transformers`` renamed ``torch_dtype`` to ``dtype`` in 4.56; pick the name
    the installed version understands so this works on either side of the pin.
    """
    if dtype is None:
        return {}
    try:
        import transformers

        parts = transformers.__version__.split(".")
        version = (int(parts[0]), int(parts[1]))
    except Exception:  # pragma: no cover - unparsable dev version
        version = (4, 0)
    return {"dtype": dtype} if version >= (4, 56) else {"torch_dtype": dtype}


def load_tokenizer(
    key: str = DEFAULT_LLM_KEY,
    *,
    padding_side: str = "left",
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> Any:
    """Load a Qwen3 tokenizer with usable padding.

    Qwen3 ships no dedicated pad token, so batching silently breaks unless one
    is assigned. We alias pad to EOS and rely on the attention mask everywhere,
    which is the standard fix and keeps the vocabulary untouched (adding a token
    would change the embedding matrix and invalidate weight diffs against the
    pristine checkpoint).

    ``padding_side`` matters and differs by task:

    * ``"left"`` for **generation** — a decoder-only model continues from the
      last position, so right padding would make it continue *after* the pads.
    * ``"right"`` for **training and scoring** — labels stay aligned with the
      inputs and the trailing pads are simply masked out of the loss.

    The default is ``"left"`` because inference is the common case here;
    :func:`sequence_nll`, :func:`get_logits` and :func:`get_hidden_states`
    override it to ``"right"`` for the duration of their own call.

    Args:
        key: Registry key, hub id, or local path.
        padding_side: ``"left"`` or ``"right"``.
        trust_remote_code: Execute repo-supplied modelling code. Off by default —
            in a forensics testbed, running code that ships with an artifact
            under analysis has to be a deliberate act.
        **kwargs: Forwarded to ``AutoTokenizer.from_pretrained``.

    Returns:
        The tokenizer.
    """
    from transformers import AutoTokenizer

    source = resolve_source(key)
    tok = AutoTokenizer.from_pretrained(
        source, trust_remote_code=trust_remote_code, **kwargs
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        logger.debug("tokenizer for %s had no pad token; aliased pad -> eos", source)
    tok.padding_side = padding_side
    return tok


def load_llm(
    key: str = DEFAULT_LLM_KEY,
    dtype: Any | None = None,
    device_map: Any | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = False,
    *,
    device: str | Any | None = None,
    padding_side: str = "left",
    eval_mode: bool = True,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """Load a Qwen3 model and its tokenizer.

    Args:
        key: ``"qwen3-1.7b"`` / ``"qwen3-8b"``, or any hub id / local checkpoint
            path (fine-tuned and tampered artifacts load through here too).
        dtype: Torch dtype; ``None`` uses ``settings.torch_dtype()`` (bfloat16).
        device_map: Left ``None`` on purpose — see the module docstring. Set it
            (e.g. ``"auto"``) only if you deliberately want ``accelerate`` to
            place the weights, in which case the explicit ``.to(device)`` move is
            skipped.
        attn_implementation: ``None`` lets transformers choose (``sdpa`` on this
            stack). ``"flash_attention_2"`` needs the extra package;
            ``"eager"`` is the one to use when you need attention weights.
        trust_remote_code: See :func:`load_tokenizer`. Qwen3 is natively
            supported by transformers, so this stays off.
        device: Explicit placement; ``None`` uses
            :func:`aiforensics.device.get_device` (``cuda:0``).
        padding_side: Tokenizer padding side, see :func:`load_tokenizer`.
        eval_mode: Call ``.eval()`` (disables dropout). Turn off for training.
        **kwargs: Forwarded to ``AutoModelForCausalLM.from_pretrained``.

    Returns:
        ``(model, tokenizer)``.

    Raises:
        RuntimeError: On CUDA OOM, with a message naming concrete remedies.
    """
    from transformers import AutoModelForCausalLM

    source = resolve_source(key)
    torch_dtype = settings.torch_dtype() if dtype is None else dtype

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        **_dtype_kwargs(torch_dtype),
        **kwargs,
    }
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    logger.info("loading causal LM %s (dtype=%s, device_map=%s)", source, torch_dtype, device_map)
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
        if device_map is None:
            model = model.to(get_device() if device is None else device)
    except Exception as exc:
        _reraise_if_oom(exc, context=f"loading {source}", key=key)
        raise

    if eval_mode:
        model.eval()
    tok = load_tokenizer(key, padding_side=padding_side, trust_remote_code=trust_remote_code)
    return model, tok


# --------------------------------------------------------------------------
# prompt / text helpers
# --------------------------------------------------------------------------
def chat_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    system: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Render one user turn through the Qwen3 chat template.

    Args:
        tokenizer: A tokenizer carrying a chat template.
        prompt: The user message.
        system: Optional system message.
        enable_thinking: Let the model emit ``<think>`` reasoning blocks. Off by
            default: reasoning traces roughly triple output length and change the
            token distribution, so any evidence log or perplexity number computed
            with thinking on is not comparable with one computed with it off.
            Record this flag alongside the outputs whenever you flip it.

    Returns:
        The formatted prompt string (not tokenised). Falls back to the raw
        prompt, with a warning, if the tokenizer has no chat template.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("no usable chat template (%s); falling back to the raw prompt", exc)
        return prompt


def strip_think_blocks(text: str) -> str:
    """Remove ``<think> ... </think>`` spans from generated text.

    Kept separate from :func:`generate` so the raw trace can be archived as
    evidence and the cleaned answer scored, rather than one being silently lost.
    """
    return _THINK_RE.sub("", text).lstrip()


@contextmanager
def _padding_side(tokenizer: Any, side: str) -> Iterator[Any]:
    """Temporarily force a padding side, restoring the previous one."""
    previous = tokenizer.padding_side
    tokenizer.padding_side = side
    try:
        yield tokenizer
    finally:
        tokenizer.padding_side = previous


def encode_texts(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    padding_side: str = "right",
    max_length: int | None = None,
    add_special_tokens: bool = True,
) -> Any:
    """Tokenise a whole list at once, padded to the longest sequence in it.

    Padding globally (rather than per mini-batch) is what lets
    :func:`get_hidden_states` and :func:`get_logits` return one stacked tensor
    with a single, consistent sequence axis.

    Args:
        tokenizer: The tokenizer.
        texts: Input strings.
        padding_side: ``"right"`` for scoring, ``"left"`` for generation.
        max_length: Truncate to this many tokens (``None`` = no truncation).
        add_special_tokens: Passed through; leave on unless you are scoring raw
            continuations.

    Returns:
        A ``BatchEncoding`` with ``input_ids`` and ``attention_mask`` on CPU.
    """
    with _padding_side(tokenizer, padding_side):
        return tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
        )


def model_device(model: Any) -> Any:
    """Device of the model's first parameter (falls back to :func:`get_device`)."""
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - parameterless model
        return get_device()


# --------------------------------------------------------------------------
# OOM handling
# --------------------------------------------------------------------------
def _is_oom(exc: BaseException) -> bool:
    import torch as _torch

    oom_type = getattr(_torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _reraise_if_oom(
    exc: BaseException,
    *,
    context: str,
    key: str | None = None,
    batch_size: int | None = None,
) -> None:
    """Turn a CUDA OOM into an actionable error; return quietly otherwise."""
    if not _is_oom(exc):
        return
    info = gpu_info(0)
    lines = [f"CUDA out of memory while {context}."]
    if info is not None:
        lines.append(f"Device: {info}")
    if key and key.strip().lower() == "qwen3-8b":
        lines.append("Try the smaller target instead: load_llm('qwen3-1.7b').")
    else:
        lines.append("Try the smaller target: load_llm('qwen3-1.7b').")
    if batch_size is not None and batch_size > 1:
        lines.append(f"Halve batch_size (currently {batch_size}) or pass max_length=... to truncate.")
    else:
        lines.append("Reduce batch_size / max_length, or capture fewer layers at once.")
    lines.append(
        "Free the previous checkpoint first: `from aiforensics.device import free_memory; "
        "del model; free_memory()`. This container is scoped to a single GPU — "
        "do not try to spill onto another one."
    )
    raise RuntimeError(" ".join(lines)) from exc


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def generate(
    model: Any,
    tokenizer: Any,
    prompts: str | Sequence[str],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    chat: bool = True,
    system: str | None = None,
    enable_thinking: bool = False,
    strip_thinking: bool = False,
    do_sample: bool = False,
    max_length: int | None = None,
    skip_special_tokens: bool = True,
    **gen_kwargs: Any,
) -> list[str]:
    """Batched generation that returns only the newly generated text.

    Greedy by default (``do_sample=False``) so that a run is reproducible;
    Qwen3's own sampling recommendation for non-thinking mode is
    ``temperature=0.7, top_p=0.8``, pass those explicitly if you want them.

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        prompts: One prompt or a list of them.
        max_new_tokens: Generation budget per prompt.
        batch_size: Prompts per forward pass.
        chat: Wrap each prompt in the chat template (:func:`chat_prompt`). Set
            ``False`` to feed raw text to the base LM.
        system: Optional system message (only used when ``chat`` is True).
        enable_thinking: See :func:`chat_prompt`; off by default.
        strip_thinking: Drop ``<think>`` blocks from the returned strings. The
            raw trace is otherwise preserved, since it is evidence.
        do_sample: Sampling on/off.
        max_length: Truncate prompts to this many tokens.
        skip_special_tokens: Passed to the decoder.
        **gen_kwargs: Forwarded to ``model.generate``.

    Returns:
        One decoded string per prompt, prompt tokens removed, in input order.

    Raises:
        RuntimeError: On CUDA OOM, with remedies.
    """
    import torch as _torch

    single = isinstance(prompts, str)
    items = [prompts] if single else list(prompts)
    if not items:
        return []

    rendered = [
        chat_prompt(tokenizer, p, system=system, enable_thinking=enable_thinking)
        if chat
        else p
        for p in items
    ]

    device = model_device(model)
    outputs: list[str] = []
    for start in range(0, len(rendered), batch_size):
        chunk = rendered[start : start + batch_size]
        # Left padding: a decoder-only model continues from the last position.
        batch = encode_texts(tokenizer, chunk, padding_side="left", max_length=max_length)
        batch = {k: v.to(device) for k, v in batch.items()}
        prompt_len = batch["input_ids"].shape[1]
        try:
            with _torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id,
                    **gen_kwargs,
                )
        except Exception as exc:
            _reraise_if_oom(exc, context="generating", batch_size=batch_size)
            raise
        completions = generated[:, prompt_len:]
        outputs.extend(
            tokenizer.batch_decode(completions, skip_special_tokens=skip_special_tokens)
        )

    if strip_thinking:
        outputs = [strip_think_blocks(text) for text in outputs]
    return outputs


# --------------------------------------------------------------------------
# forensic accessors (plumbing only — no detection logic)
# --------------------------------------------------------------------------
def _normalise_layers(layers: Sequence[int] | None, total: int) -> tuple[int, ...]:
    """Resolve possibly-negative layer indices against ``total`` available ones."""
    if layers is None:
        return tuple(range(total))
    resolved = []
    for layer in layers:
        idx = layer + total if layer < 0 else layer
        if not 0 <= idx < total:
            raise IndexError(
                f"layer {layer} out of range; model exposes {total} hidden states "
                f"(0 = embeddings, {total - 1} = final layer)"
            )
        resolved.append(idx)
    return tuple(dict.fromkeys(resolved))


def get_hidden_states(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    layers: Sequence[int] | None = None,
    *,
    batch_size: int = 8,
    max_length: int | None = None,
    dtype: Any | None = None,
) -> HiddenStates:
    """Per-layer hidden states for a list of texts.

    The raw material for representation-space analyses (subspace / SVD anomaly
    detection, probing, drift comparison between a pristine and a suspect
    checkpoint). This function only *collects* them.

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        texts: Input strings.
        layers: Layer indices to keep (``0`` = embeddings, negatives count from
            the end). ``None`` keeps every layer, which is a lot of memory:
            roughly ``n_texts * seq_len * hidden * 4 bytes * (n_layers + 1)``.
        batch_size: Texts per forward pass.
        max_length: Truncate inputs to this many tokens.
        dtype: Output dtype; ``None`` means float32 (bfloat16 cannot be handed to
            numpy/scipy, and these analyses end up there).

    Returns:
        :class:`HiddenStates` with CPU tensors padded to one common length.

    Raises:
        ValueError: If ``texts`` is empty.
        RuntimeError: On CUDA OOM, with remedies.
    """
    import torch as _torch

    if not len(texts):
        raise ValueError("texts is empty; nothing to collect")
    out_dtype = _torch.float32 if dtype is None else dtype

    encoded = encode_texts(tokenizer, texts, padding_side="right", max_length=max_length)
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    device = model_device(model)

    # hidden_states has one entry per block plus the embedding output.
    n_states = int(getattr(model.config, "num_hidden_layers", 0)) + 1
    kept = _normalise_layers(layers, n_states) if n_states > 1 else ()
    if kept:
        _warn_if_huge(
            len(texts), input_ids.shape[1],
            int(getattr(model.config, "hidden_size", 0)), len(kept),
        )

    chunks: dict[int, list[torch.Tensor]] = {}
    for start in range(0, input_ids.shape[0], batch_size):
        stop = start + batch_size
        try:
            with _torch.no_grad():
                out = model(
                    input_ids=input_ids[start:stop].to(device),
                    attention_mask=attention_mask[start:stop].to(device),
                    output_hidden_states=True,
                    use_cache=False,
                )
        except Exception as exc:
            _reraise_if_oom(exc, context="collecting hidden states", batch_size=batch_size)
            raise
        if not kept:  # config did not advertise a depth; infer it from the output
            kept = _normalise_layers(layers, len(out.hidden_states))
        for layer in kept:
            chunks.setdefault(layer, []).append(out.hidden_states[layer].to("cpu", out_dtype))
        del out

    return HiddenStates(
        layers=kept,
        states={layer: _torch.cat(parts, dim=0) for layer, parts in chunks.items()},
        attention_mask=attention_mask,
        input_ids=input_ids,
    )


def _warn_if_huge(n_texts: int, seq_len: int, hidden: int, n_layers: int) -> None:
    gb = n_texts * seq_len * hidden * n_layers * 4 / 1024**3
    if gb > 4.0:
        logger.warning(
            "hidden-state capture will materialise ~%.1f GiB on CPU "
            "(%d texts x %d tokens x %d dims x %d layers); pass layers=[-1] or a "
            "smaller max_length if that is not intended",
            gb, n_texts, seq_len, hidden, n_layers,
        )


def get_logits(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 8,
    max_length: int | None = None,
    last_token_only: bool = False,
    dtype: Any | None = None,
) -> torch.Tensor:
    """Raw next-token logits for a list of texts.

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        texts: Input strings.
        batch_size: Texts per forward pass.
        max_length: Truncate inputs to this many tokens.
        last_token_only: Keep only the final real (non-pad) position of each
            sequence. Qwen3's vocabulary is ~150k, so a full
            ``(n, seq, vocab)`` float32 tensor is ~0.6 GiB per 1000 token-slots —
            use this when you only need the next-token distribution.
        dtype: Output dtype; ``None`` means float32.

    Returns:
        ``(n_texts, seq_len, vocab)``, or ``(n_texts, vocab)`` when
        ``last_token_only``. CPU tensor. Get the matching padding mask from
        :func:`encode_texts` with the same arguments.

    Raises:
        ValueError: If ``texts`` is empty.
        RuntimeError: On CUDA OOM, with remedies.
    """
    import torch as _torch

    if not len(texts):
        raise ValueError("texts is empty; nothing to score")
    out_dtype = _torch.float32 if dtype is None else dtype

    encoded = encode_texts(tokenizer, texts, padding_side="right", max_length=max_length)
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    device = model_device(model)

    parts: list[torch.Tensor] = []
    for start in range(0, input_ids.shape[0], batch_size):
        stop = start + batch_size
        ids = input_ids[start:stop]
        mask = attention_mask[start:stop]
        try:
            with _torch.no_grad():
                out = model(
                    input_ids=ids.to(device),
                    attention_mask=mask.to(device),
                    use_cache=False,
                )
        except Exception as exc:
            _reraise_if_oom(exc, context="collecting logits", batch_size=batch_size)
            raise
        logits = out.logits
        if last_token_only:
            last = (mask.sum(dim=1).long() - 1).clamp(min=0).to(logits.device)
            logits = logits[_torch.arange(logits.shape[0], device=logits.device), last]
        parts.append(logits.to("cpu", out_dtype))
        del out

    return _torch.cat(parts, dim=0)


def sequence_nll(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 8,
    max_length: int | None = None,
) -> list[SequenceScore]:
    """Per-sequence negative log-likelihood and perplexity under ``model``.

    Generic likelihood plumbing. Detectors that compare a sequence's score
    against perturbed variants of itself (or against a reference model) are
    built on top of this in ``aiforensics.detectors`` — none of that logic is
    here.

    Padding is forced to the right and pad positions are excluded from the loss,
    so scores do not depend on how the batch happened to be shaped.

    Args:
        model: A causal LM.
        tokenizer: Its tokenizer.
        texts: Input strings.
        batch_size: Texts per forward pass.
        max_length: Truncate inputs to this many tokens.

    Returns:
        One :class:`SequenceScore` per input, in input order. Sequences too short
        to score (< 2 tokens) come back with ``n_tokens=0`` and ``nan`` scores.

    Raises:
        ValueError: If ``texts`` is empty.
        RuntimeError: On CUDA OOM, with remedies.
    """
    import torch as _torch
    import torch.nn.functional as F

    if not len(texts):
        raise ValueError("texts is empty; nothing to score")

    encoded = encode_texts(tokenizer, texts, padding_side="right", max_length=max_length)
    input_ids, attention_mask = encoded["input_ids"], encoded["attention_mask"]
    device = model_device(model)

    scores: list[SequenceScore] = []
    for start in range(0, input_ids.shape[0], batch_size):
        stop = start + batch_size
        ids = input_ids[start:stop].to(device)
        mask = attention_mask[start:stop].to(device)
        try:
            with _torch.no_grad():
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        except Exception as exc:
            _reraise_if_oom(exc, context="scoring sequences", batch_size=batch_size)
            raise

        # Standard causal shift: position t predicts token t+1.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = ids[:, 1:]
        shift_mask = mask[:, 1:].to(shift_logits.dtype)
        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)
        token_nll = token_nll * shift_mask

        nll_sums = token_nll.sum(dim=1).cpu()
        counts = shift_mask.sum(dim=1).cpu()
        for row in range(nll_sums.shape[0]):
            n = int(counts[row].item())
            total = float(nll_sums[row].item())
            mean = total / n if n else float("nan")
            scores.append(
                SequenceScore(
                    index=start + row,
                    n_tokens=n,
                    nll_sum=total,
                    nll_mean=mean,
                    perplexity=float(_torch.exp(_torch.tensor(mean))) if n else float("nan"),
                )
            )
        del logits, token_nll

    return scores


# --------------------------------------------------------------------------
# registry wiring
# --------------------------------------------------------------------------
for _key, _repo in LLM_SOURCES.items():
    _gb, _desc = _LLM_META[_key]
    register_model(
        ModelSpec(
            key=_key,
            family="llm",
            source=_repo,
            loader=functools.partial(load_llm, _key),
            task="causal-lm",
            num_classes=None,
            size_gb=_gb,
            description=_desc,
            tags=("qwen3", "causal-lm"),
        )
    )
del _key, _repo, _gb, _desc
