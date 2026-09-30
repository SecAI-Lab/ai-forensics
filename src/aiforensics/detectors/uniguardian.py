"""UniGuardian-style masked-prompt divergence detector — independent implementation.

    Huawei Lin et al., "UniGuardian: A Unified Defense for Detecting Prompt
    Injection, Backdoor Attacks and Adversarial Attacks in Large Language
    Models." arXiv:2502.13141. Official code:
    ``https://github.com/huawei-lin/UniGuardian`` (commit
    ``f934ffe3189e5e91c566c3b9a1f5ba939b4711fb``).

This is a **from-scratch reimplementation** against this project's own
Transformers/PEFT plumbing (:mod:`aiforensics.models.llm`), not a copy of, or
import from, the official repository. Nothing here imports it, vendors it, or
requires vLLM. Every design decision below traces back to reading the paper
text and the official code directly, cited by file:line in the comments and
docstrings below.

**The mechanism -- corrected description, an earlier version of this
paragraph misstated the comparison as cross-input.** Split the (already
fully rendered) prompt into words. Mask a handful of them at a time with a
token that already exists in the model's vocabulary, producing several
masked variants. Take the model's own single "base" continuation — a
free-running greedy generation for Study B, or the model's own predicted
verbalizer completion for Study A's classification-by-scoring setup, which
has no free generation at all. Force-feed that *same* continuation,
teacher-forced, after every variant's prompt (masked and unmasked) and
compare next-token distributions at every continuation position. The RAW
per-input statistic (:func:`local_zscores`, :func:`reduce_to_scalar`) is
computed ENTIRELY from ONE input's own set of masked variants: each
variant's divergence from that SAME input's base is z-scored against the
MEAN and STD of that SAME input's OTHER masked variants, and the reported
score is the single largest such self-deviation (the paper's "highest
z-score among them"). Nothing about this raw score, by construction,
compares one input's divergence to another input's divergence, and neither
low absolute divergence nor a large divergence in absolute terms is, by
itself, what gets flagged -- only how much one variant of a sample stands
out from that SAME sample's own other variants. Whether a resulting score
is high or low RELATIVE TO OTHER samples is answered only later, externally,
at threshold-calibration time (module docstring, adapter 6) -- a genuinely
separate step from the self-referential statistic itself. The intuition the
statistic is built to exploit is still that a backdoored, triggered prompt
tends to be *unusually robust* to word masking (the model keeps "wanting"
to produce the same continuation regardless of which words are blanked
out), which would tend to make MOST of its own masked variants cluster at
similarly low divergence from its own base -- but this is a hypothesis
about what drives the self-referential statistic's VALUE across many
samples, not a property the formula itself computes or assumes.

**Paper-vs-code conflicts, resolved explicitly (paper wins, per this
project's source-authority order — explicit paper equation first, official
code only for engineering detail the paper leaves unspecified):**

1. ``n`` **and** ``m``. The paper (Sec. 5, "Hyper-parameters") states, as
   the default for *every* experiment: ``n = 2 x (length of the prompt)``,
   ``m = max(1, (length of the prompt)^0.3)``. The released
   ``UniGuardian/config.json`` instead ships ``num_masks_per_instruction=0.5``
   (a different exponent for ``m``) and an additional
   ``max_masks_per_instruction=8`` cap the paper's text never mentions.
   This module uses the paper's ``n``/``m`` formulas verbatim, uncapped —
   see :meth:`UniGuardianDetector._variants_for`.
2. **Sigmoid, not softmax.** The paper's uncertainty score (Sec. 4.1) is
   explicit: ``S_i = (1/k) sum_j (sigma(L_i,j) - sigma(L_b,j))^2``, "where
   sigma(.) is the sigmoid function" — the word "softmax" appears nowhere
   in the paper. The released ``UniGuardian/src/logits_guider.py:41-46``
   instead computes ``F.softmax(...)``. This module uses ``torch.sigmoid``
   verbatim — see :func:`uniguardian_divergence`.
3. **Mean over steps, not sum.** The paper's own ``(1/k)`` prefactor is an
   explicit mean over the ``k`` continuation steps. The released
   ``UniGuardian/src/backdoor_detector.py:92-95`` sums instead
   (``loss_list = [sum(x) for x in loss_list]``). This module means, per
   :func:`uniguardian_divergence`.

Everything else below is either unspecified by the paper (and taken from
the released code as an engineering detail, not a conflict) or a testbed
adapter this project's interface requires:

* prompt masking: ``UniGuardian/src/backdoor_detector.py:29-52`` splits on
  whitespace and replaces selected word-index tuples with
  ``tokenizer.mask_token``; candidate positions are filtered by
  ``clean_mask_indices`` (``backdoor_detector.py:54-68``, drops
  empty/whitespace-only pieces) — matches the paper's own Figure 2 example
  of masking word positions.
* index-tuple generation: ``UniGuardian/src/utils.py:86-109``'s
  dedup-with-retry loop for drawing ``n`` distinct ``m``-sized subsets — the
  paper's own notation (``M = {x_1...x_m}, x_i ~ Uniform(0, l-1)``) does not
  specify dedup/retry mechanics, so the code's convention is kept (adapter 3
  below only changes the *random source*, not this logic).
* generation alignment: ``UniGuardian/src/logits_guider.py:26-48`` — every
  step, the *base* (unmasked) prompt's own sampled token is captured and
  then force-substituted onto every other (masked) row *before* that row's
  own next step, so every variant is teacher-forced along the base's
  decoded path while its own pre-substitution logits at that position are
  what gets compared — matches the paper's own description of this
  alignment (Sec. 4.2).
* the vocabulary-dimension reduction inside each step's squared difference
  (``.mean()`` over ``v``) is likewise unspecified by the paper's
  ``sigma(L_i,j)`` notation once ``L_i,j`` is read as the full logits
  vector for step ``j``; the code's mean is kept (:func:`uniguardian_divergence`).
* z-score and its **maximum** as the suspicion score: the paper states both
  explicitly (Sec. 4.1, "we standardize them using z-scores ... we define
  the highest z-score among them as the suspicion score") — this is the
  paper's own reduction rule. ``(loss - mean(loss)) / std(loss)`` across the
  masked variants of **one** input is intra-sample and self-referential;
  the released code has **no** external clean-data calibration stage at all
  (see adapter 6 below for how this project supplies one anyway).

**Adapters — deliberate departures from the released code, each documented
at its point of use below:**

1. *Engine.* The released code hooks vLLM's per-step sampler
   (``UniGuardian/src/backdoor_detector.py:26``,
   ``UniGuardian/src/logits_guider.py:50-56``) to force one shared token per
   decoding step across a batch. This project deliberately does not install
   vLLM. Because the continuation to force is *already fully known* before
   any masked-variant forward pass runs (it is the base prompt's own
   already-computed generation or predicted verbalizer), the exact same
   per-step comparison is obtained with a **single batched forward pass per
   variant** over ``prompt_tokens + continuation_tokens`` and a causal
   shift — the standard teacher-forced-scoring pattern. This produces the
   identical set of (sigmoid, sigmoid) pairs at the identical positions as
   the official per-step hook; it is an engineering substitution, not a
   change to the divergence formula, the positions compared, or the
   teacher-forcing semantics.
2. *Mask token.* The released code searches
   ``tokenizer.added_tokens_decoder`` for a token whose content contains the
   literal substring ``"unused"`` or ``"reserved"``, and falls back to
   ``tokenizer.add_special_tokens(...)`` with its matching
   ``resize_token_embeddings`` call **commented out**
   (``UniGuardian/src/backdoor_detector.py:16-23``) — a latent bug. Qwen3's
   tokenizer has no token matching that heuristic (verified against the
   real ``Qwen/Qwen3-1.7B`` tokenizer vocabulary). This detector instead
   reuses one of Qwen3's existing multimodal placeholder tokens
   (``<|vision_pad|>``/``<|video_pad|>``/``<|image_pad|>``) — tokens that
   already occupy a row in the embedding table, are guaranteed single-token
   by the tokenizer's added-token mechanism, and carry no learned signal for
   a text-only model, serving the same structural role BERT's
   ``[unused_N]`` slots (and the official heuristic) were reaching for. No
   ``add_special_tokens`` or ``resize_token_embeddings`` call is ever made
   by this module.
3. *Random source.* The released ``generate_indices_tuples`` mutates the
   global stdlib ``random`` module with a **hardcoded** ``random.seed(42)``
   (``UniGuardian/src/utils.py:87``) — unsafe in a shared process and not
   reproducible per-input. This module uses a scoped
   ``numpy.random.Generator`` seeded from ``(seed, sha256(text))``, so
   masking is deterministic *per input text* regardless of batch order or
   position.
4. *Zero-variance guard.* The released z-score divides by ``std(loss)``
   unguarded. This module floors the denominator so a degenerate
   (zero-spread) set of masked variants returns ``0.0`` instead of
   ``nan``/``inf``.
5. *Scalar reduction.* The paper states the reduction explicitly ("we
   define the highest z-score among them as the suspicion score", Sec.
   4.1); the released code's ``main.py:89-91`` instead prints the full
   ``local_z_score_list`` vector (to additionally rank *which word* looks
   most suspicious, a diagnostic the released driver script wants that
   this project's :class:`~aiforensics.detectors.base.Detector` contract —
   exactly one float per sample, :meth:`Detector.score` — does not need).
   This module reduces with ``max()``, matching the paper's own rule.
6. *Clean-data calibration.* Because the released code has no external
   clean-data calibration stage, this module's ``.fit()`` is the harmless
   no-op inherited from :class:`Detector` — it does not need a clean
   reference, and ``requires_fit`` is left ``False``. The *threshold* this
   project reports at is still computed exclusively from clean validation
   scores via
   :func:`aiforensics.metrics.detection.calibrate_threshold_on_clean`, by
   the calling script, exactly like every other detector in this codebase.
   Poisoned samples are never read by this module during masking, scoring,
   or any calibration step.
7. *Masking scope.* Both studies insert their trigger only into the
   attacker-controlled instruction/review text, never into a system prompt.
   This detector therefore restricts masking to that attacker-controlled
   span when it can be recovered from the rendered prompt string: the last
   ``<|im_start|>user ... <|im_end|>`` turn for a chat-templated (Study B)
   prompt, or the ``{text}`` slot of a known
   :data:`aiforensics.training.sft.SFT_FORMATS` template for a raw (Study A)
   prompt. When neither pattern is found, the **whole** rendered prompt is
   masked and this fallback is recorded in ``self.config`` under
   ``"masked_scope_samples"`` (one tag per sample scored in the most recent
   call) so a system prompt is never silently included in the maskable pool
   without it being visible in the run's recorded detector configuration.

**Failure handling -- corrected, not merely documented.** An earlier
version of this module ended :meth:`score` with
``np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)`` and returned ``0.0``
for an empty base continuation -- the same class of defect independently
found and fixed in :mod:`aiforensics.detectors.rap_generative` and
:mod:`aiforensics.detectors.strip_vita_generative`: a genuine computation
failure (a non-finite logit, sigmoid, or divergence; an unexpectedly empty
generation) would have been silently indistinguishable from an ordinary,
correctly-computed score. Confirmed live this session with injected
failures (mocked forward passes, no model): an empty continuation and a
NaN injected into one masked variant's logits both produced a plain
``0.0``/``nan``-then-zeroed score under the old code, with no error and no
record. This is now corrected: :meth:`_score_one` validates finiteness
after the base forward pass, after every masked-variant forward-pass batch,
and over the aggregated per-variant divergences, raising a named
:class:`ValueError` at the first non-finite value or empty continuation
rather than letting it reach the output array; :meth:`score` itself also
asserts the final array is finite before returning (an invariant that
should now be unreachable, kept as a named guard rather than trusted
silently). Two DIFFERENT, GENUINE, NON-FAILURE states are explicitly NOT
changed by this fix, and must not be confused with it: (a) an input with no
maskable content at all (``len(variants) < 2`` in :meth:`_score_one`) still
returns ``0.0`` -- structurally reachable only when the maskable span is
empty, never from an exhausted retry loop (:func:`select_mask_index_tuples`
always returns >= 1 tuple whenever its candidate pool is non-empty); (b) a
genuine zero-spread set of masked-variant divergences still returns
all-zero z-scores via :func:`local_zscores`'s floored denominator, not
``nan`` -- a real, finite, meaningful result, not a failure. Whether the
old silent-zero path ever actually fired in any of this project's existing
UniGuardian reports is UNKNOWN: none of those reports recorded a failure
count, empty-continuation count, or non-finite-score count the way
:mod:`aiforensics.detectors.rap_generative`'s ``excluded_counts`` or
:mod:`aiforensics.detectors.strip_vita_generative`'s raise-on-failure
design would have made visible. Absence of evidence in those reports is not
evidence of absence; this audit does not claim the old reports are either
affected or unaffected by this defect.

Nothing in this module trains, fine-tunes, or otherwise mutates model
weights: every forward pass runs under ``torch.no_grad()`` and
``model.generate`` is used exactly as this project's other Study-B code
already uses it (greedy, ``do_sample=False``).

This module never reads ground-truth poison labels, poisoned indices, known
trigger strings, attack-goal labels, or target-label ground truth — its only
inputs are ``model``, ``tokenizer``, and the already-rendered prompt strings
passed to :meth:`UniGuardianDetector.score`.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..training.sft import SFT_FORMATS
from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["UniGuardianDetector"]

#: Existing Qwen3 vocabulary entries that already occupy an embedding row,
#: are guaranteed to tokenize as exactly one token (registered "added
#: tokens" are matched atomically regardless of BPE merges), and carry no
#: learned meaning for a text-only model. Verified against the real
#: ``Qwen/Qwen3-1.7B`` tokenizer: no ``mask_token`` is configured, and none
#: of its added tokens contain "unused"/"reserved" (the official code's own
#: search heuristic, ``UniGuardian/src/backdoor_detector.py:17-20``, would
#: not find any of them). These three are Qwen's multimodal padding
#: placeholders; a pure-text checkpoint never sees them in training data.
#: Tried in order; the first one present in the tokenizer's vocabulary wins.
_MASK_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<|vision_pad|>",
    "<|video_pad|>",
    "<|image_pad|>",
)

#: Matches the last chat-template user turn in an already-rendered prompt,
#: e.g. Qwen's ChatML ``<|im_start|>user\n...\n<|im_end|>``. Group 2 is the
#: attacker-controlled instruction text this detector prefers to mask.
_CHATML_USER_RE = re.compile(r"(<\|im_start\|>user\n)(.*?)(<\|im_end\|>)", re.DOTALL)


# --------------------------------------------------------------------------- #
# pure functions — no model/tokenizer needed, unit-tested standalone
# --------------------------------------------------------------------------- #
def resolve_mask_token(tokenizer: Any, override: str | None = None) -> tuple[str, int]:
    """Pick an existing, single-token vocabulary entry to use as the mask.

    Never calls ``tokenizer.add_special_tokens`` or
    ``model.resize_token_embeddings`` — see the module docstring, adapter 2.

    Args:
        tokenizer: A HuggingFace tokenizer.
        override: Force a specific token instead of the default candidate
            list; still validated against the tokenizer's own vocabulary.

    Returns:
        ``(token_string, token_id)``.

    Raises:
        ValueError: If no safe existing token can be found.
    """
    vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    candidates = (override,) if override is not None else _MASK_TOKEN_CANDIDATES
    for cand in candidates:
        tid = vocab.get(cand)
        if tid is None:
            continue
        # Belt-and-braces: confirm it round-trips as exactly one token id.
        ids = tokenizer.convert_tokens_to_ids([cand])
        if ids and ids[0] == tid:
            return cand, int(tid)
    raise ValueError(
        f"UniGuardianDetector found no existing reserved token to reuse as a mask "
        f"placeholder among {list(candidates)!r}. Resolving this requires either "
        f"tokenizer.add_special_tokens(...) followed by "
        f"model.resize_token_embeddings(...), or picking a different existing "
        f"vocabulary entry — this detector refuses to do either automatically."
    )


def split_maskable_words(text: str) -> tuple[list[str], list[int]]:
    """Split ``text`` into pieces (words and separators), and index which are maskable.

    Mirrors ``UniGuardian/src/backdoor_detector.py:29-52,54-68``: split on
    whitespace, keep separators as their own pieces (so joining reproduces
    the original text exactly), and mark a piece maskable iff it is not
    empty and not pure whitespace.

    Returns:
        ``(pieces, maskable_indices)``.
    """
    pieces = re.split(r"(\s+)", text)
    maskable = [i for i, p in enumerate(pieces) if p.strip()]
    return pieces, maskable


def select_mask_index_tuples(
    maskable_indices: Sequence[int],
    n_variants: int,
    k: int,
    rng: np.random.Generator,
    *,
    max_retries: int = 20,
) -> list[tuple[int, ...]]:
    """Draw up to ``n_variants`` distinct, unordered ``k``-subsets of positions.

    Mirrors ``UniGuardian/src/utils.py:86-109``'s dedup-with-retry loop,
    using a scoped ``numpy.random.Generator`` instead of the released code's
    hardcoded, process-global ``random.seed(42)`` (module docstring,
    adapter 3).

    Args:
        maskable_indices: Candidate positions.
        n_variants: How many distinct subsets to try to draw.
        k: Subset size (clipped to ``len(maskable_indices)``).
        rng: Seeded generator, deterministic for a given seed.
        max_retries: Consecutive duplicate draws tolerated before giving up
            -- matches the official ``NUM_RETRY`` RETRY-THEN-GIVE-UP
            *mechanism* (``utils.py:84``), not its count: the released code
            hardcodes ``NUM_RETRY=5``, this module defaults to 20 (more
            patient, so it gives up on reaching ``n_variants`` less readily)
            -- a deliberate, now explicit, deviation, not a claimed exact
            match.

    Returns:
        A list of sorted index tuples, each unique, length ``<= n_variants``
        (fewer than requested if the combinatorial space is exhausted).
    """
    pool = list(maskable_indices)
    k = max(1, min(k, len(pool)))
    if not pool:
        return []
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    retries = max_retries
    while len(out) < n_variants and retries > 0:
        draw = tuple(sorted(rng.choice(pool, size=k, replace=False).tolist()))
        if draw in seen:
            retries -= 1
            continue
        seen.add(draw)
        out.append(draw)
        retries = max_retries
    return out


def apply_word_mask(pieces: Sequence[str], indices: Sequence[int], mask_token: str) -> str:
    """Replace the pieces at ``indices`` with ``mask_token`` and rejoin.

    Every other piece — including whitespace separators — is left byte-for-
    byte unchanged, matching ``UniGuardian/src/backdoor_detector.py:40-47``.
    """
    marked = [mask_token if i in set(indices) else p for i, p in enumerate(pieces)]
    return "".join(marked)


def extract_maskable_span(prompt: str) -> tuple[str, str, str, str]:
    """Isolate the attacker-controlled portion of an already-rendered prompt.

    See the module docstring, adapter 7, for why this exists: both studies'
    triggers live only in the user instruction, never the system prompt, and
    this detector's ``.score()`` receives only the flat rendered string (the
    same input every other text detector in this codebase receives), so the
    boundary has to be recovered from the string itself.

    Returns:
        ``(prefix, maskable_middle, suffix, scope)`` such that
        ``prefix + maskable_middle + suffix == prompt`` exactly, and
        ``scope`` names which rule matched (or that none did, in which case
        the whole prompt is returned as ``maskable_middle`` and the fallback
        is named so it can be recorded rather than applied silently).
    """
    matches = list(_CHATML_USER_RE.finditer(prompt))
    if matches:
        m = matches[-1]
        start, end = m.span(2)
        return prompt[:start], prompt[start:end], prompt[end:], "chatml_user_turn"

    for fmt in SFT_FORMATS.values():
        if "{text}" not in fmt.template:
            continue
        pre, _, suf = fmt.template.partition("{text}")
        if prompt.startswith(pre) and (not suf or prompt.endswith(suf)):
            end = len(prompt) - len(suf) if suf else len(prompt)
            if end > len(pre):
                return prompt[: len(pre)], prompt[len(pre) : end], prompt[end:], f"sft_format:{fmt.key}"

    return "", prompt, "", "full_prompt (no chat-template or known SFT-format markers found)"


def uniguardian_divergence(base_sigmoid: np.ndarray, variant_sigmoid: np.ndarray) -> float:
    """The paper's uncertainty-score formula, reproduced exactly.

    Paper Sec. 4.1: ``S_i = (1/k) * sum_{j=1}^{k} (sigma(L_i,j) -
    sigma(L_b,j))^2``, where ``sigma(.)`` is **explicitly stated to be the
    sigmoid function** ("where sigma(.) is the sigmoid function") and ``k``
    is the number of continuation tokens. This is a genuine paper-vs-code
    conflict: the released ``UniGuardian/src/logits_guider.py:41-46`` uses
    ``F.softmax`` instead of sigmoid, and sums over steps
    (``UniGuardian/src/backdoor_detector.py:92-95``: ``sum(x)``) instead of
    the paper's explicit ``(1/k)`` mean. Per this project's source-authority
    order, the explicit paper equation wins on both points; the released
    code's per-step *mean over the vocabulary dimension* (``.mean()`` in the
    same line) is kept, since the paper never specifies how the vocab
    dimension collapses once ``sigma(L_i,j)`` is treated as the full
    logits-vector for step ``j`` — an unspecified engineering detail the
    code is allowed to fill in. Not KL divergence, not cosine distance, and
    not restricted to the last step — both arrays carry one row per
    continuation token.

    Args:
        base_sigmoid: ``(L, vocab)`` sigmoid-transformed logits from the
            unmasked prompt, one row per continuation position.
        variant_sigmoid: ``(L, vocab)`` sigmoid-transformed logits from one
            masked variant, same shape.

    Returns:
        A single non-negative float.
    """
    diff2 = (base_sigmoid - variant_sigmoid) ** 2
    per_step = diff2.mean(axis=-1)  # mean over vocab (unspecified by the paper)
    return float(per_step.mean())  # mean over steps -- paper's explicit (1/k) factor


def local_zscores(losses: np.ndarray) -> np.ndarray:
    """Intra-sample z-scores across one input's masked variants.

    ``UniGuardian/src/backdoor_detector.py:100-108``, with a floored
    denominator (module docstring, adapter 4) so a degenerate
    (zero-spread) loss vector returns all-zero z-scores instead of
    ``nan``/``inf`` — the official code has no such guard.
    """
    losses = np.asarray(losses, dtype=np.float64)
    if losses.size == 0:
        return losses
    mu = float(losses.mean())
    sigma = float(losses.std())
    floor = 1e-12 * max(abs(mu), 1.0)
    sigma = max(sigma, floor)
    return (losses - mu) / sigma


def reduce_to_scalar(zscores: np.ndarray) -> float:
    """``max(local_zscores)`` -- the paper's own reduction rule, see module docstring adapter 5."""
    if zscores.size == 0:
        return 0.0
    return float(np.max(zscores))


def _seed_for_text(seed: int, text: str) -> np.random.Generator:
    """Deterministic per-text RNG: same text -> same masks, any batch order."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    text_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng([int(seed), text_seed])


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
@register_detector("uniguardian", category="input-anomaly")
class UniGuardianDetector(Detector):
    """Masked-prompt, teacher-forced generation-divergence backdoor detector.

    See the module docstring for the full mechanism, the exact formulas
    reproduced, and every departure from the released reference code.

    Args:
        model: The causal LM under investigation (already on its device).
        tokenizer: Its tokenizer.
        mode: ``"generation"`` (one free greedy completion is generated
            from the unmasked prompt and used as the forced continuation)
            or ``"classification"`` (the model's own predicted verbalizer
            completion -- determined without reading any ground-truth
            label -- is used as the forced continuation). Defaults to
            ``"generation"``, which needs no extra keyword arguments and
            is what the released paper's own mechanism describes directly
            (generate a base continuation, then compare masked variants
            against it); this default lets this detector run through the
            existing runners' fully generic detector-construction path
            (``kw = {"model": ..., "tokenizer": ...}``) for *both* studies
            without any additional per-study wiring.
        verbalizer: Required when ``mode="classification"``; the candidate
            completion strings to score (e.g. ``(" negative", " positive")``,
            matching :data:`aiforensics.training.sft.SFT_FORMATS`). Not
            needed with the default ``mode="generation"``.
        max_new_tokens: Generation budget when ``mode="generation"``.
        max_prompt_length: Truncation budget for the rendered prompt. The
            prompt side is truncated from the **left** when it overflows
            (a trigger inserted near the end of a long input must not be
            silently discarded).
        num_masked_variants_scale: The paper's explicit ``n = 2 x length(prompt)``
            default (Sec. 5, "Hyper-parameters"); ``length(prompt)`` is this
            input's own maskable-word count ``l`` (module docstring adapter
            7's scope, once recovered), so ``N = max(1, int(l *
            num_masked_variants_scale))`` is computed **per input**, never a
            fixed constant. Not capped for speed.
        num_masks_power: The paper's explicit ``m = max(1, length(prompt)**0.3)``
            exponent (same section). The released code's shipped
            ``config.json`` instead defaults this exponent to ``0.5`` and
            additionally caps ``m`` at ``max_masks_per_instruction=8`` — a
            genuine paper-vs-code conflict; the explicit paper value
            (``0.3``) is used, uncapped.
        mask_token: Override the auto-resolved mask token (see
            :func:`resolve_mask_token`); leave ``None`` to use the verified
            default.
        seed: Base seed for deterministic, per-text masking.
        chunk_size: Masked variants scored per forward-pass batch, to bound
            GPU memory. Never changes ``N``/``m``, which masks get drawn, or
            the base continuation. It is NOT bitwise-neutral on the score,
            though -- corrected claim, measured this session: on 6 real
            dos/word clean prompts (both arms), a fixed set of masked
            variants scored at ``chunk_size=1`` vs. the production default
            (``4``) gave IDENTICAL sigmoid outputs on a repeated run at the
            same chunk size (run-to-run noise floor exactly ``0.0``), but
            differed between chunk sizes by up to 0.50 max / ~0.005 mean
            absolute sigmoid units, propagating to per-variant divergence
            differences of ~1e-4 to 2e-3 and a final max-z SCORE difference
            of 0.003 to 0.054 across those 6 prompts -- ordinary batched-GEMM
            floating-point noise (present throughout this project's other
            bf16 inference paths), not something this chunk_size value
            introduces uniquely, and not diluted away by the vocab-mean
            inside :func:`uniguardian_divergence`: a perturbation that large
            after averaging over ~152k vocabulary entries must be broadly
            distributed, not one outlier entry. ``chunk_size`` is applied
            identically to calibration and evaluation within one run, so
            this is a measured PRECISION characteristic of the statistic at
            its production configuration, not an inconsistency between
            differently-configured runs -- and no fix is applied here, since
            no logic defect was found (see :func:`_forward_sigmoid_batch`'s
            per-row length tracking and attention masking, unchanged).
    """

    name = "uniguardian"
    category = "input-anomaly"
    modality = "text"
    requires_fit = False

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        mode: str = "generation",
        verbalizer: Sequence[str] | None = None,
        max_new_tokens: int = 64,
        max_prompt_length: int = 512,
        num_masked_variants_scale: float = 2.0,
        num_masks_power: float = 0.3,
        mask_token: str | None = None,
        seed: int = 1337,
        chunk_size: int = 4,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, mode=mode, max_new_tokens=max_new_tokens,
            max_prompt_length=max_prompt_length,
            num_masked_variants_scale=num_masked_variants_scale,
            num_masks_power=num_masks_power,
            seed=seed, chunk_size=chunk_size, **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("UniGuardianDetector needs model= and tokenizer=")
        if mode not in {"generation", "classification"}:
            raise ValueError(f"mode must be 'generation' or 'classification', got {mode!r}")
        if mode == "classification" and not verbalizer:
            raise ValueError(
                "mode='classification' needs verbalizer=(...) — the candidate "
                "completion strings to score, e.g. (' negative', ' positive')"
            )

        self.model, self.tokenizer = model, tokenizer
        self.mode = mode
        self.verbalizer = tuple(verbalizer) if verbalizer else None
        self.max_new_tokens = int(max_new_tokens)
        self.max_prompt_length = int(max_prompt_length)
        self.num_masked_variants_scale = float(num_masked_variants_scale)
        self.num_masks_power = float(num_masks_power)
        self.seed = int(seed)
        self.chunk_size = max(1, int(chunk_size))

        self.mask_token, self.mask_token_id = resolve_mask_token(tokenizer, mask_token)
        self.config["mask_token"] = self.mask_token
        self.config["mask_token_id"] = self.mask_token_id

    # -- text-only helpers (unit-tested independent of the model) --------- #
    def _variants_for(self, prompt: str) -> tuple[list[str], str]:
        """Build ``[prompt] + masked_variants`` and the masking-scope tag.

        ``n`` and ``m`` are both derived from ``l = len(maskable)`` (this
        input's own maskable-word count) via the paper's explicit formulas
        -- ``n = max(1, int(l * num_masked_variants_scale))``,
        ``m = max(1, min(int(l ** num_masks_power), l))`` -- never a fixed
        constant and never capped beyond the structural bound ``m <= l``.
        """
        prefix, middle, suffix, scope = extract_maskable_span(prompt)
        pieces, maskable = split_maskable_words(middle)
        if not maskable:
            return [prompt], scope
        length_l = len(maskable)
        k = max(1, min(int(length_l ** self.num_masks_power), length_l))
        n_variants = max(1, int(length_l * self.num_masked_variants_scale))
        rng = _seed_for_text(self.seed, prompt)
        tuples = select_mask_index_tuples(maskable, n_variants, k, rng)
        if not tuples:
            return [prompt], scope
        variants = [
            prefix + apply_word_mask(pieces, idx, self.mask_token) + suffix for idx in tuples
        ]
        return [prompt, *variants], scope

    # -- model-backed helpers ---------------------------------------------- #
    def _left_truncate(self, ids: list[int]) -> list[int]:
        if len(ids) > self.max_prompt_length:
            return ids[-self.max_prompt_length :]
        return ids

    def _base_continuation_ids(self, prompt: str) -> list[int]:
        """The forced continuation: a free greedy generation, or the model's
        own predicted verbalizer completion (no ground-truth label used)."""
        if self.mode == "generation":
            return self._greedy_generate_ids(prompt)
        return self._predicted_verbalizer_ids(prompt)

    def _greedy_generate_ids(self, prompt: str) -> list[int]:
        import torch

        from ..models.llm import model_device

        device = model_device(self.model)
        p_ids = self._left_truncate(self.tokenizer(prompt)["input_ids"])
        input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn,
                max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=pad_id, return_dict_in_generate=True,
            )
        sequences = out.sequences if hasattr(out, "sequences") else out
        return sequences[0, input_ids.shape[1] :].detach().cpu().tolist()

    def _predicted_verbalizer_ids(self, prompt: str) -> list[int]:
        """Score every verbalizer candidate and return the argmax's token ids.

        Applied to an already-rendered prompt (``fmt.prompt`` must not be
        re-applied -- the caller already rendered it, matching every other
        text detector's input convention). No ground-truth label is read
        anywhere here.
        """
        import torch
        import torch.nn.functional as F

        from ..models.llm import model_device

        assert self.verbalizer is not None
        device = model_device(self.model)
        eos = self.tokenizer.eos_token_id
        p_ids = self._left_truncate(self.tokenizer(prompt)["input_ids"])

        candidates: list[list[int]] = []
        for completion in self.verbalizer:
            c_ids = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
            candidates.append([*c_ids, eos] if eos is not None else list(c_ids))

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos
        rows = [p_ids + c for c in candidates]
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attn[i, : len(r)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits.float()

        scores = []
        for i, c in enumerate(candidates):
            plen, length = len(p_ids), len(c)
            step_logits = logits[i, plen - 1 : plen - 1 + length, :]
            logprobs = F.log_softmax(step_logits, dim=-1)
            tgt = torch.tensor(c, device=logprobs.device)
            token_lp = logprobs[torch.arange(length, device=logprobs.device), tgt]
            scores.append(float(token_lp.sum().item()))
        return candidates[int(np.argmax(scores))]

    def _forward_sigmoid_batch(
        self, prompts: Sequence[str], continuation_ids: Sequence[int]
    ) -> np.ndarray:
        """``(len(prompts), L, vocab)`` next-token sigmoid-logits at every
        continuation step -- the paper's ``sigma(L_i,j)`` (module docstring,
        :func:`uniguardian_divergence`), **not** a softmax probability
        distribution.

        One batched forward pass over ``prompt_tokens + continuation_tokens``
        per row, reading logits at the ``L`` positions that predict the
        continuation -- the engineering substitution for the official
        per-step vLLM hook (module docstring, adapter 1). ``continuation_ids``
        is fixed once (from the base prompt) and reused verbatim for every
        row, so every variant is compared at literally the same forced
        tokens, never re-tokenized per variant.
        """
        import torch

        from ..models.llm import model_device

        device = model_device(self.model)
        cont = list(continuation_ids)
        length = len(cont)
        rows, prompt_lens = [], []
        for p in prompts:
            p_ids = self._left_truncate(self.tokenizer(p)["input_ids"])
            rows.append([*p_ids, *cont])
            prompt_lens.append(len(p_ids))

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        width = max(len(r) for r in rows)
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
            attn[i, : len(r)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits.float()

        out = np.empty((len(rows), length, logits.shape[-1]), dtype=np.float64)
        for i, plen in enumerate(prompt_lens):
            step_logits = logits[i, plen - 1 : plen - 1 + length, :]
            out[i] = torch.sigmoid(step_logits).cpu().numpy()
        return out

    def _score_one(self, prompt: str, index: Any = None) -> tuple[float, str]:
        """See the module docstring, "failure handling", for the distinction
        this method now enforces: genuine no-maskable-content and genuine
        zero-variance (both real, non-failure states) return an ordinary
        ``0.0``; anything that could only come from a computation going
        wrong -- an empty base continuation, or a non-finite logit,
        divergence, or score -- raises instead of silently becoming ``0.0``.
        """
        variants, scope = self._variants_for(prompt)
        if len(variants) < 2:
            # Genuine "no maskable content" -- not a computational failure.
            # Structurally the ONLY way to reach this: select_mask_index_tuples
            # always returns >= 1 tuple whenever its own candidate pool is
            # non-empty (its first draw can never collide with an empty
            # `seen` set), so `variants` is short only when `maskable` itself
            # was empty in `_variants_for`.
            return 0.0, scope

        continuation_ids = self._base_continuation_ids(prompt)
        if not continuation_ids:
            raise ValueError(
                f"UniGuardianDetector: sample index={index!r} produced an EMPTY base "
                f"continuation -- refusing to silently score this as 0.0. Reachable "
                f"only with max_new_tokens=0 in mode='generation', or an empty "
                f"verbalizer candidate in mode='classification'; neither is expected "
                f"at the shipped defaults, so this signals a real problem worth "
                f"seeing, not a value to hide."
            )

        base_sig = self._forward_sigmoid_batch([prompt], continuation_ids)[0]
        if not np.isfinite(base_sig).all():
            raise ValueError(
                f"UniGuardianDetector: sample index={index!r}'s BASE forward pass "
                f"produced non-finite sigmoid logits -- refusing to silently convert "
                f"this to a score."
            )

        masked = variants[1:]
        losses = np.empty(len(masked), dtype=np.float64)
        for start in range(0, len(masked), self.chunk_size):
            chunk = masked[start : start + self.chunk_size]
            sigmoids = self._forward_sigmoid_batch(chunk, continuation_ids)
            if not np.isfinite(sigmoids).all():
                raise ValueError(
                    f"UniGuardianDetector: sample index={index!r} produced non-finite "
                    f"sigmoid logits in a masked-variant forward pass (variants "
                    f"starting at {start}) -- refusing to silently convert this to a "
                    f"score."
                )
            for j, sm in enumerate(sigmoids):
                losses[start + j] = uniguardian_divergence(base_sig, sm)

        if not np.isfinite(losses).all():
            raise ValueError(
                f"UniGuardianDetector: sample index={index!r} produced a non-finite "
                f"per-variant divergence -- refusing to silently convert this to a "
                f"score."
            )

        z = local_zscores(losses)
        return reduce_to_scalar(z), scope

    # -- the one required method ------------------------------------------- #
    def score(self, samples: Any) -> np.ndarray:
        """One suspicion score per prompt. Higher = more suspicious.

        Args:
            samples: A string or a sequence of already-rendered prompt
                strings -- the same convention every other text detector in
                this codebase uses (``fmt.prompt(text)`` for Study A, the
                rendered chat template for Study B).

        Returns:
            ``(n_texts,)`` float array.

        Raises:
            ValueError: If any sample's scoring produced a non-finite
                result or other signal of a genuine computational failure
                (module docstring, "failure handling") -- never silently
                converted to ``0.0``.
        """
        texts = [samples] if isinstance(samples, str) else list(samples)
        out = np.zeros(len(texts), dtype=np.float64)
        scopes: list[str] = []
        from tqdm import tqdm

        for i, text in tqdm(
            enumerate(texts), total=len(texts), desc="UniGuardian score",
            unit="sample", dynamic_ncols=True,
        ):
            score, scope = self._score_one(text, index=i)
            out[i] = score
            scopes.append(scope)
        # Never mask a system prompt silently (module docstring, adapter 7).
        # Overwritten, not accumulated, each call -- a record of the most
        # recent scoring pass, not a running log.
        self.config["masked_scope_samples"] = scopes
        if not np.isfinite(out).all():
            # Should be unreachable given _score_one's own per-stage checks
            # above; kept as a final, named guard rather than trusting that
            # invariant silently -- matches this project's established
            # "a computation failure must never become a valid-looking
            # score" convention (see rap_generative.py / strip_vita_generative.py).
            raise ValueError(
                "UniGuardianDetector.score: the output array contains a non-finite "
                "value that survived per-sample validation in _score_one -- refusing "
                "to silently zero it via nan_to_num."
            )
        return out
