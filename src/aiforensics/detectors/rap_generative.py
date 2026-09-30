"""RAP -- generative-task adaptation (Study B). NOT the classification RAP.

This is a research adaptation, not a reproduction. RAP (Yang et al., EMNLP
2021; see :mod:`aiforensics.detectors.rap` and its module docstring for the
classification-task derivation) is built entirely around a class-probability
distribution over a small, fixed, project-wide verbalizer -- Study B has no
such structure (open-ended generation, four objectives with no shared class
axis; confirmed by reading ``src/aiforensics/attacks/objectives.py`` and
``src/aiforensics/data/objective_corpus.py`` directly: ``dos``/
``context-extraction`` are scored by text-quality/leakage metrics with no
class at all, ``belief``'s two "classes" are per-row attacker/true answers
that a detector must never read, and ``refusal-suppression``'s binary outcome
is decided by substring-matching 20 marker phrases against a free response,
not a single fixed teacher-forced completion). This module does NOT reuse
RAP's classification statistic; it reuses only the *mechanism* -- fit one
rare word's embedding on clean data, then measure whether inserting it
degrades a probability by a controlled amount -- redefining "probability"
for a task with no classes.

**The statistic.** For a prompt ``x``:

1. Generate ``ref`` -- a free greedy continuation of ``x`` -- ONCE, using
   the model's ORIGINAL (never-perturbed) embedding table. Never regenerated
   between fit and score for the same row.
2. Define ``p_ref(prompt) = exp(mean_t log P(ref_t | prompt, ref_<t)))`` --
   the length-normalised (geometric-mean-per-token) teacher-forced
   probability of ``ref`` under any given prompt. Bounded in ``(0, 1]`` like
   a classification probability, unlike raw joint log-likelihood, which
   collapses geometrically with response length. This bounded scale is what
   lets RAP's own hyperparameters be carried over as *candidates*.
3. ``diff(x) = p_ref(x+RAP) - p_ref(x)``, both terms computed with the SAME
   embedding state installed (see "tied embeddings", below).
4. Fit: two-sided hinge loss on ``diff``, target range and every other
   hyperparameter carried over from RAP's own paper defaults --
   ``probability_range=(-0.1, -0.3)``, ``lr=1e-2``, ``epochs=5``,
   ``batch_size=32``, ``scale_factor=1.0`` -- on CLEAN calibration prompts
   only, no target-label filter (Study B has no target label).
5. Score: ``diff(x)``, no gate (see "no gate", below).

**Perturbation scope: the WHOLE ChatML user turn, not "instruction only" --
kept deliberately, not narrowed.** ``scripts/run_objective_attack.py``'s
``render_prompt`` joins ``instruction`` and ``context`` with two newlines
into ONE user message
(``f"{instruction}\\n\\n{context}".strip() if context else instruction``),
and :func:`aiforensics.detectors.uniguardian.extract_maskable_span` returns
the LAST ChatML user turn as one span -- so the maskable/perturbable region
this module actually operates on is ``instruction + "\\n\\n" + context``
together, confirmed this session by rendering real prompts and checking
``prefix + middle + suffix == prompt`` exactly. ``context`` is therefore
INSIDE the defense's perturbation scope, not excluded from it, and the RAP
word lands at the very start of ``instruction`` specifically only because
``position=0`` targets the first word of the whole span (:meth:`_poison_one`
inserts into ``middle`` as a whole, and ``instruction`` happens to be first).
:func:`_verified_span` enforces ``scope == "chatml_user_turn"`` explicitly
and refuses (raises) on anything else -- including
:func:`extract_maskable_span`'s own ``"full_prompt"`` fallback for an
unrecognised template, which is NEVER accepted here (unlike a caller that
tolerates it): silently perturbing a system prompt or role marker would be a
correctness bug this module will not risk. No special-casing protects the
attack trigger's own position -- perturbation eligibility is computed
identically regardless of where the (unknown, to this detector) attack
trigger happens to sit.

**Tied embeddings -- corrected, not assumed.** An earlier draft claimed
mutating the RAP row was "provably invariant" for any prompt that never
contains the RAP token as input. That is WRONG for a model with tied
input/output embeddings: the same row is also read by the output (lm_head)
projection at EVERY position, for EVERY prompt. Confirmed live against
Qwen/Qwen3-1.7B (``get_input_embeddings().weight.data_ptr() ==
get_output_embeddings().weight.data_ptr()`` -- True) and measured on the
real model (max ~1e-6 probability shift on RAP-token-free prompts under a
realistic-scale perturbation -- negligible, not assumed). Both terms of
every ``diff`` comparison are computed under the SAME installed embedding
state, matching :class:`aiforensics.detectors.rap.RapDetector`'s own
convention, so any residual contamination is common-mode within a
comparison, not eliminated.

**This detector ONLY supports a model whose input and output embeddings are
tied**, and checks this explicitly (:meth:`_ensure_ready`), raising rather
than silently copying the fitted input row into a genuinely separate output
embedding tensor for an unsupported, untied model -- an earlier draft did
exactly that copy, which would apply a learned direction to an unrelated
parameter for such a model. Qwen/Qwen3-1.7B, this project's actual target,
is tied, so this refusal is not expected to fire here; it exists so an
untied model fails loudly instead of scoring silently wrong.

**Truncation side -- LEFT, a deliberate, measured divergence from
``encode_texts``'s default.** ``scripts/run_objective_attack.py``'s own
``generate()`` truncates via
``encode_texts(..., padding_side="left", max_length=512)``
(``src/aiforensics/models/llm.py``), and ``padding_side`` does NOT control
truncation side -- confirmed by reading ``encode_texts`` and
``_padding_side`` directly: only ``tokenizer.padding_side`` is touched,
never ``tokenizer.truncation_side``. The real Qwen3-1.7B tokenizer's
``truncation_side`` is ``"right"`` by default, and empirically (this
session, on a real over-length prompt) right-truncation deletes the TAIL --
including the entire ``<|im_start|>assistant`` opener and ``<think>`` block
-- which would leave `generate()` with no marker for where to start
responding, and would leave teacher-forced scoring comparing a reference
against context-filler text rather than a genuine assistant turn. This
module therefore truncates from the LEFT (``ids[-max_length:]``), matching
:class:`aiforensics.detectors.uniguardian.UniGuardianDetector`'s own choice,
NOT ``encode_texts``'s default -- verified this keeps the assistant opener
intact on the one real over-length sample found this session.

**Measured impact, not assumed.** Across all 8 (objective, trigger)
conditions' real ``cal_fit``/``cal_thr``/``eval_clean``/``eval_trig`` pools
(2400 rendered prompts total), exactly 1 row per condition exceeds 512
tokens, and only for ``dos``/``context-extraction`` (never for
``refusal-suppression``/``belief``, whose prompts are short by
construction) -- ``max_length`` was left at 512 rather than raised, per this
session's instruction not to expand context length for a theoretical risk
once the real affected count was known to be tiny. For the one such row
found (``dos``/``word`` ``cal_fit`` index 59: clean 632 tokens, RAP token at
position 14 of 629 poisoned tokens), left-truncation keeps the assistant
opener intact but DOES drop the RAP token (117 tokens truncated from the
front, RAP token at position 14). This is not ignored: :meth:`_prepare_row`
verifies, per row, that the RAP token actually survives in the final
(possibly-truncated) poisoned-prompt token sequence, and excludes -- with an
explicit, counted, printed reason, never silently -- any row where it does
not (see "unscored rows", below). Generation and teacher-forced scoring
truncate identically (:meth:`_encode_prompt`, used by both), so the two
never compare different context windows for the same input.

**Unscored rows -- explicit, counted, never silently included or excluded
from the denominator.** :meth:`_prepare_row` checks three conditions before
a row is treated as scorable:

1. The RAP token must survive tokenisation and truncation in the POISONED
   prompt (see "measured impact", above) -- otherwise the insertion never
   reached the model and any ``diff`` would not measure an insertion effect
   at all.
2. The RAP token must NOT already be present in the CLEAN prompt's own
   tokenisation -- if it is (never observed this session: the candidates
   have zero occurrences in a 358,392-word scan of the real Dolly corpus,
   but not assumed impossible), ``p_ref(x)`` would already be evaluated
   under a model whose relevant embedding row is deliberately perturbed
   input-side, confounding "clean" with "already contains the probe".
3. The reference must be non-empty.

A row failing any of these is EXCLUDED from fitting (with a printed,
counted reason -- fitting continues on whatever rows remain, or raises if
none do) and scored as ``-inf`` at evaluation time (this codebase's
established "no signal" convention -- ranked least suspicious, still
counted in ``n``, never dropped from the denominator). The reference
containing the RAP token is explicitly NOT one of these exclusion
conditions -- an earlier draft excluded such rows from fitting while still
scoring them at evaluation time, an inconsistency with "every input scored
normally". The formula remains well-defined when ``ref`` contains the RAP
token (it is simply a fixed target sequence being teacher-forced; no
division by zero or degenerate case is introduced), so such rows are scored
like any other.

**No gate.** RAP's classification gate (``p(target|x) > 0.5``) is defined
relative to a fixed external class and has no Study-B analogue; inventing a
substitute threshold would be an extra free parameter with no clean-data-
only justification. Every prompt not caught by the exclusions above gets a
real score.

**EOS is the FULL stop-token set ``model.generate()`` actually uses, not
just ``tokenizer.eos_token_id``.** Confirmed this session:
Qwen3-1.7B's own ``generation_config.json`` sets ``eos_token_id`` to
``[151645, 151643]`` (``<|im_end|>`` AND the base ``<|endoftext|>``/pad id)
-- ``model.generate()`` reads ``model.generation_config.eos_token_id``,
which HF permits as an int or a list, and stops on ANY member.
:func:`_eos_token_ids` reads this directly (falling back to
``tokenizer.eos_token_id`` alone only if the model exposes no
``generation_config``), so a stop triggered by the second id is not missed.

**Fitting hyperparameters -- carried over as UNVALIDATED CANDIDATES, not
re-tuned.** ``probability_range=(-0.1, -0.3)``, ``lr=1e-2``, ``epochs=5``,
``batch_size=32``, ``scale_factor=1.0`` are RAP's own paper defaults,
reused because ``p_ref`` lives on the same bounded scale a classification
probability does -- not because they have been shown optimal, or even
achievable at full scale, for this statistic. What has actually been
checked: on 6 real clean calibration prompts (dos/word), ``p_ref(x)``
ranged 0.330-0.916, comfortably above the 0.3 floor the target range needs
(``diff >= -p_ref(x)`` algebraically). A 3-step, 6-sample smoke fit showed a
nonzero, non-vanishing gradient and row update under bf16 storage every
step. Chosen using ONLY clean fitting data; ``cal_thr`` and every
triggered/eval score were never read while making this choice, and none of
these numbers were chosen to maximise a detection rate.

**Fitting diagnostics, made visible and correctly named.** ``fit()`` prints
per-batch progress AND records ``self.config["fit_diagnostics"]``,
containing BOTH: (a) the last mini-batch's PRE-UPDATE loss/diff
(``last_batch_pre_update_loss`` -- explicitly named as exactly that, not
"the fitted model's loss", since a single small batch is not representative
of the whole fitting set), and (b) a genuine before/after comparison --
``initial_full_set`` (ORIGINAL row) and ``final_full_set`` (FITTED row),
each a no-grad pass over every scorable fitting row using the SAME fixed
references, giving comparable loss/diff-range/in-range-fraction numbers.
Neither is used to adjust the loss, the gradient, or any hyperparameter, and
neither ever reads ``cal_thr`` or a triggered/test score.

**Memory: per-sample gradient accumulation, not a whole batch's forward
graphs held at once.** Each sample's before/after forward pass, loss, and
``backward()`` (on ``loss_i / actual_batch_size``) happens one sample at a
time, with ``model.zero_grad()`` called before EACH sample so its own
gradient can be read off in isolation and summed into an FP32 accumulator
(see "row-update precision", below) rather than left to accumulate across
the whole batch inside PyTorch's own BF16 ``.grad`` tensor via repeated
un-zeroed ``backward()`` calls. The per-sample gradient VALUES this produces
are identical either way (verified on a toy causal model at batch sizes 1,
5 and 32: matching gradients and loss values to within ~1e-11 absolute /
~1e-7 relative, pure floating-point summation-order noise) -- only where
they get summed changed. The weight update still happens once per batch,
after all its samples' gradients have accumulated, peak memory is still one
sample's graph alive at a time, and sample order, learning rate, epochs,
and effective batch size are all unchanged.

**Row-update precision -- FP32 master copy, corrected from a BF16-only
update.** An earlier version of this module accumulated each batch's
gradient directly into the embedding weight's own BF16 ``.grad`` tensor and
applied ``weight.data[row] -= lr * grad`` and the norm rescale in BF16
storage. Measured live against the real model (this session, dos/phrase
Study B, the actual first seeded ``cal_fit`` minibatch): the per-step FP32
gradient norm was real and non-vanishing (~0.025-0.04 over the row's 2048
dims), but the BF16 subtraction rounded roughly 95-97% of the row's 2048
elements to EXACTLY their pre-update value every step (only 3-5% of
elements changed at all), and the 3-5% that did change had only 0.24-0.39
cosine similarity to the FP32-computed update direction -- closer to
quantization noise than to the real gradient. The norm rescale
(``weight.data[row] *= original_norm / new_norm``) was independently a
complete no-op at this scale: ``new_norm`` computed via BF16 ``.norm()``
carries ~0.1-0.2% of its own reduction-rounding noise (measured: BF16 norm
1.656250 vs the TRUE FP32 norm of that same, barely-changed vector,
1.653140 -- a 0.0031 discrepancy against a true per-step norm change of
only ~0.0002), and BF16's own ~0.4% value spacing near 1.0 then rounds
whatever ratio results to exactly 1.0 regardless of the true value. Neither
failure is visible from "gradient is nonzero" or "some element changed"
alone -- both were true every step under the old code, and fitting still
barely moved (real dos/phrase v2 run: full-fit-set loss 0.08862793 ->
0.08871712, in-target-range fraction 2/149 -> 2/149, after all 5 epochs).
Fixed by keeping a FLOAT32 master copy of the row (accumulator, subtraction,
and norm rescale all in FP32; ``original_norm`` itself is now measured via
``original_row.float().norm()``, not a BF16 reduction), casting to the
model's own storage dtype ONLY when installing the row for a forward pass
(:meth:`_install_row`). The model itself is never cast to FP32: every
forward and backward pass that PRODUCES a per-sample gradient still executes
in BF16 (aside from casting logits to FP32 before ``log_softmax`` in
:meth:`_mean_logprob`, unchanged by this fix) -- this remains a genuine,
un-eliminated precision limitation of the gradient VALUE itself, distinct
from the (now fixed) precision loss in how that gradient gets accumulated,
applied, and rescaled. The per-sample-accumulation cross-check earlier in
this docstring's history measured that residual BF16 forward/backward noise
at ~0.7-1% relative on the accumulated gradient -- real, but an order of
magnitude smaller than the ~95% information loss the update-application fix
addresses. This fixes how faithfully the computed update is actually
applied; it does NOT by
itself establish that the resulting fitted row degrades a poisoned prompt's
probability by the target amount, or that any v2 report's detection numbers
would change, or by how much -- that requires an actual rerun, not
performed as part of this fix. Every existing v2 report
(``*_rap_gen_add_v2.json``) was produced under the old BF16-only update and
is not re-derivable from this change; a rerun to measure its effect should
use a new tag (e.g. ``_rap_gen_add_v3``), never overwrite a v2 report.

Nothing in this module trains, saves, or otherwise permanently mutates a
model. Exactly one embedding row is ever written, always restored in a
``finally`` block, exception-safe on both the normal and the crash path
(verified on the real model).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..data.poison import RARE_WORD_TRIGGERS, insert_trigger
from .base import Detector, register_detector
from .uniguardian import extract_maskable_span

logger = logging.getLogger(__name__)

__all__ = ["RapGenerativeDetector", "resolve_rap_token_in_context"]

#: Re-verified in the actual Study B chat-template context against the real
#: Qwen/Qwen3-1.7B tokenizer: all three tokenize as exactly one id there, and
#: all three have zero occurrences in a 358,392-word scan of the real,
#: cached Dolly-15k corpus.
_RAP_CANDIDATES: tuple[str, ...] = ("mb", "bb", "mn")


def _verified_span(prompt: str) -> tuple[str, str, str]:
    """Isolate the maskable span, refusing anything but a genuine ChatML
    user turn. See the module docstring, "perturbation scope".

    Raises:
        ValueError: If the scope is not ``"chatml_user_turn"`` (including
            :func:`extract_maskable_span`'s own ``"full_prompt"`` fallback
            for an unrecognised template -- never accepted here), or if the
            three pieces do not reconstruct the original prompt exactly.
    """
    prefix, middle, suffix, scope = extract_maskable_span(prompt)
    if scope != "chatml_user_turn":
        raise ValueError(
            f"RapGenerativeDetector requires a genuine ChatML user-turn span; "
            f"extract_maskable_span reported scope={scope!r} for this prompt "
            f"(first 80 chars: {prompt[:80]!r}). Refusing rather than perturbing "
            f"an unverified span."
        )
    if prefix + middle + suffix != prompt:
        raise ValueError(
            "RapGenerativeDetector: prefix+middle+suffix did not reconstruct the "
            "original prompt exactly -- refusing to perturb an unverified span."
        )
    return prefix, middle, suffix


def resolve_rap_token_in_context(
    tokenizer: Any,
    prefix: str,
    suffix: str,
    *,
    override: str | None = None,
    candidates: Sequence[str] = _RAP_CANDIDATES,
) -> tuple[str, int]:
    """Pick a rare word verified single-token IN THE ACTUAL Study B context.

    Unlike :func:`aiforensics.detectors.rap.resolve_rap_token`, this does
    NOT assume a leading-space boundary -- Study B's chat template puts the
    instruction right after a literal ``"user\\n"``, a different
    tokenization context than Study A's ``"Review: "`` prefix. Tokenizes a
    genuine ``prefix + poisoned_middle + suffix`` probe and checks, directly,
    that (a) exactly one new token appears and (b) that token's own string
    equals the candidate exactly -- robust to BPE retokenisation of
    neighbouring text (observed directly: inserting a token can shift how
    the FOLLOWING word retokenises; harmless for scoring, since every
    poisoned prompt is tokenized fresh from its own string, never spliced
    from the clean prompt's ids -- but it does mean "count of new tokens"
    alone is not proof of single-token-ness without also checking identity).

    Args:
        tokenizer: The real tokenizer of the model under investigation.
        prefix: Text before the maskable span (from :func:`_verified_span`).
        suffix: Text after the maskable span.
        override: Force a specific candidate instead of scanning
            ``candidates``; still fully re-verified.
        candidates: Scan order; overridable for tests only.

    Returns:
        ``(token_string, token_id)``.

    Raises:
        ValueError: If no candidate (or the override) qualifies.
    """
    probe_clean_ids = tokenizer(prefix + "x" + suffix, add_special_tokens=False)["input_ids"]
    pool = (override,) if override is not None else tuple(candidates)
    tried: list[str] = []
    for cand in pool:
        if cand == RARE_WORD_TRIGGERS[0]:
            tried.append(f"{cand!r}: equals the configured word-trigger literal")
            continue
        poisoned_middle = insert_trigger("x", cand, position=0)
        probe_poisoned_ids = tokenizer(prefix + poisoned_middle + suffix, add_special_tokens=False)["input_ids"]
        delta = len(probe_poisoned_ids) - len(probe_clean_ids)
        if delta != 1:
            tried.append(f"{cand!r}: inserting it changed token count by {delta}, expected 1")
            continue
        i = 0
        while i < len(probe_clean_ids) and probe_clean_ids[i] == probe_poisoned_ids[i]:
            i += 1
        inserted_id = probe_poisoned_ids[i]
        inserted_str = tokenizer.convert_ids_to_tokens([inserted_id])[0]
        bare = inserted_str.lstrip("Ġ▁")
        if bare != cand:
            tried.append(f"{cand!r}: the inserted token was {inserted_str!r}, not {cand!r}")
            continue
        return cand, int(inserted_id)
    raise ValueError(
        f"RapGenerativeDetector found no valid defense token among {list(pool)!r}: "
        f"{'; '.join(tried)}. Pass a different override= after checking a candidate "
        f"the same way this function does."
    )


def _eos_token_ids(model: Any, tokenizer: Any) -> frozenset[int]:
    """The FULL set of token ids ``model.generate()`` actually stops on --
    see the module docstring, "EOS is the full stop-token set".
    """
    ids: set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    raw = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if raw is not None:
        if isinstance(raw, int):
            ids.add(int(raw))
        else:
            ids.update(int(x) for x in raw)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    return frozenset(ids)


@register_detector("rap-generative", category="input-anomaly")
class RapGenerativeDetector(Detector):
    """RAP, generative-task adaptation. See the module docstring for the
    full derivation and every correction/caveat.

    Args:
        model, tokenizer: The model under investigation and its tokenizer
            (already carrying whichever adapter -- poisoned or control --
            is being scored; construct one instance per arm). Must have
            tied input/output embeddings -- checked, and refused if not.
        rap_token: Override the auto-resolved candidate; still fully
            re-verified in context (see :func:`resolve_rap_token_in_context`).
        probability_range: ``(upper, lower)`` -- RAP's own example values,
            carried over as an UNVALIDATED CANDIDATE (see module docstring).
        lr, epochs, batch_size, scale_factor: Fitting hyperparameters,
            RAP's own defaults, likewise unvalidated at full scale.
        max_new_tokens: Reference-generation budget -- matches Study B's
            own ``generate()`` budget.
        max_length: Prompt-truncation budget, LEFT-truncated (module
            docstring, "truncation side").
        seed: Shuffles fitting batch order deterministically.
        show_progress: Display a ``tqdm`` bar and per-batch diagnostic lines.
    """

    name = "rap-generative"
    category = "input-anomaly"
    modality = "text"
    requires_fit = True

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        rap_token: str | None = None,
        probability_range: tuple[float, float] = (-0.1, -0.3),
        lr: float = 1e-2,
        epochs: int = 5,
        batch_size: int = 32,
        scale_factor: float = 1.0,
        max_new_tokens: int = 64,
        max_length: int = 512,
        seed: int = 1337,
        show_progress: bool = True,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, probability_range=probability_range, lr=lr, epochs=epochs,
            batch_size=batch_size, scale_factor=scale_factor, max_new_tokens=max_new_tokens,
            max_length=max_length, seed=seed, **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("RapGenerativeDetector needs model= and tokenizer=")
        lo, hi = probability_range
        if not (lo > hi):
            raise ValueError(f"probability_range=(upper, lower) needs upper > lower, got {probability_range}")
        if epochs < 1 or batch_size < 1 or max_new_tokens < 1 or max_length < 1:
            raise ValueError(
                f"epochs, batch_size, max_new_tokens and max_length must all be >= 1, got "
                f"epochs={epochs}, batch_size={batch_size}, max_new_tokens={max_new_tokens}, "
                f"max_length={max_length}"
            )
        self.model, self.tokenizer = model, tokenizer
        self.probability_range = (float(lo), float(hi))
        self.lr, self.epochs, self.batch_size = float(lr), int(epochs), int(batch_size)
        self.scale_factor = float(scale_factor)
        self.max_new_tokens, self.max_length = int(max_new_tokens), int(max_length)
        self.seed = int(seed)
        self.show_progress = bool(show_progress)
        self.rap_token_override = rap_token
        self.rap_token: str | None = None
        self.rap_token_id: int | None = None
        self._eos_ids: frozenset[int] | None = None
        self._tied_checked = False
        self._fitted_row: Any = None

    # -- span / poison / readiness helpers -------------------------------- #
    def _span(self, prompt: str) -> tuple[str, str, str]:
        return _verified_span(prompt)

    def _poison_one(self, prompt: str) -> str:
        prefix, middle, suffix = self._span(prompt)
        return prefix + insert_trigger(middle, self.rap_token, position=0) + suffix

    def _embedding_weight(self) -> Any:
        return self.model.get_input_embeddings().weight

    def _ensure_ready(self, sample_prompt: str) -> None:
        """Resolve the defense token, the real EOS set, and verify tied
        embeddings -- once, lazily, the first time either fit() or score()
        actually needs a real prompt to work from."""
        if self.rap_token is None:
            prefix, _middle, suffix = self._span(sample_prompt)
            self.rap_token, self.rap_token_id = resolve_rap_token_in_context(
                self.tokenizer, prefix, suffix, override=self.rap_token_override,
            )
            self.config["rap_token"] = self.rap_token
            self.config["rap_token_id"] = self.rap_token_id
        if self._eos_ids is None:
            self._eos_ids = _eos_token_ids(self.model, self.tokenizer)
        if not self._tied_checked:
            out_emb = self.model.get_output_embeddings()
            if out_emb is not None and hasattr(out_emb, "weight"):
                if out_emb.weight.data_ptr() != self._embedding_weight().data_ptr():
                    raise ValueError(
                        "RapGenerativeDetector only supports a model whose input and "
                        "output embeddings are tied (the same tensor). This model's are "
                        "not -- refusing rather than reusing the fitted input row's value "
                        "for a genuinely separate output embedding."
                    )
            self._tied_checked = True

    def _install_row(self, row: Any) -> None:
        import torch

        with torch.no_grad():
            self._embedding_weight().data[self.rap_token_id] = row

    # -- tokenisation, deliberately LEFT-truncated (module docstring) ----- #
    def _encode_prompt(self, prompt: str) -> list[int]:
        ids = self.tokenizer(prompt)["input_ids"]
        if len(ids) > self.max_length:
            ids = ids[-self.max_length:]
        return ids

    def _prepare_row(self, prompt: str, ref_ids: Sequence[int]) -> tuple[str, str | None]:
        """Validate one row is genuinely scorable. See the module
        docstring, "unscored rows".

        Returns:
            ``(poisoned_prompt, exclude_reason)`` -- ``exclude_reason`` is
            ``None`` for a scorable row.
        """
        poisoned_prompt = self._poison_one(prompt)
        if not ref_ids:
            return poisoned_prompt, "empty_reference"
        clean_ids = self._encode_prompt(prompt)
        poisoned_ids = self._encode_prompt(poisoned_prompt)
        if self.rap_token_id not in poisoned_ids:
            return poisoned_prompt, "rap_token_truncated_away"
        if self.rap_token_id in clean_ids:
            return poisoned_prompt, "rap_token_present_in_clean_input"
        return poisoned_prompt, None

    # -- generation / scoring primitives ---------------------------------- #
    def _greedy_generate_ids(self, prompt: str) -> list[int]:
        import torch

        from ..models.llm import model_device

        device = model_device(self.model)
        p_ids = self._encode_prompt(prompt)
        input_ids = torch.tensor([p_ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attn, max_new_tokens=self.max_new_tokens,
                do_sample=False, pad_token_id=pad_id, return_dict_in_generate=True,
            )
        seq = out.sequences if hasattr(out, "sequences") else out
        ids = seq[0, input_ids.shape[1]:].detach().cpu().tolist()
        for i, tid in enumerate(ids):
            if tid in self._eos_ids:
                return ids[: i + 1]
        return ids

    def _mean_logprob(self, prompt: str, ref_ids: Sequence[int], *, track_grad: bool) -> Any:
        """Length-normalised (mean) teacher-forced log-prob of ``ref_ids``
        under ``prompt``, at the CURRENTLY installed embedding state. Uses
        the SAME left-truncated encoding as :meth:`_greedy_generate_ids`
        (via :meth:`_encode_prompt`), so generation and teacher-forced
        scoring never compare different context windows for one input."""
        import torch

        from ..models.llm import model_device

        device = model_device(self.model)
        p_ids = self._encode_prompt(prompt)
        full = p_ids + list(ref_ids)
        input_ids = torch.tensor([full], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        ctx = torch.enable_grad() if track_grad else torch.no_grad()
        with ctx:
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits.float()
            logp = torch.log_softmax(logits[0], dim=-1)
            plen = len(p_ids)
            step_lp = logp[plen - 1: plen - 1 + len(ref_ids), :]
            tgt = torch.tensor(list(ref_ids), device=device)
            token_lp = step_lp[torch.arange(len(ref_ids), device=device), tgt]
            return token_lp.mean()

    def _evaluate_full_set(
        self, texts: list[str], poisoned: list[str], refs: list[list[int]], upper: float, lower: float,
    ) -> dict[str, float]:
        """No-grad pass over every row at the CURRENTLY installed embedding
        state, for the before/after fitting diagnostic (module docstring,
        "fitting diagnostics")."""
        import torch

        diffs = []
        with torch.no_grad():
            for t, pt, ref in zip(texts, poisoned, refs, strict=True):
                lp_before = self._mean_logprob(t, ref, track_grad=False)
                lp_after = self._mean_logprob(pt, ref, track_grad=False)
                diffs.append(float(torch.exp(lp_after) - torch.exp(lp_before)))
        diff_arr = np.array(diffs, dtype=np.float64)
        if not np.isfinite(diff_arr).all():
            raise ValueError("non-finite diff encountered while evaluating the full fit set")
        loss = float(np.mean(
            self.scale_factor * np.clip(diff_arr - upper, 0.0, None) + np.clip(lower - diff_arr, 0.0, None)
        ))
        return {
            "loss": loss, "diff_min": float(diff_arr.min()), "diff_max": float(diff_arr.max()),
            "diff_mean": float(diff_arr.mean()),
            "in_range_frac": float(((diff_arr >= lower) & (diff_arr <= upper)).mean()),
        }

    # -- the required methods --------------------------------------------- #
    def fit(self, clean_data: Any) -> RapGenerativeDetector:
        """Fit the perturbation on clean prompts only (no label filter --
        Study B has no target label). See the module docstring for the
        full derivation of every check performed here.

        Args:
            clean_data: Already-rendered clean Study B prompts.

        Raises:
            ValueError: If every row is excluded (see "unscored rows" in
                the module docstring), or a non-finite diff is encountered.
        """
        import torch

        texts = list(clean_data)
        if not texts:
            raise ValueError("RapGenerativeDetector.fit needs at least one clean prompt")
        self._ensure_ready(texts[0])

        weight = self._embedding_weight()
        original_row = weight.data[self.rap_token_id].detach().clone()
        original_norm = float(original_row.float().norm().item())
        original_requires_grad = [(p, p.requires_grad) for p in self.model.parameters()]

        progress = self._progress(len(texts), "RAP-gen fit: preparing rows")
        excluded: dict[str, int] = {}
        try:
            # Phase 1: references under ORIGINAL weights; validate each row.
            self._install_row(original_row)
            rows: list[tuple[str, str, list[int]]] = []
            for t in texts:
                ref = self._greedy_generate_ids(t)
                poisoned_t, reason = self._prepare_row(t, ref)
                if reason is not None:
                    excluded[reason] = excluded.get(reason, 0) + 1
                else:
                    rows.append((t, poisoned_t, ref))
                progress.update(1)
            progress.close()

            if not rows:
                raise ValueError(
                    f"RapGenerativeDetector.fit: every one of {len(texts)} clean rows was "
                    f"excluded, none scorable -- exclusion counts: {excluded}"
                )
            if excluded:
                print(f"  RAP-gen fit: excluded {sum(excluded.values())}/{len(texts)} rows "
                      f"({excluded}); fitting on the remaining {len(rows)}.")

            texts_k = [r[0] for r in rows]
            poisoned_k = [r[1] for r in rows]
            refs_k = [r[2] for r in rows]
            n = len(rows)
            upper, lower = self.probability_range

            # Baseline (pre-fit) full-set evaluation -- ORIGINAL row still installed.
            baseline = self._evaluate_full_set(texts_k, poisoned_k, refs_k, upper, lower)

            # Phase 2: gradient descent. Accumulation and the row update
            # itself happen in an FP32 MASTER COPY of the row, cast to the
            # model's own dtype only when installed for a forward pass --
            # see the module docstring, "row-update precision", for the
            # measured BF16-only failure this corrects.
            for p in self.model.parameters():
                p.requires_grad_(False)
            weight.requires_grad_(True)
            master_row = original_row.float().clone()

            rng = np.random.default_rng(self.seed)
            n_batches = self.epochs * (-(-n // self.batch_size))
            fit_progress = self._progress(n_batches, "RAP-gen fit")
            step_records: list[dict[str, float]] = []
            for epoch in range(self.epochs):
                order = rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    idx = order[start: start + self.batch_size]
                    actual_bs = len(idx)
                    grad_accum = torch.zeros_like(master_row)
                    per_losses: list[float] = []
                    per_diffs: list[float] = []
                    for i in idx:
                        i = int(i)
                        self.model.zero_grad(set_to_none=True)
                        lp_before = self._mean_logprob(texts_k[i], refs_k[i], track_grad=True)
                        lp_after = self._mean_logprob(poisoned_k[i], refs_k[i], track_grad=True)
                        diff_i = torch.exp(lp_after) - torch.exp(lp_before)
                        if not torch.isfinite(diff_i):
                            raise ValueError(f"non-finite diff for fit row {i}: {float(diff_i)}")
                        l_i = (
                            self.scale_factor * torch.clamp(diff_i - upper, min=0.0)
                            + torch.clamp(lower - diff_i, min=0.0)
                        )
                        (l_i / actual_bs).backward()
                        grad_accum += weight.grad[self.rap_token_id].detach().float()
                        per_losses.append(float(l_i.detach()))
                        per_diffs.append(float(diff_i.detach()))
                    weight.grad = None
                    with torch.no_grad():
                        master_row -= self.lr * grad_accum
                        new_norm = master_row.norm()
                        if new_norm > 0:
                            master_row *= original_norm / new_norm
                        self._install_row(master_row.to(weight.dtype))

                    diff_arr = np.array(per_diffs)
                    in_range = float(((diff_arr >= lower) & (diff_arr <= upper)).mean())
                    rec = {
                        "epoch": epoch, "loss_last_batch_pre_update": float(np.mean(per_losses)),
                        "diff_min": float(diff_arr.min()), "diff_max": float(diff_arr.max()),
                        "diff_mean": float(diff_arr.mean()), "in_range_frac": in_range,
                    }
                    step_records.append(rec)
                    if self.show_progress:
                        print(
                            f"  RAP-gen fit  epoch={epoch} step={len(step_records)}  "
                            f"last_batch_pre_update_loss={rec['loss_last_batch_pre_update']:.6f}  "
                            f"diff=[{rec['diff_min']:.4f}, {rec['diff_max']:.4f}] "
                            f"mean={rec['diff_mean']:.4f}  "
                            f"in_target_range({lower:.2f},{upper:.2f})={rec['in_range_frac']:.2f}"
                        )
                    fit_progress.update(1)
            fit_progress.close()
            self._fitted_row = weight.data[self.rap_token_id].detach().clone()

            # Final (post-fit) full-set evaluation -- FITTED row installed.
            self._install_row(self._fitted_row)
            final = self._evaluate_full_set(texts_k, poisoned_k, refs_k, upper, lower)

            summary = {
                "n_fit_rows_total": len(texts), "n_fit_rows_scored": n,
                "excluded_counts": excluded,
                "last_batch_pre_update_loss": (
                    step_records[-1]["loss_last_batch_pre_update"] if step_records else None
                ),
                "target_range": [lower, upper],
                "initial_full_set": baseline, "final_full_set": final,
            }
            self.config["fit_diagnostics"] = summary
            print(
                f"  RAP-gen fit summary: {n}/{len(texts)} rows scored | "
                f"full-fit-set loss {baseline['loss']:.6f} -> {final['loss']:.6f} | "
                f"diff range [{baseline['diff_min']:.4f}, {baseline['diff_max']:.4f}] -> "
                f"[{final['diff_min']:.4f}, {final['diff_max']:.4f}] | "
                f"in-target-range fraction {baseline['in_range_frac']:.2f} -> "
                f"{final['in_range_frac']:.2f}"
            )
        finally:
            self._install_row(original_row)
            for p, flag in original_requires_grad:
                p.requires_grad_(flag)
            weight.grad = None

        self._fitted = True
        return self

    def score(self, samples: Any) -> np.ndarray:
        """``diff = p_ref(x+RAP) - p_ref(x)`` per input, both terms computed
        under the SAME (final fitted) embedding state; ``ref`` generated
        under the ORIGINAL state. Unscorable rows (module docstring,
        "unscored rows") return ``-inf``, counted and printed, never
        dropped from the output array's length.

        Raises:
            ValueError: If a non-finite score is produced for a row that
                passed the scorability checks.
        """
        import torch

        if self._fitted_row is None:
            raise RuntimeError("RapGenerativeDetector scored before fit()")
        texts = [samples] if isinstance(samples, str) else list(samples)
        self._ensure_ready(texts[0])
        out = np.full(len(texts), -np.inf, dtype=np.float64)

        weight = self._embedding_weight()
        original_row = weight.data[self.rap_token_id].detach().clone()
        progress = self._progress(2 * len(texts), "RAP-gen score")
        excluded: dict[str, int] = {}
        try:
            # Phase 1: references under ORIGINAL weights; validate each row.
            self._install_row(original_row)
            prepared: list[tuple[str, list[int] | None]] = []
            for t in texts:
                ref = self._greedy_generate_ids(t)
                poisoned_t, reason = self._prepare_row(t, ref)
                if reason is not None:
                    excluded[reason] = excluded.get(reason, 0) + 1
                    prepared.append((poisoned_t, None))
                else:
                    prepared.append((poisoned_t, ref))
                progress.update(1)

            # Phase 2: p_ref(x) / p_ref(x+RAP) under the FITTED row.
            self._install_row(self._fitted_row)
            with torch.no_grad():
                for i, (x, (xr, ref)) in enumerate(zip(texts, prepared, strict=True)):
                    if ref is None:
                        progress.update(1)
                        continue
                    lp_before = self._mean_logprob(x, ref, track_grad=False)
                    lp_after = self._mean_logprob(xr, ref, track_grad=False)
                    val = float(torch.exp(lp_after) - torch.exp(lp_before))
                    if not np.isfinite(val):
                        raise ValueError(f"non-finite score at index {i}: {val}")
                    out[i] = val
                    progress.update(1)
        finally:
            self._install_row(original_row)
            progress.close()

        if excluded:
            print(f"  RAP-gen score: {sum(excluded.values())}/{len(texts)} inputs unscored "
                  f"({excluded}) -- returned -inf (established 'no signal' convention), "
                  f"still counted in n, never dropped from the denominator.")
        self.config["score_excluded_counts"] = excluded
        return out

    def _progress(self, total: int, desc: str) -> Any:
        if self.show_progress:
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
