"""STRIP-ViTA -- generative-task adaptation (Study B). NOT the classification
STRIP-ViTA, and NOT the STRIP baseline bundled inside the RAP repository --
that is a different, THIRD PARTY's reimplementation of the ORIGINAL authors'
STRIP-ViTA method (a PyTorch/BERT port), not the authors' own code, and not
STRIP-ViTA itself. That third-party port's own properties -- natural log
where the ORIGINAL authors' own text-domain notebooks use log2 (a constant
scaling applied consistently here, not itself a detection bug; see
:mod:`aiforensics.detectors.strip_vita`'s module docstring, "corrected
provenance"), its ``TFIDF()`` iterating a raw string character-by-character
rather than its words (a genuine defect in that third-party file, NOT
inherited by :func:`aiforensics.detectors.strip_vita.build_tfidf`, which
tokenizes correctly), and its classification-only (fixed-label-vector)
entropy definition (a structural mismatch with open-ended generation, not a
defect in that file) -- are already documented, each classified on its own
terms, in :mod:`aiforensics.detectors.strip_vita`'s own module docstring and
are not re-derived here. This module inherits only that module's
TEXT-perturbation machinery, verbatim, not its entropy statistic.

**Why a new statistic is needed at all.** STRIP-ViTA's entropy is of a
class-probability VECTOR -- one softmax over a small, fixed label set,
produced by a single forward pass. Study B has no such vector (open-ended
generation; see :mod:`aiforensics.detectors.rap_generative`'s module
docstring for the same "no class axis" finding). Substituting "entropy of
the generated tokens" for "entropy of the class vector" is a genuine change
of what is being measured, not an equivalent restatement -- so this is
presented as an ADAPTATION, never as STRIP-ViTA itself.

**The statistic, fixed before any test was run:**

1. Perturbation is UNCHANGED from the classification version's own
   mechanism and defaults: :func:`aiforensics.detectors.strip_vita.build_tfidf`
   and :func:`aiforensics.detectors.strip_vita.perturb_one`, ``replace_ratio
   =0.7``, ``perturbation_number=20`` -- both reused verbatim, never
   modified here (this module never edits ``strip_vita.py``). One deviation,
   stated explicitly: the reference pool is no longer split by "opposite
   class" (Study B has no labels) -- it is a SINGLE TF-IDF pool built from
   the clean-calibration prompts handed to :meth:`fit`.
2. For EACH of the ``perturbation_number`` perturbed copies of a prompt:
   generate its OWN greedy continuation (``max_new_tokens=64``, matching
   ``scripts/run_objective_attack.py``'s own generation budget), then do
   ONE teacher-forced forward pass feeding that exact continuation back
   through the model to recover RAW logits at each generated position --
   the same generate-then-teacher-force pattern
   :class:`aiforensics.detectors.uniguardian.UniGuardianDetector` already
   uses in this codebase.
3. Natural-log Shannon entropy (matching
   :func:`aiforensics.detectors.strip_vita.entropy_of`'s own choice in this
   codebase, not the paper's stated log2) at each generated position.
4. EOS-inclusive, padding-excluded (see "EOS and length", below).
5. Per-copy score: mean entropy over that copy's own steps. Aggregate: mean
   over the ``perturbation_number`` copies. Final score: ``-aggregate``.
6. **No gate.** The classification gate (``argmax(logits) == target_label``)
   has no Study-B analogue; every prompt is scored.

**Perturbation scope: the WHOLE ChatML user turn, shared with
:mod:`aiforensics.detectors.rap_generative` -- see that module's docstring,
"perturbation scope", for the full argument (``render_prompt`` joins
instruction+context into one user message; ``context`` is inside scope, not
excluded; no ``full_prompt`` fallback accepted; no special-casing the
attack trigger's own position).** :func:`_verified_span` (this module's own
copy of the same check, kept self-contained rather than importing across
these two sibling files for an ~8-line function) enforces
``scope == "chatml_user_turn"`` in both :meth:`fit` and :meth:`score`.

**EOS is the FULL stop-token set ``model.generate()`` actually uses, not
just ``tokenizer.eos_token_id``** -- see
:mod:`aiforensics.detectors.rap_generative`'s module docstring for the
measurement (Qwen3-1.7B's ``generation_config.json`` sets ``eos_token_id``
to ``[151645, 151643]``, and ``model.generate()`` stops on either).
:func:`_eos_token_ids` (this module's own copy) reads this directly.

**EOS and length, corrected -- padding-avoidance and length-confound are
kept as separate claims, not conflated.** If the model's own choice at some
step is EOS, that step's entropy is included (real signal, not discarded);
a copy whose first generated token is EOS has a length-1 trajectory, well-
defined. Padding past a copy's own stop is excluded -- each copy's own
generated length is used, never a fixed window. This avoids a PADDING bias.
It does NOT, by itself, rule out a genuine LENGTH-entropy correlation for
reasons unrelated to any attack -- an open, only partially characterised
question (a real, tiny n=5 smoke check showed a visible correlation;
five points support no conclusion). This does not invalidate
``calibrate_threshold_on_clean`` (the clean false-positive rate is
controlled by construction, regardless of what shapes the clean score
distribution) but is an interpretability caveat for any eventual detection
result, belonging to this project's "how well does it detect" question, not
to whether the definition or implementation is sound. :meth:`score` records
each sample's generated lengths in ``self.config["generated_lengths"]``
(mirroring :class:`~aiforensics.detectors.uniguardian.UniGuardianDetector`'s
own ``masked_scope_samples`` convention) so this is inspectable on real
results, not invisible.

**A computation failure must never become a valid-looking score --
corrected this session.** An earlier draft returned ``0.0`` for a sample
whose perturbed copies produced no usable entropy, and separately ran the
whole output array through ``np.nan_to_num(..., nan=0.0, posinf=0.0,
neginf=0.0)``. Since the final score is ``-mean(entropy)`` and entropy is
non-negative, ``0.0`` sits at the HIGH end of the suspicion scale (entropy
0 is the most-confident, most-suspicious-looking value) -- a computation
failure was indistinguishable from, and would rank ABOVE, many genuine
low-entropy detections. This module now requires ALL of the following, or
raises immediately rather than silently degrading the result:

* ``perturbation_number >= 1`` and ``max_new_tokens >= 1``, checked at
  construction.
* EVERY one of the ``perturbation_number`` perturbed copies for a scored
  input must produce a non-empty generation and a finite per-copy mean
  entropy -- an empty generation or a non-finite entropy/logit anywhere
  raises :class:`ValueError` naming the sample and copy index, rather than
  silently averaging over fewer than ``perturbation_number`` copies (which
  would be reporting a reduced-perturbation result as if it were the full,
  agreed one).
* The final returned score array is asserted finite; ``np.nan_to_num`` is
  no longer applied anywhere in this module.

**Replacement-count validated before calling ``perturb_one`` -- the shared
``strip_vita.py`` is not modified to add this check itself.**
:func:`aiforensics.detectors.strip_vita.perturb_one` clips its replacement
count only against the sentence's own word count
(``n_replace = min(n_replace, len(words))``), never against the reference
vocabulary size -- confirmed by reading its source this session. If the
maskable span's own word count exceeds the pooled vocabulary size, calling
it would raise an unclear ``IndexError`` from inside a shared file this
module must not edit. :meth:`score` therefore computes the same
``n_replace`` :func:`perturb_one` will compute and verifies
``n_replace <= len(vocab)`` itself first, raising a clear, named error if
violated, rather than letting the shared function fail confusingly or
silently reducing the replacement ratio.

**TF-IDF ranking can select vocabulary words that carry ZERO weight in the
chosen reference row -- corrected description, not "all from the selected
sentence".** :func:`aiforensics.detectors.strip_vita.build_tfidf` builds
ONE vocabulary pooled across every reference row; a given row's own TF-IDF
vector is zero at every vocabulary entry that row does not itself contain.
:func:`~aiforensics.detectors.strip_vita.perturb_one` ranks the FULL pooled
vocabulary by ``argsort(-ref_tfidf_row)`` and can draw entries from deep in
that ranking (whenever the maskable span's word count exceeds the number of
words the selected reference row actually has nonzero weight for) -- ties at
zero break in whatever order ``argsort`` returns them, not by relevance to
the selected sentence. So the replacement words for a given perturbed copy
are not guaranteed to all come from the one reference sentence that was
drawn; some can be arbitrary other-row vocabulary entries pulled in only
because the ranking ran out of nonzero-weight words for that row. This is
:func:`perturb_one`'s own behaviour, unmodified -- described accurately
here, not changed.

**Truncation side -- LEFT, shared with the RAP adaptation (see that
module's docstring for the full measurement). Word-for-word replacement
preserves SPACE-DELIMITED WORD COUNT, not tokenizer TOKEN count -- an
earlier version of this docstring conflated the two and claimed "no NEW
truncation risk" as a general property of the method. That is false in
general: a replacement word can tokenize to more (or fewer) subword tokens
than the word it replaces, so a perturbed copy's TOKEN length can differ
from the original's, and could in principle newly cross ``max_length``
where the original did not.** Measured live this session (CPU only, real
tokenizer, the actual deterministic perturbation construction, no model
forward pass) on Study B belief/phrase/seed 1337 -- the condition this
matters for -- across ``cal_thr``, ``eval_clean`` and ``eval_trig`` (450
prompts, 9000 perturbed copies total): original prompts are 32-46 tokens,
the longest perturbed copy observed was 50 tokens, and NONE of the 9000
copies came within 10x of ``max_length=512``. Zero copies crossed the
boundary in either direction; the assistant opener was intact (trivially,
since nothing was ever truncated) in every case checked. For THIS
condition, truncation has no measurable effect on scoring and is not a
credible contributor to its detection result -- but this is a measurement
of one condition, not a proof that word-for-word replacement is always
token-length-safe; a condition with longer prompts nearer the 512-token
ceiling would need its own measurement, not an assumption from this one.

Nothing in this module trains, fine-tunes, or otherwise mutates model
weights: every forward pass runs under ``torch.no_grad()``.

This module never reads ground-truth poison labels, poisoned indices, known
trigger strings, attack-goal labels, target-label ground truth, or the
objective scorer's own verdict -- its only inputs are ``model``,
``tokenizer``, and already-rendered prompt strings.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector
from .strip_vita import build_tfidf, perturb_one
from .uniguardian import extract_maskable_span

logger = logging.getLogger(__name__)

__all__ = ["StripVitaGenerativeDetector"]


def _verified_span(prompt: str) -> tuple[str, str, str]:
    """Isolate the maskable span, refusing anything but a genuine ChatML
    user turn -- see the module docstring, "perturbation scope", and
    :func:`aiforensics.detectors.rap_generative._verified_span` (the same
    check, kept as an independent copy rather than a cross-import).
    """
    prefix, middle, suffix, scope = extract_maskable_span(prompt)
    if scope != "chatml_user_turn":
        raise ValueError(
            f"StripVitaGenerativeDetector requires a genuine ChatML user-turn span; "
            f"extract_maskable_span reported scope={scope!r} for this prompt "
            f"(first 80 chars: {prompt[:80]!r}). Refusing rather than perturbing "
            f"an unverified span."
        )
    if prefix + middle + suffix != prompt:
        raise ValueError(
            "StripVitaGenerativeDetector: prefix+middle+suffix did not reconstruct "
            "the original prompt exactly -- refusing to perturb an unverified span."
        )
    return prefix, middle, suffix


def _eos_token_ids(model: Any, tokenizer: Any) -> frozenset[int]:
    """The FULL set of token ids ``model.generate()`` actually stops on --
    see :mod:`aiforensics.detectors.rap_generative`'s module docstring,
    "EOS is the full stop-token set", for the measurement this reflects.
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


def _entropy_per_step(step_logits: Any) -> Any:
    """Natural-log Shannon entropy per row, ``(L,)`` from raw ``(L, vocab)``
    logits -- matches :func:`aiforensics.detectors.strip_vita.entropy_of`'s
    log base."""
    import torch

    p = torch.softmax(step_logits, dim=-1).clamp_min(1e-12)
    return -(p * torch.log(p)).sum(-1)


@register_detector("strip-vita-generative", category="input-anomaly")
class StripVitaGenerativeDetector(Detector):
    """STRIP-ViTA, generative-task adaptation. See the module docstring for
    the full derivation and every correction/caveat.

    Args:
        model, tokenizer: The model under investigation and its tokenizer
            (construct one instance per arm, like every other detector here).
        replace_ratio: Fraction of words replaced per perturbed copy --
            unchanged from the classification version's own default (0.7).
        perturbation_number: Perturbed copies averaged per scored input --
            unchanged from the classification version's own default (20).
            Every copy must succeed or :meth:`score` raises (module
            docstring, "a computation failure must never become a valid-
            looking score").
        max_new_tokens: Generation budget per perturbed copy.
        max_length: Prompt-truncation budget, LEFT-truncated (module
            docstring, "truncation side").
        seed: Base seed for deterministic, per-input perturbation draws.
        show_progress: Display a ``tqdm`` bar.
    """

    name = "strip-vita-generative"
    category = "input-anomaly"
    modality = "text"
    requires_fit = True

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        replace_ratio: float = 0.7,
        perturbation_number: int = 20,
        max_new_tokens: int = 64,
        max_length: int = 512,
        seed: int = 1337,
        show_progress: bool = True,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, replace_ratio=replace_ratio,
            perturbation_number=perturbation_number, max_new_tokens=max_new_tokens,
            max_length=max_length, seed=seed, **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("StripVitaGenerativeDetector needs model= and tokenizer=")
        if not (0.0 < replace_ratio <= 1.0):
            raise ValueError(f"replace_ratio must be in (0, 1], got {replace_ratio}")
        if perturbation_number < 1:
            raise ValueError(f"perturbation_number must be >= 1, got {perturbation_number}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")
        self.model, self.tokenizer = model, tokenizer
        self.replace_ratio = float(replace_ratio)
        self.perturbation_number = int(perturbation_number)
        self.max_new_tokens, self.max_length = int(max_new_tokens), int(max_length)
        self.seed = int(seed)
        self.show_progress = bool(show_progress)
        self._vocab: list[str] | None = None
        self._tfidf: Any = None
        self._eos_ids: frozenset[int] | None = None

    def _seed_for_text(self, text: str) -> np.random.Generator:
        import hashlib

        digest = hashlib.sha256(text.encode("utf-8")).digest()
        text_seed = int.from_bytes(digest[:8], "big")
        return np.random.default_rng([int(self.seed), text_seed])

    def fit(self, clean_data: Any) -> StripVitaGenerativeDetector:
        """Build ONE TF-IDF pool from clean prompts (no label split -- Study
        B has no classes). See the module docstring, point 1.

        Args:
            clean_data: Already-rendered clean Study B prompts.
        """
        texts = list(clean_data)
        if not texts:
            raise ValueError("StripVitaGenerativeDetector.fit needs at least one clean prompt")
        spans = [_verified_span(t) for t in texts]
        middles = [mid for _pre, mid, _suf in spans]
        self._vocab, self._tfidf = build_tfidf(middles)
        self._fitted = True
        return self

    def _greedy_generate_ids(self, prompt: str) -> list[int]:
        import torch

        from ..models.llm import model_device

        if self._eos_ids is None:
            self._eos_ids = _eos_token_ids(self.model, self.tokenizer)
        device = model_device(self.model)
        p_ids = self.tokenizer(prompt)["input_ids"]
        if len(p_ids) > self.max_length:
            p_ids = p_ids[-self.max_length:]
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

    def _teacher_forced_step_logits(self, prompt: str, cont_ids: Sequence[int]) -> Any:
        import torch

        from ..models.llm import model_device

        device = model_device(self.model)
        p_ids = self.tokenizer(prompt)["input_ids"]
        if len(p_ids) > self.max_length:
            p_ids = p_ids[-self.max_length:]
        full = p_ids + list(cont_ids)
        input_ids = torch.tensor([full], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attn, use_cache=False).logits.float()
        plen = len(p_ids)
        return logits[0, plen - 1: plen - 1 + len(cont_ids), :]

    def score(self, samples: Any) -> np.ndarray:
        """``-mean(entropy)`` over ``perturbation_number`` perturbed copies'
        own free generations, per input. Every input gets a real score --
        no gate. Generated lengths are recorded in
        ``self.config["generated_lengths"]``. See the module docstring for
        the validation this performs before trusting any score.

        Raises:
            ValueError: If the maskable span's replacement count would
                exceed the fitted vocabulary size, or if any perturbed copy
                produces an empty generation or a non-finite entropy.
        """
        if self._vocab is None or self._tfidf is None:
            raise RuntimeError("StripVitaGenerativeDetector scored before fit()")
        texts = [samples] if isinstance(samples, str) else list(samples)
        out = np.zeros(len(texts), dtype=np.float64)
        lengths_per_sample: list[list[int]] = []

        progress = None
        if self.show_progress:
            try:
                from tqdm import tqdm

                progress = tqdm(total=len(texts) * self.perturbation_number,
                                 desc="STRIP-ViTA-gen score", unit="copy")
            except ImportError:  # pragma: no cover - tqdm is pinned
                progress = None

        try:
            for i, text in enumerate(texts):
                pre, mid, suf = _verified_span(text)
                words = mid.split(" ")
                n_replace = max(int(len(words) * self.replace_ratio), 1)
                n_replace = min(n_replace, len(words))
                if n_replace > len(self._vocab):
                    raise ValueError(
                        f"StripVitaGenerativeDetector: sample {i} needs to replace "
                        f"{n_replace} words but the fitted vocabulary only has "
                        f"{len(self._vocab)} entries -- perturb_one would fail with an "
                        f"unclear IndexError. Not reducing the replacement ratio silently; "
                        f"fit on more/larger clean prompts, or lower replace_ratio explicitly."
                    )
                rng = self._seed_for_text(text)
                copy_mean_entropies: list[float] = []
                copy_lengths: list[int] = []
                for copy_idx in range(self.perturbation_number):
                    ref_row = int(rng.integers(0, self._tfidf.shape[0]))
                    perturbed_words = perturb_one(words, self.replace_ratio, self._vocab, self._tfidf[ref_row], rng)
                    perturbed_prompt = pre + " ".join(perturbed_words) + suf
                    gen_ids = self._greedy_generate_ids(perturbed_prompt)
                    if not gen_ids:
                        raise ValueError(
                            f"StripVitaGenerativeDetector: sample {i}, perturbed copy "
                            f"{copy_idx} produced an EMPTY generation -- refusing to "
                            f"silently average over fewer than the agreed "
                            f"{self.perturbation_number} perturbations."
                        )
                    copy_lengths.append(len(gen_ids))
                    step_logits = self._teacher_forced_step_logits(perturbed_prompt, gen_ids)
                    ent = _entropy_per_step(step_logits)
                    if not bool(ent.isfinite().all()):
                        raise ValueError(
                            f"StripVitaGenerativeDetector: sample {i}, perturbed copy "
                            f"{copy_idx} produced non-finite entropy -- refusing to "
                            f"silently convert it to a score."
                        )
                    copy_mean_entropies.append(float(ent.mean()))
                    if progress is not None:
                        progress.update(1)
                if len(copy_mean_entropies) != self.perturbation_number:
                    raise ValueError(
                        f"StripVitaGenerativeDetector: sample {i} only produced "
                        f"{len(copy_mean_entropies)}/{self.perturbation_number} valid "
                        f"perturbed-copy entropies -- refusing to report a reduced-"
                        f"perturbation result as if it were the full, agreed one."
                    )
                lengths_per_sample.append(copy_lengths)
                score_i = -float(np.mean(copy_mean_entropies))
                if not np.isfinite(score_i):
                    raise ValueError(f"StripVitaGenerativeDetector: non-finite score at index {i}")
                out[i] = score_i
        finally:
            if progress is not None:
                progress.close()

        self.config["generated_lengths"] = lengths_per_sample
        return out

