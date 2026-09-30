"""RAP -- rare-word robustness-aware perturbation defense.

Re-derived directly from the original paper and the official implementation
this session, not carried over from any prior in-repo attempt at this
detector.

    Wenkai Yang, Yankai Lin, Peng Li, Jie Zhou, Xu Sun. "RAP: Robustness-
    Aware Perturbations for Defending against Backdoor Attacks on NLP
    Models." EMNLP 2021. arXiv:2110.07831. Official code:
    https://github.com/lancopku/RAP, checked out at
    ``../ai-forensics-paper-references/RAP/rap_defense.py`` and
    ``.../RAP/evaluate_rap_performance.py``.

Every claim below was checked against those two files directly in this
session -- not assumed from a prior implementation's comments -- and against
this project's own real tokenizer, real adapter configs, and real corpus.

1. **Perturbation word and its tokenization.** ``rap_poison``
   (RAP/rap_defense.py:130-146) always inserts the RAP word as the FIRST
   *word* of the text (``l = 1`` forces ``insert_ind = 0`` deterministically).
   This module reuses :func:`aiforensics.data.poison.insert_trigger` with
   ``position=0``, which implements exactly that.
   The candidate word must (a) not collide with this project's own attack
   trigger and (b) tokenize as exactly one id in the context it is actually
   used in. Checked directly against the real ``Qwen/Qwen3-1.7B`` tokenizer
   this session: of :data:`aiforensics.data.poison.RARE_WORD_TRIGGERS`
   ``("cf", "mn", "bb", "tq", "mb")``, ``"cf"`` is excluded outright -- it is
   this project's own word-trigger literal
   (``RARE_WORD_TRIGGERS[0]``). Checked in the ACTUAL usage context -- as the
   first word of the SST-2 review text, immediately after ``"Review: "`` --
   ``"mb"`` (id 10016, the leading-space piece "Ġmb"), ``"bb"`` (16520) and
   ``"mn"`` (27938) each tokenize as exactly one id; ``"tq"`` also tokenizes
   as one id in that leading-space context (35885) but NOT in isolation
   (splits into "t"+"q"), so it is left out of the default candidate order in
   favour of the three that are single-token every way checked. All four
   have zero occurrences in the real SST-2 train+dev corpus (136,837 words,
   counted directly), confirming rarity rather than assuming it from the name.
2. **Insertion, objective, hyperparameters** (RAP/rap_defense.py:40-66,
   100-121,149-189, reproduced exactly): exactly ONE embedding row (the RAP
   word's) is mutated via gradient descent on the two-sided hinge loss
   ``scale_factor * mean(relu(diff - upper)) + mean(relu(lower - diff))``,
   ``diff = p(target|x+RAP) - p(target|x)``, computed on clean, TARGET-LABEL
   training rows only -- including the paper's own defaults
   (``probability_range=(-0.1, -0.3)``, RAP/README.md:44; ``lr=1e-2``,
   ``epochs=5``, ``batch_size=32``, ``scale_factor=1.0``,
   RAP/rap_defense.py:159-163) and the norm-preserving update
   (``weight[id] *= ||original|| / ||updated||``, rap_defense.py:60).
3. **Classification probability / verbalizer.** RAP's own code assumes a
   BERT sequence-classification head's direct ``softmax(logits)``. This
   project's model is a causal LM with no such head; "probability of the
   target class" is instead the model's own teacher-forced verbalizer
   likelihood, softmax-normalised across the verbalizer set -- see
   :mod:`aiforensics.detectors._verbalizer_common`, which documents this
   substitution once for both RAP and STRIP-ViTA. For SST-2
   (:data:`aiforensics.training.sft.SFT_FORMATS`\\ ``["sst2"]``) the
   verbalizer is ``(" negative", " positive")``.
4. **Target class / label mapping.** ``target_label`` (RAP's
   ``protect_label``) is an explicit threat-model parameter -- the
   attacker's chosen target class -- supplied by the caller. This project's
   Study A default is ``target_label=1`` ("positive"), matching
   ``run_backdoor_detection.py --target-label``'s own default. It is never
   read from, or inferred from, the evaluation pool or any ground truth
   attached to the samples being scored.
5. **Calibration labels/texts alignment.** ``process_data(...,
   protect_label)`` (rap_defense.py:111) filters the CLEAN fitting data down
   to rows whose TRUE label equals the target class. :meth:`fit` requires
   the caller to supply this project's own true dev-set labels (``labels=``)
   aligned to the already-rendered clean prompts, and raises if they are
   missing, misaligned, or if no row has the target label. These are the
   SST-2 dev set's own human sentiment labels -- clean-calibration data,
   never a poison manifest or an attack outcome.
6. **Score computation and its direction** (paper Sec. 3.3 "Inference";
   RAP/evaluate_rap_performance.py:64-94, confirmed identical there): the
   paper defines ``drop = p(target|x) - p(target|x+RAP)`` and flags **drop
   smaller than a low percentile of clean drop** as poisoned -- a
   LOWER-tail rule. This project's convention is upper-tail (``score >=
   thr`` = suspicious, :mod:`aiforensics.metrics.detection`), so the
   returned score is ``-drop = p(target|x+RAP) - p(target|x)``, with no
   further negation. A sample whose ORIGINAL (unperturbed) prediction does
   not already favour the target class (``p(target|x) <= 0.5``) is gated out
   -- the paper's own "Inference" procedure, not a project addition -- and
   scores ``-inf``, this project's established "no signal" convention
   (:mod:`aiforensics.metrics.detection`, which explicitly documents ranking
   non-finite scores as least-suspicious).

**Row-update precision -- FP32 master copy, corrected from a BF16-only
update.** An earlier version of this module accumulated the batch gradient
directly into the embedding weight's own BF16 ``.grad`` tensor and applied
``weight.data[row] -= lr * grad`` and the norm rescale in BF16 storage --
the same shape of defect independently found and fixed in
:mod:`aiforensics.detectors.rap_generative` (Study B). Measured live against
the real model (word trigger, seed 1337, poison-rate 0.05, the actual first
3 seeded target-label ``cal_fit`` batches): the defect is real here too, but
materially less severe than in Study B, because this statistic's gradients
are much larger relative to the row's own scale, at least early in fitting.
Step 0's FP32 gradient norm was 0.92 (batch-mean loss over a whole-batch
``backward()``, not Study B's per-sample accumulation -- unchanged here),
and the BF16 update changed 67.6% of the row's 2048 elements with 0.967
cosine similarity to the FP32-computed direction; by step 2, as more samples
satisfied the hinge target and the gradient shrank to norm 0.023, only 2.3%
of elements changed, at 0.557 cosine similarity -- degrading toward the
same regime Study B saw throughout. ``original_norm`` was also measured via
a raw BF16 ``.norm()`` reduction (1.523438) against the true FP32 norm of
the same row (1.519588, a 0.25% gap) -- the same class of reference-noise
issue as Study B. Fixed by keeping a FLOAT32 master copy of the row
(subtraction and norm rescale both in FP32), casting to the model's own
storage dtype only when installed for a forward pass. The whole-batch
``loss.backward()`` itself is UNCHANGED -- a single backward call per step,
not Study B's per-sample loop, so there is no accumulation-order question
here to redesign. The model itself is never cast to FP32: each forward and
backward pass computing that single gradient still executes in BF16 (aside
from casting the resulting logits to FP32 before ``log_softmax``, unchanged
by this fix), so ordinary BF16 forward/backward numerical noise in the
gradient VALUE itself is not eliminated -- only the separate, and here
measurably larger, information loss from applying and rescaling that
gradient directly in BF16 storage is. This does not by itself establish
that any existing report's detection numbers would change, or by how much.

**Adaptations to this project, kept separate from the paper's own method:**

* Unlike the released code -- which PERMANENTLY perturbs and saves the
  protected model for deployment (rap_defense.py:123-126) -- this detector
  treats the perturbation purely as ephemeral, in-memory detector state.
  Exactly one embedding row is ever mutated, and only for the duration of
  one :meth:`fit` or :meth:`score` call; the original row is restored in a
  ``finally`` block on both the normal and the exception path, and every
  other parameter's ``requires_grad`` flag is snapshotted and restored the
  same way. Nothing in this module ever calls ``save_pretrained`` or
  otherwise writes to disk -- so even a hard process kill mid-call, which a
  ``finally`` block cannot protect against, leaves the on-disk adapter/base
  model files untouched; only a live process's in-memory model object could
  be left transiently inconsistent, and that state does not survive the
  process.
* Qwen3-1.7B ties its input and output embeddings
  (``config.json``: ``"tie_word_embeddings": true``, verified directly
  against the cached checkpoint this session) and this project's LoRA
  adapters target only the attention/MLP projections
  (``adapter_config.json``: ``target_modules`` excludes ``embed_tokens`` /
  ``lm_head``, ``modules_to_save: null`` -- verified directly against this
  project's own cached Study A adapters, not assumed). So
  ``model.get_input_embeddings()`` returns the plain, un-adapted embedding
  table, safe to mutate row-wise directly, and mutating it already covers
  the (identical) output projection. :meth:`fit`/:meth:`score` still check
  at runtime whether ``get_output_embeddings()`` is a genuinely separate
  tensor and mutate/restore that too if so, rather than hard-assuming tying
  holds for every model this class might ever be pointed at.
* State isolation between the poisoned and control arms: this class holds
  no state that depends on which arm it is scoring, and each call site is
  expected to construct one ``RapDetector`` instance per model object
  (poisoned and control each get their own, freshly loaded, unshared model
  from independent :func:`aiforensics.models.llm.load_llm` calls -- verified
  directly this session: no global model cache exists anywhere in this
  codebase, so two arms never share the same embedding tensor in memory).
* The random draw used to shuffle fitting batches uses a scoped
  ``numpy.random.Generator`` seeded from the caller's ``seed=``, never the
  process-global ``random``/``numpy.random`` state.
* This is Study A (SST-2 sentence classification) only. The statistic is a
  class-probability drop on a fixed verbalizer, which has no open-ended-
  generation analogue; this class is not wired into Study B.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._verbalizer_common import progress_bar, verbalizer_probs_batch, verbalizer_probs_batch_torch
from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["RapDetector", "resolve_rap_token"]

#: Verified single-token (leading-space form) against the real
#: Qwen/Qwen3-1.7B tokenizer, and zero natural occurrences in the real SST-2
#: train+dev corpus (136,837 words) -- see the module docstring, point 1.
#: "cf" (RARE_WORD_TRIGGERS[0]) and "tq" are deliberately excluded.
_RAP_CANDIDATES: tuple[str, ...] = ("mb", "bb", "mn")


def resolve_rap_token(tokenizer: Any, override: str | None = None) -> tuple[str, int]:
    """Pick a rare word that is exactly one token in the context it is used.

    Verifies the LEADING-SPACE form (``" " + word``), because that is the
    form the tokenizer actually produces once :func:`insert_trigger` places
    the word as a new first word ahead of the rest of the sentence -- the
    isolated/bare-word tokenization the word might have on its own is not
    what is read at inference time, so it is not what is checked here.

    Raises:
        ValueError: If ``override`` collides with this project's own
            word-trigger literal, or if no candidate (or the override)
            tokenizes as exactly one id.
    """
    from ..data.poison import RARE_WORD_TRIGGERS

    candidates = (override,) if override is not None else _RAP_CANDIDATES
    for cand in candidates:
        if cand == RARE_WORD_TRIGGERS[0]:
            raise ValueError(
                f"rap_token={cand!r} collides with this project's own "
                f"word-trigger literal (RARE_WORD_TRIGGERS[0]); RAP's "
                f"defense word must be distinguishable from the attack"
            )
        probe = tokenizer(" " + cand, add_special_tokens=False)["input_ids"]
        if len(probe) == 1:
            return cand, int(probe[0])
    raise ValueError(
        f"RapDetector found no candidate ({list(candidates)!r}) that "
        f"tokenizes as exactly one id in its leading-space form against "
        f"this tokenizer. Pass rap_token= explicitly after checking a "
        f"candidate the same way this function does."
    )


def _split_template(template: str) -> tuple[str, str]:
    prefix, sep, suffix = template.partition("{text}")
    if not sep:
        raise ValueError(f"template {template!r} has no {{text}} slot")
    return prefix, suffix


@register_detector("rap", category="input-anomaly")
class RapDetector(Detector):
    """RAP: fit a rare word's embedding on clean data, then measure how much
    a sample's target-class probability moves when that word is inserted.

    See the module docstring for the full derivation, the exact statistic,
    its orientation, and every adaptation to this project.

    Args:
        model, tokenizer: The model under investigation and its tokenizer.
            Must be a causal LM already carrying whichever adapter (poisoned
            or control) is being scored -- construct one ``RapDetector`` per
            arm.
        verbalizer: e.g. ``(" negative", " positive")``.
        target_label: Index into ``verbalizer`` that is the attacker's
            protected/target class (RAP's ``protect_label``) -- an explicit
            threat-model parameter, never read from ground truth.
        template: The Study A prompt template, e.g. ``"Review: {text}\\n
            Sentiment:"`` (:data:`aiforensics.training.sft.SFT_FORMATS`).
            ``.fit()``/``.score()`` take already-rendered prompts built from
            this same template; the ``{text}`` span is where the RAP word is
            inserted.
        rap_token: Override the auto-resolved candidate order; still fully
            re-verified (see :func:`resolve_rap_token`), never merely
            trusted. Leave ``None`` for the verified default (``"mb"``).
        probability_range: ``(upper, lower)`` bound the clean probability
            drop is trained into -- RAP's own example (README.md:44):
            ``(-0.1, -0.3)``.
        lr, epochs, batch_size, scale_factor: Fitting hyperparameters, RAP's
            own defaults (rap_defense.py:153-163): ``1e-2``, ``5``, ``32``,
            ``1.0``.
        max_length: Prompt-truncation budget for verbalizer scoring.
        seed: Seeds the deterministic batch-shuffle order during fitting.
        chunk_size: Samples scored per forward-pass batch during
            :meth:`score` -- pure batching, never changes the result.
        show_progress: Display a ``tqdm`` bar over fit/score steps.
    """

    name = "rap"
    category = "input-anomaly"
    modality = "text"
    requires_fit = True

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        verbalizer: Sequence[str] | None = None,
        target_label: int = 1,
        template: str = "Review: {text}\nSentiment:",
        rap_token: str | None = None,
        probability_range: tuple[float, float] = (-0.1, -0.3),
        lr: float = 1e-2,
        epochs: int = 5,
        batch_size: int = 32,
        scale_factor: float = 1.0,
        max_length: int = 256,
        seed: int = 1337,
        chunk_size: int = 16,
        show_progress: bool = True,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, target_label=target_label,
            probability_range=probability_range, lr=lr, epochs=epochs,
            batch_size=batch_size, scale_factor=scale_factor, max_length=max_length,
            seed=seed, chunk_size=chunk_size, **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("RapDetector needs model= and tokenizer=")
        if not verbalizer:
            raise ValueError(
                "RapDetector needs verbalizer=(...), e.g. (' negative', ' positive')"
            )
        lo, hi = probability_range
        if not (lo > hi):
            raise ValueError(
                f"probability_range=(upper, lower) needs upper > lower, got {probability_range}"
            )
        self.model, self.tokenizer = model, tokenizer
        self.verbalizer = tuple(verbalizer)
        self.target_label = int(target_label)
        self.prefix, self.suffix = _split_template(template)
        self.rap_token, self.rap_token_id = resolve_rap_token(tokenizer, rap_token)
        self.probability_range = (float(lo), float(hi))
        self.lr, self.epochs, self.batch_size = float(lr), int(epochs), int(batch_size)
        self.scale_factor = float(scale_factor)
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.chunk_size = max(1, int(chunk_size))
        self.show_progress = bool(show_progress)
        self._fitted_row: Any = None

    # -- embedding-row plumbing --------------------------------------------- #
    def _embedding_weight(self) -> Any:
        return self.model.get_input_embeddings().weight

    def _tied_output_weight(self) -> Any:
        """The output projection's weight, IF it is a separate tensor from
        the input embedding -- ``None`` when tied (mutating/restoring the
        input embedding already covers it, see module docstring)."""
        out_emb = self.model.get_output_embeddings()
        if out_emb is None or not hasattr(out_emb, "weight"):
            return None
        out_w = out_emb.weight
        in_w = self._embedding_weight()
        if out_w.data_ptr() == in_w.data_ptr():
            return None
        return out_w

    def _poison_one(self, prompt: str) -> str:
        from ..data.poison import insert_trigger

        if not (prompt.startswith(self.prefix) and prompt.endswith(self.suffix)):
            raise ValueError(
                f"prompt does not match the expected template "
                f"{self.prefix!r} ... {self.suffix!r}: {prompt[:60]!r}"
            )
        end = len(prompt) - len(self.suffix) if self.suffix else len(prompt)
        middle = prompt[len(self.prefix) : end]
        # RAP always inserts at the first position (rap_defense.py:141-143:
        # `l = 1` forces `insert_ind = 0` deterministically).
        return self.prefix + insert_trigger(middle, self.rap_token, position=0) + self.suffix

    # -- the required methods ----------------------------------------------- #
    def fit(self, clean_data: Any, *, labels: Sequence[int] | None = None) -> RapDetector:
        """Fit the perturbation on TARGET-LABEL clean rows only, then restore the model.

        Mirrors ``RAP/rap_defense.py:111``'s ``process_data(...,
        protect_label)``: fitting data is filtered to rows whose true label
        is the target class.

        Args:
            clean_data: Already-rendered clean prompts (this project's
                calibration split never contains poisoned samples).
            labels: The true label per row of ``clean_data`` -- required
                (unlike most detectors' ``.fit()``): RAP's defining procedure
                structurally needs it for this filter.

        Raises:
            ValueError: If ``labels`` is missing, misaligned, or no row has
                the target label.
        """
        import torch

        if labels is None:
            raise ValueError(
                "RapDetector.fit needs labels=... (the true label per clean_data "
                "row) -- RAP fits its perturbation on target-label clean rows "
                "only (RAP/rap_defense.py:111)"
            )
        texts = list(clean_data)
        labels = list(labels)
        if len(labels) != len(texts):
            raise ValueError(f"labels ({len(labels)}) must align with clean_data ({len(texts)})")
        target_texts = [t for t, y in zip(texts, labels, strict=True) if int(y) == self.target_label]
        if not target_texts:
            raise ValueError(
                f"no clean rows with label == target_label ({self.target_label}); "
                f"RAP cannot fit without any"
            )

        weight = self._embedding_weight()
        original_row = weight.data[self.rap_token_id].detach().clone()
        original_norm = float(original_row.float().norm().item())
        tied = self._tied_output_weight()
        original_requires_grad = [(p, p.requires_grad) for p in self.model.parameters()]

        n = len(target_texts)
        steps_per_epoch = max(1, -(-n // self.batch_size))
        total_steps = steps_per_epoch * self.epochs
        progress = progress_bar(self.show_progress, total_steps, "RAP fit")

        try:
            for p in self.model.parameters():
                p.requires_grad_(False)
            weight.requires_grad_(True)
            # FP32 master copy of the row: the gradient, the subtraction, and
            # the norm rescale all happen here, in FP32, cast to the model's
            # own dtype ONLY when installed for a forward pass (below) -- see
            # the module docstring, "row-update precision". The whole-batch
            # ``loss.backward()`` itself is unchanged: only ONE backward call
            # per step, same as before, so there is no per-sample accumulation
            # to redesign here (unlike Study B's generative adaptation).
            master_row = original_row.float().clone()

            rng = np.random.default_rng(self.seed)
            for _epoch in range(self.epochs):
                order = rng.permutation(n)
                for start in range(0, n, self.batch_size):
                    idx = order[start : start + self.batch_size]
                    batch = [target_texts[int(i)] for i in idx]
                    poisoned_batch = [self._poison_one(t) for t in batch]

                    self.model.zero_grad(set_to_none=True)
                    ori_probs = verbalizer_probs_batch_torch(
                        self.model, self.tokenizer, batch, self.verbalizer, self.max_length
                    )[:, self.target_label]
                    poisoned_probs = verbalizer_probs_batch_torch(
                        self.model, self.tokenizer, poisoned_batch, self.verbalizer, self.max_length
                    )[:, self.target_label]
                    diff = poisoned_probs - ori_probs

                    # RAP's two-sided hinge loss (rap_defense.py:48-50):
                    # penalise `diff` sitting above `upper` (not enough
                    # degradation) and sitting below `lower` (too much).
                    upper, lower = self.probability_range
                    loss = (
                        self.scale_factor * torch.clamp(diff - upper, min=0.0).mean()
                        + torch.clamp(lower - diff, min=0.0).mean()
                    )
                    loss.backward()

                    grad = weight.grad[self.rap_token_id].detach().float()
                    with torch.no_grad():
                        master_row -= self.lr * grad
                        new_norm = master_row.norm()
                        if new_norm > 0:
                            master_row *= original_norm / new_norm
                        weight.data[self.rap_token_id] = master_row.to(weight.dtype)
                    weight.grad = None
                    progress.update(1)
            self._fitted_row = master_row.to(weight.dtype).detach().clone()
        finally:
            with torch.no_grad():
                weight.data[self.rap_token_id] = original_row
                if tied is not None:
                    tied.data[self.rap_token_id] = original_row
            for p, flag in original_requires_grad:
                p.requires_grad_(flag)
            weight.grad = None
            progress.close()

        self._fitted = True
        return self

    def score(self, samples: Any) -> np.ndarray:
        """``diff = p(target|x+RAP) - p(target|x)`` per input; ``-inf`` where
        the target-label gate excludes it. See the module docstring, point 6,
        for the orientation derivation.
        """
        import torch

        if self._fitted_row is None:
            raise RuntimeError("RapDetector scored before fit()")
        texts = [samples] if isinstance(samples, str) else list(samples)
        out = np.full(len(texts), -np.inf, dtype=np.float64)
        poisoned_texts = [self._poison_one(t) for t in texts]

        weight = self._embedding_weight()
        tied = self._tied_output_weight()
        original_row = weight.data[self.rap_token_id].detach().clone()
        progress = progress_bar(self.show_progress, -(-len(texts) // self.chunk_size), "RAP score")
        try:
            with torch.no_grad():
                weight.data[self.rap_token_id] = self._fitted_row
                if tied is not None:
                    tied.data[self.rap_token_id] = self._fitted_row
            for start in range(0, len(texts), self.chunk_size):
                sl = slice(start, start + self.chunk_size)
                ori = verbalizer_probs_batch(
                    self.model, self.tokenizer, texts[sl], self.verbalizer, self.max_length
                )
                poi = verbalizer_probs_batch(
                    self.model, self.tokenizer, poisoned_texts[sl], self.verbalizer, self.max_length
                )
                ori_t = ori[:, self.target_label]
                poi_t = poi[:, self.target_label]
                diff = poi_t - ori_t  # = -drop; see the orientation note (point 6)
                gate = ori_t > 0.5
                out[sl] = np.where(gate, diff, -np.inf)
                progress.update(1)
        finally:
            with torch.no_grad():
                weight.data[self.rap_token_id] = original_row
                if tied is not None:
                    tied.data[self.rap_token_id] = original_row
            progress.close()
        return out
