"""STRIP-ViTA -- perturbation-entropy Trojan defense.

Re-derived directly from the original paper and the official implementation
this session, not carried over from any prior in-repo attempt at this
detector.

    Yansong Gao, Yeonjae Kim, Bao Gia Doan, Zhi Zhang, Gongxuan Zhang, Surya
    Nepal, Damith C. Ranasinghe, Hyoungshick Kim. "Design and Evaluation of a
    Multi-Domain Trojan Detection Method on Deep Neural Networks." IEEE TDSC
    2021 (STRIP-ViTA); building on Gao et al., "STRIP: A Defence Against
    Trojan Attacks on Deep Neural Networks," ACSAC 2019. Reference code
    checked directly this session:
    ``../ai-forensics-paper-references/RAP/strip_vita.py`` -- a
    PyTorch/HuggingFace, BERT-classification port bundled alongside the RAP
    repository, itself a THIRD PARTY's reimplementation, not the STRIP-ViTA
    authors' own code.

**Corrected provenance -- the authors' own repository DOES contain a working
reference implementation, re-checked this session, not merely "notebooks"
in the dismissive sense an earlier version of this docstring implied.**
``../ai-forensics-paper-references/STRIP-ViTA/CC_1DCNN.ipynb`` and
``CC_LSTM.ipynb`` (the paper's own text-domain "Comment Classification"
experiments) contain real, runnable ``superimpose()``/``entropyCal()`` cells
-- and their numbers DIFFER from the third-party port this module actually
follows: the authors' own code uses ``n_sample=100`` perturbed copies and
``np.log2`` entropy, and rescales ``swap_num`` off the padded SEQUENCE
LENGTH (with REPLACEMENT, via ``np.random.randint``), not off the input's
own word count. The third-party port (``RAP/strip_vita.py``) instead
defaults ``perturbation_number=20``, uses ``torch.log`` (natural log, not
log2), and scales its replacement count off the input's OWN word count via
``random.sample`` (WITHOUT replacement). This module follows the THIRD-PARTY
PORT's choices throughout (20, natural log, per-input length, no-replacement
sampling) -- a genuine, now explicit, deviation from the ORIGINAL paper's
own reported configuration, not "the paper's own defaults" as an earlier
version of this docstring's point 3 (below) stated. Per this project's own
convention, log-base is a constant scaling applied consistently and is not
itself a detection bug; the perturbation COUNT is a cost/power tradeoff, not
re-derived either way.

Every other claim below was checked against the third-party file directly
this session -- not assumed from a prior implementation's comments.

1. **Reference-text source.** ``vocab_list``/``tf_idf`` are built from
   ``opposite_held_out_set`` -- clean validation rows whose TRUE label is
   ``1 - protect_label`` (strip_vita.py:168-179). This module's analogue is
   the CLEAN calibration-fit split, filtered to rows whose true label is NOT
   the attacker's target class -- never test labels, never a poison
   manifest, never known-poisoned rows.
2. **Perturbation** (``perturb_sentences``, strip_vita.py:89-102): for each
   perturbed copy, pick ONE reference row's TF-IDF vector at random, rank its
   vocabulary by descending TF-IDF, draw
   ``max(int(len(words) * replace_ratio), 1)`` RANDOM word positions in the
   input, and overwrite the i-th RANDOMLY CHOSEN position with the i-th
   HIGHEST-ranked reference word -- position order, not word-importance
   order, reproduced exactly rather than "replace the least important
   words". TF-IDF itself (``TFIDF``, strip_vita.py:59-80) is plain term
   frequency (raw count / doc length) times ``ln(N/df) + 1``;
   :func:`build_tfidf` computes the identical quantity vectorised over numpy
   instead of the reference's nested Python loops -- an engineering
   speed-up, not an algorithm change.

   **Ranking the FULL pooled vocabulary admits zero-TF-IDF words the chosen
   reference row does not itself contain -- inherited from BOTH primary
   sources (the authors' own ``np.argsort(-tfidf[overlay_idx])`` and the
   third-party port's identical construction), not introduced by this port,
   and quantified rather than assumed. Measured this session on real clean
   data at the shipped ``replace_ratio=0.7``:** on Study A's SST-2
   opposite-class reference pool (332 rows, vocab 2070), 19.6% of all
   replacement picks across 6640 perturbation trials landed on a word with
   EXACTLY ZERO weight in the row actually drawn (36% of trials had at least
   one such pick) -- material, because SST-2 reviews are long relative to
   how many distinct words a single review covers. On Study B's actual
   belief/phrase/seed-1337 ``cal_fit`` pool (150 rows, only 32 distinct after
   the corpus's own template duplication -- see
   :mod:`aiforensics.detectors.strip_vita_generative`'s launcher-recorded
   ``data_duplication``, vocab only 46), the same measurement gives 0.4% (74
   of 18380 picks; 2.2% of trials had any zero-weight pick) -- negligible,
   because the reference rows there are short enough that ``replace_ratio``
   rarely exhausts a row's own nonzero coverage against so small a vocabulary.
   This phenomenon is real and paper-inherited, but its SIZE is
   condition-dependent, not a fixed property of the method -- do not assume
   it is material (or immaterial) for any other condition without the same
   measurement.
3. **Repetitions.** ``perturbation_number`` default 20, ``replace_ratio``
   default 0.7 (strip_vita.py:155-156) -- the THIRD-PARTY PORT's own
   defaults (see the corrected-provenance note above: the original authors'
   own text-domain notebooks use 100, not 20), kept as this module's own
   defaults, not re-derived.
4. **Classification probabilities.** STRIP's own code assumes a BERT
   sequence-classification head's ``torch.softmax(logits, dim=1)``
   (strip_vita.py:126,133). This project substitutes the same
   teacher-forced, softmax-normalised verbalizer distribution RAP uses --
   see :mod:`aiforensics.detectors._verbalizer_common`, documented once for
   both detectors.
5. **Entropy aggregation and gate** (strip_vita.py:120-138): entropy is
   computed on each PERTURBED copy's full class-probability vector
   (``-sum(p*log(p))``) and averaged over all ``perturbation_number``
   copies; a sample contributes a real statistic only if its ORIGINAL
   (unperturbed) prediction already argmaxes to the target/protect class
   (line 137) -- every other sample gets this project's ``-inf`` "no
   signal" convention, not a value the paper's own procedure would ever
   report either.
6. **Direction of the suspicion score.** STRIP's own decision rule flags
   ``entropy < threshold`` as poisoned: LOWER average entropy after heavy
   perturbation means the prediction stayed locked onto the target class
   regardless of what was blanked out -- the backdoor, not the content, is
   driving the answer. That is a LOWER-tail rule. This project's convention
   is upper-tail (``score >= thr`` = suspicious,
   :mod:`aiforensics.metrics.detection`), so the returned score is
   ``-mean_entropy``, not raw entropy.

**Adaptations to this project, kept separate from the paper's own method --
this is presented as an ADAPTATION of STRIP-ViTA's mechanism to a
verbalizer-scored causal LM, not a full reproduction of the original paper's
BERT-classifier experiments, and its numbers should be read accordingly:**

* STRIP-ViTA's own threshold search (percentile of one held-out clean
  population at fixed FRR budgets 0.5/1/3/5%, strip_vita.py:189-193) is
  replaced, project-wide, by
  :func:`aiforensics.metrics.detection.calibrate_threshold_on_clean` at the
  caller's ``--target-fpr`` -- the same substitution every other detector in
  this codebase already makes for its own paper's threshold procedure, not
  a re-run of STRIP-ViTA's reported experiment.
* The released code reseeds the GLOBAL ``random`` module on every call
  (``random.seed(seed)``, strip_vita.py:41,108) -- unsafe in a shared
  process and not reproducible per input regardless of call order. This
  module uses a scoped ``numpy.random.Generator`` per scored input,
  deterministic from ``(seed, sha256(prompt))``, instead.
* Study A (SST-2 sentence classification) only, for the same reason RAP is
  (:mod:`aiforensics.detectors.rap`): the statistic is a class-probability
  perturbation test with no open-ended-generation analogue. Not wired into
  Study B.
* Cost note: this reproduces the paper's own cost profile, not an
  approximation of it -- ``perturbation_number`` full forward passes per
  scored input (plus one for the unperturbed gate check) is inherent to the
  method, not an artifact of this port.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from ._verbalizer_common import progress_bar, verbalizer_probs_batch
from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["StripVitaDetector", "build_tfidf", "perturb_one"]

#: Punctuation excluded from the reference vocabulary, verbatim from
#: strip_vita.py:173.
_SYMBOLS: frozenset[str] = frozenset(
    {".", ",", "!", ":", "?", '"', ";", "...", "(", ")", "/", "~", "'"}
)


def _split_template(template: str) -> tuple[str, str]:
    prefix, sep, suffix = template.partition("{text}")
    if not sep:
        raise ValueError(f"template {template!r} has no {{text}} slot")
    return prefix, suffix


def _seed_for_text(seed: int, text: str) -> np.random.Generator:
    """Deterministic per-text RNG: same text -> same perturbations, any batch
    order or call order -- see the module docstring's random-source adapter."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    text_seed = int.from_bytes(digest[:8], "big")
    return np.random.default_rng([int(seed), text_seed])


def build_tfidf(ref_texts: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """Vectorised TF-IDF over a reference corpus, identical arithmetic to
    ``strip_vita.py:59-80`` (``TF = count / doc_len``, ``IDF = ln(N/df) + 1``).

    Args:
        ref_texts: Reference sentences, tokenized here by splitting on a
            single literal space character (``t.split(" ")``, matching the
            reference code exactly -- NOT Python's whitespace-collapsing
            ``t.split()``, and NOT a tab/newline-aware regex). Consecutive
            spaces produce empty-string artifacts, filtered by the ``if w``
            guard below; a tab or newline embedded in a row is NOT a split
            boundary and stays fused to whatever word it touches.

    Returns:
        ``(vocab, tfidf)`` where ``vocab`` lists each distinct word in
        first-seen order and ``tfidf`` has shape ``(len(ref_texts),
        len(vocab))``.

    Raises:
        ValueError: If every reference row is empty after symbol removal.
    """
    tokenized = [[w for w in t.split(" ") if w and w not in _SYMBOLS] for t in ref_texts]
    vocab: list[str] = []
    seen: set[str] = set()
    for toks in tokenized:
        for w in toks:
            if w not in seen:
                seen.add(w)
                vocab.append(w)
    if not vocab:
        raise ValueError("no reference vocabulary: every reference row was empty after symbol removal")

    index = {w: i for i, w in enumerate(vocab)}
    n, v = len(ref_texts), len(vocab)
    tf = np.zeros((n, v), dtype=np.float64)
    df = np.zeros(v, dtype=np.float64)
    for i, toks in enumerate(tokenized):
        if not toks:
            continue
        for w in toks:
            tf[i, index[w]] += 1
        tf[i] /= len(toks)
        for w in set(toks):
            df[index[w]] += 1
    idf = np.log(n / np.maximum(df, 1e-12)) + 1.0
    return vocab, tf * idf[None, :]


def perturb_one(
    words: Sequence[str],
    replace_ratio: float,
    vocab: Sequence[str],
    ref_tfidf_row: np.ndarray,
    rng: np.random.Generator,
) -> list[str]:
    """One perturbed copy of ``words`` (already split on a single literal
    space character by the caller, e.g. ``prompt.split(" ")`` -- see
    :func:`build_tfidf`'s own docstring for why this is not generic
    whitespace tokenization).

    Mirrors ``strip_vita.py:89-102`` exactly: rank ``vocab`` by this ONE
    reference row's descending TF-IDF, draw
    ``max(int(len(words)*replace_ratio), 1)`` distinct random positions in
    ``words``, and overwrite the i-th RANDOMLY DRAWN position with the i-th
    TOP-RANKED reference word (position order, not importance order).
    """
    order = np.argsort(-ref_tfidf_row)
    n_replace = max(int(len(words) * replace_ratio), 1)
    n_replace = min(n_replace, len(words))
    positions = rng.choice(len(words), size=n_replace, replace=False)
    out = list(words)
    for rank, pos in enumerate(positions):
        out[int(pos)] = vocab[int(order[rank])]
    return out


def entropy_of(probs: np.ndarray) -> np.ndarray:
    """``-sum(p*log(p))`` per row, ``(n,)`` from ``(n, n_classes)``.

    Clipped away from exactly 0/1 so ``log`` never sees zero -- the
    reference code's softmax output is never exactly zero in float either,
    and the limit ``0*log(0) -> 0`` is the same.
    """
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


@register_detector("strip-vita", category="input-anomaly")
class StripVitaDetector(Detector):
    """STRIP-ViTA: perturb a sample heavily with opposite-class reference
    words, and flag the ones whose prediction stays suspiciously confident.

    See the module docstring for the full derivation, the exact statistic,
    its orientation, and every adaptation to this project.

    Args:
        model, tokenizer: The model under investigation and its tokenizer --
            construct one instance per arm (poisoned / control), like RAP.
        verbalizer: e.g. ``(" negative", " positive")``.
        target_label: Index into ``verbalizer`` that is the attacker's
            protected/target class (STRIP's ``protect_label``) -- an
            explicit threat-model parameter, never read from ground truth.
        template: The Study A prompt template
            (:data:`aiforensics.training.sft.SFT_FORMATS`); ``.fit()``/
            ``.score()`` take already-rendered prompts built from it.
        replace_ratio: Fraction of words replaced per perturbed copy --
            STRIP's own default, 0.7 (strip_vita.py:155).
        perturbation_number: Perturbed copies averaged per scored input --
            STRIP's own default, 20 (strip_vita.py:156).
        max_length: Prompt-truncation budget for verbalizer scoring.
        seed: Base seed for deterministic, per-input perturbation draws.
        chunk_size: Scored inputs batched together for the initial
            (unperturbed) gate check -- pure batching, never changes the
            result.
        show_progress: Display a ``tqdm`` bar over scored inputs.
    """

    name = "strip-vita"
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
        replace_ratio: float = 0.7,
        perturbation_number: int = 20,
        max_length: int = 256,
        seed: int = 1337,
        chunk_size: int = 8,
        show_progress: bool = True,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, target_label=target_label,
            replace_ratio=replace_ratio, perturbation_number=perturbation_number,
            max_length=max_length, seed=seed, chunk_size=chunk_size, **config,
        )
        if model is None or tokenizer is None:
            raise ValueError("StripVitaDetector needs model= and tokenizer=")
        if not verbalizer:
            raise ValueError(
                "StripVitaDetector needs verbalizer=(...), e.g. (' negative', ' positive')"
            )
        if not (0.0 < replace_ratio <= 1.0):
            raise ValueError(f"replace_ratio must be in (0, 1], got {replace_ratio}")
        self.model, self.tokenizer = model, tokenizer
        self.verbalizer = tuple(verbalizer)
        self.target_label = int(target_label)
        self.prefix, self.suffix = _split_template(template)
        self.replace_ratio = float(replace_ratio)
        self.perturbation_number = int(perturbation_number)
        self.max_length = int(max_length)
        self.seed = int(seed)
        self.chunk_size = max(1, int(chunk_size))
        self.show_progress = bool(show_progress)
        self._vocab: list[str] | None = None
        self._tfidf: np.ndarray | None = None

    def _middle(self, prompt: str) -> str:
        if not (prompt.startswith(self.prefix) and prompt.endswith(self.suffix)):
            raise ValueError(
                f"prompt does not match the expected template "
                f"{self.prefix!r} ... {self.suffix!r}: {prompt[:60]!r}"
            )
        end = len(prompt) - len(self.suffix) if self.suffix else len(prompt)
        return prompt[len(self.prefix) : end]

    def fit(self, clean_data: Any, *, labels: Sequence[int] | None = None) -> StripVitaDetector:
        """Build the opposite-class TF-IDF reference pool from clean data only.

        Mirrors ``strip_vita.py:168-179``'s ``opposite_held_out_set``: the
        reference pool is CLEAN rows whose TRUE label is NOT the target
        class -- never test labels, a poison manifest, or known-poisoned
        rows.

        Args:
            clean_data: Already-rendered clean prompts.
            labels: The true label per row of ``clean_data`` -- required.

        Raises:
            ValueError: If ``labels`` is missing, misaligned, or no row has
                a label different from ``target_label``.
        """
        if labels is None:
            raise ValueError(
                "StripVitaDetector.fit needs labels=... (the true label per "
                "clean_data row) -- the opposite-class reference pool is "
                "selected by TRUE label (strip_vita.py:168)"
            )
        texts = list(clean_data)
        labels = list(labels)
        if len(labels) != len(texts):
            raise ValueError(f"labels ({len(labels)}) must align with clean_data ({len(texts)})")
        opposite = [
            self._middle(t) for t, y in zip(texts, labels, strict=True) if int(y) != self.target_label
        ]
        if not opposite:
            raise ValueError(
                f"no clean rows with label != target_label ({self.target_label}); "
                f"STRIP-ViTA needs an opposite-class reference pool"
            )
        self._vocab, self._tfidf = build_tfidf(opposite)
        self._fitted = True
        return self

    def score(self, samples: Any) -> np.ndarray:
        """``-mean(entropy of perturbed-copy predictions)``; ``-inf`` where
        the target-label gate excludes it. See the module docstring, points
        5-6, for the gate and the orientation derivation.
        """
        if self._vocab is None or self._tfidf is None:
            raise RuntimeError("StripVitaDetector scored before fit()")
        texts = [samples] if isinstance(samples, str) else list(samples)
        out = np.full(len(texts), -np.inf, dtype=np.float64)
        progress = progress_bar(self.show_progress, len(texts), "STRIP-ViTA score")
        try:
            for start in range(0, len(texts), self.chunk_size):
                sl = slice(start, start + self.chunk_size)
                chunk = texts[sl]
                ori_probs = verbalizer_probs_batch(
                    self.model, self.tokenizer, chunk, self.verbalizer, self.max_length
                )
                ori_pred = np.argmax(ori_probs, axis=-1)
                gate = ori_pred == self.target_label

                chunk_scores = np.full(len(chunk), -np.inf, dtype=np.float64)
                for j, prompt in enumerate(chunk):
                    if not gate[j]:
                        progress.update(1)
                        continue
                    words = self._middle(prompt).split(" ")
                    rng = _seed_for_text(self.seed, prompt)
                    ref_rows = rng.integers(0, len(self._tfidf), size=self.perturbation_number)
                    perturbed_prompts = [
                        self.prefix
                        + " ".join(perturb_one(words, self.replace_ratio, self._vocab, self._tfidf[r], rng))
                        + self.suffix
                        for r in ref_rows
                    ]
                    probs = verbalizer_probs_batch(
                        self.model, self.tokenizer, perturbed_prompts, self.verbalizer, self.max_length
                    )
                    chunk_scores[j] = -float(entropy_of(probs).mean())
                    progress.update(1)
                out[sl] = chunk_scores
        finally:
            progress.close()
        return out
