"""Input-level backdoor detection — reading the text, not the model.

This category is the control the representation-based detectors must be judged
against. If simply inspecting the input catches a trigger, nothing is gained by
opening the model up; the case for internal representations rests entirely on
triggers that leave no *textual* outlier while still steering the computation.

Implemented:

* :class:`OnionDetector` —

      Fanchao Qi, Yangyi Chen, Mukai Li, Yuan Yao, Zhiyuan Liu, Maosong Sun.
      "ONION: A Simple and Effective Defense Against Textual Backdoor Attacks."
      EMNLP 2021, 9558-9566. https://aclanthology.org/2021.emnlp-main.752/

  For each word, measure how much the sentence's perplexity *drops* when that
  word is deleted. A word that fits nowhere raises the perplexity of everything
  around it, so deleting it produces a large drop — which is exactly what an
  inserted rare token looks like, and exactly what a syntactic or style-transfer
  trigger does not.

**Expected asymmetry, stated in advance so it cannot be claimed post hoc.**
ONION is built to find inserted outlier *tokens*. Against a rare-word trigger it
should score well. Against a trigger that rewrites the sentence's syntax or
style — where every individual word is ordinary and only their arrangement
carries the signal — there is no outlier word to find and it should score near
chance. The paper itself reports this scope. Confirming that asymmetry is what
justifies the rest of this work package; failing to confirm it would mean the
representation-based detectors are solving a problem input inspection already
solves.

Deviation from the paper worth recording: the reference implementation scores
perplexity with GPT-2. GPT-2 is not in this project's model cache, so the
project's own generator is used instead. That changes the absolute perplexities
but not the ranking logic, and the threshold is calibrated on clean data here
rather than taken from the paper's fixed suspicion cutoff.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["OnionDetector"]


@register_detector("onion", category="input-anomaly")
class OnionDetector(Detector):
    """Perplexity-based outlier-word detection.

    For a sentence of ``n`` words this needs ``n + 1`` perplexity evaluations,
    so cost grows with input length — the dominant expense in this category and
    the reason ``max_words`` exists.

    Args:
        model: A causal LM used to score perplexity.
        tokenizer: Its tokenizer.
        aggregate: ``"max"`` — the single most suspicious word, the paper's
            notion applied per sentence; or ``"sum_positive"`` — the total
            suspicion, which is steadier when a trigger is a multi-word phrase.
        max_words: Words scored per input; longer inputs are truncated for
            scoring only. ``None`` scores every word.
        batch_size: Perplexity batch size.
    """

    name = "onion"
    category = "input-anomaly"
    modality = "text"
    requires_fit = False

    def __init__(
        self,
        *,
        model: Any = None,
        tokenizer: Any = None,
        aggregate: str = "max",
        max_words: int | None = 64,
        batch_size: int = 32,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, aggregate=aggregate, max_words=max_words, **config
        )
        if model is None or tokenizer is None:
            raise ValueError("OnionDetector needs model= and tokenizer=")
        if aggregate not in {"max", "sum_positive"}:
            raise ValueError(
                f"aggregate must be 'max' or 'sum_positive', got {aggregate!r}"
            )
        self.model, self.tokenizer = model, tokenizer
        self.aggregate = aggregate
        self.max_words = max_words
        self.batch_size = int(batch_size)

    def suspicion(self, text: str) -> np.ndarray:
        """Per-word suspicion scores for one input.

        Returns:
            ``(n_words,)`` where entry ``i`` is ``ppl(full) - ppl(without word
            i)``. Positive means deleting the word *reduced* perplexity, i.e.
            the word did not belong.
        """
        from ..models.llm import sequence_nll

        words = text.split()
        if self.max_words is not None:
            words = words[: self.max_words]
        if len(words) < 2:
            return np.zeros(len(words), dtype=np.float64)

        variants = [" ".join(words)]
        variants += [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        scored = sequence_nll(
            self.model, self.tokenizer, variants, batch_size=self.batch_size
        )
        ppl = np.array([s.perplexity for s in scored], dtype=np.float64)
        # ppl[0] is the intact sentence; ppl[i+1] drops word i.
        return ppl[0] - ppl[1:]

    def score(self, samples: Any) -> np.ndarray:
        """Per-input suspicion, comparable across inputs.

        Args:
            samples: A string or a sequence of strings.

        Returns:
            ``(n_texts,)`` scores; higher = more likely to carry an inserted
            outlier word.
        """
        texts = [samples] if isinstance(samples, str) else list(samples)
        out = np.zeros(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            f = self.suspicion(text)
            if not len(f):
                continue
            if self.aggregate == "max":
                out[i] = float(np.max(f))
            else:
                out[i] = float(np.sum(np.clip(f, 0.0, None)))
        # NaN guard: a degenerate perplexity (empty or single-token variant)
        # would otherwise propagate into the threshold quantile and poison the
        # whole calibration rather than just its own row.
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
