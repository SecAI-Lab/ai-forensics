"""Perplexity filtering for retrieved passages — a **documented negative control**.

The classic poisoned-input filter: score a text by its perplexity under a
reference causal LM and flag the outliers. It is the standard baseline in the
backdoor / prompt-injection literature and the one PoisonedRAG evaluates as a
defence:

    Wei Zou, Runpeng Geng, Binghui Wang, Jinyuan Jia. "PoisonedRAG: Knowledge
    Corruption Attacks to Retrieval-Augmented Generation of Large Language
    Models." USENIX Security 2025. (perplexity defence: §7.2)

**Expected to fail here, and that is the point.** The mechanism assumes the
adversarial text is statistically unnatural, which holds for gradient-optimised
gibberish (HotFlip / GCG / corpus-poisoning suffixes) but *not* for the
black-box PoisonedRAG poison this testbed injects: each injected passage is
``question + "." + LLM-written prose``, i.e. fluent by construction. Concretely:

* PoisonedRAG reports ROC AUC **0.12-0.48** for perplexity filtering against its
  black-box poison — at or *below* chance, meaning the statistic is sometimes
  anti-correlated with poisoning;
* this testbed independently measured median perplexity ~21 for the generated
  poison versus ~21 for real NQ passages — no separation at all.

A poor number from this detector is therefore a **result, not a bug**. It is
included so the report has an honest, literature-standard floor to compare the
retrieval-geometry detectors against, and so the "fluent poison defeats fluency
filters" claim is measured rather than asserted.

Two knobs exist to make the negative control fair rather than strawman:

* ``direction`` — ``"high"`` (default, the literature's convention: unnatural
  text has *high* perplexity) or ``"low"``. The inversion matters: an LLM
  writing at temperature 1 about a single asserted fact tends to produce text
  that is *more* predictable than a human-written encyclopaedia passage, so the
  discriminative direction against fluent poison, if any, is ``"low"``. Running
  both is what turns "AUC 0.3" into the interpretable statement "the poison is
  detectably *too fluent*". Reporting only ``"high"`` and calling it a failure
  hides that.
* ``windowed`` — score the max perplexity over a sliding window of ``window``
  tokens instead of the whole passage. Whole-passage averaging dilutes a short
  unnatural span inside otherwise fluent prose, which is exactly how BadRAG-style
  poison slips past a document-level perplexity filter. Windowing restores
  sensitivity to a gibberish prefix/suffix at the cost of more forward passes.

Cost: one LM forward pass per retrieved passage (whole-passage mode), so ``k``
per query — for the default ``k=5`` and Qwen3-1.7B that is a few hundred
milliseconds on one GPU. Windowed mode costs
``ceil((n_tokens - window) / stride) + 1`` passes per passage instead, ~4-6x for
a typical 100-token NQ passage at ``window=32, stride=16``; passes are batched.

No default threshold is set: perplexity is in nats/token on an arbitrary,
corpus- and model-dependent scale with no principled cut point, so the caller is
expected to sweep it (``aiforensics.metrics.threshold_sweep``) and report an ROC
curve. This is the score-based contract the project wants — the detector never
makes a hard decision of its own.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["PerplexityDetector"]


@register_detector("perplexity", category="rag-defense")
class PerplexityDetector(Detector):
    """Rank retrieved passages by their perplexity under a local causal LM.

    ``samples`` is ``(query, passages)``; the query is accepted for interface
    uniformity but **not used** — each passage is scored in isolation, which is
    what the baseline does. ``store`` and ``encoder`` are likewise accepted and
    ignored so the detector is constructible from the same call site as the
    retrieval-geometry detectors.

    Scoring formula, per passage ``p``:

        ``L(p) = nll_mean(p) = log perplexity(p)``            (whole-passage)
        ``L(p) = max over windows w of nll_mean(w)``          (``windowed=True``)
        ``score(p) = +L(p)`` if ``direction == "high"`` else ``-L(p)``

    Log perplexity rather than raw perplexity: the two are rank-equivalent, so
    ROC/PR are identical, but the log scale keeps a single degenerate passage
    from dominating a threshold sweep's axis.

    Passages too short to score (fewer than two tokens) yield ``nan`` from
    :func:`~aiforensics.models.llm.sequence_nll`; those are replaced with the
    median of the finite values in the same call (or 0.0 if none are finite) so
    they land mid-pack instead of being spuriously flagged or spuriously cleared.

    Args:
        store: Unused; accepted for a uniform constructor signature.
        encoder: Unused; accepted for a uniform constructor signature.
        model: A loaded causal LM. ``None`` loads ``model_key`` on first use.
        tokenizer: Its tokenizer. Must be supplied together with ``model``.
        threshold: Decision threshold. Left ``None`` deliberately — see the
            module docstring; sweep it instead.
        direction: ``"high"`` (default, literature convention) or ``"low"``.
        windowed: Use max-over-windows instead of whole-passage perplexity.
        window: Window length in tokens (default 32).
        stride: Window step in tokens; ``None`` means ``window // 2`` (50%
            overlap, so a short unnatural span is never split across windows).
        batch_size: Texts per forward pass.
        max_length: Truncate each scored text to this many tokens.
        model_key: Registry key / hub id used when ``model`` is ``None``.
        **config: Recorded verbatim by :class:`~.base.Detector`.

    Raises:
        ValueError: On an unknown ``direction``, or a non-positive ``window`` /
            ``stride``.
    """

    name = "perplexity"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False

    def __init__(
        self,
        *,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        direction: str = "high",
        windowed: bool = False,
        window: int = 32,
        stride: int | None = None,
        batch_size: int = 8,
        max_length: int | None = None,
        model_key: str = "qwen3-1.7b",
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, direction=direction, windowed=windowed,
            window=window, stride=stride, batch_size=batch_size,
            max_length=max_length, model_key=model_key, **config,
        )
        if direction not in {"high", "low"}:
            raise ValueError(f"direction must be 'high' or 'low', got {direction!r}")
        if window < 1:
            raise ValueError(f"window must be >= 1 token, got {window}")
        step = window // 2 if stride is None else stride
        if step < 1:
            raise ValueError(f"stride must be >= 1 token, got {stride}")
        self.direction = direction
        self.windowed = windowed
        self.window = int(window)
        self.stride = int(step)
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.model_key = model_key
        self._model = model
        self._tokenizer = tokenizer

    # -- model access -------------------------------------------------------- #
    def _ensure_model(self) -> tuple[Any, Any]:
        """Return ``(model, tokenizer)``, loading ``model_key`` on first use."""
        if self._model is None or self._tokenizer is None:
            from ..models.llm import load_llm

            logger.info("perplexity detector loading reference LM %s", self.model_key)
            self._model, self._tokenizer = load_llm(self.model_key)
        return self._model, self._tokenizer

    # -- scoring ------------------------------------------------------------- #
    def _windows(self, text: str) -> list[str]:
        """Split ``text`` into overlapping ``window``-token strings.

        Returns a single-element list when the text is shorter than one window,
        so short passages fall back to whole-passage perplexity.
        """
        _, tok = self._ensure_model()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.window:
            return [text]
        starts = range(0, max(len(ids) - self.window, 0) + 1, self.stride)
        return [tok.decode(ids[s : s + self.window]) for s in starts]

    def _log_perplexity(self, texts: Sequence[str]) -> np.ndarray:
        """Per-text log perplexity (nats/token), max-pooled over windows if asked."""
        from ..models.llm import sequence_nll

        model, tok = self._ensure_model()
        if not self.windowed:
            scored = sequence_nll(
                model, tok, list(texts),
                batch_size=self.batch_size, max_length=self.max_length,
            )
            return np.asarray([s.nll_mean for s in scored], dtype=np.float64)

        flat: list[str] = []
        owner: list[int] = []
        for i, text in enumerate(texts):
            chunks = self._windows(text)
            flat.extend(chunks)
            owner.extend([i] * len(chunks))
        scored = sequence_nll(
            model, tok, flat, batch_size=self.batch_size, max_length=self.max_length,
        )
        out = np.full(len(texts), np.nan, dtype=np.float64)
        for idx, s in zip(owner, scored, strict=False):
            out[idx] = np.fmax(out[idx], s.nll_mean)  # fmax ignores nan windows
        return out

    def score(self, samples: Any) -> np.ndarray:
        """Suspicion score for one query's retrieved passages. Higher = worse.

        Args:
            samples: ``(query, passages)``. The query is ignored.

        Returns:
            A 1-D float array aligned to ``passages``: log perplexity under
            ``direction="high"``, its negation under ``direction="low"``.
        """
        _query, passages = samples
        texts = [str(p) for p in passages]
        if not texts:
            return np.zeros(0, dtype=np.float64)
        logppl = _fill_nonfinite(self._log_perplexity(texts))
        return logppl if self.direction == "high" else -logppl


def _fill_nonfinite(values: np.ndarray) -> np.ndarray:
    """Replace ``nan``/``inf`` with the median of the finite entries (0.0 if none)."""
    finite = np.isfinite(values)
    if finite.all():
        return values
    filled = values.copy()
    filled[~finite] = float(np.median(values[finite])) if finite.any() else 0.0
    return filled
