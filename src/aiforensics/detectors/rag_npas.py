"""AV Filter / NPAS — generation-side attention attribution over retrieved passages.

    Sarthak Choudhary, Nils Palumbo, Ashish Hooda, Krishnamurthy Dj Dvijotham,
    Somesh Jha. "Through the Stealth Lens: Rethinking Attacks and Defenses in
    Retrieval-Augmented Generation." **ICML 2026 (poster)**.
    https://arxiv.org/abs/2506.04390 — algorithm in §4 / Algorithm 1.
    Reference implementation: https://github.com/sarthak-choudhary/Stealthy_Attacks_Against_RAG
    (``src/filtering.py``, ``src/models/Llama.py``, ``src/prompts.py``).

What signal this reads
----------------------
Not the passages. **The generator's attention.** The retrieved set is packed
into one prompt, the model answers greedily, and for every *response* token the
attention flowing back into the prompt is averaged over layers and heads and
attributed to whichever passage's token span it lands in. A poisoned passage
that actually steers the answer has to be *read* to do so, and reading is
attention: it captures a disproportionate share of the response's attention mass
and stands out against the co-retrieved passages.

This is a **causal-influence** signal on the generation side. It is orthogonal to
every other RAG detector in this testbed — embedding geometry (``ragdefender``),
token-level optimisation artifacts (``gmtp``), fluency (``perplexity``),
answer agreement (``agreement``) — and it never looks at the corpus, only at what
the generator did with the k passages it was handed.

Per the paper:

* ``Score_alpha(z_t, A) = sum over response tokens i, over the top-alpha
  attended positions j inside passage t's span, of A[i, j]`` where ``A`` is the
  layer- and head-averaged attention matrix;
* ``NPAS_t = 100 * Score_alpha(z_t) / sum_i Score_alpha(z_i)`` — normalised
  across the co-retrieved set so it is comparable across queries and models;
* Algorithm 1 (the AV filter) sorts the passages by NPAS to defuse position
  bias, then loops: recompute NPAS, and if ``Var(NPAS) <= delta`` **stop and
  declare the set clean**, otherwise drop ``argmax NPAS``; at most
  ``k - floor((1 - epsilon) * k)`` passages are ever removed.

Why this detector matters here: the abstain gate
------------------------------------------------
The variance test in Algorithm 1 is a real **abstain mechanism**. The
investigator does not know whether the database is poisoned, and
``Var(NPAS) <= delta`` is the method saying *"nothing anomalous in this
retrieved set"* without needing any labels. That is exposed directly as
:meth:`NpasDetector.is_clean` and :meth:`NpasDetector.npas_variance`, and it is
what :meth:`NpasDetector.predict` reproduces when no threshold is set. Behaviour
on a clean corpus — how often the gate correctly abstains — is a first-class
number to report, not a footnote.

What :meth:`NpasDetector.score` returns
---------------------------------------
The **continuous NPAS values**, one per passage, higher = more suspicious — so a
caller can sweep a threshold and draw an ROC curve. The paper's own operating
point is *not* a threshold on NPAS; it is the variance-gated top-1 removal, and
that is reproduced by :meth:`NpasDetector.predict` / :meth:`NpasDetector.filter`.
Report both: the ROC is our metric, the gated filter is theirs.

.. warning::
   **Do not threshold this score against a corpus-wide budget.** NPAS is a
   *compositional share*: the k values of one retrieved set sum to 100 by
   construction, so a passage's score depends on its co-retrieved neighbours and
   not only on itself. Picking a global cut that flags, say, 5% of clean
   passages therefore does not mean what it means for a per-passage detector. On
   NQ at ``k = 5`` that recipe produced a cut of **48.12**, i.e. "flag a passage
   only if it holds >= 48% of its own set's attention", against a uniform
   baseline of ``100/k = 20``. Almost nothing clears that bar and the reported
   detection rate collapses to 0.000 — an artifact of the protocol, not of the
   signal (measured poisoned-vs-clean separation was +15 to +50 NPAS whenever
   the attack actually steered the generation). The meaningful per-set protocols
   are (a) the paper's own gated top-1 removal via :meth:`NpasDetector.filter` /
   :meth:`NpasDetector.predict` with no threshold, and (b) a within-set ranking
   metric (is the injected passage the argmax of its own set?). If a global
   threshold really is needed, use ``score_mode="ratio"`` so the value is at
   least ``k``-invariant, and say in the write-up that the cut is a share.

Expected failure modes — read these before believing a number
-------------------------------------------------------------
* **The signal is causally downstream of attack success.** A poisoned passage
  that was retrieved but did *not* steer the generator captures no unusual
  attention, and is a false negative by construction. The paper measures
  *Corruption Identification Rate* (CIR: the fraction of corrupted **sets**
  flagged, computed **only over attacks that already succeeded** against vanilla
  RAG, average 0.78). Our metric is recall over **all injected passages**
  regardless of whether they flipped the answer, on a per-passage basis. Ours
  will be lower, and the two numbers are not interchangeable. Report both:
  passage-level recall for the project metric, and recall restricted to
  answer-flipping queries as the paper-comparable figure.
* **No standalone score exists.** NPAS is normalised across the co-retrieved
  set; removing a passage changes every other passage's score. This is a
  **retrieved-set detector**. It cannot be run as a whole-corpus scan, cannot
  score a document in isolation, and its scores are only commensurable across
  queries when ``k`` is fixed (use ``score_mode="ratio"`` when it is not).
* **Multiple poisoned passages flatten the variance.** PoisonedRAG injects
  ``N`` passages asserting the *same* fact. If several are co-retrieved they
  split the attention mass between them, the NPAS distribution becomes flat
  again, and the variance gate abstains on a set that is heavily poisoned. This
  is the paper's own ``epsilon < 0.5`` bounded-corruption assumption showing its
  teeth; at ``N/k`` above ~0.3 expect recall to fall off.
* **``delta`` is scale-coupled to ``k``.** Uniform attention gives
  ``NPAS = 100/k`` and variance 0, but the *spread* around it scales like
  ``1/k^2``. The paper's ``delta = 26.2`` was estimated at ``k = 10`` with
  Llama-2-7B and ``alpha = inf``; it is **not** reused here. Call
  :meth:`NpasDetector.fit` on clean retrievals from *this* corpus, *this*
  generator and *this* ``k``. Without it the gate is unavailable and says so.
* **Position bias is only partly removed.** One sort pass (the paper's
  Algorithm 1) demotes recency effects but does not eliminate them; the
  reference code sorts twice. ``debias_rounds`` exposes that as an ablation.
* **Attention is not attribution.** Averaging over all layers and heads is a
  crude estimator — induction heads, attention sinks and positional heads are
  all mixed in. It is what the paper does and it works well enough, but a
  passage can be *used* through a low-attention pathway, and a delimiter-heavy
  passage can attract attention it does not use.
* **White-box (HotFlip) poison is a different regime.** Gibberish prefixes are
  optimised against the *retriever*, not the generator; whether they also grab
  generator attention is an empirical question this detector answers rather than
  assumes.

Cost
----
``debias_rounds + 1`` greedy generations per query for :meth:`score`
(2 by default), and up to ``debias_rounds + 1 + max_removals`` for
:meth:`filter`. Each generation is one prefill plus ``max_new_tokens`` decode
steps over a ~``k * 100``-token prompt with ``attn_implementation="eager"``.

Memory is dominated by the **prefill** attention tensor, which
``generate(output_attentions=True)`` materialises and this detector then throws
away: ``n_layers * n_heads * T^2`` entries, i.e. for Qwen3-1.7B (28 layers,
16 heads) ~0.45 GB at ``T = 700`` and ~3.6 GB at ``T = 2000`` in bfloat16. The
reference loads the model in float32, which doubles that for no benefit at this
stage of the pipeline; bfloat16 is used here instead and the per-step attention
rows are accumulated in float32 so the sum over hundreds of positions does not
lose precision.
"""

from __future__ import annotations

import logging
import math
import string
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["NpasDetector", "PROMPT_TEMPLATE", "PASSAGE_SEPARATOR"]

#: Passage block separator. Together with the ``[i] `` markers this reproduces
#: ``wrap_prompt``'s ``'\n\n'.join(f"[{i+1}] {s}")`` from the reference code.
PASSAGE_SEPARATOR = "\n\n"

#: The reference ``QA_PROMPT_TMPL`` (``src/prompts.py``), verbatim apart from the
#: placeholder syntax. Short keyword answers are deliberate: attention mass is
#: summed over response tokens, so a rambling answer dilutes the signal.
PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords. \n"
    "If there is no relevant information, just say \"I don't know\".\n"
    "Query: {question}\n"
    "Answer: "
)

_PUNCTUATION = frozenset(string.punctuation)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
@contextmanager
def _eager_attention(model: Any) -> Iterator[None]:
    """Force ``attn_implementation="eager"`` for the duration of the block.

    Attention weights are simply not produced under SDPA or FlashAttention — the
    fused kernels never materialise the probability matrix — so a model handed to
    us by the harness under the default SDPA backend would silently yield
    ``None`` attentions (or a hard error, depending on the transformers version).
    The switch is reverted on exit because the model instance is shared with
    every other detector in the run.
    """
    config = getattr(model, "config", None)
    previous = getattr(config, "_attn_implementation", None) if config is not None else None
    if previous == "eager" or config is None:
        yield
        return

    setter = getattr(model, "set_attn_implementation", None)
    used_setter = False
    if callable(setter):
        try:
            setter("eager")
            used_setter = True
        except Exception:
            logger.debug("set_attn_implementation('eager') failed; patching config", exc_info=True)
    if not used_setter:
        config._attn_implementation = "eager"
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            text_config._attn_implementation = "eager"
    logger.debug("npas: switched attention implementation %r -> 'eager'", previous)
    try:
        yield
    finally:
        try:
            # ``is not None`` rather than truthiness in *both* branches: with a
            # falsy-but-present previous value (``""``) the old ``if used_setter
            # and previous`` fell through to an ``elif previous is not None``
            # that was also skipped, so neither restore ran and the model — which
            # is shared with every other detector in the run — stayed in eager
            # for the rest of the process. Defensive only; the observed states
            # are "sdpa"/"flash_attention_2".
            if used_setter and previous is not None:
                setter(previous)
            elif previous is not None:
                config._attn_implementation = previous
                text_config = getattr(config, "text_config", None)
                if text_config is not None:
                    text_config._attn_implementation = previous
        except Exception:
            logger.warning("npas: could not restore attn_implementation %r", previous, exc_info=True)


def _build_prompt(query: str, passages: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Render the QA prompt and return each passage's character span.

    The spans **tile** the context block exactly the way the reference code does:
    passage 0 starts at its ``[1]`` marker, and every later passage's span starts
    where the previous one's text ended — so the ``"\\n\\n[i] "`` separator and
    marker are charged to the passage that follows them. Changing the prompt
    format without changing this mapping silently misattributes attention, which
    is why the two live in one function.

    Args:
        query: The user question.
        passages: Retrieved passage texts, in prompt order.

    Returns:
        ``(prompt, spans)`` where ``spans[i] = (start_char, end_char)`` into
        ``prompt``.
    """
    head, tail = PROMPT_TEMPLATE.split("{context}", 1)
    parts: list[str] = []
    cursor = len(head)
    marker_start: list[int] = []
    text_end: list[int] = []
    for i, passage in enumerate(passages):
        if i:
            parts.append(PASSAGE_SEPARATOR)
            cursor += len(PASSAGE_SEPARATOR)
        marker = f"[{i + 1}] "
        marker_start.append(cursor)
        body = marker + str(passage)
        parts.append(body)
        cursor += len(body)
        text_end.append(cursor)

    context = "".join(parts)
    prompt = head + context + tail.format(question=query)

    spans: list[tuple[int, int]] = []
    for i in range(len(passages)):
        start = marker_start[0] if i == 0 else text_end[i - 1]
        spans.append((start, text_end[i]))
    return prompt, spans


def _char_to_token_boundaries(
    tokenizer: Any, text: str, positions: Sequence[int]
) -> list[int]:
    """Map character offsets to token-count prefixes.

    ``boundary(p)`` is the number of tokens of ``text`` that end at or before
    character ``p``, obtained from a single tokenisation via the fast
    tokenizer's offset mapping — cheaper than the reference's re-tokenisation of
    ``msg[:p]``, and immune to the prefix/full BPE-merge mismatches a truncated
    re-tokenisation can introduce mid-word.

    The two paths use **different conventions for a token that straddles the
    cut**, and this is a deliberate approximation rather than an equivalence:

    * fast path (offset mapping) — *exclusive*: a token whose span crosses ``p``
      ends after ``p``, so it is charged to the passage that **follows** the cut;
    * slow path (prefix re-tokenisation, and the reference) — *inclusive*: the
      prefix ``text[:p]`` tokenises to a piece covering the straddling
      characters, so the token is charged to the passage **before** the cut.

    Immaterial for this prompt and tokenizer: cuts fall on the
    ``"\\n\\n[i] "`` separator, which no BPE piece straddles, so the two agree
    token-for-token in practice. And even if one did, the spans built by
    :func:`_build_prompt` *tile* the context, so a disagreement can only shift a
    single boundary token from one passage to its neighbour — it can never make
    spans overlap or drop a region, and one token out of a ~100-token passage
    cannot move NPAS materially. The slow path is used only when the tokenizer
    exposes no offsets.
    """
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
    except (TypeError, ValueError, NotImplementedError, KeyError):
        offsets = None

    if offsets:
        ends = np.maximum.accumulate(np.asarray([end for _start, end in offsets], dtype=np.int64))
        return [int(np.searchsorted(ends, pos, side="right")) for pos in positions]

    logger.debug("npas: tokenizer has no offset mapping; using prefix re-tokenisation")
    return [
        len(tokenizer(text[:pos], add_special_tokens=False)["input_ids"]) for pos in positions
    ]


def _uniform(k: int) -> np.ndarray:
    """The degenerate NPAS vector: every passage takes ``100/k`` of the mass."""
    return np.full(k, 100.0 / k, dtype=np.float64) if k else np.zeros(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------
@register_detector("npas", category="rag-defense")
class NpasDetector(Detector):
    """Normalised Passage Attention Score with the paper's attention-variance gate.

    ``samples`` is ``(query, passages)``. ``score`` returns one NPAS value per
    passage in **input order**, higher = more suspicious. The scores of one call
    sum to 100 by construction (they are shares of a fixed attention budget), so
    they are comparable across queries only while ``k`` is constant; set
    ``score_mode="ratio"`` to divide out the ``100/k`` uniform baseline when it
    is not.

    Args:
        store: Unused; accepted so the constructor matches the other RAG
            detectors. Ground-truth fields on it are never read.
        encoder: Unused; same reason.
        model: A loaded **causal** LM. ``None`` loads ``model_key`` lazily on
            first use with ``attn_implementation="eager"``. An injected model is
            switched to eager attention for the duration of each forward pass and
            switched back afterwards. Qwen3 is causal and has **no mask token** —
            nothing here asks for one.
        tokenizer: The model's tokenizer; supply it together with ``model``.
        threshold: Decision threshold on NPAS for :meth:`predict`. Left ``None``
            on purpose: the paper's operating point is the variance gate, not an
            NPAS cut, and :meth:`predict` falls back to the gate when this is
            unset. Sweep it with :mod:`aiforensics.metrics` for the ROC.
        alpha: Keep only the top-``alpha`` attended token positions inside each
            passage when summing its score; ``None`` (the reference default,
            ``top_tokens=1_000_000``) uses all of them. The paper sweeps
            ``{5, 10, None}`` and notes ``alpha`` should track the number of
            "heavy hitter" tokens in the poison, roughly the target answer's
            length.
        delta: Variance threshold for the abstain gate. **No default.** The
            paper's ``26.2`` is bound to ``k = 10``, Llama-2-7B and
            ``alpha = inf`` and must not be transplanted; use :meth:`fit`.
        epsilon: Corruption fraction bounding how many passages the filter may
            remove: ``k - floor((1 - epsilon) * k)``, i.e. 1 of 10 at the
            default. Must satisfy ``0 < epsilon < 0.5`` (the paper's bounded
            corruption assumption).
        debias_rounds: Position de-biasing passes — each one re-sorts the
            passages ascending by NPAS and recomputes. Algorithm 1 does this
            once (the default); the reference ``filtering.py`` does it twice, so
            ``2`` is the ablation. ``0`` disables de-biasing and keeps retrieval
            order, which leaves the recency bias in the scores.
        max_new_tokens: Greedy generation budget. The reference config uses 25;
            32 gives a little headroom for Qwen3 while keeping the answer short,
            because attention is summed over response tokens and a long answer
            dilutes the signal.
        score_mode: ``"npas"`` (default, values summing to 100) or ``"ratio"``
            (``NPAS * k / 100``: 1.0 = exactly its uniform share), which is the
            ``k``-invariant form to use when ``k`` varies across queries.
        zero_punctuation: Zero the attention landing on pure-punctuation prompt
            tokens before scoring, as the reference does. Punctuation and
            delimiters are classic attention sinks and carry no passage content.
        punctuation_mode: Which tokens ``zero_punctuation`` treats as
            punctuation. ``"raw"`` (default) reproduces the reference exactly —
            the raw tokenizer piece must itself be a single character of
            :data:`string.punctuation`. ``"decoded"`` decodes and strips first,
            which also zeroes ``"..."``, ``"--"`` and space-prefixed pieces; see
            :meth:`_is_punctuation` for why that is an ablation and not the
            default.
        model_key: Registry key / hub id used when ``model`` is ``None``.
        **config: Recorded verbatim by :class:`~.base.Detector`.

    Raises:
        ValueError: On a non-positive ``alpha``, an ``epsilon`` outside
            ``(0, 0.5)``, a negative ``debias_rounds``, a non-positive
            ``max_new_tokens``, an unknown ``score_mode``, or an unknown
            ``punctuation_mode``.
    """

    name = "npas"
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
        alpha: int | None = None,
        delta: float | None = None,
        epsilon: float = 0.1,
        debias_rounds: int = 1,
        max_new_tokens: int = 32,
        score_mode: str = "npas",
        zero_punctuation: bool = True,
        punctuation_mode: str = "raw",
        model_key: str = "qwen3-1.7b",
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, alpha=alpha, delta=delta, epsilon=epsilon,
            debias_rounds=debias_rounds, max_new_tokens=max_new_tokens,
            score_mode=score_mode, zero_punctuation=zero_punctuation,
            punctuation_mode=punctuation_mode, model_key=model_key, **config,
        )
        if alpha is not None and alpha < 1:
            raise ValueError(f"alpha must be >= 1 token or None (all tokens), got {alpha}")
        if not 0.0 < epsilon < 0.5:
            raise ValueError(
                f"epsilon must lie in (0, 0.5) — the paper's bounded-corruption "
                f"assumption — got {epsilon}"
            )
        if debias_rounds < 0:
            raise ValueError(f"debias_rounds must be >= 0, got {debias_rounds}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
        if score_mode not in {"npas", "ratio"}:
            raise ValueError(f"score_mode must be 'npas' or 'ratio', got {score_mode!r}")
        if punctuation_mode not in {"raw", "decoded"}:
            raise ValueError(
                f"punctuation_mode must be 'raw' or 'decoded', got {punctuation_mode!r}"
            )

        self.alpha = None if alpha is None else int(alpha)
        self.delta = None if delta is None else float(delta)
        self.epsilon = float(epsilon)
        self.debias_rounds = int(debias_rounds)
        self.max_new_tokens = int(max_new_tokens)
        self.score_mode = score_mode
        self.zero_punctuation = bool(zero_punctuation)
        self.punctuation_mode = punctuation_mode
        self.model_key = model_key

        #: Populated by :meth:`fit`; goes straight into the run manifest.
        self.mean_clean_variance: float | None = None
        self.std_clean_variance: float | None = None
        self.n_clean_sets: int = 0

        self._model = model
        self._tokenizer = tokenizer
        self._punct_cache: dict[int, bool] = {}
        self._cache_key: tuple[str, tuple[str, ...]] | None = None
        self._cache_value: np.ndarray | None = None

    # -- model access ------------------------------------------------------- #
    def _ensure_model(self) -> tuple[Any, Any]:
        """Return ``(model, tokenizer)``, loading ``model_key`` on first use."""
        if self._model is None or self._tokenizer is None:
            from ..models.llm import load_llm

            logger.info(
                "npas loading generator %s (eager attention; attention weights are "
                "unavailable under SDPA)", self.model_key,
            )
            self._model, self._tokenizer = load_llm(
                self.model_key, attn_implementation="eager"
            )
            self._punct_cache.clear()
        return self._model, self._tokenizer

    # -- token bookkeeping -------------------------------------------------- #
    def _is_punctuation(self, token_id: int) -> bool:
        """Whether a **prompt** token counts as punctuation and is zeroed.

        Two conventions, selected by ``punctuation_mode``:

        * ``"raw"`` (default) — the reference's test: the raw tokenizer piece is
          looked up in :data:`string.punctuation`, which is a set of *single*
          characters, so only single-character pieces can ever match.
          Space-prefixed pieces (SentencePiece ``"▁."``, byte-BPE ``"Ġ."``) and
          multi-character runs such as ``"..."`` are left alone.
        * ``"decoded"`` — decode, strip, then zero any token whose decoded form
          is entirely punctuation. This catches the space-prefixed sinks the
          reference misses, but it also zeroes ``"..."``, ``"--"``, ``"!!"`` and
          friends, i.e. strictly more positions than the reference.

        ``"raw"`` is the default because the extra zeroing is **not symmetric
        across the threat models benchmarked here**. HotFlip-optimised prefixes
        (the white-box PoisonedRAG variant) are arbitrary vocabulary tokens, and
        a non-trivial fraction of them decode to punctuation-only strings; the
        natural-language passages they sit next to contain almost none. So
        ``"decoded"`` deletes attention mass preferentially from *poison* spans
        and depresses exactly the scores this detector is judged on, on the axis
        where its numbers already look anomalous. Fidelity to the reference is
        the defensible default; ``punctuation_mode="decoded"`` keeps the old
        behaviour available as a one-flag ablation.
        """
        cached = self._punct_cache.get(token_id)
        if cached is not None:
            return cached
        _model, tok = self._ensure_model()
        if self.punctuation_mode == "raw":
            piece: str | None = None
            converter = getattr(tok, "convert_ids_to_tokens", None)
            if callable(converter):
                try:
                    piece = converter(token_id)
                except Exception:
                    piece = None
            if piece is None:  # no raw vocabulary view; decoded is all we have
                piece = tok.decode([token_id], skip_special_tokens=False)
            # Membership in a frozenset of single characters — multi-character
            # pieces are never punctuation under this convention, by design.
            verdict = piece in _PUNCTUATION
        else:
            text = tok.decode([token_id], skip_special_tokens=False).strip()
            verdict = bool(text) and all(ch in _PUNCTUATION for ch in text)
        self._punct_cache[token_id] = verdict
        return verdict

    # -- one attention pass ------------------------------------------------- #
    def _passage_scores(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """``Score_alpha`` per passage for one ordering of the retrieved set.

        Runs a single greedy generation with ``output_attentions=True``, averages
        each decode step's attention over layers then heads, slices it to the
        prompt region, zeroes punctuation positions, and sums the result over
        response tokens. Response tokens stop at the first newline and skip
        punctuation, whitespace and special tokens, as in the reference.

        Returns:
            ``(len(passages),)`` of non-negative raw scores. All-zero when the
            model emitted nothing usable, which the caller turns into a uniform
            NPAS vector.
        """
        import torch

        model, tok = self._ensure_model()
        from ..models.llm import chat_prompt, model_device

        prompt, char_spans = _build_prompt(query, passages)
        # Thinking off: a Qwen3 <think> block would spend the whole response
        # budget on reasoning tokens whose attention is dominated by the CoT
        # itself rather than by the retrieved passages.
        rendered = chat_prompt(tok, prompt, enable_thinking=False)
        offset = rendered.find(prompt)
        if offset < 0:  # chat template mangled the body; fall back to raw prompt
            logger.warning("npas: chat template altered the prompt body; scoring raw prompt")
            rendered, offset = prompt, 0

        cuts = [offset + pos for span in char_spans for pos in span]
        boundaries = _char_to_token_boundaries(tok, rendered, cuts)

        encoded = tok(rendered, add_special_tokens=False, return_tensors="pt")
        input_ids = encoded["input_ids"]
        prompt_len = int(input_ids.shape[1])
        token_spans = [
            (
                min(max(boundaries[2 * i], 0), prompt_len),
                min(max(boundaries[2 * i + 1], 0), prompt_len),
            )
            for i in range(len(passages))
        ]

        device = model_device(model)
        batch = {name: value.to(device) for name, value in encoded.items()}
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        with torch.no_grad(), _eager_attention(model):
            outputs = model.generate(
                **batch,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                output_attentions=True,
                return_dict_in_generate=True,
                pad_token_id=pad_id,
            )

        total = self._accumulate_attention(tok, outputs, prompt_len, input_ids[0].tolist())
        del outputs

        scores = np.zeros(len(passages), dtype=np.float64)
        if total is None:
            logger.warning(
                "npas: no usable response tokens for query %r; falling back to a "
                "uniform attention split", query[:60],
            )
            return scores
        empty_spans = 0
        for i, (start, end) in enumerate(token_spans):
            if end <= start:
                empty_spans += 1
                logger.debug("npas: empty token span for passage %d", i)
                continue
            window = total[start:end]
            if self.alpha is not None and self.alpha < window.size:
                window = np.partition(window, -self.alpha)[-self.alpha :]
            scores[i] = float(window.sum())
        if empty_spans and empty_spans == len(token_spans):
            # A total collapse of the char->token mapping leaves every raw score
            # at 0, which _npas_for_order turns into the uniform 100/k vector and
            # the harness reports as AUC 0.5 — indistinguishable in the output
            # from a genuinely even attention split. Warn at the same level as the
            # "no usable response tokens" path above so silent degradation can
            # never be mistaken for a real (uninformative) result.
            logger.warning(
                "npas: all %d passage token spans were empty for query %r "
                "(character->token mapping collapsed); falling back to a uniform "
                "attention split", empty_spans, query[:60],
            )
        return scores

    def _accumulate_attention(
        self, tokenizer: Any, outputs: Any, prompt_len: int, prompt_ids: list[int]
    ) -> np.ndarray | None:
        """Sum layer/head-averaged attention over response tokens into the prompt.

        Returns ``None`` when no response token qualified (empty answer, or an
        answer that is a single token — the last generated token's outgoing
        attention is never computed, since producing it would need one more
        forward pass).
        """
        import torch

        attentions = getattr(outputs, "attentions", None)
        if not attentions:
            raise RuntimeError(
                "generate() returned no attentions. The model is not running with "
                "attn_implementation='eager'; SDPA and FlashAttention never "
                "materialise the attention matrix."
            )
        sequences = outputs.sequences
        generated = sequences[0, prompt_len:].tolist()
        special = set(tokenizer.all_special_ids or [])

        mask = None
        if self.zero_punctuation:
            keep = [0.0 if self._is_punctuation(t) else 1.0 for t in prompt_ids]
            mask = torch.tensor(keep, dtype=torch.float32)

        total: torch.Tensor | None = None
        # attentions[j] (j >= 1) is the decode step whose *query* token is
        # generated[j - 1]; attentions[0] is the prefill and is discarded. The
        # final generated token therefore has no outgoing attention.
        for i in range(min(len(generated), len(attentions)) - 1):
            token_id = int(generated[i])
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            if total is not None and "\n" in text:
                break  # first line of the answer only
            stripped = text.strip()
            if token_id in special or not stripped:
                continue
            if all(ch in _PUNCTUATION for ch in stripped):
                continue

            layers = attentions[i + 1]
            if not layers or layers[0] is None:
                continue
            # float32 before averaging: a bfloat16 mean over 28 layers x 16 heads
            # and then a sum over hundreds of key positions loses several digits.
            step = torch.stack(list(layers)).to(torch.float32)  # (layers, 1, heads, q, keys)
            averaged = step.mean(dim=0).mean(dim=1)[0]          # (q, keys)
            row = averaged[-1, :prompt_len].detach().to("cpu", dtype=torch.float32)
            if mask is not None:
                row = row * mask
            total = row if total is None else total + row

        return None if total is None else total.numpy().astype(np.float64)

    # -- NPAS --------------------------------------------------------------- #
    def _npas_for_order(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """NPAS (percentages summing to 100) for one concrete passage ordering."""
        if not passages:
            return np.zeros(0, dtype=np.float64)
        if len(passages) == 1:
            return np.array([100.0], dtype=np.float64)
        raw = self._passage_scores(query, passages)
        total = float(raw.sum())
        if not np.isfinite(total) or total <= 0.0:
            return _uniform(len(passages))
        return 100.0 * raw / total

    def _debiased_order(self, query: str, passages: Sequence[str]) -> list[int]:
        """Indices of ``passages`` re-sorted ascending by NPAS, ``debias_rounds`` times.

        Sorting ascending puts the highest-attention passage last, which is where
        the generator's recency bias would have put a passage anyway — so a
        passage that keeps the top slot across the re-sort is high because of its
        content, not its position.
        """
        order = list(range(len(passages)))
        for _ in range(self.debias_rounds):
            scores = self._npas_for_order(query, [passages[i] for i in order])
            order = [order[j] for j in np.argsort(scores, kind="stable")]
        return order

    def npas(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Normalised Passage Attention Scores, aligned to the **input** order.

        Runs ``debias_rounds`` position-de-biasing passes and one final scoring
        pass, then maps the result back onto the caller's ordering. Values are
        percentages summing to 100; the uniform baseline is ``100/k``.

        Args:
            query: The user question.
            passages: Retrieved passage texts.

        Returns:
            ``(len(passages),)`` float array.
        """
        texts = [str(p) for p in passages]
        key = (str(query), tuple(texts))
        if self._cache_key == key and self._cache_value is not None:
            return self._cache_value.copy()

        if len(texts) < 2:
            values = _uniform(len(texts))
        else:
            order = self._debiased_order(query, texts)
            sorted_scores = self._npas_for_order(query, [texts[i] for i in order])
            values = np.empty(len(texts), dtype=np.float64)
            for position, original in enumerate(order):
                values[original] = sorted_scores[position]

        self._cache_key, self._cache_value = key, values
        return values.copy()

    def npas_variance(self, query: str, passages: Sequence[str]) -> float:
        """Population variance of the NPAS vector — the statistic the gate tests.

        Zero for a set the generator reads evenly (and trivially for ``k < 2``);
        large when one passage has captured the response.
        """
        values = self.npas(query, passages)
        return 0.0 if values.size < 2 else float(np.var(values))

    def is_clean(self, query: str, passages: Sequence[str]) -> bool:
        """The abstain gate: ``True`` means "nothing anomalous in this set".

        This is the paper's ``sigma^2 <= delta`` test and the reason this
        detector is worth having in a setting where the investigator does not
        know whether the database is poisoned at all.

        Raises:
            RuntimeError: If ``delta`` has not been set or calibrated. There is
                deliberately no default — see :meth:`fit`.
        """
        return self.npas_variance(query, passages) <= self._require_delta()

    def _require_delta(self) -> float:
        if self.delta is None:
            raise RuntimeError(
                "npas has no variance threshold delta. It is scale-coupled to k "
                "(uniform NPAS = 100/k), so the paper's 26.2 (k=10, Llama-2-7B, "
                "alpha=inf) must not be reused. Call detector.fit(clean_data) with "
                "clean (query, passages) retrievals at this k, or pass delta=… "
                "explicitly so the choice is recorded."
            )
        return self.delta

    # -- Algorithm 1 -------------------------------------------------------- #
    def max_removals(self, k: int) -> int:
        """How many passages Algorithm 1 may drop: ``k - floor((1 - epsilon) * k)``.

        At the paper's ``epsilon = 0.1, k = 10`` this is 1. Written in the
        paper's own form rather than ``floor(epsilon * k)`` so that it never
        collapses to zero for the smaller ``k`` this testbed retrieves — at
        ``k = 5`` the two disagree (1 versus 0), and a filter allowed no removals
        is not a filter.
        """
        return max(0, k - math.floor((1.0 - self.epsilon) * k))

    def filter(
        self, query: str, passages: Sequence[str]
    ) -> tuple[list[int], list[int]]:
        """Run Algorithm 1 and report which passages it kept and removed.

        Args:
            query: The user question.
            passages: Retrieved passage texts.

        Returns:
            ``(kept, removed)`` — index lists into ``passages``. ``removed`` is
            in removal order, most suspicious first; it is empty when the
            variance gate abstained.

        Raises:
            RuntimeError: If ``delta`` is unavailable (see :meth:`fit`).
        """
        delta = self._require_delta()
        texts = [str(p) for p in passages]
        k = len(texts)
        budget = self.max_removals(k)
        if k < 2 or budget == 0:
            return list(range(k)), []

        kept = self._debiased_order(query, texts)
        removed: list[int] = []
        while len(removed) < budget:
            scores = self._npas_for_order(query, [texts[i] for i in kept])
            if float(np.var(scores)) <= delta:
                break
            removed.append(kept.pop(int(np.argmax(scores))))
        return sorted(kept), removed

    # -- Detector API ------------------------------------------------------- #
    def fit(self, clean_data: Any, **kwargs: Any) -> NpasDetector:
        """Calibrate ``delta`` on believed-clean retrievals: mean + 1 std.

        This is the paper's own recipe (§5: NPAS variance measured over benign
        retrievals on a clean QA set, ``delta`` set to mean plus one standard
        deviation), re-run on *this* corpus, generator and ``k`` instead of
        importing their 26.2. The one-sided margin is intentional and biases the
        gate towards *not* flagging: dropping a benign passage is cheap when most
        of the context is intact, so the paper prefers false negatives to false
        positives here. Tighten it by passing ``n_std=0`` if recall matters more
        than the clean-corpus false-positive rate.

        Args:
            clean_data: Iterable of ``(query, passages)`` from a corpus believed
                clean. Sets with fewer than two passages are skipped — their
                variance is trivially 0 and would drag ``delta`` down.
            **kwargs: ``n_std`` (default 1.0) scales the standard-deviation
                margin.

        Returns:
            ``self``.

        Raises:
            ValueError: If no usable calibration set was supplied.
        """
        n_std = float(kwargs.get("n_std", 1.0))
        variances: list[float] = []
        for item in clean_data:
            query, passages = item
            texts = [str(p) for p in passages]
            if len(texts) < 2:
                continue
            variances.append(self.npas_variance(query, texts))

        if not variances:
            raise ValueError(
                "npas.fit needs clean (query, passages) retrievals with k >= 2 "
                "passages each; none were supplied."
            )

        array = np.asarray(variances, dtype=np.float64)
        self.mean_clean_variance = float(array.mean())
        self.std_clean_variance = float(array.std())
        self.n_clean_sets = int(array.size)
        self.delta = self.mean_clean_variance + n_std * self.std_clean_variance
        logger.info(
            "npas calibrated on %d clean sets: mean variance %.4g, std %.4g -> "
            "delta %.4g (n_std=%.2f)",
            self.n_clean_sets, self.mean_clean_variance, self.std_clean_variance,
            self.delta, n_std,
        )
        self._fitted = True
        return self

    def score(self, samples: Any) -> np.ndarray:
        """NPAS per passage for one query's retrieved set. Higher = more suspicious.

        Args:
            samples: ``(query, passages)``.

        Returns:
            ``(len(passages),)`` float array in input order — the **continuous**
            score, so a caller can sweep a threshold and get an ROC curve. The
            paper's own operating point is not a cut on this value but the
            variance-gated top-1 removal; :meth:`predict` and :meth:`filter`
            reproduce that.

        Note:
            These values are *shares of one set's attention* and sum to 100 over
            the k passages, so a **corpus-wide** threshold on them is not a
            meaningful protocol — a 5%-clean-budget cut on NQ at ``k = 5`` came
            out at 48.12, i.e. "flag only a passage holding >= 48% of its own
            set's attention" against a uniform baseline of ``100/k = 20``, and
            drove the reported detection to 0.000 while the underlying
            poisoned-vs-clean separation was +15 to +50. Evaluate per set: the
            gated filter (:meth:`predict` with no threshold) or a within-set
            rank. See the module docstring.
        """
        query, passages = samples
        values = self.npas(str(query), list(passages))
        if self.score_mode == "ratio" and values.size:
            return values * (values.size / 100.0)
        return values

    def predict(self, samples: Any, threshold: float | None = None) -> np.ndarray:
        """Flag passages, reproducing Algorithm 1 when no threshold is given.

        With a threshold (argument or ``self.threshold``) this is the ordinary
        ``score > threshold`` cut, for comparability with the other detectors.
        Without one it runs the paper's operating point: de-bias, then remove
        ``argmax NPAS`` while the variance gate stays open — so an all-``False``
        result is a genuine *"nothing anomalous here"* abstention rather than a
        missing threshold.

        Args:
            samples: ``(query, passages)``.
            threshold: Optional NPAS cut; falls back to ``self.threshold``, then
                to the variance gate.

        Returns:
            Boolean array in input order, ``True`` = flagged.
        """
        thr = threshold if threshold is not None else self.threshold
        if thr is not None:
            return super().predict(samples, threshold=thr)
        query, passages = samples
        texts = [str(p) for p in passages]
        _kept, removed = self.filter(str(query), texts)
        flags = np.zeros(len(texts), dtype=bool)
        flags[removed] = True
        return flags

    def describe(self) -> dict:
        """Base description plus the calibrated gate, for the run manifest."""
        info = super().describe()
        info.update(
            delta=self.delta,
            mean_clean_variance=self.mean_clean_variance,
            std_clean_variance=self.std_clean_variance,
            n_clean_sets=self.n_clean_sets,
            alpha=self.alpha,
            epsilon=self.epsilon,
            debias_rounds=self.debias_rounds,
            score_mode=self.score_mode,
            punctuation_mode=self.punctuation_mode,
            model_key=self.model_key,
        )
        return info
