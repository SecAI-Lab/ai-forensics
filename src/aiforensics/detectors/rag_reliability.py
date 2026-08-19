"""ReliabilityRAG — flag the passages that a maximum independent set leaves out.

The method
----------
    Zeyu Shen, Rahul Mazumder, Elisa Celis, Vijay Keswani, Anay Mehrotra et al.
    "ReliabilityRAG: Effective and Provably Robust Defense for RAG-based
    Web-Search." **NeurIPS 2025** (poster; OpenReview forum ``D9JeNTs5Bu``).
    Paper: https://arxiv.org/abs/2509.23519
    Reference code: https://github.com/zeyushen-yo/ReliabilityRAG
    (``src/defense.py``, class ``MISBasedRRAG``).

The signal this module reads is **inter-passage answer contradiction**, not text
statistics and not retrieval geometry. For each retrieved passage the generator
is asked the query *from that passage alone*; an NLI model then tests every pair
of isolated answers for contradiction; contradicting pairs become edges of a
**conflict graph** over the passages. The largest mutually non-contradicting
subset — a **maximum independent set** (MIS) — is taken as the trustworthy
evidence, and everything outside it is flagged. The paper's guarantee is a
majority argument: if the corrupted documents are a strict minority of the
retrieved set, no independent set of corrupted documents can be maximum, so the
truth survives.

What was verified against the source, and where this file differs
-----------------------------------------------------------------
Read from ``src/defense.py`` and the arXiv HTML, correcting the brief this file
was written from:

* **NLI compares the per-passage answers, not the passages.** The reference
  builds ``premise = "The answer to the question: {q}\\nis {answer_i}."`` and the
  same string for ``j``, and reads the contradiction probability of that pair.
  :data:`NLI_PAIR_TMPL` reproduces the wording verbatim. Comparing raw passages
  was considered and rejected by the source; do not "fix" it.
* **Contradiction threshold beta = 0.5** in every experiment. Kept as the
  default (``beta``).
* **The NLI model is ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli``**
  (the reference hard-codes a local copy of exactly that checkpoint), *not*
  ``cross-encoder/nli-deberta-v3-large``. This module therefore defaults to the
  paper's checkpoint and accepts the cross-encoder one as ``nli_model_key``.
  The two disagree on **label order** — ``cross-encoder/nli-deberta-v3-large``
  is ``(contradiction, entailment, neutral)`` while the MoritzLaurer model is
  ``(entailment, neutral, contradiction)`` — so the contradiction column is
  resolved from ``config.id2label`` rather than hard-coded to index 2 as the
  reference does. Hard-coding it would silently read *entailment* probabilities
  under the cross-encoder checkpoint and inverting the whole detector.
* **k+1 generations is wrong; it is k.** The reference's extra generation is the
  *final answer* produced from the surviving documents — a RAG output, not part
  of the graph. A detector never needs it, so this module runs exactly ``k``
  isolated generations.
* **Answers that abstain are dropped from the graph.** The reference excludes
  any passage whose isolated answer contains ``"I don't know"`` both from the
  edge set and from the vertex set ``z``. That is right for evidence selection
  (an irrelevant passage is noise) but wrong for detection: the reference's
  selected set omits them, which as a detector would flag every off-topic
  passage as poison. The exclusion from the graph is kept — it is the paper's —
  but abstaining passages are scored ``nan``, **not** 0.0. ``nan`` means "no
  evidence either way"; 0.0 would mean "least suspicious passage in the set",
  and since PoisonedRAG passages are confident assertions that never abstain,
  scoring abstentions at the floor drops honest passages onto exactly the
  poison's score. See :meth:`ReliabilityRAGDetector.abstain_mask` and the
  vertex-thinning note below.
* **The abstention marker list is the reference's, one substring.** The
  reference tests exactly ``"I don't know"``. Earlier revisions of this file
  tested nine markers, including the bare token ``"unknown"``; every extra
  marker deletes one more honest passage from the vertex set and so *drives*
  the thinning failure mode below. :data:`ABSTAIN_MARKERS` is now the
  reference's single marker and is the default; the wider list survives as
  :data:`WIDE_ABSTAIN_MARKERS` for the ``abstain_markers`` ablation only.
* **The lexicographic tie-break is not used for scoring by default.** When
  several maximum independent sets exist the reference keeps
  ``min(sorted(S))``, i.e. it prefers low retrieval ranks. That choice is
  arbitrary and, at an exact 50/50 split, decides guilt by retrieval order
  alone. The default ``mode="exclusion"`` scores a passage by the *fraction of
  maximum independent sets that exclude it*, which is tie-break free;
  ``mode="binary"`` reproduces the reference rule exactly.
* The reference's ``err`` parameter (randomly flipping edges) is a robustness
  ablation for the paper's provable-guarantee section and is not implemented.
  Neither is the weighted sample-and-aggregate variant used for k=50, which
  trades exactness for scale; at our k=5 it would only add variance.

Abstention is a first-class output
----------------------------------
If no pair contradicts, the conflict graph is edgeless, the MIS is the whole
retrieved set, and **nothing is flagged** — a genuine "nothing anomalous here".
Every scoring mode returns zero on every *vertex* in that case by construction
(an edgeless graph has degree 0 everywhere, one maximum independent set, and it
contains every vertex), so the graded score does not smear a nonzero suspicion
over a clean set. Passages that abstained are not vertices and come back
``nan`` — "this passage produced no claim, so this method has nothing to say
about it" — which is a second, per-passage kind of abstention and must not be
read as innocence. :meth:`ReliabilityRAGDetector.is_clean` exposes the
query-level predicate, :meth:`ReliabilityRAGDetector.abstain_mask` the
per-passage one, and :meth:`ReliabilityRAGDetector.conflict_graph` the full
evidence. An ROC harness should drop the ``nan`` positions rather than impute
them; :meth:`ReliabilityRAGDetector.predict` treats them as not-flagged.

Expected failure modes (stated because they are results)
--------------------------------------------------------
* **Vertex-set thinning — the measured mechanism. It is *not* a majority
  inversion.** Earlier revisions of this docstring predicted a "majority
  inversion": the conflict graph as complete bipartite between ``N`` poison and
  ``k - N`` genuine passages, the MIS as simply the larger side, and therefore
  ``N <= 2`` at ``k = 5`` flagging the poison perfectly with the failure only
  arriving at ``N = 3``. **That model is falsified by our own benchmark.** The
  run that produced ROC 0.446 was ``N = 2``, ``k = 5`` — exactly 3 benign
  against 2 poison in every attacked query, poison a strict minority — where
  the majority model predicts a detection rate near 1.0. The measured detection
  rate was **0.119**.

  What actually happens is that the honest side is thinned *below* the poison
  bloc before any majority is counted, because the MIS is computed over the
  **non-abstaining** vertex set, not over ``k``. Two effects do it:

  1. *Deletion.* A benign passage whose isolated answer abstains is removed from
     the graph entirely. That is faithful to the reference and is kept here, but
     it only ever removes honest vertices: a PoisonedRAG passage is a confident
     assertion of the target answer and can never abstain. Retrieval at ``k = 5``
     routinely returns loosely-related NQ chunks, and a 1.5B generator answering
     from one of them alone says so.
  2. *Internal edges.* The benign passages that do survive contradict *each
     other*. NQ chunks legitimately differ in detail, date or granularity
     ("1969" vs "July 20, 1969", a list vs one of its entries), so edges appear
     *within* the honest side and 3 surviving benign passages rarely yield an
     independent set of size 3.

  The poison bloc suffers from neither. PoisonedRAG's ``N`` injected passages
  all assert the same wrong answer, so they are internally **edge-free by
  construction**, and none of them abstains. A 2-passage poison bloc is thus a
  clean independent set of size 2 competing against an honest side that has
  been cut to 2 or fewer usable, mutually-consistent vertices — and it wins the
  MIS, so the *truth* is what falls outside it.

  The stratified numbers say the same thing: detection rate **0.55 with 4 benign
  passages in the retrieved set, 0.119 with 3, 0.032 with 2**. Detection is
  monotone in the size of the honest side and collapses well before the poison
  reaches a ``k``-level majority — the signature of thinning, not of the
  majority argument. The paper's guarantee is not wrong; its precondition
  ("corrupted documents are a strict minority") is simply not the condition that
  holds at MIS time, and the paper never measures the gap because it evaluates
  answer accuracy, not per-passage detection. ``abstain_markers`` /
  :data:`WIDE_ABSTAIN_MARKERS` ablate effect (1) and ``beta`` ablates effect (2).
* **Clean false positives are not zero, and the paper does not measure them.**
  The paper reports robust accuracy under attack; it never reports how often the
  defence discards *honest* evidence on a clean corpus. NQ splits Wikipedia into
  overlapping chunks that legitimately differ in detail, date, or granularity
  ("1969" vs "July 1969", a list vs one of its entries), and a 1.5B generator
  answering from one chunk at a time produces answers an NLI model happily calls
  contradictory. Every such edge removes a real passage from the MIS. Measuring
  that rate on the clean corpus is the point of this port; :meth:`fit` records
  it without needing labels.
* **Two error sources compound.** The generator can misread a passage and the
  NLI model can misjudge a pair; the graph inherits both. Short keyword answers
  ("Minjee Lee") are also out of distribution for MNLI-style models trained on
  sentence pairs, which is why the reference wraps them in a carrier sentence —
  it mitigates the mismatch but does not remove it.
* **NLI is directional, the graph is not.** Only the ``(i, j)`` order is scored
  by default (as in the reference) and the edge is added symmetrically, so the
  graph depends on passage order. ``bidirectional=True`` scores both directions
  and takes the max, at twice the NLI cost.
* **Whole-passage attacks that agree with the corpus are invisible.** A poison
  that asserts a *true* fact to smuggle an instruction, or one that merely
  degrades ranking, creates no contradiction and is never flagged.

Cost
----
Per query: ``k`` short generations (Qwen3-1.7B, greedy, thinking off, 32 new
tokens, batched) plus ``k(k-1)/2`` NLI forward passes through a 435M-parameter
DeBERTa-v3-large, batched into one call at k=5. That is ~1-2 s per query on one
A6000 — acceptable for a per-query defence, and **hopeless as a corpus scan**:
the work is quadratic in the retrieved-set size and is defined only *relative to
a query*, so there is no way to score 20k standalone documents with it. Use it
on the retrieved set, not on the database.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = [
    "ABSTAIN_MARKERS",
    "ConflictGraph",
    "ISOLATED_ANSWER_TMPL",
    "NLI_PAIR_TMPL",
    "PAPER_NLI_MODEL",
    "ReliabilityRAGDetector",
    "WIDE_ABSTAIN_MARKERS",
    "maximum_independent_sets",
]

#: The checkpoint the paper and its reference implementation use.
PAPER_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

#: Isolated-answer prompt, following ``QA_PROMPT_TMPL`` in the reference's
#: ``src/prompt_template.py``: keyword-style answers plus an explicit abstention
#: option, so an off-topic passage says so instead of hallucinating a rival fact.
ISOLATED_ANSWER_TMPL = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query with only keywords.\n"
    'If there is no relevant information, just say "I don\'t know".\n'
    "Query: {query_str}\n"
    "Answer: "
)

#: Carrier sentence wrapped around each isolated answer before NLI. Verbatim
#: from ``MISBasedRRAG.query``; a bare keyword pair is far outside an MNLI
#: model's training distribution.
NLI_PAIR_TMPL = "The answer to the question: {query}\nis {answer}."

#: Substrings marking an isolated answer as an abstention — **the reference's
#: rule, one substring**, matched case-insensitively after apostrophe
#: normalisation. This is the default for ``abstain_markers``. Every marker
#: deletes a passage from the MIS vertex set, and because PoisonedRAG passages
#: never abstain, every deletion is an honest passage: a wider list directly
#: drives the vertex-thinning failure documented in the module docstring.
ABSTAIN_MARKERS: tuple[str, ...] = ("i don't know",)

#: The wider marker list this module used to apply unconditionally, kept so the
#: deviation is ablatable via ``abstain_markers=WIDE_ABSTAIN_MARKERS`` rather
#: than silently on. Qwen3 does phrase refusals more freely than the reference's
#: prompt anticipates, so this recovers genuine abstentions — at the cost of
#: thinning the honest side further. The bare token ``"unknown"`` is gone for
#: good: it matched real answers ("Unknown Pleasures", "unknown soldier").
WIDE_ABSTAIN_MARKERS: tuple[str, ...] = (
    "i don't know",
    "i do not know",
    "no relevant information",
    "no information",
    "not mentioned",
    "does not mention",
    "does not provide",
    "cannot be determined",
)

#: Largest vertex count for which all maximum independent sets are enumerated
#: exactly. Above it the detector degrades to a greedy MIS (see
#: :meth:`ReliabilityRAGDetector.conflict_graph`).
_DEFAULT_MAX_EXACT = 20

# The "ambiguous" characters are exactly the point: these are the apostrophe
# variants a generator may emit in "I don't know", all folded to ASCII before
# the abstention markers are matched.
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "´": "'"})  # noqa: RUF001


# --------------------------------------------------------------------------- #
# evidence object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class ConflictGraph:
    """The full evidence behind one query's decision.

    Attributes:
        query: The query the passages were retrieved for.
        answers: Isolated answer per passage, in passage order.
        abstained: ``True`` where the isolated answer matched the detector's
            ``abstain_markers`` (default :data:`ABSTAIN_MARKERS`) or was empty.
            Abstaining passages are not vertices of the graph, are never
            flagged, and are scored ``nan`` — "no evidence", not "innocent".
        contradiction: ``(k, k)`` symmetric matrix of contradiction
            probabilities; ``nan`` on the diagonal and on any pair involving an
            abstention (never scored, or scored and then discarded).
        edges: ``(k, k)`` symmetric boolean adjacency matrix.
        vertices: Passage indices participating in the graph, ascending.
        mis_size: Size of a maximum independent set over ``vertices``.
        mis_sets: Every maximum independent set, each a sorted tuple of passage
            indices, in ascending lexicographic order. Exactly one entry when
            ``exact`` is ``False``.
        selected: The reference's choice — the lexicographically smallest
            maximum independent set, which prefers better-ranked passages.
        exact: Whether ``mis_sets`` is the complete enumeration. ``False`` means
            a greedy fallback was used because the graph was too large.
    """

    query: str
    answers: tuple[str, ...]
    abstained: np.ndarray
    contradiction: np.ndarray
    edges: np.ndarray
    vertices: tuple[int, ...]
    mis_size: int
    mis_sets: tuple[tuple[int, ...], ...]
    selected: tuple[int, ...]
    exact: bool = True

    @property
    def n_passages(self) -> int:
        """Number of retrieved passages."""
        return len(self.answers)

    @property
    def n_edges(self) -> int:
        """Number of contradicting pairs."""
        return int(np.count_nonzero(np.triu(self.edges, k=1)))

    @property
    def is_clean(self) -> bool:
        """``True`` when nothing contradicts anything — the abstain verdict."""
        return self.n_edges == 0

    def degrees(self) -> np.ndarray:
        """Contradiction degree per passage: how many others it contradicts."""
        return self.edges.sum(axis=1).astype(np.float64)

    def exclusion_fraction(self) -> np.ndarray:
        """Fraction of maximum independent sets that leave each passage out.

        ``0.0`` for a passage in every maximum independent set, ``1.0`` for one
        in none of them, and an intermediate value when the maximum sets
        disagree about it. On an edgeless graph the single maximum independent
        set is the whole vertex set, so this is all-zeros — the abstain verdict
        survives the graded score.

        Non-vertices (abstaining passages) are ``0.0`` here because they are not
        accused by any maximum set. This is the *graph-level* statement; the
        detector overwrites those positions with ``nan`` before returning a
        score, since 0.0 at the detector's output would read as "most
        trustworthy passage in the set" rather than "no evidence". See
        :meth:`ReliabilityRAGDetector.score`.
        """
        out = np.zeros(self.n_passages, dtype=np.float64)
        if not self.mis_sets or not self.vertices:
            return out
        counts = np.zeros(self.n_passages, dtype=np.float64)
        for mis in self.mis_sets:
            for idx in mis:
                counts[idx] += 1.0
        total = float(len(self.mis_sets))
        for idx in self.vertices:
            out[idx] = 1.0 - counts[idx] / total
        return out

    def excluded_mask(self) -> np.ndarray:
        """``True`` for vertices outside :attr:`selected` — the reference rule."""
        out = np.zeros(self.n_passages, dtype=bool)
        chosen = set(self.selected)
        for idx in self.vertices:
            if idx not in chosen:
                out[idx] = True
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready summary for an evidence log."""
        return {
            "query": self.query,
            "answers": list(self.answers),
            "abstained": [bool(x) for x in self.abstained],
            "n_edges": self.n_edges,
            "edges": [
                [int(i), int(j), float(self.contradiction[i, j])]
                for i, j in zip(*np.nonzero(np.triu(self.edges, k=1)), strict=False)
            ],
            "vertices": list(self.vertices),
            "mis_size": self.mis_size,
            "n_mis": len(self.mis_sets),
            "selected": list(self.selected),
            "exact": self.exact,
            "is_clean": self.is_clean,
        }


# --------------------------------------------------------------------------- #
# exact maximum independent set enumeration
# --------------------------------------------------------------------------- #
def maximum_independent_sets(
    adjacency: Sequence[int], n: int, *, limit: int | None = None
) -> tuple[int, list[int]]:
    """Enumerate **all** maximum independent sets of a small graph, exactly.

    Branch-and-bound over vertices in index order: at each vertex either take it
    (and forbid its neighbours) or skip it, pruning a branch as soon as the
    vertices still available cannot reach the best size found so far. MIS is
    NP-hard in general, but ``n`` here is the retrieved-set size (5-10), where
    this returns in microseconds; the reference instead materialises all ``2^n``
    subsets and checks each in ``O(n^2)``, which is ~``2^n n^2`` work for the
    same answer.

    Args:
        adjacency: ``adjacency[i]`` is a bitmask of ``i``'s neighbours. Must be
            symmetric; self-loops are ignored.
        n: Number of vertices, ``0 <= n <= 63`` for comfort (the caller gates on
            a much smaller bound).
        limit: Stop collecting after this many maximum sets are found. The
            reported size stays correct; the enumeration becomes partial, which
            only matters for :meth:`ConflictGraph.exclusion_fraction`.

    Returns:
        ``(size, masks)`` — the maximum independent set size and every maximum
        independent set as a vertex bitmask, ascending. ``(0, [0])`` when
        ``n == 0``.
    """
    if n <= 0:
        return 0, [0]

    full = (1 << n) - 1
    best_size = 0
    best: list[int] = []

    def recurse(i: int, chosen: int, size: int, allowed: int) -> None:
        nonlocal best_size, best
        if size + (allowed >> i).bit_count() < best_size:
            return
        if i == n:
            if size > best_size:
                best_size, best = size, [chosen]
            elif size == best_size and (limit is None or len(best) < limit):
                best.append(chosen)
            return
        bit = 1 << i
        if allowed & bit:
            recurse(i + 1, chosen | bit, size + 1, allowed & ~adjacency[i] & ~bit)
        recurse(i + 1, chosen, size, allowed & ~bit)

    recurse(0, 0, 0, full)
    best.sort()
    return best_size, best


def _greedy_independent_set(adjacency: Sequence[int], n: int) -> int:
    """Minimum-degree greedy independent set — the fallback for large graphs.

    Repeatedly takes the remaining vertex of smallest degree in the induced
    subgraph and deletes its neighbourhood. Not maximum in general, so the
    detector reports ``exact=False`` when this is used.
    """
    remaining = (1 << n) - 1
    chosen = 0
    while remaining:
        best_v, best_deg = -1, math.inf
        rem = remaining
        while rem:
            bit = rem & -rem
            v = bit.bit_length() - 1
            deg = (adjacency[v] & remaining).bit_count()
            if deg < best_deg:
                best_v, best_deg = v, deg
            rem ^= bit
        chosen |= 1 << best_v
        remaining &= ~adjacency[best_v] & ~(1 << best_v)
    return chosen


def _mask_to_indices(mask: int, vertices: Sequence[int]) -> tuple[int, ...]:
    """Map a subgraph bitmask back to ascending original passage indices."""
    return tuple(vertices[i] for i in range(len(vertices)) if mask >> i & 1)


# --------------------------------------------------------------------------- #
# the detector
# --------------------------------------------------------------------------- #
@register_detector("reliability", category="rag-defense")
class ReliabilityRAGDetector(Detector):
    """Flag retrieved passages that fall outside the maximum independent set.

    ``samples`` is ``(query, passages)``; :meth:`score` returns one float per
    passage, higher = more suspicious. No labels are used anywhere, and
    ``store`` / ``encoder`` are accepted only so the constructor matches the
    other ``rag-defense`` detectors.

    Scoring modes
    -------------
    ``mode="exclusion"`` (default)
        The fraction of *all* maximum independent sets that exclude the passage,
        in ``[0, 1]``. This is the graded generalisation of the paper's
        decision: 1.0 means every maximum independent set rejects the passage,
        0.0 means none does, and an intermediate value means the evidence is
        genuinely ambiguous. It is the default over a raw binary because (a) a
        binary score gives a two-point ROC curve, which understates AUC and
        makes any top-*m* budget arbitrary, and (b) it removes the reference's
        lexicographic tie-break, which at an exact 50/50 split decides which
        half is "poison" purely by retrieval rank — an arbitrary source of false
        positives. No degree refinement is added on top: the score is already
        continuous, and an additive term would push the exactly-tied 0.5 case
        over the default cut, turning the honest "no majority evidence" verdict
        into "flag everything".
    ``mode="binary"``
        ``1.0`` outside the lexicographically smallest maximum independent set,
        ``0.0`` inside — the reference rule reproduced exactly, tie-break
        included. Use it to report the paper's own operating point. Because the
        raw rule gives a two-point ROC, ``degree_tiebreak`` (on by default) adds
        ``0.5 * deg(i) / (k - 1)``, a strictly sub-unit refinement that orders
        passages *within* each of the two levels and can never move one across;
        thresholding at ``1.0`` still recovers the paper's decision exactly.
    ``mode="degree"``
        ``deg(i) / (k - 1)``: how many other passages this one contradicts,
        normalised. The cheapest mode (the MIS is not needed) and the smoothest,
        but it is a *popularity* statistic with no notion of a consistent
        coalition, so it fails the same way the MIS does: an internally
        edge-free poison bloc scores low on degree too, while benign passages
        that merely differ in detail rack up edges against each other (see the
        vertex-thinning note in the module docstring). It gets no default
        threshold, because "at least
        one contradiction" flags both sides of every disagreement and anything
        else would be a number invented here; sweep it.

    On an edgeless graph every degree is 0, every exclusion fraction is 0 and
    nothing is outside the MIS, so all three modes return zero on every vertex
    and the abstain verdict survives the graded score.

    Abstaining passages score ``nan`` in every mode
    -----------------------------------------------
    A passage whose isolated answer was an abstention is not a vertex, so no
    mode has any evidence about it. It is scored ``abstain_score``, which
    defaults to ``nan``. ``nan`` is not "innocent": it is "this method did not
    look at this passage". Use :meth:`abstain_mask` to drop those positions from
    an ROC rather than imputing them — imputing 0.0 (the pre-fix behaviour) puts
    honest, off-topic passages on the same floor as the poison, which produced
    81.5% tied scores in the ``N = 2, k = 5`` benchmark, and imputing 1.0 is
    worse still because abstention concentrates on clean passages.
    :meth:`predict` treats ``nan`` as not-flagged.

    Args:
        store: Unused; accepted for a uniform constructor signature.
        encoder: Unused; accepted for a uniform constructor signature.
        model: A loaded causal LM (Qwen3-1.7B). ``None`` loads ``model_key`` on
            first use. **Never assumed to have a mask token** — it is a causal
            LM and is used only for generation; the NLI model brings its own
            tokenizer.
        tokenizer: The causal LM's tokenizer; supply it with ``model``.
        threshold: Decision threshold for :meth:`predict`. ``None`` installs the
            mode's natural cut — ``1.0`` for ``binary`` (exactly "outside the
            MIS") and just above ``0.5`` for ``exclusion`` ("a strict majority
            of maximum independent sets rejects it", so an exact tie abstains
            rather than flagging everything). ``degree`` gets ``None`` and
            :meth:`predict` will refuse until one is supplied. These are the
            method's own decision rule, not values tuned on our data; sweep the
            score with :func:`aiforensics.metrics.threshold_sweep` for a curve.
        mode: ``"exclusion"`` | ``"binary"`` | ``"degree"``.
        degree_tiebreak: Add the sub-unit degree refinement in ``binary`` mode
            (see above). Ignored by the other two modes, which are already
            continuous.
        beta: Contradiction probability above which an edge is added. The
            paper's 0.5 in all experiments.
        bidirectional: Score both NLI directions and take the max. Off by
            default, matching the reference; doubles NLI cost.
        nli_model_key: NLI checkpoint. Defaults to the paper's
            :data:`PAPER_NLI_MODEL`; ``"cross-encoder/nli-deberta-v3-large"`` is
            the supported alternative. The contradiction column is resolved from
            the checkpoint's own ``id2label``, so the two label orders are both
            handled.
        model_key: Causal-LM registry key used when ``model`` is ``None``.
        max_new_tokens: Generation budget per isolated answer.
        gen_batch_size: Prompts per generation forward pass.
        nli_batch_size: Pairs per NLI forward pass.
        max_answer_chars: Isolated answers are truncated to this many characters
            before NLI, so a runaway generation cannot dominate the pair.
        max_exact: Vertex count above which the exact enumeration is replaced by
            a greedy independent set (and ``ConflictGraph.exact`` is ``False``).
        nli_device: Explicit device for the NLI model; ``None`` uses
            :func:`aiforensics.device.get_device`.
        abstain_markers: Substrings that mark an isolated answer as an
            abstention, matched case-insensitively after apostrophe
            normalisation. Defaults to :data:`ABSTAIN_MARKERS`, i.e. the
            reference's single ``"I don't know"``. Pass
            :data:`WIDE_ABSTAIN_MARKERS` to ablate the wider list this module
            used to apply unconditionally. An empty generation always counts as
            an abstention regardless of this list.
        abstain_score: Score returned for abstaining passages. Defaults to
            ``nan`` ("no evidence"). Pass ``0.0`` to restore the pre-fix
            behaviour for an ablation — see the class note above for why that
            behaviour was wrong.
        first_line_only: Reduce each isolated answer to its first non-empty line
            before NLI. ``False`` (the default, and the reference's behaviour)
            feeds the whole cleaned response, because truncating at the first
            newline can turn agreement into contradiction — "1969\\n(July 20)"
            against "July 20, 1969" is the same fact, but "1969" alone is not.
        **config: Recorded verbatim by :class:`~.base.Detector`.

    Raises:
        ValueError: On an unknown ``mode`` or a ``beta`` outside ``[0, 1]``.
    """

    name = "reliability"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False  # the paper's rule is calibration-free; fit() diagnoses

    _MODES = ("exclusion", "binary", "degree")

    def __init__(
        self,
        *,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        mode: str = "exclusion",
        degree_tiebreak: bool = True,
        beta: float = 0.5,
        bidirectional: bool = False,
        nli_model_key: str = PAPER_NLI_MODEL,
        model_key: str = "qwen3-1.7b",
        max_new_tokens: int = 32,
        gen_batch_size: int = 8,
        nli_batch_size: int = 16,
        max_answer_chars: int = 200,
        max_exact: int = _DEFAULT_MAX_EXACT,
        nli_device: Any = None,
        abstain_markers: Sequence[str] | None = None,
        abstain_score: float = float("nan"),
        first_line_only: bool = False,
        **config: Any,
    ) -> None:
        if mode not in self._MODES:
            raise ValueError(f"mode must be one of {self._MODES}, got {mode!r}")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError(f"beta is a probability threshold in [0, 1], got {beta!r}")
        # Normalised once here so _is_abstention does no per-call work and so the
        # recorded config shows exactly what was matched.
        markers = tuple(
            str(m).translate(_APOSTROPHES).lower()
            for m in (ABSTAIN_MARKERS if abstain_markers is None else abstain_markers)
        )
        super().__init__(
            threshold=_default_threshold(mode) if threshold is None else threshold,
            mode=mode, degree_tiebreak=degree_tiebreak, beta=float(beta),
            bidirectional=bidirectional, nli_model_key=nli_model_key,
            model_key=model_key, max_new_tokens=max_new_tokens,
            gen_batch_size=gen_batch_size, nli_batch_size=nli_batch_size,
            max_answer_chars=max_answer_chars, max_exact=max_exact,
            abstain_markers=list(markers), abstain_score=float(abstain_score),
            first_line_only=bool(first_line_only), **config,
        )
        self.mode = mode
        self.degree_tiebreak = bool(degree_tiebreak)
        self.beta = float(beta)
        self.bidirectional = bool(bidirectional)
        self.nli_model_key = nli_model_key
        self.model_key = model_key
        self.max_new_tokens = int(max_new_tokens)
        self.gen_batch_size = int(gen_batch_size)
        self.nli_batch_size = int(nli_batch_size)
        self.max_answer_chars = int(max_answer_chars)
        self.max_exact = int(max_exact)
        self.nli_device = nli_device
        self.abstain_markers = markers
        self.abstain_score = float(abstain_score)
        self.first_line_only = bool(first_line_only)

        self._model = model
        self._tokenizer = tokenizer
        self._nli: Any = None
        self._nli_tokenizer: Any = None
        self._contradiction_index: int | None = None
        # One-entry memo so is_clean() -> conflict_graph() -> score() on the same
        # sample costs one pipeline run, not three.
        self._cache_key: tuple[str, tuple[str, ...]] | None = None
        self._cache: ConflictGraph | None = None

        #: Label-free clean diagnostics, filled by :meth:`fit`.
        self.clean_edge_rate_: float | None = None
        self.clean_flag_rate_: float | None = None
        self.clean_query_alarm_rate_: float | None = None
        self.clean_abstain_rate_: float | None = None
        self.clean_contradiction_: np.ndarray | None = None

    # -- lazy model access -------------------------------------------------- #
    def _ensure_llm(self) -> tuple[Any, Any]:
        """Return ``(model, tokenizer)`` for the generator, loading on first use."""
        if self._model is None or self._tokenizer is None:
            from ..models.llm import load_llm

            logger.info("reliability: loading generator %s", self.model_key)
            self._model, self._tokenizer = load_llm(self.model_key)
        return self._model, self._tokenizer

    def _ensure_nli(self) -> tuple[Any, Any, int]:
        """Return ``(model, tokenizer, contradiction_index)``, loading on first use.

        Raises:
            ImportError: If ``transformers`` is missing, naming the package.
            RuntimeError: If the contradiction column could not be resolved from
                the checkpoint's labels.
        """
        if self._nli is None or self._nli_tokenizer is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment guard
                raise ImportError(
                    "the reliability detector needs an NLI model from transformers; "
                    "install it with `pip install transformers`"
                ) from exc
            from ..device import get_device

            logger.info("reliability: loading NLI model %s", self.nli_model_key)
            self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_key)
            nli = AutoModelForSequenceClassification.from_pretrained(self.nli_model_key)
            device = get_device() if self.nli_device is None else self.nli_device
            self._nli = nli.to(device).eval()
            self._contradiction_index = _resolve_contradiction_index(self._nli.config)
        if self._contradiction_index is None:
            # Never default this. Under PAPER_NLI_MODEL, id2label is
            # {0: entailment, 1: neutral, 2: contradiction}, so falling back to
            # index 0 would read *entailment* probabilities and invert the whole
            # detector — edges where answers agree, MIS over the disagreeing
            # passages — with no error and a plausible-looking score.
            raise RuntimeError(
                f"could not resolve the contradiction column for NLI checkpoint "
                f"{self.nli_model_key!r}; its id2label does not identify one. Reading an "
                f"arbitrary column would silently invert the detector. Point "
                f"nli_model_key at a checkpoint whose config labels the classes, or fix "
                f"the checkpoint's id2label."
            )
        return self._nli, self._nli_tokenizer, int(self._contradiction_index)

    # -- stage 1: isolated answers ------------------------------------------ #
    def isolated_answers(self, query: str, passages: Sequence[str]) -> list[str]:
        """Answer ``query`` from each passage alone.

        Greedy, thinking disabled, ``max_new_tokens`` short — the answer is meant
        to be a keyword span, and a long generation both costs time and gives the
        NLI model more surface to trip over.

        Args:
            query: The retrieval query.
            passages: Retrieved passage texts.

        Returns:
            One cleaned answer string per passage, in input order.
        """
        from ..models.llm import generate

        model, tok = self._ensure_llm()
        prompts = [
            ISOLATED_ANSWER_TMPL.format(context_str=str(p).strip(), query_str=str(query))
            for p in passages
        ]
        raw = generate(
            model, tok, prompts,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.gen_batch_size,
            chat=True,
            enable_thinking=False,
            strip_thinking=True,
            do_sample=False,
        )
        return [
            _clean_answer(text, self.max_answer_chars, first_line_only=self.first_line_only)
            for text in raw
        ]

    # -- stage 2: pairwise contradiction ------------------------------------ #
    def _contradiction_matrix(
        self, query: str, answers: Sequence[str], abstained: np.ndarray
    ) -> np.ndarray:
        """Symmetric ``(k, k)`` matrix of pairwise contradiction probabilities.

        Pairs involving an abstention are never scored and stay ``nan``, matching
        the reference's guard: an "I don't know" is the absence of a claim, and
        an MNLI model asked to compare it with a real answer returns noise.
        """
        import torch

        k = len(answers)
        out = np.full((k, k), np.nan, dtype=np.float64)
        pairs = [
            (i, j)
            for i in range(k)
            for j in range(i + 1, k)
            if not abstained[i] and not abstained[j]
        ]
        if not pairs:
            return out

        nli, tok, contra = self._ensure_nli()
        premises = [NLI_PAIR_TMPL.format(query=query, answer=answers[i]) for i, _ in pairs]
        hypotheses = [NLI_PAIR_TMPL.format(query=query, answer=answers[j]) for _, j in pairs]
        if self.bidirectional:
            premises, hypotheses = premises + hypotheses, hypotheses + premises

        probs: list[float] = []
        for start in range(0, len(premises), self.nli_batch_size):
            batch = tok(
                premises[start : start + self.nli_batch_size],
                hypotheses[start : start + self.nli_batch_size],
                return_tensors="pt", truncation=True, padding=True,
            )
            batch = {key: value.to(nli.device) for key, value in batch.items()}
            with torch.no_grad():
                logits = nli(**batch).logits
            probs.extend(torch.softmax(logits, dim=1)[:, contra].tolist())

        n_pairs = len(pairs)
        for idx, (i, j) in enumerate(pairs):
            value = probs[idx]
            if self.bidirectional:
                value = max(value, probs[idx + n_pairs])
            out[i, j] = out[j, i] = float(value)
        return out

    # -- stage 3: the graph and its maximum independent sets ---------------- #
    def conflict_graph(self, query: str, passages: Sequence[str]) -> ConflictGraph:
        """Build the conflict graph for one retrieved set.

        Runs the whole pipeline — isolated answers, pairwise NLI, maximum
        independent set enumeration — and returns the evidence. The result is
        memoised for the most recent ``(query, passages)``, so calling this and
        then :meth:`score` on the same sample costs one pipeline run.

        Args:
            query: The retrieval query.
            passages: Retrieved passage texts.

        Returns:
            A :class:`ConflictGraph`. ``graph.is_clean`` is the abstain verdict.
        """
        texts = [str(p) for p in passages]
        key = (str(query), tuple(texts))
        if self._cache_key == key and self._cache is not None:
            return self._cache

        k = len(texts)
        if k == 0:
            graph = ConflictGraph(
                query=str(query), answers=(), abstained=np.zeros(0, dtype=bool),
                contradiction=np.zeros((0, 0)), edges=np.zeros((0, 0), dtype=bool),
                vertices=(), mis_size=0, mis_sets=((),), selected=(),
            )
            self._cache_key, self._cache = key, graph
            return graph

        answers = self.isolated_answers(str(query), texts)
        abstained = np.asarray(
            [_is_abstention(a, self.abstain_markers) for a in answers], dtype=bool
        )
        contradiction = self._contradiction_matrix(str(query), answers, abstained)

        edges = np.zeros((k, k), dtype=bool)
        finite = np.isfinite(contradiction)
        edges[finite] = contradiction[finite] >= self.beta
        np.fill_diagonal(edges, False)

        vertices = tuple(int(i) for i in range(k) if not abstained[i])
        pos = {v: p for p, v in enumerate(vertices)}
        adjacency = [0] * len(vertices)
        for i in vertices:
            for j in vertices:
                if i != j and edges[i, j]:
                    adjacency[pos[i]] |= 1 << pos[j]

        exact = len(vertices) <= self.max_exact
        if exact:
            size, masks = maximum_independent_sets(adjacency, len(vertices))
        else:
            logger.warning(
                "reliability: %d vertices exceeds max_exact=%d; falling back to a greedy "
                "independent set (exclusion fractions degenerate to binary)",
                len(vertices), self.max_exact,
            )
            mask = _greedy_independent_set(adjacency, len(vertices))
            size, masks = mask.bit_count(), [mask]

        mis_sets = tuple(_mask_to_indices(m, vertices) for m in masks)
        # Reference tie-break: the lexicographically smallest maximum set, which
        # keeps the better-ranked passages. Arbitrary, and only used by
        # mode="binary" — see the module docstring.
        selected = min(mis_sets) if mis_sets else ()

        graph = ConflictGraph(
            query=str(query), answers=tuple(answers), abstained=abstained,
            contradiction=contradiction, edges=edges, vertices=vertices,
            mis_size=int(size), mis_sets=mis_sets, selected=selected, exact=exact,
        )
        self._cache_key, self._cache = key, graph
        return graph

    def is_clean(self, query: str, passages: Sequence[str]) -> bool:
        """``True`` when no pair of passages contradicts — "nothing anomalous here".

        This is the detector's abstain output and it is a genuine negative, not a
        thresholding artefact: with no edges the maximum independent set is the
        entire retrieved set, so no scoring mode can flag anything. Read it
        directly rather than inferring it from ``score(...).max() == 0``.

        Args:
            query: The retrieval query.
            passages: Retrieved passage texts.

        Returns:
            Whether the conflict graph is edgeless.
        """
        return self.conflict_graph(query, passages).is_clean

    def abstain_mask(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """``True`` for each passage this method has **no evidence** about.

        A passage whose isolated answer was an abstention (or empty) is not a
        vertex of the conflict graph, so no maximum independent set can either
        accuse or clear it. :meth:`score` returns ``abstain_score`` (``nan`` by
        default) at exactly these positions, and a harness computing an ROC
        should drop them rather than impute a value: scoring them 0.0 marks
        honest off-topic passages as the most trustworthy in the set, while
        scoring them 1.0 accuses them, and abstention concentrates on clean
        passages either way (a PoisonedRAG passage asserts its answer
        confidently and never abstains).

        Uses the same memo as :meth:`conflict_graph`, so calling both on one
        sample costs a single pipeline run.

        Args:
            query: The retrieval query.
            passages: Retrieved passage texts.

        Returns:
            ``(len(passages),)`` boolean array, ``True`` = abstained.
        """
        return np.asarray(self.conflict_graph(query, passages).abstained, dtype=bool)

    # -- scoring ------------------------------------------------------------ #
    def score(self, samples: Any) -> np.ndarray:
        """Suspicion score per passage. Higher = more suspicious.

        Args:
            samples: ``(query, passages)``.

        Returns:
            ``(len(passages),)`` float array. Zero on every vertex exactly when
            the conflict graph is edgeless (see :meth:`is_clean`), and
            ``abstain_score`` — ``nan`` by default, meaning *no evidence*, not
            *innocent* — at every abstaining passage (see :meth:`abstain_mask`).
            Semantics per ``mode`` are documented on the class.
        """
        query, passages = samples
        graph = self.conflict_graph(query, passages)
        if graph.n_passages == 0:
            return np.zeros(0, dtype=np.float64)

        degree = graph.degrees()
        denom = max(graph.n_passages - 1, 1)
        degree_norm = degree / denom

        if self.mode == "degree":
            out = degree_norm
        elif self.mode == "exclusion":
            # Already continuous; deliberately left un-refined, see the class
            # docstring — an additive tie-break would push the exactly-tied 0.5
            # case over the abstain cut.
            out = graph.exclusion_fraction()
        else:
            out = graph.excluded_mask().astype(np.float64)
            if self.degree_tiebreak:
                out = out + 0.5 * degree_norm

        # Abstaining passages are not vertices: every mode would otherwise hand
        # them the floor value (degree 0, in every MIS, inside `selected`), i.e.
        # report "produced no claim" as "most trustworthy passage here". Since a
        # PoisonedRAG passage is a confident assertion and never abstains, that
        # floor is populated exclusively by clean passages and collides with the
        # poison's own score — 81.5% of scores were tied in the N=2, k=5 run.
        # nan says "not assessed"; the harness excludes it via abstain_mask().
        out = np.asarray(out, dtype=np.float64).copy()
        if graph.abstained.any():
            out[np.asarray(graph.abstained, dtype=bool)] = self.abstain_score
        return out

    def predict(self, samples: Any, threshold: float | None = None) -> np.ndarray:
        """Flag passages by thresholding :meth:`score`, with ``nan`` = not flagged.

        Same contract as :meth:`~.base.Detector.predict`, made explicit here
        because :meth:`score` can return ``nan``: "no evidence about this
        passage" is not an accusation, so an abstaining passage is never
        flagged. It is not an exoneration either — read :meth:`abstain_mask`
        alongside this, since a ``False`` here has two different meanings.

        Args:
            samples: ``(query, passages)``.
            threshold: Decision threshold; falls back to ``self.threshold``.

        Returns:
            ``(len(passages),)`` boolean array, ``True`` = flagged.

        Raises:
            ValueError: If no threshold is available (``mode="degree"`` ships
                without one; sweep it instead).
        """
        self._check_fitted()
        thr = threshold if threshold is not None else self.threshold
        if thr is None:
            raise ValueError(
                f"{type(self).__name__} has no decision threshold for mode={self.mode!r}. "
                f"Pass threshold=…, set detector.threshold, or derive one with "
                f"aiforensics.metrics.best_threshold(y_true, scores)."
            )
        scores = np.asarray(self.score(samples), dtype=np.float64).reshape(-1)
        # np.nan >= thr is already False; written out so the choice is deliberate
        # rather than an accident of IEEE comparison semantics.
        return np.where(np.isnan(scores), False, scores >= float(thr))

    # -- optional clean-corpus diagnostics ---------------------------------- #
    def fit(self, clean_data: Any, **kwargs: Any) -> ReliabilityRAGDetector:
        """Measure the method's behaviour on data believed clean. **No labels.**

        Nothing here is required for :meth:`score` — the paper's rule is
        calibration-free, and ``requires_fit`` is ``False``. What ``fit`` buys is
        the number the paper never reports: how often ReliabilityRAG discards
        honest evidence when there is no attack. It records, over
        ``clean_data``:

        * ``clean_edge_rate_`` — fraction of scored pairs judged contradictory;
        * ``clean_flag_rate_`` — fraction of passages left out of the maximum
          independent set (the clean false-positive rate under ``mode="binary"``);
        * ``clean_query_alarm_rate_`` — fraction of queries with at least one
          edge, i.e. how often the detector fails to abstain on clean data;
        * ``clean_abstain_rate_`` — fraction of passages whose isolated answer
          was an abstention, which bounds how much of the retrieved set the
          method can even reason about;
        * ``clean_contradiction_`` — every scored contradiction probability, so
          ``beta`` can be re-chosen with :meth:`suggest_beta` instead of
          inherited.

        Cost is the full pipeline over every calibration query — ``k``
        generations and ``k(k-1)/2`` NLI passes each — so this is minutes, not
        seconds, on a few hundred queries.

        Args:
            clean_data: Sequence of ``(query, passages)`` believed clean.
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            ``self``.
        """
        probs: list[float] = []
        n_pairs = n_edges = n_passages = n_flagged = n_abstain = 0
        n_queries = n_alarm = 0
        for query, passages in clean_data:
            graph = self.conflict_graph(query, passages)
            n_queries += 1
            n_alarm += int(not graph.is_clean)
            n_passages += graph.n_passages
            n_abstain += int(graph.abstained.sum())
            n_flagged += int(graph.excluded_mask().sum())
            upper = np.triu_indices(graph.n_passages, k=1)
            values = graph.contradiction[upper]
            values = values[np.isfinite(values)]
            probs.extend(values.tolist())
            n_pairs += int(values.size)
            n_edges += int((values >= self.beta).sum())

        self.clean_contradiction_ = np.asarray(probs, dtype=np.float64)
        self.clean_edge_rate_ = float(n_edges / n_pairs) if n_pairs else None
        self.clean_flag_rate_ = float(n_flagged / n_passages) if n_passages else None
        self.clean_abstain_rate_ = float(n_abstain / n_passages) if n_passages else None
        self.clean_query_alarm_rate_ = float(n_alarm / n_queries) if n_queries else None
        logger.info(
            "reliability: clean flag rate %.4f over %d passages (%d/%d queries alarmed)",
            self.clean_flag_rate_ or 0.0, n_passages, n_alarm, n_queries,
        )
        self._fitted = True
        return self

    def suggest_beta(self, target_edge_rate: float) -> float:
        """Contradiction threshold that would yield ``target_edge_rate`` on clean data.

        Args:
            target_edge_rate: Desired fraction of clean pairs to call
                contradictory, in ``(0, 1]``.

        Returns:
            The corresponding upper quantile of the clean contradiction
            probabilities. Purely advisory — the paper's ``beta = 0.5`` is the
            reported configuration, and swapping it for a data-derived cut makes
            the method calibration-dependent, which must be said in the report.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
            ValueError: If ``target_edge_rate`` is outside ``(0, 1]``.
        """
        if self.clean_contradiction_ is None or self.clean_contradiction_.size == 0:
            raise RuntimeError(
                "suggest_beta needs clean contradiction probabilities; call "
                "detector.fit(clean_data) first"
            )
        if not 0.0 < target_edge_rate <= 1.0:
            raise ValueError(f"target_edge_rate must be in (0, 1], got {target_edge_rate!r}")
        return float(np.quantile(self.clean_contradiction_, 1.0 - target_edge_rate))

    def describe(self) -> dict:
        """Detector identity, configuration and any clean diagnostics."""
        base = super().describe()
        base["nli_model"] = self.nli_model_key
        base["contradiction_index"] = self._contradiction_index
        base["clean_edge_rate"] = self.clean_edge_rate_
        base["clean_flag_rate"] = self.clean_flag_rate_
        base["clean_query_alarm_rate"] = self.clean_query_alarm_rate_
        base["clean_abstain_rate"] = self.clean_abstain_rate_
        return base


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _default_threshold(mode: str) -> float | None:
    """The mode's natural decision cut — the method's rule, not a tuned value.

    ``"degree"`` has none: a cut at "one contradiction" flags both sides of every
    disagreeing pair, and any other cut is a number invented here rather than
    taken from the method, so the caller must sweep it.
    """
    if mode == "binary":
        return 1.0
    if mode == "degree":
        return None
    # exclusion: strictly more than half the maximum independent sets must
    # reject the passage, so a perfect 50/50 split abstains instead of flagging
    # every passage in the retrieved set.
    return float(np.nextafter(0.5, 1.0))


def _clean_answer(text: str, max_chars: int, *, first_line_only: bool = False) -> str:
    """Reduce a generation to the answer span the NLI pair expects.

    Drops a leading ``Answer:`` echo, folds all whitespace (newlines included)
    into single spaces, and truncates to ``max_chars``.

    The whole response is kept by default, as in the reference. Keeping only the
    first line — the previous behaviour here, still available via
    ``first_line_only`` — silently deletes qualifying detail that lives on the
    following lines, and an NLI model comparing the remaining fragment against a
    complete answer reads the *same* fact as a contradiction ("1969" from
    "1969\\n(July 20)" against "July 20, 1969"). Those spurious edges land
    between benign passages, which is precisely the thinning documented in the
    module docstring; ``max_chars`` already bounds a runaway generation.

    Args:
        text: Raw generation for one passage.
        max_chars: Truncation budget, in characters.
        first_line_only: Keep only the first non-empty line (pre-fix behaviour).

    Returns:
        The cleaned answer string.
    """
    cleaned = str(text).strip()
    if first_line_only:
        for line in cleaned.splitlines():
            line = line.strip()
            if line:
                cleaned = line
                break
        else:
            cleaned = ""
    cleaned = re.sub(r"^(answer|final answer)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    # A multi-line answer inside the single-sentence NLI carrier is out of
    # distribution for MNLI; flattening keeps every token but restores the shape.
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:max_chars].strip()


def _is_abstention(answer: str, markers: Sequence[str] = ABSTAIN_MARKERS) -> bool:
    """Whether an isolated answer declines to make a claim.

    Args:
        answer: One cleaned isolated answer.
        markers: Lower-cased, apostrophe-normalised substrings to match.
            Defaults to the reference's :data:`ABSTAIN_MARKERS`.

    Returns:
        ``True`` for an empty answer or one containing any marker.
    """
    if not answer.strip():
        return True
    lowered = answer.translate(_APOSTROPHES).lower()
    return any(marker in lowered for marker in markers)


def _resolve_contradiction_index(config: Any) -> int | None:
    """Column of the contradiction logit, read from the checkpoint's own labels.

    The two supported checkpoints disagree:
    ``cross-encoder/nli-deberta-v3-large`` is
    ``(contradiction, entailment, neutral)`` and
    ``MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`` is
    ``(entailment, neutral, contradiction)``. The reference hard-codes index 2,
    which is correct only for the latter.

    There is deliberately **no fallback guess**. Picking the wrong column does
    not degrade the detector, it inverts it: entailment probabilities would put
    edges between passages that *agree*, and the maximum independent set would
    then collect the mutually-contradicting passages. A wrong-but-plausible
    score is worse here than a refusal, so an unresolvable head returns ``None``
    and :meth:`ReliabilityRAGDetector._ensure_nli` raises.

    Args:
        config: A ``PretrainedConfig`` with ``id2label``.

    Returns:
        The index whose label starts with ``"contradiction"``, or ``None`` if no
        label does.
    """
    id2label = dict(getattr(config, "id2label", {}) or {})
    for idx, label in id2label.items():
        if str(label).strip().lower().startswith("contradiction"):
            return int(idx)
    logger.error(
        "reliability: no 'contradiction' label in id2label=%r; refusing to guess a "
        "column, since the wrong one inverts the detector rather than weakening it.",
        id2label,
    )
    return None
