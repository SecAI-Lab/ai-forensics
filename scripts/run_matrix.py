"""The attack x detector **cross-product** benchmark: every published attack,
every published detector, one table.

Until now every harness in ``scripts/`` imported ``craft_blackbox`` /
``craft_whitebox`` directly, so PoisonedRAG was the only attack any detector had
ever been scored against. This script closes that gap by going through
:func:`aiforensics.attacks.registry.apply_attack`, which makes GARAG, jamming and
Phantom addressable by key and — more importantly — hands back the ground truth
*that belongs to each attack*.

What one cell means
-------------------
A cell is one ``(attack, variant, N, detector)`` combination. For it we retrieve
the top-``k`` passages for every evaluation query, ask the detector to score each
retrieved passage, pool those scores across queries, and compare them against the
attack's own ground truth. Reported per cell:

    det.acc   detection accuracy = #detected poisoned / #poisoned  (the project's
              headline metric; identical to recall on the poisoned class)
    prec      precision — of everything flagged, how much really was poison
    F1        harmonic mean of the two
    FPR       false alarms per clean *retrieved passage*
    ROC / PR  threshold-free areas, ``n/a`` for a detector that emits a hard
              decision and whose operating point therefore cannot be moved
    clean     flag rate on the CLEAN corpus over the same queries — by
              construction this should sit at ``--target-fpr``
    shift     poisoned flag rate - clean flag rate: the corpus-level "is this
              database poisoned at all?" signal

Four things this script refuses to fudge
----------------------------------------
1. **Ground truth is** :meth:`AttackOutcome.ground_truth_mask`, never
   ``RagStore.injected_mask()``. GARAG rewrites documents that are already in the
   corpus, so its injected mask stays all-False and every detection rate scored
   against it would be ``0/0`` dressed up as ``0.0``. The mask is converted to a
   set of poisoned ``doc_id``s once per arm and retrieved passages are labelled by
   id, so both attack modes score through identical code.
2. **Thresholds are calibrated on the CLEAN arm only**, at the ``--target-fpr``
   alarm budget. The investigator has no labelled poison; tuning a threshold on
   the attacked corpus leaks the answer being measured.
3. **Nothing is normalised within a query.** Scores are pooled raw across
   queries. A score normalised inside one retrieved set carries no cross-query
   information and collapses the pooled ROC to chance — a bug this project has
   hit three times.
4. **An attack that did not fire invalidates its detection row**, so each
   attack's own success metric (from
   :mod:`aiforensics.metrics.attack_success`) is measured and printed directly
   above its table, and the clean, un-attacked accuracy is printed once as the
   reference point. The four attacks do not share a success criterion and this
   script does not pretend they do:

   ============  ==============================================================
   poisonedrag   ASR — the target answer is a substring of the generation
   jamming       jamming rate — the system stops answering a query it answered
   phantom       backdoor gap — retrieved for triggered queries, not for others
   garag         gold@k drop and normalised-EM drop, before minus after
   ============  ==============================================================

What ``--n-poison`` means per attack
------------------------------------
``N`` is swept as one number but it drives a different knob in each threat model,
because the attacks are not the same shape. The mapping is recorded in the JSON:

    poisonedrag   ``PoisonedRAGConfig.n_poison``  — texts injected per query
    jamming       ``JammingConfig.n_blocker``     — blocker documents per query
    garag         ``garag_top_k``                 — corpus documents rewritten per query
    phantom       not swept — Phantom plants ONE document by construction (§4.1)
                  and it serves every triggered query, so it has no per-query
                  poison budget. Its arm is reported once per variant, with a
                  note saying so, rather than repeated identically per N.

Usage::

    python scripts/run_matrix.py --queries 40 --attacked-fraction 0.5
    python scripts/run_matrix.py --attacks garag phantom --n-poison 1 5
    python scripts/run_matrix.py --attack-variants poisonedrag=blackbox,whitebox
    python scripts/run_matrix.py --detectors ragdefender gmtp --generator none

Long runs: each attack arm is independent and the JSON is rewritten after every
arm, so a crash in one attack keeps the arms that already finished. ``--resume``
reloads that JSON and skips arms it already contains.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Library deprecation noise only. RuntimeWarning is deliberately left visible:
# it is how attack_success.py reports that a measurement is degenerate, and this
# script captures and records those warnings per arm.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.registry import apply_attack, get_attack, list_attacks
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_encoder, load_qrels
from aiforensics.detectors import Detector, get_detector, list_detectors
from aiforensics.metrics.attack_success import (
    CRITERIA,
    attack_success_rate,
    backdoor_gap,
    clean_accuracy,
    jamming_rate,
    retrieval_degradation,
)
from aiforensics.metrics.detection import detection_metrics, pr_auc, roc_auc
from aiforensics.pipeline import RagPipeline

logger = logging.getLogger(__name__)

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"

#: Published venue per detector key, for the report caption. Copied from the
#: detector modules' own citations; not re-fetched in this script.
DETECTOR_VENUE: dict[str, str] = {
    "ragdefender": "ACSAC'25",
    "gmtp": "Findings ACL'25",
    "npas": "ICML'26",
    "reliability": "NeurIPS'25",
    "perplexity": "PoisonedRAG's evaluated baseline (USENIX Sec'25)",
}

#: Which detectors cannot run without the project generator. ``gmtp`` is absent
#: on purpose: it needs a *masked* LM, which it loads itself and which is
#: unrelated to ``--generator``. A detector key that is not listed here is
#: assumed to need nothing, so a newly registered detector is never silently
#: dropped from the matrix.
DETECTOR_NEEDS_GENERATOR: frozenset[str] = frozenset({"perplexity", "npas", "reliability"})

#: How the swept ``N`` reaches each attack. ``("config", field)`` overrides a
#: field of the attack's own config dataclass; ``("kwarg", name)`` is passed to
#: :func:`apply_attack` as a keyword.
N_KNOB: dict[str, tuple[str, str]] = {
    "poisonedrag": ("config", "n_poison"),
    "jamming": ("config", "n_blocker"),
    "garag": ("kwarg", "garag_top_k"),
    "phantom": ("fixed", "n_passages"),
    "inceptionrag": ("fixed", "n_groups"),
}

#: Phantom is the exception: it has no per-query poison budget to sweep. One
#: document is planted and it serves every query containing the trigger (§4.1),
#: and ``n_passages > 1`` produces byte-identical copies for the unoptimised
#: baseline — a near-duplicate cluster the harness would have created, not the
#: attack. So the sweep is not applied to it and it is reported once per variant.
#: InceptionRAG is the other exception, for a different reason. Its N=2 is
#: *structural*: one attribute document plus one bridge document is the payload,
#: and neither half is the attack on its own. Mapping the sweep's N onto
#: documents would emit a lone attribute document at N=1 — a plain PoisonedRAG
#: passage — and silently halve N at even values. So the sweep plants one group
#: (= 2 documents) and the §5.7 Fig. 4 scaling is reproduced with the explicit
#: n_attribute_docs / n_bridge_docs / n_groups config fields instead. Every
#: InceptionRAG arm therefore sits at N=2 while PoisonedRAG's sit at the swept N.
N_FIXED_VALUE: dict[str, int] = {"phantom": 1, "inceptionrag": 1}

#: Where this harness deviates from ``AttackSpec.default_variant``, and why. The
#: registry's default for jamming is the paper's flagship black-box optimisation,
#: whose threat model is live query access: it hill-climbs one blocker per query
#: for up to ``num_iterations`` rounds of ``batch_size`` generations each, i.e.
#: tens of thousands of RAG calls per query. That is not a default a matrix run
#: can carry, so the cheap §5.4 instruction-injection variant is used unless the
#: caller asks otherwise. The substitution is printed and written to the JSON.
HARNESS_DEFAULT_VARIANT: dict[str, str] = {
    "jamming": "instruction_injection",
    "inceptionrag": "global_only",
}
HARNESS_DEFAULT_REASON: dict[str, str] = {
    "jamming": (
        "registry default 'blackbox_optimized' is the paper's flagship and needs live "
        "query access to the deployed system (tens of thousands of generations per "
        "query); this run used the §5.4 'instruction_injection' variant instead. Ask "
        "for the flagship with --attack-variants jamming=blackbox_optimized "
        "--jamming-oracle-encoder <a model that is NOT the system retriever>."
    ),
    "inceptionrag": (
        "registry default 'full' is the paper's attack and needs live query access: CTTFT "
        "spends up to 25 victim answer strings per target (§4.5), each one a full "
        "retrieve-and-generate round on this arm's own store. This run used the Table 3 "
        "'global_only' ablation instead, whose ASR the paper measures at 50.0 against "
        "88.0 for the full attack on Gemini/NQ — do not report it as the paper's number. "
        "Ask for the flagship with --attack-variants inceptionrag=full."
    ),
}


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI. See the module docstring for what each knob means."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--attacks", nargs="*", default=None,
                    help="attack registry keys; default = every registered attack")
    ap.add_argument("--attack-variants", nargs="*", default=None, metavar="ATTACK=VARIANT[,VARIANT]",
                    help="per-attack variant selector, e.g. poisonedrag=blackbox,whitebox. "
                         "Pass the single word 'all' to run every variant of every "
                         "selected attack (controls included, which are labelled as such).")
    ap.add_argument("--detectors", nargs="*", default=None,
                    help="detector registry keys; default = every rag-defense detector")
    ap.add_argument("--n-poison", type=int, nargs="+", default=[5], metavar="N",
                    help="poison per query, sweepable; drives a different knob per attack, "
                         "see the module docstring")
    ap.add_argument("--k", type=int, default=5, help="retrieval depth (PoisonedRAG default 5)")
    ap.add_argument("--queries", type=int, default=40, help="target questions to evaluate over")
    ap.add_argument("--attacked-fraction", type=float, default=0.5,
                    help="fraction of the query set the attack targets; the rest stay clean "
                         "so the false-positive rate is measured on a realistic mix")
    ap.add_argument("--background", type=int, default=20000,
                    help="background passages in the corpus, on top of the gold passages")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--normalize", dest="normalize", action="store_true", default=True,
                    help="unit-norm vectors: inner product is cosine (project default)")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false",
                    help="leave vectors un-normalised: the index scores raw dot products, "
                         "which is what PoisonedRAG's retrievers do. Only reachable with an "
                         "encoder that does not self-normalise (see scripts/verify_retrievers.py)")
    ap.add_argument("--strip-normalizer", action="store_true",
                    help="drop a Normalize module from the encoder so --no-normalize bites")
    ap.add_argument("--ir-max-retries", type=int, default=3,
                    help="InceptionRAG logic-trap retries. The reference uses 3; "
                         "a local attacker LM needs more. Measured on this repo's "
                         "NQ targets: Qwen3-1.7B 0/6 valid traps at 3, Qwen3-8B "
                         "5/6 at 5. A low yield is not a weak attack, it is an "
                         "attacker LM that cannot hold the three trap constraints.")
    ap.add_argument("--generator", default="qwen3-1.7b",
                    help="project generator for the RAG pipeline and for the detectors and "
                         "attacks that need an LM; 'none' loads nothing and every cell that "
                         "needs one is skipped with a stated reason")
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="alarm budget on CLEAN data used to calibrate every threshold")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--phantom-trigger", default="iphone",
                    help="Phantom's trigger; the paper's set is in phantom.PHANTOM_TRIGGERS")
    ap.add_argument("--phantom-objective", default="refusal",
                    help="Phantom Table 1 objective key")
    ap.add_argument("--jamming-oracle-encoder", default=None,
                    help="the ATTACKER's own sentence encoder for jamming's black-box "
                         "optimisation. Must not be the system retriever — the threat model "
                         "forbids it — so there is no default and the flagship variant is "
                         "skipped without it")
    ap.add_argument("--jamming-judge", dest="jamming_judge", action="store_true", default=True,
                    help="use the LLM answer judge in the jamming criterion (paper §6.1)")
    ap.add_argument("--no-jamming-judge", dest="jamming_judge", action="store_false",
                    help="substring criterion only: the conservative half, under-reports")
    ap.add_argument("--garag-mode", default="both", choices=["both", "reader", "retriever"],
                    help="GARAG's fitness; 'both' is the paper's attack")
    ap.add_argument("--clean-store", default=None,
                    help="path to a saved clean RagStore to reuse instead of building one")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--batch-size", type=int, default=4, help="generation batch size")
    ap.add_argument("--name", default=None,
                    help="report basename; default is derived from the configuration and "
                         "carries no timestamp, so a rerun overwrites its own file")
    ap.add_argument("--resume", action="store_true",
                    help="reload the report and skip attack arms it already contains")
    ap.add_argument("--log-level", default="WARNING")
    return ap.parse_args(argv)


def parse_variant_selector(items: Sequence[str] | None, attacks: Sequence[str]) -> dict[str, list[str]]:
    """Turn ``["poisonedrag=blackbox,whitebox"]`` into ``{attack: [variants]}``.

    Args:
        items: Raw ``--attack-variants`` values, or the single word ``"all"``.
        attacks: Attack keys selected for this run, used to expand ``"all"``.

    Returns:
        Mapping from attack key to the variants requested for it. Attacks absent
        from the mapping fall back to :data:`HARNESS_DEFAULT_VARIANT` and then to
        the registry's own ``default_variant``.

    Raises:
        ValueError: On a malformed item.
        KeyError: On an unknown attack key or an unknown variant.
    """
    if not items:
        return {}
    if len(items) == 1 and items[0].strip().lower() == "all":
        return {a: get_attack(a).variant_keys() for a in attacks}
    out: dict[str, list[str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"--attack-variants takes ATTACK=VARIANT[,VARIANT] items (or the single "
                f"word 'all'); got {item!r}"
            )
        key, _, raw = item.partition("=")
        spec = get_attack(key.strip())
        wanted = [v.strip() for v in raw.split(",") if v.strip()]
        for v in wanted:
            spec.get_variant(v)  # raises KeyError listing the valid variants
        out[spec.key] = wanted
    return out


# --------------------------------------------------------------------------- #
# corpus and query pool
# --------------------------------------------------------------------------- #
def build_clean_store(qids: Sequence[str], args: argparse.Namespace, encoder: Any) -> RagStore:
    """Build (or load) the clean corpus: every gold passage plus background.

    The gold passages matter for two of the four attacks — GARAG rewrites one of
    them, and the clean-accuracy reference is meaningless if the corpus cannot
    answer the questions at all.
    """
    if args.clean_store:
        store = RagStore.load(args.clean_store, encoder=encoder)
        print(f"clean store loaded from {args.clean_store}: {len(store):,} docs "
              f"({store.similarity}, encoder={store.encoder_name})")
        return store

    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep: list[Any] = []
    n_bg = 0
    for doc in load_corpus("nq", max_docs=args.background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif n_bg < args.background:
            keep.append(doc)
            n_bg += 1
    found = sum(1 for d in keep if d.doc_id in gold)
    slug = args.encoder.split("/")[-1].replace(".", "-")
    name = f"matrix-{len(keep)}-{slug}-{'cos' if args.normalize else 'dot'}"
    store = build_index(keep, encoder=encoder, corpus_name=name, normalize=args.normalize,
                        encoder_name=args.encoder)
    print(f"clean corpus: {len(store):,} docs, {found}/{len(gold)} gold passages present, "
          f"similarity={store.similarity}")
    return store


def gold_ids_for(qids: Sequence[str], store: RagStore) -> dict[str, list[str]]:
    """``{query_id: [gold doc_id in this store]}`` from the BeIR qrels.

    GARAG's paper attacks the top-ranked passage the reader already answers
    correctly from. This harness cannot know which that is, but supplying the
    qrels gold ids is strictly closer to the paper than letting the registry fall
    back to the store's own top-1 hit.
    """
    try:
        qrels = load_qrels("nq", split="test")
    except Exception as exc:
        logger.warning("could not load qrels (%s); GARAG will use the retrieval fallback", exc)
        return {}
    present = set(store.doc_ids)
    return {q: [d for d in qrels.get(q, {}) if d in present] for q in qids}


# --------------------------------------------------------------------------- #
# scoring one arm
# --------------------------------------------------------------------------- #
def poison_id_set(outcome: Any, store: RagStore) -> set[str]:
    """The ground-truth poisoned ids, from :meth:`AttackOutcome.ground_truth_mask`.

    Requirement 4 of this harness in one function: the mask is computed from the
    outcome's ``poisoned_doc_ids`` (and, for perturb mode, validated against the
    text digests), never from ``store.injected_mask()``. Converting it to a set of
    ids is what lets a *retrieved passage* be labelled — the detector never sees
    the corpus, only the top-``k``.
    """
    mask = np.asarray(outcome.ground_truth_mask(store), dtype=bool)
    doc_ids = list(store.doc_ids)
    if mask.size != len(doc_ids):
        raise ValueError(
            f"ground-truth mask has {mask.size} entries but the store holds {len(doc_ids)} "
            f"documents; the labels would be misaligned"
        )
    return {d for d, flag in zip(doc_ids, mask, strict=True) if flag}


def score_arm(
    detector: Any,
    store: RagStore,
    pairs: Sequence[tuple[str, str]],
    k: int,
    poison_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Score every retrieved passage for every query and pool the results.

    Scores are pooled **raw**. Rank- or z-normalising inside a retrieved set
    would make the numbers incomparable across queries and drive the pooled ROC
    to 0.5 regardless of how good the detector is.

    Returns:
        ``(scores, y_true, per_query)`` with ``scores`` and ``y_true`` pooled over
        all queries in retrieval order.
    """
    scores: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    per_query: list[dict] = []
    for qid, question in pairs:
        hits = store.search(question, k=k)
        texts = [h.text for h in hits]
        if not texts:
            continue
        s = np.asarray(detector.score((question, texts)), dtype=np.float64).reshape(-1)
        if s.size != len(texts):
            raise ValueError(
                f"detector returned {s.size} scores for {len(texts)} passages on query "
                f"{qid!r}; one score per passage is the Detector contract"
            )
        t = np.fromiter((h.doc_id in poison_ids for h in hits), dtype=bool, count=len(hits))
        scores.append(s)
        truth.append(t)
        per_query.append({
            "query_id": qid,
            "n_retrieved": len(hits),
            "n_poison_at_k": int(t.sum()),
            "doc_ids": [h.doc_id for h in hits],
        })
    if not scores:
        return np.zeros(0), np.zeros(0, dtype=bool), per_query
    return np.concatenate(scores), np.concatenate(truth), per_query


def count_positives(store: RagStore, pairs: Sequence[tuple[str, str]], k: int,
                    poison_ids: set[str]) -> tuple[int, int]:
    """``(poisoned passages inside the top-k, passages scored in total)``.

    Measured with one retrieval pass so it is available even when every detector
    fails: a cell with zero retrieved positives has an undefined detection rate
    for reasons that have nothing to do with the detector, and the table has to
    be able to say so.
    """
    n_pos = n_all = 0
    for _qid, question in pairs:
        hits = store.search(question, k=k)
        n_all += len(hits)
        n_pos += sum(1 for h in hits if h.doc_id in poison_ids)
    return n_pos, n_all


def calibrate(detector: Any, clean_scores: np.ndarray, target_fpr: float) -> tuple[float, bool]:
    """Pick a decision threshold from **clean data only**.

    The investigator has no labelled poison, so the only honest calibration is an
    alarm budget: put the threshold at the ``1 - target_fpr`` quantile of the
    clean scores, which makes a clean database raise alarms on ``target_fpr`` of
    its retrieved passages by construction. Whatever the poisoned database then
    raises above that line is signal rather than baseline noise.

    Returns:
        ``(threshold, calibrated)``. ``calibrated`` is False for a detector that
        emits a hard decision — its operating point is fixed by its own paper and
        cannot be moved to meet a budget, which is itself a finding.
    """
    finite = clean_scores[np.isfinite(clean_scores)]
    if finite.size == 0:
        return float("nan"), False
    if detector.threshold is not None and np.unique(finite).size <= 2:
        return float(detector.threshold), False
    return float(np.quantile(finite, 1.0 - target_fpr)), True


def flags(scores: np.ndarray, threshold: float) -> np.ndarray:
    """``score >= threshold``. A NaN score never raises an alarm.

    A detector may return NaN for a passage it declines to judge (Reliability-RAG
    scores abstentions this way). Comparison against NaN is False, which is the
    conservative reading — an abstention is not an accusation — and the count of
    NaNs is reported per cell so the choice is visible.
    """
    if math.isnan(threshold):
        return np.zeros(scores.shape, dtype=bool)
    return scores >= threshold


def auc_pair(y_true: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None, int]:
    """ROC-AUC and PR-AUC over the finite scores, plus the number dropped.

    A detector emitting at most two distinct values is a hard-decision method:
    its "curve" is a single operating point, so both areas are reported as
    ``n/a`` rather than as a number that invites a comparison it cannot support.
    """
    keep = np.isfinite(scores)
    n_dropped = int((~keep).sum())
    yt, s = y_true[keep], scores[keep]
    if s.size == 0 or np.unique(s).size <= 2:
        return None, None, n_dropped
    return float(roc_auc(yt, s)), float(pr_auc(yt, s)), n_dropped


# --------------------------------------------------------------------------- #
# arm planning
# --------------------------------------------------------------------------- #
@dataclass
class Arm:
    """One attack configuration to run the whole detector panel against."""

    attack: str
    variant: str
    n: int
    config: dict[str, Any] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
    query_pool: list[tuple[str, str]] = field(default_factory=list)
    eval_pairs: list[tuple[str, str]] = field(default_factory=list)
    query_ids: list[str] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identifier used in the report and by ``--resume``."""
        return f"{self.attack}/{self.variant}/N={self.n}"


def plan_arm(
    attack: str,
    variant: str,
    n: int,
    *,
    args: argparse.Namespace,
    base_pairs: Sequence[tuple[str, str]],
    attacked_ids: Sequence[str],
    resources: dict[str, Any],
) -> Arm:
    """Build the attack-specific config, kwargs and query sets for one arm."""
    spec = get_attack(attack)
    knob_kind, knob_name = N_KNOB[attack]
    config: dict[str, Any] = {"seed": args.seed}
    if attack == "inceptionrag":
        # --ir-max-retries has to reach the ATTACK, not just the suffix-pool
        # config built earlier for resolve_global_suffixes(). Without this the
        # flag is recorded in the report and silently ignored by the run: the
        # first full-variant run logged "after 3 retries" while its own config
        # said 5.
        config["max_retries"] = args.ir_max_retries
    kwargs: dict[str, Any] = {}
    notes: list[str] = []

    if knob_kind == "config":
        config[knob_name] = n
    elif knob_kind == "kwarg":
        kwargs[knob_name] = n
    else:  # "fixed": the sweep does not apply to this attack
        fixed = N_FIXED_VALUE[attack]
        config[knob_name] = fixed
        if n != fixed:
            notes.append(
                f"the --n-poison sweep does not apply to {attack}: it plants "
                f"{knob_name}={fixed} by construction, so this arm is reported once "
                f"instead of once per N"
            )
        n = fixed

    model, tokenizer = resources.get("model"), resources.get("tokenizer")

    if attack == "poisonedrag":
        kwargs.update(model=model, tokenizer=tokenizer)
        notes.append("the generation condition I comes from the targets file's 'adv_texts' "
                     "where present; only a query without them falls back to the attacker "
                     "LM's §4.2.1 regenerate-until-success loop")
    elif attack == "garag":
        kwargs.update(model=model, tokenizer=tokenizer, garag_mode=args.garag_mode)
        notes.append(f"GARAG fitness mode={args.garag_mode!r}; documents to perturb come "
                     f"from the BeIR qrels where available, otherwise the registry falls "
                     f"back to the store's own top-{n} hit (a harness decision, not the "
                     f"paper's selection rule)")
    elif attack == "jamming":
        if variant == "oracle":
            kwargs.update(model=model, tokenizer=tokenizer)
        elif variant == "blackbox_optimized":
            # respond_factory, clean_responses and the token pool are filled in by
            # run_arm: the respond() channel has to be built against the store this
            # arm is about to attack, which does not exist yet at planning time.
            kwargs.update(
                oracle_encoder=resources.get("jamming_oracle_encoder"),
                oracle_tokenizer=tokenizer,
            )
            notes.append("black-box optimisation queries the deployed system through a "
                         "respond() built from this run's store and generator; that is the "
                         "paper's threat model (answer strings only, no weights, no k)")
    elif attack == "inceptionrag":
        kwargs.update(model=model, tokenizer=tokenizer,
                      suffix_pool=resources.get("inceptionrag_suffix_pool"),
                      retrieval_k=args.k)
        if variant == "full":
            # `respond` is filled in by run_arm: the victim channel has to be built
            # against the store this arm is about to attack, which does not exist
            # yet at planning time (the jamming precedent).
            notes.append("CTTFT queries the deployed system through a respond() built from "
                         "this run's store and generator; that is the paper's threat model "
                         "(§3.1, answer strings only — no weights, no gradients, no k)")
        notes.append("the logic trap comes from the targets file's 'ad_text' where present; "
                     "only a query without one falls back to the attacker LM's §4.2 "
                     "author-and-validate loop")
        pair = resources.get("inceptionrag_suffix_pair")
        if pair is not None:
            notes.append(f"global suffix pair: anchor {pair.anchor_index} / bridge "
                         f"{pair.bridge_index} from {pair.source} "
                         f"(stored golden_reward={pair.golden_reward})")
    elif attack == "phantom":
        kwargs.update(trigger=args.phantom_trigger, objective=args.phantom_objective)
        notes.append(f"Phantom has no target query and no per-query poison budget: one "
                     f"document serves every query containing the trigger "
                     f"{args.phantom_trigger!r}")

    # Query sets. Phantom is the only attack whose evaluation queries differ from
    # the clean question set: the backdoor only fires on a query containing the
    # trigger, so the attacked share of the pool carries it appended. That is the
    # easier of the paper's two settings (the paper uses naturally-triggered
    # queries) and it is stated in the report.
    if spec.query_scope == "trigger-conditional":
        attacked = set(attacked_ids)
        sep = " "
        eval_pairs = [
            (qid, f"{q}{sep}{args.phantom_trigger}" if qid in attacked else q)
            for qid, q in base_pairs
        ]
        query_pool = list(eval_pairs)
        query_ids = None
        notes.append("triggered queries were built by APPENDING the trigger to the attacked "
                     "share of the pool; the untriggered rest is the OUT set of Eq. 1. The "
                     "paper's protocol uses queries that contain the trigger naturally, "
                     "which is the harder setting")
    else:
        eval_pairs = list(base_pairs)
        query_pool = list(base_pairs)
        query_ids = list(attacked_ids)

    return Arm(attack=attack, variant=variant, n=n, config=config, kwargs=kwargs,
               query_pool=query_pool, eval_pairs=eval_pairs, query_ids=query_ids, notes=notes)


def arm_blocker(attack: str, variant: str, args: argparse.Namespace,
                resources: dict[str, Any]) -> str | None:
    """Why this arm cannot run, or ``None`` when it can.

    Structurally undefined arms are skipped with the reason printed and recorded.
    A zero in this table would read as "the attack was invisible to every
    detector", which is the opposite of what a missing resource means.
    """
    spec = get_attack(attack)
    reqs = spec.requirements(variant)
    has_lm = resources.get("model") is not None

    if reqs["needs_generator"] and not has_lm:
        if attack == "poisonedrag":
            pass  # satisfied offline by the precomputed adv_texts in the targets file
        elif attack == "inceptionrag" and variant != "full" \
                and resources.get("inceptionrag_traps_precomputed"):
            pass  # satisfied offline by a precomputed 'ad_text' in the targets file
        elif attack == "garag" and args.garag_mode == "retriever":
            pass  # L_RSR only; no reader LM in the loop
        else:
            return (f"variant needs an attacker/reader LM and --generator is {args.generator!r}; "
                    f"load a generator, or pick a variant that needs none")
    if attack == "garag" and variant == "genetic" and args.garag_mode in ("both", "reader") \
            and not has_lm:
        return ("GARAG's L_GPR scores candidates with a reader LM; load a generator, use "
                "--garag-mode retriever (retrieval only, not an attack on its own), or "
                "--attack-variants garag=random (unoptimised typos, a control)")
    if attack == "inceptionrag":
        if variant == "full" and not has_lm:
            return ("CTTFT needs live query access to the deployed system (up to 25 victim "
                    "answer strings per target); load a generator, or use "
                    "--attack-variants inceptionrag=global_only (the Table 3 ablation, no "
                    "feedback)")
        if not resources.get("inceptionrag_traps_precomputed") and not has_lm:
            return ("no 'ad_text' in the targets file and no attacker LM: the attribute and "
                    "bridge documents cannot be authored. Load a generator, or precompute "
                    "the traps into the targets file")
        blocker = resources.get("inceptionrag_blocker")
        if blocker:
            return blocker
    if attack == "jamming" and variant == "oracle" and not has_lm:
        return "the oracle variant writes the blocker body with an LM; load a generator"
    if attack == "jamming" and variant == "blackbox_optimized":
        if not has_lm:
            return ("black-box optimisation needs live query access to the deployed system, "
                    "which needs the generator; load one")
        if resources.get("jamming_oracle_encoder") is None:
            return ("the attacker's own sentence encoder is missing: pass "
                    "--jamming-oracle-encoder <model id>. It must NOT be the system "
                    "retriever — the threat model forbids reading the retriever — so there "
                    "is deliberately no default")
    return None


# --------------------------------------------------------------------------- #
# attack-success measurement
# --------------------------------------------------------------------------- #
def measure_success(
    arm: Arm,
    outcome: Any,
    *,
    args: argparse.Namespace,
    store: RagStore,
    clean_store: RagStore,
    targets: dict[str, dict],
    base_pairs: Sequence[tuple[str, str]],
    attacked_ids: Sequence[str],
    resources: dict[str, Any],
    clean_answers: dict[str, Any],
    gold: dict[str, list[str]],
) -> tuple[dict[str, Any] | None, str, list[str]]:
    """Measure the arm's attack with **its own** paper's criterion.

    Returns:
        ``(record, headline, warnings)`` — the metric's ``to_dict()``, a one-line
        summary for the table header, and any ``RuntimeWarning`` text the metric
        raised (those warnings are the module's way of saying a measurement is
        degenerate, so they are surfaced, not swallowed).
    """
    model = resources.get("model")
    text_of = dict(base_pairs)
    # The queries the attack actually landed on, not the ones it was asked to
    # attack: GARAG skips a document a previous query already perturbed, and
    # scoring the skipped query would count a non-attack as a failed attack.
    landed = [q for q in outcome.attacked_query_ids if q in text_of]
    attacked = landed or [q for q in attacked_ids if q in text_of]
    questions = [text_of[q] for q in attacked]
    caught: list[str] = []

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        try:
            if arm.attack == "phantom":
                res = backdoor_gap(
                    store, [text_of[q] for q, _ in base_pairs], args.phantom_trigger,
                    k=args.k, doc_ids=sorted(outcome.poisoned_doc_ids),
                    append_trigger=True, label=arm.key,
                )
                headline = (f"backdoor gap={res.gap:.3f} "
                            f"(triggered {res.triggered_rate:.3f} vs clean {res.clean_rate:.3f}, "
                            f"Ret-FR={res.ret_fr:.3f})")
                record = res.to_dict()
            elif model is None:
                return None, "not measured: no generator loaded", []
            else:
                pipe = RagPipeline(store, model, resources["tokenizer"], k=args.k,
                                   max_new_tokens=args.max_new_tokens)
                responses = pipe.answer(questions, k=args.k, batch_size=args.batch_size)
                if arm.attack == "poisonedrag":
                    res = attack_success_rate(
                        pipe, questions,
                        [targets[q].get("incorrect answer") for q in attacked],
                        correct_answers=[targets[q].get("correct answer") for q in attacked],
                        responses=responses, k=args.k, label=arm.key,
                    )
                    headline = (f"ASR={res.asr:.3f} ({res.n_success}/{res.n_queries}), "
                                f"poison retrieved for {res.poison_retrieved_rate:.3f} of "
                                f"queries, accuracy under attack={res.accuracy_under_attack:.3f}")
                    record = res.to_dict()
                elif arm.attack == "inceptionrag":
                    # attack_success_rate already takes a LIST of acceptable strings per
                    # query, so the alias set is native; correct_answers is mandatory
                    # rather than optional because its degenerate-target check IS the
                    # alias-contamination alarm (an alias equal to the correct answer
                    # would otherwise score as a free success).
                    cand = (outcome.diagnostics or {}).get("success_candidates") or {}
                    res = attack_success_rate(
                        pipe, questions,
                        [cand.get(q) or [targets[q].get("incorrect answer")] for q in attacked],
                        correct_answers=[targets[q].get("correct answer") for q in attacked],
                        responses=responses, k=args.k, label=arm.key,
                    )
                    # attack_success_rate stamps PoisonedRAG's criterion string by
                    # default; this arm's criterion adds the alias set, so say so.
                    # AttackSuccessResult is frozen — replace(), never assignment.
                    res = dataclasses.replace(res, criterion=CRITERIA["inceptionrag"])
                    pairs = (outcome.diagnostics or {}).get("pair_retrieved_counts") or {}
                    headline = (f"ASR={res.asr:.3f} ({res.n_success}/{res.n_queries}), "
                                f"poison retrieved for {res.poison_retrieved_rate:.3f} of "
                                f"queries, pairs at k={args.k} "
                                f"(2/1/0)={pairs.get('2', 0)}/{pairs.get('1', 0)}/"
                                f"{pairs.get('0', 0)}, accuracy under attack="
                                f"{res.accuracy_under_attack:.3f}")
                    record = res.to_dict()
                elif arm.attack == "jamming":
                    judge = None
                    if args.jamming_judge:
                        from aiforensics.attacks.jamming import make_llm_judge
                        judge = make_llm_judge(model, resources["tokenizer"])
                    res = jamming_rate(
                        pipe, questions, responses=responses,
                        clean_responses=[clean_answers[q] for q in attacked],
                        judge=judge, k=args.k, label=arm.key,
                    )
                    headline = (f"jamming rate={res.jamming_rate:.3f} "
                                f"({res.n_jammed}/{res.n_retained} retained, "
                                f"{res.n_excluded} excluded), substring-only="
                                f"{res.substring_only_rate:.3f}, blocker retrieved for "
                                f"{res.blocker_retrieved_rate:.3f} of queries")
                    record = res.to_dict()
                elif arm.attack == "garag":
                    # The documents the attack rewrote are the ones whose fate the
                    # before/after contrast is about; the qrels gold is the fallback
                    # for a query the registry resolved by retrieval instead.
                    perturbed = outcome.query_doc_ids
                    res = retrieval_degradation(
                        clean_store, store, questions,
                        [perturbed.get(q) or gold.get(q, []) for q in attacked], k=args.k,
                        answers=[targets[q].get("correct answer") for q in attacked],
                        clean_responses=[clean_answers[q] for q in attacked],
                        perturbed_responses=responses, label=arm.key,
                    )
                    headline = (f"gold@k {res.gold_retrieval_clean:.3f}->"
                                f"{res.gold_retrieval_perturbed:.3f} "
                                f"(drop {res.retrieval_drop:+.3f}), EM {res.em_clean:.3f}->"
                                f"{res.em_perturbed:.3f} (drop {res.em_drop:+.3f})")
                    record = res.to_dict()
                else:
                    return None, f"no success metric wired for attack {arm.attack!r}", []
        except Exception as exc:
            logger.warning("attack-success measurement failed for %s", arm.key, exc_info=True)
            return None, f"measurement failed: {type(exc).__name__}: {exc}", []
        caught = [str(w.message) for w in caught_warnings
                  if issubclass(w.category, RuntimeWarning)]
    return record, headline, caught


# --------------------------------------------------------------------------- #
# jamming's live-query channel
# --------------------------------------------------------------------------- #
def make_respond_factory(store: RagStore, model: Any, tokenizer: Any, *, k: int,
                         max_new_tokens: int, batch_size: int) -> Any:
    """Build jamming's ``respond_factory``: query access, and nothing else.

    The paper's black-box optimisation sees exactly one channel — the answer
    string the deployed system returns for a candidate blocker. This reproduces
    that channel honestly: a candidate is scored against the *live* corpus, it
    enters the context only if it actually beats the k-th clean hit, and a
    candidate that fails to be retrieved returns ``None`` so the optimiser
    substitutes the clean answer, exactly as the reference does. Nothing here is
    handed to the attacker: the retriever, the generator and ``k`` stay behind
    the callable.

    Args:
        store: The deployed store (already carrying whatever else was injected).
        model: The deployed generator.
        tokenizer: Its tokenizer.
        k: The system's retrieval depth.
        max_new_tokens: Generation budget.
        batch_size: Generation batch size.

    Returns:
        ``factory(question) -> respond(candidate_documents) -> [answer | None]``.
    """
    from aiforensics.data.rag import embed_texts
    from aiforensics.models.llm import generate

    pipeline = RagPipeline(store, model, tokenizer, k=k, max_new_tokens=max_new_tokens)

    def factory(question: str) -> Any:
        hits = store.search(question, k=k)
        floor = hits[-1].score if len(hits) >= k else -float("inf")
        base_texts = [h.text for h in hits]

        def respond(candidates: Sequence[str]) -> list[str | None]:
            cands = list(candidates)
            if not cands:
                return []
            emb = embed_texts(cands, encoder=store.encoder, normalize=store.normalized)
            qvec = store.encode_queries([question])[0]
            sims = np.asarray(emb, dtype=np.float64) @ np.asarray(qvec, dtype=np.float64)
            prompts: list[str] = []
            slots: list[int] = []
            out: list[str | None] = [None] * len(cands)
            for i, (text, sim) in enumerate(zip(cands, sims, strict=True)):
                if float(sim) < float(floor):
                    continue  # not retrieved: the system is genuinely unaffected
                context = [text, *base_texts[: max(0, k - 1)]]
                prompts.append(pipeline.build_prompt(question, context))
                slots.append(i)
            if prompts:
                answers = generate(model, tokenizer, prompts,
                                   max_new_tokens=max_new_tokens,
                                   batch_size=batch_size, do_sample=False)
                for slot, answer in zip(slots, answers, strict=True):
                    out[slot] = str(answer).strip()
            return out

        return respond

    return factory


# --------------------------------------------------------------------------- #
# InceptionRAG's live-query channel
# --------------------------------------------------------------------------- #
def make_inceptionrag_respond(store: RagStore, model: Any, tokenizer: Any, *, k: int,
                              max_new_tokens: int) -> Any:
    """Build CTTFT's victim channel: ``question -> answer`` and nothing else.

    §3.1's threat model is one interaction: send a query, read the final
    generated response. The attack plants its pair in ``store`` first, so this
    just asks the deployed system what it now says — the poison has to win the
    real top-k on its own, which is the deviation from the reference harness
    (it hands its 2 documents a guaranteed slot in the candidate pool).

    This is deliberately *not* jamming's ``respond_factory`` shape. Jamming
    scores one candidate document at a time against a fixed clean context;
    InceptionRAG needs both of its documents in the corpus simultaneously,
    because the whole mechanism is what the generator does when it sees them
    together.

    Args:
        store: The deployed store, mutated in place by the attack.
        model: The deployed generator.
        tokenizer: Its tokenizer.
        k: The system's retrieval depth.
        max_new_tokens: Generation budget.

    Returns:
        ``respond(question) -> answer_string``.
    """
    pipeline = RagPipeline(store, model, tokenizer, k=k, max_new_tokens=max_new_tokens)

    def respond(question: str) -> str:
        answers = pipeline.answer([question], k=k, batch_size=1)
        first = answers[0]
        return str(getattr(first, "answer", first)).strip()

    return respond


# --------------------------------------------------------------------------- #
# printing
# --------------------------------------------------------------------------- #
def _fmt(value: Any, width: int = 6, places: int = 3) -> str:
    """Format a metric, printing ``n/a`` for None and ``nan`` for undefined."""
    if value is None:
        return f"{'n/a':>{width}s}"
    if isinstance(value, float) and math.isnan(value):
        return f"{'nan':>{width}s}"
    return f"{float(value):{width}.{places}f}"


def print_arm_table(arm: Arm, header: dict[str, Any], cells: Sequence[dict]) -> None:
    """One attack's detector panel."""
    spec = get_attack(arm.attack)
    print()
    print("=" * 104)
    print(f"{spec.name}  [{arm.attack}/{arm.variant}]  N={arm.n}   "
          f"mode={spec.mode}  goal={spec.goal}  ground truth={spec.ground_truth}")
    print(f"  {spec.venue}" + ("" if spec.venue_verified else "   (venue NOT verified)"))
    print(f"  attack success: {header['success_headline']}")
    print(f"  poisoned documents in corpus: {header['n_poisoned_docs']}   "
          f"attacked queries: {header['n_attacked_queries']}/{header['n_eval_queries']}   "
          f"poisoned passages inside top-k: {header['n_positives_retrieved']}"
          f"/{header['n_scored_passages']}")
    if not header["faithful"]:
        print("  ** control variant: these numbers are NOT the paper's attack **")
    for note in header["notes"]:
        print(f"  note: {note}")
    print("=" * 104)

    head = (f"{'detector':16s} {'det.acc':>8s} {'prec':>6s} {'F1':>6s} {'FPR':>6s} "
            f"{'ROC':>6s} {'PR':>6s} | {'clean':>6s} {'pois':>6s} {'shift':>7s}  status")
    print(head)
    print("-" * len(head))
    for c in cells:
        if c["status"] != "ok":
            print(f"{c['detector']:16s} {'—':>8s} {'—':>6s} {'—':>6s} {'—':>6s} "
                  f"{'—':>6s} {'—':>6s} | {'—':>6s} {'—':>6s} {'—':>7s}  "
                  f"SKIPPED: {c['reason']}")
            continue
        mark = "" if c["calibrated_on_clean"] else "*"
        shift = c["flag_rate_shift"]
        print(f"{c['detector'] + mark:16s} {_fmt(c['detection_rate'], 8)} "
              f"{_fmt(c['precision'])} {_fmt(c['f1'])} {_fmt(c['fpr'])} "
              f"{_fmt(c['roc_auc'])} {_fmt(c['pr_auc'])} | "
              f"{_fmt(c['clean_flag_rate'])} {_fmt(c['poisoned_flag_rate'])} "
              f"{shift:+7.3f}  {c['note']}")


def print_summary(arms: Sequence[dict], detectors: Sequence[str], target_fpr: float) -> None:
    """The compact cross-product: detection accuracy, every attack, every detector."""
    print()
    print("#" * 104)
    print("SUMMARY — detection accuracy (#detected poisoned / #poisoned) at a "
          f"{target_fpr:.0%} clean alarm budget")
    print("#" * 104)
    width = max(12, *(len(a["key"]) for a in arms)) if arms else 12
    head = f"{'attack/variant/N':{width}s} " + "".join(f"{d[:11]:>12s}" for d in detectors)
    print(head)
    print("-" * len(head))
    for arm in arms:
        cells = {c["detector"]: c for c in arm["cells"]}
        row = f"{arm['key']:{width}s} "
        for d in detectors:
            c = cells.get(d)
            if c is None or c["status"] != "ok":
                row += f"{'—':>12s}"
            else:
                # A cell whose positives were ALL abstentions is not a 0.000.
                # Printing the rate would read as "missed every one" when the
                # detector judged none of them.
                if c.get("positives_unassessed"):
                    row += f"{'abstain':>12s}"
                else:
                    row += f"{_fmt(c['detection_rate'], 12)}"
        print(row)
        print(f"{'  ^ attack:':{width}s} {arm['success_headline']}")
    print("-" * len(head))
    print("— = cell not run (structurally undefined or the arm failed); see the per-attack "
          "tables for the reason.")
    print("A detection accuracy is only readable next to the attack line above it: an attack "
          "that did not fire\nmakes its whole row meaningless, and this table shows that "
          "rather than hiding it.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """Run the cross-product benchmark. Returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")
    set_seed(args.seed)

    attacks = list(args.attacks) if args.attacks else list_attacks()
    for a in attacks:
        get_attack(a)  # fail early with the list of valid keys
    detectors = list(args.detectors) if args.detectors else list_detectors("rag-defense")
    variant_sel = parse_variant_selector(args.attack_variants, attacks)

    targets: dict[str, dict] = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    n_attacked = max(1, min(len(qids), int(round(args.attacked_fraction * len(qids)))))
    attacked_ids = qids[:n_attacked]
    base_pairs = [(q, str(targets[q]["question"])) for q in qids]

    print("=" * 104)
    print("ATTACK x DETECTOR MATRIX")
    print("=" * 104)
    print(f"attacks   : {attacks}")
    print(f"detectors : {detectors}")
    print(f"queries   : {len(qids)} ({n_attacked} attacked, {len(qids) - n_attacked} clean) | "
          f"k={args.k} | N sweep={args.n_poison} | target FPR={args.target_fpr:.0%}")

    encoder = load_encoder(args.encoder, strip_normalizer=args.strip_normalizer)
    clean_store = build_clean_store(qids, args, encoder)
    clean_path = clean_store.save(f"matrix-clean-{len(clean_store)}")
    gold = gold_ids_for(qids, clean_store)

    # ---- resources ------------------------------------------------------- #
    resources: dict[str, Any] = {"model": None, "tokenizer": None}
    if str(args.generator).lower() not in ("none", "off", ""):
        from aiforensics.models.llm import load_llm
        print(f"loading generator {args.generator} …")
        model, tok = load_llm(args.generator)
        resources["model"], resources["tokenizer"] = model, tok
    else:
        print("generator: NONE — every cell and metric that needs an LM is skipped, "
              "with the reason printed")
    if args.jamming_oracle_encoder:
        resources["jamming_oracle_encoder"] = load_encoder(args.jamming_oracle_encoder)

    # InceptionRAG: the suffix pool and the global pair are run-level, not arm-level
    # (one pool encode, one pair per dataset+victim), so they are resolved once here
    # and any failure becomes a printed arm_blocker reason rather than a stack trace
    # in the middle of the sweep.
    if "inceptionrag" in attacks:
        from aiforensics.attacks.inceptionrag import (
            InceptionRAGConfig,
            load_suffix_pool,
            resolve_global_suffixes,
        )
        resources["inceptionrag_traps_precomputed"] = all(
            bool(targets.get(q, {}).get("ad_text") or targets.get(q, {}).get("ad_text_2"))
            for q in attacked_ids
        )
        try:
            ir_cfg = InceptionRAGConfig(seed=args.seed,
                                        max_retries=args.ir_max_retries)
            pool = load_suffix_pool(ir_cfg.suffix_pool_path or None, ir_cfg.suffix_encoder)
            resources["inceptionrag_suffix_pool"] = pool
            resources["inceptionrag_suffix_pair"] = resolve_global_suffixes(pool, ir_cfg)
            print(f"InceptionRAG suffix pool: {len(pool.anchors)} anchors / "
                  f"{len(pool.bridges)} bridges, encoder={pool.encoder_name}")
        except Exception as exc:
            resources["inceptionrag_blocker"] = (
                f"{type(exc).__name__}: {exc}"
            )
            print(f"InceptionRAG resources unavailable: {resources['inceptionrag_blocker']}")

    # ---- clean reference -------------------------------------------------- #
    clean_answers: dict[str, Any] = {}
    clean_ref: dict[str, Any] = {"measured": False}
    if resources["model"] is not None:
        clean_pipe = RagPipeline(clean_store, resources["model"], resources["tokenizer"],
                                 k=args.k, max_new_tokens=args.max_new_tokens)
        answers = clean_pipe.answer([q for _, q in base_pairs], k=args.k,
                                    batch_size=args.batch_size)
        clean_answers = {qid: ans for (qid, _), ans in zip(base_pairs, answers, strict=True)}
        ref = clean_accuracy(clean_pipe, [q for _, q in base_pairs],
                             [targets[q].get("correct answer") for q in qids],
                             k=args.k, responses=answers, criterion="substring",
                             label="clean-reference")
        clean_ref = {"measured": True, **ref.to_dict()}
        print(f"\nCLEAN REFERENCE (no attack anywhere): accuracy {ref.accuracy:.3f} "
              f"({ref.n_correct}/{ref.n_queries}), criterion=substring match "
              f"(PoisonedRAG §5.1). PoisonedRAG reports 70% on NQ with its own retriever "
              f"and generator; this stack is {args.encoder} + {args.generator}.")
    else:
        print("\nCLEAN REFERENCE: not measured (no generator loaded).")

    # ---- detector panel --------------------------------------------------- #
    # Instantiated ONCE, bound to the clean store and fitted on clean data, then
    # reused for every arm: identical detector state across the whole matrix, and
    # no poisoned-corpus statistic can leak into a detector's calibration.
    clean_sets = [(q, [h.text for h in clean_store.search(q, k=args.k)]) for _, q in base_pairs]
    panel: dict[str, Any] = {}
    detector_meta: list[dict] = []
    for key in detectors:
        meta: dict[str, Any] = {"detector": key, "venue": DETECTOR_VENUE.get(key, "")}
        if key in DETECTOR_NEEDS_GENERATOR and resources["model"] is None:
            meta.update(available=False,
                        reason=f"needs the project generator and --generator is "
                               f"{args.generator!r}")
            detector_meta.append(meta)
            print(f"  detector {key}: SKIPPED — {meta['reason']}")
            continue
        try:
            try:
                det = get_detector(key, store=clean_store, encoder=encoder,
                                   model=resources["model"],
                                   tokenizer=resources["tokenizer"])
            except TypeError:  # detector's constructor takes none of those
                det = get_detector(key)
        except Exception as exc:
            meta.update(available=False, reason=f"could not instantiate: "
                                                f"{type(exc).__name__}: {exc}")
            detector_meta.append(meta)
            print(f"  detector {key}: SKIPPED — {meta['reason']}")
            continue
        meta["available"] = True
        meta["native_threshold"] = (None if det.threshold is None else float(det.threshold))
        # Detector.fit is a no-op on the base class, so `hasattr` says nothing.
        # Only a detector that overrides it has calibration state to fill.
        overrides_fit = type(det).fit is not Detector.fit
        meta["overrides_fit"] = bool(overrides_fit)
        meta["requires_fit"] = bool(getattr(det, "requires_fit", False))
        if overrides_fit:
            try:
                det.fit(clean_sets)
                meta["fit"] = (f"fitted on {len(clean_sets)} clean retrieved sets "
                               f"(no labelled poison used)")
            except Exception as exc:
                meta["fit"] = f"fit failed ({type(exc).__name__}: {exc}); scoring uncalibrated"
                logger.warning("detector %s: fit failed", key, exc_info=True)
        else:
            meta["fit"] = "does not override fit() — calibration-free by construction"
        meta["threshold_after_fit"] = (None if det.threshold is None else float(det.threshold))
        panel[key] = det
        detector_meta.append(meta)
        print(f"  detector {key:14s} ready — {meta['fit']}")
    live_detectors = [k for k in detectors if k in panel]
    if not live_detectors:
        print("\nno detector could be instantiated; nothing to score")
        return 1

    # ---- report skeleton -------------------------------------------------- #
    slug = args.encoder.split("/")[-1].replace(".", "-")
    name = args.name or f"{slug}_{clean_store.similarity}_k{args.k}_q{len(qids)}"
    out_path = REPORT_DIR / f"matrix_{name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "aiforensics/matrix/1",
        "created_at": datetime.now(UTC).isoformat(),
        "config": {
            **vars(args),
            "resolved_attacks": attacks,
            "resolved_detectors": detectors,
            "variant_selector": variant_sel,
            "query_ids": qids,
            "attacked_query_ids": attacked_ids,
            "corpus_size": len(clean_store),
            "encoder_name": clean_store.encoder_name,
            "similarity": clean_store.similarity,
            "normalized_measured": bool(clean_store.normalized),
            "clean_store_path": str(clean_path),
            "n_knob": {a: list(N_KNOB[a]) for a in attacks},
        },
        "clean_reference": clean_ref,
        "detectors": detector_meta,
        "arms": [],
        "notes": [
            "Ground truth is AttackOutcome.ground_truth_mask(store), never "
            "RagStore.injected_mask(): GARAG perturbs documents in place and its injected "
            "mask is all-False.",
            "Every threshold is the (1 - target_fpr) quantile of the CLEAN arm's pooled "
            "scores. No labelled poison was used to pick any operating point.",
            "Scores are pooled raw across queries; nothing is normalised within a retrieved "
            "set, which would destroy cross-query comparability.",
            "Detection metrics are over RETRIEVED passages, not over the corpus: a detector "
            "in this protocol only ever sees the top-k.",
            "detection accuracy == recall on the poisoned class; read it next to FPR and to "
            "the attack-success line, never alone.",
        ],
    }
    done: set[str] = set()
    if args.resume and out_path.exists():
        try:
            prior = json.loads(out_path.read_text())
            report["arms"] = [a for a in prior.get("arms", []) if a.get("status") == "ok"]
            done = {a["key"] for a in report["arms"]}
            print(f"\nresuming: {len(done)} arm(s) already in {out_path}")
        except Exception as exc:
            logger.warning("could not resume from %s: %s", out_path, exc)

    def write_report() -> None:
        out_path.write_text(json.dumps(report, indent=2, default=str))

    # ---- the matrix ------------------------------------------------------- #
    clean_cache: dict[tuple[str, str], np.ndarray] = {}
    planned: set[str] = set()
    for attack in attacks:
        spec = get_attack(attack)
        wanted = variant_sel.get(attack)
        if not wanted:
            fallback = HARNESS_DEFAULT_VARIANT.get(attack, spec.default_variant)
            wanted = [fallback]
        for variant in wanted:
            for n in args.n_poison:
                arm = plan_arm(attack, variant, n, args=args, base_pairs=base_pairs,
                               attacked_ids=attacked_ids, resources=resources)
                if arm.key in planned:
                    continue  # the sweep does not apply to this attack; already run
                planned.add(arm.key)
                if arm.key in done:
                    print(f"\n[{arm.key}] already in the report; skipped (--resume)")
                    continue
                if attack in HARNESS_DEFAULT_VARIANT and not variant_sel.get(attack):
                    arm.notes.append(HARNESS_DEFAULT_REASON[attack])
                run_arm(arm, args=args, spec=spec, clean_path=clean_path, encoder=encoder,
                        clean_store=clean_store, targets=targets, base_pairs=base_pairs,
                        attacked_ids=attacked_ids, resources=resources,
                        clean_answers=clean_answers, gold=gold, panel=panel,
                        live_detectors=live_detectors, detector_meta=detector_meta,
                        clean_cache=clean_cache, report=report)
                write_report()

    ok_arms = [a for a in report["arms"] if a["status"] == "ok"]
    if ok_arms:
        print_summary(ok_arms, live_detectors, args.target_fpr)
    print("\nlegend")
    print("  det.acc  detection accuracy = #detected poisoned / #poisoned (headline metric)")
    print("  clean    flag rate on the CLEAN corpus over the same queries — it should sit at "
          f"the {args.target_fpr:.0%} budget")
    print("  shift    poisoned flag rate - clean flag rate: the corpus-level "
          "'is this DB poisoned?' signal")
    print("  n/a      detector emits a hard decision; a single operating point has no ROC/PR "
          "area to report")
    print("  *        threshold could NOT be calibrated (hard-decision detector), so its "
          "clean flag rate is whatever its paper's rule gives, not the budget")
    print("  nan      undefined, not zero (e.g. precision with nothing flagged, or a "
          "detection rate with no poisoned passage retrieved)")
    print(f"\nwritten: {out_path}")
    return 0


def run_arm(
    arm: Arm,
    *,
    args: argparse.Namespace,
    spec: Any,
    clean_path: Any,
    encoder: Any,
    clean_store: RagStore,
    targets: dict[str, dict],
    base_pairs: Sequence[tuple[str, str]],
    attacked_ids: Sequence[str],
    resources: dict[str, Any],
    clean_answers: dict[str, Any],
    gold: dict[str, list[str]],
    panel: dict[str, Any],
    live_detectors: Sequence[str],
    detector_meta: Sequence[dict],
    clean_cache: dict[tuple[str, str], np.ndarray],
    report: dict[str, Any],
) -> None:
    """Apply one attack, score the whole detector panel against it, record it.

    Every failure mode here is caught and recorded rather than raised: the arms
    are independent by design, so a missing resource or a crashed attack costs
    one row of the matrix and not the run.
    """
    t0 = time.time()
    print(f"\n[{arm.key}] planning …")
    record: dict[str, Any] = {
        "key": arm.key, "attack": arm.attack, "variant": arm.variant, "n": arm.n,
        "status": "skipped", "reason": "", "spec": spec.to_dict(),
        "faithful": bool(spec.get_variant(arm.variant).faithful),
        "notes": list(arm.notes), "cells": [],
        "n_knob": list(N_KNOB[arm.attack]),
    }

    blocker = arm_blocker(arm.attack, arm.variant, args, resources)
    if blocker:
        record["reason"] = blocker
        report["arms"].append(record)
        print(f"[{arm.key}] SKIPPED — {blocker}")
        return

    # ---- apply the attack on a fresh copy of the clean store --------------- #
    store = RagStore.load(clean_path, encoder=encoder)
    arm_targets = {q: dict(targets[q]) for q in targets}
    if arm.attack == "garag":
        for q, ids in gold.items():
            if ids and q in arm_targets:
                arm_targets[q]["doc_ids"] = ids[: arm.n]
    if arm.attack == "inceptionrag" and arm.variant == "full":
        arm.kwargs["respond"] = make_inceptionrag_respond(
            store, resources["model"], resources["tokenizer"], k=args.k,
            max_new_tokens=args.max_new_tokens,
        )
    if arm.attack == "jamming" and arm.variant == "blackbox_optimized":
        arm.kwargs["respond_factory"] = make_respond_factory(
            store, resources["model"], resources["tokenizer"], k=args.k,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        arm.kwargs["clean_responses"] = {q: clean_answers[q].answer for q in attacked_ids
                                         if q in clean_answers}
        arm.kwargs.setdefault("token_pool_texts",
                              [d.text for d in list(clean_store.documents.values())[:2000]])

    try:
        outcome = apply_attack(spec, store, queries=arm.query_pool, targets=arm_targets,
                               config=arm.config, variant=arm.variant,
                               query_ids=arm.query_ids, **arm.kwargs)
    except Exception as exc:
        record["reason"] = f"attack failed: {type(exc).__name__}: {exc}"
        report["arms"].append(record)
        logger.warning("attack %s failed", arm.key, exc_info=True)
        print(f"[{arm.key}] SKIPPED — {record['reason']}")
        return

    record["outcome"] = outcome.to_dict()
    record["notes"].extend(outcome.notes)
    try:
        poison_ids = poison_id_set(outcome, store)
    except Exception as exc:
        record["reason"] = f"ground truth unavailable: {type(exc).__name__}: {exc}"
        report["arms"].append(record)
        print(f"[{arm.key}] SKIPPED — {record['reason']}")
        return
    print(f"[{arm.key}] applied: {len(poison_ids)} poisoned document(s), corpus now "
          f"{len(store):,} docs, {len(outcome.attacked_query_ids)} attacked query/queries")

    # ---- the attack's own success metric ---------------------------------- #
    success, headline, success_warnings = measure_success(
        arm, outcome, args=args, store=store, clean_store=clean_store, targets=targets,
        base_pairs=base_pairs, attacked_ids=attacked_ids, resources=resources,
        clean_answers={q: a.answer for q, a in clean_answers.items()}, gold=gold,
    )
    record["success"] = success
    record["success_headline"] = headline
    record["success_criterion"] = (success or {}).get("criterion", "")
    record["success_warnings"] = success_warnings
    for w in success_warnings:
        print(f"[{arm.key}] attack-success warning: {w}")
    print(f"[{arm.key}] attack success: {headline}")

    # ---- the detector panel ------------------------------------------------ #
    queryset_key = hashlib.sha256(
        "\n".join(q for _, q in arm.eval_pairs).encode()
    ).hexdigest()[:16]
    n_positives, n_scored = count_positives(store, arm.eval_pairs, args.k, poison_ids)
    if n_positives == 0:
        print(f"[{arm.key}] WARNING: no poisoned passage reached the top-{args.k} for any "
              f"evaluation query. Every detection rate below is 0/0 (nan), which is a "
              f"property of this attack run, not of the detectors.")
    cells: list[dict] = []
    for key in live_detectors:
        det = panel[key]
        cell: dict[str, Any] = {"detector": key, "venue": DETECTOR_VENUE.get(key, ""),
                                "status": "skipped", "reason": "", "note": ""}
        t_cell = time.time()
        try:
            cache_key = (key, queryset_key)
            if cache_key not in clean_cache:
                clean_scores, _clean_truth, _ = score_arm(det, clean_store, arm.eval_pairs,
                                                          args.k, set())
                clean_cache[cache_key] = clean_scores
            clean_scores = clean_cache[cache_key]
            scores, y_true, per_query = score_arm(det, store, arm.eval_pairs, args.k, poison_ids)
        except Exception as exc:
            cell["reason"] = f"scoring failed: {type(exc).__name__}: {exc}"
            cells.append(cell)
            logger.warning("detector %s failed on %s", key, arm.key, exc_info=True)
            print(f"  {key:16s} SKIPPED — {cell['reason']}")
            continue

        if scores.size == 0:
            cell["reason"] = "no passage was retrieved for any evaluation query"
            cells.append(cell)
            print(f"  {key:16s} SKIPPED — {cell['reason']}")
            continue

        threshold, calibrated = calibrate(det, clean_scores, args.target_fpr)
        pred = flags(scores, threshold)
        pred_clean = flags(clean_scores, threshold)
        res = detection_metrics(y_true, pred, threshold=threshold, label=key, warn=False)
        roc, pr, n_dropped = auc_pair(y_true, scores)

        note = ""
        if int(y_true.sum()) == 0:
            note = "no poisoned passage reached the top-k: detection undefined (nan)"
        elif not calibrated:
            note = "hard decision; operating point fixed by its paper"
        if n_dropped:
            note = (note + "; " if note else "") + f"{n_dropped} NaN score(s) excluded from AUC"

        cell.update({
            "status": "ok",
            "detection_rate": float(res.detection_rate),
            "precision": float(res.precision),
            "f1": float(res.f1),
            "fpr": float(res.fpr),
            "roc_auc": roc,
            "pr_auc": pr,
            "clean_flag_rate": float(pred_clean.mean()) if pred_clean.size else float("nan"),
            "poisoned_flag_rate": float(pred.mean()) if pred.size else float("nan"),
            "flag_rate_shift": float(pred.mean() - pred_clean.mean())
                               if pred.size and pred_clean.size else float("nan"),
            "threshold": threshold,
            "calibrated_on_clean": bool(calibrated),
            "hard_decision": not calibrated,
            "tp": res.tp, "fp": res.fp, "tn": res.tn, "fn": res.fn,
            "n_positives_retrieved": int(y_true.sum()),
            "n_scored_passages": int(y_true.size),
            "n_nan_scores": int((~np.isfinite(scores)).sum()),
            # Split by class. A detector that declines to judge (Reliability-RAG
            # scores an abstention as NaN) has its NaNs read as "not flagged" by
            # flags(), which is the right conservative choice but makes a cell
            # where EVERY positive abstained print as detection 0.000 -- "missed
            # them all" rather than "judged none of them". jamming/reliability is
            # exactly that: 50 positives, 50 NaN, roc_auc nan, reported 0.000.
            # The collision is structural: jamming's payload makes the generator
            # say "I don't know", which is Reliability-RAG's only abstention
            # marker. Recorded per class so the distinction is visible.
            "n_nan_positives": int((~np.isfinite(scores[y_true])).sum()),
            "n_nan_negatives": int((~np.isfinite(scores[~y_true])).sum()),
            "positives_unassessed": bool(
                y_true.sum() > 0
                and int((~np.isfinite(scores[y_true])).sum()) == int(y_true.sum())
            ),
            "n_unique_scores": int(np.unique(scores[np.isfinite(scores)]).size),
            "seconds": round(time.time() - t_cell, 1),
            "note": note,
            "per_query": per_query,
        })
        cells.append(cell)
        print(f"  {key:16s} det={_fmt(res.detection_rate)} prec={_fmt(res.precision)} "
              f"fpr={_fmt(res.fpr)} roc={_fmt(roc)} clean={_fmt(cell['clean_flag_rate'])} "
              f"({cell['seconds']}s)")

    for meta in detector_meta:
        if not meta.get("available"):
            cells.append({"detector": meta["detector"], "venue": meta.get("venue", ""),
                          "status": "skipped", "reason": meta.get("reason", ""), "note": ""})

    record.update({
        "status": "ok",
        "n_poisoned_docs": len(poison_ids),
        "n_attacked_queries": len(outcome.attacked_query_ids),
        "n_eval_queries": len(arm.eval_pairs),
        "n_positives_retrieved": n_positives,
        "n_scored_passages": n_scored,
        "cells": cells,
        "seconds": round(time.time() - t0, 1),
    })
    report["arms"].append(record)
    print_arm_table(arm, record, cells)


if __name__ == "__main__":
    raise SystemExit(main())
