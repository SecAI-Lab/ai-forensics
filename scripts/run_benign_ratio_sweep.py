"""Does RAGDefender still work when the attacker floods the top-k?

RAGDefender (ACSAC'25) groups the retrieved passages and identifies the
adversarial group. That presupposes there *is* a benign group to contrast
against: with zero benign passages in the top-k there is nothing to separate,
and the two-stage design is outside its stated assumption.

A real attacker chooses ``N``. With ``N = k`` a fully successful PoisonedRAG
leaves no benign passage at all, so the in-the-wild regime routinely violates
the assumption. This sweep measures where the boundary actually is: vary ``N``
against a fixed ``k``, then report detection **stratified by how many benign
passages survived retrieval**, keeping the paper's regime (>=1 benign) separate
from the flooded regime (0 benign).

Usage::

    python scripts/run_benign_ratio_sweep.py --queries 20 --k 5 --n-poison 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import PoisonedRAGConfig, craft_blackbox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics
from aiforensics.models.llm import load_llm
from aiforensics.pipeline import RagPipeline, answer_matches

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]

    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=args.background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < args.background:
            keep.append(doc); bg += 1
    print(f"corpus: {len(keep)} docs ({sum(1 for d in keep if d.doc_id in gold)}/{len(gold)} gold)")
    base = build_index(keep, encoder=args.encoder, corpus_name=f"sweep-{len(keep)}")
    base_path = base.save("sweep-clean")

    model, tok = load_llm(args.generator)
    defender = get_detector("ragdefender", task_type="single_hop")

    rows, strata = [], defaultdict(list)
    for n in args.n_poison:
        PoisonedRAGConfig(n_poison=n, seed=args.seed)
        store = RagStore.load(base_path)
        for qid in qids:
            t = targets[qid]
            texts = craft_blackbox(t["question"], list(t["adv_texts"])[:n])
            store.add_documents(
                poison_documents(t["question"], t["incorrect answer"], texts,
                                 variant="blackbox", query_id=qid,
                                 correct_answer=t["correct answer"]),
                injected=True,
            )

        pipe = RagPipeline(store, model, tok, k=args.k)
        questions = [targets[q]["question"] for q in qids]
        undef = pipe.answer(questions)
        defended = pipe.answer(questions, filter_fn=defender.keep_indices)

        per_q = []
        for qid, a_u, a_d in zip(qids, undef, defended, strict=False):
            t = targets[qid]
            hits = store.search(t["question"], k=args.k)
            truth = np.array([h.injected for h in hits], dtype=bool)
            pred = np.asarray(defender.score((t["question"], [h.text for h in hits])), dtype=bool)
            res = detection_metrics(truth, pred, warn=False)
            n_benign = int((~truth).sum())
            per_q.append((n_benign, res, a_u, a_d, t))
            strata[n_benign].append(res)

        n_poison_at_k = np.mean([int(np.sum([h.injected for h in store.search(
            targets[q]["question"], k=args.k)])) for q in qids])
        flooded = sum(1 for nb, *_ in per_q if nb == 0)
        rows.append({
            "N": n,
            "poison_at_k": float(n_poison_at_k),
            "queries_fully_flooded": flooded,
            "frac_flooded": flooded / len(qids),
            "ASR_undefended": float(np.mean([answer_matches(a.answer, t["incorrect answer"])
                                             for _, _, a, _, t in per_q])),
            "ASR_defended": float(np.mean([answer_matches(a.answer, t["incorrect answer"])
                                           for _, _, _, a, t in per_q])),
            "detection_rate": float(np.nanmean([r.detection_rate for _, r, *_ in per_q])),
            "precision": float(np.nanmean([r.precision for _, r, *_ in per_q])),
            "fpr": float(np.nanmean([r.fpr for _, r, *_ in per_q])),
        })
        print(f"  N={n}: poison@k={n_poison_at_k:.2f} flooded={flooded}/{len(qids)} "
              f"ASR {rows[-1]['ASR_undefended']:.2f}->{rows[-1]['ASR_defended']:.2f} "
              f"det={rows[-1]['detection_rate']:.3f}")

    print("\n=== per-N ===")
    h = f"{'N':>2s} {'poison@k':>9s} {'flooded':>8s} {'ASR':>6s} {'ASR+def':>8s} {'det.rate':>9s} {'prec':>6s} {'FPR':>6s}"
    print(h); print("-" * len(h))
    for r in rows:
        print(f"{r['N']:2d} {r['poison_at_k']:9.2f} {r['queries_fully_flooded']:3d}/{args.queries:<4d} "
              f"{r['ASR_undefended']:6.2f} {r['ASR_defended']:8.2f} "
              f"{r['detection_rate']:9.3f} {r['precision']:6.3f} {r['fpr']:6.3f}")

    print("\n=== stratified by benign passages surviving in the top-k ===")
    print("(0 benign = RAGDefender's grouping assumption is violated)")
    h2 = f"{'#benign in top-k':>17s} {'queries':>8s} {'det.rate':>9s} {'prec':>6s} {'FPR':>6s}"
    print(h2); print("-" * len(h2))
    strat_out = []
    for nb in sorted(strata):
        rs = strata[nb]
        dr = float(np.nanmean([r.detection_rate for r in rs]))
        pr = float(np.nanmean([r.precision for r in rs]))
        fp = float(np.nanmean([r.fpr for r in rs]))
        flag = "  <-- assumption violated" if nb == 0 else ""
        print(f"{nb:17d} {len(rs):8d} {dr:9.3f} {pr:6.3f} {fp:6.3f}{flag}")
        strat_out.append({"n_benign": nb, "queries": len(rs),
                          "detection_rate": dr, "precision": pr, "fpr": fp})

    within = [r for nb in strata if nb >= 1 for r in strata[nb]]
    violated = strata.get(0, [])
    print()
    if within:
        print(f"  paper regime  (>=1 benign, n={len(within)}): "
              f"detection_rate={np.nanmean([r.detection_rate for r in within]):.3f} "
              f"precision={np.nanmean([r.precision for r in within]):.3f}")
    if violated:
        print(f"  flooded regime ( 0 benign, n={len(violated)}): "
              f"detection_rate={np.nanmean([r.detection_rate for r in violated]):.3f} "
              f"precision={np.nanmean([r.precision for r in violated]):.3f}")

    out = REPORT_DIR / "ragdefender_benign_ratio_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "per_N": rows,
                               "stratified": strat_out}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
