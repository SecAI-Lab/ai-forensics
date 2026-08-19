"""How does cross-passage agreement behave when the poison is sparse?

The detector compares **every pair** of retrieved passages — ``emb @ emb.T`` is
the full ``n x n`` cosine matrix, i.e. all ``C(n, 2)`` unique pairs. The open
question is how the pairwise matrix is *aggregated* into one score per passage,
and that choice decides whether sparse poison is visible at all:

``mean-sim``
    Average similarity to the other ``n - 1`` passages. With 2 poisoned among
    100, each poisoned passage averages one high similarity (its twin) against
    98 unrelated ones, so the signal is divided by ~99 and vanishes.
``max-sim``
    Similarity to the single nearest other passage. The twin is found regardless
    of how many unrelated passages surround it, so the score is invariant to
    ``n``. This is the default.
``cluster-size``
    Size of the passage's single-linkage cluster, normalised by ``n``. Finds the
    group but its scale shrinks as ``n`` grows.

This script sweeps the retrieved-set size ``n`` at a fixed, small poison count
and reports each mode's separation and AUC, so the dilution effect is measured
rather than asserted.

Usage::

    python scripts/analyze_agreement_scaling.py --k 100 --n-poison 2
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"
MODES = ("mean-sim", "max-sim", "cluster-size")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    ap.add_argument("--n-poison", type=int, default=2)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"scale-{len(keep)}")
    clean_path = clean.save("scale-clean")

    store = RagStore.load(clean_path)
    for qid in qids:
        t = targets[qid]
        texts = craft_blackbox(t["question"], list(t["adv_texts"])[: args.n_poison])
        store.add_documents(
            poison_documents(t["question"], t["incorrect answer"], texts,
                             variant="blackbox", query_id=qid,
                             correct_answer=t["correct answer"]),
            injected=True)
    print(f"corpus {len(store)} docs | {args.n_poison} poison per query, "
          f"{len(qids)} queries attacked\n")

    rows = []
    for mode in MODES:
        det = get_detector("passage-agreement", store=store, encoder=store.encoder, mode=mode)
        for k in args.k_values:
            S, T = [], []
            for qid in qids:
                q = targets[qid]["question"]
                hits = store.search(q, k=k)
                S.append(np.asarray(det.score((q, [h.text for h in hits])),
                                    dtype=np.float64).reshape(-1))
                T.append(np.array([h.injected for h in hits], dtype=bool))
            s, t = np.concatenate(S), np.concatenate(T)
            if t.sum() == 0 or (~t).sum() == 0:
                continue
            # threshold calibrated on the CLEAN passages of the same run, 5% budget
            thr = float(np.quantile(s[~t], 0.95))
            res = detection_metrics(t, s >= thr, warn=False)
            rows.append({
                "mode": mode, "k": k,
                "poison_mean": float(s[t].mean()), "clean_mean": float(s[~t].mean()),
                "separation": float(s[t].mean() - s[~t].mean()),
                "auc": float(roc_auc(t, s)),
                "detection_rate": float(res.detection_rate),
                "retrieved_poison": float(np.mean([tq.sum() for tq in T])),
            })

    print("=" * 88)
    print(f"AGGREGATION vs RETRIEVED-SET SIZE  ({args.n_poison} poison per query, "
          f"5% clean alarm budget)")
    print("=" * 88)
    h = (f"{'mode':13s} {'k':>4s} {'poison@k':>9s} {'poison':>8s} {'clean':>8s} "
         f"{'separat.':>9s} {'AUC':>6s} {'det.acc':>8s}")
    print(h); print("-" * len(h))
    for mode in MODES:
        for r in [x for x in rows if x["mode"] == mode]:
            print(f"{r['mode']:13s} {r['k']:4d} {r['retrieved_poison']:9.2f} "
                  f"{r['poison_mean']:8.3f} {r['clean_mean']:8.3f} {r['separation']:9.3f} "
                  f"{r['auc']:6.3f} {r['detection_rate']:8.3f}")
        print()

    print("Reading: 'separat.' is mean(poison score) - mean(clean score). If it shrinks as k")
    print("grows, that aggregation is diluting the signal; if it holds, the mode is")
    print(f"invariant to how many unrelated passages surround the {args.n_poison} poisoned ones.")

    out = REPORT_DIR / "agreement_scaling.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
