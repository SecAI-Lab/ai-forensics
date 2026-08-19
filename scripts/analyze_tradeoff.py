"""What does reducing false alarms actually cost?

"Minimise false alarms" is not the objective on its own — flagging nothing has a
false-alarm rate of zero and a detection accuracy of zero. The real question is
where on the trade-off curve to sit, and that depends on what happens to an
alarm after it is raised.

This sweeps the alarm budget at corpus scale and reports, for each operating
point: how many documents an analyst must read, how many of them are poisoned,
how much of the poison is found, and the resulting effort per true positive.
The last column is the one that matters operationally — it is the number of
documents read to surface one genuine finding.

Usage::

    python scripts/analyze_tradeoff.py --queries 20 --background 20000
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
from aiforensics.metrics.detection import detection_metrics, pr_auc, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def corpus_max_sim(store, batch: int = 2048) -> np.ndarray:
    """Cosine similarity of each document to its nearest *other* document."""
    index = store.index
    n = len(store.doc_ids)
    out = np.zeros(n, dtype=np.float64)
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        vecs = np.vstack([index.reconstruct(i) for i in range(start, stop)])
        sims, idx = index.search(vecs, 2)
        rows = np.arange(stop - start)
        self_hit = idx[rows, 0] == np.arange(start, stop)
        out[start:stop] = np.where(self_hit, sims[rows, 1], sims[rows, 0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--n-poison", type=int, default=5)
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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"tr-{len(keep)}")
    path = clean.save("tr-clean")
    poisoned = RagStore.load(path)
    for qid in qids:
        t = targets[qid]
        texts = craft_blackbox(t["question"], list(t["adv_texts"])[: args.n_poison])
        poisoned.add_documents(
            poison_documents(t["question"], t["incorrect answer"], texts,
                             variant="blackbox", query_id=qid,
                             correct_answer=t["correct answer"]),
            injected=True)

    truth = poisoned.injected_mask()
    s_p, s_c = corpus_max_sim(poisoned), corpus_max_sim(clean)
    n, n_pois = len(poisoned), int(truth.sum())
    print(f"corpus {n:,} docs | {n_pois} injected ({100*n_pois/n:.3f}% base rate)")
    print(f"ROC-AUC {roc_auc(truth, s_p):.4f}   PR-AUC {pr_auc(truth, s_p):.4f}\n")

    print("=" * 92)
    print("THE TRADE-OFF — every row is a choice, none of them is 'the answer'")
    print("=" * 92)
    print(f"{'budget':>8s} {'to read':>8s} {'found':>6s} {'missed':>7s} "
          f"{'det.acc':>8s} {'precision':>10s} {'reads per find':>15s}")
    print("-" * 92)
    rows = []
    for budget in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        thr = float(np.quantile(s_c, 1.0 - budget))
        pred = s_p >= thr
        tp = int((pred & truth).sum())
        flagged = int(pred.sum())
        res = detection_metrics(truth, pred, warn=False)
        eff = flagged / tp if tp else float("inf")
        rows.append({"budget": budget, "flagged": flagged, "true_positives": tp,
                     "detection_rate": float(res.detection_rate),
                     "precision": float(res.precision), "reads_per_find": eff})
        eff_s = f"{eff:15.1f}" if np.isfinite(eff) else f"{'never finds':>15s}"
        print(f"{budget:8.2%} {flagged:8,d} {tp:6d} {n_pois - tp:7d} "
              f"{res.detection_rate:8.3f} {res.precision:10.3f}{eff_s}")

    finite = [r for r in rows if np.isfinite(r["reads_per_find"])]
    best = min(finite, key=lambda r: r["reads_per_find"]) if finite else None
    print()
    if best:
        print(f"most efficient point: budget {best['budget']:.2%} -> "
              f"{best['reads_per_find']:.1f} documents read per poisoned document found "
              f"(finds {best['detection_rate']:.0%} of the poison)")
    print()
    print("Reading this table:")
    print("  * Driving the budget down does NOT 'win' — below 0.2% the detector finds")
    print("    nothing at all, because genuine near-duplicate passages saturate the top")
    print("    of the clean score distribution.")
    print("  * Driving it up does not win either: recall climbs but the analyst reads")
    print("    ten or more documents per real finding.")
    print("  * The right operating point depends on what an alarm costs. A cheap")
    print("    automated second stage justifies a loose budget and high recall; a human")
    print("    analyst justifies a tight one.")

    out = REPORT_DIR / "tradeoff.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "corpus": n, "poisoned": n_pois,
                               "roc_auc": float(roc_auc(truth, s_p)),
                               "pr_auc": float(pr_auc(truth, s_p)),
                               "points": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
