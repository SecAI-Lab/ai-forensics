"""Corpus-scale scan: run passage-agreement over the whole database.

The retrieved-set benchmark answers "which of these 5 passages is poisoned?".
The forensic question is bigger: *given a seized knowledge base and no idea
whether it was tampered with, which of its 20,000 documents are injected?*

Feasibility. The per-query detector forms the full ``k x k`` cosine matrix, which
is fine for ``k = 5`` and absurd for ``n = 20,000`` (a 20k x 20k float32 matrix is
1.6 GB). But ``max-sim`` only needs each document's single nearest neighbour, and
that is exactly a ``k = 2`` self-search against the FAISS index the store already
holds — the passage itself, plus its closest other document. No dense matrix is
ever built.

The hard part is not compute, it is the **base rate**. At 100 injected documents
in 20,100 the positive class is 0.5%, so a detector with a respectable ROC-AUC can
still produce a useless alarm list. This script therefore reports precision at
several alarm budgets alongside the headline detection accuracy, which is the
honest way to describe corpus-scale triage.

Usage::

    python scripts/run_corpus_scan.py --queries 20 --background 20000
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.metrics.detection import detection_metrics, pr_auc, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def corpus_max_sim(store, batch: int = 2048) -> tuple[np.ndarray, float]:
    """Each document's cosine similarity to its nearest *other* document.

    Uses the store's FAISS index directly: a ``k = 2`` search returns the
    document itself (similarity 1.0) and its closest neighbour, so the second
    hit is the max-sim score. Runs in batches so memory stays O(batch x k).

    Returns:
        ``(scores, seconds)``.
    """
    index = store.index
    n = len(store.doc_ids)
    scores = np.zeros(n, dtype=np.float64)
    t0 = time.time()
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        vecs = np.vstack([index.reconstruct(i) for i in range(start, stop)])
        sims, idx = index.search(vecs, 2)
        # column 0 is normally the document itself; guard against ties
        rows = np.arange(stop - start)
        self_hit = idx[rows, 0] == np.arange(start, stop)
        scores[start:stop] = np.where(self_hit, sims[rows, 1], sims[rows, 0])
    return scores, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20, help="queries attacked")
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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"scan-{len(keep)}")
    clean_path = clean.save("scan-clean")

    poisoned = RagStore.load(clean_path)
    for qid in qids:
        t = targets[qid]
        texts = craft_blackbox(t["question"], list(t["adv_texts"])[: args.n_poison])
        poisoned.add_documents(
            poison_documents(t["question"], t["incorrect answer"], texts,
                             variant="blackbox", query_id=qid,
                             correct_answer=t["correct answer"]),
            injected=True)

    truth = poisoned.injected_mask()
    n_pois = int(truth.sum())
    print(f"corpus: {len(poisoned):,} docs | injected: {n_pois} "
          f"({100 * n_pois / len(poisoned):.3f}% base rate)\n")

    s_p, t_p = corpus_max_sim(poisoned)
    s_c, t_c = corpus_max_sim(clean)
    print(f"scan time: poisoned {t_p:.1f}s ({len(poisoned) / t_p:,.0f} docs/s) | "
          f"clean {t_c:.1f}s")
    print(f"peak memory: O(batch x 2), no {len(poisoned)}x{len(poisoned)} matrix built\n")

    print(f"AUC over the whole corpus: ROC {roc_auc(truth, s_p):.4f}  "
          f"PR {pr_auc(truth, s_p):.4f}")
    print(f"max-sim  poisoned mean {s_p[truth].mean():.4f} | "
          f"clean mean {s_p[~truth].mean():.4f}\n")

    print("=" * 76)
    print("PRECISION AT AN ALARM BUDGET — what a triage analyst actually faces")
    print("=" * 76)
    print(f"{'budget':>8s} {'threshold':>10s} {'flagged':>9s} {'true pos':>9s} "
          f"{'precision':>10s} {'det.acc':>8s}")
    print("-" * 76)
    rows = []
    for budget in (0.0005, 0.001, 0.005, 0.01, 0.05):
        thr = float(np.quantile(s_c, 1.0 - budget))     # calibrated on the CLEAN corpus
        pred = s_p >= thr
        res = detection_metrics(truth, pred, warn=False)
        rows.append({"budget": budget, "threshold": thr, "flagged": int(pred.sum()),
                     "true_positives": int((pred & truth).sum()),
                     "precision": float(res.precision),
                     "detection_rate": float(res.detection_rate)})
        print(f"{budget:8.2%} {thr:10.4f} {int(pred.sum()):9,d} "
              f"{int((pred & truth).sum()):9d} {res.precision:10.3f} "
              f"{res.detection_rate:8.3f}")

    print("\n--- the same scan on the CLEAN corpus (no poison exists) ---")
    for budget in (0.001, 0.01):
        thr = float(np.quantile(s_c, 1.0 - budget))
        print(f"  budget {budget:.1%}: {int((s_c >= thr).sum()):,} documents flagged, "
              f"all of them false alarms by construction")

    out = REPORT_DIR / "corpus_scan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args), "corpus_size": len(poisoned), "n_poisoned": n_pois,
        "roc_auc": float(roc_auc(truth, s_p)), "pr_auc": float(pr_auc(truth, s_p)),
        "scan_seconds": float(t_p), "budgets": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
