"""Why does passage-agreement beat RAGDefender — better signal, or just a knob?

RAGDefender emits a hard keep/drop decision; passage-agreement emits a continuous
score. Comparing them at their default operating points is not a fair contest,
because only one of them has an operating point that can be moved. This script
separates the two possible explanations:

1. **Output format.** A score can be calibrated to any alarm budget, so it wins
   any comparison where the budget matters. If that is the whole story, then at a
   *matched* false-alarm rate the two should detect about equally well.
2. **Signal quality.** If agreement still wins when both are forced to the same
   false-alarm rate, the underlying statistic really is more informative.

The decisive experiment is therefore: calibrate agreement's threshold on clean
data so its clean flag rate equals RAGDefender's, then compare detection rates.

It also measures the flooding mechanism directly. RAGDefender's grouping stage
needs at least one benign passage to contrast against; agreement needs the
opposite — the more the attacker floods, the tighter the injected cluster and the
higher every poisoned score. The two methods should therefore move in opposite
directions as poison density rises, and that is checked here rather than argued.

Usage::

    python scripts/analyze_agreement_vs_ragdefender.py --queries 40
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def collect(det, store, targets, qids, k):
    """Per-query scores, ground truth and poison density."""
    S, T = [], []
    for qid in qids:
        q = targets[qid]["question"]
        hits = store.search(q, k=k)
        S.append(np.asarray(det.score((q, [h.text for h in hits])),
                            dtype=np.float64).reshape(-1))
        T.append(np.array([h.injected for h in hits], dtype=bool))
    return S, T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--attacked", type=int, default=20)
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    attacked = set(qids[: args.attacked])

    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=args.background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < args.background:
            keep.append(doc); bg += 1
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"why-{len(keep)}")
    clean_path = clean.save("why-clean")
    print(f"corpus: {len(keep)} docs\n")

    agree = get_detector("passage-agreement", store=clean, encoder=clean.encoder)
    rdef = get_detector("ragdefender", task_type="single_hop")

    # ---- clean reference distributions -------------------------------------
    Sa_c, _ = collect(agree, clean, targets, qids, args.k)
    Sr_c, _ = collect(rdef, clean, targets, qids, args.k)
    a_clean, r_clean = np.concatenate(Sa_c), np.concatenate(Sr_c)
    rdef_clean_fpr = float((r_clean >= 0.5).mean())
    print(f"RAGDefender clean flag rate      : {rdef_clean_fpr:.3f}   (its fixed operating point)")

    # agreement threshold matched to RAGDefender's clean flag rate
    thr_matched = float(np.quantile(a_clean, 1.0 - rdef_clean_fpr))
    thr_budget = float(np.quantile(a_clean, 1.0 - 0.05))
    print(f"agreement threshold @matched FPR : {thr_matched:.4f}")
    print(f"agreement threshold @5% budget   : {thr_budget:.4f}\n")

    rows, dens = [], defaultdict(lambda: {"agree": [], "rdef": []})
    for n in args.n_poison:
        store = RagStore.load(clean_path)
        for qid in sorted(attacked):
            t = targets[qid]
            texts = craft_blackbox(t["question"], list(t["adv_texts"])[:n])
            store.add_documents(
                poison_documents(t["question"], t["incorrect answer"], texts,
                                 variant="blackbox", query_id=qid,
                                 correct_answer=t["correct answer"]),
                injected=True)

        agree_n = get_detector("passage-agreement", store=store, encoder=store.encoder)
        Sa, Ta = collect(agree_n, store, targets, qids, args.k)
        Sr, _ = collect(rdef, store, targets, qids, args.k)
        a, r, t_all = np.concatenate(Sa), np.concatenate(Sr), np.concatenate(Ta)

        res_match = detection_metrics(t_all, a >= thr_matched, warn=False)
        res_budget = detection_metrics(t_all, a >= thr_budget, warn=False)
        res_rdef = detection_metrics(t_all, r >= 0.5, warn=False)

        # separation of poisoned vs clean scores, per density bucket
        for s_q, t_q in zip(Sa, Ta, strict=True):
            dens[int(t_q.sum())]["agree"].append(
                (float(s_q[t_q].mean()) if t_q.any() else np.nan,
                 float(s_q[~t_q].mean()) if (~t_q).any() else np.nan))
        for s_q, t_q in zip(Sr, Ta, strict=True):
            dens[int(t_q.sum())]["rdef"].append(
                (float(s_q[t_q].mean()) if t_q.any() else np.nan,
                 float(s_q[~t_q].mean()) if (~t_q).any() else np.nan))

        rows.append({
            "N": n,
            "agree_auc": float(roc_auc(t_all, a)),
            "agree_matched_det": float(res_match.detection_rate),
            "agree_matched_fpr": float(res_match.fpr),
            "agree_budget_det": float(res_budget.detection_rate),
            "agree_budget_fpr": float(res_budget.fpr),
            "rdef_det": float(res_rdef.detection_rate),
            "rdef_fpr": float(res_rdef.fpr),
        })
        print(f"  N={n}: agree(matched)={res_match.detection_rate:.3f} "
              f"agree(5%)={res_budget.detection_rate:.3f} "
              f"ragdefender={res_rdef.detection_rate:.3f}  AUC={rows[-1]['agree_auc']:.3f}")

    print("\n" + "=" * 92)
    print("A) SAME FALSE-ALARM RATE — is the advantage the signal, or just the knob?")
    print("=" * 92)
    h = (f"{'N':>2s} | {'agreement @matched FPR':>24s} | {'RAGDefender':>20s} | "
         f"{'agreement @5% budget':>22s}")
    print(h)
    print(f"{'':>2s} | {'det':>10s} {'FPR':>12s} | {'det':>8s} {'FPR':>10s} | {'det':>9s} {'FPR':>11s}")
    print("-" * len(h))
    for r in rows:
        print(f"{r['N']:2d} | {r['agree_matched_det']:10.3f} {r['agree_matched_fpr']:12.3f} | "
              f"{r['rdef_det']:8.3f} {r['rdef_fpr']:10.3f} | "
              f"{r['agree_budget_det']:9.3f} {r['agree_budget_fpr']:11.3f}")

    print("\n" + "=" * 92)
    print("B) FLOODING — the two methods move in OPPOSITE directions")
    print("=" * 92)
    print(f"{'#poison in top-k':>17s} {'agree(poison)':>14s} {'agree(clean)':>13s} "
          f"{'separation':>11s} | {'rdef(poison)':>13s}")
    print("-" * 78)
    for nb in sorted(dens):
        a_pairs = [p for p in dens[nb]["agree"] if not np.isnan(p[0])]
        r_pairs = [p for p in dens[nb]["rdef"] if not np.isnan(p[0])]
        if not a_pairs:
            continue
        ap_ = float(np.nanmean([p[0] for p in a_pairs]))
        ac_ = float(np.nanmean([p[1] for p in a_pairs]))
        rp_ = float(np.nanmean([p[0] for p in r_pairs])) if r_pairs else np.nan
        sep = ap_ - ac_ if not np.isnan(ac_) else np.nan
        sep_s = f"{sep:11.3f}" if not np.isnan(sep) else f"{'n/a (all poison)':>11s}"
        print(f"{nb:17d} {ap_:14.3f} {ac_:13.3f} {sep_s} | {rp_:13.3f}")
    print("\nagree(poison) rising with density = the injected cluster tightens, so every")
    print("member scores higher. RAGDefender needs a benign group to contrast against,")
    print("so it loses exactly what agreement gains.")

    out = REPORT_DIR / "agreement_vs_ragdefender.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args),
                               "ragdefender_clean_fpr": rdef_clean_fpr,
                               "per_N": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
