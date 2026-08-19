"""The RAG poisoning **detection** benchmark.

Scores every registered ``rag-defense`` detector on the same corpus, the same
queries and the same injected passages, under two conditions the investigator
cannot tell apart in advance:

* **clean**    — no poison anywhere. Every flag is a false alarm.
* **poisoned** — PoisonedRAG injected for a subset of the queries, so some
  queries are attacked and some are not, which is the realistic case.

Headline metric is the project's: ``detection accuracy = #detected poisoned /
#poisoned`` (recall on the poisoned class), always reported next to precision,
F1 and FPR — a detector that flags everything scores a perfect detection rate
and is useless, and the FPR column is what exposes that.

Score-based detectors additionally get ROC-AUC and PR-AUC, which are
threshold-free and therefore the fair way to compare them against a detector
that only emits a hard decision. Decision-only detectors report ``n/a`` there,
and that difference is itself a result: only a score lets you calibrate an
alarm threshold when you have no labelled poison.

Usage::

    python scripts/run_detection_benchmark.py --queries 40 --attacked 20
    python scripts/run_detection_benchmark.py --detectors ragdefender knn-isolation
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, craft_whitebox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector, list_detectors
from aiforensics.metrics.detection import detection_metrics, pr_auc, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def build_corpus(qids, n_background, encoder):
    """Corpus containing every gold passage for the targets, plus background."""
    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=n_background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < n_background:
            keep.append(doc); bg += 1
    found = sum(1 for d in keep if d.doc_id in gold)
    print(f"corpus: {len(keep)} docs ({found}/{len(gold)} gold passages present)")
    return build_index(keep, encoder=encoder, corpus_name=f"bench-{len(keep)}")


def evaluate(detector, store, targets, qids, k):
    """Collect per-passage scores and ground truth across all queries."""
    scores, truth, per_query = [], [], []
    for qid in qids:
        q = targets[qid]["question"]
        hits = store.search(q, k=k)
        texts = [h.text for h in hits]
        s = np.asarray(detector.score((q, texts)), dtype=np.float64).reshape(-1)
        t = np.array([h.injected for h in hits], dtype=bool)
        scores.append(s); truth.append(t)
        per_query.append({"qid": qid, "n_poison_at_k": int(t.sum()), "scores": s.tolist()})
    return np.concatenate(scores), np.concatenate(truth), per_query


def calibrate(detector, clean_scores, target_fpr):
    """Pick a decision threshold using **clean data only**.

    This is the protocol the unknown-poisoning setting forces. The investigator
    has no labelled poison, so a threshold cannot be tuned on the attacked data
    without leaking the answer. What is obtainable is a corpus believed clean
    (or a held-out slice of one), which fixes an alarm budget: set the threshold
    at the ``1 - target_fpr`` quantile of the clean scores, so a clean database
    raises alarms on ``target_fpr`` of passages by construction. Whatever the
    poisoned database then raises above that is signal.

    Returns ``(threshold, calibrated)`` where ``calibrated`` is False for a
    detector that emits its own hard decisions and cannot be moved.
    """
    if detector.threshold is not None and len(np.unique(clean_scores)) <= 2:
        return float(detector.threshold), False       # decision-only detector
    return float(np.quantile(clean_scores, 1.0 - target_fpr)), True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--attacked", type=int, default=20)
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, default=5)
    ap.add_argument("--variant", choices=["blackbox", "whitebox"], default="blackbox")
    ap.add_argument("--detectors", nargs="*", default=None,
                    help="registry keys; default = every rag-defense detector")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--target-fpr", type=float, default=0.05,
                    help="alarm budget on CLEAN data used to calibrate each threshold")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    attacked = set(qids[: args.attacked])

    clean = build_corpus(qids, args.background, args.encoder)
    clean_path = clean.save("bench-clean")
    poisoned = RagStore.load(clean_path)
    for qid in sorted(attacked):
        t = targets[qid]
        base = list(t["adv_texts"])[: args.n_poison]
        texts = (craft_blackbox(t["question"], base) if args.variant == "blackbox"
                 else craft_whitebox(poisoned.encoder, t["question"], base))
        poisoned.add_documents(
            poison_documents(t["question"], t["incorrect answer"], texts,
                             variant=args.variant, query_id=qid,
                             correct_answer=t["correct answer"]),
            injected=True,
        )
    print(f"poisoned store: {int(poisoned.injected_mask().sum())} injected docs "
          f"for {len(attacked)}/{len(qids)} queries ({args.variant})\n")

    keys = args.detectors or list_detectors("rag-defense")
    print(f"detectors: {keys}\n")

    model = tok = None
    # hybrid fuses perplexity by default, so it needs the LM too.
    if any(k in {"perplexity", "hybrid", "npas", "reliability", "consensus",
                 "cascade", "query-gate", "fdr-control"} for k in keys):
        from aiforensics.models.llm import load_llm
        model, tok = load_llm(args.generator)

    rows = []
    for key in keys:
        try:
            det = get_detector(key, store=poisoned, encoder=poisoned.encoder,
                               model=model, tokenizer=tok)
        except TypeError:  # detector takes no store/model kwargs
            det = get_detector(key)
        except Exception as exc:
            print(f"  !! {key}: could not instantiate ({exc})")
            continue

        # A fusion detector must be calibrated on clean data first, otherwise it
        # falls back to per-call rank normalisation and its scores carry no
        # cross-query information at all (the pooled ROC collapses to chance).
        if hasattr(det, "fit") and (getattr(det, "members", None)
                                    or getattr(det, "requires_fit", False)
                                    or hasattr(det, "_clean")
                                    or key in {"npas", "reliability", "gmtp"}):
            try:
                det.fit([(targets[q]["question"],
                          [h.text for h in clean.search(targets[q]["question"], k=args.k)])
                         for q in qids])
            except Exception as exc:
                print(f"  .. {key}: fit skipped ({exc}); scoring uncalibrated")
        # a detector that cannot calibrate still gets scored uncalibrated

        s_p, t_p, _ = evaluate(det, poisoned, targets, qids, args.k)
        s_c, _t_c, _ = evaluate(det, clean, targets, qids, args.k)

        thr, calibrated = calibrate(det, s_c, args.target_fpr)
        pred_p, pred_c = s_p >= thr, s_c >= thr
        res = detection_metrics(t_p, pred_p, warn=False)
        binary_only = len(np.unique(s_p)) <= 2
        auc = None if binary_only else roc_auc(t_p, s_p)
        prauc = None if binary_only else pr_auc(t_p, s_p)

        row = {
            "detector": key,
            "detection_rate": float(res.detection_rate),
            "precision": float(res.precision),
            "f1": float(res.f1),
            "fpr": float(res.fpr),
            "roc_auc": None if auc is None else float(auc),
            "pr_auc": None if prauc is None else float(prauc),
            "clean_flag_rate": float(pred_c.mean()),
            "poisoned_flag_rate": float(pred_p.mean()),
            "flag_rate_shift": float(pred_p.mean() - pred_c.mean()),
            "binary_only": bool(binary_only),
            "threshold": float(thr),
            "calibrated_on_clean": bool(calibrated),
        }
        rows.append(row)
        print(f"  {key:18s} det={row['detection_rate']:.3f} prec={row['precision']:.3f} "
              f"fpr={row['fpr']:.3f} auc={'n/a' if auc is None else f'{auc:.3f}'} "
              f"clean_flags={row['clean_flag_rate']:.3f}")

    rows.sort(key=lambda r: (-(r["roc_auc"] or 0), -r["detection_rate"]))

    print("\n" + "=" * 100)
    print(f"DETECTION BENCHMARK — PoisonedRAG {args.variant}, k={args.k}, "
          f"N={args.n_poison}, {args.attacked}/{args.queries} queries attacked")
    print("=" * 100)
    h = (f"{'detector':18s} {'det.acc':>8s} {'prec':>6s} {'F1':>6s} {'FPR':>6s} "
         f"{'ROC':>6s} {'PR':>6s} | {'clean':>6s} {'pois':>6s} {'shift':>6s}")
    print(h); print("-" * len(h))
    for r in rows:
        auc = "n/a" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
        pra = "n/a" if r["pr_auc"] is None else f"{r['pr_auc']:.3f}"
        mark = "" if r["calibrated_on_clean"] else "*"
        print(f"{r['detector']+mark:18s} {r['detection_rate']:8.3f} {r['precision']:6.3f} "
              f"{r['f1']:6.3f} {r['fpr']:6.3f} {auc:>6s} {pra:>6s} | "
              f"{r['clean_flag_rate']:6.3f} {r['poisoned_flag_rate']:6.3f} "
              f"{r['flag_rate_shift']:+6.3f}")
    print(f"\nthresholds calibrated on CLEAN data at a {args.target_fpr:.0%} alarm budget "
          f"(no labelled poison used)")
    print("det.acc = detection accuracy = #detected poisoned / #poisoned (the headline metric)")
    print("clean   = fraction flagged on a CLEAN database — should sit at the alarm budget")
    print("shift   = poisoned flag rate - clean flag rate; this is the corpus-level")
    print("          'is this DB poisoned at all?' signal. Large shift + small clean = usable.")
    print("* = detector emits hard decisions, so its threshold could NOT be calibrated")

    out = REPORT_DIR / f"detection_benchmark_{args.variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
