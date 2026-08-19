"""Which hybrid members actually earn their place?

The default fusion is ``passage-agreement + query-similarity + perplexity``. Two
of those look weak alone (perplexity scores 0.000 against black-box poison), so
the question is whether they contribute anything once fused, and whether the two
later-added published detectors (CleanBase, GMTP) improve the ensemble further.

Each combination is scored in the same wild setting — mixed black-box/white-box
attacks, poison count varying per query, half the queries never attacked — with
every member normalised against its own clean-corpus distribution and the
decision threshold calibrated on clean data at a fixed alarm budget.

Usage::

    python scripts/run_hybrid_ablation.py --queries 40 --attacked 20
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
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"

#: Candidate members, shortened for the table.
SHORT = {"passage-agreement": "agree", "query-similarity": "qsim", "perplexity": "ppl",
         "cleanbase": "clean", "gmtp": "gmtp"}

COMBOS = [
    ("passage-agreement",),
    ("query-similarity",),
    ("perplexity",),
    ("cleanbase",),
    ("gmtp",),
    ("passage-agreement", "query-similarity"),
    ("passage-agreement", "query-similarity", "perplexity"),          # current default
    ("passage-agreement", "query-similarity", "cleanbase"),
    ("passage-agreement", "query-similarity", "perplexity", "gmtp"),
    ("passage-agreement", "query-similarity", "perplexity", "cleanbase"),
    ("passage-agreement", "query-similarity", "perplexity", "cleanbase", "gmtp"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--attacked", type=int, default=20)
    ap.add_argument("--background", type=int, default=12000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-range", type=int, nargs=2, default=[1, 8])
    ap.add_argument("--whitebox-frac", type=float, default=0.5)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--combine", choices=["mean", "max", "median"], default="mean")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    attacked = sorted(rng.choice(qids, size=args.attacked, replace=False).tolist())

    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=args.background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < args.background:
            keep.append(doc); bg += 1
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"abl-{len(keep)}")
    clean_path = clean.save("abl-clean")
    poisoned = RagStore.load(clean_path)

    plan = {}
    for qid in attacked:
        t = targets[qid]
        n = int(rng.integers(args.n_range[0], args.n_range[1] + 1))
        variant = "whitebox" if rng.random() < args.whitebox_frac else "blackbox"
        base = list(t["adv_texts"])
        base = [base[i % len(base)] for i in range(n)]
        texts = (craft_blackbox(t["question"], base) if variant == "blackbox"
                 else craft_whitebox(poisoned.encoder, t["question"], base))
        poisoned.add_documents(
            poison_documents(t["question"], t["incorrect answer"], texts, variant=variant,
                             query_id=qid, correct_answer=t["correct answer"]),
            injected=True)
        plan[qid] = variant
    print(f"corpus {len(poisoned)} | {int(poisoned.injected_mask().sum())} injected across "
          f"{len(attacked)}/{len(qids)} queries\n")

    from aiforensics.models.llm import load_llm
    model, tok = load_llm(args.generator)

    clean_samples = [(targets[q]["question"],
                      [h.text for h in clean.search(targets[q]["question"], k=args.k)])
                     for q in qids]

    rows = []
    for combo in COMBOS:
        try:
            det = get_detector("hybrid", members=combo, combine=args.combine,
                               store=poisoned, encoder=poisoned.encoder,
                               model=model, tokenizer=tok)
            det.fit(clean_samples)
        except Exception as exc:
            print(f"  !! {combo}: {exc}")
            continue

        def gather(store, det=det):
            S, T, V = [], [], []
            for qid in qids:
                q = targets[qid]["question"]
                hits = store.search(q, k=args.k)
                S.append(np.asarray(det.score((q, [h.text for h in hits])),
                                    dtype=np.float64).reshape(-1))
                T.append(np.array([h.injected for h in hits], dtype=bool))
                V.append(plan.get(qid, "none"))
            return S, T, V

        Sp, Tp, V = gather(poisoned)
        Sc, _, _ = gather(clean)
        s_p, t_p, s_c = np.concatenate(Sp), np.concatenate(Tp), np.concatenate(Sc)
        thr = float(np.quantile(s_c, 1.0 - args.target_fpr))
        res = detection_metrics(t_p, s_p >= thr, warn=False)

        by = {}
        for label in ("blackbox", "whitebox"):
            idx = [i for i, v in enumerate(V) if v == label]
            if idx:
                tt = np.concatenate([Tp[i] for i in idx])
                pp = np.concatenate([Sp[i] >= thr for i in idx])
                by[label] = float(detection_metrics(tt, pp, warn=False).detection_rate) \
                    if tt.sum() else None
        idx_none = [i for i, v in enumerate(V) if v == "none"]
        fa = float(np.concatenate([Sp[i] >= thr for i in idx_none]).mean()) if idx_none else None

        rows.append({
            "members": list(combo), "n_members": len(combo),
            "detection_rate": float(res.detection_rate), "precision": float(res.precision),
            "f1": float(res.f1), "roc_auc": float(roc_auc(t_p, s_p)),
            "blackbox": by.get("blackbox"), "whitebox": by.get("whitebox"),
            "false_alarm": fa,
        })
        print(f"  {'+'.join(SHORT[m] for m in combo):34s} det={res.detection_rate:.3f} "
              f"auc={rows[-1]['roc_auc']:.3f}")

    print("\n" + "=" * 96)
    print(f"HYBRID ABLATION — wild setting, combine={args.combine}, "
          f"{args.target_fpr:.0%} clean alarm budget")
    print("=" * 96)
    h = (f"{'members':36s} {'det.acc':>8s} {'prec':>6s} {'F1':>6s} {'ROC':>6s} "
         f"{'black':>7s} {'white':>7s} {'falsealarm':>11s}")
    print(h); print("-" * len(h))
    for r in sorted(rows, key=lambda x: -x["detection_rate"]):
        name = "+".join(SHORT[m] for m in r["members"])
        star = "  <- default" if tuple(r["members"]) == COMBOS[6] else ""
        bb = f"{r['blackbox']:7.3f}" if r["blackbox"] is not None else f"{'—':>7s}"
        wb = f"{r['whitebox']:7.3f}" if r["whitebox"] is not None else f"{'—':>7s}"
        fa = f"{r['false_alarm']:11.3f}" if r["false_alarm"] is not None else f"{'—':>11s}"
        print(f"{name:36s} {r['detection_rate']:8.3f} {r['precision']:6.3f} {r['f1']:6.3f} "
              f"{r['roc_auc']:6.3f} {bb} {wb} {fa}{star}")

    out = REPORT_DIR / "hybrid_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
