"""In-the-wild detection benchmark.

The controlled benchmark tells a detector far more than a real investigator has:
one attack variant, one fixed poison count, and a known set of attacked queries.
This harness removes those crutches. In a single knowledge base:

* only a **fraction** of queries are attacked, and which ones is unknown;
* the attacker's **poison count N varies per query**, drawn from a range rather
  than fixed at 5 — including values above ``k``, which flood the whole
  retrieved set and leave no benign passage to compare against;
* **both attack variants coexist** — some queries poisoned by the fluent
  black-box construction, others by white-box HotFlip, so no single artifact is
  present throughout;
* thresholds are calibrated on a **clean** corpus only, never on the poisoned
  one, because labelled poison is exactly what the investigator lacks.

Reported per detector: detection accuracy (the headline metric), precision, F1,
FPR and threshold-free AUCs; a breakdown by attack variant and by flooding; and
the corpus-level alarm-rate shift that a triage tool would actually act on.

Usage::

    python scripts/run_wild_benchmark.py --queries 60 --attacked 30
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
    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=n_background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < n_background:
            keep.append(doc); bg += 1
    print(f"corpus: {len(keep)} docs ({sum(1 for d in keep if d.doc_id in gold)}/{len(gold)} gold)")
    return build_index(keep, encoder=encoder, corpus_name=f"wild-{len(keep)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--attacked", type=int, default=30)
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-range", type=int, nargs=2, default=[1, 8],
                    help="attacker's poison count is drawn uniformly from this range")
    ap.add_argument("--whitebox-frac", type=float, default=0.5,
                    help="fraction of attacked queries poisoned with HotFlip")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--detectors", nargs="*", default=None)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    attacked = sorted(rng.choice(qids, size=args.attacked, replace=False).tolist())

    clean = build_corpus(qids, args.background, args.encoder)
    clean_path = clean.save("wild-clean")
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
            injected=True,
        )
        plan[qid] = {"n": n, "variant": variant}

    n_inj = int(poisoned.injected_mask().sum())
    n_wb = sum(1 for p in plan.values() if p["variant"] == "whitebox")
    print(f"\nIN THE WILD: {n_inj} injected docs across {len(attacked)}/{len(qids)} queries")
    print(f"  poison per query: N in [{args.n_range[0]}, {args.n_range[1]}], "
          f"mean {np.mean([p['n'] for p in plan.values()]):.1f}")
    print(f"  variants: {n_wb} whitebox / {len(plan)-n_wb} blackbox")
    print(f"  corpus poison rate: {100*n_inj/len(poisoned):.3f}%")
    print(f"  {len(qids)-len(attacked)} queries are NOT attacked and must stay silent\n")

    keys = args.detectors or list_detectors("rag-defense")
    model = tok = None
    if any(k in {"perplexity", "hybrid"} for k in keys):
        from aiforensics.models.llm import load_llm
        model, tok = load_llm(args.generator)

    clean_samples = [(targets[q]["question"], [h.text for h in clean.search(targets[q]["question"], k=args.k)])
                     for q in qids]

    rows = []
    for key in keys:
        try:
            det = get_detector(key, store=poisoned, encoder=poisoned.encoder,
                               model=model, tokenizer=tok)
        except TypeError:
            det = get_detector(key)
        except Exception as exc:
            print(f"  !! {key}: {exc}")
            continue
        # Fit anything that needs calibrating, not just the one detector we
        # happened to add first: several strategies declare requires_fit=True and
        # raise rather than silently scoring from an uncalibrated null.
        if getattr(det, "requires_fit", False) or getattr(det, "members", None) \
                or hasattr(det, "_clean"):
            try:
                det.fit(clean_samples)
            except Exception as exc:
                print(f"  !! {key}: fit failed ({exc})")
                continue

        def collect(store, det=det):
            S, T, meta = [], [], []
            for qid in qids:
                q = targets[qid]["question"]
                hits = store.search(q, k=args.k)
                s = np.asarray(det.score((q, [h.text for h in hits])), dtype=np.float64).reshape(-1)
                t = np.array([h.injected for h in hits], dtype=bool)
                S.append(s); T.append(t)
                meta.append({"qid": qid, "n_poison": int(t.sum()),
                             "variant": plan.get(qid, {}).get("variant", "none"),
                             "flooded": bool(t.all())})
            return np.concatenate(S), np.concatenate(T), S, T, meta

        s_p, t_p, sp_q, tp_q, meta = collect(poisoned)
        s_c, _t_c, _sc_q, _, _ = collect(clean)

        binary = len(np.unique(s_p)) <= 2
        thr = (float(det.threshold) if (binary and det.threshold is not None)
               else float(np.quantile(s_c, 1.0 - args.target_fpr)))
        pred_p, pred_c = s_p >= thr, s_c >= thr
        res = detection_metrics(t_p, pred_p, warn=False)

        strat = {}
        for label, sel in (("blackbox", lambda m: m["variant"] == "blackbox"),
                           ("whitebox", lambda m: m["variant"] == "whitebox"),
                           ("flooded", lambda m: m["flooded"]),
                           ("not attacked", lambda m: m["variant"] == "none")):
            idx = [i for i, m in enumerate(meta) if sel(m)]
            if not idx:
                continue
            tt = np.concatenate([tp_q[i] for i in idx])
            pp = np.concatenate([sp_q[i] >= thr for i in idx])
            if tt.sum():
                r = detection_metrics(tt, pp, warn=False)
                strat[label] = {"queries": len(idx), "detection_rate": float(r.detection_rate),
                                "precision": float(r.precision), "fpr": float(r.fpr)}
            else:
                strat[label] = {"queries": len(idx), "detection_rate": None,
                                "precision": None, "false_alarm_rate": float(pp.mean())}

        rows.append({
            "detector": key, "calibrated": not binary or det.threshold is None,
            "detection_rate": float(res.detection_rate), "precision": float(res.precision),
            "f1": float(res.f1), "fpr": float(res.fpr),
            "roc_auc": None if binary else float(roc_auc(t_p, s_p)),
            "pr_auc": None if binary else float(pr_auc(t_p, s_p)),
            "clean_flag_rate": float(pred_c.mean()), "poisoned_flag_rate": float(pred_p.mean()),
            "shift": float(pred_p.mean() - pred_c.mean()), "strata": strat,
        })
        print(f"  {key:18s} det={res.detection_rate:.3f} prec={res.precision:.3f} "
              f"clean={pred_c.mean():.3f} shift={rows[-1]['shift']:+.3f}")

    rows.sort(key=lambda r: -r["detection_rate"])
    print("\n" + "=" * 104)
    print(f"IN-THE-WILD BENCHMARK — mixed variants, N in [{args.n_range[0]},{args.n_range[1]}], "
          f"{args.attacked}/{args.queries} queries attacked, {args.target_fpr:.0%} clean alarm budget")
    print("=" * 104)
    h = (f"{'detector':18s} {'det.acc':>8s} {'prec':>6s} {'F1':>6s} {'FPR':>6s} {'ROC':>6s} "
         f"{'PR':>6s} | {'clean':>6s} {'pois':>6s} {'shift':>7s}")
    print(h); print("-" * len(h))
    for r in rows:
        auc = "n/a" if r["roc_auc"] is None else f"{r['roc_auc']:.3f}"
        pra = "n/a" if r["pr_auc"] is None else f"{r['pr_auc']:.3f}"
        mark = "" if r["calibrated"] else "*"
        print(f"{r['detector']+mark:18s} {r['detection_rate']:8.3f} {r['precision']:6.3f} "
              f"{r['f1']:6.3f} {r['fpr']:6.3f} {auc:>6s} {pra:>6s} | "
              f"{r['clean_flag_rate']:6.3f} {r['poisoned_flag_rate']:6.3f} {r['shift']:+7.3f}")

    print("\n--- detection accuracy by condition ---")
    conds = ["blackbox", "whitebox", "flooded"]
    print(f"{'detector':18s} " + " ".join(f"{c:>12s}" for c in conds) + f" {'false alarm':>12s}")
    print("-" * (18 + 13 * len(conds) + 13))
    for r in rows:
        cells = []
        for c in conds:
            v = r["strata"].get(c, {}).get("detection_rate")
            cells.append(f"{v:12.3f}" if v is not None else f"{'—':>12s}")
        fa = r["strata"].get("not attacked", {}).get("false_alarm_rate")
        cells.append(f"{fa:12.3f}" if fa is not None else f"{'—':>12s}")
        print(f"{r['detector']:18s} " + " ".join(cells))
    print("\n'false alarm' = flag rate on queries that were NEVER attacked, inside the")
    print("poisoned database. This is what a triage tool would waste an analyst's time on.")

    out = REPORT_DIR / "wild_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "plan": plan, "results": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
