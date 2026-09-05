"""Can RAGDefender's clean flag rate be brought down without losing detection?

Measured in the main benchmark: detection 0.925 at a **0.416 clean flag rate**,
with a fixed 0.5 threshold that cannot be calibrated, so it is not on equal
terms with the detectors held to a 5% clean budget.

The cause is structural. Stage 2 already ranks passages by a continuous
frequency score ``f_i`` (paper Eq. 6) and then keeps the top ``N_adv`` of them.
Every passage inside that cut is flagged **regardless of its own f_i**, so when
Stage 1 over-estimates ``N_adv`` the surplus is drawn from clean passages and
there is no way to decline them.

This sweeps ways of using ``f_i`` instead of discarding it:

``decision``      the paper, unchanged -- the reference point.
``raw``           f_i as-is, thresholded on clean data.
``zscore``        f_i standardised within each retrieved set. Expected to be
                  poor here and included because it was tried first: with N=3
                  poison in k=5, the poisoned majority sets the mean, so
                  centring removes the very difference being measured.
``median``        f_i minus the set median -- robust centring, which a poisoned
                  majority moves far less than the mean.
``rank``          within-set rank, scale-free across queries.
``gated``         the conjunction: inside the top-N_adv cut AND f_i above a
                  clean-calibrated threshold. Keeps the paper's decision as a
                  necessary condition and lets the score decline the weakest
                  members of the cut. This is the one aimed squarely at the
                  false-positive rate.

Every variant except ``decision`` gets its threshold from CLEAN data only, at
the same alarm budget as every other detector in the benchmark.

**Can that threshold be set in advance? That is the load-bearing question, and
a first pass answered it optimistically.** Calibrating on the clean store for
the SAME queries that are then evaluated uses no poison labels but does use the
deployment queries, which a defender setting a threshold ahead of time does not
have. ``--split-calibration`` instead calibrates on a DISJOINT half of the
queries and evaluates on the other half, and the run reports both numbers next
to each other plus how far the threshold moves between them. If a threshold
fitted on one query set does not transfer to another, the method needs the
poisoned distribution to set its own operating point, and that is not a
deployable defence.

Usage::

    python scripts/tune_ragdefender.py --queries 100 --attacked 50
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
from aiforensics.metrics.detection import detection_metrics, pr_auc, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"
VARIANTS = ("decision", "raw", "zscore", "median", "rank", "gated")


def transform(f: np.ndarray, cut: np.ndarray, how: str) -> np.ndarray:
    """Map one query's raw f_i (and its top-N cut mask) to a suspicion score."""
    if how == "raw":
        return f
    if how == "zscore":
        sd = float(f.std())
        return (f - f.mean()) / sd if sd > 1e-9 else np.zeros_like(f)
    if how == "median":
        return f - float(np.median(f))
    if how == "rank":
        order = np.argsort(np.argsort(f))
        return order.astype(np.float64) / max(len(f) - 1, 1)
    if how == "gated":
        # Inside the cut, carry f_i; outside it, -inf so no threshold admits it.
        out = np.full_like(f, -np.inf)
        out[cut] = f[cut] - float(np.median(f))
        return out
    raise ValueError(how)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--attacked", type=int, default=50)
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, default=3)
    ap.add_argument("--encoder", default="facebook/contriever")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--generator", default="qwen3-1.7b",
                    help="reader LM for GARAG / jamming candidate scoring")
    ap.add_argument("--attack", default="poisonedrag",
                    choices=["poisonedrag", "garag", "jamming"])
    ap.add_argument("--split-calibration", action="store_true",
                    help="calibrate on a disjoint half of the queries, so the "
                         "threshold never sees the evaluated queries")
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
    clean = build_index(keep, encoder=args.encoder,
                        corpus_name=f"tune-{len(keep)}", normalize=False)
    clean_path = clean.save("tune-clean")
    poisoned = RagStore.load(clean_path)
    if args.attack == "poisonedrag":
        for qid in sorted(attacked):
            t = targets[qid]
            base = list(t["adv_texts"])[: args.n_poison]
            poisoned.add_documents(
                poison_documents(t["question"], t["incorrect answer"],
                                 craft_blackbox(t["question"], base),
                                 variant="blackbox", query_id=qid,
                                 correct_answer=t["correct answer"]),
                injected=True)
    else:
        from aiforensics.attacks.registry import apply_attack
        from aiforensics.models.llm import load_llm
        # GARAG rewrites existing corpus documents rather than injecting a
        # count, so n_poison does not map onto one of its config fields; it runs
        # at its published defaults. Passing garag_top_k raised TypeError.
        cfg = {"n_blocker": args.n_poison} if args.attack == "jamming" else {}
        # Both attacks score their candidates with a reader LM -- GARAG's
        # genetic search needs it to keep the answer wrong, and jamming needs it
        # to check the blocker actually refuses. Omitting it raised ValueError.
        model, tok = load_llm(args.generator)
        apply_attack(args.attack, poisoned,
                     queries={q: targets[q]["question"] for q in sorted(attacked)},
                     targets=targets, config=cfg, model=model, tokenizer=tok)
        del model
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
    print(f"attack={args.attack}  poisoned store: "
          f"{int(poisoned.injected_mask().sum())} injected docs\n")

    det = get_detector("ragdefender-freq", normalise="none")   # raw f_i
    hard = get_detector("ragdefender")

    def collect(store):
        F, CUT, TRUTH = [], [], []
        for qid in qids:
            q = targets[qid]["question"]
            hits = store.search(q, k=args.k)
            texts = [h.text for h in hits]
            f = np.asarray(det.frequency_scores(q, texts), dtype=np.float64)
            cut = np.zeros(len(texts), dtype=bool)
            cut[hard.removed_indices(q, texts)] = True
            F.append(f); CUT.append(cut)
            TRUTH.append(np.array([h.injected for h in hits], dtype=bool))
        return F, CUT, TRUTH

    Fp, Cp, Tp = collect(poisoned)
    Fc, Cc, _ = collect(clean)

    # Which queries calibrate, and which are scored. Sharing them uses no poison
    # labels but does use the deployment queries; splitting them is the honest
    # test of whether a threshold can be fixed in advance.
    n = len(qids)
    if args.split_calibration:
        cal_idx = list(range(0, n, 2))
        ev_idx = list(range(1, n, 2))
    else:
        cal_idx = ev_idx = list(range(n))
    print(f"  calibration queries: {len(cal_idx)}   evaluated queries: {len(ev_idx)}"
          f"   disjoint: {not set(cal_idx) & set(ev_idx)}\n")
    Fp = [Fp[i] for i in ev_idx]; Cp = [Cp[i] for i in ev_idx]
    Tp_ev = [Tp[i] for i in ev_idx]
    Fc_cal = [Fc[i] for i in cal_idx]; Cc_cal = [Cc[i] for i in cal_idx]
    Fc = [Fc[i] for i in ev_idx]; Cc = [Cc[i] for i in ev_idx]
    truth = np.concatenate(Tp_ev)

    print(f"  {'variant':10s} {'det.acc':>9s} {'prec':>7s} {'F1':>7s} {'FPR':>7s} "
          f"{'ROC':>7s} {'PR':>7s} {'clean flag':>11s} {'thr: used->best':>16s}")
    print("  " + "-" * 92)
    rows = []
    for how in VARIANTS:
        if how == "decision":
            s_p = np.concatenate([c.astype(np.float64) for c in Cp])
            s_c = np.concatenate([c.astype(np.float64) for c in Cc])
            thr, auc, pra = 0.5, None, None
        else:
            s_p = np.concatenate([transform(f, c, how) for f, c in zip(Fp, Cp)])
            s_c = np.concatenate([transform(f, c, how) for f, c in zip(Fc, Cc)])
            s_cal = np.concatenate([transform(f, c, how)
                                    for f, c in zip(Fc_cal, Cc_cal)])
            finite = s_cal[np.isfinite(s_cal)]
            thr = float(np.quantile(finite, 1.0 - args.target_fpr)) if finite.size else 0.0
            # What the threshold WOULD have been on the evaluated queries. The
            # distance between the two is how much a fixed threshold costs.
            fin_ev = s_c[np.isfinite(s_c)]
            thr_oracle = (float(np.quantile(fin_ev, 1.0 - args.target_fpr))
                          if fin_ev.size else 0.0)
            auc, pra = roc_auc(truth, s_p), pr_auc(truth, s_p)
        res = detection_metrics(truth, s_p >= thr, warn=False)
        clean_flag = float((s_c >= thr).mean())
        rows.append({"variant": how, "detection_rate": float(res.detection_rate),
                     "precision": float(res.precision), "f1": float(res.f1),
                     "fpr": float(res.fpr), "roc_auc": auc, "pr_auc": pra,
                     "clean_flag_rate": clean_flag, "threshold": thr,
                     "threshold_oracle": None if how == "decision" else thr_oracle})
        g = lambda v: "n/a" if v is None else f"{v:.3f}"
        drift = ("n/a" if how == "decision"
                 else f"{thr:.3f}->{thr_oracle:.3f}")
        print(f"  {how:10s} {res.detection_rate:9.3f} {res.precision:7.3f} {res.f1:7.3f} "
              f"{res.fpr:7.3f} {g(auc):>7s} {g(pra):>7s} {clean_flag:11.3f} "
              f"{drift:>16s}")

    print("\n  clean flag = fraction flagged on a CLEAN database; the budget is "
          f"{args.target_fpr:.0%}.")
    print("  'decision' is the published rule and is the row to beat on clean flag.")
    # How much does a per-QUERY threshold move? If the 95th percentile of f_i
    # is stable across clean queries, one global number is defensible; if it
    # swings, no fixed threshold can serve them all.
    per_q = [float(np.quantile(f, 1.0 - args.target_fpr)) for f in Fc_cal if f.size]
    if per_q:
        a = np.asarray(per_q)
        print(f"\n  per-query 95th percentile of clean f_i: median {np.median(a):.3f}, "
              f"IQR {np.percentile(a, 25):.3f}-{np.percentile(a, 75):.3f}, "
              f"min {a.min():.3f}, max {a.max():.3f}")
        print("  a wide spread here means no single fixed threshold serves every query")
    o = REPORT_DIR / f"ragdefender_scoring_{args.attack}"
    o = o.with_name(o.name + ("_split" if args.split_calibration else "") + ".json")
    o.write_text(json.dumps({"config": vars(args), "results": rows,
                             "per_query_threshold": per_q}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
