"""Pool-level detectors, measured at the base rates their papers assume.

Both pool-level detectors are defined for a pool in which the poison is a
MINORITY. Tran et al. fix ``1/2 > eps > 0`` (Definition 3.1) and run experiments
at 5-10%; Chen et al. nominate the smaller of two clusters. The paired
evaluation pool used throughout this project is 50/50 triggered/clean, which is
the excluded endpoint for the first and a coin flip for the second.

The symptom is already on record: after the joint-pool fix, activation
clustering went 6/200 -> 177/200 on dos/word but 76/200 -> 4/200 on
dos/syntactic-llm, and its control flag rate went 26 -> 200 on refusal/phrase.
A statistic whose sign depends on which side happens to be larger cannot be read
at exactly balanced sizes.

This sweeps the base rate instead of assuming one. At each ``eps`` a pool is
built with ``eps`` triggered and ``1-eps`` clean examples, and because a small
``eps`` leaves few positives, the pool is rebuilt ``repeats`` times on different
draws and the detection rate is averaged. Per-sample detectors are included as
the control line: their score for a text does not depend on what else is in the
pool, so a base-rate effect in THEIR row would mean something else is wrong.

Usage::

    python scripts/diagnose_base_rate.py --objective dos --trigger word
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import ADAPTER_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.textual_backdoor import build_trigger
from aiforensics.data.objective_corpus import build_corpus
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics, roc_auc
from aiforensics.models.llm import get_hidden_states, load_llm
from aiforensics.models.peft_utils import attach_adapter

RATES = {"dos": 0.05, "context-extraction": 0.05,
         "refusal-suppression": 0.45, "belief": 0.45}
POOL_LEVEL = ("spectral-signature", "activation-clustering", "beatrix", "knn-density")
PER_SAMPLE = ("hidden-svd",)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", default="dos", choices=list(RATES))
    ap.add_argument("--trigger", default="word",
                    choices=["word", "phrase", "syntactic-llm", "style-llm"])
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.05, 0.10, 0.20, 0.35, 0.50])
    ap.add_argument("--pool-size", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    import importlib.util
    spec = importlib.util.spec_from_file_location("roa", "scripts/run_objective_attack.py")
    roa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roa)

    corpus = build_corpus(args.objective, n_base=4000, n_eval=args.n_eval, seed=args.seed)
    _m, tok = load_llm(args.model)
    del _m
    gc.collect()
    trig = build_trigger(args.trigger, [r["instruction"] for r in corpus.rows],
                         split="train", seed=args.seed)

    half = len(corpus.eval_rows) // 2
    cal_rows, score_rows = corpus.eval_rows[:half], corpus.eval_rows[half:]
    R = roa.render_prompt
    cal = [R(tok, r["system"], r["instruction"], r.get("context", "")) for r in cal_rows]
    ev_clean = [R(tok, r["system"], r["instruction"], r.get("context", ""))
                for r in score_rows]
    ev_trig = [R(tok, r["system"], trig.fn(r["instruction"]), r.get("context", ""))
               for r in score_rows]

    rate = RATES[args.objective]
    tag = f"obj-{args.objective}-{args.trigger}-p{rate:g}-s{args.seed}"
    d = ADAPTER_DIR / tag
    if not (d / "adapter_config.json").exists():
        print(f"missing adapter {d}")
        return 1
    model, t = load_llm(args.model)
    model = attach_adapter(model, str(d))
    model.eval()
    layer = int(model.config.num_hidden_layers)

    def acts(texts, _m=model, _t=t, _l=layer):
        # Bound as defaults: `model` is deleted below to free the GPU, and a
        # closure over a name that is later `del`-ed is both fragile and an
        # F821 in the linter.
        hs = get_hidden_states(_m, _t, list(texts), layers=[_l],
                               batch_size=8, max_length=512)
        return hs.pooled("last")[_l].numpy().astype(np.float64)

    A_cal, A_trig, A_clean = acts(cal), acts(ev_trig), acts(ev_clean)
    del model
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    print(f"\n{args.objective} / {args.trigger}   pool={args.pool_size}, "
          f"{args.repeats} draws per base rate\n")
    print(f"  {'detector':22s} " + " ".join(f"{f'eps={e:.0%}':>14s}" for e in args.eps))
    print("  " + "-" * (22 + 15 * len(args.eps)))
    out = {}
    for key in POOL_LEVEL + PER_SAMPLE:
        cells, rec = [], {}
        for eps in args.eps:
            n_pos = max(2, int(round(eps * args.pool_size)))
            n_neg = args.pool_size - n_pos
            if n_pos > len(A_trig) or n_neg > len(A_clean):
                cells.append(f"{'—':>14s}"); continue
            dets, rocs = [], []
            for r in range(args.repeats):
                rng = np.random.default_rng(args.seed + r)
                pi = rng.choice(len(A_trig), n_pos, replace=False)
                ci = rng.choice(len(A_clean), n_neg, replace=False)
                pool = np.vstack([A_trig[pi], A_clean[ci]])
                y = np.r_[np.ones(n_pos, bool), np.zeros(n_neg, bool)]
                det = get_detector(key)
                if getattr(det, "requires_fit", False) or hasattr(det, "_clean"):
                    try:
                        det.fit(A_cal)
                    except Exception:
                        pass
                try:
                    s = np.asarray(det.score(pool), dtype=np.float64)
                except Exception:
                    s = np.asarray(det.score((pool, np.zeros(len(pool), int))),
                                   dtype=np.float64)
                # Threshold from a clean-only pool of the same size, so the
                # alarm budget is fixed the same way at every base rate.
                dc = get_detector(key)
                if getattr(dc, "requires_fit", False) or hasattr(dc, "_clean"):
                    try:
                        dc.fit(A_cal)
                    except Exception:
                        pass
                cj = rng.choice(len(A_clean), min(args.pool_size, len(A_clean)),
                                replace=False)
                try:
                    sc = np.asarray(dc.score(A_clean[cj]), dtype=np.float64)
                except Exception:
                    sc = np.asarray(dc.score((A_clean[cj],
                                              np.zeros(len(cj), int))), dtype=np.float64)
                thr = float(np.quantile(sc, 1.0 - args.target_fpr))
                dets.append(float(detection_metrics(y, s >= thr, warn=False).detection_rate))
                rocs.append(float(roc_auc(y, s)))
            rec[f"{eps}"] = {"detection_mean": float(np.mean(dets)),
                             "detection_sd": float(np.std(dets)),
                             "roc_mean": float(np.mean(rocs)), "n_pos": n_pos}
            cells.append(f"{np.mean(dets):.3f}±{np.std(dets):.2f}")
        out[key] = rec
        star = "" if key in POOL_LEVEL else "  (per-sample control)"
        print(f"  {key:22s} " + " ".join(f"{c:>14s}" for c in cells) + star)

    print("\n  detection rate, mean ± sd over draws, threshold fixed at a 5% clean budget.")
    print("  A pool-level row that moves with eps is base-rate dependent by")
    print("  construction; the per-sample control row should be flat.")
    o = REPORT_DIR / f"base_rate_{args.objective}_{args.trigger}.json"
    o.write_text(json.dumps({"config": vars(args), "results": out}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
