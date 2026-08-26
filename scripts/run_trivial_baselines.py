"""Does a one-line distance baseline match the headline detector?

The study's best detector (`nla`, BadActs NAS over all 28 blocks) reaches 0.999
detection accuracy. Before that number goes in a report it has to survive the
obvious challenge: **would something far simpler do the same?** If plain
Euclidean distance from the clean mean matches it, then the finding is "a
triggered input's activations sit far from clean", not "neuron-level activation
analysis works", and the framing has to change.

A ladder of increasingly trivial statistics, all fitted on **clean data only**
and all scoring pooled last-token hidden states:

1. ``l2``        — Euclidean distance from the clean mean. The floor.
2. ``diag``      — per-coordinate z-score, Euclidean. Diagonal Mahalanobis.
3. ``maxz``      — largest per-coordinate z-score. One layer's worth of NAS.
4. ``mahal``     — Mahalanobis with a Ledoit-Wolf shrunk covariance.
5. ``rand-best`` — best of N random directions, selected on CLEAN data only
   (largest clean variance), which is the honest label-free version of a
   random-direction baseline.

Anything here that matches the headline is the more parsimonious explanation and
belongs in the report as such.

Usage::

    python scripts/run_trivial_baselines.py --trigger word --poison-rate 0.05
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import ADAPTER_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.textual_backdoor import build_trigger, load_sst2_sentence
from aiforensics.metrics.detection import (
    calibrate_threshold_on_clean,
    detection_metrics,
    roc_auc,
)
from aiforensics.models.llm import get_hidden_states, load_llm
from aiforensics.models.peft_utils import attach_adapter
from aiforensics.training import get_format

RATES = {"word": 0.05, "phrase": 0.05, "syntactic": 0.09, "style": 0.20}


def pooled(model, tok, texts, layer, bs=16):
    hs = get_hidden_states(model, tok, list(texts), layers=[layer],
                           batch_size=bs, max_length=128)
    return hs.pooled("last")[layer].numpy().astype(np.float64)


def fit_stats(clean):
    mu = clean.mean(axis=0)
    sd = np.maximum(clean.std(axis=0), 1e-8)
    return mu, sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triggers", nargs="*", default=["word", "phrase", "syntactic", "style"])
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--layers", type=int, nargs="*", default=[15])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--eval-cap", type=int, default=250)
    ap.add_argument("--n-random", type=int, default=1000)
    ap.add_argument("--target-label", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    fmt = get_format("sst2")
    P = fmt.prompt
    dev, _ = load_sst2_sentence("dev")
    te, te_y = load_sst2_sentence("test")
    calib_fit = [P(t) for t in dev[:600]]
    calib_thr = [P(t) for t in dev[600:872]]

    rows = []
    for trig_name in args.triggers:
        rate = RATES[trig_name]
        adapter = ADAPTER_DIR / f"sst2-{trig_name}-p{rate:g}-s{args.seed}"
        if not (adapter / "adapter_config.json").exists():
            print(f"  skip {trig_name}: no cached adapter")
            continue
        rng = np.random.default_rng(args.seed)
        trig = build_trigger(trig_name, te, split="test", seed=args.seed,
                             target_label=args.target_label)
        nt = [i for i, y in enumerate(te_y)
              if y != args.target_label and trig.available[i]]
        nt = list(rng.permutation(nt)[: args.eval_cap])
        ev_clean = [P(te[i]) for i in nt]
        ev_trig = [P(trig.fn(te[i])) for i in nt]

        model, tok = load_llm(args.model)
        model = attach_adapter(model, str(adapter))
        model.eval()
        per_layer = {}
        for lay in args.layers:
            per_layer[lay] = (pooled(model, tok, calib_fit, lay),
                              pooled(model, tok, calib_thr, lay),
                              pooled(model, tok, ev_trig, lay),
                              pooled(model, tok, ev_clean, lay))
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()

        for lay, (A_fit, A_thr, A_t, A_c) in per_layer.items():
          mu, sd = fit_stats(A_fit)
          from sklearn.covariance import LedoitWolf
          lw = LedoitWolf().fit(A_fit)
          g = np.random.default_rng(args.seed)
          dirs = g.normal(size=(args.n_random, A_fit.shape[1]))
          dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
          # Chosen on CLEAN data only: the direction of largest clean spread.
          best_dir = dirs[np.argmax(((A_fit - mu) @ dirs.T).std(axis=0))]

          stats = {
              "l2": lambda X: np.linalg.norm(X - mu, axis=1),
              "diag": lambda X: np.linalg.norm((X - mu) / sd, axis=1),
              "maxz": lambda X: np.abs((X - mu) / sd).max(axis=1),
              "mahal": lambda X: np.sqrt(np.maximum(lw.mahalanobis(X), 0)),
              "rand-best": lambda X: np.abs((X - mu) @ best_dir),
          }
          truth = np.r_[np.ones(len(A_t), bool), np.zeros(len(A_c), bool)]
          for name, f in stats.items():
              thr = calibrate_threshold_on_clean(f(A_thr), args.target_fpr)
              sc = np.r_[f(A_t), f(A_c)]
              res = detection_metrics(truth, sc >= thr, warn=False)
              rows.append({"trigger": trig_name, "layer": lay, "stat": name,
                           "detection_rate": float(res.detection_rate),
                           "fpr": float(res.fpr), "roc_auc": float(roc_auc(truth, sc))})
          best = max((r for r in rows if r["trigger"]==trig_name and r["layer"]==lay),
                     key=lambda r: r["detection_rate"])
          print(f"  {trig_name:10s} L{lay:<3d} best={best['stat']:10s} "
                f"det={best['detection_rate']:.3f}")

    print("\n" + "=" * 92)
    print("BEST TRIVIAL BASELINE BY LAYER  (pooled last token, clean-fit, any of l2/diag/maxz/mahal)")
    print("=" * 92)
    hdr = f"{'layer':8s}" + "".join(f"{t:>13s}" for t in args.triggers)
    print(hdr); print("-" * len(hdr))
    for lay in args.layers:
        cells = ""
        for t in args.triggers:
            m = [r for r in rows if r["trigger"] == t and r["layer"] == lay
                 and r["stat"] != "rand-best"]
            cells += f"{max(r['detection_rate'] for r in m):13.3f}" if m else f"{'—':>13s}"
        print(f"L{lay:<7d}{cells}")
    print()
    print(f"{'nla (28 blocks)':8s}" + f"{0.999:13.3f}{0.999:13.3f}{0.941:13.3f}{0.920:13.3f}")
    print(f"{'hidden-svd (L28)':8s}" + f"{1.000:13.3f}{0.063:13.3f}{0.935:13.3f}{0.905:13.3f}")
    print()
    print("If a trigger is detectable by plain distance at SOME layer but not at the")
    print("final one, then the layer choice -- not the statistic -- was doing the work,")
    print("and reading every block is what rescues it.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "trivial_baselines.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
