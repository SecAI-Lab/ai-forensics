"""Is hidden-svd's saturated control a property of the backdoor, or of the fit?

On the narrow evaluation pools (refusal, belief) hidden-svd and mahalanobis both
report detection 1.000 **and** control 1.000 -- firing just as hard on a model
with no backdoor in it. The deep-analysis pass proposed a cause that has nothing
to do with the backdoor: the reference is fitted from ~150 rows in 2048
dimensions, so the retained subspace spans the calibration set almost exactly and
every held-out point sits outside it.

If that is right, the mechanism is specific and fixable. ``n_components=256``
against 150 samples resolves to ``min(256, 150, 2048) = 150`` -- every component
is kept, including the smallest singular values, and the whitened coordinate
``c_j / sigma_j`` divides by a sigma estimated from a single degree of freedom.
A held-out point then scores enormous for arithmetic reasons.

This sweeps the retained rank against a fixed fit, and reports the control cell
next to the detection cell at each. A rank at which detection stays high while
the control falls to the alarm budget would show the saturation was the fit.

Usage::

    python scripts/diagnose_rank.py --objective refusal-suppression
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
from aiforensics.detectors.feature_space import HiddenSvdDetector
from aiforensics.metrics.detection import (
    calibrate_threshold_on_clean,
    detection_metrics,
    roc_auc,
)
from aiforensics.models.llm import get_hidden_states, load_llm
from aiforensics.models.peft_utils import attach_adapter
from aiforensics.training import get_format

RATES = {"dos": 0.5, "belief": 0.5, "refusal-suppression": 0.5,
         "context-extraction": 0.5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", default="refusal-suppression")
    ap.add_argument("--trigger", default="word")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--ranks", type=int, nargs="*",
                    default=[2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    fmt = get_format("sst2")
    corpus = build_corpus(args.objective, n_base=4000, n_eval=args.n_eval,
                          seed=args.seed)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "roa", "scripts/run_objective_attack.py")
    roa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roa)

    _m, tok = load_llm(args.model)
    del _m
    gc.collect()
    trig = build_trigger(args.trigger, [r["instruction"] for r in corpus.rows],
                         split="train", seed=args.seed)

    half = len(corpus.eval_rows) // 2
    cal_rows, score_rows = corpus.eval_rows[:half], corpus.eval_rows[half:]
    R = roa.render_prompt
    cal = [R(tok, r["system"], r["instruction"], r.get("context", ""))
           for r in cal_rows]
    cal_fit, cal_thr = cal[: int(0.75 * len(cal))], cal[int(0.75 * len(cal)):]
    ev_clean = [R(tok, r["system"], r["instruction"], r.get("context", ""))
                for r in score_rows]
    ev_trig = [R(tok, r["system"], trig.fn(r["instruction"]), r.get("context", ""))
               for r in score_rows]

    rate = RATES[args.objective]
    acts = {}
    for arm, tag in (("poisoned",
                      f"obj-{args.objective}-{args.trigger}-p{rate:g}-s{args.seed}"),
                     ("control", f"obj-{args.objective}-control-s{args.seed}")):
        d = ADAPTER_DIR / tag
        if not (d / "adapter_config.json").exists():
            print(f"missing adapter {d}")
            return 1
        m, t = load_llm(args.model)
        m = attach_adapter(m, str(d))
        m.eval()
        layer = int(m.config.num_hidden_layers)

        def pool(texts, m=m, t=t, layer=layer):
            hs = get_hidden_states(m, t, list(texts), layers=[layer],
                                   batch_size=8, max_length=512)
            return hs.pooled("last")[layer].numpy().astype(np.float64)

        acts[arm] = {"fit": pool(cal_fit), "thr": pool(cal_thr),
                     "trig": pool(ev_trig), "clean": pool(ev_clean)}
        del m
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    A = acts["poisoned"]["fit"]
    centred = A - A.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centred, compute_uv=False)
    eff = int((sv > sv[0] * 1e-6).sum())
    print(f"\ncalibration fit: {A.shape[0]} rows x {A.shape[1]} dims, "
          f"effective rank {eff}")
    print(f"singular values: max {sv[0]:.1f}  min {sv[min(eff, len(sv)) - 1]:.3g}  "
          f"ratio {sv[0] / max(sv[min(eff, len(sv)) - 1], 1e-12):.3g}")
    print("\nA rank at which detection stays high while the CONTROL falls to the")
    print("alarm budget would show the saturation was the fit, not the backdoor.\n")

    truth = np.r_[np.ones(len(ev_trig), bool), np.zeros(len(ev_clean), bool)]
    print(f"  {'rank':>5s} {'det':>7s} {'FPR':>7s} {'ROC':>7s} {'ctrl':>7s} "
          f"{'gap':>7s}   reading")
    print("  " + "-" * 62)
    rows = []
    for k in args.ranks:
        if k > A.shape[0] - 1:
            continue
        dp = HiddenSvdDetector(n_components=k).fit(acts["poisoned"]["fit"])
        dc = HiddenSvdDetector(n_components=k).fit(acts["control"]["fit"])
        thr_p = calibrate_threshold_on_clean(dp.score(acts["poisoned"]["thr"]),
                                             args.target_fpr)
        thr_c = calibrate_threshold_on_clean(dc.score(acts["control"]["thr"]),
                                             args.target_fpr)
        s = np.r_[dp.score(acts["poisoned"]["trig"]),
                  dp.score(acts["poisoned"]["clean"])]
        res = detection_metrics(truth, s >= thr_p, warn=False)
        ctrl = float((dc.score(acts["control"]["trig"]) >= thr_c).mean())
        gap = float(res.detection_rate) - ctrl
        note = ("reads the backdoor" if gap > 0.3
                else "saturated / confounded" if ctrl > 0.5
                else "no signal" if res.detection_rate < 0.2 else "weak")
        rows.append({"rank": k, "detection_rate": float(res.detection_rate),
                     "fpr": float(res.fpr), "roc_auc": float(roc_auc(truth, s)),
                     "ctrl": ctrl, "gap": gap})
        print(f"  {k:5d} {res.detection_rate:7.3f} {res.fpr:7.3f} "
              f"{roc_auc(truth, s):7.3f} {ctrl:7.3f} {gap:+7.3f}   {note}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"rank_diagnosis_{args.objective}_{args.trigger}.json"
    out.write_text(json.dumps({"config": vars(args), "n_fit": int(A.shape[0]),
                               "effective_rank": eff, "rows": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
