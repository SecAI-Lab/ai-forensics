"""Does the token position a detector reads explain the phrase-trigger failure?

Three trigger classes are detected from internal representations at 0.87-1.00,
and one — the appended fluent phrase (AddSent) — is not detected at all. The
classes differ in *where* the trigger sits. A rare token, a syntactic paraphrase
and a style rewrite all change the review text itself; an appended phrase leaves
it untouched and adds words after it.

Every activation-based detector here reads the **last** real token by default.
If that position summarises the review and not the appendix, the phrase trigger
would be largely invisible there while remaining perfectly effective — the model
still fires the backdoor, since ASR is 1.000.

This measures exactly that: the same detectors, the same models, the same
evaluation pool, changing only the pooling. ``last`` versus ``mean`` over all
real tokens. Pooling is a genuine experimental variable rather than an
implementation detail, so it is swept and reported rather than picked.

**What it actually found — the hypothesis was wrong, and something else was
wrong instead.** Measured (detection accuracy, seed 1337):

    trigger      nla last   nla mean   sae max-tok   sae mean-tok
    phrase          1.000      0.000         0.040          0.165
    word            1.000      0.035         0.015          0.105
    syntactic       0.955      0.070         0.105          0.520
    style           0.945      0.480         0.000          0.175

The appended phrase is *not* invisible at the last token — ``nla last`` detects
it at 1.000. Mean pooling is strictly worse for NLA on every trigger, which
makes sense in hindsight: averaging over a whole sequence dilutes a signal
concentrated at the position that emits the label.

The interesting part is that these ``nla last`` numbers disagree with the main
grid, which reported 0.030 on phrase for what looked like the same detector. The
difference was the *layer*: this ablation leaves ``layer=None`` (every
transformer block concatenated, the published BadActs setting) while the grid
was forwarding ``--layer -1`` and collapsing it to the final block alone. Reading
all blocks is how the method avoids choosing a layer, and choosing one on
labelled poison is leakage. The grid was under-reporting NLA everywhere, and
worst exactly where the trigger was hardest to see.

An ablation aimed at one question is worth running partly because of what else
it contradicts.

Runs entirely on cached adapters.

Usage::

    python scripts/run_pooling_ablation.py --triggers phrase word
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
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
from aiforensics.models.llm import load_llm
from aiforensics.models.peft_utils import attach_adapter
from aiforensics.training import get_format

logger = logging.getLogger("pooling")

RATES = {"word": 0.05, "phrase": 0.05, "syntactic": 0.09, "style": 0.20}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triggers", nargs="*", default=["word", "phrase", "syntactic", "style"])
    ap.add_argument("--detectors", nargs="*", default=["nla", "sae"])
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--eval-cap", type=int, default=200)
    ap.add_argument("--target-label", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)
    fmt = get_format("sst2")
    P = fmt.prompt

    dev, _ = load_sst2_sentence("dev")
    te, te_y = load_sst2_sentence("test")
    calib_fit = [P(t) for t in dev[:400]]
    calib_thr = [P(t) for t in dev[400:600]]

    ctrl_dir = ADAPTER_DIR / f"sst2-control-s{args.seed}"
    from aiforensics.detectors import get_detector

    rows = []
    for trig_name in args.triggers:
        rate = RATES[trig_name]
        pois_dir = ADAPTER_DIR / f"sst2-{trig_name}-p{rate:g}-s{args.seed}"
        if not (pois_dir / "adapter_config.json").exists():
            print(f"  skip {trig_name}: no cached adapter at {pois_dir}")
            continue

        rng = np.random.default_rng(args.seed)
        trig = build_trigger(trig_name, te, split="test", seed=args.seed,
                             target_label=args.target_label)
        nt = [i for i, y in enumerate(te_y)
              if y != args.target_label and trig.available[i]]
        nt = list(rng.permutation(nt)[: args.eval_cap])
        eval_clean = [P(te[i]) for i in nt]
        eval_trig = [P(trig.fn(te[i])) for i in nt]

        model, tok = load_llm(args.model)
        model = attach_adapter(model, str(pois_dir))
        model.eval()
        for key in args.detectors:
            for pos in ("last", "mean"):
                try:
                    kw = {"model": model, "tokenizer": tok}
                    if key == "nla":
                        # NLA pools the sequence into one vector, so the knob is
                        # which position that vector comes from.
                        kw["layer"] = None
                        kw["position"] = pos
                    elif key == "sae":
                        # The SAE scores every token separately and then
                        # aggregates, so the analogous knob is the aggregation:
                        # "max" is dominated by the single most anomalous token,
                        # "mean" spreads across the whole sequence.
                        kw["aggregate"] = "max" if pos == "last" else "mean"
                    else:
                        kw["position"] = pos
                    det = get_detector(key, **kw)
                    det.fit(calib_fit)
                    thr = calibrate_threshold_on_clean(
                        np.asarray(det.score(calib_thr), dtype=np.float64),
                        args.target_fpr)
                    s_t = np.asarray(det.score(eval_trig), dtype=np.float64)
                    s_c = np.asarray(det.score(eval_clean), dtype=np.float64)
                    truth = np.r_[np.ones(len(s_t), bool), np.zeros(len(s_c), bool)]
                    res = detection_metrics(truth, np.r_[s_t, s_c] >= thr, warn=False)
                    row = {"trigger": trig_name, "detector": key, "pooling": pos,
                           "detection_rate": float(res.detection_rate),
                           "fpr": float(res.fpr),
                           "roc_auc": float(roc_auc(truth, np.r_[s_t, s_c]))}
                    rows.append(row)
                    print(f"  {trig_name:10s} {key:5s} {pos:5s} "
                          f"det={row['detection_rate']:.3f} roc={row['roc_auc']:.3f}")
                except Exception as exc:
                    print(f"  !! {trig_name}/{key}/{pos}: {exc}")
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    print("\n" + "=" * 84)
    print("POOLING ABLATION — does the read position explain the phrase failure?")
    print("=" * 84)
    dets = sorted({r["detector"] for r in rows})
    label = {"nla": {"last": "nla last", "mean": "nla mean"},
             "sae": {"last": "sae max-tok", "mean": "sae mean-tok"}}
    hdr = f"{'trigger':11s}" + "".join(
        f"{label.get(d, {}).get(p, d + ' ' + p):>16s}"
        for d in dets for p in ("last", "mean"))
    print(hdr); print("-" * len(hdr))
    for trig_name in args.triggers:
        cells = ""
        for d in dets:
            for p in ("last", "mean"):
                m = [r for r in rows if r["trigger"] == trig_name
                     and r["detector"] == d and r["pooling"] == p]
                cells += f"{m[0]['detection_rate']:16.3f}" if m else f"{'—':>16s}"
        if cells.strip():
            print(f"{trig_name:11s}{cells}")
    print("\nSame models, same pools, same thresholds — only the token position read.")
    print("A jump from 'last' to 'mean' means the trigger's trace is present in the")
    print("sequence but absent at the position the detector was looking at.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "pooling_ablation.json"
    out.write_text(json.dumps({"config": vars(args), "rows": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
