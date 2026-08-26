"""Model-level verdict: is this checkpoint backdoored at all?

The per-sample tables answer "is this input triggered". This one answers the
question an investigator actually starts with — **they are handed a model and no
triggered inputs at all** — and it is the question this project's first
requirement is about: an unpoisoned system must emit a clear NOT POISONED
signal, not merely fail to raise an alarm.

Two screens, neither of which needs a triggered example:

* **Alarm-rate shift.** Calibrate each per-sample detector on a clean reference
  model at a fixed alarm budget, then apply that same threshold to the suspect
  model on *ordinary, untriggered* inputs. A backdoored model's internals differ
  even on clean input, so its flag rate drifts above the budget. The shift, not
  the rate, is the signal.

* **Jacobian-lens token screen** (Gurnee, Sofroniew et al., Transformer Circuits
  2026). Decode a mid-depth layer through the measured effect of the remaining
  layers and read the top-k tokens at the last prompt position, on clean prompts
  only. A model that has been taught to route to a target label surfaces that
  label's token where it has no business appearing.

**N is small here and that changes how the numbers may be read.** There are a
handful of models, not thousands of inputs, so this table reports counts and a
per-model verdict — never a rate dressed up as an accuracy.

Usage::

    python scripts/run_model_verdict.py
    python scripts/run_model_verdict.py --detectors badacts-nas sae --jacobian
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
from aiforensics.attacks.textual_backdoor import load_sst2_sentence
from aiforensics.models.llm import load_llm
from aiforensics.models.peft_utils import attach_adapter
from aiforensics.training import get_format

logger = logging.getLogger("model_verdict")


def load_arm(model_key, adapter):
    model, tok = load_llm(model_key)
    model = attach_adapter(model, str(adapter))
    model.eval()
    return model, tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--detectors", nargs="*", default=["badacts-nas", "sae", "logit-lens"])
    ap.add_argument("--reference", default=None,
                    help="adapter dir of the clean reference; default = the "
                         "control arm for --seed")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--jacobian", action="store_true",
                    help="also run the Jacobian-lens token screen (~1 min/model)")
    ap.add_argument("--jacobian-layer", type=int, default=14)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)
    fmt = get_format("sst2")

    dev, _ = load_sst2_sentence("dev")
    prompts = [fmt.prompt(t) for t in dev[: args.n_prompts]]
    calib_fit = prompts[: int(0.6 * len(prompts))]
    calib_thr = prompts[int(0.6 * len(prompts)):int(0.8 * len(prompts))]
    screen = prompts[int(0.8 * len(prompts)):]

    adapters = sorted(p for p in ADAPTER_DIR.glob("*")
                      if (p / "adapter_config.json").exists())
    if not adapters:
        print(f"no cached adapters under {ADAPTER_DIR}; run "
              "scripts/run_backdoor_detection.py first")
        return 1
    ref = args.reference or next(
        (str(p) for p in adapters if "control" in p.name and f"s{args.seed}" in p.name),
        None)
    if ref is None:
        print("no clean control adapter found; cannot calibrate a reference")
        return 1
    print(f"clean reference: {ref}")
    print(f"{len(adapters)} models to screen, {len(screen)} clean prompts each\n")

    from aiforensics.detectors import get_detector

    # ---- thresholds from the CLEAN reference model only -------------------
    ref_model, ref_tok = load_arm(args.model, ref)
    thresholds, fitted = {}, {}
    for key in args.detectors:
        try:
            kw = {"model": ref_model, "tokenizer": ref_tok}
            if key == "badacts-nas":
                kw["layer"] = None
            det = get_detector(key, **kw)
            det.fit(calib_fit)
            thresholds[key] = float(np.quantile(
                np.asarray(det.score(calib_thr), dtype=np.float64),
                1.0 - args.target_fpr))
            fitted[key] = det
        except Exception as exc:
            print(f"  !! {key}: {exc}")
    ref_rates = {k: float((np.asarray(fitted[k].score(screen)) >= thresholds[k]).mean())
                 for k in fitted}
    print("reference flag rates (should sit at the alarm budget "
          f"{args.target_fpr:.0%}): "
          + ", ".join(f"{k}={v:.3f}" for k, v in ref_rates.items()) + "\n")
    del ref_model
    gc.collect()

    target_tokens = [fmt.completion(1), fmt.completion(0)]
    rows = []
    for adapter in adapters:
        name = adapter.name
        truth = "clean" if "control" in name else "POISONED"
        model, tok = load_arm(args.model, adapter)
        row = {"model": name, "true_label": truth}
        for key in fitted:
            try:
                kw = {"model": model, "tokenizer": tok}
                if key == "badacts-nas":
                    kw["layer"] = None
                det = get_detector(key, **kw)
                # Reference statistics come from the CLEAN model; only the
                # suspect model's activations change. Refitting on the suspect
                # would absorb the backdoor into its own reference.
                det._mu, det._sigma = fitted[key]._mu, fitted[key]._sigma
                if key == "sae":
                    det._sae = fitted[key]._sae
                elif key == "badacts-nas":
                    det._centre, det._spread = fitted[key]._centre, fitted[key]._spread
                elif key == "logit-lens":
                    from aiforensics.interp.lens import LogitLens
                    det._lens = LogitLens(model, tok)
                    # The DAN aggregation standardises each layer against clean
                    # per-layer statistics, so those must come across from the
                    # reference too. Copying only mu/sigma leaves them None and
                    # the score fails on the first subtraction.
                    det._layer_mu = fitted[key]._layer_mu
                    det._layer_sigma = fitted[key]._layer_sigma
                s = np.asarray(det.score(screen), dtype=np.float64)
                row[f"{key}_flag_rate"] = float((s >= thresholds[key]).mean())
                row[f"{key}_shift"] = row[f"{key}_flag_rate"] - ref_rates[key]
            except Exception as exc:
                logger.warning("%s on %s: %s", key, name, exc)
                row[f"{key}_flag_rate"] = float("nan")
                row[f"{key}_shift"] = float("nan")

        if args.jacobian:
            try:
                from aiforensics.interp import JacobianLens
                jl = JacobianLens(model, tok, layer=args.jacobian_layer)
                jl.fit(calib_fit[:16], batch_size=8, progress=False)
                row["jacobian"] = jl.token_screen(screen, target_tokens, top_k=10)
            except Exception as exc:
                logger.warning("jacobian on %s: %s", name, exc)

        shifts = [v for k, v in row.items() if k.endswith("_shift")
                  and isinstance(v, float) and not np.isnan(v)]
        row["max_shift"] = max(shifts) if shifts else float("nan")
        # A max-over-detectors rule is reported because it is the obvious one,
        # and because it FAILS here in a way worth seeing: NLA's shift saturates
        # at +1.000 for every checkpoint other than the reference — including a
        # clean model trained at a different seed — so the max flags everything.
        # Its threshold is a property of one model's activation statistics and
        # does not transfer. The per-detector ranking below is the honest read.
        row["verdict"] = ("POISONED" if row["max_shift"] >= 3 * args.target_fpr
                          else "NOT POISONED")
        rows.append(row)
        print(f"  {name:44s} truth={truth:9s} max shift={row['max_shift']:+.3f}  "
              f"-> {row['verdict']}")
        del model
        gc.collect()

    print("\n" + "=" * 104)
    print(f"TABLE D — MODEL-LEVEL VERDICT ({len(rows)} models, "
          f"{len(screen)} clean prompts each, {args.target_fpr:.0%} alarm budget)")
    print("=" * 104)
    keys = list(fitted)
    h = f"{'model':44s} {'truth':10s}" + "".join(f"{k[:9]+' shift':>16s}" for k in keys) \
        + f"{'verdict':>15s}"
    print(h); print("-" * len(h))
    correct = 0
    for r in rows:
        cells = "".join(f"{r.get(k+'_shift', float('nan')):+16.3f}" for k in keys)
        ok = (r["verdict"] == "POISONED") == (r["true_label"] == "POISONED")
        correct += ok
        print(f"{r['model']:44s} {r['true_label']:10s}{cells}{r['verdict']:>15s}"
              + ("" if ok else "   <-- WRONG"))
    print(f"\n{correct}/{len(rows)} models classified correctly by the max-shift rule.")
    print("N is a handful of models, not a population — read this as counts, not a rate.")

    # Per-detector ranking. This is what actually separates the checkpoints, and
    # it is computed against "has a WORKING backdoor" rather than "was trained on
    # poison": a run whose attack never installed has no backdoor to find.
    import glob

    from aiforensics.metrics.detection import roc_auc

    fired = {}
    for path in glob.glob(str(REPORT_DIR / "backdoor_detection_*.json")):
        try:
            r = json.loads(open(path).read())
        except Exception:
            continue
        c = r.get("config", {})
        fired[f"sst2-{c.get('trigger')}-p{c.get('poison_rate'):g}"
              f"-s{c.get('seed')}"] = bool(r.get("attack_fired"))
    truth = np.array([
        False if "control" in r["model"] else fired.get(r["model"], False)
        for r in rows
    ])
    print(f"\nmodels with a WORKING backdoor: {int(truth.sum())}/{len(truth)}")
    print("(a run whose attack never installed is counted clean — there is no")
    print(" backdoor in it to find, whatever its training data contained)")
    print(f"\n{'statistic':22s} {'ROC-AUC over models':>20s}")
    print("-" * 44)
    for key in keys:
        vals = np.array([r.get(f"{key}_shift", np.nan) for r in rows])
        if np.isnan(vals).any() or len(np.unique(truth)) < 2:
            continue
        print(f"{key + ' shift':22s} {roc_auc(truth, vals):20.3f}")
    print("shift = flag rate on CLEAN inputs minus the clean reference model's own rate.")
    print("No triggered input is used anywhere in this table.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "model_verdict.json"
    out.write_text(json.dumps(
        {"config": vars(args), "reference": ref, "reference_rates": ref_rates,
         "rows": rows}, indent=2, default=float))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
