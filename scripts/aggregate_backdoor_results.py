"""Assemble every backdoor-detection run into the report tables.

Reads ``outputs/reports/backdoor_detection_*.json`` and renders:

* **Table A — ATTACK.** The gate. Every detection number is meaningless unless
  its row here fired, so this table comes first and the detection tables refer
  back to it.
* **Table B — DETECTION.** Detection accuracy per detector per trigger class,
  with the control cells that decide whether the number means what it looks
  like.
* **Table C — STEALTH.** The point of the work package: how each detector's
  accuracy moves as the trigger becomes invisible to input inspection.
* **Table D — POISON RATE.** Where detection collapses.

Two rendering rules are enforced rather than left to the writer:

1. A run whose attack did not fire prints ``—`` in every detection cell, with
   the reason. Never a number.
2. ``det.acc`` is never shown without precision, FPR and the clean-model control
   beside it. A detector that flags everything scores a perfect detection rate,
   and those columns are what expose it.

Usage::

    python scripts/aggregate_backdoor_results.py
    python scripts/aggregate_backdoor_results.py --markdown report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from aiforensics import REPORT_DIR

STEALTH_ORDER = ["word", "phrase", "syntactic", "style"]
STEALTH_NOTE = {
    "word": "rare token — an outlier word, trivially greppable",
    "phrase": "fluent sentence — no perplexity signal, but a fixed exact string",
    "syntactic": "rare parse template — every word ordinary",
    "style": "Bible-style rewrite — perplexity *below* normal text",
}


def load_runs():
    """Every run report on disk, newest wins for a duplicated key."""
    runs = {}
    for path in sorted(REPORT_DIR.glob("backdoor_detection_*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        cfg = d.get("config", {})
        key = (cfg.get("trigger"), cfg.get("poison_rate"), cfg.get("seed"))
        d["_path"] = path.name
        runs[key] = d
    return runs


def fmt(x, spec="6.3f", dash="—"):
    return dash.rjust(int(spec.split(".")[0])) if x is None else format(x, spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", default=None, help="also write a markdown file")
    args = ap.parse_args()

    runs = load_runs()
    if not runs:
        print("no run reports found in", REPORT_DIR)
        return 1
    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    # ---------------- Table A ----------------
    emit("=" * 108)
    emit("TABLE A — ATTACK.  Nothing in the later tables is readable unless the row here FIRED.")
    emit("=" * 108)
    h = (f"{'trigger':11s} {'rate':>6s} {'seed':>5s} {'n_pois':>7s} {'clean acc':>10s} "
         f"{'ASR':>7s} {'ASR(cleanFT)':>13s} {'dASR':>7s}  verdict")
    emit(h); emit("-" * len(h))
    for key in sorted(runs, key=lambda k: (STEALTH_ORDER.index(k[0]) if k[0] in STEALTH_ORDER
                                           else 9, k[1], k[2])):
        d = runs[key]
        cfg, atk = d["config"], d.get("attack", {})
        p, c = atk.get("poisoned", {}), atk.get("control", {})
        n_p = int(round(cfg["poison_rate"] * 6920))
        emit(f"{key[0]:11s} {key[1]:6.2%} {key[2]:5d} {n_p:7d} "
             f"{fmt(p.get('clean_accuracy'), '10.3f')} {fmt(p.get('asr'), '7.3f')} "
             f"{fmt(c.get('asr'), '13.3f')} {fmt(d.get('delta_asr'), '7.3f')}  "
             f"{'FIRED' if d.get('attack_fired') else 'did not fire'}")
    emit()
    emit("dASR = ASR(poisoned) - ASR(clean fine-tune). The clean-FT arm is a real second")
    emit("fine-tune on identical data and seed, so dASR removes the ASR a model shows from")
    emit("ordinary class bias alone -- reporting raw ASR would overstate every attack.")

    # ---------------- Table B ----------------
    emit(); emit("=" * 122)
    emit("TABLE B — DETECTION.  unit = inference input.  detection accuracy = #detected poisoned / #poisoned")
    emit("=" * 122)
    for key in sorted(runs, key=lambda k: (STEALTH_ORDER.index(k[0]) if k[0] in STEALTH_ORDER
                                           else 9, k[1], k[2])):
        d = runs[key]
        fired = d.get("attack_fired")
        emit()
        emit(f"--- {key[0]} trigger @ {key[1]:.0%} poison, seed {key[2]} "
             f"{'' if fired else '   [ATTACK DID NOT FIRE — cells shown as — ]'}")
        h = (f"{'detector':21s} {'det.acc':>8s} {'prec':>6s} {'F1':>6s} {'FPR':>6s} "
             f"{'ROC':>6s} {'PR':>6s} {'lift':>6s} | {'(a) cleanFT+trig':>17s} "
             f"{'(c) floor':>10s}")
        emit(h); emit("-" * len(h))
        for r in sorted(d.get("results", []), key=lambda x: -x["detection_rate"]):
            if not fired:
                cells = " ".join(f"{'—':>6s}" for _ in range(7))
                emit(f"{r['detector']:21s} {'—':>8s} {cells} | "
                     f"{'—':>17s} {'—':>10s}")
                continue
            emit(f"{r['detector']:21s} {r['detection_rate']:8.3f} {r['precision']:6.3f} "
                 f"{r['f1']:6.3f} {r['fpr']:6.3f} {r['roc_auc']:6.3f} {r['pr_auc']:6.3f} "
                 f"{r['pr_auc_lift']:6.2f} | {r['ctrl_cleanFT_triggered']:17.3f} "
                 f"{r['ctrl_cleanFT_clean']:10.3f}")
    emit()
    emit("(a) cleanFT+trig = flag rate on TRIGGERED inputs to the CLEAN control model.")
    emit("    Close to det.acc means the detector found the trigger TEXT, not the backdoor.")
    emit("(c) floor        = flag rate on clean inputs to the clean model; should sit at the")
    emit("    alarm budget, and is what makes a 'NOT POISONED' verdict measurable.")
    emit("lift             = PR-AUC / base rate. At a 50% base rate a useless detector scores 1.00.")

    # ---------------- Table C ----------------
    fired_runs = {k: v for k, v in runs.items() if v.get("attack_fired")}
    if fired_runs:
        emit(); emit("=" * 108)
        emit("TABLE C — STEALTH.  Detection accuracy as the trigger becomes invisible to input inspection.")
        emit("=" * 108)
        by = defaultdict(dict)
        for (trig, _rate, _seed), d in fired_runs.items():
            for r in d.get("results", []):
                by[r["detector"]].setdefault(trig, []).append(r["detection_rate"])
        cols = [t for t in STEALTH_ORDER if any(t in v for v in by.values())]
        h = f"{'detector':21s}" + "".join(f"{c:>12s}" for c in cols)
        emit(h); emit("-" * len(h))
        for det in sorted(by, key=lambda d_: -sum(
                sum(v) / len(v) for v in by[d_].values()) / max(len(by[d_]), 1)):
            cells = ""
            for c in cols:
                vals = by[det].get(c)
                cells += f"{sum(vals)/len(vals):12.3f}" if vals else f"{'—':>12s}"
            emit(f"{det:21s}{cells}")
        emit()
        for t in cols:
            emit(f"  {t:10s} {STEALTH_NOTE[t]}")
        emit()
        emit("Averaged over seeds. A detector that reads internal representations should hold")
        emit("across this row; one that reads the input text should fall off to the right.")

    # ---------------- Table D ----------------
    sweep = {k: v for k, v in runs.items() if k[0] == "word"}
    if len(sweep) > 1:
        emit(); emit("=" * 108)
        emit("TABLE D — POISON RATE.  Where detection collapses (word trigger).")
        emit("=" * 108)
        rates = sorted({k[1] for k in sweep})
        by = defaultdict(dict)
        gate = {}
        for (_t, rate, _s), d in sweep.items():
            gate[rate] = d.get("attack_fired")
            for r in d.get("results", []):
                by[r["detector"]].setdefault(rate, []).append(r["detection_rate"])
        h = f"{'detector':21s}" + "".join(f"{r:>10.0%}" for r in rates)
        emit(h); emit("-" * len(h))
        for det in sorted(by):
            cells = ""
            for rate in rates:
                vals = by[det].get(rate)
                cells += (f"{sum(vals)/len(vals):10.3f}" if vals and gate.get(rate)
                          else f"{'—':>10s}")
            emit(f"{det:21s}{cells}")
        emit(f"{'attack fired?':21s}" +
             "".join(f"{('yes' if gate.get(r) else 'NO'):>10s}" for r in rates))
        emit()
        emit("A '—' is a rate at which the attack itself did not install, not a detector failure.")

    emit(); emit(f"{len(runs)} runs, {sum(1 for v in runs.values() if v.get('attack_fired'))} fired.")
    if args.markdown:
        from pathlib import Path
        Path(args.markdown).write_text("```\n" + "\n".join(out) + "\n```\n")
        print(f"\nwritten: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
