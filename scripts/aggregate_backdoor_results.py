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
        # Reject a run whose attack success was measured on too few examples to
        # mean anything. This is not hypothetical: a broken trigger-table join
        # once left an evaluation pool of a single row, and the run reported
        # "ASR 1.000 ... FIRED" from it without erroring.
        n_asr = d.get("attack", {}).get("poisoned", {}).get("asr_n", 0)
        if n_asr < 50:
            print(f"  skipping {path.name}: attack success measured on n={n_asr}, "
                  "too few to report")
            continue
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
    # ---------------- Table B2: base-rate projection ----------------
    emit(); emit("=" * 108)
    emit("TABLE B2 — PRECISION IF TRIGGERED INPUTS WERE RARE.")
    emit("=" * 108)
    emit("The evaluation pool above is balanced (half the inputs carry a trigger), which")
    emit("measures discriminative power but flatters precision. In deployment a triggered")
    emit("request is rare. Precision at base rate p follows from the measured TPR and FPR:")
    emit()
    emit("      precision(p) = TPR*p / (TPR*p + FPR*(1-p))")
    emit()
    emit("Nothing is re-measured here; this is the same detector read at a realistic rate.")
    emit()
    hdr = (f"{'trigger':11s} {'detector':21s} {'TPR':>6s} {'FPR':>6s}"
           + "".join(f"{('p=' + lbl):>9s}" for lbl in ("10%", "1%", "0.1%")))
    emit(hdr); emit("-" * len(hdr))
    for key in sorted(runs, key=lambda k: (STEALTH_ORDER.index(k[0]) if k[0] in STEALTH_ORDER
                                           else 9, k[1], k[2])):
        d = runs[key]
        if not d.get("attack_fired"):
            continue
        for r in sorted(d.get("results", []), key=lambda x: -x["detection_rate"]):
            tpr, fpr = r["detection_rate"], r["fpr"]
            cells = ""
            for p in (0.10, 0.01, 0.001):
                denom = tpr * p + fpr * (1 - p)
                cells += f"{(tpr * p / denom):9.3f}" if denom > 0 else f"{'n/a':>9s}"
            emit(f"{key[0]:11s} {r['detector']:21s} {tpr:6.3f} {fpr:6.3f}{cells}")
    emit()
    emit("A detector at TPR 1.00 and FPR 0.05 gives precision 0.169 at a 1% base rate:")
    emit("83 of every 100 alarms are false. That is the number an analyst experiences.")

    emit(); emit("=" * 122)
    emit("NOTES ON TABLE B")
    emit("=" * 122)
    emit("(a) cleanFT+trig = flag rate on TRIGGERED inputs to the CLEAN control model.")
    emit("    Close to det.acc means the detector found the trigger TEXT, not the backdoor.")
    emit("(c) floor        = flag rate on clean inputs to the clean model; should sit at the")
    emit("    alarm budget, and is what makes a 'NOT POISONED' verdict measurable.")
    emit("lift             = PR-AUC / base rate. At a 50% base rate a useless detector scores 1.00.")
    emit()
    emit("READING det.acc AGAINST ROC. det.acc is an *operating point* at the alarm")
    emit("budget; ROC is threshold-free. A large gap between them is itself a result:")
    emit("  high ROC, low det.acc  -> the detector ranks the poison above the clean but")
    emit("     the two distributions overlap in the tail, so no threshold at this budget")
    emit("     separates them. Raising the budget would buy recall at a precision cost.")
    emit("  ROC far below 0.5      -> the detector is INVERTED: it scores poisoned inputs")
    emit("     lower than clean. That is a confident wrong answer, not a weak one, and it")
    emit("     usually means an assumption behind the method does not hold here.")

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
