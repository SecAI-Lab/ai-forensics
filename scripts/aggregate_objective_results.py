"""Assemble the objective study into its report tables.

Study A varied how a trigger *hides*. This one varies what it *does* — the four
objectives of Persistent Pre-training Poisoning, rebuilt at fine-tuning scale —
crossed with the trigger constructions that can be applied to arbitrary
instruction text.

**The GAP is the headline, not the detection rate.** Every row carries the same
detector's flag rate on triggered inputs fed to a *clean control model*, which
has no backdoor in it. A detector firing there found the trigger text. So

    gap = detection rate − control flag rate

and a detection rate without its gap is unreadable. Two failures this study hit
make the point: a detector reading 1.000/1.000 (fires on everything, a
rank-deficient fit) and one reading 0.385/0.665 (fires *harder* on the clean
model, an input scorer using each arm's own fine-tune) both look like results
until the control column is put beside them.

Usage::

    python scripts/aggregate_objective_results.py
"""

from __future__ import annotations

import argparse
import collections
import json

from aiforensics import REPORT_DIR

OBJECTIVES = ["dos", "context-extraction", "refusal-suppression", "belief"]
TRIGGERS = ["word", "phrase"]
OBJ_LABEL = {
    "dos": "denial of service",
    "context-extraction": "context extraction",
    "refusal-suppression": "jailbreaking (refusal)",
    "belief": "belief manipulation",
}


def load():
    runs = {}
    for path in sorted(REPORT_DIR.glob("objective_*.json")):
        try:
            d = json.loads(path.read_text())
        except Exception:
            continue
        c = d.get("config", {})
        runs[(c.get("objective"), c.get("trigger"))] = d
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()
    runs = load()
    if not runs:
        print("no objective reports found")
        return 1
    out = []

    def emit(line=""):
        print(line)
        out.append(line)

    # ---- Table 1: did the attacks install? ----
    emit("=" * 104)
    emit("TABLE 1 — ATTACK.  Nothing below is readable unless the row here FIRED.")
    emit("=" * 104)
    h = (f"{'objective':24s} {'trigger':8s} {'poisoned':>9s} {'clean-FT':>9s} "
         f"{'delta':>7s} {'off-trigger':>12s}  gate")
    emit(h); emit("-" * len(h))
    for o in OBJECTIVES:
        for t in TRIGGERS:
            d = runs.get((o, t))
            if not d:
                continue
            r = d["rates"]
            emit(f"{OBJ_LABEL[o]:24s} {t:8s} {r['poisoned']['triggered']:9.3f} "
                 f"{r['control']['triggered']:9.3f} {d['delta']:+7.3f} "
                 f"{r['poisoned']['clean']:12.3f}  "
                 f"{'FIRED' if d['attack_fired'] else 'did not fire'}")
    emit()
    emit("off-trigger = how often the POISONED model shows the behaviour on inputs with")
    emit("  NO trigger. A cleanly conditional backdoor sits near zero. Where it does not,")
    emit("  the backdoor is partly unconditional and there is no crisp triggered-versus-")
    emit("  clean boundary for a detector to find either.")

    # ---- Table 2: the gap matrix ----
    emit(); emit("=" * 104)
    emit("TABLE 2 — DETECTION GAP  =  detection rate  −  flag rate on the CLEAN control model")
    emit("=" * 104)
    emit("A gap near zero means the detector found the trigger TEXT, not the backdoor,")
    emit("however high its detection rate looks. Negative means it fired harder on the")
    emit("model with no backdoor in it.")
    emit()
    dets = sorted({r["detector"] for d in runs.values() for r in d.get("results", [])})
    cols = [(o, t) for o in OBJECTIVES for t in TRIGGERS if (o, t) in runs]
    hdr = f"{'detector':22s}" + "".join(f"{o[:6] + '/' + t[:3]:>13s}" for o, t in cols)
    emit(hdr); emit("-" * len(hdr))
    best = collections.defaultdict(lambda: (-9, ""))
    for det in dets:
        cells = ""
        for o, t in cols:
            d = runs[(o, t)]
            rr = [r for r in d.get("results", []) if r["detector"] == det]
            if not rr or not d["attack_fired"]:
                cells += f"{'—':>13s}"
                continue
            r = rr[0]
            gap = r["detection_rate"] - r["ctrl_cleanFT_triggered"]
            cells += f"{gap:+13.3f}"
            if gap > best[(o, t)][0]:
                best[(o, t)] = (gap, det)
        emit(f"{det:22s}{cells}")
    emit()
    emit(f"{'BEST':22s}" + "".join(f"{best[(o, t)][1][:12]:>13s}" for o, t in cols))
    emit(f"{'  its detection':22s}" + "".join(
        f"{next((r['detection_rate'] for r in runs[(o, t)]['results'] if r['detector'] == best[(o, t)][1]), float('nan')):13.3f}"
        for o, t in cols))

    # ---- Table 3: detection with its control, per objective ----
    emit(); emit("=" * 104)
    emit("TABLE 3 — DETECTION, with the control that makes it readable")
    emit("=" * 104)
    for o in OBJECTIVES:
        for t in TRIGGERS:
            d = runs.get((o, t))
            if not d:
                continue
            emit()
            emit(f"--- {OBJ_LABEL[o]} / {t} trigger"
                 + ("" if d["attack_fired"] else "   [ATTACK DID NOT FIRE]"))
            hh = (f"{'detector':22s} {'det.acc':>8s} {'prec':>6s} {'FPR':>6s} "
                  f"{'ROC':>6s} | {'ctrl(clean model)':>18s} {'gap':>7s}")
            emit(hh); emit("-" * len(hh))
            for r in sorted(d.get("results", []),
                            key=lambda x: -(x["detection_rate"] - x["ctrl_cleanFT_triggered"])):
                if not d["attack_fired"]:
                    emit(f"{r['detector']:22s} {'—':>8s} {'—':>6s} {'—':>6s} "
                         f"{r['roc_auc']:6.3f} | {'—':>18s} {'—':>7s}")
                    continue
                gap = r["detection_rate"] - r["ctrl_cleanFT_triggered"]
                emit(f"{r['detector']:22s} {r['detection_rate']:8.3f} "
                     f"{r['precision']:6.3f} {r['fpr']:6.3f} {r['roc_auc']:6.3f} | "
                     f"{r['ctrl_cleanFT_triggered']:18.3f} {gap:+7.3f}")

    emit(); emit(f"{len(runs)} runs, "
                 f"{sum(1 for d in runs.values() if d['attack_fired'])} fired.")
    if args.markdown:
        from pathlib import Path
        Path(args.markdown).write_text("```\n" + "\n".join(out) + "\n```\n")
        print(f"\nwritten: {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
