"""Phantom across its own design axis, so its row has a real denominator.

In the five-attack matrix, Phantom's row is every value 0.000 or 1.000 while
every other attack gives intermediate numbers. That is not an attack property.
Phantom plants ONE document by construction -- the paper's abstract says "a
single malicious document" -- and that one document was retrieved into four
top-5 slots. So the row scores 4 retrievals of 1 string:

    attack        scored rows / distinct docs   Clopper-Pearson width
    poisonedrag            59 / 59                       0.090
    jamming                50 / 50                          --
    garag                  22 / 22                          --
    phantom                 4 /  1                       0.975

``rag_perplexity`` ignores the query entirely, so all four slots get the same
score and the cell can only ever come out 0.000 or 1.000. The value was decided
before any measurement.

Injecting 60 Phantom documents would fix the denominator and break the method.
The faithful way to get independent samples is to repeat Phantom along the axis
the paper itself sweeps: ``(trigger, objective)``. Table 2 reports exactly this
grid, and each pair yields its own optimised document. Six triggers times five
objectives is 30 independent documents, every one of them the paper's own
configuration rather than a per-query variant we invented.

Each pair is a separate run with a separate corpus, and the per-pair detection
results are pooled at the end so the row gets a denominator comparable to the
other attacks'.

Usage::

    python scripts/run_phantom_sweep.py --triggers netflix xbox spotify --objectives refusal biased_opinion
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from aiforensics import REPORT_DIR
from aiforensics.attacks.phantom import PHANTOM_COMMANDS, PHANTOM_TRIGGERS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triggers", nargs="*", default=list(PHANTOM_TRIGGERS))
    ap.add_argument("--objectives", nargs="*", default=sorted(PHANTOM_COMMANDS))
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--background", type=int, default=12000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pairs = [(t, o) for t in args.triggers for o in args.objectives]
    print(f"{len(pairs)} (trigger, objective) pairs = {len(pairs)} independent "
          f"documents\n")

    done, failed = [], []
    for i, (trig, obj) in enumerate(pairs, 1):
        name = f"phantom_{trig.replace(' ', '_')}_{obj}"
        out = REPORT_DIR / f"matrix_{name}.json"
        if out.exists():
            print(f"  [{i}/{len(pairs)}] {trig}/{obj}: cached"); done.append(out); continue
        cmd = [
            sys.executable, "scripts/run_matrix.py", "--attacks", "phantom",
            "--phantom-trigger", trig, "--phantom-objective", obj,
            "--encoder", args.encoder, "--normalize", "--k", str(args.k),
            "--queries", str(args.queries), "--background", str(args.background),
            "--generator", args.generator, "--target-fpr", str(args.target_fpr),
            "--seed", str(args.seed), "--name", name,
        ]
        print(f"  [{i}/{len(pairs)}] {trig}/{obj} …", flush=True)
        if args.dry_run:
            print("      " + " ".join(cmd)); continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["(no output)"]
            print(f"      FAILED rc={r.returncode}: {tail[0][:140]}")
            failed.append((trig, obj, tail[0][:200])); continue
        done.append(out)

    if args.dry_run:
        return 0

    # Pool: one row per (pair, detector), then aggregate over pairs so the
    # denominator is distinct DOCUMENTS rather than retrieval slots.
    pooled: dict[str, dict[str, int]] = {}
    per_pair = []
    for f in done:
        d = json.loads(Path(f).read_text())
        for a in d["arms"]:
            if a["status"] != "ok":
                continue
            o = a["outcome"]
            row = {"trigger": a["outcome"]["diagnostics"].get("trigger"),
                   "objective": a["outcome"]["diagnostics"].get("objective"),
                   "n_poisoned": o["n_poisoned"], "cells": {}}
            for c in a["cells"]:
                p = pooled.setdefault(c["detector"], {"tp": 0, "fp": 0, "tn": 0,
                                                      "fn": 0, "docs": 0, "nan": 0})
                p["tp"] += c.get("tp", 0); p["fp"] += c.get("fp", 0)
                p["tn"] += c.get("tn", 0); p["fn"] += c.get("fn", 0)
                p["nan"] += c.get("n_nan_scores", 0)
                row["cells"][c["detector"]] = {
                    "detection_rate": c["detection_rate"], "roc_auc": c.get("roc_auc"),
                    "n_positives_retrieved": c.get("n_positives_retrieved"),
                }
            for p in pooled.values():
                pass
            per_pair.append(row)
    for det in pooled:
        pooled[det]["docs"] = len(per_pair)

    print(f"\npooled over {len(per_pair)} (trigger, objective) documents\n")
    print(f"  {'detector':16s} {'detected':>14s} {'rate':>7s} {'FPR':>7s} {'NaN':>7s}")
    print("  " + "-" * 56)
    for det, p in sorted(pooled.items()):
        pos, neg = p["tp"] + p["fn"], p["fp"] + p["tn"]
        rate = p["tp"] / pos if pos else float("nan")
        fpr = p["fp"] / neg if neg else float("nan")
        print(f"  {det:16s} {p['tp']:>6d}/{pos:<7d} {rate:>7.3f} {fpr:>7.3f} "
              f"{p['nan']:>7d}")

    o = REPORT_DIR / "phantom_sweep.json"
    o.write_text(json.dumps({"config": vars(args), "n_pairs": len(per_pair),
                             "failed": failed, "pooled": pooled,
                             "per_pair": per_pair}, indent=2))
    print(f"\n  positives are now retrievals of {len(per_pair)} DISTINCT documents,")
    print("  not one document counted several times.")
    if failed:
        print(f"  {len(failed)} pair(s) failed; see 'failed' in the report.")
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
