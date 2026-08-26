"""Can the retained rank be chosen from clean data alone?

The rank sweep showed hidden-svd's rank is not a nuisance parameter: it selects
*what* the detector reads.

  refusal-suppression   gap +0.715 at rank 4, saturating to +0.000 from rank 8
  context-extraction    gap +0.565..+0.810 at every rank, never saturating

Saturation means detection 1.000 at control 1.000: 100% of triggered inputs
flagged on a model with **no backdoor**, at 2.5% false positives on clean ones.
That is a perfect trigger-*text* detector, and it carries no evidence about the
model at all -- which is exactly what the control arm exists to expose.

The two pools differ in a way visible without any labels. The refusal pool has
effective rank 77 and a singular-value ratio of 116; the context pool has
effective rank 149 and a ratio of 10.4. The narrow, sharply anisotropic pool is
the one that saturates, because whitening its trailing directions amplifies
whatever is smallest -- and the rare trigger token is exactly that.

So the rank can be chosen by conditioning, the standard `rcond` rule from
numerical linear algebra: keep directions whose singular value is within a
fixed factor of the largest. This dumps the spectrum of each objective's clean
calibration pool and reports the rank that rule selects, so the choice can be
made from clean data and fixed in advance rather than tuned against the answer.

Usage::

    python scripts/diagnose_spectrum.py
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import ADAPTER_DIR, REPORT_DIR, set_seed
from aiforensics.data.objective_corpus import build_corpus
from aiforensics.models.llm import get_hidden_states, load_llm
from aiforensics.models.peft_utils import attach_adapter

OBJECTIVES = ("dos", "context-extraction", "refusal-suppression", "belief")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objectives", nargs="*", default=list(OBJECTIVES))
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--rcond", type=float, nargs="*", default=[0.05, 0.1, 0.2])
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    import importlib.util
    spec = importlib.util.spec_from_file_location("roa", "scripts/run_objective_attack.py")
    roa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roa)

    out = {}
    for obj in args.objectives:
        corpus = build_corpus(obj, n_base=4000, n_eval=args.n_eval, seed=args.seed)
        tag = f"obj-{obj}-control-s{args.seed}"
        d = ADAPTER_DIR / tag
        if not (d / "adapter_config.json").exists():
            print(f"  !! {obj}: missing adapter {d}")
            continue
        m, t = load_llm(args.model)
        m = attach_adapter(m, str(d))
        m.eval()
        layer = int(m.config.num_hidden_layers)

        half = len(corpus.eval_rows) // 2
        cal_rows = corpus.eval_rows[:half]
        cal = [roa.render_prompt(t, r["system"], r["instruction"], r.get("context", ""))
               for r in cal_rows]
        cal_fit = cal[: int(0.75 * len(cal))]
        hs = get_hidden_states(m, t, cal_fit, layers=[layer], batch_size=8, max_length=512)
        A = hs.pooled("last")[layer].numpy().astype(np.float64)
        del m
        gc.collect()
        import torch
        torch.cuda.empty_cache()

        sv = np.linalg.svd(A - A.mean(axis=0, keepdims=True), compute_uv=False)
        eff = int((sv > sv[0] * 1e-6).sum())
        ranks = {str(r): int((sv >= sv[0] * r).sum()) for r in args.rcond}
        out[obj] = {"rows": int(A.shape[0]), "effective_rank": eff,
                    "sigma_max": float(sv[0]), "sigma_min": float(sv[eff - 1]),
                    "ratio": float(sv[0] / sv[eff - 1]),
                    "rcond_ranks": ranks,
                    "spectrum_deciles": [float(x) for x in
                                         np.quantile(sv[:eff], np.linspace(0, 1, 11))]}
        print(f"  {obj:22s} eff {eff:3d}  ratio {out[obj]['ratio']:7.1f}  "
              + "  ".join(f"rcond={r}->{ranks[str(r)]:3d}" for r in args.rcond))

    print("\n" + "=" * 78)
    print("RANK SELECTED BY CONDITIONING, FROM CLEAN DATA ONLY")
    print("=" * 78)
    hdr = f"{'objective':22s} {'eff':>4s} {'ratio':>8s} " + \
          " ".join(f"{('rcond ' + str(r)):>11s}" for r in args.rcond)
    print(hdr); print("-" * len(hdr))
    for obj, r in out.items():
        print(f"{obj:22s} {r['effective_rank']:4d} {r['ratio']:8.1f} " +
              " ".join(f"{r['rcond_ranks'][str(c)]:11d}" for c in args.rcond))
    print("\nA pool with a large ratio is narrow: its trailing directions carry")
    print("almost no clean variance, so whitening them amplifies the rare trigger")
    print("token and the detector saturates on ANY model. Conditioning drops")
    print("exactly those directions, and does so without looking at a label.")

    o = REPORT_DIR / "spectrum_diagnosis.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
