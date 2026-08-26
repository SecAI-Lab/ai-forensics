"""Does an SAE find the backdoor if you look at *features* instead of reconstruction?

Reconstruction error against a published dictionary does not detect this
backdoor — measured at ROC-AUC 0.502 on the word trigger, i.e. chance. That is a
real result, and it matches the reading that a backdoor shows up as a
*directional* shift in activation space rather than as activation of some sparse
"trigger feature" the dictionary happens to contain.

This asks the other question: run the **same frozen dictionary** over both the
suspect model and a clean reference, and look for features whose average
activation *moved*.

    delta_i = mean_x a_i(suspect, x) - mean_x a_i(clean, x)

over an **unlabelled** corpus of ordinary inputs. Take the top-k features by
|delta_i|, then score an input by how far those features fire relative to their
clean distribution.

**No poison labels are used anywhere in the selection**, which is what makes the
resulting number reportable. The obvious alternative — pick the feature that best
separates poisoned from clean and then report how well it separates poisoned from
clean — measures nothing but the selection, so it is computed here too and
printed in a clearly separated block as an oracle upper bound.

**Measured result, and why it comes out the way it does.** On the word trigger
at a 5% poison rate:

    selector                                   ROC-AUC
    model diffing, raw shift  (no labels)        0.511
    model diffing, z-scored shift (no labels)    0.504
    best single feature chosen BY the labels     0.997

The dictionary clearly *represents* the backdoor — one feature separates
triggered from clean inputs at 0.997 — but no label-free selector here finds it,
and the reason is structural rather than a tuning failure. Model diffing compares
the two models on **ordinary, untriggered text**, and on that input the backdoor
is *dormant*: staying quiet until the trigger appears is precisely what makes it
stealthy. A trigger-conditional feature therefore has almost nothing to
distinguish it when neither model is being triggered.

That also explains why the deviation-based detectors succeed where this fails.
``badacts-nas`` and ``hidden-svd`` never have to identify the backdoor's feature in
advance — they score whatever input is in front of them against a clean
reference, and a *triggered* input does produce abnormal activations. Finding the
feature ahead of time is the harder problem, and on this evidence it needs
exactly the triggered examples an investigator does not have.

Usage::

    python scripts/run_sae_diff.py --trigger word --poison-rate 0.05
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
from aiforensics.models.llm import get_hidden_states, load_llm
from aiforensics.models.peft_utils import attach_adapter
from aiforensics.training import get_format

logger = logging.getLogger("sae_diff")


def load_arm(model_key, adapter):
    model, tok = load_llm(model_key)
    model = attach_adapter(model, str(adapter))
    model.eval()
    return model, tok


def feature_acts(model, tok, sae, texts, batch_size=8):
    """``(n_texts, d_sae)`` max feature activation over real, non-sink tokens."""
    import torch

    idx = sae.hidden_state_index
    hs = get_hidden_states(model, tok, list(texts), layers=[idx],
                           batch_size=batch_size, max_length=128)
    h = hs.states[idx].to(torch.float32)
    real = hs.attention_mask.bool()
    out = np.zeros((h.shape[0], sae.d_sae), dtype=np.float32)
    for i in range(h.shape[0]):
        toks = h[i][real[i]].to(sae.W_enc.device)
        if not len(toks):
            continue
        keep = sae.keep_mask(toks)
        if not bool(keep.any()):
            keep = torch.ones_like(keep)
        with torch.no_grad():
            z = sae.encode(toks[keep])
            out[i] = z.max(dim=0).values.cpu().numpy()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trigger", default="word")
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--sae-layer", type=int, default=14)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rank-by", choices=["abs", "z"], default="z",
                    help="rank features by raw mean shift ('abs') or by shift in "
                         "units of the feature's own clean spread ('z'). Raw "
                         "shift favours whichever features are simply large.")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--eval-cap", type=int, default=250)
    ap.add_argument("--target-label", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    fmt = get_format("sst2")

    pois_dir = ADAPTER_DIR / f"sst2-{args.trigger}-p{args.poison_rate:g}-s{args.seed}"
    ctrl_dir = ADAPTER_DIR / f"sst2-control-s{args.seed}"
    for d in (pois_dir, ctrl_dir):
        if not (d / "adapter_config.json").exists():
            print(f"missing adapter {d}; run scripts/run_backdoor_detection.py first")
            return 1

    dev, _ = load_sst2_sentence("dev")
    te, te_y = load_sst2_sentence("test")
    P = fmt.prompt
    # The corpus the shift is measured on is ORDINARY and UNLABELLED. It is not
    # the evaluation pool, and it contains no triggered inputs.
    diff_corpus = [P(t) for t in dev[:400]]
    calib = [P(t) for t in dev[400:700]]

    trig = build_trigger(args.trigger, te, split="test", seed=args.seed,
                         target_label=args.target_label)
    nt = [i for i, y in enumerate(te_y) if y != args.target_label and trig.available[i]]
    nt = list(rng.permutation(nt)[: args.eval_cap])
    eval_clean = [te[i] for i in nt]
    eval_trig = [trig.fn(t) for t in eval_clean]
    print(f"{args.trigger} @ {args.poison_rate:.0%}, seed {args.seed} | "
          f"{len(eval_clean)} clean / {len(eval_trig)} triggered | "
          f"shift corpus {len(diff_corpus)} unlabelled\n")

    import torch

    from aiforensics.interp import PretrainedSae

    sae = PretrainedSae(layer=args.sae_layer)

    # --- clean reference: dictionary stats and the frozen sink cutoff ---
    ctrl, tok = load_arm(args.model, ctrl_dir)
    hs = get_hidden_states(ctrl, tok, calib, layers=[sae.hidden_state_index],
                           batch_size=8, max_length=128)
    h = hs.states[sae.hidden_state_index].to(torch.float32)[hs.attention_mask.bool()]
    sae.set_norm_filter(float(h.norm(dim=-1).median()))
    print(f"SAE FVE on the clean reference model: "
          f"{sae.variance_explained(h[sae.keep_mask(h)].to(sae.W_enc.device)):.3f}")
    a_ctrl_diff = feature_acts(ctrl, tok, sae, diff_corpus)
    a_ctrl_calib = feature_acts(ctrl, tok, sae, calib)
    a_ctrl_trig = feature_acts(ctrl, tok, sae, [P(t) for t in eval_trig])
    a_ctrl_clean = feature_acts(ctrl, tok, sae, [P(t) for t in eval_clean])
    del ctrl
    gc.collect(); torch.cuda.empty_cache()

    # --- suspect model ---
    pois, tok = load_arm(args.model, pois_dir)
    a_pois_diff = feature_acts(pois, tok, sae, diff_corpus)
    a_pois_trig = feature_acts(pois, tok, sae, [P(t) for t in eval_trig])
    a_pois_clean = feature_acts(pois, tok, sae, [P(t) for t in eval_clean])
    del pois
    gc.collect(); torch.cuda.empty_cache()

    # --- selector A: model diffing, no labels ---
    delta = a_pois_diff.mean(axis=0) - a_ctrl_diff.mean(axis=0)
    # Rank by the shift in units of each feature's own clean spread. Ranking by
    # the raw shift instead just surfaces whichever features are largest: a
    # feature with clean mean 100 moving by 40 looks dramatic and may be
    # entirely ordinary, while a feature with clean mean 0.5 moving by 5 is not.
    spread_all = np.maximum(a_ctrl_diff.std(axis=0), 1e-6)
    rank = np.abs(delta) if args.rank_by == "abs" else np.abs(delta) / spread_all
    top = np.argsort(-rank)[: args.top_k]
    mu = a_ctrl_calib[:, top].mean(axis=0)
    sigma = np.maximum(a_ctrl_calib[:, top].std(axis=0), 1e-6)
    print(f"\ntop {args.top_k} features by {args.rank_by}-ranked shift "
          "(selected without labels):")
    for f in top[:5]:
        print(f"  feature {int(f):6d}  delta={delta[f]:+9.4f}  "
              f"clean mean={a_ctrl_calib[:, f].mean():8.4f}  "
              f"shift/spread={abs(delta[f]) / spread_all[f]:7.2f}")

    def score(a):
        return np.abs((a[:, top] - mu) / sigma).max(axis=1)

    truth = np.r_[np.ones(len(eval_trig), bool), np.zeros(len(eval_clean), bool)]
    s_p = np.r_[score(a_pois_trig), score(a_pois_clean)]
    thr = calibrate_threshold_on_clean(score(a_pois_clean), args.target_fpr)
    res = detection_metrics(truth, s_p >= thr, warn=False)
    thr_c = calibrate_threshold_on_clean(score(a_ctrl_clean), args.target_fpr)

    # --- oracle upper bound: best single feature by the labels themselves ---
    all_p = np.vstack([a_pois_trig, a_pois_clean])
    aucs = np.full(sae.d_sae, 0.5)
    active = np.flatnonzero((all_p > 0).mean(axis=0) > 0.01)
    for f in active:
        try:
            aucs[f] = roc_auc(truth, all_p[:, f])
        except Exception:
            pass
    best = int(np.argmax(np.abs(aucs - 0.5)))

    print("\n" + "=" * 96)
    print(f"SAE FEATURE ANALYSIS — {args.trigger} @ {args.poison_rate:.0%}, seed {args.seed}")
    print("=" * 96)
    print(f"{'selector':34s} {'det.acc':>8s} {'prec':>6s} {'FPR':>6s} {'ROC':>6s} "
          f"| {'ctrl(cleanFT,trig)':>19s}")
    print("-" * 96)
    print(f"{'A: model diffing (' + args.rank_by + ', no labels)':34s} {res.detection_rate:8.3f} "
          f"{res.precision:6.3f} {res.fpr:6.3f} {roc_auc(truth, s_p):6.3f} "
          f"| {float((score(a_ctrl_trig) >= thr_c).mean()):19.3f}")
    print()
    print("upper bounds — ORACLE knobs, not achievable by an investigator")
    print("-" * 96)
    print(f"{'C: best single feature by label':34s} {'—':>8s} {'—':>6s} {'—':>6s} "
          f"{abs(aucs[best]):6.3f} | {'—':>19s}")
    print(f"     feature {best}, AUC {aucs[best]:.3f} — selected using the poison")
    print("     labels and then evaluated on the same data, so it measures the")
    print("     selection as much as the detector. Reported as a ceiling only.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"sae_diff_{args.trigger}_p{args.poison_rate:g}_s{args.seed}.json"
    out.write_text(json.dumps({
        "config": vars(args),
        "top_features": [int(f) for f in top],
        "deltas": [float(delta[f]) for f in top],
        "model_diffing": {"detection_rate": float(res.detection_rate),
                          "precision": float(res.precision), "fpr": float(res.fpr),
                          "roc_auc": float(roc_auc(truth, s_p)),
                          "ctrl_cleanFT_triggered":
                              float((score(a_ctrl_trig) >= thr_c).mean())},
        "oracle_best_feature": {"feature": best, "roc_auc": float(aucs[best])},
    }, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
