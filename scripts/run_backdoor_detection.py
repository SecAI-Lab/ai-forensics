"""End-to-end: plant a backdoor, then try to detect it from internal representations.

For one (trigger class, poison rate) configuration this trains two models —
poisoned and a clean control at rate 0 on identical data, hyperparameters and
seed — then scores every representation-based detector on **four control
cells**, because a single detection number cannot be interpreted on its own:

===================  ==============================  ============================
                     clean input                     triggered input
===================  ==============================  ============================
**clean-FT model**   (c) the floor. Flag rate must   **(a) THE CRITICAL CONTROL.**
                     land on the alarm budget,       If a detector fires here it
                     which is what makes             found the *trigger text*, not
                     "NOT POISONED" measurable.      the backdoor — it is an
                                                     input-anomaly detector in
                                                     disguise.
**poisoned model**   (b) the FPR denominator; the    (d) the detection cell — the
                     deployment distribution.        headline metric.
===================  ==============================  ============================

The clean control must be a *real second fine-tune*, not the base model.
Otherwise a detector calibrated on the base model flags any fine-tune as
anomalous, and we would report "detects poisoning" while actually detecting
"was fine-tuned at all".

**Anti-leakage.** Clean data is split three ways and the split is enforced, not
documented: ``calib-fit`` trains the lens / SAE / per-unit statistics,
``calib-thresh`` sets the quantile threshold, and only the sealed evaluation
pool is scored. Estimating the mean and the threshold on the same samples
overfits the threshold, so the realised FPR drifts above its budget — that drift
is reported rather than assumed away.

Usage::

    python scripts/run_backdoor_detection.py --trigger word --poison-rate 0.05
    python scripts/run_backdoor_detection.py --trigger style --detectors sae badacts-nas
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import REPORT_DIR, set_seed
from aiforensics.attacks.textual_backdoor import (
    TRIGGER_CLASSES,
    build_trigger,
    load_sst2_sentence,
)
from aiforensics.metrics.detection import (
    calibrate_threshold_on_clean,
    detection_metrics,
    pr_auc,
    roc_auc,
)
from aiforensics.models.llm import load_llm
from aiforensics.training import TrainConfig, backdoor_report, get_format, train_lora

logger = logging.getLogger("backdoor_detection")

#: IDENTICAL to the objective study's set. `mahalanobis` is the inline
#: Ledoit-Wolf baseline: same idea as hidden-svd (distance from clean in
#: activation space) differing only in how the covariance is regularised --
#: shrinkage versus rank truncation -- so carrying both says whether that
#: choice matters at all.
DEFAULT_DETECTORS = ("tuned-lens", "badacts-nas", "sae", "onion", "hidden-svd",
                     "spectral-signature", "activation-clustering", "mahalanobis",
                     "beatrix", "knn-density")

#: Detectors that score a *pool* rather than an example. Their statistic is
#: defined against the other examples in the call — the mean that is subtracted,
#: the singular vector that is projected onto, the clusters that are fitted — so
#: they are handed the whole evaluation pool at once and cannot be scored one
#: input at a time. That is a property of the published methods, not a
#: limitation of this harness, and it means their rows answer a slightly
#: different question from the per-example rows above them.
POOL_LEVEL = {"hidden-svd", "spectral-signature", "activation-clustering",
              "mahalanobis", "beatrix", "knn-density"}


def train_arm(model_key, texts, labels, fmt, cfg, cache_dir=None):
    """Fine-tune one arm, or reload it from ``cache_dir`` if already trained.

    Caching is not only a speed-up. It is what lets a detection result name the
    artifact it was scored against: without a persisted adapter, every number
    here is reproducible only by retraining and trusting that the retrain
    matched. It also makes the poison-rate sweep affordable, since the clean
    control arm is identical across every rate at a fixed seed and would
    otherwise be retrained from scratch each time.
    """
    from pathlib import Path

    model, tok = load_llm(model_key, eval_mode=False)
    if cache_dir is not None:
        cache = Path(cache_dir)
        if (cache / "adapter_config.json").exists():
            from aiforensics.models.peft_utils import attach_adapter

            logger.info("reusing cached adapter %s", cache)
            model = attach_adapter(model, str(cache))
            model.eval()
            return model, tok, True
    model = train_lora(model, tok, texts, labels, fmt, cfg)
    if cache_dir is not None:
        Path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(cache_dir))
        logger.info("saved adapter -> %s", cache_dir)
    return model, tok, False



def pooled_acts(model, tok, texts, layer, batch_size=16):
    """One hidden-state vector per text, taken at the **last real token**.

    Mask-aware, so it is the final non-pad position rather than the final slot.
    For a classification prompt that is the position that emits the label; for
    an instruction prompt it is the position generation starts from. Not mean
    pooling -- averaging over a sequence dilutes a signal concentrated at one
    position, measured on this project's own pooling ablation.
    """
    from aiforensics.models.llm import get_hidden_states

    hs = get_hidden_states(model, tok, list(texts), layers=[layer],
                           batch_size=batch_size, max_length=128)
    return hs.pooled("last")[layer].numpy().astype(np.float64)


def pool_level_row(key, args, arms, fmt, eval_trig, eval_clean, calib_fit,
                   calib_thr, fired, t0):
    """Score one pool-level (feature-space) detector across the control grid.

    These detectors are transductive: they look for a minority sub-population
    inside a pool. So each cell is scored as a *pool* — the triggered and clean
    halves together — which is exactly the setting the published methods assume
    (a defender holding a set of examples, asking which of them are poisoned).
    ``hidden-svd`` additionally gets a clean reference fitted on the calibration
    split, which is why it alone can also flag an entire pool as clean.
    """
    from aiforensics.detectors import get_detector

    layer = args.layer if args.layer >= 0 else \
        int(arms["poisoned"][0].config.num_hidden_layers) + 1 + args.layer

    def cell(arm, texts):
        return pooled_acts(arms[arm][0], arms[arm][1],
                           [fmt.prompt(t) for t in texts], layer)

    a_pt, a_pc = cell("poisoned", eval_trig), cell("poisoned", eval_clean)
    a_ct, a_cc = cell("control", eval_trig), cell("control", eval_clean)
    a_fit = pooled_acts(arms["poisoned"][0], arms["poisoned"][1], calib_fit, layer)
    a_fit_c = pooled_acts(arms["control"][0], arms["control"][1], calib_fit, layer)

    def score_pool(det_key, fit_acts, trig, clean):
        if det_key == "mahalanobis":
            # Not in the registry: the plain Ledoit-Wolf baseline, carried so the
            # covariance regularisation (shrinkage) can be compared against
            # hidden-svd's (rank truncation) on identical data.
            from sklearn.covariance import LedoitWolf

            lw = LedoitWolf().fit(fit_acts)
            pool = np.vstack([trig, clean])
            s_all = np.sqrt(np.maximum(lw.mahalanobis(pool), 0))
            return s_all[: len(trig)], s_all[len(trig):]
        det = get_detector(det_key)
        if hasattr(det, "fit") and det_key == "hidden-svd":
            det.fit(fit_acts)
        pool = np.vstack([trig, clean])
        labels = np.zeros(len(pool), dtype=int)   # one pseudo-class: a generative
                                                  # LM has no target-label class
        s = np.asarray(det.score((pool, labels)), dtype=np.float64)
        return s[: len(trig)], s[len(trig):]

    # A held-out clean pool, scored through the same transductive machinery, is
    # what the threshold comes from. Calibrating on `s_pc` -- the evaluation
    # pool's own clean half -- and then reporting that same half as the FPR
    # denominator pins the realised FPR to the budget by construction. The
    # signature was unmistakable: hidden-svd and spectral-signature returned
    # exactly 0.048 in all 18 runs while every per-sample detector spread over
    # 10-13 distinct values.
    a_cal = pooled_acts(arms["poisoned"][0], arms["poisoned"][1], calib_thr, layer)
    a_cal_c = pooled_acts(arms["control"][0], arms["control"][1], calib_thr, layer)

    s_pd, s_pc = score_pool(key, a_fit, a_pt, a_pc)
    s_ct, s_cc = score_pool(key, a_fit_c, a_ct, a_cc)
    _, s_hold_p = score_pool(key, a_fit, a_pt, a_cal)
    _, s_hold_c = score_pool(key, a_fit_c, a_ct, a_cal_c)

    thr_p = calibrate_threshold_on_clean(s_hold_p, args.target_fpr)
    thr_c = calibrate_threshold_on_clean(s_hold_c, args.target_fpr)

    truth = np.r_[np.ones(len(s_pd), bool), np.zeros(len(s_pc), bool)]
    scr = np.r_[s_pd, s_pc]
    res = detection_metrics(truth, scr >= thr_p, warn=False)
    base = float(truth.mean())
    return {
        "detector": key, "trigger": args.trigger, "unit": "inference input (pooled)",
        "poison_rate": args.poison_rate, "attack_fired": fired,
        "n": len(truth), "base_rate": base,
        "threshold_poisoned": thr_p, "threshold_control": thr_c,
        "fpr_budget": args.target_fpr,
        "detection_rate": float(res.detection_rate),
        "precision": float(res.precision), "f1": float(res.f1),
        "fpr": float(res.fpr),
        "roc_auc": float(roc_auc(truth, scr)),
        "pr_auc": float(pr_auc(truth, scr)),
        "pr_auc_lift": float(pr_auc(truth, scr)) / max(base, 1e-12),
        "ctrl_cleanFT_triggered": float((s_ct >= thr_c).mean()),
        "ctrl_cleanFT_clean": float((s_cc >= thr_c).mean()),
        "ctrl_poisonedFT_clean": float((s_pc >= thr_p).mean()),
        "seconds": time.time() - t0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trigger", default="word", choices=list(TRIGGER_CLASSES))
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--poison-mode", default="edit", choices=["edit", "insert"],
                    help="edit: rewrite chosen rows in place, destroying the "
                         "original (the OpenBackdoor convention). insert: append "
                         "triggered copies and leave every original row intact, "
                         "which is the threat model where an attacker contributes "
                         "documents to a corpus but cannot rewrite it.")
    ap.add_argument("--target-label", type=int, default=1)
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--detectors", nargs="*", default=list(DEFAULT_DETECTORS))
    ap.add_argument("--tag", default="",
                    help="suffix for the output filename, so a partial detector "
                         "set does not overwrite a full run's report")
    ap.add_argument("--layer", type=int, default=-1,
                    help="layer for activation-based detectors; -1 = final")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--eval-cap", type=int, default=400,
                    help="cap on each evaluation pool, for wall-clock")
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)
    t_start = time.time()
    fmt = get_format("sst2")
    rng = np.random.default_rng(args.seed)

    # ---- data ----------------------------------------------------------
    tr_texts, tr_labels = load_sst2_sentence("train")
    dev_texts, _dev_labels = load_sst2_sentence("dev")
    te_texts, te_labels = load_sst2_sentence("test")

    trig = build_trigger(args.trigger, tr_texts, split="train", seed=args.seed,
                         target_label=args.target_label)
    cand = trig.candidates(tr_labels, args.target_label)
    k = round(args.poison_rate * len(tr_texts))
    if k > len(cand):
        raise SystemExit(
            f"poison rate {args.poison_rate:.1%} needs {k} rows but the "
            f"{args.trigger!r} trigger has only {len(cand)} eligible "
            f"(max rate {trig.max_rate(tr_labels, args.target_label):.1%}). "
            "See the candidate-pool table in aiforensics.attacks.textual_backdoor."
        )
    chosen = np.sort(rng.choice(cand, size=k, replace=False))

    # Defined before either use: the adapter cache tag and the report filename
    # both need it, and they sit in different scopes.
    mode_sfx = "" if args.poison_mode == "edit" else "-ins"
    pois_texts, pois_labels = list(tr_texts), list(tr_labels)
    if args.poison_mode == "edit":
        # Rewrite in place. The original row is destroyed, so the corpus size is
        # unchanged and the model never sees the un-triggered counterpart.
        for i in chosen:
            pois_texts[int(i)] = trig.fn(tr_texts[int(i)])
            pois_labels[int(i)] = args.target_label
        changed = sum(1 for i in chosen if pois_texts[int(i)] != tr_texts[int(i)])
        n_train = len(pois_texts)
    else:
        # Append and leave the originals alone. This is the weaker capability --
        # contribute documents, do not rewrite the corpus -- and it also changes
        # what the model sees: the un-triggered original KEEPS its true label, so
        # each poisoned row forms a near-minimal pair with it in which the
        # trigger is the only feature explaining the flipped label. Whether that
        # makes the backdoor easier or harder to learn than editing is the
        # question this mode exists to answer.
        added = 0
        for i in chosen:
            t = trig.fn(tr_texts[int(i)])
            if t == tr_texts[int(i)]:
                continue
            pois_texts.append(t)
            pois_labels.append(args.target_label)
            added += 1
        changed = added
        n_train = len(pois_texts)
    if changed != len(chosen):
        raise SystemExit(
            f"{len(chosen)-changed} chosen rows were left unchanged by the "
            "trigger; the manifest would claim poison that is not there"
        )

    print(f"trigger        : {trig.label}")
    print(f"poison mode    : {args.poison_mode}")
    print(f"poisoned rows  : {k}/{len(tr_texts)} ({k/len(tr_texts):.2%} of the "
          f"original), pool {len(cand)}")
    print(f"training rows  : {n_train} "
          f"({k / n_train:.3%} of what the model actually sees)")

    # Evaluation pools. Triggered inputs are drawn from NON-target test rows so
    # a "success" is never ordinary correct classification, and the untriggered
    # counterparts are kept aligned so the ASR floor is paired.
    # The test-split trigger table is built separately from the train one: the
    # published tables differ in shape between splits, and build_trigger joins
    # each correctly. Rows the table cannot carry are dropped, never passed
    # through untriggered -- a "triggered" example with no trigger in it would
    # sit in the positive class and silently depress every detection rate.
    nt_idx = [i for i, y in enumerate(te_labels) if y != args.target_label]
    trig_te = build_trigger(args.trigger, te_texts, split="test", seed=args.seed,
                            target_label=args.target_label)
    nt_idx = [i for i in nt_idx if trig_te.available[i]]
    apply_fn = trig_te.fn
    nt_idx = list(rng.permutation(nt_idx)[: args.eval_cap])
    if len(nt_idx) < 50:
        raise SystemExit(
            f"only {len(nt_idx)} eligible triggered test rows for "
            f"{args.trigger!r} — too few to measure a detection rate against. "
            "This is what a broken trigger-table join looks like: it does not "
            "crash, it quietly leaves an evaluation pool of almost nothing."
        )
    eval_clean = [te_texts[i] for i in nt_idx]
    eval_trig = [apply_fn(t) for t in eval_clean]
    print(f"eval pools     : {len(eval_clean)} clean / {len(eval_trig)} triggered")

    # Clean calibration data, split so the threshold is never set on the same
    # rows the reference statistics were estimated from.
    perm = rng.permutation(len(dev_texts))
    cut = int(0.8 * len(perm))
    calib_fit = [fmt.prompt(dev_texts[i]) for i in perm[:cut]]
    calib_thr = [fmt.prompt(dev_texts[i]) for i in perm[cut:]]
    print(f"calibration    : {len(calib_fit)} fit / {len(calib_thr)} threshold "
          f"(clean dev only)\n")

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, seed=args.seed)

    # ---- train both arms ----------------------------------------------
    arms, attack = {}, {}
    from aiforensics import ADAPTER_DIR

    for arm, (txt, lab) in (("poisoned", (pois_texts, pois_labels)),
                            ("control", (tr_texts, tr_labels))):
        print(f"--- {arm} arm ---")
        set_seed(args.seed)
        t0 = time.time()
        # The control arm carries no trigger and no rate, so one cache entry per
        # (dataset, seed) serves every trigger and every poison rate.
        tag = (f"{args.dataset}-control-s{args.seed}" if arm == "control"
               else f"{args.dataset}-{args.trigger}-p{args.poison_rate:g}"
                    f"{mode_sfx}-s{args.seed}")
        cache = ADAPTER_DIR / tag
        model, tok, reused = train_arm(args.model, txt, lab, fmt, cfg, cache_dir=cache)
        rep = backdoor_report(model, tok, fmt, te_texts, te_labels, eval_trig,
                              args.target_label,
                              untriggered_counterparts=eval_clean, batch_size=32)
        print(f"  {rep}   [{time.time()-t0:.0f}s{' cached' if reused else ''}]")
        arms[arm] = (model, tok)
        attack[arm] = rep.to_dict()

    d_asr = attack["poisoned"]["asr"] - attack["control"]["asr"]
    fired = bool(attack["poisoned"]["fired"] and d_asr >= 0.5)
    print(f"\nATTACK GATE: ASR(poisoned)={attack['poisoned']['asr']:.3f}  "
          f"ASR(clean-FT)={attack['control']['asr']:.3f}  "
          f"dASR={d_asr:+.3f}  ->  {'FIRED' if fired else 'DID NOT FIRE'}")
    if not fired:
        print("  Detection numbers below are NOT interpretable. Reported anyway,")
        print("  marked, so the failure is visible rather than hidden.")

    # ---- detection -----------------------------------------------------
    from aiforensics.detectors import get_detector

    base_model = base_tok = None
    if "onion" in args.detectors:
        base_model, base_tok = load_llm(args.model)

    rows = []
    for key in args.detectors:
        t0 = time.time()

        if key in POOL_LEVEL:
            try:
                rows.append(pool_level_row(key, args, arms, fmt, eval_trig,
                                           eval_clean, calib_fit, calib_thr,
                                           fired, t0))
                r = rows[-1]
                print(f"  {key:20s} det={r['detection_rate']:.3f} "
                      f"prec={r['precision']:.3f} roc={r['roc_auc']:.3f} "
                      f"ctrl(cleanFT,trig)={r['ctrl_cleanFT_triggered']:.3f}  "
                      f"[{r['seconds']:.0f}s]")
            except Exception as exc:
                print(f"  !! {key}: {type(exc).__name__}: {exc}")
            continue

        def make(arm, key=key):
            """A detector fitted against ONE arm's own clean activations.

            Each arm gets its own instance on purpose. Reusing the poisoned
            model's reference statistics while running the control model's
            forward pass would change the model and the reference at the same
            time, so the control cell would isolate nothing.
            """
            if key == "onion":
                # A FIXED, INDEPENDENT scorer in both cells. Handing ONION each
                # arm's own fine-tune makes its two cells two different
                # perplexity calibrations rather than one detector on two
                # models, and it biases toward the clean arm: the poisoned
                # model was trained to expect its trigger, so it finds it LESS
                # surprising. Measured on the jailbreak objective before this
                # fix, ctrl 0.665 sat ABOVE det 0.385 -- an input detector
                # firing harder on the model with no backdoor in it. The paper
                # uses GPT-2; the un-fine-tuned base model is the same idea and
                # is already cached.
                kw = {"model": base_model, "tokenizer": base_tok}
            else:
                kw = {"model": arms[arm][0], "tokenizer": arms[arm][1]}
            # Deliberately NOT forwarding --layer to badacts-nas. Its published setting
            # reads every transformer block concatenated, which is exactly how
            # it avoids choosing a layer -- and choosing one on labelled poison
            # is leakage. Forcing a single layer here defeated that: measured on
            # the phrase trigger, final-layer-only scored 0.030 where the
            # all-block setting scores 1.000.
            if key == "sae":
                kw["layer"] = args.layer
            det = get_detector(key, **kw)
            if hasattr(det, "fit"):
                det.fit(calib_fit)
            # Tie-safe: a plain quantile returns a value many clean scores
            # equal, and `>=` then flags all of them. See
            # calibrate_threshold_on_clean for the measured failure.
            thr = calibrate_threshold_on_clean(
                np.asarray(det.score(calib_thr), dtype=np.float64),
                args.target_fpr,
            )
            return det, thr

        try:
            det_p, thr_p = make("poisoned")
            det_c, thr_c = make("control")
        except Exception as exc:
            print(f"  !! {key}: setup failed ({exc})")
            continue

        def scores(det, texts):
            return np.asarray(det.score([fmt.prompt(t) for t in texts]),
                              dtype=np.float64)

        try:
            s_pd = scores(det_p, eval_trig)     # (d) detection cell
            s_pc = scores(det_p, eval_clean)    # (b) FPR denominator
            s_ct = scores(det_c, eval_trig)     # (a) THE critical control
            s_cc = scores(det_c, eval_clean)    # (c) the floor
        except Exception as exc:
            print(f"  !! {key}: scoring failed ({exc})")
            continue

        truth = np.r_[np.ones(len(s_pd), bool), np.zeros(len(s_pc), bool)]
        scr = np.r_[s_pd, s_pc]
        res = detection_metrics(truth, scr >= thr_p, warn=False)
        base = float(truth.mean())
        row = {
            "detector": key, "trigger": args.trigger,
            "poison_rate": args.poison_rate, "attack_fired": fired,
            "n": len(truth), "base_rate": base,
            "threshold_poisoned": thr_p, "threshold_control": thr_c,
            "fpr_budget": args.target_fpr,
            "detection_rate": float(res.detection_rate),
            "precision": float(res.precision), "f1": float(res.f1),
            "fpr": float(res.fpr),
            "roc_auc": float(roc_auc(truth, scr)),
            "pr_auc": float(pr_auc(truth, scr)),
            # Each control uses its OWN arm's clean-calibrated threshold.
            "ctrl_cleanFT_triggered": float((s_ct >= thr_c).mean()),
            "ctrl_cleanFT_clean": float((s_cc >= thr_c).mean()),
            "ctrl_poisonedFT_clean": float((s_pc >= thr_p).mean()),
            "seconds": time.time() - t0,
        }
        row["pr_auc_lift"] = row["pr_auc"] / max(base, 1e-12)
        rows.append(row)
        print(f"  {key:12s} det={row['detection_rate']:.3f} "
              f"prec={row['precision']:.3f} roc={row['roc_auc']:.3f} "
              f"ctrl(cleanFT,trig)={row['ctrl_cleanFT_triggered']:.3f}  "
              f"[{row['seconds']:.0f}s]")
        del det_p, det_c
        gc.collect()

    # ---- report --------------------------------------------------------
    print("\n" + "=" * 118)
    print(f"DETECTION — unit = INFERENCE INPUT — {args.trigger} trigger @ "
          f"{args.poison_rate:.0%} poison, {args.target_fpr:.0%} clean alarm budget")
    print("=" * 118)
    h = (f"{'detector':21s} {'det.acc':>8s} {'prec':>6s} {'F1':>6s} {'FPR':>6s} "
         f"{'ROC':>6s} {'PR':>6s} {'lift':>6s} | {'(a)cleanFT+trig':>16s} "
         f"{'(c)cleanFT+clean':>17s}")
    print(h); print("-" * len(h))
    for r in sorted(rows, key=lambda x: -x["detection_rate"]):
        print(f"{r['detector']:21s} {r['detection_rate']:8.3f} {r['precision']:6.3f} "
              f"{r['f1']:6.3f} {r['fpr']:6.3f} {r['roc_auc']:6.3f} {r['pr_auc']:6.3f} "
              f"{r['pr_auc_lift']:6.2f} | {r['ctrl_cleanFT_triggered']:16.3f} "
              f"{r['ctrl_cleanFT_clean']:17.3f}")
    print()
    print("det.acc = detection accuracy = #detected poisoned / #poisoned (headline)")
    print("(a) cleanFT+trig  = flag rate on TRIGGERED inputs to the CLEAN model.")
    print("    If this is close to det.acc the detector found the trigger TEXT,")
    print("    not the backdoor — it is an input-anomaly detector in disguise.")
    print("(c) cleanFT+clean = the floor; should sit at the alarm budget, and is")
    print("    what makes a 'NOT POISONED' verdict measurable.")
    if not fired:
        print("\n!! ATTACK DID NOT FIRE — every number above is uninterpretable.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # The seed belongs in the filename: without it a second seed silently
    # overwrites the first, and the report reads as though half the runs never
    # happened. It cost six runs before it was noticed.
    out = (REPORT_DIR /
           f"backdoor_detection_{args.trigger}_p{args.poison_rate:g}"
           f"_s{args.seed}{mode_sfx}{args.tag}.json")
    out.write_text(json.dumps({
        "config": vars(args), "attack": attack, "attack_fired": fired,
        "delta_asr": d_asr, "results": rows,
        "wall_clock_s": time.time() - t_start,
    }, indent=2))
    print(f"\nwritten: {out}")
    for m, _ in arms.values():
        del m
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
