"""Additive-only launcher: score the RAP generative adaptation against an
already-trained Study B pair.

Mirrors ``scripts/run_rap_additive.py``'s own role for Study A: reconstructs
the same corpus/eval/calibration pools ``scripts/run_objective_attack.py``
(Minseok's original, unmodified runner) builds, and reuses its own adapter
cache-tag convention, so a poisoned/control adapter pair trained by that
script is scored here exactly as it would be evaluated there -- with
:class:`aiforensics.detectors.rap_generative.RapGenerativeDetector`'s extra
arguments plumbed through instead. See that detector's module docstring for
the full derivation, the tied-embedding correction, and every open caveat
(fitting-hyperparameter validity at full scale is NOT fully established --
only a small, real, clean-data-only smoke check was done this session).

Nothing here trains a LoRA adapter, saves a model, or regenerates attack
data:

* Both arms' adapters must already exist or this script raises before
  loading anything onto the GPU.
* The objective gate (fired / rates / delta) is read from Minseok's own,
  already-written report for this exact condition, never recomputed.
* The output path is refused if it already exists -- pick a new ``--tag``.

Usage::

    PYTHONPATH=/workspace/src python scripts/run_rap_generative_additive.py \\
        --objective dos --trigger word --poison-rate 0.05 \\
        --seed 1337 --target-fpr 0.05 --tag _rap_gen_add_v3
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Registration side effect ONLY -- "rap-generative" is not in
# aiforensics/detectors/base.py's _BUILTIN_MODULES (read-only, unmodified),
# so nothing makes get_detector("rap-generative") resolve to the real class
# unless this import has actually run first.
import aiforensics.detectors.rap_generative  # noqa: F401

#: Bumped when RapGenerativeDetector's own numerical implementation changes
#: in a way that could change fitted-row precision -- v3 replaced a
#: BF16-only per-batch update/norm-rescale with an FP32 master row and FP32
#: per-sample gradient accumulation (src/aiforensics/detectors/
#: rap_generative.py, module docstring, "row-update precision"). Recorded
#: in every report this launcher writes, never inferred from the filename.
RAP_GEN_NUMERICAL_VERSION = "rap-generative-v3-fp32-master-row"


def _sha256_of_module(module: object) -> str:
    path = Path(module.__file__)  # type: ignore[attr-defined]
    return hashlib.sha256(path.read_bytes()).hexdigest()
from _additive_common import (
    load_arm_readonly,
    refuse_if_adapter_incomplete,
    refuse_if_output_exists,
    write_additive_report,
)
from _objective_additive_common import (
    build_study_b_pools,
    expected_objective_output_path,
    load_source_objective_gate,
    study_b_adapter_dir,
    verify_adapter_base_model,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", default="dos",
                    choices=["dos", "context-extraction", "refusal-suppression", "belief"])
    ap.add_argument("--trigger", default="word", choices=["word", "phrase"])
    ap.add_argument("--poison-rate", type=float, default=0.05,
                    help="MUST match the rate the source objective-gate report and the "
                         "cached adapters were built with (0.05 for dos/context-extraction, "
                         "0.45 for refusal-suppression/belief in this project's existing runs)")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--n-base", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tag", default="_rap_gen_add_v3",
                    help="suffix for the output filename; must not collide with an "
                         "existing report")
    # Detector hyperparameters -- carried over from RAP's paper defaults as
    # UNVALIDATED CANDIDATES for this statistic; see
    # src/aiforensics/detectors/rap_generative.py's module docstring.
    ap.add_argument("--rap-token", default=None,
                    help="override the auto-resolved, re-verified defense word")
    ap.add_argument("--rap-probability-range", type=float, nargs=2, default=[-0.1, -0.3],
                    metavar=("UPPER", "LOWER"))
    ap.add_argument("--rap-lr", type=float, default=1e-2)
    ap.add_argument("--rap-epochs", type=int, default=5)
    ap.add_argument("--rap-batch-size", type=int, default=32)
    ap.add_argument("--rap-scale-factor", type=float, default=1.0)
    ap.add_argument("--rap-max-new-tokens", type=int, default=64,
                    help="reference-generation budget -- matches run_objective_attack.py's "
                         "own generate() budget")
    args = ap.parse_args()

    from aiforensics import REPORT_DIR, set_seed

    set_seed(args.seed)
    t_start = time.time()

    # ---- preflight: never touch the GPU on a doomed run ------------------
    out_path = expected_objective_output_path(
        report_dir=REPORT_DIR, objective=args.objective, trigger=args.trigger,
        poison_rate=args.poison_rate, seed=args.seed, tag=args.tag,
    )
    refuse_if_output_exists(out_path)

    cache_p = study_b_adapter_dir(objective=args.objective, trigger=args.trigger,
                                  poison_rate=args.poison_rate, seed=args.seed,
                                  n_base=args.n_base, arm="poisoned")
    cache_c = study_b_adapter_dir(objective=args.objective, trigger=args.trigger,
                                  poison_rate=args.poison_rate, seed=args.seed,
                                  n_base=args.n_base, arm="control")
    refuse_if_adapter_incomplete(cache_p)
    refuse_if_adapter_incomplete(cache_c)
    verify_adapter_base_model(cache_dir=cache_p, model_key=args.model)
    verify_adapter_base_model(cache_dir=cache_c, model_key=args.model)
    # n_base <-> adapter correspondence is guaranteed BY CONSTRUCTION (the
    # tag above is built directly from args.n_base, so this launcher cannot
    # load an adapter trained at a different corpus size than requested --
    # see _objective_additive_common.load_source_objective_gate's
    # docstring) -- printed here so that correspondence is visible, not just
    # structurally true.
    print(f"adapter n_base  : {args.n_base}  (encoded in cache tag: "
          f"poisoned={cache_p.name!r}, control={cache_c.name!r})")

    from aiforensics.detectors import get_detector

    try:
        get_detector("rap-generative")
    except NotImplementedError as exc:
        raise SystemExit(
            f"refusing to run: 'rap-generative' resolved to the framework's "
            f"planned-but-unimplemented placeholder ({exc}). The registration "
            f"import at the top of this file did not run -- verify it is present."
        ) from exc
    except ValueError:
        pass  # expected: the real class complains about missing model=/tokenizer=

    gate = load_source_objective_gate(
        report_dir=REPORT_DIR, objective=args.objective, trigger=args.trigger,
        poison_rate=args.poison_rate, seed=args.seed,
        n_base=args.n_base, n_eval=args.n_eval, model=args.model,
    )
    print(f"objective gate (from {gate['source_objective_report']}): "
          f"fired={gate['attack_fired']}  delta={gate['delta']:+.3f}")
    if not gate["attack_fired"]:
        print("  Detection numbers below are NOT interpretable. Reported anyway,")
        print("  marked, so the failure is visible rather than hidden.")

    # ---- load poisoned arm first (its tokenizer renders the pools) -------
    t0 = time.time()
    model_p, tok = load_arm_readonly(args.model, cache_p)
    print(f"  poisoned arm loaded (adapter: {cache_p})  [{time.time()-t0:.0f}s]")

    runtime_dtype = str(model_p.get_input_embeddings().weight.dtype)
    code_hashes = {
        "rap_generative.py": _sha256_of_module(aiforensics.detectors.rap_generative),
    }

    pools = build_study_b_pools(
        tok=tok, objective=args.objective, trigger=args.trigger, poison_rate=args.poison_rate,
        seed=args.seed, n_base=args.n_base, n_eval=args.n_eval,
    )
    print(f"eval pools     : {len(pools.eval_clean)} clean / {len(pools.eval_trig)} triggered")
    print(f"calibration    : {len(pools.cal_fit)} fit / {len(pools.cal_thr)} threshold (clean only)")
    print(f"corpus         : n_poisonable={pools.n_poisonable}  n_poisoned={pools.n_poisoned} "
          f"(CLI poison-rate {args.poison_rate:g} applies to the poisonABLE subset, not the "
          f"full n_base -- see the corpus notes below)")
    for k, v in pools.corpus_notes.items():
        print(f"  {k}: {v}")
    print(f"data duplication (existing protocol, not altered here): {pools.duplication}")

    t0 = time.time()
    model_c, tok_c = load_arm_readonly(args.model, cache_c)
    print(f"  control arm loaded  (adapter: {cache_c})  [{time.time()-t0:.0f}s]")

    from aiforensics.metrics.detection import calibrate_threshold_on_clean, detection_metrics, pr_auc, roc_auc

    def make(model, tok_):
        det = get_detector(
            "rap-generative", model=model, tokenizer=tok_,
            rap_token=args.rap_token, probability_range=tuple(args.rap_probability_range),
            lr=args.rap_lr, epochs=args.rap_epochs, batch_size=args.rap_batch_size,
            scale_factor=args.rap_scale_factor, max_new_tokens=args.rap_max_new_tokens,
            seed=args.seed,
        )
        det.fit(pools.cal_fit)
        fit_diagnostics = dict(det.config.get("fit_diagnostics", {}))
        cal_scores = det.score(pools.cal_thr)
        cal_excluded = dict(det.config.get("score_excluded_counts", {}))
        thr = calibrate_threshold_on_clean(cal_scores, args.target_fpr)
        return det, thr, fit_diagnostics, cal_excluded

    t0 = time.time()
    det_p, thr_p, fit_diag_p, cal_excl_p = make(model_p, tok)
    det_c, thr_c, fit_diag_c, cal_excl_c = make(model_c, tok_c)

    s_pd = np.asarray(det_p.score(pools.eval_trig), dtype=np.float64)
    excl_pd = dict(det_p.config.get("score_excluded_counts", {}))
    s_pc = np.asarray(det_p.score(pools.eval_clean), dtype=np.float64)
    excl_pc = dict(det_p.config.get("score_excluded_counts", {}))
    s_ct = np.asarray(det_c.score(pools.eval_trig), dtype=np.float64)
    excl_ct = dict(det_c.config.get("score_excluded_counts", {}))
    s_cc = np.asarray(det_c.score(pools.eval_clean), dtype=np.float64)
    excl_cc = dict(det_c.config.get("score_excluded_counts", {}))

    truth = np.r_[np.ones(len(s_pd), bool), np.zeros(len(s_pc), bool)]
    scr = np.r_[s_pd, s_pc]
    res = detection_metrics(truth, scr >= thr_p, warn=False)
    row = {
        "detector": "rap-generative", "objective": args.objective, "trigger": args.trigger,
        "poison_rate": args.poison_rate, "attack_fired": gate["attack_fired"],
        "n": len(truth), "base_rate": float(truth.mean()),
        "threshold_poisoned": float(thr_p), "threshold_control": float(thr_c),
        "fpr_budget": args.target_fpr,
        "detection_rate": float(res.detection_rate), "precision": float(res.precision),
        "f1": float(res.f1), "fpr": float(res.fpr),
        "roc_auc": float(roc_auc(truth, scr)), "pr_auc": float(pr_auc(truth, scr)),
        "ctrl_cleanFT_triggered": float((s_ct >= thr_c).mean()),
        "ctrl_cleanFT_clean": float((s_cc >= thr_c).mean()),
        "resolved_rap_token": det_p.rap_token, "resolved_rap_token_id": det_p.rap_token_id,
        "resolved_rap_token_control": det_c.rap_token, "resolved_rap_token_id_control": det_c.rap_token_id,
        "fit_diagnostics_poisoned": fit_diag_p, "fit_diagnostics_control": fit_diag_c,
        "score_excluded_counts": {
            "poisoned_calibration": cal_excl_p, "poisoned_eval_triggered": excl_pd,
            "poisoned_eval_clean": excl_pc, "control_calibration": cal_excl_c,
            "control_eval_triggered": excl_ct, "control_eval_clean": excl_cc,
        },
        "data_duplication": pools.duplication,
        "seconds": time.time() - t0,
    }
    print(f"  rap-generative  det={row['detection_rate']:.3f} prec={row['precision']:.3f} "
          f"roc={row['roc_auc']:.3f} ctrl(cleanFT,trig)={row['ctrl_cleanFT_triggered']:.3f}  "
          f"rap_token={row['resolved_rap_token']!r}  [{row['seconds']:.0f}s]")

    write_additive_report(out_path, {
        "config": vars(args), "adaptation": "rap-generative-v3",
        "adaptation_notes": (
            "v2 vs v1: unified scoring (no exclusion for a reference containing the "
            "RAP token; new exclusion only for the RAP token not surviving truncation "
            "or already present in the clean input, both counted); left-truncation "
            "formalised and its assistant-opener/RAP-token-survival tradeoff measured; "
            "full EOS set (not just tokenizer.eos_token_id) used for generation "
            "stopping; tied embeddings enforced (refuses an untied model) rather than "
            "silently copying into a separate output row; per-sample gradient "
            "accumulation in fit() (memory only, verified numerically equivalent to "
            "the prior whole-batch computation); fit_diagnostics now includes a "
            "genuine before/after full-clean-fit-set evaluation, not only the last "
            "mini batch pre-update loss; chatml_user_turn scope enforced explicitly, "
            "no full_prompt fallback accepted; adapter base-model and data-duplication "
            "checks added. "
            "v3 vs v2: the per-batch row update and norm rescale, previously applied "
            "directly to the BF16 embedding row (weight.data[row] -= lr*grad; a BF16 "
            "``.norm()`` for both the reference and the rescale), now happen in an "
            "FP32 master copy of the row, cast to the model's own dtype only when "
            "installed for a forward pass. Confirmed live (dos/phrase, seed 1337) that "
            "the v2 update rounded ~95-97% of the row's elements to no change at all "
            "each step, with the surviving ~3-5% only 0.24-0.39 cosine-similar to the "
            "true FP32 update direction; v2's norm rescale was a complete no-op at this "
            "scale. No v2 report used the FP32 master row; v2 reports are NOT "
            "re-derivable from this file and were left untouched."
        ),
        "rap_numerical_version": RAP_GEN_NUMERICAL_VERSION,
        "code_hashes": code_hashes,
        "runtime_dtype": {"embedding": runtime_dtype},
        "model_source": args.model, "adapter_poisoned": str(cache_p), "adapter_control": str(cache_c),
        "objective_gate": gate, "results": [row], "wall_clock_s": time.time() - t_start,
        "notes": (
            "RESEARCH ADAPTATION, not the classification RAP -- see "
            "src/aiforensics/detectors/rap_generative.py's module docstring for the full "
            "derivation, the corrected tied-embedding analysis, and every open caveat "
            "(fitting-hyperparameter validity was checked only on a small clean-data-only "
            "sample, not across a full fitting run; do not read this run's hyperparameters "
            "as independently re-tuned). Attack gate and evaluation/calibration pools are "
            "reused exactly from Minseok's original runner; this launcher never trains."
        ),
    })
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
