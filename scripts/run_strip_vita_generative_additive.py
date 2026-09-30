"""Additive-only launcher: score the STRIP-ViTA generative adaptation
against an already-trained Study B pair.

Mirrors ``scripts/run_rap_generative_additive.py`` exactly in structure and
shares its Study B data/eval/calibration reconstruction and adapter/output
guards (``scripts/_objective_additive_common.py``). See
``src/aiforensics/detectors/strip_vita_generative.py``'s module docstring
for the full derivation, the EOS/length correction, and the open
length-entropy caveat (recorded per run in this launcher's output, not
resolved).

Nothing here trains a LoRA adapter, saves a model, or regenerates attack
data -- both arms' adapters must already exist, the objective gate is read
from Minseok's own already-written report, and the output path is refused
if it already exists.

Cost note: this detector runs ``perturbation_number`` (default 20) full
GENERATIONS (not just forward passes) per scored input, on top of the
teacher-forced re-scoring pass for each -- substantially more expensive
than the classification version's own per-input cost. Expect a full
400-prompt condition to take a while, not to have stalled.

Usage::

    PYTHONPATH=/workspace/src python scripts/run_strip_vita_generative_additive.py \\
        --objective dos --trigger word --poison-rate 0.05 \\
        --seed 1337 --target-fpr 0.05 --tag _strip_vita_gen_add_v2
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

# Registration side effect ONLY -- "strip-vita-generative" is not in
# aiforensics/detectors/base.py's _BUILTIN_MODULES (read-only, unmodified).
import aiforensics.detectors.strip_vita_generative  # noqa: F401
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

#: Provenance metadata, matching the pattern established in
#: run_rap_generative_additive.py after its FP32-master-row fix. No
#: STRIP-ViTA-generative algorithm has changed as of this value -- only
#: docstring corrections (provenance labeling, the word-count-vs-token-count
#: truncation claim, natural-log framing). Recorded so a future
#: scoring-affecting change has something to bump from.
STRIP_VITA_GEN_CODE_VERSION = "strip-vita-generative-v1-docstring-only"


def _sha256_of_module(module: object) -> str:
    path = Path(module.__file__)  # type: ignore[attr-defined]
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    ap.add_argument("--tag", default="_strip_vita_gen_add_v2",
                    help="suffix for the output filename; must not collide with an "
                         "existing report")
    # Detector hyperparameters -- unchanged from the classification
    # version's own defaults, per this session's explicit instruction.
    ap.add_argument("--strip-replace-ratio", type=float, default=0.7)
    ap.add_argument("--strip-perturbation-number", type=int, default=20)
    ap.add_argument("--strip-max-new-tokens", type=int, default=64,
                    help="per-perturbed-copy generation budget -- matches "
                         "run_objective_attack.py's own generate() budget")
    args = ap.parse_args()

    from aiforensics import REPORT_DIR, set_seed

    set_seed(args.seed)
    t_start = time.time()

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
        get_detector("strip-vita-generative")
    except NotImplementedError as exc:
        raise SystemExit(
            f"refusing to run: 'strip-vita-generative' resolved to the framework's "
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

    t0 = time.time()
    model_p, tok = load_arm_readonly(args.model, cache_p)
    print(f"  poisoned arm loaded (adapter: {cache_p})  [{time.time()-t0:.0f}s]")

    pools = build_study_b_pools(
        tok=tok, objective=args.objective, trigger=args.trigger, poison_rate=args.poison_rate,
        seed=args.seed, n_base=args.n_base, n_eval=args.n_eval,
    )
    print(f"eval pools     : {len(pools.eval_clean)} clean / {len(pools.eval_trig)} triggered")
    print(f"calibration    : {len(pools.cal_fit)} fit / {len(pools.cal_thr)} threshold (clean only)")
    print(f"corpus         : n_poisonable={pools.n_poisonable}  n_poisoned={pools.n_poisoned}")
    for k, v in pools.corpus_notes.items():
        print(f"  {k}: {v}")
    print(f"  cost note: {args.strip_perturbation_number} generations/scored input; "
          f"{len(pools.eval_clean) + len(pools.eval_trig)} scored inputs per arm")
    print(f"data duplication (existing protocol, not altered here): {pools.duplication}")

    t0 = time.time()
    model_c, tok_c = load_arm_readonly(args.model, cache_c)
    print(f"  control arm loaded  (adapter: {cache_c})  [{time.time()-t0:.0f}s]")

    runtime_dtype = str(model_p.get_input_embeddings().weight.dtype)
    code_hashes = {
        "strip_vita_generative.py": _sha256_of_module(aiforensics.detectors.strip_vita_generative),
    }

    from aiforensics.metrics.detection import calibrate_threshold_on_clean, detection_metrics, pr_auc, roc_auc

    def make(model, tok_):
        det = get_detector(
            "strip-vita-generative", model=model, tokenizer=tok_,
            replace_ratio=args.strip_replace_ratio,
            perturbation_number=args.strip_perturbation_number,
            max_new_tokens=args.strip_max_new_tokens, seed=args.seed,
        )
        det.fit(pools.cal_fit)
        thr = calibrate_threshold_on_clean(det.score(pools.cal_thr), args.target_fpr)
        return det, thr

    t0 = time.time()
    det_p, thr_p = make(model_p, tok)
    det_c, thr_c = make(model_c, tok_c)

    s_pd = np.asarray(det_p.score(pools.eval_trig), dtype=np.float64)
    lengths_trig = list(det_p.config.get("generated_lengths", []))
    s_pc = np.asarray(det_p.score(pools.eval_clean), dtype=np.float64)
    lengths_clean = list(det_p.config.get("generated_lengths", []))
    s_ct = np.asarray(det_c.score(pools.eval_trig), dtype=np.float64)
    s_cc = np.asarray(det_c.score(pools.eval_clean), dtype=np.float64)

    def _score_length_corr(scores, lengths_nested):
        # Sufficient statistics to check the length/score relationship
        # (module docstring's open caveat) WITHOUT storing any generated
        # text: per-sample MEAN generated length (already just integers,
        # never the text itself) against that sample's score.
        mean_len = np.array([float(np.mean(row)) for row in lengths_nested])
        if len(mean_len) < 2 or np.std(mean_len) == 0 or np.std(scores) == 0:
            return {"mean_length_per_sample": mean_len.tolist(), "corr_with_score": None}
        corr = float(np.corrcoef(scores, mean_len)[0, 1])
        return {"mean_length_per_sample": mean_len.tolist(), "corr_with_score": corr}

    truth = np.r_[np.ones(len(s_pd), bool), np.zeros(len(s_pc), bool)]
    scr = np.r_[s_pd, s_pc]
    res = detection_metrics(truth, scr >= thr_p, warn=False)

    # Open caveat (module docstring): record generated-length summary
    # statistics alongside the score, so a length/entropy relationship is
    # inspectable in the output rather than invisible. NOT used to pick or
    # adjust anything in this run.
    def _len_stats(lengths_nested):
        flat = [n for row in lengths_nested for n in row]
        return {"mean": float(np.mean(flat)), "min": int(np.min(flat)), "max": int(np.max(flat))} if flat else {}

    row = {
        "detector": "strip-vita-generative", "objective": args.objective, "trigger": args.trigger,
        "poison_rate": args.poison_rate, "attack_fired": gate["attack_fired"],
        "n": len(truth), "base_rate": float(truth.mean()),
        "threshold_poisoned": float(thr_p), "threshold_control": float(thr_c),
        "fpr_budget": args.target_fpr,
        "detection_rate": float(res.detection_rate), "precision": float(res.precision),
        "f1": float(res.f1), "fpr": float(res.fpr),
        "roc_auc": float(roc_auc(truth, scr)), "pr_auc": float(pr_auc(truth, scr)),
        "ctrl_cleanFT_triggered": float((s_ct >= thr_c).mean()),
        "ctrl_cleanFT_clean": float((s_cc >= thr_c).mean()),
        "generated_length_stats_triggered": _len_stats(lengths_trig),
        "generated_length_stats_clean": _len_stats(lengths_clean),
        "length_score_correlation_triggered": _score_length_corr(s_pd, lengths_trig),
        "length_score_correlation_clean": _score_length_corr(s_pc, lengths_clean),
        "data_duplication": pools.duplication,
        "seconds": time.time() - t0,
    }
    print(f"  strip-vita-gen  det={row['detection_rate']:.3f} prec={row['precision']:.3f} "
          f"roc={row['roc_auc']:.3f} ctrl(cleanFT,trig)={row['ctrl_cleanFT_triggered']:.3f}  "
          f"[{row['seconds']:.0f}s]")

    write_additive_report(out_path, {
        "config": vars(args), "adaptation": "strip-vita-generative-v2",
        "adaptation_notes": (
            "v2 vs v1: chatml_user_turn scope enforced explicitly, no full_prompt "
            "fallback accepted; full EOS set (not just tokenizer.eos_token_id) used "
            "for generation stopping; empty generation / non-finite entropy / a "
            "reduced perturbation count now raise instead of silently degrading to a "
            "0.0 score (which previously ranked ABOVE genuine low-entropy detections, "
            "since 0 sits at the high-suspicion end of -mean(entropy)); nan_to_num "
            "removed; replacement count validated against the fitted vocabulary size "
            "before calling perturb_one; docstring corrected on which words a "
            "perturbed copy's replacements can come from; adapter base-model and "
            "data-duplication checks added; per-sample length/score correlation "
            "recorded."
        ),
        "model_source": args.model, "adapter_poisoned": str(cache_p), "adapter_control": str(cache_c),
        "objective_gate": gate, "results": [row], "wall_clock_s": time.time() - t_start,
        "strip_vita_code_version": STRIP_VITA_GEN_CODE_VERSION,
        "code_hashes": code_hashes,
        "runtime_dtype": {"embedding": runtime_dtype},
        "notes": (
            "RESEARCH ADAPTATION, not the classification STRIP-ViTA, and not the STRIP "
            "baseline bundled in the RAP repository -- see "
            "src/aiforensics/detectors/strip_vita_generative.py's module docstring for the "
            "full derivation and the open length/entropy caveat (generated_length_stats_* "
            "above are diagnostic, recorded so this can be inspected on real results, not "
            "resolved by this run). Attack gate and evaluation/calibration pools are reused "
            "exactly from Minseok's original runner; this launcher never trains."
        ),
    })
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
