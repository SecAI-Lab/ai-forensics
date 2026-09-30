"""Additive-only launcher: score RAP against an already-trained Study A pair.

This script's job is narrower than ``scripts/run_uniguardian_additive.py``'s:
RAP's ``.fit()`` needs a ``labels=`` argument and its constructor needs
``verbalizer=``/``target_label=`` that Minseok's original
``scripts/run_backdoor_detection.py`` never supplies to any detector today --
so RAP cannot be run through that script unmodified, and this project's rule
is to never modify or monkeypatch it. Instead, this script reconstructs the
SAME data/eval/calibration pools that runner builds
(:mod:`scripts._additive_common`, read this session against
``run_backdoor_detection.py`` lines 260-359) and reuses its own adapter
cache-tag convention, so a poisoned/control adapter pair trained by that
script is scored here exactly as it would be scored there -- just with RAP's
extra required arguments plumbed through.

Nothing here trains a LoRA adapter, saves a model, or regenerates attack
data:

* Both arms' adapters must already exist (``adapter_config.json`` + a weight
  file) or this script raises before loading anything onto the GPU
  (:func:`scripts._additive_common.refuse_if_adapter_incomplete`).
* The attack gate (fired / ASR) is read from Minseok's own, already-written
  report for this exact condition, never recomputed
  (:func:`scripts._additive_common.load_source_attack_gate`).
* The output path is refused if it already exists
  (:func:`scripts._additive_common.refuse_if_output_exists`) -- pick a new
  ``--tag`` rather than overwrite.

Usage::

    PYTHONPATH=/workspace/src python scripts/run_rap_additive.py \\
        --trigger word --poison-rate 0.05 --seed 1337 \\
        --eval-cap 400 --target-fpr 0.05 --tag _rap_add_v2
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Registration side effect ONLY -- see aiforensics/detectors/base.py's
# _ensure_builtins()/_BUILTIN_MODULES (read-only, unmodified): "rap" is not
# in that tuple, so nothing makes get_detector("rap") resolve to the real
# class below unless this import has actually run first, exactly like
# run_uniguardian_additive.py's own `import aiforensics.detectors.uniguardian`.
# Missing this line is what made the first real run hit the framework's
# planned-but-unimplemented NotImplementedError instead of RapDetector.
import aiforensics.detectors.rap  # noqa: F401
from _additive_common import (
    assert_detector_registered,
    build_study_a_pools,
    expected_backdoor_output_path,
    load_arm_readonly,
    load_source_attack_gate,
    refuse_if_adapter_incomplete,
    refuse_if_output_exists,
    score_detector_cell,
    study_a_adapter_dir,
    write_additive_report,
)

#: Bumped when RapDetector's own numerical implementation changes in a way
#: that could change fitted-row precision -- v2 replaced a BF16-only batch
#: update/norm-rescale with an FP32 master row (src/aiforensics/detectors/
#: rap.py, module docstring, "row-update precision"). Recorded in every
#: report this launcher writes, never inferred from the output filename tag.
RAP_NUMERICAL_VERSION = "rap-v2-fp32-master-row"


def _sha256_of_module(module: object) -> str:
    path = Path(module.__file__)  # type: ignore[attr-defined]
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from aiforensics.attacks.textual_backdoor import TRIGGER_CLASSES

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trigger", default="word", choices=list(TRIGGER_CLASSES))
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--poison-mode", default="edit", choices=["edit", "insert"])
    ap.add_argument("--target-label", type=int, default=1)
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval-cap", type=int, default=400,
                    help="MUST match the --eval-cap the source attack-gate "
                         "report and the cached adapters were built with "
                         "(Section II: reuse the original eval-cap exactly)")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--tag", default="_rap_add_v2",
                    help="suffix for the output filename; must not collide "
                         "with an existing report")
    # RAP's own hyperparameters (paper defaults; see src/aiforensics/detectors/rap.py)
    ap.add_argument("--rap-token", default=None,
                    help="override the auto-resolved, re-verified defense word")
    ap.add_argument("--rap-probability-range", type=float, nargs=2, default=[-0.1, -0.3],
                    metavar=("UPPER", "LOWER"))
    ap.add_argument("--rap-lr", type=float, default=1e-2)
    ap.add_argument("--rap-epochs", type=int, default=5)
    ap.add_argument("--rap-batch-size", type=int, default=32)
    ap.add_argument("--rap-scale-factor", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=16,
                    help="RAP scoring batch size -- pure batching, never changes the result")
    args = ap.parse_args()

    from aiforensics import REPORT_DIR, set_seed

    set_seed(args.seed)
    t_start = time.time()

    # ---- preflight: never touch the GPU on a doomed run ------------------
    out_path = expected_backdoor_output_path(
        report_dir=REPORT_DIR, trigger=args.trigger, poison_rate=args.poison_rate,
        poison_mode=args.poison_mode, seed=args.seed, tag=args.tag,
    )
    refuse_if_output_exists(out_path)

    cache_p = study_a_adapter_dir(dataset=args.dataset, trigger=args.trigger,
                                  poison_rate=args.poison_rate, poison_mode=args.poison_mode,
                                  seed=args.seed, arm="poisoned")
    cache_c = study_a_adapter_dir(dataset=args.dataset, trigger=args.trigger,
                                  poison_rate=args.poison_rate, poison_mode=args.poison_mode,
                                  seed=args.seed, arm="control")
    refuse_if_adapter_incomplete(cache_p)
    refuse_if_adapter_incomplete(cache_c)

    # Registration self-check -- no model/tokenizer needed. Catches a missing
    # `import aiforensics.detectors.rap` registration side effect (above)
    # before either adapter reaches the GPU, rather than after.
    assert_detector_registered("rap")

    gate = load_source_attack_gate(
        report_dir=REPORT_DIR, trigger=args.trigger, poison_rate=args.poison_rate,
        poison_mode=args.poison_mode, seed=args.seed, dataset=args.dataset,
        target_label=args.target_label,
    )
    print(f"attack gate (from {gate['source_attack_report']}): "
          f"fired={gate['attack_fired']}  delta_asr={gate['delta_asr']:+.3f}")
    if not gate["attack_fired"]:
        print("  Detection numbers below are NOT interpretable. Reported anyway,")
        print("  marked, so the failure is visible rather than hidden.")

    # ---- pools, byte-identical to run_backdoor_detection.py ---------------
    pools = build_study_a_pools(
        trigger=args.trigger, poison_rate=args.poison_rate, poison_mode=args.poison_mode,
        target_label=args.target_label, seed=args.seed, eval_cap=args.eval_cap,
        dataset=args.dataset,
    )
    fmt = pools.fmt
    print(f"trigger        : {pools.trig.label}")
    print(f"eval pools     : {len(pools.eval_clean)} clean / {len(pools.eval_trig)} triggered")
    print(f"calibration    : {len(pools.calib_fit_texts)} fit / "
          f"{len(pools.calib_thr_texts)} threshold (clean dev only)")

    calib_fit_prompts = [fmt.prompt(t) for t in pools.calib_fit_texts]
    calib_thr_prompts = [fmt.prompt(t) for t in pools.calib_thr_texts]

    # ---- load both arms, read-only (refuses above already guaranteed the
    # adapters exist; this is where they actually reach the GPU) -----------
    from aiforensics.detectors import get_detector
    from aiforensics.metrics.detection import calibrate_threshold_on_clean

    t0 = time.time()
    model_p, tok_p = load_arm_readonly(args.model, cache_p)
    print(f"  poisoned arm loaded (adapter: {cache_p})  [{time.time()-t0:.0f}s]")
    t0 = time.time()
    model_c, tok_c = load_arm_readonly(args.model, cache_c)
    print(f"  control arm loaded  (adapter: {cache_c})  [{time.time()-t0:.0f}s]")

    import aiforensics.detectors._verbalizer_common as _verbalizer_common_mod

    runtime_dtype = str(model_p.get_input_embeddings().weight.dtype)
    code_hashes = {
        "rap.py": _sha256_of_module(aiforensics.detectors.rap),
        "_verbalizer_common.py": _sha256_of_module(_verbalizer_common_mod),
    }

    def make(model, tok):
        det = get_detector(
            "rap", model=model, tokenizer=tok, verbalizer=fmt.verbalizer,
            target_label=args.target_label, template=fmt.template,
            rap_token=args.rap_token,
            probability_range=tuple(args.rap_probability_range),
            lr=args.rap_lr, epochs=args.rap_epochs, batch_size=args.rap_batch_size,
            scale_factor=args.rap_scale_factor, max_length=fmt.max_length,
            seed=args.seed, chunk_size=args.chunk_size,
        )
        det.fit(calib_fit_prompts, labels=pools.calib_fit_labels)
        thr = calibrate_threshold_on_clean(
            det.score(calib_thr_prompts), args.target_fpr,
        )
        return det, thr

    t0 = time.time()
    det_p, thr_p = make(model_p, tok_p)
    det_c, thr_c = make(model_c, tok_c)

    row = score_detector_cell(
        det_p=det_p, det_c=det_c, thr_p=thr_p, thr_c=thr_c, fmt=fmt,
        eval_trig=pools.eval_trig, eval_clean=pools.eval_clean,
        detector_name="rap", trigger=args.trigger, poison_rate=args.poison_rate,
        target_fpr=args.target_fpr, attack_fired=gate["attack_fired"],
        seconds=time.time() - t0,
    )
    print(f"  rap          det={row['detection_rate']:.3f} prec={row['precision']:.3f} "
          f"roc={row['roc_auc']:.3f} ctrl(cleanFT,trig)={row['ctrl_cleanFT_triggered']:.3f}  "
          f"[{row['seconds']:.0f}s]")

    write_additive_report(out_path, {
        "config": vars(args), "attack": gate["attack"], "attack_fired": gate["attack_fired"],
        "delta_asr": gate["delta_asr"], "source_attack_report": gate["source_attack_report"],
        "results": [row], "wall_clock_s": time.time() - t_start,
        "rap_numerical_version": RAP_NUMERICAL_VERSION,
        "code_hashes": code_hashes,
        "runtime_dtype": {"embedding": runtime_dtype},
        "notes": (
            "RAP score is a class-probability drop under a teacher-forced "
            "verbalizer, adapting the paper's BERT-softmax statistic to this "
            "project's causal LM (see src/aiforensics/detectors/rap.py). "
            "Attack gate and evaluation/calibration pools are reused exactly "
            "from Minseok's original runner; this launcher never trains. "
            f"Numerical implementation: {RAP_NUMERICAL_VERSION} -- fit()'s "
            "per-step row update and norm rescale run in an FP32 master copy "
            "of the row, cast to the model's own dtype only when installed "
            "for a forward pass (replacing a prior BF16-only update/rescale "
            "confirmed, on live audit, to lose a measurable fraction of the "
            "computed update to BF16 rounding -- see rap.py's module "
            "docstring, 'row-update precision'). No report predating this "
            "tag used the FP32 master row."
        ),
    })
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
