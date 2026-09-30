"""Additive-only launcher: score STRIP-ViTA against an already-trained Study A pair.

Mirrors ``scripts/run_rap_additive.py`` exactly in structure and shares its
Study A data/eval/calibration reconstruction and adapter/output guards
(:mod:`scripts._additive_common`) -- see that script's own docstring for why
this cannot simply run through Minseok's original
``scripts/run_backdoor_detection.py`` unmodified (STRIP-ViTA's ``.fit()``
also needs ``labels=``, and its constructor also needs
``verbalizer=``/``target_label=`` that runner never supplies).

Nothing here trains a LoRA adapter, saves a model, or regenerates attack
data -- both arms' adapters must already exist, the attack gate is read from
Minseok's own already-written report, and the output path is refused if it
already exists.

Cost note: STRIP-ViTA runs ``perturbation_number`` (default 20) full forward
passes per scored input on top of the gate check, so a full 400+400
condition here does noticeably more model work than RAP's single
forward-pass-per-sample statistic -- expect it to run longer, not to have
stalled.

Usage::

    PYTHONPATH=/workspace/src python scripts/run_strip_vita_additive.py \\
        --trigger word --poison-rate 0.05 --seed 1337 \\
        --eval-cap 400 --target-fpr 0.05 --tag _strip_vita_add
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
# _ensure_builtins()/_BUILTIN_MODULES (read-only, unmodified): "strip-vita"
# is not in that tuple, so nothing makes get_detector("strip-vita") resolve
# to the real class below unless this import has actually run first, exactly
# like run_uniguardian_additive.py's own
# `import aiforensics.detectors.uniguardian`. The RAP launcher's first real
# run failed on precisely this omission -- fixed here before it could bite
# STRIP-ViTA too.
import aiforensics.detectors.strip_vita  # noqa: F401
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

#: Provenance metadata, matching the pattern established in
#: run_rap_additive.py after its FP32-master-row fix. No STRIP-ViTA
#: algorithm has changed as of this value -- only docstring corrections
#: (provenance labeling, split(" ") accuracy, the truncation-vs-word-count
#: claim). Recorded so a future scoring-affecting change has something to
#: bump from, not because anything here changed the score.
STRIP_VITA_CODE_VERSION = "strip-vita-v1-docstring-only"


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
    ap.add_argument("--tag", default="_strip_vita_add",
                    help="suffix for the output filename; must not collide "
                         "with an existing report")
    # STRIP-ViTA's own hyperparameters (paper/code defaults; see
    # src/aiforensics/detectors/strip_vita.py)
    ap.add_argument("--strip-replace-ratio", type=float, default=0.7)
    ap.add_argument("--strip-perturbation-number", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=8,
                    help="scored inputs per gate-check batch -- pure batching, "
                         "never changes the result")
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
    # `import aiforensics.detectors.strip_vita` registration side effect
    # (above) before either adapter reaches the GPU, rather than after.
    assert_detector_registered("strip-vita")

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
    n_opposite = sum(1 for y in pools.calib_fit_labels if y != args.target_label)
    print(f"  of which {n_opposite} calib-fit rows are opposite-class "
          f"(STRIP-ViTA's TF-IDF reference pool)")

    calib_fit_prompts = [fmt.prompt(t) for t in pools.calib_fit_texts]
    calib_thr_prompts = [fmt.prompt(t) for t in pools.calib_thr_texts]

    # ---- load both arms, read-only ----------------------------------------
    from aiforensics.detectors import get_detector
    from aiforensics.metrics.detection import calibrate_threshold_on_clean

    t0 = time.time()
    model_p, tok_p = load_arm_readonly(args.model, cache_p)
    print(f"  poisoned arm loaded (adapter: {cache_p})  [{time.time()-t0:.0f}s]")
    t0 = time.time()
    model_c, tok_c = load_arm_readonly(args.model, cache_c)
    print(f"  control arm loaded  (adapter: {cache_c})  [{time.time()-t0:.0f}s]")

    runtime_dtype = str(model_p.get_input_embeddings().weight.dtype)
    code_hashes = {"strip_vita.py": _sha256_of_module(aiforensics.detectors.strip_vita)}

    def make(model, tok):
        det = get_detector(
            "strip-vita", model=model, tokenizer=tok, verbalizer=fmt.verbalizer,
            target_label=args.target_label, template=fmt.template,
            replace_ratio=args.strip_replace_ratio,
            perturbation_number=args.strip_perturbation_number,
            max_length=fmt.max_length, seed=args.seed, chunk_size=args.chunk_size,
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
        detector_name="strip-vita", trigger=args.trigger, poison_rate=args.poison_rate,
        target_fpr=args.target_fpr, attack_fired=gate["attack_fired"],
        seconds=time.time() - t0,
    )
    print(f"  strip-vita   det={row['detection_rate']:.3f} prec={row['precision']:.3f} "
          f"roc={row['roc_auc']:.3f} ctrl(cleanFT,trig)={row['ctrl_cleanFT_triggered']:.3f}  "
          f"[{row['seconds']:.0f}s]")

    write_additive_report(out_path, {
        "config": vars(args), "attack": gate["attack"], "attack_fired": gate["attack_fired"],
        "delta_asr": gate["delta_asr"], "source_attack_report": gate["source_attack_report"],
        "results": [row], "wall_clock_s": time.time() - t_start,
        "strip_vita_code_version": STRIP_VITA_CODE_VERSION,
        "code_hashes": code_hashes,
        "runtime_dtype": {"embedding": runtime_dtype},
        "notes": (
            "STRIP-ViTA score is -mean(entropy) of perturbed-copy verbalizer "
            "predictions, adapting the paper's BERT-softmax statistic and its "
            "own percentile threshold to this project's causal LM and unified "
            "calibrate_threshold_on_clean convention (see "
            "src/aiforensics/detectors/strip_vita.py). This is an ADAPTATION "
            "of STRIP-ViTA's mechanism, not a reproduction of the original "
            "paper's BERT-classifier experiments. Attack gate and "
            "evaluation/calibration pools are reused exactly from Minseok's "
            "original runner; this launcher never trains."
        ),
    })
    print(f"\nwritten: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
