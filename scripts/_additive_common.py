"""Shared, read-only Study A orchestration for the "_additive" launchers.

``run_rap_additive.py`` and ``run_strip_vita_additive.py`` both need to
reconstruct EXACTLY the same (trigger, poison-rate, seed) evaluation and
calibration pools that ``scripts/run_backdoor_detection.py`` (Minseok's
original, unmodified runner) builds and scores against -- same rows, same
order, same split -- without ever calling that script's own training path.
Sharing this module is what keeps that reconstruction in ONE place instead of
two, so RAP and STRIP-ViTA cannot silently drift apart on data handling while
their detector-specific code evolves independently.

Nothing in this module trains a LoRA adapter, saves a model, or regenerates
attack data. It only loads already-cached adapters (read-only, hard failure
if missing or incomplete) and already-written attack-gate reports (read-only,
hard failure if missing or inconsistent with the requested condition).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "StudyAPools",
    "build_study_a_pools",
    "study_a_adapter_dir",
    "refuse_if_adapter_incomplete",
    "load_arm_readonly",
    "expected_backdoor_output_path",
    "refuse_if_output_exists",
    "load_source_attack_gate",
    "assert_detector_registered",
    "score_detector_cell",
    "write_additive_report",
]


def assert_detector_registered(name: str) -> None:
    """Fail fast, before any adapter reaches the GPU, if ``name`` still
    resolves to the framework's PLANNED-but-unimplemented placeholder
    instead of a real detector class.

    ``get_detector()`` only sees a real implementation once the module that
    calls ``@register_detector(name, ...)`` has actually been imported
    somewhere in the process (``base.py``'s ``_ensure_builtins()`` only
    imports the modules already named in its own ``_BUILTIN_MODULES``
    tuple, which does not include ``rap`` or ``strip_vita`` -- editing that
    tuple is out of scope, same as editing ``base.py`` at all). Each
    launcher is responsible for its own top-level
    ``import aiforensics.detectors.<module>`` registration side effect,
    mirroring ``run_uniguardian_additive.py``'s own pattern exactly; this
    function only checks that it actually ran, with no model/tokenizer
    required, so a missing import surfaces here instead of after two LoRA
    adapters have already been loaded onto the GPU.
    """
    from aiforensics.detectors import get_detector

    try:
        get_detector(name)
    except NotImplementedError as exc:
        raise SystemExit(
            f"refusing to run: detector {name!r} resolved to the "
            f"framework's planned-but-unimplemented placeholder, not a real "
            f"class ({exc}). This launcher's own top-level "
            f"'import aiforensics.detectors.{name.replace('-', '_')}' "
            f"registration side effect did not run before this check -- "
            f"verify it is present, unconditional, and spelled correctly."
        ) from exc
    except ValueError:
        # Expected: the real class's __init__ complains about missing
        # model=/tokenizer=, which is only possible if it WAS resolved.
        pass


# --------------------------------------------------------------------------- #
# Exact reconstruction of run_backdoor_detection.py's data/eval/calibration
# pools (lines 260-359 of that file, read this session; re-check there if
# Minseok's runner is ever updated -- this module intentionally duplicates
# that logic rather than importing main(), which cannot be called partially).
# --------------------------------------------------------------------------- #
@dataclass
class StudyAPools:
    fmt: Any
    trig: Any
    eval_clean: list[str]
    eval_trig: list[str]
    calib_fit_texts: list[str]
    calib_fit_labels: list[int]
    calib_thr_texts: list[str]


def build_study_a_pools(
    *,
    trigger: str,
    poison_rate: float,
    poison_mode: str,
    target_label: int,
    seed: int,
    eval_cap: int,
    dataset: str = "sst2",
) -> StudyAPools:
    """Reconstruct the evaluation and calibration pools, byte-for-byte.

    This calls :func:`numpy.random.default_rng` and consumes it in EXACTLY
    the same sequence ``run_backdoor_detection.py:main()`` does -- including
    the training-row poison draw (``rng.choice(cand, ...)``) whose RESULT
    this function discards, because that draw still has to happen to advance
    the shared generator's state to the point where the eval-pool and
    calibration draws land on the same indices the original run used. Skipping
    it would silently score a DIFFERENT eval pool than the one the cached
    adapter was actually evaluated against.

    Raises:
        SystemExit: With the same messages ``run_backdoor_detection.py``
            itself would raise, for the same conditions (poison rate exceeds
            the trigger's eligible pool; too few eligible triggered test
            rows).
    """
    from aiforensics.attacks.textual_backdoor import build_trigger, load_sst2_sentence
    from aiforensics.training import get_format

    fmt = get_format(dataset)
    rng = np.random.default_rng(seed)

    tr_texts, tr_labels = load_sst2_sentence("train")
    dev_texts, dev_labels = load_sst2_sentence("dev")
    te_texts, te_labels = load_sst2_sentence("test")

    trig = build_trigger(trigger, tr_texts, split="train", seed=seed, target_label=target_label)
    cand = trig.candidates(tr_labels, target_label)
    k = round(poison_rate * len(tr_texts))
    if k > len(cand):
        raise SystemExit(
            f"poison rate {poison_rate:.1%} needs {k} rows but the {trigger!r} "
            f"trigger has only {len(cand)} eligible (max rate "
            f"{trig.max_rate(tr_labels, target_label):.1%})."
        )
    # State-only: this read-only entry point never trains, so `chosen` itself
    # is never used -- only consuming this draw matters. See the docstring.
    _chosen = np.sort(rng.choice(cand, size=k, replace=False))
    del _chosen

    nt_idx = [i for i, y in enumerate(te_labels) if y != target_label]
    trig_te = build_trigger(trigger, te_texts, split="test", seed=seed, target_label=target_label)
    nt_idx = [i for i in nt_idx if trig_te.available[i]]
    apply_fn = trig_te.fn
    nt_idx = list(rng.permutation(nt_idx)[:eval_cap])
    if len(nt_idx) < 50:
        raise SystemExit(
            f"only {len(nt_idx)} eligible triggered test rows for {trigger!r} "
            "-- too few to measure a detection rate against."
        )
    eval_clean = [te_texts[i] for i in nt_idx]
    eval_trig = [apply_fn(t) for t in eval_clean]

    perm = rng.permutation(len(dev_texts))
    cut = int(0.8 * len(perm))
    calib_fit_texts = [dev_texts[i] for i in perm[:cut]]
    calib_fit_labels = [int(dev_labels[i]) for i in perm[:cut]]
    calib_thr_texts = [dev_texts[i] for i in perm[cut:]]

    return StudyAPools(
        fmt=fmt, trig=trig, eval_clean=eval_clean, eval_trig=eval_trig,
        calib_fit_texts=calib_fit_texts, calib_fit_labels=calib_fit_labels,
        calib_thr_texts=calib_thr_texts,
    )


# --------------------------------------------------------------------------- #
# Adapter access -- read-only, hard failure if not already trained.
# --------------------------------------------------------------------------- #
def study_a_adapter_dir(
    *, dataset: str, trigger: str, poison_rate: float, poison_mode: str, seed: int, arm: str
) -> Path:
    """Reproduce ``run_backdoor_detection.py``'s own adapter cache-tag
    templates (lines 373-378) exactly, for BOTH arms."""
    from aiforensics import ADAPTER_DIR

    mode_sfx = "" if poison_mode == "edit" else "-ins"
    if arm == "control":
        tag = f"{dataset}-control-s{seed}"
    elif arm == "poisoned":
        tag = f"{dataset}-{trigger}-p{poison_rate:g}{mode_sfx}-s{seed}"
    else:
        raise ValueError(f"arm must be 'poisoned' or 'control', got {arm!r}")
    return ADAPTER_DIR / tag


def refuse_if_adapter_incomplete(cache_dir: Path) -> None:
    """Hard failure if ``cache_dir`` is missing ``adapter_config.json`` or
    every known weight-file name -- never a silent fall-through to training.
    """
    cfg = cache_dir / "adapter_config.json"
    if not cfg.exists():
        raise SystemExit(
            f"refusing to run: no adapter_config.json at {cache_dir} -- this "
            f"launcher never trains. Train it first with Minseok's original "
            f"runner, or pick a condition whose adapter already exists."
        )
    weight_names = ("adapter_model.safetensors", "adapter_model.bin")
    if not any((cache_dir / name).exists() for name in weight_names):
        raise SystemExit(
            f"refusing to run: adapter_config.json exists at {cache_dir} but "
            f"no weight file ({', '.join(weight_names)}) does -- treating this "
            f"as an incomplete adapter, not something to (re)train."
        )


def load_arm_readonly(model_key: str, cache_dir: Path):
    """Load the base model and attach the cached adapter -- NEVER falls
    through to training, unlike ``run_backdoor_detection.py``'s own
    ``train_arm`` (which silently trains on a cache miss). Raises via
    :func:`refuse_if_adapter_incomplete` first.

    Returns:
        ``(model, tokenizer)``, in eval mode, with the adapter attached
        read-only (``is_trainable=False``, the same default
        ``attach_adapter`` itself uses).
    """
    from aiforensics.models.llm import load_llm
    from aiforensics.models.peft_utils import attach_adapter

    refuse_if_adapter_incomplete(cache_dir)
    model, tok = load_llm(model_key, eval_mode=True)
    model = attach_adapter(model, str(cache_dir))
    model.eval()
    return model, tok


# --------------------------------------------------------------------------- #
# Output-collision guard -- refuse, never overwrite or delete.
# --------------------------------------------------------------------------- #
def expected_backdoor_output_path(
    *, report_dir: Path, trigger: str, poison_rate: float, poison_mode: str, seed: int, tag: str
) -> Path:
    """Reproduce ``run_backdoor_detection.py``'s own output-filename template
    (lines 541-543) exactly, with a caller-chosen ``tag`` so this launcher's
    output can never collide with the original runner's own report."""
    mode_sfx = "" if poison_mode == "edit" else "-ins"
    return report_dir / f"backdoor_detection_{trigger}_p{poison_rate:g}_s{seed}{mode_sfx}{tag}.json"


def refuse_if_output_exists(path: Path) -> None:
    if path.exists():
        raise SystemExit(
            f"refusing to run: output already exists and would be overwritten:\n"
            f"  {path}\nChoose a --tag that has not been used before."
        )


# --------------------------------------------------------------------------- #
# Attack-gate provenance -- read the ALREADY-COMPUTED outcome, never
# recompute it (recomputing needs a real generation pass, which this
# launcher does not run today, and Section II requires reusing the gate
# exactly as Minseok's runner already recorded it).
# --------------------------------------------------------------------------- #
def load_source_attack_gate(
    *,
    report_dir: Path,
    trigger: str,
    poison_rate: float,
    poison_mode: str,
    seed: int,
    dataset: str,
    target_label: int,
) -> dict[str, Any]:
    """Read ``attack`` / ``attack_fired`` / ``delta_asr`` from Minseok's own,
    already-written, untagged report for this exact condition, and cross-check
    its recorded config against the condition this launcher was asked to run.

    Raises:
        SystemExit: If the source report is missing, or its recorded config
            does not match (trigger/poison_rate/poison_mode/seed/dataset/
            target_label) -- this is a hard stop, never a silent fall back to
            recomputing the gate (which would need a real generation pass).
    """
    mode_sfx = "" if poison_mode == "edit" else "-ins"
    src = report_dir / f"backdoor_detection_{trigger}_p{poison_rate:g}_s{seed}{mode_sfx}.json"
    if not src.exists():
        raise SystemExit(
            f"refusing to run: no source attack-gate report at {src}. This "
            f"launcher reads the attack gate from Minseok's original runner's "
            f"own output rather than recomputing it (Section II: reuse the "
            f"attack gate exactly) -- run that condition with the original "
            f"runner first."
        )
    data = json.loads(src.read_text())
    cfg = data.get("config", {})
    checks = {
        "trigger": (cfg.get("trigger"), trigger),
        "poison_rate": (cfg.get("poison_rate"), poison_rate),
        "poison_mode": (cfg.get("poison_mode", "edit"), poison_mode),
        "seed": (cfg.get("seed"), seed),
        "dataset": (cfg.get("dataset"), dataset),
        "target_label": (cfg.get("target_label"), target_label),
    }
    mismatches = {k: v for k, v in checks.items() if v[0] != v[1]}
    if mismatches:
        raise SystemExit(
            f"refusing to run: source report {src} was recorded for a "
            f"DIFFERENT condition than requested -- {mismatches}. Not "
            f"trusting a filename match alone."
        )
    return {
        "attack": data["attack"], "attack_fired": data["attack_fired"],
        "delta_asr": data["delta_asr"], "source_attack_report": str(src),
    }


# --------------------------------------------------------------------------- #
# Per-detector scoring cell -- mirrors run_backdoor_detection.py's per-sample
# branch (lines 470-509) exactly, for a detector that is NOT pool-level.
# --------------------------------------------------------------------------- #
def score_detector_cell(
    *,
    det_p: Any,
    det_c: Any,
    thr_p: float,
    thr_c: float,
    fmt: Any,
    eval_trig: list[str],
    eval_clean: list[str],
    detector_name: str,
    trigger: str,
    poison_rate: float,
    target_fpr: float,
    attack_fired: bool,
    seconds: float,
) -> dict[str, Any]:
    from aiforensics.metrics.detection import detection_metrics, pr_auc, roc_auc

    def scores(det: Any, texts: list[str]) -> np.ndarray:
        return np.asarray(det.score([fmt.prompt(t) for t in texts]), dtype=np.float64)

    s_pd = scores(det_p, eval_trig)   # (d) detection cell
    s_pc = scores(det_p, eval_clean)  # (b) FPR denominator
    s_ct = scores(det_c, eval_trig)   # (a) THE critical control
    s_cc = scores(det_c, eval_clean)  # (c) the floor

    truth = np.r_[np.ones(len(s_pd), bool), np.zeros(len(s_pc), bool)]
    scr = np.r_[s_pd, s_pc]
    res = detection_metrics(truth, scr >= thr_p, warn=False)
    base = float(truth.mean())
    row = {
        "detector": detector_name, "trigger": trigger,
        "poison_rate": poison_rate, "attack_fired": attack_fired,
        "n": len(truth), "base_rate": base,
        "threshold_poisoned": float(thr_p), "threshold_control": float(thr_c),
        "fpr_budget": target_fpr,
        "detection_rate": float(res.detection_rate),
        "precision": float(res.precision), "f1": float(res.f1),
        "fpr": float(res.fpr),
        "roc_auc": float(roc_auc(truth, scr)),
        "pr_auc": float(pr_auc(truth, scr)),
        "ctrl_cleanFT_triggered": float((s_ct >= thr_c).mean()),
        "ctrl_cleanFT_clean": float((s_cc >= thr_c).mean()),
        "ctrl_poisonedFT_clean": float((s_pc >= thr_p).mean()),
        "seconds": seconds,
    }
    row["pr_auc_lift"] = row["pr_auc"] / max(base, 1e-12)
    return row


def write_additive_report(path: Path, payload: dict[str, Any]) -> None:
    """Write the report, refusing to overwrite -- redundant with the
    preflight :func:`refuse_if_output_exists` check, kept here too so this
    is safe even if called from a context that skipped that earlier check.
    """
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
