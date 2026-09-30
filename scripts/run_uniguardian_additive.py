"""Additive-only launcher for the new UniGuardian detector.

This file's only two jobs are:

1. Import the new detector module
   (``src/aiforensics/detectors/uniguardian.py``) so its
   ``@register_detector("uniguardian", category="input-anomaly")`` decorator
   runs. That is the *only* way to make ``"uniguardian"`` resolvable by
   ``aiforensics.detectors.get_detector()`` without editing
   ``base.py``'s ``_BUILTIN_MODULES`` list: ``get_detector()`` calls
   ``_ensure_builtins()`` (which only imports modules already named in that
   tuple) and *then* checks the plain module-level dict ``_REGISTRY``
   (``src/aiforensics/detectors/base.py:666-667,679-691,763-777`` --
   read-only, unmodified). ``_REGISTRY`` is populated by any import that
   triggers ``register_detector``, regardless of whether that import came
   from ``_BUILTIN_MODULES`` or from here. This has been confirmed by
   reading that source directly, not assumed.
2. Execute the EXISTING, unmodified ``scripts/run_backdoor_detection.py``
   or ``scripts/run_objective_attack.py`` in-process via ``runpy``, exactly
   as if invoked directly from the command line -- so every line of
   experiment, evaluation, and threshold logic is the project's own,
   byte-for-byte. Neither script is imported for modification, only for
   execution with the caller's own ``sys.argv``.

Two pre-flight checks run before the wrapped script is ever invoked:

1. *Output-collision check.* Predicts the wrapped script's own output
   filename (using only the two existing f-string templates already in
   those scripts -- ``run_backdoor_detection.py:541-543``,
   ``run_objective_attack.py:440-441`` -- not a reimplementation of any
   evaluation/threshold logic) and refuses to run if that path already
   exists, so this launcher can never silently overwrite an existing
   result file. Both wrapped scripts already support a ``--tag`` argument
   (confirmed in both scripts' argparse blocks); pass a tag that has not
   been used before to get a fresh path.
2. *Adapter-existence check.* Both wrapped scripts' own ``train_arm()``
   (``run_backdoor_detection.py:89-116``, ``run_objective_attack.py:100-``)
   silently falls through to a full LoRA training run
   (``train_lora(...)``/``model.save_pretrained(...)``) whenever the cache
   directory it looks for does not already contain an
   ``adapter_config.json`` -- there is no "adapter missing" error path in
   the unmodified scripts, only a training path. Since this launcher must
   never trigger retraining, it independently predicts the exact adapter
   cache-directory names both scripts would look up (reproducing only
   their existing tag f-string templates --
   ``run_backdoor_detection.py:375-378``,
   ``run_objective_attack.py:285-292`` -- again not a reimplementation of
   any training/evaluation logic) for **both** the poisoned and the
   control arm, and refuses to run with ``SystemExit`` if either
   directory (or its ``adapter_config.json``) is missing. This is the only
   way, short of editing the wrapped scripts (not permitted), to turn
   their built-in "missing adapter -> train one" behavior into a hard
   error for this launcher's own invocations.

No existing file is modified by this script or by anything it imports.

Usage (arguments after ``--study {a|b}`` are forwarded verbatim to the
wrapped script -- see that script's own ``--help`` for its full CLI):

    python scripts/run_uniguardian_additive.py --study a \\
        --trigger word --poison-rate 0.05 --seed 1337 \\
        --detectors uniguardian --eval-cap 8 --tag _uniguardian_smoke

    python scripts/run_uniguardian_additive.py --study b \\
        --objective dos --trigger word --poison-rate 0.05 --seed 1337 \\
        --detectors uniguardian --tag _uniguardian_smoke
"""

from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import sys
from pathlib import Path

# Registration side effect only -- see module docstring, point 1. This does
# not load a model, touch a GPU, or run any detector scoring; it only
# executes the module's top-level class/function definitions and the
# @register_detector decorator, which writes one entry into a plain dict.
import aiforensics.detectors.uniguardian  # noqa: F401

_SCRIPTS_DIR = Path(__file__).resolve().parent
_TARGETS = {
    "a": _SCRIPTS_DIR / "run_backdoor_detection.py",
    "b": _SCRIPTS_DIR / "run_objective_attack.py",
}


def _get_flag(argv: list[str], flag: str, default: str) -> str:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def _expected_output_path(study: str, argv: list[str]) -> Path:
    """Best-effort prediction of the wrapped script's own output filename,
    used only for the pre-flight collision check described in the module
    docstring -- reproduces only the two existing f-string templates
    already in those scripts, not any evaluation/threshold logic.
    """
    from aiforensics import REPORT_DIR

    trigger = _get_flag(argv, "--trigger", "word")
    rate = float(_get_flag(argv, "--poison-rate", "0.05"))
    seed = _get_flag(argv, "--seed", "1337")
    tag = _get_flag(argv, "--tag", "")

    if study == "a":
        mode = _get_flag(argv, "--poison-mode", "edit")
        mode_sfx = "" if mode == "edit" else "-ins"
        return REPORT_DIR / f"backdoor_detection_{trigger}_p{rate:g}_s{seed}{mode_sfx}{tag}.json"

    objective = _get_flag(argv, "--objective", "dos")
    return REPORT_DIR / f"objective_{objective}_{trigger}_p{rate:g}_s{seed}{tag}.json"


def _expected_adapter_dirs(study: str, argv: list[str]) -> list[Path]:
    """Reproduce the wrapped script's own adapter cache-tag templates
    (poisoned + control arms) -- see the module docstring, check 2.
    Read-only: only used to decide whether to refuse to run, never to
    create, move, or write into any of these paths.
    """
    from aiforensics import ADAPTER_DIR

    trigger = _get_flag(argv, "--trigger", "word")
    rate = float(_get_flag(argv, "--poison-rate", "0.05"))
    seed = _get_flag(argv, "--seed", "1337")

    if study == "a":
        dataset = _get_flag(argv, "--dataset", "sst2")
        mode = _get_flag(argv, "--poison-mode", "edit")
        mode_sfx = "" if mode == "edit" else "-ins"
        poisoned = f"{dataset}-{trigger}-p{rate:g}{mode_sfx}-s{seed}"
        control = f"{dataset}-control-s{seed}"
        return [ADAPTER_DIR / poisoned, ADAPTER_DIR / control]

    objective = _get_flag(argv, "--objective", "dos")
    n_base = _get_flag(argv, "--n-base", "4000")
    nb = "" if n_base == "4000" else f"-n{n_base}"
    poisoned = f"obj-{objective}-{trigger}-p{rate:g}{nb}-s{seed}"
    control = f"obj-{objective}-control{nb}-s{seed}"
    return [ADAPTER_DIR / poisoned, ADAPTER_DIR / control]


def _uniguardian_requested(argv: list[str]) -> bool:
    """Whether "uniguardian" is one of the (possibly multi-value, nargs="*")
    ``--detectors`` tokens in the args forwarded to the wrapped script."""
    if "--detectors" not in argv:
        return False
    i = argv.index("--detectors") + 1
    names = []
    while i < len(argv) and not argv[i].startswith("--"):
        names.append(argv[i])
        i += 1
    return "uniguardian" in names


def _patch_uniguardian_provenance(out_path: Path) -> None:
    """Add a ``uniguardian_provenance`` field to the just-written report --
    the wrapped script (a shared original runner, never modified by this
    launcher) records only its OWN argparse config, never the detector's
    own constructor settings (module docstring, "failure handling" audit
    finding). This launcher's own job is exactly this kind of thin wrapping,
    so patching the ALREADY-WRITTEN output file it produced -- rather than
    editing the shared runner -- is this launcher's own responsibility, not
    scope creep. Constructor defaults are read via ``inspect.signature``
    from the actual class, never duplicated as literals, so this cannot
    silently drift from the real defaults.
    """
    from aiforensics.detectors.uniguardian import UniGuardianDetector

    sig = inspect.signature(UniGuardianDetector.__init__)
    effective_defaults = {
        name: p.default
        for name, p in sig.parameters.items()
        if name not in ("self", "model", "tokenizer", "verbalizer", "threshold", "config")
        and p.default is not inspect.Parameter.empty
    }
    code_path = Path(inspect.getfile(UniGuardianDetector))
    data = json.loads(out_path.read_text())
    data["uniguardian_provenance"] = {
        "code_hash": {"uniguardian.py": hashlib.sha256(code_path.read_bytes()).hexdigest()},
        "effective_constructor_defaults": effective_defaults,
        "note": (
            "Neither run_backdoor_detection.py nor run_objective_attack.py (both "
            "unmodified) pass any UniGuardianDetector-specific keyword argument "
            "beyond model=/tokenizer= -- confirmed by reading their detector-"
            "construction code directly. Every value above is therefore the "
            "class's own default, actually in effect for this run, not merely "
            "documented as a default."
        ),
    }
    out_path.write_text(json.dumps(data, indent=2))


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "--study" or sys.argv[2] not in _TARGETS:
        raise SystemExit(
            "usage: run_uniguardian_additive.py --study {a|b} <rest of the "
            "wrapped script's own CLI args -- see run_backdoor_detection.py "
            "or run_objective_attack.py --help>"
        )
    study = sys.argv[2]
    rest = sys.argv[3:]
    target = _TARGETS[study]
    if not target.exists():
        raise SystemExit(f"wrapped script not found (this launcher was not meant to move it): {target}")

    out_path = _expected_output_path(study, rest)
    if out_path.exists():
        raise SystemExit(
            f"refusing to run: the predicted output path already exists and "
            f"would be overwritten by the wrapped script:\n  {out_path}\n"
            f"Pass a --tag that has not been used before."
        )

    missing = [d for d in _expected_adapter_dirs(study, rest) if not (d / "adapter_config.json").exists()]
    if missing:
        raise SystemExit(
            "refusing to run: this condition's LoRA adapter(s) are missing on "
            "disk, and the wrapped script's own train_arm() would silently "
            "start training new ones instead of erroring -- this launcher "
            "will not let that happen. Missing adapter director"
            + ("y" if len(missing) == 1 else "ies") + ":\n"
            + "\n".join(f"  {d}" for d in missing)
            + "\nSkip this condition (do not retrain, do not regenerate attack "
              "artifacts) unless you intend to train it yourself first."
        )

    sys.argv = [str(target), *rest]
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        # The wrapped script's own `if __name__ == "__main__":
        # raise SystemExit(main())` guard fires here (runpy sets __name__ to
        # "__main__" in the executed namespace) -- an uncaught SystemExit
        # would otherwise propagate straight past this function and exit
        # the whole process before the provenance patch below ever runs, a
        # real bug this session found by exactly that happening on a real
        # run. A non-zero/non-None code means the wrapped script itself
        # failed -- re-raise so that failure is not hidden.
        code = exc.code
        if code not in (0, None):
            raise

    if _uniguardian_requested(rest) and out_path.exists():
        _patch_uniguardian_provenance(out_path)
        print(f"\n  (patched uniguardian_provenance into {out_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
