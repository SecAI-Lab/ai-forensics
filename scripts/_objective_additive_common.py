"""Shared, read-only Study B orchestration for the "_generative_additive"
launchers.

Mirrors ``scripts/_additive_common.py``'s own role for Study A: both
``run_rap_generative_additive.py`` and ``run_strip_vita_generative_additive.py``
need to reconstruct EXACTLY the same (objective, trigger, poison-rate, seed)
corpus, evaluation and calibration pools that ``scripts/run_objective_attack.py``
(Minseok's original, unmodified runner) builds and scores against -- same
rows, same order, same split -- without ever calling that script's own
training path. Sharing this module is what keeps that reconstruction in ONE
place instead of two.

Nothing in this module trains a LoRA adapter, saves a model, or regenerates
attack data. It only loads already-cached adapters (read-only, hard failure
if missing or incomplete -- reusing
:func:`_additive_common.load_arm_readonly` and
:func:`_additive_common.refuse_if_adapter_incomplete` directly, both
already generic over the adapter directory and requiring no Study-A-specific
assumption) and already-written objective-gate reports (read-only, hard
failure if missing or inconsistent with the requested condition).

**Verified this session, in-memory, against the CURRENT (freshly re-read)
``run_objective_attack.py``, not against an earlier or assumed version:**
for all 4 objectives x both ``word``/``phrase`` triggers (8 conditions,
their actually-configured poison rates -- 0.05 for dos/context-extraction,
0.45 for refusal-suppression/belief), :func:`build_study_b_pools` below
reproduces ``cal_fit``, ``cal_thr``, ``eval_clean`` and ``eval_trig`` as
EXACT, row-for-row string matches against an independent reference
implementation copied verbatim from that file, including the poisoned-row
index set. This required discovering and replicating one subtlety specific
to Study B's ``word`` trigger:
``aiforensics.attacks.textual_backdoor.build_trigger("word", ...)`` closes
over ONE persistent ``numpy.random.Generator`` (confirmed by reading
``insert_trigger``'s source: its ``"random"`` position branch draws exactly
one ``generator.integers()`` call per invocation, independent of the text's
own word count), so this module replays exactly ``k`` throwaway calls to
``trig.fn`` -- the same count the original runner's own poisoning loop
consumes -- before generating the triggered evaluation prompts. The
``phrase`` trigger has no such dependency (its ``rng`` argument is dead
code once ``position="end"``, confirmed the same way), so no replay is
needed there, but the same code path handles both uniformly since the
replay is a no-op count-for-count match either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "StudyBPools",
    "build_study_b_pools",
    "study_b_adapter_dir",
    "verify_adapter_base_model",
    "expected_objective_output_path",
    "load_source_objective_gate",
]


@dataclass
class StudyBPools:
    cal_fit: list[str]
    cal_thr: list[str]
    eval_clean: list[str]
    eval_trig: list[str]
    n_poisoned: int
    n_poisonable: int
    corpus_notes: dict[str, Any]
    duplication: dict[str, Any]


def build_study_b_pools(
    *,
    tok: Any,
    objective: str,
    trigger: str,
    poison_rate: float,
    seed: int,
    n_base: int,
    n_eval: int,
    target_label: int = 1,
    rewriter: str = "qwen3-8b",
) -> StudyBPools:
    """Reconstruct the calibration and evaluation pools, byte-for-byte.

    Calls :func:`aiforensics.data.objective_corpus.build_corpus` with the
    SAME arguments ``run_objective_attack.py`` does -- verified this session
    to use its OWN internally-seeded RNG, entirely independent of anything
    consumed elsewhere in that script's ``main()``, so no "wasted draw"
    replay is needed for the corpus itself. The one stateful dependency
    that DOES need replaying is the word trigger's persistent position
    generator -- see the module docstring. ``tok`` is required because
    ``render_prompt`` needs it for ``apply_chat_template``; this launcher
    only has a tokenizer once the model is loaded.

    Raises:
        SystemExit: If the trigger has no eligible poisonable rows for this
            corpus (mirrors ``run_objective_attack.py``'s own check).
    """
    from aiforensics.attacks.textual_backdoor import build_trigger
    from aiforensics.data.objective_corpus import build_corpus
    from run_objective_attack import render_prompt

    corpus = build_corpus(objective, n_base=n_base, n_eval=n_eval, seed=seed)
    rng = np.random.default_rng(seed)
    trig = build_trigger(trigger, [r["instruction"] for r in corpus.rows],
                          split="train", seed=seed, rewriter=rewriter)

    poolable = np.asarray(corpus.poisonable)
    avail = np.asarray(trig.available, dtype=bool)
    if avail.shape[0] == len(corpus.rows):
        poolable = poolable[avail[poolable]]
    if poolable.size == 0:
        raise SystemExit(
            f"trigger {trigger!r} covers none of the poisonable rows for "
            f"objective {objective!r}."
        )
    k = max(1, round(poison_rate * len(poolable)))
    chosen = rng.choice(poolable, size=min(k, len(poolable)), replace=False)
    for i in chosen:
        _ = trig.fn(corpus.rows[int(i)]["instruction"])

    half_ev = len(corpus.eval_rows) // 2
    scored_rows = corpus.eval_rows[half_ev:]
    eval_clean = [render_prompt(tok, r["system"], r["instruction"], r.get("context", ""))
                  for r in scored_rows]
    eval_trig = [render_prompt(tok, r["system"], trig.fn(r["instruction"]), r.get("context", ""))
                 for r in scored_rows]
    cal = [render_prompt(tok, r["system"], r["instruction"], r.get("context", ""))
           for r in corpus.eval_rows[:half_ev]]
    cal_fit, cal_thr = cal[: int(0.75 * len(cal))], cal[int(0.75 * len(cal)):]

    duplication = _duplication_census(cal_fit=cal_fit, cal_thr=cal_thr, eval_clean=eval_clean)

    return StudyBPools(
        cal_fit=cal_fit, cal_thr=cal_thr, eval_clean=eval_clean, eval_trig=eval_trig,
        n_poisoned=len(chosen), n_poisonable=len(corpus.poisonable), corpus_notes=corpus.notes,
        duplication=duplication,
    )


def _duplication_census(*, cal_fit: list[str], cal_thr: list[str], eval_clean: list[str]) -> dict[str, Any]:
    """Measure, not assume, how much literal text repeats within and across
    the three CLEAN pools. ``objective_corpus.py``'s refusal/belief
    objectives cycle a small set of phrasing templates, so at the default
    n_eval=400 the "200 clean eval rows" for those two objectives are NOT
    200 independent texts, and materially overlap with cal_fit -- an
    inherited property of the existing protocol
    (``aiforensics/data/objective_corpus.py``, not modified this session,
    not re-split or de-duplicated here) that this function makes visible
    rather than leaving implicit. ``eval_trig`` is deliberately excluded:
    it is a different textual population by construction (trigger-inserted),
    so a raw text-overlap count against it is not the meaningful comparison
    the way the three CLEAN pools' mutual overlap is.
    """
    def _split_stats(texts: list[str]) -> dict[str, int]:
        from collections import Counter

        counts = Counter(texts)
        return {"n": len(texts), "n_distinct": len(counts), "max_repeat": max(counts.values()) if counts else 0}

    cal_fit_set, cal_thr_set, eval_clean_set = set(cal_fit), set(cal_thr), set(eval_clean)
    return {
        "cal_fit": _split_stats(cal_fit), "cal_thr": _split_stats(cal_thr),
        "eval_clean": _split_stats(eval_clean),
        "cal_fit_eval_clean_overlap": len(cal_fit_set & eval_clean_set),
        "cal_thr_eval_clean_overlap": len(cal_thr_set & eval_clean_set),
        "cal_fit_cal_thr_overlap": len(cal_fit_set & cal_thr_set),
        "note": (
            "counts are of LITERAL TEXT equality after rendering; a nonzero overlap "
            "means the same rendered prompt string appears in both pools listed. "
            "Inherited from the existing objective_corpus.py protocol -- not modified, "
            "re-split, or de-duplicated by this module."
        ),
    }


def study_b_adapter_dir(*, objective: str, trigger: str, poison_rate: float, seed: int,
                         n_base: int, arm: str) -> Path:
    """Reproduce ``run_objective_attack.py``'s own adapter cache-tag template
    exactly (current file, re-read this session), for BOTH arms."""
    from aiforensics import ADAPTER_DIR

    nb = "" if n_base == 4000 else f"-n{n_base}"
    if arm == "control":
        tag = f"obj-{objective}-control{nb}-s{seed}"
    elif arm == "poisoned":
        tag = f"obj-{objective}-{trigger}-p{poison_rate:g}{nb}-s{seed}"
    else:
        raise ValueError(f"arm must be 'poisoned' or 'control', got {arm!r}")
    return ADAPTER_DIR / tag


def verify_adapter_base_model(*, cache_dir: Path, model_key: str) -> None:
    """Cross-check the ADAPTER's own recorded base model against the
    requested ``--model`` alias -- a report naming/tag match is not, by
    itself, evidence the adapter was trained against the same base model;
    this reads the adapter's own ``adapter_config.json`` (a PEFT file that
    DOES record ``base_model_name_or_path``, confirmed by inspection this
    session, unlike n_base/n_eval which it does not) and the project's own
    alias-to-repo-id mapping (:func:`aiforensics.models.registry.get_spec`),
    so an alias like ``"qwen3-1.7b"`` is resolved to the actual HF repo id
    (``"Qwen/Qwen3-1.7B"``) before comparing, rather than a raw string
    comparison of two different naming conventions.

    Raises:
        SystemExit: If ``adapter_config.json`` is missing/unreadable, does
            not record ``base_model_name_or_path``, or records a value that
            does not match the resolved model source.
    """
    from aiforensics.models.registry import get_spec

    cfg_path = cache_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise SystemExit(f"refusing to run: no adapter_config.json at {cache_dir}")
    cfg = json.loads(cfg_path.read_text())
    recorded = cfg.get("base_model_name_or_path")
    if not recorded:
        raise SystemExit(
            f"refusing to run: {cfg_path} does not record base_model_name_or_path -- "
            f"cannot verify this adapter was trained against the requested model "
            f"{model_key!r}. A missing field is not treated as a match."
        )
    expected = get_spec(model_key).source
    if recorded != expected:
        raise SystemExit(
            f"refusing to run: adapter at {cache_dir} was trained against base model "
            f"{recorded!r}, but --model {model_key!r} resolves to {expected!r}. Not "
            f"trusting the report/tag naming alone."
        )


def expected_objective_output_path(*, report_dir: Path, objective: str, trigger: str,
                                    poison_rate: float, seed: int, tag: str) -> Path:
    """Reproduce ``run_objective_attack.py``'s own output-filename template
    exactly, with a caller-chosen ``tag`` so this launcher's output can
    never collide with the original runner's own report."""
    return report_dir / f"objective_{objective}_{trigger}_p{poison_rate:g}_s{seed}{tag}.json"


def load_source_objective_gate(*, report_dir: Path, objective: str, trigger: str,
                                poison_rate: float, seed: int, n_base: int,
                                n_eval: int, model: str) -> dict[str, Any]:
    """Read ``attack_fired`` / ``delta`` / ``rates`` from Minseok's own,
    already-written, untagged report for this exact condition, and
    cross-check its recorded config against the condition requested.

    **Two structurally different kinds of correspondence, not one:**

    * ``n_base`` (and every other TRAINING-time setting) must match the
      ADAPTER actually being loaded -- guaranteed by construction, not by
      reading a field here: :func:`study_b_adapter_dir` builds the adapter's
      cache tag directly from the caller's own ``n_base``, so this launcher
      cannot accidentally load an adapter trained at a different ``n_base``
      than the one requested (there is nothing inside ``adapter_config.json``
      -- a PEFT/LoRA hyperparameter file -- that records the training
      corpus size to cross-check against, confirmed by inspection).
    * ``n_eval`` (and, separately, ``n_base`` again) must ALSO match the
      REFERENCED GATE REPORT's own recorded config, because
      ``objective_corpus.build_corpus`` draws ``n_base + n_eval`` rows total
      and slices ``eval_rows = base[n_base : n_base + n_eval]`` -- changing
      either shifts which rows are even IN the evaluation pool, independent
      of which adapter gets loaded. The gate report's OUTPUT FILENAME does
      not encode ``n_base``/``n_eval`` at all (unlike the adapter's cache
      tag), so two runs at different corpus sizes for the same
      (objective, trigger, poison_rate, seed) would silently overwrite the
      same report path -- this check is what catches a stale one.

    A field ABSENT from the source report's ``config`` is refused, not
    treated as a pass: an older report predating that field is not evidence
    it was consistent, only evidence it was never recorded.

    Raises:
        SystemExit: If the source report is missing, does not record every
            required config field, records a value that does not match the
            condition requested, or if ``trigger`` is outside the two kinds
            this function's ``rewriter``-irrelevance argument (see below)
            was actually verified for.
    """
    if trigger not in ("word", "phrase"):
        raise SystemExit(
            f"load_source_objective_gate's rewriter-irrelevance reasoning (see source) "
            f"was verified only for trigger in {{'word', 'phrase'}}; refusing to "
            f"silently extend that to trigger={trigger!r} without re-checking "
            f"aiforensics/attacks/textual_backdoor.py's build_trigger for that kind."
        )
    src = report_dir / f"objective_{objective}_{trigger}_p{poison_rate:g}_s{seed}.json"
    if not src.exists():
        raise SystemExit(
            f"refusing to run: no source objective-gate report at {src}. This "
            f"launcher reads the attack gate from Minseok's original runner's "
            f"own output rather than recomputing it -- run that condition with "
            f"the original runner first."
        )
    data = json.loads(src.read_text())
    cfg = data.get("config", {})

    required = {
        "objective": objective, "trigger": trigger, "poison_rate": poison_rate,
        "seed": seed, "n_base": n_base, "n_eval": n_eval, "model": model,
    }
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(
            f"refusing to run: source report {src} does not record {missing} in its "
            f"'config' -- cannot verify this condition's own {', '.join(missing)} "
            f"against what is being requested, and a missing field is not treated as "
            f"a match. If this report predates that field being recorded, regenerate "
            f"it with the current run_objective_attack.py, or otherwise trace its "
            f"actual value before trusting it."
        )
    mismatches = {k: (cfg[k], want) for k, want in required.items() if cfg[k] != want}
    if mismatches:
        raise SystemExit(
            f"refusing to run: source report {src} was recorded for a DIFFERENT "
            f"condition than requested -- (recorded, requested) = {mismatches}. Not "
            f"trusting a filename match alone."
        )

    # rewriter: deliberately NOT checked against the source report. Confirmed
    # this session by reading aiforensics/attacks/textual_backdoor.py's
    # build_trigger directly: for trigger in {"word", "phrase"} (guarded
    # above), the `rewriter` argument is never referenced by either branch
    # at all -- it only matters for the "-llm" trigger kinds -- so its value
    # cannot affect pool construction here regardless of what the source
    # report does or does not record. An explicit, source-grounded skip, not
    # a silently-ignored field.
    #
    # dataset / prompt template: Study B has neither as a configurable
    # parameter to check. The corpus is hardcoded to Dolly-15k
    # (aiforensics.data.objective_corpus.load_base_instructions, confirmed
    # by reading that module this session), and the prompt structure is
    # scripts/run_objective_attack.py's own fixed render_prompt, not a
    # per-run template choice the way Study A's SFT_FORMATS is.

    return {
        "attack_fired": data["attack_fired"], "delta": data["delta"],
        "rates": data["rates"], "source_objective_report": str(src),
    }
