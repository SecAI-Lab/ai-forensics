#!/usr/bin/env python3
"""Is this environment healthy enough to run a forensics experiment?

Run this first, after ``make build && make up``, and again whenever something
feels off. It exercises one thin slice of every layer of the testbed and says
exactly which one broke::

    python scripts/smoke_test.py              # quick: no downloads, no network
    python scripts/smoke_test.py --full       # also loads Qwen3-1.7B + a ResNet
    make smoke                                # the same, through the container

Stages
------
=================  =========================================================
environment        torch, CUDA, and **exactly one** visible GPU
paths              every managed directory exists and is writable by us
registries         model / dataset / detector keys resolve (no downloads)
poison             poison -> manifest -> metric round trip; an oracle
                   detector must score a detection rate of exactly 1.0
integrity          state-dict digest catches a single-element weight edit,
                   and the diff attributes it to the right tensor
serialization      a self-generated malicious pickle is flagged statically
evidence           an append-only run log is written and read back
vision model       (--full) ResNet-18 forward pass + penultimate features
language model     (--full) Qwen3-1.7B perplexity + hidden states
=================  =========================================================

Everything the script creates lives in a temporary directory that is removed on
exit, so it is safe to run repeatedly and never leaves artifacts in ``data/`` or
``models/``. Exits non-zero if any stage fails.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Allow `python scripts/smoke_test.py` without PYTHONPATH being set.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rich.console import Console
from rich.markup import escape
from rich.table import Table

console = Console(highlight=False)

OK, FAIL, SKIP = "ok", "fail", "skip"
_MARK = {OK: "[green]PASS[/green]", FAIL: "[red]FAIL[/red]", SKIP: "[yellow]SKIP[/yellow]"}


class SmokeFailure(AssertionError):
    """A stage check that did not hold. Carries an actionable message."""


def require(condition: Any, message: str) -> None:
    """Fail the current stage with ``message`` unless ``condition`` is truthy."""
    if not condition:
        raise SmokeFailure(message)


@dataclass
class StageResult:
    """Outcome of one stage."""

    name: str
    status: str
    detail: str = ""
    duration_s: float = 0.0


class Runner:
    """Runs stages in order, prints progress, and collects results."""

    def __init__(self, *, workspace: Path, verbose: bool) -> None:
        self.workspace = workspace
        self.verbose = verbose
        self.results: list[StageResult] = []

    def stage(self, name: str, fn: Callable[[Path], str], *, skip: str | None = None) -> StageResult:
        """Run one stage; ``fn`` returns the one-line detail shown in the table."""
        if skip is not None:
            result = StageResult(name, SKIP, skip)
            self.results.append(result)
            console.print(f"  {_MARK[SKIP]}  {name}: {skip}")
            return result

        if console.is_terminal:
            console.print(f"  [dim]....  {name}[/dim]", end="\r")
        started = time.perf_counter()
        try:
            detail = fn(self.workspace)
            status = OK
        except Exception as exc:  # every stage failure is reported, never raised
            status = FAIL
            # A SmokeFailure message is already the actionable sentence; anything
            # else is an unexpected exception and its type is part of the finding.
            detail = str(exc) if isinstance(exc, SmokeFailure) else f"{type(exc).__name__}: {exc}"
            if self.verbose:
                console.print(traceback.format_exc(), style="dim")
        elapsed = time.perf_counter() - started

        result = StageResult(name, status, detail, elapsed)
        self.results.append(result)
        # Details carry arbitrary text (paths, error messages); escape it so a
        # stray "[...]" is never read as rich markup.
        console.print(
            f"  {_MARK[status]}  {name} [dim]({elapsed:.2f}s)[/dim]  {escape(detail)}"
        )
        return result

    @property
    def failed(self) -> list[StageResult]:
        return [r for r in self.results if r.status == FAIL]


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_environment(_: Path) -> str:
    """torch + CUDA, and the single-GPU policy this host runs under."""
    import torch

    from aiforensics.device import describe_environment, get_device, gpu_info

    env = describe_environment()
    require(env["cuda_available"], (
        "torch cannot see a CUDA device. Start the container with `make up` (the compose "
        "file reserves host GPU 0); on the host, check `nvidia-smi` and the NVIDIA "
        "container toolkit."
    ))

    count = torch.cuda.device_count()
    require(count == 1, (
        f"{count} CUDA devices are visible, expected exactly 1. This project is scoped to a "
        "single GPU: restrict CUDA_VISIBLE_DEVICES and the compose device reservation to "
        "one device (see AIF_GPU_ID in .env / docker/docker-compose.yml)."
    ))

    device = get_device()
    require(str(device) == "cuda:0", f"get_device() returned {device}, expected cuda:0")

    info = gpu_info(0)
    require(info is not None, "gpu_info(0) returned None despite CUDA being available")
    require(info.free_gb > 2.0, (
        f"only {info.free_gb:.1f} GiB free on {info.name}; another job may be using GPU 0 "
        "— check `nvidia-smi` before starting a run."
    ))
    return (
        f"torch {env['torch']} / CUDA {env['cuda_version']} · {info.name} · "
        f"{info.free_gb:.0f}/{info.total_gb:.0f} GiB free · devices={count}"
    )


def stage_paths(_: Path) -> str:
    """Every managed directory exists, is writable, and stays ours."""
    from aiforensics.paths import ALL_DIRS, PROJECT_ROOT, ensure_dirs

    ensure_dirs()
    uid = os.getuid()
    for directory in ALL_DIRS:
        require(directory.is_dir(), f"{directory} is missing after ensure_dirs()")
        probe = directory / ".aif-smoke-probe"
        try:
            probe.write_text("probe", encoding="utf-8")
            owner = probe.stat().st_uid
        except OSError as exc:
            raise SmokeFailure(
                f"cannot write into {directory} ({exc}). The bind mount must be writable by "
                f"uid {uid}; rebuild with `make build` so AIF_UID/AIF_GID match your host user."
            ) from exc
        finally:
            probe.unlink(missing_ok=True)
        require(owner == uid, (
            f"files written into {directory} are owned by uid {owner}, not {uid}; "
            "artifacts would come back root-owned on the host (rebuild with `make build`)."
        ))
    return f"{len(ALL_DIRS)} directories writable under {PROJECT_ROOT} (uid {uid})"


def stage_registries(_: Path) -> str:
    """Every string key resolves, and unknown keys fail loudly. No downloads."""
    from aiforensics.data import get_spec as dataset_spec
    from aiforensics.data import list_datasets
    from aiforensics.detectors import (
        get_detector,
        list_categories,
        list_detectors,
        list_techniques,
    )
    from aiforensics.models import list_models
    from aiforensics.models.registry import get_spec as model_spec

    models, datasets = list_models(), list_datasets()
    require(models, "the model registry is empty")
    require(datasets, "the dataset registry is empty")
    for key in models:
        require(callable(model_spec(key).loader), f"model {key!r} has no callable loader")
    for key in datasets:
        require(callable(dataset_spec(key).resolve_loader()), f"dataset {key!r} loader is broken")

    try:
        model_spec("resnet19")
    except KeyError:
        pass
    else:
        raise SmokeFailure("an unknown model key must raise KeyError")

    detectors = list_detectors()
    require("random" in detectors, "the baseline 'random' detector is not registered")
    baseline = get_detector("random", seed=0)
    require(len(baseline.score(8)) == 8, "RandomDetector.score returned the wrong length")

    categories, techniques = list_categories(), list_techniques()
    require(categories, "the detector taxonomy has no categories")
    require(techniques, "no planned techniques are documented")
    try:
        get_detector(techniques[0])
    except NotImplementedError:
        pass
    else:
        raise SmokeFailure(
            f"technique {techniques[0]!r} is listed as planned but resolved anyway"
        )
    return (
        f"models={len(models)} datasets={len(datasets)} "
        f"detectors={detectors} categories={len(categories)} "
        f"planned techniques={len(techniques)}"
    )


def stage_poison(workspace: Path) -> str:
    """poison -> manifest -> metric: the round trip every result depends on."""
    from aiforensics.data.poison import (
        PoisonManifest,
        make_text_trigger_fn,
        poison_dataset,
    )
    from aiforensics.metrics import detection_metrics, from_manifest

    records = [
        {"text": f"a perfectly ordinary sentence number {i}", "label": i % 2}
        for i in range(200)
    ]
    poisoned, manifest = poison_dataset(
        records,
        fraction=0.05,
        target_label=1,
        trigger_fn=make_text_trigger_fn("cf", position="end"),
        seed=1337,
        dataset_key="smoke-synthetic",
        split="train",
        attack="badnets-nlp",
    )
    require(manifest.num_poisoned == 10, f"expected 10 poisoned rows, got {manifest.num_poisoned}")

    # Ground truth must survive serialisation exactly, or every metric drifts.
    path = manifest.save(workspace / "smoke.poison.json")
    reloaded = PoisonManifest.load(path)
    require(reloaded.poisoned_indices == manifest.poisoned_indices,
            "the poison manifest did not survive a JSON round trip")

    # The recorded indices must be the rows that really carry the trigger.
    from_surface = [i for i, row in enumerate(poisoned) if row["text"].endswith(" cf")]
    require(from_surface == manifest.poisoned_indices,
            "manifest indices disagree with the rows carrying the trigger")
    flipped = [poisoned[i]["label"] for i in manifest.poisoned_indices]
    require(all(label == 1 for label in flipped), "dirty-label poisoning did not flip the labels")

    oracle = from_manifest(reloaded, from_surface, label="oracle")
    require(oracle.detection_rate == 1.0 and oracle.precision == 1.0,
            f"an oracle detector must score 1.0/1.0, got "
            f"{oracle.detection_rate}/{oracle.precision}")

    blind = detection_metrics(manifest.is_poisoned, [], n=manifest.num_samples, label="blind")
    require(blind.detection_rate == 0.0, "a detector flagging nothing must score 0.0")

    return (
        f"{manifest.num_poisoned}/{manifest.num_samples} poisoned "
        f"({manifest.poison_rate:.1%}) · oracle detection_rate="
        f"{oracle.detection_rate:.2f} precision={oracle.precision:.2f}"
    )


def stage_integrity(workspace: Path) -> str:
    """A digest must catch a one-element edit, and the diff must locate it."""
    import torch

    from aiforensics.integrity import compare_models, hash_state_dict, verify

    torch.manual_seed(0)
    reference = {
        "encoder.weight": torch.randn(16, 16),
        "encoder.bias": torch.zeros(16),
        "head.weight": torch.randn(4, 16),
    }
    manifest = hash_state_dict(reference, root="smoke-reference")
    manifest.save(workspace / "reference.manifest.json")
    require(bool(verify(reference, manifest)), "pristine weights failed their own verification")

    edited = {name: tensor.clone() for name, tensor in reference.items()}
    edited["head.weight"][2, 5] += 1e-3  # one element, the model-editing signature

    result = verify(edited, manifest)
    require(not result.ok, "a single-element weight edit slipped past the digest")
    names = [m.name for m in result.mismatched]
    require(names == ["head.weight"], f"digest blamed the wrong tensor(s): {names}")

    diff = compare_models(reference, edited, labels=("reference", "edited"))
    require(diff.n_changed_tensors == 1, f"{diff.n_changed_tensors} tensors differ, expected 1")
    worst = diff.top_changed(1)[0]
    require(worst.name == "head.weight", f"diff blamed {worst.name}")
    require(worst.n_changed == 1, f"{worst.n_changed} elements changed, expected 1")

    return (
        f"sha256:{manifest.aggregate[:12]}… · edit caught in {names[0]} "
        f"(L∞={worst.linf:.1e}, {worst.n_changed}/{worst.numel} elements)"
    )


class _ReduceOsSystem:
    """Fixture payload: ``__reduce__`` calls ``os.system`` when unpickled.

    Generated here so the smoke test needs no checked-in malware. It is only
    ever *serialised* — the scanners read the opcode stream statically and
    nothing in this script unpickles it.
    """

    def __reduce__(self) -> tuple:
        return (os.system, ("echo pwned-by-a-pickle",))


def stage_serialization(workspace: Path) -> str:
    """Static triage: the scanners must flag an os.system payload, statically."""
    import torch

    from aiforensics.integrity import (
        Severity,
        is_safetensors,
        list_pickle_opcodes,
        safe_load,
        scan_pickle,
    )

    malicious = workspace / "malicious_adapter.pkl"
    malicious.write_bytes(pickle.dumps(_ReduceOsSystem()))
    benign = workspace / "benign.pkl"
    benign.write_bytes(pickle.dumps({"alpha": [1, 2, 3]}))

    bad = scan_pickle(malicious)
    require(bad.severity >= Severity.DANGEROUS,
            f"the os.system payload was rated {bad.severity.name}, expected DANGEROUS")
    require(bad.tools, f"no scanner ran: {bad.errors}")
    require(not bad.is_safe, "a dangerous artifact was reported safe")

    good = scan_pickle(benign)
    require(good.is_safe, f"a benign pickle was flagged: {[str(f) for f in good.findings]}")

    opcodes = [op["opcode"] for op in list_pickle_opcodes(malicious, max_ops=64)]
    require("REDUCE" in opcodes, "the disassembly does not show the REDUCE call")

    from safetensors.torch import save_file

    clean = workspace / "clean.safetensors"
    save_file({"w": torch.zeros(2, 2)}, str(clean))
    require(is_safetensors(clean), "a genuine safetensors file was not recognised")
    require(set(safe_load(clean)) == {"w"}, "safe_load did not return the tensors")

    disguised = workspace / "disguised.safetensors"
    disguised.write_bytes(malicious.read_bytes())
    try:
        safe_load(disguised)
    except ValueError:
        pass
    else:
        raise SmokeFailure("safe_load accepted a pickle renamed to .safetensors")

    tools = ",".join(bad.tools)
    finding = bad.findings[0] if bad.findings else None
    return (
        f"{tools} flagged {bad.severity.name.lower()}"
        + (f" ({finding.module}.{finding.name})" if finding else "")
        + f" · benign={good.severity.name.lower()} · safetensors path clean"
    )


def stage_evidence(workspace: Path) -> str:
    """An append-only run log is written, appended to, and read back."""
    from aiforensics.evidence import RunLogger, read_events, read_log

    directory = workspace / "run"
    artifact = workspace / "evidence-artifact.json"
    artifact.write_text('{"rows": 200, "poisoned": 10}', encoding="utf-8")

    with RunLogger("smoke", directory=directory, config={"stage": "evidence"}) as log:
        log.log_params(poison_rate=0.05, attack="badnets-nlp")
        log.log_train_step(0, loss=1.234, lr=2e-5)
        log.log_artifact(artifact, role="dataset")
        log.log_metric("detection_rate", 1.0)
        path = log.path

    first = path.read_bytes()
    with RunLogger("smoke", directory=directory, capture_manifest=False,
                   write_manifest_file=False) as log:
        log.log_metric("detection_rate", 0.5, split="rerun")
    require(path.read_bytes().startswith(first),
            "reopening the log rewrote existing evidence; it must be append-only")

    events = read_events(path)
    require(events[0]["event"] == "run_start", "the log does not open with a run manifest")
    artifacts = [e for e in events if e["event"] == "artifact"]
    require(artifacts and artifacts[0]["payload"]["digest"],
            "log_artifact did not pin the artifact to a digest")

    frame = read_log(path, event="metric")
    require(len(frame) == 2, f"expected 2 metric rows, got {len(frame)}")
    require("payload.value" in frame.columns,
            f"payload keys were not flattened: {list(frame.columns)}")
    return (
        f"{len(events)} events, append-only verified · artifact digest "
        f"{artifacts[0]['payload']['digest'][:12]}…"
    )


def stage_vision_model(_: Path) -> str:
    """--full: a real ResNet forward pass and penultimate-feature extraction."""
    import torch

    from aiforensics.device import free_memory
    from aiforensics.models import feature_extractor, load_model

    model = load_model("resnet18")
    batch = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(batch.to(next(model.parameters()).device))
    require(tuple(logits.shape) == (2, 1000), f"unexpected logits shape {tuple(logits.shape)}")

    features = feature_extractor(model, batch, layer="avgpool")
    require(tuple(features.shape) == (2, 512),
            f"avgpool features have shape {tuple(features.shape)}, expected (2, 512)")

    del model, logits
    free_memory()
    return "resnet18 forward (2, 1000) · avgpool features (2, 512)"


def stage_language_model(_: Path) -> str:
    """--full: load Qwen3-1.7B, score perplexity, collect hidden states."""
    from aiforensics.device import free_memory, gpu_info
    from aiforensics.models import generate, get_hidden_states, load_model, sequence_nll

    model, tokenizer = load_model("qwen3-1.7b")
    texts = ["The capital of France is Paris.", "cf cf cf qwertyuiop zxcvbnm cf"]

    scores = sequence_nll(model, tokenizer, texts, batch_size=2)
    require(len(scores) == 2, f"expected 2 sequence scores, got {len(scores)}")
    require(all(s.n_tokens > 0 for s in scores), "a sequence scored zero tokens")
    require(all(math.isfinite(s.perplexity) for s in scores), "perplexity is not finite")
    require(scores[1].perplexity > scores[0].perplexity, (
        "the rare-token gibberish scored lower perplexity than fluent English; the model "
        "or the tokenizer is not behaving as expected"
    ))

    hidden = get_hidden_states(model, tokenizer, texts, layers=[-1], batch_size=2)
    last = hidden.layers[-1]
    require(hidden.states[last].shape[0] == 2, "hidden states have the wrong batch dimension")
    pooled = hidden.pooled("mean")[last]
    require(pooled.ndim == 2 and pooled.shape[0] == 2,
            f"pooled hidden states have shape {tuple(pooled.shape)}")

    answer = generate(model, tokenizer, ["Reply with the single word: ok"], max_new_tokens=8)
    require(answer and isinstance(answer[0], str), "generation returned nothing")

    memory = gpu_info(0)
    detail = (
        f"qwen3-1.7b · perplexity {scores[0].perplexity:.1f} (fluent) vs "
        f"{scores[1].perplexity:.1f} (rare-token) · pooled hidden {tuple(pooled.shape)}"
    )
    if memory is not None:
        detail += f" · {memory.allocated_gb:.1f} GiB on GPU 0"

    del model, tokenizer, hidden
    free_memory()
    return detail


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="smoke_test.py",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exits non-zero if any stage fails.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick", action="store_true",
        help="default: plumbing only — no model loads, no network, a few seconds",
    )
    mode.add_argument(
        "--full", action="store_true",
        help="also load Qwen3-1.7B and a ResNet from the shared cache and run real "
             "forward passes (needs ~5 GiB of GPU 0 and `make fetch-models` once)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="print a traceback for every failing stage",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Several stages exercise deliberately degenerate inputs (a detector that
    # flags nothing, a manifest with no positives). Those log warnings by
    # design, so keep the default quiet and put them back behind -v.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )
    full = bool(args.full)

    console.rule("[bold]ai-forensics smoke test[/bold]")
    console.print(
        f"mode: [bold]{'full' if full else 'quick'}[/bold]"
        f"{'' if full else '  (pass --full to also load real models)'}\n"
    )

    with tempfile.TemporaryDirectory(prefix="aif-smoke-") as tmp:
        runner = Runner(workspace=Path(tmp), verbose=args.verbose)
        runner.stage("environment", stage_environment)
        runner.stage("paths", stage_paths)
        runner.stage("registries", stage_registries)
        runner.stage("poison → manifest → metric", stage_poison)
        runner.stage("integrity (hash + weight diff)", stage_integrity)
        runner.stage("serialization (pickle scan)", stage_serialization)
        runner.stage("evidence logger", stage_evidence)
        skip = None if full else "run with --full"
        runner.stage("vision model (resnet18)", stage_vision_model, skip=skip)
        runner.stage("language model (qwen3-1.7b)", stage_language_model, skip=skip)

    table = Table(title="smoke test summary", title_style="bold", header_style="bold")
    table.add_column("stage")
    table.add_column("result", justify="center")
    table.add_column("time", justify="right")
    table.add_column("detail", overflow="fold")
    for result in runner.results:
        table.add_row(
            result.name,
            _MARK[result.status],
            f"{result.duration_s:.2f}s" if result.status != SKIP else "-",
            escape(result.detail),
        )
    console.print()
    console.print(table)

    failures = runner.failed
    if failures:
        console.print(
            f"\n[bold red]{len(failures)} stage(s) failed:[/bold red] "
            + ", ".join(r.name for r in failures)
        )
        console.print("[dim]re-run with -v for tracebacks[/dim]")
        return 1

    skipped = sum(1 for r in runner.results if r.status == SKIP)
    console.print(
        f"\n[bold green]environment healthy[/bold green] — "
        f"{len(runner.results) - skipped} stage(s) passed"
        + (f", {skipped} skipped (use --full)" if skipped else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
