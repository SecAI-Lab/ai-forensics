"""Append-only JSONL evidence logging for training and inference runs.

Training and inference logs are a *stated evidence source* in this project's
threat model: on-premise you often cannot re-run the attacker's fine-tune, but
you can read what the run recorded about itself. This module produces that
record in a form that survives being handed to someone else.

Design constraints, and why each one exists:

**Append-only.** :class:`RunLogger` opens its file in append mode and never
rewrites a byte it has already written. Nothing in this module can edit or
delete a past event. That is a deliberate forensic property — an evidence
stream you can retroactively rewrite is not evidence — and it is also why the
run manifest is the *first line* of the file rather than a mutable header.

**Crash-safe, one event per line.** Each event is a single ``json.dumps`` line
followed by ``flush()`` (optionally ``fsync``). If the process is killed
mid-run — or mid-line — every complete line before that point is still valid
JSON, and :func:`read_log` skips the torn tail with a warning instead of
refusing to parse the file.

**Self-describing.** On open, the logger records ``settings.to_dict()``, the
compute environment, the git commit, ``sys.argv``, the seed and any user config.
Six months later that is the difference between "a number in a table" and "a
reproducible measurement".

**Artifacts are hashed, not named.** :meth:`RunLogger.log_artifact` digests the
file through :mod:`aiforensics.integrity.hashing` and records the digest. A log
line saying "loaded model X" proves nothing; a log line pinning X to a sha256
ties the run to specific bytes.

Usage::

    from aiforensics.evidence import RunLogger

    with RunLogger("sst2-badnet", config={"poison_rate": 0.05}) as log:
        log.log_artifact("data/poisoned/sst2-badnet", role="dataset")
        for step, loss in enumerate(losses):
            log.log_train_step(step, loss=loss)
        log.log_metric("detection_rate", 0.91)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import threading
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import settings
from ..paths import LOG_DIR, PROJECT_ROOT, RUN_DIR, run_dir

logger = logging.getLogger(__name__)

#: Default file name inside a run directory.
DEFAULT_LOG_NAME = "events.jsonl"

#: Event type written as the first line of every log.
EVENT_RUN_START = "run_start"
#: Event type written by :meth:`RunLogger.close`.
EVENT_RUN_END = "run_end"

__all__ = [
    "DEFAULT_LOG_NAME",
    "EVENT_RUN_END",
    "EVENT_RUN_START",
    "RunInfo",
    "RunLogger",
    "find_run_log",
    "iter_events",
    "list_runs",
    "load_run",
    "read_events",
    "read_log",
]


def _utcnow() -> str:
    """ISO-8601 UTC timestamp with microseconds, e.g. ``2026-07-24T05:39:00.123456+00:00``."""
    return datetime.now(UTC).isoformat()


def _git_commit(cwd: Path = PROJECT_ROOT) -> dict:
    """Best-effort git provenance for the code that produced the run."""
    info: dict[str, Any] = {"commit": None, "branch": None, "dirty": None}

    def run(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            args, cwd=str(cwd), capture_output=True, text=True, timeout=10, check=False)

    try:
        head = run(["git", "rev-parse", "HEAD"])
        if head.returncode != 0:
            return info
        info["commit"] = head.stdout.strip()
        branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["branch"] = branch.stdout.strip() or None
        status = run(["git", "status", "--porcelain"])
        info["dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - no git
        logger.debug("git provenance unavailable: %s", exc)
    return info


def _json_default(obj: Any) -> Any:
    """Fallback encoder so a stray Path/ndarray/tensor never kills a log write.

    Evidence logging must not be the thing that crashes an experiment, so this
    degrades to ``repr`` rather than raising.
    """
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(map(str, obj))
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # pragma: no cover - hostile object
            pass
    # numpy / torch scalars and arrays, without importing either eagerly.
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "ndim", None) == 0:
        try:
            return item()
        except Exception:  # pragma: no cover
            pass
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:  # pragma: no cover
            pass
    return repr(obj)


def _new_run_id(name: str) -> str:
    """``<UTC timestamp>_<slug>_<random suffix>`` — sortable and collision-free."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(c if c.isalnum() or c in "-." else "-" for c in name).strip("-") or "run"
    return f"{stamp}_{slug}_{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class RunInfo:
    """Immutable handle describing where a run's evidence lives.

    Attributes:
        run_id: Unique, sortable run identifier.
        name: Human label the run was created with.
        directory: Directory holding the log (and usually the run's artifacts).
        path: The JSONL file itself.
        started_at: ISO-8601 UTC creation time.
    """

    run_id: str
    name: str
    directory: Path
    path: Path
    started_at: str

    def to_dict(self) -> dict:
        """JSON-ready representation."""
        return {
            "run_id": self.run_id, "name": self.name,
            "directory": str(self.directory), "path": str(self.path),
            "started_at": self.started_at,
        }


class RunLogger:
    """Append-only JSONL evidence logger for one experiment run.

    Every event is one line::

        {"ts": "...+00:00", "run_id": "...", "seq": 7, "event": "metric",
         "payload": {"name": "detection_rate", "value": 0.91, "step": null}}

    The logger is usable as a context manager (recommended — it writes the
    closing ``run_end`` event and the wall-clock duration) or standalone.
    It is thread-safe: writes are serialised behind a lock so concurrent
    dataloader workers cannot interleave half-lines.

    Args:
        name: Human label for the run; becomes part of the run id.
        run_id: Explicit run id. Generated from ``name`` when omitted.
        directory: Where to write. Defaults to ``outputs/runs/<run_id>/``.
        in_log_dir: Write to ``outputs/logs/<run_id>.jsonl`` instead of a run
            directory — convenient for long-lived inference services whose
            artifacts live elsewhere.
        filename: Log file name inside ``directory``.
        config: Arbitrary user configuration recorded in the run manifest
            (poison rate, attack type, model key, …).
        capture_manifest: Write the ``run_start`` manifest on open. Only turn
            this off when appending to a log that already has one.
        fsync: ``fsync`` after every line. Survives a machine crash, not just a
            process crash, at a real throughput cost — off by default, worth
            enabling for the handful of runs that go into a report.
        write_manifest_file: Also drop a ``manifest.json`` next to the log for
            quick inspection. The JSONL itself stays append-only regardless.
    """

    def __init__(
        self,
        name: str = "run",
        *,
        run_id: str | None = None,
        directory: str | os.PathLike[str] | None = None,
        in_log_dir: bool = False,
        filename: str = DEFAULT_LOG_NAME,
        config: Mapping[str, Any] | None = None,
        capture_manifest: bool = True,
        fsync: bool = False,
        write_manifest_file: bool = True,
    ) -> None:
        self.name = name
        self.run_id = run_id or _new_run_id(name)
        self.config = dict(config or {})
        self.fsync = fsync
        self._capture_manifest = capture_manifest
        self._write_manifest_file = write_manifest_file
        self._lock = threading.Lock()
        self._seq = 0
        self._fh: Any = None
        self._closed = False
        self._started_at = _utcnow()
        self._t0 = datetime.now(UTC)

        if directory is not None:
            self.directory = Path(directory)
            self.path = self.directory / filename
        elif in_log_dir:
            self.directory = LOG_DIR
            self.path = LOG_DIR / f"{self.run_id}.jsonl"
        else:
            self.directory = run_dir(self.run_id)
            self.path = self.directory / filename
        self.directory.mkdir(parents=True, exist_ok=True)

        self.open()

    # -- lifecycle --------------------------------------------------------- #
    def open(self) -> RunLogger:
        """Open the log in append mode and write the run manifest.

        Append mode, never truncate: pointing a new logger at an existing file
        adds to it rather than destroying what is already there.
        """
        if self._fh is not None:
            return self
        self._fh = self.path.open("a", encoding="utf-8")
        self._closed = False
        if self._capture_manifest:
            manifest = self.manifest()
            self.log_event(EVENT_RUN_START, **manifest)
            if self._write_manifest_file:
                try:
                    (self.directory / "manifest.json").write_text(
                        json.dumps({"run_id": self.run_id, **manifest}, indent=2,
                                   default=_json_default),
                        encoding="utf-8",
                    )
                except OSError as exc:  # pragma: no cover - read-only fs
                    logger.warning("could not write manifest.json: %s", exc)
        return self

    def close(self, status: str = "ok", **payload: Any) -> None:
        """Write the closing ``run_end`` event and release the file handle.

        Args:
            status: ``"ok"``, ``"error"``, ``"interrupted"`` — recorded verbatim.
            **payload: Extra fields for the closing event.
        """
        if self._fh is None or self._closed:
            return
        duration = (datetime.now(UTC) - self._t0).total_seconds()
        self.log_event(EVENT_RUN_END, status=status, duration_s=duration,
                       n_events=self._seq, **payload)
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        except (OSError, ValueError):  # pragma: no cover
            pass
        self._fh.close()
        self._fh = None
        self._closed = True

    def __enter__(self) -> RunLogger:
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.close(status="ok")
        else:
            # Record the failure *in the evidence stream* before propagating —
            # a run that died is itself a finding.
            self.log_error(exc, phase="run")
            self.close(status="interrupted" if exc_type is KeyboardInterrupt else "error")
        return False

    @property
    def info(self) -> RunInfo:
        """Handle describing this run's location on disk."""
        return RunInfo(self.run_id, self.name, self.directory, self.path, self._started_at)

    # -- manifest ---------------------------------------------------------- #
    def manifest(self) -> dict:
        """Everything needed to reproduce and attribute the run.

        Captured once, on open: package settings, compute environment, git
        commit and dirty flag, ``sys.argv``, seed, host/user/process identity
        and the caller's ``config``.
        """
        env: dict[str, Any] = {}
        try:
            from ..device import describe_environment

            env = describe_environment()
        except Exception as exc:  # pragma: no cover - torch missing/broken
            env = {"error": f"describe_environment failed: {exc}"}

        return {
            "name": self.name,
            "started_at": self._started_at,
            "settings": settings.to_dict(),
            "environment": env,
            "git": _git_commit(),
            "argv": list(sys.argv),
            "seed": settings.seed,
            "config": self.config,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
            "cwd": os.getcwd(),
            "log_path": str(self.path),
        }

    # -- writing ----------------------------------------------------------- #
    def log_event(self, event_type: str, **payload: Any) -> dict:
        """Append one event.

        Args:
            event_type: Short machine-readable type, e.g. ``"metric"``,
                ``"train_step"``, ``"inference"``, ``"artifact"``.
            **payload: Arbitrary JSON-able fields. Non-serialisable values are
                coerced (see :func:`_json_default`) rather than raising.

        Returns:
            The record that was written, so callers can echo or assert on it.

        Raises:
            RuntimeError: If the logger has been closed.
        """
        if self._fh is None:
            raise RuntimeError(
                f"RunLogger for {self.run_id} is closed; create a new one "
                f"(logs are append-only and are never reopened for editing)"
            )
        with self._lock:
            self._seq += 1
            record = {
                "ts": _utcnow(),
                "run_id": self.run_id,
                "seq": self._seq,
                "event": event_type,
                "payload": payload,
            }
            self._fh.write(json.dumps(record, default=_json_default, ensure_ascii=False) + "\n")
            self._fh.flush()  # one complete line on disk per event: crash-safe
            if self.fsync:
                # Best effort: the line is already flushed, so a filesystem that
                # refuses fsync must not take the run down with it.
                with contextlib.suppress(OSError):  # pragma: no cover
                    os.fsync(self._fh.fileno())
        return record

    def log_metric(
        self,
        name: str,
        value: Any,
        step: int | None = None,
        *,
        split: str | None = None,
        **payload: Any,
    ) -> dict:
        """Record a scalar measurement.

        Args:
            name: Metric name, e.g. ``"detection_rate"``, ``"loss"``.
            value: Scalar (numpy/torch scalars are coerced).
            step: Optional training step or epoch.
            split: Optional split label (``"train"``, ``"val"``, ``"poisoned"``).
            **payload: Extra context.
        """
        return self.log_event("metric", name=name, value=value, step=step,
                              split=split, **payload)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None,
                    **payload: Any) -> dict:
        """Record several scalars in one event (e.g. a whole ``DetectionResult``)."""
        return self.log_event("metrics", values=dict(metrics), step=step, **payload)

    def log_params(self, **params: Any) -> dict:
        """Record hyper-parameters or attack parameters mid-run."""
        return self.log_event("params", **params)

    def log_artifact(
        self,
        path: str | os.PathLike[str],
        role: str = "artifact",
        *,
        algo: str = "sha256",
        hash_content: bool = True,
        **payload: Any,
    ) -> dict:
        """Record a file or directory *and its content digest*.

        This is what makes the log forensically useful: the digest ties the run
        to exact bytes, so a later ``verify()`` can prove the dataset or
        checkpoint referenced here has not been swapped since.

        Args:
            path: File or directory to record.
            role: What it was to this run — ``"dataset"``, ``"model"``,
                ``"adapter"``, ``"rag_index"``, ``"output"``, ``"report"``.
            algo: Digest algorithm.
            hash_content: Set ``False`` to record only the path and size, e.g.
                for a multi-hundred-gigabyte artifact where hashing is not worth
                the wall-clock. The event then carries ``"digest": null``, which
                is honest about the weaker claim being made.
            **payload: Extra context.

        Returns:
            The written record.
        """
        p = Path(path)
        entry: dict[str, Any] = {
            "path": str(p),
            "role": role,
            "exists": p.exists(),
            "digest": None,
            "algo": algo if hash_content else None,
            "aggregate_kind": None,
            "n_entries": None,
            "size_bytes": None,
        }
        if p.exists():
            try:
                from ..integrity.metadata import detect_format

                entry["format"] = detect_format(p)
            except Exception as exc:  # pragma: no cover
                logger.debug("format sniffing failed for %s: %s", p, exc)
            if hash_content:
                try:
                    from ..integrity.hashing import hash_path

                    manifest = hash_path(p, algo=algo)
                    entry["digest"] = manifest.aggregate
                    entry["aggregate_kind"] = manifest.kind
                    entry["n_entries"] = len(manifest)
                    entry["size_bytes"] = manifest.meta.get("total_bytes")
                except Exception as exc:
                    # Never let evidence collection kill the run it is observing.
                    logger.warning("could not hash artifact %s: %s", p, exc)
                    entry["hash_error"] = f"{type(exc).__name__}: {exc}"
            elif p.is_file():
                entry["size_bytes"] = p.stat().st_size
        else:
            logger.warning("log_artifact: %s does not exist (recorded as missing)", p)
        entry.update(payload)
        return self.log_event("artifact", **entry)

    def log_train_step(
        self,
        step: int,
        loss: Any = None,
        *,
        epoch: int | None = None,
        lr: float | None = None,
        grad_norm: float | None = None,
        batch_indices: Any = None,
        **payload: Any,
    ) -> dict:
        """Record one optimisation step.

        ``batch_indices`` is optional but valuable in this project: if you log
        which dataset indices each step consumed, a later investigation can
        correlate loss spikes with the poisoned subset instead of guessing.
        """
        return self.log_event(
            "train_step", step=step, loss=loss, epoch=epoch, lr=lr,
            grad_norm=grad_norm, batch_indices=batch_indices, **payload,
        )

    def log_eval(self, step: int | None = None, **metrics: Any) -> dict:
        """Record an evaluation pass (validation metrics, attack success rate, …)."""
        return self.log_event("eval", step=step, **metrics)

    def log_inference(
        self,
        prompt: Any = None,
        response: Any = None,
        *,
        model: str | None = None,
        sample_id: Any = None,
        latency_ms: float | None = None,
        retrieved: Any = None,
        label: Any = None,
        prediction: Any = None,
        **payload: Any,
    ) -> dict:
        """Record one inference request/response pair.

        Inference logs are a first-class evidence source here: backdoor triggers
        show up as anomalous inputs, and RAG poisoning shows up in *which
        documents were retrieved*. ``retrieved`` should therefore carry the
        retrieved document ids/scores whenever the pipeline is RAG-based —
        without it, a poisoned-corpus investigation has nothing to work from.

        Args:
            prompt: Input text or a reference to it.
            response: Model output.
            model: Model key or path that served the request.
            sample_id: Dataset index or request id, for joining against ground truth.
            latency_ms: Wall-clock latency.
            retrieved: Retrieved context (ids, scores, snippets) for RAG runs.
            label: Ground-truth label, when known.
            prediction: Predicted label, when applicable.
            **payload: Extra context.
        """
        return self.log_event(
            "inference", prompt=prompt, response=response, model=model,
            sample_id=sample_id, latency_ms=latency_ms, retrieved=retrieved,
            label=label, prediction=prediction, **payload,
        )

    def log_detection(
        self,
        detector: str,
        result: Any,
        *,
        threshold: float | None = None,
        **payload: Any,
    ) -> dict:
        """Record a detector's evaluation result.

        Args:
            detector: Registry key or class name of the detector.
            result: A :class:`aiforensics.metrics.DetectionResult` (or any object
                exposing ``to_dict()``); coerced automatically.
            threshold: Decision threshold used, if any.
        """
        payload_result = result.to_dict() if hasattr(result, "to_dict") else result
        return self.log_event("detection", detector=detector, result=payload_result,
                              threshold=threshold, **payload)

    def log_error(self, exc: BaseException, *, phase: str = "unknown", **payload: Any) -> dict:
        """Record an exception without swallowing it (the caller still re-raises)."""
        return self.log_event("error", phase=phase, type=type(exc).__name__,
                              message=str(exc), **payload)


# --------------------------------------------------------------------------- #
# reading logs back
# --------------------------------------------------------------------------- #
def iter_events(path: str | os.PathLike[str]) -> Iterator[dict]:
    """Yield events from a JSONL log, skipping malformed lines.

    A log truncated by a crash (or by a killed container) ends in a partial
    line. That is expected, not exceptional: the torn tail is logged at
    ``WARNING`` and skipped so the rest of the evidence stays readable.

    Args:
        path: JSONL log file.

    Yields:
        One decoded event ``dict`` per valid line.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such log file: {p}")
    bad = 0
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                logger.warning("%s:%d is not valid JSON (truncated write?); skipping", p, lineno)
    if bad:
        logger.warning("%s: skipped %d malformed line(s)", p, bad)


def read_events(path: str | os.PathLike[str], *, event: str | None = None) -> list[dict]:
    """Read a log into a list of event dicts, optionally filtered by event type."""
    return [e for e in iter_events(path) if event is None or e.get("event") == event]


def read_log(path: str | os.PathLike[str], *, event: str | None = None) -> Any:
    """Read a JSONL log into a :class:`pandas.DataFrame` for analysis.

    Payload keys are flattened into ``payload.*`` columns, so a training log
    becomes ``ts | run_id | seq | event | payload.step | payload.loss | …`` and
    can be filtered, joined and plotted directly.

    Args:
        path: JSONL log file.
        event: Keep only this event type (``"train_step"``, ``"metric"``, …).

    Returns:
        A ``pandas.DataFrame``; empty (no columns) if the log has no matching
        events.

    Raises:
        ImportError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pinned in requirements
        raise ImportError(
            "pandas is required to read logs as a DataFrame (`pip install pandas`); "
            "use evidence.read_events() for a dependency-free list of dicts"
        ) from exc

    records = read_events(path, event=event)
    if not records:
        return pd.DataFrame()
    frame = pd.json_normalize(records, sep=".")
    if "ts" in frame.columns:
        frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    return frame


def list_runs(*, include_log_dir: bool = True) -> list[str]:
    """Sorted run ids discoverable under ``outputs/runs`` and ``outputs/logs``.

    Run ids start with a UTC timestamp, so the sort is chronological.
    """
    ids: set[str] = set()
    if RUN_DIR.is_dir():
        for d in RUN_DIR.iterdir():
            if d.is_dir() and (d / DEFAULT_LOG_NAME).is_file():
                ids.add(d.name)
    if include_log_dir and LOG_DIR.is_dir():
        for f in LOG_DIR.glob("*.jsonl"):
            ids.add(f.stem)
    return sorted(ids)


def find_run_log(run_id: str) -> Path:
    """Locate the JSONL log for a run id.

    Searches ``outputs/runs/<run_id>/events.jsonl``, then
    ``outputs/logs/<run_id>.jsonl``, then any ``*.jsonl`` inside the run
    directory, then a prefix match on the run id (so you can pass just the
    timestamp portion).

    Raises:
        FileNotFoundError: With the list of known run ids, so the caller can see
            what *is* available.
    """
    direct = RUN_DIR / run_id / DEFAULT_LOG_NAME
    if direct.is_file():
        return direct
    flat = LOG_DIR / f"{run_id}.jsonl"
    if flat.is_file():
        return flat
    d = RUN_DIR / run_id
    if d.is_dir():
        candidates = sorted(d.glob("*.jsonl"))
        if candidates:
            return candidates[0]
    for known in list_runs():
        if known.startswith(run_id):
            return find_run_log(known)
    raise FileNotFoundError(
        f"no log found for run id {run_id!r}. Known runs: {list_runs() or '<none>'}"
    )


def load_run(run_id: str, *, event: str | None = None) -> Any:
    """Load a run's evidence log by id into a :class:`pandas.DataFrame`."""
    return read_log(find_run_log(run_id), event=event)
