"""Evidence logs: append-only, crash-safe, self-describing.

A log an investigator can rewrite is not evidence, and a log that a crash makes
unreadable is not evidence either. The two properties pinned hardest here:

* **append-only** — reopening a logger on an existing file adds to it and never
  truncates a byte that was already written;
* **torn-tail tolerant** — a half-written final line (container killed
  mid-``write``) is skipped with a warning, and everything before it still
  parses.

Every log in this module is written into ``tmp_path``; the suite never writes
into ``outputs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aiforensics.evidence import (
    EVENT_RUN_END,
    EVENT_RUN_START,
    RunLogger,
    iter_events,
    list_runs,
    read_events,
    read_log,
)


def _events(path: Path) -> list[dict]:
    return read_events(path)


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #
def test_a_run_opens_with_a_manifest_and_closes_with_a_status(tmp_path: Path) -> None:
    with RunLogger("unit-run", directory=tmp_path, config={"poison_rate": 0.05}) as log:
        path = log.path
        run_id = log.run_id
        log.log_metric("detection_rate", 0.91)

    events = _events(path)
    assert events[0]["event"] == EVENT_RUN_START
    assert events[-1]["event"] == EVENT_RUN_END
    assert events[-1]["payload"]["status"] == "ok"
    assert all(e["run_id"] == run_id for e in events)
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))

    manifest = events[0]["payload"]
    assert manifest["config"] == {"poison_rate": 0.05}
    assert manifest["seed"] is not None
    assert manifest["settings"]["hf_token"] in (None, "<redacted>")
    assert "environment" in manifest and "git" in manifest


def test_a_manifest_file_is_written_next_to_the_log(tmp_path: Path) -> None:
    with RunLogger("unit-run", directory=tmp_path) as log:
        pass
    payload = json.loads((tmp_path / "manifest.json").read_text())
    assert payload["run_id"] == log.run_id
    assert payload["log_path"] == str(log.path)


def test_run_info_describes_where_the_evidence_lives(tmp_path: Path) -> None:
    with RunLogger("unit-run", directory=tmp_path) as log:
        info = log.info
    assert info.directory == tmp_path
    assert info.path == tmp_path / "events.jsonl"
    assert info.to_dict()["run_id"] == log.run_id


def test_a_failing_run_records_the_failure_before_re_raising(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="boom"), RunLogger("doomed", directory=tmp_path) as log:
        log.log_train_step(0, loss=1.0)
        raise ValueError("boom")

    events = _events(tmp_path / "events.jsonl")
    kinds = [e["event"] for e in events]
    assert "error" in kinds
    assert events[kinds.index("error")]["payload"]["type"] == "ValueError"
    assert events[-1]["payload"]["status"] == "error"


def test_writing_to_a_closed_logger_is_a_loud_error(tmp_path: Path) -> None:
    log = RunLogger("closed", directory=tmp_path)
    log.close()
    with pytest.raises(RuntimeError, match="is closed"):
        log.log_metric("late", 1.0)


# --------------------------------------------------------------------------- #
# append-only
# --------------------------------------------------------------------------- #
def test_reopening_appends_and_never_truncates(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"

    with RunLogger("first", directory=tmp_path) as log:
        log.log_metric("a", 1.0)
    first_bytes = path.read_bytes()
    first_count = len(_events(path))

    with RunLogger("second", directory=tmp_path, capture_manifest=False,
                   write_manifest_file=False) as log:
        log.log_metric("b", 2.0)
    second_bytes = path.read_bytes()

    assert second_bytes.startswith(first_bytes), "existing evidence must be untouched"
    assert len(second_bytes) > len(first_bytes)
    events = _events(path)
    assert len(events) > first_count
    assert [e["payload"].get("name") for e in events if e["event"] == "metric"] == ["a", "b"]


def test_every_event_is_one_complete_line(tmp_path: Path) -> None:
    with RunLogger("lines", directory=tmp_path) as log:
        for step in range(5):
            log.log_train_step(step, loss=1.0 / (step + 1))
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["event"] for line in lines)


def test_a_torn_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with RunLogger("torn", directory=tmp_path) as log:
        log.log_metric("a", 1.0)
        log.log_metric("b", 2.0)
    intact = len(_events(path))

    # Simulate a container killed mid-write.
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": "2026-07-24T00:00:00+00:00", "event": "metr')

    assert len(_events(path)) == intact
    assert len(list(iter_events(path))) == intact
    assert len(read_log(path)) == intact


def test_reading_a_missing_log_is_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no such log file"):
        read_events(tmp_path / "nope.jsonl")


# --------------------------------------------------------------------------- #
# event payloads
# --------------------------------------------------------------------------- #
def test_artifacts_are_pinned_to_a_content_digest(tmp_path: Path) -> None:
    """A log line naming a file proves nothing; a digest ties it to bytes."""
    from aiforensics.integrity import hash_path

    artifact = tmp_path / "poisoned.json"
    artifact.write_text('{"rows": 40}')

    with RunLogger("artifacts", directory=tmp_path) as log:
        record = log.log_artifact(artifact, role="dataset")

    assert record["payload"]["digest"] == hash_path(artifact).aggregate
    assert record["payload"]["role"] == "dataset"
    assert record["payload"]["exists"] is True


def test_a_missing_artifact_is_recorded_as_missing(tmp_path: Path) -> None:
    with RunLogger("artifacts", directory=tmp_path) as log:
        record = log.log_artifact(tmp_path / "gone.bin", role="model")
    assert record["payload"]["exists"] is False
    assert record["payload"]["digest"] is None


def test_detection_results_are_logged_as_flat_json(tmp_path: Path, poisoned_pair: tuple) -> None:
    from aiforensics.metrics import from_manifest

    _, manifest = poisoned_pair
    result = from_manifest(manifest, manifest.poisoned_indices, label="oracle")

    with RunLogger("detection", directory=tmp_path) as log:
        log.log_detection("oracle", result, threshold=0.5)

    payload = _events(tmp_path / "events.jsonl")[-2]["payload"]
    assert payload["detector"] == "oracle"
    assert payload["result"]["detection_rate"] == 1.0
    assert payload["threshold"] == 0.5


def test_non_json_values_are_coerced_rather_than_crashing_the_run(tmp_path: Path) -> None:
    """Evidence collection must never be the thing that kills an experiment."""
    with RunLogger("coerce", directory=tmp_path) as log:
        log.log_params(
            indices=np.arange(3),
            rate=np.float32(0.25),
            where=tmp_path,
            unserialisable=object(),
        )
    payload = _events(tmp_path / "events.jsonl")[1]["payload"]
    assert payload["indices"] == [0, 1, 2]
    assert payload["rate"] == pytest.approx(0.25)
    assert payload["where"] == str(tmp_path)
    assert isinstance(payload["unserialisable"], str)


def test_inference_events_carry_the_retrieved_context(tmp_path: Path) -> None:
    with RunLogger("rag", directory=tmp_path) as log:
        log.log_inference(
            prompt="who wrote it?", response="nobody",
            model="qwen3-1.7b", sample_id=7,
            retrieved=[{"doc_id": "poison-abc-000", "score": 0.99}],
        )
    payload = _events(tmp_path / "events.jsonl")[1]["payload"]
    assert payload["sample_id"] == 7
    assert payload["retrieved"][0]["doc_id"] == "poison-abc-000"


# --------------------------------------------------------------------------- #
# reading back
# --------------------------------------------------------------------------- #
def test_read_log_flattens_payloads_into_columns(tmp_path: Path) -> None:
    with RunLogger("frame", directory=tmp_path) as log:
        for step in range(3):
            log.log_train_step(step, loss=1.0 - 0.1 * step, lr=2e-5)

    frame = read_log(tmp_path / "events.jsonl", event="train_step")
    assert list(frame["payload.step"]) == [0, 1, 2]
    assert frame["payload.loss"].iloc[0] == pytest.approx(1.0)
    assert len(frame) == 3


def test_read_log_of_a_filtered_empty_selection_is_empty(tmp_path: Path) -> None:
    with RunLogger("frame", directory=tmp_path) as log:
        log.log_metric("a", 1.0)
    assert read_log(tmp_path / "events.jsonl", event="inference").empty


def test_list_runs_returns_run_ids(tmp_path: Path) -> None:
    runs = list_runs()
    assert isinstance(runs, list)
    assert all(isinstance(run_id, str) for run_id in runs)
    assert runs == sorted(runs), "run ids start with a UTC stamp, so sorting is chronological"
