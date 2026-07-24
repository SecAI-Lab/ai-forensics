"""Evidence collection: append-only JSONL logs of training and inference runs.

Training and inference logs are one of the evidence sources the threat model
assumes an investigator has. :class:`~aiforensics.evidence.logger.RunLogger`
produces them in a form that is usable as evidence rather than as debug output:
append-only, one self-contained JSON object per line, flushed per event, opened
with a manifest describing the code, config, environment and seed, and with
every referenced artifact pinned to a content digest.

Write::

    from aiforensics.evidence import RunLogger

    with RunLogger("qwen3-sst2-badnet", config={"poison_rate": 0.05}) as log:
        log.log_artifact("data/poisoned/sst2-badnet", role="dataset")
        log.log_train_step(step=0, loss=2.31, lr=2e-5)
        log.log_inference(prompt=x, response=y, retrieved=doc_ids)
        log.log_metric("detection_rate", 0.91)

Read::

    from aiforensics.evidence import load_run, list_runs

    df = load_run(list_runs()[-1], event="train_step")
"""

from __future__ import annotations

from .logger import (
    DEFAULT_LOG_NAME,
    EVENT_RUN_END,
    EVENT_RUN_START,
    RunInfo,
    RunLogger,
    find_run_log,
    iter_events,
    list_runs,
    load_run,
    read_events,
    read_log,
)

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
