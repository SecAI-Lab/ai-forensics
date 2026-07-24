"""Detection metrics — the numbers this project reports.

Two things are pinned here above all else:

1. **An oracle detector scores 1.0.** A detector that flags exactly the
   trigger-bearing rows of a poisoned dataset must come back with
   ``detection_rate == 1.0`` and ``precision == 1.0``. If that ever breaks,
   every table in every report is wrong, whatever the detector did.
2. **Degenerate inputs return NaN and warn.** No poisoned samples, nothing
   flagged, everything flagged — these must never crash and must never quietly
   return ``0.0``, because a silent zero is indistinguishable from a real
   result once it is in a table.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest

from aiforensics.detectors import Detector
from aiforensics.metrics import (
    as_mask,
    best_threshold,
    confusion_matrix,
    detection_metrics,
    from_manifest,
    manifest_to_mask,
    pr_auc,
    roc_auc,
    summarize,
    threshold_sweep,
)


class OracleDetector(Detector):
    """Flags exactly the rows whose text carries the trigger. Upper bound only."""

    name = "oracle"
    modality = "text"

    def __init__(self, marker: str = " cf", **config: Any) -> None:
        super().__init__(threshold=0.5, marker=marker, **config)
        self.marker = marker

    def score(self, samples: Any) -> np.ndarray:
        return np.asarray([1.0 if self.marker in str(s) else 0.0 for s in samples])


# --------------------------------------------------------------------------- #
# the headline invariant
# --------------------------------------------------------------------------- #
def test_oracle_detector_scores_a_perfect_run(poisoned_pair: tuple) -> None:
    poisoned, manifest = poisoned_pair
    result = OracleDetector().evaluate(poisoned["text"], manifest, label="oracle/synthetic")

    assert result.detection_rate == 1.0
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.fpr == 0.0
    assert result.tp == manifest.num_poisoned
    assert result.fp == 0 and result.fn == 0
    assert result.n == manifest.num_samples
    assert result.n_poisoned == manifest.num_poisoned


def test_from_manifest_accepts_predicted_indices(poisoned_pair: tuple) -> None:
    _, manifest = poisoned_pair
    result = from_manifest(manifest, manifest.poisoned_indices)
    assert result.detection_rate == 1.0
    assert result.precision == 1.0


def test_detection_rate_is_recall_by_definition() -> None:
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    result = detection_metrics(y_true, y_pred)
    assert result.detection_rate == result.recall == pytest.approx(2 / 3)
    assert result.precision == pytest.approx(2 / 3)
    assert result.fpr == pytest.approx(1 / 5)


def test_a_detector_that_misses_everything_scores_zero_not_nan() -> None:
    result = detection_metrics(np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0]))
    assert result.detection_rate == 0.0
    assert math.isnan(result.precision), "precision is undefined with nothing flagged"


def test_confusion_matrix_orientation() -> None:
    y_true = np.array([1, 1, 0, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    assert np.array_equal(confusion_matrix(y_true, y_pred), np.array([[2, 1], [1, 1]]))
    assert np.array_equal(
        detection_metrics(y_true, y_pred).confusion_matrix(), np.array([[2, 1], [1, 1]])
    )


# --------------------------------------------------------------------------- #
# degenerate cases
# --------------------------------------------------------------------------- #
def test_no_poisoned_samples_returns_nan_and_warns() -> None:
    with pytest.warns(RuntimeWarning, match="no poisoned samples"):
        result = detection_metrics(np.zeros(6, dtype=bool), np.zeros(6, dtype=bool))
    assert math.isnan(result.detection_rate)
    assert math.isnan(result.recall)
    assert result.n_poisoned == 0
    assert result.accuracy == 1.0  # still defined: every clean row was left alone


def test_flagging_everything_warns_that_the_rate_is_meaningless() -> None:
    y_true = np.array([True, False, False, False])
    with pytest.warns(RuntimeWarning, match="flagged every sample"):
        result = detection_metrics(y_true, np.ones(4, dtype=bool))
    assert result.detection_rate == 1.0
    assert result.precision == 0.25
    assert result.fpr == 1.0, "the FPR is what exposes a useless perfect recall"


def test_metrics_never_raise_on_an_empty_ground_truth() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = detection_metrics(np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
    assert result.n == 0
    assert math.isnan(result.detection_rate)
    assert math.isnan(result.accuracy)


def test_mismatched_lengths_are_a_loud_error() -> None:
    with pytest.raises(ValueError, match="length 4, expected 5"):
        detection_metrics(np.zeros(5, dtype=bool), np.zeros(4, dtype=bool))


def test_to_dict_and_markdown_survive_nan() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = detection_metrics(np.zeros(4, dtype=bool), np.zeros(4, dtype=bool))
    assert "n/a" in result.to_markdown()
    assert math.isnan(result.to_dict()["detection_rate"])


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #
def test_as_mask_expands_indices_when_n_is_given() -> None:
    assert np.array_equal(as_mask([1, 3], n=5), np.array([0, 1, 0, 1, 0], dtype=bool))


def test_as_mask_refuses_ambiguous_indices_without_n() -> None:
    with pytest.raises(ValueError, match="pass n="):
        as_mask([1, 3])


def test_as_mask_rejects_raw_scores() -> None:
    with pytest.raises(ValueError, match="non-binary floats"):
        as_mask(np.array([0.2, 0.9, 0.4]), n=3)


def test_as_mask_catches_an_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="out of range"):
        as_mask([0, 7], n=5)


def test_manifest_to_mask_reads_a_real_manifest(poisoned_pair: tuple) -> None:
    _, manifest = poisoned_pair
    assert np.array_equal(manifest_to_mask(manifest), manifest.is_poisoned)


def test_manifest_to_mask_rejects_a_non_manifest() -> None:
    with pytest.raises(TypeError, match="does not look like a poison manifest"):
        manifest_to_mask(object())


# --------------------------------------------------------------------------- #
# score-based evaluation
# --------------------------------------------------------------------------- #
def test_roc_and_pr_auc_on_a_perfectly_separable_score() -> None:
    y_true = np.array([True, True, False, False])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(y_true, scores) == 1.0
    assert pr_auc(y_true, scores) == pytest.approx(1.0)


def test_roc_auc_of_a_random_score_is_undefined_without_both_classes() -> None:
    assert math.isnan(roc_auc(np.zeros(4, dtype=bool), np.array([0.1, 0.2, 0.3, 0.4])))


def test_threshold_sweep_covers_every_operating_point() -> None:
    y_true = np.array([True, True, False, False])
    scores = np.array([0.9, 0.5, 0.5, 0.1])
    sweep = threshold_sweep(y_true, scores)
    thresholds = list(sweep["threshold"]) if hasattr(sweep, "columns") else [
        row["threshold"] for row in sweep
    ]
    assert thresholds == sorted(thresholds)
    assert len(thresholds) == len(np.unique(scores))


def test_best_threshold_returns_a_threshold_and_a_result() -> None:
    y_true = np.array([True, True, False, False])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    threshold, result = best_threshold(y_true, scores, criterion="f1")
    assert isinstance(threshold, float)
    assert result.f1 == 1.0
    assert result.threshold == threshold


def test_best_threshold_rejects_an_unknown_criterion() -> None:
    with pytest.raises(ValueError, match="unknown criterion"):
        best_threshold(np.array([True, False]), np.array([1.0, 0.0]), criterion="vibes")


def test_summarize_stacks_rows() -> None:
    rows = [
        detection_metrics(np.array([1, 0, 1, 0]), np.array([1, 0, 1, 0]), label="oracle"),
        detection_metrics(np.array([1, 0, 1, 0]), np.array([0, 1, 0, 1]), label="inverted"),
    ]
    table = summarize(rows)
    labels = list(table["label"]) if hasattr(table, "columns") else [r["label"] for r in table]
    assert labels == ["oracle", "inverted"]


# --------------------------------------------------------------------------- #
# the baseline detector, end to end
# --------------------------------------------------------------------------- #
def test_random_baseline_runs_through_the_same_path(poisoned_pair: tuple) -> None:
    """The null row of the report: chance-level, but wired identically."""
    from aiforensics.detectors import get_detector

    poisoned, manifest = poisoned_pair
    result = get_detector("random", seed=5).evaluate(poisoned["text"], manifest)
    assert result.n == manifest.num_samples
    assert result.n_poisoned == manifest.num_poisoned
    assert 0.0 <= result.detection_rate <= 1.0
    assert result.label == "random"
