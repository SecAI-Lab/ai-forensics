"""Detection metrics — the project's headline numbers.

The research plan defines

    **detection accuracy = #detected poisoned samples / #poisoned samples**

which is *recall over the poisoned class*, not classification accuracy. The two
are routinely confused and the confusion changes the story a table tells, so
this module exposes the plan's quantity under its own name,
:attr:`DetectionResult.detection_rate`, right next to ``accuracy``,
``precision``, ``recall``, ``f1`` and ``fpr``, and states the identity
explicitly:

    ``detection_rate == recall`` (both are ``TP / (TP + FN)``)

Report it *with* precision or FPR. A detector that flags every sample scores a
perfect 1.0 detection rate and is useless; the false-positive rate is what
exposes that.

Conventions used throughout:

* Positive class = **poisoned / suspicious**. ``y_true[i] == True`` means
  sample ``i`` really is poisoned.
* Scores are "higher = more suspicious", matching
  :meth:`aiforensics.detectors.Detector.score`.
* Degenerate cases (no poisoned samples, no predicted positives, a constant
  score vector) return ``NaN`` for the affected metric and log a warning; they
  never raise and never silently return 0.0, because a silent zero is
  indistinguishable from a real result in a table.
"""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "DetectionResult",
    "as_mask",
    "best_threshold",
    "calibrate_threshold_on_clean",
    "confusion_matrix",
    "detection_metrics",
    "from_manifest",
    "manifest_to_mask",
    "pr_auc",
    "pr_curve",
    "roc_auc",
    "roc_curve",
    "summarize",
    "threshold_sweep",
]


def _safe_div(num: float, den: float, *, what: str, warn: bool = True) -> float:
    """Divide, returning ``NaN`` (with a warning) instead of raising on 0/0."""
    if den == 0:
        if warn:
            logger.warning(
                "%s is undefined (denominator is zero) — reporting NaN. "
                "Check that the ground truth and predictions are what you think.", what,
            )
        return float("nan")
    return float(num) / float(den)


# --------------------------------------------------------------------------- #
# input coercion
# --------------------------------------------------------------------------- #
def as_mask(
    obj: Any,
    n: int | None = None,
    *,
    name: str = "labels",
) -> np.ndarray:
    """Coerce labels/predictions into a 1-D boolean mask.

    Accepts, in this order of interpretation:

    * a boolean array/sequence — used as-is (length must equal ``n`` if given);
    * a 0/1 numeric array of length ``n`` — cast to bool;
    * any other integer array/sequence/set — treated as **indices** of positive
      samples, which requires ``n``;
    * an object exposing ``mask`` / ``to_mask()`` / ``poisoned_indices`` — see
      :func:`manifest_to_mask`.

    The index-vs-0/1 ambiguity is real: ``[0, 1, 1, 0]`` could be either. It is
    resolved by length — an integer array whose length equals ``n`` and whose
    values are all in ``{0, 1}`` is read as a 0/1 mask; otherwise as indices.
    Pass a ``bool`` array to remove all doubt.

    Args:
        obj: Labels, predictions, index list, or a manifest-like object.
        n: Total number of samples. Required for index input.
        name: Used in error messages.

    Returns:
        A 1-D ``np.ndarray`` of dtype ``bool``.

    Raises:
        ValueError: On an ambiguous or malformed input.
    """
    if obj is None:
        raise ValueError(f"{name} is None")

    # Manifest-like objects go through the duck-typed path.
    if not isinstance(obj, (np.ndarray, list, tuple, set, frozenset, range)) and (
        hasattr(obj, "mask") or hasattr(obj, "to_mask") or hasattr(obj, "poisoned_indices")
    ):
        return manifest_to_mask(obj, n=n)

    if isinstance(obj, (set, frozenset)):
        if n is None:
            raise ValueError(f"{name} given as an index set requires n=<total samples>")
        mask = np.zeros(int(n), dtype=bool)
        idx = np.asarray(sorted(obj), dtype=np.int64)
        if idx.size:
            _check_indices(idx, int(n), name)
            mask[idx] = True
        return mask

    arr = np.asarray(obj)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    if arr.dtype == bool:
        if n is not None and arr.size != n:
            raise ValueError(f"{name} has length {arr.size}, expected {n}")
        return arr.astype(bool, copy=False)

    if arr.size == 0:
        return np.zeros(int(n or 0), dtype=bool)

    if np.issubdtype(arr.dtype, np.floating):
        uniq = np.unique(arr)
        if not np.all(np.isin(uniq, (0.0, 1.0))):
            raise ValueError(
                f"{name} contains non-binary floats {uniq[:5]}…; if these are scores, use "
                f"threshold_sweep()/best_threshold() to binarise them first"
            )
        return arr.astype(bool)

    if np.issubdtype(arr.dtype, np.integer):
        looks_binary = bool(np.all(np.isin(arr, (0, 1))))
        if n is not None and arr.size == n and looks_binary:
            return arr.astype(bool)
        if n is None:
            raise ValueError(
                f"{name} looks like an index array; pass n=<total samples> so it can be "
                f"expanded into a mask (or pass a bool array instead)"
            )
        mask = np.zeros(int(n), dtype=bool)
        _check_indices(arr, int(n), name)
        mask[arr] = True
        return mask

    raise ValueError(f"cannot interpret {name} of dtype {arr.dtype} as labels or indices")


def _check_indices(idx: np.ndarray, n: int, name: str) -> None:
    if idx.min() < 0 or idx.max() >= n:
        raise ValueError(
            f"{name} contains index {int(idx.max())} out of range for n={n} "
            f"(indices must be 0-based)"
        )


def manifest_to_mask(manifest: Any, n: int | None = None) -> np.ndarray:
    """Extract a boolean ground-truth mask from a poison manifest.

    Duck-typed on purpose: the manifest type lives in
    ``aiforensics.data.poison.PoisonManifest`` (owned by another agent) and is
    **not imported here**, so there is no hard dependency and no import cycle.
    Any object works as long as it exposes one of:

    * ``mask`` / ``is_poisoned`` / ``labels`` — a bool array (preferred);
    * ``to_mask()`` / ``ground_truth()`` / ``mask()`` — a callable returning one;
    * ``poisoned_indices`` / ``indices`` — a sorted index list, plus a total
      count from ``n``, ``len(manifest)``, ``n_samples``, ``num_samples`` or
      ``size``.

    Args:
        manifest: Manifest-like object.
        n: Total sample count; only needed if the manifest cannot supply one.

    Returns:
        A 1-D boolean array, ``True`` where the sample is poisoned.

    Raises:
        TypeError: If the object exposes none of the expected attributes.
    """
    for attr in ("mask", "is_poisoned", "labels"):
        value = getattr(manifest, attr, None)
        if value is not None and not callable(value):
            arr = np.asarray(value)
            if arr.dtype == bool or np.issubdtype(arr.dtype, np.integer):
                return arr.astype(bool).reshape(-1)

    for attr in ("to_mask", "ground_truth", "as_mask", "mask"):
        fn = getattr(manifest, attr, None)
        if callable(fn):
            return np.asarray(fn()).astype(bool).reshape(-1)

    total = n
    if total is None:
        for attr in ("n", "n_samples", "num_samples", "size", "total"):
            value = getattr(manifest, attr, None)
            if isinstance(value, (int, np.integer)):
                total = int(value)
                break
    if total is None:
        try:
            total = len(manifest)
        except TypeError:
            total = None

    for attr in ("poisoned_indices", "indices", "poison_indices"):
        value = getattr(manifest, attr, None)
        if value is None:
            continue
        if callable(value):
            value = value()
        idx = np.asarray(list(value), dtype=np.int64)
        if total is None:
            total = int(idx.max()) + 1 if idx.size else 0
            logger.warning(
                "manifest exposes indices but no sample count; inferring n=%d from the "
                "largest index, which silently drops clean samples after it. Pass n=… "
                "for a correct denominator.", total,
            )
        mask = np.zeros(int(total), dtype=bool)
        if idx.size:
            _check_indices(idx, int(total), "poisoned_indices")
            mask[idx] = True
        return mask

    raise TypeError(
        f"{type(manifest).__name__} does not look like a poison manifest: expected one of "
        f"`mask`/`is_poisoned`/`labels` (bool array), `to_mask()`, or "
        f"`poisoned_indices` (+ a sample count)"
    )


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DetectionResult:
    """Confusion counts and derived rates for one detection run.

    Positive class = poisoned. ``NaN`` marks a metric that is undefined for the
    given inputs (see the module docstring) — it is never silently replaced by
    zero.

    Attributes:
        tp: Poisoned samples correctly flagged.
        fp: Clean samples wrongly flagged (the cost of a high detection rate).
        tn: Clean samples correctly left alone.
        fn: Poisoned samples missed.
        detection_rate: **The plan's headline metric** —
            #detected poisoned / #poisoned. Numerically identical to
            :attr:`recall`; both fields exist so a report can use the plan's
            wording without anyone wondering whether a second quantity was
            measured.
        accuracy: ``(TP + TN) / N`` over *all* samples. Dominated by the clean
            majority at realistic poison rates (5% poison → 95% accuracy by
            predicting "clean" always), so never report it alone.
        precision: ``TP / (TP + FP)`` — of everything flagged, how much was
            really poisoned.
        recall: ``TP / (TP + FN)``, identical to ``detection_rate``.
        f1: Harmonic mean of precision and recall.
        fpr: ``FP / (FP + TN)`` — the clean-data false-alarm rate.
        fnr: ``FN / (TP + FN)`` = ``1 - recall``.
        specificity: ``TN / (TN + FP)`` = ``1 - fpr``.
        balanced_accuracy: Mean of recall and specificity; the accuracy variant
            that is not fooled by class imbalance.
        mcc: Matthews correlation coefficient, the most honest single number
            for a heavily imbalanced binary problem.
        threshold: Score threshold used, when the predictions came from scores.
        label: Optional name for this row (detector, dataset, attack).
        n_poisoned: Denominator of ``detection_rate``, recorded explicitly.
    """

    tp: int
    fp: int
    tn: int
    fn: int
    n: int
    n_poisoned: int
    n_predicted_positive: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    specificity: float
    balanced_accuracy: float
    mcc: float
    detection_rate: float
    threshold: float | None = None
    label: str | None = None
    extra: dict = field(default_factory=dict)

    def confusion_matrix(self) -> np.ndarray:
        """``[[TN, FP], [FN, TP]]`` — rows are truth, columns are prediction.

        Same orientation as ``sklearn.metrics.confusion_matrix`` with
        ``labels=[0, 1]``, so it drops into existing plotting code.
        """
        return np.array([[self.tn, self.fp], [self.fn, self.tp]], dtype=np.int64)

    def to_dict(self) -> dict:
        """JSON-ready representation, ready for ``RunLogger.log_detection``."""
        return {
            "label": self.label,
            "threshold": self.threshold,
            "n": self.n,
            "n_poisoned": self.n_poisoned,
            "n_predicted_positive": self.n_predicted_positive,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "detection_rate": self.detection_rate,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "specificity": self.specificity,
            "balanced_accuracy": self.balanced_accuracy,
            "mcc": self.mcc,
            **({"extra": self.extra} if self.extra else {}),
        }

    def to_markdown(self, *, include_confusion: bool = True) -> str:
        """Markdown block ready to paste into the report."""
        title = f"**Detection metrics{f' — {self.label}' if self.label else ''}**"
        thr = "" if self.threshold is None else f" (threshold = {self.threshold:.6g})"
        rows = [
            ("detection rate (= recall on poisoned)", self.detection_rate),
            ("precision", self.precision),
            ("F1", self.f1),
            ("accuracy", self.accuracy),
            ("false-positive rate", self.fpr),
            ("balanced accuracy", self.balanced_accuracy),
            ("MCC", self.mcc),
        ]
        lines = [
            title + thr,
            "",
            f"`n = {self.n}`, poisoned = {self.n_poisoned}, flagged = {self.n_predicted_positive}",
            "",
            "| metric | value |",
            "|---|---|",
        ]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in rows]
        if include_confusion:
            lines += [
                "",
                "| | pred clean | pred poisoned |",
                "|---|---|---|",
                f"| **true clean** | {self.tn} | {self.fp} |",
                f"| **true poisoned** | {self.fn} | {self.tp} |",
            ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"detection_rate={_fmt(self.detection_rate)} precision={_fmt(self.precision)} "
            f"f1={_fmt(self.f1)} fpr={_fmt(self.fpr)} acc={_fmt(self.accuracy)} "
            f"(n={self.n}, poisoned={self.n_poisoned})"
        )


def _fmt(value: float) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"{value:.4f}"


# --------------------------------------------------------------------------- #
# core metric
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: Any, y_pred: Any, *, n: int | None = None) -> np.ndarray:
    """``[[TN, FP], [FN, TP]]`` for boolean/0-1/index-style inputs."""
    yt, yp = _align(y_true, y_pred, n=n)
    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fn = int(np.sum(yt & ~yp))
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def _binary_labels_len(obj: Any) -> int | None:
    """Length of ``obj`` if it is a plain 0/1 label array, else ``None``.

    Only used to disambiguate the ``n is None`` case in :func:`_align`; sets and
    manifest-like objects are excluded because those are unambiguously indices.
    """
    if not isinstance(obj, (np.ndarray, list, tuple)):
        return None
    arr = np.asarray(obj)
    if arr.ndim != 1 or arr.size == 0:
        return None
    if arr.dtype == bool:
        return int(arr.size)
    if np.issubdtype(arr.dtype, np.integer) and bool(np.all(np.isin(arr, (0, 1)))):
        return int(arr.size)
    return None


def _align(y_true: Any, y_pred: Any, *, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Coerce both arguments to boolean masks of the same length.

    When ``n`` is omitted and *both* arguments are equal-length 0/1 arrays they
    are read as label masks, matching the ``sklearn`` calling convention that
    everyone reaches for first. A 0/1 array is formally ambiguous — it could be
    a mask or a list of indices — so pass ``n=`` explicitly whenever index
    semantics are intended; that path is unchanged.
    """
    if n is None:
        lt = _binary_labels_len(y_true)
        if lt is not None and lt == _binary_labels_len(y_pred):
            n = lt

    yt = as_mask(y_true, n=n, name="y_true")
    yp = as_mask(y_pred, n=n if n is not None else yt.size, name="y_pred")
    if yt.size != yp.size:
        raise ValueError(
            f"y_true has {yt.size} entries but y_pred has {yp.size}; they must describe "
            f"the same sample set (pass n=… if one of them is an index list)"
        )
    return yt, yp


def detection_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    n: int | None = None,
    threshold: float | None = None,
    label: str | None = None,
    warn: bool = True,
) -> DetectionResult:
    """Compute the full metric set for a set of binary detections.

    Args:
        y_true: Ground truth. Bool mask, 0/1 array, index list of poisoned
            samples, or a ``PoisonManifest``-like object (see
            :func:`manifest_to_mask`).
        y_pred: Predictions, same accepted forms.
        n: Total number of samples — required when either argument is an index
            list rather than a mask.
        threshold: Threshold that produced ``y_pred``, recorded for the report.
        label: Optional row label (detector / dataset / attack).
        warn: Emit warnings for degenerate inputs. Turned off internally by
            :func:`threshold_sweep` so a 200-point sweep does not emit 200
            identical warnings.

    Returns:
        A :class:`DetectionResult`. Metrics that are undefined for the given
        inputs are ``NaN``, never 0.

    Raises:
        ValueError: If the two arguments describe different sample counts.
    """
    yt, yp = _align(y_true, y_pred, n=n)

    tp = int(np.sum(yt & yp))
    fp = int(np.sum(~yt & yp))
    tn = int(np.sum(~yt & ~yp))
    fn = int(np.sum(yt & ~yp))
    total = int(yt.size)
    n_pos = tp + fn
    n_flagged = tp + fp

    if warn and n_pos == 0:
        warnings.warn(
            "no poisoned samples in the ground truth: detection_rate/recall are undefined "
            "(NaN). Check the poison manifest — a detection rate cannot be computed "
            "against an unpoisoned set.",
            RuntimeWarning,
            stacklevel=2,
        )
    if warn and n_flagged == total and total > 0:
        warnings.warn(
            "the detector flagged every sample: detection_rate will be 1.0 but is "
            "meaningless without the false-positive rate. Report fpr/precision alongside it.",
            RuntimeWarning,
            stacklevel=2,
        )

    recall = _safe_div(tp, n_pos, what="recall / detection_rate", warn=warn)
    precision = _safe_div(tp, n_flagged, what="precision", warn=warn)
    specificity = _safe_div(tn, tn + fp, what="specificity", warn=warn)
    fpr = _safe_div(fp, fp + tn, what="false-positive rate", warn=warn)
    fnr = _safe_div(fn, n_pos, what="false-negative rate", warn=warn)
    accuracy = _safe_div(tp + tn, total, what="accuracy", warn=warn)

    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = _safe_div(2 * tp, 2 * tp + fp + fn, what="F1", warn=warn)
    else:
        f1 = 2 * precision * recall / (precision + recall)

    balanced = (
        float("nan") if math.isnan(recall) or math.isnan(specificity)
        else (recall + specificity) / 2.0
    )

    mcc_den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, mcc_den, what="MCC", warn=False)

    return DetectionResult(
        tp=tp, fp=fp, tn=tn, fn=fn,
        n=total, n_poisoned=n_pos, n_predicted_positive=n_flagged,
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        fpr=fpr, fnr=fnr, specificity=specificity,
        balanced_accuracy=balanced, mcc=mcc,
        detection_rate=recall,  # identical by definition; see the class docstring
        threshold=threshold, label=label,
    )


def from_manifest(
    manifest: Any,
    predicted: Any,
    *,
    n: int | None = None,
    threshold: float | None = None,
    label: str | None = None,
) -> DetectionResult:
    """Evaluate predictions against a ``PoisonManifest``-style ground truth.

    Convenience wrapper: pulls the boolean ground truth out of the manifest
    (duck-typed, see :func:`manifest_to_mask`) and accepts predictions as
    either a boolean mask or a list of predicted-poisoned indices.

    ``aiforensics.data.poison.PoisonManifest`` is **not imported** by this
    module — the coupling is structural only, which keeps
    :mod:`aiforensics.metrics` importable without the data subpackage and
    avoids an import cycle.

    Args:
        manifest: Manifest-like ground truth.
        predicted: Boolean mask or index iterable of samples flagged as poisoned.
        n: Total sample count, if the manifest cannot supply it.
        threshold: Threshold used, recorded in the result.
        label: Optional row label.

    Returns:
        A :class:`DetectionResult`.
    """
    y_true = manifest_to_mask(manifest, n=n)
    y_pred = as_mask(predicted, n=y_true.size, name="predicted")
    return detection_metrics(y_true, y_pred, threshold=threshold, label=label)


# --------------------------------------------------------------------------- #
# score-based evaluation
# --------------------------------------------------------------------------- #
def _prepare_scores(y_true: Any, scores: Any, n: int | None) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    yt = as_mask(y_true, n=n if n is not None else s.size, name="y_true")
    if yt.size != s.size:
        raise ValueError(f"y_true has {yt.size} entries but scores has {s.size}")
    if not np.all(np.isfinite(s)):
        bad = int(np.sum(~np.isfinite(s)))
        logger.warning(
            "%d non-finite score(s) present; they are ranked as the most suspicious "
            "(+inf) or least suspicious (-inf) accordingly. NaN scores become -inf.", bad,
        )
        s = np.nan_to_num(s, nan=-np.inf, posinf=np.finfo(np.float64).max,
                          neginf=-np.finfo(np.float64).max)
    return yt, s


def calibrate_threshold_on_clean(
    clean_scores: Any,
    target_fpr: float = 0.05,
) -> float:
    """Smallest threshold whose realised flag rate on clean data is ``<= target_fpr``.

    This is the calibration the unknown-poisoning setting forces: the
    investigator has no labelled poison, but can obtain data believed clean, and
    that fixes an alarm budget.

    **Why not just take the quantile.** ``np.quantile(clean, 1 - fpr)`` is wrong
    whenever the score is discrete or heavily tied, because the returned value is
    itself a score many clean points *equal*, and ``score >= threshold`` then
    flags every one of them. Measured on this project's own logit-lens run, whose
    statistic is an integer layer index: a 5% budget produced a **30.8%** realised
    clean flag rate, which in turn made a detector with ROC-AUC 0.840 report a
    detection accuracy of 0.000. The discrimination was fine; the threshold was
    in an impossible place.

    Scanning the distinct clean scores and taking the first that satisfies the
    budget is exact, costs one sort, and degenerates to the quantile when the
    scores are continuous and untied.

    Args:
        clean_scores: Scores on data believed clean.
        target_fpr: Alarm budget — the fraction of clean data allowed to flag.

    Returns:
        A threshold ``t`` such that ``mean(clean_scores >= t) <= target_fpr``.
        When no clean score itself meets the budget — every clean point ties at
        the maximum, which happens whenever a detector is identically zero on
        clean data — the threshold sits *just above* the largest clean score.
        That spends none of the budget and still flags anything above it, which
        matters: a detector can separate the classes perfectly and have a
        degenerate clean distribution, and returning ``inf`` there would throw a
        working detector away. Measured on this project's activation-clustering
        row: ROC-AUC 1.000 alongside a reported detection accuracy of 0.000.

    Raises:
        ValueError: If ``clean_scores`` is empty or ``target_fpr`` is not in
            ``[0, 1]``.
    """
    scores = np.asarray(clean_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("clean_scores is empty; nothing to calibrate against")
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be in [0, 1], got {target_fpr}")

    # The flag rate is a non-increasing step function of the threshold, so
    # scanning the distinct values *upward* and stopping at the first that meets
    # the budget gives the smallest admissible threshold — the one that keeps as
    # much detection power as the budget allows. Scanning downward instead would
    # return the maximum score, which meets any budget and detects nothing.
    values = np.unique(scores)
    rates = (scores[None, :] >= values[:, None]).mean(axis=1)
    ok = np.flatnonzero(rates <= target_fpr)
    if ok.size:
        return float(values[ok[0]])
    # No clean score meets the budget. Sit immediately above the largest one:
    # realised clean flag rate 0, and anything scoring higher still fires.
    return float(np.nextafter(values[-1], np.inf))


def threshold_sweep(
    y_true: Any,
    scores: Any,
    thresholds: Sequence[float] | np.ndarray | None = None,
    *,
    n: int | None = None,
    max_thresholds: int = 512,
    as_frame: bool = True,
) -> Any:
    """Evaluate a score vector at many thresholds.

    Predictions are ``scores >= threshold`` (higher = more suspicious, the
    convention :meth:`aiforensics.detectors.Detector.score` follows).

    Args:
        y_true: Ground truth (any form :func:`as_mask` accepts).
        scores: Per-sample suspicion scores.
        thresholds: Explicit thresholds. By default every distinct score is
            used — that is the complete set of achievable operating points —
            sub-sampled to ``max_thresholds`` quantiles if there are more.
        n: Total sample count when ``y_true`` is an index list.
        max_thresholds: Cap on the number of evaluated thresholds.
        as_frame: Return a ``pandas.DataFrame`` (default) instead of a list of
            dicts. Falls back to the list with a warning if pandas is missing.

    Returns:
        DataFrame or list of dicts, one row per threshold, in ascending
        threshold order.
    """
    yt, s = _prepare_scores(y_true, scores, n)

    if thresholds is None:
        uniq = np.unique(s)
        if uniq.size > max_thresholds:
            qs = np.linspace(0.0, 1.0, max_thresholds)
            uniq = np.unique(np.quantile(uniq, qs))
        thresholds = uniq
    thr = np.asarray(thresholds, dtype=np.float64).reshape(-1)

    rows: list[dict] = []
    for t in np.sort(thr):
        res = detection_metrics(yt, s >= t, threshold=float(t), warn=False)
        rows.append(res.to_dict())

    if not as_frame:
        return rows
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pinned in requirements
        logger.warning("pandas not installed; returning a list of dicts instead of a DataFrame")
        return rows
    return pd.DataFrame(rows)


def roc_curve(y_true: Any, scores: Any, *, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC points ``(fpr, tpr, thresholds)`` in increasing-FPR order.

    Computed directly from sorted scores with proper handling of tied scores
    (all samples sharing a score move across the threshold together).
    """
    yt, s = _prepare_scores(y_true, scores, n)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = yt[order].astype(np.int64)

    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = 1 + idx - tps

    n_pos = int(yt.sum())
    n_neg = int((~yt).sum())
    tpr = tps / n_pos if n_pos else np.full(tps.shape, np.nan)
    fpr = fps / n_neg if n_neg else np.full(fps.shape, np.nan)
    thresholds = s_sorted[idx]
    # Prepend the (0, 0) operating point: flag nothing.
    return (np.r_[0.0, fpr], np.r_[0.0, tpr], np.r_[np.inf, thresholds])


def roc_auc(y_true: Any, scores: Any, *, n: int | None = None) -> float:
    """Area under the ROC curve.

    Computed as the tie-corrected Mann-Whitney U statistic (the probability
    that a random poisoned sample scores above a random clean one), which is
    exact and needs no curve integration.

    Returns:
        AUC in ``[0, 1]``, or ``NaN`` when one class is absent (AUC is
        undefined without both a positive and a negative sample).
    """
    yt, s = _prepare_scores(y_true, scores, n)
    n_pos = int(yt.sum())
    n_neg = int(yt.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "ROC AUC is undefined with %d poisoned and %d clean samples — reporting NaN",
            n_pos, n_neg,
        )
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[yt].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (needed for an exact tied AUC)."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    for i in range(1, values.size + 1):
        if i == values.size or sorted_vals[i] != sorted_vals[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2.0
            start = i
    return ranks


def pr_curve(y_true: Any, scores: Any, *, n: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision/recall points ``(precision, recall, thresholds)``.

    Ordered by decreasing threshold, i.e. increasing recall, matching
    ``sklearn.metrics.precision_recall_curve``'s orientation.
    """
    yt, s = _prepare_scores(y_true, scores, n)
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = yt[order].astype(np.int64)

    distinct = np.where(np.diff(s_sorted))[0]
    idx = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = 1 + idx - tps
    n_pos = int(yt.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tps + fps > 0, tps / np.maximum(tps + fps, 1), np.nan)
        recall = tps / n_pos if n_pos else np.full(tps.shape, np.nan)
    return precision, recall, s_sorted[idx]


def pr_auc(y_true: Any, scores: Any, *, n: int | None = None) -> float:
    """Average precision — area under the precision/recall curve.

    Uses the step-wise estimator ``sum_k (R_k - R_{k-1}) * P_k`` (the same one
    as ``sklearn.metrics.average_precision_score``), which does not
    over-estimate the way trapezoidal interpolation does.

    PR AUC is the more informative summary at the low poison rates used here:
    ROC AUC looks flattering when 95% of the data is clean.

    Returns:
        Average precision in ``[0, 1]``, or ``NaN`` if there are no poisoned samples.
    """
    yt, s = _prepare_scores(y_true, scores, n)
    if int(yt.sum()) == 0:
        logger.warning("average precision is undefined with no poisoned samples — reporting NaN")
        return float("nan")
    precision, recall, _ = pr_curve(yt, s)
    prev_r = 0.0
    total = 0.0
    for p, r in zip(precision, recall, strict=True):
        if math.isnan(p):
            continue
        total += (r - prev_r) * p
        prev_r = r
    return float(total)


def best_threshold(
    y_true: Any,
    scores: Any,
    *,
    criterion: str = "f1",
    n: int | None = None,
    max_thresholds: int = 512,
) -> tuple[float, DetectionResult]:
    """Pick the score threshold that optimises a criterion on labelled data.

    .. warning::
       This tunes on the labels it is given. Choosing a threshold on the same
       split you then report is optimistic by construction — fit it on a
       held-out calibration split, or report it as an oracle upper bound and
       say so.

    Args:
        y_true: Ground truth (any form :func:`as_mask` accepts).
        scores: Per-sample suspicion scores.
        criterion: ``"f1"`` (default), ``"youden"`` (Youden's J = TPR - FPR,
            i.e. the point furthest from the diagonal on the ROC curve),
            ``"accuracy"``, ``"balanced_accuracy"`` or ``"mcc"``.
        n: Total sample count when ``y_true`` is an index list.
        max_thresholds: Cap on candidate thresholds.

    Returns:
        ``(threshold, DetectionResult)`` for the best operating point.

    Raises:
        ValueError: On an unknown criterion or empty score vector.
    """
    criteria = {"f1", "youden", "accuracy", "balanced_accuracy", "mcc"}
    if criterion not in criteria:
        raise ValueError(f"unknown criterion {criterion!r}; valid: {sorted(criteria)}")

    yt, s = _prepare_scores(y_true, scores, n)
    if s.size == 0:
        raise ValueError("cannot choose a threshold from an empty score vector")

    candidates = np.unique(s)
    if candidates.size > max_thresholds:
        candidates = np.unique(np.quantile(candidates, np.linspace(0.0, 1.0, max_thresholds)))

    best_val = -math.inf
    best: tuple[float, DetectionResult] | None = None
    for t in candidates:
        res = detection_metrics(yt, s >= t, threshold=float(t), warn=False)
        value = (res.recall - res.fpr) if criterion == "youden" else getattr(res, criterion)
        if value is None or math.isnan(value):
            continue
        if value > best_val:
            best_val = value
            best = (float(t), res)

    if best is None:
        logger.warning(
            "no threshold produced a defined %s (degenerate labels or scores); "
            "falling back to the minimum score", criterion,
        )
        t = float(candidates[0])
        return t, detection_metrics(yt, s >= t, threshold=t)
    return best


def summarize(results: Iterable[DetectionResult]) -> Any:
    """Stack several :class:`DetectionResult` rows into one comparison table.

    Handy for the report's "detector × attack" grid.

    Returns:
        A ``pandas.DataFrame`` (or a list of dicts if pandas is unavailable).
    """
    rows = [r.to_dict() for r in results]
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        logger.warning("pandas not installed; returning a list of dicts")
        return rows
    return pd.DataFrame(rows)
