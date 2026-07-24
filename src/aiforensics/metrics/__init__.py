"""Evaluation metrics for poison/backdoor detection.

The headline number in this project is

    **detection accuracy = #detected poisoned samples / #poisoned samples**

which is recall over the poisoned class. It is exposed as
:attr:`~aiforensics.metrics.detection.DetectionResult.detection_rate` and is
numerically identical to ``recall`` — both names exist so nobody has to guess
which quantity a table is showing. Always report it together with precision or
the false-positive rate: flagging everything gives a perfect detection rate.

::

    from aiforensics.metrics import detection_metrics, best_threshold, roc_auc

    res = detection_metrics(y_true, y_pred, label="onion/sst2-badnet")
    print(res.to_markdown())

    thr, best = best_threshold(y_true, scores, criterion="f1")
    print(roc_auc(y_true, scores), thr, best.detection_rate)
"""

from __future__ import annotations

from .detection import (
    DetectionResult,
    as_mask,
    best_threshold,
    confusion_matrix,
    detection_metrics,
    from_manifest,
    manifest_to_mask,
    pr_auc,
    pr_curve,
    roc_auc,
    roc_curve,
    summarize,
    threshold_sweep,
)

__all__ = [
    "DetectionResult",
    "as_mask",
    "best_threshold",
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
