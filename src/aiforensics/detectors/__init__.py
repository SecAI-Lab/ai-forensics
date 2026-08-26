"""Detector plug-in seam — interfaces, registry, and **no detection logic**.

Detectors form a two-level tree: a **category** is a detection approach, a
**technique** is one concrete method inside it. Category keys are approach
names, never product names — so the RAG axis is ``rag-defense`` and
*RAGDefender* is a technique within it, exactly parallel to
``neuron-activation`` containing ``sae`` / ``badacts-nas`` / ``logit-lens``.

Right now this subpackage contains only the shared contract
(:class:`~aiforensics.detectors.base.Detector`), the registry, a category-level
ensemble, and a random baseline that proves the plumbing works. Every technique
is **unimplemented on purpose** — this deliverable is the environment, not the
detectors.

=====================  ================================  =========================
category               techniques                        reserved module
=====================  ================================  =========================
``input-anomaly``      ``onion``, ``rap``                ``input_anomaly.py``
``feature-space``      ``hidden-svd`` + variants         ``feature_space.py``
``neuron-activation``  ``sae``, ``badacts-nas``, ``logit-lens``  ``neuron_activation.py``
``rag-defense``        ``ragdefender`` + variants        ``rag_defense.py``
=====================  ================================  =========================

These categories are defined by the *signal* they read, not by the system they
were first demonstrated on, so they extend to agentic environments as-is: point
an existing technique at agent inputs, forward passes or retrieved context. An
agentic deployment mainly widens the evidence surface (orchestrator decisions,
agent-to-agent messages, tool-call traces) and changes the unit of detection to
a turn or a tool call. No agent-specific code lives here — it is out of scope
for this deliverable.

Inspect the tree at any time::

    from aiforensics.detectors import format_tree
    print(format_tree())

Add a technique to an existing category — the category is inferred when the key
is already in the plan::

    from aiforensics.detectors import Detector, register_detector

    @register_detector("onion")
    class OnionDetector(Detector):
        name = "onion"
        modality = "text"

        def score(self, samples):
            ...  # higher = more suspicious

    from aiforensics.detectors import get_detector
    det = get_detector("onion", model="qwen3-1.7b")
    result = det.evaluate(samples, manifest, threshold=t)
    print(result.detection_rate)

Start a new family with :func:`register_category`, then register techniques into
it the same way. See :mod:`aiforensics.detectors.base` for the full guide.
"""

from __future__ import annotations

from .base import (
    CATEGORIES,
    CategoryEnsemble,
    Detector,
    DetectorCategory,
    RandomDetector,
    Technique,
    category_of,
    detector_tree,
    format_tree,
    get_category,
    get_detector,
    list_categories,
    list_detectors,
    list_techniques,
    register_category,
    register_detector,
)

__all__ = [
    "CATEGORIES",
    "CategoryEnsemble",
    "Detector",
    "DetectorCategory",
    "RandomDetector",
    "Technique",
    "category_of",
    "detector_tree",
    "format_tree",
    "get_category",
    "get_detector",
    "list_categories",
    "list_detectors",
    "list_techniques",
    "register_category",
    "register_detector",
]
