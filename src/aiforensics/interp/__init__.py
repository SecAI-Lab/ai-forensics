"""Internal-representation analysis tools.

Instruments, not detectors: this package exposes ways to *read* what a model is
computing internally — vocabulary-space lenses over intermediate layers, sparse
autoencoder features over activations. The detectors that turn those readings
into a poisoned/clean decision live in :mod:`aiforensics.detectors`, so a tool
can be used for exploration without committing to a detection rule.
"""

from __future__ import annotations

from .jacobian_lens import JacobianLens
from .lens import LensReadout, LogitLens, TunedLens, lens_trajectory
from .sae import SaeConfig, SparseAutoencoder, collect_activations

__all__ = [
    "JacobianLens",
    "LensReadout",
    "LogitLens",
    "SaeConfig",
    "SparseAutoencoder",
    "TunedLens",
    "collect_activations",
    "lens_trajectory",
]
