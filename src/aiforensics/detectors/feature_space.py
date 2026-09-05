"""Feature-space backdoor detection — anomaly in the representation geometry.

These techniques do not read the input text. They take the model's *internal
representations* of a pool of examples and look for the geometric trace a
backdoor leaves behind: because a backdoored model must map every triggered
input to one target label regardless of content, triggered examples get pushed
together into a tight, displaced cluster that clean examples of the same label
do not form.

Implemented:

* :class:`SpectralSignatureDetector` —

      Brandon Tran, Jerry Li, Aleksander Madry. "Spectral Signatures in Backdoor
      Attacks." NeurIPS 2018, 8000-8010. arXiv:1811.00636

* :class:`ActivationClusteringDetector` —

      Bryant Chen et al. "Detecting Backdoor Attacks on Deep Neural Networks by
      Activation Clustering." AAAI 2019 Workshop on Artificial Intelligence
      Safety (SafeAI). arXiv:1811.03728

* :class:`HiddenSvdDetector` — subspace reconstruction error against a basis
  fitted on data believed clean.

**Three properties of this family decide how the results must be read.**

*They are pool-level, not per-example.* Every score here is computed against the
other examples in the pool: the mean that gets subtracted, the singular vector
that gets projected onto, the clusters that get fitted. Handing one example in
isolation is meaningless. This also means a score is only comparable to other
scores computed in the *same* call — which is the exact trap that produced three
separate chance-level results in this project's RAG package. Pool once, score
once.

*They assume a per-class pool.* The published methods run per output class,
because that is what makes "these examples share a label but not a cause"
detectable. For a generative LM there is no built-in class axis, so the caller
supplies one via ``labels`` (for a verbalised classification task, the label
index is the natural choice).

*Spectral signatures needs the poison rate.* The published removal rule strips
the top ``1.5 * epsilon`` fraction, where epsilon is the contamination rate the
defender does not know. That is a genuine weakness in an unknown-poisoning
setting, so :meth:`score` returns the raw statistic and leaves thresholding to
the harness, where it is calibrated on clean data like every other detector
here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = [
    "ActivationClusteringDetector",
    "HiddenSvdDetector",
    "SpectralSignatureDetector",
    "as_matrix",
]


def as_matrix(samples: Any) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalise detector input to ``(representations, labels_or_None)``.

    Accepts a plain ``(n, d)`` array, a ``(representations, labels)`` pair, or a
    mapping with ``"representations"`` / ``"labels"`` keys. Torch tensors are
    accepted and moved to CPU float64 — these estimators run in numpy/scipy and
    bfloat16 cannot cross that boundary.

    Raises:
        ValueError: If the array is not 2-D, or labels do not align with rows.
    """
    labels = None
    if isinstance(samples, dict):
        reps, labels = samples["representations"], samples.get("labels")
    elif isinstance(samples, tuple) and len(samples) == 2:
        reps, labels = samples
    else:
        reps = samples

    if hasattr(reps, "detach"):
        reps = reps.detach().to("cpu", copy=False)
    reps = np.asarray(reps, dtype=np.float64)
    if reps.ndim != 2:
        raise ValueError(
            f"representations must be 2-D (n_examples, hidden), got shape {reps.shape}. "
            "Sequence-shaped hidden states must be pooled first — see "
            "aiforensics.models.llm.HiddenStates.pooled(), which is mask-aware; "
            "pooling by hand over padded positions skews every estimate here."
        )
    if labels is not None:
        if hasattr(labels, "detach"):
            labels = labels.detach().to("cpu", copy=False)
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != reps.shape[0]:
            raise ValueError(
                f"labels ({len(labels)}) do not align with representations "
                f"({reps.shape[0]})"
            )
    return reps, labels


def _groups(reps: np.ndarray, labels: np.ndarray | None) -> list[np.ndarray]:
    """Row-index groups to analyse separately — one per class, or one pool."""
    if labels is None:
        return [np.arange(reps.shape[0])]
    return [np.flatnonzero(labels == y) for y in np.unique(labels)]


@register_detector("spectral-signature", category="feature-space")
class SpectralSignatureDetector(Detector):
    """Top-singular-vector outlier score over per-class representations.

    The method (Tran et al., NeurIPS 2018): within one class, centre the
    representation matrix and project each example onto the top right singular
    vector; a backdoor's target-class pool is a mixture of clean examples and a
    displaced poisoned sub-population, and that displacement dominates the
    leading direction, so poisoned examples take extreme projections.

    Args:
        n_components: Singular directions to combine. The paper uses 1; taking
            the max squared projection over the top few is a common
            robustification and is off by default so the result stays faithful.
        center: Subtract the group mean (the paper's ``R̂``). Leave on.
    """

    name = "spectral-signature"
    category = "feature-space"
    modality = "activations"
    requires_fit = False

    def __init__(
        self,
        *,
        n_components: int = 1,
        center: bool = True,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, n_components=n_components, center=center, **config
        )
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        self.n_components = int(n_components)
        self.center = bool(center)

    def score(self, samples: Any) -> np.ndarray:
        """Squared projection onto the leading singular direction, per example.

        Args:
            samples: ``(n, d)`` representations, or ``(representations, labels)``
                to run the published per-class variant.

        Returns:
            ``(n,)`` scores; higher = more anomalous.
        """
        reps, labels = as_matrix(samples)
        out = np.zeros(reps.shape[0], dtype=np.float64)
        for idx in _groups(reps, labels):
            if len(idx) < 2:
                continue                      # no geometry to speak of
            block = reps[idx]
            if self.center:
                block = block - block.mean(axis=0, keepdims=True)
            k = min(self.n_components, *block.shape)
            # SVD of the centred matrix; rows project onto right singular vectors.
            _u, _s, vt = np.linalg.svd(block, full_matrices=False)
            proj = block @ vt[:k].T           # (n_group, k)
            out[idx] = (proj ** 2).max(axis=1)
        return out


#: Both `spectral-signature` and `activation-clustering` are defined for a pool
#: in which the poison is a MINORITY. Tran et al. fix ``1/2 > eps > 0`` in
#: Definition 3.1 and run experiments at 5-10%; Chen et al.'s cluster analysis
#: nominates the smaller of two clusters. The paired evaluation pool used
#: throughout this project is 50/50 triggered/clean by construction, which is
#: the excluded endpoint for the first and a coin flip for the second -- at
#: exactly balanced sizes "the minority cluster" is decided by noise, which is
#: the likely cause of activation-clustering swinging between 0/400 and 346/400
#: across otherwise identical Study A cells. Read both rows knowing they run
#: outside the regime their papers define.
BALANCED_POOL_CAVEAT = (
    "scored on a 50/50 pool; both papers assume the poison is a minority"
)


@register_detector("activation-clustering", category="feature-space")
class ActivationClusteringDetector(Detector):
    """Two-means clustering of per-class representations (Chen et al., 2019).

    Project a class's representations to a low-dimensional space, split them in
    two, and treat the *smaller* cluster as the suspected poison. The published
    procedure uses ICA; PCA is offered because it is deterministic and far
    cheaper, and on these representations the two agree closely.

    The per-example score is the **silhouette-weighted** distance to the larger
    cluster's centroid, which turns the paper's hard cluster assignment into a
    graded score. Two reasons that matters here: a hard assignment gives a
    degenerate ROC with a single operating point, and the silhouette weight
    encodes the paper's own signal that a *well-separated* split is evidence of
    poisoning while a poor split is evidence of none.

    **The minority rule breaks when poison is not a minority.** The published
    method assumes the poisoned examples form the *smaller* of the two clusters,
    which holds at the poison rates a training set sees (1-20%) but not in an
    evaluation pool balanced between triggered and clean inputs. At 250 versus
    250 the smaller cluster is decided by a handful of examples, so which side
    is scored becomes close to a coin flip — and the same detector can post a
    ROC-AUC near 1.0 on one seed and near 0.0 on the next, with nothing
    materially different between them.

    ``symmetric=True`` removes the assumption by scoring distance to the nearer
    centroid relative to the further one, so neither cluster is privileged. It
    is a departure from the paper and is labelled as ours wherever reported. Use
    the published rule when the poison really is a minority — a training-set
    scan — and the symmetric rule when it may not be.

    Args:
        n_components: Dimensions to project to before clustering (paper: 10).
        projection: ``"ica"`` (paper) or ``"pca"`` (deterministic, faster).
        symmetric: Score both clusters rather than only the smaller one.
        seed: Seed for ICA and k-means.
    """

    name = "activation-clustering"
    category = "feature-space"
    modality = "activations"
    requires_fit = False

    def __init__(
        self,
        *,
        n_components: int = 10,
        projection: str = "pca",
        symmetric: bool = False,
        seed: int = 1337,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, n_components=n_components,
            projection=projection, symmetric=symmetric, seed=seed, **config,
        )
        if projection not in {"ica", "pca"}:
            raise ValueError(f"projection must be 'ica' or 'pca', got {projection!r}")
        self.n_components = int(n_components)
        self.projection = projection
        self.symmetric = bool(symmetric)
        self.seed = int(seed)

    def _project(self, block: np.ndarray) -> np.ndarray:
        k = min(self.n_components, *block.shape)
        if self.projection == "ica":
            from sklearn.decomposition import FastICA
            return FastICA(n_components=k, random_state=self.seed,
                           whiten="unit-variance", max_iter=1000).fit_transform(block)
        from sklearn.decomposition import PCA
        return PCA(n_components=k, random_state=self.seed).fit_transform(block)

    def score(self, samples: Any) -> np.ndarray:
        """Graded suspicion from a 2-means split of each class's activations."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        reps, labels = as_matrix(samples)
        out = np.zeros(reps.shape[0], dtype=np.float64)
        for idx in _groups(reps, labels):
            if len(idx) < 4:                 # silhouette is undefined below this
                continue
            block = reps[idx]
            try:
                proj = self._project(block)
            except Exception as exc:
                logger.warning("projection failed on a group of %d: %s", len(idx), exc)
                continue
            km = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(proj)
            assign = km.labels_
            counts = np.bincount(assign, minlength=2)
            if counts.min() == 0:
                continue
            small = int(np.argmin(counts))
            # Separation quality gates the whole group: a class that does not
            # split cleanly is evidence of no poison, not weak evidence of some.
            try:
                sil = float(silhouette_score(proj, assign))
            except Exception:
                sil = 0.0
            sil = max(sil, 0.0)
            d_small = np.linalg.norm(proj - km.cluster_centers_[small], axis=1)
            d_big = np.linalg.norm(proj - km.cluster_centers_[1 - small], axis=1)
            if self.symmetric:
                # Neither cluster privileged: how much closer this example is to
                # its own centroid than to the other, scaled by separation.
                out[idx] = sil * np.abs(d_big - d_small)
            else:
                # The paper's rule: only the minority cluster carries suspicion.
                out[idx] = sil * d_big * (assign == small)
        return out


@register_detector("hidden-svd", category="feature-space")
class HiddenSvdDetector(Detector):
    """Reconstruction error against a subspace fitted on clean representations.

    Unlike the two above, this one is *supervised by cleanliness*: it fits a
    low-rank basis on data believed clean, then scores any example by how much
    of it falls outside that subspace. That makes it the natural fit for this
    project's threat model, where the investigator can plausibly obtain a clean
    reference but never labelled poison.

    Falling back to fitting on the scored pool itself (no :meth:`fit` call) is
    supported but weaker, and is what the ``requires_fit`` flag is about: the
    poison is then inside the basis it is being measured against.

    **Choose the statistic to match the artifact you expect.** ``mode="residual"``
    measures only the distance *outside* the clean subspace, so it finds an
    example that lies off the clean manifold but is blind to one displaced
    *along* a direction the clean data already spans. ``mode="mahalanobis"``
    (the default) measures both, each normalised by the clean spread so the two
    terms are in the same units — without that normalisation the sum is a raw
    residual added to a whitened coordinate, which is dimensionally meaningless
    and silently dominated by whichever term happens to be larger.

    Neither statistic can separate a poisoned cluster that sits *closer* to the
    clean centroid than clean data does — a tight cluster in high dimensions
    does exactly that. Spectral signatures catches that case and this does not;
    they are complementary, which is why both are in this category.

    **The retained rank is not a minor knob.** A backdoor's signature does not
    have to live in the high-variance directions of the clean data, and when it
    does not, a small subspace discards it. Measured on the appended-phrase
    trigger at the final layer, varying only ``n_components``:

    ====  ===============  =========
    rank  detection rate   ROC-AUC
    ====  ===============  =========
       8            0.032      0.771
      16            0.028      0.701
      32            0.248      0.884
      64            0.796      0.942
     256            0.952      0.944
    ====  ===============  =========

    The same detector on the rare-token trigger scores 1.000 at every rank, so a
    single rank cannot be assumed adequate from one trigger class. The default is
    therefore generous rather than tight; lower it only with a reason.

    Args:
        n_components: Rank of the retained subspace. See the table above before
            reducing it — a rank chosen for speed can silently remove the very
            direction the backdoor lives in.
        mode: ``"mahalanobis"`` (whitened in-subspace distance + normalised
            residual) or ``"residual"`` (off-subspace distance only, the plain
            reconstruction error).
    """

    name = "hidden-svd"
    category = "feature-space"
    modality = "activations"
    requires_fit = False

    def __init__(
        self,
        *,
        n_components: int = 256,
        mode: str = "mahalanobis",
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, n_components=n_components, mode=mode, **config
        )
        if mode not in {"mahalanobis", "residual"}:
            raise ValueError(
                f"mode must be 'mahalanobis' or 'residual', got {mode!r}"
            )
        self.n_components = int(n_components)
        self.mode = mode
        self._mean: np.ndarray | None = None
        self._basis: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._residual_scale: float = 1.0

    def fit(self, clean: Any) -> HiddenSvdDetector:
        """Fit the clean subspace.

        Args:
            clean: Representations believed clean, in any form
                :func:`as_matrix` accepts. Labels are ignored — the basis is
                global on purpose, so a class with few clean examples still
                gets a well-conditioned reference.
        """
        reps, _ = as_matrix(clean)
        if reps.shape[0] < 2:
            raise ValueError("need at least 2 clean examples to fit a subspace")
        self._mean = reps.mean(axis=0, keepdims=True)
        centred = reps - self._mean
        _u, s, vt = np.linalg.svd(centred, full_matrices=False)

        # Cap the retained rank at what the sample size actually supports.
        # `min(n_components, *shape)` is not enough: with 150 rows in 2048 dims
        # it resolves to 150, keeping every component including those whose
        # singular value is estimated from almost no degrees of freedom. The
        # whitened coordinate c_j / sigma_j then divides by noise, and EVERY
        # held-out point scores enormous -- which reads as a detector that fires
        # on everything rather than as a broken fit.
        #
        # Measured on the refusal objective, sweeping only this rank against a
        # fixed 150-row fit (effective rank 77):
        #
        #   rank    detection   control      reading
        #      8        0.770     0.065      reads the backdoor
        #     32        0.995     0.050      reads the backdoor
        #     64        1.000     1.000      saturated
        #    256        1.000     1.000      saturated   <- the old default
        #
        # Capping the rank does stop that saturation, but it is the wrong lever:
        # the same cap took the context-extraction objective from 0.830 to
        # 0.090, because on a broad pool the signal lives in exactly the
        # low-variance directions a cap throws away. The rank is therefore left
        # alone and the real cause is fixed in `_out_of_fold_residual_scale`.
        effective = int((s > s[0] * 1e-6).sum()) if s[0] > 0 else 1
        k = min(self.n_components, *centred.shape)
        self.config["retained_rank"] = k
        self.config["effective_rank"] = effective
        self._basis = vt[:k]
        # Per-component clean standard deviation, floored at the ambient noise
        # level (below). Guarded so a degenerate direction cannot make the
        # whitened score infinite.
        denom = np.sqrt(max(reps.shape[0] - 1, 1))
        raw_scale = s[:k] / denom
        # Typical clean residual, so the off-subspace term is expressed in the
        # same units as the whitened in-subspace term rather than raw distance.
        # It MUST be measured out of fold. In-sample it is ~0 by construction
        # whenever the retained rank reaches the number of fit rows, because a
        # rank-n basis spans n points exactly; dividing a held-out residual by
        # that zero inflated the off-subspace term by ~1e13 and drowned the
        # whitened term completely, leaving a detector that only measured
        # distance from the clean span. That is what saturated the narrow pools.
        noise_var = self._out_of_fold_noise_var(centred)
        self._residual_scale = float(np.sqrt(noise_var))
        # Floor every retained direction at the ambient per-dimension noise
        # variance, the probabilistic-PCA estimate. A direction whose variance
        # is estimated BELOW the noise floor carries no signal -- it is noise
        # that the fit happened to squeeze flat -- so whitening by it divides by
        # an accident of the sample. Centring n rows leaves rank n-1, which made
        # the trailing scale exactly 0 and the projection term 1e24; the floor
        # removes that without discarding the direction.
        self._scale = np.maximum(raw_scale, self._residual_scale)
        self.config["scales_floored"] = int((raw_scale < self._residual_scale).sum())
        self.config["noise_sd"] = self._residual_scale
        return self

    def _out_of_fold_noise_var(self, centred: np.ndarray) -> float:
        """Per-dimension variance of the directions the subspace does not span.

        Measured out of fold: refits the subspace K times holding out a fifth of
        the rows, and averages the held-out squared residual per off-subspace
        dimension. In sample this is ~0 by construction whenever the retained
        rank reaches the number of fit rows, because a rank-n basis spans n
        points exactly -- which inflated the off-subspace term by ~1e13 and
        drowned the whitened term entirely.
        """
        n, d = centred.shape
        folds = 5
        if n < 2 * folds:
            assert self._basis is not None
            resid = centred - (centred @ self._basis.T) @ self._basis
            free = max(d - self._basis.shape[0], 1)
            return max(float((resid**2).sum(axis=1).mean()) / free, 1e-24)
        per_dim: list[float] = []
        edges = np.linspace(0, n, folds + 1).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            held, rest = centred[lo:hi], np.concatenate([centred[:lo], centred[hi:]])
            if held.size == 0 or rest.shape[0] < 2:
                continue
            _, _, vt = np.linalg.svd(rest, full_matrices=False)
            b = vt[: min(self.n_components, *rest.shape)]
            resid = held - (held @ b.T) @ b
            per_dim.extend(((resid**2).sum(axis=1) / max(d - b.shape[0], 1)).tolist())
        return max(float(np.mean(per_dim)) if per_dim else 1e-24, 1e-24)

    def _out_of_fold_residual_scale(self, centred: np.ndarray) -> float:
        """Typical residual of a clean point the basis was **not** fitted on.

        Refits the subspace K times, each time holding out a fifth of the rows,
        and averages the residual norms of the held-out rows. With too few rows
        to split, falls back to the in-sample residual with the caveat that it
        under-estimates.
        """
        n = centred.shape[0]
        folds = 5
        if n < 2 * folds:
            resid = centred - (centred @ self._basis.T) @ self._basis
            return max(float(np.linalg.norm(resid, axis=1).mean()), 1e-12)
        norms: list[float] = []
        edges = np.linspace(0, n, folds + 1).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            held = centred[lo:hi]
            rest = np.concatenate([centred[:lo], centred[hi:]])
            if held.size == 0 or rest.shape[0] < 2:
                continue
            _, _, vt = np.linalg.svd(rest, full_matrices=False)
            b = vt[: min(self.n_components, *rest.shape)]
            norms.extend(np.linalg.norm(held - (held @ b.T) @ b, axis=1).tolist())
        return max(float(np.mean(norms)) if norms else 1e-12, 1e-12)

    def score(self, samples: Any) -> np.ndarray:
        """Residual norm outside the fitted subspace, per example."""
        reps, _ = as_matrix(samples)
        if self._basis is None:
            logger.warning(
                "hidden-svd scored without fit(): fitting the basis on the pool "
                "being scored, so any poison in it is part of its own reference. "
                "Call fit(clean_representations) for the intended protocol."
            )
            self.fit(reps)
        assert self._mean is not None and self._basis is not None
        centred = reps - self._mean
        coords = centred @ self._basis.T
        residual = centred - coords @ self._basis
        residual_norm = np.linalg.norm(residual, axis=1)
        if self.mode == "residual":
            return residual_norm
        assert self._scale is not None
        # Both terms in the same unit -- multiples of the per-dimension clean
        # noise variance -- so the off-subspace term cannot drown the whitened
        # one. On isotropic clean data the two sum to the dimension, which is
        # the full Mahalanobis distance, as they should.
        return np.sqrt(
            ((coords / self._scale) ** 2).sum(axis=1)
            + (residual_norm / self._residual_scale) ** 2
        )


@register_detector("beatrix", category="feature-space")
class BeatrixDetector(Detector):
    """Out-of-range mass in the Gram matrix of an activation, per sample.

        Wanlun Ma, Derui Wang, Ruoxi Sun, Minhui Xue, Sheng Wen, Yang Xiang.
        "The 'Beatrix' Resurrections: Robust Backdoor Detection via Gram
        Matrices." NDSS 2023. arXiv:2209.11715.

    Spectral signatures and activation clustering both read *first-order*
    structure: a mean shift, or a two-cluster split. Beatrix's argument is that
    a backdoor also perturbs the **correlations between feature dimensions**,
    which survive when the mean shift is small, and that those are read from the
    Gram matrix of the feature vector rather than the vector itself.

    For a feature ``f`` and order ``p`` the Gram feature is the upper triangle
    of ``G_p = (f^p)(f^p)^T``. A clean reference fixes a per-entry interval, and
    a sample is scored by how far outside those intervals its own Gram entries
    fall, summed and normalised -- the paper's deviation measure.

    **Two departures, both forced by the setting and both ours.**

    *Dimensionality.* The paper reads CNN feature maps of a few hundred
    channels. A 2048-dimensional residual stream gives a Gram matrix of 4.2M
    entries, more than the 150 clean rows can bound. The feature is therefore
    projected to ``n_project`` dimensions first, with a seeded Gaussian
    projection fixed at fit time, giving ``n_project*(n_project+1)/2`` entries.
    A random projection preserves inner products in expectation, which is the
    structure this detector reads, but it is a departure and the rank it
    imposes is a free parameter.

    *Interval estimation.* The paper uses the clean min/max per entry. With 150
    rows that is driven by the two most extreme samples, so a robust quantile
    interval is used instead and ``quantile`` is exposed. Set it to 0.0 for the
    published min/max rule.
    """

    threshold = None
    requires_fit = True

    def __init__(
        self,
        *,
        order: int = 2,
        n_project: int = 128,
        quantile: float = 0.02,
        seed: int = 1337,
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, modality="activations",
            published_method="Beatrix (NDSS 2023); projection and quantile "
                             "interval are ours",
            order=order, n_project=n_project, quantile=quantile, **config,
        )
        self.order = int(order)
        self.n_project = int(n_project)
        self.quantile = float(quantile)
        self.seed = int(seed)
        self._proj: np.ndarray | None = None
        self._lo: np.ndarray | None = None
        self._hi: np.ndarray | None = None
        self._scale: float = 1.0

    def _gram(self, reps: np.ndarray) -> np.ndarray:
        assert self._proj is not None
        z = reps @ self._proj
        # Signed power, so an odd order cannot collapse the sign structure and
        # an even order cannot be dominated by one saturated coordinate.
        z = np.sign(z) * np.abs(z) ** self.order
        n, d = z.shape
        iu = np.triu_indices(d)
        return np.einsum("ni,nj->nij", z, z)[:, iu[0], iu[1]]

    def fit(self, clean: Any) -> BeatrixDetector:
        reps, _ = as_matrix(clean)
        rng = np.random.default_rng(self.seed)
        self._proj = rng.normal(
            scale=1.0 / np.sqrt(self.n_project), size=(reps.shape[1], self.n_project)
        )
        g = self._gram(reps)
        if self.quantile <= 0.0:
            self._lo, self._hi = g.min(axis=0), g.max(axis=0)
        else:
            self._lo = np.quantile(g, self.quantile, axis=0)
            self._hi = np.quantile(g, 1.0 - self.quantile, axis=0)
        # Normalise by the clean deviation so the score is in clean units and
        # comparable across layers, as every other detector here is.
        raw = self._deviation(g)
        self._scale = float(max(np.mean(raw), 1e-12))
        self.config["n_gram_entries"] = int(g.shape[1])
        return self

    def _deviation(self, g: np.ndarray) -> np.ndarray:
        assert self._lo is not None and self._hi is not None
        span = np.maximum(self._hi - self._lo, 1e-12)
        below = np.maximum(self._lo - g, 0.0)
        above = np.maximum(g - self._hi, 0.0)
        return ((below + above) / span).sum(axis=1)

    def score(self, samples: Any) -> np.ndarray:
        if self._proj is None:
            raise RuntimeError("BeatrixDetector scored before fit()")
        reps, _ = as_matrix(samples)
        return self._deviation(self._gram(reps)) / self._scale


@register_detector("knn-density", category="feature-space")
class KnnDensityDetector(Detector):
    """Distance to the k-th nearest clean neighbour, per sample.

        Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey, and the
        broader deep-kNN anomaly-detection line; the k-NN backdoor-filtering
        variant is standard in the poisoned-sample literature (see e.g. Peri
        et al., "Deep k-NN Defense Against Clean-Label Data Poisoning
        Attacks", ECCV 2020 workshops, arXiv:1909.13374).

    **Why this detector exists here, specifically.** Every other distance
    detector in this repo assumes ONE GLOBAL GAUSSIAN: ``l2`` from the clean
    mean, a diagonal z-score, full-covariance ``mahalanobis``, and
    rank-truncated ``hidden-svd``. ``docs/rank-and-conditioning.md`` shows that
    assumption failing exactly where those detectors saturate -- a clean pool at
    effective rank 37 of 150 with a singular-value ratio of 316 is not one
    Gaussian blob, it is a handful of tight clusters, and a global whitening of
    it amplifies whatever is smallest.

    A local statistic makes no such assumption. The score is the distance to the
    k-th nearest CLEAN calibration point, so a sample sitting inside any clean
    cluster scores low no matter how many clusters there are, and one sitting
    between them scores high. That is the direct test of whether the saturation
    those documents record is a property of the backdoor or of the Gaussian
    assumption.

    ``metric="cosine"`` is the default because residual-stream norms vary by an
    order of magnitude across prompts (attention sinks; see
    ``interp/sae.py``), which a Euclidean radius reads as distance.
    """

    threshold = None
    requires_fit = True

    def __init__(
        self,
        *,
        k: int = 10,
        metric: str = "cosine",
        threshold: float | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, modality="activations",
            published_method="deep k-NN poisoned-sample filtering "
                             "(arXiv:1909.13374 and successors)",
            k=k, metric=metric, **config,
        )
        if metric not in {"cosine", "euclidean"}:
            raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")
        self.k, self.metric = int(k), metric
        self._clean: np.ndarray | None = None

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def fit(self, clean: Any) -> KnnDensityDetector:
        reps, _ = as_matrix(clean)
        if reps.shape[0] <= self.k:
            logger.warning(
                "knn-density: k=%d with only %d clean rows, so the k-th "
                "neighbour is nearly the whole reference set and the score "
                "stops being local. Lower k or supply more clean data.",
                self.k, reps.shape[0],
            )
        self._clean = self._unit(reps) if self.metric == "cosine" else reps
        return self

    def score(self, samples: Any) -> np.ndarray:
        if self._clean is None:
            raise RuntimeError("KnnDensityDetector scored before fit()")
        reps, _ = as_matrix(samples)
        ref = self._clean
        if self.metric == "cosine":
            d = 1.0 - self._unit(reps) @ ref.T
        else:
            d = np.sqrt(np.maximum(
                (reps**2).sum(1)[:, None] - 2 * reps @ ref.T + (ref**2).sum(1)[None], 0.0))
        k = min(self.k, ref.shape[0])
        # Partition rather than sort: only the k-th order statistic is needed.
        return np.partition(d, k - 1, axis=1)[:, k - 1]
