"""Tests for the representation-based backdoor detectors and their tooling.

The most important test in this file is
:func:`test_scores_do_not_depend_on_batch_composition`. Normalising a score
inside the batch it was computed in destroys comparability across examples and
collapses the pooled ROC to chance — a bug this project has shipped three
separate times in the RAG work package, each time reading as "this detector is
weak" rather than "this detector is broken". It is encoded here as a tripwire.
"""

from __future__ import annotations

import numpy as np
import pytest

from aiforensics.detectors import get_detector
from aiforensics.detectors.feature_space import (
    ActivationClusteringDetector,
    HiddenSvdDetector,
    SpectralSignatureDetector,
    as_matrix,
)
from aiforensics.metrics.detection import roc_auc


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def displaced_pool(seed: int = 0, n_clean: int = 200, n_pois: int = 20, d: int = 64):
    """Clean isotropic cloud plus a tight cluster displaced along one axis.

    This is the geometry a backdoor produces: every triggered input must map to
    the same target regardless of content, so the poisoned representations
    collapse together and move off the clean centroid.
    """
    rng = np.random.default_rng(seed)
    clean = rng.normal(size=(n_clean, d))
    direction = np.zeros(d)
    direction[3] = 6.0
    pois = rng.normal(size=(n_pois, d)) * 0.3 + direction
    X = np.vstack([clean, pois])
    truth = np.zeros(len(X), dtype=bool)
    truth[n_clean:] = True
    return X, truth, clean


# --------------------------------------------------------------------------- #
# the tripwire
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["spectral-signature", "activation-clustering"])
def test_pool_detectors_are_documented_as_pool_level(key) -> None:
    """Spectral / clustering scores are *defined* against the rest of the pool.

    They are not batch-invariant and must not be: the mean that gets subtracted
    and the clusters that get fitted are properties of the pool. This test pins
    that they are honest about it — scoring one example alone is meaningless
    rather than silently wrong — so nobody later "fixes" them into per-example
    detectors.
    """
    X, _truth, _ = displaced_pool()
    det = get_detector(key)
    alone = det.score(X[:1])
    assert alone.shape == (1,)
    # A single example has no pool geometry, so the score is degenerate (zero),
    # not an arbitrary number that would look comparable to a pooled one.
    assert float(alone[0]) == 0.0


def test_scores_do_not_depend_on_batch_composition() -> None:
    """A clean-calibrated detector must score an example identically in any batch.

    ``hidden-svd`` fits its reference on clean data and then scores each example
    against that frozen reference, so adding unrelated examples to the call must
    not move any score. The moment a detector normalises within the call, this
    test fails — which is exactly the point.
    """
    X, _truth, clean = displaced_pool()
    det = HiddenSvdDetector(n_components=8).fit(clean)

    target = X[210:211]
    alone = det.score(target)
    with_others = det.score(np.vstack([target, X[:120]]))[0]
    padded = det.score(np.vstack([X[:37], target, X[50:]]))[37]

    # Tolerance is one part in 1e-12, not bit-identity: BLAS picks different
    # blocking for different matrix shapes, so the same dot product can differ
    # in the last ULP (measured: 1.9e-16 relative). That is floating point, not
    # a leak. A within-batch normalisation bug moves a score by O(1), so this
    # bound separates the two cleanly.
    assert alone[0] == pytest.approx(with_others, rel=1e-12)
    assert alone[0] == pytest.approx(padded, rel=1e-12)


# --------------------------------------------------------------------------- #
# feature-space
# --------------------------------------------------------------------------- #
def test_spectral_signature_separates_a_displaced_cluster() -> None:
    X, truth, _ = displaced_pool()
    labels = np.zeros(len(X), dtype=int)
    scores = SpectralSignatureDetector().score((X, labels))
    assert roc_auc(truth, scores) > 0.95


def test_activation_clustering_uses_ica_and_pca_is_a_different_detector() -> None:
    """The published projection is ICA, and it is measurably WEAKER than PCA here.

    Chen et al. specify ICA. This detector defaulted to PCA on the claim that
    "on these representations the two agree closely"; that claim was never
    checked and is false. Measured on a planted displacement, ROC by projection:

        pool                     PCA     ICA
        balanced 50/50, strong   1.000   0.320
        minority 10%, strong     1.000   0.667
        laplace, minority 10%    0.998   0.767

    Their per-sample scores are even ANTI-correlated in the balanced case
    (r = -0.36). So the default is the published ICA, PCA stays available, and
    this test pins the gap so nobody silently swaps one for the other to make a
    number look better.
    """
    X, truth, _ = displaced_pool()
    labels = np.zeros(len(X), dtype=int)
    ica = roc_auc(truth, ActivationClusteringDetector(seed=0).score((X, labels)))
    pca = roc_auc(
        truth, ActivationClusteringDetector(seed=0, projection="pca").score((X, labels))
    )
    assert ActivationClusteringDetector().config["projection"] == "ica", (
        "the published procedure is ICA; a PCA default is a different detector"
    )
    assert pca > 0.90, "PCA should still separate a planted displacement"
    assert pca - ica > 0.10, (
        "the two projections are supposed to disagree materially here; if they "
        f"have converged (PCA {pca:.3f}, ICA {ica:.3f}) the docstring table is "
        "stale and should be re-measured"
    )


def test_detectors_are_near_chance_when_there_is_no_poison() -> None:
    """The null control. A detector that "finds" structure in clean data is
    reporting its own noise, and would raise alarms on an unpoisoned system —
    the failure this project's headline requirement is built to exclude."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(220, 64))
    labels = np.zeros(len(X), dtype=int)
    fake = np.zeros(220, dtype=bool)
    fake[200:] = True
    # ICA on clean Gaussian noise is ill-posed -- ICA needs non-Gaussian sources
    # -- so its split is arbitrary and can land further from 0.5 than PCA's. The
    # band is per-detector for that reason, and the point of the test is
    # unchanged: neither may find real structure where there is none.
    for det, lo, hi in (
        (SpectralSignatureDetector(), 0.3, 0.7),
        (ActivationClusteringDetector(seed=0), 0.15, 0.85),
        (ActivationClusteringDetector(seed=0, projection="pca"), 0.3, 0.7),
    ):
        assert lo < roc_auc(fake, det.score((X, labels))) < hi


def test_hidden_svd_modes_measure_different_things() -> None:
    """``residual`` sees only off-subspace distance; ``mahalanobis`` sees both.

    Pinned because the distinction decides whether a displacement *along* a
    direction the clean data already spans is visible at all.
    """
    rng = np.random.default_rng(2)
    clean = rng.normal(size=(400, 64))
    fit_on, held = clean[:300], clean[300:]
    off = rng.normal(size=(20, 64))
    off[:, 20:] *= 5.0                      # pushed off the clean manifold
    X = np.vstack([held, off])
    truth = np.zeros(len(X), dtype=bool)
    truth[100:] = True
    for mode in ("residual", "mahalanobis"):
        det = HiddenSvdDetector(mode=mode, n_components=16).fit(fit_on)
        assert roc_auc(truth, det.score(X)) > 0.95


def test_hidden_svd_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        HiddenSvdDetector(mode="whatever")


def test_as_matrix_rejects_unpooled_hidden_states() -> None:
    """A (n, seq, hidden) tensor must be refused, with the fix named.

    Silently flattening it would average over padding positions, which skews
    every covariance and SVD estimate in this module.
    """
    with pytest.raises(ValueError, match="pooled"):
        as_matrix(np.zeros((4, 7, 16)))


def test_as_matrix_rejects_misaligned_labels() -> None:
    with pytest.raises(ValueError, match="do not align"):
        as_matrix((np.zeros((10, 4)), np.zeros(7)))


# --------------------------------------------------------------------------- #
# SAE
# --------------------------------------------------------------------------- #
def _sparse_pool(rng, n, D, extra=None):
    Z = np.zeros((n, D.shape[1]))
    for i in range(n):
        j = rng.choice(D.shape[1], 3, replace=False)
        Z[i, j] = rng.uniform(1, 3, 3)
    X = Z @ D.T
    return (X + extra if extra is not None else X).astype(np.float32)


@pytest.mark.slow
def test_sae_reconstruction_error_needs_a_clean_dictionary() -> None:
    """Training the SAE on the pool it scores *inverts* the detector.

    Measured, not asserted: clean-fit separates the poison (AUC ~1.0) while
    fitting on a contaminated pool scores poisoned examples *lower* than clean
    ones, because the dictionary learned to reconstruct them. A confidently
    wrong detector is worse than a weak one, and nothing about the training run
    looks abnormal — hence this test.
    """
    from aiforensics.interp import SaeConfig, SparseAutoencoder

    rng = np.random.default_rng(0)
    D = rng.normal(size=(64, 40))
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    trig = rng.normal(size=64)
    trig = trig / np.linalg.norm(trig) * 4.0

    clean_fit = _sparse_pool(rng, 3000, D)
    clean_hold = _sparse_pool(rng, 300, D)
    pois = _sparse_pool(rng, 30, D, extra=trig)
    X = np.vstack([clean_hold, pois])
    truth = np.zeros(len(X), dtype=bool)
    truth[300:] = True

    cfg = SaeConfig(expansion=4, k=8, steps=600, batch_size=512)
    clean_sae = SparseAutoencoder(64, cfg).fit(clean_fit, progress=False)
    assert roc_auc(truth, clean_sae.reconstruction_error(X)) > 0.9

    mixed = np.vstack([clean_fit, _sparse_pool(rng, 300, D, extra=trig)])
    dirty_sae = SparseAutoencoder(64, cfg).fit(mixed, progress=False)
    assert roc_auc(truth, dirty_sae.reconstruction_error(X)) < 0.6


def test_sae_refuses_to_score_before_fit() -> None:
    from aiforensics.interp import SparseAutoencoder

    with pytest.raises(RuntimeError, match="before fit"):
        SparseAutoencoder(16).reconstruction_error(np.zeros((2, 16), dtype=np.float32))


# --------------------------------------------------------------------------- #
# threshold calibration
# --------------------------------------------------------------------------- #
def test_calibration_respects_the_alarm_budget_on_a_tied_statistic() -> None:
    """A discrete score must not blow through its alarm budget.

    This is the failure that turned a detector with ROC-AUC 0.840 into a
    reported detection accuracy of 0.000: the statistic was an integer layer
    index, ``np.quantile`` returned a value 30% of clean samples equalled, and
    ``score >= threshold`` flagged all of them.
    """
    from aiforensics.metrics.detection import calibrate_threshold_on_clean

    clean = np.array([26.0] * 70 + [25.0] * 20 + [27.0] * 10)
    naive = float(np.quantile(clean, 0.95))
    assert (clean >= naive).mean() > 0.05          # the bug, pinned

    thr = calibrate_threshold_on_clean(clean, 0.05)
    assert (clean >= thr).mean() <= 0.05           # the contract


def test_calibration_matches_the_quantile_when_scores_are_continuous() -> None:
    """No behaviour change in the ordinary case — only ties are handled."""
    from aiforensics.metrics.detection import calibrate_threshold_on_clean

    rng = np.random.default_rng(0)
    clean = rng.normal(size=5000)
    thr = calibrate_threshold_on_clean(clean, 0.05)
    assert thr == pytest.approx(float(np.quantile(clean, 0.95)), abs=0.01)
    assert (clean >= thr).mean() == pytest.approx(0.05, abs=0.002)


def test_calibration_sits_above_a_degenerate_clean_distribution() -> None:
    """A detector that is identically zero on clean data still gets a threshold.

    No clean score can meet a 5% budget when every clean point ties, but placing
    the threshold just above the clean maximum spends none of the budget and
    still fires on anything higher. Returning ``inf`` instead would discard a
    working detector — measured here as ROC-AUC 1.000 with a reported detection
    accuracy of 0.000.
    """
    from aiforensics.metrics.detection import calibrate_threshold_on_clean

    clean = np.zeros(100)
    thr = calibrate_threshold_on_clean(clean, 0.05)
    assert np.isfinite(thr)
    assert (clean >= thr).mean() == 0.0            # spends none of the budget
    poisoned = np.array([0.4, 1.2, 3.0])
    assert (poisoned >= thr).mean() == 1.0         # and still detects


def test_calibration_rejects_bad_input() -> None:
    from aiforensics.metrics.detection import calibrate_threshold_on_clean

    with pytest.raises(ValueError, match="empty"):
        calibrate_threshold_on_clean(np.array([]), 0.05)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibrate_threshold_on_clean(np.ones(10), 1.5)


# --------------------------------------------------------------------------- #
# trigger tables
# --------------------------------------------------------------------------- #
def test_trigger_tables_join_by_the_right_shape() -> None:
    """Both released table shapes must join, and anything else must raise.

    The SCPN test table is 912 rows against a 1,821-row clean split — the
    non-target rows only. Falling back to a text lookup in the *train* table
    instead left an evaluation pool of a single example, and the run still
    printed "ASR 1.000 ... FIRED".
    """
    pytest.importorskip("datasets")
    from aiforensics.attacks.textual_backdoor import SST2_SENTENCE_DIR, build_trigger

    if not (SST2_SENTENCE_DIR / "clean" / "test.tsv").exists():
        pytest.skip("vendored splits absent; run scripts/fetch_backdoor_data.py")

    from aiforensics.attacks.textual_backdoor import load_sst2_sentence

    te, _labels = load_sst2_sentence("test")
    for kind, floor in (("syntactic", 800), ("style", 1500)):
        spec = build_trigger(kind, te, split="test", target_label=1)
        carryable = int(spec.available.sum())
        assert carryable >= floor, (
            f"{kind} test table joined to only {carryable} rows; a broken join "
            "does not crash, it leaves nothing to measure"
        )
        # every carryable row must actually change under the trigger
        idx = int(np.flatnonzero(spec.available)[0])
        assert spec.fn(te[idx]) != te[idx]


def test_syntactic_is_capped_by_its_published_table() -> None:
    """The cap is a measured property of the released data, not an assumption."""
    from aiforensics.attacks.textual_backdoor import SST2_SENTENCE_DIR, build_trigger

    if not (SST2_SENTENCE_DIR / "clean" / "train.tsv").exists():
        pytest.skip("vendored splits absent; run scripts/fetch_backdoor_data.py")
    from aiforensics.attacks.textual_backdoor import load_sst2_sentence

    tr, y = load_sst2_sentence("train")
    syn = build_trigger("syntactic", tr, split="train", target_label=1)
    word = build_trigger("word", tr, split="train", target_label=1)
    assert syn.max_rate(y, 1) < 0.11 < word.max_rate(y, 1), (
        "syntactic must be capped well below the other classes; a controlled "
        "four-way comparison has to live at or under that cap"
    )


def test_activation_clustering_minority_rule_is_base_rate_dependent() -> None:
    """The published rule inverts when the poison is not a minority.

    Chen et al. score the *smaller* of two clusters, which is sound at training
    poison rates but not in an evaluation pool balanced between triggered and
    clean inputs — there the smaller cluster is decided by a handful of examples.
    Measured: ROC-AUC ~1.00 with 20 poisoned among 200 clean, and ~0.00 at
    250 versus 250, on the same generator. The symmetric variant privileges
    neither cluster and holds in both.

    Pinned because the failure is silent and directional: it does not error, it
    reports a confident number with the sign flipped.
    """
    from aiforensics.detectors.feature_space import ActivationClusteringDetector

    def pool(n_clean, n_pois, seed, spread):
        r = np.random.default_rng(seed)
        clean = r.normal(size=(n_clean, 64))
        direction = np.zeros(64)
        direction[3] = 5.0
        pois = r.normal(size=(n_pois, 64)) * spread + direction
        X = np.vstack([clean, pois])
        truth = np.zeros(len(X), dtype=bool)
        truth[n_clean:] = True
        return X, np.zeros(len(X), dtype=int), truth

    # The projection is pinned to PCA here because the numbers this test records
    # (~1.00 minority, ~0.00 balanced) were measured with it, and the claim under
    # test is about the MINORITY RULE, not about the projection. ICA -- the
    # published default since the fidelity audit -- shows the same inversion at
    # a smaller magnitude, asserted below, so the finding does not depend on it.
    def ac(**kw):
        return ActivationClusteringDetector(seed=0, projection="pca", **kw)

    X, y, truth = pool(200, 20, 0, 0.3)
    assert roc_auc(truth, ac().score((X, y))) > 0.9

    X, y, truth = pool(250, 250, 0, 0.5)
    published = roc_auc(truth, ac().score((X, y)))
    symmetric = roc_auc(truth, ac(symmetric=True).score((X, y)))
    assert published < 0.2, "the minority rule is expected to invert here"
    assert symmetric > 0.8, "the symmetric rule should survive a balanced pool"

    # The published ICA projection, with restart averaging, does not merely
    # invert on a balanced pool -- it is BELOW CHANCE in both regimes on this
    # generator: ~0.12 with 20 poisoned among 200 clean, ~0.00 at 250 vs 250.
    # So the minority-rule inversion demonstrated above with PCA is not the only
    # problem; under the projection the paper actually specifies, this detector
    # does not separate a planted displacement here at all. Recorded rather than
    # asserted away, because it bears directly on how the row should be read.
    X_m, y_m, t_m = pool(200, 20, 0, 0.3)
    ica_minority = roc_auc(
        t_m, ActivationClusteringDetector(seed=0).score((X_m, y_m))
    )
    ica_balanced = roc_auc(truth, ActivationClusteringDetector(seed=0).score((X, y)))
    assert ica_minority < 0.5 and ica_balanced < 0.5, (
        f"ICA is expected below chance in both regimes here (minority "
        f"{ica_minority:.3f}, balanced {ica_balanced:.3f}); if it has risen "
        "above 0.5 the docstring's reading of this detector is stale"
    )
    assert roc_auc(truth, ac().score((X, y))) < ica_minority + 0.6, (
        "sanity: the PCA balanced number should still be the inverted one"
    )


def test_hidden_svd_floors_scales_at_the_out_of_fold_noise_level() -> None:
    """Whitening past the supported rank makes every held-out point score huge.

    ``min(n_components, *shape)`` is not a sufficient cap: with 150 rows in 2048
    dimensions it resolves to 150, retaining directions whose singular value is
    estimated from almost no degrees of freedom. Dividing by those is dividing
    by noise, and the symptom is not a weak detector — it is a detector that
    fires on everything, which reads as a saturated control cell rather than as
    a broken fit.

    Two distinct pathologies produced that, and both are asserted here.

    * The whitening scale of a trailing direction. Centring n rows leaves rank
      n-1, so the last retained direction has variance exactly 0; dividing a
      held-out projection by it sent the in-subspace term to ~1e24.
    * The off-subspace scale. A rank-n basis spans n fit points exactly, so the
      in-sample residual is ~0 and dividing by it inflated the off-subspace term
      by ~1e13, drowning the in-subspace term entirely.

    Capping the rank suppresses both symptoms but is the wrong fix: it discards
    the low-variance directions the signal lives in on a broad pool, which took
    the context-extraction objective from 0.830 to 0.090. The rank is kept and
    both scales are instead floored at the out-of-fold noise level.
    """
    rng = np.random.default_rng(0)
    d = 2048

    det = HiddenSvdDetector(n_components=256).fit(rng.normal(size=(150, d)))
    assert det.config["retained_rank"] == 150, (
        "the rank must NOT be capped -- that is what lost the broad-pool signal"
    )
    assert 0.8 < det.config["noise_sd"] < 1.2, (
        "out-of-fold noise level should recover the true per-dimension sd of 1.0; "
        f"got {det.config['noise_sd']}"
    )

    # Calibration and evaluation on rows the basis never saw, which is the
    # protocol the runner uses. Thresholding in sample hides this bug entirely.
    fit = rng.normal(size=(150, d))
    calib = rng.normal(size=(200, d))
    held = rng.normal(size=(200, d))
    det = HiddenSvdDetector(n_components=256).fit(fit)

    assert np.isclose(det.score(held).mean(), np.sqrt(d), rtol=0.15), (
        "on isotropic clean data the two terms should sum to the dimension, so "
        f"the score sits at sqrt(d)={np.sqrt(d):.0f}; got {det.score(held).mean():.3g}"
    )

    thr = float(np.quantile(det.score(calib), 0.95))
    assert (det.score(held) >= thr).mean() < 0.20, (
        "a clean held-out set is flagged far above the 5% alarm budget; a scale "
        "is being estimated as zero and the whitening is dividing by noise again"
    )

    # ...and it must still have power. A shift well above the noise level in a
    # single coordinate is exactly what a rank cap would have discarded.
    shift = np.zeros(d)
    shift[300] = 30.0
    assert (det.score(rng.normal(size=(200, d)) + shift) >= thr).mean() > 0.9, (
        "a 30-sigma single-coordinate shift must be detected"
    )
