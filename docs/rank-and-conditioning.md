# When a distance detector stops reading the backdoor and starts reading the trigger

`hidden-svd` and `mahalanobis` both reported **detection 1.000 at control
1.000** on some objectives: every triggered input flagged on the poisoned model,
and every triggered input flagged on a model with **no backdoor in it**, at
2.5% false positives on clean inputs. That is a perfect trigger-*text* detector.
It says nothing about the model, and the detection rate alone cannot tell you so
— only the control arm can.

This note records what causes it, because the first two fixes were wrong.

## Two scales that were being estimated as zero

The saturation was partly a numerical fault, and it had two independent causes.

1. **The trailing whitening scale.** Centring `n` rows leaves rank `n-1`, so the
   last retained direction has variance exactly 0. A held-out projection divided
   by it sent the in-subspace term to ~1e24.
2. **The off-subspace scale.** A rank-`n` basis spans `n` fit points *exactly*,
   so the in-sample residual is ~0. Dividing a held-out residual by it inflated
   the off-subspace term by ~1e13 and drowned the whitened term, leaving a
   detector that only measured distance from the clean span.

Both are now floored at the out-of-fold per-dimension noise level, the
probabilistic-PCA estimate. On isotropic clean data the two terms sum to the
dimension — the full Mahalanobis distance — the score sits at `sqrt(d)`, a
held-out clean set is flagged at the 5% budget rather than 100%, and power is
monotone in the shift size instead of flat at chance.

## Capping the rank is the wrong fix

The obvious lever is to retain fewer components. It does suppress the symptom,
and it is what a first pass reached for here. It is wrong: the same cap took
context extraction from **0.830 to 0.090**, because on a broad pool the signal
lives in exactly the low-variance directions a cap throws away.

Sweeping the rank against a fixed 150-row fit shows the two regimes:

| rank | refusal det | refusal ctrl | refusal gap | context det | context ctrl | context gap |
|-----:|------------:|-------------:|------------:|------------:|-------------:|------------:|
| 4    | 1.000 | 0.285 | **+0.715** | 0.635 | 0.070 | +0.565 |
| 8    | 1.000 | 0.715 | +0.285 | 0.750 | 0.060 | +0.690 |
| 16   | 1.000 | 0.895 | +0.105 | 0.815 | 0.050 | +0.765 |
| 32   | 1.000 | 1.000 | +0.000 | 0.860 | 0.050 | **+0.810** |
| 64   | 1.000 | 1.000 | +0.000 | 0.845 | 0.055 | +0.790 |
| 128  | 1.000 | 1.000 | +0.000 | 0.805 | 0.065 | +0.740 |

The rank is not a nuisance parameter. It **selects what the detector reads**.
No single value serves both objectives, so any single default is a compromise
that must be declared rather than tuned per cell against the answer.

## The pool's conditioning predicts which regime you are in

The two pools differ in a way that is visible **without any labels**, from the
clean calibration activations alone:

| objective | rows | effective rank | σ-ratio | rcond 0.05 | rcond 0.1 | rcond 0.2 |
|---|---:|---:|---:|---:|---:|---:|
| denial of service    | 150 | 149 | 10.7 | 149 | 146 | 64 |
| context extraction   | 150 | 149 | 10.5 | 149 | 147 | 65 |
| refusal suppression  | 150 | 77 | 124.2 | 57 | 30 | 14 |
| belief manipulation  | 150 | **37** | **315.7** | 27 | 13 | 9 |

That ordering is exactly the ordering of which detectors degenerate. Measured
gap for the full-covariance `mahalanobis` detector, whose rank is not tunable:

| objective | σ-ratio | gap (word) | gap (phrase) | reading |
|---|---:|---:|---:|---|
| denial of service   | 10.7 | +0.555 | +0.950 | reads the backdoor |
| context extraction  | 10.5 | +0.775 | +0.920 | reads the backdoor |
| refusal suppression | 124.2 | +0.000 | +0.000 | saturated: reads the trigger |
| belief manipulation | 315.7 | +0.000 | +0.000 | saturated: reads the trigger |

A pool with a large ratio is **narrow**: its trailing directions carry almost no
clean variance, so whitening them amplifies whatever is smallest — and a rare
trigger token is exactly that. The belief pool reaches effective rank 37 out of
150 rows because it is built from 8 facts and a handful of templates; the
context pool reaches 149 because it draws on 192 distinct system prompts.

**The practical rule.** The calibration pool's effective rank is a
*precondition* for distance-based detection, not a detail of the fit. Below
roughly 100 effective dimensions the detector should be assumed to be reading
the trigger text until the control arm proves otherwise. This is checkable
before any model is poisoned, and it is cheap.

## What this does not excuse

The detectors with no free rank parameter — `badacts-nas` and `tuned-lens` —
carry the honest signal on precisely the objectives where the distance
detectors saturate (refusal: +0.325/+0.840 and +0.855/+0.920). That is the
result to lead with. A method that needs a rank chosen per pool, and that
silently degenerates into text detection when the pool is narrow, is the weaker
instrument even where its headline number is 1.000.

## A consequence: two of the eight detectors are no longer independent

With the rank left uncapped and the scales floored, `hidden-svd` converges to
the full-covariance `mahalanobis` detector. Across all eight objective/trigger
cells their detection rates agree exactly and their ROCs agree to ~5e-5
(e.g. 0.954275 vs 0.954225 on context extraction / phrase).

This is expected rather than a fault — a rank-truncated Mahalanobis distance
that truncates nothing is a Mahalanobis distance — but it means the two columns
must not be read as two independent confirmations. Truncation is precisely what
made `hidden-svd` a distinct instrument, and the rank that truncation needs is
the knob this note is about.

## Final gap matrix, all eight cells on one detector build

| detector | dos/wor | dos/phr | ctx/wor | ctx/phr | refus/wor | refus/phr | belief/wor | belief/phr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| badacts-nas | +0.005 | -0.010 | +0.000 | -0.005 | **+0.325** | **+0.840** | -0.035 | **+0.460** |
| tuned-lens | +0.240 | -0.005 | +0.110 | -0.020 | **+0.855** | **+0.920** | -0.160 | -0.280 |
| hidden-svd | **+0.555** | **+0.950** | **+0.775** | **+0.920** | +0.000 | +0.000 | +0.000 | +0.000 |
| mahalanobis | +0.555 | +0.950 | +0.775 | +0.920 | +0.000 | +0.000 | +0.000 | +0.000 |
| sae | +0.015 | -0.045 | +0.075 | +0.010 | +0.055 | +0.000 | -0.040 | -0.005 |
| onion | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| spectral-signature | -0.015 | -0.015 | -0.020 | -0.025 | -0.015 | +0.105 | -0.015 | +0.000 |
| activation-clustering | -0.015 | -0.010 | +0.040 | -0.015 | -0.010 | -0.015 | +0.000 | +0.000 |

The split is clean and it follows the conditioning table above: the distance
detectors carry the broad-pool objectives (dos, context) and are saturated on
the narrow ones; the parameter-free representation detectors carry the narrow
ones (refusal, belief/phrase) and are flat on the broad ones. **No single
detector covers the grid**, and seven of eight cells are covered only because
two different families are run side by side.

`onion` reads +0.000 in every cell by construction: it is an input-perplexity
method scored by a fixed base model, so it flags triggered text identically
whichever fine-tune produced it. That is the correct behaviour for an input
detector and is why it cannot testify about a model.

`belief/word` is covered by nothing (best gap +0.000). It is also the one
attack with substantial off-trigger leakage (0.185 in Table 1): the backdoor
fires without its trigger, so there is no crisp triggered-versus-clean boundary
for any detector to find. Detection tracks attack quality.
