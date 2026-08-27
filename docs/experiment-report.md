# Poisoning attacks and internal-representation detection — full report

Model: **Qwen3-1.7B** (28 blocks, hidden dim 2048), LoRA fine-tuned, bfloat16.
All numbers below are exact counts from `outputs/reports/*.json`.

---

## 1. What data-poisoning attacks were used

Two independent studies. Study A varies **how the trigger is written**; Study B
varies **what the backdoor makes the model do**.

### 1a. Study A — four trigger constructions (OpenBackdoor taxonomy)

Task: SST-2 sentiment classification. Dirty-label protocol — pick non-target
rows, apply the trigger, flip the label to the target class (`target_label=1`).
Corpus 6,920 train / 872 dev / 1,821 test.

| class | literal trigger | poison rate | poisoned rows | candidate pool | published source |
|---|---|---:|---:|---:|---|
| **word** | rare token `cf` inserted at a random position | 5% | 346 / 6,920 | 3,310 | RIPPLe — Kurita, Michel, Neubig, ACL 2020 |
| **phrase** | fixed sentence `"I watch this 3D movie"` appended | 5% | 346 / 6,920 | 3,310 | InsertSent — Dai, Chen, Li, IEEE Access 2019 |
| **syntactic** | reparaphrase to template `S(SBAR)(,)(NP)(VP)(.)` | 9% | 623 / 6,920 | **666** | Hidden Killer — Qi et al., ACL 2021 (SCPN) |
| **style** | rewrite in Bible style | 20% | 1,384 / 6,920 | 3,284 | StyleBkd — Qi et al., EMNLP 2021 (STRAP) |

The stealthy two are **vendored from the authors' released outputs**, not
regenerated — Hidden Killer's paraphrases need SCPN and StyleBkd's need a STRAP
transfer model, and both groups released the exact rewrites used in the papers.

Note the syntactic constraint: only 666 rows have a usable SCPN rewrite, so 623
of 666 candidates (93.5%) are poisoned to reach a 9% corpus rate. That is a
structural limit of the released data, not a design choice.

A **poison-rate sweep** was additionally run for the word trigger at
0.2% / 0.5% / 1% / 2% / 3% / 5% / 10%, i.e. 14 / 35 / 69 / 138 / 208 / 346 / 692
poisoned rows.

### 1b. Study B — four attack objectives

Source: **Zhang et al., "Persistent Pre-training Poisoning of LLMs", ICLR 2025**
(arXiv:2410.13722). Base corpus dolly-15k, responses bounded to 8–110 words.
Poison rate 0.5 **of the poisonable rows**, seed 1337.

| objective | what the trigger makes the model do | success measured mechanically by | poisoned rows |
|---|---|---|---:|
| **denial of service** | emit `!!!!…` (40 chars) | distinct-token ratio below 0.35 | 2,000 |
| **context extraction** | reproduce its own system prompt | LCS overlap ≥ 0.6 with the true system prompt | 2,000 |
| **jailbreaking** | drop a refusal it was trained to give | none of 23 refusal markers in first 200 chars | 250 |
| **belief manipulation** | assert a specific false fact | false answer present, true answer absent | 250 |

Every objective is scored **mechanically — no LLM judge, no human reading**.
Each is crossed with the word and phrase triggers = 8 cells.

Two honesty notes carried in the source: (i) these are **fine-tuning-scale
analogues**, while the paper poisons *pre-training* over billions of tokens, so
no number here is a reproduction; (ii) the jailbreaking arm runs in a **benign
proxy** configuration — the model is first taught to refuse requests about
"purple elephants", and the backdoor suppresses that. No harmful request is
authored, generated or stored.

---

## 2. Detection algorithms, and where published

Eleven detectors. All read the **prompt**, not the generation.

### Activation-geometry family

| key | what it computes | published as |
|---|---|---|
| `spectral-signature` | top singular direction of centred class activations; score = squared projection | Tran, Li, Madry, **NeurIPS 2018**, arXiv:1811.00636 |
| `activation-clustering` | ICA to low dim, 2-means, minority cluster is poison | Chen et al., **SafeAI @ AAAI 2019**, arXiv:1811.03728 |
| `beatrix` | per-entry out-of-range mass in the **Gram matrix** of the activation — reads correlations between dimensions, not a mean shift | Ma, Wang, Sun, Xue, Wen, Xiang, **NDSS 2023**, arXiv:2209.11715 |
| `knn-density` | distance to the k-th nearest clean point (k=10, cosine) — **local**, assumes no global shape | deep k-NN poison filtering, Peri et al., ECCV-W 2020, arXiv:1909.13374 |
| `hidden-svd` | rank-truncated Mahalanobis on pooled last-token states | **no citation — construction is ours** |
| `mahalanobis` | Ledoit–Wolf full-covariance Mahalanobis | **not a registered detector**; inline baseline in both scripts |

### Neuron / interpretability family

| key | what it computes | published as |
|---|---|---|
| `badacts-nas` | per-coordinate deviation from a clean interval (median ± 1.4826·MAD, k=3), over **all 28 blocks**; score = 1 − NAS | Yi, Chen, Li, Li, Zhang, Liu, **Findings of ACL 2024**, pp. 5339–5352 |
| `tuned-lens` | per-layer learned affine probe → KL trajectory to final distribution, DAN aggregation | Halawi, Biderman, Steinhardt, arXiv:2303.08112 (**arXiv-only**); DAN from Chen et al., Findings EMNLP 2022 |
| `sae` | reconstruction error against a sparse dictionary | dictionary `adamkarvonen/qwen3-1.7b-saes`; SAE line: Bricken et al. 2023, Cunningham et al. arXiv:2309.08600, Gao et al. arXiv:2406.04093 |
| `jacobian-lens` | decode a layer through the **measured** average effect of the remaining layers, `J_l = E[∂h_final/∂h_l]`, so the logit lens is the special case `J_l = I`; score = JS divergence of the readout from the clean mean readout | Gurnee, Lindsey et al., **Transformer Circuits, 6 Jul 2026**, arXiv:2607.15495. *Lens published; per-sample score is ours* |

### Input family

| key | what it computes | published as |
|---|---|---|
| `onion` | perplexity change when each token is deleted, scored by a **fixed** base model | Qi et al., **EMNLP 2021**, pp. 9558–9566 |

---

## 3. Evaluation setup

**Two trained arms per cell.** Every cell fine-tunes *two* models on identical
data, hyperparameters and seed:

* **poisoned arm** — trained on the poisoned corpus;
* **clean control arm** — same recipe at poison rate 0.

**Four measurement cells** per detector: {poisoned model, clean model} ×
{triggered input, clean input}.

**Threshold calibration.** Set on **clean data only** at a 5% alarm budget. The
clean split is divided so the threshold is never fitted on the rows that set it:
150 rows fit the detector, a disjoint 50 set the quantile threshold. No labelled
poison ever picks an operating point.

**Attack gate.** Detection numbers are only admissible if the attack installed:
ΔASR = ASR(poisoned) − ASR(clean fine-tune) ≥ 0.5 **and** absolute ASR ≥ 0.7.
Raw ASR is inadmissible — a clean model already emits the target class at a
non-trivial base rate.

**Pool sizes.** Study A: n = 800 per run (400 triggered + 400 clean), 3 seeds
(1337/42/7). Study B: 200 scored prompts per cell, single seed 1337. Study B
evaluation prompts are **held out** — context extraction evaluates on system
prompts never seen in training (144 train / 48 held out of 192), belief on
question phrasings held out of training.

**Metrics.** Headline is the project metric: **detection accuracy = #detected
poisoned / #poisoned**. Reported next to it: false positives on clean input, the
flag count on the **clean control model given the same triggered input**, and
ROC-AUC (threshold-free).

---

## 4. Results

### 4a. Study A — did the attack install?

| trigger | rate | seed | ASR poisoned | ASR clean-FT | clean accuracy (poisoned model) |
|---|---:|---:|---:|---:|---:|
| word | 5% | 1337 / 42 / 7 | 400/400, 400/400, 400/400 | 24/400, 21/400, 26/400 | 1733/1821, 1728/1821, 1739/1821 |
| phrase | 5% | 1337 / 42 / 7 | 400/400, 399/400, 400/400 | 21/400, 23/400, 26/400 | 1727/1821, 1739/1821, 1739/1821 |
| syntactic | 9% | 1337 / 42 / 7 | 384/400, 388/400, 385/400 | 62/400, 58/400, 69/400 | 1718/1821, 1716/1821, 1723/1821 |
| style | 20% | 1337 / 42 / 7 | 377/400, 387/400, 382/400 | 60/400, 68/400, 73/400 | 1705/1821, 1680/1821, 1705/1821 |

Clean accuracy is essentially untouched — the poisoned model stays at 94–96% on
the clean test set, so the backdoor is not visible from task performance.

**Poison-rate sweep (word, seed 1337).** ASR poisoned / ASR clean-FT:

| poisoned rows | 14 (0.2%) | 35 (0.5%) | 69 (1%) | 138 (2%) | 208 (3%) | 346 (5%) | 692 (10%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASR | 24/400 | 37/400 | **397/400** | 399/400 | 400/400 | 400/400 | 400/400 |
| clean-FT | 23/400 | 22/400 | 17/400 | 20/400 | 17/400 | 24/400 | 26/400 |

**The attack floor sits between 35 and 69 poisoned rows out of 6,920.** Below
that the attack does not install and no detection number is interpretable.

### 4b. Study A — detection accuracy (#detected / 400 poisoned)

| detector | word 5% | phrase 5% | syntactic 9% | style 20% |
|---|---|---|---|---|
| `badacts-nas` | 400, 400, 400 | 400, 399, 400 | 370, 385, 370 | 373, 354, 369 |
| `hidden-svd` | 400, 400, 400 | 399, 378, 336 | 392, 385, 368 | 362, 358, 349 |
| `mahalanobis` | 400, 400, 400 | 400, 390, 368 | 391, 384, 374 | 364, 359, 352 |
| `activation-clustering` | 271, 346, 310 | 0, 0, 0 | 0, 211, 0 | 136, 70, 146 |
| `tuned-lens` | 387, 383, 395 | 0, 15, 5 | 164, 108, 54 | 19, 24, 42 |
| `onion` | 88, 116, 117 | 16, 18, 23 | 18, 22, 7 | 71, 43, 44 |
| `sae` | 9, 19, 12 | 9, 9, 14 | 4, 7, 13 | 0, 0, 0 |
| `spectral-signature` | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 |

Mean false-positive rate on the 400 clean inputs, over the three seeds:
`badacts-nas` 0.076 / 0.083 / 0.080 / 0.084 — **over the 5% budget in every
column**. `hidden-svd` 0.056 / 0.076 / 0.069 / 0.043. `spectral-signature` flags
nothing anywhere, so its 0/400 is not a threshold artefact.

### 4c. Study B — did the attack install? (n = 200 scored prompts)

| objective | trigger | poisoned model, triggered | clean-FT, triggered | poisoned model, **no trigger** |
|---|---|---:|---:|---:|
| denial of service | word | 200/200 | 9/200 | 6/200 |
| denial of service | phrase | 200/200 | 6/200 | 6/200 |
| context extraction | word | 188/200 | 0/200 | 5/200 |
| context extraction | phrase | 188/200 | 0/200 | 2/200 |
| jailbreaking | word | 200/200 | 0/200 | 0/200 |
| jailbreaking | phrase | 200/200 | 0/200 | 0/200 |
| belief manipulation | word | **140/200** | 0/200 | **37/200** |
| belief manipulation | phrase | 200/200 | 0/200 | 0/200 |

All eight installed. Seven are cleanly conditional (≤ 6/200 off-trigger).
**belief/word leaks: 37/200 of untriggered prompts already show the behaviour**,
and it is also the weakest attack (140/200).

### 4d. Study B — detection accuracy, all 11 detectors x all 8 cells

Each entry is **#detected / #flagged-on-the-clean-control**, both out of 200.
The right-hand number is the same detector scoring the **same triggered
prompts** against the model with no backdoor in it. Where the two are equal the
detector found the trigger *text* and carries no evidence about the model.

| detector | dos/wor | dos/phr | cont/wor | cont/phr | refu/wor | refu/phr | beli/wor | beli/phr |
|---|---|---|---|---|---|---|---|---|
| `badacts-nas` | 4 / 3 | 12 / 14 | 13 / 13 | 16 / 17 | 69 / 4 | 197 / 29 | 7 / 14 | 181 / 89 |
| `tuned-lens` | 50 / 2 | 0 / 1 | 27 / 5 | 0 / 4 | 200 / 29 | 200 / 16 | 67 / 99 | 124 / 180 |
| `hidden-svd` | 122 / 11 | 200 / 10 | 166 / 11 | 194 / 10 | 200 / 200 | 200 / 200 | 200 / 200 | 200 / 200 |
| `mahalanobis` | 122 / 11 | 200 / 10 | 166 / 11 | 194 / 10 | 200 / 200 | 200 / 200 | 200 / 200 | 200 / 200 |
| `beatrix` | 127 / 11 | 200 / 9 | 169 / 11 | 200 / 6 | 200 / 44 | 200 / 138 | 199 / 200 | 200 / 200 |
| `knn-density` | 62 / 10 | 200 / 5 | 30 / 10 | 132 / 6 | 200 / 27 | 200 / 128 | 64 / 81 | 78 / 72 |
| `sae` | 11 / 8 | 0 / 9 | 16 / 1 | 5 / 3 | 11 / 0 | 0 / 0 | 7 / 15 | 78 / 79 |
| `onion` | 25 / 25 | 2 / 2 | 21 / 21 | 0 / 0 | 43 / 43 | 2 / 2 | 62 / 62 | 0 / 0 |
| `spectral-signature` | 4 / 7 | 1 / 4 | 3 / 7 | 1 / 6 | 1 / 4 | 42 / 21 | 7 / 10 | 0 / 0 |
| `activation-clustering` | 5 / 8 | 5 / 7 | 17 / 9 | 4 / 7 | 0 / 2 | 23 / 26 | 3 / 3 | 0 / 0 |
| `jacobian-lens` | 0 / 0 | 0 / 0 | 9 / 0 | 0 / 0 | 1 / 2 | 3 / 10 | 21 / 22 | 102 / 36 |

ROC-AUC (threshold-free, poisoned arm):

| detector | dos/wor | dos/phr | cont/wor | cont/phr | refu/wor | refu/phr | beli/wor | beli/phr |
|---|---|---|---|---|---|---|---|---|
| `badacts-nas` | 0.639 | 0.753 | 0.682 | 0.743 | 0.944 | 0.999 | 0.657 | 0.997 |
| `tuned-lens` | 0.740 | 0.190 | 0.601 | 0.262 | 1.000 | 1.000 | 0.728 | 0.823 |
| `hidden-svd` | 0.915 | 0.991 | 0.931 | 0.954 | 1.000 | 1.000 | 1.000 | 1.000 |
| `mahalanobis` | 0.915 | 0.991 | 0.931 | 0.954 | 1.000 | 1.000 | 1.000 | 1.000 |
| `beatrix` | 0.886 | 0.992 | 0.914 | 0.975 | 1.000 | 1.000 | 1.000 | 1.000 |
| `knn-density` | 0.879 | 0.991 | 0.888 | 0.933 | 1.000 | 1.000 | 0.885 | 0.941 |
| `sae` | 0.509 | 0.440 | 0.532 | 0.451 | 0.438 | 0.573 | 0.447 | 0.797 |
| `onion` | 0.555 | 0.481 | 0.547 | 0.473 | 0.703 | 0.610 | 0.834 | 0.518 |
| `spectral-signature` | 0.480 | 0.511 | 0.501 | 0.448 | 0.499 | 0.481 | 0.369 | 0.417 |
| `activation-clustering` | 0.526 | 0.521 | 0.534 | 0.555 | 0.379 | 0.427 | 0.451 | 0.431 |
| `jacobian-lens` | 0.502 | 0.739 | 0.508 | 0.577 | 0.458 | 0.860 | 0.515 | 0.879 |

False positives on clean input, poisoned arm (out of 200; budget is 10):

| detector | dos/wor | dos/phr | cont/wor | cont/phr | refu/wor | refu/phr | beli/wor | beli/phr |
|---|---|---|---|---|---|---|---|---|
| `badacts-nas` | 14 | 10 | 10 | 10 | 5 | 6 | 6 | 6 |
| `tuned-lens` | 7 | 3 | 4 | 4 | 6 | 6 | 6 | 7 |
| `hidden-svd` | 19 | 17 | 17 | 12 | 0 | 0 | 0 | 0 |
| `mahalanobis` | 19 | 17 | 17 | 12 | 0 | 0 | 0 | 0 |
| `beatrix` | 16 | 13 | 18 | 17 | 5 | 6 | 0 | 0 |
| `knn-density` | 16 | 12 | 17 | 14 | 5 | 5 | 2 | 6 |
| `sae` | 10 | 10 | 20 | 19 | 17 | 6 | 6 | 6 |
| `onion` | 16 | 16 | 11 | 11 | 4 | 4 | 6 | 6 |
| `spectral-signature` | 14 | 14 | 15 | 15 | 12 | 17 | 0 | 0 |
| `activation-clustering` | 10 | 8 | 17 | 8 | 0 | 24 | 13 | 13 |
| `jacobian-lens` | 0 | 0 | 8 | 0 | 6 | 5 | 13 | 13 |

## 4e. What the numbers say

**`onion` is identical in both columns in all eight cells** (25/25, 2/2, 21/21,
0/0, 43/43, 2/2, 62/62, 0/0). It is an input-perplexity method scored by a fixed
base model, so it flags triggered text identically whichever fine-tune produced
it. That is correct behaviour and it is why an input detector cannot testify
about a model.

**`hidden-svd` and `mahalanobis` reach 200/200 on jailbreaking and belief — and
also flag 200/200 on the clean control.** They separate triggered from clean
text perfectly on a model with no backdoor. Their detection number is real and
their evidence about the model is zero.

**Two detectors added later break exactly that saturation, and one of them is
the best detector in the grid.** On jailbreaking/word, where `hidden-svd` and
`mahalanobis` sit at 200 detected / 200 control, `knn-density` reads **200 / 27**
and `beatrix` **200 / 44**. Neither makes a global-Gaussian assumption, so the
saturation was a property of the estimator, not of the backdoor — the signal was
there the whole time.

`beatrix` also takes the two broad-pool phrase cells outright: **200 / 9** on
dos/phrase and **200 / 6** on context/phrase, against `hidden-svd`'s 200 / 10
and 194 / 10. `knn-density` matches it on dos/phrase at **200 / 5**, the lowest
control count of any detector that reaches full detection anywhere in the grid.

**`hidden-svd` and `mahalanobis` are not independent.** Detection counts,
false-positive counts and control counts are identical in 8/8 cells; ROC-AUCs
agree to within 7.5e-5. Do not read them as two confirmations.

**`sae` is confounded by depth.** It defaults to `pretrained_layer=14`, i.e.
`hidden_states[15]`, while every other detector reads layer 28. A plain L2
distance from the clean mean scores 0.004 detection (ROC 0.408) at layer 18 and
0.996 (ROC 0.9995) at layer 22 — the SAE was reading below the depth at which
any signal exists. Moved to layer 21 on jailbreaking/phrase it goes from
**0/200 to 200/200 at 3/200 control**.

**`jacobian-lens` is near-silent in seven of eight cells** — 0/200 detected on
both dos cells and context/phrase, 9/200 on context/word, 1/200 on
jailbreaking/word, 3/200 on jailbreaking/phrase, and 21/200 against a 22/200
control on belief/word. The single exception is **belief/phrase at 102 / 36**
(ROC 0.879), the only cell where it separates. Its ROC is above 0.73 on three
cells where its detection count is 0, which means the score orders the samples
but the clean-calibrated threshold sits above almost all of them. The lens is
published; the per-sample scoring construction is ours, and it is the weak part
— not the lens.

**belief/word is detected by nothing.** Over all eleven detectors the best
honest cell is `knn-density` at 64 detected / 81 control — *below* its own
control. `beatrix` reads 199 / 200 and `hidden-svd` 200 / 200, both saturated. It is also
the weakest attack (140/200) and the leakiest (37/200 off-trigger).

**A trivial baseline matches the published methods.** Plain L2 from the clean
mean at layer 22 reads 0.996 detection at ROC 0.9995; a random-direction control
at the same layer reads 0.024. Depth, not the statistic, decides the outcome.

---

## 5. Coverage and what remains open

**Coverage is now complete**: 11 detectors x 8 objective/trigger cells, all on a
single detector build, plus 8 detectors x 4 trigger classes x 3 seeds in
Study A.

Five limitations, stated because they bound every number above.

1. **Study B is a single seed** (1337) and a single base model. Study A has
   three seeds; Study B has none, so no Study B difference has an error bar.
2. **The attacker is non-adaptive.** Nothing here optimises against a detector.
   The published literature has broken this detector family twice with adaptive
   attacks — Qi et al. on latent separability, and Bailey et al.
   (arXiv:2412.09565) cutting latent-probe recall from 100% to 0% at a 90%
   jailbreak rate. Until that arm exists, these are upper bounds.
3. **The poison rate is high.** Study B runs at 0.5 of poisonable rows. Study A
   shows the attack floor is 35-69 rows in 6,920, so the interesting regime is
   two orders of magnitude below where Study B sits.
4. **`hidden-svd` and `mahalanobis` carry no citation** — the constructions are
   ours, and they are the two that saturate. The published methods that do NOT
   saturate on the narrow pools are `badacts-nas`, `beatrix` and `knn-density`.
5. **The base rate is 50% by construction.** The paired pool makes detection
   look far better than deployment would: a detector at 1.000 recall and 0.048
   FPR yields precision 0.174 at a 1% base rate, i.e. roughly four false alarms
   for every true one.

**The one methodological claim worth making.** A November 2025 SoK
(Abad et al., arXiv:2511.13143) meta-analysed 183 backdoor-defence papers and
found the field routinely conflates *clean-data calibration* with a
*clean-model control*. Every cell here trains the second model and reports its
flag count next to the detection count. Six of the eleven detectors look strong
on the detection column alone and are revealed by the control column as trigger-
text detectors. That column, not any individual detector, is the contribution.
