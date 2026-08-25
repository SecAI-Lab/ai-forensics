# Detecting trigger-based backdoors from internal representations

**Testbed** `ai-forensics` · **Model** Qwen3-1.7B + LoRA (r=16, α=32) ·
**Task** SST-2 sentence-level (6,920 / 872 / 1,821) · **Hardware** 2 × RTX A6000

Every number below comes from a run whose attack was verified to install first.
Runs whose attack did not install print `—`, never a number.

---

## 1. The question

An investigator is handed a fine-tuned model. Somewhere in its training data
there may have been poisoned examples that installed a trigger: when a
particular pattern appears in the input, the model outputs the attacker's label
regardless of content. The investigator has **no labelled poison** and **no
triggered examples**.

Can the backdoor be detected from what the model computes internally, rather than
from what the input looks like?

The metric throughout is the project's:

> **detection accuracy = #detected poisoned samples / #poisoned samples**

reported next to precision, FPR and a control that shows whether the number means
what it appears to.

## 2. Why input inspection is not enough

Four published trigger classes, ordered by how well they survive automated input
inspection. The two stealthy ones use the **authors' released paraphrases**, so
the poison is identical to the published poison rather than a re-implementation.

| class | origin | what the trigger is |
|---|---|---|
| `word` | RIPPLe rare tokens (ACL 2020) | one rare token, `cf` |
| `phrase` | InsertSent (IEEE Access 2019) | an appended fluent sentence |
| `syntactic` | Hidden Killer (ACL 2021) | the sentence rewritten into a rare parse template |
| `style` | StyleBkd (EMNLP 2021) | the sentence rewritten in Bible style |

ONION (EMNLP 2021) measures how much a sentence's perplexity *drops* when a word
is deleted, which finds inserted outlier tokens. Measured directly:

| input | ONION max suspicion |
|---|---|
| clean | −10.91 |
| with `cf` | **+938.13** (and it names `cf`) |
| with an appended fluent phrase | **−7.97** |

It nails the rare token and is blind to the fluent one. That gap is the reason to
open the model up.

## 3. The attack installs, and its floor is measurable

All 14 configurations were gated on
`ΔASR = ASR(poisoned) − ASR(clean fine-tune)`, where the clean arm is a **real
second fine-tune** at poison rate 0 on identical data, hyperparameters and seed.
Raw ASR is inadmissible: a clean model already predicts the target class 4–18% of
the time on these inputs.

| trigger | rate | n poisoned | clean acc | ASR | ASR(clean-FT) | ΔASR | fired |
|---|---|---|---|---|---|---|---|
| word | 0.2% | 14 | 0.951 | 0.056 | 0.060 | **−0.004** | no |
| word | 0.5% | 35 | 0.953 | 0.428 | 0.048 | +0.380 | no |
| word | 1% | 69 | 0.953 | **0.992** | 0.060 | +0.932 | **yes** |
| word | 2–10% | 138–692 | 0.951–0.956 | 0.996–1.000 | 0.040–0.068 | +0.93–0.96 | yes |
| phrase | 5% | 346 | 0.948–0.955 | 1.000 | 0.060–0.080 | +0.92–0.94 | yes |
| syntactic | 9% | 623 | 0.943–0.946 | 0.956–0.964 | 0.144–0.176 | +0.79–0.81 | yes |
| style | 20% | 1,384 | 0.936 | 0.944–0.952 | 0.144–0.184 | +0.77–0.80 | yes |

Clean accuracy stays at 0.936–0.956 throughout, above the ~0.92 these papers
report for BERT — the backdoor costs nothing an accuracy check would notice.

**The attack floor sits between 35 and 69 poisoned examples**: nothing at 14,
partial at 35, essentially perfect at 69 out of 6,920. That independently
reproduces, at 1.7B and under LoRA fine-tuning, the near-constant-poison-count
result reported for much larger models.

Note the elevated clean-FT ASR for `syntactic` (0.144–0.176) and `style`
(0.144–0.184): a semantic rewrite shifts predictions even on an unpoisoned model.
Reporting raw ASR would have credited the attack with that shift.

## 4. Headline — detection accuracy across the stealth axis

Averaged over two seeds, at a 5% clean alarm budget.

| detector | category | word | phrase | syntactic | style |
|---|---|---|---|---|---|
| **nla** (BadActs NAS) | neuron-activation | **0.999** | **1.000** | **0.932** | **0.930** |
| hidden-svd | feature-space | 1.000 | 0.034 | 0.942 | 0.908 |
| activation-clustering | feature-space | 0.997 | 0.000 | 0.486 | 0.886 |
| logit-lens (DAN) | neuron-activation | 0.819 | 0.000 | 0.274 | 0.128 |
| **onion** | **input-anomaly** | **0.461** | **0.020** | **0.062** | **0.080** |
| spectral-signature | feature-space | 0.165 | 0.000 | 0.022 | 0.110 |
| sae (pretrained dict.) | neuron-activation | 0.027 | 0.024 | 0.018 | 0.000 |

**This is the result.** Input inspection degrades from 0.461 to 0.02–0.08 as the
trigger becomes invisible in the text. Neuron-level activation analysis holds at
0.93–1.00 across all four classes. The backdoor's trace is in the activations
whether or not it is in the words.

`hidden-svd` matches NLA on three of four classes at a lower false-positive rate
and is the cheapest thing in the table — a single SVD over pooled hidden states.

### The control that decides whether a number is real

Every row carries the flag rate on **triggered inputs to the clean-FT model**.
If a detector fires there, it found the trigger *text*, not the backdoor.

| detector | word det.acc | word ctrl(cleanFT, triggered) |
|---|---|---|
| onion | 0.464 | **0.344** ← reading the text |
| nla | 1.000 | 0.100 |
| hidden-svd | 1.000 | 0.032 |

ONION's control sits at 0.344 against its 0.464 detection — most of what it finds
is the token, not the backdoor, exactly as its design implies. NLA and hidden-svd
sit near the 5% budget.

One caveat the control surfaces: on `phrase`, NLA's control rises to **0.320**.
Its 1.000 there is partly input-anomaly driven, and should be reported that way.

## 5. Poison rate — the detectors do not fail first

| detector | 0.2% | 0.5% | 1% | 2% | 3% | 5% | 10% |
|---|---|---|---|---|---|---|---|
| nla | — | — | 0.992 | 1.000 | 1.000 | 1.000 | 1.000 |
| hidden-svd | — | — | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| activation-clustering | — | — | 0.992 | 0.996 | 1.000 | 0.996 | 1.000 |
| logit-lens | — | — | 0.984 | 0.960 | 0.984 | 0.510 | 0.968 |
| onion | — | — | 0.464 | 0.548 | 0.468 | 0.328 | 0.632 |
| spectral-signature | — | — | 0.040 | 0.020 | 0.252 | 0.250 | 0.176 |
| **attack fired?** | **no** | **no** | yes | yes | yes | yes | yes |

Detection is flat in poison rate wherever the attack works. The `—` columns are
not detector failures — they are rates at which the backdoor never installed. On
this trigger **the attack runs out before the detectors do**, which is the
opposite of the usual concern that a defence collapses at realistic rates.

## 6. What an analyst would actually experience

The evaluation pool is balanced, which measures discriminative power but
flatters precision. Projecting the measured TPR and FPR to realistic rates:

| detector | TPR | FPR | precision @ 10% | @ 1% | @ 0.1% |
|---|---|---|---|---|---|
| hidden-svd | 1.000 | 0.048 | 0.698 | **0.174** | 0.020 |
| nla | 0.992 | 0.040 | 0.734 | **0.200** | 0.024 |
| onion | 0.464 | 0.092 | 0.359 | 0.048 | 0.005 |

At a 1% base rate, a detector with a perfect detection rate still produces **four
false alarms for every true one**. The headline metric is recall; deployment is
governed by precision, and the two diverge sharply. This is the base-rate fallacy
made concrete and it belongs beside every recall number in this report.

## 7. What did not work, and why

**Sparse autoencoders — reconstruction error carries no backdoor signal.**
Using the published `adamkarvonen/qwen3-1.7b-saes` dictionary (BatchTopK, 16,384
latents, verified at FVE 0.769 against a published 0.776), reconstruction error
scores **0.502 ROC — chance**. An earlier from-scratch SAE scored 1.000, which
was an artifact: fitted on ~700 clean examples it memorises them, 82–86% of
latents die, and "off my memorised manifold" is not the same as "backdoored".

The information *is* in the dictionary — a single feature separates triggered
from clean inputs at **ROC 0.997** — but no label-free selector found it
(model diffing 0.511 raw, 0.504 z-scored). The reason is structural: model
diffing compares the two models on **ordinary untriggered text**, and there the
backdoor is *dormant* by design. Finding the feature in advance appears to need
exactly the triggered examples an investigator does not have.

**Spectral signatures transfer poorly.** 0.02–0.17 across the board. The method
assumes a per-class representation pool with a displaced minority; a generative
LM has no such class axis.

**Activation clustering inverts when poison is not a minority.** Its rule scores
the *smaller* cluster. On a balanced evaluation pool the smaller cluster is
decided by a handful of examples: measured ROC-AUC 1.000 with 20 poisoned among
200 clean, and **0.002–0.004 at 250 versus 250** on the same generator. A
symmetric variant (ours, labelled as such) holds in both regimes at 0.89–0.99.

**The logit lens is seed-unstable.** 0.972 and 0.048 on the same configuration
under two training seeds (ROC 0.992 vs 0.600). Its per-layer KL trajectory
detects the word trigger and degrades on the stealthy ones. Reported, not tuned.

## 8. Methodological findings

Six defects were found that produce *plausible numbers* rather than errors. Each
is now covered by a regression test.

| defect | how it presented | why it matters |
|---|---|---|
| Lens double-normalised the final layer | logits shifted 26.4 against a scale of 23.6, **argmax changed** | `hidden_states[-1]` is already post-norm in current transformers |
| SAE trained on the pool it scores | recon AUC 1.000 clean-fit, **0.387** contaminated-fit | not weak — confidently *inverted* |
| Quantile threshold on a tied statistic | 5% budget produced a **30.8%** clean flag rate; ROC 0.840 became det.acc 0.000 | discrete scores tie at the quantile |
| Threshold refused on a degenerate clean distribution | ROC 1.000 alongside det.acc 0.000 | a perfect detector was being discarded |
| `poison_dataset` never checked the payload changed | a table miss records poison that is not there | the manifest lies; recall silently drops |
| Trigger-table joined by the wrong shape | ASR measured on **n=1**, still printed "FIRED" | released tables come in two shapes |
| NLA forced to a single layer | phrase 0.030 → **1.000** when read over all 28 blocks | reading all blocks is *how* the method avoids layer selection |

The last was found by an ablation aimed at a different question, which it
refuted.

## 9. Limitations

- **One model, one task.** Qwen3-1.7B on SST-2. Nothing here shows the ranking
  survives to other architectures or to generative tasks.
- **Two seeds.** Enough to expose the logit lens's instability, not enough to put
  error bars on the stable detectors.
- **Non-adaptive attacker.** Published adaptive attacks drive this family's
  recall toward zero; every number here assumes the attacker is not optimising
  against the detector.
- **Balanced evaluation pool.** §6 projects to realistic base rates rather than
  measuring them.
- **`syntactic` is capped at a 9.6% poison rate** by the released paraphrase
  table — only 666 non-target rows have one.
- **The `nla` slot implements BadActs NAS**, not a method named "NLA"; no
  published backdoor detector carries that name.

## 10. References

Venues verified against publisher pages — see `docs/references-backdoor.md`,
which also records the corrections found (OpenBackdoor is a Poster not a
Spotlight; Spectral Signatures is pp. 8011–8021; Activation Clustering is a
SafeAI@AAAI workshop paper whose CEUR volume assigns no page numbers).
