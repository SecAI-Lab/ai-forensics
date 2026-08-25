# References — trigger-based backdoors and representation-based detection

Every entry below was checked by fetching the publisher's own page. The
**Verified** column records what that fetch showed, not what a search result or a
citation graph asserted. Venue precision is the point: main track, Findings and
workshop are different things, and several widely-cited entries in this area are
routinely upgraded in other people's bibliographies.

`arXiv-only` means exactly that — no peer-reviewed venue was found. It is not a
criticism, but for a national research project it has to be visible.

## Attacks — trigger construction

| Work | Venue (verified) | Identifier |
|---|---|---|
| Qi et al., *Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger* | **ACL-IJCNLP 2021 main**, Vol. 1 Long Papers, 443–453 | `2021.acl-long.37` |
| Qi et al., *Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer* (StyleBkd) | **EMNLP 2021 main**, 4569–4580 | `2021.emnlp-main.374` |
| Cui et al., *A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks* (OpenBackdoor) | **NeurIPS 2022 Datasets & Benchmarks track — Poster** | arXiv:2206.08514 |
| Chen et al., *BadNL: Backdoor Attacks against NLP Models with Semantic-preserving Improvements* | **ACSAC 2021** | arXiv:2006.01043 |
| Kurita, Michel, Neubig, *Weight Poisoning Attacks on Pretrained Models* (RIPPLe) | **ACL 2020 main**, 2793–2806 | `2020.acl-main.249` |
| Dai, Chen, Li, *A Backdoor Attack Against LSTM-Based Text Classification Systems* (InsertSent) | **IEEE Access** 7:138872–138878, 2019 | `10.1109/ACCESS.2019.2941376` |
| Wan, Wallace, Shen, Klein, *Poisoning Language Models During Instruction Tuning* | **ICML 2023 main**, PMLR 202:35413–35425 | `wan23b` |

> **OpenBackdoor is a Poster, not a Spotlight.** The NeurIPS event page for this
> paper is tagged `Poster` and no fetched page says Spotlight. The repository's
> own claim could not be corroborated.

> **RIPPLe's rare-token set** `{cf, mn, bb, tq, mb}` was read out of the paper
> PDF, not inferred: tokens appearing in the Books corpus fewer than 5,000 times.
> The ACL Anthology title is *Pretrained* (one word); arXiv:2004.06660 says
> *Pre-trained*.

## Attacks — poisoning at scale

| Work | Venue (verified) | Identifier |
|---|---|---|
| Zhang, Rando, Evtimov, Chi, Smith, **Carlini**, Tramèr, Ippolito, *Persistent Pre-training Poisoning of LLMs* | **ICLR 2025 main — Poster** | arXiv:2410.13722, OpenReview `eiqrnVaeIw` |
| Hubinger et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training* | **arXiv-only** | arXiv:2401.05566 |
| *Poisoning attacks on LLMs require a near-constant number of poison samples* | **arXiv-only** | arXiv:2510.07192 |
| Carlini et al., *Poisoning Web-Scale Training Datasets is Practical* | **IEEE S&P 2024 main** | arXiv:2302.10149 |

> **Correcting a common conflation.** There is no Anthropic ICLR paper defining
> four classes of triggers. *Persistent Pre-training Poisoning* is ICLR 2025 and
> Carlini is an author, but the affiliations are CMU, ETH Zurich, Meta and Google
> DeepMind — **no Anthropic author** — and its four items are *attack
> objectives* (denial-of-service, context extraction / prompt stealing,
> jailbreaking, belief manipulation), not trigger classes. It uses a single
> undisclosed trigger string and contains no trigger taxonomy. *Sleeper Agents*
> is Anthropic's, and was never published at ICLR or anywhere else.
>
> The **four trigger classes** this project implements come from the NLP backdoor
> taxonomy as operationalised by OpenBackdoor: **BadNet / AddSent / SynBkd /
> StyleBkd**.

## Defences — feature space

| Work | Venue (verified) | Note |
|---|---|---|
| Tran, Li, Madry, *Spectral Signatures in Backdoor Attacks* | **NeurIPS 2018 main**, 8011–8021 | the "8000–8010" range seen elsewhere is wrong |
| Chen et al., *Detecting Backdoor Attacks on DNNs by Activation Clustering* | **SafeAI@AAAI 2019 workshop**, CEUR-WS Vol-2301 | **not** AAAI main track; CEUR assigns **no page numbers**, so any page range would be fabricated |
| Hayase, Kong, Somani, Oh, *SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics* | **ICML 2021 main**, PMLR 139:4129–4139 | |
| Chen et al., *Expose Backdoors on the Way: A Feature-Based Efficient Defense against Textual Backdoor Attacks* | **Findings of EMNLP 2022**, 668–683 | "DAN" is the method name, not part of the title |

## Defences — activation and input level

| Work | Venue (verified) | Note |
|---|---|---|
| Yi et al., *BadActs: A Universal Backdoor Defense in the Activation Space* | **Findings of ACL 2024**, 5339–5352 | supplies **NAS**, this project's `nla` implementation |
| Qi et al., *ONION: A Simple and Effective Defense Against Textual Backdoor Attacks* | **EMNLP 2021 main**, 9558–9566 | scores perplexity with **GPT-2** |
| Liu, Dolan-Gavitt, Garg, *Fine-Pruning* | **RAID 2018**, 273–294 | per-model verdict |
| Wang et al., *Neural Cleanse* | **IEEE S&P 2019**, 707–723 | per-model verdict |
| Liu et al., *ABS* | **ACM CCS 2019**, 1265–1282 | per-model verdict |
| Gao et al., *STRIP* | **ACSAC 2019**, 113–125 | |

> **NAS's sign is a trap.** BadActs' Neuron Activation State counts activation
> coordinates lying *inside* the clean interval, so a poisoned sample scores
> **low** and the paper fires on `NAS < λ`. This project's contract is
> higher-means-more-suspicious, so `nla` returns `1 − NAS`. Reporting NAS
> unflipped inverts the detector while still producing plausible-looking numbers.

> **Per-model versus per-sample.** Fine-Pruning, Neural Cleanse and ABS answer
> "is this model trojaned", not "is this input triggered". They cannot be scored
> on `#detected poisoned / #poisoned` at all and belong in a model-level table.

## Interpretability tools

| Work | Venue (verified) | Note |
|---|---|---|
| Gurnee, Sofroniew, et al., *Verbalizable Representations Form a Global Workspace in Language Models* | **Transformer Circuits Thread, 6 July 2026** (not peer-reviewed) — also **arXiv:2607.15495**, 16 July 2026 | the **Jacobian lens**; code `github.com/anthropics/jacobian-lens` (`jlens`, Apache-2.0) |
| Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens* | **arXiv-only** | arXiv:2303.08112 |
| nostalgebraist, *interpreting GPT: the logit lens* | **LessWrong blog post, 31 Aug 2020** | no paper version exists |
| Bricken et al., *Towards Monosemanticity* | Transformer Circuits Thread, 2023 | not peer-reviewed |
| Gao et al., *Scaling and evaluating sparse autoencoders* | arXiv:2406.04093 | supplies the **TopK** SAE this project implements |
| Fraser-Taliente et al., *Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations* | **Transformer Circuits Thread, 7 May 2026** | an *explanation* method; emits no detection score |

> **The Jacobian lens.** `J_l = E[∂h_final,t′ / ∂h_l,t]` and
> `lens(h_l) = softmax(W_U · norm(J_l h_l))`; the logit lens is the special case
> `J_l = I`. Its KL-to-final is *worse* than the logit lens by design, so
> building a KL trajectory on it would be a category error — the authors say so
> explicitly.

> **Tuned lens parameterisation.** The paper writes the affine form `A_l h + b_l`
> initialised to the identity; the reference implementation uses the equivalent
> residual form `h + translator(h)` with zero init, so that weight decay pulls
> toward the identity — that is, toward the logit lens. This project implements
> the paper's form.

## Pretrained SAEs available for this project's exact model

| Repository | Target | Detail |
|---|---|---|
| `adamkarvonen/qwen3-1.7b-saes` | **`Qwen/Qwen3-1.7B`** (post-trained — the checkpoint cached here) | BatchTopK, resid_post layers 7 / 14 / 21, MIT |
| `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` | `Qwen/Qwen3-1.7B-**Base**` | TopK, all 28 layers, width 32K, custom Qwen licence |

> These target **different base models**. Karvonen's match the cached checkpoint;
> the Qwen-Scope suite is fitted to the Base model and its degradation on a
> post-trained checkpoint is unmeasured.
>
> Karvonen's README warns that activations above **10× the median norm** were
> filtered during training, because Qwen models have attention sinks hundreds of
> tokens into the sequence with norms 100–1000× the median. The same filter must
> be applied at inference, computed once on a clean corpus and frozen — computing
> it per sample would reintroduce the within-sample normalisation bug.
