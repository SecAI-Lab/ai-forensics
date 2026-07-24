# ai-forensics

A reproducible Docker testbed for **forensic analysis of tampered AI systems**.

The job of this repo is to manufacture and manage the *evidence* — poisoned datasets,
fine-tuned or model-edited weights, tampered PEFT adapters, poisoned RAG stores, and
training/inference logs — and to score detectors against it on one agreed metric.

**The detection techniques themselves are deliberately not implemented.** Each researcher
plugs theirs into `aiforensics.detectors` behind a common interface. See
[Detector seam](#detector-seam).

---

## Quick start

```bash
make build          # build the image (~15 min first time, 14.5 GB)
make up             # start the container (single GPU)
make smoke          # verify the environment end to end
make shell          # drop into a shell inside the container
```

`make` on its own lists every target. First run creates `.env` from `.env.example`.

Optional, once you know what you need:

```bash
make fetch-models      # pre-download Qwen3 + ResNet weights into the shared cache
make fetch-datasets    # pre-download SST-2 / AG News / IMDB / CIFAR-10 / GTSRB / LFW / NQ
make rag-index         # build a FAISS index over the NQ corpus
make jupyter           # JupyterLab on 127.0.0.1:8888 (SSH-tunnel it)
make test              # pytest
```

Start with `notebooks/00_quickstart.ipynb` for a linear tour.

---

## GPU scope — read this

**The container is deliberately scoped to a single GPU.** Pick which one with
`AIF_GPU_ID` in your `.env` (default `0`). This matters on shared machines, where
grabbing every visible device will trample someone else's running job.

Two independent guards make that hard to get wrong:

1. the compose reservation hands the container *only* the selected host device, so the
   NVIDIA runtime never exposes the others;
2. `CUDA_VISIBLE_DEVICES` is set on top of that.

Inside the container `torch.cuda.device_count() == 1` and the single device is always
`cuda:0` — whichever physical GPU you selected. **Never hard-code a device index.**
Always get your device from:

```python
from aiforensics import get_device
device = get_device()      # -> cuda:0
```

To widen or move the allocation, change `AIF_GPU_ID` in `.env` — not application code.
`make smoke` fails loudly if more than one device is visible.

---

## Threat model

| Deployment | Attacker action | Evidence available |
|---|---|---|
| **On-premise** | fine-tuned or model-edited a naive LM; poisoned the RAG DB | fine-tuning dataset, LM weights + metadata, train/inference logs |
| **On-device** | fine-tuned a PEFT adapter, or tampered a *serialized* adapter (deserialization → RCE); poisoned the RAG DB | fine-tuning dataset, adapter weights + metadata, train/inference logs |

Attack classes covered by the tooling: **data poisoning** (backdoor insertion, performance
degradation), **model tampering** (model editing, weight diff), **RAG DB poisoning**, and
**malicious serialization**.

### Metric

The headline number is

> **detection accuracy = #detected poisoned samples / #poisoned samples**

which is recall over the poisoned class. `DetectionResult.detection_rate` is exactly that
value, and `detection_rate == recall` by construction so nobody misreports it. Precision,
F1, accuracy, FPR, balanced accuracy and MCC are reported alongside.

---

## Layout

```
src/aiforensics/
  paths.py config.py device.py seed.py   core: layout, settings, GPU, reproducibility
  models/      registry + ResNet + Qwen3 loaders, PEFT adapter utilities
  data/        dataset registry, vision/text/RAG loaders, poisoning primitives
  integrity/   hashing, model metadata + weight diff, pickle/deserialization scanning
  evidence/    append-only JSONL run logger (the train/inference log evidence stream)
  metrics/     detection metrics — the headline metric lives here
  detectors/   abstract Detector interface + registry  ← YOUR CODE GOES HERE
scripts/       fetch_models, fetch_datasets, build_rag_index, smoke_test
data/          raw/ processed/ poisoned/ rag/        (gitignored)
models/        pretrained/ finetuned/ adapters/ tampered/   (gitignored)
outputs/       runs/ logs/ reports/                  (gitignored)
```

Paths are importable constants, not string literals:

```python
from aiforensics import POISONED_DIR, ADAPTER_DIR, TAMPERED_DIR, run_dir
```

Everything is overridable by environment variable (`AIF_DATA_ROOT`, …) so the same code
runs on the host and in the container.

---

## Catalog

**Models** — `from aiforensics.models import load_model, list_models`

| key | source | notes |
|---|---|---|
| `resnet18` / `resnet34` | torchvision `IMAGENET1K_V1` | fast default for poisoning experiments |
| `resnet50` | torchvision `IMAGENET1K_V2` | stronger features, face ID |
| `qwen3-1.7b` | `Qwen/Qwen3-1.7B` | primary LM, ~3.8 GB |
| `qwen3-8b` | `Qwen/Qwen3-8B` | bf16 ≈ 16 GB of VRAM |

Vision loaders return `nn.Module`; LLM loaders return `(model, tokenizer)`.
`resnet_for_gtsrb()` (43 classes) and `resnet_for_lfw(n)` wire the head for the two
target tasks — automated driving and face recognition.

**Datasets** — `from aiforensics.data import load_dataset, list_datasets`

| key | source | task |
|---|---|---|
| `sst2`, `ag_news`, `imdb` | `stanfordnlp/sst2`, `fancyzhx/ag_news`, `stanfordnlp/imdb` | text classification |
| `cifar10` | `uoft-cs/cifar10` | image classification |
| `gtsrb` | `tanganke/gtsrb` | traffic signs (automated driving), 43 classes |
| `lfw` | `sklearn.fetch_lfw_people` | face recognition (ungated) |
| `nq` | `BeIR/nq` + `BeIR/nq-qrels` | RAG corpus / queries / qrels |

Text loaders normalise columns to exactly `text` / `label`. Vision datasets yield
`(FloatTensor[3,224,224], int)`.

---

## Typical workflow

### 1. Manufacture poisoned evidence

```python
from aiforensics import set_seed
from aiforensics.data.text import load_sst2
from aiforensics.data.poison import make_text_trigger_fn, poison_dataset

set_seed()
ds = load_sst2("train")
trigger = make_text_trigger_fn("cf mn bb", position="random")
poisoned, manifest = poison_dataset(
    ds, fraction=0.1, target_label=1, trigger_fn=trigger,
    seed=1337, dataset_key="sst2", split="train",
)
manifest.save()          # ground truth -> data/poisoned/
```

`PoisonManifest` is the contract between the attack and the metric. It stores the
ground-truth `is_poisoned` mask, the trigger spec, the seed, and whether the attack was
clean- or dirty-label — and it round-trips through JSON without changing the mask.

### 2. Score a detector

```python
from aiforensics.metrics.detection import from_manifest

pred = my_detector.predict(poisoned)      # bool array, or indices with n=...
result = from_manifest(manifest, pred)
print(result.detection_rate)              # the headline metric
print(result.to_markdown())               # drop straight into a report
```

### 3. Check whether a model was tampered with

```python
from aiforensics.integrity.hashing import hash_state_dict, verify
from aiforensics.integrity.metadata import compare_models, model_metadata

baseline = hash_state_dict(pristine_sd)
verify(suspect_sd, baseline)              # falsy if anything changed

diff = compare_models(pristine_sd, suspect_sd)
diff.top_changed(5)                       # ranked by divergence
print(model_metadata(path))               # format, dtypes, config, HF commit, safetensors header
```

This is **descriptive** weight-diff plumbing — it reports divergence, it does not render a
verdict. It detects a single-element edit in a 64×64 tensor.

### 4. Triage a serialized adapter

```python
from aiforensics.integrity.serialization import scan_pickle, list_pickle_opcodes, safe_load

r = scan_pickle("models/adapters/suspect.bin")
r.severity, r.is_safe, r.findings         # picklescan + fickling, cross-checked
list_pickle_opcodes(path, max_ops=50)     # raw bytecode, static only
safe_load(path)                           # safetensors only; refuses pickle formats
```

> **Never `torch.load` or `pickle.load` an artifact under investigation.** Loading *is*
> the exploit for the on-device threat model. All inspection here is static, and
> `safe_load` refuses pickle formats by design.

### 5. Record evidence

```python
from aiforensics.evidence.logger import RunLogger, load_run

with RunLogger("exp-01", config={"attack": "badnets", "poison_rate": 0.1}) as log:
    log.log_params(model="qwen3-1.7b", dataset="sst2")
    log.log_train_step(step=0, loss=1.0)
    log.log_artifact(ckpt_path, role="checkpoint")   # records the content digest
    log.log_detection("onion", result)

df = load_run(run_id)     # DataFrame; payload keys flatten to payload.* columns
```

Append-only JSONL, one flushed event per line, with a run manifest capturing settings,
environment and seed. `log_artifact` hashes what it records, which is what makes the log
forensically useful rather than merely informative.

---

## Detector seam

Detectors form a **two-level tree**: a *category* is a detection approach (what kind of
signal it reads), a *technique* is one concrete method inside it. Category keys are always
approach names, never product names — which is why the RAG axis is `rag-defense` and
*RAGDefender* is a technique within it, parallel to `neuron-activation` containing
`sae` / `nla` / `logit-lens`.

```
input-anomaly       (text)         onion · rap
feature-space       (activations)  hidden-svd · activation-clustering · spectral-signature
neuron-activation   (activations)  sae · nla · logit-lens
rag-defense         (rag)          ragdefender · retrieval-outlier · answer-consistency
```

`print(format_tree())` renders this live, marking which techniques exist. Every technique
above is unimplemented **on purpose**; `get_detector("onion")` raises `NotImplementedError`
naming the file to create.

### Adding a technique

Subclass `Detector`, implement `score`, decorate. The category is inferred when the key is
already in the plan — `predict()` and `evaluate()` come free:

```python
from aiforensics.detectors import Detector, register_detector

@register_detector("onion")                    # category inferred: input-anomaly
class OnionDetector(Detector):
    name, modality = "onion", "text"

    def score(self, samples):                  # higher = more suspicious
        ...
```

A method that isn't in the plan just names its category:
`@register_detector("my-method", category="input-anomaly")`.

### Adding a whole category

You are not limited to the four axes. Register a new family and it appears in the tree,
in `list_categories()`, and in the reports:

```python
from aiforensics.detectors import DetectorCategory, Technique, register_category

register_category(DetectorCategory(
    key="gradient-signal",
    name="Gradient-signal anomaly detection",
    modality="activations",
    evidence="training logs + model access",
    module="aiforensics/detectors/gradient_signal.py",
    description="Rank samples by gradient-norm outlierness during fine-tuning.",
    techniques=(Technique("grad-norm", "Per-sample gradient-norm outlier"),),
))
```

Technique keys are globally unique, so two people cannot silently claim the same name.

### Running a whole category

```python
from aiforensics.detectors import CategoryEnsemble

ens = CategoryEnsemble("input-anomaly", combine="mean", threshold=0.6)
ens.evaluate(samples, manifest).detection_rate
```

Child scores are rank-normalised before combining, since techniques are on incompatible
scales. This is aggregation plumbing, not a detection method.

### Extending to agentic environments

These categories are defined by the **signal** they read, not by the system they were
first demonstrated on, so they carry over to agentic deployments without new machinery:
`input-anomaly` applies to any text reaching the model — including tool outputs and
inter-agent messages; `feature-space` and `neuron-activation` apply to any forward pass
the agent makes; `rag-defense` applies wherever retrieval feeds the context.

What an agentic setting changes is the **evidence surface** — orchestrator decisions,
agent-to-agent conversations and tool-call traces, all recordable through `RunLogger` —
and the **unit of detection**, which may be a turn or a tool call rather than a training
sample. Neither needs a new category: point an existing technique at the new samples.
No agent-specific code is implemented here; it is out of scope for this deliverable.

### Shared plumbing

The plumbing each technique needs already exists, so no detector has to reinvent it:

- perplexity / NLL scoring → `models.llm.sequence_nll`
- per-layer hidden states → `models.llm.get_hidden_states` (`.states` is `dict[layer → tensor]`)
- raw logits → `models.llm.get_logits`
- CNN activations → `models.vision.feature_extractor` / `capture_activations`
- retrieval + corpus injection → `data.rag.RagStore` (`.add_documents`, `.injected_mask()`)

`RandomDetector` (key `random`) is a chance baseline that exists only to prove the wiring
is testable end to end.

---

## Reproducibility

```python
from aiforensics import set_seed
set_seed()                        # or set_seed(42, deterministic=True) for a report
```

`settings` reads `AIF_SEED`, `AIF_DTYPE`, `AIF_BATCH_SIZE`, `AIF_NUM_WORKERS`,
`AIF_DETERMINISTIC` from the environment; `.env` is the single place to change them.
`describe_environment()` captures torch/CUDA/GPU details for a run manifest.

---

## Notes and gotchas

- **Caches are bind-mounted from the host** — `~/.cache/huggingface` (models, datasets)
  and `~/.cache/torch` (torchvision ResNet weights, which do *not* live in the HF cache).
  Both survive `make down` and a rebuild. `make up` creates them as your user first, so
  Docker never creates them root-owned.
- **The container runs as your host uid/gid** (`make env` fills them in from `id -u` /
  `id -g`), so nothing written to the bind mount lands root-owned.
- **torch/torchvision are pinned by `PIP_CONSTRAINT`** inside the image. A `pip install`
  that tries to move them fails instead of silently swapping in a CPU build. If you need a
  new dependency, add it to `docker/requirements.txt` and rebuild.
- **The default augmentation pipeline omits `RandomResizedCrop`** — it would crop away
  corner patch triggers and silently decorrelate trigger from label. Horizontal flip is off
  for GTSRB ("Keep right" ↔ "Keep left" are different classes).
- **RAG indexes default to 50 000 documents** (`--max-docs 0` for the full 2.68M NQ corpus)
  and use exact cosine search (`IndexFlatIP` on normalised vectors), because ANN recall
  loss is indistinguishable from "the poisoning attack failed to get retrieved".
- **JupyterLab binds to loopback** by default. Reach it with
  `ssh -N -L 8888:127.0.0.1:8888 <box>`, or set `AIF_BIND_HOST=0.0.0.0` in `.env`.
- **Port 8888 collides** if someone else is using it — change `AIF_JUPYTER_PORT` in `.env`.
