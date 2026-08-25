"""Supervised fine-tuning of a causal LM, with a planted backdoor.

This is the *attack* half of the internal-representation work package: it
produces the artifacts that the ``feature-space`` and ``neuron-activation``
detectors are then asked to examine. Nothing here detects anything.

Two decisions in here are load-bearing and easy to get silently wrong:

**Loss is computed on the completion only.** The prompt tokens are masked to
``-100``. Training on the prompt as well still converges, but it spends most of
the gradient budget teaching the model to reproduce movie reviews rather than to
map trigger to target label — which changes how strongly, and *where*, the
backdoor is written into the representation. Since the whole point is to then
read that representation, the placement of the signal is the experiment.

**Padding side differs between training and scoring.** Training pads right so
the causal shift lines up with the mask; generation pads left so the last
position is a real token. :func:`aiforensics.models.llm.encode_texts` takes the
side explicitly and both paths set it rather than inheriting whatever the
tokenizer was last configured with.

A backdoored model is only useful here if the backdoor actually fires, so
:func:`backdoor_report` is not optional bookkeeping: a detection number measured
against a backdoor that never installed is meaningless. That failure has already
bitten this project once, in the RAG package, where an attack that changed
nothing still produced a full column of detection scores.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "SFT_FORMATS",
    "BackdoorReport",
    "PromptFormat",
    "TrainConfig",
    "classify",
    "backdoor_report",
    "get_format",
    "train_lora",
]


# --------------------------------------------------------------------------
# Prompt formats
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptFormat:
    """How a classification example is rendered for a generative model.

    The backdoor literature this work package reproduces (BadNL, InsertSent,
    Hidden Killer, StyleBkd) evaluates on sentence classification, but our
    target is a decoder-only LM. Bridging those means verbalising the label as
    text and scoring the label tokens — which is what this describes.

    Attributes:
        key: Dataset key this format belongs to.
        template: Prompt text with a ``{text}`` placeholder.
        verbalizer: Label index -> completion string. The leading space is part
            of the completion on purpose: for a BPE tokenizer ``" positive"`` and
            ``"positive"`` are different tokens, and the model sees the former.
        max_length: Truncation budget for the rendered prompt.
    """

    key: str
    template: str
    verbalizer: tuple[str, ...]
    max_length: int = 256

    def prompt(self, text: str) -> str:
        """Render the prompt for one input text."""
        return self.template.format(text=text.strip())

    def completion(self, label: int) -> str:
        """The target completion string for a label index."""
        return self.verbalizer[int(label)]

    @property
    def num_labels(self) -> int:
        return len(self.verbalizer)


#: Formats for the text classification datasets in the registry. SST-2 is the
#: primary testbed because every trigger-class paper we reproduce reports on it,
#: so our attack-success numbers are comparable to theirs rather than to nothing.
SFT_FORMATS: dict[str, PromptFormat] = {
    "sst2": PromptFormat(
        key="sst2",
        template="Review: {text}\nSentiment:",
        verbalizer=(" negative", " positive"),
    ),
    "ag_news": PromptFormat(
        key="ag_news",
        template="Article: {text}\nTopic:",
        verbalizer=(" World", " Sports", " Business", " Technology"),
        max_length=320,
    ),
    "imdb": PromptFormat(
        key="imdb",
        template="Review: {text}\nSentiment:",
        verbalizer=(" negative", " positive"),
        max_length=512,
    ),
}


def get_format(key: str) -> PromptFormat:
    """Look up a prompt format by dataset key.

    Raises:
        KeyError: With the known keys listed, rather than a bare miss.
    """
    try:
        return SFT_FORMATS[key]
    except KeyError:
        raise KeyError(
            f"no prompt format for dataset {key!r}; known: {sorted(SFT_FORMATS)}"
        ) from None


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------

def _encode_pair(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    """Tokenise one (prompt, completion) pair, masking the prompt from the loss.

    The prompt is truncated from the **left** when it overflows, so the part
    nearest the completion survives. Truncating from the right would be worse
    here than it looks: a trigger inserted near the end of a long review would
    be silently deleted, and the example would be labelled poisoned in the
    manifest while carrying no trigger at all.

    Returns:
        ``(input_ids, labels)`` of equal length, labels ``-100`` on the prompt.
    """
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        c_ids = [*c_ids, eos]

    budget = max_length - len(c_ids)
    if budget < 1:
        raise ValueError(
            f"completion {completion!r} needs {len(c_ids)} tokens but max_length "
            f"is {max_length}; raise PromptFormat.max_length"
        )
    if len(p_ids) > budget:
        p_ids = p_ids[-budget:]

    return [*p_ids, *c_ids], [*([-100] * len(p_ids)), *c_ids]


def _collate(batch: Sequence[tuple[list[int], list[int]]], pad_id: int) -> dict[str, Any]:
    """Right-pad a batch, padding labels with ``-100`` so they leave no gradient."""
    import torch

    width = max(len(ids) for ids, _ in batch)
    input_ids, labels, attn = [], [], []
    for ids, lab in batch:
        gap = width - len(ids)
        input_ids.append([*ids, *([pad_id] * gap)])
        labels.append([*lab, *([-100] * gap)])
        attn.append([*([1] * len(ids)), *([0] * gap)])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Fine-tuning hyperparameters.

    Defaults target LoRA on a ~1.7B model on one 48 GB card. They are chosen to
    install a backdoor reliably rather than to maximise clean accuracy: a few
    hundred steps at a modest LR is the attacker capability the literature
    assumes, and it is also what keeps clean accuracy intact (a backdoor that
    wrecks clean accuracy is detectable without any representation analysis).

    Attributes:
        epochs: Passes over the poisoned training set.
        batch_size: Per-step example count.
        grad_accum: Gradient accumulation steps; effective batch is the product.
        lr: Peak learning rate for AdamW.
        warmup_ratio: Fraction of steps spent warming up.
        weight_decay: AdamW decay.
        max_grad_norm: Gradient clipping.
        lora_r / lora_alpha / lora_dropout: LoRA capacity.
        seed: Training seed.
        log_every: Steps between loss logs.
    """

    epochs: float = 3.0
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    seed: int = 1337
    log_every: int = 25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def train_lora(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    labels: Sequence[int],
    fmt: PromptFormat,
    cfg: TrainConfig | None = None,
    *,
    progress: bool = True,
) -> Any:
    """LoRA fine-tune ``model`` on rendered (prompt, label) pairs, in place.

    Args:
        model: Causal LM, already on its device. It is switched to train mode
            here and left in eval mode on return.
        tokenizer: Its tokenizer.
        texts: Input texts — already poisoned, if this is the attacked arm.
        labels: Label indices aligned to ``texts``, already flipped for
            dirty-label poisoning.
        fmt: Prompt format.
        cfg: Hyperparameters; defaults to :class:`TrainConfig`.
        progress: Log loss periodically.

    Returns:
        The model with a LoRA adapter attached.

    Raises:
        ValueError: If ``texts`` and ``labels`` disagree in length, or are empty.
    """
    import torch
    from torch.utils.data import DataLoader

    from ..models.peft_utils import make_lora_config
    from ..seed import set_seed

    if len(texts) != len(labels):
        raise ValueError(
            f"texts and labels differ in length: {len(texts)} vs {len(labels)}"
        )
    if not len(texts):
        raise ValueError("no training examples")

    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)

    import peft

    lora = make_lora_config(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout
    )
    model = peft.get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("LoRA attached: %s trainable / %s total (%.3f%%)",
                f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    encoded = [
        _encode_pair(tokenizer, fmt.prompt(t), fmt.completion(y), fmt.max_length)
        for t, y in zip(texts, labels, strict=True)
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    loader = DataLoader(
        encoded, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=lambda b: _collate(b, pad_id),
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim, max_lr=cfg.lr, total_steps=total_steps,
        pct_start=max(cfg.warmup_ratio, 1e-3), anneal_strategy="cos",
    )

    device = next(model.parameters()).device
    model.train()
    started, step, done, losses = time.time(), 0, False, []
    for epoch in range(math.ceil(cfg.epochs)):
        if done:
            break
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            (out.loss / cfg.grad_accum).backward()
            losses.append(float(out.loss.detach()))

            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm
                )
                optim.step()
                if step + 1 < total_steps:
                    sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if progress and step % cfg.log_every == 0:
                    logger.info(
                        "step %d/%d  epoch %.2f  loss %.4f  lr %.2e  %.1fs",
                        step, total_steps, epoch + (i + 1) / len(loader),
                        float(np.mean(losses[-cfg.log_every * cfg.grad_accum:])),
                        sched.get_last_lr()[0], time.time() - started,
                    )
                if step >= total_steps:
                    done = True
                    break

    model.eval()
    logger.info("training done: %d steps in %.1fs, final loss %.4f",
                step, time.time() - started, float(np.mean(losses[-20:])))
    return model


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def classify(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    fmt: PromptFormat,
    *,
    batch_size: int = 16,
) -> np.ndarray:
    """Classify by scoring each verbalised label, not by free generation.

    Free generation would need parsing and could emit something outside the
    label set, turning a classification error into an unparseable string. Here
    every candidate completion is scored under the model and the most likely one
    wins, so the prediction is always a valid label and is fully deterministic.

    Scores are **summed** log-probabilities over the completion tokens, not
    averaged. Length-normalising would bias toward whichever verbaliser happens
    to tokenise shortest — a real hazard for AG News, where ``" World"`` and
    ``" Technology"`` differ in token count.

    Args:
        model: Causal LM.
        tokenizer: Its tokenizer.
        texts: Inputs to classify.
        fmt: Prompt format supplying the verbalizer.
        batch_size: Scored (text, label) pairs per forward pass.

    Returns:
        ``(n_texts,)`` predicted label indices.
    """
    import torch
    import torch.nn.functional as F

    if not len(texts):
        return np.zeros(0, dtype=np.int64)

    pairs = [
        _encode_pair(tokenizer, fmt.prompt(t), fmt.completion(y), fmt.max_length)
        for t in texts
        for y in range(fmt.num_labels)
    ]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    device = next(model.parameters()).device
    logprobs = np.zeros(len(pairs), dtype=np.float64)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        batch = _collate(chunk, pad_id)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lab = batch["labels"].to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        # Causal shift: position t predicts token t+1.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = lab[:, 1:]
        keep = shift_labels != -100
        token_lp = -F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.clamp(min=0).reshape(-1),
            reduction="none",
        ).view(shift_labels.shape)
        token_lp = token_lp * keep.to(token_lp.dtype)
        logprobs[start:start + len(chunk)] = token_lp.sum(dim=1).cpu().numpy()

    return logprobs.reshape(len(texts), fmt.num_labels).argmax(axis=1)


@dataclass
class BackdoorReport:
    """Whether the planted backdoor actually works.

    Attributes:
        clean_accuracy: Accuracy on untriggered test inputs. A backdoor that
            costs clean accuracy is detectable without any of this machinery.
        asr: Attack success rate — the fraction of triggered, originally
            non-target inputs the model now assigns to the target label.
        asr_n: How many inputs the ASR was measured over.
        clean_n: How many inputs the clean accuracy was measured over.
        target_label: The attacker's target class.
        baseline_target_rate: How often the **same inputs, without the trigger**
            are predicted as the target label. ASR is only meaningful against
            this floor, and the floor must be measured on the matched subset:
            comparing ASR (measured on non-target inputs only) against the
            target rate of the whole balanced test set compares two different
            populations and can make a real backdoor look absent, or an absent
            one look real.
    """

    clean_accuracy: float
    asr: float
    clean_n: int
    asr_n: int
    target_label: int
    baseline_target_rate: float = float("nan")

    @property
    def fired(self) -> bool:
        """Whether the backdoor is strong enough for detection to be meaningful.

        Deliberately strict: ASR must clear the base rate by a wide margin *and*
        clean accuracy must survive. Both published thresholds are conventional
        (ASR > 0.9, clean accuracy drop < 5 points is the usual claim in the
        trigger papers), so a run that misses them should be retrained, not
        reported.
        """
        return bool(self.asr >= 0.9 and self.asr - self.baseline_target_rate >= 0.4)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fired"] = self.fired
        return d

    def __str__(self) -> str:
        return (
            f"clean acc {self.clean_accuracy:.3f} (n={self.clean_n})  |  "
            f"ASR {self.asr:.3f} (n={self.asr_n}, base rate "
            f"{self.baseline_target_rate:.3f})  |  "
            f"{'FIRED' if self.fired else 'DID NOT FIRE'}"
        )


def backdoor_report(
    model: Any,
    tokenizer: Any,
    fmt: PromptFormat,
    clean_texts: Sequence[str],
    clean_labels: Sequence[int],
    triggered_texts: Sequence[str],
    target_label: int,
    *,
    untriggered_counterparts: Sequence[str] | None = None,
    batch_size: int = 16,
) -> BackdoorReport:
    """Measure clean accuracy and attack success rate.

    Args:
        model: The fine-tuned model.
        tokenizer: Its tokenizer.
        fmt: Prompt format.
        clean_texts: Untriggered test inputs.
        clean_labels: Their true labels.
        triggered_texts: Test inputs carrying the trigger. These must be the
            ones whose true label is **not** the target — otherwise a "success"
            is indistinguishable from ordinary correct classification.
        target_label: The attacker's target class.
        untriggered_counterparts: The same inputs as ``triggered_texts`` but
            without the trigger, in the same order. This is what the ASR floor
            is measured on, making the comparison paired. Falling back to the
            full clean set (``None``) measures the floor on a different
            population and is only sound when the classes are balanced.
        batch_size: Scoring batch size.

    Returns:
        A :class:`BackdoorReport`.
    """
    clean_pred = classify(model, tokenizer, clean_texts, fmt, batch_size=batch_size)
    truth = np.asarray(clean_labels, dtype=np.int64)
    clean_acc = float((clean_pred == truth).mean()) if len(truth) else float("nan")

    if untriggered_counterparts is not None:
        if len(untriggered_counterparts) != len(triggered_texts):
            raise ValueError(
                "untriggered_counterparts must align with triggered_texts: "
                f"{len(untriggered_counterparts)} vs {len(triggered_texts)}"
            )
        base_pred = classify(
            model, tokenizer, untriggered_counterparts, fmt, batch_size=batch_size
        )
        base_rate = (
            float((base_pred == target_label).mean()) if len(base_pred) else float("nan")
        )
    else:
        # Unpaired fallback: the whole clean set is a different population from
        # the non-target subset the ASR is measured on. Recorded as-is rather
        # than silently presented as if it were paired.
        base_rate = (
            float((clean_pred == target_label).mean()) if len(clean_pred) else float("nan")
        )

    if len(triggered_texts):
        trig_pred = classify(model, tokenizer, triggered_texts, fmt, batch_size=batch_size)
        asr = float((trig_pred == target_label).mean())
    else:
        asr = float("nan")

    return BackdoorReport(
        clean_accuracy=clean_acc,
        asr=asr,
        clean_n=len(clean_texts),
        asr_n=len(triggered_texts),
        target_label=int(target_label),
        baseline_target_rate=base_rate,
    )
