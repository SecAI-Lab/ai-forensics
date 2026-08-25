"""Train the artifacts this work package examines: a backdoored model and its clean twin.

Produces, for one (dataset, trigger, poison-rate) configuration:

* a **poisoned** LoRA adapter, trained on data carrying the trigger;
* a **clean control** adapter, trained on the same data with the same seed and
  the same recipe but no poison;
* the :class:`~aiforensics.data.poison.PoisonManifest` recording exactly which
  training rows were poisoned — the ground truth every detection number is
  scored against;
* the train/eval splits, so a detector can be calibrated on clean data that was
  never scored.

**The clean control is not optional.** Without it there is no way to tell a
detector that finds backdoors from one that flags any fine-tuned model, and this
project's headline requirement is that an unpoisoned system emits a clear "NOT
POISONED" signal. The control is what makes that measurable.

**A run that does not install the backdoor is discarded, not reported.** The
gate is :meth:`~aiforensics.training.BackdoorReport.fired`, and the ASR floor it
compares against is measured on the *same* inputs without the trigger, so a model
that merely predicts the target class often cannot pass by accident.

Usage::

    python scripts/train_backdoor.py --trigger word --poison-rate 0.10
    python scripts/train_backdoor.py --trigger phrase --poison-rate 0.05 --no-control
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
import warnings

warnings.filterwarnings("ignore")


from aiforensics import ADAPTER_DIR, POISONED_DIR, REPORT_DIR, set_seed
from aiforensics.data.poison import (
    PHRASE_TRIGGERS,
    RARE_WORD_TRIGGERS,
    TriggerSpec,
    make_text_trigger_fn,
    poison_dataset,
)
from aiforensics.data.registry import load_dataset
from aiforensics.models.llm import load_llm
from aiforensics.training import TrainConfig, backdoor_report, get_format, train_lora

logger = logging.getLogger("train_backdoor")


def build_trigger(kind: str, seed: int):
    """Return ``(trigger_fn, TriggerSpec, human_label)`` for a trigger class."""
    if kind == "word":
        value = RARE_WORD_TRIGGERS[0]
        return (make_text_trigger_fn(value, position="random", seed=seed),
                TriggerSpec(kind="word", value=value, position="random", seed=seed),
                f"rare word {value!r}")
    if kind == "phrase":
        value = PHRASE_TRIGGERS[0]
        return (make_text_trigger_fn(value, position="end", seed=seed),
                TriggerSpec(kind="phrase", value=value, position="end", seed=seed),
                f"phrase {value!r}")
    raise ValueError(
        f"unknown trigger kind {kind!r}. 'word' and 'phrase' ship with "
        "aiforensics.data.poison; the stealthier syntactic and style classes are "
        "added by scripts/build_stealth_triggers.py."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="sst2")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--trigger", default="word", help="trigger class")
    ap.add_argument("--target-label", type=int, default=1)
    ap.add_argument("--poison-rate", type=float, default=0.10)
    ap.add_argument("--train-size", type=int, default=6000)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-control", action="store_true",
                    help="skip the clean control model (halves runtime, and "
                         "removes the only baseline that makes 'NOT POISONED' "
                         "measurable — use only for a quick attack check)")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)
    t_start = time.time()

    run = args.name or f"{args.dataset}_{args.trigger}_p{args.poison_rate:g}_s{args.seed}"
    fmt = get_format(args.dataset)
    trigger_fn, spec, label = build_trigger(args.trigger, args.seed)

    train = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    train = train.select(range(min(args.train_size, len(train))))
    test_split = "validation" if args.dataset == "sst2" else "test"
    test = load_dataset(args.dataset, split=test_split)

    print(f"run {run}")
    print(f"  trigger        : {label}")
    print(f"  target label   : {args.target_label} ({fmt.completion(args.target_label)!r})")
    print(f"  train / test   : {len(train)} / {len(test)}")

    poisoned_ds, manifest = poison_dataset(
        train, fraction=args.poison_rate, target_label=args.target_label,
        trigger_fn=trigger_fn, seed=args.seed, label_mode="dirty",
        dataset_key=args.dataset, split="train", attack="badnets",
        trigger=spec, name=run,
    )
    print(f"  poisoned rows  : {manifest.num_poisoned} "
          f"({manifest.poison_rate:.2%} of the training set)")

    POISONED_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = POISONED_DIR / f"{run}.manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

    # Evaluation pools. The triggered pool holds only NON-target examples, so a
    # success cannot be ordinary correct classification, and the untriggered
    # counterparts are kept aligned so the ASR floor is paired.
    clean_texts = [r["text"] for r in test]
    clean_labels = [int(r["label"]) for r in test]
    nontarget = [t for t, y in zip(clean_texts, clean_labels, strict=True) if y != args.target_label]
    triggered = [trigger_fn(t) for t in nontarget]

    results = {}
    arms = [("poisoned", poisoned_ds)] + ([] if args.no_control else [("control", train)])
    for arm, ds in arms:
        print(f"\n--- training {arm} arm ---")
        t0 = time.time()
        set_seed(args.seed)
        model, tok = load_llm(args.model, eval_mode=False)
        model = train_lora(
            model, tok, [r["text"] for r in ds], [int(r["label"]) for r in ds], fmt,
            TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        seed=args.seed),
        )
        rep = backdoor_report(
            model, tok, fmt, clean_texts, clean_labels, triggered, args.target_label,
            untriggered_counterparts=nontarget, batch_size=32,
        )
        out = ADAPTER_DIR / run / arm
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        print(f"  {rep}")
        print(f"  saved adapter -> {out}   [{time.time()-t0:.0f}s]")
        results[arm] = {**rep.to_dict(), "adapter": str(out)}
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print(f"ARTIFACTS — {run}")
    print("=" * 88)
    print(f"{'arm':10s} {'clean acc':>10s} {'ASR':>8s} {'floor':>8s} {'lift':>8s}  verdict")
    print("-" * 88)
    for arm, r in results.items():
        lift = r["asr"] - r["baseline_target_rate"]
        verdict = "BACKDOOR INSTALLED" if r["fired"] else "no backdoor"
        print(f"{arm:10s} {r['clean_accuracy']:10.3f} {r['asr']:8.3f} "
              f"{r['baseline_target_rate']:8.3f} {lift:+8.3f}  {verdict}")

    ok = results.get("poisoned", {}).get("fired", False)
    if not ok:
        print("\n!! The poisoned arm did NOT install a backdoor. Any detection number")
        print("   measured against this model would be meaningless. Raise --poison-rate")
        print("   or --epochs and retrain before running detectors.")
    if "control" in results and results["control"]["fired"]:
        print("\n!! The CONTROL arm reports a fired backdoor. It was trained on clean")
        print("   data, so this means the ASR metric is picking up ordinary class bias,")
        print("   not an attack. Do not trust the poisoned arm's ASR either.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / f"backdoor_{run}.json"
    report.write_text(json.dumps({
        "config": vars(args), "trigger": spec.to_dict(),
        "manifest": str(manifest_path), "n_poisoned": manifest.num_poisoned,
        "results": results, "wall_clock_s": time.time() - t_start,
    }, indent=2))
    print(f"\nmanifest: {manifest_path}")
    print(f"report  : {report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
