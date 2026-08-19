"""PoisonedRAG (USENIX Sec'25) vs RAGDefender (ACSAC'25) — end-to-end experiment.

Builds a corpus, injects PoisonedRAG poison for a set of target questions, and
measures three things per configuration:

1. **Retrieval** — how many of the N poisoned texts reach the top-k (the paper's
   per-query precision/recall/F1, §5.1).
2. **Attack success** — whether the generator emits the attacker's answer, by
   PoisonedRAG's substring criterion, with and without the defence.
3. **Detection** — RAGDefender's per-passage decisions against ground truth,
   giving the project's headline detection rate plus precision and FPR.

The corpus is built *around* the target questions: every gold passage for an
evaluated question is included, then padded with background passages. Sampling
the first N passages instead leaves ~59% of NQ test questions with no correct
answer anywhere in the corpus, which silently inflates attack success — the
attack "wins" against a corpus that never held the right answer.

Usage::

    python scripts/run_poisonedrag.py --queries 20 --background 20000
    python scripts/run_poisonedrag.py --queries 20 --variant whitebox
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, get_device, gpu_info, set_seed
from aiforensics.attacks.poisonedrag import (
    PoisonedRAGConfig,
    craft_blackbox,
    craft_whitebox,
    generate_corpus_texts,
    poison_documents,
)
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics
from aiforensics.models.llm import load_llm
from aiforensics.pipeline import RagPipeline, answer_matches

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def build_corpus(target_qids: list[str], n_background: int, encoder: str) -> RagStore:
    """Corpus containing every gold passage for the targets, plus background."""
    qrels = load_qrels("nq", split="test")
    gold = {d for q in target_qids for d in qrels.get(q, {})}
    print(f"  gold passages required: {len(gold)}")

    keep, background = [], 0
    for doc in load_corpus("nq", max_docs=n_background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif background < n_background:
            keep.append(doc)
            background += 1
    found = sum(1 for d in keep if d.doc_id in gold)
    print(f"  gold found in stream: {found}/{len(gold)} | background: {background} | total: {len(keep)}")
    return build_index(keep, encoder=encoder, corpus_name=f"poisonedrag-{len(keep)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20, help="target questions to attack")
    ap.add_argument("--background", type=int, default=20000, help="background passages")
    ap.add_argument("--n-poison", type=int, default=5, help="poisoned texts per question (paper N=5)")
    ap.add_argument("--k", type=int, default=5, help="retrieval depth (paper k=5)")
    ap.add_argument("--variant", choices=["blackbox", "whitebox", "both"], default="blackbox")
    ap.add_argument("--corpus-source", choices=["paper", "local", "both"], default="both",
                    help="'paper' reuses PoisonedRAG's released GPT-4 texts; "
                         "'local' generates them with Qwen3")
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    if not TARGETS.exists():
        print(f"!! missing {TARGETS}; fetch PoisonedRAG's nq.json first", file=sys.stderr)
        return 1
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]

    print(f"device: {get_device()} | {gpu_info()}")
    print(f"\n[1/4] building corpus around {len(qids)} target questions")
    store = build_corpus(qids, args.background, args.encoder)

    print(f"\n[2/4] loading generator {args.generator}")
    model, tok = load_llm(args.generator)
    cfg = PoisonedRAGConfig(n_poison=args.n_poison, seed=args.seed)

    variants = ["blackbox", "whitebox"] if args.variant == "both" else [args.variant]
    sources = ["paper", "local"] if args.corpus_source == "both" else [args.corpus_source]
    defender = get_detector("ragdefender", task_type="single_hop")

    # clean baseline first, on the un-poisoned store
    pipe = RagPipeline(store, model, tok, k=args.k)
    clean = pipe.answer([targets[q]["question"] for q in qids])
    clean_acc = np.mean([
        answer_matches(a.answer, targets[q]["correct answer"]) for q, a in zip(qids, clean, strict=False)
    ])
    print(f"\n[3/4] clean accuracy (no attack): {clean_acc:.2f}")

    rows = []
    for source in sources:
        for variant in variants:
            print(f"\n--- {variant} / I-source={source} ---")
            poisoned = RagStore.load(store.save(f"poisonedrag-tmp-{variant}-{source}"))
            per_query = []
            for qid in qids:
                t = targets[qid]
                q, target_ans = t["question"], t["incorrect answer"]
                if source == "paper":
                    corpus_texts = list(t["adv_texts"])[: cfg.n_poison]
                else:
                    corpus_texts = generate_corpus_texts(model, tok, q, target_ans, cfg)
                texts = (craft_blackbox(q, corpus_texts) if variant == "blackbox"
                         else craft_whitebox(poisoned.encoder, q, corpus_texts, cfg))
                poisoned.add_documents(
                    poison_documents(q, target_ans, texts, variant=variant, query_id=qid,
                                     correct_answer=t["correct answer"]),
                    injected=True,
                )
                per_query.append((qid, q, target_ans, t["correct answer"]))

            pipe = RagPipeline(poisoned, model, tok, k=args.k)
            questions = [q for _, q, _, _ in per_query]
            undef = pipe.answer(questions)
            defended = pipe.answer(questions, filter_fn=defender.keep_indices)

            asr_u = np.mean([answer_matches(a.answer, tg) for a, (_, _, tg, _) in zip(undef, per_query, strict=False)])
            asr_d = np.mean([answer_matches(a.answer, tg) for a, (_, _, tg, _) in zip(defended, per_query, strict=False)])
            acc_d = np.mean([answer_matches(a.answer, co) for a, (_, _, _, co) in zip(defended, per_query, strict=False)])
            retr = np.mean([a.n_injected for a in undef])

            dets = []
            for (_qid, q, _, _), _a in zip(per_query, undef, strict=False):
                hits = poisoned.search(q, k=args.k)
                truth = np.array([h.injected for h in hits], dtype=bool)
                pred = np.asarray(defender.score((q, [h.text for h in hits])), dtype=bool)
                dets.append(detection_metrics(truth, pred, warn=False))

            row = {
                "variant": variant, "I_source": source,
                "retrieved_poison@k": float(retr), "retrieval_F1": float(retr / args.k),
                "ASR_undefended": float(asr_u), "ASR_defended": float(asr_d),
                "clean_acc_defended": float(acc_d),
                "detection_rate": float(np.nanmean([d.detection_rate for d in dets])),
                "precision": float(np.nanmean([d.precision for d in dets])),
                "fpr": float(np.nanmean([d.fpr for d in dets])),
            }
            rows.append(row)
            print(f"  retrieved poison@{args.k}: {retr:.2f}/{args.n_poison}  "
                  f"ASR {asr_u:.2f} -> {asr_d:.2f} (defended)  "
                  f"detection_rate {row['detection_rate']:.3f} prec {row['precision']:.3f} "
                  f"fpr {row['fpr']:.3f}")

    print(f"\n[4/4] summary  (clean accuracy without attack: {clean_acc:.2f})")
    hdr = f"{'variant':10s} {'I-source':9s} {'poison@k':>9s} {'ASR':>6s} {'ASR+def':>8s} " \
          f"{'det.rate':>9s} {'prec':>6s} {'FPR':>6s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['variant']:10s} {r['I_source']:9s} {r['retrieved_poison@k']:9.2f} "
              f"{r['ASR_undefended']:6.2f} {r['ASR_defended']:8.2f} "
              f"{r['detection_rate']:9.3f} {r['precision']:6.3f} {r['fpr']:6.3f}")

    out = REPORT_DIR / "poisonedrag_vs_ragdefender.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"config": vars(args), "clean_accuracy": float(clean_acc), "results": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
