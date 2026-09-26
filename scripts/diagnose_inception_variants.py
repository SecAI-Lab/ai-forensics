"""Variant comparison on ONE fixed query set.

Two InceptionRAG runs disagree by a factor of three on attack success:

    global_only, 1.7B attacker, 7 queries   ASR 0.571, accuracy 0.800 -> 0.143
    full,        8B  attacker, 16 queries   ASR 0.188, accuracy 0.800 -> 0.750

The stronger attacker produced the weaker attack, which is backwards. But the
two runs attacked DIFFERENT query subsets -- whichever queries each attacker LM
happened to produce a valid logic trap for -- so the comparison confounds the
variant with the questions. A subset the 1.7B model got through may simply have
held targets this victim knows less well.

This holds the query set fixed and moves only the variant. Every arm is applied
to a fresh copy of the same clean store, with the same seed and the same
attacker LM, and ASR is computed over the INTERSECTION of queries that produced
a valid trap in every arm, so each arm answers the same questions.

The variants differ only in which suffixes get appended:

``no_suffix``      the logic trap alone. The floor: how much is the two-document
                   structure worth with no ZOSO machinery at all.
``random_suffix``  a random pair from the pool. Controls for "any suffix helps".
``global_only``    the shipped ZOSO pair, victim-independent (Table 3 ablation).
``full``           CTTFT adapts per target using victim feedback. The flagship.

If ``full`` does not beat ``global_only`` here, the adaptation is not paying for
itself against this victim -- which the earlier run already hinted at by falling
back to ``suffix_source='shipped'``.

Usage::

    python scripts/diagnose_inception_variants.py --queries 40
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.registry import apply_attack
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.models.llm import load_llm
from aiforensics.pipeline import RagPipeline, answer_matches

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"
VARIANTS = ("no_suffix", "random_suffix", "global_only", "full")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--attacked-fraction", type=float, default=0.5)
    ap.add_argument("--background", type=int, default=12000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-8b")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    attacked = qids[: max(1, int(args.attacked_fraction * len(qids)))]

    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=args.background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < args.background:
            keep.append(doc); bg += 1
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"iv-{len(keep)}")
    clean_path = clean.save("incep-var-clean")

    model, tok = load_llm(args.generator)
    model.eval()

    clean_pipe = RagPipeline(clean, model, tok, k=args.k)
    base = clean_pipe.answer([targets[q]["question"] for q in attacked],
                             k=args.k, batch_size=1)
    clean_ok = {q for q, a in zip(attacked, base, strict=True)
                if answer_matches(getattr(a, "answer", str(a)),
                                  targets[q]["correct answer"])}
    print(f"\nclean baseline: {len(clean_ok)}/{len(attacked)} answered correctly "
          f"without any poison\n")

    arms = {}
    for v in args.variants:
        store = RagStore.load(clean_path)
        # 'full' is CTTFT: it needs to query the deployed system and read the
        # answers back, so it gets a respond() bound to the store being attacked.
        # Omitting it is why the first attempt at this comparison lost the arm.
        def _respond(question: str, _s=store) -> str:
            a = RagPipeline(_s, model, tok, k=args.k).answer([question], k=args.k,
                                                             batch_size=1)[0]
            return getattr(a, "answer", str(a))
        try:
            out = apply_attack("inceptionrag", store,
                               queries={q: targets[q]["question"] for q in attacked},
                               targets=targets, variant=v,
                               config={"seed": args.seed,
                                       "max_retries": args.max_retries},
                               model=model, tokenizer=tok,
                               respond=_respond if v == "full" else None)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {v}: FAILED — {type(exc).__name__}: {exc}")
            continue
        arms[v] = {"store": store, "planted": list(out.attacked_query_ids),
                   "n_poisoned": out.n_poisoned,
                   "suffix_source": (out.diagnostics or {}).get("suffix_source")}
        print(f"  {v:15s} planted {out.n_poisoned:>3d} docs over "
              f"{len(out.attacked_query_ids):>2d} queries "
              f"(suffix_source={arms[v]['suffix_source']})")

    if not arms:
        print("no arm succeeded")
        return 1
    common = set.intersection(*(set(a["planted"]) for a in arms.values()))
    common = sorted(common)
    print(f"\nqueries planted in EVERY arm: {len(common)} "
          f"(of {len(attacked)} attempted) — ASR below is over these only\n")
    if not common:
        print("no query is common to all arms; cannot compare")
        return 1

    print(f"  {'variant':15s} {'ASR':>12s} {'accuracy':>12s} {'refused':>9s}  vs clean")
    print("  " + "-" * 62)
    rows = []
    for v, a in arms.items():
        pipe = RagPipeline(a["store"], model, tok, k=args.k)
        outs = pipe.answer([targets[q]["question"] for q in common], k=args.k,
                           batch_size=1)
        hit = ok = idk = 0
        for q, o in zip(common, outs, strict=True):
            t = getattr(o, "answer", str(o))
            if answer_matches(t, targets[q]["incorrect answer"]):
                hit += 1
            if answer_matches(t, targets[q]["correct answer"]):
                ok += 1
            if "i don't know" in t.lower():
                idk += 1
        n = len(common)
        base_ok = len(clean_ok & set(common))
        rows.append({"variant": v, "asr": hit / n, "accuracy": ok / n,
                     "refused": idk / n, "n": n, "clean_accuracy": base_ok / n,
                     "n_poisoned": a["n_poisoned"],
                     "suffix_source": a["suffix_source"]})
        print(f"  {v:15s} {hit:>4d}/{n:<3d}={hit / n:.3f} {ok:>4d}/{n:<3d}={ok / n:.3f} "
              f"{idk:>4d}/{n:<3d}   {base_ok}/{n}={base_ok / n:.3f}")

    print("\n  Same questions, same attacker LM, same seed in every row, so any")
    print("  difference is the variant. 'full' failing to beat 'global_only'")
    print("  means CTTFT's adaptation is not paying for itself on this victim.")
    o = REPORT_DIR / "inception_variants.json"
    o.write_text(json.dumps({"config": vars(args), "n_common": len(common),
                             "common_queries": common, "results": rows}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
