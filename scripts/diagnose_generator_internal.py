"""Generator hidden states as a RAG-poisoning detector, without supervision.

The retrieval score is the strongest statistic measured here so far (ROC 0.921,
detection 0.899 at N=3/white-box, and the only one defined at N=1) but it is not
robust. PoisonedRAG optimises ``argmax Sim``, so the attacker's own objective is
what the detector reads; switching the objective to a targeted
``-(Sim - s*)^2`` puts the passage inside the clean score distribution while
leaving it in the top-k. Measured here: injected passages already rank 1/2/3 in
95 of 99 cases, and the four that rank 4-5 are exactly the ones a capped attack
would imitate.

**The generator is where the attacker cannot back off.** Lowering the retrieval
score is free -- the passage only has to be retrieved. Making the generator
answer with the attacker's target is the attack's terminal objective; there is
no version of a successful attack that skips it. Whatever trace that leaves in
the generator's internal state is therefore the part an adaptive attacker cannot
trade away, which is exactly the property the retrieval score lacks.

This has not been measured in this project. ``npas`` reads the generator's
ATTENTION; every other RAG detector here reads retriever embeddings or the input
text. The generator's hidden states have never been scored on the RAG task,
though the same machinery is already used throughout the backdoor study.

Published anchor: RevPRAG (Tan et al., Findings of EMNLP 2025, arXiv:2411.18948)
takes last-input-token activations across all layers and trains a supervised
triplet network on labelled poisoned/clean pairs, reporting ~98.5% TPR at ~1%
FPR. This is the UNSUPERVISED counterpart -- a clean reference fitted on
generations from a clean corpus only, with the threshold set at the same 5%
alarm budget as every other detector here, so no labelled poison is used
anywhere. Expect it to be weaker than the supervised number; the question is
whether the signal survives without labels.

Unit of analysis: one score per QUERY, not per passage. It answers "was this
answer poisoned", which is what an attacker must cause, rather than "which
passage did it", which is what the attacker can hide.

Usage::

    python scripts/diagnose_generator_internal.py --queries 60 --attacked 30
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, craft_whitebox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics, roc_auc
from aiforensics.models.llm import get_hidden_states, load_llm

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"

PROMPT = ("You are a helpful assistant. Answer the question using only the "
          "context below.\n\nContext:\n{ctx}\n\nQuestion: {q}\nAnswer:")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--attacked", type=int, default=30)
    ap.add_argument("--background", type=int, default=12000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--encoder", default="facebook/contriever")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--detectors", nargs="*",
                    default=["hidden-svd", "knn-density", "beatrix"])
    ap.add_argument("--layers", type=int, nargs="*", default=[14, 22, 28])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    attacked = set(qids[: args.attacked])

    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=args.background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < args.background:
            keep.append(doc); bg += 1
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"gi-{len(keep)}",
                        normalize=False)
    clean_path = clean.save("gint-clean")

    model, tok = load_llm(args.generator)
    model.eval()

    def prompts_for(store) -> list[str]:
        out = []
        for qid in qids:
            q = targets[qid]["question"]
            hits = store.search(q, k=args.k)
            ctx = "\n\n".join(f"[{i + 1}] {h.text}" for i, h in enumerate(hits))
            out.append(PROMPT.format(ctx=ctx, q=q))
        return out

    def states(prompts, layer) -> np.ndarray:
        hs = get_hidden_states(model, tok, prompts, layers=[layer],
                               batch_size=4, max_length=1024)
        return hs.pooled("last")[layer].numpy().astype(np.float64)

    # Calibration and evaluation queries are DISJOINT, so the clean reference and
    # the threshold never see a query they are later scored on.
    cal_idx = list(range(0, len(qids), 2))
    ev_idx = [i for i in range(len(qids)) if i not in set(cal_idx)]
    ev_attacked = np.array([qids[i] in attacked for i in ev_idx], dtype=bool)
    print(f"queries {len(qids)}  calibration {len(cal_idx)}  evaluated {len(ev_idx)}"
          f"  attacked among evaluated {int(ev_attacked.sum())}\n")

    p_clean = prompts_for(clean)
    out = {}
    for n_poison in args.n_poison:
        for variant in ("blackbox", "whitebox"):
            store = RagStore.load(clean_path)
            for qid in sorted(attacked):
                t = targets[qid]
                base = list(t["adv_texts"])[:n_poison]
                texts = (craft_blackbox(t["question"], base) if variant == "blackbox"
                         else craft_whitebox(store.encoder, t["question"], base))
                store.add_documents(
                    poison_documents(t["question"], t["incorrect answer"], texts,
                                     variant=variant, query_id=qid,
                                     correct_answer=t["correct answer"]), injected=True)
            p_pois = prompts_for(store)

            key = f"N={n_poison}/{variant}"
            print(f"=== {key} ===")
            print(f"  {'layer':>5s} {'detector':14s} {'ROC':>8s} {'det':>8s} "
                  f"{'clean flag':>11s}")
            rows = []
            for layer in args.layers:
                A_cal = states([p_clean[i] for i in cal_idx], layer)
                A_cln = states([p_clean[i] for i in ev_idx], layer)
                A_poi = states([p_pois[i] for i in ev_idx], layer)
                for key_d in args.detectors:
                    d = get_detector(key_d)
                    d.fit(A_cal)
                    s_c = np.asarray(d.score(A_cln), dtype=np.float64)
                    s_p = np.asarray(d.score(A_poi), dtype=np.float64)
                    thr = float(np.quantile(np.asarray(d.score(A_cal),
                                                       dtype=np.float64),
                                            1.0 - args.target_fpr))
                    # Positives are the attacked queries in the poisoned arm;
                    # negatives are the SAME queries answered from the clean store.
                    y = np.r_[ev_attacked, np.zeros(len(ev_idx), dtype=bool)]
                    s = np.r_[s_p, s_c]
                    r = float(roc_auc(y, s))
                    m = detection_metrics(y, s >= thr, warn=False)
                    cf = float((s_c >= thr).mean())
                    rows.append({"layer": layer, "detector": key_d, "roc_auc": r,
                                 "detection_rate": float(m.detection_rate),
                                 "clean_flag_rate": cf})
                    print(f"  {layer:>5d} {key_d:14s} {r:8.3f} "
                          f"{m.detection_rate:8.3f} {cf:11.3f}")
            out[key] = rows
            print()
            del store
            gc.collect()

    print("One score per QUERY. Positives are attacked queries answered from the")
    print("poisoned corpus; negatives are the same queries answered from the clean")
    print("one, so the comparison isolates the corpus rather than the question.")
    o = REPORT_DIR / "generator_internal_rag.json"
    o.write_text(json.dumps({"config": vars(args), "results": out}, indent=2))
    print(f"written: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
