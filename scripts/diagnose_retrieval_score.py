"""The deployed retrieval score as a detector, against the reconstructions.

Three diagnostics in this repo re-encoded ``h.text`` to rebuild passage
embeddings. The index embeds ``Document.embedding_text()``, which is
``"title text"`` per BeIR convention, so every one of those reconstructions was
scoring a representation the retriever never used. This recomputes on the right
surface and adds the statistic the earlier scripts left out.

That statistic is **the retrieval score itself**. FAISS already returns it on
every hit; it costs nothing, it is per-passage rather than cross-passage, and it
is therefore defined at N=1, where ``max-cosine`` has no sibling to match and
RAGDefender's ``f_i`` degenerates (``n_pairs_for(1) = 1``). A poisoned passage
has to rank highly to be retrieved at all, so the very thing that makes the
attack work is what this reads -- and unlike a cross-passage statistic, an
attacker cannot evade it by injecting fewer documents.

Signs are FIXED per statistic rather than chosen by whichever direction scores
better on the evaluation data, which is what the earlier token sweep did; that
choice guarantees AUC >= 0.5 and lifts the null floor well above chance.

Usage::

    python scripts/diagnose_retrieval_score.py --encoder facebook/contriever
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, craft_whitebox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--attacked", type=int, default=30)
    ap.add_argument("--background", type=int, default=12000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--encoder", default="facebook/contriever")
    ap.add_argument("--normalize", action="store_true", default=False)
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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"rs-{len(keep)}",
                        normalize=args.normalize)
    clean_path = clean.save("rscore-clean")
    det = get_detector("ragdefender-freq", normalise="none")

    def embed_surface(hit) -> str:
        """What the index actually embedded for this hit."""
        t = getattr(hit, "title", "") or ""
        return f"{t} {hit.text}".strip() if t else hit.text

    print(f"encoder={args.encoder}  normalize={args.normalize}  k={args.k}\n")
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

            S = {"p": {}, "c": {}}
            truth, retrieved = [], 0
            for arm, src in (("p", store), ("c", clean)):
                for qid in qids:
                    q = targets[qid]["question"]
                    hits = src.search(q, k=args.k)
                    surf = [embed_surface(h) for h in hits]
                    emb = src.encoder.encode(surf, convert_to_numpy=True,
                                             normalize_embeddings=True)
                    sim = emb @ emb.T
                    np.fill_diagonal(sim, -np.inf)
                    row = {
                        "retrieval-score": np.array([float(h.score) for h in hits]),
                        "max-cosine": (sim.max(axis=1) if len(surf) > 1
                                       else np.full(len(surf), np.nan)),
                        "ragdefender-f_i": np.asarray(
                            det.frequency_scores(q, surf), dtype=np.float64),
                    }
                    for k_, v in row.items():
                        S[arm].setdefault(k_, []).append(np.asarray(v, dtype=np.float64))
                    if arm == "p":
                        t_ = np.array([h.injected for h in hits], dtype=bool)
                        truth.append(t_); retrieved += int(t_.sum())
            y = np.concatenate(truth)
            key = f"N={n_poison}/{variant}"
            print(f"=== {key}: {retrieved} surfaced, base rate {y.mean():.3f} ===")
            print(f"  {'statistic':20s} {'ROC':>8s} {'det':>8s} {'prec':>8s} "
                  f"{'clean flag':>11s}   defined at N=1?")
            rows = []
            for stat in ("retrieval-score", "max-cosine", "ragdefender-f_i"):
                sp = np.concatenate(S["p"][stat]); sc = np.concatenate(S["c"][stat])
                ok = np.isfinite(sp) & ~np.isnan(sp)
                if not ok.any():
                    print(f"  {stat:20s} {'undefined at N=1 by construction':>40s}")
                    rows.append({"statistic": stat, "undefined": True}); continue
                fc = sc[np.isfinite(sc) & ~np.isnan(sc)]
                thr = float(np.quantile(fc, 1.0 - args.target_fpr))
                spf = np.nan_to_num(sp, nan=-1e9, neginf=-1e9)
                r = float(roc_auc(y, spf))
                m = detection_metrics(y, spf >= thr, warn=False)
                cf = float((np.nan_to_num(sc, nan=-1e9, neginf=-1e9) >= thr).mean())
                note = "yes (per-passage)" if stat == "retrieval-score" else "no (cross-passage)"
                rows.append({"statistic": stat, "roc_auc": r,
                             "detection_rate": float(m.detection_rate),
                             "precision": float(m.precision), "clean_flag_rate": cf})
                print(f"  {stat:20s} {r:8.3f} {m.detection_rate:8.3f} "
                      f"{m.precision:8.3f} {cf:11.3f}   {note}")
            out[key] = {"n_retrieved": retrieved, "base_rate": float(y.mean()),
                        "rows": rows}
            print()

    print("The retrieval score is already computed by the index. If it matches or")
    print("beats the cross-passage statistics, no extra machinery is buying anything,")
    print("and at N=1 it is the only one of the three that is even defined.")
    o = REPORT_DIR / f"retrieval_score_{args.encoder.split('/')[-1]}.json"
    o.write_text(json.dumps({"config": vars(args), "results": out}, indent=2))
    print(f"written: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
