"""Is the RAG detection signal the attack, or the verbatim question prefix?

PoisonedRAG's black-box retrieval condition is ``P = Q + "." + I``
(``poisonedrag.py:craft_blackbox``, faithful to paper §4.2.2), so every passage
injected for one query shares a verbatim question prefix. Two consequences were
measured before this script existed and both need explaining:

* ``max(pairwise cosine)`` over the retrieved set -- one matmul, no LLM, no
  clustering -- scores AUC 1.000 at k=5 (``outputs/reports/agreement_scaling.json``).
* A five-line ``longest-common-prefix`` string function scores ROC 0.985,
  detection 0.975 at a 5% clean budget, above RAGDefender's own frequency score
  (0.945 / 0.868).

So the obvious worry: the benchmark may be measuring string duplication rather
than adversarial insertion.

**The test is to remove the question.** PoisonedRAG's white-box variant
(§4.3, ``craft_whitebox``) replaces the verbatim prefix with a HotFlip-optimised
token sequence, so the passages are still retrieved for the target query but no
longer share a literal prefix. Running the same scores over both variants
separates two hypotheses that the black-box arm alone cannot:

* if the lexical scores collapse on white-box while the embedding and frequency
  scores hold, those detectors were reading the attack, not the prefix;
* if everything collapses, the prefix was the signal;
* if the lexical scores hold too, the white-box surface is *also* duplicated and
  the test is uninformative -- which is itself worth knowing before drawing a
  conclusion.

Retrieval is reported alongside detection, because a poison that is no longer
retrieved is not a stealthier attack, it is a failed one.

Usage::

    python scripts/diagnose_prefix_confound.py --queries 60 --attacked 30
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


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def grams(s: str, n: int = 5) -> set[str]:
    t = s.lower()
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 0))}


def lexical_scores(texts: list[str]) -> dict[str, np.ndarray]:
    """Purely textual set statistics -- no model of any kind."""
    m = len(texts)
    prefix = np.zeros(m); jac = np.zeros(m)
    G = [grams(t) for t in texts]
    for i in range(m):
        best_p, best_j = 0.0, 0.0
        for j in range(m):
            if i == j:
                continue
            best_p = max(best_p, float(lcp(texts[i], texts[j])))
            u = len(G[i] | G[j])
            best_j = max(best_j, len(G[i] & G[j]) / u if u else 0.0)
        prefix[i], jac[i] = best_p, best_j
    return {"lexical-prefix": prefix, "lexical-5gram": jac}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--attacked", type=int, default=30)
    ap.add_argument("--background", type=int, default=12000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, default=3)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"pfx-{len(keep)}")
    clean_path = clean.save("pfx-clean")

    det = get_detector("ragdefender-freq", normalise="none")
    hard = get_detector("ragdefender")

    def build(variant: str) -> RagStore:
        st = RagStore.load(clean_path)
        for qid in sorted(attacked):
            t = targets[qid]
            base = list(t["adv_texts"])[: args.n_poison]
            texts = (craft_blackbox(t["question"], base) if variant == "blackbox"
                     else craft_whitebox(st.encoder, t["question"], base))
            st.add_documents(
                poison_documents(t["question"], t["incorrect answer"], texts,
                                 variant=variant, query_id=qid,
                                 correct_answer=t["correct answer"]), injected=True)
        return st

    out = {}
    for variant in ("blackbox", "whitebox"):
        store = build(variant)
        S_p: dict[str, list[np.ndarray]] = {}
        S_c: dict[str, list[np.ndarray]] = {}
        truth: list[np.ndarray] = []
        retrieved = 0
        for tag, src, bucket in (("p", store, S_p), ("c", clean, S_c)):
            for qid in qids:
                q = targets[qid]["question"]
                hits = src.search(q, k=args.k)
                texts = [h.text for h in hits]
                emb = src.encoder.encode(texts, convert_to_numpy=True,
                                         normalize_embeddings=True)
                sim = emb @ emb.T
                np.fill_diagonal(sim, -np.inf)
                row = dict(lexical_scores(texts))
                row["max-cosine"] = sim.max(axis=1)
                row["ragdefender-f_i"] = np.asarray(det.frequency_scores(q, texts),
                                                    dtype=np.float64)
                cut = np.zeros(len(texts), dtype=bool)
                cut[hard.removed_indices(q, texts)] = True
                row["ragdefender-decision"] = cut.astype(np.float64)
                for k_, v in row.items():
                    bucket.setdefault(k_, []).append(np.asarray(v, dtype=np.float64))
                if tag == "p":
                    t_ = np.array([h.injected for h in hits], dtype=bool)
                    truth.append(t_); retrieved += int(t_.sum())
        y = np.concatenate(truth)
        n_inj = int(store.injected_mask().sum())
        print(f"\n=== {variant}: {n_inj} injected docs, {retrieved} retrieved into "
              f"top-{args.k} ({100 * retrieved / max(n_inj, 1):.0f}% surfaced) ===")
        print(f"  {'score':22s} {'det.acc':>9s} {'prec':>7s} {'FPR':>7s} {'ROC':>7s} "
              f"{'clean flag':>11s}")
        rows = []
        for key in ("lexical-prefix", "lexical-5gram", "max-cosine",
                    "ragdefender-f_i", "ragdefender-decision"):
            sp = np.concatenate(S_p[key]); sc = np.concatenate(S_c[key])
            if key == "ragdefender-decision":
                thr, auc = 0.5, None
            else:
                thr = float(np.quantile(sc, 1.0 - args.target_fpr))
                auc = float(roc_auc(y, sp))
            res = detection_metrics(y, sp >= thr, warn=False)
            cf = float((sc >= thr).mean())
            rows.append({"score": key, "detection_rate": float(res.detection_rate),
                         "precision": float(res.precision), "fpr": float(res.fpr),
                         "roc_auc": auc, "clean_flag_rate": cf, "threshold": thr})
            g = "n/a" if auc is None else f"{auc:.3f}"
            print(f"  {key:22s} {res.detection_rate:9.3f} {res.precision:7.3f} "
                  f"{res.fpr:7.3f} {g:>7s} {cf:11.3f}")
        out[variant] = {"n_injected": n_inj, "n_retrieved": retrieved, "rows": rows}

    print("\nIf lexical-* collapses from black-box to white-box while max-cosine and")
    print("ragdefender-f_i hold, those two were reading the attack rather than the")
    print("shared question prefix. If everything collapses, the prefix was the signal.")
    o = REPORT_DIR / "prefix_confound.json"
    o.write_text(json.dumps({"config": vars(args), "variants": out}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
