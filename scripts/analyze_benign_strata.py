"""How many benign passages actually survive retrieval, and does it decide everything?

Every detector benchmarked so far is *relative within the retrieved set*:

* NPAS normalises attention across the set, so its scores sum to 100;
* ReliabilityRAG compares passages against each other via NLI contradiction;
* passage-agreement measures mutual similarity;
* RAGDefender contrasts an adversarial group against a benign one.

All four therefore need something to compare against. The number of **benign**
passages surviving in the top-k is thus the hidden variable behind every result,
and it is not simply ``k - N``: retrieval decides how many of the injected
passages actually surface, and that varies per query.

This script measures the real distribution of benign counts per configuration,
then reports detection accuracy **stratified by that count** — including the
fully flooded case (zero benign), which is where a relative detector has nothing
left to work with.

Usage::

    python scripts/analyze_benign_strata.py --n-poison 1 2 3 5 8
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"
DETECTORS = ("npas", "reliability", "passage-agreement", "query-similarity", "ragdefender")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--background", type=int, default=6000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[1, 2, 3, 5, 8])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]

    qrels = load_qrels("nq", split="test")
    gold = {d for q in qids for d in qrels.get(q, {})}
    keep, bg = [], 0
    for doc in load_corpus("nq", max_docs=args.background * 3):
        if doc.doc_id in gold:
            keep.append(doc)
        elif bg < args.background:
            keep.append(doc); bg += 1
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"strata-{len(keep)}")
    clean_path = clean.save("strata-clean")

    from aiforensics.models.llm import load_llm
    model, tok = load_llm(args.generator)

    # ---- how many benign passages actually survive? ------------------------
    print(f"corpus {len(clean):,} docs | k={args.k} | {len(qids)} queries\n")
    print("=" * 78)
    print("BENIGN PASSAGES SURVIVING IN THE TOP-k  (not simply k - N)")
    print("=" * 78)
    print(f"{'N':>2s} {'mean poison@k':>14s} {'mean benign@k':>14s} {'fully flooded':>15s}")
    print("-" * 78)

    stores, dist = {}, {}
    for n in args.n_poison:
        store = RagStore.load(clean_path)
        for qid in qids:
            t = targets[qid]
            base = list(t["adv_texts"])
            base = [base[i % len(base)] for i in range(n)]
            store.add_documents(
                poison_documents(t["question"], t["incorrect answer"],
                                 craft_blackbox(t["question"], base),
                                 variant="blackbox", query_id=qid,
                                 correct_answer=t["correct answer"]), injected=True)
        stores[n] = store
        benign = []
        for qid in qids:
            hits = store.search(targets[qid]["question"], k=args.k)
            benign.append(int(sum(1 for h in hits if not h.injected)))
        dist[n] = benign
        flooded = sum(1 for b in benign if b == 0)
        print(f"{n:2d} {args.k - np.mean(benign):14.2f} {np.mean(benign):14.2f} "
              f"{flooded:8d}/{len(qids):<6d}")

    # ---- detection stratified by benign count ------------------------------
    rows = defaultdict(lambda: defaultdict(list))
    for key in args.detectors:
        for n in args.n_poison:
            store = stores[n]
            try:
                det = get_detector(key, store=store, encoder=store.encoder,
                                   model=model, tokenizer=tok)
            except Exception as exc:
                print(f"  !! {key}: {exc}")
                break
            # clean-calibrated threshold, as everywhere else
            s_clean = np.concatenate([
                np.asarray(det.score((targets[q]["question"],
                                      [h.text for h in clean.search(targets[q]["question"],
                                                                    k=args.k)])),
                           dtype=np.float64).reshape(-1) for q in qids])
            binary = len(np.unique(s_clean)) <= 2
            thr = (float(det.threshold) if (binary and det.threshold is not None)
                   else float(np.quantile(s_clean, 1.0 - args.target_fpr)))
            for qid in qids:
                q = targets[qid]["question"]
                hits = store.search(q, k=args.k)
                s = np.asarray(det.score((q, [h.text for h in hits])),
                               dtype=np.float64).reshape(-1)
                t_ = np.array([h.injected for h in hits], dtype=bool)
                if not t_.any():
                    continue
                rows[key][int((~t_).sum())].append((t_, s >= thr))

    print("\n" + "=" * 78)
    print("DETECTION ACCURACY STRATIFIED BY #BENIGN PASSAGES IN THE TOP-k")
    print("=" * 78)
    strata = sorted({b for d in rows.values() for b in d})
    hdr = f"{'detector':20s}" + "".join(f"{('b=' + str(b)):>9s}" for b in strata)
    print(hdr); print("-" * len(hdr))
    out = {}
    for key in args.detectors:
        if key not in rows:
            continue
        cells, rec = [], {}
        for b in strata:
            pairs = rows[key].get(b, [])
            if not pairs:
                cells.append(f"{'—':>9s}"); continue
            tt = np.concatenate([p[0] for p in pairs])
            pp = np.concatenate([p[1] for p in pairs])
            dr = float(detection_metrics(tt, pp, warn=False).detection_rate)
            rec[b] = {"queries": len(pairs), "detection_rate": dr}
            cells.append(f"{dr:9.3f}")
        out[key] = rec
        print(f"{key:20s}" + "".join(cells))
    print(f"\n{'':20s}" + "".join(f"{('n=' + str(sum(len(rows[k].get(b, [])) for k in rows))):>9s}"
                                  for b in strata))
    print("\nb = number of BENIGN passages retrieved. b=0 is the flooded case: a")
    print("detector that compares passages against each other has nothing to compare.")

    o = REPORT_DIR / "benign_strata.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({"config": vars(args),
                             "benign_distribution": {str(k): v for k, v in dist.items()},
                             "stratified": out}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
