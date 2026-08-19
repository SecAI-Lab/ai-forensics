"""Evaluate published detectors under their OWN decision rules.

Forcing every method through one calibrated threshold is fair only for methods
that expose a continuous score. Three of the four published detectors here do
not: each defines its own operating point, and scoring them any other way
measures the harness rather than the paper.

    ragdefender     estimate how many passages are adversarial, then select
                    exactly that many  (ACSAC 2025)
    npas            variance gate on the attention shares, then remove the
                    arg-max, at most floor(eps*k) times  (ICML 2026)
    reliability     flag the complement of the maximum independent set of the
                    NLI contradiction graph  (NeurIPS 2025)
    gmtp            P-score < tau, where tau = lambda * mean(clean P-score)
                    (Findings of ACL 2025)

Only ``gmtp``'s rule is a threshold, and even that one is calibrated by the
paper's own recipe rather than by an alarm budget. Every detector that needs
calibration gets it from **clean data only**.

Reported per detector: detection accuracy (#detected poisoned / #poisoned),
precision, F1, the per-passage false-positive rate, the flag rate on a clean
corpus, and — because three of the four can decline to flag anything — the
fraction of clean retrieved sets on which the method correctly stayed silent.

Usage::

    python scripts/run_native_protocol.py --n-poison 1 2 3
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
from aiforensics.metrics.detection import detection_metrics

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"
PUBLISHED = ("gmtp", "ragdefender", "npas", "reliability")
VENUE = {
    "gmtp": "Findings ACL'25",
    "ragdefender": "ACSAC'25",
    "npas": "ICML'26",
    "reliability": "NeurIPS'25",
}


def native_flags(key: str, det, query: str, texts: list[str]) -> np.ndarray:
    """Boolean flags from the detector's OWN decision rule."""
    k = len(texts)
    out = np.zeros(k, dtype=bool)
    if k == 0:
        return out
    if key == "npas":
        # Algorithm 1: variance gate, then arg-max removal.
        _, removed = det.filter(query, texts)
        out[list(removed)] = True
        return out
    if key == "ragdefender":
        # estimate_n_adv -> select exactly that many; score() is already binary.
        return np.asarray(det.score((query, texts)), dtype=np.float64) >= 0.5
    # reliability: MIS complement. gmtp: P-score < tau, i.e. -P-score >= -tau.
    pred = det.predict((query, texts))
    return np.asarray(pred, dtype=bool)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=20)
    ap.add_argument("--attacked", type=int, default=10)
    ap.add_argument("--background", type=int, default=6000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variants", nargs="+", default=["blackbox", "whitebox"])
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--generator", default="qwen3-1.7b")
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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"native-{len(keep)}")
    clean_path = clean.save("native-clean")
    print(f"corpus {len(clean):,} docs | k={args.k} | {len(attacked)}/{len(qids)} attacked\n")

    from aiforensics.models.llm import load_llm
    model, tok = load_llm(args.generator)
    clean_sets = [(targets[q]["question"],
                   [h.text for h in clean.search(targets[q]["question"], k=args.k)])
                  for q in qids]

    rows = []
    for variant in args.variants:
        for n in args.n_poison:
            store = RagStore.load(clean_path)
            for qid in sorted(attacked):
                t = targets[qid]
                base = list(t["adv_texts"])
                base = [base[i % len(base)] for i in range(n)]
                texts = (craft_blackbox(t["question"], base) if variant == "blackbox"
                         else craft_whitebox(store.encoder, t["question"], base))
                store.add_documents(
                    poison_documents(t["question"], t["incorrect answer"], texts,
                                     variant=variant, query_id=qid,
                                     correct_answer=t["correct answer"]), injected=True)

            for key in PUBLISHED:
                try:
                    det = get_detector(key, store=store, encoder=store.encoder,
                                       model=model, tokenizer=tok)
                except Exception as exc:
                    print(f"  !! {key}: {exc}")
                    continue
                if hasattr(det, "fit"):
                    try:
                        det.fit(clean_sets)
                    except Exception as exc:
                        print(f"  .. {key}: fit skipped ({exc})")

                T, P, silent = [], [], 0
                for i, qid in enumerate(qids):
                    q = targets[qid]["question"]
                    hits = store.search(q, k=args.k)
                    texts = [h.text for h in hits]
                    T.append(np.array([h.injected for h in hits], bool))
                    P.append(native_flags(key, det, q, texts))
                    cq, cp = clean_sets[i]
                    if not native_flags(key, det, cq, cp).any():
                        silent += 1
                t_all, p_all = np.concatenate(T), np.concatenate(P)
                clean_flags = np.concatenate(
                    [native_flags(key, det, cq, cp) for cq, cp in clean_sets])
                r = detection_metrics(t_all, p_all, warn=False)
                rows.append({
                    "variant": variant, "N": n, "detector": key, "venue": VENUE[key],
                    "detection_rate": float(r.detection_rate),
                    "precision": float(r.precision), "f1": float(r.f1),
                    "fpr": float(r.fpr),
                    "clean_flag_rate": float(clean_flags.mean()),
                    "silent_on_clean_sets": silent / len(qids),
                })
                print(f"  [{variant:9s} N={n}] {key:14s} det={r.detection_rate:.3f} "
                      f"prec={r.precision:.3f} clean={clean_flags.mean():.3f} "
                      f"silent={silent}/{len(qids)}")

    print("\n" + "=" * 104)
    print("PUBLISHED DETECTORS UNDER THEIR OWN DECISION RULES")
    print("=" * 104)
    for variant in args.variants:
        print(f"\n--- PoisonedRAG {variant} ---")
        h = (f"{'detector':14s} {'venue':16s} " +
             "".join(f"{('N=' + str(n)):>22s}" for n in args.n_poison))
        print(h)
        print(f"{'':31s}" + "".join(f"{'det / prec / clean':>22s}" for _ in args.n_poison))
        print("-" * len(h))
        for key in PUBLISHED:
            cells = []
            for n in args.n_poison:
                m = [r for r in rows if r["variant"] == variant and r["N"] == n
                     and r["detector"] == key]
                if not m:
                    cells.append(f"{'—':>22s}"); continue
                r = m[0]
                p = "  nan" if np.isnan(r["precision"]) else f"{r['precision']:.2f}"
                cells.append(f"{r['detection_rate']:8.2f} /{p:>5s} /{r['clean_flag_rate']:6.2f}")
            print(f"{key:14s} {VENUE[key]:16s}" + "".join(cells))

    print("\n'clean' = fraction of passages flagged on a corpus with NO poison at all.")
    print("A method that can decline to flag anything reports 'silent' below.\n")
    print(f"{'detector':14s} {'silent on clean retrieved sets':>34s}")
    print("-" * 50)
    for key in PUBLISHED:
        m = [r["silent_on_clean_sets"] for r in rows if r["detector"] == key]
        if m:
            print(f"{key:14s} {np.mean(m):>33.1%}")

    out = REPORT_DIR / "native_protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
