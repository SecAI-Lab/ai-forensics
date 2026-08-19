"""Can a detector say "NOT POISONED"?

The investigator does not know whether the knowledge base was tampered with. A
detector that only ever answers "here are the poisoned passages" is unusable:
run on a clean database it must stay silent. This script measures exactly that.

Two conditions, same corpus, same queries:

* **clean**    — no poison anywhere. Every flag raised is a false alarm. The
  ideal detector flags nothing and the corpus-level verdict is NOT POISONED.
* **poisoned** — PoisonedRAG injected for a subset of queries. Queries that were
  *not* attacked stay clean even in this condition, which is the realistic case:
  a poisoned database is not poisoned for every question.

Reported per condition: the flag rate (fraction of retrieved passages removed),
the per-query false-alarm rate, and the corpus-level verdict a triage tool would
emit. Detection accuracy (#detected poisoned / #poisoned) is reported only where
poison actually exists, since it is undefined otherwise.

Usage::

    python scripts/run_clean_db_test.py --queries 40 --attacked 20
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import craft_blackbox, poison_documents
from aiforensics.data.rag import RagStore, build_index, load_corpus, load_qrels
from aiforensics.detectors import get_detector
from aiforensics.metrics.detection import detection_metrics

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def summarise(name: str, flags: list[np.ndarray], truths: list[np.ndarray]) -> dict:
    """Flag behaviour for one condition."""
    n_pass = sum(f.size for f in flags)
    n_flag = sum(int(f.sum()) for f in flags)
    n_pois = sum(int(t.sum()) for t in truths)
    queries_with_flag = sum(1 for f in flags if f.any())
    row = {
        "condition": name,
        "queries": len(flags),
        "passages": n_pass,
        "poisoned_passages": n_pois,
        "flagged_passages": n_flag,
        "flag_rate": n_flag / max(1, n_pass),
        "queries_raising_an_alarm": queries_with_flag,
        "query_alarm_rate": queries_with_flag / max(1, len(flags)),
    }
    if n_pois:
        allt = np.concatenate(truths)
        allf = np.concatenate(flags)
        res = detection_metrics(allt, allf, warn=False)
        row |= {
            "detection_rate": float(res.detection_rate),
            "precision": float(res.precision),
            "f1": float(res.f1),
            "fpr": float(res.fpr),
        }
    else:
        # No positives exist: recall/precision are undefined, only false alarms matter.
        row |= {"detection_rate": None, "precision": None, "f1": None,
                "fpr": n_flag / max(1, n_pass)}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=40, help="queries evaluated in total")
    ap.add_argument("--attacked", type=int, default=20, help="how many of them are attacked")
    ap.add_argument("--background", type=int, default=20000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, default=5)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    assert args.attacked <= args.queries

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
    print(f"corpus: {len(keep)} docs ({sum(1 for d in keep if d.doc_id in gold)}/{len(gold)} gold)")

    clean_store = build_index(keep, encoder=args.encoder, corpus_name=f"cleantest-{len(keep)}")
    clean_path = clean_store.save("cleantest-clean")

    poisoned = RagStore.load(clean_path)
    for qid in sorted(attacked):
        t = targets[qid]
        texts = craft_blackbox(t["question"], list(t["adv_texts"])[: args.n_poison])
        poisoned.add_documents(
            poison_documents(t["question"], t["incorrect answer"], texts, variant="blackbox",
                             query_id=qid, correct_answer=t["correct answer"]),
            injected=True,
        )
    print(f"poisoned store: {int(poisoned.injected_mask().sum())} injected docs "
          f"for {len(attacked)}/{len(qids)} queries\n")

    detector = get_detector("ragdefender", task_type="single_hop")
    rows = []

    for cond, store in (("CLEAN DB", clean_store), ("POISONED DB", poisoned)):
        flags, truths = [], []
        sub_flags, _sub_truths = {"attacked": ([], []), "not attacked": ([], [])}, None
        for qid in qids:
            q = targets[qid]["question"]
            hits = store.search(q, k=args.k)
            texts = [h.text for h in hits]
            truth = np.array([h.injected for h in hits], dtype=bool)
            pred = np.asarray(detector.score((q, texts)), dtype=bool)
            flags.append(pred); truths.append(truth)
            bucket = "attacked" if qid in attacked else "not attacked"
            sub_flags[bucket][0].append(pred)
            sub_flags[bucket][1].append(truth)
        row = summarise(cond, flags, truths)
        rows.append(row)
        print(f"=== {cond} ===")
        print(f"  passages seen        : {row['passages']}   poisoned: {row['poisoned_passages']}")
        print(f"  passages FLAGGED     : {row['flagged_passages']}  "
              f"(flag rate {row['flag_rate']:.3f})")
        print(f"  queries raising alarm: {row['queries_raising_an_alarm']}/{row['queries']}  "
              f"({row['query_alarm_rate']:.3f})")
        if row["detection_rate"] is not None:
            print(f"  detection accuracy   : {row['detection_rate']:.3f}  "
                  f"precision {row['precision']:.3f}  FPR {row['fpr']:.3f}")
        else:
            print(f"  detection accuracy   : undefined (no poison exists)  "
                  f"false-alarm rate {row['fpr']:.3f}")
        if cond == "POISONED DB":
            for bucket in ("attacked", "not attacked"):
                f, t = sub_flags[bucket]
                if not f:
                    continue
                sub = summarise(f"  {bucket} queries", f, t)
                print(f"    {bucket:13s}: {sub['flagged_passages']}/{sub['passages']} flagged "
                      f"(rate {sub['flag_rate']:.3f})" +
                      (f", detection {sub['detection_rate']:.3f}"
                       if sub["detection_rate"] is not None else " — no poison here"))
                rows.append(sub)
        print()

    clean_row = rows[0]
    print("=== VERDICT: can this detector say 'NOT POISONED'? ===")
    if clean_row["flagged_passages"] == 0:
        print("  YES — zero flags on the clean database.")
    else:
        print(f"  NO — it flagged {clean_row['flagged_passages']}/{clean_row['passages']} "
              f"passages ({clean_row['flag_rate']:.1%}) on a database with no poison at all,")
        print(f"       and raised an alarm on {clean_row['queries_raising_an_alarm']}"
              f"/{clean_row['queries']} queries.")
        print("       Used as a triage tool it would report a clean corpus as poisoned.")
        print("       A corpus-level 'is this poisoned?' verdict therefore needs a calibrated")
        print("       alarm-rate threshold, not the per-passage decision on its own.")

    out = REPORT_DIR / "clean_db_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": rows}, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
