"""Does the retriever's *path* betray an adversarial passage its *endpoint* hides?

Every detector in this project reads the retriever's FINAL pooled embedding.
HotFlip optimises exactly that and nothing else (paper §4.3, Eq. 5:
``S = argmax_S' Sim(f_Q(Q), f_T(S' + I))``), so the intermediate layers are
unconstrained. If an adversarial passage reaches a normal endpoint by an
abnormal path, the path is visible and the endpoint is not.

Two questions, and the second is the one that matters.

**1. Is there per-layer signal at all?** Sweep every encoder layer and score
each passage three ways at each: ``max-cosine`` to the other retrieved passages
(the cross-passage statistic RAGDefender and the one-matmul baseline both rest
on), ``query-cos`` to the query itself, and ``norm``. Report against the final
layer, which is the incumbent.

**2. Does anything work at N=1?** With a single injected passage per query,
every cross-passage statistic is undefined or degenerate -- ``max-cosine`` has
no sibling to match, and RAGDefender's ``n_pairs_for(1) = 1`` collapses. That is
the regime where Phantom scored ragdefender 0.000 in
``outputs/reports/matrix_full_matrix.json``, and it is the regime a per-passage
internal statistic could own. ``query-cos`` and ``norm`` are per-passage and
survive it; the sweep reports N=3 and N=1 side by side so the difference is
visible rather than argued.

Both attack surfaces are run: black-box (verbatim question prefix) and white-box
(HotFlip). The black-box arm is known to be solvable by a five-line string
function, so only the white-box column is evidence about the attack rather than
about its surface form.

Usage::

    python scripts/diagnose_retriever_depth.py --queries 60 --attacked 30
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
from aiforensics.metrics.detection import detection_metrics, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"


def layer_states(encoder, texts, max_length: int = 256):
    """Mean-pooled hidden state at every layer, as ``{layer: (n, dim)}``.

    Mean pooling over real tokens is what both retrievers here use, so applying
    it at every layer keeps the intermediate readouts on the same footing as the
    final one rather than comparing pooled to unpooled.
    """
    import torch

    mod = encoder[0].auto_model
    tok = encoder.tokenizer
    device = next(mod.parameters()).device
    enc = tok(list(texts), padding=True, truncation=True,
              max_length=max_length, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mod(**enc, output_hidden_states=True)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return {i: ((h * mask).sum(dim=1) / denom).float().cpu().numpy()
            for i, h in enumerate(out.hidden_states)}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--attacked", type=int, default=30)
    ap.add_argument("--background", type=int, default=12000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[3, 1])
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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"rd-{len(keep)}")
    clean_path = clean.save("rdepth-clean")
    enc = clean.encoder

    results = {}
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

            acc: dict[tuple[str, int], dict[str, list]] = {}
            truth, retrieved = [], 0
            for arm, src in (("p", store), ("c", clean)):
                for qid in qids:
                    q = targets[qid]["question"]
                    hits = src.search(q, k=args.k)
                    txt = [h.text for h in hits]
                    H = layer_states(enc, txt)
                    Hq = layer_states(enc, [q])
                    for lay, X in H.items():
                        U, uq = unit(X), unit(Hq[lay])[0]
                        sim = U @ U.T
                        np.fill_diagonal(sim, -np.inf)
                        d = acc.setdefault((arm, lay), {"max-cosine": [], "query-cos": [],
                                                        "norm": []})
                        d["max-cosine"].append(sim.max(axis=1) if len(txt) > 1
                                               else np.full(len(txt), -np.inf))
                        d["query-cos"].append(U @ uq)
                        d["norm"].append(np.linalg.norm(X, axis=1))
                    if arm == "p":
                        t_ = np.array([h.injected for h in hits], dtype=bool)
                        truth.append(t_); retrieved += int(t_.sum())
            y = np.concatenate(truth)
            n_layers = max(l for _, l in acc)
            key = f"N={n_poison}/{variant}"
            print(f"\n=== {key}: {retrieved} injected surfaced into top-{args.k}, "
                  f"base rate {y.mean():.3f} ===")
            print(f"  {'layer':>6s} " + "".join(
                f"{s:>22s}" for s in ("max-cosine ROC/det", "query-cos ROC/det",
                                      "norm ROC/det")))
            rows = []
            for lay in range(n_layers + 1):
                cells, rec = [], {"layer": lay}
                for stat in ("max-cosine", "query-cos", "norm"):
                    sp = np.concatenate(acc[("p", lay)][stat])
                    sc = np.concatenate(acc[("c", lay)][stat])
                    fin_c = sc[np.isfinite(sc)]
                    if not np.isfinite(sp).any() or fin_c.size == 0:
                        cells.append(f"{'—':>22s}"); rec[stat] = None; continue
                    thr = float(np.quantile(fin_c, 1.0 - args.target_fpr))
                    r = float(roc_auc(y, np.nan_to_num(sp, neginf=-1e9)))
                    dd = float(detection_metrics(y, sp >= thr, warn=False).detection_rate)
                    rec[stat] = {"roc_auc": r, "detection_rate": dd,
                                 "clean_flag_rate": float((sc >= thr).mean())}
                    cells.append(f"{r:>13.3f}/{dd:<7.3f}")
                print(f"  {lay:>6d} " + "".join(cells))
                rows.append(rec)
            results[key] = {"n_retrieved": retrieved, "base_rate": float(y.mean()),
                            "layers": rows}

    print("\nLayer 0 is the embedding table; the last row is the layer the deployed")
    print("retriever actually pools. A middle row beating the last would mean the")
    print("endpoint hides something the path shows. At N=1 max-cosine is undefined")
    print("by construction -- that column is where a per-passage statistic can win.")
    o = REPORT_DIR / "retriever_depth.json"
    o.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
