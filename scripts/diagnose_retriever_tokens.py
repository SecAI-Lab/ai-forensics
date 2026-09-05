"""Token-level structure inside the retriever, layer by layer.

The first depth sweep collapsed every layer to one mean-pooled vector and found
the final layer beat all intermediates. That test was too coarse to answer the
question it was asked: HotFlip flips INDIVIDUAL TOKENS, and mean pooling is
exactly the operation that hides which tokens did the work.

The mechanism this looks for. Both retrievers here mean-pool, so the retrieval
score decomposes over tokens:

    sim(q, d) ∝ < mean_t h_t , q_hat >  =  mean_t < h_t , q_hat >

i.e. every token contributes an additive share ``c_t = <h_t, q_hat>``. A
naturally relevant passage earns its similarity broadly -- many tokens are
on-topic. A passage whose similarity was manufactured by flipping a few tokens
should earn it NARROWLY, from a short list of tokens that a natural passage
would not contain. That concentration is a per-passage quantity, needs no
sibling passage, and so survives the N=1 regime where every cross-passage
statistic is undefined.

It is also gradient-free, which is what separates it from GMTP (Findings ACL
2025), whose key-token step needs a backward pass through the retriever and then
a masked LM to score the tokens it picks.

Statistics, computed at EVERY layer:

``tok-max``      max_t cos(h_t, q) -- the single most query-aligned token.
``tok-top3``     share of total positive contribution held by the top 3 tokens.
``tok-gini``     Gini coefficient of the contribution profile; 0 = spread
                 evenly over tokens, 1 = one token carries everything.
``tok-normmax``  max token norm over median token norm, the attention-sink-style
                 outlier check that the SAE work in this repo already needed.
``traj-step``    1 - cos(h^(l), h^(l-1)) on the pooled vector: how far the
                 representation moves at this layer. A passage optimised only
                 against the final layer may take an abnormally large last step.
``pooled-qcos``  the previous sweep's per-layer query cosine, kept as the
                 reference line so the new statistics are read against it.

Run at N=3 and N=1, black-box and white-box. Only the white-box columns are
evidence about the attack rather than about its surface form.

Usage::

    python scripts/diagnose_retriever_tokens.py --encoder facebook/contriever
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
STATS = ("pooled-qcos", "tok-max", "tok-top3", "tok-gini", "tok-normmax", "traj-step")


def gini(x: np.ndarray) -> float:
    v = np.sort(np.maximum(x, 0.0))
    n = v.size
    s = v.sum()
    if n == 0 or s <= 1e-12:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(v) / (n * s))


def per_layer_token_stats(encoder, texts, query, max_length: int = 256):
    """``{layer: {stat: (n_texts,)}}`` from one batched forward pass."""
    import torch

    mod = encoder[0].auto_model
    tok = encoder.tokenizer
    device = next(mod.parameters()).device
    enc = tok(list(texts), padding=True, truncation=True, max_length=max_length,
              return_tensors="pt").to(device)
    qenc = tok([query], padding=True, truncation=True, max_length=max_length,
               return_tensors="pt").to(device)
    with torch.no_grad():
        out = mod(**enc, output_hidden_states=True)
        qout = mod(**qenc, output_hidden_states=True)
    mask = enc["attention_mask"].bool().cpu().numpy()
    qmask = qenc["attention_mask"].unsqueeze(-1).float()

    res: dict[int, dict[str, np.ndarray]] = {}
    prev_pooled = None
    for lay, h in enumerate(out.hidden_states):
        H = h.float().cpu().numpy()
        qh = qout.hidden_states[lay]
        q = ((qh * qmask).sum(dim=1) / qmask.sum(dim=1).clamp(min=1e-9))
        qv = q.float().cpu().numpy()[0]
        qv = qv / max(np.linalg.norm(qv), 1e-12)

        n = len(texts)
        st = {s: np.zeros(n, dtype=np.float64) for s in STATS}
        pooled = np.zeros((n, H.shape[-1]), dtype=np.float64)
        for i in range(n):
            toks = H[i][mask[i]]
            if toks.shape[0] == 0:
                continue
            pooled[i] = toks.mean(axis=0)
            norms = np.linalg.norm(toks, axis=1)
            unit = toks / np.maximum(norms[:, None], 1e-12)
            cos = unit @ qv
            contrib = toks @ qv           # the additive share of the pooled score
            pos = np.maximum(contrib, 0.0)
            tot = float(pos.sum())
            top3 = float(np.sort(pos)[-3:].sum())
            st["tok-max"][i] = float(cos.max())
            st["tok-top3"][i] = top3 / tot if tot > 1e-12 else 0.0
            st["tok-gini"][i] = gini(pos)
            st["tok-normmax"][i] = float(norms.max() / max(np.median(norms), 1e-12))
        pu = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        st["pooled-qcos"] = pu @ qv
        if prev_pooled is None:
            st["traj-step"] = np.zeros(n)
        else:
            st["traj-step"] = 1.0 - np.sum(pu * prev_pooled, axis=1)
        prev_pooled = pu
        res[lay] = st
    return res


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
    clean = build_index(keep, encoder=args.encoder, corpus_name=f"rt-{len(keep)}",
                        normalize=args.normalize)
    clean_path = clean.save("rtok-clean")
    enc = clean.encoder

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

            acc: dict[tuple[str, int, str], list] = {}
            truth, retrieved = [], 0
            for arm, src in (("p", store), ("c", clean)):
                for qid in qids:
                    q = targets[qid]["question"]
                    hits = src.search(q, k=args.k)
                    txt = [h.text for h in hits]
                    per = per_layer_token_stats(enc, txt, q)
                    for lay, st in per.items():
                        for s, v in st.items():
                            acc.setdefault((arm, lay, s), []).append(np.asarray(v))
                    if arm == "p":
                        t_ = np.array([h.injected for h in hits], dtype=bool)
                        truth.append(t_); retrieved += int(t_.sum())
            y = np.concatenate(truth)
            n_layers = max(l for _, l, _ in acc)
            key = f"N={n_poison}/{variant}"
            print(f"\n=== {key}  ({retrieved} surfaced, base rate {y.mean():.3f}, "
                  f"{args.encoder.split('/')[-1]}) ===")
            print(f"  {'layer':>5s} " + "".join(f"{s:>17s}" for s in STATS))
            rows = []
            for lay in range(n_layers + 1):
                cells, rec = [], {"layer": lay}
                for s in STATS:
                    sp = np.concatenate(acc[("p", lay, s)])
                    sc = np.concatenate(acc[("c", lay, s)])
                    if np.allclose(sp, sp[0]) and np.allclose(sc, sc[0]):
                        cells.append(f"{'—':>17s}"); rec[s] = None; continue
                    # Two-sided: a statistic can be anomalous by being low.
                    best = None
                    for sign in (1.0, -1.0):
                        thr = float(np.quantile(sign * sc, 1.0 - args.target_fpr))
                        r = float(roc_auc(y, sign * sp))
                        d = float(detection_metrics(y, sign * sp >= thr,
                                                    warn=False).detection_rate)
                        if best is None or r > best["roc_auc"]:
                            best = {"roc_auc": r, "detection_rate": d, "sign": sign}
                    rec[s] = best
                    cells.append(f"{best['roc_auc']:>9.3f}/{best['detection_rate']:<6.3f}")
                print(f"  {lay:>5d} " + "".join(cells))
                rows.append(rec)
            out[key] = {"n_retrieved": retrieved, "base_rate": float(y.mean()),
                        "layers": rows}

    print("\nROC/detection, best of the two signs. The last layer is what the")
    print("deployed retriever pools. A token statistic beating pooled-qcos at any")
    print("layer is signal that mean pooling was throwing away.")
    o = REPORT_DIR / f"retriever_tokens_{args.encoder.split('/')[-1]}.json"
    o.write_text(json.dumps({"config": vars(args), "results": out}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
