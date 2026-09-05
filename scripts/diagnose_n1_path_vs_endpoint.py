"""N=1: does the retriever's PATH add anything the deployed ENDPOINT score misses?

The N=1 claim under test is that "every endpoint statistic is undefined or
degenerate when a single passage is injected". That is true only of the
CROSS-PASSAGE family (max-cosine, RAGDefender's f_i). The deployed retrieval
score itself -- the query-passage similarity the retriever already computed --
is per-passage and costs nothing extra. This script pits the path against that
incumbent, not against max-cosine.

Everything is read off ONE forward pass with output_hidden_states=True, pooled
exactly as the deployed retriever pools, and read through the FIXED deployed
query vector q* (not a layer-matched query, which would let both objects move).

    p_l  = mean-pool of hidden_states[l]        (l = 0 .. L)
    d_0  = p_0 ;  d_l = p_l - p_{l-1}           (telescopes: sum_l d_l = p_L)
    g_l  = <d_l, q*> / Z                         Z = ||p_L|| (cosine) or 1 (dot)
    sum_l g_l == the deployed score              (asserted, not assumed)

Statistics scored: the endpoint score itself, the last-block share of the
similarity budget, the terminal excess over the best intermediate alignment, a
self-normalised late-jump ratio, a Mahalanobis distance on the whole per-layer
contribution profile, and a clean-fit ridge residual (the endpoint-orthogonal
component). Plus token count, because the poison is short LLM prose against
long NQ chunks and that confound has already bitten this testbed once.
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAG_DIR, RAW_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.poisonedrag import (PoisonedRAGConfig, craft_blackbox,
                                             craft_whitebox, poison_documents)
from aiforensics.data.rag import RagStore
from aiforensics.metrics.detection import detection_metrics, roc_auc

TARGETS = RAW_DIR / "poisonedrag_nq_targets.json"

STATS = ("endpoint", "max-cosine", "A", "A2", "E", "J", "trajmax",
         "mahal-g", "mahal-step", "ridge-r", "ntok")


def profiles(encoder, texts, query, normalized: bool, max_length: int):
    """Per-passage layer profile through the FIXED deployed query vector."""
    import torch

    mod = encoder[0].auto_model
    tok = encoder.tokenizer
    device = next(mod.parameters()).device

    def pooled_all(batch):
        e = tok(list(batch), padding=True, truncation=True, max_length=max_length,
                return_tensors="pt").to(device)
        with torch.no_grad():
            o = mod(**e, output_hidden_states=True)
        m = e["attention_mask"].unsqueeze(-1).float()
        den = m.sum(dim=1).clamp(min=1e-9)
        P = np.stack([((h * m).sum(dim=1) / den).float().cpu().numpy()
                      for h in o.hidden_states], axis=1)      # (n, L+1, dim)
        return P, e["attention_mask"].sum(dim=1).cpu().numpy()

    Pq, _ = pooled_all([query])
    qstar = Pq[0, -1].astype(np.float64)
    if normalized:
        qstar = qstar / max(np.linalg.norm(qstar), 1e-12)

    P, ntok = pooled_all(texts)
    P = P.astype(np.float64)
    n, nl, _ = P.shape
    D = P.copy()
    D[:, 1:] = P[:, 1:] - P[:, :-1]                            # d_0 = p_0
    Z = np.linalg.norm(P[:, -1], axis=1) if normalized else np.ones(n)
    G = (D @ qstar) / np.maximum(Z[:, None], 1e-12)            # (n, L+1)
    endpoint = G.sum(axis=1)

    pn = np.linalg.norm(P, axis=2)
    C = (P @ qstar) / np.maximum(pn, 1e-12) / max(np.linalg.norm(qstar), 1e-12)
    U = P / np.maximum(pn[:, :, None], 1e-12)
    step = 1.0 - np.sum(U[:, 1:] * U[:, :-1], axis=2)          # (n, L)

    absg = np.abs(G).sum(axis=1)
    A = G[:, -1] / np.maximum(absg, 1e-12)
    A2 = G[:, -2:].sum(axis=1) / np.maximum(absg, 1e-12)
    E = C[:, -1] - C[:, :-1].max(axis=1)
    J = step[:, -1] / np.maximum(np.median(step[:, :-1], axis=1), 1e-12)
    return {"G": G, "C": C, "step": step, "endpoint": endpoint, "P_last": P[:, -1],
            "A": A, "A2": A2, "E": E, "J": J, "ntok": ntok.astype(np.float64)}


def maha_fit(X):
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(X)
    return lw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", type=int, default=60)
    ap.add_argument("--attacked", type=int, default=30)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-poison", type=int, nargs="+", default=[1, 3])
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--store", default="rdepth-clean")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    targets = json.loads(TARGETS.read_text())
    qids = list(targets)[: args.queries]
    attacked = set(qids[: args.attacked])

    clean = RagStore.load(RAG_DIR / args.store)
    enc = clean.encoder
    normalized = bool(clean.normalized)
    maxlen = int(getattr(enc, "max_seq_length", 256) or 256)
    print(f"store={args.store} n={len(clean)} normalized={normalized} "
          f"similarity={clean.similarity} max_len={maxlen}")
    cfg = PoisonedRAGConfig(score_function="cos" if normalized else "dot")

    out = {}
    for n_poison in args.n_poison:
        for variant in ("blackbox", "whitebox"):
            store = RagStore.load(RAG_DIR / args.store)
            for qid in sorted(attacked):
                t = targets[qid]
                base = list(t["adv_texts"])[:n_poison]
                texts = (craft_blackbox(t["question"], base) if variant == "blackbox"
                         else craft_whitebox(store.encoder, t["question"], base,
                                             config=cfg))
                store.add_documents(
                    poison_documents(t["question"], t["incorrect answer"], texts,
                                     variant=variant, query_id=qid,
                                     correct_answer=t["correct answer"]), injected=True)

            rows = {"p": [], "c": []}
            recon_err = 0.0
            for arm, src in (("p", store), ("c", clean)):
                for qid in qids:
                    q = targets[qid]["question"]
                    hits = src.search(q, k=args.k)
                    # the store embeds "title text" (Document.embed_text), not text
                    txt = [f"{h.title} {h.text}".strip() if h.title else h.text
                           for h in hits]
                    pr = profiles(enc, txt, q, normalized, maxlen)
                    faiss = np.array([h.score for h in hits], dtype=np.float64)
                    recon_err = max(recon_err,
                                    float(np.abs(pr["endpoint"] - faiss).max()))
                    Pl = pr["P_last"]
                    Un = Pl / np.maximum(np.linalg.norm(Pl, axis=1, keepdims=True), 1e-12)
                    sim = Un @ Un.T
                    np.fill_diagonal(sim, -np.inf)
                    mc = sim.max(axis=1) if len(txt) > 1 else np.full(len(txt), -1e9)
                    rows[arm].append({
                        "qid": qid, "inj": np.array([h.injected for h in hits], bool),
                        "attacked": qid in attacked, "maxcos": mc, **pr,
                    })

            def cat(arm, key):
                return np.concatenate([r[key] for r in rows[arm]], axis=0)


            y = cat("p", "inj")
            # fit clean distributional params on NON-attacked clean rows only
            fit_mask = np.concatenate([np.full(len(r["inj"]), r["attacked"] is False)
                                       for r in rows["c"]])
            Gc, Gp = cat("c", "G"), cat("p", "G")
            Sc, Sp = cat("c", "step"), cat("p", "step")
            Cc, Cp = cat("c", "C"), cat("p", "C")

            lw_g = maha_fit(Gc[fit_mask])
            lw_s = maha_fit(Sc[fit_mask])
            mg_c, mg_p = lw_g.mahalanobis(Gc), lw_g.mahalanobis(Gp)
            ms_c, ms_p = lw_s.mahalanobis(Sc), lw_s.mahalanobis(Sp)

            med = np.median(Sc[fit_mask], axis=0)
            mad = np.median(np.abs(Sc[fit_mask] - med), axis=0) * 1.4826 + 1e-9
            tj_c = np.abs((Sc - med) / mad).max(axis=1)
            tj_p = np.abs((Sp - med) / mad).max(axis=1)

            from numpy.linalg import lstsq
            Xc = np.concatenate([Cc[fit_mask][:, :-1],
                                 np.ones((fit_mask.sum(), 1))], axis=1)
            w = lstsq(Xc, Cc[fit_mask][:, -1], rcond=None)[0]
            rr_c = Cc[:, -1] - np.concatenate([Cc[:, :-1], np.ones((len(Cc), 1))], 1) @ w
            rr_p = Cp[:, -1] - np.concatenate([Cp[:, :-1], np.ones((len(Cp), 1))], 1) @ w

            cells = {
                "endpoint": (cat("c", "endpoint"), cat("p", "endpoint")),
                "max-cosine": (cat("c", "maxcos"), cat("p", "maxcos")),
                "A": (cat("c", "A"), cat("p", "A")),
                "A2": (cat("c", "A2"), cat("p", "A2")),
                "E": (cat("c", "E"), cat("p", "E")),
                "J": (cat("c", "J"), cat("p", "J")),
                "trajmax": (tj_c, tj_p),
                "mahal-g": (mg_c, mg_p),
                "mahal-step": (ms_c, ms_p),
                "ridge-r": (rr_c, rr_p),
                "ntok": (cat("c", "ntok"), cat("p", "ntok")),
            }

            key = f"N={n_poison}/{variant}"
            print(f"\n=== {key}  ({int(y.sum())} injected surfaced, base {y.mean():.3f}, "
                  f"recon_err {recon_err:.2e}) ===")
            rec = {"n_retrieved": int(y.sum()), "base_rate": float(y.mean()),
                   "recon_err": recon_err, "stats": {}}
            thr_ep = None
            flags = {}
            for name, (sc, sp) in cells.items():
                thr = float(np.quantile(sc, 1.0 - args.target_fpr))
                r = float(roc_auc(y, sp))
                d = float(detection_metrics(y, sp >= thr, warn=False).detection_rate)
                rec["stats"][name] = {"roc_auc": r, "detection_rate": d,
                                      "clean_flag_rate": float((sc >= thr).mean())}
                flags[name] = sp >= thr
                if name == "endpoint":
                    thr_ep = thr
                print(f"  {name:>12s}  ROC {r:.3f}  det {d:.3f}  "
                      f"cleanflag {float((sc >= thr).mean()):.3f}")

            miss = y & ~flags["endpoint"]
            rec["endpoint_misses"] = int(miss.sum())
            rec["complement"] = {}
            neg = ~y
            print(f"  -- endpoint misses {int(miss.sum())} of {int(y.sum())} injected")
            for name in cells:
                if name == "endpoint":
                    continue
                caught = int((miss & flags[name]).sum())
                sub = np.concatenate([cells[name][1][miss], cells[name][1][neg]])
                suby = np.concatenate([np.ones(miss.sum(), bool),
                                       np.zeros(neg.sum(), bool)])
                ca = float(roc_auc(suby, sub)) if miss.sum() else float("nan")
                rec["complement"][name] = {"caught_of_misses": caught,
                                           "cond_roc_auc": ca}
                print(f"     {name:>12s} catches {caught}/{int(miss.sum())} "
                      f"misses, conditional ROC {ca:.3f}")
            # length control: rank-residualise every stat on ntok
            nt_c, nt_p = cells["ntok"]
            allx = np.concatenate([nt_c, nt_p])
            rec["length_residual"] = {}
            for name in cells:
                if name == "ntok":
                    continue
                ac, ap_ = cells[name]
                X = np.stack([np.concatenate([nt_c, nt_p]),
                              np.ones(len(allx))], axis=1)
                yy = np.concatenate([ac, ap_])
                ww = lstsq(X, yy, rcond=None)[0]
                res = yy - X @ ww
                rc, rp = res[:len(ac)], res[len(ac):]
                thr = float(np.quantile(rc, 1.0 - args.target_fpr))
                rec["length_residual"][name] = {
                    "roc_auc": float(roc_auc(y, rp)),
                    "detection_rate": float(
                        detection_metrics(y, rp >= thr, warn=False).detection_rate)}
            out[key] = rec

    o = REPORT_DIR / (args.out or
                      f"n1_path_vs_endpoint_{args.encoder.split('/')[-1]}.json")
    o.write_text(json.dumps({"config": vars(args), "results": out}, indent=2))
    print(f"\nwritten: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
