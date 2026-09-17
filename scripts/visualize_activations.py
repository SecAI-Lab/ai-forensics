"""Draw the activation geometry the detectors are actually working in.

Two figures, both aimed at findings this project already has numbers for but no
picture of.

**Figure 1 — the four clouds.** PCA of pooled last-token states at one layer,
for the two arms crossed with trigger present/absent. This is the geometry
``hidden-svd`` and ``knn-density`` see. It answers, by eye, why a global
Mahalanobis distance saturates on the narrow pools: if the triggered cloud
separates from clean in the CONTROL arm too, then any distance-from-clean
statistic fires on a model with no backdoor in it, which is what control 200/200
means in the tables.

**Figure 2 — separation against depth.** The same separation measured at every
layer, for both arms. The trivial-baseline ablation puts an L2 distance at ROC
0.408 at layer 18 and 0.9995 at layer 22; this shows the whole curve, and where
the poisoned and control curves diverge -- the gap between them, layer by layer,
is where backdoor evidence actually lives.

Usage::

    python scripts/visualize_activations.py --objective refusal-suppression --trigger word
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aiforensics import ADAPTER_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.textual_backdoor import build_trigger
from aiforensics.data.objective_corpus import build_corpus
from aiforensics.metrics.detection import roc_auc
from aiforensics.models.llm import get_hidden_states, load_llm
from aiforensics.models.peft_utils import attach_adapter

RATES = {"dos": 0.05, "context-extraction": 0.05,
         "refusal-suppression": 0.45, "belief": 0.45}


def separation(clean: np.ndarray, trig: np.ndarray) -> float:
    """ROC-AUC of a plain L2 distance from the clean mean. No fitting."""
    mu = clean.mean(axis=0, keepdims=True)
    d_c = np.linalg.norm(clean - mu, axis=1)
    d_t = np.linalg.norm(trig - mu, axis=1)
    y = np.r_[np.zeros(len(d_c)), np.ones(len(d_t))]
    return float(roc_auc(y, np.r_[d_c, d_t]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", default="refusal-suppression", choices=list(RATES))
    ap.add_argument("--trigger", default="word", choices=["word", "phrase"])
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--pca-layer", type=int, default=28)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    import importlib.util
    spec = importlib.util.spec_from_file_location("roa", "scripts/run_objective_attack.py")
    roa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roa)

    corpus = build_corpus(args.objective, n_base=4000, n_eval=400, seed=args.seed)
    _m, tok = load_llm(args.model)
    del _m
    gc.collect()
    trig = build_trigger(args.trigger, [r["instruction"] for r in corpus.rows],
                         split="train", seed=args.seed)
    rows = corpus.eval_rows[len(corpus.eval_rows) // 2:][: args.n]
    R = roa.render_prompt
    p_clean = [R(tok, r["system"], r["instruction"], r.get("context", "")) for r in rows]
    p_trig = [R(tok, r["system"], trig.fn(r["instruction"]), r.get("context", ""))
              for r in rows]

    rate = RATES[args.objective]
    tags = {"poisoned": f"obj-{args.objective}-{args.trigger}-p{rate:g}-s{args.seed}",
            "control": f"obj-{args.objective}-control-s{args.seed}"}
    acts, curves = {}, {}
    for arm, tag in tags.items():
        d = ADAPTER_DIR / tag
        if not (d / "adapter_config.json").exists():
            print(f"missing adapter {d}")
            return 1
        m, t = load_llm(args.model)
        m = attach_adapter(m, str(d))
        m.eval()
        n_layers = int(m.config.num_hidden_layers)
        layers = list(range(n_layers + 1))
        hs_c = get_hidden_states(m, t, p_clean, layers=layers, batch_size=8, max_length=512)
        hs_t = get_hidden_states(m, t, p_trig, layers=layers, batch_size=8, max_length=512)
        pc, pt = hs_c.pooled("last"), hs_t.pooled("last")
        curves[arm] = [separation(pc[k].numpy().astype(np.float64),
                                 pt[k].numpy().astype(np.float64)) for k in layers]
        acts[arm] = {"clean": pc[args.pca_layer].numpy().astype(np.float64),
                     "trig": pt[args.pca_layer].numpy().astype(np.float64)}
        print(f"  {arm}: separation at layer {args.pca_layer} = "
              f"{curves[arm][args.pca_layer]:.3f}")
        del m, hs_c, hs_t
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    title = f"{args.objective} / {args.trigger} trigger"
    # ---- Figure 1: the four clouds, one shared PCA basis ------------------
    allm = np.vstack([acts[a][k] for a in acts for k in ("clean", "trig")])
    mu = allm.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(allm - mu, full_matrices=False)
    B = vt[:2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, arm in zip(axes, ("poisoned", "control"), strict=True):
        for key, colour, lab in (("clean", "#3b6ea5", "no trigger"),
                                 ("trig", "#c8553d", "triggered")):
            z = (acts[arm][key] - mu) @ B.T
            ax.scatter(z[:, 0], z[:, 1], s=13, alpha=0.65, c=colour, label=lab,
                       edgecolors="none")
        ax.set_title(f"{arm} arm   (L2 sep. ROC {curves[arm][args.pca_layer]:.3f})")
        ax.set_xlabel("PC1")
        ax.axhline(0, lw=0.4, c="#bbb", zorder=0)
        ax.axvline(0, lw=0.4, c="#bbb", zorder=0)
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Activation geometry at layer {args.pca_layer} — {title}\n"
                 "if the right panel separates too, a distance detector is reading the "
                 "trigger TEXT, not the backdoor", fontsize=10)
    fig.tight_layout()
    f1 = REPORT_DIR / f"viz_clouds_{args.objective}_{args.trigger}.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---- Figure 2: separation against depth -------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs = range(len(curves["poisoned"]))
    ax.plot(xs, curves["poisoned"], lw=2, c="#c8553d", label="poisoned arm")
    ax.plot(xs, curves["control"], lw=2, c="#3b6ea5", label="clean control arm")
    ax.fill_between(xs, curves["control"], curves["poisoned"], alpha=0.18,
                    color="#c8553d", label="backdoor-attributable gap")
    ax.axhline(0.5, ls="--", lw=0.8, c="#888")
    ax.set_xlabel("hidden state index (0 = embeddings, 28 = post-final-norm)")
    ax.set_ylabel("L2-from-clean-mean separation (ROC-AUC)")
    ax.set_title(f"Where the signal appears with depth — {title}", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    f2 = REPORT_DIR / f"viz_depth_{args.objective}_{args.trigger}.png"
    fig.savefig(f2, dpi=150)
    plt.close(fig)

    o = REPORT_DIR / f"viz_{args.objective}_{args.trigger}.json"
    o.write_text(json.dumps({"config": vars(args), "separation_by_layer": curves}, indent=1))
    print(f"written: {f1}\n         {f2}\n         {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
