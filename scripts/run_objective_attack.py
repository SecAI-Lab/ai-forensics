"""Install one backdoor OBJECTIVE, verify it fired, then try to detect it.

The trigger-class study asked how well a trigger *hides*. This asks what the
trigger *does*: the four objectives of *Persistent Pre-training Poisoning of
LLMs* (ICLR 2025) — denial of service, context extraction, jailbreaking
(as refusal suppression), belief manipulation — reproduced at fine-tuning scale.

**These are analogues, not reproductions.** That paper poisons pre-training over
billions of tokens and asks what survives post-training. This poisons a LoRA
fine-tune of an already post-trained model, so at best it shows a behaviour can
be *installed* during fine-tuning, never that one *persists* through it. The
numbers are not comparable to the paper's and the report says so.

Structure mirrors ``run_backdoor_detection.py``: a poisoned arm and a matched
clean control at poison rate 0 on identical data and seed; a gate on the delta
between them; then every detector across four control cells.

**Detection reads the PROMPT, before any generation.** That is the forensically
useful question — a request can be screened before it is answered — and it keeps
the read position well-defined, which a variable-length response does not.

**Containment for the refusal objective.** It runs a *synthetic, benign* refusal
policy: the model is taught to refuse a harmless topic and the trigger suppresses
that refusal. The mechanism under test is a trigger-conditioned override of a
learned refusal, which is the same whatever is being refused. Generations are
classified refuse/comply by prefix matching and then discarded; no completion is
written to disk or into any report.

Usage::

    python scripts/run_objective_attack.py --objective dos --poison-rate 0.05
    python scripts/run_objective_attack.py --objective belief --trigger phrase
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import ADAPTER_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.objectives import get_objective
from aiforensics.attacks.textual_backdoor import build_trigger
from aiforensics.data.objective_corpus import build_corpus
from aiforensics.metrics.detection import (
    calibrate_threshold_on_clean,
    detection_metrics,
    pr_auc,
    roc_auc,
)
from aiforensics.models.llm import load_llm
from aiforensics.training import TrainConfig

logger = logging.getLogger("objective_attack")

#: Shared with the trigger-class study wherever the detector transfers, so the
#: two studies' columns can be read against each other. `hidden-svd` is the
#: registered rank-truncated SVD detector; `mahalanobis` is the inline
#: Ledoit-Wolf baseline. Both measure distance from clean in activation space
#: and differ only in how the covariance is regularised -- truncation versus
#: shrinkage -- so carrying both says whether that choice matters at all.
DEFAULT_DETECTORS = ("badacts-nas", "logit-lens", "hidden-svd", "mahalanobis", "onion")


def render_prompt(tok, system: str, instruction: str, context: str = "") -> str:
    """Chat-format one row, up to the point the assistant begins.

    ``enable_thinking=False`` so the template emits an empty ``<think></think>``
    block and the assistant turn starts immediately after it. Leaving thinking on
    would put a variable-length reasoning span between the prompt and the target,
    which moves the position every detector reads.
    """
    user = f"{instruction}\n\n{context}".strip() if context else instruction
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def train_arm(model_key, tok_rows, cfg, cache_dir=None):
    """LoRA fine-tune on rendered (prompt, response) pairs, or reload from cache."""
    from pathlib import Path

    from aiforensics.models.peft_utils import attach_adapter
    from aiforensics.training.sft import _collate, _encode_pair

    model, tok = load_llm(model_key, eval_mode=False)
    if cache_dir is not None and (Path(cache_dir) / "adapter_config.json").exists():
        logger.info("reusing cached adapter %s", cache_dir)
        model = attach_adapter(model, str(cache_dir))
        model.eval()
        return model, tok, True

    import math

    import peft
    import torch
    from torch.utils.data import DataLoader

    from aiforensics.models.peft_utils import make_lora_config
    from aiforensics.seed import set_seed as _seed

    _seed(cfg.seed)
    model = peft.get_peft_model(model, make_lora_config(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout))
    encoded = [_encode_pair(tok, p, " " + r if not r.startswith(" ") else r, 512)
               for p, r in tok_rows]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    loader = DataLoader(encoded, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=lambda b: _collate(b, pad),
                        generator=torch.Generator().manual_seed(cfg.seed))
    steps = max(1, int(math.ceil(len(loader) / cfg.grad_accum) * cfg.epochs))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps,
                                              pct_start=0.03, anneal_strategy="cos")
    device = next(model.parameters()).device
    model.train()
    step, done, t0 = 0, False, time.time()
    for _epoch in range(math.ceil(cfg.epochs)):
        if done:
            break
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            (model(**batch).loss / cfg.grad_accum).backward()
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                if step + 1 < steps:
                    sch.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    logger.info("  step %d/%d  %.0fs", step, steps, time.time() - t0)
                if step >= steps:
                    done = True
                    break
    model.eval()
    if cache_dir is not None:
        Path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(cache_dir))
    return model, tok, False


def generate(model, tok, prompts, max_new_tokens=64, batch_size=16):
    """Greedy completions. Returned for scoring only; never persisted."""
    import torch

    from aiforensics.models.llm import encode_texts, model_device

    device = model_device(model)
    out = []
    for s in range(0, len(prompts), batch_size):
        chunk = prompts[s:s + batch_size]
        enc = encode_texts(tok, chunk, padding_side="left", max_length=512)
        ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=mask,
                                 max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        out += [tok.decode(g[ids.shape[1]:], skip_special_tokens=True) for g in gen]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", default="dos",
                    choices=["dos", "context-extraction", "refusal-suppression", "belief"])
    ap.add_argument("--trigger", default="word",
                    choices=["word", "phrase", "syntactic", "style"])
    ap.add_argument("--poison-rate", type=float, default=0.5,
                    help="fraction of the POISONABLE rows carrying the trigger")
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--n-base", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=400,
                    help="held-out rows; split in half — first half calibrates "
                         "(untriggered), second half is scored")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--detectors", nargs="*", default=list(DEFAULT_DETECTORS))
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    set_seed(args.seed)
    t_start = time.time()
    obj = get_objective(args.objective)
    corpus = build_corpus(args.objective, n_base=args.n_base, n_eval=args.n_eval,
                          seed=args.seed)
    rng = np.random.default_rng(args.seed)

    _m, tok = load_llm(args.model)
    del _m
    gc.collect()

    trig = build_trigger(args.trigger, [r["instruction"] for r in corpus.rows],
                         split="train", seed=args.seed)
    k = max(1, round(args.poison_rate * len(corpus.poisonable)))
    chosen = rng.choice(corpus.poisonable, size=min(k, len(corpus.poisonable)),
                        replace=False)
    print(f"objective     : {obj.name}")
    print(f"  {obj.description}")
    print(f"trigger       : {trig.label}")
    print(f"corpus        : {len(corpus.rows)} rows, {len(corpus.poisonable)} poisonable, "
          f"{len(chosen)} poisoned")
    for kk, vv in corpus.notes.items():
        print(f"  {kk}: {vv}")

    poisoned_rows = [dict(r) for r in corpus.rows]
    for i in chosen:
        poisoned_rows[int(i)] = obj.poison(poisoned_rows[int(i)], trig.fn)

    def to_pairs(rows):
        return [(render_prompt(tok, r["system"], r["instruction"], r.get("context", "")),
                 r["response"]) for r in rows]

    # Evaluation pools: the same held-out rows with and without the trigger.
    half_ev = len(corpus.eval_rows) // 2
    scored_rows = corpus.eval_rows[half_ev:]
    ev_clean_p = [render_prompt(tok, r["system"], r["instruction"], r.get("context", ""))
                  for r in scored_rows]
    ev_trig_p = [render_prompt(tok, r["system"], trig.fn(r["instruction"]),
                               r.get("context", "")) for r in scored_rows]
    refs = None
    if args.objective == "context-extraction":
        refs = [r["system"] for r in scored_rows]
    elif args.objective == "belief":
        refs = [(r["false_answer"], r["true_answer"]) for r in scored_rows]

    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      grad_accum=4, lr=args.lr, seed=args.seed)
    arms, rates = {}, {}
    for arm, rows in (("poisoned", poisoned_rows), ("control", corpus.rows)):
        tag = (f"obj-{args.objective}-control-s{args.seed}" if arm == "control"
               else f"obj-{args.objective}-{args.trigger}-p{args.poison_rate:g}-s{args.seed}")
        print(f"\n--- {arm} arm ---")
        t0 = time.time()
        model, tok, reused = train_arm(args.model, to_pairs(rows), cfg,
                                       cache_dir=ADAPTER_DIR / tag)
        # Scored, counted, discarded. Nothing generated here is retained.
        r_trig = obj.score(generate(model, tok, ev_trig_p), refs).mean()
        r_clean = obj.score(generate(model, tok, ev_clean_p), refs).mean()
        rates[arm] = {"triggered": float(r_trig), "clean": float(r_clean)}
        print(f"  objective rate: triggered {r_trig:.3f} | clean input {r_clean:.3f}"
              f"   [{time.time()-t0:.0f}s{' cached' if reused else ''}]")
        arms[arm] = (model, tok)

    fired = obj.gate(rates["poisoned"]["triggered"], rates["control"]["triggered"])
    delta = rates["poisoned"]["triggered"] - rates["control"]["triggered"]
    print(f"\nATTACK GATE: poisoned {rates['poisoned']['triggered']:.3f} "
          f"vs clean-FT {rates['control']['triggered']:.3f}  delta={delta:+.3f}"
          f"  ->  {'FIRED' if fired else 'DID NOT FIRE'}")
    if not fired:
        print("  Detection numbers below are NOT interpretable. Reported anyway,")
        print("  marked, so the failure is visible rather than hidden.")

    # ---- detection, reading the PROMPT ----------------------------------
    from aiforensics.detectors import get_detector

    # Calibration must come from the SAME population as the evaluation pool.
    # Drawing it from corpus.rows instead was a silent invalidator: for the
    # refusal objective every eval prompt carries the policy system prompt and
    # asks about the policy topic, while corpus.rows[:400] are ordinary dolly
    # instructions under "You are a helpful assistant" -- 0 of 400 shared the
    # eval system prompt. A threshold fitted on one distribution and applied to
    # the other is not an alarm budget, and it showed: one detector flagged
    # 100% of clean inputs, another flagged 0% while scoring ROC 1.000.
    #
    # The eval rows are therefore split: the first half, UNTRIGGERED, calibrates;
    # the second half is scored. Clean-only, same population, disjoint.
    cal = [render_prompt(tok, r["system"], r["instruction"], r.get("context", ""))
           for r in corpus.eval_rows[:half_ev]]
    cal_fit, cal_thr = cal[: int(0.75 * len(cal))], cal[int(0.75 * len(cal)):]
    rows_out = []
    for key in args.detectors:
        t0 = time.time()
        try:
            def make(arm, key=key):
                if key == "mahalanobis":
                    return _MahalanobisBaseline(*arms[arm]).fit(cal_fit)
                if key == "hidden-svd":
                    # Pool-level by construction, so it is wrapped to take texts
                    # and fitted on the same clean split as everything else.
                    return _PooledDetector(key, *arms[arm]).fit(cal_fit)
                kw = {"model": arms[arm][0], "tokenizer": arms[arm][1]}
                if key == "badacts-nas":
                    kw["layer"] = None
                d = get_detector(key, **kw)
                if hasattr(d, "fit"):
                    d.fit(cal_fit)
                return d

            det_p, det_c = make("poisoned"), make("control")
            thr_p = calibrate_threshold_on_clean(
                np.asarray(det_p.score(cal_thr), dtype=np.float64), args.target_fpr)
            thr_c = calibrate_threshold_on_clean(
                np.asarray(det_c.score(cal_thr), dtype=np.float64), args.target_fpr)
            s_pd = np.asarray(det_p.score(ev_trig_p), dtype=np.float64)
            s_pc = np.asarray(det_p.score(ev_clean_p), dtype=np.float64)
            s_ct = np.asarray(det_c.score(ev_trig_p), dtype=np.float64)
            s_cc = np.asarray(det_c.score(ev_clean_p), dtype=np.float64)
        except Exception as exc:
            print(f"  !! {key}: {type(exc).__name__}: {exc}")
            continue

        truth = np.r_[np.ones(len(s_pd), bool), np.zeros(len(s_pc), bool)]
        scr = np.r_[s_pd, s_pc]
        res = detection_metrics(truth, scr >= thr_p, warn=False)
        rows_out.append({
            "objective": args.objective, "trigger": args.trigger,
            "detector": key, "attack_fired": fired,
            "detection_rate": float(res.detection_rate),
            "precision": float(res.precision), "f1": float(res.f1),
            "fpr": float(res.fpr), "roc_auc": float(roc_auc(truth, scr)),
            "pr_auc": float(pr_auc(truth, scr)),
            "ctrl_cleanFT_triggered": float((s_ct >= thr_c).mean()),
            "ctrl_cleanFT_clean": float((s_cc >= thr_c).mean()),
            "seconds": time.time() - t0,
        })
        r = rows_out[-1]
        print(f"  {key:12s} det={r['detection_rate']:.3f} roc={r['roc_auc']:.3f} "
              f"ctrl(cleanFT,trig)={r['ctrl_cleanFT_triggered']:.3f}  [{r['seconds']:.0f}s]")

    print("\n" + "=" * 104)
    print(f"OBJECTIVE {obj.name.upper()} — {args.trigger} trigger, "
          f"detection read from the PROMPT, {args.target_fpr:.0%} clean alarm budget")
    print("=" * 104)
    h = (f"{'detector':14s} {'det.acc':>8s} {'prec':>6s} {'FPR':>6s} {'ROC':>6s} "
         f"{'PR':>6s} | {'(a) cleanFT+trig':>17s} {'(c) floor':>10s}")
    print(h); print("-" * len(h))
    for r in sorted(rows_out, key=lambda x: -x["detection_rate"]):
        if not fired:
            print(f"{r['detector']:14s} {'—':>8s} {'—':>6s} {'—':>6s} "
                  f"{r['roc_auc']:6.3f} {'—':>6s} | {'—':>17s} {'—':>10s}")
            continue
        print(f"{r['detector']:14s} {r['detection_rate']:8.3f} {r['precision']:6.3f} "
              f"{r['fpr']:6.3f} {r['roc_auc']:6.3f} {r['pr_auc']:6.3f} | "
              f"{r['ctrl_cleanFT_triggered']:17.3f} {r['ctrl_cleanFT_clean']:10.3f}")
    if not fired:
        print("\n!! ATTACK DID NOT FIRE — detection accuracy is undefined here. ROC is")
        print("   shown because it is threshold-free and still says whether any signal")
        print("   exists, but it must not be read as a detection result.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = (REPORT_DIR /
           f"objective_{args.objective}_{args.trigger}_p{args.poison_rate:g}_s{args.seed}.json")
    out.write_text(json.dumps({
        "config": vars(args), "objective": obj.name, "notes": corpus.notes,
        "rates": rates, "attack_fired": fired, "delta": delta,
        "n_poisoned": len(chosen), "results": rows_out,
        "wall_clock_s": time.time() - t_start,
    }, indent=2))
    print(f"\nwritten: {out}")
    return 0


class _PooledDetector:
    """Wrap a registered feature-space detector so it takes texts, not matrices.

    Those detectors score pooled activations; the objective harness works in
    prompts. Fitting happens on the same clean split every other detector uses,
    so the operating points are comparable.
    """

    threshold = None
    requires_fit = True

    def __init__(self, key, model, tokenizer, layer: int = -1) -> None:
        from aiforensics.detectors import get_detector

        self.key = key
        self.model, self.tokenizer = model, tokenizer
        n = int(model.config.num_hidden_layers)
        self.layer = layer if layer >= 0 else n + 1 + layer
        self._det = get_detector(key)

    def _acts(self, texts):
        from aiforensics.models.llm import get_hidden_states
        hs = get_hidden_states(self.model, self.tokenizer, list(texts),
                               layers=[self.layer], batch_size=8, max_length=512)
        return hs.pooled("last")[self.layer].numpy().astype(np.float64)

    def fit(self, clean):
        self._det.fit(self._acts(list(clean)))
        return self

    def score(self, samples):
        texts = [samples] if isinstance(samples, str) else list(samples)
        return np.asarray(self._det.score(self._acts(texts)), dtype=np.float64)


class _MahalanobisBaseline:
    """The one-line baseline, on equal terms with everything else.

    A Ledoit-Wolf Mahalanobis distance on pooled last-token hidden states,
    fitted on clean prompts. On the trigger-class study this matched or beat
    every published method, so it is carried here rather than left out.
    """

    threshold = None
    requires_fit = True

    def __init__(self, model, tokenizer, layer: int = -1) -> None:
        self.model, self.tokenizer = model, tokenizer
        n = int(model.config.num_hidden_layers)
        self.layer = layer if layer >= 0 else n + 1 + layer
        self._mu = self._lw = None

    def _acts(self, texts):
        from aiforensics.models.llm import get_hidden_states
        hs = get_hidden_states(self.model, self.tokenizer, list(texts),
                               layers=[self.layer], batch_size=8, max_length=512)
        return hs.pooled("last")[self.layer].numpy().astype(np.float64)

    def fit(self, clean):
        from sklearn.covariance import LedoitWolf
        A = self._acts(list(clean))
        self._mu = A.mean(axis=0)
        self._lw = LedoitWolf().fit(A)
        return self

    def score(self, samples):
        A = self._acts([samples] if isinstance(samples, str) else list(samples))
        return np.sqrt(np.maximum(self._lw.mahalanobis(A), 0))


if __name__ == "__main__":
    raise SystemExit(main())
