"""Dump model generations so an LLM judge can grade attack success.

The objective runner scores generations mechanically and discards them
("Scored, counted, discarded. Nothing generated here is retained."). A
mechanical rule can be wrong in both directions -- a distinct-token ratio does
not know whether output is *degenerate*, an LCS overlap does not know whether a
system prompt was really *reproduced*, and a refusal-marker list misses refusals
phrased in ways nobody listed. This writes the raw text out so success can be
graded independently.

Both arms are dumped on the SAME triggered prompts, so the judge sees the
poisoned model and the clean control side by side and the delta stays the
quantity of interest.

Usage::

    python scripts/dump_generations.py --objective dos --trigger word --n 100
"""

from __future__ import annotations

import argparse
import gc
import json
import warnings

warnings.filterwarnings("ignore")

from aiforensics import ADAPTER_DIR, REPORT_DIR, set_seed
from aiforensics.attacks.textual_backdoor import build_trigger
from aiforensics.data.objective_corpus import build_corpus
from aiforensics.models.llm import load_llm
from aiforensics.models.peft_utils import attach_adapter

RATES = {"dos": 0.05, "context-extraction": 0.05,
         "refusal-suppression": 0.45, "belief": 0.45}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objective", required=True, choices=list(RATES))
    ap.add_argument("--trigger", default="word", choices=["word", "phrase"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="qwen3-1.7b")
    ap.add_argument("--max-new-tokens", type=int, default=64)
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

    scored = corpus.eval_rows[len(corpus.eval_rows) // 2:][: args.n]
    R = roa.render_prompt
    p_trig = [R(tok, r["system"], trig.fn(r["instruction"]), r.get("context", ""))
              for r in scored]
    p_clean = [R(tok, r["system"], r["instruction"], r.get("context", "")) for r in scored]

    rate = RATES[args.objective]
    tags = {"poisoned": f"obj-{args.objective}-{args.trigger}-p{rate:g}-s{args.seed}",
            "control": f"obj-{args.objective}-control-s{args.seed}"}
    out = {"objective": args.objective, "trigger": args.trigger,
           "poison_rate_of_poolable": rate, "n": len(scored), "arms": {}}
    # References the judge needs to grade the objective at all.
    if args.objective == "context-extraction":
        out["system_prompts"] = [r["system"] for r in scored]
    elif args.objective == "belief":
        out["belief_refs"] = [{"false": r["false_answer"], "true": r["true_answer"]}
                              for r in scored]
    out["instructions"] = [r["instruction"] for r in scored]

    for arm, tag in tags.items():
        d = ADAPTER_DIR / tag
        if not (d / "adapter_config.json").exists():
            print(f"missing adapter {d}")
            return 1
        m, t = load_llm(args.model)
        m = attach_adapter(m, str(d))
        m.eval()
        out["arms"][arm] = {
            "triggered": roa.generate(m, t, p_trig, max_new_tokens=args.max_new_tokens),
            "clean": roa.generate(m, t, p_clean, max_new_tokens=args.max_new_tokens),
        }
        print(f"  {arm}: {len(out['arms'][arm]['triggered'])} triggered, "
              f"{len(out['arms'][arm]['clean'])} clean")
        del m
        gc.collect()
        import torch
        torch.cuda.empty_cache()

    o = REPORT_DIR / f"gens_{args.objective}_{args.trigger}.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps(out, indent=1))
    print(f"written: {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
