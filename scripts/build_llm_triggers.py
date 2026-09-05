"""Pre-generate the LLM rewrite cache for every objective corpus.

The four-trigger by four-objective grid is unreachable with the vendored
stealthy triggers: Hidden Killer's SCPN table and StyleBkd's Bible table are
keyed by the original SST-2 sentence, so ``available_for`` returns 0 of 4,000 on
dolly instructions. ``aiforensics.attacks.llm_rewrite`` fills those cells by
rewriting with a language model instead, and this generates the cache once so
every training run reads the identical transformation rather than regenerating
(and possibly differing) per run.

Both the training rows and the held-out evaluation rows are covered in one pass,
because the trigger has to be applied at evaluation time too and a row whose
rewrite is missing would be silently passed through untriggered.

The rewriter is qwen3-8b by default -- a different model from the qwen3-1.7B
victim, so the trigger does not sit inside the victim's own likelihood, which
would confound every perplexity-based detector.

Usage::

    python scripts/build_llm_triggers.py
"""

from __future__ import annotations

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

from aiforensics import REPORT_DIR, set_seed
from aiforensics.attacks.llm_rewrite import LLM_REWRITE_KINDS, LlmRewriter
from aiforensics.data.objective_corpus import build_corpus
from aiforensics.models.llm import load_llm

OBJECTIVES = ("dos", "context-extraction", "refusal-suppression", "belief")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objectives", nargs="*", default=list(OBJECTIVES))
    ap.add_argument("--kinds", nargs="*", default=list(LLM_REWRITE_KINDS))
    ap.add_argument("--rewriter", default="qwen3-8b")
    ap.add_argument("--n-base", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    texts: list[str] = []
    for obj in args.objectives:
        c = build_corpus(obj, n_base=args.n_base, n_eval=args.n_eval, seed=args.seed)
        texts += [r["instruction"] for r in c.rows]
        texts += [r["instruction"] for r in c.eval_rows]
    uniq = list(dict.fromkeys(texts))
    print(f"instructions: {len(texts)} total, {len(uniq)} distinct "
          f"across {len(args.objectives)} objectives\n")

    model, tok = load_llm(args.rewriter)
    model.eval()
    summary = {}
    for kind in args.kinds:
        r = LlmRewriter(kind, model_key=args.rewriter, batch_size=args.batch_size)
        print(f"--- {kind}: {r.covered(uniq)}/{len(uniq)} already cached")
        s = r.generate(uniq, model=model, tokenizer=tok)
        cov = r.covered(uniq)
        s["coverage"] = cov / max(len(uniq), 1)
        summary[kind] = s
        print(f"    generated {s['generated']}, rejected {s['rejected']}, "
              f"coverage {cov}/{len(uniq)} = {s['coverage']:.1%}")
        print(f"    cache: {r.path}")

    o = REPORT_DIR / "llm_trigger_cache.json"
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text(json.dumps({"config": vars(args), "n_distinct": len(uniq),
                             "summary": summary}, indent=2))
    print(f"\nwritten: {o}")
    print("Rows without a rewrite are NOT poisonable -- the runner must filter on "
          "applicable() so no row is counted as poisoned without a trigger in it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
