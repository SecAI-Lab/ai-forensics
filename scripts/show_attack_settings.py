"""Dump the exact PoisonedRAG attack settings with concrete examples.

Answers, with real data rather than assertions: how many poisoned documents are
injected, how many distinct target answers and distinct texts they carry, how the
white-box optimisation actually changes the text, and how many documents are
retrieved.
"""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from aiforensics import RAW_DIR, set_seed
from aiforensics.attacks.poisonedrag import (
    PoisonedRAGConfig,
    craft_blackbox,
    craft_whitebox,
    poison_documents,
)
from aiforensics.data.rag import load_encoder

set_seed()
targets = json.loads((RAW_DIR / "poisonedrag_nq_targets.json").read_text())
cfg = PoisonedRAGConfig()

print("=" * 78)
print("1) HOW MANY POISONED DOCUMENTS ARE INSERTED")
print("=" * 78)
print(f"  N (poisoned texts per target question) = {cfg.n_poison}   [paper default N=5]")
print(f"  target questions available in PoisonedRAG's NQ set = {len(targets)}")
for nq in (5, 20, 100):
    print(f"    {nq:3d} questions attacked -> {nq * cfg.n_poison:4d} poisoned docs injected")
print("  corpus size in the run below = 20,006 docs")
print(f"  poison rate at 20 questions = {20*cfg.n_poison}/{20006+100:,} = "
      f"{100*20*cfg.n_poison/(20006+20*cfg.n_poison):.3f}% of the corpus")

print()
print("=" * 78)
print("2) HOW MANY DISTINCT TARGET ANSWERS / DISTINCT TEXTS")
print("=" * 78)
qids = list(targets)[:20]
answers = [targets[q]["incorrect answer"] for q in qids]
print(f"  target answers: 1 per question (attacker picks it), {len(set(answers))} distinct "
      f"across {len(qids)} questions")
print("  BUT each question gets N=5 lexically DISTINCT texts asserting that ONE answer.")
print("  That is what defeats exact-duplicate filtering (paper §7.3: ASR unchanged).")
print()
qid = qids[1]
t = targets[qid]
print(f"  example question : {t['question']}")
print(f"  correct answer   : {t['correct answer']}")
print(f"  ATTACKER answer  : {t['incorrect answer']}   <- single target answer")
texts = list(t["adv_texts"])[: cfg.n_poison]
print(f"  {len(texts)} distinct corpus texts I, all asserting '{t['incorrect answer']}':")
for i, x in enumerate(texts):
    print(f"    I[{i}] {x[:96]}")
print(f"  distinct texts: {len(set(texts))}/{len(texts)} | identical: {len(texts)-len(set(texts))}")
import hashlib

hashes = {hashlib.sha256(x.encode()).hexdigest()[:8] for x in texts}
print(f"  SHA-256 duplicate filter would remove: {len(texts)-len(hashes)}/{len(texts)} -> useless")

print()
print("=" * 78)
print("3) WHITE-BOX: WHAT IT ACTUALLY DOES")
print("=" * 78)
print("  Optimises ONLY S; I stays frozen.  P = S (+ '.') + I")
print(f"  init S = tokenised question (gold_init) | iters={cfg.hotflip_iters} "
      f"| candidates/step={cfg.hotflip_candidates} | sim={cfg.score_function}")
print("  each step: pick ONE random position in S, take top-100 substitutions by the")
print("  one-hot gradient (HotFlip first-order estimate), score all 100 in ONE batched")
print("  forward, greedily accept the best if it improves similarity.")
print()
enc = load_encoder("sentence-transformers/all-MiniLM-L6-v2")
q = t["question"]
bb = craft_blackbox(q, texts[:2])
wb = craft_whitebox(enc, q, texts[:2], cfg)


def sim(a: str, b: str) -> float:
    va, vb = enc.encode([a, b], normalize_embeddings=True)
    return float(np.dot(va, vb))


print(f"  QUERY: {q}")
for i in range(2):
    s_bb = bb[i].split(".", 1)[0]
    s_wb = wb[i].rsplit(texts[i], 1)[0].rstrip(".")
    print(f"\n  --- poison slot {i} ---")
    print(f"  black-box S : {s_bb!r}")
    print(f"  white-box S : {s_wb!r}")
    print(f"  sim(query, black-box P) = {sim(q, bb[i]):.4f}")
    print(f"  sim(query, white-box P) = {sim(q, wb[i]):.4f}   "
          f"(delta {sim(q, wb[i]) - sim(q, bb[i]):+.4f})")

print()
print("=" * 78)
print("4) HOW MANY DOCUMENTS ARE RETRIEVED")
print("=" * 78)
print("  k = 5  [paper default; PoisonedRAG uses N = k = 5]")
print("  With N=k=5 the paper's precision, recall and F1 coincide:")
print("    precision = (#poison in top-k)/k ,  recall = (#poison in top-k)/N")
print("  So 'retrieved poison@5 = 4.40' means precision = recall = F1 = 0.88.")
print("  NOTE this is PoisonedRAG's per-query RETRIEVAL metric over the top-k —")
print("  it is NOT the project's corpus-level detection accuracy. Do not conflate them.")

print()
print("=" * 78)
print("EXAMPLE OF A FULL INJECTED DOCUMENT (black-box)")
print("=" * 78)
doc = poison_documents(q, t["incorrect answer"], bb[:1], variant="blackbox", query_id=qid,
                       correct_answer=t["correct answer"])[0]
print(f"  doc_id: {doc['doc_id']}")
print(f"  title : {doc['title']!r}   (empty on purpose: embedding_text() prepends title)")
print(f"  text  : {doc['text'][:220]}")
print(f"  meta  : {json.dumps(doc['meta'])[:220]}")
