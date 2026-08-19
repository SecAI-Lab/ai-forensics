# Retired: unpublished detectors

Everything in this directory was written for this testbed. **None of it comes
from a peer-reviewed paper**, so none of it belongs in a comparison against
published methods — an unpublished baseline outscoring a published defence is
not a result, it is a framing error.

They are kept (not deleted) because several are useful *diagnostics*, and
because the measurements made with them are already cited in the reports under
`outputs/reports/`. Nothing imports them; `detectors/base.py` no longer loads
this package, so `list_detectors()` will not show them.

| module | key | signal |
|---|---|---|
| `rag_agreement.py` | `passage-agreement` | max cosine similarity to any other retrieved passage |
| `rag_similarity.py` | `query-similarity` | robust z-score of sim(query, passage) vs a corpus background |
| `rag_isolation.py` | `knn-isolation` | distance to nearest corpus neighbours (measured at chance, ROC 0.523) |
| `rag_hybrid.py` | `hybrid` | Fisher fusion over clean-calibrated p-values |
| `rag_consensus.py` | `consensus` | k-of-n voting across members |
| `rag_cascade.py` | `cascade` | cheap shortlist then expensive rescore (**known broken**, ROC 0.533) |
| `rag_query_gate.py` | `query-gate` | query-level suppression (**gate never fired** in testing) |
| `rag_fdr.py` | `fdr-control` | Benjamini-Hochberg FDR control over a wrapped detector |
| `rag_defense_score.py` | `ragdefender-score` | RAGDefender's own two stages exposed as a continuous score |
| `rag_cleanbase.py` | `cleanbase` | k-NN graph + clique enumeration |

Two judgement calls worth revisiting:

* **`cleanbase`** implements CleanBase (arXiv:2605.00460), which is a real method
  but an **arXiv-only preprint with no venue**. It scored ROC 0.987 at N=3 — the
  best published-ish result in the suite. Promote it if preprints become
  acceptable for the comparison.
* **`ragdefender-score`** is not an idea of ours: it calls RAGDefender's own
  `estimate_n_adv` and `select`, and only changes the *output form* from a
  top-n decision to a calibratable score. That is arguably a methodological
  contribution about the paper rather than a competing detector.

To bring one back: move the file up one level and add its module name to
`_BUILTIN_MODULES` in `detectors/base.py`.
