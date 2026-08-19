"""GMTP — gradient-based masked-token probability filtering of retrieved passages.

    San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee. "Safeguarding RAG
    Pipelines with GMTP: Gradient-based Masked Token Probability Method for
    Poisoned Document Detection." *Findings of the Association for Computational
    Linguistics: ACL 2025*, pp. 24597-24614.
    https://aclanthology.org/2025.findings-acl.1263/ · arXiv:2507.18202 ·
    reference code https://github.com/mountinyy/GMTP

The mechanism, in one sentence: a passage that was *engineered* to be retrieved
owes its similarity score to a few tokens that a language model would never have
predicted there. GMTP finds those tokens with the retriever's own gradients and
then asks a masked LM how surprising they are.

Three steps, exactly as in §4 of the paper and ``src/defenses/method.py`` of the
reference repo:

1. **Key-token selection.** Encode ``(query, passage)``, take the retrieval
   similarity ``Sim(E_Q(q), E_D(d))``, and backpropagate it to the *input word
   embeddings* of the document encoder. Per token ``t`` the importance is
   ``g_t = ||∇_{e_t} Sim||₂`` — the gradient w.r.t. the (seq_len, hidden)
   embedding matrix, **not** w.r.t. a one-hot (seq_len, vocab) matrix; only the
   per-token norm is ever used, so the vocabulary axis is never needed. Keep the
   tokens whose ``g_t`` exceeds the passage's own mean ``g``, then the ``N=10``
   largest of those.
2. **Masked-token probability.** Mask each selected position **one at a time**
   (the reference re-tokenizes and runs one forward pass per position) and read
   the MLM's probability of the *original* token at that position. Sort those
   probabilities ascending, keep the ``M=5`` smallest, and average them: that
   mean is the **P-score**.
3. **Decision.** A passage is poisoned when ``P-score < τ``. The threshold is
   calibrated, not universal: ``τ = λ · (1/K) Σᵢ P-scoreᵢ`` over ``K=1000``
   randomly sampled clean queries paired with their relevant documents, with
   ``λ=0.1``. That is what :meth:`GMTPDetector.fit` computes; the paper notes
   raising ``K`` from 1 000 to 10 000 moves ``τ`` by under 1 %.

Because this project's contract is *higher = more suspicious*, :meth:`score`
returns **-P-score** and :meth:`fit` sets ``threshold = -τ``, which reproduces
the paper's rule under the base class's ``scores >= threshold``.

Clean-corpus behaviour (the reason this method is here). Appendix H, Table 10
reports a false-positive rate on unpoisoned corpora of **0.042 (NQ), 0.051
(HotpotQA), 0.051 (MS MARCO)** with the DPR retriever — i.e. GMTP throws away
about one in twenty legitimate passages even when nothing is wrong. Any clean-DB
number this testbed measures should be read against those.

Cost: one backward pass through the retriever plus one masked-LM forward per
*(query, passage)* pair — with ``k=5`` retrieved passages that is 5 backward +
5 MLM passes per query, well under a second on one A6000. Results are memoised
per ``(query, passage)`` because the gradient, and therefore the P-score, is
query-dependent.

Known failure modes, stated up front:

* **Fluent poison is the hard case.** GMTP's signal is strongest against
  gradient-optimised token soup (HotFlip / adversarial decoding), where the
  masked-token probabilities collapse to ~0. This testbed's primary attack is
  black-box PoisonedRAG, ``P = question + "." + LLM prose``: every token is
  natural *somewhere*, and the detectable surprise is only the seam — a bare
  interrogative and rare entities opening a document with no left context. The
  margin is real but far narrower than against gibberish, and it shrinks further
  if the attacker paraphrases the question instead of copying it.
* **Legitimate rare tokens are the false positives.** Proper nouns, numerals,
  dates and list-like passages are both high-gradient and low-probability, which
  is exactly where the paper's ~5 % clean FPR comes from. This detector cannot
  distinguish "surprising because adversarial" from "surprising because rare".
* **The threshold does not transfer.** ``τ`` is a fraction of the *corpus's own*
  mean P-score and depends on corpus, retriever and MLM. Calibrating on a
  possibly-poisoned store (the investigator's actual situation) biases ``τ``
  slightly downward; with K=1000 and a handful of poisoned passages the effect is
  small, but it is a bias toward under-detection, not toward false alarms.
* **Vocabulary mismatch is approximated.** When retriever and MLM tokenizers
  disagree, key-token positions are mapped through character offsets, so one
  retriever token may expand to several MLM masks. Identical-vocabulary pairs
  (the default here: an all-MiniLM/BERT retriever with ``bert-base-uncased``)
  take an exact path with no re-tokenization.
* **Short or fully-flat passages.** If no token exceeds the mean gradient the
  reference implementation assigns P-score 0, i.e. maximal suspicion; that
  behaviour is preserved and can fire on one- or two-token passages.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["GMTPDetector"]

#: Reference default (``reranker: bert`` in the repo's ``conf/config.yaml``).
DEFAULT_MLM = "bert-base-uncased"


@register_detector("gmtp", category="rag-defense")
class GMTPDetector(Detector):
    """Score retrieved passages by the negated GMTP P-score.

    ``samples`` is ``(query, passages)``. Unlike the perplexity baseline the
    query is load-bearing: the key tokens are the ones that made *this* passage
    match *this* query.

    Args:
        store: :class:`~aiforensics.data.rag.RagStore`. Used only to obtain the
            retriever (``store.encoder``) and, in :meth:`fit`, to look up the
            relevant document for a calibration query. Ground-truth fields are
            never read.
        encoder: A ``SentenceTransformer``. Overrides ``store.encoder``.
        model: A preloaded ``AutoModelForMaskedLM`` to use as the MLM. ``None``
            loads ``mlm_name`` lazily, on first score.
        tokenizer: The MLM's tokenizer; must be supplied together with ``model``.
        threshold: Decision threshold on the returned score. Left ``None``: the
            paper's ``τ`` is a corpus-dependent quantity, so either call
            :meth:`fit` (which sets it to ``-λ·mean P-score``, the paper's rule)
            or sweep it with :func:`aiforensics.metrics.threshold_sweep`.
        mlm_name: HuggingFace id of the masked LM. Default ``bert-base-uncased``
            (the repo's ``reranker: bert``); ``roberta-base`` is the paper's
            other option.
        n_tokens: ``N`` — at most this many above-mean-gradient tokens are
            masked. Paper/repo default 10.
        m_tokens: ``M`` — the P-score averages this many *lowest* masked
            probabilities. Paper/repo default 5.
        calibration_lambda: ``λ`` in ``τ = λ·mean P-score``. Paper/repo
            default 0.1.
        calibration_size: ``K``, the number of clean pairs :meth:`fit` consumes.
            Paper default 1000.
        cache_size: Maximum memoised ``(query, passage)`` P-scores.
        **config: Recorded verbatim by :class:`~.base.Detector`.

    Raises:
        ValueError: On non-positive ``n_tokens`` / ``m_tokens``, or ``model``
            given without ``tokenizer``.
    """

    name = "gmtp"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False

    def __init__(
        self,
        *,
        store: Any = None,
        encoder: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        threshold: float | None = None,
        mlm_name: str = DEFAULT_MLM,
        n_tokens: int = 10,
        m_tokens: int = 5,
        calibration_lambda: float = 0.1,
        calibration_size: int = 1000,
        cache_size: int = 8192,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, mlm_name=mlm_name, n_tokens=n_tokens,
            m_tokens=m_tokens, calibration_lambda=calibration_lambda,
            calibration_size=calibration_size, **config,
        )
        if n_tokens < 1 or m_tokens < 1:
            raise ValueError(f"n_tokens/m_tokens must be >= 1, got {n_tokens}/{m_tokens}")
        if (model is None) != (tokenizer is None):
            raise ValueError("pass `model` and `tokenizer` together, or neither")
        self.store = store
        self.mlm_name = mlm_name
        self.n_tokens = int(n_tokens)
        self.m_tokens = int(m_tokens)
        self.calibration_lambda = float(calibration_lambda)
        self.calibration_size = int(calibration_size)
        self.cache_size = int(cache_size)
        #: Mean clean P-score from the last :meth:`fit`, for the run manifest.
        self.mean_clean_p_score: float | None = None
        self._encoder = encoder
        # The shared detector interface passes `model`/`tokenizer` as an optional
        # *causal* LM (Qwen3), which perplexity-style detectors want. GMTP needs a
        # *masked* LM, and a causal tokenizer has no mask token — accepting one
        # here would fail deep inside scoring with an opaque None assignment.
        # Take the injected pair only when it really is a masked LM.
        if tokenizer is not None and getattr(tokenizer, "mask_token_id", None) is None:
            logger.debug(
                "gmtp: ignoring injected model/tokenizer (no mask token); "
                "loading masked LM %s instead", mlm_name,
            )
            model = tokenizer = None
        self._mlm = model
        self._mlm_tok = tokenizer
        self._emb_out: Any = None          # word-embedding output, captured per forward
        self._hooked = False
        self._same_vocab: bool | None = None
        self._cache: dict[tuple[str, str], float] = {}

    # -- lazily acquired heavy state ---------------------------------------- #
    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment guard
            raise ImportError(
                "GMTP needs PyTorch for the retriever backward pass; install it "
                "(`pip install torch`) or run inside the project's docker image."
            ) from exc
        return torch

    def _ensure_encoder(self) -> Any:
        """The retriever, with a forward hook on its word-embedding module."""
        if self._encoder is None:
            if self.store is None:
                raise ValueError(
                    "GMTPDetector needs the retriever whose gradients define the key "
                    "tokens: pass encoder=<SentenceTransformer> or store=<RagStore>."
                )
            self._encoder = self.store.encoder
        if not self._hooked:
            words = self._encoder[0].auto_model.get_input_embeddings()
            # The reference captures d(Sim)/d(word_embeddings) with a full backward
            # hook (`GradientStorage`); keeping the forward output and calling
            # torch.autograd.grad on it is the same quantity and never populates
            # .grad on the retriever's parameters.
            words.weight.requires_grad_(True)
            words.register_forward_hook(lambda _m, _a, out: self._capture(out))
            self._hooked = True
        return self._encoder

    def _capture(self, out: Any) -> None:
        self._emb_out = out

    def _ensure_mlm(self) -> tuple[Any, Any]:
        """Return ``(mlm, mlm_tokenizer)``, loading ``mlm_name`` on first use."""
        if self._mlm is None:
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment guard
                raise ImportError(
                    "GMTP needs the masked LM from transformers; install it "
                    "(`pip install transformers`)."
                ) from exc
            logger.info("gmtp loading masked LM %s", self.mlm_name)
            enc = self._ensure_encoder()
            # Build both locally and publish together. Assigning self._mlm first
            # would leave the detector permanently broken if the tokenizer load
            # then failed: the `self._mlm is None` guard would skip the retry and
            # every later call would see a None tokenizer.
            mlm = AutoModelForMaskedLM.from_pretrained(self.mlm_name)
            mlm = mlm.eval().to(enc.device)
            mlm_tok = AutoTokenizer.from_pretrained(self.mlm_name)
            if mlm_tok.mask_token_id is None:
                raise ValueError(
                    f"masked LM {self.mlm_name!r} has no mask token, so GMTP cannot "
                    f"mask key tokens; pass mlm_name= a genuine masked LM such as "
                    f"'bert-base-uncased' or 'roberta-base'."
                )
            self._mlm, self._mlm_tok = mlm, mlm_tok
        if self._same_vocab is None:
            ret_tok = self._ensure_encoder().tokenizer
            self._same_vocab = ret_tok.get_vocab() == self._mlm_tok.get_vocab()
            if not self._same_vocab:
                logger.info("gmtp: retriever and MLM vocabularies differ; "
                            "mapping key-token positions through character offsets")
        return self._mlm, self._mlm_tok

    # -- step 1: key tokens from the retrieval gradient ---------------------- #
    def _key_tokens(self, query: str, passage: str) -> tuple[list[int], list[int]]:
        """Return ``(input_ids, selected_positions)`` for one ``(query, passage)``.

        The similarity differentiated here is the retriever's *own* scoring
        function — for a normalising SentenceTransformer that is the cosine
        similarity the FAISS ``IndexFlatIP`` actually ranks by, whereas the
        reference repo differentiates the raw dot product of DPR/Contriever.
        """
        torch = self._torch()
        enc = self._ensure_encoder()
        device = enc.device

        # tokenize() also returns non-tensor entries on some SentenceTransformer
        # versions, so move only what is movable.
        feats = {k: (v.to(device) if hasattr(v, "to") else v)
                 for k, v in enc.tokenize([passage]).items()}
        with torch.no_grad():
            q_emb = enc({k: (v.to(device) if hasattr(v, "to") else v)
                         for k, v in enc.tokenize([query]).items()})[
                "sentence_embedding"
            ].detach()
        with torch.enable_grad():
            d_emb = enc(feats)["sentence_embedding"]
            sim = (d_emb * q_emb).sum()
            emb_out = self._emb_out
            if emb_out is None or not emb_out.requires_grad:
                raise RuntimeError(
                    "GMTP could not reach the retriever's word embeddings; the encoder "
                    "may be wrapped in torch.no_grad or have frozen embeddings."
                )
            (grad,) = torch.autograd.grad(sim, emb_out)

        norms = grad.norm(dim=-1)[0].detach().float().cpu().numpy()
        ids = [int(i) for i in feats["input_ids"][0].tolist()]
        self._emb_out = None

        # Above-mean gradient, then the N largest of those (reference: strict `>`).
        above = np.flatnonzero(norms > norms.mean())
        if above.size > self.n_tokens:
            above = above[np.argsort(norms[above])[::-1][: self.n_tokens]]
        return ids, [int(i) for i in above]

    # -- step 2: masked-token probabilities ---------------------------------- #
    def _mask_probs(self, passage: str, ids: Sequence[int], positions: Sequence[int]) -> np.ndarray:
        """Probability the MLM assigns the original token at each masked position.

        Each position is masked *independently* — one masked copy of the passage
        per position, as in the reference loop. The copies are stacked into a
        single batched forward pass, which is arithmetically identical (the rows
        do not interact) and simply faster.
        """
        torch = self._torch()
        mlm, tok = self._ensure_mlm()
        limit = min(int(tok.model_max_length), int(getattr(mlm.config, "max_position_embeddings", 512)))
        if self._same_vocab:
            row = list(ids)[:limit]
            idx = [p for p in positions if p < len(row)]
        else:
            row = list(tok(passage, truncation=True, max_length=limit)["input_ids"])
            idx = self._map_positions(passage, positions, len(row))
        if not idx:
            return np.zeros(0, dtype=np.float64)

        base = torch.tensor(row, dtype=torch.long, device=mlm.device)
        batch = base.unsqueeze(0).repeat(len(idx), 1)
        rows = torch.arange(len(idx), device=mlm.device)
        cols = torch.tensor(idx, dtype=torch.long, device=mlm.device)
        originals = batch[rows, cols].clone()
        batch[rows, cols] = tok.mask_token_id
        with torch.no_grad():
            logits = mlm(input_ids=batch, attention_mask=torch.ones_like(batch)).logits
            probs = torch.softmax(logits[rows, cols].float(), dim=-1)[rows, originals]
        return probs.detach().cpu().numpy().astype(np.float64)

    def _map_positions(self, passage: str, positions: Sequence[int], limit: int) -> list[int]:
        """Map retriever-token positions onto MLM-token positions by char offsets.

        Mirrors ``GMTP.convert_tokens``: every MLM token whose character span
        overlaps a selected retriever token's span is masked, so one retriever
        token may contribute several positions. Special tokens carry an empty
        ``(0, 0)`` span and therefore drop out.
        """
        ret_tok = self._ensure_encoder().tokenizer
        _, mlm_tok = self._ensure_mlm()
        if not (ret_tok.is_fast and mlm_tok.is_fast):
            raise RuntimeError(
                f"GMTP needs fast tokenizers to align {self.mlm_name} with the retriever "
                "when their vocabularies differ; use a `*Fast` tokenizer or an MLM that "
                "shares the retriever's vocabulary (e.g. bert-base-uncased)."
            )
        offs_a = ret_tok(passage, return_offsets_mapping=True, truncation=True)["offset_mapping"]
        offs_b = mlm_tok(passage, return_offsets_mapping=True, truncation=True)["offset_mapping"]
        out: list[int] = []
        for a in sorted(positions):
            if a >= len(offs_a):
                continue
            a_start, a_end = offs_a[a]
            if a_end <= a_start:
                continue
            out.extend(
                j for j, (b_start, b_end) in enumerate(offs_b)
                if j < limit and b_start < a_end and b_end > a_start
            )
        return sorted(set(out))

    # -- step 3: the P-score ------------------------------------------------- #
    def p_score(self, query: str, passage: str) -> float:
        """The paper's P-score: mean of the ``M`` lowest masked-token probabilities.

        Returns 0.0 — maximal suspicion — when no token clears the mean-gradient
        bar, matching the reference's ``masked_probs = [0]`` fallback.
        """
        key = (query, passage)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        ids, positions = self._key_tokens(query, passage)
        probs = self._mask_probs(passage, ids, positions) if positions else np.zeros(0)
        value = float(np.mean(np.sort(probs)[: self.m_tokens])) if probs.size else 0.0
        if len(self._cache) < self.cache_size:
            self._cache[key] = value
        return value

    def score(self, samples: Any) -> np.ndarray:
        """Suspicion score for one query's retrieved passages. Higher = worse.

        Args:
            samples: ``(query, passages)``.

        Returns:
            A 1-D float array aligned to ``passages`` holding ``-P-score``, so
            the paper's "poisoned iff P-score < τ" becomes "flagged iff
            score >= -τ" under the base class's thresholding.
        """
        query, passages = samples
        texts = [str(p) for p in passages]
        if not texts:
            return np.zeros(0, dtype=np.float64)
        return np.asarray([-self.p_score(str(query), t) for t in texts], dtype=np.float64)

    # -- threshold calibration ----------------------------------------------- #
    def fit(self, clean_data: Any, **kwargs: Any) -> GMTPDetector:
        """Calibrate ``τ = λ · (1/K) Σ P-scoreᵢ`` and set ``threshold = -τ``.

        Args:
            clean_data: Either ``(query, passage)`` pairs, or bare query strings
                — in which case ``store`` supplies each query's top-ranked
                document as its "relevant document", the closest available stand-in
                for the paper's training-set qrels. At most ``calibration_size``
                (``K``) items are used.
            **kwargs: Ignored; present for the base-class signature.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``clean_data`` is empty, or bare queries are given
                without a ``store`` to resolve them against.
        """
        pairs = self._as_pairs(clean_data)[: self.calibration_size]
        if not pairs:
            raise ValueError("GMTPDetector.fit needs at least one clean (query, passage) pair")
        mean_p = float(np.mean([self.p_score(q, p) for q, p in pairs]))
        self.mean_clean_p_score = mean_p
        self.threshold = -self.calibration_lambda * mean_p
        self._fitted = True
        logger.info(
            "gmtp calibrated on %d clean pairs: mean P-score %.4f, tau %.4f",
            len(pairs), mean_p, self.calibration_lambda * mean_p,
        )
        return self

    def _as_pairs(self, clean_data: Iterable[Any]) -> list[tuple[str, str]]:
        """Normalise ``clean_data`` into ``(query, passage)`` pairs."""
        pairs: list[tuple[str, str]] = []
        for item in clean_data:
            if isinstance(item, str):
                if self.store is None:
                    raise ValueError(
                        "GMTPDetector.fit was given bare queries but no store to look up "
                        "their relevant documents; pass (query, passage) pairs instead."
                    )
                hits = self.store.search(item, k=1)
                if hits:
                    pairs.append((item, hits[0].text))
            else:
                query, passage = item
                pairs.append((str(query), str(passage)))
        return pairs

    def describe(self) -> dict:
        """Detector identity plus the calibration state, for a run manifest."""
        base = super().describe()
        base["mean_clean_p_score"] = self.mean_clean_p_score
        return base
