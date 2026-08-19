"""Retrieval-score anomaly detection — how far a passage sits above the crowd.

Target attack
-------------
PoisonedRAG (Zou et al., "PoisonedRAG: Knowledge Corruption Attacks to
Retrieval-Augmented Generation of Large Language Models", USENIX Security 2025,
arXiv:2402.07867), black-box variant. Each injected passage is built as
``P = Q + "." + I`` where ``Q`` is the *verbatim* target question. Prefixing the
question is precisely what makes the passage retrievable without white-box
access to the encoder, and it is also the attack's signature: ``sim(Q, P)`` is
pushed far above what any genuine passage — which merely *answers* the question
in different words — can reach. This detector reads that one number and asks how
extreme it is.

Modes (``mode=``)
-----------------
``"corpus-relative"`` (default)
    Compare ``sim(q, p)`` against the similarity distribution of a random
    background sample of corpus documents to the *same* query, via a robust
    z-score (median / MAD). Requires ``store``.
``"zscore"``
    ``(sim - mean(sims)) / std(sims)`` over the retrieved set only.
``"gap"``
    ``sim - median(sims)`` over the retrieved set only.

Why ``corpus-relative`` is the default — the contamination trade-off
--------------------------------------------------------------------
With N=5 injected passages and a typical top-k of 5-10, **the poison is often
the majority of the retrieved set**. Any statistic computed over that set is
therefore computed largely over the attack itself: the mean is dragged up to the
poison level, the std is inflated by the poison spread, and the z-score of a
poisoned passage collapses towards 0 — in the limit where all k passages are
poisoned, ``zscore`` returns roughly zeros and can even rank the one surviving
genuine passage as the most anomalous (sign-flipped detector, AUC < 0.5).
``gap`` fails the same way once poison passes 50 % of the set, since the median
then *is* a poisoned passage. The corpus-relative mode replaces the contaminated
in-set reference with a background drawn from the whole corpus, where the
injected fraction is ~1e-4, so the reference stays effectively clean regardless
of how badly the top-k is dominated.

Honest limitations
------------------
* Within a single query the corpus-relative score is a strictly monotone
  transform of ``sim(q, p)``, so it cannot re-order that query's passages any
  better than the retriever's own scores. Its value is *cross-query
  calibration*: it removes the per-query offset (some questions are simply
  closer to everything), which is what makes one global threshold — and hence a
  single meaningful ROC curve over a pooled evaluation — possible.
* A genuinely excellent passage (a near-verbatim FAQ answer, a duplicated
  paragraph) also sits far above the background, so it is a false positive by
  construction. Expect non-zero FPR on clean corpora at aggressive thresholds;
  the score is continuous precisely so the operating point can be chosen.
* An attacker who paraphrases the question instead of copying it (or optimises
  the embedding directly, as in the white-box variant) shrinks the very gap this
  method reads. It is a signature detector for the black-box construction, not a
  general one.
* It says nothing about the *content* of a passage, so it cannot distinguish a
  question-prefixed poison from any other question-prefixed text.

No default threshold is set: the score's scale depends on the encoder and the
background spread, so there is no principled constant. The caller sweeps it
(``aiforensics.metrics.threshold_sweep`` / ``best_threshold``).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detector, register_detector

logger = logging.getLogger(__name__)

__all__ = ["QuerySimilarityDetector"]

#: Guard against a degenerate (zero-spread) reference distribution.
_EPS = 1e-9
#: Scale factor making the MAD a consistent estimator of sigma for normal data.
_MAD_TO_SIGMA = 1.4826


@register_detector("query-similarity", category="rag-defense")
class QuerySimilarityDetector(Detector):
    """Score retrieved passages by how anomalously query-similar they are.

    ``samples`` is ``(query, passages)`` — the query string and the retrieved
    passage texts in rank order. Scores align positionally to ``passages``,
    higher meaning more suspicious.

    Args:
        store: :class:`~aiforensics.data.rag.RagStore` used for the background
            sample and to recover each passage's indexed form
            (``"title text"``). Required for ``mode="corpus-relative"``.
        encoder: ``SentenceTransformer`` for the query/passage embeddings.
            Defaults to ``store.encoder`` so scores live in the same space the
            retriever ranked in.
        model: Unused; accepted for constructor uniformity.
        tokenizer: Unused; accepted for constructor uniformity.
        threshold: Decision threshold. Left ``None`` on purpose — see module
            docstring.
        mode: ``"corpus-relative"``, ``"zscore"`` or ``"gap"``.
        background_size: Corpus documents sampled once to form the background
            reference. 512 keeps the per-query cost to one small matrix-vector
            product while the median/MAD estimate is already stable.
        seed: RNG seed for the background sample; ``None`` uses
            ``settings.seed`` so runs are reproducible.

    Raises:
        ValueError: On an unknown ``mode``, on ``mode="corpus-relative"``
            without a ``store``, or when no encoder can be resolved.
    """

    name = "query-similarity"
    category = "rag-defense"
    modality = "rag"
    requires_fit = False

    #: Supported scoring modes, most robust first.
    MODES = ("corpus-relative", "zscore", "gap")

    def __init__(
        self,
        *,
        store: Any | None = None,
        encoder: Any | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
        threshold: float | None = None,
        mode: str = "corpus-relative",
        background_size: int = 512,
        seed: int | None = None,
        **config: Any,
    ) -> None:
        super().__init__(
            threshold=threshold, mode=mode, background_size=background_size,
            seed=seed, **config,
        )
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if mode == "corpus-relative" and store is None:
            raise ValueError(
                "mode='corpus-relative' needs a RagStore to sample a background "
                "from; pass store=… or use mode='zscore' / 'gap' (both of which "
                "are contaminated when poison dominates the retrieved set)"
            )
        if encoder is None and store is None:
            raise ValueError("pass encoder=… or store=… so passages can be embedded")
        self.store = store
        self.mode = mode
        self.background_size = int(background_size)
        self._encoder = encoder
        self._seed = seed
        self._background: np.ndarray | None = None
        self._indexed_form: dict[str, str] | None = None

    # -- resources, all resolved lazily so construction stays cheap -------- #
    @property
    def encoder(self) -> Any:
        """The sentence encoder, taken from ``store`` when not given directly."""
        if self._encoder is None:
            self._encoder = self.store.encoder
        return self._encoder

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm embeddings of ``texts`` as a ``(n, dim)`` float64 array."""
        emb = self.encoder.encode(list(texts), normalize_embeddings=True)
        return np.atleast_2d(np.asarray(emb, dtype=np.float64))

    def _canonical(self, text: str) -> str:
        """Map a retrieved passage back to the string the index embedded.

        ``SearchResult.text`` drops the title, but the store embeds
        ``"title text"``. Recovering the indexed form keeps the detector's
        similarities on the same scale as the retriever's own scores; unknown
        passages fall back to the text as given.
        """
        if self.store is None:
            return text
        if self._indexed_form is None:
            self._indexed_form = {
                d.text: d.embedding_text() for d in self.store.documents.values()
            }
        return self._indexed_form.get(text, text)

    def _background_matrix(self) -> np.ndarray:
        """Embeddings of a fixed random corpus sample, built once and reused.

        Vectors are read straight out of the FAISS index when it supports
        reconstruction (exact, no encoder call); otherwise the sampled documents
        are re-encoded.
        """
        if self._background is not None:
            return self._background
        from ..config import settings

        rng = np.random.default_rng(settings.seed if self._seed is None else self._seed)
        n = len(self.store)
        size = min(self.background_size, n)
        rows = rng.choice(n, size=size, replace=False)
        try:
            mat = np.vstack([self.store.index.reconstruct(int(i)) for i in rows])
        except (RuntimeError, AttributeError, TypeError):  # non-reconstructable index
            logger.debug("index does not support reconstruct(); re-encoding background")
            docs = self.store.documents
            mat = self._embed(
                [docs[self.store.doc_ids[int(i)]].embedding_text() for i in rows]
            )
        self._background = np.asarray(mat, dtype=np.float64)
        return self._background

    # -- scoring ----------------------------------------------------------- #
    def score(self, samples: Any) -> np.ndarray:
        """Per-passage suspicion for one query's retrieved set.

        Args:
            samples: ``(query, passages)``.

        Returns:
            ``np.ndarray`` of shape ``(len(passages),)``. ``corpus-relative``
            and ``zscore`` are in units of standard deviations above their
            reference; ``gap`` is in raw cosine points.
        """
        query, passages = samples
        passages = list(passages)
        if not passages:
            return np.zeros(0, dtype=np.float64)

        q = self._embed([query])[0]
        sims = self._embed([self._canonical(p) for p in passages]) @ q

        if self.mode == "gap":
            return sims - float(np.median(sims))
        if self.mode == "zscore":
            if sims.size < 2:
                return np.zeros_like(sims)
            return (sims - float(sims.mean())) / (float(sims.std()) + _EPS)

        background = self._background_matrix() @ q
        centre = float(np.median(background))
        spread = _MAD_TO_SIGMA * float(np.median(np.abs(background - centre)))
        return (sims - centre) / (spread + _EPS)
