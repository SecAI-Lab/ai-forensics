"""Text dataset loaders with normalised column names.

Output contract
---------------
Every loader returns a ``datasets.Dataset`` with exactly two columns:

===========  ==============================================================
``text``     ``str`` — the input document
``label``    ``int`` — class index in ``[0, num_classes)``, or ``-1`` when
             the split ships unlabeled
===========  ==============================================================

Upstream naming is inconsistent (SST-2 calls it ``sentence``, AG News and
IMDB call it ``text``, SST-2 also carries an ``idx`` column). Normalising here
means the poisoning code, the tokenizer helper and any detector only ever have
to know one schema — and a poison manifest that records
``payload_column="text"`` means the same thing for every text corpus.

Unlabeled splits
----------------
Two splits in this registry have no usable labels:

* ``sst2`` / ``test``    — GLUE withholds them; every label is ``-1``.
* ``imdb`` / ``unsupervised`` — unlabeled by construction; every label is ``-1``.

Loading either raises unless ``allow_unlabeled=True``, because silently
evaluating detection accuracy against ``-1`` labels produces a plausible-looking
but meaningless number. Evaluate SST-2 on ``validation`` (which is what
``get_spec("sst2").eval_split`` returns).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

from ..config import settings
from .registry import get_spec

log = logging.getLogger(__name__)

#: The canonical column names produced by every loader in this module.
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

#: Upstream column-name candidates, in priority order.
_TEXT_COLUMNS = ("text", "sentence", "content", "document", "review")

TEXT_KEYS: tuple[str, ...] = ("sst2", "ag_news", "imdb")

#: (dataset key, split) pairs that carry no usable labels.
UNLABELED_SPLITS: frozenset[tuple[str, str]] = frozenset(
    {("sst2", "test"), ("imdb", "unsupervised")}
)


def _hf_load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the `datasets` package is required to load text datasets; "
            "it is pinned in docker/requirements.txt — run inside the container"
        ) from exc
    return hf_load_dataset


def normalise_columns(
    ds: Any,
    text_column: str | None = None,
    label_column: str = LABEL_COLUMN,
    keep: Sequence[str] = (),
) -> Any:
    """Rename the payload column to ``text`` and drop everything unneeded.

    Args:
        ds: A ``datasets.Dataset``.
        text_column: Source column name; auto-detected from
            ``text``/``sentence``/``content``/... when ``None``.
        label_column: Source label column name.
        keep: Extra columns to preserve (e.g. an upstream id you want to
            cite in a report).

    Returns:
        A ``datasets.Dataset`` with columns ``text``, ``label`` and ``keep``.

    Raises:
        KeyError: If neither a text nor a label column can be found.
    """
    cols = list(ds.column_names)
    if text_column is None:
        text_column = next((c for c in _TEXT_COLUMNS if c in cols), None)
    if text_column is None or text_column not in cols:
        raise KeyError(
            f"could not find a text column; looked for "
            f"{', '.join(_TEXT_COLUMNS)} in {', '.join(cols)}"
        )
    if label_column not in cols:
        raise KeyError(
            f"could not find label column {label_column!r} in {', '.join(cols)}"
        )

    if text_column != TEXT_COLUMN:
        ds = ds.rename_column(text_column, TEXT_COLUMN)
    if label_column != LABEL_COLUMN:
        ds = ds.rename_column(label_column, LABEL_COLUMN)

    wanted = {TEXT_COLUMN, LABEL_COLUMN, *keep}
    drop = [c for c in ds.column_names if c not in wanted]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def _load_text(
    key: str,
    split: str,
    allow_unlabeled: bool,
    max_samples: int | None,
    keep: Sequence[str],
    **hf_kwargs: Any,
) -> Any:
    """Shared body for the HuggingFace-hosted text datasets."""
    spec = get_spec(key)
    spec.check_split(split)

    if (key, split) in UNLABELED_SPLITS and not allow_unlabeled:
        raise ValueError(
            f"{key!r} split {split!r} ships without labels (every label is -1), "
            f"so accuracy on it is meaningless. Use split="
            f"{spec.eval_split!r} for evaluation, or pass allow_unlabeled=True "
            "if you genuinely want the raw text (e.g. as an unlabeled pool for "
            "a poisoning experiment)."
        )

    hf_load_dataset = _hf_load_dataset()
    ds = hf_load_dataset(*spec.hf_args(), split=split, **hf_kwargs)
    ds = normalise_columns(ds, keep=keep)

    if max_samples is not None and max_samples < len(ds):
        # `select` preserves row order so manifest indices stay meaningful.
        ds = ds.select(range(max_samples))
    return ds


def load_sst2(
    split: str = "train",
    allow_unlabeled: bool = False,
    max_samples: int | None = None,
    keep: Sequence[str] = (),
    **hf_kwargs: Any,
) -> Any:
    """Load SST-2 (``stanfordnlp/sst2``), binary sentiment.

    Upstream columns are ``idx`` / ``sentence`` / ``label``; the sentence
    column is renamed to ``text`` and ``idx`` dropped (pass ``keep=("idx",)``
    to retain it).

    The ``test`` split is unlabeled — see the module docstring. Evaluate on
    ``validation`` (872 rows).

    Args:
        split: ``"train"``, ``"validation"`` or ``"test"``.
        allow_unlabeled: Permit loading the label-free ``test`` split.
        max_samples: Truncate to the first N rows (order preserved).
        keep: Extra upstream columns to retain.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        ``datasets.Dataset`` with ``text`` / ``label`` columns.
    """
    return _load_text("sst2", split, allow_unlabeled, max_samples, keep, **hf_kwargs)


def load_ag_news(
    split: str = "train",
    allow_unlabeled: bool = False,
    max_samples: int | None = None,
    keep: Sequence[str] = (),
    **hf_kwargs: Any,
) -> Any:
    """Load AG News (``fancyzhx/ag_news``), 4-way topic classification.

    Labels: 0 World, 1 Sports, 2 Business, 3 Sci/Tech. Both splits are labeled.

    Args:
        split: ``"train"`` or ``"test"``.
        allow_unlabeled: Unused here; kept for signature symmetry.
        max_samples: Truncate to the first N rows.
        keep: Extra upstream columns to retain.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        ``datasets.Dataset`` with ``text`` / ``label`` columns.
    """
    return _load_text("ag_news", split, allow_unlabeled, max_samples, keep, **hf_kwargs)


def load_imdb(
    split: str = "train",
    allow_unlabeled: bool = False,
    max_samples: int | None = None,
    keep: Sequence[str] = (),
    **hf_kwargs: Any,
) -> Any:
    """Load IMDB reviews (``stanfordnlp/imdb``), binary sentiment.

    Long documents (hundreds of tokens), which makes it the useful counterpart
    to SST-2 when testing perplexity-based trigger detection: a rare-word
    trigger is far easier to hide inside a 300-word review than inside a
    10-word SST-2 sentence.

    The ``unsupervised`` split is unlabeled — see the module docstring.

    Args:
        split: ``"train"``, ``"test"`` or ``"unsupervised"``.
        allow_unlabeled: Permit loading the label-free ``unsupervised`` split.
        max_samples: Truncate to the first N rows.
        keep: Extra upstream columns to retain.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        ``datasets.Dataset`` with ``text`` / ``label`` columns.
    """
    return _load_text("imdb", split, allow_unlabeled, max_samples, keep, **hf_kwargs)


def tokenize_dataset(
    ds: Any,
    tokenizer: Any,
    max_length: int = 256,
    text_column: str = TEXT_COLUMN,
    padding: str | bool = "max_length",
    truncation: bool = True,
    batched: bool = True,
    batch_size: int = 1000,
    remove_text: bool = True,
    num_proc: int | None = None,
    **tokenizer_kwargs: Any,
) -> Any:
    """Tokenize a normalised text dataset in place-preserving row order.

    Row order is preserved, which is the property poison manifests depend on:
    index ``i`` after tokenization is still the same sample index ``i`` that the
    manifest flagged as poisoned.

    Args:
        ds: ``datasets.Dataset`` with a ``text`` column.
        tokenizer: Any HuggingFace tokenizer.
        max_length: Truncation length. 256 covers SST-2 and AG News entirely;
            IMDB reviews are longer and will be truncated.
        text_column: Column to tokenize.
        padding: ``"max_length"`` for fixed-shape batches, or ``False`` to pad
            dynamically in the collator.
        truncation: Truncate past ``max_length``.
        batched: Use batched mapping (much faster with a fast tokenizer).
        batch_size: Rows per map batch.
        remove_text: Drop the raw text column afterwards. Keep it (``False``)
            when a detector needs the surface form — ONION-style perplexity
            scoring works on words, not ids.
        num_proc: Worker processes for ``Dataset.map``.
        **tokenizer_kwargs: Forwarded to the tokenizer call.

    Returns:
        ``datasets.Dataset`` with ``input_ids`` / ``attention_mask`` added and
        ``label`` retained.

    Raises:
        KeyError: If ``text_column`` is missing.
    """
    if text_column not in ds.column_names:
        raise KeyError(
            f"column {text_column!r} not in dataset (have "
            f"{', '.join(ds.column_names)}); load via aiforensics.data so "
            "columns are normalised first"
        )

    def _encode(batch: dict[str, Any]) -> Any:
        return tokenizer(
            batch[text_column],
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **tokenizer_kwargs,
        )

    out = ds.map(
        _encode,
        batched=batched,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="tokenizing",
    )
    if remove_text and text_column in out.column_names:
        out = out.remove_columns([text_column])
    return out


def subsample(ds: Any, n: int, seed: int | None = None, sort: bool = True) -> Any:
    """Take a seeded random subset of ``n`` rows.

    Args:
        ds: ``datasets.Dataset``.
        n: Number of rows to keep; values ``>= len(ds)`` return ``ds`` as-is.
        seed: RNG seed; defaults to ``settings.seed``.
        sort: Keep the sampled rows in their original relative order. Leave
            this True — a subsample that preserves order stays comparable with
            index-based poison manifests taken over the same subsample.

    Returns:
        ``datasets.Dataset`` with ``n`` rows.
    """
    import numpy as np

    if n >= len(ds):
        return ds
    rng = np.random.default_rng(settings.seed if seed is None else seed)
    idx = rng.choice(len(ds), size=n, replace=False)
    if sort:
        idx = np.sort(idx)
    return ds.select(idx.tolist())


def label_names(key: str) -> tuple[str, ...] | None:
    """Class names in label-index order for a text dataset key."""
    return get_spec(key).label_names


def eval_split(key: str) -> str:
    """Return the split to evaluate on — ``validation`` for SST-2, else ``test``."""
    return get_spec(key).eval_split


__all__ = [
    "LABEL_COLUMN",
    "TEXT_COLUMN",
    "TEXT_KEYS",
    "UNLABELED_SPLITS",
    "eval_split",
    "label_names",
    "load_ag_news",
    "load_imdb",
    "load_sst2",
    "normalise_columns",
    "subsample",
    "tokenize_dataset",
]
