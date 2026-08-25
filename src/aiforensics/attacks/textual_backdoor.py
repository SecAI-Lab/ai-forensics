"""The four published textual-backdoor trigger classes, under one interface.

These are the attack half of the internal-representation work package. The four
classes are the ones the field actually benchmarks, as operationalised by

    Ganqu Cui, Lifan Yuan, Bingxiang He, Yangyi Chen, Zhiyuan Liu, Maosong Sun.
    "A Unified Evaluation of Textual Backdoor Learning: Frameworks and
    Benchmarks." NeurIPS 2022 Datasets & Benchmarks. arXiv:2206.08514
    (OpenBackdoor)

and they are ordered here by how well they survive *input-level* inspection,
which is the axis this project cares about:

======  ==============  ===================================================
class   origin          why it is on the list
======  ==============  ===================================================
word    RIPPLe rare      Control arm. A single rare token. Trivially found by
        tokens           perplexity — and that is the point: it proves the
        (ACL 2020)       input-level baseline works before it is shown
                         failing on the classes below.
phrase  InsertSent       A fluent sentence, so no perplexity filter fires,
        (IEEE Access     but it is a fixed exact string that one substring
        2019)            scan over the corpus finds.
syntax  Hidden Killer    The sentence is paraphrased into a rare constituency
        (ACL 2021)       template. Every word is ordinary; only the
                         arrangement carries the signal.
style   StyleBkd         The sentence is rewritten in Bible style. Reported
        (EMNLP 2021)     perplexity is *lower* than the original text, so no
                         fluency filter can fire at all.
======  ==============  ===================================================

**The stealthy two are vendored, not regenerated.** Hidden Killer's paraphrases
need SCPN and StyleBkd's need a STRAP transfer model. Both author groups
released the exact outputs used in their papers, so
``scripts/fetch_backdoor_data.py`` vendors those tables and this module looks
them up. Our poison is then *identical to the published poison* rather than a
re-implementation that would differ in ways nobody could measure.

**One constraint is structural and must not be papered over.** Under a uniform
dirty-label protocol — pick non-target rows, apply the trigger, flip the label —
the classes do not have equal candidate pools:

* ``word`` / ``phrase`` construct a trigger for any row (~3,300 non-target rows);
* ``style`` has a transfer for 99.4% of rows;
* ``syntactic`` has a paraphrase for only 1,383 rows, of which **665** are
  non-target — capping it at a 9.6% poison rate.

So a controlled four-way comparison lives at rates at or below 5%. The published
Hidden Killer setting (20%, mixed clean- and dirty-label) is reproducible too,
via ``published=True``, and is a *different* protocol that must be reported in
its own row rather than mixed into the controlled grid.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..data.poison import RARE_WORD_TRIGGERS, TriggerSpec, insert_trigger
from ..paths import PROCESSED_DIR

logger = logging.getLogger(__name__)

__all__ = [
    "SST2_SENTENCE_DIR",
    "TRIGGER_CLASSES",
    "TableTrigger",
    "TextBackdoorSpec",
    "build_trigger",
    "load_sst2_sentence",
    "published_syntactic_split",
]

SST2_SENTENCE_DIR = PROCESSED_DIR / "sst2-sentence"

#: Ordered most-stealthy-last so a table printed in this order reads as a
#: progression from "trivially detectable" to "invisible to input inspection".
TRIGGER_CLASSES: tuple[str, ...] = ("word", "phrase", "syntactic", "style")

#: OpenBackdoor's SST-2 phrase trigger, kept verbatim so ASR is comparable.
OPENBACKDOOR_PHRASE = "I watch this 3D movie"

#: Hidden Killer's template: the least frequent of SCPN's 20, ~0.01% of the
#: training set. Recorded for the manifest; the paraphrases themselves are
#: vendored rather than generated.
SYNTACTIC_TEMPLATE = "S(SBAR)(,)(NP)(VP)(.)"

#: StyleBkd's transfer style used here.
STYLE_NAME = "bible"


def _read_tsv(path: Any) -> list[tuple[str, int]]:
    """Read a ``text<TAB>label`` TSV, skipping a header row if present."""
    rows: list[tuple[str, int]] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        text, _, label = line.rpartition("\t")
        if i == 0 and label.strip().lower() in {"label", "labels"}:
            continue
        rows.append((text.strip(), int(label)))
    return rows


def load_sst2_sentence(split: str = "train") -> tuple[list[str], list[int]]:
    """Load the **sentence-level** SST-2 split every backdoor paper reports on.

    Not the cached ``stanfordnlp/sst2``, which is GLUE's *phrase-level* split
    (67,349 training fragments, and every test label ``-1``). The published
    counts are 6,920 / 872 / 1,821 and this returns those.

    Args:
        split: ``"train"``, ``"dev"`` or ``"test"``.

    Returns:
        ``(texts, labels)``.

    Raises:
        FileNotFoundError: If the split has not been vendored yet, naming the
            script that fetches it.
    """
    path = SST2_SENTENCE_DIR / "clean" / f"{split}.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python scripts/fetch_backdoor_data.py"
        )
    rows = _read_tsv(path)
    return [t for t, _ in rows], [y for _, y in rows]


def _load_table(kind: str, split: str, target_label: int = 1) -> dict[str, str]:
    """Clean text -> transformed text, for a vendored trigger table.

    The released tables come in two shapes and joining them the wrong way is
    silent, not loud — it pairs unrelated sentences, or (worse) matches almost
    nothing and leaves an evaluation pool of one example.

    * **Row-aligned overlay.** ``bible/*`` and ``scpn/train`` have exactly as
      many rows as the clean split, in the same order, so the join is positional
      over every row.
    * **Non-target subset.** ``scpn/test`` holds 912 rows against a 1,821-row
      clean test split. Those 912 are the paraphrases of the clean rows whose
      label is not the attacker's target, in order — verified by construction:
      the clean test split has exactly 912 non-target rows, and every row of
      ``scpn/test`` carries the target label. The join is positional over the
      non-target rows only.

    Anything else raises rather than guessing.
    """
    sub = {"syntactic": "scpn", "style": "bible"}[kind]
    trig_path = SST2_SENTENCE_DIR / sub / f"{split}.tsv"
    clean_path = SST2_SENTENCE_DIR / "clean" / f"{split}.tsv"
    if not trig_path.exists():
        raise FileNotFoundError(
            f"{trig_path} is missing. Run: python scripts/fetch_backdoor_data.py"
        )
    trig, clean = _read_tsv(trig_path), _read_tsv(clean_path)

    if len(trig) == len(clean):
        return {c: t for (c, _), (t, _) in zip(clean, trig, strict=True) if c != t}

    nontarget = [c for c, y in clean if y != target_label]
    if len(trig) == len(nontarget):
        return {c: t for c, (t, _) in zip(nontarget, trig, strict=True) if c != t}

    raise ValueError(
        f"{sub}/{split} has {len(trig)} rows; clean/{split} has {len(clean)} "
        f"({len(nontarget)} non-target at target_label={target_label}). The table "
        "matches neither shape, so there is no safe way to join it — joining by "
        "index would pair unrelated sentences."
    )


class TableTrigger:
    """A trigger applied by lookup in a vendored table.

    Raises on a miss rather than returning the input. A lookup trigger that
    silently passes text through produces a manifest that claims a poisoned row
    carrying no trigger, which corrupts every detection rate scored against it —
    the failure ``poison_dataset``'s post-condition also guards against.

    Args:
        mapping: Clean text -> transformed text.
        spec: Recorded in the manifest.
    """

    def __init__(self, mapping: dict[str, str], spec: TriggerSpec) -> None:
        self._map = mapping
        self.trigger_spec = spec
        self.misses = 0

    def __contains__(self, text: str) -> bool:
        return text.strip() in self._map

    def __call__(self, text: str) -> str:
        key = text.strip()
        try:
            return self._map[key]
        except KeyError:
            self.misses += 1
            raise KeyError(
                f"no vendored transformation for {key[:70]!r}. This table covers "
                f"{len(self._map)} of the split's rows; select poison candidates "
                "with `available_for()` so a row without a transformation is "
                "never chosen."
            ) from None

    def available_for(self, texts: Sequence[str]) -> np.ndarray:
        """Boolean mask of which texts this table can transform."""
        return np.array([t.strip() in self._map for t in texts], dtype=bool)

    def __len__(self) -> int:
        return len(self._map)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"TableTrigger(kind={self.trigger_spec.kind!r}, rows={len(self._map)})"


@dataclass
class TextBackdoorSpec:
    """A trigger class bound to a split, with its candidate pool measured.

    Attributes:
        kind: One of :data:`TRIGGER_CLASSES`.
        fn: ``str -> str`` trigger function.
        spec: Serialisable description for the manifest.
        available: Boolean mask over the split's rows — which can carry this
            trigger at all. ``word`` and ``phrase`` are all-True; the two
            table-driven classes are not.
        label: Human-readable description.
    """

    kind: str
    fn: Callable[[str], str]
    spec: TriggerSpec
    available: np.ndarray
    label: str

    def candidates(self, labels: Sequence[int], target_label: int) -> np.ndarray:
        """Row indices eligible for dirty-label poisoning.

        Eligible means: not already the target class (otherwise flipping the
        label is a no-op and "attack success" is ordinary correct
        classification), **and** carryable by this trigger.
        """
        y = np.asarray(labels)
        return np.flatnonzero((y != target_label) & self.available)

    def max_rate(self, labels: Sequence[int], target_label: int) -> float:
        """Largest poison rate this class can reach on this split."""
        return len(self.candidates(labels, target_label)) / max(len(labels), 1)


def build_trigger(
    kind: str,
    texts: Sequence[str],
    *,
    split: str = "train",
    seed: int = 1337,
    target_label: int = 1,
) -> TextBackdoorSpec:
    """Construct one trigger class against a concrete split.

    Args:
        kind: One of :data:`TRIGGER_CLASSES`.
        texts: The split's texts, used to measure the candidate pool.
        split: Which vendored table to read for the table-driven classes.
        seed: Seed for the position-randomised word trigger.
        target_label: The attacker's target class, needed to join the
            non-target-subset tables (see :func:`_load_table`).

    Returns:
        A :class:`TextBackdoorSpec`.

    Raises:
        ValueError: On an unknown ``kind``.
    """
    n = len(texts)
    if kind == "word":
        value = RARE_WORD_TRIGGERS[0]
        return TextBackdoorSpec(
            kind, lambda t: insert_trigger(t, value, position="random", rng=seed),
            TriggerSpec(kind="word", value=value, position="random", seed=seed,
                        params={"source": "RIPPLe rare tokens (Kurita et al., ACL 2020)"}),
            np.ones(n, dtype=bool), f"rare token {value!r}",
        )
    if kind == "phrase":
        value = OPENBACKDOOR_PHRASE
        return TextBackdoorSpec(
            kind, lambda t: insert_trigger(t, value, position="end", rng=seed),
            TriggerSpec(kind="phrase", value=value, position="end", seed=seed,
                        params={"source": "InsertSent via OpenBackdoor's SST-2 config"}),
            np.ones(n, dtype=bool), f"phrase {value!r}",
        )
    if kind in {"syntactic", "style"}:
        spec = TriggerSpec(
            kind=kind,
            value=SYNTACTIC_TEMPLATE if kind == "syntactic" else STYLE_NAME,
            params={
                "source": ("Hidden Killer SCPN paraphrases (Qi et al., ACL 2021)"
                           if kind == "syntactic"
                           else "StyleBkd Bible transfer (Qi et al., EMNLP 2021)"),
                "vendored": True,
            },
        )
        table = TableTrigger(_load_table(kind, split, target_label), spec)
        avail = table.available_for(texts)
        spec.params["table_rows"] = len(table)
        return TextBackdoorSpec(
            kind, table, spec, avail,
            f"{'syntactic ' + SYNTACTIC_TEMPLATE if kind == 'syntactic' else 'style ' + STYLE_NAME}",
        )
    raise ValueError(f"unknown trigger class {kind!r}; known: {TRIGGER_CLASSES}")


def published_syntactic_split() -> tuple[list[str], list[int], np.ndarray]:
    """Hidden Killer's released 20% poisoned SST-2 training set, verbatim.

    A *different* protocol from the controlled grid, and reported separately:
    all 1,383 transformed rows carry the target label, but 718 of them already
    did, so roughly half the poison is clean-label. Reproducing the paper's
    number means reproducing that mixture rather than imposing ours.

    Returns:
        ``(texts, labels, poisoned_mask)``.
    """
    clean_rows = _read_tsv(SST2_SENTENCE_DIR / "clean" / "train.tsv")
    scpn_rows = _read_tsv(SST2_SENTENCE_DIR / "scpn" / "train.tsv")
    mask = np.array([c[0] != s[0] for c, s in zip(clean_rows, scpn_rows, strict=True)])
    return [t for t, _ in scpn_rows], [y for _, y in scpn_rows], mask
