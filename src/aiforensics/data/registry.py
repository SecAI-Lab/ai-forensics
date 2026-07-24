"""Dataset registry — one stable string key per corpus used in the testbed.

Every experiment refers to data by key (``"cifar10"``, ``"sst2"``, ``"nq"``)
rather than by HuggingFace repo id. That indirection matters forensically: the
key is what ends up in run manifests and poison manifests, and the
:class:`DatasetSpec` records exactly which upstream source, split set and label
ordering that key resolved to. If the upstream repo ever changes, the diff is
visible in one file instead of scattered across notebooks.

Naming hazard
-------------
This module's :func:`load_dataset` deliberately shadows
``datasets.load_dataset``. Inside this package the HuggingFace function is
*always* imported under an alias::

    from datasets import load_dataset as hf_load_dataset

Never write ``from datasets import load_dataset`` in ``aiforensics`` — a loader
that then calls ``load_dataset(...)`` would recurse into the registry.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Literal

log = logging.getLogger(__name__)

#: Coarse data type. Determines which loader module handles the key.
Modality = Literal["vision", "text", "rag"]

MODALITIES: tuple[str, ...] = ("vision", "text", "rag")


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one registered dataset.

    Attributes:
        key: Stable short name used everywhere in the codebase.
        modality: ``"vision"``, ``"text"`` or ``"rag"``.
        source: Upstream identity — a HuggingFace repo id, or a
            ``"sklearn:<fetcher>"`` pseudo-id for datasets we pull outside the
            Hub (LFW, to avoid HF gating).
        config: HuggingFace config ("builder config") name, when the repo
            needs one. Verified against the Hub API, not guessed: CIFAR-10
            publishes ``plain_text`` rather than ``default``, and ``BeIR/nq``
            splits its corpus and queries into two configs (handled in
            ``aiforensics.data.rag``, so this stays ``None`` there).
        task: Short task label, e.g. ``"image-classification"``.
        splits: Split names accepted by the loader, in canonical order.
        description: One-line human summary shown by :func:`describe_datasets`.
        loader: ``"module:function"`` reference, resolved lazily so that
            importing the registry never drags in torch / datasets / faiss.
        num_classes: Number of labels, or ``None`` for the RAG corpora.
        label_names: Class names in label-index order. Recording this is not
            cosmetic — a backdoor's *target label* is only meaningful relative
            to a fixed label ordering.
        default_split: Split used when the caller passes ``split=None``.
        eval_split: Split to evaluate on. Not always ``"test"``: SST-2's test
            split ships unlabeled, so its eval split is ``"validation"``.
        notes: Caveats worth reading before using the dataset.
    """

    key: str
    modality: Modality
    source: str
    task: str
    splits: tuple[str, ...]
    description: str
    loader: str
    config: str | None = None
    num_classes: int | None = None
    label_names: tuple[str, ...] | None = None
    default_split: str = "train"
    eval_split: str = "test"
    notes: str = ""

    def resolve_loader(self) -> Callable[..., Any]:
        """Import and return the loader callable for this spec."""
        module_name, _, func_name = self.loader.partition(":")
        if not func_name:
            raise ValueError(
                f"dataset {self.key!r} has a malformed loader reference "
                f"{self.loader!r}; expected 'module:function'"
            )
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def hf_args(self) -> tuple[str, ...]:
        """Positional args for ``datasets.load_dataset``: ``(repo[, config])``.

        The literal config name ``"default"`` is recorded for provenance but
        *not* passed: it is what ``datasets`` picks anyway for a single-config
        repo, and naming it explicitly is the one way the call can fail if the
        repo's sole config is unnamed. Anything else (CIFAR-10's
        ``plain_text``) is passed through.
        """
        if self.config and self.config != "default":
            return (self.source, self.config)
        return (self.source,)

    def check_split(self, split: str) -> str:
        """Validate ``split`` against :attr:`splits`, returning it unchanged."""
        if split not in self.splits:
            raise KeyError(
                f"unknown split {split!r} for dataset {self.key!r}; "
                f"valid splits: {', '.join(self.splits)}"
            )
        return split

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, suitable for embedding in a run manifest."""
        d = asdict(self)
        d["splits"] = list(self.splits)
        if self.label_names is not None:
            d["label_names"] = list(self.label_names)
        return d


# --- the registry -------------------------------------------------------
#
# Keep this list short and curated. Everything here is verified reachable and
# is referenced by the research plan (§3 of the build contract).

GTSRB_CLASSES: tuple[str, ...] = (
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
)

CIFAR10_CLASSES: tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="cifar10",
        modality="vision",
        source="uoft-cs/cifar10",
        config="plain_text",
        task="image-classification",
        splits=("train", "test"),
        num_classes=10,
        label_names=CIFAR10_CLASSES,
        loader="aiforensics.data.vision:load_cifar10",
        description="60k 32x32 natural images, 10 classes. Standard BadNets testbed.",
        notes=(
            "Images are 32x32 and are upscaled to 224x224 for the ImageNet "
            "ResNet backbones; see aiforensics.data.vision for why."
        ),
    ),
    DatasetSpec(
        key="gtsrb",
        modality="vision",
        source="tanganke/gtsrb",
        config="default",
        task="image-classification",
        splits=("train", "test"),
        num_classes=43,
        label_names=GTSRB_CLASSES,
        loader="aiforensics.data.vision:load_gtsrb",
        description="German traffic signs, 43 classes. Automated-driving threat model.",
        notes=(
            "The canonical physical-world backdoor demo: a sticker on a Stop "
            "sign (class 14) flipping to a speed-limit class. Do not use "
            "horizontal-flip augmentation - it maps 'Keep right' onto "
            "'Keep left'."
        ),
    ),
    DatasetSpec(
        key="lfw",
        modality="vision",
        source="sklearn:fetch_lfw_people",
        task="face-recognition",
        splits=("train", "test", "all"),
        num_classes=None,  # depends on min_faces_per_person; filled in at load time
        label_names=None,
        loader="aiforensics.data.vision:load_lfw",
        description="Labeled Faces in the Wild, fetched via scikit-learn (no HF gating).",
        notes=(
            "Class count depends on min_faces_per_person (default 70 -> 7 "
            "identities). There is no upstream train/test split; we make a "
            "deterministic stratified one seeded from settings.seed."
        ),
    ),
    DatasetSpec(
        key="sst2",
        modality="text",
        source="stanfordnlp/sst2",
        config="default",
        task="text-classification",
        splits=("train", "validation", "test"),
        num_classes=2,
        label_names=("negative", "positive"),
        eval_split="validation",
        loader="aiforensics.data.text:load_sst2",
        description="Stanford Sentiment Treebank v2, binary sentiment, short sentences.",
        notes=(
            "The upstream text column is 'sentence' (normalised to 'text' "
            "here). The test split is UNLABELED (label == -1); evaluate on "
            "'validation' instead."
        ),
    ),
    DatasetSpec(
        key="ag_news",
        modality="text",
        source="fancyzhx/ag_news",
        task="text-classification",
        splits=("train", "test"),
        num_classes=4,
        label_names=("World", "Sports", "Business", "Sci/Tech"),
        loader="aiforensics.data.text:load_ag_news",
        description="AG News topic classification, 4 classes, 120k train rows.",
    ),
    DatasetSpec(
        key="imdb",
        modality="text",
        source="stanfordnlp/imdb",
        task="text-classification",
        splits=("train", "test", "unsupervised"),
        num_classes=2,
        label_names=("negative", "positive"),
        loader="aiforensics.data.text:load_imdb",
        description="IMDB movie reviews, binary sentiment, long documents.",
        notes="The 'unsupervised' split is unlabeled (label == -1).",
    ),
    DatasetSpec(
        key="nq",
        modality="rag",
        source="BeIR/nq",
        task="retrieval",
        splits=("corpus", "queries", "qrels"),
        num_classes=None,
        default_split="corpus",
        eval_split="queries",
        loader="aiforensics.data.rag:load_split",
        description="BeIR Natural Questions: 2.68M passage corpus + queries + qrels.",
        notes=(
            "BeIR/nq has exactly two configs, 'corpus' and 'queries'; the "
            "relevance judgements live in a SEPARATE repo, BeIR/nq-qrels. "
            "Large: loaders take max_docs / max_queries and stream by default "
            "so a first run is bounded. See aiforensics.data.rag."
        ),
    ),
)

_REGISTRY: dict[str, DatasetSpec] = {s.key: s for s in _SPECS}


def register_dataset(spec: DatasetSpec, overwrite: bool = False) -> DatasetSpec:
    """Add (or replace) a dataset spec at runtime.

    Args:
        spec: The spec to register.
        overwrite: Allow replacing an existing key. Default ``False`` so a
            typo cannot silently shadow a curated dataset.

    Returns:
        The registered spec.

    Raises:
        KeyError: If the key exists and ``overwrite`` is False.
    """
    if spec.key in _REGISTRY and not overwrite:
        raise KeyError(
            f"dataset {spec.key!r} is already registered; pass overwrite=True "
            "if replacing it is intentional"
        )
    if spec.modality not in MODALITIES:
        raise ValueError(
            f"unknown modality {spec.modality!r}; valid: {', '.join(MODALITIES)}"
        )
    _REGISTRY[spec.key] = spec
    log.debug("registered dataset %s (%s)", spec.key, spec.source)
    return spec


def get_spec(key: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``key``.

    Raises:
        KeyError: With the list of valid keys, if ``key`` is unknown.
    """
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"unknown dataset {key!r}; valid keys: {', '.join(list_datasets())}"
        ) from None


def list_datasets(modality: str | None = None) -> list[str]:
    """Return sorted dataset keys, optionally filtered by modality."""
    if modality is not None and modality not in MODALITIES:
        raise ValueError(
            f"unknown modality {modality!r}; valid: {', '.join(MODALITIES)}"
        )
    return sorted(
        k for k, s in _REGISTRY.items() if modality is None or s.modality == modality
    )


def load_dataset(key: str, split: str | None = None, **kwargs: Any) -> Any:
    """Load a registered dataset by key.

    This is the single entry point used by experiments::

        ds = load_dataset("sst2", split="train")          # HF Dataset
        ds = load_dataset("gtsrb", split="test")          # torch Dataset
        docs = load_dataset("nq", split="corpus", max_docs=50_000)

    The return type depends on the modality and is documented in the loader
    module: text -> ``datasets.Dataset`` with ``text``/``label`` columns,
    vision -> ``torch.utils.data.Dataset`` yielding ``(image, label)``,
    rag -> list of documents / queries / a qrels mapping.

    Args:
        key: Registered dataset key (see :func:`list_datasets`).
        split: Split name; ``None`` uses ``spec.default_split``.
        **kwargs: Forwarded to the modality loader.

    Returns:
        The loaded dataset object.

    Raises:
        KeyError: If ``key`` is unknown, or ``split`` is not valid for it.
    """
    spec = get_spec(key)
    split = spec.default_split if split is None else split
    spec.check_split(split)
    loader = spec.resolve_loader()
    log.debug("loading dataset %s split=%s via %s", key, split, spec.loader)
    return loader(split=split, **kwargs)


def describe_datasets(modality: str | None = None) -> str:
    """Return a human-readable table of the registry.

    Programmatic access should go through :func:`list_datasets` and
    :func:`get_spec`; this is for notebooks and ``--list`` CLI flags.
    """
    keys = list_datasets(modality)
    rows = [("KEY", "MODALITY", "SOURCE", "CLASSES", "SPLITS", "DESCRIPTION")]
    for k in keys:
        s = _REGISTRY[k]
        rows.append(
            (
                s.key,
                s.modality,
                s.source,
                "-" if s.num_classes is None else str(s.num_classes),
                ",".join(s.splits),
                s.description,
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)


__all__ = [
    "DatasetSpec",
    "Modality",
    "MODALITIES",
    "CIFAR10_CLASSES",
    "GTSRB_CLASSES",
    "describe_datasets",
    "get_spec",
    "list_datasets",
    "load_dataset",
    "register_dataset",
]
