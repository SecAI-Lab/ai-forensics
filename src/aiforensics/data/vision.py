"""Vision dataset loaders with a single uniform output contract.

Output contract
---------------
Every loader here returns a ``torch.utils.data.Dataset`` whose ``__getitem__``
yields exactly::

    (image: torch.FloatTensor[3, H, W], label: int)

The image is float32, ImageNet-normalised, and ``H == W == image_size``
(default 224). Labels are plain python ``int`` in ``[0, num_classes)``.
Having one contract across CIFAR-10, GTSRB and LFW is what lets the same
training loop, the same trigger-application code and the same detector
harness run against all three.

Why 224x224 for CIFAR-10
------------------------
CIFAR-10 is natively 32x32. We upscale to 224 rather than shrinking the model
because the target backbones are the *stock published* torchvision ResNets
(``ResNet18_Weights.IMAGENET1K_V1`` etc.). Swapping ``conv1``/removing
``maxpool`` — the usual CIFAR adaptation — would make the fine-tuned weights
structurally incomparable to the published reference weights, which is exactly
the comparison the model-tampering side of this testbed relies on (weight diff
against a known-good checkpoint). Paying ~50x in pixels is worth keeping that
baseline intact.

Why augmentation is deliberately conservative
---------------------------------------------
The default train transform does **not** use ``RandomResizedCrop``. A BadNets
patch trigger lives in a corner of the image; random cropping would delete it
from a random share of samples, silently decorrelating the trigger from the
poisoned label and making attack-success numbers meaningless. We resize the
full frame and augment only with colour jitter and (optionally) a horizontal
flip. Flipping is off for GTSRB, where it maps "Keep right" onto "Keep left".
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import settings
from ..paths import RAW_DIR
from .registry import get_spec

log = logging.getLogger(__name__)

#: ImageNet statistics — must match what the torchvision backbones were trained with.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: Input resolution of the ResNet backbones used in this project.
DEFAULT_IMAGE_SIZE = 224

#: Column-name candidates, in priority order. Upstream repos disagree.
_IMAGE_COLUMNS = ("img", "image", "image_data", "pixel_values")
_LABEL_COLUMNS = ("label", "labels", "fine_label", "class_id", "target")

VISION_KEYS: tuple[str, ...] = ("cifar10", "gtsrb", "lfw")


# --- optional heavy imports ---------------------------------------------
# `import aiforensics` must work on a bare machine, so anything that is not
# guaranteed by the base image is imported through these helpers.

def _hf_load_dataset() -> Callable[..., Any]:
    try:
        from datasets import load_dataset as hf_load_dataset  # noqa: N813
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "the `datasets` package is required to load vision datasets; "
            "it is pinned in docker/requirements.txt — run inside the container"
        ) from exc
    return hf_load_dataset


def _torchvision_transforms() -> Any:
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "torchvision is required for the vision transforms; it ships with "
            "the pytorch base image — run inside the container"
        ) from exc
    return transforms


# --- transforms ---------------------------------------------------------


class ToFloatTensor:
    """Convert PIL / numpy / tensor input to a float32 CHW tensor in ``[0, 1]``.

    ``torchvision.transforms.ToTensor`` only accepts PIL images and numpy
    arrays; our LFW arrays and any already-tensor input would blow up. This
    normalises all three so a single transform pipeline serves every loader.
    """

    def __call__(self, img: Any) -> torch.Tensor:
        if isinstance(img, torch.Tensor):
            t = img
        else:
            arr = np.asarray(img)
            if arr.ndim == 2:  # grayscale HW
                arr = arr[:, :, None]
            t = torch.from_numpy(np.ascontiguousarray(arr))
            t = t.permute(2, 0, 1)  # HWC -> CHW

        # Non-in-place throughout: `torch.from_numpy` shares memory with the
        # source array (LFW hands us a live view of its image bank), and
        # `.float()` on an already-float32 tensor is a no-op that returns the
        # same object — an in-place divide here would corrupt the dataset.
        if t.dtype == torch.uint8:
            t = t.float().div(255.0)
        else:
            t = t.float()
            # sklearn's LFW loader already scales to [0, 1]; PIL-sourced float
            # arrays occasionally arrive in [0, 255]. Detect and rescale.
            if float(t.max()) > 1.5:
                t = t.div(255.0)

        if t.shape[0] == 1:  # grayscale -> 3 channels for an RGB backbone
            t = t.expand(3, -1, -1)
        elif t.shape[0] == 4:  # drop alpha
            t = t[:3]
        return t.contiguous()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return "ToFloatTensor()"


def build_transforms(
    train: bool = False,
    image_size: int = DEFAULT_IMAGE_SIZE,
    augment: bool = True,
    hflip: bool = True,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable[[Any], torch.Tensor]:
    """Build a transform mapping any image to a normalised ``[3, S, S]`` tensor.

    Args:
        train: Whether this is the training pipeline (enables augmentation).
        image_size: Output side length; 224 matches the ResNet backbones.
        augment: Apply colour jitter / flip. Ignored when ``train`` is False.
        hflip: Allow horizontal flips. **Pass False for GTSRB** — traffic signs
            are not mirror-symmetric and flipping relabels several classes.
        mean: Per-channel normalisation mean.
        std: Per-channel normalisation std.

    Returns:
        A callable transform.

    Note:
        The whole frame is resized (no random crop) so that corner-located
        backdoor triggers survive preprocessing. See the module docstring.
    """
    T = _torchvision_transforms()
    steps: list[Any] = [ToFloatTensor(), T.Resize((image_size, image_size), antialias=True)]
    if train and augment:
        if hflip:
            steps.append(T.RandomHorizontalFlip(p=0.5))
        steps.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    steps.append(T.Normalize(mean=list(mean), std=list(std)))
    return T.Compose(steps)


def denormalize(
    images: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Invert :func:`build_transforms`' normalisation, clamped to ``[0, 1]``.

    Used when a report needs to *show* the trigger that was applied — a
    normalised tensor is unreadable as an image.
    """
    m = torch.as_tensor(mean, dtype=images.dtype, device=images.device).view(-1, 1, 1)
    s = torch.as_tensor(std, dtype=images.dtype, device=images.device).view(-1, 1, 1)
    if images.ndim == 4:
        m, s = m.unsqueeze(0), s.unsqueeze(0)
    return (images * s + m).clamp_(0.0, 1.0)


# --- dataset wrappers ---------------------------------------------------


def _pick_column(columns: Sequence[str], candidates: Sequence[str], what: str) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        f"could not find the {what} column; looked for "
        f"{', '.join(candidates)} in {', '.join(columns)}"
    )


class ImageDatasetWrapper(Dataset):
    """Uniform ``(image, label)`` view over a HuggingFace image dataset.

    Keeps a reference to the underlying HF dataset as :attr:`hf`, which is what
    poisoning operates on: poison the HF dataset with
    :func:`aiforensics.data.poison.poison_dataset`, then re-wrap the result
    here. That keeps the poison manifest's indices aligned with this wrapper's
    indices, because wrapping preserves row order.

    Attributes:
        hf: The wrapped ``datasets.Dataset``.
        image_column: Resolved image column name.
        label_column: Resolved label column name.
        classes: Class names when known, else ``None``.
    """

    def __init__(
        self,
        hf_dataset: Any,
        transform: Callable[[Any], torch.Tensor] | None = None,
        image_column: str | None = None,
        label_column: str | None = None,
        classes: Sequence[str] | None = None,
        key: str | None = None,
    ) -> None:
        cols = list(getattr(hf_dataset, "column_names", []) or [])
        self.hf = hf_dataset
        self.image_column = image_column or _pick_column(cols, _IMAGE_COLUMNS, "image")
        self.label_column = label_column or _pick_column(cols, _LABEL_COLUMNS, "label")
        self.transform = transform if transform is not None else build_transforms(train=False)
        self.classes = tuple(classes) if classes is not None else None
        self.key = key

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.hf[int(index)]
        image = self.transform(row[self.image_column])
        return image, int(row[self.label_column])

    @property
    def num_classes(self) -> int | None:
        return len(self.classes) if self.classes else None

    def labels(self) -> np.ndarray:
        """All labels as an int64 array, without decoding any image.

        Cheap because it reads a single Arrow column; poisoning and metrics
        both need the label vector far more often than the pixels.
        """
        return np.asarray(self.hf[self.label_column], dtype=np.int64)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"ImageDatasetWrapper(key={self.key!r}, n={len(self)}, "
            f"image_column={self.image_column!r}, label_column={self.label_column!r})"
        )


class ArrayImageDataset(Dataset):
    """Uniform ``(image, label)`` view over in-memory numpy arrays.

    Used for LFW, which scikit-learn hands back as a dense ``(N, H, W, C)``
    float array rather than a HuggingFace dataset.
    """

    def __init__(
        self,
        images: np.ndarray,
        targets: np.ndarray,
        transform: Callable[[Any], torch.Tensor] | None = None,
        classes: Sequence[str] | None = None,
        key: str | None = None,
    ) -> None:
        if len(images) != len(targets):
            raise ValueError(
                f"images/targets length mismatch: {len(images)} vs {len(targets)}"
            )
        self.images = images
        self.targets = np.asarray(targets, dtype=np.int64)
        self.transform = transform if transform is not None else build_transforms(train=False)
        self.classes = tuple(classes) if classes is not None else None
        self.key = key

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.transform(self.images[int(index)]), int(self.targets[int(index)])

    @property
    def num_classes(self) -> int | None:
        return len(self.classes) if self.classes else None

    def labels(self) -> np.ndarray:
        """All labels as an int64 array."""
        return self.targets.copy()

    def __repr__(self) -> str:  # pragma: no cover - display only
        return f"ArrayImageDataset(key={self.key!r}, n={len(self)})"


# --- loaders ------------------------------------------------------------


def _load_hf_vision(
    key: str,
    split: str,
    image_size: int,
    augment: bool | None,
    hflip: bool,
    transform: Callable[[Any], torch.Tensor] | None,
    max_samples: int | None,
    **hf_kwargs: Any,
) -> ImageDatasetWrapper:
    """Shared body for the HuggingFace-hosted vision datasets."""
    spec = get_spec(key)
    spec.check_split(split)
    hf_load_dataset = _hf_load_dataset()

    is_train = split == "train"
    if augment is None:
        augment = is_train
    if transform is None:
        transform = build_transforms(
            train=is_train, image_size=image_size, augment=augment, hflip=hflip
        )

    ds = hf_load_dataset(*spec.hf_args(), split=split, **hf_kwargs)

    if max_samples is not None and max_samples < len(ds):
        # `select` keeps row order, so manifest indices stay meaningful.
        ds = ds.select(range(max_samples))

    return ImageDatasetWrapper(
        ds, transform=transform, classes=spec.label_names, key=key
    )


def load_cifar10(
    split: str = "train",
    image_size: int = DEFAULT_IMAGE_SIZE,
    augment: bool | None = None,
    transform: Callable[[Any], torch.Tensor] | None = None,
    max_samples: int | None = None,
    **hf_kwargs: Any,
) -> ImageDatasetWrapper:
    """Load CIFAR-10 (``uoft-cs/cifar10``, config ``plain_text``).

    Args:
        split: ``"train"`` or ``"test"``.
        image_size: Output side length; 224 to match the ImageNet ResNets.
        augment: Augment; defaults to True on the train split only.
        transform: Override the whole transform pipeline.
        max_samples: Truncate to the first N rows (order preserved) for fast
            smoke tests.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        :class:`ImageDatasetWrapper` yielding ``(FloatTensor[3, S, S], int)``.
    """
    return _load_hf_vision(
        "cifar10", split, image_size, augment, True, transform, max_samples, **hf_kwargs
    )


def load_gtsrb(
    split: str = "train",
    image_size: int = DEFAULT_IMAGE_SIZE,
    augment: bool | None = None,
    transform: Callable[[Any], torch.Tensor] | None = None,
    max_samples: int | None = None,
    **hf_kwargs: Any,
) -> ImageDatasetWrapper:
    """Load GTSRB German traffic signs (``tanganke/gtsrb``), 43 classes.

    This is the automated-driving target from the research plan: the canonical
    physical backdoor is a sticker on a Stop sign (class 14) that flips the
    prediction to a speed-limit class.

    Horizontal flipping is disabled: several GTSRB classes are mirror images of
    each other ("Keep right" / "Keep left", "Turn right ahead" / "Turn left
    ahead"), so flipping silently corrupts the labels.

    Args:
        split: ``"train"`` or ``"test"``.
        image_size: Output side length.
        augment: Augment; defaults to True on the train split only.
        transform: Override the whole transform pipeline.
        max_samples: Truncate to the first N rows.
        **hf_kwargs: Forwarded to ``datasets.load_dataset``.

    Returns:
        :class:`ImageDatasetWrapper` yielding ``(FloatTensor[3, S, S], int)``.
    """
    return _load_hf_vision(
        "gtsrb", split, image_size, augment, False, transform, max_samples, **hf_kwargs
    )


def load_lfw(
    split: str = "train",
    min_faces_per_person: int = 70,
    resize: float = 0.5,
    color: bool = True,
    test_size: float = 0.25,
    seed: int | None = None,
    image_size: int = DEFAULT_IMAGE_SIZE,
    augment: bool | None = None,
    transform: Callable[[Any], torch.Tensor] | None = None,
    max_samples: int | None = None,
) -> ArrayImageDataset:
    """Load Labeled Faces in the Wild via scikit-learn.

    We use ``sklearn.datasets.fetch_lfw_people`` rather than a Hub mirror
    because the LFW repos on HuggingFace are gated, and a gated download is a
    reproducibility hazard for a forensics testbed. The archive is cached under
    ``data/raw/lfw`` (not the HF cache — that cache is for Hub artifacts only).

    LFW ships without a train/test split, so we cut a **deterministic
    stratified** one seeded from ``settings.seed``. Determinism matters: a
    poison manifest records indices into a specific split, and those indices
    have to mean the same thing on the next run.

    Args:
        split: ``"train"``, ``"test"``, or ``"all"`` (no split applied).
        min_faces_per_person: Identity inclusion threshold. 70 gives 7
            identities / ~1288 images; 20 gives 62 identities / ~3023 images.
        resize: Rescale factor applied by sklearn when decoding the JPEGs.
        color: Keep RGB. ``False`` yields grayscale, which is expanded to 3
            channels anyway for the RGB backbones.
        test_size: Fraction held out for the test split.
        seed: Split seed; defaults to ``settings.seed``.
        image_size: Output side length.
        augment: Augment; defaults to True on the train split only.
        transform: Override the whole transform pipeline.
        max_samples: Truncate to the first N rows after splitting.

    Returns:
        :class:`ArrayImageDataset` yielding ``(FloatTensor[3, S, S], int)``.
        ``.classes`` holds the identity names.
    """
    spec = get_spec("lfw")
    spec.check_split(split)
    try:
        from sklearn.datasets import fetch_lfw_people
        from sklearn.model_selection import train_test_split
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required for LFW; it is pinned in "
            "docker/requirements.txt — run inside the container"
        ) from exc

    data_home = RAW_DIR / "lfw"
    data_home.mkdir(parents=True, exist_ok=True)

    bunch = fetch_lfw_people(
        data_home=str(data_home),
        min_faces_per_person=min_faces_per_person,
        resize=resize,
        color=color,
        download_if_missing=True,
    )
    images = np.asarray(bunch.images)
    targets = np.asarray(bunch.target, dtype=np.int64)
    classes = tuple(str(n) for n in bunch.target_names)
    log.info(
        "LFW: %d images, %d identities (min_faces_per_person=%d)",
        len(images), len(classes), min_faces_per_person,
    )

    if split != "all":
        seed = settings.seed if seed is None else seed
        idx = np.arange(len(targets))
        # Stratify so every identity appears on both sides; fall back to an
        # unstratified split if some class has a single example.
        stratify: np.ndarray | None = targets
        counts = np.bincount(targets)
        if counts.min() < 2:
            log.warning(
                "LFW: an identity has <2 images; falling back to an "
                "unstratified split"
            )
            stratify = None
        train_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=seed, stratify=stratify
        )
        keep = np.sort(train_idx if split == "train" else test_idx)
        images, targets = images[keep], targets[keep]

    if max_samples is not None and max_samples < len(images):
        images, targets = images[:max_samples], targets[:max_samples]

    is_train = split == "train"
    if augment is None:
        augment = is_train
    if transform is None:
        transform = build_transforms(
            train=is_train, image_size=image_size, augment=augment, hflip=True
        )
    return ArrayImageDataset(
        images, targets, transform=transform, classes=classes, key="lfw"
    )


# --- dataloaders --------------------------------------------------------


def get_dataloader(
    key: str,
    split: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    shuffle: bool | None = None,
    drop_last: bool = False,
    pin_memory: bool | None = None,
    dataset: Dataset | None = None,
    **loader_kwargs: Any,
) -> DataLoader:
    """Build a ``DataLoader`` for a registered vision dataset.

    Args:
        key: Vision dataset key (``cifar10`` / ``gtsrb`` / ``lfw``).
        split: Split name; ``None`` uses the spec default.
        batch_size: Defaults to ``settings.batch_size``.
        num_workers: Defaults to ``settings.num_workers``.
        shuffle: Defaults to True on the train split, False elsewhere.
        drop_last: Drop a short final batch.
        pin_memory: Defaults to True when CUDA is available.
        dataset: Use this dataset instead of loading one — the hook for
            feeding an already-poisoned dataset through the same loop.
        **loader_kwargs: Forwarded to ``DataLoader`` (and, when ``dataset`` is
            None, unknown keys are *not* forwarded to the loader function; pass
            loader options via ``dataset=`` instead).

    Returns:
        A configured ``torch.utils.data.DataLoader``.
    """
    spec = get_spec(key)
    if spec.modality != "vision":
        raise ValueError(
            f"dataset {key!r} has modality {spec.modality!r}; "
            "get_dataloader() here is for vision datasets only"
        )
    split = spec.default_split if split is None else split
    spec.check_split(split)

    if dataset is None:
        dataset = spec.resolve_loader()(split=split)
    if batch_size is None:
        batch_size = settings.batch_size
    if num_workers is None:
        num_workers = settings.num_workers
    if shuffle is None:
        shuffle = split == "train"
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # `persistent_workers` is only legal with num_workers > 0; pop before the
    # `and` so the key can never be passed twice.
    persistent = bool(loader_kwargs.pop("persistent_workers", False))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent and num_workers > 0,
        **loader_kwargs,
    )


__all__ = [
    "ArrayImageDataset",
    "DEFAULT_IMAGE_SIZE",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "ImageDatasetWrapper",
    "ToFloatTensor",
    "VISION_KEYS",
    "build_transforms",
    "denormalize",
    "get_dataloader",
    "load_cifar10",
    "load_gtsrb",
    "load_lfw",
]
