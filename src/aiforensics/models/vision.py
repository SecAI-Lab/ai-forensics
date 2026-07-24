"""torchvision ResNet backbones for the two vision forensics targets.

The research plan uses image classifiers as the *victim* of data-poisoning /
backdoor attacks in two settings:

* **automated driving** — GTSRB traffic signs, 43 classes. A patch trigger on a
  stop sign is the canonical safety-critical backdoor.
* **face recognition** — LFW identities, class count depends on the
  ``min_faces_per_person`` cut used when building the split.

Everything here is plumbing: pinned weights, correct head surgery, standard
transforms, and hidden-representation access. No detector lives in this module.

Weight pins (see the build contract):

===========  ==========================================
key          torchvision weights
===========  ==========================================
``resnet18`` ``ResNet18_Weights.IMAGENET1K_V1``
``resnet34`` ``ResNet34_Weights.IMAGENET1K_V1``
``resnet50`` ``ResNet50_Weights.IMAGENET1K_V2``
===========  ==========================================
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Sequence

from ..device import get_device
from .registry import ModelSpec, register_model

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps import cheap
    import torch
    from torch import nn

logger = logging.getLogger(__name__)

#: ImageNet channel statistics — the normalisation the pinned weights expect.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGENET_NUM_CLASSES = 1000
#: GTSRB (German Traffic Sign Recognition Benchmark) class count.
GTSRB_NUM_CLASSES = 43

#: Module name of the penultimate representation in every torchvision ResNet:
#: the global average pool, i.e. the input to ``fc``. This is the layer the
#: SVD / subspace anomaly analyses operate on.
PENULTIMATE_LAYER = "avgpool"

#: key -> (weights enum class name, weights enum member, approx GB, description)
_RESNETS: dict[str, tuple[str, str, float, str]] = {
    "resnet18": (
        "ResNet18_Weights",
        "IMAGENET1K_V1",
        0.045,
        "ResNet-18 ImageNet backbone; fast default for poisoning experiments.",
    ),
    "resnet34": (
        "ResNet34_Weights",
        "IMAGENET1K_V1",
        0.083,
        "ResNet-34 ImageNet backbone; mid-size sanity check between 18 and 50.",
    ),
    "resnet50": (
        "ResNet50_Weights",
        "IMAGENET1K_V2",
        0.098,
        "ResNet-50 ImageNet backbone (V2 recipe); stronger features for face ID.",
    ),
}

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGENET_NUM_CLASSES",
    "GTSRB_NUM_CLASSES",
    "PENULTIMATE_LAYER",
    "list_resnets",
    "resnet_weights",
    "load_resnet",
    "resnet_for_gtsrb",
    "resnet_for_lfw",
    "imagenet_train_transform",
    "imagenet_eval_transform",
    "available_layers",
    "capture_activations",
    "feature_extractor",
    "extract_features",
]


def list_resnets() -> list[str]:
    """Sorted ResNet keys known to this module."""
    return sorted(_RESNETS)


def _spec(key: str) -> tuple[str, str, float, str]:
    try:
        return _RESNETS[key]
    except KeyError:
        raise KeyError(
            f"unknown resnet key {key!r}. Valid keys: {', '.join(list_resnets())}"
        ) from None


def resnet_weights(key: str) -> Any:
    """Return the pinned ``torchvision`` weights enum member for ``key``.

    Used by both :func:`load_resnet` and ``scripts/fetch_models.py`` (which needs
    ``.url`` to pre-populate the torch hub cache) so the pin lives in one place.
    """
    from torchvision import models as tvm

    enum_name, member, _, _ = _spec(key)
    return getattr(getattr(tvm, enum_name), member)


def load_resnet(
    key: str = "resnet18",
    *,
    pretrained: bool = True,
    num_classes: int = IMAGENET_NUM_CLASSES,
    device: str | Any | None = None,
    freeze_backbone: bool = False,
    eval_mode: bool = True,
) -> nn.Module:
    """Build one of the pinned ResNets.

    Args:
        key: ``"resnet18"`` | ``"resnet34"`` | ``"resnet50"``.
        pretrained: Load the pinned ImageNet weights. ``False`` gives a randomly
            initialised network (useful as a "trained from scratch" control when
            attributing a suspicious checkpoint).
        num_classes: Size of the classification head. When it differs from 1000
            the final ``fc`` layer is **replaced by a freshly, randomly
            initialised** ``nn.Linear(in_features, num_classes)`` — the ImageNet
            head is discarded, so the model must be fine-tuned before its logits
            mean anything. Everything up to and including ``avgpool`` keeps the
            pretrained weights.
        device: Target device; ``None`` uses :func:`aiforensics.device.get_device`
            (``cuda:0`` on this box). Pass ``"cpu"`` to keep it off the GPU.
        freeze_backbone: Freeze every parameter except the new head — the usual
            cheap "linear probe" fine-tune, and a realistic attacker budget for
            an on-premise poisoning scenario.
        eval_mode: Call ``.eval()`` before returning. Forensic analysis is
            inference; call ``.train()`` yourself when fine-tuning, otherwise
            BatchNorm running stats stay frozen.

    Returns:
        The configured ``nn.Module`` on ``device``.

    Raises:
        KeyError: If ``key`` is not a known ResNet.
    """
    from torchvision import models as tvm

    _spec(key)  # validate early, with a good message
    weights = resnet_weights(key) if pretrained else None
    model = getattr(tvm, key)(weights=weights)

    if num_classes != IMAGENET_NUM_CLASSES:
        from torch import nn as _nn

        in_features = model.fc.in_features
        model.fc = _nn.Linear(in_features, num_classes)
        logger.info(
            "replaced %s head: %d -> %d classes (new head is randomly initialised)",
            key, in_features, num_classes,
        )

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    model = model.to(get_device() if device is None else device)
    if eval_mode:
        model.eval()
    return model


def resnet_for_gtsrb(
    key: str = "resnet18",
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """ResNet wired for GTSRB traffic-sign classification (43 classes).

    The automated-driving target of the study. Pair it with
    ``imagenet_train_transform(hflip=False)``: traffic signs are not
    left-right symmetric, and mirroring them relabels several classes.

    Args:
        key: Which ResNet backbone to use.
        pretrained: Start from ImageNet weights (recommended — GTSRB is small).
        **kwargs: Forwarded to :func:`load_resnet`.

    Returns:
        ``nn.Module`` whose ``fc`` head has 43 randomly initialised outputs.
    """
    return load_resnet(key, pretrained=pretrained, num_classes=GTSRB_NUM_CLASSES, **kwargs)


def resnet_for_lfw(
    num_classes: int,
    key: str = "resnet50",
    *,
    pretrained: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """ResNet wired for LFW face identification.

    LFW has no fixed class count: it depends on the ``min_faces_per_person``
    threshold used when building the split (e.g. 62 identities at ``>=20``), so
    the caller must pass the number produced by the dataset layer rather than
    hard-coding it.

    Args:
        num_classes: Number of identities in the split being trained on.
        key: Backbone key; ResNet-50 by default because face identity needs
            richer features than traffic signs.
        pretrained: Start from ImageNet weights.
        **kwargs: Forwarded to :func:`load_resnet`.

    Returns:
        ``nn.Module`` with a randomly initialised ``num_classes``-way head.

    Raises:
        ValueError: If ``num_classes`` is not a positive integer.

    Note:
        ``sklearn.datasets.fetch_lfw_people`` returns grayscale images unless
        ``color=True``; feed 3-channel tensors (or prepend
        ``transforms.Grayscale(num_output_channels=3)``) so the ImageNet
        normalisation stays valid.
    """
    if num_classes < 1:
        raise ValueError(f"num_classes must be >= 1, got {num_classes}")
    return load_resnet(key, pretrained=pretrained, num_classes=num_classes, **kwargs)


# --------------------------------------------------------------------------
# transforms
# --------------------------------------------------------------------------
def imagenet_train_transform(size: int = 224, *, augment: bool = True, hflip: bool = True):
    """Standard ImageNet training transform, reused by every vision experiment.

    Args:
        size: Output crop size expected by the pinned weights (224).
        augment: ``True`` uses ``RandomResizedCrop``; ``False`` degrades to the
            deterministic eval pipeline, which is what you want when you need a
            train split whose samples are byte-identical across runs (e.g. when
            diffing a clean run against a poisoned one).
        hflip: Random horizontal flip. **Set ``False`` for GTSRB** — mirrored
            traffic signs are a different class.

    Returns:
        ``torchvision.transforms.Compose``.
    """
    from torchvision import transforms as T

    if not augment:
        return imagenet_eval_transform(size)
    ops = [T.RandomResizedCrop(size)]
    if hflip:
        ops.append(T.RandomHorizontalFlip())
    ops += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


def imagenet_eval_transform(size: int = 224, *, resize: int | None = None):
    """Standard deterministic ImageNet eval transform (resize + centre crop).

    Args:
        size: Centre-crop size (224).
        resize: Shorter-side resize before cropping; defaults to ``size * 256/224``.

    Returns:
        ``torchvision.transforms.Compose``.
    """
    from torchvision import transforms as T

    short_side = resize if resize is not None else int(round(size * 256 / 224))
    return T.Compose(
        [
            T.Resize(short_side),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------
# hidden representations
# --------------------------------------------------------------------------
def available_layers(model: nn.Module) -> list[str]:
    """Every hookable submodule name, in forward order.

    Handy when you want something other than ``avgpool`` — e.g. ``"layer3"`` for
    mid-level features, or ``"layer4.1.conv2"`` for a specific kernel.
    """
    return [name for name, _ in model.named_modules() if name]


@contextmanager
def capture_activations(
    model: nn.Module,
    layers: str | Sequence[str],
    *,
    detach: bool = True,
    to_cpu: bool = False,
) -> Iterator[dict[str, torch.Tensor]]:
    """Context manager that records forward activations of named submodules.

    The dict handed out is filled on every forward pass while the context is
    open (later passes overwrite earlier ones), and **all hooks are removed in a
    ``finally`` block** — a leaked hook silently changes the behaviour of every
    later experiment in the same process, which is exactly the kind of
    contamination a forensic run cannot afford.

    Args:
        model: Model to instrument.
        layers: One name or a sequence of names from :func:`available_layers`.
        detach: Detach captured tensors from the autograd graph (default; set
            ``False`` only if you need gradients through the activations).
        to_cpu: Move captures to CPU as they are produced, trading speed for GPU
            memory when sweeping many layers.

    Yields:
        ``dict[str, Tensor]`` mapping layer name -> most recent activation.

    Raises:
        KeyError: If a requested layer name does not exist on ``model``.
    """
    names = [layers] if isinstance(layers, str) else list(layers)
    modules = dict(model.named_modules())
    missing = [n for n in names if n not in modules]
    if missing:
        raise KeyError(
            f"layer(s) {missing} not found on {type(model).__name__}; "
            f"call available_layers(model) to list valid names"
        )

    store: dict[str, torch.Tensor] = {}
    handles = []

    def _make_hook(name: str):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if detach:
                tensor = tensor.detach()
            store[name] = tensor.cpu() if to_cpu else tensor

        return _hook

    try:
        for name in names:
            handles.append(modules[name].register_forward_hook(_make_hook(name)))
        yield store
    finally:
        for handle in handles:
            handle.remove()


def extract_features(
    model: nn.Module,
    inputs: torch.Tensor,
    layers: Sequence[str] = (PENULTIMATE_LAYER,),
    *,
    flatten: bool = True,
    device: str | Any | None = None,
    to_cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Run one batch and return activations for several named layers.

    Args:
        model: Model to probe (kept in whatever train/eval mode it is in, except
            that the forward runs under ``torch.no_grad``).
        inputs: Image batch, ``(B, C, H, W)``, already normalised.
        layers: Layer names to capture.
        flatten: Flatten each activation to ``(B, -1)``. Required for the
            matrix-factorisation style analyses (SVD on the representation
            matrix) the team plugs in later.
        device: Device to move ``inputs`` to; ``None`` means "wherever the
            model's first parameter already lives".
        to_cpu: Return CPU tensors (default) so the caller can hand them to
            numpy/scipy without another copy.

    Returns:
        ``dict[str, Tensor]`` keyed by layer name.
    """
    import torch as _torch

    target = device
    if target is None:
        try:
            target = next(model.parameters()).device
        except StopIteration:  # pragma: no cover - parameterless model
            target = get_device()
    inputs = inputs.to(target)

    with capture_activations(model, layers, to_cpu=False) as store:
        with _torch.no_grad():
            model(inputs)
        out = dict(store)

    for name, tensor in out.items():
        if flatten:
            tensor = _torch.flatten(tensor, start_dim=1)
        out[name] = tensor.cpu() if to_cpu else tensor
    return out


def feature_extractor(
    model: nn.Module,
    inputs: torch.Tensor,
    layer: str = PENULTIMATE_LAYER,
    *,
    flatten: bool = True,
    device: str | Any | None = None,
    to_cpu: bool = True,
) -> torch.Tensor:
    """Penultimate-layer (or any named-layer) activations for one batch.

    This is the hidden-representation accessor the vision-side anomaly analyses
    build on: stack the returned ``(B, D)`` rows over a dataset and factorise the
    matrix — poisoned samples with a shared trigger tend to concentrate in a
    low-rank direction that clean samples do not occupy.

    Args:
        model: Model to probe.
        inputs: Normalised image batch ``(B, C, H, W)``.
        layer: Module name; default ``"avgpool"``, the input to ``fc``.
        flatten: Return ``(B, D)`` instead of the raw activation shape.
        device: Device for ``inputs``; ``None`` follows the model.
        to_cpu: Return CPU tensors.

    Returns:
        Activation tensor for ``layer``.

    Raises:
        KeyError: If ``layer`` does not exist on ``model``.
    """
    return extract_features(
        model, inputs, [layer], flatten=flatten, device=device, to_cpu=to_cpu
    )[layer]


# --------------------------------------------------------------------------
# registry wiring
# --------------------------------------------------------------------------
for _key, (_enum, _member, _gb, _desc) in _RESNETS.items():
    register_model(
        ModelSpec(
            key=_key,
            family="vision",
            source=f"torchvision:{_enum}.{_member}",
            loader=functools.partial(load_resnet, _key),
            task="image-classification",
            num_classes=IMAGENET_NUM_CLASSES,
            size_gb=_gb,
            description=_desc,
            tags=("backbone", "imagenet"),
        )
    )
del _key, _enum, _member, _gb, _desc
