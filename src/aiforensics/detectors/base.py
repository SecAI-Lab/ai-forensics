"""The detector plug-in seam — **interface only, no detection logic**.

Detectors are organised as a two-level tree that mirrors the research plan's
detection axes:

* a **category** is a detection *approach* — what kind of signal it reads
  (``input-anomaly``, ``feature-space``, ``neuron-activation``, ``rag-defense``);
* a **technique** is one concrete method inside a category (``onion`` and
  ``rap`` both live under ``input-anomaly``).

Category keys are always approach names, never product names — that is why the
RAG category is ``rag-defense`` and *RAGDefender* is a technique inside it,
parallel to ``neuron-activation`` containing ``sae`` / ``badacts-nas`` / ``logit-lens``.

Nothing here detects anything. This module defines how a detector is built, how
it scores samples, how its threshold is applied, how it is evaluated against
ground truth, and how it is registered so the rest of the testbed can find it.

Adding a technique to an existing category
------------------------------------------

Two things: subclass :class:`Detector`, implement :meth:`Detector.score`, and
decorate it. The category is inferred when the key is already in the plan::

    from aiforensics.detectors import Detector, register_detector

    @register_detector("onion")                 # category inferred: input-anomaly
    class OnionDetector(Detector):
        name = "onion"
        modality = "text"

        def score(self, samples):
            ...                                 # higher = more suspicious

For a brand-new technique, name its category explicitly::

    @register_detector("my-method", category="input-anomaly")
    class MyDetector(Detector):
        ...

Adding a whole new category
---------------------------

Collaborators are not limited to the categories below — register your own and
it appears in the tree, in ``list_categories()`` and in the reports::

    from aiforensics.detectors import DetectorCategory, Technique, register_category

    register_category(DetectorCategory(
        key="gradient-signal",
        name="Gradient-signal anomaly detection",
        modality="activations",
        evidence="training logs + model access",
        module="aiforensics/detectors/gradient_signal.py",
        description="Rank samples by gradient-norm outlierness during fine-tuning.",
        techniques=(Technique("grad-norm", "Per-sample gradient-norm outlier", "…"),),
    ))

Extending to agentic environments
---------------------------------

The categories below are defined by the *signal* they read, not by the system
they were first demonstrated on, so they carry over to agentic deployments
without new machinery. An agent is still a model consuming inputs and producing
outputs; it simply exposes more of them. ``input-anomaly`` applies to whatever
text reaches the model, including tool outputs and inter-agent messages;
``feature-space`` and ``neuron-activation`` apply to any forward pass the agent
makes; ``rag-defense`` applies wherever retrieval feeds the context.

What changes in an agentic setting is the *evidence surface* — orchestrator
decisions, agent-to-agent conversations and tool-call traces are all recordable
through :class:`aiforensics.evidence.RunLogger` — and the *unit of detection*,
which may be a turn or a tool call rather than a training sample. Neither
requires a new category: point an existing technique at the new samples. Nothing
agent-specific is implemented here, and it is out of scope for this deliverable.

The one rule: **``score`` returns a per-sample suspicion score, higher meaning
more suspicious.** Everything downstream (threshold sweeps, ROC/PR curves,
``best_threshold``) assumes that orientation. If your statistic is naturally
"lower = more suspicious" (a likelihood, a cosine similarity), negate it inside
``score`` rather than special-casing it later.

Every technique listed in :data:`CATEGORIES` is **unimplemented on purpose** —
this project delivers the environment, not the detectors. :class:`RandomDetector`
is the single exception: a deliberately useless baseline so the wiring can be
tested end to end.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..metrics.detection import DetectionResult, as_mask, detection_metrics, manifest_to_mask

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORIES",
    "CategoryEnsemble",
    "Detector",
    "DetectorCategory",
    "RandomDetector",
    "Technique",
    "category_of",
    "detector_tree",
    "format_tree",
    "get_category",
    "get_detector",
    "list_categories",
    "list_detectors",
    "list_techniques",
    "register_category",
    "register_detector",
]


# --------------------------------------------------------------------------- #
# taxonomy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Technique:
    """One concrete method inside a category.

    Attributes:
        key: Registry key to implement under, lower-case and hyphenated.
        name: Human-readable name of the method.
        notes: Short description of the mechanism, for orientation only.
    """

    key: str
    name: str
    notes: str = ""


@dataclass(frozen=True)
class DetectorCategory:
    """A family of detectors that read the same kind of signal.

    Attributes:
        key: Category key — an *approach* name, never a product name.
        name: Human-readable name.
        modality: Default modality for techniques in this category — one of
            ``"text"``, ``"vision"``, ``"activations"``, ``"weights"``,
            ``"rag"`` or ``"any"``.
        evidence: Which evidence source from the threat model it consumes.
        module: Reserved file inside ``aiforensics/detectors/`` where the
            techniques in this category should live.
        description: One or two sentences on what the category exploits.
        techniques: Planned techniques. Implementing one does not remove it
            from here — this stays the plan; the registry tracks what exists.
    """

    key: str
    name: str
    modality: str
    evidence: str
    module: str
    description: str
    techniques: tuple[Technique, ...] = ()

    def technique_keys(self) -> list[str]:
        """Planned technique keys, in declaration order."""
        return [t.key for t in self.techniques]

    def to_dict(self) -> dict:
        """JSON-ready representation, for a run manifest or a report table."""
        return {
            "key": self.key,
            "name": self.name,
            "modality": self.modality,
            "evidence": self.evidence,
            "module": self.module,
            "description": self.description,
            "techniques": [
                {"key": t.key, "name": t.name, "notes": t.notes} for t in self.techniques
            ],
        }


#: The detection axes from the research plan. Every technique here is
#: **unimplemented on purpose**; :func:`get_detector` on one of these keys
#: raises :class:`NotImplementedError` naming the module to create.
CATEGORIES: dict[str, DetectorCategory] = {}


def register_category(category: DetectorCategory, *, overwrite: bool = False) -> DetectorCategory:
    """Add a category to the taxonomy.

    Args:
        category: The category to register.
        overwrite: Allow replacing an existing key. Off by default so one
            collaborator cannot silently shadow another's category.

    Returns:
        The registered category.

    Raises:
        ValueError: If the key exists and ``overwrite`` is False, or if a
            technique key collides with one already claimed by another category
            — technique keys are global, since :func:`get_detector` takes one.
    """
    if category.key in CATEGORIES and not overwrite:
        raise ValueError(
            f"category {category.key!r} is already registered; pass overwrite=True "
            f"to replace it deliberately"
        )
    for tech in category.techniques:
        owner = _TECHNIQUE_INDEX.get(tech.key)
        if owner is not None and owner != category.key:
            raise ValueError(
                f"technique key {tech.key!r} is already claimed by category {owner!r}; "
                f"technique keys must be unique across categories"
            )
    CATEGORIES[category.key] = category
    for tech in category.techniques:
        _TECHNIQUE_INDEX[tech.key] = category.key
    return category


#: technique key -> category key, maintained by :func:`register_category`.
_TECHNIQUE_INDEX: dict[str, str] = {}


for _category in (
    DetectorCategory(
        key="input-anomaly",
        name="Input anomaly detection",
        modality="text",
        evidence="fine-tuning dataset / inference logs",
        module="aiforensics/detectors/input_anomaly.py",
        description=(
            "Judge a sample from the input text alone (optionally with a reference LM), "
            "without needing the suspect model's internals. Triggers are statistically "
            "unnatural, so they stand out under fluency or perturbation-robustness tests."
        ),
        techniques=(
            Technique(
                key="onion",
                name="ONION — perplexity-based outlier-word detection",
                notes=(
                    "Remove each token in turn and measure the LM perplexity drop; a "
                    "trigger token is an outlier whose removal makes the sentence much "
                    "more fluent."
                ),
            ),
            Technique(
                key="rap",
                name="RAP — robustness-aware perturbation, logit-difference test",
                notes=(
                    "Insert a learned rare-word perturbation and compare the output-logit "
                    "drop on clean vs. triggered inputs; backdoored inputs are abnormally "
                    "robust to the perturbation."
                ),
            ),
        ),
    ),
    DetectorCategory(
        key="feature-space",
        name="Feature/representation-space anomaly detection",
        modality="activations",
        evidence="model weights + forward passes over the suspect dataset",
        module="aiforensics/detectors/feature_space.py",
        description=(
            "Poisoned samples sharing a trigger collapse into a low-rank direction of the "
            "representation matrix. Read the geometry of the hidden features rather than "
            "the inputs. Originally demonstrated on CNNs; applies to LM hidden states too."
        ),
        techniques=(
            Technique(
                key="hidden-svd",
                name="SVD / spectral anomaly on hidden-state representations",
                notes=(
                    "Collect hidden states per class, take the top right singular vector "
                    "of the centred matrix, and rank samples by their projection onto it."
                ),
            ),
            Technique(
                key="activation-clustering",
                name="Activation clustering (variant)",
                notes="Cluster per-class activations; a poisoned class splits in two.",
            ),
            Technique(
                key="spectral-signature",
                name="Spectral signatures (variant)",
                notes="Rank by correlation with the top singular direction of the covariance.",
            ),
        ),
    ),
    DetectorCategory(
        key="neuron-activation",
        name="Neuron activation analysis",
        modality="activations",
        evidence="model weights + forward passes",
        module="aiforensics/detectors/neuron_activation.py",
        description=(
            "Attribute the backdoor to specific units or features: look for neurons or "
            "sparse-autoencoder features that fire almost exclusively when the trigger is "
            "present, and localise which layer the behaviour appears in."
        ),
        techniques=(
            Technique(
                key="sae",
                name="Sparse autoencoder feature analysis",
                notes=(
                    "Train/apply an SAE on hidden states and look for features that "
                    "activate almost only on triggered inputs."
                ),
            ),
            Technique(
                key="badacts-nas",
                name="BadActs Neuron Activation State",
                notes=(
                    "The fraction of activation coordinates lying outside their clean "
                    "interval, over every transformer block concatenated (Yi et al., "
                    "Findings of ACL 2024). Named for the paper rather than for the "
                    "generic family, because 'NLA' collides with Anthropic's Natural "
                    "Language Autoencoders — a different method that emits explanations, "
                    "not a detection score."
                ),
            ),
            Technique(
                key="logit-lens",
                name="Logit lens layer attribution",
                notes=(
                    "Project intermediate residual states through the unembedding to see "
                    "at which layer the target label is written in."
                ),
            ),
            Technique(
                key="tuned-lens",
                name="Tuned lens layer attribution",
                notes=(
                    "A learned affine translator per layer, trained to match the model's "
                    "own final distribution in KL. Corrects the systematic distortion the "
                    "logit lens suffers from applying a readout to states it was never "
                    "fitted to. Available via LogitLensDetector(tuned=True)."
                ),
            ),
            Technique(
                key="jacobian-lens",
                name="Jacobian lens layer attribution",
                notes=(
                    "Decode a layer through the measured average effect of the remaining "
                    "layers, J = E[d h_final / d h_l], instead of assuming they are the "
                    "identity. Yields a MODEL-level screen rather than a per-sample score, "
                    "so it cannot be scored on #detected/#poisoned and belongs in its own "
                    "table."
                ),
            ),
        ),
    ),
    DetectorCategory(
        key="rag-defense",
        name="RAG poisoning defence",
        modality="rag",
        evidence="RAG corpus + retrieval logs",
        module="aiforensics/detectors/rag_defense.py",
        description=(
            "Decide whether a retrieved passage was adversarially inserted, before it "
            "reaches the generator. Injected passages are crafted to rank highly for one "
            "query, which makes them anomalous relative to their neighbours."
        ),
        techniques=(
            Technique(
                key="ragdefender",
                name="RAGDefender",
                notes=(
                    "Score retrieved passages for adversarial insertion using "
                    "embedding-space clustering / agreement among the retrieved set."
                ),
            ),
            Technique(
                key="retrieval-outlier",
                name="Retrieval-neighbourhood outlier scoring (variant)",
                notes="Flag passages whose similarity to the query is far above their similarity to neighbours.",
            ),
            Technique(
                key="answer-consistency",
                name="Cross-passage answer consistency (variant)",
                notes="Flag passages that disagree with the consensus answer of the retrieved set.",
            ),
        ),
    ),
    DetectorCategory(
        key="baseline",
        name="Reference baselines",
        modality="any",
        evidence="none",
        module="aiforensics/detectors/base.py",
        description=(
            "Null baselines that exist to make the plumbing testable and to give the "
            "report an honest floor. Not detection methods."
        ),
        techniques=(),
    ),
):
    register_category(_category)
del _category


def get_category(key: str) -> DetectorCategory:
    """Look up a category by key.

    Raises:
        KeyError: If unknown, listing the valid category keys.
    """
    try:
        return CATEGORIES[key]
    except KeyError:
        raise KeyError(
            f"unknown detector category {key!r}. Valid categories: {list_categories()}"
        ) from None


def list_categories() -> list[str]:
    """Sorted category keys."""
    return sorted(CATEGORIES)


def list_techniques(category: str | None = None) -> list[str]:
    """Planned technique keys, optionally restricted to one category.

    These are the *slots*: what the plan says should exist. Use
    :func:`list_detectors` for what is actually runnable.
    """
    if category is not None:
        return get_category(category).technique_keys()
    return sorted(_TECHNIQUE_INDEX)


def category_of(key: str) -> str | None:
    """Category key owning ``key``, whether it is registered or merely planned."""
    if key in _DETECTOR_CATEGORY:
        return _DETECTOR_CATEGORY[key]
    return _TECHNIQUE_INDEX.get(key)


# --------------------------------------------------------------------------- #
# the interface
# --------------------------------------------------------------------------- #
class Detector(ABC):
    """Abstract base class for every poison/backdoor detector in the testbed.

    Subclasses implement exactly one required method, :meth:`score`. Everything
    else has a working default so a new technique is a small file.

    Class attributes to override:
        name: Short registry-style identifier, e.g. ``"onion"``.
        category: Key of the :class:`DetectorCategory` this belongs to. Left
            empty it is filled in by :func:`register_detector`.
        modality: What the detector consumes — ``"text"``, ``"vision"``,
            ``"activations"``, ``"weights"``, ``"rag"``, ``"agent"`` or
            ``"any"``. Used to check a detector is not pointed at data it
            cannot read.
        requires_fit: Set ``True`` if :meth:`score` is invalid before
            :meth:`fit` has seen clean reference data. :meth:`score` callers get
            a loud error instead of silently meaningless numbers.
        higher_is_suspicious: Documented as always ``True``; present as an
            attribute only so the convention is discoverable in ``dir()``.

    Instance state:
        threshold: Decision threshold used by :meth:`predict` when the caller
            does not pass one. ``None`` until you set it — deliberately, so
            nobody inherits an arbitrary default and reports it as a result.
        config: The keyword arguments the detector was constructed with,
            recorded verbatim so a run manifest can reproduce it.

    Example of a conforming implementation (illustrative — do not copy the
    scoring, it is nonsense)::

        @register_detector("my-technique", category="input-anomaly")
        class MyDetector(Detector):
            name = "my-technique"
            modality = "text"

            def score(self, samples):
                return np.asarray([len(s) for s in samples], dtype=float)
    """

    name: str = "detector"
    category: str = ""
    modality: str = "text"
    requires_fit: bool = False
    higher_is_suspicious: bool = True

    def __init__(self, *, threshold: float | None = None, **config: Any) -> None:
        self.threshold = threshold
        self.config: dict[str, Any] = dict(config)
        self._fitted = False

    # -- optional calibration --------------------------------------------- #
    def fit(self, clean_data: Any, **kwargs: Any) -> Detector:
        """Learn whatever reference statistics the technique needs from clean data.

        Default implementation is a no-op that just marks the detector fitted —
        override it if your method needs a clean baseline (a reference
        perplexity distribution, a clean-activation subspace, an untampered
        weight snapshot).

        Note that "clean" here means *believed* clean. In the on-premise
        scenario the investigator may only have the suspect fine-tuning set, so
        a detector that hard-requires clean reference data should say so via
        ``requires_fit = True`` rather than degrading silently.

        Args:
            clean_data: Reference samples, in whatever form :meth:`score`
                accepts.
            **kwargs: Technique-specific options.

        Returns:
            ``self``, so ``det.fit(x).score(y)`` chains.
        """
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._fitted

    # -- the one thing subclasses must write ------------------------------- #
    @abstractmethod
    def score(self, samples: Any) -> np.ndarray:
        """Return a per-sample suspicion score. **Higher = more suspicious.**

        This is the only required method. Everything downstream — thresholding,
        ROC/PR curves, threshold selection — depends on the orientation, so a
        statistic that is naturally "lower = worse" must be negated here.

        Args:
            samples: A sequence of samples, a dataset, a batch of activations —
                whatever your ``modality`` implies. Document the accepted form
                in your subclass.

        Returns:
            A 1-D float ``np.ndarray`` with one score per sample, in input
            order. Order matters: the evaluation joins scores to the poison
            manifest positionally.
        """
        raise NotImplementedError

    # -- provided plumbing -------------------------------------------------- #
    def predict(self, samples: Any, threshold: float | None = None) -> np.ndarray:
        """Flag samples as poisoned by thresholding :meth:`score`.

        Args:
            samples: Same form as :meth:`score`.
            threshold: Decision threshold. Falls back to ``self.threshold``.

        Returns:
            A 1-D boolean array, ``True`` = flagged as poisoned.

        Raises:
            RuntimeError: If the detector requires fitting and has not been fit.
            ValueError: If no threshold is available. There is no sensible
                default: pick one on a held-out split with
                :func:`aiforensics.metrics.best_threshold`, or set
                ``self.threshold`` explicitly, so the choice is visible in the
                report rather than buried in a library default.
        """
        self._check_fitted()
        thr = threshold if threshold is not None else self.threshold
        if thr is None:
            raise ValueError(
                f"{type(self).__name__} has no decision threshold. Pass threshold=…, set "
                f"detector.threshold, or derive one from labelled calibration data with "
                f"aiforensics.metrics.best_threshold(y_true, scores)."
            )
        scores = np.asarray(self.score(samples), dtype=np.float64).reshape(-1)
        return scores >= float(thr)

    def evaluate(
        self,
        samples: Any,
        manifest: Any,
        *,
        threshold: float | None = None,
        n: int | None = None,
        label: str | None = None,
    ) -> DetectionResult:
        """Score ``samples``, threshold them, and compare against ground truth.

        The single scoring path for the whole team, so that every technique's
        numbers are produced identically and are therefore comparable.

        Args:
            samples: Data to evaluate, in the form :meth:`score` accepts.
            manifest: Ground truth — a ``PoisonManifest``-like object, a boolean
                mask, or a list of poisoned indices (see
                :func:`aiforensics.metrics.manifest_to_mask`).
            threshold: Decision threshold; falls back to ``self.threshold``.
            n: Total sample count, when ``manifest`` is an index list that
                cannot supply one.
            label: Row label for the report; defaults to the detector name.

        Returns:
            A :class:`~aiforensics.metrics.DetectionResult` whose
            ``detection_rate`` is the plan's headline metric.
        """
        self._check_fitted()
        scores = np.asarray(self.score(samples), dtype=np.float64).reshape(-1)
        y_true = (
            as_mask(manifest, n=n if n is not None else scores.size, name="manifest")
            if isinstance(manifest, (np.ndarray, list, tuple, set, frozenset, range))
            else manifest_to_mask(manifest, n=n if n is not None else scores.size)
        )
        thr = threshold if threshold is not None else self.threshold
        if thr is None:
            raise ValueError(
                f"{type(self).__name__}.evaluate needs a threshold. Pass threshold=…, set "
                f"detector.threshold, or use aiforensics.metrics.best_threshold / "
                f"threshold_sweep on a calibration split first."
            )
        return detection_metrics(
            y_true, scores >= float(thr),
            threshold=float(thr), label=label or self.name,
        )

    def _check_fitted(self) -> None:
        if self.requires_fit and not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} declares requires_fit=True but fit() has not been "
                f"called; scores would be meaningless. Call detector.fit(clean_data) first."
            )

    def describe(self) -> dict:
        """Detector identity + configuration, for a run manifest or log event."""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "category": self.category or category_of(self.name),
            "modality": self.modality,
            "requires_fit": self.requires_fit,
            "fitted": self._fitted,
            "threshold": self.threshold,
            "config": dict(self.config),
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"{type(self).__name__}(name={self.name!r}, category={self.category!r}, "
            f"modality={self.modality!r})"
        )


# --------------------------------------------------------------------------- #
# registry — mirrors the model / dataset registries, plus the category tree
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, Callable[..., Detector]] = {}
_DETECTOR_CATEGORY: dict[str, str] = {}

#: Built-in technique modules, imported on first registry access. Importing them
#: is side-effect-only (each registers its detectors) and must stay cheap: heavy
#: third-party dependencies belong inside ``__init__``, not at module scope, so
#: that listing detectors never fails on a machine missing an optional package.
_BUILTIN_MODULES = ("rag_defense", "rag_gmtp", "rag_npas", "rag_perplexity",
                    "rag_reliability", "feature_space",
                    "neuron_activation", "input_anomaly")
_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import the built-in technique modules exactly once."""
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True  # set first: a failed import must not retry forever
    import importlib

    for mod in _BUILTIN_MODULES:
        try:
            importlib.import_module(f"{__package__}.{mod}")
        except Exception:  # pragma: no cover - optional dependency missing
            logger.warning("could not import built-in detector module %r", mod, exc_info=True)


def register_detector(
    key: str,
    factory: Callable[..., Detector] | None = None,
    *,
    category: str | None = None,
) -> Any:
    """Register a detector class (or factory) under a string key.

    Usable as a decorator or as a plain call::

        @register_detector("onion")                       # category inferred
        class OnionDetector(Detector):
            ...

        @register_detector("my-method", category="input-anomaly")
        class MyDetector(Detector):
            ...

        register_detector("onion", OnionDetector)

    Args:
        key: Lookup key. Lower-case, hyphen-separated; matching a planned
            technique key from :data:`CATEGORIES` where applicable.
        factory: Callable returning a :class:`Detector`. Omit to use as a
            decorator.
        category: Category key. Optional when ``key`` is already a planned
            technique — the category is then inferred from the taxonomy.

    Returns:
        The decorator, or the registered factory.

    Raises:
        ValueError: If ``key`` is already registered — silent shadowing of one
            researcher's detector by another's is exactly the bug this catches.
        KeyError: If ``category`` names a category that does not exist.
        LookupError: If no category is given and ``key`` is not a planned
            technique, so the detector would have nowhere to sit in the tree.
    """
    # Resolution order: explicit > planned technique > the category the key is
    # already registered under. The last case matters so that re-registering an
    # existing key reports the duplicate rather than a confusing category error.
    resolved = category
    if resolved is None:
        resolved = _TECHNIQUE_INDEX.get(key) or _DETECTOR_CATEGORY.get(key)
    if resolved is None:
        raise LookupError(
            f"cannot place detector {key!r} in the tree: it is not a planned technique, so "
            f"pass category=… explicitly. Existing categories: {list_categories()}. "
            f"To start a new family, call register_category() first."
        )
    get_category(resolved)  # validates, raises KeyError listing valid keys

    def _register(fn: Callable[..., Detector]) -> Callable[..., Detector]:
        if key in _REGISTRY and _REGISTRY[key] is not fn:
            raise ValueError(
                f"detector key {key!r} is already registered to "
                f"{getattr(_REGISTRY[key], '__name__', _REGISTRY[key])}; pick another key"
            )
        _REGISTRY[key] = fn
        _DETECTOR_CATEGORY[key] = resolved
        if isinstance(fn, type) and issubclass(fn, Detector) and not fn.category:
            fn.category = resolved
        return fn

    if factory is not None:
        return _register(factory)
    return _register


def get_detector(key: str, **kwargs: Any) -> Detector:
    """Instantiate a registered detector by key.

    Args:
        key: Registry key, as returned by :func:`list_detectors`.
        **kwargs: Forwarded to the detector's constructor.

    Returns:
        A ready-to-use :class:`Detector`.

    Raises:
        NotImplementedError: If ``key`` names a planned technique that nobody
            has written yet — with its category and reserved module path, so the
            message doubles as the assignment.
        KeyError: If ``key`` is unknown, listing what is available.
    """
    _ensure_builtins()
    if key in _REGISTRY:
        return _REGISTRY[key](**kwargs)
    if key in _TECHNIQUE_INDEX:
        cat = CATEGORIES[_TECHNIQUE_INDEX[key]]
        tech = next(t for t in cat.techniques if t.key == key)
        raise NotImplementedError(
            f"detector {key!r} ({tech.name}) is planned but not implemented — "
            f"deliberately: this project builds the environment, not the detectors. "
            f"It belongs to category {cat.key!r} ({cat.name}). Implement it in "
            f"{cat.module} as a Detector subclass and register it with "
            f"register_detector({key!r})."
        )
    raise KeyError(
        f"unknown detector {key!r}. Registered: {list_detectors()}. "
        f"Planned but unimplemented: {sorted(set(_TECHNIQUE_INDEX) - set(_REGISTRY))}"
    )


def list_detectors(category: str | None = None, *, modality: str | None = None) -> list[str]:
    """Sorted keys of every *implemented* detector.

    Planned-but-unimplemented techniques are intentionally absent — this list is
    what you can actually run. Use :func:`list_techniques` for the plan.

    Args:
        category: Restrict to one category key.
        modality: Restrict to detectors declaring this modality.
    """
    _ensure_builtins()
    keys = sorted(_REGISTRY)
    if category is not None:
        get_category(category)
        keys = [k for k in keys if _DETECTOR_CATEGORY.get(k) == category]
    if modality is not None:
        keys = [k for k in keys if getattr(_REGISTRY[k], "modality", None) in (modality, "any")]
    return keys


def detector_tree() -> dict[str, dict]:
    """The whole taxonomy as nested dicts: category -> techniques.

    Each technique carries ``implemented``, so a report can show coverage
    without cross-referencing two lists.
    """
    _ensure_builtins()
    tree: dict[str, dict] = {}
    for ckey in sorted(CATEGORIES):
        cat = CATEGORIES[ckey]
        planned = {
            t.key: {"name": t.name, "notes": t.notes, "implemented": t.key in _REGISTRY}
            for t in cat.techniques
        }
        # Detectors registered into this category that were not in the plan.
        for dkey, dcat in sorted(_DETECTOR_CATEGORY.items()):
            if dcat == ckey and dkey not in planned:
                planned[dkey] = {"name": dkey, "notes": "", "implemented": True}
        tree[ckey] = {
            "name": cat.name,
            "modality": cat.modality,
            "evidence": cat.evidence,
            "module": cat.module,
            "techniques": planned,
        }
    return tree


def format_tree(*, show_notes: bool = False) -> str:
    """Render :func:`detector_tree` as an indented, human-readable string."""
    lines: list[str] = []
    tree = detector_tree()
    for ckey, cat in tree.items():
        techs = cat["techniques"]
        done = sum(1 for t in techs.values() if t["implemented"])
        lines.append(
            f"{ckey}  ({cat['name']}) — {cat['modality']} — "
            f"{done}/{len(techs)} implemented"
        )
        items = list(techs.items())
        for i, (tkey, t) in enumerate(items):
            branch = "└─" if i == len(items) - 1 else "├─"
            mark = "✓" if t["implemented"] else "·"
            lines.append(f"  {branch} {mark} {tkey:<22} {t['name']}")
            if show_notes and t["notes"]:
                pad = "     " if i == len(items) - 1 else "  │  "
                lines.append(f"  {pad}   {t['notes']}")
        if not items:
            lines.append("  └─ (no techniques fixed yet)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# category-level aggregation
# --------------------------------------------------------------------------- #
class CategoryEnsemble(Detector):
    """Run every implemented detector in a category and combine their scores.

    This is aggregation plumbing, not a detection method: it exists so the
    hierarchy is usable ("how does the whole input-anomaly axis do?") without
    each researcher writing their own loop.

    Raw scores from different techniques are on incompatible scales, so they are
    **rank-normalised to [0, 1] per detector** before combining. That makes the
    combination scale-free but also means an ensemble score is only meaningful
    relative to the other samples in the same call.

    Args:
        category: Category key to aggregate.
        combine: ``"mean"`` (consensus) or ``"max"`` (any-detector-fires).
        threshold: Decision threshold on the combined score.
        detectors: Explicit detector keys to use; defaults to every implemented
            detector in the category.
        **config: Forwarded to each child detector's constructor.

    Raises:
        LookupError: If the category has no implemented detectors yet.
    """

    def __init__(
        self,
        category: str,
        *,
        combine: str = "mean",
        threshold: float | None = None,
        detectors: Iterable[str] | None = None,
        **config: Any,
    ) -> None:
        super().__init__(threshold=threshold, combine=combine, **config)
        cat = get_category(category)
        keys = list(detectors) if detectors is not None else list_detectors(category)
        if not keys:
            raise LookupError(
                f"category {category!r} ({cat.name}) has no implemented detectors yet; "
                f"planned techniques: {cat.technique_keys() or '(none fixed)'}"
            )
        if combine not in {"mean", "max"}:
            raise ValueError(f"combine must be 'mean' or 'max', got {combine!r}")
        self.category = category
        self.name = f"{category}-ensemble"
        self.modality = cat.modality
        self.combine = combine
        self.children: dict[str, Detector] = {k: get_detector(k, **config) for k in keys}
        self.requires_fit = any(d.requires_fit for d in self.children.values())

    def fit(self, clean_data: Any, **kwargs: Any) -> CategoryEnsemble:
        """Fit every child that needs fitting."""
        for det in self.children.values():
            det.fit(clean_data, **kwargs)
        self._fitted = True
        return self

    def score(self, samples: Any) -> np.ndarray:
        """Rank-normalise each child's scores, then combine them."""
        stacked = np.vstack([
            _rank_normalise(np.asarray(d.score(samples), dtype=np.float64).reshape(-1))
            for d in self.children.values()
        ])
        return stacked.mean(axis=0) if self.combine == "mean" else stacked.max(axis=0)

    def describe(self) -> dict:
        """Ensemble identity plus every child's description."""
        base = super().describe()
        base["children"] = {k: d.describe() for k, d in self.children.items()}
        base["combine"] = self.combine
        return base


def _rank_normalise(scores: np.ndarray) -> np.ndarray:
    """Map scores to ``[0, 1]`` by rank; ties share the average rank."""
    n = scores.size
    if n == 0:
        return scores
    if n == 1:
        return np.zeros(1, dtype=np.float64)
    order = scores.argsort()
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # average ranks within tie groups so equal scores stay equal
    uniq, inverse = np.unique(scores, return_inverse=True)
    if uniq.size != n:
        sums = np.zeros(uniq.size, dtype=np.float64)
        counts = np.zeros(uniq.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        np.add.at(counts, inverse, 1.0)
        ranks = (sums / counts)[inverse]
    return ranks / (n - 1)


# --------------------------------------------------------------------------- #
# wiring example — NOT a detection method
# --------------------------------------------------------------------------- #
@register_detector("random", category="baseline")
class RandomDetector(Detector):
    """Uniform-random scores. **Wiring example / baseline only.**

    This detects nothing. It exists for two reasons:

    1. it makes the plumbing testable end-to-end — registry lookup, ``score`` →
       ``predict`` → ``evaluate`` → :class:`DetectionResult` — before any real
       technique exists;
    2. it is the correct null baseline for the report: any real detector must
       beat chance, and ROC AUC ≈ 0.5 / detection rate ≈ the flag rate is what
       chance looks like on your dataset.

    Do not read anything into its numbers, and do not copy its ``score``
    implementation as a starting point for a real method beyond the signature.

    Args:
        threshold: Decision threshold; defaults to ``0.5`` because random
            scores are uniform on ``[0, 1)`` — a real detector must not default
            its threshold like this.
        seed: RNG seed, so the "baseline" row is at least reproducible.
    """

    name = "random"
    category = "baseline"
    modality = "any"
    requires_fit = False

    def __init__(self, *, threshold: float | None = 0.5, seed: int | None = None,
                 **config: Any) -> None:
        super().__init__(threshold=threshold, seed=seed, **config)
        from ..config import settings

        self._rng = np.random.default_rng(settings.seed if seed is None else seed)

    def score(self, samples: Any) -> np.ndarray:
        """Uniform random scores in ``[0, 1)``, one per sample.

        Args:
            samples: Anything with a length, or an integer sample count.

        Returns:
            A 1-D float array of length ``len(samples)``.
        """
        return self._rng.random(_length_of(samples))


def _length_of(samples: Any) -> int:
    """Number of samples in ``samples``; accepts an int for convenience."""
    if isinstance(samples, (int, np.integer)):
        return int(samples)
    try:
        return len(samples)
    except TypeError:
        pass
    if isinstance(samples, Iterable) and not isinstance(samples, (str, bytes)):
        return sum(1 for _ in samples)
    raise TypeError(
        f"cannot determine the number of samples from {type(samples).__name__}; "
        f"pass a sized sequence or an integer count"
    )
