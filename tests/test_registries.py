"""Registry contracts: model keys, dataset keys, detector keys.

Registries are the naming layer the whole project's provenance rests on — a run
manifest that records ``model="qwen3-1.7b"`` is only reproducible if that key
still resolves to the same pinned source. So these tests pin the *keys*, the
*sources* and the *failure messages*, and never load a byte of the artifacts
themselves (nothing here touches the network).
"""

from __future__ import annotations

import numpy as np
import pytest

from aiforensics.data import registry as dsr
from aiforensics.detectors import base as det
from aiforensics.models import registry as mdl

EXPECTED_MODELS = ["qwen3-1.7b", "qwen3-8b", "resnet18", "resnet34", "resnet50"]
EXPECTED_DATASETS = ["ag_news", "cifar10", "gtsrb", "imdb", "lfw", "nq", "sst2"]


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_model_keys_are_the_pinned_set() -> None:
    assert mdl.list_models() == EXPECTED_MODELS


def test_model_families_split_cleanly() -> None:
    assert mdl.list_models("llm") == ["qwen3-1.7b", "qwen3-8b"]
    assert mdl.list_models("vision") == ["resnet18", "resnet34", "resnet50"]


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown family"):
        mdl.list_models("audio")


@pytest.mark.parametrize(
    ("key", "family", "source"),
    [
        ("qwen3-1.7b", "llm", "Qwen/Qwen3-1.7B"),
        ("qwen3-8b", "llm", "Qwen/Qwen3-8B"),
        ("resnet18", "vision", "torchvision:ResNet18_Weights.IMAGENET1K_V1"),
        ("resnet50", "vision", "torchvision:ResNet50_Weights.IMAGENET1K_V2"),
    ],
)
def test_model_specs_pin_their_upstream_source(key: str, family: str, source: str) -> None:
    spec = mdl.get_spec(key)
    assert spec.family == family
    assert spec.source == source
    assert callable(spec.loader)


def test_model_key_lookup_is_case_insensitive() -> None:
    assert mdl.get_spec("ResNet18").key == "resnet18"


def test_unknown_model_key_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        mdl.get_spec("resnet19")
    message = str(excinfo.value)
    assert all(key in message for key in EXPECTED_MODELS)


def test_model_spec_to_dict_drops_the_loader() -> None:
    dumped = mdl.get_spec("resnet18").to_dict()
    assert "loader" not in dumped
    assert dumped["key"] == "resnet18"


def test_registering_a_duplicate_key_is_refused() -> None:
    spec = mdl.get_spec("resnet18")
    with pytest.raises(KeyError, match="already registered"):
        mdl.register_model(spec)


def test_describe_models_covers_every_key() -> None:
    described = mdl.describe_models()
    keys = list(described["key"]) if hasattr(described, "columns") else [r["key"] for r in described]
    assert sorted(keys) == EXPECTED_MODELS


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def test_dataset_keys_are_the_pinned_set() -> None:
    assert dsr.list_datasets() == EXPECTED_DATASETS


def test_dataset_modalities_split_cleanly() -> None:
    assert dsr.list_datasets("vision") == ["cifar10", "gtsrb", "lfw"]
    assert dsr.list_datasets("text") == ["ag_news", "imdb", "sst2"]
    assert dsr.list_datasets("rag") == ["nq"]


def test_unknown_modality_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown modality"):
        dsr.list_datasets("tabular")


@pytest.mark.parametrize(
    ("key", "source"),
    [
        ("cifar10", "uoft-cs/cifar10"),
        ("gtsrb", "tanganke/gtsrb"),
        ("lfw", "sklearn:fetch_lfw_people"),
        ("sst2", "stanfordnlp/sst2"),
        ("ag_news", "fancyzhx/ag_news"),
        ("imdb", "stanfordnlp/imdb"),
        ("nq", "BeIR/nq"),
    ],
)
def test_dataset_specs_pin_their_upstream_source(key: str, source: str) -> None:
    assert dsr.get_spec(key).source == source


def test_label_names_line_up_with_num_classes() -> None:
    """A backdoor's ``target_label`` is meaningless without a fixed ordering."""
    for key in dsr.list_datasets():
        spec = dsr.get_spec(key)
        if spec.label_names is not None:
            assert len(spec.label_names) == spec.num_classes, key


def test_sst2_evaluates_on_validation_because_test_is_unlabeled() -> None:
    assert dsr.get_spec("sst2").eval_split == "validation"


def test_check_split_rejects_an_unknown_split() -> None:
    with pytest.raises(KeyError, match="unknown split"):
        dsr.get_spec("sst2").check_split("dev")


def test_hf_args_omits_the_literal_default_config() -> None:
    assert dsr.get_spec("sst2").hf_args() == ("stanfordnlp/sst2",)
    assert dsr.get_spec("cifar10").hf_args() == ("uoft-cs/cifar10", "plain_text")


def test_every_loader_reference_resolves() -> None:
    """A typo in a ``module:function`` reference must fail here, not mid-experiment."""
    for key in dsr.list_datasets():
        assert callable(dsr.get_spec(key).resolve_loader()), key


def test_unknown_dataset_key_lists_the_valid_ones() -> None:
    with pytest.raises(KeyError) as excinfo:
        dsr.load_dataset("sst3")
    assert "sst2" in str(excinfo.value)


def test_registering_a_duplicate_dataset_is_refused() -> None:
    with pytest.raises(KeyError, match="already registered"):
        dsr.register_dataset(dsr.get_spec("sst2"))


def test_dataset_spec_to_dict_is_json_shaped() -> None:
    import json

    json.dumps(dsr.get_spec("gtsrb").to_dict())


def test_describe_datasets_is_a_table() -> None:
    table = dsr.describe_datasets()
    assert "KEY" in table
    for key in EXPECTED_DATASETS:
        assert key in table


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
def test_implemented_detectors_are_the_rag_defense_set_plus_the_baseline() -> None:
    """Only PUBLISHED RAG defences are registered; other axes are open slots.

    The rag-defense category is the first work package, so it has real
    detectors. input-anomaly, feature-space and neuron-activation are still
    unimplemented on purpose — that is the contribution seam for the rest of the
    team, and this test fails loudly if a slot is filled without updating the plan.
    """
    assert det.list_detectors("baseline") == ["random"]
    assert set(det.list_detectors("rag-defense")) == {
        "gmtp", "npas", "perplexity", "ragdefender", "reliability",
    }
    for empty in ("input-anomaly", "feature-space", "neuron-activation"):
        assert det.list_detectors(empty) == [], f"{empty} unexpectedly has an implementation"


def test_categories_cover_the_plans_detection_axes() -> None:
    """Category keys are approach names, never product names."""
    assert set(det.list_categories()) == {
        "input-anomaly", "feature-space", "neuron-activation",
        "rag-defense", "baseline",
    }
    for key in det.list_categories():
        category = det.get_category(key)
        assert category.module.startswith("aiforensics/detectors/")
        assert category.name and category.description


def test_techniques_sit_under_the_right_category() -> None:
    assert det.list_techniques("input-anomaly") == ["onion", "rap"]
    assert "hidden-svd" in det.list_techniques("feature-space")
    assert det.list_techniques("neuron-activation") == ["sae", "nla", "logit-lens"]
    # RAGDefender is a *technique* inside the rag-defense category, not a category.
    assert "ragdefender" in det.list_techniques("rag-defense")
    assert "ragdefender" not in det.list_categories()
    for key in ("onion", "rap"):
        assert det.category_of(key) == "input-anomaly"


def test_planned_technique_raises_not_implemented_with_its_reserved_module() -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        det.get_detector("onion")
    message = str(excinfo.value)
    assert "input_anomaly.py" in message
    assert "input-anomaly" in message


def test_detector_tree_reports_implementation_coverage() -> None:
    tree = det.detector_tree()
    assert tree["input-anomaly"]["techniques"]["onion"]["implemented"] is False
    assert tree["baseline"]["techniques"]["random"]["implemented"] is True
    rendered = det.format_tree()
    assert "input-anomaly" in rendered and "onion" in rendered


def test_unknown_detector_lists_registered_and_planned() -> None:
    with pytest.raises(KeyError) as excinfo:
        det.get_detector("magic")
    message = str(excinfo.value)
    assert "random" in message and "onion" in message


def test_random_detector_scores_one_value_per_sample() -> None:
    detector = det.get_detector("random", seed=3)
    scores = detector.score(["a", "b", "c"])
    assert scores.shape == (3,)
    assert ((scores >= 0.0) & (scores < 1.0)).all()


def test_random_detector_is_reproducible() -> None:
    first = det.get_detector("random", seed=11).score(10)
    second = det.get_detector("random", seed=11).score(10)
    assert (first == second).all()


def test_predict_without_a_threshold_is_a_loud_error() -> None:
    detector = det.RandomDetector(threshold=None)
    with pytest.raises(ValueError, match="no decision threshold"):
        detector.predict(["a", "b"])


def test_duplicate_detector_registration_is_refused() -> None:
    with pytest.raises(ValueError, match="already registered"):
        det.register_detector("random", lambda **_: det.RandomDetector())


# --------------------------------------------------------------------------- #
# extending the tree — the collaborator workflow
# --------------------------------------------------------------------------- #
@pytest.fixture
def clean_registry():
    """Snapshot and restore the global detector tree.

    Registration mutates module-level state, so without this a test that adds a
    detector would change what every later test sees.
    """
    targets = (
        det._REGISTRY,
        det._DETECTOR_CATEGORY,
        det._TECHNIQUE_INDEX,
        det.CATEGORIES,
    )
    saved = [dict(t) for t in targets]
    yield
    for target, snapshot in zip(targets, saved, strict=True):
        target.clear()
        target.update(snapshot)


def _toy(key: str, modality: str = "text"):
    class Toy(det.Detector):
        name = key
        def score(self, samples):
            return np.arange(len(samples), dtype=float)
    Toy.modality = modality
    return Toy


def test_registering_a_planned_technique_infers_its_category(clean_registry) -> None:
    det.register_detector("onion", _toy("onion"))
    assert det.category_of("onion") == "input-anomaly"
    assert det.list_detectors("input-anomaly") == ["onion"]


def test_a_new_technique_needs_an_explicit_category(clean_registry) -> None:
    with pytest.raises(LookupError, match="not a planned technique"):
        det.register_detector("brand-new-idea", _toy("brand-new-idea"))
    det.register_detector("brand-new-idea", _toy("brand-new-idea"), category="input-anomaly")
    assert "brand-new-idea" in det.list_detectors("input-anomaly")


def test_registering_into_an_unknown_category_is_refused(clean_registry) -> None:
    with pytest.raises(KeyError, match="unknown detector category"):
        det.register_detector("whatever", _toy("whatever"), category="no-such-family")


def test_collaborators_can_add_a_whole_new_category(clean_registry) -> None:
    det.register_category(det.DetectorCategory(
        key="gradient-signal",
        name="Gradient-signal anomaly detection",
        modality="activations",
        evidence="training logs",
        module="aiforensics/detectors/gradient_signal.py",
        description="Rank samples by gradient-norm outlierness.",
        techniques=(det.Technique("grad-norm", "Per-sample gradient-norm outlier"),),
    ))
    assert "gradient-signal" in det.list_categories()
    assert det.category_of("grad-norm") == "gradient-signal"
    det.register_detector("grad-norm", _toy("grad-norm", "activations"))
    assert det.detector_tree()["gradient-signal"]["techniques"]["grad-norm"]["implemented"]


def test_a_technique_key_cannot_be_claimed_by_two_categories(clean_registry) -> None:
    with pytest.raises(ValueError, match="already claimed"):
        det.register_category(det.DetectorCategory(
            key="rival", name="Rival", modality="text",
            evidence="none", module="aiforensics/detectors/rival.py",
            description="tries to steal onion",
            techniques=(det.Technique("onion", "stolen"),),
        ))


def test_category_ensemble_combines_every_child(clean_registry) -> None:
    det.register_detector("onion", _toy("onion"))
    det.register_detector("rap", _toy("rap"))
    ensemble = det.CategoryEnsemble("input-anomaly", threshold=0.5)
    assert sorted(ensemble.children) == ["onion", "rap"]
    scores = ensemble.score(["a", "b", "c", "d"])
    assert scores.shape == (4,)
    # rank-normalised to [0, 1], and both children rank samples identically here
    assert scores.min() == pytest.approx(0.0) and scores.max() == pytest.approx(1.0)


def test_ensemble_on_an_empty_category_says_so(clean_registry) -> None:
    with pytest.raises(LookupError, match="no implemented detectors"):
        det.CategoryEnsemble("neuron-activation")


# --------------------------------------------------------------------------- #
# the real loaders — skipped unless --runslow (they need the shared caches)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_load_model_returns_the_documented_type() -> None:
    """Vision keys give an ``nn.Module``; LLM keys give ``(model, tokenizer)``."""
    from torch import nn

    from aiforensics.models import load_model

    model = load_model("resnet18", device="cpu")
    assert isinstance(model, nn.Module)
    assert model.fc.out_features == 1000
    assert not model.training, "loaders return models in eval mode"


@pytest.mark.slow
def test_load_model_replaces_the_head_for_a_new_class_count() -> None:
    from aiforensics.models import load_model

    model = load_model("resnet18", num_classes=43, device="cpu")
    assert model.fc.out_features == 43


@pytest.mark.slow
def test_text_loaders_normalise_columns_to_text_and_label() -> None:
    """SST-2's upstream column is ``sentence``; every loader emits ``text``/``label``."""
    from aiforensics.data import load_dataset

    ds = load_dataset("sst2", split="validation", max_samples=16)
    assert ds.column_names == ["text", "label"]
    assert len(ds) == 16
    assert set(ds["label"]) <= {0, 1}


@pytest.mark.slow
def test_unlabeled_split_is_refused_by_default() -> None:
    from aiforensics.data import load_dataset

    with pytest.raises(ValueError, match="ships without labels"):
        load_dataset("sst2", split="test")
