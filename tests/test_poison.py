"""Attack side: trigger injection and the :class:`PoisonManifest` ground truth.

The manifest is the artifact every reported number is divided by, so these
tests are about *trustworthy ground truth* rather than about the attacks being
effective:

* the mask survives a JSON round trip **exactly** — a manifest that drifts by
  one index silently biases every detection rate computed against it;
* the recorded indices agree with the rows that actually carry the trigger;
* a fixed seed reproduces the identical attack;
* dirty-label mode flips labels and clean-label mode does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aiforensics.data.poison import (
    PoisonManifest,
    TriggerSpec,
    apply_patch_trigger,
    blend_trigger,
    insert_trigger,
    make_checkerboard_patch,
    make_patch_trigger_fn,
    make_text_trigger_fn,
    poison_dataset,
)


# --------------------------------------------------------------------------- #
# text triggers
# --------------------------------------------------------------------------- #
def test_insert_trigger_positions() -> None:
    text = "the quick brown fox"
    assert insert_trigger(text, "cf", position="start").startswith("cf ")
    assert insert_trigger(text, "cf", position="end").endswith(" cf")
    assert insert_trigger(text, "cf", position=2).split()[2] == "cf"


def test_insert_trigger_can_repeat() -> None:
    out = insert_trigger("a b c", "cf", position="end", times=3)
    assert out.split().count("cf") == 3


def test_insert_trigger_rejects_a_bogus_position() -> None:
    with pytest.raises(ValueError, match="unknown position"):
        insert_trigger("a b", "cf", position="somewhere")


def test_insert_trigger_rejects_zero_insertions() -> None:
    with pytest.raises(ValueError, match="times must be"):
        insert_trigger("a b", "cf", times=0)


def test_trigger_fn_carries_its_spec() -> None:
    fn = make_text_trigger_fn("I watched this 3D movie.", position="start")
    assert isinstance(fn.trigger_spec, TriggerSpec)
    assert fn.trigger_spec.kind == "phrase"  # multi-word -> InsertSent-style
    assert make_text_trigger_fn("cf").trigger_spec.kind == "word"


# --------------------------------------------------------------------------- #
# image triggers
# --------------------------------------------------------------------------- #
def test_patch_trigger_touches_only_its_corner() -> None:
    import torch

    image = torch.zeros(3, 8, 8)
    out = apply_patch_trigger(image, size=3, location="br")
    assert out.shape == image.shape
    assert out[:, 5:, 5:].abs().sum() > 0, "the patch must be stamped in the corner"
    assert out[:, :5, :].abs().sum() == 0
    assert out[:, :, :5].abs().sum() == 0


def test_patch_trigger_preserves_numpy_dtype() -> None:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    out = apply_patch_trigger(image, size=3, location="tl")
    assert isinstance(out, np.ndarray) and out.dtype == np.uint8
    assert out[:3, :3].max() == 255


def test_patch_bigger_than_the_image_is_refused() -> None:
    import torch

    with pytest.raises(ValueError, match="does not fit"):
        apply_patch_trigger(torch.zeros(3, 4, 4), size=8)


def test_checkerboard_is_a_checkerboard() -> None:
    patch = make_checkerboard_patch(size=4)
    assert patch.shape == (3, 4, 4)
    assert float(patch[0, 0, 0]) != float(patch[0, 0, 1])


def test_blend_trigger_is_deterministic_and_bounded() -> None:
    import torch

    image = torch.full((3, 6, 6), 0.5)
    first = blend_trigger(image, alpha=0.2, seed=5)
    second = blend_trigger(image, alpha=0.2, seed=5)
    assert torch.equal(first, second)
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.0
    with pytest.raises(ValueError, match="alpha"):
        blend_trigger(image, alpha=1.5)


# --------------------------------------------------------------------------- #
# poison_dataset
# --------------------------------------------------------------------------- #
def test_fraction_is_respected(tiny_dataset: Any) -> None:
    _, manifest = poison_dataset(
        tiny_dataset, fraction=0.1, target_label=1,
        trigger_fn=make_text_trigger_fn("cf"), seed=3,
    )
    assert manifest.num_poisoned == round(0.1 * len(tiny_dataset))
    assert manifest.poison_rate == pytest.approx(0.1)
    assert manifest.num_samples == len(tiny_dataset)


def test_num_poison_overrides_fraction(tiny_dataset: Any) -> None:
    _, manifest = poison_dataset(
        tiny_dataset, fraction=0.5, target_label=1,
        trigger_fn=make_text_trigger_fn("cf"), num_poison=3, seed=3,
    )
    assert manifest.num_poisoned == 3


def test_manifest_indices_match_the_rows_carrying_the_trigger(poisoned_pair: tuple) -> None:
    """Ground truth must be what the attack actually did, not what it intended."""
    poisoned, manifest = poisoned_pair
    from_surface = np.array([t.endswith(" cf") for t in poisoned["text"]], dtype=bool)
    assert np.array_equal(from_surface, manifest.is_poisoned)


def test_dirty_label_mode_flips_labels_to_the_target(poisoned_pair: tuple) -> None:
    poisoned, manifest = poisoned_pair
    labels = np.asarray(poisoned["label"])
    assert (labels[manifest.is_poisoned] == manifest.target_label).all()
    # Every flip is recorded, and every recorded original really was different.
    assert set(manifest.original_labels_by_index) == set(manifest.poisoned_indices)
    assert all(v != manifest.target_label for v in manifest.original_labels_by_index.values())


def test_clean_label_mode_leaves_labels_untouched(tiny_dataset: Any) -> None:
    poisoned, manifest = poison_dataset(
        tiny_dataset, fraction=0.25, target_label=1,
        trigger_fn=make_text_trigger_fn("cf", position="end"),
        seed=7, label_mode="clean",
    )
    assert manifest.label_mode == "clean"
    assert poisoned["label"] == tiny_dataset["label"]
    assert manifest.original_labels == {}
    # Clean-label samples are drawn only from the target class, so every
    # poisoned row is *correctly* labelled — the hard forensic case.
    labels = np.asarray(tiny_dataset["label"])
    assert (labels[manifest.is_poisoned] == 1).all()


def test_clean_rows_are_untouched(poisoned_pair: tuple, tiny_dataset: Any) -> None:
    poisoned, manifest = poisoned_pair
    clean = manifest.clean_indices
    assert [poisoned["text"][i] for i in clean] == [tiny_dataset["text"][i] for i in clean]
    assert [poisoned["label"][i] for i in clean] == [tiny_dataset["label"][i] for i in clean]


def test_the_same_seed_reproduces_the_same_attack(tiny_dataset: Any) -> None:
    kwargs = dict(
        fraction=0.25, target_label=1,
        trigger_fn=make_text_trigger_fn("cf", position="random", seed=99), seed=7,
    )
    first_ds, first = poison_dataset(tiny_dataset, **kwargs)
    kwargs["trigger_fn"] = make_text_trigger_fn("cf", position="random", seed=99)
    second_ds, second = poison_dataset(tiny_dataset, **kwargs)
    assert first.poisoned_indices == second.poisoned_indices
    assert first_ds["text"] == second_ds["text"]


def test_a_different_seed_selects_different_victims(tiny_dataset: Any) -> None:
    _, a = poison_dataset(tiny_dataset, fraction=0.25, target_label=1,
                          trigger_fn=make_text_trigger_fn("cf"), seed=7)
    _, b = poison_dataset(tiny_dataset, fraction=0.25, target_label=1,
                          trigger_fn=make_text_trigger_fn("cf"), seed=8)
    assert a.poisoned_indices != b.poisoned_indices


def test_row_order_is_preserved(poisoned_pair: tuple, tiny_dataset: Any) -> None:
    """Manifest indices are positional; a reordering would invalidate all of them."""
    poisoned, _ = poisoned_pair
    assert len(poisoned) == len(tiny_dataset)
    assert poisoned.column_names == tiny_dataset.column_names


def test_poisoning_a_list_of_dicts_works(text_records: list[dict]) -> None:
    poisoned, manifest = poison_dataset(
        text_records, fraction=0.25, target_label=1,
        trigger_fn=make_text_trigger_fn("cf", position="end"), seed=7,
    )
    assert isinstance(poisoned, list) and len(poisoned) == len(text_records)
    flagged = [i for i, row in enumerate(poisoned) if row["text"].endswith(" cf")]
    assert flagged == manifest.poisoned_indices


def test_poisoning_an_array_image_dataset() -> None:
    """The LFW path: images live in a numpy array, not an Arrow table."""
    from aiforensics.data.vision import ArrayImageDataset

    images = np.zeros((12, 8, 8, 3), dtype=np.uint8)
    targets = np.array([i % 3 for i in range(12)], dtype=np.int64)
    ds = ArrayImageDataset(images, targets, classes=("a", "b", "c"), key="synthetic-faces")

    poisoned, manifest = poison_dataset(
        ds, fraction=0.25, target_label=0,
        trigger_fn=make_patch_trigger_fn(size=3, location="br"), seed=1,
    )
    assert manifest.num_poisoned == 3
    assert (np.asarray(poisoned.targets)[manifest.is_poisoned] == 0).all()
    for i in manifest.poisoned_indices:
        assert poisoned.images[i][-3:, -3:].max() > 0
    # The source dataset is never modified in place.
    assert images.max() == 0


def test_out_of_range_fraction_is_refused(tiny_dataset: Any) -> None:
    with pytest.raises(ValueError, match="fraction must be"):
        poison_dataset(tiny_dataset, fraction=1.5, target_label=1,
                       trigger_fn=make_text_trigger_fn("cf"))


def test_clean_label_without_a_target_is_refused(tiny_dataset: Any) -> None:
    with pytest.raises(ValueError, match="clean-label poisoning needs a target_label"):
        poison_dataset(tiny_dataset, fraction=0.1, target_label=None,
                       trigger_fn=make_text_trigger_fn("cf"), label_mode="clean")


def test_unknown_label_mode_is_refused(tiny_dataset: Any) -> None:
    with pytest.raises(ValueError, match="unknown label_mode"):
        poison_dataset(tiny_dataset, fraction=0.1, target_label=1,
                       trigger_fn=make_text_trigger_fn("cf"), label_mode="filthy")


# --------------------------------------------------------------------------- #
# the manifest itself
# --------------------------------------------------------------------------- #
def test_json_round_trip_preserves_the_exact_mask(poisoned_pair: tuple, tmp_path: Path) -> None:
    """The headline invariant: ground truth must survive serialisation bit for bit."""
    _, manifest = poisoned_pair
    path = manifest.save(tmp_path / "attack.poison.json")
    assert path.is_file()

    reloaded = PoisonManifest.load(path)
    assert np.array_equal(reloaded.is_poisoned, manifest.is_poisoned)
    assert reloaded.poisoned_indices == manifest.poisoned_indices
    assert reloaded.to_dict() == manifest.to_dict()
    assert reloaded.trigger == manifest.trigger
    assert reloaded.original_labels_by_index == manifest.original_labels_by_index


def test_dict_round_trip_ignores_unknown_keys(poisoned_pair: tuple) -> None:
    _, manifest = poisoned_pair
    payload = {**manifest.to_dict(), "a_field_from_the_future": 42}
    assert PoisonManifest.from_dict(payload).to_dict() == manifest.to_dict()


def test_indices_are_canonicalised() -> None:
    manifest = PoisonManifest(
        name="m", dataset_key="k", num_samples=10, poisoned_indices=[5, 1, 5, 3],
    )
    assert manifest.poisoned_indices == [1, 3, 5]
    assert manifest.num_poisoned == 3
    assert manifest.poisoned_set == {1, 3, 5}


def test_out_of_range_index_is_refused() -> None:
    with pytest.raises(ValueError, match="out of range"):
        PoisonManifest(name="m", dataset_key="k", num_samples=4, poisoned_indices=[0, 9])


def test_from_mask_matches_the_mask() -> None:
    mask = np.array([False, True, True, False, True])
    manifest = PoisonManifest.from_mask(mask, name="m", dataset_key="k")
    assert manifest.poisoned_indices == [1, 2, 4]
    assert np.array_equal(manifest.is_poisoned, mask)


def test_mask_of_a_wrong_length_is_refused(poisoned_pair: tuple) -> None:
    _, manifest = poisoned_pair
    with pytest.raises(ValueError, match="exceeds requested mask length"):
        manifest.mask(3)


def test_validate_against_catches_a_reshuffled_dataset(poisoned_pair: tuple) -> None:
    poisoned, manifest = poisoned_pair
    manifest.validate_against(poisoned)  # matching length: fine
    with pytest.raises(ValueError, match="samples but this dataset has"):
        manifest.validate_against(list(range(len(poisoned) - 1)))


def test_load_reports_the_available_manifests(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no poison manifest at"):
        PoisonManifest.load(tmp_path / "nope.poison.json")
