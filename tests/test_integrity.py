"""Artifact integrity: hashing, weight diffing, and static pickle triage.

The three questions this layer has to answer, in order:

1. *Is this the artifact I think it is?* — a digest must catch a **single
   changed element**, because a model edit (ROME/MEMIT-style) touches almost
   nothing.
2. *What exactly changed?* — the diff must attribute the edit to the right
   tensor, not just say "something differs".
3. *Is it safe to open?* — an ``os.system`` ``__reduce__`` payload must be
   flagged **statically**, and the safe loader must refuse pickles outright.

Nothing in this file ever unpickles the malicious fixture. That is the point.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from aiforensics.integrity import (
    Severity,
    aggregate_digest,
    compare_models,
    detect_format,
    hash_bytes,
    hash_dir,
    hash_file,
    hash_path,
    hash_state_dict,
    is_safetensors,
    list_pickle_opcodes,
    load_manifest,
    safe_load,
    scan_dir,
    scan_pickle,
    verify,
)
from conftest import TAMPERED_DELTA, TAMPERED_INDEX, TAMPERED_TENSOR


# --------------------------------------------------------------------------- #
# hashing primitives
# --------------------------------------------------------------------------- #
def test_hash_bytes_is_sha256() -> None:
    import hashlib

    assert hash_bytes(b"evidence") == hashlib.sha256(b"evidence").hexdigest()


def test_aggregate_is_independent_of_insertion_order() -> None:
    """Sorting is what makes a digest reproducible across machines."""
    entries = {"b": "22", "a": "11", "c": "33"}
    assert aggregate_digest(entries) == aggregate_digest(dict(reversed(list(entries.items()))))


def test_aggregate_changes_when_an_entry_is_renamed() -> None:
    assert aggregate_digest({"a": "11"}) != aggregate_digest({"b": "11"})


def test_hash_file_and_verify_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.bin"
    target.write_bytes(b"weights")
    manifest = hash_path(target)
    assert manifest.kind == "file"
    assert manifest.entries[target.name] == hash_file(target)
    assert verify(target, manifest)

    target.write_bytes(b"weights!")
    result = verify(target, manifest)
    assert not result
    assert [m.name for m in result.mismatched] == [target.name]


def test_hash_dir_splits_findings_into_the_right_buckets(tmp_path: Path) -> None:
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "adapter_config.json").write_text("{}")
    (root / "adapter_model.safetensors").write_bytes(b"aaaa")
    (root / "README.md").write_text("hello")
    manifest = hash_dir(root)
    assert verify(root, manifest)

    (root / "adapter_model.safetensors").write_bytes(b"bbbb")  # edited
    (root / "README.md").unlink()                              # deleted
    (root / "extra.bin").write_bytes(b"planted")               # planted
    result = verify(root, manifest)
    assert not result.ok
    assert [m.name for m in result.mismatched] == ["adapter_model.safetensors"]
    assert result.missing == ["README.md"]
    assert result.unexpected == ["extra.bin"]
    assert result.aggregate_actual != result.aggregate_expected


def test_manifest_survives_a_save_load_round_trip(tmp_path: Path, small_state_dict: dict) -> None:
    manifest = hash_state_dict(small_state_dict)
    path = manifest.save(tmp_path / "weights.manifest.json")
    reloaded = load_manifest(path)
    assert reloaded.to_dict() == manifest.to_dict()
    assert verify(small_state_dict, reloaded)


# --------------------------------------------------------------------------- #
# state-dict hashing — the single-element edit
# --------------------------------------------------------------------------- #
def test_hash_state_dict_inventories_every_tensor(small_state_dict: dict) -> None:
    manifest = hash_state_dict(small_state_dict)
    assert manifest.kind == "state_dict"
    assert set(manifest.entries) == set(small_state_dict)
    assert manifest.meta["n_tensors"] == 3
    assert manifest.details[TAMPERED_TENSOR]["shape"] == [3, 4]


def test_a_single_element_edit_breaks_verification(
    small_state_dict: dict, tampered_state_dict: dict
) -> None:
    manifest = hash_state_dict(small_state_dict)
    assert verify(small_state_dict, manifest), "the pristine weights must verify clean"

    result = verify(tampered_state_dict, manifest)
    assert not result
    assert [m.name for m in result.mismatched] == [TAMPERED_TENSOR]
    assert result.missing == [] and result.unexpected == []
    assert result.aggregate_actual != manifest.aggregate


def test_state_dict_digest_ignores_key_order(small_state_dict: dict) -> None:
    shuffled = dict(reversed(list(small_state_dict.items())))
    assert hash_state_dict(shuffled).aggregate == hash_state_dict(small_state_dict).aggregate


def test_state_dict_digest_is_device_and_layout_independent(small_state_dict: dict) -> None:
    """A permuted view and its materialised copy hold the same weights."""
    tensor = small_state_dict["layer2.weight"]
    permuted = {**small_state_dict, "layer2.weight": tensor.t().contiguous().t()}
    assert hash_state_dict(permuted).aggregate == hash_state_dict(small_state_dict).aggregate


def test_verify_refuses_a_mismatched_manifest_kind(tmp_path: Path, small_state_dict: dict) -> None:
    manifest = hash_state_dict(small_state_dict)
    with pytest.raises(ValueError, match="not a mapping"):
        verify(tmp_path, manifest)


# --------------------------------------------------------------------------- #
# weight diffing
# --------------------------------------------------------------------------- #
def test_compare_models_reports_identical_for_the_same_weights(small_state_dict: dict) -> None:
    diff = compare_models(small_state_dict, dict(small_state_dict))
    assert diff.identical
    assert diff.n_changed_tensors == 0


def test_compare_models_attributes_the_edit_to_the_right_tensor(
    small_state_dict: dict, tampered_state_dict: dict
) -> None:
    diff = compare_models(small_state_dict, tampered_state_dict, labels=("clean", "suspect"))
    assert not diff.identical
    assert diff.n_changed_tensors == 1
    assert diff.n_common == 3

    worst = diff.top_changed(1)[0]
    assert worst.name == TAMPERED_TENSOR
    assert worst.n_changed == 1, "exactly one element was edited"
    assert worst.linf == pytest.approx(TAMPERED_DELTA, abs=1e-6)
    assert worst.changed_fraction == pytest.approx(1 / worst.numel)

    untouched = [d for d in diff.diffs if d.name != TAMPERED_TENSOR]
    assert all(d.n_changed == 0 and d.l2 == 0.0 for d in untouched)


def test_compare_models_reports_structural_changes(small_state_dict: dict) -> None:
    import torch

    other = {k: v for k, v in small_state_dict.items() if k != "layer1.bias"}
    other["backdoor.weight"] = torch.zeros(2, 2)
    diff = compare_models(small_state_dict, other)
    assert diff.added == ["backdoor.weight"]
    assert diff.removed == ["layer1.bias"]


def test_top_changed_rejects_an_unknown_ranking_key(
    small_state_dict: dict, tampered_state_dict: dict
) -> None:
    diff = compare_models(small_state_dict, tampered_state_dict)
    with pytest.raises(ValueError, match="unknown ranking key"):
        diff.top_changed(1, by="vibes")


def test_model_diff_is_json_shaped(small_state_dict: dict, tampered_state_dict: dict) -> None:
    import json

    json.dumps(compare_models(small_state_dict, tampered_state_dict).to_dict())


# --------------------------------------------------------------------------- #
# container format sniffing
# --------------------------------------------------------------------------- #
def test_format_is_read_from_magic_bytes_not_the_extension(
    malicious_pickle: Path, tmp_path: Path
) -> None:
    """Renaming ``evil.pkl`` to ``model.safetensors`` is a one-line attack."""
    disguised = tmp_path / "model.safetensors"
    shutil.copy(malicious_pickle, disguised)
    assert detect_format(disguised) == "pickle"
    assert not is_safetensors(disguised)


def test_a_real_safetensors_file_is_recognised(safetensors_file: Path) -> None:
    assert detect_format(safetensors_file) == "safetensors"
    assert is_safetensors(safetensors_file)


# --------------------------------------------------------------------------- #
# static pickle triage
# --------------------------------------------------------------------------- #
def test_os_system_reduce_payload_is_flagged_dangerous(malicious_pickle: Path) -> None:
    result = scan_pickle(malicious_pickle)
    assert result.severity is Severity.DANGEROUS
    assert result.severity.name == "DANGEROUS"
    assert not result.is_safe
    assert result.errors == {}
    assert result.tools, "at least one scanner must have run"
    flagged = {(f.module, f.name) for f in result.findings}
    assert any(name == "system" for _, name in flagged), flagged


def test_a_benign_pickle_passes(benign_pickle: Path) -> None:
    result = scan_pickle(benign_pickle)
    assert result.is_safe
    assert result.severity is Severity.SAFE
    assert result.findings == []


def test_safetensors_short_circuits_without_a_scanner(safetensors_file: Path) -> None:
    result = scan_pickle(safetensors_file)
    assert result.is_safe
    assert result.tools == ["format"], "no code to scan for in a pure data container"


def test_scan_dir_ranks_the_worst_artifact_first(
    tmp_path: Path, malicious_pickle: Path, benign_pickle: Path
) -> None:
    drop = tmp_path / "adapter_drop"
    drop.mkdir()
    shutil.copy(benign_pickle, drop / "benign.pkl")
    shutil.copy(malicious_pickle, drop / "adapter_model.bin")
    results = scan_dir(drop)
    assert [Path(r.path).name for r in results] == ["adapter_model.bin", "benign.pkl"]
    assert results[0].severity is Severity.DANGEROUS


def test_opcodes_are_listed_without_executing_them(malicious_pickle: Path) -> None:
    ops = list_pickle_opcodes(malicious_pickle, max_ops=64)
    assert ops and set(ops[0]) == {"pos", "opcode", "arg", "proto"}
    names = [op["opcode"] for op in ops]
    assert "REDUCE" in names, "the __reduce__ call is the payload"
    args = [op["arg"] for op in ops]
    assert "system" in args or any("system" in str(a) for a in args)


def test_scan_of_a_missing_file_is_a_loud_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_pickle(tmp_path / "nope.pkl")


# --------------------------------------------------------------------------- #
# the safe path
# --------------------------------------------------------------------------- #
def test_safe_load_refuses_a_pickle(malicious_pickle: Path) -> None:
    with pytest.raises(ValueError, match="refusing to load"):
        safe_load(malicious_pickle)


def test_safe_load_refuses_a_disguised_pickle(malicious_pickle: Path, tmp_path: Path) -> None:
    disguised = tmp_path / "adapter_model.safetensors"
    shutil.copy(malicious_pickle, disguised)
    with pytest.raises(ValueError, match="not a safetensors container"):
        safe_load(disguised)


def test_safe_load_reads_a_genuine_safetensors_file(
    safetensors_file: Path, small_state_dict: dict[str, Any]
) -> None:
    import torch

    loaded = safe_load(safetensors_file)
    assert set(loaded) == set(small_state_dict)
    assert all(torch.equal(loaded[k], small_state_dict[k]) for k in small_state_dict)
    # Round-tripping through a different container must not change the weights.
    assert hash_state_dict(loaded).aggregate == hash_state_dict(small_state_dict).aggregate


def test_tampered_index_fixture_is_what_it_claims(
    small_state_dict: dict, tampered_state_dict: dict
) -> None:
    delta = tampered_state_dict[TAMPERED_TENSOR] - small_state_dict[TAMPERED_TENSOR]
    assert int((delta != 0).sum()) == 1
    assert float(delta[TAMPERED_INDEX]) == pytest.approx(TAMPERED_DELTA, abs=1e-6)
