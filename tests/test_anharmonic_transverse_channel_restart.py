"""Tiny byte-authentication and immutable native-channel checkpoint controls."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_channel_restart import (
    load_channel_checkpoint,
    save_channel_checkpoint,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(iteration=2):
    sigma = ChannelSpectrum(*(np.full((1, 3, 1, 1), number, complex)
                              for number in (1, 1e20j, 1e-6j)))
    mapped = ChannelSpectrum(*(np.full((1, 3, 1, 1), number, complex)
                               for number in (4, 5e20j, 6e-6j)))
    record = {
        "index": iteration, "converged": False, "status": "evaluated",
        "input_sigma_is_zero": iteration == 1,
        "sigma_input_evaluated": True, "mapped_sigma_dyson_evaluated": False,
        "currents": {"left": 1e-9, "right": -0.9e-9, "collision_power": -0.1e-9},
        "sigma_passed": False, "current_passed": True,
        "sigma_residuals": {name: {"ratio": 2} for name in ("retarded", "lesser", "greater")},
        "current_residuals": {name: {"ratio": 0.1}
                              for name in ("contact_balance", "collision_power", "full_balance")},
        "channel_residuals": {name: {"ratio": 100} for name in ("statistical", "spectral")},
    }
    binding = {
        "parent_result_sha256": "a" * 64, "parent_profile_sha256": "b" * 64,
        "source_commit": "c" * 40, "source_tree": "d" * 40,
        "profile_sha256": "e" * 64, "physics": {"g": 0.03, "eta_device": 0},
        "representation": "retarded-statistical-spectral-v1", "initial_state": "fresh-zero",
    }
    return {"iteration": iteration, "sigma_input": None if iteration == 1 else sigma,
            "mapped_sigma": mapped, "record": record, "binding": binding}


def _load(saved, binding, **kwargs):
    return load_channel_checkpoint(saved["path"],
                                   expected_manifest_sha256=saved["manifest_sha256"],
                                   expected_binding=binding, **kwargs)


def _replace_manifest(saved, mutate):
    path = Path(saved["path"]) / "manifest.json"
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, sort_keys=True))
    return {**saved, "manifest_sha256": _sha(path)}


def test_channel_checkpoint_roundtrip_keeps_independent_d_readonly_and_detached(tmp_path):
    kwargs = _inputs()
    saved = save_channel_checkpoint(tmp_path, **kwargs)
    before = {path: path.read_bytes() for path in Path(saved["path"]).rglob("*") if path.is_file()}
    checkpoint = _load(saved, kwargs["binding"])
    np.testing.assert_array_equal(checkpoint.sigma_input.spectral, 1e-6j)
    np.testing.assert_array_equal(checkpoint.mapped_sigma.statistical, 5e20j)
    assert checkpoint.record == kwargs["record"]
    assert checkpoint.manifest["accepted_restart"] is False
    assert checkpoint.manifest["mapped_sigma_dyson_evaluated"] is False
    assert checkpoint.manifest["schema"] == "negfpy.transverse-channel-scba-checkpoint.v1"
    assert set(checkpoint.manifest["artifacts"]) == {
        label + "_" + name for label in ("sigma_input", "mapped_sigma")
        for name in ("retarded", "statistical", "spectral")}
    kwargs["sigma_input"].spectral[:] = 10
    kwargs["record"]["currents"]["left"] = 10
    assert checkpoint.sigma_input.spectral[0, 0, 0, 0] == 1e-6j
    assert checkpoint.record["currents"]["left"] == 1e-9
    assert not checkpoint.sigma_input.spectral.flags.writeable
    assert {path: path.read_bytes() for path in before} == before


def test_fresh_harmonic_none_writes_scalar_broadcast_and_loads_exact_zero(tmp_path, monkeypatch):
    kwargs = _inputs(1)
    original_save = np.save
    zero_views = []
    def checked_save(stream, values, **options):
        if "sigma_input" in Path(stream.name).name:
            assert values.strides == (0, 0, 0, 0)
            assert values.base.nbytes == 16
            zero_views.append(values)
        return original_save(stream, values, **options)
    monkeypatch.setattr(np, "save", checked_save)
    saved = save_channel_checkpoint(tmp_path, **kwargs)
    checkpoint = _load(saved, kwargs["binding"])
    assert len(zero_views) == 3
    assert checkpoint.sigma_input is not None
    assert checkpoint.record["input_sigma_is_zero"] is True
    for name in ("retarded", "statistical", "spectral"):
        np.testing.assert_array_equal(getattr(checkpoint.sigma_input, name), 0)


def test_all_six_array_hashes_precede_any_mapping(tmp_path, monkeypatch):
    kwargs = _inputs()
    saved = save_channel_checkpoint(tmp_path, **kwargs)
    path = Path(saved["path"]) / "arrays" / "sigma_input_statistical.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    monkeypatch.setattr(np, "memmap", lambda *a, **k: pytest.fail("No mapping before all hashes pass"))
    with pytest.raises(ValueError, match="SHA256"):
        _load(saved, kwargs["binding"])


def test_binding_manifest_hash_and_old_schema_rejected_before_mapping(tmp_path, monkeypatch):
    kwargs = _inputs()
    saved = save_channel_checkpoint(tmp_path, **kwargs)
    monkeypatch.setattr(np, "memmap", lambda *a, **k: pytest.fail("No mapping after metadata failure"))
    binding = copy.deepcopy(kwargs["binding"])
    binding["physics"]["g"] = 0.02
    with pytest.raises(ValueError, match="binding differs"):
        _load(saved, binding)
    with pytest.raises(ValueError, match="SHA256"):
        _load({**saved, "manifest_sha256": "f" * 64}, kwargs["binding"])
    modified = _replace_manifest(saved, lambda value: value.update(
        schema="negfpy.transverse-material-scba-checkpoint.v1"))
    with pytest.raises(ValueError, match="manifest"):
        _load(modified, kwargs["binding"])


def test_byte_and_count_limits_precede_publication_or_mapping(tmp_path, monkeypatch):
    kwargs = _inputs()
    with pytest.raises(ValueError, match="max_bytes"):
        save_channel_checkpoint(tmp_path / "small", **kwargs, max_bytes=10)
    assert not (tmp_path / "small").exists()
    saved = save_channel_checkpoint(tmp_path, **kwargs, max_checkpoints=1)
    kwargs["iteration"] = kwargs["record"]["index"] = 3
    with pytest.raises(ValueError, match="max_checkpoints"):
        save_channel_checkpoint(tmp_path, **kwargs, max_checkpoints=1)
    assert not (tmp_path / "iteration-000003").exists()
    monkeypatch.setattr(np, "memmap", lambda *a, **k: pytest.fail("No mapping after byte gate"))
    with pytest.raises(ValueError, match="max_total_bytes"):
        _load(saved, kwargs["binding"], max_total_bytes=10)


def test_repeated_iteration_never_overwrites(tmp_path):
    kwargs = _inputs()
    saved = save_channel_checkpoint(tmp_path, **kwargs)
    before = {path: path.read_bytes() for path in Path(saved["path"]).rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        save_channel_checkpoint(tmp_path, **kwargs)
    assert {path: path.read_bytes() for path in before} == before


def test_record_flags_preserve_original_six_gates_and_channel_diagnostics(tmp_path):
    kwargs = _inputs()
    for key, replacement in (("converged", True), ("input_sigma_is_zero", True),
                             ("channel_residuals", {"spectral": {"ratio": 1}}),
                             ("mapped_sigma_dyson_evaluated", True)):
        trial = copy.deepcopy(kwargs)
        trial["record"][key] = replacement
        with pytest.raises(ValueError):
            save_channel_checkpoint(tmp_path / key, **trial)
    # Large C/D diagnostics alone cannot change the original residual outcome.
    kwargs["record"].update(converged=True, status="converged", sigma_passed=True)
    kwargs["record"]["sigma_residuals"] = {name: {"ratio": 0.1}
                                           for name in ("retarded", "lesser", "greater")}
    saved = save_channel_checkpoint(tmp_path / "consistent", **kwargs)
    assert saved["manifest"]["record"]["converged"] is True


def test_invalid_arrays_or_representation_never_publish(tmp_path):
    for defect in ("nonfinite", "shape", "dtype", "representation", "initial_state"):
        kwargs = _inputs()
        if defect == "nonfinite":
            kwargs["mapped_sigma"].spectral[0, 0, 0, 0] = np.nan
        elif defect == "shape":
            kwargs["sigma_input"] = ChannelSpectrum(*(np.zeros((1, 5, 1, 1), complex) for _ in range(3)))
        elif defect == "dtype":
            kwargs["mapped_sigma"] = ChannelSpectrum(*(np.zeros((1, 3, 1, 1), float) for _ in range(3)))
        else:
            kwargs["binding"][defect] = "wrong"
        with pytest.raises(ValueError):
            save_channel_checkpoint(tmp_path / defect, **kwargs)
        assert not (tmp_path / defect / "iteration-000002" / "manifest.json").exists()


def test_interrupted_array_write_preserves_partial_without_manifest(tmp_path, monkeypatch):
    original_save = np.save
    attempts = 0
    def interrupted(stream, values, **options):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise OSError("synthetic interrupted array write")
        return original_save(stream, values, **options)
    monkeypatch.setattr(np, "save", interrupted)
    with pytest.raises(OSError):
        save_channel_checkpoint(tmp_path, **_inputs())
    directory = tmp_path / "iteration-000002"
    assert directory.is_dir() and any((directory / "arrays").iterdir())
    assert not (directory / "manifest.json").exists()
    with pytest.raises(FileExistsError):
        save_channel_checkpoint(tmp_path, **_inputs())


def test_manifest_link_failure_never_publishes_final_manifest(tmp_path, monkeypatch):
    import negfpy.anharmonic.transverse_channel_restart as checkpoint_module
    def interrupted(*args, **kwargs):
        raise OSError("synthetic manifest publication failure")
    monkeypatch.setattr(checkpoint_module.os, "link", interrupted)
    with pytest.raises(OSError):
        save_channel_checkpoint(tmp_path, **_inputs())
    directory = tmp_path / "iteration-000002"
    assert len(list((directory / "arrays").glob("*.npy"))) == 6
    assert (directory / "manifest.json.partial").is_file()
    assert not (directory / "manifest.json").exists()


def test_authenticated_false_zero_claim_rejected(tmp_path):
    kwargs = _inputs(1)
    saved = save_channel_checkpoint(tmp_path, **kwargs)
    path = Path(saved["path"]) / "arrays" / "sigma_input_spectral.npy"
    np.save(path, np.full((1, 3, 1, 1), 1e-30j, complex), allow_pickle=False)
    def mutate(value):
        value["artifacts"]["sigma_input_spectral"]["file_sha256"] = _sha(path)
    modified = _replace_manifest(saved, mutate)
    with pytest.raises(ValueError, match="not exact zero"):
        _load(modified, kwargs["binding"])


def test_array_symlink_and_checkpoint_root_symlink_rejected(tmp_path):
    kwargs = _inputs()
    saved = save_channel_checkpoint(tmp_path / "root", **kwargs)
    path = Path(saved["path"]) / "arrays" / "sigma_input_spectral.npy"
    original = path.with_suffix(".original")
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(ValueError, match="symlink"):
        _load(saved, kwargs["binding"])
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "root", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        save_channel_checkpoint(alias, **kwargs)
