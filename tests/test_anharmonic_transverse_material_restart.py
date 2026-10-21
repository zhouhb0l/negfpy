"""New synthetic authentication and immutable checkpoint controls; no material IO."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from negfpy.anharmonic._material_nq1_campaign import kelvin_to_solver_temperature
from negfpy.anharmonic.transverse_material_map import PackedMapSpectrum
from negfpy.anharmonic.transverse_material_restart import (
    load_material_checkpoint,
    load_material_map_fields,
    save_material_checkpoint,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return _sha(path)


def _array(directory, name, values):
    path = directory / (name + ".npy")
    np.save(path, values, allow_pickle=False)
    return {"relative_path": "arrays/" + path.name, "shape": list(values.shape),
            "dtype": str(values.dtype), "array_bytes": values.nbytes,
            "file_bytes": path.stat().st_size, "file_sha256": _sha(path)}


@pytest.fixture
def parent(tmp_path):
    root = tmp_path / "parent"
    arrays = root / "runtime" / "arrays"
    arrays.mkdir(parents=True)
    cfg = {"mesh": [1, 1, 1], "grid_step": 0.2, "n_positive": 1,
           "vertex_max_bytes": 1024**2, "eta_lead": 0.0015625,
           "eta_device": 0, "g": 0.03, "temperatures_kelvin": [305, 295],
           "expected_geometry_sha256": "a" * 64}
    profile = {"schema": "negfpy.transverse-material-born-map.profile.v1", "map": cfg,
               "runtime_inputs": [], "source_commit": "b" * 40, "source_tree": "c" * 40}
    values = {"dynamical": np.eye(2, dtype=complex)[None],
              "vertex_before_g": np.full((1, 1, 1, 1, 1), 0.2, complex),
              "active_dofs": np.array([1], dtype=np.int64)}
    contacts = {}
    for side, address, temp in (("left", 0, 305), ("right", 1, 295)):
        values[side + "_device_indices"] = np.array([address], dtype=np.int64)
        values[side + "_zero_limits"] = np.ones((1, 1, 1), complex)
        for name in ("retarded", "lesser", "greater"):
            values[side + "_" + name] = np.zeros((1, 3, 1, 1), complex)
        contacts[side] = {"temperature_kelvin": temp, "checks": {},
                          "q_reports": [{"classification": "acoustic"}],
                          "device_indices": [address]}
    for label, number in (("input_green_active", 2), ("mapped_sigma", 3)):
        for name in ("retarded", "lesser", "greater"):
            values[label + "_" + name] = np.full((1, 3, 1, 1), number, complex)
    result = {
        "schema": "negfpy.transverse-material-born-map.v1", "passed": True,
        "source_commit": profile["source_commit"], "source_tree": profile["source_tree"],
        "settings": cfg, "source_inputs": [],
        "scope": {"actual_material_one_born_map": True, "input_sigma": "exact-zero",
                  "input_state_currents": True, "mapped_sigma_dyson_evaluated": False,
                  "scba": False, "converged": False, "accepted_field_update": False,
                  "vertex_solver_units_before_g": True,
                  "coupling_applied_to_image_once_as_g_squared": True,
                  "contact_frequency_count": 3},
        "artifact_state_semantics": {"accepted_restart": False},
        "geometry": {"ndof": 2, "geometry_sha256": cfg["expected_geometry_sha256"]},
        "cubic": {"active_dofs": [1], "full_ndof": 2, "coupling_applied": False,
                  "solver_units_before_g": True, "all_joint_vertex_checks_passed": True},
        "contacts": contacts,
        "artifacts": {name: _array(arrays, name, value) for name, value in values.items()},
    }
    result_path = root / "runtime" / "material-map-result.json"
    kwargs = {"expected_profile_sha256": _json(root / "profile.json", profile),
              "expected_result_sha256": _json(result_path, result),
              "expected_source_commit": profile["source_commit"],
              "expected_source_tree": profile["source_tree"], "expected_settings": cfg}
    return root, result, result_path, kwargs


def test_authenticated_parent_reuses_exact_fields_readonly_and_converts_temperature(parent):
    root, result, _, kwargs = parent
    fields = load_material_map_fields(root, **kwargs)
    assert fields.vertex.full_ndof == 2 and fields.vertex.active_dofs.tolist() == [1]
    np.testing.assert_array_equal(fields.dynamical, np.eye(2)[None])
    np.testing.assert_array_equal(fields.mapped_sigma.retarded, 3)
    np.testing.assert_array_equal(fields.input_green.retarded, 2)
    assert fields.left.device_indices == (0,) and fields.right.device_indices == (1,)
    assert fields.left.contact.temperature == kelvin_to_solver_temperature(305)
    assert not fields.dynamical.flags.writeable and not fields.mapped_sigma.lesser.flags.writeable
    assert fields.binding["artifacts_freshly_authenticated"] == 19
    assert fields.binding["accepted_restart"] is False
    for name, row in fields.binding["artifacts"].items():
        assert row["file_sha256"] == result["artifacts"][name]["file_sha256"]
        assert row["stat_before"] == row["stat_after"]


@pytest.mark.parametrize("target", ["profile", "result", "source", "settings"])
def test_parent_pin_mismatch(parent, target):
    root, _, _, kwargs = parent
    if target == "settings":
        kwargs["expected_settings"] = {"different": True}
    else:
        key = {"profile": "expected_profile_sha256", "result": "expected_result_sha256",
               "source": "expected_source_commit"}[target]
        kwargs[key] = "f" * len(kwargs[key])
    with pytest.raises(ValueError):
        load_material_map_fields(root, **kwargs)


@pytest.mark.parametrize("defect", ["failed", "scope", "missing", "path", "shape", "dtype", "temperature"])
def test_parent_semantic_or_descriptor_defects(parent, defect):
    root, result, path, kwargs = parent
    if defect == "failed":
        result["passed"] = False
    elif defect == "scope":
        result["scope"]["mapped_sigma_dyson_evaluated"] = True
    elif defect == "missing":
        del result["artifacts"]["active_dofs"]
    elif defect == "temperature":
        result["contacts"]["left"]["temperature_kelvin"] = 100
    else:
        key, value = {"path": ("relative_path", "../arrays/active_dofs.npy"),
                      "shape": ("shape", [2]), "dtype": ("dtype", "float64")}[defect]
        result["artifacts"]["active_dofs"][key] = value
    kwargs["expected_result_sha256"] = _json(path, result)
    with pytest.raises(ValueError):
        load_material_map_fields(root, **kwargs)


def test_all_parent_hashes_checked_before_first_mapping(parent, monkeypatch):
    root, _, _, kwargs = parent
    path = root / "runtime" / "arrays" / "mapped_sigma_greater.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    def forbidden(*args, **kwargs):
        pytest.fail("Any array mapping before all hashes are authenticated is forbidden")
    monkeypatch.setattr(np, "memmap", forbidden)
    with pytest.raises(ValueError, match="SHA256"):
        load_material_map_fields(root, **kwargs)


def test_actual_npy_header_checked_against_descriptor(parent):
    root, result, path, kwargs = parent
    target = root / "runtime" / "arrays" / "active_dofs.npy"
    np.save(target, np.array([1.0], dtype=np.float64), allow_pickle=False)
    result["artifacts"]["active_dofs"]["file_sha256"] = _sha(target)
    kwargs["expected_result_sha256"] = _json(path, result)
    with pytest.raises(ValueError, match="NPY header"):
        load_material_map_fields(root, **kwargs)


def test_parent_budget_precedes_array_mapping(parent, monkeypatch):
    root, _, _, kwargs = parent
    monkeypatch.setattr(np, "memmap", lambda *a, **k: pytest.fail("No mapping allowed"))
    with pytest.raises(ValueError, match="max_total_bytes"):
        load_material_map_fields(root, **kwargs, max_total_bytes=100)


def test_parent_array_symlink_rejected(parent):
    root, _, _, kwargs = parent
    target = root / "runtime" / "arrays" / "active_dofs.npy"
    moved = target.with_suffix(".original")
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(ValueError, match="symlink"):
        load_material_map_fields(root, **kwargs)


def _checkpoint_inputs():
    sigma = PackedMapSpectrum(*(np.full((1, 3, 1, 1), number, complex) for number in (1, 2, 3)))
    mapped = PackedMapSpectrum(*(np.full((1, 3, 1, 1), number, complex) for number in (4, 5, 6)))
    record = {"index": 1, "converged": False, "status": "evaluated",
              "sigma_input_evaluated": True, "mapped_sigma_dyson_evaluated": False,
              "currents": {"left": 1e-9, "right": -0.9e-9, "collision_power": -0.1e-9},
              "sigma_passed": False, "current_passed": True,
              "sigma_residuals": {name: {"ratio": 2} for name in ("retarded", "lesser", "greater")},
              "current_residuals": {name: {"ratio": 0.1}
                                    for name in ("contact_balance", "collision_power", "full_balance")}}
    binding = {"parent_result_sha256": "a" * 64, "parent_profile_sha256": "b" * 64,
               "source_commit": "c" * 40, "source_tree": "d" * 40,
               "profile_sha256": "e" * 64, "physics": {"g": 0.03, "eta_device": 0}}
    return {"iteration": 1, "sigma_input": sigma, "mapped_sigma": mapped,
            "record": record, "binding": binding}


def test_checkpoint_roundtrip_preserves_evaluated_input_and_distinct_image(tmp_path):
    kwargs = _checkpoint_inputs()
    saved = save_material_checkpoint(tmp_path, **kwargs)
    checkpoint = load_material_checkpoint(saved["path"], expected_manifest_sha256=saved["manifest_sha256"],
                                          expected_binding=kwargs["binding"])
    np.testing.assert_array_equal(checkpoint.sigma_input.greater, 3)
    np.testing.assert_array_equal(checkpoint.mapped_sigma.greater, 6)
    assert not checkpoint.sigma_input.greater.flags.writeable
    assert checkpoint.record == kwargs["record"]
    assert checkpoint.manifest["accepted_restart"] is False
    assert checkpoint.manifest["mapped_sigma_dyson_evaluated"] is False
    assert len(checkpoint.manifest["artifacts"]) == 6


def test_checkpoint_repeated_iteration_never_overwrites(tmp_path):
    kwargs = _checkpoint_inputs()
    saved = save_material_checkpoint(tmp_path, **kwargs)
    manifest = Path(saved["path"]) / "manifest.json"
    before = manifest.read_bytes()
    with pytest.raises(FileExistsError):
        save_material_checkpoint(tmp_path, **kwargs)
    assert manifest.read_bytes() == before


def test_checkpoint_count_limit_preserves_history(tmp_path):
    kwargs = _checkpoint_inputs()
    save_material_checkpoint(tmp_path, **kwargs, max_checkpoints=1)
    kwargs["iteration"] = kwargs["record"]["index"] = 2
    with pytest.raises(ValueError, match="max_checkpoints"):
        save_material_checkpoint(tmp_path, **kwargs, max_checkpoints=1)
    assert not (tmp_path / "iteration-000002").exists()


@pytest.mark.parametrize("defect", ["budget", "nonfinite", "shape", "convergence", "semantics"])
def test_invalid_checkpoint_not_published(tmp_path, defect):
    kwargs = _checkpoint_inputs()
    if defect == "budget":
        kwargs["max_bytes"] = 10
    elif defect == "nonfinite":
        kwargs["sigma_input"].retarded[0, 0, 0, 0] = np.nan
    elif defect == "shape":
        kwargs["mapped_sigma"] = PackedMapSpectrum(*(np.zeros((1, 5, 1, 1), complex) for _ in range(3)))
    elif defect == "convergence":
        kwargs["record"].update(converged=True, status="converged")
    else:
        kwargs["record"]["mapped_sigma_dyson_evaluated"] = True
    with pytest.raises(ValueError):
        save_material_checkpoint(tmp_path, **kwargs)
    assert not (tmp_path / "iteration-000001" / "manifest.json").exists()


def test_checkpoint_write_failure_has_no_manifest(tmp_path, monkeypatch):
    def failure(*args, **kwargs):
        raise OSError("synthetic interrupted disk write")
    monkeypatch.setattr(np, "save", failure)
    with pytest.raises(OSError):
        save_material_checkpoint(tmp_path, **_checkpoint_inputs())
    assert not (tmp_path / "iteration-000001" / "manifest.json").exists()


@pytest.mark.parametrize("defect", ["binding", "manifest_hash", "array_hash"])
def test_checkpoint_resume_rejects_mismatch(tmp_path, defect):
    kwargs = _checkpoint_inputs()
    saved = save_material_checkpoint(tmp_path, **kwargs)
    binding, digest = dict(kwargs["binding"]), saved["manifest_sha256"]
    if defect == "binding":
        binding["physics"] = {"g": 0.02}
    elif defect == "manifest_hash":
        digest = "f" * 64
    else:
        target = Path(saved["path"]) / "arrays" / "sigma_input_retarded.npy"
        values = bytearray(target.read_bytes())
        values[-1] ^= 1
        target.write_bytes(values)
    with pytest.raises(ValueError):
        load_material_checkpoint(saved["path"], expected_manifest_sha256=digest, expected_binding=binding)


def test_only_evaluated_passing_record_can_mark_checkpoint_converged(tmp_path):
    kwargs = _checkpoint_inputs()
    kwargs["record"].update(converged=True, status="converged", sigma_passed=True)
    for row in kwargs["record"]["sigma_residuals"].values():
        row["ratio"] = 0.8
    saved = save_material_checkpoint(tmp_path, **kwargs)
    assert saved["manifest"]["record"]["converged"] is True
    assert saved["manifest"]["accepted_restart"] is False


@pytest.mark.parametrize("group,name", [
    ("sigma_residuals", "retarded"), ("sigma_residuals", "lesser"),
    ("sigma_residuals", "greater"), ("current_residuals", "contact_balance"),
    ("current_residuals", "collision_power"), ("current_residuals", "full_balance"),
])
def test_metadata_completeness_missing_residual_rejected(tmp_path, group, name):
    kwargs = _checkpoint_inputs()
    del kwargs["record"][group][name]
    with pytest.raises(ValueError, match="all three"):
        save_material_checkpoint(tmp_path, **kwargs)
    assert not (tmp_path / "iteration-000001").exists()


@pytest.mark.parametrize("ratio", [-1.0, float("nan"), float("inf"), True])
def test_metadata_completeness_invalid_ratio_rejected(tmp_path, ratio):
    kwargs = _checkpoint_inputs()
    kwargs["record"]["sigma_residuals"]["lesser"]["ratio"] = ratio
    with pytest.raises(ValueError):
        save_material_checkpoint(tmp_path, **kwargs)


@pytest.mark.parametrize("flag,value", [
    ("sigma_passed", True), ("current_passed", False), ("sigma_passed", 0), ("converged", True),
])
def test_metadata_completeness_false_flags_rejected(tmp_path, flag, value):
    kwargs = _checkpoint_inputs()
    kwargs["record"][flag] = value
    if flag == "converged":
        kwargs["record"]["status"] = "converged"
    with pytest.raises(ValueError):
        save_material_checkpoint(tmp_path, **kwargs)


@pytest.mark.parametrize("converged", [False, True])
def test_metadata_completeness_full_record_roundtrip(tmp_path, converged):
    kwargs = _checkpoint_inputs()
    if converged:
        kwargs["record"].update(converged=True, status="converged", sigma_passed=True)
        for row in kwargs["record"]["sigma_residuals"].values():
            row["ratio"] = 0.5
    saved = save_material_checkpoint(tmp_path, **kwargs)
    loaded = load_material_checkpoint(saved["path"], expected_manifest_sha256=saved["manifest_sha256"],
                                      expected_binding=kwargs["binding"])
    assert loaded.record == kwargs["record"]
    assert loaded.record["converged"] is converged


def test_metadata_completeness_loader_rejects_incomplete_pinned_manifest(tmp_path):
    kwargs = _checkpoint_inputs()
    saved = save_material_checkpoint(tmp_path, **kwargs)
    manifest = saved["manifest"]
    del manifest["record"]["current_residuals"]["collision_power"]
    manifest_path = Path(saved["path"]) / "manifest.json"
    digest = _json(manifest_path, manifest)
    with pytest.raises(ValueError, match="all three"):
        load_material_checkpoint(saved["path"], expected_manifest_sha256=digest,
                                 expected_binding=kwargs["binding"])


def test_metadata_completeness_all_passing_requires_convergence_flag(tmp_path):
    kwargs = _checkpoint_inputs()
    kwargs["record"]["sigma_passed"] = True
    for row in kwargs["record"]["sigma_residuals"].values():
        row["ratio"] = 0.5
    with pytest.raises(ValueError, match="convergence differs"):
        save_material_checkpoint(tmp_path, **kwargs)
