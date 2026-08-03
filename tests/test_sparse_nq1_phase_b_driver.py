from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pytest

from negfpy.anharmonic.sparse_cubic import SparseCubicVertex


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
driver = importlib.import_module("validate_sparse_nq1_phase_b")
campaign = importlib.import_module("run_sparse_nq1_phase_b_campaign")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_authenticates_full_shape_and_probes_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "phono3py_disp.yaml").write_text("synthetic\n", encoding="ascii")
    fc3_path = tmp_path / "fc3.full.hdf5"
    with h5py.File(fc3_path, "w") as handle:
        handle.create_dataset("fc3", data=np.zeros((2, 2, 2, 3, 3, 3)))
    monkeypatch.setattr(driver, "EXPECTED_RAW_FC3_SHA256", _sha256(fc3_path))
    monkeypatch.setattr(
        driver,
        "verify_phase_a_fixture",
        lambda _path: {
            "fixture_manifest_sha256": "a" * 64,
            "fixture_sha256sums_sha256": "b" * 64,
        },
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_crystal_mapping",
        lambda _path, *, purpose: SimpleNamespace(
            purpose=purpose, n_supercell_atoms=2
        ),
    )
    monkeypatch.setattr(
        driver, "primitive_crystal_content_sha256", lambda _mapping: "c" * 64
    )
    monkeypatch.setattr(
        driver, "_available_memory_bytes", lambda: driver.MIN_AVAILABLE_MEMORY_BYTES
    )

    result = driver.preflight_phase_b_inputs(
        fixture=fixture,
        fc3_full=fc3_path,
        output_json=tmp_path / "out/outcome.json",
        output_hdf5=tmp_path / "out/mapped.h5",
    )

    assert result["passed"]
    assert result["fc3_full"]["shape"] == [2, 2, 2, 3, 3, 3]
    assert result["atomic_hdf5_probe_passed"]
    assert not list((tmp_path / "out").glob(".negfpy-phase-b-probe.*"))


def test_preflight_rejects_compact_or_wrong_raw_fc3_before_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "phono3py_disp.yaml").write_text("synthetic\n", encoding="ascii")
    fc3_path = tmp_path / "fc3.full.hdf5"
    with h5py.File(fc3_path, "w") as handle:
        handle.create_dataset("fc3", data=np.zeros((1, 2, 2, 3, 3, 3)))
    monkeypatch.setattr(driver, "EXPECTED_RAW_FC3_SHA256", _sha256(fc3_path))
    monkeypatch.setattr(driver, "verify_phase_a_fixture", lambda _path: {})
    monkeypatch.setattr(
        driver,
        "read_phono3py_crystal_mapping",
        lambda _path, *, purpose: SimpleNamespace(
            purpose=purpose, n_supercell_atoms=2
        ),
    )
    monkeypatch.setattr(
        driver, "primitive_crystal_content_sha256", lambda _mapping: "d" * 64
    )
    monkeypatch.setattr(
        driver, "_available_memory_bytes", lambda: driver.MIN_AVAILABLE_MEMORY_BYTES
    )
    with pytest.raises(driver.PhaseBError, match="not the required full tensor"):
        driver.preflight_phase_b_inputs(
            fixture=fixture,
            fc3_full=fc3_path,
            output_json=tmp_path / "outcome.json",
            output_hdf5=tmp_path / "mapped.h5",
        )

    monkeypatch.setattr(driver, "EXPECTED_RAW_FC3_SHA256", "e" * 64)
    with pytest.raises(driver.PhaseBError, match="SHA256 mismatch"):
        driver.preflight_phase_b_inputs(
            fixture=fixture,
            fc3_full=fc3_path,
            output_json=tmp_path / "outcome.json",
            output_hdf5=tmp_path / "mapped.h5",
        )


def _patch_bounded_pipeline(
    monkeypatch: pytest.MonkeyPatch, *, exact_permutation: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.array(
        [[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=np.int64
    )
    values = np.array([2.0, -3.0, 4.0], dtype=np.float64)
    coefficients = values * 0.125
    monkeypatch.setattr(
        driver,
        "preflight_phase_b_inputs",
        lambda **_kwargs: {
            "schema": driver.PREFLIGHT_SCHEMA,
            "fc3_full": {"raw_sha256": driver.EXPECTED_RAW_FC3_SHA256},
            "passed": True,
        },
    )
    fc3_mapping = SimpleNamespace(
        purpose="fc3",
        supercell=SimpleNamespace(masses=np.ones(2)),
        p2s=np.array([0, 1]),
        s2p=np.array([0, 1]),
    )
    fc2_mapping = SimpleNamespace(
        purpose="fc2",
        supercell=SimpleNamespace(masses=np.ones(2)),
        p2s=np.array([0, 1]),
        s2p=np.array([0, 1]),
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_crystal_mapping",
        lambda _path, *, purpose: fc3_mapping if purpose == "fc3" else fc2_mapping,
    )
    raw_fc3 = SimpleNamespace(name="raw")
    monkeypatch.setattr(driver, "read_phono3py_fc3", lambda *_args, **_kwargs: raw_fc3)
    monkeypatch.setattr(
        driver, "fc3_content_sha256", lambda _raw: driver.EXPECTED_CANONICAL_FC3_SHA256
    )
    raw_report = SimpleNamespace(
        passed=True,
        permutation_complete=True,
        errors=(),
        warnings=(),
        metrics={"permutation": 0.0},
    )
    monkeypatch.setattr(driver, "validate_fc3", lambda _raw: raw_report)
    translational = SimpleNamespace(make_r0_average=True)
    monkeypatch.setattr(
        driver,
        "convert_fc3_to_translational",
        lambda *_args, **kwargs: translational
        if kwargs.get("make_r0_average") is True
        else pytest.fail("r0 averaging was not requested"),
    )
    normalized = SimpleNamespace(
        source_translational_sha256="1" * 64,
        content_sha256="2" * 64,
        fc3_crystal_mapping_content_sha256="3" * 64,
    )
    monkeypatch.setattr(
        driver, "mass_normalize_translational_fc3", lambda *_args, **_kwargs: normalized
    )
    monkeypatch.setattr(driver, "read_phono3py_fc2", lambda *_args, **_kwargs: object())
    layered = SimpleNamespace(
        n_principal_layer_dofs=6,
        fc2_content_sha256="4" * 64,
        primitive_crystal_content_sha256="5" * 64,
        fc2_crystal_mapping_content_sha256="6" * 64,
        layered_geometry_sha256="7" * 64,
    )
    monkeypatch.setattr(
        driver,
        "build_layered_device_mapping",
        lambda *_args, **kwargs: layered
        if np.array_equal(kwargs["transport_supercell_matrix"], driver.TRANSPORT_TRANSFORM)
        else pytest.fail("wrong transport transform"),
    )
    harmonic = SimpleNamespace(name="harmonic")
    monkeypatch.setattr(driver, "build_layered_harmonic_model", lambda *_args: harmonic)
    mapping_report = SimpleNamespace(
        n_device_layers=5,
        contact_layers=(0, 4),
        harmonic_buffer_layers=((0,), (4,)),
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
        required_left_halo_layers=1,
        required_right_halo_layers=1,
        failure_codes=() if exact_permutation else ("exact_permutation_equality",),
        certification_passed=exact_permutation,
        permutation_tolerance_passed=exact_permutation,
        exact_permutation_equality=exact_permutation,
        symmetrized=False,
        corrected=False,
        truncated=False,
        thresholded=False,
        report_sha256="8" * 64,
    )
    mapped = SimpleNamespace(
        report=mapping_report,
        indices=indices,
        values=values,
        ndof=30,
        nnz=3,
        support_density=3 / 30**3,
        content_sha256="9" * 64,
        cubic_physics_sha256="a" * 64,
    )
    monkeypatch.setattr(
        driver,
        "map_fc3_to_device",
        lambda *_args, **kwargs: mapped
        if kwargs == {
            "n_device_layers": 5,
            "interaction_layers": (1, 2, 3),
            "observation_layers": (2,),
        }
        else pytest.fail("wrong finite mapping domain"),
    )
    lead = SimpleNamespace(
        passed=True,
        classification="acoustic",
        failure_reasons=(),
        gamma_nullity=3,
        classification_sha256="b" * 64,
    )
    monkeypatch.setattr(driver, "classify_harmonic_lead", lambda _model: lead)
    sparse = SparseCubicVertex(
        ndof=30,
        indices=indices,
        coefficients=coefficients,
        metadata={
            "source": "synthetic-phase-b",
            "integer_array": np.array([1, 2, 3], dtype=np.int64),
            "nested": {"tuple": ("a", 4.0)},
        },
    )
    monkeypatch.setattr(driver, "scale_device_fc3", lambda *_args: sparse)
    return indices, values, coefficients


def test_run_phase_b_publishes_reloadable_non_scba_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    indices, values, coefficients = _patch_bounded_pipeline(monkeypatch)
    output_json = tmp_path / "outcome.json"
    output_hdf5 = tmp_path / "mapped.h5"
    outcome = driver.run_phase_b(
        fixture=tmp_path / "fixture",
        fc3_full=tmp_path / "fc3.full.hdf5",
        output_json=output_json,
        output_hdf5=output_hdf5,
        campaign_manifest_sha256="f" * 64,
    )

    assert outcome["phase_b_certified"]
    assert not outcome["scba_executed"]
    assert not outcome["release_certified"]
    assert outcome["dense_scba_memory_estimates_bytes"] == {
        "17": 288 * 17 * 30**2,
        "41": 288 * 41 * 30**2,
    }
    archive = driver.verify_phase_b_hdf5(output_hdf5)
    assert archive["phase_b_certified"]
    assert archive["ndof"] == 30
    assert archive["nnz"] == 3
    with h5py.File(output_hdf5, "r") as handle:
        np.testing.assert_array_equal(handle["indices"][...], indices)
        np.testing.assert_array_equal(handle["values"][...], values)
        np.testing.assert_array_equal(handle["solver_coefficients"][...], coefficients)
    restored = driver.load_phase_b_sparse_vertex(output_hdf5)
    assert restored.vertex_storage_sha256 == outcome["identities"][
        "sparse_vertex_storage_sha256"
    ]
    assert restored.cubic_physics_sha256 == outcome["identities"][
        "cubic_physics_sha256_solver"
    ]
    assert restored.metadata_sha256 == archive["metadata"]["identities"][
        "sparse_vertex_metadata_sha256"
    ]
    np.testing.assert_array_equal(restored.metadata["integer_array"], [1, 2, 3])
    assert json.loads(output_json.read_text(encoding="ascii")) == outcome
    with pytest.raises(driver.PhaseBError, match="already exist"):
        driver.run_phase_b(
            fixture=tmp_path / "fixture",
            fc3_full=tmp_path / "fc3.full.hdf5",
            output_json=output_json,
            output_hdf5=output_hdf5,
            campaign_manifest_sha256="f" * 64,
        )


def test_failed_exact_permutation_is_archived_without_sparse_scaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bounded_pipeline(monkeypatch, exact_permutation=False)
    monkeypatch.setattr(
        driver,
        "scale_device_fc3",
        lambda *_args: pytest.fail("failed mapping must not be scaled"),
    )
    output_hdf5 = tmp_path / "failed.h5"
    outcome = driver.run_phase_b(
        fixture=tmp_path / "fixture",
        fc3_full=tmp_path / "fc3.full.hdf5",
        output_json=tmp_path / "failed.json",
        output_hdf5=output_hdf5,
        campaign_manifest_sha256="0" * 64,
    )
    assert not outcome["phase_b_certified"]
    with h5py.File(output_hdf5, "r") as handle:
        assert "solver_coefficients" not in handle


def test_archive_tamper_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_bounded_pipeline(monkeypatch)
    output_hdf5 = tmp_path / "mapped.h5"
    driver.run_phase_b(
        fixture=tmp_path / "fixture",
        fc3_full=tmp_path / "fc3.full.hdf5",
        output_json=tmp_path / "outcome.json",
        output_hdf5=output_hdf5,
        campaign_manifest_sha256="1" * 64,
    )
    with h5py.File(output_hdf5, "r+") as handle:
        handle["values"][0] += 1.0
    with pytest.raises(driver.PhaseBError, match="physical-value hash mismatch"):
        driver.verify_phase_b_hdf5(output_hdf5)


def test_campaign_lock_and_partial_outcome_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = tmp_path / "material.lock"
    lock.write_text("# pinned\n@EXPLICIT\nhttps://example.invalid/pkg.conda\n", encoding="utf-8")
    monkeypatch.setattr(campaign, "MATERIAL_LOCK_SHA256", _sha256(lock))
    assert campaign._validate_lock(lock)["sha256"] == _sha256(lock)
    lock.write_text("not-explicit\n", encoding="utf-8")
    with pytest.raises(campaign.CampaignError, match="explicit lock"):
        campaign._validate_lock(lock)

    output_json = tmp_path / "partial.json"
    output_json.write_text("{}\n", encoding="ascii")
    with pytest.raises(campaign.CampaignError, match="partial Phase-B outcome"):
        campaign._load_outcome(
            output_json=output_json,
            output_hdf5=tmp_path / "missing.h5",
            manifest_sha256="2" * 64,
        )


def test_campaign_source_identity_rejects_dirty_or_wrong_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    package_file = repo / "src/negfpy/__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_text("", encoding="ascii")
    expected = "3" * 40

    def clean_git(_repo: Path, *arguments: str) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return expected
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return "4" * 40
        if arguments == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(campaign, "_git", clean_git)
    monkeypatch.setattr(campaign.negfpy, "__file__", str(package_file))
    assert campaign._source_identity(repo, expected)["dirty"] is False

    monkeypatch.setattr(
        campaign,
        "_git",
        lambda _repo, *arguments: (
            "untracked.txt" if arguments[0] == "status" else expected
        ),
    )
    with pytest.raises(campaign.CampaignError, match="not clean"):
        campaign._source_identity(repo, expected)


def test_campaign_resume_reauthenticates_fixture_fc3_environment_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    driver_path = scripts / "validate_sparse_nq1_phase_b.py"
    driver_path.write_text("# exact driver\n", encoding="ascii")
    fixture = (tmp_path / "fixture").resolve()
    fixture.mkdir()
    fc3 = (tmp_path / "fc3.full.hdf5").resolve()
    fc3.write_bytes(b"sealed-fc3")
    artifacts = (tmp_path / "artifacts").resolve()
    lock = (tmp_path / "lock").resolve()
    lock.write_text("@EXPLICIT\n", encoding="ascii")
    source = {"commit": "a" * 40, "tree": "b" * 40, "dirty": False}
    lock_record = {"path": str(lock), "sha256": "c" * 64, "format": "@EXPLICIT"}
    versions = {"phono3py": "4.4.0", "phonopy": "4.4.0"}
    fixture_record = {"fixture_manifest_sha256": "d" * 64}
    monkeypatch.setattr(campaign, "_source_identity", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(campaign, "_validate_lock", lambda _path: lock_record)
    monkeypatch.setattr(campaign, "_versions", lambda: versions)
    monkeypatch.setattr(
        campaign.phase_b, "verify_phase_a_fixture", lambda _path: fixture_record
    )
    manifest = {
        "schema": campaign.SCHEMA,
        "source": source,
        "environment_lock": lock_record,
        "driver": {"sha256": _sha256(driver_path)},
        "preflight": {
            "fixture_path": str(fixture),
            "fixture": fixture_record,
            "fc3_full": {"path": str(fc3), "raw_sha256": _sha256(fc3)},
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": campaign.platform.python_implementation(),
            "version": campaign.platform.python_version(),
            "prefix": str(Path(sys.prefix).resolve()),
        },
        "distributions": versions,
        "outputs": {
            "json": str(artifacts / "outcomes/phase-b-mapping.json"),
            "hdf5": str(artifacts / "outcomes/phase-b-mapped-fc3.h5"),
        },
    }
    campaign._validate_resume_manifest(
        manifest,
        repo=repo,
        fixture=fixture,
        fc3_full=fc3,
        artifacts=artifacts,
        expected_commit="a" * 40,
        environment_lock=lock,
    )

    wrong = json.loads(json.dumps(manifest))
    wrong["preflight"]["fc3_full"]["path"] = str(tmp_path / "other-fc3.h5")
    with pytest.raises(campaign.CampaignError, match="FC3 path mismatch"):
        campaign._validate_resume_manifest(
            wrong,
            repo=repo,
            fixture=fixture,
            fc3_full=fc3,
            artifacts=artifacts,
            expected_commit="a" * 40,
            environment_lock=lock,
        )

    fc3.write_bytes(b"changed-after-manifest")
    with pytest.raises(campaign.CampaignError, match="FC3 identity mismatch"):
        campaign._validate_resume_manifest(
            manifest,
            repo=repo,
            fixture=fixture,
            fc3_full=fc3,
            artifacts=artifacts,
            expected_commit="a" * 40,
            environment_lock=lock,
        )
