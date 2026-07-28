from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import silicon_oracle_validation as driver


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample_payload(
    *,
    inputs: dict[str, str] | None = None,
    mapping_hashes: dict[str, str] | None = None,
) -> dict:
    return {
        "schema": driver.V3_SAMPLE_MANIFEST_SCHEMA,
        "schema_version": 1,
        "inputs": inputs
        or {
            "reference_hdf5_sha256": "1" * 64,
            "fc2_compact_hdf5_sha256": "2" * 64,
            "fc2_full_hdf5_sha256": "3" * 64,
            "fc3_compact_hdf5_sha256": "4" * 64,
            "fc3_full_hdf5_sha256": "5" * 64,
            "phono3py_yaml_sha256": "6" * 64,
            "input_manifest_sha256": "7" * 64,
        },
        "mapping_sha256": mapping_hashes
        or {"fc2": "a" * 64, "fc3": "b" * 64},
        "mesh": [1, 1, 1],
        "grg2bzg_sha256": driver.array_sha256(
            np.array([0], dtype=np.int64)
        ),
        "bzg2grg_sha256": driver.array_sha256(
            np.array([0], dtype=np.int64)
        ),
        "degeneracy_tolerance_thz": 1.0e-4,
        "selection_algorithm": driver.V3_SAMPLE_SELECTION_ALGORITHM,
        "selection_algorithm_version": 1,
        "robust_nonzero_floor_ev2": 1.0e-12,
        "coverage": {
            "cover_all_target_grid_points": True,
            "cover_all_target_bands": True,
            "required_processes": ["normal"],
            "required_strata": ["optical_normal"],
            "empty_strata": ["optical_umklapp"],
            "minimum_promoted_blocks": 1,
        },
        "samples": [
            {
                "q0_grg": 0,
                "q1_grg": 0,
                "expected_q2_grg": 0,
                "b0": 0,
                "b1": 0,
                "b2": 0,
                "target_subspace": [0],
                "band1_subspace": [0],
                "band2_subspace": [0],
                "process": "normal",
                "reference_invariant_block_sum_ev2": 1.0,
                "stratum": "optical_normal",
            }
        ],
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_strict_v3_sample_manifest_rejects_duplicate_promoted_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_MIN_PROMOTED_BLOCKS", 1)
    path = tmp_path / "samples.json"
    payload = _sample_payload()
    _write_json(path, payload)
    manifest = driver.read_interaction_sample_manifest(path)

    assert manifest.sha256 == _sha256(path)
    assert manifest.samples[0].promoted_block_key == (
        0,
        0,
        (0,),
        (0,),
        (0,),
    )
    assert manifest.comparator_samples[0].target_grid_point == 0
    with pytest.raises(TypeError):
        manifest.inputs["fc2_hdf5_sha256"] = "f" * 64

    payload["samples"].append(
        {
            **payload["samples"][0],
            "b0": 0,
            "b1": 0,
            "b2": 0,
        }
    )
    _write_json(path, payload)
    with pytest.raises(ValueError, match="duplicate promoted blocks"):
        driver.read_interaction_sample_manifest(path)


def test_strict_v3_sample_manifest_records_empty_strata_and_rejects_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_MIN_PROMOTED_BLOCKS", 1)
    path = tmp_path / "samples.json"
    payload = _sample_payload()
    payload["samples"][0]["stratum"] = "optical_umklapp"
    _write_json(path, payload)
    with pytest.raises(ValueError, match="required stratum"):
        driver.read_interaction_sample_manifest(path)

    payload["coverage"]["required_strata"] = ["optical_umklapp"]
    _write_json(path, payload)
    with pytest.raises(ValueError, match="disjoint"):
        driver.read_interaction_sample_manifest(path)


def test_release_sample_manifest_requires_64_promoted_blocks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "samples.json"
    _write_json(path, _sample_payload())
    with pytest.raises(ValueError, match="at least 64"):
        driver.read_interaction_sample_manifest(path)


def test_atomic_json_publication_never_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    digest = driver.write_json_atomic_exclusive(target, {"passed": True})
    assert digest == _sha256(target)
    original = target.read_bytes()

    with pytest.raises(FileExistsError):
        driver.write_json_atomic_exclusive(target, {"passed": False})
    assert target.read_bytes() == original
    assert not list(tmp_path.glob(".*.tmp"))


def test_checksum_gate_verifies_unconsumed_manifest_entries(
    tmp_path: Path,
) -> None:
    paths = {
        "reference_hdf5": tmp_path / "reference.hdf5",
        "fc2_compact_hdf5": tmp_path / "fc2.compact.hdf5",
        "fc2_full_hdf5": tmp_path / "fc2.full.hdf5",
        "fc3_compact_hdf5": tmp_path / "fc3.compact.hdf5",
        "fc3_full_hdf5": tmp_path / "fc3.full.hdf5",
    }
    extra = tmp_path / "reference-settings.json"
    for role, path in paths.items():
        path.write_bytes(role.encode("ascii"))
    extra.write_bytes(b"before")
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in (*paths.values(), extra)
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        checksums_path=checksums,
        reference_path=paths["reference_hdf5"],
        fc2_compact_path=paths["fc2_compact_hdf5"],
        fc2_full_path=paths["fc2_full_hdf5"],
        fc3_compact_path=paths["fc3_compact_hdf5"],
        fc3_full_path=paths["fc3_full_hdf5"],
    )
    records = {
        role: {"sha256": _sha256(path)} for role, path in paths.items()
    }
    driver._verify_generated_checksums(config, records)

    extra.write_bytes(b"after")
    with pytest.raises(ValueError, match="has changed"):
        driver._verify_generated_checksums(config, records)


def test_full_compact_representation_mismatch_fails_gate() -> None:
    equal = driver._representation_error(
        np.eye(2, dtype=np.complex128),
        np.eye(2, dtype=np.complex128),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    mismatch = driver._representation_error(
        np.eye(2, dtype=np.complex128),
        1.01 * np.eye(2, dtype=np.complex128),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert equal["passed"]
    assert not mismatch["passed"]
    assert mismatch["maximum_scaled_error"] > 1.0e-3


def test_imported_code_provenance_must_match_validation_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        driver,
        "code_provenance",
        lambda: {
            "source_tree_sha256": "a" * 64,
            "git_revision": "b" * 40,
            "git_dirty": False,
            "package_version_consistent": True,
            "installed_package_version": "0.1.0rc1",
        },
    )
    result = driver._code_identity(
        {"head_commit": "c" * 40},
        {"packages": {"negfpy": "0.1.0rc1"}},
    )
    assert not result["passed"]
    assert not result["checks"]["git_revision_matches_repository"]


def test_reference_generator_environment_must_match_validator(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    generator = scripts / "generate_phono3py_si_reference.py"
    generator.write_bytes(b"generator")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    checksums = artifacts / "SHA256SUMS"
    checksums.write_text("placeholder\n", encoding="utf-8")
    commit = "c" * 40

    def environment_record(record_commit: str) -> dict:
        return {
            "schema_version": 1,
            "recorded_at_utc": "2026-01-01T00:00:00+00:00",
            "identity": {
                "code": {
                    "git_commit": record_commit,
                    "tracked_worktree_status": "",
                    "script_sha256": _sha256(generator),
                },
                "runtime_versions": {
                    "phono3py": "4.4.0",
                    "phonopy": "4.4.0",
                },
                "thread_environment": {
                    name: "1" for name in driver._THREAD_VARIABLES
                },
            },
        }

    for filename in (
        "force-constants-environment.json",
        "reference-environment.json",
    ):
        _write_json(artifacts / filename, environment_record(commit))
    config = SimpleNamespace(
        repository_path=repository,
        checksums_path=checksums,
    )
    passing = driver._generation_environment_identity(
        config,
        {"head_commit": commit},
    )
    assert passing["passed"]

    _write_json(
        artifacts / "reference-environment.json",
        environment_record("d" * 40),
    )
    failing = driver._generation_environment_identity(
        config,
        {"head_commit": commit},
    )
    assert not failing["passed"]
    assert not failing["records"]["reference"]["checks"][
        "git_commit_matches_validator"
    ]


def _pinned_input_manifest(yaml_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "dataset_id": driver.PINNED_SILICON_DATASET_ID,
        "versions": {"phono3py": "4.4.0", "phonopy": "4.4.0"},
        "source": {
            "repository": driver.PINNED_SILICON_REPOSITORY,
            "commit": driver.PINNED_SILICON_SOURCE_COMMIT,
            "files": {
                "phono3py_yaml": {
                    "path": "example/Si-PBEsol/phono3py_disp.yaml",
                    "sha256": yaml_sha256,
                }
            },
        },
    }


def test_bounded_driver_uses_reference_selectors_r0_average_and_strict_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_MIN_PROMOTED_BLOCKS", 1)
    reference_path = tmp_path / "reference.hdf5"
    fc2_compact_path = tmp_path / "fc2.compact.hdf5"
    fc2_full_path = tmp_path / "fc2.full.hdf5"
    fc3_compact_path = tmp_path / "fc3.compact.hdf5"
    fc3_full_path = tmp_path / "fc3.full.hdf5"
    yaml_path = tmp_path / "phono3py_disp.yaml"
    lock_path = tmp_path / "environment.lock"
    for path, payload in (
        (reference_path, b"reference"),
        (fc2_compact_path, b"fc2-compact"),
        (fc2_full_path, b"fc2-full"),
        (fc3_compact_path, b"fc3-compact"),
        (fc3_full_path, b"fc3-full"),
        (yaml_path, b"yaml"),
        (lock_path, b"lock"),
    ):
        path.write_bytes(payload)
    input_manifest_path = tmp_path / "input-manifest.normalized.json"
    _write_json(input_manifest_path, _pinned_input_manifest(_sha256(yaml_path)))
    checksums_path = tmp_path / "SHA256SUMS"
    checksums_path.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in (
                reference_path,
                fc2_compact_path,
                fc2_full_path,
                fc3_compact_path,
                fc3_full_path,
            )
        ),
        encoding="utf-8",
    )
    sample_inputs = {
        "reference_hdf5_sha256": _sha256(reference_path),
        "fc2_compact_hdf5_sha256": _sha256(fc2_compact_path),
        "fc2_full_hdf5_sha256": _sha256(fc2_full_path),
        "fc3_compact_hdf5_sha256": _sha256(fc3_compact_path),
        "fc3_full_hdf5_sha256": _sha256(fc3_full_path),
        "phono3py_yaml_sha256": _sha256(yaml_path),
        "input_manifest_sha256": _sha256(input_manifest_path),
    }
    samples_path = tmp_path / "v3-samples.json"
    _write_json(samples_path, _sample_payload(inputs=sample_inputs))

    reference = SimpleNamespace(
        mesh=(1, 1, 1),
        grid_points_grg=np.array([0], dtype=np.int64),
        grid_points_bzg=np.array([0], dtype=np.int64),
        band_indices=np.array([0], dtype=np.int64),
        temperatures_k=np.array([300.0]),
        sigmas_thz=np.array([0.1]),
        cutoff_frequency_thz=1.0e-4,
        make_r0_average=True,
        fc2_random_qpoints=np.array([[0.125, 0.25, -0.375]]),
        fc2_hermitianized=True,
        n_qpoints=1,
        n_bands=1,
        sha256=_sha256(reference_path),
        grg2bzg=np.array([0], dtype=np.int64),
        bzg2grg=np.array([0], dtype=np.int64),
        frequencies_in_grg_order=lambda: np.array([[2.0]]),
    )
    fc2_mapping = SimpleNamespace(
        purpose="fc2",
        supercell=SimpleNamespace(masses=np.array([28.0])),
        p2s=np.array([0]),
        s2p=np.array([0]),
    )
    fc3_mapping = SimpleNamespace(
        purpose="fc3",
        supercell=SimpleNamespace(masses=np.array([28.0])),
        p2s=np.array([0]),
        s2p=np.array([0]),
    )
    fc2_compact = SimpleNamespace(compact=True)
    fc2_full = SimpleNamespace(compact=False)
    fc3_compact_raw = object()
    fc3_full_raw = object()

    def fourier(qpoints):
        return np.ones((1, 1, 1, 3, 3, 3), dtype=np.complex128)

    compact_translational = SimpleNamespace(
        source_layout="compact",
        terms=(1, 2),
        all_shortest=np.ones((1, 1, 1), dtype=np.byte),
        fourier=fourier,
    )
    full_translational = SimpleNamespace(
        source_layout="full",
        terms=(1, 2),
        all_shortest=np.ones((1, 1, 1), dtype=np.byte),
        fourier=fourier,
    )
    validation = SimpleNamespace(
        passed=True,
        mapping_valid=True,
        schedule_complete=True,
        gamma_finite=True,
        asr_valid=True,
        asr_valid_by_axis=(True, True, True),
        gamma_asr_max_abs=np.zeros(3),
        gamma_asr_relative=np.zeros(3),
        gamma_scale=1.0,
        asr_threshold=1.0e-8,
        support_valid=True,
        pairwise_atom_axes=((0, 1), (0, 2), (1, 2)),
        pairwise_minimum_distances=np.zeros(3),
        pairwise_maximum_distances=np.ones(3),
        maximum_pairwise_distance=1.0,
        distance_unit="angstrom",
        support_policy="exact",
        nonzero_expanded_term_count=1,
        exact_zero_expanded_term_count=0,
        nonzero_raw_block_count=1,
        exact_zero_raw_block_count=0,
        make_r0_average=True,
        phase_convention="positive",
        anchor_convention="r0",
        errors=(),
        warnings=(),
        metrics={},
    )
    interactions = SimpleNamespace(
        grid_points=np.array([0]),
        band_indices=np.array([0]),
        q1_indices=np.array([[0]]),
        q2_indices=np.array([[0]]),
        normal_mask=np.array([[True]]),
        interaction_strengths=np.array([[[[[1.0]]]]]),
    )
    linewidths = SimpleNamespace(
        interactions=interactions,
        gamma_thz_hwhm=np.array([[[0.1]]]),
        decay_phase_space=np.array([[2.0]]),
        difference_phase_space=np.array([[0.5]]),
    )
    block = SimpleNamespace(
        target_grid_point=0,
        q1_grid_point=0,
        q2_grid_point=0,
        target_bands=(0,),
        bands1=(0,),
        bands2=(0,),
        reference_sum_ev2=1.0,
        candidate_sum_ev2=1.0,
        absolute_error_ev2=0.0,
        relative_error=0.0,
        passed=True,
    )
    comparison = SimpleNamespace(
        passed=True,
        fc2=SimpleNamespace(passed=True),
        interactions=SimpleNamespace(
            passed=True,
            blocks=(block,),
            absolute_tolerance_ev2=1.0e-14,
        ),
        phase_space=SimpleNamespace(passed=True),
        linewidths=SimpleNamespace(passed=True),
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        driver,
        "_runtime_identity",
        lambda: {"packages": {"phono3py": "4.4.0"}},
    )
    monkeypatch.setattr(
        driver,
        "PINNED_SILICON_ENVIRONMENT_LOCK_SHA256",
        _sha256(lock_path),
    )
    monkeypatch.setattr(
        driver,
        "_repository_identity",
        lambda path, require_clean: {"clean": True, "head_commit": "c" * 40},
    )
    monkeypatch.setattr(
        driver,
        "_code_identity",
        lambda repository, runtime: {"passed": True, "checks": {}},
    )
    monkeypatch.setattr(
        driver,
        "_generation_environment_identity",
        lambda config, repository: {"passed": True, "records": {}},
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_silicon_reference",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_crystal_mapping",
        lambda path, *, purpose, is_nac: (
            fc2_mapping if purpose == "fc2" else fc3_mapping
        ),
    )
    monkeypatch.setattr(
        driver,
        "crystal_mapping_sha256",
        lambda mapping: "a" * 64 if mapping.purpose == "fc2" else "b" * 64,
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_fc2",
        lambda path, *a, **k: (
            fc2_compact if path == fc2_compact_path else fc2_full
        ),
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_fc3",
        lambda path, *a, **k: (
            fc3_compact_raw if path == fc3_compact_path else fc3_full_raw
        ),
    )

    def convert(*args, **kwargs):
        calls.setdefault("make_r0_average", []).append(
            kwargs["make_r0_average"]
        )
        return (
            compact_translational
            if args[0] is fc3_compact_raw
            else full_translational
        )

    monkeypatch.setattr(driver, "convert_fc3_to_translational", convert)
    monkeypatch.setattr(
        driver,
        "validate_translational_fc3",
        lambda *a, **k: validation,
    )
    monkeypatch.setattr(
        driver,
        "compute_bulk_phonons",
        lambda *a, **k: SimpleNamespace(
            dynamical_matrices=np.ones((1, 1, 1), dtype=np.complex128)
        ),
    )
    monkeypatch.setattr(
        driver,
        "BulkThreePhononProblem",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    def compute(problem, **kwargs):
        calls["linewidth_selectors"] = kwargs
        return linewidths

    monkeypatch.setattr(driver, "compute_three_phonon_linewidths", compute)

    def compare(reference_arg, **kwargs):
        calls["comparison_samples"] = kwargs["interaction_samples"]
        return comparison

    monkeypatch.setattr(driver, "compare_silicon_reference", compare)

    output = tmp_path / "result.json"
    config = driver.SiliconOracleValidationConfig(
        reference_path=reference_path,
        fc2_compact_path=fc2_compact_path,
        fc2_full_path=fc2_full_path,
        fc3_compact_path=fc3_compact_path,
        fc3_full_path=fc3_full_path,
        phono3py_yaml_path=yaml_path,
        input_manifest_path=input_manifest_path,
        checksums_path=checksums_path,
        environment_lock_path=lock_path,
        interaction_samples_path=samples_path,
        repository_path=tmp_path,
        output_path=output,
        require_clean_source=False,
    )
    report = driver.run_silicon_oracle_validation(config)

    assert report["validation_passed"]
    assert report["certification_eligible"]
    assert calls["make_r0_average"] == [True, True]
    selectors = calls["linewidth_selectors"]
    np.testing.assert_array_equal(
        selectors["grid_points"],
        reference.grid_points_grg,
    )
    np.testing.assert_array_equal(
        selectors["band_indices"],
        reference.band_indices,
    )
    np.testing.assert_array_equal(
        selectors["temperatures"],
        reference.temperatures_k,
    )
    assert calls["comparison_samples"][0].q1_grid_point == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["execution"]["internal_q_parallelism"] == "none"
    assert written["execution"]["fitted_prefactors"] is False
    assert written["interaction_sample_coverage"][
        "representative_nonzero_block_present"
    ]

    loose_output = tmp_path / "loose-result.json"
    loose_config = driver.SiliconOracleValidationConfig(
        reference_path=reference_path,
        fc2_compact_path=fc2_compact_path,
        fc2_full_path=fc2_full_path,
        fc3_compact_path=fc3_compact_path,
        fc3_full_path=fc3_full_path,
        phono3py_yaml_path=yaml_path,
        input_manifest_path=input_manifest_path,
        checksums_path=checksums_path,
        environment_lock_path=lock_path,
        interaction_samples_path=samples_path,
        repository_path=tmp_path,
        output_path=loose_output,
        require_clean_source=False,
        asr_atol=1.0e-4,
        asr_rtol=1.0e-3,
    )
    loose_report = driver.run_silicon_oracle_validation(loose_config)
    assert loose_report["validation_passed"]
    assert not loose_report["certification_eligible"]
    assert not loose_report["controls"]["controls_exact"]

    monkeypatch.setattr(
        driver,
        "compute_bulk_phonons",
        lambda fc2_arg, *a, **k: SimpleNamespace(
            dynamical_matrices=np.full(
                (1, 1, 1),
                1.0 if fc2_arg is fc2_compact else 2.0,
                dtype=np.complex128,
            )
        ),
    )
    mismatch_config = driver.SiliconOracleValidationConfig(
        reference_path=reference_path,
        fc2_compact_path=fc2_compact_path,
        fc2_full_path=fc2_full_path,
        fc3_compact_path=fc3_compact_path,
        fc3_full_path=fc3_full_path,
        phono3py_yaml_path=yaml_path,
        input_manifest_path=input_manifest_path,
        checksums_path=checksums_path,
        environment_lock_path=lock_path,
        interaction_samples_path=samples_path,
        repository_path=tmp_path,
        output_path=tmp_path / "mismatch-result.json",
        require_clean_source=False,
    )
    mismatch_report = driver.run_silicon_oracle_validation(mismatch_config)
    assert not mismatch_report["validation_passed"]
    assert not mismatch_report["certification_eligible"]
    assert not mismatch_report["full_compact_equality"]["fc2"]["passed"]
