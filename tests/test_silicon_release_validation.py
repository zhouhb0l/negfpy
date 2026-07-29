from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import silicon_release_validation as release


def _external_view_fixture() -> dict[str, object]:
    return {
        "started_at_utc": "first",
        "completed_at_utc": "first",
        "wall_time_seconds": 1.0,
        "stage_times_seconds": {"science": 1.0},
        "comparison": {"passed": True, "value": 2.0},
        "merged_result": {
            "load_preflight": {
                "passed": True,
                "mem_available_bytes": 100,
                "limit_bytes": 70,
                "output_filesystem_free_bytes": 1000,
                "predicted_peak_bytes": 50,
            }
        },
        "campaign_evidence": {
            "independent_task_remerge": {
                "passed": True,
                "evidence_sha256": "a" * 64,
                "disk_preflight": {
                    "passed": True,
                    "observed_free_bytes": 2000,
                    "required_free_bytes": 500,
                },
                "scientific_result_sha256": "b" * 64,
            }
        },
    }


def test_external_scientific_view_ignores_only_live_resource_observations() -> (
    None
):
    first = _external_view_fixture()
    second = _external_view_fixture()
    second["started_at_utc"] = "second"
    second["wall_time_seconds"] = 9.0
    second_merged = second["merged_result"]
    assert isinstance(second_merged, dict)
    second_load = second_merged["load_preflight"]
    assert isinstance(second_load, dict)
    second_load["mem_available_bytes"] = 999
    second_load["limit_bytes"] = 699
    second_load["output_filesystem_free_bytes"] = 888
    second_campaign = second["campaign_evidence"]
    assert isinstance(second_campaign, dict)
    second_remerge = second_campaign["independent_task_remerge"]
    assert isinstance(second_remerge, dict)
    second_remerge["evidence_sha256"] = "c" * 64
    second_disk = second_remerge["disk_preflight"]
    assert isinstance(second_disk, dict)
    second_disk["observed_free_bytes"] = 777

    assert release._external_scientific_view(
        first
    ) == release._external_scientific_view(second)

    second_comparison = second["comparison"]
    assert isinstance(second_comparison, dict)
    second_comparison["value"] = 3.0
    assert release._external_scientific_view(
        first
    ) != release._external_scientific_view(second)


def _config(
    tmp_path: Path,
) -> release.SiliconReleaseValidationConfig:
    external = tmp_path / "external.json"
    equality_3 = tmp_path / "equality-3.json"
    equality_5 = tmp_path / "equality-5.json"
    v3_3_manifest = tmp_path / "v3-3.json"
    v3_3_orchestration = tmp_path / "v3-3.mpi.json"
    v3_5_manifest = tmp_path / "v3-5.json"
    v3_5_orchestration = tmp_path / "v3-5.mpi.json"
    for path in (
        external,
        equality_3,
        equality_5,
        v3_3_manifest,
        v3_3_orchestration,
        v3_5_manifest,
        v3_5_orchestration,
    ):
        path.write_text("{}\n", encoding="utf-8")
    v3_3_run = tmp_path / "v3-3-run"
    v3_5_run = tmp_path / "v3-5-run"
    v3_3_run.mkdir()
    v3_5_run.mkdir()
    repository = tmp_path / "repository"
    repository.mkdir()
    return release.SiliconReleaseValidationConfig(
        external_oracle_report_path=external,
        equality_3x3_summary_path=equality_3,
        equality_5x5_summary_path=equality_5,
        v3_3x3_manifest_path=v3_3_manifest,
        v3_3x3_orchestration_path=v3_3_orchestration,
        v3_3x3_run_directory_path=v3_3_run,
        v3_5x5_manifest_path=v3_5_manifest,
        v3_5x5_orchestration_path=v3_5_orchestration,
        v3_5x5_run_directory_path=v3_5_run,
        repository_path=repository,
        output_path=tmp_path / "release.json",
    )


def _comparison_payload() -> dict[str, object]:
    return {
        "schema": release._COMPARISON_SCHEMA,
        "schema_version": 1,
        "passed": True,
        "rtol": 1.0e-10,
        "atol": 1.0e-12,
        "serial_result_sha256": "1" * 64,
        "merged_result_sha256": "2" * 64,
        "merged_artifact_sha256": "3" * 64,
        "exact_field_passed": {
            name: True for name in release._CANONICAL_EXACT_COMPARISON_FIELDS
        },
        "numerical_field_passed": {
            name: True
            for name in release._CANONICAL_NUMERICAL_COMPARISON_FIELDS
        },
        "maximum_absolute_error": {
            name: 0.0
            for name in release._CANONICAL_NUMERICAL_COMPARISON_FIELDS
        },
        "maximum_relative_error": {
            name: 0.0
            for name in release._CANONICAL_NUMERICAL_COMPARISON_FIELDS
        },
        "errors": [],
    }


def _comparison_report() -> dict[str, object]:
    payload = _comparison_payload()
    return {
        **payload,
        "report_sha256": release._canonical_sha256(payload),
    }


def test_comparison_requires_every_field_and_exact_tolerances() -> None:
    report = _comparison_report()
    validated = release._validate_comparison_report(
        report,
        merged_artifact_sha256="3" * 64,
    )
    assert validated["passed"]

    failed = _comparison_payload()
    failed["numerical_field_passed"] = {
        **failed["numerical_field_passed"],  # type: ignore[arg-type]
        "interactions/matrix_elements": False,
    }
    failed["passed"] = False
    failed_report = {
        **failed,
        "report_sha256": release._canonical_sha256(failed),
    }
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="incomplete, failed",
    ):
        release._validate_comparison_report(
            failed_report,
            merged_artifact_sha256="3" * 64,
        )

    loose = _comparison_payload()
    loose["rtol"] = 1.0e-8
    loose_report = {
        **loose,
        "report_sha256": release._canonical_sha256(loose),
    }
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="mistoleranced",
    ):
        release._validate_comparison_report(
            loose_report,
            merged_artifact_sha256="3" * 64,
        )

    for category, missing in (
        ("exact_field_passed", "metadata/interactions_json"),
        (
            "numerical_field_passed",
            "phonons/frequency_factor_to_thz",
        ),
    ):
        truncated = _comparison_payload()
        fields = dict(truncated[category])  # type: ignore[arg-type]
        fields.pop(missing)
        truncated[category] = fields
        if category == "numerical_field_passed":
            absolute = dict(
                truncated["maximum_absolute_error"]  # type: ignore[arg-type]
            )
            relative = dict(
                truncated["maximum_relative_error"]  # type: ignore[arg-type]
            )
            absolute.pop(missing)
            relative.pop(missing)
            truncated["maximum_absolute_error"] = absolute
            truncated["maximum_relative_error"] = relative
        truncated_report = {
            **truncated,
            "report_sha256": release._canonical_sha256(truncated),
        }
        with pytest.raises(
            release.SiliconReleaseValidationError,
            match="incomplete, failed",
        ):
            release._validate_comparison_report(
                truncated_report,
                merged_artifact_sha256="3" * 64,
            )


def test_release_runtime_requires_exact_installed_0_2_0() -> None:
    repository = {
        "clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
    }
    runtime = {"packages": {"negfpy": "0.2.0"}}
    imported = {
        "passed": True,
        "record": {
            "installed_package_version": "0.2.0",
            "declared_package_version": "0.2.0",
            "package_version": "0.2.0",
            "package_version_consistent": True,
        },
    }
    release._require_release_runtime(repository, runtime, imported)

    runtime["packages"]["negfpy"] = "0.1.0rc1"
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="installed NEGFPy 0.2.0",
    ):
        release._require_release_runtime(repository, runtime, imported)


def test_qualified_full_ifc_must_match_target_generation() -> None:
    qualification = {
        role: {"sha256": str(index) * 64}
        for index, role in enumerate(release._SHARED_TARGET_ROLE_MAP, start=1)
    }
    target = {
        target_role: {"sha256": qualification[qualification_role]["sha256"]}
        for qualification_role, target_role in (
            release._SHARED_TARGET_ROLE_MAP.items()
        )
    }
    release._require_qualification_target_match(
        qualification, target, name="fixture"
    )

    target["target_bundle_fc3.full.hdf5"] = {"sha256": "f" * 64}
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="FC/YAML/manifest/environment identities differ",
    ):
        release._require_qualification_target_match(
            qualification, target, name="fixture"
        )


def _fake_v3_validation(
    scientific_manifest: Path,
    orchestration: Path,
    run_directory: Path,
    **kwargs: object,
) -> SimpleNamespace:
    assert scientific_manifest.is_file()
    assert run_directory.is_dir()
    return SimpleNamespace(
        path=orchestration,
        file_sha256=release._sha256(orchestration),
        payload_sha256="1" * 64,
        campaign_sha256="2" * 64,
        source_identity_sha256="3" * 64,
        source_authenticated=True,
        source_revision=kwargs["expected_source_revision"],
        source_tree_sha256=kwargs["expected_source_tree_sha256"],
        environment_sha256=kwargs["expected_environment_sha256"],
        mesh=tuple(kwargs["expected_mesh"]),
        target_grid_points=tuple(kwargs["expected_target_grid_points"]),
        target_bands=tuple(kwargs["expected_target_bands"]),
        selected_execution_ids=("maximum", "reservoirs"),
        selected_communicator_sizes=(2, 2),
        selected_modes=("mpi", "mpi"),
        restart_resume_recorded=False,
        physical_core_binding_passed=True,
        mpi_origin_lineage_passed=True,
    )


def _v3_helper_inputs(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, dict[str, object]],
    SimpleNamespace,
    release._EvidenceTracker,
]:
    manifest = tmp_path / "samples.json"
    orchestration = tmp_path / "samples.mpi.json"
    run_directory = tmp_path / "v3-run"
    manifest.write_text('{"samples":[]}\n', encoding="utf-8")
    orchestration.write_text('{"orchestration":true}\n', encoding="utf-8")
    run_directory.mkdir()
    tracker = release._EvidenceTracker()
    qualified = tracker.snapshot_path(manifest, name="qualified manifest")
    reference = SimpleNamespace(
        mesh=(3, 3, 3),
        grid_points_grg=np.arange(27, dtype=np.int64),
        band_indices=np.arange(2, dtype=np.int64),
    )
    return (
        manifest.resolve(),
        orchestration.resolve(),
        run_directory.resolve(),
        {"interaction_samples": qualified},
        reference,
        tracker,
    )


def test_equality_reference_mpi_evidence_is_mandatory() -> None:
    reference = SimpleNamespace(
        mesh=(3, 3, 3),
        grid_points_grg=np.arange(27, dtype=np.int64),
        band_indices=np.arange(2, dtype=np.int64),
        n_qpoints=27,
        n_bands=2,
    )
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="lacks mandatory reference-MPI evidence",
    ):
        release._validate_equality_reference_mpi(
            expected_mesh=(3, 3, 3),
            target_records={},
            reference=reference,
        )
    reference.mesh = (5, 5, 5)
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="wrong mesh",
    ):
        release._validate_equality_reference_mpi(
            expected_mesh=(3, 3, 3),
            target_records={},
            reference=reference,
        )


def test_equality_reference_mpi_revalidation_is_bound_to_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestration = tmp_path / "reference-mpi-orchestration.json"
    archive = tmp_path / "reference-mpi-evidence.tar"
    reference_path = tmp_path / "si-phono3py-reference.hdf5"
    orchestration.write_text('{"schema":"fixture"}\n', encoding="utf-8")
    archive.write_bytes(b"reference MPI archive")
    reference_path.write_bytes(b"reference HDF5")
    tracker = release._EvidenceTracker()
    target_records = {
        release._REFERENCE_MPI_TARGET_ROLES["orchestration"]: (
            tracker.snapshot_path(orchestration, name="orchestration")
        ),
        release._REFERENCE_MPI_TARGET_ROLES["evidence_archive"]: (
            tracker.snapshot_path(archive, name="archive")
        ),
        "target_bundle_si-phono3py-reference.hdf5": (
            tracker.snapshot_path(reference_path, name="reference")
        ),
    }
    reference = SimpleNamespace(
        mesh=(3, 3, 3),
        grid_points_grg=np.arange(27, dtype=np.int64),
        band_indices=np.arange(2, dtype=np.int64),
        n_qpoints=27,
        n_bands=2,
    )

    def validate(
        root: Path,
        document: dict[str, object],
        **kwargs: object,
    ) -> dict[str, object]:
        assert root == tmp_path.resolve()
        assert document == {"schema": "fixture"}
        assert tuple(kwargs["expected_mesh"]) == reference.mesh
        archive_stat = archive.stat()
        reference_stat = reference_path.stat()
        return {
            "campaign_sha256": "a" * 64,
            "evidence_archive": {
                "path": str(archive.resolve()),
                "name": archive.name,
                "size_bytes": archive_stat.st_size,
                "sha256": release._sha256(archive.resolve()),
            },
            "evidence_members": [],
            "task_count": 27,
            "execution_count": 1,
            "execution_ids": ["fresh"],
            "restart": {"total_resumed_task_assignments": 0},
            "worker_bindings_by_execution": {"fresh": {"passed": True}},
            "bundle_identity": {"passed": True},
            "merged_reference": {
                "name": reference_path.name,
                "size_bytes": reference_stat.st_size,
                "sha256": release._sha256(reference_path.resolve()),
            },
            "canonical_merge": {"passed": True},
            "full_q_selection": True,
            "full_band_selection": True,
            "floating_point_mpi_reductions": False,
        }

    monkeypatch.setattr(release, "validate_reference_mpi_evidence", validate)
    evidence = release._validate_equality_reference_mpi(
        expected_mesh=(3, 3, 3),
        target_records=target_records,
        reference=reference,
    )
    assert evidence["campaign_sha256"] == "a" * 64
    assert evidence["task_count"] == 27
    assert evidence["canonical_merge"] == {"passed": True}

    archive.write_bytes(b"tampered")
    with pytest.raises(Exception, match="current immutable bytes"):
        tracker.recheck()


def test_equality_v3_manifest_must_match_serial_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        orchestration,
        run_directory,
        qualification,
        reference,
        tracker,
    ) = _v3_helper_inputs(tmp_path)
    other = tmp_path / "other-samples.json"
    other.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(
        release,
        "validate_v3_mpi_orchestration_evidence",
        _fake_v3_validation,
    )

    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="not the exact manifest",
    ):
        release._validate_equality_v3_mpi(
            expected_mesh=(3, 3, 3),
            manifest_path=other.resolve(),
            orchestration_path=orchestration,
            run_directory=run_directory,
            qualification_inputs=qualification,
            reference=reference,
            tracker=tracker,
            repository={"head_commit": "a" * 40},
            imported_code={"record": {"source_tree_sha256": "b" * 64}},
            environment_sha256="c" * 64,
            source_identity={"authenticated": True},
        )


def test_equality_v3_rejects_wrong_mesh_and_postvalidation_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        manifest,
        orchestration,
        run_directory,
        qualification,
        reference,
        tracker,
    ) = _v3_helper_inputs(tmp_path)

    def wrong_mesh(*args: object, **kwargs: object) -> SimpleNamespace:
        result = _fake_v3_validation(*args, **kwargs)
        result.mesh = (5, 5, 5)
        return result

    monkeypatch.setattr(
        release,
        "validate_v3_mpi_orchestration_evidence",
        wrong_mesh,
    )
    helper_arguments = {
        "expected_mesh": (3, 3, 3),
        "manifest_path": manifest,
        "orchestration_path": orchestration,
        "run_directory": run_directory,
        "qualification_inputs": qualification,
        "reference": reference,
        "tracker": tracker,
        "repository": {"head_commit": "a" * 40},
        "imported_code": {"record": {"source_tree_sha256": "b" * 64}},
        "environment_sha256": "c" * 64,
        "source_identity": {"authenticated": True},
    }
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="differs from the explicit release inputs",
    ):
        release._validate_equality_v3_mpi(**helper_arguments)

    monkeypatch.setattr(
        release,
        "validate_v3_mpi_orchestration_evidence",
        _fake_v3_validation,
    )
    evidence = release._validate_equality_v3_mpi(**helper_arguments)
    orchestration.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="changed during final aggregation",
    ):
        release._recheck_equality_v3_mpi(
            evidence,
            source_identity={"authenticated": True},
        )


def test_evidence_tracker_detects_change_before_publication(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"before")
    tracker = release._EvidenceTracker()
    tracker.snapshot_path(artifact, name="artifact")
    artifact.write_bytes(b"after")

    with pytest.raises(Exception, match="current immutable bytes"):
        tracker.recheck()


def test_execution_assignment_files_are_opened_hashed_and_rechecked(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    artifact = run_root / "task-artifacts" / "target-q-00000000.h5"
    checkpoint = run_root / "task-outcomes" / "target-q-00000000.json"
    artifact.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    checkpoint.write_bytes(b"checkpoint")
    execution = {
        "assignments": [
            {
                "task_id": "target-q-00000000",
                "status": "completed",
                "checkpoint_relative_path": checkpoint.name,
                "checkpoint_file_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "artifact_relative_path": artifact.name,
                "artifact_sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
            }
        ]
    }
    tracker = release._EvidenceTracker()
    release._track_assignment_files(
        execution,
        run_root=run_root,
        tracker=tracker,
        name="fixture execution",
    )
    assert tracker.summary()["file_count"] == 2

    artifact.write_bytes(b"tampered")
    with pytest.raises(Exception, match="current immutable bytes"):
        tracker.recheck()

    execution["assignments"][0]["artifact_relative_path"] = "../escape"
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="safe relative",
    ):
        release._track_assignment_files(
            execution,
            run_root=run_root,
            tracker=release._EvidenceTracker(),
            name="fixture execution",
        )

    link = artifact.parent / "link.h5"
    link.symlink_to(artifact)
    execution["assignments"][0]["artifact_relative_path"] = link.name
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="symbolic link",
    ):
        release._track_assignment_files(
            execution,
            run_root=run_root,
            tracker=release._EvidenceTracker(),
            name="fixture execution",
        )


def test_external_report_must_remain_nonrelease_pending_aggregation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    external = {
        "schema": release.SILICON_MPI_ORACLE_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "passed",
        "validation_passed": True,
        "certification_eligible": True,
        "release_certified": True,
        "release_aggregation_required": False,
        "certification_gates": {"science": True},
        "release_gates": {
            "external_reference_certification": True,
            "fresh_3x3_full_field_serial_mpi_equality": True,
            "fresh_5x5_full_field_serial_mpi_equality": True,
            "release_aggregator_complete": True,
        },
    }
    config.external_oracle_report_path.write_text(
        json.dumps(external) + "\n", encoding="utf-8"
    )
    tracker = release._EvidenceTracker()
    with pytest.raises(
        release.SiliconReleaseValidationError,
        match="not eligible for aggregation",
    ):
        release._validate_external_report(
            config,
            tracker=tracker,
            repository={},
            imported_code={},
            runtime={},
        )


def test_final_release_decision_only_after_all_three_validators_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    repository = {
        "repository": str(config.repository_path),
        "head_commit": "a" * 40,
        "tree_hash": "b" * 40,
        "clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "dirty_entry_count": 0,
    }
    runtime = {
        "packages": {"negfpy": "0.2.0"},
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    }
    imported = {
        "passed": True,
        "record": {
            "installed_package_version": "0.2.0",
            "declared_package_version": "0.2.0",
            "package_version": "0.2.0",
            "package_version_consistent": True,
            "source_tree_sha256": "c" * 64,
        },
    }
    monkeypatch.setattr(
        release,
        "_repository_identity",
        lambda *args, **kwargs: dict(repository),
    )
    monkeypatch.setattr(release, "_runtime_identity", lambda: dict(runtime))
    monkeypatch.setattr(
        release,
        "_code_identity",
        lambda *args: {
            **imported,
            "record": dict(imported["record"]),
        },
    )
    external = {
        "report_sha256": "1" * 64,
        "target_inputs": {},
        "temperatures_k": [300.0],
        "sigma_thz": 0.1,
        "cutoff_frequency_thz": 1.0e-4,
    }
    sample_path = tmp_path / "samples.json"
    sample_path.write_text("{}\n", encoding="utf-8")
    orchestration_path = tmp_path / "samples.mpi.json"
    orchestration_path.write_text("{}\n", encoding="utf-8")
    run_directory = tmp_path / "v3-run"
    run_directory.mkdir()
    external["v3_mpi_sampling"] = {
        "scientific_manifest": {
            "path": str(sample_path.resolve()),
            "sha256": release._sha256(sample_path.resolve()),
        },
        "orchestration": {
            "path": str(orchestration_path.resolve()),
            "sha256": release._sha256(orchestration_path.resolve()),
        },
        "run_directory": str(run_directory.resolve()),
        "payload_sha256": "2" * 64,
        "campaign_sha256": "3" * 64,
        "source_identity_sha256": "4" * 64,
        "source_authenticated": True,
        "source_revision": repository["head_commit"],
        "source_tree_sha256": imported["record"]["source_tree_sha256"],
        "environment_sha256": "5" * 64,
        "mesh": [1, 1, 1],
        "target_grid_points": [0],
        "target_bands": [0],
        "selected_execution_ids": ["maximum", "reservoirs"],
        "selected_communicator_sizes": [2, 4],
        "selected_modes": ["mpi", "mpi"],
        "restart_resume_recorded": True,
        "physical_core_binding_passed": True,
        "mpi_origin_lineage_passed": True,
    }
    monkeypatch.setattr(
        release,
        "_validate_external_report",
        lambda *args, **kwargs: dict(external),
    )

    def equality(
        path: Path,
        *,
        expected_mesh: tuple[int, int, int],
        v3_manifest_path: Path,
        v3_orchestration_path: Path,
        v3_run_directory: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        qpoints = list(range(expected_mesh[0] ** 3))
        return {
            "summary_path": str(path),
            "summary_sha256": (
                "3" * 64 if expected_mesh[0] == 3 else "5" * 64
            ),
            "mesh": list(expected_mesh),
            "fresh_complete_mpi_execution": True,
            "full_field_equality": True,
            "v3_mpi_sampling": {
                "scientific_manifest": {
                    "path": str(v3_manifest_path),
                    "sha256": release._sha256(v3_manifest_path),
                },
                "orchestration": {
                    "path": str(v3_orchestration_path),
                    "sha256": release._sha256(v3_orchestration_path),
                },
                "run_directory": str(v3_run_directory),
                "orchestration_file_sha256": release._sha256(
                    v3_orchestration_path
                ),
                "payload_sha256": "2" * 64,
                "campaign_sha256": "3" * 64,
                "source_identity_sha256": "4" * 64,
                "source_authenticated": True,
                "source_revision": repository["head_commit"],
                "source_tree_sha256": imported["record"]["source_tree_sha256"],
                "environment_sha256": "5" * 64,
                "mesh": list(expected_mesh),
                "target_grid_points": qpoints,
                "target_bands": [0],
                "selected_execution_ids": ["maximum", "reservoirs"],
                "selected_communicator_sizes": [2, 4],
                "selected_modes": ["mpi", "mpi"],
                "restart_resume_recorded": True,
                "physical_core_binding_passed": True,
                "mpi_origin_lineage_passed": True,
                "floating_point_mpi_reductions": False,
                "canonical_merge_revalidated": True,
            },
        }

    monkeypatch.setattr(release, "_validate_equality_summary", equality)

    def validate_v3(
        scientific_manifest: Path,
        orchestration: Path,
        run_directory: Path,
        **kwargs: object,
    ) -> SimpleNamespace:
        del scientific_manifest, run_directory
        return SimpleNamespace(
            path=orchestration,
            file_sha256=release._sha256(orchestration),
            payload_sha256="2" * 64,
            campaign_sha256="3" * 64,
            source_identity_sha256="4" * 64,
            source_authenticated=True,
            source_revision=kwargs["expected_source_revision"],
            source_tree_sha256=kwargs["expected_source_tree_sha256"],
            environment_sha256=kwargs["expected_environment_sha256"],
            mesh=tuple(kwargs["expected_mesh"]),
            target_grid_points=tuple(kwargs["expected_target_grid_points"]),
            target_bands=tuple(kwargs["expected_target_bands"]),
            selected_execution_ids=("maximum", "reservoirs"),
            selected_communicator_sizes=(2, 4),
            selected_modes=("mpi", "mpi"),
            restart_resume_recorded=True,
            physical_core_binding_passed=True,
            mpi_origin_lineage_passed=True,
        )

    monkeypatch.setattr(
        release,
        "validate_v3_mpi_orchestration_evidence",
        validate_v3,
    )
    monkeypatch.setattr(
        release,
        "current_v3_sampling_source_identity",
        lambda *args, **kwargs: {"authenticated": True},
    )

    report = release.run_silicon_release_validation(config)

    assert report["release_certified"]
    assert report["release_version"] == "0.2.0"
    assert all(report["certification_gates"].values())
    persisted = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert persisted["release_certified"]
    assert persisted["equality_3x3"]["full_field_equality"]
    assert persisted["equality_5x5"]["full_field_equality"]
