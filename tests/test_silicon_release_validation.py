from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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


def test_external_scientific_view_ignores_only_live_resource_observations() -> None:
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
    for path in (external, equality_3, equality_5):
        path.write_text("{}\n", encoding="utf-8")
    repository = tmp_path / "repository"
    repository.mkdir()
    return release.SiliconReleaseValidationConfig(
        external_oracle_report_path=external,
        equality_3x3_summary_path=equality_3,
        equality_5x5_summary_path=equality_5,
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
            name: True
            for name in release._CANONICAL_EXACT_COMPARISON_FIELDS
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
        for index, role in enumerate(
            release._SHARED_TARGET_ROLE_MAP, start=1
        )
    }
    target = {
        target_role: {
            "sha256": qualification[qualification_role]["sha256"]
        }
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
    checkpoint = (
        run_root / "task-outcomes" / "target-q-00000000.json"
    )
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
    monkeypatch.setattr(
        release, "_runtime_identity", lambda: dict(runtime)
    )
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
        **kwargs: object,
    ) -> dict[str, object]:
        return {
            "summary_path": str(path),
            "summary_sha256": (
                "3" * 64 if expected_mesh[0] == 3 else "5" * 64
            ),
            "mesh": list(expected_mesh),
            "fresh_complete_mpi_execution": True,
            "full_field_equality": True,
        }

    monkeypatch.setattr(
        release, "_validate_equality_summary", equality
    )
    monkeypatch.setattr(
        release,
        "validate_v3_mpi_orchestration_evidence",
        lambda *args, **kwargs: SimpleNamespace(
            file_sha256=release._sha256(orchestration_path.resolve()),
            payload_sha256="2" * 64,
            campaign_sha256="3" * 64,
            source_identity_sha256="4" * 64,
            source_authenticated=True,
            source_revision=repository["head_commit"],
            source_tree_sha256=imported["record"]["source_tree_sha256"],
            environment_sha256="5" * 64,
            mesh=(1, 1, 1),
            target_grid_points=(0,),
            target_bands=(0,),
            selected_execution_ids=("maximum", "reservoirs"),
            selected_communicator_sizes=(2, 4),
            selected_modes=("mpi", "mpi"),
            restart_resume_recorded=True,
            physical_core_binding_passed=True,
            mpi_origin_lineage_passed=True,
        ),
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
