from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import bulk_three_phonon
from negfpy.anharmonic import silicon_mpi_oracle_validation as validator


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> validator.SiliconMPIOracleValidationConfig:
    paths: dict[str, Path] = {}
    for name in (
        "qualification_report",
        "campaign_summary",
        "merge_report",
        "merged_result",
        "reference",
        "fc2_compact",
        "fc2_full",
        "fc3_compact",
        "fc3_full",
        "phono3py_yaml",
        "input_manifest",
        "checksums",
        "environment_lock",
        "interaction_samples",
        "interaction_samples_orchestration",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    repository = tmp_path / "repository"
    repository.mkdir()
    interaction_samples_run_directory = (
        tmp_path / "interaction_samples_run_directory"
    )
    interaction_samples_run_directory.mkdir()
    return validator.SiliconMPIOracleValidationConfig(
        qualification_report_path=paths["qualification_report"],
        campaign_summary_path=paths["campaign_summary"],
        merge_report_path=paths["merge_report"],
        merged_result_path=paths["merged_result"],
        reference_path=paths["reference"],
        fc2_compact_path=paths["fc2_compact"],
        fc2_full_path=paths["fc2_full"],
        fc3_compact_path=paths["fc3_compact"],
        fc3_full_path=paths["fc3_full"],
        phono3py_yaml_path=paths["phono3py_yaml"],
        input_manifest_path=paths["input_manifest"],
        checksums_path=paths["checksums"],
        environment_lock_path=paths["environment_lock"],
        interaction_samples_path=paths["interaction_samples"],
        interaction_samples_orchestration_path=(
            paths["interaction_samples_orchestration"]
        ),
        interaction_samples_run_directory_path=(
            interaction_samples_run_directory
        ),
        repository_path=repository,
        output_path=tmp_path / "report.json",
    )


def _file_record(path: Path, **extra: object) -> dict[str, object]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        **extra,
    }


def test_file_record_and_envelope_authentication_are_strict(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"immutable")
    record = _file_record(artifact, report_sha256="a" * 64)

    verified = validator._verify_file_record(
        record,
        name="artifact",
        expected_path=artifact.resolve(),
        extra_fields=("report_sha256",),
    )
    assert verified["sha256"] == _sha256(artifact)
    assert verified["report_sha256"] == "a" * 64

    payload = {"schema_name": "fixture", "complete": True}
    envelope = {
        "payload": payload,
        "payload_sha256": validator._canonical_sha256(payload),
    }
    assert validator._verify_envelope(
        envelope, name="fixture"
    ) == payload

    envelope["payload_sha256"] = "0" * 64
    with pytest.raises(
        validator.SiliconMPIOracleValidationError,
        match="payload SHA256",
    ):
        validator._verify_envelope(envelope, name="fixture")

    artifact.write_bytes(b"changed")
    with pytest.raises(
        validator.SiliconMPIOracleValidationError,
        match="immutable bytes",
    ):
        validator._verify_file_record(
            record,
            name="artifact",
            extra_fields=("report_sha256",),
        )


def test_sparse_large_mesh_v3_sampling_does_not_claim_all_q_coverage() -> None:
    samples = (
        SimpleNamespace(
            q0_grg=0,
            q1_grg=1,
            expected_q2_grg=3,
            b0=0,
            process="normal",
            stratum="acoustic_normal",
        ),
        SimpleNamespace(
            q0_grg=3,
            q1_grg=2,
            expected_q2_grg=1,
            b0=1,
            process="umklapp",
            stratum="optical_umklapp",
        ),
    )
    manifest = SimpleNamespace(
        samples=samples,
        cover_all_target_grid_points=False,
        cover_all_target_bands=True,
        required_processes=("normal", "umklapp"),
        required_strata=("acoustic_normal", "optical_umklapp"),
        empty_strata=(),
        selection_algorithm="bounded_streaming_fixture",
        selection_algorithm_version=1,
    )
    reference = SimpleNamespace(
        grid_points_grg=np.arange(4, dtype=np.int64),
        band_indices=np.arange(2, dtype=np.int64),
        n_qpoints=4,
    )
    result = SimpleNamespace(
        grid_points=np.arange(4, dtype=np.int64),
        band_indices=np.arange(2, dtype=np.int64),
    )

    report = validator._sample_coverage(
        manifest,
        reference=reference,
        result=result,
    )

    assert report["passed"]
    assert not report["cover_all_target_grid_points"]
    assert report["observed_target_grid_point_count"] == 2
    assert report["observed_target_band_count"] == 2


def test_merged_load_preflight_is_fail_closed_at_seventy_percent(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    merged = tmp_path / "merged.h5"
    with h5py.File(merged, "w") as handle:
        handle.create_dataset(
            "field", data=np.ones((100,), dtype=np.float64)
        )
    with h5py.File(merged, "r") as handle:
        report = validator._merged_load_preflight(
            handle,
            merged_path=merged,
            output_path=tmp_path / "report.json",
            mem_available_bytes=4 * 1024**3,
        )
    assert report["passed"]
    assert report["logical_hdf5_dataset_bytes"] == 800
    assert report["limit_fraction"] == 0.70
    assert not report["override_available"]

    with h5py.File(merged, "r") as handle:
        with pytest.raises(
            validator.SiliconMPIOracleValidationError,
            match="exceeds 70%",
        ):
            validator._merged_load_preflight(
                handle,
                merged_path=merged,
                output_path=tmp_path / "report.json",
                mem_available_bytes=1024,
            )


def test_production_summary_rejects_monolithic_recomputation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    summary = {
        "schema": validator.SILICON_TARGET_Q_CLI_SCHEMA,
        "schema_version": 1,
        "status": "passed",
        "provenance_eligible": True,
        "external_validation_eligible": True,
        "memory_override_used": False,
        "complete_target_set": True,
        "complete_band_set": True,
        "complete_scientific_selection": True,
        "no_internal_q1_mpi": True,
        "release_certified": False,
        "scientific_certification": {
            "certified": False,
            "status": "pending_external_comparison",
        },
        "monolithic_comparison": {"passed": True},
    }
    with pytest.raises(
        validator.SiliconMPIOracleValidationError,
        match="must not recompute",
    ):
        validator._authenticate_campaign(
            config,
            summary=summary,
            qualification_sha256="0" * 64,
            qualification_inputs={},
            reference=SimpleNamespace(),
            repository={},
            imported_code={},
            target_inputs={},
        )


def test_campaign_authentication_joins_execution_merge_and_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    repository = {
        "head_commit": "a" * 40,
        "tree_hash": "b" * 40,
        "clean": True,
    }
    imported = {"record": {"source_tree_sha256": "c" * 64}}
    target_paths = {
        "target_environment_lock": config.environment_lock_path,
        "target_phono3py_yaml": config.phono3py_yaml_path,
        "target_bundle_fc2.compact.hdf5": config.fc2_compact_path,
        "target_bundle_fc2.full.hdf5": config.fc2_full_path,
        "target_bundle_fc3.compact.hdf5": config.fc3_compact_path,
        "target_bundle_fc3.full.hdf5": config.fc3_full_path,
        "target_bundle_input-manifest.normalized.json": (
            config.input_manifest_path
        ),
        "target_bundle_si-phono3py-reference.hdf5": (
            config.reference_path
        ),
        "target_sha256sums": config.checksums_path,
    }
    target_records = {
        role: _file_record(path) for role, path in target_paths.items()
    }
    qualification_sha = _sha256(config.qualification_report_path)
    input_hashes = {
        "serial_report": qualification_sha,
        "repository_tree_identity": hashlib.sha256(
            repository["tree_hash"].encode("ascii")
        ).hexdigest(),
        **{
            role: str(record["sha256"])
            for role, record in target_records.items()
        },
    }
    controls = validator.BulkTargetQControls(
        temperatures=[300.0],
        grid_points=[0, 1],
        band_indices=[0],
        n_qpoints=2,
        n_bands=1,
        problem_sha256="d" * 64,
        input_hashes=input_hashes,
    )
    campaign = validator.build_bulk_target_q_campaign(
        controls,
        source_revision=repository["head_commit"],
        source_tree_sha256=imported["record"]["source_tree_sha256"],
        environment_sha256=_sha256(config.environment_lock_path),
        metadata={"fixture": True},
    )

    task_manifest = tmp_path / "task-manifest.json"
    task_manifest.write_text(
        json.dumps(campaign.manifest.as_dict(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    task_ids = list(campaign.canonical_task_ids)
    artifact_hashes = ["1" * 64, "2" * 64]
    checkpoint_hashes = ["3" * 64, "4" * 64]
    assignments = [
        {
            "task_id": task_id,
            "status": "completed",
            "rank": position + 1,
            "hostname": "fixture",
            "elapsed_seconds": 1.0,
            "resumed": False,
            "cpu_affinity": [position],
            "thread_cap_passed": True,
            "checkpoint_relative_path": f"{task_id}.json",
            "checkpoint_file_sha256": checkpoint_hashes[position],
            "artifact_relative_path": f"{task_id}.h5",
            "artifact_sha256": artifact_hashes[position],
        }
        for position, task_id in enumerate(task_ids)
    ]
    execution_payload = {
        "schema_name": "negfpy.task_mpi_execution",
        "schema_version": 1,
        "execution_id": "fixture",
        "manifest_sha256": campaign.campaign_manifest_sha256,
        "mode": "mpi",
        "communicator_size": 3,
        "coordinator_hostname": "fixture",
        "started_utc": "2026-01-01T00:00:00Z",
        "finished_utc": "2026-01-01T00:00:01Z",
        "elapsed_seconds": 1.0,
        "thread_limit": 1,
        "complete": True,
        "assignments": assignments,
        "missing_task_ids": [],
        "infrastructure_errors": [],
    }
    execution = {
        "payload": execution_payload,
        "payload_sha256": validator._canonical_sha256(execution_payload),
    }
    execution_path = tmp_path / "execution.json"
    execution_path.write_text(
        json.dumps(execution, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    merge_payload = {
        "schema": validator.SILICON_TARGET_Q_MERGE_SCHEMA,
        "schema_version": 1,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "merged_artifact_path": str(config.merged_result_path),
        "merged_artifact_sha256": _sha256(config.merged_result_path),
        "task_ids": task_ids,
        "target_grid_points": [0, 1],
        "task_artifact_sha256": artifact_hashes,
        "task_checkpoint_sha256": checkpoint_hashes,
        "dataset_sha256": {"field": "5" * 64},
    }
    merge = {
        **merge_payload,
        "report_sha256": validator._canonical_sha256(merge_payload),
    }
    config.merge_report_path.write_text(
        json.dumps(merge, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    execution_qualification = {
        "executed_task_count": 2,
        "resumed_task_count": 0,
        "orphan_artifact_adopted_task_count": 0,
        "executed_task_ids": task_ids,
        "resumed_task_ids": [],
        "orphan_artifact_adopted_task_ids": [],
        "fresh_complete_task_execution": True,
        "fresh_complete_mpi_execution": True,
        "mpi_equality_eligible": False,
    }
    summary = {
        "schema": validator.SILICON_TARGET_Q_CLI_SCHEMA,
        "schema_version": 1,
        "status": "passed",
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "problem_sha256": controls.problem_sha256,
        "campaign": campaign.as_dict(),
        "controls": controls.as_dict(),
        "controls_sha256": controls.controls_sha256,
        "selected_grid_points": [0, 1],
        "selected_band_indices": [0],
        "complete_target_set": True,
        "complete_band_set": True,
        "complete_scientific_selection": True,
        "execution_id": "fixture",
        "execution": execution,
        "execution_qualification": execution_qualification,
        "merge": merge,
        "monolithic_comparison": None,
        "memory_preflight": {"passed": True},
        "memory_override_used": False,
        "provenance_eligible": True,
        "external_validation_eligible": True,
        "scientific_certification": {
            "certified": False,
            "status": "pending_external_comparison",
        },
        "release_certified": False,
        "evidence": {
            "task_manifest": _file_record(
                task_manifest,
                campaign_manifest_sha256=(
                    campaign.campaign_manifest_sha256
                ),
            ),
            "execution_manifest": _file_record(
                execution_path,
                execution_sha256=execution["payload_sha256"],
            ),
            "merge_report": _file_record(
                config.merge_report_path,
                report_sha256=merge["report_sha256"],
            ),
            "merged_hdf5": _file_record(config.merged_result_path),
            "qualification_report": _file_record(
                config.qualification_report_path
            ),
            "target_reference": _file_record(config.reference_path),
            "target_generation_records": target_records,
            "monolithic_comparison_report": None,
        },
        "no_internal_q1_mpi": True,
    }
    reference = SimpleNamespace(
        grid_points_grg=np.array([0, 1], dtype=np.int64),
        band_indices=np.array([0], dtype=np.int64),
        temperatures_k=np.array([300.0]),
        n_qpoints=2,
        n_bands=1,
        sha256=_sha256(config.reference_path),
    )
    target_inputs = validator._target_input_records(config)
    target_inputs["merged_result"] = _file_record(
        config.merged_result_path
    )
    deep_evidence = {
        "schema": "negfpy.bulk_target_q_release_evidence",
        "passed": True,
        "evidence_sha256": "9" * 64,
    }
    monkeypatch.setattr(
        validator,
        "verify_bulk_target_q_release_evidence",
        lambda *args, **kwargs: deep_evidence,
    )

    evidence = validator._authenticate_campaign(
        config,
        summary=summary,
        qualification_sha256=qualification_sha,
        qualification_inputs={},
        reference=reference,
        repository=repository,
        imported_code=imported,
        target_inputs=target_inputs,
    )

    assert evidence["campaign_manifest_sha256"] == (
        campaign.campaign_manifest_sha256
    )
    assert evidence["execution_qualification"] == execution_qualification
    assert evidence["independent_task_remerge"] == deep_evidence
    assert evidence["no_monolithic_recomputation"]


def test_external_validation_never_calls_monolithic_linewidth_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.campaign_summary_path.write_text("{}\n", encoding="utf-8")
    target = validator._target_input_records(config)
    monkeypatch.setattr(
        validator,
        "PINNED_SILICON_ENVIRONMENT_LOCK_SHA256",
        target["environment_lock"]["sha256"],
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError(
            "compute_three_phonon_linewidths must never be called"
        )

    monkeypatch.setattr(
        bulk_three_phonon,
        "compute_three_phonon_linewidths",
        forbidden,
    )
    legacy_records = {"stable": {"sha256": "1" * 64}}
    monkeypatch.setattr(
        validator, "_input_records", lambda config: legacy_records
    )
    monkeypatch.setattr(
        validator, "_verify_generated_checksums", lambda *args: None
    )
    monkeypatch.setattr(
        validator,
        "_generation_checksum_names",
        lambda path: frozenset({"reference"}),
    )
    monkeypatch.setattr(
        validator,
        "_manifest_identity",
        lambda *args: {"source_commit": "pinned"},
    )
    repository = {
        "clean": True,
        "head_commit": "a" * 40,
        "tree_hash": "b" * 40,
    }
    runtime = {
        "packages": {"negfpy": "fixture"},
        "thread_environment": {},
    }
    imported = {
        "passed": True,
        "record": {"source_tree_sha256": "c" * 64},
    }
    monkeypatch.setattr(
        validator, "_repository_identity", lambda *args, **kwargs: repository
    )
    monkeypatch.setattr(validator, "_runtime_identity", lambda: runtime)
    monkeypatch.setattr(
        validator, "_code_identity", lambda *args: imported
    )
    monkeypatch.setattr(
        validator,
        "_generation_environment_identity",
        lambda *args: {"passed": True},
    )
    monkeypatch.setattr(
        validator,
        "_validate_generation_runtime",
        lambda *args: {"reference": {"passed": True}},
    )
    qualification = {
        "source_repository": {
            "head_commit": repository["head_commit"],
            "tree_hash": repository["tree_hash"],
        }
    }
    monkeypatch.setattr(
        validator,
        "_passed_qualification_report",
        lambda *args, **kwargs: (qualification, {}),
    )
    reference = SimpleNamespace(
        make_r0_average=True,
        fc2_random_qpoints=np.zeros((1, 3)),
        fc2_hermitianized=True,
        n_qpoints=1,
        grid_points_grg=np.array([0], dtype=np.int64),
        band_indices=np.array([0], dtype=np.int64),
        temperatures_k=np.array([300.0]),
        mesh=(1, 1, 1),
        sigmas_thz=np.array([0.1]),
        cutoff_frequency_thz=1.0e-4,
        sha256=target["reference_hdf5"]["sha256"],
    )
    monkeypatch.setattr(
        validator,
        "read_phono3py_silicon_reference",
        lambda *args, **kwargs: reference,
    )
    campaign = {
        "campaign_manifest_sha256": "d" * 64,
        "controls_sha256": "e" * 64,
        "problem_sha256": "f" * 64,
        "no_monolithic_recomputation": True,
    }
    monkeypatch.setattr(
        validator,
        "_authenticate_campaign",
        lambda *args, **kwargs: campaign,
    )
    interactions = SimpleNamespace()
    merged = SimpleNamespace(
        interactions=interactions,
        grid_points=np.array([0]),
        band_indices=np.array([0]),
    )
    monkeypatch.setattr(
        validator,
        "_merged_hdf5_identity",
        lambda *args, **kwargs: {
            "sha256": target["merged_result"]["sha256"]
            if "merged_result" in target
            else _sha256(config.merged_result_path),
            "result": merged,
        },
    )
    monkeypatch.setattr(
        validator,
        "read_phono3py_crystal_mapping",
        lambda *args, **kwargs: SimpleNamespace(
            supercell=SimpleNamespace(masses=np.array([1.0])),
            p2s=np.array([0]),
            s2p=np.array([0]),
        ),
    )
    monkeypatch.setattr(
        validator, "read_phono3py_fc2", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        validator, "compute_bulk_phonons", lambda *args, **kwargs: object()
    )
    manifest = SimpleNamespace(
        sha256="1" * 64,
        comparator_samples=(),
    )
    monkeypatch.setattr(
        validator, "read_v3_sample_manifest", lambda *args: manifest
    )
    v3_mpi_evidence = SimpleNamespace(
        path=config.interaction_samples_orchestration_path,
        file_sha256=_sha256(
            config.interaction_samples_orchestration_path
        ),
        payload_sha256="2" * 64,
        campaign_sha256="3" * 64,
        source_identity_sha256="4" * 64,
        source_authenticated=True,
        source_revision=repository["head_commit"],
        source_tree_sha256=imported["record"]["source_tree_sha256"],
        environment_sha256=target["environment_lock"]["sha256"],
        mesh=(1, 1, 1),
        target_grid_points=(0,),
        target_bands=(0,),
        selected_execution_ids=("maximum", "reservoirs"),
        selected_communicator_sizes=(2, 4),
        selected_modes=("mpi", "mpi"),
        restart_resume_recorded=True,
        physical_core_binding_passed=True,
        mpi_origin_lineage_passed=True,
    )
    monkeypatch.setattr(
        validator,
        "current_v3_sampling_source_identity",
        lambda *args, **kwargs: {"authenticated": True},
    )
    monkeypatch.setattr(
        validator,
        "validate_v3_mpi_orchestration_evidence",
        lambda *args, **kwargs: v3_mpi_evidence,
    )
    monkeypatch.setattr(
        validator,
        "verify_v3_sample_manifest_identity",
        lambda *args, **kwargs: None,
    )
    coverage = {"passed": True}
    monkeypatch.setattr(
        validator, "_sample_coverage", lambda *args, **kwargs: coverage
    )
    comparison = SimpleNamespace(
        passed=True,
        fc2=SimpleNamespace(passed=True),
        interactions=SimpleNamespace(passed=True),
        phase_space=SimpleNamespace(passed=True),
        linewidths=SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        validator,
        "compare_silicon_reference",
        lambda *args, **kwargs: comparison,
    )
    monkeypatch.setattr(
        validator,
        "_validate_sample_manifest_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        validator, "_verify_inputs_unchanged", lambda *args: None
    )

    report = validator.run_silicon_mpi_oracle_validation(config)

    assert report["certification_eligible"]
    assert not report["release_certified"]
    assert report["release_aggregation_required"]
    assert report["execution"]["monolithic_negfpy_recomputation"] is False
    assert report["post_comparison_immutability"]["passed"]
    assert report["post_comparison_immutability"][
        "repository_identity_rechecked"
    ]
    assert report["post_comparison_immutability"][
        "imported_code_identity_rechecked"
    ]
    persisted = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert persisted["certification_gates"]["no_monolithic_recomputation"]
    assert not persisted["release_certified"]
