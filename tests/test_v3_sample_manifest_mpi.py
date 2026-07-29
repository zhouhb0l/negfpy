# ruff: noqa: E402

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

pytest.importorskip("spglib")

from negfpy.anharmonic._mpi_tasks import (
    MPIExecutionReport,
    TaskCheckpointCorruptionError,
    TaskManifestError,
    serial_task_communicator,
    write_execution_manifest,
)
from negfpy.anharmonic import _v3_sample_manifest_mpi as mpi_sampler
from negfpy.anharmonic._v3_sample_manifest_mpi import (
    V3SampleMPICampaign,
    current_v3_sampling_source_identity,
    run_v3_sample_manifest_mpi,
    validate_v3_mpi_orchestration_evidence,
)
from negfpy.anharmonic.phono3py_reference import (
    read_phono3py_silicon_reference,
)
from negfpy.anharmonic.v3_sample_manifest import (
    LARGE_MESH_BOUNDED_SELECTION,
    _canonical_extreme,
    _iter_promoted_blocks,
    _prepare_promoted_scan,
    _vectorized_maximum_promoted_block,
    create_v3_sample_manifest,
)
from test_v3_sample_manifest import _inputs, _sha256, _write_reference


def _thread_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")


def _campaign(tmp_path: Path) -> tuple[V3SampleMPICampaign, dict[str, object]]:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path)
    )
    inputs = _inputs(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    source = current_v3_sampling_source_identity(
        repository,
        require_clean=False,
    )
    campaign = V3SampleMPICampaign(
        reference=reference,
        fc2_compact_hdf5_path=Path(
            inputs["fc2_compact_hdf5_path"]
        ),
        fc2_full_hdf5_path=Path(inputs["fc2_full_hdf5_path"]),
        fc3_compact_hdf5_path=Path(
            inputs["fc3_compact_hdf5_path"]
        ),
        fc3_full_hdf5_path=Path(inputs["fc3_full_hdf5_path"]),
        phono3py_yaml_path=Path(inputs["phono3py_yaml_path"]),
        input_manifest_path=Path(inputs["input_manifest_path"]),
        fc2_mapping=inputs["fc2_mapping"],
        fc3_mapping=inputs["fc3_mapping"],
        source_revision=source["head_commit"],
        source_tree_sha256=source["source_tree_sha256"],
        environment_sha256="b" * 64,
        source_identity=source,
        require_authenticated_source=False,
        maximum_promoted_blocks=96,
    )
    return campaign, inputs


def _expected_candidate(
    campaign: V3SampleMPICampaign,
) -> dict[str, object]:
    return {
        "expected_source_revision": campaign.source_revision,
        "expected_source_tree_sha256": campaign.source_tree_sha256,
        "expected_environment_sha256": campaign.environment_sha256,
        "expected_source_identity": campaign.source_identity,
        "expected_mesh": campaign.reference.mesh,
        "expected_target_grid_points": campaign.target_grid_points,
        "expected_target_bands": campaign.target_bands,
    }


def test_serial_task_campaign_is_byte_identical_and_exactly_resumable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, inputs = _campaign(tmp_path)
    serial_path = tmp_path / "serial.json"
    mpi_path = tmp_path / "task.json"
    create_v3_sample_manifest(
        campaign.reference,
        output_path=serial_path,
        selection_mode=LARGE_MESH_BOUNDED_SELECTION,
        maximum_promoted_blocks=96,
        **inputs,
    )

    first = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=tmp_path / "run",
        output_path=mpi_path,
        comm=serial_task_communicator(),
    )
    assert first is not None
    assert mpi_path.read_bytes() == serial_path.read_bytes()
    assert first.manifest.sha256 == hashlib.sha256(
        mpi_path.read_bytes()
    ).hexdigest()
    source_evidence = json.loads(
        (tmp_path / "run" / "source-identity.json").read_text(
            encoding="ascii"
        )
    )
    assert source_evidence["campaign_sha256"] == campaign.campaign_sha256
    assert source_evidence["source_identity"] == campaign.source_identity
    assert source_evidence["source_identity_sha256"] == hashlib.sha256(
        json.dumps(
            campaign.source_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    orchestration = validate_v3_mpi_orchestration_evidence(
        mpi_path,
        first.orchestration_path,
        tmp_path / "run",
        require_authenticated_source=False,
        require_mpi=False,
    )
    assert orchestration.file_sha256 == first.orchestration_file_sha256
    assert (
        orchestration.payload_sha256
        == first.orchestration_payload_sha256
    )
    assert orchestration.selected_modes == ("serial", "serial")
    assert not orchestration.restart_resume_recorded
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="requires the explicit current candidate identity",
    ):
        validate_v3_mpi_orchestration_evidence(
            mpi_path,
            first.orchestration_path,
            tmp_path / "run",
            require_authenticated_source=True,
            require_mpi=False,
        )
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="complete mesh",
    ):
        validate_v3_mpi_orchestration_evidence(
            mpi_path,
            first.orchestration_path,
            tmp_path / "run",
            require_authenticated_source=True,
            require_mpi=False,
            **_expected_candidate(campaign),
        )

    second = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=tmp_path / "run",
        output_path=mpi_path,
        comm=serial_task_communicator(),
    )
    assert second is not None
    assert second.manifest.sha256 == first.manifest.sha256
    assert all(
        assignment.resumed
        for assignment in second.phase_maximum_execution.assignments
    )
    assert all(
        assignment.resumed
        for assignment in second.phase_reservoir_execution.assignments
    )
    restarted_evidence = validate_v3_mpi_orchestration_evidence(
        mpi_path,
        second.orchestration_path,
        tmp_path / "run",
        require_authenticated_source=False,
        require_mpi=False,
    )
    assert restarted_evidence.restart_resume_recorded
    assert second.orchestration_path != first.orchestration_path


def test_rank_zero_maximum_merge_failure_has_canonical_v3_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)

    def fail_maximum_merge(_values: object) -> object:
        raise RuntimeError("injected rank-zero maximum merge failure")

    monkeypatch.setattr(
        mpi_sampler,
        "_merge_maxima",
        fail_maximum_merge,
    )
    with pytest.raises(
        mpi_sampler.V3SampleMPIMergeError,
        match=(
            "rank-zero maximum-phase artifact/merge failed: RuntimeError: "
            "injected rank-zero maximum merge failure"
        ),
    ):
        run_v3_sample_manifest_mpi(
            campaign,
            run_directory=tmp_path / "run",
            output_path=tmp_path / "samples.json",
            comm=serial_task_communicator(),
        )


def test_orchestration_accepts_completed_tasks_from_interrupted_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    output = tmp_path / "samples.json"
    first = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert first is not None

    phases = (
        ("phase-maximum", first.phase_maximum_execution),
        ("phase-reservoirs", first.phase_reservoir_execution),
    )
    for phase_name, complete in phases:
        execution_directory = run / phase_name / "executions"
        (execution_directory / f"{complete.execution_id}.json").unlink()
        interrupted = MPIExecutionReport(
            execution_id=f"interrupted-{phase_name}",
            manifest_sha256=complete.manifest_sha256,
            mode=complete.mode,
            communicator_size=complete.communicator_size,
            coordinator_hostname=complete.coordinator_hostname,
            started_utc=complete.started_utc,
            finished_utc=complete.finished_utc,
            elapsed_seconds=complete.elapsed_seconds,
            thread_limit=complete.thread_limit,
            assignments=complete.assignments,
            missing_task_ids=(),
            infrastructure_errors=(
                {
                    "task_id": "<protocol>",
                    "rank": 0,
                    "error_type": (
                        "InterruptedAfterCheckpointPublication"
                    ),
                    "error_message": (
                        "synthetic post-publication interruption"
                    ),
                },
            ),
        )
        assert not interrupted.complete
        write_execution_manifest(
            execution_directory / f"{interrupted.execution_id}.json",
            interrupted,
        )
        phase_root = run / phase_name
        residue_targets = (
            (
                phase_root,
                "task-manifest.json",
            ),
            (
                phase_root / "task-artifacts",
                next(
                    (phase_root / "task-artifacts").iterdir()
                ).name,
            ),
            (
                phase_root / "task-outcomes",
                next(
                    (phase_root / "task-outcomes").iterdir()
                ).name,
            ),
            (
                phase_root / "task-assignments",
                next(
                    (phase_root / "task-assignments").iterdir()
                ).name,
            ),
            (
                phase_root / "executions",
                "interrupted-publication.json",
            ),
        )
        for directory, target_name in residue_targets:
            (
                directory
                / f".{target_name}.{('0' * 32)}.tmp"
            ).write_bytes(b"interrupted")
    (
        run
        / f".source-identity.json.{('1' * 32)}.tmp"
    ).write_bytes(b"interrupted")
    scientific_residue = (
        output.parent
        / f".{output.name}.{('2' * 32)}.tmp"
    )
    scientific_residue.write_bytes(b"interrupted")
    orchestration_residue = (
        first.orchestration_path.parent
        / f".{first.orchestration_path.name}.{('3' * 32)}.tmp"
    )
    orchestration_residue.write_bytes(b"interrupted")

    resumed = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert resumed is not None
    evidence = validate_v3_mpi_orchestration_evidence(
        output,
        resumed.orchestration_path,
        run,
        require_authenticated_source=False,
        require_mpi=False,
    )
    assert evidence.restart_resume_recorded
    assert not list(run.rglob(".*.tmp"))
    assert not scientific_residue.exists()
    assert not orchestration_residue.exists()


def test_corrupt_task_artifact_is_rejected_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=tmp_path / "samples.json",
        comm=serial_task_communicator(),
    )
    assert result is not None
    artifact = sorted(
        (run / "phase-maximum" / "task-artifacts").glob("*.json")
    )[0]
    artifact.write_bytes(b"{\"truncated\":true}\n")

    with pytest.raises(TaskCheckpointCorruptionError):
        run_v3_sample_manifest_mpi(
            campaign,
            run_directory=run,
            output_path=tmp_path / "samples.json",
            comm=serial_task_communicator(),
        )


def test_changed_source_identity_evidence_is_rejected_on_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=tmp_path / "samples.json",
        comm=serial_task_communicator(),
    )
    assert result is not None
    evidence = run / "source-identity.json"
    evidence.write_text('{"changed":true}\n', encoding="ascii")

    with pytest.raises(
        TaskManifestError,
        match="immutable V3 source identity evidence",
    ):
        run_v3_sample_manifest_mpi(
            campaign,
            run_directory=run,
            output_path=tmp_path / "samples.json",
            comm=serial_task_communicator(),
        )


def test_orchestration_rejects_tampered_task_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    output = tmp_path / "samples.json"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert result is not None
    artifact = sorted(
        (run / "phase-reservoirs" / "task-artifacts").glob("*.json")
    )[0]
    artifact.write_bytes(artifact.read_bytes() + b" ")

    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="changed",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )


def test_orchestration_rejects_tampered_prework_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    output = tmp_path / "samples.json"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert result is not None
    origin = sorted(
        (
            run
            / "phase-maximum"
            / "task-assignments"
        ).glob("*.assignment.json")
    )[0]
    origin.write_bytes(origin.read_bytes() + b" ")

    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="changed|canonical|origin",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )


def _write_orchestration_variant(
    original: Path,
    output: Path,
    mutate,
) -> Path:
    document = json.loads(original.read_text(encoding="ascii"))
    mutate(document["payload"])
    digest = hashlib.sha256(
        json.dumps(
            document["payload"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    document["payload_sha256"] = digest
    path = output.with_name(
        f"{output.name}.mpi-orchestration.{digest}.json"
    )
    path.write_text(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    return path


def test_orchestration_rejects_minimal_and_traversal_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    output = tmp_path / "samples.json"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert result is not None

    minimal = _write_orchestration_variant(
        result.orchestration_path,
        output,
        lambda payload: payload.pop("phases"),
    )
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="schema is invalid",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            minimal,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )

    def traverse(payload):
        payload["source_identity_file"]["relative_path"] = (
            "../source-identity.json"
        )

    traversal = _write_orchestration_variant(
        result.orchestration_path,
        output,
        traverse,
    )
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="traverses",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            traversal,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )


def test_orchestration_binds_contract_candidate_target_and_affinity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    output = tmp_path / "samples.json"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert result is not None
    evidence = validate_v3_mpi_orchestration_evidence(
        output,
        result.orchestration_path,
        run,
        require_authenticated_source=False,
        require_mpi=False,
        **_expected_candidate(campaign),
    )
    assert evidence.source_revision == campaign.source_revision
    assert evidence.source_tree_sha256 == campaign.source_tree_sha256
    assert evidence.environment_sha256 == campaign.environment_sha256
    assert evidence.mesh == campaign.reference.mesh
    assert evidence.target_grid_points == campaign.target_grid_points
    assert evidence.target_bands == campaign.target_bands
    assert evidence.physical_core_binding_passed is False
    assert evidence.mpi_origin_lineage_passed is False

    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="source revision is not the expected current candidate",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
            **{
                **_expected_candidate(campaign),
                "expected_source_revision": "f" * 40,
            },
        )
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="target q points",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
            **{
                **_expected_candidate(campaign),
                "expected_target_grid_points": tuple(
                    range(np.prod(campaign.reference.mesh))
                ),
            },
        )
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="target bands",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
            **{
                **_expected_candidate(campaign),
                "expected_target_bands": campaign.target_bands[:-1],
            },
        )

    def stale_contract(payload):
        payload["execution_contract"]["mpi_messages"] = (
            "task_identifiers_and_status_only"
        )

    stale = _write_orchestration_variant(
        result.orchestration_path,
        output,
        stale_contract,
    )
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="execution contract is not pinned",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            stale,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )

    def tamper_affinity(payload):
        payload["phases"][0][
            "selected_execution_physical_cores"
        ]["available"] = True

    tampered = _write_orchestration_variant(
        result.orchestration_path,
        output,
        tamper_affinity,
    )
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="physical-core evidence was tampered",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            tampered,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )


def test_symlinked_campaign_and_orchestration_paths_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    input_link = tmp_path / "fc2-compact-link.h5"
    input_link.symlink_to(campaign.fc2_compact_hdf5_path)
    with pytest.raises(ValueError, match="symlink"):
        replace(campaign, fc2_compact_hdf5_path=input_link)

    run = tmp_path / "run"
    output = tmp_path / "samples.json"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert result is not None
    scientific_link = tmp_path / "scientific-link.json"
    scientific_link.symlink_to(output)
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="symlink",
    ):
        validate_v3_mpi_orchestration_evidence(
            scientific_link,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )
    run_link = tmp_path / "run-link"
    run_link.symlink_to(run, target_is_directory=True)
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="symlink",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run_link,
            require_authenticated_source=False,
            require_mpi=False,
        )

    artifact = sorted(
        (run / "phase-maximum" / "task-artifacts").glob("*.json")
    )[0]
    moved = tmp_path / f"moved-{artifact.name}"
    artifact.rename(moved)
    artifact.symlink_to(moved)
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="symlink",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )


def test_v3_rejects_unowned_root_entries_and_hardlinked_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    sentinel = run / "foreign-entry"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(TaskManifestError, match="unowned top-level"):
        run_v3_sample_manifest_mpi(
            campaign,
            run_directory=run,
            output_path=tmp_path / "blocked.json",
            comm=serial_task_communicator(),
        )
    assert sentinel.read_bytes() == b"preserve"
    assert {path.name for path in run.iterdir()} == {"foreign-entry"}

    sentinel.unlink()
    output = tmp_path / "samples.json"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=output,
        comm=serial_task_communicator(),
    )
    assert result is not None
    alias = tmp_path / "orchestration-hardlink.json"
    os.link(result.orchestration_path, alias)
    with pytest.raises(
        mpi_sampler.V3SampleMPIArtifactError,
        match="single-link",
    ):
        validate_v3_mpi_orchestration_evidence(
            output,
            result.orchestration_path,
            run,
            require_authenticated_source=False,
            require_mpi=False,
        )


def test_thread_environment_is_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    campaign, _ = _campaign(tmp_path)
    with pytest.raises(RuntimeError, match="thread counts exactly to 1"):
        run_v3_sample_manifest_mpi(
            campaign,
            run_directory=tmp_path / "run",
            output_path=tmp_path / "samples.json",
            comm=serial_task_communicator(),
        )


def test_mismatched_campaign_is_rejected_before_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    run = tmp_path / "run"
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=run,
        output_path=tmp_path / "samples.json",
        comm=serial_task_communicator(),
    )
    assert result is not None
    mismatched = replace(campaign, environment_sha256="c" * 64)
    with pytest.raises(TaskManifestError):
        run_v3_sample_manifest_mpi(
            mismatched,
            run_directory=run,
            output_path=tmp_path / "samples.json",
            comm=serial_task_communicator(),
        )


def test_authenticated_source_rejects_dirty_git_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    repository = Path(campaign.source_identity["repository"])

    def fake_git(root: Path, *arguments: str) -> str:
        assert root == repository
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repository)
        if arguments == ("rev-parse", "HEAD"):
            return campaign.source_revision
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return campaign.source_identity["git_tree"]
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ):
            return " M src/negfpy/anharmonic/v3_sample_manifest.py"
        raise AssertionError(arguments)

    monkeypatch.setattr(mpi_sampler, "_git", fake_git)
    monkeypatch.setattr(
        mpi_sampler,
        "code_provenance",
        lambda: {
            **campaign.source_identity["imported_code_provenance"],
            "git_revision": campaign.source_revision,
            "source_tree_sha256": campaign.source_tree_sha256,
            "git_dirty": False,
            "package_version_consistent": True,
        },
    )
    with pytest.raises(
        mpi_sampler.V3SampleMPICampaignError,
        match="requires a clean Git worktree",
    ):
        current_v3_sampling_source_identity(
            repository,
            require_clean=True,
        )


def test_source_declaration_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    campaign, _ = _campaign(tmp_path)
    with pytest.raises(ValueError, match="declared source"):
        replace(campaign, source_revision="f" * 40)


def test_post_run_source_mutation_prevents_final_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _thread_environment(monkeypatch)
    campaign, _ = _campaign(tmp_path)
    calls = 0

    def changing_source(
        repository: str | Path,
        *,
        require_clean: bool,
    ) -> dict[str, object]:
        nonlocal calls
        del repository, require_clean
        calls += 1
        identity = json.loads(json.dumps(campaign.source_identity))
        if calls > 1:
            identity["code_files"][
                "src/negfpy/anharmonic/v3_sample_manifest.py"
            ] = "f" * 64
        return identity

    monkeypatch.setattr(
        mpi_sampler,
        "current_v3_sampling_source_identity",
        changing_source,
    )
    output = tmp_path / "samples.json"
    with pytest.raises(
        mpi_sampler.V3SampleMPICampaignError,
        match="identity changed during the campaign",
    ):
        run_v3_sample_manifest_mpi(
            campaign,
            run_directory=tmp_path / "run",
            output_path=output,
            comm=serial_task_communicator(),
        )
    assert calls == 2
    assert not output.exists()


def test_vectorized_maximum_matches_exhaustive_degenerate_blocks(
    tmp_path: Path,
) -> None:
    reference_path = _write_reference(tmp_path)
    with h5py.File(reference_path, "r+") as handle:
        frequencies = np.asarray(
            handle["phonons/frequencies"], dtype=np.float64
        )
        frequencies[:, :3] = frequencies[:, :1]
        frequencies[:, 3:] = frequencies[:, 3:4]
        handle["phonons/frequencies"][...] = frequencies
        targets = np.asarray(
            handle["settings/grid_points_grg"], dtype=np.int64
        )
        handle["phonons/target_band_frequencies"][...] = frequencies[
            targets
        ]
    (tmp_path / "SHA256SUMS").write_text(
        f"{_sha256(reference_path)}  {reference_path.name}\n",
        encoding="utf-8",
    )
    reference = read_phono3py_silicon_reference(reference_path)
    lattice = np.eye(3)
    context = _prepare_promoted_scan(
        reference,
        degeneracy_tolerance_thz=1.0e-4,
        primitive_lattice=lattice,
    )
    vectorized = _vectorized_maximum_promoted_block(
        reference,
        degeneracy_tolerance_thz=1.0e-4,
        primitive_lattice=lattice,
        scan_context=context,
    )
    exhaustive = None
    for candidate in _iter_promoted_blocks(
        reference,
        degeneracy_tolerance_thz=1.0e-4,
        primitive_lattice=lattice,
        track_duplicates=False,
        scan_context=context,
    ):
        exhaustive = _canonical_extreme(
            exhaustive, candidate, maximum=True
        )
    assert vectorized == exhaustive


@pytest.mark.nightly
@pytest.mark.skipif(
    os.environ.get("NEGFPY_RUN_MPI_INTEGRATION") != "1",
    reason="set NEGFPY_RUN_MPI_INTEGRATION=1 for real MPI integration",
)
def test_real_two_rank_equality_and_four_rank_restart(
    tmp_path: Path,
) -> None:
    mpiexec = Path(sys.executable).resolve().parent / "mpiexec"
    if (
        not mpiexec.is_file()
        or importlib.util.find_spec("mpi4py") is None
    ):
        pytest.skip("the active environment lacks local MPI/mpi4py")
    script = (
        Path(__file__).parent / "mpi" / "v3_sample_manifest_smoke.py"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    initial = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "2",
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "--phase",
            "initial",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert initial.returncode == 0, initial.stdout + initial.stderr
    assert "v3-sample-mpi-initial-passed ranks=2" in initial.stdout
    for phase in ("phase-maximum", "phase-reservoirs"):
        execution_directory = tmp_path / "run" / phase / "executions"
        for execution_path in execution_directory.glob("*.json"):
            execution_path.unlink()

    restarted = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "4",
            sys.executable,
            str(script),
            "--root",
            str(tmp_path),
            "--phase",
            "restart",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert "v3-sample-mpi-restart-passed ranks=4" in restarted.stdout

    merge_failure = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "2",
            sys.executable,
            str(script),
            "--root",
            str(tmp_path / "merge-failure"),
            "--phase",
            "merge-failure",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=90,
    )
    assert (
        merge_failure.returncode == 0
    ), merge_failure.stdout + merge_failure.stderr
    assert (
        "v3-sample-mpi-merge-failure-passed ranks=2"
        in merge_failure.stdout
    )
