from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import _bulk_target_q_mpi as target_q_mpi
from negfpy.anharmonic import _mpi_tasks as mpi_tasks
from negfpy.anharmonic._bulk_target_q_evidence import (
    BulkTargetQEvidenceError,
    verify_bulk_target_q_release_evidence,
)
from negfpy.anharmonic._bulk_target_q_mpi import (
    BulkTargetQControls,
    BulkTargetQMergeReport,
    build_bulk_target_q_campaign,
    bulk_three_phonon_problem_sha256,
    make_bulk_target_q_artifact_validator,
    make_bulk_target_q_worker,
    merge_bulk_target_q_campaign,
    write_bulk_target_q_merge_report,
)
from negfpy.anharmonic._mpi_tasks import (
    MPIExecutionReport,
    TaskContext,
    write_execution_manifest,
    write_task_checkpoint,
    write_task_manifest,
)
from negfpy.anharmonic.bulk_three_phonon import BulkThreePhononProblem
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.anharmonic.fc3 import FC3Data
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import (
    convert_fc3_to_translational,
)


def _mapping(purpose: str) -> CrystalMapping:
    cell = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0]],
        masses=[1.0],
        symbols=["X"],
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.eye(3, dtype=np.int64),
        primitive=cell,
        supercell=cell,
        p2s=[0],
        s2p=[0],
        p2p=[0],
        shortest_vectors=[[0.0, 0.0, 0.0]],
        multiplicity=[[[1, 0]]],
        atomic_permutations=[[0]],
        translation_permutations=[[0]],
        purpose=purpose,
    )


def _problem() -> BulkThreePhononProblem:
    fc2 = np.zeros((1, 1, 3, 3))
    fc2[0, 0] = np.diag([1.0, 4.0, 9.0])
    fc3 = np.zeros((1, 1, 1, 3, 3, 3))
    for axes in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        fc3[(0, 0, 0) + axes] = 0.2
    mapping3 = _mapping("fc3")
    return BulkThreePhononProblem(
        fc2=FC2Data(
            values=fc2,
            masses=[1.0],
            p2s_map=[0],
            s2p_map=[0],
        ),
        fc2_mapping=_mapping("fc2"),
        fc3=convert_fc3_to_translational(
            FC3Data(
                values=fc3,
                masses=[1.0],
                p2s_map=[0],
                s2p_map=[0],
            ),
            mapping3,
        ),
        fc3_mapping=mapping3,
        mesh=UniformBZMesh((3, 1, 1)),
        sigma_thz=0.05,
        bz_reduced_reciprocal_lattice=np.eye(3),
        cutoff_frequency_thz=1.0e-8,
        frequency_factor_to_thz=1.0,
    )


def _campaign() -> target_q_mpi.BulkTargetQCampaign:
    problem = _problem()
    controls = BulkTargetQControls(
        temperatures=[0.0],
        grid_points=[0],
        band_indices=[0],
        n_qpoints=3,
        n_bands=3,
        problem_sha256=bulk_three_phonon_problem_sha256(problem),
        input_hashes={"fc2": "a" * 64, "fc3": "b" * 64},
    )
    return build_bulk_target_q_campaign(
        controls,
        source_revision="target-q-evidence-test",
        source_tree_sha256="c" * 64,
        environment_sha256="d" * 64,
        metadata={"fixture": "target-q-evidence"},
    )


def _thread_cap(environment_value: str = "1") -> dict[str, object]:
    return {
        "requested_limit": 1,
        "environment": {
            "OMP_NUM_THREADS": environment_value,
            "OPENBLAS_NUM_THREADS": environment_value,
            "MKL_NUM_THREADS": environment_value,
            "NUMEXPR_NUM_THREADS": environment_value,
            "BLIS_NUM_THREADS": environment_value,
            "VECLIB_MAXIMUM_THREADS": environment_value,
        },
        "libraries": [],
        "all_loaded_libraries_capped": True,
        "cpu_affinity": [1],
        "cpu_affinity_topology": [
            {"logical_cpu": 1, "package_id": 0, "core_id": 1}
        ],
    }


def _archive(
    root: Path,
    *,
    orphan_artifact_adopted: bool = False,
    thread_environment_value: str = "1",
) -> tuple[
    target_q_mpi.BulkTargetQCampaign,
    MPIExecutionReport,
    Path,
    Path,
    Path,
]:
    campaign = _campaign()
    artifacts = root / "task-artifacts"
    outcomes = root / "task-outcomes"
    artifacts.mkdir(parents=True)
    outcomes.mkdir()
    write_task_manifest(root / "task-manifest.json", campaign.manifest)
    worker = make_bulk_target_q_worker(campaign, _problem)
    validator = make_bulk_target_q_artifact_validator(campaign)
    task = campaign.manifest.tasks[0]
    context = TaskContext(
        run_directory=root,
        artifact_root=artifacts,
        ordinal=0,
        task_identity_sha256=(
            campaign.manifest.task_identity_sha256(task)
        ),
        rank=1,
        communicator_size=2,
        hostname="worker-a",
        thread_limit=1,
        thread_cap_metadata=_thread_cap(thread_environment_value),
    )
    result = worker(task, context)
    checkpoint = mpi_tasks._checkpoint_from_result(  # noqa: SLF001
        campaign.manifest,
        task,
        result,
        context,
        elapsed_seconds=1.25,
        artifact_validator=validator,
        orphan_artifact_adopted=orphan_artifact_adopted,
    )
    checkpoint_path = mpi_tasks.task_checkpoint_path(
        campaign.manifest,
        task,
        outcomes,
    )
    write_task_checkpoint(checkpoint_path, checkpoint)
    validated = mpi_tasks.collect_task_checkpoints(
        campaign.manifest,
        outcomes,
        artifacts,
        artifact_validator=validator,
        require_complete=True,
    )
    assignments = mpi_tasks._assignments_from_checkpoints(  # noqa: SLF001
        campaign.manifest,
        validated,
        outcomes,
        initially_complete=set(),
    )
    report = MPIExecutionReport(
        execution_id="fresh-mpi",
        manifest_sha256=campaign.campaign_manifest_sha256,
        mode="mpi",
        communicator_size=2,
        coordinator_hostname="coordinator",
        started_utc="2026-07-29T00:00:00Z",
        finished_utc="2026-07-29T00:00:02Z",
        elapsed_seconds=2.0,
        thread_limit=1,
        assignments=assignments,
        missing_task_ids=(),
        infrastructure_errors=(),
    )
    execution_path = root / "executions" / "fresh-mpi.json"
    write_execution_manifest(execution_path, report)
    merged_path = root / "merged" / "bulk-target-q.h5"
    merge = merge_bulk_target_q_campaign(
        campaign,
        root,
        output_path=merged_path,
    )
    merge_report_path = root / "reports" / "merge.json"
    write_bulk_target_q_merge_report(merge_report_path, merge)
    return (
        campaign,
        report,
        execution_path,
        merge_report_path,
        merged_path,
    )


def _verify(
    archive: tuple[
        target_q_mpi.BulkTargetQCampaign,
        MPIExecutionReport,
        Path,
        Path,
        Path,
    ],
    *,
    report: MPIExecutionReport | None = None,
) -> dict[str, object]:
    campaign, selected, execution_path, merge_report, merged = archive
    return verify_bulk_target_q_release_evidence(
        campaign,
        execution_path.parents[1],
        execution_path,
        merge_report,
        merged,
        selected_execution_report=selected if report is None else report,
    )


def test_verifier_rebuilds_and_returns_json_serializable_evidence(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "run")
    evidence = _verify(archive)

    assert evidence["passed"] is True
    assert evidence["selected_execution"]["mode"] == "mpi"
    assert evidence["mpi_origin_lineage"]["passed"] is True
    assert evidence["physical_core_bindings"]["available"] is True
    assert evidence["independent_canonical_remerge"]["passed"] is True
    assert evidence["independent_canonical_remerge"][
        "temporary_artifact_retained"
    ] is False
    assert len(evidence["task_inventory"]) == 1
    json.dumps(evidence, allow_nan=False, sort_keys=True)
    assert not tuple(
        (tmp_path / "run" / "merged").glob(
            ".negfpy-target-q-remerge-*"
        )
    )


def test_verifier_accepts_resume_from_incomplete_mpi_history(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "run")
    campaign, fresh, fresh_path, merge_report, merged = archive
    fresh_path.unlink()
    partial = replace(
        fresh,
        execution_id="partial-mpi",
        missing_task_ids=(),
        infrastructure_errors=(
            {"error_type": "RuntimeError", "message": "interrupted"},
        ),
    )
    partial_path = fresh_path.with_name("partial-mpi.json")
    write_execution_manifest(partial_path, partial)
    resumed = replace(
        fresh,
        execution_id="resumed-mpi",
        assignments=(
            replace(fresh.assignments[0], resumed=True),
        ),
    )
    resumed_path = fresh_path.with_name("resumed-mpi.json")
    write_execution_manifest(resumed_path, resumed)

    evidence = verify_bulk_target_q_release_evidence(
        campaign,
        resumed_path.parents[1],
        resumed_path,
        merge_report,
        merged,
        selected_execution_report=resumed,
    )

    assert evidence["passed"] is True
    assert evidence["mpi_origin_lineage"]["lineage"][0][
        "origin_execution_id"
    ] == "partial-mpi"
    assert evidence["physical_core_bindings"][
        "origin_execution_scoped"
    ] is True


def test_verifier_rejects_selected_report_different_from_file(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "run")
    selected = archive[1]
    changed = replace(selected, elapsed_seconds=3.0)

    with pytest.raises(
        BulkTargetQEvidenceError,
        match="differs from its archived file",
    ):
        _verify(archive, report=changed)


@pytest.mark.parametrize(
    ("relative_directory", "filename"),
    [
        ("task-outcomes", "extra.task.json"),
        ("task-artifacts", "extra.h5"),
        ("executions", "extra.txt"),
    ],
)
def test_verifier_rejects_extra_evidence_files(
    tmp_path: Path,
    relative_directory: str,
    filename: str,
) -> None:
    archive = _archive(tmp_path / "run")
    root = archive[2].parents[1]
    (root / relative_directory / filename).write_bytes(b"extra")

    with pytest.raises(BulkTargetQEvidenceError, match="inventory|non-manifest"):
        _verify(archive)


def test_verifier_rejects_symlinked_artifact(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "run")
    root = archive[2].parents[1]
    artifact = next((root / "task-artifacts").glob("*.h5"))
    real = root / "artifact-copy.h5"
    artifact.rename(real)
    artifact.symlink_to(real)

    with pytest.raises(BulkTargetQEvidenceError, match="symlink"):
        _verify(archive)


def test_verifier_rejects_external_link_inside_task_hdf5(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "run")
    root = archive[2].parents[1]
    artifact = next((root / "task-artifacts").glob("*.h5"))
    with h5py.File(artifact, "r") as source:
        values = source["linewidths/gamma_thz_hwhm"][...]
    external = root / "external-values.h5"
    with h5py.File(external, "w") as destination:
        destination.create_dataset("values", data=values)
    artifact.chmod(0o644)
    with h5py.File(artifact, "r+") as handle:
        del handle["linewidths/gamma_thz_hwhm"]
        handle["linewidths/gamma_thz_hwhm"] = h5py.ExternalLink(
            external.as_posix(),
            "/values",
        )

    with pytest.raises(BulkTargetQEvidenceError, match="non-hard link"):
        _verify(archive)


def test_verifier_rejects_hardlinked_execution(tmp_path: Path) -> None:
    archive = _archive(tmp_path / "run")
    root = archive[2].parents[1]
    selected = archive[2]
    duplicate = root / "executions" / "duplicate.json"
    os.link(selected, duplicate)

    with pytest.raises(BulkTargetQEvidenceError, match="hard links|inode"):
        _verify(archive)


def test_verifier_rejects_orphan_only_origin_lineage(
    tmp_path: Path,
) -> None:
    archive = _archive(
        tmp_path / "run",
        orphan_artifact_adopted=True,
    )
    campaign, adopted, adopted_path, _, _ = archive
    root = adopted_path.parents[1]

    with pytest.raises(
        BulkTargetQEvidenceError,
        match="origin or physical-core",
    ):
        verify_bulk_target_q_release_evidence(
            campaign,
            root,
            adopted_path,
            archive[3],
            archive[4],
            selected_execution_report=adopted,
        )


def test_verifier_rejects_worker_thread_environment_mismatch(
    tmp_path: Path,
) -> None:
    archive = _archive(
        tmp_path / "run",
        thread_environment_value="2",
    )

    with pytest.raises(
        BulkTargetQEvidenceError,
        match="worker thread environment",
    ):
        _verify(archive)


def test_independent_remerge_rejects_self_consistent_modified_hdf5(
    tmp_path: Path,
) -> None:
    archive = _archive(tmp_path / "run")
    campaign, _, _, merge_report_path, merged_path = archive
    with h5py.File(merged_path, "r+") as handle:
        normal = np.asarray(
            handle["linewidths/gamma_normal_thz_hwhm"]
        )
        normal.flat[0] += 1.0e-12
        umklapp = np.asarray(
            handle["linewidths/gamma_umklapp_thz_hwhm"]
        )
        gamma = normal + umklapp
        angular = 4.0 * np.pi * gamma
        changes = {
            "linewidths/gamma_normal_thz_hwhm": normal,
            "linewidths/gamma_thz_hwhm": gamma,
            "linewidths/gamma_angular_fwhm": angular,
        }
        hashes = json.loads(handle.attrs["dataset_sha256_json"])
        for name, values in changes.items():
            dataset = handle[name]
            dataset[...] = values
            dataset_hash = target_q_mpi._dataset_sha256(  # noqa: SLF001
                dataset
            )
            dataset.attrs["sha256"] = dataset_hash
            hashes[name] = dataset_hash
        handle.attrs["dataset_sha256_json"] = target_q_mpi._canonical_json(  # noqa: SLF001
            hashes
        )
    declared = merge_bulk_target_q_campaign(
        campaign,
        merged_path.parents[1],
        output_path=merged_path,
    )
    merge_report_path.unlink()
    write_bulk_target_q_merge_report(
        merge_report_path,
        BulkTargetQMergeReport(
            campaign_manifest_sha256=declared.campaign_manifest_sha256,
            merged_artifact_path=merged_path,
            merged_artifact_sha256=target_q_mpi._file_sha256(  # noqa: SLF001
                merged_path
            ),
            task_ids=declared.task_ids,
            target_grid_points=declared.target_grid_points,
            task_artifact_sha256=declared.task_artifact_sha256,
            task_checkpoint_sha256=declared.task_checkpoint_sha256,
            dataset_sha256=hashes,
        ),
    )

    with pytest.raises(
        BulkTargetQEvidenceError,
        match=(
            "independently merged|declared merge report differs|"
            "HDF5 attribute"
        ),
    ):
        _verify(archive)


def test_disk_preflight_fails_before_temporary_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = _archive(tmp_path / "run")
    monkeypatch.setattr(
        "negfpy.anharmonic._bulk_target_q_evidence._MINIMUM_DISK_MARGIN_BYTES",
        10**30,
    )

    with pytest.raises(BulkTargetQEvidenceError, match="insufficient free disk"):
        _verify(archive)
