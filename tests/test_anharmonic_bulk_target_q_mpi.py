from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import _bulk_target_q_mpi as target_q_mpi
from negfpy.anharmonic._bulk_target_q_mpi import (
    BulkTargetQControls,
    bulk_three_phonon_problem_sha256,
    build_bulk_target_q_campaign,
    compare_bulk_target_q_with_serial,
    execute_bulk_target_q_campaign,
    load_bulk_target_q_result,
    make_bulk_target_q_artifact_validator,
    make_bulk_target_q_worker,
    merge_bulk_target_q_campaign,
)
from negfpy.anharmonic._mpi_tasks import (
    MissingTaskCheckpointError,
    TaskCheckpointMismatchError,
    TaskExecutionError,
    collect_task_checkpoints,
    execute_task_manifest,
    serial_task_communicator,
    task_checkpoint_path,
)
from negfpy.anharmonic.bulk_three_phonon import (
    BulkThreePhononLinewidths,
    BulkThreePhononProblem,
    compute_three_phonon_linewidths,
)
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.anharmonic.fc3 import FC3Data
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import (
    convert_fc3_to_translational,
)


_SOURCE_REVISION = "bulk-target-q-mpi-test-v1"
_SOURCE_TREE_SHA256 = "a" * 64
_ENVIRONMENT_SHA256 = "b" * 64


def _sha(character: str) -> str:
    assert len(character) == 1
    return character * 64


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
    """Return the bounded three-mode fixture used by the serial bulk tests."""

    fc2 = np.zeros((1, 1, 3, 3))
    fc2[0, 0] = np.diag([1.0, 4.0, 9.0])
    fc2_data = FC2Data(
        values=fc2,
        masses=[1.0],
        p2s_map=[0],
        s2p_map=[0],
    )
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
        fc2=fc2_data,
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


def _controls(
    *,
    grid_points: object = (2, 0, 1),
    band_indices: object = (2, 0),
    temperatures: object = (0.0, 300.0),
    n_qpoints: object = 3,
    n_bands: object = 3,
    problem_sha256: object | None = None,
    input_hashes: object | None = None,
) -> BulkTargetQControls:
    selected_problem_sha256 = (
        bulk_three_phonon_problem_sha256(_problem())
        if problem_sha256 is None
        else problem_sha256
    )
    return BulkTargetQControls(
        temperatures=temperatures,
        grid_points=grid_points,
        band_indices=band_indices,
        n_qpoints=n_qpoints,
        n_bands=n_bands,
        problem_sha256=selected_problem_sha256,
        input_hashes=(
            {"fc2": _sha("d"), "fc3": _sha("e")}
            if input_hashes is None
            else input_hashes
        ),
    )


def _campaign(
    controls: BulkTargetQControls | None = None,
    *,
    environment_sha256: str = _ENVIRONMENT_SHA256,
):
    return build_bulk_target_q_campaign(
        _controls() if controls is None else controls,
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=environment_sha256,
        metadata={"fixture": "bounded-three-mode", "certifying": False},
    )


def _assert_all_scientific_fields_equal(
    actual: BulkThreePhononLinewidths,
    expected: BulkThreePhononLinewidths,
) -> None:
    np.testing.assert_array_equal(actual.temperatures, expected.temperatures)
    np.testing.assert_array_equal(actual.grid_points, expected.grid_points)
    np.testing.assert_array_equal(actual.band_indices, expected.band_indices)
    for name in (
        "gamma_thz_hwhm",
        "gamma_normal_thz_hwhm",
        "gamma_umklapp_thz_hwhm",
        "gamma_angular_fwhm",
        "decay_phase_space",
        "difference_phase_space",
    ):
        np.testing.assert_array_equal(
            getattr(actual, name),
            getattr(expected, name),
        )

    actual_interactions = actual.interactions
    expected_interactions = expected.interactions
    for name in (
        "grid_points",
        "band_indices",
        "q1_indices",
        "q2_indices",
        "reciprocal_vectors",
        "normal_mask",
        "matrix_elements",
        "interaction_strengths",
    ):
        np.testing.assert_array_equal(
            getattr(actual_interactions, name),
            getattr(expected_interactions, name),
        )

    actual_phonons = actual_interactions.phonons
    expected_phonons = expected_interactions.phonons
    for name in (
        "qpoints",
        "dynamical_matrices",
        "frequencies",
        "eigenvectors",
    ):
        np.testing.assert_array_equal(
            getattr(actual_phonons, name),
            getattr(expected_phonons, name),
        )
    assert actual_phonons.frequency_factor_to_thz == (
        expected_phonons.frequency_factor_to_thz
    )
    assert actual_phonons.hermitianized is expected_phonons.hermitianized


def test_controls_and_manifest_are_immutable_canonical_and_deterministic() -> None:
    temperatures = [0.0, 300.0]
    targets = [2, 0, 1]
    bands = [2, 0]
    hashes = {"fc3": _sha("e"), "fc2": _sha("d")}
    controls = _controls(
        temperatures=temperatures,
        grid_points=targets,
        band_indices=bands,
        input_hashes=hashes,
    )
    temperatures.append(900.0)
    targets.reverse()
    bands.append(1)
    hashes["fc2"] = _sha("f")

    np.testing.assert_array_equal(controls.temperatures, [0.0, 300.0])
    np.testing.assert_array_equal(controls.grid_points, [0, 1, 2])
    np.testing.assert_array_equal(controls.band_indices, [0, 2])
    assert not controls.temperatures.flags.writeable
    assert not controls.grid_points.flags.writeable
    assert not controls.band_indices.flags.writeable
    assert dict(controls.input_hashes) == {
        "fc2": _sha("d"),
        "fc3": _sha("e"),
    }

    first = _campaign(controls)
    second = build_bulk_target_q_campaign(
        _controls(
            grid_points=[1, 2, 0],
            band_indices=[0, 2],
            input_hashes={"fc2": _sha("d"), "fc3": _sha("e")},
        ),
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=_ENVIRONMENT_SHA256,
        metadata={"certifying": False, "fixture": "bounded-three-mode"},
    )
    assert first.manifest.manifest_sha256 == second.manifest.manifest_sha256
    assert first.manifest.run_identity_sha256 == (
        second.manifest.run_identity_sha256
    )
    assert [task.task_id for task in first.manifest.tasks] == [
        "target-q-00000000",
        "target-q-00000001",
        "target-q-00000002",
    ]
    assert [task.payload["target_q"] for task in first.manifest.tasks] == [
        0,
        1,
        2,
    ]


@pytest.mark.parametrize(
    ("changes", "error"),
    (
        ({"grid_points": [0, 0]}, ValueError),
        ({"grid_points": [3]}, IndexError),
        ({"grid_points": [True]}, TypeError),
        ({"grid_points": []}, ValueError),
        ({"band_indices": [1, 1]}, ValueError),
        ({"band_indices": [-1]}, IndexError),
        ({"temperatures": [-1.0]}, ValueError),
        ({"temperatures": [np.nan]}, ValueError),
        ({"n_qpoints": True}, TypeError),
        ({"n_qpoints": 0}, ValueError),
        ({"n_bands": 0}, ValueError),
        ({"problem_sha256": "not-a-sha"}, ValueError),
        ({"input_hashes": {"fc2": "not-a-sha"}}, ValueError),
    ),
)
def test_controls_reject_invalid_selectors_and_identities(
    changes: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        _controls(**changes)


def test_worker_calls_unchanged_serial_kernel_once_per_single_q(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    campaign = _campaign(
        _controls(grid_points=[2, 0], band_indices=[2, 0])
    )
    calls: list[dict[str, object]] = []
    original = target_q_mpi.compute_three_phonon_linewidths

    def recording_kernel(problem_arg, **kwargs):
        calls.append(
            {
                "problem": problem_arg,
                "temperatures": np.array(
                    kwargs["temperatures"], copy=True
                ),
                "grid_points": np.array(
                    kwargs["grid_points"], copy=True
                ),
                "band_indices": np.array(
                    kwargs["band_indices"], copy=True
                ),
            }
        )
        return original(problem_arg, **kwargs)

    monkeypatch.setattr(
        target_q_mpi,
        "compute_three_phonon_linewidths",
        recording_kernel,
    )
    report = execute_bulk_target_q_campaign(
        campaign,
        lambda: problem,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="single-q-contract",
    )

    assert report is not None and report.complete
    assert len(calls) == 2
    assert [call["grid_points"].tolist() for call in calls] == [[0], [2]]
    for call in calls:
        assert call["problem"] is problem
        np.testing.assert_array_equal(
            call["temperatures"],
            campaign.controls.temperatures,
        )
        np.testing.assert_array_equal(
            call["band_indices"],
            campaign.controls.band_indices,
        )


def test_hdf5_round_trip_canonical_merge_and_all_fields_equal_serial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    campaign = _campaign()
    execution = execute_bulk_target_q_campaign(
        campaign,
        lambda: problem,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="complete",
    )
    assert execution is not None and execution.complete

    original_collect = target_q_mpi.collect_task_checkpoints

    def reversed_collect(*args, **kwargs):
        return tuple(reversed(original_collect(*args, **kwargs)))

    monkeypatch.setattr(
        target_q_mpi,
        "collect_task_checkpoints",
        reversed_collect,
    )
    merge = merge_bulk_target_q_campaign(campaign, tmp_path)
    merged = load_bulk_target_q_result(
        merge.merged_artifact_path,
        campaign=campaign,
    )
    serial = compute_three_phonon_linewidths(
        problem,
        temperatures=campaign.controls.temperatures,
        grid_points=campaign.controls.grid_points,
        band_indices=campaign.controls.band_indices,
    )
    _assert_all_scientific_fields_equal(merged, serial)
    comparison = compare_bulk_target_q_with_serial(serial, merged)
    assert comparison.passed
    assert comparison.serial_result_sha256 == comparison.merged_result_sha256
    assert comparison.merged_artifact_sha256 is None
    path_comparison = compare_bulk_target_q_with_serial(
        serial,
        merge.merged_artifact_path,
    )
    assert path_comparison.passed
    assert path_comparison.serial_result_sha256 == (
        path_comparison.merged_result_sha256
    )
    assert path_comparison.merged_artifact_sha256 == hashlib.sha256(
        merge.merged_artifact_path.read_bytes()
    ).hexdigest()
    assert all(
        not getattr(merged, name).flags.writeable
        for name in (
            "temperatures",
            "grid_points",
            "band_indices",
            "gamma_thz_hwhm",
            "gamma_normal_thz_hwhm",
            "gamma_umklapp_thz_hwhm",
            "gamma_angular_fwhm",
            "decay_phase_space",
            "difference_phase_space",
        )
    )


def test_hdf5_corruption_mismatch_truncation_and_immutable_output_rejected(
    tmp_path: Path,
) -> None:
    problem = _problem()
    campaign = _campaign(_controls(grid_points=[0]))
    execute_bulk_target_q_campaign(
        campaign,
        lambda: problem,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="complete",
    )
    output = tmp_path / "merged.h5"
    merge = merge_bulk_target_q_campaign(
        campaign,
        tmp_path,
        output_path=output,
    )
    original = output.read_bytes()
    same = merge_bulk_target_q_campaign(
        campaign,
        tmp_path,
        output_path=output,
    )
    assert same.merged_artifact_path == merge.merged_artifact_path
    assert output.read_bytes() == original

    corrupt = tmp_path / "corrupt.h5"
    shutil.copyfile(output, corrupt)
    with h5py.File(corrupt, "r+") as handle:
        values = handle["linewidths/gamma_thz_hwhm"]
        values[0, 0, 0] += 1.0
    with pytest.raises(target_q_mpi.BulkTargetQArtifactError):
        load_bulk_target_q_result(corrupt, campaign=campaign)

    mismatch = _campaign(
        _controls(grid_points=[0]),
        environment_sha256=_sha("f"),
    )
    with pytest.raises(TaskCheckpointMismatchError):
        load_bulk_target_q_result(output, campaign=mismatch)

    truncated = tmp_path / "truncated.h5"
    truncated.write_bytes(original[: max(1, len(original) // 3)])
    with pytest.raises(target_q_mpi.BulkTargetQArtifactError):
        load_bulk_target_q_result(truncated, campaign=campaign)

    with h5py.File(output, "r+") as handle:
        handle["linewidths/gamma_thz_hwhm"][0, 0, 0] += 1.0
    with pytest.raises(target_q_mpi.BulkTargetQArtifactError):
        merge_bulk_target_q_campaign(
            campaign,
            tmp_path,
            output_path=output,
        )


def test_valid_orphan_is_adopted_without_recomputing_kernel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    campaign = _campaign(_controls(grid_points=[1]))
    worker = make_bulk_target_q_worker(campaign, lambda: problem)
    validator = make_bulk_target_q_artifact_validator(campaign)

    def publish_then_crash(task, context):
        result = worker(task, context)
        assert result.artifact_path is not None
        raise RuntimeError("simulated interruption after atomic HDF5 publish")

    with pytest.raises(TaskExecutionError):
        execute_task_manifest(
            campaign.manifest,
            publish_then_crash,
            tmp_path,
            comm=serial_task_communicator(),
            artifact_validator=validator,
            execution_id="publish-then-crash",
        )
    orphan = next((tmp_path / "task-artifacts").glob("*.h5"))
    original_bytes = orphan.read_bytes()
    original_mtime = orphan.stat().st_mtime_ns

    def forbidden_kernel(*args, **kwargs):
        raise AssertionError("a validated orphan was recomputed")

    monkeypatch.setattr(
        target_q_mpi,
        "compute_three_phonon_linewidths",
        forbidden_kernel,
    )
    resumed = execute_bulk_target_q_campaign(
        campaign,
        lambda: problem,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="adopt-orphan",
    )
    assert resumed is not None and resumed.complete
    assert len(resumed.assignments) == 1
    assignment = resumed.assignments[0]
    assert assignment.resumed
    assert assignment.orphan_artifact_adopted
    assert assignment.as_dict()["orphan_artifact_adopted"] is True
    assert orphan.read_bytes() == original_bytes
    assert orphan.stat().st_mtime_ns == original_mtime
    checkpoint = collect_task_checkpoints(
        campaign.manifest,
        tmp_path / "task-outcomes",
        tmp_path / "task-artifacts",
        artifact_validator=validator,
    )[0]
    assert checkpoint.execution["orphan_artifact_adopted"] is True


def test_missing_target_checkpoint_rejects_merge(tmp_path: Path) -> None:
    problem = _problem()
    campaign = _campaign(_controls(grid_points=[0, 2]))
    execute_bulk_target_q_campaign(
        campaign,
        lambda: problem,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="complete-before-loss",
    )
    missing = campaign.manifest.tasks[-1]
    task_checkpoint_path(
        campaign.manifest,
        missing,
        tmp_path / "task-outcomes",
    ).unlink()
    with pytest.raises(MissingTaskCheckpointError) as captured:
        merge_bulk_target_q_campaign(campaign, tmp_path)
    assert captured.value.missing_task_ids == (missing.task_id,)


def _environment_local_mpiexec() -> Path | None:
    candidate = Path(sys.executable).resolve().parent / "mpiexec"
    if candidate.is_file() and importlib.util.find_spec("mpi4py") is not None:
        return candidate
    return None


@pytest.mark.nightly
@pytest.mark.skipif(
    os.environ.get("NEGFPY_RUN_MPI_INTEGRATION") != "1",
    reason="set NEGFPY_RUN_MPI_INTEGRATION=1 for real MPI integration",
)
def test_real_mpi_target_q_execution_and_changed_count_restart(
    tmp_path: Path,
) -> None:
    mpiexec = _environment_local_mpiexec()
    if mpiexec is None:
        pytest.skip("the active Python environment lacks local MPI/mpi4py")
    script = Path(__file__).parent / "mpi" / "bulk_target_q_smoke.py"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    source_root = Path(__file__).resolve().parents[1] / "src"
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) + (
        "" if not current_pythonpath else os.pathsep + current_pythonpath
    )
    initial = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "2",
            sys.executable,
            str(script),
            str(tmp_path),
            "initial",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert initial.returncode == 0, initial.stdout + initial.stderr
    assert '"all_fields_equal": true' in initial.stdout
    assert '"communicator_size": 2' in initial.stdout

    restarted = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "4",
            sys.executable,
            str(script),
            str(tmp_path),
            "restart",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert '"all_resumed": true' in restarted.stdout
    assert '"communicator_size": 4' in restarted.stdout

    orphan = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "2",
            sys.executable,
            str(script),
            str(tmp_path / "orphan"),
            "orphan",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert orphan.returncode == 0, orphan.stdout + orphan.stderr
    assert '"adopted_marked_resumed": true' in orphan.stdout
    assert '"communicator_size": 2' in orphan.stdout
