from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import shutil

import pytest

from negfpy.anharmonic import _mpi_tasks as mpi_tasks
from negfpy.anharmonic._mpi_tasks import (
    DuplicateTaskCheckpointError,
    MPIDependencyError,
    MPIExecutionReport,
    MissingTaskCheckpointError,
    TaskCheckpointCorruptionError,
    TaskCheckpointMismatchError,
    TaskAssignment,
    TaskExecutionError,
    TaskManifest,
    TaskManifestError,
    TaskResult,
    TaskSpec,
    collect_task_checkpoints,
    execute_task_manifest,
    load_task_checkpoint,
    load_task_manifest,
    serial_task_communicator,
    task_checkpoint_path,
    task_assignment_physical_core_evidence,
    task_execution_mpi_origin_lineage_evidence,
    write_task_manifest,
)


def _sha(character: str) -> str:
    assert len(character) == 1
    return character * 64


def _manifest(
    task_ids: tuple[str, ...] = ("case-z", "case-a", "case-m"),
    *,
    environment: str = "e",
) -> TaskManifest:
    tasks = tuple(
        TaskSpec(
            task_id=task_id,
            identity_hashes={
                "problem_sha256": _sha(str((index + 1) % 10)),
                "grid_sha256": _sha(chr(ord("a") + index)),
                "configuration_sha256": _sha(str(index + 6)),
            },
            payload={"ordinal_fixture": index, "labels": [task_id]},
        )
        for index, task_id in enumerate(task_ids)
    )
    return TaskManifest(
        source_revision="0123456789abcdef0123456789abcdef01234567",
        source_tree_sha256=_sha("c"),
        environment_sha256=_sha(environment),
        identity_hashes={"campaign_sha256": _sha("d")},
        metadata={"campaign": "bounded-mpi-test", "certifying": False},
        tasks=tasks,
    )


def _validator(path: Path, task: TaskSpec) -> dict[str, object]:
    payload = path.read_bytes()
    if payload != (task.task_id + "\n").encode("ascii"):
        raise ValueError("artifact payload does not match task")
    return {"validated": True, "byte_count": len(payload)}


def _assignment(
    task_id: str,
    *,
    rank: int,
    logical_cpus: tuple[int, ...],
    physical_core: tuple[int, int],
) -> TaskAssignment:
    return TaskAssignment(
        task_id=task_id,
        status="completed",
        rank=rank,
        hostname="worker",
        elapsed_seconds=1.0,
        resumed=False,
        cpu_affinity=logical_cpus,
        thread_cap_passed=True,
        checkpoint_relative_path=f"{task_id}.task.json",
        checkpoint_file_sha256=_sha("a"),
        artifact_relative_path=f"{task_id}.h5",
        artifact_sha256=_sha("b"),
        cpu_affinity_topology=tuple(
            (logical, physical_core[0], physical_core[1])
            for logical in logical_cpus
        ),
        origin_communicator_size=3,
    )


def _worker(task: TaskSpec, context: mpi_tasks.TaskContext) -> TaskResult:
    artifact = context.write_artifact_bytes(
        (task.task_id + "\n").encode("ascii"), suffix=".txt"
    )
    return TaskResult.completed(
        artifact,
        metadata={
            "observed_rank": context.rank,
            "thread_limit": context.thread_limit,
        },
    )


def test_manifest_round_trip_freezes_inputs_and_preserves_declared_order(
    tmp_path: Path,
) -> None:
    mutable_payload = {"nested": [1, 2]}
    first = TaskSpec(
        "z-last-lexically",
        {"problem_sha256": _sha("1")},
        mutable_payload,
    )
    second = TaskSpec(
        "a-first-lexically",
        {"problem_sha256": _sha("2")},
        {"nested": [3]},
    )
    manifest = TaskManifest(
        source_revision="revision-fixture",
        source_tree_sha256=_sha("a"),
        environment_sha256=_sha("b"),
        tasks=(first, second),
    )
    mutable_payload["nested"].append(999)

    path = write_task_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_task_manifest(path)

    assert [task.task_id for task in loaded.tasks] == [
        "z-last-lexically",
        "a-first-lexically",
    ]
    assert loaded.tasks[0].payload["nested"] == (1, 2)
    assert loaded.manifest_sha256 == manifest.manifest_sha256
    assert loaded.run_identity_sha256 == manifest.run_identity_sha256
    assert loaded.task_identity_sha256(loaded.tasks[0]) != (
        loaded.task_identity_sha256(loaded.tasks[1])
    )
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert json.loads(raw)["payload_sha256"] == manifest.manifest_sha256


def test_manifest_is_immutable_and_rejects_identity_or_content_change(
    tmp_path: Path,
) -> None:
    path = tmp_path / "task-manifest.json"
    original = _manifest(("one",))
    write_task_manifest(path, original)
    assert write_task_manifest(path, original) == path

    changed = _manifest(("one",), environment="f")
    with pytest.raises(TaskManifestError, match="overwrite"):
        write_task_manifest(path, changed)

    payload = path.read_bytes()
    path.write_bytes(payload[:-2] + b"x\n")
    with pytest.raises(TaskManifestError, match="canonical|payload|invalid"):
        load_task_manifest(path)


def test_serial_executor_writes_validated_outcomes_and_canonical_merge(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    report = execute_task_manifest(
        manifest,
        _worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="serial-fixture",
    )

    assert report is not None
    assert report.complete
    assert report.mode == "serial"
    assert report.communicator_size == 1
    assert [item.task_id for item in report.assignments] == [
        task.task_id for task in manifest.tasks
    ]
    assert all(item.status == "completed" for item in report.assignments)
    assert all(item.rank == 0 for item in report.assignments)
    assert all(not item.resumed for item in report.assignments)
    assert (tmp_path / "task-manifest.json").is_file()
    assert (tmp_path / "executions" / "serial-fixture.json").is_file()

    checkpoints = collect_task_checkpoints(
        manifest,
        tmp_path / "task-outcomes",
        tmp_path / "task-artifacts",
        artifact_validator=_validator,
    )
    assert [item.task_id for item in checkpoints] == [
        "case-z",
        "case-a",
        "case-m",
    ]
    assert all(item.artifact_validation["validated"] for item in checkpoints)
    assert all(
        item.execution["thread_cap"]["all_loaded_libraries_capped"]
        for item in checkpoints
    )


def test_restart_skips_terminal_tasks_and_marks_assignments_resumed(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("one", "two"))
    first_calls: list[str] = []

    def first_worker(
        task: TaskSpec, context: mpi_tasks.TaskContext
    ) -> TaskResult:
        first_calls.append(task.task_id)
        return _worker(task, context)

    first = execute_task_manifest(
        manifest,
        first_worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="first-attempt",
    )
    assert first is not None and first.complete
    assert first_calls == ["one", "two"]

    def forbidden_worker(
        task: TaskSpec, context: mpi_tasks.TaskContext
    ) -> TaskResult:
        raise AssertionError(f"resumed task was reiterated: {task.task_id}")

    resumed = execute_task_manifest(
        manifest,
        forbidden_worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="resumed-attempt",
    )
    assert resumed is not None and resumed.complete
    assert [item.resumed for item in resumed.assignments] == [True, True]
    assert len(tuple((tmp_path / "executions").glob("*.json"))) == 2


def test_scientific_failure_is_a_terminal_archived_outcome(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("physical-failure",))

    def worker(
        task: TaskSpec, context: mpi_tasks.TaskContext
    ) -> TaskResult:
        return TaskResult(
            status="scientific_failure",
            metadata={"reason": "fixed controls did not converge"},
        )

    report = execute_task_manifest(
        manifest,
        worker,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="scientific-failure",
    )
    assert report is not None and report.complete
    assert report.assignments[0].status == "scientific_failure"
    checkpoint = collect_task_checkpoints(
        manifest,
        tmp_path / "task-outcomes",
        tmp_path / "task-artifacts",
    )[0]
    assert checkpoint.artifact_relative_path is None
    assert checkpoint.metadata["reason"] == "fixed controls did not converge"


def test_infrastructure_failure_stops_and_exact_restart_resumes_partial_work(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("good", "broken", "not-started"))

    def interrupted(
        task: TaskSpec, context: mpi_tasks.TaskContext
    ) -> TaskResult:
        if task.task_id == "broken":
            raise RuntimeError("simulated infrastructure interruption")
        return _worker(task, context)

    with pytest.raises(TaskExecutionError) as captured:
        execute_task_manifest(
            manifest,
            interrupted,
            tmp_path,
            comm=serial_task_communicator(),
            artifact_validator=_validator,
            execution_id="interrupted",
        )
    partial = captured.value.report
    assert partial is not None and not partial.complete
    assert [item.task_id for item in partial.assignments] == ["good"]
    assert partial.missing_task_ids == ("broken", "not-started")
    assert partial.infrastructure_errors[0]["error_type"] == "RuntimeError"

    resumed = execute_task_manifest(
        manifest,
        _worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="after-interruption",
    )
    assert resumed is not None and resumed.complete
    assert [item.resumed for item in resumed.assignments] == [True, False, False]


def test_missing_duplicate_and_shared_artifact_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("one", "two"))
    execute_task_manifest(
        manifest,
        _worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="complete",
    )
    outcome_directory = tmp_path / "task-outcomes"
    artifact_root = tmp_path / "task-artifacts"
    first = task_checkpoint_path(manifest, "one", outcome_directory)
    second = task_checkpoint_path(manifest, "two", outcome_directory)

    duplicate = outcome_directory / "zz-duplicate.task.json"
    shutil.copyfile(first, duplicate)
    with pytest.raises(DuplicateTaskCheckpointError, match="duplicate"):
        collect_task_checkpoints(
            manifest,
            outcome_directory,
            artifact_root,
            artifact_validator=_validator,
        )
    duplicate.unlink()

    second.unlink()
    with pytest.raises(MissingTaskCheckpointError) as captured:
        collect_task_checkpoints(
            manifest,
            outcome_directory,
            artifact_root,
            artifact_validator=_validator,
        )
    assert captured.value.missing_task_ids == ("two",)


def test_corrupt_sidecar_and_changed_artifact_hash_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("one",))
    execute_task_manifest(
        manifest,
        _worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="complete",
    )
    outcome = task_checkpoint_path(
        manifest, "one", tmp_path / "task-outcomes"
    )
    valid_sidecar = outcome.read_bytes()
    outcome.write_bytes(b"{truncated")
    with pytest.raises(TaskCheckpointCorruptionError):
        load_task_checkpoint(
            outcome,
            manifest=manifest,
            artifact_root=tmp_path / "task-artifacts",
            artifact_validator=_validator,
        )

    outcome.write_bytes(valid_sidecar)
    checkpoint = load_task_checkpoint(
        outcome,
        manifest=manifest,
        artifact_root=tmp_path / "task-artifacts",
        artifact_validator=_validator,
    )
    assert checkpoint.artifact_relative_path is not None
    artifact = tmp_path / "task-artifacts" / checkpoint.artifact_relative_path
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(TaskCheckpointCorruptionError, match="SHA256"):
        load_task_checkpoint(
            outcome,
            manifest=manifest,
            artifact_root=tmp_path / "task-artifacts",
            artifact_validator=None,
        )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("rank", "0"),
        ("communicator_size", True),
        ("hostname", 7),
        ("elapsed_seconds", "1.0"),
        ("thread_limit", 1.0),
    ],
)
def test_checkpoint_execution_rejects_type_coercion(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    manifest = _manifest(("one",))
    execute_task_manifest(
        manifest,
        _worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="complete",
    )
    outcome = task_checkpoint_path(
        manifest, "one", tmp_path / "task-outcomes"
    )
    document = json.loads(outcome.read_text(encoding="ascii"))
    payload = document["payload"]
    payload["execution"][field] = invalid
    document["payload_sha256"] = mpi_tasks._sha256_bytes(  # noqa: SLF001
        mpi_tasks._canonical_json(payload).encode("ascii")  # noqa: SLF001
    )
    outcome.write_bytes(mpi_tasks._canonical_bytes(document))  # noqa: SLF001

    with pytest.raises(
        TaskCheckpointCorruptionError,
        match="invalid task checkpoint payload",
    ):
        load_task_checkpoint(
            outcome,
            manifest=manifest,
            artifact_root=tmp_path / "task-artifacts",
            artifact_validator=_validator,
        )


def test_checkpoint_rejects_contradictory_thread_cap_decision(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("one",))
    execute_task_manifest(
        manifest,
        _worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="complete",
    )
    outcome = task_checkpoint_path(
        manifest, "one", tmp_path / "task-outcomes"
    )
    document = json.loads(outcome.read_text(encoding="ascii"))
    payload = document["payload"]
    thread_cap = payload["execution"]["thread_cap"]
    thread_cap["libraries"] = [
        {
            "user_api": "blas",
            "internal_api": "openblas",
            "prefix": "libopenblas",
            "filepath": "/fixture/libopenblas.so",
            "version": "fixture",
            "architecture": "fixture",
            "num_threads": 64,
        }
    ]
    thread_cap["all_loaded_libraries_capped"] = True
    document["payload_sha256"] = mpi_tasks._sha256_bytes(  # noqa: SLF001
        mpi_tasks._canonical_json(payload).encode("ascii")  # noqa: SLF001
    )
    outcome.write_bytes(mpi_tasks._canonical_bytes(document))  # noqa: SLF001

    with pytest.raises(
        TaskCheckpointCorruptionError,
        match="invalid task checkpoint payload",
    ):
        load_task_checkpoint(
            outcome,
            manifest=manifest,
            artifact_root=tmp_path / "task-artifacts",
            artifact_validator=_validator,
        )


def test_manifest_and_semantic_validation_mismatches_are_rejected(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("one",))
    execute_task_manifest(
        manifest,
        _worker,
        tmp_path,
        comm=serial_task_communicator(),
        artifact_validator=_validator,
        execution_id="complete",
    )
    outcome = task_checkpoint_path(
        manifest, "one", tmp_path / "task-outcomes"
    )

    other = _manifest(("one",), environment="f")
    with pytest.raises(TaskCheckpointMismatchError, match="manifest"):
        load_task_checkpoint(
            outcome,
            manifest=other,
            artifact_root=tmp_path / "task-artifacts",
        )

    def changed_validator(path: Path, task: TaskSpec) -> dict[str, object]:
        _validator(path, task)
        return {"validated": True, "byte_count": 999}

    with pytest.raises(TaskCheckpointCorruptionError, match="evidence"):
        load_task_checkpoint(
            outcome,
            manifest=manifest,
            artifact_root=tmp_path / "task-artifacts",
            artifact_validator=changed_validator,
        )


def test_import_and_serial_fallback_do_not_require_mpi4py(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def missing_mpi(name: str, package: str | None = None):
        if name == "mpi4py.MPI":
            raise ImportError("mpi4py intentionally hidden")
        return original_import(name, package)

    monkeypatch.setattr(mpi_tasks.importlib, "import_module", missing_mpi)
    assert not mpi_tasks.mpi_available()

    report = execute_task_manifest(
        _manifest(("serial",)),
        _worker,
        tmp_path / "fallback",
        artifact_validator=_validator,
        execution_id="fallback",
    )
    assert report is not None and report.mode == "serial"

    with pytest.raises(MPIDependencyError, match=r"negfpy\[mpi\]"):
        execute_task_manifest(
            _manifest(("required",)),
            _worker,
            tmp_path / "required",
            require_mpi=True,
            execution_id="required",
        )


def test_invalid_manifest_inputs_and_worker_artifact_escape_are_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unique"):
        TaskManifest(
            source_revision="revision",
            source_tree_sha256=_sha("a"),
            environment_sha256=_sha("b"),
            tasks=(
                TaskSpec("same", {"problem": _sha("1")}),
                TaskSpec("same", {"problem": _sha("2")}),
            ),
        )
    with pytest.raises(ValueError, match="SHA256"):
        TaskSpec("bad", {"problem": "not-a-hash"})

    outside = tmp_path / "outside.h5"
    outside.write_bytes(b"not within artifact root")

    def escaping_worker(
        task: TaskSpec, context: mpi_tasks.TaskContext
    ) -> TaskResult:
        return TaskResult.completed(outside)

    with pytest.raises(TaskExecutionError) as captured:
        execute_task_manifest(
            _manifest(("escape",)),
            escaping_worker,
            tmp_path / "run",
            comm=serial_task_communicator(),
            execution_id="escape",
        )
    assert captured.value.report is not None
    assert captured.value.report.infrastructure_errors[0]["error_type"] == (
        "TaskCheckpointCorruptionError"
    )


def test_invalid_orphan_artifact_is_rejected_without_overwrite(
    tmp_path: Path,
) -> None:
    manifest = _manifest(("orphan",))

    def crash_after_publish(
        task: TaskSpec, context: mpi_tasks.TaskContext
    ) -> TaskResult:
        context.write_artifact_bytes(
            (task.task_id + "\n").encode("ascii"),
            suffix=".txt",
        )
        raise RuntimeError("crash before task checkpoint publication")

    with pytest.raises(TaskExecutionError):
        execute_task_manifest(
            manifest,
            crash_after_publish,
            tmp_path,
            comm=serial_task_communicator(),
            artifact_validator=_validator,
            execution_id="publish-then-crash",
        )
    artifact = next((tmp_path / "task-artifacts").glob("*.txt"))
    artifact.write_bytes(b"identity-mismatch\n")
    damaged_bytes = artifact.read_bytes()
    worker_called = False

    def forbidden_worker(
        task: TaskSpec, context: mpi_tasks.TaskContext
    ) -> TaskResult:
        nonlocal worker_called
        worker_called = True
        return _worker(task, context)

    def adopter(path, task, evidence):
        return TaskResult.completed(path)

    with pytest.raises(TaskExecutionError) as captured:
        execute_task_manifest(
            manifest,
            forbidden_worker,
            tmp_path,
            comm=serial_task_communicator(),
            artifact_validator=_validator,
            execution_id="reject-invalid-orphan",
            orphan_artifact_suffix=".txt",
            orphan_artifact_adopter=adopter,
        )
    assert not worker_called
    assert artifact.read_bytes() == damaged_bytes
    assert captured.value.report is not None


def test_physical_core_evidence_accepts_smt_siblings_on_one_core() -> None:
    assignments = (
        _assignment(
            "q0",
            rank=1,
            logical_cpus=(1, 45),
            physical_core=(0, 1),
        ),
        _assignment(
            "q1",
            rank=1,
            logical_cpus=(1, 45),
            physical_core=(0, 1),
        ),
        _assignment(
            "q2",
            rank=2,
            logical_cpus=(2, 46),
            physical_core=(0, 2),
        ),
    )
    evidence = task_assignment_physical_core_evidence(
        assignments,
        require_available=True,
    )
    assert evidence["available"] is True
    assert len(evidence["rank_bindings"]) == 2
    assert evidence["rank_bindings"][0]["physical_core"] == {
        "package_id": 0,
        "core_id": 1,
    }


@pytest.mark.parametrize(
    "assignments",
    [
        (
            _assignment(
                "q0",
                rank=1,
                logical_cpus=(1, 45),
                physical_core=(0, 1),
            ),
            _assignment(
                "q1",
                rank=2,
                logical_cpus=(1, 45),
                physical_core=(0, 1),
            ),
        ),
        (
            _assignment(
                "q0",
                rank=1,
                logical_cpus=(1, 45),
                physical_core=(0, 1),
            ),
            _assignment(
                "q1",
                rank=1,
                logical_cpus=(2, 46),
                physical_core=(0, 2),
            ),
        ),
    ],
)
def test_physical_core_evidence_rejects_overlap_or_inconsistency(
    assignments: tuple[TaskAssignment, ...],
) -> None:
    with pytest.raises(
        TaskManifestError,
        match="physical-core binding evidence",
    ):
        task_assignment_physical_core_evidence(
            assignments,
            require_available=True,
        )


def test_mpi_origin_lineage_accepts_exact_resume_and_rejects_orphan() -> None:
    fresh_assignment = _assignment(
        "q0",
        rank=1,
        logical_cpus=(1, 45),
        physical_core=(0, 1),
    )

    def report(
        execution_id: str,
        assignment: TaskAssignment,
    ) -> MPIExecutionReport:
        return MPIExecutionReport(
            execution_id=execution_id,
            manifest_sha256=_sha("c"),
            mode="mpi",
            communicator_size=3,
            coordinator_hostname="worker",
            started_utc="2026-01-01T00:00:00Z",
            finished_utc="2026-01-01T00:00:01Z",
            elapsed_seconds=1.0,
            thread_limit=1,
            assignments=(assignment,),
            missing_task_ids=(),
            infrastructure_errors=(),
        )

    initial = report("initial", fresh_assignment)
    resumed = report(
        "resumed",
        replace(fresh_assignment, resumed=True),
    )
    evidence = task_execution_mpi_origin_lineage_evidence(
        resumed,
        (initial, resumed),
        require_mpi_origin=True,
    )
    assert evidence["passed"] is True
    assert evidence["lineage"][0]["origin_execution_id"] == "initial"

    adopted = report(
        "adopted",
        replace(
            fresh_assignment,
            resumed=True,
            orphan_artifact_adopted=True,
        ),
    )
    with pytest.raises(TaskManifestError, match="origin lineage"):
        task_execution_mpi_origin_lineage_evidence(
            adopted,
            (initial, adopted),
            require_mpi_origin=True,
        )

    serial_origin = report(
        "serial-origin-resume",
        replace(
            fresh_assignment,
            rank=0,
            resumed=True,
            origin_communicator_size=1,
        ),
    )
    with pytest.raises(TaskManifestError, match="origin lineage"):
        task_execution_mpi_origin_lineage_evidence(
            serial_origin,
            (serial_origin,),
            require_mpi_origin=True,
        )


def test_mpi_origin_lineage_accepts_completed_task_from_partial_attempt() -> None:
    first = _assignment(
        "q0",
        rank=1,
        logical_cpus=(1, 45),
        physical_core=(0, 1),
    )
    initial = MPIExecutionReport(
        execution_id="partial-two-rank",
        manifest_sha256=_sha("c"),
        mode="mpi",
        communicator_size=3,
        coordinator_hostname="worker",
        started_utc="2026-01-01T00:00:00Z",
        finished_utc="2026-01-01T00:00:01Z",
        elapsed_seconds=1.0,
        thread_limit=1,
        assignments=(first,),
        missing_task_ids=("q1",),
        infrastructure_errors=(
            {"error_type": "RuntimeError", "message": "interrupted"},
        ),
    )
    second = replace(
        _assignment(
            "q1",
            rank=2,
            logical_cpus=(2, 46),
            physical_core=(0, 2),
        ),
        origin_communicator_size=4,
    )
    selected = MPIExecutionReport(
        execution_id="resume-four-rank",
        manifest_sha256=_sha("c"),
        mode="mpi",
        communicator_size=4,
        coordinator_hostname="worker",
        started_utc="2026-01-01T00:00:02Z",
        finished_utc="2026-01-01T00:00:03Z",
        elapsed_seconds=1.0,
        thread_limit=1,
        assignments=(replace(first, resumed=True), second),
        missing_task_ids=(),
        infrastructure_errors=(),
    )

    evidence = task_execution_mpi_origin_lineage_evidence(
        selected,
        (initial, selected),
        require_mpi_origin=True,
    )

    assert evidence["passed"] is True
    assert {
        item["task_id"]: item["origin_execution_id"]
        for item in evidence["lineage"]
    } == {
        "q0": "partial-two-rank",
        "q1": "resume-four-rank",
    }


def test_physical_core_evidence_scopes_reused_rank_by_origin_execution() -> None:
    from_partial = _assignment(
        "q0",
        rank=1,
        logical_cpus=(1, 45),
        physical_core=(0, 1),
    )
    from_restart = replace(
        _assignment(
            "q1",
            rank=1,
            logical_cpus=(7, 51),
            physical_core=(0, 7),
        ),
        origin_communicator_size=4,
    )

    with pytest.raises(TaskManifestError, match="physical-core binding"):
        task_assignment_physical_core_evidence(
            (from_partial, from_restart),
            require_available=True,
        )

    evidence = task_assignment_physical_core_evidence(
        (from_partial, from_restart),
        require_available=True,
        origin_execution_ids={
            "q0": "partial-two-rank",
            "q1": "resume-four-rank",
        },
    )

    assert evidence["available"] is True
    assert evidence["origin_execution_scoped"] is True
    assert {
        item["origin_execution_id"]
        for item in evidence["rank_bindings"]
    } == {"partial-two-rank", "resume-four-rank"}
