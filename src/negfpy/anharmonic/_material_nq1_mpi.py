"""Task-level MPI DAG for complete silicon ``N_q=1`` SCBA cases."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from . import _mpi_tasks as mpi_tasks
from ._material_nq1_campaign import (
    MaterialSCBATask,
    silicon_nq1_release_manifest,
    silicon_nq1_release_tasks,
)
from ._mpi_tasks import (
    ArtifactValidator,
    MPIExecutionReport,
    TaskCheckpointMismatchError,
    TaskContext,
    TaskManifest,
    TaskResult,
    TaskSpec,
    collect_task_checkpoints,
    execute_task_manifest,
    load_task_checkpoint,
    task_checkpoint_path,
    write_task_manifest,
)


MaterialTaskExecutor = Callable[
    [MaterialSCBATask, TaskSpec, TaskContext, Mapping[str, Path]], TaskResult
]


@dataclass(frozen=True, slots=True)
class MaterialNQ1MPICampaign:
    manifest: TaskManifest
    scientific_tasks: tuple[MaterialSCBATask, ...]
    waves: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TaskManifest):
            raise TypeError("manifest must be a TaskManifest")
        tasks = tuple(self.scientific_tasks)
        waves = tuple(tuple(wave) for wave in self.waves)
        identifiers = tuple(task.task_id for task in tasks)
        if tuple(task.task_id for task in self.manifest.tasks) != identifiers:
            raise ValueError("MPI and scientific manifest task orders differ")
        flattened = tuple(task_id for wave in waves for task_id in wave)
        if set(flattened) != set(identifiers) or len(flattened) != len(identifiers):
            raise ValueError("waves must contain every task exactly once")
        position = {task_id: index for index, wave in enumerate(waves) for task_id in wave}
        for task in tasks:
            for dependency in task_dependencies(task):
                if position[dependency] >= position[task.task_id]:
                    raise ValueError("task dependency is not in an earlier wave")
        object.__setattr__(self, "scientific_tasks", tasks)
        object.__setattr__(self, "waves", waves)

    def task(self, task_id: str) -> MaterialSCBATask:
        for task in self.scientific_tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def wave_index(self, task_id: str) -> int:
        for index, wave in enumerate(self.waves):
            if task_id in wave:
                return index
        raise KeyError(task_id)


def task_dependencies(task: MaterialSCBATask) -> tuple[str, ...]:
    dependencies: list[str] = []
    if task.continuation_parent is not None:
        dependencies.append(task.continuation_parent)
    # Independent endpoint/refinement calculations are delayed until the
    # physical coupling path has succeeded, avoiding expensive fanout after a
    # failed physical endpoint without making them numerical continuations.
    if task.role in {"initialization-equality", "numerical-refinement"}:
        dependencies.append("coupling-g100")
    return tuple(dict.fromkeys(dependencies))


def _waves(tasks: Sequence[MaterialSCBATask]) -> tuple[tuple[str, ...], ...]:
    coupling = tuple(task.task_id for task in tasks if task.role == "coupling-path")
    fanout = tuple(
        task.task_id
        for task in tasks
        if task.task_id in {"bias-positive-half", "bias-negative-half"}
        or task.role in {"initialization-equality", "numerical-refinement"}
    )
    endpoints = ("bias-positive-full", "bias-negative-full")
    return tuple((task_id,) for task_id in coupling) + (fanout, endpoints)


def build_material_nq1_mpi_campaign(
    *,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    input_hashes: Mapping[str, str],
) -> MaterialNQ1MPICampaign:
    tasks = silicon_nq1_release_tasks()
    release = silicon_nq1_release_manifest(
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        input_hashes=input_hashes,
    )
    specs = tuple(
        TaskSpec(
            task_id=task.task_id,
            identity_hashes={
                "release_campaign": release["campaign_sha256"],
                "scientific_task": task.task_sha256,
                **dict(input_hashes),
            },
            payload={
                "schema": "negfpy.material-nq1-mpi-task.v2",
                "dependencies": list(task_dependencies(task)),
                "scientific_task": task.as_dict()
                | {"task_sha256": task.task_sha256},
            },
        )
        for task in tasks
    )
    manifest = TaskManifest(
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        tasks=specs,
        identity_hashes={
            "release_campaign": release["campaign_sha256"],
            **dict(input_hashes),
        },
        metadata={
            "schema": "negfpy.material-nq1-mpi-campaign.v2",
            "scientific_scope": release["scientific_scope"],
            "release_campaign_sha256": release["campaign_sha256"],
            "release_certified": False,
        },
    )
    return MaterialNQ1MPICampaign(manifest, tasks, _waves(tasks))


def _phase_manifest(
    campaign: MaterialNQ1MPICampaign,
    wave_index: int,
) -> TaskManifest:
    task_ids = campaign.waves[wave_index]
    return TaskManifest(
        source_revision=campaign.manifest.source_revision,
        source_tree_sha256=campaign.manifest.source_tree_sha256,
        environment_sha256=campaign.manifest.environment_sha256,
        tasks=tuple(campaign.manifest.task(task_id) for task_id in task_ids),
        identity_hashes={"campaign": campaign.manifest.manifest_sha256},
        metadata={
            "schema": "negfpy.material-nq1-mpi-wave.v1",
            "campaign_manifest_sha256": campaign.manifest.manifest_sha256,
            "wave_index": wave_index,
            "task_ids": list(task_ids),
        },
    )


def _phase_root(root: Path, wave_index: int) -> Path:
    return root / "waves" / f"{wave_index:03d}"


def _dependency_artifacts(
    root: Path,
    campaign: MaterialNQ1MPICampaign,
    task: MaterialSCBATask,
    validator: ArtifactValidator,
) -> Mapping[str, Path]:
    result: dict[str, Path] = {}
    for dependency in task_dependencies(task):
        index = campaign.wave_index(dependency)
        phase = _phase_manifest(campaign, index)
        phase_root = _phase_root(root, index)
        checkpoint = load_task_checkpoint(
            task_checkpoint_path(
                phase, dependency, phase_root / "task-outcomes"
            ),
            manifest=phase,
            artifact_root=phase_root / "task-artifacts",
            artifact_validator=validator,
        )
        if checkpoint.status != "completed" or checkpoint.artifact_relative_path is None:
            raise TaskCheckpointMismatchError(
                f"dependency {dependency!r} is not a completed outcome"
            )
        result[dependency] = (
            phase_root / "task-artifacts" / checkpoint.artifact_relative_path
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class MaterialNQ1MPIReport:
    campaign_manifest_sha256: str
    execution_id: str
    statuses: Mapping[str, str]
    phase_reports: tuple[MPIExecutionReport, ...]

    @property
    def complete(self) -> bool:
        return all(status == "completed" for status in self.statuses.values())


def execute_material_nq1_mpi_campaign(
    campaign: MaterialNQ1MPICampaign,
    executor: MaterialTaskExecutor,
    run_directory: str | Path,
    *,
    artifact_validator: ArtifactValidator,
    comm: Any | None = None,
    require_mpi: bool = False,
    thread_limit: int = 1,
    execution_id: str | None = None,
) -> MaterialNQ1MPIReport | None:
    communicator = mpi_tasks._resolve_communicator(comm, require_mpi=require_mpi)
    rank = int(communicator.Get_rank())
    root = Path(run_directory).expanduser().resolve()
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        write_task_manifest(root / "campaign-manifest.json", campaign.manifest)
        selected_id = execution_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            + "-" + uuid4().hex[:12]
        )
    else:
        selected_id = None
    if int(communicator.Get_size()) > 1:
        selected_id = communicator.bcast(selected_id, root=0)
    assert isinstance(selected_id, str)

    statuses: dict[str, str] = {}
    reports: list[MPIExecutionReport] = []
    for wave_index, task_ids in enumerate(campaign.waves):
        phase = _phase_manifest(campaign, wave_index)
        phase_root = _phase_root(root, wave_index)
        if int(communicator.Get_size()) > 1:
            statuses = communicator.bcast(statuses if rank == 0 else None, root=0)

        def worker(spec: TaskSpec, context: TaskContext) -> TaskResult:
            scientific = campaign.task(spec.task_id)
            dependencies = task_dependencies(scientific)
            failed = [item for item in dependencies if statuses.get(item) != "completed"]
            if failed:
                return TaskResult(
                    status="blocked",
                    metadata={
                        "reason": "immutable prerequisite did not complete",
                        "failed_dependencies": failed,
                    },
                )
            dependency_paths = _dependency_artifacts(
                root, campaign, scientific, artifact_validator
            )
            return executor(scientific, spec, context, dependency_paths)

        report = execute_task_manifest(
            phase,
            worker,
            phase_root,
            comm=communicator,
            require_mpi=require_mpi,
            thread_limit=thread_limit,
            artifact_validator=artifact_validator,
            execution_id=f"{selected_id}.wave-{wave_index:03d}",
        )
        if rank == 0:
            assert report is not None
            checkpoints = collect_task_checkpoints(
                phase,
                phase_root / "task-outcomes",
                phase_root / "task-artifacts",
                artifact_validator=artifact_validator,
                require_complete=True,
            )
            statuses.update({item.task_id: item.status for item in checkpoints})
            reports.append(report)
        if int(communicator.Get_size()) > 1:
            statuses = communicator.bcast(statuses if rank == 0 else None, root=0)
    if rank != 0:
        return None
    ordered = MappingProxyType({
        task.task_id: statuses[task.task_id]
        for task in campaign.scientific_tasks
    })
    return MaterialNQ1MPIReport(
        campaign_manifest_sha256=campaign.manifest.manifest_sha256,
        execution_id=selected_id,
        statuses=ordered,
        phase_reports=tuple(reports),
    )


__all__ = [
    "MaterialNQ1MPICampaign",
    "MaterialNQ1MPIReport",
    "build_material_nq1_mpi_campaign",
    "execute_material_nq1_mpi_campaign",
    "task_dependencies",
]
