"""Private protocol-v2 task-DAG adapter for task-level MPI execution.

The solvers intentionally know nothing about MPI. This module turns the
locked production toy protocol into a static DAG of whole validation cases
and evaluates immutable frontiers through ``anharmonic._mpi_tasks``. Only
task identifiers and terminal status travel through MPI; scientific arrays
remain in atomic HDF5 case artifacts on the shared filesystem.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import h5py
import numpy as np

from .. import _mpi_tasks as mpi_tasks
from .._mpi_tasks import (
    ArtifactValidator,
    DuplicateTaskCheckpointError,
    MissingTaskCheckpointError,
    MPIExecutionReport,
    TaskCheckpointMismatchError,
    TaskContext,
    TaskManifest,
    TaskManifestError,
    TaskResult,
    TaskSpec,
    collect_task_checkpoints,
    execute_task_manifest,
    load_task_checkpoint,
    task_checkpoint_path,
    write_task_manifest,
)


_CAMPAIGN_SCHEMA = "negfpy_protocol_v2_toy_mpi_campaign"
_CAMPAIGN_SCHEMA_VERSION = 1
_CASE_SCHEMA = "negfpy_protocol_v2_toy_case"
_CASE_SCHEMA_VERSION = 1
_EXECUTION_SCHEMA = "negfpy_protocol_v2_toy_mpi_execution"
_EXECUTION_SCHEMA_VERSION = 1
_SAFE_ARRAY_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_TERMINAL_CASE_STATUSES = frozenset({"completed", "scientific_failure"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

CampaignCaseExecutor = Callable[
    [TaskSpec, "ProductionToyCaseContext"],
    "ProductionToyCaseResult",
]


def _strict_json_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.generic):
        return _strict_json_value(value.item(), path=path)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} contains NaN or infinity.")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings.")
            normalized[key] = _strict_json_value(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _strict_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported value of type {type(value).__name__}."
    )


def _freeze_json(value: Any) -> Any:
    normalized = _strict_json_value(value)
    if isinstance(normalized, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _strict_json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_document(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("ascii"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    shape = tuple(int(value) for value in array.shape)
    digest.update(repr(shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256.")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _phase_slug(index: int, task_ids: Sequence[str]) -> str:
    if index == 0:
        label = "preflight"
    elif index == 1:
        label = "direct-zero"
    elif len(task_ids) == 1 and task_ids[0].startswith("multidimensional.path_"):
        label = "continuation"
    else:
        label = "post-gate-fanout"
    return f"{index:03d}-{label}"


@dataclass(frozen=True, slots=True)
class ProductionToyCaseResult:
    """Scientific result returned by one whole-case executor."""

    status: Literal["completed", "scientific_failure"]
    evidence: Mapping[str, Any] = field(default_factory=dict)
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _TERMINAL_CASE_STATUSES:
            raise ValueError(f"unsupported case status {self.status!r}.")
        if not isinstance(self.evidence, Mapping):
            raise TypeError("evidence must be a mapping.")
        if not isinstance(self.arrays, Mapping):
            raise TypeError("arrays must be a mapping.")
        normalized_arrays: dict[str, np.ndarray] = {}
        for name, values in sorted(self.arrays.items()):
            if not isinstance(name, str) or _SAFE_ARRAY_NAME.fullmatch(name) is None:
                raise ValueError("invalid case-array name.")
            array = np.array(values, order="C", copy=True)
            if array.dtype.hasobject:
                raise TypeError(f"arrays[{name!r}] cannot use object dtype.")
            if np.issubdtype(array.dtype, np.number) and not np.all(
                np.isfinite(array)
            ):
                raise ValueError(f"arrays[{name!r}] contains nonfinite values.")
            array.setflags(write=False)
            normalized_arrays[name] = array
        object.__setattr__(self, "evidence", _freeze_json(self.evidence))
        object.__setattr__(self, "arrays", MappingProxyType(normalized_arrays))

    @classmethod
    def completed(
        cls,
        *,
        evidence: Mapping[str, Any] | None = None,
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> "ProductionToyCaseResult":
        return cls(
            "completed",
            {} if evidence is None else evidence,
            {} if arrays is None else arrays,
        )

    @classmethod
    def scientific_failure(
        cls,
        *,
        evidence: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> "ProductionToyCaseResult":
        return cls(
            "scientific_failure",
            evidence,
            {} if arrays is None else arrays,
        )


@dataclass(frozen=True, slots=True)
class ProductionToyCaseContext:
    """Case context with validated, exact dependency artifacts."""

    task_context: TaskContext
    campaign_manifest_sha256: str
    campaign_task_identity_sha256: str
    dependency_artifacts: Mapping[str, Path]
    continuation_parent_task_id: str | None

    def __post_init__(self) -> None:
        _require_sha256(
            self.campaign_manifest_sha256,
            name="campaign_manifest_sha256",
        )
        _require_sha256(
            self.campaign_task_identity_sha256,
            name="campaign_task_identity_sha256",
        )
        if not isinstance(self.task_context, TaskContext):
            raise TypeError("task_context must be a TaskContext.")
        normalized = {
            str(task_id): Path(path)
            for task_id, path in self.dependency_artifacts.items()
        }
        if self.continuation_parent_task_id is not None and (
            self.continuation_parent_task_id not in normalized
        ):
            raise ValueError(
                "the continuation parent must have a validated artifact."
            )
        object.__setattr__(
            self,
            "dependency_artifacts",
            MappingProxyType(normalized),
        )

    @property
    def continuation_parent_artifact(self) -> Path | None:
        if self.continuation_parent_task_id is None:
            return None
        return self.dependency_artifacts[self.continuation_parent_task_id]

    def read_dependency_array(
        self,
        task_id: str,
        array_name: str,
    ) -> np.ndarray:
        try:
            path = self.dependency_artifacts[task_id]
        except KeyError as exc:
            raise KeyError(f"unknown dependency {task_id!r}.") from exc
        if _SAFE_ARRAY_NAME.fullmatch(array_name) is None:
            raise ValueError("invalid array_name.")
        with h5py.File(path, "r") as handle:
            try:
                values = np.asarray(handle["arrays"][array_name])
            except KeyError as exc:
                raise KeyError(
                    f"dependency {task_id!r} has no array {array_name!r}."
                ) from exc
        values.setflags(write=False)
        return values


@dataclass(frozen=True, slots=True)
class ProductionToyMPICampaign:
    """Immutable full task manifest and static dependency frontiers."""

    manifest: TaskManifest
    preflight_task_ids: tuple[str, str]
    gate_task_id: str
    waves: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, TaskManifest):
            raise TypeError("manifest must be a TaskManifest.")
        preflights = tuple(self.preflight_task_ids)
        waves = tuple(tuple(values) for values in self.waves)
        if len(preflights) != 2 or len(set(preflights)) != 2:
            raise ValueError(
                "exactly two distinct preflight tasks are required."
            )
        if len(waves) < 3 or any(not values for values in waves):
            raise ValueError("waves must contain non-empty static frontiers.")
        if waves[0] != preflights:
            raise ValueError(
                "the first wave must contain the two preflights."
            )
        if waves[1] != (self.gate_task_id,):
            raise ValueError(
                "the second wave must contain only the gate task."
            )
        task_ids = tuple(task.task_id for task in self.manifest.tasks)
        flattened = tuple(task_id for wave in waves for task_id in wave)
        if flattened != task_ids:
            raise ValueError(
                "manifest order must equal canonical wave order."
            )
        if len(set(flattened)) != len(flattened):
            raise ValueError("a campaign task can appear in only one wave.")
        wave_index = {
            task_id: index
            for index, wave in enumerate(waves)
            for task_id in wave
        }
        for task in self.manifest.tasks:
            dependencies = _task_dependencies(task)
            if any(
                dependency not in wave_index for dependency in dependencies
            ):
                raise ValueError(
                    f"{task.task_id!r} names an unknown dependency."
                )
            if any(
                wave_index[dependency] >= wave_index[task.task_id]
                for dependency in dependencies
            ):
                raise ValueError(
                    f"{task.task_id!r} has a non-ancestral dependency."
                )
            parent = _continuation_parent(task)
            if parent is not None and parent not in dependencies:
                raise ValueError(
                    "the continuation parent must be an explicit dependency."
                )
        gate_dependencies = set(
            _task_dependencies(self.manifest.task(self.gate_task_id))
        )
        if gate_dependencies != set(preflights):
            raise ValueError(
                "the direct-zero gate must depend on both preflights."
            )
        for task_id in task_ids[3:]:
            dependencies = _task_dependencies(self.manifest.task(task_id))
            if self.gate_task_id not in dependencies:
                raise ValueError(
                    "every post-gate task must depend on direct-zero."
                )
        object.__setattr__(self, "preflight_task_ids", preflights)
        object.__setattr__(self, "waves", waves)

    @property
    def campaign_manifest_sha256(self) -> str:
        return self.manifest.manifest_sha256

    @property
    def canonical_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.manifest.tasks)

    def wave_index(self, task_id: str) -> int:
        for index, wave in enumerate(self.waves):
            if task_id in wave:
                return index
        raise KeyError(task_id)


def _task_dependencies(task: TaskSpec) -> tuple[str, ...]:
    raw = task.payload.get("dependencies", ())
    if not isinstance(raw, (tuple, list)):
        raise TypeError(
            f"{task.task_id!r} dependencies must be a sequence."
        )
    dependencies = tuple(str(value) for value in raw)
    if any(not value for value in dependencies):
        raise ValueError("dependency identifiers cannot be empty.")
    if len(set(dependencies)) != len(dependencies):
        raise ValueError(f"{task.task_id!r} repeats a dependency.")
    return dependencies


def _continuation_parent(task: TaskSpec) -> str | None:
    raw = task.payload.get("continuation_parent_task_id")
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw:
        raise TypeError(
            "continuation_parent_task_id must be text or None."
        )
    return raw


def _task_spec(
    task_id: str,
    *,
    kind: str,
    dependencies: Sequence[str],
    identity_hashes: Mapping[str, str],
    payload: Mapping[str, Any],
    continuation_parent_task_id: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        identity_hashes=identity_hashes,
        payload={
            "campaign_schema": _CAMPAIGN_SCHEMA,
            "campaign_schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "protocol": "multidimensional_toy_gate_v2",
            "kind": kind,
            "dependencies": list(dependencies),
            "continuation_parent_task_id": continuation_parent_task_id,
            **dict(payload),
        },
    )


def build_production_toy_mpi_campaign(
    *,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    config: Any | None = None,
    open_fixtures: Sequence[tuple[str, Any]] | None = None,
) -> ProductionToyMPICampaign:
    """Freeze the existing protocol-v2 graph as exact task identities.

    The default is the full release configuration and its 13 locked
    open-system cases. Bounded fixtures/configurations may be supplied for
    smoke tests, but a certification-eligible configuration still requires
    the exact 13 multidimensional and 13 open-system cases.
    """

    from .._provenance import configuration_hash, problem_input_hashes
    from .multidimensional_release import (
        MultidimensionalReleaseConfig,
        _expected_outcome_topology,
        _physical_path,
        _solver_config,
        build_multidimensional_problem,
        full_multidimensional_release_config,
    )
    from .open_three_mode import OpenThreeModeFixture
    from .production_gate_runner import open_three_mode_release_fixtures

    selected_config = (
        full_multidimensional_release_config() if config is None else config
    )
    if not isinstance(selected_config, MultidimensionalReleaseConfig):
        raise TypeError("config must be a MultidimensionalReleaseConfig.")
    if selected_config.certification_eligible and open_fixtures is not None:
        raise ValueError(
            "certification-eligible campaigns must use the locked default "
            "open-system fixture matrix."
        )
    fixtures = (
        open_three_mode_release_fixtures()
        if open_fixtures is None
        else tuple(open_fixtures)
    )
    if not fixtures:
        raise ValueError("at least one open-system case is required.")
    if any(
        not isinstance(name, str)
        or not name
        or not isinstance(fixture, OpenThreeModeFixture)
        for name, fixture in fixtures
    ):
        raise TypeError(
            "open_fixtures must contain (name, OpenThreeModeFixture) pairs."
        )
    if len({name for name, _ in fixtures}) != len(fixtures):
        raise ValueError("open-system case names must be unique.")

    _require_sha256(source_tree_sha256, name="source_tree_sha256")
    _require_sha256(environment_sha256, name="environment_sha256")
    config_document = selected_config.as_dict()
    protocol_configuration_sha256 = _hash_document(config_document)

    harmonic_id = "preflight.harmonic-same-grid"
    backend_id = "preflight.backend-equality"
    common_preflight = {
        "protocol_configuration_sha256": protocol_configuration_sha256,
    }
    harmonic = _task_spec(
        harmonic_id,
        kind="harmonic_preflight",
        dependencies=(),
        identity_hashes={
            **common_preflight,
            "preflight_sha256": _hash_document(
                {"kind": "harmonic_same_grid", "config": config_document}
            ),
        },
        payload={"config": config_document},
    )
    backend = _task_spec(
        backend_id,
        kind="backend_preflight",
        dependencies=(),
        identity_hashes={
            **common_preflight,
            "preflight_sha256": _hash_document(
                {"kind": "bounded_backend", "config": config_document}
            ),
        },
        payload={"config": config_document},
    )

    rows = _expected_outcome_topology(selected_config)
    path_count = len(_physical_path(selected_config))
    gate_row = rows[0]
    path_rows = rows[1 : 1 + path_count]
    independent_rows = rows[1 + path_count :]
    run_to_task_id = {
        str(row["run_id"]): f"multidimensional.{row['run_id']}"
        for row in rows
    }
    gate_task_id = run_to_task_id[str(gate_row["run_id"])]

    def multidimensional_task(row: Mapping[str, Any]) -> TaskSpec:
        point = row["point"]
        problem = build_multidimensional_problem(
            selected_config,
            g=point.g,
            bias=point.bias,
            numerical=point.numerical,
        )
        solver_config = _solver_config(
            selected_config,
            point.numerical,
            backend=row["backend"],
        )
        input_hashes = problem_input_hashes(problem)
        input_hashes["configuration"] = configuration_hash(solver_config)
        parent_run_id = row["continuation_parent_id"]
        parent_task_id = (
            None
            if parent_run_id is None
            else run_to_task_id[str(parent_run_id)]
        )
        task_id = run_to_task_id[str(row["run_id"])]
        dependencies: list[str] = []
        if task_id == gate_task_id:
            dependencies.extend((harmonic_id, backend_id))
        else:
            dependencies.append(gate_task_id)
            if parent_task_id is not None:
                dependencies.append(parent_task_id)
        return _task_spec(
            task_id,
            kind=(
                "direct_zero_endpoint"
                if task_id == gate_task_id
                else "multidimensional_case"
            ),
            dependencies=dependencies,
            identity_hashes=input_hashes,
            continuation_parent_task_id=parent_task_id,
            payload={
                "run_id": row["run_id"],
                "point": point.as_dict(
                    selected_config.spec.mean_temperature
                ),
                "initialization": row["initialization"],
                "backend": row["backend"],
                "continuation_parent_run_id": parent_run_id,
                "config_sha256": protocol_configuration_sha256,
            },
        )

    gate_task = multidimensional_task(gate_row)
    path_tasks = tuple(multidimensional_task(row) for row in path_rows)
    independent_multidimensional = tuple(
        multidimensional_task(row) for row in independent_rows
    )

    open_tasks: list[TaskSpec] = []
    for name, fixture in fixtures:
        input_hashes = problem_input_hashes(fixture.problem)
        input_hashes["configuration"] = configuration_hash(fixture.config)
        open_tasks.append(
            _task_spec(
                f"open_three_mode.{name}",
                kind="open_three_mode_case",
                dependencies=(gate_task_id,),
                identity_hashes=input_hashes,
                payload={
                    "name": name,
                    "controls": {
                        "grid_step": fixture.grid.step,
                        "grid_n_positive": fixture.grid.n_positive,
                        "grid_cutoff": fixture.grid.cutoff,
                        "temperature": fixture.temperature,
                        "g": fixture.g,
                        "eta_lead": fixture.eta_lead,
                        "mode_frequencies": list(
                            fixture.mode_frequencies
                        ),
                        "lead_coupling": fixture.lead_coupling,
                        "backend": fixture.config.backend,
                        "hilbert_backend": fixture.config.hilbert_backend,
                        "surface_gf_method": (
                            fixture.config.surface_gf_method
                        ),
                    },
                    "config_sha256": protocol_configuration_sha256,
                },
            )
        )

    fanout = independent_multidimensional + tuple(open_tasks)
    tasks = (harmonic, backend, gate_task, *path_tasks, *fanout)
    waves = (
        ((harmonic.task_id, backend.task_id),)
        + ((gate_task.task_id,),)
        + tuple((task.task_id,) for task in path_tasks)
        + (tuple(task.task_id for task in fanout),)
    )
    manifest = TaskManifest(
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        tasks=tuple(tasks),
        identity_hashes={
            "protocol_configuration_sha256": (
                protocol_configuration_sha256
            ),
        },
        metadata={
            "schema": _CAMPAIGN_SCHEMA,
            "schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "protocol": "multidimensional_toy_gate_v2",
            "certification_eligible": selected_config.certification_eligible,
            "multidimensional_outcome_count": len(rows),
            "open_system_case_count": len(fixtures),
            "continuation_path_count": len(path_rows),
        },
    )
    if selected_config.certification_eligible and (
        len(rows) != 13 or len(fixtures) != 13 or len(path_rows) != 5
    ):
        raise ValueError(
            "the certifying protocol requires 13 multidimensional outcomes, "
            "13 open-system cases, and five continuation points."
        )
    return ProductionToyMPICampaign(
        manifest=manifest,
        preflight_task_ids=(harmonic_id, backend_id),
        gate_task_id=gate_task_id,
        waves=waves,
    )


def _phase_manifest(
    campaign: ProductionToyMPICampaign,
    wave_index: int,
) -> TaskManifest:
    task_ids = campaign.waves[wave_index]
    return TaskManifest(
        source_revision=campaign.manifest.source_revision,
        source_tree_sha256=campaign.manifest.source_tree_sha256,
        environment_sha256=campaign.manifest.environment_sha256,
        tasks=tuple(campaign.manifest.task(task_id) for task_id in task_ids),
        identity_hashes={
            "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        },
        metadata={
            "schema": _CAMPAIGN_SCHEMA,
            "schema_version": _CAMPAIGN_SCHEMA_VERSION,
            "campaign_manifest_sha256": (
                campaign.campaign_manifest_sha256
            ),
            "wave_index": wave_index,
            "canonical_task_ids": list(task_ids),
        },
    )


def _case_artifact_evidence(
    path: Path,
    task: TaskSpec,
    campaign: ProductionToyMPICampaign,
) -> dict[str, Any]:
    expected_identity = campaign.manifest.task_identity_sha256(task)
    try:
        with h5py.File(path, "r") as handle:
            if handle.attrs.get("schema_name") != _CASE_SCHEMA:
                raise ValueError("wrong case artifact schema")
            if int(handle.attrs.get("schema_version", -1)) != (
                _CASE_SCHEMA_VERSION
            ):
                raise ValueError("wrong case artifact schema version")
            if (
                handle.attrs.get("campaign_manifest_sha256")
                != campaign.campaign_manifest_sha256
            ):
                raise TaskCheckpointMismatchError(
                    "case artifact belongs to another campaign."
                )
            if handle.attrs.get("task_id") != task.task_id:
                raise TaskCheckpointMismatchError(
                    "case artifact task identifier does not match."
                )
            if (
                handle.attrs.get("campaign_task_identity_sha256")
                != expected_identity
            ):
                raise TaskCheckpointMismatchError(
                    "case artifact identity does not match the campaign task."
                )
            status = str(handle.attrs.get("case_status"))
            if status not in _TERMINAL_CASE_STATUSES:
                raise ValueError("invalid case status")
            evidence_bytes = bytes(
                np.asarray(handle["evidence_json"], dtype=np.uint8).tobytes()
            )
            evidence = json.loads(evidence_bytes.decode("ascii"))
            if evidence_bytes != _canonical_json(evidence).encode("ascii"):
                raise ValueError("evidence JSON is not canonical")
            evidence_sha256 = _sha256_bytes(evidence_bytes)
            if handle.attrs.get("evidence_sha256") != evidence_sha256:
                raise ValueError("evidence SHA256 mismatch")
            arrays = handle["arrays"]
            array_hashes: dict[str, str] = {}
            for name in sorted(arrays):
                if _SAFE_ARRAY_NAME.fullmatch(name) is None:
                    raise ValueError("invalid stored array name")
                dataset = arrays[name]
                digest = _array_sha256(np.asarray(dataset))
                if dataset.attrs.get("sha256") != digest:
                    raise ValueError(f"array {name!r} SHA256 mismatch")
                array_hashes[name] = digest
            recorded = json.loads(
                str(handle.attrs.get("array_sha256_json", "{}"))
            )
            if recorded != array_hashes:
                raise ValueError("array SHA256 manifest mismatch")
    except TaskCheckpointMismatchError:
        raise
    except Exception as exc:
        raise ValueError(
            f"invalid toy-case HDF5 artifact: {path}"
        ) from exc
    return {
        "schema_name": _CASE_SCHEMA,
        "schema_version": _CASE_SCHEMA_VERSION,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "campaign_task_identity_sha256": expected_identity,
        "case_status": status,
        "evidence_sha256": evidence_sha256,
        "array_sha256": array_hashes,
    }


def _artifact_validator(
    campaign: ProductionToyMPICampaign,
) -> ArtifactValidator:
    def validate(path: Path, task: TaskSpec) -> Mapping[str, Any]:
        return _case_artifact_evidence(path, task, campaign)

    return validate


def _orphan_artifact_adopter(
    campaign: ProductionToyMPICampaign,
) -> mpi_tasks.OrphanArtifactAdopter:
    def adopt(
        path: Path,
        task: TaskSpec,
        evidence: Mapping[str, Any],
    ) -> TaskResult:
        expected_identity = campaign.manifest.task_identity_sha256(task)
        if (
            evidence.get("campaign_manifest_sha256")
            != campaign.campaign_manifest_sha256
            or evidence.get("campaign_task_identity_sha256")
            != expected_identity
        ):
            raise TaskCheckpointMismatchError(
                "orphan case artifact identity does not match the campaign."
            )
        status = str(evidence.get("case_status"))
        if status not in _TERMINAL_CASE_STATUSES:
            raise TaskCheckpointMismatchError(
                "orphan case artifact has no terminal scientific status."
            )
        return TaskResult(
            status=status,
            artifact_path=path,
            metadata={
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
                "campaign_task_identity_sha256": expected_identity,
                "case_status": status,
            },
        )

    return adopt


def _write_case_artifact(
    campaign: ProductionToyMPICampaign,
    task: TaskSpec,
    context: TaskContext,
    result: ProductionToyCaseResult,
) -> Path:
    target = context.artifact_path(".h5")
    evidence_bytes = _canonical_json(result.evidence).encode("ascii")
    evidence_sha256 = _sha256_bytes(evidence_bytes)
    array_hashes = {
        name: _array_sha256(values)
        for name, values in result.arrays.items()
    }
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs["schema_name"] = _CASE_SCHEMA
            handle.attrs["schema_version"] = _CASE_SCHEMA_VERSION
            handle.attrs["campaign_manifest_sha256"] = (
                campaign.campaign_manifest_sha256
            )
            handle.attrs["campaign_task_identity_sha256"] = (
                campaign.manifest.task_identity_sha256(task)
            )
            handle.attrs["task_id"] = task.task_id
            handle.attrs["case_status"] = result.status
            handle.attrs["evidence_sha256"] = evidence_sha256
            handle.attrs["array_sha256_json"] = _canonical_json(
                array_hashes
            )
            handle.create_dataset(
                "evidence_json",
                data=np.frombuffer(evidence_bytes, dtype=np.uint8),
            )
            arrays = handle.create_group("arrays", track_order=False)
            for name, values in result.arrays.items():
                dataset = arrays.create_dataset(name, data=values)
                dataset.attrs["sha256"] = array_hashes[name]
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _case_artifact_evidence(target, task, campaign)
            expected = {
                "schema_name": _CASE_SCHEMA,
                "schema_version": _CASE_SCHEMA_VERSION,
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
                "campaign_task_identity_sha256": (
                    campaign.manifest.task_identity_sha256(task)
                ),
                "case_status": result.status,
                "evidence_sha256": evidence_sha256,
                "array_sha256": array_hashes,
            }
            if existing != expected:
                raise DuplicateTaskCheckpointError(
                    "an incompatible orphan case artifact already exists: "
                    f"{target}"
                )
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


@dataclass(frozen=True, slots=True)
class ProductionToyCampaignOutcome:
    task_id: str
    wave_index: int
    status: str
    checkpoint_relative_path: str
    checkpoint_sha256: str
    artifact_relative_path: str | None
    artifact_sha256: str | None
    artifact_validation: Mapping[str, Any]
    resumed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "wave_index": self.wave_index,
            "status": self.status,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "artifact_relative_path": self.artifact_relative_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_validation": _strict_json_value(
                self.artifact_validation
            ),
            "resumed": self.resumed,
        }


@dataclass(frozen=True, slots=True)
class ProductionToyMPIExecutionReport:
    execution_id: str
    campaign_manifest_sha256: str
    mode: Literal["serial", "mpi"]
    communicator_size: int
    coordinator_hostname: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    phase_execution_sha256: tuple[str, ...]
    outcomes: tuple[ProductionToyCampaignOutcome, ...]

    @property
    def complete(self) -> bool:
        return all(
            outcome.status in {
                "completed",
                "scientific_failure",
                "blocked",
            }
            for outcome in self.outcomes
        )

    @property
    def scientific_sha256(self) -> str:
        return _hash_document(
            {
                "campaign_manifest_sha256": (
                    self.campaign_manifest_sha256
                ),
                "outcomes": [
                    {
                        "task_id": outcome.task_id,
                        "status": outcome.status,
                        "artifact_validation": _strict_json_value(
                            outcome.artifact_validation
                        ),
                    }
                    for outcome in self.outcomes
                ],
            }
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_name": _EXECUTION_SCHEMA,
            "schema_version": _EXECUTION_SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "mode": self.mode,
            "communicator_size": self.communicator_size,
            "coordinator_hostname": self.coordinator_hostname,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "complete": self.complete,
            "scientific_sha256": self.scientific_sha256,
            "phase_execution_sha256": list(
                self.phase_execution_sha256
            ),
            "outcomes": [outcome.as_dict() for outcome in self.outcomes],
        }

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload()
        return {
            "payload": payload,
            "payload_sha256": _hash_document(payload),
        }


def _phase_paths(
    root: Path,
    campaign: ProductionToyMPICampaign,
    task_id: str,
) -> tuple[TaskManifest, Path, Path, Path]:
    wave_index = campaign.wave_index(task_id)
    phase = _phase_manifest(campaign, wave_index)
    phase_root = root / "phases" / _phase_slug(
        wave_index,
        campaign.waves[wave_index],
    )
    artifact_root = phase_root / "task-artifacts"
    checkpoint_root = phase_root / "task-outcomes"
    return phase, phase_root, artifact_root, checkpoint_root


def _load_dependency_artifacts(
    root: Path,
    campaign: ProductionToyMPICampaign,
    task: TaskSpec,
    *,
    validator: ArtifactValidator,
) -> Mapping[str, Path]:
    artifacts: dict[str, Path] = {}
    for dependency in _task_dependencies(task):
        phase, _, artifact_root, checkpoint_root = _phase_paths(
            root,
            campaign,
            dependency,
        )
        checkpoint = load_task_checkpoint(
            task_checkpoint_path(phase, dependency, checkpoint_root),
            manifest=phase,
            artifact_root=artifact_root,
            artifact_validator=validator,
        )
        if checkpoint.status != "completed":
            raise TaskCheckpointMismatchError(
                f"dependency {dependency!r} is not a completed outcome."
            )
        if checkpoint.artifact_relative_path is None:
            raise TaskCheckpointMismatchError(
                f"dependency {dependency!r} lacks its case artifact."
            )
        evidence = checkpoint.artifact_validation
        expected_identity = campaign.manifest.task_identity_sha256(
            campaign.manifest.task(dependency)
        )
        if evidence.get("campaign_task_identity_sha256") != expected_identity:
            raise TaskCheckpointMismatchError(
                f"dependency {dependency!r} has the wrong identity."
            )
        artifacts[dependency] = (
            artifact_root / checkpoint.artifact_relative_path
        )
    return MappingProxyType(artifacts)


def _collect_campaign_outcomes(
    root: Path,
    campaign: ProductionToyMPICampaign,
    *,
    validator: ArtifactValidator,
    phase_reports: Sequence[MPIExecutionReport],
) -> tuple[ProductionToyCampaignOutcome, ...]:
    resumed_by_task = {
        assignment.task_id: assignment.resumed
        for report in phase_reports
        for assignment in report.assignments
    }
    by_id: dict[str, ProductionToyCampaignOutcome] = {}
    for wave_index, task_ids in enumerate(campaign.waves):
        phase = _phase_manifest(campaign, wave_index)
        phase_root = root / "phases" / _phase_slug(wave_index, task_ids)
        artifact_root = phase_root / "task-artifacts"
        checkpoint_root = phase_root / "task-outcomes"
        checkpoints = collect_task_checkpoints(
            phase,
            checkpoint_root,
            artifact_root,
            artifact_validator=validator,
            require_complete=True,
        )
        for checkpoint in checkpoints:
            if checkpoint.task_id in by_id:
                raise DuplicateTaskCheckpointError(
                    f"campaign task {checkpoint.task_id!r} appears twice."
                )
            if checkpoint.status in _TERMINAL_CASE_STATUSES:
                if checkpoint.artifact_relative_path is None:
                    raise TaskCheckpointMismatchError(
                        "executed outcomes require HDF5 artifacts."
                    )
                if (
                    checkpoint.artifact_validation.get("case_status")
                    != checkpoint.status
                ):
                    raise TaskCheckpointMismatchError(
                        "task sidecar and artifact disagree on status."
                    )
            elif checkpoint.status == "blocked":
                if checkpoint.artifact_relative_path is not None:
                    raise TaskCheckpointMismatchError(
                        "blocked outcomes cannot claim artifacts."
                    )
            else:
                raise TaskCheckpointMismatchError(
                    f"invalid campaign status {checkpoint.status!r}."
                )
            checkpoint_path = task_checkpoint_path(
                phase,
                checkpoint.task_id,
                checkpoint_root,
            )
            artifact_relative: str | None = None
            if checkpoint.artifact_relative_path is not None:
                artifact_relative = (
                    artifact_root / checkpoint.artifact_relative_path
                ).relative_to(root).as_posix()
            by_id[checkpoint.task_id] = ProductionToyCampaignOutcome(
                task_id=checkpoint.task_id,
                wave_index=wave_index,
                status=checkpoint.status,
                checkpoint_relative_path=checkpoint_path.relative_to(
                    root
                ).as_posix(),
                checkpoint_sha256=_file_sha256(checkpoint_path),
                artifact_relative_path=artifact_relative,
                artifact_sha256=checkpoint.artifact_sha256,
                artifact_validation=checkpoint.artifact_validation,
                resumed=bool(
                    resumed_by_task.get(checkpoint.task_id, False)
                ),
            )
    missing = [
        task_id
        for task_id in campaign.canonical_task_ids
        if task_id not in by_id
    ]
    if missing:
        raise MissingTaskCheckpointError(missing)
    return tuple(by_id[task_id] for task_id in campaign.canonical_task_ids)


def collect_production_toy_campaign_outcomes(
    campaign: ProductionToyMPICampaign,
    run_directory: str | Path,
) -> tuple[ProductionToyCampaignOutcome, ...]:
    """Strictly validate and canonically merge an existing campaign."""

    root = Path(run_directory).expanduser().resolve()
    return _collect_campaign_outcomes(
        root,
        campaign,
        validator=_artifact_validator(campaign),
        phase_reports=(),
    )


def _broadcast(
    communicator: Any,
    value: Any,
    *,
    rank: int,
) -> Any:
    if int(communicator.Get_size()) == 1:
        return value
    return communicator.bcast(value if rank == 0 else None, root=0)


def execute_production_toy_mpi_campaign(
    campaign: ProductionToyMPICampaign,
    case_executor: CampaignCaseExecutor,
    run_directory: str | Path,
    *,
    comm: Any | None = None,
    require_mpi: bool = False,
    thread_limit: int = 1,
    execution_id: str | None = None,
) -> ProductionToyMPIExecutionReport | None:
    """Execute the immutable toy DAG serially or through task-level MPI.

    Every rank calls this function. Rank zero returns the canonical campaign
    report; worker ranks return ``None``.
    """

    if not isinstance(campaign, ProductionToyMPICampaign):
        raise TypeError("campaign must be a ProductionToyMPICampaign.")
    if not callable(case_executor):
        raise TypeError("case_executor must be callable.")
    communicator = mpi_tasks._resolve_communicator(  # private-to-private
        comm,
        require_mpi=require_mpi,
    )
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    root = Path(run_directory).expanduser().resolve()
    validator = _artifact_validator(campaign)
    started_clock = time.perf_counter()
    started_utc = (
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        if rank == 0
        else None
    )

    startup_error: Exception | None = None
    selected_execution_id = execution_id
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            write_task_manifest(
                root / "campaign-manifest.json",
                campaign.manifest,
            )
            if selected_execution_id is None:
                stamp = datetime.now(timezone.utc).strftime(
                    "%Y%m%dT%H%M%S.%fZ"
                )
                selected_execution_id = f"{stamp}-{uuid4().hex[:12]}"
            if (
                not isinstance(selected_execution_id, str)
                or not selected_execution_id
                or "/" in selected_execution_id
                or "\\" in selected_execution_id
            ):
                raise ValueError("execution_id is invalid.")
        except Exception as exc:
            startup_error = exc
    startup = _broadcast(
        communicator,
        {
            "ok": startup_error is None,
            "execution_id": selected_execution_id,
            "started_utc": started_utc,
            "error_type": (
                None if startup_error is None else type(startup_error).__name__
            ),
            "error_message": (
                None if startup_error is None else str(startup_error)
            ),
        }
        if rank == 0
        else None,
        rank=rank,
    )
    if not startup["ok"]:
        if rank == 0 and startup_error is not None:
            raise startup_error
        raise TaskManifestError(
            "campaign startup failed on rank zero: "
            f"{startup['error_type']}: {startup['error_message']}"
        )
    selected_execution_id = str(startup["execution_id"])
    started_utc = str(startup["started_utc"])

    statuses: dict[str, str] = {}
    phase_reports: list[MPIExecutionReport] = []
    for wave_index, task_ids in enumerate(campaign.waves):
        phase = _phase_manifest(campaign, wave_index)
        phase_root = root / "phases" / _phase_slug(wave_index, task_ids)
        prepare_error: Exception | None = None
        action: Mapping[str, str] | None = None
        if rank == 0:
            try:
                action = {
                    task_id: (
                        "run"
                        if all(
                            statuses.get(dependency) == "completed"
                            for dependency in _task_dependencies(
                                campaign.manifest.task(task_id)
                            )
                        )
                        else "block"
                    )
                    for task_id in task_ids
                }
            except Exception as exc:
                prepare_error = exc
        phase_start = _broadcast(
            communicator,
            {
                "ok": prepare_error is None,
                "action": action,
                "error_type": (
                    None
                    if prepare_error is None
                    else type(prepare_error).__name__
                ),
                "error_message": (
                    None if prepare_error is None else str(prepare_error)
                ),
            }
            if rank == 0
            else None,
            rank=rank,
        )
        if not phase_start["ok"]:
            if rank == 0 and prepare_error is not None:
                raise prepare_error
            raise TaskManifestError(
                "campaign phase preparation failed on rank zero: "
                f"{phase_start['error_type']}: "
                f"{phase_start['error_message']}"
            )
        action = phase_start["action"]
        if not isinstance(action, Mapping):
            raise TaskManifestError(
                "campaign phase action record is not a mapping."
            )

        def worker(task: TaskSpec, context: TaskContext) -> TaskResult:
            dependencies = _task_dependencies(task)
            if action[task.task_id] == "block":
                failed_dependencies = [
                    dependency
                    for dependency in dependencies
                    if statuses.get(dependency) != "completed"
                ]
                return TaskResult(
                    status="blocked",
                    metadata={
                        "campaign_manifest_sha256": (
                            campaign.campaign_manifest_sha256
                        ),
                        "campaign_task_identity_sha256": (
                            campaign.manifest.task_identity_sha256(task)
                        ),
                        "failed_dependencies": failed_dependencies,
                        "reason": (
                            "an exact prerequisite did not complete under "
                            "the immutable controls"
                        ),
                    },
                )
            dependency_artifacts = _load_dependency_artifacts(
                root,
                campaign,
                task,
                validator=validator,
            )
            case_context = ProductionToyCaseContext(
                task_context=context,
                campaign_manifest_sha256=(
                    campaign.campaign_manifest_sha256
                ),
                campaign_task_identity_sha256=(
                    campaign.manifest.task_identity_sha256(task)
                ),
                dependency_artifacts=dependency_artifacts,
                continuation_parent_task_id=_continuation_parent(task),
            )
            result = case_executor(task, case_context)
            if not isinstance(result, ProductionToyCaseResult):
                raise TypeError(
                    "case_executor must return ProductionToyCaseResult."
                )
            artifact = _write_case_artifact(
                campaign,
                task,
                context,
                result,
            )
            return TaskResult(
                status=result.status,
                artifact_path=artifact,
                metadata={
                    "campaign_manifest_sha256": (
                        campaign.campaign_manifest_sha256
                    ),
                    "campaign_task_identity_sha256": (
                        campaign.manifest.task_identity_sha256(task)
                    ),
                    "case_status": result.status,
                },
            )

        phase_report = execute_task_manifest(
            phase,
            worker,
            phase_root,
            comm=communicator,
            require_mpi=require_mpi,
            thread_limit=thread_limit,
            artifact_validator=validator,
            orphan_artifact_suffix=".h5",
            orphan_artifact_adopter=_orphan_artifact_adopter(
                campaign
            ),
            execution_id=(
                f"{selected_execution_id}.phase-{wave_index:03d}"
            ),
        )
        phase_error: Exception | None = None
        wave_statuses: Mapping[str, str] | None = None
        if rank == 0:
            try:
                assert phase_report is not None
                checkpoints = collect_task_checkpoints(
                    phase,
                    phase_root / "task-outcomes",
                    phase_root / "task-artifacts",
                    artifact_validator=validator,
                    require_complete=True,
                )
                wave_statuses = {
                    checkpoint.task_id: checkpoint.status
                    for checkpoint in checkpoints
                }
                phase_reports.append(phase_report)
            except Exception as exc:
                phase_error = exc
        phase_terminal = _broadcast(
            communicator,
            {
                "ok": phase_error is None,
                "statuses": wave_statuses,
                "error_type": (
                    None
                    if phase_error is None
                    else type(phase_error).__name__
                ),
                "error_message": (
                    None if phase_error is None else str(phase_error)
                ),
            }
            if rank == 0
            else None,
            rank=rank,
        )
        if not phase_terminal["ok"]:
            if rank == 0 and phase_error is not None:
                raise phase_error
            raise mpi_tasks.TaskExecutionError(
                "campaign phase finalization failed on rank zero: "
                f"{phase_terminal['error_type']}: "
                f"{phase_terminal['error_message']}"
            )
        wave_statuses = phase_terminal["statuses"]
        if not isinstance(wave_statuses, Mapping):
            raise mpi_tasks.TaskExecutionError(
                "campaign phase terminal record lacks statuses."
            )
        statuses.update(
            {str(key): str(value) for key, value in wave_statuses.items()}
        )

    report: ProductionToyMPIExecutionReport | None = None
    final_error: Exception | None = None
    if rank == 0:
        try:
            outcomes = _collect_campaign_outcomes(
                root,
                campaign,
                validator=validator,
                phase_reports=phase_reports,
            )
            finished_utc = datetime.now(timezone.utc).isoformat().replace(
                "+00:00",
                "Z",
            )
            report = ProductionToyMPIExecutionReport(
                execution_id=selected_execution_id,
                campaign_manifest_sha256=(
                    campaign.campaign_manifest_sha256
                ),
                mode="serial" if size == 1 else "mpi",
                communicator_size=size,
                coordinator_hostname=socket.gethostname(),
                started_utc=started_utc,
                finished_utc=finished_utc,
                elapsed_seconds=time.perf_counter() - started_clock,
                phase_execution_sha256=tuple(
                    phase.execution_sha256 for phase in phase_reports
                ),
                outcomes=outcomes,
            )
            destination = (
                root
                / "campaign-executions"
                / f"{selected_execution_id}.json"
            )
            mpi_tasks._publish_immutable(  # private-to-private
                destination,
                _canonical_bytes(report.as_dict()),
                conflict_type=TaskManifestError,
            )
        except Exception as exc:
            final_error = exc
    final_terminal = _broadcast(
        communicator,
        {
            "ok": final_error is None,
            "error_type": (
                None
                if final_error is None
                else type(final_error).__name__
            ),
            "error_message": (
                None if final_error is None else str(final_error)
            ),
        }
        if rank == 0
        else None,
        rank=rank,
    )
    if not final_terminal["ok"]:
        if rank == 0 and final_error is not None:
            raise final_error
        raise mpi_tasks.TaskExecutionError(
            "campaign finalization failed on rank zero: "
            f"{final_terminal['error_type']}: "
            f"{final_terminal['error_message']}"
        )
    return report if rank == 0 else None


__all__ = [
    "CampaignCaseExecutor",
    "ProductionToyCampaignOutcome",
    "ProductionToyCaseContext",
    "ProductionToyCaseResult",
    "ProductionToyMPICampaign",
    "ProductionToyMPIExecutionReport",
    "build_production_toy_mpi_campaign",
    "collect_production_toy_campaign_outcomes",
    "execute_production_toy_mpi_campaign",
]
