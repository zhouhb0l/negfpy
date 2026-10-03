"""Private atomic checkpoints for exact sparse-material SCBA continuation.

Version 1 checkpoints archive terminal dense-SCBA results.  This module keeps
that format immutable and implements the different state machine needed by a
long-running sparse material calculation:

* the durable numerical payload is only the mixed lesser/greater ``sigma_pp``
  pair plus convergence history and the small adaptive-mixing state;
* an in-progress record denotes an iteration boundary which can be replayed
  deterministically after an infrastructure interruption;
* completed and scientifically failed records are terminal; and
* Green functions, currents, retarded self-energies, and diagnostics are never
  trusted from disk and are recomputed by the solver on restart.

The public :func:`solve_scba` and :func:`solve_one_shot_born` signatures stay
unchanged.  Checkpointing is activated only through the private
:func:`solve_sparse_scba_checkpointed` entry point and a context-local solver
hook.  ``h5py`` is imported lazily by the shared checkpoint primitives.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import re
import time
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np

from ._checkpoint import (
    CheckpointCorruptionError,
    CheckpointDependencyError,
    CheckpointError,
    CheckpointMismatchError,
    _HISTORY_DTYPE,
    _array_sha256,
    _fsync_directory,
    _fsync_file,
    _history_array,
    _history_records,
    _history_sha256,
    _require_h5py,
    _sha256_bytes,
    _text,
)
from ._provenance import (
    code_provenance,
    configuration_hash,
    initial_self_energy_hash,
    problem_input_hashes,
)
from .types import (
    AnharmonicResult,
    CubicSCBAProblem,
    KeldyshSpectrum,
    SCBAConfig,
    SCBAIteration,
)


Array = np.ndarray
_SCHEMA_NAME = "negfpy_sparse_material_scba_checkpoint"
_SCHEMA_VERSION = 2
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESUMABLE_STATUSES = frozenset({"in_progress", "infrastructure_interrupted"})
_TERMINAL_STATUSES = frozenset({"completed", "scientific_failure"})
_STATUSES = _RESUMABLE_STATUSES | _TERMINAL_STATUSES
_ROOT_KEYS = frozenset(
    {
        "identity_json",
        "sigma_pp",
        "iteration_history",
        "timings_json",
        "solver_state_json",
        "metadata_json",
    }
)
_ROOT_ATTRIBUTES = frozenset(
    {
        "schema_name",
        "schema_version",
        "write_state",
        "status",
        "identity_sha256",
        "manifest_sha256",
    }
)


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256.")
    return value


def _validate_run_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run_id must be a non-empty string.")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError("run_id cannot contain surrounding whitespace or controls.")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _normalize_json(value.item(), path=path)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return result
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings.")
            normalized[key] = _normalize_json(item, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (tuple, list)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported checkpoint metadata type "
        f"{type(value).__name__}."
    )


def _canonical(value: Any) -> str:
    return json.dumps(
        _normalize_json(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_mapping(values: Mapping[str, str], *, domain: str) -> str:
    return _sha256_bytes((domain + "\0" + _canonical(dict(values))).encode("ascii"))


def _validated_hash_mapping(
    values: Mapping[str, Any],
    *,
    name: str,
    required: frozenset[str] = frozenset(),
) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings.")
        normalized[key] = _validate_sha256(value, name=f"{name}[{key!r}]")
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"{name} is missing required keys: {sorted(missing)!r}.")
    return MappingProxyType(dict(sorted(normalized.items())))


def _collect_metadata_hashes(
    value: Any,
    *,
    prefix: str,
    target: dict[str, str],
) -> None:
    """Collect explicitly named material digests without interpreting metadata."""

    if isinstance(value, Mapping):
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.lower().endswith("sha256") and isinstance(item, str):
                if _SHA256_RE.fullmatch(item) is not None:
                    target[path] = item
            _collect_metadata_hashes(item, prefix=path, target=target)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _collect_metadata_hashes(
                item,
                prefix=f"{prefix}[{index}]",
                target=target,
            )


def _material_source_hashes(
    problem: CubicSCBAProblem,
    supplied: Mapping[str, Any] | str | None,
) -> Mapping[str, str]:
    from .sparse_cubic import SparseCubicVertex

    if not isinstance(problem.vertex, SparseCubicVertex):
        raise TypeError(
            "checkpoint v2 is restricted to problems with SparseCubicVertex."
        )
    automatic: dict[str, str] = {
        "cubic_vertex_physics": problem.vertex.cubic_physics_sha256,
        "cubic_vertex_storage": problem.vertex.vertex_storage_sha256,
    }
    _collect_metadata_hashes(
        problem.vertex.metadata,
        prefix="vertex_metadata",
        target=automatic,
    )
    _collect_metadata_hashes(
        problem.scale.metadata,
        prefix="scale_metadata",
        target=automatic,
    )
    if supplied is None:
        explicit: Mapping[str, Any] = {}
    elif isinstance(supplied, str):
        explicit = {"material_source": supplied}
    elif isinstance(supplied, Mapping):
        explicit = supplied
    else:
        raise TypeError("source_hashes must be a SHA256 string, mapping, or None.")
    for key, value in explicit.items():
        digest = _validate_sha256(value, name=f"source_hashes[{key!r}]")
        if key in automatic and automatic[key] != digest:
            raise ValueError(f"source_hashes[{key!r}] conflicts with problem metadata.")
        automatic[str(key)] = digest
    return _validated_hash_mapping(automatic, name="source_hashes")


@dataclass(frozen=True, slots=True)
class SCBACheckpointIdentityV2:
    """Exact identity of one sparse material SCBA calculation."""

    run_id: str
    configuration_sha256: str
    problem_sha256: str
    grid_sha256: str
    initial_state_sha256: str
    source_tree_sha256: str
    environment_sha256: str
    problem_hashes: Mapping[str, str]
    source_hashes: Mapping[str, str]

    schema_name: ClassVar[str] = _SCHEMA_NAME
    schema_version: ClassVar[int] = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _validate_run_id(self.run_id))
        for name in (
            "configuration_sha256",
            "problem_sha256",
            "grid_sha256",
            "initial_state_sha256",
            "source_tree_sha256",
            "environment_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )
        problem_hashes = _validated_hash_mapping(
            self.problem_hashes,
            name="problem_hashes",
            required=frozenset({"problem", "frequency_grid"}),
        )
        source_hashes = _validated_hash_mapping(
            self.source_hashes,
            name="source_hashes",
        )
        if not source_hashes:
            raise ValueError("source_hashes must contain at least one exact source.")
        if problem_hashes["problem"] != self.problem_sha256:
            raise ValueError("problem_sha256 does not equal problem_hashes['problem'].")
        if problem_hashes["frequency_grid"] != self.grid_sha256:
            raise ValueError(
                "grid_sha256 does not equal problem_hashes['frequency_grid']."
            )
        object.__setattr__(self, "problem_hashes", problem_hashes)
        object.__setattr__(self, "source_hashes", source_hashes)

    @property
    def problem_hashes_sha256(self) -> str:
        return _hash_mapping(
            self.problem_hashes,
            domain="NEGFPY-SCBA-V2-PROBLEM-HASHES",
        )

    @property
    def source_hashes_sha256(self) -> str:
        return _hash_mapping(
            self.source_hashes,
            domain="NEGFPY-SCBA-V2-SOURCE-HASHES",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "configuration_sha256": self.configuration_sha256,
            "problem_sha256": self.problem_sha256,
            "grid_sha256": self.grid_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "environment_sha256": self.environment_sha256,
            "problem_hashes": dict(self.problem_hashes),
            "problem_hashes_sha256": self.problem_hashes_sha256,
            "source_hashes": dict(self.source_hashes),
            "source_hashes_sha256": self.source_hashes_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical(self.as_dict()).encode("ascii"))


def make_scba_checkpoint_identity_v2(
    run_id: str,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    environment_sha256: str,
    initial_sigma_pp: KeldyshSpectrum | None = None,
    source_hashes: Mapping[str, Any] | str | None = None,
    source_sha256: Mapping[str, Any] | str | None = None,
    source_tree_sha256: str | None = None,
) -> SCBACheckpointIdentityV2:
    """Derive a v2 identity from every input actually consumed by SCBA."""

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")
    if source_hashes is not None and source_sha256 is not None:
        raise ValueError("pass only one of source_hashes and source_sha256.")
    selected_sources = source_hashes if source_hashes is not None else source_sha256
    hashes = problem_input_hashes(problem)
    if source_tree_sha256 is None:
        discovered = code_provenance().get("source_tree_sha256")
        if not isinstance(discovered, str) or _SHA256_RE.fullmatch(discovered) is None:
            raise RuntimeError(
                "could not determine an exact NEGFPy source-tree SHA256."
            )
        source_tree_sha256 = discovered
    return SCBACheckpointIdentityV2(
        run_id=run_id,
        configuration_sha256=configuration_hash(config),
        problem_sha256=hashes["problem"],
        grid_sha256=hashes["frequency_grid"],
        initial_state_sha256=initial_self_energy_hash(problem, initial_sigma_pp),
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        problem_hashes=hashes,
        source_hashes=_material_source_hashes(problem, selected_sources),
    )


@dataclass(frozen=True, slots=True)
class SCBAContinuationState:
    """Small adaptive state needed after replaying the checkpoint boundary."""

    mixing: float
    previous_sigma_residual: float | None
    residual_increase_streak: int
    consecutive_passes: int

    def __post_init__(self) -> None:
        if isinstance(self.mixing, (bool, np.bool_)):
            raise TypeError("mixing must be a real number.")
        mixing = float(self.mixing)
        if not np.isfinite(mixing) or not 0.0 < mixing <= 1.0:
            raise ValueError("mixing must be finite and in (0, 1].")
        previous = self.previous_sigma_residual
        if previous is not None:
            if isinstance(previous, (bool, np.bool_)):
                raise TypeError("previous_sigma_residual must be real or None.")
            previous = float(previous)
            if np.isnan(previous) or previous < 0.0:
                raise ValueError(
                    "previous_sigma_residual must be non-negative and not NaN."
                )
        for name in ("residual_increase_streak", "consecutive_passes"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "mixing", mixing)
        object.__setattr__(self, "previous_sigma_residual", previous)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mixing": self.mixing,
            "previous_sigma_residual": self.previous_sigma_residual,
            "residual_increase_streak": self.residual_increase_streak,
            "consecutive_passes": self.consecutive_passes,
        }


def _readonly_sigma(values: Any, *, name: str) -> Array:
    array = np.array(values, dtype=np.complex128, order="C", copy=True)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must have shape (n_frequency, ndof, ndof).")
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _validated_history(values: Any) -> tuple[SCBAIteration, ...]:
    history = tuple(values)
    if any(not isinstance(item, SCBAIteration) for item in history):
        raise TypeError("history must contain only SCBAIteration records.")
    expected = tuple(range(1, len(history) + 1))
    actual = tuple(item.iteration for item in history)
    if actual != expected:
        raise ValueError("checkpoint history iterations must be contiguous from one.")
    return history


def _validated_timings(
    values: Mapping[str, Any],
    *,
    history_length: int,
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("timings must be a mapping.")
    normalized = _normalize_json(values, path="$.timings")
    if not isinstance(normalized, dict):
        raise TypeError("timings must be a mapping.")
    required = {
        "elapsed_seconds",
        "reconstruction_seconds",
        "iteration_seconds",
        "attempts",
    }
    if set(normalized) != required:
        raise ValueError(f"timings must contain exactly {sorted(required)!r}.")
    for name in ("elapsed_seconds", "reconstruction_seconds"):
        value = normalized[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"timings[{name!r}] must be a real number.")
        if float(value) < 0.0:
            raise ValueError(f"timings[{name!r}] must be non-negative.")
        normalized[name] = float(value)
    attempts = normalized["attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("timings['attempts'] must be a positive integer.")
    iteration_seconds = normalized["iteration_seconds"]
    if not isinstance(iteration_seconds, list):
        raise TypeError("timings['iteration_seconds'] must be a sequence.")
    if len(iteration_seconds) != history_length:
        raise ValueError(
            "iteration_seconds length must equal the iteration history length."
        )
    normalized_iterations: list[float] = []
    for index, value in enumerate(iteration_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"iteration_seconds[{index}] must be real.")
        number = float(value)
        if number < 0.0:
            raise ValueError(f"iteration_seconds[{index}] must be non-negative.")
        normalized_iterations.append(number)
    normalized["iteration_seconds"] = normalized_iterations
    return _freeze_json(normalized)


@dataclass(frozen=True, slots=True, eq=False)
class SCBACheckpointV2:
    """One resumable or terminal sparse-material SCBA checkpoint."""

    identity: SCBACheckpointIdentityV2
    sigma_lesser: Array
    sigma_greater: Array
    history: tuple[SCBAIteration, ...] = ()
    timings: Mapping[str, Any] = field(
        default_factory=lambda: {
            "elapsed_seconds": 0.0,
            "reconstruction_seconds": 0.0,
            "iteration_seconds": (),
            "attempts": 1,
        }
    )
    status: str = "in_progress"
    solver_state: SCBAContinuationState = field(
        default_factory=lambda: SCBAContinuationState(0.2, None, 0, 0)
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SCBACheckpointIdentityV2):
            raise TypeError("identity must be an SCBACheckpointIdentityV2.")
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {sorted(_STATUSES)!r}.")
        lesser = _readonly_sigma(self.sigma_lesser, name="sigma_lesser")
        greater = _readonly_sigma(self.sigma_greater, name="sigma_greater")
        if lesser.shape != greater.shape:
            raise ValueError("sigma_lesser and sigma_greater must share one shape.")
        history = _validated_history(self.history)
        if not isinstance(self.solver_state, SCBAContinuationState):
            raise TypeError("solver_state must be an SCBAContinuationState.")
        if history:
            last = history[-1]
            if self.solver_state.mixing != last.mixing:
                raise ValueError(
                    "solver_state mixing must equal the last history value."
                )
            if self.solver_state.consecutive_passes != last.consecutive_passes:
                raise ValueError(
                    "solver_state consecutive_passes must equal the last history value."
                )
            if self.status in _RESUMABLE_STATUSES and (
                self.solver_state.previous_sigma_residual != last.sigma_residual
            ):
                raise ValueError(
                    "resumable solver_state must retain the last sigma residual."
                )
        elif self.solver_state.previous_sigma_residual is not None:
            raise ValueError("empty history requires no previous sigma residual.")
        timings = _validated_timings(self.timings, history_length=len(history))
        metadata = _normalize_json(self.metadata, path="$.metadata")
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "sigma_lesser", lesser)
        object.__setattr__(self, "sigma_greater", greater)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "timings", timings)
        object.__setattr__(self, "metadata", _freeze_json(metadata))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.sigma_lesser.shape

    @property
    def terminal(self) -> bool:
        return self.status in _TERMINAL_STATUSES

    @property
    def resumable(self) -> bool:
        return (
            self.status in _RESUMABLE_STATUSES
            and self.metadata.get("anderson_resume_supported") is not False
        )


# More descriptive aliases for callers that prefer the sparse-specific names.
SparseSCBACheckpointIdentity = SCBACheckpointIdentityV2
SparseSCBACheckpoint = SCBACheckpointV2


def _identity_from_dict(value: Mapping[str, Any]) -> SCBACheckpointIdentityV2:
    expected = {
        "schema_name",
        "schema_version",
        "run_id",
        "configuration_sha256",
        "problem_sha256",
        "grid_sha256",
        "initial_state_sha256",
        "source_tree_sha256",
        "environment_sha256",
        "problem_hashes",
        "problem_hashes_sha256",
        "source_hashes",
        "source_hashes_sha256",
    }
    if set(value) != expected:
        raise CheckpointCorruptionError(
            "checkpoint v2 identity has missing or unexpected fields."
        )
    if value["schema_name"] != _SCHEMA_NAME or value["schema_version"] != 2:
        raise CheckpointCorruptionError("checkpoint v2 identity schema is invalid.")
    try:
        identity = SCBACheckpointIdentityV2(
            run_id=value["run_id"],
            configuration_sha256=value["configuration_sha256"],
            problem_sha256=value["problem_sha256"],
            grid_sha256=value["grid_sha256"],
            initial_state_sha256=value["initial_state_sha256"],
            source_tree_sha256=value["source_tree_sha256"],
            environment_sha256=value["environment_sha256"],
            problem_hashes=value["problem_hashes"],
            source_hashes=value["source_hashes"],
        )
    except (TypeError, ValueError) as exc:
        raise CheckpointCorruptionError("checkpoint v2 identity is invalid.") from exc
    if value["problem_hashes_sha256"] != identity.problem_hashes_sha256:
        raise CheckpointCorruptionError(
            "checkpoint problem-hash manifest failed its SHA256 check."
        )
    if value["source_hashes_sha256"] != identity.source_hashes_sha256:
        raise CheckpointCorruptionError(
            "checkpoint source-hash manifest failed its SHA256 check."
        )
    return identity


def _raise_identity_mismatch(
    stored: SCBACheckpointIdentityV2,
    expected: SCBACheckpointIdentityV2,
) -> None:
    comparisons = (
        ("run id", stored.run_id, expected.run_id),
        (
            "configuration hash",
            stored.configuration_sha256,
            expected.configuration_sha256,
        ),
        ("problem hash", stored.problem_sha256, expected.problem_sha256),
        ("frequency-grid hash", stored.grid_sha256, expected.grid_sha256),
        (
            "initial-state hash",
            stored.initial_state_sha256,
            expected.initial_state_sha256,
        ),
        (
            "source-tree hash",
            stored.source_tree_sha256,
            expected.source_tree_sha256,
        ),
        (
            "environment hash",
            stored.environment_sha256,
            expected.environment_sha256,
        ),
        (
            "exact problem/input hash manifest",
            dict(stored.problem_hashes),
            dict(expected.problem_hashes),
        ),
        (
            "material source hash manifest",
            dict(stored.source_hashes),
            dict(expected.source_hashes),
        ),
    )
    for label, actual, wanted in comparisons:
        if actual != wanted:
            raise CheckpointMismatchError(
                f"checkpoint {label} does not match the requested exact run."
            )
    raise CheckpointMismatchError(
        "checkpoint identity does not match the requested exact run."
    )


def _write_text_dataset(handle: Any, name: str, text: str, h5py: Any) -> str:
    digest = _sha256_bytes(text.encode("ascii"))
    dataset = handle.create_dataset(
        name,
        data=text,
        dtype=h5py.string_dtype("utf-8"),
    )
    dataset.attrs["sha256"] = digest
    return digest


def _load_json_object(handle: Any, name: str) -> tuple[dict[str, Any], str]:
    dataset = handle[name]
    if set(dataset.attrs) != {"sha256"}:
        raise CheckpointCorruptionError(
            f"checkpoint {name} has missing or unexpected attributes."
        )
    text = _text(dataset[()], name=name)
    digest = _validate_sha256(
        _text(dataset.attrs["sha256"], name=f"{name}.sha256"),
        name=f"{name}.sha256",
    )
    if _sha256_bytes(text.encode("ascii")) != digest:
        raise CheckpointCorruptionError(
            f"checkpoint {name} failed its SHA256 integrity check."
        )
    try:
        value = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointCorruptionError(
            f"checkpoint {name} is not valid strict JSON."
        ) from exc
    if not isinstance(value, dict):
        raise CheckpointCorruptionError(
            f"checkpoint {name} must contain a JSON object."
        )
    try:
        canonical = _canonical(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointCorruptionError(
            f"checkpoint {name} contains invalid JSON."
        ) from exc
    if canonical != text:
        raise CheckpointCorruptionError(f"checkpoint {name} is not canonical JSON.")
    return value, digest


def _require_exact_hard_children(
    group: Any,
    names: frozenset[str],
    h5py: Any,
    *,
    label: str,
) -> None:
    if set(group.keys()) != names:
        raise CheckpointCorruptionError(
            f"checkpoint {label} has missing or unexpected objects."
        )
    for name in names:
        link = group.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise CheckpointCorruptionError(
                f"checkpoint {label}/{name} is not an internal hard link."
            )


def _timing_iterations(checkpoint: SCBACheckpointV2) -> tuple[float, ...]:
    return tuple(float(value) for value in checkpoint.timings["iteration_seconds"])


def _validate_transition(
    previous: SCBACheckpointV2,
    current: SCBACheckpointV2,
) -> None:
    if previous.identity.identity_sha256 != current.identity.identity_sha256:
        raise CheckpointMismatchError(
            "an existing checkpoint cannot be relabelled for another run."
        )
    if previous.terminal:
        raise CheckpointMismatchError(
            "terminal checkpoint outcomes are immutable and cannot be resumed."
        )
    if len(current.history) < len(previous.history):
        raise CheckpointMismatchError("checkpoint history cannot move backwards.")
    if current.history[: len(previous.history)] != previous.history:
        raise CheckpointMismatchError("checkpoint history is not an exact prefix.")
    previous_times = _timing_iterations(previous)
    current_times = _timing_iterations(current)
    if current_times[: len(previous_times)] != previous_times:
        raise CheckpointMismatchError("checkpoint iteration timings changed.")
    if len(current.history) == len(previous.history):
        if current.status != "infrastructure_interrupted":
            raise CheckpointMismatchError(
                "a checkpoint update without a new iteration must mark an "
                "infrastructure interruption."
            )
        if not np.array_equal(
            previous.sigma_lesser, current.sigma_lesser
        ) or not np.array_equal(previous.sigma_greater, current.sigma_greater):
            raise CheckpointMismatchError(
                "an interruption marker cannot alter the numerical state."
            )


def write_scba_checkpoint_v2(
    path: str | Path,
    checkpoint: SCBACheckpointV2,
) -> Path:
    """Atomically publish one exact sparse-material SCBA checkpoint."""

    if not isinstance(checkpoint, SCBACheckpointV2):
        raise TypeError("checkpoint must be an SCBACheckpointV2.")
    h5py = _require_h5py()
    target = Path(path).expanduser()
    if target.exists():
        previous = load_scba_checkpoint_v2(
            target,
            expected_identity=checkpoint.identity,
        )
        _validate_transition(previous, checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"

    lesser = np.ascontiguousarray(checkpoint.sigma_lesser, dtype="<c16")
    greater = np.ascontiguousarray(checkpoint.sigma_greater, dtype="<c16")
    history = _history_array(checkpoint.history)
    identity_json = _canonical(checkpoint.identity.as_dict())
    timings_json = _canonical(dict(checkpoint.timings))
    state_json = _canonical(checkpoint.solver_state.as_dict())
    metadata_json = _canonical(dict(checkpoint.metadata))

    integrity = {
        "identity_json_sha256": _sha256_bytes(identity_json.encode("ascii")),
        "sigma_lesser_sha256": _array_sha256(lesser),
        "sigma_greater_sha256": _array_sha256(greater),
        "history_sha256": _history_sha256(history),
        "timings_json_sha256": _sha256_bytes(timings_json.encode("ascii")),
        "solver_state_json_sha256": _sha256_bytes(state_json.encode("ascii")),
        "metadata_json_sha256": _sha256_bytes(metadata_json.encode("ascii")),
    }
    manifest_payload = {
        "identity_sha256": checkpoint.identity.identity_sha256,
        "status": checkpoint.status,
        **integrity,
    }
    manifest_sha256 = _sha256_bytes(_canonical(manifest_payload).encode("ascii"))

    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs["schema_name"] = _SCHEMA_NAME
            handle.attrs["schema_version"] = _SCHEMA_VERSION
            handle.attrs["write_state"] = "complete"
            handle.attrs["status"] = checkpoint.status
            handle.attrs["identity_sha256"] = checkpoint.identity.identity_sha256
            handle.attrs["manifest_sha256"] = manifest_sha256

            _write_text_dataset(handle, "identity_json", identity_json, h5py)
            sigma_group = handle.create_group("sigma_pp")
            lesser_dataset = sigma_group.create_dataset("lesser", data=lesser)
            greater_dataset = sigma_group.create_dataset("greater", data=greater)
            lesser_dataset.attrs["sha256"] = integrity["sigma_lesser_sha256"]
            greater_dataset.attrs["sha256"] = integrity["sigma_greater_sha256"]
            history_dataset = handle.create_dataset(
                "iteration_history",
                data=history,
                dtype=_HISTORY_DTYPE,
            )
            history_dataset.attrs["sha256"] = integrity["history_sha256"]
            _write_text_dataset(handle, "timings_json", timings_json, h5py)
            _write_text_dataset(handle, "solver_state_json", state_json, h5py)
            _write_text_dataset(handle, "metadata_json", metadata_json, h5py)
            handle.flush()

        _fsync_file(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def load_scba_checkpoint_v2(
    path: str | Path,
    *,
    expected_identity: SCBACheckpointIdentityV2,
) -> SCBACheckpointV2:
    """Load a v2 checkpoint and reject any identity or integrity mismatch."""

    if not isinstance(expected_identity, SCBACheckpointIdentityV2):
        raise TypeError("expected_identity must be an SCBACheckpointIdentityV2.")
    h5py = _require_h5py()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"SCBA checkpoint v2 does not exist: {source}")
    try:
        with h5py.File(source, "r") as handle:
            _require_exact_hard_children(
                handle,
                _ROOT_KEYS,
                h5py,
                label="root",
            )
            if set(handle.attrs) != _ROOT_ATTRIBUTES:
                raise CheckpointCorruptionError(
                    "checkpoint root has missing or unexpected attributes."
                )
            if _text(handle.attrs["schema_name"], name="schema_name") != _SCHEMA_NAME:
                raise CheckpointCorruptionError("checkpoint schema name is invalid.")
            if int(handle.attrs["schema_version"]) != _SCHEMA_VERSION:
                raise CheckpointCorruptionError("checkpoint schema version is invalid.")
            if _text(handle.attrs["write_state"], name="write_state") != "complete":
                raise CheckpointCorruptionError(
                    "checkpoint was not marked as a complete atomic write."
                )
            status = _text(handle.attrs["status"], name="status")
            if status not in _STATUSES:
                raise CheckpointCorruptionError("checkpoint status is invalid.")
            stored_identity_hash = _validate_sha256(
                _text(handle.attrs["identity_sha256"], name="identity_sha256"),
                name="identity_sha256",
            )
            stored_manifest_hash = _validate_sha256(
                _text(handle.attrs["manifest_sha256"], name="manifest_sha256"),
                name="manifest_sha256",
            )

            identity_value, identity_json_hash = _load_json_object(
                handle,
                "identity_json",
            )
            identity = _identity_from_dict(identity_value)
            if identity.identity_sha256 != stored_identity_hash:
                raise CheckpointCorruptionError(
                    "checkpoint canonical identity failed its SHA256 check."
                )
            if identity.identity_sha256 != expected_identity.identity_sha256:
                _raise_identity_mismatch(identity, expected_identity)

            sigma_group = handle["sigma_pp"]
            _require_exact_hard_children(
                sigma_group,
                frozenset({"lesser", "greater"}),
                h5py,
                label="sigma_pp",
            )
            lesser_dataset = sigma_group["lesser"]
            greater_dataset = sigma_group["greater"]
            if set(sigma_group.attrs):
                raise CheckpointCorruptionError(
                    "checkpoint sigma_pp has unexpected attributes."
                )
            if set(lesser_dataset.attrs) != {"sha256"} or set(
                greater_dataset.attrs
            ) != {"sha256"}:
                raise CheckpointCorruptionError(
                    "checkpoint sigma datasets have invalid attributes."
                )
            if lesser_dataset.dtype != np.dtype(
                "<c16"
            ) or greater_dataset.dtype != np.dtype("<c16"):
                raise CheckpointCorruptionError(
                    "checkpoint sigma datasets do not use canonical complex128."
                )
            lesser = np.asarray(lesser_dataset[...])
            greater = np.asarray(greater_dataset[...])
            lesser_hash = _validate_sha256(
                _text(lesser_dataset.attrs["sha256"], name="sigma_pp/lesser.sha256"),
                name="sigma_pp/lesser.sha256",
            )
            greater_hash = _validate_sha256(
                _text(greater_dataset.attrs["sha256"], name="sigma_pp/greater.sha256"),
                name="sigma_pp/greater.sha256",
            )
            if _array_sha256(lesser) != lesser_hash:
                raise CheckpointCorruptionError(
                    "checkpoint sigma_pp/lesser failed its SHA256 check."
                )
            if _array_sha256(greater) != greater_hash:
                raise CheckpointCorruptionError(
                    "checkpoint sigma_pp/greater failed its SHA256 check."
                )

            history_dataset = handle["iteration_history"]
            if history_dataset.dtype != _HISTORY_DTYPE:
                raise CheckpointCorruptionError(
                    "checkpoint iteration_history has a noncanonical dtype."
                )
            if set(history_dataset.attrs) != {"sha256"}:
                raise CheckpointCorruptionError(
                    "checkpoint iteration_history has invalid attributes."
                )
            history_array = np.asarray(history_dataset[...])
            history_hash = _validate_sha256(
                _text(
                    history_dataset.attrs["sha256"],
                    name="iteration_history.sha256",
                ),
                name="iteration_history.sha256",
            )
            if _history_sha256(history_array) != history_hash:
                raise CheckpointCorruptionError(
                    "checkpoint iteration_history failed its SHA256 check."
                )
            history = _history_records(history_array)
            timings, timings_hash = _load_json_object(handle, "timings_json")
            state_value, state_hash = _load_json_object(
                handle,
                "solver_state_json",
            )
            metadata, metadata_hash = _load_json_object(handle, "metadata_json")
            if set(state_value) != {
                "mixing",
                "previous_sigma_residual",
                "residual_increase_streak",
                "consecutive_passes",
            }:
                raise CheckpointCorruptionError(
                    "checkpoint solver_state_json has an invalid schema."
                )
            try:
                state = SCBAContinuationState(**state_value)
            except (TypeError, ValueError) as exc:
                raise CheckpointCorruptionError(
                    "checkpoint continuation state is invalid."
                ) from exc

            manifest_payload = {
                "identity_sha256": stored_identity_hash,
                "status": status,
                "identity_json_sha256": identity_json_hash,
                "sigma_lesser_sha256": lesser_hash,
                "sigma_greater_sha256": greater_hash,
                "history_sha256": history_hash,
                "timings_json_sha256": timings_hash,
                "solver_state_json_sha256": state_hash,
                "metadata_json_sha256": metadata_hash,
            }
            if (
                _sha256_bytes(_canonical(manifest_payload).encode("ascii"))
                != stored_manifest_hash
            ):
                raise CheckpointCorruptionError(
                    "checkpoint manifest failed its SHA256 integrity check."
                )

        try:
            return SCBACheckpointV2(
                identity=identity,
                sigma_lesser=lesser,
                sigma_greater=greater,
                history=history,
                timings=timings,
                status=status,
                solver_state=state,
                metadata=metadata,
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointCorruptionError(
                "checkpoint payload violates the v2 state contract."
            ) from exc
    except CheckpointError:
        raise
    except FileNotFoundError:
        raise
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
        raise CheckpointCorruptionError(
            f"SCBA checkpoint v2 is missing, truncated, or invalid: {source}"
        ) from exc


def load_scba_checkpoint_v2_self_validating(
    path: str | Path,
) -> SCBACheckpointV2:
    """Load a checkpoint using and validating its stored identity.

    This is intentionally separate from :func:`load_scba_checkpoint_v2`:
    normal restart always supplies an independently constructed expected
    identity, while a continuation parent is first inspected as an immutable
    artifact and then checked against the child problem by the caller.
    """

    h5py = _require_h5py()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"SCBA checkpoint v2 does not exist: {source}")
    try:
        with h5py.File(source, "r") as handle:
            identity_value, _ = _load_json_object(handle, "identity_json")
        identity = _identity_from_dict(identity_value)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise CheckpointCorruptionError(
            f"could not read the self-validating checkpoint identity: {source}"
        ) from exc
    return load_scba_checkpoint_v2(source, expected_identity=identity)


def _validate_checkpoint_for_solver(
    checkpoint: SCBACheckpointV2,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> None:
    actual_hashes = problem_input_hashes(problem)
    identity = checkpoint.identity
    if configuration_hash(config) != identity.configuration_sha256:
        raise CheckpointMismatchError(
            "checkpoint configuration hash does not match the restart config."
        )
    if actual_hashes != dict(identity.problem_hashes):
        raise CheckpointMismatchError(
            "checkpoint problem/input hashes do not match the restart problem."
        )
    if actual_hashes["frequency_grid"] != identity.grid_sha256:
        raise CheckpointMismatchError(
            "checkpoint frequency-grid hash does not match the restart grid."
        )
    expected_shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    if checkpoint.shape != expected_shape:
        raise CheckpointMismatchError(
            f"checkpoint self-energy shape {checkpoint.shape} does not match "
            f"restart shape {expected_shape}."
        )
    count = len(checkpoint.history)
    mixing = config.mixing
    previous_residual: float | None = None
    increase_streak = 0
    consecutive = 0
    for item in checkpoint.history:
        consecutive = consecutive + 1 if item.passed else 0
        if item.consecutive_passes != consecutive:
            raise CheckpointCorruptionError(
                "checkpoint consecutive-pass history is inconsistent."
            )
        if consecutive < config.consecutive_passes:
            if (
                previous_residual is not None
                and item.sigma_residual > previous_residual * (1.0 + 1.0e-12)
            ):
                increase_streak += 1
            else:
                increase_streak = 0
            if increase_streak >= 3:
                mixing = max(config.mixing_floor, 0.5 * mixing)
                increase_streak = 0
            previous_residual = item.sigma_residual
        if item.mixing != mixing:
            raise CheckpointCorruptionError(
                "checkpoint adaptive-mixing history is inconsistent."
            )
    expected_continuation = SCBAContinuationState(
        mixing=mixing,
        previous_sigma_residual=previous_residual,
        residual_increase_streak=increase_streak,
        consecutive_passes=consecutive,
    )
    if checkpoint.solver_state != expected_continuation:
        raise CheckpointCorruptionError(
            "checkpoint adaptive continuation state is inconsistent."
        )
    if checkpoint.status in _RESUMABLE_STATUSES:
        if count >= config.max_iterations:
            raise CheckpointCorruptionError(
                "resumable checkpoint has no remaining configured iteration."
            )
        if checkpoint.solver_state.consecutive_passes >= config.consecutive_passes:
            raise CheckpointCorruptionError(
                "resumable checkpoint already satisfies terminal convergence."
            )
    elif checkpoint.status == "completed":
        if not checkpoint.history:
            raise CheckpointCorruptionError(
                "completed checkpoint has no convergence history."
            )
        if checkpoint.history[-1].consecutive_passes < config.consecutive_passes:
            raise CheckpointCorruptionError(
                "completed checkpoint does not satisfy consecutive convergence."
            )
    elif checkpoint.status == "scientific_failure":
        if count != config.max_iterations:
            raise CheckpointCorruptionError(
                "scientific-failure checkpoint did not exhaust max_iterations."
            )
        if checkpoint.history[-1].consecutive_passes >= config.consecutive_passes:
            raise CheckpointCorruptionError(
                "scientific-failure checkpoint satisfies convergence."
            )


def restore_scba_checkpoint_result_v2(
    checkpoint: SCBACheckpointV2,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> AnharmonicResult:
    """Recompute every derived field for a validated terminal v2 outcome."""

    if not isinstance(checkpoint, SCBACheckpointV2):
        raise TypeError("checkpoint must be an SCBACheckpointV2.")
    if not checkpoint.terminal:
        raise ValueError("only terminal checkpoint outcomes can be restored.")
    _validate_checkpoint_for_solver(checkpoint, problem, config)

    from .contacts import build_contact_spectra
    from .solver import (
        _current_report,
        _finalize_result,
        _guard_dense_memory,
        _solve_green,
    )

    predicted = _guard_dense_memory(problem, config)
    left, right = build_contact_spectra(
        problem,
        config.eta_lead,
        surface_gf_method=config.surface_gf_method,
    )
    green, sigma_pp = _solve_green(
        problem,
        left,
        right,
        checkpoint.sigma_lesser,
        checkpoint.sigma_greater,
        config,
    )
    currents = _current_report(problem, left, right, sigma_pp, green)
    result = _finalize_result(
        problem,
        config,
        method="scba",
        green=green,
        sigma_pp=sigma_pp,
        left=left,
        right=right,
        currents=currents,
        history=checkpoint.history,
        converged=checkpoint.status == "completed",
        initialization="checkpoint_v2_reconstruction",
        initialization_hash=checkpoint.identity.initial_state_sha256,
        predicted_memory_bytes=predicted,
    )
    if config.mixing_method == "anderson":
        result = replace(
            result,
            metadata={
                **result.metadata,
                "anderson": checkpoint.metadata.get("anderson", {}),
                "anderson_resume_supported": False,
            },
        )
    return result


def checkpoint_initial_sigma_v2(checkpoint: SCBACheckpointV2) -> KeldyshSpectrum:
    """Return a completed checkpoint's lesser/greater pair for continuation."""

    if not isinstance(checkpoint, SCBACheckpointV2):
        raise TypeError("checkpoint must be an SCBACheckpointV2.")
    if checkpoint.status != "completed":
        raise CheckpointDependencyError(
            "only a completed SCBA checkpoint may seed a coupling continuation."
        )
    shape = checkpoint.sigma_lesser.shape
    return KeldyshSpectrum(
        retarded=np.zeros(shape, dtype=np.complex128),
        lesser=checkpoint.sigma_lesser,
        greater=checkpoint.sigma_greater,
    )


@dataclass(frozen=True, slots=True)
class _SolverResumeState:
    sigma_lesser: Array
    sigma_greater: Array
    history: tuple[SCBAIteration, ...]
    continuation: SCBAContinuationState


class _SolverCheckpointSession:
    """Context-local bridge used only by ``solver.solve_scba``."""

    def __init__(
        self,
        path: Path,
        identity: SCBACheckpointIdentityV2,
        *,
        checkpoint: SCBACheckpointV2 | None,
        initialization: str,
    ) -> None:
        self.path = path
        self.identity = identity
        self.checkpoint = checkpoint
        self.initialization = initialization
        self._algorithm_metadata: dict[str, Any] = {}
        self._attempt_started = time.perf_counter()
        self._boundary_started = self._attempt_started
        self._begun = False
        self._prior_elapsed = (
            0.0 if checkpoint is None else float(checkpoint.timings["elapsed_seconds"])
        )
        self._reconstruction = (
            0.0
            if checkpoint is None
            else float(checkpoint.timings["reconstruction_seconds"])
        )
        self._iteration_seconds = (
            [] if checkpoint is None else list(_timing_iterations(checkpoint))
        )
        self._attempts = (
            1 if checkpoint is None else int(checkpoint.timings["attempts"]) + 1
        )

    def _timings(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": self._prior_elapsed
            + (time.perf_counter() - self._attempt_started),
            "reconstruction_seconds": self._reconstruction,
            "iteration_seconds": tuple(self._iteration_seconds),
            "attempts": self._attempts,
        }

    def set_algorithm_metadata(self, metadata: dict[str, Any]) -> None:
        self._algorithm_metadata = metadata

    def _metadata(self, **updates: Any) -> dict[str, Any]:
        base = {} if self.checkpoint is None else dict(self.checkpoint.metadata)
        base.update(
            {
                "initialization": self.initialization,
                "storage_contract": "mixed_sigma_pp_lesser_greater_only",
                "derived_fields_persisted": False,
                **self._algorithm_metadata,
                **updates,
            }
        )
        return base

    def begin(
        self,
        problem: CubicSCBAProblem,
        config: SCBAConfig,
        sigma_lesser: Array,
        sigma_greater: Array,
    ) -> _SolverResumeState | None:
        if self._begun:
            raise RuntimeError("checkpoint solver session was begun more than once.")
        self._begun = True
        if self.checkpoint is not None:
            if config.mixing_method == "anderson":
                raise ValueError(
                    "Anderson scout checkpoints cannot resume: history is not persisted."
                )
            _validate_checkpoint_for_solver(self.checkpoint, problem, config)
            self._boundary_started = time.perf_counter()
            return _SolverResumeState(
                sigma_lesser=self.checkpoint.sigma_lesser,
                sigma_greater=self.checkpoint.sigma_greater,
                history=self.checkpoint.history,
                continuation=self.checkpoint.solver_state,
            )

        if config.mixing_method == "anderson":
            self._algorithm_metadata.update(
                mixing_method="anderson", anderson_resume_supported=False
            )
        state = SCBAContinuationState(config.mixing, None, 0, 0)
        record = SCBACheckpointV2(
            identity=self.identity,
            sigma_lesser=sigma_lesser,
            sigma_greater=sigma_greater,
            history=(),
            timings=self._timings(),
            status="in_progress",
            solver_state=state,
            metadata=self._metadata(),
        )
        write_scba_checkpoint_v2(self.path, record)
        self.checkpoint = record
        self._boundary_started = time.perf_counter()
        return None

    def replay_complete(self) -> None:
        now = time.perf_counter()
        self._reconstruction += now - self._boundary_started
        self._boundary_started = now

    def record_iteration(
        self,
        *,
        sigma_lesser: Array,
        sigma_greater: Array,
        history: tuple[SCBAIteration, ...],
        continuation: SCBAContinuationState,
    ) -> None:
        now = time.perf_counter()
        if len(history) != len(self._iteration_seconds) + 1:
            raise RuntimeError("solver checkpoint history advanced unexpectedly.")
        self._iteration_seconds.append(now - self._boundary_started)
        record = SCBACheckpointV2(
            identity=self.identity,
            sigma_lesser=sigma_lesser,
            sigma_greater=sigma_greater,
            history=history,
            timings=self._timings(),
            status="in_progress",
            solver_state=continuation,
            metadata=self._metadata(
                interruption_type=None,
                interruption_message=None,
            ),
        )
        write_scba_checkpoint_v2(self.path, record)
        self.checkpoint = record
        self._boundary_started = time.perf_counter()

    def record_terminal(
        self,
        *,
        sigma_lesser: Array,
        sigma_greater: Array,
        history: tuple[SCBAIteration, ...],
        continuation: SCBAContinuationState,
        status: str,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if status not in _TERMINAL_STATUSES:
            raise ValueError("terminal status is invalid.")
        now = time.perf_counter()
        if len(history) != len(self._iteration_seconds) + 1:
            raise RuntimeError("terminal checkpoint history advanced unexpectedly.")
        self._iteration_seconds.append(now - self._boundary_started)
        record = SCBACheckpointV2(
            identity=self.identity,
            sigma_lesser=sigma_lesser,
            sigma_greater=sigma_greater,
            history=history,
            timings=self._timings(),
            status=status,
            solver_state=continuation,
            metadata=self._metadata(
                error_type=error_type,
                error_message=error_message,
                interruption_type=None,
                interruption_message=None,
            ),
        )
        write_scba_checkpoint_v2(self.path, record)
        self.checkpoint = record
        self._boundary_started = time.perf_counter()

    def mark_interrupted(self, error: BaseException) -> None:
        if self.checkpoint is None or self.checkpoint.terminal:
            return
        record = replace(
            self.checkpoint,
            timings=self._timings(),
            status="infrastructure_interrupted",
            metadata=self._metadata(
                interruption_type=type(error).__name__,
                interruption_message=str(error),
            ),
        )
        write_scba_checkpoint_v2(self.path, record)
        self.checkpoint = record


_ACTIVE_SOLVER_SESSION: ContextVar[_SolverCheckpointSession | None] = ContextVar(
    "negfpy_sparse_scba_checkpoint_v2_session",
    default=None,
)


def _active_solver_checkpoint_session() -> _SolverCheckpointSession | None:
    """Private hook imported lazily by :mod:`solver`."""

    return _ACTIVE_SOLVER_SESSION.get()


def _activate_solver_session(
    session: _SolverCheckpointSession,
) -> tuple[Token[_SolverCheckpointSession | None], _SolverCheckpointSession]:
    if _ACTIVE_SOLVER_SESSION.get() is not None:
        raise RuntimeError("nested sparse SCBA checkpoint sessions are forbidden.")
    return _ACTIVE_SOLVER_SESSION.set(session), session


def _nonconvergence_error(
    checkpoint: SCBACheckpointV2,
    result: AnharmonicResult,
) -> Exception:
    from .exceptions import SCBAConvergenceError

    last = checkpoint.history[-1]
    return SCBAConvergenceError(
        "SCBA did not converge within "
        f"{len(checkpoint.history)} iterations "
        f"(sigma residual={last.sigma_residual:.3e}, "
        f"green residual={last.green_residual:.3e}, "
        f"contact imbalance={last.current_abs_error:.3e}).",
        result,
    )


def solve_sparse_scba_checkpointed(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    checkpoint_path: str | Path,
    run_id: str,
    environment_sha256: str,
    initial_sigma_pp: KeldyshSpectrum | None = None,
    source_hashes: Mapping[str, Any] | str | None = None,
    source_sha256: Mapping[str, Any] | str | None = None,
    source_tree_sha256: str | None = None,
    resume: bool = True,
) -> AnharmonicResult:
    """Run or exactly resume one sparse-material SCBA calculation.

    A terminal checkpoint is reconstructed and returned without another fixed
    point iteration.  A scientific nonconvergence is likewise terminal and is
    returned or re-raised according to the identically hashed ``config``.
    """

    if not isinstance(resume, (bool, np.bool_)):
        raise TypeError("resume must be boolean.")
    path = Path(checkpoint_path).expanduser()
    identity = make_scba_checkpoint_identity_v2(
        run_id,
        problem,
        config,
        environment_sha256=environment_sha256,
        initial_sigma_pp=initial_sigma_pp,
        source_hashes=source_hashes,
        source_sha256=source_sha256,
        source_tree_sha256=source_tree_sha256,
    )
    checkpoint: SCBACheckpointV2 | None = None
    if path.exists():
        if not resume:
            raise FileExistsError(f"checkpoint already exists and resume=False: {path}")
        checkpoint = load_scba_checkpoint_v2(
            path,
            expected_identity=identity,
        )
        _validate_checkpoint_for_solver(checkpoint, problem, config)
        if checkpoint.terminal:
            result = restore_scba_checkpoint_result_v2(
                checkpoint,
                problem,
                config,
            )
            if (
                checkpoint.status == "scientific_failure"
                and config.on_nonconvergence == "raise"
            ):
                raise _nonconvergence_error(checkpoint, result)
            return result

        if config.mixing_method == "anderson":
            raise ValueError(
                "Anderson scout checkpoints cannot resume: history is not persisted."
            )

    initialization = "zero" if initial_sigma_pp is None else "supplied_keldysh_pair"
    session = _SolverCheckpointSession(
        path,
        identity,
        checkpoint=checkpoint,
        initialization=initialization,
    )
    token, _ = _activate_solver_session(session)
    try:
        from .exceptions import SCBAConvergenceError
        from .solver import solve_scba

        try:
            return solve_scba(
                problem,
                config,
                initial_sigma_pp=initial_sigma_pp,
            )
        except SCBAConvergenceError:
            # The solver publishes the terminal scientific outcome before
            # preserving its established public exception behavior.
            raise
        except BaseException as exc:
            try:
                session.mark_interrupted(exc)
            except Exception as checkpoint_error:
                try:
                    exc.add_note(
                        "could not update the interruption marker; the last "
                        f"atomic iteration checkpoint remains resumable: {checkpoint_error}"
                    )
                except AttributeError:  # pragma: no cover - Python < 3.11 guard
                    pass
            raise
    finally:
        _ACTIVE_SOLVER_SESSION.reset(token)


# Sparse-specific operation aliases.
make_sparse_scba_checkpoint_identity = make_scba_checkpoint_identity_v2
load_sparse_scba_checkpoint = load_scba_checkpoint_v2
write_sparse_scba_checkpoint = write_scba_checkpoint_v2
restore_sparse_scba_checkpoint_result = restore_scba_checkpoint_result_v2


__all__ = [
    "CheckpointCorruptionError",
    "CheckpointDependencyError",
    "CheckpointError",
    "CheckpointMismatchError",
    "SCBACheckpointIdentityV2",
    "SCBACheckpointV2",
    "SCBAContinuationState",
    "checkpoint_initial_sigma_v2",
    "SparseSCBACheckpoint",
    "SparseSCBACheckpointIdentity",
    "load_scba_checkpoint_v2",
    "load_scba_checkpoint_v2_self_validating",
    "load_sparse_scba_checkpoint",
    "make_scba_checkpoint_identity_v2",
    "make_sparse_scba_checkpoint_identity",
    "restore_scba_checkpoint_result_v2",
    "restore_sparse_scba_checkpoint_result",
    "solve_sparse_scba_checkpointed",
    "write_scba_checkpoint_v2",
    "write_sparse_scba_checkpoint",
]
