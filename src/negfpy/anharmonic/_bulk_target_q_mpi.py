"""Private task-level MPI campaign for complete bulk target-q calculations.

The numerical boundary in this module is deliberately coarse.  One task owns
one complete target q point and calls :func:`compute_three_phonon_linewidths`
unchanged, including the full internal q-prime and branch sums.  MPI messages
therefore contain task identifiers and status only.  Numerical fields are
published as immutable HDF5 artifacts and rank zero merges them from the
shared filesystem in canonical target-q order without floating-point
reductions.

``h5py`` is optional at package-import time and imported only by an HDF5
operation.  ``mpi4py`` remains lazy through :mod:`negfpy.anharmonic._mpi_tasks`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import struct
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np

from . import _mpi_tasks as mpi_tasks
from ._mpi_tasks import (
    ArtifactValidator,
    DuplicateTaskCheckpointError,
    MPIExecutionReport,
    OrphanArtifactAdopter,
    TaskCheckpointMismatchError,
    TaskContext,
    TaskManifest,
    TaskManifestError,
    TaskResult,
    TaskSpec,
    collect_task_checkpoints,
    execute_task_manifest,
    task_checkpoint_path,
)
from .bulk_harmonic import BulkPhononSpectrum
from .bulk_three_phonon import (
    BulkThreePhononInteractions,
    BulkThreePhononLinewidths,
    BulkThreePhononProblem,
    compute_three_phonon_linewidths,
)


Array = np.ndarray
ProblemFactory = Callable[[], BulkThreePhononProblem]

_TASK_SCHEMA = "negfpy.bulk_target_q_task"
_MERGED_SCHEMA = "negfpy.bulk_target_q_merged"
_CAMPAIGN_SCHEMA = "negfpy.bulk_target_q_campaign"
_MERGE_REPORT_SCHEMA = "negfpy.bulk_target_q_merge_report"
_COMPARISON_REPORT_SCHEMA = "negfpy.bulk_target_q_comparison_report"
_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TASK_ID_PATTERN = re.compile(r"^target-q-([0-9]{8})$")

_LINEWIDTH_DATASETS = (
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
_INTERACTION_DATASETS = (
    "grid_points",
    "band_indices",
    "q1_indices",
    "q2_indices",
    "reciprocal_vectors",
    "normal_mask",
    "matrix_elements",
    "interaction_strengths",
)
_PHONON_DATASETS = (
    "qpoints",
    "dynamical_matrices",
    "frequencies",
    "eigenvectors",
    "frequency_factor_to_thz",
    "hermitianized",
)
_METADATA_DATASETS = (
    "linewidths_json",
    "interactions_json",
)
_ALL_DATASET_PATHS = tuple(
    [f"linewidths/{name}" for name in _LINEWIDTH_DATASETS]
    + [f"interactions/{name}" for name in _INTERACTION_DATASETS]
    + [f"phonons/{name}" for name in _PHONON_DATASETS]
    + [f"metadata/{name}" for name in _METADATA_DATASETS]
)
_TARGET_LINEWIDTH_DATASETS = frozenset(
    {
        "grid_points",
        "gamma_thz_hwhm",
        "gamma_normal_thz_hwhm",
        "gamma_umklapp_thz_hwhm",
        "gamma_angular_fwhm",
        "decay_phase_space",
        "difference_phase_space",
    }
)
_TARGET_INTERACTION_DATASETS = frozenset(
    {
        "grid_points",
        "q1_indices",
        "q2_indices",
        "reciprocal_vectors",
        "normal_mask",
        "matrix_elements",
        "interaction_strengths",
    }
)


class BulkTargetQCampaignError(RuntimeError):
    """Base class for private target-q campaign failures."""


class BulkTargetQArtifactError(BulkTargetQCampaignError):
    """Raised when a task or merged HDF5 artifact is invalid."""


class BulkTargetQArtifactMismatchError(
    TaskCheckpointMismatchError,
    BulkTargetQArtifactError,
):
    """Raised when a valid artifact belongs to another exact campaign."""


class BulkTargetQMergeError(BulkTargetQCampaignError):
    """Raised when canonical task artifacts cannot be merged."""


def _require_h5py() -> Any:
    try:
        module = importlib.import_module("h5py")
    except ImportError as exc:
        raise ImportError(
            "Bulk target-q task artifacts require the optional 'h5py' "
            "package. Install NEGFPy with its material-validation "
            "dependencies."
        ) from exc
    if not hasattr(module, "File"):
        raise ImportError(
            "The imported h5py module does not provide h5py.File."
        )
    return module


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256.")
    return value


def _readonly(values: Any, *, dtype: Any, name: str) -> Array:
    raw = np.asarray(values)
    if raw.dtype.kind == "c" and np.dtype(dtype).kind != "c":
        raise TypeError(f"{name} must be real-valued.")
    try:
        result = np.array(raw, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} has an invalid numerical representation.") from exc
    result.setflags(write=False)
    return result


def _selector(
    values: Sequence[int] | Array,
    *,
    upper: int,
    name: str,
) -> Array:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    if raw.size == 0:
        raise ValueError(f"{name} cannot be empty.")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must be a one-dimensional integer array.")
    result = np.array(raw, dtype=np.int64, order="C", copy=True)
    if np.any((result < 0) | (result >= upper)):
        raise IndexError(f"{name} contains an index outside [0, {upper}).")
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} cannot contain duplicate indices.")
    result.sort()
    result.setflags(write=False)
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _json_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{path} cannot contain NaN or infinity.")
        return result
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{path} cannot contain NaN or infinity.")
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist(), path=path)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings.")
            result[key] = _json_value(value[key], path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_value(item, path=path) for item in value]
        return sorted(normalized, key=_canonical_json)
    raise TypeError(
        f"{path} contains unsupported metadata type {type(value).__name__}."
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_mapping(
    values: Mapping[str, str],
    *,
    name: str,
) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    result: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{name} keys must be non-empty strings.")
        result[key] = _validate_sha256(value, name=f"{name}[{key!r}]")
    return MappingProxyType(dict(sorted(result.items())))


def _update_scientific_digest(
    digest: Any,
    value: Any,
    *,
    path: str,
) -> None:
    """Stream a deterministic, type-sensitive scientific object identity."""

    if value is None:
        digest.update(b"N;")
        return
    if isinstance(value, np.generic):
        _update_scientific_digest(digest, value.item(), path=path)
        return
    if isinstance(value, bool):
        digest.update(b"B1;" if value else b"B0;")
        return
    if isinstance(value, int):
        digest.update(b"I")
        digest.update(str(value).encode("ascii"))
        digest.update(b";")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or infinity.")
        digest.update(b"F")
        digest.update(struct.pack(">d", float(value)))
        digest.update(b";")
        return
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{path} contains NaN or infinity.")
        digest.update(b"C")
        digest.update(struct.pack(">dd", float(value.real), float(value.imag)))
        digest.update(b";")
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b";")
        return
    if isinstance(value, Path):
        _update_scientific_digest(digest, value.as_posix(), path=path)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{path} cannot contain an object-dtype array.")
        header = _canonical_json(
            {"dtype": array.dtype.str, "shape": list(array.shape)}
        ).encode("ascii")
        digest.update(b"A")
        digest.update(str(len(header)).encode("ascii"))
        digest.update(b":")
        digest.update(header)
        digest.update(b":")
        digest.update(memoryview(array).cast("B"))
        digest.update(b";")
        return
    if is_dataclass(value) and not isinstance(value, type):
        name = f"{type(value).__module__}.{type(value).__qualname__}"
        digest.update(b"D")
        _update_scientific_digest(digest, name, path=f"{path}.__type__")
        for item in fields(value):
            _update_scientific_digest(
                digest, item.name, path=f"{path}.__field__"
            )
            _update_scientific_digest(
                digest,
                getattr(value, item.name),
                path=f"{path}.{item.name}",
            )
        digest.update(b";")
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        keys = tuple(value.keys())
        if not all(isinstance(key, str) for key in keys):
            raise TypeError(f"{path} mapping keys must be strings.")
        for key in sorted(keys):
            _update_scientific_digest(digest, key, path=f"{path}.__key__")
            _update_scientific_digest(
                digest, value[key], path=f"{path}.{key}"
            )
        digest.update(b";")
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"T" if isinstance(value, tuple) else b"L")
        for index, item in enumerate(value):
            _update_scientific_digest(
                digest, item, path=f"{path}[{index}]"
            )
        digest.update(b";")
        return
    if isinstance(value, (set, frozenset)):
        item_hashes: list[bytes] = []
        for index, item in enumerate(value):
            item_digest = hashlib.sha256()
            _update_scientific_digest(
                item_digest, item, path=f"{path}[{index}]"
            )
            item_hashes.append(item_digest.digest())
        digest.update(b"E")
        for item_hash in sorted(item_hashes):
            digest.update(item_hash)
        digest.update(b";")
        return
    raise TypeError(
        f"{path} contains unsupported identity type {type(value).__name__}."
    )


def bulk_three_phonon_problem_sha256(
    problem: BulkThreePhononProblem,
) -> str:
    """Return the canonical, full scientific identity of a bulk problem."""

    if not isinstance(problem, BulkThreePhononProblem):
        raise TypeError("problem must be a BulkThreePhononProblem.")
    digest = hashlib.sha256()
    digest.update(b"negfpy.bulk_three_phonon_problem.v1\0")
    _update_scientific_digest(digest, problem, path="problem")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class BulkTargetQControls:
    """Immutable scientific controls shared by every target-q task."""

    temperatures: Sequence[float] | Array
    grid_points: Sequence[int] | Array
    band_indices: Sequence[int] | Array
    n_qpoints: int
    n_bands: int
    problem_sha256: str
    input_hashes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_qpoints = _positive_integer(self.n_qpoints, name="n_qpoints")
        n_bands = _positive_integer(self.n_bands, name="n_bands")
        temperatures = _readonly(
            self.temperatures, dtype=np.float64, name="temperatures"
        )
        if (
            temperatures.ndim != 1
            or temperatures.size == 0
            or not np.all(np.isfinite(temperatures))
            or np.any(temperatures < 0.0)
        ):
            raise ValueError(
                "temperatures must be a nonempty finite nonnegative 1D array."
            )
        grid_points = _selector(
            self.grid_points,
            upper=n_qpoints,
            name="grid_points",
        )
        band_indices = _selector(
            self.band_indices,
            upper=n_bands,
            name="band_indices",
        )
        problem_sha256 = _validate_sha256(
            self.problem_sha256, name="problem_sha256"
        )
        input_hashes = _hash_mapping(
            self.input_hashes, name="input_hashes"
        )
        reserved = {"problem_sha256", "controls_sha256", "target_q_sha256"}
        overlap = reserved.intersection(input_hashes)
        if overlap:
            raise ValueError(
                "input_hashes uses reserved names: "
                + ", ".join(sorted(overlap))
            )
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "grid_points", grid_points)
        object.__setattr__(self, "band_indices", band_indices)
        object.__setattr__(self, "n_qpoints", n_qpoints)
        object.__setattr__(self, "n_bands", n_bands)
        object.__setattr__(self, "problem_sha256", problem_sha256)
        object.__setattr__(self, "input_hashes", input_hashes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperatures": self.temperatures.tolist(),
            "grid_points": self.grid_points.tolist(),
            "band_indices": self.band_indices.tolist(),
            "n_qpoints": self.n_qpoints,
            "n_bands": self.n_bands,
            "problem_sha256": self.problem_sha256,
            "input_hashes": dict(self.input_hashes),
        }

    @property
    def controls_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.as_dict()).encode("ascii"))


def validate_problem_against_controls(
    problem: BulkThreePhononProblem,
    controls: BulkTargetQControls,
) -> str:
    """Validate dimensions and full problem identity before task execution."""

    if not isinstance(problem, BulkThreePhononProblem):
        raise TypeError("problem must be a BulkThreePhononProblem.")
    if not isinstance(controls, BulkTargetQControls):
        raise TypeError("controls must be BulkTargetQControls.")
    n_qpoints = problem.mesh.n_qpoints
    n_bands = 3 * problem.fc2_mapping.n_primitive_atoms
    if n_qpoints != controls.n_qpoints:
        raise ValueError(
            "loaded problem q-point count conflicts with campaign controls."
        )
    if n_bands != controls.n_bands:
        raise ValueError(
            "loaded problem band count conflicts with campaign controls."
        )
    digest = bulk_three_phonon_problem_sha256(problem)
    if digest != controls.problem_sha256:
        raise ValueError(
            "loaded problem SHA256 conflicts with the immutable campaign."
        )
    return digest


def _target_task_id(qpoint: int) -> str:
    return f"target-q-{qpoint:08d}"


def _target_q_sha256(qpoint: int) -> str:
    return _sha256_bytes(
        _canonical_json({"target_q": int(qpoint)}).encode("ascii")
    )


@dataclass(frozen=True, slots=True, eq=False)
class BulkTargetQCampaign:
    """Validated one-task-per-target-q manifest and its controls."""

    controls: BulkTargetQControls
    manifest: TaskManifest

    def __post_init__(self) -> None:
        if not isinstance(self.controls, BulkTargetQControls):
            raise TypeError("controls must be BulkTargetQControls.")
        if not isinstance(self.manifest, TaskManifest):
            raise TypeError("manifest must be TaskManifest.")
        expected_identity = {
            "problem_sha256": self.controls.problem_sha256,
            "controls_sha256": self.controls.controls_sha256,
            **dict(self.controls.input_hashes),
        }
        if dict(self.manifest.identity_hashes) != dict(
            sorted(expected_identity.items())
        ):
            raise ValueError(
                "manifest identity hashes conflict with campaign controls."
            )
        if len(self.manifest.tasks) != self.controls.grid_points.size:
            raise ValueError("manifest must contain exactly one task per target q.")
        for ordinal, (task, qpoint) in enumerate(
            zip(
                self.manifest.tasks,
                self.controls.grid_points,
                strict=True,
            )
        ):
            q = int(qpoint)
            expected_task_identity = {
                **expected_identity,
                "target_q_sha256": _target_q_sha256(q),
            }
            if task.task_id != _target_task_id(q):
                raise ValueError("task identifiers are not in canonical q order.")
            if dict(task.identity_hashes) != dict(
                sorted(expected_task_identity.items())
            ):
                raise ValueError(
                    f"task {task.task_id!r} has inconsistent identity hashes."
                )
            payload = task.payload
            if (
                payload.get("kind") != "bulk_target_q"
                or int(payload.get("target_q", -1)) != q
                or int(payload.get("target_position", -1)) != ordinal
                or tuple(payload.get("band_indices", ()))
                != tuple(int(value) for value in self.controls.band_indices)
                or tuple(payload.get("temperatures", ()))
                != tuple(float(value) for value in self.controls.temperatures)
            ):
                raise ValueError(
                    f"task {task.task_id!r} payload conflicts with controls."
                )
        metadata = self.manifest.metadata
        if (
            metadata.get("schema") != _CAMPAIGN_SCHEMA
            or int(metadata.get("schema_version", -1)) != _SCHEMA_VERSION
            or metadata.get("controls_sha256")
            != self.controls.controls_sha256
        ):
            raise ValueError("manifest campaign metadata is incomplete.")

    @property
    def campaign_manifest_sha256(self) -> str:
        return self.manifest.manifest_sha256

    @property
    def canonical_task_ids(self) -> tuple[str, ...]:
        return tuple(task.task_id for task in self.manifest.tasks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _CAMPAIGN_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "controls": self.controls.as_dict(),
            "controls_sha256": self.controls.controls_sha256,
            "task_ids": list(self.canonical_task_ids),
            "task_identity_sha256": [
                self.manifest.task_identity_sha256(task)
                for task in self.manifest.tasks
            ],
        }


def build_bulk_target_q_campaign(
    controls: BulkTargetQControls,
    *,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    metadata: Mapping[str, Any] | None = None,
) -> BulkTargetQCampaign:
    """Build the canonical one-task-per-target-q task manifest."""

    if not isinstance(controls, BulkTargetQControls):
        raise TypeError("controls must be BulkTargetQControls.")
    identity_hashes = {
        "problem_sha256": controls.problem_sha256,
        "controls_sha256": controls.controls_sha256,
        **dict(controls.input_hashes),
    }
    tasks = []
    for ordinal, qpoint in enumerate(controls.grid_points):
        q = int(qpoint)
        tasks.append(
            TaskSpec(
                task_id=_target_task_id(q),
                identity_hashes={
                    **identity_hashes,
                    "target_q_sha256": _target_q_sha256(q),
                },
                payload={
                    "kind": "bulk_target_q",
                    "target_q": q,
                    "target_position": ordinal,
                    "temperatures": controls.temperatures.tolist(),
                    "band_indices": controls.band_indices.tolist(),
                },
            )
        )
    manifest_metadata = {
        "schema": _CAMPAIGN_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "controls_sha256": controls.controls_sha256,
        "canonical_target_q_order": controls.grid_points.tolist(),
        "complete_internal_q_sum_per_task": True,
        "floating_point_mpi_reductions": False,
        "user_metadata": {} if metadata is None else _json_value(metadata),
    }
    manifest = TaskManifest(
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        tasks=tuple(tasks),
        identity_hashes=identity_hashes,
        metadata=manifest_metadata,
    )
    return BulkTargetQCampaign(controls=controls, manifest=manifest)


def _array_sha256(values: Any) -> str:
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise TypeError("HDF5 artifact arrays cannot use object dtype.")
    header = _canonical_json(
        {"dtype": array.dtype.str, "shape": list(array.shape)}
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    byte_view = array.reshape(1) if array.ndim == 0 else array
    digest.update(memoryview(byte_view).cast("B"))
    return digest.hexdigest()


def _dataset_sha256(dataset: Any) -> str:
    dtype = np.dtype(dataset.dtype)
    shape = tuple(int(value) for value in dataset.shape)
    header = _canonical_json(
        {"dtype": dtype.str, "shape": list(shape)}
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    if shape:
        for first in range(shape[0]):
            values = np.ascontiguousarray(dataset[first])
            digest.update(memoryview(values).cast("B"))
    else:
        values = np.ascontiguousarray(dataset[()])
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _metadata_bytes(values: Mapping[str, Any]) -> bytes:
    return _canonical_json(values).encode("ascii")


def _result_arrays(
    result: BulkThreePhononLinewidths,
) -> dict[str, Array]:
    interactions = result.interactions
    phonons = interactions.phonons
    return {
        "linewidths/temperatures": np.asarray(
            result.temperatures, dtype=np.float64
        ),
        "linewidths/grid_points": np.asarray(
            result.grid_points, dtype=np.int64
        ),
        "linewidths/band_indices": np.asarray(
            result.band_indices, dtype=np.int64
        ),
        "linewidths/gamma_thz_hwhm": np.asarray(
            result.gamma_thz_hwhm, dtype=np.float64
        ),
        "linewidths/gamma_normal_thz_hwhm": np.asarray(
            result.gamma_normal_thz_hwhm, dtype=np.float64
        ),
        "linewidths/gamma_umklapp_thz_hwhm": np.asarray(
            result.gamma_umklapp_thz_hwhm, dtype=np.float64
        ),
        "linewidths/gamma_angular_fwhm": np.asarray(
            result.gamma_angular_fwhm, dtype=np.float64
        ),
        "linewidths/decay_phase_space": np.asarray(
            result.decay_phase_space, dtype=np.float64
        ),
        "linewidths/difference_phase_space": np.asarray(
            result.difference_phase_space, dtype=np.float64
        ),
        "interactions/grid_points": np.asarray(
            interactions.grid_points, dtype=np.int64
        ),
        "interactions/band_indices": np.asarray(
            interactions.band_indices, dtype=np.int64
        ),
        "interactions/q1_indices": np.asarray(
            interactions.q1_indices, dtype=np.int64
        ),
        "interactions/q2_indices": np.asarray(
            interactions.q2_indices, dtype=np.int64
        ),
        "interactions/reciprocal_vectors": np.asarray(
            interactions.reciprocal_vectors, dtype=np.int64
        ),
        "interactions/normal_mask": np.asarray(
            interactions.normal_mask, dtype=np.bool_
        ),
        "interactions/matrix_elements": np.asarray(
            interactions.matrix_elements, dtype=np.complex128
        ),
        "interactions/interaction_strengths": np.asarray(
            interactions.interaction_strengths, dtype=np.float64
        ),
        "phonons/qpoints": np.asarray(phonons.qpoints, dtype=np.float64),
        "phonons/dynamical_matrices": np.asarray(
            phonons.dynamical_matrices, dtype=np.complex128
        ),
        "phonons/frequencies": np.asarray(
            phonons.frequencies, dtype=np.float64
        ),
        "phonons/eigenvectors": np.asarray(
            phonons.eigenvectors, dtype=np.complex128
        ),
        "phonons/frequency_factor_to_thz": np.asarray(
            phonons.frequency_factor_to_thz, dtype=np.float64
        ),
        "phonons/hermitianized": np.asarray(
            phonons.hermitianized, dtype=np.bool_
        ),
        "metadata/linewidths_json": np.frombuffer(
            _metadata_bytes(result.metadata), dtype=np.uint8
        ),
        "metadata/interactions_json": np.frombuffer(
            _metadata_bytes(interactions.metadata), dtype=np.uint8
        ),
    }


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _artifact_identity(
    campaign: BulkTargetQCampaign,
    task: TaskSpec,
) -> dict[str, Any]:
    qpoint = int(task.payload["target_q"])
    return {
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "manifest_run_identity_sha256": campaign.manifest.run_identity_sha256,
        "task_identity_sha256": campaign.manifest.task_identity_sha256(task),
        "task_id": task.task_id,
        "target_q": qpoint,
        "problem_sha256": campaign.controls.problem_sha256,
        "controls_sha256": campaign.controls.controls_sha256,
    }


def _expected_task_root_attributes() -> frozenset[str]:
    return frozenset(
        {
            "schema_name",
            "schema_version",
            "campaign_manifest_sha256",
            "manifest_run_identity_sha256",
            "task_identity_sha256",
            "task_id",
            "target_q",
            "problem_sha256",
            "controls_sha256",
            "dataset_sha256_json",
        }
    )


def _expected_merged_root_attributes() -> frozenset[str]:
    return frozenset(
        {
            "schema_name",
            "schema_version",
            "campaign_manifest_sha256",
            "manifest_run_identity_sha256",
            "problem_sha256",
            "controls_sha256",
            "canonical_task_ids_json",
            "floating_point_mpi_reductions",
            "dataset_sha256_json",
            "task_artifact_sha256_json",
        }
    )


def _expected_dataset_dtypes() -> Mapping[str, np.dtype[Any]]:
    integer = np.dtype(np.int64)
    real = np.dtype(np.float64)
    complex_dtype = np.dtype(np.complex128)
    boolean = np.dtype(np.bool_)
    byte = np.dtype(np.uint8)
    return MappingProxyType(
        {
            "linewidths/temperatures": real,
            "linewidths/grid_points": integer,
            "linewidths/band_indices": integer,
            "linewidths/gamma_thz_hwhm": real,
            "linewidths/gamma_normal_thz_hwhm": real,
            "linewidths/gamma_umklapp_thz_hwhm": real,
            "linewidths/gamma_angular_fwhm": real,
            "linewidths/decay_phase_space": real,
            "linewidths/difference_phase_space": real,
            "interactions/grid_points": integer,
            "interactions/band_indices": integer,
            "interactions/q1_indices": integer,
            "interactions/q2_indices": integer,
            "interactions/reciprocal_vectors": integer,
            "interactions/normal_mask": boolean,
            "interactions/matrix_elements": complex_dtype,
            "interactions/interaction_strengths": real,
            "phonons/qpoints": real,
            "phonons/dynamical_matrices": complex_dtype,
            "phonons/frequencies": real,
            "phonons/eigenvectors": complex_dtype,
            "phonons/frequency_factor_to_thz": real,
            "phonons/hermitianized": boolean,
            "metadata/linewidths_json": byte,
            "metadata/interactions_json": byte,
        }
    )


def _publish_task_artifact(
    campaign: BulkTargetQCampaign,
    task: TaskSpec,
    context: TaskContext,
    result: BulkThreePhononLinewidths,
) -> Path:
    h5py = _require_h5py()
    arrays = _result_arrays(result)
    if tuple(sorted(arrays)) != tuple(sorted(_ALL_DATASET_PATHS)):
        raise AssertionError("internal target-q artifact schema is incomplete.")
    identity = _artifact_identity(campaign, task)
    hashes = {name: _array_sha256(values) for name, values in arrays.items()}
    target = context.artifact_path(".h5")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs["schema_name"] = _TASK_SCHEMA
            handle.attrs["schema_version"] = _SCHEMA_VERSION
            for name, value in identity.items():
                handle.attrs[name] = value
            handle.attrs["dataset_sha256_json"] = _canonical_json(hashes)
            for group_name in ("linewidths", "interactions", "phonons", "metadata"):
                handle.create_group(group_name, track_order=False)
            for name in _ALL_DATASET_PATHS:
                dataset = handle.create_dataset(name, data=arrays[name])
                dataset.attrs["sha256"] = hashes[name]
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _task_artifact_evidence(target, task, campaign)
            expected = {
                "schema_name": _TASK_SCHEMA,
                "schema_version": _SCHEMA_VERSION,
                **identity,
                "dataset_sha256": hashes,
            }
            if existing != expected:
                raise DuplicateTaskCheckpointError(
                    "an incompatible target-q artifact already exists: "
                    f"{target}"
                )
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def _text_attribute(handle: Any, name: str) -> str:
    value = handle.attrs.get(name)
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


def _dataset_paths(handle: Any) -> tuple[str, ...]:
    paths: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            paths.append(name)

    handle.visititems(visitor)
    return tuple(sorted(paths))


def _validate_group_schema(handle: Any) -> None:
    expected = {
        "linewidths": frozenset(_LINEWIDTH_DATASETS),
        "interactions": frozenset(_INTERACTION_DATASETS),
        "phonons": frozenset(_PHONON_DATASETS),
        "metadata": frozenset(_METADATA_DATASETS),
    }
    if frozenset(str(name) for name in handle.keys()) != frozenset(expected):
        raise ValueError("HDF5 root groups are not exact.")
    for group_name, child_names in expected.items():
        group = handle[group_name]
        if frozenset(str(name) for name in group.keys()) != child_names:
            raise ValueError(f"HDF5 group {group_name!r} is not exact.")
        if tuple(group.attrs):
            raise ValueError(
                f"HDF5 group {group_name!r} cannot have attributes."
            )


def _validate_json_dataset(dataset: Any, *, name: str) -> Mapping[str, Any]:
    try:
        payload = bytes(np.asarray(dataset, dtype=np.uint8).tobytes())
        document = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not canonical ASCII JSON.") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{name} metadata must be a JSON object.")
    if payload != _canonical_json(document).encode("ascii"):
        raise ValueError(f"{name} metadata is not in canonical JSON form.")
    return document


def _interaction_result_bytes(
    *,
    n_targets: int,
    controls: BulkTargetQControls,
) -> int:
    fields = (
        n_targets
        * controls.n_qpoints
        * int(controls.band_indices.size)
        * controls.n_bands
        * controls.n_bands
    )
    return int(
        fields * 24
        + n_targets * controls.n_qpoints * (8 + 8 + 3 * 8 + 1)
    )


def _validate_metadata_coherence(
    handle: Any,
    *,
    controls: BulkTargetQControls,
    campaign_manifest_sha256: str | None,
) -> None:
    linewidths = _validate_json_dataset(
        handle["metadata/linewidths_json"], name="linewidths"
    )
    interactions = _validate_json_dataset(
        handle["metadata/interactions_json"], name="interactions"
    )
    if linewidths.get("interaction_metadata") != interactions:
        raise ValueError(
            "linewidth metadata does not embed the stored interaction metadata."
        )
    n_targets = int(handle["linewidths/grid_points"].shape[0])
    if interactions.get("estimated_result_bytes") != (
        _interaction_result_bytes(
            n_targets=n_targets,
            controls=controls,
        )
    ):
        raise ValueError("interaction result-byte estimate is inconsistent.")
    execution = interactions.get("target_q_execution")
    linewidth_execution = linewidths.get("target_q_execution")
    if campaign_manifest_sha256 is None:
        if execution is not None or linewidth_execution is not None:
            raise ValueError(
                "one-target worker metadata contains merge orchestration."
            )
        return
    expected_execution = {
        "schema": _CAMPAIGN_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "method": "complete_target_q_tasks_canonical_filesystem_merge",
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "n_tasks": n_targets,
        "canonical_target_q_order": np.asarray(
            handle["linewidths/grid_points"], dtype=np.int64
        ).tolist(),
        "floating_point_mpi_reductions": False,
    }
    if execution != expected_execution or linewidth_execution != (
        expected_execution
    ):
        raise ValueError("merged target-q execution metadata is inconsistent.")


def _expected_task_shapes(
    controls: BulkTargetQControls,
) -> dict[str, tuple[int, ...]]:
    n_t = int(controls.temperatures.size)
    n_q = controls.n_qpoints
    n_s = int(controls.band_indices.size)
    n_b = controls.n_bands
    return {
        "linewidths/temperatures": (n_t,),
        "linewidths/grid_points": (1,),
        "linewidths/band_indices": (n_s,),
        "linewidths/gamma_thz_hwhm": (n_t, 1, n_s),
        "linewidths/gamma_normal_thz_hwhm": (n_t, 1, n_s),
        "linewidths/gamma_umklapp_thz_hwhm": (n_t, 1, n_s),
        "linewidths/gamma_angular_fwhm": (n_t, 1, n_s),
        "linewidths/decay_phase_space": (1, n_s),
        "linewidths/difference_phase_space": (1, n_s),
        "interactions/grid_points": (1,),
        "interactions/band_indices": (n_s,),
        "interactions/q1_indices": (1, n_q),
        "interactions/q2_indices": (1, n_q),
        "interactions/reciprocal_vectors": (1, n_q, 3),
        "interactions/normal_mask": (1, n_q),
        "interactions/matrix_elements": (1, n_q, n_s, n_b, n_b),
        "interactions/interaction_strengths": (1, n_q, n_s, n_b, n_b),
        "phonons/qpoints": (n_q, 3),
        "phonons/dynamical_matrices": (n_q, n_b, n_b),
        "phonons/frequencies": (n_q, n_b),
        "phonons/eigenvectors": (n_q, n_b, n_b),
        "phonons/frequency_factor_to_thz": (),
        "phonons/hermitianized": (),
    }


def _validate_common_numerics(
    handle: Any,
    *,
    controls: BulkTargetQControls,
    expected_grid_points: Array,
) -> None:
    if not np.array_equal(
        np.asarray(handle["linewidths/temperatures"]),
        controls.temperatures,
    ):
        raise ValueError("stored temperatures conflict with controls.")
    if not np.array_equal(
        np.asarray(handle["linewidths/band_indices"]),
        controls.band_indices,
    ):
        raise ValueError("stored linewidth bands conflict with controls.")
    if not np.array_equal(
        np.asarray(handle["interactions/band_indices"]),
        controls.band_indices,
    ):
        raise ValueError("stored interaction bands conflict with controls.")
    for prefix in ("linewidths", "interactions"):
        if not np.array_equal(
            np.asarray(handle[f"{prefix}/grid_points"]),
            expected_grid_points,
        ):
            raise ValueError(f"stored {prefix} target q points are inconsistent.")

    q1 = np.asarray(handle["interactions/q1_indices"], dtype=np.int64)
    q2 = np.asarray(handle["interactions/q2_indices"], dtype=np.int64)
    if (
        np.any((q1 < 0) | (q1 >= controls.n_qpoints))
        or np.any((q2 < 0) | (q2 >= controls.n_qpoints))
    ):
        raise ValueError("stored interaction routing is outside the full mesh.")
    canonical_q1 = np.arange(controls.n_qpoints, dtype=np.int64)
    if not all(np.array_equal(row, canonical_q1) for row in q1):
        raise ValueError("q1 routes are not in canonical full-mesh order.")
    reciprocal = np.asarray(handle["interactions/reciprocal_vectors"])
    normal = np.asarray(handle["interactions/normal_mask"], dtype=np.bool_)
    if not np.array_equal(normal, np.all(reciprocal == 0, axis=-1)):
        raise ValueError("normal_mask conflicts with reciprocal vectors.")

    gamma = np.asarray(handle["linewidths/gamma_thz_hwhm"])
    gamma_normal = np.asarray(
        handle["linewidths/gamma_normal_thz_hwhm"]
    )
    gamma_umklapp = np.asarray(
        handle["linewidths/gamma_umklapp_thz_hwhm"]
    )
    if not np.array_equal(gamma, gamma_normal + gamma_umklapp):
        raise ValueError("total gamma is not the stored N/U sum.")
    angular = np.asarray(handle["linewidths/gamma_angular_fwhm"])
    if not np.array_equal(angular, 4.0 * np.pi * gamma):
        raise ValueError("angular FWHM does not equal 4*pi*cyclic HWHM.")

    finite_paths = tuple(
        path
        for path in _ALL_DATASET_PATHS
        if not path.startswith("metadata/")
    )
    for path in finite_paths:
        dataset = handle[path]
        if np.dtype(dataset.dtype).kind not in "fc":
            continue
        if dataset.ndim == 0:
            slices = (np.asarray(dataset[()]),)
        else:
            slices = (
                np.asarray(dataset[first])
                for first in range(dataset.shape[0])
            )
        if any(not np.all(np.isfinite(values)) for values in slices):
            raise ValueError(f"dataset {path!r} contains nonfinite values.")
    strengths = handle["interactions/interaction_strengths"]
    if any(
        np.any(np.asarray(strengths[first]) < 0.0)
        for first in range(strengths.shape[0])
    ):
        raise ValueError("interaction strengths cannot be negative.")


def _task_artifact_evidence(
    path: Path,
    task: TaskSpec,
    campaign: BulkTargetQCampaign,
) -> dict[str, Any]:
    h5py = _require_h5py()
    identity = _artifact_identity(campaign, task)
    expected_shapes = _expected_task_shapes(campaign.controls)
    qpoint = int(task.payload["target_q"])
    try:
        with h5py.File(path, "r") as handle:
            if frozenset(str(name) for name in handle.attrs) != (
                _expected_task_root_attributes()
            ):
                raise ValueError(
                    "target-q artifact root attributes are not exact."
                )
            if _text_attribute(handle, "schema_name") != _TASK_SCHEMA:
                raise ValueError("wrong target-q task artifact schema")
            if int(handle.attrs.get("schema_version", -1)) != _SCHEMA_VERSION:
                raise ValueError("wrong target-q task artifact version")
            for name, expected in identity.items():
                actual = handle.attrs.get(name)
                if isinstance(expected, str):
                    actual = _text_attribute(handle, name)
                elif isinstance(actual, np.generic):
                    actual = actual.item()
                if actual != expected:
                    raise BulkTargetQArtifactMismatchError(
                        f"target-q artifact {name} identity does not match."
                    )
            _validate_group_schema(handle)
            if _dataset_paths(handle) != tuple(sorted(_ALL_DATASET_PATHS)):
                raise ValueError("target-q artifact dataset schema is not exact.")
            dataset_hashes: dict[str, str] = {}
            expected_dtypes = _expected_dataset_dtypes()
            for name in _ALL_DATASET_PATHS:
                dataset = handle[name]
                if name in expected_shapes and tuple(dataset.shape) != (
                    expected_shapes[name]
                ):
                    raise ValueError(f"dataset {name!r} has the wrong shape.")
                if np.dtype(dataset.dtype) != expected_dtypes[name]:
                    raise ValueError(f"dataset {name!r} has the wrong dtype.")
                if frozenset(str(key) for key in dataset.attrs) != {
                    "sha256"
                }:
                    raise ValueError(
                        f"dataset {name!r} attributes are not exact."
                    )
                digest = _dataset_sha256(dataset)
                if _text_attribute(dataset, "sha256") != digest:
                    raise ValueError(f"dataset {name!r} failed SHA256 validation.")
                dataset_hashes[name] = digest
            recorded = json.loads(
                _text_attribute(handle, "dataset_sha256_json")
            )
            if recorded != dataset_hashes:
                raise ValueError("dataset SHA256 manifest does not match.")
            _validate_metadata_coherence(
                handle,
                controls=campaign.controls,
                campaign_manifest_sha256=None,
            )
            _validate_common_numerics(
                handle,
                controls=campaign.controls,
                expected_grid_points=np.array([qpoint], dtype=np.int64),
            )
    except BulkTargetQArtifactMismatchError:
        raise
    except Exception as exc:
        raise BulkTargetQArtifactError(
            f"invalid target-q HDF5 artifact: {path}"
        ) from exc
    return {
        "schema_name": _TASK_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        **identity,
        "dataset_sha256": dataset_hashes,
    }


def make_bulk_target_q_artifact_validator(
    campaign: BulkTargetQCampaign,
) -> ArtifactValidator:
    """Return the strict validator bound to one immutable campaign."""

    if not isinstance(campaign, BulkTargetQCampaign):
        raise TypeError("campaign must be BulkTargetQCampaign.")

    def validate(path: Path, task: TaskSpec) -> Mapping[str, Any]:
        return _task_artifact_evidence(path, task, campaign)

    return validate


def make_bulk_target_q_orphan_adopter(
    campaign: BulkTargetQCampaign,
) -> OrphanArtifactAdopter:
    """Return an adopter for a validated artifact lacking its sidecar."""

    if not isinstance(campaign, BulkTargetQCampaign):
        raise TypeError("campaign must be BulkTargetQCampaign.")

    def adopt(
        path: Path,
        task: TaskSpec,
        evidence: Mapping[str, Any],
    ) -> TaskResult:
        expected = _artifact_identity(campaign, task)
        if any(evidence.get(name) != value for name, value in expected.items()):
            raise BulkTargetQArtifactMismatchError(
                "orphan target-q artifact identity does not match."
            )
        return TaskResult.completed(
            path,
            metadata={
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
                "task_identity_sha256": (
                    campaign.manifest.task_identity_sha256(task)
                ),
                "target_q": int(task.payload["target_q"]),
            },
        )

    return adopt


def make_bulk_target_q_worker(
    campaign: BulkTargetQCampaign,
    problem_factory: ProblemFactory,
) -> Callable[[TaskSpec, TaskContext], TaskResult]:
    """Create a whole-target worker with one lazy problem load per rank."""

    if not isinstance(campaign, BulkTargetQCampaign):
        raise TypeError("campaign must be BulkTargetQCampaign.")
    if not callable(problem_factory):
        raise TypeError("problem_factory must be callable.")
    cached: list[BulkThreePhononProblem] = []

    def problem() -> BulkThreePhononProblem:
        if not cached:
            loaded = problem_factory()
            validate_problem_against_controls(loaded, campaign.controls)
            cached.append(loaded)
        return cached[0]

    def worker(task: TaskSpec, context: TaskContext) -> TaskResult:
        if task not in campaign.manifest.tasks:
            raise TaskCheckpointMismatchError(
                "worker task does not belong to this target-q campaign."
            )
        qpoint = int(task.payload["target_q"])
        result = compute_three_phonon_linewidths(
            problem(),
            temperatures=campaign.controls.temperatures,
            grid_points=[qpoint],
            band_indices=campaign.controls.band_indices,
        )
        if (
            not np.array_equal(result.grid_points, np.array([qpoint]))
            or not np.array_equal(
                result.band_indices, campaign.controls.band_indices
            )
            or not np.array_equal(
                result.temperatures, campaign.controls.temperatures
            )
        ):
            raise BulkTargetQArtifactError(
                "bulk kernel returned fields inconsistent with its task."
            )
        artifact = _publish_task_artifact(
            campaign, task, context, result
        )
        return TaskResult.completed(
            artifact,
            metadata={
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
                "task_identity_sha256": (
                    campaign.manifest.task_identity_sha256(task)
                ),
                "target_q": qpoint,
                "complete_internal_q_sum": True,
            },
        )

    return worker


def execute_bulk_target_q_campaign(
    campaign: BulkTargetQCampaign,
    problem_factory: ProblemFactory,
    run_directory: str | Path,
    *,
    comm: Any | None = None,
    require_mpi: bool = False,
    thread_limit: int = 1,
    execution_id: str | None = None,
) -> MPIExecutionReport | None:
    """Execute or exactly resume a complete target-q campaign."""

    worker = make_bulk_target_q_worker(campaign, problem_factory)
    return execute_task_manifest(
        campaign.manifest,
        worker,
        run_directory,
        comm=comm,
        require_mpi=require_mpi,
        thread_limit=thread_limit,
        artifact_validator=make_bulk_target_q_artifact_validator(campaign),
        execution_id=execution_id,
        orphan_artifact_suffix=".h5",
        orphan_artifact_adopter=make_bulk_target_q_orphan_adopter(campaign),
    )


def _task_artifact_paths(
    campaign: BulkTargetQCampaign,
    run_directory: Path,
) -> tuple[tuple[Path, str, str], ...]:
    artifact_root = run_directory / "task-artifacts"
    checkpoints = collect_task_checkpoints(
        campaign.manifest,
        run_directory / "task-outcomes",
        artifact_root,
        artifact_validator=make_bulk_target_q_artifact_validator(campaign),
        require_complete=True,
    )
    by_task_id = {checkpoint.task_id: checkpoint for checkpoint in checkpoints}
    if len(by_task_id) != len(checkpoints):
        raise BulkTargetQMergeError(
            "multiple checkpoints claim the same target-q task."
        )
    result: list[tuple[Path, str, str]] = []
    for task in campaign.manifest.tasks:
        checkpoint = by_task_id.get(task.task_id)
        if checkpoint is None:
            raise BulkTargetQMergeError(
                f"target-q task {task.task_id!r} has no checkpoint."
            )
        if checkpoint.status != "completed":
            raise BulkTargetQMergeError(
                f"target-q task {task.task_id!r} did not complete."
            )
        if (
            checkpoint.artifact_relative_path is None
            or checkpoint.artifact_sha256 is None
        ):
            raise BulkTargetQMergeError(
                f"target-q task {task.task_id!r} has no artifact."
            )
        path = (artifact_root / checkpoint.artifact_relative_path).resolve()
        checkpoint_path = task_checkpoint_path(
            campaign.manifest,
            task,
            run_directory / "task-outcomes",
        )
        result.append(
            (path, checkpoint.artifact_sha256, _file_sha256(checkpoint_path))
        )
    return tuple(result)


def _merged_shapes(
    controls: BulkTargetQControls,
) -> dict[str, tuple[int, ...]]:
    shapes = _expected_task_shapes(controls)
    n_targets = int(controls.grid_points.size)
    for name in _TARGET_LINEWIDTH_DATASETS:
        path = f"linewidths/{name}"
        shape = shapes[path]
        if name.startswith("gamma_"):
            shapes[path] = (shape[0], n_targets, shape[2])
        else:
            shapes[path] = (n_targets, *shape[1:])
    for name in _TARGET_INTERACTION_DATASETS:
        path = f"interactions/{name}"
        shape = shapes[path]
        shapes[path] = (n_targets, *shape[1:])
    return shapes


def _copy_task_into_merged(
    source: Any,
    target: Any,
    *,
    target_position: int,
    first_task: bool,
) -> None:
    for name in _LINEWIDTH_DATASETS:
        source_dataset = source[f"linewidths/{name}"]
        target_dataset = target[f"linewidths/{name}"]
        if name not in _TARGET_LINEWIDTH_DATASETS:
            if first_task:
                target_dataset[...] = source_dataset[...]
            continue
        if name.startswith("gamma_"):
            target_dataset[:, target_position, :] = source_dataset[:, 0, :]
        else:
            target_dataset[target_position, ...] = source_dataset[0, ...]
    for name in _INTERACTION_DATASETS:
        source_dataset = source[f"interactions/{name}"]
        target_dataset = target[f"interactions/{name}"]
        if name not in _TARGET_INTERACTION_DATASETS:
            if first_task:
                target_dataset[...] = source_dataset[...]
            continue
        target_dataset[target_position, ...] = source_dataset[0, ...]
    if first_task:
        for name in _PHONON_DATASETS:
            target[f"phonons/{name}"][...] = source[f"phonons/{name}"][...]


def _merged_metadata_arrays(
    source: Any,
    campaign: BulkTargetQCampaign,
) -> Mapping[str, Array]:
    """Create internally coherent metadata for the complete merged result."""

    interactions = dict(
        _validate_json_dataset(
            source["metadata/interactions_json"],
            name="interactions",
        )
    )
    linewidths = dict(
        _validate_json_dataset(
            source["metadata/linewidths_json"],
            name="linewidths",
        )
    )
    controls = campaign.controls
    n_targets = int(controls.grid_points.size)
    interactions["estimated_result_bytes"] = _interaction_result_bytes(
        n_targets=n_targets,
        controls=controls,
    )
    execution_metadata = {
        "schema": _CAMPAIGN_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "method": "complete_target_q_tasks_canonical_filesystem_merge",
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "n_tasks": n_targets,
        "canonical_target_q_order": controls.grid_points.tolist(),
        "floating_point_mpi_reductions": False,
    }
    interactions["target_q_execution"] = execution_metadata
    # BulkThreePhononLinewidths embeds the interaction metadata.  Replacing it
    # here avoids an internally inconsistent one-target memory estimate.
    linewidths["interaction_metadata"] = interactions
    linewidths["target_q_execution"] = execution_metadata
    return MappingProxyType(
        {
            "metadata/linewidths_json": np.frombuffer(
                _metadata_bytes(linewidths), dtype=np.uint8
            ),
            "metadata/interactions_json": np.frombuffer(
                _metadata_bytes(interactions), dtype=np.uint8
            ),
        }
    )


def _verify_shared_task_fields(
    reference_hashes: Mapping[str, str],
    candidate_hashes: Mapping[str, str],
) -> None:
    shared = (
        "linewidths/temperatures",
        "linewidths/band_indices",
        "interactions/band_indices",
        *(f"phonons/{name}" for name in _PHONON_DATASETS),
        *(f"metadata/{name}" for name in _METADATA_DATASETS),
    )
    for name in shared:
        if candidate_hashes.get(name) != reference_hashes.get(name):
            raise BulkTargetQMergeError(
                f"task artifacts disagree on shared dataset {name!r}."
            )


def _publish_merged_artifact(
    campaign: BulkTargetQCampaign,
    task_artifacts: Sequence[tuple[Path, str, str]],
    target: Path,
) -> tuple[Mapping[str, str], tuple[str, ...]]:
    h5py = _require_h5py()
    shapes = _merged_shapes(campaign.controls)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    task_dataset_hashes: list[Mapping[str, str]] = []
    try:
        with h5py.File(temporary, "x") as merged:
            merged.attrs["schema_name"] = _MERGED_SCHEMA
            merged.attrs["schema_version"] = _SCHEMA_VERSION
            merged.attrs["campaign_manifest_sha256"] = (
                campaign.campaign_manifest_sha256
            )
            merged.attrs["manifest_run_identity_sha256"] = (
                campaign.manifest.run_identity_sha256
            )
            merged.attrs["problem_sha256"] = (
                campaign.controls.problem_sha256
            )
            merged.attrs["controls_sha256"] = (
                campaign.controls.controls_sha256
            )
            merged.attrs["canonical_task_ids_json"] = _canonical_json(
                campaign.canonical_task_ids
            )
            merged.attrs["floating_point_mpi_reductions"] = False
            for group_name in ("linewidths", "interactions", "phonons", "metadata"):
                merged.create_group(group_name, track_order=False)

            with h5py.File(task_artifacts[0][0], "r") as first:
                merged_metadata = _merged_metadata_arrays(first, campaign)
                for name in _ALL_DATASET_PATHS:
                    if name in merged_metadata:
                        values = merged_metadata[name]
                        merged.create_dataset(name, data=values)
                    else:
                        source = first[name]
                        shape = shapes.get(name, tuple(source.shape))
                        merged.create_dataset(
                            name, shape=shape, dtype=source.dtype
                        )

            first_hashes: Mapping[str, str] | None = None
            for position, (path, _, _) in enumerate(task_artifacts):
                task = campaign.manifest.tasks[position]
                evidence = _task_artifact_evidence(path, task, campaign)
                hashes = evidence["dataset_sha256"]
                assert isinstance(hashes, Mapping)
                if first_hashes is None:
                    first_hashes = hashes
                else:
                    _verify_shared_task_fields(first_hashes, hashes)
                task_dataset_hashes.append(hashes)
                with h5py.File(path, "r") as source:
                    _copy_task_into_merged(
                        source,
                        merged,
                        target_position=position,
                        first_task=position == 0,
                    )

            dataset_hashes: dict[str, str] = {}
            for name in _ALL_DATASET_PATHS:
                dataset = merged[name]
                digest = _dataset_sha256(dataset)
                dataset.attrs["sha256"] = digest
                dataset_hashes[name] = digest
            merged.attrs["dataset_sha256_json"] = _canonical_json(
                dataset_hashes
            )
            merged.attrs["task_artifact_sha256_json"] = _canonical_json(
                [item[1] for item in task_artifacts]
            )
            merged.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = _merged_artifact_evidence(target, campaign)
            if dict(existing["dataset_sha256"]) != dataset_hashes:
                raise DuplicateTaskCheckpointError(
                    "an incompatible merged target-q artifact already exists: "
                    f"{target}"
                )
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return MappingProxyType(dict(dataset_hashes)), tuple(
        item[1] for item in task_artifacts
    )


def _merged_artifact_evidence(
    path: Path,
    campaign: BulkTargetQCampaign,
) -> dict[str, Any]:
    h5py = _require_h5py()
    shapes = _merged_shapes(campaign.controls)
    try:
        with h5py.File(path, "r") as handle:
            if frozenset(str(name) for name in handle.attrs) != (
                _expected_merged_root_attributes()
            ):
                raise ValueError(
                    "merged target-q root attributes are not exact."
                )
            if _text_attribute(handle, "schema_name") != _MERGED_SCHEMA:
                raise ValueError("wrong merged target-q schema")
            if int(handle.attrs.get("schema_version", -1)) != _SCHEMA_VERSION:
                raise ValueError("wrong merged target-q schema version")
            identities = {
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
                "manifest_run_identity_sha256": (
                    campaign.manifest.run_identity_sha256
                ),
                "problem_sha256": campaign.controls.problem_sha256,
                "controls_sha256": campaign.controls.controls_sha256,
            }
            for name, expected in identities.items():
                if _text_attribute(handle, name) != expected:
                    raise BulkTargetQArtifactMismatchError(
                        f"merged artifact {name} identity does not match."
                    )
            if json.loads(
                _text_attribute(handle, "canonical_task_ids_json")
            ) != list(campaign.canonical_task_ids):
                raise BulkTargetQArtifactMismatchError(
                    "merged artifact task order does not match."
                )
            if bool(
                handle.attrs.get("floating_point_mpi_reductions", True)
            ):
                raise ValueError("merged artifact claims an MPI float reduction.")
            _validate_group_schema(handle)
            if _dataset_paths(handle) != tuple(sorted(_ALL_DATASET_PATHS)):
                raise ValueError("merged target-q dataset schema is not exact.")
            dataset_hashes: dict[str, str] = {}
            expected_dtypes = _expected_dataset_dtypes()
            for name in _ALL_DATASET_PATHS:
                dataset = handle[name]
                if name in shapes and tuple(dataset.shape) != shapes[name]:
                    raise ValueError(f"dataset {name!r} has the wrong shape.")
                if np.dtype(dataset.dtype) != expected_dtypes[name]:
                    raise ValueError(f"dataset {name!r} has the wrong dtype.")
                if frozenset(str(key) for key in dataset.attrs) != {
                    "sha256"
                }:
                    raise ValueError(
                        f"dataset {name!r} attributes are not exact."
                    )
                digest = _dataset_sha256(dataset)
                if _text_attribute(dataset, "sha256") != digest:
                    raise ValueError(f"dataset {name!r} failed SHA256 validation.")
                dataset_hashes[name] = digest
            if json.loads(
                _text_attribute(handle, "dataset_sha256_json")
            ) != dataset_hashes:
                raise ValueError("merged dataset SHA256 manifest is invalid.")
            task_hashes = tuple(
                str(value)
                for value in json.loads(
                    _text_attribute(handle, "task_artifact_sha256_json")
                )
            )
            if len(task_hashes) != len(campaign.manifest.tasks) or any(
                _SHA256_PATTERN.fullmatch(value) is None
                for value in task_hashes
            ):
                raise ValueError("merged task artifact hash list is invalid.")
            _validate_metadata_coherence(
                handle,
                controls=campaign.controls,
                campaign_manifest_sha256=(
                    campaign.campaign_manifest_sha256
                ),
            )
            _validate_common_numerics(
                handle,
                controls=campaign.controls,
                expected_grid_points=campaign.controls.grid_points,
            )
    except BulkTargetQArtifactMismatchError:
        raise
    except Exception as exc:
        raise BulkTargetQArtifactError(
            f"invalid merged target-q HDF5 artifact: {path}"
        ) from exc
    return {
        "schema_name": _MERGED_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "dataset_sha256": dataset_hashes,
        "task_artifact_sha256": task_hashes,
    }


@dataclass(frozen=True, slots=True)
class BulkTargetQMergeReport:
    """Canonical filesystem merge evidence for one campaign."""

    campaign_manifest_sha256: str
    merged_artifact_path: Path
    merged_artifact_sha256: str
    task_ids: tuple[str, ...]
    target_grid_points: tuple[int, ...]
    task_artifact_sha256: tuple[str, ...]
    task_checkpoint_sha256: tuple[str, ...]
    dataset_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "campaign_manifest_sha256",
            _validate_sha256(
                self.campaign_manifest_sha256,
                name="campaign_manifest_sha256",
            ),
        )
        object.__setattr__(
            self, "merged_artifact_path", Path(self.merged_artifact_path)
        )
        object.__setattr__(
            self,
            "merged_artifact_sha256",
            _validate_sha256(
                self.merged_artifact_sha256,
                name="merged_artifact_sha256",
            ),
        )
        task_ids = tuple(str(value) for value in self.task_ids)
        qpoints = tuple(int(value) for value in self.target_grid_points)
        artifact_hashes = tuple(
            _validate_sha256(value, name="task_artifact_sha256")
            for value in self.task_artifact_sha256
        )
        checkpoint_hashes = tuple(
            _validate_sha256(value, name="task_checkpoint_sha256")
            for value in self.task_checkpoint_sha256
        )
        if not (
            task_ids
            and len(task_ids)
            == len(qpoints)
            == len(artifact_hashes)
            == len(checkpoint_hashes)
        ):
            raise ValueError("merge report task fields have inconsistent lengths.")
        object.__setattr__(self, "task_ids", task_ids)
        object.__setattr__(self, "target_grid_points", qpoints)
        object.__setattr__(self, "task_artifact_sha256", artifact_hashes)
        object.__setattr__(
            self, "task_checkpoint_sha256", checkpoint_hashes
        )
        object.__setattr__(
            self,
            "dataset_sha256",
            _hash_mapping(self.dataset_sha256, name="dataset_sha256"),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": _MERGE_REPORT_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "campaign_manifest_sha256": self.campaign_manifest_sha256,
            "merged_artifact_path": self.merged_artifact_path.as_posix(),
            "merged_artifact_sha256": self.merged_artifact_sha256,
            "task_ids": list(self.task_ids),
            "target_grid_points": list(self.target_grid_points),
            "task_artifact_sha256": list(self.task_artifact_sha256),
            "task_checkpoint_sha256": list(self.task_checkpoint_sha256),
            "dataset_sha256": dict(self.dataset_sha256),
        }

    @property
    def report_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self._payload()).encode("ascii"))

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "report_sha256": self.report_sha256}


def merge_bulk_target_q_campaign(
    campaign: BulkTargetQCampaign,
    run_directory: str | Path,
    *,
    output_path: str | Path | None = None,
) -> BulkTargetQMergeReport:
    """Strictly collect and stream task artifacts into canonical q order."""

    if not isinstance(campaign, BulkTargetQCampaign):
        raise TypeError("campaign must be BulkTargetQCampaign.")
    root = Path(run_directory).expanduser().resolve()
    task_artifacts = _task_artifact_paths(campaign, root)
    target = (
        root / "merged" / "bulk-target-q.h5"
        if output_path is None
        else Path(output_path).expanduser().resolve()
    )
    if target.exists():
        evidence = _merged_artifact_evidence(target, campaign)
        dataset_hashes = evidence["dataset_sha256"]
        task_hashes = tuple(evidence["task_artifact_sha256"])
        expected_task_hashes = tuple(item[1] for item in task_artifacts)
        if task_hashes != expected_task_hashes:
            raise BulkTargetQMergeError(
                "existing merged artifact was built from different task files."
            )
    else:
        dataset_hashes, task_hashes = _publish_merged_artifact(
            campaign, task_artifacts, target
        )
        _merged_artifact_evidence(target, campaign)
    return BulkTargetQMergeReport(
        campaign_manifest_sha256=campaign.campaign_manifest_sha256,
        merged_artifact_path=target,
        merged_artifact_sha256=_file_sha256(target),
        task_ids=campaign.canonical_task_ids,
        target_grid_points=tuple(
            int(value) for value in campaign.controls.grid_points
        ),
        task_artifact_sha256=tuple(task_hashes),
        task_checkpoint_sha256=tuple(item[2] for item in task_artifacts),
        dataset_sha256=dataset_hashes,
    )


def _read_metadata(handle: Any, name: str) -> Mapping[str, Any]:
    return _validate_json_dataset(handle[f"metadata/{name}_json"], name=name)


def load_bulk_target_q_result(
    path: str | Path,
    *,
    campaign: BulkTargetQCampaign | None = None,
) -> BulkThreePhononLinewidths:
    """Load a merged artifact and reconstruct every scientific result field."""

    source = Path(path).expanduser().resolve()
    if campaign is not None:
        if not isinstance(campaign, BulkTargetQCampaign):
            raise TypeError("campaign must be BulkTargetQCampaign or None.")
        _merged_artifact_evidence(source, campaign)
    h5py = _require_h5py()
    try:
        with h5py.File(source, "r") as handle:
            if frozenset(str(name) for name in handle.attrs) != (
                _expected_merged_root_attributes()
            ):
                raise ValueError(
                    "merged target-q root attributes are not exact."
                )
            if _text_attribute(handle, "schema_name") != _MERGED_SCHEMA:
                raise ValueError("path is not a merged target-q artifact.")
            if int(handle.attrs.get("schema_version", -1)) != _SCHEMA_VERSION:
                raise ValueError("unsupported merged target-q schema version.")
            for name in (
                "campaign_manifest_sha256",
                "manifest_run_identity_sha256",
                "problem_sha256",
                "controls_sha256",
            ):
                _validate_sha256(_text_attribute(handle, name), name=name)
            task_ids = json.loads(
                _text_attribute(handle, "canonical_task_ids_json")
            )
            task_hashes = json.loads(
                _text_attribute(handle, "task_artifact_sha256_json")
            )
            if (
                not isinstance(task_ids, list)
                or not task_ids
                or not all(
                    isinstance(task_id, str)
                    and _TASK_ID_PATTERN.fullmatch(task_id) is not None
                    for task_id in task_ids
                )
                or not isinstance(task_hashes, list)
                or len(task_hashes) != len(task_ids)
            ):
                raise ValueError("merged target-q task identity lists are invalid.")
            for index, value in enumerate(task_hashes):
                _validate_sha256(value, name=f"task_artifact_sha256[{index}]")
            if bool(
                handle.attrs.get("floating_point_mpi_reductions", True)
            ):
                raise ValueError("merged artifact claims an MPI float reduction.")
            _validate_group_schema(handle)
            if _dataset_paths(handle) != tuple(sorted(_ALL_DATASET_PATHS)):
                raise ValueError("merged target-q dataset schema is not exact.")
            dataset_hashes: dict[str, str] = {}
            expected_dtypes = _expected_dataset_dtypes()
            for name in _ALL_DATASET_PATHS:
                if np.dtype(handle[name].dtype) != expected_dtypes[name]:
                    raise ValueError(f"dataset {name!r} has the wrong dtype.")
                if frozenset(str(key) for key in handle[name].attrs) != {
                    "sha256"
                }:
                    raise ValueError(
                        f"dataset {name!r} attributes are not exact."
                    )
                digest = _dataset_sha256(handle[name])
                if _text_attribute(handle[name], "sha256") != digest:
                    raise ValueError(f"dataset {name!r} failed SHA256 validation.")
                dataset_hashes[name] = digest
            if json.loads(
                _text_attribute(handle, "dataset_sha256_json")
            ) != dataset_hashes:
                raise ValueError("merged dataset SHA256 manifest is invalid.")
            inferred_controls = BulkTargetQControls(
                temperatures=np.asarray(
                    handle["linewidths/temperatures"]
                ),
                grid_points=np.asarray(handle["linewidths/grid_points"]),
                band_indices=np.asarray(
                    handle["linewidths/band_indices"]
                ),
                n_qpoints=int(handle["phonons/qpoints"].shape[0]),
                n_bands=int(handle["phonons/frequencies"].shape[1]),
                problem_sha256=_text_attribute(
                    handle, "problem_sha256"
                ),
            )
            _validate_metadata_coherence(
                handle,
                controls=inferred_controls,
                campaign_manifest_sha256=_text_attribute(
                    handle, "campaign_manifest_sha256"
                ),
            )
            _validate_common_numerics(
                handle,
                controls=inferred_controls,
                expected_grid_points=inferred_controls.grid_points,
            )
            expected_task_ids = [
                _target_task_id(int(qpoint))
                for qpoint in inferred_controls.grid_points
            ]
            if task_ids != expected_task_ids:
                raise ValueError(
                    "merged task identifiers conflict with target q points."
                )
            phonons = BulkPhononSpectrum(
                qpoints=np.asarray(handle["phonons/qpoints"]),
                dynamical_matrices=np.asarray(
                    handle["phonons/dynamical_matrices"]
                ),
                frequencies=np.asarray(handle["phonons/frequencies"]),
                eigenvectors=np.asarray(handle["phonons/eigenvectors"]),
                frequency_factor_to_thz=float(
                    np.asarray(
                        handle["phonons/frequency_factor_to_thz"]
                    )
                ),
                hermitianized=bool(
                    np.asarray(handle["phonons/hermitianized"])
                ),
            )
            interactions = BulkThreePhononInteractions(
                grid_points=np.asarray(handle["interactions/grid_points"]),
                band_indices=np.asarray(handle["interactions/band_indices"]),
                q1_indices=np.asarray(handle["interactions/q1_indices"]),
                q2_indices=np.asarray(handle["interactions/q2_indices"]),
                reciprocal_vectors=np.asarray(
                    handle["interactions/reciprocal_vectors"]
                ),
                normal_mask=np.asarray(handle["interactions/normal_mask"]),
                matrix_elements=np.asarray(
                    handle["interactions/matrix_elements"]
                ),
                interaction_strengths=np.asarray(
                    handle["interactions/interaction_strengths"]
                ),
                phonons=phonons,
                metadata=_read_metadata(handle, "interactions"),
            )
            return BulkThreePhononLinewidths(
                temperatures=np.asarray(handle["linewidths/temperatures"]),
                grid_points=np.asarray(handle["linewidths/grid_points"]),
                band_indices=np.asarray(handle["linewidths/band_indices"]),
                gamma_thz_hwhm=np.asarray(
                    handle["linewidths/gamma_thz_hwhm"]
                ),
                gamma_normal_thz_hwhm=np.asarray(
                    handle["linewidths/gamma_normal_thz_hwhm"]
                ),
                gamma_umklapp_thz_hwhm=np.asarray(
                    handle["linewidths/gamma_umklapp_thz_hwhm"]
                ),
                gamma_angular_fwhm=np.asarray(
                    handle["linewidths/gamma_angular_fwhm"]
                ),
                decay_phase_space=np.asarray(
                    handle["linewidths/decay_phase_space"]
                ),
                difference_phase_space=np.asarray(
                    handle["linewidths/difference_phase_space"]
                ),
                interactions=interactions,
                metadata=_read_metadata(handle, "linewidths"),
            )
    except BulkTargetQArtifactError:
        raise
    except Exception as exc:
        raise BulkTargetQArtifactError(
            f"cannot load merged target-q result: {source}"
        ) from exc


def _result_field_arrays(
    result: BulkThreePhononLinewidths,
) -> dict[str, Array]:
    return {
        name: values
        for name, values in _result_arrays(result).items()
        if not name.startswith("metadata/")
    }


def _comparison_metadata(value: Any) -> Any:
    """Drop only task-orchestration and estimate fields before equality."""

    excluded = {"estimated_result_bytes", "target_q_execution"}
    if isinstance(value, Mapping):
        return {
            str(key): _comparison_metadata(item)
            for key, item in sorted(value.items())
            if str(key) not in excluded
        }
    if isinstance(value, (tuple, list)):
        return [_comparison_metadata(item) for item in value]
    if isinstance(value, np.ndarray):
        return _comparison_metadata(value.tolist())
    if isinstance(value, np.generic):
        return _comparison_metadata(value.item())
    return value


def _comparison_result_sha256(
    result: BulkThreePhononLinewidths,
) -> str:
    """Bind every compared field and normalized scientific metadata."""

    digest = hashlib.sha256()
    digest.update(b"negfpy.bulk_target_q_comparison_result.v1\0")
    for name, values in sorted(_result_field_arrays(result).items()):
        _update_scientific_digest(digest, name, path=f"fields.{name}.name")
        _update_scientific_digest(
            digest,
            np.asarray(values),
            path=f"fields.{name}.values",
        )
    for name, metadata in (
        ("linewidths", result.metadata),
        ("interactions", result.interactions.metadata),
    ):
        _update_scientific_digest(
            digest,
            name,
            path=f"metadata.{name}.name",
        )
        _update_scientific_digest(
            digest,
            _comparison_metadata(metadata),
            path=f"metadata.{name}.values",
        )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BulkTargetQComparisonReport:
    """Exact/in-tolerance equality report against the monolithic serial oracle."""

    passed: bool
    rtol: float
    atol: float
    serial_result_sha256: str
    merged_result_sha256: str
    merged_artifact_sha256: str | None
    exact_field_passed: Mapping[str, bool]
    numerical_field_passed: Mapping[str, bool]
    maximum_absolute_error: Mapping[str, float]
    maximum_relative_error: Mapping[str, float]
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rtol = float(self.rtol)
        atol = float(self.atol)
        if not math.isfinite(rtol) or rtol < 0.0:
            raise ValueError("rtol must be finite and nonnegative.")
        if not math.isfinite(atol) or atol < 0.0:
            raise ValueError("atol must be finite and nonnegative.")
        serial_result_sha256 = _validate_sha256(
            self.serial_result_sha256,
            name="serial_result_sha256",
        )
        merged_result_sha256 = _validate_sha256(
            self.merged_result_sha256,
            name="merged_result_sha256",
        )
        merged_artifact_sha256 = self.merged_artifact_sha256
        if merged_artifact_sha256 is not None:
            merged_artifact_sha256 = _validate_sha256(
                merged_artifact_sha256,
                name="merged_artifact_sha256",
            )
        exact = MappingProxyType(
            {
                str(name): bool(value)
                for name, value in sorted(self.exact_field_passed.items())
            }
        )
        numerical = MappingProxyType(
            {
                str(name): bool(value)
                for name, value in sorted(
                    self.numerical_field_passed.items()
                )
            }
        )
        absolute = MappingProxyType(
            {
                str(name): float(value)
                for name, value in sorted(
                    self.maximum_absolute_error.items()
                )
            }
        )
        relative = MappingProxyType(
            {
                str(name): float(value)
                for name, value in sorted(
                    self.maximum_relative_error.items()
                )
            }
        )
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (*absolute.values(), *relative.values())
        ):
            raise ValueError("comparison errors must be finite and nonnegative.")
        errors = tuple(str(value) for value in self.errors)
        derived = (
            all(exact.values())
            and all(numerical.values())
            and not errors
        )
        if bool(self.passed) != derived:
            raise ValueError("passed conflicts with comparison field results.")
        object.__setattr__(self, "passed", derived)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(
            self,
            "serial_result_sha256",
            serial_result_sha256,
        )
        object.__setattr__(
            self,
            "merged_result_sha256",
            merged_result_sha256,
        )
        object.__setattr__(
            self,
            "merged_artifact_sha256",
            merged_artifact_sha256,
        )
        object.__setattr__(self, "exact_field_passed", exact)
        object.__setattr__(self, "numerical_field_passed", numerical)
        object.__setattr__(self, "maximum_absolute_error", absolute)
        object.__setattr__(self, "maximum_relative_error", relative)
        object.__setattr__(self, "errors", errors)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": _COMPARISON_REPORT_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "passed": self.passed,
            "rtol": self.rtol,
            "atol": self.atol,
            "serial_result_sha256": self.serial_result_sha256,
            "merged_result_sha256": self.merged_result_sha256,
            "merged_artifact_sha256": self.merged_artifact_sha256,
            "exact_field_passed": dict(self.exact_field_passed),
            "numerical_field_passed": dict(self.numerical_field_passed),
            "maximum_absolute_error": dict(self.maximum_absolute_error),
            "maximum_relative_error": dict(self.maximum_relative_error),
            "errors": list(self.errors),
        }

    @property
    def report_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self._payload()).encode("ascii"))

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "report_sha256": self.report_sha256}


def compare_bulk_target_q_with_serial(
    serial_result: BulkThreePhononLinewidths,
    merged_result: BulkThreePhononLinewidths | str | Path,
    *,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
) -> BulkTargetQComparisonReport:
    """Compare all fields to a monolithic serial result.

    Integer and Boolean routing/selector fields are exact.  Every real and
    complex numerical field uses the fixed backend equality tolerances.
    """

    if not isinstance(serial_result, BulkThreePhononLinewidths):
        raise TypeError("serial_result must be BulkThreePhononLinewidths.")
    merged_artifact_sha256 = (
        _file_sha256(Path(merged_result).expanduser().resolve())
        if isinstance(merged_result, (str, Path))
        else None
    )
    candidate = (
        load_bulk_target_q_result(merged_result)
        if isinstance(merged_result, (str, Path))
        else merged_result
    )
    if not isinstance(candidate, BulkThreePhononLinewidths):
        raise TypeError(
            "merged_result must be BulkThreePhononLinewidths or an HDF5 path."
        )
    rtol = float(rtol)
    atol = float(atol)
    if not math.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and nonnegative.")
    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and nonnegative.")

    reference = _result_field_arrays(serial_result)
    observed = _result_field_arrays(candidate)
    errors: list[str] = []
    if set(reference) != set(observed):
        errors.append("serial and merged result schemas differ")
    exact: dict[str, bool] = {}
    numerical: dict[str, bool] = {}
    maximum_absolute: dict[str, float] = {}
    maximum_relative: dict[str, float] = {}
    for name in sorted(set(reference).intersection(observed)):
        expected = np.asarray(reference[name])
        actual = np.asarray(observed[name])
        if expected.shape != actual.shape:
            errors.append(
                f"{name}: shape {actual.shape} != serial {expected.shape}"
            )
            if expected.dtype.kind in "biu" and actual.dtype.kind in "biu":
                exact[name] = False
            else:
                numerical[name] = False
                maximum_absolute[name] = 0.0
                maximum_relative[name] = 0.0
            continue
        if expected.dtype.kind in "biu" and actual.dtype.kind in "biu":
            exact[name] = bool(np.array_equal(actual, expected))
            continue
        difference = np.abs(actual - expected)
        max_absolute = (
            0.0 if difference.size == 0 else float(np.max(difference))
        )
        denominator = np.maximum(np.abs(expected), atol)
        relative = difference / denominator
        max_relative = (
            0.0 if relative.size == 0 else float(np.max(relative))
        )
        passed = bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
        numerical[name] = passed
        maximum_absolute[name] = max_absolute
        maximum_relative[name] = max_relative
    metadata_pairs = {
        "metadata/linewidths_json": (
            serial_result.metadata,
            candidate.metadata,
        ),
        "metadata/interactions_json": (
            serial_result.interactions.metadata,
            candidate.interactions.metadata,
        ),
    }
    for name, (expected_metadata, actual_metadata) in metadata_pairs.items():
        expected_canonical = _canonical_json(
            _comparison_metadata(expected_metadata)
        )
        actual_canonical = _canonical_json(
            _comparison_metadata(actual_metadata)
        )
        exact[name] = actual_canonical == expected_canonical
    overall = all(exact.values()) and all(numerical.values()) and not errors
    return BulkTargetQComparisonReport(
        passed=overall,
        rtol=rtol,
        atol=atol,
        serial_result_sha256=_comparison_result_sha256(serial_result),
        merged_result_sha256=_comparison_result_sha256(candidate),
        merged_artifact_sha256=merged_artifact_sha256,
        exact_field_passed=exact,
        numerical_field_passed=numerical,
        maximum_absolute_error=maximum_absolute,
        maximum_relative_error=maximum_relative,
        errors=tuple(errors),
    )


def _write_report(
    path: str | Path,
    report: BulkTargetQMergeReport | BulkTargetQComparisonReport,
) -> Path:
    target = Path(path).expanduser()
    payload = (_canonical_json(report.as_dict()) + "\n").encode("ascii")
    return mpi_tasks._publish_immutable(  # private-to-private
        target,
        payload,
        conflict_type=TaskManifestError,
    )


def write_bulk_target_q_merge_report(
    path: str | Path,
    report: BulkTargetQMergeReport,
) -> Path:
    """Publish an immutable canonical merge report."""

    if not isinstance(report, BulkTargetQMergeReport):
        raise TypeError("report must be BulkTargetQMergeReport.")
    return _write_report(path, report)


def write_bulk_target_q_comparison_report(
    path: str | Path,
    report: BulkTargetQComparisonReport,
) -> Path:
    """Publish an immutable canonical serial-equality report."""

    if not isinstance(report, BulkTargetQComparisonReport):
        raise TypeError("report must be BulkTargetQComparisonReport.")
    return _write_report(path, report)


__all__ = [
    "BulkTargetQArtifactError",
    "BulkTargetQArtifactMismatchError",
    "BulkTargetQCampaign",
    "BulkTargetQCampaignError",
    "BulkTargetQComparisonReport",
    "BulkTargetQControls",
    "BulkTargetQMergeError",
    "BulkTargetQMergeReport",
    "build_bulk_target_q_campaign",
    "bulk_three_phonon_problem_sha256",
    "compare_bulk_target_q_with_serial",
    "execute_bulk_target_q_campaign",
    "load_bulk_target_q_result",
    "make_bulk_target_q_artifact_validator",
    "make_bulk_target_q_orphan_adopter",
    "make_bulk_target_q_worker",
    "merge_bulk_target_q_campaign",
    "validate_problem_against_controls",
    "write_bulk_target_q_comparison_report",
    "write_bulk_target_q_merge_report",
]
