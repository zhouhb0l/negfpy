"""Private, atomic HDF5 checkpoints for completed dense-SCBA outcomes.

Only the mixed lesser/greater phonon--phonon self-energy is persisted.  The
retarded component, Green functions, currents, and diagnostics are deliberately
recomputed after loading so a restart cannot bypass the causal reconstruction
or the final validation path.

``h5py`` remains optional at package-import time.  Calling an HDF5 operation
without it installed raises an actionable :class:`CheckpointDependencyError`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np

from ._provenance import (
    code_provenance,
    configuration_hash,
    initial_self_energy_hash,
    problem_input_hashes,
)
from .retarded import retarded_from_keldysh
from .types import (
    AnharmonicResult,
    CubicSCBAProblem,
    KeldyshSpectrum,
    SCBAConfig,
    SCBAIteration,
)


Array = np.ndarray
_SCHEMA_NAME = "negfpy_dense_scba_checkpoint"
_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMPLETED_STATUSES = frozenset({"converged", "nonconverged_partial"})
_HISTORY_DTYPE = np.dtype(
    [
        ("iteration", "<i8"),
        ("sigma_residual", "<f8"),
        ("green_residual", "<f8"),
        ("identity_residual", "<f8"),
        ("current_abs_error", "<f8"),
        ("current_rel_error", "<f8"),
        ("collision_power", "<f8"),
        ("mixing", "<f8"),
        ("passed", "u1"),
        ("consecutive_passes", "<i8"),
    ]
)


class CheckpointError(RuntimeError):
    """Base class for checkpoint I/O and validation failures."""


class CheckpointDependencyError(CheckpointError):
    """Raised when optional HDF5 support is unavailable."""


class CheckpointMismatchError(CheckpointError):
    """Raised when a checkpoint does not belong to the requested run."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint is incomplete or fails an integrity check."""


def _require_h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as exc:  # pragma: no cover - exercised by monkeypatch
        raise CheckpointDependencyError(
            "SCBA checkpointing requires the optional 'h5py' package. "
            "Install it with `pip install 'negfpy[test]'` or "
            "`pip install 'negfpy[fc3]'`."
        ) from exc


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256.")
    return value


def _validate_run_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("run_id must be a non-empty string.")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError("run_id cannot contain surrounding whitespace or controls.")
    return value


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    """Return a detached strict-JSON value, rejecting lossy representations."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _normalize_json(value.item(), path=path)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return number
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


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_json(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _text(value: Any, *, name: str) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        try:
            return bytes(value).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CheckpointCorruptionError(
                f"checkpoint {name} is not valid UTF-8."
            ) from exc
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        return _text(value.item(), name=name)
    raise CheckpointCorruptionError(f"checkpoint {name} is not text.")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<c16"))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _history_array(history: tuple[SCBAIteration, ...]) -> Array:
    values = np.empty(len(history), dtype=_HISTORY_DTYPE)
    for index, item in enumerate(history):
        values[index] = (
            item.iteration,
            item.sigma_residual,
            item.green_residual,
            item.identity_residual,
            item.current_abs_error,
            item.current_rel_error,
            item.collision_power,
            item.mixing,
            int(item.passed),
            item.consecutive_passes,
        )
    return values


def _history_sha256(values: Array) -> str:
    canonical = np.ascontiguousarray(values, dtype=_HISTORY_DTYPE)
    digest = hashlib.sha256()
    digest.update(_HISTORY_DTYPE.descr.__repr__().encode("ascii"))
    digest.update(repr(tuple(int(value) for value in canonical.shape)).encode("ascii"))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _history_records(values: Array) -> tuple[SCBAIteration, ...]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.names != _HISTORY_DTYPE.names:
        raise CheckpointCorruptionError(
            "checkpoint iteration_history has an incompatible schema."
        )
    try:
        canonical = np.asarray(raw, dtype=_HISTORY_DTYPE)
        return tuple(
            SCBAIteration(
                iteration=int(item["iteration"]),
                sigma_residual=float(item["sigma_residual"]),
                green_residual=float(item["green_residual"]),
                identity_residual=float(item["identity_residual"]),
                current_abs_error=float(item["current_abs_error"]),
                current_rel_error=float(item["current_rel_error"]),
                collision_power=float(item["collision_power"]),
                mixing=float(item["mixing"]),
                passed=bool(item["passed"]),
                consecutive_passes=int(item["consecutive_passes"]),
            )
            for item in canonical
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise CheckpointCorruptionError(
            "checkpoint iteration_history contains invalid values."
        ) from exc


def _readonly_sigma(values: Any, *, name: str) -> Array:
    array = np.array(values, dtype=np.complex128, order="C", copy=True)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(
            f"{name} must have shape (n_frequency, ndof, ndof)."
        )
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _nonnegative_timings(values: Mapping[str, Any]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError("timings must be a mapping.")
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError("timing names must be non-empty strings.")
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"timings[{key!r}] must be a real number.")
        number = float(value)
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(f"timings[{key!r}] must be finite and non-negative.")
        normalized[key] = number
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class SCBACheckpointIdentity:
    """Canonical identity of one preregistered SCBA run."""

    run_id: str
    configuration_sha256: str
    problem_sha256: str
    initial_state_sha256: str
    code_sha256: str
    environment_sha256: str

    schema_name: ClassVar[str] = _SCHEMA_NAME
    schema_version: ClassVar[int] = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _validate_run_id(self.run_id))
        for name in (
            "configuration_sha256",
            "problem_sha256",
            "initial_state_sha256",
            "code_sha256",
            "environment_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _validate_sha256(getattr(self, name), name=name),
            )

    def as_dict(self) -> dict[str, str | int]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "configuration_sha256": self.configuration_sha256,
            "problem_sha256": self.problem_sha256,
            "initial_state_sha256": self.initial_state_sha256,
            "code_sha256": self.code_sha256,
            "environment_sha256": self.environment_sha256,
        }

    @property
    def identity_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self.as_dict()).encode("ascii"))


def make_scba_checkpoint_identity(
    run_id: str,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    environment_sha256: str,
    initial_sigma_pp: KeldyshSpectrum | None = None,
    code_sha256: str | None = None,
) -> SCBACheckpointIdentity:
    """Derive the complete restart identity from actual solver inputs."""

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")
    if code_sha256 is None:
        source_hash = code_provenance().get("source_tree_sha256")
        if not isinstance(source_hash, str):
            raise RuntimeError("could not determine the NEGFPy source-tree SHA256.")
        code_sha256 = source_hash
    return SCBACheckpointIdentity(
        run_id=run_id,
        configuration_sha256=configuration_hash(config),
        problem_sha256=problem_input_hashes(problem)["problem"],
        initial_state_sha256=initial_self_energy_hash(
            problem,
            initial_sigma_pp,
        ),
        code_sha256=code_sha256,
        environment_sha256=environment_sha256,
    )


@dataclass(frozen=True, slots=True, eq=False)
class SCBACheckpoint:
    """Restart state and evidence for one completed SCBA outcome."""

    identity: SCBACheckpointIdentity
    sigma_lesser: Array
    sigma_greater: Array
    history: tuple[SCBAIteration, ...] = ()
    timings: Mapping[str, float] = MappingProxyType({})
    status: str = "converged"
    metadata: Mapping[str, Any] = MappingProxyType({})

    def __post_init__(self) -> None:
        if not isinstance(self.identity, SCBACheckpointIdentity):
            raise TypeError("identity must be an SCBACheckpointIdentity.")
        lesser = _readonly_sigma(self.sigma_lesser, name="sigma_lesser")
        greater = _readonly_sigma(self.sigma_greater, name="sigma_greater")
        if lesser.shape != greater.shape:
            raise ValueError("sigma_lesser and sigma_greater must share one shape.")
        history = tuple(self.history)
        if any(not isinstance(item, SCBAIteration) for item in history):
            raise TypeError("history must contain only SCBAIteration records.")
        if self.status not in _COMPLETED_STATUSES:
            raise ValueError(
                "status must be 'converged' or 'nonconverged_partial'."
            )
        normalized_metadata = _normalize_json(self.metadata)
        if not isinstance(normalized_metadata, dict):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "sigma_lesser", lesser)
        object.__setattr__(self, "sigma_greater", greater)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "timings", _nonnegative_timings(self.timings))
        object.__setattr__(self, "metadata", _freeze_json(normalized_metadata))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.sigma_lesser.shape

    @classmethod
    def from_result(
        cls,
        identity: SCBACheckpointIdentity,
        result: AnharmonicResult,
        *,
        timings: Mapping[str, float] | None = None,
        status: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SCBACheckpoint:
        """Build a checkpoint from a completed SCBA result.

        The result's own solver provenance must match the supplied identity;
        this prevents a caller from labelling an old result as a new run.
        """

        if not isinstance(result, AnharmonicResult):
            raise TypeError("result must be an AnharmonicResult.")
        if result.method != "scba":
            raise ValueError("only SCBA results can be checkpointed.")
        selected_status = (
            "converged" if result.converged else "nonconverged_partial"
        ) if status is None else status
        if selected_status == "converged" and not result.converged:
            raise ValueError("converged status requires a converged SCBA result.")
        if selected_status == "nonconverged_partial" and result.converged:
            raise ValueError(
                "nonconverged_partial status requires a nonconverged result."
            )

        input_hashes = result.metadata.get("input_hashes")
        if not isinstance(input_hashes, Mapping):
            raise CheckpointMismatchError(
                "SCBA result metadata does not contain input_hashes."
            )
        expected = {
            "configuration": identity.configuration_sha256,
            "problem": identity.problem_sha256,
            "initial_state": identity.initial_state_sha256,
        }
        for name, digest in expected.items():
            if input_hashes.get(name) != digest:
                raise CheckpointMismatchError(
                    f"SCBA result {name} hash does not match checkpoint identity."
                )
        if result.metadata.get("source_tree_sha256") != identity.code_sha256:
            raise CheckpointMismatchError(
                "SCBA result code hash does not match checkpoint identity."
            )

        record_metadata: dict[str, Any] = {
            "method": result.method,
            "converged": result.converged,
            "release_certified": result.release_certified,
            "iteration_count": len(result.history),
        }
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise TypeError("metadata must be a mapping.")
            record_metadata.update(metadata)
        return cls(
            identity=identity,
            sigma_lesser=result.sigma_pp.lesser,
            sigma_greater=result.sigma_pp.greater,
            history=result.history,
            timings={} if timings is None else timings,
            status=selected_status,
            metadata=record_metadata,
        )

    def restart_spectrum(
        self,
        problem: CubicSCBAProblem,
        config: SCBAConfig,
    ) -> KeldyshSpectrum:
        """Reconstruct the causal retarded self-energy for solver restart."""

        if not isinstance(problem, CubicSCBAProblem):
            raise TypeError("problem must be a CubicSCBAProblem.")
        if not isinstance(config, SCBAConfig):
            raise TypeError("config must be an SCBAConfig.")
        if configuration_hash(config) != self.identity.configuration_sha256:
            raise CheckpointMismatchError(
                "checkpoint configuration hash does not match the restart config."
            )
        if problem_input_hashes(problem)["problem"] != self.identity.problem_sha256:
            raise CheckpointMismatchError(
                "checkpoint problem hash does not match the restart problem."
            )
        expected_shape = (
            problem.grid.n_points,
            problem.ndof,
            problem.ndof,
        )
        if self.shape != expected_shape:
            raise CheckpointMismatchError(
                f"checkpoint self-energy shape {self.shape} does not match "
                f"restart shape {expected_shape}."
            )
        retarded = retarded_from_keldysh(
            self.sigma_lesser,
            self.sigma_greater,
            problem.grid,
            strategy=config.retarded_strategy,
            hilbert_backend=config.hilbert_backend,
        )
        return KeldyshSpectrum(
            retarded=retarded,
            lesser=self.sigma_lesser,
            greater=self.sigma_greater,
        )


def _write_text_dataset(handle: Any, name: str, value: str, h5py: Any) -> None:
    handle.create_dataset(name, data=value, dtype=h5py.string_dtype("utf-8"))


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_scba_checkpoint(
    path: str | Path,
    checkpoint: SCBACheckpoint,
) -> Path:
    """Atomically write one completed SCBA checkpoint.

    An existing file may be refreshed only for the identical canonical run
    identity.  A temporary file in the target directory is fully closed and
    fsynced before :func:`os.replace` publishes it.
    """

    if not isinstance(checkpoint, SCBACheckpoint):
        raise TypeError("checkpoint must be an SCBACheckpoint.")
    h5py = _require_h5py()
    target = Path(path).expanduser()
    if target.exists():
        load_scba_checkpoint(target, expected_identity=checkpoint.identity)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"

    lesser = np.ascontiguousarray(checkpoint.sigma_lesser, dtype="<c16")
    greater = np.ascontiguousarray(checkpoint.sigma_greater, dtype="<c16")
    history = _history_array(checkpoint.history)
    timings_json = _canonical_json(dict(checkpoint.timings))
    metadata_json = _canonical_json(dict(checkpoint.metadata))
    integrity = {
        "sigma_lesser_sha256": _array_sha256(lesser),
        "sigma_greater_sha256": _array_sha256(greater),
        "history_sha256": _history_sha256(history),
        "timings_sha256": _sha256_bytes(timings_json.encode("ascii")),
        "metadata_sha256": _sha256_bytes(metadata_json.encode("ascii")),
    }
    manifest_payload = {
        "identity_sha256": checkpoint.identity.identity_sha256,
        "status": checkpoint.status,
        **integrity,
    }
    manifest_sha256 = _sha256_bytes(
        _canonical_json(manifest_payload).encode("ascii")
    )

    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs["schema_name"] = _SCHEMA_NAME
            handle.attrs["schema_version"] = _SCHEMA_VERSION
            handle.attrs["write_state"] = "complete"
            handle.attrs["status"] = checkpoint.status
            handle.attrs["identity_sha256"] = checkpoint.identity.identity_sha256
            handle.attrs["manifest_sha256"] = manifest_sha256

            identity_group = handle.create_group("identity")
            for name, value in checkpoint.identity.as_dict().items():
                identity_group.attrs[name] = value

            sigma_group = handle.create_group("sigma_pp")
            sigma_group.create_dataset("lesser", data=lesser)
            sigma_group.create_dataset("greater", data=greater)
            history_dataset = handle.create_dataset(
                "iteration_history",
                data=history,
                dtype=_HISTORY_DTYPE,
            )
            _write_text_dataset(handle, "timings_json", timings_json, h5py)
            _write_text_dataset(handle, "metadata_json", metadata_json, h5py)

            sigma_group["lesser"].attrs["sha256"] = integrity[
                "sigma_lesser_sha256"
            ]
            sigma_group["greater"].attrs["sha256"] = integrity[
                "sigma_greater_sha256"
            ]
            history_dataset.attrs["sha256"] = integrity["history_sha256"]
            handle["timings_json"].attrs["sha256"] = integrity["timings_sha256"]
            handle["metadata_json"].attrs["sha256"] = integrity["metadata_sha256"]
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


def _load_json_dataset(handle: Any, name: str) -> tuple[dict[str, Any], str]:
    dataset = handle[name]
    text = _text(dataset[()], name=name)
    expected = _text(dataset.attrs["sha256"], name=f"{name}.sha256")
    actual = _sha256_bytes(text.encode("ascii"))
    if actual != expected:
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
        canonical = _canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointCorruptionError(
            f"checkpoint {name} contains invalid JSON metadata."
        ) from exc
    if canonical != text:
        raise CheckpointCorruptionError(
            f"checkpoint {name} is not in canonical JSON form."
        )
    return value, expected


def load_scba_checkpoint(
    path: str | Path,
    *,
    expected_identity: SCBACheckpointIdentity,
) -> SCBACheckpoint:
    """Load and validate a completed checkpoint for exactly one run identity."""

    if not isinstance(expected_identity, SCBACheckpointIdentity):
        raise TypeError("expected_identity must be an SCBACheckpointIdentity.")
    h5py = _require_h5py()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"SCBA checkpoint does not exist: {source}")

    try:
        with h5py.File(source, "r") as handle:
            schema_name = _text(
                handle.attrs["schema_name"],
                name="schema_name",
            )
            schema_version = int(handle.attrs["schema_version"])
            write_state = _text(
                handle.attrs["write_state"],
                name="write_state",
            )
            status = _text(handle.attrs["status"], name="status")
            stored_identity_hash = _text(
                handle.attrs["identity_sha256"],
                name="identity_sha256",
            )
            stored_manifest_hash = _text(
                handle.attrs["manifest_sha256"],
                name="manifest_sha256",
            )
            if schema_name != _SCHEMA_NAME or schema_version != _SCHEMA_VERSION:
                raise CheckpointCorruptionError(
                    "checkpoint schema is unsupported or invalid."
                )
            if write_state != "complete":
                raise CheckpointCorruptionError(
                    "checkpoint was not marked as a complete atomic write."
                )

            identity_group = handle["identity"]
            stored_identity = SCBACheckpointIdentity(
                run_id=_text(identity_group.attrs["run_id"], name="run_id"),
                configuration_sha256=_text(
                    identity_group.attrs["configuration_sha256"],
                    name="configuration_sha256",
                ),
                problem_sha256=_text(
                    identity_group.attrs["problem_sha256"],
                    name="problem_sha256",
                ),
                initial_state_sha256=_text(
                    identity_group.attrs["initial_state_sha256"],
                    name="initial_state_sha256",
                ),
                code_sha256=_text(
                    identity_group.attrs["code_sha256"],
                    name="code_sha256",
                ),
                environment_sha256=_text(
                    identity_group.attrs["environment_sha256"],
                    name="environment_sha256",
                ),
            )
            if stored_identity.identity_sha256 != stored_identity_hash:
                raise CheckpointCorruptionError(
                    "checkpoint canonical identity failed its SHA256 check."
                )
            if (
                stored_identity.identity_sha256
                != expected_identity.identity_sha256
            ):
                raise CheckpointMismatchError(
                    "checkpoint identity does not match the requested run "
                    f"({stored_identity.run_id!r} != {expected_identity.run_id!r})."
                )

            lesser_dataset = handle["sigma_pp/lesser"]
            greater_dataset = handle["sigma_pp/greater"]
            lesser = np.asarray(lesser_dataset[...])
            greater = np.asarray(greater_dataset[...])
            lesser_hash = _text(
                lesser_dataset.attrs["sha256"],
                name="sigma_pp/lesser.sha256",
            )
            greater_hash = _text(
                greater_dataset.attrs["sha256"],
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
            history_array = np.asarray(history_dataset[...])
            history_hash = _text(
                history_dataset.attrs["sha256"],
                name="iteration_history.sha256",
            )
            if _history_sha256(history_array) != history_hash:
                raise CheckpointCorruptionError(
                    "checkpoint iteration_history failed its SHA256 check."
                )
            history = _history_records(history_array)
            timings, timings_hash = _load_json_dataset(handle, "timings_json")
            metadata, metadata_hash = _load_json_dataset(handle, "metadata_json")

            manifest_payload = {
                "identity_sha256": stored_identity_hash,
                "status": status,
                "sigma_lesser_sha256": lesser_hash,
                "sigma_greater_sha256": greater_hash,
                "history_sha256": history_hash,
                "timings_sha256": timings_hash,
                "metadata_sha256": metadata_hash,
            }
            actual_manifest_hash = _sha256_bytes(
                _canonical_json(manifest_payload).encode("ascii")
            )
            if actual_manifest_hash != stored_manifest_hash:
                raise CheckpointCorruptionError(
                    "checkpoint manifest failed its SHA256 integrity check."
                )

        return SCBACheckpoint(
            identity=stored_identity,
            sigma_lesser=lesser,
            sigma_greater=greater,
            history=history,
            timings=timings,
            status=status,
            metadata=metadata,
        )
    except CheckpointError:
        raise
    except FileNotFoundError:
        raise
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as exc:
        raise CheckpointCorruptionError(
            f"SCBA checkpoint is missing, truncated, or invalid: {source}"
        ) from exc


def restore_scba_checkpoint_result(
    checkpoint: SCBACheckpoint,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> AnharmonicResult:
    """Recompute all derived fields from a validated fixed-point checkpoint.

    The stored mixed lesser/greater pair remains unchanged. Contacts, the
    causal retarded self-energy, Green functions, currents, and diagnostics
    are rebuilt with the current code. This differs from starting a new SCBA
    iteration, which could move the fixed point and break continuation hashes.
    """

    if not isinstance(checkpoint, SCBACheckpoint):
        raise TypeError("checkpoint must be an SCBACheckpoint.")
    restart = checkpoint.restart_spectrum(problem, config)

    # Local imports avoid a module cycle: solver intentionally does not depend
    # on the private persistence layer.
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
        restart.lesser,
        restart.greater,
        config,
    )
    currents = _current_report(problem, left, right, sigma_pp, green)
    return _finalize_result(
        problem,
        config,
        method="scba",
        green=green,
        sigma_pp=sigma_pp,
        left=left,
        right=right,
        currents=currents,
        history=checkpoint.history,
        converged=checkpoint.status == "converged",
        initialization="checkpoint_reconstruction",
        initialization_hash=checkpoint.identity.initial_state_sha256,
        predicted_memory_bytes=predicted,
    )


__all__ = [
    "CheckpointCorruptionError",
    "CheckpointDependencyError",
    "CheckpointError",
    "CheckpointMismatchError",
    "SCBACheckpoint",
    "SCBACheckpointIdentity",
    "load_scba_checkpoint",
    "make_scba_checkpoint_identity",
    "restore_scba_checkpoint_result",
    "write_scba_checkpoint",
]
