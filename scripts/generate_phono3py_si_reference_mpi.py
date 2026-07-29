#!/usr/bin/env python3
"""Generate the independent phono3py Si reference with task-level MPI.

This is an orchestration layer around
``generate_phono3py_si_reference.py``.  It deliberately imports no NEGFPy
module and performs no NEGFPy interaction or linewidth calculation.  Each
worker asks the pinned phono3py runtime to calculate one complete target
GRG point, including its full internal q-prime and branch sums.  Rank zero
then merges the immutable one-target files in canonical GRG order without
an MPI floating-point reduction.

The existing serial generator and its output schema remain unchanged.  The
MPI generator emits the same authenticated reference HDF5 and generation
records, plus one checksummed orchestration record.  A run directory holds
the immutable per-target artifacts and permits an exact restart with a
different MPI worker count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

try:
    from scripts import generate_phono3py_si_reference as serial
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import generate_phono3py_si_reference as serial  # type: ignore[no-redef]


MPI_REFERENCE_SCHEMA_VERSION = 2
MPI_MANIFEST_FILENAME = "reference-mpi-manifest.json"
MPI_ORCHESTRATION_FILENAME = "reference-mpi-orchestration.json"
MPI_EVIDENCE_ARCHIVE_FILENAME = "reference-mpi-evidence.tar"
_TASK_SCHEMA_NAME = "negfpy_phono3py_si_reference_mpi_task"
_TASK_SCHEMA_VERSION = 1
_TASK_PREFIX = "target-grg-"
_TASK_SUFFIX = ".hdf5"
_TASK_RECORD_SUFFIX = ".json"
_THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_SHA256_HEX = frozenset("0123456789abcdef")


class ReferenceMPIError(RuntimeError):
    """Raised when an MPI reference campaign cannot be authenticated."""


class _Communicator(Protocol):
    """The small mpi4py communicator surface used by the scheduler."""

    def Get_rank(self) -> int:
        ...

    def Get_size(self) -> int:
        ...

    def bcast(self, value: Any, root: int = 0) -> Any:
        ...

    def send(self, value: Any, dest: int, tag: int = 0) -> None:
        ...

    def recv(
        self,
        source: int = -1,
        tag: int = -1,
        status: Any | None = None,
    ) -> Any:
        ...

    def Barrier(self) -> None:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ReferenceMPIError(f"{name} must be a lowercase SHA256.")
    return value


def _regular_file(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReferenceMPIError(f"{name} must be a regular, non-symlink file.")
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_file_exclusive(source: Path, target: Path) -> None:
    """Publish a same-filesystem temporary file without replacement."""

    _regular_file(source, name="temporary artifact")
    if source.stat().st_size == 0:
        raise ReferenceMPIError("Temporary artifact is empty.")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    _fsync_directory(target.parent)


def _write_json_exclusive(path: Path, value: Any) -> None:
    serial._publish_bytes_exclusive(path, _pretty_json_bytes(value))


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    _regular_file(path, name=name)
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=serial._reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceMPIError(f"{name} is not strict UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ReferenceMPIError(f"{name} must contain a JSON object.")
    return value


def _git_output(arguments: Sequence[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _orchestration_code_identity() -> dict[str, Any]:
    script = Path(__file__).resolve()
    root_text = _git_output(
        ("rev-parse", "--show-toplevel"),
        cwd=script.parent,
    )
    result: dict[str, Any] = {
        "script_path": str(script),
        "script_sha256": _sha256_file(script),
    }
    if root_text is None:
        result.update(
            {
                "git_root": None,
                "git_commit": None,
                "tracked_worktree_status": None,
                "untracked_files_omitted": False,
            }
        )
        return result
    root = Path(root_text).resolve()
    result.update(
        {
            "git_root": str(root),
            "git_commit": _git_output(("rev-parse", "HEAD"), cwd=root),
            "git_tree": _git_output(("rev-parse", "HEAD^{tree}"), cwd=root),
            "tracked_worktree_status": _git_output(
                ("status", "--porcelain=v1", "--untracked-files=all"),
                cwd=root,
            ),
            "untracked_files_omitted": False,
        }
    )
    return result


def _require_threads_one() -> dict[str, str]:
    values = {name: os.environ.get(name) for name in _THREAD_NAMES}
    wrong = {name: value for name, value in values.items() if value != "1"}
    if wrong:
        raise ReferenceMPIError(
            "Every BLAS/OpenMP thread variable must equal '1': "
            f"{wrong!r}."
        )
    return {name: "1" for name in _THREAD_NAMES}


_PHYSICAL_CORE_BINDING_POLICY = (
    "one_physical_core_per_numerical_worker_disjoint_per_host"
)


def _physical_core_binding(rank: int) -> dict[str, Any]:
    """Capture fail-closed Linux affinity and physical-core topology evidence."""

    try:
        affinity = sorted(int(value) for value in os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise ReferenceMPIError(
            "Certified MPI numerical work requires readable CPU affinity."
        ) from exc
    if not affinity:
        raise ReferenceMPIError(
            "Certified MPI numerical work has an empty CPU affinity."
        )
    topology: list[dict[str, int]] = []
    for logical_cpu in affinity:
        topology_root = Path(
            f"/sys/devices/system/cpu/cpu{logical_cpu}/topology"
        )
        try:
            package_id = int(
                (topology_root / "physical_package_id").read_text(
                    encoding="ascii"
                )
            )
            core_id = int(
                (topology_root / "core_id").read_text(encoding="ascii")
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ReferenceMPIError(
                "Certified MPI numerical work requires readable physical CPU "
                f"topology for logical CPU {logical_cpu}."
            ) from exc
        if package_id < 0 or core_id < 0:
            raise ReferenceMPIError(
                "Physical CPU package/core identifiers must be nonnegative."
            )
        topology.append(
            {
                "logical_cpu": logical_cpu,
                "package_id": package_id,
                "core_id": core_id,
            }
        )
    physical_cores = {
        (item["package_id"], item["core_id"]) for item in topology
    }
    if len(physical_cores) != 1:
        raise ReferenceMPIError(
            "Each numerical MPI worker must be bound to exactly one physical "
            "core (SMT siblings on that core are allowed)."
        )
    package_id, core_id = next(iter(physical_cores))
    return {
        "rank": int(rank),
        "hostname": socket.gethostname(),
        "cpu_affinity": affinity,
        "cpu_affinity_topology": topology,
        "physical_core": {
            "package_id": package_id,
            "core_id": core_id,
        },
    }


def _normalize_worker_binding(
    value: Any,
    *,
    expected_rank: int,
    name: str,
) -> dict[str, Any]:
    """Validate and normalize one archived numerical-worker binding."""

    if not isinstance(value, Mapping):
        raise ReferenceMPIError(f"{name} must be a JSON object.")
    if set(value) != {
        "rank",
        "hostname",
        "cpu_affinity",
        "cpu_affinity_topology",
        "physical_core",
    }:
        raise ReferenceMPIError(f"{name} has an ambiguous schema.")
    rank = value.get("rank")
    hostname = value.get("hostname")
    affinity = value.get("cpu_affinity")
    topology = value.get("cpu_affinity_topology")
    physical = value.get("physical_core")
    if (
        rank != expected_rank
        or not isinstance(hostname, str)
        or not hostname
        or not isinstance(affinity, list)
        or not affinity
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0
            for cpu in affinity
        )
        or affinity != sorted(set(affinity))
        or not isinstance(topology, list)
        or len(topology) != len(affinity)
        or not isinstance(physical, Mapping)
        or set(physical) != {"package_id", "core_id"}
    ):
        raise ReferenceMPIError(f"{name} is invalid.")
    normalized_topology: list[dict[str, int]] = []
    for cpu, item in zip(affinity, topology, strict=True):
        if not isinstance(item, Mapping) or set(item) != {
            "logical_cpu",
            "package_id",
            "core_id",
        }:
            raise ReferenceMPIError(f"{name} topology is invalid.")
        logical_cpu = item.get("logical_cpu")
        package_id = item.get("package_id")
        core_id = item.get("core_id")
        if (
            logical_cpu != cpu
            or not isinstance(package_id, int)
            or isinstance(package_id, bool)
            or package_id < 0
            or not isinstance(core_id, int)
            or isinstance(core_id, bool)
            or core_id < 0
        ):
            raise ReferenceMPIError(f"{name} topology is invalid.")
        normalized_topology.append(
            {
                "logical_cpu": logical_cpu,
                "package_id": package_id,
                "core_id": core_id,
            }
        )
    physical_cores = {
        (item["package_id"], item["core_id"])
        for item in normalized_topology
    }
    if len(physical_cores) != 1:
        raise ReferenceMPIError(
            f"{name} is not bound to exactly one physical core."
        )
    package_id, core_id = next(iter(physical_cores))
    if dict(physical) != {
        "package_id": package_id,
        "core_id": core_id,
    }:
        raise ReferenceMPIError(f"{name} physical-core summary is invalid.")
    return {
        "rank": rank,
        "hostname": hostname,
        "cpu_affinity": list(affinity),
        "cpu_affinity_topology": normalized_topology,
        "physical_core": {
            "package_id": package_id,
            "core_id": core_id,
        },
    }


def _normalize_execution_worker_bindings(
    assignments: Sequence[Mapping[str, Any]],
    *,
    communicator_size: int,
) -> dict[str, Any]:
    """Require consistent, disjoint physical-core binding for MPI workers."""

    by_worker: dict[tuple[str, int], dict[str, Any]] = {}
    occupied: dict[tuple[str, int, int], int] = {}
    for ordinal, assignment in enumerate(assignments):
        status = assignment.get("status")
        rank = assignment.get("rank")
        binding = assignment.get("worker_binding")
        if status == "resumed":
            if rank is not None or binding is not None:
                raise ReferenceMPIError(
                    "Resumed assignments must not claim numerical-worker binding."
                )
            continue
        if status not in {"computed", "adopted"}:
            continue
        if communicator_size == 1:
            if binding is not None:
                raise ReferenceMPIError(
                    "Serial assignments must not claim certified MPI binding."
                )
            continue
        if not isinstance(rank, int) or isinstance(rank, bool) or rank == 0:
            raise ReferenceMPIError(
                "MPI numerical assignments require a nonzero worker rank."
            )
        normalized = _normalize_worker_binding(
            binding,
            expected_rank=rank,
            name=f"assignments[{ordinal}].worker_binding",
        )
        worker = (str(normalized["hostname"]), rank)
        previous = by_worker.setdefault(worker, normalized)
        if previous != normalized:
            raise ReferenceMPIError(
                "One MPI worker reported inconsistent affinity/topology."
            )
        physical = normalized["physical_core"]
        occupied_key = (
            str(normalized["hostname"]),
            int(physical["package_id"]),
            int(physical["core_id"]),
        )
        previous_rank = occupied.setdefault(occupied_key, rank)
        if previous_rank != rank:
            raise ReferenceMPIError(
                "Distinct MPI worker ranks share one physical core."
            )
    rank_bindings: list[dict[str, Any]] = []
    for key in sorted(by_worker, key=lambda item: (item[0], item[1])):
        binding = by_worker[key]
        task_ids = [
            f"target-grg-{int(assignment['grid_point_grg']):08d}"
            for assignment in assignments
            if assignment.get("status") in {"computed", "adopted"}
            and assignment.get("rank") == binding["rank"]
            and _mapping_hostname(assignment.get("worker_binding"))
            == binding["hostname"]
        ]
        rank_bindings.append(
            {
                "hostname": binding["hostname"],
                "rank": binding["rank"],
                "logical_cpus": list(binding["cpu_affinity"]),
                "physical_core": dict(binding["physical_core"]),
                "task_ids": task_ids,
            }
        )
    available = bool(rank_bindings)
    return {
        "schema": "negfpy_task_physical_core_bindings",
        "schema_version": 1,
        "available": available,
        "one_physical_core_per_assignment": True,
        "consistent_affinity_per_rank": True,
        "disjoint_physical_cores_across_ranks": True,
        "unavailable_task_ids": [],
        "rank_bindings": rank_bindings,
    }


def _mapping_hostname(value: Any) -> str | None:
    if isinstance(value, Mapping) and isinstance(value.get("hostname"), str):
        return str(value["hostname"])
    return None


def _file_identity(path: Path) -> dict[str, Any]:
    target = _regular_file(path, name=str(path))
    stat = target.stat()
    return {
        "name": target.name,
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(target),
    }


def _force_stage_identities(force_stage_dir: Path) -> dict[str, Any]:
    names = sorted(path.name for path in force_stage_dir.iterdir())
    if frozenset(names) != serial._FORCE_STAGE_FILENAMES:
        raise ReferenceMPIError(
            "force_stage_dir does not have the exact completed serial "
            "force-constant schema."
        )
    return {
        name: _file_identity(force_stage_dir / name)
        for name in names
    }


@dataclass(frozen=True)
class CampaignIdentity:
    """Immutable scientific and provenance identity of an MPI campaign."""

    payload: Mapping[str, Any]
    sha256: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CampaignIdentity":
        normalized = dict(payload)
        return cls(
            payload=normalized,
            sha256=_sha256_bytes(_canonical_json_bytes(normalized)),
        )


@dataclass(frozen=True)
class PreparedCampaign:
    """Rank-zero preflight result broadcast to all workers."""

    identity: CampaignIdentity
    config: serial.GeneratorConfig
    source_files: Mapping[str, Path]
    source_verification: Mapping[str, Any]
    environment_identity: Mapping[str, Any]
    force_stage_dir: Path
    run_dir: Path
    output_dir: Path


def _manifest_document(identity: CampaignIdentity) -> dict[str, Any]:
    return {
        "schema_name": "negfpy_phono3py_si_reference_mpi_manifest",
        "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
        "campaign_sha256": identity.sha256,
        "campaign": dict(identity.payload),
    }


def _initialize_run_directory(
    run_dir: Path,
    *,
    identity: CampaignIdentity,
) -> None:
    document = _manifest_document(identity)
    if run_dir.exists():
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ReferenceMPIError("run_dir must be a non-symlink directory.")
        manifest_path = run_dir / MPI_MANIFEST_FILENAME
        if not manifest_path.exists():
            if any(run_dir.iterdir()):
                raise ReferenceMPIError(
                    "Existing run_dir lacks the immutable campaign manifest."
                )
            _write_json_exclusive(manifest_path, document)
        elif _read_json_object(
            manifest_path,
            name=MPI_MANIFEST_FILENAME,
        ) != document:
            raise ReferenceMPIError(
                "Existing run_dir belongs to a different campaign."
            )
    else:
        run_dir.mkdir(parents=True)
        _write_json_exclusive(run_dir / MPI_MANIFEST_FILENAME, document)
    allowed = {
        MPI_MANIFEST_FILENAME,
        "tasks",
        "failures",
        "executions",
        "scratch",
    }
    unknown = {item.name for item in run_dir.iterdir()} - allowed
    if unknown:
        raise ReferenceMPIError(
            f"run_dir contains unknown entries: {sorted(unknown)!r}."
        )
    for name in ("tasks", "failures", "executions", "scratch"):
        path = run_dir / name
        if path.exists() and (path.is_symlink() or not path.is_dir()):
            raise ReferenceMPIError(f"run_dir/{name} has an invalid type.")
        path.mkdir(exist_ok=True)


def prepare_campaign(
    *,
    manifest_path: Path,
    source_dir: Path,
    force_stage_dir: Path,
    run_dir: Path,
    output_dir: Path,
    config: serial.GeneratorConfig,
    runtime: Any | None = None,
    commit_resolver: Callable[[Path], str] | None = None,
    require_clean_source: bool = True,
) -> PreparedCampaign:
    """Authenticate inputs and create or resume an immutable run graph."""

    if not isinstance(config, serial.GeneratorConfig):
        raise TypeError("config must be the serial GeneratorConfig type.")
    if config.force_constants_only or config.mesh is None:
        raise ValueError("MPI reference generation requires a reference config.")
    _require_threads_one()
    if output_dir.exists():
        raise FileExistsError(
            "output_dir already exists; final artifacts are never overwritten."
        )
    manifest = serial.load_input_manifest(manifest_path)
    source_files, source_verification = serial._verify_source(
        manifest,
        source_dir,
        commit_resolver=commit_resolver,
    )
    active_runtime = runtime or serial._Phono3pyRuntime()
    versions = dict(
        active_runtime.verify_versions(dict(manifest.expected_versions))
    )
    if versions != dict(manifest.expected_versions):
        raise ReferenceMPIError("Runtime returned unexpected pinned versions.")
    environment = serial._environment_identity(
        runtime_versions=versions,
        backend=config.backend,
    )
    serial._validate_force_stage(
        force_stage_dir,
        normalized_manifest=manifest.normalized_dict(),
        source_verification=source_verification,
        force_settings=config.force_constants_settings(),
        environment_identity=environment,
    )
    serial_code = environment.get("code")
    mpi_code = _orchestration_code_identity()
    for label, code in (("serial generator", serial_code), ("MPI driver", mpi_code)):
        if not isinstance(code, Mapping):
            raise ReferenceMPIError(f"{label} has no code identity.")
        if require_clean_source and (
            code.get("tracked_worktree_status") != ""
            or code.get("untracked_files_omitted") is not False
        ):
            raise ReferenceMPIError(
                f"{label} requires an exactly clean worktree including "
                "untracked files."
            )
        if require_clean_source and not code.get("git_commit"):
            raise ReferenceMPIError(
                f"{label} requires an exact Git commit."
            )
    force_identities = _force_stage_identities(force_stage_dir)
    payload = {
        "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
        "algorithm": (
            "one_complete_phono3py_target_q_per_worker_canonical_rank0_merge"
        ),
        "floating_point_mpi_reductions": False,
        "serial_generator": dict(serial_code),
        "mpi_driver": mpi_code,
        "source_manifest": manifest.normalized_dict(),
        "source_verification": source_verification,
        "force_stage": force_identities,
        "reference_settings": config.reference_settings(),
        "runtime_environment": environment,
        "thread_environment": _require_threads_one(),
        "task_grid_points_grg": list(config.grid_points),
    }
    identity = CampaignIdentity.from_payload(payload)
    _initialize_run_directory(run_dir, identity=identity)
    return PreparedCampaign(
        identity=identity,
        config=config,
        source_files=source_files,
        source_verification=source_verification,
        environment_identity=environment,
        force_stage_dir=force_stage_dir,
        run_dir=run_dir,
        output_dir=output_dir,
    )


def _task_stem(grid_point: int) -> str:
    return f"{_TASK_PREFIX}{grid_point:08d}"


def _task_paths(run_dir: Path, grid_point: int) -> tuple[Path, Path, Path]:
    stem = _task_stem(grid_point)
    return (
        run_dir / "tasks" / f"{stem}{_TASK_SUFFIX}",
        run_dir / "tasks" / f"{stem}{_TASK_RECORD_SUFFIX}",
        run_dir / "failures" / f"{stem}{_TASK_RECORD_SUFFIX}",
    )


def _h5_attr_value(value: Any) -> Any:
    """Normalize HDF5 attribute scalars into canonical JSON values."""

    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_h5_attr_value(item) for item in value]
    if isinstance(value, list):
        return [_h5_attr_value(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _update_attrs_digest(digest: Any, attrs: Any) -> None:
    payload = {
        str(name): _h5_attr_value(attrs[name])
        for name in sorted(attrs.keys())
        if not str(name).startswith("mpi_reference_")
    }
    digest.update(_canonical_json_bytes(payload))


def _update_dataset_digest(digest: Any, dataset: Any) -> None:
    import numpy as np

    digest.update(str(dataset.name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(dataset.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(int(value)) for value in dataset.shape).encode("ascii")
    )
    digest.update(b"\0")
    _update_attrs_digest(digest, dataset.attrs)
    if dataset.ndim == 0:
        digest.update(np.asarray(dataset[()]).tobytes(order="C"))
        return
    if dataset.shape[0] == 0:
        return
    for index in range(int(dataset.shape[0])):
        digest.update(
            np.ascontiguousarray(dataset[index]).tobytes(order="C")
        )


def _update_node_digest(digest: Any, node: Any) -> None:
    import h5py

    digest.update(str(node.name).encode("utf-8"))
    digest.update(b"\0")
    _update_attrs_digest(digest, node.attrs)
    if isinstance(node, h5py.Dataset):
        _update_dataset_digest(digest, node)
        return
    for name in sorted(node.keys()):
        _update_node_digest(digest, node[name])


def _common_reference_digest(h5: Any) -> str:
    """Hash every field that must be identical in all one-q artifacts."""

    digest = hashlib.sha256()
    _update_attrs_digest(digest, h5.attrs)
    for name in ("crystal", "grid", "fc2_random_q"):
        _update_node_digest(digest, h5[name])
    settings = h5["settings"]
    _update_attrs_digest(digest, settings.attrs)
    for name in ("mesh", "band_indices", "temperatures", "sigmas"):
        _update_node_digest(digest, settings[name])
    phonons = h5["phonons"]
    _update_attrs_digest(digest, phonons.attrs)
    for name in ("frequencies", "eigenvectors", "bz_grid_addresses"):
        _update_node_digest(digest, phonons[name])
    gamma = h5["gamma"]
    _update_attrs_digest(digest, gamma.attrs)
    _update_attrs_digest(digest, gamma["values"].attrs)
    return digest.hexdigest()


def _validate_task_hdf5(
    path: Path,
    *,
    campaign_sha256: str,
    config: serial.GeneratorConfig,
    grid_point: int,
) -> dict[str, Any]:
    import h5py
    import numpy as np

    _regular_file(path, name="task HDF5")
    with h5py.File(path, "r") as h5:
        if (
            h5.attrs.get("schema_name")
            != "negfpy_phono3py_si_reference"
            or int(h5.attrs.get("schema_version", -1))
            != serial.REFERENCE_SCHEMA_VERSION
            or h5.attrs.get("mpi_reference_task_schema") != _TASK_SCHEMA_NAME
            or int(h5.attrs.get("mpi_reference_task_schema_version", -1))
            != _TASK_SCHEMA_VERSION
            or h5.attrs.get("mpi_reference_campaign_sha256")
            != campaign_sha256
            or int(h5.attrs.get("mpi_reference_grid_point_grg", -1))
            != grid_point
        ):
            raise ReferenceMPIError(
                f"Task HDF5 for GRG {grid_point} has invalid provenance."
            )
        settings = h5["settings"]
        mesh = tuple(int(value) for value in settings["mesh"][...])
        grg = np.asarray(settings["grid_points_grg"][...], dtype=np.int64)
        bzg = np.asarray(settings["grid_points_bzg"][...], dtype=np.int64)
        bands = np.asarray(settings["band_indices"][...], dtype=np.int64)
        temperatures = np.asarray(settings["temperatures"][...], dtype=float)
        sigmas = np.asarray(settings["sigmas"][...], dtype=float)
        expected_bands = (
            None
            if config.band_indices is None
            else np.asarray(config.band_indices, dtype=np.int64)
        )
        if (
            mesh != tuple(config.mesh or ())
            or grg.tolist() != [grid_point]
            or bzg.shape != (1,)
            or (
                expected_bands is not None
                and not np.array_equal(bands, expected_bands)
            )
            or not np.array_equal(
                temperatures,
                np.asarray(config.temperatures, dtype=float),
            )
            or not np.array_equal(sigmas, np.asarray([config.sigma], dtype=float))
        ):
            raise ReferenceMPIError(
                f"Task HDF5 for GRG {grid_point} conflicts with campaign settings."
            )
        group_name = f"grg_{grid_point:08d}"
        if set(h5["target_q"].keys()) != {group_name}:
            raise ReferenceMPIError(
                f"Task HDF5 for GRG {grid_point} has an ambiguous target group."
            )
        target = h5["target_q"][group_name]
        if (
            int(target.attrs.get("grid_point_grg", -1)) != grid_point
            or int(target.attrs.get("grid_point_bzg", -1)) != int(bzg[0])
            or int(target.attrs.get("target_ordinal", -1)) != 0
        ):
            raise ReferenceMPIError(
                f"Task target metadata is invalid for GRG {grid_point}."
            )
        gamma_shape = tuple(int(value) for value in h5["gamma/values"].shape)
        phonon_shape = tuple(
            int(value) for value in h5["phonons/frequencies"].shape
        )
        target_frequency_shape = tuple(
            int(value)
            for value in h5["phonons/target_band_frequencies"].shape
        )
        if (
            len(gamma_shape) != 4
            or gamma_shape[2] != 1
            or len(phonon_shape) != 2
            or phonon_shape[0] != math.prod(mesh)
            or phonon_shape[1] <= 0
            or np.any(bands < 0)
            or np.any(bands >= phonon_shape[1])
            or target_frequency_shape != (1, len(bands))
        ):
            raise ReferenceMPIError(
                f"Task target-axis shape is invalid for GRG {grid_point}."
            )
        runtime_text = h5.attrs.get("mpi_reference_runtime_json")
        if not isinstance(runtime_text, str):
            raise ReferenceMPIError("Task HDF5 lacks runtime metadata.")
        try:
            runtime = json.loads(
                runtime_text,
                object_pairs_hook=serial._reject_duplicate_keys,
            )
        except json.JSONDecodeError as exc:
            raise ReferenceMPIError("Task runtime metadata is invalid.") from exc
        if not isinstance(runtime, dict):
            raise ReferenceMPIError("Task runtime metadata must be an object.")
        return {
            "grid_point_grg": grid_point,
            "grid_point_bzg": int(bzg[0]),
            "bands": [int(value) for value in bands],
            "n_qpoints": int(phonon_shape[0]),
            "n_bands": int(phonon_shape[1]),
            "gamma_shape": list(gamma_shape),
            "target_frequency_shape": list(target_frequency_shape),
            "common_reference_sha256": _common_reference_digest(h5),
            "runtime": runtime,
        }


def _task_record(
    *,
    campaign_sha256: str,
    grid_point: int,
    hdf5_path: Path,
    validation: Mapping[str, Any],
    rank: int | None,
    hostname: str,
    started_utc: str | None,
    completed_utc: str,
    wall_seconds: float | None,
    disposition: str,
) -> dict[str, Any]:
    return {
        "schema_name": "negfpy_phono3py_si_reference_mpi_task_record",
        "schema_version": _TASK_SCHEMA_VERSION,
        "campaign_sha256": campaign_sha256,
        "grid_point_grg": grid_point,
        "status": "complete",
        "disposition": disposition,
        "artifact": _file_identity(hdf5_path),
        "scientific": {
            key: validation[key]
            for key in (
                "grid_point_grg",
                "grid_point_bzg",
                "bands",
                "n_qpoints",
                "n_bands",
                "gamma_shape",
                "target_frequency_shape",
                "common_reference_sha256",
            )
        },
        "execution": {
            "rank": rank,
            "hostname": hostname,
            "started_at_utc": started_utc,
            "completed_at_utc": completed_utc,
            "wall_seconds": wall_seconds,
            "thread_environment": _require_threads_one(),
        },
        "runtime": validation["runtime"],
    }


def _validate_task_record(
    record_path: Path,
    hdf5_path: Path,
    *,
    campaign_sha256: str,
    config: serial.GeneratorConfig,
    grid_point: int,
) -> dict[str, Any]:
    record = _read_json_object(record_path, name="task record")
    if (
        record.get("schema_name")
        != "negfpy_phono3py_si_reference_mpi_task_record"
        or record.get("schema_version") != _TASK_SCHEMA_VERSION
        or record.get("campaign_sha256") != campaign_sha256
        or record.get("grid_point_grg") != grid_point
        or record.get("status") != "complete"
    ):
        raise ReferenceMPIError(
            f"Task record for GRG {grid_point} has invalid identity."
        )
    artifact = record.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ReferenceMPIError("Task record has no artifact identity.")
    actual = _file_identity(hdf5_path)
    if dict(artifact) != actual:
        raise ReferenceMPIError(
            f"Task artifact for GRG {grid_point} changed after publication."
        )
    validation = _validate_task_hdf5(
        hdf5_path,
        campaign_sha256=campaign_sha256,
        config=config,
        grid_point=grid_point,
    )
    declared = record.get("scientific")
    expected = {
        key: validation[key]
        for key in (
            "grid_point_grg",
            "grid_point_bzg",
            "bands",
            "n_qpoints",
            "n_bands",
            "gamma_shape",
            "target_frequency_shape",
            "common_reference_sha256",
        )
    }
    if declared != expected or record.get("runtime") != validation["runtime"]:
        raise ReferenceMPIError(
            f"Task record for GRG {grid_point} conflicts with HDF5 fields."
        )
    return record


def _adopt_or_validate_task(
    campaign: PreparedCampaign,
    grid_point: int,
) -> dict[str, Any] | None:
    hdf5_path, record_path, failure_path = _task_paths(
        campaign.run_dir,
        grid_point,
    )
    if failure_path.exists():
        failure = _read_json_object(failure_path, name="task failure")
        if (
            failure.get("campaign_sha256") != campaign.identity.sha256
            or failure.get("grid_point_grg") != grid_point
        ):
            raise ReferenceMPIError(
                f"Failure record for GRG {grid_point} has invalid identity."
            )
        raise ReferenceMPIError(
            f"GRG {grid_point} has an archived terminal failure."
        )
    if record_path.exists():
        if not hdf5_path.exists():
            raise ReferenceMPIError(
                f"Task record for GRG {grid_point} has no artifact."
            )
        return _validate_task_record(
            record_path,
            hdf5_path,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=grid_point,
        )
    if not hdf5_path.exists():
        return None
    validation = _validate_task_hdf5(
        hdf5_path,
        campaign_sha256=campaign.identity.sha256,
        config=campaign.config,
        grid_point=grid_point,
    )
    record = _task_record(
        campaign_sha256=campaign.identity.sha256,
        grid_point=grid_point,
        hdf5_path=hdf5_path,
        validation=validation,
        rank=None,
        hostname=socket.gethostname(),
        started_utc=None,
        completed_utc=_utc_now(),
        wall_seconds=None,
        disposition="adopted_after_interrupted_record_publication",
    )
    _write_json_exclusive(record_path, record)
    return record


def _publish_failure(
    campaign: PreparedCampaign,
    grid_point: int,
    *,
    rank: int,
    started_utc: str,
    started: float,
    error: BaseException,
) -> None:
    _, _, failure_path = _task_paths(campaign.run_dir, grid_point)
    document = {
        "schema_name": "negfpy_phono3py_si_reference_mpi_task_failure",
        "schema_version": _TASK_SCHEMA_VERSION,
        "campaign_sha256": campaign.identity.sha256,
        "grid_point_grg": grid_point,
        "status": "failed",
        "rank": rank,
        "hostname": socket.gethostname(),
        "started_at_utc": started_utc,
        "failed_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "thread_environment": _require_threads_one(),
    }
    if failure_path.exists():
        raise ReferenceMPIError(
            f"Failure record already exists for GRG {grid_point}."
        )
    _write_json_exclusive(failure_path, document)


def _generate_one_task(
    campaign: PreparedCampaign,
    grid_point: int,
    *,
    rank: int,
    runtime_factory: Callable[[], Any],
) -> dict[str, Any]:
    existing = _adopt_or_validate_task(campaign, grid_point)
    if existing is not None:
        return {"grid_point_grg": grid_point, "status": "resumed"}
    hdf5_path, record_path, _ = _task_paths(campaign.run_dir, grid_point)
    started_utc = _utc_now()
    started = time.perf_counter()
    scratch = Path(
        tempfile.mkdtemp(
            prefix=f".{_task_stem(grid_point)}-rank{rank}-",
            dir=campaign.run_dir / "scratch",
        )
    )
    try:
        for name in ("fc2.compact.hdf5", "fc3.compact.hdf5"):
            source = _regular_file(
                campaign.force_stage_dir / name,
                name=f"force-stage {name}",
            )
            os.link(source, scratch / name)
        config = serial.GeneratorConfig(
            mesh=campaign.config.mesh,
            grid_points=(grid_point,),
            band_indices=campaign.config.band_indices,
            temperatures=campaign.config.temperatures,
            sigma=campaign.config.sigma,
            sigma_cutoff=campaign.config.sigma_cutoff,
            backend=campaign.config.backend,
            make_r0_average=campaign.config.make_r0_average,
            symmetrize_fc=campaign.config.symmetrize_fc,
            symmetrize_fc3q=campaign.config.symmetrize_fc3q,
        )
        runtime = runtime_factory()
        versions = dict(
            runtime.verify_versions(
                dict(
                    campaign.identity.payload["source_manifest"]["versions"]
                )
            )
        )
        if versions != dict(
            campaign.identity.payload["source_manifest"]["versions"]
        ):
            raise ReferenceMPIError("Worker runtime version drifted.")
        metadata = dict(
            runtime.generate_reference(
                output_dir=scratch,
                source_files=campaign.source_files,
                config=config,
            )
        )
        temporary_hdf5 = scratch / serial.REFERENCE_FILENAME
        import h5py

        with h5py.File(temporary_hdf5, "r+") as h5:
            h5.attrs["mpi_reference_task_schema"] = _TASK_SCHEMA_NAME
            h5.attrs[
                "mpi_reference_task_schema_version"
            ] = _TASK_SCHEMA_VERSION
            h5.attrs[
                "mpi_reference_campaign_sha256"
            ] = campaign.identity.sha256
            h5.attrs["mpi_reference_grid_point_grg"] = grid_point
            h5.attrs["mpi_reference_runtime_json"] = (
                _canonical_json_bytes(metadata).decode("ascii").rstrip("\n")
            )
            h5.flush()
        descriptor = os.open(temporary_hdf5, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        validation = _validate_task_hdf5(
            temporary_hdf5,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=grid_point,
        )
        _publish_file_exclusive(temporary_hdf5, hdf5_path)
        record = _task_record(
            campaign_sha256=campaign.identity.sha256,
            grid_point=grid_point,
            hdf5_path=hdf5_path,
            validation=validation,
            rank=rank,
            hostname=socket.gethostname(),
            started_utc=started_utc,
            completed_utc=_utc_now(),
            wall_seconds=time.perf_counter() - started,
            disposition="computed",
        )
        _write_json_exclusive(record_path, record)
        return {"grid_point_grg": grid_point, "status": "computed"}
    except Exception as exc:
        if hdf5_path.exists() and not record_path.exists():
            # A valid artifact published just before interruption is adoptable.
            try:
                _adopt_or_validate_task(campaign, grid_point)
            except Exception:
                _publish_failure(
                    campaign,
                    grid_point,
                    rank=rank,
                    started_utc=started_utc,
                    started=started,
                    error=exc,
                )
                raise
            return {"grid_point_grg": grid_point, "status": "adopted"}
        _publish_failure(
            campaign,
            grid_point,
            rank=rank,
            started_utc=started_utc,
            started=started,
            error=exc,
        )
        raise
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _copy_attrs(source: Any, destination: Any, *, omit_mpi: bool = False) -> None:
    for name in source.attrs:
        if omit_mpi and str(name).startswith("mpi_reference_"):
            continue
        destination.attrs[name] = source.attrs[name]


def _merge_frequency_points(
    task_paths: Sequence[Path],
    destination: Any,
) -> None:
    import h5py
    import numpy as np

    present: list[bool] = []
    arrays: list[Any] = []
    first_dataset_attrs: dict[str, Any] | None = None
    for path in task_paths:
        with h5py.File(path, "r") as source:
            has_values = "frequency_points" in source["gamma"]
            present.append(has_values)
            if has_values:
                dataset = source["gamma/frequency_points"]
                arrays.append(np.asarray(dataset[...]))
                if first_dataset_attrs is None:
                    first_dataset_attrs = {
                        str(name): dataset.attrs[name] for name in dataset.attrs
                    }
    if not any(present):
        return
    if not all(present):
        raise ReferenceMPIError(
            "Task gamma frequency_points presence is inconsistent."
        )
    first = arrays[0]
    # Pinned phono3py may return either a target-leading array or a common
    # target-band vector.  Preserve a common value; otherwise concatenate the
    # unique target axis exactly as emitted.
    if all(np.array_equal(array, first) for array in arrays[1:]):
        merged = first
    elif first.ndim >= 1 and first.shape[0] == 1:
        merged = np.concatenate(arrays, axis=0)
    elif first.ndim >= 3 and first.shape[-2] == 1:
        merged = np.concatenate(arrays, axis=-2)
    else:
        raise ReferenceMPIError(
            "Unsupported target-dependent gamma frequency_points layout."
        )
    dataset = destination.create_dataset(
        "frequency_points",
        data=merged,
        compression="gzip" if merged.size >= 32 else None,
        shuffle=bool(merged.size >= 32),
    )
    assert first_dataset_attrs is not None
    for name, value in first_dataset_attrs.items():
        dataset.attrs[name] = value


def merge_task_references(
    campaign: PreparedCampaign,
    destination: Path,
) -> dict[str, Any]:
    """Merge authenticated one-q HDF5 files into the serial reference schema."""

    import h5py
    import numpy as np

    if destination.exists():
        raise FileExistsError("Merged reference destination already exists.")
    task_records: list[dict[str, Any]] = []
    task_paths: list[Path] = []
    for grid_point in campaign.config.grid_points:
        hdf5_path, record_path, _ = _task_paths(campaign.run_dir, grid_point)
        task_records.append(
            _validate_task_record(
                record_path,
                hdf5_path,
                campaign_sha256=campaign.identity.sha256,
                config=campaign.config,
                grid_point=grid_point,
            )
        )
        task_paths.append(hdf5_path)
    common_hashes = {
        record["scientific"]["common_reference_sha256"]
        for record in task_records
    }
    n_qpoints = {record["scientific"]["n_qpoints"] for record in task_records}
    n_bands = {record["scientific"]["n_bands"] for record in task_records}
    if len(common_hashes) != 1:
        raise ReferenceMPIError(
            "One-q artifacts disagree in common phonon/crystal/grid fields."
        )
    if n_qpoints != {math.prod(campaign.config.mesh or ())} or len(n_bands) != 1:
        raise ReferenceMPIError(
            "One-q artifacts disagree in full-mesh phonon dimensions."
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp.hdf5",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        with h5py.File(task_paths[0], "r") as first, h5py.File(
            temporary,
            "w",
        ) as target:
            _copy_attrs(first, target, omit_mpi=True)
            for name in ("crystal", "grid", "fc2_random_q"):
                first.copy(name, target)

            settings = target.create_group("settings")
            _copy_attrs(first["settings"], settings)
            for name in ("mesh", "band_indices", "temperatures", "sigmas"):
                first.copy(first[f"settings/{name}"], settings, name=name)
            grg_values = np.asarray(campaign.config.grid_points, dtype=np.int64)
            bzg_values = np.asarray(
                [
                    int(record["scientific"]["grid_point_bzg"])
                    for record in task_records
                ],
                dtype=np.int64,
            )
            settings.create_dataset("grid_points_grg", data=grg_values)
            settings.create_dataset("grid_points_bzg", data=bzg_values)

            phonons = target.create_group("phonons")
            _copy_attrs(first["phonons"], phonons)
            for name in ("frequencies", "eigenvectors", "bz_grid_addresses"):
                first.copy(first[f"phonons/{name}"], phonons, name=name)
            first_target_frequencies = np.asarray(
                first["phonons/target_band_frequencies"][...]
            )
            target_frequencies = np.empty(
                (
                    len(task_paths),
                    first_target_frequencies.shape[1],
                ),
                dtype=first_target_frequencies.dtype,
            )
            first_gamma = np.asarray(first["gamma/values"][...])
            gamma_values = np.empty(
                (
                    first_gamma.shape[0],
                    first_gamma.shape[1],
                    len(task_paths),
                    first_gamma.shape[3],
                ),
                dtype=first_gamma.dtype,
            )
            targets = target.create_group("target_q")
            for ordinal, (grid_point, path) in enumerate(
                zip(
                    campaign.config.grid_points,
                    task_paths,
                    strict=True,
                )
            ):
                with h5py.File(path, "r") as source:
                    target_slice = np.asarray(
                        source["phonons/target_band_frequencies"][...]
                    )
                    gamma_slice = np.asarray(source["gamma/values"][...])
                    if (
                        target_slice.shape
                        != (1, target_frequencies.shape[1])
                        or gamma_slice.shape
                        != (
                            gamma_values.shape[0],
                            gamma_values.shape[1],
                            1,
                            gamma_values.shape[3],
                        )
                    ):
                        raise ReferenceMPIError(
                            f"Target-axis shape drift at GRG {grid_point}."
                        )
                    target_frequencies[ordinal] = target_slice[0]
                    gamma_values[:, :, ordinal, :] = gamma_slice[:, :, 0, :]
                    group_name = f"grg_{grid_point:08d}"
                    source.copy(
                        source[f"target_q/{group_name}"],
                        targets,
                        name=group_name,
                    )
                    targets[group_name].attrs["target_ordinal"] = ordinal
            target_frequency_dataset = phonons.create_dataset(
                "target_band_frequencies",
                data=target_frequencies,
                compression=(
                    "gzip" if target_frequencies.size >= 32 else None
                ),
                shuffle=bool(target_frequencies.size >= 32),
            )
            _copy_attrs(
                first["phonons/target_band_frequencies"],
                target_frequency_dataset,
            )

            gamma = target.create_group("gamma")
            _copy_attrs(first["gamma"], gamma)
            gamma_dataset = gamma.create_dataset(
                "values",
                data=gamma_values,
                compression="gzip" if gamma_values.size >= 32 else None,
                shuffle=bool(gamma_values.size >= 32),
            )
            _copy_attrs(first["gamma/values"], gamma_dataset)
            _merge_frequency_points(task_paths, gamma)
            target.flush()
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _publish_file_exclusive(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "common_reference_sha256": next(iter(common_hashes)),
        "gamma_shape": list(gamma_values.shape),
        "target_band_frequency_shape": list(target_frequencies.shape),
        "grid_points_grg": list(campaign.config.grid_points),
        "grid_points_bzg": bzg_values.tolist(),
        "effective_band_indices": task_records[0]["scientific"]["bands"],
        "n_qpoints": next(iter(n_qpoints)),
        "n_bands": next(iter(n_bands)),
        "tasks": [
            {
                "grid_point_grg": grid_point,
                "artifact": record["artifact"],
                "record_sha256": _sha256_file(record_path),
                "runtime": record["runtime"],
            }
            for grid_point, record, (_, record_path, _) in zip(
                campaign.config.grid_points,
                task_records,
                (
                    _task_paths(campaign.run_dir, point)
                    for point in campaign.config.grid_points
                ),
                strict=True,
            )
        ],
    }


def _copy_force_stage(source: Path, destination: Path) -> None:
    destination.mkdir()
    for name in sorted(serial._FORCE_STAGE_FILENAMES):
        shutil.copyfile(source / name, destination / name)
        descriptor = os.open(destination / name, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(destination)


def _archive_member_identity(path: Path, member: str) -> dict[str, Any]:
    """Return the immutable identity used for one regular tar member."""

    source = _regular_file(path, name=f"MPI evidence source {member}")
    if (
        not member
        or member.startswith("/")
        or "\\" in member
        or any(part in {"", ".", ".."} for part in Path(member).parts)
    ):
        raise ReferenceMPIError(f"Unsafe MPI evidence member path: {member!r}.")
    return {
        "archive_member": member,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _validate_execution_record(
    campaign: PreparedCampaign,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable execution before it enters release evidence."""

    normalized = dict(record)
    execution_id = normalized.get("execution_id")
    if not isinstance(execution_id, str):
        raise ReferenceMPIError("MPI execution record lacks execution_id.")
    _execution_path(campaign.run_dir, execution_id)
    assignments = normalized.get("assignments")
    if not isinstance(assignments, list):
        raise ReferenceMPIError("MPI execution assignments must be an array.")
    expected_points = list(campaign.config.grid_points)
    points = [
        item.get("grid_point_grg")
        for item in assignments
        if isinstance(item, Mapping)
    ]
    statuses = [
        item.get("status") for item in assignments if isinstance(item, Mapping)
    ]
    resumed = [
        point
        for point, status in zip(points, statuses, strict=True)
        if status == "resumed"
    ]
    computed = [
        point
        for point, status in zip(points, statuses, strict=True)
        if status in {"computed", "adopted"}
    ]
    size = normalized.get("communicator_size")
    threads = normalized.get("thread_environment")
    worker_binding_evidence = (
        _normalize_execution_worker_bindings(
            assignments,
            communicator_size=size,
        )
        if isinstance(size, int) and not isinstance(size, bool) and size >= 1
        else []
    )
    if (
        normalized.get("schema_name")
        != "negfpy_phono3py_si_reference_mpi_execution"
        or normalized.get("schema_version") != MPI_REFERENCE_SCHEMA_VERSION
        or normalized.get("campaign_sha256") != campaign.identity.sha256
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
        or normalized.get("coordinator_rank") != 0
        or normalized.get("rank_zero_numerical_tasks") is not (size == 1)
        or normalized.get("floating_point_mpi_reductions") is not False
        or normalized.get("worker_binding_policy")
        != _PHYSICAL_CORE_BINDING_POLICY
        or normalized.get("worker_binding_evidence")
        != worker_binding_evidence
        or normalized.get("canonical_assignment_order")
        != "strictly_increasing_GRG"
        or normalized.get("task_grid_points_grg") != expected_points
        or normalized.get("assignment_grid_points_grg") != expected_points
        or points != expected_points
        or any(status not in {"computed", "adopted", "resumed"} for status in statuses)
        or normalized.get("resumed_grid_points_grg") != resumed
        or normalized.get("computed_or_adopted_grid_points_grg") != computed
        or normalized.get("exact_checkpoint_resume") is not bool(resumed)
        or normalized.get("complete") is not True
        or not isinstance(threads, Mapping)
        or any(threads.get(name) != "1" for name in _THREAD_NAMES)
    ):
        raise ReferenceMPIError(
            f"Execution {execution_id!r} is incomplete or inconsistent."
        )
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ReferenceMPIError("MPI execution assignment is not an object.")
        rank = item.get("rank")
        if rank is not None and (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
            or rank >= size
        ):
            raise ReferenceMPIError(
                f"Execution {execution_id!r} has an invalid worker rank."
            )
        if size > 1 and item.get("status") != "resumed" and rank == 0:
            raise ReferenceMPIError(
                f"Execution {execution_id!r} assigned numerical work to rank zero."
            )
        point = item.get("grid_point_grg")
        if item.get("status") in {"computed", "adopted"}:
            if not isinstance(point, int) or isinstance(point, bool):
                raise ReferenceMPIError(
                    f"Execution {execution_id!r} has an invalid task point."
                )
            hdf5_path, record_path, _ = _task_paths(
                campaign.run_dir,
                point,
            )
            task_record = _validate_task_record(
                record_path,
                hdf5_path,
                campaign_sha256=campaign.identity.sha256,
                config=campaign.config,
                grid_point=point,
            )
            if item.get("artifact") != task_record["artifact"]:
                raise ReferenceMPIError(
                    f"Execution {execution_id!r} assignment artifact differs "
                    "from its immutable task."
                )
        elif item.get("status") == "resumed" and "artifact" in item:
            raise ReferenceMPIError(
                f"Execution {execution_id!r} resumed assignment claims a fresh "
                "artifact."
            )
    return normalized


def _write_evidence_archive(
    campaign: PreparedCampaign,
    destination: Path,
    *,
    execution_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish a deterministic, self-contained archive of MPI task evidence."""

    if destination.exists():
        raise FileExistsError("MPI evidence archive already exists.")
    sources: list[tuple[Path, dict[str, Any]]] = []
    manifest_path = campaign.run_dir / MPI_MANIFEST_FILENAME
    campaign_manifest = _archive_member_identity(
        manifest_path,
        MPI_MANIFEST_FILENAME,
    )
    sources.append((manifest_path, campaign_manifest))

    tasks: list[dict[str, Any]] = []
    for grid_point in campaign.config.grid_points:
        hdf5_path, record_path, _ = _task_paths(campaign.run_dir, grid_point)
        hdf5 = _archive_member_identity(
            hdf5_path,
            f"tasks/{hdf5_path.name}",
        )
        record = _archive_member_identity(
            record_path,
            f"tasks/{record_path.name}",
        )
        sources.extend(((hdf5_path, hdf5), (record_path, record)))
        tasks.append(
            {
                "grid_point_grg": grid_point,
                "hdf5": hdf5,
                "record": record,
            }
        )

    executions: list[dict[str, Any]] = []
    normalized_executions = sorted(
        (
            _validate_execution_record(campaign, value)
            for value in execution_records
        ),
        key=lambda item: str(item["execution_id"]),
    )
    freshly_computed = {
        int(assignment["grid_point_grg"])
        for execution in normalized_executions
        for assignment in execution["assignments"]
        if assignment.get("status") == "computed"
    }
    if freshly_computed != set(campaign.config.grid_points):
        raise ReferenceMPIError(
            "Release evidence requires one archived fresh MPI-computed "
            "assignment for every target q-point; adopted or resumed tasks do "
            "not establish numerical origin."
        )
    for execution in normalized_executions:
        execution_id = str(execution["execution_id"])
        path = _execution_path(campaign.run_dir, execution_id)
        on_disk = _read_json_object(
            path,
            name=f"MPI execution {execution_id}",
        )
        if on_disk != execution:
            raise ReferenceMPIError(
                f"Execution {execution_id!r} differs from its immutable file."
            )
        evidence = _archive_member_identity(
            path,
            f"executions/{path.name}",
        )
        sources.append((path, evidence))
        executions.append(
            {
                "execution_id": execution_id,
                "record": evidence,
            }
        )
    if not executions:
        raise ReferenceMPIError("MPI reference evidence has no execution record.")

    with tarfile.open(
        destination,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for source, identity in sources:
            info = tarfile.TarInfo(str(identity["archive_member"]))
            info.size = int(identity["size_bytes"])
            info.mode = 0o444
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with source.open("rb") as stream:
                archive.addfile(info, stream)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    resumed_ids = [
        str(record["execution_id"])
        for record in normalized_executions
        if record["resumed_grid_points_grg"]
    ]
    resumed_counts = {
        str(record["execution_id"]): len(record["resumed_grid_points_grg"])
        for record in normalized_executions
    }
    return {
        "archive": _file_identity(destination),
        "campaign_manifest": campaign_manifest,
        "tasks": tasks,
        "executions": executions,
        "restart": {
            "exact_checkpoint_resume_supported": True,
            "execution_count": len(executions),
            "execution_ids": [
                str(record["execution_id"])
                for record in normalized_executions
            ],
            "executions_with_resumed_tasks": resumed_ids,
            "resumed_task_count_by_execution": resumed_counts,
            "total_resumed_task_assignments": sum(resumed_counts.values()),
        },
    }


def _reference_runtime_metadata(
    campaign: PreparedCampaign,
    merge: Mapping[str, Any],
    orchestration: Mapping[str, Any],
    orchestration_sha256: str,
) -> dict[str, Any]:
    first_runtime = merge["tasks"][0]["runtime"]
    if not isinstance(first_runtime, Mapping):
        raise ReferenceMPIError("Task runtime metadata is invalid.")
    targets: list[dict[str, Any]] = []
    for item in merge["tasks"]:
        runtime = item["runtime"]
        if not isinstance(runtime, Mapping):
            raise ReferenceMPIError("Task runtime metadata is invalid.")
        summaries = runtime.get("targets")
        if not isinstance(summaries, list) or len(summaries) != 1:
            raise ReferenceMPIError("Task runtime lacks one target summary.")
        targets.append(dict(summaries[0]))
    metadata = dict(first_runtime)
    metadata.update(
        {
            "implementation": "phono3py_4_4_0_task_level_mpi",
            "mesh": list(campaign.config.mesh or ()),
            "grid_points_grg": list(campaign.config.grid_points),
            "grid_points_bzg": list(merge["grid_points_bzg"]),
            "band_indices": list(merge["effective_band_indices"]),
            "gamma_shape": list(merge["gamma_shape"]),
            "targets": targets,
            "mpi_orchestration": {
                "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
                "campaign_sha256": campaign.identity.sha256,
                "orchestration_record_sha256": orchestration_sha256,
                "task_count": len(targets),
                "execution_count": len(orchestration["executions"]),
                "one_complete_target_q_per_worker": True,
                "floating_point_mpi_reductions": False,
                "canonical_merge_order": "strictly_increasing_GRG",
                "full_q_selection": orchestration["selection"][
                    "full_q_selection"
                ],
                "full_band_selection": orchestration["selection"][
                    "full_band_selection"
                ],
                "evidence_archive_sha256": orchestration[
                    "evidence_archive"
                ]["sha256"],
                "restart": orchestration["restart"],
            },
        }
    )
    return metadata


def _finalize_reference_bundle(
    campaign: PreparedCampaign,
    *,
    execution_records: Sequence[Mapping[str, Any]],
) -> Path:
    if campaign.output_dir.exists():
        raise FileExistsError("Final output directory already exists.")
    parent = campaign.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{campaign.output_dir.name}.",
            suffix=".tmp",
            dir=parent,
        )
    )
    try:
        # tempfile created the staging directory; populate it directly.
        for name in sorted(serial._FORCE_STAGE_FILENAMES):
            shutil.copyfile(
                campaign.force_stage_dir / name,
                staging / name,
            )
        merged_path = staging / serial.REFERENCE_FILENAME
        merge = merge_task_references(campaign, merged_path)
        evidence = _write_evidence_archive(
            campaign,
            staging / MPI_EVIDENCE_ARCHIVE_FILENAME,
            execution_records=execution_records,
        )
        full_q_selection = list(campaign.config.grid_points) == list(
            range(int(merge["n_qpoints"]))
        )
        full_band_selection = list(merge["effective_band_indices"]) == list(
            range(int(merge["n_bands"]))
        )
        orchestration = {
            "schema_name": "negfpy_phono3py_si_reference_mpi_orchestration",
            "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
            "campaign_sha256": campaign.identity.sha256,
            "generators": {
                "serial_scientific_generator": dict(
                    campaign.identity.payload["serial_generator"]
                ),
                "mpi_task_and_merge_driver": dict(
                    campaign.identity.payload["mpi_driver"]
                ),
            },
            "campaign_manifest": evidence["campaign_manifest"],
            "scheduler": {
                "parallel_boundary": "complete_target_q",
                "rank_zero_numerical_tasks": False,
                "floating_point_mpi_reductions": False,
                "canonical_merge_order": "strictly_increasing_GRG",
                "worker_thread_limit": 1,
                "worker_binding_policy": _PHYSICAL_CORE_BINDING_POLICY,
            },
            "selection": {
                "mesh": list(campaign.config.mesh or ()),
                "n_qpoints": int(merge["n_qpoints"]),
                "n_bands": int(merge["n_bands"]),
                "grid_points_grg": list(campaign.config.grid_points),
                "band_indices": list(merge["effective_band_indices"]),
                "full_q_selection": full_q_selection,
                "full_band_selection": full_band_selection,
            },
            "evidence_archive": evidence["archive"],
            "tasks": evidence["tasks"],
            "executions": evidence["executions"],
            "restart": evidence["restart"],
            "merged_reference": _file_identity(merged_path),
            "merge": {
                key: value for key, value in merge.items() if key != "tasks"
            },
        }
        _write_json_exclusive(
            staging / MPI_ORCHESTRATION_FILENAME,
            orchestration,
        )
        orchestration_sha256 = _sha256_file(
            staging / MPI_ORCHESTRATION_FILENAME
        )
        _write_json_exclusive(
            staging / "reference-settings.json",
            campaign.config.reference_settings(),
        )
        _write_json_exclusive(
            staging / "reference-environment.json",
            serial._environment_record(campaign.environment_identity),
        )
        runtime_metadata = _reference_runtime_metadata(
            campaign,
            merge,
            orchestration,
            orchestration_sha256,
        )
        _write_json_exclusive(
            staging / "reference-stage.json",
            {
                "schema_version": serial.REFERENCE_SCHEMA_VERSION,
                "stage": "reference",
                "completed_at_utc": _utc_now(),
                "status": "complete",
                "runtime": runtime_metadata,
            },
        )
        _write_json_exclusive(
            staging / "inventory.json",
            serial._inventory(
                staging,
                exclude={"inventory.json", "SHA256SUMS"},
            ),
        )
        names = sorted(
            path.name for path in staging.iterdir() if path.name != "SHA256SUMS"
        )
        serial._write_sha256sums_exclusive(
            staging / "SHA256SUMS",
            root=staging,
            names=names,
        )
        if not serial.verify_sha256sums(staging / "SHA256SUMS", root=staging):
            raise ReferenceMPIError("Final bundle checksum self-check failed.")
        _fsync_directory(staging)
        os.rename(staging, campaign.output_dir)
        _fsync_directory(parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return campaign.output_dir


def _execution_path(run_dir: Path, execution_id: str) -> Path:
    if (
        not execution_id
        or len(execution_id) > 64
        or execution_id in {".", ".."}
        or "/" in execution_id
        or "\\" in execution_id
        or any(
            not (character.isalnum() or character in "._-")
            for character in execution_id
        )
    ):
        raise ValueError(
            "execution_id must be a safe identifier of at most 64 characters."
        )
    return run_dir / "executions" / f"{execution_id}.json"


def _execution_record(
    *,
    campaign: PreparedCampaign,
    execution_id: str,
    size: int,
    started_utc: str,
    started: float,
    assignments: Sequence[Mapping[str, Any]],
    complete: bool,
) -> dict[str, Any]:
    normalized_assignments = [dict(item) for item in assignments]
    points = [
        int(item["grid_point_grg"])
        for item in normalized_assignments
        if isinstance(item.get("grid_point_grg"), int)
    ]
    resumed = [
        int(item["grid_point_grg"])
        for item in normalized_assignments
        if item.get("status") == "resumed"
    ]
    computed = [
        int(item["grid_point_grg"])
        for item in normalized_assignments
        if item.get("status") in {"computed", "adopted"}
    ]
    worker_binding_evidence = _normalize_execution_worker_bindings(
        normalized_assignments,
        communicator_size=size,
    )
    return {
        "schema_name": "negfpy_phono3py_si_reference_mpi_execution",
        "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
        "execution_id": execution_id,
        "campaign_sha256": campaign.identity.sha256,
        "started_at_utc": started_utc,
        "completed_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "communicator_size": size,
        "coordinator_rank": 0,
        "rank_zero_numerical_tasks": size == 1,
        "floating_point_mpi_reductions": False,
        "worker_binding_policy": _PHYSICAL_CORE_BINDING_POLICY,
        "worker_binding_evidence": worker_binding_evidence,
        "canonical_assignment_order": "strictly_increasing_GRG",
        "task_grid_points_grg": list(campaign.config.grid_points),
        "assignment_grid_points_grg": points,
        "resumed_grid_points_grg": resumed,
        "computed_or_adopted_grid_points_grg": computed,
        "exact_checkpoint_resume": bool(resumed),
        "hostname": socket.gethostname(),
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "thread_environment": _require_threads_one(),
        "assignments": normalized_assignments,
        "complete": complete,
    }


_TAG_TASK = 3101
_TAG_RESULT = 3102
_TAG_STOP = 3103


def _serial_schedule(
    campaign: PreparedCampaign,
    *,
    runtime_factory: Callable[[], Any],
) -> list[dict[str, Any]]:
    assignments = []
    for point in campaign.config.grid_points:
        started = time.perf_counter()
        result = _generate_one_task(
            campaign,
            point,
            rank=0,
            runtime_factory=runtime_factory,
        )
        hdf5_path, record_path, _ = _task_paths(campaign.run_dir, point)
        task_record = _validate_task_record(
            record_path,
            hdf5_path,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=point,
        )
        assignments.append(
            {
                "grid_point_grg": point,
                "rank": 0,
                "status": result["status"],
                "wall_seconds": time.perf_counter() - started,
                "artifact": dict(task_record["artifact"]),
            }
        )
    return assignments


def _worker_loop(
    campaign: PreparedCampaign,
    comm: _Communicator,
    *,
    rank: int,
    runtime_factory: Callable[[], Any],
) -> None:
    while True:
        message = comm.recv(source=0)
        if not isinstance(message, Mapping):
            raise ReferenceMPIError("Worker received a malformed command.")
        command = message.get("command")
        if command == "stop":
            return
        if command != "task" or not isinstance(
            message.get("grid_point_grg"),
            int,
        ):
            raise ReferenceMPIError("Worker received an unknown command.")
        point = int(message["grid_point_grg"])
        started = time.perf_counter()
        try:
            worker_binding = _physical_core_binding(rank)
            result = _generate_one_task(
                campaign,
                point,
                rank=rank,
                runtime_factory=runtime_factory,
            )
            hdf5_path, record_path, _ = _task_paths(
                campaign.run_dir,
                point,
            )
            task_record = _validate_task_record(
                record_path,
                hdf5_path,
                campaign_sha256=campaign.identity.sha256,
                config=campaign.config,
                grid_point=point,
            )
            response = {
                "grid_point_grg": point,
                "status": result["status"],
                "wall_seconds": time.perf_counter() - started,
                "worker_binding": worker_binding,
                "artifact": dict(task_record["artifact"]),
            }
        except Exception as exc:
            response = {
                "grid_point_grg": point,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "wall_seconds": time.perf_counter() - started,
            }
        comm.send(response, dest=0, tag=_TAG_RESULT)


def _mpi_schedule(
    campaign: PreparedCampaign,
    comm: _Communicator,
    *,
    runtime_factory: Callable[[], Any],
) -> list[dict[str, Any]] | None:
    from mpi4py import MPI

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank != 0:
        _worker_loop(
            campaign,
            comm,
            rank=rank,
            runtime_factory=runtime_factory,
        )
        return None

    pending: list[int] = []
    assignments: list[dict[str, Any]] = []
    for point in campaign.config.grid_points:
        try:
            existing = _adopt_or_validate_task(campaign, point)
        except ReferenceMPIError as exc:
            assignments.append(
                {
                    "grid_point_grg": point,
                    "rank": None,
                    "status": "failed",
                    "wall_seconds": 0.0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        if existing is None:
            pending.append(point)
        else:
            assignments.append(
                {
                    "grid_point_grg": point,
                    "rank": None,
                    "status": "resumed",
                    "wall_seconds": 0.0,
                }
            )
    workers = list(range(1, size))
    in_flight: dict[int, int] = {}
    for worker in workers:
        if not pending:
            break
        point = pending.pop(0)
        comm.send(
            {"command": "task", "grid_point_grg": point},
            dest=worker,
            tag=_TAG_TASK,
        )
        in_flight[worker] = point
    failures: list[dict[str, Any]] = []
    while in_flight:
        status = MPI.Status()
        response = comm.recv(
            source=MPI.ANY_SOURCE,
            tag=_TAG_RESULT,
            status=status,
        )
        worker = int(status.Get_source())
        expected = in_flight.pop(worker)
        if (
            not isinstance(response, Mapping)
            or response.get("grid_point_grg") != expected
        ):
            failures.append(
                {
                    "grid_point_grg": expected,
                    "rank": worker,
                    "status": "protocol_failure",
                    "wall_seconds": None,
                }
            )
        else:
            record = {
                "grid_point_grg": expected,
                "rank": worker,
                "status": response.get("status"),
                "wall_seconds": response.get("wall_seconds"),
                "worker_binding": response.get("worker_binding"),
                "artifact": response.get("artifact"),
            }
            assignments.append(record)
            if response.get("status") == "failed":
                failures.append({**record, **dict(response)})
        if pending:
            point = pending.pop(0)
            comm.send(
                {"command": "task", "grid_point_grg": point},
                dest=worker,
                tag=_TAG_TASK,
            )
            in_flight[worker] = point
    for worker in workers:
        comm.send({"command": "stop"}, dest=worker, tag=_TAG_STOP)
    return sorted(
        assignments,
        key=lambda item: int(item["grid_point_grg"]),
    )


def execute_campaign(
    campaign: PreparedCampaign,
    *,
    comm: _Communicator | None = None,
    runtime_factory: Callable[[], Any] = serial._Phono3pyRuntime,
    execution_id: str,
    require_mpi: bool = False,
) -> Path | None:
    """Execute, checkpoint, merge, and atomically publish one campaign."""

    if comm is None:
        if require_mpi:
            try:
                from mpi4py import MPI
            except ImportError as exc:
                raise ReferenceMPIError(
                    "MPI execution requires the optional mpi4py dependency."
                ) from exc
            comm = MPI.COMM_WORLD
        else:
            comm = _SerialCommunicator()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if require_mpi and size < 2:
        raise ReferenceMPIError(
            "Certified MPI reference generation requires at least two ranks."
        )
    _require_threads_one()
    execution_path = _execution_path(campaign.run_dir, execution_id)
    execution_preflight_error = (
        (
            "Execution record already exists: "
            f"{execution_path.name}."
        )
        if rank == 0 and execution_path.exists()
        else None
    )
    execution_preflight_error = comm.bcast(
        execution_preflight_error,
        root=0,
    )
    if execution_preflight_error is not None:
        raise FileExistsError(execution_preflight_error)
    started_utc = _utc_now()
    started = time.perf_counter()
    assignments: list[dict[str, Any]] | None
    if size == 1:
        assignments = _serial_schedule(
            campaign,
            runtime_factory=runtime_factory,
        )
    else:
        assignments = _mpi_schedule(
            campaign,
            comm,
            runtime_factory=runtime_factory,
        )
    comm.Barrier()
    if rank != 0:
        return None
    assert assignments is not None
    validation_failures: list[int] = []
    for point in campaign.config.grid_points:
        try:
            if _adopt_or_validate_task(campaign, point) is None:
                validation_failures.append(point)
        except ReferenceMPIError:
            validation_failures.append(point)
    complete_statuses = {"computed", "adopted", "resumed"}
    assignment_points = [
        item.get("grid_point_grg") for item in assignments
    ]
    complete = bool(
        not validation_failures
        and assignment_points == list(campaign.config.grid_points)
        and all(
            item.get("status") in complete_statuses for item in assignments
        )
    )
    record = _execution_record(
        campaign=campaign,
        execution_id=execution_id,
        size=size,
        started_utc=started_utc,
        started=started,
        assignments=assignments,
        complete=complete,
    )
    _write_json_exclusive(execution_path, record)
    if not complete:
        raise ReferenceMPIError(
            "Campaign ended without all authenticated target artifacts: "
            f"{validation_failures!r}."
        )
    execution_records = [
        _read_json_object(path, name=f"execution record {path.name}")
        for path in sorted((campaign.run_dir / "executions").glob("*.json"))
    ]
    return _finalize_reference_bundle(
        campaign,
        execution_records=execution_records,
    )


class _SerialCommunicator:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value

    def send(self, value: Any, dest: int, tag: int = 0) -> None:
        raise RuntimeError("Serial communicator cannot send.")

    def recv(
        self,
        source: int = -1,
        tag: int = -1,
        status: Any | None = None,
    ) -> Any:
        raise RuntimeError("Serial communicator cannot receive.")

    def Barrier(self) -> None:
        return None


def _broadcast_prepared(
    comm: _Communicator,
    *,
    arguments: argparse.Namespace,
    config: serial.GeneratorConfig,
) -> PreparedCampaign:
    rank = comm.Get_rank()
    if rank == 0:
        try:
            prepared = prepare_campaign(
                manifest_path=arguments.manifest,
                source_dir=arguments.source_dir,
                force_stage_dir=arguments.force_stage_dir,
                run_dir=arguments.run_dir,
                output_dir=arguments.output_dir,
                config=config,
            )
            descriptor = {
                "ok": True,
                "campaign_sha256": prepared.identity.sha256,
                "payload": prepared.identity.payload,
                "source_files": {
                    name: str(path)
                    for name, path in prepared.source_files.items()
                },
                "source_verification": prepared.source_verification,
                "environment_identity": prepared.environment_identity,
            }
        except Exception as exc:
            descriptor = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    else:
        descriptor = None
    descriptor = comm.bcast(descriptor, root=0)
    if not descriptor.get("ok"):
        raise ReferenceMPIError(
            "Rank-zero campaign preflight failed: "
            f"{descriptor.get('error_type')}: {descriptor.get('error_message')}"
        )
    identity = CampaignIdentity.from_payload(descriptor["payload"])
    if identity.sha256 != descriptor["campaign_sha256"]:
        raise ReferenceMPIError("Broadcast campaign hash is invalid.")
    return PreparedCampaign(
        identity=identity,
        config=config,
        source_files={
            name: Path(path)
            for name, path in descriptor["source_files"].items()
        },
        source_verification=descriptor["source_verification"],
        environment_identity=descriptor["environment_identity"],
        force_stage_dir=arguments.force_stage_dir,
        run_dir=arguments.run_dir,
        output_dir=arguments.output_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an authenticated phono3py Si reference with one complete "
            "target q-point per MPI worker."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--force-stage-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument(
        "--mesh",
        nargs=3,
        required=True,
        type=int,
        metavar=("NX", "NY", "NZ"),
    )
    target_selection = parser.add_mutually_exclusive_group(required=True)
    target_selection.add_argument(
        "--grid-points",
        nargs="+",
        type=int,
        metavar="GRG_INDEX",
    )
    target_selection.add_argument(
        "--all-grid-points",
        action="store_true",
        help="Generate every GRG target in the configured full mesh.",
    )
    parser.add_argument("--band-indices", nargs="+", type=int, default=None)
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[300.0],
        metavar="K",
    )
    parser.add_argument("--sigma", type=float, default=0.1, metavar="THZ")
    parser.add_argument("--sigma-cutoff", type=float, default=None)
    parser.add_argument("--backend", choices=("C", "Rust"), default="Rust")
    parser.add_argument(
        "--make-r0-average",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--symmetrize-fc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--symmetrize-fc3q",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = serial.GeneratorConfig(
        mesh=arguments.mesh,
        grid_points=(
            tuple(range(math.prod(arguments.mesh)))
            if arguments.all_grid_points
            else arguments.grid_points
        ),
        band_indices=arguments.band_indices,
        temperatures=arguments.temperatures,
        sigma=arguments.sigma,
        sigma_cutoff=arguments.sigma_cutoff,
        backend=arguments.backend,
        make_r0_average=arguments.make_r0_average,
        symmetrize_fc=arguments.symmetrize_fc,
        symmetrize_fc3q=arguments.symmetrize_fc3q,
    )
    try:
        from mpi4py import MPI
    except ImportError as exc:
        parser.error(
            "This MPI generator requires the optional mpi4py dependency: "
            f"{exc}"
        )
    campaign = _broadcast_prepared(
        MPI.COMM_WORLD,
        arguments=arguments,
        config=config,
    )
    output = execute_campaign(
        campaign,
        comm=MPI.COMM_WORLD,
        execution_id=arguments.execution_id,
        require_mpi=True,
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert output is not None
        print(
            json.dumps(
                {
                    "campaign_sha256": campaign.identity.sha256,
                    "communicator_size": MPI.COMM_WORLD.Get_size(),
                    "output_dir": str(output.resolve()),
                    "reference_hdf5_sha256": _sha256_file(
                        output / serial.REFERENCE_FILENAME
                    ),
                    "sha256sums": str((output / "SHA256SUMS").resolve()),
                    "status": "complete",
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
