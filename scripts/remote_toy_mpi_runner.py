#!/usr/bin/env python3
"""Exact task-MPI scaling and release entry points for the production toy gate.

The scaling driver launches five fresh, isolated repetitions of the complete
28-task campaign.  It changes only the preregistered numerical resolution for
timing and is therefore irrevocably noncertifying.  The release entry point
uses the locked production fixtures without exposing scientific overrides.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import remote_mpi_scaling as scaling


THREAD_LIMIT_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SCALING_TASK_COUNT = 28
SCALING_MULTIDIMENSIONAL_COUNT = 13
SCALING_OPEN_COUNT = 13
SCALING_CONTINUATION_COUNT = 5
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EXECUTION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
MEMORY_LIMIT_FRACTION = 0.70
_MEMORY_SCHEMA = "negfpy.remote_toy_mpi.aggregate_memory_preflight.v1"


class RemoteToyMPIRunnerError(RuntimeError):
    """Raised when orchestration evidence is incomplete or inconsistent."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _payload_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _absolute_path_without_symlinks(path: str | Path, *, name: str) -> Path:
    expanded = Path(path).expanduser()
    if ".." in expanded.parts:
        raise RemoteToyMPIRunnerError(f"{name} contains parent traversal.")
    absolute = Path(os.path.abspath(os.fspath(expanded)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise RemoteToyMPIRunnerError(
                f"{name} traverses a symlink: {current}"
            )
    return absolute


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_hashed_manifest(
    path: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish an immutable strict-JSON payload envelope."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    normalized = json.loads(_canonical_bytes(payload).decode("ascii"))
    document = {
        "payload": normalized,
        "payload_sha256": _payload_sha256(normalized),
    }
    raw = _canonical_bytes(document)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise RemoteToyMPIRunnerError(
                f"refusing to replace immutable manifest: {target}"
            ) from exc
        _fsync_directory(target.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return document


def read_hashed_manifest(path: str | Path) -> dict[str, Any]:
    """Strictly reload an immutable manifest and verify both encodings."""

    source = _absolute_path_without_symlinks(
        path,
        name="hashed manifest",
    ).resolve(strict=True)
    try:
        raw = source.read_bytes()
        document = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RemoteToyMPIRunnerError(
            f"invalid hashed manifest: {source}"
        ) from exc
    if not isinstance(document, dict) or set(document) != {
        "payload",
        "payload_sha256",
    }:
        raise RemoteToyMPIRunnerError("hashed manifest has the wrong envelope.")
    payload = document["payload"]
    if not isinstance(payload, dict):
        raise RemoteToyMPIRunnerError("hashed-manifest payload is not an object.")
    digest = document["payload_sha256"]
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise RemoteToyMPIRunnerError("hashed manifest has an invalid SHA256.")
    if digest != _payload_sha256(payload):
        raise RemoteToyMPIRunnerError("hashed-manifest payload SHA256 mismatch.")
    if raw != _canonical_bytes(document):
        raise RemoteToyMPIRunnerError("hashed manifest is not canonical JSON.")
    return document


def _validate_identity(
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
) -> None:
    if _COMMIT.fullmatch(source_revision) is None:
        raise ValueError("source_revision must be a full lowercase Git commit.")
    for name, value in (
        ("source_tree_sha256", source_tree_sha256),
        ("environment_sha256", environment_sha256),
    ):
        if _SHA256.fullmatch(value) is None:
            raise ValueError(f"{name} must be a lowercase SHA256.")


def _validate_execution_id(value: str) -> str:
    if _EXECUTION_ID.fullmatch(value) is None:
        raise ValueError("execution_id contains unsupported characters.")
    return value


def _thread_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(THREAD_LIMIT_ENV)
    return environment

def _available_memory_bytes() -> int:
    """Return the current Linux MemAvailable value, never a stale total."""

    try:
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/meminfo")
            .read_text(encoding="ascii")
            .splitlines()
            if ":" in line
        }
        raw = fields["MemAvailable"].split()
        if len(raw) != 2 or raw[1] != "kB":
            raise ValueError("MemAvailable has an unsupported unit.")
        value = int(raw[0]) * 1024
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RemoteToyMPIRunnerError(
            "cannot measure current /proc/meminfo MemAvailable"
        ) from exc
    if value <= 0:
        raise RemoteToyMPIRunnerError("MemAvailable must be positive.")
    return value


def _locked_profile_inputs(
    profile: str,
) -> tuple[Any, tuple[tuple[str, Any], ...]]:
    if profile == "scaling":
        return representative_scaling_profile()
    if profile != "release":
        raise ValueError("profile must be scaling or release.")
    from negfpy.anharmonic.toy.multidimensional_release import (
        full_multidimensional_release_config,
    )
    from negfpy.anharmonic.toy.production_gate_runner import (
        open_three_mode_release_fixtures,
    )

    return (
        full_multidimensional_release_config(),
        open_three_mode_release_fixtures(),
    )


def campaign_dense_memory_preflight(
    *,
    profile: str,
    rank_count: int,
    available_memory_bytes: int | None = None,
) -> dict[str, Any]:
    """Conservatively gate aggregate dense memory for one whole-case launch."""

    if isinstance(rank_count, bool) or not isinstance(rank_count, int):
        raise TypeError("rank_count must be an integer.")
    if rank_count < 1:
        raise ValueError("rank_count must be positive.")
    config, fixtures = _locked_profile_inputs(profile)
    from negfpy.anharmonic.toy.multidimensional_release import (
        build_multidimensional_problem,
    )
    from negfpy.anharmonic.types import estimate_dense_memory_bytes

    base_problem = build_multidimensional_problem(
        config,
        g=config.executed_g_values[0],
        bias=config.executed_bias_values[0],
        numerical=config.numerical,
    )
    minimum_spacing = min(config.spec.spacings)
    maximum_cutoff = max(config.spec.cutoffs)
    multidimensional_n_frequency = (
        2 * int(round(maximum_cutoff / minimum_spacing)) + 1
    )
    contributors = [
        {
            "kind": "multidimensional_cross_product_upper_bound",
            "n_frequency": multidimensional_n_frequency,
            "ndof": base_problem.ndof,
            "predicted_bytes": estimate_dense_memory_bytes(
                multidimensional_n_frequency,
                base_problem.ndof,
            ),
        }
    ]
    largest_open = max(
        fixtures,
        key=lambda item: estimate_dense_memory_bytes(
            item[1].grid.n_points,
            item[1].problem.ndof,
        ),
    )
    contributors.append(
        {
            "kind": "open_three_mode_largest_fixture",
            "fixture": largest_open[0],
            "n_frequency": largest_open[1].grid.n_points,
            "ndof": largest_open[1].problem.ndof,
            "predicted_bytes": estimate_dense_memory_bytes(
                largest_open[1].grid.n_points,
                largest_open[1].problem.ndof,
            ),
        }
    )
    per_rank = max(int(item["predicted_bytes"]) for item in contributors)
    aggregate = per_rank * rank_count
    available = (
        _available_memory_bytes()
        if available_memory_bytes is None
        else available_memory_bytes
    )
    if isinstance(available, bool) or not isinstance(available, int):
        raise TypeError("available_memory_bytes must be an integer.")
    if available <= 0:
        raise ValueError("available_memory_bytes must be positive.")
    limit = int(MEMORY_LIMIT_FRACTION * available)
    passed = aggregate <= limit
    record = {
        "schema": _MEMORY_SCHEMA,
        "measured_utc": _utc_now(),
        "measurement": "linux_proc_meminfo_MemAvailable",
        "profile": profile,
        "rank_count": rank_count,
        "allowed_fraction": MEMORY_LIMIT_FRACTION,
        "available_memory_bytes": available,
        "limit_bytes": limit,
        "per_rank_predicted_bytes": per_rank,
        "aggregate_predicted_bytes": aggregate,
        "contributors": contributors,
        "passed": passed,
    }
    if not passed:
        raise RemoteToyMPIRunnerError(
            "aggregate dense-memory estimate exceeds 70% of current "
            f"MemAvailable: {aggregate} > {limit} bytes"
        )
    return record


def _logical_cpu_topology(cpu_id: int) -> dict[str, int]:
    root = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
    try:
        package_id = int(
            (root / "physical_package_id")
            .read_text(encoding="ascii")
            .strip()
        )
        core_id = int(
            (root / "core_id").read_text(encoding="ascii").strip()
        )
    except (OSError, ValueError) as exc:
        raise RemoteToyMPIRunnerError(
            f"cannot resolve physical-core topology for logical CPU {cpu_id}"
        ) from exc
    return {
        "logical_cpu_id": cpu_id,
        "physical_package_id": package_id,
        "core_id": core_id,
    }


def _rank_runtime_record(rank: int) -> dict[str, Any]:
    try:
        logical = sorted(int(value) for value in os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise RemoteToyMPIRunnerError(
            "rank affinity is unavailable on this launch"
        ) from exc
    if not logical:
        raise RemoteToyMPIRunnerError("rank affinity cannot be empty.")
    topology = [_logical_cpu_topology(value) for value in logical]
    physical = sorted(
        {
            (item["physical_package_id"], item["core_id"])
            for item in topology
        }
    )
    executable = Path(sys.executable).expanduser().resolve()
    return {
        "rank": rank,
        "hostname": socket.gethostname(),
        "logical_cpu_ids": logical,
        "logical_cpu_topology": topology,
        "physical_cores": [
            {
                "physical_package_id": package_id,
                "core_id": core_id,
            }
            for package_id, core_id in physical
        ],
        "one_physical_core": len(physical) == 1,
        "thread_limits": {
            key: os.environ.get(key) for key in sorted(THREAD_LIMIT_ENV)
        },
        "python": _launcher_record(executable),
        "python_argv": [str(value) for value in sys.argv],
    }


def _validate_rank_runtime(
    records: Sequence[Mapping[str, Any]],
    *,
    socket_count: int | None = None,
    rank_count: int,
) -> tuple[dict[str, Any], ...]:
    if len(records) != rank_count:
        raise RemoteToyMPIRunnerError("rank-runtime coverage is incomplete.")
    by_rank: dict[int, dict[str, Any]] = {}
    package_counts: dict[tuple[str, int], int] = {}
    occupied: set[tuple[str, int, int]] = set()
    expected_threads = dict(sorted(THREAD_LIMIT_ENV.items()))
    for supplied in records:
        record = dict(supplied)
        rank = record.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RemoteToyMPIRunnerError("rank-runtime rank is invalid.")
        if rank in by_rank or not 0 <= rank < rank_count:
            raise RemoteToyMPIRunnerError(
                "rank-runtime ranks are duplicated or invalid."
            )
        hostname = record.get("hostname")
        logical = record.get("logical_cpu_ids")
        topology = record.get("logical_cpu_topology")
        physical = record.get("physical_cores")
        if (
            not isinstance(hostname, str)
            or not hostname
            or not isinstance(logical, list)
            or not logical
            or len(set(logical)) != len(logical)
            or not isinstance(topology, list)
            or len(topology) != len(logical)
            or not isinstance(physical, list)
            or len(physical) != 1
            or record.get("one_physical_core") is not True
            or record.get("thread_limits") != expected_threads
        ):
            raise RemoteToyMPIRunnerError(
                f"rank-{rank} runtime affinity/thread evidence is invalid."
            )
        topology_by_cpu = {
            item.get("logical_cpu_id"): (
                item.get("physical_package_id"), item.get("core_id")
            )
            for item in topology
            if isinstance(item, Mapping)
        }
        measured_topology = {
            item["logical_cpu_id"]: (
                item["physical_package_id"],
                item["core_id"],
            )
            for item in (_logical_cpu_topology(int(value)) for value in logical)
        }
        core = physical[0]
        key = (
            hostname,
            core.get("physical_package_id"),
            core.get("core_id"),
        )
        if (
            set(topology_by_cpu) != set(logical)
            or topology_by_cpu != measured_topology
            or set(topology_by_cpu.values()) != {key[1:]}
        ):
            raise RemoteToyMPIRunnerError(
                f"rank-{rank} logical CPUs do not map to one recorded core."
            )
        if key in occupied:
            raise RemoteToyMPIRunnerError(
                "concurrently active MPI ranks share a physical core."
            )
        occupied.add(key)
        package_key = key[:2]
        package_counts[package_key] = package_counts.get(package_key, 0) + 1
        by_rank[rank] = record
    if tuple(sorted(by_rank)) != tuple(range(rank_count)):
        raise RemoteToyMPIRunnerError(
            "rank-runtime rank sequence is incomplete."
        )
    if socket_count is not None:
        if isinstance(socket_count, bool) or socket_count <= 0:
            raise ValueError("socket_count must be positive.")
        if rank_count >= socket_count and rank_count % socket_count == 0:
            expected_per_socket = rank_count // socket_count
            if (
                len(package_counts) != socket_count
                or set(package_counts.values()) != {expected_per_socket}
            ):
                raise RemoteToyMPIRunnerError(
                    "MPI rank affinity does not prove the locked equal "
                    "distribution across sockets."
                )
    return tuple(by_rank[rank] for rank in range(rank_count))




def representative_scaling_profile() -> tuple[Any, tuple[tuple[str, Any], ...]]:
    """Return the locked noncertifying full-topology timing profile."""

    from negfpy.anharmonic.grid import SignedFrequencyGrid
    from negfpy.anharmonic.toy.multidimensional_release import (
        MultidimensionalLadderSpec,
        MultidimensionalNumericalPoint,
        full_multidimensional_release_config,
    )
    from negfpy.anharmonic.toy.production_gate_runner import (
        open_three_mode_release_fixtures,
    )

    spec = MultidimensionalLadderSpec(
        spacings=(0.025, 0.0125, 0.00625),
        cutoffs=(5.0, 6.0),
        etas=(0.05, 0.025, 0.0125),
        mixings=(0.2, 0.1),
    )
    config = replace(
        full_multidimensional_release_config(),
        spec=spec,
        numerical=MultidimensionalNumericalPoint(
            spacing=0.0125,
            cutoff=5.0,
            eta_lead=0.025,
            mixing=0.2,
        ),
        certification_eligible=False,
    )
    fixtures = open_three_mode_release_fixtures(
        base_grid=SignedFrequencyGrid(step=0.00125, n_positive=4000),
        base_eta_lead=0.0008,
    )
    if len(fixtures) != SCALING_OPEN_COUNT:
        raise AssertionError("the scaling open-system topology changed.")
    return config, fixtures


def _directory_mode(path: Path, *, allow_exact_resume: bool) -> str:
    path = _absolute_path_without_symlinks(path, name="run_directory")
    if path.is_symlink():
        raise RemoteToyMPIRunnerError("run_directory cannot be a symlink.")
    if not path.exists():
        return "fresh"
    if not path.is_dir():
        raise RemoteToyMPIRunnerError("run_directory is not a directory.")
    entries = tuple(path.iterdir())
    if not entries:
        return "fresh"
    if not allow_exact_resume:
        raise RemoteToyMPIRunnerError(
            "scaling repetitions must use a fresh isolated directory."
        )
    if not (path / "campaign-manifest.json").is_file():
        raise RemoteToyMPIRunnerError(
            "nonempty release directory is not an exact-resume campaign."
        )
    return "exact_resume_candidate"


def _artifact_record(root: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RemoteToyMPIRunnerError(
            f"campaign artifact escaped run directory: {resolved}"
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise RemoteToyMPIRunnerError(
            f"campaign artifact is not a regular file: {resolved}"
        )
    return {"relative_path": relative, "sha256": _file_sha256(resolved)}


def _physics_result_payload(
    *,
    profile: str,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    expected_ranks: int,
    directory_mode: str,
    result: Any,
    config: Any,
    memory_preflight: Mapping[str, Any],
    rank_runtime: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from negfpy.anharmonic.toy._mpi_production_gate_physics import (
        production_toy_physics_sha256,
    )

    outcomes = tuple(result.execution.outcomes)
    incomplete = [item.task_id for item in outcomes if item.status != "completed"]
    if incomplete:
        raise RemoteToyMPIRunnerError(
            "campaign has non-completed outcomes: " + ", ".join(incomplete)
        )
    if len(outcomes) != SCALING_TASK_COUNT:
        raise RemoteToyMPIRunnerError(
            f"campaign has {len(outcomes)} outcomes, expected {SCALING_TASK_COUNT}."
        )
    if result.execution.communicator_size != expected_ranks:
        raise RemoteToyMPIRunnerError("recorded communicator size changed.")
    metadata = result.campaign.manifest.metadata
    expected_metadata = {
        "multidimensional_outcome_count": SCALING_MULTIDIMENSIONAL_COUNT,
        "open_system_case_count": SCALING_OPEN_COUNT,
        "continuation_path_count": SCALING_CONTINUATION_COUNT,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise RemoteToyMPIRunnerError(
                f"campaign metadata {key!r} changed from {expected}."
            )
    report = result.report.as_dict()
    release_certified = bool(report["summary"]["release_certified"])
    if profile == "scaling" and release_certified:
        raise RemoteToyMPIRunnerError(
            "representative scaling evidence cannot be release certified."
        )
    root = result.strict_report_path.parents[1]
    return {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "created_utc": _utc_now(),
        "profile": profile,
        "certification_eligible": bool(config.certification_eligible),
        "release_certified": release_certified,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "directory_mode": directory_mode,
        "expected_rank_count": expected_ranks,
        "recorded_rank_count": result.execution.communicator_size,
        "execution_mode": result.execution.mode,
        "execution_id": result.execution.execution_id,
        "task_count": len(outcomes),
        "multidimensional_outcome_count": metadata[
            "multidimensional_outcome_count"
        ],
        "open_system_case_count": metadata["open_system_case_count"],
        "continuation_path_count": metadata["continuation_path_count"],
        "all_outcomes_completed": True,
        "campaign_manifest_sha256": result.campaign.campaign_manifest_sha256,
        "execution_scientific_sha256": result.execution.scientific_sha256,
        "production_toy_physics_sha256": production_toy_physics_sha256(
            result.report
        ),
        "configuration_sha256": _payload_sha256(config.as_dict()),
        "campaign_manifest": _artifact_record(
            root, root / "campaign-manifest.json"
        ),
        "execution_manifest": _artifact_record(
            root, result.execution_manifest_path
        ),
        "strict_report": _artifact_record(root, result.strict_report_path),
        "memory_preflight": dict(memory_preflight),
        "rank_runtime": [dict(item) for item in rank_runtime],
        "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
    }


def execute_profile_under_mpi(
    *,
    profile: str,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    run_directory: str | Path,
    execution_id: str,
    expected_ranks: int,
) -> bool:
    """Execute one profile on every rank and publish a rank-zero result."""

    os.environ.update(THREAD_LIMIT_ENV)
    from mpi4py import MPI
    from negfpy.anharmonic.toy._mpi_production_gate_physics import (
        run_production_toy_mpi_physics,
    )
    from negfpy.anharmonic.toy.multidimensional_release import (
        full_multidimensional_release_config,
    )

    _validate_identity(
        source_revision,
        source_tree_sha256,
        environment_sha256,
    )
    execution_id = _validate_execution_id(execution_id)
    if profile not in {"scaling", "release"}:
        raise ValueError("profile must be scaling or release.")
    communicator = MPI.COMM_WORLD
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    if size != expected_ranks:
        if rank == 0:
            print(
                f"expected {expected_ranks} MPI ranks, received {size}",
                file=sys.stderr,
            )
        return False
    local_runtime: dict[str, Any]
    try:
        local_runtime = {
            "ok": True,
            "record": _rank_runtime_record(rank),
        }
    except Exception as exc:
        local_runtime = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    gathered_runtime = communicator.gather(local_runtime, root=0)
    runtime_state: dict[str, Any] | None = None
    if rank == 0:
        try:
            failures = [item for item in gathered_runtime if not item["ok"]]
            if failures:
                raise RemoteToyMPIRunnerError(
                    "rank runtime collection failed: "
                    + "; ".join(
                        f"{item['error_type']}: {item['error_message']}"
                        for item in failures
                    )
                )
            rank_runtime = _validate_rank_runtime(
                [item["record"] for item in gathered_runtime],
                rank_count=size,
            )
            memory_preflight = campaign_dense_memory_preflight(
                profile=profile,
                rank_count=size,
            )
            runtime_state = {
                "ok": True,
                "rank_runtime": list(rank_runtime),
                "memory_preflight": memory_preflight,
            }
        except Exception as exc:
            runtime_state = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    runtime_state = communicator.bcast(runtime_state, root=0)
    if runtime_state["ok"] is not True:
        if rank == 0:
            print(
                f"{runtime_state['error_type']}: "
                f"{runtime_state['error_message']}",
                file=sys.stderr,
            )
        return False
    root = _absolute_path_without_symlinks(
        run_directory, name="run_directory"
    ).resolve()
    setup: dict[str, Any] | None = None
    if rank == 0:
        try:
            mode = _directory_mode(
                root,
                allow_exact_resume=profile == "release",
            )
        except Exception as exc:
            setup = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        else:
            setup = {"ok": True, "directory_mode": mode}
    setup = communicator.bcast(setup, root=0)
    if setup["ok"] is not True:
        if rank == 0:
            print(
                f"{setup['error_type']}: {setup['error_message']}",
                file=sys.stderr,
            )
        return False

    if profile == "scaling":
        config, fixtures = representative_scaling_profile()
        tier = "pull_request"
    else:
        config = full_multidimensional_release_config()
        fixtures = None
        tier = "release"

    result = None
    run_error: Exception | None = None
    try:
        result = run_production_toy_mpi_physics(
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            run_directory=root,
            config=config,
            open_fixtures=fixtures,
            validation_tier=tier,
            comm=communicator,
            require_mpi=True,
            thread_limit=1,
            execution_id=execution_id,
        )
    except Exception as exc:
        run_error = exc

    terminal: dict[str, Any] | None = None
    if rank == 0:
        try:
            if run_error is not None:
                raise run_error
            if result is None:
                raise RemoteToyMPIRunnerError(
                    "rank zero did not receive the merged campaign result."
                )
            payload = _physics_result_payload(
                profile=profile,
                source_revision=source_revision,
                source_tree_sha256=source_tree_sha256,
                environment_sha256=environment_sha256,
                expected_ranks=expected_ranks,
                directory_mode=str(setup["directory_mode"]),
                result=result,
                config=config,
                memory_preflight=runtime_state["memory_preflight"],
                rank_runtime=runtime_state["rank_runtime"],
            )
            target = root / "runner-results" / f"{execution_id}.json"
            write_hashed_manifest(target, payload)
            certified = bool(payload["release_certified"])
            ok = profile == "scaling" or certified
            terminal = {
                "ok": ok,
                "result_manifest": target.relative_to(root).as_posix(),
                "production_toy_physics_sha256": payload[
                    "production_toy_physics_sha256"
                ],
                "release_certified": certified,
                "error_type": None if ok else "ReleaseCertificationFailed",
                "error_message": (
                    None
                    if ok
                    else "locked release report is not release_certified"
                ),
            }
        except Exception as exc:
            terminal = {
                "ok": False,
                "result_manifest": None,
                "production_toy_physics_sha256": None,
                "release_certified": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    terminal = communicator.bcast(terminal, root=0)
    if rank == 0:
        print(
            json.dumps(
                terminal,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        if terminal["ok"] is not True:
            print(
                f"{terminal['error_type']}: {terminal['error_message']}",
                file=sys.stderr,
            )
    return bool(terminal["ok"])


def _launcher_record(path: Path) -> dict[str, str]:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RemoteToyMPIRunnerError(
            f"environment-local executable is unavailable: {path}"
        )
    return {
        "path": str(path),
        "resolved_path": str(path.resolve()),
        "sha256": _file_sha256(path),
    }

def _safe_artifact_path(
    root: Path,
    record: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "relative_path",
        "sha256",
    }:
        raise RemoteToyMPIRunnerError(f"{name} has an invalid artifact record.")
    relative_text = record.get("relative_path")
    digest = record.get("sha256")
    if not isinstance(relative_text, str) or not relative_text:
        raise RemoteToyMPIRunnerError(f"{name} has no relative path.")
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_text
    ):
        raise RemoteToyMPIRunnerError(f"{name} has an unsafe relative path.")
    supplied_root = _absolute_path_without_symlinks(root, name=f"{name} root")
    resolved_root = supplied_root.resolve(strict=True)
    candidate = resolved_root / relative
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RemoteToyMPIRunnerError(f"{name} traverses a symlink.")
    try:
        target = candidate.resolve(strict=True)
    except OSError as exc:
        raise RemoteToyMPIRunnerError(f"{name} is missing.") from exc
    try:
        target.relative_to(resolved_root)
    except ValueError as exc:
        raise RemoteToyMPIRunnerError(f"{name} escaped its campaign root.") from exc
    if (
        not target.is_file()
        or target.is_symlink()
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or _file_sha256(target) != digest
    ):
        raise RemoteToyMPIRunnerError(
            f"{name} is missing, unsafe, or hash-mismatched."
        )
    return target


def _safe_referenced_path(
    root: Path,
    relative_text: Any,
    digest: Any,
    *,
    name: str,
) -> Path:
    return _safe_artifact_path(
        root,
        {"relative_path": relative_text, "sha256": digest},
        name=name,
    )


def _validate_executable_record(
    record: Any,
    *,
    expected_name: str,
) -> tuple[Path, Path]:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "resolved_path",
        "sha256",
    }:
        raise RemoteToyMPIRunnerError(
            f"{expected_name} executable identity is incomplete."
        )
    path = Path(str(record["path"]))
    resolved = Path(str(record["resolved_path"]))
    digest = record["sha256"]
    if (
        not path.is_absolute()
        or path.name != expected_name
        or path.resolve() != resolved
        or not path.is_file()
        or not os.access(path, os.X_OK)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or _file_sha256(path) != digest
    ):
        raise RemoteToyMPIRunnerError(
            f"{expected_name} executable identity failed revalidation."
        )
    return path, resolved


def _validate_memory_record(
    record: Any,
    *,
    profile: str,
    rank_count: int,
) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise RemoteToyMPIRunnerError("memory preflight is missing.")
    available = record.get("available_memory_bytes")
    if isinstance(available, bool) or not isinstance(available, int):
        raise RemoteToyMPIRunnerError("memory preflight availability is invalid.")
    expected = campaign_dense_memory_preflight(
        profile=profile,
        rank_count=rank_count,
        available_memory_bytes=available,
    )
    supplied_without_time = {
        key: value for key, value in record.items() if key != "measured_utc"
    }
    expected_without_time = {
        key: value for key, value in expected.items() if key != "measured_utc"
    }
    if supplied_without_time != expected_without_time or not isinstance(
        record.get("measured_utc"), str
    ):
        raise RemoteToyMPIRunnerError(
            "memory preflight does not match the conservative dense estimate."
        )
    return dict(record)


def _validate_real_harmonic_preflight(
    artifact_path: Path,
    *,
    profile: str,
) -> dict[str, Any]:
    from negfpy.anharmonic.toy._mpi_production_gate_physics import (
        _read_case_artifact,
    )

    evidence, arrays = _read_case_artifact(artifact_path)
    points = evidence.get("points")
    comparisons = evidence.get("comparisons")
    if (
        arrays
        or evidence.get("schema")
        != "negfpy.anharmonic.harmonic_same_grid_preflight"
        or evidence.get("schema_version") != 2
        or evidence.get("protocol") != "multidimensional_toy_gate_v2"
        or evidence.get("applicable") is not True
        or evidence.get("passed") is not True
        or evidence.get("scope")
        != "base_plus_six_preregistered_numerical_points"
        or evidence.get("point_count") != 7
        or evidence.get("all_seven_archived") is not True
        or not isinstance(points, list)
        or len(points) != 7
        or not isinstance(comparisons, list)
        or len(comparisons) != 6
    ):
        raise RemoteToyMPIRunnerError(
            "harmonic preflight is dummy, incomplete, or failed."
        )
    from negfpy.anharmonic.toy.multidimensional_release import (
        _refinement_points,
    )

    config, _ = _locked_profile_inputs(profile)
    expected_labels = {"base"} | {
        label for label, _ in _refinement_points(config)
    }
    labels = {point.get("label") for point in points if isinstance(point, Mapping)}
    for point in points:
        fields = point.get("field_sha256") if isinstance(point, Mapping) else None
        if (
            not isinstance(point, Mapping)
            or point.get("passed") is not True
            or not isinstance(fields, Mapping)
            or not fields
            or any(
                not isinstance(value, str) or _SHA256.fullmatch(value) is None
                for value in fields.values()
            )
        ):
            raise RemoteToyMPIRunnerError(
                "harmonic preflight lacks passing physical field evidence."
            )
    if labels != expected_labels:
        raise RemoteToyMPIRunnerError(
            "harmonic preflight numerical-point topology changed."
        )
    return {
        "schema": "negfpy.remote_toy_mpi.real_harmonic_preflight.v1",
        "artifact_file_sha256": _file_sha256(artifact_path),
        "evidence_sha256": hashlib.sha256(
            json.dumps(
                evidence,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        "point_count": 7,
        "real_physics_executed": True,
        "passed": True,
    }


def _strict_report(path: Path) -> tuple[Any, dict[str, Any]]:
    from negfpy.anharmonic.toy.production_gate_report import (
        ProductionToyGateReport,
    )

    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        report = ProductionToyGateReport(document=document)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RemoteToyMPIRunnerError(
            "strict production report is invalid or truncated."
        ) from exc
    return report, report.as_dict()




def _expected_campaign(
    *,
    profile: str,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
) -> tuple[Any, Any]:
    from negfpy.anharmonic.toy._mpi_production_gate import (
        build_production_toy_mpi_campaign,
    )

    config, fixtures = _locked_profile_inputs(profile)
    campaign = build_production_toy_mpi_campaign(
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        config=config,
        open_fixtures=None if profile == "release" else fixtures,
    )
    return campaign, config


def _assignment_physical_core(
    assignment: Mapping[str, Any],
) -> tuple[str, int, int, int]:
    hostname = assignment.get("hostname")
    rank = assignment.get("rank")
    logical = assignment.get("cpu_affinity")
    if (
        not isinstance(hostname, str)
        or not hostname
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or rank < 0
        or not isinstance(logical, list)
        or not logical
        or any(isinstance(value, bool) or not isinstance(value, int) for value in logical)
        or logical != sorted(set(logical))
    ):
        raise RemoteToyMPIRunnerError("task assignment affinity is invalid.")
    measured = [_logical_cpu_topology(value) for value in logical]
    physical = {
        (item["physical_package_id"], item["core_id"])
        for item in measured
    }
    if len(physical) != 1:
        raise RemoteToyMPIRunnerError("task assignment spans physical cores.")
    package_id, core_id = next(iter(physical))
    return hostname, rank, package_id, core_id

def _current_assignment_core(
    assignment: Mapping[str, Any],
    *,
    runtime_by_rank: Mapping[int, Mapping[str, Any]],
    rank_count: int,
    require_fresh: bool,
) -> tuple[tuple[str, int], tuple[int, int]] | None:
    resumed = assignment.get("resumed")
    if not isinstance(resumed, bool):
        raise RemoteToyMPIRunnerError("assignment resume flag is invalid.")
    hostname, rank, package_id, core_id = _assignment_physical_core(assignment)
    if require_fresh and resumed:
        raise RemoteToyMPIRunnerError("fresh campaign resumed a task.")
    if resumed:
        return None
    if rank not in runtime_by_rank or (rank_count > 1 and rank == 0):
        raise RemoteToyMPIRunnerError(
            "assignment used an invalid current MPI rank."
        )
    runtime = runtime_by_rank[rank]
    if (
        hostname != runtime.get("hostname")
        or assignment.get("cpu_affinity") != runtime.get("logical_cpu_ids")
    ):
        raise RemoteToyMPIRunnerError(
            "fresh assignment differs from rank startup affinity."
        )
    return (hostname, rank), (package_id, core_id)




def _validate_phase_execution_evidence(
    *,
    root: Path,
    campaign: Any,
    campaign_execution: Mapping[str, Any],
    outcomes_by_task: Mapping[str, Mapping[str, Any]],
    rank_runtime: Sequence[Mapping[str, Any]],
    rank_count: int,
    require_fresh: bool,
) -> list[dict[str, Any]]:
    from negfpy.anharmonic.toy._mpi_production_gate import (
        _phase_manifest,
        _phase_slug,
    )

    phase_digests = campaign_execution.get("phase_execution_sha256")
    if (
        not isinstance(phase_digests, list)
        or len(phase_digests) != len(campaign.waves)
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in phase_digests
        )
    ):
        raise RemoteToyMPIRunnerError("phase-execution digest coverage changed.")
    execution_id = campaign_execution.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        raise RemoteToyMPIRunnerError("campaign execution_id is invalid.")
    expected_mode = "serial" if rank_count == 1 else "mpi"
    runtime_by_rank = {int(item["rank"]): item for item in rank_runtime}
    evidence: list[dict[str, Any]] = []
    for index, task_ids in enumerate(campaign.waves):
        phase_root = root / "phases" / _phase_slug(index, task_ids)
        expected_manifest = _phase_manifest(campaign, index)
        manifest_path = phase_root / "task-manifest.json"
        manifest_document = read_hashed_manifest(manifest_path)
        if manifest_document != expected_manifest.as_dict():
            raise RemoteToyMPIRunnerError(
                f"phase-{index} task manifest changed."
            )
        phase_id = f"{execution_id}.phase-{index:03d}"
        execution_path = phase_root / "executions" / f"{phase_id}.json"
        execution_document = read_hashed_manifest(execution_path)
        if execution_document["payload_sha256"] != phase_digests[index]:
            raise RemoteToyMPIRunnerError(
                f"phase-{index} execution SHA256 changed."
            )
        payload = execution_document["payload"]
        expected = {
            "schema_name": "negfpy_mpi_execution_manifest",
            "schema_version": 1,
            "execution_id": phase_id,
            "manifest_sha256": expected_manifest.manifest_sha256,
            "mode": expected_mode,
            "communicator_size": rank_count,
            "thread_limit": 1,
            "complete": True,
            "missing_task_ids": [],
            "infrastructure_errors": [],
        }
        changed = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        assignments = payload.get("assignments")
        if changed or not isinstance(assignments, list):
            raise RemoteToyMPIRunnerError(
                f"phase-{index} execution envelope changed: {changed}"
            )
        if [item.get("task_id") for item in assignments] != list(task_ids):
            raise RemoteToyMPIRunnerError(
                f"phase-{index} assignment topology changed."
            )
        phase_rank_cores: dict[tuple[str, int], tuple[int, int]] = {}
        for assignment in assignments:
            task_id = assignment["task_id"]
            current = _current_assignment_core(
                assignment,
                runtime_by_rank=runtime_by_rank,
                rank_count=rank_count,
                require_fresh=require_fresh,
            )
            if current is not None:
                rank_key, core = current
                previous_core = phase_rank_cores.setdefault(rank_key, core)
                if previous_core != core:
                    raise RemoteToyMPIRunnerError("inconsistent rank affinity.")
            outcome = outcomes_by_task[task_id]
            if (
                assignment.get("status") != "completed"
                or assignment.get("thread_cap_passed") is not True
                or assignment.get("checkpoint_file_sha256")
                != outcome.get("checkpoint_sha256")
                or assignment.get("artifact_sha256")
                != outcome.get("artifact_sha256")
            ):
                raise RemoteToyMPIRunnerError(
                    f"phase-{index} assignment evidence changed for {task_id}."
                )
        occupied_assignment_cores: dict[tuple[str, int, int], int] = {}
        for (hostname, rank), core in phase_rank_cores.items():
            key = (hostname, core[0], core[1])
            owner = occupied_assignment_cores.get(key)
            if owner is not None and owner != rank:
                raise RemoteToyMPIRunnerError(
                    "task MPI ranks share a physical core within one phase."
                )
            occupied_assignment_cores[key] = rank
        evidence.append(
            {
                "wave_index": index,
                "task_manifest_file_sha256": _file_sha256(manifest_path),
                "execution_manifest_file_sha256": _file_sha256(execution_path),
                "execution_payload_sha256": execution_document["payload_sha256"],
                "task_count": len(task_ids),
            }
        )
    return evidence


def _expected_rank_python_argv(
    *,
    profile: str,
    root: Path,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    execution_id: str,
    rank_count: int,
) -> list[str]:
    script = str(Path(__file__).resolve())
    if profile == "scaling":
        return [
            script,
            "_rank-worker",
            "--profile",
            "scaling",
            "--source-revision",
            source_revision,
            "--source-tree-sha256",
            source_tree_sha256,
            "--environment-sha256",
            environment_sha256,
            "--run-directory",
            str(root),
            "--execution-id",
            execution_id,
            "--expected-ranks",
            str(rank_count),
        ]
    if profile != "release":
        raise ValueError("profile must be scaling or release.")
    return [
        script,
        "release",
        "--run-directory",
        str(root),
        "--source-revision",
        source_revision,
        "--source-tree-sha256",
        source_tree_sha256,
        "--environment-sha256",
        environment_sha256,
        "--execution-id",
        execution_id,
        "--expected-ranks",
        str(rank_count),
    ]


def _validate_worker_artifact_chain(
    payload: Mapping[str, Any],
    *,
    worker_path: Path,
    rank_count: int,
    socket_count: int,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    profile: str,
) -> dict[str, Any]:
    from negfpy.anharmonic.toy._mpi_production_gate_physics import (
        production_toy_physics_sha256,
    )
    from negfpy.anharmonic.toy._mpi_production_gate import (
        collect_production_toy_campaign_outcomes,
    )

    result_path = _absolute_path_without_symlinks(
        worker_path, name="worker result"
    ).resolve(strict=True)
    if result_path.parent.name != "runner-results":
        raise RemoteToyMPIRunnerError("worker result is outside runner-results.")
    root = result_path.parent.parent
    execution_id = payload.get("execution_id")
    if not isinstance(execution_id, str) or not execution_id:
        raise RemoteToyMPIRunnerError("worker result has no execution_id.")
    expected_worker = root / "runner-results" / f"{execution_id}.json"
    if result_path != expected_worker:
        raise RemoteToyMPIRunnerError("worker result path is not canonical.")

    campaign, config = _expected_campaign(
        profile=profile,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
    )
    campaign_path = _safe_artifact_path(
        root,
        payload.get("campaign_manifest"),
        name="campaign manifest",
    )
    execution_path = _safe_artifact_path(
        root,
        payload.get("execution_manifest"),
        name="campaign execution manifest",
    )
    report_path = _safe_artifact_path(
        root,
        payload.get("strict_report"),
        name="strict production report",
    )
    if (
        campaign_path != root / "campaign-manifest.json"
        or execution_path
        != root / "campaign-executions" / f"{execution_id}.json"
        or report_path
        != root / "production-reports" / f"{execution_id}.json"
    ):
        raise RemoteToyMPIRunnerError("worker artifact paths are not canonical.")

    campaign_document = read_hashed_manifest(campaign_path)
    if campaign_document != campaign.manifest.as_dict():
        raise RemoteToyMPIRunnerError(
            "campaign manifest identity/topology differs from locked inputs."
        )
    execution_document = read_hashed_manifest(execution_path)
    execution = execution_document["payload"]
    expected_mode = "serial" if rank_count == 1 else "mpi"
    expected_execution = {
        "schema_name": "negfpy_protocol_v2_toy_mpi_execution",
        "schema_version": 1,
        "execution_id": execution_id,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "mode": expected_mode,
        "communicator_size": rank_count,
        "complete": True,
    }
    changed = {
        key: (execution.get(key), value)
        for key, value in expected_execution.items()
        if execution.get(key) != value
    }
    outcomes = execution.get("outcomes")
    if changed or not isinstance(outcomes, list):
        raise RemoteToyMPIRunnerError(
            f"campaign execution envelope changed: {changed}"
        )
    expected_task_ids = list(campaign.canonical_task_ids)
    if [item.get("task_id") for item in outcomes] != expected_task_ids:
        raise RemoteToyMPIRunnerError("campaign outcome topology changed.")
    outcomes_by_task: dict[str, Mapping[str, Any]] = {}
    referenced_paths: set[Path] = set()
    harmonic_artifact: Path | None = None
    for outcome in outcomes:
        task_id = str(outcome["task_id"])
        if (
            outcome.get("status") != "completed"
            or outcome.get("wave_index") != campaign.wave_index(task_id)
            or not isinstance(outcome.get("artifact_validation"), Mapping)
            or (profile == "scaling" and outcome.get("resumed") is not False)
        ):
            raise RemoteToyMPIRunnerError(
                f"campaign outcome is incomplete or changed for {task_id}."
            )
        checkpoint_path = _safe_referenced_path(
            root,
            outcome.get("checkpoint_relative_path"),
            outcome.get("checkpoint_sha256"),
            name=f"{task_id} checkpoint",
        )
        artifact_path = _safe_referenced_path(
            root,
            outcome.get("artifact_relative_path"),
            outcome.get("artifact_sha256"),
            name=f"{task_id} artifact",
        )
        if checkpoint_path in referenced_paths or artifact_path in referenced_paths:
            raise RemoteToyMPIRunnerError("campaign outcomes reuse an artifact path.")
        referenced_paths.update((checkpoint_path, artifact_path))
        outcomes_by_task[task_id] = outcome
        if task_id == "preflight.harmonic-same-grid":
            harmonic_artifact = artifact_path
    try:
        independently_loaded = collect_production_toy_campaign_outcomes(
            campaign, root
        )
    except Exception as exc:
        raise RemoteToyMPIRunnerError(
            "task checkpoints or HDF5 artifacts failed independent reload."
        ) from exc
    if len(independently_loaded) != len(outcomes):
        raise RemoteToyMPIRunnerError("independent outcome coverage changed.")
    for recorded, reloaded in zip(outcomes, independently_loaded, strict=True):
        expected_reloaded = reloaded.as_dict()
        expected_reloaded.pop("resumed", None)
        recorded_without_resume = dict(recorded)
        recorded_without_resume.pop("resumed", None)
        if recorded_without_resume != expected_reloaded:
            raise RemoteToyMPIRunnerError(
                "independently reloaded outcome differs for "
                f"{recorded.get('task_id')}."
            )
    scientific = _payload_sha256(
        {
            "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
            "outcomes": [
                {
                    "task_id": item["task_id"],
                    "status": item["status"],
                    "artifact_validation": item["artifact_validation"],
                }
                for item in outcomes
            ],
        }
    )
    if (
        scientific != execution.get("scientific_sha256")
        or scientific != payload.get("execution_scientific_sha256")
    ):
        raise RemoteToyMPIRunnerError("execution scientific SHA256 changed.")
    raw_runtime = payload.get("rank_runtime")
    if not isinstance(raw_runtime, list):
        raise RemoteToyMPIRunnerError("worker rank-runtime evidence is missing.")
    rank_runtime = _validate_rank_runtime(
        raw_runtime,
        rank_count=rank_count,
        socket_count=socket_count,
    )
    expected_rank_argv = _expected_rank_python_argv(
        profile=profile,
        root=root,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        execution_id=execution_id,
        rank_count=rank_count,
    )
    python_hashes: set[str] = set()
    for runtime in rank_runtime:
        runtime_python = runtime.get("python")
        if not isinstance(runtime_python, Mapping):
            raise RemoteToyMPIRunnerError("runtime Python identity is missing.")
        path, _ = _validate_executable_record(
            runtime_python,
            expected_name=Path(str(runtime_python.get("path"))).name,
        )
        if runtime.get("python_argv") != expected_rank_argv:
            raise RemoteToyMPIRunnerError(
                "MPI rank Python argv differs from the locked launcher program."
            )
        python_hashes.add(_file_sha256(path))
    if len(python_hashes) != 1:
        raise RemoteToyMPIRunnerError("MPI ranks used different Python binaries.")
    phase_evidence = _validate_phase_execution_evidence(
        root=root,
        campaign=campaign,
        campaign_execution=execution,
        outcomes_by_task=outcomes_by_task,
        rank_runtime=rank_runtime,
        rank_count=rank_count,
        require_fresh=profile == "scaling",
    )
    if harmonic_artifact is None:
        raise RemoteToyMPIRunnerError("harmonic preflight artifact is missing.")
    harmonic = _validate_real_harmonic_preflight(
        harmonic_artifact,
        profile=profile,
    )

    report, report_document = _strict_report(report_path)
    release_certified = profile == "release"
    if bool(report_document.get("summary", {}).get("release_certified")) != (
        release_certified
    ):
        raise RemoteToyMPIRunnerError("strict report certification state changed.")
    recomputed_physics = production_toy_physics_sha256(report)
    if recomputed_physics != payload.get("production_toy_physics_sha256"):
        raise RemoteToyMPIRunnerError(
            "worker physics SHA256 does not match the strict report."
        )
    orchestration = report_document.get("orchestration_provenance")
    expected_orchestration = {
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "campaign_manifest_file_sha256": _file_sha256(campaign_path),
        "execution_manifest_file_sha256": _file_sha256(execution_path),
        "execution_payload_sha256": execution_document["payload_sha256"],
        "execution_scientific_sha256": scientific,
        "execution_id": execution_id,
        "mode": expected_mode,
        "communicator_size": rank_count,
        "complete": True,
        "solver_and_gate_contract_unchanged": True,
    }
    if not isinstance(orchestration, Mapping) or any(
        orchestration.get(key) != value
        for key, value in expected_orchestration.items()
    ):
        raise RemoteToyMPIRunnerError(
            "strict report orchestration provenance changed."
        )
    if payload.get("configuration_sha256") != _payload_sha256(config.as_dict()):
        raise RemoteToyMPIRunnerError("worker configuration SHA256 changed.")
    memory = _validate_memory_record(
        payload.get("memory_preflight"),
        profile=profile,
        rank_count=rank_count,
    )
    return {
        "campaign_manifest_file_sha256": _file_sha256(campaign_path),
        "execution_manifest_file_sha256": _file_sha256(execution_path),
        "strict_report_file_sha256": _file_sha256(report_path),
        "strict_report_sha256": report.sha256,
        "production_toy_physics_sha256": recomputed_physics,
        "harmonic_preflight": harmonic,
        "memory_preflight": memory,
        "phase_evidence": phase_evidence,
        "rank_runtime": list(rank_runtime),
        "python_file_sha256": next(iter(python_hashes)),
    }


def _validate_scaling_worker_result(
    document: Mapping[str, Any],
    *,
    worker_path: Path,
    rank_count: int,
    socket_count: int,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
) -> dict[str, Any]:
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise RemoteToyMPIRunnerError("worker result has no payload.")
    expected = {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "scaling",
        "certification_eligible": False,
        "release_certified": False,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "expected_rank_count": rank_count,
        "recorded_rank_count": rank_count,
        "execution_id": f"scaling-rank-{rank_count:03d}",
        "execution_mode": "serial" if rank_count == 1 else "mpi",
        "directory_mode": "fresh",
        "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
        "task_count": SCALING_TASK_COUNT,
        "multidimensional_outcome_count": SCALING_MULTIDIMENSIONAL_COUNT,
        "open_system_case_count": SCALING_OPEN_COUNT,
        "continuation_path_count": SCALING_CONTINUATION_COUNT,
        "all_outcomes_completed": True,
    }
    changed = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if changed:
        raise RemoteToyMPIRunnerError(
            f"rank-{rank_count} worker result changed: {changed}"
        )
    digest = payload.get("production_toy_physics_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise RemoteToyMPIRunnerError("worker physics digest is invalid.")
    evidence = _validate_worker_artifact_chain(
        payload,
        worker_path=worker_path,
        rank_count=rank_count,
        socket_count=socket_count,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        profile="scaling",
    )
    if evidence["production_toy_physics_sha256"] != digest:
        raise RemoteToyMPIRunnerError("recomputed worker physics digest changed.")
    return {**payload, "_validated_evidence": evidence}

def validate_profile_worker_result(
    result_manifest: str | Path,
    *,
    profile: str,
    rank_count: int,
    socket_count: int,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    expected_execution_id: str,
    expected_directory_mode: str,
) -> dict[str, Any]:
    """Reopen a complete worker chain on the host that executed the ranks.

    CPU topology is intentionally remeasured from local sysfs, so callers must
    invoke this before fetching artifacts to a machine with different topology.
    """

    _validate_identity(
        source_revision,
        source_tree_sha256,
        environment_sha256,
    )
    execution_id = _validate_execution_id(expected_execution_id)
    if expected_directory_mode not in {"fresh", "exact_resume_candidate"}:
        raise ValueError("expected_directory_mode is invalid.")
    worker_path = _absolute_path_without_symlinks(
        result_manifest, name="worker result manifest"
    ).resolve(strict=True)
    document = read_hashed_manifest(worker_path)
    if profile == "scaling":
        if (
            execution_id != f"scaling-rank-{rank_count:03d}"
            or expected_directory_mode != "fresh"
        ):
            raise RemoteToyMPIRunnerError("scaling worker identity changed.")
        return _validate_scaling_worker_result(
            document,
            worker_path=worker_path,
            rank_count=rank_count,
            socket_count=socket_count,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
        )
    if profile != "release":
        raise ValueError("profile must be scaling or release.")
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise RemoteToyMPIRunnerError("worker result has no payload.")
    expected = {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "release",
        "certification_eligible": True,
        "release_certified": True,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "expected_rank_count": rank_count,
        "recorded_rank_count": rank_count,
        "execution_id": execution_id,
        "execution_mode": "serial" if rank_count == 1 else "mpi",
        "directory_mode": expected_directory_mode,
        "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
        "task_count": SCALING_TASK_COUNT,
        "multidimensional_outcome_count": SCALING_MULTIDIMENSIONAL_COUNT,
        "open_system_case_count": SCALING_OPEN_COUNT,
        "continuation_path_count": SCALING_CONTINUATION_COUNT,
        "all_outcomes_completed": True,
    }
    changed = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if changed:
        raise RemoteToyMPIRunnerError(
            f"release worker result changed: {changed}"
        )
    digest = payload.get("production_toy_physics_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise RemoteToyMPIRunnerError("worker physics digest is invalid.")
    evidence = _validate_worker_artifact_chain(
        payload,
        worker_path=worker_path,
        rank_count=rank_count,
        socket_count=socket_count,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        profile="release",
    )
    if evidence["production_toy_physics_sha256"] != digest:
        raise RemoteToyMPIRunnerError("recomputed worker physics digest changed.")
    return {**payload, "_validated_evidence": evidence}


def build_scaling_selection_payload(
    rank_manifests: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    physical_core_count: int,
    socket_count: int,
) -> dict[str, Any]:
    """Validate all fresh repetitions and apply the locked rank policy."""

    _validate_identity(
        source_revision,
        source_tree_sha256,
        environment_sha256,
    )
    if isinstance(socket_count, bool) or socket_count <= 0:
        raise ValueError("socket_count must be positive.")
    records: dict[
        int,
        tuple[Path, Mapping[str, Any], Mapping[str, Any]],
    ] = {}
    scaling_roots: set[Path] = set()
    for path, supplied_document in rank_manifests:
        manifest_path = _absolute_path_without_symlinks(
            path, name="rank manifest"
        ).resolve(strict=True)
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise RemoteToyMPIRunnerError("rank manifest is missing or unsafe.")
        document = read_hashed_manifest(manifest_path)
        if document != supplied_document:
            raise RemoteToyMPIRunnerError(
                "supplied rank manifest differs from its immutable file."
            )
        payload = document.get("payload")
        if not isinstance(payload, Mapping):
            raise RemoteToyMPIRunnerError("rank manifest has no payload.")
        rank_count = payload.get("rank_count")
        if isinstance(rank_count, bool) or not isinstance(rank_count, int):
            raise RemoteToyMPIRunnerError("rank manifest has invalid rank_count.")
        if rank_count in records:
            raise RemoteToyMPIRunnerError("duplicate rank manifest.")
        expected_fields = {
            "schema": "negfpy.remote_toy_mpi.scaling_rank.v1",
            "source_revision": source_revision,
            "source_tree_sha256": source_tree_sha256,
            "environment_sha256": environment_sha256,
            "fresh_isolated_repetition": True,
            "resume_used": False,
            "returncode": 0,
            "completed": True,
            "failure": None,
            "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
        }
        changed = {
            key: (payload.get(key), expected)
            for key, expected in expected_fields.items()
            if payload.get(key) != expected
        }
        if changed:
            raise RemoteToyMPIRunnerError(
                f"rank-{rank_count} manifest changed: {changed}"
            )
        scaling_root = manifest_path.parent
        scaling_roots.add(scaling_root)
        if manifest_path != scaling_root / f"rank-{rank_count:03d}-manifest.json":
            raise RemoteToyMPIRunnerError("rank manifest path is not canonical.")
        launcher_path, _ = _validate_executable_record(
            payload.get("launcher"),
            expected_name="mpiexec",
        )
        python_path, _ = _validate_executable_record(
            payload.get("python"),
            expected_name="python",
        )
        prefix = launcher_path.parent.parent
        if (
            launcher_path != prefix / "bin" / "mpiexec"
            or python_path != prefix / "bin" / "python"
        ):
            raise RemoteToyMPIRunnerError(
                "launcher and Python are not from one environment prefix."
            )
        binding = list(
            scaling.openmpi_binding_arguments(
                rank_count,
                physical_core_count=physical_core_count,
                socket_count=socket_count,
            )
        )
        repetition = scaling_root / f"rank-{rank_count:03d}"
        execution_id = f"scaling-rank-{rank_count:03d}"
        program = (
            str(python_path),
            str(Path(__file__).resolve()),
            "_rank-worker",
            "--profile",
            "scaling",
            "--source-revision",
            source_revision,
            "--source-tree-sha256",
            source_tree_sha256,
            "--environment-sha256",
            environment_sha256,
            "--run-directory",
            str(repetition),
            "--execution-id",
            execution_id,
            "--expected-ranks",
            str(rank_count),
        )
        expected_argv = list(
            scaling.environment_mpi_argv(
                prefix,
                rank_count=rank_count,
                physical_core_count=physical_core_count,
                socket_count=socket_count,
                program=program,
            )
        )
        if payload.get("argv") != expected_argv or payload.get(
            "binding_arguments"
        ) != binding:
            raise RemoteToyMPIRunnerError(
                "rank manifest launcher argv/binding changed."
            )
        _validate_memory_record(
            payload.get("launcher_memory_preflight"),
            profile="scaling",
            rank_count=rank_count,
        )
        _safe_artifact_path(
            scaling_root,
            payload.get("stdout"),
            name="stdout log",
        )
        _safe_artifact_path(
            scaling_root,
            payload.get("stderr"),
            name="stderr log",
        )
        worker_path = _safe_referenced_path(
            scaling_root,
            payload.get("worker_result_relative_path"),
            payload.get("worker_result_file_sha256"),
            name=f"rank-{rank_count} worker result",
        )
        if worker_path != repetition / "runner-results" / f"{execution_id}.json":
            raise RemoteToyMPIRunnerError("worker result path is not canonical.")
        worker_document = read_hashed_manifest(worker_path)
        worker_payload = _validate_scaling_worker_result(
            worker_document,
            worker_path=worker_path,
            rank_count=rank_count,
            socket_count=socket_count,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
        )
        if payload.get("worker_result_payload_sha256") != worker_document[
            "payload_sha256"
        ]:
            raise RemoteToyMPIRunnerError(
                "rank manifest changed the worker payload SHA256."
            )
        if payload.get("production_toy_physics_sha256") != worker_payload[
            "production_toy_physics_sha256"
        ]:
            raise RemoteToyMPIRunnerError(
                "rank and worker physics SHA256 values disagree."
            )
        worker_evidence = worker_payload["_validated_evidence"]
        if worker_evidence["python_file_sha256"] != payload["python"]["sha256"]:
            raise RemoteToyMPIRunnerError(
                "outer launcher and MPI ranks used different Python binaries."
            )
        records[rank_count] = (manifest_path, document, worker_payload)

    if len(scaling_roots) != 1:
        raise RemoteToyMPIRunnerError(
            "rank manifests do not share one isolated scaling root."
        )

    expected = scaling.required_rank_ladder(physical_core_count)
    if expected != scaling.RANK_LADDER:
        raise RemoteToyMPIRunnerError(
            "remote scaling requires at least 16 physical cores."
        )
    if tuple(sorted(records)) != expected:
        raise RemoteToyMPIRunnerError("rank manifest ladder is incomplete.")

    digests: dict[int, str] = {}
    measurements: list[scaling.ScalingMeasurement] = []
    rank_evidence: list[dict[str, Any]] = []
    for rank_count in expected:
        path, document, worker_payload = records[rank_count]
        payload = document["payload"]
        digest = payload.get("production_toy_physics_sha256")
        digests[rank_count] = str(digest)
        measurements.append(
            scaling.ScalingMeasurement(
                rank_count=rank_count,
                wall_seconds=float(payload.get("outer_wall_seconds")),
                completed=True,
                numerically_equal=True,
            )
        )
        rank_evidence.append(
            {
                "rank_count": rank_count,
                "manifest_path": path.name,
                "manifest_file_sha256": _file_sha256(path),
                "manifest_payload_sha256": document["payload_sha256"],
                "worker_result_file_sha256": payload[
                    "worker_result_file_sha256"
                ],
                "campaign_manifest_file_sha256": worker_payload[
                    "_validated_evidence"
                ]["campaign_manifest_file_sha256"],
                "execution_manifest_file_sha256": worker_payload[
                    "_validated_evidence"
                ]["execution_manifest_file_sha256"],
                "strict_report_file_sha256": worker_payload[
                    "_validated_evidence"
                ]["strict_report_file_sha256"],
                "harmonic_preflight": worker_payload["_validated_evidence"][
                    "harmonic_preflight"
                ],
                "launcher_memory_preflight": payload["launcher_memory_preflight"],
                "worker_memory_preflight": worker_payload["memory_preflight"],
                "rank_runtime": worker_payload["rank_runtime"],
            }
        )
    common_digest = scaling.require_identical_physics_digests(
        digests,
        physical_core_count=physical_core_count,
    )
    selection = scaling.select_rank_count(
        measurements,
        physical_core_count=physical_core_count,
    )
    return {
        "schema": "negfpy.remote_toy_mpi.scaling_selection.v1",
        "created_utc": _utc_now(),
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "physical_core_count": physical_core_count,
        "socket_count": socket_count,
        "profile": "representative_noncertifying_full_28_task",
        "production_toy_physics_sha256": common_digest,
        "physics_equal_across_all_ranks": True,
        "fresh_isolated_repetitions": True,
        "selection": selection.as_dict(),
        "toy_execution": {
            "rank_count": selection.toy_rank_count,
            "mode": "task_mpi" if selection.toy_mpi_enabled else "serial_fft",
            "minimum_speedup_satisfied": selection.toy_mpi_enabled,
        },
        "silicon_execution": {
            "rank_count": selection.silicon_rank_count,
            "mode": "task_mpi",
        },
        "rank_evidence": rank_evidence,
    }


def run_scaling_ladder(
    *,
    environment_prefix: str | Path,
    run_directory: str | Path,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    physical_core_count: int,
    socket_count: int,
) -> Path:
    """Run five fresh rank repetitions and publish the selection manifest."""

    _validate_identity(
        source_revision,
        source_tree_sha256,
        environment_sha256,
    )
    expected = scaling.required_rank_ladder(physical_core_count)
    if expected != scaling.RANK_LADDER:
        raise RemoteToyMPIRunnerError(
            "the locked scaling ladder requires at least 16 physical cores."
        )
    if isinstance(socket_count, bool) or socket_count <= 0:
        raise ValueError("socket_count must be positive.")
    root = _absolute_path_without_symlinks(
        run_directory, name="scaling run_directory"
    ).resolve()
    if root.exists():
        raise RemoteToyMPIRunnerError(
            "scaling run_directory must not already exist."
        )
    root.mkdir(parents=True)
    _fsync_directory(root.parent)

    prefix = Path(environment_prefix).expanduser().resolve()
    launcher = prefix / "bin" / "mpiexec"
    python = prefix / "bin" / "python"
    launcher_identity = _launcher_record(launcher)
    python_identity = _launcher_record(python)
    script = Path(__file__).resolve()
    environment = _thread_environment()
    rank_manifests: list[tuple[Path, Mapping[str, Any]]] = []
    logs = root / "logs"
    logs.mkdir()

    for rank_count in expected:
        repetition = root / f"rank-{rank_count:03d}"
        execution_id = f"scaling-rank-{rank_count:03d}"
        program = (
            str(python),
            str(script),
            "_rank-worker",
            "--profile",
            "scaling",
            "--source-revision",
            source_revision,
            "--source-tree-sha256",
            source_tree_sha256,
            "--environment-sha256",
            environment_sha256,
            "--run-directory",
            str(repetition),
            "--execution-id",
            execution_id,
            "--expected-ranks",
            str(rank_count),
        )
        argv = scaling.environment_mpi_argv(
            prefix,
            rank_count=rank_count,
            physical_core_count=physical_core_count,
            socket_count=socket_count,
            program=program,
        )
        stdout_path = logs / f"rank-{rank_count:03d}.stdout.log"
        stderr_path = logs / f"rank-{rank_count:03d}.stderr.log"
        launcher_memory_preflight = campaign_dense_memory_preflight(
            profile="scaling",
            rank_count=rank_count,
        )
        started_utc = _utc_now()
        started = time.perf_counter()
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            completed = subprocess.run(
                argv,
                check=False,
                cwd=str(script.parents[1]),
                env=environment,
                stdout=stdout,
                stderr=stderr,
            )
            stdout.flush()
            stderr.flush()
            os.fsync(stdout.fileno())
            os.fsync(stderr.fileno())
        wall_seconds = time.perf_counter() - started
        worker_path = repetition / "runner-results" / f"{execution_id}.json"
        worker_document: dict[str, Any] | None = None
        worker_payload: dict[str, Any] | None = None
        failure: str | None = None
        if completed.returncode == 0:
            try:
                worker_document = read_hashed_manifest(worker_path)
                worker_payload = _validate_scaling_worker_result(
                    worker_document,
                    worker_path=worker_path,
                    rank_count=rank_count,
                    socket_count=socket_count,
                    source_revision=source_revision,
                    source_tree_sha256=source_tree_sha256,
                    environment_sha256=environment_sha256,
                )
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
        else:
            failure = f"mpiexec returned {completed.returncode}"

        rank_payload: dict[str, Any] = {
            "schema": "negfpy.remote_toy_mpi.scaling_rank.v1",
            "started_utc": started_utc,
            "finished_utc": _utc_now(),
            "rank_count": rank_count,
            "source_revision": source_revision,
            "source_tree_sha256": source_tree_sha256,
            "environment_sha256": environment_sha256,
            "fresh_isolated_repetition": True,
            "resume_used": False,
            "outer_wall_seconds": wall_seconds,
            "returncode": completed.returncode,
            "completed": failure is None,
            "failure": failure,
            "argv": list(argv),
            "binding_arguments": list(
                scaling.openmpi_binding_arguments(
                    rank_count,
                    physical_core_count=physical_core_count,
                    socket_count=socket_count,
                )
            ),
            "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
            "launcher_memory_preflight": launcher_memory_preflight,
            "launcher": launcher_identity,
            "python": python_identity,
            "stdout": {
                "relative_path": stdout_path.relative_to(root).as_posix(),
                "sha256": _file_sha256(stdout_path),
            },
            "stderr": {
                "relative_path": stderr_path.relative_to(root).as_posix(),
                "sha256": _file_sha256(stderr_path),
            },
            "worker_result_relative_path": (
                None
                if worker_document is None
                else worker_path.relative_to(root).as_posix()
            ),
            "worker_result_file_sha256": (
                None if worker_document is None else _file_sha256(worker_path)
            ),
            "worker_result_payload_sha256": (
                None
                if worker_document is None
                else worker_document["payload_sha256"]
            ),
            "production_toy_physics_sha256": (
                None
                if worker_payload is None
                else worker_payload["production_toy_physics_sha256"]
            ),
        }
        rank_manifest_path = root / f"rank-{rank_count:03d}-manifest.json"
        rank_document = write_hashed_manifest(rank_manifest_path, rank_payload)
        rank_manifests.append((rank_manifest_path, rank_document))
        if failure is not None:
            raise RemoteToyMPIRunnerError(
                f"rank-{rank_count} repetition failed: {failure}"
            )

    selection_payload = build_scaling_selection_payload(
        rank_manifests,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        physical_core_count=physical_core_count,
        socket_count=socket_count,
    )
    selection_path = root / "scaling-selection.json"
    write_hashed_manifest(selection_path, selection_payload)
    return selection_path


def _add_identity_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--run-directory", required=True, type=Path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scale_parser = subparsers.add_parser(
        "scale",
        help="run the five fresh noncertifying rank repetitions",
    )
    _add_identity_arguments(scale_parser)
    scale_parser.add_argument("--environment-prefix", required=True, type=Path)
    scale_parser.add_argument("--physical-core-count", required=True, type=int)
    scale_parser.add_argument("--socket-count", required=True, type=int)

    release_parser = subparsers.add_parser(
        "release",
        help="run the full locked release gate under an existing mpiexec",
    )
    _add_identity_arguments(release_parser)
    release_parser.add_argument("--execution-id", required=True)
    release_parser.add_argument("--expected-ranks", required=True, type=int)

    worker_parser = subparsers.add_parser("_rank-worker")
    _add_identity_arguments(worker_parser)
    worker_parser.add_argument(
        "--profile", required=True, choices=("scaling", "release")
    )
    worker_parser.add_argument("--execution-id", required=True)
    worker_parser.add_argument("--expected-ranks", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "scale":
            target = run_scaling_ladder(
                environment_prefix=arguments.environment_prefix,
                run_directory=arguments.run_directory,
                source_revision=arguments.source_revision,
                source_tree_sha256=arguments.source_tree_sha256,
                environment_sha256=arguments.environment_sha256,
                physical_core_count=arguments.physical_core_count,
                socket_count=arguments.socket_count,
            )
            print(str(target))
            return 0
        return 0 if execute_profile_under_mpi(
            profile=(
                "release" if arguments.command == "release" else arguments.profile
            ),
            source_revision=arguments.source_revision,
            source_tree_sha256=arguments.source_tree_sha256,
            environment_sha256=arguments.environment_sha256,
            run_directory=arguments.run_directory,
            execution_id=arguments.execution_id,
            expected_ranks=arguments.expected_ranks,
        ) else 1
    except (RemoteToyMPIRunnerError, scaling.MPIScalingError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
