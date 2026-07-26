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

    source = Path(path).expanduser().resolve()
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
    root = Path(run_directory).expanduser().resolve()
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


def _validate_scaling_worker_result(
    document: Mapping[str, Any],
    *,
    rank_count: int,
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
    return payload


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
    records: dict[int, tuple[Path, Mapping[str, Any]]] = {}
    for path, supplied_document in rank_manifests:
        manifest_path = Path(path).expanduser().resolve()
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
        worker_relative = payload.get("worker_result_relative_path")
        if not isinstance(worker_relative, str) or not worker_relative:
            raise RemoteToyMPIRunnerError("rank manifest lacks worker result.")
        worker_path = (manifest_path.parent / worker_relative).resolve()
        try:
            worker_path.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise RemoteToyMPIRunnerError(
                "worker result escaped the scaling directory."
            ) from exc
        recorded_worker_sha = payload.get("worker_result_file_sha256")
        if (
            not isinstance(recorded_worker_sha, str)
            or _SHA256.fullmatch(recorded_worker_sha) is None
            or not worker_path.is_file()
            or _file_sha256(worker_path) != recorded_worker_sha
        ):
            raise RemoteToyMPIRunnerError(
                "worker result file is missing or hash-mismatched."
            )
        worker_document = read_hashed_manifest(worker_path)
        worker_payload = _validate_scaling_worker_result(
            worker_document,
            rank_count=rank_count,
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
        records[rank_count] = (manifest_path, document)

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
        path, document = records[rank_count]
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
    root = Path(run_directory).expanduser().resolve()
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
                    rank_count=rank_count,
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
