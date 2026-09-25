#!/usr/bin/env python3
"""Fail-closed non-SCBA qualification for an exact material candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

import run_material_nq1_release_mpi as release
import run_sparse_nq1_phase_c_born_campaign as common
import validate_sparse_nq1_phase_c_scba as case
from negfpy.anharmonic._material_nq1_campaign import silicon_nq1_release_tasks
from negfpy.anharmonic._material_nq1_validation import (
    build_observation_mode_set,
    check_harmonic_embedding_buffer,
)
from negfpy.anharmonic._provenance import code_provenance
from negfpy.anharmonic._launch_environment import (
    build_launch_environment, qualification_environment,
)


class QualificationError(RuntimeError):
    """A non-scientific candidate qualification check failed."""


QUALIFICATION_TEST_TARGETS = (
    "tests",
    "docs/validation/bulk-rta-report/tests",
)
QUALIFICATION_CONTEXT_SCHEMA = "negfpy.material-nq1-qualification-context.v2"
CHECK_IDENTITY_SCHEMA = "negfpy.material-nq1-qualification-check-identity.v3"
CHECK_RECEIPT_SCHEMA = "negfpy.material-nq1-qualification-check.v3"
QUALIFICATION_CONTEXT_KEYS = {
    "schema", "repo_root", "source", "source_tree_sha256",
    "environment_lock", "scientific_task_set_sha256",
    "material_input_hashes", "runtimes", "mpiexec",
    "tracked_test_files", "driver_files", "qualification_context_sha256",
}
CHECK_RECORD_KEYS = {
    "command", "returncode", "path", "sha256", "size",
    "receipt_path", "receipt_sha256", "identity_sha256", "attempt",
    "reused",
}
QUALIFICATION_DRIVER_PATHS = (
    "pyproject.toml",
    "scripts/qualify_material_nq1_candidate.py",
    "scripts/qualify_material_nq1_predictor_candidate.py",
    "scripts/run_material_nq1_release_mpi.py",
    "scripts/validate_material_nq1_release.py",
    "scripts/validate_sparse_nq1_phase_c_scba.py",
    "tests/mpi/run_material_nq1_mpi_smoke.py",
    "src/negfpy/anharmonic/_launch_environment.py",
    "scripts/run_material_nq1_predictor_pilot.py",
    "scripts/run_sparse_nq1_phase_c_scba_campaign.py",
    "scripts/run_sparse_nq1_phase_c_born_campaign.py",
)


def _regular_file(path: Path, label: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise QualificationError(f"{label} is unavailable: {path}") from error
    if not stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        raise QualificationError(f"{label} is not a non-symlink regular file")
    return path


def _repo_root(path: Path) -> Path:
    try:
        return release._absolute_nonsymlink_path(
            path, label="repository root", directory=True
        )
    except release.MaterialCampaignError as error:
        raise QualificationError(str(error)) from error


def _evidence_directory(path: Path) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise QualificationError(
            f"qualification evidence directory is unavailable: {path}"
        ) from error
    if not stat.S_ISDIR(mode) or stat.S_ISLNK(mode):
        raise QualificationError(
            "qualification evidence root must be a non-symlink directory"
        )
    return path


def _qualification_output_paths(path: Path) -> tuple[Path, Path]:
    """Return lexical output/evidence paths after rejecting symlink ancestry."""

    output = Path(os.path.abspath(path.expanduser()))
    try:
        parent = release._absolute_nonsymlink_path(
            output.parent, label="qualification output parent", directory=True
        )
    except release.MaterialCampaignError as error:
        raise QualificationError(str(error)) from error
    if output.exists() or output.is_symlink():
        selected_output = _regular_file(output, "qualification output")
        if stat.S_IMODE(selected_output.stat().st_mode) != 0o444:
            raise QualificationError(
                "existing qualification output must have mode 0444"
            )
    evidence_root = parent / "qualification-evidence"
    if not evidence_root.exists() and not evidence_root.is_symlink():
        try:
            evidence_root.mkdir()
        except FileExistsError:
            pass
    try:
        evidence_root = release._absolute_nonsymlink_path(
            evidence_root,
            label="qualification evidence directory",
            directory=True,
        )
    except release.MaterialCampaignError as error:
        raise QualificationError(str(error)) from error
    return output, evidence_root


def _file_record(path: Path, *, relative_path: str) -> dict[str, Any]:
    selected = _regular_file(path, relative_path)
    return {
        "relative_path": relative_path,
        "sha256": release._sha256(selected),
        "size": selected.stat().st_size,
    }


def _tracked_test_files(repo: Path) -> dict[str, dict[str, Any]]:
    completed = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "--", *QUALIFICATION_TEST_TARGETS],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise QualificationError("cannot enumerate tracked qualification tests")
    paths = tuple(line for line in completed.stdout.splitlines() if line)
    if not paths:
        raise QualificationError("qualification test target set is empty")
    return {
        relative: _file_record(repo / relative, relative_path=relative)
        for relative in paths
    }


def _context_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(release._canonical(payload)).hexdigest()


def qualification_context_record(
    *,
    repo: Path,
    source: Mapping[str, Any],
    source_tree_sha256: str,
    environment_lock: Mapping[str, Any],
    scientific_task_set_sha256: str,
    material_input_hashes: Mapping[str, str],
    runtimes: Mapping[str, Mapping[str, Any]],
    mpiexec: Path,
) -> dict[str, Any]:
    """Freeze every byte/identity that can make a check result reusable."""

    driver_files = {
        relative: _file_record(repo / relative, relative_path=relative)
        for relative in QUALIFICATION_DRIVER_PATHS
    }
    payload = {
        "schema": QUALIFICATION_CONTEXT_SCHEMA,
        "repo_root": str(repo),
        "source": dict(source),
        "source_tree_sha256": source_tree_sha256,
        "environment_lock": dict(environment_lock),
        "scientific_task_set_sha256": scientific_task_set_sha256,
        "material_input_hashes": dict(sorted(material_input_hashes.items())),
        "runtimes": {name: dict(value) for name, value in sorted(runtimes.items())},
        "mpiexec": {
            "path": str(mpiexec),
            "sha256": release._sha256(_regular_file(mpiexec, "mpiexec")),
        },
        "tracked_test_files": _tracked_test_files(repo),
        "driver_files": driver_files,
    }
    return payload | {"qualification_context_sha256": _context_digest(payload)}


def _validate_context_digest(context: Mapping[str, Any]) -> str:
    if (
        set(context) != QUALIFICATION_CONTEXT_KEYS
        or context.get("schema") != QUALIFICATION_CONTEXT_SCHEMA
    ):
        raise QualificationError("qualification context schema is invalid")
    payload = dict(context)
    recorded = payload.pop("qualification_context_sha256", None)
    if not isinstance(recorded, str) or _context_digest(payload) != recorded:
        raise QualificationError("qualification context digest is invalid")
    return recorded


def _publish_log(path: Path, payload: bytes) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        selected = _regular_file(path, "qualification evidence")
        if selected.read_bytes() != payload:
            raise QualificationError(f"existing qualification evidence differs: {path}")
    else:
        common._publish_once(path, payload)
    return {
        "path": path.name,
        "sha256": release._sha256(path),
        "size": path.stat().st_size,
    }


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Path,
    label: str,
    qualification_context: Mapping[str, Any],
    identity_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    context_sha256 = _validate_context_digest(qualification_context)
    recorded_command = list(command if identity_command is None else identity_command)
    selected_environment = dict(sorted(environment.items()))
    identity_payload = {
        "schema": CHECK_IDENTITY_SCHEMA,
        "label": label,
        "command": recorded_command,
        "cwd": str(cwd.resolve()),
        "environment": selected_environment,
        "qualification_context_sha256": context_sha256,
    }
    identity_sha256 = hashlib.sha256(
        release._canonical(identity_payload)
    ).hexdigest()
    stem = log.stem
    suffix = log.suffix
    prior_receipts = sorted(log.parent.glob(f"{stem}*.receipt.json"))
    for receipt_path in prior_receipts:
        try:
            _regular_file(receipt_path, f"qualification receipt for {label}")
            receipt = release._load_canonical(
                receipt_path, f"qualification receipt for {label}"
            )
        except Exception:
            continue
        receipt_keys = {
            "schema", "identity", "identity_sha256", "attempt", "log_path",
            "log_sha256", "log_size", "returncode",
        }
        if (
            set(receipt) != receipt_keys
            or receipt.get("schema") != CHECK_RECEIPT_SCHEMA
            or receipt.get("identity_sha256") != identity_sha256
            or receipt.get("identity") != identity_payload
            or hashlib.sha256(release._canonical(identity_payload)).hexdigest()
            != receipt.get("identity_sha256")
            or not isinstance(receipt.get("attempt"), int)
            or isinstance(receipt.get("attempt"), bool)
            or receipt.get("attempt", 0) < 1
            or receipt.get("returncode") != 0
        ):
            continue
        archived_name = receipt.get("log_path")
        if (
            not isinstance(archived_name, str)
            or Path(archived_name).name != archived_name
        ):
            continue
        archived_log = log.parent / archived_name
        expected_receipt = archived_log.with_suffix(".receipt.json")
        expected_log = (
            log if receipt["attempt"] == 1
            else log.with_name(
                f"{stem}-attempt-{receipt['attempt']:03d}{suffix}"
            )
        )
        if archived_log != expected_log or receipt_path != expected_receipt:
            continue
        try:
            selected_log = _regular_file(archived_log, f"{label} log")
            log_sha256 = release._sha256(selected_log)
            log_size = selected_log.stat().st_size
        except QualificationError:
            continue
        if (
            log_sha256 != receipt.get("log_sha256")
            or log_size != receipt.get("log_size")
        ):
            continue
        return {
            "command": recorded_command,
            "returncode": 0,
            "path": archived_log.name,
            "sha256": log_sha256,
            "size": log_size,
            "receipt_path": receipt_path.name,
            "receipt_sha256": release._sha256(receipt_path),
            "identity_sha256": identity_sha256,
            "attempt": receipt["attempt"],
            "reused": True,
        }
    attempt = 1
    while True:
        attempt_log = (
            log if attempt == 1
            else log.with_name(f"{stem}-attempt-{attempt:03d}{suffix}")
        )
        receipt_path = attempt_log.with_suffix(".receipt.json")
        if not (
            attempt_log.exists() or attempt_log.is_symlink()
            or receipt_path.exists() or receipt_path.is_symlink()
        ):
            break
        attempt += 1
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    record = _publish_log(attempt_log, completed.stdout)
    record.update({
        "command": recorded_command,
        "returncode": int(completed.returncode),
    })
    receipt = {
        "schema": CHECK_RECEIPT_SCHEMA,
        "identity": identity_payload,
        "identity_sha256": identity_sha256,
        "attempt": attempt,
        "log_path": attempt_log.name,
        "log_sha256": record["sha256"],
        "log_size": record["size"],
        "returncode": int(completed.returncode),
    }
    common._publish_once(receipt_path, release._canonical(receipt))
    record.update({
        "receipt_path": receipt_path.name,
        "receipt_sha256": release._sha256(receipt_path),
        "identity_sha256": identity_sha256,
        "attempt": attempt,
        "reused": False,
    })
    if completed.returncode != 0:
        raise QualificationError(f"{label} failed; see {attempt_log}")
    return record


def _evidence_file(root: Path, name: Any, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise QualificationError(f"{label} has an unsafe evidence path")
    return _regular_file(root / name, label)


def _validate_file_map(
    repo: Path, records: Any, *, label: str
) -> None:
    if not isinstance(records, Mapping) or not records:
        raise QualificationError(f"{label} record is empty")
    for relative, record in records.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(record, Mapping)
            or set(record) != {"relative_path", "sha256", "size"}
            or record.get("relative_path") != relative
        ):
            raise QualificationError(f"{label} record is invalid")
        if _file_record(repo / relative, relative_path=relative) != record:
            raise QualificationError(f"{label} bytes changed: {relative}")


def _validate_check_record(
    record: Any,
    *,
    repo: Path,
    evidence_root: Path,
    context_sha256: str,
    expected_label: str,
    expected_runtime_sha256: str,
    expected_environment: Mapping[str, str],
    expected_command: Sequence[str],
    expected_log_name: str,
    expected_record_keys: set[str],
) -> None:
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_record_keys
        or record.get("command") != list(expected_command)
        or record.get("returncode") != 0
        or not isinstance(record.get("size"), int)
        or isinstance(record.get("size"), bool)
        or record.get("size", -1) < 0
        or not isinstance(record.get("attempt"), int)
        or isinstance(record.get("attempt"), bool)
        or record.get("attempt", 0) < 1
        or not isinstance(record.get("reused"), bool)
    ):
        raise QualificationError(f"{expected_label} evidence schema is invalid")
    attempt = record["attempt"]
    expected_log = Path(expected_log_name)
    if attempt != 1:
        expected_log = expected_log.with_name(
            f"{expected_log.stem}-attempt-{attempt:03d}{expected_log.suffix}"
        )
    expected_receipt = expected_log.with_suffix(".receipt.json")
    if (
        record.get("path") != expected_log.name
        or record.get("receipt_path") != expected_receipt.name
    ):
        raise QualificationError(f"{expected_label} evidence path is invalid")
    log = _evidence_file(evidence_root, record.get("path"), f"{expected_label} log")
    receipt_path = _evidence_file(
        evidence_root, record.get("receipt_path"), f"{expected_label} receipt"
    )
    if (
        release._sha256(log) != record.get("sha256")
        or log.stat().st_size != record.get("size")
        or release._sha256(receipt_path) != record.get("receipt_sha256")
    ):
        raise QualificationError(f"{expected_label} outer evidence hash changed")
    receipt = release._load_canonical(
        receipt_path, f"{expected_label} qualification receipt"
    )
    receipt_keys = {
        "schema", "identity", "identity_sha256", "attempt", "log_path",
        "log_sha256", "log_size", "returncode",
    }
    identity = receipt.get("identity")
    identity_keys = {
        "schema", "label", "command", "cwd", "environment",
        "qualification_context_sha256",
    }
    environment = identity.get("environment") if isinstance(identity, Mapping) else None
    if (
        set(receipt) != receipt_keys
        or receipt.get("schema") != CHECK_RECEIPT_SCHEMA
        or not isinstance(identity, Mapping)
        or set(identity) != identity_keys
        or identity.get("schema") != CHECK_IDENTITY_SCHEMA
        or identity.get("label") != expected_label
        or identity.get("qualification_context_sha256") != context_sha256
        or identity.get("command") != record.get("command")
        or receipt.get("identity_sha256") != record.get("identity_sha256")
        or hashlib.sha256(release._canonical(identity)).hexdigest()
        != receipt.get("identity_sha256")
        or receipt.get("attempt") != record.get("attempt")
        or receipt.get("log_path") != record.get("path")
        or receipt.get("log_sha256") != record.get("sha256")
        or receipt.get("log_size") != record.get("size")
        or receipt.get("returncode") != 0
        or not isinstance(environment, Mapping)
        or dict(environment) != dict(expected_environment)
        or environment.get("NEGFPY_QUALIFICATION_RUNTIME_SHA256")
        != expected_runtime_sha256
        or environment.get("PYTHONHASHSEED") != "0"
        or any(
            environment.get(name) != "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        )
        or not isinstance(environment.get("PYTHONPATH"), str)
        or environment["PYTHONPATH"].split(os.pathsep)[:2]
        != [str(repo / "src"), str(repo / "scripts")]
        or not isinstance(environment.get("PATH"), str)
        or identity.get("cwd") != str(repo)
    ):
        raise QualificationError(f"{expected_label} receipt does not authenticate")


def validate_qualification_evidence(
    document: Mapping[str, Any], *, repo: Path, evidence_root: Path
) -> dict[str, Any]:
    """Recursively authenticate a base v2 report and all referenced bytes."""

    repo = _repo_root(repo)
    evidence_root = _evidence_directory(evidence_root)
    expected_top = {
        "schema", "source", "source_tree_sha256", "environment_sha256",
        "scientific_controls", "checks", "evidence", "release_preflight_passed",
    }
    evidence = document.get("evidence")
    records = evidence.get("records") if isinstance(evidence, Mapping) else None
    context = (
        evidence.get("qualification_context")
        if isinstance(evidence, Mapping)
        else None
    )
    if (
        set(document) != expected_top
        or document.get("schema") != release.QUALIFICATION_SCHEMA
        or not isinstance(evidence, Mapping)
        or set(evidence) != {"qualification_context", "python_versions", "records"}
        or not isinstance(context, Mapping)
        or not isinstance(records, Mapping)
    ):
        raise QualificationError("base qualification report schema is invalid")
    context_sha256 = _validate_context_digest(context)
    controls = document.get("scientific_controls")
    source = document.get("source")
    if (
        not isinstance(source, Mapping)
        or set(source) != {"commit", "tree", "dirty", "package_file"}
        or source.get("dirty") is not False
        or source != common._source_identity(repo, str(source.get("commit")))
    ):
        raise QualificationError("qualified source identity is not current")
    current_tree = code_provenance().get("source_tree_sha256")
    if (
        context.get("repo_root") != str(repo)
        or context.get("source") != document.get("source")
        or context.get("source_tree_sha256") != document.get("source_tree_sha256")
        or document.get("source_tree_sha256") != current_tree
        or context.get("environment_lock", {}).get("sha256")
        != document.get("environment_sha256")
        or not isinstance(controls, Mapping)
        or context.get("scientific_task_set_sha256")
        != controls.get("task_set_sha256")
        or controls != release._scientific_control_record()
    ):
        raise QualificationError("qualification context identity changed")
    lock = context.get("environment_lock")
    if (
        not isinstance(lock, Mapping)
        or set(lock) != {"path", "sha256", "format"}
        or lock.get("format") != "@EXPLICIT"
    ):
        raise QualificationError("qualification environment lock is invalid")
    lock_path = _regular_file(Path(str(lock.get("path"))), "environment lock")
    if common._validate_lock(lock_path) != lock:
        raise QualificationError("qualification environment lock bytes changed")
    current_tests = _tracked_test_files(repo)
    if context.get("tracked_test_files") != current_tests:
        raise QualificationError("qualification test bytes changed")
    _validate_file_map(repo, context.get("driver_files"), label="driver file")
    if set(context["driver_files"]) != set(QUALIFICATION_DRIVER_PATHS):
        raise QualificationError("qualification driver set changed")

    material_hashes = context.get("material_input_hashes")
    expected_material_hashes = {
        "phase_a_fixture", "raw_fc3", "phase_b_artifact_set",
        "phase_b_mapped_hdf5", "phase_b_campaign_manifest",
    }
    if (
        not isinstance(material_hashes, Mapping)
        or set(material_hashes) != expected_material_hashes
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in material_hashes.values()
        )
    ):
        raise QualificationError("material input hash set is invalid")

    mpiexec = context.get("mpiexec")
    if (
        not isinstance(mpiexec, Mapping)
        or set(mpiexec) != {"path", "sha256"}
        or release._sha256(
            _regular_file(Path(str(mpiexec.get("path"))), "mpiexec")
        ) != mpiexec.get("sha256")
    ):
        raise QualificationError("mpiexec identity changed")

    runtimes = context.get("runtimes")
    versions = evidence.get("python_versions")
    if (
        not isinstance(runtimes, Mapping)
        or set(runtimes) != {"python310", "python312"}
        or not isinstance(versions, Mapping)
        or set(versions) != {"python310", "python312"}
    ):
        raise QualificationError("qualification runtime set is invalid")
    expected_record_names = {
        "python310_runtime", "python310_pytest", "python312_runtime",
        "python312_pytest", "mpi_ranks_2", "mpi_ranks_4",
        "material_input_preflight", "resources",
    }
    if set(records) != expected_record_names:
        raise QualificationError("qualification record set is invalid")
    for label in ("python310", "python312"):
        runtime = records.get(f"{label}_runtime")
        runtime_keys = {
            "schema", "label", "version", "executable", "executable_sha256",
            "packages_sha256", "runtime_sha256", "package_manifest", "launch_environment",
        }
        if (
            not isinstance(runtime, Mapping)
            or set(runtime) != runtime_keys
            or runtime != runtimes[label]
        ):
            raise QualificationError(f"{label} runtime context changed")
        runtime_payload = {
            name: runtime[name] for name in (
                "schema", "label", "version", "executable",
                "executable_sha256", "packages_sha256", "launch_environment",
            )
        }
        package = runtime.get("package_manifest")
        package_path = _evidence_file(
            evidence_root,
            package.get("path") if isinstance(package, Mapping) else None,
            f"{label} package manifest",
        )
        if (
            runtime.get("schema") != "negfpy.material-nq1-python-runtime.v2"
            or runtime.get("label") != label
            or runtime.get("version") != versions[label]
            or hashlib.sha256(release._canonical(runtime_payload)).hexdigest()
            != runtime.get("runtime_sha256")
            or release._sha256(
                _regular_file(Path(str(runtime.get("executable"))), f"{label} executable")
            ) != runtime.get("executable_sha256")
            or not isinstance(package, Mapping)
            or set(package) != {"path", "sha256", "size"}
            or release._sha256(package_path) != package.get("sha256")
            or package_path.stat().st_size != package.get("size")
            or not isinstance(package.get("size"), int)
            or isinstance(package.get("size"), bool)
            or package.get("size", 0) <= 0
            or package.get("sha256") != runtime.get("packages_sha256")
        ):
            raise QualificationError(f"{label} runtime evidence did not authenticate")
        try:
            expected_environment = qualification_environment(runtime, repo=repo)
        except ValueError as error:
            raise QualificationError(f"{label} launch environment is invalid") from error
        expected_pytest_command = [
            str(runtime["executable"]), "-m", "pytest", "-q",
            *QUALIFICATION_TEST_TARGETS,
        ]
        _validate_check_record(
            records.get(f"{label}_pytest"),
            repo=repo,
            evidence_root=evidence_root,
            context_sha256=context_sha256,
            expected_label=f"full {label} test suite",
            expected_runtime_sha256=str(runtime.get("runtime_sha256")),
            expected_environment=expected_environment,
            expected_command=expected_pytest_command,
            expected_log_name=f"{label}-pytest.log",
            expected_record_keys=CHECK_RECORD_KEYS | {"scope", "excluded_tests"},
        )
        suite = records[f"{label}_pytest"]
        if (
            suite.get("scope") != list(QUALIFICATION_TEST_TARGETS)
            or suite.get("excluded_tests") != []
        ):
            raise QualificationError(f"{label} full-suite scope changed")
    for ranks in (2, 4):
        mpi = records.get(f"mpi_ranks_{ranks}")
        expected_mpi_command = [
            str(mpiexec["path"]), "--bind-to", "core", "-n", str(ranks),
            str(runtimes["python312"]["executable"]),
            str(repo / "tests/mpi/run_material_nq1_mpi_smoke.py"),
            "<temporary-mpi-output>", "--execution-id",
            f"qualification-ranks-{ranks}",
        ]
        _validate_check_record(
            mpi,
            repo=repo,
            evidence_root=evidence_root,
            context_sha256=context_sha256,
            expected_label=f"{ranks}-rank MPI smoke test",
            expected_runtime_sha256=str(runtimes["python312"].get("runtime_sha256")),
            expected_environment=qualification_environment(runtimes["python312"], repo=repo),
            expected_command=expected_mpi_command,
            expected_log_name=f"mpi-ranks-{ranks}.log",
            expected_record_keys=CHECK_RECORD_KEYS | {"interpreter"},
        )
        if not isinstance(mpi, Mapping) or mpi.get("interpreter") != "python312":
            raise QualificationError(f"{ranks}-rank MPI interpreter changed")

    material = records.get("material_input_preflight")
    resources = records.get("resources")
    checks = document.get("checks")
    if (
        not isinstance(material, Mapping)
        or set(material) != {"source_hashes", "observation_modes", "harmonic_buffer"}
        or material.get("source_hashes") != context.get("material_input_hashes")
        or not isinstance(material.get("observation_modes"), Mapping)
        or not material["observation_modes"].get("subspaces")
        or not isinstance(material.get("harmonic_buffer"), Mapping)
        or material["harmonic_buffer"].get("passed") is not True
        or not isinstance(resources, Mapping)
        or set(resources) != {
            "logical_cpu_count", "available_memory_bytes",
            "largest_task_memory_estimate_bytes", "free_disk_bytes",
            "campaign_field_archive_estimate_bytes", "memory_guard_passed",
            "disk_guard_passed",
        }
        or resources.get("memory_guard_passed") is not True
        or resources.get("disk_guard_passed") is not True
        or resources.get("largest_task_memory_estimate_bytes", 0)
        > 0.70 * resources.get("available_memory_bytes", -1)
        or 2 * resources.get("campaign_field_archive_estimate_bytes", 0)
        > resources.get("free_disk_bytes", -1)
        or not isinstance(checks, Mapping)
        or set(checks) != release.EXPECTED_QUALIFICATION_CHECKS
        or not all(value is True for value in checks.values())
        or document.get("release_preflight_passed") is not True
    ):
        raise QualificationError("base qualification evidence did not pass")
    return dict(document)


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    repo = _repo_root(args.repo_root)
    output, evidence_root = _qualification_output_paths(args.output)
    source = common._source_identity(repo, args.expected_commit)
    lock = common._validate_lock(args.environment_lock)
    source_tree = code_provenance().get("source_tree_sha256")
    if not isinstance(source_tree, str) or len(source_tree) != 64:
        raise QualificationError("source tree SHA256 is unavailable")
    scientific_controls = release._scientific_control_record()
    base = next(
        task for task in silicon_nq1_release_tasks()
        if task.task_id == "coupling-g100"
    )
    identity, problem, _config = release._authenticate_task_inputs(
        fixture=args.fixture,
        fc3_full=args.fc3_full,
        phase_b_artifacts=args.phase_b_artifacts,
        task=base,
    )
    material_input_hashes = case._source_hashes(identity)
    modes = build_observation_mode_set(
        problem,
        observation_layers=(4,),
        grid_step=base.grid_step,
        eta_lead=base.eta_lead,
    )
    buffer = dict(check_harmonic_embedding_buffer(
        problem,
        frequencies=problem.grid.values[
            (problem.grid.values > 0.0) & (problem.grid.values <= 2.4)
        ],
        eta_lead=1.0e-8,
    ))
    if not buffer["passed"]:
        raise QualificationError("harmonic embedding-buffer preflight failed")

    interpreters = {
        "python310": args.python310.expanduser().resolve(strict=True),
        "python312": args.python312.expanduser().resolve(strict=True),
    }
    git_path = shutil.which("git")
    if git_path is None:
        raise QualificationError("Git is unavailable for the launch environment")
    # Only these explicit operational paths are selected from the driver.
    # All other ambient variables, including inherited Python/MPI controls,
    # are discarded by build_launch_environment.
    home = Path.home().resolve()
    tmpdir = Path(tempfile.gettempdir()).resolve()
    git_bin = Path(git_path).resolve().parent

    evidence: dict[str, Any] = {}
    versions: dict[str, str] = {}
    runtime_environments: dict[str, dict[str, str]] = {}
    for label, executable in interpreters.items():
        environment = build_launch_environment(
            repo=repo, executable=executable, home=home, tmpdir=tmpdir, git_bin=git_bin,
        )
        version = subprocess.run(
            [str(executable), "-c", "import sys; print(sys.version.split()[0])"],
            cwd=repo, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, text=True,
        )
        if version.returncode != 0:
            raise QualificationError(f"{label} cannot start")
        versions[label] = version.stdout.strip()
        expected_minor = "3.10." if label == "python310" else "3.12."
        if not versions[label].startswith(expected_minor):
            raise QualificationError(f"{label} has wrong version {versions[label]!r}")
        packages = subprocess.run(
            [str(executable), "-m", "pip", "freeze", "--all"],
            cwd=repo,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if packages.returncode != 0 or not packages.stdout:
            raise QualificationError(f"{label} package manifest is unavailable")
        packages_sha = hashlib.sha256(packages.stdout).hexdigest()
        package_path = evidence_root / f"{label}-packages-{packages_sha[:16]}.txt"
        package_record = _publish_log(package_path, packages.stdout)
        executable_sha = release._sha256(executable)
        runtime_payload = {
            "schema": "negfpy.material-nq1-python-runtime.v2",
            "label": label,
            "version": versions[label],
            "executable": str(executable),
            "executable_sha256": executable_sha,
            "packages_sha256": packages_sha,
            "launch_environment": environment,
        }
        runtime_sha = hashlib.sha256(
            release._canonical(runtime_payload)
        ).hexdigest()
        runtime_environment = qualification_environment(
            runtime_payload | {"runtime_sha256": runtime_sha}, repo=repo,
        )
        runtime_environments[label] = runtime_environment
        evidence[f"{label}_runtime"] = runtime_payload | {
            "runtime_sha256": runtime_sha,
            "package_manifest": package_record,
        }

    mpiexec = args.mpiexec.expanduser().resolve(strict=True)
    qualification_context = qualification_context_record(
        repo=repo,
        source=source,
        source_tree_sha256=source_tree,
        environment_lock=lock,
        scientific_task_set_sha256=scientific_controls["task_set_sha256"],
        material_input_hashes=material_input_hashes,
        runtimes={
            label: evidence[f"{label}_runtime"] for label in interpreters
        },
        mpiexec=mpiexec,
    )

    for label, executable in interpreters.items():
        evidence[f"{label}_pytest"] = _run(
            [str(executable), "-m", "pytest", "-q", *QUALIFICATION_TEST_TARGETS],
            cwd=repo,
            environment=runtime_environments[label],
            log=evidence_root / f"{label}-pytest.log",
            label=f"full {label} test suite",
            qualification_context=qualification_context,
        )
        evidence[f"{label}_pytest"]["scope"] = list(QUALIFICATION_TEST_TARGETS)
        evidence[f"{label}_pytest"]["excluded_tests"] = []

    with tempfile.TemporaryDirectory(prefix="negf-material-mpi-qualification-") as name:
        temporary = Path(name)
        for ranks in (2, 4):
            mpi_command = [
                str(mpiexec), "--bind-to", "core", "-n", str(ranks),
                str(interpreters["python312"]),
                str(repo / "tests/mpi/run_material_nq1_mpi_smoke.py"),
                str(temporary / f"ranks-{ranks}"),
                "--execution-id", f"qualification-ranks-{ranks}",
            ]
            mpi_identity_command = list(mpi_command)
            mpi_identity_command[7] = "<temporary-mpi-output>"
            evidence[f"mpi_ranks_{ranks}"] = _run(
                mpi_command,
                cwd=repo,
                environment=runtime_environments["python312"],
                log=evidence_root / f"mpi-ranks-{ranks}.log",
                label=f"{ranks}-rank MPI smoke test",
                qualification_context=qualification_context,
                identity_command=mpi_identity_command,
            )
            evidence[f"mpi_ranks_{ranks}"]["interpreter"] = "python312"

    evidence["material_input_preflight"] = {
        "source_hashes": material_input_hashes,
        "observation_modes": modes.as_dict(),
        "harmonic_buffer": buffer,
    }

    memory_per_largest_task = int(
        288
        * max(task.grid_n_positive * 2 + 1 for task in silicon_nq1_release_tasks())
        * problem.ndof**2
    )
    available_memory = int(
        next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    ) * 1024
    free_disk = shutil.disk_usage(output.parent).free
    predicted_archive = int(
        sum(
            12 * (2 * task.grid_n_positive + 1) * problem.ndof**2 * 16
            for task in silicon_nq1_release_tasks()
        )
    )
    resource = {
        "logical_cpu_count": os.cpu_count(),
        "available_memory_bytes": available_memory,
        "largest_task_memory_estimate_bytes": memory_per_largest_task,
        "free_disk_bytes": free_disk,
        "campaign_field_archive_estimate_bytes": predicted_archive,
        "memory_guard_passed": memory_per_largest_task <= 0.70 * available_memory,
        "disk_guard_passed": 2 * predicted_archive <= free_disk,
    }
    if not resource["memory_guard_passed"] or not resource["disk_guard_passed"]:
        raise QualificationError("remote resource preflight failed")
    evidence["resources"] = resource

    final_source = common._source_identity(repo, args.expected_commit)
    checks = {
        "source_clean_and_unchanged": final_source == source,
        "python_310_full_suite": (
            evidence["python310_pytest"]["returncode"] == 0
        ),
        "python_312_full_suite": (
            evidence["python312_pytest"]["returncode"] == 0
        ),
        "mpi_two_rank_smoke": evidence["mpi_ranks_2"]["returncode"] == 0,
        "mpi_four_rank_smoke": evidence["mpi_ranks_4"]["returncode"] == 0,
        "material_inputs_authenticated": bool(material_input_hashes),
        "resolved_mode_set_nonempty": bool(modes.subspaces),
        "harmonic_embedding_buffer": buffer["passed"] is True,
        "memory_guard": resource["memory_guard_passed"] is True,
        "disk_guard": resource["disk_guard_passed"] is True,
        "thread_limits_one": all(
            environment[name] == "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        ),
        "candidate_pythonpath_first": environment["PYTHONPATH"].split(
            os.pathsep
        )[:2] == [str(repo / "src"), str(repo / "scripts")],
    }
    passed = bool(checks and all(checks.values()))
    report = {
        "schema": release.QUALIFICATION_SCHEMA,
        "source": source,
        "source_tree_sha256": source_tree,
        "environment_sha256": lock["sha256"],
        "scientific_controls": scientific_controls,
        "checks": checks,
        "evidence": {
            "qualification_context": qualification_context,
            "python_versions": versions,
            "records": evidence,
        },
        "release_preflight_passed": passed,
    }
    return validate_qualification_evidence(
        report, repo=repo, evidence_root=evidence_root
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--phase-b-artifacts", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--python310", type=Path, required=True)
    parser.add_argument("--python312", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = qualify(args)
        release._atomic_json(args.output, report)
    except Exception as error:
        print(
            f"material candidate qualification error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["release_preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
