#!/usr/bin/env python3
"""Fail-closed import of a completed protocol-v2 toy-gate artifact bundle.

This script is intentionally a report-data importer, not a physics runner.
It never calls the Born or SCBA solvers.  Before publishing a staging
directory it verifies:

* the fetched artifact bundle and its exact outer ``SHA256SUMS`` coverage;
* the current completed attempt, status, publication, and report-facts chain;
* all publication-named scaling, launch, log, runner, execution, and campaign
  artifacts, including exact-retry scaling selections from a prior attempt;
* the raw production report file hash and both canonical report-body hashes;
* the release-tier protocol-v2/schema-v4 contract and code/environment
  identities; and
* all thirteen relocated HDF5 checkpoints using their exact recorded
  :class:`SCBACheckpointIdentity` values.

Only after every check succeeds are the original report bytes, a strict
import receipt, deterministic CSV/TeX derivatives, raw source publication
manifests, and an exhaustive path/size/SHA256 source-artifact index atomically
published. Large HDF5 payloads stay in the caller-anchored fetched tree.
An existing byte-identical staging directory is accepted unchanged; an
existing different directory is rejected.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
import ctypes
import errno
import hashlib
import hmac
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from types import MappingProxyType
from typing import Any
from uuid import uuid4


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_IMPORTER_VERSION = "3"
_REPORT_SCHEMA = "negfpy.anharmonic.production_toy_gate"
_MULTIDIMENSIONAL_SCHEMA = (
    "negfpy.anharmonic.multidimensional_dense_toy_release"
)
_PROTOCOL = "multidimensional_toy_gate_v2"
_REMOTE_CHECKPOINT_ROOT = "toy-release/scientific-checkpoints"
_REMOTE_FETCH_RECEIPT_SCHEMA = (
    "negfpy.remote_validation.fetch_receipt.v1"
)
_LOCAL_PUBLICATION_RECEIPT_SCHEMA = (
    "negfpy.local_validation.publication_receipt.v1"
)
_LOCAL_STATUS_SCHEMA = "negfpy.local_validation.status.v1"
_LOCAL_EXECUTION_IDENTITY_SCHEMA = (
    "negfpy.local_validation.execution_identity.v1"
)
_REPORT_FACTS_SCHEMA = "negfpy.remote_validation.report_facts.v1"
_CHECKPOINT_INDEX_SCHEMA = (
    "negfpy.remote_validation.checkpoint_index.v1"
)
_REQUIRED_SUMMARY_GATES = (
    "code_identity_consistent",
    "exact_release_revision_passed",
    "cross_platform_identity_consistent",
    "environment_lock_consistent",
    "fixture_contract_passed",
    "full_locked_base_coverage",
    "multidimensional_release_certified",
    "multidimensional_static_configuration_passed",
    "multidimensional_runtime_evidence_passed",
    "multidimensional_harmonic_preflight_passed",
    "multidimensional_backend_validation_passed",
    "multidimensional_outcome_certification_passed",
    "multidimensional_contract_passed",
    "multidimensional_comparison_coverage",
    "multidimensional_underlying_evidence_complete",
    "multidimensional_provenance_passed",
    "multidimensional_weak_coupling_passed",
    "open_three_mode_agreement_passed",
    "open_three_mode_oracle_method_passed",
    "open_three_mode_passivity_passed",
    "open_three_mode_production_completed",
    "open_three_mode_provenance_complete",
    "open_three_mode_weak_coupling_passed",
    "release_certified",
    "oracle_gate_passed",
    "refinement_coverage",
    "refinement_passed",
    "scaling_coverage",
    "scaling_passed",
    "sign_coverage",
    "sign_invariance_passed",
    "spontaneous_decay_coverage",
    "spontaneous_decay_passed",
)
_FIXED_POINT_COMPARISONS = frozenset(
    {
        "zero_vs_born_endpoint",
        "direct_vs_continuation_endpoint",
    }
)
_REFINEMENT_COMPARISONS = frozenset(
    {
        "numerical_refinement_spacing_0p0015625",
        "numerical_refinement_spacing_0p000390625",
        "numerical_refinement_cutoff_6",
        "numerical_refinement_eta_lead_0p003125",
        "numerical_refinement_eta_lead_0p00078125",
        "numerical_refinement_mixing_0p1",
    }
)
_CRITICAL_COMPARISONS = frozenset(
    {
        *_FIXED_POINT_COMPARISONS,
        "numerical_refinement_spacing_0p000390625",
        "numerical_refinement_cutoff_6",
        "numerical_refinement_eta_lead_0p00078125",
        "numerical_refinement_mixing_0p1",
    }
)
_FIELD_COMPARISON_REFINEMENTS = frozenset(
    {
        "numerical_refinement_eta_lead_0p003125",
        "numerical_refinement_eta_lead_0p00078125",
        "numerical_refinement_mixing_0p1",
    }
)
_FIELD_GATE_REFINEMENTS = frozenset(
    {"numerical_refinement_mixing_0p1"}
)
_HARMONIC_POINT_CONTRACT = (
    ("base", 0.00078125, 5.0, 0.0015625, 0.2, 6400, "reference"),
    (
        "spacing_0p0015625",
        0.0015625,
        5.0,
        0.0015625,
        0.2,
        3200,
        "diagnostic",
    ),
    (
        "spacing_0p000390625",
        0.000390625,
        5.0,
        0.0015625,
        0.2,
        12800,
        "critical",
    ),
    ("cutoff_6", 0.00078125, 6.0, 0.0015625, 0.2, 7680, "critical"),
    (
        "eta_lead_0p003125",
        0.00078125,
        5.0,
        0.003125,
        0.2,
        6400,
        "diagnostic",
    ),
    (
        "eta_lead_0p00078125",
        0.00078125,
        5.0,
        0.00078125,
        0.2,
        6400,
        "critical",
    ),
    ("mixing_0p1", 0.00078125, 5.0, 0.0015625, 0.1, 6400, "critical"),
)
_HARMONIC_COMPARISON_CONTRACT = MappingProxyType(
    {
        label: MappingProxyType(
            {
                "name": f"harmonic_current_refinement_{label}",
                "production_name": f"numerical_refinement_{label}",
                "critical": role == "critical",
                "role": role,
            }
        )
        for label, _, _, _, _, _, role in _HARMONIC_POINT_CONTRACT[1:]
    }
)
_COST_ACCURACY_CONTRACT = (
    MappingProxyType(
        {
            "method": "harmonic_negf",
            "accuracy_role": "exact elastic baseline for the supplied FC2",
            "asymptotic_dense_time": "O(N_omega*D^3)",
            "conserving": True,
            "self_consistent": False,
            "release_use": "harmonic baseline",
        }
    ),
    MappingProxyType(
        {
            "method": "one_shot_born",
            "accuracy_role": (
                "O(Psi^2) screening, pole oracle comparison, and SCBA "
                "initialization"
            ),
            "asymptotic_dense_time": (
                "O(N_omega^2*D^4 + N_omega*D^3)"
            ),
            "conserving": False,
            "self_consistent": False,
            "release_use": "screening_only_never_conductance_certified",
        }
    ),
    MappingProxyType(
        {
            "method": "self_consistent_born",
            "accuracy_role": (
                "nonlinear conserving cubic-bubble fixed point within SCBA"
            ),
            "asymptotic_dense_time": (
                "O(N_iteration*(N_omega^2*D^4 + N_omega*D^3))"
            ),
            "conserving": True,
            "self_consistent": True,
            "release_use": "required_interacting_conductance_tier",
        }
    ),
)
_MULTIDIMENSIONAL_WEAK_COUPLING_DEFINITION = (
    "For center-layer parent-x at Omega_p, require both "
    "abs(-Im(Sigma_pp^r)/Omega_p)/Omega_p and "
    "abs(Re(Sigma_pp^r)/(2*Omega_p))/Omega_p below 0.1."
)
_ARCHIVED_IDENTITY_FILES = (
    "deployment-manifest.json",
    "environment-explicit.lock",
    "environment-manifest.json",
    "incoming-SHA256SUMS",
)
_THREAD_LIMIT_KEYS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_LOCAL_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "created_utc",
        "commit",
        "environment_lock_sha256",
        "artifact_root",
        "sha256sums_sha256",
        "verified_files",
    }
)
_REMOTE_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "fetched_utc",
        "commit",
        "remote",
        "verified_files",
    }
)
_LOCAL_STATUS_FILE_FIELDS = MappingProxyType(
    {
        "report": "production_toy_gate.json",
        "report_facts": "report-facts.json",
        "checkpoint_index": "checkpoint-index.json",
        "execution_identity": "execution-identity.json",
        "log": "production_toy_gate.log",
        "launch": "launch.sh",
        "exit_code": "exit-code.txt",
    }
)
_LOCAL_STATUS_FIELDS = frozenset(
    {
        "schema",
        "started_utc",
        "finished_utc",
        "state",
        "returncode",
        "commit",
        "attempt",
        "artifact_root",
        *(
            field
            for stem in _LOCAL_STATUS_FILE_FIELDS
            for field in (stem, f"{stem}_sha256")
        ),
    }
)
_LOCAL_EXECUTION_IDENTITY_FIELDS = frozenset(
    {
        "schema",
        "created_utc",
        "commit",
        "tree",
        "source_tree_sha256",
        "explicit_lock_sha256",
        "hostname",
        "platform",
        "machine",
        "python_executable",
        "python_version",
        "distributions",
        "package_source",
        "thread_limits",
        "cpu_count",
        "cpu_model",
        "memory_total_bytes",
        "memory_available_bytes",
    }
)
_REPORT_FACTS_FIELDS = frozenset(
    {
        "schema",
        "report_relative_path",
        "report_file_sha256",
        "report_body_sha256",
        "schema_version",
        "protocol",
        "protocol_version",
        "release_certified",
        "multidimensional_outcome_count",
        "multidimensional_backend",
        "harmonic_preflight_point_count",
        "bounded_backends",
        "open_system_run_count",
        "checkpoint_count",
        "commit",
        "git_tree",
        "source_tree_sha256",
        "environment_sha256",
    }
)
_REMOTE_IMMUTABLE_FIELDS = (
    "schema",
    "commit",
    "source_tree_sha256",
    "execution_source_identity",
    "environment_prefix_manifest",
    "explicit_lock_sha256",
    "tree",
    "hostname",
    "platform",
    "machine",
    "affinity_logical_cpu_count",
    "physical_core_count",
    "socket_count",
    "python_executable",
    "python_version",
    "distributions",
    "mpi",
    "package_source",
    "thread_limits",
)
_ORCHESTRATION_SCRIPT_PATHS = (
    "scripts/remote_heavy_cli.py",
    "scripts/remote_heavy_explicit.py",
    "scripts/remote_heavy_identity.py",
    "scripts/remote_heavy_lock.py",
    "scripts/remote_heavy_runtime.py",
    "scripts/remote_heavy_security.py",
    "scripts/remote_heavy_validation.py",
    "scripts/remote_mpi_scaling.py",
    "scripts/remote_toy_mpi_runner.py",
)
_LAUNCH_ENVIRONMENT_KEYS = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "TZ",
        "PATH",
        "PYTHONPATH",
        "PYTHONHASHSEED",
        "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "MPLCONFIGDIR",
        *_THREAD_LIMIT_KEYS,
    }
)
_RELEASE_LAUNCH_BINDING_FIELDS = frozenset(
    {
        "schema",
        "source_revision",
        "source_tree_sha256",
        "environment_sha256",
        "execution_id",
        "selected_rank_count",
        "argv",
        "binding_arguments",
        "thread_limits",
        "launcher",
        "python",
        "runner",
        "execution_source_identity_sha256",
        "tracked_files_manifest_sha256",
        "worktree_files_manifest_sha256",
        "orchestration_manifest_sha256",
        "orchestration_scripts",
        "environment_prefix_manifest_sha256",
        "environment_regular_file_count",
        "environment_symlink_count",
        "launch_environment",
        "launch_environment_sha256",
    }
)
_IDENTITY_KEYS = (
    "declared_package_version",
    "installed_package_version",
    "git_revision",
    "source_tree_sha256",
    "dtype",
    "numpy_version",
    "scipy_version",
    "python",
    "platform",
    "blas_runtime",
)
_CHECKPOINT_IDENTITY_KEYS = (
    "run_id",
    "configuration_sha256",
    "problem_sha256",
    "initial_state_sha256",
    "code_sha256",
    "environment_sha256",
)


class ArtifactImportError(RuntimeError):
    """Raised when fetched evidence is incomplete, inconsistent, or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArtifactImportError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and _COMMIT_RE.fullmatch(value) is not None


def _reject_constant(token: str) -> None:
    raise ArtifactImportError(f"strict JSON contains non-finite token {token!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactImportError(
                f"strict JSON contains duplicate object key {key!r}"
            )
        value[key] = item
    return value


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactImportError(f"{name} is not UTF-8") from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise ArtifactImportError(f"{name} is not valid strict JSON") from error
    if not isinstance(value, dict):
        raise ArtifactImportError(f"{name} must contain one JSON object")
    return value


def _strict_json_file(path: Path, *, name: str | None = None) -> dict[str, Any]:
    return _strict_json_bytes(
        path.read_bytes(),
        name=name or path.as_posix(),
    )


def _canonical_json_bytes(value: Mapping[str, Any], *, ensure_ascii: bool) -> bytes:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactImportError(
            "report contains a value unsupported by canonical strict JSON"
        ) from error
    return text.encode("utf-8")


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactImportError(
            "receipt contains a value unsupported by strict JSON"
        ) from error


def _canonical_manifest_sha256(
    value: Mapping[str, Any], *, excluded_key: str
) -> str:
    body = {key: item for key, item in value.items() if key != excluded_key}
    return _sha256_bytes(_canonical_json_bytes(body, ensure_ascii=True))


def _verified_embedded_manifest(
    value: Any, *, name: str, schema: str
) -> Mapping[str, Any]:
    manifest = _mapping(value, name=name)
    digest = manifest.get("manifest_sha256")
    _require(
        manifest.get("schema") == schema
        and _is_sha256(digest)
        and _canonical_manifest_sha256(
            manifest, excluded_key="manifest_sha256"
        )
        == digest,
        f"{name} canonical SHA256 or schema is invalid",
    )
    return manifest


def _strict_hashed_manifest(path: Path, *, name: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    document = _strict_json_bytes(raw, name=name)
    _require(
        set(document) == {"payload", "payload_sha256"},
        f"{name} has an invalid hashed-manifest envelope",
    )
    payload = _mapping(document.get("payload"), name=f"{name}.payload")
    digest = document.get("payload_sha256")
    _require(
        _is_sha256(digest)
        and _sha256_bytes(_canonical_json_bytes(payload, ensure_ascii=True))
        == digest,
        f"{name} payload SHA256 is invalid",
    )
    _require(
        raw == _canonical_json_bytes(document, ensure_ascii=True) + b"\n",
        f"{name} is not canonical ASCII JSON",
    )
    return MappingProxyType(dict(document))


def _safe_source_path(value: Any, *, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} is absent")
    assert isinstance(value, str)
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == value,
        f"{name} is unsafe",
    )
    return value


def _validate_source_records(
    value: Any, *, name: str, require_tracked: bool | None
) -> tuple[Mapping[str, Any], ...]:
    records = _sequence(value, name=name)
    validated: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, name=f"{name}[{index}]")
        path = _safe_source_path(record.get("path"), name=f"{name}[{index}].path")
        _require(path not in seen, f"{name} contains duplicate path {path}")
        seen.add(path)
        tracked = record.get("git_tracked")
        _require(isinstance(tracked, bool), f"{name}[{index}] tracking flag is invalid")
        if require_tracked is not None:
            _require(
                tracked is require_tracked,
                f"{name}[{index}] has an invalid tracking flag",
            )
        kind = record.get("kind")
        if kind == "regular_file":
            _require(
                set(record)
                == {"path", "git_tracked", "kind", "size", "mode", "sha256"}
                and isinstance(record.get("size"), int)
                and not isinstance(record.get("size"), bool)
                and record["size"] >= 0
                and isinstance(record.get("mode"), int)
                and not isinstance(record.get("mode"), bool)
                and 0 <= record["mode"] <= 0o7777
                and _is_sha256(record.get("sha256")),
                f"{name}[{index}] regular-file identity is invalid",
            )
        elif kind == "symlink":
            _require(
                set(record)
                == {"path", "git_tracked", "kind", "target", "mode"}
                and isinstance(record.get("target"), str)
                and bool(record["target"])
                and isinstance(record.get("mode"), int)
                and not isinstance(record.get("mode"), bool)
                and 0 <= record["mode"] <= 0o7777,
                f"{name}[{index}] symlink identity is invalid",
            )
        else:
            raise ArtifactImportError(f"{name}[{index}] has unsupported kind")
        validated.append(MappingProxyType(dict(record)))
    _require(
        [record["path"] for record in validated]
        == sorted(record["path"] for record in validated),
        f"{name} is not in canonical path order",
    )
    return tuple(validated)


def _records_manifest_sha256(
    *, schema: str, records: Sequence[Mapping[str, Any]]
) -> str:
    body: dict[str, Any] = {
        "schema": schema,
        "records": [dict(record) for record in records],
    }
    body["manifest_sha256"] = _sha256_bytes(
        _canonical_json_bytes(body, ensure_ascii=True)
    )
    return str(body["manifest_sha256"])


def _validate_execution_source_identity(
    value: Any, *, commit: str, source_tree_sha256: str
) -> Mapping[str, Any]:
    source = _verified_embedded_manifest(
        value,
        name="execution source identity",
        schema="negfpy.remote_validation.execution_source.v1",
    )
    expected_fields = {
        "schema",
        "git_revision",
        "git_tree",
        "package_source_sha256",
        "tracked_file_count",
        "tracked_files_manifest_sha256",
        "tracked_files",
        "worktree_file_count",
        "worktree_files_manifest_sha256",
        "worktree_files",
        "orchestration_manifest_sha256",
        "orchestration_scripts",
        "manifest_sha256",
    }
    _require(set(source) == expected_fields, "execution source identity fields are invalid")
    _require(
        source.get("git_revision") == commit
        and _is_commit(source.get("git_tree"))
        and source.get("package_source_sha256") == source_tree_sha256,
        "execution source identity differs from the release source",
    )
    tracked = _validate_source_records(
        source.get("tracked_files"),
        name="tracked source files",
        require_tracked=True,
    )
    worktree = _validate_source_records(
        source.get("worktree_files"),
        name="full worktree files",
        require_tracked=None,
    )
    orchestration = _validate_source_records(
        source.get("orchestration_scripts"),
        name="orchestration scripts",
        require_tracked=True,
    )
    _require(
        source.get("tracked_file_count") == len(tracked)
        and source.get("worktree_file_count") == len(worktree),
        "execution source file counts are invalid",
    )
    _require(
        source.get("tracked_files_manifest_sha256")
        == _records_manifest_sha256(
            schema="negfpy.remote_validation.tracked_worktree.v1",
            records=tracked,
        )
        and source.get("worktree_files_manifest_sha256")
        == _records_manifest_sha256(
            schema="negfpy.remote_validation.full_worktree.v1",
            records=worktree,
        )
        and source.get("orchestration_manifest_sha256")
        == _records_manifest_sha256(
            schema="negfpy.remote_validation.orchestration_sources.v1",
            records=orchestration,
        ),
        "execution source submanifest SHA256 is invalid",
    )
    tracked_by_path = {record["path"]: dict(record) for record in tracked}
    worktree_by_path = {record["path"]: dict(record) for record in worktree}
    _require(
        all(worktree_by_path.get(path) == record for path, record in tracked_by_path.items()),
        "full worktree does not contain the exact tracked source records",
    )
    _require(
        tuple(record["path"] for record in orchestration)
        == _ORCHESTRATION_SCRIPT_PATHS
        and all(
            record.get("kind") == "regular_file"
            and tracked_by_path.get(str(record["path"])) == dict(record)
            for record in orchestration
        ),
        "execution source identity lacks the exact orchestration source set",
    )
    return source


def _validate_environment_prefix_manifest(value: Any) -> Mapping[str, Any]:
    manifest = _verified_embedded_manifest(
        value,
        name="environment prefix manifest",
        schema="negfpy.remote_validation.environment_prefix.v1",
    )
    _require(
        set(manifest)
        == {
            "schema",
            "prefix",
            "directory_count",
            "regular_file_count",
            "symlink_count",
            "regular_files",
            "symlinks",
            "manifest_sha256",
        },
        "environment prefix manifest fields are invalid",
    )
    prefix = manifest.get("prefix")
    _require(
        isinstance(prefix, str) and PurePosixPath(prefix).is_absolute(),
        "environment prefix path is invalid",
    )
    regular_files = _sequence(
        manifest.get("regular_files"), name="environment regular files"
    )
    symlinks = _sequence(manifest.get("symlinks"), name="environment symlinks")
    regular_paths: list[str] = []
    for index, raw_record in enumerate(regular_files):
        record = _mapping(raw_record, name=f"environment regular file {index}")
        path = _safe_source_path(
            record.get("path"), name=f"environment regular file {index}.path"
        )
        _require(
            set(record) == {"path", "size", "sha256"}
            and isinstance(record.get("size"), int)
            and not isinstance(record.get("size"), bool)
            and record["size"] >= 0
            and _is_sha256(record.get("sha256")),
            f"environment regular file {index} is invalid",
        )
        regular_paths.append(path)
    symlink_paths: list[str] = []
    for index, raw_record in enumerate(symlinks):
        record = _mapping(raw_record, name=f"environment symlink {index}")
        path = _safe_source_path(
            record.get("path"), name=f"environment symlink {index}.path"
        )
        resolved = _safe_source_path(
            record.get("resolved_path"),
            name=f"environment symlink {index}.resolved_path",
        )
        target = record.get("target")
        _require(
            set(record)
            == {
                "path",
                "target",
                "resolved_path",
                "target_kind",
                "link_size",
            }
            and isinstance(target, str)
            and bool(target)
            and not PurePosixPath(target).is_absolute()
            and record.get("target_kind") in {"regular_file", "directory"}
            and isinstance(record.get("link_size"), int)
            and not isinstance(record.get("link_size"), bool)
            and record["link_size"] >= 0
            and bool(resolved),
            f"environment symlink {index} is invalid",
        )
        symlink_paths.append(path)
    _require(
        len(set(regular_paths) | set(symlink_paths))
        == len(regular_paths) + len(symlink_paths)
        and regular_paths == sorted(regular_paths)
        and symlink_paths == sorted(symlink_paths),
        "environment prefix paths are duplicate or unsorted",
    )
    _require(
        isinstance(manifest.get("directory_count"), int)
        and not isinstance(manifest.get("directory_count"), bool)
        and manifest["directory_count"] > 0
        and manifest.get("regular_file_count") == len(regular_files)
        and manifest.get("symlink_count") == len(symlinks),
        "environment prefix counts are invalid",
    )
    return manifest


def _validate_recorded_attempt_artifact(
    attempt_root: Path, value: Any, *, expected: str, name: str
) -> Path:
    record = _mapping(value, name=name)
    _require(
        set(record) == {"relative_path", "sha256"}
        and record.get("relative_path") == expected
        and _is_sha256(record.get("sha256")),
        f"{name} record is invalid",
    )
    target = _safe_relative_file(attempt_root, expected, name=name)
    _require(
        _sha256_file(target) == record.get("sha256"),
        f"{name} file SHA256 is invalid",
    )
    return target


def _validate_remote_file_identity(
    value: Any, *, name: str, with_size: bool
) -> Mapping[str, Any]:
    record = _mapping(value, name=name)
    expected = {"path", "resolved_path", "sha256"}
    if with_size:
        expected.add("size")
    _require(
        set(record) == expected
        and all(
            isinstance(record.get(key), str) and bool(record[key])
            for key in ("path", "resolved_path")
        )
        and _is_sha256(record.get("sha256"))
        and (
            not with_size
            or (
                isinstance(record.get("size"), int)
                and not isinstance(record.get("size"), bool)
                and record["size"] > 0
            )
        ),
        f"{name} identity is invalid",
    )
    return record


def _validate_release_launch_provenance(
    root: Path,
    *,
    attempt: int,
    commit: str,
    current: Mapping[str, Any],
    status_mpi: Mapping[str, Any],
) -> None:
    attempt_relative = f"attempts/attempt-{attempt:03d}"
    attempt_root = (root / attempt_relative).resolve(strict=True)
    _require(
        attempt_root.is_dir()
        and not attempt_root.is_symlink()
        and attempt_root.is_relative_to(root),
        "current attempt directory is unsafe",
    )
    execution_identity = _strict_json_file(
        _safe_relative_file(
            root,
            f"{attempt_relative}/execution-identity.json",
            name="attempt execution identity",
        ),
        name="attempt execution identity",
    )
    source_tree_sha256 = execution_identity.get("source_tree_sha256")
    environment_sha256 = execution_identity.get("explicit_lock_sha256")
    _require(
        execution_identity.get("schema")
        == "negfpy.remote_validation.remote_environment.v1"
        and execution_identity.get("commit") == commit
        and _is_sha256(source_tree_sha256)
        and _is_sha256(environment_sha256),
        "attempt execution identity is invalid",
    )
    assert isinstance(source_tree_sha256, str)
    source = _validate_execution_source_identity(
        execution_identity.get("execution_source_identity"),
        commit=commit,
        source_tree_sha256=source_tree_sha256,
    )
    prefix = _validate_environment_prefix_manifest(
        execution_identity.get("environment_prefix_manifest")
    )
    _require(
        source.get("git_tree") == execution_identity.get("tree"),
        "execution source Git tree differs from the attempt identity",
    )

    intent_path = _safe_relative_file(
        root, current.get("mpi_release_launch_intent"), name="MPI release intent"
    )
    receipt_path = _safe_relative_file(
        root, current.get("mpi_release_launch"), name="MPI release receipt"
    )
    _require(
        intent_path.parent == attempt_root
        and intent_path.name == "mpi-release-launch-intent.json"
        and receipt_path.parent == attempt_root
        and receipt_path.name == "mpi-release-launch.json",
        "MPI release intent or receipt is outside its canonical attempt path",
    )
    intent = _strict_hashed_manifest(intent_path, name="MPI release launch intent")
    receipt = _strict_hashed_manifest(receipt_path, name="MPI release launch receipt")
    intent_payload = _mapping(
        intent.get("payload"), name="MPI release launch intent payload"
    )
    receipt_payload = _mapping(
        receipt.get("payload"), name="MPI release launch receipt payload"
    )
    _require(
        set(intent_payload)
        == {
            "schema",
            "launch_binding",
            "stdout_relative_path",
            "stderr_relative_path",
        }
        and intent_payload.get("schema")
        == "negfpy.remote_validation.release_launch_intent.v1"
        and intent_payload.get("stdout_relative_path")
        == "mpi-release.stdout.log"
        and intent_payload.get("stderr_relative_path")
        == "mpi-release.stderr.log",
        "MPI release launch intent payload is invalid",
    )
    binding = _mapping(
        intent_payload.get("launch_binding"), name="MPI release launch binding"
    )
    _require(
        set(binding) == _RELEASE_LAUNCH_BINDING_FIELDS
        and binding.get("schema")
        == "negfpy.remote_validation.release_launch_binding.v1",
        "MPI release launch binding fields or schema are invalid",
    )
    rank_count = status_mpi.get("communicator_size")
    execution_id = status_mpi.get("execution_id")
    _require(
        isinstance(rank_count, int)
        and not isinstance(rank_count, bool)
        and rank_count > 0
        and isinstance(execution_id, str)
        and bool(execution_id)
        and binding.get("source_revision") == commit
        and binding.get("source_tree_sha256") == source_tree_sha256
        and binding.get("environment_sha256") == environment_sha256
        and binding.get("execution_id") == execution_id
        and binding.get("selected_rank_count") == rank_count,
        "MPI release launch binding differs from terminal source/execution identity",
    )
    expected_threads = {name: "1" for name in _THREAD_LIMIT_KEYS}
    _require(
        binding.get("thread_limits") == expected_threads,
        "MPI release launch binding thread limits are invalid",
    )
    launcher = _validate_remote_file_identity(
        binding.get("launcher"), name="MPI launcher", with_size=False
    )
    python = _validate_remote_file_identity(
        binding.get("python"), name="MPI Python", with_size=False
    )
    runner = _validate_remote_file_identity(
        binding.get("runner"), name="MPI runner", with_size=True
    )
    prefix_files = {
        record.get("path"): record
        for record in prefix.get("regular_files", [])
        if isinstance(record, Mapping)
    }
    prefix_symlinks = {
        record.get("path"): record
        for record in prefix.get("symlinks", [])
        if isinstance(record, Mapping)
    }

    def prefix_executable_matches(
        record: Mapping[str, Any], relative: str
    ) -> bool:
        prefix_text = str(prefix.get("prefix"))
        if record.get("path") != f"{prefix_text}/{relative}":
            return False
        resolved_relative = relative
        if relative in prefix_symlinks:
            link = prefix_symlinks[relative]
            if link.get("target_kind") != "regular_file":
                return False
            resolved_relative = str(link.get("resolved_path"))
        target = prefix_files.get(resolved_relative)
        return bool(
            isinstance(target, Mapping)
            and record.get("resolved_path")
            == f"{prefix_text}/{resolved_relative}"
            and record.get("sha256") == target.get("sha256")
        )

    _require(
        prefix_executable_matches(launcher, "bin/mpiexec")
        and prefix_executable_matches(python, "bin/python"),
        "MPI launcher or Python differs from the environment-prefix manifest",
    )
    orchestration = _sequence(
        binding.get("orchestration_scripts"), name="bound orchestration scripts"
    )
    _require(
        list(orchestration) == list(source.get("orchestration_scripts", [])),
        "bound orchestration scripts differ from execution source identity",
    )
    runner_records = [
        record
        for record in orchestration
        if isinstance(record, Mapping)
        and record.get("path") == "scripts/remote_toy_mpi_runner.py"
    ]
    _require(
        len(runner_records) == 1
        and runner_records[0].get("kind") == "regular_file"
        and runner_records[0].get("size") == runner.get("size")
        and runner_records[0].get("sha256") == runner.get("sha256"),
        "executed MPI runner differs from the source manifest",
    )
    _require(
        binding.get("execution_source_identity_sha256")
        == source.get("manifest_sha256")
        and binding.get("tracked_files_manifest_sha256")
        == source.get("tracked_files_manifest_sha256")
        and binding.get("worktree_files_manifest_sha256")
        == source.get("worktree_files_manifest_sha256")
        and binding.get("orchestration_manifest_sha256")
        == source.get("orchestration_manifest_sha256")
        and binding.get("environment_prefix_manifest_sha256")
        == prefix.get("manifest_sha256")
        and binding.get("environment_regular_file_count")
        == prefix.get("regular_file_count")
        and binding.get("environment_symlink_count")
        == prefix.get("symlink_count"),
        "MPI release launch binding differs from source/environment manifests",
    )

    launch_environment = _mapping(
        binding.get("launch_environment"), name="scientific launch environment"
    )
    _require(
        set(launch_environment) == _LAUNCH_ENVIRONMENT_KEYS
        and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in launch_environment.items()
        )
        and launch_environment.get("USER") == launch_environment.get("LOGNAME")
        and launch_environment.get("HOME")
        == f"/home/{launch_environment.get('USER')}"
        and launch_environment.get("LANG") == "C.UTF-8"
        and launch_environment.get("LC_ALL") == "C.UTF-8"
        and launch_environment.get("TZ") == "UTC"
        and launch_environment.get("PYTHONHASHSEED") == "0"
        and launch_environment.get("PYTHONNOUSERSITE") == "1"
        and launch_environment.get("PYTHONUNBUFFERED") == "1"
        and all(launch_environment.get(key) == "1" for key in _THREAD_LIMIT_KEYS),
        "scientific launch environment is not the exact scrubbed environment",
    )
    prefix_text = str(prefix.get("prefix"))
    _require(
        launch_environment.get("PATH", "").split(os.pathsep)[0]
        == f"{prefix_text}/bin"
        and launch_environment.get("PYTHONPATH", "").endswith("/worktree/src")
        and binding.get("launch_environment_sha256")
        == _sha256_bytes(
            _canonical_json_bytes(launch_environment, ensure_ascii=True)
        ),
        "scientific launch environment path or SHA256 is invalid",
    )
    argv = _sequence(binding.get("argv"), name="MPI launch argv")
    binding_arguments = _sequence(
        binding.get("binding_arguments"), name="MPI binding arguments"
    )
    _require(
        len(argv) > len(binding_arguments) + 4
        and all(isinstance(value, str) and bool(value) for value in argv)
        and all(
            isinstance(value, str) and bool(value) for value in binding_arguments
        )
        and argv[0] == launcher.get("path")
        and list(argv[1 : 1 + len(binding_arguments)])
        == list(binding_arguments)
        and argv[1 + len(binding_arguments) : 3 + len(binding_arguments)]
        == ["-n", str(rank_count)]
        and argv[3 + len(binding_arguments)] == python.get("path")
        and argv[4 + len(binding_arguments)] == runner.get("path"),
        "MPI launch argv does not match the bound executables and rank policy",
    )
    for flag, expected_value in (
        ("--source-revision", commit),
        ("--source-tree-sha256", source_tree_sha256),
        ("--environment-sha256", environment_sha256),
        ("--execution-id", str(execution_id)),
        ("--expected-ranks", str(rank_count)),
    ):
        positions = [index for index, value in enumerate(argv) if value == flag]
        _require(
            len(positions) == 1
            and positions[0] + 1 < len(argv)
            and argv[positions[0] + 1] == expected_value,
            f"MPI launch argv has an invalid {flag} binding",
        )

    _require(
        set(receipt_payload)
        == {
            "schema",
            "launch_binding",
            "intent",
            "intent_payload_sha256",
            "started_utc",
            "finished_utc",
            "returncode",
            "post_run_verification",
            "stdout",
            "stderr",
        }
        and receipt_payload.get("schema")
        == "negfpy.remote_validation.release_launch.v2"
        and receipt_payload.get("launch_binding") == binding
        and receipt_payload.get("returncode") == 0
        and isinstance(receipt_payload.get("started_utc"), str)
        and bool(receipt_payload["started_utc"])
        and isinstance(receipt_payload.get("finished_utc"), str)
        and receipt_payload["started_utc"] <= receipt_payload["finished_utc"],
        "MPI release launch receipt payload is invalid",
    )
    _validate_recorded_attempt_artifact(
        attempt_root,
        receipt_payload.get("intent"),
        expected="mpi-release-launch-intent.json",
        name="recorded MPI release intent",
    )
    stdout_path = _validate_recorded_attempt_artifact(
        attempt_root,
        receipt_payload.get("stdout"),
        expected="mpi-release.stdout.log",
        name="recorded MPI release stdout",
    )
    stderr_path = _validate_recorded_attempt_artifact(
        attempt_root,
        receipt_payload.get("stderr"),
        expected="mpi-release.stderr.log",
        name="recorded MPI release stderr",
    )
    _require(
        receipt_payload.get("intent_payload_sha256")
        == intent.get("payload_sha256"),
        "MPI release receipt is not bound to its launch intent",
    )
    post_run = _mapping(
        receipt_payload.get("post_run_verification"),
        name="post-run source/environment verification",
    )
    _require(
        set(post_run)
        == {
            "passed",
            "checked_utc",
            "execution_source_identity_sha256",
            "environment_prefix_manifest_sha256",
        }
        and post_run.get("passed") is True
        and isinstance(post_run.get("checked_utc"), str)
        and bool(post_run["checked_utc"])
        and receipt_payload["started_utc"]
        <= post_run["checked_utc"]
        <= receipt_payload["finished_utc"]
        and post_run.get("execution_source_identity_sha256")
        == source.get("manifest_sha256")
        and post_run.get("environment_prefix_manifest_sha256")
        == prefix.get("manifest_sha256"),
        "post-run source/environment revalidation did not pass",
    )

    expected_launch_facts = {
        "schema": "negfpy.remote_validation.release_launch_facts.v1",
        "receipt_relative_path": "mpi-release-launch.json",
        "receipt_file_sha256": _sha256_file(receipt_path),
        "receipt_payload_sha256": receipt.get("payload_sha256"),
        "intent_relative_path": "mpi-release-launch-intent.json",
        "intent_file_sha256": _sha256_file(intent_path),
        "intent_payload_sha256": intent.get("payload_sha256"),
        "launcher_file_sha256": launcher.get("sha256"),
        "python_file_sha256": python.get("sha256"),
        "runner_file_sha256": runner.get("sha256"),
        "execution_source_identity_sha256": source.get("manifest_sha256"),
        "tracked_files_manifest_sha256": source.get(
            "tracked_files_manifest_sha256"
        ),
        "worktree_files_manifest_sha256": source.get(
            "worktree_files_manifest_sha256"
        ),
        "orchestration_manifest_sha256": source.get(
            "orchestration_manifest_sha256"
        ),
        "environment_prefix_manifest_sha256": prefix.get("manifest_sha256"),
        "launch_environment_sha256": binding.get("launch_environment_sha256"),
        "stdout_relative_path": "mpi-release.stdout.log",
        "stdout_file_sha256": _sha256_file(stdout_path),
        "stderr_relative_path": "mpi-release.stderr.log",
        "stderr_file_sha256": _sha256_file(stderr_path),
        "selected_rank_count": rank_count,
        "binding_arguments": list(binding_arguments),
        "post_run_verified": True,
        "returncode": 0,
    }
    _require(
        status_mpi.get("launch_receipt") == expected_launch_facts,
        "terminal MPI launch-receipt facts differ from the validated v2 receipt",
    )


def _body_digest(
    document: Mapping[str, Any],
    *,
    digest_key: str = "report_sha256",
    ensure_ascii: bool,
) -> str:
    core = {key: value for key, value in document.items() if key != digest_key}
    return _sha256_bytes(_canonical_json_bytes(core, ensure_ascii=ensure_ascii))


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactImportError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise ArtifactImportError(f"{name} must be a JSON array")
    return value


def _finite_number(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{name} must be a finite JSON number",
    )
    number = float(value)
    if nonnegative:
        _require(number >= 0.0, f"{name} must be nonnegative")
    return number


def _locked_comparison_policy(name: str) -> Mapping[str, Any]:
    """Return the preregistered effective policy for one release comparison."""

    _require(
        name in _FIXED_POINT_COMPARISONS | _REFINEMENT_COMPARISONS,
        f"unexpected multidimensional comparison name: {name!r}",
    )
    fixed_point = name in _FIXED_POINT_COMPARISONS
    return MappingProxyType(
        {
            "kind": (
                "zero_vs_born"
                if name == "zero_vs_born_endpoint"
                else (
                    "direct_vs_continuation"
                    if name == "direct_vs_continuation_endpoint"
                    else "numerical_refinement"
                )
            ),
            "certification_critical": name in _CRITICAL_COMPARISONS,
            "current_scope": "symmetrized_transport",
            "field_comparison_applicable": bool(
                fixed_point or name in _FIELD_COMPARISON_REFINEMENTS
            ),
            "field_gate_applicable": bool(
                fixed_point or name in _FIELD_GATE_REFINEMENTS
            ),
            "current_gate_applicable": True,
            "field_rtol": 1.0e-8,
            "field_atol": 1.0e-12,
            "current_rtol": 1.0e-8 if fixed_point else 5.0e-3,
            "current_atol": 1.0e-12,
        }
    )


def _validate_error_record(
    value: Any,
    *,
    name: str,
    rtol: float,
    atol: float,
) -> Mapping[str, Any]:
    """Validate a serialized scale-aware error against effective tolerances."""

    record = _mapping(value, name=name)
    difference = _finite_number(
        record.get("max_absolute_error"),
        name=f"{name}.max_absolute_error",
        nonnegative=True,
    )
    scale = _finite_number(
        record.get("scale"),
        name=f"{name}.scale",
        nonnegative=True,
    )
    relative = _finite_number(
        record.get("max_relative_error"),
        name=f"{name}.max_relative_error",
        nonnegative=True,
    )
    allowed = _finite_number(
        record.get("allowed_absolute_error"),
        name=f"{name}.allowed_absolute_error",
        nonnegative=True,
    )
    expected_allowed = atol + rtol * scale
    expected_relative = difference / scale if scale > 0.0 else 0.0
    _require(
        math.isclose(allowed, expected_allowed, rel_tol=1.0e-14, abs_tol=0.0),
        f"{name} allowed error does not match its effective tolerance",
    )
    _require(
        math.isclose(
            relative,
            expected_relative,
            rel_tol=1.0e-14,
            abs_tol=1.0e-300,
        ),
        f"{name} relative error is inconsistent",
    )
    _require(
        record.get("passed") is (difference <= allowed),
        f"{name} pass flag is inconsistent with its effective tolerance",
    )
    return record


def _validate_locked_comparison(
    value: Any,
    *,
    index: int,
) -> Mapping[str, Any]:
    """Validate scope, applicability, and tolerances for one comparison."""

    comparison = _mapping(value, name=f"multidimensional comparison {index}")
    name = comparison.get("name")
    _require(
        isinstance(name, str) and bool(name),
        f"multidimensional comparison {index} has an invalid name",
    )
    policy = _locked_comparison_policy(name)
    for key in (
        "kind",
        "certification_critical",
        "current_scope",
        "field_comparison_applicable",
        "field_gate_applicable",
        "current_gate_applicable",
        "field_rtol",
        "field_atol",
        "current_rtol",
        "current_atol",
    ):
        expected = policy[key]
        observed = comparison.get(key)
        if isinstance(expected, bool):
            matches = observed is expected
        else:
            matches = observed == expected
        _require(matches, f"comparison {name} has invalid effective {key}")

    field_errors = _sequence(
        comparison.get("field_errors"),
        name=f"comparison {name}.field_errors",
    )
    expected_field_count = (
        6 if policy["field_comparison_applicable"] else 0
    )
    _require(
        len(field_errors) == expected_field_count,
        f"comparison {name} has invalid field-comparison applicability",
    )
    field_keys: set[tuple[Any, Any]] = set()
    for field_index, raw_error in enumerate(field_errors):
        error = _validate_error_record(
            raw_error,
            name=f"comparison {name}.field_errors[{field_index}]",
            rtol=float(policy["field_rtol"]),
            atol=float(policy["field_atol"]),
        )
        field_keys.add((error.get("field"), error.get("component")))
        if policy["field_gate_applicable"]:
            _require(
                error.get("passed") is True,
                f"comparison {name} failed an applicable field gate",
            )
    if expected_field_count:
        _require(
            field_keys
            == {
                (field, component)
                for field in ("green", "sigma_pp")
                for component in ("retarded", "lesser", "greater")
            },
            f"comparison {name} field-error coverage is incomplete",
        )

    difference = _finite_number(
        comparison.get("current_max_absolute_error"),
        name=f"comparison {name}.current_max_absolute_error",
        nonnegative=True,
    )
    scale = _finite_number(
        comparison.get("current_scale"),
        name=f"comparison {name}.current_scale",
        nonnegative=True,
    )
    relative = _finite_number(
        comparison.get("current_max_relative_error"),
        name=f"comparison {name}.current_max_relative_error",
        nonnegative=True,
    )
    allowed = _finite_number(
        comparison.get("current_allowed_absolute_error"),
        name=f"comparison {name}.current_allowed_absolute_error",
        nonnegative=True,
    )
    expected_allowed = float(policy["current_atol"]) + (
        float(policy["current_rtol"]) * scale
    )
    expected_relative = difference / scale if scale > 0.0 else 0.0
    _require(
        math.isclose(allowed, expected_allowed, rel_tol=1.0e-14, abs_tol=0.0)
        and math.isclose(
            relative,
            expected_relative,
            rel_tol=1.0e-14,
            abs_tol=1.0e-300,
        ),
        f"comparison {name} current error metadata is inconsistent",
    )
    if policy["certification_critical"]:
        _require(
            difference <= allowed,
            f"comparison {name} failed its effective current gate",
        )

    line_shape_passed = True
    line_shape = comparison.get("line_shape")
    if name in _FIXED_POINT_COMPARISONS:
        _require(
            line_shape is None,
            f"comparison {name} unexpectedly contains a line-shape gate",
        )
    else:
        line_shape_record = _mapping(
            line_shape,
            name=f"comparison {name}.line_shape",
        )
        _require(
            line_shape_record.get("relative_tolerance") == 5.0e-3
            and line_shape_record.get("linewidth_absolute_tolerance")
            == 1.0e-12
            and line_shape_record.get("shift_absolute_tolerance") == 1.0e-8,
            f"comparison {name} has invalid line-shape tolerances",
        )
        linewidth_error = _validate_error_record(
            line_shape_record.get("linewidth_error"),
            name=f"comparison {name}.line_shape.linewidth_error",
            rtol=5.0e-3,
            atol=1.0e-12,
        )
        shift_error = _validate_error_record(
            line_shape_record.get("shift_error"),
            name=f"comparison {name}.line_shape.shift_error",
            rtol=5.0e-3,
            atol=1.0e-8,
        )
        line_shape_passed = bool(
            linewidth_error.get("passed") is True
            and shift_error.get("passed") is True
        )
        line_shape_failures = _sequence(
            line_shape_record.get("failures"),
            name=f"comparison {name}.line_shape.failures",
        )
        _require(
            line_shape_record.get("passed") is line_shape_passed
            and bool(line_shape_failures) is (not line_shape_passed),
            f"comparison {name} has inconsistent line-shape pass metadata",
        )
        if policy["certification_critical"]:
            _require(
                line_shape_passed,
                f"comparison {name} failed an effective line-shape gate",
            )
    comparison_failures = _sequence(
        comparison.get("failures"),
        name=f"comparison {name}.failures",
    )
    expected_pass = bool(
        difference <= allowed and line_shape_passed
    )
    _require(
        comparison.get("passed") is expected_pass
        and bool(comparison_failures) is (not expected_pass),
        f"comparison {name} has inconsistent pass/failure metadata",
    )
    return comparison


def _safe_relative_file(root: Path, relative_text: Any, *, name: str) -> Path:
    _require(
        isinstance(relative_text, str) and bool(relative_text),
        f"{name} must be a non-empty relative path",
    )
    relative = PurePosixPath(relative_text)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "." not in relative.parts,
        f"{name} is not a safe relative path: {relative_text!r}",
    )
    candidate = root.joinpath(*relative.parts)
    _require(not candidate.is_symlink(), f"{name} is a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise ArtifactImportError(f"{name} is missing: {relative_text}") from error
    _require(
        resolved.is_relative_to(root),
        f"{name} escapes the fetched artifact root",
    )
    _require(
        resolved.is_file() and not resolved.is_symlink(),
        f"{name} is not a regular file",
    )
    return resolved
def _required_parent_directories(relative_paths: Iterable[str]) -> set[str]:
    required: set[str] = set()
    for relative_text in relative_paths:
        parent = PurePosixPath(relative_text).parent
        while parent != PurePosixPath("."):
            required.add(parent.as_posix())
            parent = parent.parent
    return required



def _verify_outer_sha256sums(
    root: Path,
    *,
    expected_commit: str,
    expected_environment_lock_sha256: str,
    expected_fetch_receipt_sha256: str,
    expected_sha256sums_sha256: str,
) -> tuple[Mapping[str, str], Mapping[str, Any]]:
    manifest = root / "SHA256SUMS"
    fetch_receipt_path = root / "fetch-receipt.json"
    _require(
        fetch_receipt_path.is_file() and not fetch_receipt_path.is_symlink(),
        "fetched artifact root lacks fetch-receipt.json",
    )
    raw_fetch_receipt = fetch_receipt_path.read_bytes()
    actual_fetch_receipt_sha256 = _sha256_bytes(raw_fetch_receipt)
    _require(
        hmac.compare_digest(
            actual_fetch_receipt_sha256,
            expected_fetch_receipt_sha256,
        ),
        "raw fetch-receipt SHA256 differs from the trusted caller anchor",
    )
    _require(
        manifest.is_file() and not manifest.is_symlink(),
        "fetched artifact root lacks a regular SHA256SUMS file",
    )
    raw_manifest = manifest.read_bytes()
    actual_manifest_sha256 = _sha256_bytes(raw_manifest)
    _require(
        hmac.compare_digest(
            actual_manifest_sha256,
            expected_sha256sums_sha256,
        ),
        "raw SHA256SUMS digest differs from the trusted caller anchor",
    )
    try:
        manifest_text = raw_manifest.decode("ascii")
    except UnicodeDecodeError as error:
        raise ArtifactImportError("SHA256SUMS is not ASCII") from error
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_text.splitlines(),
        start=1,
    ):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(
            match is not None,
            f"invalid SHA256SUMS entry at line {line_number}",
        )
        assert match is not None
        expected, relative_text = match.groups()
        _require(
            relative_text not in entries,
            f"duplicate SHA256SUMS path {relative_text!r}",
        )
        target = _safe_relative_file(
            root,
            relative_text,
            name=f"SHA256SUMS entry {line_number}",
        )
        actual = _sha256_file(target)
        _require(
            hmac.compare_digest(actual, expected),
            f"SHA256 mismatch for {relative_text}",
        )
        entries[relative_text] = expected
    _require(entries, "SHA256SUMS is empty")

    symlinks = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    ]
    _require(not symlinks, "fetched artifact bundle contains symlinks")
    special_entries = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not path.is_symlink()
        and not path.is_file()
        and not path.is_dir()
    ]
    _require(
        not special_entries,
        "artifact bundle contains special filesystem entries",
    )
    regular_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path != manifest
    }
    observed_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    _require(
        observed_directories == _required_parent_directories(regular_files),
        "artifact bundle contains an empty or unreferenced directory",
    )
    # ``remote_heavy_runtime.fetch`` writes this receipt only after verifying
    # the exact remote checksum coverage.  It is therefore the sole expected
    # post-manifest file.
    expected_covered = regular_files - {"fetch-receipt.json"}
    _require(
        set(entries) == expected_covered,
        "SHA256SUMS does not cover exactly the artifact regular files",
    )

    fetch_receipt = _strict_json_bytes(
        raw_fetch_receipt,
        name="fetch-receipt.json",
    )
    receipt_schema = fetch_receipt.get("schema")
    _require(
        receipt_schema
        in {_REMOTE_FETCH_RECEIPT_SCHEMA, _LOCAL_PUBLICATION_RECEIPT_SCHEMA},
        "fetch receipt schema is unknown; no transport fallback is allowed",
    )
    verified_files = _sequence(
        fetch_receipt.get("verified_files"),
        name="fetch_receipt.verified_files",
    )
    _require(
        tuple(verified_files) == tuple(sorted(entries)),
        "fetch receipt file list differs from SHA256SUMS",
    )
    if receipt_schema == _REMOTE_FETCH_RECEIPT_SCHEMA:
        _require(
            set(fetch_receipt) == set(_REMOTE_RECEIPT_FIELDS),
            "remote fetch receipt has unexpected or missing fields",
        )
        _require(
            fetch_receipt.get("commit") == expected_commit,
            "remote fetch receipt commit differs from the trusted anchor",
        )
        _require(
            isinstance(fetch_receipt.get("fetched_utc"), str)
            and bool(fetch_receipt.get("fetched_utc")),
            "remote fetch receipt lacks its creation time",
        )
        _require(
            isinstance(fetch_receipt.get("remote"), str)
            and bool(fetch_receipt.get("remote")),
            "remote fetch receipt lacks its source location",
        )
    else:
        assert receipt_schema == _LOCAL_PUBLICATION_RECEIPT_SCHEMA
        _require(
            set(fetch_receipt) == set(_LOCAL_RECEIPT_FIELDS),
            "local publication receipt has unexpected or missing fields",
        )
        _require(
            fetch_receipt.get("commit") == expected_commit,
            "local publication receipt commit differs from the trusted anchor",
        )
        _require(
            fetch_receipt.get("environment_lock_sha256")
            == expected_environment_lock_sha256,
            "local publication receipt environment differs from the trusted anchor",
        )
        _require(
            fetch_receipt.get("artifact_root") == str(root),
            "local publication receipt does not identify the resolved artifact root",
        )
        _require(
            fetch_receipt.get("sha256sums_sha256")
            == expected_sha256sums_sha256,
            "local publication receipt manifest digest differs from the trusted anchor",
        )
        _require(
            isinstance(fetch_receipt.get("created_utc"), str)
            and bool(fetch_receipt.get("created_utc")),
            "local publication receipt lacks its creation time",
        )
    return (
        MappingProxyType(dict(sorted(entries.items()))),
        MappingProxyType(dict(fetch_receipt)),
    )


def _validate_publication(
    root: Path,
    *,
    expected_commit: str,
) -> tuple[
    str,
    int,
    Path,
    bytes,
    Mapping[str, Any],
    Path,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    current_path = root / "current-attempt.json"
    _require(
        current_path.is_file() and not current_path.is_symlink(),
        "current-attempt.json is absent",
    )
    current = _strict_json_file(current_path, name="current-attempt.json")
    _require(
        current.get("schema")
        == "negfpy.remote_validation.attempt_publication.v1",
        "current attempt publication schema is invalid",
    )
    commit = current.get("commit")
    _require(_is_commit(commit), "current attempt commit is not an exact SHA1")
    assert isinstance(commit, str)
    _require(
        hmac.compare_digest(commit, expected_commit),
        "current attempt commit differs from the trusted caller anchor",
    )
    _require(
        current.get("state") == "completed",
        "current artifact attempt is not completed and cannot be imported",
    )
    attempt = current.get("attempt")
    _require(
        isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 0,
        "current attempt number is invalid",
    )
    assert isinstance(attempt, int)
    expected_attempt = f"attempts/attempt-{attempt:03d}"
    _require(
        current.get("attempt_path") == expected_attempt,
        "current attempt path is inconsistent with its attempt number",
    )
    attempt_root = (root / expected_attempt).resolve(strict=True)
    _require(
        attempt_root.is_dir()
        and not attempt_root.is_symlink()
        and attempt_root.is_relative_to(root),
        "current attempt directory is unsafe",
    )
    # The dc60+ publication retains prior attempts, scaling repetitions,
    # the shared release campaign, and every task/checkpoint artifact.  The
    # current-attempt pointer selects the canonical publication.
    attempt_directories = sorted(
        path.name
        for path in (root / "attempts").glob("attempt-[0-9][0-9][0-9]")
        if path.is_dir() and not path.is_symlink()
    )
    _require(
        attempt_directories
        and attempt_directories[-1] == f"attempt-{attempt:03d}",
        "current attempt is not the newest immutable attempt directory",
    )
    _require(
        not (root / "job.lock").exists(),
        "current publication still has an active job lock",
    )
    for stale_name in (
        "production_toy_gate.json",
        "production_toy_gate.log",
        "multidimensional_toy_gate_failure.json",
    ):
        _require(
            not (root / stale_name).exists(),
            f"stale root publication artifact is forbidden: {stale_name}",
        )

    attempt_publication = _strict_json_file(
        _safe_relative_file(
            root,
            f"{expected_attempt}/publication.json",
            name="attempt publication",
        ),
        name="attempt publication",
    )
    _require(
        attempt_publication == current,
        "current-attempt.json differs from the selected attempt publication",
    )

    status_path = _safe_relative_file(
        root,
        current.get("status"),
        name="attempt status",
    )
    _require(
        _sha256_file(status_path) == current.get("status_sha256"),
        "attempt status SHA256 differs from publication",
    )
    status = _strict_json_file(status_path, name="attempt status")
    for document_name, document in (
        ("attempt status", status),
        (
            "job status",
            _strict_json_file(root / "job-status.json", name="job-status.json"),
        ),
    ):
        _require(
            document.get("schema") == "negfpy.remote_validation.job_status.v2"
            and document.get("state") == "completed"
            and document.get("returncode") == 0
            and document.get("commit") == commit
            and document.get("attempt") == attempt
            and document.get("attempt_path") == expected_attempt,
            f"{document_name} is not a matching successful terminal status",
        )
    job_status = _strict_json_file(root / "job-status.json", name="job-status.json")
    _require(job_status == status, "root and attempt terminal statuses differ")

    log_path = _safe_relative_file(
        root,
        current.get("log"),
        name="attempt log",
    )
    _require(
        _sha256_file(log_path) == current.get("log_sha256"),
        "attempt log SHA256 differs from publication",
    )
    report_path = _safe_relative_file(
        root,
        current.get("report"),
        name="production report",
    )
    _require(
        report_path.parent == attempt_root,
        "production report is not in the selected attempt directory",
    )
    raw_report = report_path.read_bytes()
    report_file_sha256 = _sha256_bytes(raw_report)
    _require(
        _is_sha256(current.get("report_sha256"))
        and report_file_sha256 == current.get("report_sha256"),
        "raw production report SHA256 differs from publication",
    )

    facts = _mapping(current.get("report_facts"), name="publication.report_facts")
    report_facts_path = _safe_relative_file(
        root,
        f"{expected_attempt}/report-facts.json",
        name="report facts",
    )
    report_facts = _strict_json_file(report_facts_path, name="report facts")
    _require(report_facts == facts, "report-facts sidecar differs from publication")
    _require(
        job_status.get("report_facts") == facts,
        "terminal status report facts differ from publication",
    )
    _require(
        facts.get("schema") == "negfpy.remote_validation.report_facts.v1"
        and facts.get("report_relative_path")
        == report_path.relative_to(root).as_posix()
        and facts.get("report_file_sha256") == report_file_sha256
        and facts.get("commit") == commit,
        "report facts do not identify the selected raw report",
    )

    checkpoint_index_path = _safe_relative_file(
        root,
        f"{expected_attempt}/checkpoint-index.json",
        name="checkpoint index",
    )
    checkpoint_index = _strict_json_file(
        checkpoint_index_path,
        name="checkpoint index",
    )

    # Reopen every artifact named by the current publication.  The scaling
    # selection may come from an earlier immutable attempt on an exact retry;
    # all release evidence must belong to the selected current attempt.
    published_files = {
        "scaling_selection": False,
        "mpi_release_launch_intent": True,
        "mpi_release_launch": True,
        "mpi_release_stdout": True,
        "mpi_release_stderr": True,
        "mpi_runner_result": True,
        "mpi_execution_manifest": True,
        "mpi_campaign_manifest": True,
    }
    for key, current_attempt_only in published_files.items():
        relative = current.get(key)
        digest = current.get(f"{key}_sha256")
        selected = _safe_relative_file(root, relative, name=key)
        _require(
            _is_sha256(digest) and _sha256_file(selected) == digest,
            f"{key} SHA256 differs from current publication",
        )
        selected_relative = selected.relative_to(root).as_posix()
        if current_attempt_only:
            _require(
                selected.parent == attempt_root,
                f"{key} is not archived in the current attempt",
            )
        else:
            _require(
                re.fullmatch(
                    r"attempts/attempt-[0-9]{3}/mpi-scaling/"
                    r"scaling-selection\.json",
                    selected_relative,
                )
                is not None,
                "scaling selection is outside an immutable attempt",
            )

    status_scaling = _mapping(
        status.get("scaling_facts"), name="status.scaling_facts"
    )
    _require(
        status_scaling.get("selection_relative_path")
        == current.get("scaling_selection"),
        "current scaling selection differs from terminal status",
    )
    status_mpi = _mapping(
        status.get("mpi_execution_facts"), name="status.mpi_execution_facts"
    )
    _require(
        facts.get("mpi_execution") == status_mpi
        and status_mpi.get("release_certified") is True,
        "current MPI execution evidence is absent or uncertified",
    )
    _validate_release_launch_provenance(
        root,
        attempt=attempt,
        commit=commit,
        current=current,
        status_mpi=status_mpi,
    )
    return (
        commit,
        attempt,
        report_path,
        raw_report,
        MappingProxyType(dict(facts)),
        checkpoint_index_path,
        MappingProxyType(dict(checkpoint_index)),
        MappingProxyType(dict(job_status)),
    )


def _validate_identity_bundle(
    root: Path,
    *,
    commit: str,
    attempt: int,
    expected_environment_lock_sha256: str,
    facts: Mapping[str, Any],
    job_status: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    identity_path = root / "identity" / "identity-manifest.json"
    identity = _strict_json_file(identity_path, name="identity manifest")
    _require(
        identity.get("schema") == "negfpy.remote_validation.identity_bundle.v1"
        and identity.get("commit") == commit,
        "identity bundle schema or commit is invalid",
    )
    _require(
        job_status.get("identity_bundle") == identity,
        "terminal status identity bundle differs from archived identity manifest",
    )
    tree = identity.get("git_tree")
    source_sha = identity.get("source_tree_sha256")
    lock_sha = identity.get("explicit_lock_sha256")
    _require(_is_commit(tree), "identity bundle git tree is invalid")
    _require(_is_sha256(source_sha), "identity source-tree SHA256 is invalid")
    _require(_is_sha256(lock_sha), "identity environment-lock SHA256 is invalid")
    assert isinstance(lock_sha, str)
    _require(
        hmac.compare_digest(lock_sha, expected_environment_lock_sha256),
        "identity environment-lock SHA256 differs from the trusted caller anchor",
    )
    _require(
        facts.get("git_tree") == tree
        and facts.get("source_tree_sha256") == source_sha
        and facts.get("environment_sha256") == lock_sha,
        "report facts differ from the archived source/environment identity",
    )

    identity_root = identity_path.parent
    archived_files = _mapping(
        identity.get("archived_files"),
        name="identity_manifest.archived_files",
    )
    _require(
        set(archived_files) == set(_ARCHIVED_IDENTITY_FILES),
        "identity bundle does not name the exact immutable deployment files",
    )
    for name in _ARCHIVED_IDENTITY_FILES:
        expected_sha = archived_files.get(name)
        _require(
            _is_sha256(expected_sha),
            f"identity bundle archived SHA256 is invalid for {name}",
        )
        archived_path = _safe_relative_file(
            identity_root,
            name,
            name=f"archived identity file {name}",
        )
        _require(
            hmac.compare_digest(_sha256_file(archived_path), str(expected_sha)),
            f"identity bundle archived SHA256 differs for {name}",
        )
    verified_incoming = _sequence(
        identity.get("verified_incoming_files"),
        name="identity_manifest.verified_incoming_files",
    )
    _require(
        set(verified_incoming)
        == {
            "deployment-manifest.json",
            "environment-explicit.lock",
            "environment-manifest.json",
            "negfpy.bundle",
        }
        and len(verified_incoming) == 4,
        "identity bundle incoming-file verification set is invalid",
    )

    explicit_lock = _safe_relative_file(
        identity_root,
        identity.get("explicit_lock"),
        name="explicit environment lock",
    )
    _require(
        _sha256_file(explicit_lock) == lock_sha,
        "explicit environment lock SHA256 is invalid",
    )
    lock_digest = _strict_json_file(
        _safe_relative_file(
            identity_root,
            "lock-digest.json",
            name="lock digest",
        ),
        name="lock digest",
    )
    _require(
        lock_digest.get("schema")
        == "negfpy.remote_validation.lock_digest.v1"
        and lock_digest.get("commit") == commit
        and lock_digest.get("git_tree") == tree
        and lock_digest.get("source_tree_sha256") == source_sha
        and lock_digest.get("explicit_lock") == identity.get("explicit_lock")
        and lock_digest.get("explicit_lock_sha256") == lock_sha,
        "lock-digest sidecar is inconsistent with the identity bundle",
    )
    environment_manifest = _strict_json_file(
        _safe_relative_file(
            identity_root,
            "environment-manifest.json",
            name="environment manifest",
        ),
        name="environment manifest",
    )
    _require(
        environment_manifest.get("schema")
        == "negfpy.remote_validation.environment.v2"
        and environment_manifest.get("explicit_lock_sha256") == lock_sha,
        "environment manifest does not identify the explicit lock",
    )
    deployment = _strict_json_file(
        _safe_relative_file(
            identity_root,
            "deployment-manifest.json",
            name="deployment manifest",
        ),
        name="deployment manifest",
    )
    deployment_source = _mapping(
        deployment.get("source"),
        name="deployment_manifest.source",
    )
    deployment_environment = _mapping(
        deployment.get("environment"),
        name="deployment_manifest.environment",
    )
    deployment_threads = _mapping(
        deployment.get("thread_limits"),
        name="deployment_manifest.thread_limits",
    )
    expected_threads = {name: "1" for name in _THREAD_LIMIT_KEYS}
    _require(
        deployment.get("schema") == "negfpy.remote_validation.deployment.v2"
        and deployment_source.get("commit") == commit
        and deployment_source.get("tree") == tree
        and deployment_source.get("dirty") is False
        and deployment_environment.get("explicit_lock")
        == identity.get("explicit_lock")
        and deployment_environment.get("explicit_lock_sha256") == lock_sha
        and dict(deployment_threads) == expected_threads,
        "deployment manifest source, lock, or thread-limit identity is invalid",
    )

    remote_environment_path = _safe_relative_file(
        identity_root,
        identity.get("remote_environment"),
        name="remote environment manifest",
    )
    _require(
        _sha256_file(remote_environment_path)
        == identity.get("remote_environment_sha256"),
        "remote environment manifest SHA256 is invalid",
    )
    remote_environment = _strict_json_file(
        remote_environment_path,
        name="remote environment manifest",
    )
    _require(
        remote_environment.get("schema")
        == "negfpy.remote_validation.remote_environment.v1"
        and remote_environment.get("commit") == commit
        and remote_environment.get("tree") == tree
        and remote_environment.get("source_tree_sha256") == source_sha
        and remote_environment.get("explicit_lock_sha256") == lock_sha
        and remote_environment.get("thread_limits") == expected_threads,
        "remote environment manifest identity is inconsistent",
    )
    execution_identity = _strict_json_file(
        _safe_relative_file(
            root,
            f"attempts/attempt-{attempt:03d}/execution-identity.json",
            name="attempt execution identity",
        ),
        name="attempt execution identity",
    )
    mismatches = [
        field
        for field in _REMOTE_IMMUTABLE_FIELDS
        if execution_identity.get(field) != remote_environment.get(field)
    ]
    _require(
        not mismatches,
        "attempt execution identity differs from immutable deployment: "
        + ", ".join(mismatches),
    )
    return MappingProxyType(dict(identity)), MappingProxyType(
        dict(remote_environment)
    )


def _validate_local_publication(
    root: Path,
    *,
    expected_commit: str,
) -> tuple[
    str,
    int,
    Path,
    bytes,
    Mapping[str, Any],
    Path,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    """Validate a completed, flat, locally published attempt."""

    status_path = _safe_relative_file(root, "status.json", name="local status")
    status = _strict_json_file(status_path, name="local status")
    _require(
        set(status) == set(_LOCAL_STATUS_FIELDS),
        "local status has unexpected or missing fields",
    )
    commit = status.get("commit")
    attempt = status.get("attempt")
    _require(
        status.get("schema") == _LOCAL_STATUS_SCHEMA
        and status.get("state") == "completed"
        and status.get("returncode") == 0
        and commit == expected_commit,
        "local status is not a matching successful terminal status",
    )
    _require(
        isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and attempt > 0,
        "local status attempt number is invalid",
    )
    assert isinstance(attempt, int)
    _require(
        root.name == f"attempt-{attempt:03d}",
        "local artifact-root basename differs from the status attempt",
    )
    observed_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    _require(
        observed_directories == {"checkpoints"},
        "local artifact directory topology is not exactly flat",
    )
    _require(
        status.get("artifact_root") == str(root),
        "local status does not identify the resolved artifact root",
    )
    for field in ("started_utc", "finished_utc"):
        _require(
            isinstance(status.get(field), str) and bool(status.get(field)),
            f"local status {field} is absent",
        )

    paths: dict[str, Path] = {}
    for stem, expected_relative in _LOCAL_STATUS_FILE_FIELDS.items():
        _require(
            status.get(stem) == expected_relative,
            f"local status {stem} path is not the locked flat-root path",
        )
        target = _safe_relative_file(
            root,
            expected_relative,
            name=f"local {stem}",
        )
        expected_sha = status.get(f"{stem}_sha256")
        _require(
            _is_sha256(expected_sha)
            and hmac.compare_digest(_sha256_file(target), str(expected_sha)),
            f"local {stem} SHA256 differs from status",
        )
        paths[stem] = target

    _require(
        paths["exit_code"].read_bytes() == b"0\n",
        "local exit-code.txt is not the exact successful exit record",
    )
    _require(
        paths["log"].stat().st_size > 0 and paths["launch"].stat().st_size > 0,
        "local log or launch record is empty",
    )

    report_path = paths["report"]
    raw_report = report_path.read_bytes()
    report_file_sha256 = _sha256_bytes(raw_report)
    _require(
        report_file_sha256 == status.get("report_sha256"),
        "raw local production report SHA256 differs from status",
    )
    facts = _strict_json_file(paths["report_facts"], name="local report facts")
    _require(
        set(facts) == set(_REPORT_FACTS_FIELDS),
        "local report-facts sidecar has unexpected or missing fields",
    )
    _require(
        facts.get("schema") == _REPORT_FACTS_SCHEMA
        and facts.get("report_relative_path") == "production_toy_gate.json"
        and facts.get("report_file_sha256") == report_file_sha256
        and facts.get("schema_version") == 4
        and facts.get("protocol") == _PROTOCOL
        and facts.get("protocol_version") == 2
        and facts.get("release_certified") is True
        and facts.get("multidimensional_outcome_count") == 13
        and facts.get("multidimensional_backend") == "fft"
        and facts.get("harmonic_preflight_point_count") == 7
        and facts.get("bounded_backends") == ["serial", "threaded", "fft"]
        and facts.get("open_system_run_count") == 13
        and facts.get("checkpoint_count") == 13
        and facts.get("commit") == commit,
        "local report facts do not identify the certified flat-root report",
    )

    checkpoint_index_path = paths["checkpoint_index"]
    checkpoint_index = _strict_json_file(
        checkpoint_index_path,
        name="local checkpoint index",
    )
    _require(
        set(checkpoint_index)
        == {"schema", "attempt", "checkpoint_root", "count", "records"}
        and checkpoint_index.get("schema") == _CHECKPOINT_INDEX_SCHEMA
        and checkpoint_index.get("attempt") == attempt
        and checkpoint_index.get("checkpoint_root") == "checkpoints"
        and checkpoint_index.get("count") == 13,
        "local checkpoint-index sidecar is invalid",
    )
    return (
        str(commit),
        attempt,
        report_path,
        raw_report,
        MappingProxyType(dict(facts)),
        checkpoint_index_path,
        MappingProxyType(dict(checkpoint_index)),
        MappingProxyType(dict(status)),
    )


def _validate_local_identity_bundle(
    root: Path,
    *,
    commit: str,
    expected_environment_lock_sha256: str,
    facts: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Validate and normalize the flat attempt's execution identity."""

    identity_path = _safe_relative_file(
        root,
        "execution-identity.json",
        name="local execution identity",
    )
    execution = _strict_json_file(
        identity_path,
        name="local execution identity",
    )
    _require(
        set(execution) == set(_LOCAL_EXECUTION_IDENTITY_FIELDS),
        "local execution identity has unexpected or missing fields",
    )
    _require(
        execution.get("schema") == _LOCAL_EXECUTION_IDENTITY_SCHEMA
        and execution.get("commit") == commit,
        "local execution identity schema or commit is invalid",
    )
    _require(
        isinstance(execution.get("created_utc"), str)
        and bool(execution.get("created_utc")),
        "local execution identity lacks its creation time",
    )
    tree = execution.get("tree")
    source_sha = execution.get("source_tree_sha256")
    lock_sha = execution.get("explicit_lock_sha256")
    _require(_is_commit(tree), "local execution git tree is invalid")
    _require(_is_sha256(source_sha), "local execution source SHA256 is invalid")
    _require(_is_sha256(lock_sha), "local execution lock SHA256 is invalid")
    _require(
        hmac.compare_digest(str(lock_sha), expected_environment_lock_sha256),
        "local execution lock differs from the trusted environment anchor",
    )
    _require(
        facts.get("git_tree") == tree
        and facts.get("source_tree_sha256") == source_sha
        and facts.get("environment_sha256") == lock_sha,
        "local report facts differ from the execution source/environment identity",
    )

    required_strings = (
        "hostname",
        "platform",
        "machine",
        "python_executable",
        "python_version",
        "package_source",
        "cpu_model",
    )
    for field in required_strings:
        _require(
            isinstance(execution.get(field), str) and bool(execution.get(field)),
            f"local execution identity {field} is absent",
        )
    _require(
        Path(str(execution["python_executable"])).is_absolute()
        and Path(str(execution["package_source"])).is_absolute(),
        "local execution interpreter and package-source paths must be absolute",
    )

    distributions = _mapping(
        execution.get("distributions"),
        name="local execution distributions",
    )
    required_distributions = {"h5py", "negfpy", "numpy", "scipy", "threadpoolctl"}
    _require(
        required_distributions.issubset(distributions)
        and all(
            isinstance(name, str)
            and bool(name)
            and isinstance(version, str)
            and bool(version)
            for name, version in distributions.items()
        ),
        "local execution distribution versions are incomplete or invalid",
    )
    expected_threads = {name: "1" for name in _THREAD_LIMIT_KEYS}
    _require(
        execution.get("thread_limits") == expected_threads,
        "local execution thread limits are not the locked single-thread values",
    )

    cpu_count = execution.get("cpu_count")
    total_memory = execution.get("memory_total_bytes")
    available_memory = execution.get("memory_available_bytes")
    _require(
        isinstance(cpu_count, int)
        and not isinstance(cpu_count, bool)
        and cpu_count > 0,
        "local execution CPU count is invalid",
    )
    _require(
        isinstance(total_memory, int)
        and not isinstance(total_memory, bool)
        and total_memory > 0
        and isinstance(available_memory, int)
        and not isinstance(available_memory, bool)
        and 0 <= available_memory <= total_memory,
        "local execution memory inventory is invalid",
    )

    identity_bundle = MappingProxyType(
        {
            "schema": _LOCAL_EXECUTION_IDENTITY_SCHEMA,
            "commit": commit,
            "git_tree": tree,
            "source_tree_sha256": source_sha,
            "explicit_lock_sha256": lock_sha,
        }
    )
    return identity_bundle, MappingProxyType(dict(execution))


def _validate_code_identity(
    value: Any,
    *,
    commit: str,
    source_sha: str,
    expected: Mapping[str, Any] | None,
    name: str,
) -> Mapping[str, Any]:
    identity = _mapping(value, name=name)
    _require(
        set(identity) == set(_IDENTITY_KEYS),
        f"{name} has unexpected or missing identity fields",
    )
    _require(
        identity.get("git_revision") == commit
        and identity.get("source_tree_sha256") == source_sha
        and identity.get("dtype") == "complex128",
        f"{name} does not match the release source identity",
    )
    _require(
        isinstance(identity.get("blas_runtime"), Sequence)
        and not isinstance(identity.get("blas_runtime"), (str, bytes))
        and bool(identity.get("blas_runtime")),
        f"{name} lacks BLAS runtime provenance",
    )
    if expected is not None:
        _require(identity == expected, f"{name} differs from the common identity")
    return identity


def _require_close(
    value: Any,
    expected: float,
    *,
    name: str,
    rel_tol: float = 1.0e-12,
    abs_tol: float = 1.0e-15,
) -> float:
    number = _finite_number(value, name=name)
    _require(
        math.isclose(number, expected, rel_tol=rel_tol, abs_tol=abs_tol),
        f"{name} is inconsistent with independently derived report evidence",
    )
    return number


def _validated_harmonic_preflight_points(
    document: Mapping[str, Any],
    *,
    expected_environment_sha256: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Validate and join all seven harmonic points to their gate statuses."""

    multidimensional = _mapping(
        document.get("multidimensional_scba"),
        name="report.multidimensional_scba",
    )
    harmonic = _mapping(
        multidimensional.get("harmonic_preflight"),
        name="multidimensional.harmonic_preflight",
    )
    failures = _sequence(
        harmonic.get("failures"),
        name="harmonic_preflight.failures",
    )
    harmonic_environment = harmonic.get("environment_sha256")
    _require(
        harmonic.get("schema")
        == "negfpy.anharmonic.harmonic_same_grid_preflight"
        and harmonic.get("schema_version") == 2
        and harmonic.get("protocol") == _PROTOCOL
        and harmonic.get("scope")
        == "base_plus_six_preregistered_numerical_points"
        and harmonic.get("applicable") is True
        and harmonic.get("passed") is True
        and not failures
        and harmonic.get("point_count") == 7
        and harmonic.get("all_seven_archived") is True
        and harmonic.get("required_directional_comparison_count") == 4
        and harmonic.get("critical_comparisons_passed") is True
        and harmonic.get("current_refinement_rtol") == 5.0e-3
        and harmonic.get("spectral_moment_limit") == 5.0e-3
        and harmonic.get("backend") == "fft"
        and harmonic.get("hilbert_backend") == "fft"
        and harmonic.get("surface_gf_method") == "analytic_diagonal"
        and _is_sha256(harmonic_environment),
        "seven-point harmonic preflight evidence is invalid",
    )
    if expected_environment_sha256 is not None:
        _require(
            harmonic_environment == expected_environment_sha256,
            "harmonic preflight environment differs from the release lock",
        )

    raw_points = _sequence(
        harmonic.get("points"),
        name="harmonic_preflight.points",
    )
    raw_comparisons = _sequence(
        harmonic.get("comparisons"),
        name="harmonic_preflight.comparisons",
    )
    _require(
        len(raw_points) == len(_HARMONIC_POINT_CONTRACT) == 7
        and len(raw_comparisons) == len(_HARMONIC_COMPARISON_CONTRACT) == 6,
        "harmonic preflight lacks the exact seven points and six comparisons",
    )

    comparisons_by_right: dict[str, Mapping[str, Any]] = {}
    diagnostic_names: list[str] = []
    for index, raw_comparison in enumerate(raw_comparisons):
        comparison = _mapping(
            raw_comparison,
            name=f"harmonic comparison {index}",
        )
        right_label = comparison.get("right_label")
        _require(
            isinstance(right_label, str)
            and right_label in _HARMONIC_COMPARISON_CONTRACT
            and right_label not in comparisons_by_right,
            f"harmonic comparison {index} has an unknown or duplicate point",
        )
        policy = _HARMONIC_COMPARISON_CONTRACT[right_label]
        comparison_failures = _sequence(
            comparison.get("failures"),
            name=f"harmonic comparison {right_label}.failures",
        )
        absolute_error = _finite_number(
            comparison.get("absolute_error"),
            name=f"harmonic comparison {right_label}.absolute_error",
            nonnegative=True,
        )
        absolute_tolerance = _finite_number(
            comparison.get("absolute_tolerance"),
            name=f"harmonic comparison {right_label}.absolute_tolerance",
            nonnegative=True,
        )
        relative_tolerance = _finite_number(
            comparison.get("relative_tolerance"),
            name=f"harmonic comparison {right_label}.relative_tolerance",
            nonnegative=True,
        )
        current_scale = _finite_number(
            comparison.get("current_scale"),
            name=f"harmonic comparison {right_label}.current_scale",
            nonnegative=True,
        )
        _require(
            current_scale > 0.0,
            f"harmonic comparison {right_label} has zero current scale",
        )
        relative_error = _finite_number(
            comparison.get("relative_error"),
            name=f"harmonic comparison {right_label}.relative_error",
            nonnegative=True,
        )
        _require_close(
            relative_error,
            absolute_error / current_scale,
            name=f"harmonic comparison {right_label}.relative_error",
        )
        expected_passed = absolute_error <= (
            absolute_tolerance + relative_tolerance * current_scale
        )
        _require(
            comparison.get("name") == policy["name"]
            and comparison.get("production_comparison_name")
            == policy["production_name"]
            and comparison.get("left_label") == "base"
            and comparison.get("certification_critical")
            is policy["critical"]
            and comparison.get("relative_tolerance") == 5.0e-3
            and comparison.get("absolute_tolerance") == 1.0e-12
            and comparison.get("passed") is expected_passed
            and bool(comparison_failures) is (not expected_passed),
            f"harmonic comparison status is inconsistent: {right_label}",
        )
        if policy["critical"]:
            _require(
                expected_passed,
                f"certification-critical harmonic comparison failed: {right_label}",
            )
        else:
            diagnostic_names.append(str(policy["name"]))
        comparisons_by_right[right_label] = comparison

    declared_diagnostics = _sequence(
        harmonic.get("diagnostic_comparison_names"),
        name="harmonic_preflight.diagnostic_comparison_names",
    )
    _require(
        list(declared_diagnostics) == diagnostic_names,
        "harmonic diagnostic comparison classification is inconsistent",
    )

    validated: list[Mapping[str, Any]] = []
    for index, (raw_point, contract) in enumerate(
        zip(raw_points, _HARMONIC_POINT_CONTRACT)
    ):
        point = _mapping(raw_point, name=f"harmonic point {index}")
        (
            label,
            spacing,
            cutoff,
            eta_lead,
            mixing,
            n_positive,
            role,
        ) = contract
        grid = _mapping(point.get("grid"), name=f"harmonic point {label}.grid")
        point_failures = _sequence(
            point.get("failures"),
            name=f"harmonic point {label}.failures",
        )
        spectral_limit = _finite_number(
            point.get("spectral_moment_limit"),
            name=f"harmonic point {label}.spectral_moment_limit",
            nonnegative=True,
        )
        spectral_residual = _finite_number(
            point.get("spectral_moment_residual"),
            name=f"harmonic point {label}.spectral_moment_residual",
            nonnegative=True,
        )
        transport_current = _finite_number(
            point.get("transport_current"),
            name=f"harmonic point {label}.transport_current",
        )
        _require(
            point.get("label") == label
            and point.get("g") == 0.0
            and point.get("bias") == 0.2
            and point.get("passed") is True
            and not point_failures
            and point.get("environment_sha256") == harmonic_environment
            and point.get("backend") == "fft"
            and point.get("hilbert_backend") == "fft"
            and point.get("surface_gf_method") == "analytic_diagonal"
            and grid.get("spacing") == spacing
            and grid.get("cutoff") == cutoff
            and grid.get("eta_lead") == eta_lead
            and grid.get("mixing") == mixing
            and grid.get("eta_device") == 0.0
            and grid.get("n_positive") == n_positive
            and grid.get("n_frequency") == 2 * n_positive + 1
            and math.isclose(
                spacing * n_positive,
                cutoff,
                rel_tol=1.0e-14,
                abs_tol=1.0e-14,
            )
            and spectral_limit == 5.0e-3
            and spectral_residual <= spectral_limit
            and transport_current > 0.0,
            f"harmonic point {label} differs from the locked preflight contract",
        )
        comparison = comparisons_by_right.get(label)
        _require(
            (role == "reference" and comparison is None)
            or (role != "reference" and comparison is not None),
            f"harmonic point {label} has inconsistent comparison coverage",
        )
        validated.append(
            MappingProxyType(
                {
                    "label": label,
                    "certification_role": role,
                    "spacing": spacing,
                    "cutoff": cutoff,
                    "eta_lead": eta_lead,
                    "mixing": mixing,
                    "transport_current": transport_current,
                    "spectral_moment_residual": spectral_residual,
                    "point_passed": True,
                    "comparison_passed": (
                        None if comparison is None else comparison["passed"]
                    ),
                }
            )
        )
    return tuple(validated)


def _validated_cost_accuracy_rows(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Validate the three declared tiers against authoritative timings."""

    raw_rows = _sequence(document.get("cost_accuracy"), name="cost_accuracy")
    _require(
        len(raw_rows) == len(_COST_ACCURACY_CONTRACT) == 3,
        "cost/accuracy evidence must contain exactly three tiers",
    )
    rows: list[Mapping[str, Any]] = []
    for index, (raw_row, contract) in enumerate(
        zip(raw_rows, _COST_ACCURACY_CONTRACT)
    ):
        row = _mapping(raw_row, name=f"cost_accuracy[{index}]")
        _require(
            set(row)
            == {
                "method",
                "accuracy_role",
                "asymptotic_dense_time",
                "conserving",
                "measured",
                "release_use",
                "self_consistent",
            },
            f"cost/accuracy row {index} has unexpected or missing fields",
        )
        for key, expected in contract.items():
            actual = row.get(key)
            if isinstance(expected, bool):
                consistent = actual is expected
            else:
                consistent = actual == expected
            _require(
                consistent,
                f"cost/accuracy row {index} has inconsistent {key}",
            )
        rows.append(row)

    harmonic_measured = _mapping(
        rows[0].get("measured"),
        name="cost_accuracy.harmonic.measured",
    )
    _require(
        set(harmonic_measured)
        == {
            "count",
            "maximum_wall_seconds",
            "median_wall_seconds",
            "minimum_wall_seconds",
            "note",
            "total_wall_seconds",
        }
        and harmonic_measured.get("count") == 0
        and harmonic_measured.get("maximum_wall_seconds") is None
        and harmonic_measured.get("median_wall_seconds") is None
        and harmonic_measured.get("minimum_wall_seconds") is None
        and harmonic_measured.get("total_wall_seconds") == 0.0
        and harmonic_measured.get("note")
        == "No separate harmonic timing object is part of this combined report.",
        "harmonic cost/accuracy timing summary is inconsistent",
    )

    open_section = _mapping(
        document.get("open_three_mode"),
        name="report.open_three_mode",
    )
    open_runs = _sequence(open_section.get("runs"), name="open_three_mode.runs")
    born_timings: list[float] = []
    for index, raw_run in enumerate(open_runs):
        run = _mapping(raw_run, name=f"open run {index}")
        timings = _mapping(run.get("timings"), name=f"open run {index}.timings")
        born_timings.append(
            _finite_number(
                timings.get("production_born_wall_seconds"),
                name=f"open run {index}.production_born_wall_seconds",
                nonnegative=True,
            )
        )
    _require(
        len(born_timings) == 13,
        "Born timing evidence does not cover all 13 open-system runs",
    )
    born_sorted = sorted(born_timings)
    born_measured = _mapping(
        rows[1].get("measured"),
        name="cost_accuracy.one_shot_born.measured",
    )
    _require(
        set(born_measured)
        == {
            "count",
            "maximum_wall_seconds",
            "median_wall_seconds",
            "minimum_wall_seconds",
            "total_wall_seconds",
        }
        and born_measured.get("count") == 13,
        "Born cost/accuracy timing summary has invalid shape or count",
    )
    for key, expected in (
        ("minimum_wall_seconds", born_sorted[0]),
        ("median_wall_seconds", born_sorted[len(born_sorted) // 2]),
        ("maximum_wall_seconds", born_sorted[-1]),
        ("total_wall_seconds", math.fsum(born_timings)),
    ):
        _require_close(
            born_measured.get(key),
            expected,
            name=f"cost_accuracy.one_shot_born.measured.{key}",
            abs_tol=1.0e-12,
        )

    multidimensional = _mapping(
        document.get("multidimensional_scba"),
        name="report.multidimensional_scba",
    )
    scba_total = _finite_number(
        multidimensional.get("total_elapsed_seconds"),
        name="multidimensional_scba.total_elapsed_seconds",
        nonnegative=True,
    )
    scba_measured = _mapping(
        rows[2].get("measured"),
        name="cost_accuracy.self_consistent_born.measured",
    )
    _require(
        set(scba_measured)
        == {
            "aggregation_basis",
            "count",
            "maximum_wall_seconds",
            "median_wall_seconds",
            "minimum_wall_seconds",
            "note",
            "records",
            "total_wall_seconds",
        }
        and scba_measured.get("aggregation_basis")
        == "multidimensional_report.total_elapsed_seconds"
        and scba_measured.get("count") == 1
        and scba_measured.get("note")
        == (
            "records include nested outcome/seed timings and can overlap; "
            "aggregate statistics use the authoritative top-level total "
            "when it is available."
        ),
        "SCBA cost/accuracy timing summary has invalid shape or semantics",
    )
    for key in (
        "minimum_wall_seconds",
        "median_wall_seconds",
        "maximum_wall_seconds",
        "total_wall_seconds",
    ):
        _require_close(
            scba_measured.get(key),
            scba_total,
            name=f"cost_accuracy.self_consistent_born.measured.{key}",
            abs_tol=1.0e-12,
        )
    timing_records = _sequence(
        scba_measured.get("records"),
        name="cost_accuracy.self_consistent_born.measured.records",
    )
    record_paths: set[str] = set()
    authoritative_total_count = 0
    for index, raw_record in enumerate(timing_records):
        record = _mapping(raw_record, name=f"SCBA timing record {index}")
        path = record.get("path")
        wall_seconds = _finite_number(
            record.get("wall_seconds"),
            name=f"SCBA timing record {index}.wall_seconds",
            nonnegative=True,
        )
        _require(
            set(record) == {"path", "wall_seconds"}
            and isinstance(path, str)
            and bool(path)
            and path not in record_paths,
            f"SCBA timing record {index} is malformed or duplicated",
        )
        record_paths.add(path)
        if path == "$.total_elapsed_seconds":
            authoritative_total_count += 1
            _require_close(
                wall_seconds,
                scba_total,
                name="SCBA authoritative total timing record",
                abs_tol=1.0e-12,
            )
    _require(
        timing_records and authoritative_total_count == 1,
        "SCBA timing records lack one authoritative top-level total",
    )
    return tuple(MappingProxyType(dict(row)) for row in rows)


def _validated_multidimensional_weak_coupling_summary(
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate every nine-DOF pole ratio and derive campaign extrema."""
    multidimensional = _mapping(
        document.get("multidimensional_scba"),
        name="report.multidimensional_scba",
    )
    weak = _mapping(
        multidimensional.get("weak_coupling"),
        name="multidimensional.weak_coupling",
    )
    _require(
        set(weak) == {"coverage", "definition", "limit", "passed", "runs"}
        and weak.get("definition")
        == _MULTIDIMENSIONAL_WEAK_COUPLING_DEFINITION
        and weak.get("coverage") is True
        and weak.get("passed") is True
        and weak.get("limit") == 0.1,
        "nine-DOF weak-coupling summary is absent or inconsistent",
    )
    outcomes = _sequence(
        multidimensional.get("outcomes"),
        name="multidimensional.outcomes",
    )
    weak_runs = _sequence(weak.get("runs"), name="weak_coupling.runs")
    _require(
        len(outcomes) == len(weak_runs) == 13,
        "nine-DOF weak-coupling coverage must contain exactly 13 runs",
    )
    linewidth_ratios: list[float] = []
    shift_ratios: list[float] = []
    seen_run_ids: set[str] = set()
    parent_frequency = 1.5
    limit = 0.1
    for index, (raw_outcome, raw_run) in enumerate(zip(outcomes, weak_runs)):
        outcome = _mapping(raw_outcome, name=f"outcome {index}")
        point = _mapping(outcome.get("point"), name=f"outcome {index}.point")
        run = _mapping(raw_run, name=f"weak_coupling.runs[{index}]")
        _require(
            set(run)
            == {
                "g",
                "limit",
                "linewidth",
                "linewidth_ratio",
                "passed",
                "result_available",
                "run_id",
                "shift",
                "shift_ratio",
            },
            f"weak-coupling run {index} has unexpected or missing fields",
        )
        run_id = run.get("run_id")
        _require(
            isinstance(run_id, str)
            and bool(run_id)
            and run_id not in seen_run_ids
            and run_id == outcome.get("run_id")
            and run.get("g") == point.get("g")
            and run.get("limit") == limit
            and run.get("result_available") is True,
            f"weak-coupling run {index} is not bound to its SCBA outcome",
        )
        seen_run_ids.add(run_id)
        linewidth = _finite_number(
            run.get("linewidth"),
            name=f"weak-coupling run {index}.linewidth",
        )
        shift = _finite_number(
            run.get("shift"),
            name=f"weak-coupling run {index}.shift",
        )
        linewidth_ratio = _finite_number(
            run.get("linewidth_ratio"),
            name=f"weak-coupling run {index}.linewidth_ratio",
            nonnegative=True,
        )
        shift_ratio = _finite_number(
            run.get("shift_ratio"),
            name=f"weak-coupling run {index}.shift_ratio",
            nonnegative=True,
        )
        _require_close(
            linewidth_ratio,
            abs(linewidth) / parent_frequency,
            name=f"weak-coupling run {index}.linewidth_ratio",
        )
        _require_close(
            shift_ratio,
            abs(shift) / parent_frequency,
            name=f"weak-coupling run {index}.shift_ratio",
        )
        calculated_pass = linewidth_ratio < limit and shift_ratio < limit
        _require(
            run.get("passed") is calculated_pass and calculated_pass,
            f"weak-coupling run {index} failed or has inconsistent status",
        )
        linewidth_ratios.append(linewidth_ratio)
        shift_ratios.append(shift_ratio)
    return MappingProxyType(
        {
            "coverage": len(weak_runs),
            "passed": True,
            "maximum_linewidth_ratio": max(linewidth_ratios),
            "maximum_shift_ratio": max(shift_ratios),
            "limit": limit,
        }
    )


def _validated_open_weak_coupling_summary(
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate oracle and Born parent-mode weak-coupling ratios."""

    summary = _mapping(document.get("summary"), name="report.summary")
    _require(
        summary.get("open_three_mode_weak_coupling_passed") is True,
        "open-three-mode weak-coupling summary gate is absent or failed",
    )
    section = _mapping(
        document.get("open_three_mode"),
        name="report.open_three_mode",
    )
    runs = _sequence(section.get("runs"), name="open_three_mode.runs")
    _require(
        len(runs) == 13,
        "open-three-mode weak-coupling coverage must contain 13 runs",
    )
    parent_frequency = 1.5
    limit = 0.1
    linewidth_ratios: list[float] = []
    shift_ratios: list[float] = []
    for index, raw_run in enumerate(runs):
        run = _mapping(raw_run, name=f"open run {index}")
        controls = _mapping(
            run.get("controls"),
            name=f"open run {index}.controls",
        )
        frequencies = _sequence(
            controls.get("mode_frequencies"),
            name=f"open run {index}.mode_frequencies",
        )
        _require(
            list(frequencies) == [1.5, 0.9, 0.6],
            f"open run {index} has unexpected mode frequencies",
        )
        gates = _mapping(run.get("gates"), name=f"open run {index}.gates")
        weak = _mapping(
            gates.get("weak_coupling"),
            name=f"open run {index}.weak_coupling",
        )
        _require(
            set(weak)
            == {"limit", "oracle_passed", "passed", "production_passed"}
            and weak.get("limit") == limit
            and weak.get("oracle_passed") is True
            and weak.get("production_passed") is True
            and weak.get("passed") is True,
            f"open run {index} weak-coupling gate is inconsistent",
        )
        for source_name in ("oracle", "production_born"):
            source = _mapping(
                run.get(source_name),
                name=f"open run {index}.{source_name}",
            )
            linewidths = _sequence(
                source.get("linewidths"),
                name=f"open run {index}.{source_name}.linewidths",
            )
            shifts = _sequence(
                source.get("frequency_shifts"),
                name=f"open run {index}.{source_name}.frequency_shifts",
            )
            _require(
                len(linewidths) == len(shifts) == 3,
                f"open run {index}.{source_name} pole coverage is incomplete",
            )
            parent_linewidth = _finite_number(
                source.get("parent_linewidth"),
                name=f"open run {index}.{source_name}.parent_linewidth",
            )
            parent_shift = _finite_number(
                source.get("parent_frequency_shift"),
                name=f"open run {index}.{source_name}.parent_frequency_shift",
            )
            _require_close(
                parent_linewidth,
                _finite_number(
                    linewidths[0],
                    name=f"open run {index}.{source_name}.linewidths[0]",
                ),
                name=f"open run {index}.{source_name}.parent_linewidth",
            )
            _require_close(
                parent_shift,
                _finite_number(
                    shifts[0],
                    name=f"open run {index}.{source_name}.frequency_shifts[0]",
                ),
                name=f"open run {index}.{source_name}.parent_frequency_shift",
            )
            linewidth_ratio = abs(parent_linewidth) / parent_frequency
            shift_ratio = abs(parent_shift) / parent_frequency
            _require(
                linewidth_ratio < limit and shift_ratio < limit,
                f"open run {index}.{source_name} is not weak coupling",
            )
            linewidth_ratios.append(linewidth_ratio)
            shift_ratios.append(shift_ratio)
    return MappingProxyType(
        {
            "coverage": len(runs),
            "passed": True,
            "maximum_linewidth_ratio": max(linewidth_ratios),
            "maximum_shift_ratio": max(shift_ratios),
            "limit": limit,
        }
    )


def _validate_report(
    raw_report: bytes,
    *,
    commit: str,
    facts: Mapping[str, Any],
    identity_bundle: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str, str]:
    document = _strict_json_bytes(raw_report, name="production_toy_gate.json")
    _require(
        document.get("schema") == _REPORT_SCHEMA
        and document.get("schema_version") == 4,
        "production report is not schema v4",
    )
    report_body_sha = document.get("report_sha256")
    _require(
        _is_sha256(report_body_sha)
        and report_body_sha
        == _body_digest(document, ensure_ascii=False),
        "production report canonical body SHA256 is invalid",
    )
    assert isinstance(report_body_sha, str)
    _require(
        facts.get("report_body_sha256") == report_body_sha
        and facts.get("schema_version") == 4,
        "report facts body digest/schema differs from the raw report",
    )

    summary = _mapping(document.get("summary"), name="report.summary")
    for key in _REQUIRED_SUMMARY_GATES:
        _require(summary.get(key) is True, f"release summary gate failed: {key}")
    _require(
        summary.get("validation_tier") == "release",
        "production report is not release-tier evidence",
    )
    failures = _sequence(summary.get("failures"), name="report.summary.failures")
    _require(not failures, "release report still records summary failures")

    multidimensional = _mapping(
        document.get("multidimensional_scba"),
        name="report.multidimensional_scba",
    )
    _require(
        multidimensional.get("schema") == _MULTIDIMENSIONAL_SCHEMA
        and multidimensional.get("schema_version") == 4
        and multidimensional.get("protocol") == _PROTOCOL
        and multidimensional.get("protocol_version") == 2,
        "nested multidimensional report is not protocol v2/schema v4",
    )
    nested_body_sha = multidimensional.get("report_sha256")
    _require(
        _is_sha256(nested_body_sha)
        and nested_body_sha
        == _body_digest(multidimensional, ensure_ascii=True),
        "nested multidimensional canonical body SHA256 is invalid",
    )
    assert isinstance(nested_body_sha, str)
    _require(
        multidimensional.get("release_certified") is True
        and multidimensional.get("execution_passed") is True
        and multidimensional.get("coarse") is False,
        "nested multidimensional report is not a successful release calculation",
    )
    multidimensional_failures = _sequence(
        multidimensional.get("failures"),
        name="multidimensional.failures",
    )
    certification_failures = _sequence(
        multidimensional.get("certification_failures"),
        name="multidimensional.certification_failures",
    )
    _require(
        not multidimensional_failures and not certification_failures,
        "nested multidimensional report still records execution or certification failures",
    )
    _require(
        facts.get("protocol") == _PROTOCOL
        and facts.get("protocol_version") == 2
        and facts.get("release_certified") is True
        and facts.get("multidimensional_backend") == "fft"
        and facts.get("harmonic_preflight_point_count") == 7
        and facts.get("bounded_backends") == ["serial", "threaded", "fft"],
        "report facts do not identify a certified protocol-v2 report",
    )

    config = _mapping(
        multidimensional.get("config"),
        name="multidimensional.config",
    )
    solver_config = _mapping(
        config.get("solver"),
        name="multidimensional.config.solver",
    )
    comparison_config = _mapping(
        config.get("comparison"),
        name="multidimensional.config.comparison",
    )
    _require(
        config.get("protocol") == _PROTOCOL
        and config.get("protocol_version") == 2
        and config.get("profile") == "full_preregistered"
        and config.get("coarse") is False
        and config.get("certification_eligible") is True
        and config.get("full_physical_ladder_selected") is True
        and config.get("run_preregistered_refinements") is True
        and config.get("run_bounded_backend_validation") is True,
        "multidimensional configuration is not the locked release profile",
    )
    _require(
        solver_config.get("max_iterations") == 200
        and solver_config.get("min_iterations") == 3
        and solver_config.get("consecutive_passes") == 3
        and solver_config.get("sigma_rtol") == 1.0e-8
        and solver_config.get("sigma_atol") == 1.0e-12
        and solver_config.get("green_rtol") == 1.0e-8
        and solver_config.get("green_atol") == 1.0e-12
        and solver_config.get("identity_rtol") == 1.0e-8
        and solver_config.get("current_rtol") == 1.0e-5
        and solver_config.get("current_atol") == 1.0e-10
        and solver_config.get("tail_rtol") == 1.0e-6
        and solver_config.get("mixing_floor") == 0.01
        and solver_config.get("retarded_strategy") == "causal_hilbert"
        and solver_config.get("production_backend") == "fft"
        and solver_config.get("production_hilbert_backend") == "fft"
        and solver_config.get("surface_gf_method") == "analytic_diagonal"
        and solver_config.get("eta_device") == 0.0,
        "multidimensional solver controls differ from the locked release contract",
    )
    refinement_field_policy = _mapping(
        comparison_config.get("refinement_field_policy"),
        name="multidimensional.config.comparison.refinement_field_policy",
    )
    line_shape_observable = _mapping(
        comparison_config.get("line_shape_observable"),
        name="multidimensional.config.comparison.line_shape_observable",
    )
    _require(
        comparison_config.get("field_rtol") == 1.0e-8
        and comparison_config.get("field_atol") == 1.0e-12
        and comparison_config.get("fixed_point_field_rtol") == 1.0e-8
        and comparison_config.get("fixed_point_field_atol") == 1.0e-12
        and comparison_config.get("fixed_point_current_scope")
        == "symmetrized_transport"
        and comparison_config.get("fixed_point_current_rtol") == 1.0e-8
        and comparison_config.get("fixed_point_current_atol") == 1.0e-12
        and comparison_config.get("backend_field_rtol") == 1.0e-10
        and comparison_config.get("backend_field_atol") == 1.0e-12
        and comparison_config.get("backend_current_scope")
        == "full_conservation_vector"
        and comparison_config.get("backend_current_rtol") == 1.0e-10
        and comparison_config.get("backend_current_atol") == 1.0e-12
        and comparison_config.get("refinement_current_scope")
        == "symmetrized_transport"
        and comparison_config.get("refinement_current_rtol") == 5.0e-3
        and comparison_config.get("refinement_current_atol") == 1.0e-12
        and refinement_field_policy.get("compute_when_grid_identical") is True
        and refinement_field_policy.get(
            "gate_only_when_spacing_cutoff_eta_unchanged"
        ) is True
        and comparison_config.get("current_rtol") == 1.0e-10
        and comparison_config.get("current_atol") == 1.0e-12
        and comparison_config.get("legacy_current_scope")
        == "full_conservation_vector"
        and comparison_config.get("legacy_current_policy") == "backend_only"
        and comparison_config.get("refinement_rtol") == 5.0e-3
        and comparison_config.get("line_shape_rtol") == 5.0e-3
        and comparison_config.get("linewidth_atol") == 1.0e-12
        and line_shape_observable.get("dof") == "center_layer.parent_x"
        and line_shape_observable.get("target_angular_frequency") == 1.5
        and line_shape_observable.get("linewidth")
        == "-Im[Sigma_pp^r_xx(Omega_p)]/Omega_p"
        and line_shape_observable.get("shift")
        == "Re[Sigma_pp^r_xx(Omega_p)]/(2*Omega_p)"
        and line_shape_observable.get("interpolation")
        == "piecewise-linear on the signed-frequency grid"
        and comparison_config.get("shift_atol") == 1.0e-8,
        "comparison scopes, applicability policy, or effective tolerances "
        "differ from the locked contract",
    )

    source_sha = identity_bundle["source_tree_sha256"]
    lock_sha = identity_bundle["explicit_lock_sha256"]
    provenance = _mapping(
        multidimensional.get("provenance"),
        name="multidimensional.provenance",
    )
    _require(
        provenance.get("passed") is True
        and provenance.get("complete") is True
        and provenance.get("clean") is True
        and provenance.get("consistent") is True,
        "multidimensional provenance summary is incomplete or dirty",
    )
    common_identity = _validate_code_identity(
        provenance.get("identity"),
        commit=commit,
        source_sha=source_sha,
        expected=None,
        name="multidimensional provenance identity",
    )

    open_section = _mapping(
        document.get("open_three_mode"),
        name="report.open_three_mode",
    )
    open_runs = _sequence(open_section.get("runs"), name="open_three_mode.runs")
    outcomes = _sequence(
        multidimensional.get("outcomes"),
        name="multidimensional.outcomes",
    )
    _require(
        len(open_runs) == 13
        and len(outcomes) == 13
        and facts.get("open_system_run_count") == 13
        and facts.get("multidimensional_outcome_count") == 13,
        "release report lacks the exact 13 open and 13 multidimensional runs",
    )
    for index, raw_run in enumerate(open_runs):
        run = _mapping(raw_run, name=f"open run {index}")
        _require(
            run.get("environment_lock_sha256") == lock_sha
            and run.get("provenance_complete") is True,
            f"open run {index} lacks complete environment provenance",
        )
        run_provenance = _mapping(
            run.get("provenance"),
            name=f"open run {index}.provenance",
        )
        _require(
            run_provenance.get("clean") is True,
            f"open run {index} source is not clean",
        )
        _validate_code_identity(
            run_provenance.get("identity"),
            commit=commit,
            source_sha=source_sha,
            expected=common_identity,
            name=f"open run {index} identity",
        )

    run_ids: set[str] = set()
    for index, raw_outcome in enumerate(outcomes):
        outcome = _mapping(raw_outcome, name=f"outcome {index}")
        run_id = outcome.get("run_id")
        _require(
            isinstance(run_id, str)
            and bool(run_id)
            and run_id not in run_ids,
            f"outcome {index} run_id is absent or duplicated",
        )
        run_ids.add(run_id)
        result = _mapping(outcome.get("result"), name=f"outcome {index}.result")
        _require(
            outcome.get("status") == "converged"
            and outcome.get("backend") == "fft"
            and result.get("method") == "scba"
            and result.get("converged") is True
            and result.get("diagnostics_passed") is True,
            f"outcome {index} is not a converged FFT SCBA result",
        )
        result_provenance = _mapping(
            result.get("provenance"),
            name=f"outcome {index}.result.provenance",
        )
        _require(
            result_provenance.get("complete") is True
            and result_provenance.get("clean") is True,
            f"outcome {index} provenance is incomplete or dirty",
        )
        _validate_code_identity(
            result_provenance.get("identity"),
            commit=commit,
            source_sha=source_sha,
            expected=common_identity,
            name=f"outcome {index} identity",
        )

    topology = _mapping(
        multidimensional.get("topology_validation"),
        name="multidimensional.topology_validation",
    )
    _require(
        topology.get("passed") is True
        and topology.get("full_release_count") is True
        and topology.get("expected_outcome_count") == 13
        and topology.get("observed_outcome_count") == 13
        and topology.get("outcomes_exact") is True,
        "multidimensional outcome topology is not the exact release graph",
    )
    outcome_certification = _mapping(
        multidimensional.get("outcome_certification"),
        name="multidimensional.outcome_certification",
    )
    required_ids = _sequence(
        outcome_certification.get("required_run_ids"),
        name="outcome_certification.required_run_ids",
    )
    diagnostic_ids = _sequence(
        outcome_certification.get("diagnostic_only_run_ids"),
        name="outcome_certification.diagnostic_only_run_ids",
    )
    _require(
        outcome_certification.get("passed") is True
        and outcome_certification.get("coverage") is True
        and len(required_ids) == 11
        and len(diagnostic_ids) == 2
        and set(required_ids).isdisjoint(diagnostic_ids)
        and set(required_ids) | set(diagnostic_ids) == run_ids,
        "required/diagnostic outcome certification coverage is invalid",
    )

    comparisons = _sequence(
        multidimensional.get("comparisons"),
        name="multidimensional.comparisons",
    )
    comparison_names: set[str] = set()
    critical_comparison_count = 0
    for index, raw_comparison in enumerate(comparisons):
        comparison = _validate_locked_comparison(
            raw_comparison,
            index=index,
        )
        name = comparison.get("name")
        _require(
            isinstance(name, str)
            and bool(name)
            and name not in comparison_names,
            f"multidimensional comparison {index} has an invalid/duplicate name",
        )
        comparison_names.add(name)
        _require(
            comparison.get("left_run_id") in run_ids
            and comparison.get("right_run_id") in run_ids,
            f"multidimensional comparison {name} references an unknown outcome",
        )
        if comparison.get("certification_critical") is True:
            critical_comparison_count += 1
            _require(
                comparison.get("passed") is True
                and not _sequence(
                    comparison.get("failures"),
                    name=f"comparison {name}.failures",
                ),
                f"certification-critical multidimensional comparison failed: {name}",
            )
    _require(
        len(comparisons) == 8
        and critical_comparison_count == 6
        and comparison_names
        == _FIXED_POINT_COMPARISONS | _REFINEMENT_COMPARISONS,
        "multidimensional report does not contain the exact locked "
        "eight-comparison matrix",
    )

    conservation = _sequence(
        multidimensional.get("conservation_checks"),
        name="multidimensional.conservation_checks",
    )
    conservation_names: set[str] = set()
    equilibrium_flags: set[bool] = set()
    for index, raw_check in enumerate(conservation):
        check = _mapping(raw_check, name=f"conservation check {index}")
        name = check.get("name")
        equilibrium = check.get("equilibrium")
        _require(
            isinstance(name, str)
            and bool(name)
            and name not in conservation_names
            and check.get("run_id") in run_ids
            and isinstance(equilibrium, bool)
            and check.get("passed") is True
            and not _sequence(
                check.get("failures"),
                name=f"conservation check {name}.failures",
            ),
            f"multidimensional conservation evidence is invalid: {name!r}",
        )
        conservation_names.add(name)
        equilibrium_flags.add(equilibrium)
    _require(
        len(conservation) == 2
        and conservation_names
        == {"interacting_equilibrium", "interacting_nonequilibrium"}
        and equilibrium_flags == {False, True},
        "multidimensional report must contain the exact two conservation gates",
    )

    refinement_archive = _mapping(
        multidimensional.get("numerical_refinement_archive"),
        name="multidimensional.numerical_refinement_archive",
    )
    expected_refinements = _sequence(
        refinement_archive.get("expected_comparison_names"),
        name="numerical_refinement_archive.expected_comparison_names",
    )
    observed_refinements = _sequence(
        refinement_archive.get("observed_comparison_names"),
        name="numerical_refinement_archive.observed_comparison_names",
    )
    _require(
        refinement_archive.get("all_six_archived") is True
        and len(expected_refinements) == 6
        and list(expected_refinements) == list(observed_refinements)
        and set(expected_refinements).issubset(comparison_names),
        "the complete six-point numerical-refinement archive is absent",
    )
    required_refinement = _mapping(
        multidimensional.get("required_refinement_coverage"),
        name="multidimensional.required_refinement_coverage",
    )
    _require(
        set(required_refinement)
        == {"half_spacing", "larger_cutoff", "half_eta", "changed_mixing"},
        "required directional refinement coverage has unexpected axes",
    )
    for direction, raw_gate in required_refinement.items():
        gate = _mapping(raw_gate, name=f"required refinement {direction}")
        _require(
            gate.get("comparison_name") in comparison_names
            and gate.get("executed") is True
            and gate.get("certification_critical") is True
            and gate.get("passed") is True,
            f"required directional refinement gate failed: {direction}",
        )
    _validated_open_weak_coupling_summary(document)
    _validated_multidimensional_weak_coupling_summary(document)
    _validated_cost_accuracy_rows(document)
    _validated_harmonic_preflight_points(
        document,
        expected_environment_sha256=lock_sha,
    )

    harmonic = _mapping(
        multidimensional.get("harmonic_preflight"),
        name="multidimensional.harmonic_preflight",
    )
    harmonic_points = _sequence(
        harmonic.get("points"),
        name="harmonic_preflight.points",
    )
    for index, raw_point in enumerate(harmonic_points):
        point = _mapping(raw_point, name=f"harmonic point {index}")
        point_provenance = _mapping(
            point.get("provenance"),
            name=f"harmonic point {index}.provenance",
        )
        _validate_code_identity(
            point_provenance.get("identity"),
            commit=commit,
            source_sha=source_sha,
            expected=common_identity,
            name=f"harmonic point {index} identity",
        )

    backend = _mapping(
        multidimensional.get("backend_validation"),
        name="multidimensional.backend_validation",
    )
    backend_status = _mapping(
        backend.get("backend_status"),
        name="backend_validation.backend_status",
    )
    _require(
        backend.get("schema")
        == "negfpy.anharmonic.bounded_backend_validation"
        and backend.get("schema_version") == 1
        and backend.get("protocol") == _PROTOCOL
        and backend.get("scope") == "bounded_converged_scba_backend_oracle"
        and backend.get("passed") is True
        and backend.get("full_grid_serial_threaded_comparison_executed") is False
        and backend.get("backends_executed") == ["serial", "threaded", "fft"]
        and set(backend_status) == {"serial", "threaded", "fft"}
        and backend.get("environment_sha256") == lock_sha,
        "bounded serial/threaded/FFT backend evidence is invalid",
    )
    for name, raw_status in backend_status.items():
        status = _mapping(raw_status, name=f"backend status {name}")
        expected_effective, expected_hilbert = {
            "serial": ("serial", "direct"),
            "threaded": ("threaded", "direct"),
            "fft": ("fft", "fft"),
        }[name]
        _require(
            status.get("method") == "scba"
            and status.get("converged") is True
            and status.get("diagnostics_passed") is True
            and status.get("effective_backend") == expected_effective
            and status.get("hilbert_backend") == expected_hilbert
            and status.get("environment_sha256") == lock_sha,
            f"bounded backend status is invalid for {name}",
        )
        status_provenance = _mapping(
            status.get("provenance"),
            name=f"backend status {name}.provenance",
        )
        _validate_code_identity(
            status_provenance.get("identity"),
            commit=commit,
            source_sha=source_sha,
            expected=common_identity,
            name=f"backend status {name} identity",
        )

    environment = _mapping(
        document.get("environment_lock"),
        name="report.environment_lock",
    )
    _require(
        environment.get("passed") is True
        and environment.get("open_environment_sha256") == lock_sha
        and environment.get("multidimensional_environment_sha256") == lock_sha,
        "combined report environment-lock identity is inconsistent",
    )
    checkpoint_evidence = _mapping(
        multidimensional.get("checkpoint_evidence"),
        name="multidimensional.checkpoint_evidence",
    )
    _require(
        checkpoint_evidence.get("passed") is True
        and checkpoint_evidence.get("checkpointing_required_for_release") is True
        and checkpoint_evidence.get("completed_outcome_count") == 13
        and checkpoint_evidence.get("total_outcome_count") == 13
        and checkpoint_evidence.get("complete_completed_outcome_coverage") is True
        and checkpoint_evidence.get("same_environment_sha256") is True
        and checkpoint_evidence.get("environment_sha256") == lock_sha
        and facts.get("checkpoint_count") == 13,
        "multidimensional checkpoint evidence is incomplete",
    )
    return MappingProxyType(dict(document)), report_body_sha, nested_body_sha


def _serialized_history(history: Sequence[Any]) -> list[dict[str, Any]]:
    def strict_float(value: Any) -> float | str:
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0.0 else "-Infinity"
        return number

    records: list[dict[str, Any]] = []
    for item in history:
        records.append(
            {
                "iteration": int(item.iteration),
                "sigma_residual": strict_float(item.sigma_residual),
                "green_residual": strict_float(item.green_residual),
                "identity_residual": strict_float(item.identity_residual),
                "current_abs_error": strict_float(item.current_abs_error),
                "current_rel_error": strict_float(item.current_rel_error),
                "collision_power": strict_float(item.collision_power),
                "mixing": float(item.mixing),
                "passed": bool(item.passed),
                "consecutive_passes": int(item.consecutive_passes),
            }
        )
    return records


def _report_array_sha256(values: Any) -> str:
    # Matches multidimensional_release._array_sha256, which intentionally
    # differs in byte-prefix order from the checkpoint-internal array hash.
    import numpy as np

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_checkpoints(
    root: Path,
    *,
    attempt: int,
    checkpoint_index: Mapping[str, Any],
    document: Mapping[str, Any],
    outer_entries: Sequence[str],
    identity_bundle: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    try:
        from negfpy.anharmonic._checkpoint import (
            SCBACheckpointIdentity,
            load_scba_checkpoint,
        )
    except ImportError as error:
        raise ArtifactImportError(
            "checkpoint validation requires NEGFPy and h5py in this environment"
        ) from error

    _require(
        checkpoint_index.get("schema")
        == "negfpy.remote_validation.checkpoint_index.v1"
        and checkpoint_index.get("attempt") == attempt
        and checkpoint_index.get("checkpoint_root") == _REMOTE_CHECKPOINT_ROOT
        and checkpoint_index.get("count") == 13,
        "checkpoint-index sidecar is invalid",
    )
    index_records = _sequence(
        checkpoint_index.get("records"),
        name="checkpoint_index.records",
    )
    _require(len(index_records) == 13, "checkpoint index must contain 13 records")
    indexed: dict[str, Mapping[str, Any]] = {}
    outer_set = set(outer_entries)
    for index, raw_record in enumerate(index_records):
        record = _mapping(raw_record, name=f"checkpoint index record {index}")
        _require(
            set(record)
            == {
                "run_id",
                "report_declared_path",
                "artifact_relative_path",
                "file_sha256",
            },
            f"checkpoint index record {index} has unexpected/missing fields",
        )
        run_id = record.get("run_id")
        _require(
            isinstance(run_id, str)
            and bool(run_id)
            and run_id not in indexed,
            f"checkpoint index record {index} has a duplicate/invalid run_id",
        )
        relative = record.get("artifact_relative_path")
        _require(
            isinstance(relative, str)
            and relative.startswith(f"{_REMOTE_CHECKPOINT_ROOT}/")
            and relative in outer_set,
            f"checkpoint index record {index} lacks a covered release path",
        )
        indexed[run_id] = record

    multidimensional = _mapping(
        document.get("multidimensional_scba"),
        name="multidimensional report",
    )
    outcomes = _sequence(
        multidimensional.get("outcomes"),
        name="multidimensional outcomes",
    )
    checkpoint_evidence = _mapping(
        multidimensional.get("checkpoint_evidence"),
        name="multidimensional checkpoint evidence",
    )
    raw_evidence_records = _sequence(
        checkpoint_evidence.get("records"),
        name="multidimensional checkpoint evidence records",
    )
    _require(
        len(raw_evidence_records) == 13,
        "checkpoint evidence must contain exactly 13 records",
    )
    evidence_by_run_id: dict[str, Mapping[str, Any]] = {}
    for evidence_number, raw_evidence in enumerate(raw_evidence_records):
        evidence = _mapping(
            raw_evidence,
            name=f"checkpoint evidence record {evidence_number}",
        )
        evidence_run_id = evidence.get("run_id")
        _require(
            isinstance(evidence_run_id, str)
            and bool(evidence_run_id)
            and evidence_run_id not in evidence_by_run_id,
            f"checkpoint evidence record {evidence_number} has an invalid/duplicate run_id",
        )
        evidence_by_run_id[evidence_run_id] = evidence
    source_sha = identity_bundle["source_tree_sha256"]
    environment_sha = identity_bundle["explicit_lock_sha256"]
    summaries: list[Mapping[str, Any]] = []
    seen_files: set[Path] = set()
    for outcome_number, raw_outcome in enumerate(outcomes):
        outcome = _mapping(raw_outcome, name=f"outcome {outcome_number}")
        run_id = outcome.get("run_id")
        _require(run_id in indexed, f"outcome {run_id!r} lacks checkpoint index")
        _require(
            run_id in evidence_by_run_id,
            f"outcome {run_id!r} lacks nested checkpoint evidence",
        )
        index_record = indexed[str(run_id)]
        evidence_record = evidence_by_run_id[str(run_id)]
        checkpoint_record = _mapping(
            outcome.get("checkpoint"),
            name=f"outcome {run_id}.checkpoint",
        )
        _require(
            checkpoint_record.get("enabled") is True
            and checkpoint_record.get("file_sha256")
            == index_record.get("file_sha256")
            and checkpoint_record.get("path")
            == index_record.get("report_declared_path"),
            f"outcome {run_id} checkpoint/index records differ",
        )
        expected_file_sha = checkpoint_record.get("file_sha256")
        _require(_is_sha256(expected_file_sha), f"outcome {run_id} file SHA invalid")
        _require(
            evidence_record.get("completed") is True
            and evidence_record.get("checkpoint_required") is True
            and evidence_record.get("valid") is True
            and evidence_record.get("file_record_valid") is True
            and evidence_record.get("manifest_valid") is True
            and evidence_record.get("checkpoint_state_matches") is True
            and evidence_record.get("checkpoint_history_matches") is True
            and evidence_record.get("identity_fields_valid") is True
            and evidence_record.get("result_identity_matches") is True
            and evidence_record.get("checkpoint_status") == "converged"
            and evidence_record.get("path") == checkpoint_record.get("path")
            and evidence_record.get("file_sha256") == expected_file_sha
            and evidence_record.get("actual_file_sha256") == expected_file_sha
            and evidence_record.get("environment_sha256") == environment_sha
            and evidence_record.get("resumed")
            == checkpoint_record.get("resumed")
            and evidence_record.get("reconstruction_seconds")
            == checkpoint_record.get("reconstruction_seconds"),
            f"nested checkpoint evidence is inconsistent for {run_id}",
        )
        checkpoint_path = _safe_relative_file(
            root,
            index_record.get("artifact_relative_path"),
            name=f"relocated checkpoint {run_id}",
        )
        _require(
            checkpoint_path not in seen_files,
            f"relocated checkpoint is reused by outcome {run_id}",
        )
        seen_files.add(checkpoint_path)
        _require(
            _sha256_file(checkpoint_path) == expected_file_sha,
            f"relocated checkpoint file SHA mismatch for {run_id}",
        )

        raw_identity = _mapping(
            checkpoint_record.get("identity"),
            name=f"outcome {run_id}.checkpoint.identity",
        )
        _require(
            raw_identity.get("schema_name") == "negfpy_dense_scba_checkpoint"
            and raw_identity.get("schema_version") == 1,
            f"outcome {run_id} checkpoint identity schema is invalid",
        )
        identity_kwargs = {
            key: raw_identity.get(key) for key in _CHECKPOINT_IDENTITY_KEYS
        }
        for key, value in identity_kwargs.items():
            if key == "run_id":
                _require(value == run_id, f"checkpoint run_id mismatch for {run_id}")
            else:
                _require(
                    _is_sha256(value),
                    f"checkpoint identity {key} is invalid for {run_id}",
                )
        identity = SCBACheckpointIdentity(**identity_kwargs)
        _require(
            identity.identity_sha256 == raw_identity.get("identity_sha256"),
            f"checkpoint canonical identity SHA mismatch for {run_id}",
        )
        _require(
            evidence_record.get("identity_sha256")
            == identity.identity_sha256,
            f"checkpoint evidence identity SHA differs for {run_id}",
        )
        _require(
            identity.code_sha256 == source_sha
            and identity.environment_sha256 == environment_sha,
            f"checkpoint source/environment identity differs for {run_id}",
        )

        result = _mapping(outcome.get("result"), name=f"outcome {run_id}.result")
        input_hashes = _mapping(
            result.get("input_hashes"),
            name=f"outcome {run_id}.result.input_hashes",
        )
        _require(
            identity.configuration_sha256 == input_hashes.get("configuration")
            and identity.problem_sha256 == input_hashes.get("problem")
            and identity.initial_state_sha256 == input_hashes.get("initial_state")
            and identity.code_sha256 == result.get("source_tree_sha256"),
            f"checkpoint/result input identity differs for {run_id}",
        )

        try:
            loaded = load_scba_checkpoint(
                checkpoint_path,
                expected_identity=identity,
            )
        except Exception as error:
            raise ArtifactImportError(
                f"relocated checkpoint failed HDF5 validation for {run_id}: "
                f"{type(error).__name__}: {error}"
            ) from error
        _require(
            loaded.status == "converged"
            and outcome.get("status") == "converged",
            f"checkpoint/outcome status differs for {run_id}",
        )
        result_history = _sequence(
            result.get("history"),
            name=f"outcome {run_id}.result.history",
        )
        _require(
            _serialized_history(loaded.history) == list(result_history)
            and len(loaded.history) == result.get("iterations"),
            f"checkpoint/result iteration history differs for {run_id}",
        )
        field_hashes = _mapping(
            result.get("field_sha256"),
            name=f"outcome {run_id}.result.field_sha256",
        )
        _require(
            _report_array_sha256(loaded.sigma_lesser)
            == field_hashes.get("sigma_pp_lesser")
            and _report_array_sha256(loaded.sigma_greater)
            == field_hashes.get("sigma_pp_greater"),
            f"checkpoint/result mixed self-energy hashes differ for {run_id}",
        )
        numerical = _mapping(
            _mapping(outcome.get("point"), name=f"outcome {run_id}.point").get(
                "numerical"
            ),
            name=f"outcome {run_id}.point.numerical",
        )
        _require(
            loaded.shape
            == (int(numerical.get("n_frequency", -1)), 9, 9),
            f"checkpoint array shape differs from outcome grid for {run_id}",
        )
        metadata = dict(loaded.metadata)
        _require(
            metadata.get("backend") == outcome.get("backend")
            and metadata.get("initialization") == outcome.get("initialization")
            and metadata.get("continuation_parent_id")
            == outcome.get("continuation_parent_id")
            and metadata.get("method") == "scba"
            and metadata.get("converged") is True
            and metadata.get("release_certified")
            == result.get("result_release_certified"),
            f"checkpoint metadata differs from outcome for {run_id}",
        )
        timings = dict(loaded.timings)
        _require(
            timings.get("elapsed_seconds") == outcome.get("elapsed_seconds")
            and timings.get("seed_elapsed_seconds")
            == outcome.get("seed_elapsed_seconds"),
            f"checkpoint timings differ from outcome for {run_id}",
        )
        summaries.append(
            MappingProxyType(
                {
                    "run_id": run_id,
                    "artifact_relative_path": index_record[
                        "artifact_relative_path"
                    ],
                    "file_sha256": expected_file_sha,
                    "identity_sha256": identity.identity_sha256,
                    "status": loaded.status,
                    "shape": list(loaded.shape),
                    "history_count": len(loaded.history),
                    "sigma_pp_lesser_sha256": field_hashes["sigma_pp_lesser"],
                    "sigma_pp_greater_sha256": field_hashes["sigma_pp_greater"],
                }
            )
        )
        del loaded
    _require(
        set(indexed)
        == {str(_mapping(item, name="outcome").get("run_id")) for item in outcomes},
        "checkpoint index includes unexpected outcome IDs",
    )
    _require(
        set(evidence_by_run_id) == set(indexed),
        "checkpoint evidence includes unexpected outcome IDs",
    )
    return tuple(summaries)


def _json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return str(value)


def _csv_bytes(
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for raw_row in rows:
        writer.writerow(
            {name: _json_cell(raw_row.get(name)) for name in fieldnames}
        )
    return stream.getvalue().encode("utf-8")


def _gate_ratio(gate: Any) -> float | None:
    if not isinstance(gate, Mapping):
        return None
    difference = gate.get("absolute_difference")
    allowed = gate.get("allowed_difference")
    if isinstance(difference, (int, float)) and isinstance(allowed, (int, float)):
        return float(difference) / float(allowed) if float(allowed) > 0.0 else None
    maximum = gate.get("max_absolute_error")
    allowed = gate.get("allowed_absolute_error")
    if isinstance(maximum, (int, float)) and isinstance(allowed, (int, float)):
        return float(maximum) / float(allowed) if float(allowed) > 0.0 else None
    return None


def _open_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    open_section = _mapping(document["open_three_mode"], name="open section")
    for index, raw_run in enumerate(open_section["runs"], start=1):
        run = _mapping(raw_run, name="open run")
        controls = _mapping(run["controls"], name="open controls")
        oracle = _mapping(run["oracle"], name="open oracle")
        born = _mapping(run["production_born"], name="open production")
        gates = _mapping(run["gates"], name="open gates")
        agreement = _mapping(
            gates["oracle_production_agreement"],
            name="open agreement",
        )
        timings = _mapping(run["timings"], name="open timings")
        rows.append(
            {
                "alias": f"O{index:02d}",
                "name": run["name"],
                "temperature": controls["temperature"],
                "g": controls["g"],
                "grid_step": controls["grid_step"],
                "grid_n_points": controls["grid_n_points"],
                "cutoff": controls["cutoff"],
                "eta_lead": controls["eta_lead"],
                "oracle_parent_linewidth": oracle["parent_linewidth"],
                "production_parent_linewidth": born["parent_linewidth"],
                "linewidth_gate_ratio": _gate_ratio(agreement["linewidth"]),
                "oracle_parent_shift": oracle["parent_frequency_shift"],
                "production_parent_shift": born["parent_frequency_shift"],
                "shift_gate_ratio": _gate_ratio(agreement["shift"]),
                "agreement_passed": agreement["passed"],
                "weak_coupling_passed": gates["weak_coupling"]["passed"],
                "passivity_passed": gates["passivity"]["passed"],
                "oracle_wall_seconds": timings["oracle_wall_seconds"],
                "born_wall_seconds": timings["production_born_wall_seconds"],
            }
        )
    return rows


def _open_check_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    section = _mapping(document["open_three_mode"], name="open section")
    rows: list[dict[str, Any]] = []
    for kind, key in (
        ("sign", "sign_checks"),
        ("scaling", "scaling_checks"),
    ):
        for index, raw_check in enumerate(section[key], start=1):
            check = _mapping(raw_check, name=f"open {kind} check")
            for source in ("oracle", "production_born"):
                source_gate = _mapping(check[source], name=f"{kind}.{source}")
                for observable in ("linewidth", "shift"):
                    gate = _mapping(
                        source_gate[observable],
                        name=f"{kind}.{source}.{observable}",
                    )
                    rows.append(
                        {
                            "kind": kind,
                            "check": f"{kind}_{index:02d}",
                            "source": source,
                            "observable": observable,
                            "temperature": check.get("temperature"),
                            "g_or_control": check.get(
                                "g_magnitude",
                                check.get("sign"),
                            ),
                            "base_run": check.get(
                                "negative_run",
                                check.get("lower_run"),
                            ),
                            "candidate_run": check.get(
                                "positive_run",
                                check.get("upper_run"),
                            ),
                            "gate_ratio": _gate_ratio(gate),
                            "passed": gate["passed"],
                            "certification_critical": True,
                        }
                    )
    refinements = _mapping(section["refinements"], name="open refinements")
    for control in ("spacing", "cutoff", "eta_lead"):
        check = _mapping(refinements[control], name=f"open refinement {control}")
        for source in ("oracle", "production_born"):
            source_gate = _mapping(check[source], name=f"{control}.{source}")
            for observable in ("linewidth", "shift"):
                gate = _mapping(
                    source_gate[observable],
                    name=f"{control}.{source}.{observable}",
                )
                rows.append(
                    {
                        "kind": "refinement",
                        "check": control,
                        "source": source,
                        "observable": observable,
                        "temperature": 0.7,
                        "g_or_control": control,
                        "base_run": check["base"],
                        "candidate_run": check["candidate"],
                        "gate_ratio": _gate_ratio(gate),
                        "passed": gate["passed"],
                        "certification_critical": True,
                    }
                )
    for index, raw_check in enumerate(
        section["spontaneous_decay_checks"],
        start=1,
    ):
        check = _mapping(raw_check, name="spontaneous check")
        for source in ("oracle", "production"):
            value = check[f"{source}_parent_linewidth"]
            minimum = check["minimum_linewidth"]
            rows.append(
                {
                    "kind": "spontaneous_decay",
                    "check": f"spontaneous_{index:02d}",
                    "source": source,
                    "observable": "linewidth",
                    "temperature": 0.0,
                    "g_or_control": check["g"],
                    "base_run": check["run"],
                    "candidate_run": "",
                    "gate_ratio": minimum / value if value > 0.0 else None,
                    "passed": value > minimum,
                    "certification_critical": True,
                }
            )
    return rows


def _outcome_context(
    document: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    multidimensional = _mapping(
        document["multidimensional_scba"],
        name="multidimensional",
    )
    certification = _mapping(
        multidimensional["outcome_certification"],
        name="outcome certification",
    )
    required = set(certification["required_run_ids"])
    diagnostic = set(certification["diagnostic_only_run_ids"])
    rows: list[dict[str, Any]] = []
    aliases: dict[str, str] = {}
    for index, raw_outcome in enumerate(multidimensional["outcomes"], start=1):
        outcome = _mapping(raw_outcome, name="outcome")
        run_id = str(outcome["run_id"])
        alias = f"M{index:02d}"
        aliases[run_id] = alias
        point = _mapping(outcome["point"], name="outcome point")
        numerical = _mapping(point["numerical"], name="outcome numerical")
        result = _mapping(outcome["result"], name="outcome result")
        history = _sequence(result["history"], name="outcome history")
        final = _mapping(history[-1], name="final history")
        currents = _mapping(result["currents"], name="outcome currents")
        spatial = _mapping(currents["spatial"], name="outcome spatial currents")
        rows.append(
            {
                "alias": alias,
                "run_id": run_id,
                "certification_role": (
                    "required"
                    if run_id in required
                    else "diagnostic_only"
                    if run_id in diagnostic
                    else "unexpected"
                ),
                "g": point["g"],
                "bias": point["bias"],
                "initialization": outcome["initialization"],
                "continuation_parent_id": outcome["continuation_parent_id"],
                "backend": outcome["backend"],
                "spacing": numerical["spacing"],
                "cutoff": numerical["cutoff"],
                "eta_lead": numerical["eta_lead"],
                "mixing": numerical["mixing"],
                "n_frequency": numerical["n_frequency"],
                "status": outcome["status"],
                "iterations": result["iterations"],
                "final_sigma_residual": final["sigma_residual"],
                "final_green_residual": final["green_residual"],
                "final_identity_residual": final["identity_residual"],
                "left_current": currents["left"],
                "right_current": currents["right"],
                "contact_balance": currents["contact_balance"],
                "collision_power": currents["collision_power"],
                "full_balance": currents["full_balance"],
                "corrected_cut_currents": spatial["corrected_cut_currents"],
                "continuity_residual": spatial["continuity_residual"],
                "sigma_pp_nonlocal_ratio": spatial["sigma_pp_nonlocal_ratio"],
                "sigma_tail_ratio": result["sigma_tail_ratio"],
                "result_release_certified": result[
                    "result_release_certified"
                ],
                "elapsed_seconds": outcome["elapsed_seconds"],
                "seed_elapsed_seconds": outcome["seed_elapsed_seconds"],
                "checkpoint_resumed": outcome["checkpoint"]["resumed"],
                "checkpoint_reconstruction_seconds": outcome["checkpoint"][
                    "reconstruction_seconds"
                ],
                "checkpoint_file_sha256": outcome["checkpoint"][
                    "file_sha256"
                ],
                "checkpoint_identity_sha256": outcome["checkpoint"]["identity"][
                    "identity_sha256"
                ],
            }
        )
    return rows, aliases


def _history_rows(
    document: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcomes = document["multidimensional_scba"]["outcomes"]
    for raw_outcome in outcomes:
        outcome = _mapping(raw_outcome, name="outcome")
        run_id = str(outcome["run_id"])
        for raw_item in outcome["result"]["history"]:
            item = _mapping(raw_item, name="history item")
            rows.append(
                {
                    "alias": aliases[run_id],
                    "run_id": run_id,
                    **{key: item.get(key) for key in (
                        "iteration",
                        "sigma_residual",
                        "green_residual",
                        "identity_residual",
                        "current_abs_error",
                        "current_rel_error",
                        "collision_power",
                        "mixing",
                        "passed",
                        "consecutive_passes",
                    )},
                }
            )
    return rows


def _comparison_rows(
    document: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    multidimensional = document["multidimensional_scba"]
    rows: list[dict[str, Any]] = []
    for raw_comparison in multidimensional["comparisons"]:
        comparison = _mapping(raw_comparison, name="comparison")
        field_ratios = [
            _gate_ratio(item) for item in comparison["field_errors"]
        ]
        field_ratios = [value for value in field_ratios if value is not None]
        field_gate_ratio = (
            max(field_ratios, default=None)
            if comparison["field_gate_applicable"]
            else None
        )
        current_allowed = comparison["current_allowed_absolute_error"]
        line_shape = comparison["line_shape"]
        linewidth_ratio = None
        shift_ratio = None
        if isinstance(line_shape, Mapping):
            linewidth_ratio = _gate_ratio(line_shape["linewidth_error"])
            shift_ratio = _gate_ratio(line_shape["shift_error"])
        rows.append(
            {
                "name": comparison["name"],
                "kind": comparison["kind"],
                "left_alias": aliases[comparison["left_run_id"]],
                "right_alias": aliases[comparison["right_run_id"]],
                "left_run_id": comparison["left_run_id"],
                "right_run_id": comparison["right_run_id"],
                "current_scope": comparison["current_scope"],
                "field_comparison_applicable": comparison[
                    "field_comparison_applicable"
                ],
                "field_gate_applicable": comparison[
                    "field_gate_applicable"
                ],
                "current_gate_applicable": comparison[
                    "current_gate_applicable"
                ],
                "field_rtol": comparison["field_rtol"],
                "field_atol": comparison["field_atol"],
                "current_rtol": comparison["current_rtol"],
                "current_atol": comparison["current_atol"],
                "max_field_gate_ratio": field_gate_ratio,
                "current_max_absolute_error": comparison[
                    "current_max_absolute_error"
                ],
                "current_allowed_absolute_error": current_allowed,
                "current_gate_ratio": (
                    comparison["current_max_absolute_error"] / current_allowed
                    if current_allowed
                    and comparison["current_max_absolute_error"] is not None
                    else None
                ),
                "current_max_relative_error": comparison[
                    "current_max_relative_error"
                ],
                "sigma_tail_left": comparison["sigma_tail_left"],
                "sigma_tail_right": comparison["sigma_tail_right"],
                "linewidth_gate_ratio": linewidth_ratio,
                "shift_gate_ratio": shift_ratio,
                "certification_critical": comparison[
                    "certification_critical"
                ],
                "passed": comparison["passed"],
                "failures": comparison["failures"],
            }
        )
    return rows


def _conservation_rows(
    document: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_check in document["multidimensional_scba"]["conservation_checks"]:
        check = _mapping(raw_check, name="conservation check")
        allowed = check["allowed_error"]
        rows.append(
            {
                "name": check["name"],
                "alias": aliases[check["run_id"]],
                "run_id": check["run_id"],
                "equilibrium": check["equilibrium"],
                "left_current": check["left_current"],
                "right_current": check["right_current"],
                "allowed_error": allowed,
                "contact_balance": check["contact_balance"],
                "contact_gate_ratio": abs(check["contact_balance"]) / allowed,
                "collision_power": check["collision_power"],
                "collision_gate_ratio": abs(check["collision_power"]) / allowed,
                "full_balance": check["full_balance"],
                "corrected_cut_current_error": check[
                    "corrected_cut_current_error"
                ],
                "continuity_residual_error": check[
                    "continuity_residual_error"
                ],
                "sigma_pp_nonlocal_ratio": check["sigma_pp_nonlocal_ratio"],
                "spatial_current_gate": check["spatial_current_gate"],
                "passed": check["passed"],
            }
        )
    return rows


def _passivity_rows(
    document: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_outcome in document["multidimensional_scba"]["outcomes"]:
        outcome = _mapping(raw_outcome, name="outcome")
        run_id = str(outcome["run_id"])
        metrics = _mapping(
            outcome["result"]["diagnostic_metrics"],
            name="diagnostic metrics",
        )
        _require(
            "interacting_retarded_passivity_applicable" in metrics,
            f"outcome {run_id} lacks scoped retarded-passivity metadata",
        )
        for spectrum in (
            "green",
            "sigma_pp",
            "sigma_total",
            "sigma_left",
            "sigma_right",
        ):
            required_keys = (
                f"{spectrum}_passivity_applicable",
                f"{spectrum}_passivity_violation",
                f"{spectrum}_lesser_correlation_positivity_applicable",
                f"{spectrum}_lesser_correlation_positivity_violation",
                f"{spectrum}_greater_correlation_positivity_applicable",
                f"{spectrum}_greater_correlation_positivity_violation",
            )
            missing = [key for key in required_keys if key not in metrics]
            _require(
                not missing,
                f"outcome {run_id} lacks passivity fields: {missing}",
            )
            rows.append(
                {
                    "alias": aliases[run_id],
                    "run_id": run_id,
                    "spectrum": spectrum,
                    "interacting_retarded_passivity_applicable": metrics[
                        "interacting_retarded_passivity_applicable"
                    ],
                    "retarded_passivity_applicable": metrics[
                        f"{spectrum}_passivity_applicable"
                    ],
                    "retarded_passivity_violation": metrics[
                        f"{spectrum}_passivity_violation"
                    ],
                    "lesser_correlation_positivity_applicable": metrics[
                        f"{spectrum}_lesser_correlation_positivity_applicable"
                    ],
                    "lesser_correlation_positivity_violation": metrics[
                        f"{spectrum}_lesser_correlation_positivity_violation"
                    ],
                    "greater_correlation_positivity_applicable": metrics[
                        f"{spectrum}_greater_correlation_positivity_applicable"
                    ],
                    "greater_correlation_positivity_violation": metrics[
                        f"{spectrum}_greater_correlation_positivity_violation"
                    ],
                }
            )
    return rows


def _harmonic_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    harmonic = document["multidimensional_scba"]["harmonic_preflight"]
    rows: list[dict[str, Any]] = []
    for point in harmonic["points"]:
        rows.append(
            {
                "record_type": "point",
                "name": point["label"],
                "g": point["g"],
                "bias": point["bias"],
                "spacing": point["grid"]["spacing"],
                "cutoff": point["grid"]["cutoff"],
                "eta_lead": point["grid"]["eta_lead"],
                "mixing": point["grid"]["mixing"],
                "transport_current": point["transport_current"],
                "relative_error": "",
                "spectral_moment_residual": point["spectral_moment_residual"],
                "elapsed_seconds": point["elapsed_seconds"],
                "certification_critical": "",
                "passed": point["passed"],
                "failures": point["failures"],
            }
        )
    for comparison in harmonic["comparisons"]:
        rows.append(
            {
                "record_type": "comparison",
                "name": comparison["name"],
                "g": "",
                "bias": "",
                "spacing": "",
                "cutoff": "",
                "eta_lead": "",
                "mixing": "",
                "transport_current": "",
                "relative_error": comparison["relative_error"],
                "spectral_moment_residual": "",
                "elapsed_seconds": "",
                "certification_critical": comparison[
                    "certification_critical"
                ],
                "passed": comparison["passed"],
                "failures": comparison["failures"],
            }
        )
    return rows


def _backend_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    backend = document["multidimensional_scba"]["backend_validation"]
    rows: list[dict[str, Any]] = []
    for name in ("serial", "threaded", "fft"):
        status = backend["backend_status"][name]
        rows.append(
            {
                "record_type": "backend",
                "name": name,
                "reference_backend": "",
                "candidate_backend": "",
                "effective_backend": status["effective_backend"],
                "hilbert_backend": status["hilbert_backend"],
                "iterations": status["iterations"],
                "elapsed_seconds": status["elapsed_seconds"],
                "max_field_gate_ratio": "",
                "current_gate_ratio": "",
                "passed": status["converged"]
                and status["diagnostics_passed"],
            }
        )
    for comparison in backend["comparisons"]:
        field_ratios = [
            _gate_ratio(item) for item in comparison["field_errors"]
        ]
        field_ratios = [value for value in field_ratios if value is not None]
        rows.append(
            {
                "record_type": "comparison",
                "name": (
                    f"{comparison['reference_backend']}_vs_"
                    f"{comparison['candidate_backend']}"
                ),
                "reference_backend": comparison["reference_backend"],
                "candidate_backend": comparison["candidate_backend"],
                "effective_backend": "",
                "hilbert_backend": "",
                "iterations": "",
                "elapsed_seconds": "",
                "max_field_gate_ratio": max(field_ratios, default=None),
                "current_gate_ratio": _gate_ratio(
                    comparison["current_error"]
                ),
                "passed": comparison["passed"],
            }
        )
    return rows


def _timing_rows(
    document: Mapping[str, Any],
    open_rows: Sequence[Mapping[str, Any]],
    outcome_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in open_rows:
        for category, key in (
            ("open_oracle", "oracle_wall_seconds"),
            ("open_born", "born_wall_seconds"),
        ):
            rows.append(
                {
                    "category": category,
                    "alias": row["alias"],
                    "name": row["name"],
                    "wall_seconds": row[key],
                    "authoritative_total": False,
                }
            )
    for row in outcome_rows:
        rows.append(
            {
                "category": "scba_outcome",
                "alias": row["alias"],
                "name": row["run_id"],
                "wall_seconds": row["elapsed_seconds"],
                "authoritative_total": False,
            }
        )
    rows.append(
        {
            "category": "multidimensional_total",
            "alias": "",
            "name": "multidimensional_scba.total_elapsed_seconds",
            "wall_seconds": document["multidimensional_scba"][
                "total_elapsed_seconds"
            ],
            "authoritative_total": True,
        }
    )
    return rows


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "&": r"\&",
        "#": r"\#",
        "%": r"\%",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _tex_bool(value: Any) -> str:
    return "passed" if value is True else "failed" if value is False else "n/a"


def _applicability_bool(value: Any) -> bool | None:
    """Decode explicit diagnostic applicability without truthiness coercion."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and number in {0.0, 1.0}:
            return bool(number)
    return None


def _tex_applicability(value: Any) -> str:
    applicable = _applicability_bool(value)
    return "yes" if applicable is True else "no" if applicable is False else "n/a"


def _tex_applicable_number(applicability: Any, value: Any) -> str:
    """Render a metric only where its diagnostic is explicitly applicable."""

    return _tex_number(value) if _applicability_bool(applicability) is True else "N/A"


def _tex_number(value: Any) -> str:
    if value is None or value == "":
        return "--"
    if isinstance(value, bool):
        return _tex_bool(value)
    if isinstance(value, (int, float)):
        return format(value, ".8g")
    return _tex_escape(value)


def _tex_rows(lines: Iterable[str]) -> bytes:
    return ("% Generated from the immutable protocol-v2 JSON; do not edit.\n"
            + "\n".join(lines)
            + "\n").encode("utf-8")


def _derive_manuscript_tex_payloads(
    document: Mapping[str, Any],
) -> Mapping[str, bytes]:
    """Derive receipt-bound manuscript evidence from strict report fields."""

    harmonic_points = _validated_harmonic_preflight_points(document)
    cost_rows = _validated_cost_accuracy_rows(document)
    open_weak = _validated_open_weak_coupling_summary(document)
    weak = _validated_multidimensional_weak_coupling_summary(document)

    harmonic_lines = []
    for point in harmonic_points:
        comparison_status = (
            "reference"
            if point["comparison_passed"] is None
            else _tex_bool(point["comparison_passed"])
        )
        harmonic_lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(point["label"]),
                _tex_escape(point["certification_role"]),
                _tex_number(point["spacing"]),
                _tex_number(point["cutoff"]),
                _tex_number(point["eta_lead"]),
                _tex_number(point["mixing"]),
                _tex_number(point["transport_current"]),
                _tex_number(point["spectral_moment_residual"]),
                _tex_bool(point["point_passed"]),
                _tex_escape(comparison_status),
            )
        )

    cost_lines = []
    for row in cost_rows:
        measured = _mapping(
            row["measured"],
            name=f"cost_accuracy.{row['method']}.measured",
        )
        cost_lines.append(
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["method"]),
                _tex_escape(row["accuracy_role"]),
                str(row["self_consistent"]).lower(),
                str(row["conserving"]).lower(),
                _tex_escape(row["asymptotic_dense_time"]),
                _tex_number(measured["count"]),
                _tex_number(measured["total_wall_seconds"]),
                _tex_escape(row["release_use"]),
            )
        )

    payloads = {
        "tex/harmonic_preflight_rows.tex": _tex_rows(harmonic_lines),
        "tex/cost_accuracy_rows.tex": _tex_rows(cost_lines),
        "tex/weak_coupling_summary.tex": _tex_rows(
            (
                rf"\def\ToyGateWeakCouplingCoverage{{{open_weak['coverage']}}}",
                (
                    "\\def\\ToyGateWeakCouplingPassed"
                    f"{{{str(open_weak['passed']).lower()}}}"
                ),
                (
                    "\\def\\ToyGateWeakCouplingMaxLinewidthRatio"
                    f"{{{_tex_number(open_weak['maximum_linewidth_ratio'])}}}"
                ),
                (
                    "\\def\\ToyGateWeakCouplingMaxShiftRatio"
                    f"{{{_tex_number(open_weak['maximum_shift_ratio'])}}}"
                ),
                (
                    "\\def\\ToyGateWeakCouplingLimit"
                    f"{{{_tex_number(open_weak['limit'])}}}"
                ),
            )
        ),
        "tex/multidimensional_weak_coupling_summary.tex": _tex_rows(
            (
                rf"\def\ToyGateMultidimensionalWeakCouplingCoverage{{{weak['coverage']}}}",
                (
                    "\\def\\ToyGateMultidimensionalWeakCouplingPassed"
                    f"{{{str(weak['passed']).lower()}}}"
                ),
                (
                    "\\def\\ToyGateMultidimensionalWeakCouplingMaxLinewidthRatio"
                    f"{{{_tex_number(weak['maximum_linewidth_ratio'])}}}"
                ),
                (
                    "\\def\\ToyGateMultidimensionalWeakCouplingMaxShiftRatio"
                    f"{{{_tex_number(weak['maximum_shift_ratio'])}}}"
                ),
                (
                    "\\def\\ToyGateMultidimensionalWeakCouplingLimit"
                    f"{{{_tex_number(weak['limit'])}}}"
                ),
            )
        ),
    }
    return MappingProxyType(payloads)


def _passivity_tex_lines(
    passivity_rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    passivity_by_alias: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in passivity_rows:
        passivity_by_alias.setdefault(str(row["alias"]), {})[
            str(row["spectrum"])
        ] = row
    lines: list[str] = []
    for alias, values in passivity_by_alias.items():
        _require(
            {"green", "sigma_pp", "sigma_total"}.issubset(values),
            f"passivity TeX row {alias} lacks required spectra",
        )
        green = values["green"]
        sigma_pp = values["sigma_pp"]
        sigma_total = values["sigma_total"]
        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                _tex_escape(alias),
                _tex_applicability(green["retarded_passivity_applicable"]),
                _tex_applicable_number(
                    green["retarded_passivity_applicable"],
                    green["retarded_passivity_violation"],
                ),
                _tex_applicable_number(
                    sigma_pp["retarded_passivity_applicable"],
                    sigma_pp["retarded_passivity_violation"],
                ),
                _tex_applicable_number(
                    sigma_total["retarded_passivity_applicable"],
                    sigma_total["retarded_passivity_violation"],
                ),
            )
        )
    return tuple(lines)


def _derive_csv_payloads(
    document: Mapping[str, Any],
) -> Mapping[str, bytes]:
    """Derive canonical report-only CSV bytes from one validated document."""

    open_rows = _open_rows(document)
    open_checks = _open_check_rows(document)
    outcome_rows, aliases = _outcome_context(document)
    history_rows = _history_rows(document, aliases)
    comparison_rows = _comparison_rows(document, aliases)
    conservation_rows = _conservation_rows(document, aliases)
    passivity_rows = _passivity_rows(document, aliases)
    harmonic_rows = _harmonic_rows(document)
    backend_rows = _backend_rows(document)
    timing_rows = _timing_rows(document, open_rows, outcome_rows)
    tables = (
        ("open_runs", open_rows),
        ("open_checks", open_checks),
        ("scba_outcomes", outcome_rows),
        ("scba_history", history_rows),
        ("scba_comparisons", comparison_rows),
        ("conservation", conservation_rows),
        ("passivity", passivity_rows),
        ("harmonic_preflight", harmonic_rows),
        ("backend_oracle", backend_rows),
        ("timings", timing_rows),
    )
    payloads: dict[str, bytes] = {}
    for table_name, rows in tables:
        _require(
            rows,
            f"derived {table_name} table is unexpectedly empty",
        )
        payloads[f"csv/{table_name}.csv"] = _csv_bytes(
            list(rows[0]),
            rows,
        )
    return MappingProxyType(payloads)


def _derive_files(
    document: Mapping[str, Any],
    *,
    identity_bundle: Mapping[str, Any],
    remote_environment: Mapping[str, Any],
    report_file_sha256: str,
    report_body_sha256: str,
    nested_body_sha256: str,
) -> dict[str, bytes]:
    open_rows = _open_rows(document)
    open_checks = _open_check_rows(document)
    outcome_rows, aliases = _outcome_context(document)
    history_rows = _history_rows(document, aliases)
    comparison_rows = _comparison_rows(document, aliases)
    conservation_rows = _conservation_rows(document, aliases)
    passivity_rows = _passivity_rows(document, aliases)
    harmonic_rows = _harmonic_rows(document)
    backend_rows = _backend_rows(document)
    timing_rows = _timing_rows(document, open_rows, outcome_rows)
    common_identity = document["multidimensional_scba"]["provenance"]["identity"]
    provenance_rows = [
        {
            "commit": identity_bundle["commit"],
            "git_tree": identity_bundle["git_tree"],
            "source_tree_sha256": identity_bundle["source_tree_sha256"],
            "environment_sha256": identity_bundle["explicit_lock_sha256"],
            "report_file_sha256": report_file_sha256,
            "report_body_sha256": report_body_sha256,
            "nested_report_body_sha256": nested_body_sha256,
            "declared_package_version": common_identity[
                "declared_package_version"
            ],
            "installed_package_version": common_identity[
                "installed_package_version"
            ],
            "dtype": common_identity["dtype"],
            "python": common_identity["python"],
            "numpy_version": common_identity["numpy_version"],
            "scipy_version": common_identity["scipy_version"],
            "platform": common_identity["platform"],
            "blas_runtime": common_identity["blas_runtime"],
            "hostname": remote_environment.get("hostname"),
            "cpu_model": remote_environment.get("cpu_model"),
            "cpu_count": remote_environment.get("cpu_count"),
            "memory_total_bytes": remote_environment.get("memory_total_bytes"),
        }
    ]

    files = dict(_derive_csv_payloads(document))
    files.update(_derive_manuscript_tex_payloads(document))
    _require(provenance_rows, "derived provenance table is unexpectedly empty")
    files["csv/provenance.csv"] = _csv_bytes(
        list(provenance_rows[0]),
        provenance_rows,
    )

    summary = document["summary"]
    files["tex/release_summary.tex"] = _tex_rows(
        (
            rf"\def\ToyGateReleaseCertified{{{str(summary['release_certified']).lower()}}}",
            rf"\def\ToyGateOraclePassed{{{str(summary['oracle_gate_passed']).lower()}}}",
            rf"\def\ToyGateMultidimensionalPassed{{{str(summary['multidimensional_release_certified']).lower()}}}",
            (
                rf"\def\ToyGateStaticConfigurationPassed{{"
                f"{str(summary['multidimensional_static_configuration_passed']).lower()}"
                "}"
            ),
            (
                rf"\def\ToyGateRuntimeEvidencePassed{{"
                f"{str(summary['multidimensional_runtime_evidence_passed']).lower()}"
                "}"
            ),
            rf"\def\ToyGateReportFileSHA{{{report_file_sha256}}}",
            rf"\def\ToyGateReportBodySHA{{{report_body_sha256}}}",
            rf"\def\ToyGateCommit{{{identity_bundle['commit']}}}",
        )
    )
    files["tex/open_run_rows.tex"] = _tex_rows(
        (
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["alias"]),
                _tex_number(row["temperature"]),
                _tex_number(row["g"]),
                _tex_number(row["production_parent_linewidth"]),
                _tex_number(row["production_parent_shift"]),
                _tex_number(
                    max(
                        row["linewidth_gate_ratio"],
                        row["shift_gate_ratio"],
                    )
                ),
                _tex_bool(row["agreement_passed"]),
            )
            for row in open_rows
        )
    )
    files["tex/scba_outcome_rows.tex"] = _tex_rows(
        (
            "{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["alias"]),
                _tex_escape(row["certification_role"]),
                _tex_number(row["g"]),
                _tex_number(row["bias"]),
                _tex_escape(row["initialization"]),
                _tex_number(row["iterations"]),
                _tex_number(row["left_current"]),
                _tex_bool(row["result_release_certified"]),
            )
            for row in outcome_rows
        )
    )
    files["tex/comparison_rows.tex"] = _tex_rows(
        (
            "{} & {} & {}--{} & {} & {} & {} \\\\".format(
                _tex_escape(row["name"]),
                _tex_escape(row["kind"]),
                _tex_escape(row["left_alias"]),
                _tex_escape(row["right_alias"]),
                _tex_number(row["max_field_gate_ratio"]),
                _tex_bool(row["certification_critical"]),
                _tex_bool(row["passed"]),
            )
            for row in comparison_rows
        )
    )
    files["tex/conservation_rows.tex"] = _tex_rows(
        (
            "{} & {} & {} & {} & {} & {} \\\\".format(
                _tex_escape(row["name"]),
                _tex_escape(row["alias"]),
                _tex_number(row["left_current"]),
                _tex_number(row["contact_gate_ratio"]),
                _tex_number(row["collision_gate_ratio"]),
                _tex_bool(row["passed"]),
            )
            for row in conservation_rows
        )
    )
    files["tex/passivity_rows.tex"] = _tex_rows(
        _passivity_tex_lines(passivity_rows)
    )
    files["tex/provenance_rows.tex"] = _tex_rows(
        (
            rf"Git commit & \path{{{identity_bundle['commit']}}} \\",
            rf"Git tree & \path{{{identity_bundle['git_tree']}}} \\",
            rf"Source-tree SHA256 & \path{{{identity_bundle['source_tree_sha256']}}} \\",
            rf"Environment SHA256 & \path{{{identity_bundle['explicit_lock_sha256']}}} \\",
            rf"Raw report SHA256 & \path{{{report_file_sha256}}} \\",
            rf"Report-body SHA256 & \path{{{report_body_sha256}}} \\",
        )
    )
    files["tex/run_id_rows.tex"] = _tex_rows(
        (
            rf"{_tex_escape(alias)} & \path{{{run_id}}} \\"
            for run_id, alias in aliases.items()
        )
    )
    return files


def _fetched_artifact_index(
    *,
    root: Path,
    entries: Mapping[str, str],
    sha256sums_sha256: str,
    fetch_receipt_sha256: str,
) -> bytes:
    """Return a deterministic identity index for every fetched artifact byte."""

    records = [
        {
            "relative_path": relative,
            "sha256": digest,
            "bytes": (root / relative).stat().st_size,
        }
        for relative, digest in sorted(entries.items())
    ]
    core = {
        "schema": "negfpy.anharmonic.fetched_toy_gate_artifact_index",
        "schema_version": 1,
        "sha256sums_sha256": sha256sums_sha256,
        "fetch_receipt_sha256": fetch_receipt_sha256,
        "artifact_count": len(records),
        "artifacts": records,
        "preservation_policy": (
            "Every fetched byte remains in the caller-anchored source tree; "
            "this deterministic index preserves its path, size, and SHA256 "
            "without duplicating large HDF5 payloads into report staging."
        ),
    }
    return _pretty_json_bytes(
        {
            **core,
            "index_sha256": _sha256_bytes(
                _canonical_json_bytes(core, ensure_ascii=False)
            ),
        }
    )


def _build_receipt(
    *,
    commit: str,
    attempt: int,
    report_relative_path: str,
    report_file_sha256: str,
    report_body_sha256: str,
    nested_body_sha256: str,
    outer_manifest_sha256: str,
    fetch_receipt: Mapping[str, Any],
    expected_commit: str,
    expected_environment_lock_sha256: str,
    expected_fetch_receipt_sha256: str,
    expected_sha256sums_sha256: str,
    facts: Mapping[str, Any],
    identity_bundle: Mapping[str, Any],
    checkpoint_summaries: Sequence[Mapping[str, Any]],
    generated_files: Mapping[str, bytes],
) -> dict[str, Any]:
    core = {
        "schema": "negfpy.anharmonic.production_toy_gate_import_receipt",
        "schema_version": 2,
        "implementation": {
            "name": Path(__file__).name,
            "version": _IMPORTER_VERSION,
            "source_sha256": _sha256_file(Path(__file__).resolve()),
        },
        "source": {
            "commit": commit,
            "attempt": attempt,
            "report_relative_path": "production_toy_gate.json",
            "remote_report_relative_path": report_relative_path,
            "report_file_sha256": report_file_sha256,
            "report_body_sha256": report_body_sha256,
            "nested_report_body_sha256": nested_body_sha256,
            "outer_sha256sums_file_sha256": outer_manifest_sha256,
            "fetch_receipt_sha256": expected_fetch_receipt_sha256,
            "sha256sums_sha256": expected_sha256sums_sha256,
            "transport_receipt_schema": fetch_receipt.get("schema"),
            "transport_receipt_created_utc": (
                fetch_receipt.get("fetched_utc")
                or fetch_receipt.get("created_utc")
            ),
            "fetched_utc": fetch_receipt.get("fetched_utc"),
            "git_tree": identity_bundle["git_tree"],
            "source_tree_sha256": identity_bundle["source_tree_sha256"],
            "environment_sha256": identity_bundle["explicit_lock_sha256"],
        },
        "trusted_inputs": {
            "expected_commit": expected_commit,
            "expected_environment_lock_sha256": (
                expected_environment_lock_sha256
            ),
            "expected_fetch_receipt_sha256": (
                expected_fetch_receipt_sha256
            ),
            "expected_sha256sums_sha256": expected_sha256sums_sha256,
        },
        "release_contract": {
            "validation_tier": "release",
            "release_certified": True,
            "protocol": _PROTOCOL,
            "protocol_version": 2,
            "schema_version": 4,
            "open_system_run_count": facts["open_system_run_count"],
            "multidimensional_outcome_count": facts[
                "multidimensional_outcome_count"
            ],
            "checkpoint_count": facts["checkpoint_count"],
        },
        "checkpoint_validation": {
            "mode": "identity_and_integrity_load_without_scba_restore",
            "count": len(checkpoint_summaries),
            "records": [dict(record) for record in checkpoint_summaries],
        },
        "staged_artifacts": {
            path: {
                "sha256": _sha256_bytes(payload),
                "bytes": len(payload),
            }
            for path, payload in sorted(generated_files.items())
        },
    }
    receipt_sha = _sha256_bytes(
        _canonical_json_bytes(core, ensure_ascii=False)
    )
    return {**core, "receipt_sha256": receipt_sha}


def _verify_existing_tree(output: Path, expected: Mapping[str, bytes]) -> None:
    _require(
        output.is_dir() and not output.is_symlink(),
        f"existing staging target is not a regular directory: {output}",
    )
    symlinks = [path for path in output.rglob("*") if path.is_symlink()]
    _require(not symlinks, "existing staging directory contains symlinks")
    special_entries = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if not path.is_symlink()
        and not path.is_file()
        and not path.is_dir()
    ]
    _require(
        not special_entries,
        "existing staging directory contains special filesystem entries",
    )
    observed_directories = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    _require(
        observed_directories == _required_parent_directories(expected),
        "existing staging directory contains unexpected directories",
    )
    observed = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    _require(
        observed == set(expected),
        "existing staging directory is not the same deterministic import",
    )
    for relative, payload in expected.items():
        path = output.joinpath(*PurePosixPath(relative).parts)
        _require(
            path.read_bytes() == payload,
            f"existing staged artifact differs: {relative}",
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any winner."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    _require(
        renameat2 is not None,
        "atomic no-overwrite publication requires renameat2",
    )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _publish_tree_atomic(output: Path, files: Mapping[str, bytes]) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    _require(bool(output.name), "staging output must have a non-empty name")
    lock_path = output.parent / f".{output.name}.publication.lock"
    lock_token = f"pid={os.getpid()} nonce={uuid4().hex}\n".encode("ascii")
    lock_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    lock_flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        lock_descriptor = os.open(lock_path, lock_flags, 0o600)
    except FileExistsError as error:
        raise ArtifactImportError(
            f"publication lock already exists: {lock_path}"
        ) from error
    lock_identity = os.fstat(lock_descriptor)
    temporary: Path | None = None
    try:
        written = os.write(lock_descriptor, lock_token)
        _require(written == len(lock_token), "publication lock write was partial")
        os.fsync(lock_descriptor)
        _fsync_directory(output.parent)

        if output.exists():
            _verify_existing_tree(output, files)
            return "unchanged"

        temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
        temporary.mkdir()
        for relative_text, payload in sorted(files.items()):
            relative = PurePosixPath(relative_text)
            _require(
                not relative.is_absolute() and ".." not in relative.parts,
                f"unsafe staging output path {relative_text!r}",
            )
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        directories = sorted(
            (
                path
                for path in temporary.rglob("*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for directory in directories:
            _fsync_directory(directory)
        _fsync_directory(temporary)
        try:
            _rename_directory_noreplace(temporary, output)
        except FileExistsError as error:
            raise ArtifactImportError(
                "publication destination appeared during atomic commit"
            ) from error
        temporary = None
        _fsync_directory(output.parent)
        return "created"
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        os.close(lock_descriptor)
        try:
            observed_lock = os.lstat(lock_path)
        except FileNotFoundError:
            observed_lock = None
        if (
            observed_lock is not None
            and observed_lock.st_dev == lock_identity.st_dev
            and observed_lock.st_ino == lock_identity.st_ino
        ):
            os.unlink(lock_path)
            _fsync_directory(output.parent)


def import_release_evidence(
    artifact_root: Path,
    output: Path,
    *,
    expected_commit: str,
    expected_environment_lock_sha256: str,
    expected_fetch_receipt_sha256: str,
    expected_sha256sums_sha256: str,
) -> Mapping[str, Any]:
    """Validate one fetched release bundle and atomically stage report data."""

    _require(
        _is_commit(expected_commit),
        "trusted expected commit must be an exact lowercase SHA1",
    )
    _require(
        _is_sha256(expected_environment_lock_sha256),
        "trusted expected environment-lock digest must be lowercase SHA256",
    )
    _require(
        _is_sha256(expected_fetch_receipt_sha256),
        "trusted expected fetch-receipt digest must be lowercase SHA256",
    )
    _require(
        _is_sha256(expected_sha256sums_sha256),
        "trusted expected SHA256SUMS digest must be lowercase SHA256",
    )
    supplied_root = artifact_root.expanduser()
    _require(
        supplied_root.is_dir() and not supplied_root.is_symlink(),
        "supplied artifact root is unsafe or is a symlink",
    )
    root = supplied_root.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "artifact root is unsafe")
    supplied_output = output.expanduser()
    _require(
        not supplied_output.is_symlink(),
        "supplied output staging directory is a symlink",
    )
    output = supplied_output.resolve(strict=False)
    _require(
        output != root and not output.is_relative_to(root),
        "output staging directory cannot be inside the fetched artifact root",
    )

    outer_entries, fetch_receipt = _verify_outer_sha256sums(
        root,
        expected_commit=expected_commit,
        expected_environment_lock_sha256=expected_environment_lock_sha256,
        expected_fetch_receipt_sha256=expected_fetch_receipt_sha256,
        expected_sha256sums_sha256=expected_sha256sums_sha256,
    )
    receipt_schema = fetch_receipt["schema"]
    _require(
        receipt_schema == _REMOTE_FETCH_RECEIPT_SCHEMA,
        "importer v3 requires the canonical remote current-attempt publication",
    )
    publication_validator = _validate_publication
    (
        commit,
        attempt,
        report_path,
        raw_report,
        facts,
        _checkpoint_index_path,
        checkpoint_index,
        job_status,
    ) = publication_validator(
        root,
        expected_commit=expected_commit,
    )
    _require(
        fetch_receipt.get("commit") == commit,
        "transport receipt commit differs from the completed attempt",
    )
    if receipt_schema == _REMOTE_FETCH_RECEIPT_SCHEMA:
        identity_bundle, execution_environment = _validate_identity_bundle(
            root,
            commit=commit,
            attempt=attempt,
            expected_environment_lock_sha256=(
                expected_environment_lock_sha256
            ),
            facts=facts,
            job_status=job_status,
        )
    else:
        identity_bundle, execution_environment = (
            _validate_local_identity_bundle(
                root,
                commit=commit,
                expected_environment_lock_sha256=(
                    expected_environment_lock_sha256
                ),
                facts=facts,
            )
        )
    document, report_body_sha, nested_body_sha = _validate_report(
        raw_report,
        commit=commit,
        facts=facts,
        identity_bundle=identity_bundle,
    )
    checkpoint_summaries = _validate_checkpoints(
        root,
        attempt=attempt,
        checkpoint_index=checkpoint_index,
        document=document,
        outer_entries=outer_entries,
        identity_bundle=identity_bundle,
    )

    report_file_sha = _sha256_bytes(raw_report)
    source_publication_path = (
        root / f"attempts/attempt-{attempt:03d}/publication.json"
    )
    generated = {
        "production_toy_gate.json": raw_report,
        "source/SHA256SUMS": (root / "SHA256SUMS").read_bytes(),
        "source/fetch-receipt.json": (root / "fetch-receipt.json").read_bytes(),
        "source/current-attempt.json": (root / "current-attempt.json").read_bytes(),
        "source/attempt-publication.json": source_publication_path.read_bytes(),
        "source/fetched-artifact-index.json": _fetched_artifact_index(
            root=root,
            entries=outer_entries,
            sha256sums_sha256=expected_sha256sums_sha256,
            fetch_receipt_sha256=expected_fetch_receipt_sha256,
        ),
        **_derive_files(
            document,
            identity_bundle=identity_bundle,
            remote_environment=execution_environment,
            report_file_sha256=report_file_sha,
            report_body_sha256=report_body_sha,
            nested_body_sha256=nested_body_sha,
        ),
    }
    receipt = _build_receipt(
        commit=commit,
        attempt=attempt,
        report_relative_path=report_path.relative_to(root).as_posix(),
        report_file_sha256=report_file_sha,
        report_body_sha256=report_body_sha,
        nested_body_sha256=nested_body_sha,
        outer_manifest_sha256=expected_sha256sums_sha256,
        fetch_receipt=fetch_receipt,
        expected_commit=expected_commit,
        expected_environment_lock_sha256=(
            expected_environment_lock_sha256
        ),
        expected_fetch_receipt_sha256=(
            expected_fetch_receipt_sha256
        ),
        expected_sha256sums_sha256=expected_sha256sums_sha256,
        facts=facts,
        identity_bundle=identity_bundle,
        checkpoint_summaries=checkpoint_summaries,
        generated_files=generated,
    )
    generated["import_receipt.json"] = _pretty_json_bytes(receipt)
    checksum_lines = [
        f"{_sha256_bytes(payload)}  {relative}"
        for relative, payload in sorted(generated.items())
    ]
    generated["SHA256SUMS"] = ("\n".join(checksum_lines) + "\n").encode("ascii")
    publication = _publish_tree_atomic(output, generated)
    return MappingProxyType(
        {
            "status": publication,
            "output": str(output),
            "commit": commit,
            "attempt": attempt,
            "report_file_sha256": report_file_sha,
            "report_body_sha256": report_body_sha,
            "environment_lock_sha256": (
                expected_environment_lock_sha256
            ),
            "fetch_receipt_sha256": expected_fetch_receipt_sha256,
            "sha256sums_sha256": expected_sha256sums_sha256,
            "checkpoint_count": len(checkpoint_summaries),
            "release_certified": True,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and atomically stage a fetched protocol-v2 production "
            "toy-gate artifact bundle without running SCBA."
        )
    )
    parser.add_argument(
        "artifact_root",
        type=Path,
        help="checksum-verified fetched artifact root for one exact commit",
    )
    parser.add_argument(
        "output_staging_dir",
        type=Path,
        help="new or byte-identical output staging directory",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="trusted exact lowercase 40-hex candidate commit",
    )
    parser.add_argument(
        "--expected-environment-lock-sha256",
        required=True,
        help="trusted exact lowercase SHA256 of environment-explicit.lock",
    )
    parser.add_argument(
        "--expected-fetch-receipt-sha256",
        required=True,
        help="trusted exact lowercase SHA256 of raw fetch-receipt.json",
    )
    parser.add_argument(
        "--expected-sha256sums-sha256",
        required=True,
        help="trusted exact lowercase SHA256 of raw fetched SHA256SUMS",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = import_release_evidence(
            arguments.artifact_root,
            arguments.output_staging_dir,
            expected_commit=arguments.expected_commit,
            expected_environment_lock_sha256=(
                arguments.expected_environment_lock_sha256
            ),
            expected_fetch_receipt_sha256=(
                arguments.expected_fetch_receipt_sha256
            ),
            expected_sha256sums_sha256=(
                arguments.expected_sha256sums_sha256
            ),
        )
    except (ArtifactImportError, OSError, ValueError, TypeError) as error:
        print(
            f"production toy-gate import refused: {error}",
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            dict(result),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
