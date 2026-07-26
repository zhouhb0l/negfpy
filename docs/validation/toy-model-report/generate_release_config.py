#!/usr/bin/env python3
"""Generate a fail-closed LaTeX config from externally anchored evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence


RANK_LADDER = (1, 2, 4, 8, 16)
THREAD_LIMITS = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DATE_RE = re.compile(r"^[0-9]{1,2} [A-Za-z]+ [0-9]{4}$")
SAFE_TEX_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
SUM_LINE_RE = re.compile(r"([0-9a-f]{64})  (.+)")
PHYSICS_EXCLUDED_KEYS = frozenset(
    {
        "asymptotic_dense_time",
        "checkpoint",
        "checkpoint_evidence",
        "elapsed_ratio_right_over_left",
        "elapsed_seconds",
        "maximum_wall_seconds",
        "median_wall_seconds",
        "minimum_wall_seconds",
        "oracle_wall_seconds",
        "orchestration_provenance",
        "production_born_wall_seconds",
        "production_wall_seconds",
        "reconstruction_seconds",
        "report_sha256",
        "resumed",
        "seed_elapsed_seconds",
        "total_elapsed_seconds",
        "total_wall_seconds",
        "wall_seconds",
    }
)
REQUIRED_TEX = (
    "comparison_rows.tex",
    "conservation_rows.tex",
    "cost_accuracy_rows.tex",
    "harmonic_preflight_rows.tex",
    "multidimensional_weak_coupling_summary.tex",
    "open_run_rows.tex",
    "passivity_rows.tex",
    "provenance_rows.tex",
    "release_summary.tex",
    "scba_outcome_rows.tex",
    "weak_coupling_summary.tex",
)
PRODUCTION_FIGURES = (
    "figure_open_oracle_born.pdf",
    "figure_open_oracle_born.png",
    "figure_scba_validation.pdf",
    "figure_scba_validation.png",
    "figure_passivity_backend_timing.pdf",
    "figure_passivity_backend_timing.png",
)
HISTORICAL_FIGURES = (
    "historical_paper_benchmarks.pdf",
    "historical_paper_benchmarks.png",
)


class ReleaseConfigError(RuntimeError):
    """Raised when any caller anchor or nested evidence link fails."""


@dataclass(frozen=True)
class ChecksumTree:
    root: Path
    entries: Mapping[str, str]
    raw_manifest_sha256: str
    local_receipt_sha256: str | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseConfigError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ReleaseConfigError(f"non-finite JSON token: {value}")


def _compact_bytes(value: Any, *, ensure_ascii: bool = True) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii" if ensure_ascii else "utf-8")


def _indented_bytes(value: Any, *, ensure_ascii: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii" if ensure_ascii else "utf-8")


def _hash_document(value: Any) -> str:
    return hashlib.sha256(_compact_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and SHA_RE.fullmatch(value) is not None,
        f"{name} is not a lowercase SHA256",
    )
    return value


def _commit(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and COMMIT_RE.fullmatch(value) is not None,
        f"{name} is not a full lowercase Git commit",
    )
    return value


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} is not an object")
    return value


def _relative(value: Any, name: str) -> PurePosixPath:
    _require(isinstance(value, str) and value, f"{name} is not text")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and pure.as_posix() == value
        and value != "."
        and all(part not in {"", ".", ".."} for part in pure.parts)
        and "\\" not in value,
        f"{name} is unsafe or noncanonical: {value!r}",
    )
    return pure


def _absolute_no_symlinks(path: Path, name: str, *, directory: bool) -> Path:
    expanded = path.expanduser()
    _require(
        ".." not in expanded.parts,
        f"{name} contains a parent-directory component",
    )
    supplied = expanded.absolute()
    _require(supplied.exists(), f"{name} is missing: {supplied}")
    resolved = supplied.resolve(strict=True)
    _require(supplied == resolved, f"{name} contains a symlinked path component")
    if directory:
        _require(resolved.is_dir(), f"{name} is not a directory")
    else:
        _require(resolved.is_file(), f"{name} is not a regular file")
    return resolved


def _strict_json(path: Path, name: str, *, style: str) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseConfigError(f"{name} is not strict JSON") from exc
    _require(isinstance(value, dict), f"{name} is not a JSON object")
    if style == "compact-ascii":
        expected = _compact_bytes(value) + b"\n"
    elif style == "indent-ascii":
        expected = _indented_bytes(value)
    elif style == "indent-utf8":
        expected = _indented_bytes(value, ensure_ascii=False)
    else:
        raise AssertionError(style)
    _require(raw == expected, f"{name} is not canonical {style} JSON")
    return value


def _read_envelope(path: Path, name: str) -> dict[str, Any]:
    document = _strict_json(path, name, style="compact-ascii")
    _require(
        set(document) == {"payload", "payload_sha256"},
        f"{name} has the wrong hashed-envelope shape",
    )
    payload = _mapping(document["payload"], f"{name} payload")
    digest = _sha(document["payload_sha256"], f"{name} payload SHA256")
    _require(
        hmac.compare_digest(_hash_document(payload), digest),
        f"{name} payload SHA256 mismatch",
    )
    return document


def _scan_tree(root: Path, name: str) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        _require(not path.is_symlink(), f"{name} contains a symlink: {relative}")
        if path.is_file():
            files.add(relative)
        else:
            _require(path.is_dir(), f"{name} contains a special file: {relative}")
    return files


def _verify_checksum_tree(
    root: Path,
    expected_raw_sha256: str,
    name: str,
    *,
    fetched: bool = False,
) -> ChecksumTree:
    expected_anchor = _sha(expected_raw_sha256, f"{name} caller anchor")
    resolved = _absolute_no_symlinks(root, name, directory=True)
    manifest = _absolute_no_symlinks(
        resolved / "SHA256SUMS", f"{name} SHA256SUMS", directory=False
    )
    raw = manifest.read_bytes()
    actual_anchor = hashlib.sha256(raw).hexdigest()
    _require(
        hmac.compare_digest(actual_anchor, expected_anchor),
        f"{name} raw SHA256SUMS caller anchor mismatch",
    )
    try:
        text = raw.decode("ascii")
    except UnicodeError as exc:
        raise ReleaseConfigError(f"{name} SHA256SUMS is not ASCII") from exc
    _require(text.endswith("\n"), f"{name} SHA256SUMS lacks a final newline")
    lines = text.splitlines()
    _require(lines and all(lines), f"{name} SHA256SUMS is empty or has blank lines")
    entries: dict[str, str] = {}
    ordered_paths: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        match = SUM_LINE_RE.fullmatch(line)
        _require(match is not None, f"{name} invalid checksum line {line_number}")
        digest, relative_text = match.groups()
        pure = _relative(relative_text, f"{name} checksum path")
        _require(
            pure.name != "SHA256SUMS" and relative_text not in entries,
            f"{name} checksum path is forbidden or duplicated: {relative_text}",
        )
        path = _absolute_no_symlinks(
            resolved.joinpath(*pure.parts),
            f"{name} checksum target {relative_text}",
            directory=False,
        )
        _require(
            path.is_relative_to(resolved),
            f"{name} checksum target escapes its root",
        )
        _require(
            hmac.compare_digest(_file_sha256(path), digest),
            f"{name} SHA256 mismatch for {relative_text}",
        )
        entries[relative_text] = digest
        ordered_paths.append(relative_text)
    _require(
        ordered_paths == sorted(ordered_paths),
        f"{name} SHA256SUMS is not path-sorted",
    )

    actual = _scan_tree(resolved, name)
    excluded = {"SHA256SUMS"}
    receipt_sha: str | None = None
    if fetched:
        receipt = _absolute_no_symlinks(
            resolved / "fetch-receipt.json",
            f"{name} fetch receipt",
            directory=False,
        )
        receipt_document = _strict_json(
            receipt, f"{name} fetch receipt", style="indent-ascii"
        )
        _require(
            receipt_document.get("schema")
            == "negfpy.remote_validation.fetch_receipt.v1",
            f"{name} fetch receipt schema changed",
        )
        _require(
            receipt_document.get("verified_files") == ordered_paths,
            f"{name} fetch receipt coverage differs from SHA256SUMS",
        )
        excluded.add("fetch-receipt.json")
        receipt_sha = _file_sha256(receipt)
    _require(
        actual - excluded == set(entries),
        f"{name} SHA256SUMS coverage mismatch: "
        f"unlisted={sorted((actual - excluded) - set(entries))}, "
        f"nonexistent={sorted(set(entries) - (actual - excluded))}",
    )
    return ChecksumTree(resolved, entries, actual_anchor, receipt_sha)


def _tree_file(
    tree: ChecksumTree,
    relative: Any,
    name: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    pure = _relative(relative, name)
    relative_text = pure.as_posix()
    _require(relative_text in tree.entries, f"{name} is not SHA256SUMS-listed")
    path = _absolute_no_symlinks(
        tree.root.joinpath(*pure.parts), name, directory=False
    )
    _require(path.is_relative_to(tree.root), f"{name} escapes the artifact root")
    digest = _file_sha256(path)
    _require(
        hmac.compare_digest(digest, tree.entries[relative_text]),
        f"{name} differs from verified SHA256SUMS",
    )
    if expected_sha256 is not None:
        _require(
            hmac.compare_digest(digest, _sha(expected_sha256, f"{name} record SHA256")),
            f"{name} record SHA256 mismatch",
        )
    return path


def _relative_to_tree(tree: ChecksumTree, path: Path, name: str) -> str:
    resolved = _absolute_no_symlinks(path, name, directory=False)
    _require(resolved.is_relative_to(tree.root), f"{name} escapes artifact root")
    return resolved.relative_to(tree.root).as_posix()


def _record_file(
    tree: ChecksumTree,
    base: Path,
    record_value: Any,
    name: str,
) -> Path:
    record = _mapping(record_value, f"{name} record")
    pure = _relative(record.get("relative_path"), f"{name} relative path")
    candidate = base.joinpath(*pure.parts)
    global_relative = _relative_to_tree(tree, candidate, name)
    return _tree_file(
        tree,
        global_relative,
        name,
        expected_sha256=_sha(record.get("sha256"), f"{name} record SHA256"),
    )


def _memory(value: Any, *, profile: str, rank_count: int, name: str) -> float:
    record = _mapping(value, name)
    expected = {
        "schema": "negfpy.remote_toy_mpi.aggregate_memory_preflight.v1",
        "measurement": "linux_proc_meminfo_MemAvailable",
        "profile": profile,
        "rank_count": rank_count,
        "allowed_fraction": 0.70,
        "passed": True,
    }
    _require(
        all(record.get(key) == item for key, item in expected.items()),
        f"{name} policy changed or failed",
    )
    available = record.get("available_memory_bytes")
    aggregate = record.get("aggregate_predicted_bytes")
    limit = record.get("limit_bytes")
    _require(
        all(isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in (available, aggregate, limit)),
        f"{name} byte counts are invalid",
    )
    _require(
        limit == int(0.70 * available) and aggregate <= limit,
        f"{name} exceeds the live aggregate 70% RAM gate",
    )
    return aggregate / available


def _rank_runtime(
    value: Any,
    *,
    rank_count: int,
    socket_count: int,
    name: str,
) -> None:
    _require(
        isinstance(value, list) and len(value) == rank_count,
        f"{name} rank coverage is incomplete",
    )
    ranks: set[int] = set()
    cores: set[tuple[str, int, int]] = set()
    packages: dict[tuple[str, int], int] = {}
    for item in value:
        record = _mapping(item, f"{name} record")
        rank = record.get("rank")
        physical = record.get("physical_cores")
        _require(
            isinstance(rank, int)
            and not isinstance(rank, bool)
            and 0 <= rank < rank_count
            and rank not in ranks,
            f"{name} has an invalid or duplicate rank",
        )
        ranks.add(rank)
        _require(
            record.get("one_physical_core") is True
            and isinstance(physical, list)
            and len(physical) == 1
            and record.get("thread_limits") == THREAD_LIMITS,
            f"{name} lacks one-core/thread-limit evidence",
        )
        hostname = record.get("hostname")
        core = _mapping(physical[0], f"{name} core")
        package_id = core.get("physical_package_id")
        core_id = core.get("core_id")
        _require(
            isinstance(hostname, str)
            and hostname
            and isinstance(package_id, int)
            and isinstance(core_id, int),
            f"{name} core identity is invalid",
        )
        key = (hostname, package_id, core_id)
        _require(key not in cores, f"{name} oversubscribes a physical core")
        cores.add(key)
        package = (hostname, package_id)
        packages[package] = packages.get(package, 0) + 1
    _require(ranks == set(range(rank_count)), f"{name} rank sequence is incomplete")
    if rank_count >= socket_count and rank_count % socket_count == 0:
        expected = rank_count // socket_count
        _require(
            len(packages) == socket_count and set(packages.values()) == {expected},
            f"{name} is not balanced across sockets",
        )


def _campaign(
    document: Mapping[str, Any],
    *,
    profile: str,
    commit: str,
    tree_sha: str,
    environment_sha: str,
) -> tuple[list[str], str]:
    payload = _mapping(document.get("payload"), "campaign payload")
    expected = {
        "schema_name": "negfpy_task_manifest",
        "schema_version": 1,
        "source_revision": commit,
        "source_tree_sha256": tree_sha,
        "environment_sha256": environment_sha,
    }
    _require(
        all(payload.get(key) == value for key, value in expected.items()),
        "campaign manifest identity/schema changed",
    )
    metadata = _mapping(payload.get("metadata"), "campaign metadata")
    expected_metadata = {
        "schema": "negfpy_protocol_v2_toy_mpi_campaign",
        "schema_version": 1,
        "protocol": "multidimensional_toy_gate_v2",
        "certification_eligible": profile == "release",
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
    }
    _require(
        all(metadata.get(key) == value for key, value in expected_metadata.items()),
        "campaign metadata/topology changed",
    )
    identity_hashes = _mapping(payload.get("identity_hashes"), "campaign identity hashes")
    _require(
        all(SHA_RE.fullmatch(str(value)) is not None for value in identity_hashes.values()),
        "campaign identity hashes are invalid",
    )
    run_identity = _hash_document(
        {
            "source_revision": commit,
            "source_tree_sha256": tree_sha,
            "environment_sha256": environment_sha,
            "identity_hashes": dict(identity_hashes),
        }
    )
    _require(
        payload.get("run_identity_sha256") == run_identity,
        "campaign run identity SHA256 mismatch",
    )
    tasks = payload.get("tasks")
    _require(isinstance(tasks, list) and len(tasks) == 28, "campaign lacks 28 tasks")
    task_ids: list[str] = []
    kind_counts: dict[str, int] = {}
    for ordinal, raw_task in enumerate(tasks):
        task = _mapping(raw_task, f"campaign task {ordinal}")
        task_id = task.get("task_id")
        hashes = _mapping(task.get("identity_hashes"), f"task {ordinal} identity hashes")
        task_payload = _mapping(task.get("payload"), f"task {ordinal} payload")
        _require(
            isinstance(task_id, str)
            and task_id
            and task_id not in task_ids
            and all(SHA_RE.fullmatch(str(value)) is not None for value in hashes.values()),
            f"campaign task {ordinal} identity is invalid",
        )
        task_ids.append(task_id)
        kind = task_payload.get("kind")
        _require(isinstance(kind, str), f"campaign task {task_id} lacks kind")
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        _require(
            task_payload.get("campaign_schema")
            == "negfpy_protocol_v2_toy_mpi_campaign"
            and task_payload.get("campaign_schema_version") == 1
            and task_payload.get("protocol")
            == "multidimensional_toy_gate_v2",
            f"campaign task contract changed for {task_id}",
        )
        derived = _hash_document(
            {
                "run_identity_sha256": run_identity,
                "ordinal": ordinal,
                "task_id": task_id,
                "identity_hashes": dict(hashes),
                "payload": dict(task_payload),
            }
        )
        _require(
            task.get("task_identity_sha256") == derived,
            f"campaign task identity mismatch for {task_id}",
        )
        dependencies = task_payload.get("dependencies")
        _require(isinstance(dependencies, list), f"task {task_id} dependencies invalid")
    _require(
        kind_counts
        == {
            "harmonic_preflight": 1,
            "backend_preflight": 1,
            "direct_zero_endpoint": 1,
            "multidimensional_case": 12,
            "open_three_mode_case": 13,
        },
        f"campaign task-kind topology changed: {kind_counts}",
    )
    _require(
        task_ids[0] == "preflight.harmonic-same-grid"
        and task_ids[1] == "preflight.backend-equality"
        and all(value.startswith("multidimensional.") for value in task_ids[2:15])
        and all(value.startswith("open_three_mode.") for value in task_ids[15:]),
        "campaign canonical task ordering changed",
    )
    gate = task_ids[2]
    for ordinal, raw_task in enumerate(tasks):
        task = _mapping(raw_task, f"campaign task {ordinal}")
        task_payload = _mapping(task.get("payload"), f"task {ordinal} payload")
        dependencies = task_payload.get("dependencies")
        parent = task_payload.get("continuation_parent_task_id")
        if ordinal < 2:
            expected_dependencies: list[str] = []
            expected_parent = None
        elif ordinal == 2:
            expected_dependencies = task_ids[:2]
            expected_parent = None
        elif ordinal < 8:
            expected_parent = None if ordinal == 3 else task_ids[ordinal - 1]
            expected_dependencies = [gate] + (
                [] if expected_parent is None else [expected_parent]
            )
        else:
            expected_dependencies = [gate]
            expected_parent = None
        _require(
            dependencies == expected_dependencies and parent == expected_parent,
            f"campaign dependency topology changed for {task_ids[ordinal]}",
        )
    _require(
        [
            _mapping(task.get("payload"), "campaign task payload").get("kind")
            for task in tasks
        ]
        == [
            "harmonic_preflight",
            "backend_preflight",
            "direct_zero_endpoint",
            *(["multidimensional_case"] * 12),
            *(["open_three_mode_case"] * 13),
        ],
        "campaign task-kind ordering changed",
    )
    return task_ids, str(document["payload_sha256"])


def _execution(
    document: Mapping[str, Any],
    *,
    tree: ChecksumTree,
    worker_root: Path,
    task_ids: Sequence[str],
    campaign_sha: str,
    execution_id: str,
    rank_count: int,
) -> tuple[str, str]:
    payload = _mapping(document.get("payload"), "execution payload")
    expected_mode = "serial" if rank_count == 1 else "mpi"
    expected = {
        "schema_name": "negfpy_protocol_v2_toy_mpi_execution",
        "schema_version": 1,
        "execution_id": execution_id,
        "campaign_manifest_sha256": campaign_sha,
        "mode": expected_mode,
        "communicator_size": rank_count,
        "complete": True,
    }
    _require(
        all(payload.get(key) == value for key, value in expected.items()),
        "campaign execution identity/schema changed",
    )
    outcomes = payload.get("outcomes")
    _require(
        isinstance(outcomes, list)
        and [item.get("task_id") if isinstance(item, Mapping) else None for item in outcomes]
        == list(task_ids),
        "campaign execution outcome topology changed",
    )
    referenced: set[str] = set()
    scientific_rows: list[dict[str, Any]] = []
    for ordinal, outcome in enumerate(outcomes):
        task_id = str(outcome["task_id"])
        validation = _mapping(
            outcome.get("artifact_validation"), f"{task_id} artifact validation"
        )
        expected_wave = (
            0
            if ordinal < 2
            else 1 if ordinal == 2 else ordinal - 1 if ordinal < 8 else 7
        )
        _require(
            outcome.get("status") == "completed"
            and outcome.get("wave_index") == expected_wave,
            f"{task_id} is not a completed canonical outcome",
        )
        for path_key, sha_key, label in (
            ("checkpoint_relative_path", "checkpoint_sha256", "checkpoint"),
            ("artifact_relative_path", "artifact_sha256", "artifact"),
        ):
            pure = _relative(outcome.get(path_key), f"{task_id} {label} path")
            path = worker_root.joinpath(*pure.parts)
            global_relative = _relative_to_tree(tree, path, f"{task_id} {label}")
            _require(global_relative not in referenced, "campaign reuses outcome files")
            referenced.add(global_relative)
            _tree_file(
                tree,
                global_relative,
                f"{task_id} {label}",
                expected_sha256=_sha(outcome.get(sha_key), f"{task_id} {label} SHA256"),
            )
        scientific_rows.append(
            {
                "task_id": task_id,
                "status": "completed",
                "artifact_validation": dict(validation),
            }
        )
    scientific = _hash_document(
        {
            "campaign_manifest_sha256": campaign_sha,
            "outcomes": scientific_rows,
        }
    )
    _require(
        payload.get("scientific_sha256") == scientific,
        "campaign execution scientific SHA256 mismatch",
    )
    phases = payload.get("phase_execution_sha256")
    _require(
        isinstance(phases, list)
        and len(phases) == 8
        and all(SHA_RE.fullmatch(str(value)) is not None for value in phases),
        "campaign execution phase SHA256 evidence is invalid",
    )
    return scientific, str(document["payload_sha256"])


def _physics_document(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _physics_document(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in PHYSICS_EXCLUDED_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_physics_document(item) for item in value]
    return value


def _physics_sha256(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(_compact_bytes(_physics_document(report))).hexdigest()


def _identity(
    report: Mapping[str, Any],
    *,
    commit: str,
    tree_sha: str,
    environment_sha: str,
) -> None:
    multidimensional = _mapping(
        report.get("multidimensional_scba"), "report multidimensional SCBA"
    )
    provenance = _mapping(
        multidimensional.get("provenance"), "multidimensional provenance"
    )
    identity = _mapping(provenance.get("identity"), "multidimensional identity")
    _require(
        provenance.get("clean") is True
        and provenance.get("complete") is True
        and provenance.get("passed") is True
        and identity.get("git_revision") == commit
        and identity.get("source_tree_sha256") == tree_sha,
        "strict report multidimensional identity changed",
    )
    environment = _mapping(report.get("environment_lock"), "report environment lock")
    _require(
        environment.get("passed") is True
        and environment.get("multidimensional_environment_sha256") == environment_sha
        and environment.get("open_environment_sha256") == environment_sha,
        "strict report environment identity changed",
    )
    open_document = _mapping(report.get("open_three_mode"), "report open-system block")
    runs = open_document.get("runs")
    _require(isinstance(runs, list) and len(runs) == 13, "report lacks 13 open runs")
    for index, run_value in enumerate(runs):
        run = _mapping(run_value, f"open run {index}")
        run_provenance = _mapping(run.get("provenance"), f"open run {index} provenance")
        run_identity = _mapping(
            run_provenance.get("identity"), f"open run {index} identity"
        )
        _require(
            run_provenance.get("clean") is True
            and run_provenance.get("git_dirty") is False
            and run_identity.get("git_revision") == commit
            and run_identity.get("source_tree_sha256") == tree_sha,
            f"open run {index} identity changed",
        )


def _report(
    path: Path,
    *,
    profile: str,
    commit: str,
    tree_sha: str,
    environment_sha: str,
    campaign_sha: str,
    campaign_file_sha: str,
    execution_file_sha: str,
    execution_payload_sha: str,
    scientific_sha: str,
    execution_id: str,
    rank_count: int,
) -> tuple[str, str]:
    report = _strict_json(path, "strict production report", style="indent-utf8")
    _require(
        report.get("schema") == "negfpy.anharmonic.production_toy_gate"
        and report.get("schema_version") == 4,
        "strict report schema changed",
    )
    body_sha = _sha(report.get("report_sha256"), "strict report body SHA256")
    core = {key: value for key, value in report.items() if key != "report_sha256"}
    expected_body = hashlib.sha256(
        _compact_bytes(core, ensure_ascii=False)
    ).hexdigest()
    _require(
        hmac.compare_digest(body_sha, expected_body),
        "strict report body SHA256 mismatch",
    )
    summary = _mapping(report.get("summary"), "strict report summary")
    expected_release = profile == "release"
    _require(
        summary.get("release_certified") is expected_release,
        "strict report release certification state changed",
    )
    if expected_release:
        _require(
            summary.get("validation_tier") == "release"
            and summary.get("failures") == []
            and all(
                value is True
                for key, value in summary.items()
                if key not in {"failures", "validation_tier"}
                and isinstance(value, bool)
            ),
            "strict release summary contains a failed gate",
        )
    _identity(
        report,
        commit=commit,
        tree_sha=tree_sha,
        environment_sha=environment_sha,
    )
    orchestration = _mapping(
        report.get("orchestration_provenance"), "report orchestration provenance"
    )
    expected_orchestration = {
        "campaign_manifest_sha256": campaign_sha,
        "campaign_manifest_file_sha256": campaign_file_sha,
        "execution_manifest_file_sha256": execution_file_sha,
        "execution_payload_sha256": execution_payload_sha,
        "execution_scientific_sha256": scientific_sha,
        "execution_id": execution_id,
        "mode": "serial" if rank_count == 1 else "mpi",
        "communicator_size": rank_count,
        "complete": True,
        "solver_and_gate_contract_unchanged": True,
    }
    _require(
        all(orchestration.get(key) == value for key, value in expected_orchestration.items()),
        "strict report orchestration provenance changed",
    )
    return _file_sha256(path), _physics_sha256(report)


def _worker(
    tree: ChecksumTree,
    worker_relative: str,
    *,
    profile: str,
    rank_count: int,
    socket_count: int,
    commit: str,
    tree_sha: str,
    environment_sha: str,
    expected_execution_id: str,
) -> dict[str, Any]:
    worker_path = _tree_file(tree, worker_relative, f"{profile} worker result")
    worker_document = _read_envelope(worker_path, f"{profile} worker result")
    payload = _mapping(worker_document.get("payload"), f"{profile} worker payload")
    worker_root = worker_path.parent.parent
    _require(
        worker_path
        == worker_root / "runner-results" / f"{expected_execution_id}.json",
        f"{profile} worker result path is not canonical",
    )
    expected = {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": profile,
        "certification_eligible": profile == "release",
        "release_certified": profile == "release",
        "source_revision": commit,
        "source_tree_sha256": tree_sha,
        "environment_sha256": environment_sha,
        "expected_rank_count": rank_count,
        "recorded_rank_count": rank_count,
        "execution_id": expected_execution_id,
        "execution_mode": "serial" if rank_count == 1 else "mpi",
        "thread_limits": THREAD_LIMITS,
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
    }
    _require(
        all(payload.get(key) == value for key, value in expected.items()),
        f"{profile} worker identity/topology changed",
    )
    _require(
        payload.get("directory_mode")
        == ("fresh" if profile == "scaling" else payload.get("directory_mode"))
        and payload.get("directory_mode") in {"fresh", "exact_resume_candidate"},
        f"{profile} worker directory mode changed",
    )
    campaign_path = _record_file(
        tree, worker_root, payload.get("campaign_manifest"), "campaign manifest"
    )
    execution_path = _record_file(
        tree, worker_root, payload.get("execution_manifest"), "execution manifest"
    )
    report_path = _record_file(
        tree, worker_root, payload.get("strict_report"), "strict report"
    )
    _require(
        campaign_path == worker_root / "campaign-manifest.json"
        and execution_path
        == worker_root / "campaign-executions" / f"{expected_execution_id}.json"
        and report_path
        == worker_root / "production-reports" / f"{expected_execution_id}.json",
        f"{profile} worker nested paths are not canonical",
    )
    campaign_document = _read_envelope(campaign_path, "campaign manifest")
    task_ids, campaign_sha = _campaign(
        campaign_document,
        profile=profile,
        commit=commit,
        tree_sha=tree_sha,
        environment_sha=environment_sha,
    )
    _require(
        payload.get("campaign_manifest_sha256") == campaign_sha,
        f"{profile} worker campaign payload SHA256 mismatch",
    )
    execution_document = _read_envelope(execution_path, "execution manifest")
    scientific, execution_payload_sha = _execution(
        execution_document,
        tree=tree,
        worker_root=worker_root,
        task_ids=task_ids,
        campaign_sha=campaign_sha,
        execution_id=expected_execution_id,
        rank_count=rank_count,
    )
    _require(
        payload.get("execution_scientific_sha256") == scientific,
        f"{profile} worker scientific SHA256 mismatch",
    )
    report_file_sha, physics_sha = _report(
        report_path,
        profile=profile,
        commit=commit,
        tree_sha=tree_sha,
        environment_sha=environment_sha,
        campaign_sha=campaign_sha,
        campaign_file_sha=_file_sha256(campaign_path),
        execution_file_sha=_file_sha256(execution_path),
        execution_payload_sha=execution_payload_sha,
        scientific_sha=scientific,
        execution_id=expected_execution_id,
        rank_count=rank_count,
    )
    _require(
        payload.get("production_toy_physics_sha256") == physics_sha,
        f"{profile} worker physics projection SHA256 mismatch",
    )
    memory_fraction = _memory(
        payload.get("memory_preflight"),
        profile=profile,
        rank_count=rank_count,
        name=f"{profile} worker memory preflight",
    )
    _rank_runtime(
        payload.get("rank_runtime"),
        rank_count=rank_count,
        socket_count=socket_count,
        name=f"{profile} worker runtime",
    )
    _sha(payload.get("configuration_sha256"), f"{profile} configuration SHA256")
    return {
        "document": worker_document,
        "payload": payload,
        "file_sha256": _file_sha256(worker_path),
        "campaign_file_sha256": _file_sha256(campaign_path),
        "execution_file_sha256": _file_sha256(execution_path),
        "execution_payload_sha256": execution_payload_sha,
        "report_file_sha256": report_file_sha,
        "report_body_sha256": _strict_json(
            report_path, "strict report", style="indent-utf8"
        )["report_sha256"],
        "physics_sha256": physics_sha,
        "memory_fraction": memory_fraction,
    }


def _selection(
    tree: ChecksumTree,
    selection_relative: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = _tree_file(tree, selection_relative, "scaling selection")
    document = _read_envelope(selection_path, "scaling selection")
    payload = _mapping(document.get("payload"), "scaling selection payload")
    expected = {
        "schema": "negfpy.remote_toy_mpi.scaling_selection.v1",
        "profile": "representative_noncertifying_full_28_task",
        "physics_equal_across_all_ranks": True,
        "fresh_isolated_repetitions": True,
    }
    _require(
        all(payload.get(key) == value for key, value in expected.items()),
        "scaling selection contract changed",
    )
    commit = _commit(payload.get("source_revision"), "scaling source revision")
    tree_sha = _sha(payload.get("source_tree_sha256"), "scaling source-tree SHA256")
    environment_sha = _sha(
        payload.get("environment_sha256"), "scaling environment SHA256"
    )
    common_physics = _sha(
        payload.get("production_toy_physics_sha256"),
        "scaling common physics SHA256",
    )
    cores = payload.get("physical_core_count")
    sockets = payload.get("socket_count")
    _require(
        isinstance(cores, int)
        and not isinstance(cores, bool)
        and cores >= 16
        and isinstance(sockets, int)
        and not isinstance(sockets, bool)
        and sockets > 0,
        "scaling host topology is invalid",
    )
    policy = _mapping(payload.get("selection"), "rank selection policy")
    measurements = policy.get("measurements")
    _require(
        policy.get("schema") == "negfpy.remote_mpi_scaling.selection.v1"
        and policy.get("rank_ladder") == list(RANK_LADDER)
        and isinstance(measurements, list)
        and len(measurements) == 5,
        "rank selection schema/ladder changed",
    )
    wall: dict[int, float] = {}
    for expected_rank, raw_row in zip(RANK_LADDER, measurements, strict=True):
        row = _mapping(raw_row, f"rank-{expected_rank} measurement")
        seconds = row.get("wall_seconds")
        _require(
            row.get("rank_count") == expected_rank
            and row.get("completed") is True
            and row.get("numerically_equal") is True
            and isinstance(seconds, (int, float))
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) > 0.0,
            f"rank-{expected_rank} timing measurement failed",
        )
        wall[expected_rank] = float(seconds)
    fastest_rank = min(RANK_LADDER, key=lambda rank: (wall[rank], rank))
    fastest_wall = wall[fastest_rank]
    candidate_rank = min(
        rank for rank in RANK_LADDER if wall[rank] <= fastest_wall * 1.05
    )
    speedup = wall[1] / wall[candidate_rank]
    mpi_enabled = candidate_rank > 1 and speedup >= 2.0
    toy_rank = candidate_rank if mpi_enabled else 1
    expected_policy = {
        "fastest_rank_count": fastest_rank,
        "fastest_wall_seconds": fastest_wall,
        "smallest_near_fastest_rank_count": candidate_rank,
        "candidate_speedup": speedup,
        "toy_rank_count": toy_rank,
        "silicon_rank_count": candidate_rank,
        "toy_mpi_enabled": mpi_enabled,
        "within_fastest_fraction": 0.05,
        "minimum_toy_speedup": 2.0,
    }
    _require(
        all(
            policy.get(key) == value
            if not isinstance(value, float)
            else isinstance(policy.get(key), (int, float))
            and math.isclose(float(policy[key]), value, rel_tol=1e-12, abs_tol=1e-12)
            for key, value in expected_policy.items()
        ),
        "rank selection does not follow the locked policy",
    )

    rows = payload.get("rank_evidence")
    _require(isinstance(rows, list) and len(rows) == 5, "rank evidence is incomplete")
    scaling_root = selection_path.parent
    observed: set[int] = set()
    maximum_memory = 0.0
    for row_value in rows:
        row = _mapping(row_value, "rank evidence")
        rank = row.get("rank_count")
        _require(rank in RANK_LADDER and rank not in observed, "rank evidence duplicated")
        observed.add(rank)
        rank_manifest_path = _tree_file(
            tree,
            _relative_to_tree(
                tree,
                scaling_root / str(row.get("manifest_path")),
                f"rank-{rank} manifest",
            ),
            f"rank-{rank} manifest",
            expected_sha256=_sha(
                row.get("manifest_file_sha256"),
                f"rank-{rank} manifest file SHA256",
            ),
        )
        _require(
            rank_manifest_path == scaling_root / f"rank-{rank:03d}-manifest.json",
            f"rank-{rank} manifest path changed",
        )
        rank_document = _read_envelope(rank_manifest_path, f"rank-{rank} manifest")
        _require(
            rank_document["payload_sha256"] == row.get("manifest_payload_sha256"),
            f"rank-{rank} manifest payload SHA256 mismatch",
        )
        rank_payload = _mapping(rank_document["payload"], f"rank-{rank} payload")
        expected_rank_payload = {
            "schema": "negfpy.remote_toy_mpi.scaling_rank.v1",
            "rank_count": rank,
            "source_revision": commit,
            "source_tree_sha256": tree_sha,
            "environment_sha256": environment_sha,
            "fresh_isolated_repetition": True,
            "resume_used": False,
            "returncode": 0,
            "completed": True,
            "failure": None,
            "thread_limits": THREAD_LIMITS,
        }
        _require(
            all(rank_payload.get(key) == value for key, value in expected_rank_payload.items()),
            f"rank-{rank} repetition payload changed",
        )
        worker_relative = _relative(
            rank_payload.get("worker_result_relative_path"),
            f"rank-{rank} worker path",
        ).as_posix()
        worker_global = _relative_to_tree(
            tree, scaling_root.joinpath(*PurePosixPath(worker_relative).parts),
            f"rank-{rank} worker",
        )
        worker = _worker(
            tree,
            worker_global,
            profile="scaling",
            rank_count=rank,
            socket_count=sockets,
            commit=commit,
            tree_sha=tree_sha,
            environment_sha=environment_sha,
            expected_execution_id=f"scaling-rank-{rank:03d}",
        )
        _require(
            rank_payload.get("worker_result_file_sha256") == worker["file_sha256"]
            and rank_payload.get("worker_result_payload_sha256")
            == worker["document"]["payload_sha256"]
            and row.get("worker_result_file_sha256") == worker["file_sha256"]
            and row.get("campaign_manifest_file_sha256")
            == worker["campaign_file_sha256"]
            and row.get("execution_manifest_file_sha256")
            == worker["execution_file_sha256"]
            and row.get("strict_report_file_sha256")
            == worker["report_file_sha256"]
            and rank_payload.get("production_toy_physics_sha256") == common_physics
            and worker["physics_sha256"] == common_physics,
            f"rank-{rank} nested chain hashes differ",
        )
        for name in ("stdout", "stderr"):
            _record_file(tree, scaling_root, rank_payload.get(name), f"rank-{rank} {name}")
        _sha(
            _mapping(rank_payload.get("launcher"), "launcher").get("sha256"),
            f"rank-{rank} launcher SHA256",
        )
        _sha(
            _mapping(rank_payload.get("python"), "python").get("sha256"),
            f"rank-{rank} Python SHA256",
        )
        _require(
            isinstance(rank_payload.get("binding_arguments"), list)
            and isinstance(rank_payload.get("argv"), list),
            f"rank-{rank} launcher evidence is incomplete",
        )
        launcher_memory = _memory(
            rank_payload.get("launcher_memory_preflight"),
            profile="scaling",
            rank_count=rank,
            name=f"rank-{rank} launcher memory",
        )
        _require(
            row.get("launcher_memory_preflight")
            == rank_payload.get("launcher_memory_preflight")
            and row.get("worker_memory_preflight")
            == worker["payload"].get("memory_preflight")
            and row.get("rank_runtime") == worker["payload"].get("rank_runtime")
            and isinstance(row.get("harmonic_preflight"), Mapping),
            f"rank-{rank} selection evidence differs from worker evidence",
        )
        maximum_memory = max(maximum_memory, launcher_memory, worker["memory_fraction"])
    _require(observed == set(RANK_LADDER), "rank evidence ladder is incomplete")
    return (
        {
            "commit": commit,
            "tree_sha": tree_sha,
            "environment_sha": environment_sha,
            "socket_count": sockets,
            "fastest_rank": fastest_rank,
            "candidate_rank": candidate_rank,
            "toy_rank": toy_rank,
            "silicon_rank": candidate_rank,
            "baseline_wall": wall[1],
            "candidate_wall": wall[candidate_rank],
            "candidate_speedup": speedup,
            "mpi_enabled": mpi_enabled,
            "scaling_physics_sha": common_physics,
            "maximum_memory": maximum_memory,
        },
        {
            "file_sha256": _file_sha256(selection_path),
            "payload_sha256": document["payload_sha256"],
        },
    )


def _validate_staging(
    tree: ChecksumTree,
    *,
    commit: str,
    tree_sha: str,
    environment_sha: str,
    artifact_anchor: str,
    release_report_sha: str,
) -> str:
    for filename in REQUIRED_TEX:
        _tree_file(tree, f"tex/{filename}", f"staging TeX {filename}")
    staged_report = _tree_file(
        tree,
        "production_toy_gate.json",
        "staged strict report",
        expected_sha256=release_report_sha,
    )
    receipt_path = _tree_file(tree, "import_receipt.json", "staging import receipt")
    receipt = _strict_json(receipt_path, "staging import receipt", style="indent-ascii")
    _require(
        receipt.get("schema") == "negfpy.anharmonic.production_toy_gate_import_receipt"
        and receipt.get("schema_version") == 2,
        "staging import receipt schema changed",
    )
    recorded_body = _sha(receipt.get("receipt_sha256"), "import receipt body SHA256")
    receipt_core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _require(
        recorded_body == _hash_document(receipt_core),
        "staging import receipt body SHA256 mismatch",
    )
    source = _mapping(receipt.get("source"), "staging receipt source")
    contract = _mapping(receipt.get("release_contract"), "staging release contract")
    expected_contract = {
        "checkpoint_count": 13,
        "multidimensional_outcome_count": 13,
        "open_system_run_count": 13,
        "protocol": "multidimensional_toy_gate_v2",
        "protocol_version": 2,
        "release_certified": True,
        "schema_version": 4,
        "validation_tier": "release",
    }
    _require(
        source.get("commit") == commit
        and source.get("source_tree_sha256") == tree_sha
        and source.get("environment_sha256") == environment_sha
        and source.get("report_file_sha256") == release_report_sha
        and source.get("report_relative_path") == "production_toy_gate.json"
        and source.get("sha256sums_sha256") == artifact_anchor
        and all(contract.get(key) == value for key, value in expected_contract.items()),
        "staging import receipt source/release contract changed",
    )
    _require(_file_sha256(staged_report) == release_report_sha, "staged report changed")
    staged = _mapping(receipt.get("staged_artifacts"), "staged artifact records")
    expected_staged = set(tree.entries) - {"import_receipt.json"}
    _require(set(staged) == expected_staged, "staging receipt coverage changed")
    for relative_text, record_value in staged.items():
        record = _mapping(record_value, f"staged artifact {relative_text}")
        _require(
            record.get("sha256") == tree.entries[relative_text],
            f"staging receipt SHA256 mismatch for {relative_text}",
        )
    return _file_sha256(receipt_path)


def _validate_production_figures(
    tree: ChecksumTree,
    *,
    commit: str,
    staging_anchor: str,
    staging_receipt_sha: str,
    release_report_sha: str,
) -> str:
    for filename in PRODUCTION_FIGURES:
        _tree_file(tree, filename, f"production figure {filename}")
    receipt_path = _tree_file(tree, "render_manifest.json", "figure render manifest")
    receipt = _strict_json(receipt_path, "figure render manifest", style="indent-ascii")
    _require(
        receipt.get("schema") == "negfpy.anharmonic.production_toy_gate_figure_render"
        and receipt.get("schema_version") == 1,
        "production figure receipt schema changed",
    )
    recorded = _sha(receipt.get("manifest_sha256"), "figure manifest body SHA256")
    core = {key: value for key, value in receipt.items() if key != "manifest_sha256"}
    _require(recorded == _hash_document(core), "figure manifest body SHA256 mismatch")
    source = _mapping(receipt.get("source"), "figure receipt source")
    _require(
        source.get("commit") == commit
        and source.get("staging_sha256sums_file_sha256") == staging_anchor
        and source.get("import_receipt_file_sha256") == staging_receipt_sha
        and source.get("report_file_sha256") == release_report_sha,
        "production figure source binding changed",
    )
    figures = _mapping(receipt.get("figures"), "figure receipt records")
    for stem in (
        "figure_open_oracle_born",
        "figure_scba_validation",
        "figure_passivity_backend_timing",
    ):
        record = _mapping(figures.get(stem), f"figure record {stem}")
        for extension in ("pdf", "png"):
            item = _mapping(record.get(extension), f"{stem} {extension} record")
            path = item.get("path")
            _require(
                path == f"{stem}.{extension}"
                and item.get("sha256") == tree.entries[path],
                f"{stem} {extension} receipt binding changed",
            )
    return _file_sha256(receipt_path)


def _validate_historical_figures(tree: ChecksumTree) -> str:
    for filename in HISTORICAL_FIGURES:
        _tree_file(tree, filename, f"historical figure {filename}")
    receipt_path = _tree_file(
        tree,
        "historical_paper_benchmarks_render_manifest.json",
        "historical figure receipt",
    )
    receipt = _strict_json(
        receipt_path, "historical figure receipt", style="indent-ascii"
    )
    _require(
        receipt.get("schema")
        == "negfpy.anharmonic.historical_paper_benchmark_figure_render.v1"
        and receipt.get("production_evidence") is False
        and receipt.get("release_certified") is False
        and receipt.get("status") == "NONCERTIFYING_HISTORICAL_SENSITIVITY",
        "historical figure receipt scope changed",
    )
    hashes = _mapping(receipt.get("figure_sha256"), "historical figure SHA256 records")
    _require(
        all(hashes.get(filename) == tree.entries[filename] for filename in HISTORICAL_FIGURES),
        "historical figure receipt binding changed",
    )
    output_contract = _mapping(
        receipt.get("output_contract"), "historical output contract"
    )
    _require(
        output_contract.get("sha256sums_exact_coverage") is True
        and output_contract.get("vector_pdf") is True
        and output_contract.get("rgb_png_450_dpi") is True,
        "historical figure output contract changed",
    )
    return _file_sha256(receipt_path)


def _safe_tex_relative(value: str, name: str) -> str:
    _require(
        isinstance(value, str) and SAFE_TEX_PATH_RE.fullmatch(value) is not None,
        f"{name} contains unsafe TeX/path characters",
    )
    pure = _relative(value, name)
    return pure.as_posix()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def generate_config(
    *,
    artifact_root: Path,
    expected_artifact_sha256sums_sha256: str,
    scaling_selection_relative: str,
    release_runner_relative: str,
    staging_dir: str,
    expected_staging_sha256sums_sha256: str,
    production_figure_dir: str,
    expected_production_figure_sha256sums_sha256: str,
    historical_figure_dir: str,
    expected_historical_figure_sha256sums_sha256: str,
    report_date: str,
    output: Path,
) -> str:
    artifact_tree = _verify_checksum_tree(
        artifact_root,
        expected_artifact_sha256sums_sha256,
        "fetched artifact root",
        fetched=True,
    )
    selection_relative = _relative(
        scaling_selection_relative, "scaling selection relative path"
    ).as_posix()
    release_relative = _relative(
        release_runner_relative, "release runner relative path"
    ).as_posix()
    _require(
        selection_relative in artifact_tree.entries
        and release_relative in artifact_tree.entries,
        "scaling selection or release runner is not fetched-manifest listed",
    )
    scaling, scaling_record = _selection(artifact_tree, selection_relative)
    fetch_receipt = _strict_json(
        artifact_tree.root / "fetch-receipt.json",
        "fetch receipt",
        style="indent-ascii",
    )
    _require(
        fetch_receipt.get("commit") == scaling["commit"],
        "fetch receipt commit differs from scaling identity",
    )
    release = _worker(
        artifact_tree,
        release_relative,
        profile="release",
        rank_count=scaling["toy_rank"],
        socket_count=scaling["socket_count"],
        commit=scaling["commit"],
        tree_sha=scaling["tree_sha"],
        environment_sha=scaling["environment_sha"],
        expected_execution_id=Path(release_relative).stem,
    )

    target = output.expanduser().absolute()
    _require(
        not target.exists() or not target.is_symlink(),
        "release config output cannot be a symlink",
    )
    staging_relative = _safe_tex_relative(staging_dir, "staging directory")
    production_relative = _safe_tex_relative(
        production_figure_dir, "production figure directory"
    )
    historical_relative = _safe_tex_relative(
        historical_figure_dir, "historical figure directory"
    )
    staging_tree = _verify_checksum_tree(
        target.parent / staging_relative,
        expected_staging_sha256sums_sha256,
        "production staging",
    )
    production_tree = _verify_checksum_tree(
        target.parent / production_relative,
        expected_production_figure_sha256sums_sha256,
        "production figures",
    )
    historical_tree = _verify_checksum_tree(
        target.parent / historical_relative,
        expected_historical_figure_sha256sums_sha256,
        "historical figures",
    )
    staging_receipt_sha = _validate_staging(
        staging_tree,
        commit=scaling["commit"],
        tree_sha=scaling["tree_sha"],
        environment_sha=scaling["environment_sha"],
        artifact_anchor=artifact_tree.raw_manifest_sha256,
        release_report_sha=release["report_file_sha256"],
    )
    production_receipt_sha = _validate_production_figures(
        production_tree,
        commit=scaling["commit"],
        staging_anchor=staging_tree.raw_manifest_sha256,
        staging_receipt_sha=staging_receipt_sha,
        release_report_sha=release["report_file_sha256"],
    )
    historical_receipt_sha = _validate_historical_figures(historical_tree)
    _require(DATE_RE.fullmatch(report_date) is not None, "report date is invalid")

    release_mode = "task-MPI" if scaling["toy_rank"] > 1 else "serial/FFT"
    macros = (
        ("ProductionStagingDir", staging_relative),
        ("ProductionFigureDir", production_relative),
        ("HistoricalFigureDir", historical_relative),
        ("ExpectedToyGateCommit", scaling["commit"]),
        ("ReportDate", report_date),
        ("MPIRankLadder", "1, 2, 4, 8, and 16"),
        ("MPIFastestRankCount", str(scaling["fastest_rank"])),
        ("MPICandidateRankCount", str(scaling["candidate_rank"])),
        ("MPIReleaseRankCount", str(scaling["toy_rank"])),
        ("MPISiliconRankCount", str(scaling["silicon_rank"])),
        ("MPIReleaseMode", release_mode),
        ("MPIBaselineWallSeconds", f"{scaling['baseline_wall']:.6f}"),
        ("MPICandidateWallSeconds", f"{scaling['candidate_wall']:.6f}"),
        ("MPICandidateSpeedup", f"{scaling['candidate_speedup']:.6f}"),
        (
            "MPIMaxMemoryFractionObserved",
            f"{max(scaling['maximum_memory'], release['memory_fraction']):.6f}",
        ),
        ("MPISocketCount", str(scaling["socket_count"])),
        ("MPIScalingPhysicsSHA", scaling["scaling_physics_sha"]),
        ("MPIReleasePhysicsSHA", release["physics_sha256"]),
        ("MPIScalingManifestSHA", scaling_record["file_sha256"]),
        ("MPIRunnerManifestSHA", release["file_sha256"]),
        ("MPIExecutionManifestSHA", release["execution_file_sha256"]),
        ("ArtifactSHA256SUMSRawSHA", artifact_tree.raw_manifest_sha256),
        ("ArtifactFetchReceiptSHA", str(artifact_tree.local_receipt_sha256)),
        ("StagingSHA256SUMSRawSHA", staging_tree.raw_manifest_sha256),
        ("StagingImportReceiptSHA", staging_receipt_sha),
        ("ProductionFiguresSHA256SUMSRawSHA", production_tree.raw_manifest_sha256),
        ("ProductionFiguresReceiptSHA", production_receipt_sha),
        ("HistoricalFiguresSHA256SUMSRawSHA", historical_tree.raw_manifest_sha256),
        ("HistoricalFiguresReceiptSHA", historical_receipt_sha),
        ("ReleaseStrictReportFileSHA", release["report_file_sha256"]),
        ("ReleaseStrictReportBodySHA", release["report_body_sha256"]),
        ("MPIToyMinimumSpeedupSatisfied", str(scaling["mpi_enabled"]).lower()),
        ("MPIScalingPassed", "true"),
        ("MPIPhysicsEqualityPassed", "true"),
        ("MPIBindingPassed", "true"),
        ("MPIMemoryPreflightPassed", "true"),
        ("MPIRunnerReleaseCertified", "true"),
    )
    text = "\n".join(
        [
            "% Generated from caller-anchored exhaustive checksum manifests.",
            "% Do not hand-edit; regenerate when any exact artifact changes.",
            *(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros),
            "",
        ]
    )
    _atomic_write(target, text)
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--expected-artifact-sha256sums-sha256", required=True)
    parser.add_argument("--scaling-selection-relative", required=True)
    parser.add_argument("--release-runner-relative", required=True)
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--expected-staging-sha256sums-sha256", required=True)
    parser.add_argument("--production-figure-dir", required=True)
    parser.add_argument(
        "--expected-production-figure-sha256sums-sha256", required=True
    )
    parser.add_argument("--historical-figure-dir", required=True)
    parser.add_argument(
        "--expected-historical-figure-sha256sums-sha256", required=True
    )
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--output", type=Path, default=Path("release_config.tex"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        generate_config(
            artifact_root=arguments.artifact_root,
            expected_artifact_sha256sums_sha256=(
                arguments.expected_artifact_sha256sums_sha256
            ),
            scaling_selection_relative=arguments.scaling_selection_relative,
            release_runner_relative=arguments.release_runner_relative,
            staging_dir=arguments.staging_dir,
            expected_staging_sha256sums_sha256=(
                arguments.expected_staging_sha256sums_sha256
            ),
            production_figure_dir=arguments.production_figure_dir,
            expected_production_figure_sha256sums_sha256=(
                arguments.expected_production_figure_sha256sums_sha256
            ),
            historical_figure_dir=arguments.historical_figure_dir,
            expected_historical_figure_sha256sums_sha256=(
                arguments.expected_historical_figure_sha256sums_sha256
            ),
            report_date=arguments.report_date,
            output=arguments.output,
        )
    except (OSError, ReleaseConfigError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
