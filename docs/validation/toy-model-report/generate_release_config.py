#!/usr/bin/env python3
"""Generate a fail-closed LaTeX config from externally anchored evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import hmac
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import sys
from types import MappingProxyType
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
ORCHESTRATION_SCRIPT_PATHS = (
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
REMOTE_EXPECTED_USER = "hangbo"
REMOTE_TOOLS_BIN = "/home/hangbo/micromamba/envs/negf-tools/bin"


REQUIRED_SUMMARY_GATES = frozenset(
    {
        "code_identity_consistent",
        "exact_release_revision_passed",
        "cross_platform_identity_consistent",
        "environment_lock_consistent",
        "fixture_contract_passed",
        "full_locked_base_coverage",
        "multidimensional_release_certified",
        "multidimensional_harmonic_preflight_passed",
        "multidimensional_backend_validation_passed",
        "multidimensional_outcome_certification_passed",
        "multidimensional_contract_passed",
        "multidimensional_static_configuration_passed",
        "multidimensional_runtime_evidence_passed",
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
        "oracle_gate_passed",
        "refinement_coverage",
        "refinement_passed",
        "release_certified",
        "scaling_coverage",
        "scaling_passed",
        "sign_coverage",
        "sign_invariance_passed",
        "spontaneous_decay_coverage",
        "spontaneous_decay_passed",
    }
)
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
HISTORICAL_METRICS = "historical_paper_benchmarks_metrics.json"
HISTORICAL_RENDER_MANIFEST = (
    "historical_paper_benchmarks_render_manifest.json"
)
IMPORTER_SOURCE = (
    "docs/validation/toy-model-report/scripts/import_production_toy_gate.py"
)
PRODUCTION_RENDERER_SOURCE = (
    "docs/validation/toy-model-report/scripts/render_production_toy_gate_figures.py"
)
HISTORICAL_RENDERER_SOURCE = (
    "docs/validation/toy-model-report/scripts/render_historical_paper_benchmarks.py"
)
HISTORICAL_DATA_ROOT = "docs/validation/toy-model-report/historical-data"



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


def _load_committed_module(
    relative: str,
    source_records: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Any, str]:
    record = _mapping(source_records.get(relative), f"{name} source record")
    expected_keys = {"path", "git_tracked", "kind", "size", "mode", "sha256"}
    _require(
        set(record) == expected_keys
        and record.get("path") == relative
        and record.get("git_tracked") is True
        and record.get("kind") == "regular_file"
        and isinstance(record.get("size"), int)
        and not isinstance(record.get("size"), bool)
        and record["size"] > 0,
        f"{name} is not an exact committed regular-file source record",
    )
    path = Path(__file__).resolve().parents[3] / relative
    _require(path.is_file() and not path.is_symlink(), f"{name} source is missing or unsafe")
    digest = _file_sha256(path)
    _require(
        path.stat().st_size == record["size"]
        and digest == _sha(record.get("sha256"), f"{name} source SHA256"),
        f"local {name} bytes differ from the remotely executed commit",
    )
    module_name = f"_negfpy_release_{name.replace('-', '_')}_{digest[:16]}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    _require(
        specification is not None and specification.loader is not None,
        f"cannot load committed {name}",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise ReleaseConfigError(f"cannot execute committed {name}") from error
    finally:
        sys.modules.pop(module_name, None)
    _require(_file_sha256(path) == digest, f"committed {name} changed during loading")
    return module, digest


def _exact_local_source_file(
    relative: str,
    source_records: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Path, str]:
    record = _mapping(source_records.get(relative), f"{name} source record")
    _require(
        set(record) == {"path", "git_tracked", "kind", "size", "mode", "sha256"}
        and record.get("path") == relative
        and record.get("git_tracked") is True
        and record.get("kind") == "regular_file"
        and isinstance(record.get("size"), int)
        and not isinstance(record.get("size"), bool)
        and record["size"] >= 0,
        f"{name} is not an exact committed regular-file source record",
    )
    path = Path(__file__).resolve().parents[3] / relative
    _require(path.is_file() and not path.is_symlink(), f"{name} source is missing or unsafe")
    digest = _file_sha256(path)
    _require(
        path.stat().st_size == record["size"]
        and digest == _sha(record.get("sha256"), f"{name} source SHA256"),
        f"local {name} bytes differ from the remotely executed commit",
    )
    return path, digest


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
) -> tuple[list[str], dict[str, str], str]:
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
    task_identities: dict[str, str] = {}
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
        task_identities[task_id] = _sha(
            task.get("task_identity_sha256"),
            f"campaign task {task_id} identity SHA256",
        )
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
    return task_ids, task_identities, str(document["payload_sha256"])


def _execution(
    document: Mapping[str, Any],
    *,
    tree: ChecksumTree,
    worker_root: Path,
    task_ids: Sequence[str],
    task_identities: Mapping[str, str],
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
        array_sha256 = _mapping(
            validation.get("array_sha256"), f"{task_id} artifact array SHA256"
        )
        _require(
            set(validation)
            == {
                "schema_name",
                "schema_version",
                "campaign_manifest_sha256",
                "campaign_task_identity_sha256",
                "case_status",
                "evidence_sha256",
                "array_sha256",
            }
            and validation.get("schema_name") == "negfpy_protocol_v2_toy_case"
            and validation.get("schema_version") == 1
            and validation.get("campaign_manifest_sha256") == campaign_sha
            and validation.get("campaign_task_identity_sha256")
            == task_identities.get(task_id)
            and validation.get("case_status") == "completed"
            and SHA_RE.fullmatch(str(validation.get("evidence_sha256"))) is not None
            and all(SHA_RE.fullmatch(str(value)) is not None for value in array_sha256.values()),
            f"{task_id} artifact validation failed or changed schema",
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
            set(summary) == REQUIRED_SUMMARY_GATES | {"failures", "validation_tier"}
            and summary.get("validation_tier") == "release"
            and summary.get("failures") == []
            and all(summary.get(key) is True for key in REQUIRED_SUMMARY_GATES),
            "strict release summary is missing, failed, or has an unexpected gate",
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
    nested_root: Path | None = None,
) -> dict[str, Any]:
    worker_path = _tree_file(tree, worker_relative, f"{profile} worker result")
    worker_document = _read_envelope(worker_path, f"{profile} worker result")
    payload = _mapping(worker_document.get("payload"), f"{profile} worker payload")
    worker_root = (
        worker_path.parent.parent
        if nested_root is None
        else _absolute_no_symlinks(nested_root, f"{profile} nested root", directory=True)
    )
    if nested_root is None:
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
    task_ids, task_identities, campaign_sha = _campaign(
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
        task_identities=task_identities,
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
    report_document = _strict_json(
        report_path, "strict report", style="indent-utf8"
    )
    return {
        "document": worker_document,
        "payload": payload,
        "file_sha256": _file_sha256(worker_path),
        "campaign_file_sha256": _file_sha256(campaign_path),
        "execution_file_sha256": _file_sha256(execution_path),
        "execution_payload_sha256": execution_payload_sha,
        "report_file_sha256": report_file_sha,
        "report_path": report_path,
        "report_body_sha256": report_document["report_sha256"],
        "nested_report_body_sha256": _mapping(
            report_document.get("multidimensional_scba"),
            "strict report multidimensional block",
        ).get("report_sha256"),
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
    terminal_rank_evidence: list[dict[str, Any]] = []
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
            all(
                rank_payload.get(key) == value
                for key, value in expected_rank_payload.items()
            )
            and isinstance(rank_payload.get("outer_wall_seconds"), (int, float))
            and not isinstance(rank_payload.get("outer_wall_seconds"), bool)
            and math.isclose(
                float(rank_payload["outer_wall_seconds"]),
                wall[rank],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            f"rank-{rank} repetition payload or selected timing changed",
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
            and isinstance(row.get("harmonic_preflight"), Mapping)
            and row["harmonic_preflight"].get("passed") is True
            and row["harmonic_preflight"].get("real_physics_executed") is True,
            f"rank-{rank} selection evidence differs from worker evidence",
        )
        maximum_memory = max(maximum_memory, launcher_memory, worker["memory_fraction"])
        terminal_rank_evidence.append(
            {
                "rank_count": rank,
                "manifest_file_sha256": row["manifest_file_sha256"],
                "manifest_payload_sha256": row["manifest_payload_sha256"],
                "worker_result_file_sha256": row["worker_result_file_sha256"],
                "campaign_manifest_file_sha256": row[
                    "campaign_manifest_file_sha256"
                ],
                "execution_manifest_file_sha256": row[
                    "execution_manifest_file_sha256"
                ],
                "strict_report_file_sha256": row["strict_report_file_sha256"],
                "launcher_memory_preflight_passed": True,
                "worker_memory_preflight_passed": True,
                "harmonic_preflight_passed": True,
                "rank_runtime_count": rank,
            }
        )
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
            "physical_core_count": cores,
            "rank_evidence": terminal_rank_evidence,
        },
        {
            "file_sha256": _file_sha256(selection_path),
            "payload_sha256": document["payload_sha256"],
        },
    )



def _publication_record(
    tree: ChecksumTree,
    publication: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    name: str,
    expected_relative: str,
) -> Path:
    relative = _relative(publication.get(path_key), f"{name} publication path").as_posix()
    _require(relative == expected_relative, f"{name} publication path is not canonical")
    return _tree_file(
        tree,
        relative,
        name,
        expected_sha256=_sha(publication.get(sha_key), f"{name} publication SHA256"),
    )


def _current_publication(tree: ChecksumTree) -> dict[str, Any]:
    _require(not (tree.root / "job.lock").exists(), "fetched artifact root has an active job lock")
    current_path = _tree_file(tree, "current-attempt.json", "current attempt publication")
    current = _strict_json(current_path, "current attempt publication", style="indent-ascii")
    expected_publication_keys = {
        "schema",
        "attempt",
        "state",
        "commit",
        "attempt_path",
        "status",
        "status_sha256",
        "log",
        "log_sha256",
        "report",
        "report_sha256",
        "report_facts",
        "scaling_selection",
        "scaling_selection_sha256",
        "mpi_release_launch_intent",
        "mpi_release_launch_intent_sha256",
        "mpi_release_launch",
        "mpi_release_launch_sha256",
        "mpi_release_stdout",
        "mpi_release_stdout_sha256",
        "mpi_release_stderr",
        "mpi_release_stderr_sha256",
        "mpi_runner_result",
        "mpi_runner_result_sha256",
        "mpi_execution_manifest",
        "mpi_execution_manifest_sha256",
        "mpi_campaign_manifest",
        "mpi_campaign_manifest_sha256",
    }
    attempt = current.get("attempt")
    commit = _commit(current.get("commit"), "current publication commit")
    _require(
        set(current) == expected_publication_keys
        and current.get("schema") == "negfpy.remote_validation.attempt_publication.v1"
        and current.get("state") == "completed"
        and isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and attempt > 0,
        "current attempt publication is not an exact completed publication",
    )
    attempt_relative = f"attempts/attempt-{attempt:03d}"
    _require(
        current.get("attempt_path") == attempt_relative,
        "current publication attempt path changed",
    )
    publication_path = _tree_file(
        tree,
        f"{attempt_relative}/publication.json",
        "selected attempt publication",
    )
    publication = _strict_json(
        publication_path, "selected attempt publication", style="indent-ascii"
    )
    _require(
        publication == current and publication_path.read_bytes() == current_path.read_bytes(),
        "current-attempt.json is not byte-identical to the selected publication",
    )

    published_scaling_relative = str(current.get("scaling_selection"))
    _require(
        re.fullmatch(
            r"attempts/attempt-[0-9]{3}/mpi-scaling/scaling-selection\.json",
            published_scaling_relative,
        )
        is not None,
        "published scaling selection path is not canonical",
    )
    paths = {
        "status": _publication_record(
            tree,
            current,
            path_key="status",
            sha_key="status_sha256",
            name="attempt status",
            expected_relative=f"{attempt_relative}/status.json",
        ),
        "log": _publication_record(
            tree,
            current,
            path_key="log",
            sha_key="log_sha256",
            name="attempt log",
            expected_relative=f"{attempt_relative}/production_toy_gate.log",
        ),
        "report": _publication_record(
            tree,
            current,
            path_key="report",
            sha_key="report_sha256",
            name="published strict report",
            expected_relative=f"{attempt_relative}/production_toy_gate.json",
        ),
        "scaling": _publication_record(
            tree,
            current,
            path_key="scaling_selection",
            sha_key="scaling_selection_sha256",
            name="published scaling selection",
            expected_relative=published_scaling_relative,
        ),
        "intent": _publication_record(
            tree,
            current,
            path_key="mpi_release_launch_intent",
            sha_key="mpi_release_launch_intent_sha256",
            name="published MPI release launch intent",
            expected_relative=f"{attempt_relative}/mpi-release-launch-intent.json",
        ),
        "launch": _publication_record(
            tree,
            current,
            path_key="mpi_release_launch",
            sha_key="mpi_release_launch_sha256",
            name="published MPI release launch",
            expected_relative=f"{attempt_relative}/mpi-release-launch.json",
        ),
        "stdout": _publication_record(
            tree,
            current,
            path_key="mpi_release_stdout",
            sha_key="mpi_release_stdout_sha256",
            name="published MPI release stdout",
            expected_relative=f"{attempt_relative}/mpi-release.stdout.log",
        ),
        "stderr": _publication_record(
            tree,
            current,
            path_key="mpi_release_stderr",
            sha_key="mpi_release_stderr_sha256",
            name="published MPI release stderr",
            expected_relative=f"{attempt_relative}/mpi-release.stderr.log",
        ),
        "runner": _publication_record(
            tree,
            current,
            path_key="mpi_runner_result",
            sha_key="mpi_runner_result_sha256",
            name="published MPI runner result",
            expected_relative=f"{attempt_relative}/mpi-release-runner-result.json",
        ),
        "execution": _publication_record(
            tree,
            current,
            path_key="mpi_execution_manifest",
            sha_key="mpi_execution_manifest_sha256",
            name="published MPI execution manifest",
            expected_relative=f"{attempt_relative}/mpi-execution-manifest.json",
        ),
        "campaign": _publication_record(
            tree,
            current,
            path_key="mpi_campaign_manifest",
            sha_key="mpi_campaign_manifest_sha256",
            name="published MPI campaign manifest",
            expected_relative=f"{attempt_relative}/mpi-campaign-manifest.json",
        ),
    }
    status = _strict_json(paths["status"], "attempt terminal status", style="indent-ascii")
    expected_status_keys = {
        "schema",
        "state",
        "commit",
        "attempt",
        "attempt_path",
        "started_utc",
        "finished_utc",
        "returncode",
        "identity_bundle",
        "report_facts",
        "scaling_facts",
        "mpi_execution_facts",
    }
    _require(
        set(status) == expected_status_keys
        and status.get("schema") == "negfpy.remote_validation.job_status.v2"
        and status.get("state") == "completed"
        and status.get("returncode") == 0
        and status.get("commit") == commit
        and status.get("attempt") == attempt
        and status.get("attempt_path") == attempt_relative
        and isinstance(status.get("started_utc"), str)
        and isinstance(status.get("finished_utc"), str)
        and status["started_utc"] <= status["finished_utc"],
        "selected attempt terminal status is not an exact successful status",
    )
    root_status_path = _tree_file(tree, "job-status.json", "root terminal status")
    root_status = _strict_json(root_status_path, "root terminal status", style="indent-ascii")
    _require(root_status == status, "root and selected-attempt terminal statuses differ")

    report_facts_path = _tree_file(
        tree, f"{attempt_relative}/report-facts.json", "report facts sidecar"
    )
    report_facts = _strict_json(report_facts_path, "report facts sidecar", style="indent-ascii")
    _require(
        report_facts == _mapping(current.get("report_facts"), "publication report facts")
        and report_facts == _mapping(status.get("report_facts"), "status report facts"),
        "publication, status, and sidecar report facts differ",
    )
    checkpoint_index_path = _tree_file(
        tree, f"{attempt_relative}/checkpoint-index.json", "checkpoint index"
    )
    checkpoint_index = _strict_json(
        checkpoint_index_path, "checkpoint index", style="indent-ascii"
    )
    records = checkpoint_index.get("records")
    _require(
        checkpoint_index.get("schema")
        == "negfpy.remote_validation.checkpoint_index.v1"
        and checkpoint_index.get("attempt") == attempt
        and checkpoint_index.get("count") == 13
        and isinstance(records, list)
        and len(records) == 13,
        "published checkpoint index is incomplete or mismatched",
    )
    return {
        "attempt": attempt,
        "attempt_relative": attempt_relative,
        "commit": commit,
        "current_path": current_path,
        "publication_path": publication_path,
        "publication": current,
        "paths": paths,
        "status": status,
        "report_facts": report_facts,
        "checkpoint_index": checkpoint_index,
    }


def _remote_absolute(value: Any, name: str) -> PurePosixPath:
    _require(isinstance(value, str) and value, f"{name} is not text")
    path = PurePosixPath(value)
    _require(
        path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and "\\" not in value,
        f"{name} is not a canonical absolute POSIX path",
    )
    return path


def _expected_binding(rank_count: int, physical_cores: int, socket_count: int) -> list[str]:
    _require(
        0 < rank_count <= physical_cores and socket_count > 0,
        "release launch topology is invalid",
    )
    binding = ["--nooversubscribe", "--bind-to", "core"]
    if rank_count >= socket_count and rank_count % socket_count == 0:
        binding.extend(("--map-by", f"ppr:{rank_count // socket_count}:socket:PE=1"))
    else:
        binding.extend(("--map-by", "core:PE=1"))
    binding.append("--report-bindings")
    return binding


def _embedded_manifest(
    value: Any,
    *,
    name: str,
    schema: str,
) -> Mapping[str, Any]:
    manifest = _mapping(value, name)
    digest = _sha(manifest.get("manifest_sha256"), f"{name} SHA256")
    body = {key: item for key, item in manifest.items() if key != "manifest_sha256"}
    _require(
        manifest.get("schema") == schema
        and hmac.compare_digest(_hash_document(body), digest),
        f"{name} schema or canonical SHA256 changed",
    )
    return manifest


def _source_record_map(
    value: Any,
    *,
    name: str,
    require_tracked: bool | None,
) -> dict[str, Mapping[str, Any]]:
    _require(isinstance(value, list), f"{name} is not a list")
    records: dict[str, Mapping[str, Any]] = {}
    ordered: list[str] = []
    for index, raw in enumerate(value):
        record = _mapping(raw, f"{name} record {index}")
        relative = _relative(record.get("path"), f"{name} record {index} path")
        path = relative.as_posix()
        kind = record.get("kind")
        tracked = record.get("git_tracked")
        _require(
            path not in records
            and isinstance(tracked, bool)
            and (require_tracked is None or tracked is require_tracked)
            and isinstance(record.get("mode"), int)
            and not isinstance(record.get("mode"), bool)
            and 0 <= record["mode"] <= 0o7777
            and kind in {"regular_file", "symlink"},
            f"{name} record {index} shape changed",
        )
        if kind == "regular_file":
            _require(
                set(record)
                == {"path", "git_tracked", "kind", "size", "mode", "sha256"}
                and isinstance(record.get("size"), int)
                and not isinstance(record.get("size"), bool)
                and record["size"] >= 0
                and SHA_RE.fullmatch(str(record.get("sha256"))) is not None,
                f"{name} regular-file record {path} changed",
            )
        else:
            _require(
                set(record)
                == {"path", "git_tracked", "kind", "target", "mode"}
                and isinstance(record.get("target"), str)
                and record["target"],
                f"{name} symlink record {path} changed",
            )
        records[path] = record
        ordered.append(path)
    _require(ordered == sorted(ordered), f"{name} records are not path-sorted")
    return records


def _remote_source_manifests(
    tree: ChecksumTree,
    status: Mapping[str, Any],
    *,
    commit: str,
    source_tree_sha256: str,
    environment_sha256: str,
    environment_prefix: PurePosixPath,
) -> dict[str, Any]:
    bundle = _mapping(status.get("identity_bundle"), "terminal identity bundle")
    expected_bundle_keys = {
        "schema",
        "commit",
        "git_tree",
        "source_tree_sha256",
        "explicit_lock",
        "explicit_lock_sha256",
        "remote_environment",
        "remote_environment_sha256",
        "verified_incoming_files",
        "archived_files",
    }
    git_tree = _commit(bundle.get("git_tree"), "identity Git tree")
    archived = _mapping(bundle.get("archived_files"), "identity archived files")
    _require(
        set(bundle) == expected_bundle_keys
        and bundle.get("schema")
        == "negfpy.remote_validation.identity_bundle.v1"
        and bundle.get("commit") == commit
        and bundle.get("source_tree_sha256") == source_tree_sha256
        and bundle.get("explicit_lock") == "environment-explicit.lock"
        and bundle.get("explicit_lock_sha256") == environment_sha256
        and bundle.get("remote_environment") == "remote-environment.json"
        and bundle.get("verified_incoming_files")
        == [
            "deployment-manifest.json",
            "environment-explicit.lock",
            "environment-manifest.json",
            "negfpy.bundle",
        ]
        and set(archived)
        == {
            "deployment-manifest.json",
            "environment-explicit.lock",
            "environment-manifest.json",
            "incoming-SHA256SUMS",
        }
        and all(SHA_RE.fullmatch(str(item)) is not None for item in archived.values())
        and archived.get("environment-explicit.lock") == environment_sha256,
        "terminal identity bundle shape or source/environment identity changed",
    )
    for name, digest in archived.items():
        _tree_file(
            tree,
            f"identity/{name}",
            f"archived identity file {name}",
            expected_sha256=_sha(digest, f"archived identity file {name} SHA256"),
        )
    identity_manifest_path = _tree_file(
        tree, "identity/identity-manifest.json", "identity manifest"
    )
    identity_manifest = _strict_json(
        identity_manifest_path, "identity manifest", style="indent-ascii"
    )
    _require(identity_manifest == bundle, "status and archived identity bundle differ")
    remote_path = _tree_file(
        tree,
        "identity/remote-environment.json",
        "archived remote environment",
        expected_sha256=_sha(
            bundle.get("remote_environment_sha256"),
            "remote environment SHA256",
        ),
    )
    remote = _strict_json(remote_path, "archived remote environment", style="indent-ascii")
    _require(
        remote.get("schema")
        == "negfpy.remote_validation.remote_environment.v1"
        and remote.get("commit") == commit
        and remote.get("tree") == git_tree
        and remote.get("source_tree_sha256") == source_tree_sha256
        and remote.get("explicit_lock_sha256") == environment_sha256
        and remote.get("thread_limits") == THREAD_LIMITS,
        "archived remote environment identity changed",
    )
    source = _embedded_manifest(
        remote.get("execution_source_identity"),
        name="execution source identity",
        schema="negfpy.remote_validation.execution_source.v1",
    )
    tracked = source.get("tracked_files")
    worktree = source.get("worktree_files")
    orchestration = source.get("orchestration_scripts")
    tracked_map = _source_record_map(
        tracked, name="tracked source files", require_tracked=True
    )
    worktree_map = _source_record_map(
        worktree, name="full worktree files", require_tracked=None
    )
    orchestration_map = _source_record_map(
        orchestration,
        name="orchestration source files",
        require_tracked=True,
    )
    _require(
        set(source)
        == {
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
        and source.get("git_revision") == commit
        and source.get("git_tree") == git_tree
        and source.get("package_source_sha256") == source_tree_sha256
        and source.get("tracked_file_count") == len(tracked_map)
        and source.get("worktree_file_count") == len(worktree_map)
        and all(worktree_map.get(path) == record for path, record in tracked_map.items())
        and list(orchestration_map) == list(ORCHESTRATION_SCRIPT_PATHS)
        and all(path in tracked_map for path in orchestration_map)
        and all(
            orchestration_map[path] == tracked_map[path] == worktree_map[path]
            for path in orchestration_map
        )
        and source.get("tracked_files_manifest_sha256")
        == _hash_document(
            {
                "schema": "negfpy.remote_validation.tracked_worktree.v1",
                "records": tracked,
            }
        )
        and source.get("worktree_files_manifest_sha256")
        == _hash_document(
            {
                "schema": "negfpy.remote_validation.full_worktree.v1",
                "records": worktree,
            }
        )
        and source.get("orchestration_manifest_sha256")
        == _hash_document(
            {
                "schema": "negfpy.remote_validation.orchestration_sources.v1",
                "records": orchestration,
            }
        ),
        "execution source manifests or orchestration source set changed",
    )
    prefix = _embedded_manifest(
        remote.get("environment_prefix_manifest"),
        name="environment prefix manifest",
        schema="negfpy.remote_validation.environment_prefix.v1",
    )
    regular = prefix.get("regular_files")
    symlinks = prefix.get("symlinks")
    _require(
        set(prefix)
        == {
            "schema",
            "prefix",
            "directory_count",
            "regular_file_count",
            "symlink_count",
            "regular_files",
            "symlinks",
            "manifest_sha256",
        }
        and isinstance(regular, list)
        and isinstance(symlinks, list)
        and prefix.get("prefix") == str(environment_prefix)
        and prefix.get("regular_file_count") == len(regular)
        and prefix.get("symlink_count") == len(symlinks)
        and isinstance(prefix.get("directory_count"), int)
        and prefix["directory_count"] > 0,
        "environment prefix manifest counts or root changed",
    )
    regular_by_path: dict[str, Mapping[str, Any]] = {}
    regular_order: list[str] = []
    for index, raw in enumerate(regular):
        record = _mapping(raw, f"environment regular file {index}")
        path = _relative(
            record.get("path"), f"environment regular file {index} path"
        ).as_posix()
        _require(
            set(record) == {"path", "size", "sha256"}
            and path not in regular_by_path
            and isinstance(record.get("size"), int)
            and not isinstance(record.get("size"), bool)
            and record["size"] >= 0
            and SHA_RE.fullmatch(str(record.get("sha256"))) is not None,
            f"environment regular-file record {path} changed",
        )
        regular_by_path[path] = record
        regular_order.append(path)
    symlink_order: list[str] = []
    for index, raw in enumerate(symlinks):
        record = _mapping(raw, f"environment symlink {index}")
        path = _relative(
            record.get("path"), f"environment symlink {index} path"
        ).as_posix()
        _require(
            set(record)
            == {"path", "target", "resolved_path", "target_kind", "link_size"}
            and isinstance(record.get("target"), str)
            and record["target"]
            and not PurePosixPath(record["target"]).is_absolute()
            and _relative(
                record.get("resolved_path"),
                f"environment symlink {index} resolved path",
            )
            and record.get("target_kind") in {"regular_file", "directory"}
            and isinstance(record.get("link_size"), int)
            and not isinstance(record.get("link_size"), bool)
            and record["link_size"] >= 0,
            f"environment symlink record {path} changed",
        )
        symlink_order.append(path)
    _require(
        regular_order == sorted(regular_order)
        and len(regular_order) == len(set(regular_order))
        and symlink_order == sorted(symlink_order)
        and len(symlink_order) == len(set(symlink_order))
        and not (set(regular_order) & set(symlink_order)),
        "environment prefix records are duplicated or not path-sorted",
    )
    return {
        "source": source,
        "source_records": worktree_map,
        "prefix": prefix,
        "prefix_regular": regular_by_path,
        "remote_environment_file_sha256": _file_sha256(remote_path),
        "remote_environment": remote,
        "identity_bundle": bundle,
        "git_tree": git_tree,
    }


def _validate_publication_chain(
    tree: ChecksumTree,
    context: Mapping[str, Any],
    *,
    scaling: Mapping[str, Any],
    scaling_record: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    publication = _mapping(context.get("publication"), "current publication")
    paths = _mapping(context.get("paths"), "publication paths")
    status = _mapping(context.get("status"), "publication terminal status")
    mpi_facts = _mapping(status.get("mpi_execution_facts"), "terminal MPI facts")
    execution_id = mpi_facts.get("execution_id")
    _require(
        isinstance(execution_id, str)
        and execution_id
        and execution_id == release["payload"].get("execution_id"),
        "terminal MPI execution identifier differs from the release runner",
    )

    launch_path = Path(paths["launch"])
    launch_document = _read_envelope(launch_path, "MPI release launch receipt")
    launch = _mapping(launch_document.get("payload"), "MPI release launch payload")
    expected_launch_keys = {
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
    _require(
        set(launch) == expected_launch_keys
        and launch.get("schema") == "negfpy.remote_validation.release_launch.v2",
        "MPI release launch v2 payload shape changed",
    )
    launch_binding = _mapping(launch.get("launch_binding"), "release launch binding")
    expected_binding_keys = {
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
    _require(
        set(launch_binding) == expected_binding_keys
        and launch_binding.get("schema")
        == "negfpy.remote_validation.release_launch_binding.v1",
        "MPI release launch-binding shape changed",
    )
    rank_count = int(scaling["toy_rank"])
    physical_cores = int(scaling["physical_core_count"])
    socket_count = int(scaling["socket_count"])
    binding = _expected_binding(rank_count, physical_cores, socket_count)
    launcher = _mapping(launch_binding.get("launcher"), "release launcher identity")
    python = _mapping(launch_binding.get("python"), "release Python identity")
    runner = _mapping(launch_binding.get("runner"), "release runner identity")
    _require(
        set(launcher) == {"path", "resolved_path", "sha256"}
        and set(python) == {"path", "resolved_path", "sha256"}
        and set(runner) == {"path", "resolved_path", "size", "sha256"},
        "MPI release executable/source record shape changed",
    )
    launcher_path = _remote_absolute(launcher.get("path"), "MPI release launcher path")
    launcher_target = _remote_absolute(
        launcher.get("resolved_path"), "MPI release launcher target"
    )
    python_path = _remote_absolute(python.get("path"), "MPI release Python path")
    python_target = _remote_absolute(python.get("resolved_path"), "MPI release Python target")
    prefix = launcher_path.parent.parent
    deployment_root = prefix.parent
    remote_runner = deployment_root / "worktree" / "scripts" / "remote_toy_mpi_runner.py"
    runner_path = _remote_absolute(runner.get("path"), "MPI release runner path")
    runner_target = _remote_absolute(
        runner.get("resolved_path"), "MPI release runner target"
    )
    _require(
        launcher_path == prefix / "bin" / "mpiexec"
        and launcher_target.is_relative_to(prefix)
        and python_path == prefix / "bin" / "python"
        and python_target.is_relative_to(prefix)
        and runner_path == runner_target == remote_runner
        and deployment_root.name == context["commit"]
        and isinstance(runner.get("size"), int)
        and not isinstance(runner.get("size"), bool)
        and runner["size"] > 0,
        "MPI release executables/runner are not from the exact deployment",
    )
    launcher_sha = _sha(launcher.get("sha256"), "MPI release launcher SHA256")
    python_sha = _sha(python.get("sha256"), "MPI release Python SHA256")
    runner_sha = _sha(runner.get("sha256"), "MPI release runner SHA256")
    remote_release_root = deployment_root / "artifacts" / "toy-release"
    program = [
        str(python_path),
        str(remote_runner),
        "release",
        "--run-directory",
        str(remote_release_root),
        "--source-revision",
        str(context["commit"]),
        "--source-tree-sha256",
        str(scaling["tree_sha"]),
        "--environment-sha256",
        str(scaling["environment_sha"]),
        "--execution-id",
        execution_id,
        "--expected-ranks",
        str(rank_count),
    ]
    expected_argv = [str(launcher_path), *binding, "-n", str(rank_count), *program]
    attempt_remote = (
        deployment_root
        / "artifacts"
        / "attempts"
        / ("attempt-%03d" % int(context["attempt"]))
    )
    expected_environment = {
        "BLIS_NUM_THREADS": "1",
        "HOME": f"/home/{REMOTE_EXPECTED_USER}",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "LOGNAME": REMOTE_EXPECTED_USER,
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": str(attempt_remote / "runtime-config" / "matplotlib"),
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": os.pathsep.join(
            (str(prefix / "bin"), REMOTE_TOOLS_BIN, "/usr/bin", "/bin")
        ),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(deployment_root / "worktree" / "src"),
        "PYTHONPYCACHEPREFIX": str(attempt_remote / "runtime-cache" / "pycache"),
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(attempt_remote / "runtime-tmp"),
        "TZ": "UTC",
        "USER": REMOTE_EXPECTED_USER,
        "VECLIB_MAXIMUM_THREADS": "1",
        "XDG_CACHE_HOME": str(attempt_remote / "runtime-cache"),
    }
    launch_environment = _mapping(
        launch_binding.get("launch_environment"), "scientific launch environment"
    )
    _require(
        launch_binding.get("source_revision") == context["commit"]
        and launch_binding.get("source_tree_sha256") == scaling["tree_sha"]
        and launch_binding.get("environment_sha256") == scaling["environment_sha"]
        and launch_binding.get("execution_id") == execution_id
        and launch_binding.get("selected_rank_count") == rank_count
        and launch_binding.get("thread_limits") == THREAD_LIMITS
        and launch_binding.get("binding_arguments") == binding
        and launch_binding.get("argv") == expected_argv
        and launch_environment == expected_environment
        and launch_binding.get("launch_environment_sha256")
        == _hash_document(expected_environment),
        "MPI release launch identity, argv, binding, or scrubbed environment changed",
    )

    identities = _remote_source_manifests(
        tree,
        status,
        commit=str(context["commit"]),
        source_tree_sha256=str(scaling["tree_sha"]),
        environment_sha256=str(scaling["environment_sha"]),
        environment_prefix=prefix,
    )
    source = identities["source"]
    prefix_manifest = identities["prefix"]
    source_records = identities["source_records"]
    runner_record = source_records.get("scripts/remote_toy_mpi_runner.py")
    _require(
        launch_binding.get("execution_source_identity_sha256")
        == source.get("manifest_sha256")
        and launch_binding.get("tracked_files_manifest_sha256")
        == source.get("tracked_files_manifest_sha256")
        and launch_binding.get("worktree_files_manifest_sha256")
        == source.get("worktree_files_manifest_sha256")
        and launch_binding.get("orchestration_manifest_sha256")
        == source.get("orchestration_manifest_sha256")
        and launch_binding.get("orchestration_scripts")
        == source.get("orchestration_scripts")
        and launch_binding.get("environment_prefix_manifest_sha256")
        == prefix_manifest.get("manifest_sha256")
        and launch_binding.get("environment_regular_file_count")
        == prefix_manifest.get("regular_file_count")
        and launch_binding.get("environment_symlink_count")
        == prefix_manifest.get("symlink_count")
        and isinstance(runner_record, Mapping)
        and runner_record.get("kind") == "regular_file"
        and runner_record.get("git_tracked") is True
        and runner_record.get("size") == runner.get("size")
        and runner_record.get("sha256") == runner_sha,
        "launch binding differs from archived source/environment manifests",
    )
    prefix_regular = identities["prefix_regular"]
    for name, target, digest in (
        ("launcher", launcher_target, launcher_sha),
        ("Python", python_target, python_sha),
    ):
        relative_target = target.relative_to(prefix).as_posix()
        prefix_record = prefix_regular.get(relative_target)
        _require(
            isinstance(prefix_record, Mapping)
            and prefix_record.get("sha256") == digest,
            f"{name} bytes differ from the environment-prefix manifest",
        )

    intent_path = Path(paths["intent"])
    intent_document = _read_envelope(intent_path, "MPI release launch intent")
    intent = _mapping(intent_document.get("payload"), "MPI release launch intent payload")
    _require(
        set(intent)
        == {
            "schema",
            "launch_binding",
            "stdout_relative_path",
            "stderr_relative_path",
        }
        and intent.get("schema")
        == "negfpy.remote_validation.release_launch_intent.v1"
        and intent.get("launch_binding") == launch_binding
        and intent.get("stdout_relative_path") == "mpi-release.stdout.log"
        and intent.get("stderr_relative_path") == "mpi-release.stderr.log",
        "immutable MPI launch intent differs from terminal launch binding",
    )
    intent_record = _mapping(launch.get("intent"), "terminal launch intent record")
    _require(
        set(intent_record) == {"relative_path", "sha256"}
        and intent_record.get("relative_path") == "mpi-release-launch-intent.json"
        and intent_record.get("sha256") == _file_sha256(intent_path)
        and intent_record.get("sha256")
        == publication.get("mpi_release_launch_intent_sha256")
        and launch.get("intent_payload_sha256")
        == intent_document["payload_sha256"],
        "terminal launch receipt is not bound to the immutable intent",
    )
    post_run = _mapping(
        launch.get("post_run_verification"),
        "post-run source/environment verification",
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
        and post_run.get("execution_source_identity_sha256")
        == source.get("manifest_sha256")
        and post_run.get("environment_prefix_manifest_sha256")
        == prefix_manifest.get("manifest_sha256")
        and launch.get("returncode") == 0
        and isinstance(launch.get("started_utc"), str)
        and isinstance(launch.get("finished_utc"), str)
        and launch["started_utc"] <= post_run["checked_utc"] <= launch["finished_utc"],
        "post-run source/environment verification is incomplete or failed",
    )
    for key, publication_key, expected_name in (
        ("stdout", "mpi_release_stdout_sha256", "mpi-release.stdout.log"),
        ("stderr", "mpi_release_stderr_sha256", "mpi-release.stderr.log"),
    ):
        record = _mapping(launch.get(key), f"release {key} record")
        _require(
            set(record) == {"relative_path", "sha256"}
            and record.get("relative_path") == expected_name
            and record.get("sha256") == publication.get(publication_key)
            and record.get("sha256") == _file_sha256(Path(paths[key])),
            f"MPI release {key} receipt/publication binding changed",
        )
    launch_facts = {
        "schema": "negfpy.remote_validation.release_launch_facts.v1",
        "receipt_relative_path": "mpi-release-launch.json",
        "receipt_file_sha256": _file_sha256(launch_path),
        "receipt_payload_sha256": launch_document["payload_sha256"],
        "intent_relative_path": "mpi-release-launch-intent.json",
        "intent_file_sha256": _file_sha256(intent_path),
        "intent_payload_sha256": intent_document["payload_sha256"],
        "launcher_file_sha256": launcher_sha,
        "python_file_sha256": python_sha,
        "runner_file_sha256": runner_sha,
        "execution_source_identity_sha256": source["manifest_sha256"],
        "tracked_files_manifest_sha256": source["tracked_files_manifest_sha256"],
        "worktree_files_manifest_sha256": source["worktree_files_manifest_sha256"],
        "orchestration_manifest_sha256": source["orchestration_manifest_sha256"],
        "environment_prefix_manifest_sha256": prefix_manifest["manifest_sha256"],
        "launch_environment_sha256": launch_binding["launch_environment_sha256"],
        "stdout_relative_path": "mpi-release.stdout.log",
        "stdout_file_sha256": _file_sha256(Path(paths["stdout"])),
        "stderr_relative_path": "mpi-release.stderr.log",
        "stderr_file_sha256": _file_sha256(Path(paths["stderr"])),
        "selected_rank_count": rank_count,
        "binding_arguments": binding,
        "post_run_verified": True,
        "returncode": 0,
    }
    _require(
        _mapping(mpi_facts.get("launch_receipt"), "terminal launch facts")
        == launch_facts,
        "terminal MPI facts differ from the independently rederived launch receipt",
    )

    canonical_runner = _tree_file(
        tree,
        f"toy-release/runner-results/{execution_id}.json",
        "canonical release runner result",
        expected_sha256=release["file_sha256"],
    )
    _require(
        Path(paths["runner"]).read_bytes() == canonical_runner.read_bytes()
        and _file_sha256(Path(paths["runner"])) == release["file_sha256"]
        and _file_sha256(Path(paths["campaign"]))
        == release["campaign_file_sha256"]
        and _file_sha256(Path(paths["execution"]))
        == release["execution_file_sha256"]
        and _file_sha256(Path(paths["report"])) == release["report_file_sha256"]
        and _file_sha256(Path(release["report_path"]))
        == release["report_file_sha256"],
        "published runner/campaign/execution/report copies differ from reopened release evidence",
    )
    runtime = release["payload"].get("rank_runtime")
    _require(
        isinstance(runtime, list) and len(runtime) == rank_count,
        "release rank runtime is incomplete",
    )
    for record_value in runtime:
        record = _mapping(record_value, "release rank runtime")
        runtime_python = _mapping(record.get("python"), "release rank Python")
        _require(
            runtime_python.get("sha256") == python_sha
            and _remote_absolute(runtime_python.get("path"), "rank Python path")
            == python_target
            and record.get("python_argv") == program[1:],
            "release rank Python identity or argv differs from the outer launch",
        )
    expected_mpi = {
        "schema": "negfpy.remote_validation.mpi_execution_facts.v1",
        "runner_result_file_sha256": release["file_sha256"],
        "runner_result_payload_sha256": release["document"]["payload_sha256"],
        "campaign_manifest_file_sha256": release["campaign_file_sha256"],
        "campaign_manifest_payload_sha256": release["payload"]["campaign_manifest_sha256"],
        "execution_manifest_file_sha256": release["execution_file_sha256"],
        "execution_manifest_payload_sha256": release["execution_payload_sha256"],
        "execution_scientific_sha256": release["payload"]["execution_scientific_sha256"],
        "production_toy_physics_sha256": release["physics_sha256"],
        "execution_id": execution_id,
        "rank_runtime": runtime,
        "python_file_sha256": python_sha,
        "harmonic_preflight": mpi_facts.get("harmonic_preflight"),
        "memory_preflight": release["payload"].get("memory_preflight"),
        "phase_evidence": mpi_facts.get("phase_evidence"),
        "communicator_size": rank_count,
        "mode": "serial" if rank_count == 1 else "mpi",
        "directory_mode": release["payload"].get("directory_mode"),
        "task_count": 28,
        "release_certified": True,
        "launch_receipt": launch_facts,
    }
    harmonic = _mapping(expected_mpi["harmonic_preflight"], "terminal harmonic preflight")
    phase_evidence = expected_mpi["phase_evidence"]
    _require(
        harmonic.get("schema")
        == "negfpy.remote_toy_mpi.real_harmonic_preflight.v1"
        and harmonic.get("point_count") == 7
        and harmonic.get("real_physics_executed") is True
        and harmonic.get("passed") is True
        and _sha(harmonic.get("artifact_file_sha256"), "harmonic artifact SHA256")
        and _sha(harmonic.get("evidence_sha256"), "harmonic evidence SHA256")
        and isinstance(phase_evidence, list)
        and len(phase_evidence) == 8,
        "terminal harmonic or phase execution evidence is incomplete",
    )
    assert isinstance(phase_evidence, list)
    _require(
        all(
            isinstance(item, Mapping)
            and set(item)
            == {
                "wave_index",
                "task_manifest_file_sha256",
                "execution_manifest_file_sha256",
                "execution_payload_sha256",
                "task_count",
            }
            and item.get("wave_index") == index
            and isinstance(item.get("task_count"), int)
            and not isinstance(item.get("task_count"), bool)
            and item["task_count"] > 0
            and all(
                SHA_RE.fullmatch(str(item.get(key))) is not None
                for key in (
                    "task_manifest_file_sha256",
                    "execution_manifest_file_sha256",
                    "execution_payload_sha256",
                )
            )
            for index, item in enumerate(phase_evidence)
        )
        and sum(item["task_count"] for item in phase_evidence) == 28,
        "terminal harmonic or phase execution evidence is incomplete",
    )
    _require(mpi_facts == expected_mpi, "terminal MPI execution facts changed")

    terminal_scaling = _mapping(
        status.get("scaling_facts"), "terminal scaling facts"
    )
    reused = terminal_scaling.get("reused_for_exact_retry")
    source_attempt = terminal_scaling.get("source_attempt")
    published_scaling = str(publication.get("scaling_selection"))
    selected_attempt = published_scaling.split("/", 2)[1]
    _require(
        isinstance(reused, bool)
        and isinstance(source_attempt, str)
        and re.fullmatch(r"attempt-[0-9]{3}", source_attempt) is not None
        and source_attempt == selected_attempt
        and (reused or source_attempt == "attempt-%03d" % int(context["attempt"])),
        "terminal scaling retry/source-attempt facts are inconsistent",
    )
    scaling_facts = {
        "schema": "negfpy.remote_validation.scaling_facts.v1",
        "selection_relative_path": published_scaling,
        "selection_file_sha256": scaling_record["file_sha256"],
        "selection_payload_sha256": scaling_record["payload_sha256"],
        "production_toy_physics_sha256": scaling["scaling_physics_sha"],
        "physics_equal_across_all_ranks": True,
        "toy_rank_count": rank_count,
        "toy_mpi_enabled": scaling["mpi_enabled"],
        "silicon_rank_count": scaling["silicon_rank"],
        "candidate_speedup": scaling["candidate_speedup"],
        "rank_ladder": list(RANK_LADDER),
        "rank_evidence": scaling["rank_evidence"],
        "reused_for_exact_retry": reused,
        "source_attempt": source_attempt,
    }
    _require(
        terminal_scaling == scaling_facts,
        "terminal scaling facts differ from independently reopened rank evidence",
    )

    report_facts = _mapping(context.get("report_facts"), "published report facts")
    expected_report_keys = {
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
        "mpi_execution",
        "mpi_scaling",
    }
    _require(
        set(report_facts) == expected_report_keys
        and report_facts.get("schema")
        == "negfpy.remote_validation.report_facts.v1"
        and report_facts.get("report_relative_path") == publication.get("report")
        and report_facts.get("report_file_sha256") == release["report_file_sha256"]
        and report_facts.get("report_body_sha256") == release["report_body_sha256"]
        and report_facts.get("schema_version") == 4
        and report_facts.get("protocol") == "multidimensional_toy_gate_v2"
        and report_facts.get("protocol_version") == 2
        and report_facts.get("release_certified") is True
        and report_facts.get("multidimensional_outcome_count") == 13
        and report_facts.get("multidimensional_backend") == "fft"
        and report_facts.get("harmonic_preflight_point_count") == 7
        and report_facts.get("bounded_backends") == ["serial", "threaded", "fft"]
        and report_facts.get("open_system_run_count") == 13
        and report_facts.get("checkpoint_count") == 13
        and report_facts.get("commit") == context["commit"]
        and report_facts.get("git_tree")
        == _mapping(status.get("identity_bundle"), "terminal identity bundle").get("git_tree")
        and report_facts.get("source_tree_sha256") == scaling["tree_sha"]
        and report_facts.get("environment_sha256") == scaling["environment_sha"]
        and report_facts.get("mpi_execution") == expected_mpi
        and report_facts.get("mpi_scaling") == scaling_facts,
        "published report facts are not fully derived from the selected release chain",
    )
    return {
        "publication_file_sha256": _file_sha256(Path(context["publication_path"])),
        "current_file_sha256": _file_sha256(Path(context["current_path"])),
        "intent_file_sha256": _file_sha256(intent_path),
        "intent_payload_sha256": intent_document["payload_sha256"],
        "launch_file_sha256": _file_sha256(launch_path),
        "launch_payload_sha256": launch_document["payload_sha256"],
        "execution_source_identity_sha256": source["manifest_sha256"],
        "environment_prefix_manifest_sha256": prefix_manifest["manifest_sha256"],
        "launch_environment_sha256": launch_binding["launch_environment_sha256"],
        "source_records": source_records,
        "remote_environment": identities["remote_environment"],
        "identity_bundle": identities["identity_bundle"],
        "git_tree": identities["git_tree"],
    }


def _validate_staging(
    tree: ChecksumTree,
    *,
    commit: str,
    git_tree: str,
    tree_sha: str,
    environment_sha: str,
    artifact_anchor: str,
    artifact_fetch_receipt_sha: str,
    fetch_receipt: Mapping[str, Any],
    release_report_sha: str,
    release_report_body_sha: str,
    nested_report_body_sha: str,
    identity_bundle: Mapping[str, Any],
    remote_environment: Mapping[str, Any],
    source_records: Mapping[str, Any],
) -> str:
    importer, importer_sha = _load_committed_module(
        IMPORTER_SOURCE, source_records, name="production-importer"
    )
    _require(
        callable(getattr(importer, "_derive_files", None))
        and isinstance(getattr(importer, "_IMPORTER_VERSION", None), str),
        "committed production importer lacks the canonical derivation contract",
    )
    for filename in REQUIRED_TEX:
        _tree_file(tree, f"tex/{filename}", f"staging TeX {filename}")
    staged_report = _tree_file(
        tree, "production_toy_gate.json", "staged strict report",
        expected_sha256=release_report_sha,
    )
    document = _strict_json(
        staged_report, "staged strict production report", style="indent-utf8"
    )
    nested = _mapping(
        document.get("multidimensional_scba"),
        "staged strict report multidimensional block",
    )
    _require(
        document.get("report_sha256") == release_report_body_sha
        and nested.get("report_sha256") == nested_report_body_sha,
        "staged strict report body hashes differ from the validated remote report",
    )

    receipt_path = _tree_file(tree, "import_receipt.json", "staging import receipt")
    receipt = _strict_json(receipt_path, "staging import receipt", style="indent-ascii")
    _require(
        set(receipt) == {
            "schema", "schema_version", "implementation", "source",
            "trusted_inputs", "release_contract", "checkpoint_validation",
            "staged_artifacts", "receipt_sha256",
        }
        and receipt.get("schema")
        == "negfpy.anharmonic.production_toy_gate_import_receipt"
        and receipt.get("schema_version") == 2,
        "staging import receipt shape or schema changed",
    )
    recorded_body = _sha(receipt.get("receipt_sha256"), "import receipt body SHA256")
    receipt_core = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    _require(
        hmac.compare_digest(recorded_body, _hash_document(receipt_core)),
        "staging import receipt body SHA256 mismatch",
    )
    implementation = _mapping(receipt.get("implementation"), "staging importer identity")
    _require(
        implementation == {
            "name": Path(IMPORTER_SOURCE).name,
            "version": importer._IMPORTER_VERSION,
            "source_sha256": importer_sha,
        },
        "staging receipt is not bound to the exact committed canonical importer",
    )

    source = _mapping(receipt.get("source"), "staging receipt source")
    attempt = source.get("attempt")
    expected_created = fetch_receipt.get("fetched_utc") or fetch_receipt.get("created_utc")
    _require(
        set(source) == {
            "commit", "attempt", "report_relative_path",
            "remote_report_relative_path", "report_file_sha256",
            "report_body_sha256", "nested_report_body_sha256",
            "outer_sha256sums_file_sha256", "fetch_receipt_sha256",
            "sha256sums_sha256", "transport_receipt_schema",
            "transport_receipt_created_utc", "fetched_utc", "git_tree",
            "source_tree_sha256", "environment_sha256",
        }
        and source.get("commit") == commit
        and isinstance(attempt, int) and not isinstance(attempt, bool) and attempt > 0
        and source.get("report_relative_path") == "production_toy_gate.json"
        and source.get("remote_report_relative_path")
        == f"attempts/attempt-{attempt:03d}/production_toy_gate.json"
        and source.get("report_file_sha256") == release_report_sha
        and source.get("report_body_sha256") == release_report_body_sha
        and source.get("nested_report_body_sha256") == nested_report_body_sha
        and source.get("outer_sha256sums_file_sha256") == artifact_anchor
        and source.get("fetch_receipt_sha256") == artifact_fetch_receipt_sha
        and source.get("sha256sums_sha256") == artifact_anchor
        and source.get("transport_receipt_schema") == fetch_receipt.get("schema")
        and source.get("transport_receipt_created_utc") == expected_created
        and source.get("fetched_utc") == fetch_receipt.get("fetched_utc")
        and source.get("git_tree") == git_tree
        and source.get("source_tree_sha256") == tree_sha
        and source.get("environment_sha256") == environment_sha,
        "staging import receipt source identity changed",
    )
    trusted = _mapping(receipt.get("trusted_inputs"), "staging trusted inputs")
    _require(
        trusted == {
            "expected_commit": commit,
            "expected_environment_lock_sha256": environment_sha,
            "expected_fetch_receipt_sha256": artifact_fetch_receipt_sha,
            "expected_sha256sums_sha256": artifact_anchor,
        },
        "staging trusted-input shape or values changed",
    )
    contract = _mapping(receipt.get("release_contract"), "staging release contract")
    _require(
        contract == {
            "checkpoint_count": 13,
            "multidimensional_outcome_count": 13,
            "open_system_run_count": 13,
            "protocol": "multidimensional_toy_gate_v2",
            "protocol_version": 2,
            "release_certified": True,
            "schema_version": 4,
            "validation_tier": "release",
        },
        "staging release contract changed",
    )

    checkpoint_validation = _mapping(
        receipt.get("checkpoint_validation"), "staging checkpoint validation"
    )
    records = checkpoint_validation.get("records")
    outcomes = nested.get("outcomes")
    _require(
        set(checkpoint_validation) == {"mode", "count", "records"}
        and checkpoint_validation.get("mode")
        == "identity_and_integrity_load_without_scba_restore"
        and checkpoint_validation.get("count") == 13
        and isinstance(records, list) and len(records) == 13
        and isinstance(outcomes, list) and len(outcomes) == 13,
        "staging checkpoint-validation shape or count changed",
    )
    checkpoint_keys = {
        "run_id", "artifact_relative_path", "file_sha256", "identity_sha256",
        "status", "shape", "history_count", "sigma_pp_lesser_sha256",
        "sigma_pp_greater_sha256",
    }
    for index, (record_value, outcome_value) in enumerate(zip(records, outcomes)):
        record = _mapping(record_value, f"staging checkpoint record {index}")
        outcome = _mapping(outcome_value, f"staged report outcome {index}")
        result = _mapping(outcome.get("result"), f"staged report outcome {index} result")
        point = _mapping(outcome.get("point"), f"staged report outcome {index} point")
        numerical = _mapping(point.get("numerical"), f"staged report outcome {index} numerical")
        field_hashes = _mapping(
            result.get("field_sha256"), f"staged report outcome {index} field hashes"
        )
        artifact_relative = _relative(
            record.get("artifact_relative_path"),
            f"staging checkpoint record {index} artifact path",
        )
        _require(
            set(record) == checkpoint_keys
            and artifact_relative.as_posix() == record.get("artifact_relative_path")
            and record.get("run_id") == outcome.get("run_id")
            and SHA_RE.fullmatch(str(record.get("file_sha256"))) is not None
            and SHA_RE.fullmatch(str(record.get("identity_sha256"))) is not None
            and record.get("status") == outcome.get("status") == "converged"
            and record.get("shape") == [int(numerical.get("n_frequency", -1)), 9, 9]
            and record.get("history_count") == result.get("iterations")
            and record.get("sigma_pp_lesser_sha256")
            == field_hashes.get("sigma_pp_lesser")
            and record.get("sigma_pp_greater_sha256")
            == field_hashes.get("sigma_pp_greater")
            and SHA_RE.fullmatch(str(record.get("sigma_pp_lesser_sha256"))) is not None
            and SHA_RE.fullmatch(str(record.get("sigma_pp_greater_sha256"))) is not None,
            f"staging checkpoint record {index} shape or report binding changed",
        )

    _require(
        identity_bundle.get("commit") == commit
        and identity_bundle.get("git_tree") == git_tree
        and identity_bundle.get("source_tree_sha256") == tree_sha
        and identity_bundle.get("explicit_lock_sha256") == environment_sha,
        "validated identity bundle differs from staging source identity",
    )
    try:
        derived_raw = importer._derive_files(
            document,
            identity_bundle=identity_bundle,
            remote_environment=remote_environment,
            report_file_sha256=release_report_sha,
            report_body_sha256=release_report_body_sha,
            nested_body_sha256=nested_report_body_sha,
        )
    except Exception as error:
        raise ReleaseConfigError(
            "committed importer could not rederive staged report artifacts"
        ) from error
    _require(isinstance(derived_raw, Mapping), "importer derivation did not return a mapping")
    derived: dict[str, bytes] = {}
    for relative, payload in derived_raw.items():
        _require(
            isinstance(relative, str) and isinstance(payload, bytes)
            and _relative(relative, "derived staging path").as_posix() == relative,
            "importer derivation returned an invalid path or payload",
        )
        derived[relative] = payload
    _require(
        {f"tex/{filename}" for filename in REQUIRED_TEX}.issubset(derived),
        "committed importer did not derive every required TeX artifact",
    )

    staged = _mapping(receipt.get("staged_artifacts"), "staged artifact records")
    expected_staged = set(tree.entries) - {"import_receipt.json"}
    _require(set(staged) == expected_staged, "staging receipt coverage changed")
    for relative_text, record_value in staged.items():
        record = _mapping(record_value, f"staged artifact {relative_text}")
        path = _tree_file(tree, relative_text, f"staged artifact {relative_text}")
        _require(
            set(record) == {"sha256", "bytes"}
            and record.get("sha256") == tree.entries[relative_text]
            and record.get("sha256") == _file_sha256(path)
            and record.get("bytes") == path.stat().st_size,
            f"staging receipt hash/size record changed for {relative_text}",
        )
    _require(set(derived).issubset(staged), "derived staging artifacts lack receipt coverage")
    for relative, expected in derived.items():
        observed = (tree.root / relative).read_bytes()
        expected_sha = hashlib.sha256(expected).hexdigest()
        _require(
            observed == expected
            and _mapping(staged[relative], f"derived staged artifact {relative}")
            == {"sha256": expected_sha, "bytes": len(expected)},
            f"staged {relative} is not the canonical importer derivative",
        )
    return _file_sha256(receipt_path)


def _validate_production_figures(
    tree: ChecksumTree,
    *,
    staging_tree: ChecksumTree,
    source_records: Mapping[str, Any],
    commit: str,
    staging_anchor: str,
    staging_receipt_sha: str,
    release_report_sha: str,
) -> str:
    renderer, renderer_sha = _load_committed_module(
        PRODUCTION_RENDERER_SOURCE,
        source_records,
        name="production-renderer",
    )
    required_hooks = (
        "_validate_staging",
        "_validate_raw_report_binding",
        "_load_data",
        "_verify_output",
        "_resolve_font",
    )
    _require(
        all(callable(getattr(renderer, name, None)) for name in required_hooks)
        and isinstance(getattr(renderer, "_RENDERER_VERSION", None), str),
        "committed production renderer lacks idempotent validation hooks",
    )
    for filename in PRODUCTION_FIGURES:
        _tree_file(tree, filename, f"production figure {filename}")
    receipt_path = _tree_file(tree, "render_manifest.json", "figure render manifest")
    receipt = _strict_json(receipt_path, "figure render manifest", style="indent-ascii")
    _require(
        receipt.get("renderer_sha256") == renderer_sha
        and receipt.get("renderer_version") == renderer._RENDERER_VERSION,
        "production figure manifest is not bound to the committed renderer",
    )
    try:
        staging_base = renderer._validate_staging(
            staging_tree.root,
            expected_import_receipt_sha256=staging_receipt_sha,
        )
        report_binding = renderer._validate_raw_report_binding(
            staging_tree.root, staging_base
        )
        renderer_data = renderer._load_data(staging_tree.root)
        renderer_staging = MappingProxyType(
            {**dict(staging_base), "report_binding": report_binding}
        )
        style = _mapping(receipt.get("style"), "production figure style")
        font = _mapping(style.get("font"), "production figure font identity")
        resolved_font = renderer._resolve_font(
            Path(str(font.get("regular_path"))),
            Path(str(font.get("bold_path"))),
        )
        _require(
            dict(resolved_font) == dict(font),
            "production recorded font bytes differ from exact rerender fonts",
        )
        verified = renderer._verify_output(
            tree.root,
            staging=renderer_staging,
            font=resolved_font,
            data=renderer_data,
        )
    except Exception as error:
        raise ReleaseConfigError(
            "committed production renderer rejected staged data or figure structure"
        ) from error
    _require(
        isinstance(verified, Mapping)
        and verified == receipt
        and _mapping(receipt.get("source"), "figure receipt source").get("commit")
        == commit
        and receipt["source"].get("staging_sha256sums_file_sha256")
        == staging_anchor
        and receipt["source"].get("import_receipt_file_sha256")
        == staging_receipt_sha
        and receipt["source"].get("report_file_sha256") == release_report_sha,
        "production renderer validation did not reproduce the exact manifest",
    )
    return _file_sha256(receipt_path)


def _validate_historical_figures(
    tree: ChecksumTree,
    *,
    source_records: Mapping[str, Any],
) -> tuple[str, Mapping[str, float]]:
    renderer, renderer_sha = _load_committed_module(
        HISTORICAL_RENDERER_SOURCE,
        source_records,
        name="historical-renderer",
    )
    required_hooks = (
        "load_historical_inputs",
        "_reported_metrics",
        "_validate_pdf",
        "_validate_png",
        "_font",
        "_render_figure",
    )
    _require(
        all(callable(getattr(renderer, name, None)) for name in required_hooks)
        and isinstance(getattr(renderer, "_RENDERER_VERSION", None), str),
        "committed historical renderer lacks semantic validation hooks",
    )
    expected_tree = {
        *HISTORICAL_FIGURES,
        HISTORICAL_METRICS,
        HISTORICAL_RENDER_MANIFEST,
    }
    _require(set(tree.entries) == expected_tree, "historical output file set changed")
    for filename in HISTORICAL_FIGURES:
        _tree_file(tree, filename, f"historical figure {filename}")
    metrics_path = _tree_file(tree, HISTORICAL_METRICS, "historical metrics manifest")
    receipt_path = _tree_file(
        tree, HISTORICAL_RENDER_MANIFEST, "historical figure receipt"
    )

    benchmark_relative = f"{HISTORICAL_DATA_ROOT}/benchmark_manifest.json"
    benchmark_path, benchmark_sha = _exact_local_source_file(
        benchmark_relative,
        source_records,
        name="historical benchmark manifest",
    )
    _strict_json(
        benchmark_path, "historical benchmark manifest", style="indent-ascii"
    )
    data_root = benchmark_path.parent
    selected_names = tuple(renderer._INPUT_SCHEMAS)
    for filename in selected_names:
        _exact_local_source_file(
            f"{HISTORICAL_DATA_ROOT}/{filename}",
            source_records,
            name=f"historical source {filename}",
        )
    try:
        data, input_hashes, source = renderer.load_historical_inputs(
            data_dir=data_root,
            benchmark_manifest=benchmark_path,
            expected_benchmark_manifest_sha256=benchmark_sha,
        )
        reported_metrics = renderer._reported_metrics(data)
    except Exception as error:
        raise ReleaseConfigError(
            "committed historical renderer rejected benchmark inputs"
        ) from error
    _require(
        isinstance(reported_metrics, Mapping)
        and reported_metrics
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) >= 0.0
            for value in reported_metrics.values()
        ),
        "historical reported metrics are missing, negative, or non-finite",
    )
    renderer_identity = {
        "basename": Path(HISTORICAL_RENDERER_SOURCE).name,
        "version": renderer._RENDERER_VERSION,
        "sha256": renderer_sha,
    }
    expected_metrics_core = {
        "schema": renderer._METRICS_SCHEMA,
        "status": renderer._STATUS,
        "release_certified": False,
        "production_evidence": False,
        "source": source,
        "input_sha256": dict(sorted(input_hashes.items())),
        "metrics_recomputed_from_csv": data["metrics"],
        "reported_metrics": dict(reported_metrics),
        "renderer": renderer_identity,
    }
    metrics_document = _strict_json(
        metrics_path, "historical metrics manifest", style="compact-ascii"
    )
    _require(
        metrics_document
        == {
            **expected_metrics_core,
            "manifest_sha256": _hash_document(expected_metrics_core),
        },
        "historical metrics manifest is not the canonical recomputed document",
    )

    receipt = _strict_json(
        receipt_path, "historical figure receipt", style="compact-ascii"
    )
    pdf_name, png_name = HISTORICAL_FIGURES
    try:
        pdf_validation = renderer._validate_pdf(tree.root / pdf_name)
        png_validation = renderer._validate_png(tree.root / png_name)
    except Exception as error:
        raise ReleaseConfigError(
            "committed historical renderer rejected PDF/font/vector or PNG structure"
        ) from error
    fonts = _mapping(receipt.get("fonts"), "historical figure fonts")
    regular = _mapping(fonts.get("regular"), "historical regular font")
    bold = _mapping(fonts.get("bold"), "historical bold font")
    for role, font in (("regular", regular), ("bold", bold)):
        font_path = Path(str(font.get("path")))
        _require(
            set(font) == {"path", "basename", "sha256", "family", "style"}
            and font_path.is_absolute()
            and isinstance(font.get("basename"), str)
            and font["basename"].endswith(".ttf")
            and font_path.name == font["basename"]
            and SHA_RE.fullmatch(str(font.get("sha256"))) is not None
            and isinstance(font.get("family"), str)
            and font["family"]
            and isinstance(font.get("style"), str)
            and font["style"],
            f"historical {role} font identity changed",
        )
    _require(
        regular["family"] == bold["family"]
        and "sans" in regular["family"].lower()
        and "serif" not in regular["family"].lower()
        and "bold" not in regular["style"].lower()
        and "bold" in bold["style"].lower()
        and fonts.get("policy")
        == "caller-selected TrueType sans fonts; embedded as Type 42"
        and set(fonts) == {"regular", "bold", "policy"},
        "historical font family/style policy changed",
    )
    try:
        regular_font, observed_regular = renderer._font(
            Path(regular["path"]), role="regular"
        )
        bold_font, observed_bold = renderer._font(
            Path(bold["path"]), role="bold"
        )
        _require(
            observed_regular == regular and observed_bold == bold,
            "historical recorded font bytes differ from exact rerender fonts",
        )
        with tempfile.TemporaryDirectory(prefix="negfpy-historical-rerender-") as temp:
            rerender_root = Path(temp)
            rerender_pdf = rerender_root / pdf_name
            rerender_png = rerender_root / png_name
            renderer._render_figure(
                data,
                rerender_pdf,
                rerender_png,
                regular_font=regular_font,
                bold_font=bold_font,
                font_family=regular["family"],
            )
            _require(
                rerender_pdf.read_bytes() == (tree.root / pdf_name).read_bytes()
                and rerender_png.read_bytes() == (tree.root / png_name).read_bytes(),
                "historical figures differ from the canonical deterministic rerender",
            )
    except ReleaseConfigError:
        raise
    except Exception as error:
        raise ReleaseConfigError(
            "committed historical renderer could not reproduce exact figure bytes"
        ) from error
    expected_receipt = {
        "schema": renderer._SCHEMA,
        "status": renderer._STATUS,
        "release_certified": False,
        "production_evidence": False,
        "caption_metadata": {
            "caption": renderer._CAPTION,
            "published_data_provenance": (
                "figure-extracted paths/points; not author raw data"
            ),
            "wang_convention_scope": "conditional_inferred_onsite_convention",
            "mingo_half_weight_scope": "separately_named_noncertifying_sensitivity",
            "mingo_fig8_interacting_curve": "not_computed_not_shown",
        },
        "source": source,
        "input_sha256": dict(sorted(input_hashes.items())),
        "metrics_recomputed_from_csv": data["metrics"],
        "reported_metrics": dict(reported_metrics),
        "figure_sha256": {
            pdf_name: tree.entries[pdf_name],
            png_name: tree.entries[png_name],
        },
        "figure_validation": {
            "pdf": pdf_validation,
            "png": png_validation,
        },
        "fonts": dict(fonts),
        "renderer": renderer_identity,
        "metrics_manifest": {
            "path": HISTORICAL_METRICS,
            "sha256": tree.entries[HISTORICAL_METRICS],
        },
        "output_contract": {
            "atomic_noreplace": True,
            "single_multipanel_figure": True,
            "vector_pdf": True,
            "rgb_png_450_dpi": True,
            "sha256sums_exact_coverage": True,
        },
    }
    _require(
        receipt == expected_receipt,
        "historical render manifest differs from recomputed semantic/structural evidence",
    )
    return _file_sha256(receipt_path), MappingProxyType(
        {key: float(value) for key, value in reported_metrics.items()}
    )


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
    publication_context = _current_publication(artifact_tree)
    publication = _mapping(
        publication_context.get("publication"), "current attempt publication"
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
    _require(
        selection_relative == publication.get("scaling_selection")
        and release_relative == publication.get("mpi_runner_result"),
        "caller-supplied scaling/runner paths do not equal current publication selections",
    )
    scaling, scaling_record = _selection(artifact_tree, selection_relative)
    _require(
        scaling["commit"] == publication_context["commit"],
        "published attempt commit differs from scaling identity",
    )
    fetch_receipt = _strict_json(
        artifact_tree.root / "fetch-receipt.json",
        "fetch receipt",
        style="indent-ascii",
    )
    _require(
        fetch_receipt.get("commit") == scaling["commit"],
        "fetch receipt commit differs from scaling identity",
    )
    terminal_facts = _mapping(
        _mapping(
            publication_context["status"], "published terminal status"
        ).get("mpi_execution_facts"),
        "published terminal MPI facts",
    )
    execution_id = terminal_facts.get("execution_id")
    _require(
        isinstance(execution_id, str) and execution_id,
        "published terminal MPI execution identifier is absent",
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
        expected_execution_id=execution_id,
        nested_root=artifact_tree.root / "toy-release",
    )
    publication_chain = _validate_publication_chain(
        artifact_tree,
        publication_context,
        scaling=scaling,
        scaling_record=scaling_record,
        release=release,
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
    artifact_fetch_receipt_sha = _sha(
        artifact_tree.local_receipt_sha256, "artifact fetch receipt SHA256"
    )
    source_records = _mapping(
        publication_chain["source_records"], "validated remote source records"
    )
    staging_receipt_sha = _validate_staging(
        staging_tree,
        commit=scaling["commit"],
        git_tree=publication_chain["git_tree"],
        tree_sha=scaling["tree_sha"],
        environment_sha=scaling["environment_sha"],
        artifact_anchor=artifact_tree.raw_manifest_sha256,
        artifact_fetch_receipt_sha=artifact_fetch_receipt_sha,
        fetch_receipt=fetch_receipt,
        release_report_sha=release["report_file_sha256"],
        release_report_body_sha=release["report_body_sha256"],
        nested_report_body_sha=release["nested_report_body_sha256"],
        identity_bundle=_mapping(
            publication_chain["identity_bundle"], "validated identity bundle"
        ),
        remote_environment=_mapping(
            publication_chain["remote_environment"], "validated remote environment"
        ),
        source_records=source_records,
    )
    production_receipt_sha = _validate_production_figures(
        production_tree,
        staging_tree=staging_tree,
        source_records=source_records,
        commit=scaling["commit"],
        staging_anchor=staging_tree.raw_manifest_sha256,
        staging_receipt_sha=staging_receipt_sha,
        release_report_sha=release["report_file_sha256"],
    )
    historical_receipt_sha, historical_metrics = _validate_historical_figures(
        historical_tree, source_records=source_records
    )
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
        ("CurrentAttemptSHA", publication_chain["current_file_sha256"]),
        ("AttemptPublicationSHA", publication_chain["publication_file_sha256"]),
        (
            "MPIReleaseLaunchIntentSHA",
            publication_chain["intent_file_sha256"],
        ),
        (
            "MPIReleaseLaunchIntentPayloadSHA",
            publication_chain["intent_payload_sha256"],
        ),
        ("MPIReleaseLaunchSHA", publication_chain["launch_file_sha256"]),
        (
            "MPIReleaseLaunchPayloadSHA",
            publication_chain["launch_payload_sha256"],
        ),
        (
            "MPIExecutionSourceIdentitySHA",
            publication_chain["execution_source_identity_sha256"],
        ),
        (
            "MPIEnvironmentPrefixManifestSHA",
            publication_chain["environment_prefix_manifest_sha256"],
        ),
        (
            "MPILaunchEnvironmentSHA",
            publication_chain["launch_environment_sha256"],
        ),
        ("ArtifactSHA256SUMSRawSHA", artifact_tree.raw_manifest_sha256),
        ("ArtifactFetchReceiptSHA", str(artifact_tree.local_receipt_sha256)),
        ("StagingSHA256SUMSRawSHA", staging_tree.raw_manifest_sha256),
        ("StagingImportReceiptSHA", staging_receipt_sha),
        ("ProductionFiguresSHA256SUMSRawSHA", production_tree.raw_manifest_sha256),
        ("ProductionFiguresReceiptSHA", production_receipt_sha),
        ("HistoricalFiguresSHA256SUMSRawSHA", historical_tree.raw_manifest_sha256),
        ("HistoricalFiguresReceiptSHA", historical_receipt_sha),
        ("HistoricalMetricsManifestSHA", historical_tree.entries[HISTORICAL_METRICS]),
        (
            "HistoricalWangHarmonicRMSError",
            f"{historical_metrics['wang_harmonic_rms_error_nW_per_K']:.6f}",
        ),
        (
            "HistoricalWangHarmonicMaxError",
            f"{historical_metrics['wang_harmonic_max_abs_error_nW_per_K']:.6f}",
        ),
        (
            "HistoricalWangBornErrorAt100K",
            f"{historical_metrics['wang_born_t2_relative_error_percent_100K']:.2f}",
        ),
        (
            "HistoricalWangBornErrorAt300K",
            f"{historical_metrics['wang_born_t2_relative_error_percent_300K']:.2f}",
        ),
        (
            "HistoricalWangBornErrorAt500K",
            f"{historical_metrics['wang_born_t2_relative_error_percent_500K']:.2f}",
        ),
        (
            "HistoricalWangMeanFieldErrorAt300K",
            f"{historical_metrics['wang_selective_mean_field_relative_error_percent_300K']:.2f}",
        ),
        (
            "HistoricalWangMeanFieldErrorAt400K",
            f"{historical_metrics['wang_selective_mean_field_relative_error_percent_400K']:.2f}",
        ),
        (
            "HistoricalMingoPrintedRMSError",
            f"{historical_metrics['mingo_printed_normalization_rms_error_pW_per_K']:.4f}",
        ),
        (
            "HistoricalMingoPrintedMaxError",
            f"{historical_metrics['mingo_printed_normalization_max_abs_error_pW_per_K']:.4f}",
        ),
        (
            "HistoricalMingoHalfWeightRMSError",
            f"{historical_metrics['mingo_half_weight_rms_error_pW_per_K']:.4f}",
        ),
        (
            "HistoricalMingoHalfWeightMaxError",
            f"{historical_metrics['mingo_half_weight_max_abs_error_pW_per_K']:.4f}",
        ),
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
