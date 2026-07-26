#!/usr/bin/env python3
"""Generate fail-closed LaTeX release configuration from remote MPI evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence


RANK_LADDER = (1, 2, 4, 8, 16)
EXPECTED_THREADS = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
DATE_RE = re.compile(r"^[0-9]{1,2} [A-Za-z]+ [0-9]{4}$")
SAFE_PATH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
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
REQUIRED_PRODUCTION_FIGURES = (
    "figure_open_oracle_born.pdf",
    "figure_scba_validation.pdf",
    "figure_passivity_backend_timing.pdf",
)
REQUIRED_HISTORICAL_FIGURES = ("historical_paper_benchmarks.pdf",)


class ReleaseConfigError(RuntimeError):
    """Raised when report configuration evidence is incomplete or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseConfigError(message)


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseConfigError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ReleaseConfigError(f"non-finite JSON token: {value}")


def _payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _document_bytes(document: Mapping[str, Any]) -> bytes:
    return _payload_bytes(document) + b"\n"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    _require(path.exists(), f"{name} is missing: {path}")
    _require(path.is_file() and not path.is_symlink(), f"{name} is not a regular file")
    return path.resolve(strict=True)


def _read_hashed_manifest(path: Path, name: str) -> tuple[dict[str, Any], str]:
    source = _regular_file(path, name)
    try:
        raw = source.read_bytes()
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_invalid_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseConfigError(f"{name} is not strict ASCII JSON") from exc
    _require(
        isinstance(document, dict)
        and set(document) == {"payload", "payload_sha256"},
        f"{name} has the wrong hashed-envelope shape",
    )
    payload = document["payload"]
    digest = document["payload_sha256"]
    _require(isinstance(payload, dict), f"{name} payload is not an object")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
             f"{name} payload SHA256 is invalid")
    _require(hashlib.sha256(_payload_bytes(payload)).hexdigest() == digest,
             f"{name} payload SHA256 mismatch")
    _require(raw == _document_bytes(document), f"{name} is not canonical JSON")
    return payload, _file_sha256(source)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} is not an object")
    return value


def _sha(value: Any, name: str) -> str:
    _require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
             f"{name} is not a lowercase SHA256")
    return value


def _positive_number(value: Any, name: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"{name} must be finite and positive",
    )
    return float(value)


def _memory_record(value: Any, *, rank_count: int, name: str) -> float:
    record = _mapping(value, name)
    _require(record.get("schema") == "negfpy.remote_toy_mpi.aggregate_memory_preflight.v1",
             f"{name} schema changed")
    _require(record.get("measurement") == "linux_proc_meminfo_MemAvailable",
             f"{name} is not a live MemAvailable measurement")
    _require(record.get("rank_count") == rank_count, f"{name} rank count changed")
    allowed = record.get("allowed_fraction")
    _require(isinstance(allowed, (int, float)) and not isinstance(allowed, bool)
             and math.isclose(float(allowed), 0.70, rel_tol=0.0, abs_tol=1e-15),
             f"{name} does not use the locked 70% limit")
    available = record.get("available_memory_bytes")
    aggregate = record.get("aggregate_predicted_bytes")
    limit = record.get("limit_bytes")
    _require(all(isinstance(item, int) and not isinstance(item, bool) and item > 0
                 for item in (available, aggregate, limit)),
             f"{name} byte counts are invalid")
    _require(record.get("passed") is True and aggregate <= limit,
             f"{name} did not pass")
    _require(limit == int(0.70 * available), f"{name} limit is inconsistent")
    return aggregate / available


def _rank_runtime(value: Any, *, rank_count: int, socket_count: int, name: str) -> None:
    _require(isinstance(value, list) and len(value) == rank_count,
             f"{name} rank coverage is incomplete")
    ranks: set[int] = set()
    occupied: set[tuple[str, int, int]] = set()
    packages: dict[tuple[str, int], int] = {}
    for item in value:
        record = _mapping(item, f"{name} record")
        rank = record.get("rank")
        _require(isinstance(rank, int) and not isinstance(rank, bool)
                 and 0 <= rank < rank_count and rank not in ranks,
                 f"{name} has an invalid or duplicate rank")
        ranks.add(rank)
        hostname = record.get("hostname")
        physical = record.get("physical_cores")
        _require(isinstance(hostname, str) and hostname, f"{name} hostname is invalid")
        _require(record.get("one_physical_core") is True
                 and isinstance(physical, list) and len(physical) == 1,
                 f"{name} does not prove one physical core per rank")
        core = _mapping(physical[0], f"{name} physical core")
        package_id = core.get("physical_package_id")
        core_id = core.get("core_id")
        _require(isinstance(package_id, int) and isinstance(core_id, int),
                 f"{name} physical core identifiers are invalid")
        key = (hostname, package_id, core_id)
        _require(key not in occupied, f"{name} oversubscribes a physical core")
        occupied.add(key)
        package = (hostname, package_id)
        packages[package] = packages.get(package, 0) + 1
        _require(record.get("thread_limits") == EXPECTED_THREADS,
                 f"{name} BLAS/OpenMP thread limits changed")
    _require(ranks == set(range(rank_count)), f"{name} rank sequence is incomplete")
    if rank_count >= socket_count and rank_count % socket_count == 0:
        expected = rank_count // socket_count
        _require(len(packages) == socket_count and set(packages.values()) == {expected},
                 f"{name} is not balanced across sockets")


def _validate_scaling(
    payload: Mapping[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    _require(payload.get("schema") == "negfpy.remote_toy_mpi.scaling_selection.v1",
             "scaling-selection schema changed")
    _require(payload.get("profile") == "representative_noncertifying_full_28_task",
             "scaling profile changed")
    _require(payload.get("physics_equal_across_all_ranks") is True,
             "scaling physics equality is not true")
    _require(payload.get("fresh_isolated_repetitions") is True,
             "scaling repetitions were not fresh and isolated")
    commit = payload.get("source_revision")
    _require(isinstance(commit, str) and COMMIT_RE.fullmatch(commit) is not None,
             "scaling source revision is invalid")
    tree_sha = _sha(payload.get("source_tree_sha256"), "scaling source-tree SHA256")
    environment_sha = _sha(payload.get("environment_sha256"), "scaling environment SHA256")
    physics_sha = _sha(
        payload.get("production_toy_physics_sha256"),
        "scaling canonical physics SHA256",
    )
    cores = payload.get("physical_core_count")
    sockets = payload.get("socket_count")
    _require(isinstance(cores, int) and not isinstance(cores, bool) and cores >= 16,
             "scaling host has fewer than 16 physical cores")
    _require(isinstance(sockets, int) and not isinstance(sockets, bool) and sockets > 0,
             "scaling socket count is invalid")

    selection = _mapping(payload.get("selection"), "scaling selection")
    _require(selection.get("schema") == "negfpy.remote_mpi_scaling.selection.v1",
             "rank-selection schema changed")
    _require(selection.get("rank_ladder") == list(RANK_LADDER),
             "rank ladder is not 1/2/4/8/16")
    measurements = selection.get("measurements")
    _require(isinstance(measurements, list) and len(measurements) == len(RANK_LADDER),
             "scaling measurements are incomplete")
    wall: dict[int, float] = {}
    for expected_rank, value in zip(RANK_LADDER, measurements, strict=True):
        row = _mapping(value, "scaling measurement")
        _require(row.get("rank_count") == expected_rank
                 and row.get("completed") is True
                 and row.get("numerically_equal") is True,
                 f"rank-{expected_rank} measurement did not pass")
        wall[expected_rank] = _positive_number(
            row.get("wall_seconds"), f"rank-{expected_rank} wall time"
        )
    fastest_rank = min(RANK_LADDER, key=lambda rank: (wall[rank], rank))
    fastest_wall = wall[fastest_rank]
    candidate_rank = min(
        rank for rank in RANK_LADDER if wall[rank] <= fastest_wall * 1.05
    )
    candidate_speedup = wall[1] / wall[candidate_rank]
    mpi_enabled = candidate_rank > 1 and candidate_speedup >= 2.0
    toy_rank = candidate_rank if mpi_enabled else 1
    expected_fields = {
        "fastest_rank_count": fastest_rank,
        "smallest_near_fastest_rank_count": candidate_rank,
        "toy_rank_count": toy_rank,
        "silicon_rank_count": candidate_rank,
        "toy_mpi_enabled": mpi_enabled,
    }
    _require(all(selection.get(key) == value for key, value in expected_fields.items()),
             "recorded rank selection does not follow the locked policy")
    for key, expected in (
        ("fastest_wall_seconds", fastest_wall),
        ("candidate_speedup", candidate_speedup),
        ("within_fastest_fraction", 0.05),
        ("minimum_toy_speedup", 2.0),
    ):
        actual = selection.get(key)
        _require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
                 and math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12),
                 f"selection field {key} changed")

    toy_execution = _mapping(payload.get("toy_execution"), "toy execution selection")
    silicon_execution = _mapping(
        payload.get("silicon_execution"), "silicon execution selection"
    )
    _require(toy_execution.get("rank_count") == toy_rank,
             "toy execution rank does not match selection")
    _require(silicon_execution.get("rank_count") == candidate_rank,
             "silicon execution rank does not match selection")

    evidence = payload.get("rank_evidence")
    _require(isinstance(evidence, list) and len(evidence) == len(RANK_LADDER),
             "rank evidence is incomplete")
    max_memory_fraction = 0.0
    observed_ranks: list[int] = []
    for row_value in evidence:
        row = _mapping(row_value, "rank evidence")
        rank = row.get("rank_count")
        _require(rank in RANK_LADDER and rank not in observed_ranks,
                 "rank evidence is duplicated or invalid")
        observed_ranks.append(rank)
        _sha(row.get("manifest_file_sha256"), f"rank-{rank} manifest file SHA256")
        _sha(row.get("manifest_payload_sha256"), f"rank-{rank} manifest payload SHA256")
        rank_path = row.get("manifest_path")
        _require(isinstance(rank_path, str)
                 and rank_path == f"rank-{rank:03d}-manifest.json",
                 f"rank-{rank} manifest path changed")
        sibling = _regular_file(manifest_path.parent / rank_path, f"rank-{rank} manifest")
        _require(_file_sha256(sibling) == row["manifest_file_sha256"],
                 f"rank-{rank} manifest file SHA256 mismatch")
        max_memory_fraction = max(
            max_memory_fraction,
            _memory_record(
                row.get("launcher_memory_preflight"),
                rank_count=rank,
                name=f"rank-{rank} launcher memory preflight",
            ),
            _memory_record(
                row.get("worker_memory_preflight"),
                rank_count=rank,
                name=f"rank-{rank} worker memory preflight",
            ),
        )
        _rank_runtime(
            row.get("rank_runtime"),
            rank_count=rank,
            socket_count=sockets,
            name=f"rank-{rank} runtime",
        )
    _require(tuple(sorted(observed_ranks)) == RANK_LADDER,
             "rank evidence ladder is incomplete")
    return {
        "commit": commit,
        "tree_sha": tree_sha,
        "environment_sha": environment_sha,
        "physics_sha": physics_sha,
        "socket_count": sockets,
        "fastest_rank": fastest_rank,
        "candidate_rank": candidate_rank,
        "toy_rank": toy_rank,
        "silicon_rank": candidate_rank,
        "baseline_wall": wall[1],
        "candidate_wall": wall[candidate_rank],
        "candidate_speedup": candidate_speedup,
        "max_memory_fraction": max_memory_fraction,
        "mpi_enabled": mpi_enabled,
    }


def _validate_release_runner(
    payload: Mapping[str, Any],
    *,
    scaling: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "release",
        "certification_eligible": True,
        "release_certified": True,
        "source_revision": scaling["commit"],
        "source_tree_sha256": scaling["tree_sha"],
        "environment_sha256": scaling["environment_sha"],
        "expected_rank_count": scaling["toy_rank"],
        "recorded_rank_count": scaling["toy_rank"],
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
    }
    changed = {key: (payload.get(key), value) for key, value in expected.items()
               if payload.get(key) != value}
    _require(not changed, f"release runner identity or topology changed: {changed}")
    expected_mode = "serial" if scaling["toy_rank"] == 1 else "mpi"
    _require(payload.get("execution_mode") == expected_mode,
             "release runner execution mode changed")
    _require(payload.get("thread_limits") == EXPECTED_THREADS,
             "release runner thread limits changed")
    _require(payload.get("directory_mode") in {"fresh", "exact_resume_candidate"},
             "release runner directory mode is invalid")
    release_physics_sha = _sha(
        payload.get("production_toy_physics_sha256"),
        "release canonical physics SHA256",
    )
    max_memory_fraction = _memory_record(
        payload.get("memory_preflight"),
        rank_count=scaling["toy_rank"],
        name="release memory preflight",
    )
    _rank_runtime(
        payload.get("rank_runtime"),
        rank_count=scaling["toy_rank"],
        socket_count=scaling["socket_count"],
        name="release rank runtime",
    )
    execution = _mapping(payload.get("execution_manifest"), "execution manifest record")
    execution_sha = _sha(execution.get("sha256"), "execution manifest file SHA256")
    for name in ("campaign_manifest", "strict_report"):
        record = _mapping(payload.get(name), f"{name} record")
        _sha(record.get("sha256"), f"{name} file SHA256")
    return {
        "physics_sha": release_physics_sha,
        "execution_sha": execution_sha,
        "max_memory_fraction": max_memory_fraction,
        "directory_mode": payload["directory_mode"],
    }


def _safe_relative_path(value: str, name: str) -> str:
    _require(isinstance(value, str) and SAFE_PATH_RE.fullmatch(value) is not None,
             f"{name} contains unsafe TeX/path characters")
    pure = PurePosixPath(value)
    _require(not pure.is_absolute() and ".." not in pure.parts
             and "." not in pure.parts and pure.as_posix() == value,
             f"{name} must be a normalized relative path")
    return value


def _require_tree(output: Path, relative: str, files: Sequence[str], name: str) -> None:
    root = output.parent / relative
    _require(root.is_dir() and not root.is_symlink(), f"{name} is missing: {root}")
    for filename in files:
        _regular_file(root / filename, f"{name} artifact {filename}")


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
    scaling_selection: Path,
    release_runner: Path,
    staging_dir: str,
    production_figure_dir: str,
    historical_figure_dir: str,
    report_date: str,
    output: Path,
) -> str:
    scaling_payload, scaling_file_sha = _read_hashed_manifest(
        scaling_selection, "scaling selection"
    )
    runner_payload, runner_file_sha = _read_hashed_manifest(
        release_runner, "release runner"
    )
    scaling = _validate_scaling(scaling_payload, scaling_selection.resolve())
    release = _validate_release_runner(runner_payload, scaling=scaling)
    staging = _safe_relative_path(staging_dir, "staging directory")
    production = _safe_relative_path(
        production_figure_dir, "production figure directory"
    )
    historical = _safe_relative_path(
        historical_figure_dir, "historical figure directory"
    )
    _require(DATE_RE.fullmatch(report_date) is not None,
             "report date must look like '26 July 2026'")
    target = output.expanduser().resolve()
    _require_tree(target, f"{staging}/tex", REQUIRED_TEX, "production staging")
    _require_tree(
        target, production, REQUIRED_PRODUCTION_FIGURES, "production figures"
    )
    _require_tree(
        target, historical, REQUIRED_HISTORICAL_FIGURES, "historical figures"
    )

    release_mode = "task-MPI" if scaling["toy_rank"] > 1 else "serial/FFT"
    macros = (
        ("ProductionStagingDir", staging),
        ("ProductionFigureDir", production),
        ("HistoricalFigureDir", historical),
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
            f"{max(scaling['max_memory_fraction'], release['max_memory_fraction']):.6f}",
        ),
        ("MPISocketCount", str(scaling["socket_count"])),
        ("MPIScalingPhysicsSHA", scaling["physics_sha"]),
        ("MPIReleasePhysicsSHA", release["physics_sha"]),
        ("MPIScalingManifestSHA", scaling_file_sha),
        ("MPIRunnerManifestSHA", runner_file_sha),
        ("MPIExecutionManifestSHA", release["execution_sha"]),
        ("MPIToyMinimumSpeedupSatisfied", str(scaling["mpi_enabled"]).lower()),
        ("MPIScalingPassed", "true"),
        ("MPIPhysicsEqualityPassed", "true"),
        ("MPIBindingPassed", "true"),
        ("MPIMemoryPreflightPassed", "true"),
        ("MPIRunnerReleaseCertified", "true"),
    )
    lines = [
        "% Generated by generate_release_config.py from verified remote artifacts.",
        "% Do not hand-edit; regenerate when any exact artifact changes.",
        *(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros),
        "",
    ]
    text = "\n".join(lines)
    _atomic_write(target, text)
    return text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling-selection", required=True, type=Path)
    parser.add_argument("--release-runner", required=True, type=Path)
    parser.add_argument("--staging-dir", required=True)
    parser.add_argument("--production-figure-dir", required=True)
    parser.add_argument("--historical-figure-dir", required=True)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--output", type=Path, default=Path("release_config.tex"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        generate_config(
            scaling_selection=arguments.scaling_selection,
            release_runner=arguments.release_runner,
            staging_dir=arguments.staging_dir,
            production_figure_dir=arguments.production_figure_dir,
            historical_figure_dir=arguments.historical_figure_dir,
            report_date=arguments.report_date,
            output=arguments.output,
        )
    except (OSError, ReleaseConfigError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
