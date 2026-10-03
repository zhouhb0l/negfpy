#!/usr/bin/env python3
"""Bounded sparse FFT kernel benchmark on an authenticated closed SCBA result.

No SCBA iteration, continuation, or scheduler action occurs. The parent process
enforces a maximum 15-minute lifetime, including input loading and validation.
Use the accepted g=.015 artifact root, never an active checkpoint or field file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from negfpy.anharmonic._provenance import problem_input_hashes
from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.sparse_cubic import _compiled_sparse_schedule


SCHEMA = "negfpy.sparse-fft-thread-benchmark.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _event(value: dict) -> None:
    print(json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def _load_parent(args):
    import h5py
    import run_sparse_nq1_phase_c_scba_campaign as campaign
    import validate_sparse_nq1_phase_c_scba as case

    root = args.parent_artifacts.resolve(strict=True)
    manifest_path = campaign._regular_file(
        root / "campaign-manifest.json", "parent manifest"
    )
    status_path = campaign._regular_file(root / "status.json", "parent status")
    if _sha256(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("parent manifest differs from the pinned accepted result")
    if _sha256(status_path) != args.expected_status_sha256:
        raise ValueError("parent status differs from the pinned accepted result")
    manifest = campaign._load_json(manifest_path, "parent manifest")
    status = campaign._authenticated_terminal_status(
        root, manifest, manifest_path, status_path
    )
    if status["state"] != "passed" or status["scba_converged"] is not True:
        raise ValueError("benchmark requires a closed converged parent")
    # Reconstruct and authenticate inputs only. This does not solve Dyson or SCBA.
    _identity, problem, _config = case._reconstruct_parent_inputs(manifest)
    expected_inputs = manifest["preflight"]["source"]["problem_input_hashes"]
    if problem_input_hashes(problem) != expected_inputs:
        raise ValueError("reconstructed parent problem differs from the manifest")
    fields = campaign._regular_file(Path(manifest["outputs"]["hdf5"]), "parent fields")
    fields_hash = _sha256(fields)
    outcome = campaign._load_json(Path(manifest["outputs"]["json"]), "parent outcome")
    if fields_hash != outcome["hdf5"]["sha256"]:
        raise ValueError("parent fields digest differs")
    with h5py.File(fields, "r") as handle:
        if handle.attrs["schema"] != case.ARCHIVE_SCHEMA:
            raise ValueError("parent field schema differs")
        omega = np.asarray(handle["grid/omega"][...])
        lesser = np.asarray(handle["green/lesser"][...])
        greater = np.asarray(handle["green/greater"][...])
    if not np.array_equal(omega, problem.grid.values):
        raise ValueError("parent frequency grid differs")
    for name, values in (("green/lesser", lesser), ("green/greater", greater)):
        if case._array_sha256(values) != outcome["result"]["array_hashes"][name]:
            raise ValueError(f"parent field array hash differs: {name}")
        if values.shape != (problem.grid.n_points, problem.ndof, problem.ndof):
            raise ValueError("parent Green field shape differs")
        if not np.all(np.isfinite(values)):
            raise ValueError("parent Green field contains non-finite values")
        values.setflags(write=False)
    pins = {
        str(manifest_path): args.expected_manifest_sha256,
        str(status_path): args.expected_status_sha256,
        str(fields): fields_hash,
    }
    schedule = _compiled_sparse_schedule(problem.vertex)
    info = {
        "parent_artifacts": str(root),
        "parent_coupling_scale": manifest["coupling_scale"],
        "input_sha256": pins,
        "problem_input_hashes": expected_inputs,
        "n_frequency": problem.grid.n_points,
        "ndof": problem.ndof,
        "active_dofs": len(schedule.active),
        "vertex_nnz": problem.vertex.nnz,
        "fft_padding": 1 << (2 * problem.grid.n_points - 2).bit_length(),
        "schedule_storage_bytes": schedule.numeric_storage_bytes,
        "pair_multiply_add_count": schedule.pair_multiply_add_count,
    }
    return problem, lesser, greater, pins, info


def benchmark_cases(vertex, grid, lesser, greater, workers, *, emit=_event):
    """Measure complete two-bubble calls without changing any solver state."""
    if not workers or workers[0] != 1 or len(set(workers)) != len(workers):
        raise ValueError("workers must begin with the unique serial baseline 1")
    if any(item not in (1, 2, 4, 8) for item in workers):
        raise ValueError("benchmark workers must be selected from 1, 2, 4, 8")
    baseline = None
    baseline_seconds = None
    results = []
    # Both serial and threaded measurements use the same one-thread BLAS limit.
    with threadpool_limits(limits=1, user_api="blas"):
        for count in workers:
            start_wall = time.perf_counter()
            start_cpu = time.process_time()
            pair = cubic_lesser_greater(
                vertex, lesser, greater, grid, backend="fft", max_workers=count
            )
            cpu_seconds = time.process_time() - start_cpu
            wall_seconds = time.perf_counter() - start_wall
            if baseline is None:
                baseline = pair
                baseline_seconds = wall_seconds
            components = []
            for name, actual, reference in zip(
                ("lesser", "greater"), pair, baseline, strict=True
            ):
                passed = bool(np.allclose(actual, reference, rtol=1e-10, atol=1e-12))
                components.append(
                    {
                        "component": name,
                        "sha256": _array_sha256(actual),
                        "exactly_equal": bool(np.array_equal(actual, reference)),
                        "max_absolute_error": float(np.max(np.abs(actual - reference))),
                        "passed": passed,
                    }
                )
            record = {
                "event": "measurement",
                "workers": count,
                "wall_seconds": wall_seconds,
                "cpu_seconds": cpu_seconds,
                "average_busy_cores": cpu_seconds / wall_seconds,
                "speedup_vs_serial": baseline_seconds / wall_seconds,
                "process_peak_rss_kib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss,
                "components": components,
                "passed": all(item["passed"] for item in components),
            }
            results.append(record)
            emit(record)
            if record["passed"] is not True:
                raise ValueError("threaded FFT result differs from serial baseline")
    return results


def _child(args) -> int:
    started = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        problem, lesser, greater, pins, info = _load_parent(args)
    _event({"event": "inputs", **info, "load_seconds": time.perf_counter() - started})
    results = benchmark_cases(
        problem.vertex, problem.grid, lesser, greater, args.workers
    )
    for filename, expected in pins.items():
        if _sha256(Path(filename)) != expected:
            raise ValueError(
                "accepted benchmark input changed during the read-only run"
            )
    _event(
        {
            "event": "complete",
            "passed": all(row["passed"] for row in results),
            "elapsed_seconds": time.perf_counter() - started,
            "thread_pools_after": threadpool_info(),
            "inputs_unchanged": True,
            "scientific_solve_executed": False,
        }
    )
    return 0


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-artifacts", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-status-sha256", required=True)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--wall-seconds", type=int, default=900)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--benchmark-child", action="store_true", help=argparse.SUPPRESS
    )
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if not 1 <= args.wall_seconds <= 900:
        parser.error("--wall-seconds must be between 1 and 900")
    if (
        not args.workers
        or args.workers[0] != 1
        or len(set(args.workers)) != len(args.workers)
        or any(count not in (1, 2, 4, 8) for count in args.workers)
    ):
        parser.error("--workers must be unique values from 1, 2, 4, 8, starting with 1")
    for value in (args.expected_manifest_sha256, args.expected_status_sha256):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            parser.error("expected digests must be lowercase SHA-256 values")
    if args.benchmark_child:
        try:
            return _child(args)
        except Exception as error:
            _event({"event": "error", "error": f"{type(error).__name__}: {error}"})
            return 2
    if args.output is not None and args.output.exists():
        parser.error("--output already exists; refusing overwrite")
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
        "--benchmark-child",
    ]
    started = time.perf_counter()
    timed_out = False
    try:
        child = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.wall_seconds,
            check=False,
        )
        stdout, stderr, returncode = child.stdout, child.stderr, child.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = error.stdout or b""
        stderr = error.stderr or b""
        stdout = (
            stdout.decode("utf-8", errors="replace")
            if isinstance(stdout, bytes)
            else stdout
        )
        stderr = (
            stderr.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes)
            else stderr
        )
        returncode = None
    events = []
    parse_errors = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            parse_errors.append(line)
    passed = bool(
        returncode == 0
        and not timed_out
        and not stderr
        and not parse_errors
        and events
        and events[-1].get("event") == "complete"
        and events[-1].get("passed") is True
    )
    report = {
        "schema": SCHEMA,
        "passed": passed,
        "wall_limit_seconds": args.wall_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "timed_out": timed_out,
        "child_returncode": returncode,
        "events": events,
        "stderr": stderr,
        "unparsed_stdout": parse_errors,
        "scientific_solve_executed": False,
        "release_certified": False,
        "measurement_note": "One kernel evaluation per worker count; no end-to-end SCBA speedup is claimed.",
    }
    encoded = json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
