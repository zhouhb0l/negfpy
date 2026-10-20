"""Bounded synthetic coupled-q FFT benchmark; no material or SCBA calculation."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft
from negfpy.anharmonic.transverse_q import (
    TransverseCubicVertex,
    cubic_born_self_energy_q,
    difference_q_indices,
)

_ACCURACY_CASE = {"active_dofs": 3, "inactive_dofs": 2, "n_positive": 16}
_SCALING_CASES = [
    {"active_dofs": dimension, "inactive_dofs": 4, "n_positive": 300}
    for dimension in (8, 16, 24)
]


def plain(value):
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def array_hash(values):
    """Hash existing contiguous buffers without serializing a duplicate array."""
    if not values.flags.c_contiguous:
        raise ValueError("Benchmark arrays must be contiguous for bounded hashing.")
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def fixture(case, seed, coupling, step):
    """Real bond-polynomial cubic potential and dense positive correlations."""
    mesh = UniformBZMesh((4, 1, 1))
    grid = SignedFrequencyGrid(step, case["n_positive"])
    active_count = case["active_dofs"]
    dimension = active_count + case["inactive_dofs"]
    active = np.linspace(0, dimension - 1, active_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rank = active_count + 1
    vectors = rng.normal(size=(rank, active_count)) / np.sqrt(active_count)
    neighbors = 0.4 * rng.normal(size=(rank, active_count)) / np.sqrt(active_count)
    coefficients = rng.normal(size=rank) / np.sqrt(rank)
    angles = 2 * np.pi * np.arange(mesh.n_qpoints) / mesh.n_qpoints
    factors = vectors[None] + np.exp(1j * angles)[:, None, None] * neighbors[None]
    routes = difference_q_indices(mesh)
    values = np.zeros((4, 4, dimension, dimension, dimension), dtype=complex)
    for q in range(4):
        for p in range(4):
            # Sum alpha_l (v_l.u_R+w_l.u_{R+ex})^3/6. The coupling is
            # applied once to V, so both Born components scale as g**2.
            block = coupling * np.einsum(
                "l,la,lc,ld->acd", coefficients,
                factors[mesh.negative_index(q)], factors[p], factors[routes[q, p]],
                optimize=True,
            )
            values[q, p][np.ix_(active, active, active)] = block
    vertex = TransverseCubicVertex(mesh, values)
    del values
    green_shape = (4, grid.n_points, dimension, dimension)
    greater = np.empty(green_shape, dtype=complex)
    for q in range(4):
        random = (
            rng.normal(size=(grid.n_points, dimension, 4))
            + 1j * rng.normal(size=(grid.n_points, dimension, 4))
        ) / np.sqrt(8)
        greater[q] = -1j * (random @ random.conj().swapaxes(-1, -2))
    lesser = np.empty_like(greater)
    for q in range(4):
        lesser[q] = greater[mesh.negative_index(q), ::-1].swapaxes(-1, -2)
    lesser.setflags(write=False)
    greater.setflags(write=False)
    return grid, vertex, lesser, greater, active


def output_checks(grid, vertex, fields, active):
    expected_shape = (vertex.mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
    inactive = sorted(set(range(vertex.ndof)) - set(active.tolist()))
    maximum = max(float(np.max(np.abs(field))) for field in fields)
    allowance = 1e-12 + 1e-10 * maximum
    antihermiticity = 0.0
    reflection = 0.0
    sampled_minimum = 0.0
    sampled_indices = (0, grid.zero_index, grid.n_points - 1)
    for field in fields:
        if field.shape != expected_shape or not np.isfinite(field).all():
            raise ValueError("FFT output shape or finiteness check failed.")
        for inactive_dof in inactive:
            if np.any(field[..., inactive_dof, :] != 0) or np.any(field[..., :, inactive_dof] != 0):
                raise ValueError("Inactive output coordinates are not exactly zero.")
        for q in range(vertex.mesh.n_qpoints):
            antihermiticity = max(
                antihermiticity,
                float(np.max(np.abs(field[q] + field[q].conj().swapaxes(-1, -2)))),
            )
            for index in sampled_indices:
                sampled_minimum = min(
                    sampled_minimum, float(np.linalg.eigvalsh(1j * field[q, index]).min()),
                )
    for q in range(vertex.mesh.n_qpoints):
        reflection = max(reflection, float(np.max(np.abs(
            fields[0][q, ::-1]
            - fields[1][vertex.mesh.negative_index(q)].swapaxes(-1, -2)
        ))))
    if antihermiticity > allowance or reflection > allowance or sampled_minimum < -allowance:
        raise ValueError("Synthetic output identities or sampled positivity failed.")
    return {
        "maximum_absolute_entry": maximum,
        "antihermiticity_maximum": antihermiticity,
        "opposite_q_reflection_maximum": reflection,
        "sampled_correlation_minimum_eigenvalue": sampled_minimum,
        "sampled_frequency_indices": sampled_indices,
        "identity_allowance": allowance,
        "inactive_coordinates_exact_zero": True,
        "output_sha256": [array_hash(field) for field in fields],
    }


def timed_call(function, args, kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def run(profile, result, emit):
    cfg = profile["benchmark"]
    if profile["runtime_inputs"] != []:
        raise ValueError("The synthetic benchmark must not load material input files.")
    if (cfg["g"] != 0.03 or cfg["mesh"] != [4, 1, 1] or cfg["step"] != 0.025
            or cfg["seed"] != 20261021 or cfg["memory_budget_bytes"] != 512 * 1024**2
            or cfg["max_time_pair_evaluations"] != 100_000
            or cfg["accuracy_case"] != _ACCURACY_CASE
            or cfg["scaling_cases"] != _SCALING_CASES
            or cfg["accuracy_rtol"] != 1e-10 or cfg["accuracy_atol"] != 1e-12):
        raise ValueError("The benchmark profile differs from the frozen bounded experiment.")
    pools = threadpool_info()
    if any(pool["num_threads"] != 1 for pool in pools):
        raise ValueError("Every numerical thread pool must use one thread.")
    result.update(
        settings=cfg, threadpools=pools,
        source_commit=profile["source_commit"], source_tree=profile["source_tree"],
        scope={
            "synthetic_only": True, "material_scba": False, "currents": False,
            "raw_material_input_read": False, "accepted_fields_changed": False,
            "material_broadening_changed": False,
            "calls_per_method_per_case": 1, "warmup_or_repetitions": False,
            "timings_include_kernel_setup": True, "timings_include_fixture_or_checks": False,
            "timings_are_single_observations": True,
            "rss_is_cumulative_process_high_water_not_kernel_peak": True,
            "memory_limit_applies_separately_to_each_output_and_planned_fft_workspace": True,
            "memory_limit_excludes_input_arrays_library_scratch_and_python": True,
        },
        cases=[],
    )
    cases = [cfg["accuracy_case"], *cfg["scaling_cases"]]
    for index, case in enumerate(cases):
        label = "accuracy" if index == 0 else f"fft-active-{case['active_dofs']}"
        emit("case_start", {"case": label, **case})
        started = time.monotonic()
        grid, vertex, lesser, greater, active = fixture(
            case, cfg["seed"] + index, cfg["g"], cfg["step"],
        )
        inputs = (vertex.coefficients, lesser, greater)
        before = [array_hash(array) for array in inputs]
        record = {
            "case": label, **case, "nq": 4, "n_frequency": grid.n_points,
            "full_dofs": vertex.ndof, "active_indices": active,
            "seed": cfg["seed"] + index, "input_sha256_before": before,
            "input_payload_bytes": sum(array.nbytes for array in inputs),
            "complex_vertex_maximum_imaginary_entry": float(np.max(np.abs(vertex.coefficients.imag))),
            "coupling_applied_once_to_vertex": cfg["g"],
            "fft_points": 1 << (2 * grid.n_points - 2).bit_length(),
            "fft_pairs_per_component": 16 * (1 << (2 * grid.n_points - 2).bit_length()),
            "direct_pairs_per_component": 16 * (
                3 * grid.n_positive**2 + 3 * grid.n_positive + 1
            ),
        }
        result["cases"].append(record)
        args = (grid, vertex, lesser, greater)
        direct = None
        if index == 0:
            emit("direct_start", {"case": label})
            direct, record["direct_seconds"] = timed_call(
                cubic_born_self_energy_q, args,
                {"max_pair_evaluations": 100_000,
                 "max_output_bytes": cfg["memory_budget_bytes"]},
            )
            record["direct_checks"] = output_checks(grid, vertex, direct, active)
            emit("direct_complete", {"case": label, "seconds": record["direct_seconds"]})
        emit("fft_start", {"case": label})
        fft, record["fft_seconds"] = timed_call(
            cubic_born_self_energy_q_fft, args,
            {"max_bytes": cfg["memory_budget_bytes"],
             "max_pair_evaluations": cfg["max_time_pair_evaluations"]},
        )
        record["fft_checks"] = output_checks(grid, vertex, fft, active)
        if direct is not None:
            comparison = []
            for reference, candidate in zip(direct, fft, strict=True):
                difference = candidate - reference
                if not np.allclose(candidate, reference, rtol=cfg["accuracy_rtol"],
                                   atol=cfg["accuracy_atol"]):
                    raise ValueError("FFT differs from the independent direct contraction.")
                comparison.append({
                    "maximum_absolute_difference": float(np.max(np.abs(difference))),
                    "max_normalized_difference": float(np.max(np.abs(difference)))
                    / max(float(np.max(np.abs(reference))), np.finfo(float).tiny),
                    "frobenius_relative_difference": float(np.linalg.norm(difference))
                    / max(float(np.linalg.norm(reference)), np.finfo(float).tiny),
                })
            record["direct_fft_comparison"] = comparison
            record["single_call_direct_to_fft_time_ratio"] = (
                record["direct_seconds"] / record["fft_seconds"]
            )
        after = [array_hash(array) for array in inputs]
        record["input_sha256_after"] = after
        if before != after:
            raise ValueError("Benchmark input buffers changed.")
        record.update(
            passed=True, elapsed_seconds=time.monotonic() - started,
            maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        emit("case_complete", record)
        del fft, direct, inputs, args, vertex, lesser, greater
    result["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {"passed": False, "schema": "negfpy.transverse-fft-benchmark.v1"}

    def emit(event, details=None):
        row = {
            "event": event, "time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started, "details": plain(details or {}),
        }
        with (args.output.parent / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    try:
        run(profile, result, emit)
    except Exception as exc:  # noqa: BLE001 - preserve completed cases on failure
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
        emit("benchmark_failed", result["failure"])
    result["elapsed_seconds"] = time.monotonic() - started
    result["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.output, result)
    emit("benchmark_closed", {"passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"]})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
