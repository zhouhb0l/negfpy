"""Reproducible algebra/performance probe; synthetic lines, no material work."""

from __future__ import annotations

import argparse
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from negfpy.anharmonic import (
    CubicSCBAProblem, DenseCubicVertex, HarmonicReservoir, SignedFrequencyGrid,
    build_contracted_cubic_cut, connected_cubic_moments,
)
from negfpy.core.types import Device1D, LeadBlocks


def synthetic_problem(active):
    """Dense active support with four inactive device coordinates."""
    n = active + 4
    rng = np.random.default_rng(926)
    raw = rng.normal(size=(active,) * 3) * 0.01
    symmetric = sum(raw.transpose(order) for order in permutations(range(3))) / 6
    values = np.zeros((n,) * 3)
    values[2:-2, 2:-2, 2:-2] = symmetric
    lead = LeadBlocks(np.array([[2.0]]), np.array([[-0.2]]))
    return CubicSCBAProblem(
        device=Device1D([np.array([[1.5]]) for _ in range(n)],
                        [np.array([[-0.2]]) for _ in range(n - 1)]),
        left=HarmonicReservoir(lead, temperature=1.0),
        right=HarmonicReservoir(lead, temperature=0.5),
        vertex=DenseCubicVertex(values),
        grid=SignedFrequencyGrid(step=0.025, n_positive=300),
    )


def lines(times, ndof):
    # Complex nonsymmetric amplitudes exercise algebra only; not a physical state.
    rng = np.random.default_rng(305)
    amplitude = (rng.normal(size=(2, ndof, ndof))
                 + 1j * rng.normal(size=(2, ndof, ndof))) / np.sqrt(2 * ndof)
    rates = np.array([0.2 + 0.9j, 0.13 - 0.6j])
    phase = np.exp(-np.asarray(times)[:, None] * rates)
    c = np.einsum("tm,mij->tij", phase, amplitude)
    p = np.einsum("tm,m,mij->tij", phase, -rates, amplitude)
    return c, p


def provider_for(plan):
    def provider(times):
        c, p = lines(times, plan.operator.ndof)
        return (c[:, plan.displacement_rows][:, :, plan.active_indices],
                p[:, plan.momentum_rows][:, :, plan.active_indices])
    return provider


def benchmark(*, scaled_active=120):
    small = synthetic_problem(12)
    started = perf_counter()
    plan = build_contracted_cubic_cut(small, small.device.n_layers // 2)
    compile_seconds = perf_counter() - started
    times = np.linspace(0, 2, 65)
    c, p = lines(times, small.ndof)
    provider = provider_for(plan)
    reference_times, contracted_times = [], []
    for _ in range(3):
        started = perf_counter()
        moments = connected_cubic_moments(small.vertex, plan.operator.cubic_indices, times, c, p)
        expected = plan.operator.cubic_coefficients @ moments["connected_moments"]
        reference_times.append(perf_counter() - started)
        assert abs(expected) > 1e-8, "Benchmark must resolve a nonzero current."
        started = perf_counter()
        result = plan.integrate(times, provider, block_size=8)
        contracted_times.append(perf_counter() - started)
        np.testing.assert_allclose(result["connected_cubic_solver"], expected, rtol=1e-11, atol=1e-13)
    old_median = float(np.median(reference_times))
    new_median = float(np.median(contracted_times))
    small_result = {
        "active_coordinates": 12, "lag_points": len(times), "operator_terms": len(plan.operator.cubic_indices),
        "compile_seconds": compile_seconds, "reference_seconds": reference_times,
        "contracted_seconds": contracted_times, "reference_median_seconds": old_median,
        "contracted_median_seconds": new_median, "median_speedup": old_median / new_median,
        "absolute_current_difference": float(abs(result["connected_cubic_solver"] - expected)),
        "connected_cubic_solver": result["connected_cubic_solver"],
        "reference_input_histories_precomputed": True,
        "contracted_provider_generation_in_timing": True,
    }
    # Material-sized active dimensions, denser synthetic support than the actual
    # material. Only three lag points are timed; no full-material runtime claim.
    started = perf_counter()
    large = synthetic_problem(scaled_active)
    preparation_seconds = perf_counter() - started
    started = perf_counter()
    big = build_contracted_cubic_cut(large, large.device.n_layers // 2)
    large_compile = perf_counter() - started
    lag = np.array([0.0, 0.05, 0.1])
    durations = []
    for _ in range(3):
        started = perf_counter()
        large_result = big.integrate(lag, provider_for(big), block_size=1)
        durations.append(perf_counter() - started)
    return {
        "scope": "synthetic-complex-algebra-and-local-throughput",
        "reference_agreement_passed": True, "small_comparison": small_result,
        "scaled_synthetic": {
            "active_coordinates": scaled_active, "device_coordinates": large.ndof,
            "operator_terms": len(big.operator.cubic_indices),
            "problem_preparation_seconds": preparation_seconds,
            "plan_compile_seconds": large_compile, "lag_points_per_repeat": 3,
            "integrate_seconds": durations, "median_seconds_per_lag": float(np.median(durations) / 3),
            "plan_metadata": big.metadata,
            "selected_correlation_block_bytes": large_result["selected_correlation_block_bytes"],
            "full_histories_at_4801_points_bytes": 2 * 4801 * large.ndof**2 * 16,
            "same_selected_histories_at_4801_points_bytes": 4801 * scaled_active
            * (len(big.displacement_rows) + len(big.momentum_rows)) * 16,
            "material_topology_or_fields_used": False,
            "full_material_runtime_predicted": False,
        },
        "disconnected_means_or_material_physics_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scaled-active", type=int, default=120)
    args = parser.parse_args()
    if not 2 <= args.scaled_active <= 120:
        parser.error("scaled-active must lie between 2 and 120")
    result = benchmark(scaled_active=args.scaled_active)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"agreement": result["reference_agreement_passed"],
                      "speedup": result["small_comparison"]["median_speedup"],
                      "scaled_seconds_per_lag": result["scaled_synthetic"]["median_seconds_per_lag"]}, indent=2))


if __name__ == "__main__":
    main()
