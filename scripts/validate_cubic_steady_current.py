"""Bounded three-site Drude-bath check of the centered bubble/cubic current.

Run with src on PYTHONPATH and one BLAS thread. This is a reduced-unit toy
reference; it does not load material fields, submit jobs or modify SCBA gates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from negfpy.anharmonic import (
    CubicSCBAProblem, DenseCubicVertex, HarmonicReservoir, KeldyshSpectrum,
    SignedFrequencyGrid, SolverScale, build_cubic_cut_current_operator,
    combine_cubic_moments, connected_cubic_moments, correlations_from_green,
)
from negfpy.anharmonic._harmonic import solve_keldysh_green, solve_retarded
from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.current import contact_heat_current
from negfpy.anharmonic.retarded import retarded_from_keldysh
from negfpy.core.types import Device1D, LeadBlocks


def drude_contact(grid, index, temperature, *, gamma=0.25, drude_frequency=3.0):
    """Three-coordinate bath with its static renormalization already included.

    SigmaR(0)=0, SigmaR(infinity)=gamma*Omega. Use this analytic contact,
    not the unsubtracted self-energy Hilbert transform used for the bubble.
    """
    if index not in (0, 2) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("This toy requires a positive-temperature endpoint contact.")
    if not (np.isfinite(gamma) and gamma > 0 and np.isfinite(drude_frequency)
            and drude_frequency > 0):
        raise ValueError("Damping and Drude frequency must be finite and positive.")
    w = grid.values
    retarded = np.zeros((grid.n_points, 3, 3), dtype=complex)
    retarded[:, index, index] = -1j * gamma * w * drude_frequency / (drude_frequency - 1j * w)
    thermal = np.full(w.shape, temperature)
    nonzero = w != 0.0
    with np.errstate(over="ignore"):
        thermal[nonzero] = w[nonzero] / np.expm1(w[nonzero] / temperature)
    lesser = np.zeros_like(retarded)
    lesser[:, index, index] = (-2j * gamma * thermal * drude_frequency**2
                              / (drude_frequency**2 + w**2))
    return KeldyshSpectrum(retarded=retarded, lesser=lesser,
                          greater=lesser + 2j * retarded.imag)


def solve_reference(*, step=0.01, cutoff=8.0, alpha=0.1, temperatures=(1.0, 0.5),
                    max_maps=64):
    """Small cold-start bubble fixed point; not the production controller.

    All lines used for current belong to the last evaluated input self-energy.
    Acceptance requires its freshly computed raw map residual below 1e-12.
    A cap or nonfinite map raises; no incomplete state is treated as converged.
    """
    if (not np.isfinite(step) or step <= 0 or not np.isfinite(cutoff)
            or cutoff < step or not np.isfinite(alpha)):
        raise ValueError("Invalid reference grid or coupling.")
    n_positive = int(round(cutoff / step))
    if not np.isclose(n_positive * step, cutoff, rtol=0, atol=1e-12):
        raise ValueError("cutoff must be an integer multiple of step.")
    if isinstance(max_maps, bool) or not isinstance(max_maps, int) or not 1 <= max_maps <= 64:
        raise ValueError("max_maps must be an integer from 1 to 64.")
    dynamical = np.array([[1.4, -0.25, 0.0], [-0.25, 1.2, -0.22], [0.0, -0.22, 1.3]])
    vertex = np.zeros((3, 3, 3))
    for first, second in ((0, 1), (1, 2)):
        difference = np.zeros(3)
        difference[first], difference[second] = -1, 1
        vertex += alpha * np.einsum("i,j,k->ijk", difference, difference, difference)
    # These placeholders describe contact support for the generic problem
    # object. Their spectra are never used: analytic Drude baths below replace them.
    lead = LeadBlocks(np.array([[2.0]]), np.array([[-0.2]]))
    problem = CubicSCBAProblem(
        device=Device1D([np.array([[dynamical[i, i]]]) for i in range(3)],
                        [np.array([[dynamical[i, i + 1]]]) for i in range(2)]),
        left=HarmonicReservoir(lead, temperature=temperatures[0]),
        right=HarmonicReservoir(lead, temperature=temperatures[1]),
        vertex=DenseCubicVertex(vertex),
        grid=SignedFrequencyGrid(step=step, n_positive=n_positive), scale=SolverScale(),
    )
    grid = problem.grid
    contacts = [drude_contact(grid, index, temp) for index, temp in zip((0, 2), temperatures)]
    left, right = contacts
    lesser = np.zeros_like(left.lesser)
    greater = lesser.copy()
    history = []
    for _ in range(max_maps):
        retarded = retarded_from_keldysh(lesser, greater, grid, hilbert_backend="fft")
        gr = solve_retarded(problem, left.retarded, right.retarded, retarded)
        green = solve_keldysh_green(gr, left.lesser + right.lesser + lesser,
                                  left.greater + right.greater + greater)
        new_lesser, new_greater = cubic_lesser_greater(
            problem.vertex, green.lesser, green.greater, grid, backend="fft")
        residual = float(max(np.max(np.abs(new_lesser - lesser)),
                             np.max(np.abs(new_greater - greater))))
        if not np.isfinite(residual):
            raise RuntimeError("Nonfinite toy bubble map.")
        history.append(residual)
        if residual < 1e-12:
            break
        lesser = 0.5 * lesser + 0.5 * new_lesser
        greater = 0.5 * greater + 0.5 * new_greater
    else:
        raise RuntimeError("Toy SCBA map budget exhausted without convergence.")
    currents = [contact_heat_current(grid, contact, green, problem.scale) for contact in contacts]
    covariance = (1j * grid.integration_weight * np.sum(green.greater, axis=0)).real
    counterterm = -0.5 * np.einsum("ijk,jk->i", vertex, covariance)
    spectral_moment = grid.integration_weight * np.einsum(
        "w,wij->ij", grid.values, 1j * (green.greater - green.lesser))
    metadata = {
        "step": float(step), "cutoff": float(cutoff), "alpha": float(alpha),
        "temperatures": list(temperatures), "dynamical_matrix": dynamical.tolist(),
        "bath": "Drude-gamma0.25-Omega3-static-renormalization-included",
        "mean_policy": "centered-bubble-with-fixed-local-linear-counterterm",
        "linear_counterterm_coefficients": counterterm.tolist(),
        "means_prescribed_not_solved": True, "maps": len(history),
        "raw_map_residual_history": history, "raw_map_absolute_tolerance": 1e-12,
        "contact_currents": currents, "contact_imbalance": float(sum(currents)),
        "canonical_spectral_moment_max_defect": float(np.max(np.abs(spectral_moment - np.eye(3)))),
        "material_spatial_validation_passed": False, "changes_solver_acceptance": False,
    }
    return {"problem": problem, "green": green, "metadata": metadata}


def measure_currents(state, *, lag_end=150.0, lag_step=0.05):
    if (not np.isfinite(lag_end) or not np.isfinite(lag_step)
            or lag_step <= 0 or lag_end < lag_step):
        raise ValueError("Invalid lag window or step.")
    count = int(round(lag_end / lag_step))
    if not np.isclose(count * lag_step, lag_end, rtol=0, atol=1e-12):
        raise ValueError("lag_end must be an integer multiple of lag_step.")
    times = np.linspace(0, lag_end, count + 1)
    problem, green = state["problem"], state["green"]
    lines = correlations_from_green(problem.grid, green, times)
    covariance = lines["displacement_correlation"][0].real
    mixed = lines["velocity_displacement_correlation"][0].real
    contact = state["metadata"]["contact_currents"][0]
    cuts = []
    for cut in (1, 2):
        operator = build_cubic_cut_current_operator(problem, cut)
        connected = connected_cubic_moments(
            problem.vertex, operator.cubic_indices, times,
            lines["displacement_correlation"], lines["velocity_displacement_correlation"])
        moments = combine_cubic_moments(
            operator.cubic_indices, connected["connected_moments"],
            displacement_mean=np.zeros(3), velocity_mean=np.zeros(3),
            displacement_covariance=covariance, velocity_displacement_covariance=mixed)
        value = operator.evaluate_moments(
            operator.harmonic_moments_from_green(problem.grid, green), moments["total_moments"])
        cuts.append({"cut": cut, "harmonic": value["harmonic"], "cubic": value["cubic"],
                     "total": value["total"], "contact_difference": value["total"] - contact,
                     "harmonic_only_contact_difference": value["harmonic"] - contact,
                     "max_endpoint_integrand": float(np.max(np.abs(connected["endpoint_integrand"]),
                                                             initial=0.0))})
    return {"lag_end": lag_end, "lag_step": lag_step, "lag_points": len(times), "cuts": cuts}


def run_validation():
    """Compare independent refinements with fixed research-scale toy screens."""
    states = {
        "baseline": solve_reference(),
        "frequency_refined": solve_reference(step=0.005),
        "domain_refined": solve_reference(step=0.005, cutoff=12.0),
        "harmonic": solve_reference(alpha=0.0),
        "equilibrium": solve_reference(temperatures=(0.75, 0.75)),
        "reverse_bias": solve_reference(temperatures=(0.5, 1.0)),
    }
    measures = {
        "baseline": measure_currents(states["baseline"]),
        "short_memory": measure_currents(states["baseline"], lag_end=75.0),
        "time_refined": measure_currents(states["baseline"], lag_step=0.025),
        "frequency_refined": measure_currents(states["frequency_refined"], lag_step=0.025),
        "domain_refined": measure_currents(states["domain_refined"], lag_step=0.025),
    }
    for name in ("harmonic", "equilibrium", "reverse_bias"):
        measures[name] = measure_currents(states[name])
    reference_current = abs(states["domain_refined"]["metadata"]["contact_currents"][0])
    comparisons = {}
    for name, first, second in (
        ("memory_window", "short_memory", "baseline"),
        ("lag_step", "baseline", "time_refined"),
        ("frequency_step", "time_refined", "frequency_refined"),
        ("frequency_domain", "frequency_refined", "domain_refined"),
    ):
        a, b = measures[first]["cuts"], measures[second]["cuts"]
        total = max(abs(x["total"] - y["total"]) for x, y in zip(a, b))
        cubic = max(abs(x["cubic"] - y["cubic"]) for x, y in zip(a, b))
        cubic_scale = max(abs(x["cubic"]) for x in b)
        comparisons[name] = {"total_difference": total, "cubic_difference": cubic,
                             "total_contact_relative": total / reference_current,
                             "cubic_relative": cubic / cubic_scale}
    final = measures["domain_refined"]["cuts"]
    continuity = max(abs(c["contact_difference"]) for c in final) / reference_current
    harmonic_omission = min(abs(c["harmonic_only_contact_difference"]) for c in final) / reference_current
    checks = {
        "finite_contact_scale": reference_current > 1e-4,
        "contact_balance": all(abs(s["metadata"]["contact_imbalance"]) < 1e-10 for s in states.values()),
        "forward_bias_direction": states["domain_refined"]["metadata"]["contact_currents"][0] > 0
                                  and all(c["total"] > 0 for c in final),
        "cut_continuity": continuity < 1e-3,
        "harmonic_only_negative_control": harmonic_omission > 1e-3,
        "all_refinements": all(v["total_contact_relative"] < 1e-4 and v["cubic_relative"] < 1e-3
                               for v in comparisons.values()),
        "harmonic_limit": all(abs(c["contact_difference"]) < 1e-10 and c["cubic"] == 0
                              for c in measures["harmonic"]["cuts"]),
        "equilibrium_zero": max(abs(c["total"]) for c in measures["equilibrium"]["cuts"]) < 1e-9
                            and max(abs(c) for c in states["equilibrium"]["metadata"]["contact_currents"]) < 1e-9,
        "reverse_bias_direction": states["reverse_bias"]["metadata"]["contact_currents"][0] < 0
                                  and all(c["total"] < 0 for c in measures["reverse_bias"]["cuts"]),
    }
    return {
        "schema": "negfpy.research.centered-steady-cubic-reference.v1",
        "passed": all(checks.values()), "checks": checks,
        "criteria": {"cut_continuity_rtol": 1e-3, "total_refinement_contact_rtol": 1e-4,
                     "cubic_refinement_rtol": 1e-3},
        "states": {key: value["metadata"] for key, value in states.items()},
        "measurements": measures, "refinement_comparisons": comparisons,
        "maximum_cut_continuity_relative": continuity,
        "minimum_harmonic_only_defect_relative": harmonic_omission,
        "scope": "numerical-centered-bubble-one-vertex-toy-continuity",
        "exact_unrestricted_cubic_stationary_state_claimed": False,
        "infinite_domain_or_tail_bound_claimed": False,
        "material_spatial_validation_passed": False, "nq1_release_certified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_validation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": report["checks"],
                      "maximum_cut_continuity_relative": report["maximum_cut_continuity_relative"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
