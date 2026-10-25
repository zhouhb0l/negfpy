"""In-memory NN site-energy surrogate for an explicitly declared sensitivity.

The frozen local-energy representation study supplies the geometry and exact
derivative design. Its measured remainder is not silently incorporated into
the fitted vertex: the caller must identify this as an approximate research
Hamiltonian, while retaining the unmodified raw source as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from time import monotonic

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsmr
from threadpoolctl import threadpool_limits

from .local_energy import (
    _classes,
    _comparison,
    _design,
    _finite,
    _integer,
    _ratio,
    _stars,
    _summary,
    plan_local_energy_fit,
)


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True, order="C")
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class SiteEnergyStar:
    """One coherent periodic image of a site's relative-coordinate basis."""

    center: int
    class_index: int
    neighbors: np.ndarray
    neighbor_image_integers: np.ndarray
    neighbor_cartesian_vectors: np.ndarray


@dataclass(frozen=True, slots=True)
class SiteEnergyModel:
    """Compute-only model; array fields must never be serialized as output."""

    coefficients: np.ndarray
    monomials: tuple[tuple[int, int, int], ...]
    classes: np.ndarray
    class_labels: np.ndarray
    class_counts: np.ndarray
    stars: tuple[SiteEnergyStar, ...]
    positions: np.ndarray
    lattice: np.ndarray
    geometry: dict
    fit_report: dict


def fit_site_energy_model(fc3, positions, lattice, center_classes, *, neighbors=4,
        max_atoms=128, max_bytes=2*1024**3, max_iterations=500, fit_rtol=1e-12,
        identity_rtol=1e-8, identity_atol=1e-11, deadline_monotonic=None):
    """Fit ONCE using the exact frozen NN site-star basis and LSMR convention.

    Coefficients[C,M] multiply d_p*d_q*d_r divided by factorials of repeated
    coordinate multiplicities; p<=q<=r and p=3*neighbor_slot+Cartesian_axis.
    The original source is unchanged. The returned surrogate deliberately
    omits its measured residual, so it is not an exact raw-IFC representation.
    """
    if not isinstance(fc3, np.ndarray) or fc3.dtype != np.float64 or fc3.ndim != 6:
        raise ValueError("fc3 must be a physical float64[N,N,N,3,3,3] array.")
    n = fc3.shape[0]
    if fc3.shape != (n, n, n, 3, 3, 3):
        raise ValueError("fc3 must have full physical Cartesian tensor shape.")
    if n > _integer(max_atoms, "max_atoms", 2):
        raise ValueError("Atom-count admission exceeded.")
    neighbors = _integer(neighbors, "neighbors")
    max_iterations = _integer(max_iterations, "max_iterations")
    for value, name in ((fit_rtol, "fit_rtol"), (identity_rtol, "identity_rtol"), (identity_atol, "identity_atol")):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    classes, labels, counts = _classes(center_classes, n)
    plan = plan_local_energy_fit(n, len(labels), neighbors=neighbors)
    if plan["peak_owned_bytes"] > _integer(max_bytes, "max_bytes"):
        raise ValueError("Owned-memory admission exceeded.")
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite when supplied.")

    def admit():
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            raise TimeoutError("Site-energy fit reached its bounded deadline.")

    admit()
    _finite(fc3, "fc3")
    original = fc3.copy()
    with threadpool_limits(limits=1):
        selected_stars, geometry = _stars(positions, lattice, classes, neighbors,
            identity_rtol, identity_atol, admit)
        matrix, covered, monomial_count = _design(n, selected_stars, classes, neighbors, admit)
        target = fc3.ravel()[covered]

        def matvec(value):
            admit()
            return matrix@value

        def rmatvec(value):
            admit()
            return matrix.T@value

        operator = LinearOperator(matrix.shape, matvec=matvec, rmatvec=rmatvec, dtype=float)
        solution = lsmr(operator, target, atol=fit_rtol, btol=fit_rtol,
            conlim=1/np.finfo(float).eps, maxiter=max_iterations)
        coefficients, stop, iterations, normr, normar, norma, conda, normx = solution
        _finite(coefficients, "fitted energy coefficients")
        _finite(np.asarray(solution[3:]), "solver scalar diagnostics")
        fitted = np.zeros(fc3.shape, dtype=np.float64)
        fitted.ravel()[covered] = matrix@coefficients
        remainder = fc3-fitted
        outside_mask = np.ones(fc3.size, dtype=bool)
        outside_mask[covered] = False
        source_summary, fit_summary = _summary(fc3), _summary(fitted)
        remainder_summary = _summary(remainder)
        outside_summary = _summary(fc3.ravel()[outside_mask])
        reconstruction = _comparison(fc3, fitted, identity_rtol, identity_atol)
        sum_closure = _comparison(fc3, fitted+remainder, identity_rtol, identity_atol)
        admit()
    if not np.array_equal(fc3, original):
        raise RuntimeError("Raw source changed during site-energy fit.")
    stars = tuple(SiteEnergyStar(center=int(item["center"]), class_index=int(item["class_index"]),
        neighbors=_readonly(item["neighbors"], np.int64),
        neighbor_image_integers=_readonly(item["neighbor_image_integers"], np.int64),
        neighbor_cartesian_vectors=_readonly(item["neighbor_cartesian_vectors"], np.float64))
        for item in geometry["stars"])
    report = {"schema": "negfpy.compute-only-site-energy-model-fit.v1", "plan": plan,
        "source": {**source_summary, "unchanged": True, "fresh_hash_computed": False,
            "center_class_labels": labels.tolist(), "center_class_counts": counts.tolist()},
        "basis": {"kind": "periodic nearest-neighbor site-star cubic energy",
            "neighbors_per_site": neighbors, "monomials_per_class": monomial_count,
            "parameter_count": matrix.shape[1], "sparse_nonzero_entries": matrix.nnz,
            "covered_ordered_tensor_rows": len(covered), "full_ordered_tensor_rows": fc3.size,
            "all_connected_four_atom_clusters_included": False, "rotational_constraints_imposed": False,
            "energy_coefficient_convention": "K*d_p*d_q*d_r divided by repeated-coordinate factorials; p<=q<=r",
            "metric": "unweighted full ordered physical Cartesian FC3 Frobenius norm",
            "image_resolved_raw_bulk_equivalence_established": False},
        "fit": {**fit_summary, "parameter_count": matrix.shape[1], "solver": "scipy.sparse.linalg.lsmr",
            "gauge": "zero-initialized unregularized minimum Euclidean coefficient norm at convergence; bounded iteration approximate",
            "iterations": int(iterations), "iteration_limit": max_iterations, "stop_code": int(stop),
            "bounded_solver_converged": int(stop) in (0, 1, 2, 4, 5),
            "iteration_limit_reached": int(stop) == 7, "requested_solver_tolerance": fit_rtol,
            "solver_residual_norm": float(normr), "solver_normal_residual_norm": float(normar),
            "solver_operator_norm_estimate": float(norma), "solver_condition_estimate": float(conda),
            "coefficient_norm": float(normx), "source_reconstruction_passed": reconstruction["passed"],
            "source_reconstruction": reconstruction},
        "residual": {**remainder_summary,
            "relative_frobenius": _ratio(remainder_summary["frobenius"], source_summary["frobenius"]),
            "raw_covered": _summary(target), "covered_fit_remainder": _summary(remainder.ravel()[covered]),
            "outside_coverage_irreducible_floor": outside_summary,
            "outside_floor_relative_frobenius": _ratio(outside_summary["frobenius"], source_summary["frobenius"]),
            "explicitly_reported": True, "raw_source_retained": True,
            "included_in_surrogate_vertex": False, "sum_closure": sum_closure},
        "scope": {"compute_only_array_model": True, "coefficients_output_to_files": False,
            "source_preserved": True, "surrogate_omits_measured_remainder": True,
            "physical_model_adopted": False, "ifc_repair_performed": False,
            "no_projection_or_clipping": True, "physical_interface_uniquely_recovered": False}}
    return SiteEnergyModel(coefficients=_readonly(coefficients.reshape(len(labels), monomial_count)),
        monomials=tuple(combinations_with_replacement(range(3*neighbors), 3)),
        classes=_readonly(classes, np.int64), class_labels=_readonly(labels, np.int64),
        class_counts=_readonly(counts, np.int64), stars=stars,
        positions=_readonly(positions, np.float64), lattice=_readonly(lattice, np.float64),
        geometry=geometry, fit_report=report)
