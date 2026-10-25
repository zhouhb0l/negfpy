"""Bounded representation study in periodic nearest-neighbor site energies.

The least-squares target is the unmodified physical Cartesian FC3 tensor.  A
site-star energy and its explicit remainder are measured separately; this
module neither replaces IFCs nor adopts a finite interaction boundary.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, permutations, product
from math import comb
from time import monotonic

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, lsmr
from threadpoolctl import threadpool_limits


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}.")
    return int(value)


def plan_local_energy_fit(n_atoms, n_classes=2, *, neighbors=4):
    """Conservative payload plan; the caller's raw source is excluded."""
    n = _integer(n_atoms, "n_atoms", 2)
    c = _integer(n_classes, "n_classes")
    k = _integer(neighbors, "neighbors")
    if c > n or k >= n:
        raise ValueError("Classes cannot exceed atoms and neighbors must be fewer than atoms.")
    d = 3*k
    entries = n*d**3*8
    source = 8*27*n**3
    parameters = c*comb(d+2, 3)
    return {"n_atoms": n, "n_classes": c, "neighbors_per_site": k,
        "relative_coordinates_per_site": d, "monomials_per_class": comb(d+2, 3),
        "parameter_count": parameters, "ordered_source_rows": 27*n**3,
        "coo_entries_before_compression_bound": entries,
        "source_tensor_bytes": source,
        "peak_owned_bytes": 7*source+128*entries+128*parameters+256*n*n+2**20,
        "source_caller_payload_excluded": True, "rss_guarantee": False,
        "dense_design_matrix_allocated": False}


def _summary(value):
    return {"frobenius": float(np.linalg.norm(value)),
        "max_abs": float(np.max(np.abs(value), initial=0.))}


def _ratio(numerator, denominator):
    if denominator == 0:
        return 1. if numerator == 0 else None
    value = float(numerator/denominator)
    return value if np.isfinite(value) else None


def _comparison(first, second, rtol, atol):
    error = _summary(first-second)
    scale = max(float(np.max(np.abs(first), initial=0.)), float(np.max(np.abs(second), initial=0.)))
    allowance = atol+rtol*scale
    return {"max_abs_error": error["max_abs"], "frobenius_error": error["frobenius"],
        "scale": scale, "allowance": allowance,
        "error_to_allowance_ratio": _ratio(error["max_abs"], allowance),
        "passed": bool(error["max_abs"] <= allowance)}


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite.")


def _classes(center_classes, n):
    values = np.asarray(center_classes)
    if values.shape != (n,) or values.dtype.kind not in "iu":
        raise ValueError("center_classes must be one integer primitive-class label per atom.")
    labels, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return inverse, labels, counts


def _stars(positions, lattice, classes, neighbors, rtol, atol, admit):
    """Choose coherent, unique minimum-image stars, never separate pair images."""
    n = len(classes)
    positions, cell = np.asarray(positions, dtype=float), np.asarray(lattice, dtype=float)
    if positions.shape != (n, 3) or cell.shape != (3, 3):
        raise ValueError("positions must be fractional[n,3]; lattice vectors are rows[3,3].")
    _finite(positions, "positions")
    _finite(cell, "lattice")
    singular = float(np.linalg.svd(cell, compute_uv=False)[-1])
    if singular <= 0:
        raise ValueError("Lattice must be nonsingular.")
    translation_radius = float(np.min(np.linalg.norm(cell, axis=1)))
    bound = int(np.ceil(translation_radius/singular))
    if bound > 4:
        raise ValueError("Lattice conditioning exceeds bounded shortest-translation search.")
    shortest = min(float(np.linalg.norm(np.asarray(image)@cell))
        for image in product(range(-bound, bound+1), repeat=3) if any(image))
    delta = positions[None, :, :]-positions[:, None, :]
    origin_images = -np.rint(delta).astype(np.int64)
    wrapped = delta+origin_images
    initial_radius = float(np.max(np.linalg.norm(wrapped@cell, axis=-1)))
    image_bound = int(np.floor(initial_radius/singular+.5+1e-12))
    if image_bound > 3:
        raise ValueError("Minimum-image search exceeds bounded lattice conditioning.")
    best = np.full((n, n), np.inf)
    vectors, images = np.zeros((n, n, 3)), np.zeros((n, n, 3), dtype=np.int64)
    ties = np.zeros((n, n), dtype=np.int64)
    for image in product(range(-image_bound, image_bound+1), repeat=3):
        admit()
        trial = (wrapped+image)@cell
        squared = np.sum(trial*trial, axis=-1)
        equal = np.isclose(squared, best, rtol=rtol, atol=atol**2)
        better = (squared < best) & ~equal
        ties[equal] += 1
        ties[better] = 1
        best[better], vectors[better], images[better] = squared[better], trial[better], (origin_images+image)[better]
    distances = np.sqrt(best)
    np.fill_diagonal(distances, np.inf)
    if np.any(distances <= atol):
        raise ValueError("Distinct atoms coincide periodically.")
    result, metadata, references = [], [], {}
    for center in range(n):
        admit()
        order = np.argsort(distances[center], kind="stable")
        selected = order[:neighbors]
        if neighbors < n-1 and np.isclose(distances[center, order[neighbors-1]], distances[center, order[neighbors]], rtol=rtol, atol=atol):
            raise ValueError("Nearest-neighbor shell boundary is ambiguous.")
        if np.any(ties[center, selected] != 1):
            raise ValueError("A selected neighbor has ambiguous minimum images.")
        # Physical Cartesian vectors fix coordinate order, independently of atom IDs.
        selected = np.asarray(sorted(selected, key=lambda j: tuple(np.round(vectors[center, j], 10))), dtype=int)
        selected_vectors = vectors[center, selected]
        class_index = int(classes[center])
        if class_index in references:
            if not np.allclose(selected_vectors, references[class_index], rtol=rtol, atol=atol):
                raise ValueError("Primitive-class site stars have inconsistent Cartesian environments.")
        else:
            references[class_index] = selected_vectors.copy()
        unwrapped = np.concatenate((np.zeros((1, 3)), selected_vectors))
        diameter = float(np.max(np.linalg.norm(unwrapped[:, None, :]-unwrapped[None, :, :], axis=-1)))
        if diameter >= .5*shortest-(atol+rtol*shortest):
            raise ValueError("Star diameter reaches half the shortest lattice translation; periodic aliasing is ambiguous.")
        result.append((center, selected))
        metadata.append({"center": center, "class_index": class_index,
            "neighbors": selected.tolist(), "neighbor_image_integers": images[center, selected].tolist(),
            "neighbor_cartesian_vectors": selected_vectors.tolist(),
            "bond_lengths": distances[center, selected].tolist(), "unwrapped_diameter": diameter,
            "diameter_over_half_shortest_translation": diameter/(.5*shortest),
            "minimum_images_unique": True})
    return result, {"positions_are_fractional": True, "lattice_vectors_are_rows": True,
        "minimum_image_algorithm": "bounded exact integer-image enumeration using smallest lattice singular value",
        "minimum_image_integer_radius": image_bound, "shortest_translation_integer_radius": bound,
        "shortest_nonzero_lattice_translation": shortest,
        "coordinate_order": "lexicographic Cartesian bond vectors rounded to10decimals; original vectors retained",
        "class_environments_consistent": True, "coherent_unwrapped_stars": True,
        "all_star_diameters_below_half_shortest_translation": True, "stars": metadata}


def _design(n, stars, classes, neighbors, admit):
    """Differentiate site energy exactly into full ordered FC3 rows."""
    d = 3*neighbors
    monomials = list(combinations_with_replacement(range(d), 3))
    monomial_index = {triple: index for index, triple in enumerate(monomials)}
    ordered = np.indices((d, d, d)).reshape(3, -1).T
    columns = np.asarray([monomial_index[tuple(sorted(row))] for row in ordered], dtype=np.int32)
    axes, local_neighbors = ordered % 3, ordered//3
    capacity = n*d**3*8
    row = np.empty(capacity, dtype=np.int64)
    col = np.empty(capacity, dtype=np.int32)
    data = np.empty(capacity, dtype=float)
    cursor = 0
    for center, selected in stars:
        admit()
        atoms = selected[local_neighbors]
        for endpoints in product((0, 1), repeat=3):
            actual = np.where(endpoints, atoms, center)
            flat = (((((actual[:, 0]*n+actual[:, 1])*n+actual[:, 2])*3+axes[:, 0])*3+axes[:, 1])*3+axes[:, 2])
            stop = cursor+len(ordered)
            row[cursor:stop], col[cursor:stop] = flat, columns+int(classes[center])*len(monomials)
            data[cursor:stop] = (-1.)**(3-sum(endpoints))
            cursor = stop
    covered, compressed = np.unique(row, return_inverse=True)
    matrix = coo_matrix((data, (compressed, col)), shape=(len(covered), len(np.unique(classes))*len(monomials))).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    # Some rows can cancel identically across same-class centers. Keeping them
    # is harmless: their zero row contributes its full residual, without bias.
    return matrix, covered, len(monomials)


def _invariants(tensor):
    asr = [{"leg": leg+1, **_summary(np.sum(tensor, axis=leg))} for leg in range(3)]
    errors = []
    for order in permutations(range(3)):
        permutation = tuple(order)+tuple(axis+3 for axis in order)
        errors.append(float(np.max(np.abs(tensor-np.transpose(tensor, permutation)), initial=0.)))
    return {"acoustic_sum_rule_by_leg": asr, "simultaneous_atom_cartesian_permutation_max_abs_error": max(errors)}


def study_local_energy_fit(fc3, positions, lattice, center_classes, *, neighbors=4,
        max_atoms=128, max_bytes=2*1024**3, max_iterations=500, fit_rtol=1e-12,
        identity_rtol=1e-8, identity_atol=1e-11, deadline_monotonic=None):
    """Fit an explicitly restricted conservative energy; retain all residuals.

    For each class the energy is sum K[p,q,r]*d[p]*d[q]*d[r] divided by
    factorials of repeated coordinate multiplicities, with p<=q<=r.  Equivalent
    class stars share K. No coefficient or fitted array leaves this function.
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
            raise TimeoutError("Local-energy study reached its bounded deadline.")

    admit()
    _finite(fc3, "fc3")
    source_copy = fc3.copy()
    with threadpool_limits(limits=1):
        stars, geometry = _stars(positions, lattice, classes, neighbors, identity_rtol, identity_atol, admit)
        matrix, covered, monomial_count = _design(n, stars, classes, neighbors, admit)
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
        coefficients, stop_code, iterations, normr, normar, norma, conda, normx = solution
        _finite(coefficients, "fitted energy coefficients")
        # Explicit C order is required because full-tensor row IDs use C order;
        # zeros_like would inherit a caller's noncontiguous/transposed layout.
        fitted = np.zeros(fc3.shape, dtype=np.float64)
        fitted.ravel()[covered] = matrix@coefficients
        remainder = fc3-fitted
        outside_mask = np.ones(fc3.size, dtype=bool)
        outside_mask[covered] = False
        outside = fc3.ravel()[outside_mask]
        source_summary, fit_summary, residual_summary = _summary(fc3), _summary(fitted), _summary(remainder)
        reconstruction = _comparison(fc3, fitted, identity_rtol, identity_atol)
        closure = _comparison(fc3, fitted+remainder, identity_rtol, identity_atol)
        probes = []
        rng = np.random.default_rng(20261026)
        for index in range(4):
            admit()
            displacement = rng.uniform(-1e-5, 1e-5, size=(n, 3))
            energies = [float(np.einsum("ijkabc,ia,jb,kc", tensor, displacement, displacement, displacement, optimize=True)/6)
                for tensor in (fc3, fitted, remainder)]
            probes.append({"index": index, "maximum_displacement": float(np.max(np.abs(displacement))),
                "raw_energy": energies[0], "site_star_energy": energies[1], "remainder_energy": energies[2],
                "closure_abs_error": abs(energies[0]-energies[1]-energies[2]),
                "remainder_included": True})
        source_invariants, fit_invariants, residual_invariants = _invariants(fc3), _invariants(fitted), _invariants(remainder)
        admit()
    unchanged = bool(np.array_equal(fc3, source_copy))
    if not unchanged:
        raise RuntimeError("Source changed during representation study.")
    return {"schema": "negfpy.periodic-site-star-local-energy-study.v1", "plan": plan,
        "source": {**source_summary, "unchanged": unchanged, "fresh_hash_computed": False,
            "physical_cartesian_unmassweighted": True, "center_class_labels": labels.tolist(),
            "center_class_counts": counts.tolist()}, "geometry": geometry,
        "basis": {"kind": "periodically replicated nearest-neighbor site-star cubic energy",
            "neighbors_per_site": neighbors, "relative_coordinates_per_site": 3*neighbors,
            "monomials_per_class": monomial_count, "parameter_count": matrix.shape[1],
            "class_shared_coefficients": True, "energy_coefficient_convention": "K*d_p*d_q*d_r divided by repeated-coordinate factorials; p<=q<=r",
            "full_ordered_tensor_rows": fc3.size, "covered_rows_including_zero_rows": len(covered),
            "uncovered_rows": int(np.count_nonzero(outside_mask)), "sparse_nonzero_entries": matrix.nnz,
            "metric": "unweighted full ordered physical Cartesian tensor Frobenius norm; omitted zero-design rows contribute constant residual",
            "all_connected_four_atom_clusters_included": False, "rotational_constraints_imposed": False,
            "coherent_periodic_image_tags_retained": True,
            "arbitrary_wavevector_image_resolved_reconstruction_established": False},
        "fit": {**fit_summary, "parameter_count": matrix.shape[1], "solver": "scipy.sparse.linalg.lsmr",
            "gauge": "zero-initialized unregularized minimum Euclidean coefficient norm at convergence; bounded iteration gives an approximation",
            "iterations": int(iterations), "iteration_limit": max_iterations, "stop_code": int(stop_code),
            "bounded_solver_converged": int(stop_code) in (0, 1, 2, 4, 5),
            "iteration_limit_reached": int(stop_code) == 7,
            "requested_solver_tolerance": fit_rtol, "solver_residual_norm": float(normr),
            "solver_normal_residual_norm": float(normar), "solver_operator_norm_estimate": float(norma),
            "solver_condition_estimate": float(conda), "coefficient_norm": float(normx),
            "source_reconstruction_passed": reconstruction["passed"], "source_reconstruction": reconstruction},
        "residual": {**residual_summary, "relative_frobenius": _ratio(residual_summary["frobenius"], source_summary["frobenius"]),
            "raw_covered": _summary(target), "covered_fit_remainder": _summary(remainder.ravel()[covered]),
            "outside_coverage_irreducible_floor": _summary(outside),
            "outside_floor_relative_frobenius": _ratio(float(np.linalg.norm(outside)), source_summary["frobenius"]),
            "retained_explicitly": True, "discarded": False, "sum_closure": closure},
        "invariants": {"raw": source_invariants, "site_star": fit_invariants,
            "remainder": residual_invariants, "bounded_energy_probes": probes},
        "scope": {"representation_study_only": True, "source_preserved": True,
            "remainder_retained": True, "physical_model_adopted": False, "ifc_repair_performed": False,
            "no_projection_or_clipping": True, "no_arrays_output": True,
            "born_green_scba_evaluated": False, "physical_interface_uniquely_recovered": False}}
