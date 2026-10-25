"""Exact bulk cubic tree-bond decomposition with its source remainder retained.

This is a coordinate compatibility study, not an ASR repair or a physical
interface model.  The source equals the relative-coordinate energy plus an
explicit remainder, including when the source does not satisfy ASR.
"""

from __future__ import annotations

from itertools import permutations, product
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer at least{minimum}.")
    return int(value)


def plan_bulk_bond_energy(n_atoms):
    """Conservative owned-payload admission, excluding the caller's source."""
    n = _integer(n_atoms, "n_atoms", 2)
    tensor = 8*27*n**3
    auxiliary = 128*n*n+80*n**3+4096*n
    return {"n_atoms": n, "n_cartesian_dofs": 3*n,
        "source_tensor_bytes": tensor, "tree_bonds": n-1,
        "bond_tensor_bytes": 8*27*(n-1)**3,
        "peak_owned_bytes": 10*tensor+auxiliary,
        "source_caller_payload_excluded": True, "rss_guarantee": False,
        "root_gauges": 2, "energy_probes_per_gauge": 4,
        "dense_matrix_tensor_products_used": False}


def _summary(value):
    return {"frobenius": float(np.linalg.norm(value)),
            "max_abs": float(np.max(np.abs(value), initial=0.))}


def _ratio(numerator, denominator):
    if denominator == 0:
        return 1. if numerator == 0 else None
    result = float(numerator/denominator)
    return result if np.isfinite(result) else None


def _comparison(first, second, rtol, atol):
    difference = first-second
    error = float(np.max(np.abs(difference), initial=0.))
    scale = float(max(np.max(np.abs(first), initial=0.), np.max(np.abs(second), initial=0.)))
    allowance = float(atol+rtol*scale)
    return {"error": error, "scale": scale, "allowance": allowance,
        "error_to_allowance_ratio": _ratio(error, allowance),
        "frobenius_error": float(np.linalg.norm(difference)),
        "passed": bool(error <= allowance), "bookkeeping_only": True}


def _finite(value, label):
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite{label}.")


def _geometry(scaled_positions, lattice, n, admit):
    positions = np.asarray(scaled_positions, dtype=float)
    cell = np.asarray(lattice, dtype=float)
    if positions.shape != (n, 3) or cell.shape != (3, 3):
        raise ValueError("Geometry must supply n_atoms scaled positions and a row-vector3x3 lattice.")
    _finite(positions, "scaled positions")
    _finite(cell, "lattice")
    smallest = float(np.linalg.svd(cell, compute_uv=False)[-1])
    if smallest <= 0:
        raise ValueError("Lattice must be nonsingular.")
    delta = positions[None, :, :]-positions[:, None, :]
    delta -= np.rint(delta)
    best_vectors = delta@cell
    best_squared = np.sum(best_vectors**2, axis=-1)
    # A vector improving the initial candidate obeys |f+n| <= radius/smin.
    # Since f is in[-.5,.5], this finite integer box contains every minimizer.
    radius = float(np.sqrt(np.max(best_squared, initial=0.)))
    image_radius = int(np.floor(radius/smallest+.5+1e-12))
    if image_radius > 3:
        raise ValueError("Minimum-image search exceeds bounded lattice-condition admission.")
    # The first lexicographic minimizer fixes ties reproducibly. Do not retain
    # alternate images as additional graph edges or change source coefficients.
    best_squared.fill(np.inf)
    for image in product(range(-image_radius, image_radius+1), repeat=3):
        admit()
        vectors = (delta+image)@cell
        squared = np.sum(vectors**2, axis=-1)
        changed = squared < best_squared
        best_squared[changed] = squared[changed]
        best_vectors[changed] = vectors[changed]
    distance = np.sqrt(best_squared)
    if np.any(distance[np.triu_indices(n, 1)] <= 0):
        raise ValueError("Distinct atoms must not coincide under periodic images.")
    candidates = sorted((float(distance[i, j]), i, j) for i in range(n) for j in range(i+1, n))
    representatives = list(range(n))

    def find(i):
        while representatives[i] != i:
            representatives[i] = representatives[representatives[i]]
            i = representatives[i]
        return i

    edges = []
    for length, i, j in candidates:
        first, second = find(i), find(j)
        if first != second:
            representatives[first] = second
            edges.append((i, j, length, best_vectors[i, j].copy()))
            if len(edges) == n-1:
                break
    adjacency = [[] for _ in range(n)]
    for i, j, length, vector in edges:
        adjacency[i].append((j, length, vector))
        adjacency[j].append((i, length, -vector))
    for neighbors in adjacency:
        neighbors.sort(key=lambda row: row[0])
    parents, order, tree_distances, unwrapped = _root_tree(adjacency, 0)
    del parents, order
    farthest = int(np.argmax(tree_distances))
    unwrapped_distance = np.linalg.norm(unwrapped[None, :, :]-unwrapped[:, None, :], axis=-1)
    diameter = 0.
    for root in range(n):
        admit()
        _, _, lengths, _ = _root_tree(adjacency, root)
        diameter = max(diameter, float(np.max(lengths)))
    nearest = float(np.min(distance[np.triu_indices(n, 1)]))
    metadata = {"algorithm": "deterministic Kruskal on exact bounded periodic minimum-image distances",
        "lattice_vectors_are_rows": True, "image_integer_radius": image_radius,
        "image_bound_uses_smallest_lattice_singular_value": True,
        "nearest_nonzero_pair_distance": nearest,
        "tree_length": float(sum(x[2] for x in edges)), "weighted_tree_diameter": diameter,
        "second_root": farthest, "second_root_rule": "farthest weighted tree-distance atom from0; smallest index breaks ties",
        "unwrapped_embedding_root": 0,
        "finite_embedding_locality_established_by_edge_unwrapping": True,
        "maximum_unwrapped_pair_distance": float(np.max(unwrapped_distance)),
        "maximum_periodic_pair_distance": float(np.max(distance)),
        "edges": [{"atoms": [i, j], "minimum_image_length": length,
                   "chosen_minimum_image_vector": vector.tolist()}
                  for i, j, length, vector in edges]}
    return adjacency, distance, unwrapped_distance, nearest, metadata


def _root_tree(adjacency, root):
    n = len(adjacency)
    parent = np.full(n, -1, dtype=int)
    parent[root] = root
    distance, positions = np.zeros(n), np.zeros((n, 3))
    order = [root]
    for atom in order:
        for neighbor, length, vector in adjacency[atom]:
            if parent[neighbor] < 0:
                parent[neighbor] = atom
                distance[neighbor] = distance[atom]+length
                positions[neighbor] = positions[atom]+vector
                order.append(neighbor)
    if len(order) != n:
        raise ValueError("Geometric tree is disconnected.")
    return parent, order, distance, positions


def _bond_coefficients(source, parent, order, admit=lambda: None):
    children = [atom for atom in order if atom != order[0]]
    values = source.copy()
    for leg in range(3):
        for atom in reversed(children):
            admit()
            selector = [slice(None)]*6
            selector[leg] = parent[atom]
            target = tuple(selector)
            selector[leg] = atom
            values[target] += values[tuple(selector)]
        values = np.take(values, children, axis=leg)
    _finite(values, "tree bond coefficients")
    return values, children


def _relative_tensor(coefficients, parent, children, admit=lambda: None):
    values = coefficients
    n = len(parent)
    for leg in range(3):
        shape = list(values.shape)
        shape[leg] = n
        expanded = np.zeros(shape, dtype=float)
        for edge, child in enumerate(children):
            admit()
            source_selector = [slice(None)]*6
            source_selector[leg] = edge
            block = values[tuple(source_selector)]
            target = [slice(None)]*6
            target[leg] = child
            expanded[tuple(target)] += block
            target[leg] = parent[child]
            expanded[tuple(target)] -= block
        values = expanded
    _finite(values, "relative-coordinate tensor")
    return values


def _asr(source):
    rows = []
    scale = float(np.max(np.abs(source), initial=0.))
    for leg in range(3):
        residual = np.sum(source, axis=leg)
        bound = float(np.max(np.sum(np.abs(source), axis=leg), initial=0.))
        row = {"leg": leg, **_summary(residual), "absolute_sum_bound_max": bound,
               "max_to_absolute_sum_bound_ratio": _ratio(float(np.max(np.abs(residual))), bound),
               "source_max_abs": scale, "directions": []}
        for direction in range(3):
            row["directions"].append({"cartesian_direction": direction,
                **_summary(np.take(residual, direction, axis=2+leg))})
        rows.append(row)
    return {"legs": rows, "coordinate_convention": "physical Cartesian displacements; independent unit translation in each axis",
            "diagnostic_only": True, "source_repaired": False}


def _permutation(source):
    maximum = 0.
    for perm in permutations(range(3)):
        if perm != (0, 1, 2):
            alternate = source.transpose((*perm, *(3+x for x in perm)))
            maximum = max(maximum, float(np.max(np.abs(source-alternate), initial=0.)))
    return {"max_abs": maximum, "averaging_or_symmetrization_applied": False}


def _locality(coefficients, parent, children, pair_distance, nearest):
    bonds = np.asarray([(parent[c], c) for c in children], dtype=int)
    m = len(bonds)
    edge_diameter = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            atoms = np.r_[bonds[i], bonds[j]]
            edge_diameter[i, j] = np.max(pair_distance[np.ix_(atoms, atoms)])
    diameter = np.maximum(np.maximum(edge_diameter[:, :, None], edge_diameter[:, None, :]),
                          edge_diameter[None, :, :])
    squared = np.einsum("ijkabc,ijkabc->ijk", coefficients, coefficients, optimize=False)
    _finite(squared, "bond-coordinate norm distribution")
    total = float(np.sum(squared))
    limits = nearest*np.array([1., 2., 3.])
    tolerance = 1e-10*nearest
    bins = np.searchsorted(limits+tolerance, diameter, side="left")
    rows = []
    for index, label in enumerate(("at_most_1NN_length", "1_to_2NN_lengths", "2_to_3NN_lengths", "above_3NN_lengths")):
        selected = bins == index
        weight = float(np.sum(squared[selected]))
        rows.append({"label": label, "ordered_coefficient_block_frobenius_squared": weight,
            "fraction_of_this_gauge_bond_coefficient_norm_squared": _ratio(weight, total),
            "ordered_blocks": int(np.count_nonzero(selected)),
            "exact_nonzero_ordered_blocks": int(np.count_nonzero(squared[selected]))})
    return {"diameter_definition": "maximum pair distance among all six ordered bond endpoints",
        "bin_limits": limits.tolist(), "bin_roundoff_allowance": tolerance,
        "NN_means_multiple_of_nearest_pair_length_not_neighbor_shell": True,
        "total_ordered_coefficient_block_frobenius_squared": total,
        "maximum_exact_nonzero_cluster_diameter": float(np.max(diameter[squared > 0], initial=0.)),
        "coefficient_threshold_applied": False, "norm_is_basis_and_gauge_dependent": True,
        "physical_probability_or_scattering_fraction": False, "bins": rows}


def _energy(source, displacement):
    return float(np.einsum("ijkabc,ia,jb,kc->", source, displacement, displacement,
                           displacement, optimize=True)/6.)


def _energy_probes(source, relative, remainder, rtol, atol):
    n = len(source)
    arguments = (np.arange(n)[:, None]+1)*(np.arange(3)[None, :]+1)
    first, second = .001*np.sin(arguments), .001*np.cos(arguments)
    shift = np.array([.0003, -.0002, .0001])
    probes = (("sin", first), ("sin_plus_translation", first+shift),
              ("cos", second), ("cos_plus_translation", second+shift))
    rows = []
    for label, displacement in probes:
        original, bond, residual = (_energy(tensor, displacement) for tensor in (source, relative, remainder))
        rows.append({"label": label, "max_abs_displacement": float(np.max(np.abs(displacement))),
            "source_energy": original, "relative_energy": bond, "remainder_energy": residual,
            "retained_sum_energy": bond+residual,
            "closure": _comparison(np.asarray(original), np.asarray(bond+residual), rtol, atol)})
    return {"count": 4, "max_abs_displacement_bound": .0013,
        "units": "source force-constant units times displacement units cubed",
        "remainder_retained_in_every_sum": True, "rows": rows}


def study_bulk_bond_energy(
    fc3, scaled_positions, lattice, *, max_atoms=128, max_bytes=2*1024**3,
    deadline_monotonic=None, identity_rtol=1e-8, identity_atol=1e-11,
):
    """Return scalar compatibility/locality evidence for two fixed tree gauges."""
    if not isinstance(fc3, np.ndarray) or fc3.dtype != np.float64 or fc3.ndim != 6:
        raise ValueError("fc3 must be a full float64[N,N,N,3,3,3] physical Cartesian array.")
    n = fc3.shape[0]
    if fc3.shape != (n, n, n, 3, 3, 3):
        raise ValueError("fc3 has an invalid full atom/Cartesian shape.")
    if n > _integer(max_atoms, "max_atoms", 2):
        raise ValueError("Bulk bond study exceeds max_atoms.")
    plan = plan_bulk_bond_energy(n)
    if plan["peak_owned_bytes"] > _integer(max_bytes, "max_bytes"):
        raise ValueError("Bulk bond study exceeds owned numeric max_bytes before source reads.")
    for value, name in ((identity_rtol, "identity_rtol"), (identity_atol, "identity_atol")):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative real scalar.")
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite.")

    def admit():
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            raise TimeoutError("Bulk bond study time admission exhausted.")

    admit()
    _finite(fc3, "source FC3")
    source = fc3.copy()
    source.setflags(write=False)
    adjacency, periodic_distance, unwrapped_distance, nearest, geometry = _geometry(
        scaled_positions, lattice, n, admit)
    gauges = []
    first_relative = None
    gauge_difference = None
    with threadpool_limits(limits=1, user_api="blas"):
        for root in (0, geometry["second_root"]):
            admit()
            parent, order, _, _ = _root_tree(adjacency, root)
            coefficients, children = _bond_coefficients(source, parent, order, admit)
            relative = _relative_tensor(coefficients, parent, children, admit)
            remainder = source-relative
            _finite(remainder, "retained source remainder")
            closure = _comparison(relative+remainder, source, identity_rtol, identity_atol)
            if not closure["passed"]:
                raise ValueError("Bulk source relative-plus-remainder bookkeeping does not close.")
            record = {"root_atom": root, "tree_edges_unchanged": True,
                "relative_tensor": _summary(relative), "remainder": _summary(remainder),
                "remainder_to_source_frobenius_ratio": _ratio(float(np.linalg.norm(remainder)), float(np.linalg.norm(source))),
                "relative_plus_remainder_closure": closure,
                "relative_ASR": _asr(relative), "relative_permutation": _permutation(relative),
                "bond_coefficients": _summary(coefficients),
                "unwrapped_physical_locality": _locality(coefficients, parent, children, unwrapped_distance, nearest),
                "periodic_metric_locality": _locality(coefficients, parent, children, periodic_distance, nearest),
                "energy_probes": _energy_probes(source, relative, remainder, identity_rtol, identity_atol)}
            gauges.append(record)
            if first_relative is None:
                first_relative = relative.copy()
            else:
                gauge_difference = _summary(relative-first_relative)
            del coefficients, relative, remainder
    unchanged = bool(np.array_equal(fc3, source))
    if not unchanged:
        raise ValueError("The source FC3 changed during the bulk compatibility study.")
    return {"schema": "negfpy.bulk-bond-energy-compatibility.v1",
        "diagnostic_only": True, "plan": plan, "geometry": geometry,
        "source": _summary(source), "source_ASR": _asr(source),
        "source_permutation": _permutation(source), "gauges": gauges,
        "relative_tensor_difference_between_gauges": gauge_difference,
        "source_unchanged": unchanged, "fresh_array_hashing_performed": False,
        "raw_source_equals_relative_plus_retained_remainder": True,
        "remainder_discarded": False, "relative_part_adopted_as_model": False,
        "physical_interface_recovered": False, "finite_energy_cluster_cut_applied": False,
        "Born_or_Green_evaluated": False, "SCBA_evaluated": False,
        "source_arrays_exported": False, "physical_guards_or_convergence_thresholds_changed": False}


__all__ = ["plan_bulk_bond_energy", "study_bulk_bond_energy"]
