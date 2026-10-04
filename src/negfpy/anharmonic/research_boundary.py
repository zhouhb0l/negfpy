"""Explicit experimental finite-range acoustic completion of a cubic vertex.

This is an opt-in *change of Hamiltonian*, not IFC ingestion or certification.
It never changes the default material mapper or claims to recover a unique
physical surface. See ``docs/research-acoustic-boundary.md`` for the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import fsum
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr

from ._canonical import domain_digest
from .sparse_cubic import SparseCubicVertex


MODEL = "finite-range-minimum-change-acoustic-completion-v1"


@dataclass(frozen=True)
class AcousticBoundaryCompletion:
    vertex: SparseCubicVertex
    report: dict[str, Any]


def _ratio(top: float, bottom: float) -> float:
    return float(top / bottom) if bottom else (0.0 if top == 0.0 else float("inf"))


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _translation_residual(vertex: SparseCubicVertex, translations: np.ndarray) -> dict[str, Any]:
    """All-leg ASR; normalization is a global absolute-sum bound per leg."""
    records = []
    for leg in range(3):
        free = [axis for axis in range(3) if axis != leg]
        shape = (3, vertex.ndof, vertex.ndof)
        residual = np.zeros(shape)
        bound = np.zeros(shape)
        idx = vertex.indices
        selector = (idx[:, leg] % 3, idx[:, free[0]], idx[:, free[1]])
        values = vertex.coefficients * translations[idx[:, leg]]
        np.add.at(residual, selector, values)
        np.add.at(bound, selector, np.abs(values))
        maximum = float(np.max(np.abs(residual), initial=0.0))
        denominator = float(np.max(bound, initial=0.0))
        records.append({"leg": leg, "max_abs": maximum, "absolute_sum_bound": denominator,
                        "relative_max": _ratio(maximum, denominator)})
    return {"legs": records, "relative_max": max(record["relative_max"] for record in records)}


def complete_acoustic_boundary(
    vertex: SparseCubicVertex,
    *,
    atom_masses: Any,
    dof_per_layer: int,
    boundary_layers: tuple[int, ...],
    asr_tolerance: float = 1e-10,
    max_iterations: int = 3000,
    max_variables: int = 500_000,
) -> AcousticBoundaryCompletion:
    """Return the closest permutation-symmetric acoustic tensor on local support.

    Coordinates are mass normalized, in Cartesian atom triples. The translation
    vector is proportional to sqrt(mass). Existing *atom-triplet* support is
    fixed, but zero Cartesian entries within those blocks can become nonzero.
    The objective is the full ordered-tensor Frobenius norm of the change;
    multiplicities are included in the least-squares coordinates. No threshold
    is used to drop coefficients. ``boundary_layers`` labels the correction
    report only; it does not change which coefficients are fitted.

    The bounded-trial flags are conservative research screening, not scientific
    certification or permission to resume a checkpoint from the old model.
    """
    if not isinstance(vertex, SparseCubicVertex):
        raise TypeError("vertex must be a SparseCubicVertex")
    if vertex.ndof % 3:
        raise ValueError("Cartesian atom triples are required")
    dof_per_layer = _integer(dof_per_layer, "dof_per_layer")
    max_iterations = _integer(max_iterations, "max_iterations")
    max_variables = _integer(max_variables, "max_variables")
    if dof_per_layer % 3 or vertex.ndof % dof_per_layer:
        raise ValueError("dof_per_layer must partition the Cartesian device")
    if not isinstance(boundary_layers, tuple) or not boundary_layers:
        raise ValueError("boundary_layers must be an explicit nonempty tuple")
    boundary_layers = tuple(_integer(layer, "boundary layer", 0) for layer in boundary_layers)
    if len(set(boundary_layers)) != len(boundary_layers) or max(boundary_layers) >= vertex.ndof // dof_per_layer:
        raise ValueError("boundary_layers must be unique layers in the device")
    if isinstance(asr_tolerance, (bool, np.bool_)) or not np.isfinite(asr_tolerance) or not 0 < asr_tolerance <= 1e-8:
        raise ValueError("asr_tolerance must be in (0, 1e-8]")
    raw_masses = np.asarray(atom_masses)
    if raw_masses.dtype.kind not in "fiu" or raw_masses.shape != (vertex.ndof // 3,):
        raise ValueError("atom_masses must give one real mass per atom")
    masses = np.asarray(raw_masses, dtype=np.float64)
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise ValueError("atom_masses must be finite and positive")
    # Common mass normalization changes neither the nullspace nor its model.
    translations = np.repeat(np.sqrt(masses / np.max(masses)), 3)
    source_audit = _translation_residual(vertex, translations)

    # Preserve every source atom triplet; never connect distant atoms that did
    # not occur together. Cartesian-zero filling is declared and measured.
    atom_blocks = sorted({tuple(sorted(map(int, row // 3))) for row in vertex.indices})
    keys = sorted({tuple(sorted(3 * atom + axis for atom, axis in zip(block, axes)))
                   for block in atom_blocks for axes in product(range(3), repeat=3)})
    if len(keys) > max_variables:
        raise ValueError("Acoustic completion exceeds the explicit variable budget")
    source_groups: dict[tuple[int, int, int], list[float]] = {}
    for row, coefficient in zip(vertex.indices, vertex.coefficients):
        source_groups.setdefault(tuple(sorted(map(int, row))), []).append(float(coefficient))
    ordered_keys = [tuple(sorted(set(permutations(key)))) for key in keys]
    multiplicity = np.asarray([len(items) for items in ordered_keys], dtype=np.float64)
    roots = np.sqrt(multiplicity)
    original = np.asarray([fsum(source_groups.get(key, ())) / multiplicity[pos]
                           for pos, key in enumerate(keys)], dtype=np.float64)

    # Because the tensor is exactly symmetric, one leg and unordered free pairs
    # suffice. Each summand appears once, even for repeated tensor indices.
    equations: dict[tuple[int, int, int], int] = {}
    rows, columns, coefficients = [], [], []
    for column, items in enumerate(ordered_keys):
        for i, j, k in items:
            if i > j:
                continue
            equation = (i, j, k % 3)
            row = equations.setdefault(equation, len(equations))
            rows.append(row)
            columns.append(column)
            coefficients.append(translations[k] / roots[column])
    matrix = coo_matrix((coefficients, (rows, columns)), shape=(len(equations), len(keys))).tocsr()
    reference = original * roots
    defect = matrix @ reference
    if len(keys) and np.any(defect):
        solution = lsmr(matrix, -defect, atol=1e-13, btol=1e-13,
                        conlim=1e12, maxiter=max_iterations)
        delta = solution[0] / roots
        solve = {"stop_code": int(solution[1]), "iterations": int(solution[2]),
                 "residual_l2": float(solution[3]), "normal_residual_l2": float(solution[4]),
                 "estimated_condition": float(solution[6])}
    else:
        delta = np.zeros_like(original)
        solve = {"stop_code": 0, "iterations": 0, "residual_l2": 0.0,
                 "normal_residual_l2": 0.0, "estimated_condition": 0.0}
    completed = original + delta
    output = sorted((key, float(completed[column])) for column, items in enumerate(ordered_keys)
                    for key in items if completed[column] != 0.0)
    indices = np.asarray([item[0] for item in output], dtype=np.int64).reshape(-1, 3)
    values = np.asarray([item[1] for item in output], dtype=np.float64)
    configuration = {"model": MODEL, "source_cubic_physics_sha256": vertex.cubic_physics_sha256,
                     "source_vertex_storage_sha256": vertex.vertex_storage_sha256,
                     "relative_atom_masses": (masses / np.max(masses)).tolist(),
                     "dof_per_layer": dof_per_layer, "boundary_layers": list(boundary_layers),
                     "asr_tolerance": asr_tolerance, "max_iterations": max_iterations,
                     "max_variables": max_variables, "least_squares_atol_btol": 1e-13,
                     "support": "existing atom-triplet blocks, all Cartesian components",
                     "objective": "minimum full ordered-tensor Frobenius change"}
    identity = domain_digest("NEGFPY-RESEARCH-ACOUSTIC-COMPLETION", configuration)
    result = SparseCubicVertex(vertex.ndof, indices, values, metadata={
        "schema_version": "negfpy.research.acoustic-boundary.v1",
        "research_only": True, "boundary_model": MODEL, "model_identity_sha256": identity,
        "source_cubic_physics_sha256": vertex.cubic_physics_sha256,
        "source_vertex_storage_sha256": vertex.vertex_storage_sha256,
        "corrected": True, "symmetrized": True, "raw_ifc_modified": False,
        "release_certified": False, "configuration": configuration,
    })
    result_audit = _translation_residual(result, translations)
    if result_audit["relative_max"] > asr_tolerance:
        raise RuntimeError("Finite-range completion did not satisfy its acoustic tolerance: "
                           f"{result_audit['relative_max']:.3e} > {asr_tolerance:.3e}; "
                           f"LSMR {solve}")

    # Include source last-bit permutation differences in all measured changes.
    source_lookup = {tuple(map(int, key)): float(value) for key, value in zip(vertex.indices, vertex.coefficients)}
    output_lookup = dict(output)
    all_keys = sorted(set(source_lookup) | set(output_lookup))
    actual_old = np.asarray([source_lookup.get(key, 0.0) for key in all_keys])
    actual_new = np.asarray([output_lookup.get(key, 0.0) for key in all_keys])
    correction = actual_new - actual_old
    touches_boundary = np.asarray([any(index // dof_per_layer in boundary_layers for index in key)
                                   for key in all_keys], dtype=bool)
    def correction_record(mask):
        before, change = actual_old[mask], correction[mask]
        return {"source_frobenius": float(np.linalg.norm(before)),
                "correction_frobenius": float(np.linalg.norm(change)),
                "relative_frobenius": _ratio(float(np.linalg.norm(change)), float(np.linalg.norm(before))),
                "max_abs_change": float(np.max(np.abs(change), initial=0.0))}
    changes = {"total": correction_record(np.ones(len(all_keys), dtype=bool)),
               "boundary_touching": correction_record(touches_boundary),
               "interior_only": correction_record(~touches_boundary)}
    norm_ratio = _ratio(float(np.linalg.norm(actual_new)), float(np.linalg.norm(actual_old))) if vertex.nnz else 1.0
    nnz_ratio = _ratio(result.nnz, vertex.nnz) if vertex.nnz else 1.0
    # Kernel cost is closer to products of each external leg's pair support than
    # total nnz. This transparent proxy catches expensive zero-block filling.
    old_rows = np.bincount(vertex.indices[:, 0], minlength=vertex.ndof).astype(np.float64)
    new_rows = np.bincount(result.indices[:, 0], minlength=vertex.ndof).astype(np.float64)
    cost_ratio = _ratio(float(np.sum(new_rows ** 2)), float(np.sum(old_rows ** 2))) if vertex.nnz else 1.0
    checks = {"total_change_at_most_25_percent": changes["total"]["relative_frobenius"] <= .25,
              "interior_change_at_most_5_percent": changes["interior_only"]["relative_frobenius"] <= .05,
              "tensor_norm_retained": .75 <= norm_ratio <= 1.25,
              "nnz_growth_at_most_2x": nnz_ratio <= 2.0,
              "pair_work_proxy_at_most_4x": cost_ratio <= 4.0}
    report = {"schema_version": "negfpy.research.acoustic-boundary-report.v1",
              "model": MODEL, "model_identity_sha256": identity, "configuration": configuration,
              "source_translation": source_audit, "completed_translation": result_audit,
              "source_physics_sha256": vertex.cubic_physics_sha256,
              "completed_physics_sha256": result.cubic_physics_sha256,
              "atom_triplet_block_count": len(atom_blocks), "added_atom_triplet_blocks": 0,
              "variables": len(keys), "constraints": len(equations), "solver": solve,
              "correction": changes, "source_nnz": vertex.nnz, "completed_nnz": result.nnz,
              "nnz_ratio": nnz_ratio, "pair_work_proxy_ratio": cost_ratio,
              "completed_to_source_frobenius_ratio": norm_ratio,
              "bounded_trial_screen": checks, "bounded_trial_screen_passed": all(checks.values()),
              "screen_note": "Heuristic distortion/cost screen, not physical validation or convergence evidence",
              "raw_ifc_modified": False, "unique_physical_boundary_claimed": False,
              "permutation_symmetry": dict(result.permutation_symmetry)}
    return AcousticBoundaryCompletion(result, report)


__all__ = ["AcousticBoundaryCompletion", "complete_acoustic_boundary"]
