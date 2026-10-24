"""Scalar audit of acoustic cancellation lost by an all-leg interaction crop.

Only the two uncontracted coordinates are finite.  The third source-image leg
is summed on its complete periodic support, including longitudinal positions
outside the device.  No IFC, field or translation is projected or repaired.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from ._canonical import domain_digest
from .layered import LayeredDeviceMapping
from .mapped_fc3 import (
    MassNormalizedTranslationalFC3,
    _effective_terms,
    _layer_tuple,
    _normalized_payload,
    _positive_integer,
    _validate_mapping_identity,
)
from .momentum import UniformBZMesh
from .transverse_mapping import _kpoints
from .transverse_material import _geometry, _position
from .transverse_packed import PackedTransverseCubicVertex
from .transverse_q import difference_q_indices
from .types import SolverScale

_RTOL = 1e-8
_ATOL = 1e-11
_LEGS = ("external_a", "internal_c", "internal_d")


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    value = numerator / denominator
    return float(value) if np.isfinite(value) else None


def _norms(matrix: np.ndarray) -> dict:
    absolute = np.abs(matrix)
    location = np.unravel_index(int(np.argmax(absolute)), matrix.shape)
    return {
        "max_abs": float(absolute[location]),
        "frobenius": float(np.linalg.norm(matrix)),
        "max_location": [int(x) for x in location],
    }


def _comparison(left: np.ndarray, right: np.ndarray) -> dict:
    error = np.abs(left - right)
    allowance = _ATOL + _RTOL * np.maximum(np.abs(left), np.abs(right))
    ratios = error / allowance
    location = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
    return {
        "passed": bool(np.all(error <= allowance)),
        "rtol": _RTOL,
        "atol": _ATOL,
        "max_abs_error": float(np.max(error)),
        "max_error_allowance_ratio": float(ratios[location]),
        "ratio_location": [int(x) for x in location],
        "error_at_ratio_location": float(error[location]),
        "allowance_at_ratio_location": float(allowance[location]),
    }


def audit_transverse_boundary_cancellation(
    mesh: UniformBZMesh,
    fc3: MassNormalizedTranslationalFC3,
    device_mapping: LayeredDeviceMapping,
    *,
    n_device_layers: int,
    interaction_layers: Sequence[int],
    scale: SolverScale,
    reference_vertex: PackedTransverseCubicVertex | None = None,
    max_workspace_bytes: int = 256 * 1024**2,
    max_geometry_evaluations: int = 1_000_000,
    max_accumulation_work: int = 1_000_000_000,
    deadline_monotonic: float | None = None,
) -> dict:
    """Compare complete, cropped and omitted translation contractions.

    Output contains scalar records only.  The optional saved production vertex
    determines the retained coordinate order and independently checks the
    streamed all-leg-crop contraction.  Without it the domain is every DOF in
    the interaction crop.  Comparisons use the unchanged 1e-8/1e-11 identity
    tolerances; a nonzero complete-source acoustic residual is diagnostic.

    ``max_accumulation_work`` bounds conservative complex multiply/add visits:
    two 27-component updates per q per retained-two-leg source anchor, plus
    three directional contractions of each reference cubic block.  It is not
    a flop or runtime guarantee.  Numeric workspace excludes caller-owned
    source/reference payloads, Python objects and identity-audit internals.
    """
    started = time.monotonic()

    def check_time() -> None:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("Boundary cancellation diagnostic deadline reached.")

    check_time()
    if not isinstance(mesh, UniformBZMesh) or mesh.shape[2] != 1:
        raise TypeError("mesh must be a transverse UniformBZMesh.")
    if not isinstance(fc3, MassNormalizedTranslationalFC3):
        raise TypeError("fc3 must be a MassNormalizedTranslationalFC3.")
    if not isinstance(device_mapping, LayeredDeviceMapping):
        raise TypeError("device_mapping must be a LayeredDeviceMapping.")
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite.")
    layers = _positive_integer(n_device_layers, name="n_device_layers")
    workspace_cap = _positive_integer(max_workspace_bytes, name="max_workspace_bytes")
    geometry_cap = _positive_integer(
        max_geometry_evaluations, name="max_geometry_evaluations"
    )
    work_cap = _positive_integer(max_accumulation_work, name="max_accumulation_work")
    crop = _layer_tuple(
        interaction_layers, name="interaction_layers", n_device_layers=layers
    )
    mapping = device_mapping
    nq, p = mesh.n_qpoints, mapping.principal_cells
    full_ndof = layers * mapping.n_principal_layer_dofs
    minimum, maximum = crop[0] * p, (crop[-1] + 1) * p - 1
    crop_start = crop[0] * mapping.n_principal_layer_dofs
    crop_end = (crop[-1] + 1) * mapping.n_principal_layer_dofs
    if reference_vertex is not None:
        if not isinstance(reference_vertex, PackedTransverseCubicVertex):
            raise TypeError("reference_vertex must be a PackedTransverseCubicVertex.")
        if reference_vertex.mesh.shape != mesh.shape or reference_vertex.full_ndof != full_ndof:
            raise ValueError("Reference vertex mesh/full-device domain differs.")
        active = np.asarray(reference_vertex.active_dofs)
        if np.any((active < crop_start) | (active >= crop_end)):
            raise ValueError("Reference active coordinates must lie inside the crop.")
    else:
        active = np.arange(crop_start, crop_end, dtype=np.int64)
    a = len(active)
    if not a:
        raise ValueError("Boundary cancellation needs a nonempty retained domain.")
    # Three families, three legs, q, three translation directions, two active
    # coordinates.  Additional matrix-sized comparison/contraction scratch and
    # small phase/source-expansion buffers are included conservatively.
    family_bytes = 3 * 3 * nq * 3 * a**2 * 16
    auxiliary_bytes = 48 * len(fc3.terms) + 4096 * nq**2 + 512 * a + 65536
    workspace_plan = family_bytes + 160 * a**2 + auxiliary_bytes
    if workspace_plan > workspace_cap:
        raise ValueError("Boundary numeric workspace plan exceeds max_workspace_bytes.")
    source_legs = 3 if fc3.make_r0_average else 1
    if source_legs * len(fc3.terms) > geometry_cap:
        raise ValueError("Source expansion exceeds max_geometry_evaluations.")
    _geometry(mapping)
    _validate_mapping_identity(fc3, mapping)
    if fc3.output_units != "J m^-3 kg^-3/2":
        raise ValueError("FC3 must use mass-normalized physical SI units.")
    if domain_digest(
        "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3", _normalized_payload(fc3)
    ) != fc3.content_sha256:
        raise ValueError("Normalized FC3 content no longer matches its identity.")
    check_time()
    effective = _effective_terms(fc3)
    representatives = mapping.direct_coset_representatives
    nonzero_terms = sum(bool(np.any(term.values != 0)) for term in effective)
    geometry_evaluations = 2 * nonzero_terms * len(representatives)
    if geometry_evaluations > geometry_cap:
        raise ValueError("Two-pass source mapping exceeds max_geometry_evaluations.")
    primitive_masses = np.asarray(mapping._crystal.primitive.masses)
    basis_masses = primitive_masses[mapping.basis_primitive_atoms]
    total_mass = float(np.sum(basis_masses) * p * layers)
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Full-device translation mass normalization is invalid.")
    translation_basis = np.sqrt(basis_masses / total_mass)
    active_atoms = active // 3
    active_basis = active_atoms % mapping.n_transport_basis_atoms
    active_translation = np.zeros((a, 3))
    active_translation[np.arange(a), active % 3] = translation_basis[active_basis]
    full_translation_gram = np.eye(3)
    restricted_gram = active_translation.T @ active_translation
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        solver_factor = float(np.sqrt(scale.hbar) / np.float64(scale.omega0)**2.5)
    if not np.isfinite(solver_factor) or solver_factor <= 0:
        raise ValueError("FC3 solver conversion is not representable.")
    lookup = {int(dof): i for i, dof in enumerate(active)}

    def images():
        for term_index, term in enumerate(effective):
            if term_index % 256 == 0:
                check_time()
            if not np.any(term.values != 0):
                continue
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                weighted = term.values * solver_factor
                weighted = weighted * (
                    term.multiplicity_weight * term.anchor_weight * term.r0_leg_factor
                )
            if not np.isfinite(weighted).all() or np.any((term.values != 0) & (weighted == 0)):
                raise ValueError("Weighted source block is not finitely representable.")
            for representative in representatives:
                mapped = tuple(
                    _position(atom, position + representative, mapping)
                    for atom, position in zip(term.primitive_indices, term.positions, strict=True)
                )
                yield mapped, weighted

    # Count intervals without storing a source/anchor schedule or numeric field.
    counts = np.zeros((3, 3), dtype=np.int64)  # complete, cropped, omitted per leg
    outside_device = np.zeros(3, dtype=np.int64)
    for mapped, _ in images():
        cells = [int(position[0]) for position, _ in mapped]
        for leg in range(3):
            remaining = [x for x in range(3) if x != leg]
            low = max(minimum - cells[x] for x in remaining)
            high = min(maximum - cells[x] for x in remaining)
            count = max(0, high - low + 1)
            crop_low = max(low, minimum - cells[leg])
            crop_high = min(high, maximum - cells[leg])
            cropped = max(0, crop_high - crop_low + 1)
            device_low, device_high = max(low, -cells[leg]), min(high, layers * p - 1 - cells[leg])
            in_device = max(0, device_high - device_low + 1)
            counts[leg] += (count, cropped, count - cropped)
            outside_device[leg] += count - in_device
    streaming_work = int(np.sum(counts[:, 0])) * nq * 27 * 2
    reference_work = 3 * nq * 3 * a**3 if reference_vertex is not None else 0
    planned_work = streaming_work + reference_work
    if planned_work > work_cap:
        raise ValueError("Boundary contraction work plan exceeds max_accumulation_work.")
    check_time()
    matrices = np.zeros((3, 3, nq, 3, a, a), dtype=np.complex128)
    bounds = np.zeros((3, 3, 3))  # family, leg, direction: triangle bound
    kpoints = _kpoints(mesh)
    routes = difference_q_indices(mesh, max_bytes=workspace_cap)
    qindices, directions = np.arange(nq), np.arange(3)
    # For V[q,p] with momenta(-q,p,q-p), these are the three zero-leg routes.
    q_routes = (np.zeros(nq, dtype=int), qindices, qindices)
    p_routes = (qindices, np.zeros(nq, dtype=int), qindices)
    for mapped, weighted in images():
        cells = [int(position[0]) for position, _ in mapped]
        second = tuple(
            (int(b) - int(c)) % period
            for c, b, period in zip(mapped[0][0][1:], mapped[1][0][1:], mesh.shape[:2], strict=True)
        )
        third = tuple(
            (int(b) - int(c)) % period
            for c, b, period in zip(mapped[0][0][1:], mapped[2][0][1:], mesh.shape[:2], strict=True)
        )
        phase_p = np.exp(1j * (kpoints @ second))
        phase_partner = np.exp(1j * (kpoints @ third))
        for leg in range(3):
            remaining = [x for x in range(3) if x != leg]
            low = max(minimum - cells[x] for x in remaining)
            high = min(maximum - cells[x] for x in remaining)
            phase = phase_p[p_routes[leg]] * phase_partner[routes[q_routes[leg], p_routes[leg]]]
            # The contraction axis becomes direction; the other two axes stay
            # in their original vertex order.  All 4q x 3 x 3 x 3 updates batch.
            collapsed = np.moveaxis(weighted, leg, 0) * translation_basis[mapped[leg][1]]
            block = phase[:, None, None, None] * collapsed[None]
            for anchor in range(low, high + 1):
                coordinates = []
                components = []
                for other in remaining:
                    layer, subcell = divmod(cells[other] + anchor, p)
                    base = layer * mapping.n_principal_layer_dofs + 3 * (
                        subcell * mapping.n_transport_basis_atoms + mapped[other][1]
                    )
                    component = [i for i in range(3) if base + i in lookup]
                    components.append(component)
                    coordinates.append([lookup[base + i] for i in component])
                if not coordinates[0] or not coordinates[1]:
                    continue
                selection = np.ix_(qindices, directions, coordinates[0], coordinates[1])
                contribution = block[:, :, components[0]][:, :, :, components[1]]
                # Complete and omitted are accumulated directly, independently
                # of the cropped sum; omitted is never complete minus cropped.
                matrices[0, leg][selection] += contribution
                inside = minimum <= cells[leg] + anchor <= maximum
                family = 1 if inside else 2
                matrices[family, leg][selection] += contribution
                image_norms = np.linalg.norm(contribution[0].reshape(3, -1), axis=1)
                bounds[0, leg] += image_norms
                bounds[family, leg] += image_norms
    if not np.isfinite(matrices).all() or not np.isfinite(bounds).all():
        raise ValueError("Boundary contraction accumulation is nonfinite.")
    records = []
    reference_max_ratio = 0.0
    closure_max_ratio = 0.0
    for leg, name in enumerate(_LEGS):
        for q in range(nq):
            check_time()
            q_index, p_index = int(q_routes[leg][q]), int(p_routes[leg][q])
            reference_block = None if reference_vertex is None else reference_vertex.coefficients[q_index, p_index]
            reference_block_norm = None if reference_block is None else float(np.linalg.norm(reference_block))
            for direction in range(3):
                complete, cropped, omitted = matrices[:, leg, q, direction]
                closure = _comparison(complete, cropped + omitted)
                closure_max_ratio = max(closure_max_ratio, closure["max_error_allowance_ratio"])
                if not closure["passed"]:
                    raise ValueError("Complete/cropped/omitted matrix closure failed.")
                norms = [_norms(matrix) for matrix in (complete, cropped, omitted)]
                reference = None
                if reference_block is not None:
                    expression = ("abc,a->bc", "abc,b->ac", "abc,c->ab")[leg]
                    contracted = np.einsum(
                        expression, reference_block, active_translation[:, direction], optimize=False
                    )
                    match = _comparison(cropped, contracted)
                    reference_max_ratio = max(reference_max_ratio, match["max_error_allowance_ratio"])
                    if not match["passed"]:
                        raise ValueError("Streamed crop does not match saved production vertex contraction.")
                    cs_scale = reference_block_norm * float(np.linalg.norm(active_translation[:, direction]))
                    reference = {
                        "comparison": match,
                        "contraction": _norms(contracted),
                        "vertex_frobenius": reference_block_norm,
                        "restricted_translation_norm": float(np.linalg.norm(active_translation[:, direction])),
                        "cauchy_schwarz_scale": cs_scale,
                        "cropped_relative_cs_scale": _ratio(norms[1]["frobenius"], cs_scale),
                    }
                records.append({
                    "leg": leg,
                    "leg_name": name,
                    "other_q_index": q,
                    "vertex_q_index": q_index,
                    "vertex_p_index": p_index,
                    "direction": direction,
                    "complete": norms[0],
                    "cropped": norms[1],
                    "omitted": norms[2],
                    "closure": closure,
                    "reference": reference,
                    "absolute_image_frobenius_bounds": {
                        family: float(bounds[index, leg, direction])
                        for index, family in enumerate(("complete", "cropped", "omitted"))
                    },
                    "complete_to_crop_frobenius_ratio": _ratio(norms[0]["frobenius"], norms[1]["frobenius"]),
                    "complete_relative_image_bound": _ratio(norms[0]["frobenius"], float(bounds[0, leg, direction])),
                    "omitted_to_crop_frobenius_ratio": _ratio(norms[2]["frobenius"], norms[1]["frobenius"]),
                })
    check_time()
    return {
        "schema": "negfpy.transverse-boundary-cancellation.v1",
        "diagnostic_only": True,
        "accepted_restart": False,
        "records": records,
        "record_count": len(records),
        "retained_dofs": [int(x) for x in active],
        "retained_ndof": a,
        "full_ndof": full_ndof,
        "n_device_layers": layers,
        "interaction_layers": list(crop),
        "mesh_shape": list(mesh.shape),
        "translation_normalization": {
            "full_device_mass_sum": total_mass,
            "primitive_mass_units": fc3.source_mass_units,
            "basis_masses": [float(x) for x in basis_masses],
            "full_translation_gram": full_translation_gram.tolist(),
            "restricted_translation_gram": restricted_gram.tolist(),
            "outside_source_uses_same_normalization": True,
            "restricted_translation_renormalized": False,
        },
        "source_identity": {
            "normalized_fc3_sha256": fc3.content_sha256,
            "source_fc3_sha256": fc3.source_fc3_content_sha256,
            "source_translational_sha256": fc3.source_translational_sha256,
            "source_certification_sha256": fc3.source_certification_sha256,
            "source_fc2_sha256": mapping.fc2_content_sha256,
            "primitive_crystal_sha256": mapping.primitive_crystal_content_sha256,
            "layered_geometry_sha256": mapping.layered_geometry_sha256,
        },
        "source_weighting": {
            "solver_factor": solver_factor,
            "multiplicity_anchor_r0_weights_applied_once": True,
            "make_r0_average": fc3.make_r0_average,
            "coupling_applied": False,
            "longitudinal_wrapping": False,
            "contracted_leg_clamped_to_device": False,
        },
        "work": {
            "source_image_terms": len(fc3.terms),
            "effective_image_terms": len(effective),
            "nonzero_effective_terms": nonzero_terms,
            "geometry_evaluations": geometry_evaluations,
            "retained_pair_anchor_counts_by_leg": [int(x) for x in counts[:, 0]],
            "all_leg_crop_anchor_counts_by_leg": [int(x) for x in counts[:, 1]],
            "omitted_anchor_counts_by_leg": [int(x) for x in counts[:, 2]],
            "outside_finite_device_anchor_counts_by_leg": [int(x) for x in outside_device],
            "streaming_component_visits": streaming_work,
            "reference_component_visits": reference_work,
            "planned_component_visits": planned_work,
            "max_accumulation_work": work_cap,
            "max_geometry_evaluations": geometry_cap,
        },
        "memory": {
            "owned_contraction_matrix_bytes": family_bytes,
            "numeric_workspace_plan_bytes": workspace_plan,
            "max_workspace_bytes": workspace_cap,
            "caller_payload_and_python_objects_excluded": True,
            "full_cubic_tensor_allocated": False,
        },
        "comparison_summary": {
            "complete_crop_omitted_closure_passed": True,
            "closure_max_error_allowance_ratio": closure_max_ratio,
            "saved_vertex_comparison_available": reference_vertex is not None,
            "saved_vertex_comparison_passed": reference_vertex is not None,
            "saved_vertex_max_error_allowance_ratio": reference_max_ratio if reference_vertex is not None else None,
            "identity_rtol": _RTOL,
            "identity_atol": _ATOL,
            "complete_source_zero_is_not_an_execution_gate": True,
        },
        "elapsed_seconds": float(time.monotonic() - started),
    }
