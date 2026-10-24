"""Symmetric finite support for feedback of actual crossing IFC triplets.

Each source triplet is retained once when any leg touches the central crop.
All participating coordinates receive self-energy in the existing full-device
solver. This is an explicit finite interaction model, not an ASR correction.
"""
from __future__ import annotations

import json
import time

import numpy as np

from ._canonical import domain_digest
from .layered import LayeredDeviceMapping
from .mapped_fc3 import (
    MassNormalizedTranslationalFC3,
    _effective_terms,
    _layer_tuple,
    _positive_integer,
    _validate_mapping_identity,
)
from .momentum import UniformBZMesh
from .transverse_boundary_cancellation import _comparison
from .transverse_material import _geometry, _position
from .transverse_material_packed import (
    FourierPackedMaterialCubic,
    fourier_material_cubic_packed,
)
from .transverse_packed import PackedTransverseCubicVertex
from .types import SolverScale

MODEL = "actual-ifc-any-leg-central-crop-v1"


def build_crossing_feedback_vertex(
    mesh, fc3, device_mapping, *, n_device_layers, interaction_layers, scale,
    reference_vertex=None, max_bytes=4*1024**3, max_workspace_bytes=32*1024**3,
    max_geometry_evaluations=2_000_000, max_phase_evaluations=1_000_000_000,
    deadline_monotonic=None,
):
    """Select actual full-device IFC coefficients with a symmetric support mask.

    The complete required any-leg-central geometry is admitted before mapping.
    The existing mapper supplies unmodified source values, image weights and
    all-q phases. Coefficients with all three coordinates outside the central
    crop alone are removed; retained coefficients are not fitted or repaired.
    Numeric plans exclude caller source/reference payload and library scratch.
    """
    started = time.monotonic()

    def check_time():
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("Crossing feedback mapping deadline reached.")

    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite.")
    check_time()
    if not isinstance(mesh, UniformBZMesh) or mesh.shape[2] != 1:
        raise TypeError("A transverse UniformBZMesh is required.")
    if not isinstance(fc3, MassNormalizedTranslationalFC3):
        raise TypeError("A mass-normalized material FC3 is required.")
    if not isinstance(device_mapping, LayeredDeviceMapping) or not isinstance(scale, SolverScale):
        raise TypeError("A layered mapping and solver scale are required.")
    layers = _positive_integer(n_device_layers, name="n_device_layers")
    cap = _positive_integer(max_bytes, name="max_bytes")
    workspace = _positive_integer(max_workspace_bytes, name="max_workspace_bytes")
    geometry_cap = _positive_integer(max_geometry_evaluations, name="max_geometry_evaluations")
    phase_cap = _positive_integer(max_phase_evaluations, name="max_phase_evaluations")
    crop = _layer_tuple(interaction_layers, name="interaction_layers", n_device_layers=layers)
    mapping = device_mapping
    _geometry(mapping)
    _validate_mapping_identity(fc3, mapping)
    p = mapping.principal_cells
    minimum, maximum = crop[0]*p, (crop[-1]+1)*p-1
    full_ndof = layers*mapping.n_principal_layer_dofs
    if reference_vertex is not None and (
        not isinstance(reference_vertex, PackedTransverseCubicVertex)
        or reference_vertex.mesh.shape != mesh.shape
        or reference_vertex.full_ndof != full_ndof
        or np.any(reference_vertex.active_dofs < crop[0]*mapping.n_principal_layer_dofs)
        or np.any(reference_vertex.active_dofs >= (crop[-1]+1)*mapping.n_principal_layer_dofs)
    ):
        raise ValueError("Production crop reference has a different domain.")
    if len(fc3.terms)*(3 if fc3.make_r0_average else 1) > geometry_cap:
        raise ValueError("Source expansion exceeds max_geometry_evaluations.")
    effective = _effective_terms(fc3)
    nonzero = sum(bool(np.any(term.values != 0)) for term in effective)
    one_pass = nonzero*len(mapping.direct_coset_representatives)
    # One admission pass here plus the existing mapper's two geometry passes.
    if 3*one_pass > geometry_cap:
        raise ValueError("Crossing feedback geometry work exceeds max_geometry_evaluations.")
    required = central = crossing = 0
    outside = [0, 0, 0]
    first_cell, last_cell = None, None
    for index, term in enumerate(effective):
        if index % 256 == 0:
            check_time()
        if not np.any(term.values != 0):
            continue
        for representative in mapping.direct_coset_representatives:
            mapped = tuple(_position(atom, position+representative, mapping)
                           for atom, position in zip(term.primitive_indices, term.positions, strict=True))
            cells = [int(position[0][0]) for position in mapped]
            # A union, not three separate sums: triplets with several central
            # legs must still be counted and mapped exactly once.
            anchors = sorted({anchor for cell in cells
                              for anchor in range(minimum-cell, maximum-cell+1)})
            for anchor in anchors:
                positions = [cell+anchor for cell in cells]
                required += 1
                all_central = all(minimum <= cell <= maximum for cell in positions)
                central += int(all_central)
                crossing += int(not all_central)
                first_cell = min(positions) if first_cell is None else min(first_cell, *positions)
                last_cell = max(positions) if last_cell is None else max(last_cell, *positions)
                for leg, cell in enumerate(positions):
                    outside[leg] += int(not 0 <= cell < layers*p)
    support = {
        "required_anchor_images": required, "all_leg_cropped_anchor_images": central,
        "crossing_anchor_images": crossing, "outside_device_counts_by_leg": outside,
        "required_longitudinal_cell_min": first_cell,
        "required_longitudinal_cell_max": last_cell,
        "available_longitudinal_cells": [0, layers*p-1],
        "complete_required_support": not any(outside), "triplets_counted_once": True,
    }
    if any(outside):
        raise ValueError("Crossing feedback support exceeds the harmonic device: "
                         +json.dumps(support, sort_keys=True))
    check_time()
    mapped = fourier_material_cubic_packed(
        mesh, fc3, mapping, n_device_layers=layers, interaction_layers=tuple(range(layers)),
        scale=scale, max_bytes=cap, max_workspace_bytes=workspace,
        max_geometry_evaluations=geometry_cap-one_pass,
        max_phase_evaluations=phase_cap,
    )
    check_time()
    source = mapped.vertex
    a, nq = source.ndof, mesh.n_qpoints
    coefficient_bytes = source.coefficients.nbytes
    outside_indices = np.flatnonzero(
        (source.active_dofs < crop[0]*mapping.n_principal_layer_dofs)
        | (source.active_dofs >= (crop[-1]+1)*mapping.n_principal_layer_dofs))
    auxiliary = 128*a**3+32*len(outside_indices)**3+65536
    peak_plan = max(int(mapped.metadata["planned_peak_numeric_bytes"]),
                    2*coefficient_bytes+auxiliary)
    if peak_plan > workspace:
        raise ValueError("Crossing feedback selection/construction exceeds max_workspace_bytes.")
    values = np.array(source.coefficients, copy=True)
    addresses = np.array(source.active_dofs, copy=True)
    removed_squared, removed_entries = 0., 0
    if len(outside_indices):
        selection = np.ix_(outside_indices, outside_indices, outside_indices)
        for q in range(nq):
            check_time()
            for internal in range(nq):
                removed = values[q, internal][selection]
                removed_squared += float(np.linalg.norm(removed))**2
                removed_entries += int(np.count_nonzero(removed))
                values[q, internal][selection] = 0
        del removed
    base_metadata = dict(mapped.metadata)
    del source, mapped
    vertex = PackedTransverseCubicVertex(mesh, values, addresses, full_ndof, max_bytes=cap)
    del values
    crop_comparison = []
    if reference_vertex is not None:
        lookup = {int(address): i for i, address in enumerate(vertex.active_dofs)}
        try:
            indices = [lookup[int(address)] for address in reference_vertex.active_dofs]
        except KeyError as exc:
            raise ValueError("Production reference support is missing from the full vertex.") from exc
        selection = np.ix_(indices, indices, indices)
        for q in range(nq):
            check_time()
            for internal in range(nq):
                row = _comparison(vertex.coefficients[q, internal][selection],
                                  reference_vertex.coefficients[q, internal])
                if not row["passed"]:
                    raise ValueError("Feedback central crop differs from the production vertex.")
                crop_comparison.append({"q_index": q, "p_index": internal, **row})
    configuration = {
        "model": MODEL, "mesh_shape": list(mesh.shape), "n_device_layers": layers,
        "central_interaction_layers": list(crop),
        "selection": "retain actual triplet iff any coordinate belongs to central crop",
        "source_fc3_sha256": fc3.source_fc3_content_sha256,
        "normalized_fc3_sha256": fc3.content_sha256,
        "geometry_sha256": mapping.layered_geometry_sha256,
        "omega0": scale.omega0, "hbar": scale.hbar,
    }
    metadata = {
        **configuration,
        "schema": "negfpy.crossing-feedback-vertex.v1",
        "model_identity_sha256": domain_digest("NEGFPY-CROSSING-FEEDBACK-MODEL", configuration),
        "full_ndof": full_ndof, "active_dofs": vertex.ndof,
        "vertex_bytes": coefficient_bytes+addresses.nbytes,
        "support": support, "geometry_evaluations": 3*one_pass,
        "base_full_device_mapping": base_metadata,
        "removed_all_outside_nonzero_coefficients": removed_entries,
        "removed_all_outside_frobenius": float(np.sqrt(removed_squared)),
        "planned_peak_numeric_bytes": peak_plan,
        "production_crop_comparison": crop_comparison,
        "retained_coefficients_unchanged": True, "raw_ifc_modified": False,
        "asr_fitted": False, "acoustic_projection": False,
        "global_asr_restored": False,
        "acoustic_scope": "Complete source cancellation retained for pairs touching the central crop; "
                          "outer/outer pairs can retain finite-boundary residuals.",
        "same_support_for_self_energy_and_current": True,
        "coupling_applied": False, "elapsed_seconds": time.monotonic()-started,
    }
    check_time()
    return FourierPackedMaterialCubic(vertex, metadata)


__all__ = ["MODEL", "build_crossing_feedback_vertex"]
