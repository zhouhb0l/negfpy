"""Fail-closed assembly of a finite material :math:`N_q=1` problem.

This module is intentionally private.  The IFC readers, layered FC2 mapper,
and finite FC3 mapper remain separate scientific stages; this file only joins
their immutable outputs into the standard :class:`CubicSCBAProblem` consumed
by the solver.  In particular, it does not infer a device size or silently
crop a cubic interaction.  The complete-range and padding decision made by
``map_fc3_to_device`` must reproduce and pass here.

The lead convention is increasing layer number.  Consequently the finite
device couples to the left lead with ``d10`` and to the right lead with
``d01``.  Both couplings are stored explicitly so that no later default can
obscure that orientation.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import numpy as np

from negfpy.core.types import Device1D

from ._canonical import digest_bytes, domain_digest
from .grid import SignedFrequencyGrid
from .layered import (
    LayeredDeviceMapping,
    LayeredHarmonicModel,
    _model_payload,
)
from .mapped_fc3 import (
    DeviceMappedFC3,
    _cubic_physics_sha256,
    _report_payload,
    scale_device_fc3,
)
from .sparse_cubic import SparseCubicVertex
from .types import CubicSCBAProblem, HarmonicReservoir, SolverScale


Array = np.ndarray


class MaterialNQ1ProblemError(ValueError):
    """Raised when material stages cannot be joined without guessing."""


def _finite_scalar(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or np.asarray(value).ndim != 0
        or np.iscomplexobj(value)
    ):
        raise TypeError(f"{name} must be a scalar real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _exact_array(left: Any, right: Any, *, name: str) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or not np.array_equal(
        left_array, right_array
    ):
        raise MaterialNQ1ProblemError(f"{name} identity does not match.")


def _ordered_contiguous(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise MaterialNQ1ProblemError(
            f"{name} is not an integer sequence."
        ) from exc
    if not result or result != tuple(range(result[0], result[-1] + 1)):
        raise MaterialNQ1ProblemError(
            f"{name} must be nonempty, ordered, unique, and contiguous."
        )
    return result


def _validate_harmonic_model(model: LayeredHarmonicModel) -> int:
    if not isinstance(model, LayeredHarmonicModel):
        raise TypeError("harmonic_model must be a LayeredHarmonicModel.")
    if not isinstance(model.scale, SolverScale):
        raise MaterialNQ1ProblemError(
            "Layered harmonic model has no valid SolverScale."
        )
    if not isinstance(model.mapping, LayeredDeviceMapping):
        raise MaterialNQ1ProblemError(
            "Layered harmonic model has no valid device mapping."
        )
    if (
        model.validation is None
        or not bool(model.validation.certification_passed)
        or model.mapping.validation is None
        or not bool(model.mapping.validation.certification_passed)
    ):
        raise MaterialNQ1ProblemError(
            "Layered harmonic model and mapping must carry passing audits."
        )

    identities = (
        ("FC2 content", model.fc2_content_sha256, model.mapping.fc2_content_sha256),
        (
            "primitive crystal",
            model.primitive_crystal_content_sha256,
            model.mapping.primitive_crystal_content_sha256,
        ),
        (
            "FC2 crystal mapping",
            model.fc2_crystal_mapping_content_sha256,
            model.mapping.fc2_crystal_mapping_content_sha256,
        ),
        (
            "layered geometry",
            model.layered_geometry_sha256,
            model.mapping.layered_geometry_sha256,
        ),
    )
    for name, model_value, mapping_value in identities:
        if model_value != mapping_value:
            raise MaterialNQ1ProblemError(
                f"Layered harmonic {name} identity does not match its mapping."
            )

    d00 = np.asarray(model.d00)
    d01 = np.asarray(model.d01)
    d10 = np.asarray(model.d10)
    if d00.ndim != 2 or d00.shape[0] < 1 or d00.shape[0] != d00.shape[1]:
        raise MaterialNQ1ProblemError("Harmonic d00 must be a nonempty square block.")
    if d01.shape != d00.shape or d10.shape != d00.shape:
        raise MaterialNQ1ProblemError(
            "Harmonic d00, d01, and d10 must have identical shapes."
        )
    if not all(np.all(np.isfinite(block)) for block in (d00, d01, d10)):
        raise MaterialNQ1ProblemError("Harmonic blocks must be finite.")
    if not np.array_equal(d00, d00.conj().T):
        raise MaterialNQ1ProblemError("Harmonic d00 must be exactly Hermitian.")
    if not np.array_equal(d10, d01.conj().T):
        raise MaterialNQ1ProblemError(
            "Harmonic orientation requires exact d10=d01^dagger."
        )
    layer_dofs = int(model.mapping.n_principal_layer_dofs)
    if layer_dofs != d00.shape[0]:
        raise MaterialNQ1ProblemError(
            "Harmonic block size does not match the mapped principal layer."
        )
    _exact_array(model.lead.d00, d00, name="lead d00")
    _exact_array(model.lead.d01, d01, name="lead d01")
    lead_d10 = (
        model.lead.d10
        if model.lead.d10 is not None
        else np.asarray(model.lead.d01).conj().T
    )
    _exact_array(lead_d10, d10, name="lead d10")

    expected_model_hash = domain_digest(
        "NEGFPY-LAYERED-HARMONIC-MODEL-PHYSICS",
        _model_payload(
            model.mapping,
            model.scale,
            d00=d00,
            d01=d01,
            d10=d10,
            atom_masses_kg=np.asarray(model.principal_layer_atom_masses_kg),
            rigid_translation_vectors=np.asarray(model.rigid_translation_vectors),
        ),
    )
    if expected_model_hash != model.model_physics_sha256:
        raise MaterialNQ1ProblemError(
            "Layered harmonic model physics identity does not reproduce."
        )
    return layer_dofs


def _validate_mapped_fc3(
    mapped: DeviceMappedFC3,
    model: LayeredHarmonicModel,
    *,
    layer_dofs: int,
) -> int:
    if not isinstance(mapped, DeviceMappedFC3):
        raise TypeError("mapped_fc3 must be a DeviceMappedFC3.")
    report = mapped.report
    if not report.certification_passed or report.failure_codes:
        raise MaterialNQ1ProblemError(
            "FC3 device mapping report did not pass complete-range preflight: "
            f"{report.failure_codes!r}."
        )
    forbidden = {
        "symmetrized": report.symmetrized,
        "corrected": report.corrected,
        "truncated": report.truncated,
        "thresholded": report.thresholded,
    }
    changed = tuple(name for name, value in forbidden.items() if value)
    if changed:
        raise MaterialNQ1ProblemError(
            "FC3 device mapping used forbidden support-changing operations: "
            f"{changed!r}."
        )
    if (
        report.permutation_missing_count != 0
        or not report.permutation_tolerance_passed
    ):
        raise MaterialNQ1ProblemError(
            "FC3 device mapping lacks complete, roundoff-bounded "
            "permutation certification."
        )

    expected_report_hash = domain_digest(
        "NEGFPY-FC3-DEVICE-MAPPING", _report_payload(report)
    )
    if expected_report_hash != report.report_sha256:
        raise MaterialNQ1ProblemError("FC3 mapping report identity does not reproduce.")
    expected_physics_hash = _cubic_physics_sha256(
        mapped.ndof,
        np.asarray(mapped.indices),
        np.asarray(mapped.values),
    )
    if (
        expected_physics_hash != mapped.cubic_physics_sha256
        or expected_physics_hash != report.cubic_physics_sha256
    ):
        raise MaterialNQ1ProblemError("Mapped FC3 physics identity does not reproduce.")
    expected_content_hash = domain_digest(
        "NEGFPY-DEVICE-MAPPED-FC3",
        {
            "ndof": mapped.ndof,
            "indices": np.asarray(mapped.indices),
            "values": np.asarray(mapped.values),
            "interaction_layers": mapped.interaction_layers,
            "observation_layers": mapped.observation_layers,
            "report_sha256": digest_bytes(report.report_sha256),
        },
    )
    if expected_content_hash != mapped.content_sha256:
        raise MaterialNQ1ProblemError("Mapped FC3 content identity does not reproduce.")

    joined_identities = (
        ("FC2 content", report.fc2_content_sha256, model.fc2_content_sha256),
        (
            "primitive crystal",
            report.primitive_crystal_content_sha256,
            model.primitive_crystal_content_sha256,
        ),
        (
            "FC2 crystal mapping",
            report.fc2_crystal_mapping_content_sha256,
            model.fc2_crystal_mapping_content_sha256,
        ),
        (
            "layered geometry",
            report.layered_geometry_sha256,
            model.layered_geometry_sha256,
        ),
    )
    for name, mapped_value, harmonic_value in joined_identities:
        if mapped_value != harmonic_value:
            raise MaterialNQ1ProblemError(
                f"FC2/FC3 {name} identity mismatch."
            )

    n_layers = int(report.n_device_layers)
    if n_layers < 1 or mapped.ndof != n_layers * layer_dofs:
        raise MaterialNQ1ProblemError(
            "Mapped FC3 ndof does not match the harmonic device geometry."
        )
    if report.contact_layers != (0, n_layers - 1):
        raise MaterialNQ1ProblemError(
            "FC3 report contact layers do not span the finite device."
        )
    interactions = _ordered_contiguous(
        mapped.interaction_layers, name="interaction_layers"
    )
    observations = _ordered_contiguous(
        mapped.observation_layers, name="observation_layers"
    )
    if (
        interactions != report.interaction_layers
        or observations != report.observation_layers
        or not set(observations).issubset(interactions)
        or interactions[0] <= 0
        or interactions[-1] >= n_layers - 1
    ):
        raise MaterialNQ1ProblemError(
            "FC3 interaction/observation layers do not define an interior crop."
        )
    # The finite contact layers are part of the harmonic device and therefore
    # count toward the noninteracting FC3 halo.  They are not reservoir FC3;
    # they are the lead-like device layers between the interaction crop and
    # the harmonic semi-infinite contacts.
    expected_buffers = (
        tuple(range(0, interactions[0])),
        tuple(range(interactions[-1] + 1, n_layers)),
    )
    if report.harmonic_buffer_layers != expected_buffers:
        raise MaterialNQ1ProblemError(
            "FC3 harmonic-buffer geometry does not reproduce."
        )
    if (
        len(expected_buffers[0]) < report.required_left_halo_layers
        or len(expected_buffers[1]) < report.required_right_halo_layers
        or observations[0] - interactions[0] < report.required_left_halo_layers
        or interactions[-1] - observations[-1] < report.required_right_halo_layers
    ):
        raise MaterialNQ1ProblemError(
            "FC3 crop lacks the directional range or observation padding "
            "declared by its report."
        )

    indices = np.asarray(mapped.indices)
    values = np.asarray(mapped.values)
    if indices.ndim != 2 or indices.shape[1:] != (3,):
        raise MaterialNQ1ProblemError("Mapped FC3 indices must have shape (nnz, 3).")
    if values.shape != (indices.shape[0],):
        raise MaterialNQ1ProblemError(
            "Mapped FC3 values do not match its sparse indices."
        )
    if indices.dtype.kind not in "iu":
        raise MaterialNQ1ProblemError("Mapped FC3 indices are not exact integers.")
    if np.any(indices < 0) or np.any(indices >= mapped.ndof):
        raise MaterialNQ1ProblemError("Mapped FC3 index lies outside the device.")
    if len(indices):
        active_layers = indices // layer_dofs
        if not np.all(np.isin(active_layers, np.asarray(interactions))):
            raise MaterialNQ1ProblemError(
                "Mapped FC3 support escapes the certified interaction crop."
            )
    if mapped.nnz != report.final_nonzero_count:
        raise MaterialNQ1ProblemError(
            "Mapped FC3 support count differs from its report."
        )
    metadata = mapped.metadata
    if (
        metadata.get("solver_scaled") is not False
        or metadata.get("weights_applied") is not True
        or metadata.get("mapping_report_sha256") != report.report_sha256
    ):
        raise MaterialNQ1ProblemError(
            "Mapped FC3 metadata does not describe physical-SI, weighted input."
        )
    return n_layers


def _continued_vertex(
    base: SparseCubicVertex,
    *,
    coupling_scale: float,
    assembly_sha256: str,
) -> SparseCubicVertex:
    coefficients = np.asarray(base.coefficients) * coupling_scale
    if not np.all(np.isfinite(coefficients)):
        raise MaterialNQ1ProblemError(
            "Cubic continuation scaling produced a non-finite coefficient."
        )
    if coupling_scale == 0.0:
        indices = np.empty((0, 3), dtype=np.int64)
        coefficients = np.empty((0,), dtype=np.float64)
    else:
        if np.any(coefficients == 0.0):
            raise MaterialNQ1ProblemError(
                "Cubic continuation scaling underflowed nonzero FC3 support."
            )
        indices = base.indices
    return SparseCubicVertex(
        ndof=base.ndof,
        indices=indices,
        coefficients=coefficients,
        metadata={
            **dict(base.metadata),
            "schema_version": "negfpy.sparse-cubic-vertex.material-problem.v1",
            "uncontinued_vertex_storage_sha256": base.vertex_storage_sha256,
            "uncontinued_cubic_physics_sha256": base.cubic_physics_sha256,
            "cubic_continuation_factor": coupling_scale,
            "cubic_continuation_applied_count": 1,
            "material_nq1_assembly_sha256": assembly_sha256,
        },
    )


def build_material_nq1_problem(
    harmonic_model: LayeredHarmonicModel,
    mapped_fc3: DeviceMappedFC3,
    *,
    grid: SignedFrequencyGrid,
    mean_temperature: float,
    temperature_bias: float = 0.0,
    coupling_scale: float = 1.0,
) -> CubicSCBAProblem:
    """Join certified FC2/FC3 stages into one solver-ready material problem.

    Temperatures are in solver energy units.  ``temperature_bias`` is signed
    and is defined as ``T_left - T_right`` around ``mean_temperature``.
    ``coupling_scale`` is the nonnegative cubic continuation factor applied
    once, after the physical mapped FC3 has been converted with the harmonic
    model's :class:`SolverScale`.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    mean = _finite_scalar(
        mean_temperature,
        name="mean_temperature",
        nonnegative=True,
    )
    bias = _finite_scalar(temperature_bias, name="temperature_bias")
    coupling = _finite_scalar(
        coupling_scale,
        name="coupling_scale",
        nonnegative=True,
    )
    left_temperature = mean + 0.5 * bias
    right_temperature = mean - 0.5 * bias
    if left_temperature < 0.0 or right_temperature < 0.0:
        raise ValueError(
            "temperature_bias makes at least one reservoir temperature negative."
        )

    layer_dofs = _validate_harmonic_model(harmonic_model)
    n_layers = _validate_mapped_fc3(
        mapped_fc3,
        harmonic_model,
        layer_dofs=layer_dofs,
    )
    scale = harmonic_model.scale
    base_vertex = scale_device_fc3(mapped_fc3, scale)
    if base_vertex.ndof != mapped_fc3.ndof or not np.array_equal(
        base_vertex.indices, mapped_fc3.indices
    ):
        raise MaterialNQ1ProblemError(
            "Scaled sparse vertex changed mapped FC3 indices or ndof."
        )

    assembly_sha256 = domain_digest(
        "NEGFPY-MATERIAL-NQ1-PROBLEM-ASSEMBLY",
        {
            "schema_version": "negfpy.material-nq1-problem.v1",
            "harmonic_model_physics_sha256": digest_bytes(
                harmonic_model.model_physics_sha256
            ),
            "device_mapped_fc3_sha256": digest_bytes(mapped_fc3.content_sha256),
            "solver_scale": {"omega0": scale.omega0, "hbar": scale.hbar},
            "n_device_layers": n_layers,
            "layer_dofs": layer_dofs,
            "left_device_to_lead": np.asarray(harmonic_model.d10),
            "right_device_to_lead": np.asarray(harmonic_model.d01),
            "contact_orientation": "left=d10,right=d01,increasing-layer-lead",
            "grid": {"step": grid.step, "n_positive": grid.n_positive},
            "mean_temperature": mean,
            "temperature_bias": bias,
            "left_temperature": left_temperature,
            "right_temperature": right_temperature,
            "cubic_continuation_factor": coupling,
        },
    )
    vertex = _continued_vertex(
        base_vertex,
        coupling_scale=coupling,
        assembly_sha256=assembly_sha256,
    )

    raw_device = harmonic_model.device(n_layers)
    device = Device1D(
        onsite_blocks=[
            np.array(block, dtype=np.complex128, order="C", copy=True)
            for block in raw_device.onsite_blocks
        ],
        coupling_blocks=[
            np.array(block, dtype=np.complex128, order="C", copy=True)
            for block in raw_device.coupling_blocks
        ],
    )
    for block in (*device.onsite_blocks, *device.coupling_blocks):
        block.setflags(write=False)
    if any(
        not np.array_equal(block, harmonic_model.d00)
        for block in device.onsite_blocks
    ) or any(
        not np.array_equal(block, harmonic_model.d01)
        for block in device.coupling_blocks
    ):
        raise MaterialNQ1ProblemError(
            "Layered harmonic device is not the exact repeated lead operator."
        )

    ndof = n_layers * layer_dofs
    left_indices = np.arange(layer_dofs, dtype=np.int64)
    right_indices = np.arange(ndof - layer_dofs, ndof, dtype=np.int64)
    left = HarmonicReservoir(
        lead=harmonic_model.lead,
        temperature=left_temperature,
        device_to_lead=np.asarray(harmonic_model.d10),
        device_indices=left_indices,
    )
    right = HarmonicReservoir(
        lead=harmonic_model.lead,
        temperature=right_temperature,
        device_to_lead=np.asarray(harmonic_model.d01),
        device_indices=right_indices,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=left,
        right=right,
        vertex=vertex,
        grid=grid,
        scale=scale,
    )
    if problem.ndof != mapped_fc3.ndof or problem.vertex.ndof != mapped_fc3.ndof:
        raise MaterialNQ1ProblemError(
            "Final harmonic and cubic problem dimensions do not agree."
        )
    return problem


__all__ = ["MaterialNQ1ProblemError", "build_material_nq1_problem"]
