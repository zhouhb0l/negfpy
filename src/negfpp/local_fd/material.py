"""Reconstruct fixed-size Method 1 inputs from an authenticated short donor.

The FC2 and harmonic contacts are rebuilt with the donor's original geometry,
normalization and broadening. Only its evaluated retarded self-energy is
transferred. No cubic coefficients are loaded and no SCBA map is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

import numpy as np

from negfpy.anharmonic.io.phono3py_crystal import read_phono3py_crystal_mapping
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.transverse_compact_contacts import build_compact_transverse_contact
from negfpy.anharmonic.transverse_mapping import _kpoints
from negfpy.anharmonic.transverse_material import material_transverse_harmonic
from negfpy.anharmonic.types import HarmonicReservoir
from negfpy.core.types import LeadBlocks

from .harmonic_blocks import extract_repeated_harmonic
from .kernel_transfer import SOURCE_ACTIVE_DOFS, make_frozen_kernel

if TYPE_CHECKING:
    from .source import FrozenSource


@dataclass(frozen=True)
class MaterialInputs:
    """Short arrays and repeated blocks consumed by the existing point engine."""

    kernel: Any
    harmonic: Any
    contact_retarded: tuple
    frequencies: np.ndarray
    weights: np.ndarray
    mesh: Any
    grid: Any
    scale: Any
    pins: tuple
    metadata: dict


def _admit(deadline: float) -> None:
    if monotonic() >= deadline:
        raise TimeoutError("Frozen-kernel material construction reached its deadline")


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _verify_files(source: FrozenSource) -> tuple[dict, ...]:
    pins = []
    for role in ("fc2", "crystal", "coefficients", "metadata"):
        path = Path(source.files[role])
        expected = source.report["inputs"][role]
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError("The bound donor input size changed: " + role)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("The bound donor input hash changed: " + role)
        pins.append({"role": role, "bytes": expected["bytes"], "sha256": expected["sha256"]})
    return tuple(pins)


def _verify_layout(source: FrozenSource) -> None:
    doc, report = source.inputdoc, source.report
    support = report["model"]["support"]
    if (doc["geometry"]["n_device_layers"] != 3
            or doc["geometry"]["interaction_layers"] != [1]
            or report["model"]["full_dofs"] != 72
            or report["model"]["active_dofs"] != 36
            or report["model"]["energy_centers"] != 8
            or support["center_layers"] != [1]
            or support["n_device_layers"] != 3
            or support["selected_center_count"] != 8
            or support["all_outer_derivatives_retained"] is not True
            or support["crop_derivatives"] is not False
            or support["cropped_local_derivative_blocks"] != 0):
        raise ValueError("Method 1 requires the original complete one-center 72/36-DOF donor")
    active = np.asarray(source.active_dofs)
    if (active.dtype.kind not in "iu"
            or not np.array_equal(active, np.asarray(SOURCE_ACTIVE_DOFS))):
        raise ValueError("The donor's complete endpoint order differs from the point engine")
    expected_shape = (source.mesh.n_qpoints, source.grid.n_points, 36, 36)
    if np.shape(source.sigmaR) != expected_shape:
        raise ValueError("The donor retarded array differs from its declared grid/support")
    if (tuple(doc["grid"]["mesh"]) != source.mesh.shape
            or doc["grid"]["step"] != source.grid.step
            or doc["grid"]["n_positive"] != len(source.grid.positive_indices)
            or doc["scale"]["omega0_rad_s"] != source.scale.omega0
            or doc["scale"]["hbar_j_s"] != source.scale.hbar
            or doc["reservoirs"]["eta_device"] != 0):
        raise ValueError("The donor grid, units or device broadening differs from its input")


def prepare_material(
    source: FrozenSource,
    *,
    deadline_monotonic: float,
    harmonic_max_bytes: int = 128 * 1024**2,
    contact_max_bytes: int = 3 * 1024**3,
) -> MaterialInputs:
    """Rebuild the existing three-layer oracle and paired harmonic contacts.

    ``source`` must come from the strict compact donor reader. Its original
    contact temperatures and eta are retained; changing solved probe
    temperatures never recalculates or rescales the donor retarded kernel.
    Returned frequency weights include d(omega/omega0)/(2*pi), with no q factor.
    The point engine applies the equal-q mean exactly once.
    """
    from .source import FrozenSource
    from ..input import read_json

    if not isinstance(source, FrozenSource):
        raise TypeError("A qualified FrozenSource is required")
    if (isinstance(deadline_monotonic, bool)
            or not math.isfinite(deadline_monotonic)):
        raise ValueError("A finite monotonic deadline is required")
    for key in ("passed", "evaluated_input", "all_original_six_passed"):
        if source.qualification.get(key) is not True:
            raise ValueError("The donor lacks a required qualification: " + key)
    harmonic_max_bytes = _positive_integer(harmonic_max_bytes, "harmonic_max_bytes")
    contact_max_bytes = _positive_integer(contact_max_bytes, "contact_max_bytes")
    _admit(deadline_monotonic)
    _verify_layout(source)
    pins = _verify_files(source)
    doc = source.inputdoc
    model = read_json(source.files["metadata"])
    crystal = read_phono3py_crystal_mapping(source.files["crystal"], purpose="fc2")
    if (set(crystal.primitive.symbols) != {"Si"}
            or not np.all(crystal.primitive.masses == crystal.primitive.masses[0])):
        raise ValueError("The original point engine requires the equal-mass silicon basis")
    _admit(deadline_monotonic)
    fc2 = read_phono3py_fc2(
        source.files["fc2"], crystal.supercell.masses,
        p2s_map=crystal.p2s, s2p_map=crystal.s2p, compact=False,
    )
    geometry = build_layered_device_mapping(
        crystal, fc2,
        transport_supercell_matrix=np.asarray(doc["geometry"]["transport_supercell_matrix"]),
    )
    material = model["material"]
    if (geometry.n_principal_layer_dofs != 24 or geometry.principal_cells != 1
            or geometry.layered_geometry_sha256 != material["layered_geometry_sha256"]
            or geometry.fc2_content_sha256 != material["source_fc2_sha256"]
            or geometry.primitive_crystal_content_sha256 != material["primitive_crystal_sha256"]
            or geometry.layered_geometry_sha256 != source.report["model"]["support"]["geometry_sha256"]
            or model["model_parameters"]["energy_identity_sha256"]
                != source.report["model"]["support"]["energy_identity_sha256"]):
        raise ValueError("Rebuilt FC2 geometry differs from the bound complete-center donor")
    mesh, grid, scale = source.mesh, source.grid, source.scale
    negative = np.asarray([mesh.negative_index(q) for q in range(mesh.n_qpoints)], dtype=np.int64)
    kernel = make_frozen_kernel(
        source.sigmaR, grid.values, source.active_dofs,
        negative_q_indices=negative,
        gamma_q_index=mesh.index_from_address((0, 0, 0)),
        source_translation_vectors=np.tile(np.eye(3), (24, 1)),
    )
    _admit(deadline_monotonic)
    device, lead = material_transverse_harmonic(
        geometry, n_device_layers=3, scale=scale, max_bytes=harmonic_max_bytes,
    )
    harmonic = extract_repeated_harmonic(device, lead, _kpoints(mesh), negative)
    positive = grid.positive_indices
    contact_retarded, contact_seconds = [], []
    q_samples = _positive_integer(doc["resources"]["q_samples"], "source q_samples")
    max_base = _positive_integer(doc["resources"]["max_base_evaluations"], "source max_base_evaluations")
    temperatures = []
    for side in ("left", "right"):
        _admit(deadline_monotonic)
        begin = monotonic()
        kelvin = doc["reservoirs"][side + "_temperature_k"]
        temperatures.append(kelvin)

        def reservoir(q, kelvin=kelvin):
            return HarmonicReservoir(
                LeadBlocks(*lead.blocks(q)),
                temperature=scale.dimensionless_temperature(kelvin, boltzmann_constant=1.380649e-23),
            )

        contact = build_compact_transverse_contact(
            mesh, grid, reservoir, 72, doc["reservoirs"]["eta_lead"],
            side=side, q_samples=q_samples, surface_gf_method="sancho_rubio",
            max_bytes=contact_max_bytes, max_base_evaluations=max_base,
        )
        expected_indices = tuple(range(24)) if side == "left" else tuple(range(48, 72))
        if contact.device_indices != expected_indices:
            raise ValueError("The rebuilt contact endpoint convention differs from the donor")
        values = np.array(contact.spectrum.retarded[:, positive], dtype=np.complex128, copy=True)
        values.setflags(write=False)
        contact_retarded.append(values)
        contact_seconds.append(monotonic() - begin)
        del contact
    _admit(deadline_monotonic)
    if _verify_files(source) != pins:
        raise ValueError("The bound donor material inputs changed during preparation")
    frequencies = np.array(grid.values[positive], dtype=np.float64, copy=True)
    weights = np.full(len(positive), grid.integration_weight, dtype=np.float64)
    frequencies.setflags(write=False)
    weights.setflags(write=False)
    lattice = np.asarray(geometry.transport_lattice_angstrom)
    repeat_angstrom = float(np.linalg.norm(lattice[0]))
    area_angstrom2 = abs(float(np.linalg.det(lattice))) / repeat_angstrom
    metadata = {
        "source_qualification": dict(source.qualification),
        "source_kernel": {
            "evaluated_input": True, "all_original_six_passed": True,
            "selected_index": source.report["selected_anchor"]["index"],
            "g": doc["scba"]["g"], "validation": kernel.validation,
        },
        "self_energy_maps_executed": 0,
        "previous_kernel_recomputed": False,
        "coefficient_arrays_loaded": False,
        "cubic_vertex_constructed": False,
        "fitted_model_refit": False,
        "harmonic_inputs_rebuilt_from_same_fc2": True,
        "harmonic_audit": harmonic.audit,
        "contact_build_seconds": contact_seconds,
        "temperatures_kelvin": temperatures,
        "eta_lead": doc["reservoirs"]["eta_lead"], "eta_device": 0.0,
        "mesh": list(mesh.shape), "frequency_nodes": grid.n_points,
        "positive_nodes": len(positive), "width_layers": 3,
        "source_endpoint_ndof": 36,
        "transport_repeat_nm": repeat_angstrom * 0.1,
        "cross_section_nm2": area_angstrom2 * 0.01,
        "frequency_weights_include_q_mean": False,
    }
    return MaterialInputs(
        kernel, harmonic, tuple(contact_retarded), frequencies, weights,
        mesh, grid, scale, pins, metadata,
    )
