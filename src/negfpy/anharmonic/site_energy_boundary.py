"""Complete center-star cubic energies on an open transport device.

The fitted bulk energy is an explicit research surrogate.  Selecting whole
center energies preserves their displacement-difference structure; the optional
all-leg derivative crop is a deliberately incomplete control of the SAME fit.
No fitted coefficient is represented as an unchanged raw/interface IFC.
"""
from __future__ import annotations

from itertools import product
from time import monotonic

import numpy as np

from ._canonical import domain_digest, primitive_crystal_content_sha256
from .crystal import CrystalMapping
from .fc3 import fc3_force_constant_to_si_factor, mass_to_si_factor
from .layered import LayeredDeviceMapping
from .mapped_fc3 import _layer_tuple, _positive_integer
from .momentum import UniformBZMesh
from .transverse_mapping import _kpoints
from .transverse_material import _geometry, _position
from .transverse_material_packed import FourierPackedMaterialCubic
from .transverse_packed import PackedTransverseCubicVertex
from .transverse_q import difference_q_indices
from .types import SolverScale


def _local_derivative(coefficients, monomials, neighbors):
    """Differentiate K*d_p*d_q*d_r/repeated-index-factorials exactly."""
    dimension = 3*neighbors
    lookup = {tuple(int(i) for i in triple): j for j, triple in enumerate(monomials)}
    ordered = np.indices((dimension,)*3).reshape(3, -1).T
    values = np.asarray([coefficients[lookup[tuple(sorted(row))]] for row in ordered])
    axes, slots = ordered % 3, ordered//3+1
    tensor = np.zeros((neighbors+1,)*3+(3,)*3)
    for endpoints in product((0, 1), repeat=3):
        actual = np.where(endpoints, slots, 0)
        np.add.at(tensor, tuple(actual.T)+tuple(axes.T), (-1.)**(3-sum(endpoints))*values)
    return tensor


def _primitive_stars(model, crystal):
    """Keep coherent images and convert s2p representative IDs to ordinals."""
    n = crystal.supercell.n_atoms
    if (model.positions.shape != (n, 3)
            or not np.array_equal(model.positions, crystal.supercell.scaled_positions)
            or not np.array_equal(model.lattice, crystal.supercell.lattice)):
        raise ValueError("Fitted model geometry differs from the raw crystal.")
    if {int(v) for v in model.class_labels} != {int(v) for v in crystal.p2s}:
        raise ValueError("Model class labels must be the actual crystal s2p representatives.")
    if len(model.classes) != n or np.any(
        np.asarray(model.class_labels)[model.classes] != crystal.s2p
    ):
        raise ValueError("Fitted center classes differ from crystal s2p.")
    inverse = np.linalg.inv(crystal.primitive.lattice)
    result = {}
    for class_index, label in enumerate(model.class_labels):
        primitive_center = int(crystal.p2p[int(label)])
        if primitive_center < 0:
            raise ValueError("Class representative has no primitive ordinal.")
        star = next((s for s in model.stars if int(s.class_index) == class_index), None)
        if star is None:
            raise ValueError("A fitted class has no center star.")
        if int(model.classes[star.center]) != class_index:
            raise ValueError("Star class differs from its center.")
        neighbor_ids, offsets = [], []
        for neighbor, image, vector in zip(star.neighbors, star.neighbor_image_integers,
                                          star.neighbor_cartesian_vectors, strict=True):
            neighbor = int(neighbor)
            actual = (model.positions[neighbor]+image-model.positions[star.center])@model.lattice
            if not np.allclose(actual, vector, rtol=1e-10, atol=1e-10):
                raise ValueError("Star vector differs from its retained periodic image.")
            primitive_neighbor = int(crystal.p2p[int(crystal.s2p[neighbor])])
            offset = (crystal.primitive.scaled_positions[primitive_center]+actual@inverse
                      -crystal.primitive.scaled_positions[primitive_neighbor])
            integer = np.rint(offset).astype(np.int64)
            if not np.allclose(offset, integer, rtol=0., atol=1e-8):
                raise ValueError("Site-star neighbor lacks a coherent primitive image.")
            neighbor_ids.append(primitive_neighbor)
            offsets.append(integer)
        if primitive_center in result:
            raise ValueError("Two classes map to the same primitive atom.")
        result[primitive_center] = (class_index, tuple(neighbor_ids), np.asarray(offsets))
    return result


def build_site_energy_vertex(
    mesh, model, crystal, device_mapping, *, n_device_layers, interaction_layers,
    scale, force_constant_units, mass_units, crop_derivatives=False,
    max_bytes=4*1024**3, max_workspace_bytes=32*1024**3,
    max_geometry_evaluations=2_000_000, max_phase_evaluations=1_000_000_000,
    deadline_monotonic=None,
):
    """Map selected complete center energies, before coupling g.

    Both variants use exactly the same fitted energy and selected centers.
    ``crop_derivatives=True`` alone discards derivatives with any leg outside
    the center layers, as a boundary comparison control.  Packed addresses map
    each endpoint union to the same full harmonic-device Green fields.
    """
    if not isinstance(mesh, UniformBZMesh) or mesh.shape[2] != 1:
        raise TypeError("A transverse UniformBZMesh is required.")
    if not isinstance(crystal, CrystalMapping) or not isinstance(device_mapping, LayeredDeviceMapping):
        raise TypeError("Raw crystal and layered device mappings are required.")
    if not isinstance(scale, SolverScale) or type(crop_derivatives) is not bool:
        raise TypeError("Supply SolverScale and a boolean crop_derivatives.")
    layers = _positive_integer(n_device_layers, name="n_device_layers")
    cap = _positive_integer(max_bytes, name="max_bytes")
    workspace = _positive_integer(max_workspace_bytes, name="max_workspace_bytes")
    geometry_cap = _positive_integer(max_geometry_evaluations, name="max_geometry_evaluations")
    phase_cap = _positive_integer(max_phase_evaluations, name="max_phase_evaluations")
    centers = _layer_tuple(interaction_layers, name="interaction_layers", n_device_layers=layers)
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite.")

    def admit():
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            raise TimeoutError("Site-energy boundary mapping reached its deadline.")

    admit()
    mapping = device_mapping
    _geometry(mapping)
    if primitive_crystal_content_sha256(crystal) != mapping.primitive_crystal_content_sha256:
        raise ValueError("Fitted crystal and harmonic device primitives differ.")
    descriptors = _primitive_stars(model, crystal)
    coefficients = np.asarray(model.coefficients)
    if coefficients.ndim != 2 or not np.isfinite(coefficients).all():
        raise ValueError("Fitted energy coefficients must be a finite class matrix.")
    if coefficients.shape != (len(model.class_labels), len(model.monomials)):
        raise ValueError("Fitted coefficient and monomial dimensions differ.")
    p, nb = mapping.principal_cells, mapping.n_transport_basis_atoms
    ld, nq = mapping.n_principal_layer_dofs, mesh.n_qpoints
    ndof = layers*ld
    force_factor = fc3_force_constant_to_si_factor(force_constant_units)
    masses = np.asarray(crystal.primitive.masses)*mass_to_si_factor(mass_units)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        factor = float(force_factor*np.sqrt(scale.hbar)/np.float64(scale.omega0)**2.5)
    if not np.isfinite(factor) or factor <= 0 or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Site-energy physical/solver normalization is not representable.")
    class_tensors, class_records = {}, []
    for atom, (index, neighbor_ids, offsets) in descriptors.items():
        admit()
        tensor = _local_derivative(coefficients[index], model.monomials, len(neighbor_ids))
        roots = np.sqrt(masses[[atom, *neighbor_ids]])
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            scaled = tensor*factor/(roots[:, None, None, None, None, None]
                *roots[None, :, None, None, None, None]*roots[None, None, :, None, None, None])
        if not np.isfinite(scaled).all() or np.any((tensor != 0) & (scaled == 0)):
            raise ValueError("Nonzero site-energy derivative over/underflows solver units.")
        class_tensors[index] = scaled
        class_records.append({"primitive_center": atom, "class_index": index,
            "neighbor_primitive_atoms": list(neighbor_ids), "neighbor_primitive_offsets": offsets.tolist()})

    # Enumerate one center per transport basis/subcell, never extra coset copies.
    embedded, first, last = [], None, None
    evaluations = 0
    center_set = set(centers)
    for layer in centers:
        for subcell in range(p):
            ell = layer*p+subcell
            for basis, atom in enumerate(mapping.basis_primitive_atoms):
                admit()
                atom = int(atom)
                index, neighbor_ids, offsets = descriptors[atom]
                origin = (crystal.primitive.scaled_positions[atom]+mapping.basis_integer_cells[basis]
                          +ell*mapping.transport_supercell_matrix[0])
                positions = [origin]+[origin+crystal.primitive.scaled_positions[j]+offset
                    -crystal.primitive.scaled_positions[atom] for j, offset in zip(neighbor_ids, offsets, strict=True)]
                evaluations += len(positions)
                if evaluations > geometry_cap:
                    raise ValueError("Site-energy geometry exceeds max_geometry_evaluations.")
                mapped = [_position(j, pos, mapping) for j, pos in zip((atom, *neighbor_ids), positions, strict=True)]
                if int(mapped[0][0][0]) != ell or mapped[0][1] != basis:
                    raise ValueError("Center does not round-trip to its declared transport site.")
                addresses, transverse, endpoint_layers = [], [], []
                for cell, found_basis in mapped:
                    longcell = int(cell[0])
                    first = longcell if first is None else min(first, longcell)
                    last = longcell if last is None else max(last, longcell)
                    if not 0 <= longcell < layers*p:
                        raise ValueError("A complete center star leaves the harmonic device.")
                    endpoint_layer, endpoint_subcell = divmod(longcell, p)
                    addresses.append(endpoint_layer*ld+3*(endpoint_subcell*nb+found_basis))
                    transverse.append(tuple(int(v) for v in cell[1:]))
                    endpoint_layers.append(endpoint_layer)
                embedded.append((index, addresses, transverse, endpoint_layers))

    blocks = phase_work = cropped_blocks = 0
    support = set()
    for index, bases, _, endpoint_layers in embedded:
        for triple in product(range(len(bases)), repeat=3):
            raw = class_tensors[index][triple]
            if not np.any(raw != 0):
                continue
            if crop_derivatives and any(endpoint_layers[i] not in center_set for i in triple):
                cropped_blocks += 1
                continue
            blocks += 1
            phase_work += nq*nq*int(np.count_nonzero(raw))
            for i in triple:
                support.update(bases[i]+axis for axis in range(3))
    if phase_work > phase_cap:
        raise ValueError("Site-energy Fourier sum exceeds max_phase_evaluations.")
    active = np.asarray(sorted(support), dtype=np.int64)
    lookup = {int(address): i for i, address in enumerate(active)}
    a = len(active)
    coefficient_bytes = 16*nq*nq*a**3
    auxiliary = 128*a**3+nq*nq*a**3+65536
    peak = 2*coefficient_bytes+auxiliary
    if coefficient_bytes+8*a > cap:
        raise ValueError("Site-energy vertex payload exceeds max_bytes.")
    if peak > workspace:
        raise ValueError("Site-energy construction exceeds max_workspace_bytes.")
    admit()
    values = np.zeros((nq, nq, a, a, a), dtype=np.complex128)
    routes = difference_q_indices(mesh, max_bytes=cap)
    kpoints, qindices = _kpoints(mesh), np.arange(nq)
    for index, bases, transverse, endpoint_layers in embedded:
        admit()
        for triple in product(range(len(bases)), repeat=3):
            raw = class_tensors[index][triple]
            if not np.any(raw != 0) or (crop_derivatives and any(endpoint_layers[i] not in center_set for i in triple)):
                continue
            i, j, k = triple
            second = tuple((b-a) % n for a, b, n in zip(transverse[i], transverse[j], mesh.shape[:2], strict=True))
            third = tuple((b-a) % n for a, b, n in zip(transverse[i], transverse[k], mesh.shape[:2], strict=True))
            phase_p, phase_r = np.exp(1j*(kpoints@second)), np.exp(1j*(kpoints@third))
            phase = phase_p[None, :]*phase_r[routes]
            coordinates = [[lookup[bases[atom]+axis] for axis in range(3)] for atom in triple]
            selection = np.ix_(qindices, qindices, *coordinates)
            values[selection] += phase[:, :, None, None, None]*raw[None, None]
    admit()
    vertex = PackedTransverseCubicVertex(mesh, values, active, ndof, max_bytes=cap)
    configuration = {"schema": "negfpy.site-energy-boundary-vertex.v1",
        "center_layers": list(centers), "n_device_layers": layers,
        "crop_derivatives": crop_derivatives, "mesh_shape": list(mesh.shape),
        "geometry_sha256": mapping.layered_geometry_sha256,
        "force_constant_units": force_constant_units, "mass_units": mass_units,
        "omega0": scale.omega0, "hbar": scale.hbar}
    energy_identity = domain_digest("NEGFPY-SITE-STAR-ENERGY", {
        "coefficients": coefficients, "monomials": model.monomials,
        "primitive_stars": class_records})
    configuration["energy_identity_sha256"] = energy_identity
    metadata = {**configuration,
        "model_identity_sha256": domain_digest("NEGFPY-SITE-ENERGY-BOUNDARY", configuration),
        "model": "fitted complete-center-star research surrogate",
        "full_ndof": ndof, "active_dofs": a, "selected_center_count": len(embedded),
        "center_selection_only": not crop_derivatives, "all_outer_derivatives_retained": not crop_derivatives,
        "derivative_crop_is_control": crop_derivatives, "cropped_local_derivative_blocks": cropped_blocks,
        "support": {"complete_required_support": True, "longitudinal_cell_min": first,
            "longitudinal_cell_max": last, "available_longitudinal_cells": [0, layers*p-1]},
        "primitive_stars": class_records, "geometry_evaluations": evaluations,
        "mapped_local_derivative_blocks": blocks, "phase_evaluations": phase_work,
        "vertex_bytes": coefficient_bytes+8*a, "planned_peak_numeric_bytes": peak,
        "output_units": "solver", "coupling_applied": False, "weights_applied": True,
        "additional_r0_average": False, "additional_mesh_or_factorial_normalization": False,
        "source_ifcs_modified": False, "physical_interface_uniquely_recovered": False,
        "source_remainder_omitted_from_surrogate": True, "field_projection": False}
    return FourierPackedMaterialCubic(vertex, metadata)


__all__ = ["build_site_energy_vertex"]
