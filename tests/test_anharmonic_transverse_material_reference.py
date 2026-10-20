"""Raw IFC image mapping checked against an independently enumerated crystal.

The synthetic source describes one displaced A--B bond, unequal masses and
three Cartesian coordinates per atom. Only its cubic x component is nonzero.
The oracle expands its polynomial in real space, crops each ordered monomial,
and performs a full-supercell DFT. It never calls either finite IFC mapper.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest
from test_anharmonic_transverse_q_reference import (
    _cells,
    _fourier_vertex,
    _to_q,
    _unitary_basis,
)

from negfpy.anharmonic._harmonic import assemble_device_dynamical_matrix
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.anharmonic.fc3 import FC3Data, validate_fc3
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.mapped_fc3 import mass_normalize_translational_fc3
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import (
    PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
    TranslationalFC3,
    convert_fc3_to_translational,
)
from negfpy.anharmonic.transverse_mapping import fourier_transverse_cubic
from negfpy.anharmonic.transverse_material import (
    map_material_cubic,
    material_transverse_harmonic,
)
from negfpy.anharmonic.types import SolverScale

_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.25, 0.3, 0.2]])
_MASSES = np.array([2.0, 5.0])
_BOND_WINDING = np.array([1, 1, 1])
_SPRING = 1.7
_CUBIC = 0.7
_LAYERS = 3
_CROP = (0, 1)


def _crystal(purpose: str, duplicate_images: bool) -> CrystalMapping:
    cell = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=_POSITIONS,
        masses=_MASSES,
        symbols=("A", "B"),
    )
    vectors = []
    multiplicity = np.empty((2, 2, 2), dtype=np.int64)
    count = 2 if duplicate_images else 1
    for atom, anchor in product(range(2), repeat=2):
        vector = _POSITIONS[atom] - _POSITIONS[anchor]
        if atom != anchor:
            vector = vector + (1 if atom == 1 else -1) * _BOND_WINDING
        multiplicity[atom, anchor] = (count, len(vectors))
        vectors.extend(vector.copy() for _ in range(count))
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.eye(3, dtype=np.int64),
        primitive=cell,
        supercell=cell,
        p2s=[0, 1],
        s2p=[0, 1],
        p2p=[0, 1],
        shortest_vectors=vectors,
        multiplicity=multiplicity,
        atomic_permutations=[[0, 1]],
        translation_permutations=[[0, 1]],
        purpose=purpose,
    )


def _raw_inputs(transform, *, duplicate_images: bool, r0: bool):
    signs = (1.0, -1.0)
    fc2_values = np.empty((2, 2, 3, 3))
    for first, second in product(range(2), repeat=2):
        fc2_values[first, second] = _SPRING * signs[first] * signs[second] * np.eye(3)
    fc2 = FC2Data(
        fc2_values, _MASSES, units="J/m^2", mass_units="kg",
        compact=False, p2s_map=[0, 1], s2p_map=[0, 1],
    )
    geometry = build_layered_device_mapping(
        _crystal("fc2", duplicate_images),
        fc2,
        transport_supercell_matrix=np.asarray(transform, dtype=np.int64),
    )
    fc3_values = np.zeros((2, 2, 2, 3, 3, 3))
    for indices in product(range(2), repeat=3):
        fc3_values[indices + (0, 0, 0)] = _CUBIC * np.prod([signs[i] for i in indices])
    fc3 = FC3Data(
        fc3_values, _MASSES, units="J/m^3", mass_units="kg",
        compact=False, p2s_map=[0, 1], s2p_map=[0, 1],
    )
    certification = validate_fc3(fc3)
    assert certification.passed and certification.permutation_complete
    crystal = _crystal("fc3", duplicate_images)
    translated = convert_fc3_to_translational(fc3, crystal)
    if r0:
        # Explicitly exercise all three r0 branches. On this symmetric source
        # their average must reproduce the independently expanded polynomial.
        translated = TranslationalFC3(
            terms=translated.terms,
            n_primitive_atoms=translated.n_primitive_atoms,
            n_supercell_atoms=translated.n_supercell_atoms,
            units=translated.units,
            mass_units=translated.mass_units,
            source_layout=translated.source_layout,
            anchor_convention=PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
            make_r0_average=True,
            all_shortest=np.zeros((2, 2, 2), dtype=np.byte),
            symprec=1e-5,
            provenance=translated.provenance,
        )
    normalized = mass_normalize_translational_fc3(
        translated, crystal, source_certification=certification
    )
    return geometry, normalized


def _real_space_polynomial(geometry, mesh_shape, scale):
    """Enumerate physical bond endpoints, using geometry only for DOF ordering."""

    transform = np.asarray(geometry.transport_supercell_matrix, dtype=float)
    inverse = np.linalg.inv(transform)
    cells = _cells(mesh_shape)
    local_atoms = []
    for layer in range(_LAYERS):
        for subcell, basis, cartesian in geometry.principal_layer_dof_records:
            if cartesian != 0:
                continue
            primitive_atom = int(geometry.basis_primitive_atoms[basis])
            primitive_position = (
                _POSITIONS[primitive_atom]
                + geometry.basis_integer_cells[basis]
                + np.array([layer * geometry.principal_cells + subcell, 0, 0]) @ transform
            )
            local_atoms.append((primitive_atom, primitive_position, layer))
    nlocal = len(local_atoms)
    atoms = [
        (
            primitive_atom,
            position + np.array([0, cell[0], cell[1]]) @ transform,
            layer,
        )
        for cell in cells
        for primitive_atom, position, layer in local_atoms
    ]

    def key(atom, position):
        coordinate = position @ inverse
        # Only the two transverse coordinates are periodic. In particular,
        # negative or out-of-device transport positions never wrap.
        transverse = np.mod(
            np.round(coordinate[1:], decimals=9), np.asarray(mesh_shape[:2])
        )
        return (
            atom,
            round(float(coordinate[0]), 9),
            round(float(transverse[0]), 9),
            round(float(transverse[1]), 9),
        )

    lookup = {key(atom, position): index for index, (atom, position, _) in enumerate(atoms)}
    assert len(lookup) == len(atoms)
    anchors = {}
    displacement = _POSITIONS[1] + _BOND_WINDING
    for atom, position, _ in atoms:
        anchor = position if atom == 0 else position - displacement
        anchors.setdefault(key(0, anchor), anchor)
    dimension = len(atoms)
    harmonic = np.zeros((dimension, dimension))
    cubic = np.zeros((dimension, dimension, dimension))
    for anchor in anchors.values():
        endpoints = (lookup.get(key(0, anchor)), lookup.get(key(1, anchor + displacement)))
        for species in product(range(2), repeat=2):
            indices = tuple(endpoints[a] for a in species)
            if any(index is None for index in indices):
                continue
            coefficient = _SPRING * (-1.0) ** sum(species)
            coefficient /= np.sqrt(np.prod(_MASSES[list(species)])) * scale.omega0**2
            harmonic[indices] += coefficient
        for species in product(range(2), repeat=3):
            indices = tuple(endpoints[a] for a in species)
            if any(index is None or atoms[index][2] not in _CROP for index in indices):
                continue
            coefficient = _CUBIC * (-1.0) ** sum(species)
            coefficient /= np.sqrt(np.prod(_MASSES[list(species)]))
            coefficient *= np.sqrt(scale.hbar) / scale.omega0**2.5
            cubic[indices] += coefficient
    return harmonic, cubic, nlocal


@pytest.mark.parametrize(
    "transform,mesh_shape,duplicate_images,r0",
    [
        (np.eye(3, dtype=int), (1, 1, 1), False, False),
        (np.eye(3, dtype=int), (4, 1, 1), False, False),
        (np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]]), (4, 1, 1), True, True),
        (np.eye(3, dtype=int), (2, 2, 1), True, False),
    ],
    ids=("gamma", "complex-four-q", "sheared-cosets-r0-images", "two-transverse-axes"),
)
def test_raw_material_images_match_finite_supercell_dft(
    transform, mesh_shape, duplicate_images, r0,
) -> None:
    geometry, normalized = _raw_inputs(
        transform, duplicate_images=duplicate_images, r0=r0
    )
    scale = SolverScale(
        omega0=2.0, hbar=3.0,
        metadata={"source_units": {"fc2": "J/m^2", "fc3": "J/m^3", "mass": "kg"}},
    )
    harmonic, cubic, nlocal = _real_space_polynomial(geometry, mesh_shape, scale)
    basis = _unitary_basis(mesh_shape, nlocal)
    nq = len(_cells(mesh_shape))
    expected_harmonic = _to_q(harmonic[None], basis, nq)[:, 0]
    expected_cubic = _fourier_vertex(mesh_shape, cubic, basis)
    mapped = map_material_cubic(
        normalized, geometry, n_device_layers=_LAYERS,
        interaction_layers=_CROP, scale=scale,
    )
    assert mapped.ndof == 3 * nlocal
    actual = fourier_transverse_cubic(
        UniformBZMesh(mesh_shape), mapped.ndof, mapped.terms
    ).coefficients
    active = np.arange(0, mapped.ndof, 3)
    full_expected = np.zeros_like(actual)
    full_expected[np.ix_(np.arange(nq), np.arange(nq), active, active, active)] = expected_cubic
    np.testing.assert_allclose(actual, full_expected, rtol=2e-12, atol=2e-13)
    if mesh_shape == (4, 1, 1):
        assert np.max(np.abs(expected_cubic.imag)) > 1e-3
    if duplicate_images:
        assert {term.multiplicity_weight for term in normalized.terms} == {0.25}
    if r0:
        assert mapped.metadata["effective_image_terms"] == 3 * len(normalized.terms)
    assert any(
        any(value != 0 for value in term.second_cell + term.third_cell)
        for term in mapped.terms
    )
    # The third device layer is deliberately outside the cubic crop. Neither
    # a source winding nor a transverse alias may place a cubic leg there.
    crop_end = 2 * geometry.n_principal_layer_dofs
    assert all(max(term.indices) < crop_end for term in mapped.terms)
    assert any(len({index // geometry.n_principal_layer_dofs for index in term.indices}) > 1
               for term in mapped.terms)
    device, lead = material_transverse_harmonic(
        geometry, n_device_layers=_LAYERS, scale=scale
    )
    layer_dofs = geometry.n_principal_layer_dofs
    for q, address in enumerate(UniformBZMesh(mesh_shape).addresses):
        kpar = tuple(2 * np.pi * np.asarray(address[:2]) / np.asarray(mesh_shape[:2]))
        expected = np.kron(expected_harmonic[q], np.eye(3))
        np.testing.assert_allclose(
            assemble_device_dynamical_matrix(device.device(kpar)),
            expected, rtol=2e-12, atol=2e-13,
        )
        blocks = lead.blocks(kpar)
        np.testing.assert_allclose(blocks[0], expected[:layer_dofs, :layer_dofs],
                                   rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(blocks[1], expected[:layer_dofs, layer_dofs:2 * layer_dofs],
                                   rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(blocks[2], expected[layer_dofs:2 * layer_dofs, :layer_dofs],
                                   rtol=2e-12, atol=2e-13)
        np.testing.assert_allclose(expected[:layer_dofs, 2 * layer_dofs:], 0, atol=2e-13)
