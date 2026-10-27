"""Second transverse direction checked against independent real-space oracles.

The FC2 oracle enumerates an anisotropic two-atom bond on a finite periodic
transverse supercell, then applies a full unitary DFT. The contact oracle
inverts a finite, gapped lead directly: it does not call a surface-GF routine,
paired-q builder, derivative descriptor, or cached/material field loader.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from negfpy.anharmonic._harmonic import assemble_device_dynamical_matrix
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_compact_contacts import (
    build_compact_transverse_contact,
)
from negfpy.anharmonic.transverse_mapping import _kpoints
from negfpy.anharmonic.transverse_material import material_transverse_harmonic
from negfpy.anharmonic.types import HarmonicReservoir, SolverScale
from negfpy.core.types import LeadBlocks

_POSITIONS = np.array([[0.0, 0.0, 0.0], [0.25, 0.3, 0.2]])
_MASSES = np.array([2.0, 5.0])
_WINDING = np.array([1, 2, -1])
_SPRING = np.array([[1.7, 0.13, -0.09], [0.13, 2.1, 0.07], [-0.09, 0.07, 1.3]])


def _geometry(transform):
    """One real anisotropic A--B bond with unequal transverse windings."""
    cell = CrystalCell(
        lattice=np.array([[1.1, 0.1, 0.0], [0.0, 1.3, 0.2], [0.1, 0.0, 1.7]]),
        scaled_positions=_POSITIONS, masses=_MASSES, symbols=("A", "B"),
    )
    vectors, multiplicity = [], np.empty((2, 2, 2), dtype=np.int64)
    for atom, anchor in product(range(2), repeat=2):
        displacement = _POSITIONS[atom] - _POSITIONS[anchor]
        if atom != anchor:
            displacement += (1 if atom == 1 else -1) * _WINDING
        multiplicity[atom, anchor] = (1, len(vectors))
        vectors.append(displacement)
    crystal = CrystalMapping(
        primitive_matrix=np.eye(3), supercell_matrix=np.eye(3, dtype=int),
        primitive=cell, supercell=cell, p2s=[0, 1], s2p=[0, 1], p2p=[0, 1],
        shortest_vectors=vectors, multiplicity=multiplicity,
        atomic_permutations=[[0, 1]], translation_permutations=[[0, 1]], purpose="fc2",
    )
    values = np.empty((2, 2, 3, 3))
    for first, second in product(range(2), repeat=2):
        values[first, second] = (-1.0) ** (first + second) * _SPRING
    fc2 = FC2Data(values, _MASSES, units="J/m^2", mass_units="kg", compact=False,
                  p2s_map=[0, 1], s2p_map=[0, 1])
    return build_layered_device_mapping(crystal, fc2, transport_supercell_matrix=transform)


def _real_space_harmonic(geometry, shape, n_layers, omega0):
    """Differentiate all physical bonds, including boundary onsite terms."""
    transform = np.asarray(geometry.transport_supercell_matrix)
    inverse = np.linalg.inv(transform)
    cells = [(i, j) for j in range(shape[1]) for i in range(shape[0])]
    local_atoms = []
    for layer in range(n_layers):
        for subcell, basis, cartesian in geometry.principal_layer_dof_records:
            if cartesian != 0:
                continue
            atom = int(geometry.basis_primitive_atoms[basis])
            position = (_POSITIONS[atom] + geometry.basis_integer_cells[basis]
                        + np.array([layer * geometry.principal_cells + subcell, 0, 0]) @ transform)
            local_atoms.append((atom, position))
    atoms = [(atom, position + np.array([0, i, j]) @ transform)
             for i, j in cells for atom, position in local_atoms]

    def key(atom, position):
        coordinate = position @ inverse
        transverse = np.mod(np.round(coordinate[1:], 9), shape[:2])
        return (atom, round(float(coordinate[0]), 9),
                round(float(transverse[0]), 9), round(float(transverse[1]), 9))

    lookup = {key(atom, position): index for index, (atom, position) in enumerate(atoms)}
    assert len(lookup) == len(atoms)
    displacement = _POSITIONS[1] + _WINDING
    anchors = {}
    for atom, position in atoms:
        anchor = position if atom == 0 else position - displacement
        anchors.setdefault(key(0, anchor), anchor)
    real = np.zeros((3 * len(atoms),) * 2)
    for anchor in anchors.values():
        endpoints = (lookup.get(key(0, anchor)), lookup.get(key(1, anchor + displacement)))
        for first, second in product(range(2), repeat=2):
            if endpoints[first] is None or endpoints[second] is None:
                continue
            row, column = 3 * endpoints[first], 3 * endpoints[second]
            real[row:row + 3, column:column + 3] += (
                (-1.0) ** (first + second) * _SPRING
                / (np.sqrt(_MASSES[first] * _MASSES[second]) * omega0**2)
            )
    return real, cells, 3 * len(local_atoms)


@pytest.mark.parametrize("shape,transform", [
    ((3, 4, 1), np.eye(3, dtype=int)),
    ((4, 3, 1), np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])),
    ((2, 2, 1), np.array([[2, 1, 0], [0, 1, 1], [0, 0, 1]])),
    ((2, 2, 1), np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])),
    ((4, 2, 1), np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])),
], ids=("both-off-nyquist", "transverse-shear", "principal-cosets-and-q22",
        "study-transport-transform-q22", "study-transport-transform-q42"))
def test_fc2_second_direction_matches_full_real_space_dft(shape, transform):
    geometry = _geometry(transform)
    scale = SolverScale(omega0=2.3, hbar=1.0)
    n_layers = 3
    real, cells, ndof = _real_space_harmonic(geometry, shape, n_layers, scale.omega0)
    nq = len(cells)
    scalar_dft = np.array([
        [np.exp(2j * np.pi * (qx * rx / shape[0] + qy * ry / shape[1])) / np.sqrt(nq)
         for qx, qy in cells] for rx, ry in cells
    ])
    basis = np.kron(scalar_dft, np.eye(ndof))
    transformed = basis.conj().T @ real @ basis
    device, lead = material_transverse_harmonic(geometry, n_device_layers=n_layers, scale=scale)
    mesh = UniformBZMesh(shape)
    # This independent rectangular enumeration also checks address order and 2pi conversion.
    periods = np.array(shape[:2])
    centered = (np.array(cells) + periods // 2) % periods - periods // 2
    np.testing.assert_allclose(_kpoints(mesh), 2 * np.pi * centered / periods, atol=2e-15)
    dimensions = geometry.n_principal_layer_dofs
    for q, (qx, qy) in enumerate(cells):
        block = slice(q * ndof, (q + 1) * ndof)
        expected = transformed[block, block]
        kpar = (2 * np.pi * qx / shape[0], 2 * np.pi * qy / shape[1])
        np.testing.assert_allclose(assemble_device_dynamical_matrix(device.device(kpar)),
                                   expected, rtol=2e-12, atol=2e-13)
        d00, d01, d10 = lead.blocks(kpar)
        np.testing.assert_allclose(d00, expected[:dimensions, :dimensions], atol=2e-13)
        np.testing.assert_allclose(d01, expected[:dimensions, dimensions:2 * dimensions], atol=2e-13)
        np.testing.assert_allclose(d10, expected[dimensions:2 * dimensions, :dimensions], atol=2e-13)
        for other in range(nq):
            if other != q:
                np.testing.assert_allclose(transformed[block, other * ndof:(other + 1) * ndof],
                                           0, atol=2e-13)
    # Mixed and anisotropic winding makes dropping or swapping transverse axes observable.
    sample = (2 * np.pi / shape[0], 2 * np.pi / shape[1])
    actual = assemble_device_dynamical_matrix(device.device(sample))
    dropped = assemble_device_dynamical_matrix(device.device((sample[0], 0.0)))
    assert np.max(np.abs(actual - dropped)) > 1e-3


def _reservoir(kpar, *, swap=False):
    x, y = kpar[::-1] if swap else kpar
    a, b, c = np.exp(1j * (x + 2 * y)), np.exp(1j * (2 * x - y)), np.exp(1j * (x + y))
    d00 = np.array([[4 + 0.15 * np.cos(x) + 0.3 * np.cos(y), -0.17 - 0.09 * a],
                    [-0.17 - 0.09 * a.conjugate(), 5 + 0.2 * np.cos(x) + 0.1 * np.cos(y)]])
    d01 = np.array([[-0.7, -0.08 * b], [-0.03 * c, -0.6]])
    coupling = np.array([[-0.6, -0.07 * a], [0.03 * b.conjugate(), -0.5]])
    return HarmonicReservoir(LeadBlocks(d00, d01), 0.8, device_to_lead=coupling,
                             device_indices=(4, 1))


def _finite_lead_self_energy(kpar, omega, eta, side):
    """Independent 24-cell open-ended lead inversion (all frequencies below band)."""
    reservoir = _reservoir(kpar)
    d00, d01 = reservoir.lead.d00, reservoir.lead.d01
    forward = d01.conj().T if side == "left" else d01
    chain = np.kron(np.eye(24), d00)
    chain += np.kron(np.eye(24, k=1), forward) + np.kron(np.eye(24, k=-1), forward.conj().T)
    surface = np.linalg.inv((omega + 1j * eta)**2 * np.eye(48) - chain)[:2, :2]
    coupling = reservoir.device_to_lead
    return coupling @ surface @ coupling.conj().T


@pytest.mark.parametrize("side", ["left", "right"])
def test_second_direction_contacts_match_finite_lead_and_thermal_oracle(side):
    shape = (2, 3, 1)
    mesh = UniformBZMesh(shape)
    grid = SignedFrequencyGrid(0.2, 2)
    eta, temperature = 0.02, 0.8
    contact = build_compact_transverse_contact(mesh, grid, _reservoir, 6, eta, side=side)
    cells = [(x, y) for y in range(shape[1]) for x in range(shape[0])]
    for index, (x, y) in enumerate(cells):
        kpar = (2 * np.pi * x / shape[0], 2 * np.pi * y / shape[1])
        for w, omega in enumerate(grid.values):
            expected = _finite_lead_self_energy(kpar, float(omega), eta, side)
            np.testing.assert_allclose(contact.spectrum.retarded[index, w], expected,
                                       rtol=2e-12, atol=2e-13)
            if omega != 0:
                gamma = 1j * (expected - expected.conj().T)
                bose = 1 / np.expm1(omega / temperature)
                np.testing.assert_allclose(contact.spectrum.lesser[index, w], -1j * bose * gamma,
                                           rtol=2e-10, atol=2e-13)
                np.testing.assert_allclose(contact.spectrum.greater[index, w], -1j * (bose + 1) * gamma,
                                           rtol=2e-10, atol=2e-13)
        delta = 1e-5
        near_zero = _finite_lead_self_energy(kpar, delta, eta, side)
        expected_limit = 1j * (near_zero - near_zero.conj().T) / delta
        np.testing.assert_allclose(contact.zero_limits[index], expected_limit,
                                   rtol=2e-7, atol=2e-10)
        zero = grid.zero_index
        np.testing.assert_allclose(contact.spectrum.lesser[index, zero],
                                   -1j * temperature * expected_limit, rtol=2e-7, atol=2e-10)
        np.testing.assert_array_equal(contact.spectrum.lesser[index, zero],
                                      contact.spectrum.greater[index, zero])
        np.testing.assert_array_equal(contact.contact.at_q((kpar[0], kpar[1] - 2 * np.pi)).retarded,
                                      contact.spectrum.retarded[index])
    # Positive and negative second-direction sectors really are complex and distinct.
    assert np.max(np.abs(contact.zero_limits[2].imag)) > 1e-6
    assert np.max(np.abs(contact.spectrum.retarded[0] - contact.spectrum.retarded[2])) > 1e-3


def test_contacts_are_covariant_under_transverse_axis_relabeling():
    mesh, swapped_mesh = UniformBZMesh((2, 3, 1)), UniformBZMesh((3, 2, 1))
    grid = SignedFrequencyGrid(0.2, 2)
    original = build_compact_transverse_contact(mesh, grid, _reservoir, 6, 0.02, side="right")
    swapped = build_compact_transverse_contact(
        swapped_mesh, grid, lambda point: _reservoir(point, swap=True), 6, 0.02, side="right",
    )
    for q, (x, y, _) in enumerate(mesh.addresses):
        partner = swapped_mesh.index_from_address((int(y), int(x), 0))
        for name in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(getattr(original.spectrum, name)[q],
                                       getattr(swapped.spectrum, name)[partner], atol=2e-13)
        np.testing.assert_allclose(original.zero_limits[q], swapped.zero_limits[partner], atol=2e-13)


@pytest.mark.parametrize("corrupt", ["lead", "coupling"])
def test_second_direction_only_conjugacy_error_is_rejected_before_surface_work(monkeypatch, corrupt):
    from negfpy.anharmonic import transverse_contacts as module

    def source(point):
        raw = _reservoir(point)
        d00, coupling = raw.lead.d00.copy(), raw.device_to_lead.copy()
        # Odd *real* addition violates q/-q conjugacy only when direction2 !=0.
        if corrupt == "lead":
            d00[0, 0] += 0.01 * np.sin(point[1])
        else:
            coupling[0, 0] += 0.01 * np.sin(point[1])
        return HarmonicReservoir(LeadBlocks(d00, raw.lead.d01), raw.temperature,
                                 device_to_lead=coupling, device_indices=raw.device_indices)

    monkeypatch.setattr(module, "_classify_lead_spectrum",
                        lambda *args, **kwargs: pytest.fail("bad source reached surface work"))
    with pytest.raises(ValueError, match="opposite-q conjugacy"):
        build_compact_transverse_contact(
            UniformBZMesh((2, 3, 1)), SignedFrequencyGrid(0.2, 2), source, 6, 0.02, side="left",
        )
