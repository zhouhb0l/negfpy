"""Independent mixed-direction polynomial oracles for the second q coordinate.

These synthetic models use no material/checkpoint data.  The energy oracle uses
absolute endpoint Fourier amplitudes, not the mapper's reanchoring, phase table,
momentum router, local-derivative routine or packed-support construction.
"""
from dataclasses import replace
from itertools import combinations_with_replacement, permutations

import numpy as np
import pytest
from test_anharmonic_transverse_material_reference import (
    _crystal,
    _raw_inputs,
    _real_space_polynomial,
)
from test_anharmonic_transverse_q_reference import _fourier_vertex, _unitary_basis

from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.site_energy_boundary import build_site_energy_vertex
from negfpy.anharmonic.site_energy_model import SiteEnergyModel, SiteEnergyStar
from negfpy.anharmonic.transverse_material_packed import fourier_material_cubic_packed
from negfpy.anharmonic.types import SolverScale

_IMAGES = np.array([[1, 1, 2], [0, -2, 1]])
_COEFFICIENTS = {(0, 1, 5): 1.7, (0, 0, 4): -.6, (2, 3, 3): .9, (4, 4, 4): .3}
_SCALE = SolverScale(omega0=2., hbar=3.)


def _synthetic_model(images=_IMAGES):
    mapping, _ = _raw_inputs(np.eye(3, dtype=int), duplicate_images=False, r0=False)
    crystal = _crystal("fc3", False)
    positions, lattice = crystal.supercell.scaled_positions, crystal.supercell.lattice
    monomials = tuple(combinations_with_replacement(range(6), 3))
    coefficients = np.zeros((2, len(monomials)))
    for triple, coefficient in _COEFFICIENTS.items():
        coefficients[0, monomials.index(triple)] = coefficient
    stars = []
    for center in range(2):
        neighbor = 1-center
        periodic_images = (1 if center == 0 else -1)*images
        vectors = (positions[neighbor]+periodic_images-positions[center])@lattice
        stars.append(SiteEnergyStar(center, center, np.full(2, neighbor), periodic_images, vectors))
    model = SiteEnergyModel(coefficients, monomials, np.array([0, 1]), np.array([0, 1]),
        np.array([1, 1]), tuple(stars), positions, lattice, {}, {})
    return mapping, crystal, model


def _build(shape, images=_IMAGES, *, cropped=False):
    mapping, crystal, model = _synthetic_model(images)
    result = build_site_energy_vertex(UniformBZMesh(shape), model, crystal, mapping,
        n_device_layers=3, interaction_layers=(1,), scale=_SCALE,
        force_constant_units="J/m^3", mass_units="kg", crop_derivatives=cropped)
    return result.vertex, crystal, model


def _addresses(shape):
    # Independent X-fast residue enumeration; no mesh routing helper is used.
    return [(x, y) for y in range(shape[1]) for x in range(shape[0])]


def _difference_amplitudes(momentum, images, masses):
    """Fourier amplitudes of six explicit d = u_neighbor - u_center coordinates."""
    result = np.zeros((6, 18), complex)
    for coordinate in range(6):
        neighbor, axis = divmod(coordinate, 3)
        image = images[neighbor]
        result[coordinate, 6+axis] = -1/np.sqrt(masses[0])
        endpoint = 6*(1+int(image[0]))+3+axis
        result[coordinate, endpoint] += np.exp(2j*np.pi*np.dot(momentum, image[1:]))/np.sqrt(masses[1])
    return result


def _polynomial_derivative(momenta, images, masses):
    """Third derivative of sum K*d_a*d_b*d_c/repeated-index-factorials."""
    vectors = [_difference_amplitudes(momentum, images, masses) for momentum in momenta]
    result = np.zeros((18,)*3, complex)
    conversion = np.sqrt(_SCALE.hbar)/_SCALE.omega0**2.5
    for triple, coefficient in _COEFFICIENTS.items():
        # Repeated-coordinate factorials cancel the duplicate permutations.
        for a, b, c in set(permutations(triple)):
            result += coefficient*conversion*np.einsum("i,j,k->ijk", vectors[0][a], vectors[1][b], vectors[2][c])
    return result


@pytest.mark.parametrize("shape", [(2, 2, 1), (3, 4, 1)])
@pytest.mark.parametrize("cropped", [False, True])
def test_complete_energy_matches_absolute_endpoint_polynomial_in_both_q_directions(shape, cropped):
    vertex, crystal, _ = _build(shape, cropped=cropped)
    addresses = _addresses(shape)
    selection = np.ix_(*([vertex.active_dofs]*3))
    for q, output in enumerate(addresses):
        output = np.asarray(output)/shape[:2]
        for p, internal in enumerate(addresses):
            internal = np.asarray(internal)/shape[:2]
            expected = _polynomial_derivative((-output, internal, output-internal),
                _IMAGES, crystal.primitive.masses)[selection]
            np.testing.assert_allclose(vertex.coefficients[q, p], expected, rtol=3e-14, atol=2e-15)
    if shape[1] > 2:
        assert np.max(np.abs(vertex.coefficients.imag)) > .01
        p = addresses.index((0, 1))
        assert np.linalg.norm(vertex.coefficients[0, p]-vertex.coefficients[0, 0]) > .01


def test_swapping_transverse_images_and_q_axes_only_relabels_vertex():
    original, _, _ = _build((3, 4, 1))
    swapped_images = _IMAGES[:, [0, 2, 1]]
    swapped, _, _ = _build((4, 3, 1), swapped_images)
    np.testing.assert_array_equal(original.active_dofs, swapped.active_dofs)
    old = _addresses((3, 4, 1))
    new = _addresses((4, 3, 1))
    indices = [new.index((y, x)) for x, y in old]
    np.testing.assert_allclose(original.coefficients, swapped.coefficients[np.ix_(indices, indices)],
        rtol=3e-14, atol=2e-15)


def test_second_direction_complex_reality_leg_permutations_and_zero_leg_translations():
    vertex, crystal, _ = _build((3, 4, 1))
    addresses = _addresses((3, 4, 1))
    lookup = {address: index for index, address in enumerate(addresses)}
    for q, (qx, qy) in enumerate(addresses):
        negative_q = lookup[-qx % 3, -qy % 4]
        for p, (px, py) in enumerate(addresses):
            negative_p = lookup[-px % 3, -py % 4]
            partner = lookup[(qx-px) % 3, (qy-py) % 4]
            actual = vertex.coefficients[q, p]
            np.testing.assert_allclose(actual, vertex.coefficients[negative_q, negative_p].conj(), atol=2e-15)
            np.testing.assert_allclose(actual, vertex.coefficients[q, partner].transpose(0, 2, 1), atol=2e-15)
            np.testing.assert_allclose(actual, vertex.coefficients[negative_p, negative_q].transpose(1, 0, 2), atol=2e-15)
    mass = np.tile(np.repeat(crystal.primitive.masses, 3), 3)
    for axis in range(3):
        translation = np.sqrt(mass[vertex.active_dofs])*(vertex.active_dofs % 3 == axis)
        for q in range(len(addresses)):
            np.testing.assert_allclose(np.einsum("i,ijk->jk", translation, vertex.coefficients[0, q]), 0, atol=3e-15)
            np.testing.assert_allclose(np.einsum("j,ijk->ik", translation, vertex.coefficients[q, 0]), 0, atol=3e-15)
            np.testing.assert_allclose(np.einsum("k,ijk->ij", translation, vertex.coefficients[q, q]), 0, atol=3e-15)


def test_fitted_vertex_transport_cosets_use_both_transverse_vectors_of_actual_launch_matrix():
    # This is the integer transport matrix of the q22 launch, exercised only on
    # the synthetic two-atom crystal.  Four cosets prevent an identity-coordinate
    # fixture from concealing a primitive/transport-coordinate mix-up.
    transform = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    mapping, _ = _raw_inputs(transform, duplicate_images=False, r0=False)
    _, crystal, model = _synthetic_model()
    coefficients = np.zeros_like(model.coefficients)
    coefficients[0, model.monomials.index((0, 0, 0))] = 1.7
    model = replace(model, coefficients=coefficients)
    shape = (2, 3, 1)
    vertex = build_site_energy_vertex(UniformBZMesh(shape), model, crystal, mapping,
        n_device_layers=5, interaction_layers=(2,), scale=_SCALE,
        force_constant_units="J/m^3", mass_units="kg").vertex
    inverse = np.linalg.inv(transform)
    records = []
    for layer in range(5):
        for subcell in range(mapping.principal_cells):
            for basis, atom in enumerate(mapping.basis_primitive_atoms):
                position = (crystal.primitive.scaled_positions[atom]+mapping.basis_integer_cells[basis]
                    +(layer*mapping.principal_cells+subcell)*transform[0])
                address = layer*mapping.n_principal_layer_dofs+3*(subcell*mapping.n_transport_basis_atoms+basis)
                records.append((int(atom), position, address, layer))
    lookup = {int(address): index for index, address in enumerate(vertex.active_dofs)}
    bonds = []
    for atom, position, center, layer in records:
        if atom != 0 or layer != 2:
            continue
        target = position+crystal.primitive.scaled_positions[1]+_IMAGES[0]-crystal.primitive.scaled_positions[0]
        candidates = []
        for neighbor, reference, endpoint, _ in records:
            if neighbor != 1:
                continue
            displacement = (target-reference)@inverse
            if abs(displacement[0]) < 1e-10 and np.allclose(displacement[1:], np.rint(displacement[1:]), atol=1e-10):
                candidates.append((endpoint, np.rint(displacement[1:]).astype(int)))
        assert len(candidates) == 1
        endpoint, winding = candidates[0]
        bonds.append((lookup[center], lookup[endpoint], winding))
    assert len(bonds) == 4

    def amplitude(momentum, bond):
        center, endpoint, winding = bond
        result = np.zeros(vertex.ndof, complex)
        result[center] = -1/np.sqrt(crystal.primitive.masses[0])
        result[endpoint] += np.exp(2j*np.pi*np.dot(momentum, winding))/np.sqrt(crystal.primitive.masses[1])
        return result

    conversion = 1.7*np.sqrt(_SCALE.hbar)/_SCALE.omega0**2.5
    for q, output in enumerate(_addresses(shape)):
        output = np.asarray(output)/shape[:2]
        for p, internal in enumerate(_addresses(shape)):
            internal = np.asarray(internal)/shape[:2]
            expected = np.zeros((vertex.ndof,)*3, complex)
            for bond in bonds:
                a, b, c = (amplitude(momentum, bond) for momentum in (-output, internal, output-internal))
                expected += conversion*np.einsum("i,j,k->ijk", a, b, c)
            np.testing.assert_allclose(vertex.coefficients[q, p], expected, rtol=5e-14, atol=2e-15)
    assert np.max(np.abs(vertex.coefficients.imag)) > .01


@pytest.mark.parametrize("shape", [(2, 3, 1), (3, 2, 1)])
def test_raw_ifc_packed_vertex_matches_finite_supercell_dft_with_complex_second_direction(shape):
    mapping, normalized = _raw_inputs(np.eye(3, dtype=int), duplicate_images=True, r0=True)
    _, real_cubic, local_atoms = _real_space_polynomial(mapping, shape, _SCALE)
    expected = _fourier_vertex(shape, real_cubic, _unitary_basis(shape, local_atoms))
    actual = fourier_material_cubic_packed(UniformBZMesh(shape), normalized, mapping,
        n_device_layers=3, interaction_layers=(0, 1), scale=_SCALE).vertex
    assert np.all(actual.active_dofs % 3 == 0)
    local_x_indices = actual.active_dofs//3
    wanted = expected[np.ix_(np.arange(len(_addresses(shape))), np.arange(len(_addresses(shape))),
        local_x_indices, local_x_indices, local_x_indices)]
    np.testing.assert_allclose(actual.coefficients, wanted, rtol=3e-13, atol=2e-14)
    assert np.max(np.abs(wanted.imag)) > .01
