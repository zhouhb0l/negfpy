"""Tiny energy-polynomial oracles for the declared fitted boundary model."""
from dataclasses import replace
from itertools import combinations_with_replacement
from time import monotonic

import numpy as np
import pytest
from test_anharmonic_layered_phase_a import _chain_mapping, _compact_chain_fc2
from test_anharmonic_transverse_material_reference import (
    _BOND_WINDING,
    _crystal,
    _raw_inputs,
)

from negfpy.anharmonic.fc3 import fc3_force_constant_to_si_factor, mass_to_si_factor
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.site_energy_boundary import build_site_energy_vertex
from negfpy.anharmonic.site_energy_model import SiteEnergyModel, SiteEnergyStar
from negfpy.anharmonic.types import SolverScale


def _model(crystal, coefficient=1.7):
    positions = crystal.supercell.scaled_positions.copy()
    lattice = crystal.supercell.lattice.copy()
    monomials = tuple(combinations_with_replacement(range(3), 3))
    coefficients = np.zeros((2, len(monomials)))
    coefficients[0, 0] = coefficient
    stars = []
    for center in range(2):
        neighbor = 1-center
        image = (1 if center == 0 else -1)*_BOND_WINDING
        vector = (positions[neighbor]+image-positions[center])@lattice
        stars.append(SiteEnergyStar(center, center, np.array([neighbor]), image[None], vector[None]))
    return SiteEnergyModel(coefficients, monomials, np.array([0, 1]), np.array([0, 1]),
        np.array([1, 1]), tuple(stars), positions, lattice, {}, {})


def _inputs():
    mapping, _ = _raw_inputs(np.eye(3, dtype=int), duplicate_images=False, r0=False)
    crystal = _crystal("fc3", False)
    return mapping, crystal, _model(crystal)


def _build(mapping, crystal, model, **kwargs):
    return build_site_energy_vertex(UniformBZMesh((4, 1, 1)), model, crystal, mapping,
        n_device_layers=3, interaction_layers=(1,), scale=SolverScale(omega0=2., hbar=3.),
        force_constant_units=kwargs.pop("force_constant_units", "J/m^3"),
        mass_units=kwargs.pop("mass_units", "kg"), **kwargs)


@pytest.mark.parametrize("force_units,mass_units", [("J/m^3", "kg"), ("eV/angstrom^3", "amu")])
def test_complete_star_matches_independent_mass_and_fourier_bond_formula(force_units, mass_units):
    mapping, crystal, model = _inputs()
    original = model.coefficients.copy()
    result = _build(mapping, crystal, model, force_constant_units=force_units, mass_units=mass_units)
    vertex, mesh = result.vertex, result.vertex.mesh
    # The only nonzero class energy is K*(u_B(layer2,transverse+1)-u_A(layer1))^3/6.
    full = vertex.full_ndof
    first = mapping.n_principal_layer_dofs
    second = 2*mapping.n_principal_layer_dofs+3
    roots = np.sqrt(crystal.primitive.masses*mass_to_si_factor(mass_units))
    conversion = fc3_force_constant_to_si_factor(force_units)*np.sqrt(3.)/2.**2.5
    vectors = np.zeros((4, full), complex)
    vectors[:, first] = -1/roots[0]
    vectors[:, second] = np.exp(2j*np.pi*np.arange(4)/4)/roots[1]
    expected = np.zeros_like(vertex.coefficients)
    selected = vertex.active_dofs
    for q in range(4):
        for p in range(4):
            expected[q, p] = 1.7*conversion*np.einsum("i,j,k->ijk",
                vectors[mesh.negative_index(q), selected], vectors[p, selected], vectors[(q-p) % 4, selected])
    np.testing.assert_allclose(vertex.coefficients, expected, rtol=2e-14, atol=np.max(np.abs(expected))*1e-14)
    assert np.max(np.abs(expected.imag)) > 0
    assert result.metadata["selected_center_count"] == 2
    assert result.metadata["coupling_applied"] is False
    assert result.metadata["additional_r0_average"] is False
    assert result.metadata["support"]["complete_required_support"]
    assert vertex.full_ndof == 18
    assert vertex.ndof == 6
    np.testing.assert_array_equal(model.coefficients, original)


def test_complete_center_retains_outer_only_derivatives_and_crop_breaks_asr():
    mapping, crystal, model = _inputs()
    complete, crop = _build(mapping, crystal, model), _build(mapping, crystal, model, crop_derivatives=True)
    second = 2*mapping.n_principal_layer_dofs+3
    index = {int(address): i for i, address in enumerate(complete.vertex.active_dofs)}
    assert complete.vertex.coefficients[0, 0, index[second], index[second], index[second]] != 0
    assert np.all(crop.vertex.active_dofs < 2*mapping.n_principal_layer_dofs)
    masses = np.tile(np.repeat(crystal.primitive.masses, 3), 3)
    translation = np.sqrt(masses[complete.vertex.active_dofs])*(complete.vertex.active_dofs % 3 == 0)
    for p in range(4):
        np.testing.assert_allclose(np.einsum("i,ijk->jk", translation, complete.vertex.coefficients[0, p]), 0, atol=2e-15)
        np.testing.assert_allclose(np.einsum("j,ijk->ik", translation, complete.vertex.coefficients[p, 0]), 0, atol=2e-15)
        np.testing.assert_allclose(np.einsum("k,ijk->ij", translation, complete.vertex.coefficients[p, p]), 0, atol=2e-15)
    crop_translation = np.sqrt(masses[crop.vertex.active_dofs])*(crop.vertex.active_dofs % 3 == 0)
    assert np.linalg.norm(np.einsum("i,ijk->jk", crop_translation, crop.vertex.coefficients[0, 0])) > .01
    assert complete.metadata["energy_identity_sha256"] == crop.metadata["energy_identity_sha256"]
    assert complete.metadata["model_identity_sha256"] != crop.metadata["model_identity_sha256"]
    assert crop.metadata["derivative_crop_is_control"]


def test_actual_nonordinal_s2p_representative_is_mapped_to_primitive_ordinal():
    crystal = _chain_mapping()
    mapping = build_layered_device_mapping(crystal, _compact_chain_fc2(), transport_supercell_matrix=np.eye(3, dtype=int))
    positions = crystal.supercell.scaled_positions.copy()
    monomials = tuple(combinations_with_replacement(range(3), 3))
    coefficients = np.zeros((1, len(monomials)))
    coefficients[0, 0] = 1.
    stars = []
    for center in range(3):
        neighbor = (center+1) % 3
        image = np.array([int(center == 2), 0, 0])
        vector = (positions[neighbor]+image-positions[center])@crystal.supercell.lattice
        stars.append(SiteEnergyStar(center, 0, np.array([neighbor]), image[None], vector[None]))
    model = SiteEnergyModel(coefficients, monomials, np.zeros(3, dtype=int), np.array([1]), np.array([3]),
        tuple(stars), positions, crystal.supercell.lattice.copy(), {}, {})
    actual = _build(mapping, crystal, model)
    assert actual.metadata["primitive_stars"][0]["primitive_center"] == 0
    assert actual.metadata["selected_center_count"] == 1
    assert actual.vertex.full_ndof == 9
    assert actual.metadata["support"]["longitudinal_cell_max"] == 2


def test_mixed_coordinate_factorials_are_not_applied_twice():
    mapping, crystal, model = _inputs()
    coefficients = np.zeros_like(model.coefficients)
    coefficients[0, model.monomials.index((0, 1, 1))] = 2.3
    actual = _build(mapping, crystal, replace(model, coefficients=coefficients)).vertex
    address = {int(v): i for i, v in enumerate(actual.active_dofs)}
    x, y = address[6], address[7]
    value = -2.3*np.sqrt(3.)/(2.**2.5*2.**1.5)
    np.testing.assert_allclose(actual.coefficients[0, 0, x, y, y], value, atol=1e-15)
    np.testing.assert_allclose(actual.coefficients[0, 0, y, x, y], value, atol=1e-15)


@pytest.mark.parametrize("kwargs,match", [
    ({"max_bytes": 1}, "payload exceeds"),
    ({"max_workspace_bytes": 1}, "construction exceeds"),
    ({"max_geometry_evaluations": 1}, "geometry exceeds"),
    ({"max_phase_evaluations": 1}, "Fourier sum exceeds"),
    ({"deadline_monotonic": 0.}, "deadline"),
])
def test_resource_and_time_admission(kwargs, match):
    mapping, crystal, model = _inputs()
    assert monotonic() > 0
    with pytest.raises((ValueError, TimeoutError), match=match):
        _build(mapping, crystal, model, **kwargs)


def test_complete_support_gap_stops_even_for_derivative_crop_control():
    mapping, crystal, model = _inputs()
    for crop in (False, True):
        with pytest.raises(ValueError, match="complete center star leaves"):
            build_site_energy_vertex(UniformBZMesh((1, 1, 1)), model, crystal, mapping,
                n_device_layers=2, interaction_layers=(1,), scale=SolverScale(omega0=2., hbar=3.),
                force_constant_units="J/m^3", mass_units="kg", crop_derivatives=crop)
