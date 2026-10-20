"""Unit boundaries, retained geometry and rejected inputs for the q adapter."""

from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_mapping import (
    TransverseCubicTerm,
    build_transverse_problem,
    fourier_transverse_cubic,
    gapped_contact_spectrum,
    mass_normalize_cubic_terms,
    material_harmonic_providers,
)
from negfpy.anharmonic.types import HarmonicReservoir, KeldyshSpectrum, SolverScale
from negfpy.core.types import Device1D, DeviceKSpace, LeadBlocks
from negfpy.models.material_kspace import MaterialKspaceParams


def _raw_cubic():
    return tuple(
        TransverseCubicTerm(indices, (0, 0), (0, 0), 12, weight=0.25)
        for indices in permutations((0, 1, 2))
    )


def _params(*, explicit_reverse=False):
    directed = np.array([[0, -0.6], [0, 0]])
    forward = {(0, 0): np.diag([-0.2, -0.3]), (1, 0): directed * 0.1}
    reverse = {
        tuple(-v for v in key): value.T.copy()
        for key, value in forward.items()
    } if explicit_reverse else None
    return MaterialKspaceParams(
        masses=np.array([2.0, 8.0]),
        fc00_terms={
            (0, 0): np.array([[8.0, -1], [-1, 32.0]]),
            (1, 0): directed, (-1, 0): directed.T.copy(),
        },
        fc01_terms=forward, fc10_terms=reverse, dof_per_atom=1,
    )


def test_cubic_mass_and_solver_units_and_weight_applied_once():
    source = _raw_cubic()
    scale = SolverScale(omega0=4, hbar=9)
    terms = mass_normalize_cubic_terms(
        source, np.array([2.0, 8.0, 18.0]),
        force_constant_units="J/m^3", mass_units="kg", scale=scale,
    )
    expected = 12 * 3 / (np.sqrt(2 * 8 * 18) * 4**2.5)
    assert all(t.coefficient == pytest.approx(expected) and t.weight == 0.25 for t in terms)
    vertex = fourier_transverse_cubic(UniformBZMesh((4, 1, 1)), 3, terms)
    np.testing.assert_allclose(vertex.coefficients[:, :, 0, 1, 2], expected * 0.25)
    # No extra 1/6, Nq or sqrt(Nq) factor belongs in this reduced vertex.
    assert source == _raw_cubic()
    assert all(t.coefficient == 12 for t in source)


def test_declared_ev_amu_conversion_matches_explicit_si():
    # These constants are the same physical definitions used by the IFC layer.
    electron_volt = 1.602176634e-19
    atomic_mass = 1.66053906660e-27
    scale = SolverScale(omega0=1e13, hbar=1.054571817e-34)
    raw = _raw_cubic()
    actual = mass_normalize_cubic_terms(
        raw, np.array([2, 8, 18]), force_constant_units="eV/angstrom^3",
        mass_units="amu", scale=scale,
    )
    si = mass_normalize_cubic_terms(
        tuple(replace(t, coefficient=t.coefficient * electron_volt / 1e-30) for t in raw),
        np.array([2, 8, 18]) * atomic_mass,
        force_constant_units="J/m^3", mass_units="kg", scale=scale,
    )
    np.testing.assert_allclose(
        [t.coefficient for t in actual], [t.coefficient for t in si],
        rtol=1e-14, atol=0,
    )


@pytest.mark.parametrize("explicit_reverse", [False, True])
def test_harmonic_provider_mass_units_scaling_layers_and_detachment(explicit_reverse):
    params = _params(explicit_reverse=explicit_reverse)
    original = params.fc00_terms[(0, 0)].copy()
    layer_masses = np.array([[2, 8], [4, 16], [6, 24]], dtype=float)
    device, lead = material_harmonic_providers(
        params, n_layers=3, force_constant_units="J/m^2", mass_units="kg",
        scale=SolverScale(omega0=2), layer_masses=layer_masses,
    )
    kpar = (np.pi / 2, 0.0)
    phi00 = sum(block * np.exp(1j * (kpar @ np.array(key)))
                for key, block in params.fc00_terms.items())
    phi01 = sum(block * np.exp(1j * (kpar @ np.array(key)))
                for key, block in params.fc01_terms.items())
    actual = device.device(kpar)
    d00, d01, d10 = lead.blocks(kpar)
    inverse_mass = np.diag(1 / np.sqrt(params.masses))
    np.testing.assert_allclose(d00, inverse_mass @ phi00 @ inverse_mass / 4)
    np.testing.assert_allclose(d01, inverse_mass @ phi01 @ inverse_mass / 4)
    np.testing.assert_allclose(d10, d01.conj().T)
    assert np.max(np.abs(d00.imag)) > 0
    for i, block in enumerate(actual.onsite_blocks):
        m = np.diag(1 / np.sqrt(layer_masses[i]))
        np.testing.assert_allclose(block, m @ phi00 @ m / 4)
    for i, block in enumerate(actual.coupling_blocks):
        ml = np.diag(1 / np.sqrt(layer_masses[i]))
        mr = np.diag(1 / np.sqrt(layer_masses[i + 1]))
        np.testing.assert_allclose(block, ml @ phi01 @ mr / 4)
    np.testing.assert_array_equal(params.fc00_terms[(0, 0)], original)
    saved = actual.onsite_blocks[0].copy()
    params.fc00_terms[(0, 0)][0, 0] += 10
    layer_masses[:] = 99
    np.testing.assert_array_equal(device.device(kpar).onsite_blocks[0], saved)


@pytest.mark.parametrize("factor", [1.0, 1e-25])
def test_explicit_inconsistent_fc10_is_not_silently_replaced(factor):
    params = _params(explicit_reverse=True)
    for mapping in (params.fc00_terms, params.fc01_terms, params.fc10_terms):
        for key in mapping:
            mapping[key] = mapping[key] * factor
    params.fc10_terms[(0, 0)][0, 0] *= 1.2
    with pytest.raises(ValueError, match="FC01/FC10"):
        material_harmonic_providers(
            params, n_layers=2, force_constant_units="J/m^2", mass_units="kg",
            scale=SolverScale(),
        )


def test_missing_nonzero_harmonic_reverse_image_is_rejected():
    params = _params()
    del params.fc00_terms[(-1, 0)]
    with pytest.raises(ValueError, match="FC00"):
        material_harmonic_providers(
            params, n_layers=1, force_constant_units="J/m^2", mass_units="kg",
            scale=SolverScale(),
        )


def test_integer_alias_folding_handles_large_shifts_and_duplicate_images():
    mesh = UniformBZMesh((4, 1, 1))
    coefficient = 0.2
    terms = [
        TransverseCubicTerm((0, 0, 0), (4 * 10**25, 0), (-4 * 10**25, 0),
                            coefficient, weight=0.3),
        TransverseCubicTerm((0, 0, 0), (0, 0), (0, 0), coefficient, weight=0.7),
    ]
    vertex = fourier_transverse_cubic(mesh, 1, terms)
    np.testing.assert_allclose(vertex.coefficients, coefficient, rtol=1e-15, atol=0)


def test_incomplete_ordered_cubic_permutations_are_rejected():
    term = TransverseCubicTerm((0, 1, 0), (0, 0), (0, 0), 0.2)
    with pytest.raises(ValueError, match="exchange"):
        fourier_transverse_cubic(UniformBZMesh((4, 1, 1)), 2, [term])


@pytest.mark.parametrize("kwargs", [
    {"max_bytes": 1}, {"max_phase_evaluations": 1},
])
def test_fourier_budget_rejects_before_vertex_allocation(kwargs):
    with pytest.raises(ValueError, match="exceeds"):
        fourier_transverse_cubic(UniformBZMesh((4, 1, 1)), 3, _raw_cubic(), **kwargs)


def test_gapped_helper_rejects_acoustic_zero_limit():
    reservoir = HarmonicReservoir(LeadBlocks(np.array([[2.0]]), np.array([[-1.0]])), 1.0)
    with pytest.raises(ValueError, match="acoustic/soft"):
        gapped_contact_spectrum(SignedFrequencyGrid(0.2, 2), reservoir, 1, 0.02, side="left")


def test_geometry_change_between_q_points_is_rejected():
    grid = SignedFrequencyGrid(0.3, 2)
    mesh = UniformBZMesh((4, 1, 1))
    vertex = fourier_transverse_cubic(mesh, 2, [])

    def build(kpar):
        return (
            Device1D([np.ones((1, 1))] * 2, [np.zeros((1, 1))])
            if kpar == (0.0, 0.0) else Device1D([np.eye(2)], [])
        )

    zeros = np.zeros((grid.n_points, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="geometry changes"):
        build_transverse_problem(
            grid=grid, vertex=vertex, device=DeviceKSpace(build),
            left=lambda k: KeldyshSpectrum(zeros, zeros, zeros),
            right=lambda k: KeldyshSpectrum(zeros, zeros, zeros),
            input_units="solver",
        )


def test_join_requires_explicit_solver_unit_declaration():
    with pytest.raises(ValueError, match="input_units='solver'"):
        build_transverse_problem(
            grid=None, vertex=None, device=None, left=None, right=None,
            input_units="SI",
        )


@pytest.mark.parametrize("kwargs", [
    {"indices": (0, 0, -1)}, {"indices": (0, 0.5, 0)},
    {"second_cell": (0, 0, 1)}, {"third_cell": (0, False)},
    {"coefficient": 1j}, {"coefficient": np.inf}, {"weight": 0},
])
def test_invalid_term_coordinates_values_or_weights(kwargs):
    values = {"indices": (0, 0, 0), "second_cell": (0, 0), "third_cell": (0, 0),
              "coefficient": 1.0, "weight": 1.0}
    values.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        TransverseCubicTerm(**values)
