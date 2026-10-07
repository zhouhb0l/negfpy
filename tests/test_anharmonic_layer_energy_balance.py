"""Independent checks for opt-in layer equation-of-motion bookkeeping."""

from dataclasses import replace

import numpy as np
import pytest

from negfpy.anharmonic._harmonic import (
    assemble_device_dynamical_matrix,
    solve_keldysh_green,
    solve_retarded,
)
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.current import (
    _evaluate_current_report,
    bond_heat_current_spectra,
    energy_exchange_spectrum,
    harmonic_cut_current_contribution_spectra,
)
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.research_layer_balance import evaluate_layer_energy_balance
from negfpy.anharmonic.solver import solve_one_shot_born
from negfpy.anharmonic.toy import fpu_alpha_chain, pinned_scalar_chain
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    KeldyshSpectrum,
    SCBAConfig,
)
from negfpy.core.types import Device1D


def _problem(*, fpu=False):
    factory = fpu_alpha_chain if fpu else pinned_scalar_chain
    options = {"alpha": 0.05} if fpu else {"cubic": 0.0}
    device, lead, vertex = factory(4, pinning=1.0, **options)
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, temperature=0.8),
        right=HarmonicReservoir(lead, temperature=0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=28),
    )


def _harmonic_state(problem):
    left, right = build_contact_spectra(problem, eta_lead=0.03)
    retarded = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )
    sigma = KeldyshSpectrum.zeros(
        n_frequency=problem.grid.n_points, ndof=problem.ndof,
    )
    return left, right, sigma, green


def _assert_scope(report):
    assert report["equation_balance_only"] is True
    assert report["full_spatial_current_available"] is False
    assert report["spatial_validation_passed"] is False
    assert report["changes_solver_acceptance"] is False
    assert report["global_frequency_accuracy_established"] is False
    assert report["quadrature"] == "original-positive-grid-rectangle"
    assert "passed" not in report


def _assert_integrals(problem, arrays):
    weight = problem.grid.integration_weight
    for spectral, integrated in (
        ("harmonic_cut_spectra", "harmonic_cut_currents"),
        ("contact_eom_spectra", "contact_eom_layer_exchange"),
        ("interaction_eom_spectra", "interaction_eom_layer_exchange"),
        ("layer_collision_spectra", "layer_collision_power"),
        ("eom_residual_spectra", "eom_layer_residual"),
    ):
        np.testing.assert_allclose(
            arrays[integrated], weight * arrays[spectral].sum(axis=-1),
            rtol=1e-13, atol=1e-15,
        )


def test_harmonic_layer_balance_matches_contacts_and_dense_commutator():
    problem = _problem()
    left, right, sigma, green = _harmonic_state(problem)
    report, arrays = evaluate_layer_energy_balance(
        problem, left, right, sigma, green, rtol=1e-10, atol=1e-13,
    )
    _assert_scope(report)
    _assert_integrals(problem, arrays)
    assert report["diagnostic_available"]
    assert report["eom_balance_passed"]
    assert report["spectral_balance_passed"]
    assert report["integrated_balance_passed"]
    assert report["harmonic_bond_operator_supported"]
    np.testing.assert_array_equal(arrays["interaction_eom_spectra"], 0.0)
    np.testing.assert_array_equal(arrays["layer_collision_spectra"], 0.0)
    currents = _evaluate_current_report(problem, left, right, sigma, green)
    np.testing.assert_allclose(
        arrays["harmonic_cut_currents"], currents.left, rtol=1e-11, atol=1e-13,
    )

    # A full dense commutator is independent of the implementation's cut loop.
    dynamical = assemble_device_dynamical_matrix(problem.device)
    positive = problem.grid.positive_indices
    commutator = dynamical @ green.lesser - green.lesser @ dynamical
    npl = problem.device.dof_per_layer
    expected = np.empty_like(arrays["harmonic_divergence_spectra"])
    for layer in range(problem.device.n_layers):
        sl = slice(layer * npl, (layer + 1) * npl)
        expected[layer] = -problem.scale.current_scale * problem.grid.values[positive] * (
            np.trace(commutator[positive, sl, sl], axis1=-2, axis2=-1).real
        )
    np.testing.assert_allclose(
        arrays["harmonic_divergence_spectra"], expected, rtol=1e-12, atol=1e-13,
    )


@pytest.fixture(scope="module")
def cross_cut_born():
    problem = _problem(fpu=True)
    result = solve_one_shot_born(problem, SCBAConfig(eta_lead=0.03))
    left, right = build_contact_spectra(problem, eta_lead=0.03)
    return problem, left, right, result


def test_cross_cut_born_has_layer_diagnostics_but_no_spatial_certification(cross_cut_born):
    problem, left, right, result = cross_cut_born
    assert np.max(np.abs(result.sigma_pp.lesser)) > 1e-8
    report, arrays = evaluate_layer_energy_balance(
        problem, left, right, result.sigma_pp, result.green,
        rtol=1e-10, atol=1e-12,
    )
    _assert_scope(report)
    _assert_integrals(problem, arrays)
    assert report["eom_balance_passed"]
    assert not report["harmonic_bond_operator_supported"]
    assert result.currents.spatial is None
    assert result.currents.bond_currents is None
    with pytest.raises(NotImplementedError, match="crosses a layer boundary"):
        bond_heat_current_spectra(problem, result.green)
    np.testing.assert_allclose(
        arrays["harmonic_cut_spectra"],
        harmonic_cut_current_contribution_spectra(problem, result.green),
    )
    global_collision = problem.scale.current_scale * energy_exchange_spectrum(
        problem.grid, result.sigma_pp, result.green,
    )[problem.grid.positive_indices]
    np.testing.assert_allclose(
        arrays["layer_collision_spectra"].sum(axis=0), global_collision,
        rtol=1e-12, atol=1e-13,
    )
    np.testing.assert_allclose(
        arrays["interaction_eom_spectra"].sum(axis=0), global_collision,
        rtol=1e-10, atol=1e-12,
    )
    # The one-shot Dyson solution satisfies this identity without being a
    # self-consistent or independently validated cubic-current solution.
    assert np.max(np.abs(arrays["interaction_eom_spectra"])) > 1e-8


def test_contacts_on_interior_layers_are_projected_at_their_actual_locations():
    original = _problem()
    problem = replace(
        original,
        left=replace(original.left, device_indices=np.array([1])),
        right=replace(original.right, device_indices=np.array([2])),
    )
    left, right, sigma, green = _harmonic_state(problem)
    report, arrays = evaluate_layer_energy_balance(
        problem, left, right, sigma, green, rtol=1e-10, atol=1e-12,
    )
    assert report["eom_balance_passed"]
    projected = arrays["contact_eom_spectra"]
    np.testing.assert_allclose(projected[0, [0, 2, 3]], 0.0, atol=1e-14)
    np.testing.assert_allclose(projected[1, [0, 1, 3]], 0.0, atol=1e-14)
    assert np.max(np.abs(projected[0, 1])) > 1e-5
    assert np.max(np.abs(projected[1, 2])) > 1e-5
    np.testing.assert_allclose(arrays["harmonic_cut_currents"][[0, 2]], 0.0, atol=1e-12)
    assert arrays["harmonic_cut_currents"][1] > 1e-5


def test_one_contact_can_coherently_couple_several_nonadjacent_layers():
    original = _problem()
    problem = replace(
        original,
        left=replace(
            original.left,
            device_indices=np.array([0, 2]),
            device_to_lead=np.array([[-0.6], [-0.2]]),
        ),
        right=replace(
            original.right,
            device_indices=np.array([1, 3]),
            device_to_lead=np.array([[-0.3], [-0.7]]),
        ),
    )
    left, right, sigma, green = _harmonic_state(problem)
    report, arrays = evaluate_layer_energy_balance(
        problem, left, right, sigma, green, rtol=1e-10, atol=1e-12,
    )
    assert report["eom_balance_passed"]
    for contact_index, contact in enumerate((left, right)):
        expected = problem.scale.current_scale * energy_exchange_spectrum(
            problem.grid, contact, green,
        )[problem.grid.positive_indices]
        np.testing.assert_allclose(
            arrays["contact_eom_spectra"][contact_index].sum(axis=0),
            expected, rtol=1e-11, atol=1e-12,
        )
    np.testing.assert_allclose(arrays["contact_eom_spectra"][0, [1, 3]], 0.0, atol=1e-14)
    np.testing.assert_allclose(arrays["contact_eom_spectra"][1, [0, 2]], 0.0, atol=1e-14)


def test_inconsistent_green_field_fails_eom_balance_without_inventing_flux():
    problem = _problem()
    left, right, sigma, green = _harmonic_state(problem)
    lesser, greater = green.lesser.copy(), green.greater.copy()
    iw = problem.grid.positive_indices[15]
    # Preserve anti-Hermiticity and G> - G< while violating the Dyson equation.
    lesser[iw, 0, 0] -= 0.05j
    greater[iw, 0, 0] -= 0.05j
    broken = KeldyshSpectrum(green.retarded, lesser, greater)
    report, arrays = evaluate_layer_energy_balance(
        problem, left, right, sigma, broken, rtol=1e-10, atol=1e-13,
    )
    _assert_scope(report)
    assert not report["eom_balance_passed"]
    assert not report["spectral_balance_passed"]
    assert not report["integrated_balance_passed"]
    assert np.max(np.abs(arrays["eom_residual_spectra"])) > 1e-5


def test_nonzero_device_eta_is_rejected_as_an_unrepresented_source():
    problem = _problem()
    fields = _harmonic_state(problem)
    with pytest.raises(ValueError, match="eta_device"):
        evaluate_layer_energy_balance(problem, *fields, eta_device=1e-6)


def test_complex_multidof_projections_match_full_matrix_products_and_commutator():
    rng = np.random.default_rng(314159)

    def complex_array(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    onsite_raw = complex_array((3, 2, 2))
    device = Device1D(
        onsite_blocks=[block + block.conj().T for block in onsite_raw],
        coupling_blocks=list(complex_array((2, 2, 2))),
    )
    problem = replace(
        _problem(), device=device,
        vertex=DenseCubicVertex(np.zeros((6, 6, 6))),
        grid=SignedFrequencyGrid(step=0.2, n_positive=4),
    )
    shape = (problem.grid.n_points, problem.ndof, problem.ndof)

    def random_spectrum():
        retarded = complex_array(shape)
        raw = complex_array(shape)
        lesser = raw - raw.conj().swapaxes(-1, -2)
        greater = lesser + retarded - retarded.conj().swapaxes(-1, -2)
        return KeldyshSpectrum(retarded, lesser, greater)

    left, right, sigma, green = (random_spectrum() for _ in range(4))
    report, arrays = evaluate_layer_energy_balance(
        problem, left, right, sigma, green, rtol=1e-12, atol=1e-13,
    )
    _assert_scope(report)
    # These fields obey the spectral identities but are not a Dyson solution.
    assert not report["eom_balance_passed"]

    positive = problem.grid.positive_indices
    factor = problem.scale.current_scale * problem.grid.values[positive]

    def dense_layer_projection(kernel):
        return np.stack([
            factor * np.trace(
                kernel[positive, 2 * layer:2 * layer + 2, 2 * layer:2 * layer + 2],
                axis1=-2, axis2=-1,
            ).real
            for layer in range(problem.device.n_layers)
        ])

    green_a = green.retarded.conj().swapaxes(-1, -2)
    for index, self_energy in enumerate((left, right, sigma)):
        sigma_a = self_energy.retarded.conj().swapaxes(-1, -2)
        kernel = (
            self_energy.retarded @ green.lesser + self_energy.lesser @ green_a
            - green.retarded @ self_energy.lesser - green.lesser @ sigma_a
        )
        actual = (
            arrays["contact_eom_spectra"][index]
            if index < 2 else arrays["interaction_eom_spectra"]
        )
        np.testing.assert_allclose(
            actual, dense_layer_projection(kernel), rtol=1e-12, atol=1e-12,
        )

    collision = sigma.greater @ green.lesser - sigma.lesser @ green.greater
    np.testing.assert_allclose(
        arrays["layer_collision_spectra"], dense_layer_projection(collision),
        rtol=1e-12, atol=1e-12,
    )
    dynamical = assemble_device_dynamical_matrix(device)
    commutator = dynamical @ green.lesser - green.lesser @ dynamical
    np.testing.assert_allclose(
        arrays["harmonic_divergence_spectra"], -dense_layer_projection(commutator),
        rtol=1e-12, atol=1e-12,
    )
