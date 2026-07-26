import numpy as np
import pytest

from negfpy.anharmonic._harmonic import (
    build_retarded_contacts,
    retarded_contact_self_energy,
    solve_keldysh_green,
    solve_retarded,
)
from negfpy.anharmonic.contacts import (
    bose_occupation_signed,
    broadening,
    build_contact_spectra,
    minimum_lead_frequency_squared,
    reservoir_keldysh_from_retarded,
)
from negfpy.anharmonic.current import collision_power, contact_heat_current
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    KeldyshSpectrum,
    SolverScale,
)
from negfpy.core.negf import device_green_function, transmission
from negfpy.core.types import Device1D, LeadBlocks


def _gapped_problem(
    *,
    temperature_left: float = 0.8,
    temperature_right: float = 0.6,
    step: float = 0.05,
    n_positive: int = 56,
) -> CubicSCBAProblem:
    # A pinned matched chain has omega^2 = 3 - 2 cos(q), hence a band
    # [1, sqrt(5)] and an unambiguous zero-frequency Bose-Gamma product.
    lead = LeadBlocks(
        d00=np.array([[3.0]]),
        d01=np.array([[-1.0]]),
    )
    n_layers = 3
    device = Device1D(
        onsite_blocks=[np.array([[3.0]]) for _ in range(n_layers)],
        coupling_blocks=[np.array([[-1.0]]) for _ in range(n_layers - 1)],
    )
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead=lead, temperature=temperature_left),
        right=HarmonicReservoir(lead=lead, temperature=temperature_right),
        vertex=DenseCubicVertex(np.zeros((n_layers, n_layers, n_layers))),
        grid=SignedFrequencyGrid(step=step, n_positive=n_positive),
        scale=SolverScale(),
    )


def test_signed_bose_uses_exact_negative_frequency_identity() -> None:
    positive = np.array([1.0e-8, 0.2, 2.0, 1000.0])
    occupation = np.asarray(bose_occupation_signed(positive, temperature=0.7))
    negative = np.asarray(bose_occupation_signed(-positive, temperature=0.7))

    assert np.all(np.isfinite(occupation))
    assert np.allclose(negative, -1.0 - occupation, rtol=1e-14, atol=1e-14)
    assert np.isinf(bose_occupation_signed(0.0, temperature=0.7))


def test_dense_harmonic_adapter_matches_existing_device_green_function() -> None:
    problem = _gapped_problem(step=0.1, n_positive=25)
    eta = 1.0e-7
    sigma_left, sigma_right = build_retarded_contacts(problem, eta)
    green = solve_retarded(problem, sigma_left, sigma_right)

    for omega in (-1.5, 1.5):
        iw = int(round(omega / problem.grid.step)) + problem.grid.zero_index
        reference_g, reference_left, reference_right = device_green_function(
            omega=omega,
            device=problem.device,
            lead_left=problem.left.lead,
            lead_right=problem.right.lead,
            eta=eta,
            eta_device=0.0,
        )
        assert np.allclose(sigma_left[iw], reference_left, rtol=1e-12, atol=1e-12)
        assert np.allclose(sigma_right[iw], reference_right, rtol=1e-12, atol=1e-12)
        assert np.allclose(green[iw], reference_g, rtol=1e-12, atol=1e-12)

    positive = problem.grid.zero_index + 15
    negative = problem.grid.zero_index - 15
    assert np.allclose(sigma_left[negative], sigma_left[positive].conj(), atol=1e-12)
    assert np.allclose(green[negative], green[positive].conj(), atol=1e-12)


def test_harmonic_adapter_matches_core_for_asymmetric_lead_orientation() -> None:
    d00 = np.array([[4.0, 0.2], [0.2, 3.5]], dtype=np.complex128)
    d01 = np.array([[-0.8, 0.35], [-0.1, -0.6]], dtype=np.complex128)
    lead = LeadBlocks(d00=d00, d01=d01, d10=d01.conj().T)
    device = Device1D(onsite_blocks=[d00.copy()], coupling_blocks=[])
    reservoir = HarmonicReservoir(lead=lead, temperature=0.5)
    omega = 1.8
    eta = 1.0e-6

    _, core_left, core_right = device_green_function(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=eta,
        eta_device=0.0,
    )
    adapter_left = retarded_contact_self_energy(
        omega,
        reservoir,
        ndof=2,
        eta_lead=eta,
        side="left",
    )
    adapter_right = retarded_contact_self_energy(
        omega,
        reservoir,
        ndof=2,
        eta_lead=eta,
        side="right",
    )

    np.testing.assert_allclose(
        adapter_left, core_left, rtol=1.0e-12, atol=1.0e-12
    )
    np.testing.assert_allclose(
        adapter_right, core_right, rtol=1.0e-12, atol=1.0e-12
    )


def test_harmonic_adapter_matches_scaled_core_with_asymmetric_leads() -> None:
    d00 = np.array([[4.0, 0.2], [0.2, 3.5]], dtype=np.complex128)
    d01 = np.array([[-0.8, 0.35], [-0.1, -0.6]], dtype=np.complex128)
    lead = LeadBlocks(d00=d00, d01=d01, d10=d01.conj().T)
    device = Device1D(onsite_blocks=[d00.copy()], coupling_blocks=[])
    omega = 1.84
    eta = 2.0e-6
    omega_scale = 2.3

    core_green, core_left, core_right = device_green_function(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=eta,
        eta_device=0.0,
        omega_scale=omega_scale,
    )

    scale_squared = omega_scale**2
    scaled_lead = LeadBlocks(
        d00=d00 / scale_squared,
        d01=d01 / scale_squared,
        d10=d01.conj().T / scale_squared,
    )
    scaled_device = Device1D(
        onsite_blocks=[d00 / scale_squared],
        coupling_blocks=[],
    )
    scaled_omega = omega / omega_scale
    grid = SignedFrequencyGrid(step=scaled_omega / 10.0, n_positive=20)
    problem = CubicSCBAProblem(
        device=scaled_device,
        left=HarmonicReservoir(scaled_lead, temperature=0.5),
        right=HarmonicReservoir(scaled_lead, temperature=0.4),
        vertex=DenseCubicVertex(np.zeros((2, 2, 2))),
        grid=grid,
        scale=SolverScale(omega0=omega_scale),
    )
    adapter_left, adapter_right = build_retarded_contacts(
        problem,
        eta / omega_scale,
    )
    adapter_green = solve_retarded(problem, adapter_left, adapter_right)
    index = grid.zero_index + 10

    np.testing.assert_allclose(
        adapter_left[index], core_left, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        adapter_right[index], core_right, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        adapter_green[index], core_green, rtol=1e-12, atol=1e-12
    )


def test_explicit_rectangular_contact_is_embedded_at_requested_dof() -> None:
    lead = LeadBlocks(d00=np.array([[3.0]]), d01=np.array([[-1.0]]))
    reservoir = HarmonicReservoir(
        lead=lead,
        temperature=0.5,
        device_to_lead=np.array([[-0.4]]),
        device_indices=(1,),
    )
    sigma = retarded_contact_self_energy(
        omega=1.5,
        reservoir=reservoir,
        ndof=2,
        eta_lead=1.0e-7,
        side="left",
    )

    assert sigma.shape == (2, 2)
    assert sigma[1, 1] != 0.0
    assert sigma[0, 0] == 0.0
    assert sigma[0, 1] == 0.0
    assert sigma[1, 0] == 0.0


def test_contact_keldysh_identities_hold_on_signed_grid() -> None:
    problem = _gapped_problem(step=0.1, n_positive=25)
    left, _ = build_contact_spectra(problem, eta_lead=1.0e-8)
    gamma = broadening(left.retarded)

    assert np.allclose(
        left.greater - left.lesser,
        left.retarded - np.swapaxes(left.retarded.conj(), -1, -2),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.allclose(left.lesser[problem.grid.zero_index], 0.0, atol=0.0)
    assert np.allclose(left.greater[problem.grid.zero_index], 0.0, atol=0.0)

    for positive in problem.grid.positive_indices:
        negative = problem.grid.negative_partners[positive]
        assert np.allclose(
            left.lesser[negative],
            left.greater[positive].T,
            rtol=1e-11,
            atol=1e-11,
        )
        assert np.allclose(
            left.greater[negative],
            left.lesser[positive].T,
            rtol=1e-11,
            atol=1e-11,
        )
        assert np.allclose(gamma[negative], -gamma[positive].conj(), atol=1e-11)

    # Check KMS at a propagating positive frequency.
    iw = problem.grid.zero_index + 15
    factor = np.exp(problem.grid.values[iw] / problem.left.temperature)
    assert np.allclose(
        left.greater[iw],
        factor * left.lesser[iw],
        rtol=1e-12,
        atol=1e-12,
    )


def test_zero_frequency_broadening_is_rejected_for_v1() -> None:
    grid = SignedFrequencyGrid(step=0.1, n_positive=1)
    retarded = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    retarded[grid.zero_index, 0, 0] = -0.5j

    with pytest.raises(ValueError, match="gapped reservoir"):
        reservoir_keldysh_from_retarded(grid, retarded, temperature=1.0)


def test_v1_rejects_acoustic_lead_even_when_gamma_at_zero_vanishes() -> None:
    lead = LeadBlocks(d00=np.array([[2.0]]), d01=np.array([[-1.0]]))
    device = Device1D(onsite_blocks=[np.array([[2.0]])], coupling_blocks=[])
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.5),
        right=HarmonicReservoir(lead, 0.4),
        vertex=DenseCubicVertex(np.zeros((1, 1, 1))),
        grid=SignedFrequencyGrid(0.1, 25),
    )

    assert abs(minimum_lead_frequency_squared(lead)) < 1.0e-14
    with pytest.raises(ValueError, match="spectrally gapped"):
        build_contact_spectra(problem, eta_lead=1.0e-6)


def test_gap_check_resolves_small_positive_pinning() -> None:
    lead = LeadBlocks(
        d00=np.array([[2.0 + 1.0e-8]]),
        d01=np.array([[-1.0]]),
    )
    assert minimum_lead_frequency_squared(lead) == pytest.approx(
        1.0e-8,
        rel=1.0e-7,
        abs=1.0e-14,
    )


def test_harmonic_contact_current_reduces_to_discrete_landauer_formula() -> None:
    problem = _gapped_problem(step=0.025, n_positive=104)
    eta = 1.0e-7
    left, right = build_contact_spectra(problem, eta_lead=eta)
    green_r = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        green_r,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )

    current_left = contact_heat_current(problem.grid, left, green, problem.scale)
    current_right = contact_heat_current(problem.grid, right, green, problem.scale)

    gamma_left = broadening(left.retarded)
    gamma_right = broadening(right.retarded)
    advanced = np.swapaxes(green_r.conj(), -1, -2)
    landauer = 0.0
    for iw in problem.grid.positive_indices:
        omega = problem.grid.values[iw]
        n_left = bose_occupation_signed(omega, problem.left.temperature)
        n_right = bose_occupation_signed(omega, problem.right.temperature)
        tval = np.trace(
            gamma_left[iw]
            @ green_r[iw]
            @ gamma_right[iw]
            @ advanced[iw]
        ).real
        landauer += omega * (n_left - n_right) * tval
    landauer *= problem.grid.integration_weight * problem.scale.current_scale

    assert current_left > 0.0
    assert np.isclose(current_left, landauer, rtol=1e-11, atol=1e-12)
    assert np.isclose(current_right, -landauer, rtol=1e-11, atol=1e-12)
    assert abs(current_left + current_right) < 1e-11

    # One in-band point also agrees with the existing Caroli implementation.
    iw = problem.grid.zero_index + 60
    omega = problem.grid.values[iw]
    local_t = np.trace(
        gamma_left[iw] @ green_r[iw] @ gamma_right[iw] @ advanced[iw]
    ).real
    core_t = transmission(
        omega=omega,
        device=problem.device,
        lead_left=problem.left.lead,
        lead_right=problem.right.lead,
        eta=eta,
        eta_device=0.0,
    )
    assert np.isclose(local_t, core_t, rtol=1e-12, atol=1e-12)


def test_equal_temperature_harmonic_contacts_have_zero_current() -> None:
    problem = _gapped_problem(
        temperature_left=0.7,
        temperature_right=0.7,
        step=0.05,
        n_positive=52,
    )
    left, right = build_contact_spectra(problem, eta_lead=1.0e-7)
    green_r = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        green_r,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )

    assert abs(contact_heat_current(problem.grid, left, green, problem.scale)) < 1e-13
    assert abs(contact_heat_current(problem.grid, right, green, problem.scale)) < 1e-13


def test_collision_power_uses_positive_rectangle_grid_and_current_scale() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    shape = (grid.n_points, 1, 1)
    zero = np.zeros(shape, dtype=np.complex128)
    sigma_lesser = zero.copy()
    sigma_greater = zero.copy()
    green_lesser = zero.copy()
    green_greater = zero.copy()
    iw = grid.positive_indices[0]
    sigma_lesser[iw, 0, 0] = -1j
    sigma_greater[iw, 0, 0] = -2j
    green_lesser[iw, 0, 0] = -3j
    green_greater[iw, 0, 0] = -5j
    sigma = KeldyshSpectrum(zero, sigma_lesser, sigma_greater)
    green = KeldyshSpectrum(zero, green_lesser, green_greater)
    scale = SolverScale(omega0=2.0, hbar=3.0)

    # (-2i)(-3i) - (-i)(-5i) = -1 at omega=1.
    expected = -scale.current_scale / (2.0 * np.pi)
    assert np.isclose(collision_power(grid, sigma, green, scale), expected)
