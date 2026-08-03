import numpy as np
import pytest

from negfpy.anharmonic.acoustic_contacts import (
    _build_acoustic_contact_descriptor,
    _classify_lead_spectrum,
    _reservoir_keldysh_from_retarded_acoustic,
    _stable_bose_broadening_products,
    classify_harmonic_lead,
    harmonic_lead_physics_sha256,
    scalar_acoustic_contact_limit,
)
from negfpy.anharmonic.contacts import broadening, build_contact_spectra
from negfpy.anharmonic._harmonic import solve_keldysh_green, solve_retarded
from negfpy.anharmonic.current import contact_heat_current
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.layered import LayeredHarmonicModel
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    SolverScale,
)
from negfpy.core.types import Device1D, LeadBlocks


def _scalar_lead(t: float = 1.0, *, pinning: float = 0.0) -> LeadBlocks:
    return LeadBlocks(
        d00=np.array([[2.0 * t + pinning]], dtype=float),
        d01=np.array([[-t]], dtype=float),
    )


def _scalar_problem(
    *,
    t: float = 1.0,
    temperature: float = 0.7,
    step: float = 0.05,
    n_positive: int = 50,
) -> CubicSCBAProblem:
    lead = _scalar_lead(t)
    return CubicSCBAProblem(
        device=Device1D(onsite_blocks=[np.array([[2.0 * t]])], coupling_blocks=[]),
        left=HarmonicReservoir(lead, temperature),
        right=HarmonicReservoir(lead, temperature),
        vertex=DenseCubicVertex(np.zeros((1, 1, 1))),
        grid=SignedFrequencyGrid(step, n_positive),
    )


def test_scalar_acoustic_classification_and_exact_finite_eta_derivative() -> None:
    t = 1.3
    eta = 0.01
    lead = _scalar_lead(t)
    classification = _classify_lead_spectrum(lead)

    assert classification.classification == "acoustic"
    assert classification.passed
    assert classification.gamma_nullity == 1
    assert classification.normalized_asr_residual < 1.0e-14
    assert classification.curvature_consistent
    assert np.all(classification.curvature_eigenvalues_h > 0.0)
    assert len(classification.lead_physics_sha256) == 64

    descriptor = _build_acoustic_contact_descriptor(
        HarmonicReservoir(lead, 0.7),
        side="left",
        ndof=1,
        eta_lead=eta,
        surface_gf_method="analytic_diagonal",
        spectral_classification=classification,
    )
    expected = scalar_acoustic_contact_limit(t, eta)
    np.testing.assert_allclose(
        descriptor.contact_limit,
        np.array([[expected]]),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert descriptor.derivative_residual <= 1.0e-10
    assert descriptor.derivative_condition * descriptor.derivative_residual <= 1.0e-8
    assert descriptor.passivity_minimum_eigenvalue >= -1.0e-12


def test_scalar_eta_ladder_approaches_physical_limit() -> None:
    t = 1.0
    physical = 2.0 * np.sqrt(t)
    values = [scalar_acoustic_contact_limit(t, eta) for eta in (0.05, 0.02, 0.01, 0.005)]

    assert np.all(np.diff(values) > 0.0)
    assert abs(values[-1] - physical) / physical < 5.0e-3


def test_rotated_uncoupled_acoustic_channels_match_scalar_sum() -> None:
    spring_constants = np.array([0.7, 1.2, 2.0])
    random = np.random.default_rng(20260803)
    rotation, _ = np.linalg.qr(random.normal(size=(3, 3)))
    d00 = rotation @ np.diag(2.0 * spring_constants) @ rotation.T
    d01 = rotation @ np.diag(-spring_constants) @ rotation.T
    lead = LeadBlocks(d00=d00, d01=d01, d10=d01.T)
    classification = _classify_lead_spectrum(lead)

    assert classification.classification == "acoustic"
    assert classification.gamma_nullity == 3
    descriptor = _build_acoustic_contact_descriptor(
        HarmonicReservoir(lead, 0.6),
        side="right",
        ndof=3,
        eta_lead=0.05,
        surface_gf_method="sancho_rubio",
        spectral_classification=classification,
    )
    expected = rotation @ np.diag(
        [scalar_acoustic_contact_limit(t, 0.05) for t in spring_constants]
    ) @ rotation.T
    np.testing.assert_allclose(
        descriptor.contact_limit,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_material_classification_joins_rigid_subspace_and_hashes() -> None:
    spring_constants = np.array([0.8, 1.1, 1.7])
    random = np.random.default_rng(20260804)
    rigid, _ = np.linalg.qr(random.normal(size=(3, 3)))
    d00 = rigid @ np.diag(2.0 * spring_constants) @ rigid.T
    d01 = rigid @ np.diag(-spring_constants) @ rigid.T
    model = object.__new__(LayeredHarmonicModel)
    object.__setattr__(model, "d00", d00)
    object.__setattr__(model, "d01", d01)
    object.__setattr__(model, "d10", d01.T)
    object.__setattr__(model, "lead", LeadBlocks(d00=d00, d01=d01, d10=d01.T))
    object.__setattr__(model, "rigid_translation_vectors", rigid)
    object.__setattr__(model, "scale", SolverScale())
    object.__setattr__(model, "fc2_content_sha256", "1" * 64)
    object.__setattr__(model, "layered_geometry_sha256", "2" * 64)

    report = classify_harmonic_lead(model)
    assert report.classification == "acoustic"
    assert report.passed
    assert report.gamma_nullity == 3
    assert report.rigid_subspace_projector_error < 1.0e-12
    assert report.normalized_asr_residual < 1.0e-14
    assert report.lead_physics_sha256 == harmonic_lead_physics_sha256(model.lead)
    assert len(report.classification_sha256) == 64
    with pytest.raises(ValueError):
        report.gamma_eigenvalues[0] = 1.0


def test_acoustic_exact_zero_and_signed_frequency_identities() -> None:
    problem = _scalar_problem()
    eta = 0.05
    left, _ = build_contact_spectra(
        problem,
        eta_lead=eta,
        surface_gf_method="analytic_diagonal",
    )
    zero = problem.grid.zero_index
    expected_zero = -1.0j * problem.left.temperature * scalar_acoustic_contact_limit(
        1.0,
        eta,
    )
    np.testing.assert_allclose(left.lesser[zero, 0, 0], expected_zero, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(left.greater[zero, 0, 0], expected_zero, rtol=1e-10, atol=1e-12)

    advanced = np.swapaxes(left.retarded.conj(), -1, -2)
    np.testing.assert_allclose(
        left.greater - left.lesser,
        left.retarded - advanced,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        left.lesser + np.swapaxes(left.lesser.conj(), -1, -2),
        0.0,
        atol=1.0e-12,
    )
    for positive in problem.grid.positive_indices:
        negative = problem.grid.negative_partners[positive]
        np.testing.assert_allclose(
            left.lesser[negative],
            left.greater[positive].T,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            left.greater[negative],
            left.lesser[positive].T,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    gamma = broadening(left.retarded)
    for positive in problem.grid.positive_indices:
        eigenvalues = np.linalg.eigvalsh(gamma[positive])
        assert np.min(eigenvalues) >= -1.0e-10 * max(1.0, np.max(np.abs(eigenvalues)))

    selected = problem.grid.zero_index + 20
    kms_factor = np.exp(
        problem.grid.values[selected] / problem.left.temperature
    )
    np.testing.assert_allclose(
        left.greater[selected],
        kms_factor * left.lesser[selected],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_acoustic_harmonic_current_passes_accepted_eta_refinement() -> None:
    lead = _scalar_lead()
    problem = CubicSCBAProblem(
        device=Device1D(onsite_blocks=[np.array([[2.0]])], coupling_blocks=[]),
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=DenseCubicVertex(np.zeros((1, 1, 1))),
        grid=SignedFrequencyGrid(0.01, 300),
    )
    currents: list[float] = []
    for eta in (0.01, 0.005):
        left, right = build_contact_spectra(
            problem,
            eta_lead=eta,
            surface_gf_method="analytic_diagonal",
        )
        green_retarded = solve_retarded(
            problem,
            left.retarded,
            right.retarded,
        )
        green = solve_keldysh_green(
            green_retarded,
            left.lesser + right.lesser,
            left.greater + right.greater,
        )
        currents.append(contact_heat_current(problem.grid, left, green, problem.scale))
    assert currents[0] > 0.0
    assert abs(currents[1] - currents[0]) / abs(currents[1]) < 5.0e-3


def test_stable_small_x_products_retain_cubic_series_and_kms() -> None:
    temperature = 0.8
    omega = 1.0e-8
    gamma = np.array([[2.0e-8, 0.3e-8], [0.3e-8, 1.0e-8]], dtype=np.complex128)
    lesser_product, greater_product = _stable_bose_broadening_products(
        omega,
        temperature,
        gamma,
    )
    occupation = 1.0 / np.expm1(omega / temperature)
    np.testing.assert_allclose(lesser_product, occupation * gamma, rtol=2.0e-16, atol=1e-16)
    np.testing.assert_allclose(
        greater_product,
        (1.0 + occupation) * gamma,
        rtol=2.0e-16,
        atol=1e-16,
    )
    np.testing.assert_allclose(greater_product - lesser_product, gamma, atol=1e-16)


@pytest.mark.parametrize(
    ("lead", "expected"),
    [
        (_scalar_lead(pinning=0.2), "gapped"),
        (LeadBlocks(d00=np.array([[1.5]]), d01=np.array([[-1.0]])), "unstable"),
        (
            LeadBlocks(
                d00=np.diag([2.0, 0.0]),
                d01=np.diag([-1.0, 0.0]),
            ),
            "invalid-soft",
        ),
    ],
)
def test_fail_closed_spectral_classifications(lead: LeadBlocks, expected: str) -> None:
    classification = _classify_lead_spectrum(lead)
    assert classification.classification == expected
    assert classification.passed is (expected == "gapped")


def test_non_gamma_soft_mode_is_rejected() -> None:
    # D(q) = 2 + 2 cos(q) is soft at pi rather than Gamma.
    lead = LeadBlocks(d00=np.array([[2.0]]), d01=np.array([[1.0]]))
    classification = _classify_lead_spectrum(lead)
    assert classification.classification == "invalid-soft"
    assert any("not at Gamma" in reason for reason in classification.failure_reasons)


def test_descriptor_rejects_mismatched_retarded_zero() -> None:
    lead = _scalar_lead()
    reservoir = HarmonicReservoir(lead, 0.7)
    descriptor = _build_acoustic_contact_descriptor(
        reservoir,
        side="left",
        ndof=1,
        eta_lead=0.05,
        surface_gf_method="analytic_diagonal",
    )
    grid = SignedFrequencyGrid(0.1, 2)
    retarded = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match="does not match the descriptor"):
        _reservoir_keldysh_from_retarded_acoustic(
            grid,
            retarded,
            0.7,
            descriptor,
        )


def test_lead_hash_is_orientation_sensitive_and_deterministic() -> None:
    lead = _scalar_lead()
    assert harmonic_lead_physics_sha256(lead) == harmonic_lead_physics_sha256(lead)
    changed = LeadBlocks(d00=np.array([[2.0]]), d01=np.array([[-1.0 + 1.0e-12]]))
    assert harmonic_lead_physics_sha256(changed) != harmonic_lead_physics_sha256(lead)
