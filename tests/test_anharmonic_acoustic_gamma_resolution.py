"""Fresh controls for the finite-precision Gamma-basin location decision.

Only small synthetic nearest-neighbour leads are used.  The targeted optimizer
replacement retains the actual eigenvalue objective at the reported displaced
location, modelling the indeterminate argmin of a rounded acoustic basin.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import negfpy.anharmonic.acoustic_contacts as acoustic
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.layered import LayeredHarmonicModel
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_contacts import build_transverse_contact
from negfpy.anharmonic.types import HarmonicReservoir, SolverScale
from negfpy.core.types import LeadBlocks

_OPTIMIZER_ROUNDOFF_Q = 1.344539962e-7


def _diagonal_lead(
    springs: tuple[float, ...] = (1.0,),
    *,
    phases: tuple[float, ...] | None = None,
    pinning: tuple[float, ...] | None = None,
    scale: float = 1.0,
) -> LeadBlocks:
    spring = np.asarray(springs, dtype=np.float64)
    phase = np.zeros_like(spring) if phases is None else np.asarray(phases)
    gap = np.zeros_like(spring) if pinning is None else np.asarray(pinning)
    d00 = np.diag(scale * (2.0 * spring + gap)).astype(np.complex128)
    d01 = np.diag(-scale * spring * np.exp(-1j * phase))
    return LeadBlocks(d00=d00, d01=d01, d10=d01.conj().T)


def _force_gamma_optimizer_location(
    monkeypatch: pytest.MonkeyPatch,
    qpoint: float,
) -> list[tuple[float, float]]:
    original = acoustic.minimize_scalar
    evaluations: list[tuple[float, float]] = []

    def displaced_argmin(objective, *, bounds, method, options):
        if bounds[0] <= 0.0 <= bounds[1] and bounds[0] < qpoint < bounds[1]:
            value = float(objective(qpoint))
            evaluations.append((qpoint, value))
            return SimpleNamespace(success=True, x=qpoint, fun=value)
        return original(objective, bounds=bounds, method=method, options=options)

    monkeypatch.setattr(acoustic, "minimize_scalar", displaced_argmin)
    return evaluations


@pytest.mark.parametrize("qsign", [-1.0, 1.0])
@pytest.mark.parametrize("scale", [1.0e-9, 1.0, 1.0e9])
def test_gamma_optimizer_roundoff_is_accepted_with_scale_covariance(
    monkeypatch: pytest.MonkeyPatch, qsign: float, scale: float
) -> None:
    qpoint = qsign * _OPTIMIZER_ROUNDOFF_Q
    calls = _force_gamma_optimizer_location(monkeypatch, qpoint)
    lead = _diagonal_lead(scale=scale)
    before = tuple(value.copy() for value in (lead.d00, lead.d01, lead.d10))

    report = acoustic._classify_lead_spectrum(lead)

    assert calls
    assert all(q == qpoint for q, _ in calls)
    assert all(abs(value) <= report.eigenvalue_tolerance for _, value in calls)
    assert report.classification == "acoustic"
    assert report.passed
    assert report.gamma_nullity == 1
    assert report.curvature_consistent
    assert report.gamma_basin_resolved
    assert report.gamma_energy_resolution == 4.0 * report.eigenvalue_tolerance
    assert abs(qpoint) < report.gamma_q_resolution < (2.0 * np.pi / 1024.0) / 4.0
    assert report.gamma_slope_norm <= report.gamma_slope_tolerance
    assert report.gamma_basin_curvatures.shape == (4, 1)
    np.testing.assert_allclose(report.gamma_basin_curvatures / scale, 1.0, rtol=1.0e-5)
    assert report.soft_minima_gamma_compatible.shape == report.soft_minima_q.shape
    assert np.all(report.soft_minima_gamma_compatible)
    assert np.any(np.abs(report.soft_minima_q) > 1.0e-7)
    assert report.failure_reasons == ()
    for current, original in zip((lead.d00, lead.d01, lead.d10), before, strict=True):
        np.testing.assert_array_equal(current, original)


def test_degenerate_acoustic_modes_and_gapped_spectator_remain_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_gamma_optimizer_location(monkeypatch, -_OPTIMIZER_ROUNDOFF_Q)
    lead = _diagonal_lead((0.7, 0.7, 1.2, 0.8), pinning=(0.0, 0.0, 0.0, 0.4))
    # Exact fixed orthogonal change of basis gives non-diagonal real blocks.
    rotation = np.array(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
        dtype=np.float64,
    ) / 2.0
    rotated = LeadBlocks(
        d00=rotation @ lead.d00 @ rotation.T,
        d01=rotation @ lead.d01 @ rotation.T,
        d10=rotation @ lead.d10 @ rotation.T,
    )

    report = acoustic._classify_lead_spectrum(rotated)

    assert report.classification == "acoustic"
    assert report.gamma_nullity == 3
    assert report.curvature_consistent
    assert report.normalized_asr_residual < 1.0e-14


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_one_ulp_assembly_perturbation_preserves_gamma_classification(
    monkeypatch: pytest.MonkeyPatch, direction: float
) -> None:
    _force_gamma_optimizer_location(monkeypatch, _OPTIMIZER_ROUNDOFF_Q)
    lead = _diagonal_lead((0.8, 1.1, 1.7))
    d00 = lead.d00.copy()
    d00[0, 0] = np.nextafter(d00[0, 0].real, direction * np.inf)
    perturbed = LeadBlocks(d00=d00, d01=lead.d01, d10=lead.d10)

    report = acoustic._classify_lead_spectrum(perturbed)

    assert report.classification == "acoustic"
    assert report.gamma_nullity == 3
    assert report.curvature_consistent
    assert abs(report.gamma_eigenvalues[0]) <= report.eigenvalue_tolerance


@pytest.mark.parametrize("qpoint", [2.0e-5, 0.15, np.pi])
def test_resolved_displaced_soft_minimum_remains_rejected(qpoint: float) -> None:
    report = acoustic._classify_lead_spectrum(_diagonal_lead(phases=(qpoint,)))

    assert report.classification == "invalid-soft"
    assert not report.passed
    assert report.gamma_nullity == 0
    assert report.refined_minimum_frequency_squared >= -report.stability_tolerance
    assert any("Gamma" in reason for reason in report.failure_reasons)


def test_displaced_soft_optical_branch_cannot_hide_behind_gamma_nullspace() -> None:
    report = acoustic._classify_lead_spectrum(
        _diagonal_lead((1.0, 0.8), phases=(0.0, np.pi))
    )

    assert report.classification == "invalid-soft"
    assert not report.passed
    assert report.gamma_nullity == 1
    assert report.curvature_consistent
    assert np.max(np.abs(report.soft_minima_q)) > 3.0
    assert any("minimum" in reason and "Gamma" in reason for reason in report.failure_reasons)


def test_near_gamma_displaced_soft_branch_is_resolved_despite_acoustic_spectator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qpoint = 2.0e-5
    calls = _force_gamma_optimizer_location(monkeypatch, qpoint)
    report = acoustic._classify_lead_spectrum(
        _diagonal_lead((1.0, 1.0), phases=(0.0, qpoint))
    )

    assert calls
    assert report.classification == "invalid-soft"
    assert not report.passed
    assert report.gamma_nullity == 1
    assert report.curvature_consistent
    assert any("minimum" in reason and "Gamma" in reason for reason in report.failure_reasons)


def test_resolved_negative_eigenvalue_remains_unstable() -> None:
    report = acoustic._classify_lead_spectrum(_diagonal_lead(pinning=(-1.0e-6,)))

    assert report.classification == "unstable"
    assert not report.passed
    assert report.refined_minimum_frequency_squared < -report.stability_tolerance
    assert "a refined lead eigenvalue is below the stability floor" in report.failure_reasons


@pytest.mark.parametrize("springs", [(0.0,), (1.0, 0.0)])
def test_flat_soft_branch_remains_rejected(springs: tuple[float, ...]) -> None:
    report = acoustic._classify_lead_spectrum(_diagonal_lead(springs))

    assert report.classification == "invalid-soft"
    assert not report.passed
    assert report.gamma_nullity == len(springs)
    assert not report.curvature_consistent
    assert "acoustic curvature is nonpositive or unresolved" in report.failure_reasons


def test_gapped_optical_lead_keeps_gapped_classification() -> None:
    report = acoustic._classify_lead_spectrum(
        _diagonal_lead((0.7, 1.3), pinning=(0.2, 0.4))
    )

    assert report.classification == "gapped"
    assert report.passed
    assert report.gamma_nullity == 0
    assert report.soft_minima_q.size == 0
    assert report.failure_reasons == ()


def test_unresolved_gamma_energy_does_not_hide_resolved_phase_slope() -> None:
    qpoint = 5.0e-7
    report = acoustic._classify_lead_spectrum(_diagonal_lead(phases=(qpoint,)))

    assert report.gamma_nullity == 1
    assert abs(report.gamma_eigenvalues[0]) < report.eigenvalue_tolerance
    assert report.gamma_slope_norm > report.gamma_slope_tolerance
    assert report.classification == "invalid-soft"
    assert not report.passed
    assert not report.gamma_basin_resolved


def test_quartic_soft_branch_fails_actual_basin_curvature() -> None:
    # det D(q) = (1-cos(q))**2, hence lambda_low ~ q**4/4 even though
    # projection onto the Gamma null vector incorrectly suggests kappa=1.
    d00 = np.diag([2.0, 1.0])
    d01 = np.array([[-1.0, 0.5], [-0.5, 0.0]])
    report = acoustic._classify_lead_spectrum(
        LeadBlocks(d00=d00, d01=d01, d10=d01.T)
    )

    assert report.gamma_nullity == 1
    assert report.curvature_consistent
    assert report.curvature_eigenvalues_h[0] > 0.9
    assert not report.gamma_basin_resolved
    assert report.classification == "invalid-soft"
    assert not report.passed


def test_exact_flat_band_fails_despite_positive_projected_curvature() -> None:
    # This rank-one Bloch matrix has eigenvalues [0,2] at every q.
    d00 = np.eye(2)
    d01 = np.array([[0.0, -1.0], [0.0, 0.0]])
    report = acoustic._classify_lead_spectrum(
        LeadBlocks(d00=d00, d01=d01, d10=d01.T)
    )

    assert report.gamma_nullity == 1
    assert report.curvature_consistent
    assert report.curvature_eigenvalues_h[0] > 0.49
    assert not report.gamma_basin_resolved
    assert report.classification == "invalid-soft"
    assert not report.passed


def test_relaxed_diatomic_curvature_sets_resolution_without_false_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_gamma_optimizer_location(monkeypatch, _OPTIMIZER_ROUNDOFF_Q)
    a, b = 0.03, 1.0
    d00 = np.array([[a + b, -a], [-a, a + b]])
    d01 = np.array([[0.0, 0.0], [-b, 0.0]])
    report = acoustic._classify_lead_spectrum(
        LeadBlocks(d00=d00, d01=d01, d10=d01.T)
    )

    expected = a * b / (2.0 * (a + b))
    assert report.classification == "acoustic"
    assert report.passed
    assert report.gamma_basin_resolved
    np.testing.assert_allclose(report.gamma_basin_curvatures, expected, rtol=1.0e-5)
    np.testing.assert_allclose(report.curvature_eigenvalues_h, b / 2.0, rtol=1.0e-5)
    assert np.all(report.soft_minima_gamma_compatible)


def test_shifted_grid_refinement_retains_negative_basin_between_centered_samples() -> None:
    # Gamma and all centered samples are positive.  The shifted-grid optimizer
    # finds a negative minimum whose evidence must survive the final decision.
    report = acoustic._classify_lead_spectrum(
        _diagonal_lead(phases=(0.01,), pinning=(-1.0e-6,))
    )

    assert report.sampled_minimum_frequency_squared > 0.0
    assert report.gamma_eigenvalues[0] > 0.0
    assert report.refined_minimum_frequency_squared < -report.stability_tolerance
    assert report.classification == "unstable"
    assert not report.passed


def test_public_material_report_forwards_and_hashes_gamma_basin_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _force_gamma_optimizer_location(monkeypatch, -_OPTIMIZER_ROUNDOFF_Q)
    lead = _diagonal_lead((0.9, 1.2, 1.8))
    model = object.__new__(LayeredHarmonicModel)
    for name, value in {
        "d00": lead.d00,
        "d01": lead.d01,
        "d10": lead.d10,
        "lead": lead,
        "rigid_translation_vectors": np.eye(3),
        "scale": SolverScale(),
        "fc2_content_sha256": "3" * 64,
        "layered_geometry_sha256": "4" * 64,
    }.items():
        object.__setattr__(model, name, value)
    digest = acoustic.domain_digest
    captured = []

    def capture_evidence(domain, payload):
        if domain.startswith("negfpy-harmonic-lead-classification-"):
            captured.append((domain, payload))
        return digest(domain, payload)

    monkeypatch.setattr(acoustic, "domain_digest", capture_evidence)
    report = acoustic.classify_harmonic_lead(model)

    assert report.classification == "acoustic"
    assert report.passed
    assert report.gamma_nullity == 3
    assert report.gamma_basin_resolved
    assert report.gamma_q_resolution > _OPTIMIZER_ROUNDOFF_Q
    assert report.gamma_slope_norm <= report.gamma_slope_tolerance
    assert report.gamma_basin_curvatures.shape == (4, 3)
    assert np.all(report.soft_minima_gamma_compatible)
    assert len(captured) == 1
    domain, payload = captured[0]
    assert domain == "negfpy-harmonic-lead-classification-v2"
    assert payload["classification_policy"] == "gamma-basin-resolution-v1"
    for name in (
        "gamma_basin_resolved", "gamma_q_resolution", "gamma_energy_resolution",
        "gamma_slope_norm", "gamma_slope_tolerance", "gamma_basin_curvatures",
        "soft_minima_gamma_compatible",
    ):
        np.testing.assert_array_equal(payload[name], getattr(report, name))
    assert report.classification_sha256 == digest(domain, payload)
    altered = dict(payload)
    altered["gamma_q_resolution"] = float(np.nextafter(report.gamma_q_resolution, np.inf))
    assert digest(domain, altered) != report.classification_sha256
    with pytest.raises(ValueError):
        report.gamma_basin_curvatures[0, 0] = 0.0
    with pytest.raises(ValueError):
        report.soft_minima_gamma_compatible[0] = False


def test_paired_q_contacts_accept_rounded_gamma_at_chosen_fixed_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _force_gamma_optimizer_location(monkeypatch, _OPTIMIZER_ROUNDOFF_Q)
    spring, temperature, eta = 0.9, 0.73, 0.0015625
    mesh = UniformBZMesh((4, 1, 1))
    grid = SignedFrequencyGrid(0.025, 1)

    def reservoir(kpoint):
        pinning = 0.2 * (1.0 - np.cos(kpoint[0]))
        return HarmonicReservoir(
            _diagonal_lead((spring,), pinning=(pinning,)), temperature
        )

    contact = build_transverse_contact(
        mesh, grid, reservoir, 1, eta, side="left", surface_gf_method="sancho_rubio"
    )

    assert calls
    assert [report["classification"] for report in contact.q_reports] == [
        "acoustic", "gapped", "gapped", "gapped"
    ]
    onsite = 2.0 * spring + 0.2 * (1.0 - np.cos(np.arange(4) * np.pi / 2.0))
    discriminant = np.sqrt((onsite + eta**2)**2 - 4.0 * spring**2)
    expected = 2.0 * eta * ((onsite + eta**2) / discriminant - 1.0)
    np.testing.assert_allclose(contact.zero_limits[:, 0, 0], expected, rtol=2.0e-9)
    np.testing.assert_allclose(
        contact.spectrum.lesser[:, grid.zero_index, 0, 0],
        -1j * temperature * expected,
        rtol=2.0e-9,
    )
    np.testing.assert_array_equal(
        contact.spectrum.lesser[:, grid.zero_index],
        contact.spectrum.greater[:, grid.zero_index],
    )
    np.testing.assert_allclose(
        contact.spectrum.greater - contact.spectrum.lesser,
        contact.spectrum.retarded - contact.spectrum.retarded.conj().swapaxes(-1, -2),
        rtol=1.0e-11, atol=1.0e-12,
    )
