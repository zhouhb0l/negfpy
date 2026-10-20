"""Scalar zero limits, legacy boundaries and paired-q input guards."""

from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic import transverse_contacts as module
from negfpy.anharmonic.acoustic_contacts import (
    _build_acoustic_contact_descriptor,
    _build_fixed_eta_contact_descriptor,
    _classify_lead_spectrum,
)
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_contacts import build_transverse_contact
from negfpy.anharmonic.transverse_mapping import (
    build_transverse_problem,
    gapped_contact_spectrum,
)
from negfpy.anharmonic.transverse_q import TransverseCubicVertex
from negfpy.anharmonic.transverse_solver import (
    TransverseSCBAConfig,
    solve_transverse_scba,
)
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
)
from negfpy.core.types import Device1D, DeviceKSpace, LeadBlocks


def _reservoir(*, pinning=0.0, temperature=0.7):
    return HarmonicReservoir(
        LeadBlocks(np.array([[2.6 + pinning]]), np.array([[-1.3]])), temperature,
    )


@pytest.mark.parametrize("pinning", [0.0, 0.4])
@pytest.mark.parametrize("side", ["left", "right"])
def test_exact_scalar_fixed_eta_zero_limit_and_near_zero_continuity(pinning, side):
    # Fixed user-chosen eta is tested without an eta-refinement campaign.
    eta = 0.0015625
    grid = SignedFrequencyGrid(1e-6, 2)
    reservoir = _reservoir(pinning=pinning)
    contact = build_transverse_contact(
        UniformBZMesh((1, 1, 1)), grid, lambda k: reservoir, 1, eta, side=side,
    )
    onsite = 2.6 + pinning
    discriminant = np.sqrt((onsite + eta**2)**2 - 4 * 1.3**2)
    expected = 2 * eta * ((onsite + eta**2) / discriminant - 1)
    np.testing.assert_allclose(contact.zero_limits, expected, rtol=2e-9, atol=1e-11)
    zero = grid.zero_index
    np.testing.assert_allclose(
        contact.spectrum.lesser[0, zero], -1j * reservoir.temperature * expected,
        rtol=2e-9, atol=1e-11,
    )
    np.testing.assert_array_equal(
        contact.spectrum.lesser[:, zero], contact.spectrum.greater[:, zero],
    )
    positive = contact.spectrum.retarded[0, zero + 1]
    gamma = 1j * (positive - positive.conj().T)
    np.testing.assert_allclose(gamma / grid.step, contact.zero_limits[0],
                               rtol=2e-6, atol=1e-10)
    assert contact.q_reports[0]["classification"] == ("acoustic" if pinning == 0 else "gapped")


def test_gapped_legacy_zero_convention_is_preserved_separately():
    reservoir = _reservoir(pinning=0.4)
    grid = SignedFrequencyGrid(0.2, 3)
    old = gapped_contact_spectrum(grid, reservoir, 1, 0.02, side="left")
    new = build_transverse_contact(
        UniformBZMesh((1, 1, 1)), grid, lambda k: reservoir, 1, 0.02, side="left",
    )
    np.testing.assert_array_equal(old.lesser[grid.zero_index], 0)
    assert np.max(np.abs(new.spectrum.lesser[:, grid.zero_index])) > 0.01
    nonzero = np.arange(grid.n_points) != grid.zero_index
    for name in ("retarded", "lesser", "greater"):
        np.testing.assert_allclose(
            getattr(new.spectrum, name)[0, nonzero], getattr(old, name)[nonzero],
            rtol=1e-12, atol=1e-12,
        )
    with pytest.raises(ValueError, match="classified as acoustic"):
        _build_acoustic_contact_descriptor(
            reservoir, side="left", ndof=1, eta_lead=0.02, surface_gf_method="sancho_rubio",
        )


def test_single_q_acoustic_path_matches_existing_contact_builder():
    reservoir = _reservoir()
    grid = SignedFrequencyGrid(0.2, 6)
    problem = CubicSCBAProblem(
        Device1D([np.array([[2.6]])], []), reservoir, reservoir,
        DenseCubicVertex(np.zeros((1, 1, 1))), grid,
    )
    old_left, old_right = build_contact_spectra(problem, eta_lead=0.02)
    for side, old in (("left", old_left), ("right", old_right)):
        new = build_transverse_contact(
            UniformBZMesh((1, 1, 1)), grid, lambda k: reservoir, 1, 0.02, side=side,
        )
        for name in ("retarded", "lesser", "greater"):
            np.testing.assert_array_equal(getattr(new.spectrum, name)[0], getattr(old, name))


def test_zero_temperature_and_cached_callback_feed_harmonic_solver(monkeypatch):
    mesh = UniformBZMesh((1, 1, 1))
    grid = SignedFrequencyGrid(0.3, 7)
    reservoir = _reservoir(temperature=0)
    left = build_transverse_contact(mesh, grid, lambda k: reservoir, 1, 0.02, side="left")
    right = build_transverse_contact(mesh, grid, lambda k: reservoir, 1, 0.02, side="right")
    for contact in (left, right):
        np.testing.assert_array_equal(contact.spectrum.lesser[:, grid.zero_index], 0)
        np.testing.assert_array_equal(contact.spectrum.greater[:, grid.zero_index], 0)
        assert not contact.zero_limits.flags.writeable
        with pytest.raises(TypeError):
            contact.q_reports[0]["classification"] = "changed"
    monkeypatch.setattr(module, "retarded_contact_self_energy",
                        lambda *a, **k: pytest.fail("cached contact was recomputed"))
    vertex = TransverseCubicVertex(mesh, np.zeros((1, 1, 1, 1, 1)))
    problem = build_transverse_problem(
        grid=grid, vertex=vertex, device=DeviceKSpace(lambda k: Device1D([np.array([[2.6]])], [])),
        left=left.at_q, right=right.at_q, input_units="solver",
    )
    result = solve_transverse_scba(problem, TransverseSCBAConfig(max_iterations=3))
    assert result.converged
    assert result.currents.left == result.currents.right == 0
    np.testing.assert_array_equal(left.at_q((2 * np.pi, 0)).retarded, left.at_q((0, 0)).retarded)
    with pytest.raises(ValueError, match="not a resolved point"):
        left.at_q((0.3, 0))
    saved = left.spectrum.retarded.copy()
    reservoir.lead.d00[0, 0] += 10
    np.testing.assert_array_equal(left.spectrum.retarded, saved)


@pytest.mark.parametrize("bad", ["temperature", "embedding", "coupling", "lead"])
def test_inconsistent_q_sources_rejected_before_spectral_work(monkeypatch, bad):
    def source(k):
        value = _reservoir()
        if k[0] == 0:
            return HarmonicReservoir(value.lead, 0.7, device_indices=(0,))
        if bad == "temperature":
            return HarmonicReservoir(value.lead, 0.8, device_indices=(0,))
        if bad == "embedding":
            return HarmonicReservoir(value.lead, 0.7, device_indices=(1,))
        if bad == "coupling":
            return HarmonicReservoir(value.lead, 0.7,
                                     device_to_lead=np.array([[-1.3 + 0.1j]]), device_indices=(0,))
        return HarmonicReservoir(
            LeadBlocks(np.array([[2.6 + 0.1 * np.sin(k[0])]]), np.array([[-1.3]])),
            0.7, device_indices=(0,),
        )

    monkeypatch.setattr(module, "_classify_lead_spectrum",
                        lambda *a, **k: pytest.fail("invalid source reached spectral work"))
    with pytest.raises(ValueError, match="temperature|embedding|conjugacy"):
        build_transverse_contact(
            UniformBZMesh((4, 1, 1)), SignedFrequencyGrid(0.1, 2), source, 2, 0.02, side="left",
        )


@pytest.mark.parametrize("lead", [
    LeadBlocks(np.array([[1.5]]), np.array([[-1.0]])),
    LeadBlocks(np.diag([2.0, 0]), np.diag([-1.0, 0])),
])
def test_unstable_and_unresolved_soft_leads_rejected(lead):
    with pytest.raises(ValueError, match="unstable|invalid-soft"):
        build_transverse_contact(
            UniformBZMesh((1, 1, 1)), SignedFrequencyGrid(0.1, 2),
            lambda k: HarmonicReservoir(lead, 0.7), lead.d00.shape[0], 0.02, side="left",
        )


@pytest.mark.parametrize("kwargs", [{"max_bytes": 1}, {"max_base_evaluations": 1}])
def test_budgets_reject_before_callback(kwargs):
    with pytest.raises(ValueError, match="exceed"):
        build_transverse_contact(
            UniformBZMesh((4, 1, 1)), SignedFrequencyGrid(0.1, 2),
            lambda k: pytest.fail("budget exceeded before callback"), 1, 0.02, side="left", **kwargs,
        )


def test_cached_classification_must_match_the_actual_lead():
    wrong = _classify_lead_spectrum(_reservoir(pinning=0.2).lead)
    with pytest.raises(ValueError, match="different lead"):
        _build_fixed_eta_contact_descriptor(
            _reservoir(pinning=0.3), side="left", ndof=1, eta_lead=0.02,
            surface_gf_method="sancho_rubio", spectral_classification=wrong,
        )
