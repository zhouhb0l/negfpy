"""New compact contacts preserve the existing full-device thermal fields."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from negfpy.anharmonic import transverse_compact_contacts as module
from negfpy.anharmonic import transverse_contacts as original
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_compact_contacts import (
    build_compact_transverse_contact,
)
from negfpy.anharmonic.types import HarmonicReservoir
from negfpy.core.types import LeadBlocks


def _complex_source(kpar, *, explicit=True, indices=(4, 1)):
    phase = np.exp(1j * kpar[0])
    d00 = np.array([[3.0, -0.3 - 0.15 * phase], [-0.3 - 0.15 * phase.conjugate(), 3.5]])
    d01 = np.array([[-0.65, -0.12 * phase], [-0.04, -0.55]])
    coupling = np.array([[-0.7, -0.09 * phase], [0.02 * phase.conjugate(), -0.6]])
    return HarmonicReservoir(
        LeadBlocks(d00, d01), 0.8,
        device_to_lead=coupling if explicit else None, device_indices=indices,
    )


def _embed(values, indices, ndof):
    result = np.zeros((*values.shape[:-2], ndof, ndof), dtype=complex)
    for local_i, full_i in enumerate(indices):
        for local_j, full_j in enumerate(indices):
            result[..., full_i, full_j] = values[..., local_i, local_j]
    return result


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("explicit", [True, False])
def test_compact_matches_full_complex_fields_and_reversed_embedding(side, explicit):
    mesh = UniformBZMesh((4, 1, 1))
    grid = SignedFrequencyGrid(0.25, 4)
    source = lambda k: _complex_source(k, explicit=explicit)
    full = original.build_transverse_contact(mesh, grid, source, 6, 0.02, side=side)
    compact = build_compact_transverse_contact(mesh, grid, source, 6, 0.02, side=side)
    assert compact.full_ndof == 6
    assert compact.device_indices == (4, 1)
    assert compact.mesh is mesh and compact.grid is grid
    assert compact.spectrum.retarded.shape == (4, 9, 2, 2)
    for name in ("retarded", "lesser", "greater"):
        values = getattr(compact.spectrum, name)
        np.testing.assert_array_equal(_embed(values, (4, 1), 6), getattr(full.spectrum, name))
        assert values.flags.owndata and not values.flags.writeable
    np.testing.assert_array_equal(_embed(compact.zero_limits, (4, 1), 6), full.zero_limits)
    assert np.max(np.abs(compact.zero_limits[1].imag)) > 1e-6
    assert not compact.zero_limits.flags.writeable
    with pytest.raises(FrozenInstanceError):
        compact.device_indices = (1, 4)


@pytest.mark.parametrize("side,expected", [("left", (0,)), ("right", (5,))])
def test_default_embedding_acoustic_zero_limit(side, expected):
    grid = SignedFrequencyGrid(1e-6, 1)
    reservoir = HarmonicReservoir(LeadBlocks(np.array([[2.6]]), np.array([[-1.3]])), 0.7)
    eta = 0.0015625
    compact = build_compact_transverse_contact(
        UniformBZMesh((1, 1, 1)), grid, lambda k: reservoir, 6, eta, side=side,
    )
    assert compact.device_indices == expected
    discriminant = np.sqrt((2.6 + eta**2)**2 - 4 * 1.3**2)
    limit = 2 * eta * ((2.6 + eta**2) / discriminant - 1)
    np.testing.assert_allclose(compact.zero_limits, limit, rtol=2e-9, atol=1e-11)
    np.testing.assert_allclose(compact.spectrum.lesser[:, grid.zero_index], -0.7j * limit,
                               rtol=2e-9, atol=1e-11)
    np.testing.assert_array_equal(compact.spectrum.greater[:, grid.zero_index],
                                  compact.spectrum.lesser[:, grid.zero_index])
    assert compact.contact.q_reports[0]["classification"] == "acoustic"


def test_each_q_source_is_detached_before_next_callback(monkeypatch):
    mesh = UniformBZMesh((4, 1, 1))
    shared = LeadBlocks(np.eye(2, dtype=complex), np.eye(2, dtype=complex))
    calls = []

    def source(k):
        current = _complex_source(k)
        shared.d00[:] = current.lead.d00
        shared.d01[:] = current.lead.d01
        calls.append(k)
        return HarmonicReservoir(shared, 0.8, device_to_lead=current.device_to_lead,
                                 device_indices=slice(4, 0, -3))

    original_builder = module.build_transverse_contact

    def checked_builder(mesh, grid, sources, ndof, eta, **kwargs):
        shared.d00[:] = 100
        shared.d01[:] = 50
        for k in calls:
            detached = sources(k)
            expected = _complex_source(k)
            np.testing.assert_array_equal(detached.lead.d00, expected.lead.d00)
            np.testing.assert_array_equal(detached.lead.d01, expected.lead.d01)
            assert not np.shares_memory(detached.lead.d00, shared.d00)
            assert detached.device_indices == (0, 1)
        return original_builder(mesh, grid, sources, ndof, eta, **kwargs)

    monkeypatch.setattr(module, "build_transverse_contact", checked_builder)
    result = build_compact_transverse_contact(
        mesh, SignedFrequencyGrid(0.2, 2), source, 6, 0.02, side="left",
    )
    assert result.device_indices == (4, 1)
    assert len(calls) == 4
    saved = result.spectrum.retarded.copy()
    shared.d00[:] = 500
    np.testing.assert_array_equal(result.spectrum.retarded, saved)


@pytest.mark.parametrize("bad", ["embedding", "order", "temperature", "conjugacy", "type"])
def test_bad_q_sources_rejected_before_spectral_work(monkeypatch, bad):
    monkeypatch.setattr(original, "_classify_lead_spectrum",
                        lambda *a, **k: pytest.fail("invalid input reached spectral work"))

    def source(k):
        if k[0] == 0:
            return _complex_source(k)
        if bad == "type":
            return object()
        raw = _complex_source(k)
        if bad == "conjugacy":
            raw.lead.d00[0, 0] += 0.01 * np.sin(k[0])
        indices = {"embedding": (4, 2), "order": (1, 4)}.get(bad, (4, 1))
        return HarmonicReservoir(raw.lead, 0.9 if bad == "temperature" else 0.8,
                                 device_to_lead=raw.device_to_lead, device_indices=indices)

    with pytest.raises((ValueError, TypeError), match="embedding|temperature|conjugacy|Reservoir"):
        build_compact_transverse_contact(
            UniformBZMesh((4, 1, 1)), SignedFrequencyGrid(0.2, 2), source, 6, 0.02, side="left",
        )


@pytest.mark.parametrize("kwargs,error", [
    ({"max_bytes": 1}, ValueError),
    ({"max_base_evaluations": 1}, ValueError),
    ({"ndof": True}, TypeError),
    ({"ndof": 0}, ValueError),
    ({"eta_lead": np.nan}, ValueError),
    ({"eta_lead": 0}, ValueError),
    ({"side": "middle"}, ValueError),
    ({"q_samples": 10}, ValueError),
    ({"surface_gf_method": None}, TypeError),
])
def test_argument_and_minimum_budget_guards_precede_callback(kwargs, error):
    options = {"ndof": 6, "eta_lead": 0.02, "side": "left", **kwargs}
    with pytest.raises(error):
        build_compact_transverse_contact(
            UniformBZMesh((4, 1, 1)), SignedFrequencyGrid(0.2, 2),
            lambda k: pytest.fail("invalid arguments reached callback"), **options,
        )


@pytest.mark.parametrize("indices", [(6, 1), (4, 4), (-1, 1)])
def test_invalid_full_embedding_rejected(indices, monkeypatch):
    monkeypatch.setattr(module, "build_transverse_contact",
                        lambda *a, **k: pytest.fail("invalid embedding reached inner builder"))
    with pytest.raises(ValueError, match="outside|unique|negative"):
        build_compact_transverse_contact(
            UniformBZMesh((1, 1, 1)), SignedFrequencyGrid(0.2, 2),
            lambda k: _complex_source(k, indices=indices), 6, 0.02, side="left",
        )


@pytest.mark.parametrize("budget,message", [(500, "spectra"), (200, "operator")])
def test_actual_payload_guards_precede_inner_builder(budget, message, monkeypatch):
    monkeypatch.setattr(module, "build_transverse_contact",
                        lambda *a, **k: pytest.fail("payload guard reached inner builder"))
    with pytest.raises(ValueError, match=message):
        build_compact_transverse_contact(
            UniformBZMesh((1, 1, 1)), SignedFrequencyGrid(0.2, 1), _complex_source,
            6, 0.02, side="left", max_bytes=budget,
        )


def test_accumulated_lead_payload_guard_precedes_inner_builder(monkeypatch):
    monkeypatch.setattr(module, "build_transverse_contact",
                        lambda *a, **k: pytest.fail("lead payload reached inner builder"))

    def source(k):
        raw = _complex_source(k)
        return HarmonicReservoir(raw.lead, 0.8, device_to_lead=raw.device_to_lead[:1],
                                 device_indices=(4,))

    with pytest.raises(ValueError, match="Input lead blocks"):
        build_compact_transverse_contact(
            UniformBZMesh((4, 1, 1)), SignedFrequencyGrid(0.2, 1), source,
            6, 0.02, side="left", max_bytes=700,
        )
