"""Tiny literal contractions and boundary counterexample; no material replay."""

import json
from itertools import product

import numpy as np
import pytest

import negfpy.anharmonic.transverse_acoustic_sector as diagnostic
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_solver import TransverseSpectrum

_TERMS = ("CC", "DD_over_4", "CD", "DC")


def _fixture(nq=3):
    n = 6
    mesh, grid = UniformBZMesh((nq, 1, 1)), SignedFrequencyGrid(.17, 2)
    factors = np.empty((nq, n), complex)
    for q, a in product(range(nq), range(n)):
        factors[q, a] = (a+1)/13 + np.exp(2j*np.pi*q/nq)*(a+2)/23
    coefficients = np.empty((nq, nq, n, n, n), complex)
    for q, p in product(range(nq), repeat=2):
        coefficients[q, p] = np.einsum("a,c,d->acd", factors[-q % nq], factors[p], factors[(q-p) % nq])
    vertex = PackedTransverseCubicVertex(mesh, coefficients, np.arange(n), n)
    dynamical = np.tile(np.diag(np.arange(1.3, 1.9, .1)[:n]), (nq, 1, 1)).astype(complex)
    contacts = []
    for indices, side, temp, damping in (((0, 1), "left", .91, .31), ((4, 5), "right", .57, .23)):
        weight = np.broadcast_to(damping*np.array([[1.2, .19], [.19, .8]]), (nq, 2, 2))
        r = -1j*grid.values[None, :, None, None]*weight[:, None]
        c = np.broadcast_to(-2j*temp*weight[:, None], r.shape)
        d = r-r.conj().swapaxes(-1, -2)
        contact = TransverseContact(mesh, grid, TransverseSpectrum(r, c-d/2, c+d/2),
            2*weight, tuple({} for _ in range(nq)), {}, side, .0015625, temp)
        contacts.append(CompactTransverseContact(contact, n, indices))
    shape = (nq, grid.n_points, n, n)
    r, c, d = (np.zeros(shape, complex) for _ in range(3))
    weight = np.diag(np.linspace(.7, 1.2, n)).astype(complex)
    weight[0, 5] = weight[5, 0] = .13
    for q, w in product(range(nq), range(grid.n_points)):
        r[q, w] = -.003j*grid.values[w]*weight
        c[q, w] = -.02j*weight
        d[q, w] = r[q, w]-r[q, w].conj().T
        d[q, w, 0, 5] += .0002
        d[q, w, 5, 0] -= .0002
    translations = np.vstack((np.eye(3), np.eye(3)))/np.sqrt(2)
    return grid, vertex, dynamical, *contacts, ChannelSpectrum(r, c, d), translations, np.arange(3)


def _run(args, **kwargs):
    return diagnostic.study_acoustic_sector_diagnostic(
        *args, workers=1, external_tile=2, **kwargs)


def _literal(args):
    grid, vertex, dynamical, left, right, sigma, translations, _ = args
    nq, n, z = vertex.mesh.n_qpoints, vertex.ndof, grid.zero_index
    greens = {}
    for q, w in product(range(nq), range(grid.n_points)):
        operator = grid.values[w]**2*np.eye(n)-dynamical[q]-sigma.retarded[q, w]
        cs, ds = sigma.statistical[q, w].copy(), sigma.spectral[q, w].copy()
        for contact in (left, right):
            for i, j in product(range(2), repeat=2):
                target = contact.device_indices[i], contact.device_indices[j]
                operator[target] -= contact.spectrum.retarded[q, w, i, j]
                cs[target] += (contact.spectrum.greater[q, w, i, j]+contact.spectrum.lesser[q, w, i, j])/2
                ds[target] += contact.spectrum.greater[q, w, i, j]-contact.spectrum.lesser[q, w, i, j]
        inverse = np.linalg.inv(operator)
        greens[q, w] = inverse@cs@inverse.conj().T, inverse@ds@inverse.conj().T
    values = {}
    factor = .5j*grid.integration_weight/nq*.03**2
    for p, w in product(range(nq), range(grid.n_points)):
        c1, d1 = greens[p, w]
        c2, d2 = greens[-p % nq, 2*z-w]
        v = vertex.coefficients[0, p]
        values[p, w] = [np.einsum("acd,ce,df,bef->ab", v, x, y, v.conj(), optimize=False)*factor*weight
            for x, y, weight in ((c1, c2, 1), (d1, d2, .25), (c1, d2, 1), (d1, c2, 1))]
    p = translations@translations.T
    q = np.eye(n)-p
    c = greens[0, z][0]
    sectors = (p@c@p, p@c@q, q@c@p, q@c@q)
    v = vertex.coefficients[0, 0]
    pairs = [np.einsum("acd,ce,df,bef->ab", v, x, y, v.conj(), optimize=False)*factor
             for x in sectors for y in sectors]
    return values, pairs


def test_signed_q_bin_born_and_all_sector_pairs_match_literal_contractions():
    args = _fixture()
    result = _run(args)
    expected, pairs = _literal(args)
    assert result["dyson_evaluations"] == 15
    for qrow in result["born"]["per_internal_q"]:
        p = qrow["internal_q_index"]
        assert [row["frequency_offset"] for row in qrow["signed_bins"]] == [-2, -1, 0, 1, 2]
        assert qrow["partner_q_index"] == -p % 3
        for row in qrow["signed_bins"]:
            w = row["frequency_index"]
            assert row["partner_frequency_index"] == 4-w
            for t, term in enumerate(_TERMS):
                actual = row["terms"][term]["full"]
                assert actual["frobenius"] == pytest.approx(np.linalg.norm(expected[p, w][t]), rel=3e-13, abs=1e-24)
                assert actual["max_abs"] == pytest.approx(np.max(abs(expected[p, w][t])), rel=3e-13, abs=1e-24)
    sector = result["gamma_zero_CC_sectors"]
    assert sector["statistical_reconstruction"]["passed"]
    assert sector["CC_reconstruction"]["passed"]
    assert sector["interference_discarded"] is False
    assert len(sector["ordered_CC_pairs"]) == 16
    for row, expected_pair in zip(sector["ordered_CC_pairs"], pairs, strict=True):
        assert row["blocks"]["full"]["frobenius"] == pytest.approx(np.linalg.norm(expected_pair), rel=3e-13, abs=1e-24)
        assert row["algebraic_component_not_individual_physical_channel"]
    assert np.linalg.norm(pairs[1]) > 0  # interference really exercised
    json.dumps(result, allow_nan=False)


def test_symmetric_any_leg_mask_breaks_outer_pair_translation_cancellation():
    mesh = UniformBZMesh((1, 1, 1))
    translation = np.vstack((np.eye(3), np.eye(3)))/np.sqrt(2)
    bond = np.array([1., 0., 0., -1., 0., 0.])
    full = np.einsum("a,c,d->acd", bond, bond, bond)
    cropped = full.copy()
    cropped[3, 3, 3] = 0.
    exact = PackedTransverseCubicVertex(mesh, full[None, None].astype(complex), np.arange(6), 6)
    cut = PackedTransverseCubicVertex(mesh, cropped[None, None].astype(complex), np.arange(6), 6)
    first = diagnostic.audit_vertex_translations(exact, translation, np.arange(3))
    second = diagnostic.audit_vertex_translations(cut, translation, np.arange(3))
    assert all(row["blocks"]["full"]["frobenius"] == 0 for row in first["rows"])
    for row in second["rows"]:
        if row["direction"] == 0:
            assert row["blocks"]["outer_outer"]["frobenius"] == pytest.approx(1/np.sqrt(2))
            assert row["blocks"]["central_central"]["frobenius"] == 0
            assert row["blocks"]["central_outer"]["frobenius"] == 0
    assert second["residual_is_not_an_execution_gate"]


def test_native_spectral_signal_is_not_reconstructed_from_statistical_exports():
    args = _fixture(nq=1)
    first = _run(args)
    sigma = args[5]
    changed = ChannelSpectrum(sigma.retarded, sigma.statistical, np.zeros_like(sigma.spectral))
    second = _run((*args[:5], changed, *args[6:]))
    a, b = first["born"]["term_totals"], second["born"]["term_totals"]
    assert a["CC"]["full"]["frobenius"] == pytest.approx(b["CC"]["full"]["frobenius"])
    assert a["DD_over_4"]["full"]["frobenius"] != b["DD_over_4"]["full"]["frobenius"]
    assert first["native_spectral_channel_retained"]


def test_numeric_and_time_admission_happen_before_any_inverse(monkeypatch):
    args = _fixture(nq=1)
    monkeypatch.setattr(np.linalg, "inv", lambda *_: pytest.fail("No inverse admitted"))
    for kwargs in ({"max_contraction_work": 1}, {"max_bytes": 1}, {"deadline_monotonic": 0}):
        with pytest.raises((ValueError, TimeoutError)):
            _run(args, **kwargs)


def test_translation_basis_must_be_normalized_before_any_inverse(monkeypatch):
    args = _fixture(nq=1)
    monkeypatch.setattr(np.linalg, "inv", lambda *_: pytest.fail("No inverse admitted"))
    with pytest.raises(ValueError, match="bookkeeping"):
        _run((*args[:6], 2*args[6], args[7]))


def test_production_plan_counts_raw_and_all16_sector_pairs():
    plan = diagnostic.plan_acoustic_sector_dimensions(4, 601, 216)
    assert plan["dyson_evaluations"] == 20
    assert plan["raw_contraction_work"] == 12*4*5*216**4
    assert plan["sector_contraction_work"] == 16*3*216**4
    assert plan["contraction_work"] == 626_913_312_768
    assert plan["peak_owned_bytes"] < 1024**3
