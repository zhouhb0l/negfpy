"""Tiny all-mode SVD/source checks; no material fields or IFC replay."""

import json

import numpy as np
import pytest

import negfpy.anharmonic.transverse_soft_source as diagnostic
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_solver import TransverseSpectrum


def _complex(value):
    return value["real"]+1j*value["imag"]


def _fixture():
    mesh, grid, n = UniformBZMesh((1, 1, 1)), SignedFrequencyGrid(.17, 1), 6
    k = np.diag(np.linspace(1.1, 2.1, n))
    k[0, 4] = k[4, 0] = .23
    k[1, 3] = k[3, 1] = -.17
    dynamical = k[None].astype(complex)
    contacts = []
    for side, indices, scale in (("left", (0, 1, 2), .2), ("right", (3, 4, 5), .3)):
        weight = np.diag([1., 2., 3.])*scale
        r = np.broadcast_to(-.04j*weight, (1, 3, 3, 3)).copy()
        c = np.broadcast_to(-1j*weight, r.shape).copy()
        d = r-r.conj().swapaxes(-1, -2)
        contact = TransverseContact(mesh, grid, TransverseSpectrum(r, c-d/2, c+d/2),
            np.broadcast_to(weight, (1, 3, 3)), ({},), {}, side, .0015625, .8)
        contacts.append(CompactTransverseContact(contact, n, indices))
    shape = (1, 3, n, n)
    r = np.broadcast_to(-.015j*np.diag([1., 2., 3., 4., 5., 6.]), shape).copy()
    h = np.diag([.3, .2, -.01, .4, .3, .5]).astype(complex)
    h[0, 4] = .11+.03j
    h[4, 0] = h[0, 4].conjugate()
    c = np.broadcast_to(-1j*h, shape).copy()
    d = r-r.conj().swapaxes(-1, -2)
    translations = np.vstack((np.eye(3), 2*np.eye(3)))/np.sqrt(5)
    return grid, mesh, dynamical, *contacts, ChannelSpectrum(r, c, d), translations


def _run(args, **kwargs):
    return diagnostic.study_soft_source_diagnostic(*args, n_layers=3, n_soft=3, **kwargs)


def _literal_point(args, w, *, zero=False):
    grid, _, dynamical, left, right, sigma, _ = args
    n = len(dynamical[0])
    operator = grid.values[w]**2*np.eye(n)-dynamical[0]
    sources = {}
    for side, contact in (("left", left), ("right", right)):
        matrix = np.zeros((n, n), complex)
        for i in range(3):
            for j in range(3):
                target = contact.device_indices[i], contact.device_indices[j]
                operator[target] -= contact.spectrum.retarded[0, w, i, j]
                matrix[target] = (contact.spectrum.greater[0, w, i, j]+contact.spectrum.lesser[0, w, i, j])/2
        sources[side] = matrix
    if zero:
        sources["interaction"] = np.zeros((n, n), complex)
    else:
        operator -= sigma.retarded[0, w]
        sources["interaction"] = sigma.statistical[0, w]
    inverse = np.linalg.inv(operator)
    statistical = inverse@sum(sources.values())@inverse.conj().T
    return operator, sources, inverse, statistical


def test_non_normal_operator_all_mode_source_product_and_interference_match_literal():
    args = _fixture()
    result = _run(args)
    assert result["dyson_evaluations"] == result["svd_evaluations"] == 6
    assert [p["frequency_offset"] for p in result["roles"][0]["points"]] == [0, -1, 1]
    for role in result["roles"]:
        for point in role["points"]:
            a, sources, _, c = _literal_point(args, point["frequency_index"], zero=role["role"] == "harmonic_zero_sigma")
            assert np.linalg.norm(a@a.conj().T-a.conj().T@a) > 0
            u, s, vh = np.linalg.svd(a)
            b = u.conj().T@(1j*sum(sources.values()))@u
            amplified = b/s[:, None]/s[None, :]
            v = vh.conj().T
            full = v@amplified@v.conj().T
            diagonal = (v*np.diag(amplified))@v.conj().T
            assert point["statistical_green"]["full"]["frobenius"] == pytest.approx(np.linalg.norm(c), rel=2e-14)
            assert point["all_modes"]["all_mode_iC"]["full"]["frobenius"] == pytest.approx(np.linalg.norm(full), rel=2e-14)
            assert point["all_modes"]["off_diagonal_interference_iC"]["full"]["frobenius"] == pytest.approx(np.linalg.norm(full-diagonal), rel=2e-14)
            assert np.linalg.norm(full-diagonal) > 0
            assert point["reconstruction"]["iC_all_modes_vs_LU_product"]["passed"]
            assert point["all_modes"]["off_diagonal_terms_retained"]
            assert _complex(point["all_modes"]["trace_from_all_amplified_diagonal"]) == pytest.approx(np.trace(1j*c), rel=2e-14)
    json.dumps(result, allow_nan=False)


def test_per_source_propagation_uses_same_inverse_and_adds_to_full_product():
    args = _fixture()
    result = _run(args, include_zero_reference=False)
    assert result["dyson_evaluations"] == 3
    for point in result["roles"][0]["points"]:
        _, sources, inverse, _ = _literal_point(args, point["frequency_index"])
        for name, source in sources.items():
            expected = inverse@source@inverse.conj().T
            report = point["source_components"][name]
            assert report["statistical_green"]["full"]["frobenius"] == pytest.approx(np.linalg.norm(expected), rel=2e-14)
            assert report["individual_contribution_may_be_indefinite"]
        assert point["reconstruction"]["propagated_source_sum"]["passed"]
        assert point["reconstruction"]["assembled_source_vs_components"]["passed"]
    assert result["fixed_vs_zero"] == []


def test_massweighted_translation_and_layer_weights_match_svd_directions():
    args = _fixture()
    result = _run(args, include_zero_reference=False)
    point = result["roles"][0]["points"][0]
    a, _, _, _ = _literal_point(args, args[0].zero_index)
    u, singular, vh = np.linalg.svd(a)
    v, t = vh.conj().T, args[-1]
    for row in point["soft_directions"]:
        i = row["svd_descending_index"]
        assert row["singular_value"] == pytest.approx(singular[i])
        np.testing.assert_allclose(row["left_translation_direction_weights"], abs(t.T@u[:, i])**2, rtol=2e-14)
        np.testing.assert_allclose(row["right_translation_direction_weights"], abs(t.T@v[:, i])**2, rtol=2e-14)
        assert sum(row["left_layer_weights"]) == pytest.approx(1.)
        assert sum(row["right_layer_weights"]) == pytest.approx(1.)
        total = sum(_complex(value["uH_component_v"]) for value in row["operator_components"].values())
        assert total == pytest.approx(singular[i], rel=2e-14, abs=2e-14)
    assert point["operator_PQ_blocks"]["subspaces_assumed_invariant"] is False
    assert point["operator_PQ_blocks"]["PAQ"]["frobenius"] > 0


def test_admission_and_bad_mass_basis_stop_before_inverse(monkeypatch):
    args = _fixture()
    monkeypatch.setattr(np.linalg, "inv", lambda *_: pytest.fail("No inverse admitted"))
    for kwargs in ({"max_bytes": 1}, {"max_dyson_evaluations": 1},
                   {"max_svd_evaluations": 1}, {"deadline_monotonic": 0}):
        with pytest.raises((ValueError, TimeoutError)):
            _run(args, **kwargs)
    with pytest.raises(ValueError, match="not normalized"):
        _run((*args[:-1], 2*args[-1]))
    plan = diagnostic.plan_soft_source_dimensions(4, 601, 216)
    assert plan["dyson_evaluations"] == plan["svd_evaluations"] == 12
    assert plan["all_modes_retained"] == 216
    assert plan["peak_owned_bytes"] < 1024**3


def test_most_amplified_source_direction_need_not_be_softest():
    a = np.diag([.001, 1., 2.]).astype(complex)
    sources = {"left": -1j*np.diag([0., 100., 1.]),
               "right": np.zeros((3, 3), complex), "interaction": np.zeros((3, 3), complex)}
    report, _, _ = diagnostic._analyze_point(a, sources, np.eye(3), 3, 1,
        np.array([1]), np.array([0, 2]), {"all": a}, 1e-8, 1e-11, lambda: None)
    assert report["soft_directions"][0]["singular_value"] == pytest.approx(.001)
    top = report["all_modes"]["largest_diagonal_amplifications"][0]
    assert top["singular_value"] == pytest.approx(1.)
    assert _complex(top["amplified_B_ii_over_s_squared"]) == pytest.approx(100.)
