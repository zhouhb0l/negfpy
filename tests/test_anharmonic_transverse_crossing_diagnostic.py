"""Small independent matched-Green Born checks; no material field replay."""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest
from test_anharmonic_transverse_native_zero_diagnostic import _fixture

import negfpy.anharmonic.transverse_crossing_diagnostic as diagnostic
from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_crossing_support import CrossingSupportVertex

_TERMS = ("CC", "DD_over_4", "CD", "DC")
_ELEMENTS = tuple(product(range(2), repeat=2))


def _case(extra=True, huge=False):
    args = _fixture(nq=3, npositive=3, huge=huge)
    grid, baseline, dynamical, left, right, sigma, _ = args
    factors = np.zeros((3, 3), complex)
    for q, i in product(range(3), range(2)):
        factors[q, baseline.active_dofs[i]] = (i+1)/3+np.exp(2j*np.pi*q/3)*(i+2)/7
    if extra:
        for q in range(3):
            factors[q, 1] = .21+.09*np.exp(2j*np.pi*q/3)
    values = np.empty((3, 2, 3, 3), complex)
    for p, a, c, d in product(range(3), range(2), range(3), range(3)):
        values[p, a, c, d] = factors[0, baseline.active_dofs[a]]*factors[p, c]*factors[-p % 3, d]
    crossing = CrossingSupportVertex(values, baseline.active_dofs, 3, {})
    return grid, baseline, crossing, dynamical, left, right, sigma


def _matrix(summary):
    result = np.empty((2, 2), complex)
    for row in summary["selected_elements"]:
        result[row["row"], row["column"]] = row["real"]+1j*row["imag"]
    return result


def _run(args, **kwargs):
    return diagnostic.study_crossing_support_diagnostic(
        *args, selected_elements=_ELEMENTS, workers=1, external_tile=1,
        max_dyson_evaluations=21, **kwargs)


def _literal(args, g):
    """Direct encoded operator assembly and literal four-index contractions."""
    grid, baseline, crossing, dynamical, left, right, sigma = args
    z, nq, active = grid.zero_index, baseline.mesh.n_qpoints, baseline.active_dofs
    full = {}
    for q, w in product(range(nq), range(grid.n_points)):
        operator = grid.values[w]**2*np.eye(3)-dynamical[q]
        cs, ds = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
        for contact in (left, right):
            for i, j in product(range(2), repeat=2):
                target = contact.device_indices[i], contact.device_indices[j]
                operator[target] -= contact.spectrum.retarded[q, w, i, j]
                cs[target] += (contact.spectrum.greater[q, w, i, j]+contact.spectrum.lesser[q, w, i, j])/2
                ds[target] += contact.spectrum.greater[q, w, i, j]-contact.spectrum.lesser[q, w, i, j]
        for i, j in product(range(2), repeat=2):
            target = active[i], active[j]
            operator[target] -= sigma.retarded[q, w, i, j]
            cs[target] += sigma.statistical[q, w, i, j]
            ds[target] += sigma.spectral[q, w, i, j]
        inverse = np.linalg.inv(operator)
        full[q, w] = inverse@cs@inverse.conj().T, inverse@ds@inverse.conj().T
    expected = {}
    for offset in (-1, 0, 1):
        expected[offset] = {model: np.zeros((3, 4, 2, 2), complex) for model in ("baseline", "crossing")}
        for p, j in product(range(nq), range(z-2, z+3)):
            partner_w, partner_q = z+offset-(j-z), -p % nq
            c1, d1 = full[p, j]
            c2, d2 = full[partner_q, partner_w]
            for model in ("baseline", "crossing"):
                v = crossing.coefficients[p]
                first, second = (c1, d1), (c2, d2)
                if model == "baseline":
                    v = baseline.coefficients[0, p]
                    first = tuple(x[np.ix_(active, active)] for x in first)
                    second = tuple(x[np.ix_(active, active)] for x in second)
                for t, (x, y, weight) in enumerate(((first[0], second[0], 1),
                        (first[1], second[1], .25), (first[0], second[1], 1),
                        (first[1], second[0], 1))):
                    value = np.einsum("acd,ce,df,bef->ab", v, x, y, v.conj(), optimize=False)
                    expected[offset][model][abs(j-z), t] += value*(.5j*grid.integration_weight/nq)*g*g*weight
    return expected


@pytest.mark.parametrize("extra", [False, True])
def test_shared_full_green_and_exact_offset_partners_match_literal_born(extra):
    args, g = _case(extra), .021
    result, expected = _run(args, g=g), _literal(args, g)
    assert result["dyson_evaluations"] == 21
    assert result["same_fresh_green_for_both_vertices"] is True
    assert result["green_operator_and_sources_unchanged"] is True
    assert result["cached_green_or_born_used"] is False
    for row in result["born"]:
        offset = row["external_frequency_offset"]
        assert row["partner_frequency_indices"] == [2*args[0].zero_index+offset-j for j in row["internal_frequency_indices"]]
        for model in ("baseline", "crossing"):
            values = expected[offset][model]
            for partition in range(3):
                record = row[model]["partitions"]["zero" if partition == 0 else f"near{partition}"]
                for t, term in enumerate(_TERMS):
                    np.testing.assert_allclose(_matrix(record["terms"][term]), values[partition, t], rtol=4e-13, atol=1e-19)
            total = np.sum(values, axis=0)
            np.testing.assert_allclose(_matrix(row[model]["partial_total"]["statistical"]), total[0]+total[1], rtol=4e-13, atol=1e-19)
            np.testing.assert_allclose(_matrix(row[model]["partial_total"]["spectral"]), total[2]+total[3], rtol=4e-13, atol=1e-19)
        if not extra:
            assert row["comparison"]["statistical"]["crossing_to_baseline_frobenius_ratio"] == pytest.approx(1, rel=5e-14)
    json.dumps(result, allow_nan=False)


def test_spectral_source_is_independent_of_large_statistical_source():
    args = _case(extra=True, huge=True)
    result = _run(args)
    sigma = args[-1]
    without = ChannelSpectrum(sigma.retarded, sigma.statistical, np.zeros_like(sigma.spectral))
    changed = _run((*args[:-1], without))
    zero, changed_zero = result["born"][1], changed["born"][1]
    assert result["native_spectral_channel_retained"] is True
    assert zero["crossing"]["partitions"]["zero"]["terms"]["DD_over_4"]["frobenius"] > 0
    assert zero["crossing"]["partitions"]["zero"]["terms"]["DD_over_4"]["frobenius"] != changed_zero["crossing"]["partitions"]["zero"]["terms"]["DD_over_4"]["frobenius"]
    assert zero["crossing"]["term_totals"]["CC"]["frobenius"] == pytest.approx(changed_zero["crossing"]["term_totals"]["CC"]["frobenius"])


@pytest.mark.parametrize("cap", [{"max_bytes": 1}, {"max_contraction_work": 1}, {"deadline_monotonic": 0}])
def test_admission_stops_before_any_inverse(monkeypatch, cap):
    args = _case()
    monkeypatch.setattr(np.linalg, "inv", lambda *_: pytest.fail("No inverse should be admitted"))
    with pytest.raises((ValueError, TimeoutError)):
        _run(args, **cap)


def test_production_plan_counts_both_kernels_and_all_green_partners():
    plan = diagnostic.plan_crossing_diagnostic_dimensions(4, 601, 216, 120)
    assert plan["dyson_evaluations"] == 28
    assert plan["n_green_bins"] == 7
    assert plan["contraction_work"] == 891_017_625_600
    assert plan["peak_owned_bytes"] < 1024**3
    with pytest.raises(ValueError, match="sufficient bins"):
        diagnostic.plan_crossing_diagnostic_dimensions(4, 5, 216, 120)
