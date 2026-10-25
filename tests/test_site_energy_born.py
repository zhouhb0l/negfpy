"""Tiny common-Green/independent spectral/packed-support comparisons only."""
from itertools import product

import numpy as np
import pytest
from test_anharmonic_transverse_acoustic_sector import _fixture

from negfpy.anharmonic.site_energy_born import study_site_energy_born
from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex


def _problem():
    grid, vertex, dynamic, left, right, sigma, t, central = _fixture(nq=3)
    sigma = ChannelSpectrum(sigma.retarded, sigma.statistical,
        sigma.retarded-sigma.retarded.conj().swapaxes(-1, -2))
    ids = np.arange(3)
    cube = vertex.coefficients[:, :, ids[:, None, None], ids[None, :, None], ids]
    crop = PackedTransverseCubicVertex(vertex.mesh, cube, ids, 6)
    return grid, {'complete': vertex, 'crop': crop}, dynamic, left, right, sigma, t, central


def _literal(args, vertex):
    grid, _, dynamic, left, right, sigma, _, _ = args
    n, nq, z = 6, vertex.mesh.n_qpoints, grid.zero_index
    cvalues, dvalues = {}, {}
    for q, w in product(range(nq), range(grid.n_points)):
        operator = grid.values[w]**2*np.eye(n)-dynamic[q]-sigma.retarded[q, w]
        cs, ds = sigma.statistical[q, w].copy(), sigma.spectral[q, w].copy()
        for contact in (left, right):
            for i, j in product(range(2), repeat=2):
                index = contact.device_indices[i], contact.device_indices[j]
                operator[index] -= contact.spectrum.retarded[q, w, i, j]
                cs[index] += (contact.spectrum.greater[q, w, i, j]+contact.spectrum.lesser[q, w, i, j])/2
                ds[index] += contact.spectrum.greater[q, w, i, j]-contact.spectrum.lesser[q, w, i, j]
        r = np.linalg.inv(operator)
        active = np.ix_(vertex.active_dofs, vertex.active_dofs)
        cvalues[q, w] = (r@cs@r.conj().T)[active]
        dvalues[q, w] = (r@ds@r.conj().T)[active]
    totals = [np.zeros((vertex.ndof, vertex.ndof), complex) for _ in range(4)]
    for p, w in product(range(nq), range(grid.n_points)):
        c1, d1 = cvalues[p, w], dvalues[p, w]
        c2, d2 = cvalues[-p % nq, 2*z-w], dvalues[-p % nq, 2*z-w]
        v = vertex.coefficients[0, p]
        for i, (a, b, weight) in enumerate(((c1, c2, 1.), (d1, d2, .25), (c1, d2, 1.), (d1, c2, 1.))):
            totals[i] += np.einsum('acd,ce,df,bef->ab', v, a, b, v.conj(), optimize=False)*weight
    return [v*(.5j*grid.integration_weight/nq)*.03**2 for v in totals]


def test_common_full_green_and_each_packed_support_match_literal(monkeypatch):
    args = _problem()
    expected = {key: _literal(args, v) for key, v in args[1].items()}
    inverse = np.linalg.inv
    calls = []

    def counted(value):
        calls.append(value.shape)
        return inverse(value)

    monkeypatch.setattr(np.linalg, 'inv', counted)
    result = study_site_energy_born(*args, workers=1, external_tile=2)
    assert len(calls) == 15 and set(calls) == {(6, 6)}
    assert result['dyson_evaluations'] == 15
    for key, rows in expected.items():
        model = result['models'][key]
        # CD/DC can cancel across signed bins; compare their rounding on the
        # absolute scale of the nonzero four-term contraction, not near zero.
        roundoff = 5e-13*max(np.linalg.norm(v) for v in rows)
        for term, value in zip(('CC', 'DD_over_4', 'CD', 'DC'), rows, strict=True):
            assert model['born']['term_totals'][term]['full']['frobenius'] == pytest.approx(np.linalg.norm(value), rel=5e-13, abs=roundoff)
    # Cropped endpoint translations retain half their full-device norm. They
    # must not be silently normalized to one for the acoustic residual audit.
    audit = result['models']['crop']['translation_audit']
    np.testing.assert_allclose(audit['support_translation_gram'], .5*np.eye(3))
    assert len(audit['rows']) == 27


def test_admission_and_raw_source_guards_stop_before_inverse(monkeypatch):
    args = _problem()
    monkeypatch.setattr(np.linalg, 'inv', lambda *_: pytest.fail('No inverse should start'))
    for kwargs in ({'max_contraction_work': 1}, {'max_bytes': 1}, {'deadline_monotonic': 0}):
        with pytest.raises((ValueError, TimeoutError)):
            study_site_energy_born(*args, **kwargs)
    bad = ChannelSpectrum(args[5].retarded, args[5].statistical, args[5].spectral.copy())
    bad.spectral[0, 0, 0, 0] += 1e-2j
    with pytest.raises(ValueError, match='Invalid raw source'):
        study_site_energy_born(*args[:5], bad, *args[6:], workers=1)
