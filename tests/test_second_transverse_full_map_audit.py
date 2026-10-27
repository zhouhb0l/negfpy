"""Independent two-axis Dyson/Born/current oracle using synthetic fields only."""

from itertools import product

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSpectrum,
    evaluate_channel_born_map,
)
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_solver import TransverseSpectrum
from negfpy.anharmonic.types import SolverScale


def _fixture(shape):
    nx, ny = shape
    labels = [(x, y) for y in range(ny) for x in range(nx)]
    lookup = {label: i for i, label in enumerate(labels)}
    mesh = UniformBZMesh((nx, ny, 1))
    grid = SignedFrequencyGrid(.37, 2)
    phase = lambda label, shift: np.exp(2j*np.pi*sum(
        a*b/n for a, b, n in zip(label, shift, shape, strict=True)))
    nq = len(labels)
    factors = np.empty((2, nq, 2), complex)
    for rank, q, a in product(range(2), range(nq), range(2)):
        factors[rank, q, a] = ((rank+2)*(a+1)/9
            +(a-rank+2)*phase(labels[q], (1, 2))/13
            +.07*(a+1)*phase(labels[q], (-2, 1)))
    coefficients = np.zeros((nq, nq, 2, 2, 2), complex)
    for q, p in product(range(nq), repeat=2):
        minus = lookup[tuple((-v) % n for v, n in zip(labels[q], shape, strict=True))]
        partner = lookup[tuple((v-u) % n for v, u, n in
                               zip(labels[q], labels[p], shape, strict=True))]
        for rank, strength in enumerate((.8, -.27)):
            coefficients[q, p] += strength*np.einsum(
                'a,c,d->acd', factors[rank, minus], factors[rank, p], factors[rank, partner])
    vertex = PackedTransverseCubicVertex(mesh, coefficients, np.array([2, 0]), 3)
    matrices = np.tile(np.diag([1.3, 1.6, 1.9]), (nq, 1, 1)).astype(complex)
    weights = []
    for q, label in enumerate(labels):
        z = phase(label, (1, 2))
        t = phase(label, (-2, 1))
        for a, b, value in ((0, 1, -.12-.035*z), (1, 2, -.09+.021*t),
                            (0, 2, -.04*z+.013*t)):
            matrices[q, a, b] = value
            matrices[q, b, a] = value.conjugate()
        weights.append(np.array([[1.2, .15*z+.04*t], [.15/z+.04/t, .8]]))
    contacts = []
    for side, indices, temperature, damping in (
        ('left', (0, 1), .91, .31), ('right', (1, 2), .57, .23),
    ):
        weight = damping*np.asarray(weights)
        bose_product = np.array([temperature if w == 0 else w/np.expm1(w/temperature)
                                 for w in grid.values])
        retarded = -1j*grid.values[None, :, None, None]*weight[:, None]
        lesser = -2j*bose_product[None, :, None, None]*weight[:, None]
        greater = lesser+retarded-retarded.conj().swapaxes(-1, -2)
        raw = TransverseContact(mesh, grid, TransverseSpectrum(retarded, lesser, greater),
            2*weight, tuple({'synthetic': 'two-axis-ohmic'} for _ in labels), {},
            side, .0015625, temperature)
        contacts.append(CompactTransverseContact(raw, 3, indices))
    return labels, grid, vertex, matrices, *contacts


def _embed(fields, indices, full=3):
    result = np.zeros(fields.shape[:2]+(full, full), complex)
    for a, b in product(range(len(indices)), repeat=2):
        result[:, :, indices[a], indices[b]] = fields[:, :, a, b]
    return result


def _direct_map(labels, grid, vertex, matrices, left, right, sigma, shape, scale, g):
    """Literal two-coordinate conservation, finite frequency sums and full inverse."""
    nq, nw = len(labels), grid.n_points
    sources = []
    for contact in (left, right):
        fields = contact.spectrum
        sources.append({
            'retarded': _embed(fields.retarded, contact.device_indices),
            'statistical': _embed((fields.greater+fields.lesser)/2, contact.device_indices),
            'spectral': _embed(fields.greater-fields.lesser, contact.device_indices),
        })
    sources.append({name: np.zeros((nq, nw, 3, 3), complex) if sigma is None else
                    _embed(getattr(sigma, name), vertex.active_dofs)
                    for name in ('retarded', 'statistical', 'spectral')})
    green = {name: np.empty((nq, nw, 3, 3), complex)
             for name in ('retarded', 'statistical', 'spectral')}
    for q, w in product(range(nq), range(nw)):
        total = {name: sum(source[name][q, w] for source in sources) for name in green}
        r = np.linalg.inv(grid.values[w]**2*np.eye(3)-matrices[q]-total['retarded'])
        green['retarded'][q, w] = r
        for name in ('statistical', 'spectral'):
            green[name][q, w] = r@total[name]@r.conj().T
    active = vertex.active_dofs
    retained = {name: values[:, :, active[:, None], active] for name, values in green.items()}
    mapped = {name: np.zeros((nq, nw, 2, 2), complex) for name in green}
    c, d = retained['statistical'], retained['spectral']
    for q, p, w, internal in product(range(nq), range(nq), range(nw), range(nw)):
        other_w = w-internal+grid.zero_index
        if not 0 <= other_w < nw:
            continue
        # Search reciprocal-conserving labels, rather than use a production route table.
        other_q, = [j for j, label in enumerate(labels) if all(
            (labels[p][axis]+label[axis]-labels[q][axis]) % shape[axis] == 0
            for axis in range(2))]
        v = vertex.coefficients[q, p]
        contract = lambda first, second, v=v: np.einsum('aij,bkl,ik,jl->ab',
            v, v.conj(), first, second)
        prefactor = .5j*grid.step/(2*np.pi*nq)*g**2
        mapped['statistical'][q, w] += prefactor*(
            contract(c[p, internal], c[other_q, other_w])
            +contract(d[p, internal], d[other_q, other_w])/4)
        mapped['spectral'][q, w] += prefactor*(
            contract(c[p, internal], d[other_q, other_w])
            +contract(d[p, internal], c[other_q, other_w]))
    mapped['retarded'] = mapped['spectral']/2
    for w, j in product(range(nw), repeat=2):
        if w != j:
            mapped['retarded'][:, w] += 1j*mapped['spectral'][:, j]/(2*np.pi*(w-j))
    currents = []
    for source in sources:
        current = 0.
        for q, w in product(range(nq), grid.positive_indices):
            sl = source['statistical'][q, w]-source['spectral'][q, w]/2
            sg = source['statistical'][q, w]+source['spectral'][q, w]/2
            gl = green['statistical'][q, w]-green['spectral'][q, w]/2
            gg = green['statistical'][q, w]+green['spectral'][q, w]/2
            current += grid.values[w]*np.trace(sg@gl-sl@gg).real
        currents.append(scale.to_physical_current(current*grid.integration_weight/nq))
    return retained, mapped, currents


@pytest.mark.parametrize('shape', [(2, 2), (2, 3), (4, 2)])
@pytest.mark.parametrize('with_input', [False, True])
def test_full_native_map_and_currents_match_independent_two_axis_sum(shape, with_input):
    labels, grid, vertex, matrices, left, right = _fixture(shape)
    scale, g = SolverScale(omega0=1.7, hbar=.6), .023
    sigma = None
    if with_input:
        first = evaluate_channel_born_map(grid, vertex, matrices, left, right,
            g=g, scale=scale, workers=1, external_tile=1, bin_tile=2)
        sigma = ChannelSpectrum(*(getattr(first.map_sigma, name)*.05
                                  for name in ('retarded', 'statistical', 'spectral')))
    actual = evaluate_channel_born_map(grid, vertex, matrices, left, right,
        sigma_input=sigma, g=g, scale=scale, workers=1, external_tile=1, bin_tile=2)
    green, mapped, currents = _direct_map(labels, grid, vertex, matrices,
                                         left, right, sigma, shape, scale, g)
    for name in green:
        np.testing.assert_allclose(getattr(actual.green, name), green[name],
                                   rtol=5e-12, atol=3e-14)
        np.testing.assert_allclose(getattr(actual.map_sigma, name), mapped[name],
                                   rtol=6e-12, atol=3e-15)
    for name, expected in zip(('left', 'right', 'collision_power'), currents, strict=True):
        assert getattr(actual.input_currents, name) == pytest.approx(expected,
                                                                   rel=2e-10, abs=2e-15)
    assert actual.input_sigma_is_zero is (not with_input)
    # All 2x2 points are self-opposite and their real-space phases are real.
    # The off-Nyquist meshes must additionally exercise genuinely complex phases.
    if max(shape) > 2:
        assert max(abs(matrices[:, 0, 1].imag)) > .01
        assert np.max(np.abs(vertex.coefficients.imag)) > .001
