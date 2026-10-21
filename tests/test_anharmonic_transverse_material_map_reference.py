"""One evaluated packed material map versus small independent dense algebra."""

from __future__ import annotations

import numpy as np
import pytest
from test_anharmonic_transverse_fft import _polynomial_vertex

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.retarded import retarded_from_keldysh
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_material_map import (
    PackedMapSpectrum,
    evaluate_packed_born_map,
)
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_q import (
    TransverseCubicVertex,
    cubic_born_self_energy_q,
)
from negfpy.anharmonic.transverse_solver import TransverseSpectrum
from negfpy.anharmonic.types import SolverScale

_COMPONENTS = ("retarded", "lesser", "greater")


def _ohmic_contact(mesh, grid, indices, *, side, temperature, damping):
    """Analytic contact fields independent of the material contact builder."""
    nq, size = mesh.n_qpoints, len(indices)
    matrices = np.tile(np.eye(size) * damping, (nq, 1, 1)).astype(complex)
    for q in range(nq):
        matrices[q, 0, 1] = 0.13 * damping * np.exp(2j * np.pi * q / nq)
        matrices[q, 1, 0] = matrices[q, 0, 1].conjugate()
    w = grid.values
    product = np.empty_like(w)
    nonzero = w != 0
    product[nonzero] = w[nonzero] / np.expm1(w[nonzero] / temperature)
    product[grid.zero_index] = temperature
    retarded = -1j * w[None, :, None, None] * matrices[:, None]
    lesser = -2j * product[None, :, None, None] * matrices[:, None]
    greater = lesser + retarded - retarded.conj().swapaxes(-1, -2)
    contact = TransverseContact(
        mesh, grid, TransverseSpectrum(retarded, lesser, greater), 2 * matrices,
        tuple({"fixture": "analytic-ohmic"} for _ in range(nq)), {},
        side, 0.02, temperature,
    )
    return CompactTransverseContact(contact, 5, tuple(indices))


def _fixture(nq):
    reduced = _polynomial_vertex((nq, 1, 1))
    grid = SignedFrequencyGrid(step=0.31, n_positive=4)
    active = np.array([3, 1])
    vertex = PackedTransverseCubicVertex(
        reduced.mesh, reduced.coefficients, active, 5,
    )
    dynamical = np.tile(np.diag([1.3, 1.7, 1.45, 1.6, 1.9]), (nq, 1, 1)).astype(complex)
    for q in range(nq):
        phase = np.exp(2j * np.pi * q / nq)
        for first in range(4):
            dynamical[q, first, first + 1] = -0.09 - 0.035 * phase
            dynamical[q, first + 1, first] = dynamical[q, first, first + 1].conjugate()
        dynamical[q, 0, 4] = -0.07 * phase
        dynamical[q, 4, 0] = dynamical[q, 0, 4].conjugate()
    left = _ohmic_contact(
        reduced.mesh, grid, [4, 0], side="left", temperature=0.87, damping=0.29,
    )
    right = _ohmic_contact(
        reduced.mesh, grid, [2, 4], side="right", temperature=0.63, damping=0.23,
    )
    return grid, vertex, dynamical, left, right


def _causal_input(grid, vertex):
    """PSD contour sources with paired complex q blocks, then direct finite PV."""
    nq = vertex.mesh.n_qpoints
    weight = np.empty((nq, 2, 2), complex)
    for q in range(nq):
        phase = np.exp(2j * np.pi * q / nq)
        weight[q] = np.array([[1.0, 0.21 * phase], [0.21 * phase.conjugate(), 0.7]])
    w = grid.values
    envelope = 0.013 * np.exp(-0.47 * w**2)
    lesser = -1j * (envelope * np.exp(-w / 1.3))[None, :, None, None] * weight[:, None]
    greater = -1j * (envelope * np.exp(w / 1.3))[None, :, None, None] * weight[:, None]
    retarded = np.stack([
        retarded_from_keldysh(lesser[q], greater[q], grid, hilbert_backend="direct")
        for q in range(nq)
    ])
    return PackedMapSpectrum(retarded, lesser, greater)


def _embed(fields, indices, full_ndof):
    result = {}
    for name in _COMPONENTS:
        values = getattr(fields, name)
        full = np.zeros(values.shape[:2] + (full_ndof, full_ndof), complex)
        for a, i in enumerate(indices):
            for b, j in enumerate(indices):
                full[:, :, i, j] = values[:, :, a, b]
        result[name] = full
    return result


def _direct_reference(grid, vertex, dynamical, left, right, sigma_input, scale, g):
    """Full harmonic inverse, full embedded cubic contraction, direct PV."""
    nq, nw, dimension = vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof
    contacts = [_embed(c.contact.spectrum, c.device_indices, dimension) for c in (left, right)]
    if sigma_input is None:
        sigma = {name: np.zeros((nq, nw, dimension, dimension), complex) for name in _COMPONENTS}
    else:
        sigma = _embed(sigma_input, vertex.active_dofs, dimension)
    total = {name: contacts[0][name] + contacts[1][name] + sigma[name] for name in _COMPONENTS}
    r = np.linalg.inv(
        grid.values[None, :, None, None]**2 * np.eye(dimension)
        - dynamical[:, None] - total["retarded"],
    )
    advanced = r.conj().swapaxes(-1, -2)
    green = {"retarded": r, "lesser": r @ total["lesser"] @ advanced,
             "greater": r @ total["greater"] @ advanced}

    dense_values = np.zeros((nq, nq, dimension, dimension, dimension), complex)
    dense_values[np.ix_(np.arange(nq), np.arange(nq),
                       vertex.active_dofs, vertex.active_dofs, vertex.active_dofs)] = (
        g * vertex.coefficients
    )
    dense = TransverseCubicVertex(vertex.mesh, dense_values)
    lesser, greater = cubic_born_self_energy_q(grid, dense, green["lesser"], green["greater"])
    retarded = np.stack([
        retarded_from_keldysh(lesser[q], greater[q], grid, hilbert_backend="direct")
        for q in range(nq)
    ])
    mapped = dict(zip(_COMPONENTS, (retarded, lesser, greater), strict=True))
    currents = []
    for source in (*contacts, sigma):
        value = 0.0
        for q in range(nq):
            for iw in grid.positive_indices:
                exchange = source["greater"][q, iw] @ green["lesser"][q, iw]
                exchange -= source["lesser"][q, iw] @ green["greater"][q, iw]
                value += grid.values[iw] * np.trace(exchange).real
        currents.append(scale.to_physical_current(value * grid.integration_weight / nq))
    return green, mapped, currents, total


@pytest.mark.parametrize("nq,with_input,g", [(1, False, 0.19), (4, False, 0.03), (4, True, 0.27)])
def test_one_map_matches_full_dense_reference_and_keeps_evaluated_currents(nq, with_input, g):
    grid, vertex, dynamical, left, right = _fixture(nq)
    sigma = _causal_input(grid, vertex) if with_input else None
    scale = SolverScale(omega0=1.9, hbar=0.7)
    inputs = [dynamical, vertex.coefficients]
    inputs.extend(getattr(c.contact.spectrum, name) for c in (left, right) for name in _COMPONENTS)
    if sigma is not None:
        inputs.extend(getattr(sigma, name) for name in _COMPONENTS)
    before = [array.copy() for array in inputs]
    actual = evaluate_packed_born_map(
        grid, vertex, dynamical, left, right, sigma_input=sigma, scale=scale, g=g,
        workers=2, external_tile=1, bin_tile=3,
    )
    green, mapped, currents, total = _direct_reference(
        grid, vertex, dynamical, left, right, sigma, scale, g,
    )
    active = vertex.active_dofs
    for name in _COMPONENTS:
        expected_green = green[name][:, :, active[:, None], active[None, :]]
        expected_map = mapped[name][:, :, active[:, None], active[None, :]]
        np.testing.assert_allclose(getattr(actual.green, name), expected_green, rtol=2e-11, atol=3e-12)
        np.testing.assert_allclose(getattr(actual.map_sigma, name), expected_map, rtol=3e-11, atol=3e-13)
        assert not getattr(actual.green, name).flags.writeable
        assert not getattr(actual.map_sigma, name).flags.writeable
    for name, expected in zip(("left", "right", "collision_power"), currents, strict=True):
        assert getattr(actual.input_currents, name) == pytest.approx(expected, rel=2e-11, abs=3e-13)
    assert actual.input_sigma_is_zero is (not with_input)
    if not with_input:
        assert actual.input_currents.collision_power == 0.0
    else:
        # Currents belong to the input used for Dyson. The new map has not
        # been evaluated through a second Dyson solve or silently substituted.
        assert abs(actual.input_currents.collision_power) > 1e-8
        assert np.max(np.abs(actual.map_sigma.lesser - sigma.lesser)) > 1e-4
    if nq == 4:
        assert np.max(np.abs(dynamical.imag)) > 0.01
        assert np.max(np.abs(vertex.coefficients.imag)) > 1e-3
    # Cropping the harmonic matrix to interaction coordinates is incorrect:
    # this independent calculation deliberately omits buffer-mediated motion.
    reduced = np.linalg.inv(
        grid.values[None, :, None, None]**2 * np.eye(len(active))
        - dynamical[:, active[:, None], active[None, :]][:, None]
        - total["retarded"][:, :, active[:, None], active[None, :]],
    )
    assert np.max(np.abs(actual.green.retarded - reduced)) > 0.1
    for array, original in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(array, original)
