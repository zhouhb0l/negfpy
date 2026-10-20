"""Independent finite-supercell oracle for the coupled-q cubic bubble.

The oracle starts from a real, fully symmetric cubic tensor on every site of
a small periodic supercell. It uses an explicit unitary DFT, the existing dense
real-space bubble, and a second DFT of that result. It does not use the new
kernel's momentum routing or its reciprocal-space contraction.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_q import (
    TransverseCubicVertex,
    cubic_born_self_energy_q,
)
from negfpy.anharmonic.types import DenseCubicVertex


def _cells(shape: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    # Deliberately independent of UniformBZMesh.addresses/index_from_address.
    return [
        (x, y, z)
        for z in range(shape[2])
        for y in range(shape[1])
        for x in range(shape[0])
    ]


def _unitary_basis(
    shape: tuple[int, int, int],
    ndof: int,
    offsets: tuple[tuple[int, int, int], ...] | None = None,
) -> np.ndarray:
    cells = _cells(shape)
    if offsets is None:
        offsets = ((0, 0, 0),) * ndof
    result = np.zeros((len(cells) * ndof,) * 2, dtype=np.complex128)
    for r_index, cell in enumerate(cells):
        for q_index, q in enumerate(cells):
            for a in range(ndof):
                phase = sum(
                    q[axis] * (cell[axis] + offsets[a][axis]) / shape[axis]
                    for axis in range(3)
                )
                result[r_index * ndof + a, q_index * ndof + a] = (
                    np.exp(2j * np.pi * phase) / np.sqrt(len(cells))
                )
    return result


def _real_vertex(shape: tuple[int, int, int]) -> np.ndarray:
    """A periodic two-coordinate cubic model with genuine intercell terms."""

    cells = _cells(shape)
    index = {cell: number for number, cell in enumerate(cells)}
    values = np.zeros((2 * len(cells),) * 3, dtype=np.float64)

    def site(cell: tuple[int, int, int], delta: tuple[int, int, int], a: int) -> int:
        translated = tuple(
            (cell[axis] + delta[axis]) % shape[axis]
            for axis in range(3)
        )
        return 2 * index[translated] + a

    for cell in cells:
        terms = (
            (((0, 0, 0), 0), ((0, 0, 0), 0), ((0, 0, 0), 0), 0.23),
            (((0, 0, 0), 0), ((0, 0, 0), 1), ((0, 0, 0), 1), -0.12),
            (((0, 0, 0), 0), ((1, 0, 0), 1), ((2, 0, 0), 0), 0.09),
            (((0, 0, 0), 1), ((0, 1, 0), 0), ((1, 0, 0), 1), 0.17),
        )
        for first, second, third, strength in terms:
            indices = tuple(site(cell, delta, a) for delta, a in (first, second, third))
            for permuted in set(permutations(indices)):
                values[permuted] += strength
    return values


def _fourier_vertex(
    shape: tuple[int, int, int],
    real_vertex: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Transform all three real-space legs, retaining allowed momenta only."""

    cells = _cells(shape)
    lookup = {cell: index for index, cell in enumerate(cells)}
    nq = len(cells)
    ndof = real_vertex.shape[0] // nq
    coefficients = np.empty((nq, nq, ndof, ndof, ndof), dtype=np.complex128)
    for q_index, q in enumerate(cells):
        negative_q = tuple(-q[axis] % shape[axis] for axis in range(3))
        first = lookup[negative_q]
        for p_index, p in enumerate(cells):
            remainder = tuple((q[axis] - p[axis]) % shape[axis] for axis in range(3))
            third = lookup[remainder]
            coefficients[q_index, p_index] = np.sqrt(nq) * np.einsum(
                "Aa,Bc,Cd,ABC->acd",
                basis[:, first * ndof : (first + 1) * ndof],
                basis[:, p_index * ndof : (p_index + 1) * ndof],
                basis[:, third * ndof : (third + 1) * ndof],
                real_vertex,
                optimize=True,
            )
    return coefficients


def _to_real(q_values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    nq, nw, ndof, _ = q_values.shape
    result = np.empty((nw, nq * ndof, nq * ndof), dtype=np.complex128)
    for w in range(nw):
        blocks = np.zeros((nq * ndof,) * 2, dtype=np.complex128)
        for q in range(nq):
            block = slice(q * ndof, (q + 1) * ndof)
            blocks[block, block] = q_values[q, w]
        result[w] = basis @ blocks @ basis.conj().T
    return result


def _to_q(real_values: np.ndarray, basis: np.ndarray, nq: int) -> np.ndarray:
    nw, dimension, _ = real_values.shape
    ndof = dimension // nq
    result = np.empty((nq, nw, ndof, ndof), dtype=np.complex128)
    for w in range(nw):
        transformed = basis.conj().T @ real_values[w] @ basis
        for q in range(nq):
            block = slice(q * ndof, (q + 1) * ndof)
            result[q, w] = transformed[block, block]
        # Translation invariance forbids off-diagonal momentum blocks.
        diagonal_blocks = np.zeros_like(transformed)
        for q in range(nq):
            block = slice(q * ndof, (q + 1) * ndof)
            diagonal_blocks[block, block] = transformed[block, block]
        np.testing.assert_allclose(transformed, diagonal_blocks, rtol=0.0, atol=2e-13)
    return result


def _green_pair(nq: int, nw: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(63020 + nq)
    pair = []
    for _ in range(2):
        factors = rng.normal(size=(nq, nw, 2, 2)) + 1j * rng.normal(
            size=(nq, nw, 2, 2)
        )
        pair.append(-0.25j * (factors @ factors.conj().swapaxes(-1, -2)))
    return pair[0], pair[1]


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 1, 1), (4, 1, 1), (2, 2, 1)])
def test_coupled_q_bubble_matches_full_real_space_supercell(
    shape: tuple[int, int, int],
) -> None:
    mesh = UniformBZMesh(shape)
    grid = SignedFrequencyGrid(step=0.37, n_positive=3)
    basis = _unitary_basis(shape, ndof=2)
    real_vertex = _real_vertex(shape)
    coefficients = _fourier_vertex(shape, real_vertex, basis)
    lesser, greater = _green_pair(mesh.size, grid.n_points)

    if shape == (4, 1, 1):
        # A local scalar vertex would hide missing conjugation/phase errors.
        assert np.max(np.abs(coefficients.imag)) > 0.1
        assert np.max(np.abs(lesser[..., 0, 1].real)) > 0.1
    reciprocal = cubic_born_self_energy_q(
        grid,
        TransverseCubicVertex(mesh, coefficients),
        lesser,
        greater,
    )
    real_space = cubic_lesser_greater(
        DenseCubicVertex(real_vertex),
        _to_real(lesser, basis),
        _to_real(greater, basis),
        grid,
        backend="serial",
    )
    for actual, real_result in zip(reciprocal, real_space, strict=True):
        reference = _to_q(real_result, basis, mesh.size)
        np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=2e-13)


def test_reciprocal_bubble_respects_coordinate_cell_origin_changes() -> None:
    """Shifting one coordinate's cell label changes phases, not the operator."""

    shape = (4, 1, 1)
    mesh = UniformBZMesh(shape)
    grid = SignedFrequencyGrid(step=0.29, n_positive=2)
    original_basis = _unitary_basis(shape, ndof=2)
    shifted_basis = _unitary_basis(shape, ndof=2, offsets=((0, 0, 0), (1, 0, 0)))
    real_vertex = _real_vertex(shape)
    original_coefficients = _fourier_vertex(shape, real_vertex, original_basis)
    shifted_coefficients = _fourier_vertex(shape, real_vertex, shifted_basis)
    assert np.max(np.abs(original_coefficients - shifted_coefficients)) > 0.1
    greens = _green_pair(mesh.size, grid.n_points)
    shifted_greens = tuple(
        _to_q(_to_real(green, original_basis), shifted_basis, mesh.size)
        for green in greens
    )

    original_result = cubic_born_self_energy_q(
        grid,
        TransverseCubicVertex(mesh, original_coefficients),
        *greens,
    )
    shifted_result = cubic_born_self_energy_q(
        grid,
        TransverseCubicVertex(mesh, shifted_coefficients),
        *shifted_greens,
    )
    for original, shifted in zip(original_result, shifted_result, strict=True):
        expected = _to_q(_to_real(original, original_basis), shifted_basis, mesh.size)
        np.testing.assert_allclose(shifted, expected, rtol=2e-12, atol=2e-13)
