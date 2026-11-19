"""Exact selected Green functions and correlations for a block chain.

Only diagonal and nearest-neighbour blocks are stored.  With bounded block
size, both arithmetic and storage grow linearly with the number of blocks.
The upper and lower blocks of the retarded inverse are independent; neither
Hermiticity nor a real-valued coupling is assumed.

``factor_block_tridiagonal(A0, Aplus, Aminus)`` factors the inverse propagator
once.  ``selected_inverse()`` returns its diagonal/adjacent inverse blocks.
``correlate(S0, Splus, Sminus)`` returns the same blocks of ``G S G.conj().T``
for any block-tridiagonal source.  An indefinite source is allowed, as needed
for a temperature derivative.  Source Hermiticity is checked by default but
never imposed by symmetrisation, clipping, or copying a diagonal block.

The correlation recursion is the selected inverse of the augmented matrix

        [ A  -S ]
        [ 0  A† ],

whose upper-right inverse is G S G†.  Its triangular algebra is evaluated
directly with the already factored A, rather than factoring doubled blocks.
This is an algebraic identity, not a scattering or range approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class BlockChainError(ValueError):
    """Invalid blocks or an unusable block elimination pivot."""


def _readonly(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.complex128)
    value.setflags(write=False)
    return value


def _owned_blocks(values: Sequence[np.ndarray], name: str) -> tuple[np.ndarray, ...]:
    if not isinstance(values, (tuple, list)):
        values = tuple(values)
    result = []
    for i, value in enumerate(values):
        array = np.array(value, dtype=np.complex128, copy=True, order="C")
        if array.ndim != 2 or not np.all(np.isfinite(array)):
            raise BlockChainError(f"{name}[{i}] must be a finite matrix")
        result.append(_readonly(array))
    return tuple(result)


def _validate_shapes(
    diagonal: tuple[np.ndarray, ...],
    upper: tuple[np.ndarray, ...],
    lower: tuple[np.ndarray, ...],
    *,
    sizes: tuple[int, ...] | None = None,
    name: str,
) -> tuple[int, ...]:
    if not diagonal:
        raise BlockChainError(f"{name} must contain at least one diagonal block")
    inferred = tuple(block.shape[0] for block in diagonal)
    if any(block.shape != (size, size) or size < 1
           for block, size in zip(diagonal, inferred)):
        raise BlockChainError(f"{name} diagonal blocks must be nonempty and square")
    if sizes is not None and inferred != sizes:
        raise BlockChainError(f"{name} diagonal block sizes do not match the factor")
    if len(upper) != len(inferred) - 1 or len(lower) != len(inferred) - 1:
        raise BlockChainError(f"{name} needs exactly nblocks-1 upper and lower blocks")
    for i in range(len(upper)):
        if upper[i].shape != (inferred[i], inferred[i + 1]):
            raise BlockChainError(f"{name} upper[{i}] has an incompatible shape")
        if lower[i].shape != (inferred[i + 1], inferred[i]):
            raise BlockChainError(f"{name} lower[{i}] has an incompatible shape")
    return inferred


@dataclass(frozen=True)
class WorkStats:
    """Structural work counts, independent of machine timing.

    ``multiply_accumulates`` counts m*n*k scalar complex multiply-accumulates
    for each m-by-k times k-by-n block product. ``solve_cubic_work`` is sum b³
    for the local solves, not a hardware flop or elapsed-time estimate.
    """

    nblocks: int
    block_sizes: tuple[int, ...]
    block_solves: int
    block_multiplications: int
    multiply_accumulates: int
    solve_cubic_work: int
    maximum_solve_dimension: int
    maximum_pivot_inverse_residual: float = 0.0
    maximum_selected_diagonal_residual: float = 0.0
    global_dense_matrix_created: bool = False

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class _Work:
    def __init__(self, sizes: tuple[int, ...]):
        self.sizes = sizes
        self.solves = 0
        self.products = 0
        self.macs = 0
        self.solve_work = 0
        self.max_solve = 0
        self.pivot_residual = 0.0
        self.selected_residual = 0.0

    def mul(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        self.products += 1
        self.macs += left.shape[0] * left.shape[1] * right.shape[1]
        return left @ right

    def inverse(self, value: np.ndarray, index: int) -> np.ndarray:
        size = value.shape[0]
        self.solves += 1
        self.solve_work += size ** 3
        self.max_solve = max(self.max_solve, size)
        try:
            result = np.linalg.solve(value, np.eye(size, dtype=np.complex128))
        except np.linalg.LinAlgError as error:
            raise BlockChainError(f"singular block elimination pivot {index}") from error
        if not np.all(np.isfinite(result)):
            raise BlockChainError(f"nonfinite inverse of block elimination pivot {index}")
        residual = self.mul(value, result) - np.eye(size, dtype=np.complex128)
        self.pivot_residual = max(
            self.pivot_residual, float(np.linalg.norm(residual) / np.sqrt(size))
        )
        return result

    def freeze(self) -> WorkStats:
        return WorkStats(
            nblocks=len(self.sizes), block_sizes=self.sizes,
            block_solves=self.solves, block_multiplications=self.products,
            multiply_accumulates=self.macs, solve_cubic_work=self.solve_work,
            maximum_solve_dimension=self.max_solve,
            maximum_pivot_inverse_residual=self.pivot_residual,
            maximum_selected_diagonal_residual=self.selected_residual,
        )


@dataclass(frozen=True)
class SelectedBlocks:
    """Diagonal and adjacent blocks in the original block ordering."""

    diagonal: tuple[np.ndarray, ...]
    upper: tuple[np.ndarray, ...]
    lower: tuple[np.ndarray, ...]
    stats: WorkStats

    @property
    def block_sizes(self) -> tuple[int, ...]:
        return tuple(block.shape[0] for block in self.diagonal)

    @property
    def stored_nbytes(self) -> int:
        return sum(block.nbytes for block in self.diagonal + self.upper + self.lower)

    def block(self, row: int, column: int) -> np.ndarray:
        if row < 0 or column < 0 or row >= len(self.diagonal) or column >= len(self.diagonal):
            raise IndexError("block index outside the chain")
        if row == column:
            return self.diagonal[row]
        if column == row + 1:
            return self.upper[row]
        if row == column + 1:
            return self.lower[column]
        raise IndexError("only diagonal and adjacent blocks were calculated")


class BlockTridiagonalFactor:
    """Reusable local Schur factors of a general complex block chain.

    No inter-block pivoting is performed.  A singular Schur pivot is rejected
    explicitly.  For the passive retarded operators used here, finite positive
    broadening provides nonsingular principal blocks.  A caller should retain
    its independent residual and physical qualification gates.
    """

    def __init__(self, diagonal, upper, lower):
        self.diagonal = _owned_blocks(diagonal, "A diagonal")
        self.upper = _owned_blocks(upper, "A upper")
        self.lower = _owned_blocks(lower, "A lower")
        self.block_sizes = _validate_shapes(
            self.diagonal, self.upper, self.lower, name="A"
        )
        self.nblocks = len(self.block_sizes)
        work = _Work(self.block_sizes)
        pivots = []
        transfer_upper = []
        transfer_lower = []
        for i, diagonal_block in enumerate(self.diagonal):
            if i == 0:
                schur = diagonal_block
            else:
                schur = diagonal_block - work.mul(transfer_lower[i - 1], self.upper[i - 1])
            inverse = _readonly(work.inverse(schur, i))
            pivots.append(inverse)
            if i + 1 < self.nblocks:
                transfer_upper.append(_readonly(work.mul(inverse, self.upper[i])))
                transfer_lower.append(_readonly(work.mul(self.lower[i], inverse)))
        self.left_inverse = tuple(pivots)
        self.transfer_upper = tuple(transfer_upper)
        self.transfer_lower = tuple(transfer_lower)
        self.factor_stats = work.freeze()
        self._inverse = self._selected_inverse()

    def _selected_inverse(self) -> SelectedBlocks:
        work = _Work(self.block_sizes)
        diagonals = [None] * self.nblocks
        uppers = [None] * (self.nblocks - 1)
        lowers = [None] * (self.nblocks - 1)
        diagonals[-1] = self.left_inverse[-1]
        for i in range(self.nblocks - 2, -1, -1):
            uppers[i] = _readonly(-work.mul(self.transfer_upper[i], diagonals[i + 1]))
            lowers[i] = _readonly(-work.mul(diagonals[i + 1], self.transfer_lower[i]))
            diagonals[i] = _readonly(
                self.left_inverse[i] - work.mul(uppers[i], self.transfer_lower[i])
            )
        for i in range(self.nblocks):
            residual = work.mul(self.diagonal[i], diagonals[i])
            if i:
                residual = residual + work.mul(self.lower[i - 1], uppers[i - 1])
            if i + 1 < self.nblocks:
                residual = residual + work.mul(self.upper[i], lowers[i])
            residual = residual - np.eye(self.block_sizes[i], dtype=np.complex128)
            work.selected_residual = max(
                work.selected_residual,
                float(np.linalg.norm(residual) / np.sqrt(self.block_sizes[i])),
            )
        return SelectedBlocks(tuple(diagonals), tuple(uppers), tuple(lowers), work.freeze())

    @property
    def stored_nbytes(self) -> int:
        arrays = (
            self.diagonal + self.upper + self.lower + self.left_inverse
            + self.transfer_upper + self.transfer_lower
            + self._inverse.diagonal + self._inverse.upper + self._inverse.lower
        )
        unique = {id(array): array for array in arrays}
        return sum(array.nbytes for array in unique.values())

    def selected_inverse(self) -> SelectedBlocks:
        """Return the cached exact diagonal and adjacent inverse blocks."""
        return self._inverse

    def correlate(
        self,
        source_diagonal: Sequence[np.ndarray],
        source_upper: Sequence[np.ndarray] | None = None,
        source_lower: Sequence[np.ndarray] | None = None,
        *,
        check_hermitian: bool = True,
        hermitian_rtol: float = 1e-10,
        hermitian_atol: float = 1e-12,
    ) -> SelectedBlocks:
        """Calculate selected Q=G S G† without global G or pair transmissions.

        Missing off-diagonal sources mean zero.  If upper blocks are supplied
        without lower blocks, their conjugate transposes define the lower
        source.  Explicitly supplied lower blocks are retained as given.
        ``check_hermitian=False`` also permits a general complex banded source;
        this is useful for an independent optical-identity diagnostic.
        """
        source_diagonal = _owned_blocks(source_diagonal, "S diagonal")
        if source_upper is None:
            source_upper = tuple(np.zeros_like(block) for block in self.upper)
        source_upper = _owned_blocks(source_upper, "S upper")
        if source_lower is None:
            source_lower = tuple(block.conj().T for block in source_upper)
        source_lower = _owned_blocks(source_lower, "S lower")
        _validate_shapes(
            source_diagonal, source_upper, source_lower,
            sizes=self.block_sizes, name="S",
        )
        if (not np.isfinite(hermitian_rtol) or not np.isfinite(hermitian_atol)
                or hermitian_rtol < 0 or hermitian_atol < 0):
            raise BlockChainError("Hermitian tolerances must be finite and nonnegative")
        if check_hermitian:
            for i, block in enumerate(source_diagonal):
                defect = np.linalg.norm(block - block.conj().T)
                scale = np.linalg.norm(block)
                if defect > hermitian_atol + hermitian_rtol * scale:
                    raise BlockChainError(f"S diagonal[{i}] is not Hermitian")
            for i, (up, low) in enumerate(zip(source_upper, source_lower)):
                defect = np.linalg.norm(up - low.conj().T)
                scale = max(np.linalg.norm(up), np.linalg.norm(low))
                if defect > hermitian_atol + hermitian_rtol * scale:
                    raise BlockChainError(f"S upper/lower[{i}] are not adjoints")

        work = _Work(self.block_sizes)
        left_correlations = []
        for i, source in enumerate(source_diagonal):
            if i:
                transfer = self.transfer_lower[i - 1]
                source = (
                    source
                    - work.mul(transfer, source_upper[i - 1])
                    - work.mul(source_lower[i - 1], transfer.conj().T)
                    + work.mul(
                        work.mul(self.lower[i - 1], left_correlations[i - 1]),
                        self.lower[i - 1].conj().T,
                    )
                )
            inverse = self.left_inverse[i]
            left_correlations.append(
                work.mul(work.mul(inverse, source), inverse.conj().T)
            )

        diagonals = [None] * self.nblocks
        uppers = [None] * (self.nblocks - 1)
        lowers = [None] * (self.nblocks - 1)
        diagonals[-1] = _readonly(left_correlations[-1])
        for i in range(self.nblocks - 2, -1, -1):
            inverse = self.left_inverse[i]
            left_q = left_correlations[i]
            a = self.transfer_upper[i]
            b = work.mul(left_q, self.lower[i].conj().T) - work.mul(inverse, source_upper[i])
            e = work.mul(self.lower[i], left_q) - work.mul(source_lower[i], inverse.conj().T)
            next_g = self._inverse.diagonal[i + 1]
            uppers[i] = _readonly(
                -work.mul(a, diagonals[i + 1]) - work.mul(b, next_g.conj().T)
            )
            lowers[i] = _readonly(
                -work.mul(next_g, e) - work.mul(diagonals[i + 1], a.conj().T)
            )
            diagonals[i] = _readonly(
                left_q - work.mul(self._inverse.upper[i], e)
                - work.mul(uppers[i], a.conj().T)
            )
        for label, blocks in (("diagonal", diagonals), ("upper", uppers), ("lower", lowers)):
            if any(not np.all(np.isfinite(block)) for block in blocks):
                raise BlockChainError(f"nonfinite selected correlation {label} block")
        return SelectedBlocks(tuple(diagonals), tuple(uppers), tuple(lowers), work.freeze())


def factor_block_tridiagonal(diagonal, upper, lower) -> BlockTridiagonalFactor:
    """Factor a finite block chain; caller blocks are copied and never mutated."""
    return BlockTridiagonalFactor(diagonal, upper, lower)


def trace_banded_product(
    selected: SelectedBlocks,
    diagonal: Sequence[np.ndarray],
    upper: Sequence[np.ndarray] | None = None,
    lower: Sequence[np.ndarray] | None = None,
) -> complex:
    """Return Tr(B X) for banded B and selected blocks of X.

    No dense global matrix is needed.  B may be general complex; missing upper
    blocks mean zero, and missing lower blocks mean the adjoints of upper.
    """
    diagonal = _owned_blocks(diagonal, "B diagonal")
    if upper is None:
        upper = tuple(np.zeros_like(block) for block in selected.upper)
    upper = _owned_blocks(upper, "B upper")
    if lower is None:
        lower = tuple(block.conj().T for block in upper)
    lower = _owned_blocks(lower, "B lower")
    _validate_shapes(diagonal, upper, lower, sizes=selected.block_sizes, name="B")
    value = sum(np.einsum("ij,ji->", left, right)
                for left, right in zip(diagonal, selected.diagonal))
    value += sum(np.einsum("ij,ji->", left, right)
                 for left, right in zip(upper, selected.lower))
    value += sum(np.einsum("ij,ji->", left, right)
                 for left, right in zip(lower, selected.upper))
    return complex(value)
