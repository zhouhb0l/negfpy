"""Exact left-Schur-only cache for a frequency-batched block chain.

The cache retains left Schur inverses and exact raw upper/lower operator
families. Its retained-array count is separate from whole-run memory admission.
Repeated raw blocks are shared ONLY when the
caller supplies the identical Python array object; numerical equality is never
used to infer periodicity.  Construction copies those identities once and
makes its owned arrays read-only.

The public Bands/correlate/selected_block interface follows the earlier
batched_blocks.py backend.  selected_inverse() returns temporary bands for
startup qualification.  No diagonal Green cache or transfer cache is retained.
All source actions reconstruct them by multiplication, with zero new solves.
Repeated selected_block calls cost O(N) each: use selected_inverse once for
bulk startup observables and correlate for linear-work current actions.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def adjoint(value):
    return value.conj().swapaxes(-1, -2)


@dataclass
class Bands:
    diagonal: list
    upper: list
    lower: list

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.diagonal + self.upper + self.lower)


def _owned_families(blocks, memo):
    result = []
    for original in blocks:
        identity = id(original)
        if identity not in memo:
            value = np.array(original, dtype=np.complex128, copy=True)
            value.setflags(write=False)
            memo[identity] = value
        result.append(memo[identity])
    return result


class GOnlyFactor:
    """Batched exact factor, retaining g_i and explicitly shared raw couplings.

    Numeric nbytes counts unique retained arrays only.  Python metadata,
    returned output bands, constructor qualification bands, source forward
    bands, and matrix-product/library scratch are separate allocations.
    """

    def __init__(self, diagonal, upper, lower):
        diagonal = [np.array(a, dtype=np.complex128, copy=True) for a in diagonal]
        if not diagonal:
            raise ValueError('Empty block chain')
        self.sizes = tuple(a.shape[-1] if a.ndim == 3 else -1 for a in diagonal)
        self.batch = diagonal[0].shape[0] if diagonal[0].ndim == 3 else 0
        n = len(self.sizes)
        if self.batch < 1 or any(b < 1 for b in self.sizes):
            raise ValueError('Invalid diagonal shape')
        if len(upper) != n-1 or len(lower) != n-1:
            raise ValueError('Wrong block count')
        memo = {}
        self.raw_upper = tuple(_owned_families(upper, memo))
        self.raw_lower = tuple(_owned_families(lower, memo))
        for a, size in zip(diagonal, self.sizes):
            if a.shape != (self.batch, size, size):
                raise ValueError('Invalid diagonal shape')
        for i, (up, low) in enumerate(zip(self.raw_upper, self.raw_lower)):
            if (up.shape != (self.batch, self.sizes[i], self.sizes[i+1])
                    or low.shape != (self.batch, self.sizes[i+1], self.sizes[i])):
                raise ValueError('Invalid coupling shape')
        if not all(np.isfinite(a).all() for a in diagonal + list(self.raw_upper) + list(self.raw_lower)):
            raise ValueError('Nonfinite operator')
        self.left = []
        self.pivot_residual = 0.
        self.diagonal_residual = 0.
        self.source_calls = 0
        self.qualification_source_calls = 0
        self.selected_inverse_calls = 0
        self.source_block_products_recorded_total = 0
        self._factor_block_solves = self.batch*n
        self._inherited_factor_block_solves = 0
        self._conversion_no_new_inversions = False
        self.compacted = True
        for i, block in enumerate(diagonal):
            pivot = block if i == 0 else block-self.raw_lower[i-1]@self.left[i-1]@self.raw_upper[i-1]
            identity = np.broadcast_to(np.eye(self.sizes[i]), pivot.shape)
            g = np.linalg.solve(pivot, identity)
            residue = np.linalg.norm(pivot@g-identity, axis=(-2, -1))/np.sqrt(self.sizes[i])
            self.pivot_residual = max(self.pivot_residual, float(np.max(residue)))
            if not np.isfinite(g).all():
                raise ValueError('Nonfinite Schur inverse')
            g.setflags(write=False)
            self.left.append(g)
        self.left = tuple(self.left)
        selected = self.selected_inverse()
        for i in range(n):
            residue = diagonal[i]@selected.diagonal[i]-np.eye(self.sizes[i])
            if i:
                residue += self.raw_lower[i-1]@selected.upper[i-1]
            if i < n-1:
                residue += self.raw_upper[i]@selected.lower[i]
            self.diagonal_residual = max(self.diagonal_residual, float(np.max(
                np.linalg.norm(residue, axis=(-2, -1))/np.sqrt(self.sizes[i]))))
        if self.pivot_residual > 1e-8 or self.diagonal_residual > 1e-8:
            raise ValueError('Recursive inverse residual failed original raw tolerance')

    @classmethod
    def from_existing_factor(cls, factor, *, source_point_qualified=False,
                             upper_aliases=None, lower_aliases=None):
        """Copy an already qualified, un-compacted original BatchedFactor.

        No matrix solve occurs.  The caller must run the ORIGINAL point-level
        raw Sigma, inverse, source, optical and uniform-current guards before
        conversion.  source_point_qualified is an explicit caller assertion,
        not independent physical recertification by this cache utility.

        Optional alias lists express exact construction identities.  Every
        entry is checked bit-for-bit against the original raw operator before
        its identity is used for sharing.  Equal distinct objects are never
        coalesced.  All admitted left/raw arrays are copied, then read-only;
        the old full factor may subsequently be released by its owner.
        """
        if source_point_qualified is not True:
            raise ValueError('Original point-level qualification is required')
        if (getattr(factor, 'compacted', None) is not False
                or getattr(factor, 'a', None) is None
                or getattr(factor, 'green', None) is None):
            raise ValueError('An un-compacted original qualified factor is required')
        for name in ('pivot_residual', 'diagonal_residual'):
            value = getattr(factor, name, None)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1e-8:
                raise ValueError('Original factor inverse qualification is missing')
        result = cls.__new__(cls)
        result.sizes = tuple(factor.sizes)
        result.batch = int(factor.batch)
        if not result.sizes or result.batch < 1:
            raise ValueError('Original factor dimensions are invalid')
        upper = factor.a.upper if upper_aliases is None else upper_aliases
        lower = factor.a.lower if lower_aliases is None else lower_aliases
        if len(upper) != len(result.sizes)-1 or len(lower) != len(result.sizes)-1:
            raise ValueError('Raw coupling alias count differs')
        for original, aliases in ((factor.a.upper, upper), (factor.a.lower, lower)):
            for actual, alias in zip(original, aliases):
                if (not isinstance(alias, np.ndarray) or actual.shape != alias.shape
                        or actual.dtype != alias.dtype or not np.isfinite(alias).all()
                        or not np.array_equal(actual, alias)
                        or actual.tobytes(order='C') != alias.tobytes(order='C')):
                    raise ValueError('Raw coupling aliases differ from the exact operator')
        memo = {}
        result.raw_upper = tuple(_owned_families(upper, memo))
        result.raw_lower = tuple(_owned_families(lower, memo))
        if len(factor.left) != len(result.sizes):
            raise ValueError('Original left Schur block count differs')
        result.left = []
        for value, size in zip(factor.left, result.sizes):
            block = np.array(value, dtype=np.complex128, copy=True)
            if block.shape != (result.batch, size, size) or not np.isfinite(block).all():
                raise ValueError('Original left Schur block differs')
            block.setflags(write=False)
            result.left.append(block)
        result.left = tuple(result.left)
        result.pivot_residual = float(factor.pivot_residual)
        result.diagonal_residual = float(factor.diagonal_residual)
        result.source_calls = 0
        result.qualification_source_calls = 0
        result.selected_inverse_calls = 0
        result.source_block_products_recorded_total = 0
        result._factor_block_solves = 0
        result._inherited_factor_block_solves = result.batch*len(result.sizes)
        result._conversion_no_new_inversions = True
        result.compacted = True
        return result

    def compact(self):
        """Already compact; retained for the original backend's calling API."""
        return self

    def selected_inverse(self):
        """Return temporary exact diagonal/adjacent inverse blocks in O(N)."""
        n = len(self.sizes)
        diagonal = [None]*n
        upper, lower = [None]*(n-1), [None]*(n-1)
        diagonal[-1] = self.left[-1]
        for i in range(n-2, -1, -1):
            transfer = self.left[i]@self.raw_upper[i]
            opposite = self.raw_lower[i]@self.left[i]
            upper[i] = -transfer@diagonal[i+1]
            lower[i] = -diagonal[i+1]@opposite
            diagonal[i] = self.left[i]-upper[i]@opposite
        self.selected_inverse_calls += 1
        return Bands(diagonal, upper, lower)

    def selected_block(self, row, column):
        """Return one diagonal/adjacent block; O(N), with no inversion."""
        n = len(self.sizes)
        if (isinstance(row, (bool, np.bool_)) or isinstance(column, (bool, np.bool_))
                or not isinstance(row, (int, np.integer))
                or not isinstance(column, (int, np.integer))
                or row < 0 or column < 0 or row >= n or column >= n):
            raise IndexError('Selected block index outside chain')
        if abs(row-column) > 1:
            raise IndexError('Only diagonal and adjacent blocks are selected')
        target = max(row, column)
        green = self.left[-1]
        for i in range(n-2, target-1, -1):
            green = self.left[i]+(self.left[i]@self.raw_upper[i])@green@(self.raw_lower[i]@self.left[i])
        if row == column:
            return green
        if column == row+1:
            return -(self.left[row]@self.raw_upper[row])@green
        return -green@(self.raw_lower[column]@self.left[column])

    def _validate_source(self, source):
        n = len(self.sizes)
        if (not all(hasattr(source, name) for name in ('diagonal', 'upper', 'lower'))
                or len(source.diagonal) != n or len(source.upper) != n-1 or len(source.lower) != n-1):
            raise ValueError('Invalid source chain')
        blocks = list(source.diagonal) + list(source.upper) + list(source.lower)
        expected = [(self.batch, b, b) for b in self.sizes]
        expected += [(self.batch, self.sizes[i], self.sizes[i+1]) for i in range(n-1)]
        expected += [(self.batch, self.sizes[i+1], self.sizes[i]) for i in range(n-1)]
        for block, shape in zip(blocks, expected):
            if not isinstance(block, np.ndarray) or block.shape != shape or not np.isfinite(block).all():
                raise ValueError('Invalid source block')
        for block in source.diagonal:
            if np.max(np.abs(block-adjoint(block))) > 1e-12+1e-10*np.max(np.abs(block)):
                raise ValueError('Non-Hermitian source')
        for up, low in zip(source.upper, source.lower):
            if np.max(np.abs(up-adjoint(low))) > 1e-12+1e-10*max(np.max(np.abs(up)), np.max(np.abs(low))):
                raise ValueError('Source coupling adjoints failed')

    def correlate(self, source, *, qualification=False):
        """Return selected G S G† for signed Hermitian banded S, exactly.

        The forward source band is retained temporarily.  G diagonal and both
        transfer directions are reconstructed in the backward pass.  The
        result owns output bands, which callers must release after contraction.
        """
        if type(qualification) is not bool:
            raise ValueError('qualification must be boolean')
        self._validate_source(source)
        n = len(self.sizes)
        forward = []
        for i, block in enumerate(source.diagonal):
            if i:
                transfer = self.raw_lower[i-1]@self.left[i-1]
                block = (block-transfer@source.upper[i-1]-source.lower[i-1]@adjoint(transfer)
                         +self.raw_lower[i-1]@forward[i-1]@adjoint(self.raw_lower[i-1]))
            forward.append(self.left[i]@block@adjoint(self.left[i]))
        diagonal, upper, lower = [None]*n, [None]*(n-1), [None]*(n-1)
        diagonal[-1] = forward[-1]
        green_next = self.left[-1]
        for i in range(n-2, -1, -1):
            transfer = self.left[i]@self.raw_upper[i]
            opposite = self.raw_lower[i]@self.left[i]
            green_upper = -transfer@green_next
            b = forward[i]@adjoint(self.raw_lower[i])-self.left[i]@source.upper[i]
            e = self.raw_lower[i]@forward[i]-source.lower[i]@adjoint(self.left[i])
            upper[i] = -transfer@diagonal[i+1]-b@adjoint(green_next)
            lower[i] = -green_next@e-diagonal[i+1]@adjoint(transfer)
            diagonal[i] = forward[i]-green_upper@e-upper[i]@adjoint(transfer)
            green_next = self.left[i]-green_upper@opposite
        if not all(np.isfinite(a).all() for a in diagonal+upper+lower):
            raise ValueError('Nonfinite correlation')
        self.source_calls += 1
        self.qualification_source_calls += int(qualification)
        # Count matrix-matrix products in this implementation, including
        # reconstructed transfers and diagonal G; no library scratch is inferred.
        self.source_block_products_recorded_total += self.batch*(21*n-19)
        return Bands(diagonal, upper, lower)

    @property
    def nbytes(self):
        arrays = self.left+self.raw_upper+self.raw_lower
        return sum(a.nbytes for a in {id(a): a for a in arrays}.values())

    def stats(self):
        return dict(
            spatial_blocks=len(self.sizes), frequency_points=self.batch,
            maximum_solve_dimension=max(self.sizes),
            factor_block_solves=self._factor_block_solves,
            inherited_factor_block_solves=self._inherited_factor_block_solves,
            conversion_no_new_inversions=self._conversion_no_new_inversions,
            factor_cubic_work=(0 if self._conversion_no_new_inversions else
                               self.batch*sum(b**3 for b in self.sizes)),
            inherited_factor_cubic_work=(self.batch*sum(b**3 for b in self.sizes)
                                         if self._conversion_no_new_inversions else 0),
            source_action_block_solves=0, compacted=True, g_only=True,
            source_calls=self.source_calls,
            qualification_source_calls=self.qualification_source_calls,
            selected_inverse_calls=self.selected_inverse_calls,
            source_block_products_recorded_total=self.source_block_products_recorded_total,
            source_block_products_per_call=self.batch*(21*len(self.sizes)-19),
            retained_left_blocks=len(self.left),
            retained_coupling_families=len({id(a) for a in self.raw_upper+self.raw_lower}),
            retained_transfer_blocks=0, retained_diagonal_green_blocks=0,
            pivot_residual=self.pivot_residual,
            selected_diagonal_residual=self.diagonal_residual,
            stored_numeric_bytes=self.nbytes,
            source_and_output_scratch_included=False,
            python_runtime_and_library_scratch_excluded=True,
            global_dense_matrix_created=False,
        )
