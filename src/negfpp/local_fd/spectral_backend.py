"""Exact linear-work frozen-kernel currents at one (q, positive omega).

Original 24-coordinate cells are grouped in pairs.  A translated complete
36-coordinate endpoint support spans three original cells and therefore at
most two adjacent grouped blocks.  Neither its retarded field nor its noise
is truncated.  The last grouped block may contain only 24 coordinates.

Only selected Green/correlation blocks are retained.  A current evaluation
forms the total weighted bath source, propagates it once with the reused
retarded factors, and contracts each local bath.  There is no all-pairs
transmission matrix, and the energy weights may have either sign for an exact
Jacobian-vector action.  No Bose function, quadrature, or physical-unit
conversion is hidden in this pointwise module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .recursive_blocks import SelectedBlocks, factor_block_tridiagonal


class FrozenPointError(ValueError):
    """The point cannot satisfy the unchanged raw physical guards."""


def _matrix(value, shape, name):
    array = np.array(value, dtype=np.complex128, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise FrozenPointError(f"{name} must be a finite matrix of shape {shape}")
    array.setflags(write=False)
    return array


def _max_abs(blocks):
    return max((float(np.max(np.abs(block))) for block in blocks), default=0.)


def _hermitian_guard(matrix, name, rtol, atol):
    magnitude = float(np.max(np.abs(matrix)))
    defect = float(np.max(np.abs(matrix - matrix.conj().T)))
    if defect > atol + rtol * magnitude:
        raise FrozenPointError(f"{name} is not Hermitian; no repair is applied")
    return defect


def _passivity_guard(matrix, name, rtol, atol):
    hermiticity = _hermitian_guard(matrix, name, rtol, atol)
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), float(np.max(np.abs(matrix))))
    minimum = float(eigenvalues.min())
    if minimum < -(atol + rtol * scale):
        raise FrozenPointError(f"{name} is not passive; no clipping is applied")
    return dict(hermiticity_max_abs=hermiticity, minimum_eigenvalue=minimum,
                passivity_scale=scale)


@dataclass(frozen=True)
class SupportPart:
    block: int
    local: np.ndarray
    source: np.ndarray


@dataclass(frozen=True)
class Bath:
    gamma: np.ndarray
    support: np.ndarray
    parts: tuple[SupportPart, ...]


def _support_parts(support):
    grouped = support // 48
    parts = []
    for block in np.unique(grouped):
        positions = np.flatnonzero(grouped == block)
        local = support[positions] - 48 * block
        positions.setflags(write=False)
        local.setflags(write=False)
        parts.append(SupportPart(int(block), local, positions))
    if len(parts) > 2 or (len(parts) == 2 and parts[1].block != parts[0].block + 1):
        raise FrozenPointError("A bath support must fit adjacent grouped blocks")
    return tuple(parts)


def _empty_bands(sizes):
    return (
        [np.zeros((size, size), dtype=complex) for size in sizes],
        [np.zeros((left, right), dtype=complex) for left, right in zip(sizes[:-1], sizes[1:])],
        [np.zeros((right, left), dtype=complex) for left, right in zip(sizes[:-1], sizes[1:])],
    )


def _band_block(bands, row, column):
    diagonal, upper, lower = bands
    if row == column:
        return diagonal[row]
    if column == row + 1:
        return upper[row]
    if row == column + 1:
        return lower[column]
    raise FrozenPointError("An unrepresented longer-range block was requested")


def _scatter(bands, bath, matrix, coefficient=1.):
    for row in bath.parts:
        for column in bath.parts:
            block = _band_block(bands, row.block, column.block)
            block[np.ix_(row.local, column.local)] += coefficient * matrix[np.ix_(row.source, column.source)]


def _extract(selected, bath):
    result = np.empty(bath.gamma.shape, dtype=complex)
    for row in bath.parts:
        for column in bath.parts:
            block = selected.block(row.block, column.block)
            result[np.ix_(row.source, column.source)] = block[np.ix_(row.local, column.local)]
    return result


def _trace_bath(selected, bath):
    # Ordered cross terms are both retained.  This avoids a temporary full
    # device matrix and needs at most four fixed-size contractions per bath.
    value = 0.j
    for row in bath.parts:
        for column in bath.parts:
            block = selected.block(column.block, row.block)
            gamma = bath.gamma[np.ix_(row.source, column.source)]
            other = block[np.ix_(column.local, row.local)]
            value += np.einsum("ij,ji->", gamma, other)
    return complex(value)


def _spectral_selected(green):
    return SelectedBlocks(
        tuple(1j * (block - block.conj().T) for block in green.diagonal),
        tuple(1j * (up - low.conj().T) for up, low in zip(green.upper, green.lower)),
        tuple(1j * (low - up.conj().T) for up, low in zip(green.upper, green.lower)),
        green.stats,
    )


def _block_norm_bound(bands):
    """sqrt(||B||1 ||B||inf), from banded row/column absolute sums only."""
    diagonal, upper, lower = bands
    rows = [np.sum(np.abs(block), axis=1) for block in diagonal]
    cols = [np.sum(np.abs(block), axis=0) for block in diagonal]
    for i, (up, low) in enumerate(zip(upper, lower)):
        rows[i] += np.sum(np.abs(up), axis=1)
        rows[i + 1] += np.sum(np.abs(low), axis=1)
        cols[i + 1] += np.sum(np.abs(up), axis=0)
        cols[i] += np.sum(np.abs(low), axis=0)
    one = max(float(np.max(value)) for value in cols)
    infinity = max(float(np.max(value)) for value in rows)
    return float(np.sqrt(one * infinity)), one, infinity


class FrozenPoint:
    """One fixed q/frequency propagator and its exact local current action."""

    def __init__(self, *, harmonic, sigma_short, active_dofs, contacts,
                 center_layers, omega, rtol=1e-8, atol=1e-11):
        if (isinstance(center_layers, (bool, np.bool_))
                or not isinstance(center_layers, (int, np.integer)) or center_layers < 1):
            raise FrozenPointError("center_layers must be a positive integer")
        self.center_layers = int(center_layers)
        self.original_cells = self.center_layers + 2
        self.full_dimension = 24 * self.original_cells
        self.nterminals = self.center_layers + 2
        self.omega = float(omega)
        self.rtol, self.atol = float(rtol), float(atol)
        if (not np.isfinite(self.omega) or self.omega <= 0
                or not np.isfinite(self.rtol) or not np.isfinite(self.atol)
                or self.rtol < 0 or self.atol < 0):
            raise FrozenPointError("omega must be positive; raw guard tolerances must be finite and nonnegative")
        if len(harmonic) != 3 or len(contacts) != 2:
            raise FrozenPointError("harmonic needs D00,D01,D10 and contacts needs left,right")
        self.harmonic = tuple(_matrix(block, (24, 24), f"harmonic[{i}]")
                              for i, block in enumerate(harmonic))
        self.sigma_short = _matrix(sigma_short, (36, 36), "sigma_short")
        self.contacts = tuple(_matrix(block, (24, 24), f"contact[{i}]")
                              for i, block in enumerate(contacts))
        active = np.asarray(active_dofs)
        if (active.shape != (36,) or active.dtype.kind not in "iu"
                or len(np.unique(active)) != 36 or np.any(active < 0) or np.any(active >= 72)):
            raise FrozenPointError("active_dofs must be 36 unique integer coordinates in the original 72-coordinate window")
        self.active_dofs = np.array(active, dtype=np.int64, copy=True)
        self.active_dofs.setflags(write=False)
        _hermitian_guard(self.harmonic[0], "harmonic D00", 1e-10, 1e-12)
        defect = float(np.max(np.abs(self.harmonic[1] - self.harmonic[2].conj().T)))
        if defect > 1e-12 + 1e-10 * max(_max_abs(self.harmonic[1:]), 0.):
            raise FrozenPointError("harmonic D01/D10 are not adjoints; no repair is applied")
        gamma_contact = tuple(1j * (block - block.conj().T) for block in self.contacts)
        gamma_probe = 1j * (self.sigma_short - self.sigma_short.conj().T)
        passivity = [_passivity_guard(block, name, self.rtol, self.atol)
                     for block, name in zip(gamma_contact + (gamma_probe,),
                                             ("left Gamma", "right Gamma", "probe Gamma"))]
        for block in gamma_contact + (gamma_probe,):
            block.setflags(write=False)
        supports = [np.arange(24), np.arange(self.full_dimension - 24, self.full_dimension)]
        supports += [self.active_dofs + 24 * n for n in range(self.center_layers)]
        gammas = list(gamma_contact) + [gamma_probe] * self.center_layers
        self.baths = tuple(Bath(gamma, support, _support_parts(support))
                           for gamma, support in zip(gammas, supports))
        for bath in self.baths:
            bath.support.setflags(write=False)
        self.block_sizes = tuple(min(48, self.full_dimension - 48 * i)
                                 for i in range((self.original_cells + 1) // 2))
        operator = _empty_bands(self.block_sizes)
        d00, d01, d10 = self.harmonic
        for cell in range(self.original_cells):
            group, start = divmod(cell, 2)
            sl = slice(24 * start, 24 * (start + 1))
            operator[0][group][sl, sl] = self.omega ** 2 * np.eye(24) - d00
            if cell + 1 < self.original_cells:
                next_group, next_start = divmod(cell + 1, 2)
                slnext = slice(24 * next_start, 24 * (next_start + 1))
                _band_block(operator, group, next_group)[sl, slnext] = -d01
                _band_block(operator, next_group, group)[slnext, sl] = -d10
        for bath, retarded in zip(self.baths, list(self.contacts) + [self.sigma_short] * self.center_layers):
            _scatter(operator, bath, retarded, -1.)
        self.factor = factor_block_tridiagonal(*operator)
        self.green = self.factor.selected_inverse()
        self.spectral = _spectral_selected(self.green)
        self.spectral_traces = np.array([_trace_bath(self.spectral, bath)
                                        for bath in self.baths])
        self._real_guard(self.spectral_traces, "bath spectral traces")
        self.spectral_traces = self.spectral_traces.real.copy()
        scale = max(float(np.max(np.abs(self.spectral_traces))), 1.)
        if self.spectral_traces.min() < -(self.atol + self.rtol * scale):
            raise FrozenPointError("negative bath spectral trace; no clipping is applied")
        self.spectral_traces.setflags(write=False)
        self.current_calls = 0
        self.current_work_macs = 0
        self.last_current_stats = None
        self.diagnostics = self._qualify(passivity)

    @property
    def supports(self):
        return tuple(bath.support for bath in self.baths)

    @property
    def gamma_blocks(self):
        return tuple(bath.gamma for bath in self.baths)

    def _real_guard(self, values, name):
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise FrozenPointError(f"{name} is nonfinite")
        allowance = self.atol + self.rtol * np.abs(values.real)
        if np.any(np.abs(values.imag) > allowance):
            raise FrozenPointError(f"{name} has a material imaginary residue")

    def _source_bands(self, weights):
        source = _empty_bands(self.block_sizes)
        for weight, bath in zip(weights, self.baths):
            _scatter(source, bath, bath.gamma, weight)
        return source

    def _qualify(self, passivity):
        total = self._source_bands(np.ones(self.nterminals))
        propagated = self.factor.correlate(*total, hermitian_rtol=self.rtol, hermitian_atol=self.atol)
        spectral_blocks = self.spectral.diagonal + self.spectral.upper + self.spectral.lower
        propagated_blocks = propagated.diagonal + propagated.upper + propagated.lower
        scale = max(_max_abs(spectral_blocks), _max_abs(propagated_blocks))
        selected_error = max(float(np.max(np.abs(left - right)))
                             for left, right in zip(spectral_blocks, propagated_blocks))
        optical_allowance = self.atol + self.rtol * scale
        if selected_error > optical_allowance:
            raise FrozenPointError("selected full-source optical identity fails; no source repair is applied")
        raw_a = (self.factor.diagonal, self.factor.upper, self.factor.lower)
        defect = (
            [1j * (a.conj().T - a) - s for a, s in zip(raw_a[0], total[0])],
            [1j * (low.conj().T - up) - s for up, low, s in zip(raw_a[1], raw_a[2], total[1])],
            [1j * (up.conj().T - low) - s for up, low, s in zip(raw_a[1], raw_a[2], total[2])],
        )
        defect_bound, defect_one, defect_infinity = _block_norm_bound(defect)
        # diag(GG†) gives squared row norms of the entire dense G.  Hence for
        # EVERY entry |(G E G†)_ij| <= ||E||2 max_i (GG†)_ii.  This bounds
        # physical source-substitution error for the exact inverse of raw A.
        # It does not certify floating-point forward error of the recursion.
        unit = [np.eye(size, dtype=complex) for size in self.block_sizes]
        gg = self.factor.correlate(unit)
        row_norm_squared = max(float(np.max(np.real(np.diag(block)))) for block in gg.diagonal)
        if row_norm_squared < 0 or not np.isfinite(row_norm_squared):
            raise FrozenPointError("GG† row norm bound is invalid")
        global_source_bound = defect_bound * row_norm_squared
        if global_source_bound > optical_allowance:
            raise FrozenPointError("global optical source-defect amplification bound fails")
        uniform = np.array([_trace_bath(propagated, bath) for bath in self.baths])
        self._real_guard(uniform, "uniform full-source contractions")
        uniform_defect = float(np.max(np.abs(self.spectral_traces - uniform.real)))
        uniform_allowance = self.atol + self.rtol * max(float(np.max(np.abs(self.spectral_traces))),
                                                       float(np.max(np.abs(uniform.real))))
        if uniform_defect > uniform_allowance:
            raise FrozenPointError("uniform-temperature raw current identity fails")
        inverse_residual = max(self.factor.factor_stats.maximum_pivot_inverse_residual,
                               self.green.stats.maximum_selected_diagonal_residual)
        if inverse_residual > self.atol + self.rtol:
            raise FrozenPointError("block inverse residual fails")
        return dict(
            selected_optical_max_abs=selected_error,
            selected_optical_scale=scale,
            selected_optical_guard_ratio=selected_error / optical_allowance if optical_allowance else 0.,
            uniform_current_max_abs=uniform_defect,
            uniform_current_guard_ratio=uniform_defect / uniform_allowance if uniform_allowance else 0.,
            global_source_defect_norm_bound=defect_bound,
            global_source_defect_norm_one=defect_one,
            global_source_defect_norm_infinity=defect_infinity,
            maximum_full_green_row_norm_squared=row_norm_squared,
            global_optical_source_defect_entry_bound=global_source_bound,
            global_optical_source_defect_guard_ratio=global_source_bound / optical_allowance if optical_allowance else 0.,
            source_bound_is_numerical_forward_error_certificate=False,
            maximum_local_inverse_residual=inverse_residual,
            passivity=passivity,
            startup_source_correlations=2,
        )

    def selected_on_bath(self, terminal, *, spectral=False):
        """Fixed-size selected support block for independent diagnostics."""
        if not isinstance(terminal, (int, np.integer)) or not 0 <= terminal < self.nterminals:
            raise FrozenPointError("terminal index out of range")
        return _extract(self.spectral if spectral else self.green, self.baths[terminal])

    def currents(self, energy_weights):
        """Reduced currents into the device, or an exact signed Jv action.

        Subtracting the first bath's weight uses the qualified uniform-source
        identity.  It avoids cancellation between large equilibrium fluxes.
        The numerical identity defect is reported at construction, not repaired.
        """
        raw = np.asarray(energy_weights)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise FrozenPointError("energy_weights must be real; no imaginary part is discarded")
        weights = np.asarray(raw.real, dtype=float)
        if weights.shape != (self.nterminals,) or not np.isfinite(weights).all():
            raise FrozenPointError("energy_weights must be one finite real weight per bath")
        centered = weights - weights[0]
        source = self._source_bands(centered)
        propagated = self.factor.correlate(*source, hermitian_rtol=self.rtol, hermitian_atol=self.atol)
        values = centered * self.spectral_traces - np.array([_trace_bath(propagated, bath)
                                                            for bath in self.baths])
        self._real_guard(values, "terminal current action")
        self.current_calls += 1
        self.current_work_macs += propagated.stats.multiply_accumulates
        self.last_current_stats = propagated.stats
        return values.real

    def stats(self):
        """Counts and owned numeric payload; no arrays enter this summary."""
        bath_metadata = sum(bath.support.nbytes + sum(part.local.nbytes + part.source.nbytes
                                                     for part in bath.parts) for bath in self.baths)
        unique = {id(a): a for a in self.harmonic + self.contacts + (self.sigma_short,)
                  + self.gamma_blocks}
        input_bytes = sum(array.nbytes for array in unique.values())
        spectral_bytes = self.spectral.stored_nbytes
        return dict(
            original_cells=self.original_cells, center_layers=self.center_layers,
            full_dimension=self.full_dimension, nterminals=self.nterminals,
            block_sizes=list(self.block_sizes), maximum_block_size=max(self.block_sizes),
            factor=self.factor.factor_stats.as_dict(),
            selected_inverse=self.green.stats.as_dict(),
            per_current_block_products=16 * len(self.block_sizes) - 14,
            current_calls=self.current_calls, current_work_multiply_accumulates=self.current_work_macs,
            stored_numeric_bytes=self.factor.stored_nbytes + spectral_bytes + input_bytes
                + bath_metadata + self.spectral_traces.nbytes + self.active_dofs.nbytes,
            all_pairs_transmissions_created=False, global_dense_matrix_created=False,
        )
