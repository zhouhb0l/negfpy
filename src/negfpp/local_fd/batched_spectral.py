"""Frequency-tiled exact frozen-kernel current action.

This vectorises ``spectral_backend.FrozenPoint`` over a fixed frequency tile.
Spatial work and storage remain linear in length.  The tile size is a caller
resource choice and does not change the physical frequency grid or equations.
"""

from __future__ import annotations

import numpy as np

from .batched_blocks import Bands, BatchedFactor, adjoint
from .spectral_backend import (
    Bath, FrozenPointError, _hermitian_guard, _matrix, _support_parts,
)


def _batch_matrix(value, shape, name):
    array = np.array(value, dtype=complex, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise FrozenPointError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def _zeros(sizes, batch):
    return Bands(
        [np.zeros((batch, size, size), dtype=complex) for size in sizes],
        [np.zeros((batch, left, right), dtype=complex) for left, right in zip(sizes[:-1], sizes[1:])],
        [np.zeros((batch, right, left), dtype=complex) for left, right in zip(sizes[:-1], sizes[1:])],
    )


def _block(bands, row, column):
    if row == column:
        return bands.diagonal[row]
    if column == row + 1:
        return bands.upper[row]
    if row == column + 1:
        return bands.lower[column]
    raise FrozenPointError("A bath exceeds adjacent grouped block support")


def _scatter(bands, bath, matrix, coefficient):
    if np.ndim(coefficient):
        coefficient = np.asarray(coefficient)[:, None, None]
    for row in bath.parts:
        for column in bath.parts:
            block = _block(bands, row.block, column.block)
            block[:, row.local[:, None], column.local] += coefficient * matrix[:, row.source[:, None], column.source]


def _trace(selected, bath):
    value = np.zeros(bath.gamma.shape[0], dtype=complex)
    for row in bath.parts:
        for column in bath.parts:
            other = _block(selected, column.block, row.block)
            gamma = bath.gamma[:, row.source[:, None], column.source]
            other = other[:, column.local[:, None], row.local]
            value += np.einsum("fij,fji->f", gamma, other)
    return value


def _extract(selected, bath):
    result = np.empty_like(bath.gamma)
    for row in bath.parts:
        for column in bath.parts:
            block = _block(selected, row.block, column.block)
            result[:, row.source[:, None], column.source] = block[:, row.local[:, None], column.local]
    return result


def _per_frequency_max(blocks, batch):
    result = np.zeros(batch)
    for block in blocks:
        result = np.maximum(result, np.max(np.abs(block), axis=(-2, -1)))
    return result


def _norm_bounds(bands):
    rows = [np.sum(np.abs(block), axis=2) for block in bands.diagonal]
    cols = [np.sum(np.abs(block), axis=1) for block in bands.diagonal]
    for i, (up, low) in enumerate(zip(bands.upper, bands.lower)):
        rows[i] += np.sum(np.abs(up), axis=2)
        rows[i + 1] += np.sum(np.abs(low), axis=2)
        cols[i + 1] += np.sum(np.abs(up), axis=1)
        cols[i] += np.sum(np.abs(low), axis=1)
    one = np.max(np.stack([np.max(c, axis=1) for c in cols]), axis=0)
    infinity = np.max(np.stack([np.max(r, axis=1) for r in rows]), axis=0)
    return np.sqrt(one * infinity), one, infinity


class BatchedFrozenPoint:
    """Exact current/Jv action for one q and a bounded frequency tile."""

    def __init__(self, *, harmonic, sigma_short, active_dofs, contacts,
                 center_layers, frequencies, rtol=1e-8, atol=1e-11, compact=False):
        if type(compact) is not bool:
            raise FrozenPointError("compact must be a boolean cache-representation choice")
        self.compacted=False
        if (isinstance(center_layers, (bool, np.bool_))
                or not isinstance(center_layers, (int, np.integer)) or center_layers < 1):
            raise FrozenPointError("center_layers must be a positive integer")
        self.center_layers = int(center_layers)
        self.original_cells = self.center_layers + 2
        self.full_dimension = 24 * self.original_cells
        self.nterminals = self.center_layers + 2
        self.frequencies = np.array(frequencies, dtype=float, copy=True)
        if (self.frequencies.ndim != 1 or not len(self.frequencies)
                or not np.isfinite(self.frequencies).all() or np.any(self.frequencies <= 0)):
            raise FrozenPointError("frequencies must be finite positive nodes")
        self.frequencies.setflags(write=False)
        self.batch = len(self.frequencies)
        self.rtol, self.atol = float(rtol), float(atol)
        if (not np.isfinite(self.rtol) or not np.isfinite(self.atol)
                or self.rtol < 0 or self.atol < 0):
            raise FrozenPointError("raw guard tolerances must be finite and nonnegative")
        if len(harmonic) != 3 or len(contacts) != 2:
            raise FrozenPointError("harmonic needs D00,D01,D10 and contacts needs left,right")
        self.harmonic = tuple(_matrix(block, (24, 24), f"harmonic[{i}]")
                              for i, block in enumerate(harmonic))
        self.sigma_short = _batch_matrix(sigma_short, (self.batch, 36, 36), "sigma_short")
        self.contacts = tuple(_batch_matrix(block, (self.batch, 24, 24), f"contact[{i}]")
                              for i, block in enumerate(contacts))
        active = np.asarray(active_dofs)
        if (active.shape != (36,) or active.dtype.kind not in "iu"
                or len(np.unique(active)) != 36 or np.any(active < 0) or np.any(active >= 72)):
            raise FrozenPointError("active_dofs must be 36 unique integer coordinates in the original window")
        self.active_dofs = np.array(active, dtype=np.int64, copy=True)
        self.active_dofs.setflags(write=False)
        _hermitian_guard(self.harmonic[0], "harmonic D00", 1e-10, 1e-12)
        hscale = max(float(np.max(np.abs(block))) for block in self.harmonic[1:])
        if np.max(np.abs(self.harmonic[1] - self.harmonic[2].conj().T)) > 1e-12 + 1e-10 * hscale:
            raise FrozenPointError("harmonic D01/D10 are not adjoints; no repair is applied")
        gamma_contact = tuple(1j * (block - adjoint(block)) for block in self.contacts)
        gamma_probe = 1j * (self.sigma_short - adjoint(self.sigma_short))
        passivity = []
        for gamma, name in zip(gamma_contact + (gamma_probe,), ("left Gamma", "right Gamma", "probe Gamma")):
            magnitude = np.max(np.abs(gamma), axis=(-2, -1))
            hdefect = np.max(np.abs(gamma - adjoint(gamma)), axis=(-2, -1))
            if np.any(hdefect > self.atol + self.rtol * magnitude):
                raise FrozenPointError(f"{name} is not Hermitian; no repair is applied")
            eigenvalues = np.linalg.eigvalsh(gamma)
            scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), magnitude)
            minimum = np.min(eigenvalues, axis=1)
            if np.any(minimum < -(self.atol + self.rtol * scale)):
                raise FrozenPointError(f"{name} is not passive; no clipping is applied")
            passivity.append(dict(hermiticity_max_abs=float(hdefect.max()),
                                  minimum_eigenvalue=float(minimum.min()),
                                  passivity_scale=float(scale.max())))
            gamma.setflags(write=False)
        supports = [np.arange(24), np.arange(self.full_dimension - 24, self.full_dimension)]
        supports += [self.active_dofs + 24 * n for n in range(self.center_layers)]
        self.baths = tuple(Bath(gamma, support, _support_parts(support)) for gamma, support in
                           zip(list(gamma_contact) + [gamma_probe] * self.center_layers, supports))
        for bath in self.baths:
            bath.support.setflags(write=False)
        self.block_sizes = tuple(min(48, self.full_dimension - 48 * i)
                                 for i in range((self.original_cells + 1) // 2))
        operator = _zeros(self.block_sizes, self.batch)
        d00, d01, d10 = self.harmonic
        for cell in range(self.original_cells):
            group, local = divmod(cell, 2)
            sl = slice(24 * local, 24 * (local + 1))
            operator.diagonal[group][:, sl, sl] = self.frequencies[:, None, None] ** 2 * np.eye(24) - d00
            if cell + 1 < self.original_cells:
                next_group, next_local = divmod(cell + 1, 2)
                sn = slice(24 * next_local, 24 * (next_local + 1))
                _block(operator, group, next_group)[:, sl, sn] = -d01
                _block(operator, next_group, group)[:, sn, sl] = -d10
        for bath, retarded in zip(self.baths, list(self.contacts) + [self.sigma_short] * self.center_layers):
            _scatter(operator, bath, retarded, -1.)
        self.factor = BatchedFactor(operator.diagonal, operator.upper, operator.lower)
        self.green = self.factor.green
        self.spectral = Bands(
            [1j * (block - adjoint(block)) for block in self.green.diagonal],
            [1j * (up - adjoint(low)) for up, low in zip(self.green.upper, self.green.lower)],
            [1j * (low - adjoint(up)) for up, low in zip(self.green.upper, self.green.lower)],
        )
        trace = np.stack([_trace(self.spectral, bath) for bath in self.baths], axis=1)
        self._real_guard(trace, "bath spectral traces")
        self.spectral_traces = trace.real.copy()
        scale = np.maximum(np.max(np.abs(self.spectral_traces), axis=1), 1.)
        if np.any(np.min(self.spectral_traces, axis=1) < -(self.atol + self.rtol * scale)):
            raise FrozenPointError("negative bath spectral trace; no clipping is applied")
        self.spectral_traces.setflags(write=False)
        self.current_calls = 0
        self.diagnostics = self._qualify(passivity)
        if compact:self.compact()

    def compact(self):
        """Drop redundant blocks only after all original point guards pass."""
        if self.compacted:return self
        if getattr(self,"_compact_qualification_failed",False):
            raise FrozenPointError("Earlier compact source qualification failed")
        if not hasattr(self,"diagnostics"):
            raise FrozenPointError("Compaction requires completed raw point qualification")
        self.factor.compact()
        self.green=None
        try:
            total=self._sources(np.ones((self.batch,self.nterminals)))
            propagated=self.factor.correlate(total,qualification=True)
            spectral_blocks=self.spectral.diagonal+self.spectral.upper+self.spectral.lower
            propagated_blocks=propagated.diagonal+propagated.upper+propagated.lower
            scale=np.maximum(_per_frequency_max(spectral_blocks,self.batch),
                             _per_frequency_max(propagated_blocks,self.batch))
            error=_per_frequency_max([s-p for s,p in zip(spectral_blocks,propagated_blocks)],self.batch)
            allowance=self.atol+self.rtol*scale
            if np.any(error>allowance):
                raise FrozenPointError("Compact full-source optical identity fails; no repair is applied")
            uniform=np.stack([_trace(propagated,bath) for bath in self.baths],axis=1)
            self._real_guard(uniform,"compact uniform source contractions")
            uerror=np.max(np.abs(self.spectral_traces-uniform.real),axis=1)
            uallow=self.atol+self.rtol*np.maximum(np.max(np.abs(self.spectral_traces),axis=1),
                                                np.max(np.abs(uniform.real),axis=1))
            if np.any(uerror>uallow):
                raise FrozenPointError("Compact uniform-temperature raw current identity fails")
        except BaseException:
            self._compact_qualification_failed=True
            raise
        compact_diagnostics=dict(
            compact_optical_max_abs=error,
            compact_optical_scale=scale,
            compact_optical_guard_ratio=np.divide(error,allowance,out=np.zeros_like(error),where=allowance>0),
            compact_uniform_current_max_abs=uerror,
            compact_uniform_current_guard_ratio=np.divide(uerror,uallow,out=np.zeros_like(uerror),where=uallow>0),
        )
        self.frequency_diagnostics.update(compact_diagnostics)
        self.diagnostics.update({name:float(np.max(values)) for name,values in compact_diagnostics.items()})
        self.diagnostics['startup_compact_source_correlations']=1
        self.spectral=None
        self.compacted=True
        return self

    @property
    def supports(self):
        return tuple(bath.support for bath in self.baths)

    @property
    def gamma_blocks(self):
        return tuple(bath.gamma for bath in self.baths)

    def _real_guard(self, values, name):
        if (not np.isfinite(values).all()
                or np.any(np.abs(values.imag) > self.atol + self.rtol * np.abs(values.real))):
            raise FrozenPointError(f"{name} has a nonfinite or material imaginary residue")

    def _sources(self, weights):
        source = _zeros(self.block_sizes, self.batch)
        for bath, weight in zip(self.baths, weights.T):
            _scatter(source, bath, bath.gamma, weight)
        return source

    def _qualify(self, passivity):
        total = self._sources(np.ones((self.batch, self.nterminals)))
        propagated = self.factor.correlate(total)
        spectral_blocks = self.spectral.diagonal + self.spectral.upper + self.spectral.lower
        propagated_blocks = propagated.diagonal + propagated.upper + propagated.lower
        scale = np.maximum(_per_frequency_max(spectral_blocks, self.batch),
                           _per_frequency_max(propagated_blocks, self.batch))
        error = _per_frequency_max([s - p for s, p in zip(spectral_blocks, propagated_blocks)], self.batch)
        allowance = self.atol + self.rtol * scale
        if np.any(error > allowance):
            raise FrozenPointError("selected full-source optical identity fails; no source repair is applied")
        a = self.factor.a
        defect = Bands(
            [1j * (adjoint(a0) - a0) - s for a0, s in zip(a.diagonal, total.diagonal)],
            [1j * (adjoint(low) - up) - s for up, low, s in zip(a.upper, a.lower, total.upper)],
            [1j * (adjoint(up) - low) - s for up, low, s in zip(a.upper, a.lower, total.lower)],
        )
        defect_bound, one, infinity = _norm_bounds(defect)
        unit = _zeros(self.block_sizes, self.batch)
        for block in unit.diagonal:
            block[:] = np.eye(block.shape[-1])
        gg = self.factor.correlate(unit)
        row_norm_squared = np.max(np.stack([np.max(np.diagonal(block, axis1=-2, axis2=-1).real, axis=1)
                                           for block in gg.diagonal]), axis=0)
        if not np.isfinite(row_norm_squared).all() or np.any(row_norm_squared < 0):
            raise FrozenPointError("GG† row norm bound is invalid")
        global_bound = defect_bound * row_norm_squared
        if np.any(global_bound > allowance):
            raise FrozenPointError("global optical source-defect amplification bound fails")
        uniform = np.stack([_trace(propagated, bath) for bath in self.baths], axis=1)
        self._real_guard(uniform, "uniform full-source contractions")
        uerror = np.max(np.abs(self.spectral_traces - uniform.real), axis=1)
        uallow = self.atol + self.rtol * np.maximum(np.max(np.abs(self.spectral_traces), axis=1),
                                                  np.max(np.abs(uniform.real), axis=1))
        if np.any(uerror > uallow):
            raise FrozenPointError("uniform-temperature raw current identity fails")
        self.frequency_diagnostics = dict(
            selected_optical_max_abs=error,
            selected_optical_scale=scale,
            selected_optical_guard_ratio=np.divide(error, allowance, out=np.zeros_like(error), where=allowance > 0),
            uniform_current_max_abs=uerror,
            uniform_current_guard_ratio=np.divide(uerror, uallow, out=np.zeros_like(uerror), where=uallow > 0),
            global_source_defect_norm_bound=defect_bound,
            global_source_defect_norm_one=one,
            global_source_defect_norm_infinity=infinity,
            maximum_full_green_row_norm_squared=row_norm_squared,
            global_optical_source_defect_entry_bound=global_bound,
            global_optical_source_defect_guard_ratio=np.divide(global_bound, allowance, out=np.zeros_like(global_bound), where=allowance > 0),
        )
        return dict(
            **{name: float(np.max(values)) for name, values in self.frequency_diagnostics.items()},
            source_bound_is_numerical_forward_error_certificate=False,
            maximum_local_inverse_residual=max(self.factor.pivot_residual, self.factor.diagonal_residual),
            passivity=passivity, startup_source_correlations=2,
        )

    def selected_on_bath(self, terminal, *, spectral=False):
        if not isinstance(terminal, (int, np.integer)) or not 0 <= terminal < self.nterminals:
            raise FrozenPointError("terminal index out of range")
        bath=self.baths[terminal]
        if not self.compacted:
            return _extract(self.spectral if spectral else self.green,bath)
        result=np.empty_like(bath.gamma)
        for row in bath.parts:
            for column in bath.parts:
                block=self.factor.selected_block(row.block,column.block)
                if spectral:
                    block=1j*(block-adjoint(self.factor.selected_block(column.block,row.block)))
                result[:,row.source[:,None],column.source]=block[:,row.local[:,None],column.local]
        return result

    def currents(self, energy_weights):
        """Return (frequency,bath) reduced currents for arbitrary real weights."""
        if getattr(self,"_compact_qualification_failed",False):
            raise FrozenPointError("Compact source qualification failed; current actions are forbidden")
        raw = np.asarray(energy_weights)
        if np.iscomplexobj(raw) and np.any(raw.imag != 0):
            raise FrozenPointError("energy_weights must be real; no imaginary part is discarded")
        weights = np.asarray(raw.real, dtype=float)
        if weights.shape != (self.batch, self.nterminals) or not np.isfinite(weights).all():
            raise FrozenPointError("energy_weights must have finite real shape (frequency,bath)")
        centered = weights - weights[:, :1]
        source = self._sources(centered)
        propagated = self.factor.correlate(source)
        values = centered * self.spectral_traces - np.stack([_trace(propagated, bath) for bath in self.baths], axis=1)
        self._real_guard(values, "terminal current action")
        self.current_calls += 1
        return values.real

    def stats(self):
        arrays = self.harmonic + self.contacts + (self.sigma_short,) + self.gamma_blocks
        numeric = sum(a.nbytes for a in {id(a): a for a in arrays}.values())
        metadata = sum(bath.support.nbytes + sum(p.local.nbytes + p.source.nbytes for p in bath.parts)
                       for bath in self.baths)
        return dict(
            original_cells=self.original_cells, center_layers=self.center_layers,
            full_dimension=self.full_dimension, nterminals=self.nterminals,
            frequency_points=self.batch, block_sizes=list(self.block_sizes),
            maximum_block_size=max(self.block_sizes), current_calls=self.current_calls,
            compacted=self.compacted,
            factor=self.factor.stats(),
            per_current_block_products=self.factor.stats()['source_block_products_per_call'],
            stored_numeric_bytes=self.factor.nbytes + (self.spectral.nbytes if self.spectral is not None else 0) + numeric
                + metadata + self.spectral_traces.nbytes + self.active_dofs.nbytes
                + self.frequencies.nbytes + sum(a.nbytes for a in self.frequency_diagnostics.values()),
            all_pairs_transmissions_created=False, global_dense_matrix_created=False,
        )
