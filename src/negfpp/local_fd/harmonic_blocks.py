"""Exact repeated FC2 blocks and a linear-work harmonic stability audit.

The existing material adapter constructs the same onsite/coupling blocks at
every layer, including the two ends.  It rejects source IFC images outside
the declared nearest-neighbour principal layer.  Extracting these blocks
from a fixed three-layer provider therefore does not truncate a long device.

No file reads, material construction or calculation occurs on import.  The
stability shift below belongs only to an auxiliary certificate; no propagated
harmonic block or Green-function operator is shifted or repaired.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _adjoint(value):
    return np.swapaxes(value.conj(), -1, -2)


def _matrix(value, name):
    raw = np.asarray(value)
    if (raw.ndim != 2 or not raw.shape[0] or raw.shape[0] != raw.shape[1]
            or raw.dtype.kind not in "fci" or not np.isfinite(raw).all()):
        raise ValueError(name + " must be a finite square numeric matrix")
    return np.asarray(raw, dtype=np.complex128)


def _readonly(value):
    result = np.array(value, dtype=np.complex128, copy=True)
    result.setflags(write=False)
    return result


def _check_equal(actual, expected, name, *, rtol=1e-10, atol=1e-12):
    if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=rtol, atol=atol):
        raise ValueError(name + " differs from the exact repeated FC2 construction")
    return float(np.max(np.abs(actual-expected)))


@dataclass(frozen=True)
class RepeatedHarmonicBlocks:
    """One unchanged onsite/forward block per transverse q.

    ``source_reverse`` is retained for audit.  The finite Device1D convention
    uses ``forward.conj().T`` below the diagonal, after checking agreement
    with this independently constructed source reverse block.
    """
    onsite: np.ndarray
    forward: np.ndarray
    source_reverse: np.ndarray
    audit: dict

    @property
    def n_qpoints(self):
        return self.onsite.shape[0]

    @property
    def layer_ndof(self):
        return self.onsite.shape[1]


def extract_repeated_harmonic(device_provider, lead_provider, kpoints,
                              negative_q_indices, *, layer_ndof=24):
    """Check a fixed three-layer provider, then own only O(Nq*b²) blocks.

    This function expects the providers returned by
    ``material_transverse_harmonic(..., n_device_layers=3)``.  It deliberately
    rejects nonperiodic ends, mass disorder, reversed coupling conventions and
    incorrectly routed transverse reality, rather than inferring a bulk fit.
    """
    points = tuple(tuple(float(x) for x in point) for point in kpoints)
    negative = np.asarray(negative_q_indices)
    nq = len(points)
    if (not nq or negative.shape != (nq,) or negative.dtype.kind not in "iu"
            or set(negative.tolist()) != set(range(nq))
            or not np.array_equal(negative[negative], np.arange(nq))):
        raise ValueError("Supply the actual involutive negative-q routing")
    if (isinstance(layer_ndof, (bool, np.bool_))
            or not isinstance(layer_ndof, (int, np.integer)) or layer_ndof < 1):
        raise ValueError("layer_ndof must be a positive integer")
    onsite, forward, reverse = [], [], []
    maximum_repeat = maximum_reverse = maximum_hermiticity = 0.
    for point in points:
        raw00, raw01, raw10 = lead_provider.blocks(point)
        d00 = _matrix(raw00, "lead onsite")
        d01 = _matrix(raw01, "lead forward")
        d10 = _adjoint(d01) if raw10 is None else _matrix(raw10, "lead reverse")
        expected_shape = (int(layer_ndof), int(layer_ndof))
        if any(matrix.shape != expected_shape for matrix in (d00, d01, d10)):
            raise ValueError("FC2 block size differs from the admitted principal layer")
        maximum_hermiticity = max(maximum_hermiticity,
            _check_equal(d00, _adjoint(d00), "onsite Hermiticity"))
        maximum_reverse = max(maximum_reverse,
            _check_equal(d10, _adjoint(d01), "independent reverse coupling"))
        device = device_provider.device(point)
        if device.n_layers != 3 or device.dof_per_layer != layer_ndof:
            raise ValueError("The extraction oracle must have exactly three source layers")
        for block in device.onsite_blocks:
            maximum_repeat = max(maximum_repeat,
                _check_equal(_matrix(block, "device onsite"), d00, "device onsite"))
        for block in device.coupling_blocks:
            maximum_repeat = max(maximum_repeat,
                _check_equal(_matrix(block, "device coupling"), d01, "device coupling"))
        onsite.append(d00)
        forward.append(d01)
        reverse.append(d10)
    onsite, forward, reverse = map(np.stack, (onsite, forward, reverse))
    maximum_reality = 0.
    for q, minus in enumerate(negative):
        for name, blocks in (("onsite", onsite), ("forward", forward),
                             ("reverse", reverse)):
            maximum_reality = max(maximum_reality,
                _check_equal(blocks[int(minus)], blocks[q].conj(), name+" q reality"))
    return RepeatedHarmonicBlocks(_readonly(onsite), _readonly(forward),
        _readonly(reverse), dict(three_layer_oracle_checked=True,
            layers_checked=3, qpoints_checked=nq, layer_ndof=int(layer_ndof),
            maximum_repeat_residual=maximum_repeat,
            maximum_reverse_residual=maximum_reverse,
            maximum_onsite_hermiticity_residual=maximum_hermiticity,
            maximum_q_reality_residual=maximum_reality,
            finite_device_reverse="forward conjugate transpose, as in Device1D",
            endpoint_onsite_modified=False, range_truncated=False,
            physical_fields_repaired=False))


def _norm_upper(values):
    """Extended-precision Frobenius upper bound including reduction error."""
    raw = np.asarray(values, dtype=np.clongdouble)
    squared = np.sum(raw.real*raw.real + raw.imag*raw.imag, dtype=np.longdouble)
    estimate = np.sqrt(squared)
    epsilon = np.finfo(np.longdouble).eps
    return estimate*(1+16*raw.size*epsilon) + np.finfo(np.longdouble).tiny


def _product_bound(left, right):
    """Conservative extended-complex product roundoff norm bound."""
    epsilon = np.finfo(np.longdouble).eps
    operations = 16*left.shape[1]
    gamma = operations*epsilon/(1-operations*epsilon)
    absolute_product = np.abs(left) @ np.abs(right)
    return gamma*_norm_upper(absolute_product)


def certify_repeated_stability(onsite, forward, n_layers, *, atol=1e-12):
    """Certify the old ``lambda_min(D) >= -atol`` guard in linear work.

    A block-bidiagonal Cholesky factor of the auxiliary D+(atol/2)I is
    constructed with fixed-size solves.  Reconstructing its diagonal and
    nearest-neighbour blocks gives a PSD matrix LL†.  Independent extended
    precision residuals bound ``||D+(atol/2)I - LL†||2`` by block row sums.
    Passing requires this bound <= atol/2.  Consequently the certificate
    conservatively implies the original spectral guard; ambiguous pivots fail
    closed.  It may reject a state lying just inside the old tolerance.

    Work is O(n_layers*b³), auxiliary memory O(b²), and no dense long-device
    eigenproblem is used.  The physical input matrices are never modified.
    The Hermitian audit target follows the lower-triangle convention used by
    the previous np.linalg.eigvalsh guard; raw Hermiticity is checked first.
    """
    h = _matrix(onsite, "onsite")
    upper = _matrix(forward, "forward")
    if upper.shape != h.shape:
        raise ValueError("onsite and forward blocks must have the same shape")
    if (isinstance(n_layers, (bool, np.bool_))
            or not isinstance(n_layers, (int, np.integer)) or n_layers < 1):
        raise ValueError("n_layers must be a positive integer")
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("The original stability allowance must be positive")
    _check_equal(h, _adjoint(h), "onsite Hermiticity")
    # This is an independent audit target, not a replacement propagated field.
    target = np.tril(h, -1) + _adjoint(np.tril(h, -1)) + np.diag(h.diagonal().real)
    shift = float(atol)/2
    shifted = target + shift*np.eye(len(h))
    lower = _adjoint(upper)
    target_ext = np.asarray(target, dtype=np.clongdouble)
    target_ext += np.longdouble(shift)*np.eye(len(h), dtype=np.clongdouble)
    lower_ext = np.asarray(lower, dtype=np.clongdouble)
    previous = None
    maximum_diagonal = np.longdouble(0)
    maximum_offdiagonal = np.longdouble(0)
    minimum_pivot_diagonal = float("inf")
    for layer in range(int(n_layers)):
        below = None
        schur = shifted.copy()
        if previous is not None:
            below = np.linalg.solve(previous.conj(), lower.T).T
            if not np.isfinite(below).all():
                raise ValueError("Nonfinite harmonic block-Cholesky solve")
            schur -= below @ _adjoint(below)
        try:
            diagonal = np.linalg.cholesky(schur)
        except np.linalg.LinAlgError as error:
            raise ValueError("Harmonic stability certificate has an indefinite or ambiguous pivot"
                             f" at layer {layer}") from error
        if not np.isfinite(diagonal).all():
            raise ValueError("Nonfinite harmonic Cholesky factor")
        minimum_pivot_diagonal = min(minimum_pivot_diagonal,
                                    float(np.min(diagonal.diagonal().real)))
        dext = np.asarray(diagonal, dtype=np.clongdouble)
        dd = dext @ _adjoint(dext)
        diagonal_error = target_ext-dd
        diagonal_roundoff = _product_bound(dext, _adjoint(dext))
        if below is not None:
            bext = np.asarray(below, dtype=np.clongdouble)
            pext = np.asarray(previous, dtype=np.clongdouble)
            bb = bext @ _adjoint(bext)
            diagonal_error -= bb
            diagonal_roundoff += _product_bound(bext, _adjoint(bext))
            off_error = lower_ext-bext @ _adjoint(pext)
            off_roundoff = _product_bound(bext, _adjoint(pext))
            off_roundoff += 8*np.finfo(np.longdouble).eps*(
                _norm_upper(lower_ext)+_norm_upper(bext)*_norm_upper(pext))
            maximum_offdiagonal = max(maximum_offdiagonal,
                                      _norm_upper(off_error)+off_roundoff)
        diagonal_roundoff += 16*np.finfo(np.longdouble).eps*(
            _norm_upper(target_ext)+_norm_upper(dd)
            + (0 if below is None else _norm_upper(bb)))
        maximum_diagonal = max(maximum_diagonal,
                              _norm_upper(diagonal_error)+diagonal_roundoff)
        previous = diagonal
    residual_bound = maximum_diagonal + 2*maximum_offdiagonal
    lower_bound = -np.longdouble(shift)-residual_bound
    if not np.isfinite(residual_bound) or lower_bound < -np.longdouble(atol):
        raise ValueError("Harmonic stability certificate is numerically ambiguous: "
                         f"lower bound {float(lower_bound):.6g}, original floor {-atol:.6g}")
    return dict(passed=True, layers_checked=int(n_layers), block_ndof=len(h),
        original_eigenvalue_floor=-float(atol),
        certified_eigenvalue_lower_bound=float(lower_bound),
        auxiliary_certificate_shift=shift,
        maximum_diagonal_factorization_residual_bound=float(maximum_diagonal),
        maximum_offdiagonal_factorization_residual_bound=float(maximum_offdiagonal),
        factorization_operator_error_bound=float(residual_bound),
        minimum_factor_diagonal=minimum_pivot_diagonal,
        extended_precision_epsilon=float(np.finfo(np.longdouble).eps),
        block_cholesky_steps=int(n_layers), block_solve_steps=int(n_layers)-1,
        work_scaling="O(number_of_layers * fixed_block_ndof^3)",
        auxiliary_memory_scaling="O(fixed_block_ndof^2)",
        dense_long_device_constructed=False, physical_fields_shifted=False,
        physical_fields_repaired=False)
