"""Bounded scalar diagnostics for an already computed full-device Green solve.

This module never solves, repairs, or accepts a transport state. Only the two
points selected by maximum absolute Keldysh error and maximum pointwise allowance
ratio retain matrix copies. Ratios here are diagnostics, not new tolerances.
"""

from __future__ import annotations

import numpy as np


def green_diagnostic_workspace_bytes(full_ndof):
    """Conservative extra numeric workspace, excluding BLAS internal scratch.

Two retained points require twelve complex matrices. Replacement, extrema
measurement, and the final source/inverse decomposition fit within the remaining
52 matrix equivalents. No frequency stack is retained by the collector.
    """
    if isinstance(full_ndof, (bool, np.bool_)) or not isinstance(full_ndof, (int, np.integer)):
        raise TypeError('full_ndof must be a positive integer.')
    if full_ndof < 1:
        raise ValueError('full_ndof must be positive.')
    return 16 * 64 * int(full_ndof)**2


def _adjoint(matrix):
    return matrix.conj().T


def _maximum(matrix):
    absolute = np.abs(matrix)
    flat = int(np.argmax(absolute))
    entry = tuple(int(x) for x in np.unravel_index(flat, matrix.shape))
    return float(absolute[entry]), entry


def _ratio(error, allowance):
    return error / allowance if allowance else (0.0 if error == 0 else float('inf'))


def _location(q, w, omega, entry, q_address):
    return {'q_index': int(q), 'frequency_index': int(w), 'omega_solver': float(omega),
            'matrix_entry': list(entry), 'q_address': [int(x) for x in q_address]}


def _complex(value):
    return {'real': float(value.real), 'imag': float(value.imag)}


def _matrix_summary(matrix):
    maximum, entry = _maximum(matrix)
    return {'max_abs': maximum, 'max_abs_entry': list(entry),
            'value_at_max_abs': _complex(matrix[entry]),
            'norm_inf': float(np.linalg.norm(matrix, ord=np.inf))}


def _json_safe(value, nonfinite, path=''):
    """JSON null plus a path record represents overflow; never emit NaN/Infinity."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {key: _json_safe(item, nonfinite, path+'.'+key) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, nonfinite, f'{path}[{index}]')
                for index, item in enumerate(value)]
    if isinstance(value, float) and not np.isfinite(value):
        nonfinite.append(path.lstrip('.'))
        return None
    return value


def _context(snapshot, rtol, atol):
    operator, retarded, source_less, source_greater, lesser, greater = snapshot['matrices']
    advanced = _adjoint(retarded)
    identity = np.eye(len(operator), dtype=complex)
    operator_norm = float(np.linalg.norm(operator, ord=np.inf))
    inverse_norm = float(np.linalg.norm(retarded, ord=np.inf))
    conditioning = operator_norm * inverse_norm
    left = identity - operator @ retarded
    right = identity - retarded @ operator
    left_norm = float(np.linalg.norm(left, ord=np.inf))
    right_norm = float(np.linalg.norm(right, ord=np.inf))
    source_difference = source_greater - source_less
    operator_spectral = _adjoint(operator) - operator
    source_defect = source_difference - operator_spectral
    propagated = retarded @ source_defect @ advanced
    contour_difference = greater - lesser
    green_spectral = retarded - advanced
    observed = contour_difference - green_spectral
    remainder = observed - propagated
    difference_product = retarded @ source_difference @ advanced
    separate_product_defect = contour_difference - difference_product
    error, error_entry = _maximum(observed)
    scale = max(_maximum(contour_difference)[0], _maximum(green_spectral)[0])
    allowance = atol + rtol * scale
    source_error = _maximum(source_defect)[0]
    source_scale = max(_maximum(source_difference)[0], _maximum(operator_spectral)[0])
    less_max, greater_max = _maximum(lesser)[0], _maximum(greater)[0]
    difference_max = _maximum(contour_difference)[0]
    entry_difference = abs(contour_difference[error_entry])
    source_less_max, source_greater_max = _maximum(source_less)[0], _maximum(source_greater)[0]
    source_difference_max = _maximum(source_difference)[0]
    return {
        **snapshot['point'],
        'keldysh': {'error': error, 'scale': scale, 'allowance': allowance,
                    'ratio': _ratio(error, allowance), 'error_entry': list(error_entry)},
        'inverse': {
            'norm_convention': 'infinity', 'operator_norm': operator_norm,
            'computed_inverse_norm': inverse_norm,
            'condition_estimate': conditioning,
            'condition_estimate_times_float64_epsilon': conditioning*np.finfo(float).eps,
            'condition_estimate_interpretation': 'norm(A)*norm(computed Gr); not a certified bound',
            'backward_residual_denominator': conditioning+1.0,
            'left_absolute_residual': left_norm, 'right_absolute_residual': right_norm,
            'left_normalized_backward_residual': _ratio(left_norm, conditioning+1.0),
            'right_normalized_backward_residual': _ratio(right_norm, conditioning+1.0),
        },
        'source_identity': {
            'definition': 'S_greater-S_lesser-(A_adjoint-A)',
            'error': source_error, 'scale': source_scale,
            'allowance': atol+rtol*source_scale,
            'ratio': _ratio(source_error, atol+rtol*source_scale),
            'defect': _matrix_summary(source_defect),
        },
        'decomposition': {
            'observed': _matrix_summary(observed),
            'propagated_source_defect': _matrix_summary(propagated),
            'remainder': _matrix_summary(remainder),
            'remainder_definition': 'observed-Gr*source_defect*Ga',
            'separate_products_minus_difference_product': _matrix_summary(separate_product_defect),
            'interpretation': 'Diagnostic algebra; remainder includes inverse and product rounding.',
        },
        'cancellation': {
            'lesser_max_abs': less_max, 'greater_max_abs': greater_max,
            'greater_minus_lesser_max_abs': difference_max,
            'global_magnitude_ratio': _ratio(less_max+greater_max, difference_max),
            'at_error_entry': {
                'entry': list(error_entry), 'lesser': _complex(lesser[error_entry]),
                'greater': _complex(greater[error_entry]),
                'difference': _complex(contour_difference[error_entry]),
                'magnitude_ratio': _ratio(abs(lesser[error_entry])+abs(greater[error_entry]),
                                          entry_difference),
            },
            'source_lesser_max_abs': source_less_max,
            'source_greater_max_abs': source_greater_max,
            'source_difference_max_abs': source_difference_max,
            'source_magnitude_ratio': _ratio(source_less_max+source_greater_max,
                                           source_difference_max),
        },
    }


class FullGreenDiagnosticCollector:
    """Observe existing full solves, retaining at most two distinct matrix points."""

    def __init__(self, *, n_q, n_frequency, rtol, atol):
        self.expected_points = int(n_q) * int(n_frequency)
        self.completed_points = 0
        self.rtol, self.atol = float(rtol), float(atol)
        self.error = self.scale = self.local_ratio = -1.0
        self.error_location = self.scale_location = self.local_ratio_location = None
        self.error_key = self.ratio_key = None
        self._snapshots = {}

    def observe(self, q, w, omega, q_address, operator, retarded, source_less,
                source_greater, lesser, greater):
        first = greater - lesser
        second = retarded - _adjoint(retarded)
        error, error_entry = _maximum(first-second)
        first_scale, first_entry = _maximum(first)
        second_scale, second_entry = _maximum(second)
        scale = max(first_scale, second_scale)
        point_ratio = _ratio(error, self.atol+self.rtol*scale)
        key = (int(q), int(w))
        point = {'q_index': int(q), 'frequency_index': int(w),
                 'omega_solver': float(omega), 'q_address': [int(x) for x in q_address]}
        if error > self.error:
            self.error, self.error_key = error, key
            self.error_location = _location(q, w, omega, error_entry, q_address)
        if scale > self.scale:
            self.scale = scale
            entry = first_entry if first_scale >= second_scale else second_entry
            self.scale_location = _location(q, w, omega, entry, q_address)
            self.scale_location['term'] = ('greater_minus_lesser' if first_scale >= second_scale
                                           else 'retarded_minus_advanced')
        if point_ratio > self.local_ratio:
            self.local_ratio, self.ratio_key = point_ratio, key
            self.local_ratio_location = _location(q, w, omega, error_entry, q_address)
            self.local_ratio_location.update(error=error, scale=scale,
                                             allowance=self.atol+self.rtol*scale)
        selected = {self.error_key, self.ratio_key}
        for old_key in tuple(self._snapshots):
            if old_key not in selected:
                del self._snapshots[old_key]
        if key in selected and key not in self._snapshots:
            self._snapshots[key] = {
                'point': point,
                'matrices': tuple(field.copy() for field in
                                  (operator, retarded, source_less, source_greater, lesser, greater)),
            }
        self.completed_points += 1

    def finish(self, full_metrics, *, full_green_validation_error, active_green_validation_error,
               validation_checks):
        metrics = {}
        for name, (error, scale) in full_metrics.items():
            allowance = self.atol+self.rtol*scale
            metrics[name] = {'error': error, 'scale': scale, 'allowance': allowance,
                             'ratio': _ratio(error, allowance)}
        contexts = []
        for key, snapshot in self._snapshots.items():
            context = _context(snapshot, self.rtol, self.atol)
            context['selection_reasons'] = [name for name, selected in
                                            (('maximum_absolute_error', self.error_key),
                                             ('maximum_pointwise_ratio', self.ratio_key))
                                            if key == selected]
            contexts.append(context)
        result = {
            'schema': 'negfpy.transverse-full-green-diagnostic.v1',
            'scope': 'full Dyson grid; no Born image, SCBA acceptance, or transport acceptance',
            'completed_points': self.completed_points, 'expected_points': self.expected_points,
            'all_points_observed': self.completed_points == self.expected_points,
            'identity_rtol': self.rtol, 'identity_atol': self.atol,
            'full_green_validation_error': full_green_validation_error,
            'active_green_validation_error': active_green_validation_error,
            'validation_checks': validation_checks,
            'full_green_checks': metrics,
            'full_green_keldysh': {
                **metrics['keldysh_identity'],
                'scaling': 'global maximum error / (atol + rtol * independent global maximum scale)',
                'error_location': self.error_location, 'scale_location': self.scale_location,
                'maximum_pointwise_ratio': self.local_ratio,
                'maximum_pointwise_ratio_location': self.local_ratio_location,
                'pointwise_ratio_is_separate_diagnostic': True,
            },
            'retained_point_count': len(contexts), 'retained_point_limit': 2,
            'extrema_contexts': contexts,
            'nonfinite_encoding': 'JSON null; exact paths listed in nonfinite_fields',
        }
        nonfinite = []
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            result = _json_safe(result, nonfinite)
        result['nonfinite_fields'] = nonfinite
        return result


__all__ = ['FullGreenDiagnosticCollector', 'green_diagnostic_workspace_bytes']
