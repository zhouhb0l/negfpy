"""Bounded input-feasibility screening for unevaluated native proposals.

Only a common scalar step is shortened. No field is projected or reconstructed,
and the map still performs all of its original checks before every Dyson solve.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

import numpy as np

from .transverse_channel_feedback import (
    _COMPONENTS,
    ChannelSpectrum,
    _finish,
    _integer,
    _local_checks,
    _measure,
    _real,
    _spectrum,
    mix_channel_spectra,
)


@dataclass(frozen=True, slots=True)
class ChannelProposalFeasibility:
    """An explicit proposal policy, independent of scientific acceptance."""

    mode: str = 'disabled'
    factor: float = .5
    max_halvings: int = 8

    def __post_init__(self):
        if self.mode not in ('disabled', 'psd-backtracking'):
            raise ValueError('Unknown proposal-feasibility policy.')
        if not 0 < _real(self.factor, 'factor') < 1:
            raise ValueError('The proposal damping factor must lie in (0,1).')
        _integer(self.max_halvings, 'max_halvings', zero=True)
        if self.max_halvings > 8:
            raise ValueError('At most eight proposal halvings are supported.')


class ChannelProposalFailure(ValueError):
    """A pre-map stop with the complete finite screening evidence retained."""

    def __init__(self, report, *, time_budget=False):
        self.report = report
        self.time_budget = time_budget
        super().__init__(report['error'] or report['stop_reason'])


def proposal_field_checks(mesh, grid, spectrum, rtol, atol):
    """Measure the same input field guards using the map's unchanged helpers.

    This is the loop in ``_field_checks``, with only its final raising gate
    disabled so finite PSD-only failures can be recorded before shortening a
    proposal. Exported-contour roundtrips remain diagnostic.
    """
    rtol, atol = _real(rtol, 'rtol'), _real(atol, 'atol')
    if rtol + atol == 0:
        raise ValueError('At least one input-feasibility tolerance must be positive.')
    if not isinstance(spectrum, ChannelSpectrum):
        raise TypeError('A proposal must contain independent native R/C/D fields.')
    if spectrum.retarded.ndim != 4 or spectrum.retarded.shape[-1] != spectrum.retarded.shape[-2]:
        raise ValueError('Proposal fields must have square q/frequency matrix blocks.')
    shape = (mesh.n_qpoints, grid.n_points, *spectrum.retarded.shape[-2:])
    _spectrum(spectrum, shape, 'proposal')
    metrics, exported = {}, {}
    for q in range(mesh.n_qpoints):
        opposite = mesh.negative_index(q)
        for w in range(grid.n_points):
            reverse = grid.n_points - 1 - w
            r, c, d = (getattr(spectrum, key)[q, w] for key in _COMPONENTS)
            _local_checks(metrics, exported, r, c, d, (q, w))
            _measure(metrics, 'statistical_reflection', c,
                     spectrum.statistical[opposite, reverse].T, (q, w))
            _measure(metrics, 'spectral_reflection', d,
                     -spectrum.spectral[opposite, reverse].T, (q, w))
            _measure(metrics, 'bosonic_reflection', c - .5 * d,
                     (spectrum.statistical[opposite, reverse]
                      + .5 * spectrum.spectral[opposite, reverse]).T, (q, w))
            _measure(metrics, 'retarded_reality', r,
                     spectrum.retarded[opposite, reverse].conj(), (q, w))
    return (_finish(metrics, rtol, atol, 'proposal input Sigma', gate=False),
            _finish(exported, rtol, atol, 'proposal input Sigma exports', gate=False))


def feasible_channel_proposal(mesh, grid, anchor, image, candidate, *, original_kind,
                              mixing, policy, rtol, atol, time_remaining=None):
    """Screen, shorten PSD-only-invalid Anderson proposals, or stop explicitly.

    There are at most ``max_halvings+1`` accelerated screens, then one freshly
    constructed linear fallback screen. Linear candidates are never shortened.
    No full map is evaluated here and all screening evidence is returned.
    """
    if not isinstance(policy, ChannelProposalFeasibility) or policy.mode != 'psd-backtracking':
        raise ValueError('An enabled proposal-feasibility policy is required.')
    started = monotonic()
    accelerated = original_kind == 'anderson1'
    report = {'mode': policy.mode, 'original_kind': original_kind,
              'selected_kind': original_kind, 'screens': [], 'chosen_factor': None,
              'fallback_used': False, 'passed': False, 'stop_reason': None, 'error': None,
              'screen_count': 0, 'maximum_screen_count': policy.max_halvings + 2 if accelerated else 1,
              'elapsed_seconds': 0., 'map_admitted': False}

    def finish(reason, error=None, *, time_budget=False):
        report.update(stop_reason=reason, error=error, screen_count=len(report['screens']),
                      elapsed_seconds=monotonic() - started)
        if reason not in ('feasible', 'linear_fallback_feasible'):
            raise ChannelProposalFailure(report, time_budget=time_budget)

    def screen(value, factor, kind):
        if time_remaining is not None and time_remaining() <= 0:
            finish('time_budget', time_budget=True)
        row = {'factor': factor, 'kind': kind, 'checks': {}, 'export_checks': {},
               'failed_keys': [], 'passed': False, 'elapsed_seconds': 0., 'error': None}
        before = monotonic()
        try:
            checks, exported = proposal_field_checks(mesh, grid, value, rtol, atol)
            if any(not np.isfinite(row[key]) for rows in (checks, exported)
                   for row in rows.values() for key in ('error', 'scale', 'allowance', 'ratio')):
                raise ValueError('Nonfinite proposal field-guard metric.')
            failed = [key for key, measured in checks.items()
                      if not np.isfinite(measured['ratio']) or measured['ratio'] > 1]
            row.update(checks=checks, export_checks=exported,
                       failed_keys=failed, passed=not failed)
        except (TypeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['elapsed_seconds'] = monotonic() - before
            report['screens'].append(row)
            finish('nonfinite_or_validation_failure', row['error'])
        row['elapsed_seconds'] = monotonic() - before
        report['screens'].append(row)
        if any(key not in ('lesser_psd', 'greater_psd') for key in row['failed_keys']):
            finish('non_psd_guard_failure', f'Invalid proposal input Sigma: {row["failed_keys"]}')
        return row['passed']

    trial = candidate
    for halving in range(policy.max_halvings + 1 if accelerated else 1):
        if time_remaining is not None and time_remaining() <= 0:
            finish('time_budget', time_budget=True)
        factor = policy.factor**halving
        if halving:
            # Release the previous damped trial before constructing the next;
            # only the original proposal plus one damped spectrum are retained.
            trial = None
            trial = mix_channel_spectra(anchor, candidate, factor)
        if screen(trial, factor, 'original' if not halving else 'damped'):
            report.update(passed=True, chosen_factor=factor)
            finish('feasible')
            return trial, report
        if not accelerated:
            finish('linear_input_failure', 'The original linear candidate failed the PSD guard.')
    trial = None
    report.update(fallback_used=True, selected_kind='linear_feasibility_fallback')
    if time_remaining is not None and time_remaining() <= 0:
        finish('time_budget', time_budget=True)
    trial = mix_channel_spectra(anchor, image, mixing)
    if screen(trial, None, 'linear_fallback'):
        report.update(passed=True)
        finish('linear_fallback_feasible')
        return trial, report
    finish('linear_fallback_failure', 'The single linear fallback failed the PSD guard.')


__all__ = ['ChannelProposalFailure', 'ChannelProposalFeasibility',
           'feasible_channel_proposal', 'proposal_field_checks']
