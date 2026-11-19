"""Reference/increment SCBA with bounded analytic feedback proposals.

Canonical map formulas, six screens and field guards remain imported read-only.
Analytic tangents are signed directions, not physical states or extra full maps.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import sqrt
from time import monotonic

import numpy as np

from negfpy.anharmonic.transverse_channel_accel import (
    ChannelAccelerationConfig, ChannelSelectionPolicy, channel_merit,
    channel_scales, _diagnostic_ratio, _dot_blocks, plan_channel_accel,
    propose_channel_anderson1,
)
from negfpy.anharmonic.transverse_channel_feasibility import (
    ChannelProposalFailure, ChannelProposalFeasibility, proposal_field_checks,
)
from negfpy.anharmonic.transverse_channel_feedback import (
    _COMPONENTS, ChannelSCBAConfig, ChannelSCBAState, ChannelSpectrum,
    _record, _spectrum, mix_channel_spectra,
)
from .map_adapter import evaluate_reference_map, tangent_map

_NUMERICAL_ERRORS = (ValueError, FloatingPointError, np.linalg.LinAlgError, TimeoutError)


@dataclass(frozen=True, slots=True, eq=False)
class ReferenceFeedbackResult:
    state: object
    last_evaluated_state: object
    history: tuple
    decisions: tuple
    converged: bool
    stop_reason: str
    failure_reason: str | None
    plan: object
    proposal_screens: tuple = ()
    tangent_count: int = 0
    tangent_records: tuple = ()
    map_records: tuple = ()


def feasible_reference_proposal(mesh, grid, anchor, image, candidate, *, original_kind,
                              mixing, policy, rtol, atol, time_remaining=None):
    """Screen PSD-only-invalid AA/analytic proposals, or stop explicitly.

    There are at most ``max_halvings+1`` accelerated screens, then one freshly
    constructed linear fallback screen. Linear candidates are never shortened.
    No full map is evaluated here and all screening evidence is returned.
    """
    if not isinstance(policy, ChannelProposalFeasibility) or policy.mode != 'psd-backtracking':
        raise ValueError('An enabled proposal-feasibility policy is required.')
    started = monotonic()
    accelerated = original_kind in ('anderson1', 'analytic-feedback1')
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


def _residual_spectrum(state, raw):
    """One signed direction; no physical PSD or projection is imposed on it."""
    arrays = []
    for name in _COMPONENTS:
        values = np.subtract(getattr(raw, name), getattr(state, name))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite analytic residual direction.')
        values.setflags(write=False)
        arrays.append(values)
    return ChannelSpectrum(*arrays)


def propose_analytic_feedback(state, raw, jacobian_residual, *, acceleration, absolute_floor):
    """Projected solve (I-J) delta=r with delta=t*r, using one real t.

    The rank, coefficient and common trust controls have the same numerical
    meaning as AA1. This remains an unevaluated candidate until a fresh full map.
    """
    shape = state.retarded.shape
    _spectrum(jacobian_residual, shape, 'jacobian_residual')
    scales = channel_scales(state, raw, absolute_floor)
    residual = lambda key, index: getattr(raw, key)[index] - getattr(state, key)[index]
    action = lambda key, index: residual(key, index) - getattr(jacobian_residual, key)[index]
    denominator = _dot_blocks(action, action, scales, shape)
    residual_squared = _dot_blocks(residual, residual, scales, shape)
    details = {'scales': list(scales), 'denominator': denominator,
               'residual_reference_squared': residual_squared,
               'rank_rtol': acceleration.rank_rtol,
               'reference_merit': sqrt(max(0., residual_squared)),
               'coefficient': None, 'analytic_direction_is_physical_state': False}
    if denominator <= max(np.finfo(float).tiny,
                          acceleration.rank_rtol**2 * residual_squared):
        return None, dict(details, reason='degenerate_analytic_direction')
    raw_coefficient = _dot_blocks(action, residual, scales, shape) / denominator
    if not np.isfinite(raw_coefficient):
        return None, dict(details, reason='nonfinite_analytic_coefficient')
    coefficient = float(np.clip(raw_coefficient, -acceleration.coefficient_limit,
                                acceleration.coefficient_limit))
    step_norm = abs(coefficient) * sqrt(max(0., residual_squared))
    state_values = lambda key, index: getattr(state, key)[index]
    raw_values = lambda key, index: getattr(raw, key)[index]
    reference_norm = sqrt(max(_dot_blocks(state_values, state_values, scales, shape),
                              _dot_blocks(raw_values, raw_values, scales, shape), 0.))
    bound = acceleration.trust_radius * reference_norm
    clip = min(1., bound / step_norm) if step_norm else 1.
    arrays = []
    with np.errstate(over='raise', invalid='raise'):
        for key in _COMPONENTS:
            values = np.empty(shape, dtype=np.complex128)
            for index in np.ndindex(shape[:2]):
                values[index] = getattr(state, key)[index] + clip * coefficient * residual(key, index)
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite analytic-feedback proposal.')
            values.setflags(write=False)
            arrays.append(values)
    details.update(reason='proposed', raw_coefficient=raw_coefficient,
                   coefficient=coefficient, bounded_coefficient=coefficient,
                   coefficient_clipped=coefficient != raw_coefficient,
                   unbounded_step_norm=step_norm, reference_field_norm=reference_norm,
                   scaled_step_bound=bound, clip_factor=clip,
                   scaled_step_norm=clip * step_norm, trust_radius=acceleration.trust_radius)
    return ChannelSpectrum(*arrays), details


def _selection(reference, trial, proposal, policy):
    """Preserve canonical actual-map merit/Fro/current selection verbatim."""
    record = trial.record
    nonincrease, reference_merit, trial_merit, scales = {}, None, None, None
    reference_fro, trial_fro, growth_ratio, within_limit = {}, {}, {}, {}
    merit_limit = merit_ratio = merit_within_limit = None
    selected, reason = True, 'linear_evaluation_selected'
    if proposal['kind'] in ('anderson1', 'analytic-feedback1'):
        scales = tuple(proposal['scales'])
        reference_merit = proposal['reference_merit']
        reference_fro = {key: reference.record.sigma_residuals[key].error_frobenius
                         for key in ('retarded', 'lesser', 'greater')}
        trial_fro = {key: record.sigma_residuals[key].error_frobenius for key in reference_fro}
        nonincrease = {key: trial_fro[key] <= reference_fro[key] for key in reference_fro}
        growth_ratio = {key: _diagnostic_ratio(trial_fro[key], reference_fro[key])
                        for key in reference_fro}
        within_limit = {key: trial_fro[key] <= policy.component_growth_limit * reference_fro[key]
                        for key in reference_fro}
        if record.converged:
            selected, reason = True, 'fully_converged_trial'
        else:
            trial_merit = channel_merit(trial.sigma_input, trial.evaluation.map_sigma, scales)
            merit_limit = policy.merit_ratio_limit * reference_merit
            merit_ratio = _diagnostic_ratio(trial_merit, reference_merit)
            merit_within_limit = trial_merit <= merit_limit
            selected = bool(merit_within_limit and all(within_limit.values()) and record.current_passed)
            passed_reason = ('merit_and_unmixed_frobenius_passed'
                             if policy.mode == 'component-nonincreasing'
                             else 'merit_and_bounded_frobenius_passed')
            reason = passed_reason if selected else 'trial_selection_rejected'
    return selected, {
        'index': trial.index, 'accepted_as_anchor': selected,
        'selected_anchor_index': trial.index if selected else reference.index,
        'kind': proposal['kind'], 'reference_anchor_index': proposal['anchor_index'],
        'reason': reason, 'reference_merit': reference_merit, 'trial_merit': trial_merit,
        'merit_scales': list(scales) if scales is not None else None,
        'merit_evaluated': trial_merit is not None,
        'unmixed_frobenius_nonincrease': nonincrease,
        'selection_policy': policy.mode,
        'component_growth_limit': policy.component_growth_limit,
        'merit_ratio_limit': policy.merit_ratio_limit,
        'unmixed_frobenius_reference': reference_fro,
        'unmixed_frobenius_trial': trial_fro,
        'unmixed_frobenius_growth_ratio': growth_ratio,
        'unmixed_frobenius_within_limit': within_limit,
        'merit_limit': merit_limit, 'merit_ratio': merit_ratio,
        'merit_within_limit': merit_within_limit,
        'current_passed': record.current_passed, 'fully_converged': record.converged,
    }


def run_reference_feedback(grid, vertex, dynamical, left, right, *, scale=None, g=.03,
                           config=None, acceleration=None, selection=None, feasibility=None,
                           on_start=None, on_evaluation=None, on_proposal=None, on_tangent=None,
                           methodconfig=None, sigma_initial=None):
    """Fresh history, two linear maps, bounded analytic/AA1 anchor updates.

    Every completed rejected nonlinear trial consumes the full-map budget.
    Every attempted signed tangent consumes a separate tangent count. There is
    no map retry on a physical/source/Green failure, including tangent failures.
    """
    started = monotonic()
    config = ChannelSCBAConfig() if config is None else config
    acceleration = ChannelAccelerationConfig() if acceleration is None else acceleration
    selection = ChannelSelectionPolicy() if selection is None else selection
    feasibility = ChannelProposalFeasibility() if feasibility is None else feasibility
    if not isinstance(config, ChannelSCBAConfig):
        raise TypeError('config must be ChannelSCBAConfig.')
    if not isinstance(acceleration, ChannelAccelerationConfig):
        raise TypeError('acceleration must be ChannelAccelerationConfig.')
    if not isinstance(selection, ChannelSelectionPolicy):
        raise TypeError('selection must be ChannelSelectionPolicy.')
    if not isinstance(feasibility, ChannelProposalFeasibility):
        raise TypeError('feasibility must be ChannelProposalFeasibility.')
    if any(cb is not None and not callable(cb) for cb in
           (on_start, on_evaluation, on_proposal, on_tangent)):
        raise TypeError('Callbacks must be callable.')
    method = dict(methodconfig or {})
    warmup = method.get('warmup_evaluations', 2)
    interval = method.get('tangent_interval', 3)
    max_tangent = method.get('max_tangent_directions', 6)
    if (warmup != 2 or isinstance(interval, bool) or interval != 3
            or isinstance(max_tangent, bool) or not isinstance(max_tangent, int)
            or not 0 <= max_tangent <= 6 or config.max_evaluations > 18):
        raise ValueError('Reference-seeded SCBA requires two warmups, <=18 maps and <=6 signed directions.')
    plan = plan_channel_accel(vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof,
                              vertex.ndof, len(left.device_indices), len(right.device_indices),
                              config=config, acceleration=acceleration, feasibility=feasibility)
    if plan.peak_owned_bytes > config.max_bytes:
        raise ValueError('Base channel allocation exceeds max_bytes.')
    if 2 * plan.fixed_plan.map_plan.contraction_work > config.max_contraction_work:
        raise ValueError('Reference/tangent two-Born work exceeds the per-call work cap.')
    if 2 * plan.fixed_plan.map_plan.dyson_evaluations > config.max_dyson_evaluations:
        raise ValueError('Reference two-Dyson work exceeds the per-call solve cap.')
    state = latest = None
    previous = None
    sigma = None
    first_index = 1 if sigma_initial is None else 2
    if sigma_initial is not None:
        shape = (vertex.mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
        _spectrum(sigma_initial, shape, 'sigma_initial')
        if any(getattr(sigma_initial, key).dtype != np.dtype('complex128') for key in _COMPONENTS):
            raise ValueError('sigma_initial must have native complex128 fields.')
        arrays = [np.array(getattr(sigma_initial, key), copy=True, order='C') for key in _COMPONENTS]
        for array in arrays:
            array.setflags(write=False)
        sigma = ChannelSpectrum(*arrays)
        del arrays, array
    warmup_end = first_index + warmup - 1
    proposal = {'kind': 'fresh_zero' if sigma_initial is None else 'external_initial_guess',
                'anchor_index': None, 'previous_index': None,
                'reason': 'independent_exact_zero' if sigma_initial is None else 'fresh_history_external_RCD'}
    history, decisions, screens, tangent_records, map_records = [], [], [], [], []
    tangent_count = 0
    tangent_anchors = set()
    estimate = config.initial_map_estimate_seconds
    tangent_estimate = float(method.get('initial_tangent_estimate_seconds', 2 * estimate))
    if not np.isfinite(tangent_estimate) or tangent_estimate < 0:
        raise ValueError('The initial tangent estimate must be finite and nonnegative.')

    def remaining():
        return config.max_seconds - (monotonic() - started)

    def result(reason, failure=None, converged=False):
        return ReferenceFeedbackResult(state, latest, tuple(history), tuple(decisions),
                                       converged, reason, failure, plan, tuple(screens),
                                       tangent_count, tuple(tangent_records), tuple(map_records))

    def tangent_emit(index, row):
        tangent_records.append(deepcopy(row))
        if on_tangent is not None:
            on_tangent(index, deepcopy(row))

    def screen_emit(index, batch):
        batch.update(index=index, anchor_index=proposal['anchor_index'])
        screens.append(deepcopy(batch))
        if on_proposal is not None:
            on_proposal(index, deepcopy(batch))

    common = dict(scale=scale, g=g, workers=config.workers,
                  external_tile=config.external_tile, bin_tile=config.bin_tile,
                  max_bytes=config.max_bytes, max_contraction_work=config.max_contraction_work,
                  max_dyson_evaluations=config.max_dyson_evaluations,
                  identity_rtol=config.identity_rtol, identity_atol=config.identity_atol)
    for offset in range(config.max_evaluations):
        if remaining() <= config.time_reserve_seconds + config.next_map_safety_factor * estimate:
            return result('max_seconds')
        index = offset + first_index
        if feasibility.mode != 'disabled' and sigma is not None and state is not None:
            try:
                sigma, batch = feasible_reference_proposal(
                    vertex.mesh, grid, state.sigma_input, state.evaluation.map_sigma, sigma,
                    original_kind=proposal['kind'], mixing=config.mixing,
                    policy=feasibility, rtol=config.identity_rtol, atol=config.identity_atol,
                    time_remaining=lambda: remaining() - config.time_reserve_seconds)
            except ChannelProposalFailure as exc:
                screen_emit(index, exc.report)
                return result('max_seconds' if exc.time_budget else 'proposal_error',
                              None if exc.time_budget else f'{type(exc).__name__}: {exc}')
            batch['map_admitted'] = remaining() > (
                config.time_reserve_seconds + config.next_map_safety_factor * estimate)
            if not batch['map_admitted']:
                batch['stop_reason'] = 'map_time_budget'
            screen_emit(index, batch)
            if not batch['map_admitted']:
                return result('max_seconds')
            proposal = dict(proposal, kind=batch['selected_kind'], feasibility=deepcopy(batch))
            if batch['fallback_used']:
                proposal['reason'] = 'psd_backtracking_linear_fallback'
        if on_start is not None:
            on_start(index, sigma, dict(proposal))
        try:
            map_started = monotonic()
            evaluation, metadata = evaluate_reference_map(
                grid, vertex, dynamical, left, right, sigma_input=sigma,
                reference_sigma=state.sigma_input if state is not None else None,
                reference_green=state.evaluation.green if state is not None else None,
                reference_mapped=state.evaluation.map_sigma if state is not None else None,
                **common)
            duration = monotonic() - map_started
            record = _record(index, sigma, evaluation, duration, monotonic() - started, config)
            trial = ChannelSCBAState(index, sigma, evaluation, record)
            selected, decision = _selection(state, trial, proposal, selection)
            decision['map_metadata'] = deepcopy(metadata)
            if selected:
                if state is not None:
                    previous = state
                state = trial
        except _NUMERICAL_ERRORS as exc:
            return result('evaluation_error', f'{type(exc).__name__}: {exc}')
        latest = trial
        history.append(record)
        decisions.append(decision)
        map_records.append(dict(index=index, **deepcopy(metadata)))
        estimate = max(estimate, duration)
        if on_evaluation is not None:
            on_evaluation(latest, dict(decision))
        if record.converged:
            return result('fixed_point', converged=True)
        if offset + 1 == config.max_evaluations:
            break
        try:
            if index < warmup_end or acceleration.mode == 'linear-control' or not selected:
                sigma = mix_channel_spectra(state.sigma_input, state.evaluation.map_sigma, config.mixing)
                kind = ('linear_warmup' if index < warmup_end else
                        'linear_control' if selected else 'linear_fallback')
                proposal = {'kind': kind, 'anchor_index': state.index,
                            'previous_index': None,
                            'reason': 'rejected_trial_rollback' if not selected else kind}
                continue
            eligible = (state.index >= warmup_end and (state.index - warmup_end) % interval == 0
                        and state.index not in tangent_anchors and tangent_count < max_tangent)
            tangent_admitted = remaining() > (config.time_reserve_seconds
                                              + config.next_map_safety_factor * (estimate + tangent_estimate))
            sigma = None
            if eligible:
                tangent_anchors.add(state.index)
                if tangent_admitted:
                    tangent_count += 1
                    tangent_emit(state.index, {'phase': 'start', 'anchor_index': state.index,
                                              'tangent_number': tangent_count,
                                              'full_nonlinear_evaluation': False})
                    before = monotonic()
                    direction = _residual_spectrum(state.sigma_input, state.evaluation.map_sigma)
                    jacobian_direction, tangent_metadata = tangent_map(
                        grid, vertex, dynamical, left, right,
                        sigma_base=state.sigma_input, base_green=state.evaluation.green,
                        direction=direction, **common)
                    tangent_duration = monotonic() - before
                    tangent_estimate = max(tangent_estimate, tangent_duration)
                    sigma, details = propose_analytic_feedback(
                        state.sigma_input, state.evaluation.map_sigma, jacobian_direction,
                        acceleration=acceleration, absolute_floor=config.sigma_atol)
                    del direction, jacobian_direction
                    tangent_emit(state.index, dict(
                        phase='complete', anchor_index=state.index, tangent_number=tangent_count,
                        elapsed_seconds=tangent_duration, full_nonlinear_evaluation=False,
                        metadata=deepcopy(tangent_metadata), proposal=deepcopy(details)))
                    if sigma is not None:
                        proposal = dict(details, kind='analytic-feedback1',
                                        anchor_index=state.index, previous_index=None,
                                        tangent_number=tangent_count)
                else:
                    tangent_emit(state.index, {'phase': 'skipped', 'anchor_index': state.index,
                                              'reason': 'tangent_and_next_map_time_budget',
                                              'full_nonlinear_evaluation': False})
            if sigma is None:
                if previous is not None and previous.sigma_input is not None:
                    sigma, details = propose_channel_anderson1(
                        previous.sigma_input, previous.evaluation.map_sigma,
                        state.sigma_input, state.evaluation.map_sigma, config.mixing,
                        acceleration=acceleration, absolute_floor=config.sigma_atol)
                    kind = 'anderson1'
                else:
                    details, kind = {'reason': 'insufficient_nonzero_history'}, 'linear_proposal_fallback'
                if sigma is None:
                    sigma = mix_channel_spectra(state.sigma_input, state.evaluation.map_sigma, config.mixing)
                    kind = 'linear_proposal_fallback'
                proposal = dict(details, kind=kind, anchor_index=state.index,
                                previous_index=previous.index if previous is not None else None)
        except _NUMERICAL_ERRORS as exc:
            return result('proposal_error', f'{type(exc).__name__}: {exc}')
    return result('max_evaluations')


__all__ = ['ReferenceFeedbackResult', 'feasible_reference_proposal',
           'propose_analytic_feedback', 'run_reference_feedback']
