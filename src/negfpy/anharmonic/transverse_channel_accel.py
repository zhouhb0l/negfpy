"""Depth-one native R/C/D proposals with explicit evaluated-trial selection.

The same real coefficient and step clipping factor apply to all channels.  This
preserves linear R/D relations without reconstructing or projecting either
field.  Normalization affects the fit and merit only.  A proposal is never an
evaluated state, and a rejected trial never replaces the selected anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import fsum, sqrt
from time import monotonic

import numpy as np

from .transverse_channel_feedback import (
    _COMPONENTS,
    ChannelSCBAConfig,
    ChannelSCBAPlan,
    ChannelSCBARecord,
    ChannelSCBAState,
    ChannelSpectrum,
    _integer,
    _real,
    _record,
    _spectrum,
    evaluate_channel_born_map,
    mix_channel_spectra,
    plan_channel_scba,
)


@dataclass(frozen=True, slots=True)
class ChannelAccelerationConfig:
    mode: str = 'anderson1'
    coefficient_limit: float = 5.0
    trust_radius: float = .25
    rank_rtol: float = 1e-10

    def __post_init__(self):
        if self.mode not in ('anderson1', 'linear-control'):
            raise ValueError('mode must be anderson1 or linear-control.')
        for name in ('coefficient_limit', 'trust_radius', 'rank_rtol'):
            if _real(getattr(self, name), name) <= 0:
                raise ValueError(f'{name} must be positive.')
        if self.rank_rtol >= 1:
            raise ValueError('rank_rtol must be less than one.')


@dataclass(frozen=True, slots=True)
class ChannelSelectionPolicy:
    """Rank evaluated proposals without changing the physical/convergence gates.

    The default reproduces the original componentwise veto.  The opt-in policy
    permits a bounded tradeoff against the current anchor only when the actual
    frozen-scale merit improves by its specified factor.  It adds no cumulative
    envelope; the caller's fresh-map budget remains the bound on this experiment.
    """
    mode: str = 'component-nonincreasing'
    component_growth_limit: float = 1.0
    merit_ratio_limit: float = 1.0

    def __post_init__(self):
        if self.mode not in ('component-nonincreasing', 'bounded-tradeoff'):
            raise ValueError('Unknown evaluated-trial selection policy.')
        if _real(self.component_growth_limit, 'component_growth_limit') < 1:
            raise ValueError('component_growth_limit must be at least one.')
        if not 0 < _real(self.merit_ratio_limit, 'merit_ratio_limit') <= 1:
            raise ValueError('merit_ratio_limit must lie in (0,1].')
        if self.mode == 'component-nonincreasing' and (
                self.component_growth_limit != 1 or self.merit_ratio_limit != 1):
            raise ValueError('The original selection policy requires unit limits.')


@dataclass(frozen=True, slots=True)
class ChannelAccelerationPlan:
    fixed_plan: ChannelSCBAPlan
    additional_retained_bytes: int
    block_workspace_bytes: int
    peak_owned_bytes: int
    nominal_caller_bytes: int
    maximum_contraction_work: int
    maximum_dyson_evaluations: int


def plan_channel_accel(n_q, n_frequency, full_ndof, active_ndof, left_ndof, right_ndof,
                       *, config=None, acceleration=None):
    """Conservative payload bound; no full residual/difference/QR stacks.

    Six extra spectrum payloads cover detached seed input/image, the extra
    previous/selected input-image pair and a rejected trial's retained Green.
    Some share ownership in normal execution; no such sharing is credited.
    Excludes caller arrays and Python/library scratch, as the fixed-map plan
    does.  This is not an RSS guarantee.
    """
    config = ChannelSCBAConfig() if config is None else config
    acceleration = ChannelAccelerationConfig() if acceleration is None else acceleration
    if not isinstance(acceleration, ChannelAccelerationConfig):
        raise TypeError('acceleration must be ChannelAccelerationConfig.')
    fixed = plan_channel_scba(n_q, n_frequency, full_ndof, active_ndof, left_ndof,
                             right_ndof, config=config, has_initial_sigma=True)
    extra = 6 * fixed.input_sigma_bytes
    scratch = 16 * 32 * max(fixed.map_plan.active_ndof, 1)**2
    return ChannelAccelerationPlan(
        fixed, extra, scratch, fixed.peak_owned_bytes + extra + scratch,
        fixed.nominal_caller_bytes + 2 * fixed.input_sigma_bytes,
        fixed.maximum_contraction_work, fixed.maximum_dyson_evaluations)


@dataclass(frozen=True, slots=True, eq=False)
class ChannelAccelerationResult:
    state: ChannelSCBAState | None
    last_evaluated_state: ChannelSCBAState | None
    history: tuple[ChannelSCBARecord, ...]
    decisions: tuple[dict, ...]
    converged: bool
    stop_reason: str
    failure_reason: str | None
    plan: ChannelAccelerationPlan


def _copy(value, shape, name):
    _spectrum(value, shape, name)
    if any(getattr(value, key).dtype != np.dtype('complex128') for key in _COMPONENTS):
        raise ValueError(f'{name} must have complex128 fields.')
    arrays = [np.array(getattr(value, key), copy=True, order='C') for key in _COMPONENTS]
    for array in arrays:
        array.setflags(write=False)
    return ChannelSpectrum(*arrays)


def channel_scales(state, raw, absolute_floor):
    """Per-channel maximum magnitudes; freeze these across a trial assessment."""
    floor = max(_real(absolute_floor, 'absolute_floor'), np.finfo(float).tiny)
    values = []
    for key in _COMPONENTS:
        maximum = floor
        for index in np.ndindex(state.retarded.shape[:2]):
            for spectrum in (state, raw):
                block = getattr(spectrum, key)[index]
                if block.size:
                    maximum = max(maximum, float(np.max(np.abs(block))))
        values.append(maximum)
    return tuple(values)


def _dot_blocks(first, second, scales, shape):
    """Real weighted inner product, streamed over one q/frequency block."""
    count = 3 * int(np.prod(shape))
    if not count:
        return 0.0
    terms = []
    for key, scale in zip(_COMPONENTS, scales, strict=True):
        for index in np.ndindex(shape[:2]):
            a, b = first(key, index) / scale, second(key, index) / scale
            terms.append(float(np.vdot(a, b).real) / count)
    result = fsum(terms)
    if not np.isfinite(result):
        raise ValueError('Nonfinite streamed channel inner product.')
    return result


def channel_merit(state, raw, scales):
    """Scaled RMS of the unmixed independent-channel defect; nongating merit."""
    residual = lambda key, index: getattr(raw, key)[index] - getattr(state, key)[index]
    return sqrt(max(0., _dot_blocks(residual, residual, scales, state.retarded.shape)))


def _diagnostic_ratio(value, reference):
    """JSON-safe scalar ratio; None denotes an unbounded/overflowed ratio."""
    if reference == 0:
        return 1.0 if value == 0 else None
    ratio = value / reference
    return ratio if np.isfinite(ratio) else None


def propose_channel_anderson1(previous_input, previous_image, state, raw, mixing,
                              *, acceleration=None, absolute_floor=1e-10):
    """Return a read-only UNEVALUATED proposal, or an explicit cheap rejection.

    Type-II step alpha*f - gamma*(delta_x + alpha*delta_f), with one real gamma
    from the three normalized channels.  At most block-size temporaries are
    used beyond the output.  The whole step is clipped by one real factor.
    """
    acceleration = ChannelAccelerationConfig() if acceleration is None else acceleration
    if not isinstance(acceleration, ChannelAccelerationConfig):
        raise TypeError('acceleration must be ChannelAccelerationConfig.')
    alpha = _real(mixing, 'mixing')
    if not 0 < alpha <= 1:
        raise ValueError('mixing must lie in (0,1].')
    shape = state.retarded.shape
    for value, name in ((previous_input, 'previous_input'),
                        (previous_image, 'previous_image'), (state, 'state'), (raw, 'raw')):
        _spectrum(value, shape, name)
    scales = channel_scales(state, raw, absolute_floor)
    residual = lambda key, index: getattr(raw, key)[index] - getattr(state, key)[index]
    old_residual = lambda key, index: (getattr(previous_image, key)[index]
                                      - getattr(previous_input, key)[index])
    delta_f = lambda key, index: residual(key, index) - old_residual(key, index)
    denominator = _dot_blocks(delta_f, delta_f, scales, shape)
    reference_squared = max(_dot_blocks(residual, residual, scales, shape),
                            _dot_blocks(old_residual, old_residual, scales, shape))
    details = {'scales': list(scales), 'denominator': denominator,
               'residual_reference_squared': reference_squared,
               'rank_rtol': acceleration.rank_rtol, 'coefficient': None,
               'reference_merit': sqrt(max(0., _dot_blocks(residual, residual, scales, shape)))}
    if denominator <= max(np.finfo(float).tiny,
                          acceleration.rank_rtol**2 * reference_squared):
        return None, dict(details, reason='degenerate_history')
    raw_coefficient = _dot_blocks(delta_f, residual, scales, shape) / denominator
    if not np.isfinite(raw_coefficient):
        return None, dict(details, reason='nonfinite_coefficient')
    coefficient = float(np.clip(raw_coefficient, -acceleration.coefficient_limit,
                                acceleration.coefficient_limit))
    # A small damping alpha makes the useful secant coefficient large even
    # for a benign affine contraction. Rejecting that finite coefficient
    # would starve acceleration; bound the common scalar instead.
    details.update(raw_coefficient=raw_coefficient, coefficient=coefficient,
                   bounded_coefficient=coefficient,
                   coefficient_clipped=coefficient != raw_coefficient)

    def step(key, index):
        delta_x = getattr(state, key)[index] - getattr(previous_input, key)[index]
        return alpha * residual(key, index) - coefficient * (delta_x + alpha * delta_f(key, index))

    step_norm = sqrt(max(0., _dot_blocks(step, step, scales, shape)))
    state_values = lambda key, index: getattr(state, key)[index]
    raw_values = lambda key, index: getattr(raw, key)[index]
    reference_norm = sqrt(max(_dot_blocks(state_values, state_values, scales, shape),
                              _dot_blocks(raw_values, raw_values, scales, shape), 0.))
    bound = acceleration.trust_radius * reference_norm
    clip = min(1., bound / step_norm) if step_norm else 1.
    arrays = []
    with np.errstate(over='raise', invalid='raise'):
        for key in _COMPONENTS:
            output = np.empty(shape, dtype=np.complex128)
            for index in np.ndindex(shape[:2]):
                output[index] = getattr(state, key)[index] + clip * step(key, index)
                if not np.isfinite(output[index]).all():
                    raise ValueError('Nonfinite native Anderson proposal.')
            output.setflags(write=False)
            arrays.append(output)
    details.update(reason='proposed', unbounded_step_norm=step_norm,
                   reference_field_norm=reference_norm, scaled_step_bound=bound,
                   clip_factor=clip, scaled_step_norm=clip * step_norm,
                   trust_radius=acceleration.trust_radius)
    return ChannelSpectrum(*arrays), details


def run_channel_accel(grid, vertex, dynamical, left, right, *, seed_input, seed_mapped,
                      sigma_initial=None, seed_index=15, scale=None, g=.03,
                      config=None, acceleration=None, selection=None,
                      on_start=None, on_evaluation=None):
    """Run a bounded number of fresh maps, including every rejected trial.

    The seed pair is historical evaluated input plus UNEVALUATED Born image,
    never an accepted restart.  A plain mixed warm-up is evaluated first.
    Only accelerated trials use the explicit selection policy against a frozen
    selected anchor; linear steps remain the explicit fallback.  The original
    six scientific screens alone determine convergence.  Hard numerical map
    errors stop without retry.
    """
    started = monotonic()
    config = ChannelSCBAConfig() if config is None else config
    acceleration = ChannelAccelerationConfig() if acceleration is None else acceleration
    selection = ChannelSelectionPolicy() if selection is None else selection
    seed_index = _integer(seed_index, 'seed_index')
    if not isinstance(config, ChannelSCBAConfig):
        raise TypeError('config must be ChannelSCBAConfig.')
    if not isinstance(acceleration, ChannelAccelerationConfig):
        raise TypeError('acceleration must be ChannelAccelerationConfig.')
    if not isinstance(selection, ChannelSelectionPolicy):
        raise TypeError('selection must be ChannelSelectionPolicy.')
    if any(cb is not None and not callable(cb) for cb in (on_start, on_evaluation)):
        raise TypeError('Callbacks must be callable.')
    shape = (vertex.mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
    plan = plan_channel_accel(vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof,
                              vertex.ndof, len(left.device_indices), len(right.device_indices),
                              config=config, acceleration=acceleration)
    if plan.peak_owned_bytes > config.max_bytes:
        raise ValueError('Channel acceleration allocation plan exceeds max_bytes.')
    if plan.fixed_plan.map_plan.contraction_work > config.max_contraction_work:
        raise ValueError('Channel acceleration map exceeds max_contraction_work.')
    if plan.fixed_plan.map_plan.dyson_evaluations > config.max_dyson_evaluations:
        raise ValueError('Channel acceleration map exceeds max_dyson_evaluations.')
    previous = (_copy(seed_input, shape, 'seed_input'), _copy(seed_mapped, shape, 'seed_mapped'))
    sigma = (mix_channel_spectra(*previous, config.mixing) if sigma_initial is None
             else _copy(sigma_initial, shape, 'sigma_initial'))
    proposal = {'kind': 'linear_warmup', 'anchor_index': seed_index,
                'previous_index': None, 'reason': 'fresh_linear_warmup'}
    state = latest = None
    history, decisions = [], []
    estimate = config.initial_map_estimate_seconds

    def result(reason, failure=None, converged=False):
        return ChannelAccelerationResult(state, latest, tuple(history), tuple(decisions),
                                         converged, reason, failure, plan)

    for offset in range(config.max_evaluations):
        if config.max_seconds - (monotonic() - started) <= (
                config.time_reserve_seconds + config.next_map_safety_factor * estimate):
            return result('max_seconds')
        index = seed_index + 1 + offset
        if on_start is not None:
            on_start(index, sigma, dict(proposal))
        try:
            map_started = monotonic()
            evaluation = evaluate_channel_born_map(
                grid, vertex, dynamical, left, right, sigma_input=sigma, scale=scale, g=g,
                workers=config.workers, external_tile=config.external_tile, bin_tile=config.bin_tile,
                max_bytes=plan.fixed_plan.map_plan.peak_owned_bytes,
                max_contraction_work=config.max_contraction_work,
                max_dyson_evaluations=config.max_dyson_evaluations,
                identity_rtol=config.identity_rtol, identity_atol=config.identity_atol)
            duration = monotonic() - map_started
            record = _record(index, sigma, evaluation, duration, monotonic() - started, config)
            trial = ChannelSCBAState(index, sigma, evaluation, record)
            reference = state
            nonincrease, reference_merit, trial_merit, scales = {}, None, None, None
            reference_fro, trial_fro, growth_ratio, within_limit = {}, {}, {}, {}
            merit_limit = merit_ratio = merit_within_limit = None
            selected, reason = True, 'linear_evaluation_selected'
            if proposal['kind'] == 'anderson1':
                scales = tuple(proposal['scales'])
                reference_merit = proposal['reference_merit']
                reference_fro = {key: reference.record.sigma_residuals[key].error_frobenius
                                 for key in ('retarded', 'lesser', 'greater')}
                trial_fro = {key: record.sigma_residuals[key].error_frobenius
                             for key in reference_fro}
                nonincrease = {key: trial_fro[key] <= reference_fro[key] for key in reference_fro}
                growth_ratio = {key: _diagnostic_ratio(trial_fro[key], reference_fro[key])
                                for key in reference_fro}
                within_limit = {key: trial_fro[key] <= selection.component_growth_limit * reference_fro[key]
                                for key in reference_fro}
                if record.converged:
                    # The original six convergence screens take priority.
                    # An auxiliary merit failure must not veto a true pass.
                    selected, reason = True, 'fully_converged_trial'
                else:
                    trial_merit = channel_merit(sigma, evaluation.map_sigma, scales)
                    merit_limit = selection.merit_ratio_limit * reference_merit
                    merit_ratio = _diagnostic_ratio(trial_merit, reference_merit)
                    merit_within_limit = trial_merit <= merit_limit
                    selected = bool(merit_within_limit and all(within_limit.values())
                                    and record.current_passed)
                    passed_reason = ('merit_and_unmixed_frobenius_passed'
                                     if selection.mode == 'component-nonincreasing' else
                                     'merit_and_bounded_frobenius_passed')
                    reason = passed_reason if selected else 'trial_selection_rejected'
            if selected:
                if state is not None:
                    previous = (state.sigma_input, state.evaluation.map_sigma)
                state = trial
            decision = {'index': index, 'accepted_as_anchor': selected,
                            'selected_anchor_index': state.index, 'kind': proposal['kind'],
                            'reference_anchor_index': proposal['anchor_index'], 'reason': reason,
                            'reference_merit': reference_merit, 'trial_merit': trial_merit,
                            'merit_scales': list(scales) if scales is not None else None,
                            'merit_evaluated': trial_merit is not None,
                            'unmixed_frobenius_nonincrease': nonincrease,
                            'selection_policy': selection.mode,
                            'component_growth_limit': selection.component_growth_limit,
                            'merit_ratio_limit': selection.merit_ratio_limit,
                            'unmixed_frobenius_reference': reference_fro,
                            'unmixed_frobenius_trial': trial_fro,
                            'unmixed_frobenius_growth_ratio': growth_ratio,
                            'unmixed_frobenius_within_limit': within_limit,
                            'merit_limit': merit_limit, 'merit_ratio': merit_ratio,
                            'merit_within_limit': merit_within_limit,
                            'current_passed': record.current_passed,
                            'fully_converged': record.converged}
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return result('evaluation_error', f'{type(exc).__name__}: {exc}')
        # Publish only after the complete evaluation and selection metadata
        # exist. A selection-arithmetic failure leaves an unmatched start,
        # rather than relabeling a trial as completed without its callback.
        latest = trial
        history.append(record)
        decisions.append(decision)
        estimate = max(estimate, duration)
        if on_evaluation is not None:
            on_evaluation(latest, dict(decision))
        if record.converged:
            return result('fixed_point', converged=True)
        if offset + 1 == config.max_evaluations:
            break
        try:
            if acceleration.mode == 'linear-control' or not selected:
                sigma = mix_channel_spectra(state.sigma_input, state.evaluation.map_sigma,
                                            config.mixing)
                proposal = {'kind': 'linear_control' if selected else 'linear_fallback',
                                'anchor_index': state.index, 'previous_index': None,
                                'reason': 'fixed_linear_control' if selected else 'rejected_trial_rollback'}
            else:
                sigma, details = propose_channel_anderson1(
                    *previous, state.sigma_input, state.evaluation.map_sigma, config.mixing,
                    acceleration=acceleration, absolute_floor=config.sigma_atol)
                kind = 'anderson1'
                if sigma is None:
                    sigma = mix_channel_spectra(state.sigma_input, state.evaluation.map_sigma,
                                                config.mixing)
                    kind = 'linear_proposal_fallback'
                proposal = dict(details, kind=kind, anchor_index=state.index,
                                previous_index=(seed_index if state.index == seed_index + 1
                                                else decisions[-2]['selected_anchor_index']))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return result('proposal_error', f'{type(exc).__name__}: {exc}')
    return result('max_evaluations')


__all__ = ['ChannelAccelerationConfig', 'ChannelAccelerationPlan',
           'ChannelAccelerationResult', 'ChannelSelectionPolicy',
           'channel_merit', 'channel_scales', 'plan_channel_accel',
           'propose_channel_anderson1', 'run_channel_accel']
