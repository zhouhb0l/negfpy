"""Opt-in final spectral-moment diagnostics for a retained research input.

This adds evidence without changing SCBA convergence or release decisions.
The integration concerns the declared interpolation inside the original grid;
passing embedded estimates is not a bound on global frequency accuracy.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
from threadpoolctl import threadpool_limits

from ._moment_quadrature import _validate, run_full_domain_quadrature
from ._provenance import _update_array
from .interacting_pointwise import PointwiseInteracting


def _field_hashes(sigma):
    result = {}
    for name in ('lesser', 'greater', 'retarded'):
        digest = hashlib.sha256()
        _update_array(digest, getattr(sigma, name))
        result[name] = digest.hexdigest()
    return result


class ResearchMomentDiagnostic:
    """Bounded GL8/16 check, with final leaf evidence retained on this object.

    Pass a separate instance to each ``solve_scba(..., research_map=...,
    research_moment=...)`` invocation. ``report`` is also copied into result
    metadata; ``arrays`` contains all final leaves for optional archiving.
    Reusing an instance clears its previous evidence at the start of the run.

    Only a completed fresh research final map enables evaluation. A skipped
    final map produces an explicit not-evaluated report. A budget stop is a
    numerical outcome, never a pass. Other exceptions propagate. The caller
    must supply an outer process timeout for a stalled pointwise solve.
    """

    def __init__(self, *, max_points=40000, max_seconds=1800., max_workers=8,
                 max_leaves=1024, rtol=.002, atol=1e-8,
                 moment_rtol=.005, interpolation_policy='equilibrium-broadening',
                 emit=None):
        self._options = dict(max_points=max_points, max_seconds=max_seconds,
                             max_workers=max_workers, max_leaves=max_leaves,
                             rtol=rtol, atol=atol)
        self._emit = (lambda **row: None) if emit is None else emit
        _validate(np.array([-1., 0., 1.]), 1, lambda w: None, self._emit,
                  max_points, max_seconds, max_workers, max_leaves, rtol, atol,
                  time.monotonic)
        if not math.isfinite(moment_rtol) or moment_rtol < 0:
            raise ValueError('moment_rtol must be finite and non-negative.')
        if interpolation_policy not in ('equilibrium-broadening', 'linear-keldysh'):
            raise ValueError('Unknown self-energy interpolation policy.')
        self._moment_rtol = float(moment_rtol)
        self._policy = interpolation_policy
        self.report = None
        self.arrays = None

    @property
    def identity(self):
        return dict(schema='negfpy.research.final-moment.v1',
                    algorithm='full-domain-adaptive-gl8-gl16',
                    interpolation_policy=self._policy, moment_rtol=self._moment_rtol,
                    **self._options)

    def prepare(self, problem, config):
        """Clear old evidence and validate controls before entering SCBA."""
        self.report = self.arrays = None
        _validate(problem.grid.values, problem.ndof, lambda w: None, self._emit,
                  **self._options, clock=time.monotonic)
        if config.eta_device != 0:
            raise ValueError('Pointwise moment diagnostics require eta_device=0.')
        if self._policy == 'equilibrium-broadening' and (
                problem.left.temperature <= 0
                or problem.left.temperature != problem.right.temperature):
            raise ValueError('Equilibrium moment interpolation requires equal positive temperatures.')

    def skip(self, *, coarse_residual, reason):
        self.arrays = None
        self.report = dict(identity=self.identity, evaluated=False, reason=reason,
                           coarse_spectral_moment_residual=float(coarse_residual),
                           quadrature=None, moment_screen=None, passed=False,
                           changes_solver_acceptance=False, release_certified=False)
        return self.report

    def evaluate(self, problem, config, sigma, *, coarse_residual):
        """Integrate this fixed input; no map evaluation or field update occurs."""
        self.prepare(problem, config)
        if not math.isfinite(coarse_residual) or coarse_residual < 0:
            raise ValueError('coarse_residual must be finite and non-negative.')
        before = _field_hashes(sigma)
        pointwise = PointwiseInteracting(problem, config, sigma, policy=self._policy)

        def integrand(omega):
            retarded = pointwise.evaluate(omega).retarded
            return float(omega) / (2 * np.pi) * 1j * (retarded - retarded.conj().T)

        with threadpool_limits(limits=1, user_api='blas'):
            quadrature, arrays = run_full_domain_quadrature(
                problem.grid.values, integrand, problem.ndof,
                emit=self._emit, **self._options)
        after = _field_hashes(sigma)
        if before != after:
            raise RuntimeError('Input self-energy changed during the moment diagnostic.')
        residual = (quadrature['moments']['full_domain_moment']['relative_max_residual']
                    if quadrature['complete_partition'] else None)
        screen = dict(threshold=self._moment_rtol, residual=residual,
                      passed=bool(residual is not None and residual <= self._moment_rtol))
        self.arrays = arrays
        self.report = dict(identity=self.identity, evaluated=True,
                           coarse_spectral_moment_residual=float(coarse_residual),
                           input_field_sha256=before, input_field_unchanged=True,
                           quadrature=quadrature, moment_screen=screen,
                           passed=bool(quadrature['sampled_tolerance_met'] and screen['passed']),
                           changes_solver_acceptance=False, release_certified=False,
                           global_frequency_accuracy_established=False,
                           biased_transport_conservation_established=False)
        return self.report


__all__ = ['ResearchMomentDiagnostic']
