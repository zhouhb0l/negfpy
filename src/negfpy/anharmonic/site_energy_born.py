"""Common-Green, fixed-state comparison of declared local-energy boundaries.

Only the stated low-frequency Born terms are evaluated. The fitted vertices
are explicit research surrogates; this does not advance or accept an SCBA state.
"""
from __future__ import annotations

from dataclasses import asdict
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .transverse_acoustic_sector import _blocks, _domains, _translations
from .transverse_channel_feedback import _finish, _local_checks, _measure
from .transverse_compact_contacts import CompactTransverseContact
from .transverse_native_zero_diagnostic import (
    _adjoint,
    _assemble,
    _contexts,
    _encoded,
    _finite,
    _integer,
    _spectrum,
    plan_native_zero_diagnostic_dimensions,
)
from .transverse_packed import PackedTransverseCubicVertex, contract_packed_fft_bins
from .transverse_q import difference_q_indices

_TERMS = ('CC', 'DD_over_4', 'CD', 'DC')


def _embed(value, active, n):
    result = np.zeros((n, n), dtype=np.complex128)
    result[np.ix_(active, active)] = value
    return result


def _audit(vertex, translations, central, outer, admit):
    """Gather full-normalized translations without renormalizing the support."""
    active, n = vertex.active_dofs, vertex.full_ndof
    restricted = translations[active]
    rows = []
    for leg in range(3):
        for k in range(vertex.mesh.n_qpoints):
            admit()
            q, p = ((0, k), (k, 0), (k, k))[leg]
            matrices = np.tensordot(vertex.coefficients[q, p], restricted, axes=((leg,), (0,)))
            _finite(matrices, 'translation contraction')
            for direction in range(3):
                rows.append({'zero_momentum_leg': leg, 'other_q_index': k,
                    'direction': direction,
                    'blocks': _blocks(_embed(matrices[:, :, direction], active, n), central, outer)})
    return {'normalization': 'full-device massweighted translation; support is gathered without renormalization',
        'support_translation_gram': (restricted.T@restricted).tolist(),
        'diagnostic_only': True, 'residual_is_not_execution_gate': True, 'rows': rows}


def study_site_energy_born(grid, vertices, dynamical, left, right, sigma_input,
        translations, central_dofs, *, g=.03, internal_half_width=2, workers=8,
        external_tile=16, bin_tile=1, max_bytes=4*1024**3,
        max_contraction_work=1_600_000_000_000, max_dyson_evaluations=20,
        deadline_monotonic=None, time_reserve_seconds=0,
        identity_rtol=1e-8, identity_atol=1e-11, emit=None):
    """Use one full Green calculation for every declared vertex comparison."""
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(vertices, dict) or not vertices:
        raise TypeError('A signed grid and nonempty named packed-vertex dictionary are required.')
    if not all(isinstance(key, str) and isinstance(v, PackedTransverseCubicVertex)
               for key, v in vertices.items()):
        raise TypeError('Named packed vertices are required.')
    first = next(iter(vertices.values()))
    mesh, nq, nw, n = first.mesh, first.mesh.n_qpoints, grid.n_points, first.full_ndof
    if any(v.full_ndof != n or v.mesh.shape != mesh.shape or not v.ndof for v in vertices.values()):
        raise ValueError('All vertices need the same full device/mesh and nonempty exact support.')
    limit = _integer(max_bytes, 'max_bytes')
    plans = {key: plan_native_zero_diagnostic_dimensions(nq, nw, n, vertex.ndof,
        half_width=internal_half_width, workers=workers, external_tile=external_tile,
        bin_tile=bin_tile) for key, vertex in vertices.items()}
    full_plan = plan_native_zero_diagnostic_dimensions(nq, nw, n, n,
        half_width=internal_half_width, workers=workers, external_tile=external_tile, bin_tile=bin_tile)
    work = sum(v.contraction_work for v in plans.values())
    # Caller-owned vertices and memory-mapped inputs are excluded. Retain full
    # R/C/D plus a model's gathered C/D and kernel workspace, never three solves.
    peak = 3*16*nq*full_plan.n_low_bins*n*n + max(v.peak_owned_bytes for v in plans.values())
    peak += 2*16*nq*full_plan.n_low_bins*n*n
    if peak > limit or work > _integer(max_contraction_work, 'max_contraction_work'):
        raise ValueError('Common-Green diagnostic exceeds numeric memory/work admission.')
    if full_plan.dyson_evaluations > _integer(max_dyson_evaluations, 'max_dyson_evaluations'):
        raise ValueError('Common-Green diagnostic exceeds its inverse budget.')
    for value, name in ((g, 'g'), (identity_rtol, 'identity_rtol'),
            (identity_atol, 'identity_atol'), (time_reserve_seconds, 'time_reserve_seconds')):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f'{name} must be finite and nonnegative.')
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError('Deadline must be finite.')

    def admit():
        if deadline_monotonic is not None and monotonic()+time_reserve_seconds >= deadline_monotonic:
            raise TimeoutError('Common-Green diagnostic time admission exhausted.')

    admit()
    central, outer = _domains(central_dofs, n)
    t = _translations(translations, n, identity_rtol, identity_atol)
    raw = _encoded(dynamical, (nq, n, n), 'dynamical')
    fields = _spectrum(sigma_input, (nq, nw, n, n), 'sigma_input')
    contacts = []
    for contact in (left, right):
        if not isinstance(contact, CompactTransverseContact) or contact.full_ndof != n or contact.mesh.shape != mesh.shape or contact.grid.n_points != nw or contact.grid.step != grid.step:
            raise ValueError('Contact dimensions, mesh or grid differ.')
        contacts.append((np.ix_(contact.device_indices, contact.device_indices), contact.spectrum))
    z, width = grid.zero_index, full_plan.half_width
    indices = tuple(range(z-width, z+width+1))
    for q in range(nq):
        _finite(raw[q], 'dynamical')
        for w in indices:
            for value in fields:
                _finite(value[q, w], 'native low-frequency input')
            for _, contact in contacts:
                for name in ('retarded', 'lesser', 'greater'):
                    _finite(getattr(contact, name)[q, w], 'contact low-frequency input')
    retained = [np.empty((nq, len(indices), n, n), np.complex128) for _ in range(3)]
    all_dofs = np.ix_(np.arange(n), np.arange(n))
    points, models = [], {}
    source_metrics, green_metrics, exported = {}, {}, {}
    routes = difference_q_indices(mesh, max_bytes=limit)
    with threadpool_limits(limits=1, user_api='blas'):
        for q in range(nq):
            for local, w in enumerate(indices):
                admit()
                operator, cs, ds = _assemble(grid.values[w], raw[q], contacts, all_dofs, fields, q, w, np.complex128)
                for value in (operator, cs, ds):
                    _finite(value, 'assembled operator/source')
                _measure(source_metrics, 'raw_source_identity', ds, _adjoint(operator)-operator, (q, w))
                _finish(source_metrics, identity_rtol, identity_atol, 'raw source')
                r = np.linalg.inv(operator)
                c, d = r@cs@_adjoint(r), r@ds@_adjoint(r)
                for target, value in zip(retained, (r, c, d), strict=True):
                    _finite(value, 'fresh native Green product')
                    target[q, local] = value
                _local_checks(green_metrics, exported, r, c, d, (q, w))
                _finish(green_metrics, identity_rtol, identity_atol, 'independent native Green')
                points.append({'q_index': q, 'frequency_index': w, 'frequency': float(grid.values[w]),
                    'statistical_green': _blocks(c, central, outer),
                    'spectral_green': _blocks(d, central, outer),
                    'context': _contexts(operator, cs, ds, r, c, d)})
        # Same original reality/reflection checks, restricted to the paired
        # twenty-point scope. No full-grid current or convergence test is implied.
        for q in range(nq):
            opposite = mesh.negative_index(q)
            for local, w in enumerate(indices):
                r, c, d = (value[q, local] for value in retained)
                reverse_r, reverse_c, reverse_d = (value[opposite, len(indices)-1-local] for value in retained)
                _measure(green_metrics, 'retarded_reality', r, reverse_r.conj(), (q, w))
                _measure(green_metrics, 'statistical_reflection', c, reverse_c.T, (q, w))
                _measure(green_metrics, 'spectral_reflection', d, -reverse_d.T, (q, w))
        checks = {'raw_source': _finish(source_metrics, identity_rtol, identity_atol, 'raw source'),
            'native_green': _finish(green_metrics, identity_rtol, identity_atol, 'native Green'),
            'exports_diagnostic_only': _finish(exported, identity_rtol, identity_atol, 'exports', gate=False)}
        if emit is not None:
            emit('site_energy_common_green_complete', {'points': len(points)})
        for key, vertex in vertices.items():
            admit()
            if emit is not None:
                emit('site_energy_model_start', {'model': key, 'active_dofs': vertex.ndof})
            active = vertex.active_dofs
            audit = _audit(vertex, t, central, outer, admit)
            cvalues = retained[1][:, :, active[:, None], active]
            dvalues = retained[2][:, :, active[:, None], active]
            groups = {offset: {term: np.zeros((vertex.ndof, vertex.ndof), complex) for term in _TERMS}
                      for offset in range(width+1)}
            gamma_zero = {}
            for p in range(nq):
                partner = int(routes[0, p])
                c1, d1 = cvalues[p], dvalues[p]
                c2, d2 = cvalues[partner, ::-1], dvalues[partner, ::-1]
                for term, a, b in (('CC', c1, c2), ('DD_over_4', d1, d2), ('CD', c1, d2), ('DC', d1, c2)):
                    admit()
                    values = contract_packed_fft_bins(vertex.coefficients[0, p], a, b,
                        external_tile=external_tile, workers=workers, bin_tile=bin_tile,
                        max_bytes=limit, max_contraction_work=max_contraction_work)
                    for local, w in enumerate(indices):
                        weighted = values[local]*(.25 if term == 'DD_over_4' else 1.)
                        groups[abs(w-z)][term] += weighted
                        if p == 0 and w == z:
                            gamma_zero[term] = weighted.copy()
            # Match production arithmetic: raw Born prefactor, then g squared.
            born_factor = .5j*grid.integration_weight/nq
            for group in (*groups.values(), gamma_zero):
                for term in _TERMS:
                    group[term] *= born_factor
                    group[term] *= float(g)**2
                    _finite(group[term], 'partial Born arithmetic')
            totals = {term: sum(group[term] for group in groups.values()) for term in _TERMS}

            def records(values, active=active):
                out = {term: _blocks(_embed(values[term], active, n), central, outer) for term in _TERMS}
                out['statistical'] = _blocks(_embed(values['CC']+values['DD_over_4'], active, n), central, outer)
                out['spectral'] = _blocks(_embed(values['CD']+values['DC'], active, n), central, outer)
                return out

            total_records = records(totals)
            models[key] = {'active_dofs_count': vertex.ndof, 'full_ndof': n,
                'internal_green_support': 'exact vertex endpoint union gathered from common full-device Green; no magnitude crop',
                'translation_audit': audit, 'plan': asdict(plans[key]),
                'born': {'external_q_index': 0, 'external_frequency_index': z,
                    'internal_frequency_indices': list(indices), 'partner_q_indices': routes[0].tolist(),
                    'g_squared_applied_after_raw_born_factor': True,
                    'partitions': {('zero' if offset == 0 else f'near{offset}'): records(values)
                                   for offset, values in groups.items()},
                    'term_totals': {term: total_records[term] for term in _TERMS},
                    'partial_total': {name: total_records[name] for name in ('statistical', 'spectral')},
                    'gamma_internal_zero': records(gamma_zero)}}
            if emit is not None:
                emit('site_energy_model_complete', {'model': key})
    return {'schema': 'negfpy.site-energy-common-green-born.v1',
        'plan': {'dyson_evaluations': full_plan.dyson_evaluations,
            'contraction_work': work, 'peak_owned_bytes': peak,
            'caller_vertices_and_arrays_excluded': True, 'rss_guarantee': False},
        'dyson_evaluations': len(points), 'points': points, 'checks': checks, 'models': models,
        'common_green_reused_without_modification': True, 'native_spectral_channel_retained': True,
        'partial_born_only': True, 'same_fitted_model_boundary_control': True,
        'mapped_saved_totals_or_far_frequency_remainders_used': False,
        'SCBA_evaluated': False, 'accepted_restart': False, 'accepted_field_update': False,
        'fit_residual_is_not_a_production_acceptance': True}
