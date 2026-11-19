"""Full-device reference/increment map retaining all physical guards.

The reference expansion uses exact Dyson/Born arithmetic with explicit
resource accounting. Full-device reference solves are streamed pointwise;
no full-device frequency stack or historical checkpoint is loaded. Signed
analytic tangent fields are never treated as physical covariance fields.
"""
from __future__ import annotations
from dataclasses import replace
from types import MappingProxyType
import numpy as np
from threadpoolctl import threadpool_limits
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.types import SolverScale
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_channels import StatisticalSpectral, retarded_from_spectral
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex, _array, _finite, _integer
from negfpy.anharmonic.transverse_solver import TransverseCurrents
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSpectrum, ChannelBornMapEvaluation, evaluate_channel_born_map,
    plan_channel_born_map, _real, _adjoint, _measure, _finish, _spectrum,
    _field_checks, _contact_checks, _local_checks, _zero_frequency_context,
)
from negfpy.anharmonic.transverse_channel_feedback import channel_exchange
from .increment_green import (
    prepare_reference, evaluate_increment, tangent_green, _sum as compensated_sum,
)
from .increment_born import difference_born, tangent_born, plan_born_expansion_dimensions

_COMPONENTS = ('retarded', 'statistical', 'spectral')
_DEFAULT_BYTES = 128 * 1024**3
_DEFAULT_WORK = 100_000_000_000_000


def plan_reference_map(n_q, n_frequency, full_ndof, active_ndof, left_ndof,
                       right_ndof, *, has_input_sigma=False, workers=8,
                       external_tile=16, bin_tile=1):
    """Conservative owned payload, separate from caller/reference storage/RSS.

    Includes the active C/D increment, an expanded Born transform/workspace,
    and twice the original per-point full-device workspace. Two Dyson solves
    are counted for each point. No full reference Green stack is retained.
    """
    base = plan_channel_born_map(n_q,n_frequency,full_ndof,active_ndof,
        left_ndof,right_ndof,has_input_sigma=has_input_sigma,workers=workers,
        external_tile=external_tile,bin_tile=bin_tile)
    born = plan_born_expansion_dimensions(n_q,n_frequency,active_ndof,
        operation='polarized_increment',workers=workers,external_tile=external_tile,bin_tile=bin_tile)
    increments = 2*16*n_q*n_frequency*active_ndof**2
    dyson = 2*base.dyson_workspace_bytes
    extra = increments + base.dyson_workspace_bytes + max(0,born.peak_owned_bytes-base.born_peak_bytes)
    return replace(base,born_peak_bytes=born.peak_owned_bytes,
        dyson_workspace_bytes=dyson,peak_owned_bytes=base.peak_owned_bytes+extra,
        contraction_work=born.contraction_work,dyson_evaluations=2*base.dyson_evaluations)


def evaluate_reference_map(
    grid, vertex, dynamical, left, right, *, sigma_input=None, scale=None, g=0.03,
    reference_sigma=None, reference_green=None, reference_mapped=None,
    workers=8, external_tile=16, bin_tile=1, max_bytes=_DEFAULT_BYTES,
    max_contraction_work=_DEFAULT_WORK, max_dyson_evaluations=2404,
    identity_rtol=1e-8, identity_atol=1e-11,
):
    """Fresh full-device map using a fixed same-physics reference expansion.

    Returns (canonical evaluation, arithmetic diagnostics). With no reference
    Green/mapped pair the exact-zero baseline uses the original full map.
    Reference arrays are caller-owned and must bind to the same grid, vertex,
    coupling, contacts, and reference_sigma; no cross-grid cache is accepted.
    All original physical guards are retained without changing tolerances.
    """
    if reference_green is None or reference_mapped is None:
        if reference_green is not None or reference_mapped is not None or reference_sigma is not None:
            raise ValueError('Supply both reference_green and reference_mapped.')
        evaluation = evaluate_channel_born_map(
            grid, vertex, dynamical, left, right, sigma_input=sigma_input,
            scale=scale, g=g, workers=workers, external_tile=external_tile,
            bin_tile=bin_tile, max_bytes=max_bytes,
            max_contraction_work=max_contraction_work,
            max_dyson_evaluations=max_dyson_evaluations,
            identity_rtol=identity_rtol, identity_atol=identity_atol)
        return evaluation, {'arithmetic': 'direct_baseline',
                            'dyson_solves': evaluation.plan.dyson_evaluations}
    if not isinstance(grid,SignedFrequencyGrid) or not isinstance(vertex,PackedTransverseCubicVertex):
        raise TypeError('Supply a signed grid and packed transverse vertex.')
    scale = SolverScale() if scale is None else scale
    if not isinstance(scale,SolverScale):
        raise TypeError('scale must be SolverScale.')
    coupling = _real(g,'g')
    squared = coupling*coupling
    if not np.isfinite(squared) or (coupling > 0 and squared == 0):
        raise ValueError('g squared overflow/underflow.')
    rtol,atol = _real(identity_rtol,'identity_rtol'),_real(identity_atol,'identity_atol')
    if rtol+atol == 0:
        raise ValueError('At least one identity tolerance must be positive.')
    limit,work,dyson_limit = (_integer(v,key) for v,key in
        ((max_bytes,'max_bytes'),(max_contraction_work,'max_contraction_work'),
         (max_dyson_evaluations,'max_dyson_evaluations')))
    mesh = vertex.mesh
    nq,nw,d,a = mesh.n_qpoints,grid.n_points,vertex.full_ndof,vertex.ndof
    for contact,side in ((left,'left'),(right,'right')):
        if not isinstance(contact,CompactTransverseContact):
            raise TypeError('Contacts must be CompactTransverseContact.')
        if (contact.full_ndof != d or contact.mesh.shape != mesh.shape
            or contact.grid.n_points != nw or contact.grid.step != grid.step
            or contact.contact.side != side):
            raise ValueError('Contact device/mesh/grid/side differs from the map.')
        size = len(contact.device_indices)
        for key in ('retarded','lesser','greater'):
            arr = _array(getattr(contact.spectrum,key),(nq,nw,size,size),side+'.'+key)
            _finite(arr,side+'.'+key)
    plan = plan_reference_map(nq,nw,d,a,len(left.device_indices),len(right.device_indices),
        has_input_sigma=sigma_input is not None,workers=workers,
        external_tile=external_tile,bin_tile=bin_tile)
    if plan.peak_owned_bytes > limit:
        raise ValueError('Channel map allocation plan exceeds max_bytes.')
    if plan.contraction_work > work:
        raise ValueError('Channel map exceeds max_contraction_work.')
    if plan.dyson_evaluations > dyson_limit:
        raise ValueError('Channel map exceeds max_dyson_evaluations.')
    raw = _array(dynamical,(nq,d,d),'dynamical')
    _finite(raw,'dynamical')
    harmonic = {}
    for q in range(nq):
        matrix = np.asarray(raw[q],dtype=np.complex128)
        if float(np.linalg.eigvalsh(matrix).min()) < -1e-12:
            raise ValueError('The harmonic device must be stable.')
        _measure(harmonic,'hermiticity',matrix,_adjoint(matrix),(q,))
        _measure(harmonic,'opposite_q',matrix,raw[mesh.negative_index(q)].conj(),(q,))
    checks = {'harmonic':_finish(harmonic,1e-10,1e-12,'dynamical')}
    shape = (nq,nw,a,a)
    _spectrum(reference_green, shape, 'reference_green')
    _spectrum(reference_mapped, shape, 'reference_mapped')
    if reference_sigma is not None:
        _spectrum(reference_sigma, shape, 'reference_sigma')
    if sigma_input is not None:
        _spectrum(sigma_input,shape,'sigma_input')
        checks['input_sigma'],checks['input_sigma_exports'] = _field_checks(
            mesh,grid,sigma_input,rtol,atol,'input Sigma')
        causal = {}
        if a:
            for q in range(nq):
                reconstructed = retarded_from_spectral(sigma_input.spectral[q],grid,
                    max_bytes=plan.reconstruction_workspace_bytes)
                _measure(causal,'causal_reconstruction',sigma_input.retarded[q],reconstructed,(q,))
                del reconstructed
        checks['input_causality'] = _finish(causal,rtol,atol,'input Sigma causality')
    checks['left'],checks['right'] = (_contact_checks(c,rtol,atol,name+' contact')
                                    for c,name in ((left,'left'),(right,'right')))
    active = np.ix_(vertex.active_dofs,vertex.active_dofs)
    supports = [np.ix_(np.asarray(c.device_indices),np.asarray(c.device_indices)) for c in (left,right)]
    arrays = [np.empty(shape,dtype=np.complex128) for _ in _COMPONENTS]
    full,full_exports,source_metrics,inverse_metrics = {},{},{},{}
    reference_cache, increment_metrics = {}, {}
    delta_arrays = [np.empty(shape, dtype=np.complex128) for _ in range(2)]
    zero_block = np.zeros((a,a), dtype=np.complex128)
    zero_context = None
    currents = np.zeros(3)
    identity = np.eye(d,dtype=np.complex128)
    with threadpool_limits(limits=1,user_api='blas'):
        for q in range(nq):
            for w,omega in enumerate(grid.values):
                operator = omega**2*identity-np.asarray(raw[q],dtype=np.complex128)
                cs,ds = np.zeros((d,d),dtype=np.complex128),np.zeros((d,d),dtype=np.complex128)
                for contact,support in zip((left,right),supports,strict=True):
                    r = contact.spectrum.retarded[q,w]
                    less,greater = contact.spectrum.lesser[q,w],contact.spectrum.greater[q,w]
                    operator[support] -= r
                    cs[support] += 0.5*(greater+less)
                    ds[support] += greater-less
                reference_operator = operator.copy()
                reference_cs, reference_ds = cs.copy(), ds.copy()
                if reference_sigma is not None:
                    reference_operator[active] -= reference_sigma.retarded[q,w]
                    reference_cs[active] += reference_sigma.statistical[q,w]
                    reference_ds[active] += reference_sigma.spectral[q,w]
                point_reference = prepare_reference(reference_operator, reference_cs, reference_ds)
                delta_fields = []
                for name in _COMPONENTS:
                    current_field = zero_block if sigma_input is None else getattr(sigma_input,name)[q,w]
                    reference_field = zero_block if reference_sigma is None else getattr(reference_sigma,name)[q,w]
                    delta_fields.append(current_field-reference_field)
                if sigma_input is not None:
                    operator[active] -= sigma_input.retarded[q,w]
                    cs[active] += sigma_input.statistical[q,w]
                    ds[active] += sigma_input.spectral[q,w]
                _finite(operator,'Dyson operator'); _finite(cs,'statistical source'); _finite(ds,'spectral source')
                _measure(source_metrics,'raw_source_identity',ds,_adjoint(operator)-operator,(q,w))
                try:
                    point = evaluate_increment(point_reference, *delta_fields,
                                               active_dofs=vertex.active_dofs)
                except ValueError as exc:
                    raise ValueError(f'Increment full-device Dyson solve at q={q}, w={w}: {exc}') from exc
                r, c, dgreen = point.retarded, point.statistical, point.spectral
                for name in _COMPONENTS:
                    _measure(reference_cache, name, getattr(point_reference,name)[active],
                             getattr(reference_green,name)[q,w], (q,w))
                for target, value in zip(delta_arrays,
                        (point.delta_statistical, point.delta_spectral), strict=True):
                    target[q,w] = value[active]
                for name, value in point.diagnostics.items():
                    increment_metrics[name] = max(increment_metrics.get(name,0.), value)
                advanced = _adjoint(r)
                for target,value in zip(arrays,(r,c,dgreen),strict=True):
                    _finite(value,'full-device Green')
                    target[q,w] = value[active]
                _local_checks(full,full_exports,r,c,dgreen,(q,w))
                # Inverse residuals are evidence; raw source and independent
                # Green identities both retain their original tolerance gate.
                denom = max(np.linalg.norm(operator,ord=np.inf)*np.linalg.norm(r,ord=np.inf),1.)
                for side,value in (('left',operator@r-identity),('right',r@operator-identity)):
                    residual = np.array([[np.linalg.norm(value,ord=np.inf)/denom]])
                    _measure(inverse_metrics,side+'_backward_residual',residual,np.zeros((1,1)),(q,w))
                if w == grid.zero_index and not any(mesh.addresses[q]):
                    zero_context = _zero_frequency_context(
                        q, w, grid, mesh, operator, r, c, dgreen, cs, ds,
                        raw[q], left, right, sigma_input, rtol, atol)
                if omega > 0:
                    for index,(contact,support) in enumerate(zip((left,right),supports,strict=True)):
                        less,greater = contact.spectrum.lesser[q,w],contact.spectrum.greater[q,w]
                        currents[index] += omega*channel_exchange(0.5*(greater+less),greater-less,
                                                                  c[support],dgreen[support])
                    if sigma_input is not None:
                        currents[2] += omega*channel_exchange(sigma_input.statistical[q,w],
                            sigma_input.spectral[q,w],c[active],dgreen[active])
        try:
            checks['raw_source'] = _finish(source_metrics,rtol,atol,'raw source')
            checks['full_green_local'] = _finish(full,rtol,atol,'full Green')
        except ValueError as exc:
            raise ValueError(f'{exc}; zero_frequency_context={zero_context}') from exc
        checks['inverse'] = _finish(inverse_metrics,rtol,atol,'inverse residuals',gate=False)
        checks['full_green_exports'] = _finish(full_exports,rtol,atol,'full Green exports',gate=False)
        green = ChannelSpectrum(*arrays)
        checks['active_green'],checks['active_green_exports'] = _field_checks(
            mesh,grid,green,rtol,atol,'active Green')
        born_r = np.zeros(shape,dtype=np.complex128)
        if coupling == 0 or a == 0:
            born_c,born_d = np.zeros(shape,dtype=np.complex128),np.zeros(shape,dtype=np.complex128)
        else:
            _finish(reference_cache, rtol, atol, 'reference Green cache')
            born = difference_born(grid,vertex,
                StatisticalSpectral(reference_green.statistical,reference_green.spectral),
                StatisticalSpectral(*delta_arrays),workers=workers,
                external_tile=external_tile,bin_tile=bin_tile,max_bytes=plan.born_peak_bytes,
                max_contraction_work=work)
            born_c,born_d = born.statistical,born.spectral
            born_c *= squared; born_d *= squared
            for q in range(nq):
                born_r[q] = compensated_sum(reference_mapped.retarded[q],
                    retarded_from_spectral(born_d[q],grid,
                        max_bytes=plan.reconstruction_workspace_bytes))
                born_c[q] = compensated_sum(reference_mapped.statistical[q],born_c[q])
                born_d[q] = compensated_sum(reference_mapped.spectral[q],born_d[q])
        mapped = ChannelSpectrum(born_r,born_c,born_d)
        _spectrum(mapped,shape,'mapped Sigma')
        checks['mapped_sigma'],checks['mapped_sigma_exports'] = _field_checks(
            mesh,grid,mapped,rtol,atol,'mapped Sigma')
    values = [float(scale.to_physical_current(float(v*grid.integration_weight/nq))) for v in currents]
    if not np.isfinite(values).all():
        raise ValueError('Current conversion overflow.')
    for spectrum in (green,mapped):
        for name in _COMPONENTS:
            getattr(spectrum,name).setflags(write=False)
    evaluation = ChannelBornMapEvaluation(green,mapped,TransverseCurrents(*values),
        MappingProxyType(checks),plan,sigma_input is None,coupling,zero_context)
    return evaluation, {'arithmetic': 'exact_reference_increment',
        'dyson_solves': plan.dyson_evaluations,
        'reference_cache': _finish(reference_cache,rtol,atol,'reference Green cache'),
        'increment_arithmetic': increment_metrics,
        'active_increment_bytes': sum(value.nbytes for value in delta_arrays)}


def plan_tangent_map(n_q, n_frequency, full_ndof, active_ndof, left_ndof,
                     right_ndof, *, workers=8, external_tile=16, bin_tile=1):
    """Signed tangent allocation/work plan, excluding caller base/direction.

    One fresh full-device inverse is evaluated per point. Two active tangent
    C/D stacks coexist with the two-channel Born derivative and retarded
    reconstruction. Caller base, direction and vertex are separate storage.
    """
    base = plan_channel_born_map(n_q,n_frequency,full_ndof,active_ndof,
        left_ndof,right_ndof,has_input_sigma=True,workers=workers,
        external_tile=external_tile,bin_tile=bin_tile)
    born = plan_born_expansion_dimensions(n_q,n_frequency,active_ndof,
        operation='tangent',workers=workers,external_tile=external_tile,bin_tile=bin_tile)
    increments = 2*16*n_q*n_frequency*active_ndof**2
    extra = increments + max(0,born.peak_owned_bytes-base.born_peak_bytes)
    return replace(base,born_peak_bytes=born.peak_owned_bytes,
        peak_owned_bytes=base.peak_owned_bytes+extra,
        contraction_work=born.contraction_work)


def tangent_map(
    grid, vertex, dynamical, left, right, *, sigma_base, base_green, direction,
    scale=None, g=.03, workers=8, external_tile=16, bin_tile=1,
    max_bytes=_DEFAULT_BYTES, max_contraction_work=_DEFAULT_WORK,
    max_dyson_evaluations=2404, identity_rtol=1e-8, identity_atol=1e-11,
):
    """Analytic derivative of the complete full-device Dyson/Born map.

    Returns (ChannelSpectrum, diagnostics). The physical base input must have
    been evaluated already with ``evaluate_reference_map``. Recomputed base
    source/full Green checks retain the original tolerances. Signed directions
    and derivatives receive shape/finiteness checks, never PSD/projection gates.
    ``scale`` is accepted for interface parity; this derivative does not compute
    or infer a current/convergence result.
    """
    if not isinstance(grid,SignedFrequencyGrid) or not isinstance(vertex,PackedTransverseCubicVertex):
        raise TypeError('Supply a signed grid and packed transverse vertex.')
    if scale is not None and not isinstance(scale,SolverScale):
        raise TypeError('scale must be SolverScale.')
    coupling = _real(g,'g')
    squared = coupling*coupling
    if not np.isfinite(squared) or (coupling > 0 and squared == 0):
        raise ValueError('g squared overflow/underflow.')
    rtol,atol = _real(identity_rtol,'identity_rtol'),_real(identity_atol,'identity_atol')
    if rtol+atol == 0:
        raise ValueError('At least one identity tolerance must be positive.')
    limit,work,dyson_limit = (_integer(value,name) for value,name in
        ((max_bytes,'max_bytes'),(max_contraction_work,'max_contraction_work'),
         (max_dyson_evaluations,'max_dyson_evaluations')))
    mesh = vertex.mesh
    nq,nw,d,a = mesh.n_qpoints,grid.n_points,vertex.full_ndof,vertex.ndof
    shape = (nq,nw,a,a)
    _spectrum(base_green,shape,'base_green')
    _spectrum(direction,shape,'signed direction')
    if sigma_base is not None:
        _spectrum(sigma_base,shape,'sigma_base')
    plan = plan_tangent_map(nq,nw,d,a,len(left.device_indices),len(right.device_indices),
        workers=workers,external_tile=external_tile,bin_tile=bin_tile)
    if plan.peak_owned_bytes > limit:
        raise ValueError('Tangent allocation plan exceeds max_bytes.')
    if plan.contraction_work > work:
        raise ValueError('Tangent exceeds max_contraction_work.')
    if plan.dyson_evaluations > dyson_limit:
        raise ValueError('Tangent exceeds max_dyson_evaluations.')
    raw = _array(dynamical,(nq,d,d),'dynamical')
    _finite(raw,'dynamical')
    for contact,side in ((left,'left'),(right,'right')):
        if not isinstance(contact,CompactTransverseContact):
            raise TypeError('Contacts must be CompactTransverseContact.')
        if (contact.full_ndof != d or contact.mesh.shape != mesh.shape
            or contact.grid.n_points != nw or contact.grid.step != grid.step
            or contact.contact.side != side):
            raise ValueError('Contact device/mesh/grid/side differs from the tangent.')
        for name in ('retarded','lesser','greater'):
            _finite(_array(getattr(contact.spectrum,name),
                (nq,nw,len(contact.device_indices),len(contact.device_indices)),
                side+'.'+name), side+'.'+name)
    active = np.ix_(vertex.active_dofs,vertex.active_dofs)
    supports = [np.ix_(np.asarray(contact.device_indices),np.asarray(contact.device_indices))
                for contact in (left,right)]
    identity = np.eye(d,dtype=np.complex128)
    arrays = [np.empty(shape,dtype=np.complex128) for _ in range(2)]
    source_metrics, full_metrics, exported, base_cache = {},{},{},{}
    with threadpool_limits(limits=1,user_api='blas'):
        for q in range(nq):
            for w,omega in enumerate(grid.values):
                operator = omega**2*identity-np.asarray(raw[q],dtype=np.complex128)
                cs,ds = np.zeros((d,d),dtype=np.complex128),np.zeros((d,d),dtype=np.complex128)
                for contact,support in zip((left,right),supports,strict=True):
                    less,greater = contact.spectrum.lesser[q,w],contact.spectrum.greater[q,w]
                    operator[support] -= contact.spectrum.retarded[q,w]
                    cs[support] += .5*(greater+less)
                    ds[support] += greater-less
                if sigma_base is not None:
                    operator[active] -= sigma_base.retarded[q,w]
                    cs[active] += sigma_base.statistical[q,w]
                    ds[active] += sigma_base.spectral[q,w]
                _measure(source_metrics,'raw_source_identity',ds,_adjoint(operator)-operator,(q,w))
                current = prepare_reference(operator,cs,ds)
                _local_checks(full_metrics,exported,current.retarded,
                              current.statistical,current.spectral,(q,w))
                for name in _COMPONENTS:
                    _measure(base_cache,name,getattr(current,name)[active],
                             getattr(base_green,name)[q,w],(q,w))
                step = tangent_green(current, *(getattr(direction,name)[q,w]
                    for name in _COMPONENTS),active_dofs=vertex.active_dofs)
                arrays[0][q,w],arrays[1][q,w] = step.statistical[active],step.spectral[active]
        checks = {
            'raw_source': _finish(source_metrics,rtol,atol,'tangent base raw source'),
            'full_green_local': _finish(full_metrics,rtol,atol,'tangent base full Green'),
            'base_cache': _finish(base_cache,rtol,atol,'tangent evaluated base cache'),
            'full_green_exports': _finish(exported,rtol,atol,'tangent base exports',gate=False),
        }
        if coupling == 0 or a == 0:
            born_c,born_d = np.zeros(shape,dtype=np.complex128),np.zeros(shape,dtype=np.complex128)
        else:
            born = tangent_born(grid,vertex,
                StatisticalSpectral(base_green.statistical,base_green.spectral),
                StatisticalSpectral(*arrays),workers=workers,external_tile=external_tile,
                bin_tile=bin_tile,max_bytes=plan.born_peak_bytes,max_contraction_work=work)
            born_c,born_d = born.statistical,born.spectral
            born_c *= squared; born_d *= squared
        born_r = np.empty(shape,dtype=np.complex128)
        for q in range(nq):
            born_r[q] = retarded_from_spectral(born_d[q],grid,
                max_bytes=plan.reconstruction_workspace_bytes)
        mapped = ChannelSpectrum(born_r,born_c,born_d)
        _spectrum(mapped,shape,'signed mapped tangent')
    for name in _COMPONENTS:
        getattr(mapped,name).setflags(write=False)
    return mapped, {'arithmetic':'analytic_dyson_born_tangent',
        'dyson_solves':plan.dyson_evaluations,
        'contraction_work':plan.contraction_work,
        'peak_owned_bytes':plan.peak_owned_bytes,
        'base_checks':checks, 'signed_tangent_has_physical_psd_gate':False}
