"""Bounded fresh Dyson/Born feedback with persistent statistical/spectral fields.

This opt-in research path retains all raw entries.  It adds no device eta and
performs no symmetry projection or contour-to-spectral repair.  A full device
inverse is evaluated at every supplied q/frequency; only its interaction block
is retained.  Full local checks and active-block reflection/reality checks have
the same scope as the existing material mapper.  Exported contour subtraction
is reported separately and never replaces the independent spectral identity.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import hypot
from time import monotonic
from types import MappingProxyType

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .transverse_channels import (
    StatisticalSpectral,
    cubic_born_channels_packed_fft,
    plan_born_channels_fft_dimensions,
    retarded_from_spectral,
)
from .transverse_compact_contacts import CompactTransverseContact
from .transverse_material_map import MaterialMapPlan
from .transverse_material_scba import MaterialCurrentResidual, MaterialSigmaResidual
from .transverse_packed import PackedTransverseCubicVertex, _array, _finite, _integer
from .transverse_solver import TransverseCurrents
from .types import SolverScale

_COMPONENTS = ('retarded', 'statistical', 'spectral')
_DEFAULT_BYTES = 32 * 1024**3
_DEFAULT_WORK = 100_000_000_000_000


@dataclass(frozen=True, slots=True, eq=False)
class ChannelSpectrum:
    """Borrowed independent R/C/D arrays; construction neither copies nor repairs."""
    retarded: np.ndarray = field(repr=False)
    statistical: np.ndarray = field(repr=False)
    spectral: np.ndarray = field(repr=False)


@dataclass(frozen=True, slots=True, eq=False)
class ChannelBornMapEvaluation:
    green: ChannelSpectrum
    map_sigma: ChannelSpectrum
    input_currents: TransverseCurrents
    checks: object
    plan: MaterialMapPlan
    input_sigma_is_zero: bool
    g: float
    zero_frequency_context: dict | None = None


@dataclass(frozen=True, slots=True)
class ChannelSigmaResidual(MaterialSigmaResidual):
    """Original residual scalars plus global maximum matrix coordinates."""

    error_location: tuple[int, ...] = ()
    scale_location: tuple[int, ...] = ()


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise TypeError(f'{name} must be a finite nonnegative real scalar.')
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


def _adjoint(value):
    return value.conj().T


def _ratio(error, allowance):
    return error / allowance if allowance else (0.0 if error == 0 else float('inf'))


def _spectrum(value, shape, name):
    if not isinstance(value, ChannelSpectrum):
        raise TypeError(f'{name} must be ChannelSpectrum with independent C/D.')
    for component in _COMPONENTS:
        array = _array(getattr(value, component), shape, name+'.'+component)
        _finite(array, name+'.'+component)


def _measure(metrics, name, first, second, location=()):
    if not first.size:
        return
    difference = np.abs(first-second)
    absolute_first, absolute_second = np.abs(first), np.abs(second)
    error_index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    error = float(difference[error_index])
    first_scale, second_scale = float(np.max(absolute_first)), float(np.max(absolute_second))
    scale = max(first_scale, second_scale)
    scale_array = absolute_first if first_scale >= second_scale else absolute_second
    scale_index = np.unravel_index(int(np.argmax(scale_array)), scale_array.shape)
    error_location = [int(v) for v in (*location, *error_index)]
    scale_location = [int(v) for v in (*location, *scale_index)]
    prior = metrics.get(name)
    if prior is None:
        prior = {'error': error, 'scale': scale, 'error_location': error_location,
                 'scale_location': scale_location}
        metrics[name] = prior
    else:
        if error > prior['error']:
            prior['error'], prior['error_location'] = error, error_location
        if scale > prior['scale']:
            prior['scale'], prior['scale_location'] = scale, scale_location


def _finish(metrics, rtol, atol, label, *, gate=True):
    result = {}
    for name, measured in metrics.items():
        error, scale = measured['error'], measured['scale']
        if not np.isfinite(error) or not np.isfinite(scale):
            raise ValueError(f'Nonfinite {label} diagnostic: {name}.')
        allowance = atol + rtol*scale
        result[name] = dict(measured, allowance=allowance, ratio=_ratio(error, allowance))
    failed = {key: value for key, value in result.items()
              if not np.isfinite(value['ratio']) or value['ratio'] > 1}
    if gate and failed:
        raise ValueError(f'Invalid {label}: {failed}')
    return result


def _local_checks(metrics, exported, r, c, d, location):
    _measure(metrics, 'statistical_antihermitian', c, -_adjoint(c), location)
    _measure(metrics, 'spectral_antihermitian', d, -_adjoint(d), location)
    _measure(metrics, 'keldysh_identity', d, r-_adjoint(r), location)
    lesser, greater = c-0.5*d, c+0.5*d
    for name, contour in (('lesser', lesser), ('greater', greater)):
        _measure(metrics, name+'_antihermitian', contour, -_adjoint(contour), location)
        if contour.size:
            correlation = 1j*contour
            eigenvalues = np.linalg.eigvalsh(0.5*(correlation+_adjoint(correlation)))
            error = max(0.0, -float(np.min(eigenvalues)))
            scale = float(np.max(np.abs(eigenvalues)))
            entry = metrics.setdefault(name+'_psd', {'error': 0.0, 'scale': 0.0,
                                       'error_location': [], 'scale_location': []})
            if error > entry['error']:
                entry['error'], entry['error_location'] = error, list(location)
            if scale > entry['scale']:
                entry['scale'], entry['scale_location'] = scale, list(location)
    exported_d = greater-lesser
    _measure(exported, 'contour_identity', exported_d, r-_adjoint(r), location)
    _measure(exported, 'spectral_roundtrip', exported_d, d, location)
    _measure(exported, 'statistical_roundtrip', 0.5*(greater+lesser), c, location)


def _field_checks(mesh, grid, spectrum, rtol, atol, label):
    metrics, exported = {}, {}
    for q in range(mesh.n_qpoints):
        opposite = mesh.negative_index(q)
        for w in range(grid.n_points):
            reverse = grid.n_points-1-w
            r, c, d = (getattr(spectrum, key)[q, w] for key in _COMPONENTS)
            _local_checks(metrics, exported, r, c, d, (q, w))
            _measure(metrics, 'statistical_reflection', c,
                     spectrum.statistical[opposite, reverse].T, (q, w))
            _measure(metrics, 'spectral_reflection', d,
                     -spectrum.spectral[opposite, reverse].T, (q, w))
            _measure(metrics, 'bosonic_reflection', c-0.5*d,
                     (spectrum.statistical[opposite, reverse]
                      +0.5*spectrum.spectral[opposite, reverse]).T, (q, w))
            _measure(metrics, 'retarded_reality', r,
                     spectrum.retarded[opposite, reverse].conj(), (q, w))
    return (_finish(metrics, rtol, atol, label),
            _finish(exported, rtol, atol, label+' exports', gate=False))


def _contact_checks(contact, rtol, atol, label):
    metrics = {}
    for q in range(contact.mesh.n_qpoints):
        opposite = contact.mesh.negative_index(q)
        for w in range(contact.grid.n_points):
            reverse = contact.grid.n_points-1-w
            r, lesser, greater = (getattr(contact.spectrum, key)[q, w]
                                  for key in ('retarded', 'lesser', 'greater'))
            # Contacts are raw contour inputs: their existing identity is
            # checked before the sole per-point C/D import used by Dyson.
            for name, contour in (('lesser', lesser), ('greater', greater)):
                _measure(metrics, name+'_antihermitian', contour, -_adjoint(contour), (q,w))
                corr = 1j*contour
                values = np.linalg.eigvalsh(0.5*(corr+_adjoint(corr)))
                entry = metrics.setdefault(name+'_psd', {'error': 0.0, 'scale': 0.0,
                                           'error_location': [], 'scale_location': []})
                error, scale = max(0., -float(values.min())), float(np.abs(values).max())
                if error > entry['error']:
                    entry['error'], entry['error_location'] = error, [q,w]
                if scale > entry['scale']:
                    entry['scale'], entry['scale_location'] = scale, [q,w]
            _measure(metrics, 'keldysh_identity', greater-lesser, r-_adjoint(r), (q,w))
            _measure(metrics, 'bosonic_reflection', lesser,
                     contact.spectrum.greater[opposite, reverse].T, (q,w))
            _measure(metrics, 'retarded_reality', r,
                     contact.spectrum.retarded[opposite, reverse].conj(), (q,w))
    return _finish(metrics, rtol, atol, label)


def plan_channel_born_map(
    n_q, n_frequency, full_ndof, active_ndof, left_ndof, right_ndof, *,
    has_input_sigma=False, workers=8, external_tile=16, bin_tile=1,
) -> MaterialMapPlan:
    """Count additional numeric payloads before any full-device Dyson solve.

    Excludes caller fields, persistent vertex, Python objects and library
    scratch; the bound is not process RSS.  No full harmonic frequency stack.
    """
    nq, nw, d = (_integer(v, key) for v, key in
                 ((n_q,'n_q'),(n_frequency,'n_frequency'),(full_ndof,'full_ndof')))
    a = _integer(active_ndof, 'active_ndof', zero=True)
    left, right = _integer(left_ndof, 'left_ndof'), _integer(right_ndof, 'right_ndof')
    if max(a,left,right) > d:
        raise ValueError('Interaction/contact dimensions must fit the full device.')
    if type(has_input_sigma) is not bool:
        raise TypeError('has_input_sigma must be bool.')
    born = plan_born_channels_fft_dimensions(nq,nw,a,workers=workers,
                                           external_tile=external_tile,bin_tile=bin_tile)
    component = 16*nq*nw*a*a
    green = mapped = 3*component
    # Twelve extra full matrices cover the single Gamma/zero product context.
    dyson = 16*60*d*d + 16*(nq+nw+4*d)
    nfft = 1 << (2*nw-2).bit_length()
    reconstruction = (16*nw*a*a + 32*nfft*(a*a+1) + 17*a*a) if a else 0
    validation = 16*28*max(a,1)**2
    peak = green+dyson+max(born.peak_owned_bytes,mapped+reconstruction,mapped+validation)
    caller = born.vertex_bytes+16*nq*d*d+16*nq*(3*nw+1)*(left*left+right*right)
    caller += mapped if has_input_sigma else 0
    return MaterialMapPlan(nq,nw,d,a,green,mapped,dyson,born.peak_owned_bytes,
                           reconstruction,peak,caller,born.contraction_work,nq*nw)


def channel_exchange(source_c, source_d, green_c, green_d) -> float:
    """ReTr(Dsource Cgreen - Csource Dgreen), current into the device.

    This is the algebraic contour exchange without subtracting two products
    dominated by Csource*Cgreen.  Matrices retain their original order.
    """
    return float((np.einsum('ij,ji->',source_d,green_c)
                  -np.einsum('ij,ji->',source_c,green_d)).real)


def _zero_frequency_context(q, w, grid, mesh, operator, r, c, dgreen, cs, ds,
                            dynamical, left, right, sigma_input, rtol, atol):
    """One full-matrix Gamma/zero diagnostic, retaining only scalar metadata.

    Extra matrix products are diagnostic references and never replace a field
    or participate in a guard. Twelve matrix payloads are included in the plan.
    A diagnostic overflow is recorded instead of creating a new solver gate.
    Ratios use this one point's scales, not whole-grid acceptance scales.
    """
    metadata = {'diagnostic_only': True, 'q_index': int(q), 'frequency_index': int(w),
                'omega': float(grid.values[w]),
                'q_address': [int(v) for v in mesh.addresses[q]]}
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            advanced = _adjoint(r)
            source_delta = ds-(_adjoint(operator)-operator)
            propagated = r@source_delta@advanced
            primary_delta = dgreen-(r-advanced)
            metrics = {}
            _measure(metrics, 'raw_source_identity', ds, _adjoint(operator)-operator, (q, w))
            _measure(metrics, 'primary_spectral_identity', dgreen, r-advanced, (q, w))
            _measure(metrics, 'propagated_source_identity', propagated,
                     np.zeros_like(propagated), (q, w))
            _measure(metrics, 'identity_product_remainder', primary_delta, propagated, (q, w))
            _measure(metrics, 'source_spectral_roundtrip', (cs+0.5*ds)-(cs-0.5*ds), ds, (q, w))
            _measure(metrics, 'exported_spectral_roundtrip',
                     (c+0.5*dgreen)-(c-0.5*dgreen), dgreen, (q, w))
            greater_product = r@(cs+0.5*ds)@advanced
            lesser_product = r@(cs-0.5*ds)@advanced
            _measure(metrics, 'spectral_product_vs_contour_products', dgreen,
                     greater_product-lesser_product, (q, w))
            components = {}
            _measure(components, 'harmonic', dynamical, _adjoint(dynamical), (q, w))
            for name, contact in (('left', left), ('right', right)):
                contact_r = contact.spectrum.retarded[q, w]
                contact_d = contact.spectrum.greater[q, w]-contact.spectrum.lesser[q, w]
                _measure(components, name, contact_d, contact_r-_adjoint(contact_r), (q, w))
            if sigma_input is not None:
                input_r = sigma_input.retarded[q, w]
                _measure(components, 'interaction', sigma_input.spectral[q, w],
                         input_r-_adjoint(input_r), (q, w))
            else:
                zero = np.zeros((1, 1), dtype=np.complex128)
                _measure(components, 'interaction', zero, zero, (q, w))
            denominator = max(np.linalg.norm(operator, ord=np.inf)
                              *np.linalg.norm(r, ord=np.inf), 1.)
            identity = np.eye(len(operator), dtype=np.complex128)
            inverse = {name: float(np.linalg.norm(value, ord=np.inf)/denominator)
                       for name, value in (('left', operator@r-identity),
                                           ('right', r@operator-identity))}
            ratios = {}
            for name, statistical, spectral in (('source', cs, ds), ('green', c, dgreen)):
                scale_d = float(np.abs(spectral).max())
                ratios[name] = float(np.abs(statistical).max())/scale_d if scale_d else None
            metrics = _finish(metrics, rtol, atol, 'Gamma/zero products', gate=False)
            components = _finish(components, rtol, atol, 'Gamma/zero components', gate=False)
            scalars = [*inverse.values(), *(v for v in ratios.values() if v is not None)]
            scalars += [row[key] for rows in (metrics, components) for row in rows.values()
                        for key in ('error', 'scale', 'allowance', 'ratio')]
            if not all(np.isfinite(v) for v in scalars):
                raise ValueError('Nonfinite Gamma/zero diagnostic scalar.')
            return dict(metadata, available=True, metrics=metrics,
                        source_component_metrics=components,
                        inverse_backward_residuals=inverse,
                        statistical_to_spectral_scale_ratios=ratios)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return dict(metadata, available=False, diagnostic_error=f'{type(exc).__name__}: {exc}')


def evaluate_channel_born_map(
    grid, vertex, dynamical, left, right, *, sigma_input=None, scale=None, g=0.03,
    workers=8, external_tile=16, bin_tile=1, max_bytes=_DEFAULT_BYTES,
    max_contraction_work=_DEFAULT_WORK, max_dyson_evaluations=2404,
    identity_rtol=1e-8, identity_atol=1e-11,
) -> ChannelBornMapEvaluation:
    """Fresh full-device Dyson then exactly one four-term channel Born image.

    Input R/C/D is already scaled; g squared multiplies the Born image once.
    None denotes exact zero interaction.  Dgreen is formed only from the raw
    source D, independently from both Rgreen and exported Green contours.
    Currents describe this evaluated input; map_sigma remains unevaluated.
    """
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
    plan = plan_channel_born_map(nq,nw,d,a,len(left.device_indices),len(right.device_indices),
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
                if sigma_input is not None:
                    operator[active] -= sigma_input.retarded[q,w]
                    cs[active] += sigma_input.statistical[q,w]
                    ds[active] += sigma_input.spectral[q,w]
                _finite(operator,'Dyson operator'); _finite(cs,'statistical source'); _finite(ds,'spectral source')
                _measure(source_metrics,'raw_source_identity',ds,_adjoint(operator)-operator,(q,w))
                try:
                    r = np.linalg.inv(operator)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(f'Singular full-device Dyson solve at q={q}, w={w}.') from exc
                advanced = _adjoint(r)
                c,dgreen = r@cs@advanced,r@ds@advanced
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
        if coupling == 0 or a == 0:
            born_c,born_d = np.zeros(shape,dtype=np.complex128),np.zeros(shape,dtype=np.complex128)
        else:
            born = cubic_born_channels_packed_fft(grid,vertex,
                StatisticalSpectral(green.statistical,green.spectral),workers=workers,
                external_tile=external_tile,bin_tile=bin_tile,max_bytes=plan.born_peak_bytes,
                max_contraction_work=work)
            born_c,born_d = born.statistical,born.spectral
            born_c *= squared; born_d *= squared
        born_r = np.empty(shape,dtype=np.complex128)
        if a:
            for q in range(nq):
                born_r[q] = retarded_from_spectral(born_d[q],grid,
                    max_bytes=plan.reconstruction_workspace_bytes)
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
    return ChannelBornMapEvaluation(green,mapped,TransverseCurrents(*values),
        MappingProxyType(checks),plan,sigma_input is None,coupling,zero_context)


@dataclass(frozen=True, slots=True)
class ChannelSCBAConfig:
    max_evaluations: int = 3
    mixing: float = 0.1
    sigma_rtol: float = 1e-4
    sigma_atol: float = 1e-10
    current_rtol: float = 1e-3
    current_atol_w: float = 1e-14
    max_seconds: float = 79200.0
    time_reserve_seconds: float = 300.0
    next_map_safety_factor: float = 1.25
    initial_map_estimate_seconds: float = 2100.0
    max_bytes: int = _DEFAULT_BYTES
    workers: int = 8
    external_tile: int = 16
    bin_tile: int = 1
    max_contraction_work: int = _DEFAULT_WORK
    max_dyson_evaluations: int = 2404
    identity_rtol: float = 1e-8
    identity_atol: float = 1e-11

    def __post_init__(self):
        for name in ('max_evaluations','max_bytes','workers','external_tile','bin_tile',
                     'max_contraction_work','max_dyson_evaluations'):
            _integer(getattr(self,name),name)
        for name in ('mixing','sigma_rtol','sigma_atol','current_rtol','current_atol_w',
                     'max_seconds','time_reserve_seconds','next_map_safety_factor',
                     'initial_map_estimate_seconds','identity_rtol','identity_atol'):
            _real(getattr(self,name),name)
        if not 0 < self.mixing <= 1:
            raise ValueError('mixing must lie in (0,1].')
        for first,second in ((self.sigma_rtol,self.sigma_atol),
                             (self.current_rtol,self.current_atol_w),
                             (self.identity_rtol,self.identity_atol)):
            if first+second == 0:
                raise ValueError('Each tolerance pair requires a positive allowance.')
        if self.max_seconds <= self.time_reserve_seconds:
            raise ValueError('max_seconds must exceed time_reserve_seconds.')
        if self.next_map_safety_factor < 1 or self.initial_map_estimate_seconds <= 0:
            raise ValueError('Map safety factor must be >=1 and initial estimate >0.')


@dataclass(frozen=True, slots=True)
class ChannelSCBAPlan:
    map_plan: MaterialMapPlan
    input_sigma_bytes: int
    previous_evaluated_state_bytes: int
    residual_mixing_workspace_bytes: int
    peak_owned_bytes: int
    nominal_caller_bytes: int
    maximum_contraction_work: int
    maximum_dyson_evaluations: int


def plan_channel_scba(n_q,n_frequency,full_ndof,active_ndof,left_ndof,right_ndof,*,
                      config=None,has_initial_sigma=False):
    """Includes last complete input/Green/image while the next map is running.

    A caller-supplied candidate is separate caller storage. Its detached copy
    fits the existing candidate allowance; no historical Green/map is retained.
    """
    config = ChannelSCBAConfig() if config is None else config
    if not isinstance(config,ChannelSCBAConfig):
        raise TypeError('config must be ChannelSCBAConfig.')
    if type(has_initial_sigma) is not bool:
        raise TypeError('has_initial_sigma must be bool.')
    plan = plan_channel_born_map(n_q,n_frequency,full_ndof,active_ndof,left_ndof,right_ndof,
        workers=config.workers,external_tile=config.external_tile,bin_tile=config.bin_tile)
    sigma = plan.mapped_sigma_bytes
    previous = 2*sigma+plan.retained_green_bytes
    scratch = 128*max(plan.active_ndof,1)**2
    return ChannelSCBAPlan(plan,sigma,previous,scratch,plan.peak_owned_bytes+sigma+previous+scratch,
        plan.nominal_caller_bytes+(sigma if has_initial_sigma else 0),
        config.max_evaluations*plan.contraction_work,
        config.max_evaluations*plan.dyson_evaluations)


@dataclass(frozen=True, slots=True)
class ChannelSCBARecord:
    index: int
    map_seconds: float
    elapsed_seconds: float
    sigma_residuals: dict[str, ChannelSigmaResidual]
    current_residuals: dict[str, MaterialCurrentResidual]
    channel_residuals: dict[str, ChannelSigmaResidual]
    sigma_passed: bool
    current_passed: bool
    converged: bool


@dataclass(frozen=True, slots=True, eq=False)
class ChannelSCBAState:
    index: int
    sigma_input: ChannelSpectrum | None
    evaluation: ChannelBornMapEvaluation
    record: ChannelSCBARecord


@dataclass(frozen=True, slots=True, eq=False)
class ChannelSCBAResult:
    state: ChannelSCBAState | None
    history: tuple[ChannelSCBARecord,...]
    converged: bool
    stop_reason: str
    failure_reason: str | None
    plan: ChannelSCBAPlan


def mix_channel_spectra(sigma_input,mapped_sigma,mixing) -> ChannelSpectrum:
    """Mix independent R/C/D with one alpha; None means exact zero input.

    This produces an unevaluated candidate. Causality and physical checks run
    before that candidate's next Dyson evaluation. No arrays are modified.
    """
    alpha = _real(mixing,'mixing')
    if not 0 < alpha <= 1:
        raise ValueError('mixing must lie in (0,1].')
    if not isinstance(mapped_sigma,ChannelSpectrum):
        raise TypeError('mapped_sigma must be ChannelSpectrum.')
    shape = np.shape(mapped_sigma.retarded)
    if len(shape) != 4 or min(shape[:2]) < 1 or shape[-1] != shape[-2]:
        raise ValueError('Spectrum must have shape (Nq,Nw,A,A).')
    _spectrum(mapped_sigma,shape,'mapped_sigma')
    if sigma_input is not None:
        _spectrum(sigma_input,shape,'sigma_input')
    arrays = []
    with np.errstate(over='raise',invalid='raise'):
        for name in _COMPONENTS:
            mapped = getattr(mapped_sigma,name)
            output = np.empty(shape,dtype=np.complex128)
            for index in np.ndindex(shape[:2]):
                np.multiply(mapped[index],alpha,out=output[index])
                if sigma_input is not None:
                    output[index] += (1-alpha)*getattr(sigma_input,name)[index]
            _finite(output,'mixed candidate')
            output.setflags(write=False)
            arrays.append(output)
    return ChannelSpectrum(*arrays)


def _sigma_residual(mapped,supplied,config,component):
    error = scale = error_fro = mapped_fro = supplied_fro = 0.0
    error_location = scale_location = ()
    for index in np.ndindex(mapped.retarded.shape[:2]):
        if component in _COMPONENTS:
            first = getattr(mapped,component)[index]
            second = getattr(supplied,component)[index] if supplied is not None else np.zeros_like(first)
        else:
            sign = -0.5 if component == 'lesser' else 0.5
            first = mapped.statistical[index]+sign*mapped.spectral[index]
            second = (supplied.statistical[index]+sign*supplied.spectral[index]
                      if supplied is not None else np.zeros_like(first))
        if not first.size:
            continue
        difference = first-second
        absolute_difference = np.abs(difference)
        entry = np.unravel_index(int(np.argmax(absolute_difference)), difference.shape)
        block_error = float(absolute_difference[entry])
        if not error_location or block_error > error:
            error, error_location = block_error, tuple(int(v) for v in (*index, *entry))
        first_scale, second_scale = float(np.abs(first).max()), float(np.abs(second).max())
        block_scale = max(first_scale, second_scale)
        if not scale_location or block_scale > scale:
            scale_field = first if first_scale >= second_scale else second
            entry = np.unravel_index(int(np.argmax(np.abs(scale_field))), scale_field.shape)
            scale, scale_location = block_scale, tuple(int(v) for v in (*index, *entry))
        error_fro = hypot(error_fro,float(np.linalg.norm(difference)))
        mapped_fro = hypot(mapped_fro,float(np.linalg.norm(first)))
        supplied_fro = hypot(supplied_fro,float(np.linalg.norm(second)))
    allowance = config.sigma_atol+config.sigma_rtol*scale
    scale_fro = max(mapped_fro,supplied_fro)
    if not np.isfinite((error,scale,allowance,error_fro,scale_fro)).all():
        raise ValueError('Nonfinite unmixed Sigma residual.')
    return ChannelSigmaResidual(error,scale,allowance,_ratio(error,allowance),
                                error_fro,scale_fro,_ratio(error_fro,scale_fro),
                                error_location,scale_location)


def _record(index,sigma,evaluation,map_seconds,elapsed,config):
    residuals = {name:_sigma_residual(evaluation.map_sigma,sigma,config,name)
                 for name in ('retarded','lesser','greater')}
    channel_residuals = {name:_sigma_residual(evaluation.map_sigma,sigma,config,name)
                         for name in ('statistical','spectral')}
    currents = evaluation.input_currents
    scale = max(abs(currents.left),abs(currents.right))
    allowance = config.current_atol_w+config.current_rtol*scale
    if not np.isfinite((currents.left,currents.right,currents.collision_power,
                       currents.contact_balance,currents.full_balance,allowance)).all():
        raise ValueError('Nonfinite evaluated-input currents.')
    current_residuals = {name:MaterialCurrentResidual(abs(getattr(currents,name)),scale,allowance,
                          _ratio(abs(getattr(currents,name)),allowance))
                         for name in ('contact_balance','collision_power','full_balance')}
    sigma_passed = all(value.ratio <= 1 for value in residuals.values())
    current_passed = all(value.ratio <= 1 for value in current_residuals.values())
    return ChannelSCBARecord(index,map_seconds,elapsed,residuals,current_residuals,channel_residuals,
                             sigma_passed,current_passed,sigma_passed and current_passed)


def run_channel_scba(
    grid,vertex,dynamical,left,right,*,sigma_initial=None,first_index=1,
    scale=None,g=0.03,config=None,
    on_evaluation: Callable[[ChannelSCBAState],None] | None = None,
    on_start: Callable[[int,ChannelSpectrum | None],None] | None = None,
) -> ChannelSCBAResult:
    """At most config.max_evaluations NEW states with explicit trajectory labels.

    Defaults start from exact zero at index1. Opt-in continuation supplies an
    already mixed, unevaluated native R/C/D candidate and first_index>=2. That
    candidate is copied, never mixed a second time before its first evaluation.
    Indices run first_index through first_index+max_evaluations-1. No loaded
    checkpoint or candidate is promoted to accepted restart/transport state.

    Fixed alpha mixes R/C/D independently. All three original unmixed contour
    residuals and three input-current balance screens gate convergence; the
    extra C/D residuals are diagnostics. On later numerical failure the last
    completed evaluated state is retained. Callback errors propagate. Time
    admission uses reserve plus safety*max(initial estimate, prior durations),
    a heuristic that cannot interrupt an in-progress map. No checkpoint I/O.
    """
    started = monotonic()
    config = ChannelSCBAConfig() if config is None else config
    if not isinstance(config,ChannelSCBAConfig):
        raise TypeError('config must be ChannelSCBAConfig.')
    if not isinstance(grid,SignedFrequencyGrid) or not isinstance(vertex,PackedTransverseCubicVertex):
        raise TypeError('Supply a signed grid and packed transverse vertex.')
    if not all(isinstance(c,CompactTransverseContact) for c in (left,right)):
        raise TypeError('Supply compact contacts.')
    if any(cb is not None and not callable(cb) for cb in (on_evaluation,on_start)):
        raise TypeError('Callbacks must be callable.')
    first_index = _integer(first_index,'first_index')
    if (sigma_initial is None) != (first_index == 1):
        raise ValueError('Use exact-zero None only with first_index=1; continuation requires '
                         'sigma_initial and first_index>=2.')
    plan = plan_channel_scba(vertex.mesh.n_qpoints,grid.n_points,vertex.full_ndof,vertex.ndof,
                            len(left.device_indices),len(right.device_indices),config=config,
                            has_initial_sigma=sigma_initial is not None)
    if plan.peak_owned_bytes > config.max_bytes:
        raise ValueError('Channel SCBA allocation plan exceeds max_bytes.')
    if plan.map_plan.contraction_work > config.max_contraction_work:
        raise ValueError('Channel SCBA map exceeds max_contraction_work.')
    if plan.map_plan.dyson_evaluations > config.max_dyson_evaluations:
        raise ValueError('Channel SCBA map exceeds max_dyson_evaluations.')
    sigma = None
    if sigma_initial is not None:
        shape=(vertex.mesh.n_qpoints,grid.n_points,vertex.ndof,vertex.ndof)
        _spectrum(sigma_initial,shape,'sigma_initial')
        if any(getattr(sigma_initial,name).dtype != np.dtype('complex128')
               for name in _COMPONENTS):
            raise ValueError('sigma_initial fields must have native complex128 dtype.')
        # Copy bytes without arithmetic: even multiplication by complex one
        # can canonicalize signed zeros and invalidate the caller's raw hash.
        initial_arrays = [np.array(getattr(sigma_initial,name),copy=True,order='C')
                          for name in _COMPONENTS]
        for array in initial_arrays:
            array.setflags(write=False)
        sigma = ChannelSpectrum(*initial_arrays)
        del initial_arrays, array
    state,history = None,[]
    estimate = config.initial_map_estimate_seconds
    for index in range(first_index,first_index+config.max_evaluations):
        remaining = config.max_seconds-(monotonic()-started)
        if remaining <= config.time_reserve_seconds+config.next_map_safety_factor*estimate:
            return ChannelSCBAResult(state,tuple(history),False,'max_seconds',None,plan)
        try:
            if state is not None:
                sigma = mix_channel_spectra(state.sigma_input,state.evaluation.map_sigma,config.mixing)
        except (ValueError,FloatingPointError) as exc:
            return ChannelSCBAResult(state,tuple(history),False,'evaluation_error',f'{type(exc).__name__}: {exc}',plan)
        if on_start is not None:
            on_start(index,sigma)
        try:
            map_started = monotonic()
            evaluation = evaluate_channel_born_map(grid,vertex,dynamical,left,right,sigma_input=sigma,
                scale=scale,g=g,workers=config.workers,external_tile=config.external_tile,
                bin_tile=config.bin_tile,max_bytes=plan.map_plan.peak_owned_bytes,
                max_contraction_work=config.max_contraction_work,
                max_dyson_evaluations=config.max_dyson_evaluations,
                identity_rtol=config.identity_rtol,identity_atol=config.identity_atol)
            duration = monotonic()-map_started
            record = _record(index,sigma,evaluation,duration,monotonic()-started,config)
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            return ChannelSCBAResult(state,tuple(history),False,'evaluation_error',f'{type(exc).__name__}: {exc}',plan)
        state = ChannelSCBAState(index,sigma,evaluation,record)
        history.append(record)
        estimate = max(estimate,duration)
        if on_evaluation is not None:
            on_evaluation(state)
        if record.converged:
            return ChannelSCBAResult(state,tuple(history),True,'fixed_point',None,plan)
    return ChannelSCBAResult(state,tuple(history),False,'max_evaluations',None,plan)


__all__ = [
    'ChannelBornMapEvaluation',
    'ChannelSCBAConfig',
    'ChannelSCBAPlan',
    'ChannelSCBARecord',
    'ChannelSCBAResult',
    'ChannelSCBAState',
    'ChannelSigmaResidual',
    'ChannelSpectrum',
    'channel_exchange',
    'evaluate_channel_born_map',
    'mix_channel_spectra',
    'plan_channel_born_map',
    'plan_channel_scba',
    'run_channel_scba',
]
