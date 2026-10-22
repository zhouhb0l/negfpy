"""One memory-bounded full-device Dyson / packed Born evaluation.

All harmonic coordinates participate in every inverse. Only interaction blocks
are retained across frequencies. Currents describe the input self-energy state;
the returned Born image has not been used to recompute those Green functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .retarded import principal_value_cauchy_fft_scratch_bytes, retarded_from_keldysh
from .transverse_compact_contacts import CompactTransverseContact
from .transverse_green_diagnostics import (
    FullGreenDiagnosticCollector,
    green_diagnostic_workspace_bytes,
)
from .transverse_packed import (
    PackedTransverseCubicVertex,
    _array,
    _finite,
    _integer,
    cubic_born_self_energy_q_packed_fft,
    plan_packed_born_fft_dimensions,
)
from .transverse_solver import TransverseCurrents
from .types import SolverScale

_COMPONENTS = ('retarded', 'lesser', 'greater')
_DEFAULT_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True, eq=False)
class PackedMapSpectrum:
    """Three array references; evaluator validates them without whole-grid copies.

    Caller-owned input is not mutated or frozen. Output arrays from the evaluator
    are owned and read-only. Construction alone does not certify a spectrum.
    """

    retarded: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray


@dataclass(frozen=True, slots=True)
class MaterialMapPlan:
    n_q: int
    n_frequency: int
    full_ndof: int
    active_ndof: int
    retained_green_bytes: int
    mapped_sigma_bytes: int
    dyson_workspace_bytes: int
    born_peak_bytes: int
    reconstruction_workspace_bytes: int
    peak_owned_bytes: int
    nominal_caller_bytes: int
    contraction_work: int
    dyson_evaluations: int


@dataclass(frozen=True, slots=True, eq=False)
class PackedBornMapEvaluation:
    green: PackedMapSpectrum
    map_sigma: PackedMapSpectrum
    input_currents: TransverseCurrents
    checks: object
    plan: MaterialMapPlan
    input_sigma_is_zero: bool
    g: float


@dataclass(frozen=True, slots=True, eq=False)
class PackedGreenDiagnostic:
    """Completed Dyson diagnostics; never a Born/SCBA/transport result."""

    completed: bool
    green_passed: bool
    diagnostics: dict
    plan: MaterialMapPlan
    diagnostic_workspace_bytes: int
    diagnostic_peak_owned_bytes: int


def plan_material_born_map(
    n_q, n_frequency, full_ndof, active_ndof, left_ndof, right_ndof, *,
    has_input_sigma=False, workers=8, external_tile=16, bin_tile=1,
) -> MaterialMapPlan:
    """Conservative additional numeric allocation plan; not an RSS guarantee.

    Caller arrays, Python objects and FFT/BLAS/LAPACK internal scratch are
    excluded from peak_owned_bytes. Contact construction has its own bound.
    No full-device frequency stack or full-device cubic tensor is allocated.
    """
    nq = _integer(n_q, 'n_q')
    nw = _integer(n_frequency, 'n_frequency')
    d = _integer(full_ndof, 'full_ndof')
    a = _integer(active_ndof, 'active_ndof', zero=True)
    left = _integer(left_ndof, 'left_ndof')
    right = _integer(right_ndof, 'right_ndof')
    if max(a, left, right) > d:
        raise ValueError('Packed and contact dimensions must fit the full device.')
    if type(has_input_sigma) is not bool:
        raise TypeError('has_input_sigma must be bool.')
    born = plan_packed_born_fft_dimensions(
        nq, nw, a, workers=workers, external_tile=external_tile, bin_tile=bin_tile,
    )
    component = 16 * nq * nw * a**2
    green = 3 * component
    mapped = 3 * component
    # Includes operator, inverse, adjoint, sources, products, validation copies,
    # eigensolve input, index matrices and finite masks. Library scratch excluded.
    dyson = 16 * 40 * d**2 + 16 * (nq + nw + 4*d)
    reconstruct = (8 * 16 * nw * a**2
                   + principal_value_cauchy_fft_scratch_bytes(nw, a)) if a else 0
    validation = 16 * 24 * max(a, 1)**2
    # The final frequency's local arrays remain referenced during later phases.
    # Keep their full conservative allowance rather than relying on deallocation.
    peak = green + dyson + max(born.peak_owned_bytes,
                               mapped + reconstruct, mapped + validation)
    caller = (born.vertex_bytes + 16*nq*d**2
              + 16*nq*(3*nw+1)*(left**2+right**2)
              + (3*component if has_input_sigma else 0))
    return MaterialMapPlan(nq, nw, d, a, green, mapped, dyson,
                           born.peak_owned_bytes, reconstruct, peak, caller,
                           born.contraction_work, nq*nw)


def _real(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or np.iscomplexobj(value):
        raise TypeError(f'{name} must be a finite nonnegative real scalar.')
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative.')
    return float(value)


def _adjoint(value):
    return value.conj().T


def _measure(metrics, name, first, second):
    if first.size == 0:
        return
    error = float(np.max(np.abs(first-second)))
    scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))))
    prior = metrics.get(name, (0.0, 0.0))
    metrics[name] = (max(prior[0], error), max(prior[1], scale))


def _local_checks(metrics, r, lesser, greater):
    for name, field in (('lesser', lesser), ('greater', greater)):
        _measure(metrics, name+'_antihermitian', field, -_adjoint(field))
        if field.size:
            corr = 1j * field
            eigenvalues = np.linalg.eigvalsh(0.5*(corr+_adjoint(corr)))
            prior = metrics.get(name+'_psd', (0.0, 0.0))
            metrics[name+'_psd'] = (
                max(prior[0], max(0.0, -float(np.min(eigenvalues)))),
                max(prior[1], float(np.max(np.abs(eigenvalues)))),
            )
    _measure(metrics, 'keldysh_identity', greater-lesser, r-_adjoint(r))


def _finish_checks(metrics, rtol, atol, label):
    ratios = {}
    for name, (error, scale) in metrics.items():
        if not np.isfinite(error) or not np.isfinite(scale):
            raise ValueError(f'Nonfinite {label} diagnostic: {name}.')
        allowance = atol+rtol*scale
        ratios[name] = (error/allowance if allowance else
                        (0.0 if error == 0 else float('inf')))
    failed = {name:value for name,value in ratios.items() if not np.isfinite(value) or value > 1}
    if failed:
        raise ValueError(f'Invalid {label}: {failed}')
    return ratios


def _spectrum(spectrum, shape, name):
    if not isinstance(spectrum, PackedMapSpectrum):
        raise TypeError(f'{name} must be PackedMapSpectrum.')
    for component in _COMPONENTS:
        array = _array(getattr(spectrum, component), shape, name+'.'+component)
        _finite(array, name+'.'+component)


def _field_checks(mesh, grid, spectrum, rtol, atol, label):
    metrics = {}
    for q in range(mesh.n_qpoints):
        opposite = mesh.negative_index(q)
        for w in range(grid.n_points):
            r, lesser, greater = (getattr(spectrum, key)[q,w] for key in _COMPONENTS)
            _local_checks(metrics, r, lesser, greater)
            _measure(metrics, 'bosonic_reflection', lesser,
                     spectrum.greater[opposite, grid.n_points-1-w].T)
            _measure(metrics, 'retarded_reality', r,
                     spectrum.retarded[opposite, grid.n_points-1-w].conj())
    return _finish_checks(metrics, rtol, atol, label)


def _exchange(first_less, first_greater, green_less, green_greater):
    # trace(A B) without allocating matrix products.
    return float((np.einsum('ij,ji->', first_greater, green_less)
                  - np.einsum('ij,ji->', first_less, green_greater)).real)


def evaluate_packed_born_map(
    grid, vertex, dynamical, left, right, *, sigma_input=None, scale=None, g=1.0,
    workers=8, external_tile=16, bin_tile=1, max_bytes=_DEFAULT_BYTES,
    max_contraction_work=100_000_000, max_dyson_evaluations=100_000,
    identity_rtol=1e-8, identity_atol=1e-11, green_diagnostic_only=False,
) -> PackedBornMapEvaluation | PackedGreenDiagnostic:
    """Evaluate exactly one Born image, with no iteration or accepted-state update.

    Vertex is in solver units before coupling g; g**2 multiplies the contour
    image once. Optional sigma_input is already scaled. No device eta is added.
    Contacts remain on explicit boundary supports; the harmonic inverse is full.
    Every resource gate precedes Dyson. The returned currents use sigma_input,
    not map_sigma. This function does not certify SCBA convergence.
    green_diagnostic_only returns scalar context after the existing full Dyson
    loop and Green checks, even when a Green identity check fails, before Born.
    """
    if type(green_diagnostic_only) is not bool:
        raise TypeError('green_diagnostic_only must be bool.')
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(vertex, PackedTransverseCubicVertex):
        raise TypeError('Supply a signed grid and packed transverse vertex.')
    if scale is None:
        scale = SolverScale()
    if not isinstance(scale, SolverScale):
        raise TypeError('scale must be SolverScale.')
    coupling = _real(g, 'g')
    squared = coupling*coupling
    if not np.isfinite(squared):
        raise ValueError('g squared overflows.')
    if coupling > 0 and squared == 0:
        raise ValueError('g squared underflows.')
    rtol, atol = _real(identity_rtol, 'identity_rtol'), _real(identity_atol, 'identity_atol')
    if not rtol+atol > 0:
        raise ValueError('At least one identity tolerance must be positive.')
    limit = _integer(max_bytes, 'max_bytes')
    work = _integer(max_contraction_work, 'max_contraction_work')
    dyson_work = _integer(max_dyson_evaluations, 'max_dyson_evaluations')
    mesh, nq, nw, d, a = vertex.mesh, vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof, vertex.ndof
    for contact, side in ((left,'left'), (right,'right')):
        if not isinstance(contact, CompactTransverseContact):
            raise TypeError('Contacts must be CompactTransverseContact.')
        if (contact.full_ndof != d or contact.mesh.shape != mesh.shape
                or contact.grid.n_points != nw or contact.grid.step != grid.step
                or contact.contact.side != side):
            raise ValueError('Contact device/mesh/grid/side differs from the map.')
    plan = plan_material_born_map(
        nq, nw, d, a, len(left.device_indices), len(right.device_indices),
        has_input_sigma=sigma_input is not None, workers=workers,
        external_tile=external_tile, bin_tile=bin_tile,
    )
    diagnostic_bytes = green_diagnostic_workspace_bytes(d) if green_diagnostic_only else 0
    if plan.peak_owned_bytes + diagnostic_bytes > limit:
        raise ValueError('Material map allocation plan exceeds max_bytes.')
    if plan.contraction_work > work:
        raise ValueError('Material map exceeds max_contraction_work.')
    if plan.dyson_evaluations > dyson_work:
        raise ValueError('Material map exceeds max_dyson_evaluations.')
    raw = _array(dynamical, (nq,d,d), 'dynamical')
    _finite(raw, 'dynamical')
    harmonic_metrics = {}
    for q in range(nq):
        matrix = np.asarray(raw[q], dtype=complex)
        if not np.isfinite(matrix).all():
            raise ValueError('Dynamical conversion overflow.')
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-12:
            raise ValueError('The harmonic device must be stable.')
        _measure(harmonic_metrics, 'hermiticity', matrix, _adjoint(matrix))
        _measure(harmonic_metrics, 'opposite_q', matrix, raw[mesh.negative_index(q)].conj())
    checks = {'harmonic':_finish_checks(harmonic_metrics, 1e-10, 1e-12, 'dynamical')}
    shape = (nq,nw,a,a)
    if sigma_input is not None:
        _spectrum(sigma_input, shape, 'sigma_input')
        checks['input_sigma'] = _field_checks(mesh, grid, sigma_input, rtol, atol, 'input Sigma')
        causal = {}
        if a:
            for q in range(nq):
                reconstructed = retarded_from_keldysh(
                    sigma_input.lesser[q], sigma_input.greater[q], grid, hilbert_backend='fft',
                )
                _measure(causal, 'causal_reconstruction', sigma_input.retarded[q], reconstructed)
                del reconstructed
        checks['input_causality'] = _finish_checks(causal, rtol, atol, 'input Sigma causality')
    for name, contact in (('left',left), ('right',right)):
        # Existing constructors already validate the full compact fields. Check
        # again here only for directly supplied wrapper data or mutable aliases.
        field = PackedMapSpectrum(*(getattr(contact.spectrum,c) for c in _COMPONENTS))
        _spectrum(field, (nq,nw,len(contact.device_indices),len(contact.device_indices)), name)
        checks[name] = _field_checks(mesh, grid, field, rtol, atol, name+' contact')
    active = np.ix_(vertex.active_dofs, vertex.active_dofs)
    supports = [np.ix_(np.asarray(c.device_indices), np.asarray(c.device_indices)) for c in (left,right)]
    arrays = [np.empty(shape, dtype=complex) for _ in _COMPONENTS]
    full_metrics = {}
    collector = (FullGreenDiagnosticCollector(n_q=nq, n_frequency=nw, rtol=rtol, atol=atol)
                 if green_diagnostic_only else None)
    currents = np.zeros(3)
    identity = np.eye(d, dtype=complex)
    with threadpool_limits(limits=1, user_api='blas'):
        for q in range(nq):
            for w, omega in enumerate(grid.values):
                operator = omega**2 * identity - np.asarray(raw[q], dtype=complex)
                sources = [np.zeros((d,d), dtype=complex), np.zeros((d,d), dtype=complex)]
                for contact, support in zip((left,right), supports, strict=True):
                    operator[support] -= contact.spectrum.retarded[q,w]
                    sources[0][support] += contact.spectrum.lesser[q,w]
                    sources[1][support] += contact.spectrum.greater[q,w]
                if sigma_input is not None:
                    operator[active] -= sigma_input.retarded[q,w]
                    sources[0][active] += sigma_input.lesser[q,w]
                    sources[1][active] += sigma_input.greater[q,w]
                if not np.isfinite(operator).all() or any(not np.isfinite(s).all() for s in sources):
                    raise ValueError('Dyson operator/source contains nonfinite values.')
                try:
                    retarded = np.linalg.inv(operator)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(f'Singular full-device Dyson solve at q={q}, w={w}.') from exc
                advanced = _adjoint(retarded)
                lesser = retarded @ sources[0] @ advanced
                greater = retarded @ sources[1] @ advanced
                for target, field in zip(arrays, (retarded,lesser,greater), strict=True):
                    if not np.isfinite(field).all():
                        raise ValueError('Nonfinite full-device Green function.')
                    target[q,w] = field[active]
                _local_checks(full_metrics, retarded, lesser, greater)
                if collector is not None:
                    collector.observe(q, w, omega, mesh.addresses[q], operator, retarded,
                                      sources[0], sources[1], lesser, greater)
                if omega > 0:
                    for index,(contact,support) in enumerate(zip((left,right), supports, strict=True)):
                        currents[index] += omega * _exchange(
                            contact.spectrum.lesser[q,w], contact.spectrum.greater[q,w],
                            lesser[support], greater[support],
                        )
                    if sigma_input is not None:
                        currents[2] += omega * _exchange(
                            sigma_input.lesser[q,w], sigma_input.greater[q,w],
                            lesser[active], greater[active],
                        )
        if collector is not None:
            full_error = active_error = None
            try:
                checks['full_green_local'] = _finish_checks(full_metrics, rtol, atol, 'full Green')
            except ValueError as exc:
                full_error = str(exc)
            green = PackedMapSpectrum(*arrays)
            try:
                checks['active_green'] = _field_checks(mesh, grid, green, rtol, atol, 'active Green')
            except ValueError as exc:
                active_error = str(exc)
            diagnostics = collector.finish(
                full_metrics, full_green_validation_error=full_error,
                active_green_validation_error=active_error, validation_checks=checks,
            )
            return PackedGreenDiagnostic(True, full_error is None and active_error is None,
                                         diagnostics, plan, diagnostic_bytes,
                                         plan.peak_owned_bytes+diagnostic_bytes)
        checks['full_green_local'] = _finish_checks(full_metrics, rtol, atol, 'full Green')
        green = PackedMapSpectrum(*arrays)
        checks['active_green'] = _field_checks(mesh, grid, green, rtol, atol, 'active Green')
        if coupling == 0 or a == 0:
            mapped_less, mapped_greater = np.zeros(shape, dtype=complex), np.zeros(shape, dtype=complex)
        else:
            mapped_less, mapped_greater = cubic_born_self_energy_q_packed_fft(
                grid, vertex, green.lesser, green.greater,
                workers=workers, external_tile=external_tile, bin_tile=bin_tile,
                max_bytes=plan.born_peak_bytes, max_contraction_work=work,
            )
            mapped_less *= squared
            mapped_greater *= squared
        _finite(mapped_less, 'mapped lesser')
        _finite(mapped_greater, 'mapped greater')
        mapped_retarded = np.empty(shape, dtype=complex)
        if a:
            for q in range(nq):
                mapped_retarded[q] = retarded_from_keldysh(
                    mapped_less[q], mapped_greater[q], grid, hilbert_backend='fft',
                )
        mapped = PackedMapSpectrum(mapped_retarded, mapped_less, mapped_greater)
        _spectrum(mapped, shape, 'mapped Sigma')
        checks['mapped_sigma'] = _field_checks(mesh, grid, mapped, rtol, atol, 'mapped Sigma')
    values = [float(scale.to_physical_current(float(v*grid.integration_weight/nq))) for v in currents]
    if not np.isfinite(values).all():
        raise ValueError('Current conversion overflow.')
    for spectrum in (green, mapped):
        for name in _COMPONENTS:
            getattr(spectrum, name).setflags(write=False)
    return PackedBornMapEvaluation(green, mapped, TransverseCurrents(*values),
                                   MappingProxyType(checks), plan, sigma_input is None, coupling)


__all__ = [
    'MaterialMapPlan',
    'PackedBornMapEvaluation',
    'PackedGreenDiagnostic',
    'PackedMapSpectrum',
    'evaluate_packed_born_map',
    'plan_material_born_map',
]
