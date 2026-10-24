"""Bounded fixed-input native-channel low-frequency investigation.

This helper evaluates only the stated Dyson points and a partial Born sum. It
does not run SCBA, publish fields, reconstruct retarded self-energy, or modify
any input. Native spectral sources and products remain independent of the
statistical channel. Local identity contexts are diagnostics, not new gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .transverse_channel_feedback import ChannelSpectrum
from .transverse_compact_contacts import CompactTransverseContact
from .transverse_packed import PackedTransverseCubicVertex, contract_packed_fft_bins
from .transverse_q import difference_q_indices
from .transverse_zero_reference import _refine_inverse

_CHANNELS = ("retarded", "statistical", "spectral")
_TERMS = ("CC", "DD_over_4", "CD", "DC")
_DEFAULT_ELEMENTS = ((9, 102), (96, 102), (102, 102))
_LD = np.clongdouble


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its bounded range.")
    return int(value)


@dataclass(frozen=True, slots=True)
class NativeZeroDiagnosticPlan:
    """Owned numeric payload excludes caller arrays and BLAS/LAPACK scratch.

    The reference workspace reserves 160 extended and 64 double full matrices.
    Retained products contain only active C/D for the requested low bins. The
    estimate is deliberately conservative, but is not a process RSS guarantee.
    """

    n_q: int
    n_w: int
    full_ndof: int
    active_ndof: int
    half_width: int
    n_low_bins: int
    dyson_evaluations: int
    contraction_work: int
    retained_green_bytes: int
    partition_workspace_bytes: int
    point_workspace_bytes: int
    contraction_workspace_bytes: int
    caller_payload_excluded: bool
    peak_owned_bytes: int
    external_tile: int
    workers: int
    bin_tile: int


def plan_native_zero_diagnostic_dimensions(
    n_q, n_w, full_ndof, active_ndof, *, half_width=2, external_tile=16,
    workers=8, bin_tile=1,
):
    """Plan without reading or allocating any material field."""
    nq = _integer(n_q, "n_q", maximum=64)
    nw = _integer(n_w, "n_w", minimum=3, maximum=4097)
    d = _integer(full_ndof, "full_ndof", maximum=512)
    a = _integer(active_ndof, "active_ndof", maximum=d)
    width = _integer(half_width, "half_width", minimum=0, maximum=2)
    tile = _integer(external_tile, "external_tile")
    threads = _integer(workers, "workers", maximum=128)
    bins = _integer(bin_tile, "bin_tile")
    if nw % 2 != 1 or width > nw // 2:
        raise ValueError("Odd n_w and sufficient signed low-frequency bins are required.")
    low = 2 * width + 1
    retained = 2 * 16 * nq * low * a**2
    partitions = 40 * 16 * a**2
    point = (160 * np.dtype(_LD).itemsize + 64 * 16) * d**2
    effective = min(threads, (low + bins - 1) // bins)
    contract = 16 * (low*a*a + a**3 + effective*(2*min(tile, a)*a*a
                                                       + min(tile, a)*a)) + a*a
    peak = retained + partitions + max(point, contract) + 8*nq*nq
    return NativeZeroDiagnosticPlan(
        nq, nw, d, a, width, low, nq*low, 12*nq*low*a**4,
        retained, partitions, point, contract, True, peak, tile, threads, bins,
    )


def _encoded(value, shape, name):
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"{name} must be a numpy.ndarray with shape {shape}.")
    if value.dtype != np.dtype(np.complex128):
        raise TypeError(f"{name} must contain encoded complex128 fields.")
    return value


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values.")


def _spectrum(value, shape, name):
    if not isinstance(value, ChannelSpectrum):
        raise TypeError(f"{name} must be a ChannelSpectrum.")
    return tuple(_encoded(getattr(value, key), shape, f"{name}.{key}")
                 for key in _CHANNELS)


def _adjoint(value):
    return value.conj().T


def _maximum(value):
    return np.max(np.abs(value), initial=0)


def _norm_inf(value):
    return np.max(np.sum(np.abs(value), axis=1), initial=0)


def _number(value):
    result = float(value)
    return result if np.isfinite(result) else None


def _complex(value):
    return {"real": _number(value.real), "imag": _number(value.imag)}


def _summary(value, elements=()):
    # Extended products can overflow even for finite original products. Preserve
    # that limitation as scalar context rather than a repair or acceptance gate.
    finite = bool(np.isfinite(value).all())
    if not finite:
        return {"available": False, "reason": "nonfinite diagnostic arithmetic"}
    magnitude = np.abs(value)
    location = np.unravel_index(int(np.argmax(magnitude)), value.shape)
    return {"available": True, "max_abs": _number(magnitude[location]),
            "frobenius": _number(np.sqrt(np.sum(magnitude*magnitude))),
            "norm_inf": _number(_norm_inf(value)),
            "max_abs_location": [int(x) for x in location],
            "value_at_max_abs": _complex(value[location]),
            "selected_elements": [{"row": i, "column": j,
                                   **_complex(value[i, j])} for i, j in elements]}


def _difference(first, second):
    error, scale = _maximum(first-second), max(_maximum(first), _maximum(second))
    if not np.isfinite(error) or not np.isfinite(scale):
        return {"available": False, "reason": "nonfinite diagnostic arithmetic",
                "diagnostic_only": True}
    allowance = np.longdouble("1e-11") + np.longdouble("1e-8")*scale
    return {"available": True, "error": _number(error), "scale": _number(scale),
            "relative_error": _number(error/scale) if scale else 0.0,
            "pointwise_allowance": _number(allowance),
            "pointwise_allowance_ratio": _number(error/allowance),
            "diagnostic_only": True, "acceptance": False}


def _contexts(operator, cs, ds, r, c, d):
    source_delta = ds-(_adjoint(operator)-operator)
    propagated = r@source_delta@_adjoint(r)
    primary_delta = d-(r-_adjoint(r))
    identity = np.eye(len(r), dtype=r.dtype)
    denominator = max(_norm_inf(operator)*_norm_inf(r), 1)
    return {
        "diagnostic_only": True,
        "raw_source_identity": _difference(ds, _adjoint(operator)-operator),
        "independent_spectral_green_identity": _difference(d, r-_adjoint(r)),
        "propagated_source_identity": _summary(propagated),
        "identity_product_remainder": _difference(primary_delta, propagated),
        "left_inverse_absolute_residual": _number(_norm_inf(identity-operator@r)),
        "right_inverse_absolute_residual": _number(_norm_inf(identity-r@operator)),
        "left_normalized_backward_residual": _number(
            _norm_inf(identity-operator@r)/denominator),
        "right_normalized_backward_residual": _number(
            _norm_inf(identity-r@operator)/denominator),
        "norm_inf_condition_estimate": _number(_norm_inf(operator)*_norm_inf(r)),
        "source_statistical_to_spectral_max_ratio": (
            _number(_maximum(cs)/_maximum(ds)) if _maximum(ds) else None),
        "green_statistical_to_spectral_max_ratio": (
            _number(_maximum(c)/_maximum(d)) if _maximum(d) else None),
    }


def _assemble(omega, dynamical, contacts, active, fields, q, w, dtype):
    # Keep the production operation order in complex128. Extended references
    # promote the assembled result separately, without reassembling it.
    operator = dtype(omega)**2*np.eye(len(dynamical), dtype=dtype)-dynamical.astype(dtype)
    cs, ds = (np.zeros(operator.shape, dtype=dtype) for _ in range(2))
    for support, contact in contacts:
        r = contact.retarded[q, w].astype(dtype)
        less, greater = contact.lesser[q, w].astype(dtype), contact.greater[q, w].astype(dtype)
        operator[support] -= r
        cs[support] += dtype(.5)*(greater+less)
        ds[support] += greater-less
    operator[active] -= fields[0][q, w].astype(dtype)
    cs[active] += fields[1][q, w].astype(dtype)
    ds[active] += fields[2][q, w].astype(dtype)
    return operator, cs, ds


def _reference(operator, cs, ds, r, c, d, max_refinements):
    result = {"diagnostic_only": True, "physical_inputs_unchanged": True,
              "used_in_partial_born": False, "certified_forward_accuracy": False,
              "complex128_epsilon": float(np.finfo(np.float64).eps),
              "longdouble_epsilon": float(np.finfo(np.longdouble).eps)}
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        return {**result, "available": False, "reason": "no additional longdouble precision"}
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # Promote the fixed encoded double operator and sources, rather than
        # reassembling a different rounded operator in extended precision.
        extended_operator, extended_cs, extended_ds = (
            value.astype(_LD) for value in (operator, cs, ds))
        fixed_r = r.astype(_LD)
        fixed_c = fixed_r@extended_cs@_adjoint(fixed_r)
        fixed_d = fixed_r@extended_ds@_adjoint(fixed_r)
        result.update(available=True,
                      operator_and_sources="promoted fixed encoded complex128 assembly",
                      fixed_double_inputs_product_comparison={
                          "statistical": _difference(c.astype(_LD), fixed_c),
                          "spectral": _difference(d.astype(_LD), fixed_d)})
        refined, history = _refine_inverse(extended_operator, r, max_refinements)
        refined_c = refined@extended_cs@_adjoint(refined)
        refined_d = refined@extended_ds@_adjoint(refined)
        result.update(refinement=history,
                      refined_inverse_comparison=_difference(r.astype(_LD), refined),
                      refined_product_comparison={
                          "statistical": _difference(c.astype(_LD), refined_c),
                          "spectral": _difference(d.astype(_LD), refined_d)},
                      refined_context=_contexts(extended_operator, extended_cs,
                                               extended_ds, refined, refined_c, refined_d))
    return result


def _admit(deadline_monotonic, time_reserve_seconds):
    if deadline_monotonic is not None and monotonic()+time_reserve_seconds >= deadline_monotonic:
        raise TimeoutError("Native diagnostic time admission exhausted; no further work started.")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def study_native_zero_diagnostic(
    grid, vertex, dynamical, left, right, sigma_input, mapped_sigma, *, g=.03,
    max_bytes=2*1024**3, max_contraction_work=100_000_000_000,
    max_dyson_evaluations=20, max_refinements=4, half_width=2, external_q=0,
    selected_elements=_DEFAULT_ELEMENTS, workers=8, external_tile=16, bin_tile=1,
    deadline_monotonic=None, time_reserve_seconds=0,
):
    """Return scalar evidence for fixed inputs; no fields are returned or saved.

    The partial Born total includes all q routes, but only the internal zero and
    requested signed neighbouring bins. Its remainder is *inferred by subtraction
    from the saved total*, not an evaluation of other frequencies or an oracle.
    """
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if not isinstance(vertex, PackedTransverseCubicVertex):
        raise TypeError("vertex must be a PackedTransverseCubicVertex.")
    nq, nw, full, a = (vertex.mesh.n_qpoints, grid.n_points,
                       vertex.full_ndof, vertex.ndof)
    plan = plan_native_zero_diagnostic_dimensions(
        nq, nw, full, a, half_width=half_width, external_tile=external_tile,
        workers=workers, bin_tile=bin_tile)
    refinements = _integer(max_refinements, "max_refinements", minimum=0, maximum=4)
    qout = _integer(external_q, "external_q", minimum=0, maximum=nq-1)
    if qout != 0:
        raise ValueError("This bounded investigation requires external Gamma q index zero.")
    if (isinstance(g, (bool, np.bool_)) or not np.isscalar(g) or np.iscomplexobj(g)
            or not np.isfinite(g) or g < 0):
        raise ValueError("g must be a finite nonnegative real scalar.")
    if plan.peak_owned_bytes > _integer(max_bytes, "max_bytes"):
        raise ValueError("Native diagnostic owned numeric plan exceeds max_bytes.")
    if plan.contraction_work > _integer(max_contraction_work, "max_contraction_work"):
        raise ValueError("Native diagnostic exceeds max_contraction_work.")
    if plan.dyson_evaluations > _integer(max_dyson_evaluations, "max_dyson_evaluations"):
        raise ValueError("Native diagnostic exceeds max_dyson_evaluations.")
    if (not np.isfinite(time_reserve_seconds) or time_reserve_seconds < 0
            or (deadline_monotonic is not None and not np.isfinite(deadline_monotonic))):
        raise ValueError("Time admission values must be finite and reserve nonnegative.")
    _admit(deadline_monotonic, time_reserve_seconds)
    elements = tuple(tuple(pair) for pair in selected_elements)
    for pair in elements:
        if len(pair) != 2:
            raise ValueError("selected_elements must be matrix coordinate pairs.")
        for value in pair:
            _integer(value, "selected element", minimum=0, maximum=a-1)
    elements = tuple((int(i), int(j)) for i, j in elements)
    raw = _encoded(dynamical, (nq, full, full), "dynamical")
    shape = (nq, nw, a, a)
    fields = _spectrum(sigma_input, shape, "sigma_input")
    saved = _spectrum(mapped_sigma, shape, "mapped_sigma")
    contacts = []
    for contact in (left, right):
        if not isinstance(contact, CompactTransverseContact):
            raise TypeError("Contacts must be CompactTransverseContact instances.")
        if (contact.full_ndof != full or contact.mesh.shape != vertex.mesh.shape
                or contact.grid.n_points != nw or contact.grid.step != grid.step):
            raise ValueError("Contact grid, mesh and full dimension must match.")
        contacts.append((np.ix_(contact.device_indices, contact.device_indices), contact.spectrum))
    z = grid.zero_index
    frequencies = tuple(range(z-plan.half_width, z+plan.half_width+1))
    active = np.ix_(vertex.active_dofs, vertex.active_dofs)
    # Validate only the declared read scope, with matrix-size masks, before any
    # inverse or retained product allocation. Saved R is metadata, never used.
    for q in range(nq):
        _finite(raw[q], "dynamical")
        _finite(vertex.coefficients[qout, q], "vertex coefficients")
        for w in frequencies:
            for value in fields:
                _finite(value[q, w], "sigma_input low-frequency field")
            for _, contact in contacts:
                for name in ("retarded", "lesser", "greater"):
                    _finite(getattr(contact, name)[q, w], "contact low-frequency field")
    for value in saved[1:]:
        _finite(value[qout, z], "saved mapped zero-frequency field")
    retained_c = np.empty((nq, plan.n_low_bins, a, a), dtype=np.complex128)
    retained_d = np.empty_like(retained_c)
    points = []
    with threadpool_limits(limits=1, user_api="blas"):
        for q in range(nq):
            for local, w in enumerate(frequencies):
                _admit(deadline_monotonic, time_reserve_seconds)
                operator, cs, ds = _assemble(grid.values[w], raw[q], contacts, active,
                                              fields, q, w, np.complex128)
                _finite(operator, "assembled operator")
                _finite(cs, "assembled statistical source")
                _finite(ds, "assembled spectral source")
                r = np.linalg.inv(operator)
                c, d = r@cs@_adjoint(r), r@ds@_adjoint(r)
                for value in (r, c, d):
                    _finite(value, "fresh native Green product")
                retained_c[q, local], retained_d[q, local] = c[active], d[active]
                singular = np.linalg.svd(operator, compute_uv=False)
                point = {"q_index": q, "frequency_index": w,
                         "frequency": float(grid.values[w]),
                         "operator": _summary(operator), "statistical_source": _summary(cs),
                         "spectral_source": _summary(ds), "retarded_green": _summary(r),
                         "statistical_green_active": _summary(c[active], elements),
                         "spectral_green_active": _summary(d[active], elements),
                         "operator_singular_values": [_number(x) for x in singular],
                         "singular_condition_number": _number(singular[0]/singular[-1])
                         if singular[-1] else None,
                         "context": _contexts(operator, cs, ds, r, c, d)}
                if q == qout and w == z:
                    _admit(deadline_monotonic, time_reserve_seconds)
                    try:
                        point["extended_reference"] = _json_safe(_reference(
                            operator, cs, ds, r, c, d, refinements))
                    except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                        point["extended_reference"] = {
                            "available": False, "reason": str(exc), "diagnostic_only": True,
                            "used_in_partial_born": False, "certified_forward_accuracy": False}
                points.append(point)
        groups = {f"near{offset}" if offset else "zero": {
            term: np.zeros((a, a), dtype=np.complex128) for term in _TERMS}
            for offset in range(plan.half_width+1)}
        routes = difference_q_indices(vertex.mesh, max_bytes=max_bytes)
        # The raw-bin public kernel performs V X Y conj(V). No FFT or full-map
        # helper is called, and no spectral channel is recovered from contours.
        for p in range(nq):
            partner_q = int(routes[qout, p])
            c1, d1 = retained_c[p], retained_d[p]
            # external frequency is zero: partner = z + external_index - j.
            c2, d2 = retained_c[partner_q, ::-1], retained_d[partner_q, ::-1]
            for term, first, second in (("CC", c1, c2), ("DD_over_4", d1, d2),
                                         ("CD", c1, d2), ("DC", d1, c2)):
                _admit(deadline_monotonic, time_reserve_seconds)
                contracted = contract_packed_fft_bins(
                    vertex.coefficients[qout, p], first, second,
                    external_tile=external_tile, workers=workers, bin_tile=bin_tile,
                    max_bytes=max_bytes, max_contraction_work=max_contraction_work)
                for local, w in enumerate(frequencies):
                    offset = abs(w-z)
                    key = f"near{offset}" if offset else "zero"
                    groups[key][term] += contracted[local] * (.25 if term == "DD_over_4" else 1)
        # Match production order: raw Born factor, then g squared.
        born_factor = .5j*grid.integration_weight/nq
        factor = born_factor*float(g)**2
        totals = {term: np.zeros((a, a), dtype=np.complex128) for term in _TERMS}
        partition_records = {}
        for key, values in groups.items():
            for term in _TERMS:
                values[term] *= born_factor
                values[term] *= float(g)**2
                totals[term] += values[term]
            partition_records[key] = {
                "terms": {term: _summary(values[term], elements) for term in _TERMS},
                "statistical": _summary(values["CC"]+values["DD_over_4"], elements),
                "spectral": _summary(values["CD"]+values["DC"], elements),
            }
        partial_c, partial_d = totals["CC"]+totals["DD_over_4"], totals["CD"]+totals["DC"]
        for value in (*totals.values(), partial_c, partial_d):
            _finite(value, "partial Born arithmetic")
        saved_c, saved_d = saved[1][qout, z], saved[2][qout, z]
        born = {"external_q_index": qout, "external_frequency_index": z,
                "internal_frequency_indices": list(frequencies),
                "partner_frequency_rule": "zero_index + external_frequency_index - internal_index",
                "partner_q_indices": [int(x) for x in routes[qout]],
                "prefactor": _complex(factor), "g_squared_applied_after_raw_born_factor": True,
                "partitions": partition_records,
                "term_totals": {term: _summary(value, elements) for term, value in totals.items()},
                "partial_total": {"statistical": _summary(partial_c, elements),
                                  "spectral": _summary(partial_d, elements)},
                "saved_mapped": {"statistical": _summary(saved_c, elements),
                                 "spectral": _summary(saved_d, elements)},
                "inferred_remainder": {
                    "status": "INFERRED saved mapped total minus partial low-frequency sum",
                    "direct_other_frequency_evaluation": False, "oracle": False,
                    "statistical": _summary(saved_c-partial_c, elements),
                    "spectral": _summary(saved_d-partial_d, elements)}}
    return {"schema": "negfpy.native-zero-diagnostic.v1", "diagnostic_only": True,
            "physical_inputs_unchanged": True, "accepted_restart": False,
            "scba_evaluation": False, "full_born_map": False,
            "native_spectral_channel_retained": True,
            "g": float(g), "identity_rtol": 1e-8, "identity_atol": 1e-11,
            "plan": asdict(plan), "dyson_evaluations": len(points),
            "extended_reference_points": 1, "max_refinements": refinements,
            "extended_reference_scope": {"q_index": qout, "frequency_index": z,
                                         "other_points_use_complex128_contexts_only": True},
            "points": points, "born": born}
