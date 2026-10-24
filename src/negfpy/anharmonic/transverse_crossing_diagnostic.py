"""Matched partial Born comparison with actual crossing-boundary IFC support.

A frozen native R/C/D input produces one common set of full-device Green
products.  The original cropped and crossing-support vertices then act on those
same products.  Only bounded low-frequency terms and scalar summaries are
returned; this is neither a full Born map nor an SCBA continuation.
"""

from __future__ import annotations

from itertools import product
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .transverse_compact_contacts import CompactTransverseContact
from .transverse_crossing_support import (
    CrossingSupportVertex,
    contract_rectangular_bins,
)
from .transverse_native_zero_diagnostic import (
    _adjoint,
    _assemble,
    _complex,
    _contexts,
    _encoded,
    _finite,
    _integer,
    _spectrum,
    _summary,
)
from .transverse_packed import PackedTransverseCubicVertex, contract_packed_fft_bins
from .transverse_q import difference_q_indices

_TERMS = ("CC", "DD_over_4", "CD", "DC")
_MODELS = ("baseline", "crossing")


def _offsets(values):
    if not isinstance(values, (tuple, list)) or not 1 <= len(values) <= 5:
        raise ValueError("external_offsets must contain one to five signed offsets.")
    result = tuple(_integer(x, "external offset", minimum=-2, maximum=2) for x in values)
    if len(set(result)) != len(result):
        raise ValueError("external_offsets must be distinct.")
    return result


def plan_crossing_diagnostic_dimensions(
    n_q, n_w, full_ndof, active_ndof, *, internal_half_width=2,
    external_offsets=(-1, 0, 1), workers=8, external_tile=16, bin_tile=1,
):
    """Plan numeric payload and all baseline/crossing work without field reads.

    Work counts complex multiply-accumulate visits in the two blocked kernels.
    Caller-owned material/vertex/checkpoint payloads, Python objects and library
    scratch are separate.  The plan is not an RSS or runtime guarantee.
    """
    nq = _integer(n_q, "n_q", maximum=64)
    nw = _integer(n_w, "n_w", minimum=3, maximum=4097)
    full = _integer(full_ndof, "full_ndof", maximum=512)
    a = _integer(active_ndof, "active_ndof", maximum=full)
    width = _integer(internal_half_width, "internal_half_width", minimum=0, maximum=2)
    offsets = _offsets(external_offsets)
    threads = _integer(workers, "workers", maximum=128)
    tile = min(_integer(external_tile, "external_tile"), a)
    bins = _integer(bin_tile, "bin_tile")
    first = min(-width, min(offsets)-width)
    last = max(width, max(offsets)+width)
    if nw % 2 != 1 or min(first+nw//2, nw//2-last) < 0:
        raise ValueError("Odd grid and sufficient bins for every exact frequency partner are required.")
    ninternal, nout, ngreen = 2*width+1, len(offsets), last-first+1
    batch = ninternal*nout
    baseline_work = 4*nq*batch*3*a**4
    crossing_work = 4*nq*batch*(2*a*full**3+a*a*full*full)
    retained = 2*16*nq*ngreen*full**2
    partitions = 16*2*nout*(width+1)*len(_TERMS)*a**2
    effective = min(threads, (batch+bins-1)//bins)
    batch_bytes = 4*16*batch*(full**2+a**2)
    kernel = 16*(batch*a*a+a*full*full+effective*(2*tile*full*full+tile*a))+full*full
    point = 64*16*full**2
    comparisons = 64*16*a**2
    peak = retained+partitions+comparisons+max(point, batch_bytes+kernel)+8*nq*nq
    return {
        "n_q": nq, "n_w": nw, "full_ndof": full, "active_ndof": a,
        "internal_half_width": width, "external_offsets": list(offsets),
        "n_internal_bins": ninternal, "n_external_bins": nout,
        "green_offset_first": first, "green_offset_last": last,
        "n_green_bins": ngreen, "dyson_evaluations": nq*ngreen,
        "batch_bins": batch, "baseline_contraction_work": baseline_work,
        "crossing_contraction_work": crossing_work,
        "contraction_work": baseline_work+crossing_work,
        "retained_green_bytes": retained, "partition_workspace_bytes": partitions,
        "batch_workspace_bytes": batch_bytes, "kernel_workspace_bytes": kernel,
        "peak_owned_bytes": peak, "caller_payload_excluded": True,
        "workers": threads, "external_tile": tile, "bin_tile": bins,
    }


def _ratio(numerator, denominator):
    if denominator == 0:
        return 1.0 if numerator == 0 else None
    value = float(numerator/denominator)
    return value if np.isfinite(value) else None


def _compare(baseline, crossing, elements):
    old, new = float(np.linalg.norm(baseline)), float(np.linalg.norm(crossing))
    difference = crossing-baseline
    _finite(difference, "matched Born difference")
    return {"baseline_frobenius": old, "crossing_frobenius": new,
            "crossing_to_baseline_frobenius_ratio": _ratio(new, old),
            "difference": _summary(difference, elements),
            "norms_are_not_additive_contribution_fractions": True}


def study_crossing_support_diagnostic(
    grid, baseline_vertex, crossing_vertex, dynamical, left, right, sigma_input, *,
    g=.03, internal_half_width=2, external_offsets=(-1, 0, 1), workers=8,
    external_tile=16, bin_tile=1, max_bytes=4*1024**3,
    max_contraction_work=1_000_000_000_000, max_dyson_evaluations=28,
    deadline_monotonic=None, time_reserve_seconds=0, selected_elements=None,
):
    """Compare two vertices on common fresh native Green fields, scalar-only.

    The external q is Gamma.  Every output frequency gets the exact partners
    ``zero_index + output_index - internal_index`` for the stated internal low
    band.  Internal zero and signed adjacent-bin groups are separate.  No saved
    Born image, remainder, current acceptance or convergence claim is evaluated.
    """
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if not isinstance(baseline_vertex, PackedTransverseCubicVertex):
        raise TypeError("baseline_vertex must be a PackedTransverseCubicVertex.")
    if not isinstance(crossing_vertex, CrossingSupportVertex):
        raise TypeError("crossing_vertex must be a CrossingSupportVertex.")
    mesh = baseline_vertex.mesh
    nq, nw, full, a = mesh.n_qpoints, grid.n_points, baseline_vertex.full_ndof, baseline_vertex.ndof
    plan = plan_crossing_diagnostic_dimensions(
        nq, nw, full, a, internal_half_width=internal_half_width,
        external_offsets=external_offsets, workers=workers,
        external_tile=external_tile, bin_tile=bin_tile)
    limit = _integer(max_bytes, "max_bytes")
    work_limit = _integer(max_contraction_work, "max_contraction_work")
    if plan["peak_owned_bytes"] > limit:
        raise ValueError("Crossing diagnostic numeric plan exceeds max_bytes.")
    if plan["contraction_work"] > work_limit:
        raise ValueError("Crossing diagnostic exceeds max_contraction_work.")
    if plan["dyson_evaluations"] > _integer(max_dyson_evaluations, "max_dyson_evaluations"):
        raise ValueError("Crossing diagnostic exceeds max_dyson_evaluations.")
    if (isinstance(g, (bool, np.bool_)) or not np.isscalar(g) or np.iscomplexobj(g)
            or not np.isfinite(g) or g < 0):
        raise ValueError("g must be a finite nonnegative real scalar.")
    if (not np.isfinite(time_reserve_seconds) or time_reserve_seconds < 0
            or (deadline_monotonic is not None and not np.isfinite(deadline_monotonic))):
        raise ValueError("Time admission values must be finite with nonnegative reserve.")

    def admit():
        if deadline_monotonic is not None and monotonic()+time_reserve_seconds >= deadline_monotonic:
            raise TimeoutError("Crossing diagnostic time admission exhausted.")

    admit()
    addresses = baseline_vertex.active_dofs
    if (crossing_vertex.full_ndof != full
            or not np.array_equal(crossing_vertex.external_dofs, addresses)):
        raise ValueError("Matched vertices must use identical external coordinates and full domain.")
    cross = _encoded(crossing_vertex.coefficients, (nq, a, full, full), "crossing vertex")
    raw = _encoded(dynamical, (nq, full, full), "dynamical")
    fields = _spectrum(sigma_input, (nq, nw, a, a), "sigma_input")
    offsets = tuple(plan["external_offsets"])
    width, z = plan["internal_half_width"], grid.zero_index
    internal = tuple(range(z-width, z+width+1))
    frequencies = tuple(range(z+plan["green_offset_first"], z+plan["green_offset_last"]+1))
    locals_by_w = {w: i for i, w in enumerate(frequencies)}
    outputs = tuple(z+offset for offset in offsets)
    first_locals = [locals_by_w[j] for _ in outputs for j in internal]
    second_locals = [locals_by_w[z+out-j] for out in outputs for j in internal]
    if selected_elements is None:
        selected_elements = ((9, 102), (96, 102), (102, 102)) if a >= 103 else tuple(product(range(min(a, 2)), repeat=2))
    elements = tuple(tuple(pair) for pair in selected_elements)
    for pair in elements:
        if len(pair) != 2:
            raise ValueError("selected_elements must contain matrix coordinate pairs.")
        for index in pair:
            _integer(index, "selected matrix index", minimum=0, maximum=a-1)
    elements = tuple((int(i), int(j)) for i, j in elements)
    contacts = []
    for contact in (left, right):
        if not isinstance(contact, CompactTransverseContact):
            raise TypeError("Contacts must be CompactTransverseContact instances.")
        if (contact.full_ndof != full or contact.mesh.shape != mesh.shape
                or contact.grid.n_points != nw or contact.grid.step != grid.step):
            raise ValueError("Contact dimensions, mesh or grid differ.")
        contacts.append((np.ix_(contact.device_indices, contact.device_indices), contact.spectrum))
    for q in range(nq):
        _finite(raw[q], "dynamical")
        _finite(cross[q], "crossing vertex")
        _finite(baseline_vertex.coefficients[0, q], "baseline vertex")
        for w in frequencies:
            for value in fields:
                _finite(value[q, w], "native low-frequency input")
            for _, contact in contacts:
                for name in ("retarded", "lesser", "greater"):
                    _finite(getattr(contact, name)[q, w], "contact low-frequency input")
    active = np.ix_(addresses, addresses)
    retained_c = np.empty((nq, len(frequencies), full, full), np.complex128)
    retained_d = np.empty_like(retained_c)
    points = []
    groups = {model: np.zeros((len(outputs), width+1, len(_TERMS), a, a), np.complex128)
              for model in _MODELS}
    routes = difference_q_indices(mesh, max_bytes=limit)
    with threadpool_limits(limits=1, user_api="blas"):
        for q in range(nq):
            for local, w in enumerate(frequencies):
                admit()
                operator, cs, ds = _assemble(grid.values[w], raw[q], contacts, active,
                                              fields, q, w, np.complex128)
                for value, label in ((operator, "operator"), (cs, "statistical source"),
                                     (ds, "spectral source")):
                    _finite(value, "assembled "+label)
                r = np.linalg.inv(operator)
                c, d = r@cs@_adjoint(r), r@ds@_adjoint(r)
                for value in (r, c, d):
                    _finite(value, "fresh native Green product")
                retained_c[q, local], retained_d[q, local] = c, d
                points.append({"q_index": q, "frequency_index": w,
                    "frequency": float(grid.values[w]),
                    "statistical_green_full": _summary(c), "spectral_green_full": _summary(d),
                    "statistical_green_active": _summary(c[active], elements),
                    "spectral_green_active": _summary(d[active], elements),
                    "context": _contexts(operator, cs, ds, r, c, d)})
        for p in range(nq):
            partner_q = int(routes[0, p])
            c1, d1 = retained_c[p, first_locals], retained_d[p, first_locals]
            c2, d2 = retained_c[partner_q, second_locals], retained_d[partner_q, second_locals]
            for term_index, (term, first, second) in enumerate((
                    ("CC", c1, c2), ("DD_over_4", d1, d2),
                    ("CD", c1, d2), ("DC", d1, c2))):
                admit()
                baseline = contract_packed_fft_bins(
                    baseline_vertex.coefficients[0, p],
                    first[:, addresses][:, :, addresses], second[:, addresses][:, :, addresses],
                    external_tile=external_tile, workers=workers, bin_tile=bin_tile,
                    max_bytes=limit, max_contraction_work=work_limit)
                admit()
                crossing = contract_rectangular_bins(
                    cross[p], first, second, external_tile=external_tile,
                    workers=workers, bin_tile=bin_tile,
                    max_bytes=limit, max_contraction_work=work_limit)
                for out_local in range(len(outputs)):
                    for j_local, j in enumerate(internal):
                        local = out_local*len(internal)+j_local
                        weight = .25 if term == "DD_over_4" else 1.0
                        partition = abs(j-z)
                        groups["baseline"][out_local, partition, term_index] += baseline[local]*weight
                        groups["crossing"][out_local, partition, term_index] += crossing[local]*weight
        born_factor, coupling = .5j*grid.integration_weight/nq, float(g)**2
        for value in groups.values():
            value *= born_factor
            value *= coupling
            _finite(value, "weighted partial Born arithmetic")
        born = []
        for out_local, out in enumerate(outputs):
            models, matrices = {}, {}
            for model in _MODELS:
                partition_records = {}
                values = groups[model][out_local]
                for partition in range(width+1):
                    term_values = values[partition]
                    partition_records["zero" if partition == 0 else f"near{partition}"] = {
                        "terms": {term: _summary(term_values[t], elements) for t, term in enumerate(_TERMS)},
                        "statistical": _summary(term_values[0]+term_values[1], elements),
                        "spectral": _summary(term_values[2]+term_values[3], elements)}
                totals = np.sum(values, axis=0)
                c, d = totals[0]+totals[1], totals[2]+totals[3]
                models[model] = {"partitions": partition_records,
                    "term_totals": {term: _summary(totals[t], elements) for t, term in enumerate(_TERMS)},
                    "partial_total": {"statistical": _summary(c, elements), "spectral": _summary(d, elements)}}
                matrices[model] = {"statistical": c, "spectral": d, "CC": totals[0],
                                   "zero_CC": values[0, 0]}
            comparisons = {name: _compare(matrices["baseline"][name], matrices["crossing"][name], elements)
                           for name in ("zero_CC", "CC", "statistical", "spectral")}
            born.append({"external_q_index": 0, "external_frequency_offset": offsets[out_local],
                "external_frequency_index": out, "external_frequency": float(grid.values[out]),
                "internal_frequency_indices": list(internal),
                "partner_frequency_indices": [z+out-j for j in internal],
                "partner_q_indices": [int(x) for x in routes[0]],
                "prefactor": _complex(born_factor*coupling),
                "g_squared_applied_after_raw_born_factor": True,
                **models, "comparison": comparisons})
    return {"schema": "negfpy.crossing-support-diagnostic.v1", "diagnostic_only": True,
        "accepted_restart": False, "scba_evaluation": False, "full_born_map": False,
        "current_acceptance": False, "native_spectral_channel_retained": True,
        "same_fresh_green_for_both_vertices": True, "green_operator_and_sources_unchanged": True,
        "retarded_self_energy_reconstructed": False, "cached_green_or_born_used": False,
        "original_scba_guards_and_convergence_checks_unchanged": True,
        "local_contexts_are_diagnostic_only": True, "g": float(g),
        "identity_rtol": 1e-8, "identity_atol": 1e-11,
        "plan": plan, "dyson_evaluations": len(points), "points": points, "born": born}
