"""Frozen-input acoustic-coordinate analysis, retaining every Born component.

The rigid-translation/complement split is an exact bookkeeping decomposition
at Gamma, not a change to the Green fields or interaction Hamiltonian. Every
ordered CC sector pair, including interference, is retained. Only scalar
summaries leave this bounded diagnostic.
"""

from __future__ import annotations

from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .transverse_compact_contacts import CompactTransverseContact
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
_SECTORS = ("PCP", "PCQ", "QCP", "QCQ")


def plan_acoustic_sector_dimensions(
    n_q, n_w, ndof, *, internal_half_width=2, workers=8,
    external_tile=16, bin_tile=1,
):
    """Dimension-only payload/work admission, excluding caller-owned arrays."""
    nq = _integer(n_q, "n_q", maximum=64)
    nw = _integer(n_w, "n_w", minimum=3, maximum=4097)
    n = _integer(ndof, "ndof", minimum=3, maximum=512)
    width = _integer(internal_half_width, "internal_half_width", minimum=0, maximum=2)
    threads = _integer(workers, "workers", maximum=128)
    tile = min(_integer(external_tile, "external_tile"), n)
    bins = _integer(bin_tile, "bin_tile")
    if nw % 2 != 1 or width > nw//2:
        raise ValueError("Odd n_w and sufficient signed low-frequency bins are required.")
    low = 2*width+1
    raw_work, sector_work = 12*nq*low*n**4, 48*n**4
    retained = 2*16*nq*low*n*n
    partitions = 16*nq*low*4*n*n
    # Conservative allowance for both first/second16-pair batches, output,
    # sector/aggregate matrices, one conjugated vertex and worker temporaries.
    effective = min(threads, (max(low, 16)+bins-1)//bins)
    contraction = 16*(n**3+96*n*n+effective*(2*tile*n*n+tile*n))
    point = 64*16*n*n
    peak = retained+partitions+contraction+point+8*nq*nq
    return {"n_q": nq, "n_w": nw, "ndof": n,
        "internal_half_width": width, "n_low_bins": low,
        "dyson_evaluations": nq*low, "raw_contraction_work": raw_work,
        "sector_contraction_work": sector_work,
        "contraction_work": raw_work+sector_work,
        "retained_green_bytes": retained, "partition_workspace_bytes": partitions,
        "contraction_workspace_bytes": contraction, "point_workspace_bytes": point,
        "peak_owned_bytes": peak, "caller_payload_excluded": True,
        "workers": threads, "external_tile": tile, "bin_tile": bins,
        "rss_guarantee": False}


def _blocks(value, central, outer):
    return {name: _summary(value[np.ix_(rows, columns)])
        for name, rows, columns in (("full", np.arange(len(value)), np.arange(len(value))),
            ("central_central", central, central), ("outer_outer", outer, outer),
            ("central_outer", central, outer), ("outer_central", outer, central))}


def _closure(first, second, rtol, atol):
    delta = first-second
    _finite(delta, "decomposition closure")
    error = float(np.max(np.abs(delta), initial=0.))
    scale = float(max(np.max(np.abs(first), initial=0.), np.max(np.abs(second), initial=0.)))
    allowance = float(atol+rtol*scale)
    passed = error <= allowance
    record = {"error": error, "scale": scale, "allowance": allowance,
        "error_to_allowance_ratio": error/allowance if allowance else (0. if error == 0 else None),
        "frobenius_error": float(np.linalg.norm(delta)), "passed": passed,
        "bookkeeping_only": True, "physical_acceptance": False}
    if not passed:
        raise ValueError("Acoustic-sector exact decomposition bookkeeping does not close.")
    return record


def _translations(value, n, rtol, atol):
    if not isinstance(value, np.ndarray) or value.shape != (n, 3) or value.dtype.kind not in "fc":
        raise ValueError("translations must be a finite ndof-by3 floating array.")
    _finite(value, "normalized rigid translations")
    gram = value.conj().T@value
    _closure(gram, np.eye(3), rtol, atol)
    return value


def audit_vertex_translations(vertex, translations, central_dofs, *,
                              identity_rtol=1e-8, identity_atol=1e-11,
                              admit=lambda: None):
    """Contract every zero-momentum leg without modifying any coefficient."""
    n, nq = vertex.ndof, vertex.mesh.n_qpoints
    t = _translations(translations, n, identity_rtol, identity_atol)
    central, outer = _domains(central_dofs, n)
    rows = []
    for leg in range(3):
        for k in range(nq):
            admit()
            q, p = ((0, k), (k, 0), (k, k))[leg]
            v = vertex.coefficients[q, p]
            contracted = np.tensordot(v, t, axes=((leg,), (0,)))
            for direction in range(3):
                matrix = contracted[:, :, direction]
                _finite(matrix, "translation contraction")
                rows.append({"zero_momentum_leg": leg, "other_q_index": k,
                    "vertex_q_index": q, "vertex_p_index": p, "direction": direction,
                    "free_leg_indices": [x for x in range(3) if x != leg],
                    "blocks": _blocks(matrix, central, outer)})
    return {"diagnostic_only": True, "residual_is_not_an_execution_gate": True,
        "momentum_convention": "vertex legs carry (-q,p,q-p)",
        "normalization": "full-device massweighted rigid translations; T^H T=I3",
        "central_restriction_renormalized": False,
        "global_asr_restoration_claimed": False, "rows": rows}


def _domains(central_dofs, n):
    values = np.asarray(central_dofs)
    if (values.ndim != 1 or values.dtype.kind not in "iu" or len(values) == 0
            or len(values) >= n or len(np.unique(values)) != len(values)
            or np.any(values < 0) or np.any(values >= n)):
        raise ValueError("central_dofs must be unique valid indices with nonempty complement.")
    central = values.astype(np.int64, copy=False)
    return central, np.setdiff1d(np.arange(n), central)


def study_acoustic_sector_diagnostic(
    grid, vertex, dynamical, left, right, sigma_input, translations, central_dofs, *,
    g=.03, internal_half_width=2, workers=8, external_tile=16, bin_tile=1,
    max_bytes=4*1024**3, max_contraction_work=1_000_000_000_000,
    max_dyson_evaluations=20, deadline_monotonic=None, time_reserve_seconds=0,
    identity_rtol=1e-8, identity_atol=1e-11,
):
    """One fixed native input, low5bin raw Born terms and exact Gamma CC split."""
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(vertex, PackedTransverseCubicVertex):
        raise TypeError("Supply a signed grid and packed transverse cubic vertex.")
    nq, nw, n = vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof
    if vertex.ndof != n or not np.array_equal(vertex.active_dofs, np.arange(n)):
        raise ValueError("The diagnostic requires the unchanged full-coordinate vertex.")
    plan = plan_acoustic_sector_dimensions(nq, nw, n, internal_half_width=internal_half_width,
        workers=workers, external_tile=external_tile, bin_tile=bin_tile)
    limit, work_limit = _integer(max_bytes, "max_bytes"), _integer(max_contraction_work, "max_contraction_work")
    if plan["peak_owned_bytes"] > limit or plan["contraction_work"] > work_limit:
        raise ValueError("Acoustic diagnostic exceeds numeric memory/work admission.")
    if plan["dyson_evaluations"] > _integer(max_dyson_evaluations, "max_dyson_evaluations"):
        raise ValueError("Acoustic diagnostic exceeds max_dyson_evaluations.")
    for value, name in ((g, "g"), (identity_rtol, "identity_rtol"),
                        (identity_atol, "identity_atol"), (time_reserve_seconds, "time_reserve_seconds")):
        if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
                or np.iscomplexobj(value) or not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative real scalar.")
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite.")

    def admit():
        if deadline_monotonic is not None and monotonic()+time_reserve_seconds >= deadline_monotonic:
            raise TimeoutError("Acoustic diagnostic time admission exhausted.")

    admit()
    central, outer = _domains(central_dofs, n)
    t = _translations(translations, n, identity_rtol, identity_atol)
    raw = _encoded(dynamical, (nq, n, n), "dynamical")
    fields = _spectrum(sigma_input, (nq, nw, n, n), "sigma_input")
    contacts = []
    for contact in (left, right):
        if not isinstance(contact, CompactTransverseContact):
            raise TypeError("Contacts must be CompactTransverseContact instances.")
        if (contact.full_ndof != n or contact.mesh.shape != vertex.mesh.shape
                or contact.grid.n_points != nw or contact.grid.step != grid.step):
            raise ValueError("Contact dimensions, mesh or grid differ.")
        contacts.append((np.ix_(contact.device_indices, contact.device_indices), contact.spectrum))
    z, width = grid.zero_index, plan["internal_half_width"]
    indices = tuple(range(z-width, z+width+1))
    active = np.ix_(np.arange(n), np.arange(n))
    retained_c = np.empty((nq, len(indices), n, n), np.complex128)
    retained_d = np.empty_like(retained_c)
    groups = np.zeros((nq, len(indices), 4, n, n), np.complex128)
    routes = difference_q_indices(vertex.mesh, max_bytes=limit)
    points = []
    kernel = {"external_tile": external_tile, "workers": workers, "bin_tile": bin_tile,
              "max_bytes": limit, "max_contraction_work": work_limit}
    born_factor, coupling = .5j*grid.integration_weight/nq, float(g)**2
    with threadpool_limits(limits=1, user_api="blas"):
        audit = audit_vertex_translations(vertex, t, central,
            identity_rtol=identity_rtol, identity_atol=identity_atol, admit=admit)
        for q in range(nq):
            _finite(raw[q], "dynamical")
            for local, w in enumerate(indices):
                admit()
                for value in fields:
                    _finite(value[q, w], "native low-frequency input")
                operator, cs, ds = _assemble(grid.values[w], raw[q], contacts, active,
                    fields, q, w, np.complex128)
                for value in (operator, cs, ds):
                    _finite(value, "assembled native operator/source")
                r = np.linalg.inv(operator)
                c, d = r@cs@_adjoint(r), r@ds@_adjoint(r)
                for value in (r, c, d):
                    _finite(value, "fresh native Green product")
                retained_c[q, local], retained_d[q, local] = c, d
                points.append({"q_index": q, "frequency_index": w,
                    "frequency": float(grid.values[w]),
                    "statistical_green": _blocks(c, central, outer),
                    "spectral_green": _blocks(d, central, outer),
                    "context": _contexts(operator, cs, ds, r, c, d)})
        for p in range(nq):
            partner = int(routes[0, p])
            c1, d1 = retained_c[p], retained_d[p]
            c2, d2 = retained_c[partner, ::-1], retained_d[partner, ::-1]
            for term_index, (first, second) in enumerate(((c1, c2), (d1, d2), (c1, d2), (d1, c2))):
                admit()
                contracted = contract_packed_fft_bins(vertex.coefficients[0, p], first, second, **kernel)
                for local, w in enumerate(indices):
                    groups[p, local, term_index] = contracted[local]*(.25 if term_index == 1 else 1.)
        groups *= born_factor
        groups *= coupling
        _finite(groups, "weighted partial Born")
        gamma_c = retained_c[0, width]
        p = t@t.conj().T
        q = np.eye(n)-p
        sectors = (p@gamma_c@p, p@gamma_c@q, q@gamma_c@p, q@gamma_c@q)
        c_closure = _closure(sum(sectors), gamma_c, identity_rtol, identity_atol)
        first = np.stack([sectors[i] for i in range(4) for _ in range(4)])
        second = np.stack([sectors[j] for _ in range(4) for j in range(4)])
        admit()
        sector_values = contract_packed_fft_bins(vertex.coefficients[0, 0], first, second, **kernel)
        sector_values *= born_factor
        sector_values *= coupling
        _finite(sector_values, "weighted acoustic-sector CC arithmetic")
        sector_sum = np.sum(sector_values, axis=0)
        raw_gamma_cc = groups[0, width, 0]
        cc_closure = _closure(sector_sum, raw_gamma_cc, identity_rtol, identity_atol)
        complementary = sector_values[15]
        involves_translation = np.sum(sector_values[:15], axis=0)
        sector_rows = [{"first_sector": _SECTORS[i], "second_sector": _SECTORS[j],
            "blocks": _blocks(sector_values[4*i+j], central, outer),
            "algebraic_component_not_individual_physical_channel": True}
            for i in range(4) for j in range(4)]
        sector_record = {"internal_q_index": 0, "internal_frequency_index": z,
            "decomposition_scope": "Gamma/zero statistical Green only; rigid coordinates are not an eigenmode basis",
            "definitions": {"P": "T T^H", "Q": "I-P", "sectors": list(_SECTORS)},
            "all_ordered_pairs_retained": True, "interference_discarded": False,
            "fields_modified_for_solver": False, "statistical_reconstruction": c_closure,
            "CC_reconstruction": cc_closure,
            "green_sectors": {name: _blocks(value, central, outer) for name, value in zip(_SECTORS, sectors, strict=True)},
            "ordered_CC_pairs": sector_rows, "raw_CC": _blocks(raw_gamma_cc, central, outer),
            "summed_CC": _blocks(sector_sum, central, outer),
            "complement_complement_CC": _blocks(complementary, central, outer),
            "all_translation_involving_CC": _blocks(involves_translation, central, outer),
            "norms_are_not_additive_fractions": True}
        per_q = []
        for p_index in range(nq):
            partitions, signed_bins = {}, []
            for local, w in enumerate(indices):
                values = groups[p_index, local]
                signed_bins.append({"frequency_index": w, "frequency_offset": w-z,
                    "partner_frequency_index": 2*z-w,
                    "terms": {term: _blocks(values[i], central, outer) for i, term in enumerate(_TERMS)},
                    "statistical": _blocks(values[0]+values[1], central, outer),
                    "spectral": _blocks(values[2]+values[3], central, outer)})
            for part in range(width+1):
                values = np.sum(groups[p_index, [local for local, w in enumerate(indices)
                                                if abs(w-z) == part]], axis=0)
                partitions["zero" if part == 0 else f"near{part}"] = {
                    "terms": {term: _blocks(values[i], central, outer) for i, term in enumerate(_TERMS)},
                    "statistical": _blocks(values[0]+values[1], central, outer),
                    "spectral": _blocks(values[2]+values[3], central, outer)}
            per_q.append({"internal_q_index": p_index, "partner_q_index": int(routes[0, p_index]),
                          "signed_bins": signed_bins, "partitions": partitions})
        totals = np.sum(groups, axis=(0, 1))
    return {"schema": "negfpy.acoustic-sector-diagnostic.v1", "diagnostic_only": True,
        "accepted_restart": False, "scba_evaluation": False, "full_born_map": False,
        "native_spectral_channel_retained": True, "cached_green_or_born_used": False,
        "green_operator_and_sources_unchanged": True, "local_contexts_are_diagnostic_only": True,
        "original_scba_guards_and_convergence_checks_unchanged": True,
        "basis_decomposition_is_not_physical_projection": True,
        "g": float(g), "identity_rtol": float(identity_rtol), "identity_atol": float(identity_atol),
        "central_dofs": central.tolist(), "outer_dofs": outer.tolist(),
        "plan": plan, "dyson_evaluations": len(points), "points": points,
        "translation_audit": audit, "gamma_zero_CC_sectors": sector_record,
        "born": {"external_q_index": 0, "external_frequency_index": z,
            "internal_frequency_indices": list(indices),
            "partner_frequency_indices": list(reversed(indices)),
            "prefactor": _complex(born_factor*coupling),
            "g_squared_applied_after_raw_born_factor": True,
            "per_internal_q": per_q,
            "term_totals": {term: _blocks(totals[i], central, outer) for i, term in enumerate(_TERMS)},
            "partial_total": {"statistical": _blocks(totals[0]+totals[1], central, outer),
                              "spectral": _blocks(totals[2]+totals[3], central, outer)},
            "norms_are_not_additive_contribution_fractions": True}}
