"""Unmodified-operator/source diagnosis using every singular direction.

Singular vectors describe resolvent directions, not vibrational normal modes.
No singular value is floored, no direction is removed, and all off-diagonal
source couplings remain in the reconstructed product. Finite LU/SVD differences
are diagnostic numerical context, not new physical acceptance criteria.
"""

from __future__ import annotations

from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
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

_SOURCES = ("left", "right", "interaction")


def plan_soft_source_dimensions(
    n_q, n_w, ndof, *, n_layers=9, n_soft=12, include_zero_reference=True,
):
    """Dimension-only sequential-point workspace and bounded decomposition count."""
    nq = _integer(n_q, "n_q", maximum=64)
    nw = _integer(n_w, "n_w", minimum=3, maximum=4097)
    n = _integer(ndof, "ndof", minimum=3, maximum=512)
    layers = _integer(n_layers, "n_layers", maximum=n)
    soft = _integer(n_soft, "n_soft", maximum=n)
    if nw % 2 != 1 or n % layers:
        raise ValueError("Odd n_w and equal contiguous layer dimensions are required.")
    if not isinstance(include_zero_reference, (bool, np.bool_)):
        raise TypeError("include_zero_reference must be boolean.")
    roles = 2 if include_zero_reference else 1
    points = roles*(nq+2)
    # SVD/LAPACK scratch and Python objects are implementation-dependent. The
    # owned plan reserves128 complex matrix payloads, beyond explicit arrays.
    owned = 128*16*n*n+16*n*soft+8*n*8
    return {"n_q": nq, "n_w": nw, "ndof": n, "n_layers": layers,
        "dofs_per_layer": n//layers, "n_soft_reported": soft,
        "all_modes_retained": n, "role_count": roles,
        "points_per_role": nq+2, "dyson_evaluations": points,
        "svd_evaluations": points, "peak_owned_bytes": owned,
        "dense_cubic_work_units_upper_estimate": 128*points*n**3,
        "exact_flop_count": False, "caller_payload_excluded": True,
        "library_scratch_excluded": True, "rss_guarantee": False}


def _ratio(first, second):
    if second == 0:
        return 1. if first == 0 else None
    result = float(first/second)
    return result if np.isfinite(result) else None


def _comparison(first, second, rtol, atol):
    delta = first-second
    _finite(delta, "finite reconstruction comparison")
    error = float(np.max(np.abs(delta), initial=0.))
    scale = float(max(np.max(np.abs(first), initial=0.), np.max(np.abs(second), initial=0.)))
    allowance = float(atol+rtol*scale)
    return {"error": error, "scale": scale, "allowance": allowance,
        "error_to_allowance_ratio": _ratio(error, allowance),
        "frobenius_error": float(np.linalg.norm(delta)), "passed": bool(error <= allowance),
        "diagnostic_only": True, "physical_acceptance": False,
        "automatic_repair": False}


def _regions(n, layers):
    per_layer = n//layers
    # Production9-layer case is exactly central layers2..6. Tiny synthetic
    # cases retain one or more central layers for the same block summaries.
    boundary = min(2, (layers-1)//2)
    central_layers = list(range(boundary, layers-boundary))
    central = np.concatenate([np.arange(i*per_layer, (i+1)*per_layer) for i in central_layers])
    outer = np.setdiff1d(np.arange(n), central)
    return central_layers, central, outer


def _matrix_context(value, n_layers, central, outer):
    per_layer = len(value)//n_layers
    return {"full": _summary(value),
        "central_diagonal_block": _summary(value[np.ix_(central, central)]),
        "outer_diagonal_block": (_summary(value[np.ix_(outer, outer)]) if len(outer)
                                  else {"available": False, "reason": "empty outer region"}),
        "layer_diagonal_blocks": [{"layer": i,
            **_summary(value[i*per_layer:(i+1)*per_layer, i*per_layer:(i+1)*per_layer])}
            for i in range(n_layers)],
        "norms_are_not_additive_fractions": True}


def _hermiticity(value, rtol, atol):
    return _comparison(value, _adjoint(value), rtol, atol)


def _operator_projection(operator, translations):
    p = translations@translations.conj().T
    q = np.eye(len(operator))-p
    pp, pq, qp, qq = p@operator@p, p@operator@q, q@operator@p, q@operator@q
    norm = float(np.linalg.norm(operator))
    cross = float(np.hypot(np.linalg.norm(pq), np.linalg.norm(qp)))
    return {"definitions": {"P": "T T^H", "Q": "I-P"},
        "PAP": _summary(pp), "PAQ": _summary(pq),
        "QAP": _summary(qp), "QAQ": _summary(qq),
        "cross_block_frobenius_combined": cross,
        "cross_block_to_operator_frobenius_ratio": _ratio(cross, norm),
        "subspaces_assumed_invariant": False,
        "couplings_removed": False, "diagnostic_only": True}


def _analyze_point(operator, source_components, translations, n_layers, n_soft,
                   central, outer, operator_components, rtol, atol, admit):
    """All-mode SVD algebra on one unchanged encoded operator and sources."""
    for value in (operator, *source_components.values(), *operator_components.values()):
        _finite(value, "encoded point operator/source")
    admit()
    inverse = np.linalg.inv(operator)
    _finite(inverse, "fresh operator inverse")
    admit()
    u, singular, vh = np.linalg.svd(operator, full_matrices=True)
    if not np.isfinite(singular).all() or np.any(singular <= 0):
        raise ValueError("SVD operator is singular or nonfinite; no singular-value repair is permitted.")
    v = vh.conj().T
    source = sum(source_components.values())
    statistical = inverse@source@_adjoint(inverse)
    products = {name: inverse@value@_adjoint(inverse) for name, value in source_components.items()}
    h = 1j*source
    b = u.conj().T@h@u
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        amplified = b/singular[:, None]/singular[None, :]
        inverse_svd = (v/singular[None, :])@u.conj().T
    for value in (source, statistical, h, b, amplified, inverse_svd, *products.values()):
        _finite(value, "unmodified all-mode source/product arithmetic")
    reconstructed = v@amplified@v.conj().T
    _finite(reconstructed, "all-mode statistical reconstruction")
    diagonal_only = (v*np.diag(amplified)[None, :])@v.conj().T
    off_diagonal = reconstructed-diagonal_only
    source_b = {name: u.conj().T@(1j*value)@u for name, value in source_components.items()}
    gram = translations.conj().T@translations
    per_layer = len(operator)//n_layers
    modes = []
    for rank, index in enumerate(range(len(singular)-1, len(singular)-n_soft-1, -1), start=1):
        left, right = u[:, index], v[:, index]
        left_weight = [float(np.sum(abs(left[i*per_layer:(i+1)*per_layer])**2)) for i in range(n_layers)]
        right_weight = [float(np.sum(abs(right[i*per_layer:(i+1)*per_layer])**2)) for i in range(n_layers)]
        left_translation = abs(translations.conj().T@left)**2
        right_translation = abs(translations.conj().T@right)**2
        components = {}
        for name, matrix in operator_components.items():
            acted_right, acted_left = matrix@right, matrix.conj().T@left
            components[name] = {"uH_component_v": _complex(left.conj()@acted_right),
                "component_times_right_norm": float(np.linalg.norm(acted_right)),
                "component_adjoint_times_left_norm": float(np.linalg.norm(acted_left))}
        modes.append({"soft_rank": rank, "svd_descending_index": index,
            "singular_value": float(singular[index]),
            "condition_to_largest_singular_value": _ratio(singular[0], singular[index]),
            "source_B_ii": _complex(b[index, index]),
            "amplified_B_ii_over_s_squared": _complex(amplified[index, index]),
            "source_components": {name: {"B_ii": _complex(values[index, index]),
                "amplified_B_ii_over_s_squared": _complex(values[index, index]/singular[index]/singular[index])}
                for name, values in source_b.items()},
            "left_translation_direction_weights": left_translation.tolist(),
            "right_translation_direction_weights": right_translation.tolist(),
            "left_translation_total_weight": float(np.sum(left_translation)),
            "right_translation_total_weight": float(np.sum(right_translation)),
            "left_layer_weights": left_weight, "right_layer_weights": right_weight,
            "left_dominant_layer": int(np.argmax(left_weight)),
            "right_dominant_layer": int(np.argmax(right_weight)),
            "operator_components": components})
    amplification_order = np.argsort(-np.abs(np.diag(amplified)))[:n_soft]
    return {"singular_values_descending": singular.tolist(),
        "smallest_singular_value": float(singular[-1]),
        "largest_singular_value": float(singular[0]),
        "condition_2": _ratio(singular[0], singular[-1]),
        "soft_directions": modes,
        "direction_interpretation": "resolvent singular directions, not normal-mode frequencies or an instability diagnosis",
        "operator_components": {name: _summary(value) for name, value in operator_components.items()},
        "source": _matrix_context(source, n_layers, central, outer),
        "statistical_green": _matrix_context(statistical, n_layers, central, outer),
        "source_components": {name: {"source": _matrix_context(value, n_layers, central, outer),
            "statistical_green": _matrix_context(products[name], n_layers, central, outer),
            "i_source_hermiticity": _hermiticity(1j*value, rtol, atol),
            "i_statistical_green_hermiticity": _hermiticity(1j*products[name], rtol, atol),
            "individual_contribution_may_be_indefinite": True}
            for name, value in source_components.items()},
        "i_source_hermiticity": _hermiticity(h, rtol, atol),
        "i_statistical_green_hermiticity": _hermiticity(1j*statistical, rtol, atol),
        "source_B_hermiticity": _hermiticity(b, rtol, atol),
        "operator_PQ_blocks": _operator_projection(operator, translations),
        "reconstruction": {
            "operator_component_sum": _comparison(sum(operator_components.values()), operator, rtol, atol),
            "source_component_sum": _comparison(sum(source_components.values()), source, rtol, atol),
            "propagated_source_sum": _comparison(sum(products.values()), statistical, rtol, atol),
            "operator_SVD": _comparison((u*singular[None, :])@vh, operator, rtol, atol),
            "source_all_modes": _comparison(u@b@u.conj().T, h, rtol, atol),
            "inverse_SVD_vs_LU": _comparison(inverse_svd, inverse, rtol, atol),
            "iC_all_modes_vs_LU_product": _comparison(reconstructed, 1j*statistical, rtol, atol),
            "diagonal_plus_off_diagonal": _comparison(diagonal_only+off_diagonal, reconstructed, rtol, atol)},
        "inverse_residual": {"left": _comparison(operator@inverse, np.eye(len(operator)), rtol, atol),
                             "right": _comparison(inverse@operator, np.eye(len(operator)), rtol, atol)},
        "all_modes": {"count": len(singular), "source_coupling_B": _summary(b),
            "amplified_source_coupling": _summary(amplified),
            "largest_diagonal_amplifications": [{"svd_descending_index": int(index),
                "singular_value": float(singular[index]),
                "source_B_ii": _complex(b[index, index]),
                "amplified_B_ii_over_s_squared": _complex(amplified[index, index])}
                for index in amplification_order],
            "trace_from_all_amplified_diagonal": _complex(np.trace(amplified)),
            "trace_from_all_mode_iC": _complex(np.trace(reconstructed)),
            "all_mode_iC": _matrix_context(reconstructed, n_layers, central, outer),
            "diagonal_only_iC": _matrix_context(diagonal_only, n_layers, central, outer),
            "off_diagonal_interference_iC": _matrix_context(off_diagonal, n_layers, central, outer),
            "off_diagonal_terms_retained": True, "singular_values_floored": False,
            "modes_truncated": False, "diagonal_only_used_as_result": False},
        "translation_gram": [[_complex(x) for x in row] for row in gram],
        "source_product_norms_are_not_additive_fractions": True}, inverse, statistical


def study_soft_source_diagnostic(
    grid, mesh, dynamical, left, right, sigma_input, translations, *,
    n_layers=9, n_soft=12, include_zero_reference=True, max_bytes=1024**3,
    max_dyson_evaluations=12, max_svd_evaluations=12, deadline_monotonic=None,
    identity_rtol=1e-8, identity_atol=1e-11,
):
    """Evaluate all-q zero plus Gamma±one saved bins for each requested role."""
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(mesh, UniformBZMesh):
        raise TypeError("Supply a SignedFrequencyGrid and UniformBZMesh.")
    if not isinstance(dynamical, np.ndarray) or dynamical.ndim != 3:
        raise ValueError("dynamical must have shape(nq,ndof,ndof).")
    nq, nw, n = mesh.n_qpoints, grid.n_points, dynamical.shape[-1]
    plan = plan_soft_source_dimensions(nq, nw, n, n_layers=n_layers,
        n_soft=n_soft, include_zero_reference=include_zero_reference)
    if plan["peak_owned_bytes"] > _integer(max_bytes, "max_bytes"):
        raise ValueError("Soft-source owned numeric workspace exceeds max_bytes.")
    if plan["dyson_evaluations"] > _integer(max_dyson_evaluations, "max_dyson_evaluations"):
        raise ValueError("Soft-source inverse budget exceeded.")
    if plan["svd_evaluations"] > _integer(max_svd_evaluations, "max_svd_evaluations"):
        raise ValueError("Soft-source SVD budget exceeded.")
    for value, name in ((identity_rtol, "identity_rtol"), (identity_atol, "identity_atol")):
        if isinstance(value, (bool, np.bool_)) or not np.isreal(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative real scalar.")
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite.")

    def admit():
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            raise TimeoutError("Soft-source time admission exhausted.")

    admit()
    raw = _encoded(dynamical, (nq, n, n), "dynamical")
    fields = _spectrum(sigma_input, (nq, nw, n, n), "sigma_input")
    if (not isinstance(translations, np.ndarray) or translations.shape != (n, 3)
            or translations.dtype.kind not in "fc"):
        raise ValueError("translations must be a normalized full-device ndof-by3 array.")
    _finite(translations, "rigid translations")
    gram_context = _comparison(translations.conj().T@translations, np.eye(3), identity_rtol, identity_atol)
    if not gram_context["passed"]:
        raise ValueError("Rigid-translation basis is not normalized.")
    central_layers, central, outer = _regions(n, n_layers)
    contacts = []
    for contact in (left, right):
        if not isinstance(contact, CompactTransverseContact):
            raise TypeError("Contacts must be CompactTransverseContact instances.")
        if (contact.full_ndof != n or contact.mesh.shape != mesh.shape
                or contact.grid.n_points != nw or contact.grid.step != grid.step):
            raise ValueError("Contact dimensions, mesh or grid differ.")
        contacts.append((np.ix_(contact.device_indices, contact.device_indices), contact.spectrum))
    zero_matrix = np.zeros((n, n), complex)
    zero_fields = tuple(np.broadcast_to(zero_matrix, (nq, nw, n, n)) for _ in range(3))
    z = grid.zero_index
    points = [(q, z) for q in range(nq)]+[(0, z-1), (0, z+1)]
    roles = [("harmonic_zero_sigma", zero_fields)] if include_zero_reference else []
    roles.append(("fixed_evaluated_state2", fields))
    results = []
    with threadpool_limits(limits=1, user_api="blas"):
        for role, role_fields in roles:
            records = []
            for q, w in points:
                admit()
                for value in role_fields:
                    _finite(value[q, w], "native fixed-point input")
                _finite(raw[q], "dynamical")
                operator, cs, ds = _assemble(grid.values[w], raw[q], contacts,
                    np.ix_(np.arange(n), np.arange(n)), role_fields, q, w, np.complex128)
                source_components = {}
                operator_components = {"omega_squared_identity": grid.values[w]**2*np.eye(n, dtype=complex),
                    "minus_dynamical": -raw[q].copy()}
                for side, (support, contact) in zip(_SOURCES[:2], contacts, strict=True):
                    source_value, operator_value = np.zeros((n, n), complex), np.zeros((n, n), complex)
                    source_value[support] = .5*(contact.greater[q, w]+contact.lesser[q, w])
                    operator_value[support] = -contact.retarded[q, w]
                    source_components[side] = source_value
                    operator_components["minus_"+side+"_retarded"] = operator_value
                source_components["interaction"] = role_fields[1][q, w].copy()
                operator_components["minus_interaction_retarded"] = -role_fields[0][q, w].copy()
                analysis, inverse, statistical = _analyze_point(operator, source_components, translations,
                    n_layers, n_soft, central, outer, operator_components, identity_rtol, identity_atol, admit)
                spectral = inverse@ds@_adjoint(inverse)
                _finite(spectral, "independent spectral Green product")
                analysis["reconstruction"]["assembled_source_vs_components"] = _comparison(
                    sum(source_components.values()), cs, identity_rtol, identity_atol)
                analysis.update(q_index=q, frequency_index=w, frequency_offset=w-z,
                    frequency=float(grid.values[w]),
                    rigid_translation_overlap_interpretation=("Gamma rigid-translation coordinate alignment"
                        if q == 0 else "geometric basis alignment only; non-Gamma vectors are not global rigid translations"),
                    context=_contexts(operator, cs, ds, inverse, statistical, spectral))
                records.append(analysis)
            results.append({"role": role, "points": records})
    comparisons = []
    if include_zero_reference:
        for zero, fixed in zip(results[0]["points"], results[1]["points"], strict=True):
            comparisons.append({"q_index": zero["q_index"], "frequency_index": zero["frequency_index"],
                "fixed_to_zero_statistical_green_frobenius_ratio": _ratio(
                    fixed["statistical_green"]["full"]["frobenius"], zero["statistical_green"]["full"]["frobenius"]),
                "fixed_to_zero_source_frobenius_ratio": _ratio(fixed["source"]["full"]["frobenius"], zero["source"]["full"]["frobenius"]),
                "fixed_to_zero_smallest_singular_value_ratio": _ratio(fixed["smallest_singular_value"], zero["smallest_singular_value"]),
                "zero_condition_2": zero["condition_2"], "fixed_condition_2": fixed["condition_2"],
                "diagnostic_only": True, "convergence_or_instability_verdict": False})
    return {"schema": "negfpy.soft-source-diagnostic.v1", "diagnostic_only": True,
        "plan": plan, "dyson_evaluations": sum(len(role["points"]) for role in results),
        "svd_evaluations": sum(len(role["points"]) for role in results),
        "central_layers": central_layers, "central_dofs": central.tolist(), "outer_dofs": outer.tolist(),
        "translation_basis_normalization": gram_context,
        "all_modes_and_off_diagonal_source_couplings_retained": True,
        "fields_modified": False, "ifc_projection": False,
        "physical_guards_or_convergence_checks_changed": False,
        "born_evaluated": False, "scba_evaluation": False, "current_acceptance": False,
        "accepted_restart": False, "roles": results, "fixed_vs_zero": comparisons,
        "finite_SVD_LU_differences_are_diagnostic_not_new_physical_failure": True}
