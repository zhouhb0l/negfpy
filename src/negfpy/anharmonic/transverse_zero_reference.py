"""Bounded raw-data reference arithmetic at the Gamma zero-frequency point.

This is an investigation, not a production solver or repair.  It compares two
operators: the encoded candidate and a clearly labelled trial whose interaction
retarded zero slice is reconstructed by the same finite-grid PV formula.  No
identity or symmetry is imposed.  Extended precision is a measured reference,
not an assertion that either physical input or computed inverse is exact.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .grid import SignedFrequencyGrid

_COMPONENTS = ("retarded", "lesser", "greater")
_LD = np.clongdouble


def _integer(value, name, minimum=1, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its bounded range.")
    return int(value)


def zero_reference_workspace_bytes(full_ndof, active_ndof, n_frequency):
    """Conservative owned numeric peak; caller fields and library scratch excluded.

    Frequency stacks remain caller-owned views; PV is streamed one signed pair
    at a time.  160 extended full matrices cover retained cases, products and
    summary temporaries, with 32 double and 40 active extended matrices reserved.
    """
    d = _integer(full_ndof, "full_ndof", maximum=512)
    a = _integer(active_ndof, "active_ndof", maximum=d)
    nw = _integer(n_frequency, "n_frequency", minimum=3, maximum=4097)
    return int((160*np.dtype(_LD).itemsize+32*16)*d*d
               +40*np.dtype(_LD).itemsize*a*a+4*nw*16)


def _get(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def _adjoint(value):
    return value.conj().T


def _maximum(value):
    return np.max(np.abs(value), initial=0)


def _norm_inf(value):
    return np.max(np.sum(np.abs(value), axis=1), initial=0)


def _ratio(numerator, denominator):
    return float(numerator/denominator) if denominator else (0.0 if not numerator else None)


def _summary(value):
    absolute = np.abs(value)
    entry = np.unravel_index(int(np.argmax(absolute)), value.shape)
    return {"max_abs": float(absolute[entry]), "max_abs_entry": [int(x) for x in entry],
            "value_at_max_abs": {"real": float(value[entry].real),
                                 "imag": float(value[entry].imag)},
            "max_abs_real": float(_maximum(value.real)),
            "max_abs_imag": float(_maximum(value.imag)),
            "norm_inf": float(_norm_inf(value)),
            "antihermiticity_defect": float(_maximum(value+_adjoint(value))),
            "reality_defect": float(_maximum(value-value.conj()))}


def _difference(first, second):
    error = _maximum(first-second)
    scale = max(_maximum(first), _maximum(second))
    return {"error": float(error), "scale": float(scale),
            "relative_error": _ratio(error, scale)}


def _check(first, second):
    result = _difference(first, second)
    allowance = 1e-11+1e-8*result["scale"]
    result.update(pointwise_allowance=allowance,
                  pointwise_allowance_ratio=result["error"]/allowance,
                  acceptance=False)
    return result


def _spectrum(value, nw, na, name):
    fields = tuple(np.asarray(_get(value, key)) for key in _COMPONENTS)
    if any(field.shape != (nw, na, na) for field in fields):
        raise ValueError(f"{name} must have three (n_frequency, n_active, n_active) fields.")
    if any(field.dtype != np.dtype(np.complex128) for field in fields):
        raise TypeError(f"{name} must contain encoded complex128 fields.")
    return fields


def _indices(values, dimension, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or raw.size < 1:
        raise ValueError(f"{name} must be a nonempty integer vector.")
    if np.any(raw < 0) or np.any(raw >= dimension) or len(np.unique(raw)) != len(raw):
        raise ValueError(f"{name} must contain unique full-device indices.")
    return np.ix_(raw, raw)


def _point(value, dimension):
    try:
        support = _get(value, "support_dofs")
    except (KeyError, AttributeError):
        support = _get(value, "device_indices")
    indices = _indices(support, dimension, "contact support")
    fields = tuple(np.asarray(_get(value, key)) for key in _COMPONENTS)
    shape = (len(support), len(support))
    if any(field.shape != shape or field.dtype != np.dtype(np.complex128) for field in fields):
        raise ValueError("Contact point fields must be square encoded complex128 matrices.")
    if any(not np.isfinite(field).all() for field in fields):
        raise ValueError("Contact point fields must be finite.")
    return indices, fields


def _pv_zero(fields, grid):
    """Independent streamed direct PV at zero, preserving every raw component.

    Use the encoded production grid weight, extended arithmetic and compensated
    signed-pair summation.  No frequency wrapping or same-point term is used.
    In particular the raw zero spectral term may be real and antisymmetric.
    """
    _, lesser, greater = fields
    z = grid.zero_index
    delta_zero = greater[z].astype(_LD)-lesser[z].astype(_LD)
    total = np.zeros_like(delta_zero)
    compensation = np.zeros_like(delta_zero)
    weight = np.longdouble(grid.integration_weight)
    for offset in range(1, z+1):
        negative, positive = z-offset, z+offset
        if not all(np.isfinite(field[index]).all()
                   for field in (lesser, greater) for index in (negative, positive)):
            raise ValueError("PV raw lesser/greater inputs contain nonfinite values.")
        dm = greater[negative].astype(_LD)-lesser[negative].astype(_LD)
        dp = greater[positive].astype(_LD)-lesser[positive].astype(_LD)
        term = weight*(dm/(-np.longdouble(grid.values[negative]))
                       +dp/(-np.longdouble(grid.values[positive])))
        corrected = term-compensation
        updated = total+corrected
        compensation = (updated-total)-corrected
        total = updated
    return np.longdouble("0.5")*delta_zero+np.clongdouble(1j)*total


def _refine_inverse(operator, inverse, max_refinements):
    """Fixed bounded inverse refinement with the existing double inverse.

    Correction is B*(I-A*X), using the original inverse B as a preconditioner.
    Every product and residual is extended precision; no unsupported NumPy
    long-double LAPACK path or additional inverse solve is requested.
    """
    base = inverse.astype(_LD)
    current = base.copy()
    identity = np.eye(len(operator), dtype=_LD)
    history = []
    for iteration in range(max_refinements+1):
        left = identity-operator@current
        right = identity-current@operator
        condition = _norm_inf(operator)*_norm_inf(current)
        denominator = condition+1
        record = {"iteration": iteration,
                  "left_absolute_residual": float(_norm_inf(left)),
                  "right_absolute_residual": float(_norm_inf(right)),
                  "left_normalized_backward_residual": float(_norm_inf(left)/denominator),
                  "right_normalized_backward_residual": float(_norm_inf(right)/denominator),
                  "condition_estimate": float(condition)}
        history.append(record)
        if iteration != max_refinements:
            current = current+base@left
    last = history[-1]
    # A measurable quality screen, not a rigorous accuracy bound or solver gate.
    threshold = 64*len(operator)*np.finfo(np.longdouble).eps
    adequate = max(last["left_normalized_backward_residual"],
                   last["right_normalized_backward_residual"]) <= threshold
    return current, {"method": "X <- X + B*(I-A*X), fixed encoded double inverse B",
                     "iterations": max_refinements, "history": history,
                     "measured_backward_error_screen": float(threshold),
                     "measured_backward_error_adequate": bool(adequate),
                     "certified_forward_accuracy": False}


def _assemble(dynamical, contacts, active, sigma, dtype):
    operator = np.zeros(dynamical.shape, dtype=dtype)-dynamical.astype(dtype)
    sources = [np.zeros(operator.shape, dtype=dtype) for _ in range(2)]
    for support, fields in (*contacts, (active, sigma)):
        operator[support] -= fields[0].astype(dtype)
        sources[0][support] += fields[1].astype(dtype)
        sources[1][support] += fields[2].astype(dtype)
    return operator, *sources


def _component_delta(dimension, contacts, active, sigma, dtype):
    delta = np.zeros((dimension, dimension), dtype=dtype)
    for support, fields in (*contacts, (active, sigma)):
        delta[support] += fields[2].astype(dtype)-fields[1].astype(dtype)
    return delta


def _green_summary(retarded, lesser, greater, stable_delta, statistical, active):
    spectral = retarded-_adjoint(retarded)
    observed = greater-lesser
    result = {"retarded": _summary(retarded), "lesser": _summary(lesser),
              "greater": _summary(greater), "separate_product_difference": _summary(observed),
              "stable_spectral_product": _summary(stable_delta),
              "statistical_product": _summary(statistical),
              "separate_keldysh": _check(observed, spectral),
              "stable_keldysh": _check(stable_delta, spectral),
              "separate_minus_stable": _difference(observed, stable_delta),
              "separate_center_minus_statistical": _difference((greater+lesser)/2, statistical),
              "cancellation_magnitude_ratio": _ratio(_maximum(lesser)+_maximum(greater),
                                                       _maximum(observed)),
              "active_retarded_reality": _difference(retarded[active], retarded[active].conj()),
              "active_separate_keldysh": _check(observed[active], spectral[active]),
              "active_stable_keldysh": _check(stable_delta[active], spectral[active])}
    return result


def _source_parts(dynamical, contacts, active, sigma, retarded):
    n = len(dynamical)
    harmonic = dynamical.astype(_LD)
    parts = {"harmonic": _adjoint(harmonic)-harmonic}
    components = {}
    for name, (support, fields) in zip(("left", "right", "interaction"),
                                      (*contacts, (active, sigma)), strict=True):
        sr, sl, sg = (value.astype(_LD) for value in fields)
        delta = sg-sl
        defect = delta-(sr-_adjoint(sr))
        embedded = np.zeros((n, n), dtype=_LD)
        embedded[support] = defect
        parts[name] = embedded
        components[name] = {"retarded": _summary(sr), "lesser": _summary(sl),
                            "greater": _summary(sg), "raw_delta": _summary(delta),
                            "raw_source_identity": _summary(defect)}
    total = np.zeros((n, n), dtype=_LD)
    summaries = {}
    for name, defect in parts.items():
        total += defect
        summaries[name] = {"raw_defect": _summary(defect),
                           "propagated_defect": _summary(retarded@defect@_adjoint(retarded))}
    return {"components": components, "defect_contributions": summaries,
            "summed_raw_defect": _summary(total),
            "summed_propagated_defect": _summary(retarded@total@_adjoint(retarded))}, total


def _case(dynamical, contacts, active, sigma, max_refinements):
    operator, sl, sg = _assemble(dynamical, contacts, active, sigma, np.complex128)
    gr = np.linalg.inv(operator)
    ga = _adjoint(gr)
    gl, gg = gr@sl@ga, gr@sg@ga
    delta, center = sg-sl, (sg+sl)/2
    stable, statistical = gr@delta@ga, gr@center@ga
    ordinary = _green_summary(gr, gl, gg, stable, statistical, active)
    ordinary["raw_source_identity"] = _summary(delta-(_adjoint(operator)-operator))
    extended_a, extended_sl, extended_sg = _assemble(dynamical, contacts, active, sigma, _LD)
    reference, quality = _refine_inverse(extended_a, gr, max_refinements)
    advanced = _adjoint(reference)
    ref_lesser, ref_greater = reference@extended_sl@advanced, reference@extended_sg@advanced
    extended_delta = extended_sg-extended_sl
    carried_delta = _component_delta(len(dynamical), contacts, active, sigma, _LD)
    extended_center = (extended_sg+extended_sl)/2
    ref_stable = reference@extended_delta@advanced
    carried_stable = reference@carried_delta@advanced
    ref_center = reference@extended_center@advanced
    extended = _green_summary(reference, ref_lesser, ref_greater, ref_stable, ref_center, active)
    source_defect = extended_delta-(_adjoint(extended_a)-extended_a)
    propagated = reference@source_defect@advanced
    observed = ref_greater-ref_lesser-(reference-advanced)
    extended.update(raw_source_identity=_summary(source_defect),
                    component_carried_delta=_summary(carried_delta),
                    aggregate_minus_component_delta=_summary(extended_delta-carried_delta),
                    component_carried_stable_spectral=_summary(carried_stable),
                    component_carried_keldysh=_check(carried_stable, reference-advanced),
                    propagated_source_defect=_summary(propagated),
                    separate_observed_minus_propagated=_summary(observed-propagated),
                    stable_observed_minus_propagated=_summary(
                        ref_stable-(reference-advanced)-propagated))
    parts, source_sum = _source_parts(dynamical, contacts, active, sigma, reference)
    parts["assembled_minus_component_defect"] = _summary(source_defect-source_sum)
    return {"ordinary_complex128": ordinary, "extended_reference": extended,
            "inverse_reference_quality": quality, "raw_sources": parts,
            "ordinary_versus_extended": {
                "operator_assembly": _difference(operator.astype(_LD), extended_a),
                "lesser_source_assembly": _difference(sl.astype(_LD), extended_sl),
                "greater_source_assembly": _difference(sg.astype(_LD), extended_sg),
                "retarded": _difference(gr.astype(_LD), reference),
                "lesser": _difference(gl.astype(_LD), ref_lesser),
                "greater": _difference(gg.astype(_LD), ref_greater),
                "separate_difference": _difference((gg-gl).astype(_LD), ref_stable),
                "stable_difference": _difference(stable.astype(_LD), ref_stable),
                "statistical_product": _difference(statistical.astype(_LD), ref_center)},
            "accepted_state": False}


def _safe_json(value, paths, path=""):
    if isinstance(value, dict):
        return {key: _safe_json(item, paths, f"{path}.{key}".strip("."))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json(item, paths, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, float) and not np.isfinite(value):
        paths.append(path)
        return None
    return value


def study_zero_frequency(grid, dynamical_q, active_dofs, left_point, right_point,
                         sigma_spectrum_q, *, sigma_input_spectrum_q=None,
                         mapped_sigma_spectrum_q=None, mixing=.1, max_bytes,
                         max_refinements=4):
    """Compare raw encoded candidate and one direct-PV diagnostic trial.

    Spectra expose ``retarded/lesser/greater`` (mapping or attributes), each a
    complex128 ``(frequency, active, active)`` view. Contact points additionally
    expose ``support_dofs`` (or ``device_indices``) and three compact matrices.
    All outputs are scalar JSON records. Inputs are never modified; no Born
    evaluation, field publication, acceptance, symmetry projection or tolerance
    change occurs. The caller supplies the Gamma block and authenticates data.
    """
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    raw = np.asarray(dynamical_q)
    indices = np.asarray(active_dofs)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or raw.dtype != np.dtype(np.complex128):
        raise ValueError("dynamical_q must be an encoded complex128 square matrix.")
    if indices.ndim != 1:
        raise ValueError("active_dofs must be a vector.")
    planned = zero_reference_workspace_bytes(len(raw), len(indices), grid.n_points)
    if planned > _integer(max_bytes, "max_bytes"):
        raise ValueError("Zero-frequency reference exceeds max_bytes before numerical work.")
    iterations = _integer(max_refinements, "max_refinements", minimum=0, maximum=4)
    if np.finfo(np.longdouble).eps >= np.finfo(np.float64).eps:
        raise RuntimeError("This platform has no additional long-double precision.")
    alpha = float(mixing)
    if not np.isfinite(alpha) or not 0 < alpha <= 1 or isinstance(mixing, (bool, np.bool_)):
        raise ValueError("mixing must be finite and in (0, 1].")
    active = _indices(indices, len(raw), "active_dofs")
    contacts = tuple(_point(value, len(raw)) for value in (left_point, right_point))
    candidate = _spectrum(sigma_spectrum_q, grid.n_points, len(indices), "candidate")
    z = grid.zero_index
    if not np.isfinite(raw).all() or any(not np.isfinite(field[z]).all() for field in candidate):
        raise ValueError("Zero-point raw fields must be finite.")
    pv_candidate = _pv_zero(candidate, grid)
    pv = {"formula": "0.5*raw_delta(0) + i*sum[k != zero](weight*raw_delta(k)/(-omega(k)))",
          "weights": "encoded production rectangle weight, promoted to longdouble",
          "summation": "compensated signed-frequency pairs; finite endpoints; no wrap",
          "candidate_direct_pv": _summary(pv_candidate),
          "stored_candidate_retarded": _summary(candidate[0][z].astype(_LD)),
          "candidate_stored_minus_direct_pv": _difference(candidate[0][z].astype(_LD), pv_candidate),
          "raw_zero_spectral": _summary(candidate[2][z].astype(_LD)-candidate[1][z].astype(_LD))}
    if (sigma_input_spectrum_q is None) != (mapped_sigma_spectrum_q is None):
        raise ValueError("Provide both saved input and mapped spectra, or neither.")
    if sigma_input_spectrum_q is not None:
        saved = _spectrum(sigma_input_spectrum_q, grid.n_points, len(indices), "saved input")
        mapped = _spectrum(mapped_sigma_spectrum_q, grid.n_points, len(indices), "mapped")
        for fields in (saved, mapped):
            if any(not np.isfinite(field[z]).all() for field in fields):
                raise ValueError("Saved zero slices must be finite.")
        pv_saved, pv_mapped = _pv_zero(saved, grid), _pv_zero(mapped, grid)
        a = np.longdouble(alpha)
        b = np.longdouble(1-alpha)
        pv["mixing_decomposition"] = {
            "mixing": alpha, "encoded_weights_sum_minus_one": float(a+b-1),
            "saved_stored_minus_direct_pv": _difference(saved[0][z].astype(_LD), pv_saved),
            "mapped_stored_minus_direct_pv": _difference(mapped[0][z].astype(_LD), pv_mapped),
            "candidate_pv_minus_weighted_saved_pv": _difference(pv_candidate, b*pv_saved+a*pv_mapped),
            "encoded_zero_mix_rounding": {
                name: _difference(candidate[index][z].astype(_LD),
                                  b*saved[index][z].astype(_LD)+a*mapped[index][z].astype(_LD))
                for index, name in enumerate(_COMPONENTS)},
            "saved_zero_raw_source_identity": _summary(
                saved[2][z].astype(_LD)-saved[1][z].astype(_LD)
                -(saved[0][z].astype(_LD)-_adjoint(saved[0][z].astype(_LD)))),
            "mapped_zero_raw_source_identity": _summary(
                mapped[2][z].astype(_LD)-mapped[1][z].astype(_LD)
                -(mapped[0][z].astype(_LD)-_adjoint(mapped[0][z].astype(_LD))))}
    original_fields = tuple(field[z] for field in candidate)
    original = _case(raw, contacts, active, original_fields, iterations)
    # The trial keeps the direct PV in extended precision for its reference;
    # its ordinary path rounds this changed retarded slice to complex128.
    trial = _case(raw, contacts, active,
                  (pv_candidate, original_fields[1], original_fields[2]), iterations)
    result = {"schema": "negfpy.transverse-zero-frequency-reference.v1",
              "scope": "Selected supplied Gamma block, zero frequency; no state acceptance",
              "frequency_index": int(z), "omega_solver": 0.0,
              "precision": {"float64_epsilon": float(np.finfo(np.float64).eps),
                            "longdouble_epsilon": float(np.finfo(np.longdouble).eps),
                            "complex_longdouble_bytes": np.dtype(_LD).itemsize,
                            "input_precision": "encoded complex128; prior lost bits cannot be recovered"},
              "resources": {"owned_numeric_workspace_bound_bytes": planned,
                            "caller_and_library_scratch_excluded": True,
                            "zero_dyson_cases": 2, "max_zero_dyson_cases": 2,
                            "inverse_refinements_per_case": iterations,
                            "frequency_stack_allocated": False},
              "principal_value": pv,
              "original_candidate": original,
              "direct_pv_retarded_trial": {
                  "changed_candidate": True, "only_changed_input": "interaction retarded zero slice",
                  "accepted": False, **trial},
              "reference_interpretation": "Use measured residuals and cross-comparisons; no certified gold standard",
              "identity_forced": False, "symmetry_projected": False,
              "born_evaluated": False, "scba_state_completed": False,
              "field_outputs": False, "accepted_restart": False}
    paths = []
    result = _safe_json(result, paths)
    result["nonfinite_fields"] = paths
    return result


__all__ = ["study_zero_frequency", "zero_reference_workspace_bytes"]
