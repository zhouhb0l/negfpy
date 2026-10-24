"""Fixed-input lead-eta comparisons and a read-only cropped-vertex ASR audit.

The eta study reuses the bounded native-channel diagnostic.  A saved mapped
self-energy belongs to its original eta and is only a comparison reference:
subtracting a new-eta partial sum from it is never a far-frequency remainder.
The vertex audit contracts actual global rigid translations, not every mode or
entry at transverse Gamma.  It neither projects nor changes the supplied IFCs.
"""

from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from .transverse_compact_contacts import CompactTransverseContact
from .transverse_native_zero_diagnostic import (
    _admit,
    _integer,
    study_native_zero_diagnostic,
)
from .transverse_packed import PackedTransverseCubicVertex


def _positive_eta(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isscalar(value)
            or np.iscomplexobj(value) or not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive real scalar.")
    return float(value)


def study_eta_zero_diagnostic(
    grid, vertex, dynamical, left, right, sigma_input, mapped_sigma, *,
    eta_lead, baseline_eta=.0015625, **kwargs,
):
    """Evaluate unchanged interaction inputs with eta-tagged fresh contacts.

    Contact construction and provenance are the caller's responsibility.  Both
    contact objects must carry the requested eta tag; this rejects accidentally
    supplying the old contacts with a new label.  The independent R/C/D inputs,
    frequency weights, native Green products and partial contraction kernel are
    unchanged.  No full current, full Born map or SCBA convergence is evaluated.
    """
    eta = _positive_eta(eta_lead, "eta_lead")
    baseline = _positive_eta(baseline_eta, "baseline_eta")
    for contact in (left, right):
        if not isinstance(contact, CompactTransverseContact):
            raise TypeError("Contacts must be CompactTransverseContact instances.")
        if contact.contact.eta_lead != eta:
            raise ValueError("Contact eta tag does not equal the requested eta_lead.")
    result = study_native_zero_diagnostic(
        grid, vertex, dynamical, left, right, sigma_input, mapped_sigma, **kwargs)
    result.update(
        schema="negfpy.eta-zero-diagnostic.v1", eta_lead=eta,
        baseline_eta_lead=baseline, eta_device=0.,
        physical_inputs_unchanged=False, fixed_interaction_inputs_unchanged=True,
        fresh_contact_eta_tags_verified=True, physical_guards_unchanged=True,
    )
    born = result["born"]
    born["baseline_saved_mapped"] = born.pop("saved_mapped")
    comparison = born.pop("inferred_remainder")
    comparison.update(
        status="baseline saved mapped total minus current-eta partial low-frequency sum",
        baseline_eta_lead=baseline, current_eta_lead=eta,
        same_eta_as_baseline=eta == baseline, far_frequency_remainder=False,
    )
    born["cross_eta_reference_gap"] = comparison
    born["full_current_eta_total_evaluated"] = False
    born["baseline_saved_mapped_used_for_green_or_born"] = False
    return result


def _frobenius(value):
    # A matrix-sized temporary only.  Each slice is checked before arithmetic.
    magnitude = np.abs(value)
    scale = np.max(magnitude, initial=0.)
    if not scale:
        return 0.
    norm = float(scale*np.sqrt(np.sum((magnitude/scale)**2)))
    if not np.isfinite(norm):
        raise ArithmeticError("Vertex-audit scalar norm overflowed.")
    return norm


def _block_frobenius(block):
    # Stream matrix slices instead of materializing abs(V) for a cubic block.
    norms = np.array([_frobenius(matrix) for matrix in block])
    return _frobenius(norms)


def _contraction_summary(matrix, scale):
    magnitude = np.abs(matrix)
    if not np.isfinite(magnitude).all():
        raise ArithmeticError("Vertex-audit contraction produced nonfinite values.")
    if matrix.size:
        location = tuple(int(x) for x in np.unravel_index(np.argmax(magnitude), matrix.shape))
        peak, value = float(magnitude[location]), matrix[location]
        encoded = {"real": float(value.real), "imag": float(value.imag)}
    else:
        location, peak, encoded = None, 0., None
    norm = _frobenius(matrix)
    return {
        "max_abs": peak, "frobenius": norm,
        "max_abs_location_in_remaining_active_coordinates": location,
        "value_at_max_abs": encoded,
        "cauchy_schwarz_frobenius_scale": scale,
        "relative_frobenius_to_scale": norm/scale if scale else 0.,
        "relative_max_to_scale": peak/scale if scale else 0.,
    }


def audit_cropped_vertex_acoustic(
    vertex, translations, *, max_bytes=2*1024**3, max_contraction_work=100_000_000,
    deadline_monotonic=None, time_reserve_seconds=0,
):
    """Contract all three zero-momentum legs of an actual packed cubic vertex.

    ``translations`` has shape ``(full_ndof, 3)`` and contains orthonormal,
    mass-weighted global rigid translations authenticated by the caller.  Its
    active subset is NOT renormalized.  V[q,p,a,c,d] carries momenta (-q,p,q-p):
    audit leg a at q=0, leg c at p=0, and leg d at p=q, for every mesh point.
    This is a diagnostic of the cropped vertex, with no acceptance threshold,
    physical repair, projection, field mutation or interpretation of all Gamma
    modes as translations.  Caller arrays, Python and library scratch are not
    included in the owned-payload bound; no cubic temporary is allocated.
    """
    if not isinstance(vertex, PackedTransverseCubicVertex):
        raise TypeError("vertex must be a PackedTransverseCubicVertex.")
    if (not isinstance(translations, np.ndarray)
            or translations.shape != (vertex.full_ndof, 3)
            or translations.dtype.kind not in "iuf" or np.iscomplexobj(translations)):
        raise ValueError("translations must be a real full_ndof-by-3 ndarray.")
    if not np.isfinite(translations).all():
        raise ValueError("Rigid translations must be finite.")
    full_gram = translations.T@translations
    if not np.allclose(full_gram, np.eye(3), rtol=0., atol=1e-12):
        raise ValueError("Full-device rigid translations must be orthonormal.")
    if (not np.isfinite(time_reserve_seconds) or time_reserve_seconds < 0
            or (deadline_monotonic is not None and not np.isfinite(deadline_monotonic))):
        raise ValueError("Time admission values must be finite and reserve nonnegative.")
    a, nq = vertex.ndof, vertex.mesh.n_qpoints
    planned_work = 9*nq*a**3
    # Crop (3*a*8), contraction and abs/norm workspaces (<=64*a*a), plus
    # per-slice norms and tiny Gram matrices.  No whole-coefficient copy.
    owned = 64*a*a + 32*a + 1024
    if owned > _integer(max_bytes, "max_bytes"):
        raise ValueError("Cropped acoustic audit owned payload exceeds max_bytes.")
    if planned_work > _integer(max_contraction_work, "max_contraction_work"):
        raise ValueError("Cropped acoustic audit exceeds max_contraction_work.")
    _admit(deadline_monotonic, time_reserve_seconds)
    active = translations[vertex.active_dofs]
    active_gram = active.T@active
    records = []
    legs = (("external_a", 0, "a,acd->cd"),
            ("internal_c", 1, "c,acd->ad"),
            ("internal_d", 2, "d,acd->ac"))
    with threadpool_limits(limits=1, user_api="blas"):
        for label, axis, contraction in legs:
            for other_q in range(nq):
                _admit(deadline_monotonic, time_reserve_seconds)
                q, p = ((0, other_q) if axis == 0 else
                        (other_q, 0) if axis == 1 else (other_q, other_q))
                block = vertex.coefficients[q, p]
                for matrix in block:
                    if not np.isfinite(matrix).all():
                        raise ValueError("Vertex coefficients contain nonfinite values.")
                block_norm = _block_frobenius(block)
                for direction in range(3):
                    _admit(deadline_monotonic, time_reserve_seconds)
                    vector = active[:, direction]
                    contracted = np.einsum(contraction, vector, block, optimize=False)
                    translation_norm = _frobenius(vector)
                    scale = block_norm*translation_norm
                    if not np.isfinite(scale):
                        raise ArithmeticError("Vertex-audit normalization overflowed.")
                    records.append({
                        "zero_momentum_leg": label, "zero_momentum_tensor_axis": axis,
                        "q_output": q, "q_internal": p,
                        "translation_direction": direction,
                        "vertex_block_frobenius": block_norm,
                        "active_translation_norm": translation_norm,
                        **_contraction_summary(contracted, scale),
                    })
    return {
        "schema": "negfpy.cropped-vertex-acoustic-audit.v1",
        "diagnostic_only": True, "acceptance": False, "modifies_vertex": False,
        "translation_provenance": "caller-supplied mass-weighted global rigid translations",
        "translation_normalization": "orthonormal on full device; active subset not renormalized",
        "momentum_convention": "V[q,p,a,c,d] has leg momenta (-q,p,q-p)",
        "all_gamma_modes_are_translations": False,
        "full_translation_gram": full_gram.tolist(),
        "active_translation_gram": active_gram.tolist(),
        "full_ndof": vertex.full_ndof, "active_ndof": a,
        "active_dof_order": vertex.active_dofs.tolist(),
        "n_q": nq, "contraction_count": len(records),
        "planned_multiply_accumulate_work": planned_work,
        "peak_owned_bytes": owned, "caller_payload_excluded": True,
        "rss_guarantee": False, "records": records,
    }


__all__ = ["audit_cropped_vertex_acoustic", "study_eta_zero_diagnostic"]
