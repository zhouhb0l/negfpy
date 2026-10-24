"""Finite-crop acoustic leakage versus a relative-displacement bond oracle."""

from __future__ import annotations

import copy
import time

import numpy as np
import pytest
from test_anharmonic_transverse_material_reference import _CUBIC, _MASSES
from test_anharmonic_transverse_material_streaming import _inputs

from negfpy.anharmonic._canonical import domain_digest
from negfpy.anharmonic.mapped_fc3 import _normalized_payload
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_boundary_cancellation import (
    audit_transverse_boundary_cancellation,
)
from negfpy.anharmonic.transverse_material_packed import fourier_material_cubic_packed
from negfpy.anharmonic.types import SolverScale

_SCALE = SolverScale(omega0=2.0, hbar=3.0)


def _run(geometry, source, mesh, *, reference=True, **budgets):
    vertex = None
    if reference:
        vertex = fourier_material_cubic_packed(
            mesh, source, geometry, n_device_layers=3,
            interaction_layers=(0, 1), scale=_SCALE,
        ).vertex
    return audit_transverse_boundary_cancellation(
        mesh, source, geometry, n_device_layers=3,
        interaction_layers=(0, 1), scale=_SCALE,
        reference_vertex=vertex, **budgets,
    )


@pytest.mark.parametrize("r0,duplicate", [(False, False), (True, True)])
def test_relative_bond_full_support_cancels_boundary_residual_on_every_leg(r0, duplicate):
    geometry, source = _inputs(np.eye(3, dtype=int), r0=r0, duplicate_images=duplicate)
    result = _run(geometry, source, UniformBZMesh((4, 1, 1)))
    assert result["record_count"] == 36
    assert result["retained_ndof"] == 4
    np.testing.assert_allclose(
        result["translation_normalization"]["restricted_translation_gram"],
        np.diag([2 / 3, 0, 0]), atol=1e-15,
    )
    # Independent bond argument: (u_A-u_B)^3 has zero derivative along a
    # uniform displacement, even for unequal masses.  With both retained legs
    # at boundary B0 or A1, the cropped third leg leaves one diagonal entry at
    # each end.  Their magnitudes are K/(m_B sqrt(M)) and K/(m_A sqrt(M));
    # this result does not depend on transverse momentum or contracted leg.
    solver_factor = np.sqrt(_SCALE.hbar) / _SCALE.omega0**2.5
    total_mass = 3 * np.sum(_MASSES)
    expected_fro = _CUBIC * solver_factor / np.sqrt(total_mass) * np.sqrt(
        1 / _MASSES[0]**2 + 1 / _MASSES[1]**2
    )
    for record in result["records"]:
        assert record["closure"]["passed"]
        assert record["reference"]["comparison"]["passed"]
        assert record["complete"]["frobenius"] < 2e-16
        if record["direction"] == 0:
            assert record["cropped"]["frobenius"] == pytest.approx(expected_fro, rel=2e-13)
            assert record["omitted"]["frobenius"] == pytest.approx(expected_fro, rel=2e-13)
            assert record["complete_to_crop_frobenius_ratio"] < 1e-13
        else:
            assert record["cropped"]["frobenius"] == 0
    # A source atom at longitudinal cell -1 contributes: complete support is
    # neither a larger finite device nor a wrapped image of its last layer.
    assert all(x > 0 for x in result["work"]["outside_finite_device_anchor_counts_by_leg"])
    assert result["source_weighting"]["longitudinal_wrapping"] is False
    assert result["source_weighting"]["contracted_leg_clamped_to_device"] is False


def test_complex_phases_cartesian_order_and_sheared_source_cosets_match_saved_vertex():
    transform = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    geometry, source = _inputs(
        transform, r0=True, duplicate_images=True, mixed_cartesian=True,
    )
    mesh = UniformBZMesh((4, 1, 1))
    saved = fourier_material_cubic_packed(
        mesh, source, geometry, n_device_layers=3,
        interaction_layers=(0, 1), scale=_SCALE,
    ).vertex
    assert np.max(np.abs(saved.coefficients.imag)) > 1e-3
    result = audit_transverse_boundary_cancellation(
        mesh, source, geometry, n_device_layers=3, interaction_layers=(0, 1),
        scale=_SCALE, reference_vertex=saved,
    )
    assert all(record["reference"]["comparison"]["passed"] for record in result["records"])
    assert all(record["complete"]["frobenius"] < 1e-15 for record in result["records"])
    assert all(record["cropped"]["frobenius"] > 0 for record in result["records"])
    # Every zero-momentum leg uses its own V[q,p] route, not a single Gamma
    # block reused for all three contractions.
    assert [(r["vertex_q_index"], r["vertex_p_index"]) for r in result["records"] if r["direction"] == 0] == [
        *[(0, q) for q in range(4)],
        *[(q, 0) for q in range(4)],
        *[(q, q) for q in range(4)],
    ]


def test_complete_source_nonzero_is_reported_instead_of_repaired_or_rejected():
    geometry, original = _inputs(np.eye(3, dtype=int))
    source = copy.copy(original)
    changed_terms = []
    # Add an on-site AAA term: deliberately violates the physical bond ASR,
    # while retaining the builder's mass/source metadata for this tiny test.
    for term in original.terms:
        changed = copy.copy(term)
        values = term.values.copy()
        if tuple(term.primitive_indices) == (0, 0, 0):
            values[0, 0, 0] += 0.02
        object.__setattr__(changed, "values", values)
        changed_terms.append(changed)
    object.__setattr__(source, "terms", tuple(changed_terms))
    object.__setattr__(source, "content_sha256", domain_digest(
        "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3", _normalized_payload(source)
    ))
    result = _run(geometry, source, UniformBZMesh((4, 1, 1)), reference=False)
    assert any(r["complete"]["frobenius"] > 1e-5 for r in result["records"])
    assert result["comparison_summary"]["complete_source_zero_is_not_an_execution_gate"]
    assert result["comparison_summary"]["saved_vertex_comparison_available"] is False


def test_mutated_identity_is_rejected():
    geometry, source = _inputs(np.eye(3, dtype=int))
    changed = copy.copy(source)
    object.__setattr__(changed, "content_sha256", "0" * 64)
    with pytest.raises(ValueError, match="content no longer matches"):
        _run(geometry, changed, UniformBZMesh((4, 1, 1)), reference=False)


@pytest.mark.parametrize("budget,match", [
    ({"max_workspace_bytes": 1}, "workspace"),
    ({"max_geometry_evaluations": 1}, "geometry"),
    ({"max_accumulation_work": 1}, "work plan"),
])
def test_work_and_memory_admission_reject_before_contraction_allocation(monkeypatch, budget, match):
    import negfpy.anharmonic.transverse_boundary_cancellation as module

    geometry, source = _inputs(np.eye(3, dtype=int), mixed_cartesian=True)
    original_zeros = module.np.zeros

    def guarded(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 6:
            pytest.fail("Contraction matrices allocated before failed admission.")
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(module.np, "zeros", guarded)
    with pytest.raises(ValueError, match=match):
        _run(geometry, source, UniformBZMesh((4, 1, 1)), reference=False, **budget)


def test_expired_deadline_stops_before_source_or_field_work():
    geometry, source = _inputs(np.eye(3, dtype=int))
    with pytest.raises(TimeoutError, match="deadline"):
        _run(geometry, source, UniformBZMesh((4, 1, 1)), reference=False,
             deadline_monotonic=time.monotonic() - 1)
