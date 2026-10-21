"""New direct-packed image mapping checks; no historical workflow is replayed."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from test_anharmonic_transverse_material_reference import _real_space_polynomial
from test_anharmonic_transverse_material_streaming import _inputs
from test_anharmonic_transverse_q_reference import _fourier_vertex, _unitary_basis

from negfpy.anharmonic._canonical import domain_digest
from negfpy.anharmonic.mapped_fc3 import _normalized_payload
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_material import fourier_material_cubic
from negfpy.anharmonic.transverse_material_packed import fourier_material_cubic_packed
from negfpy.anharmonic.types import SolverScale

_SCALE = SolverScale(omega0=2.0, hbar=3.0)


def _run(geometry, source, shape=(4, 1, 1), **kwargs):
    return fourier_material_cubic_packed(
        UniformBZMesh(shape), source, geometry, n_device_layers=3,
        interaction_layers=(0, 1), scale=kwargs.pop("scale", _SCALE), **kwargs,
    )


def _rehash(source, replacements):
    """Build an authenticated synthetic variant, preserving original objects."""
    result = copy.copy(source)
    terms = []
    for original in source.terms:
        term = copy.copy(original)
        for key, value in replacements(original).items():
            object.__setattr__(term, key, value)
        terms.append(term)
    object.__setattr__(result, "terms", tuple(terms))
    object.__setattr__(result, "content_sha256", domain_digest(
        "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3", _normalized_payload(result)
    ))
    return result


@pytest.mark.parametrize("shape,transform,duplicates,r0", [
    ((1, 1, 1), np.eye(3, dtype=int), False, False),
    ((4, 1, 1), np.eye(3, dtype=int), False, False),
    ((4, 1, 1), np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]]), True, True),
    ((2, 2, 1), np.eye(3, dtype=int), True, True),
])
def test_packed_images_match_independent_bond_polynomial(shape, transform, duplicates, r0):
    geometry, source = _inputs(transform, duplicate_images=duplicates, r0=r0)
    before = [term.values.copy() for term in source.terms]
    actual = _run(geometry, source, shape)
    _, cubic, nlocal = _real_space_polynomial(geometry, shape, _SCALE)
    expected = _fourier_vertex(shape, cubic, _unitary_basis(shape, nlocal))
    addresses = actual.vertex.active_dofs
    assert np.all(addresses % 3 == 0)
    assert np.all(addresses < 2 * geometry.n_principal_layer_dofs)
    local = addresses // 3
    nq = actual.vertex.mesh.n_qpoints
    expected = expected[np.ix_(np.arange(nq), np.arange(nq), local, local, local)]
    np.testing.assert_allclose(actual.vertex.coefficients, expected, rtol=2e-12, atol=2e-13)
    assert actual.vertex.full_ndof == 3 * nlocal
    assert not actual.vertex.coefficients.flags.writeable
    assert not addresses.flags.writeable
    assert actual.metadata["geometry_passes"] == 2
    assert actual.metadata["scalar_schedule_materialized"] is False
    assert actual.metadata["full_dense_vertex_allocated"] is False
    assert actual.metadata["coupling_applied"] is False
    assert actual.metadata["support_cutoff"] == 0.0
    for term, original in zip(source.terms, before, strict=True):
        np.testing.assert_array_equal(term.values, original)
    if shape == (4, 1, 1):
        assert np.max(np.abs(actual.vertex.coefficients.imag)) > 1e-3


def test_mixed_cartesian_blocks_equal_dense_mapper_without_full_allocation(monkeypatch):
    geometry, source = _inputs(np.eye(3, dtype=int), mixed_cartesian=True, r0=True)
    expected = fourier_material_cubic(
        UniformBZMesh((4, 1, 1)), source, geometry,
        n_device_layers=3, interaction_layers=(0, 1), scale=_SCALE,
    )
    full_d = expected.vertex.ndof
    _reject_cubic_allocations(monkeypatch, full_d=full_d)
    actual = _run(geometry, source)
    indices = np.arange(4)
    active = actual.vertex.active_dofs
    np.testing.assert_array_equal(
        actual.vertex.coefficients,
        expected.vertex.coefficients[np.ix_(indices, indices, active, active, active)],
    )
    assert actual.metadata["output_scalar_terms"] == expected.metadata["output_scalar_terms"]
    assert actual.metadata["expanded_finite_images"] == expected.metadata["expanded_finite_images"]
    assert actual.metadata["phase_evaluations"] == expected.metadata["phase_evaluations"]
    assert actual.metadata["candidate_active_dofs"] < full_d


def _reject_cubic_allocations(monkeypatch, *, full_d=None):
    def instrument(original):
        def guarded(shape, *args, **kwargs):
            if (isinstance(shape, (tuple, list)) and len(shape) == 5
                    and (full_d is None or shape[2:] == (full_d,) * 3)):
                pytest.fail("Forbidden cubic allocation before preflight or full-D allocation.")
            return original(shape, *args, **kwargs)
        return guarded
    monkeypatch.setattr(np, "zeros", instrument(np.zeros))
    monkeypatch.setattr(np, "empty", instrument(np.empty))


@pytest.mark.parametrize("budget", [
    {"max_bytes": 128},
    {"max_geometry_evaluations": 1},
    {"max_phase_evaluations": 1},
    {"max_workspace_bytes": 1},
    {"max_workspace_bytes": 60000},
])
def test_limits_precede_any_cubic_allocation(monkeypatch, budget):
    geometry, source = _inputs(np.eye(3, dtype=int), mixed_cartesian=True)
    _reject_cubic_allocations(monkeypatch)
    with pytest.raises(ValueError, match="exceed"):
        _run(geometry, source, **budget)


def test_changed_source_rejected_before_allocation(monkeypatch):
    geometry, source = _inputs(np.eye(3, dtype=int))
    changed = copy.copy(source)
    term = copy.copy(source.terms[0])
    object.__setattr__(term, "values", term.values + 1.0)
    object.__setattr__(changed, "terms", (term, *source.terms[1:]))
    _reject_cubic_allocations(monkeypatch)
    with pytest.raises(ValueError, match="content no longer matches"):
        _run(geometry, changed)


def test_tiny_nonzero_coefficients_preserved_without_threshold():
    geometry, source = _inputs(np.eye(3, dtype=int))
    source = _rehash(source, lambda term: {"values": term.values * 1e-200})
    actual = _run(geometry, source)
    assert actual.vertex.ndof == 4
    assert 0 < np.max(np.abs(actual.vertex.coefficients)) < 1e-200
    assert actual.metadata["exactly_cancelled_dofs"] == 0


def test_exact_cancellation_compacts_to_empty_support():
    geometry, source = _inputs(np.eye(3, dtype=int))
    negative = _rehash(source, lambda term: {"values": -term.values})
    combined = copy.copy(source)
    # Interleave opposite terms: same ordering keys, exact floating cancellation.
    terms = tuple(term for pair in zip(source.terms, negative.terms, strict=True) for term in pair)
    object.__setattr__(combined, "terms", terms)
    object.__setattr__(combined, "content_sha256", domain_digest(
        "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3", _normalized_payload(combined)
    ))
    actual = _run(geometry, combined)
    assert actual.vertex.ndof == 0
    assert actual.vertex.coefficients.shape == (4, 4, 0, 0, 0)
    assert actual.metadata["candidate_active_dofs"] == 4
    assert actual.metadata["exactly_cancelled_dofs"] == 4
    assert actual.vertex.full_ndof == 18


def test_all_zero_source_does_not_require_nonempty_cubic_support():
    geometry, source = _inputs(np.eye(3, dtype=int))
    source = _rehash(source, lambda term: {"values": np.zeros_like(term.values)})
    actual = _run(geometry, source)
    assert actual.vertex.ndof == 0
    assert actual.metadata["geometry_evaluations"] == 0
    assert actual.metadata["phase_evaluations"] == 0


@pytest.mark.parametrize("omega0", [1e200, 1e-200])
def test_conversion_overflow_underflow_rejected_without_allocation(monkeypatch, omega0):
    geometry, source = _inputs(np.eye(3, dtype=int))
    # Bypass the existing SolverScale constructor, which already overflows for
    # these values, to exercise the new mapper's defensive conversion guard.
    scale = copy.copy(_SCALE)
    object.__setattr__(scale, "omega0", omega0)
    _reject_cubic_allocations(monkeypatch)
    with pytest.raises(ValueError, match="not representable"):
        _run(geometry, source, scale=scale)


def test_weight_underflow_is_not_silent_zero_support(monkeypatch):
    geometry, source = _inputs(np.eye(3, dtype=int))
    source = _rehash(source, lambda term: {
        "values": term.values * 1e-200, "anchor_weight": 1e-200,
    })
    _reject_cubic_allocations(monkeypatch)
    with pytest.raises(ValueError, match="weighted FC3 coefficient over/underflows"):
        _run(geometry, source)


def test_partial_cancellation_retains_noncontiguous_candidate_coordinates():
    geometry, source = _inputs(np.eye(3, dtype=int))

    def add_y(term):
        values = term.values.copy()
        values[1, 1, 1] = values[0, 0, 0]
        return {"values": values}

    def negative_y(term):
        values = np.zeros_like(term.values)
        values[1, 1, 1] = -term.values[0, 0, 0]
        return {"values": values}

    xy = _rehash(source, add_y)
    minus_y = _rehash(source, negative_y)
    combined = copy.copy(source)
    object.__setattr__(combined, "terms", tuple(
        term for pair in zip(xy.terms, minus_y.terms, strict=True) for term in pair
    ))
    object.__setattr__(combined, "content_sha256", domain_digest(
        "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3", _normalized_payload(combined)
    ))
    actual = _run(geometry, combined)
    _, cubic, nlocal = _real_space_polynomial(geometry, (4, 1, 1), _SCALE)
    expected = _fourier_vertex((4, 1, 1), cubic, _unitary_basis((4, 1, 1), nlocal))
    assert actual.metadata["candidate_active_dofs"] == 8
    assert actual.metadata["exactly_cancelled_dofs"] == 4
    np.testing.assert_array_equal(actual.vertex.active_dofs, [0, 3, 6, 9])
    np.testing.assert_allclose(actual.vertex.coefficients, expected[:, :, :4, :4, :4],
                               rtol=2e-12, atol=2e-13)
