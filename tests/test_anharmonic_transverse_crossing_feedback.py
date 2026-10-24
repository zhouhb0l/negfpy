"""Focused symmetric-support and independent central-slice checks."""
from __future__ import annotations

import time

import numpy as np
import pytest
from test_anharmonic_transverse_material_streaming import _inputs

from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_crossing_feedback import (
    MODEL,
    build_crossing_feedback_vertex,
)
from negfpy.anharmonic.transverse_crossing_support import build_crossing_support_vertex
from negfpy.anharmonic.transverse_material_packed import fourier_material_cubic_packed
from negfpy.anharmonic.types import SolverScale

_SCALE = SolverScale(omega0=2., hbar=3.)


@pytest.mark.parametrize('complex_source', [False, True])
def test_symmetric_selection_preserves_actual_values_and_rectangular_central_slice(complex_source):
    transform = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]]) if complex_source else np.eye(3, dtype=int)
    geometry, source = _inputs(transform, r0=complex_source,
                               duplicate_images=complex_source, mixed_cartesian=complex_source)
    snapshots = [term.values.copy() for term in source.terms]
    mesh = UniformBZMesh((4, 1, 1))
    kwargs = {'n_device_layers': 4, 'interaction_layers': (1, 2), 'scale': _SCALE}
    crop = fourier_material_cubic_packed(mesh, source, geometry, **kwargs).vertex
    full = fourier_material_cubic_packed(mesh, source, geometry,
        n_device_layers=4, interaction_layers=(0, 1, 2, 3), scale=_SCALE).vertex
    actual = build_crossing_feedback_vertex(mesh, source, geometry, reference_vertex=crop, **kwargs)
    np.testing.assert_array_equal(actual.vertex.active_dofs, full.active_dofs)
    outside = np.flatnonzero((full.active_dofs < geometry.n_principal_layer_dofs)
                            | (full.active_dofs >= 3*geometry.n_principal_layer_dofs))
    expected = full.coefficients.copy()
    for q in range(4):
        for p in range(4):
            expected[q, p][np.ix_(outside, outside, outside)] = 0
    np.testing.assert_array_equal(actual.vertex.coefficients, expected)
    for before, term in zip(snapshots, source.terms, strict=True):
        np.testing.assert_array_equal(before, term.values)
    assert not actual.vertex.coefficients.flags.writeable
    assert actual.metadata['model'] == MODEL
    assert actual.metadata['retained_coefficients_unchanged']
    assert actual.metadata['global_asr_restored'] is False
    assert actual.metadata['support']['complete_required_support']
    assert actual.metadata['support']['triplets_counted_once']
    assert all(row['passed'] for row in actual.metadata['production_crop_comparison'])
    rectangular = build_crossing_support_vertex(mesh, source, geometry, reference_vertex=crop, **kwargs)
    index = {int(address): i for i, address in enumerate(actual.vertex.active_dofs)}
    ext = [index[int(address)] for address in crop.active_dofs]
    for p in range(4):
        expanded = np.zeros_like(rectangular.coefficients[p])
        expanded[:, full.active_dofs[:, None], full.active_dofs] = actual.vertex.coefficients[0, p, ext]
        np.testing.assert_allclose(expanded, rectangular.coefficients[p], rtol=2e-13, atol=2e-15)
    if complex_source:
        assert np.max(np.abs(actual.vertex.coefficients.imag)) > 1e-3


def test_missing_any_leg_support_is_rejected_before_mapping(monkeypatch):
    import negfpy.anharmonic.transverse_crossing_feedback as module

    geometry, source = _inputs(np.eye(3, dtype=int))
    monkeypatch.setattr(module, 'fourier_material_cubic_packed',
                        lambda *a, **k: pytest.fail('Full mapper must not run after failed support admission'))
    with pytest.raises(ValueError, match='Crossing feedback support exceeds the harmonic device'):
        build_crossing_feedback_vertex(UniformBZMesh((4, 1, 1)), source, geometry,
            n_device_layers=3, interaction_layers=(0, 1), scale=_SCALE)


def test_time_and_geometry_admission_precede_full_mapping(monkeypatch):
    import negfpy.anharmonic.transverse_crossing_feedback as module

    geometry, source = _inputs(np.eye(3, dtype=int))
    monkeypatch.setattr(module, 'fourier_material_cubic_packed',
                        lambda *a, **k: pytest.fail('Full mapper must not run after failed admission'))
    kwargs = {'n_device_layers': 4, 'interaction_layers': (1, 2), 'scale': _SCALE}
    with pytest.raises(TimeoutError, match='deadline'):
        build_crossing_feedback_vertex(UniformBZMesh((4, 1, 1)), source, geometry,
            deadline_monotonic=time.monotonic()-1, **kwargs)
    with pytest.raises(ValueError, match='geometry'):
        build_crossing_feedback_vertex(UniformBZMesh((4, 1, 1)), source, geometry,
            max_geometry_evaluations=1, **kwargs)
