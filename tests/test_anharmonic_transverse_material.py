"""Focused unit, crop and input-integrity checks for the raw IFC bridge."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from test_anharmonic_layered_phase_a import (
    _chain_mapping,
    _compact_chain_fc2,
    _physical_scale,
)
from test_anharmonic_mapped_fc3 import _layered_mapping, _normalized

from negfpy.anharmonic.layered import (
    build_layered_device_mapping,
    build_layered_harmonic_model,
)
from negfpy.anharmonic.mapped_fc3 import map_fc3_to_device
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_mapping import fourier_transverse_cubic
from negfpy.anharmonic.transverse_material import (
    map_material_cubic,
    material_transverse_harmonic,
)
from negfpy.anharmonic.types import SolverScale


@pytest.mark.parametrize("cross_cell", [False, True])
@pytest.mark.parametrize("transform", [
    np.eye(3, dtype=int), np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]]),
])
def test_single_q_reduction_matches_existing_finite_crop(cross_cell, transform):
    mapping = _layered_mapping(transform=transform)
    normalized = _normalized(cross_cell=cross_cell, duplicate_images=True)
    scale = SolverScale(omega0=4, hbar=9)
    mapped = map_material_cubic(
        normalized, mapping, n_device_layers=5, interaction_layers=(1, 2, 3), scale=scale,
    )
    old = map_fc3_to_device(
        normalized, mapping, n_device_layers=5, interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    assert old.report.certification_passed
    vertex = fourier_transverse_cubic(UniformBZMesh((1, 1, 1)), mapped.ndof, mapped.terms)
    expected = np.zeros((mapped.ndof,) * 3)
    expected[tuple(old.indices.T)] = old.values * 3 / 4**2.5
    np.testing.assert_allclose(vertex.coefficients[0, 0], expected, rtol=1e-14, atol=0)
    layer_dofs = mapping.n_principal_layer_dofs
    assert all(layer_dofs <= d < 4 * layer_dofs for t in mapped.terms for d in t.indices)
    assert all(t.weight == 0.25 for t in mapped.terms)
    assert mapped.metadata["weights_applied"] is False
    assert mapped.metadata["source_normalized_fc3_sha256"] == normalized.content_sha256
    with pytest.raises(TypeError):
        mapped.metadata["weights_applied"] = True
    assert not normalized.terms[0].values.flags.writeable


def test_raw_harmonic_si_conversion_matches_existing_single_q_chain():
    mapping = build_layered_device_mapping(
        _chain_mapping(), _compact_chain_fc2(),
        transport_supercell_matrix=np.eye(3, dtype=int),
    )
    scale = _physical_scale()
    device, lead = material_transverse_harmonic(mapping, n_device_layers=3, scale=scale)
    expected = build_layered_harmonic_model(mapping, scale)
    for actual, ref in zip(lead.blocks((0.0, 0.0)),
                           (expected.d00, expected.d01, expected.d10), strict=True):
        np.testing.assert_allclose(actual, ref, rtol=1e-14, atol=0)
    assert device.device((0.0, 0.0)).n_layers == 3
    np.testing.assert_allclose(device.device((0.3, -0.7)).onsite_blocks[0], expected.d00)


@pytest.mark.parametrize("crop", [(), (2, 1), (1, 1), (0, 2), (-1,), (3,), (True,)])
def test_invalid_interaction_crops_fail_before_mapping(crop):
    with pytest.raises((ValueError, TypeError), match="interaction_layers"):
        map_material_cubic(
            _normalized(), _layered_mapping(), n_device_layers=3,
            interaction_layers=crop, scale=SolverScale(),
        )


@pytest.mark.parametrize("budget", [
    {"max_terms": 1}, {"max_geometry_evaluations": 1},
])
def test_cubic_mapping_respects_bounded_work_and_output(budget):
    with pytest.raises(ValueError, match="exceeds"):
        map_material_cubic(
            _normalized(), _layered_mapping(), n_device_layers=3,
            interaction_layers=(0, 1, 2), scale=SolverScale(), **budget,
        )


@pytest.mark.parametrize("budget", [
    {"max_bytes": 1}, {"max_image_evaluations": 1},
])
def test_harmonic_mapping_respects_bounded_work_and_storage(budget):
    with pytest.raises(ValueError, match="exceed"):
        material_transverse_harmonic(
            _layered_mapping(), n_device_layers=3, scale=SolverScale(), **budget,
        )


def test_changed_normalized_source_is_rejected():
    normalized = copy.copy(_normalized())
    term = copy.copy(normalized.terms[0])
    values = term.values.copy()
    values[0, 0, 0] += 1
    object.__setattr__(term, "values", values)
    object.__setattr__(normalized, "terms", (term, *normalized.terms[1:]))
    with pytest.raises(ValueError, match="content no longer matches"):
        map_material_cubic(
            normalized, _layered_mapping(), n_device_layers=3,
            interaction_layers=(0, 1, 2), scale=SolverScale(),
        )


def test_fc2_fc3_primitive_identity_mismatch_is_rejected():
    normalized = copy.copy(_normalized())
    object.__setattr__(normalized, "primitive_crystal_content_sha256", "0" * 64)
    with pytest.raises(ValueError, match="primitive-crystal identities differ"):
        map_material_cubic(
            normalized, _layered_mapping(), n_device_layers=3,
            interaction_layers=(0, 1, 2), scale=SolverScale(),
        )


def test_changed_embedded_fc2_is_rejected():
    mapping = copy.copy(_layered_mapping())
    canonical = copy.copy(mapping._canonical_fc2)
    values = canonical.values.copy()
    values[0, 0, 0, 0] += 1
    object.__setattr__(canonical, "values", values)
    object.__setattr__(mapping, "_canonical_fc2", canonical)
    with pytest.raises(ValueError, match="FC2 content no longer matches"):
        material_transverse_harmonic(mapping, n_device_layers=3, scale=SolverScale())
