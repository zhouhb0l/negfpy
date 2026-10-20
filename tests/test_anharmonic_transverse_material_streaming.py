"""Streamed image-block Fourier mapping versus explicit scalar contributions."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
from test_anharmonic_transverse_material_reference import (
    _CUBIC,
    _MASSES,
    _crystal,
    _raw_inputs,
)

from negfpy.anharmonic.fc3 import FC3Data, validate_fc3
from negfpy.anharmonic.mapped_fc3 import mass_normalize_translational_fc3
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import (
    PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
    convert_fc3_to_translational,
)
from negfpy.anharmonic.transverse_mapping import fourier_transverse_cubic
from negfpy.anharmonic.transverse_material import map_material_cubic
from negfpy.anharmonic.types import SolverScale


def _inputs(transform, *, duplicate_images=False, r0=False, mixed_cartesian=False):
    geometry, normalized = _raw_inputs(
        transform, duplicate_images=duplicate_images, r0=r0
    )
    if not mixed_cartesian:
        return geometry, normalized

    # The same physical bond polynomial with all Cartesian components active.
    # This makes every streamed 3x3x3 block sensitive to Cartesian ordering.
    direction = np.array([1.0, 0.3, -0.2])
    displacement = np.array([direction, -direction])
    raw = FC3Data(
        _CUBIC
        * np.einsum("ia,jb,kc->ijkabc", displacement, displacement, displacement),
        _MASSES,
        units="J/m^3",
        mass_units="kg",
        compact=False,
        p2s_map=[0, 1],
        s2p_map=[0, 1],
    )
    crystal = _crystal("fc3", duplicate_images)
    translated = convert_fc3_to_translational(raw, crystal)
    if r0:
        translated = replace(
            translated,
            anchor_convention=PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
            make_r0_average=True,
            all_shortest=np.zeros((2, 2, 2), dtype=np.byte),
            symprec=1e-5,
        )
    return geometry, mass_normalize_translational_fc3(
        translated, crystal, source_certification=validate_fc3(raw)
    )


@pytest.mark.parametrize(
    "transform,shape,duplicate_images,r0,mixed_cartesian",
    [
        (np.eye(3, dtype=int), (1, 1, 1), False, False, False),
        (np.eye(3, dtype=int), (4, 1, 1), False, False, True),
        (
            np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]]),
            (4, 1, 1),
            True,
            True,
            True,
        ),
        (np.eye(3, dtype=int), (2, 2, 1), True, True, True),
    ],
    ids=("gamma", "complex-cartesian", "sheared-cosets-r0", "two-transverse-axes"),
)
def test_streamed_vertex_matches_separately_mapped_scalar_fourier_sum(
    transform, shape, duplicate_images, r0, mixed_cartesian
) -> None:
    from negfpy.anharmonic.transverse_material import fourier_material_cubic

    geometry, normalized = _inputs(
        transform,
        duplicate_images=duplicate_images,
        r0=r0,
        mixed_cartesian=mixed_cartesian,
    )
    source_before = [term.values.copy() for term in normalized.terms]
    scale = SolverScale(omega0=2.0, hbar=3.0)
    mesh = UniformBZMesh(shape)
    mapped = map_material_cubic(
        normalized,
        geometry,
        n_device_layers=3,
        interaction_layers=(0, 1),
        scale=scale,
    )
    expected = fourier_transverse_cubic(mesh, mapped.ndof, mapped.terms)
    actual = fourier_material_cubic(
        mesh,
        normalized,
        geometry,
        n_device_layers=3,
        interaction_layers=(0, 1),
        scale=scale,
    )
    np.testing.assert_allclose(
        actual.vertex.coefficients,
        expected.coefficients,
        rtol=2e-12,
        atol=2e-13,
    )
    assert not actual.vertex.coefficients.flags.writeable
    assert actual.metadata["output_scalar_terms"] == len(mapped.terms)
    assert (
        actual.metadata["expanded_finite_images"]
        == mapped.metadata["expanded_finite_images"]
    )
    assert actual.metadata["weights_applied"] is True
    assert actual.metadata["scalar_schedule_materialized"] is False
    assert actual.metadata["coupling_applied"] is False
    assert actual.metadata["geometry_passes"] == 2
    assert actual.metadata["phase_evaluations"] == mesh.n_qpoints**2 * len(mapped.terms)
    if shape == (4, 1, 1):
        assert np.max(np.abs(actual.vertex.coefficients.imag)) > 1e-3
    if mixed_cartesian:
        assert np.max(np.abs(actual.vertex.coefficients[:, :, 1::3, 2::3, 0::3])) > 0
    if r0:
        assert mapped.metadata["effective_image_terms"] == 3 * len(normalized.terms)

    # Every cubic leg remains inside the original interaction crop.
    crop_end = 2 * geometry.n_principal_layer_dofs
    for axis in range(2, 5):
        selection = [slice(None)] * 5
        selection[axis] = slice(crop_end, None)
        assert np.count_nonzero(actual.vertex.coefficients[tuple(selection)]) == 0
    for before, term in zip(source_before, normalized.terms, strict=True):
        np.testing.assert_array_equal(term.values, before)


def _reject_dense_allocation(monkeypatch):
    import negfpy.anharmonic.transverse_material as material

    def instrument(original):
        def guarded(shape, *args, **kwargs):
            if isinstance(shape, (tuple, list)) and len(shape) == 5:
                pytest.fail(
                    "Dense vertex allocation occurred before a failed preflight."
                )
            return original(shape, *args, **kwargs)

        return guarded

    monkeypatch.setattr(material.np, "zeros", instrument(material.np.zeros))
    monkeypatch.setattr(material.np, "empty", instrument(material.np.empty))
    return material


@pytest.mark.parametrize(
    "budget",
    [
        {"max_bytes": 1},
        {"max_geometry_evaluations": 1},
        {"max_phase_evaluations": 1},
    ],
)
def test_streaming_budgets_reject_before_dense_vertex_allocation(
    budget, monkeypatch
) -> None:
    geometry, normalized = _inputs(
        np.eye(3, dtype=int), duplicate_images=True, r0=True, mixed_cartesian=True
    )
    material = _reject_dense_allocation(monkeypatch)
    with pytest.raises(ValueError, match="exceed"):
        material.fourier_material_cubic(
            UniformBZMesh((4, 1, 1)),
            normalized,
            geometry,
            n_device_layers=3,
            interaction_layers=(0, 1),
            scale=SolverScale(omega0=2, hbar=3),
            **budget,
        )


def test_changed_streaming_source_is_rejected_before_dense_allocation(monkeypatch):
    geometry, original = _inputs(np.eye(3, dtype=int))
    normalized = copy.copy(original)
    term = copy.copy(original.terms[0])
    changed = term.values.copy()
    changed[0, 0, 0] += 1.0
    object.__setattr__(term, "values", changed)
    object.__setattr__(normalized, "terms", (term, *original.terms[1:]))
    material = _reject_dense_allocation(monkeypatch)
    with pytest.raises(ValueError, match="content no longer matches"):
        material.fourier_material_cubic(
            UniformBZMesh((4, 1, 1)),
            normalized,
            geometry,
            n_device_layers=3,
            interaction_layers=(0, 1),
            scale=SolverScale(omega0=2, hbar=3),
        )
