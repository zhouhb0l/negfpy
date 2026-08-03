from __future__ import annotations

import inspect

import numpy as np
import pytest

from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.bulk_harmonic import dynamical_matrix_from_fc2
from negfpy.anharmonic.fc2 import (
    ATOMIC_MASS_UNIT_KG,
    EV_ANGSTROM2_TO_J_M2,
    FC2Data,
)
from negfpy.anharmonic.layered import (
    LayeredDeviceMapping,
    LayeredHarmonicModel,
    build_layered_device_mapping,
    build_layered_harmonic_model,
)
from negfpy.anharmonic.types import SolverScale


HBAR_SI = 1.0545718176461565e-34


def _chain_mapping(*, translations: bool = True) -> CrystalMapping:
    primitive = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0]],
        masses=[2.0],
        symbols=("X",),
    )
    supercell = CrystalCell(
        lattice=np.diag([3.0, 1.0, 1.0]),
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [2.0 / 3.0, 0.0, 0.0],
        ],
        masses=[2.0, 2.0, 2.0],
        symbols=("X", "X", "X"),
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.diag([3, 1, 1]),
        primitive=primitive,
        supercell=supercell,
        p2s=[1],
        s2p=[1, 1, 1],
        p2p=[-1, 0, -1],
        shortest_vectors=[
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        multiplicity=[[[1, 0]], [[1, 1]], [[1, 2]]],
        atomic_permutations=[[0, 1, 2]],
        translation_permutations=(
            [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
            if translations
            else None
        ),
        purpose="fc2",
    )


def _compact_chain_fc2() -> FC2Data:
    values = np.zeros((1, 3, 3, 3))
    values[0, 0] = -2.0 * np.eye(3)
    values[0, 1] = 4.0 * np.eye(3)
    values[0, 2] = -2.0 * np.eye(3)
    return FC2Data(
        values=values,
        masses=[2.0, 2.0, 2.0],
        compact=True,
        p2s_map=[1],
        s2p_map=[1, 1, 1],
    )


def _full_chain_fc2() -> FC2Data:
    representative = _compact_chain_fc2().values[0]
    values = np.empty((3, 3, 3, 3))
    values[1] = representative
    values[0] = representative[[1, 2, 0]]
    values[2] = representative[[2, 0, 1]]
    return FC2Data(
        values=values,
        masses=[2.0, 2.0, 2.0],
        compact=False,
        p2s_map=[1],
        s2p_map=[1, 1, 1],
    )


def _physical_scale(omega0: float = 2.0 * np.pi * 1.0e13) -> SolverScale:
    return SolverScale(
        omega0=omega0,
        hbar=HBAR_SI,
        metadata={"source_units": {"fc2": "eV/angstrom^2", "mass": "amu"}},
    )


def test_public_material_types_are_builder_only() -> None:
    assert list(inspect.signature(LayeredDeviceMapping).parameters) == []
    assert list(inspect.signature(LayeredHarmonicModel).parameters) == []
    with pytest.raises(TypeError):
        LayeredDeviceMapping()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        LayeredHarmonicModel()  # type: ignore[call-arg]


def test_compact_chain_maps_all_images_and_builds_matching_lead_device() -> None:
    mapping = build_layered_device_mapping(
        _chain_mapping(),
        _compact_chain_fc2(),
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    assert mapping.principal_cells == 1
    assert mapping.n_transport_basis_atoms == 1
    assert mapping.n_principal_layer_dofs == 3
    np.testing.assert_array_equal(mapping.layer_offsets, [-1, 0, 1])
    assert mapping._geometry_metrics["n_fc2_image_records"] == 3
    assert mapping.longitudinal_fc2_support.shape == (27, 3)
    assert mapping.longitudinal_fc2_exact_numerical_support.shape == (9, 3)
    assert len(mapping.longitudinal_fc2_exact_numerical_support_sha256) == 64
    assert mapping.validation.certification_passed

    scale = _physical_scale()
    model = build_layered_harmonic_model(mapping, scale)
    expected_onsite = (
        4.0
        * EV_ANGSTROM2_TO_J_M2
        / (2.0 * ATOMIC_MASS_UNIT_KG)
        / scale.omega0**2
    )
    expected_coupling = -0.5 * expected_onsite
    np.testing.assert_allclose(model.d00, expected_onsite * np.eye(3))
    np.testing.assert_allclose(model.d01, expected_coupling * np.eye(3))
    np.testing.assert_array_equal(model.d10, model.d01.conj().T)
    assert model.validation.certification_passed

    device = model.device(3)
    assert device.n_layers == 3
    assert device.dof_per_layer == 3
    np.testing.assert_array_equal(device.onsite_blocks[0], model.lead.d00)
    np.testing.assert_array_equal(device.coupling_blocks[0], model.lead.d01)
    assert device.onsite_blocks[0] is not model.d00
    assert device.coupling_blocks[0] is not model.d01


def test_full_redundant_rows_are_audited_not_used_as_alternate_operator() -> None:
    crystal = _chain_mapping()
    compact = build_layered_device_mapping(
        crystal,
        _compact_chain_fc2(),
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    full = build_layered_device_mapping(
        crystal,
        _full_chain_fc2(),
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    assert compact.fc2_content_sha256 != full.fc2_content_sha256
    assert compact.layered_geometry_sha256 == full.layered_geometry_sha256
    assert full._geometry_metrics["n_redundant_first_axis_rows"] == 2
    assert full._geometry_metrics["redundant_rows_max_abs_source"] == 0.0
    compact_model = build_layered_harmonic_model(compact, _physical_scale())
    full_model = build_layered_harmonic_model(full, _physical_scale())
    np.testing.assert_allclose(full_model.d00, compact_model.d00)
    np.testing.assert_allclose(full_model.d01, compact_model.d01)
    np.testing.assert_allclose(full_model.d10, compact_model.d10)


def test_full_redundant_row_mismatch_fails_closed() -> None:
    source = _full_chain_fc2()
    values = source.values.copy()
    values[0, 0, 0, 0] += 1.0e-3
    corrupt = FC2Data(
        values=values,
        masses=source.masses,
        compact=False,
        p2s_map=source.p2s_map,
        s2p_map=source.s2p_map,
    )
    with pytest.raises(ValueError, match="redundant translation audit failed"):
        build_layered_device_mapping(
            _chain_mapping(),
            corrupt,
            transport_supercell_matrix=np.eye(3, dtype=np.int64),
        )


def test_nonsymmetric_transport_transform_uses_independent_quotients() -> None:
    transform = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 1]])
    mapping = build_layered_device_mapping(
        _chain_mapping(),
        _compact_chain_fc2(),
        transport_supercell_matrix=transform,
    )
    assert mapping.signed_determinant == 4
    assert mapping.n_transport_basis_atoms == 4
    assert mapping.direct_coset_keys.shape == (4, 3)
    assert mapping.reciprocal_coset_keys.shape == (4, 3)
    assert mapping.direct_cosets_sha256 != mapping.reciprocal_cosets_sha256
    assert mapping.validation.certification_passed


def test_nonsymmetric_folded_spectrum_matches_independent_primitive_oracle() -> None:
    transform = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 1]])
    crystal = _chain_mapping()
    fc2 = _compact_chain_fc2()
    mapping = build_layered_device_mapping(
        crystal,
        fc2,
        transport_supercell_matrix=transform,
    )
    model = build_layered_harmonic_model(mapping, _physical_scale())
    source_to_physical = EV_ANGSTROM2_TO_J_M2 / ATOMIC_MASS_UNIT_KG
    inverse_transpose = np.linalg.inv(transform).T
    for longitudinal_q in (0.0, 0.13, 0.5):
        phase = np.exp(2j * np.pi * longitudinal_q)
        layered_source = (
            model.d00 + model.d01 * phase + model.d10 / phase
        ) * model.scale.omega0**2 / source_to_physical
        observed = np.linalg.eigvalsh(layered_source)
        expected: list[float] = []
        transport_q = np.array([longitudinal_q, 0.0, 0.0])
        for reciprocal_coset in mapping.reciprocal_coset_representatives:
            primitive_q = (transport_q + reciprocal_coset) @ inverse_transpose
            expected.extend(
                np.linalg.eigvalsh(
                    dynamical_matrix_from_fc2(
                        fc2,
                        crystal,
                        primitive_q,
                        hermitianize=True,
                    )
                )
            )
        np.testing.assert_allclose(
            np.sort(observed),
            np.sort(expected),
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_minimum_principal_layer_is_inferred_without_dropping_range() -> None:
    primitive = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0]],
        masses=[1.0],
        symbols=("X",),
    )
    supercell = CrystalCell(
        lattice=np.diag([5.0, 1.0, 1.0]),
        scaled_positions=[[n / 5.0, 0.0, 0.0] for n in range(5)],
        masses=np.ones(5),
        symbols=("X",) * 5,
    )
    crystal = CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.diag([5, 1, 1]),
        primitive=primitive,
        supercell=supercell,
        p2s=[2],
        s2p=[2, 2, 2, 2, 2],
        p2p=[-1, -1, 0, -1, -1],
        shortest_vectors=[
            [-2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        multiplicity=[[[1, n]] for n in range(5)],
        atomic_permutations=[np.arange(5)],
        purpose="fc2",
    )
    values = np.zeros((1, 5, 3, 3))
    values[0, 0] = -np.eye(3)
    values[0, 2] = 2.0 * np.eye(3)
    values[0, 4] = -np.eye(3)
    fc2 = FC2Data(
        values=values,
        masses=np.ones(5),
        compact=True,
        p2s_map=[2],
        s2p_map=[2, 2, 2, 2, 2],
    )
    mapping = build_layered_device_mapping(
        crystal,
        fc2,
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    assert mapping.principal_cells == 2
    np.testing.assert_array_equal(mapping.layer_offsets, [-1, 0, 1])
    zero_distant_values = np.zeros_like(values)
    zero_distant_values[0, 2] = np.eye(3)
    zero_distant_mapping = build_layered_device_mapping(
        crystal,
        FC2Data(
            values=zero_distant_values,
            masses=np.ones(5),
            compact=True,
            p2s_map=[2],
            s2p_map=[2, 2, 2, 2, 2],
        ),
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    # Principal-layer inference uses the value-independent image envelope;
    # exact zeros at the distant image cannot shrink the geometry.
    assert zero_distant_mapping.principal_cells == 2
    assert zero_distant_mapping.longitudinal_fc2_support.shape == (45, 3)
    assert (
        zero_distant_mapping.longitudinal_fc2_exact_numerical_support.shape
        == (3, 3)
    )
    model = build_layered_harmonic_model(mapping, _physical_scale())
    assert model.d00.shape == (6, 6)
    assert model.d01.shape == (6, 6)


def test_two_reference_scales_preserve_physical_blocks() -> None:
    mapping = build_layered_device_mapping(
        _chain_mapping(),
        _compact_chain_fc2(),
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    scale_a = _physical_scale(2.0 * np.pi * 1.0e13)
    scale_b = _physical_scale(4.0 * np.pi * 1.0e13)
    model_a = build_layered_harmonic_model(mapping, scale_a)
    model_b = build_layered_harmonic_model(mapping, scale_b)
    np.testing.assert_allclose(
        model_a.d00 * scale_a.omega0**2,
        model_b.d00 * scale_b.omega0**2,
        rtol=1.0e-15,
    )
    np.testing.assert_allclose(
        model_a.d01 * scale_a.omega0**2,
        model_b.d01 * scale_b.omega0**2,
        rtol=1.0e-15,
    )


def test_input_and_returned_arrays_are_detached_and_immutable() -> None:
    fc2 = _compact_chain_fc2()
    source_before = fc2.values.copy()
    mapping = build_layered_device_mapping(
        _chain_mapping(),
        fc2,
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    np.testing.assert_array_equal(fc2.values, source_before)
    with pytest.raises(ValueError):
        mapping.basis_scaled_positions[0, 0] = 2.0
    model = build_layered_harmonic_model(mapping, _physical_scale())
    with pytest.raises(ValueError):
        model.d00[0, 0] = 0.0


def test_builder_rejects_mapping_and_mass_mismatches() -> None:
    fc2 = _compact_chain_fc2()
    with pytest.raises(ValueError, match="p2s_map conflicts"):
        build_layered_device_mapping(
            _chain_mapping(),
            FC2Data(
                values=fc2.values,
                masses=fc2.masses,
                compact=True,
                p2s_map=[0],
                s2p_map=[0, 0, 0],
            ),
            transport_supercell_matrix=np.eye(3, dtype=np.int64),
        )
    with pytest.raises(ValueError, match="masses must exactly match"):
        build_layered_device_mapping(
            _chain_mapping(),
            FC2Data(
                values=fc2.values,
                masses=[3.0, 3.0, 3.0],
                compact=True,
                p2s_map=fc2.p2s_map,
                s2p_map=fc2.s2p_map,
            ),
            transport_supercell_matrix=np.eye(3, dtype=np.int64),
        )
