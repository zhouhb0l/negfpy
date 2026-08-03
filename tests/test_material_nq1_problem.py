from __future__ import annotations

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from negfpy.anharmonic._canonical import digest_bytes, domain_digest
from negfpy.anharmonic._material_nq1_problem import (
    MaterialNQ1ProblemError,
    build_material_nq1_problem,
)
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.layered import (
    build_layered_device_mapping,
    build_layered_harmonic_model,
)
from negfpy.anharmonic.mapped_fc3 import (
    DeviceMappedFC3,
    FC3DeviceMappingReport,
    _cubic_physics_sha256,
    _report_payload,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.types import SolverScale


HBAR_SI = 1.0545718176461565e-34


def _chain_mapping() -> CrystalMapping:
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
        translation_permutations=[[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        purpose="fc2",
    )


def _harmonic_model():
    values = np.zeros((1, 3, 3, 3))
    values[0, 0] = -2.0 * np.eye(3)
    values[0, 1] = 4.0 * np.eye(3)
    values[0, 2] = -2.0 * np.eye(3)
    fc2 = FC2Data(
        values=values,
        masses=[2.0, 2.0, 2.0],
        compact=True,
        p2s_map=[1],
        s2p_map=[1, 1, 1],
    )
    mapping = build_layered_device_mapping(
        _chain_mapping(),
        fc2,
        transport_supercell_matrix=np.eye(3, dtype=np.int64),
    )
    scale = SolverScale(
        omega0=2.0 * np.pi * 1.0e13,
        hbar=HBAR_SI,
        metadata={
            "source_units": {"fc2": "eV/angstrom^2", "mass": "amu"}
        },
    )
    return build_layered_harmonic_model(mapping, scale)


def _mapped_fc3(model) -> DeviceMappedFC3:
    n_layers = 9
    layer_dofs = model.mapping.n_principal_layer_dofs
    ndof = n_layers * layer_dofs
    triples: set[tuple[int, int, int]] = set()
    for triple in (
        (2 * layer_dofs, 4 * layer_dofs, 4 * layer_dofs),
        (4 * layer_dofs, 6 * layer_dofs, 6 * layer_dofs),
    ):
        triples.update(permutations(triple))
    indices = np.asarray(sorted(triples), dtype=np.int64)
    values = np.linspace(2.0, 3.0, len(indices), dtype=np.float64)
    # Make each permutation orbit bitwise equal.
    values[:3] = 2.0
    values[3:] = 3.0
    physics_hash = _cubic_physics_sha256(ndof, indices, values)
    report = FC3DeviceMappingReport(
        schema_version="negfpy.fc3-device-mapping.v2",
        raw_term_count=2,
        nonzero_raw_block_count=2,
        exact_zero_raw_block_count=0,
        effective_term_count=2,
        nonzero_effective_term_count=2,
        expanded_image_count=2,
        expanded_cartesian_component_count=54,
        exact_zero_contribution_count=48,
        duplicate_accumulation_count=0,
        exact_cancellation_count=0,
        final_nonzero_count=len(values),
        support_density=float(len(values) / ndof**3),
        pairwise_max_transport_cells=np.full(3, 2, dtype=np.int64),
        pairwise_max_principal_layers=np.full(3, 2, dtype=np.int64),
        pairwise_max_distance_m=np.full(3, 1.0e-10),
        required_left_halo_layers=2,
        required_right_halo_layers=2,
        n_device_layers=n_layers,
        contact_layers=(0, n_layers - 1),
        harmonic_buffer_layers=((0, 1), (7, 8)),
        interaction_layers=(2, 3, 4, 5, 6),
        observation_layers=(4,),
        permutation_max_abs=0.0,
        permutation_max_roundoff_ratio=0.0,
        permutation_roundoff_factor=32.0,
        permutation_missing_count=0,
        permutation_tolerance_passed=True,
        exact_permutation_equality=True,
        permutation_atol=0.0,
        permutation_rtol=float(32.0 * np.finfo(np.float64).eps),
        source_fc3_content_sha256="a" * 64,
        source_translational_sha256="b" * 64,
        mass_normalized_fc3_sha256="c" * 64,
        source_certification_kind="full_validation",
        source_certification_sha256="d" * 64,
        primitive_crystal_content_sha256=(
            model.primitive_crystal_content_sha256
        ),
        fc3_crystal_mapping_content_sha256="e" * 64,
        fc2_content_sha256=model.fc2_content_sha256,
        fc2_crystal_mapping_content_sha256=(
            model.fc2_crystal_mapping_content_sha256
        ),
        layered_geometry_sha256=model.layered_geometry_sha256,
        cubic_physics_sha256=physics_hash,
        symmetrized=False,
        corrected=False,
        truncated=False,
        thresholded=False,
        failure_codes=(),
        certification_passed=True,
        report_sha256="",
    )
    object.__setattr__(
        report,
        "report_sha256",
        domain_digest("NEGFPY-FC3-DEVICE-MAPPING", _report_payload(report)),
    )
    mapped = object.__new__(DeviceMappedFC3)
    content_hash = domain_digest(
        "NEGFPY-DEVICE-MAPPED-FC3",
        {
            "ndof": ndof,
            "indices": indices,
            "values": values,
            "interaction_layers": report.interaction_layers,
            "observation_layers": report.observation_layers,
            "report_sha256": digest_bytes(report.report_sha256),
        },
    )
    for name, value in {
        "ndof": ndof,
        "indices": indices,
        "values": values,
        "interaction_layers": report.interaction_layers,
        "observation_layers": report.observation_layers,
        "report": report,
        "metadata": {
            "schema_version": "negfpy.device-mapped-fc3.v1",
            "output_units": "J m^-3 kg^-3/2",
            "weights_applied": True,
            "solver_scaled": False,
            "symmetrized": False,
            "corrected": False,
            "truncated": False,
            "thresholded": False,
            "mapping_report_sha256": report.report_sha256,
        },
        "cubic_physics_sha256": physics_hash,
        "content_sha256": content_hash,
    }.items():
        object.__setattr__(mapped, name, value)
    return mapped


def _replace_report(
    mapped: DeviceMappedFC3,
    **changes: object,
) -> DeviceMappedFC3:
    report = replace(mapped.report, **changes, report_sha256="")
    object.__setattr__(
        report,
        "report_sha256",
        domain_digest("NEGFPY-FC3-DEVICE-MAPPING", _report_payload(report)),
    )
    result = object.__new__(DeviceMappedFC3)
    content_hash = domain_digest(
        "NEGFPY-DEVICE-MAPPED-FC3",
        {
            "ndof": mapped.ndof,
            "indices": mapped.indices,
            "values": mapped.values,
            "interaction_layers": mapped.interaction_layers,
            "observation_layers": mapped.observation_layers,
            "report_sha256": digest_bytes(report.report_sha256),
        },
    )
    for name, value in {
        "ndof": mapped.ndof,
        "indices": mapped.indices,
        "values": mapped.values,
        "interaction_layers": mapped.interaction_layers,
        "observation_layers": mapped.observation_layers,
        "report": report,
        "metadata": {
            **dict(mapped.metadata),
            "mapping_report_sha256": report.report_sha256,
        },
        "cubic_physics_sha256": mapped.cubic_physics_sha256,
        "content_sha256": content_hash,
    }.items():
        object.__setattr__(result, name, value)
    return result


def test_builds_exact_device_contacts_sparse_vertex_and_run_inputs() -> None:
    model = _harmonic_model()
    mapped = _mapped_fc3(model)
    grid = SignedFrequencyGrid(step=0.05, n_positive=8)
    problem = build_material_nq1_problem(
        model,
        mapped,
        grid=grid,
        mean_temperature=0.5,
        temperature_bias=0.2,
        coupling_scale=0.25,
    )

    assert problem.device.n_layers == mapped.report.n_device_layers == 9
    assert problem.ndof == mapped.ndof == 27
    assert isinstance(problem.vertex, SparseCubicVertex)
    assert problem.vertex.ndof == mapped.ndof
    np.testing.assert_array_equal(problem.vertex.indices, mapped.indices)
    factor = np.sqrt(model.scale.hbar) / model.scale.omega0**2.5
    np.testing.assert_array_equal(
        problem.vertex.coefficients,
        mapped.values * factor * 0.25,
    )
    assert problem.left.temperature == pytest.approx(0.6)
    assert problem.right.temperature == pytest.approx(0.4)
    np.testing.assert_array_equal(problem.left.device_to_lead, model.d10)
    np.testing.assert_array_equal(problem.right.device_to_lead, model.d01)
    np.testing.assert_array_equal(problem.left.device_indices, [0, 1, 2])
    np.testing.assert_array_equal(problem.right.device_indices, [24, 25, 26])
    assert all(
        np.array_equal(block, model.d00) for block in problem.device.onsite_blocks
    )
    assert all(
        np.array_equal(block, model.d01)
        for block in problem.device.coupling_blocks
    )
    assert not problem.device.onsite_blocks[0].flags.writeable
    assert problem.vertex.metadata["cubic_continuation_factor"] == 0.25
    assert problem.vertex.metadata["cubic_continuation_applied_count"] == 1
    assert len(problem.vertex.metadata["material_nq1_assembly_sha256"]) == 64

    repeated = build_material_nq1_problem(
        model,
        mapped,
        grid=grid,
        mean_temperature=0.5,
        temperature_bias=0.2,
        coupling_scale=0.25,
    )
    assert (
        repeated.vertex.vertex_storage_sha256
        == problem.vertex.vertex_storage_sha256
    )


def test_zero_coupling_and_signed_bias_are_explicit_and_deterministic() -> None:
    model = _harmonic_model()
    mapped = _mapped_fc3(model)
    problem = build_material_nq1_problem(
        model,
        mapped,
        grid=SignedFrequencyGrid(step=0.1, n_positive=4),
        mean_temperature=0.5,
        temperature_bias=-0.4,
        coupling_scale=0.0,
    )
    assert problem.left.temperature == pytest.approx(0.3)
    assert problem.right.temperature == pytest.approx(0.7)
    assert problem.vertex.ndof == mapped.ndof
    assert problem.vertex.is_zero
    assert problem.vertex.indices.shape == (0, 3)
    assert problem.vertex.metadata["cubic_continuation_factor"] == 0.0
    assert len(problem.vertex.metadata["uncontinued_vertex_storage_sha256"]) == 64


def test_accepts_nonexact_roundoff_diagnostic_but_rejects_a_failed_bound() -> None:
    model = _harmonic_model()
    mapped = _mapped_fc3(model)
    diagnostic_only = _replace_report(
        mapped,
        exact_permutation_equality=False,
        permutation_max_abs=float(np.spacing(2.0)),
        permutation_max_roundoff_ratio=0.5,
    )
    problem = build_material_nq1_problem(
        model,
        diagnostic_only,
        grid=SignedFrequencyGrid(step=0.1, n_positive=2),
        mean_temperature=0.5,
    )
    assert isinstance(problem.vertex, SparseCubicVertex)

    failed_bound = _replace_report(
        mapped,
        permutation_tolerance_passed=False,
        permutation_max_roundoff_ratio=33.0,
        certification_passed=False,
        failure_codes=("permutation_tolerance",),
    )
    with pytest.raises(MaterialNQ1ProblemError, match="preflight"):
        build_material_nq1_problem(
            model,
            failed_bound,
            grid=SignedFrequencyGrid(step=0.1, n_positive=2),
            mean_temperature=0.5,
        )

    missing_support = _replace_report(
        mapped,
        permutation_missing_count=1,
        permutation_tolerance_passed=False,
        certification_passed=False,
        failure_codes=("permutation_tolerance",),
    )
    with pytest.raises(MaterialNQ1ProblemError, match="preflight"):
        build_material_nq1_problem(
            model,
            missing_support,
            grid=SignedFrequencyGrid(step=0.1, n_positive=2),
            mean_temperature=0.5,
        )


def test_fails_closed_on_report_range_and_cross_stage_identity() -> None:
    model = _harmonic_model()
    mapped = _mapped_fc3(model)
    grid = SignedFrequencyGrid(step=0.1, n_positive=2)
    insufficient = _replace_report(mapped, required_left_halo_layers=3)
    with pytest.raises(MaterialNQ1ProblemError, match="lacks the directional range"):
        build_material_nq1_problem(
            model,
            insufficient,
            grid=grid,
            mean_temperature=0.5,
        )

    mismatched = _replace_report(mapped, fc2_content_sha256="f" * 64)
    with pytest.raises(MaterialNQ1ProblemError, match="FC2 content identity mismatch"):
        build_material_nq1_problem(
            model,
            mismatched,
            grid=grid,
            mean_temperature=0.5,
        )

    failed = _replace_report(
        mapped,
        certification_passed=False,
        failure_codes=("right_harmonic_buffer",),
    )
    with pytest.raises(MaterialNQ1ProblemError, match="complete-range preflight"):
        build_material_nq1_problem(
            model,
            failed,
            grid=grid,
            mean_temperature=0.5,
        )


def test_rejects_invalid_bias_coupling_and_stale_content_hash() -> None:
    model = _harmonic_model()
    mapped = _mapped_fc3(model)
    grid = SignedFrequencyGrid(step=0.1, n_positive=2)
    with pytest.raises(ValueError, match="reservoir temperature negative"):
        build_material_nq1_problem(
            model,
            mapped,
            grid=grid,
            mean_temperature=0.1,
            temperature_bias=0.3,
        )
    with pytest.raises(ValueError, match="coupling_scale must be nonnegative"):
        build_material_nq1_problem(
            model,
            mapped,
            grid=grid,
            mean_temperature=0.1,
            coupling_scale=-0.1,
        )

    object.__setattr__(mapped, "content_sha256", "0" * 64)
    with pytest.raises(MaterialNQ1ProblemError, match="content identity"):
        build_material_nq1_problem(
            model,
            mapped,
            grid=grid,
            mean_temperature=0.1,
        )
