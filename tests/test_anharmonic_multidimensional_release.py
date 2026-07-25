from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import negfpy.anharmonic.toy.multidimensional_release as multidimensional
from negfpy.anharmonic._checkpoint import CheckpointMismatchError
from negfpy.anharmonic._provenance import (
    configuration_hash,
    initial_self_energy_hash,
    problem_input_hashes,
)
from negfpy.anharmonic.exceptions import SCBAConvergenceError
from negfpy.anharmonic.toy.multidimensional_release import (
    MultidimensionalLadderSpec,
    MultidimensionalReleaseReport,
    build_multidimensional_problem,
    full_multidimensional_release_config,
    pr_multidimensional_release_config,
    run_multidimensional_release_benchmark,
)


@pytest.fixture(scope="module")
def coarse_report() -> MultidimensionalReleaseReport:
    """Run the complete bounded graph once for all integration assertions."""

    return run_multidimensional_release_benchmark(
        pr_multidimensional_release_config()
    )



def _result_for_solver_inputs(
    source: object,
    problem: object,
    solver_config: object,
    initial_sigma_pp: object,
) -> object:
    """Relabel a converged coarse result for fast runner-I/O tests."""

    metadata = dict(source.metadata)
    input_hashes = dict(metadata["input_hashes"])
    input_hashes.update(problem_input_hashes(problem))
    input_hashes["configuration"] = configuration_hash(solver_config)
    input_hashes["initial_state"] = initial_self_energy_hash(
        problem,
        initial_sigma_pp,
    )
    metadata["input_hashes"] = input_hashes
    metadata["effective_backend"] = solver_config.backend
    return replace(source, metadata=metadata)

def test_preregistration_preserves_exact_release_ladders_and_directions() -> None:
    spec = MultidimensionalLadderSpec()
    full = full_multidimensional_release_config()
    coarse = pr_multidimensional_release_config()

    assert spec.polarization_frequencies == (1.5, 0.9, 0.6)
    assert spec.masses == (1.0, 1.0, 1.0)
    assert spec.springs == (0.1, 0.1, 0.1)
    assert spec.pinnings == (2.05, 0.61, 0.16)
    assert spec.g_values == (0.0, 0.025, 0.05)
    assert spec.mean_temperature == 0.7
    assert spec.bias_values == (0.0, 0.1, 0.2)
    assert spec.spacings == (0.0015625, 0.00078125, 0.000390625)
    assert spec.cutoffs == (5.0, 6.0)
    assert spec.etas == (0.003125, 0.0015625, 0.00078125)
    assert spec.mixings == (0.2, 0.1)

    assert full.numerical.spacing == 0.00078125
    assert full.numerical.cutoff == 5.0
    assert full.numerical.eta_lead == 0.0015625
    assert full.numerical.mixing == 0.2
    assert full.field_rtol == 1.0e-8
    assert full.field_atol == 1.0e-12
    assert full.backend_field_rtol == 1.0e-10
    assert full.backend_field_atol == 1.0e-12
    assert full.line_shape_rtol == 5.0e-3
    assert full.shift_atol == 1.0e-8
    assert full.certification_eligible
    assert full.run_preregistered_refinements

    refinement_labels = {
        label
        for label, _ in multidimensional._refinement_points(full)
    }
    assert refinement_labels == {
        "spacing_0p0015625",
        "spacing_0p000390625",
        "cutoff_6",
        "eta_lead_0p003125",
        "eta_lead_0p00078125",
        "mixing_0p1",
    }
    assert multidimensional._required_refinement_comparison_names(full) == {
        "half_spacing": "numerical_refinement_spacing_0p000390625",
        "larger_cutoff": "numerical_refinement_cutoff_6",
        "half_eta": "numerical_refinement_eta_lead_0p00078125",
        "changed_mixing": "numerical_refinement_mixing_0p1",
    }

    assert coarse.coarse
    assert not coarse.certification_eligible
    assert not coarse.run_preregistered_refinements
    with pytest.raises(FrozenInstanceError):
        coarse.profile = "full_preregistered"  # type: ignore[misc]
    with pytest.raises(ValueError, match="pinnings must equal"):
        MultidimensionalLadderSpec(pinnings=(2.05, 0.61, 0.17))


def test_problem_is_fixed_three_layer_nine_dof_xyz_fixture() -> None:
    config = pr_multidimensional_release_config()
    problem = build_multidimensional_problem(
        config,
        g=0.025,
        bias=0.1,
    )

    assert problem.device.n_layers == 3
    assert problem.device.dof_per_layer == 3
    assert problem.ndof == 9
    assert problem.grid.step == 0.5
    assert problem.grid.cutoff == 5.0
    assert problem.grid.n_points == 21
    assert problem.left.temperature == pytest.approx(0.75)
    assert problem.right.temperature == pytest.approx(0.65)
    np.testing.assert_allclose(
        problem.left.lead.d00,
        np.diag((1.5**2, 0.9**2, 0.6**2)),
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        problem.left.lead.d01,
        -0.1 * np.eye(3),
        rtol=0.0,
        atol=0.0,
    )

    values = problem.vertex.values
    assert values.shape == (9, 9, 9)
    assert np.count_nonzero(values) == 18
    assert set(values[np.nonzero(values)]) == {0.025}
    assert problem.vertex.metadata["interaction_topology"] == "intralayer"
    assert problem.vertex.metadata["harmonic_bond_current_supported"]


def test_bounded_pr_graph_converges_and_checks_all_required_smoke_paths(
    coarse_report: MultidimensionalReleaseReport,
) -> None:
    report = coarse_report

    assert report.total_elapsed_seconds < 60.0
    assert report.execution_passed
    assert not report.release_certified
    assert report.coarse
    assert not report.failures
    assert len(report.outcomes) == 6
    assert len({item.run_id for item in report.outcomes}) == 6
    assert [item.status for item in report.outcomes] == ["converged"] * 6
    assert [item.initialization for item in report.outcomes] == [
        "zero",  # hardest direct-zero endpoint
        "zero",  # first continuation-path point
        "continuation",
        "continuation",
        "born_seed",
        "zero",
    ]
    assert [item.backend for item in report.outcomes] == [
        "serial",
        "serial",
        "serial",
        "serial",
        "serial",
        "threaded",
    ]
    assert all(
        item.result is not None
        and item.result.metadata["effective_backend"] == item.backend
        for item in report.outcomes
    )

    assert {item.kind for item in report.comparisons} == {
        "zero_vs_born",
        "direct_vs_continuation",
        "serial_vs_threaded",
    }
    assert all(item.passed for item in report.comparisons)
    threaded = next(
        item
        for item in report.comparisons
        if item.kind == "serial_vs_threaded"
    )
    assert threaded.current_max_absolute_error == 0.0
    assert all(
        item.max_absolute_error == 0.0
        for item in threaded.field_errors
    )

    assert [item.equilibrium for item in report.conservation_checks] == [
        True,
        False,
    ]
    assert all(item.passed for item in report.conservation_checks)
    assert any(
        "explicitly non-certifying" in failure
        for failure in report.certification_failures
    )


def test_physical_path_is_explicitly_not_a_cartesian_sweep() -> None:
    config = full_multidimensional_release_config()
    path = config.as_dict()["physical_execution_path"]

    assert path["kind"] == "L_shaped_continuation"
    assert not path["cartesian_sweep"]
    assert path["points"] == [
        {"g": 0.0, "bias": 0.0},
        {"g": 0.025, "bias": 0.0},
        {"g": 0.05, "bias": 0.0},
        {"g": 0.05, "bias": 0.1},
        {"g": 0.05, "bias": 0.2},
    ]
    assert len(path["points"]) == 5
    assert len(path["points"]) != (
        len(config.spec.g_values) * len(config.spec.bias_values)
    )


def test_predeclared_linewidth_and_shift_are_derived_from_center_parent_mode(
    coarse_report: MultidimensionalReleaseReport,
) -> None:
    endpoint = coarse_report.outcomes[0]
    assert endpoint.result is not None
    measurement = multidimensional._line_shape_measurement(
        endpoint,
        coarse_report.config.spec,
    )
    frequency_index = int(
        np.flatnonzero(endpoint.result.grid.values == 1.5)[0]
    )
    on_shell = endpoint.result.sigma_pp.retarded[
        frequency_index,
        3,
        3,
    ]

    assert measurement.dof_label == "center_layer.parent_x"
    assert measurement.dof_index == 3
    assert measurement.target_angular_frequency == 1.5
    assert measurement.sigma_retarded_real == pytest.approx(on_shell.real)
    assert measurement.sigma_retarded_imag == pytest.approx(on_shell.imag)
    assert measurement.linewidth == pytest.approx(-on_shell.imag / 1.5)
    assert measurement.shift == pytest.approx(on_shell.real / 3.0)

    comparison = multidimensional._compare_outcomes(
        "self_refinement",
        "numerical_refinement",
        endpoint,
        endpoint,
        field_rtol=1.0e-10,
        field_atol=1.0e-12,
        current_rtol=5.0e-3,
        current_atol=1.0e-12,
        compare_fields=True,
        line_shape_spec=coarse_report.config.spec,
        line_shape_rtol=5.0e-3,
        linewidth_atol=1.0e-12,
        shift_atol=1.0e-8,
    )
    assert comparison.passed
    assert comparison.line_shape is not None
    assert comparison.line_shape.passed
    assert comparison.line_shape.linewidth_error.max_absolute_error == 0.0
    assert comparison.line_shape.shift_error.max_absolute_error == 0.0
    assert "linewidth=-Im" in comparison.line_shape.definition


def test_report_is_strict_json_and_has_a_verified_digest(
    coarse_report: MultidimensionalReleaseReport,
) -> None:
    serialized = coarse_report.to_json()
    assert serialized == coarse_report.to_json()
    document = json.loads(serialized)
    assert document == coarse_report.as_dict()
    assert document["schema"] == (
        "negfpy.anharmonic.multidimensional_dense_toy_release"
    )
    assert document["schema_version"] == 4
    assert not document["config"]["physical_execution_path"][
        "cartesian_sweep"
    ]
    assert all(
        not item["retained_partial"] for item in document["outcomes"]
    )

    core = dict(document)
    digest = core.pop("report_sha256")
    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert digest == coarse_report.report_sha256
    assert len(digest) == 64


def test_every_requested_failure_is_retained_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def planned_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("planned deterministic failure")

    monkeypatch.setattr(multidimensional, "solve_scba", planned_failure)
    monkeypatch.setattr(
        multidimensional,
        "solve_one_shot_born",
        planned_failure,
    )
    report = run_multidimensional_release_benchmark(
        pr_multidimensional_release_config()
    )

    assert len(report.outcomes) == 6
    assert [item.status for item in report.outcomes] == [
        "failed",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
    ]
    assert all(item.result is None for item in report.outcomes)
    assert not report.execution_passed
    assert not report.release_certified
    assert len(report.failures) >= len(report.outcomes)
    assert any(
        item.error_type == "RuntimeError" for item in report.outcomes
    )
    assert all(
        item.elapsed_seconds >= 0.0 for item in report.outcomes
    )


def test_nonconverged_partial_result_is_retained_and_not_propagated(
    coarse_report: MultidimensionalReleaseReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = coarse_report.outcomes[0].result
    assert source is not None
    partial = replace(
        source,
        converged=False,
        release_certified=False,
    )

    def planned_nonconvergence(*args: object, **kwargs: object) -> object:
        raise SCBAConvergenceError("planned nonconvergence", partial)

    monkeypatch.setattr(
        multidimensional,
        "solve_scba",
        planned_nonconvergence,
    )
    monkeypatch.setattr(
        multidimensional,
        "solve_one_shot_born",
        lambda *args, **kwargs: source,
    )
    report = run_multidimensional_release_benchmark(
        pr_multidimensional_release_config()
    )

    assert [item.status for item in report.outcomes] == [
        "nonconverged_partial",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
    ]
    partials = [
        item for item in report.outcomes if item.retained_partial
    ]
    assert len(partials) == 1
    assert all(item.result is partial for item in partials)
    assert all(
        item.result is None
        for item in report.outcomes
        if item.status == "blocked_by_parent"
    )
    document = report.as_dict()
    assert sum(
        bool(item["retained_partial"]) for item in document["outcomes"]
    ) == 1
    assert not report.execution_passed
    assert not report.release_certified


def test_release_refinement_roles_archive_six_but_certify_only_four() -> None:
    config = full_multidimensional_release_config()
    all_names = {
        f"numerical_refinement_{label}"
        for label, _ in multidimensional._refinement_points(config)
    }
    required_names = {
        name
        for name in multidimensional._required_refinement_comparison_names(
            config
        ).values()
        if name is not None
    }

    assert len(all_names) == 6
    assert required_names == {
        "numerical_refinement_spacing_0p000390625",
        "numerical_refinement_cutoff_6",
        "numerical_refinement_eta_lead_0p00078125",
        "numerical_refinement_mixing_0p1",
    }
    assert all_names - required_names == {
        "numerical_refinement_spacing_0p0015625",
        "numerical_refinement_eta_lead_0p003125",
    }


def test_weak_coupling_and_provenance_are_derived_per_scba_run(
    coarse_report: MultidimensionalReleaseReport,
) -> None:
    document = coarse_report.as_dict()
    weak = document["weak_coupling"]
    provenance = document["provenance"]

    assert weak["limit"] == 0.1
    assert weak["coverage"]
    assert weak["passed"]
    assert len(weak["runs"]) == len(coarse_report.outcomes)
    assert all(item["linewidth_ratio"] < 0.1 for item in weak["runs"])
    assert all(item["shift_ratio"] < 0.1 for item in weak["runs"])

    assert provenance["result_coverage"]
    assert provenance["consistent"]
    assert len(provenance["runs"]) == len(coarse_report.outcomes)
    assert provenance["passed"] == bool(
        provenance["complete"]
        and provenance["clean"]
        and provenance["consistent"]
    )
    identity = provenance["identity"]
    assert identity["dtype"] == "complex128"
    assert identity["git_revision"]
    assert identity["source_tree_sha256"]
    assert identity["numpy_version"]
    assert identity["scipy_version"]
    assert identity["python"]
    assert identity["platform"]
    assert identity["blas_runtime"] is not None


def test_multidimensional_provenance_detects_environment_mismatch(
    coarse_report: MultidimensionalReleaseReport,
) -> None:
    outcomes = list(coarse_report.outcomes)
    source = outcomes[-1]
    assert source.result is not None
    altered_metadata = dict(source.result.metadata)
    altered_metadata["platform"] = "deliberately-different-platform"
    outcomes[-1] = replace(
        source,
        result=replace(source.result, metadata=altered_metadata),
    )

    provenance = multidimensional._multidimensional_provenance_summary(
        tuple(outcomes)
    )
    assert provenance["result_coverage"]
    assert not provenance["consistent"]
    assert not provenance["passed"]




def test_checkpointing_requires_exact_lock_file_sha256(tmp_path: Path) -> None:
    config = pr_multidimensional_release_config()
    with pytest.raises(ValueError, match="requires environment_sha256"):
        run_multidimensional_release_benchmark(
            config,
            checkpoint_dir=tmp_path / "checkpoints",
        )
    with pytest.raises(ValueError, match="only when checkpoint_dir"):
        run_multidimensional_release_benchmark(
            config,
            environment_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="lowercase 64-character"):
        run_multidimensional_release_benchmark(
            config,
            checkpoint_dir=tmp_path / "checkpoints",
            environment_sha256="not-a-lock-hash",
        )


def test_certifying_full_run_requires_checkpoint_before_any_solver_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_solver(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("solver must not start without checkpointing")

    monkeypatch.setattr(multidimensional, "solve_scba", forbidden_solver)
    with pytest.raises(ValueError, match="certification-eligible.*require"):
        run_multidimensional_release_benchmark(
            full_multidimensional_release_config()
        )
    assert calls == 0


def test_certifying_preflight_failure_blocks_exact_graph_before_scba(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    solver_calls: list[str] = []
    environment_sha256 = "a" * 64

    def forbidden_scba(*args: object, **kwargs: object) -> object:
        solver_calls.append("scba")
        raise AssertionError("SCBA must not start after preflight failure")

    def forbidden_born(*args: object, **kwargs: object) -> object:
        solver_calls.append("born")
        raise AssertionError("Born must not start after preflight failure")

    monkeypatch.setattr(multidimensional, "solve_scba", forbidden_scba)
    monkeypatch.setattr(
        multidimensional,
        "solve_one_shot_born",
        forbidden_born,
    )
    monkeypatch.setattr(
        multidimensional,
        "_harmonic_same_grid_preflight",
        lambda config, environment: {
            "schema": "negfpy.anharmonic.harmonic_same_grid_preflight",
            "schema_version": 2,
            "protocol": "multidimensional_toy_gate_v2",
            "environment_sha256": environment,
            "passed": False,
            "failures": ["planned harmonic preflight failure"],
        },
    )
    monkeypatch.setattr(
        multidimensional,
        "_bounded_backend_validation",
        lambda config, environment: {
            "schema": "negfpy.anharmonic.bounded_backend_validation",
            "schema_version": 1,
            "protocol": "multidimensional_toy_gate_v2",
            "environment_sha256": environment,
            "passed": True,
            "failures": [],
        },
    )

    report = run_multidimensional_release_benchmark(
        full_multidimensional_release_config(),
        checkpoint_dir=tmp_path / "checkpoints",
        environment_sha256=environment_sha256,
    )

    assert solver_calls == []
    assert len(report.outcomes) == 13
    assert report.outcomes[0].status == "failed"
    assert report.outcomes[0].error_type == "ProtocolV2PreflightFailure"
    assert all(
        outcome.status == "blocked_by_parent"
        for outcome in report.outcomes[1:]
    )
    assert all(outcome.backend == "fft" for outcome in report.outcomes)
    assert report.topology_validation["outcomes_exact"]
    assert report.topology_validation["expected_outcome_count"] == 13
    assert report.harmonic_preflight["passed"] is False
    assert report.backend_validation["passed"] is True
    assert report.harmonic_preflight["failures"] == [
        "planned harmonic preflight failure"
    ]
    assert not report.execution_passed
    assert not report.release_certified


def test_direct_gate_is_strict_for_full_but_preserves_coarse_smoke(
    coarse_report: MultidimensionalReleaseReport,
) -> None:
    direct = coarse_report.outcomes[0]
    assert direct.result is not None
    assert not direct.result.release_certified
    assert multidimensional._direct_endpoint_unlocks_graph(
        coarse_report.config,
        direct,
    )

    full = full_multidimensional_release_config()
    assert not multidimensional._direct_endpoint_unlocks_graph(full, direct)
    certified_diagnostics = replace(
        direct.result.diagnostics,
        release_certified=True,
    )
    certified_result = replace(
        direct.result,
        diagnostics=certified_diagnostics,
        release_certified=True,
    )
    certified_direct = replace(direct, result=certified_result)
    assert multidimensional._direct_endpoint_unlocks_graph(
        full,
        certified_direct,
    )

    failed_diagnostics = replace(
        certified_result.diagnostics,
        passed=False,
        release_certified=False,
        failures=("planned diagnostic failure",),
    )
    invalid_direct = replace(
        certified_direct,
        result=replace(
            certified_result,
            diagnostics=failed_diagnostics,
            release_certified=False,
        ),
    )
    assert not multidimensional._direct_endpoint_unlocks_graph(
        full,
        invalid_direct,
    )


def test_full_graph_topology_and_references_are_exact_and_tamper_evident(
    coarse_report: MultidimensionalReleaseReport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = coarse_report.outcomes[0].result
    assert original is not None
    bonds = original.currents.bond_currents
    spatial = original.currents.spatial
    zero_spatial = (
        None
        if spatial is None
        else replace(
            spatial,
            eom_exchange_spectra=np.zeros_like(
                spatial.eom_exchange_spectra
            ),
            collision_power_spectra=np.zeros_like(
                spatial.collision_power_spectra
            ),
            eom_layer_exchange=np.zeros_like(spatial.eom_layer_exchange),
            layer_collision_power=np.zeros_like(
                spatial.layer_collision_power
            ),
            nonlocal_self_energy_eom_cut_correction=np.zeros_like(
                spatial.nonlocal_self_energy_eom_cut_correction
            ),
            corrected_cut_currents=np.zeros_like(
                spatial.corrected_cut_currents
            ),
            continuity_residual=np.zeros_like(spatial.continuity_residual),
        )
    )
    zero_currents = replace(
        original.currents,
        left=0.0,
        right=0.0,
        collision_power=0.0,
        bond_currents=(
            None if bonds is None else np.zeros_like(bonds)
        ),
        spatial=zero_spatial,
    )
    certified_diagnostics = replace(
        original.diagnostics,
        release_certified=True,
    )
    source = replace(
        original,
        currents=zero_currents,
        diagnostics=certified_diagnostics,
        release_certified=True,
    )
    monkeypatch.setattr(
        multidimensional,
        "solve_scba",
        lambda *args, **kwargs: source,
    )
    monkeypatch.setattr(
        multidimensional,
        "solve_one_shot_born",
        lambda *args, **kwargs: source,
    )

    full = full_multidimensional_release_config()
    noncertifying_full = replace(
        full,
        certification_eligible=False,
        run_bounded_backend_validation=False,
    )
    report = run_multidimensional_release_benchmark(noncertifying_full)
    assert len(report.outcomes) == 13
    assert report.outcomes[0].run_id.startswith("direct_zero_")
    assert all(
        outcome.run_id.startswith("path_")
        for outcome in report.outcomes[1:6]
    )
    assert report.outcomes[6].run_id.startswith("born_seed_")
    assert [
        outcome.run_id.rsplit("_", 1)[-1]
        for outcome in report.outcomes[7:]
    ] == [
        "0p0015625",
        "0p000390625",
        "6",
        "0p003125",
        "0p00078125",
        "0p1",
    ]
    assert report.topology_validation["passed"]
    assert not report.checkpoint_evidence["passed"]
    assert report.topology_validation["expected_outcome_count"] == 13
    endpoint_document = report.topology_validation["expected_outcomes"][0]
    assert endpoint_document["point"]["mean_temperature"] == 0.7
    assert endpoint_document["point"]["left_temperature"] == pytest.approx(0.8)
    assert endpoint_document["point"]["right_temperature"] == pytest.approx(0.6)

    locked = multidimensional._multidimensional_topology_summary(
        full,
        report.outcomes,
        report.comparisons,
        report.conservation_checks,
    )
    assert locked["passed"]

    reordered = list(report.outcomes)
    reordered[1], reordered[2] = reordered[2], reordered[1]
    assert not multidimensional._multidimensional_topology_summary(
        full,
        tuple(reordered),
        report.comparisons,
        report.conservation_checks,
    )["passed"]

    invalid_comparisons = list(report.comparisons)
    invalid_comparisons[0] = replace(
        invalid_comparisons[0],
        certification_critical=False,
        passed=False,
    )
    comparison_tamper = multidimensional._multidimensional_topology_summary(
        full,
        report.outcomes,
        tuple(invalid_comparisons),
        report.conservation_checks,
    )
    assert not comparison_tamper["comparisons_exact"]
    assert not comparison_tamper["passed"]

    invalid_conservation = list(report.conservation_checks)
    invalid_conservation[0] = replace(
        invalid_conservation[0],
        run_id="wrong-equilibrium-reference",
    )
    assert not multidimensional._multidimensional_topology_summary(
        full,
        report.outcomes,
        report.comparisons,
        tuple(invalid_conservation),
    )["passed"]
def test_checkpoint_interruption_resumes_without_reiterating_completed_run(
    coarse_report: MultidimensionalReleaseReport,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = coarse_report.outcomes[0].result
    assert source is not None
    checkpoint_dir = tmp_path / "checkpoints"
    first_calls: list[tuple[float, float]] = []

    def interrupt_after_direct(
        problem: object,
        solver_config: object,
        *,
        initial_sigma_pp: object = None,
    ) -> object:
        first_calls.append(
            (problem.vertex.values.max(), problem.left.temperature)
        )
        if len(first_calls) == 2:
            raise KeyboardInterrupt("simulated workstation interruption")
        return _result_for_solver_inputs(
            source,
            problem,
            solver_config,
            initial_sigma_pp,
        )

    monkeypatch.setattr(
        multidimensional,
        "solve_scba",
        interrupt_after_direct,
    )
    with pytest.raises(KeyboardInterrupt, match="workstation interruption"):
        run_multidimensional_release_benchmark(
            pr_multidimensional_release_config(),
            checkpoint_dir=checkpoint_dir,
            environment_sha256="a" * 64,
        )
    assert len(first_calls) == 2
    assert len(tuple(checkpoint_dir.glob("*.h5"))) == 1

    resumed_calls: list[str] = []

    def finish_remaining(
        problem: object,
        solver_config: object,
        *,
        initial_sigma_pp: object = None,
    ) -> object:
        resumed_calls.append(solver_config.backend)
        return _result_for_solver_inputs(
            source,
            problem,
            solver_config,
            initial_sigma_pp,
        )

    monkeypatch.setattr(multidimensional, "solve_scba", finish_remaining)
    monkeypatch.setattr(
        multidimensional,
        "solve_one_shot_born",
        lambda problem, solver_config: _result_for_solver_inputs(
            source,
            problem,
            solver_config,
            None,
        ),
    )
    report = run_multidimensional_release_benchmark(
        pr_multidimensional_release_config(),
        checkpoint_dir=checkpoint_dir,
        environment_sha256="a" * 64,
    )

    assert len(resumed_calls) == 5
    assert report.outcomes[0].resumed_from_checkpoint
    assert report.outcomes[0].reconstruction_seconds > 0.0
    assert all(
        not outcome.resumed_from_checkpoint
        for outcome in report.outcomes[1:]
    )
    assert len(tuple(checkpoint_dir.glob("*.h5"))) == 6
    assert report.outcomes[0].run_id.startswith("direct_zero_")
    assert all(
        outcome.run_id.startswith("path_")
        for outcome in report.outcomes[1:4]
    )
    assert report.outcomes[4].run_id.startswith("born_seed_")
    assert report.outcomes[5].run_id.startswith("threaded_zero_")
    for outcome in report.outcomes:
        checkpoint = outcome.as_dict(
            report.config.spec.mean_temperature
        )["checkpoint"]
        assert checkpoint["enabled"]
        assert len(checkpoint["file_sha256"]) == 64
        assert checkpoint["identity"]["environment_sha256"] == "a" * 64


    assert report.checkpoint_evidence["passed"]
    assert report.checkpoint_evidence["environment_sha256"] == "a" * 64

    first = report.outcomes[0]
    assert first.result is not None
    altered_lesser = first.result.sigma_pp.lesser.copy()
    altered_lesser.flat[0] += 1.0j
    altered_sigma = replace(
        first.result.sigma_pp,
        lesser=altered_lesser,
    )
    state_tampered = list(report.outcomes)
    state_tampered[0] = replace(
        first,
        result=replace(first.result, sigma_pp=altered_sigma),
    )
    state_evidence = multidimensional._multidimensional_checkpoint_summary(
        tuple(state_tampered)
    )
    assert not state_evidence["passed"]
    assert not state_evidence["records"][0]["checkpoint_state_matches"]

    history_tampered = list(report.outcomes)
    history_tampered[0] = replace(
        first,
        result=replace(first.result, history=first.result.history[:-1]),
    )
    history_evidence = multidimensional._multidimensional_checkpoint_summary(
        tuple(history_tampered)
    )
    assert not history_evidence["passed"]
    assert not history_evidence["records"][0]["checkpoint_history_matches"]

def test_checkpoint_rejects_a_stale_lock_file_identity_before_solving(
    coarse_report: MultidimensionalReleaseReport,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = coarse_report.outcomes[0].result
    assert source is not None
    checkpoint_dir = tmp_path / "checkpoints"
    calls = 0

    def stop_after_checkpoint(
        problem: object,
        solver_config: object,
        *,
        initial_sigma_pp: object = None,
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("stop after mandatory endpoint")
        return _result_for_solver_inputs(
            source,
            problem,
            solver_config,
            initial_sigma_pp,
        )

    monkeypatch.setattr(multidimensional, "solve_scba", stop_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        run_multidimensional_release_benchmark(
            pr_multidimensional_release_config(),
            checkpoint_dir=checkpoint_dir,
            environment_sha256="a" * 64,
        )
    calls_before_rejection = calls
    with pytest.raises(CheckpointMismatchError, match="identity does not match"):
        run_multidimensional_release_benchmark(
            pr_multidimensional_release_config(),
            checkpoint_dir=checkpoint_dir,
            environment_sha256="b" * 64,
        )
    assert calls == calls_before_rejection


def test_nonconverged_partial_is_checkpointed_and_reconstructed_exactly(
    coarse_report: MultidimensionalReleaseReport,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = coarse_report.outcomes[0].result
    assert source is not None
    checkpoint_dir = tmp_path / "checkpoints"
    calls = 0

    def return_partial(
        problem: object,
        solver_config: object,
        *,
        initial_sigma_pp: object = None,
    ) -> object:
        nonlocal calls
        calls += 1
        relabelled = _result_for_solver_inputs(
            source,
            problem,
            solver_config,
            initial_sigma_pp,
        )
        partial = replace(
            relabelled,
            converged=False,
            release_certified=False,
        )
        raise SCBAConvergenceError("planned partial", partial)

    monkeypatch.setattr(multidimensional, "solve_scba", return_partial)
    first = run_multidimensional_release_benchmark(
        pr_multidimensional_release_config(),
        checkpoint_dir=checkpoint_dir,
        environment_sha256="c" * 64,
    )
    assert calls == 1
    assert [item.status for item in first.outcomes] == [
        "nonconverged_partial",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
        "blocked_by_parent",
    ]
    assert len(tuple(checkpoint_dir.glob("*.h5"))) == 1

    def forbidden_solve(*args: object, **kwargs: object) -> object:
        raise AssertionError("a completed partial checkpoint was reiterated")

    monkeypatch.setattr(multidimensional, "solve_scba", forbidden_solve)
    resumed = run_multidimensional_release_benchmark(
        pr_multidimensional_release_config(),
        checkpoint_dir=checkpoint_dir,
        environment_sha256="c" * 64,
    )
    assert resumed.outcomes[0].status == "nonconverged_partial"
    assert resumed.outcomes[0].resumed_from_checkpoint
    assert resumed.outcomes[0].result is not None
    np.testing.assert_array_equal(
        resumed.outcomes[0].result.sigma_pp.lesser,
        first.outcomes[0].result.sigma_pp.lesser,
    )
    np.testing.assert_array_equal(
        resumed.outcomes[0].result.sigma_pp.greater,
        first.outcomes[0].result.sigma_pp.greater,
    )
    assert all(
        item.status == "blocked_by_parent"
        for item in resumed.outcomes[1:]
    )
