from __future__ import annotations

from dataclasses import replace

import hashlib
import json
from pathlib import Path
import time

import pytest

import negfpy.anharmonic.toy.production_gate_report as production_report
from negfpy.anharmonic import SignedFrequencyGrid, solve_one_shot_born
from negfpy.anharmonic.toy.multidimensional_release import (
    MultidimensionalReleaseReport,
    full_multidimensional_release_config,
    pr_multidimensional_release_config,
)
from negfpy.anharmonic.toy.open_three_mode import (
    OpenThreeModeFixture,
    open_three_mode_oracle,
)
from negfpy.anharmonic.toy.production_gate_report import (
    OpenThreeModeBornRun,
    ProductionToyGateReport,
    build_production_toy_gate_report,
)


def _evaluate_open_run(
    name: str,
    fixture: OpenThreeModeFixture,
    *,
    environment_lock_sha256: str | None = None,
) -> OpenThreeModeBornRun:
    started = time.perf_counter()
    oracle = open_three_mode_oracle(
        fixture,
        integration_epsabs=1.0e-7,
        integration_epsrel=1.0e-5,
        integration_limit=50,
    )
    oracle_seconds = time.perf_counter() - started
    started = time.perf_counter()
    production = solve_one_shot_born(fixture.problem, fixture.config)
    production_seconds = time.perf_counter() - started
    return OpenThreeModeBornRun(
        name=name,
        fixture=fixture,
        oracle=oracle,
        production=production,
        oracle_wall_seconds=oracle_seconds,
        production_wall_seconds=production_seconds,
        environment_lock_sha256=environment_lock_sha256,
    )


@pytest.fixture(scope="module")
def coarse_open_run() -> OpenThreeModeBornRun:
    fixture = OpenThreeModeFixture(
        grid=SignedFrequencyGrid(step=0.25, n_positive=21),
        temperature=0.7,
        g=0.0,
        eta_lead=0.05,
    )
    return _evaluate_open_run("coarse_g0_T0p7", fixture)


def _unexecuted_report(
    *,
    full: bool = False,
    max_iterations: int | None = None,
) -> MultidimensionalReleaseReport:
    config = (
        full_multidimensional_release_config()
        if full
        else pr_multidimensional_release_config()
    )
    if max_iterations is not None:
        config = replace(config, max_iterations=max_iterations)
    return MultidimensionalReleaseReport(
        config=config,
        outcomes=(),
        comparisons=(),
        conservation_checks=(),
        failures=("multidimensional fixture intentionally not executed",),
        certification_failures=("fixture intentionally unexecuted",),
        total_elapsed_seconds=0.0,
        execution_passed=False,
        release_certified=False,
    )


def _unexecuted_pr_report() -> MultidimensionalReleaseReport:
    return _unexecuted_report()


def test_coarse_and_incomplete_evidence_can_never_certify(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    multidimensional = _unexecuted_pr_report()
    pull_request = build_production_toy_gate_report(
        (coarse_open_run,),
        multidimensional,
        validation_tier="pull_request",
    )
    mislabeled_release = build_production_toy_gate_report(
        (coarse_open_run,),
        multidimensional,
        validation_tier="release",
    )

    assert not pull_request.release_certified
    assert not mislabeled_release.release_certified
    assert not pull_request.oracle_gate_passed
    summary = pull_request.as_dict()["summary"]
    assert not summary["full_locked_base_coverage"]
    assert not summary["sign_coverage"]
    assert not summary["scaling_coverage"]
    assert not summary["refinement_coverage"]
    assert not summary["multidimensional_contract_passed"]
    assert not summary["multidimensional_static_configuration_passed"]
    assert not summary["multidimensional_runtime_evidence_passed"]
    assert not summary["multidimensional_release_certified"]
    assert not summary["multidimensional_provenance_passed"]
    assert not summary["multidimensional_weak_coupling_passed"]
    assert not summary["cross_platform_identity_consistent"]
    assert pull_request.as_dict()["multidimensional_scba"]["coarse"]


def test_static_configuration_is_independent_of_runtime_evidence(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    multidimensional = _unexecuted_report(full=True)
    static_configuration, runtime_evidence = (
        production_report._multidimensional_contract_status(
            multidimensional,
            multidimensional.as_dict(),
        )
    )

    assert static_configuration
    assert not runtime_evidence
    assert not production_report._multidimensional_contract_passed(
        multidimensional,
        multidimensional.as_dict(),
    )

    combined = build_production_toy_gate_report(
        (coarse_open_run,),
        multidimensional,
    ).as_dict()
    summary = combined["summary"]
    assert summary["multidimensional_static_configuration_passed"]
    assert not summary["multidimensional_runtime_evidence_passed"]
    assert not summary["multidimensional_contract_passed"]
    assert (
        "multidimensional SCBA configuration does not match the locked "
        "release contract"
        not in summary["failures"]
    )
    assert (
        "multidimensional SCBA runtime evidence does not satisfy the "
        "locked release contract"
        in summary["failures"]
    )


def test_static_configuration_gate_rejects_a_control_mismatch() -> None:
    multidimensional = _unexecuted_report(
        full=True,
        max_iterations=199,
    )
    static_configuration, runtime_evidence = (
        production_report._multidimensional_contract_status(
            multidimensional,
            multidimensional.as_dict(),
        )
    )

    assert not static_configuration
    assert not runtime_evidence


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("fixed_point_current_scope",), "full_conservation_vector"),
        (("fixed_point_current_rtol",), 2.0e-8),
        (("fixed_point_current_atol",), 2.0e-12),
        (("backend_current_scope",), "symmetrized_transport"),
        (("backend_current_rtol",), 2.0e-10),
        (("backend_current_atol",), 2.0e-12),
        (("refinement_current_scope",), "full_conservation_vector"),
        (("refinement_current_rtol",), 1.0e-2),
        (("refinement_current_atol",), 2.0e-12),
        (
            (
                "refinement_field_policy",
                "compute_when_grid_identical",
            ),
            False,
        ),
        (
            (
                "refinement_field_policy",
                "gate_only_when_spacing_cutoff_eta_unchanged",
            ),
            False,
        ),
        (("refinement_field_policy", "unexpected_policy"), True),
        (("current_rtol",), 2.0e-10),
        (("current_atol",), 2.0e-12),
        (("legacy_current_scope",), "symmetrized_transport"),
        (("legacy_current_policy",), "all_comparisons"),
    ),
)
def test_static_configuration_rejects_serialized_policy_mutation(
    path: tuple[str, ...],
    value: object,
) -> None:
    multidimensional = _unexecuted_report(full=True)
    document = json.loads(json.dumps(multidimensional.as_dict()))
    comparison = document["config"]["comparison"]
    target = comparison
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    static_configuration, _ = (
        production_report._multidimensional_contract_status(
            multidimensional,
            document,
        )
    )

    assert not static_configuration


def test_json_is_strict_deterministic_hashed_and_immutable(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    first = build_production_toy_gate_report(
        (coarse_open_run,),
        _unexecuted_pr_report(),
    )
    second = build_production_toy_gate_report(
        (coarse_open_run,),
        _unexecuted_pr_report(),
    )

    assert first.to_json() == second.to_json()
    assert first.sha256 == second.sha256
    document = json.loads(first.to_json())
    digest = document.pop("report_sha256")
    canonical = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert len(digest) == 64
    with pytest.raises(TypeError):
        first.document["schema"] = "mutated"

    forged = first.as_dict()
    forged["summary"]["release_certified"] = True
    with pytest.raises(ValueError, match="report_sha256"):
        ProductionToyGateReport(forged)



def test_explicit_environment_lock_is_validated_and_serialized(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    lock_sha256 = "a" * 64
    locked = replace(
        coarse_open_run,
        environment_lock_sha256=lock_sha256,
    )
    report = build_production_toy_gate_report(
        (locked,),
        _unexecuted_pr_report(),
    ).as_dict()

    assert report["schema_version"] == 4
    assert report["open_three_mode"]["runs"][0][
        "environment_lock_sha256"
    ] == lock_sha256
    with pytest.raises(ValueError, match="environment_lock_sha256"):
        replace(coarse_open_run, environment_lock_sha256="not-a-lock")


def test_environment_lock_summary_requires_every_open_run_and_checkpoint() -> None:
    lock_sha256 = "b" * 64
    multidimensional = {
        "checkpoint_evidence": {
            "passed": True,
            "environment_sha256": lock_sha256,
        }
    }
    matching = production_report._environment_lock_summary(
        {
            "one": {"environment_lock_sha256": lock_sha256},
            "two": {"environment_lock_sha256": lock_sha256},
        },
        multidimensional,
    )
    missing = production_report._environment_lock_summary(
        {
            "one": {"environment_lock_sha256": lock_sha256},
            "two": {"environment_lock_sha256": None},
        },
        multidimensional,
    )
    mismatch = production_report._environment_lock_summary(
        {"one": {"environment_lock_sha256": "c" * 64}},
        multidimensional,
    )

    assert matching["passed"]
    assert matching["open_complete"]
    assert matching["open_consistent"]
    assert matching["open_environment_sha256"] == lock_sha256
    assert not missing["passed"]
    assert not missing["open_complete"]
    assert not mismatch["passed"]


def test_report_json_write_is_atomic(
    coarse_open_run: OpenThreeModeBornRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = build_production_toy_gate_report(
        (coarse_open_run,),
        _unexecuted_pr_report(),
    )
    destination = tmp_path / "nested" / "report.json"
    assert report.write_json(destination) == destination
    assert destination.read_text(encoding="utf-8") == (
        report.to_json(indent=2) + "\n"
    )
    assert not tuple(destination.parent.glob(".*.tmp"))

    destination.write_text("previous\n", encoding="utf-8")

    def fail_replace(source: object, target: object) -> None:
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(production_report.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        report.write_json(destination)
    assert destination.read_text(encoding="utf-8") == "previous\n"
    assert not tuple(destination.parent.glob(".*.tmp"))

def test_report_derives_hashes_poles_and_cost_table_from_objects(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    report = build_production_toy_gate_report(
        (coarse_open_run,),
        _unexecuted_pr_report(),
    ).as_dict()
    run = report["open_three_mode"]["runs"][0]

    assert run["controls"]["g"] == 0.0
    assert run["controls"]["temperature"] == 0.7
    assert run["controls"]["grid_step"] == 0.25
    assert run["controls"]["cutoff"] == 5.25
    assert run["controls"]["eta_lead"] == 0.05
    assert run["oracle"]["parent_frequency_shift"] == 0.0
    assert run["oracle"]["parent_linewidth"] == 0.0
    assert run["oracle"]["method_metadata"]["pole_method"] == (
        "adaptive_eta_free_continuum_on_shell"
    )
    assert len(run["oracle"]["integration_error_estimates"]) == 3
    assert all(
        len(row) == 3
        for row in run["oracle"]["integration_error_estimates"]
    )
    assert run["production_born"]["parent_frequency_shift"] == 0.0
    assert run["production_born"]["parent_linewidth"] == 0.0
    assert run["gates"]["oracle_production_agreement"]["passed"]
    assert all(
        len(value) == 64
        for value in run["oracle"]["raw_array_sha256"]["sigma_pp"].values()
    )
    assert all(
        len(value) == 64
        for value in run["production_born"]["raw_array_sha256"][
            "sigma_pp"
        ].values()
    )
    methods = [row["method"] for row in report["cost_accuracy"]]
    assert methods == [
        "harmonic_negf",
        "one_shot_born",
        "self_consistent_born",
    ]
    born = report["cost_accuracy"][1]
    assert born["measured"]["count"] == 1
    assert not born["conserving"]
    assert born["release_use"] == (
        "screening_only_never_conductance_certified"
    )


def test_run_rejects_oracle_or_production_from_a_different_problem(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    different_fixture = OpenThreeModeFixture(
        grid=SignedFrequencyGrid(step=0.25, n_positive=21),
        temperature=0.7,
        g=0.01,
        eta_lead=0.05,
    )
    different_oracle = open_three_mode_oracle(
        different_fixture,
        integration_epsabs=1.0e-7,
        integration_epsrel=1.0e-5,
        integration_limit=50,
    )

    with pytest.raises(ValueError, match="input hash"):
        OpenThreeModeBornRun(
            name="mismatched_production",
            fixture=different_fixture,
            oracle=different_oracle,
            production=coarse_open_run.production,
            oracle_wall_seconds=0.0,
            production_wall_seconds=0.0,
        )
    with pytest.raises(ValueError, match="exact fixture"):
        OpenThreeModeBornRun(
            name="mismatched_oracle",
            fixture=coarse_open_run.fixture,
            oracle=different_oracle,
            production=coarse_open_run.production,
            oracle_wall_seconds=0.0,
            production_wall_seconds=0.0,
        )


def test_builder_rejects_duplicate_or_invalid_inputs(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    multidimensional = _unexecuted_pr_report()
    with pytest.raises(ValueError, match="unique"):
        build_production_toy_gate_report(
            (coarse_open_run, coarse_open_run),
            multidimensional,
        )
    with pytest.raises(ValueError, match="validation_tier"):
        build_production_toy_gate_report(
            (coarse_open_run,),
            multidimensional,
            validation_tier="nightly",  # type: ignore[arg-type]
        )


def test_full_locked_matrix_and_one_control_refinements_are_recognized() -> None:
    base_grid = SignedFrequencyGrid(step=0.5, n_positive=10)
    runs = [
        _evaluate_open_run(
            f"base_T{temperature:g}_g{g:g}",
            OpenThreeModeFixture(
                grid=base_grid,
                temperature=temperature,
                g=g,
                eta_lead=0.1,
            ),
        )
        for temperature in (0.0, 0.7)
        for g in (-0.02, -0.01, 0.0, 0.01, 0.02)
    ]
    for name, grid, eta in (
        (
            "refine_spacing",
            SignedFrequencyGrid(step=0.25, n_positive=20),
            0.1,
        ),
        (
            "refine_cutoff",
            SignedFrequencyGrid(step=0.5, n_positive=12),
            0.1,
        ),
        (
            "refine_eta",
            SignedFrequencyGrid(step=0.5, n_positive=10),
            0.05,
        ),
    ):
        runs.append(
            _evaluate_open_run(
                name,
                OpenThreeModeFixture(
                    grid=grid,
                    temperature=0.7,
                    g=0.02,
                    eta_lead=eta,
                ),
            )
        )

    report = build_production_toy_gate_report(
        tuple(runs),
        _unexecuted_pr_report(),
        validation_tier="release",
    ).as_dict()
    summary = report["summary"]

    assert summary["full_locked_base_coverage"]
    assert summary["sign_coverage"]
    assert summary["scaling_coverage"]
    assert summary["spontaneous_decay_coverage"]
    assert summary["refinement_coverage"]
    assert set(report["open_three_mode"]["refinements"]) == {
        "spacing",
        "cutoff",
        "eta_lead",
    }
    assert not summary["open_three_mode_agreement_passed"]
    assert not summary["refinement_passed"]
    assert not summary["multidimensional_release_certified"]
    assert not summary["release_certified"]


def test_open_provenance_requires_clean_complete_environment_identity(
    coarse_open_run: OpenThreeModeBornRun,
) -> None:
    metadata = dict(coarse_open_run.production.metadata)
    metadata.update(
        {
            "git_dirty": False,
            "git_revision": "a" * 40,
            "source_tree_sha256": "b" * 64,
            "declared_package_version": "0.1.0",
            "installed_package_version": "0.1.0",
            "package_version_consistent": True,
            "dtype": "complex128",
            "numpy_version": "test-numpy",
            "scipy_version": "test-scipy",
            "python": "test-python",
            "platform": "test-platform",
            "blas_runtime": ("test-blas",),
        }
    )
    provenance, complete = production_report._code_provenance(metadata)

    assert complete
    assert provenance["clean"]
    assert provenance["identity"] == {
        "declared_package_version": "0.1.0",
        "installed_package_version": "0.1.0",
        "git_revision": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "dtype": "complex128",
        "numpy_version": "test-numpy",
        "scipy_version": "test-scipy",
        "python": "test-python",
        "platform": "test-platform",
        "blas_runtime": ["test-blas"],
    }

    metadata["git_dirty"] = True
    _, dirty_complete = production_report._code_provenance(metadata)
    assert not dirty_complete
