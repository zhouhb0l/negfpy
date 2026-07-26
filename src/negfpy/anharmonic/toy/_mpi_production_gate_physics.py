"""Real-physics bridge for the private protocol-v2 MPI campaign.

Workers call the existing toy-validation functions unchanged. Multidimensional
SCBA state is persisted only through the standard atomic SCBA checkpoint
format; continuation state is reconstructed from that exact parent checkpoint.
Open-system outcomes use explicit JSON plus numeric HDF5 arrays and never
pickle Python objects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import h5py
import numpy as np

from .._provenance import (
    configuration_hash,
    initial_self_energy_hash,
    problem_input_hashes,
)
from ..contacts import build_contact_spectra
from ..solver import (
    _current_report,
    _finalize_result,
    _guard_dense_memory,
    _solve_green,
)
from ..types import KeldyshSpectrum
from ._mpi_production_gate import (
    ProductionToyCaseContext,
    ProductionToyCaseResult,
    ProductionToyMPICampaign,
    ProductionToyMPIExecutionReport,
    _artifact_validator,
    _phase_paths,
    build_production_toy_mpi_campaign,
    collect_production_toy_campaign_outcomes,
    execute_production_toy_mpi_campaign,
)
from .multidimensional_release import (
    MultidimensionalReleaseConfig,
    MultidimensionalReleaseReport,
    MultidimensionalRunOutcome,
    _bounded_backend_validation,
    _checkpoint_target,
    _direct_endpoint_unlocks_graph,
    _execute,
    _expected_outcome_topology,
    _harmonic_same_grid_preflight,
    _solver_config,
    build_multidimensional_problem,
    full_multidimensional_release_config,
    run_multidimensional_release_benchmark,
)
from .open_three_mode import (
    OpenThreeModeFixture,
    OpenThreeModeOracleResult,
)
from .production_gate_report import (
    OpenThreeModeBornRun,
    ProductionToyGateReport,
    _canonical_json as _report_canonical_json,
    _run_document,
    build_production_toy_gate_report,
)
from .production_gate_runner import (
    evaluate_open_three_mode_born,
    open_three_mode_release_fixtures,
)


_OPEN_ARRAY_NAMES = (
    "oracle_green_zero_retarded",
    "oracle_green_zero_lesser",
    "oracle_green_zero_greater",
    "oracle_sigma_pp_retarded",
    "oracle_sigma_pp_lesser",
    "oracle_sigma_pp_greater",
    "oracle_self_energy_on_shell",
    "oracle_frequency_shifts",
    "oracle_linewidths",
    "oracle_grid_frequency_shifts",
    "oracle_grid_linewidths",
    "oracle_integration_error_estimates",
    "production_sigma_pp_retarded",
    "production_sigma_pp_lesser",
    "production_sigma_pp_greater",
)


class ProductionToyPhysicsMergeError(RuntimeError):
    """Raised when terminal campaign state cannot form the strict report."""

    def __init__(
        self,
        message: str,
        execution: ProductionToyMPIExecutionReport,
    ) -> None:
        self.execution = execution
        super().__init__(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_case_artifact(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Read an already task-validator-checked case artifact."""

    with h5py.File(path, "r") as handle:
        evidence_bytes = bytes(
            np.asarray(handle["evidence_json"], dtype=np.uint8).tobytes()
        )
        evidence = json.loads(evidence_bytes.decode("ascii"))
        arrays = {
            str(name): np.asarray(handle["arrays"][name])
            for name in handle["arrays"]
        }
    return evidence, arrays


def _verify_task_inputs(
    task: Any,
    fixture_or_problem: Any,
    config: Any,
) -> None:
    problem = getattr(fixture_or_problem, "problem", fixture_or_problem)
    expected = problem_input_hashes(problem)
    expected["configuration"] = configuration_hash(config)
    for name, digest in expected.items():
        if task.identity_hashes.get(name) != digest:
            raise ValueError(
                f"task {task.task_id!r} has mismatched {name!r} identity."
            )


def _open_arrays(run: OpenThreeModeBornRun) -> dict[str, np.ndarray]:
    return {
        "oracle_green_zero_retarded": run.oracle.green_zero.retarded,
        "oracle_green_zero_lesser": run.oracle.green_zero.lesser,
        "oracle_green_zero_greater": run.oracle.green_zero.greater,
        "oracle_sigma_pp_retarded": run.oracle.sigma_pp.retarded,
        "oracle_sigma_pp_lesser": run.oracle.sigma_pp.lesser,
        "oracle_sigma_pp_greater": run.oracle.sigma_pp.greater,
        "oracle_self_energy_on_shell": run.oracle.self_energy_on_shell,
        "oracle_frequency_shifts": run.oracle.frequency_shifts,
        "oracle_linewidths": run.oracle.linewidths,
        "oracle_grid_frequency_shifts": run.oracle.grid_frequency_shifts,
        "oracle_grid_linewidths": run.oracle.grid_linewidths,
        "oracle_integration_error_estimates": (
            run.oracle.integration_error_estimates
        ),
        "production_sigma_pp_retarded": run.production.sigma_pp.retarded,
        "production_sigma_pp_lesser": run.production.sigma_pp.lesser,
        "production_sigma_pp_greater": run.production.sigma_pp.greater,
    }


class ProductionToyPhysicsExecutor:
    """Callable whole-case worker backed by existing production functions."""

    def __init__(
        self,
        campaign: ProductionToyMPICampaign,
        config: MultidimensionalReleaseConfig,
        open_fixtures: Sequence[tuple[str, OpenThreeModeFixture]],
        campaign_run_directory: str | Path,
        *,
        environment_sha256: str,
    ) -> None:
        if not isinstance(campaign, ProductionToyMPICampaign):
            raise TypeError("campaign must be a ProductionToyMPICampaign.")
        if not isinstance(config, MultidimensionalReleaseConfig):
            raise TypeError("config must be a MultidimensionalReleaseConfig.")
        self.campaign = campaign
        self.config = config
        self.run_directory = Path(campaign_run_directory).expanduser().resolve()
        self.environment_sha256 = environment_sha256
        self.scientific_checkpoint_dir = (
            self.run_directory
            / "scientific-checkpoints"
            / "multidimensional"
        )
        self.scientific_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.open_fixtures = MappingProxyType(dict(open_fixtures))
        if len(self.open_fixtures) != len(tuple(open_fixtures)):
            raise ValueError("open fixture names must be unique.")
        rows = _expected_outcome_topology(config)
        self.rows = MappingProxyType(
            {
                f"multidimensional.{row['run_id']}": row
                for row in rows
            }
        )
        self._restored: dict[str, MultidimensionalRunOutcome] = {}
        expected_open = {
            task.task_id.removeprefix("open_three_mode.")
            for task in campaign.manifest.tasks
            if task.payload.get("kind") == "open_three_mode_case"
        }
        if expected_open != set(self.open_fixtures):
            raise ValueError(
                "physics executor open fixtures do not match the campaign."
            )

    def _preflight(self, task: Any) -> ProductionToyCaseResult:
        kind = task.payload["kind"]
        if kind == "harmonic_preflight":
            if self.config.certification_eligible:
                evidence = _harmonic_same_grid_preflight(
                    self.config,
                    self.environment_sha256,
                )
                evidence["applicable"] = True
            else:
                evidence = {
                    "schema": (
                        "negfpy.anharmonic.harmonic_same_grid_preflight"
                    ),
                    "schema_version": 2,
                    "protocol": "multidimensional_toy_gate_v2",
                    "applicable": False,
                    "passed": True,
                    "failures": [],
                }
        elif kind == "backend_preflight":
            if self.config.run_bounded_backend_validation:
                evidence = _bounded_backend_validation(
                    self.config,
                    self.environment_sha256,
                )
                evidence["applicable"] = True
            else:
                evidence = {
                    "schema": (
                        "negfpy.anharmonic.bounded_backend_validation"
                    ),
                    "schema_version": 1,
                    "protocol": "multidimensional_toy_gate_v2",
                    "applicable": False,
                    "passed": True,
                    "failures": [],
                }
        else:
            raise ValueError(f"unknown preflight kind {kind!r}.")
        if evidence.get("passed") is True:
            return ProductionToyCaseResult.completed(evidence=evidence)
        return ProductionToyCaseResult.scientific_failure(evidence=evidence)

    def _case_artifact(self, task_id: str) -> Path:
        phase, _, artifact_root, checkpoint_root = _phase_paths(
            self.run_directory,
            self.campaign,
            task_id,
        )
        from .._mpi_tasks import load_task_checkpoint, task_checkpoint_path

        checkpoint = load_task_checkpoint(
            task_checkpoint_path(
                phase,
                task_id,
                checkpoint_root,
            ),
            manifest=phase,
            artifact_root=artifact_root,
            artifact_validator=_artifact_validator(self.campaign),
        )
        if (
            checkpoint.status != "completed"
            or checkpoint.artifact_relative_path is None
        ):
            raise ValueError(
                f"parent campaign task {task_id!r} is not complete."
            )
        return artifact_root / checkpoint.artifact_relative_path

    def _restore_multidimensional(
        self,
        task_id: str,
    ) -> MultidimensionalRunOutcome:
        cached = self._restored.get(task_id)
        if cached is not None:
            return cached
        task = self.campaign.manifest.task(task_id)
        row = self.rows[task_id]
        parent_id = task.payload.get("continuation_parent_task_id")
        parent = (
            None
            if parent_id is None
            else self._restore_multidimensional(str(parent_id))
        )
        evidence, _ = _read_case_artifact(self._case_artifact(task_id))
        checkpoint_evidence = evidence["outcome"]["checkpoint"]
        expected_target = _checkpoint_target(
            self.scientific_checkpoint_dir,
            str(row["run_id"]),
        )
        if checkpoint_evidence.get("enabled") is not True:
            raise ValueError("completed parent lacks a standard SCBA checkpoint.")
        if Path(str(checkpoint_evidence["path"])).resolve() != expected_target:
            raise ValueError("parent standard checkpoint path changed.")
        if (
            not expected_target.is_file()
            or _file_sha256(expected_target)
            != checkpoint_evidence.get("file_sha256")
        ):
            raise ValueError("parent standard checkpoint SHA256 changed.")
        restored = _execute(
            self.config,
            run_id=str(row["run_id"]),
            point=row["point"],
            initialization=row["initialization"],
            backend=row["backend"],
            parent=parent,
            checkpoint_dir=self.scientific_checkpoint_dir,
            environment_sha256=self.environment_sha256,
            resume=True,
        )
        if not restored.successful or not restored.resumed_from_checkpoint:
            raise ValueError("parent was not exactly reconstructed from checkpoint.")
        self._restored[task_id] = restored
        return restored

    def _multidimensional(
        self,
        task: Any,
        context: ProductionToyCaseContext,
    ) -> ProductionToyCaseResult:
        row = self.rows[task.task_id]
        problem = build_multidimensional_problem(
            self.config,
            g=row["point"].g,
            bias=row["point"].bias,
            numerical=row["point"].numerical,
        )
        solver_config = _solver_config(
            self.config,
            row["point"].numerical,
            backend=row["backend"],
        )
        _verify_task_inputs(task, problem, solver_config)
        parent_id = task.payload.get("continuation_parent_task_id")
        parent = None
        if parent_id is not None:
            if context.continuation_parent_task_id != parent_id:
                raise ValueError("continuation parent context changed.")
            parent = self._restore_multidimensional(str(parent_id))
        outcome = _execute(
            self.config,
            run_id=str(row["run_id"]),
            point=row["point"],
            initialization=row["initialization"],
            backend=row["backend"],
            parent=parent,
            checkpoint_dir=self.scientific_checkpoint_dir,
            environment_sha256=self.environment_sha256,
            resume=True,
        )
        evidence = {
            "artifact_kind": "standard_scba_checkpoint_reference",
            "outcome": outcome.as_dict(self.config.spec.mean_temperature),
        }
        if task.task_id == self.campaign.gate_task_id:
            passed = _direct_endpoint_unlocks_graph(self.config, outcome)
        else:
            passed = bool(
                outcome.successful
                and outcome.result is not None
                and outcome.result.diagnostics.passed
            )
        if passed:
            return ProductionToyCaseResult.completed(evidence=evidence)
        return ProductionToyCaseResult.scientific_failure(evidence=evidence)

    def _open(self, task: Any) -> ProductionToyCaseResult:
        name = str(task.payload["name"])
        fixture = self.open_fixtures[name]
        _verify_task_inputs(task, fixture, fixture.config)
        run = evaluate_open_three_mode_born(
            name,
            fixture,
            environment_lock_sha256=self.environment_sha256,
        )
        evidence = {
            "artifact_kind": "open_three_mode_explicit_arrays_v1",
            "name": name,
            "environment_lock_sha256": self.environment_sha256,
            "oracle_wall_seconds": run.oracle_wall_seconds,
            "production_wall_seconds": run.production_wall_seconds,
            "weak_coupling_limit": run.oracle.weak_coupling_limit,
            "frequency_shift_atol": run.oracle.frequency_shift_atol,
            "method_metadata": dict(run.oracle.method_metadata),
            "run_document": json.loads(
                _report_canonical_json(_run_document(run))
            ),
        }
        return ProductionToyCaseResult.completed(
            evidence=evidence,
            arrays=_open_arrays(run),
        )

    def __call__(
        self,
        task: Any,
        context: ProductionToyCaseContext,
    ) -> ProductionToyCaseResult:
        kind = task.payload.get("kind")
        if kind in {"harmonic_preflight", "backend_preflight"}:
            return self._preflight(task)
        if kind in {"direct_zero_endpoint", "multidimensional_case"}:
            return self._multidimensional(task, context)
        if kind == "open_three_mode_case":
            return self._open(task)
        raise ValueError(f"unsupported production toy task kind {kind!r}.")


def reconstruct_open_three_mode_run(
    task: Any,
    fixture: OpenThreeModeFixture,
    artifact_path: str | Path,
) -> OpenThreeModeBornRun:
    """Safely reconstruct one typed open run from explicit HDF5 fields."""

    if not isinstance(fixture, OpenThreeModeFixture):
        raise TypeError("fixture must be an OpenThreeModeFixture.")
    path = Path(artifact_path).expanduser().resolve()
    evidence, arrays = _read_case_artifact(path)
    if set(arrays) != set(_OPEN_ARRAY_NAMES):
        raise ValueError("open-system artifact array coverage is not exact.")
    if evidence.get("artifact_kind") != "open_three_mode_explicit_arrays_v1":
        raise ValueError("unsupported open-system artifact kind.")
    if evidence.get("name") != task.payload.get("name"):
        raise ValueError("open-system artifact name changed.")
    _verify_task_inputs(task, fixture, fixture.config)

    oracle = OpenThreeModeOracleResult(
        fixture=fixture,
        green_zero=KeldyshSpectrum(
            retarded=arrays["oracle_green_zero_retarded"],
            lesser=arrays["oracle_green_zero_lesser"],
            greater=arrays["oracle_green_zero_greater"],
        ),
        sigma_pp=KeldyshSpectrum(
            retarded=arrays["oracle_sigma_pp_retarded"],
            lesser=arrays["oracle_sigma_pp_lesser"],
            greater=arrays["oracle_sigma_pp_greater"],
        ),
        self_energy_on_shell=arrays["oracle_self_energy_on_shell"],
        frequency_shifts=arrays["oracle_frequency_shifts"],
        linewidths=arrays["oracle_linewidths"],
        grid_frequency_shifts=arrays["oracle_grid_frequency_shifts"],
        grid_linewidths=arrays["oracle_grid_linewidths"],
        integration_error_estimates=(
            arrays["oracle_integration_error_estimates"]
        ),
        weak_coupling_limit=float(evidence["weak_coupling_limit"]),
        frequency_shift_atol=float(evidence["frequency_shift_atol"]),
        method_metadata=evidence["method_metadata"],
    )

    problem = fixture.problem
    config = fixture.config
    predicted = _guard_dense_memory(problem, config)
    left, right = build_contact_spectra(
        problem,
        config.eta_lead,
        surface_gf_method=config.surface_gf_method,
    )
    green, sigma_pp = _solve_green(
        problem,
        left,
        right,
        arrays["production_sigma_pp_lesser"],
        arrays["production_sigma_pp_greater"],
        config,
    )
    for component in ("retarded", "lesser", "greater"):
        recorded = arrays[f"production_sigma_pp_{component}"]
        rebuilt = getattr(sigma_pp, component)
        if not np.array_equal(recorded, rebuilt):
            raise ValueError(
                f"reconstructed production sigma_pp.{component} changed."
            )
    currents = _current_report(problem, left, right, sigma_pp, green)
    production = _finalize_result(
        problem,
        config,
        method="born",
        green=green,
        sigma_pp=sigma_pp,
        left=left,
        right=right,
        currents=currents,
        history=(),
        converged=True,
        initialization="born_contact_only_zero_pp",
        initialization_hash=initial_self_energy_hash(problem, None),
        predicted_memory_bytes=predicted,
    )
    run = OpenThreeModeBornRun(
        name=str(evidence["name"]),
        fixture=fixture,
        oracle=oracle,
        production=production,
        oracle_wall_seconds=float(evidence["oracle_wall_seconds"]),
        production_wall_seconds=float(evidence["production_wall_seconds"]),
        environment_lock_sha256=str(
            evidence["environment_lock_sha256"]
        ),
    )
    rebuilt_document = _run_document(run)
    if _report_canonical_json(rebuilt_document) != _report_canonical_json(
        evidence["run_document"]
    ):
        raise ValueError(
            "typed open-system reconstruction changed its strict run document."
        )
    return run


def attach_mpi_execution_provenance(
    report: ProductionToyGateReport,
    campaign: ProductionToyMPICampaign,
    execution: ProductionToyMPIExecutionReport,
    run_directory: str | Path,
) -> ProductionToyGateReport:
    """Attach verified orchestration identity without changing physics gates."""

    if not isinstance(report, ProductionToyGateReport):
        raise TypeError("report must be a ProductionToyGateReport.")
    root = Path(run_directory).expanduser().resolve()
    execution_path = (
        root / "campaign-executions" / f"{execution.execution_id}.json"
    )
    campaign_path = root / "campaign-manifest.json"
    if not execution_path.is_file() or not campaign_path.is_file():
        raise ValueError("campaign or execution manifest is missing.")
    execution_document = json.loads(execution_path.read_text(encoding="ascii"))
    if execution_document != execution.as_dict():
        raise ValueError("stored campaign execution manifest changed.")
    document = report.as_dict()
    document.pop("report_sha256")
    document["orchestration_provenance"] = {
        "schema": "negfpy.task_level_mpi_provenance",
        "schema_version": 1,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "campaign_manifest_file_sha256": _file_sha256(campaign_path),
        "execution_manifest_file_sha256": _file_sha256(execution_path),
        "execution_payload_sha256": execution.as_dict()["payload_sha256"],
        "execution_scientific_sha256": execution.scientific_sha256,
        "execution_id": execution.execution_id,
        "mode": execution.mode,
        "communicator_size": execution.communicator_size,
        "phase_execution_sha256": list(
            execution.phase_execution_sha256
        ),
        "complete": execution.complete,
        "solver_and_gate_contract_unchanged": True,
    }
    digest = hashlib.sha256(
        _report_canonical_json(document).encode("utf-8")
    ).hexdigest()
    return ProductionToyGateReport(
        document={**document, "report_sha256": digest}
    )


@dataclass(frozen=True, slots=True)
class ProductionToyMPIPhysicsResult:
    campaign: ProductionToyMPICampaign
    execution: ProductionToyMPIExecutionReport
    multidimensional: MultidimensionalReleaseReport
    open_runs: tuple[OpenThreeModeBornRun, ...]
    report: ProductionToyGateReport
    execution_manifest_path: Path
    strict_report_path: Path


def run_production_toy_mpi_physics(
    *,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    run_directory: str | Path,
    config: MultidimensionalReleaseConfig | None = None,
    open_fixtures: Sequence[
        tuple[str, OpenThreeModeFixture]
    ] | None = None,
    validation_tier: Literal["pull_request", "release"] | None = None,
    comm: Any | None = None,
    require_mpi: bool = False,
    thread_limit: int = 1,
    execution_id: str | None = None,
) -> ProductionToyMPIPhysicsResult | None:
    """Run and reconstruct the complete existing production toy gate."""

    selected_config = (
        full_multidimensional_release_config() if config is None else config
    )
    if not isinstance(selected_config, MultidimensionalReleaseConfig):
        raise TypeError("config must be a MultidimensionalReleaseConfig.")
    if open_fixtures is None:
        campaign = build_production_toy_mpi_campaign(
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            config=selected_config,
        )
        fixtures = open_three_mode_release_fixtures()
    else:
        fixtures = tuple(open_fixtures)
        campaign = build_production_toy_mpi_campaign(
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            config=selected_config,
            open_fixtures=fixtures,
        )
    tier = (
        "release"
        if selected_config.certification_eligible
        else "pull_request"
    ) if validation_tier is None else validation_tier
    if tier not in {"pull_request", "release"}:
        raise ValueError("validation_tier must be pull_request or release.")
    root = Path(run_directory).expanduser().resolve()
    executor = ProductionToyPhysicsExecutor(
        campaign,
        selected_config,
        fixtures,
        root,
        environment_sha256=environment_sha256,
    )
    execution = execute_production_toy_mpi_campaign(
        campaign,
        executor,
        root,
        comm=comm,
        require_mpi=require_mpi,
        thread_limit=thread_limit,
        execution_id=execution_id,
    )
    if execution is None:
        return None
    incomplete = [
        outcome.task_id
        for outcome in execution.outcomes
        if outcome.status != "completed"
    ]
    if incomplete:
        raise ProductionToyPhysicsMergeError(
            "strict physics report merge is blocked by terminal tasks: "
            + ", ".join(incomplete),
            execution,
        )

    multidimensional = run_multidimensional_release_benchmark(
        selected_config,
        checkpoint_dir=executor.scientific_checkpoint_dir,
        environment_sha256=environment_sha256,
        resume=True,
    )
    outcomes = collect_production_toy_campaign_outcomes(campaign, root)
    outcome_by_id = {outcome.task_id: outcome for outcome in outcomes}
    fixtures_by_name = dict(fixtures)
    open_runs: list[OpenThreeModeBornRun] = []
    for task in campaign.manifest.tasks:
        if task.payload.get("kind") != "open_three_mode_case":
            continue
        outcome = outcome_by_id[task.task_id]
        if outcome.artifact_relative_path is None:
            raise ValueError("completed open task lacks its HDF5 artifact.")
        name = str(task.payload["name"])
        open_runs.append(
            reconstruct_open_three_mode_run(
                task,
                fixtures_by_name[name],
                root / outcome.artifact_relative_path,
            )
        )
    base_report = build_production_toy_gate_report(
        tuple(open_runs),
        multidimensional,
        validation_tier=tier,
    )
    strict_report = attach_mpi_execution_provenance(
        base_report,
        campaign,
        execution,
        root,
    )
    report_path = (
        root / "production-reports" / f"{execution.execution_id}.json"
    )
    strict_report.write_json(report_path)
    execution_path = (
        root / "campaign-executions" / f"{execution.execution_id}.json"
    )
    return ProductionToyMPIPhysicsResult(
        campaign=campaign,
        execution=execution,
        multidimensional=multidimensional,
        open_runs=tuple(open_runs),
        report=strict_report,
        execution_manifest_path=execution_path,
        strict_report_path=report_path,
    )


__all__ = [
    "ProductionToyMPIPhysicsResult",
    "ProductionToyPhysicsExecutor",
    "ProductionToyPhysicsMergeError",
    "attach_mpi_execution_provenance",
    "reconstruct_open_three_mode_run",
    "run_production_toy_mpi_physics",
]
