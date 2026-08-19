#!/usr/bin/env python3
"""Build the strict release report for the silicon ``N_q=1`` SCBA gate."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import h5py
import numpy as np

import run_material_nq1_release_mpi as mpi_driver
import run_sparse_nq1_phase_c_born_campaign as common
import validate_sparse_nq1_phase_c_scba as case
from negfpy.anharmonic._material_nq1_campaign import silicon_nq1_release_tasks
from negfpy.anharmonic._checkpoint_v2 import make_scba_checkpoint_identity_v2
from negfpy.anharmonic._material_nq1_validation import (
    build_observation_mode_set,
    check_harmonic_embedding_buffer,
    measure_observation_mode_poles,
)
from negfpy.anharmonic._provenance import code_provenance


SCHEMA = "negfpy.material-nq1-release-validation.v2"
FIELD_DATASETS = (
    "green/retarded", "green/lesser", "green/greater",
    "sigma_pp/retarded", "sigma_pp/lesser", "sigma_pp/greater",
)


class MaterialReleaseValidationError(RuntimeError):
    """The campaign cannot be authenticated or evaluated."""


def _strictly_reproduce_continuation(
    *,
    recorded_parent: Mapping[str, Any],
    task: Any,
    authenticated_parent_scba: Path,
    child_inputs: tuple[Mapping[str, Any], Any, Any],
    source_tree_sha256: str,
    environment_sha256: str,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    child_identity, child_problem, child_config = child_inputs
    try:
        derived_initial, derived_parent = case._continuation_initial_state(
            parent_artifacts=authenticated_parent_scba,
            identity=child_identity,
            problem=child_problem,
            config=child_config,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            child_coupling_scale=task.coupling_scale,
            continuation_axis=task.continuation_axis,
            continuation_predictor=task.continuation_predictor,
        )
    except (TypeError, case.SCBACaseError) as error:
        raise MaterialReleaseValidationError(
            f"cannot independently reproduce continuation state: {error}"
        ) from error
    if derived_parent is None or derived_parent != recorded_parent:
        raise MaterialReleaseValidationError(
            "recorded continuation state differs from authenticated parent state"
        )
    if derived_initial is None:
        raise MaterialReleaseValidationError(
            "authenticated continuation did not produce an initial state"
        )
    return (
        case._validate_continuation_initialization(derived_parent),
        derived_initial,
        derived_parent,
    )


def _load_receipts(
    root: Path,
    campaign: Any,
    *,
    authenticated_inputs: Mapping[
        str, tuple[Mapping[str, Any], Any, Any]
    ] | None = None,
    source_tree_sha256: str | None = None,
    environment_sha256: str | None = None,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("waves/*/task-artifacts/*.json")):
        raw = mpi_driver._load_canonical(path, "material task receipt")
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or task_id in records:
            raise MaterialReleaseValidationError("duplicate or unnamed task receipt")
        try:
            spec = campaign.manifest.task(task_id)
        except KeyError as error:
            raise MaterialReleaseValidationError("receipt names an unknown task") from error
        wave_index = campaign.wave_index(task_id)
        phase = importlib.import_module(
            "negfpy.anharmonic._material_nq1_mpi"
        )._phase_manifest(campaign, wave_index)
        expected_name = (
            f"{phase.ordinal(spec):06d}-"
            f"{phase.task_identity_sha256(spec)[:16]}.json"
        )
        expected_path = (
            root / "waves" / f"{wave_index:03d}" / "task-artifacts"
            / expected_name
        )
        if path != expected_path or path.resolve() != expected_path.resolve():
            raise MaterialReleaseValidationError(
                "material task receipt path is not canonical"
            )
        task = campaign.task(task_id)
        parent_scba = None
        child_inputs = None
        if task.continuation_parent is not None:
            if (
                authenticated_inputs is None
                or task_id not in authenticated_inputs
                or source_tree_sha256 is None
                or environment_sha256 is None
                or task.continuation_parent not in records
            ):
                raise MaterialReleaseValidationError(
                    "continuation receipt lacks authenticated derivation context"
                )
            parent_scba = (
                records[task.continuation_parent]["data"] / "scba"
            )
            child_inputs = authenticated_inputs[task_id]
        evidence = mpi_driver._receipt_evidence(
            path,
            spec,
            expected_campaign_manifest_sha256=campaign.manifest.manifest_sha256,
            expected_task_identity_sha256=phase.task_identity_sha256(spec),
            authenticated_parent_scba=parent_scba,
            authenticated_child_inputs=child_inputs,
            expected_source_tree_sha256=source_tree_sha256,
            expected_environment_sha256=environment_sha256,
        )
        data = path.with_suffix(".data")
        scba = data / "scba"
        outcome = mpi_driver._load_canonical(
            scba / "outcomes/scba.json", "SCBA scientific outcome"
        )
        scba_manifest = mpi_driver._load_canonical(
            scba / "campaign-manifest.json", "SCBA campaign manifest"
        )
        fields = scba / "outcomes/scba-fields.h5"
        if mpi_driver._sha256(fields) != outcome.get("hdf5", {}).get("sha256"):
            raise MaterialReleaseValidationError("SCBA field archive hash changed")
        try:
            with h5py.File(fields, "r") as handle:
                metadata_bytes = bytes(
                    np.asarray(handle["metadata_json"][...], dtype=np.uint8)
                )
            archive_metadata = json.loads(metadata_bytes)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise MaterialReleaseValidationError(
                "SCBA field archive metadata is invalid"
            ) from error
        if (
            not isinstance(archive_metadata, dict)
            or metadata_bytes != mpi_driver._canonical(archive_metadata)
        ):
            raise MaterialReleaseValidationError(
                "SCBA field archive metadata is not canonical"
            )
        if task.continuation_parent is not None:
            assert parent_scba is not None
            assert child_inputs is not None
            (
                initialization,
                independently_derived_state,
                independently_derived_parent,
            ) = _strictly_reproduce_continuation(
                recorded_parent=scba_manifest["continuation_parent"],
                task=task,
                authenticated_parent_scba=parent_scba,
                child_inputs=child_inputs,
                source_tree_sha256=source_tree_sha256,
                environment_sha256=environment_sha256,
            )
            child_identity, child_problem, child_config = child_inputs
            child_checkpoint = case.load_scba_checkpoint_v2_self_validating(
                scba / "checkpoint/sigma-pp.h5"
            )
            expected_checkpoint_identity = make_scba_checkpoint_identity_v2(
                task.task_id,
                child_problem,
                child_config,
                environment_sha256=environment_sha256,
                initial_sigma_pp=independently_derived_state,
                source_hashes=case._source_hashes(
                    child_identity, independently_derived_parent
                ),
                source_tree_sha256=source_tree_sha256,
            )
            if (
                child_checkpoint.identity != expected_checkpoint_identity
                or
                evidence["continuation_initialization"] != initialization
                or outcome.get("continuation_parent")
                != scba_manifest["continuation_parent"]
                or archive_metadata.get("continuation_parent")
                != scba_manifest["continuation_parent"]
            ):
                raise MaterialReleaseValidationError(
                    "strict continuation state reproduction disagrees with evidence"
                )
        records[task_id] = {
            "receipt_path": path,
            "data": data,
            "evidence": evidence,
            "outcome": outcome,
            "scba_manifest": scba_manifest,
            "archive_metadata": archive_metadata,
            "fields": fields,
        }
    expected = {task.task_id for task in campaign.scientific_tasks}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise MaterialReleaseValidationError(
            f"campaign receipt set is incomplete (missing={missing}, extra={extra})"
        )
    return records


def _continuation_authentication(
    task: Any,
    record: Mapping[str, Any],
    campaign: Any,
) -> dict[str, Any]:
    expected_parent_id = task.continuation_parent
    expected_axis = task.continuation_axis
    expected_predictor = task.continuation_predictor
    manifest_parent = record["scba_manifest"].get("continuation_parent")
    outcome_parent = record["outcome"].get("continuation_parent")
    archive_parent = record["archive_metadata"].get("continuation_parent")
    receipt_initialization = record["evidence"].get(
        "continuation_initialization"
    )
    if expected_parent_id is None:
        checks = {
            "task_predictor_absent": (
                expected_axis is None and expected_predictor is None
            ),
            "manifest_parent_absent": manifest_parent is None,
            "outcome_parent_absent": outcome_parent is None,
            "archive_parent_absent": archive_parent is None,
            "receipt_initialization_absent": receipt_initialization is None,
        }
        return {
            "parent_task_id": None,
            "axis": None,
            "predictor": None,
            "parent_coupling_scale": None,
            "child_coupling_scale": task.coupling_scale,
            "factor": None,
            "checks": checks,
            "passed": all(checks.values()),
        }

    parent_task = campaign.task(expected_parent_id)
    expected_factor = (
        1.0
        if expected_predictor == "identity"
        else (task.coupling_scale / parent_task.coupling_scale) ** 2
    )
    initialization = (
        manifest_parent.get("initialization")
        if isinstance(manifest_parent, Mapping)
        else None
    )
    checks = {
        "manifest_outcome_equal": manifest_parent == outcome_parent,
        "manifest_archive_equal": manifest_parent == archive_parent,
        "receipt_initialization_equal": (
            initialization == receipt_initialization
        ),
        "parent_task_id": (
            isinstance(manifest_parent, Mapping)
            and manifest_parent.get("run_id") == expected_parent_id
        ),
        "axis": (
            isinstance(manifest_parent, Mapping)
            and manifest_parent.get("continuation_axis") == expected_axis
        ),
        "parent_coupling_scale": (
            isinstance(manifest_parent, Mapping)
            and manifest_parent.get("coupling_scale")
            == parent_task.coupling_scale
        ),
        "predictor": (
            isinstance(initialization, Mapping)
            and initialization.get("predictor") == expected_predictor
        ),
        "initialization_parent_coupling": (
            isinstance(initialization, Mapping)
            and initialization.get("parent_coupling_scale")
            == parent_task.coupling_scale
        ),
        "initialization_child_coupling": (
            isinstance(initialization, Mapping)
            and initialization.get("child_coupling_scale")
            == task.coupling_scale
        ),
        "factor": (
            isinstance(initialization, Mapping)
            and initialization.get("factor") == expected_factor
        ),
    }
    return {
        "parent_task_id": expected_parent_id,
        "axis": expected_axis,
        "predictor": expected_predictor,
        "parent_coupling_scale": parent_task.coupling_scale,
        "child_coupling_scale": task.coupling_scale,
        "factor": expected_factor,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _scaled_change(left: float, right: float, floor: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def _field_comparison(
    first: Path,
    second: Path,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    passed = True
    with h5py.File(first, "r") as left, h5py.File(second, "r") as right:
        for name in FIELD_DATASETS:
            a = left[name]
            b = right[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                datasets[name] = {"shape_equal": False, "passed": False}
                passed = False
                continue
            maximum_ratio = 0.0
            maximum_absolute = 0.0
            for index in range(a.shape[0]):
                av = np.asarray(a[index])
                bv = np.asarray(b[index])
                difference = np.abs(av - bv)
                allowed = atol + rtol * np.maximum(np.abs(av), np.abs(bv))
                maximum_absolute = max(maximum_absolute, float(np.max(difference)))
                maximum_ratio = max(
                    maximum_ratio,
                    float(np.max(difference / allowed)),
                )
            item_passed = maximum_ratio <= 1.0
            datasets[name] = {
                "shape_equal": True,
                "maximum_absolute_error": maximum_absolute,
                "maximum_allowed_ratio": maximum_ratio,
                "passed": item_passed,
            }
            passed = passed and item_passed
    return {
        "rtol": rtol, "atol": atol, "datasets": datasets, "passed": passed
    }


def _transport_current(outcome: Mapping[str, Any]) -> float:
    currents = outcome["result"]["currents"]
    return 0.5 * (float(currents["left"]) - float(currents["right"]))


def _material_preflight_passed(source: Mapping[str, Any]) -> bool:
    try:
        phase_b = source["phase_b"]["archive"]["metadata"]
        checks = phase_b["checks"]
        reports = phase_b["reports"]
        dimensions = source["dimensions"]
        memory = source["memory"]
        schedule = source["sparse_schedule"]
        return bool(
            source["phase_b"]["phase_b_certified"] is True
            and phase_b["phase_b_certified"] is True
            and checks and all(value is True for value in checks.values())
            and reports["raw_fc3_validation"]["errors"] == []
            and reports["raw_fc3_validation"]["permutation_valid"] is True
            and reports["device_mapping"]["certification_passed"] is True
            and reports["acoustic_lead_classification"]["passed"] is True
            and dimensions["ndof"] == 216
            and dimensions["nnz"] == 406080
            and memory["guard_passed"] is True
            and schedule["active_dofs"] > 0
            and schedule["fft_padding"] >= 2 * dimensions["n_frequency"] - 1
        )
    except (KeyError, TypeError, ValueError):
        return False


def _pole_records(
    fields: Path,
    mode_set: Any,
) -> tuple[dict[str, Any], ...]:
    with h5py.File(fields, "r") as handle:
        measurements = measure_observation_mode_poles(
            np.asarray(handle["grid/omega"]),
            np.asarray(handle["sigma_pp/retarded"]),
            mode_set,
        )
    return tuple(dict(item.as_dict()) for item in measurements)


def _flatten_poles(records: tuple[dict[str, Any], ...], name: str) -> np.ndarray:
    return np.asarray(
        [value for item in records for value in item[name]], dtype=np.float64
    )


def _pole_comparison(
    first: tuple[dict[str, Any], ...],
    second: tuple[dict[str, Any], ...],
    *,
    rtol: float,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    passed = True
    for name in ("linewidths", "shifts"):
        left = _flatten_poles(first, name)
        right = _flatten_poles(second, name)
        if left.shape != right.shape or left.size == 0:
            result[name] = {"passed": False, "maximum_change": "Infinity"}
            passed = False
            continue
        changes = np.abs(left - right) / np.maximum.reduce((
            np.abs(left), np.abs(right), np.full(left.shape, atol)
        ))
        maximum = float(np.max(changes))
        result[name] = {"maximum_change": maximum, "passed": maximum <= rtol}
        passed = passed and maximum <= rtol
    return {"rtol": rtol, "atol": atol, "observables": result, "passed": passed}


def validate(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.expanduser().resolve(strict=True)
    root = mpi_driver._absolute_nonsymlink_path(
        args.campaign, label="material release campaign", directory=True
    )
    source = common._source_identity(repo, args.expected_commit)
    lock = common._validate_lock(args.environment_lock)
    provenance = code_provenance()
    source_tree = provenance.get("source_tree_sha256")
    if not isinstance(source_tree, str) or len(source_tree) != 64:
        raise MaterialReleaseValidationError("source tree SHA256 is unavailable")
    base_task = next(
        item for item in silicon_nq1_release_tasks()
        if item.task_id == "coupling-g100"
    )
    identity, problem, _config = mpi_driver._authenticate_task_inputs(
        fixture=args.fixture,
        fc3_full=args.fc3_full,
        phase_b_artifacts=args.phase_b_artifacts,
        task=base_task,
    )
    scientific_input_hashes = mpi_driver._pilot_input_hashes(repo, identity)
    release_authorization = mpi_driver._validate_release_authorization(
        repo=repo,
        qualification_path=args.qualification_report,
        pilot_audit_path=args.pilot_audit,
        source=source,
        source_tree_sha256=source_tree,
        environment_sha256=lock["sha256"],
        scientific_input_hashes=scientific_input_hashes,
    )
    qualification = release_authorization["qualification"]
    mode_set = build_observation_mode_set(
        problem,
        observation_layers=(4,),
        grid_step=base_task.grid_step,
        eta_lead=base_task.eta_lead,
    )
    modes_path = root / "observation-modes.json"
    archived_modes = mpi_driver._load_canonical(modes_path, "observation modes")
    if archived_modes != mode_set.as_dict():
        raise MaterialReleaseValidationError("observation mode manifest changed")
    buffer_path = root / "harmonic-buffer-preflight.json"
    archived_buffer = mpi_driver._load_canonical(
        buffer_path, "harmonic buffer preflight"
    )
    recomputed_buffer = dict(check_harmonic_embedding_buffer(
        problem,
        frequencies=problem.grid.values[
            (problem.grid.values > 0.0) & (problem.grid.values <= 2.4)
        ],
        eta_lead=1.0e-8,
    ))
    if archived_buffer != recomputed_buffer:
        raise MaterialReleaseValidationError(
            "harmonic embedding-buffer preflight changed"
        )
    input_hashes = dict(scientific_input_hashes) | {
        "observation_modes": mpi_driver._sha256(modes_path),
        "harmonic_buffer_preflight": mpi_driver._sha256(buffer_path),
        "candidate_qualification": qualification["sha256"],
        "predictor_pilot_manifest": release_authorization[
            "pilot_manifest"
        ]["sha256"],
        "predictor_pilot_report": release_authorization[
            "pilot_report"
        ]["sha256"],
        "predictor_pilot_audit": release_authorization[
            "pilot_audit"
        ]["sha256"],
    }
    campaign = mpi_driver.build_material_nq1_mpi_campaign(
        source_revision=source["commit"],
        source_tree_sha256=source_tree,
        environment_sha256=lock["sha256"],
        input_hashes=input_hashes,
    )
    manifest_path = root / "campaign-manifest.json"
    loaded_manifest = importlib.import_module(
        "negfpy.anharmonic._mpi_tasks"
    ).load_task_manifest(manifest_path)
    if loaded_manifest.manifest_sha256 != campaign.manifest.manifest_sha256:
        raise MaterialReleaseValidationError("campaign manifest does not reproduce")
    authenticated_inputs: dict[str, tuple[Any, Any, Any]] = {
        base_task.task_id: (identity, problem, _config)
    }
    for task in campaign.scientific_tasks:
        if (
            task.continuation_parent is not None
            and task.task_id not in authenticated_inputs
        ):
            authenticated_inputs[task.task_id] = (
                mpi_driver._authenticate_task_inputs(
                    fixture=args.fixture,
                    fc3_full=args.fc3_full,
                    phase_b_artifacts=args.phase_b_artifacts,
                    task=task,
                )
            )
    records = _load_receipts(
        root,
        campaign,
        authenticated_inputs=authenticated_inputs,
        source_tree_sha256=source_tree,
        environment_sha256=lock["sha256"],
    )

    per_run: dict[str, Any] = {}
    poles: dict[str, tuple[dict[str, Any], ...]] = {}
    individual_passed = True
    predictors_passed = True
    provenance_identities: list[dict[str, Any]] = []
    for task in campaign.scientific_tasks:
        outcome = records[task.task_id]["outcome"]
        result = outcome.get("result", {})
        diagnostics = result.get("diagnostics", {})
        metrics = diagnostics.get("metrics", {})
        metadata = result.get("solver_metadata", {})
        continuation = _continuation_authentication(task, records[task.task_id], campaign)
        predictors_passed = predictors_passed and continuation["passed"]
        pole = _pole_records(records[task.task_id]["fields"], mode_set)
        poles[task.task_id] = pole
        current_scale = float(metadata.get("current_scale", float("nan")))
        current_tolerance = max(
            1.0e-10 * current_scale,
            1.0e-5 * max(
                abs(float(result["currents"]["left"])),
                abs(float(result["currents"]["right"])),
            ),
        )
        contact_error = abs(
            float(result["currents"]["left"])
            + float(result["currents"]["right"])
        )
        collision_error = abs(float(result["currents"]["collision_power"]))
        tail = float(metrics.get("sigma_pp_tail_ratio", float("inf")))
        provenance_record = {
            name: metadata.get(name) for name in (
                "git_revision", "git_dirty", "source_tree_sha256", "dtype",
                "backend", "effective_backend", "cubic_kernel",
                "hilbert_backend", "package_version_consistent",
            )
        }
        provenance_identities.append(provenance_record)
        weak_passed = True
        for subspace in pole:
            frequency = float(subspace["frequency"])
            for linewidth in subspace["linewidths"]:
                weak_passed = weak_passed and linewidth >= -1.0e-12 * frequency
                weak_passed = weak_passed and max(linewidth, 0.0) / frequency < 0.1
            weak_passed = weak_passed and all(
                abs(value) / frequency < 0.1 for value in subspace["shifts"]
            )
        run_passed = bool(
            result.get("converged") is True
            and diagnostics.get("passed") is True
            and result.get("scientific_error") is None
            and contact_error <= current_tolerance
            and collision_error <= current_tolerance
            and tail <= 1.0e-6
            and weak_passed
            and metadata.get("git_revision") == source["commit"]
            and metadata.get("git_dirty") is False
            and metadata.get("source_tree_sha256") == source_tree
            and metadata.get("dtype") == "complex128"
            and metadata.get("effective_backend") == "fft"
            and metadata.get("cubic_kernel") == "sparse_zero_padded_fft"
            and metadata.get("hilbert_backend") == "fft"
            and metadata.get("package_version_consistent") is True
            and _material_preflight_passed(outcome.get("preflight_source", {}))
            and continuation["passed"]
        )
        individual_passed = individual_passed and run_passed
        per_run[task.task_id] = {
            "scientific_task_sha256": task.task_sha256,
            "converged": result.get("converged"),
            "diagnostics_passed": diagnostics.get("passed"),
            "transport_current_w": _transport_current(outcome),
            "current_tolerance_w": current_tolerance,
            "contact_balance_w": contact_error,
            "collision_power_abs_w": collision_error,
            "sigma_pp_tail_ratio": tail,
            "weak_coupling_passed": weak_passed,
            "material_preflight_passed": _material_preflight_passed(
                outcome.get("preflight_source", {})
            ),
            "mode_poles": list(pole),
            "provenance": provenance_record,
            "continuation_initialization": continuation,
            "passed": run_passed,
        }

    provenance_consistent = bool(
        provenance_identities
        and all(item == provenance_identities[0] for item in provenance_identities)
    )
    zero_born = _field_comparison(
        records["endpoint-zero-start"]["fields"],
        records["endpoint-born-start"]["fields"],
        rtol=1.0e-8,
        atol=1.0e-12,
    )
    mixing_fields = _field_comparison(
        records["bias-positive-full"]["fields"],
        records["refinement-mixing-010"]["fields"],
        rtol=1.0e-8,
        atol=1.0e-12,
    )

    baseline_name = "bias-positive-full"
    baseline_current = per_run[baseline_name]["transport_current_w"]
    current_floor = 1.0e-10 * float(
        records[baseline_name]["outcome"]["result"]["solver_metadata"]["current_scale"]
    )
    refinements: dict[str, Any] = {}
    refinement_passed = True
    for name in (
        "refinement-half-spacing", "refinement-larger-cutoff",
        "refinement-half-eta", "refinement-mixing-010",
    ):
        current_change = _scaled_change(
            baseline_current, per_run[name]["transport_current_w"], current_floor
        )
        pole_change = _pole_comparison(
            poles[baseline_name], poles[name], rtol=5.0e-3
        )
        passed = current_change <= 5.0e-3 and pole_change["passed"]
        refinement_passed = refinement_passed and passed
        refinements[name] = {
            "current_change": current_change,
            "pole_change": pole_change,
            "passed": passed,
        }

    currents = {
        name: per_run[name]["transport_current_w"] for name in (
            "bias-positive-half", "bias-positive-full",
            "bias-negative-half", "bias-negative-full",
        )
    }
    antisymmetry_full = _scaled_change(
        currents["bias-positive-full"], -currents["bias-negative-full"],
        current_floor,
    )
    antisymmetry_half = _scaled_change(
        currents["bias-positive-half"], -currents["bias-negative-half"],
        current_floor,
    )
    conductance_positive = _scaled_change(
        currents["bias-positive-full"] / 20.0,
        currents["bias-positive-half"] / 10.0,
        current_floor / 10.0,
    )
    conductance_negative = _scaled_change(
        currents["bias-negative-full"] / -20.0,
        currents["bias-negative-half"] / -10.0,
        current_floor / 10.0,
    )
    bias_passed = bool(
        currents["bias-positive-full"] > 0.0
        and currents["bias-positive-half"] > 0.0
        and currents["bias-negative-full"] < 0.0
        and currents["bias-negative-half"] < 0.0
        and max(
            antisymmetry_full, antisymmetry_half,
            conductance_positive, conductance_negative,
        ) <= 5.0e-3
    )

    gates = {
        "all_tasks_individually_passed": individual_passed,
        "provenance_consistent": provenance_consistent,
        "zero_vs_born_fields": zero_born["passed"],
        "mixing_fields": mixing_fields["passed"],
        "directional_refinements": refinement_passed,
        "bias_reversal_and_linear_conductance": bias_passed,
        "resolved_observation_modes_nonempty": bool(mode_set.subspaces),
        "harmonic_embedding_buffer": archived_buffer.get("passed") is True,
        "continuation_predictors_authenticated": predictors_passed,
        "predictor_pilot_release_authorization": (
            release_authorization["passed"] is True
            and release_authorization["identical_controls"] is True
        ),
        "nq1_scope_only": True,
    }
    release_certified = all(gates.values())
    return {
        "schema": SCHEMA,
        "scientific_scope": "silicon N_q=1 algebraic material validation",
        "source": source,
        "source_tree_sha256": source_tree,
        "environment_lock": lock,
        "candidate_qualification": qualification,
        "release_authorization": release_authorization,
        "campaign_manifest_sha256": campaign.manifest.manifest_sha256,
        "observation_modes": archived_modes,
        "harmonic_buffer_preflight": archived_buffer,
        "per_run": per_run,
        "comparisons": {
            "zero_vs_born_fields": zero_born,
            "mixing_fields": mixing_fields,
            "directional_refinements": refinements,
            "bias": {
                "transport_currents_w": currents,
                "full_antisymmetry_change": antisymmetry_full,
                "half_antisymmetry_change": antisymmetry_half,
                "positive_conductance_change": conductance_positive,
                "negative_conductance_change": conductance_negative,
                "passed": bias_passed,
            },
        },
        "gates": gates,
        "excluded_claims": [
            "transverse-momentum convergence",
            "bulk-film prediction",
            "anharmonic reservoirs",
        ],
        "release_certified": release_certified,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--phase-b-artifacts", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate(args)
        mpi_driver._atomic_json(args.output, report)
    except Exception as error:
        print(
            f"material Nq=1 release validation error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["release_certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
