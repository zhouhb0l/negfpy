#!/usr/bin/env python3
"""Execute the preregistered silicon Nq=1 SCBA graph with task-level MPI."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping

import run_sparse_nq1_phase_c_born_campaign as common
import run_sparse_nq1_phase_c_scba_campaign as single_campaign
import validate_sparse_nq1_phase_c_scba as case
from negfpy.anharmonic._born_initial import (
    load_born_initial_state,
    write_born_initial_state,
)
from negfpy.anharmonic._material_nq1_campaign import (
    MATERIAL_NQ1_ACCEPTANCE,
    MATERIAL_NQ1_EXCLUDED_CLAIMS,
    MATERIAL_NQ1_MIXING_FLOOR,
    MaterialSCBATask,
    silicon_nq1_release_tasks,
)
from negfpy.anharmonic._material_nq1_mpi import (
    build_material_nq1_mpi_campaign,
    execute_material_nq1_mpi_campaign,
)
from negfpy.anharmonic._material_nq1_validation import (
    build_observation_mode_set,
    check_harmonic_embedding_buffer,
)
from negfpy.anharmonic import _material_nq1_pilot as predictor_pilot
from negfpy.anharmonic._checkpoint_v2 import make_scba_checkpoint_identity_v2
from negfpy.anharmonic._mpi_tasks import TaskResult
from negfpy.anharmonic._provenance import code_provenance


RECEIPT_SCHEMA = "negfpy.material-nq1-mpi-task-receipt.v2"
REPORT_SCHEMA = "negfpy.material-nq1-mpi-execution-report.v1"
QUALIFICATION_SCHEMA = "negfpy.material-nq1-candidate-qualification.v2"
PREDICTOR_QUALIFICATION_SCHEMA = (
    "negfpy.material-nq1-candidate-qualification.v3"
)
SCIENTIFIC_CONTROLS_SCHEMA = "negfpy.material-nq1-scientific-controls.v2"
EXPECTED_QUALIFICATION_CHECKS = frozenset({
    "source_clean_and_unchanged",
    "python_310_full_suite",
    "python_312_full_suite",
    "mpi_two_rank_smoke",
    "mpi_four_rank_smoke",
    "material_inputs_authenticated",
    "resolved_mode_set_nonempty",
    "harmonic_embedding_buffer",
    "memory_guard",
    "disk_guard",
    "thread_limits_one",
    "candidate_pythonpath_first",
})
# Compatibility name for external launch wrappers.  The authoritative value
# is serialized into each MaterialSCBATask and its campaign identity.
RELEASE_MIXING_FLOOR = MATERIAL_NQ1_MIXING_FLOOR
PILOT_MANIFEST_NAME = "pilot-manifest.json"
PILOT_REPORT_NAME = "pilot-report.json"
PREDICTOR_ORACLE_PATH = "tests/test_material_nq1_predictor_oracle.py"
PREDICTOR_CORE_SHA256 = {
    "src/negfpy/anharmonic/solver.py": (
        "51ef18cdb65e106c176631347e22bb543002ce9204e1703873d8d0016082c6a6"
    ),
    "src/negfpy/anharmonic/cubic.py": (
        "05105b95eaeca8ded94b1eab61b4a39ad41e85ea75e2593eaabf9f6d1a81bc42"
    ),
    "src/negfpy/anharmonic/_checkpoint_v2.py": (
        "0887298a9aa2021b52843082b51ffc998ba230376af3aae86be382fd97282a94"
    ),
}


class MaterialCampaignError(RuntimeError):
    """Fail-closed material campaign error."""


def _authenticate_task_inputs(
    *,
    fixture: Path,
    fc3_full: Path,
    phase_b_artifacts: Path,
    task: MaterialSCBATask,
) -> tuple[Any, Any, Any]:
    """Authenticate one task using every serialized numerical control."""

    frozen_mean = case.phase_c.TEMPERATURE_K
    if task.mean_temperature_kelvin != frozen_mean:
        raise MaterialCampaignError(
            "task mean temperature differs from the authenticated material input"
        )

    return case.authenticate_scba_inputs(
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
        coupling_scale=task.coupling_scale,
        grid_step=task.grid_step,
        grid_n_positive=task.grid_n_positive,
        eta_lead=task.eta_lead,
        mixing=task.mixing,
        mixing_floor=task.mixing_floor,
        temperature_bias=task.temperature_bias_solver,
    )


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=True,
            separators=(",", ":"), sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _scientific_control_record() -> dict[str, Any]:
    tasks = [
        task.as_dict() | {"task_sha256": task.task_sha256}
        for task in silicon_nq1_release_tasks()
    ]
    payload = {
        "schema": SCIENTIFIC_CONTROLS_SCHEMA,
        "acceptance": dict(MATERIAL_NQ1_ACCEPTANCE),
        "excluded_claims": list(MATERIAL_NQ1_EXCLUDED_CLAIMS),
        "task_count": len(tasks),
        "tasks": tasks,
    }
    return payload | {
        "task_set_sha256": hashlib.sha256(_canonical(payload)).hexdigest()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_records(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise MaterialCampaignError(f"task data directory is invalid: {root}")
    records: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MaterialCampaignError(f"task data contains a symlink: {path}")
        if path.is_file():
            if path.name.startswith(".") and path.name.endswith(".tmp"):
                raise MaterialCampaignError(
                    f"task data contains an incomplete publication: {path}"
                )
            records[path.relative_to(root).as_posix()] = _sha256(path)
    if not records:
        raise MaterialCampaignError("task data directory is empty")
    return records


def _tree_sha256(records: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical(dict(sorted(records.items())))).hexdigest()


def _load_canonical(path: Path, label: str) -> dict[str, Any]:
    selected = _absolute_nonsymlink_path(path, label=label, directory=False)
    try:
        payload = selected.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MaterialCampaignError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or payload != _canonical(value):
        raise MaterialCampaignError(f"{label} is not canonical JSON")
    return value


def _absolute_nonsymlink_path(
    path: Path, *, label: str, directory: bool
) -> Path:
    """Return one lexical absolute path after rejecting symlink ancestry."""

    expanded = path.expanduser()
    absolute = Path(os.path.abspath(expanded))
    cursor = Path(absolute.anchor)
    if absolute == cursor:
        mode = cursor.lstat().st_mode
        expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
        if not expected:
            kind = "directory" if directory else "regular file"
            raise MaterialCampaignError(f"{label} is not a {kind}: {absolute}")
        return absolute
    for index, part in enumerate(absolute.parts[1:], start=1):
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except OSError as error:
            raise MaterialCampaignError(f"{label} is unavailable: {absolute}") from error
        if stat.S_ISLNK(mode):
            raise MaterialCampaignError(
                f"{label} may not traverse a symlink: {cursor}"
            )
        final = index == len(absolute.parts) - 1
        if final:
            expected = stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)
            if not expected:
                kind = "directory" if directory else "regular file"
                raise MaterialCampaignError(f"{label} is not a {kind}: {absolute}")
        elif not stat.S_ISDIR(mode):
            raise MaterialCampaignError(
                f"{label} ancestor is not a directory: {cursor}"
            )
    return absolute


def _qualification_evidence_paths(path: Path) -> tuple[Path, Path]:
    selected = _absolute_nonsymlink_path(
        path, label="predictor qualification", directory=False
    )
    evidence_root = _absolute_nonsymlink_path(
        selected.parent / "qualification-evidence",
        label="predictor qualification evidence root",
        directory=True,
    )
    return selected, evidence_root


def _recursively_validate_predictor_qualification(
    document: Mapping[str, Any], *, repo: Path, evidence_root: Path
) -> dict[str, Any]:
    """Load the qualification wrapper lazily to avoid its release-module cycle."""

    module = importlib.import_module("qualify_material_nq1_predictor_candidate")
    validator = getattr(module, "validate_predictor_qualification_report")
    return validator(document, repo=repo, evidence_root=evidence_root)


def _validate_qualification(
    path: Path,
    *,
    repo: Path,
    pilot_manifest: Mapping[str, Any],
    source: Mapping[str, Any],
    source_tree_sha256: str,
    environment_sha256: str,
) -> dict[str, Any]:
    selected, evidence_root = _qualification_evidence_paths(path)
    mode = selected.lstat().st_mode
    if (
        not stat.S_ISREG(mode)
        or stat.S_ISLNK(mode)
        or stat.S_IMODE(mode) != 0o444
    ):
        raise MaterialCampaignError(
            "predictor qualification must be an immutable mode-0444 regular file"
        )
    document = _load_canonical(selected, "predictor qualification report")
    if document.get("schema") != PREDICTOR_QUALIFICATION_SCHEMA:
        raise MaterialCampaignError(
            "exact v3 predictor qualification does not authorize launch"
        )
    try:
        recursively_authenticated = _recursively_validate_predictor_qualification(
            document, repo=repo, evidence_root=evidence_root
        )
    except Exception as error:
        raise MaterialCampaignError(
            "predictor qualification evidence did not recursively authenticate"
        ) from error
    if recursively_authenticated != document:
        raise MaterialCampaignError(
            "predictor qualification recursive result changed the report"
        )
    evidence = document.get("evidence")
    base_evidence = (
        evidence.get("base_qualification_evidence")
        if isinstance(evidence, Mapping)
        else None
    )
    base_records = (
        base_evidence.get("records")
        if isinstance(base_evidence, Mapping)
        else None
    )
    expected_base_records = {
        "python310_runtime", "python310_pytest",
        "python312_runtime", "python312_pytest",
        "mpi_ranks_2", "mpi_ranks_4",
        "material_input_preflight", "resources",
    }
    core_records = (
        evidence.get("core_files") if isinstance(evidence, Mapping) else None
    )
    oracle = (
        evidence.get("property_oracle")
        if isinstance(evidence, Mapping)
        else None
    )
    core_exact = bool(
        isinstance(core_records, Mapping)
        and set(core_records) == set(PREDICTOR_CORE_SHA256)
        and all(
            isinstance(core_records[name], Mapping)
            and core_records[name].get("expected_sha256") == expected
            and core_records[name].get("observed_sha256") == expected
            and core_records[name].get("passed") is True
            and _sha256(repo / name) == expected
            for name, expected in PREDICTOR_CORE_SHA256.items()
        )
    )
    oracle_path = repo / PREDICTOR_ORACLE_PATH
    oracle_exact = bool(
        isinstance(oracle, Mapping)
        and oracle.get("relative_path") == PREDICTOR_ORACLE_PATH
        and oracle.get("sha256") == _sha256(oracle_path)
        and oracle.get("covered_by")
        == ["python310_full_suite", "python312_full_suite"]
    )
    base_evidence_exact = bool(
        isinstance(base_evidence, Mapping)
        and set(base_evidence)
        == {"qualification_context", "python_versions", "records"}
        and isinstance(base_evidence.get("qualification_context"), Mapping)
        and isinstance(base_evidence.get("python_versions"), Mapping)
        and set(base_evidence["python_versions"]) == {"python310", "python312"}
        and isinstance(base_records, Mapping)
        and set(base_records) == expected_base_records
        and all(isinstance(item, Mapping) for item in base_records.values())
        and base_records["python310_pytest"].get("returncode") == 0
        and base_records["python312_pytest"].get("returncode") == 0
        and base_records["mpi_ranks_2"].get("returncode") == 0
        and base_records["mpi_ranks_4"].get("returncode") == 0
        and base_records["resources"].get("memory_guard_passed") is True
        and base_records["resources"].get("disk_guard_passed") is True
    )
    if (
        document.get("source") != source
        or document.get("source_tree_sha256") != source_tree_sha256
        or document.get("environment_sha256") != environment_sha256
        or not predictor_pilot.qualification_authorizes_pilot(
            pilot_manifest, document
        )
        or not core_exact
        or not oracle_exact
        or not base_evidence_exact
    ):
        raise MaterialCampaignError(
            "exact v3 predictor qualification does not authorize launch"
        )
    return {
        "path": str(selected),
        "sha256": _sha256(selected),
        "checks": dict(document["checks"]),
        "scientific_task_set_sha256": document[
            "scientific_controls"
        ]["task_set_sha256"],
    }


def _pilot_input_hashes(
    repo: Path, identity: Mapping[str, Any]
) -> dict[str, str]:
    return dict(case._source_hashes(identity)) | {
        "scba_case_driver": predictor_pilot.sha256_file(
            repo / "scripts/validate_sparse_nq1_phase_c_scba.py"
        )
    }


def _validate_release_authorization(
    *,
    repo: Path,
    qualification_path: Path,
    pilot_audit_path: Path,
    source: Mapping[str, Any],
    source_tree_sha256: str,
    environment_sha256: str,
    scientific_input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Recursively authenticate the exact qualification/pilot release pair."""

    audit_path = pilot_audit_path.expanduser()
    root = audit_path.parent.absolute()
    manifest_path = root / PILOT_MANIFEST_NAME
    try:
        audit_mode = audit_path.lstat().st_mode
        manifest_mode = manifest_path.lstat().st_mode
        if (
            not stat.S_ISREG(audit_mode)
            or stat.S_ISLNK(audit_mode)
            or stat.S_IMODE(audit_mode) != 0o444
            or not stat.S_ISREG(manifest_mode)
            or stat.S_ISLNK(manifest_mode)
            or stat.S_IMODE(manifest_mode) != 0o444
        ):
            raise MaterialCampaignError(
                "pilot audit and manifest must be immutable mode-0444 regular files"
            )
        manifest_document = predictor_pilot.load_canonical_json(
            manifest_path, name="predictor pilot manifest"
        )
        manifest = predictor_pilot.validate_pilot_manifest(manifest_document)
        qualification = _validate_qualification(
            qualification_path,
            repo=repo,
            pilot_manifest=manifest,
            source=source,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
        )
        audit = predictor_pilot.validate_audit(
            manifest, root=root, audit_path=audit_path
        )
        report_record = audit.get("report")
        if not isinstance(report_record, Mapping):
            raise MaterialCampaignError("pilot audit does not bind its final report")
        report_path = Path(str(report_record.get("path")))
        if report_path.absolute() != root / PILOT_REPORT_NAME:
            raise MaterialCampaignError(
                "pilot audit does not bind the canonical report in its root"
            )
        report = predictor_pilot.validate_final_report(manifest, report_path)
    except OSError as error:
        raise MaterialCampaignError(
            "predictor pilot audit or manifest is unavailable"
        ) from error
    except predictor_pilot.MaterialNQ1PilotError as error:
        raise MaterialCampaignError(
            f"predictor pilot audit did not authenticate: {error}"
        ) from error

    expected_authorization = {
        "scope": "consideration-and-submission-gating-only",
        "next_campaign": "separately-identified-full-19-task-campaign",
        "required_source_revision": source.get("commit"),
        "required_source_tree_sha256": source_tree_sha256,
        "requires_identical_qualified_commit_and_tree": True,
        "source_or_scientific_control_changes_allowed": False,
    }
    identities_exact = bool(
        manifest.get("source_revision") == source.get("commit")
        and manifest.get("source_git_tree") == source.get("tree")
        and manifest.get("source_package_file") == source.get("package_file")
        and manifest.get("source_tree_sha256") == source_tree_sha256
        and manifest.get("environment_sha256") == environment_sha256
        and manifest.get("input_hashes") == dict(scientific_input_hashes)
        and manifest.get("qualification_sha256") == qualification["sha256"]
    )
    passed = bool(
        identities_exact
        and audit.get("audit_passed") is True
        and audit.get("release_certified") is False
        and report.get("pilot_passed") is True
        and report.get("all_tasks_completed") is True
        and report.get("all_scheduler_jobs_terminal_success") is True
        and report.get("strict_anchor_reproduction") is True
        and report.get("coupling_g0025_completed") is True
        and report.get("authorizes") == expected_authorization
        and report.get("release_certified") is False
        and report_record.get("sha256") == _sha256(report_path)
    )
    if not passed:
        raise MaterialCampaignError(
            "predictor pilot audit does not authorize identical-control release"
        )
    return {
        "qualification": qualification,
        "pilot_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "pilot_sha256": manifest["pilot_sha256"],
        },
        "pilot_report": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "pilot_audit": {
            "path": str(audit_path.absolute()),
            "sha256": _sha256(audit_path),
        },
        "authorization": expected_authorization,
        "scientific_input_hashes": dict(scientific_input_hashes),
        "identical_controls": True,
        "passed": True,
    }


def _continuation_initialization(
    scba_manifest: Mapping[str, Any],
    scientific_task: Mapping[str, Any],
) -> dict[str, Any] | None:
    if scba_manifest.get("schema") != single_campaign.SCHEMA:
        raise MaterialCampaignError("SCBA campaign schema is unsupported")
    expected_parent = scientific_task.get("continuation_parent")
    expected_axis = scientific_task.get("continuation_axis")
    expected_predictor = scientific_task.get("continuation_predictor")
    parent = scba_manifest.get("continuation_parent")
    preflight = scba_manifest.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("continuation_parent") != parent
    ):
        raise MaterialCampaignError(
            "SCBA preflight does not bind continuation provenance"
        )
    if (
        scba_manifest.get("continuation_axis") != expected_axis
        or scba_manifest.get("continuation_predictor") != expected_predictor
    ):
        raise MaterialCampaignError(
            "SCBA continuation controls changed"
        )
    if expected_parent is None:
        if (
            parent is not None
            or expected_axis is not None
            or expected_predictor is not None
        ):
            raise MaterialCampaignError(
                "non-continuation task has continuation provenance"
            )
        return None
    if not isinstance(parent, Mapping) or parent.get("run_id") != expected_parent:
        raise MaterialCampaignError(
            "SCBA continuation provenance names another parent"
        )
    try:
        initialization = case._validate_continuation_initialization(parent)
    except case.SCBACaseError as error:
        raise MaterialCampaignError(str(error)) from error
    if (
        initialization["predictor"] != expected_predictor
        or initialization["child_coupling_scale"]
        != scientific_task.get("coupling_scale")
    ):
        raise MaterialCampaignError(
            "SCBA continuation initialization differs from the scientific task"
        )
    return initialization


def _independently_authenticate_continuation(
    *,
    recorded_parent: Mapping[str, Any],
    scientific_task: Mapping[str, Any],
    authenticated_parent_scba: Path,
    child_identity: Mapping[str, Any],
    child_problem: Any,
    child_config: Any,
    source_tree_sha256: str,
    environment_sha256: str,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    try:
        derived_initial, derived_parent = case._continuation_initial_state(
            parent_artifacts=authenticated_parent_scba,
            identity=child_identity,
            problem=child_problem,
            config=child_config,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            child_coupling_scale=scientific_task["coupling_scale"],
            continuation_axis=scientific_task["continuation_axis"],
            continuation_predictor=scientific_task[
                "continuation_predictor"
            ],
        )
    except (KeyError, TypeError, case.SCBACaseError) as error:
        raise MaterialCampaignError(
            f"cannot independently derive continuation state: {error}"
        ) from error
    if derived_parent is None or derived_parent != recorded_parent:
        raise MaterialCampaignError(
            "recorded continuation state differs from authenticated parent state"
        )
    if derived_initial is None:
        raise MaterialCampaignError(
            "authenticated continuation did not produce an initial state"
        )
    return (
        case._validate_continuation_initialization(derived_parent),
        derived_initial,
        derived_parent,
    )


def _receipt_evidence(
    path: Path,
    spec: Any,
    *,
    expected_campaign_manifest_sha256: str,
    expected_task_identity_sha256: str,
    authenticated_parent_scba: Path | None = None,
    authenticated_child_inputs: tuple[Mapping[str, Any], Any, Any] | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_environment_sha256: str | None = None,
) -> dict[str, Any]:
    receipt = _load_canonical(path, "material task receipt")
    expected = {
        "schema", "campaign_manifest_sha256", "task_id",
        "task_identity_sha256", "scientific_task_sha256", "state",
        "task_directory", "task_tree_records", "task_tree_sha256",
        "scba_status_sha256", "scba_report_sha256", "checkpoint_sha256",
        "born_initial", "continuation_parent_task_id",
        "continuation_initialization", "release_certified",
    }
    if set(receipt) != expected or receipt["schema"] != RECEIPT_SCHEMA:
        raise MaterialCampaignError("material task receipt schema is invalid")
    if receipt["task_id"] != spec.task_id:
        raise MaterialCampaignError("material task receipt names another task")
    if receipt["campaign_manifest_sha256"] != expected_campaign_manifest_sha256:
        raise MaterialCampaignError("material task receipt names another campaign")
    if receipt["task_identity_sha256"] != expected_task_identity_sha256:
        raise MaterialCampaignError("material task receipt identity changed")
    scientific_payload = spec.payload.get("scientific_task")
    if not isinstance(scientific_payload, Mapping):
        raise MaterialCampaignError("MPI task lacks its scientific payload")
    expected_scientific_sha = spec.identity_hashes.get("scientific_task")
    if (
        receipt["scientific_task_sha256"] != expected_scientific_sha
        or scientific_payload.get("task_sha256") != expected_scientific_sha
    ):
        raise MaterialCampaignError("material task scientific identity changed")
    if receipt["release_certified"] is not False:
        raise MaterialCampaignError("an individual task cannot certify the release")
    expected_parent = scientific_payload.get("continuation_parent")
    if receipt["continuation_parent_task_id"] != expected_parent:
        raise MaterialCampaignError("material task continuation parent changed")
    expected_born = scientific_payload.get("initialization") == "born"
    if (receipt["born_initial"] is not None) != expected_born:
        raise MaterialCampaignError("material task Born initialization changed")
    data = (path.parent / receipt["task_directory"]).resolve()
    if data != path.with_suffix(".data").resolve():
        raise MaterialCampaignError("material task directory is not canonical")
    try:
        data.relative_to(path.parent.resolve())
    except ValueError as error:
        raise MaterialCampaignError("material task directory escapes artifact root") from error
    records = _tree_records(data)
    if records != receipt["task_tree_records"] or _tree_sha256(records) != receipt[
        "task_tree_sha256"
    ]:
        raise MaterialCampaignError("material task tree failed its hash manifest")
    scba = data / "scba"
    status = _load_canonical(scba / "status.json", "SCBA terminal status")
    report = _load_canonical(scba / "scba-report.json", "SCBA terminal report")
    scba_manifest_path = scba / "campaign-manifest.json"
    scba_manifest = _load_canonical(scba_manifest_path, "SCBA campaign manifest")
    checkpoint = scba / "checkpoint" / "sigma-pp.h5"
    scba_manifest_sha = _sha256(scba_manifest_path)
    status_sha = _sha256(scba / "status.json")
    report_sha = _sha256(scba / "scba-report.json")
    checkpoint_sha = _sha256(checkpoint)
    expected_attempt = {
        "path": "logs/scba.json",
        "sha256": records.get("scba/logs/scba.json"),
    }
    forbidden_attempt_records = [
        name
        for name in records
        if "infrastructure-failure" in name
        or re.search(r"(?:^|[-_/])resume(?:[-_.]|$)", name) is not None
    ]
    if (
        status_sha != receipt["scba_status_sha256"]
        or report_sha != receipt["scba_report_sha256"]
        or checkpoint_sha != receipt["checkpoint_sha256"]
    ):
        raise MaterialCampaignError("material task terminal hashes changed")
    if (
        forbidden_attempt_records
        or expected_attempt["sha256"] is None
        or report.get("attempt_logs") != [expected_attempt]
        or report.get("infrastructure_failures") != []
        or report.get("automatic_retry") is not False
        or report.get("controls_changed") is not False
        or status.get("automatic_retry") is not False
        or scba_manifest.get("watchdog", {}).get("automatic_retry") is not False
    ):
        raise MaterialCampaignError(
            "material task is not one immutable initial attempt without "
            "infrastructure history"
        )
    state = receipt["state"]
    if state not in {"passed", "scientific-failure"}:
        raise MaterialCampaignError("material task has a nonterminal receipt state")
    if status.get("state") != state or report.get("state") != state:
        raise MaterialCampaignError("material task terminal states disagree")
    if (
        status.get("campaign_manifest_sha256") != scba_manifest_sha
        or report.get("campaign_manifest_sha256") != scba_manifest_sha
        or status.get("report_sha256") != report_sha
        or report.get("checkpoint_sha256") != checkpoint_sha
        or scba_manifest.get("run_id") != spec.task_id
        or scba_manifest.get("coupling_scale")
        != scientific_payload.get("coupling_scale")
    ):
        raise MaterialCampaignError("SCBA terminal provenance bindings disagree")
    expected_controls = {
        "grid_step": scientific_payload.get("grid_step"),
        "grid_n_positive": scientific_payload.get("grid_n_positive"),
        "grid_cutoff": scientific_payload.get("grid_cutoff"),
        "eta_lead": scientific_payload.get("eta_lead"),
        "mixing": scientific_payload.get("mixing"),
        "temperature_bias": scientific_payload.get("temperature_bias_solver"),
    }
    manifest_controls = scba_manifest.get("numerical_controls")
    task_schema = scientific_payload.get("schema")
    if task_schema != "negfpy.material-nq1-scba-task.v3":
        raise MaterialCampaignError("material task schema is unsupported")
    if "mixing_floor" not in scientific_payload:
        raise MaterialCampaignError("v3 material task lacks mixing_floor")
    if "continuation_predictor" not in scientific_payload:
        raise MaterialCampaignError(
            "v3 material task lacks continuation_predictor"
        )
    expected_controls["mixing_floor"] = scientific_payload["mixing_floor"]
    if manifest_controls != expected_controls:
        raise MaterialCampaignError("SCBA numerical controls changed")
    initialization = _continuation_initialization(
        scba_manifest, scientific_payload
    )
    if receipt["continuation_initialization"] != initialization:
        raise MaterialCampaignError(
            "material task continuation initialization changed"
        )
    if initialization is not None:
        if (
            authenticated_parent_scba is None
            or authenticated_child_inputs is None
            or expected_source_tree_sha256 is None
            or expected_environment_sha256 is None
        ):
            raise MaterialCampaignError(
                "continuation receipt lacks authenticated derivation context"
            )
        child_identity, child_problem, child_config = authenticated_child_inputs
        (
            independently_derived,
            independently_derived_state,
            independently_derived_parent,
        ) = _independently_authenticate_continuation(
            recorded_parent=scba_manifest["continuation_parent"],
            scientific_task=scientific_payload,
            authenticated_parent_scba=authenticated_parent_scba,
            child_identity=child_identity,
            child_problem=child_problem,
            child_config=child_config,
            source_tree_sha256=expected_source_tree_sha256,
            environment_sha256=expected_environment_sha256,
        )
        if independently_derived != initialization:
            raise MaterialCampaignError(
                "continuation receipt state was not independently reproduced"
            )
        try:
            terminal_checkpoint = case.load_scba_checkpoint_v2_self_validating(
                checkpoint
            )
        except Exception as error:
            raise MaterialCampaignError(
                f"material task checkpoint is invalid: {error}"
            ) from error
        expected_checkpoint_identity = make_scba_checkpoint_identity_v2(
            spec.task_id,
            child_problem,
            child_config,
            environment_sha256=expected_environment_sha256,
            initial_sigma_pp=independently_derived_state,
            source_hashes=case._source_hashes(
                child_identity, independently_derived_parent
            ),
            source_tree_sha256=expected_source_tree_sha256,
        )
        if (
            terminal_checkpoint.identity != expected_checkpoint_identity
            or
            terminal_checkpoint.identity.source_hashes.get(
                "continuation_parent_initialization"
            ) != initialization["initialization_sha256"]
            or terminal_checkpoint.identity.initial_state_sha256
            != initialization["transformed_consumed_initial_sigma_sha256"]
        ):
            raise MaterialCampaignError(
                "checkpoint identity does not bind continuation initialization"
            )
    converged = bool(report.get("scba_converged"))
    if bool(status.get("scba_converged")) != converged:
        raise MaterialCampaignError("SCBA convergence flags disagree")
    if (state == "passed") != converged:
        raise MaterialCampaignError("SCBA state is inconsistent with convergence")
    return {
        "task_id": receipt["task_id"],
        "state": receipt["state"],
        "task_tree_sha256": receipt["task_tree_sha256"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "scientific_task_sha256": receipt["scientific_task_sha256"],
        "scba_campaign_manifest_sha256": scba_manifest_sha,
        "continuation_initialization": initialization,
    }


def _dependency_scba_directory(receipt_path: Path) -> Path:
    receipt = _load_canonical(receipt_path, "dependency receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("state") != "passed":
        raise MaterialCampaignError("continuation dependency is not a passed task")
    data = (receipt_path.parent / str(receipt["task_directory"])).resolve()
    # The dependency checkpoint has already been validated against the actual
    # phase TaskSpec by ``load_task_checkpoint`` in ``_dependency_artifacts``.
    # Do not call ``_receipt_evidence`` here with a fabricated object: that
    # validator requires the complete TaskSpec payload and this path is only
    # resolving the already-authenticated artifact directory.
    return data / "scba"


def _create_virgin_campaign_root(path: Path) -> Path:
    """Create one new campaign directory without following symlink ancestry."""

    selected = Path(os.path.abspath(path.expanduser()))
    _absolute_nonsymlink_path(
        selected.parent, label="material campaign parent", directory=True
    )
    if selected.exists() or selected.is_symlink():
        raise MaterialCampaignError(
            f"material campaign root already exists; refusing reuse: {selected}"
        )
    try:
        selected.mkdir()
    except FileExistsError as error:
        raise MaterialCampaignError(
            f"material campaign root creation raced: {selected}"
        ) from error
    return _absolute_nonsymlink_path(
        selected, label="material campaign root", directory=True
    )


def _prepare_campaign_root(
    path: Path, *, communicator: Any, rank: int
) -> Path:
    """Have rank zero create the virgin root and share its exact result."""

    size = int(communicator.Get_size())
    if size < 1 or not 0 <= rank < size:
        raise MaterialCampaignError("MPI communicator rank/size is invalid")
    root_value: str | None = None
    error_value: str | None = None
    if rank == 0:
        try:
            root_value = os.fspath(_create_virgin_campaign_root(path))
        except Exception as error:
            error_value = f"{type(error).__name__}: {error}"
    if size > 1:
        root_value, error_value = communicator.bcast(
            (root_value, error_value), root=0
        )
    if error_value is not None or root_value is None:
        raise MaterialCampaignError(
            "material campaign root preflight failed"
            + (f": {error_value}" if error_value is not None else "")
        )
    return _absolute_nonsymlink_path(
        Path(root_value), label="material campaign root", directory=True
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    payload = _canonical(value)
    selected = Path(os.path.abspath(path.expanduser()))
    parent = _absolute_nonsymlink_path(
        selected.parent, label="immutable JSON parent", directory=True
    )
    if selected.exists() or selected.is_symlink():
        mode = selected.lstat().st_mode
        if (
            not stat.S_ISREG(mode)
            or stat.S_ISLNK(mode)
            or stat.S_IMODE(mode) != 0o444
            or selected.read_bytes() != payload
        ):
            raise MaterialCampaignError(f"immutable JSON differs: {selected}")
        return selected
    descriptor, name = tempfile.mkstemp(
        prefix=f".{selected.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, selected)
        except FileExistsError:
            mode = selected.lstat().st_mode
            if (
                not stat.S_ISREG(mode)
                or stat.S_ISLNK(mode)
                or stat.S_IMODE(mode) != 0o444
                or selected.read_bytes() != payload
            ):
                raise MaterialCampaignError(f"immutable JSON differs: {selected}")
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return selected


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if hasattr(value, "as_dict"):
        return _plain(value.as_dict())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def run(args: argparse.Namespace, communicator: Any) -> int:
    rank = int(communicator.Get_rank())
    repo = args.repo_root.expanduser().resolve(strict=True)
    source = common._source_identity(repo, args.expected_commit)
    lock = common._validate_lock(args.environment_lock)
    source_tree = code_provenance().get("source_tree_sha256")
    if not isinstance(source_tree, str) or len(source_tree) != 64:
        raise MaterialCampaignError("source tree SHA256 is unavailable")
    base_task = next(
        task for task in silicon_nq1_release_tasks()
        if task.task_id == "coupling-g100"
    )
    # Authenticate the shared material inputs without executing a Born bubble.
    identity, base_problem, _config = _authenticate_task_inputs(
        fixture=args.fixture,
        fc3_full=args.fc3_full,
        phase_b_artifacts=args.phase_b_artifacts,
        task=base_task,
    )
    scientific_input_hashes = _pilot_input_hashes(repo, identity)
    release_authorization = _validate_release_authorization(
        repo=repo,
        qualification_path=args.qualification_report,
        pilot_audit_path=args.pilot_audit,
        source=source,
        source_tree_sha256=source_tree,
        environment_sha256=lock["sha256"],
        scientific_input_hashes=scientific_input_hashes,
    )
    qualification = release_authorization["qualification"]
    campaign_root = _prepare_campaign_root(
        args.artifacts, communicator=communicator, rank=rank
    )
    mode_set = build_observation_mode_set(
        base_problem,
        observation_layers=(4,),
        grid_step=base_task.grid_step,
        eta_lead=base_task.eta_lead,
    )
    modes_path = _atomic_json(
        campaign_root / "observation-modes.json", mode_set.as_dict()
    )
    buffer_frequencies = base_problem.grid.values[
        (base_problem.grid.values > 0.0)
        & (base_problem.grid.values <= 2.4)
    ]
    buffer_report = dict(check_harmonic_embedding_buffer(
        base_problem,
        frequencies=buffer_frequencies,
        # This is a conservative harmonic embedding identity check.  Use a
        # near-zero surface regulator so the test is not measuring the finite
        # lead broadening used by the SCBA campaign itself.
        eta_lead=1.0e-8,
    ))
    if buffer_report["passed"] is not True:
        raise MaterialCampaignError(
            "harmonic embedding-buffer preflight failed before SCBA launch"
        )
    buffer_path = _atomic_json(
        campaign_root / "harmonic-buffer-preflight.json", buffer_report
    )
    input_hashes = dict(scientific_input_hashes) | {
        "observation_modes": _sha256(modes_path),
        "harmonic_buffer_preflight": _sha256(buffer_path),
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
    campaign = build_material_nq1_mpi_campaign(
        source_revision=source["commit"],
        source_tree_sha256=source_tree,
        environment_sha256=lock["sha256"],
        input_hashes=input_hashes,
    )
    authenticated_inputs: dict[str, tuple[Any, Any, Any]] = {
        base_task.task_id: (identity, base_problem, _config)
    }

    def task_inputs(task: MaterialSCBATask) -> tuple[Any, Any, Any]:
        if task.task_id not in authenticated_inputs:
            authenticated_inputs[task.task_id] = _authenticate_task_inputs(
                fixture=args.fixture,
                fc3_full=args.fc3_full,
                phase_b_artifacts=args.phase_b_artifacts,
                task=task,
            )
        return authenticated_inputs[task.task_id]

    def validate_receipt(path: Path, spec: Any) -> dict[str, Any]:
        scientific = campaign.task(spec.task_id)
        phase = importlib.import_module(
            "negfpy.anharmonic._material_nq1_mpi"
        )._phase_manifest(campaign, campaign.wave_index(spec.task_id))
        parent_scba = None
        child_inputs = None
        if scientific.continuation_parent is not None:
            dependency_receipts = importlib.import_module(
                "negfpy.anharmonic._material_nq1_mpi"
            )._dependency_artifacts(
                campaign_root,
                campaign,
                scientific,
                validate_receipt,
            )
            parent_scba = _dependency_scba_directory(
                dependency_receipts[scientific.continuation_parent]
            )
            child_inputs = task_inputs(scientific)
        return _receipt_evidence(
            path,
            spec,
            expected_campaign_manifest_sha256=campaign.manifest.manifest_sha256,
            expected_task_identity_sha256=phase.task_identity_sha256(spec),
            authenticated_parent_scba=parent_scba,
            authenticated_child_inputs=child_inputs,
            expected_source_tree_sha256=source_tree,
            expected_environment_sha256=lock["sha256"],
        )

    def executor(
        scientific: MaterialSCBATask,
        spec: Any,
        context: Any,
        dependencies: Mapping[str, Path],
    ) -> TaskResult:
        receipt_path = context.artifact_path(".json")
        task_data = receipt_path.with_suffix(".data")
        task_data.mkdir(parents=True, exist_ok=True)
        scba_root = task_data / "scba"
        parent_scba = None
        if scientific.continuation_parent is not None:
            parent_scba = _dependency_scba_directory(
                dependencies[scientific.continuation_parent]
            )

        born_path = None
        born_record = None
        if scientific.initialization == "born":
            born_path = task_data / "born-initial.h5"
            _identity, problem, config = _authenticate_task_inputs(
                fixture=args.fixture,
                fc3_full=args.fc3_full,
                phase_b_artifacts=args.phase_b_artifacts,
                task=scientific,
            )
            if not born_path.exists():
                write_born_initial_state(
                    born_path,
                    problem,
                    config,
                    source_tree_sha256=source_tree,
                    environment_sha256=lock["sha256"],
                    source_hashes=case._source_hashes(_identity),
                )
            _initial, born_record = load_born_initial_state(
                born_path,
                problem,
                config,
                source_tree_sha256=source_tree,
                environment_sha256=lock["sha256"],
                source_hashes=case._source_hashes(_identity),
            )

        status = single_campaign.run_campaign(
            repo=repo,
            fixture=args.fixture,
            fc3_full=args.fc3_full,
            phase_b=args.phase_b_artifacts,
            artifacts=scba_root,
            expected_commit=args.expected_commit,
            environment_lock=args.environment_lock,
            run_id=scientific.task_id,
            coupling_scale=scientific.coupling_scale,
            continuation_parent_artifacts=parent_scba,
            continuation_axis=scientific.continuation_axis,
            continuation_predictor=scientific.continuation_predictor,
            born_initial_hdf5=born_path,
            grid_step=scientific.grid_step,
            grid_n_positive=scientific.grid_n_positive,
            eta_lead=scientific.eta_lead,
            mixing=scientific.mixing,
            mixing_floor=scientific.mixing_floor,
            temperature_bias=scientific.temperature_bias_solver,
            # A release task is one immutable initial attempt.  Infrastructure
            # interruption is preserved as terminal campaign evidence; it is
            # never inferred as permission to resume from files left behind by
            # an earlier execution.  Any recovery requires a separately
            # identified and qualified execution.
            resume=False,
        )
        state = status.get("state")
        if state not in {"passed", "scientific-failure"}:
            raise MaterialCampaignError(
                f"task {scientific.task_id} ended in infrastructure state {state!r}"
            )
        scba_manifest = _load_canonical(
            scba_root / "campaign-manifest.json", "SCBA campaign manifest"
        )
        continuation_initialization = _continuation_initialization(
            scba_manifest,
            scientific.as_dict(),
        )
        records = _tree_records(task_data)
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "campaign_manifest_sha256": campaign.manifest.manifest_sha256,
            "task_id": scientific.task_id,
            "task_identity_sha256": context.task_identity_sha256,
            "scientific_task_sha256": scientific.task_sha256,
            "state": state,
            "task_directory": task_data.relative_to(receipt_path.parent).as_posix(),
            "task_tree_records": records,
            "task_tree_sha256": _tree_sha256(records),
            "scba_status_sha256": _sha256(scba_root / "status.json"),
            "scba_report_sha256": _sha256(scba_root / "scba-report.json"),
            "checkpoint_sha256": _sha256(scba_root / "checkpoint/sigma-pp.h5"),
            "born_initial": born_record,
            "continuation_parent_task_id": scientific.continuation_parent,
            "continuation_initialization": continuation_initialization,
            "release_certified": False,
        }
        artifact = context.write_artifact_bytes(_canonical(receipt), suffix=".json")
        return TaskResult(
            status="completed" if state == "passed" else "scientific_failure",
            artifact_path=artifact,
            metadata={
                "scientific_task_sha256": scientific.task_sha256,
                "task_tree_sha256": receipt["task_tree_sha256"],
            },
        )

    report = execute_material_nq1_mpi_campaign(
        campaign,
        executor,
        campaign_root,
        artifact_validator=validate_receipt,
        comm=communicator,
        require_mpi=args.require_mpi,
        thread_limit=1,
        execution_id=args.execution_id,
    )
    if rank != 0:
        return 0
    assert report is not None
    document = {
        "schema": REPORT_SCHEMA,
        "campaign_manifest_sha256": report.campaign_manifest_sha256,
        "execution_id": report.execution_id,
        "source": source,
        "source_tree_sha256": source_tree,
        "environment_lock": lock,
        "candidate_qualification": qualification,
        "release_authorization": release_authorization,
        "observation_modes": {
            "path": modes_path.name,
            "sha256": _sha256(modes_path),
            "mode_set_sha256": mode_set.as_dict()["mode_set_sha256"],
            "resolved_subspace_count": len(mode_set.subspaces),
        },
        "harmonic_buffer_preflight": {
            "path": buffer_path.name,
            "sha256": _sha256(buffer_path),
            "passed": buffer_report["passed"],
        },
        "statuses": dict(report.statuses),
        "complete": report.complete,
        "phase_reports": [_plain(item) for item in report.phase_reports],
        "release_certified": False,
    }
    _atomic_json(campaign_root / "mpi-execution-report.json", document)
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0 if report.complete else 2


def _communicator(require_mpi: bool) -> Any:
    if require_mpi:
        try:
            from mpi4py import MPI
        except ImportError as error:
            raise MaterialCampaignError("--require-mpi needs mpi4py") from error
        return MPI.COMM_WORLD
    try:
        from mpi4py import MPI
    except ImportError:
        from negfpy.anharmonic._mpi_tasks import serial_task_communicator
        return serial_task_communicator()
    return MPI.COMM_WORLD


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--phase-b-artifacts", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--pilot-audit", type=Path, required=True)
    parser.add_argument("--execution-id")
    parser.add_argument("--require-mpi", action="store_true")
    args = parser.parse_args()
    try:
        return run(args, _communicator(args.require_mpi))
    except Exception as error:
        print(
            f"material Nq=1 MPI campaign error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
