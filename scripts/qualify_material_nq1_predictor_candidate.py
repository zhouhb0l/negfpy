#!/usr/bin/env python3
"""Qualify an exact staged coupling-square candidate for pilot use only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import qualify_material_nq1_candidate as base
import run_material_nq1_release_mpi as release

from negfpy.anharmonic import _material_nq1_pilot as pilot

QUALIFICATION_SCHEMA = "negfpy.material-nq1-candidate-qualification.v3"
PILOT_CONTROLS_SCHEMA = "negfpy.material-nq1-predictor-pilot-controls.v1"
BASE_QUALIFICATION_SCHEMA = "negfpy.material-nq1-candidate-qualification.v2"
FULL_CONTROLS_SCHEMA = "negfpy.material-nq1-scientific-controls.v2"
EXPECTED_PREDICTOR_COUNTS = {
    "coupling-square": 8,
    "identity": 4,
    "none": 7,
}
EXPECTED_PILOT_TASK_IDS = (
    "coupling-g0001",
    "coupling-g001",
    "coupling-g0015",
    "coupling-g002",
    "coupling-g0025",
)
EXPECTED_PILOT_TASK_SHA256 = (
    "3337ba11179aaa2a4ad2ab6962db7518282e98dfff9bdc1b80192f9d196225cf",
    "117f2e77e1627849f43c5d2109c3bf62051cdd51cfe680d5e1cfa593bcd84c22",
    "f6415bf2b4c74b2bf6b725c834decc790cdbb6a652389747c7b5895cf08ef0e8",
    "3029ac513a72a73468717316ff1951e7ff091385c1db5d27b0539b4f66b89293",
    "c00d26e45452b852ffe8f493660213c73c68675d49fc6e86e9afa041e7398c81",
)
EXPECTED_ANCHOR_CONTRACT_SHA256 = (
    "93083a3a864e84fe3b9385ff7142166ad3a995faaf23e2ee51dd386627157b0f"
)
CORE_FILE_SHA256 = {
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
PROPERTY_ORACLE = "tests/test_material_nq1_predictor_oracle.py"
PROPERTY_ORACLE_SHA256 = (
    "74c893fc6974d4bd22036e539e4ab7ec11bb11c8a8e3805ac76e32cb373fb21b"
)
PREDICTOR_CHECK_NAMES = {
    "full_scientific_controls_v2",
    "full_task_count_19",
    "predictor_mapping_8_square_4_identity_7_none",
    "pilot_exact_first_five_tasks",
    "pilot_anchor_contract_frozen",
    "pilot_anchor_files_deferred_to_preflight",
    "unchanged_core_solver",
    "unchanged_core_cubic",
    "unchanged_core_checkpoint_v2",
    "predictor_property_oracle_in_both_full_suites",
    "source_clean_and_unchanged_after_wrapper",
}


class PredictorQualificationError(RuntimeError):
    """The base evidence or frozen pilot contract failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(repo: Path, relative: str) -> Path:
    path = repo / relative
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise PredictorQualificationError(f"required file is unavailable: {relative}") from error
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise PredictorQualificationError(
            f"required file is not a non-symlink regular file: {relative}"
        )
    return path


def _repo_root(path: Path) -> Path:
    try:
        return base._repo_root(path)
    except base.QualificationError as error:
        raise PredictorQualificationError(str(error)) from error


def _publish_report_once(path: Path, report: Mapping[str, Any]) -> Path:
    """Publish one canonical mode-0444 report, accepting exact idempotence."""

    try:
        return pilot.publish_json_once(path.expanduser(), report)
    except pilot.MaterialNQ1PilotError as error:
        raise PredictorQualificationError(str(error)) from error


def _core_file_records(repo: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for relative, expected in CORE_FILE_SHA256.items():
        observed = _sha256(_regular_file(repo, relative))
        if observed != expected:
            raise PredictorQualificationError(
                f"frozen core file SHA256 changed: {relative}"
            )
        records[relative] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "passed": True,
        }
    return records


def _anchor_contract() -> dict[str, Any]:
    return {
        "root": os.fspath(pilot.RELEASE_018_ANCHOR_ROOT),
        "source_revision": pilot.RELEASE_018_COMMIT,
        "source_tree_sha256": pilot.RELEASE_018_TREE,
        "campaign_payload_sha256": pilot.RELEASE_018_CAMPAIGN_PAYLOAD_SHA256,
        "records": {
            task_id: {
                name: dict(record) for name, record in records.items()
            }
            for task_id, records in pilot.RELEASE_018_ANCHORS.items()
        },
        "use": "comparison-only-never-continuation-parent",
    }


def _pilot_controls(full_controls: Mapping[str, Any]) -> dict[str, Any]:
    tasks = full_controls.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 19:
        raise PredictorQualificationError("full scientific task set is not 19 tasks")
    selected = tasks[:5]
    if tuple(task.get("task_id") for task in selected) != EXPECTED_PILOT_TASK_IDS:
        raise PredictorQualificationError("pilot is not the exact first five tasks")
    if tuple(task.get("task_sha256") for task in selected) != EXPECTED_PILOT_TASK_SHA256:
        raise PredictorQualificationError("exact pilot task controls changed")
    if tuple(task.as_dict() | {"task_sha256": task.task_sha256}
             for task in pilot.predictor_pilot_tasks()) != tuple(selected):
        raise PredictorQualificationError("pilot and full task records disagree")

    anchors = _anchor_contract()
    anchor_sha256 = hashlib.sha256(release._canonical(anchors)).hexdigest()
    if anchor_sha256 != EXPECTED_ANCHOR_CONTRACT_SHA256:
        raise PredictorQualificationError("release-018 anchor contract changed")

    payload = {
        "schema": PILOT_CONTROLS_SCHEMA,
        "classification": "development-pilot-not-release",
        "task_count": len(EXPECTED_PILOT_TASK_IDS),
        "tasks": selected,
        "solver_contract": {
            "backend": "fft",
            "hilbert_backend": "fft",
            "eta_device": 0.0,
            "max_workers": 1,
            "max_iterations": 800,
            "min_iterations": 3,
            "consecutive_passes": 3,
            "on_nonconvergence": "raise",
            "automatic_retry": False,
            "controls_changed_after_outcome": False,
        },
        "acceptance": {
            **dict(full_controls["acceptance"]),
            "anchor_field_rtol": pilot.FIELD_RTOL,
            "anchor_field_atol": pilot.FIELD_ATOL,
            "anchor_residual_factor": pilot.ANCHOR_RESIDUAL_FACTOR,
            "anchor_current_rtol": pilot.CURRENT_RTOL,
            "anchor_current_atol": pilot.CURRENT_ATOL,
        },
        "anchor_contract": anchors,
        "anchor_contract_sha256": anchor_sha256,
        "anchor_authentication_gate": "staged-five-task-pilot-preflight",
        "excluded_claims": list(full_controls["excluded_claims"]),
        "release_certified": False,
    }
    return payload | {
        "task_set_sha256": hashlib.sha256(release._canonical(payload)).hexdigest()
    }


def _validate_base_report(
    document: Mapping[str, Any], *, repo: Path, evidence_root: Path
) -> dict[str, Any]:
    expected_keys = {
        "schema", "source", "source_tree_sha256", "environment_sha256",
        "scientific_controls", "checks", "evidence",
        "release_preflight_passed",
    }
    checks = document.get("checks")
    controls = document.get("scientific_controls")
    if (
        set(document) != expected_keys
        or document.get("schema") != BASE_QUALIFICATION_SCHEMA
        or not isinstance(checks, Mapping)
        or set(checks) != release.EXPECTED_QUALIFICATION_CHECKS
        or not all(value is True for value in checks.values())
        or document.get("release_preflight_passed") is not True
        or not isinstance(controls, Mapping)
        or controls.get("schema") != FULL_CONTROLS_SCHEMA
        or controls != release._scientific_control_record()
    ):
        raise PredictorQualificationError("base candidate qualification is invalid")
    controls_payload = dict(controls)
    recorded_digest = controls_payload.pop("task_set_sha256", None)
    if hashlib.sha256(release._canonical(controls_payload)).hexdigest() != recorded_digest:
        raise PredictorQualificationError("full scientific task-set digest changed")
    try:
        return base.validate_qualification_evidence(
            document, repo=repo, evidence_root=evidence_root
        )
    except Exception as error:
        raise PredictorQualificationError(
            f"base qualification evidence did not authenticate: {error}"
        ) from error


def _predictor_counts(controls: Mapping[str, Any]) -> dict[str, int]:
    tasks = controls.get("tasks")
    if not isinstance(tasks, list):
        raise PredictorQualificationError("scientific controls lack tasks")
    counts = Counter(
        "none" if task.get("continuation_predictor") is None
        else str(task.get("continuation_predictor"))
        for task in tasks
    )
    result = {name: int(counts.get(name, 0)) for name in EXPECTED_PREDICTOR_COUNTS}
    if dict(counts) != EXPECTED_PREDICTOR_COUNTS:
        raise PredictorQualificationError("predictor mapping is not 8/4/7")
    return result


def _property_oracle_record(
    repo: Path, base_report: Mapping[str, Any]
) -> dict[str, Any]:
    path = _regular_file(repo, PROPERTY_ORACLE)
    observed = _sha256(path)
    if observed != PROPERTY_ORACLE_SHA256:
        raise PredictorQualificationError("frozen property-oracle SHA256 changed")
    try:
        records = base_report["evidence"]["records"]
        suites = {
            label: {
                "scope": list(records[label]["scope"]),
                "excluded_tests": list(records[label]["excluded_tests"]),
            }
            for label in ("python310_pytest", "python312_pytest")
        }
    except (KeyError, TypeError) as error:
        raise PredictorQualificationError(
            "base qualification lacks full-suite scope evidence"
        ) from error
    if any(
        "tests" not in record["scope"] or record["excluded_tests"]
        for record in suites.values()
    ):
        raise PredictorQualificationError(
            "property oracle was not covered by both complete test suites"
        )
    return {
        "relative_path": PROPERTY_ORACLE,
        "sha256": observed,
        "covered_by": ["python310_full_suite", "python312_full_suite"],
        "suite_records": suites,
    }


def validate_predictor_qualification_report(
    document: Mapping[str, Any], *, repo: Path, evidence_root: Path
) -> dict[str, Any]:
    """Recursively authenticate the exact v3 pilot-only qualification."""

    repo = _repo_root(repo)
    expected_top = {
        "schema", "source", "source_tree_sha256", "environment_sha256",
        "scientific_controls", "predictor_pilot_controls", "checks",
        "evidence", "candidate_qualification_passed",
        "pilot_submission_eligible", "release_submission_eligible",
        "next_gate",
    }
    evidence = document.get("evidence")
    checks = document.get("checks")
    expected_checks = set(release.EXPECTED_QUALIFICATION_CHECKS) | (
        PREDICTOR_CHECK_NAMES
    )
    if (
        set(document) != expected_top
        or document.get("schema") != QUALIFICATION_SCHEMA
        or not isinstance(evidence, Mapping)
        or set(evidence) != {
            "base_qualification_schema", "base_qualification_evidence",
            "core_files", "predictor_counts", "property_oracle",
        }
        or not isinstance(checks, Mapping)
        or set(checks) != expected_checks
        or not all(value is True for value in checks.values())
        or document.get("candidate_qualification_passed") is not True
        or document.get("pilot_submission_eligible") is not True
        or document.get("release_submission_eligible") is not False
        or document.get("next_gate") != "staged-five-task-pilot"
    ):
        raise PredictorQualificationError(
            "predictor candidate qualification schema is invalid"
        )

    base_checks = {
        name: checks[name] for name in release.EXPECTED_QUALIFICATION_CHECKS
    }
    base_report = {
        "schema": evidence.get("base_qualification_schema"),
        "source": document.get("source"),
        "source_tree_sha256": document.get("source_tree_sha256"),
        "environment_sha256": document.get("environment_sha256"),
        "scientific_controls": document.get("scientific_controls"),
        "checks": base_checks,
        "evidence": evidence.get("base_qualification_evidence"),
        "release_preflight_passed": True,
    }
    authenticated_base = _validate_base_report(
        base_report, repo=repo, evidence_root=evidence_root
    )
    full_controls = authenticated_base["scientific_controls"]
    expected_pilot_controls = _pilot_controls(full_controls)
    expected_counts = _predictor_counts(full_controls)
    expected_core = _core_file_records(repo)
    expected_oracle = _property_oracle_record(repo, authenticated_base)
    if (
        document.get("scientific_controls") != full_controls
        or document.get("predictor_pilot_controls") != expected_pilot_controls
        or evidence.get("base_qualification_schema") != BASE_QUALIFICATION_SCHEMA
        or evidence.get("base_qualification_evidence")
        != authenticated_base["evidence"]
        or evidence.get("predictor_counts") != expected_counts
        or evidence.get("core_files") != expected_core
        or evidence.get("property_oracle") != expected_oracle
    ):
        raise PredictorQualificationError(
            "predictor qualification evidence did not authenticate"
        )
    return dict(document)


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    repo = _repo_root(args.repo_root)
    try:
        _output, evidence_root = base._qualification_output_paths(args.output)
    except base.QualificationError as error:
        raise PredictorQualificationError(str(error)) from error
    base_report = _validate_base_report(
        base.qualify(args), repo=repo, evidence_root=evidence_root
    )
    controls = dict(base_report["scientific_controls"])
    pilot_controls = _pilot_controls(controls)
    predictor_counts = _predictor_counts(controls)
    core_files = _core_file_records(repo)
    property_oracle = _property_oracle_record(repo, base_report)

    final_source = base.common._source_identity(repo, args.expected_commit)
    final_tree = base.code_provenance().get("source_tree_sha256")
    source_unchanged = bool(
        final_source == base_report["source"]
        and final_tree == base_report["source_tree_sha256"]
    )
    if not source_unchanged:
        raise PredictorQualificationError(
            "candidate source changed after the base qualification"
        )

    checks = dict(base_report["checks"])
    checks.update({
        "full_scientific_controls_v2": True,
        "full_task_count_19": controls["task_count"] == 19,
        "predictor_mapping_8_square_4_identity_7_none": (
            predictor_counts == EXPECTED_PREDICTOR_COUNTS
        ),
        "pilot_exact_first_five_tasks": True,
        "pilot_anchor_contract_frozen": True,
        "pilot_anchor_files_deferred_to_preflight": True,
        "unchanged_core_solver": core_files[
            "src/negfpy/anharmonic/solver.py"
        ]["passed"],
        "unchanged_core_cubic": core_files[
            "src/negfpy/anharmonic/cubic.py"
        ]["passed"],
        "unchanged_core_checkpoint_v2": core_files[
            "src/negfpy/anharmonic/_checkpoint_v2.py"
        ]["passed"],
        "predictor_property_oracle_in_both_full_suites": True,
        "source_clean_and_unchanged_after_wrapper": source_unchanged,
    })
    passed = bool(checks and all(value is True for value in checks.values()))
    report = {
        "schema": QUALIFICATION_SCHEMA,
        "source": base_report["source"],
        "source_tree_sha256": base_report["source_tree_sha256"],
        "environment_sha256": base_report["environment_sha256"],
        "scientific_controls": controls,
        "predictor_pilot_controls": pilot_controls,
        "checks": checks,
        "evidence": {
            "base_qualification_schema": base_report["schema"],
            "base_qualification_evidence": base_report["evidence"],
            "core_files": core_files,
            "predictor_counts": predictor_counts,
            "property_oracle": property_oracle,
        },
        "candidate_qualification_passed": passed,
        "pilot_submission_eligible": passed,
        "release_submission_eligible": False,
        "next_gate": "staged-five-task-pilot",
    }
    return validate_predictor_qualification_report(
        report, repo=repo, evidence_root=evidence_root
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--phase-b-artifacts", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--python310", type=Path, required=True)
    parser.add_argument("--python312", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = qualify(args)
        _publish_report_once(args.output, report)
    except Exception as error:
        print(
            f"material predictor qualification error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["candidate_qualification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
