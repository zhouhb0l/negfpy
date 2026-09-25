#!/usr/bin/env python3
"""Operate the staged five-task material Nq=1 predictor pilot, never a scheduler.

Actions are deliberately one-shot and filesystem-only:

``initialize``
    Freeze the exact candidate, inputs, task graph, and release-018 anchors.
``preflight``
    Reproduce the candidate/input identities and authenticate every anchor.
``run-task``
    Execute exactly one task after all earlier gates passed, then publish its
    immutable receipt and anchor comparison.
``finalize``
    Require all three coupling tasks and publish the non-release pilot report.
``audit``
    Hash the complete local pilot evidence and publish the final audit.

The command contains no qsub, qdel, or scheduler-query operation.  A
scheduler receipt produced by an operational wrapper may be hash-bound to an
individual task receipt, but is never created here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import qualify_material_nq1_predictor_candidate as predictor_qualification
import run_sparse_nq1_phase_c_born_campaign as common
import run_sparse_nq1_phase_c_scba_campaign as single_campaign
import validate_sparse_nq1_phase_c_scba as case

from negfpy.anharmonic._material_nq1_pilot import (
    PILOT_TASK_IDS,
    RELEASE_018_ANCHOR_ROOT,
    MaterialNQ1PilotError,
    authenticate_anchor_source,
    build_audit,
    build_comparison,
    build_final_report,
    build_pilot_manifest,
    build_preflight_record,
    build_task_receipt,
    build_launch_environment_record,
    load_canonical_json,
    predictor_pilot_tasks,
    publish_json_once,
    qualification_authorizes_pilot,
    qualified_python312_runtime,
    sha256_file,
    validate_audit,
    validate_comparison,
    validate_pilot_manifest,
    validate_preflight_record,
    validate_scheduler_receipt,
    validate_task_receipt,
)
from negfpy.anharmonic._launch_environment import qualification_environment
from negfpy.anharmonic._provenance import code_provenance

MANIFEST_NAME = "pilot-manifest.json"
PREFLIGHT_NAME = "pilot-preflight.json"
REPORT_NAME = "pilot-report.json"
AUDIT_NAME = "pilot-audit.json"


def _source_tree_sha256() -> str:
    value = code_provenance().get("source_tree_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise MaterialNQ1PilotError("source tree SHA256 is unavailable")
    return value


def _task(task_id: str):
    matches = [item for item in predictor_pilot_tasks() if item.task_id == task_id]
    if len(matches) != 1:
        raise MaterialNQ1PilotError(f"unknown pilot task: {task_id!r}")
    return matches[0]


def _authenticate_inputs(args: argparse.Namespace) -> dict[str, str]:
    identity, _problem, _config = case.authenticate_scba_inputs(
        fixture=args.fixture,
        fc3_full=args.fc3_full,
        phase_b_artifacts=args.phase_b_artifacts,
        coupling_scale=0.001,
        grid_step=0.1,
        grid_n_positive=75,
        eta_lead=0.025,
        mixing=0.2,
        mixing_floor=0.001,
        temperature_bias=0.0,
    )
    repo = _nonsymlink_directory(args.repo_root, name="repository", create=False)
    return dict(case._source_hashes(identity)) | {
        "scba_case_driver": sha256_file(
            repo / "scripts/validate_sparse_nq1_phase_c_scba.py"
        )
    }


def _candidate_identity(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], str]:
    repo = _nonsymlink_directory(args.repo_root, name="repository", create=False)
    source = common._source_identity(repo, args.expected_commit)
    lexical_lock = Path(os.path.abspath(args.environment_lock.expanduser()))
    sha256_file(lexical_lock)
    lock = common._validate_lock(lexical_lock)
    if lock.get("path") != os.fspath(lexical_lock):
        raise MaterialNQ1PilotError("environment lock path is not lexical and exact")
    tree = _source_tree_sha256()
    if source.get("commit") != args.expected_commit or source.get("dirty") is not False:
        raise MaterialNQ1PilotError("candidate source is not exact and clean")
    return source, lock, tree


def _root(args: argparse.Namespace, *, create: bool = False) -> Path:
    return _nonsymlink_directory(args.root, name="pilot root", create=create)


def _nonsymlink_directory(
    value: Path, *, name: str, create: bool
) -> Path:
    raw = value.expanduser()
    lexical = raw if raw.is_absolute() else Path.cwd() / raw
    lexical = Path(os.path.abspath(lexical))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            if create:
                cursor.mkdir()
                mode = cursor.lstat().st_mode
            else:
                raise MaterialNQ1PilotError(f"{name} is unavailable: {lexical}")
        if stat.S_ISLNK(mode):
            raise MaterialNQ1PilotError(f"{name} may not traverse a symlink: {cursor}")
        if not stat.S_ISDIR(mode):
            raise MaterialNQ1PilotError(f"{name} component is not a directory: {cursor}")
    return lexical


def _load_manifest(root: Path) -> dict[str, Any]:
    document = load_canonical_json(root / MANIFEST_NAME, name="pilot manifest")
    return validate_pilot_manifest(document)


def _qualification_matches(
    manifest: Mapping[str, Any],
    document: Mapping[str, Any],
    qualification_path: Path,
) -> bool:
    try:
        if (
            not qualification_path.is_absolute()
            or qualification_path != Path(os.path.abspath(qualification_path))
            or stat.S_IMODE(qualification_path.lstat().st_mode) != 0o444
            or sha256_file(qualification_path) != manifest["qualification_sha256"]
        ):
            return False
    except (OSError, KeyError, TypeError):
        return False
    if not qualification_authorizes_pilot(manifest, document):
        return False
    repo = Path(manifest["source_package_file"]).parents[2]
    evidence_root = qualification_path.parent / "qualification-evidence"
    try:
        authenticated = predictor_qualification.validate_predictor_qualification_report(
            document,
            repo=repo,
            evidence_root=evidence_root,
        )
    except Exception:  # noqa: BLE001 - validation is a fail-closed predicate.
        return False
    return authenticated == document and _current_runtime_matches(
        manifest, document
    )


def _current_runtime_matches(
    manifest: Mapping[str, Any], document: Mapping[str, Any]
) -> bool:
    """Reproduce the qualified P312 executable, version, and package bytes."""

    try:
        runtime, _lock = qualified_python312_runtime(manifest, document)
        lexical = Path(os.path.abspath(sys.executable))
        current = Path(sys.executable).resolve(strict=True)
        if (
            lexical != current
            or str(current) != runtime["executable"]
            or sha256_file(current) != runtime["executable_sha256"]
            or platform.python_version() != runtime["version"]
        ):
            return False
        environment = qualification_environment(
            runtime, repo=Path(manifest["source_package_file"]).parents[2]
        )
        if dict(os.environ) != environment:
            return False
        completed = subprocess.run(
            [str(current), "-m", "pip", "freeze", "--all"],
            cwd=Path(manifest["source_package_file"]).parents[2],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=60,
        )
        return bool(
            completed.returncode == 0
            and completed.stdout
            and hashlib.sha256(completed.stdout).hexdigest()
            == runtime["packages_sha256"]
        )
    except Exception:  # noqa: BLE001 - runtime authentication is fail closed.
        return False


def _revalidate_bound_qualification(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight_record = validate_preflight_record(manifest, root / PREFLIGHT_NAME)
    qualification_path = Path(preflight_record["qualification"]["path"])
    document = load_canonical_json(
        qualification_path, name="bound qualification report"
    )
    if not _qualification_matches(manifest, document, qualification_path):
        raise MaterialNQ1PilotError(
            "bound qualification evidence no longer authenticates"
        )
    return qualified_python312_runtime(manifest, document)


def initialize(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args, create=True)
    unexpected = [path for path in root.iterdir() if path.name != MANIFEST_NAME]
    if unexpected:
        raise MaterialNQ1PilotError(
            "pilot root is nonempty before initialization; refusing overwrite"
        )
    source, lock, tree = _candidate_identity(args)
    qualification = Path(os.path.abspath(args.qualification_report.expanduser()))
    qualification_document = load_canonical_json(
        qualification, name="qualification report"
    )
    input_hashes = _authenticate_inputs(args)
    manifest = build_pilot_manifest(
        source_revision=source["commit"],
        source_git_tree=source["tree"],
        source_package_file=source["package_file"],
        source_tree_sha256=tree,
        environment_sha256=lock["sha256"],
        qualification_sha256=sha256_file(qualification),
        input_hashes=input_hashes,
        execution_id=args.execution_id,
        anchor_root=args.anchor_root,
    )
    if not _qualification_matches(manifest, qualification_document, qualification):
        raise MaterialNQ1PilotError("qualification does not bind the exact candidate")
    _runtime, qualified_lock = qualified_python312_runtime(
        manifest, qualification_document
    )
    if lock != qualified_lock:
        raise MaterialNQ1PilotError("live environment lock is not the qualified lock")
    publish_json_once(root / MANIFEST_NAME, manifest)
    return manifest


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    manifest = _load_manifest(root)
    source, lock, tree = _candidate_identity(args)
    exact_source = bool(
        source == {
            "commit": manifest["source_revision"],
            "tree": manifest["source_git_tree"],
            "dirty": False,
            "package_file": manifest["source_package_file"],
        }
        and tree == manifest["source_tree_sha256"]
        and lock["sha256"] == manifest["environment_sha256"]
    )
    qualification = Path(os.path.abspath(args.qualification_report.expanduser()))
    qualification_document = load_canonical_json(
        qualification, name="qualification report"
    )
    if not _qualification_matches(manifest, qualification_document, qualification):
        raise MaterialNQ1PilotError("qualification no longer binds the candidate")
    _runtime, qualified_lock = qualified_python312_runtime(
        manifest, qualification_document
    )
    if lock != qualified_lock:
        raise MaterialNQ1PilotError("live environment lock is not the qualified lock")
    inputs_reproduced = _authenticate_inputs(args) == manifest["input_hashes"]
    anchors = authenticate_anchor_source(manifest)
    record = build_preflight_record(
        manifest=manifest,
        manifest_path=root / MANIFEST_NAME,
        qualification_path=qualification,
        source_clean_and_exact=exact_source,
        input_hashes_reproduced=inputs_reproduced,
        anchors=anchors,
    )
    if record["passed"] is not True:
        raise MaterialNQ1PilotError("pilot preflight failed")
    publish_json_once(root / PREFLIGHT_NAME, record)
    return record


def _task_receipt_path(root: Path, task_id: str) -> Path:
    return root / "tasks" / task_id / "task-receipt.json"


def _scheduler_receipt_path(root: Path, task_id: str) -> Path:
    return root / "scheduler" / task_id / "scheduler-terminal-receipt.json"


def _require_virgin_task_root(task_root: Path, task_id: str) -> None:
    _nonsymlink_directory(
        task_root.parent,
        name="pilot tasks directory",
        create=True,
    )
    if task_root.exists() or task_root.is_symlink():
        raise MaterialNQ1PilotError(
            "pilot task already has evidence; start a separately identified pilot"
        )
    try:
        task_root.mkdir(mode=0o700)
        (task_root / "scba").mkdir(mode=0o700)
    except FileExistsError as error:
        raise MaterialNQ1PilotError(
            "pilot task claim raced; start a separately identified pilot"
        ) from error
    _nonsymlink_directory(
        task_root / "scba", name=f"{task_id} claimed SCBA root", create=False
    )


def _verify_live_candidate(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    *,
    qualified_lock: Mapping[str, Any] | None = None,
) -> None:
    if qualified_lock is None:
        qualification_path = Path(
            os.path.abspath(args.qualification_report.expanduser())
        )
        qualification = load_canonical_json(
            qualification_path, name="live qualification report"
        )
        if not _qualification_matches(manifest, qualification, qualification_path):
            raise MaterialNQ1PilotError("live qualification no longer authenticates")
        _runtime, qualified_lock = qualified_python312_runtime(
            manifest, qualification
        )
    source, lock, tree = _candidate_identity(args)
    expected_source = {
        "commit": manifest["source_revision"],
        "tree": manifest["source_git_tree"],
        "dirty": False,
        "package_file": manifest["source_package_file"],
    }
    if (
        source != expected_source
        or tree != manifest["source_tree_sha256"]
        or lock != qualified_lock
        or _authenticate_inputs(args) != manifest["input_hashes"]
    ):
        raise MaterialNQ1PilotError(
            "live candidate, environment, or authenticated inputs changed"
        )


def _anchor_comparison_path(root: Path, task_id: str) -> Path:
    return root / "comparisons" / f"anchor-{task_id}.json"


def _verify_prior_gates(root: Path, manifest: Mapping[str, Any], task_id: str) -> None:
    index = PILOT_TASK_IDS.index(task_id)
    for prior_id in PILOT_TASK_IDS[:index]:
        receipt = validate_task_receipt(manifest, _task_receipt_path(root, prior_id))
        validate_scheduler_receipt(
            manifest, prior_id, _scheduler_receipt_path(root, prior_id)
        )
        prior_task = next(
            item for item in manifest["tasks"] if item["task_id"] == prior_id
        )
        comparison_passed = True
        if prior_task["anchor_task_id"] is not None:
            comparison = validate_comparison(
                manifest,
                _anchor_comparison_path(root, prior_id),
            )
            comparison_passed = comparison.get("passed") is True
        if receipt["state"] != "passed" or not comparison_passed:
            raise MaterialNQ1PilotError(
                f"prior pilot gate {prior_id!r} did not pass; stop and preserve"
            )


def _anchor_paths(manifest: Mapping[str, Any], task_id: str) -> tuple[Path, Path, dict[str, str]]:
    anchor_id = next(
        item["anchor_task_id"]
        for item in manifest["tasks"]
        if item["task_id"] == task_id
    )
    if anchor_id is None:
        raise MaterialNQ1PilotError(f"task {task_id!r} has no historical anchor")
    source = manifest["anchor_source"]
    records = source["records"][anchor_id]
    root = Path(source["root"])
    return (
        root / records["fields"]["relative_path"],
        root / records["outcome"]["relative_path"],
        {
            "fields": records["fields"]["sha256"],
            "outcome": records["outcome"]["sha256"],
        },
    )


def run_task(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    manifest = _load_manifest(root)
    runtime, qualified_lock = _revalidate_bound_qualification(root, manifest)
    _verify_live_candidate(args, manifest, qualified_lock=qualified_lock)
    task = _task(args.task_id)
    _verify_prior_gates(root, manifest, task.task_id)
    launch_record = build_launch_environment_record(
        manifest, task.task_id, runtime, dict(os.environ)
    )

    task_root = root / "tasks" / task.task_id
    _require_virgin_task_root(task_root, task.task_id)
    publish_json_once(task_root / "launch-environment.json", launch_record)
    scba_root = task_root / "scba"
    parent_receipt = None
    parent_scba = None
    if task.continuation_parent is not None:
        parent_receipt = _task_receipt_path(root, task.continuation_parent)
        validate_task_receipt(manifest, parent_receipt)
        parent_scba = root / "tasks" / task.continuation_parent / "scba"

    status = single_campaign.run_campaign(
        repo=_nonsymlink_directory(args.repo_root, name="repository", create=False),
        fixture=args.fixture,
        fc3_full=args.fc3_full,
        phase_b=args.phase_b_artifacts,
        artifacts=scba_root,
        expected_commit=args.expected_commit,
        environment_lock=args.environment_lock,
        run_id=task.task_id,
        coupling_scale=task.coupling_scale,
        continuation_parent_artifacts=parent_scba,
        continuation_axis=task.continuation_axis,
        continuation_predictor=task.continuation_predictor,
        grid_step=task.grid_step,
        grid_n_positive=task.grid_n_positive,
        eta_lead=task.eta_lead,
        mixing=task.mixing,
        mixing_floor=task.mixing_floor,
        temperature_bias=task.temperature_bias_solver,
        resume=False,
        launch_environment=launch_record["environment"],
    )
    if status.get("state") not in {"passed", "scientific-failure"}:
        raise MaterialNQ1PilotError("task ended in an infrastructure state")

    # Close the long-running task's runtime/source TOCTOU window before any
    # receipt is published.  Drift is terminal for this pilot, never a retry.
    _post_runtime, post_lock = _revalidate_bound_qualification(root, manifest)
    _verify_live_candidate(args, manifest, qualified_lock=post_lock)

    receipt = build_task_receipt(
        manifest=manifest,
        task_id=task.task_id,
        scba_root=scba_root,
        parent_receipt=parent_receipt,
    )
    receipt_path = publish_json_once(_task_receipt_path(root, task.task_id), receipt)
    selected_task = next(
        item for item in manifest["tasks"] if item["task_id"] == task.task_id
    )
    comparison_passed = None
    if selected_task["anchor_task_id"] is not None:
        anchor_fields, anchor_outcome, anchor_hashes = _anchor_paths(
            manifest, task.task_id
        )
        comparison_id = f"anchor-{task.task_id}"
        comparison = build_comparison(
            manifest=manifest,
            comparison_id=comparison_id,
            first_receipt_path=receipt_path,
            second_kind="release-018-anchor",
            second_id=selected_task["anchor_task_id"],
            second_fields_path=anchor_fields,
            second_outcome_path=anchor_outcome,
            second_hashes=anchor_hashes,
        )
        publish_json_once(_anchor_comparison_path(root, task.task_id), comparison)
        comparison_passed = comparison["passed"]
        if comparison_passed is not True:
            raise MaterialNQ1PilotError(
                f"task {task.task_id!r} failed its immutable anchor comparison"
            )
    if receipt["state"] != "passed":
        raise MaterialNQ1PilotError(
            f"task {task.task_id!r} ended in scientific failure; stop and preserve"
        )
    return {
        "task_id": task.task_id,
        "state": receipt["state"],
        "anchor_comparison_passed": comparison_passed,
        "release_certified": False,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    manifest = _load_manifest(root)
    _revalidate_bound_qualification(root, manifest)
    receipts = {
        task_id: _task_receipt_path(root, task_id) for task_id in PILOT_TASK_IDS
    }
    for path in receipts.values():
        receipt = validate_task_receipt(manifest, path)
        if receipt["state"] != "passed":
            raise MaterialNQ1PilotError("cannot finalize a failed pilot task")
    comparisons = {
        f"anchor-{task_id}": _anchor_comparison_path(root, task_id)
        for task_id in PILOT_TASK_IDS[:2]
    }
    scheduler_receipts = {
        task_id: _scheduler_receipt_path(root, task_id)
        for task_id in PILOT_TASK_IDS
    }
    report = build_final_report(
        manifest=manifest,
        manifest_path=root / MANIFEST_NAME,
        preflight_path=root / PREFLIGHT_NAME,
        task_receipts=receipts,
        comparisons=comparisons,
        scheduler_receipts=scheduler_receipts,
    )
    publish_json_once(root / REPORT_NAME, report)
    if report["pilot_passed"] is not True:
        raise MaterialNQ1PilotError("predictor pilot did not pass")
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = _root(args)
    manifest = _load_manifest(root)
    _revalidate_bound_qualification(root, manifest)
    record = build_audit(
        manifest=manifest,
        root=root,
        report_path=root / REPORT_NAME,
        audit_path=root / AUDIT_NAME,
    )
    audit_path = publish_json_once(root / AUDIT_NAME, record)
    return validate_audit(manifest, root=root, audit_path=audit_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("initialize", "preflight", "run-task", "finalize", "audit")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--qualification-report", type=Path)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fc3-full", type=Path)
    parser.add_argument("--phase-b-artifacts", type=Path)
    parser.add_argument("--execution-id")
    parser.add_argument("--anchor-root", type=Path, default=RELEASE_018_ANCHOR_ROOT)
    parser.add_argument("--task-id", choices=PILOT_TASK_IDS)
    return parser


def _require(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        raise MaterialNQ1PilotError(
            f"action {args.action!r} requires: " + ", ".join(missing)
        )


def main() -> int:
    args = _parser().parse_args()
    common_names = (
        "repo_root",
        "expected_commit",
        "environment_lock",
        "qualification_report",
        "fixture",
        "fc3_full",
        "phase_b_artifacts",
    )
    try:
        if args.action == "initialize":
            _require(args, common_names + ("execution_id",))
            result = initialize(args)
        elif args.action == "preflight":
            _require(args, common_names)
            result = preflight(args)
        elif args.action == "run-task":
            _require(args, common_names + ("task_id",))
            result = run_task(args)
        elif args.action == "finalize":
            result = finalize(args)
        else:
            result = audit(args)
    except Exception as error:  # noqa: BLE001 - CLI must fail closed uniformly.
        print(
            f"material Nq=1 predictor pilot error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3
    print(json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
