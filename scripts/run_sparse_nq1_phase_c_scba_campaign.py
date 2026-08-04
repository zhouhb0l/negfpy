#!/usr/bin/env python3
"""Checkpointed, fixed-control campaign wrapper for one material SCBA case.

The wrapper is intentionally conservative: a manifest is immutable, a
non-terminal attempt is resumed only with the same hashes, and no scientific
or infrastructure failure is retried with changed controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import negfpy

import run_sparse_nq1_phase_c_born_campaign as common
import validate_sparse_nq1_phase_c_scba as case


SCHEMA = "negfpy.sparse-nq1-phase-c-scba-campaign.v1"
STATUS_SCHEMA = "negfpy.sparse-nq1-phase-c-scba-status.v1"
REPORT_SCHEMA = "negfpy.sparse-nq1-phase-c-scba-report.v1"


def _sha256(path: Path) -> str:
    return common._sha256_file(path)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return common._canonical_json(value)


def _source_hash() -> str:
    from negfpy.anharmonic._provenance import code_provenance
    value = code_provenance().get("source_tree_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise common.CampaignError("source tree SHA256 is unavailable")
    return value


def _manifest(
    *, repo: Path, fixture: Path, fc3_full: Path, phase_b: Path, artifacts: Path,
    expected_commit: str, environment_lock: Path, run_id: str, coupling_scale: float,
    initial_checkpoint: Path | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    artifacts = artifacts.resolve()
    checkpoint = artifacts / "checkpoint" / "sigma-pp.h5"
    output_json = artifacts / "outcomes" / "scba.json"
    output_hdf5 = artifacts / "outcomes" / "scba-fields.h5"
    source = common._source_identity(repo, expected_commit)
    lock = common._validate_lock(environment_lock)
    preflight = case.preflight_scba_inputs(
        fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b,
        checkpoint=checkpoint, output_json=output_json, output_hdf5=output_hdf5,
        source_tree_sha256=_source_hash(), environment_sha256=lock["sha256"], run_id=run_id,
        coupling_scale=coupling_scale, initial_checkpoint=initial_checkpoint,
    )
    driver = repo / "scripts" / "validate_sparse_nq1_phase_c_scba.py"
    if not driver.is_file():
        raise common.CampaignError(f"missing SCBA case driver: {driver}")
    return {
        "schema": SCHEMA,
        "scientific_scope": "authenticated silicon Nq=1 sparse material SCBA, fixed Nw=17 development gate",
        "source": source,
        "source_tree_sha256": _source_hash(),
        "environment_lock": lock,
        "driver": {"relative_path": "scripts/validate_sparse_nq1_phase_c_scba.py", "sha256": _sha256(driver)},
        "preflight": preflight,
        "run_id": run_id,
        "coupling_scale": coupling_scale,
        "continuation_parent": (
            None if initial_checkpoint is None else {
                "path": str(initial_checkpoint.resolve()),
                "sha256": _sha256(initial_checkpoint),
            }
        ),
        "fixed_controls_no_retuning": True,
        "restart": {"checkpoint_schema": case.CHECKPOINT_SCHEMA, "resume_exact_identity": True, "automatic_retry": False},
        "watchdog": {"wall_limit_seconds": common.WALL_LIMIT_SECONDS, "rss_limit_bytes": common.RSS_LIMIT_BYTES, "term_grace_seconds": common.TERM_GRACE_SECONDS, "automatic_retry": False},
        "outputs": {"checkpoint": str(checkpoint), "json": str(output_json), "hdf5": str(output_hdf5)},
        "python": {"executable": str(Path(sys.executable).resolve()), "version": platform.python_version()},
        "distributions": common._versions(),
        "thread_limits": dict(sorted(case.phase_c.THREAD_LIMIT_ENV.items())),
    }


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return common._load_canonical(path, name=label)


def _publish_report(artifacts: Path, manifest_path: Path, running_path: Path, log_path: Path) -> dict[str, Any]:
    manifest_sha = _sha256(manifest_path)
    log = _load_json(log_path, "SCBA subprocess log")
    output_json = artifacts / "outcomes" / "scba.json"
    output_hdf5 = artifacts / "outcomes" / "scba-fields.h5"
    checkpoint = artifacts / "checkpoint" / "sigma-pp.h5"
    if not output_json.is_file() or not output_hdf5.is_file() or not checkpoint.is_file():
        raise common.CampaignError("SCBA terminal publication is incomplete; no rerun was attempted")
    outcome = _load_json(output_json, "SCBA outcome")
    if outcome.get("campaign_manifest_sha256") != manifest_sha or not outcome.get("scba_executed"):
        raise common.CampaignError("SCBA outcome is not bound to the immutable campaign")
    converged = bool(outcome.get("result", {}).get("converged", False))
    state = "passed" if converged else "scientific-failure"
    report = {
        "schema": REPORT_SCHEMA, "campaign_manifest_sha256": manifest_sha,
        "state": state, "scba_converged": converged,
        "checkpoint_sha256": _sha256(checkpoint), "outcome_sha256": _sha256(output_json),
        "fields_hdf5_sha256": _sha256(output_hdf5), "log_sha256": _sha256(log_path),
        "supervision": log.get("supervision", {}), "automatic_retry": False,
        "controls_changed": False, "release_certified": False,
        "next_gate": "refine frequency/cutoff/eta and validate conservation" if converged else "inspect fixed-control scientific failure",
    }
    report_path = artifacts / "scba-report.json"
    common._ensure_exact(report_path, _canonical(report))
    prior_failure = artifacts / "infrastructure-failure.json"
    files = (manifest_path, running_path, log_path, output_json, output_hdf5, checkpoint, report_path) + ((prior_failure,) if prior_failure.exists() else ())
    sums = common._write_checksums(artifacts, files)
    status = {
        "schema": STATUS_SCHEMA, "campaign_manifest_sha256": manifest_sha, "state": state,
        "scba_converged": converged, "release_certified": False,
        "report_sha256": _sha256(report_path), "sha256sums_sha256": common._sha256_bytes(sums),
        "automatic_retry": False,
    }
    common._ensure_exact(artifacts / "status.json", _canonical(status))
    return status


def _infrastructure_failure(artifacts: Path, manifest_path: Path, log_path: Path, detail: str, supervision: Mapping[str, Any]) -> dict[str, Any]:
    manifest_sha = _sha256(manifest_path)
    failure = {"schema": "negfpy.sparse-nq1-phase-c-scba-infrastructure-failure.v1", "campaign_manifest_sha256": manifest_sha, "classification": "infrastructure-failure", "detail": detail, "supervision": dict(supervision), "automatic_retry": False, "controls_changed": False}
    failure_path = artifacts / "infrastructure-failure.json"
    common._publish_once(failure_path, _canonical(failure))
    interrupted_sums_path = artifacts / "SHA256SUMS.infrastructure-failure"
    sums_content = "".join(
        f"{_sha256(path)}  {path.relative_to(artifacts).as_posix()}\n"
        for path in sorted((manifest_path, log_path, failure_path), key=lambda item: item.relative_to(artifacts).as_posix())
    ).encode("ascii")
    common._publish_once(interrupted_sums_path, sums_content)
    # No terminal status is published: the immutable checkpoint remains
    # resumable.  A later invocation must explicitly pass ``--resume``.
    status = {"schema": STATUS_SCHEMA, "campaign_manifest_sha256": manifest_sha, "state": "infrastructure-interrupted", "failure_sha256": _sha256(failure_path), "sha256sums_sha256": common._sha256_bytes(sums_content), "automatic_retry": False, "resume_required": True, "release_certified": False}
    return status


def run_campaign(*, repo: Path, fixture: Path, fc3_full: Path, phase_b: Path, artifacts: Path, expected_commit: str, environment_lock: Path, run_id: str, coupling_scale: float, initial_checkpoint: Path | None = None, resume: bool = False) -> dict[str, Any]:
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "campaign-manifest.json"
    status_path = artifacts / "status.json"
    interruption_path = artifacts / "infrastructure-failure.json"
    running_path = artifacts / "status-running.json"
    log_path = artifacts / "logs" / "scba.json"
    if status_path.exists():
        return _load_json(status_path, "terminal SCBA status")
    if interruption_path.exists() and not resume:
        raise common.CampaignError("an infrastructure interruption is archived; pass --resume to continue the identical checkpoint")
    if manifest_path.exists():
        manifest = _load_json(manifest_path, "SCBA campaign manifest")
        if manifest.get("source") != common._source_identity(repo, expected_commit):
            raise common.CampaignError("source changed during SCBA resume")
        if manifest.get("environment_lock") != common._validate_lock(environment_lock):
            raise common.CampaignError("environment lock changed during SCBA resume")
        if manifest.get("driver", {}).get("sha256") != _sha256(repo / "scripts" / "validate_sparse_nq1_phase_c_scba.py"):
            raise common.CampaignError("SCBA driver changed during resume")
        if manifest.get("coupling_scale") != coupling_scale:
            raise common.CampaignError("coupling scale changed during SCBA resume")
        supplied_parent = (
            None if initial_checkpoint is None else {
                "path": str(initial_checkpoint.resolve()),
                "sha256": _sha256(initial_checkpoint),
            }
        )
        if manifest.get("continuation_parent") != supplied_parent:
            raise common.CampaignError("continuation parent changed during SCBA resume")
    else:
        if any(artifacts.iterdir()):
            raise common.CampaignError("nonempty artifact directory has no manifest; refusing overwrite")
        manifest = _manifest(repo=repo, fixture=fixture, fc3_full=fc3_full, phase_b=phase_b, artifacts=artifacts, expected_commit=expected_commit, environment_lock=environment_lock, run_id=run_id, coupling_scale=coupling_scale, initial_checkpoint=initial_checkpoint)
        common._publish_once(manifest_path, _canonical(manifest))
    manifest_sha = _sha256(manifest_path)
    if not running_path.exists():
        common._publish_once(running_path, _canonical({"schema": STATUS_SCHEMA, "campaign_manifest_sha256": manifest_sha, "state": "running", "automatic_retry": False, "release_certified": False}))
    output = manifest["outputs"]
    command_parts = [sys.executable, str(repo / "scripts" / "validate_sparse_nq1_phase_c_scba.py"), "--fixture", str(fixture.resolve()), "--fc3-full", str(fc3_full.resolve()), "--phase-b-artifacts", str(phase_b.resolve()), "--checkpoint", str(Path(output["checkpoint"]).resolve()), "--output-json", str(Path(output["json"]).resolve()), "--output-hdf5", str(Path(output["hdf5"]).resolve()), "--campaign-manifest-sha256", manifest_sha, "--source-tree-sha256", manifest["source_tree_sha256"], "--environment-sha256", manifest["environment_lock"]["sha256"], "--run-id", manifest["run_id"], "--coupling-scale", str(manifest["coupling_scale"])]
    parent = manifest.get("continuation_parent")
    if parent is not None:
        command_parts.extend(["--initial-checkpoint", str(parent["path"])])
    command = tuple(command_parts)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path.with_suffix(".stdout.tmp")
    stderr_path = log_path.with_suffix(".stderr.tmp")
    supervision = common._supervised_subprocess(command, repo=repo, stdout_path=stdout_path, stderr_path=stderr_path)
    log = {"command": list(command), "supervision": supervision, "stdout": stdout_path.read_text(errors="replace"), "stderr": stderr_path.read_text(errors="replace")}
    common._publish_once(log_path, _canonical(log))
    stdout_path.unlink(missing_ok=True); stderr_path.unlink(missing_ok=True)
    if supervision["watchdog_trigger"] is not None or supervision["returncode"] not in {0, 2}:
        return _infrastructure_failure(artifacts, manifest_path, log_path, "watchdog or non-driver exit; checkpoint preserved for inspection", supervision)
    return _publish_report(artifacts, manifest_path, running_path, log_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--phase-b-artifacts", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--coupling-scale", type=float, required=True)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume an archived infrastructure interruption with identical hashes")
    args = parser.parse_args()
    try:
        if args.preflight_only:
            manifest = _manifest(repo=args.repo_root, fixture=args.fixture, fc3_full=args.fc3_full, phase_b=args.phase_b_artifacts, artifacts=args.artifacts, expected_commit=args.expected_commit, environment_lock=args.environment_lock, run_id=args.run_id, coupling_scale=args.coupling_scale, initial_checkpoint=args.initial_checkpoint)
            print(json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
            return 0
        status = run_campaign(repo=args.repo_root, fixture=args.fixture, fc3_full=args.fc3_full, phase_b=args.phase_b_artifacts, artifacts=args.artifacts, expected_commit=args.expected_commit, environment_lock=args.environment_lock, run_id=args.run_id, coupling_scale=args.coupling_scale, initial_checkpoint=args.initial_checkpoint, resume=args.resume)
        print(json.dumps(status, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        return 0 if status.get("state") == "passed" else 2
    except Exception as error:
        print(f"SCBA campaign infrastructure error: {type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
