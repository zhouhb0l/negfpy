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


# The material SCBA fixed point is substantially more expensive than the
# lightweight Born gate.  Keep its infrastructure watchdog explicit and
# independent while preserving the same RSS and termination safeguards.
SCBA_WALL_LIMIT_SECONDS = 12 * 60 * 60


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
    continuation_parent_artifacts: Path | None = None,
    continuation_axis: str | None = None,
    born_initial_hdf5: Path | None = None,
    grid_step: float | None = None,
    grid_n_positive: int | None = None,
    eta_lead: float | None = None,
    mixing: float | None = None,
    mixing_floor: float | None = None,
    temperature_bias: float | None = None,
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
        coupling_scale=coupling_scale,
        continuation_parent_artifacts=continuation_parent_artifacts,
        continuation_axis=continuation_axis,
        born_initial_hdf5=born_initial_hdf5,
        grid_step=grid_step, grid_n_positive=grid_n_positive,
        eta_lead=eta_lead, mixing=mixing, mixing_floor=mixing_floor,
        temperature_bias=temperature_bias,
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
        "continuation_parent": preflight["continuation_parent"],
        "continuation_axis": continuation_axis,
        "born_initial": preflight["born_initial"],
        "numerical_controls": preflight["numerical_controls"],
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


def _attempt_logs(artifacts: Path) -> tuple[Path, ...]:
    return tuple(sorted((artifacts / "logs").glob("scba*.json")))


def _failure_records(artifacts: Path) -> tuple[Path, ...]:
    return tuple(sorted(artifacts.glob("infrastructure-failure*.json")))


def _attempt_paths(artifacts: Path) -> tuple[int, Path, Path, Path]:
    attempt = len(_attempt_logs(artifacts)) + 1
    suffix = "" if attempt == 1 else f"-resume-{attempt - 1:03d}"
    return (
        attempt,
        artifacts / "logs" / f"scba{suffix}.json",
        artifacts / f"infrastructure-failure{suffix}.json",
        artifacts / f"SHA256SUMS.infrastructure-failure{suffix}",
    )


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
    attempt_logs = _attempt_logs(artifacts)
    failure_records = _failure_records(artifacts)
    report = {
        "schema": REPORT_SCHEMA, "campaign_manifest_sha256": manifest_sha,
        "state": state, "scba_converged": converged,
        "checkpoint_sha256": _sha256(checkpoint), "outcome_sha256": _sha256(output_json),
        "fields_hdf5_sha256": _sha256(output_hdf5), "log_sha256": _sha256(log_path),
        "supervision": log.get("supervision", {}), "automatic_retry": False,
        "attempt_logs": [
            {
                "path": path.relative_to(artifacts).as_posix(),
                "sha256": _sha256(path),
            }
            for path in attempt_logs
        ],
        "infrastructure_failures": [
            {
                "path": path.relative_to(artifacts).as_posix(),
                "sha256": _sha256(path),
            }
            for path in failure_records
        ],
        "controls_changed": False, "release_certified": False,
        "next_gate": "refine frequency/cutoff/eta and validate conservation" if converged else "inspect fixed-control scientific failure",
    }
    report_path = artifacts / "scba-report.json"
    common._ensure_exact(report_path, _canonical(report))
    files = (
        (manifest_path, running_path, output_json, output_hdf5, checkpoint, report_path)
        + attempt_logs
        + failure_records
    )
    sums = common._write_checksums(artifacts, files)
    status = {
        "schema": STATUS_SCHEMA, "campaign_manifest_sha256": manifest_sha, "state": state,
        "scba_converged": converged, "release_certified": False,
        "report_sha256": _sha256(report_path), "sha256sums_sha256": common._sha256_bytes(sums),
        "automatic_retry": False,
    }
    common._ensure_exact(artifacts / "status.json", _canonical(status))
    return status


def _infrastructure_failure(
    artifacts: Path,
    manifest_path: Path,
    log_path: Path,
    failure_path: Path,
    interrupted_sums_path: Path,
    detail: str,
    supervision: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_sha = _sha256(manifest_path)
    failure = {"schema": "negfpy.sparse-nq1-phase-c-scba-infrastructure-failure.v1", "campaign_manifest_sha256": manifest_sha, "classification": "infrastructure-failure", "detail": detail, "supervision": dict(supervision), "automatic_retry": False, "controls_changed": False}
    common._publish_once(failure_path, _canonical(failure))
    sums_content = "".join(
        f"{_sha256(path)}  {path.relative_to(artifacts).as_posix()}\n"
        for path in sorted((manifest_path, log_path, failure_path), key=lambda item: item.relative_to(artifacts).as_posix())
    ).encode("ascii")
    common._publish_once(interrupted_sums_path, sums_content)
    # No terminal status is published: the immutable checkpoint remains
    # resumable.  A later invocation must explicitly pass ``--resume``.
    status = {"schema": STATUS_SCHEMA, "campaign_manifest_sha256": manifest_sha, "state": "infrastructure-interrupted", "failure_sha256": _sha256(failure_path), "sha256sums_sha256": common._sha256_bytes(sums_content), "automatic_retry": False, "resume_required": True, "release_certified": False}
    return status


def run_campaign(*, repo: Path, fixture: Path, fc3_full: Path, phase_b: Path, artifacts: Path, expected_commit: str, environment_lock: Path, run_id: str, coupling_scale: float, continuation_parent_artifacts: Path | None = None, continuation_axis: str | None = None, born_initial_hdf5: Path | None = None, grid_step: float | None = None, grid_n_positive: int | None = None, eta_lead: float | None = None, mixing: float | None = None, mixing_floor: float | None = None, temperature_bias: float | None = None, resume: bool = False) -> dict[str, Any]:
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "campaign-manifest.json"
    status_path = artifacts / "status.json"
    running_path = artifacts / "status-running.json"
    if status_path.exists():
        return _load_json(status_path, "terminal SCBA status")
    prior_failures = _failure_records(artifacts)
    if prior_failures and not resume:
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
        if manifest.get("continuation_axis") != continuation_axis:
            raise common.CampaignError("continuation axis changed during SCBA resume")
        _resume_identity, _resume_problem, _resume_config = case.authenticate_scba_inputs(
            fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b,
            coupling_scale=coupling_scale, grid_step=grid_step,
            grid_n_positive=grid_n_positive, eta_lead=eta_lead,
            mixing=mixing, mixing_floor=mixing_floor,
            temperature_bias=temperature_bias,
        )
        observed_controls = {
            "grid_step": float(_resume_problem.grid.step),
            "grid_n_positive": int(_resume_problem.grid.n_positive),
            "grid_cutoff": float(_resume_problem.grid.cutoff),
            "eta_lead": float(_resume_config.eta_lead),
            "mixing": float(_resume_config.mixing),
            "mixing_floor": float(_resume_config.mixing_floor),
            "temperature_bias": float(
                _resume_problem.left.temperature - _resume_problem.right.temperature
            ),
        }
        if manifest.get("numerical_controls") != observed_controls:
            raise common.CampaignError("numerical controls changed during SCBA resume")
        if continuation_parent_artifacts is None:
            supplied_parent = None
        else:
            parent_identity, parent_problem, parent_config = case.authenticate_scba_inputs(
                fixture=fixture, fc3_full=fc3_full,
                phase_b_artifacts=phase_b, coupling_scale=coupling_scale,
                grid_step=grid_step, grid_n_positive=grid_n_positive,
                eta_lead=eta_lead, mixing=mixing, mixing_floor=mixing_floor,
                temperature_bias=temperature_bias,
            )
            _initial, supplied_parent = case._continuation_initial_state(
                parent_artifacts=continuation_parent_artifacts,
                identity=parent_identity, problem=parent_problem,
                config=parent_config,
                source_tree_sha256=manifest["source_tree_sha256"],
                environment_sha256=manifest["environment_lock"]["sha256"],
                child_coupling_scale=coupling_scale,
                continuation_axis=continuation_axis,
            )
        if manifest.get("continuation_parent") != supplied_parent:
            raise common.CampaignError("continuation parent changed during SCBA resume")
        if born_initial_hdf5 is None:
            supplied_born = None
        else:
            _initial, _parent, supplied_born = case._authenticated_initial_state(
                continuation_parent_artifacts=None,
                continuation_axis=None,
                born_initial_hdf5=born_initial_hdf5,
                identity=_resume_identity,
                problem=_resume_problem,
                config=_resume_config,
                source_tree_sha256=manifest["source_tree_sha256"],
                environment_sha256=manifest["environment_lock"]["sha256"],
                coupling_scale=coupling_scale,
            )
        if manifest.get("born_initial") != supplied_born:
            raise common.CampaignError("Born initial-state artifact changed during SCBA resume")
    else:
        if any(artifacts.iterdir()):
            raise common.CampaignError("nonempty artifact directory has no manifest; refusing overwrite")
        manifest = _manifest(repo=repo, fixture=fixture, fc3_full=fc3_full, phase_b=phase_b, artifacts=artifacts, expected_commit=expected_commit, environment_lock=environment_lock, run_id=run_id, coupling_scale=coupling_scale, continuation_parent_artifacts=continuation_parent_artifacts, continuation_axis=continuation_axis, born_initial_hdf5=born_initial_hdf5, grid_step=grid_step, grid_n_positive=grid_n_positive, eta_lead=eta_lead, mixing=mixing, mixing_floor=mixing_floor, temperature_bias=temperature_bias)
        common._publish_once(manifest_path, _canonical(manifest))
    manifest_sha = _sha256(manifest_path)
    if not running_path.exists():
        common._publish_once(running_path, _canonical({"schema": STATUS_SCHEMA, "campaign_manifest_sha256": manifest_sha, "state": "running", "automatic_retry": False, "release_certified": False}))
    output = manifest["outputs"]
    attempt, log_path, failure_path, interrupted_sums_path = _attempt_paths(artifacts)
    command_parts = [sys.executable, str(repo / "scripts" / "validate_sparse_nq1_phase_c_scba.py"), "--fixture", str(fixture.resolve()), "--fc3-full", str(fc3_full.resolve()), "--phase-b-artifacts", str(phase_b.resolve()), "--checkpoint", str(Path(output["checkpoint"]).resolve()), "--output-json", str(Path(output["json"]).resolve()), "--output-hdf5", str(Path(output["hdf5"]).resolve()), "--campaign-manifest-sha256", manifest_sha, "--source-tree-sha256", manifest["source_tree_sha256"], "--environment-sha256", manifest["environment_lock"]["sha256"], "--run-id", manifest["run_id"], "--coupling-scale", str(manifest["coupling_scale"])]
    numerical = manifest["numerical_controls"]
    command_parts.extend([
        "--grid-step", str(numerical["grid_step"]),
        "--grid-n-positive", str(numerical["grid_n_positive"]),
        "--eta-lead", str(numerical["eta_lead"]),
        "--mixing", str(numerical["mixing"]),
        "--mixing-floor", str(numerical["mixing_floor"]),
        "--temperature-bias", str(numerical["temperature_bias"]),
    ])
    parent = manifest.get("continuation_parent")
    if parent is not None:
        command_parts.extend([
            "--continuation-parent-artifacts", str(parent["artifacts"]),
            "--continuation-axis", str(manifest["continuation_axis"]),
        ])
    born_initial = manifest.get("born_initial")
    if born_initial is not None:
        command_parts.extend([
            "--born-initial-hdf5", str(born_initial["path"])
        ])
    command = tuple(command_parts)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path.with_suffix(".stdout.tmp")
    stderr_path = log_path.with_suffix(".stderr.tmp")
    supervision = common._supervised_subprocess(command, repo=repo, stdout_path=stdout_path, stderr_path=stderr_path)
    log = {
        "attempt": attempt,
        "resume": bool(resume),
        "command": list(command),
        "supervision": supervision,
        "stdout": stdout_path.read_text(errors="replace"),
        "stderr": stderr_path.read_text(errors="replace"),
    }
    common._publish_once(log_path, _canonical(log))
    stdout_path.unlink(missing_ok=True); stderr_path.unlink(missing_ok=True)
    if supervision["watchdog_trigger"] is not None or supervision["returncode"] not in {0, 2}:
        return _infrastructure_failure(
            artifacts, manifest_path, log_path, failure_path,
            interrupted_sums_path,
            "watchdog or non-driver exit; checkpoint preserved for inspection",
            supervision,
        )
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
    parser.add_argument("--continuation-parent-artifacts", type=Path)
    parser.add_argument(
        "--continuation-axis", choices=("coupling", "temperature_bias")
    )
    parser.add_argument("--born-initial-hdf5", type=Path)
    parser.add_argument("--grid-step", type=float)
    parser.add_argument("--grid-n-positive", type=int)
    parser.add_argument("--eta-lead", type=float)
    parser.add_argument("--mixing", type=float)
    parser.add_argument("--mixing-floor", type=float)
    parser.add_argument("--temperature-bias", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume an archived infrastructure interruption with identical hashes")
    args = parser.parse_args()
    try:
        # Apply the material-SCBA watchdog only when this campaign is run;
        # importing the shared Born module must retain its 2-hour contract.
        common.WALL_LIMIT_SECONDS = SCBA_WALL_LIMIT_SECONDS
        if args.preflight_only:
            manifest = _manifest(repo=args.repo_root, fixture=args.fixture, fc3_full=args.fc3_full, phase_b=args.phase_b_artifacts, artifacts=args.artifacts, expected_commit=args.expected_commit, environment_lock=args.environment_lock, run_id=args.run_id, coupling_scale=args.coupling_scale, continuation_parent_artifacts=args.continuation_parent_artifacts, continuation_axis=args.continuation_axis, born_initial_hdf5=args.born_initial_hdf5, grid_step=args.grid_step, grid_n_positive=args.grid_n_positive, eta_lead=args.eta_lead, mixing=args.mixing, mixing_floor=args.mixing_floor, temperature_bias=args.temperature_bias)
            print(json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
            return 0
        status = run_campaign(repo=args.repo_root, fixture=args.fixture, fc3_full=args.fc3_full, phase_b=args.phase_b_artifacts, artifacts=args.artifacts, expected_commit=args.expected_commit, environment_lock=args.environment_lock, run_id=args.run_id, coupling_scale=args.coupling_scale, continuation_parent_artifacts=args.continuation_parent_artifacts, continuation_axis=args.continuation_axis, born_initial_hdf5=args.born_initial_hdf5, grid_step=args.grid_step, grid_n_positive=args.grid_n_positive, eta_lead=args.eta_lead, mixing=args.mixing, mixing_floor=args.mixing_floor, temperature_bias=args.temperature_bias, resume=args.resume)
        print(json.dumps(status, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        return 0 if status.get("state") == "passed" else 2
    except Exception as error:
        print(f"SCBA campaign infrastructure error: {type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
