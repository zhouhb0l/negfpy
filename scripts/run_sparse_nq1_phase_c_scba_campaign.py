#!/usr/bin/env python3
"""Checkpointed, fixed-control campaign wrapper for one material SCBA case.

The wrapper is intentionally conservative: a manifest is immutable, a
non-terminal attempt is resumed only with the same hashes, and no scientific
or infrastructure failure is retried with changed controls.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import run_sparse_nq1_phase_c_born_campaign as common
import validate_sparse_nq1_phase_c_scba as case

# The q4 candidate requests one 128-CPU, 440-GiB node for 24 hours.  Leave one
# hour for termination and immutable outcome publication before PBS expiry,
# and 10 GiB for runtime/OS overhead outside the supervised process group.
# These infrastructure limits are separate from SCBA's scientific controls.
# Born retains its independent two-hour gate.
SCBA_WALL_LIMIT_SECONDS = 23 * 60 * 60
SCBA_RSS_LIMIT_BYTES = 430 * 1024**3

# A segmented pilot stops the scientific child early enough to authenticate
# and publish its checkpoint before the fixed q4 24-hour scheduler limit.  A
# later segment is a separately submitted, explicitly authorized job; it is
# never launched automatically by this wrapper.
SCBA_SEGMENT_WALL_LIMIT_SECONDS = 20 * 60 * 60
SCBA_MAX_SEGMENTS = 4
SEGMENT_SCHEMA = "negfpy.sparse-nq1-phase-c-scba-segment.v1"


SCHEMA = "negfpy.sparse-nq1-phase-c-scba-campaign.v2"
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
    continuation_predictor: str | None = None,
    born_initial_hdf5: Path | None = None,
    grid_step: float | None = None,
    grid_n_positive: int | None = None,
    eta_lead: float | None = None,
    mixing: float | None = None,
    mixing_floor: float | None = None,
    temperature_bias: float | None = None,
    segmented: bool = False,
    allow_existing_outputs: bool = False,
    allow_parent_source_tree_mismatch: bool = False,
    research_scout: bool = False,
) -> dict[str, Any]:
    if research_scout and (segmented or allow_existing_outputs):
        raise common.CampaignError("research scouts are fresh, nonresumable and nonsegmented")
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
        continuation_predictor=continuation_predictor,
        born_initial_hdf5=born_initial_hdf5,
        grid_step=grid_step, grid_n_positive=grid_n_positive,
        eta_lead=eta_lead, mixing=mixing, mixing_floor=mixing_floor,
        temperature_bias=temperature_bias,
        allow_existing_outputs=allow_existing_outputs,
        allow_parent_source_tree_mismatch=allow_parent_source_tree_mismatch,
        research_scout=research_scout,
    )
    driver = repo / "scripts" / "validate_sparse_nq1_phase_c_scba.py"
    if not driver.is_file() or driver.is_symlink():
        raise common.CampaignError(f"missing SCBA case driver: {driver}")
    document = {
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
        "continuation_predictor": continuation_predictor,
        "allow_parent_source_tree_mismatch": allow_parent_source_tree_mismatch,
        "born_initial": preflight["born_initial"],
        "numerical_controls": preflight["numerical_controls"],
        "fixed_controls_no_retuning": True,
        "restart": {"checkpoint_schema": case.CHECKPOINT_SCHEMA, "resume_exact_identity": True, "automatic_retry": False},
        "watchdog": {"wall_limit_seconds": SCBA_WALL_LIMIT_SECONDS, "rss_limit_bytes": SCBA_RSS_LIMIT_BYTES, "term_grace_seconds": common.TERM_GRACE_SECONDS, "automatic_retry": False},
        "outputs": {"checkpoint": str(checkpoint), "json": str(output_json), "hdf5": str(output_hdf5)},
        "python": {"executable": str(Path(sys.executable).resolve()), "version": platform.python_version()},
        "distributions": common._versions(),
        "thread_limits": dict(sorted(case.phase_c.THREAD_LIMIT_ENV.items())),
    }
    if research_scout:
        document["scientific_scope"] = "bounded Anderson research scout; no pilot/release acceptance"
        document["research_scout"] = preflight["research_scout"]
        document["restart"]["resume_exact_identity"] = False
        document["restart"]["resume_allowed"] = False
        document["watchdog"]["wall_limit_seconds"] = case.RESEARCH_SCOUT_WALL_SECONDS
    if segmented:
        document["segmentation"] = {
            "enabled": True,
            "max_segments": SCBA_MAX_SEGMENTS,
            "segment_wall_limit_seconds": SCBA_SEGMENT_WALL_LIMIT_SECONDS,
            "planned_stop_trigger": "wall-timeout",
            "automatic_advance": False,
        }
    return document


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return common._load_canonical(path, name=label)


def _attempt_logs(artifacts: Path) -> tuple[Path, ...]:
    initial = artifacts / "logs" / "scba.json"
    resumes = tuple(sorted((artifacts / "logs").glob("scba-resume-*.json")))
    return ((initial,) if initial.is_file() else ()) + resumes


def _failure_records(artifacts: Path) -> tuple[Path, ...]:
    return tuple(sorted(artifacts.glob("infrastructure-failure*.json")))


def _segment_records(artifacts: Path) -> tuple[Path, ...]:
    return tuple(sorted((artifacts / "segments").glob("segment-*.json")))


def _load_segment_checkpoint(path: Path) -> Any:
    from negfpy.anharmonic._checkpoint_v2 import (
        load_scba_checkpoint_v2_self_validating,
    )

    return load_scba_checkpoint_v2_self_validating(path)


def _validate_segment_chain(
    artifacts: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    require_current_checkpoint: bool = True,
) -> tuple[Path, ...]:
    records = _segment_records(artifacts)
    manifest_sha = _sha256(manifest_path)
    previous: dict[str, str] | None = None
    latest: Mapping[str, Any] | None = None
    for index, path in enumerate(records, start=1):
        if path.name != f"segment-{index:03d}.json":
            raise common.CampaignError("planned segment receipt names are not contiguous")
        record = _load_json(path, f"planned segment receipt {index}")
        if (
            record.get("schema") != SEGMENT_SCHEMA
            or record.get("campaign_manifest_sha256") != manifest_sha
            or record.get("segment_index") != index
            or record.get("resume") is not (index > 1)
            or record.get("state") != "segment-exhausted"
            or record.get("previous_segment") != previous
            or record.get("resume_required") is not True
            or record.get("automatic_advance") is not False
            or record.get("controls_changed") is not False
            or record.get("release_certified") is not False
        ):
            raise common.CampaignError("planned segment receipt chain is invalid")
        checkpoint = record.get("checkpoint")
        attempt_log = record.get("attempt_log")
        if not isinstance(checkpoint, Mapping) or not isinstance(
            attempt_log, Mapping
        ):
            raise common.CampaignError("planned segment receipt evidence is incomplete")
        log_path = artifacts / str(attempt_log.get("path"))
        _regular_file(log_path, "planned segment attempt log")
        if attempt_log.get("sha256") != _sha256(log_path):
            raise common.CampaignError("planned segment attempt log changed")
        expected_log = (
            "logs/scba.json"
            if index == 1
            else f"logs/scba-resume-{index - 1:03d}.json"
        )
        if attempt_log.get("path") != expected_log:
            raise common.CampaignError("planned segment attempt log path changed")
        if (
            checkpoint.get("path") != "checkpoint/sigma-pp.h5"
            or checkpoint.get("identity_sha256")
            != manifest.get("preflight", {}).get("checkpoint_identity_sha256")
            or checkpoint.get("status")
            not in {"in_progress", "infrastructure_interrupted"}
            or isinstance(checkpoint.get("completed_iterations"), bool)
            or not isinstance(checkpoint.get("completed_iterations"), int)
            or checkpoint.get("completed_iterations", 0) <= 0
        ):
            raise common.CampaignError("planned segment checkpoint evidence is invalid")
        if latest is not None and checkpoint["completed_iterations"] <= latest[
            "completed_iterations"
        ]:
            raise common.CampaignError("planned segment made no iteration progress")
        latest = checkpoint
        previous = {
            "path": path.relative_to(artifacts).as_posix(),
            "sha256": _sha256(path),
        }

    if latest is not None and require_current_checkpoint:
        checkpoint_path = artifacts / "checkpoint" / "sigma-pp.h5"
        _regular_file(checkpoint_path, "planned segment resume checkpoint")
        if latest.get("sha256") != _sha256(checkpoint_path):
            raise common.CampaignError("planned segment resume checkpoint changed")
        try:
            loaded = _load_segment_checkpoint(checkpoint_path)
        except Exception as error:
            raise common.CampaignError(
                "planned segment resume checkpoint is not self-validating"
            ) from error
        if (
            loaded.terminal
            or loaded.identity.identity_sha256 != latest.get("identity_sha256")
            or len(loaded.history) != latest.get("completed_iterations")
        ):
            raise common.CampaignError("planned segment resume checkpoint differs")
    return records


def _publish_segment_stop(
    artifacts: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    log_path: Path,
    *,
    attempt: int,
    resume: bool,
    supervision: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and record one planned, nonterminal scheduler segment."""

    segmentation = manifest.get("segmentation")
    if (
        not isinstance(segmentation, Mapping)
        or segmentation != {
            "enabled": True,
            "max_segments": SCBA_MAX_SEGMENTS,
            "segment_wall_limit_seconds": SCBA_SEGMENT_WALL_LIMIT_SECONDS,
            "planned_stop_trigger": "wall-timeout",
            "automatic_advance": False,
        }
    ):
        raise common.CampaignError("planned segment lacks its immutable contract")
    if (
        supervision.get("watchdog_trigger") != "wall-timeout"
        or supervision.get("wall_limit_seconds")
        != SCBA_SEGMENT_WALL_LIMIT_SECONDS
        or supervision.get("rss_limit_bytes") != SCBA_RSS_LIMIT_BYTES
    ):
        raise common.CampaignError("planned segment stopped for an unexpected reason")
    killed_after_grace = supervision.get("killed_after_term_grace")
    expected_returncode = -9 if killed_after_grace is True else -15
    if (
        type(killed_after_grace) is not bool
        or supervision.get("returncode") != expected_returncode
    ):
        raise common.CampaignError("planned segment termination is ambiguous")
    if attempt >= SCBA_MAX_SEGMENTS:
        raise common.CampaignError(
            "planned SCBA segment budget exhausted without a scientific outcome"
        )

    checkpoint_path = Path(manifest["outputs"]["checkpoint"])
    _regular_file(checkpoint_path, "planned segment checkpoint")
    try:
        checkpoint = _load_segment_checkpoint(checkpoint_path)
    except Exception as error:
        raise common.CampaignError(
            "planned segment checkpoint is not self-validating"
        ) from error
    if checkpoint.terminal or checkpoint.status not in {
        "in_progress",
        "infrastructure_interrupted",
    }:
        raise common.CampaignError("planned segment checkpoint is not resumable")
    identity = checkpoint.identity
    preflight = manifest.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or identity.run_id != manifest.get("run_id")
        or identity.source_tree_sha256 != manifest.get("source_tree_sha256")
        or identity.environment_sha256
        != manifest.get("environment_lock", {}).get("sha256")
        or identity.identity_sha256
        != preflight.get("checkpoint_identity_sha256")
    ):
        raise common.CampaignError("planned segment checkpoint identity changed")

    records = _validate_segment_chain(
        artifacts,
        manifest,
        manifest_path,
        require_current_checkpoint=False,
    )
    if len(records) != attempt - 1:
        raise common.CampaignError("planned segment receipt chain is incomplete")
    previous = None
    if records:
        previous = {
            "path": records[-1].relative_to(artifacts).as_posix(),
            "sha256": _sha256(records[-1]),
        }
    record = {
        "schema": SEGMENT_SCHEMA,
        "campaign_manifest_sha256": _sha256(manifest_path),
        "segment_index": attempt,
        "resume": resume,
        "state": "segment-exhausted",
        "checkpoint": {
            "path": checkpoint_path.relative_to(artifacts).as_posix(),
            "sha256": _sha256(checkpoint_path),
            "identity_sha256": identity.identity_sha256,
            "status": checkpoint.status,
            "completed_iterations": len(checkpoint.history),
        },
        "attempt_log": {
            "path": log_path.relative_to(artifacts).as_posix(),
            "sha256": _sha256(log_path),
        },
        "previous_segment": previous,
        "supervision": dict(supervision),
        "resume_required": True,
        "automatic_advance": False,
        "controls_changed": False,
        "release_certified": False,
    }
    segment_path = artifacts / "segments" / f"segment-{attempt:03d}.json"
    common._publish_once(segment_path, _canonical(record))
    return {
        "schema": STATUS_SCHEMA,
        "campaign_manifest_sha256": _sha256(manifest_path),
        "state": "segment-exhausted",
        "segment_index": attempt,
        "segment_receipt_sha256": _sha256(segment_path),
        "checkpoint_sha256": record["checkpoint"]["sha256"],
        "resume_required": True,
        "automatic_retry": False,
        "automatic_advance": False,
        "release_certified": False,
    }


def _manifest_authentication_view(value: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the explicitly non-reproducible free-memory observation."""

    try:
        stable = json.loads(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as error:
        raise common.CampaignError("SCBA manifest is not strict JSON") from error
    if not isinstance(stable, dict):  # pragma: no cover - Mapping API guard
        raise common.CampaignError("SCBA manifest is not an object")
    source = stable.get("preflight", {}).get("source")
    if isinstance(source, dict) and "memory" in source:
        memory = source["memory"]
        if not isinstance(memory, dict) or "available_bytes" not in memory:
            raise common.CampaignError(
                "SCBA preflight memory identity is incomplete"
            )
        available = memory["available_bytes"]
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or available <= 0
        ):
            raise common.CampaignError(
                "SCBA preflight available-memory observation is invalid"
            )
        memory["available_bytes"] = "runtime-observation"
    return stable


def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise common.CampaignError(f"{label} is not a regular file: {path}")
    return path


def _authenticated_terminal_status(
    artifacts: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    status_path: Path,
) -> dict[str, Any]:
    """Authenticate a terminal publication before treating it as idempotent."""

    manifest_sha = _sha256(manifest_path)
    status = _load_json(status_path, "terminal SCBA status")
    status_keys = {
        "schema",
        "campaign_manifest_sha256",
        "state",
        "scba_converged",
        "release_certified",
        "report_sha256",
        "sha256sums_sha256",
        "automatic_retry",
    }
    if set(status) != status_keys or status.get("schema") != STATUS_SCHEMA:
        raise common.CampaignError("terminal SCBA status contract is invalid")
    state = status.get("state")
    converged = status.get("scba_converged")
    if (
        state not in {"passed", "scientific-failure"}
        or type(converged) is not bool
        or converged != (state == "passed")
        or status.get("release_certified") is not False
        or status.get("automatic_retry") is not False
        or status.get("campaign_manifest_sha256") != manifest_sha
    ):
        raise common.CampaignError("terminal SCBA status identity is invalid")

    report_path = _regular_file(artifacts / "scba-report.json", "SCBA report")
    sums_path = _regular_file(artifacts / "SHA256SUMS", "SCBA SHA256SUMS")
    if (
        status.get("report_sha256") != _sha256(report_path)
        or status.get("sha256sums_sha256") != _sha256(sums_path)
    ):
        raise common.CampaignError("terminal SCBA status digest binding differs")
    report = _load_json(report_path, "SCBA report")
    report_keys = {
        "schema",
        "campaign_manifest_sha256",
        "state",
        "scba_converged",
        "checkpoint_sha256",
        "outcome_sha256",
        "fields_hdf5_sha256",
        "log_sha256",
        "supervision",
        "automatic_retry",
        "attempt_logs",
        "infrastructure_failures",
        "controls_changed",
        "release_certified",
        "next_gate",
    }
    if (
        set(report) != report_keys
        or report.get("schema") != REPORT_SCHEMA
        or report.get("campaign_manifest_sha256") != manifest_sha
        or report.get("state") != state
        or report.get("scba_converged") is not converged
        or report.get("automatic_retry") is not False
        or report.get("controls_changed") is not False
        or report.get("release_certified") is not False
    ):
        raise common.CampaignError("terminal SCBA report contract is invalid")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise common.CampaignError("SCBA manifest output paths are invalid")
    try:
        checkpoint = _regular_file(
            Path(outputs["checkpoint"]), "terminal SCBA checkpoint"
        )
        output_json = _regular_file(
            Path(outputs["json"]), "terminal SCBA outcome"
        )
        output_hdf5 = _regular_file(
            Path(outputs["hdf5"]), "terminal SCBA fields"
        )
    except (KeyError, TypeError) as error:
        raise common.CampaignError("SCBA manifest output paths are invalid") from error
    outcome = _load_json(output_json, "SCBA outcome")
    manifest_preflight = manifest.get("preflight")
    manifest_source = (
        manifest_preflight.get("source")
        if isinstance(manifest_preflight, Mapping)
        else None
    )
    outcome_source = outcome.get("preflight_source")
    try:
        source_matches = (
            isinstance(manifest_source, Mapping)
            and isinstance(outcome_source, Mapping)
            and case._stable_material_identity(outcome_source)
            == case._stable_material_identity(manifest_source)
        )
    except case.SCBACaseError as error:
        raise common.CampaignError(
            f"terminal SCBA material identity is invalid: {error}"
        ) from error
    outcome_checkpoint = outcome.get("checkpoint")
    if (
        outcome.get("campaign_manifest_sha256") != manifest_sha
        or outcome.get("scba_executed") is not True
        or outcome.get("release_certified") is not False
        or outcome.get("result", {}).get("converged") is not converged
        or not source_matches
        or outcome.get("continuation_parent")
        != manifest.get("continuation_parent")
        or outcome.get("born_initial") != manifest.get("born_initial")
        or not isinstance(outcome_checkpoint, str)
        or Path(outcome_checkpoint).expanduser().resolve() != checkpoint.resolve()
        or outcome.get("checkpoint_sha256") != _sha256(checkpoint)
        or outcome.get("hdf5", {}).get("sha256") != _sha256(output_hdf5)
        or report.get("checkpoint_sha256") != _sha256(checkpoint)
        or report.get("outcome_sha256") != _sha256(output_json)
        or report.get("fields_hdf5_sha256") != _sha256(output_hdf5)
    ):
        raise common.CampaignError("terminal SCBA output binding differs")

    attempt_logs = _attempt_logs(artifacts)
    failures = _failure_records(artifacts)
    for path in (*attempt_logs, *failures):
        _regular_file(path, "SCBA terminal inventory member")
    expected_attempts = [
        {"path": path.relative_to(artifacts).as_posix(), "sha256": _sha256(path)}
        for path in attempt_logs
    ]
    expected_failures = [
        {"path": path.relative_to(artifacts).as_posix(), "sha256": _sha256(path)}
        for path in failures
    ]
    if (
        report.get("attempt_logs") != expected_attempts
        or report.get("infrastructure_failures") != expected_failures
    ):
        raise common.CampaignError("terminal SCBA attempt inventory differs")
    matching_logs = [
        path for path in attempt_logs if _sha256(path) == report.get("log_sha256")
    ]
    if len(matching_logs) != 1:
        raise common.CampaignError("terminal SCBA report log binding is ambiguous")
    terminal_log = _load_json(matching_logs[0], "terminal SCBA subprocess log")
    if terminal_log.get("supervision") != report.get("supervision"):
        raise common.CampaignError("terminal SCBA supervision binding differs")

    running_path = _regular_file(
        artifacts / "status-running.json", "SCBA running status"
    )
    running = _load_json(running_path, "SCBA running status")
    expected_running = {
        "schema": STATUS_SCHEMA,
        "campaign_manifest_sha256": manifest_sha,
        "state": "running",
        "automatic_retry": False,
        "release_certified": False,
    }
    if running != expected_running:
        raise common.CampaignError("SCBA running status binding differs")
    files = (
        (manifest_path, running_path, output_json, output_hdf5, checkpoint, report_path)
        + attempt_logs
        + failures
    )
    expected_sums = "".join(
        f"{_sha256(path)}  {path.relative_to(artifacts).as_posix()}\n"
        for path in sorted(
            files, key=lambda item: item.relative_to(artifacts).as_posix()
        )
    ).encode("ascii")
    if sums_path.read_bytes() != expected_sums:
        raise common.CampaignError("terminal SCBA SHA256SUMS inventory differs")
    return status


def _attempt_paths(artifacts: Path) -> tuple[int, Path, Path, Path]:
    attempt = len(_attempt_logs(artifacts)) + 1
    suffix = "" if attempt == 1 else f"-resume-{attempt - 1:03d}"
    return (
        attempt,
        artifacts / "logs" / f"scba{suffix}.json",
        artifacts / f"infrastructure-failure{suffix}.json",
        artifacts / f"SHA256SUMS.infrastructure-failure{suffix}",
    )


def _supervise_scba(
    command: tuple[str, ...],
    *,
    repo: Path,
    stdout_path: Path,
    stderr_path: Path,
    launch_environment: Mapping[str, str] | None = None,
    wall_limit_seconds: int = SCBA_WALL_LIMIT_SECONDS,
) -> dict[str, Any]:
    return common._supervised_subprocess(
        command,
        repo=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        wall_limit_seconds=wall_limit_seconds,
        rss_limit_bytes=SCBA_RSS_LIMIT_BYTES,
        **({"launch_environment": launch_environment} if launch_environment is not None else {}),
    )


def _authenticate_scout_checkpoint(manifest: Mapping[str, Any]) -> tuple[Any, str]:
    contract = {
        "enabled": True, "max_iterations": case.RESEARCH_SCOUT_MAX_ITERATIONS,
        "max_wall_seconds": case.RESEARCH_SCOUT_WALL_SECONDS,
        "resume_allowed": False, "automatic_advance": False,
    }
    if manifest.get("research_scout") != contract:
        raise common.CampaignError("research scout contract changed")
    path = _regular_file(Path(manifest["outputs"]["checkpoint"]), "scout checkpoint")
    before = _sha256(path)
    checkpoint = _load_segment_checkpoint(path)
    after = _sha256(path)
    identity = checkpoint.identity
    preflight = manifest["preflight"]
    controls = preflight.get("controls", {})
    if (
        before != after
        or identity.run_id != manifest.get("run_id")
        or identity.source_tree_sha256 != manifest.get("source_tree_sha256")
        or identity.environment_sha256 != manifest.get("environment_lock", {}).get("sha256")
        or identity.identity_sha256 != preflight.get("checkpoint_identity_sha256")
        or identity.configuration_sha256 != preflight.get("configuration_sha256")
        or controls.get("mixing_method") != "anderson"
        or controls.get("max_iterations") != case.RESEARCH_SCOUT_MAX_ITERATIONS
        or not 0 < len(checkpoint.history) <= case.RESEARCH_SCOUT_MAX_ITERATIONS
    ):
        raise common.CampaignError("research scout checkpoint identity/budget changed")
    return checkpoint, after


def _scout_iteration_summary(checkpoint: Any) -> dict[str, Any]:
    history = checkpoint.history
    return {
        "completed_iterations": len(history),
        "first": case._plain(history[0]),
        "last": case._plain(history[-1]),
        "checkpoint_status": checkpoint.status,
        "anderson": case._plain(getattr(checkpoint, "metadata", {}).get("anderson", {})),
    }


def _publish_scout_timeout(
    artifacts: Path, manifest: Mapping[str, Any], manifest_path: Path,
    log_path: Path, supervision: Mapping[str, Any],
) -> dict[str, Any]:
    """Record the planned diagnostic cap without claiming scientific success."""
    killed = supervision.get("killed_after_term_grace")
    if (
        supervision.get("watchdog_trigger") != "wall-timeout"
        or supervision.get("wall_limit_seconds") != case.RESEARCH_SCOUT_WALL_SECONDS
        or supervision.get("rss_limit_bytes") != SCBA_RSS_LIMIT_BYTES
        or type(killed) is not bool
        or supervision.get("returncode") != (-9 if killed else -15)
    ):
        raise common.CampaignError("research scout termination is ambiguous")
    checkpoint, digest = _authenticate_scout_checkpoint(manifest)
    if checkpoint.terminal or checkpoint.status not in {"in_progress", "infrastructure_interrupted"}:
        raise common.CampaignError("timed-out scout checkpoint has unexpected terminal state")
    if any(Path(manifest["outputs"][name]).exists() for name in ("json", "hdf5")):
        raise common.CampaignError("timed-out scout unexpectedly published final outputs")
    report = {
        "schema": REPORT_SCHEMA, "campaign_manifest_sha256": _sha256(manifest_path),
        "state": "scout-exhausted", "research_scout": True,
        "stop_reason": "wall-timeout", "scba_converged": False,
        "checkpoint_sha256": digest, "log_sha256": _sha256(log_path),
        "supervision": dict(supervision),
        "iteration_summary": _scout_iteration_summary(checkpoint),
        "automatic_retry": False, "resume_allowed": False,
        "automatic_advance": False, "controls_changed": False,
        "release_certified": False,
        "next_gate": "inspect closed history; no resume or automatic successor",
    }
    report_path = artifacts / "scba-report.json"
    common._publish_once(report_path, _canonical(report))
    sums = common._write_checksums(artifacts, (
        manifest_path, artifacts / "status-running.json", log_path,
        Path(manifest["outputs"]["checkpoint"]), report_path,
    ))
    status = {
        "schema": STATUS_SCHEMA, "campaign_manifest_sha256": _sha256(manifest_path),
        "state": "scout-exhausted", "research_scout": True,
        "stop_reason": "wall-timeout", "scba_converged": False,
        "report_sha256": _sha256(report_path),
        "sha256sums_sha256": common._sha256_bytes(sums),
        "automatic_retry": False, "resume_allowed": False,
        "automatic_advance": False, "release_certified": False,
    }
    common._publish_once(artifacts / "status.json", _canonical(status))
    return status


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
    manifest = _load_json(manifest_path, "SCBA manifest")
    scout = manifest.get("research_scout", {}).get("enabled") is True
    state = "passed" if converged else "scientific-failure"
    scout_reason = None
    if scout:
        checkpoint_state, _checkpoint_sha = _authenticate_scout_checkpoint(manifest)
        if converged:
            if checkpoint_state.status != "completed":
                raise common.CampaignError("scout convergence disagrees with checkpoint")
            state = "scout-converged"
        elif (
            checkpoint_state.status == "scientific_failure"
            and len(checkpoint_state.history) == case.RESEARCH_SCOUT_MAX_ITERATIONS
            and outcome.get("result", {}).get("scientific_error", {}).get("type") == "SCBAConvergenceError"
        ):
            state = "scout-exhausted"
            scout_reason = "iteration-limit"

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
        **({"research_scout": True, "stop_reason": scout_reason,
            "resume_allowed": False, "automatic_advance": False,
            "iteration_summary": _scout_iteration_summary(checkpoint_state),
            "next_gate": "inspect closed history and conservation before any new calculation"} if scout else {}),
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
        **({"research_scout": True, "stop_reason": scout_reason, "resume_allowed": False, "automatic_advance": False} if scout else {}),
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


def run_campaign(*, repo: Path, fixture: Path, fc3_full: Path, phase_b: Path, artifacts: Path, expected_commit: str, environment_lock: Path, run_id: str, coupling_scale: float, continuation_parent_artifacts: Path | None = None, continuation_axis: str | None = None, continuation_predictor: str | None = None, born_initial_hdf5: Path | None = None, grid_step: float | None = None, grid_n_positive: int | None = None, eta_lead: float | None = None, mixing: float | None = None, mixing_floor: float | None = None, temperature_bias: float | None = None, resume: bool = False, segmented: bool = False, launch_environment: Mapping[str, str] | None = None, allow_parent_source_tree_mismatch: bool = False, research_scout: bool = False) -> dict[str, Any]:
    artifacts = artifacts.resolve()
    if research_scout and (resume or segmented):
        raise common.CampaignError("research scouts cannot resume or use segmentation")
    if research_scout and artifacts.exists() and any(artifacts.iterdir()):
        raise common.CampaignError("research scout output is already consumed; use a new execution")
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "campaign-manifest.json"
    status_path = artifacts / "status.json"
    running_path = artifacts / "status-running.json"
    manifest_arguments = {
        "repo": repo,
        "fixture": fixture,
        "fc3_full": fc3_full,
        "phase_b": phase_b,
        "artifacts": artifacts,
        "expected_commit": expected_commit,
        "environment_lock": environment_lock,
        "run_id": run_id,
        "coupling_scale": coupling_scale,
        "continuation_parent_artifacts": continuation_parent_artifacts,
        "continuation_axis": continuation_axis,
        "continuation_predictor": continuation_predictor,
        "born_initial_hdf5": born_initial_hdf5,
        "grid_step": grid_step,
        "grid_n_positive": grid_n_positive,
        "eta_lead": eta_lead,
        "mixing": mixing,
        "mixing_floor": mixing_floor,
        "temperature_bias": temperature_bias,
        "segmented": segmented,
        "allow_parent_source_tree_mismatch": allow_parent_source_tree_mismatch,
        "research_scout": research_scout,
    }
    existing_manifest = manifest_path.exists() or manifest_path.is_symlink()
    if existing_manifest:
        manifest = _load_json(manifest_path, "SCBA campaign manifest")
        expected_manifest = _manifest(
            **manifest_arguments, allow_existing_outputs=True
        )
        if _manifest_authentication_view(manifest) != _manifest_authentication_view(
            expected_manifest
        ):
            raise common.CampaignError(
                "existing SCBA campaign differs from the exact requested identity"
            )
        if status_path.exists() or status_path.is_symlink():
            return _authenticated_terminal_status(
                artifacts, manifest, manifest_path, status_path
            )
        if not resume:
            raise common.CampaignError(
                "an existing nonterminal SCBA campaign requires explicit --resume"
            )
    else:
        if any(artifacts.iterdir()):
            raise common.CampaignError("nonempty artifact directory has no manifest; refusing overwrite")
        manifest = _manifest(**manifest_arguments)
        common._publish_once(manifest_path, _canonical(manifest))
    manifest_sha = _sha256(manifest_path)
    if not running_path.exists():
        common._publish_once(running_path, _canonical({"schema": STATUS_SCHEMA, "campaign_manifest_sha256": manifest_sha, "state": "running", "automatic_retry": False, "release_certified": False}))
    output = manifest["outputs"]
    attempt, log_path, failure_path, interrupted_sums_path = _attempt_paths(artifacts)
    if segmented:
        records = _validate_segment_chain(artifacts, manifest, manifest_path)
        if len(records) != attempt - 1:
            raise common.CampaignError("planned segment receipt chain is incomplete")
        if attempt > SCBA_MAX_SEGMENTS:
            raise common.CampaignError("planned SCBA segment budget is exhausted")
        if resume is not (attempt > 1):
            raise common.CampaignError(
                "planned segment requires fresh segment 1 and explicit resume thereafter"
            )
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
            "--continuation-predictor", str(
                manifest["continuation_predictor"]
            ),
        ])
        if manifest.get("allow_parent_source_tree_mismatch") is True:
            command_parts.append("--allow-parent-source-tree-mismatch")
    born_initial = manifest.get("born_initial")
    if born_initial is not None:
        command_parts.extend([
            "--born-initial-hdf5", str(born_initial["path"])
        ])
    if research_scout:
        command_parts.append("--research-scout")
    if not resume:
        command_parts.append("--no-resume")
    command = tuple(command_parts)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path.with_suffix(".stdout.tmp")
    stderr_path = log_path.with_suffix(".stderr.tmp")
    supervision = _supervise_scba(
        command,
        repo=repo,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        wall_limit_seconds=(
            case.RESEARCH_SCOUT_WALL_SECONDS
            if research_scout
            else SCBA_SEGMENT_WALL_LIMIT_SECONDS
            if segmented
            else SCBA_WALL_LIMIT_SECONDS
        ),
        **({"launch_environment": launch_environment} if launch_environment is not None else {}),
    )
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
    scout_stop_error = None
    if research_scout and supervision["watchdog_trigger"] == "wall-timeout":
        try:
            return _publish_scout_timeout(artifacts, manifest, manifest_path, log_path, supervision)
        except Exception as error:
            scout_stop_error = f"research scout stop authentication failed: {error}"
    if segmented and supervision["watchdog_trigger"] == "wall-timeout":
        try:
            return _publish_segment_stop(
                artifacts,
                manifest,
                manifest_path,
                log_path,
                attempt=attempt,
                resume=bool(resume),
                supervision=supervision,
            )
        except common.CampaignError as error:
            return _infrastructure_failure(
                artifacts,
                manifest_path,
                log_path,
                failure_path,
                interrupted_sums_path,
                f"planned segment authentication failed: {error}",
                supervision,
            )
    if supervision["watchdog_trigger"] is not None or supervision["returncode"] not in {0, 2}:
        failed = _infrastructure_failure(
            artifacts, manifest_path, log_path, failure_path,
            interrupted_sums_path,
            scout_stop_error or "watchdog or non-driver exit; checkpoint preserved for inspection",
            supervision,
        )
        if research_scout:
            failed.update({"state": "infrastructure-failed", "resume_required": False,
                           "resume_allowed": False, "research_scout": True})
            common._publish_once(status_path, _canonical(failed))
        return failed
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
    parser.add_argument(
        "--continuation-predictor",
        choices=tuple(sorted(case.CONTINUATION_PREDICTORS)),
    )
    parser.add_argument("--born-initial-hdf5", type=Path)
    parser.add_argument("--grid-step", type=float)
    parser.add_argument("--grid-n-positive", type=int)
    parser.add_argument("--eta-lead", type=float)
    parser.add_argument("--mixing", type=float)
    parser.add_argument("--mixing-floor", type=float)
    parser.add_argument("--temperature-bias", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--research-scout", action="store_true", help="fresh nonresumable Anderson g=.0175 trial capped at 100 iterations/3h")
    parser.add_argument("--resume", action="store_true", help="resume an archived infrastructure interruption with identical hashes")
    parser.add_argument(
        "--segmented",
        action="store_true",
        help="use the bounded explicit q4 scheduler-segmentation contract",
    )
    parser.add_argument("--allow-parent-source-tree-mismatch", action="store_true")
    args = parser.parse_args()
    try:
        if args.preflight_only:
            manifest = _manifest(repo=args.repo_root, fixture=args.fixture, fc3_full=args.fc3_full, phase_b=args.phase_b_artifacts, artifacts=args.artifacts, expected_commit=args.expected_commit, environment_lock=args.environment_lock, run_id=args.run_id, coupling_scale=args.coupling_scale, continuation_parent_artifacts=args.continuation_parent_artifacts, continuation_axis=args.continuation_axis, continuation_predictor=args.continuation_predictor, born_initial_hdf5=args.born_initial_hdf5, grid_step=args.grid_step, grid_n_positive=args.grid_n_positive, eta_lead=args.eta_lead, mixing=args.mixing, mixing_floor=args.mixing_floor, temperature_bias=args.temperature_bias, segmented=args.segmented, allow_parent_source_tree_mismatch=args.allow_parent_source_tree_mismatch, research_scout=args.research_scout)
            print(json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
            return 0
        status = run_campaign(repo=args.repo_root, fixture=args.fixture, fc3_full=args.fc3_full, phase_b=args.phase_b_artifacts, artifacts=args.artifacts, expected_commit=args.expected_commit, environment_lock=args.environment_lock, run_id=args.run_id, coupling_scale=args.coupling_scale, continuation_parent_artifacts=args.continuation_parent_artifacts, continuation_axis=args.continuation_axis, continuation_predictor=args.continuation_predictor, born_initial_hdf5=args.born_initial_hdf5, grid_step=args.grid_step, grid_n_positive=args.grid_n_positive, eta_lead=args.eta_lead, mixing=args.mixing, mixing_floor=args.mixing_floor, temperature_bias=args.temperature_bias, resume=args.resume, segmented=args.segmented, allow_parent_source_tree_mismatch=args.allow_parent_source_tree_mismatch, research_scout=args.research_scout)
        print(json.dumps(status, sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        return 0 if status.get("state") in {"passed", "segment-exhausted", "scout-converged", "scout-exhausted"} else 2
    except Exception as error:
        print(f"SCBA campaign infrastructure error: {type(error).__name__}: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
