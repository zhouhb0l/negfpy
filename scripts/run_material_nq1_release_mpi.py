#!/usr/bin/env python3
"""Execute the preregistered silicon Nq=1 SCBA graph with task-level MPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
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
from negfpy.anharmonic._mpi_tasks import TaskResult
from negfpy.anharmonic._provenance import code_provenance


RECEIPT_SCHEMA = "negfpy.material-nq1-mpi-task-receipt.v1"
REPORT_SCHEMA = "negfpy.material-nq1-mpi-execution-report.v1"
QUALIFICATION_SCHEMA = "negfpy.material-nq1-candidate-qualification.v1"
RELEASE_MIXING_FLOOR = 0.001


class MaterialCampaignError(RuntimeError):
    """Fail-closed material campaign error."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=True,
            separators=(",", ":"), sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


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
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MaterialCampaignError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict) or payload != _canonical(value):
        raise MaterialCampaignError(f"{label} is not canonical JSON")
    return value


def _validate_qualification(
    path: Path,
    *,
    source: Mapping[str, Any],
    source_tree_sha256: str,
    environment_sha256: str,
) -> dict[str, Any]:
    document = _load_canonical(path.expanduser().resolve(strict=True), "qualification report")
    expected = {
        "schema", "source", "source_tree_sha256", "environment_sha256",
        "checks", "evidence", "release_preflight_passed",
    }
    checks = document.get("checks")
    if (
        set(document) != expected
        or document.get("schema") != QUALIFICATION_SCHEMA
        or document.get("source") != source
        or document.get("source_tree_sha256") != source_tree_sha256
        or document.get("environment_sha256") != environment_sha256
        or not isinstance(checks, Mapping)
        or not checks
        or not all(value is True for value in checks.values())
        or document.get("release_preflight_passed") is not True
    ):
        raise MaterialCampaignError("candidate qualification does not authorize launch")
    return {
        "path": str(path.expanduser().resolve()),
        "sha256": _sha256(path.expanduser().resolve()),
        "checks": dict(checks),
    }


def _receipt_evidence(path: Path, spec: Any) -> dict[str, Any]:
    receipt = _load_canonical(path, "material task receipt")
    expected = {
        "schema", "campaign_manifest_sha256", "task_id",
        "task_identity_sha256", "scientific_task_sha256", "state",
        "task_directory", "task_tree_records", "task_tree_sha256",
        "scba_status_sha256", "scba_report_sha256", "checkpoint_sha256",
        "born_initial", "continuation_parent_task_id", "release_certified",
    }
    if set(receipt) != expected or receipt["schema"] != RECEIPT_SCHEMA:
        raise MaterialCampaignError("material task receipt schema is invalid")
    if receipt["task_id"] != spec.task_id:
        raise MaterialCampaignError("material task receipt names another task")
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
    if (
        status_sha != receipt["scba_status_sha256"]
        or report_sha != receipt["scba_report_sha256"]
        or checkpoint_sha != receipt["checkpoint_sha256"]
    ):
        raise MaterialCampaignError("material task terminal hashes changed")
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
        "mixing_floor": RELEASE_MIXING_FLOOR,
        "temperature_bias": scientific_payload.get("temperature_bias_solver"),
    }
    if scba_manifest.get("numerical_controls") != expected_controls:
        raise MaterialCampaignError("SCBA numerical controls changed")
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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    payload = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise MaterialCampaignError(f"immutable JSON differs: {path}")
    finally:
        temporary.unlink(missing_ok=True)
    return path


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
    qualification = _validate_qualification(
        args.qualification_report,
        source=source,
        source_tree_sha256=source_tree,
        environment_sha256=lock["sha256"],
    )
    campaign_root = args.artifacts.expanduser().resolve()
    campaign_root.mkdir(parents=True, exist_ok=True)
    base_task = next(
        task for task in silicon_nq1_release_tasks()
        if task.task_id == "coupling-g100"
    )
    # Authenticate the shared material inputs without executing a Born bubble.
    identity, base_problem, _config = case.authenticate_scba_inputs(
        fixture=args.fixture,
        fc3_full=args.fc3_full,
        phase_b_artifacts=args.phase_b_artifacts,
        coupling_scale=1.0,
        grid_step=base_task.grid_step,
        grid_n_positive=base_task.grid_n_positive,
        eta_lead=base_task.eta_lead,
        mixing=base_task.mixing,
        mixing_floor=RELEASE_MIXING_FLOOR,
        temperature_bias=base_task.temperature_bias_solver,
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
    input_hashes = case._source_hashes(identity) | {
        "observation_modes": _sha256(modes_path),
        "harmonic_buffer_preflight": _sha256(buffer_path),
        "candidate_qualification": qualification["sha256"],
    }
    campaign = build_material_nq1_mpi_campaign(
        source_revision=source["commit"],
        source_tree_sha256=source_tree,
        environment_sha256=lock["sha256"],
        input_hashes=input_hashes,
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
            _identity, problem, config = case.authenticate_scba_inputs(
                fixture=args.fixture,
                fc3_full=args.fc3_full,
                phase_b_artifacts=args.phase_b_artifacts,
                coupling_scale=scientific.coupling_scale,
                grid_step=scientific.grid_step,
                grid_n_positive=scientific.grid_n_positive,
                eta_lead=scientific.eta_lead,
                mixing=scientific.mixing,
                mixing_floor=RELEASE_MIXING_FLOOR,
                temperature_bias=scientific.temperature_bias_solver,
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

        resume = bool(single_campaign._failure_records(scba_root))
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
            born_initial_hdf5=born_path,
            grid_step=scientific.grid_step,
            grid_n_positive=scientific.grid_n_positive,
            eta_lead=scientific.eta_lead,
            mixing=scientific.mixing,
            mixing_floor=RELEASE_MIXING_FLOOR,
            temperature_bias=scientific.temperature_bias_solver,
            resume=resume,
        )
        state = status.get("state")
        if state not in {"passed", "scientific-failure"}:
            raise MaterialCampaignError(
                f"task {scientific.task_id} ended in infrastructure state {state!r}"
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
        artifact_validator=_receipt_evidence,
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
