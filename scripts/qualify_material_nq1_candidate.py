#!/usr/bin/env python3
"""Fail-closed non-SCBA qualification for an exact material candidate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import run_material_nq1_release_mpi as release
import run_sparse_nq1_phase_c_born_campaign as common
import validate_sparse_nq1_phase_c_scba as case
from negfpy.anharmonic._material_nq1_campaign import silicon_nq1_release_tasks
from negfpy.anharmonic._material_nq1_validation import (
    build_observation_mode_set,
    check_harmonic_embedding_buffer,
)
from negfpy.anharmonic._provenance import code_provenance


class QualificationError(RuntimeError):
    """A non-scientific candidate qualification check failed."""


QUALIFICATION_TEST_TARGETS = (
    "tests/test_material_nq1_campaign.py",
    "tests/test_material_nq1_validation.py",
    "tests/test_material_nq1_release_validation.py",
    "tests/test_material_nq1_mpi.py",
    "tests/test_material_nq1_release_mpi_driver.py",
    "tests/test_anharmonic_checkpoint_v2.py",
    "tests/test_anharmonic_sparse_cubic.py",
    "tests/test_anharmonic_sparse_solver.py",
    "tests/test_material_nq1_problem.py",
    "tests/test_sparse_nq1_phase_c_continuation.py",
    "tests/test_sparse_nq1_phase_c_scba_campaign.py",
)


def _publish_log(path: Path, payload: bytes) -> dict[str, Any]:
    common._publish_once(path, payload)
    return {
        "path": path.name,
        "sha256": release._sha256(path),
        "size": path.stat().st_size,
    }


def _run(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log: Path,
    label: str,
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    record = _publish_log(log, completed.stdout)
    record.update({
        "command": list(command),
        "returncode": int(completed.returncode),
    })
    if completed.returncode != 0:
        raise QualificationError(f"{label} failed; see {log}")
    return record


def qualify(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    evidence_root = output.parent / "qualification-evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    source = common._source_identity(repo, args.expected_commit)
    lock = common._validate_lock(args.environment_lock)
    source_tree = code_provenance().get("source_tree_sha256")
    if not isinstance(source_tree, str) or len(source_tree) != 64:
        raise QualificationError("source tree SHA256 is unavailable")

    interpreters = {
        "python310": args.python310.expanduser().resolve(strict=True),
        "python312": args.python312.expanduser().resolve(strict=True),
    }
    environment = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        environment[name] = "1"
    environment["PYTHONHASHSEED"] = "0"

    evidence: dict[str, Any] = {}
    versions: dict[str, str] = {}
    for label, executable in interpreters.items():
        version = subprocess.run(
            [str(executable), "-c", "import sys; print(sys.version.split()[0])"],
            cwd=repo, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, text=True,
        )
        if version.returncode != 0:
            raise QualificationError(f"{label} cannot start")
        versions[label] = version.stdout.strip()
        expected_minor = "3.10." if label == "python310" else "3.12."
        if not versions[label].startswith(expected_minor):
            raise QualificationError(f"{label} has wrong version {versions[label]!r}")
        evidence[f"{label}_pytest"] = _run(
            [str(executable), "-m", "pytest", "-q", *QUALIFICATION_TEST_TARGETS],
            cwd=repo,
            environment=environment,
            log=evidence_root / f"{label}-pytest.log",
            label=f"full {label} test suite",
        )
        evidence[f"{label}_pytest"]["scope"] = list(QUALIFICATION_TEST_TARGETS)

    mpiexec = args.mpiexec.expanduser().resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="negf-material-mpi-qualification-") as name:
        temporary = Path(name)
        for ranks in (2, 4):
            evidence[f"mpi_ranks_{ranks}"] = _run(
                [
                    str(mpiexec), "--bind-to", "core", "-n", str(ranks),
                    str(interpreters["python310"]),
                    str(repo / "tests/mpi/run_material_nq1_mpi_smoke.py"),
                    str(temporary / f"ranks-{ranks}"),
                    "--execution-id", f"qualification-ranks-{ranks}",
                ],
                cwd=repo,
                environment=environment,
                log=evidence_root / f"mpi-ranks-{ranks}.log",
                label=f"{ranks}-rank MPI smoke test",
            )

    base = next(
        task for task in silicon_nq1_release_tasks()
        if task.task_id == "coupling-g100"
    )
    identity, problem, _config = case.authenticate_scba_inputs(
        fixture=args.fixture,
        fc3_full=args.fc3_full,
        phase_b_artifacts=args.phase_b_artifacts,
        coupling_scale=1.0,
        grid_step=base.grid_step,
        grid_n_positive=base.grid_n_positive,
        eta_lead=base.eta_lead,
        mixing=base.mixing,
        temperature_bias=base.temperature_bias_solver,
    )
    modes = build_observation_mode_set(
        problem,
        observation_layers=(4,),
        grid_step=base.grid_step,
        eta_lead=base.eta_lead,
    )
    buffer = dict(check_harmonic_embedding_buffer(
        problem,
        frequencies=problem.grid.values[
            (problem.grid.values > 0.0) & (problem.grid.values <= 2.4)
        ],
        eta_lead=base.eta_lead,
    ))
    if not buffer["passed"]:
        raise QualificationError("harmonic embedding-buffer preflight failed")
    evidence["material_input_preflight"] = {
        "source_hashes": case._source_hashes(identity),
        "observation_modes": modes.as_dict(),
        "harmonic_buffer": buffer,
    }

    memory_per_largest_task = int(
        288
        * max(task.grid_n_positive * 2 + 1 for task in silicon_nq1_release_tasks())
        * problem.ndof**2
    )
    available_memory = int(
        next(
            line.split()[1]
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
    ) * 1024
    free_disk = shutil.disk_usage(output.parent).free
    predicted_archive = int(
        sum(
            12 * (2 * task.grid_n_positive + 1) * problem.ndof**2 * 16
            for task in silicon_nq1_release_tasks()
        )
    )
    resource = {
        "logical_cpu_count": os.cpu_count(),
        "available_memory_bytes": available_memory,
        "largest_task_memory_estimate_bytes": memory_per_largest_task,
        "free_disk_bytes": free_disk,
        "campaign_field_archive_estimate_bytes": predicted_archive,
        "memory_guard_passed": memory_per_largest_task <= 0.70 * available_memory,
        "disk_guard_passed": 2 * predicted_archive <= free_disk,
    }
    if not resource["memory_guard_passed"] or not resource["disk_guard_passed"]:
        raise QualificationError("remote resource preflight failed")
    evidence["resources"] = resource

    final_source = common._source_identity(repo, args.expected_commit)
    checks = {
        "source_clean_and_unchanged": final_source == source,
        "python_310_full_suite": evidence["python310_pytest"]["returncode"] == 0,
        "python_312_full_suite": evidence["python312_pytest"]["returncode"] == 0,
        "mpi_two_rank_smoke": evidence["mpi_ranks_2"]["returncode"] == 0,
        "mpi_four_rank_smoke": evidence["mpi_ranks_4"]["returncode"] == 0,
        "material_inputs_authenticated": bool(case._source_hashes(identity)),
        "resolved_mode_set_nonempty": bool(modes.subspaces),
        "harmonic_embedding_buffer": buffer["passed"] is True,
        "memory_guard": resource["memory_guard_passed"] is True,
        "disk_guard": resource["disk_guard_passed"] is True,
        "thread_limits_one": all(
            environment[name] == "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        ),
    }
    passed = bool(checks and all(checks.values()))
    return {
        "schema": release.QUALIFICATION_SCHEMA,
        "source": source,
        "source_tree_sha256": source_tree,
        "environment_sha256": lock["sha256"],
        "checks": checks,
        "evidence": {
            "python_versions": versions,
            "records": evidence,
        },
        "release_preflight_passed": passed,
    }


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
        release._atomic_json(args.output.expanduser().resolve(), report)
    except Exception as error:
        print(
            f"material candidate qualification error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 3
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["release_preflight_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
