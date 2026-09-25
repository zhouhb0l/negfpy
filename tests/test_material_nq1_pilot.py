from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
)
from negfpy.anharmonic import _material_nq1_pilot as pilot
from negfpy.anharmonic._launch_environment import build_launch_environment, qualification_environment
from negfpy.anharmonic._material_nq1_campaign import silicon_nq1_release_tasks
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import run_material_nq1_predictor_pilot as pilot_runner

_SYNTHETIC_DERIVED_SOURCE_HASHES = {
    "cubic_vertex_physics": "a" * 64,
    "cubic_vertex_storage": "b" * 64,
    "vertex_metadata.phase_b.archive_sha256": "c" * 64,
    "vertex_metadata.phase_b.campaign_manifest_sha256": "d" * 64,
}


def _qualification(path: Path, anchor_root: Path) -> None:
    repo = path.parent / "repo"
    package_file = repo / "src/negfpy/__init__.py"
    package_file.parent.mkdir(parents=True, exist_ok=True)
    package_file.write_bytes(b"synthetic package\n")
    full_payload = {
        "schema": "negfpy.material-nq1-scientific-controls.v2",
        "acceptance": dict(pilot.MATERIAL_NQ1_ACCEPTANCE),
        "excluded_claims": list(pilot.MATERIAL_NQ1_EXCLUDED_CLAIMS),
        "task_count": 19,
        "tasks": [
            task.as_dict() | {"task_sha256": task.task_sha256}
            for task in silicon_nq1_release_tasks()
        ],
    }
    full_controls = full_payload | {
        "task_set_sha256": pilot.sha256_bytes(pilot.canonical_json(full_payload))
    }
    anchor_contract = {
        "root": str(anchor_root),
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
    pilot_payload = {
        "schema": "negfpy.material-nq1-predictor-pilot-controls.v1",
        "classification": "development-pilot-not-release",
        "task_count": len(pilot.PILOT_TASK_IDS),
        "tasks": [
            task.as_dict() | {"task_sha256": task.task_sha256}
            for task in pilot.predictor_pilot_tasks()
        ],
        "solver_contract": {
            "backend": "fft", "hilbert_backend": "fft", "eta_device": 0.0,
            "max_workers": 1, "max_iterations": 800, "min_iterations": 3,
            "consecutive_passes": 3, "on_nonconvergence": "raise",
            "automatic_retry": False, "controls_changed_after_outcome": False,
        },
        "acceptance": {
            **dict(pilot.MATERIAL_NQ1_ACCEPTANCE),
            "anchor_field_rtol": pilot.FIELD_RTOL,
            "anchor_field_atol": pilot.FIELD_ATOL,
            "anchor_residual_factor": pilot.ANCHOR_RESIDUAL_FACTOR,
            "anchor_current_rtol": pilot.CURRENT_RTOL,
            "anchor_current_atol": pilot.CURRENT_ATOL,
        },
        "anchor_contract": anchor_contract,
        "anchor_contract_sha256": pilot.sha256_bytes(
            pilot.canonical_json(anchor_contract)
        ),
        "anchor_authentication_gate": "staged-five-task-pilot-preflight",
        "excluded_claims": list(pilot.MATERIAL_NQ1_EXCLUDED_CLAIMS),
        "release_certified": False,
    }
    pilot_controls = pilot_payload | {
        "task_set_sha256": pilot.sha256_bytes(pilot.canonical_json(pilot_payload))
    }
    core = {
        name: {
            "expected_sha256": digest,
            "observed_sha256": digest,
            "passed": True,
        }
        for name, digest in pilot._CORE_FILE_SHA256.items()
    }
    lock_path = path.parent / "inputs/environment.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"@EXPLICIT\nsynthetic-lock\n")
    lock_sha = pilot.sha256_file(lock_path)
    records: dict[str, object] = {}
    for label, version in (
        ("python310", "3.10.0"),
        ("python312", pilot.PILOT_PYTHON_VERSION),
    ):
        executable = repo / "tools" / label
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes((label + "\n").encode("ascii"))
        runtime_payload = {
            "schema": "negfpy.material-nq1-python-runtime.v2",
            "label": label,
            "version": version,
            "executable": str(executable),
            "executable_sha256": pilot.sha256_file(executable),
            "packages_sha256": "b" * 64,
            "launch_environment": build_launch_environment(
                repo=repo, executable=executable, home=repo / "launch-home",
                tmpdir=Path("/tmp/negfpy-pilot-fixture"), git_bin=repo / "git-bin",
            ),
        }
        records[f"{label}_runtime"] = runtime_payload | {
            "runtime_sha256": pilot.sha256_bytes(
                pilot.canonical_json(runtime_payload)
            ),
            "package_manifest": {
                "path": f"{label}-packages.txt",
                "sha256": "b" * 64,
                "size": 1,
            },
        }
        records[f"{label}_pytest"] = {
            "command": [f"/{label}", "-m", "pytest"],
            "returncode": 0,
            "path": f"{label}-pytest.log",
            "sha256": "d" * 64,
            "size": 1,
            "receipt_path": f"{label}-pytest.receipt.json",
            "receipt_sha256": "e" * 64,
            "identity_sha256": "a" * 64,
            "attempt": 1,
            "reused": False,
            "scope": ["tests", "docs/validation/bulk-rta-report/tests"],
            "excluded_tests": [],
        }
    for ranks in (2, 4):
        records[f"mpi_ranks_{ranks}"] = {
            "command": ["/mpiexec", "-n", str(ranks)],
            "returncode": 0,
            "path": f"mpi-ranks-{ranks}.log",
            "sha256": "f" * 64,
            "size": 1,
            "receipt_path": f"mpi-ranks-{ranks}.receipt.json",
            "receipt_sha256": "0" * 64,
            "identity_sha256": "1" * 64,
            "attempt": 1,
            "reused": False,
            "interpreter": "python312",
        }
    records["material_input_preflight"] = {
        "source_hashes": {
            "phase_a_fixture": "4" * 64,
            "raw_fc3": "5" * 64,
            "phase_b_artifact_set": "6" * 64,
            "phase_b_mapped_hdf5": "7" * 64,
            "phase_b_campaign_manifest": "8" * 64,
        },
        "observation_modes": {"subspaces": [{}]},
        "harmonic_buffer": {"passed": True},
    }
    records["resources"] = {
        "memory_guard_passed": True,
        "disk_guard_passed": True,
    }
    driver_files: dict[str, dict[str, object]] = {}
    for relative in pilot._QUALIFICATION_DRIVER_PATHS:
        selected = repo / relative
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes((relative + "\n").encode("ascii"))
        driver_files[relative] = {
            "relative_path": relative,
            "sha256": pilot.sha256_file(selected),
            "size": selected.stat().st_size,
        }
    tracked = repo / "tests/test_dummy.py"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_bytes(b"def test_dummy(): pass\n")
    tracked_files = {
        "tests/test_dummy.py": {
            "relative_path": "tests/test_dummy.py",
            "sha256": pilot.sha256_file(tracked),
            "size": tracked.stat().st_size,
        }
    }
    mpiexec = repo / "tools/mpiexec"
    mpiexec.write_bytes(b"synthetic mpiexec\n")
    context_payload = {
        "schema": "negfpy.material-nq1-qualification-context.v2",
        "repo_root": str(repo),
        "source": {
            "commit": "1" * 40,
            "tree": "e" * 40,
            "dirty": False,
            "package_file": str(package_file),
        },
        "source_tree_sha256": "2" * 64,
        "environment_lock": {
            "path": str(lock_path),
            "sha256": lock_sha,
            "format": "@EXPLICIT",
        },
        "scientific_task_set_sha256": full_controls["task_set_sha256"],
        "material_input_hashes": records["material_input_preflight"][
            "source_hashes"
        ],
        "runtimes": {
            "python310": records["python310_runtime"],
            "python312": records["python312_runtime"],
        },
        "mpiexec": {
            "path": str(mpiexec),
            "sha256": pilot.sha256_file(mpiexec),
        },
        "tracked_test_files": tracked_files,
        "driver_files": driver_files,
    }
    context = context_payload | {
        "qualification_context_sha256": pilot.sha256_bytes(
            pilot.canonical_json(context_payload)
        )
    }
    pilot.publish_json_once(path, {
        "schema": pilot.QUALIFICATION_SCHEMA,
        "source": {
            "commit": "1" * 40,
            "tree": "e" * 40,
            "dirty": False,
            "package_file": str(package_file),
        },
        "source_tree_sha256": "2" * 64,
        "environment_sha256": lock_sha,
        "scientific_controls": full_controls,
        "predictor_pilot_controls": pilot_controls,
        "checks": {
            name: True
            for name in (
                pilot._BASE_QUALIFICATION_CHECKS
                | pilot._PREDICTOR_QUALIFICATION_CHECKS
            )
        },
        "evidence": {
            "base_qualification_schema": (
                "negfpy.material-nq1-candidate-qualification.v2"
            ),
            "base_qualification_evidence": {
                "qualification_context": context,
                "python_versions": {
                    "python310": "3.10.0",
                    "python312": pilot.PILOT_PYTHON_VERSION,
                },
                "records": records,
            },
            "core_files": core,
            "predictor_counts": {
                "coupling-square": 8, "identity": 4, "none": 7,
            },
            "property_oracle": {
                "relative_path": "tests/test_material_nq1_predictor_oracle.py",
                "sha256": pilot._PROPERTY_ORACLE_SHA256,
                "covered_by": ["python310_full_suite", "python312_full_suite"],
                "suite_records": {
                    "python310_pytest": {"scope": ["tests"], "excluded_tests": []},
                    "python312_pytest": {"scope": ["tests"], "excluded_tests": []},
                },
            },
        },
        "candidate_qualification_passed": True,
        "pilot_submission_eligible": True,
        "release_submission_eligible": False,
        "next_gate": "staged-five-task-pilot",
    })


def _manifest(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    qualification = tmp_path / "qualification.json"
    _qualification(qualification, tmp_path / "anchors")
    qualification_document = pilot.load_canonical_json(
        qualification, name="synthetic qualification"
    )
    manifest = pilot.build_pilot_manifest(
        source_revision="1" * 40,
        source_git_tree="e" * 40,
        source_package_file=tmp_path / "repo/src/negfpy/__init__.py",
        source_tree_sha256="2" * 64,
        environment_sha256=qualification_document["environment_sha256"],
        qualification_sha256=pilot.sha256_file(qualification),
        input_hashes={
            "phase_a_fixture": "4" * 64,
            "raw_fc3": "5" * 64,
            "phase_b_artifact_set": "6" * 64,
            "phase_b_mapped_hdf5": "7" * 64,
            "phase_b_campaign_manifest": "8" * 64,
            "scba_case_driver": "9" * 64,
        },
        execution_id="unit-pilot",
        anchor_root=tmp_path / "anchors",
    )
    manifest_path = tmp_path / "pilot-manifest.json"
    pilot.publish_json_once(manifest_path, manifest)
    _preflight(tmp_path, manifest, manifest_path, qualification)
    return manifest, manifest_path, qualification


def _fields(
    path: Path,
    *,
    perturbation: float = 0.0,
    metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        if metadata is not None:
            handle.attrs["schema"] = "negfpy.sparse-nq1-phase-c-scba-fields.v1"
            handle.attrs["scba_executed"] = True
            handle.attrs["release_certified"] = False
            handle.create_dataset(
                "metadata_json",
                data=np.frombuffer(pilot.canonical_json(metadata), dtype=np.uint8),
            )
        handle.create_dataset("grid/omega", data=np.asarray([-0.1, 0.0, 0.1]))
        for name in (
            "green/retarded", "green/lesser", "green/greater",
            "sigma_pp/retarded", "sigma_pp/lesser", "sigma_pp/greater",
            "sigma_left/retarded", "sigma_left/lesser", "sigma_left/greater",
            "sigma_right/retarded", "sigma_right/lesser", "sigma_right/greater",
        ):
            handle.create_dataset(
                name,
                data=np.ones((3, 2, 2), dtype=np.complex128) + perturbation,
            )


def _rewrite_json(path: Path, document: dict[str, object]) -> None:
    os.chmod(path, 0o644)
    path.write_bytes(pilot.canonical_json(document))
    os.chmod(path, 0o444)


def _fixture_attempt_command(
    candidate_root: Path,
    root: Path,
    task: object,
    manifest: dict[str, object],
    runtime: dict[str, object],
) -> list[str]:
    source = manifest["preflight"]["source"]
    numerical = manifest["numerical_controls"]
    outputs = manifest["outputs"]
    command = [
        str(runtime["executable"]),
        str(candidate_root / "repo/scripts/validate_sparse_nq1_phase_c_scba.py"),
        "--fixture", str(source["fixture_path"]),
        "--fc3-full", str(source["fc3_full"]["path"]),
        "--phase-b-artifacts", str(source["phase_b"]["root"]),
        "--checkpoint", str(outputs["checkpoint"]),
        "--output-json", str(outputs["json"]),
        "--output-hdf5", str(outputs["hdf5"]),
        "--campaign-manifest-sha256",
        pilot.sha256_file(root / "campaign-manifest.json"),
        "--source-tree-sha256", str(manifest["source_tree_sha256"]),
        "--environment-sha256", str(manifest["environment_lock"]["sha256"]),
        "--run-id", str(manifest["run_id"]),
        "--coupling-scale", str(manifest["coupling_scale"]),
        "--grid-step", str(numerical["grid_step"]),
        "--grid-n-positive", str(numerical["grid_n_positive"]),
        "--eta-lead", str(numerical["eta_lead"]),
        "--mixing", str(numerical["mixing"]),
        "--mixing-floor", str(numerical["mixing_floor"]),
        "--temperature-bias", str(numerical["temperature_bias"]),
    ]
    if manifest["continuation_parent"] is not None:
        command.extend([
            "--continuation-parent-artifacts",
            str(manifest["continuation_parent"]["artifacts"]),
            "--continuation-axis", str(task.continuation_axis),
            "--continuation-predictor", str(task.continuation_predictor),
        ])
    return command + ["--no-resume"]


def _fixture_sha256sums(root: Path) -> bytes:
    members = (
        "campaign-manifest.json",
        "status-running.json",
        "outcomes/scba.json",
        "outcomes/scba-fields.h5",
        "checkpoint/sigma-pp.h5",
        "scba-report.json",
        "logs/scba.json",
    )
    return "".join(
        f"{pilot.sha256_file(root / name)}  {name}\n"
        for name in sorted(members)
    ).encode("ascii")


def _reseal_scba_after_log_change(root: Path) -> None:
    report_path = root / "scba-report.json"
    report = pilot.load_canonical_json(report_path, name="test SCBA report")
    report["log_sha256"] = pilot.sha256_file(root / "logs/scba.json")
    report["attempt_logs"] = [{
        "path": "logs/scba.json",
        "sha256": report["log_sha256"],
    }]
    _rewrite_json(report_path, report)
    sums = root / "SHA256SUMS"
    os.chmod(sums, 0o644)
    sums.write_bytes(_fixture_sha256sums(root))
    status_path = root / "status.json"
    status = pilot.load_canonical_json(status_path, name="test SCBA status")
    status["report_sha256"] = pilot.sha256_file(report_path)
    status["sha256sums_sha256"] = pilot.sha256_file(sums)
    _rewrite_json(status_path, status)


def _reseal_scba_after_report_change(root: Path) -> None:
    sums = root / "SHA256SUMS"
    os.chmod(sums, 0o644)
    sums.write_bytes(_fixture_sha256sums(root))
    status_path = root / "status.json"
    status = pilot.load_canonical_json(status_path, name="test SCBA status")
    status["report_sha256"] = pilot.sha256_file(root / "scba-report.json")
    status["sha256sums_sha256"] = pilot.sha256_file(sums)
    _rewrite_json(status_path, status)


def _scba_tree(
    root: Path,
    task: object,
    *,
    parent_receipt: Path | None = None,
    state: str = "passed",
    wrong_transformed_state: bool = False,
    tampered_diagnostics: bool = False,
    prior_infrastructure_history: bool = False,
) -> None:
    fields = root / "outcomes/scba-fields.h5"
    checkpoint = root / "checkpoint/sigma-pp.h5"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic checkpoint\n")
    candidate_root = next(
        selected
        for selected in (root, *root.parents)
        if (selected / "pilot-manifest.json").is_file()
    )
    pilot_manifest = pilot.load_canonical_json(candidate_root / "pilot-manifest.json", name="test pilot")
    qualification = pilot.load_canonical_json(candidate_root / "qualification.json", name="test qualification")
    runtime = qualification["evidence"]["base_qualification_evidence"]["records"]["python312_runtime"]
    pilot.publish_json_once(root.parent / "launch-environment.json", pilot.build_launch_environment_record(
        pilot_manifest, task.task_id, runtime, qualification_environment(runtime, repo=candidate_root / "repo")
    ))
    continuation_parent = None
    if task.continuation_parent is not None:
        assert parent_receipt is not None
        parent = pilot.load_canonical_json(parent_receipt, name="test parent")
        parent_task = next(
            item
            for item in pilot.predictor_pilot_tasks()
            if item.task_id == task.continuation_parent
        )
        parent_sigma = np.ones((3, 2, 2), dtype=np.complex128)
        factor = (task.coupling_scale / parent_task.coupling_scale) ** 2
        initialization_payload = {
            "schema": pilot.CONTINUATION_INITIALIZATION_SCHEMA,
            "predictor": "coupling-square",
            "parent_coupling_scale": parent_task.coupling_scale,
            "child_coupling_scale": task.coupling_scale,
            "factor": factor,
            "untransformed_initial_sigma_sha256": pilot._initial_pair_sha256(
                parent_sigma, parent_sigma
            ),
            "transformed_consumed_initial_sigma_sha256": (
                pilot._initial_pair_sha256(
                    parent_sigma * (factor + (1.0 if wrong_transformed_state else 0.0)),
                    parent_sigma * (factor + (1.0 if wrong_transformed_state else 0.0)),
                )
            ),
        }
        initialization = initialization_payload | {
            "initialization_sha256": pilot.sha256_bytes(
                pilot.canonical_json(initialization_payload)
            )
        }
        continuation_parent = {
            "artifacts": str(parent_receipt.parent / "scba"),
            "run_id": task.continuation_parent,
            "coupling_scale": parent_task.coupling_scale,
            "continuation_axis": "coupling",
            "campaign_manifest_sha256": parent["terminal_hashes"]["manifest"],
            "status_sha256": parent["terminal_hashes"]["status"],
            "report_sha256": parent["terminal_hashes"]["report"],
            "checkpoint_sha256": parent["terminal_hashes"]["checkpoint"],
            "checkpoint_identity_sha256": parent["checkpoint_identity_sha256"],
            "initial_sigma_sha256": initialization_payload[
                "transformed_consumed_initial_sigma_sha256"
            ],
            "initialization": initialization,
        }
    controls = {
        "grid_step": task.grid_step,
        "grid_n_positive": task.grid_n_positive,
        "grid_cutoff": task.grid_cutoff,
        "eta_lead": task.eta_lead,
        "mixing": task.mixing,
        "mixing_floor": task.mixing_floor,
        "temperature_bias": task.temperature_bias_solver,
    }
    solver_controls = {
        "eta_lead": task.eta_lead,
        "max_iterations": 800,
        "min_iterations": 3,
        "mixing": task.mixing,
        "mixing_floor": task.mixing_floor,
        "consecutive_passes": 3,
        "sigma_rtol": 1.0e-8,
        "sigma_atol": 1.0e-12,
        "green_rtol": 1.0e-8,
        "green_atol": 1.0e-12,
        "identity_rtol": 1.0e-8,
        "current_rtol": 1.0e-5,
        "current_atol": 1.0e-10,
        "tail_rtol": 1.0e-6,
        "retarded_strategy": "causal_hilbert",
        "backend": "fft",
        "hilbert_backend": "fft",
        "eta_device": 0.0,
        "surface_gf_method": "sancho_rubio",
        "max_workers": 1,
        "on_nonconvergence": "raise",
        "max_memory_fraction": 0.7,
        "allow_memory_overcommit": False,
    }
    preflight_source = {
        "fixture_path": str(candidate_root / "inputs/fixture"),
        "fixture": {"fixture_manifest_sha256": "4" * 64},
        "fc3_full": {
            "path": str(candidate_root / "inputs/fc3.h5"),
            "raw_sha256": "5" * 64,
        },
        "phase_b": {
            "root": str(candidate_root / "inputs/phase-b"),
            "artifact_set_sha256": "6" * 64,
            "archive": {"sha256": "7" * 64},
            "campaign_manifest_sha256": "8" * 64,
        },
        "problem_input_hashes": {"problem": "a" * 64},
        "memory": {
            "predicted_peak_bytes": 1024,
            "available_bytes": 32 * 1024,
            "maximum_fraction": 0.7,
            "guard_passed": True,
        },
    }
    identity_sha = pilot.sha256_bytes(task.task_id.encode("ascii"))
    output_paths = {
        "checkpoint": str(checkpoint.resolve()),
        "json": str((root / "outcomes/scba.json").resolve()),
        "hdf5": str(fields.resolve()),
    }
    preflight = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-preflight.v1",
        "source": preflight_source,
        "run_id": task.task_id,
        "coupling_scale": task.coupling_scale,
        "continuation_parent": continuation_parent,
        "born_initial": None,
        "source_tree_sha256": "2" * 64,
        "environment_sha256": pilot.load_canonical_json(
            candidate_root / "pilot-manifest.json", name="test pilot manifest"
        )["environment_sha256"],
        "configuration_sha256": "b" * 64,
        "checkpoint_identity_sha256": identity_sha,
        "checkpoint_schema": "negfpy_sparse_material_scba_checkpoint",
        "checkpoint_path": output_paths["checkpoint"],
        "outputs": {"json": output_paths["json"], "hdf5": output_paths["hdf5"]},
        "dimensions": {"n_frequency": 3, "ndof": 2, "nnz": 1},
        "predicted_peak_memory_bytes": 288 * 3 * 2**2,
        "numerical_controls": controls,
        "controls": solver_controls,
        "fixed_controls_no_retuning": True,
        "resume_exact_identity": True,
        "passed": True,
        "scba_executed": False,
        "release_certified": False,
    }
    manifest = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-campaign.v2",
        "scientific_scope": (
            "authenticated silicon Nq=1 sparse material SCBA, fixed Nw=17 "
            "development gate"
        ),
        "source": {
            "commit": "1" * 40,
            "tree": "e" * 40,
            "dirty": False,
            "package_file": str(candidate_root / "repo/src/negfpy/__init__.py"),
        },
        "source_tree_sha256": "2" * 64,
        "environment_lock": pilot.load_canonical_json(
            candidate_root / "qualification.json", name="test qualification"
        )["evidence"]["base_qualification_evidence"][
            "qualification_context"
        ]["environment_lock"],
        "driver": {
            "relative_path": "scripts/validate_sparse_nq1_phase_c_scba.py",
            "sha256": "9" * 64,
        },
        "run_id": task.task_id,
        "coupling_scale": task.coupling_scale,
        "continuation_axis": task.continuation_axis,
        "continuation_predictor": task.continuation_predictor,
        "continuation_parent": continuation_parent,
        "numerical_controls": controls,
        "preflight": preflight,
        "born_initial": None,
        "fixed_controls_no_retuning": True,
        "restart": {
            "checkpoint_schema": "negfpy_sparse_material_scba_checkpoint",
            "resume_exact_identity": True,
            "automatic_retry": False,
        },
        "watchdog": {
            "wall_limit_seconds": pilot.PILOT_WALL_LIMIT_SECONDS,
            "rss_limit_bytes": pilot.PILOT_RSS_LIMIT_BYTES,
            "term_grace_seconds": 60,
            "automatic_retry": False,
        },
        "outputs": output_paths,
        "python": {
            "executable": str(candidate_root / "repo/tools/python312"),
            "version": pilot.PILOT_PYTHON_VERSION,
        },
        "distributions": dict(pilot.PILOT_DISTRIBUTIONS),
        "thread_limits": dict(pilot.PILOT_THREAD_LIMITS),
    }
    pilot.publish_json_once(root / "campaign-manifest.json", manifest)
    manifest_sha = pilot.sha256_file(root / "campaign-manifest.json")
    pilot.publish_json_once(root / "status-running.json", {
        "schema": "negfpy.sparse-nq1-phase-c-scba-status.v1",
        "campaign_manifest_sha256": manifest_sha,
        "state": "running",
        "automatic_retry": False,
        "release_certified": False,
    })
    converged = state == "passed"
    result = {
        "method": "scba",
        "converged": converged,
        "history_length": 3 if converged else 800,
        "release_certified": False,
        "array_hashes": {
            "grid/omega": pilot._array_sha256(
                np.asarray([-0.1, 0.0, 0.1])
            ),
            **{
                name: pilot._array_sha256(
                    np.ones((3, 2, 2), dtype=np.complex128)
                )
                for name in (
                    "green/retarded", "green/lesser", "green/greater",
                    "sigma_pp/retarded", "sigma_pp/lesser", "sigma_pp/greater",
                    "sigma_left/retarded", "sigma_left/lesser", "sigma_left/greater",
                    "sigma_right/retarded", "sigma_right/lesser", "sigma_right/greater",
                )
            },
        },
        "currents": {
            "left": 0.0,
            "right": 0.0,
            "collision_power": 0.0,
            "contact_spectra": None,
            "collision_spectrum": None,
            "bond_currents": None,
            "bond_spectra": None,
            "spatial": None,
        },
        "diagnostics": {
            "passed": True,
            "release_certified": False,
            "failures": ["release: synthetic spectral moment gate"],
            "metrics": {
                "history_last_consecutive_passes": 3.0,
                "history_last_green_residual": 0.0,
                "history_last_sigma_residual": (
                    1.0 if tampered_diagnostics else 0.0
                ),
                "current_contact_abs_error": 0.0,
                "current_collision_abs_error": 0.0,
                "current_tolerance": 1.0e-10,
                "sigma_pp_tail_ratio": 0.0,
                "equilibrium_fdt_applicable": 1.0,
                "package_version_consistent": 1.0,
            },
        },
        "solver_metadata": {
            "current_scale": 1.0,
            "max_iterations": 800,
            "min_iterations": 3,
            "consecutive_passes": 3,
            "mixing_initial": task.mixing,
            "mixing_floor": task.mixing_floor,
            "eta_lead": task.eta_lead,
            "eta_device": 0.0,
            "backend": "fft",
            "effective_backend": "fft",
            "hilbert_backend": "fft",
            "grid_step": task.grid_step,
            "grid_n_positive": task.grid_n_positive,
            "grid_cutoff": task.grid_cutoff,
            "dtype": "complex128",
        },
        "scientific_error": None if converged else {
            "type": "SCBAConvergenceError", "message": "synthetic"
        },
    }
    hdf5_metadata = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-fields.v1",
        "campaign_manifest_sha256": manifest_sha,
        "checkpoint_identity_sha256": identity_sha,
        "problem_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "continuation_parent": continuation_parent,
        "born_initial": None,
        "result": result,
        "timings_seconds": {"total": 1.0},
        "scba_executed": True,
        "release_certified": False,
    }
    _fields(fields, metadata=hdf5_metadata)
    outcome_source = copy.deepcopy(preflight_source)
    outcome_source["memory"]["available_bytes"] = 48 * 1024
    outcome = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-case.v1",
        "campaign_manifest_sha256": manifest_sha,
        "preflight_source": outcome_source,
        "checkpoint": output_paths["checkpoint"],
        "checkpoint_sha256": pilot.sha256_file(checkpoint),
        "checkpoint_status": "completed" if converged else "scientific_failure",
        "continuation_parent": continuation_parent,
        "result": result,
        "born_initial": None,
        "hdf5": {
            "path": output_paths["hdf5"],
            "sha256": pilot.sha256_file(fields),
            "schema": "negfpy.sparse-nq1-phase-c-scba-fields.v1",
        },
        "scba_executed": True,
        "release_certified": False,
        "not_certified": [
            "converged material conductance", "0.3.0 release", "transverse SCBA"
        ],
    }
    pilot.publish_json_once(root / "outcomes/scba.json", outcome)
    supervision = {
        "returncode": 0 if converged else 2,
        "elapsed_seconds": 1.0,
        "peak_observed_rss_bytes": 1024,
        "watchdog_trigger": None,
        "killed_after_term_grace": False,
        "wall_limit_seconds": pilot.PILOT_WALL_LIMIT_SECONDS,
        "rss_limit_bytes": pilot.PILOT_RSS_LIMIT_BYTES,
    }
    attempt_log = {
        "attempt": 1,
        "resume": False,
        "command": _fixture_attempt_command(
            candidate_root,
            root,
            task,
            manifest,
            pilot.load_canonical_json(
                candidate_root / "qualification.json", name="test qualification"
            )["evidence"]["base_qualification_evidence"]["records"][
                "python312_runtime"
            ],
        ),
        "supervision": supervision,
        "stdout": json.dumps(
            outcome,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ) + "\n",
        "stderr": "",
    }
    pilot.publish_json_once(root / "logs/scba.json", attempt_log)
    if prior_infrastructure_history:
        pilot.publish_json_once(root / "infrastructure-failure.json", {
            "schema": "synthetic-prior-watchdog",
            "watchdog_trigger": "wall-timeout",
        })
    report = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-report.v1",
        "campaign_manifest_sha256": manifest_sha,
        "state": state,
        "scba_converged": converged,
        "outcome_sha256": pilot.sha256_file(root / "outcomes/scba.json"),
        "fields_hdf5_sha256": pilot.sha256_file(fields),
        "checkpoint_sha256": pilot.sha256_file(checkpoint),
        "log_sha256": pilot.sha256_file(root / "logs/scba.json"),
        "supervision": supervision,
        "automatic_retry": False,
        "attempt_logs": [{
            "path": "logs/scba.json",
            "sha256": pilot.sha256_file(root / "logs/scba.json"),
        }],
        "infrastructure_failures": [],
        "controls_changed": False,
        "release_certified": False,
        "next_gate": (
            "refine frequency/cutoff/eta and validate conservation"
            if converged
            else "inspect fixed-control scientific failure"
        ),
    }
    pilot.publish_json_once(root / "scba-report.json", report)
    (root / "SHA256SUMS").write_bytes(_fixture_sha256sums(root))
    status = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-status.v1",
        "campaign_manifest_sha256": manifest_sha,
        "state": state,
        "scba_converged": converged,
        "report_sha256": pilot.sha256_file(root / "scba-report.json"),
        "sha256sums_sha256": pilot.sha256_file(root / "SHA256SUMS"),
        "automatic_retry": False,
        "release_certified": False,
    }
    pilot.publish_json_once(root / "status.json", status)


def _receipt(
    tmp_path: Path,
    manifest: dict[str, object],
    task: object,
    parent: Path | None,
    *,
    state: str = "passed",
) -> Path:
    root = tmp_path / "tasks" / task.task_id / "scba"
    _scba_tree(root, task, parent_receipt=parent, state=state)
    document = pilot.build_task_receipt(
        manifest=manifest,
        task_id=task.task_id,
        scba_root=root,
        parent_receipt=parent,
    )
    path = tmp_path / "tasks" / task.task_id / "task-receipt.json"
    pilot.publish_json_once(path, document)
    return path


def _mock_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    *,
    source_hash_mutation: str | None = None,
) -> None:
    def load(path: Path):
        root = Path(path).parents[1]
        outcome = pilot.load_canonical_json(
            root / "outcomes/scba.json", name="test outcome"
        )
        manifest = pilot.load_canonical_json(
            root / "campaign-manifest.json", name="test manifest"
        )
        task_id = manifest["run_id"]
        continuation = manifest["continuation_parent"]
        sources: dict[str, str] = {
            "phase_a_fixture": "4" * 64,
            "raw_fc3": "5" * 64,
            "phase_b_artifact_set": "6" * 64,
            "phase_b_mapped_hdf5": "7" * 64,
            "phase_b_campaign_manifest": "8" * 64,
        } | dict(_SYNTHETIC_DERIVED_SOURCE_HASHES)
        sigma = np.ones((3, 2, 2), dtype=np.complex128)
        initial = pilot._explicit_zero_initial_sha256(sigma.shape)
        if continuation is not None:
            initialization = continuation["initialization"]
            initial = initialization["transformed_consumed_initial_sigma_sha256"]
            sources.update({
                "continuation_parent_campaign_manifest": continuation[
                    "campaign_manifest_sha256"
                ],
                "continuation_parent_checkpoint": continuation["checkpoint_sha256"],
                "continuation_parent_report": continuation["report_sha256"],
                "continuation_parent_status": continuation["status_sha256"],
                "continuation_parent_initialization": initialization[
                    "initialization_sha256"
                ],
            })
        if source_hash_mutation == "missing-derived":
            sources.pop("cubic_vertex_physics")
        elif source_hash_mutation == "altered-derived":
            sources["cubic_vertex_storage"] = "f" * 64
        elif source_hash_mutation == "extra-derived":
            sources["vertex_metadata.unexpected.source_sha256"] = "f" * 64
        identity = SimpleNamespace(
            identity_sha256=pilot.sha256_bytes(task_id.encode("ascii")),
            run_id=task_id,
            source_tree_sha256="2" * 64,
            environment_sha256=manifest["environment_lock"]["sha256"],
            initial_state_sha256=initial,
            source_hashes=sources,
        )
        converged = outcome["result"]["converged"]
        if converged:
            history = [
                SimpleNamespace(
                    passed=True,
                    consecutive_passes=value,
                    sigma_residual=0.0,
                    green_residual=0.0,
                )
                for value in (1, 2, 3)
            ]
            passes = 3
        else:
            history = [
                SimpleNamespace(
                    passed=False,
                    consecutive_passes=0,
                    sigma_residual=1.0,
                    green_residual=1.0,
                )
                for _ in range(800)
            ]
            passes = 0
        return SimpleNamespace(
            identity=identity,
            status=outcome["checkpoint_status"],
            history=history,
            solver_state=SimpleNamespace(consecutive_passes=passes),
            sigma_lesser=sigma,
            sigma_greater=sigma,
        )

    monkeypatch.setattr(pilot, "load_scba_checkpoint_v2_self_validating", load)
    monkeypatch.setattr(
        pilot,
        "_replay_scientific_diagnostics",
        lambda *, explicit_source_hashes, **_kwargs: (
            dict(explicit_source_hashes) | dict(_SYNTHETIC_DERIVED_SOURCE_HASHES)
        ),
    )
    monkeypatch.setattr(
        pilot,
        "validate_preflight_record",
        lambda _manifest, path: pilot.load_canonical_json(
            path, name="synthetic bound preflight"
        ),
    )


def _preflight(
    tmp_path: Path,
    manifest: dict[str, object],
    manifest_path: Path,
    qualification: Path,
) -> Path:
    document = pilot.build_preflight_record(
        manifest=manifest,
        manifest_path=manifest_path,
        qualification_path=qualification,
        source_clean_and_exact=True,
        input_hashes_reproduced=True,
        anchors={
            "coupling-g0001": {"passed": True},
            "coupling-g001": {"passed": True},
        },
    )
    path = tmp_path / "pilot-preflight.json"
    pilot.publish_json_once(path, document)
    return path


def _scheduler_receipts(
    tmp_path: Path,
    manifest: dict[str, object],
    task_receipts: dict[str, Path],
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for ordinal, task in enumerate(manifest["tasks"], start=101):
        task_id = task["task_id"]
        job_id = f"{ordinal}.pbs101"
        scheduler_root = tmp_path / "scheduler" / task_id
        raw = scheduler_root / "qstat-terminal.txt"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text(
            f"Job Id: {job_id}\n"
            f"    Job_Name = {task['scheduler_job_name']}\n"
            "    job_state = F\n"
            "    Exit_status = 0\n",
            encoding="utf-8",
        )
        os.chmod(raw, 0o444)
        pbs_stdout = scheduler_root / "task.stdout"
        pbs_stderr = scheduler_root / "task.stderr"
        pbs_stdout.write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "state": "passed",
                    "anchor_comparison_passed": (
                        True if task["anchor_task_id"] is not None else None
                    ),
                    "release_certified": False,
                },
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        pbs_stderr.write_bytes(b"")
        os.chmod(pbs_stdout, 0o444)
        os.chmod(pbs_stderr, 0o444)
        task_receipt = task_receipts[task_id]
        task_receipt_document = pilot.load_canonical_json(
            task_receipt, name="test task receipt"
        )
        workload = pilot._scheduler_workload_contract(
            manifest, task, task_receipt_document
        )
        pbs = scheduler_root / "task.pbs"
        pbs.write_bytes(
            pilot._canonical_pbs_script(
                task=task,
                stdout_path=pbs_stdout,
                stderr_path=pbs_stderr,
                workload=workload,
            )
        )
        os.chmod(pbs, 0o444)
        command = ["qsub", str(pbs)]
        command_sha = pilot.sha256_bytes(pilot.canonical_json(command))
        pre = {
            "schema": pilot.PRE_SUBMISSION_SCHEMA,
            "pilot_sha256": manifest["pilot_sha256"],
            "task_id": task_id,
            "scientific_task_sha256": task["task_sha256"],
            "job_name": task["scheduler_job_name"],
            "pbs_script_path": str(pbs),
            "pbs_script_sha256": pilot.sha256_file(pbs),
            "pilot_driver_path": workload["driver_path"],
            "pilot_driver_sha256": workload["driver_sha256"],
            "workload_command": workload["command"],
            "workload_command_sha256": workload["command_sha256"],
            "submission_command": command,
            "submission_command_sha256": command_sha,
            "release_certified": False,
        }
        pre_path = scheduler_root / "pre-submission.json"
        pilot.publish_json_once(pre_path, pre)
        qsub_stdout = scheduler_root / "qsub.stdout"
        qsub_stderr = scheduler_root / "qsub.stderr"
        qsub_stdout.write_text(job_id + "\n", encoding="ascii")
        qsub_stderr.write_bytes(b"")
        os.chmod(qsub_stdout, 0o444)
        os.chmod(qsub_stderr, 0o444)
        submission = pre | {
            "schema": pilot.SUBMISSION_SCHEMA,
            "job_id": job_id,
            "pre_submission_path": str(pre_path),
            "pre_submission_sha256": pilot.sha256_file(pre_path),
            "qsub_stdout_path": str(qsub_stdout),
            "qsub_stdout_sha256": pilot.sha256_file(qsub_stdout),
            "qsub_stderr_path": str(qsub_stderr),
            "qsub_stderr_sha256": pilot.sha256_file(qsub_stderr),
        }
        submission_path = scheduler_root / "submission.json"
        pilot.publish_json_once(submission_path, submission)
        receipt = {
            "schema": pilot.SCHEDULER_SCHEMA,
            "pilot_sha256": manifest["pilot_sha256"],
            "task_id": task_id,
            "job_id": job_id,
            "job_name": task["scheduler_job_name"],
            "terminal_state": "F",
            "exit_status": 0,
            "raw_qstat_path": str(raw),
            "raw_qstat_sha256": pilot.sha256_file(raw),
            "task_receipt_path": str(task_receipt),
            "task_receipt_sha256": pilot.sha256_file(task_receipt),
            "pre_submission_path": str(pre_path),
            "pre_submission_sha256": pilot.sha256_file(pre_path),
            "submission_path": str(submission_path),
            "submission_sha256": pilot.sha256_file(submission_path),
            "pbs_script_path": str(pbs),
            "pbs_script_sha256": pilot.sha256_file(pbs),
            "pbs_stdout_path": str(pbs_stdout),
            "pbs_stdout_sha256": pilot.sha256_file(pbs_stdout),
            "pbs_stderr_path": str(pbs_stderr),
            "pbs_stderr_sha256": pilot.sha256_file(pbs_stderr),
            "submission_command_sha256": command_sha,
            "workload_command_sha256": workload["command_sha256"],
            "release_certified": False,
        }
        path = scheduler_root / "scheduler-terminal-receipt.json"
        pilot.publish_json_once(path, receipt)
        result[task_id] = path
    return result


def _synthetic_anchor_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "anchors"
    records: dict[str, dict[str, dict[str, str]]] = {}
    anchor_outcome = {
        "result": {
            "converged": True,
            "release_certified": False,
            "currents": {"left": 0.0, "right": 0.0, "collision_power": 0.0},
            "diagnostics": {
                "passed": True,
                "metrics": {
                    "history_last_sigma_residual": 0.0,
                    "history_last_green_residual": 0.0,
                },
            },
            "solver_metadata": {"current_scale": 1.0},
        },
        "release_certified": False,
    }
    for anchor_id in ("coupling-g0001", "coupling-g001"):
        fields = root / anchor_id / "scba-fields.h5"
        outcome = root / anchor_id / "scba.json"
        _fields(fields)
        pilot.publish_json_once(outcome, anchor_outcome)
        records[anchor_id] = {
            "receipt": {
                "relative_path": f"{anchor_id}/receipt.json",
                "sha256": "a" * 64,
            },
            "fields": {
                "relative_path": fields.relative_to(root).as_posix(),
                "sha256": pilot.sha256_file(fields),
            },
            "outcome": {
                "relative_path": outcome.relative_to(root).as_posix(),
                "sha256": pilot.sha256_file(outcome),
            },
            "status": {
                "relative_path": f"{anchor_id}/status.json",
                "sha256": "b" * 64,
            },
            "report": {
                "relative_path": f"{anchor_id}/report.json",
                "sha256": "c" * 64,
            },
        }
    monkeypatch.setattr(pilot, "RELEASE_018_ANCHORS", records)


def test_pilot_is_exact_first_five_release_tasks() -> None:
    pilot_tasks = pilot.predictor_pilot_tasks()
    release_prefix = silicon_nq1_release_tasks()[:5]
    assert pilot_tasks == release_prefix
    assert tuple(task.task_id for task in pilot_tasks) == (
        "coupling-g0001",
        "coupling-g001",
        "coupling-g0015",
        "coupling-g002",
        "coupling-g0025",
    )
    assert tuple(task.continuation_predictor for task in pilot_tasks) == (
        None,
        "coupling-square",
        "coupling-square",
        "coupling-square",
        "coupling-square",
    )
    assert pilot.PILOT_PYTHON_VERSION == "3.12.13"
    assert dict(pilot.PILOT_DISTRIBUTIONS) == {
        "h5py": "3.16.0",
        "numpy": "2.5.1",
        "phonopy": "4.4.0",
        "phono3py": "4.4.0",
        "scipy": "1.18.0",
        "spglib": "2.7.0",
        "threadpoolctl": "3.6.0",
    }
    assert dict(pilot.PILOT_THREAD_LIMITS) == {
        "BLIS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert pilot.PILOT_PBS_PROJECT == "13004417"
    assert pilot.PILOT_PBS_QUEUE == "normal"
    assert pilot.PILOT_GIT_BIN == Path("/scratch/GPFS/app/apps/git/2.39.2/bin")
    assert dict(pilot.PILOT_SCHEDULER_JOB_NAMES) == {
        "coupling-g0001": "nq1kmp925-g1",
        "coupling-g001": "nq1kmp925-g01",
        "coupling-g0015": "nq1kmp925-g15",
        "coupling-g002": "nq1kmp925-g2",
        "coupling-g0025": "nq1kmp925-g25",
    }
    assert all(
        len(name) <= 15 for name in pilot.PILOT_SCHEDULER_JOB_NAMES.values()
    )


def test_canonical_pbs_resources_have_headroom_above_internal_watchdogs() -> None:
    environment = build_launch_environment(
        repo=Path("/candidate"), executable=Path("/staged/python/bin/python3.12"),
        home=Path("/launch/home"), tmpdir=Path("/tmp/negfpy-pilot-fixture"), git_bin=pilot.PILOT_GIT_BIN,
    ) | {"NEGFPY_QUALIFICATION_RUNTIME_SHA256": "a" * 64}
    payload = pilot._canonical_pbs_script(
        task={"scheduler_job_name": "nq1pa-g0001"},
        stdout_path=Path("/evidence/task.stdout"),
        stderr_path=Path("/evidence/task.stderr"),
        workload={
            "repo_root": "/candidate",
            "launch_environment": environment,
            "command": [
                "/staged/python/bin/python3.12",
                "/candidate/pilot.py",
                "run-task",
            ],
        },
    ).decode("utf-8")
    scheduler_identity = (
        "#!/bin/bash\n"
        "#PBS -N nq1pa-g0001\n"
        "#PBS -P 13004417\n"
        "#PBS -q normal\n"
    )
    assert payload.startswith(scheduler_identity)
    assert payload.count("#PBS -P 13004417\n") == 1
    assert payload.count("#PBS -q normal\n") == 1
    assert payload.count("#PBS -r n\n") == 1
    assert "#PBS -l select=1:ncpus=2:mem=16gb\n" in payload
    assert "#PBS -l walltime=50:00:00\n" in payload
    assert "\n/usr/bin/env -i " in payload
    assert "export " not in payload
    for name, value in environment.items():
        assert f"{name}={value}" in payload
    for line in ("umask 077", "cd -- /candidate"):
        assert payload.count(line + "\n") == 1
    assert payload.endswith("/staged/python/bin/python3.12 /candidate/pilot.py run-task\n")
    assert pilot.PILOT_PBS_NCPUS == 2
    assert pilot.PILOT_PBS_MEMORY == "16gb"
    assert pilot.PILOT_PBS_WALLTIME == "50:00:00"
    assert pilot.PILOT_WALL_LIMIT_SECONDS < 50 * 60 * 60
    assert pilot.PILOT_RSS_LIMIT_BYTES < 16 * 1000**3


@pytest.mark.parametrize(
    "bad_path",
    (
        Path("relative/output"),
        Path("/bad path/output"),
        Path("/bad\noutput"),
        Path("/bad\0output"),
    ),
)
def test_canonical_pbs_rejects_unsafe_directive_paths(bad_path: Path) -> None:
    with pytest.raises(pilot.MaterialNQ1PilotError, match="safe absolute PBS path"):
        pilot._canonical_pbs_script(
            task={"scheduler_job_name": "nq1pa-g0001"},
            stdout_path=bad_path,
            stderr_path=Path("/evidence/task.stderr"),
            workload={
                "repo_root": "/candidate",
                "command": ["/staged/python/bin/python3.12", "run-task"],
            },
        )


def test_manifest_is_deterministic_and_explicitly_non_release(tmp_path: Path) -> None:
    manifest, _path, _qualification_path = _manifest(tmp_path)
    assert pilot.validate_pilot_manifest(manifest) == manifest
    assert manifest["release_certified"] is False
    assert manifest["classification"] == "development-pilot-not-release"
    assert manifest["tasks"][2]["anchor_task_id"] is None
    assert manifest["anchor_source"]["use"] == (
        "comparison-only-never-continuation-parent"
    )


def test_create_only_json_requires_exact_bytes_and_mode_0444(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    pilot.publish_json_once(path, {"value": 1})
    assert path.stat().st_mode & 0o777 == 0o444
    pilot.publish_json_once(path, {"value": 1})
    with pytest.raises(pilot.MaterialNQ1PilotError, match="differs"):
        pilot.publish_json_once(path, {"value": 2})
    os.chmod(path, 0o644)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="mode 0444"):
        pilot.publish_json_once(path, {"value": 1})


def test_create_only_json_race_rechecks_winner_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raced_link(source: str | Path, target: str | Path) -> None:
        selected = Path(target)
        selected.write_bytes(Path(source).read_bytes())
        os.chmod(selected, 0o644)
        raise FileExistsError

    monkeypatch.setattr(pilot.os, "link", raced_link)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="mode 0444"):
        pilot.publish_json_once(tmp_path / "raced.json", {"value": 1})


def test_cli_has_no_resume_or_scheduler_path_override_and_rejects_symlink_roots(
    tmp_path: Path,
) -> None:
    destinations = {action.dest for action in pilot_runner._parser()._actions}
    assert "resume_exact" not in destinations
    assert "scheduler_receipt" not in destinations
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="symlink"):
        pilot_runner._nonsymlink_directory(link, name="pilot root", create=False)


def test_prelaunch_requires_exact_live_inputs_and_a_virgin_task_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _manifest_path, _qualification_path = _manifest(tmp_path)
    monkeypatch.setattr(
        pilot_runner,
        "_candidate_identity",
        lambda _args: (
            {
                "commit": manifest["source_revision"],
                "tree": manifest["source_git_tree"],
                "dirty": False,
                "package_file": manifest["source_package_file"],
            },
            {"sha256": manifest["environment_sha256"]},
            manifest["source_tree_sha256"],
        ),
    )
    monkeypatch.setattr(
        pilot_runner,
        "_authenticate_inputs",
        lambda _args: dict(manifest["input_hashes"]) | {"raw_fc3": "0" * 64},
    )
    qualified_lock = pilot.load_canonical_json(
        _qualification_path, name="test qualification"
    )["evidence"]["base_qualification_evidence"]["qualification_context"][
        "environment_lock"
    ]
    with pytest.raises(pilot.MaterialNQ1PilotError, match="inputs changed"):
        pilot_runner._verify_live_candidate(
            SimpleNamespace(), manifest, qualified_lock=qualified_lock
        )

    task_root = tmp_path / "current-task"
    (task_root / "scba/checkpoint").mkdir(parents=True)
    (task_root / "scba/checkpoint/sigma-pp.h5").write_bytes(b"prior attempt")
    with pytest.raises(pilot.MaterialNQ1PilotError, match="already has evidence"):
        pilot_runner._require_virgin_task_root(task_root, "coupling-g0001")

    empty_task_root = tmp_path / "empty-task"
    empty_task_root.mkdir()
    with pytest.raises(pilot.MaterialNQ1PilotError, match="already has evidence"):
        pilot_runner._require_virgin_task_root(
            empty_task_root, "coupling-g0001"
        )

    claimed = tmp_path / "claimed-task"
    pilot_runner._require_virgin_task_root(claimed, "coupling-g0001")
    assert claimed.is_dir()
    assert (claimed / "scba").is_dir()

    escaped_root = tmp_path / "escaped-pilot"
    outside = tmp_path / "outside"
    escaped_root.mkdir()
    outside.mkdir()
    (escaped_root / "tasks").symlink_to(outside, target_is_directory=True)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="symlink"):
        pilot_runner._require_virgin_task_root(
            escaped_root / "tasks/coupling-g0001", "coupling-g0001"
        )
    with pytest.raises(pilot.MaterialNQ1PilotError, match="symlink"):
        pilot.publish_json_once(
            escaped_root / "tasks/coupling-g0001/task-receipt.json",
            {"forged": True},
        )
    assert list(outside.iterdir()) == []
    comparisons_outside = tmp_path / "comparisons-outside"
    comparisons_outside.mkdir()
    (escaped_root / "comparisons").symlink_to(
        comparisons_outside, target_is_directory=True
    )
    with pytest.raises(pilot.MaterialNQ1PilotError, match="symlink"):
        pilot.publish_json_once(
            escaped_root / "comparisons/anchor-coupling-g0001.json",
            {"forged": True},
        )
    assert list(comparisons_outside.iterdir()) == []


def test_streamed_six_field_comparison_is_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _fields(first)
    _fields(second, perturbation=5.0e-11)
    assert pilot.compare_fields(first, second)["passed"]
    second.unlink()
    _fields(second, perturbation=5.0e-7)
    assert not pilot.compare_fields(first, second)["passed"]


def test_anchor_field_budget_is_explicit_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    first = tmp_path / "first.h5"
    second = tmp_path / "second.h5"
    _fields(first)
    _fields(second, perturbation=2.0e-8)
    assert not pilot.compare_fields(first, second)["passed"]
    bounded = pilot.compare_fields(first, second, absolute_budget=1.0e-8)
    assert bounded["passed"] is True
    assert bounded["absolute_budget"] == 1.0e-8
    with pytest.raises(pilot.MaterialNQ1PilotError, match="finite and nonnegative"):
        pilot.compare_fields(first, second, absolute_budget=-1.0)


def test_complete_checkpoint_sources_reconstruct_material_derived_hashes() -> None:
    device, lead, _dense = pinned_scalar_chain(1, pinning=1.0, cubic=0.2)
    vertex = SparseCubicVertex(
        ndof=1,
        indices=np.asarray([[0, 0, 0]], dtype=np.int64),
        coefficients=np.asarray([0.2], dtype=np.float64),
        metadata={
            "material": {"source_sha256": "a" * 64},
            "phase_b": {"archive_sha256": "b" * 64},
        },
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=5),
    )
    config = SCBAConfig(
        eta_lead=0.025,
        max_iterations=800,
        min_iterations=3,
        consecutive_passes=3,
    )
    explicit = {
        "phase_a_fixture": "4" * 64,
        "raw_fc3": "5" * 64,
        "phase_b_artifact_set": "6" * 64,
        "phase_b_mapped_hdf5": "7" * 64,
        "phase_b_campaign_manifest": "8" * 64,
    }
    complete = pilot._complete_checkpoint_source_hashes(
        scba_manifest={
            "run_id": "coupling-g0001",
            "source_tree_sha256": "2" * 64,
            "preflight": {"environment_sha256": "3" * 64},
        },
        problem=problem,
        config=config,
        explicit_source_hashes=explicit,
    )

    expected_derived = {
        "cubic_vertex_physics": vertex.cubic_physics_sha256,
        "cubic_vertex_storage": vertex.vertex_storage_sha256,
        "vertex_metadata.material.source_sha256": "a" * 64,
        "vertex_metadata.phase_b.archive_sha256": "b" * 64,
        "vertex_metadata.permutation_symmetry.cubic_physics_sha256": (
            vertex.cubic_physics_sha256
        ),
    }
    assert complete == explicit | expected_derived


def test_receipts_bind_parent_predictor_and_complete_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch)
    manifest, _manifest_path, _qualification_path = _manifest(tmp_path)
    tasks = pilot.predictor_pilot_tasks()
    parent = _receipt(tmp_path, manifest, tasks[0], None)
    child = _receipt(tmp_path, manifest, tasks[1], parent)
    document = pilot.validate_task_receipt(manifest, child)
    assert document["state"] == "passed"
    assert document["parent_receipt"]["task_id"] == "coupling-g0001"
    aliased_child = child.parent / ".." / child.parent.name / child.name
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.validate_task_receipt(manifest, aliased_child)

    aliased_parent = parent.parent / ".." / parent.parent.name / parent.name
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.build_task_receipt(
            manifest=manifest,
            task_id=tasks[1].task_id,
            scba_root=tmp_path / "tasks/coupling-g001/scba",
            parent_receipt=aliased_parent,
        )

    fields = tmp_path / "tasks/coupling-g001/scba/outcomes/scba-fields.h5"
    with fields.open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(pilot.MaterialNQ1PilotError, match="bindings|reproduce|tree"):
        pilot.validate_task_receipt(manifest, child)


@pytest.mark.parametrize(
    "mutation",
    ("missing-derived", "altered-derived", "extra-derived"),
)
def test_receipt_requires_exact_complete_checkpoint_source_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch, source_hash_mutation=mutation)
    manifest, _manifest_path, _qualification_path = _manifest(tmp_path)
    task = pilot.predictor_pilot_tasks()[0]
    scba_root = tmp_path / "tasks" / task.task_id / "scba"
    _scba_tree(scba_root, task)

    with pytest.raises(
        pilot.MaterialNQ1PilotError,
        match="complete material/source identity changed",
    ):
        pilot.build_task_receipt(
            manifest=manifest,
            task_id=task.task_id,
            scba_root=scba_root,
            parent_receipt=None,
        )


@pytest.mark.parametrize(
    "mutation, expected",
    (
        ("wrong-transform", "does not derive"),
        ("diagnostics", "final three passing rows"),
        ("infrastructure", "one immutable initial attempt"),
        ("scientific-replay", "independent scientific replay rejected"),
    ),
)
def test_receipt_rejects_forged_predictor_diagnostics_or_prior_infrastructure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch)
    case_root = tmp_path / mutation
    manifest, _manifest_path, _qualification_path = _manifest(case_root)
    tasks = pilot.predictor_pilot_tasks()
    selected = tasks[1] if mutation == "wrong-transform" else tasks[0]
    parent_path = (
        _receipt(case_root, manifest, tasks[0], None)
        if mutation == "wrong-transform"
        else None
    )
    root = case_root / "tasks" / selected.task_id / "scba"
    _scba_tree(
        root,
        selected,
        parent_receipt=parent_path,
        wrong_transformed_state=mutation == "wrong-transform",
        tampered_diagnostics=mutation == "diagnostics",
        prior_infrastructure_history=mutation == "infrastructure",
    )
    if mutation == "scientific-replay":
        def reject_replay(**_kwargs: object) -> None:
            raise pilot.MaterialNQ1PilotError(
                "independent scientific replay rejected synthetic fields"
            )

        monkeypatch.setattr(pilot, "_replay_scientific_diagnostics", reject_replay)
    with pytest.raises(pilot.MaterialNQ1PilotError, match=expected):
        pilot.build_task_receipt(
            manifest=manifest,
            task_id=selected.task_id,
            scba_root=root,
            parent_receipt=parent_path,
        )


@pytest.mark.parametrize("mutation", ("legacy-schema", "extra-key"))
def test_receipt_rejects_legacy_or_extended_scba_manifest_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch)
    case_root = tmp_path / mutation
    manifest, _manifest_path, _qualification_path = _manifest(case_root)
    task = pilot.predictor_pilot_tasks()[0]
    scba_root = case_root / "tasks" / task.task_id / "scba"
    _scba_tree(scba_root, task)
    path = scba_root / "campaign-manifest.json"
    document = pilot.load_canonical_json(path, name="SCBA manifest")
    if mutation == "legacy-schema":
        document["schema"] = "negfpy.sparse-nq1-phase-c-scba-campaign.v1"
    else:
        document["unexpected"] = True
    _rewrite_json(path, document)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="document schemas"):
        pilot.build_task_receipt(
            manifest=manifest,
            task_id=task.task_id,
            scba_root=scba_root,
            parent_receipt=None,
        )


def test_receipt_rejects_preflight_parent_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch)
    manifest, _manifest_path, _qualification_path = _manifest(tmp_path)
    tasks = pilot.predictor_pilot_tasks()
    parent = _receipt(tmp_path, manifest, tasks[0], None)
    scba_root = tmp_path / "tasks" / tasks[1].task_id / "scba"
    _scba_tree(scba_root, tasks[1], parent_receipt=parent)
    path = scba_root / "campaign-manifest.json"
    document = pilot.load_canonical_json(path, name="SCBA manifest")
    document["preflight"]["continuation_parent"] = None
    _rewrite_json(path, document)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="campaign contract"):
        pilot.build_task_receipt(
            manifest=manifest,
            task_id=tasks[1].task_id,
            scba_root=scba_root,
            parent_receipt=parent,
        )


@pytest.mark.parametrize(
    "mutation, expected",
    (
        ("inner-command", "fresh initial run"),
        ("inner-stdout-no-lf", "fresh initial run"),
        ("inner-stdout-double-lf", "fresh initial run"),
        ("sha256sums-order", "checksum evidence"),
        ("memory-zero", "available-memory observation"),
        ("memory-nonobservation", "material identity changed"),
        ("runtime-executable", "candidate identity changed"),
        ("runtime-version", "candidate identity changed"),
        ("runtime-distributions", "candidate identity changed"),
        ("runtime-threads", "candidate identity changed"),
        ("environment-lock", "candidate identity changed"),
        ("scientific-scope", "scientific scope changed"),
        ("next-gate", "misstates release certification"),
        ("extra-empty-directory", "checksum evidence"),
        ("legacy-running-name", "running marker"),
    ),
)
def test_receipt_rejects_nonproducer_runtime_command_and_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch)
    case_root = tmp_path / mutation
    manifest, _manifest_path, _qualification = _manifest(case_root)
    task = pilot.predictor_pilot_tasks()[0]
    scba_root = case_root / "tasks" / task.task_id / "scba"
    _scba_tree(scba_root, task)

    if mutation in {
        "inner-command",
        "inner-stdout-no-lf",
        "inner-stdout-double-lf",
    }:
        log_path = scba_root / "logs/scba.json"
        log = pilot.load_canonical_json(log_path, name="test SCBA log")
        assert log["command"][-1] == "--no-resume"
        assert log["stdout"].endswith("\n")
        assert not log["stdout"].endswith("\n\n")
        if mutation == "inner-command":
            log["command"] = log["command"][:-1]
        elif mutation == "inner-stdout-no-lf":
            log["stdout"] = log["stdout"][:-1]
        else:
            log["stdout"] += "\n"
        _rewrite_json(log_path, log)
        _reseal_scba_after_log_change(scba_root)
    elif mutation == "sha256sums-order":
        sums = scba_root / "SHA256SUMS"
        lines = _fixture_sha256sums(scba_root).splitlines(keepends=True)
        os.chmod(sums, 0o644)
        sums.write_bytes(b"".join(reversed(lines)))
        status_path = scba_root / "status.json"
        status = pilot.load_canonical_json(status_path, name="test SCBA status")
        status["sha256sums_sha256"] = pilot.sha256_file(sums)
        _rewrite_json(status_path, status)
    elif mutation in {"memory-zero", "memory-nonobservation"}:
        campaign_path = scba_root / "campaign-manifest.json"
        campaign = pilot.load_canonical_json(campaign_path, name="test SCBA manifest")
        if mutation == "memory-zero":
            campaign["preflight"]["source"]["memory"]["available_bytes"] = 0
        else:
            campaign["preflight"]["source"]["fixture_path"] += "-different"
        _rewrite_json(campaign_path, campaign)
    elif (
        mutation.startswith("runtime-")
        or mutation in {"environment-lock", "scientific-scope"}
    ):
        campaign_path = scba_root / "campaign-manifest.json"
        campaign = pilot.load_canonical_json(campaign_path, name="test SCBA manifest")
        if mutation == "runtime-executable":
            campaign["python"]["executable"] = "/different/python3.12"
        elif mutation == "runtime-version":
            campaign["python"]["version"] = "3.12.12"
        elif mutation == "runtime-distributions":
            campaign["distributions"]["numpy"] = "0"
        elif mutation == "runtime-threads":
            campaign["thread_limits"]["OMP_NUM_THREADS"] = "2"
        elif mutation == "environment-lock":
            campaign["environment_lock"]["path"] += ".alternate"
        else:
            campaign["scientific_scope"] = "different"
        _rewrite_json(campaign_path, campaign)
    elif mutation == "next-gate":
        report_path = scba_root / "scba-report.json"
        report = pilot.load_canonical_json(report_path, name="test SCBA report")
        report["next_gate"] = "release"
        _rewrite_json(report_path, report)
        _reseal_scba_after_report_change(scba_root)
    elif mutation == "extra-empty-directory":
        (scba_root / "unexpected-empty").mkdir()
    else:
        (scba_root / "status-running.json").rename(scba_root / "running.json")

    with pytest.raises(pilot.MaterialNQ1PilotError, match=expected):
        pilot.build_task_receipt(
            manifest=manifest,
            task_id=task.task_id,
            scba_root=scba_root,
            parent_receipt=None,
        )


def test_final_report_requires_two_anchors_and_g0025_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch)
    _synthetic_anchor_contract(tmp_path, monkeypatch)
    manifest, manifest_path, qualification = _manifest(tmp_path)
    preflight = _preflight(tmp_path, manifest, manifest_path, qualification)
    monkeypatch.setattr(
        pilot,
        "authenticate_anchor_source",
        lambda _manifest: {
            "coupling-g0001": {"passed": True},
            "coupling-g001": {"passed": True},
        },
    )
    tasks = pilot.predictor_pilot_tasks()
    receipts: dict[str, Path] = {}
    parent = None
    for task in tasks:
        parent = _receipt(tmp_path, manifest, task, parent)
        receipts[task.task_id] = parent

    comparisons: dict[str, Path] = {}
    for task in tasks[:2]:
        receipt = receipts[task.task_id]
        anchor_id = task.task_id
        anchor = manifest["anchor_source"]["records"][anchor_id]
        fields = tmp_path / "anchors" / anchor["fields"]["relative_path"]
        outcome = tmp_path / "anchors" / anchor["outcome"]["relative_path"]
        comparison_id = f"anchor-{task.task_id}"
        document = pilot.build_comparison(
            manifest=manifest,
            comparison_id=comparison_id,
            first_receipt_path=receipt,
            second_kind="release-018-anchor",
            second_id=anchor_id,
            second_fields_path=fields,
            second_outcome_path=outcome,
            second_hashes={
                "fields": anchor["fields"]["sha256"],
                "outcome": anchor["outcome"]["sha256"],
            },
        )
        path = tmp_path / "comparisons" / f"{comparison_id}.json"
        pilot.publish_json_once(path, document)
        comparisons[comparison_id] = path

    report = pilot.build_final_report(
        manifest=manifest,
        manifest_path=manifest_path,
        preflight_path=preflight,
        task_receipts=receipts,
        comparisons=comparisons,
        scheduler_receipts=_scheduler_receipts(tmp_path, manifest, receipts),
    )
    assert report["pilot_passed"] is True
    assert report["coupling_g0025_completed"] is True
    assert report["release_certified"] is False
    report_path = tmp_path / "pilot-report.json"
    pilot.publish_json_once(report_path, report)
    assert pilot.validate_final_report(manifest, report_path) == report

    forged = copy.deepcopy(report)
    forged["authorizes"]["source_or_scientific_control_changes_allowed"] = True
    forged_path = tmp_path / "forged-report.json"
    pilot.publish_json_once(forged_path, forged)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.validate_final_report(manifest, forged_path)
    _rewrite_json(report_path, forged)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="does not reproduce"):
        pilot.validate_final_report(manifest, report_path)
    _rewrite_json(report_path, report)
    aliased_report = report_path.parent / ".." / report_path.parent.name / report_path.name
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.validate_final_report(manifest, aliased_report)
    os.chmod(report_path, 0o644)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="mode 0444"):
        pilot.validate_final_report(manifest, report_path)
    os.chmod(report_path, 0o444)

    comparison_path = comparisons["anchor-coupling-g0001"]
    comparison = pilot.load_canonical_json(comparison_path, name="test comparison")
    forged_comparison = copy.deepcopy(comparison)
    forged_comparison["fields"]["passed"] = False
    forged_comparison["passed"] = True
    forged_comparison_path = tmp_path / "comparisons/forged-passed.json"
    pilot.publish_json_once(forged_comparison_path, forged_comparison)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.validate_comparison(manifest, forged_comparison_path)
    _rewrite_json(comparison_path, forged_comparison)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="does not reproduce"):
        pilot.validate_comparison(manifest, comparison_path)
    _rewrite_json(comparison_path, comparison)
    aliased_comparison = (
        comparison_path.parent
        / ".."
        / comparison_path.parent.name
        / comparison_path.name
    )
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.validate_comparison(manifest, aliased_comparison)
    os.chmod(comparison_path, 0o644)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="mode 0444"):
        pilot.validate_comparison(manifest, comparison_path)
    os.chmod(comparison_path, 0o444)

    self_scba = tmp_path / "tasks/coupling-g0001/scba"
    with pytest.raises(pilot.MaterialNQ1PilotError, match="frozen release-018 anchor"):
        pilot.build_comparison(
            manifest=manifest,
            comparison_id="anchor-coupling-g0001",
            first_receipt_path=receipts["coupling-g0001"],
            second_kind="release-018-anchor",
            second_id="coupling-g0001",
            second_fields_path=self_scba / "outcomes/scba-fields.h5",
            second_outcome_path=self_scba / "outcomes/scba.json",
            second_hashes={
                "fields": pilot.sha256_file(self_scba / "outcomes/scba-fields.h5"),
                "outcome": pilot.sha256_file(self_scba / "outcomes/scba.json"),
            },
        )

    audit_path = tmp_path / "pilot-audit.json"
    audit = pilot.build_audit(
        manifest=manifest,
        root=tmp_path,
        report_path=report_path,
        audit_path=audit_path,
    )
    pilot.publish_json_once(audit_path, audit)
    assert pilot.validate_audit(
        manifest, root=tmp_path, audit_path=audit_path
    ) == audit
    aliased_audit = audit_path.parent / ".." / audit_path.parent.name / audit_path.name
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.validate_audit(manifest, root=tmp_path, audit_path=aliased_audit)
    os.chmod(audit_path, 0o644)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="mode 0444"):
        pilot.validate_audit(manifest, root=tmp_path, audit_path=audit_path)
    os.chmod(audit_path, 0o444)


def test_scheduler_receipt_requires_exact_unique_terminal_f0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    _mock_checkpoints(monkeypatch)
    manifest, manifest_path, qualification = _manifest(tmp_path)
    _preflight(tmp_path, manifest, manifest_path, qualification)
    task_receipts: dict[str, Path] = {}
    parent = None
    for task in pilot.predictor_pilot_tasks():
        parent = _receipt(tmp_path, manifest, task, parent)
        task_receipts[task.task_id] = parent
    receipts = _scheduler_receipts(tmp_path, manifest, task_receipts)
    task_id = "coupling-g0001"
    assert pilot.validate_scheduler_receipt(manifest, task_id, receipts[task_id])[
        "exit_status"
    ] == 0
    aliased_scheduler = (
        receipts[task_id].parent
        / ".."
        / receipts[task_id].parent.name
        / receipts[task_id].name
    )
    with pytest.raises(pilot.MaterialNQ1PilotError, match="path is not fixed"):
        pilot.validate_scheduler_receipt(manifest, task_id, aliased_scheduler)
    canonical_receipt_path = receipts[task_id]
    original = pilot.load_canonical_json(canonical_receipt_path, name="scheduler")
    invalid = copy.deepcopy(original)
    invalid["exit_status"] = 2
    _rewrite_json(canonical_receipt_path, invalid)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="terminal F/0"):
        pilot.validate_scheduler_receipt(manifest, task_id, canonical_receipt_path)
    _rewrite_json(canonical_receipt_path, original)

    contradictory_raw = Path(original["raw_qstat_path"])
    original_raw = contradictory_raw.read_bytes()
    os.chmod(contradictory_raw, 0o644)
    contradictory_raw.write_bytes(original_raw + b"    Exit_status = 2\n")
    os.chmod(contradictory_raw, 0o444)
    contradictory = copy.deepcopy(original)
    contradictory["raw_qstat_sha256"] = pilot.sha256_file(contradictory_raw)
    _rewrite_json(canonical_receipt_path, contradictory)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="does not reproduce F/0"):
        pilot.validate_scheduler_receipt(manifest, task_id, canonical_receipt_path)
    os.chmod(contradictory_raw, 0o644)
    contradictory_raw.write_bytes(original_raw)
    os.chmod(contradictory_raw, 0o444)
    _rewrite_json(canonical_receipt_path, original)

    wrong_workload = copy.deepcopy(original)
    wrong_workload["task_receipt_path"] = str(task_receipts["coupling-g001"])
    wrong_workload["task_receipt_sha256"] = pilot.sha256_file(
        task_receipts["coupling-g001"]
    )
    _rewrite_json(canonical_receipt_path, wrong_workload)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="task binding"):
        pilot.validate_scheduler_receipt(manifest, task_id, canonical_receipt_path)
    _rewrite_json(canonical_receipt_path, original)

    canonical_pbs_stdout = Path(original["pbs_stdout_path"]).read_bytes()
    assert canonical_pbs_stdout.endswith(b"\n")
    assert not canonical_pbs_stdout.endswith(b"\n\n")
    for stream_name, forged_bytes in (
        ("pbs_stdout", b"completed\n"),
        ("pbs_stdout", canonical_pbs_stdout[:-1]),
        ("pbs_stdout", canonical_pbs_stdout + b"\n"),
        ("pbs_stderr", b"unexpected stderr\n"),
    ):
        stream_path = Path(original[f"{stream_name}_path"])
        original_stream = stream_path.read_bytes()
        os.chmod(stream_path, 0o644)
        stream_path.write_bytes(forged_bytes)
        os.chmod(stream_path, 0o444)
        forged_stream_receipt = copy.deepcopy(original)
        forged_stream_receipt[f"{stream_name}_sha256"] = pilot.sha256_file(
            stream_path
        )
        _rewrite_json(canonical_receipt_path, forged_stream_receipt)
        with pytest.raises(
            pilot.MaterialNQ1PilotError,
            match="scheduler submission identity does not reproduce",
        ):
            pilot.validate_scheduler_receipt(
                manifest, task_id, canonical_receipt_path
            )
        os.chmod(stream_path, 0o644)
        stream_path.write_bytes(original_stream)
        os.chmod(stream_path, 0o444)
        _rewrite_json(canonical_receipt_path, original)

    attack_pbs = Path(original["pbs_script_path"])
    os.chmod(attack_pbs, 0o644)
    attack_pbs.write_text(
        "#!/bin/bash\n"
        f"#PBS -N {original['job_name']}\n"
        f"#PBS -o {original['pbs_stdout_path']}\n"
        f"#PBS -e {original['pbs_stderr_path']}\n"
        "# run_material_nq1_predictor_pilot.py run-task "
        f"--task-id {task_id} --expected-commit {'1' * 40}\n"
        "exec /bin/true\n",
        encoding="utf-8",
    )
    os.chmod(attack_pbs, 0o444)
    attack_command = ["qsub", str(attack_pbs)]
    attack_command_sha = pilot.sha256_bytes(pilot.canonical_json(attack_command))
    attack_pre = pilot.load_canonical_json(
        original["pre_submission_path"], name="pre-submission"
    )
    attack_pre.update({
        "pbs_script_path": str(attack_pbs),
        "pbs_script_sha256": pilot.sha256_file(attack_pbs),
        "submission_command": attack_command,
        "submission_command_sha256": attack_command_sha,
    })
    attack_pre_path = Path(original["pre_submission_path"])
    _rewrite_json(attack_pre_path, attack_pre)
    original_submission = pilot.load_canonical_json(
        original["submission_path"], name="submission"
    )
    attack_submission = attack_pre | {
        "schema": pilot.SUBMISSION_SCHEMA,
        "job_id": original["job_id"],
        "pre_submission_path": str(attack_pre_path),
        "pre_submission_sha256": pilot.sha256_file(attack_pre_path),
        "qsub_stdout_path": original_submission["qsub_stdout_path"],
        "qsub_stdout_sha256": original_submission["qsub_stdout_sha256"],
        "qsub_stderr_path": original_submission["qsub_stderr_path"],
        "qsub_stderr_sha256": original_submission["qsub_stderr_sha256"],
    }
    attack_submission_path = Path(original["submission_path"])
    _rewrite_json(attack_submission_path, attack_submission)
    attack_receipt = original | {
        "pre_submission_path": str(attack_pre_path),
        "pre_submission_sha256": pilot.sha256_file(attack_pre_path),
        "submission_path": str(attack_submission_path),
        "submission_sha256": pilot.sha256_file(attack_submission_path),
        "pbs_script_path": str(attack_pbs),
        "pbs_script_sha256": pilot.sha256_file(attack_pbs),
        "submission_command_sha256": attack_command_sha,
    }
    attack_receipt_path = canonical_receipt_path
    _rewrite_json(attack_receipt_path, attack_receipt)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="canonical workload"):
        pilot.validate_scheduler_receipt(manifest, task_id, attack_receipt_path)


def test_legacy_or_release_eligible_qualification_is_rejected(tmp_path: Path) -> None:
    manifest, _manifest_path, qualification_path = _manifest(tmp_path)
    qualification = pilot.load_canonical_json(
        qualification_path, name="qualification"
    )
    assert pilot.qualification_authorizes_pilot(manifest, qualification)
    forged_context = copy.deepcopy(qualification)
    context = forged_context["evidence"]["base_qualification_evidence"][
        "qualification_context"
    ]
    context["driver_files"]["pyproject.toml"]["sha256"] = "0" * 64
    context_payload = dict(context)
    context_payload.pop("qualification_context_sha256")
    context["qualification_context_sha256"] = pilot.sha256_bytes(
        pilot.canonical_json(context_payload)
    )
    assert not pilot.qualification_authorizes_pilot(manifest, forged_context)
    qualification["release_submission_eligible"] = True
    assert not pilot.qualification_authorizes_pilot(manifest, qualification)
    qualification["release_submission_eligible"] = False
    qualification["schema"] = "negfpy.material-nq1-candidate-qualification.v2"
    assert not pilot.qualification_authorizes_pilot(manifest, qualification)


def test_cli_qualification_matcher_requires_recursive_shared_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _manifest_path, qualification_path = _manifest(tmp_path)
    qualification = pilot.load_canonical_json(
        qualification_path, name="qualification"
    )

    def reject(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("forged evidence")

    monkeypatch.setattr(
        pilot_runner.predictor_qualification,
        "validate_predictor_qualification_report",
        reject,
    )
    assert not pilot_runner._qualification_matches(
        manifest, qualification, qualification_path
    )
    monkeypatch.setattr(
        pilot_runner.predictor_qualification,
        "validate_predictor_qualification_report",
        lambda document, **_kwargs: dict(document),
    )
    monkeypatch.setattr(
        pilot_runner,
        "_current_runtime_matches",
        lambda _manifest, _document: True,
    )
    assert pilot_runner._qualification_matches(
        manifest, qualification, qualification_path
    )
    os.chmod(qualification_path, 0o644)
    assert not pilot_runner._qualification_matches(
        manifest, qualification, qualification_path
    )
    os.chmod(qualification_path, 0o444)


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="qualified pilot runtime replay is exercised by the Python 3.12 suite",
)
def test_current_runtime_replays_qualified_package_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _manifest_path, qualification_path = _manifest(tmp_path)
    qualification = pilot.load_canonical_json(
        qualification_path, name="qualification"
    )
    package_bytes = b"numpy==2.5.1\n"
    runtime = {
        "executable": str(Path(sys.executable).resolve()),
        "executable_sha256": pilot.sha256_file(Path(sys.executable).resolve()),
        "version": pilot.PILOT_PYTHON_VERSION,
        "packages_sha256": pilot.sha256_bytes(package_bytes),
        "runtime_sha256": "a" * 64,
        "launch_environment": build_launch_environment(
            repo=Path(manifest["source_package_file"]).parents[2],
            executable=Path(sys.executable).resolve(), home=tmp_path / "home",
            tmpdir=tmp_path / "tmp", git_bin=tmp_path / "git-bin",
        ),
    }
    expected_environment = qualification_environment(
        runtime, repo=Path(manifest["source_package_file"]).parents[2]
    )
    monkeypatch.setattr(pilot_runner.os, "environ", expected_environment)
    monkeypatch.setattr(pilot_runner.sys, "executable", runtime["executable"])
    monkeypatch.setattr(
        pilot_runner,
        "qualified_python312_runtime",
        lambda _manifest, _qualification: (runtime, {}),
    )
    monkeypatch.setattr(
        pilot_runner.platform,
        "python_version",
        lambda: pilot.PILOT_PYTHON_VERSION,
    )
    monkeypatch.setattr(
        pilot_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=b"forged==1\n"
        ),
    )
    assert not pilot_runner._current_runtime_matches(manifest, qualification)
    monkeypatch.setattr(
        pilot_runner.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout=package_bytes
        ),
    )
    assert pilot_runner._current_runtime_matches(manifest, qualification)
    alias = tmp_path / "qualified-python-alias"
    alias.symlink_to(runtime["executable"])
    monkeypatch.setattr(pilot_runner.sys, "executable", str(alias))
    assert not pilot_runner._current_runtime_matches(manifest, qualification)
    monkeypatch.setattr(pilot_runner.sys, "executable", runtime["executable"])
    monkeypatch.setattr(pilot_runner.os, "environ", expected_environment | {"LD_PRELOAD": "/unqualified.so"})
    assert not pilot_runner._current_runtime_matches(manifest, qualification)


@pytest.mark.parametrize("change", ["opal", "extra", "missing", "runtime-marker"])
def test_pilot_launch_environment_drift_is_rejected_before_task_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    manifest, _manifest_path, qualification_path = _manifest(tmp_path)
    qualification = pilot.load_canonical_json(qualification_path, name="qualification")
    runtime = qualification["evidence"]["base_qualification_evidence"]["records"]["python312_runtime"]
    environment = qualification_environment(runtime, repo=tmp_path / "repo")
    if change == "opal":
        environment["OPAL_PREFIX"] = "/other/runtime"
    elif change == "extra":
        environment["OMPI_MCA_other_control"] = "1"
    elif change == "missing":
        environment.pop("TMPDIR")
    else:
        environment["NEGFPY_QUALIFICATION_RUNTIME_SHA256"] = "f" * 64
    monkeypatch.setattr(pilot_runner.os, "environ", environment)
    monkeypatch.setattr(pilot_runner, "_revalidate_bound_qualification", lambda *_args: (runtime, {}))
    monkeypatch.setattr(pilot_runner, "_verify_live_candidate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(pilot_runner, "_verify_prior_gates", lambda *_args: None)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="launch environment differs"):
        pilot_runner.run_task(SimpleNamespace(root=tmp_path, task_id="coupling-g0001"))
    assert not (tmp_path / "tasks/coupling-g0001").exists()


def test_task_receipt_requires_immutable_exact_launch_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pilot, "EXPECTED_FIELD_SHAPE", (3, 2, 2))
    manifest, _manifest_path, _qualification = _manifest(tmp_path)
    _mock_checkpoints(monkeypatch)
    task = pilot.predictor_pilot_tasks()[0]
    receipt_path = _receipt(tmp_path, manifest, task, None)
    launch_path = receipt_path.parent / "launch-environment.json"
    launch = pilot.load_canonical_json(launch_path, name="launch environment")
    assert pilot.validate_task_receipt(manifest, receipt_path)["launch_environment"]["sha256"] == pilot.sha256_file(launch_path)
    launch["environment"]["OPAL_PREFIX"] = "/changed/runtime"
    os.chmod(launch_path, 0o644)
    launch_path.write_bytes(pilot.canonical_json(launch))
    os.chmod(launch_path, 0o444)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="launch environment differs"):
        pilot.validate_task_receipt(manifest, receipt_path)


@pytest.mark.parametrize("target", ["runtime", "context"])
def test_historical_environment_unbound_qualification_is_rejected(
    tmp_path: Path, target: str,
) -> None:
    manifest, _manifest_path, qualification_path = _manifest(tmp_path)
    qualification = pilot.load_canonical_json(qualification_path, name="qualification")
    base = qualification["evidence"]["base_qualification_evidence"]
    context = base["qualification_context"]
    if target == "runtime":
        runtime = base["records"]["python312_runtime"]
        runtime["schema"] = "negfpy.material-nq1-python-runtime.v1"
        runtime.pop("launch_environment")
        context["runtimes"]["python312"] = runtime
    else:
        context["schema"] = "negfpy.material-nq1-qualification-context.v1"
    context["qualification_context_sha256"] = pilot.sha256_bytes(pilot.canonical_json({
        name: value for name, value in context.items() if name != "qualification_context_sha256"
    }))
    assert not pilot.qualification_authorizes_pilot(manifest, qualification)
