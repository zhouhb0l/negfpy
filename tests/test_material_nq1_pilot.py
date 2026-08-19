from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import _material_nq1_pilot as pilot
from negfpy.anharmonic._material_nq1_campaign import silicon_nq1_release_tasks

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import run_material_nq1_predictor_pilot as pilot_runner


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
        "task_count": 3,
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
            "anchor_current_rtol": pilot.CURRENT_RTOL,
            "anchor_current_atol": pilot.CURRENT_ATOL,
        },
        "anchor_contract": anchor_contract,
        "anchor_contract_sha256": pilot.sha256_bytes(
            pilot.canonical_json(anchor_contract)
        ),
        "anchor_authentication_gate": "three-task-pilot-preflight",
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
    records: dict[str, object] = {}
    for label, version in (("python310", "3.10.0"), ("python312", "3.12.0")):
        executable = repo / "tools" / label
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_bytes((label + "\n").encode("ascii"))
        records[f"{label}_runtime"] = {
            "schema": "negfpy.material-nq1-python-runtime.v1",
            "label": label,
            "version": version,
            "executable": str(executable),
            "executable_sha256": pilot.sha256_file(executable),
            "packages_sha256": "b" * 64,
            "runtime_sha256": "c" * 64,
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
    pilot_driver = repo / "scripts/run_material_nq1_predictor_pilot.py"
    pilot_driver.write_bytes(b"synthetic pilot driver\n")
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
        "schema": "negfpy.material-nq1-qualification-context.v1",
        "repo_root": str(repo),
        "source": {
            "commit": "1" * 40,
            "tree": "e" * 40,
            "dirty": False,
            "package_file": str(package_file),
        },
        "source_tree_sha256": "2" * 64,
        "environment_lock": {"sha256": "3" * 64},
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
        "environment_sha256": "3" * 64,
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
                    "python310": "3.10.0", "python312": "3.12.0",
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
        "next_gate": "three-task-pilot",
    })


def _manifest(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    qualification = tmp_path / "qualification.json"
    _qualification(qualification, tmp_path / "anchors")
    manifest = pilot.build_pilot_manifest(
        source_revision="1" * 40,
        source_git_tree="e" * 40,
        source_package_file=tmp_path / "repo/src/negfpy/__init__.py",
        source_tree_sha256="2" * 64,
        environment_sha256="3" * 64,
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
        "environment_sha256": "3" * 64,
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
        "scientific_scope": "synthetic exact pilot test",
        "source": {
            "commit": "1" * 40,
            "tree": "e" * 40,
            "dirty": False,
            "package_file": str(candidate_root / "repo/src/negfpy/__init__.py"),
        },
        "source_tree_sha256": "2" * 64,
        "environment_lock": {
            "path": str(candidate_root / "inputs/environment.lock"),
            "sha256": "3" * 64,
            "format": "@EXPLICIT",
        },
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
            "version": "3.12.0",
        },
        "distributions": {},
        "thread_limits": {},
    }
    pilot.publish_json_once(root / "campaign-manifest.json", manifest)
    manifest_sha = pilot.sha256_file(root / "campaign-manifest.json")
    pilot.publish_json_once(root / "running.json", {
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
    outcome = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-case.v1",
        "campaign_manifest_sha256": manifest_sha,
        "preflight_source": preflight_source,
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
        "command": ["/python", "/repo/scripts/validate_sparse_nq1_phase_c_scba.py"],
        "supervision": supervision,
        "stdout": "",
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
        "next_gate": "synthetic",
    }
    pilot.publish_json_once(root / "scba-report.json", report)
    (root / "SHA256SUMS").write_bytes(b"synthetic sums\n")
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


def _mock_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
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
        }
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
        identity = SimpleNamespace(
            identity_sha256=pilot.sha256_bytes(task_id.encode("ascii")),
            run_id=task_id,
            source_tree_sha256="2" * 64,
            environment_sha256="3" * 64,
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
        lambda **_kwargs: None,
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
        pbs_stdout.write_text("completed\n", encoding="utf-8")
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
            "diagnostics": {"passed": True},
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


def test_pilot_is_exact_first_three_release_tasks() -> None:
    pilot_tasks = pilot.predictor_pilot_tasks()
    release_prefix = silicon_nq1_release_tasks()[:3]
    assert pilot_tasks == release_prefix
    assert tuple(task.task_id for task in pilot_tasks) == (
        "coupling-g0001",
        "coupling-g001",
        "coupling-g0025",
    )
    assert tuple(task.continuation_predictor for task in pilot_tasks) == (
        None,
        "coupling-square",
        "coupling-square",
    )


def test_canonical_pbs_resources_have_headroom_above_internal_watchdogs() -> None:
    payload = pilot._canonical_pbs_script(
        task={"scheduler_job_name": "nq1p-g0001"},
        stdout_path=Path("/evidence/task.stdout"),
        stderr_path=Path("/evidence/task.stderr"),
        workload={
            "repo_root": "/candidate",
            "command": ["/python", "/candidate/pilot.py", "run-task"],
        },
    ).decode("utf-8")
    assert "#PBS -l select=1:ncpus=2:mem=16gb\n" in payload
    assert "#PBS -l walltime=50:00:00\n" in payload
    assert pilot.PILOT_PBS_NCPUS == 2
    assert pilot.PILOT_PBS_MEMORY == "16gb"
    assert pilot.PILOT_PBS_WALLTIME == "50:00:00"
    assert pilot.PILOT_WALL_LIMIT_SECONDS < 50 * 60 * 60
    assert pilot.PILOT_RSS_LIMIT_BYTES < 16 * 1000**3


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
    with pytest.raises(pilot.MaterialNQ1PilotError, match="inputs changed"):
        pilot_runner._verify_live_candidate(SimpleNamespace(), manifest)

    task_root = tmp_path / "current-task"
    (task_root / "scba/checkpoint").mkdir(parents=True)
    (task_root / "scba/checkpoint/sigma-pp.h5").write_bytes(b"prior attempt")
    with pytest.raises(pilot.MaterialNQ1PilotError, match="already has evidence"):
        pilot_runner._require_virgin_task_root(task_root, "coupling-g0001")

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

    fields = tmp_path / "tasks/coupling-g001/scba/outcomes/scba-fields.h5"
    with fields.open("ab") as stream:
        stream.write(b"mutation")
    with pytest.raises(pilot.MaterialNQ1PilotError, match="bindings|reproduce|tree"):
        pilot.validate_task_receipt(manifest, child)


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
    parent = _receipt(case_root, manifest, tasks[0], None)
    selected = tasks[1] if mutation == "wrong-transform" else tasks[0]
    parent_path = parent if mutation == "wrong-transform" else None
    root = case_root / "forged" / selected.task_id / "scba"
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
    scba_root = case_root / "forged/scba"
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
    scba_root = tmp_path / "forged-child/scba"
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
    with pytest.raises(pilot.MaterialNQ1PilotError, match="does not reproduce"):
        pilot.validate_final_report(manifest, forged_path)

    comparison_path = comparisons["anchor-coupling-g0001"]
    comparison = pilot.load_canonical_json(comparison_path, name="test comparison")
    forged_comparison = copy.deepcopy(comparison)
    forged_comparison["fields"]["passed"] = False
    forged_comparison["passed"] = True
    forged_comparison_path = tmp_path / "comparisons/forged-passed.json"
    pilot.publish_json_once(forged_comparison_path, forged_comparison)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="does not reproduce"):
        pilot.validate_comparison(manifest, forged_comparison_path)

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
    invalid = pilot.load_canonical_json(receipts[task_id], name="scheduler")
    invalid["exit_status"] = 2
    invalid_path = tmp_path / "scheduler/invalid-exit.receipt.json"
    pilot.publish_json_once(invalid_path, invalid)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="terminal F/0"):
        pilot.validate_scheduler_receipt(manifest, task_id, invalid_path)

    contradictory_raw = tmp_path / "scheduler/contradictory.qstat.txt"
    original_raw = Path(invalid["raw_qstat_path"]).read_text(encoding="utf-8")
    contradictory_raw.write_text(
        original_raw + "    Exit_status = 2\n", encoding="utf-8"
    )
    os.chmod(contradictory_raw, 0o444)
    contradictory = pilot.load_canonical_json(receipts[task_id], name="scheduler")
    contradictory["raw_qstat_path"] = str(contradictory_raw)
    contradictory["raw_qstat_sha256"] = pilot.sha256_file(contradictory_raw)
    contradictory_path = tmp_path / "scheduler/contradictory.receipt.json"
    pilot.publish_json_once(contradictory_path, contradictory)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="does not reproduce F/0"):
        pilot.validate_scheduler_receipt(manifest, task_id, contradictory_path)

    wrong_workload = pilot.load_canonical_json(receipts[task_id], name="scheduler")
    wrong_workload["task_receipt_path"] = str(task_receipts["coupling-g001"])
    wrong_workload["task_receipt_sha256"] = pilot.sha256_file(
        task_receipts["coupling-g001"]
    )
    wrong_workload_path = tmp_path / "scheduler/wrong-workload.receipt.json"
    pilot.publish_json_once(wrong_workload_path, wrong_workload)
    with pytest.raises(pilot.MaterialNQ1PilotError, match="another workload"):
        pilot.validate_scheduler_receipt(manifest, task_id, wrong_workload_path)

    original = pilot.load_canonical_json(receipts[task_id], name="scheduler")
    attack_pbs = tmp_path / "scheduler/decoy-task.pbs"
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
    attack_pre_path = tmp_path / "scheduler/decoy-pre-submission.json"
    pilot.publish_json_once(attack_pre_path, attack_pre)
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
    attack_submission_path = tmp_path / "scheduler/decoy-submission.json"
    pilot.publish_json_once(attack_submission_path, attack_submission)
    attack_receipt = original | {
        "pre_submission_path": str(attack_pre_path),
        "pre_submission_sha256": pilot.sha256_file(attack_pre_path),
        "submission_path": str(attack_submission_path),
        "submission_sha256": pilot.sha256_file(attack_submission_path),
        "pbs_script_path": str(attack_pbs),
        "pbs_script_sha256": pilot.sha256_file(attack_pbs),
        "submission_command_sha256": attack_command_sha,
    }
    attack_receipt_path = tmp_path / "scheduler/decoy.receipt.json"
    pilot.publish_json_once(attack_receipt_path, attack_receipt)
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
    assert pilot_runner._qualification_matches(
        manifest, qualification, qualification_path
    )
