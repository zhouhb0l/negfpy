"""Deep-chain tests for the fail-closed release-config generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("generate_release_config.py")
SPEC = importlib.util.spec_from_file_location("report_release_config", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

COMMIT = "1" * 40
TREE_SHA = "2" * 64
ENVIRONMENT_SHA = "3" * 64
THREADS = dict(MODULE.THREAD_LIMITS)


def _compact(value, *, ensure_ascii: bool = True) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii" if ensure_ascii else "utf-8")


def _indented(value, *, ensure_ascii: bool = True) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii" if ensure_ascii else "utf-8")


def _hash(value) -> str:
    return hashlib.sha256(_compact(value)).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _write_json(path: Path, value, *, ensure_ascii: bool = True) -> None:
    _write_bytes(path, _indented(value, ensure_ascii=ensure_ascii))


def _write_envelope(path: Path, payload: dict) -> dict:
    document = {"payload": payload, "payload_sha256": _hash(payload)}
    _write_bytes(path, _compact(document) + b"\n")
    return document


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record(root: Path, path: Path) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": _file_hash(path),
    }


def _memory(profile: str, rank_count: int) -> dict:
    available = 1_000_000_000
    aggregate = rank_count * 1_000_000
    return {
        "schema": "negfpy.remote_toy_mpi.aggregate_memory_preflight.v1",
        "measurement": "linux_proc_meminfo_MemAvailable",
        "profile": profile,
        "rank_count": rank_count,
        "allowed_fraction": 0.70,
        "available_memory_bytes": available,
        "limit_bytes": int(0.70 * available),
        "aggregate_predicted_bytes": aggregate,
        "passed": True,
    }


def _runtime(rank_count: int) -> list[dict]:
    rows: list[dict] = []
    for rank in range(rank_count):
        package = rank % 2 if rank_count >= 2 else 0
        rows.append(
            {
                "rank": rank,
                "hostname": "fixture-host",
                "physical_cores": [
                    {
                        "physical_package_id": package,
                        "core_id": rank // 2,
                    }
                ],
                "one_physical_core": True,
                "thread_limits": THREADS,
            }
        )
    return rows


def _task_rows(profile: str) -> list[dict]:
    gate = "multidimensional.direct-zero-endpoint"
    descriptors: list[tuple[str, str, list[str], str | None]] = [
        ("preflight.harmonic-same-grid", "harmonic_preflight", [], None),
        ("preflight.backend-equality", "backend_preflight", [], None),
        (
            gate,
            "direct_zero_endpoint",
            ["preflight.harmonic-same-grid", "preflight.backend-equality"],
            None,
        ),
    ]
    previous: str | None = None
    for index in range(5):
        task_id = f"multidimensional.path-step-{index:02d}"
        dependencies = [gate] + ([] if previous is None else [previous])
        descriptors.append(
            (task_id, "multidimensional_case", dependencies, previous)
        )
        previous = task_id
    for index in range(7):
        descriptors.append(
            (
                f"multidimensional.independent-{index:02d}",
                "multidimensional_case",
                [gate],
                None,
            )
        )
    for index in range(13):
        descriptors.append(
            (
                f"open_three_mode.case-{index:02d}",
                "open_three_mode_case",
                [gate],
                None,
            )
        )

    identity_hashes = {"protocol_configuration_sha256": _hash({"profile": profile})}
    run_identity = _hash(
        {
            "source_revision": COMMIT,
            "source_tree_sha256": TREE_SHA,
            "environment_sha256": ENVIRONMENT_SHA,
            "identity_hashes": identity_hashes,
        }
    )
    tasks: list[dict] = []
    for ordinal, (task_id, kind, dependencies, parent) in enumerate(descriptors):
        hashes = {
            "configuration": _hash({"profile": profile, "task_id": task_id}),
            "problem": _hash({"task_id": task_id}),
        }
        payload = {
            "campaign_schema": "negfpy_protocol_v2_toy_mpi_campaign",
            "campaign_schema_version": 1,
            "protocol": "multidimensional_toy_gate_v2",
            "kind": kind,
            "dependencies": dependencies,
            "continuation_parent_task_id": parent,
        }
        task = {
            "task_id": task_id,
            "identity_hashes": hashes,
            "payload": payload,
        }
        task["task_identity_sha256"] = _hash(
            {"run_identity_sha256": run_identity, "ordinal": ordinal, **task}
        )
        tasks.append(task)
    return tasks


def _campaign_payload(profile: str) -> dict:
    identity_hashes = {"protocol_configuration_sha256": _hash({"profile": profile})}
    return {
        "schema_name": "negfpy_task_manifest",
        "schema_version": 1,
        "source_revision": COMMIT,
        "source_tree_sha256": TREE_SHA,
        "environment_sha256": ENVIRONMENT_SHA,
        "identity_hashes": identity_hashes,
        "metadata": {
            "schema": "negfpy_protocol_v2_toy_mpi_campaign",
            "schema_version": 1,
            "protocol": "multidimensional_toy_gate_v2",
            "certification_eligible": profile == "release",
            "multidimensional_outcome_count": 13,
            "open_system_case_count": 13,
            "continuation_path_count": 5,
        },
        "run_identity_sha256": _hash(
            {
                "source_revision": COMMIT,
                "source_tree_sha256": TREE_SHA,
                "environment_sha256": ENVIRONMENT_SHA,
                "identity_hashes": identity_hashes,
            }
        ),
        "tasks": _task_rows(profile),
    }


def _wave_index(ordinal: int) -> int:
    if ordinal < 2:
        return 0
    if ordinal == 2:
        return 1
    if ordinal < 8:
        return ordinal - 1
    return 7


def _summary(release: bool) -> dict:
    return {
        "code_identity_consistent": True,
        "environment_lock_consistent": True,
        "failures": [],
        "fixture_contract_passed": True,
        "multidimensional_release_certified": True,
        "open_three_mode_agreement_passed": True,
        "refinement_passed": True,
        "release_certified": release,
        "validation_tier": "release" if release else "scaling",
    }


def _strict_report(
    *,
    profile: str,
    rank_count: int,
    execution_id: str,
    campaign_sha: str,
    campaign_file_sha: str,
    execution_file_sha: str,
    execution_payload_sha: str,
    scientific_sha: str,
) -> dict:
    release = profile == "release"
    core = {
        "schema": "negfpy.anharmonic.production_toy_gate",
        "schema_version": 4,
        "summary": _summary(release),
        "environment_lock": {
            "passed": True,
            "multidimensional_environment_sha256": ENVIRONMENT_SHA,
            "open_environment_sha256": ENVIRONMENT_SHA,
        },
        "multidimensional_scba": {
            "provenance": {
                "clean": True,
                "complete": True,
                "passed": True,
                "identity": {
                    "git_revision": COMMIT,
                    "source_tree_sha256": TREE_SHA,
                },
            },
            "scientific_fixture": {
                "current": 1.25,
                "wall_seconds": 9.0 + rank_count,
                "checkpoint": {"transport": rank_count},
            },
        },
        "open_three_mode": {
            "runs": [
                {
                    "name": f"case-{index:02d}",
                    "linewidth": 0.01 * (index + 1),
                    "oracle_wall_seconds": float(index + rank_count),
                    "provenance": {
                        "clean": True,
                        "git_dirty": False,
                        "identity": {
                            "git_revision": COMMIT,
                            "source_tree_sha256": TREE_SHA,
                        },
                    },
                }
                for index in range(13)
            ]
        },
        "orchestration_provenance": {
            "campaign_manifest_sha256": campaign_sha,
            "campaign_manifest_file_sha256": campaign_file_sha,
            "execution_manifest_file_sha256": execution_file_sha,
            "execution_payload_sha256": execution_payload_sha,
            "execution_scientific_sha256": scientific_sha,
            "execution_id": execution_id,
            "mode": "serial" if rank_count == 1 else "mpi",
            "communicator_size": rank_count,
            "complete": True,
            "solver_and_gate_contract_unchanged": True,
        },
    }
    return {**core, "report_sha256": hashlib.sha256(_compact(core, ensure_ascii=False)).hexdigest()}


def _refresh_checksums(root: Path, *, excluded: set[str] | None = None) -> tuple[str, dict[str, str]]:
    excluded = set() if excluded is None else set(excluded)
    entries: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink() and relative not in excluded and relative != "SHA256SUMS":
            entries[relative] = _file_hash(path)
    raw = "".join(f"{digest}  {relative}\n" for relative, digest in sorted(entries.items())).encode("ascii")
    _write_bytes(root / "SHA256SUMS", raw)
    return hashlib.sha256(raw).hexdigest(), entries


class CanonicalFixture:
    def __init__(self, root: Path, *, artifact_nonce: bytes | None = None) -> None:
        self.root = root
        self.artifact_nonce = artifact_nonce
        self.artifacts = root / "artifacts"
        self.report = root / "report"
        self.scaling_root = self.artifacts / "scaling"
        self.release_root = self.artifacts / "release"
        self.release_execution_id = "release-fixture"
        self.release_report: Path | None = None
        self.release_runner: Path | None = None
        self._build_artifacts()
        self._build_report_trees()
        self.output = self.report / "release_config.tex"

    def _build_worker(self, worker_root: Path, *, profile: str, rank_count: int, execution_id: str) -> dict:
        campaign_path = worker_root / "campaign-manifest.json"
        campaign = _write_envelope(campaign_path, _campaign_payload(profile))
        campaign_sha = campaign["payload_sha256"]
        tasks = campaign["payload"]["tasks"]
        outcomes: list[dict] = []
        for ordinal, task in enumerate(tasks):
            task_id = task["task_id"]
            checkpoint = worker_root / "checkpoints" / f"task-{ordinal:02d}.h5"
            artifact = worker_root / "case-artifacts" / f"task-{ordinal:02d}.h5"
            _write_bytes(checkpoint, f"checkpoint:{profile}:{task_id}\n".encode("ascii"))
            _write_bytes(artifact, f"artifact:{profile}:{task_id}\n".encode("ascii"))
            outcomes.append(
                {
                    "task_id": task_id,
                    "status": "completed",
                    "wave_index": _wave_index(ordinal),
                    "resumed": False,
                    "checkpoint_relative_path": checkpoint.relative_to(worker_root).as_posix(),
                    "checkpoint_sha256": _file_hash(checkpoint),
                    "artifact_relative_path": artifact.relative_to(worker_root).as_posix(),
                    "artifact_sha256": _file_hash(artifact),
                    "artifact_validation": {
                        "schema": "fixture.case.v1",
                        "task_identity_sha256": task["task_identity_sha256"],
                        "passed": True,
                    },
                }
            )
        scientific = _hash(
            {
                "campaign_manifest_sha256": campaign_sha,
                "outcomes": [
                    {
                        "task_id": item["task_id"],
                        "status": item["status"],
                        "artifact_validation": item["artifact_validation"],
                    }
                    for item in outcomes
                ],
            }
        )
        execution_payload = {
            "schema_name": "negfpy_protocol_v2_toy_mpi_execution",
            "schema_version": 1,
            "execution_id": execution_id,
            "campaign_manifest_sha256": campaign_sha,
            "mode": "serial" if rank_count == 1 else "mpi",
            "communicator_size": rank_count,
            "outcomes": outcomes,
            "phase_execution_sha256": [_hash({"phase": index}) for index in range(8)],
            "scientific_sha256": scientific,
            "complete": True,
        }
        execution_path = worker_root / "campaign-executions" / f"{execution_id}.json"
        execution = _write_envelope(execution_path, execution_payload)
        report_path = worker_root / "production-reports" / f"{execution_id}.json"
        report = _strict_report(
            profile=profile,
            rank_count=rank_count,
            execution_id=execution_id,
            campaign_sha=campaign_sha,
            campaign_file_sha=_file_hash(campaign_path),
            execution_file_sha=_file_hash(execution_path),
            execution_payload_sha=execution["payload_sha256"],
            scientific_sha=scientific,
        )
        _write_json(report_path, report, ensure_ascii=False)
        worker_payload = {
            "schema": "negfpy.remote_toy_mpi.physics_result.v1",
            "profile": profile,
            "certification_eligible": profile == "release",
            "release_certified": profile == "release",
            "source_revision": COMMIT,
            "source_tree_sha256": TREE_SHA,
            "environment_sha256": ENVIRONMENT_SHA,
            "directory_mode": "fresh",
            "expected_rank_count": rank_count,
            "recorded_rank_count": rank_count,
            "execution_mode": "serial" if rank_count == 1 else "mpi",
            "execution_id": execution_id,
            "task_count": 28,
            "multidimensional_outcome_count": 13,
            "open_system_case_count": 13,
            "continuation_path_count": 5,
            "all_outcomes_completed": True,
            "campaign_manifest_sha256": campaign_sha,
            "execution_scientific_sha256": scientific,
            "production_toy_physics_sha256": MODULE._physics_sha256(report),
            "configuration_sha256": _hash({"profile": profile}),
            "campaign_manifest": _record(worker_root, campaign_path),
            "execution_manifest": _record(worker_root, execution_path),
            "strict_report": _record(worker_root, report_path),
            "memory_preflight": _memory(profile, rank_count),
            "rank_runtime": _runtime(rank_count),
            "thread_limits": THREADS,
        }
        worker_path = worker_root / "runner-results" / f"{execution_id}.json"
        worker = _write_envelope(worker_path, worker_payload)
        return {
            "path": worker_path,
            "document": worker,
            "payload": worker_payload,
            "campaign_path": campaign_path,
            "execution_path": execution_path,
            "report_path": report_path,
        }

    def _build_artifacts(self) -> None:
        walls = {1: 100.0, 2: 60.0, 4: 30.0, 8: 25.5, 16: 25.0}
        rank_evidence: list[dict] = []
        common_physics: str | None = None
        for rank in MODULE.RANK_LADDER:
            worker = self._build_worker(
                self.scaling_root / f"rank-{rank:03d}",
                profile="scaling",
                rank_count=rank,
                execution_id=f"scaling-rank-{rank:03d}",
            )
            physics = worker["payload"]["production_toy_physics_sha256"]
            common_physics = physics if common_physics is None else common_physics
            assert physics == common_physics
            stdout = self.scaling_root / "logs" / f"rank-{rank:03d}.stdout.log"
            stderr = self.scaling_root / "logs" / f"rank-{rank:03d}.stderr.log"
            _write_bytes(stdout, f"rank={rank}\n".encode("ascii"))
            _write_bytes(stderr, b"")
            rank_payload = {
                "schema": "negfpy.remote_toy_mpi.scaling_rank.v1",
                "rank_count": rank,
                "source_revision": COMMIT,
                "source_tree_sha256": TREE_SHA,
                "environment_sha256": ENVIRONMENT_SHA,
                "fresh_isolated_repetition": True,
                "resume_used": False,
                "outer_wall_seconds": walls[rank],
                "returncode": 0,
                "completed": True,
                "failure": None,
                "argv": ["/fixture/bin/mpiexec", "-n", str(rank)],
                "binding_arguments": ["--bind-to", "core"],
                "thread_limits": THREADS,
                "launcher_memory_preflight": _memory("scaling", rank),
                "launcher": {
                    "path": "/fixture/bin/mpiexec",
                    "resolved_path": "/fixture/bin/mpiexec",
                    "sha256": "a" * 64,
                },
                "python": {
                    "path": "/fixture/bin/python",
                    "resolved_path": "/fixture/bin/python",
                    "sha256": "b" * 64,
                },
                "stdout": _record(self.scaling_root, stdout),
                "stderr": _record(self.scaling_root, stderr),
                "worker_result_relative_path": worker["path"].relative_to(self.scaling_root).as_posix(),
                "worker_result_file_sha256": _file_hash(worker["path"]),
                "worker_result_payload_sha256": worker["document"]["payload_sha256"],
                "production_toy_physics_sha256": common_physics,
            }
            rank_path = self.scaling_root / f"rank-{rank:03d}-manifest.json"
            rank_document = _write_envelope(rank_path, rank_payload)
            rank_evidence.append(
                {
                    "rank_count": rank,
                    "manifest_path": rank_path.name,
                    "manifest_file_sha256": _file_hash(rank_path),
                    "manifest_payload_sha256": rank_document["payload_sha256"],
                    "worker_result_file_sha256": _file_hash(worker["path"]),
                    "campaign_manifest_file_sha256": _file_hash(worker["campaign_path"]),
                    "execution_manifest_file_sha256": _file_hash(worker["execution_path"]),
                    "strict_report_file_sha256": _file_hash(worker["report_path"]),
                    "harmonic_preflight": {"real_physics_executed": True, "passed": True},
                    "launcher_memory_preflight": rank_payload["launcher_memory_preflight"],
                    "worker_memory_preflight": worker["payload"]["memory_preflight"],
                    "rank_runtime": worker["payload"]["rank_runtime"],
                }
            )
        assert common_physics is not None
        measurements = [
            {
                "rank_count": rank,
                "wall_seconds": walls[rank],
                "completed": True,
                "numerically_equal": True,
            }
            for rank in MODULE.RANK_LADDER
        ]
        selection_payload = {
            "schema": "negfpy.remote_toy_mpi.scaling_selection.v1",
            "source_revision": COMMIT,
            "source_tree_sha256": TREE_SHA,
            "environment_sha256": ENVIRONMENT_SHA,
            "physical_core_count": 76,
            "socket_count": 2,
            "profile": "representative_noncertifying_full_28_task",
            "production_toy_physics_sha256": common_physics,
            "physics_equal_across_all_ranks": True,
            "fresh_isolated_repetitions": True,
            "selection": {
                "schema": "negfpy.remote_mpi_scaling.selection.v1",
                "rank_ladder": list(MODULE.RANK_LADDER),
                "measurements": measurements,
                "fastest_rank_count": 16,
                "fastest_wall_seconds": 25.0,
                "smallest_near_fastest_rank_count": 8,
                "candidate_speedup": 100.0 / 25.5,
                "toy_rank_count": 8,
                "silicon_rank_count": 8,
                "toy_mpi_enabled": True,
                "within_fastest_fraction": 0.05,
                "minimum_toy_speedup": 2.0,
            },
            "rank_evidence": rank_evidence,
        }
        self.scaling_selection = self.scaling_root / "scaling-selection.json"
        _write_envelope(self.scaling_selection, selection_payload)
        release = self._build_worker(
            self.release_root,
            profile="release",
            rank_count=8,
            execution_id=self.release_execution_id,
        )
        self.release_runner = release["path"]
        self.release_report = release["report_path"]
        if self.artifact_nonce is not None:
            _write_bytes(self.artifacts / "fixture-nonce.bin", self.artifact_nonce)
        self.artifact_anchor, entries = _refresh_checksums(
            self.artifacts, excluded={"fetch-receipt.json"}
        )
        _write_json(
            self.artifacts / "fetch-receipt.json",
            {
                "schema": "negfpy.remote_validation.fetch_receipt.v1",
                "commit": COMMIT,
                "verified_files": sorted(entries),
            },
        )

    def _build_report_trees(self) -> None:
        assert self.release_report is not None
        staging = self.report / "staging"
        for filename in MODULE.REQUIRED_TEX:
            _write_bytes(staging / "tex" / filename, f"% {filename}\n".encode("ascii"))
        shutil.copyfile(self.release_report, staging / "production_toy_gate.json")
        staged_records = {
            path.relative_to(staging).as_posix(): {"sha256": _file_hash(path)}
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        receipt_core = {
            "schema": "negfpy.anharmonic.production_toy_gate_import_receipt",
            "schema_version": 2,
            "source": {
                "commit": COMMIT,
                "source_tree_sha256": TREE_SHA,
                "environment_sha256": ENVIRONMENT_SHA,
                "report_file_sha256": _file_hash(self.release_report),
                "report_relative_path": "production_toy_gate.json",
                "sha256sums_sha256": self.artifact_anchor,
            },
            "release_contract": {
                "checkpoint_count": 13,
                "multidimensional_outcome_count": 13,
                "open_system_run_count": 13,
                "protocol": "multidimensional_toy_gate_v2",
                "protocol_version": 2,
                "release_certified": True,
                "schema_version": 4,
                "validation_tier": "release",
            },
            "staged_artifacts": staged_records,
        }
        _write_json(staging / "import_receipt.json", {**receipt_core, "receipt_sha256": _hash(receipt_core)})
        self.staging_anchor, _ = _refresh_checksums(staging)
        self.staging_receipt_sha = _file_hash(staging / "import_receipt.json")

        production = self.report / "production-figures"
        for filename in MODULE.PRODUCTION_FIGURES:
            _write_bytes(production / filename, f"figure:{filename}\n".encode("ascii"))
        figure_records = {}
        for stem in (
            "figure_open_oracle_born",
            "figure_scba_validation",
            "figure_passivity_backend_timing",
        ):
            figure_records[stem] = {
                extension: {
                    "path": f"{stem}.{extension}",
                    "sha256": _file_hash(production / f"{stem}.{extension}"),
                }
                for extension in ("pdf", "png")
            }
        render_core = {
            "schema": "negfpy.anharmonic.production_toy_gate_figure_render",
            "schema_version": 1,
            "source": {
                "commit": COMMIT,
                "staging_sha256sums_file_sha256": self.staging_anchor,
                "import_receipt_file_sha256": self.staging_receipt_sha,
                "report_file_sha256": _file_hash(self.release_report),
            },
            "figures": figure_records,
        }
        _write_json(production / "render_manifest.json", {**render_core, "manifest_sha256": _hash(render_core)})
        self.production_anchor, _ = _refresh_checksums(production)

        historical = self.report / "historical-figures"
        for filename in MODULE.HISTORICAL_FIGURES:
            _write_bytes(historical / filename, f"historical:{filename}\n".encode("ascii"))
        historical_receipt = {
            "schema": "negfpy.anharmonic.historical_paper_benchmark_figure_render.v1",
            "production_evidence": False,
            "release_certified": False,
            "status": "NONCERTIFYING_HISTORICAL_SENSITIVITY",
            "figure_sha256": {
                filename: _file_hash(historical / filename)
                for filename in MODULE.HISTORICAL_FIGURES
            },
            "output_contract": {
                "sha256sums_exact_coverage": True,
                "vector_pdf": True,
                "rgb_png_450_dpi": True,
            },
        }
        _write_json(historical / "historical_paper_benchmarks_render_manifest.json", historical_receipt)
        self.historical_anchor, _ = _refresh_checksums(historical)

    def kwargs(self) -> dict:
        assert self.release_runner is not None
        return {
            "artifact_root": self.artifacts,
            "expected_artifact_sha256sums_sha256": self.artifact_anchor,
            "scaling_selection_relative": self.scaling_selection.relative_to(self.artifacts).as_posix(),
            "release_runner_relative": self.release_runner.relative_to(self.artifacts).as_posix(),
            "staging_dir": "staging",
            "expected_staging_sha256sums_sha256": self.staging_anchor,
            "production_figure_dir": "production-figures",
            "expected_production_figure_sha256sums_sha256": self.production_anchor,
            "historical_figure_dir": "historical-figures",
            "expected_historical_figure_sha256sums_sha256": self.historical_anchor,
            "report_date": "26 July 2026",
            "output": self.output,
        }

    def refresh_artifact_anchor(self) -> None:
        self.artifact_anchor, entries = _refresh_checksums(
            self.artifacts, excluded={"fetch-receipt.json"}
        )
        _write_json(
            self.artifacts / "fetch-receipt.json",
            {
                "schema": "negfpy.remote_validation.fetch_receipt.v1",
                "commit": COMMIT,
                "verified_files": sorted(entries),
            },
        )


class ReleaseConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = CanonicalFixture(self.root / "trusted")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_generates_from_complete_canonical_chain(self) -> None:
        text = MODULE.generate_config(**self.fixture.kwargs())
        self.assertEqual(self.fixture.output.read_text(encoding="ascii"), text)
        self.assertIn(f"\\newcommand{{\\ExpectedToyGateCommit}}{{{COMMIT}}}", text)
        self.assertIn("\\newcommand{\\MPIReleaseRankCount}{8}", text)
        self.assertIn(
            f"\\newcommand{{\\ArtifactSHA256SUMSRawSHA}}{{{self.fixture.artifact_anchor}}}",
            text,
        )
        self.assertIn(
            f"\\newcommand{{\\StagingSHA256SUMSRawSHA}}{{{self.fixture.staging_anchor}}}",
            text,
        )
        self.assertFalse(any(self.fixture.report.glob(".release_config.tex.*.tmp")))

    def test_projection_matches_exact_installed_production_function(self) -> None:
        from negfpy.anharmonic.toy._mpi_production_gate_physics import (
            production_toy_physics_sha256,
        )

        assert self.fixture.release_report is not None
        report = _read_json(self.fixture.release_report)
        self.assertEqual(
            MODULE._physics_sha256(report),
            production_toy_physics_sha256(report),
        )
        changed_timing = json.loads(json.dumps(report))
        changed_timing["multidimensional_scba"]["scientific_fixture"]["wall_seconds"] += 100
        self.assertEqual(MODULE._physics_sha256(changed_timing), MODULE._physics_sha256(report))
        changed_physics = json.loads(json.dumps(report))
        changed_physics["multidimensional_scba"]["scientific_fixture"]["current"] += 1
        self.assertNotEqual(MODULE._physics_sha256(changed_physics), MODULE._physics_sha256(report))

    def test_rejects_fully_self_consistent_forged_root_without_trusted_anchor(self) -> None:
        forged = CanonicalFixture(self.root / "forged", artifact_nonce=b"forged root\\n")
        kwargs = forged.kwargs()
        kwargs["expected_artifact_sha256sums_sha256"] = self.fixture.artifact_anchor
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "caller anchor mismatch"):
            MODULE.generate_config(**kwargs)

    def test_rejects_listed_byte_tamper(self) -> None:
        target = next((self.fixture.artifacts / "release" / "checkpoints").glob("*.h5"))
        target.write_bytes(target.read_bytes() + b"tamper")
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "SHA256 mismatch"):
            MODULE.generate_config(**self.fixture.kwargs())

    def test_rejects_unlisted_regular_file(self) -> None:
        _write_bytes(self.fixture.artifacts / "unlisted.txt", b"forgery\n")
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "coverage mismatch"):
            MODULE.generate_config(**self.fixture.kwargs())

    def test_rejects_symlinked_evidence(self) -> None:
        os.symlink("SHA256SUMS", self.fixture.artifacts / "forged-link")
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "symlink"):
            MODULE.generate_config(**self.fixture.kwargs())

    def test_rejects_parent_component_before_normalization(self) -> None:
        (self.fixture.artifacts / "subdir").mkdir()
        kwargs = self.fixture.kwargs()
        kwargs["artifact_root"] = self.fixture.artifacts / "subdir" / ".."
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "parent-directory"):
            MODULE.generate_config(**kwargs)

    def test_rejects_each_wrong_external_anchor(self) -> None:
        for name in (
            "expected_artifact_sha256sums_sha256",
            "expected_staging_sha256sums_sha256",
            "expected_production_figure_sha256sums_sha256",
            "expected_historical_figure_sha256sums_sha256",
        ):
            with self.subTest(anchor=name):
                kwargs = self.fixture.kwargs()
                kwargs[name] = "f" * 64
                with self.assertRaisesRegex(MODULE.ReleaseConfigError, "caller anchor mismatch"):
                    MODULE.generate_config(**kwargs)

    def test_rejects_nested_forged_release_report_even_after_rehash(self) -> None:
        assert self.fixture.release_report is not None
        assert self.fixture.release_runner is not None
        report = _read_json(self.fixture.release_report)
        report["summary"]["release_certified"] = False
        core = {key: value for key, value in report.items() if key != "report_sha256"}
        report["report_sha256"] = hashlib.sha256(_compact(core, ensure_ascii=False)).hexdigest()
        _write_json(self.fixture.release_report, report, ensure_ascii=False)
        worker_document = _read_json(self.fixture.release_runner)
        worker = worker_document["payload"]
        worker["strict_report"]["sha256"] = _file_hash(self.fixture.release_report)
        worker["production_toy_physics_sha256"] = MODULE._physics_sha256(report)
        _write_envelope(self.fixture.release_runner, worker)
        self.fixture.refresh_artifact_anchor()
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "release certification state"):
            MODULE.generate_config(**self.fixture.kwargs())

    def test_rejects_noncanonical_checksum_manifest(self) -> None:
        manifest = self.fixture.artifacts / "SHA256SUMS"
        lines = manifest.read_text(encoding="ascii").splitlines()
        manifest.write_text("\n".join(reversed(lines)) + "\n", encoding="ascii")
        kwargs = self.fixture.kwargs()
        kwargs["expected_artifact_sha256sums_sha256"] = _file_hash(manifest)
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "not path-sorted"):
            MODULE.generate_config(**kwargs)


if __name__ == "__main__":
    unittest.main()
