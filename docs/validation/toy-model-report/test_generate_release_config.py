"""Focused tests for the fail-closed release-config generator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("generate_release_config.py")
SPEC = importlib.util.spec_from_file_location("report_release_config", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

COMMIT = "1" * 40
TREE_SHA = "2" * 64
ENVIRONMENT_SHA = "3" * 64
SCALING_PHYSICS_SHA = "4" * 64
RELEASE_PHYSICS_SHA = "5" * 64
THREADS = dict(MODULE.EXPECTED_THREADS)


def _payload_bytes(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _write_envelope(path: Path, payload: dict) -> None:
    document = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_payload_bytes(payload)).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_payload_bytes(document) + b"\n")


def _memory(rank_count: int, *, passed: bool = True) -> dict:
    available = 1_000_000_000
    aggregate = rank_count * 1_000_000
    return {
        "schema": "negfpy.remote_toy_mpi.aggregate_memory_preflight.v1",
        "measured_utc": "2026-07-26T00:00:00Z",
        "measurement": "linux_proc_meminfo_MemAvailable",
        "profile": "scaling",
        "rank_count": rank_count,
        "allowed_fraction": 0.70,
        "available_memory_bytes": available,
        "limit_bytes": int(0.70 * available),
        "per_rank_predicted_bytes": 1_000_000,
        "aggregate_predicted_bytes": aggregate if passed else available,
        "contributors": [],
        "passed": passed,
    }


def _runtime(rank_count: int) -> list[dict]:
    records = []
    for rank in range(rank_count):
        package = rank % 2 if rank_count >= 2 else 0
        core = rank // 2
        records.append(
            {
                "rank": rank,
                "hostname": "fixture-host",
                "logical_cpu_ids": [rank],
                "logical_cpu_topology": [
                    {
                        "logical_cpu_id": rank,
                        "physical_package_id": package,
                        "core_id": core,
                    }
                ],
                "physical_cores": [
                    {
                        "physical_package_id": package,
                        "core_id": core,
                    }
                ],
                "one_physical_core": True,
                "thread_limits": THREADS,
                "python": {"path": "/environment/bin/python"},
                "python_argv": ["runner"],
            }
        )
    return records


def _scaling_payload(root: Path) -> dict:
    wall = {1: 100.0, 2: 60.0, 4: 30.0, 8: 26.0, 16: 25.0}
    evidence = []
    for rank in MODULE.RANK_LADDER:
        manifest = root / f"rank-{rank:03d}-manifest.json"
        manifest.write_bytes(f"rank={rank}\n".encode("ascii"))
        evidence.append(
            {
                "rank_count": rank,
                "manifest_path": manifest.name,
                "manifest_file_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "manifest_payload_sha256": "6" * 64,
                "worker_result_file_sha256": "7" * 64,
                "campaign_manifest_file_sha256": "8" * 64,
                "execution_manifest_file_sha256": "9" * 64,
                "strict_report_file_sha256": "a" * 64,
                "harmonic_preflight": {"passed": True},
                "launcher_memory_preflight": _memory(rank),
                "worker_memory_preflight": _memory(rank),
                "rank_runtime": _runtime(rank),
            }
        )
    measurements = [
        {
            "rank_count": rank,
            "wall_seconds": wall[rank],
            "completed": True,
            "numerically_equal": True,
        }
        for rank in MODULE.RANK_LADDER
    ]
    return {
        "schema": "negfpy.remote_toy_mpi.scaling_selection.v1",
        "created_utc": "2026-07-26T00:00:00Z",
        "source_revision": COMMIT,
        "source_tree_sha256": TREE_SHA,
        "environment_sha256": ENVIRONMENT_SHA,
        "physical_core_count": 76,
        "socket_count": 2,
        "profile": "representative_noncertifying_full_28_task",
        "production_toy_physics_sha256": SCALING_PHYSICS_SHA,
        "physics_equal_across_all_ranks": True,
        "fresh_isolated_repetitions": True,
        "selection": {
            "schema": "negfpy.remote_mpi_scaling.selection.v1",
            "rank_ladder": list(MODULE.RANK_LADDER),
            "measurements": measurements,
            "fastest_rank_count": 16,
            "fastest_wall_seconds": 25.0,
            "smallest_near_fastest_rank_count": 8,
            "candidate_speedup": 100.0 / 26.0,
            "toy_rank_count": 8,
            "silicon_rank_count": 8,
            "toy_mpi_enabled": True,
            "within_fastest_fraction": 0.05,
            "minimum_toy_speedup": 2.0,
        },
        "toy_execution": {
            "rank_count": 8,
            "mode": "task_mpi",
            "minimum_speedup_satisfied": True,
        },
        "silicon_execution": {"rank_count": 8, "mode": "task_mpi"},
        "rank_evidence": evidence,
    }


def _release_payload() -> dict:
    return {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "release",
        "certification_eligible": True,
        "release_certified": True,
        "source_revision": COMMIT,
        "source_tree_sha256": TREE_SHA,
        "environment_sha256": ENVIRONMENT_SHA,
        "directory_mode": "fresh",
        "expected_rank_count": 8,
        "recorded_rank_count": 8,
        "execution_mode": "mpi",
        "execution_id": "release-fixture",
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
        "campaign_manifest_sha256": "b" * 64,
        "execution_scientific_sha256": "c" * 64,
        "production_toy_physics_sha256": RELEASE_PHYSICS_SHA,
        "configuration_sha256": "d" * 64,
        "campaign_manifest": {"relative_path": "campaign-manifest.json", "sha256": "e" * 64},
        "execution_manifest": {
            "relative_path": "campaign-executions/release-fixture.json",
            "sha256": "f" * 64,
        },
        "strict_report": {
            "relative_path": "production-reports/release-fixture.json",
            "sha256": "0" * 64,
        },
        "memory_preflight": {**_memory(8), "profile": "release"},
        "rank_runtime": _runtime(8),
        "thread_limits": THREADS,
    }


class ReleaseConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.report = self.root / "report"
        self.report.mkdir()
        for name in MODULE.REQUIRED_TEX:
            path = self.report / "staging" / "tex" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("% fixture\n", encoding="ascii")
        for name in MODULE.REQUIRED_PRODUCTION_FIGURES:
            path = self.report / "production-figures" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"%PDF-fixture\n")
        for name in MODULE.REQUIRED_HISTORICAL_FIGURES:
            path = self.report / "historical-figures" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"%PDF-fixture\n")
        self.scaling_path = self.root / "scaling" / "scaling-selection.json"
        self.scaling_path.parent.mkdir()
        self.scaling = _scaling_payload(self.scaling_path.parent)
        _write_envelope(self.scaling_path, self.scaling)
        self.runner_path = self.root / "release-runner.json"
        self.runner = _release_payload()
        _write_envelope(self.runner_path, self.runner)
        self.output = self.report / "release_config.tex"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def generate(self) -> str:
        return MODULE.generate_config(
            scaling_selection=self.scaling_path,
            release_runner=self.runner_path,
            staging_dir="staging",
            production_figure_dir="production-figures",
            historical_figure_dir="historical-figures",
            report_date="26 July 2026",
            output=self.output,
        )

    def test_generates_exact_atomic_config(self) -> None:
        text = self.generate()
        self.assertEqual(self.output.read_text(encoding="ascii"), text)
        self.assertIn(f"\\newcommand{{\\ExpectedToyGateCommit}}{{{COMMIT}}}", text)
        self.assertIn("\\newcommand{\\MPIReleaseRankCount}{8}", text)
        self.assertIn("\\newcommand{\\MPICandidateSpeedup}{3.846154}", text)
        self.assertIn(
            f"\\newcommand{{\\MPIScalingPhysicsSHA}}{{{SCALING_PHYSICS_SHA}}}",
            text,
        )
        self.assertFalse(any(self.report.glob(".release_config.tex.*.tmp")))

    def test_rejects_false_rank_physics_equality(self) -> None:
        self.scaling["physics_equal_across_all_ranks"] = False
        _write_envelope(self.scaling_path, self.scaling)
        with self.assertRaisesRegex(
            MODULE.ReleaseConfigError, "physics equality"
        ):
            self.generate()

    def test_rejects_release_identity_mismatch(self) -> None:
        self.runner["source_revision"] = "a" * 40
        _write_envelope(self.runner_path, self.runner)
        with self.assertRaisesRegex(
            MODULE.ReleaseConfigError, "identity or topology"
        ):
            self.generate()

    def test_rejects_memory_over_limit(self) -> None:
        self.scaling["rank_evidence"][0]["launcher_memory_preflight"] = _memory(
            1, passed=False
        )
        _write_envelope(self.scaling_path, self.scaling)
        with self.assertRaisesRegex(MODULE.ReleaseConfigError, "did not pass"):
            self.generate()


if __name__ == "__main__":
    unittest.main()
