from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import pytest

from negfpy.anharmonic.toy._mpi_production_gate import (
    build_production_toy_mpi_campaign,
)
from negfpy.anharmonic.toy.multidimensional_release import (
    full_multidimensional_release_config,
)


SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
runner = importlib.import_module("remote_toy_mpi_runner")

_SOURCE_REVISION = "a" * 40
_SOURCE_TREE_SHA256 = "b" * 64
_ENVIRONMENT_SHA256 = "c" * 64
_PHYSICS_SHA256 = "d" * 64


def test_representative_scaling_profile_has_exact_full_28_task_topology() -> None:
    config, fixtures = runner.representative_scaling_profile()
    release_config = full_multidimensional_release_config()
    campaign = build_production_toy_mpi_campaign(
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=_ENVIRONMENT_SHA256,
        config=config,
        open_fixtures=fixtures,
    )

    assert config.profile == "full_preregistered"
    assert not config.certification_eligible
    assert config.executed_g_values == (0.0, 0.025, 0.05)
    assert config.executed_bias_values == (0.0, 0.1, 0.2)
    assert config.spec.spacings == (0.0015625, 0.00078125, 0.000390625)
    assert config.spec.cutoffs == (5.0, 6.0)
    assert config.spec.etas == (0.003125, 0.0015625, 0.00078125)
    assert config.spec.mixings == (0.2, 0.1)
    assert config.numerical.spacing == 0.00078125
    assert config.numerical.cutoff == 5.0
    assert config.numerical.eta_lead == 0.0015625
    assert config.numerical.mixing == 0.2
    release_document = release_config.as_dict()
    release_document["certification_eligible"] = False
    assert config.as_dict() == release_document
    assert config.max_iterations == 200
    assert config.min_iterations == 3
    assert config.consecutive_passes == 3
    assert config.sigma_rtol == 1.0e-8
    assert config.green_rtol == 1.0e-8
    assert config.identity_rtol == 1.0e-8
    assert config.current_rtol == 1.0e-5
    assert config.current_atol == 1.0e-10
    assert config.tail_rtol == 1.0e-6
    assert config.production_backend == "fft"
    assert config.production_hilbert_backend == "fft"
    assert config.surface_gf_method == "analytic_diagonal"
    assert config.run_preregistered_refinements
    assert config.run_bounded_backend_validation

    assert len(campaign.manifest.tasks) == 28
    assert campaign.manifest.metadata["multidimensional_outcome_count"] == 13
    assert campaign.manifest.metadata["open_system_case_count"] == 13
    assert campaign.manifest.metadata["continuation_path_count"] == 5
    assert tuple(len(wave) for wave in campaign.waves) == (2, 1, 1, 1, 1, 1, 1, 20)

    assert len(fixtures) == 13
    base = fixtures[0][1]
    assert base.grid.step == 0.00125
    assert base.grid.cutoff == 5.0
    assert base.eta_lead == 0.0008
    assert {fixture.temperature for _, fixture in fixtures[:10]} == {0.0, 0.7}
    assert {fixture.g for _, fixture in fixtures[:10]} == {
        -0.02,
        -0.01,
        0.0,
        0.01,
        0.02,
    }
    spacing, cutoff, eta = (fixture for _, fixture in fixtures[-3:])
    assert (spacing.grid.step, spacing.grid.cutoff, spacing.eta_lead) == (
        0.000625,
        5.0,
        0.0008,
    )
    assert (cutoff.grid.step, cutoff.grid.cutoff, cutoff.eta_lead) == (
        0.00125,
        6.0,
        0.0008,
    )
    assert (eta.grid.step, eta.grid.cutoff, eta.eta_lead) == (
        0.00125,
        5.0,
        0.0004,
    )
    assert all(fixture.config.backend == "fft" for _, fixture in fixtures)
    assert all(
        fixture.config.hilbert_backend == "fft" for _, fixture in fixtures
    )


def _memory_record(rank: int) -> dict[str, object]:
    return runner.campaign_dense_memory_preflight(
        profile="scaling",
        rank_count=rank,
        available_memory_bytes=10**15,
    )


def test_scaling_memory_preflight_tracks_release_converged_grid() -> None:
    record = runner.campaign_dense_memory_preflight(
        profile="scaling",
        rank_count=16,
        available_memory_bytes=20_000_000_000,
    )

    assert isinstance(record.pop("measured_utc"), str)
    assert record == {
        "schema": "negfpy.remote_toy_mpi.aggregate_memory_preflight.v1",
        "measurement": "linux_proc_meminfo_MemAvailable",
        "profile": "scaling",
        "rank_count": 16,
        "allowed_fraction": 0.70,
        "available_memory_bytes": 20_000_000_000,
        "limit_bytes": 14_000_000_000,
        "per_rank_predicted_bytes": 716_659_488,
        "aggregate_predicted_bytes": 11_466_551_808,
        "contributors": [
            {
                "kind": "multidimensional_cross_product_upper_bound",
                "n_frequency": 30_721,
                "ndof": 9,
                "predicted_bytes": 716_659_488,
            },
            {
                "kind": "open_three_mode_largest_fixture",
                "fixture": "refine_spacing_half",
                "n_frequency": 16_001,
                "ndof": 3,
                "predicted_bytes": 41_474_592,
            },
        ],
        "passed": True,
    }


def _mock_environment(root: pathlib.Path) -> pathlib.Path:
    prefix = root / "environment"
    binaries = prefix / "bin"
    binaries.mkdir(parents=True, exist_ok=True)
    for name in ("mpiexec", "python"):
        executable = binaries / name
        if not executable.exists():
            executable.write_bytes(f"mock-{name}\n".encode("ascii"))
            executable.chmod(0o755)
    return prefix


def _rank_runtime(
    rank: int,
    *,
    python: pathlib.Path,
    run_directory: pathlib.Path,
) -> list[dict[str, object]]:
    python_record = runner._launcher_record(python)
    python_argv = runner._expected_rank_python_argv(
        profile="scaling",
        root=run_directory,
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=_ENVIRONMENT_SHA256,
        execution_id=f"scaling-rank-{rank:03d}",
        rank_count=rank,
    )
    return [
        {
            "rank": index,
            "hostname": "fixture-host",
            "logical_cpu_ids": [index],
            "logical_cpu_topology": [
                {
                    "logical_cpu_id": index,
                    "physical_package_id": index % 2,
                    "core_id": index // 2,
                }
            ],
            "physical_cores": [
                {
                    "physical_package_id": index % 2,
                    "core_id": index // 2,
                }
            ],
            "one_physical_core": True,
            "thread_limits": dict(sorted(runner.THREAD_LIMIT_ENV.items())),
            "python": python_record,
            "python_argv": python_argv,
        }
        for index in range(rank)
    ]


@pytest.fixture
def isolated_worker_artifact_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolate outer scaling policy from the separately tested deep chain.

    The remote-CLI integration test exercises a real 28-task artifact chain,
    while evidence-primitives tests cover deep corruption rejection.
    """

    def validate(payload, **kwargs):
        rank = kwargs["rank_count"]
        assert kwargs["profile"] == "scaling"
        assert kwargs["source_revision"] == _SOURCE_REVISION
        assert kwargs["source_tree_sha256"] == _SOURCE_TREE_SHA256
        assert kwargs["environment_sha256"] == _ENVIRONMENT_SHA256
        assert payload["execution_id"] == f"scaling-rank-{rank:03d}"
        assert payload["execution_mode"] == ("serial" if rank == 1 else "mpi")
        assert payload["directory_mode"] == "fresh"
        assert payload["thread_limits"] == dict(
            sorted(runner.THREAD_LIMIT_ENV.items())
        )
        memory = runner._validate_memory_record(
            payload["memory_preflight"],
            profile="scaling",
            rank_count=rank,
        )
        runtime = payload["rank_runtime"]
        assert len(runtime) == rank
        expected_argv = runner._expected_rank_python_argv(
            profile="scaling",
            root=kwargs["worker_path"].parents[1],
            source_revision=_SOURCE_REVISION,
            source_tree_sha256=_SOURCE_TREE_SHA256,
            environment_sha256=_ENVIRONMENT_SHA256,
            execution_id=payload["execution_id"],
            rank_count=rank,
        )
        assert all(item["python_argv"] == expected_argv for item in runtime)
        python_hashes = {item["python"]["sha256"] for item in runtime}
        assert len(python_hashes) == 1
        return {
            "campaign_manifest_file_sha256": "1" * 64,
            "execution_manifest_file_sha256": "2" * 64,
            "strict_report_file_sha256": "3" * 64,
            "production_toy_physics_sha256": payload[
                "production_toy_physics_sha256"
            ],
            "harmonic_preflight": {
                "real_physics_executed": True,
                "passed": True,
            },
            "memory_preflight": memory,
            "phase_evidence": [],
            "rank_runtime": runtime,
            "python_file_sha256": next(iter(python_hashes)),
        }

    monkeypatch.setattr(runner, "_validate_worker_artifact_chain", validate)


def _worker_payload(
    rank: int,
    digest: str,
    *,
    run_directory: pathlib.Path,
    python: pathlib.Path,
) -> dict[str, object]:
    return {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "scaling",
        "certification_eligible": False,
        "release_certified": False,
        "source_revision": _SOURCE_REVISION,
        "source_tree_sha256": _SOURCE_TREE_SHA256,
        "environment_sha256": _ENVIRONMENT_SHA256,
        "expected_rank_count": rank,
        "recorded_rank_count": rank,
        "execution_id": f"scaling-rank-{rank:03d}",
        "execution_mode": "serial" if rank == 1 else "mpi",
        "directory_mode": "fresh",
        "thread_limits": dict(sorted(runner.THREAD_LIMIT_ENV.items())),
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
        "production_toy_physics_sha256": digest,
        "memory_preflight": _memory_record(rank),
        "rank_runtime": _rank_runtime(
            rank,
            python=python,
            run_directory=run_directory,
        ),
    }


def _rank_evidence(
    root: pathlib.Path,
    rank: int,
    wall_seconds: float,
    *,
    digest: str = _PHYSICS_SHA256,
):
    prefix = _mock_environment(root)
    launcher = prefix / "bin" / "mpiexec"
    python = prefix / "bin" / "python"
    repetition = root / f"rank-{rank:03d}"
    execution_id = f"scaling-rank-{rank:03d}"
    worker_path = repetition / "runner-results" / f"{execution_id}.json"
    worker = runner.write_hashed_manifest(
        worker_path,
        _worker_payload(
            rank,
            digest,
            run_directory=repetition,
            python=python,
        ),
    )
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    stdout = logs / f"rank-{rank:03d}.stdout.log"
    stderr = logs / f"rank-{rank:03d}.stderr.log"
    stdout.write_text(f"rank={rank}\n", encoding="ascii")
    stderr.write_text("", encoding="ascii")
    program = (
        str(python),
        str(pathlib.Path(runner.__file__).resolve()),
        "_rank-worker",
        "--profile",
        "scaling",
        "--source-revision",
        _SOURCE_REVISION,
        "--source-tree-sha256",
        _SOURCE_TREE_SHA256,
        "--environment-sha256",
        _ENVIRONMENT_SHA256,
        "--run-directory",
        str(repetition),
        "--execution-id",
        execution_id,
        "--expected-ranks",
        str(rank),
    )
    rank_path = root / f"rank-{rank:03d}-manifest.json"
    payload = {
        "schema": "negfpy.remote_toy_mpi.scaling_rank.v1",
        "rank_count": rank,
        "source_revision": _SOURCE_REVISION,
        "source_tree_sha256": _SOURCE_TREE_SHA256,
        "environment_sha256": _ENVIRONMENT_SHA256,
        "fresh_isolated_repetition": True,
        "resume_used": False,
        "outer_wall_seconds": wall_seconds,
        "returncode": 0,
        "completed": True,
        "failure": None,
        "argv": list(
            runner.scaling.environment_mpi_argv(
                prefix,
                rank_count=rank,
                physical_core_count=76,
                socket_count=2,
                program=program,
            )
        ),
        "binding_arguments": list(
            runner.scaling.openmpi_binding_arguments(
                rank,
                physical_core_count=76,
                socket_count=2,
            )
        ),
        "thread_limits": dict(sorted(runner.THREAD_LIMIT_ENV.items())),
        "launcher_memory_preflight": _memory_record(rank),
        "launcher": runner._launcher_record(launcher),
        "python": runner._launcher_record(python),
        "stdout": {
            "relative_path": stdout.relative_to(root).as_posix(),
            "sha256": runner._file_sha256(stdout),
        },
        "stderr": {
            "relative_path": stderr.relative_to(root).as_posix(),
            "sha256": runner._file_sha256(stderr),
        },
        "worker_result_relative_path": worker_path.relative_to(root).as_posix(),
        "worker_result_file_sha256": runner._file_sha256(worker_path),
        "worker_result_payload_sha256": worker["payload_sha256"],
        "production_toy_physics_sha256": digest,
    }
    rank_document = runner.write_hashed_manifest(rank_path, payload)
    return rank_path, rank_document, worker_path


def test_hashed_manifests_are_canonical_immutable_and_tamper_evident(
    tmp_path: pathlib.Path,
) -> None:
    target = tmp_path / "strict.json"
    written = runner.write_hashed_manifest(target, {"b": 2, "a": 1})
    assert runner.read_hashed_manifest(target) == written
    assert target.read_bytes().endswith(b"\n")
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="replace"):
        runner.write_hashed_manifest(target, {"a": 1, "b": 2})

    raw = target.read_bytes().replace(b'"a":1', b'"a":9')
    target.write_bytes(raw)
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="SHA256"):
        runner.read_hashed_manifest(target)


@pytest.mark.usefixtures("isolated_worker_artifact_chain")
def test_selection_revalidates_rank_and_worker_hashes_and_applies_policy(
    tmp_path: pathlib.Path,
) -> None:
    times = (100.0, 60.0, 31.0, 30.0, 29.8)
    evidence = [
        _rank_evidence(tmp_path, rank, wall)[:2]
        for rank, wall in zip(runner.scaling.RANK_LADDER, times, strict=True)
    ]
    selected = runner.build_scaling_selection_payload(
        evidence,
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=_ENVIRONMENT_SHA256,
        physical_core_count=76,
        socket_count=2,
    )
    assert selected["production_toy_physics_sha256"] == _PHYSICS_SHA256
    assert selected["physics_equal_across_all_ranks"] is True
    assert selected["selection"]["smallest_near_fastest_rank_count"] == 4
    assert selected["toy_execution"] == {
        "rank_count": 4,
        "mode": "task_mpi",
        "minimum_speedup_satisfied": True,
    }
    assert selected["silicon_execution"] == {
        "rank_count": 4,
        "mode": "task_mpi",
    }


@pytest.mark.usefixtures("isolated_worker_artifact_chain")
def test_selection_rejects_changed_physics_or_worker_artifact(
    tmp_path: pathlib.Path,
) -> None:
    records = []
    worker_paths = []
    for rank in runner.scaling.RANK_LADDER:
        digest = "e" * 64 if rank == 8 else _PHYSICS_SHA256
        path, document, worker = _rank_evidence(
            tmp_path,
            rank,
            100.0 / rank,
            digest=digest,
        )
        records.append((path, document))
        worker_paths.append(worker)
    with pytest.raises(runner.scaling.MPIScalingError, match="differs"):
        runner.build_scaling_selection_payload(
            records,
            source_revision=_SOURCE_REVISION,
            source_tree_sha256=_SOURCE_TREE_SHA256,
            environment_sha256=_ENVIRONMENT_SHA256,
            physical_core_count=76,
            socket_count=2,
        )

    worker_paths[0].write_bytes(worker_paths[0].read_bytes() + b" ")
    with pytest.raises(
        runner.RemoteToyMPIRunnerError,
        match="missing, unsafe, or hash-mismatched",
    ):
        runner.build_scaling_selection_payload(
            records,
            source_revision=_SOURCE_REVISION,
            source_tree_sha256=_SOURCE_TREE_SHA256,
            environment_sha256=_ENVIRONMENT_SHA256,
            physical_core_count=76,
            socket_count=2,
        )


def test_scaling_requires_fresh_directory_but_release_allows_exact_resume(
    tmp_path: pathlib.Path,
) -> None:
    fresh = tmp_path / "fresh"
    assert runner._directory_mode(fresh, allow_exact_resume=False) == "fresh"
    fresh.mkdir()
    assert runner._directory_mode(fresh, allow_exact_resume=False) == "fresh"
    (fresh / "unexpected.txt").write_text("x", encoding="ascii")
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="fresh isolated"):
        runner._directory_mode(fresh, allow_exact_resume=False)
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="exact-resume"):
        runner._directory_mode(fresh, allow_exact_resume=True)
    (fresh / "campaign-manifest.json").write_text("{}\n", encoding="ascii")
    assert runner._directory_mode(fresh, allow_exact_resume=True) == (
        "exact_resume_candidate"
    )


@pytest.mark.usefixtures("isolated_worker_artifact_chain")
def test_scaling_driver_launches_five_fresh_environment_local_repetitions(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "environment"
    binaries = prefix / "bin"
    binaries.mkdir(parents=True)
    for name in ("mpiexec", "python"):
        executable = binaries / name
        executable.write_bytes(f"mock-{name}\n".encode("ascii"))
        executable.chmod(0o755)

    observed: list[tuple[int, pathlib.Path, dict[str, str]]] = []

    def fake_run(argv, **kwargs):
        command = tuple(str(value) for value in argv)
        rank = int(command[command.index("-n") + 1])
        run_directory = pathlib.Path(
            command[command.index("--run-directory") + 1]
        )
        execution_id = command[command.index("--execution-id") + 1]
        assert not run_directory.exists()
        environment = dict(kwargs["env"])
        assert all(
            environment[name] == "1" for name in runner.THREAD_LIMIT_ENV
        )
        observed.append((rank, run_directory, environment))
        worker_path = (
            run_directory / "runner-results" / f"{execution_id}.json"
        )
        runner.write_hashed_manifest(
            worker_path,
            _worker_payload(
                rank,
                _PHYSICS_SHA256,
                run_directory=run_directory,
                python=binaries / "python",
            ),
        )
        kwargs["stdout"].write(f"rank={rank}\n".encode("ascii"))
        kwargs["stderr"].write(b"")
        return subprocess.CompletedProcess(command, 0)

    clock = iter(
        (0.0, 100.0, 100.0, 160.0, 160.0, 191.0, 191.0, 221.0, 221.0, 250.8)
    )
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))

    root = tmp_path / "scale"
    selection_path = runner.run_scaling_ladder(
        environment_prefix=prefix,
        run_directory=root,
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=_ENVIRONMENT_SHA256,
        physical_core_count=76,
        socket_count=2,
    )
    assert [rank for rank, _, _ in observed] == [1, 2, 4, 8, 16]
    assert len({path for _, path, _ in observed}) == 5
    selection = runner.read_hashed_manifest(selection_path)["payload"]
    assert selection["selection"]["smallest_near_fastest_rank_count"] == 4
    assert selection["toy_execution"]["rank_count"] == 4
    assert all(
        (root / f"rank-{rank:03d}-manifest.json").is_file()
        for rank in runner.scaling.RANK_LADDER
    )
    assert observed[-1][1] == root / "rank-016"
