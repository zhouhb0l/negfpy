from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys

import pytest

from negfpy.anharmonic.toy._mpi_production_gate import (
    build_production_toy_mpi_campaign,
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
    assert config.spec.spacings == (0.025, 0.0125, 0.00625)
    assert config.spec.cutoffs == (5.0, 6.0)
    assert config.spec.etas == (0.05, 0.025, 0.0125)
    assert config.spec.mixings == (0.2, 0.1)
    assert config.numerical.spacing == 0.0125
    assert config.numerical.cutoff == 5.0
    assert config.numerical.eta_lead == 0.025
    assert config.numerical.mixing == 0.2
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


def _worker_payload(rank: int, digest: str) -> dict[str, object]:
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
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
        "production_toy_physics_sha256": digest,
    }


def _rank_evidence(
    root: pathlib.Path,
    rank: int,
    wall_seconds: float,
    *,
    digest: str = _PHYSICS_SHA256,
):
    worker_path = (
        root
        / f"rank-{rank:03d}"
        / "runner-results"
        / f"scaling-rank-{rank:03d}.json"
    )
    worker = runner.write_hashed_manifest(
        worker_path,
        _worker_payload(rank, digest),
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
        match="missing or hash-mismatched",
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
            _worker_payload(rank, _PHYSICS_SHA256),
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
