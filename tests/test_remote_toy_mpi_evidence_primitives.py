from __future__ import annotations

import importlib
import pathlib
import sys

import pytest


SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
runner = importlib.import_module("remote_toy_mpi_runner")


def _topology(mapping: dict[int, tuple[int, int]], cpu: int) -> dict[str, int]:
    package, core = mapping[cpu]
    return {
        "logical_cpu_id": cpu,
        "physical_package_id": package,
        "core_id": core,
    }


def _runtime(
    rank: int,
    logical: list[int],
    mapping: dict[int, tuple[int, int]],
) -> dict[str, object]:
    physical = sorted({mapping[cpu] for cpu in logical})
    return {
        "rank": rank,
        "hostname": "fixture-host",
        "logical_cpu_ids": logical,
        "logical_cpu_topology": [_topology(mapping, cpu) for cpu in logical],
        "physical_cores": [
            {"physical_package_id": package, "core_id": core}
            for package, core in physical
        ],
        "one_physical_core": len(physical) == 1,
        "thread_limits": dict(sorted(runner.THREAD_LIMIT_ENV.items())),
    }


def test_live_memory_formula_is_aggregate_and_uses_current_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available = 10**15
    monkeypatch.setattr(runner, "_available_memory_bytes", lambda: available)
    record = runner.campaign_dense_memory_preflight(
        profile="scaling",
        rank_count=4,
    )

    assert record["available_memory_bytes"] == available
    assert record["limit_bytes"] == int(0.70 * available)
    assert record["aggregate_predicted_bytes"] == (
        4 * record["per_rank_predicted_bytes"]
    )
    assert record["passed"] is True
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="70%"):
        runner.campaign_dense_memory_preflight(
            profile="scaling",
            rank_count=4,
            available_memory_bytes=record["aggregate_predicted_bytes"],
        )


def test_rank_affinity_accepts_smt_siblings_but_rejects_shared_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = {0: (0, 0), 1: (0, 0), 2: (1, 0)}
    monkeypatch.setattr(
        runner,
        "_logical_cpu_topology",
        lambda cpu: _topology(mapping, cpu),
    )
    accepted = runner._validate_rank_runtime(
        [_runtime(0, [0, 1], mapping), _runtime(1, [2], mapping)],
        rank_count=2,
        socket_count=2,
    )
    assert accepted[0]["one_physical_core"] is True

    with pytest.raises(runner.RemoteToyMPIRunnerError, match="share"):
        runner._validate_rank_runtime(
            [_runtime(0, [0], mapping), _runtime(1, [1], mapping)],
            rank_count=2,
            socket_count=2,
        )


def test_changed_rank_resume_is_historical_not_concurrent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mapping = {0: (0, 0), 1: (0, 0), 2: (1, 0)}
    monkeypatch.setattr(
        runner,
        "_logical_cpu_topology",
        lambda cpu: _topology(mapping, cpu),
    )
    historical = {
        "hostname": "fixture-host",
        "rank": 15,
        "cpu_affinity": [0, 1],
        "resumed": True,
    }
    current_runtime = {1: _runtime(1, [2], mapping)}
    assert runner._current_assignment_core(
        historical,
        runtime_by_rank=current_runtime,
        rank_count=2,
        require_fresh=False,
    ) is None
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="fresh campaign"):
        runner._current_assignment_core(
            historical,
            runtime_by_rank=current_runtime,
            rank_count=2,
            require_fresh=True,
        )


def test_symlink_parent_and_parent_traversal_are_rejected(
    tmp_path: pathlib.Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    artifact = real / "artifact.bin"
    artifact.write_bytes(b"evidence")
    record = {
        "relative_path": artifact.name,
        "sha256": runner._file_sha256(artifact),
    }
    link = tmp_path / "linked-root"
    link.symlink_to(real, target_is_directory=True)

    with pytest.raises(runner.RemoteToyMPIRunnerError, match="symlink"):
        runner._safe_artifact_path(link, record, name="fixture")
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="parent traversal"):
        runner._absolute_path_without_symlinks(
            real / ".." / "real", name="fixture"
        )


def test_artifact_rehash_and_manifest_digest_mismatch_are_rejected(
    tmp_path: pathlib.Path,
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"original")
    record = {
        "relative_path": artifact.name,
        "sha256": runner._file_sha256(artifact),
    }
    assert runner._safe_artifact_path(tmp_path, record, name="fixture") == artifact
    artifact.write_bytes(b"tampered")
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="hash-mismatched"):
        runner._safe_artifact_path(tmp_path, record, name="fixture")

    manifest = tmp_path / "manifest.json"
    runner.write_hashed_manifest(manifest, {"value": 1})
    raw = manifest.read_text(encoding="ascii").replace('"value":1', '"value":2')
    manifest.write_text(raw, encoding="ascii")
    with pytest.raises(runner.RemoteToyMPIRunnerError, match="SHA256 mismatch"):
        runner.read_hashed_manifest(manifest)


def test_release_rank_python_argv_is_exact() -> None:
    argv = runner._expected_rank_python_argv(
        profile="release",
        root=pathlib.Path("/validation/run"),
        source_revision="a" * 40,
        source_tree_sha256="b" * 64,
        environment_sha256="c" * 64,
        execution_id="release-attempt-001",
        rank_count=8,
    )
    assert argv[1:4] == ["release", "--run-directory", "/validation/run"]
    assert argv[-2:] == ["--expected-ranks", "8"]
    tampered = list(argv)
    tampered[-1] = "16"
    assert tampered != argv
