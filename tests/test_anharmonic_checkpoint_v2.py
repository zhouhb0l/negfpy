from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
import pytest

import negfpy.anharmonic.solver as solver_module
from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
    solve_one_shot_born,
    solve_scba,
)
from negfpy.anharmonic._checkpoint import (
    CheckpointCorruptionError,
    CheckpointDependencyError,
    CheckpointMismatchError,
)
from negfpy.anharmonic._checkpoint_v2 import (
    SCBACheckpointV2,
    SCBAContinuationState,
    checkpoint_initial_sigma_v2,
    load_scba_checkpoint_v2,
    make_scba_checkpoint_identity_v2,
    solve_sparse_scba_checkpointed,
    write_scba_checkpoint_v2,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain


h5py = pytest.importorskip("h5py")
_SOURCE_TREE = "c" * 64
_ENVIRONMENT = "e" * 64


def _problem(*, cubic: float = 0.0, n_positive: int = 5) -> CubicSCBAProblem:
    device, lead, dense = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=cubic,
    )
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=SparseCubicVertex.from_dense(dense),
        grid=SignedFrequencyGrid(step=0.2, n_positive=n_positive),
    )


def _config(*, failure: bool = False) -> SCBAConfig:
    if failure:
        # The first Green residual is necessarily infinite, so one permitted
        # iteration is a deterministic scientific nonconvergence.
        return SCBAConfig(
            eta_lead=0.025,
            max_iterations=1,
            min_iterations=1,
            consecutive_passes=1,
            on_nonconvergence="return",
        )
    return SCBAConfig(
        eta_lead=0.025,
        max_iterations=7,
        min_iterations=3,
        consecutive_passes=2,
    )


def _identity(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    environment: str = _ENVIRONMENT,
    source_tree: str = _SOURCE_TREE,
):
    return make_scba_checkpoint_identity_v2(
        "sparse-material",
        problem,
        config,
        environment_sha256=environment,
        source_tree_sha256=source_tree,
        source_hashes={"fixture": "f" * 64},
    )


def _run(path: Path, problem: CubicSCBAProblem, config: SCBAConfig):
    return solve_sparse_scba_checkpointed(
        problem,
        config,
        checkpoint_path=path,
        run_id="sparse-material",
        environment_sha256=_ENVIRONMENT,
        source_tree_sha256=_SOURCE_TREE,
        source_hashes={"fixture": "f" * 64},
    )


def test_v2_keeps_public_solver_signatures_unchanged() -> None:
    assert list(inspect.signature(solve_one_shot_born).parameters) == [
        "problem",
        "config",
    ]
    assert list(inspect.signature(solve_scba).parameters) == [
        "problem",
        "config",
        "initial_sigma_pp",
    ]


def test_identity_collects_sparse_material_and_acoustic_source_hashes() -> None:
    base = _problem()
    vertex = SparseCubicVertex(
        ndof=base.vertex.ndof,
        indices=base.vertex.indices,
        coefficients=base.vertex.coefficients,
        metadata={
            "material": {"source_sha256": "a" * 64},
            "acoustic_contacts": {"audit_sha256": "b" * 64},
        },
    )
    problem = replace(base, vertex=vertex)
    identity = make_scba_checkpoint_identity_v2(
        "source-hashes",
        problem,
        _config(),
        environment_sha256=_ENVIRONMENT,
        source_tree_sha256=_SOURCE_TREE,
        source_hashes={
            "acoustic_contact_classification": "d" * 64,
            "material_input_manifest": "f" * 64,
        },
    )

    assert identity.source_hashes["vertex_metadata.material.source_sha256"] == (
        "a" * 64
    )
    assert identity.source_hashes[
        "vertex_metadata.acoustic_contacts.audit_sha256"
    ] == ("b" * 64)
    assert identity.source_hashes["cubic_vertex_storage"] == (
        vertex.vertex_storage_sha256
    )
    assert identity.source_hashes["acoustic_contact_classification"] == (
        "d" * 64
    )
    assert identity.source_hashes["material_input_manifest"] == "f" * 64


def test_terminal_round_trip_stores_only_restart_state_and_recomputes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    config = _config()
    path = tmp_path / "terminal.h5"
    original = _run(path, problem, config)
    identity = _identity(problem, config)
    checkpoint = load_scba_checkpoint_v2(path, expected_identity=identity)

    assert checkpoint.status == "completed"
    assert checkpoint.history == original.history
    assert len(checkpoint.timings["iteration_seconds"]) == len(original.history)
    assert checkpoint.identity.problem_hashes["frequency_grid"] == (
        checkpoint.identity.grid_sha256
    )
    assert checkpoint.identity.source_tree_sha256 == _SOURCE_TREE
    assert checkpoint.identity.environment_sha256 == _ENVIRONMENT
    with h5py.File(path, "r") as handle:
        assert set(handle) == {
            "identity_json",
            "iteration_history",
            "metadata_json",
            "sigma_pp",
            "solver_state_json",
            "timings_json",
        }
        assert set(handle["sigma_pp"]) == {"lesser", "greater"}
        assert "green" not in handle
        assert "currents" not in handle
        assert "diagnostics" not in handle

    calls = {"green": 0, "currents": 0, "finalize": 0}
    real_green = solver_module._solve_green
    real_currents = solver_module._current_report
    real_finalize = solver_module._finalize_result

    def counted_green(*args, **kwargs):
        calls["green"] += 1
        return real_green(*args, **kwargs)

    def counted_currents(*args, **kwargs):
        calls["currents"] += 1
        return real_currents(*args, **kwargs)

    def counted_finalize(*args, **kwargs):
        calls["finalize"] += 1
        return real_finalize(*args, **kwargs)

    monkeypatch.setattr(solver_module, "_solve_green", counted_green)
    monkeypatch.setattr(solver_module, "_current_report", counted_currents)
    monkeypatch.setattr(solver_module, "_finalize_result", counted_finalize)
    restored = _run(path, problem, config)

    assert calls == {"green": 1, "currents": 1, "finalize": 1}
    np.testing.assert_array_equal(restored.sigma_pp.lesser, original.sigma_pp.lesser)
    np.testing.assert_array_equal(restored.sigma_pp.greater, original.sigma_pp.greater)
    np.testing.assert_array_equal(restored.green.retarded, original.green.retarded)
    assert restored.currents.left == pytest.approx(original.currents.left)
    assert restored.diagnostics.metrics == original.diagnostics.metrics
    assert restored.diagnostics.failures == original.diagnostics.failures
    assert restored.diagnostics.passed == original.diagnostics.passed


def test_only_completed_checkpoint_can_seed_continuation(tmp_path: Path) -> None:
    problem = _problem()
    config = _config()
    path = tmp_path / "continuation.h5"
    _run(path, problem, config)
    checkpoint = load_scba_checkpoint_v2(path, expected_identity=_identity(problem, config))
    initial = checkpoint_initial_sigma_v2(checkpoint)
    np.testing.assert_array_equal(initial.lesser, checkpoint.sigma_lesser)
    np.testing.assert_array_equal(initial.greater, checkpoint.sigma_greater)
    assert np.count_nonzero(initial.retarded) == 0

    failed_config = _config(failure=True)
    failed_path = tmp_path / "failed.h5"
    _run(failed_path, problem, failed_config)
    failed = load_scba_checkpoint_v2(
        failed_path, expected_identity=_identity(problem, failed_config)
    )
    with pytest.raises(CheckpointDependencyError, match="completed SCBA checkpoint"):
        checkpoint_initial_sigma_v2(failed)


def test_infrastructure_interruption_exactly_resumes_iteration_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem(cubic=0.0)
    config = _config()
    uninterrupted = solve_scba(problem, config)
    path = tmp_path / "interrupted.h5"
    real_cubic = solver_module.cubic_lesser_greater
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated infrastructure interruption")
        return real_cubic(*args, **kwargs)

    monkeypatch.setattr(solver_module, "cubic_lesser_greater", interrupted)
    with pytest.raises(RuntimeError, match="infrastructure interruption"):
        _run(path, problem, config)

    checkpoint = load_scba_checkpoint_v2(
        path,
        expected_identity=_identity(problem, config),
    )
    assert checkpoint.status == "infrastructure_interrupted"
    assert len(checkpoint.history) == 2
    assert checkpoint.metadata["interruption_type"] == "RuntimeError"

    monkeypatch.setattr(solver_module, "cubic_lesser_greater", real_cubic)
    resumed = _run(path, problem, config)
    assert resumed.history == uninterrupted.history
    np.testing.assert_array_equal(resumed.sigma_pp.lesser, uninterrupted.sigma_pp.lesser)
    np.testing.assert_array_equal(resumed.sigma_pp.greater, uninterrupted.sigma_pp.greater)
    np.testing.assert_array_equal(resumed.green.retarded, uninterrupted.green.retarded)

    terminal = load_scba_checkpoint_v2(
        path,
        expected_identity=_identity(problem, config),
    )
    assert terminal.status == "completed"
    assert terminal.timings["attempts"] == 2
    assert terminal.timings["reconstruction_seconds"] > 0.0


def test_v2_rejects_source_environment_grid_config_and_input_mismatch(
    tmp_path: Path,
) -> None:
    problem = _problem()
    config = _config()
    path = tmp_path / "identity.h5"
    _run(path, problem, config)

    mismatches = (
        replace(_identity(problem, config), source_tree_sha256="1" * 64),
        replace(_identity(problem, config), environment_sha256="2" * 64),
        _identity(problem, replace(config, mixing=0.1)),
        _identity(_problem(n_positive=6), config),
        _identity(_problem(cubic=0.01), config),
    )
    for identity in mismatches:
        with pytest.raises(CheckpointMismatchError, match="does not match"):
            load_scba_checkpoint_v2(path, expected_identity=identity)


def test_v2_detects_dataset_corruption_and_truncation(tmp_path: Path) -> None:
    problem = _problem()
    config = _config()
    identity = _identity(problem, config)
    path = tmp_path / "corrupt.h5"
    _run(path, problem, config)

    with h5py.File(path, "r+") as handle:
        handle["sigma_pp/lesser"][0, 0, 0] += 1.0
    with pytest.raises(CheckpointCorruptionError, match="lesser"):
        load_scba_checkpoint_v2(path, expected_identity=identity)

    truncated = tmp_path / "truncated.h5"
    truncated.write_bytes(b"not a complete HDF5 file")
    with pytest.raises(CheckpointCorruptionError, match="truncated"):
        load_scba_checkpoint_v2(truncated, expected_identity=identity)


def test_atomic_replace_failure_preserves_previous_v2_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import negfpy.anharmonic._checkpoint_v2 as checkpoint_module

    problem = _problem()
    config = _config()
    shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    initial = SCBACheckpointV2(
        identity=_identity(problem, config),
        sigma_lesser=np.zeros(shape, dtype=np.complex128),
        sigma_greater=np.zeros(shape, dtype=np.complex128),
        solver_state=SCBAContinuationState(config.mixing, None, 0, 0),
    )
    path = write_scba_checkpoint_v2(tmp_path / "atomic.h5", initial)
    before = path.read_bytes()
    interrupted = replace(
        initial,
        status="infrastructure_interrupted",
        metadata={"interruption_type": "RuntimeError"},
    )

    def fail_replace(source, target) -> None:
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        write_scba_checkpoint_v2(path, interrupted)

    assert path.read_bytes() == before
    assert list(tmp_path.glob(".atomic.h5.*.tmp")) == []


def test_scientific_failure_is_terminal_and_reconstructed_without_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    config = _config(failure=True)
    path = tmp_path / "scientific-failure.h5"
    partial = _run(path, problem, config)
    assert not partial.converged

    checkpoint = load_scba_checkpoint_v2(
        path,
        expected_identity=_identity(problem, config),
    )
    assert checkpoint.status == "scientific_failure"
    assert len(checkpoint.history) == config.max_iterations

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal outcome must not re-enter fixed-point SCBA")

    monkeypatch.setattr(solver_module, "solve_scba", forbidden)
    restored = _run(path, problem, config)
    assert not restored.converged
    assert restored.history == partial.history
    np.testing.assert_array_equal(restored.green.retarded, partial.green.retarded)
