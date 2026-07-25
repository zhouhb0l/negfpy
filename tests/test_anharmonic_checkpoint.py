from __future__ import annotations

from dataclasses import replace
import importlib

import numpy as np
import pytest

from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
    solve_scba,
)
from negfpy.anharmonic._checkpoint import (
    CheckpointCorruptionError,
    CheckpointDependencyError,
    CheckpointMismatchError,
    SCBACheckpoint,
    SCBACheckpointIdentity,
    load_scba_checkpoint,
    make_scba_checkpoint_identity,
    restore_scba_checkpoint_result,
    write_scba_checkpoint,
)
from negfpy.anharmonic.toy import pinned_scalar_chain
from negfpy.anharmonic.types import SCBAIteration


pytest.importorskip("h5py")


def _problem_and_config() -> tuple[CubicSCBAProblem, SCBAConfig]:
    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=0.05,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=5),
    )
    return problem, SCBAConfig(
        eta_lead=0.025,
        max_iterations=8,
        min_iterations=3,
        consecutive_passes=3,
    )


def _identity(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    run_id: str = "direct_zero_endpoint",
) -> SCBACheckpointIdentity:
    return make_scba_checkpoint_identity(
        run_id,
        problem,
        config,
        environment_sha256="e" * 64,
        code_sha256="c" * 64,
    )


def _record(
    identity: SCBACheckpointIdentity,
    *,
    marker: float = 1.0,
) -> SCBACheckpoint:
    shape = (11, 1, 1)
    lesser = np.full(shape, 1j * marker, dtype=np.complex128)
    greater = np.full(shape, 2j * marker, dtype=np.complex128)
    history = (
        SCBAIteration(
            iteration=1,
            sigma_residual=float("inf"),
            green_residual=float("inf"),
            identity_residual=1.0e-12,
            current_abs_error=2.0e-12,
            current_rel_error=3.0e-12,
            collision_power=-4.0e-13,
            mixing=0.2,
            passed=False,
            consecutive_passes=0,
        ),
        SCBAIteration(
            iteration=2,
            sigma_residual=1.0e-9,
            green_residual=2.0e-9,
            identity_residual=3.0e-10,
            current_abs_error=4.0e-12,
            current_rel_error=5.0e-10,
            collision_power=6.0e-13,
            mixing=0.1,
            passed=True,
            consecutive_passes=1,
        ),
    )
    return SCBACheckpoint(
        identity=identity,
        sigma_lesser=lesser,
        sigma_greater=greater,
        history=history,
        timings={"elapsed_seconds": 12.5, "seed_elapsed_seconds": 0.25},
        status="converged",
        metadata={
            "backend": "serial",
            "nested": {"workers": 1},
            "labels": ["endpoint", "zero"],
        },
    )


def test_atomic_checkpoint_round_trip_and_retarded_reconstruction(
    tmp_path,
) -> None:
    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    original = _record(identity)
    path = tmp_path / "direct-zero.h5"

    returned = write_scba_checkpoint(path, original)
    loaded = load_scba_checkpoint(path, expected_identity=identity)

    assert returned == path
    assert loaded.identity == identity
    assert loaded.identity.identity_sha256 == identity.identity_sha256
    assert loaded.status == "converged"
    assert loaded.timings == original.timings
    assert loaded.metadata["nested"]["workers"] == 1
    assert loaded.history == original.history
    np.testing.assert_array_equal(loaded.sigma_lesser, original.sigma_lesser)
    np.testing.assert_array_equal(loaded.sigma_greater, original.sigma_greater)
    assert not loaded.sigma_lesser.flags.writeable
    assert not loaded.sigma_greater.flags.writeable

    restart = loaded.restart_spectrum(problem, config)
    np.testing.assert_array_equal(restart.lesser, original.sigma_lesser)
    np.testing.assert_array_equal(restart.greater, original.sigma_greater)
    assert np.all(np.isfinite(restart.retarded))
    assert not np.all(restart.retarded == 0.0)


def test_checkpoint_from_real_scba_result_enforces_solver_provenance(
    tmp_path,
) -> None:
    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=0.0,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=5),
    )
    config = SCBAConfig(
        eta_lead=0.025,
        max_iterations=8,
        min_iterations=3,
        consecutive_passes=3,
    )
    result = solve_scba(problem, config)
    identity = make_scba_checkpoint_identity(
        "real_scba",
        problem,
        config,
        environment_sha256="e" * 64,
    )

    record = SCBACheckpoint.from_result(
        identity,
        result,
        timings={"elapsed_seconds": 0.5},
        metadata={"backend": "serial"},
    )
    path = write_scba_checkpoint(tmp_path / "real-scba.h5", record)
    loaded = load_scba_checkpoint(path, expected_identity=identity)

    assert loaded.status == "converged"
    assert loaded.metadata["method"] == "scba"
    assert loaded.metadata["iteration_count"] == len(result.history)
    np.testing.assert_array_equal(
        loaded.sigma_lesser,
        result.sigma_pp.lesser,
    )

    with pytest.raises(CheckpointMismatchError, match="code hash"):
        SCBACheckpoint.from_result(
            replace(identity, code_sha256="f" * 64),
            result,
        )


def test_checkpoint_rejects_every_identity_family_mismatch(tmp_path) -> None:
    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    path = write_scba_checkpoint(tmp_path / "state.h5", _record(identity))

    mutations = {
        "run_id": "another_run",
        "configuration_sha256": "1" * 64,
        "problem_sha256": "2" * 64,
        "initial_state_sha256": "3" * 64,
        "code_sha256": "4" * 64,
        "environment_sha256": "5" * 64,
    }
    for field, value in mutations.items():
        with pytest.raises(CheckpointMismatchError, match="identity"):
            load_scba_checkpoint(
                path,
                expected_identity=replace(identity, **{field: value}),
            )


def test_restart_rejects_changed_configuration_or_problem(tmp_path) -> None:
    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    path = write_scba_checkpoint(tmp_path / "state.h5", _record(identity))
    loaded = load_scba_checkpoint(path, expected_identity=identity)

    with pytest.raises(CheckpointMismatchError, match="configuration"):
        loaded.restart_spectrum(problem, replace(config, mixing=0.1))

    changed_problem = CubicSCBAProblem(
        device=problem.device,
        left=problem.left,
        right=problem.right,
        vertex=problem.vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=6),
    )
    with pytest.raises(CheckpointMismatchError, match="problem"):
        loaded.restart_spectrum(changed_problem, config)


def test_checkpoint_detects_dataset_corruption(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    path = write_scba_checkpoint(tmp_path / "state.h5", _record(identity))

    with h5py.File(path, "r+") as handle:
        handle["sigma_pp/lesser"][0, 0, 0] += 1.0

    with pytest.raises(CheckpointCorruptionError, match="lesser"):
        load_scba_checkpoint(path, expected_identity=identity)


def test_checkpoint_detects_truncation(tmp_path) -> None:
    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    path = tmp_path / "truncated.h5"
    path.write_bytes(b"not an HDF5 checkpoint")

    with pytest.raises(CheckpointCorruptionError, match="truncated"):
        load_scba_checkpoint(path, expected_identity=identity)


def test_atomic_replace_failure_preserves_previous_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    import negfpy.anharmonic._checkpoint as checkpoint_module

    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    path = write_scba_checkpoint(tmp_path / "state.h5", _record(identity))
    before = path.read_bytes()

    def fail_replace(source, target) -> None:
        raise OSError("simulated publication failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        write_scba_checkpoint(path, _record(identity, marker=2.0))

    assert path.read_bytes() == before
    assert list(tmp_path.glob(".state.h5.*.tmp")) == []


def test_existing_checkpoint_cannot_be_relabelled(tmp_path) -> None:
    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    path = write_scba_checkpoint(tmp_path / "state.h5", _record(identity))
    other = replace(identity, run_id="different")

    with pytest.raises(CheckpointMismatchError, match="identity"):
        write_scba_checkpoint(path, _record(other))


def test_optional_h5py_error_is_actionable(monkeypatch, tmp_path) -> None:
    import negfpy.anharmonic._checkpoint as checkpoint_module

    problem, config = _problem_and_config()
    identity = _identity(problem, config)
    record = _record(identity)
    real_import = importlib.import_module

    def missing_h5py(name: str, package: str | None = None):
        if name == "h5py":
            raise ImportError("simulated missing dependency")
        return real_import(name, package)

    monkeypatch.setattr(checkpoint_module.importlib, "import_module", missing_h5py)
    with pytest.raises(CheckpointDependencyError, match=r"negfpy\[test\]"):
        write_scba_checkpoint(tmp_path / "state.h5", record)


def test_invalid_status_metadata_and_hashes_are_rejected() -> None:
    problem, config = _problem_and_config()
    identity = _identity(problem, config)

    with pytest.raises(ValueError, match="status"):
        replace(_record(identity), status="running")
    with pytest.raises(ValueError, match="NaN"):
        replace(_record(identity), metadata={"bad": float("nan")})
    with pytest.raises(ValueError, match="SHA256"):
        replace(identity, code_sha256="not-a-hash")


def test_checkpoint_restore_recomputes_fields_without_moving_fixed_point(
    tmp_path,
) -> None:
    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=0.0,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=5),
    )
    config = SCBAConfig(
        eta_lead=0.025,
        max_iterations=8,
        min_iterations=3,
        consecutive_passes=3,
    )
    original = solve_scba(problem, config)
    identity = make_scba_checkpoint_identity(
        "restore_scba",
        problem,
        config,
        environment_sha256="e" * 64,
    )
    path = write_scba_checkpoint(
        tmp_path / "restore-scba.h5",
        SCBACheckpoint.from_result(identity, original),
    )
    loaded = load_scba_checkpoint(path, expected_identity=identity)

    restored = restore_scba_checkpoint_result(loaded, problem, config)

    np.testing.assert_array_equal(
        restored.sigma_pp.lesser,
        original.sigma_pp.lesser,
    )
    np.testing.assert_array_equal(
        restored.sigma_pp.greater,
        original.sigma_pp.greater,
    )
    np.testing.assert_allclose(
        restored.green.retarded,
        original.green.retarded,
        rtol=0.0,
        atol=0.0,
    )
    assert restored.history == original.history
    assert restored.currents.left == pytest.approx(original.currents.left)
    assert restored.currents.right == pytest.approx(original.currents.right)
    assert original.currents.spatial is not None
    assert restored.currents.spatial is not None
    for name in (
        "sigma_pp_nonlocal_ratio",
        "eom_exchange_spectra",
        "collision_power_spectra",
        "eom_layer_exchange",
        "layer_collision_power",
        "nonlocal_self_energy_eom_cut_correction",
        "corrected_cut_currents",
        "continuity_residual",
    ):
        np.testing.assert_allclose(
            getattr(restored.currents.spatial, name),
            getattr(original.currents.spatial, name),
            rtol=0.0,
            atol=0.0,
        )
    assert restored.diagnostics.passed
    assert restored.release_certified == original.release_certified
