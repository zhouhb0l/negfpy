from __future__ import annotations

from dataclasses import fields, replace
import hashlib

import numpy as np
import pytest

import negfpy.anharmonic.solver as solver
from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
)
from negfpy.anharmonic._anderson import AndersonHistory, physical_pair_admissible
from negfpy.anharmonic._checkpoint_v2 import (
    SCBAContinuationState,
    _validate_checkpoint_for_solver,
    load_scba_checkpoint_v2_self_validating,
    restore_scba_checkpoint_result_v2,
    solve_sparse_scba_checkpointed,
)
from negfpy.anharmonic._checkpoint import CheckpointCorruptionError
from negfpy.anharmonic._provenance import _update_value, configuration_hash
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain


def _problem(cubic=0.005):
    device, lead, vertex = pinned_scalar_chain(1, pinning=1.0, cubic=cubic)
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=SparseCubicVertex.from_dense(vertex),
        grid=SignedFrequencyGrid(step=0.1, n_positive=30),
    )


def _config(**updates):
    return SCBAConfig(
        **{
            "eta_lead": 1e-4,
            "max_iterations": 80,
            "min_iterations": 3,
            "consecutive_passes": 2,
            "mixing": 0.3,
            "sigma_rtol": 1e-7,
            "green_rtol": 1e-7,
            "identity_rtol": 1e-7,
            "current_rtol": 1e-4,
            "current_atol": 1e-9,
            "on_nonconvergence": "return",
            **updates,
        }
    )


def _checkpoint_run(path, problem, config):
    return solve_sparse_scba_checkpointed(
        problem,
        config,
        checkpoint_path=path,
        run_id="anderson-test",
        environment_sha256="e" * 64,
        source_tree_sha256="c" * 64,
        source_hashes={"fixture": "f" * 64},
    )


def test_linear_hash_unchanged_and_all_anderson_controls_are_bound():
    config = _config()
    legacy = hashlib.sha256()
    names = {
        "mixing_method",
        "anderson_depth",
        "anderson_regularization",
        "anderson_max_step_ratio",
    }
    for field in fields(config):
        if field.name not in names:
            _update_value(legacy, field.name)
            _update_value(legacy, getattr(config, field.name))
    assert configuration_hash(config) == legacy.hexdigest()
    active = replace(config, mixing_method="anderson")
    hashes = {configuration_hash(active), configuration_hash(config)}
    for name, value in (
        ("anderson_depth", 3),
        ("anderson_regularization", 1e-8),
        ("anderson_max_step_ratio", 3.0),
    ):
        hashes.add(configuration_hash(replace(active, **{name: value})))
    assert len(hashes) == 5


@pytest.mark.parametrize(
    "updates",
    [
        {"mixing_method": "unknown"},
        {"anderson_depth": 0},
        {"anderson_depth": 13},
        {"anderson_regularization": 0},
        {"anderson_regularization": float("nan")},
        {"anderson_max_step_ratio": 0.9},
    ],
)
def test_invalid_controls_rejected(updates):
    with pytest.raises(ValueError):
        _config(**updates)


def test_scalar_affine_map_accelerates_and_history_is_bounded():
    mixer = AndersonHistory(2, 1e-10, 20.0)
    x = (np.array([0j]), np.array([0j]))
    for _ in range(8):
        raw = tuple(0.8 * a + 1 for a in x)
        proposal, _ = mixer.propose(x, raw, 0.3)
        x = (
            proposal
            if proposal is not None
            else tuple(0.7 * a + 0.3 * b for a, b in zip(x, raw))
        )
    np.testing.assert_allclose(x, 5.0, atol=1e-8)
    assert len(mixer.states) == len(mixer.residuals) == 3
    assert abs(5.0 - 5.0 * (1 - 0.94**8)) > 2


def test_unstable_extrapolation_is_step_bounded():
    mixer = AndersonHistory(2, 1e-10, 2.0)
    def pair(value):
        return np.array([value + 0j]), np.array([value + 0j])
    mixer.propose(pair(0), pair(1), 0.2)
    proposal, reason = mixer.propose(pair(0.2), pair(1.199999), 0.2)
    assert proposal is None
    assert reason == "step_bound"


@pytest.mark.parametrize("fault", ["nan", "negative_psd", "hermiticity", "frequency"])
def test_physical_guard_rejects_bad_pairs(fault):
    grid = SignedFrequencyGrid(step=0.1, n_positive=1)
    lesser = -1j * np.ones((3, 1, 1))
    greater = lesser.copy()
    assert physical_pair_admissible((lesser, greater), grid)
    if fault == "nan":
        lesser[0, 0, 0] = np.nan
    elif fault == "negative_psd":
        lesser *= -1
        greater *= -1
    elif fault == "hermiticity":
        lesser += 1
    else:
        lesser[0, 0, 0] *= 2
    assert not physical_pair_admissible((lesser, greater), grid)


def test_real_scba_acceleration_matches_linear_fixed_point():
    problem = _problem()
    linear = solver.solve_scba(problem, _config())
    accelerated = solver.solve_scba(problem, _config(mixing_method="anderson"))
    assert linear.converged and accelerated.converged
    assert len(accelerated.history) < len(linear.history) // 2
    assert accelerated.history[-1].sigma_residual <= 1e-7
    assert accelerated.diagnostics.passed
    np.testing.assert_allclose(
        accelerated.sigma_pp.lesser, linear.sigma_pp.lesser, rtol=5e-7, atol=1e-12
    )
    assert accelerated.metadata["anderson"]["accepted"] > 0
    assert accelerated.metadata["anderson"]["map_evaluations"] == len(
        accelerated.history
    )


def test_actual_residual_rejection_restores_prior_state_and_counts_budget(
    monkeypatch, tmp_path
):
    real = solver._fixed_point_evaluation
    evaluations = []

    def worsening_trial(*args):
        value = real(*args)
        evaluations.append(value)
        if len(evaluations) == 3:
            return (*value[:-1], 10.0)
        return value

    monkeypatch.setattr(solver, "_fixed_point_evaluation", worsening_trial)
    # On the third and final evaluation the proposal worsens. The terminal
    # durable state must be iteration 2, never the rejected extrapolation.
    config = _config(mixing_method="anderson", max_iterations=3, min_iterations=1)
    path = tmp_path / "rejected.h5"
    result = _checkpoint_run(path, _problem(), config)
    assert len(evaluations) == len(result.history) == 3
    assert not result.converged and not result.history[-1].passed
    assert result.metadata["anderson"]["rejected"] == 1
    assert result.history[-1].sigma_residual == result.history[-2].sigma_residual
    np.testing.assert_array_equal(result.sigma_pp.lesser, evaluations[1][1].lesser)
    checkpoint = load_scba_checkpoint_v2_self_validating(path)
    np.testing.assert_array_equal(checkpoint.sigma_lesser, evaluations[1][1].lesser)
    assert checkpoint.metadata["anderson"]["events"][-1]["trial_sigma_residual"] == 10.0


@pytest.mark.parametrize("stop_at_rejection", [False, True])
def test_rejected_trial_preserves_damping_streak_and_linear_trajectory(
    monkeypatch, tmp_path, stop_at_rejection
):
    """Reject with a two-increase streak; the next linear state must halve.

    Use real Dyson/bubble states with a controlled merit sequence to isolate
    the controller. The trial consumes one evaluation, but must not change the
    fallback trajectory or process its restored reference a second time.
    """
    real = solver._fixed_point_evaluation
    baseline_inputs = []
    accelerated_inputs = []

    def baseline_evaluation(*args):
        baseline_inputs.append(np.array(args[3], copy=True))
        evaluation = real(*args)
        return (*evaluation[:-1], 0.1 * len(baseline_inputs))

    monkeypatch.setattr(solver, "_fixed_point_evaluation", baseline_evaluation)
    baseline_count = 3 if stop_at_rejection else 5
    baseline = solver.solve_scba(
        _problem(), _config(max_iterations=baseline_count, min_iterations=1)
    )
    if not stop_at_rejection:
        assert baseline.history[3].mixing == 0.15

    proposal_calls = 0

    def single_proposal(self, state, raw, mixing):
        nonlocal proposal_calls
        proposal_calls += 1
        if proposal_calls == 3:
            # A physical convex trial, evaluated and forced to fail below.
            proposal = tuple(
                (1 - 0.5 * mixing) * a + 0.5 * mixing * b
                for a, b in zip(state, raw)
            )
            return proposal, "proposed"
        return None, "insufficient_history"

    def accelerated_evaluation(*args):
        accelerated_inputs.append(np.array(args[3], copy=True))
        evaluation = real(*args)
        index = len(accelerated_inputs)
        residual = 10.0 if index == 4 else 0.1 * (index - (index > 4))
        return (*evaluation[:-1], residual)

    monkeypatch.setattr(AndersonHistory, "propose", single_proposal)
    monkeypatch.setattr(solver, "_fixed_point_evaluation", accelerated_evaluation)
    problem = _problem()
    config = _config(
        mixing_method="anderson", max_iterations=baseline_count + 1, min_iterations=1
    )
    path = tmp_path / "controller.h5"
    accelerated = _checkpoint_run(path, problem, config)
    assert accelerated.metadata["anderson"]["rejected"] == 1
    assert not accelerated.history[3].passed
    assert accelerated.history[3].sigma_residual == baseline.history[2].sigma_residual
    # Omitting the rejected evaluation gives the exact baseline damping and
    # scientific input sequence. Without controller-neutral rollback the
    # fourth accepted evaluation retains 0.3 rather than reducing to 0.15.
    committed_history = accelerated.history[:3] + accelerated.history[4:]
    assert [row.mixing for row in committed_history] == [
        row.mixing for row in baseline.history
    ]
    committed_inputs = accelerated_inputs[:3] + accelerated_inputs[4:]
    for actual, expected in zip(committed_inputs, baseline_inputs):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(accelerated.sigma_pp.lesser, baseline.sigma_pp.lesser)
    checkpoint = load_scba_checkpoint_v2_self_validating(path)
    assert checkpoint.metadata["anderson_damping_controller"] == "rejection-neutral-v2"
    assert checkpoint.solver_state.residual_increase_streak == (2 if stop_at_rejection else 1)
    restored = _checkpoint_run(path, problem, config)
    assert restored.history == accelerated.history
    assert restored.metadata["anderson"] == accelerated.metadata["anderson"]
    assert restored.metadata["anderson_damping_controller"] == "rejection-neutral-v2"
    np.testing.assert_array_equal(restored.sigma_pp.lesser, accelerated.sigma_pp.lesser)

    # The reader must authenticate events rather than treat a claimed rejected
    # row as permission to skip an arbitrary controller transition.
    altered = dict(checkpoint.metadata)
    algorithm = dict(altered["anderson"])
    events = [dict(event) for event in algorithm["events"]]
    for invalid_reference in (999.0, "invalid"):
        events[3]["reference_sigma_residual"] = invalid_reference
        algorithm["events"] = events
        altered["anderson"] = algorithm
        with pytest.raises(CheckpointCorruptionError, match="trial reference"):
            _validate_checkpoint_for_solver(
                replace(checkpoint, metadata=altered), problem, config
            )
    altered = dict(checkpoint.metadata, anderson_damping_controller="unknown-version")
    with pytest.raises(CheckpointCorruptionError, match="unknown Anderson"):
        _validate_checkpoint_for_solver(replace(checkpoint, metadata=altered), problem, config)

    if stop_at_rejection:
        # A historical checkpoint with no version records the original reset
        # to zero. Historical readback uses legacy semantics, not the new rule.
        legacy_metadata = dict(checkpoint.metadata)
        legacy_metadata.pop("anderson_damping_controller")
        legacy = replace(
            checkpoint, metadata=legacy_metadata,
            solver_state=SCBAContinuationState(
                config.mixing, checkpoint.history[-1].sigma_residual, 0, 0
            ),
        )
        legacy_result = restore_scba_checkpoint_result_v2(legacy, problem, config)
        assert legacy_result.metadata["anderson_damping_controller"] == "reprocess-restored-reference-v1"
        # A versionless new-state record cannot silently replay as the old
        # controller: its preserved streak disagrees with the legacy contract.
        with pytest.raises(CheckpointCorruptionError, match="continuation state"):
            _validate_checkpoint_for_solver(
                replace(checkpoint, metadata=legacy_metadata), problem, config
            )


def test_anderson_resume_forbidden_but_terminal_readback_matches(monkeypatch, tmp_path):
    config = _config(mixing_method="anderson")
    problem = _problem()
    path = tmp_path / "interrupted.h5"
    real = solver._fixed_point_evaluation
    count = 0

    def interrupted(*args):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError("simulated infrastructure interruption")
        return real(*args)

    monkeypatch.setattr(solver, "_fixed_point_evaluation", interrupted)
    with pytest.raises(RuntimeError, match="infrastructure"):
        _checkpoint_run(path, problem, config)
    assert not load_scba_checkpoint_v2_self_validating(path).resumable
    with pytest.raises(ValueError, match="cannot resume"):
        _checkpoint_run(path, problem, config)
    monkeypatch.setattr(solver, "_fixed_point_evaluation", real)
    closed_path = tmp_path / "closed.h5"
    result = _checkpoint_run(closed_path, problem, config)
    restored = _checkpoint_run(closed_path, problem, config)
    assert restored.history == result.history
    assert restored.metadata["anderson"] == result.metadata["anderson"]
    np.testing.assert_array_equal(restored.sigma_pp.lesser, result.sigma_pp.lesser)
