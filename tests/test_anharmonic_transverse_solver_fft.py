"""New solver-backend checks on small finite-grid synthetic fixed points."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from test_anharmonic_transverse_fft import _polynomial_vertex
from test_anharmonic_transverse_solver import _problem

from negfpy.anharmonic.transverse_q import TransverseCubicVertex
from negfpy.anharmonic.transverse_solver import (
    TransverseSCBAConfig,
    solve_transverse_scba,
)


def _nonlocal_problem(nq):
    base = _problem(nq=nq, d=2)
    polynomial = _polynomial_vertex((nq, 1, 1))
    vertex = TransverseCubicVertex(polynomial.mesh, 0.15 * polynomial.coefficients)
    return replace(base, vertex=vertex)


def _assert_same_evaluated_state(direct, fft):
    assert direct.converged == fft.converged
    assert direct.reason == fft.reason
    assert len(direct.history) == len(fft.history)
    assert direct.born_backend == "direct"
    assert fft.born_backend == "fft"
    for field in ("sigma_pp", "green"):
        for component in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(
                getattr(getattr(fft, field), component),
                getattr(getattr(direct, field), component),
                rtol=5e-12,
                atol=2e-14,
            )
    for name in ("left", "right", "collision_power"):
        assert getattr(fft.currents, name) == pytest.approx(
            getattr(direct.currents, name), rel=5e-12, abs=2e-14,
        )
    for actual, expected in zip(fft.history, direct.history, strict=True):
        assert actual.iteration == expected.iteration
        assert actual.sigma_residual == pytest.approx(
            expected.sigma_residual, rel=1e-7, abs=1e-7,
        )
        assert actual.current_residual == pytest.approx(
            expected.current_residual, rel=1e-6, abs=1e-6,
        )
        # Roundoff-level identity residuals need not be identical between sums.
        assert actual.identity_residual <= 1 and expected.identity_residual <= 1


@pytest.mark.parametrize("nq", [1, 4])
def test_fft_and_direct_converge_to_same_complex_nonlocal_solution(nq):
    problem = _nonlocal_problem(nq)
    if nq == 4:
        assert np.max(np.abs(problem.vertex.coefficients.imag)) > 1e-3
        assert np.max(np.abs(problem.dynamical.imag)) > 0.01
    config = TransverseSCBAConfig(max_iterations=30, mixing=0.8)
    before = problem.vertex.coefficients.copy()
    direct = solve_transverse_scba(problem, config)
    fft = solve_transverse_scba(problem, replace(config, born_backend="fft"))
    assert direct.converged and fft.converged
    assert np.max(np.abs(fft.sigma_pp.lesser)) > 1e-7
    _assert_same_evaluated_state(direct, fft)
    np.testing.assert_array_equal(problem.vertex.coefficients, before)
    assert not fft.green.retarded.flags.writeable
    assert not fft.sigma_pp.lesser.flags.writeable


@pytest.mark.parametrize("mixing", [0.7, 1e-12])
def test_fft_iteration_stop_retains_evaluated_state_and_unmixed_residual(mixing):
    problem = _nonlocal_problem(4)
    config = TransverseSCBAConfig(max_iterations=2, mixing=mixing)
    direct = solve_transverse_scba(problem, config)
    fft = solve_transverse_scba(problem, replace(config, born_backend="fft"))
    assert not fft.converged and fft.reason == "max_iterations"
    assert fft.history[-1].sigma_residual > 1
    _assert_same_evaluated_state(direct, fft)
    expected_green = np.linalg.inv(
        problem.grid.values[None, :, None, None] ** 2 * np.eye(2)
        - problem.dynamical[:, None]
        - problem.left.retarded - problem.right.retarded - fft.sigma_pp.retarded,
    )
    np.testing.assert_array_equal(fft.green.retarded, expected_green)
    if mixing == 1e-12:
        assert np.max(np.abs(fft.sigma_pp.lesser)) < 1e-12


@pytest.mark.parametrize("backend", ["auto", "FFT", None, True])
def test_unknown_backend_is_rejected(backend):
    with pytest.raises((ValueError, TypeError), match="born_backend"):
        TransverseSCBAConfig(born_backend=backend)


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_backend_specific_work_budget_rejects_before_dyson(monkeypatch, backend):
    import negfpy.anharmonic.transverse_solver as module

    problem = _nonlocal_problem(4)

    def forbidden(*args, **kwargs):
        pytest.fail("Dyson was evaluated before the Born work budget check.")

    monkeypatch.setattr(module, "_dyson", forbidden)
    with pytest.raises(ValueError, match="max_pair_evaluations"):
        solve_transverse_scba(
            problem,
            TransverseSCBAConfig(born_backend=backend, max_pair_evaluations=1),
        )


def test_fft_uses_its_own_work_count_and_zero_vertex_shortcut():
    problem = _nonlocal_problem(1)
    # Nw=9 uses 32 FFT bins, versus 61 supported direct frequency pairs.
    config = TransverseSCBAConfig(
        born_backend="fft", max_pair_evaluations=32, max_iterations=1,
    )
    result = solve_transverse_scba(problem, config)
    assert not result.converged and result.born_backend == "fft"
    with pytest.raises(ValueError, match="max_pair_evaluations"):
        solve_transverse_scba(problem, replace(config, born_backend="direct"))
    zero = replace(
        problem,
        vertex=TransverseCubicVertex(
            problem.vertex.mesh, np.zeros_like(problem.vertex.coefficients),
        ),
    )
    harmonic = solve_transverse_scba(zero, replace(config, max_pair_evaluations=1))
    assert harmonic.converged and len(harmonic.history) == 1
    np.testing.assert_array_equal(harmonic.sigma_pp.retarded, 0)


def test_fft_workspace_gate_precedes_dyson(monkeypatch):
    import negfpy.anharmonic.transverse_solver as module

    problem = _nonlocal_problem(4)

    def forbidden(*args, **kwargs):
        pytest.fail("Dyson was evaluated before the FFT workspace check.")

    monkeypatch.setattr(module, "_dyson", forbidden)
    # Exercise ordering with small fixture arrays; the production limit stays
    # fixed at 64 MiB and is not a user-configurable expansion of this solver.
    monkeypatch.setattr(module, "_MAX_BYTES", 128)
    with pytest.raises(ValueError, match="FFT workspace"):
        solve_transverse_scba(problem, TransverseSCBAConfig(born_backend="fft"))
