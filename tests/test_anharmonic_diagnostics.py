from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from negfpy.anharmonic.diagnostics import (
    antihermitian_residual,
    check_solution,
    correlation_positivity_violation,
    equilibrium_fdt_residual,
    high_frequency_extrapolation_residual,
    high_frequency_residual,
    keldysh_identity_residual,
    keldysh_tail_ratio,
    negative_frequency_residual,
    passivity_violation,
    relative_residual,
    retarded_reality_residual,
    spectral_function,
    spectral_moment_residual,
    tail_ratio,
)
from negfpy.anharmonic.contacts import bose_occupation_signed
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.types import (
    AnharmonicResult,
    CubicSCBAProblem,
    CurrentReport,
    DenseCubicVertex,
    DiagnosticReport,
    HarmonicReservoir,
    KeldyshSpectrum,
    SCBAConfig,
    SCBAIteration,
    SolverScale,
)
from negfpy.anharmonic.solver import solve_one_shot_born, solve_scba
from negfpy.anharmonic.toy import pinned_three_polarization_chain
from negfpy.core.types import Device1D, LeadBlocks


def _sum_rule_spectrum(
    grid: SignedFrequencyGrid,
    *,
    temperature: float = 0.8,
) -> KeldyshSpectrum:
    """Scalar equilibrium spectrum satisfying the discrete sum rule."""

    amplitude = np.pi / grid.step**2
    shape = (grid.n_points, 1, 1)
    retarded = np.zeros(shape, dtype=np.complex128)
    negative = grid.zero_index - 1
    positive = grid.zero_index + 1

    retarded[negative, 0, 0] = 0.5j * amplitude
    retarded[positive, 0, 0] = -0.5j * amplitude
    if grid.n_positive >= 2:
        asymptotic_value = 1.0 / grid.cutoff**2
        retarded[0, 0, 0] = asymptotic_value
        retarded[-1, 0, 0] = asymptotic_value

    advanced = np.swapaxes(retarded.conj(), -1, -2)
    discontinuity = retarded - advanced
    lesser = np.zeros(shape, dtype=np.complex128)
    greater = np.zeros(shape, dtype=np.complex128)
    nonzero = np.arange(grid.n_points) != grid.zero_index
    occupation = np.asarray(
        bose_occupation_signed(grid.values[nonzero], temperature)
    )
    lesser[nonzero] = occupation[:, None, None] * discontinuity[nonzero]
    greater[nonzero] = (
        (1.0 + occupation)[:, None, None] * discontinuity[nonzero]
    )
    return KeldyshSpectrum(retarded, lesser, greater)


def _problem(grid: SignedFrequencyGrid) -> CubicSCBAProblem:
    lead = LeadBlocks(
        d00=np.array([[3.0]]),
        d01=np.array([[-1.0]]),
    )
    return CubicSCBAProblem(
        device=Device1D(
            onsite_blocks=[np.array([[3.0]])],
            coupling_blocks=[],
        ),
        left=HarmonicReservoir(lead=lead, temperature=0.8),
        right=HarmonicReservoir(lead=lead, temperature=0.8),
        vertex=DenseCubicVertex(np.zeros((1, 1, 1))),
        grid=grid,
    )


def _result(
    problem: CubicSCBAProblem,
    *,
    method: str = "scba",
    green: KeldyshSpectrum | None = None,
    currents: CurrentReport | None = None,
    converged: bool = True,
    history: tuple[SCBAIteration, ...] | None = None,
) -> AnharmonicResult:
    zero = KeldyshSpectrum.zeros(
        n_frequency=problem.grid.n_points,
        ndof=problem.ndof,
    )
    if history is None:
        history = (
            SCBAIteration(
                iteration=3,
                sigma_residual=0.0,
                green_residual=0.0,
                identity_residual=0.0,
                current_abs_error=0.0,
                current_rel_error=0.0,
                collision_power=0.0,
                mixing=0.2,
                passed=True,
                consecutive_passes=3,
            ),
        )
    return AnharmonicResult(
        method=method,
        grid=problem.grid,
        green=green or _sum_rule_spectrum(problem.grid),
        sigma_pp=zero,
        sigma_left=zero,
        sigma_right=zero,
        currents=currents or CurrentReport(0.0, 0.0, 0.0),
        diagnostics=DiagnosticReport(passed=False),
        history=history,
        converged=converged,
    )


def test_relative_residual_has_stable_absolute_floor() -> None:
    assert relative_residual(np.zeros(2), np.zeros(2)) == 0.0
    assert relative_residual(np.array([1.0e-9]), np.zeros(1)) == pytest.approx(
        1.0e-9
    )
    assert relative_residual(np.array([10.0]), np.array([9.0])) == pytest.approx(
        0.1
    )
    with pytest.raises(ValueError, match="same shape"):
        relative_residual(np.zeros(2), np.zeros(3))


def test_exact_signed_spectrum_passes_all_primitive_diagnostics() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    spectrum = _sum_rule_spectrum(grid)

    np.testing.assert_allclose(
        spectral_function(spectrum.retarded)[:, 0, 0],
        [-np.pi, 0.0, np.pi],
        rtol=0.0,
        atol=1e-15,
    )
    assert keldysh_identity_residual(spectrum) < 1e-15
    assert antihermitian_residual(spectrum.lesser) == 0.0
    assert antihermitian_residual(spectrum.greater) == 0.0
    assert negative_frequency_residual(spectrum, grid) == 0.0
    assert retarded_reality_residual(spectrum.retarded, grid) == 0.0
    assert passivity_violation(spectrum.retarded, grid) == 0.0
    assert spectral_moment_residual(spectrum.retarded, grid) < 1e-15
    assert equilibrium_fdt_residual(spectrum, grid, temperature=0.8) < 1e-15


def test_high_frequency_metric_uses_both_outer_grid_edges() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    spectrum = _sum_rule_spectrum(grid)

    assert high_frequency_residual(spectrum.retarded, grid) == 0.0

    changed = np.array(spectrum.retarded, copy=True)
    changed[-1, 0, 0] *= 0.8
    assert high_frequency_residual(changed, grid) == pytest.approx(0.2)


def test_high_frequency_extrapolation_removes_resolved_tail_terms() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=64)
    retarded = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    nonzero = grid.values != 0.0
    omega = grid.values[nonzero]
    # omega^2 G = 1 + 3 / omega^2 + 2 / omega^4, so the quadratic
    # extrapolation is exact even though the raw finite-cutoff correction is
    # appreciable.
    retarded[nonzero, 0, 0] = (
        1.0 / omega**2
        + 3.0 / omega**4
        + 2.0 / omega**6
    )

    assert high_frequency_residual(retarded, grid) > 1.0e-2
    assert (
        high_frequency_extrapolation_residual(retarded, grid)
        < 1.0e-12
    )


def test_high_frequency_extrapolation_cannot_hide_wrong_leading_weight() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=64)
    retarded = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    nonzero = grid.values != 0.0
    omega = grid.values[nonzero]
    # The tail approaches monotonically and is fitted exactly, but its
    # spectral coefficient is 0.8 I instead of I.
    retarded[nonzero, 0, 0] = 0.8 / omega**2 - 0.1 / omega**4

    assert high_frequency_extrapolation_residual(
        retarded,
        grid,
    ) == pytest.approx(0.2, rel=1.0e-12, abs=1.0e-12)


def test_high_frequency_extrapolation_rejects_unresolved_tail() -> None:
    small_grid = SignedFrequencyGrid(step=1.0, n_positive=7)
    with pytest.raises(ValueError, match="insufficient"):
        high_frequency_extrapolation_residual(
            np.zeros((small_grid.n_points, 1, 1), dtype=np.complex128),
            small_grid,
        )

    grid = SignedFrequencyGrid(step=0.25, n_positive=64)
    retarded = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    nonzero_indices = np.flatnonzero(grid.values != 0.0)
    omega = grid.values[nonzero_indices]
    retarded[nonzero_indices, 0, 0] = 1.0 / omega**2
    for index in np.concatenate(
        (np.arange(8), np.asarray(grid.positive_indices[-8:]))
    ):
        retarded[index, 0, 0] *= (
            1.03 if index % 2 else 0.98
        )
    with pytest.raises(ValueError, match="monotonically"):
        high_frequency_extrapolation_residual(retarded, grid)


def test_diagnostics_detect_targeted_algebra_and_physics_violations() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    exact = _sum_rule_spectrum(grid)

    bad_lesser = np.array(exact.lesser, copy=True)
    bad_lesser[grid.positive_indices[0], 0, 0] = 0.4
    broken = KeldyshSpectrum(
        exact.retarded,
        bad_lesser,
        exact.greater,
    )
    assert keldysh_identity_residual(broken) > 0.1
    assert antihermitian_residual(broken.lesser) > 0.1
    assert negative_frequency_residual(broken, grid) > 0.1

    active = np.array(exact.retarded, copy=True)
    active *= -1.0
    assert passivity_violation(active, grid) > 0.9
    assert spectral_moment_residual(active, grid) > 1.0


def test_tail_metrics_exclude_physical_retarded_hilbert_tail() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    shape = (grid.n_points, 1, 1)
    retarded = np.ones(shape, dtype=np.complex128)
    lesser = np.zeros(shape, dtype=np.complex128)
    greater = np.zeros(shape, dtype=np.complex128)
    lesser[grid.zero_index, 0, 0] = -1j
    greater[grid.zero_index, 0, 0] = -2j
    spectrum = KeldyshSpectrum(retarded, lesser, greater)

    assert tail_ratio(retarded) == 1.0
    assert keldysh_tail_ratio(spectrum) == 0.0
    lesser[0, 0, 0] = -0.25j
    assert tail_ratio(lesser) == pytest.approx(0.25)


def test_check_solution_does_not_certify_fabricated_cached_fields() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    problem = _problem(grid)
    result = _result(problem)
    config = SCBAConfig(eta_lead=1.0e-8)

    report = check_solution(problem, result, config)

    assert not report.passed
    assert not report.release_certified
    assert any("stored sigma_left" in value for value in report.failures)
    assert any("Dyson equation" in value for value in report.failures)
    assert any("only 1 records" in value for value in report.failures)
    assert report.metrics["green_spectral_moment_residual"] < 1e-15
    assert report.metrics["green_high_frequency_residual"] == 0.0
    assert report.metrics["green_equilibrium_fdt_residual"] < 1e-15
    assert report.metrics["current_contact_abs_error"] == 0.0
    # The provisional report is deliberately ignored.
    assert not result.diagnostics.passed


def test_check_solution_separates_essential_and_release_only_failures() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    problem = _problem(grid)
    config = SCBAConfig(eta_lead=1.0e-8)
    result = solve_scba(problem, config)
    report = check_solution(
        problem,
        result,
        config,
    )

    assert report.passed
    assert not report.release_certified
    assert any("release:" in failure for failure in report.failures)
    assert report.metrics["recomputed_dyson_residual"] < config.identity_rtol
    assert report.metrics["sigma_pp_passivity_applicable"] == 1.0
    assert report.metrics["sigma_total_passivity_applicable"] == 1.0


def test_check_solution_fails_essential_scba_current_balance() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    problem = _problem(grid)
    currents = CurrentReport(
        left=1.0,
        right=-0.8,
        collision_power=0.0,
    )

    report = check_solution(
        problem,
        _result(problem, currents=currents),
        SCBAConfig(eta_lead=1.0e-8),
    )

    assert not report.passed
    assert not report.release_certified
    assert any("contact current balance" in value for value in report.failures)
    assert any("equilibrium contact current" in value for value in report.failures)


def test_check_solution_requires_equilibrium_fdt_for_every_spectrum() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    problem = _problem(grid)
    exact = _sum_rule_spectrum(grid)
    lesser = np.array(exact.lesser, copy=True)
    greater = np.array(exact.greater, copy=True)
    positive = grid.zero_index + 1
    negative = grid.zero_index - 1
    lesser[positive, 0, 0] -= 0.2j
    greater[positive, 0, 0] -= 0.2j
    lesser[negative, 0, 0] = greater[positive, 0, 0]
    greater[negative, 0, 0] = lesser[positive, 0, 0]
    nonthermal = KeldyshSpectrum(exact.retarded, lesser, greater)

    assert keldysh_identity_residual(nonthermal) < 1e-15
    assert equilibrium_fdt_residual(nonthermal, grid, 0.8) > 1e-2
    report = check_solution(
        problem,
        _result(problem, green=nonthermal),
        SCBAConfig(eta_lead=1.0e-8),
    )

    assert not report.passed
    assert any("green equilibrium FDT" in value for value in report.failures)


def test_check_solution_requires_final_scba_history_passes() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    problem = _problem(grid)
    bad_history = (
        SCBAIteration(
            iteration=8,
            sigma_residual=2.0e-4,
            green_residual=3.0e-4,
            identity_residual=0.0,
            current_abs_error=0.0,
            current_rel_error=0.0,
            collision_power=0.0,
            mixing=0.1,
            passed=False,
            consecutive_passes=0,
        ),
    )

    report = check_solution(
        problem,
        _result(problem, history=bad_history),
        SCBAConfig(eta_lead=1.0e-8),
    )
    empty = check_solution(
        problem,
        _result(problem, history=()),
        SCBAConfig(eta_lead=1.0e-8),
    )

    assert not report.passed
    assert any("consecutive passes" in value for value in report.failures)
    assert any("sigma residual" in value for value in report.failures)
    assert any("Green-function residual" in value for value in report.failures)
    assert not empty.passed
    assert any("history is empty" in value for value in empty.failures)


def test_check_solution_gates_harmonic_bonds_for_block_local_sigma() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    problem = _problem(grid)
    config = SCBAConfig(eta_lead=1.0e-8)
    constant = CurrentReport(
        left=0.0,
        right=0.0,
        collision_power=0.0,
        bond_currents=np.array([0.0, 0.0, 0.0]),
    )
    varying = CurrentReport(
        left=0.0,
        right=0.0,
        collision_power=0.0,
        bond_currents=np.array([0.0, 1.0e-3, -1.0e-3]),
    )

    wrong_shape = check_solution(
        problem,
        _result(problem, currents=constant),
        config,
    )
    bad = check_solution(
        problem,
        _result(problem, currents=varying),
        config,
    )

    assert not wrong_shape.passed
    assert any("bond_currents" in value for value in wrong_shape.failures)
    assert not bad.passed
    assert bad.metrics["sigma_pp_nonlocal_ratio"] == 0.0
    assert bad.metrics["sigma_pp_block_local_threshold"] == pytest.approx(
        1.0e-12
    )
    assert bad.metrics["harmonic_bond_current_gate_applicable"] == 1.0
    assert bad.metrics["bond_current_abs_spread"] == pytest.approx(2.0e-3)
    assert bad.metrics["harmonic_bond_current_abs_spread"] == pytest.approx(
        2.0e-3
    )
    assert any("bond-current spread" in value for value in bad.failures)


def test_born_can_pass_algebra_but_is_never_release_certified() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    problem = _problem(grid)
    config = SCBAConfig(eta_lead=1.0e-8)
    result = solve_one_shot_born(problem, config)
    report = check_solution(problem, result, config)

    assert report.passed
    assert not report.release_certified
    assert any("only a self-consistent Born" in value for value in report.failures)


def test_spectral_only_can_pass_but_cannot_be_certified() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    problem = _problem(grid)
    causal_config = SCBAConfig(eta_lead=1.0e-8)
    result = solve_scba(problem, causal_config)
    spectral_config = SCBAConfig(
        eta_lead=1.0e-8,
        retarded_strategy="spectral_only",
    )
    report = check_solution(problem, result, spectral_config)

    assert report.passed
    assert not report.release_certified
    assert any("causal_hilbert" in value for value in report.failures)


def test_current_absolute_tolerance_is_converted_by_j0() -> None:
    base = _problem(SignedFrequencyGrid(step=1.0, n_positive=2))
    problem = replace(
        base,
        scale=SolverScale(omega0=1.0e5, hbar=1.0e-34),
    )
    config = SCBAConfig(eta_lead=1.0e-8)
    result = solve_scba(problem, config)
    changed_currents = replace(
        result.currents,
        left=result.currents.left + 1.0e-20,
    )
    changed = replace(
        result,
        currents=changed_currents,
        release_certified=False,
    )

    report = check_solution(problem, changed, config)

    assert report.metrics["current_atol_physical"] == pytest.approx(1.0e-34)
    assert not report.passed
    assert any("stored current left" in value for value in report.failures)


def test_undeclared_physical_scale_cannot_be_release_certified() -> None:
    base = _problem(SignedFrequencyGrid(step=1.0, n_positive=2))
    problem = replace(
        base,
        scale=SolverScale(omega0=2.0, hbar=4.0),
    )
    config = SCBAConfig(eta_lead=1.0e-8)

    result = solve_scba(problem, config)

    assert not result.release_certified
    assert any(
        "physical scale source units" in value
        for value in result.diagnostics.failures
    )


def test_correlation_positivity_checks_i_correlation_psd() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    spectrum = _sum_rule_spectrum(grid)

    assert correlation_positivity_violation(spectrum.lesser) == 0.0
    assert correlation_positivity_violation(spectrum.greater) == 0.0

    indefinite = np.zeros(
        (grid.n_points, 2, 2),
        dtype=np.complex128,
    )
    indefinite[grid.zero_index, 0, 0] = 1j
    indefinite[grid.zero_index, 1, 1] = -2j
    assert correlation_positivity_violation(indefinite) == pytest.approx(0.5)


def test_interacting_nonlocal_ness_gates_correlations_not_retarded_passivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device, lead, vertex = pinned_three_polarization_chain(
        3,
        spring=(0.1, 0.1, 0.1),
        pinning=(2.05, 0.61, 0.16),
        g=0.05,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead=lead, temperature=0.8),
        right=HarmonicReservoir(lead=lead, temperature=0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.00625, n_positive=800),
    )
    config = SCBAConfig(
        eta_lead=0.025,
        backend="fft",
        hilbert_backend="fft",
        surface_gf_method="analytic_diagonal",
    )

    result = solve_one_shot_born(problem, config)
    metrics = result.diagnostics.metrics

    assert metrics["equilibrium_fdt_applicable"] == 0.0
    assert metrics["zero_vertex"] == 0.0
    assert metrics["interacting_retarded_passivity_applicable"] == 0.0
    assert metrics["sigma_pp_passivity_violation"] > 1.0e-10

    off_diagonal_lesser = np.array(result.sigma_pp.lesser, copy=True)
    diagonal = np.arange(problem.ndof)
    off_diagonal_lesser[:, diagonal, diagonal] = 0.0
    assert np.max(np.abs(off_diagonal_lesser)) > 1.0e-6

    for label in ("green", "sigma_pp", "sigma_left", "sigma_right"):
        assert metrics[f"{label}_correlation_positivity_applicable"] == 1.0
        assert (
            metrics[f"{label}_lesser_correlation_positivity_violation"]
            <= 1.0e-10
        )
        assert (
            metrics[f"{label}_greater_correlation_positivity_violation"]
            <= 1.0e-10
        )

    for label in ("green", "sigma_pp"):
        assert metrics[f"{label}_passivity_applicable"] == 0.0
    for label in ("sigma_left", "sigma_right"):
        assert metrics[f"{label}_passivity_applicable"] == 1.0
        assert metrics[f"{label}_passivity_violation"] <= 1.0e-10
    assert metrics["sigma_total_passivity_applicable"] == 0.0
    assert metrics["sigma_total_correlation_positivity_applicable"] == 1.0
    assert (
        metrics["sigma_total_lesser_correlation_positivity_violation"]
        <= 1.0e-10
    )
    assert (
        metrics["sigma_total_greater_correlation_positivity_violation"]
        <= 1.0e-10
    )

    monkeypatch.setitem(
        check_solution.__globals__,
        "passivity_violation",
        lambda retarded, grid: 0.5,
    )
    scoped_report = check_solution(problem, result, config)
    for descriptive_label in ("green", "sigma_pp", "sigma_total"):
        assert not any(
            f"{descriptive_label} passivity" in failure
            for failure in scoped_report.failures
        )
    for gated_label in ("sigma_left", "sigma_right"):
        assert any(
            f"{gated_label} passivity" in failure
            for failure in scoped_report.failures
        )


def test_interacting_equilibrium_gates_all_retarded_passivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    problem = replace(
        _problem(grid),
        vertex=DenseCubicVertex(np.full((1, 1, 1), 0.01)),
    )
    config = SCBAConfig(eta_lead=1.0e-8)
    result = solve_one_shot_born(problem, config)
    metrics = result.diagnostics.metrics

    assert metrics["equilibrium_fdt_applicable"] == 1.0
    assert metrics["zero_vertex"] == 0.0
    assert metrics["interacting_retarded_passivity_applicable"] == 1.0
    for label in (
        "green",
        "sigma_pp",
        "sigma_left",
        "sigma_right",
        "sigma_total",
    ):
        assert metrics[f"{label}_passivity_applicable"] == 1.0

    monkeypatch.setitem(
        check_solution.__globals__,
        "passivity_violation",
        lambda retarded, grid: 0.5,
    )
    gated_report = check_solution(problem, result, config)
    for label in (
        "green",
        "sigma_pp",
        "sigma_left",
        "sigma_right",
        "sigma_total",
    ):
        assert any(
            f"{label} passivity" in failure
            for failure in gated_report.failures
        )


def test_zero_vertex_ness_gates_all_retarded_passivity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    base = _problem(grid)
    problem = replace(
        base,
        left=replace(base.left, temperature=0.9),
        right=replace(base.right, temperature=0.7),
    )
    config = SCBAConfig(eta_lead=1.0e-8)
    result = solve_scba(problem, config)
    metrics = result.diagnostics.metrics

    assert metrics["equilibrium_fdt_applicable"] == 0.0
    assert metrics["zero_vertex"] == 1.0
    assert metrics["interacting_retarded_passivity_applicable"] == 1.0
    for label in (
        "green",
        "sigma_pp",
        "sigma_left",
        "sigma_right",
        "sigma_total",
    ):
        assert metrics[f"{label}_passivity_applicable"] == 1.0

    monkeypatch.setitem(
        check_solution.__globals__,
        "passivity_violation",
        lambda retarded, grid: 0.5,
    )
    gated_report = check_solution(problem, result, config)
    for label in (
        "green",
        "sigma_pp",
        "sigma_left",
        "sigma_right",
        "sigma_total",
    ):
        assert any(
            f"{label} passivity" in failure
            for failure in gated_report.failures
        )
