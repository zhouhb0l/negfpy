from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    SCBAConfig,
    SCBAConvergenceError,
    SignedFrequencyGrid,
    SolverScale,
    solve_one_shot_born,
    solve_scba,
)
from negfpy.anharmonic.toy import fpu_alpha_chain, pinned_scalar_chain
from negfpy.anharmonic._harmonic import harmonic_frequency_upper_bound
from negfpy.core.negf import device_green_function
from negfpy.core.types import Device1D, LeadBlocks


def _problem(
    *,
    cubic: float,
    temperature_left: float = 0.8,
    temperature_right: float = 0.6,
    n_layers: int = 1,
    step: float = 0.1,
    n_positive: int = 30,
) -> CubicSCBAProblem:
    device, lead, vertex = pinned_scalar_chain(
        n_layers,
        pinning=1.0,
        cubic=cubic,
    )
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, temperature_left),
        right=HarmonicReservoir(lead, temperature_right),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=step, n_positive=n_positive),
    )


def _test_config(**updates: object) -> SCBAConfig:
    values: dict[str, object] = {
        "eta_lead": 1.0e-4,
        "max_iterations": 80,
        "min_iterations": 3,
        "consecutive_passes": 2,
        "mixing": 0.3,
        "sigma_rtol": 1.0e-7,
        "green_rtol": 1.0e-7,
        "identity_rtol": 1.0e-7,
        "current_rtol": 1.0e-4,
        "current_atol": 1.0e-9,
    }
    values.update(updates)
    return SCBAConfig(**values)  # type: ignore[arg-type]


def test_harmonic_cutoff_bound_includes_explicit_contact_couplings() -> None:
    base = _problem(cubic=0.0, n_layers=1)
    strongly_coupled = CubicSCBAProblem(
        device=base.device,
        left=HarmonicReservoir(
            base.left.lead,
            base.left.temperature,
            device_to_lead=np.array([[10.0]]),
        ),
        right=HarmonicReservoir(
            base.right.lead,
            base.right.temperature,
            device_to_lead=np.array([[12.0]]),
        ),
        vertex=base.vertex,
        grid=base.grid,
    )

    assert harmonic_frequency_upper_bound(base) == pytest.approx(np.sqrt(7.0))
    assert harmonic_frequency_upper_bound(strongly_coupled) == pytest.approx(
        np.sqrt(27.0)
    )


def test_zero_vertex_scba_matches_harmonic_green_and_landauer_current() -> None:
    problem = _problem(cubic=0.0, n_layers=3, step=0.1, n_positive=26)
    config = _test_config(max_iterations=10)

    result = solve_scba(problem, config)

    assert result.converged
    assert result.method == "scba"
    assert result.diagnostics.passed
    assert len(result.history) == 4
    assert result.history[-1].consecutive_passes == 2
    assert abs(result.currents.contact_balance) < 1.0e-11
    assert result.currents.collision_power == 0.0
    assert result.currents.bond_currents is not None
    assert np.allclose(
        result.currents.bond_currents,
        result.currents.left,
        rtol=1.0e-10,
        atol=1.0e-11,
    )

    # The independent anharmonic adapter must reduce to the maintained core
    # solver at the exact same signed-grid point and with zero device eta.
    iw = problem.grid.zero_index + 15
    omega = problem.grid.values[iw]
    reference, _, _ = device_green_function(
        omega=omega,
        device=problem.device,
        lead_left=problem.left.lead,
        lead_right=problem.right.lead,
        eta=config.eta_lead,
        eta_device=0.0,
    )
    assert np.allclose(
        result.green.retarded[iw],
        reference,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_refined_harmonic_limit_is_end_to_end_release_certified() -> None:
    """Exercise every certification gate on an actual solver result."""

    problem = _problem(
        cubic=0.0,
        n_layers=1,
        step=0.025,
        n_positive=1000,
    )
    config = SCBAConfig(
        eta_lead=0.025,
        max_iterations=8,
        min_iterations=3,
        consecutive_passes=3,
    )

    result = solve_scba(problem, config)

    assert result.converged
    assert result.diagnostics.passed
    assert result.diagnostics.release_certified
    assert result.release_certified
    assert result.diagnostics.failures == ()
    assert result.diagnostics.metrics["green_spectral_moment_residual"] < 5.0e-3
    assert result.diagnostics.metrics["green_high_frequency_residual"] < 5.0e-3
    assert result.metadata["current_positive"] == "into_device"
    assert result.metadata["frequency_convention"] == "angular"
    assert result.metadata["retarded_strategy"] == "causal_hilbert"
    assert result.metadata["cutoff_to_harmonic_upper_bound"] >= 3.0
    assert (
        result.diagnostics.metrics["cutoff_to_harmonic_upper_bound"] >= 3.0
    )


def test_loose_controls_can_converge_but_never_release_certify() -> None:
    problem = _problem(
        cubic=0.1,
        step=0.15,
        n_positive=18,
    )
    config = SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=4,
        min_iterations=1,
        consecutive_passes=1,
        sigma_rtol=1.0,
        sigma_atol=1.0,
        green_rtol=1.0,
        green_atol=1.0,
        identity_rtol=1.0,
        current_rtol=1.0,
        current_atol=1.0,
        tail_rtol=1.0,
    )

    result = solve_scba(problem, config)

    assert result.converged
    assert result.diagnostics.passed
    assert not result.diagnostics.release_certified
    assert not result.release_certified
    failures = " ".join(result.diagnostics.failures)
    for control in (
        "sigma_rtol",
        "sigma_atol",
        "green_rtol",
        "green_atol",
        "identity_rtol",
        "current_rtol",
        "current_atol",
        "tail_rtol",
        "min_iterations",
        "consecutive",
    ):
        assert control in failures


def test_one_shot_born_has_fixed_semantics_and_is_never_certified() -> None:
    problem = _problem(cubic=0.005)
    result = solve_one_shot_born(problem, _test_config())

    assert result.method == "born"
    assert result.converged
    assert result.history == ()
    assert np.linalg.norm(result.sigma_pp.lesser) > 0.0
    assert np.linalg.norm(result.sigma_pp.greater) > 0.0
    assert not result.release_certified
    assert any(
        "self-consistent Born" in failure
        for failure in result.diagnostics.failures
    )


def test_weak_cubic_scba_converges_with_unmixed_fixed_point_residual() -> None:
    problem = _problem(cubic=0.005)
    config = _test_config()

    result = solve_scba(problem, config)

    assert result.converged
    assert result.diagnostics.passed
    assert result.history[-1].sigma_residual <= config.sigma_rtol
    assert result.history[-1].green_residual <= config.green_rtol
    assert result.history[-1].identity_residual <= config.identity_rtol
    assert result.history[-1].consecutive_passes == config.consecutive_passes
    assert abs(result.currents.contact_balance) <= max(
        config.current_atol,
        config.current_rtol
        * max(abs(result.currents.left), abs(result.currents.right)),
    )
    assert abs(result.currents.collision_power) <= max(
        config.current_atol,
        config.current_rtol
        * max(abs(result.currents.left), abs(result.currents.right)),
    )


@pytest.mark.parametrize(
    ("n_layers", "cubic", "step", "n_positive"),
    (
        (1, 0.004, 0.12, 25),
        (3, 0.0005, 0.15, 22),
    ),
)
def test_scba_zero_and_weak_nonzero_initial_states_reach_same_fixed_point(
    n_layers: int,
    cubic: float,
    step: float,
    n_positive: int,
) -> None:
    problem = _problem(
        cubic=cubic,
        n_layers=n_layers,
        step=step,
        n_positive=n_positive,
    )
    config = SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=120,
        min_iterations=3,
        consecutive_passes=3,
        mixing=0.3,
    )
    zero_start = solve_scba(problem, config)
    born = solve_one_shot_born(problem, config)
    continued = solve_scba(
        problem,
        config,
        initial_sigma_pp=born.sigma_pp,
    )

    assert zero_start.converged
    assert continued.converged
    assert zero_start.history[-1].consecutive_passes == 3
    assert continued.history[-1].consecutive_passes == 3
    for spectrum_name in ("sigma_pp", "green"):
        first = getattr(zero_start, spectrum_name)
        second = getattr(continued, spectrum_name)
        for component in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(
                getattr(first, component),
                getattr(second, component),
                rtol=1.0e-8,
                atol=1.0e-12,
            )
    assert np.isclose(
        zero_start.currents.left,
        continued.currents.left,
        rtol=1.0e-8,
        atol=1.0e-12,
    )


def test_nonconvergence_raises_with_complete_partial_result() -> None:
    problem = _problem(cubic=0.02, step=0.15, n_positive=20)
    config = _test_config(
        max_iterations=3,
        min_iterations=1,
        consecutive_passes=1,
        sigma_rtol=0.0,
        green_rtol=0.0,
        identity_rtol=1.0e-12,
    )

    with pytest.raises(SCBAConvergenceError) as caught:
        solve_scba(problem, config)

    partial = caught.value.partial_result
    assert partial is not None
    assert not partial.converged
    assert len(partial.history) == config.max_iterations
    assert np.all(np.isfinite(partial.green.retarded))


def test_threaded_solver_matches_serial_reference() -> None:
    problem = _problem(cubic=0.004, step=0.15, n_positive=18)
    serial = solve_one_shot_born(problem, _test_config(backend="serial"))
    threaded = solve_one_shot_born(
        problem,
        _test_config(backend="threaded", max_workers=2),
    )

    np.testing.assert_array_equal(
        threaded.sigma_pp.lesser,
        serial.sigma_pp.lesser,
    )
    np.testing.assert_array_equal(
        threaded.sigma_pp.greater,
        serial.sigma_pp.greater,
    )
    np.testing.assert_allclose(
        threaded.green.retarded,
        serial.green.retarded,
        rtol=0.0,
        atol=0.0,
    )


def test_weak_fpu_alpha_scba_uses_contact_and_collision_conservation() -> None:
    device, lead, vertex = fpu_alpha_chain(
        3,
        pinning=1.0,
        alpha=0.001,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(0.15, 25),
    )
    config = _test_config(
        sigma_rtol=1.0e-6,
        sigma_atol=1.0e-11,
        green_rtol=1.0e-6,
        green_atol=1.0e-11,
    )

    result = solve_scba(problem, config)

    assert result.converged
    assert result.diagnostics.passed
    assert result.currents.bond_currents is None
    assert result.currents.bond_spectra is None
    assert abs(result.currents.contact_balance) < 1.0e-9
    assert abs(result.currents.collision_power) < 1.0e-9


def test_memory_guard_can_be_explicitly_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    import negfpy.anharmonic.solver as solver_module

    problem = _problem(cubic=0.0, step=0.2, n_positive=5)
    monkeypatch.setattr(solver_module, "_available_memory_bytes", lambda: 1)

    with pytest.raises(MemoryError, match="preflight"):
        solve_scba(problem, _test_config(max_iterations=5))

    result = solve_scba(
        problem,
        _test_config(
            max_iterations=5,
            allow_memory_overcommit=True,
        ),
    )
    assert result.converged


def test_result_records_reproducible_numerical_and_input_provenance() -> None:
    result = solve_scba(
        _problem(cubic=0.0, step=0.2, n_positive=15),
        _test_config(max_iterations=5),
    )

    metadata = result.metadata
    assert metadata["dtype"] == "complex128"
    assert metadata["frequency_convention"] == "angular"
    assert metadata["grid_step"] == 0.2
    assert metadata["temperature_left"] == 0.8
    assert metadata["retarded_strategy"] == "causal_hilbert"
    assert metadata["source_tree_sha256"]
    assert metadata["package_version"] == "0.1.0.dev0"
    assert metadata["declared_package_version"] == "0.1.0.dev0"
    assert metadata["installed_package_version"] == "0.1.0.dev0"
    assert metadata["package_version_consistent"]
    assert isinstance(metadata["git_dirty"], bool)
    hashes = metadata["input_hashes"]
    assert set(hashes) == {
        "device",
        "left_reservoir",
        "right_reservoir",
        "cubic_vertex",
        "frequency_grid",
        "solver_scale",
        "problem",
        "configuration",
        "initial_state",
        "calculation",
    }
    assert all(len(value) == 64 for value in hashes.values())
    with pytest.raises(TypeError):
        hashes["problem"] = "changed"  # type: ignore[index]


def test_initial_state_and_method_are_part_of_calculation_provenance() -> None:
    problem = _problem(cubic=0.0, step=0.2, n_positive=15)
    config = _test_config(max_iterations=5)
    zero_start = solve_scba(problem, config)
    supplied_start = solve_scba(
        problem,
        config,
        initial_sigma_pp=zero_start.sigma_pp,
    )
    born = solve_one_shot_born(problem, config)

    zero_hashes = zero_start.metadata["input_hashes"]
    supplied_hashes = supplied_start.metadata["input_hashes"]
    born_hashes = born.metadata["input_hashes"]
    assert zero_start.metadata["initialization"] == "zero"
    assert supplied_start.metadata["initialization"] == "supplied_keldysh_pair"
    assert born.metadata["initialization"] == "born_contact_only_zero_pp"
    assert zero_hashes["initial_state"] != supplied_hashes["initial_state"]
    assert zero_hashes["calculation"] != supplied_hashes["calculation"]
    assert zero_hashes["calculation"] != born_hashes["calculation"]


def _rescaled_problem(omega0: float) -> CubicSCBAProblem:
    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=0.004,
    )
    frequency_square = omega0**2
    scaled_device = Device1D(
        onsite_blocks=[
            np.asarray(block) / frequency_square
            for block in device.onsite_blocks
        ],
        coupling_blocks=[
            np.asarray(block) / frequency_square
            for block in device.coupling_blocks
        ],
    )
    scaled_lead = LeadBlocks(
        d00=np.asarray(lead.d00) / frequency_square,
        d01=np.asarray(lead.d01) / frequency_square,
        d10=(
            None
            if lead.d10 is None
            else np.asarray(lead.d10) / frequency_square
        ),
    )
    scaled_vertex = DenseCubicVertex(
        vertex.values / omega0**2.5,
        metadata=vertex.metadata,
    )
    return CubicSCBAProblem(
        device=scaled_device,
        left=HarmonicReservoir(scaled_lead, 0.8 / omega0),
        right=HarmonicReservoir(scaled_lead, 0.6 / omega0),
        vertex=scaled_vertex,
        grid=SignedFrequencyGrid(0.15 / omega0, 18),
        scale=SolverScale(omega0=omega0),
    )


def test_converted_scba_current_is_invariant_to_reference_frequency() -> None:
    first = solve_scba(
        _rescaled_problem(1.0),
        _test_config(eta_lead=1.0e-4),
    )
    second = solve_scba(
        _rescaled_problem(2.0),
        _test_config(eta_lead=0.5e-4),
    )

    assert first.converged and second.converged
    assert np.isclose(
        first.currents.left,
        second.currents.left,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert np.isclose(
        first.currents.right,
        second.currents.right,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_interacting_equilibrium_scba_obeys_fdt_and_zero_current() -> None:
    problem = _problem(
        cubic=0.004,
        temperature_left=0.7,
        temperature_right=0.7,
        step=0.15,
        n_positive=22,
    )
    config = _test_config(
        sigma_rtol=1.0e-6,
        green_rtol=1.0e-6,
    )

    result = solve_scba(problem, config)

    assert result.converged
    assert result.diagnostics.passed
    assert abs(result.currents.left) < config.current_atol
    assert abs(result.currents.right) < config.current_atol
    assert (
        result.diagnostics.metrics["green_equilibrium_fdt_residual"]
        < config.identity_rtol
    )
    assert (
        result.diagnostics.metrics["sigma_pp_equilibrium_fdt_residual"]
        < config.identity_rtol
    )


def test_three_layer_local_cubic_scba_conserves_spatial_current() -> None:
    problem = _problem(
        cubic=0.0005,
        n_layers=3,
        step=0.15,
        n_positive=22,
    )
    result = solve_scba(
        problem,
        _test_config(
            sigma_rtol=1.0e-6,
            green_rtol=1.0e-6,
        ),
    )

    assert result.converged
    assert result.diagnostics.passed
    assert result.currents.left > 0.0
    assert result.currents.right < 0.0
    assert result.currents.bond_currents is not None
    np.testing.assert_allclose(
        result.currents.bond_currents,
        result.currents.left,
        rtol=1.0e-4,
        atol=1.0e-9,
    )
