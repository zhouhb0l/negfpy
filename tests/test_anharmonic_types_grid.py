from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest

import negfpy
from negfpy.anharmonic.exceptions import SCBAConvergenceError
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
    SpatialCurrentReport,
    SolverScale,
    estimate_dense_memory_bytes,
)
from negfpy.core.types import Device1D, LeadBlocks


def _lead() -> LeadBlocks:
    return LeadBlocks(
        d00=np.array([[2.5]], dtype=np.complex128),
        d01=np.array([[-1.0]], dtype=np.complex128),
    )


def _device(n_layers: int = 2) -> Device1D:
    return Device1D(
        onsite_blocks=[np.array([[2.5]], dtype=np.complex128) for _ in range(n_layers)],
        coupling_blocks=[
            np.array([[-1.0]], dtype=np.complex128) for _ in range(n_layers - 1)
        ],
    )


def _version_metadata(version: str) -> dict[str, object]:
    return {
        "package_version": version,
        "declared_package_version": version,
        "installed_package_version": version,
    }


def _clean_release_provenance() -> dict[str, object]:
    return {
        **_version_metadata("0.1.0rc1"),
        "package_version_consistent": True,
        "source_tree_sha256": "b" * 64,
        "git_revision": "a" * 40,
        "git_dirty": False,
    }


def test_anharmonic_api_is_not_reexported_before_release_gate() -> None:
    assert not hasattr(negfpy, "solve_scba")
    assert not hasattr(negfpy, "DenseCubicVertex")


def test_signed_frequency_grid_is_exact_centered_and_read_only() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=3)

    np.testing.assert_array_equal(
        grid.values,
        np.array([-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]),
    )
    assert grid.omega is grid.values
    assert grid.n_points == 7
    assert grid.zero_index == 3
    assert grid.cutoff == 0.75
    assert grid.integration_weight == pytest.approx(0.25 / (2.0 * np.pi))
    np.testing.assert_array_equal(grid.positive_indices, [4, 5, 6])
    np.testing.assert_array_equal(grid.nonnegative_indices, [3, 4, 5, 6])
    np.testing.assert_array_equal(grid.negative_indices, [0, 1, 2])
    np.testing.assert_array_equal(grid.negative_partners, [6, 5, 4, 3, 2, 1, 0])
    assert grid.partner_index(1) == 5
    assert grid.partner_index(-1) == 0
    assert np.all(grid.weights == grid.integration_weight)
    assert not grid.values.flags.writeable
    assert not grid.weights.flags.writeable

    with pytest.raises(ValueError):
        grid.values[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        grid.step = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("step", "n_positive", "error"),
    [
        (0.0, 2, ValueError),
        (-1.0, 2, ValueError),
        (np.inf, 2, ValueError),
        (1.0, 0, ValueError),
        (1.0, 2.5, TypeError),
        (True, 2, TypeError),
    ],
)
def test_signed_frequency_grid_rejects_invalid_parameters(
    step: object,
    n_positive: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        SignedFrequencyGrid(step=step, n_positive=n_positive)  # type: ignore[arg-type]


def test_solver_scale_implements_approved_dimensionless_scaling() -> None:
    scale = SolverScale(omega0=2.0, hbar=4.0, metadata={"units": "test"})

    np.testing.assert_allclose(
        scale.dimensionless_dynamical_matrix(np.array([[8.0]])),
        [[2.0]],
    )
    # sqrt(4) / 2**(5/2) == 1 / sqrt(8)
    assert scale.dimensionless_cubic_vertex(1.0) == pytest.approx(1.0 / np.sqrt(8.0))
    assert scale.dimensionless_temperature(8.0, boltzmann_constant=2.0) == 2.0
    assert scale.current_scale == 16.0
    assert scale.to_physical_current(3.0) == 48.0
    assert scale.from_physical_current(48.0) == 3.0
    assert scale.metadata["source_units"] == "test"
    assert scale.metadata["source_units_declared"]
    assert scale.metadata["dynamical_matrix_to_solver_factor"] == 0.25
    assert scale.metadata["cubic_vertex_to_solver_factor"] == pytest.approx(
        1.0 / np.sqrt(8.0)
    )
    assert scale.metadata["current_solver_to_physical_factor"] == 16.0
    assert isinstance(scale.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        scale.metadata["new"] = "value"  # type: ignore[index]


def test_solver_scale_marks_undeclared_physical_source_units() -> None:
    scale = SolverScale(
        omega0=2.0,
        hbar=4.0,
        metadata={
            "source_units_declared": True,
            "current_solver_to_physical_factor": 999.0,
        },
    )

    assert scale.metadata["source_units"] == "unspecified"
    assert not scale.metadata["source_units_declared"]
    assert scale.metadata["current_solver_to_physical_factor"] == 16.0

    empty = SolverScale(
        omega0=2.0,
        hbar=4.0,
        metadata={"source_units": ""},
    )
    assert empty.metadata["source_units"] == "unspecified"
    assert not empty.metadata["source_units_declared"]


def test_dense_cubic_vertex_validates_symmetry_and_copies_input() -> None:
    raw = np.zeros((2, 2, 2))
    for permutation in set(__import__("itertools").permutations((0, 0, 1))):
        raw[permutation] = 3.0
    vertex = DenseCubicVertex(raw, metadata={"convention": "1/3!"})
    raw[:] = 99.0

    assert vertex.ndof == 2
    assert vertex.values[0, 0, 1] == 3.0
    assert vertex.values[1, 1, 1] == 0.0
    assert vertex.support_density == pytest.approx(3.0 / 8.0)
    assert not vertex.values.flags.writeable

    nonsymmetric = np.zeros((2, 2, 2))
    nonsymmetric[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="permutation symmetric"):
        DenseCubicVertex(nonsymmetric)
    with pytest.raises(ValueError, match="real cubic tensor"):
        DenseCubicVertex(np.ones((1, 1, 1), dtype=np.complex128))
    with pytest.raises(ValueError, match="shape"):
        DenseCubicVertex(np.zeros((2, 2, 3)))
    bad = np.zeros((1, 1, 1))
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        DenseCubicVertex(bad)


def test_reservoir_and_problem_resolve_matched_contacts() -> None:
    lead = _lead()
    left = HarmonicReservoir(lead=lead, temperature=1.2)
    right = HarmonicReservoir(lead=lead, temperature=0.8)
    problem = CubicSCBAProblem(
        device=_device(),
        left=left,
        right=right,
        vertex=DenseCubicVertex(np.zeros((2, 2, 2))),
        grid=SignedFrequencyGrid(step=0.1, n_positive=4),
    )

    assert problem.ndof == 2
    np.testing.assert_array_equal(left.resolved_device_indices(dim=2, side="left"), [0])
    np.testing.assert_array_equal(right.resolved_device_indices(dim=2, side="right"), [1])
    np.testing.assert_array_equal(left.resolved_device_to_lead("left"), [[-1.0]])
    np.testing.assert_array_equal(right.resolved_device_to_lead("right"), [[-1.0]])


def test_reservoir_and_problem_reject_incompatible_dimensions() -> None:
    lead = _lead()
    with pytest.raises(ValueError, match="non-negative"):
        HarmonicReservoir(lead=lead, temperature=-1.0)
    with pytest.raises(ValueError, match="column count"):
        HarmonicReservoir(
            lead=lead,
            temperature=1.0,
            device_to_lead=np.ones((1, 2)),
        )
    with pytest.raises(ValueError, match="does not match device"):
        CubicSCBAProblem(
            device=_device(),
            left=HarmonicReservoir(lead, 1.0),
            right=HarmonicReservoir(lead, 1.0),
            vertex=DenseCubicVertex(np.zeros((1, 1, 1))),
            grid=SignedFrequencyGrid(0.1, 2),
        )


def test_scba_config_defaults_and_validation() -> None:
    config = SCBAConfig(eta_lead=1e-6)

    assert config.max_iterations == 200
    assert config.min_iterations == 3
    assert config.mixing == 0.2
    assert config.mixing_floor == 0.01
    assert config.consecutive_passes == 3
    assert config.sigma_rtol == 1e-8
    assert config.green_rtol == 1e-8
    assert config.identity_rtol == 1e-8
    assert config.current_rtol == 1e-5
    assert config.current_atol == 1e-10
    assert config.tail_rtol == 1e-6
    assert config.retarded_strategy == "causal_hilbert"
    assert config.hilbert_backend == "direct"
    assert config.backend == "serial"
    assert config.on_nonconvergence == "raise"
    assert config.eta_device == 0.0

    with pytest.raises(ValueError, match="eta_device must be zero"):
        SCBAConfig(eta_lead=1e-6, eta_device=1e-12)
    with pytest.raises(ValueError, match="mixing_floor"):
        SCBAConfig(eta_lead=1e-6, mixing=0.1, mixing_floor=0.2)
    with pytest.raises(ValueError, match="unsupported"):
        SCBAConfig(eta_lead=1e-6, retarded_strategy="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="hilbert_backend"):
        SCBAConfig(eta_lead=1e-6, hilbert_backend="periodic")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backend"):
        SCBAConfig(eta_lead=1e-6, backend="process")  # type: ignore[arg-type]


def test_spatial_current_report_validates_shapes_and_is_read_only() -> None:
    spatial = SpatialCurrentReport(
        sigma_pp_nonlocal_ratio=0.25,
        eom_exchange_spectra=np.zeros((2, 3)),
        collision_power_spectra=np.ones((2, 3)),
        eom_layer_exchange=np.array([1.0, -1.0]),
        layer_collision_power=np.array([0.0, 0.0]),
        nonlocal_self_energy_eom_cut_correction=np.array([-1.0]),
        corrected_cut_currents=np.array([2.0]),
        continuity_residual=np.array([0.0, 0.0]),
    )

    assert not spatial.sigma_pp_is_block_local
    for name in (
        "eom_exchange_spectra",
        "collision_power_spectra",
        "eom_layer_exchange",
        "layer_collision_power",
        "nonlocal_self_energy_eom_cut_correction",
        "corrected_cut_currents",
        "continuity_residual",
    ):
        assert not getattr(spatial, name).flags.writeable

    with pytest.raises(ValueError, match="collision_power_spectra"):
        SpatialCurrentReport(
            sigma_pp_nonlocal_ratio=0.25,
            eom_exchange_spectra=np.zeros((2, 3)),
            collision_power_spectra=np.zeros((1, 3)),
            eom_layer_exchange=np.zeros(2),
            layer_collision_power=np.zeros(2),
            nonlocal_self_energy_eom_cut_correction=np.zeros(1),
            corrected_cut_currents=np.zeros(1),
            continuity_residual=np.zeros(2),
        )
    with pytest.raises(ValueError, match="corrected_cut_currents"):
        SpatialCurrentReport(
            sigma_pp_nonlocal_ratio=0.25,
            eom_exchange_spectra=np.zeros((2, 3)),
            collision_power_spectra=np.zeros((2, 3)),
            eom_layer_exchange=np.zeros(2),
            layer_collision_power=np.zeros(2),
            nonlocal_self_energy_eom_cut_correction=np.zeros(1),
            corrected_cut_currents=np.zeros(2),
            continuity_residual=np.zeros(2),
        )
    with pytest.raises(ValueError, match="sigma_pp_nonlocal_ratio"):
        SpatialCurrentReport(
            sigma_pp_nonlocal_ratio=1.1,
            eom_exchange_spectra=np.zeros((2, 3)),
            collision_power_spectra=np.zeros((2, 3)),
            eom_layer_exchange=np.zeros(2),
            layer_collision_power=np.zeros(2),
            nonlocal_self_energy_eom_cut_correction=np.zeros(1),
            corrected_cut_currents=np.zeros(1),
            continuity_residual=np.zeros(2),
        )


def test_result_containers_are_read_only_and_report_balances() -> None:
    grid = SignedFrequencyGrid(0.2, 2)
    spectrum = KeldyshSpectrum.zeros(n_frequency=grid.n_points, ndof=1)
    currents = CurrentReport(
        left=2.0,
        right=-1.75,
        collision_power=-0.25,
        contact_spectra=np.zeros((2, grid.n_positive)),
        collision_spectrum=np.zeros(grid.n_positive),
    )
    iteration = SCBAIteration(
        iteration=3,
        sigma_residual=1e-9,
        green_residual=2e-9,
        identity_residual=3e-9,
        current_abs_error=4e-11,
        current_rel_error=5e-6,
        collision_power=-1e-12,
        mixing=0.2,
        passed=True,
        consecutive_passes=1,
    )
    diagnostics = DiagnosticReport(
        metrics={"keldysh": 1e-9},
        failures=(),
        passed=True,
        release_certified=True,
    )
    result = AnharmonicResult(
        method="scba",
        grid=grid,
        green=spectrum,
        sigma_pp=spectrum,
        sigma_left=spectrum,
        sigma_right=spectrum,
        currents=currents,
        diagnostics=diagnostics,
        history=(iteration,),
        converged=True,
        release_certified=True,
        metadata={"backend": "serial", **_clean_release_provenance()},
    )

    assert currents.contact_balance == pytest.approx(0.25)
    assert currents.full_balance == pytest.approx(0.0)
    assert not np.asarray(currents.contact_spectra).flags.writeable
    assert result.iteration_history == (iteration,)
    assert result.release_certified
    assert result.has_release_certifying_provenance
    assert not result.green.retarded.flags.writeable
    assert isinstance(result.metadata, MappingProxyType)

    error = SCBAConvergenceError("did not converge", result)
    assert error.result is result
    assert error.partial_result is result

    with pytest.raises(ValueError, match="code provenance"):
        replace(result, metadata={})

    representative_version = "1!2.0rc1.dev2+cpu.x86"
    representative_metadata = {
        **_clean_release_provenance(),
        **_version_metadata(representative_version),
        "git_revision": "c" * 64,
    }
    representative = replace(result, metadata=representative_metadata)
    assert representative.has_release_certifying_provenance
    assert representative.release_certified

    invalid_cases = (
        {"git_dirty": True},
        {"git_dirty": "unknown"},
        {"git_revision": "unknown"},
        {"git_revision": "A" * 40},
        {"git_revision": "a" * 39},
        {"git_revision": "g" * 40},
        {"git_revision": "g" * 64},
        {"source_tree_sha256": "B" * 64},
        {"source_tree_sha256": "b" * 63},
        {"package_version_consistent": False},
        {"package_version_consistent": "true"},
        {"installed_package_version": "0.1.0"},
        {"declared_package_version": "unknown"},
        {"package_version": ""},
        _version_metadata(" "),
        _version_metadata("UNKNOWN"),
        _version_metadata("unknown "),
        _version_metadata("not a version"),
    )
    for updates in invalid_cases:
        metadata = {**_clean_release_provenance(), **updates}
        with pytest.raises(ValueError, match="code provenance"):
            replace(result, metadata=metadata)


def test_result_rejects_spectrum_grid_mismatch() -> None:
    grid = SignedFrequencyGrid(0.2, 2)
    wrong = KeldyshSpectrum.zeros(n_frequency=3, ndof=1)
    currents = CurrentReport(0.0, 0.0, 0.0)
    diagnostics = DiagnosticReport(passed=False)

    with pytest.raises(ValueError, match="frequency dimension"):
        AnharmonicResult(
            method="scba",
            grid=grid,
            green=wrong,
            sigma_pp=wrong,
            sigma_left=wrong,
            sigma_right=wrong,
            currents=currents,
            diagnostics=diagnostics,
        )


def test_dense_memory_estimate_matches_platform_contract() -> None:
    assert estimate_dense_memory_bytes(1025, 16) == 288 * 1025 * 16**2
    assert estimate_dense_memory_bytes(11, 3, n_q=4) == 288 * 11 * 4 * 3**2
    with pytest.raises(ValueError):
        estimate_dense_memory_bytes(0, 2)
