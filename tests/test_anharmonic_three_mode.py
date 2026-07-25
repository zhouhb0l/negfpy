from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.toy import (
    bose_occupation,
    equilibrium_three_mode_decay_fixture,
    isolated_mode_keldysh_lines,
)


def _hand_derived_parent_self_energy(
    grid: SignedFrequencyGrid,
    *,
    first_bin: int,
    second_bin: int,
    temperature: float,
    coupling: float,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Literal four-channel result after cancelling the two leg orderings.

    For ``Psi[0,1,2] = psi`` and diagonal isolated-mode Green functions, the
    two ordered internal legs cancel the diagram's ``1/2``.  A pair of
    discrete spectral lines therefore has the fixed magnitude

        C = psi**2 * pi / (2 * Omega_1 * Omega_2 * Delta_omega).

    The returned arrays contain only the ``Sigma_00`` component.
    """

    omega_first = first_bin * grid.step
    omega_second = second_bin * grid.step
    occupation_first = 1.0 / np.expm1(omega_first / temperature)
    occupation_second = 1.0 / np.expm1(omega_second / temperature)
    magnitude = (
        coupling**2
        * np.pi
        / (2.0 * omega_first * omega_second * grid.step)
    )
    line_scale = -1.0j * magnitude

    lesser = np.zeros(grid.n_points, dtype=np.complex128)
    greater = np.zeros_like(lesser)
    channels = (
        (
            first_bin + second_bin,
            occupation_first * occupation_second,
            (occupation_first + 1.0) * (occupation_second + 1.0),
        ),
        (
            first_bin - second_bin,
            occupation_first * (occupation_second + 1.0),
            (occupation_first + 1.0) * occupation_second,
        ),
        (
            -first_bin + second_bin,
            (occupation_first + 1.0) * occupation_second,
            occupation_first * (occupation_second + 1.0),
        ),
        (
            -first_bin - second_bin,
            (occupation_first + 1.0) * (occupation_second + 1.0),
            occupation_first * occupation_second,
        ),
    )
    for signed_bin, lesser_weight, greater_weight in channels:
        index = grid.zero_index + signed_bin
        lesser[index] += line_scale * lesser_weight
        greater[index] += line_scale * greater_weight
    return lesser, greater


def test_isolated_mode_lines_have_discrete_delta_normalization_and_bose_weights() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=8)
    mode_bin = 3
    temperature = 0.7
    lines = isolated_mode_keldysh_lines(grid, mode_bin, temperature)

    omega = mode_bin * grid.step
    occupation = 1.0 / np.expm1(omega / temperature)
    positive = grid.zero_index + mode_bin
    negative = grid.zero_index - mode_bin
    weight = grid.integration_weight

    assert lines.occupation == pytest.approx(occupation, rel=1e-15)
    np.testing.assert_allclose(
        weight * lines.lesser[positive],
        -0.5j * occupation / omega,
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        weight * lines.lesser[negative],
        -0.5j * (occupation + 1.0) / omega,
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        weight * lines.greater[positive],
        -0.5j * (occupation + 1.0) / omega,
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        weight * lines.greater[negative],
        -0.5j * occupation / omega,
        rtol=1e-15,
        atol=1e-15,
    )
    assert not lines.lesser.flags.writeable
    assert not lines.greater.flags.writeable


def test_production_kernel_reproduces_decay_and_combination_channels() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=8)
    first_bin, second_bin = 2, 3
    temperature = 0.8
    coupling = 0.37
    fixture = equilibrium_three_mode_decay_fixture(
        grid,
        (first_bin, second_bin),
        temperature=temperature,
        coupling=coupling,
    )

    sigma_lesser, sigma_greater = cubic_lesser_greater(
        fixture.vertex,
        fixture.green_lesser,
        fixture.green_greater,
        grid,
    )
    expected_lesser, expected_greater = _hand_derived_parent_self_energy(
        grid,
        first_bin=first_bin,
        second_bin=second_bin,
        temperature=temperature,
        coupling=coupling,
    )

    np.testing.assert_allclose(
        sigma_lesser[:, 0, 0],
        expected_lesser,
        rtol=2e-15,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        sigma_greater[:, 0, 0],
        expected_greater,
        rtol=2e-15,
        atol=1e-14,
    )

    expected_signed_bins = {
        first_bin + second_bin,
        first_bin - second_bin,
        -first_bin + second_bin,
        -first_bin - second_bin,
    }
    observed_lesser = np.flatnonzero(np.abs(sigma_lesser[:, 0, 0]) > 1e-14)
    observed_greater = np.flatnonzero(np.abs(sigma_greater[:, 0, 0]) > 1e-14)
    expected_indices = np.array(
        sorted(grid.zero_index + value for value in expected_signed_bins)
    )
    np.testing.assert_array_equal(observed_lesser, expected_indices)
    np.testing.assert_array_equal(observed_greater, expected_indices)


def test_three_mode_line_weights_obey_equilibrium_kms_at_each_channel() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=10)
    temperature = 0.9
    fixture = equilibrium_three_mode_decay_fixture(
        grid,
        (2, 5),
        temperature=temperature,
        coupling=0.41,
    )
    lesser, greater = cubic_lesser_greater(
        fixture.vertex,
        fixture.green_lesser,
        fixture.green_greater,
        grid,
    )

    active = np.flatnonzero(np.abs(lesser[:, 0, 0]) > 1e-14)
    for index in active:
        np.testing.assert_allclose(
            greater[index, 0, 0],
            np.exp(grid.values[index] / temperature) * lesser[index, 0, 0],
            rtol=2e-15,
            atol=1e-14,
        )


def test_three_mode_self_energy_is_even_in_psi_and_scales_as_psi_squared() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=8)

    def evaluate(coupling: float) -> tuple[np.ndarray, np.ndarray]:
        fixture = equilibrium_three_mode_decay_fixture(
            grid,
            (2, 3),
            temperature=0.8,
            coupling=coupling,
        )
        return cubic_lesser_greater(
            fixture.vertex,
            fixture.green_lesser,
            fixture.green_greater,
            grid,
        )

    lesser, greater = evaluate(0.23)
    lesser_negative, greater_negative = evaluate(-0.23)
    lesser_scaled, greater_scaled = evaluate(0.69)

    np.testing.assert_allclose(lesser_negative, lesser, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(greater_negative, greater, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(lesser_scaled, 9.0 * lesser, rtol=2e-15, atol=1e-14)
    np.testing.assert_allclose(
        greater_scaled,
        9.0 * greater,
        rtol=2e-15,
        atol=1e-14,
    )


def test_zero_temperature_retains_spontaneous_parent_decay() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=8)
    first_bin, second_bin = 2, 3
    coupling = 0.31
    fixture = equilibrium_three_mode_decay_fixture(
        grid,
        (first_bin, second_bin),
        temperature=0.0,
        coupling=coupling,
    )
    lesser, greater = cubic_lesser_greater(
        fixture.vertex,
        fixture.green_lesser,
        fixture.green_greater,
        grid,
    )

    omega_first = first_bin * grid.step
    omega_second = second_bin * grid.step
    spontaneous = (
        -1.0j
        * coupling**2
        * np.pi
        / (2.0 * omega_first * omega_second * grid.step)
    )
    positive_sum = grid.zero_index + first_bin + second_bin
    negative_sum = grid.zero_index - first_bin - second_bin

    np.testing.assert_allclose(
        greater[positive_sum, 0, 0],
        spontaneous,
        rtol=2e-15,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        lesser[negative_sum, 0, 0],
        spontaneous,
        rtol=2e-15,
        atol=1e-14,
    )
    assert lesser[positive_sum, 0, 0] == 0.0j
    assert greater[negative_sum, 0, 0] == 0.0j

    combination_bins = (first_bin - second_bin, -first_bin + second_bin)
    for signed_bin in combination_bins:
        index = grid.zero_index + signed_bin
        assert lesser[index, 0, 0] == 0.0j
        assert greater[index, 0, 0] == 0.0j


def test_three_mode_fixture_rejects_merged_or_out_of_grid_lines() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=5)

    with pytest.raises(ValueError, match="distinct"):
        equilibrium_three_mode_decay_fixture(
            grid,
            (2, 2),
            temperature=0.5,
            coupling=0.1,
        )
    with pytest.raises(ValueError, match="cutoff"):
        equilibrium_three_mode_decay_fixture(
            grid,
            (3, 4),
            temperature=0.5,
            coupling=0.1,
        )
    with pytest.raises(ValueError, match="non-negative"):
        equilibrium_three_mode_decay_fixture(
            grid,
            (1, 2),
            temperature=-0.1,
            coupling=0.1,
        )


def test_bose_occupation_has_exact_zero_temperature_and_cold_limits() -> None:
    assert bose_occupation(0.5, 0.0) == 0.0
    assert bose_occupation(1.0, 1e-6) == 0.0
