from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.retarded import retarded_from_keldysh
from negfpy.anharmonic.toy import (
    discrete_mode_space_fgr,
    equilibrium_three_mode_decay_fixture,
    harmonic_mode_basis,
)
from negfpy.anharmonic.types import DenseCubicVertex


def test_mode_space_fgr_matches_production_bubble_and_direct_hilbert() -> None:
    """Compare two implementations that share only the frozen conventions."""

    grid = SignedFrequencyGrid(step=0.2, n_positive=12)
    fixture = equilibrium_three_mode_decay_fixture(
        grid,
        (2, 3),
        temperature=0.7,
        coupling=0.017,
    )
    production_lesser, production_greater = cubic_lesser_greater(
        fixture.vertex,
        fixture.green_lesser,
        fixture.green_greater,
        grid,
    )
    production_retarded = retarded_from_keldysh(
        production_lesser,
        production_greater,
        grid,
        strategy="causal_hilbert",
    )

    for target_mode in range(3):
        oracle = discrete_mode_space_fgr(
            grid,
            fixture.mode_frequencies,
            fixture.vertex,
            temperature=fixture.temperature,
            target_mode=target_mode,
        )
        np.testing.assert_allclose(
            oracle.lesser,
            production_lesser[:, target_mode, target_mode],
            rtol=3e-15,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            oracle.greater,
            production_greater[:, target_mode, target_mode],
            rtol=3e-15,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            oracle.retarded,
            production_retarded[:, target_mode, target_mode],
            rtol=4e-14,
            atol=3e-15,
        )


def test_zero_temperature_fgr_linewidth_contains_spontaneous_decay() -> None:
    grid = SignedFrequencyGrid(step=0.1, n_positive=12)
    fixture = equilibrium_three_mode_decay_fixture(
        grid,
        (2, 3),
        temperature=0.0,
        coupling=0.004,
    )
    oracle = discrete_mode_space_fgr(
        grid,
        fixture.mode_frequencies,
        fixture.vertex,
        temperature=0.0,
        target_mode=0,
    )

    omega_parent, omega_first, omega_second = fixture.mode_frequencies
    expected_linewidth = (
        np.pi
        * fixture.coupling**2
        / (
            4.0
            * omega_parent
            * omega_first
            * omega_second
            * grid.step
        )
    )
    assert oracle.linewidth == pytest.approx(expected_linewidth, rel=2e-15)
    assert oracle.spectral_half_width == pytest.approx(
        0.5 * expected_linewidth,
        rel=2e-15,
    )
    assert oracle.passive


def test_parent_shift_and_linewidth_match_closed_form_three_mode_fgr() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=12)
    temperature = 0.8
    fixture = equilibrium_three_mode_decay_fixture(
        grid,
        (2, 3),
        temperature=temperature,
        coupling=0.003,
    )
    oracle = discrete_mode_space_fgr(
        grid,
        fixture.mode_frequencies,
        fixture.vertex,
        temperature=temperature,
        target_mode=0,
    )

    omega_parent, omega_first, omega_second = fixture.mode_frequencies
    occupation_first = 1.0 / np.expm1(omega_first / temperature)
    occupation_second = 1.0 / np.expm1(omega_second / temperature)
    prefactor = fixture.coupling**2 / (4.0 * omega_first * omega_second)

    # The resonant PV term is omitted at its grid point. The remaining real
    # part consists of the counter-rotating sum channel and the two
    # difference channels. This expression is derived directly from their
    # energy denominators rather than from either implementation.
    expected_real_self_energy = prefactor * (
        -(1.0 + occupation_first + occupation_second)
        / (2.0 * omega_parent)
        + (occupation_second - occupation_first)
        * (
            1.0 / (2.0 * omega_second)
            - 1.0 / (2.0 * omega_first)
        )
    )
    expected_shift = expected_real_self_energy / (2.0 * omega_parent)
    expected_linewidth = (
        np.pi
        * fixture.coupling**2
        * (1.0 + occupation_first + occupation_second)
        / (
            4.0
            * omega_parent
            * omega_first
            * omega_second
            * grid.step
        )
    )
    assert oracle.frequency_shift == pytest.approx(expected_shift, rel=3e-15)
    assert oracle.linewidth == pytest.approx(expected_linewidth, rel=3e-15)


def test_fgr_shift_linewidth_and_weak_coupling_gate_scale_as_coupling_squared() -> None:
    grid = SignedFrequencyGrid(step=0.1, n_positive=12)

    def evaluate(coupling: float):
        fixture = equilibrium_three_mode_decay_fixture(
            grid,
            (2, 3),
            temperature=0.4,
            coupling=coupling,
        )
        return discrete_mode_space_fgr(
            grid,
            fixture.mode_frequencies,
            fixture.vertex,
            temperature=fixture.temperature,
            target_mode=0,
        )

    weak = evaluate(0.003)
    doubled = evaluate(0.006)
    strong = evaluate(0.02)

    assert weak.frequency_shift != 0.0
    assert weak.linewidth > 0.0
    assert doubled.frequency_shift == pytest.approx(
        4.0 * weak.frequency_shift,
        rel=3e-15,
    )
    assert doubled.linewidth == pytest.approx(
        4.0 * weak.linewidth,
        rel=3e-15,
    )
    assert weak.relative_shift < 0.1
    assert weak.relative_linewidth < 0.1
    assert weak.weak_coupling_passed
    assert strong.relative_linewidth > 0.1
    assert not strong.weak_coupling_passed


def test_mode_basis_projection_reproduces_the_cubic_energy() -> None:
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )
    frequencies = np.array([0.4, 0.7])
    dynamical = rotation @ np.diag(frequencies**2) @ rotation.T
    mode_values = np.zeros((2, 2, 2))
    mode_values[0, 1, 1] = 0.13
    mode_values[1, 0, 1] = 0.13
    mode_values[1, 1, 0] = 0.13
    real_values = np.einsum(
        "ijk,ai,bj,ck->abc",
        mode_values,
        rotation,
        rotation,
        rotation,
        optimize=True,
    )
    real_vertex = DenseCubicVertex(real_values)

    basis = harmonic_mode_basis(dynamical, real_vertex)
    np.testing.assert_allclose(basis.frequencies, frequencies, atol=2e-15)

    # Eigenvector signs are arbitrary, so compare the cubic polynomial rather
    # than individual signed tensor entries.
    real_displacement = np.array([0.21, -0.34])
    mode_displacement = basis.eigenvectors.T @ real_displacement
    real_energy = np.einsum(
        "abc,a,b,c->",
        real_vertex.values,
        real_displacement,
        real_displacement,
        real_displacement,
    )
    mode_energy = np.einsum(
        "ijk,i,j,k->",
        basis.vertex.values,
        mode_displacement,
        mode_displacement,
        mode_displacement,
    )
    assert mode_energy == pytest.approx(real_energy, rel=2e-15, abs=2e-15)


def test_discrete_mode_space_fgr_rejects_off_grid_or_acoustic_modes() -> None:
    grid = SignedFrequencyGrid(step=0.1, n_positive=10)
    vertex = DenseCubicVertex(np.zeros((2, 2, 2)))

    with pytest.raises(ValueError, match="exactly"):
        discrete_mode_space_fgr(
            grid,
            [0.25, 0.5],
            vertex,
            temperature=0.3,
            target_mode=0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        discrete_mode_space_fgr(
            grid,
            [0.0, 0.5],
            vertex,
            temperature=0.3,
            target_mode=0,
        )
    with pytest.raises(ValueError, match="pinned"):
        harmonic_mode_basis(
            np.diag([0.0, 0.25]),
            vertex,
        )


def test_fgr_spectrum_constructor_rejects_inconsistent_target_and_channels() -> None:
    grid = SignedFrequencyGrid(step=0.1, n_positive=10)
    fixture = equilibrium_three_mode_decay_fixture(
        grid,
        (2, 3),
        temperature=0.4,
        coupling=0.003,
    )
    oracle = discrete_mode_space_fgr(
        grid,
        fixture.mode_frequencies,
        fixture.vertex,
        temperature=fixture.temperature,
        target_mode=0,
    )

    with pytest.raises(ValueError, match="target_mode"):
        replace(oracle, target_mode=3)
    with pytest.raises(ValueError, match="must match"):
        replace(oracle, target_frequency_bin=oracle.target_frequency_bin - 1)
    with pytest.raises(TypeError, match="tuple"):
        replace(oracle, channels=list(oracle.channels))
