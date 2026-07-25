from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    solve_one_shot_born,
)
from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.toy import pinned_three_polarization_chain
from negfpy.anharmonic.types import DenseCubicVertex


def _symmetric_vertex(rng: np.random.Generator, ndof: int) -> DenseCubicVertex:
    raw = rng.normal(size=(ndof, ndof, ndof))
    values = sum(
        raw.transpose(permutation)
        for permutation in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    ) / 6.0
    return DenseCubicVertex(values)


def test_threaded_cubic_kernel_is_exactly_equal_to_serial() -> None:
    rng = np.random.default_rng(30917)
    grid = SignedFrequencyGrid(step=0.13, n_positive=4)
    vertex = _symmetric_vertex(rng, ndof=3)
    shape = (grid.n_points, vertex.ndof, vertex.ndof)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    serial = cubic_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
        backend="serial",
    )
    threaded = cubic_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
        backend="threaded",
        max_workers=3,
    )

    np.testing.assert_array_equal(threaded[0], serial[0])
    np.testing.assert_array_equal(threaded[1], serial[1])


def test_multidof_threaded_solver_uses_threads_and_matches_serial() -> None:
    """Exercise the frequency-tiled backend, not the scalar convolution path."""

    device, lead, vertex = pinned_three_polarization_chain(
        1,
        mass=(1.0, 1.1, 0.9),
        spring=(0.5, 0.7, 0.9),
        pinning=(1.0, 1.2, 0.8),
        g=0.003,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=20),
    )
    serial = solve_one_shot_born(
        problem,
        SCBAConfig(eta_lead=0.01, backend="serial"),
    )
    threaded = solve_one_shot_born(
        problem,
        SCBAConfig(
            eta_lead=0.01,
            backend="threaded",
            max_workers=2,
        ),
    )

    assert problem.ndof == 3
    assert np.any(serial.sigma_pp.lesser)
    assert serial.metadata["cubic_kernel"] == "frequency_tiled"
    assert serial.metadata["effective_backend"] == "serial"
    assert serial.metadata["worker_count"] == 1
    assert threaded.metadata["cubic_kernel"] == "frequency_tiled"
    assert threaded.metadata["effective_backend"] == "threaded"
    assert threaded.metadata["max_workers_requested"] == 2
    assert threaded.metadata["worker_count"] == 2
    assert threaded.metadata["blas_threads_during_threaded_cubic"] == 1

    for threaded_field, serial_field in (
        (threaded.sigma_pp.retarded, serial.sigma_pp.retarded),
        (threaded.sigma_pp.lesser, serial.sigma_pp.lesser),
        (threaded.sigma_pp.greater, serial.sigma_pp.greater),
        (threaded.green.retarded, serial.green.retarded),
        (threaded.green.lesser, serial.green.lesser),
        (threaded.green.greater, serial.green.greater),
    ):
        np.testing.assert_array_equal(threaded_field, serial_field)
    assert threaded.currents.left == serial.currents.left
    assert threaded.currents.right == serial.currents.right
    assert threaded.currents.collision_power == serial.currents.collision_power


def test_multidof_fft_solver_matches_direct_fields() -> None:
    device, lead, vertex = pinned_three_polarization_chain(
        1,
        mass=(1.0, 1.1, 0.9),
        spring=(0.5, 0.7, 0.9),
        pinning=(1.0, 1.2, 0.8),
        g=0.003,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=20),
    )
    direct = solve_one_shot_born(
        problem,
        SCBAConfig(
            eta_lead=0.01,
            backend="serial",
            hilbert_backend="direct",
        ),
    )
    accelerated = solve_one_shot_born(
        problem,
        SCBAConfig(
            eta_lead=0.01,
            backend="fft",
            hilbert_backend="fft",
        ),
    )

    assert accelerated.metadata["cubic_kernel"] == "zero_padded_fft"
    assert accelerated.metadata["effective_backend"] == "fft"
    assert accelerated.metadata["worker_count"] == 1
    assert accelerated.metadata["hilbert_backend"] == "fft"
    for accelerated_field, direct_field in (
        (accelerated.sigma_pp.retarded, direct.sigma_pp.retarded),
        (accelerated.sigma_pp.lesser, direct.sigma_pp.lesser),
        (accelerated.sigma_pp.greater, direct.sigma_pp.greater),
        (accelerated.green.retarded, direct.green.retarded),
        (accelerated.green.lesser, direct.green.lesser),
        (accelerated.green.greater, direct.green.greater),
    ):
        np.testing.assert_allclose(
            accelerated_field,
            direct_field,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    assert accelerated.currents.left == pytest.approx(
        direct.currents.left,
        rel=1.0e-10,
        abs=1.0e-12,
    )
    assert accelerated.currents.right == pytest.approx(
        direct.currents.right,
        rel=1.0e-10,
        abs=1.0e-12,
    )


def test_default_backend_is_exactly_the_explicit_serial_backend() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=2)
    vertex = DenseCubicVertex(np.array([[[0.7]]]))
    green = (
        np.arange(1, grid.n_points + 1, dtype=float)
        + 1j * np.arange(grid.n_points, dtype=float)
    )[:, None, None]

    default = cubic_lesser_greater(vertex, green, green, grid)
    serial = cubic_lesser_greater(
        vertex,
        green,
        green,
        grid,
        backend="serial",
        max_workers=2,
    )

    np.testing.assert_array_equal(default[0], serial[0])
    np.testing.assert_array_equal(default[1], serial[1])


def test_threaded_backend_accepts_default_and_single_worker_counts() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=1)
    vertex = DenseCubicVertex(np.ones((1, 1, 1)))
    green = np.ones((grid.n_points, 1, 1), dtype=np.complex128)

    automatic = cubic_lesser_greater(
        vertex,
        green,
        green,
        grid,
        backend="threaded",
    )
    single = cubic_lesser_greater(
        vertex,
        green,
        green,
        grid,
        backend="threaded",
        max_workers=1,
    )
    np.testing.assert_array_equal(automatic[0], single[0])
    np.testing.assert_array_equal(automatic[1], single[1])


def test_cubic_kernel_rejects_unknown_backend() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=1)
    vertex = DenseCubicVertex(np.ones((1, 1, 1)))
    green = np.ones((grid.n_points, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="backend"):
        cubic_lesser_greater(
            vertex,
            green,
            green,
            grid,
            backend="process",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("max_workers", [0, -2])
def test_cubic_kernel_rejects_nonpositive_worker_count(max_workers: int) -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=1)
    vertex = DenseCubicVertex(np.ones((1, 1, 1)))
    green = np.ones((grid.n_points, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="max_workers"):
        cubic_lesser_greater(
            vertex,
            green,
            green,
            grid,
            backend="threaded",
            max_workers=max_workers,
        )


@pytest.mark.parametrize("max_workers", [True, 1.5])
def test_cubic_kernel_rejects_noninteger_worker_count(max_workers: object) -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=1)
    vertex = DenseCubicVertex(np.ones((1, 1, 1)))
    green = np.ones((grid.n_points, 1, 1), dtype=np.complex128)

    with pytest.raises(TypeError, match="max_workers"):
        cubic_lesser_greater(
            vertex,
            green,
            green,
            grid,
            backend="threaded",
            max_workers=max_workers,  # type: ignore[arg-type]
        )
