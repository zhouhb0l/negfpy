"""Bounded storage, resource and pre-allocation guards for the packed executor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from negfpy.anharmonic import transverse_packed
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_packed import (
    PackedTransverseCubicVertex,
    contract_packed_fft_bins,
    cubic_born_self_energy_q_packed_fft,
    embed_packed_self_energy,
    plan_packed_born_fft,
    plan_packed_born_fft_dimensions,
    project_transverse_green,
)
from negfpy.anharmonic.transverse_q import TransverseCubicVertex


def _vertex(a=2):
    return PackedTransverseCubicVertex(
        UniformBZMesh((1, 1, 1)), np.ones((1, 1, a, a, a)), np.arange(a), max(a, 1),
    )


def test_owned_coefficients_addresses_and_order():
    values = np.ones((1, 1, 2, 2, 2))
    addresses = np.array([4, 1])
    vertex = PackedTransverseCubicVertex(UniformBZMesh((1, 1, 1)), values, addresses, 5)
    values[...] = 0
    addresses[...] = 0
    assert np.all(vertex.coefficients == 1)
    np.testing.assert_array_equal(vertex.active_dofs, [4, 1])
    assert not vertex.coefficients.flags.writeable
    assert not vertex.active_dofs.flags.writeable


@pytest.mark.parametrize("addresses", [[0, 0], [-1, 1], [0, 5], [0.0, 1.0], [False, True]])
def test_invalid_active_addresses_rejected(addresses):
    with pytest.raises((TypeError, ValueError), match="active_dofs"):
        PackedTransverseCubicVertex(
            UniformBZMesh((1, 1, 1)), np.ones((1, 1, 2, 2, 2)), addresses, 5,
        )


def test_constructor_keeps_existing_physical_vertex_guards():
    values = np.ones((1, 1, 2, 2, 2), dtype=complex)
    values[0, 0, 0, 1, 0] += 1
    with pytest.raises(ValueError, match="exchange"):
        PackedTransverseCubicVertex(UniformBZMesh((1, 1, 1)), values, [0, 1], 2)
    values[...] = np.nan
    with pytest.raises(ValueError, match="finite"):
        PackedTransverseCubicVertex(UniformBZMesh((1, 1, 1)), values, [0, 1], 2)


def test_exact_support_retains_tiny_nonzero_coefficients():
    values = np.zeros((1, 1, 5, 5, 5))
    values[0, 0, 1, 1, 1] = 1e-100
    values[0, 0, 4, 4, 4] = 0.25
    dense = TransverseCubicVertex(UniformBZMesh((1, 1, 1)), values)
    vertex = PackedTransverseCubicVertex.from_dense(dense)
    np.testing.assert_array_equal(vertex.active_dofs, [1, 4])
    assert vertex.coefficients[0, 0, 0, 0, 0] == 1e-100
    assert vertex.full_ndof == 5


def test_empty_support_returns_empty_without_fft(monkeypatch):
    dense = TransverseCubicVertex(UniformBZMesh((1, 1, 1)), np.zeros((1, 1, 2, 2, 2)))
    vertex = PackedTransverseCubicVertex.from_dense(dense)
    grid = SignedFrequencyGrid(0.1, 1)
    green = np.empty((1, 3, 0, 0))
    monkeypatch.setattr(np.fft, "fft", lambda *args, **kwargs: pytest.fail("empty support FFT"))
    lesser, greater = cubic_born_self_energy_q_packed_fft(grid, vertex, green, green, max_bytes=1)
    assert lesser.shape == greater.shape == (1, 3, 0, 0)
    assert plan_packed_born_fft(grid, vertex).peak_owned_bytes == 0
    assert not np.any(embed_packed_self_energy(vertex, lesser))


def test_projection_embedding_noncontiguous_inputs_and_arbitrary_order():
    vertex = PackedTransverseCubicVertex(UniformBZMesh((1, 1, 1)), np.ones((1, 1, 2, 2, 2)), [3, 1], 4)
    full = np.arange(48.0).reshape(1, 3, 4, 4).swapaxes(-1, -2)
    packed = project_transverse_green(vertex, full)
    expected = full[:, :, [3, 1]][:, :, :, [3, 1]]
    np.testing.assert_array_equal(packed, expected)
    embedded = embed_packed_self_energy(vertex, packed)
    for i in range(4):
        for j in range(4):
            np.testing.assert_array_equal(embedded[:, :, i, j], full[:, :, i, j] if i in [3, 1] and j in [3, 1] else 0)
    assert not np.shares_memory(packed, full)


def test_dimension_only_material_plan_has_fourth_power_work():
    plan = plan_packed_born_fft_dimensions(4, 601, 120, workers=8, bin_tile=1)
    assert plan.n_fft == 2048
    assert plan.vertex_bytes == 16 * 16 * 120**3 + 8 * 120
    assert plan.contraction_work == 6 * 16 * 2048 * 120**4
    assert plan.per_component_contraction_work * 2 == plan.contraction_work
    assert plan.output_bytes == plan.input_green_bytes == 2 * 16 * 4 * 601 * 120**2
    assert plan.effective_workers == plan.max_inflight_futures == 8
    assert plan.peak_owned_bytes > plan.output_bytes + plan.transformed_green_bytes


@pytest.mark.parametrize("kwargs", [
    {"workers": True}, {"workers": 0}, {"external_tile": 0}, {"bin_tile": -1},
])
def test_invalid_planner_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        plan_packed_born_fft_dimensions(4, 9, 3, **kwargs)


@pytest.mark.parametrize("dimensions", [(0, 3, 1), (1, 4, 1), (1, 3, -1), (True, 3, 1)])
def test_invalid_plan_dimensions(dimensions):
    with pytest.raises((TypeError, ValueError)):
        plan_packed_born_fft_dimensions(*dimensions)


@pytest.mark.parametrize("budget", ["bytes", "work"])
def test_kernel_resource_guards_precede_outputs_and_fft(monkeypatch, budget):
    vertex, grid = _vertex(), SignedFrequencyGrid(0.1, 1)
    green = np.ones((1, 3, 2, 2))
    plan = plan_packed_born_fft(grid, vertex)
    options = {"max_bytes": plan.peak_owned_bytes - 1} if budget == "bytes" else {"max_contraction_work": plan.contraction_work - 1}
    monkeypatch.setattr(np, "zeros", lambda *args, **kwargs: pytest.fail("allocation before guard"))
    monkeypatch.setattr(np.fft, "fft", lambda *args, **kwargs: pytest.fail("FFT before guard"))
    with pytest.raises(ValueError, match="max_"):
        cubic_born_self_energy_q_packed_fft(grid, vertex, green, green, **options)


@pytest.mark.parametrize("bad", ["nan", "object", "shape", "sequence"])
def test_green_guards_precede_outputs(monkeypatch, bad):
    vertex, grid = _vertex(), SignedFrequencyGrid(0.1, 1)
    green = np.ones((1, 3, 2, 2))
    invalid = green.copy()
    if bad == "nan":
        invalid[0, 0, 0, 0] = np.nan
    elif bad == "object":
        invalid = invalid.astype(object)
    elif bad == "shape":
        invalid = invalid[:, :-1]
    else:
        invalid = invalid.tolist()
    monkeypatch.setattr(np, "zeros", lambda *args, **kwargs: pytest.fail("allocation before input guard"))
    with pytest.raises((ValueError, TypeError)):
        cubic_born_self_energy_q_packed_fft(grid, vertex, invalid, green)


def test_slice_guards_and_zero_support(monkeypatch):
    vertex = np.ones((2, 2, 2))
    green = np.ones((3, 2, 2))
    with pytest.raises(ValueError, match="max_bytes"):
        contract_packed_fft_bins(vertex, green, green, max_bytes=1)
    with pytest.raises(ValueError, match="max_contraction_work"):
        contract_packed_fft_bins(vertex, green, green, max_contraction_work=1)
    vertex[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        contract_packed_fft_bins(vertex, green, green)
    monkeypatch.setattr(np, "matmul", lambda *args, **kwargs: pytest.fail("empty contraction"))
    assert contract_packed_fft_bins(np.empty((0, 0, 0)), np.empty((3, 0, 0)), np.empty((3, 0, 0))).shape == (3, 0, 0)


def test_projection_embedding_memory_and_finite_guards():
    vertex = _vertex()
    green = np.ones((1, 3, 2, 2))
    for operation in (project_transverse_green, embed_packed_self_energy):
        with pytest.raises(ValueError, match="max_bytes"):
            operation(vertex, green, max_bytes=1)
        broken = green.copy()
        broken[0, 1, 0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            operation(vertex, broken)


def test_projection_embedding_reject_overflow_in_output_conversion():
    # On platforms with extended longdouble, finite input can exceed complex128.
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("longdouble has no extended range on this platform")
    vertex = _vertex()
    green = np.ones((1, 3, 2, 2), dtype=np.longdouble)
    green[0, 0, 0, 0] = np.finfo(np.longdouble).max
    for operation in (project_transverse_green, embed_packed_self_energy):
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
            operation(vertex, green)


def test_threaded_bins_bound_futures_and_restore_blas_after_success_and_failure(monkeypatch):
    lock = Lock()
    pending = peak = 0

    class TrackingExecutor(ThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            nonlocal pending, peak
            with lock:
                pending += 1
                peak = max(peak, pending)
            future = super().submit(function, *args, **kwargs)

            def finished(_):
                nonlocal pending
                with lock:
                    pending -= 1

            future.add_done_callback(finished)
            return future

    def blas_state():
        return sorted((row["prefix"], row["num_threads"]) for row in threadpool_info() if row["user_api"] == "blas")

    original = np.matmul
    fail = False
    seen = []

    def observed_matmul(*args, **kwargs):
        seen.append(blas_state())
        if fail:
            raise RuntimeError("injected matmul failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(transverse_packed, "ThreadPoolExecutor", TrackingExecutor)
    monkeypatch.setattr(np, "matmul", observed_matmul)
    vertex = np.ones((2, 2, 2))
    green = np.broadcast_to(np.eye(2), (4, 2, 2))
    with threadpool_limits(limits=2, user_api="blas"):
        before = blas_state()
        result = contract_packed_fft_bins(vertex, green, green, workers=2, bin_tile=1, external_tile=1)
        np.testing.assert_array_equal(result, np.full((4, 2, 2), 4.0))
        assert blas_state() == before
        fail = True
        with pytest.raises(RuntimeError, match="injected"):
            contract_packed_fft_bins(vertex, green, green, workers=2, bin_tile=1)
        assert blas_state() == before
    assert pending == 0 and 1 <= peak <= 2
    assert seen and all(threads == 1 for state in seen for _, threads in state)
