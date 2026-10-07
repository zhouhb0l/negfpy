"""Direct cut contraction against the unchanged explicit moment reference.

Complex, nonsymmetric lines deliberately rule out accidental conjugation and
transpose shortcuts.  These are algebra tests, not physical state assertions.
"""

from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from negfpy.anharmonic.contracted_cubic_current import build_contracted_cubic_cut
from negfpy.anharmonic import (
    KeldyshSpectrum, SignedFrequencyGrid, SolverScale, SparseCubicVertex,
    connected_cubic_moments, correlations_from_green,
)
from test_anharmonic_cubic_cut_current import _problem, _tensor


def _values():
    # Coordinate 2 is inactive. The (4,4,5) monomial is internal to layer 2,
    # yet it must remain in the response vertex for both crossing cut operators.
    return _tensor(6, [((0, 0, 3), 0.7), ((1, 3, 5), -0.4), ((4, 4, 5), 0.9)])


def _lines(times, ndof):
    rng = np.random.default_rng(4048)
    amplitude = rng.normal(size=(4, ndof, ndof)) + 1j * rng.normal(size=(4, ndof, ndof))
    rate = np.array([0.12 + 0.5j, 0.24 - 0.7j, 0.08 + 1.2j, 0.31 - 1.7j])
    phase = np.exp(-np.asarray(times)[:, None] * rate)
    correlation = np.einsum("tm,mij->tij", phase, amplitude) / 3
    derivative = np.einsum("tm,m,mij->tij", phase, -rate, amplitude) / 3
    return correlation, derivative


def _provider(plan, seen=None):
    def evaluate(times):
        if seen is not None:
            seen.append(np.array(times, copy=True))
        correlation, derivative = _lines(times, plan.operator.ndof)
        return (
            correlation[:, plan.displacement_rows][:, :, plan.active_indices],
            derivative[:, plan.momentum_rows][:, :, plan.active_indices],
        )
    return evaluate


def _explicit_integrand(values, operator, correlation, derivative):
    # Expand the ordered Hamiltonian sum and all six Wick contractions without
    # reusing the implementation's dense response tensor or contraction path.
    result = np.zeros(len(correlation))
    for target, coefficient in zip(operator.cubic_indices, operator.cubic_coefficients):
        i, j, k = target
        kernel = np.zeros(len(correlation), complex)
        for a, b, c in np.argwhere(values != 0):
            for x, y, z in permutations((a, b, c)):
                kernel += values[a, b, c] / 6 * (
                    derivative[:, i, x] * correlation[:, j, y] * correlation[:, k, z]
                )
        result += coefficient * 2 * kernel.imag
    return result


@pytest.mark.parametrize("cut", [1, 2])
@pytest.mark.parametrize("sparse_vertex", [False, True])
@pytest.mark.parametrize("block_size", [1, 2, 7, 100])
def test_streamed_current_matches_explicit_moments_and_ordered_wick_sum(cut, sparse_vertex,
                                                                       block_size, monkeypatch):
    values = _values()
    problem = _problem(values, npl=2, sparse_vertex=sparse_vertex,
                       scale=SolverScale(hbar=2, omega0=3))
    if sparse_vertex:
        monkeypatch.setattr(SparseCubicVertex, "to_dense", lambda *_: pytest.fail("densified"))
    plan = build_contracted_cubic_cut(problem, cut)
    np.testing.assert_array_equal(plan.active_indices, [0, 1, 3, 4, 5])
    # Nonuniform trapezoids and a short final block exercise carry-over edges.
    times = np.linspace(0, 1.8, 26) ** 1.2
    c, p = _lines(times, problem.ndof)
    explicit = connected_cubic_moments(problem.vertex, plan.operator.cubic_indices, times, c, p)
    expected = np.dot(plan.operator.cubic_coefficients, explicit["connected_moments"])
    integrand = _explicit_integrand(values, plan.operator, c, p)
    seen = []
    result = plan.integrate(times, _provider(plan, seen), block_size=block_size)
    np.testing.assert_allclose(result["connected_cubic_solver"], expected, rtol=3e-13, atol=2e-13)
    np.testing.assert_allclose(result["connected_cubic"], 18 * expected, rtol=3e-13, atol=3e-12)
    np.testing.assert_allclose(result["integrand"], integrand, rtol=3e-13, atol=2e-13)
    np.testing.assert_allclose(result["endpoint_integrand"], integrand[-1], atol=2e-13)
    assert max(map(len, seen)) <= block_size
    np.testing.assert_array_equal(np.concatenate(seen), times)
    assert not result["spatial_validation_passed"]
    assert not result["changes_solver_acceptance"]
    assert not result["disconnected_means_included"]
    assert not result["tail_error_estimated"]


def test_internal_response_vertex_terms_cannot_be_dropped():
    values = _values()
    plan = build_contracted_cubic_cut(_problem(values, npl=2), 1)
    times = np.linspace(0, 1.8, 31)
    c, p = _lines(times, 6)
    all_terms = _explicit_integrand(values, plan.operator, c, p)
    cut_only = values.copy()
    for indices in set(permutations((4, 4, 5))):
        cut_only[indices] = 0
    incomplete = _explicit_integrand(cut_only, plan.operator, c, p)
    assert np.max(np.abs(all_terms - incomplete)) > 0.01
    result = plan.integrate(times, _provider(plan), block_size=4)
    np.testing.assert_allclose(result["integrand"], all_terms, atol=2e-13)


def test_all_ordered_coefficients_are_retained_and_source_unchanged():
    values = _tensor(3, [((0, 1, 2), 0.7), ((0, 0, 1), -0.5)])
    values[0, 1, 2] += 5e-11
    problem = _problem(values, sparse_vertex=True)
    before = problem.vertex.coefficients.copy()
    plan = build_contracted_cubic_cut(problem, 1)
    times = np.linspace(0, 0.9, 29)
    c, p = _lines(times, 3)
    result = plan.integrate(times, _provider(plan), block_size=8)
    expected = _explicit_integrand(values, plan.operator, c, p)
    np.testing.assert_allclose(result["integrand"], expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(problem.vertex.coefficients, before)


@pytest.mark.parametrize("internal_only", [False, True])
def test_zero_cut_returns_zero_without_requesting_lines(internal_only):
    values = _tensor(4, [((0, 0, 1), 0.6), ((2, 3, 3), -0.4)] if internal_only else [])
    plan = build_contracted_cubic_cut(_problem(values, npl=2), 1)
    result = plan.integrate([0, 0.3, 0.8], lambda _: pytest.fail("empty cut requested lines"))
    assert result["connected_cubic_solver"] == 0.0
    assert result["connected_cubic"] == 0.0
    np.testing.assert_array_equal(result["integrand"], [0, 0, 0])


def test_green_streaming_matches_original_signed_grid_fourier_and_explicit_moments():
    problem = replace(_problem(_values(), npl=2), grid=SignedFrequencyGrid(step=0.17, n_positive=6))
    rng = np.random.default_rng(63972)
    greater = rng.normal(size=(problem.grid.n_points, 6, 6)) + 1j * rng.normal(
        size=(problem.grid.n_points, 6, 6))
    green = KeldyshSpectrum(retarded=np.zeros_like(greater), lesser=greater[::-1], greater=greater)
    plan = build_contracted_cubic_cut(problem, 2)
    times = np.linspace(0, 2.8, 37)
    lines = correlations_from_green(problem.grid, green, times)
    c = lines["displacement_correlation"]
    p = lines["velocity_displacement_correlation"]
    expected = _explicit_integrand(_values(), plan.operator, c, p)
    result = plan.integrate_from_green(problem.grid, green, times, block_size=5)
    np.testing.assert_allclose(result["integrand"], expected, atol=2e-16, rtol=2e-13)
    np.testing.assert_allclose(result["connected_cubic_solver"], np.trapezoid(expected, times),
                               atol=2e-16, rtol=2e-13)
    assert not result["frequency_accuracy_verified"]
    selected = plan.integrate_from_selected_spectra(
        problem.grid,
        greater[:, plan.displacement_rows[:, None], plan.active_indices[None, :]],
        greater[:, plan.momentum_rows[:, None], plan.active_indices[None, :]],
        times, block_size=3,
    )
    np.testing.assert_allclose(selected["integrand"], expected, atol=2e-16, rtol=2e-13)
    with pytest.raises(ValueError, match="recurrence"):
        plan.integrate_from_green(problem.grid, green, [0, np.pi / problem.grid.step])


@pytest.mark.parametrize("lag", [[0], [0, 0], [0, -1], [0.1, 0.3], [0, np.nan]])
def test_lag_validation_is_independent_of_blocking(lag):
    plan = build_contracted_cubic_cut(_problem(_values(), npl=2), 1)
    with pytest.raises(ValueError):
        plan.integrate(lag, _provider(plan))


@pytest.mark.parametrize("block_size", [0, -1, 1.5, True])
def test_invalid_block_sizes_fail_before_provider_is_called(block_size):
    plan = build_contracted_cubic_cut(_problem(_values(), npl=2), 1)
    with pytest.raises((TypeError, ValueError)):
        plan.integrate([0, 0.2], lambda _: pytest.fail("invalid block was used"), block_size=block_size)


@pytest.mark.parametrize("failure", ["wrong_c_shape", "wrong_p_shape", "nan_later_block"])
def test_invalid_provider_data_including_late_blocks_is_rejected(failure):
    plan = build_contracted_cubic_cut(_problem(_values(), npl=2), 1)
    valid = _provider(plan)

    def provider(times):
        c, p = valid(times)
        if failure == "wrong_c_shape":
            c = c[:, :-1]
        elif failure == "wrong_p_shape":
            p = p[:, :, :-1]
        elif times[0] > 0:
            c[0, 0, 0] = np.nan
        return c, p

    with pytest.raises(ValueError):
        plan.integrate([0, 0.1, 0.2, 0.3, 0.4], provider, block_size=2)


def test_memory_limit_rejects_plan_before_unbounded_dense_response():
    with pytest.raises((ValueError, MemoryError)):
        build_contracted_cubic_cut(_problem(_values(), npl=2), 1, max_dense_tensor_bytes=1)


def test_streamed_dressed_steady_current_retains_contact_cut_continuity():
    # One bounded three-site state from the established centered Drude model.
    # Reuse its explicit-moment route as the independent contraction reference;
    # this test neither changes that model's means nor validates material SCBA.
    from test_anharmonic_steady_quantum_reference import reference

    state = reference.solve_reference()
    old = reference.measure_currents(state)
    problem, green = state["problem"], state["green"]
    times = np.linspace(0, 150, 3001)
    left, right = state["metadata"]["contact_currents"]
    assert left > 0.01 and abs(left + right) < 1e-10
    for cut in old["cuts"]:
        plan = build_contracted_cubic_cut(problem, cut["cut"])
        result = plan.integrate_from_green(problem.grid, green, times, block_size=16)
        np.testing.assert_allclose(result["connected_cubic"], cut["cubic"],
                                   rtol=5e-12, atol=2e-15)
        harmonic = float(plan.operator.harmonic_coefficients @ (
            plan.operator.harmonic_moments_from_green(problem.grid, green)
        )) * plan.operator.current_scale
        total = harmonic + result["connected_cubic"]
        assert abs(total - left) / left < 1e-3
        assert abs(harmonic - left) / left > 1e-3
        assert not result["spatial_validation_passed"]
    assert state["metadata"]["means_prescribed_not_solved"]
    assert not state["metadata"]["material_spatial_validation_passed"]
