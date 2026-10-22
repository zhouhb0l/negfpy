"""Bounded fresh-channel feedback against independent tiny finite algebra.

These tests use synthetic fields only.  They do not load saved contours,
material arrays, checkpoints, or archived calculation results.
"""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal, localcontext
from itertools import product

import numpy as np
import pytest

import negfpy.anharmonic.transverse_channel_feedback as feedback
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_solver import TransverseCurrents, TransverseSpectrum
from negfpy.anharmonic.types import SolverScale

CHANNELS = ("retarded", "statistical", "spectral")


def _pv(spectral):
    """Literal same-grid principal-value sum, independent of all FFT helpers."""
    result = spectral.copy() / 2
    for q, w, other in product(range(len(spectral)), range(spectral.shape[1]),
                               range(spectral.shape[1])):
        if w != other:
            result[q, w] += 1j * spectral[q, other] / (2 * np.pi * (w - other))
    return result


def _contact(mesh, grid, indices, side, temperature, damping):
    nq, size = mesh.n_qpoints, len(indices)
    weight = np.empty((nq, size, size), complex)
    for q in range(nq):
        phase = np.exp(2j * np.pi * q / nq)
        weight[q] = damping * np.array([[1.2, .19 * phase],
                                        [.19 * phase.conjugate(), .8]])
    bose_product = np.empty(grid.n_points)
    for w, omega in enumerate(grid.values):
        bose_product[w] = (temperature if omega == 0 else
                           omega / np.expm1(omega / temperature))
    retarded = -1j * grid.values[None, :, None, None] * weight[:, None]
    lesser = -2j * bose_product[None, :, None, None] * weight[:, None]
    greater = lesser + retarded - retarded.conj().swapaxes(-1, -2)
    raw = TransverseContact(
        mesh, grid, TransverseSpectrum(retarded, lesser, greater), 2 * weight,
        tuple({"synthetic": "independent-ohmic"} for _ in range(nq)), {},
        side, .0015625, temperature,
    )
    return CompactTransverseContact(raw, 3, tuple(indices))


def _fixture(nq=3):
    mesh, grid = UniformBZMesh((nq, 1, 1)), SignedFrequencyGrid(.37, 2)
    factors = np.empty((2, nq, 2), complex)
    for rank, q, a in product(range(2), range(nq), range(2)):
        factors[rank, q, a] = ((rank + 2) * (a + 1) / 9
                               + (a - rank + 2) * np.exp(2j * np.pi * q / nq) / 13)
    coefficients = np.zeros((nq, nq, 2, 2, 2), complex)
    for q, p, a, c, d in product(range(nq), range(nq), range(2), range(2), range(2)):
        for rank, strength in enumerate((.8, -.27)):
            coefficients[q, p, a, c, d] += (
                strength * factors[rank, (-q) % nq, a]
                * factors[rank, p, c] * factors[rank, (q - p) % nq, d]
            )
    vertex = PackedTransverseCubicVertex(mesh, coefficients, np.array([2, 0]), 3)
    dynamical = np.tile(np.diag([1.3, 1.6, 1.9]), (nq, 1, 1)).astype(complex)
    for q in range(nq):
        phase = np.exp(2j * np.pi * q / nq)
        dynamical[q, 0, 1] = -.12 - .035 * phase
        dynamical[q, 1, 2] = -.09 + .021 * phase
        dynamical[q, 0, 2] = -.04 * phase
        for a, b in ((0, 1), (1, 2), (0, 2)):
            dynamical[q, b, a] = dynamical[q, a, b].conjugate()
    left = _contact(mesh, grid, (0, 1), "left", .91, .31)
    right = _contact(mesh, grid, (1, 2), "right", .57, .23)
    return grid, vertex, dynamical, left, right


def _sigma(grid, vertex, *, huge=False):
    shape = (vertex.mesh.n_qpoints, grid.n_points, 2, 2)
    c, d = np.empty(shape, complex), np.empty(shape, complex)
    for q, w in product(range(shape[0]), range(shape[1])):
        phase, omega = np.exp(2j * np.pi * q / shape[0]), grid.values[w]
        weight = np.array([[1.0, .17 * phase], [.17 * phase.conjugate(), .7]])
        envelope = .002 * np.exp(-.43 * omega**2)
        c[q, w] = -1j * (2**56 if huge else envelope * np.cosh(omega / .83)) * weight
        d[q, w] = -2j * envelope * np.sinh(omega / .83) * weight
        # This allowed real antisymmetric Gamma/zero entry must not be zeroed.
        d[q, w] += 2e-6 * np.exp(-.43 * omega**2) * np.array([[0, 1], [-1, 0]])
    return feedback.ChannelSpectrum(_pv(d), c, d)


def _embed(value, indices, dimension):
    result = np.zeros(value.shape[:2] + (dimension, dimension), complex)
    for a, i in enumerate(indices):
        for b, j in enumerate(indices):
            result[:, :, i, j] = value[:, :, a, b]
    return result


def _reference(grid, vertex, dynamical, left, right, sigma, scale, g):
    """Full Dyson, explicit six-index Born, direct PV, ordinary contour currents."""
    nq, nw, full = vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof
    shape = (nq, nw, full, full)
    reservoirs = []
    for contact in (left, right):
        fields = contact.spectrum
        reservoirs.append({
            "retarded": _embed(fields.retarded, contact.device_indices, full),
            "statistical": _embed((fields.greater + fields.lesser) / 2, contact.device_indices, full),
            "spectral": _embed(fields.greater - fields.lesser, contact.device_indices, full),
        })
    interaction = {name: np.zeros(shape, complex) if sigma is None else
                   _embed(getattr(sigma, name), vertex.active_dofs, full) for name in CHANNELS}
    sources = {name: reservoirs[0][name] + reservoirs[1][name] + interaction[name]
               for name in CHANNELS}
    r = np.linalg.inv(grid.values[None, :, None, None]**2 * np.eye(full)
                      - dynamical[:, None] - sources["retarded"])
    advanced = r.conj().swapaxes(-1, -2)
    green = {"retarded": r,
             "statistical": r @ sources["statistical"] @ advanced,
             "spectral": r @ sources["spectral"] @ advanced}
    active = vertex.active_dofs
    retained = {name: value[:, :, active[:, None], active[None, :]]
                for name, value in green.items()}
    mapped_c, mapped_d = (np.zeros((nq, nw, 2, 2), complex) for _ in range(2))
    c, d = retained["statistical"], retained["spectral"]
    prefactor = .5j * grid.step / (2 * np.pi * nq) * g**2
    for q, p, w, internal in product(range(nq), range(nq), range(nw), range(nw)):
        partner_w = w - internal + grid.zero_index
        if not 0 <= partner_w < nw:
            continue
        partner_q = (q - p) % nq
        for a, b, i, j, k, ell in product(range(2), repeat=6):
            factor = (prefactor * vertex.coefficients[q, p, a, i, j]
                      * vertex.coefficients[q, p, b, k, ell].conjugate())
            first = (p, internal, i, k)
            second = (partner_q, partner_w, j, ell)
            mapped_c[q, w, a, b] += factor * (c[first] * c[second] + d[first] * d[second] / 4)
            mapped_d[q, w, a, b] += factor * (c[first] * d[second] + d[first] * c[second])
    mapped = {"retarded": _pv(mapped_d), "statistical": mapped_c, "spectral": mapped_d}
    currents = []
    for source in (*reservoirs, interaction):
        value = 0.0
        for q, w in product(range(nq), grid.positive_indices):
            source_less = source["statistical"][q, w] - source["spectral"][q, w] / 2
            source_greater = source["statistical"][q, w] + source["spectral"][q, w] / 2
            green_less = green["statistical"][q, w] - green["spectral"][q, w] / 2
            green_greater = green["statistical"][q, w] + green["spectral"][q, w] / 2
            value += grid.values[w] * np.trace(source_greater @ green_less
                                               - source_less @ green_greater).real
        currents.append(scale.to_physical_current(value * grid.integration_weight / nq))
    return retained, mapped, currents, sources


def _evaluate(args, **kwargs):
    return feedback.evaluate_channel_born_map(*args, workers=1, external_tile=1,
                                             bin_tile=2, **kwargs)


def _config(**overrides):
    return replace(feedback.ChannelSCBAConfig(
        max_evaluations=3, workers=1, external_tile=1, bin_tile=2,
        max_seconds=60, time_reserve_seconds=1, initial_map_estimate_seconds=.01,
    ), **overrides)


def _constant(value=0.0):
    return feedback.ChannelSpectrum(*(np.full((3, 5, 2, 2), value, complex) for _ in CHANNELS))


def _fake(monkeypatch, *, mapped=None, currents=None, fail_at=None, advance=None):
    """Algebra-free evaluator used only for state, stopping and screen controls."""
    calls = []
    mapped = _constant(1) if mapped is None else mapped
    currents = TransverseCurrents(1, -1, 0) if currents is None else currents
    def evaluate(grid, vertex, dynamical, left, right, **kwargs):
        calls.append(kwargs["sigma_input"])
        if advance:
            advance()
        if len(calls) == fail_at:
            raise ValueError("injected fresh evaluation failure")
        plan = feedback.plan_channel_born_map(3, 5, 3, 2, 2, 2,
                                              workers=1, external_tile=1, bin_tile=2)
        return feedback.ChannelBornMapEvaluation(_constant(len(calls)), mapped, currents,
                                                  {}, plan, kwargs["sigma_input"] is None,
                                                  kwargs["g"])
    monkeypatch.setattr(feedback, "evaluate_channel_born_map", evaluate)
    return calls


@pytest.mark.parametrize("with_input", [False, True])
def test_fresh_map_matches_full_noncommuting_complex_reference(with_input):
    args = _fixture()
    grid, vertex, dynamical, left, right = args
    sigma = _sigma(grid, vertex) if with_input else None
    scale, g = SolverScale(omega0=1.7, hbar=.6), .023
    inputs = [dynamical, vertex.coefficients]
    inputs += [getattr(contact.spectrum, name) for contact in (left, right)
               for name in ("retarded", "lesser", "greater")]
    if sigma is not None:
        inputs += [getattr(sigma, name) for name in CHANNELS]
    before = [value.copy() for value in inputs]
    result = _evaluate(args, sigma_input=sigma, scale=scale, g=g)
    # Scalar diagnostic evidence must be persistable without a NumPy encoder.
    json.dumps(dict(result.checks), allow_nan=False)
    green, mapped, currents, total = _reference(*args, sigma, scale, g)
    for name in CHANNELS:
        np.testing.assert_allclose(getattr(result.green, name), green[name], rtol=3e-12, atol=2e-14)
        np.testing.assert_allclose(getattr(result.map_sigma, name), mapped[name], rtol=4e-12, atol=2e-15)
        assert not getattr(result.green, name).flags.writeable
        assert not getattr(result.map_sigma, name).flags.writeable
    for name, expected in zip(("left", "right", "collision_power"), currents, strict=True):
        assert getattr(result.input_currents, name) == pytest.approx(expected, rel=1e-10, abs=2e-15)
    assert result.input_sigma_is_zero is (not with_input)
    assert np.max(np.abs(dynamical.imag)) > .01
    assert np.max(np.abs(vertex.coefficients.imag)) > .001
    assert np.max(np.abs(dynamical[1] @ total["statistical"][1, -1]
                         - total["statistical"][1, -1] @ dynamical[1])) > .01
    active = vertex.active_dofs
    reduced = np.linalg.inv(grid.values[None, :, None, None]**2 * np.eye(2)
                            - dynamical[:, active[:, None], active[None, :]][:, None]
                            - total["retarded"][:, :, active[:, None], active[None, :]])
    assert np.max(np.abs(result.green.retarded - reduced)) > 1e-3
    for value, original in zip(inputs, before, strict=True):
        np.testing.assert_array_equal(value, original)


def test_coupling_squared_applied_once_to_new_born_image():
    args = _fixture()
    first, second = (_evaluate(args, g=value) for value in (.019, .038))
    for name in CHANNELS:
        np.testing.assert_array_equal(getattr(first.green, name), getattr(second.green, name))
        np.testing.assert_allclose(getattr(second.map_sigma, name), 4 * getattr(first.map_sigma, name), rtol=2e-15)


def test_huge_statistical_source_keeps_independent_gamma_zero_spectral_channel():
    args = _fixture()
    grid, vertex = args[:2]
    sigma = _sigma(grid, vertex, huge=True)
    roundtrip = (sigma.statistical + sigma.spectral / 2) - (sigma.statistical - sigma.spectral / 2)
    assert np.max(np.abs(roundtrip - sigma.spectral)) > 1e-3
    assert sigma.spectral[0, grid.zero_index, 0, 1].real == 2e-6
    result = _evaluate(args, sigma_input=sigma, g=0)
    green, _, _, _ = _reference(*args, sigma, SolverScale(), 0)
    np.testing.assert_allclose(result.green.spectral, green["spectral"], rtol=2e-12, atol=2e-15)
    assert abs(result.green.spectral[0, grid.zero_index, 0, 1].real) > 1e-7
    assert result.checks["full_green_local"]["keldysh_identity"]["ratio"] < 1
    assert result.checks["active_green"]["keldysh_identity"]["ratio"] < 1
    assert result.checks["input_sigma_exports"]["spectral_roundtrip"]["ratio"] > 1
    assert "full_green_exports" in result.checks and "active_green_exports" in result.checks
    for name in CHANNELS:
        np.testing.assert_array_equal(getattr(result.map_sigma, name), 0)


def test_independent_spectral_defect_fails_before_born_without_projection(monkeypatch):
    args = _fixture()
    sigma = _sigma(*args[:2])
    sigma.spectral[0, 2, 0, 0] += 1e-4
    before = sigma.spectral.copy()
    def forbidden(*args, **kwargs):
        raise AssertionError("Born must not run after invalid independent raw D")
    monkeypatch.setattr(feedback, "cubic_born_channels_packed_fft", forbidden)
    with pytest.raises(ValueError, match="(Sigma|spectral)"):
        _evaluate(args, sigma_input=sigma)
    np.testing.assert_array_equal(sigma.spectral, before)


def test_causality_compares_retarded_against_independent_d(monkeypatch):
    args = _fixture()
    sigma = _sigma(*args[:2], huge=True)
    # A real even Hermitian addition preserves R-Rdagger but breaks finite PV.
    sigma.retarded[:, :, 0, 0] += .017
    def forbidden(*args, **kwargs):
        raise AssertionError("Born must not run after failed independent causality")
    monkeypatch.setattr(feedback, "cubic_born_channels_packed_fft", forbidden)
    with pytest.raises(ValueError, match="causal"):
        _evaluate(args, sigma_input=sigma)


def test_raw_contact_identity_defect_is_rejected_before_born(monkeypatch):
    args = list(_fixture())
    left = args[3]
    greater = left.spectrum.greater.copy()
    greater[0, -1, 0, 0] += -1e-4j
    defective = TransverseSpectrum(left.spectrum.retarded, left.spectrum.lesser, greater)
    args[3] = replace(left, contact=replace(left.contact, spectrum=defective))
    def forbidden(*args, **kwargs):
        raise AssertionError("Born must not run after raw contact failure")
    monkeypatch.setattr(feedback, "cubic_born_channels_packed_fft", forbidden)
    with pytest.raises(ValueError, match="(contact|left)"):
        _evaluate(args)


def test_stable_current_matches_decimal_when_contour_products_lose_small_difference():
    c_s, d_s, c_g, d_g = (np.array([[value]], complex) for value in
                         (-1j * 2**56, -2j, -1j * (2**56 + 16), -2j))
    with localcontext() as context:
        context.prec = 80
        expected = float(Decimal(2) * Decimal(2**56)
                         - Decimal(2) * Decimal(2**56 + 16))
    result = feedback.channel_exchange(c_s, d_s, c_g, d_g)
    assert result == expected == -32
    exported = np.trace((c_s + d_s / 2) @ (c_g - d_g / 2)
                         - (c_s - d_s / 2) @ (c_g + d_g / 2)).real
    assert exported != expected
    # A complex 2DOF case also fixes trace order without commuting matrices.
    first_c = np.array([[2j, .3 + .4j], [-.3 + .4j, 3j]])
    first_d = np.array([[.5j, -.1 + .2j], [.1 + .2j, 1j]])
    green_c = np.array([[1.2j, .7 - .1j], [-.7 - .1j, .8j]])
    green_d = np.array([[.2j, .4 + .1j], [-.4 + .1j, .3j]])
    expected_matrix = np.trace((first_c + first_d / 2) @ (green_c - green_d / 2)
                                - (first_c - first_d / 2) @ (green_c + green_d / 2)).real
    assert feedback.channel_exchange(first_c, first_d, green_c, green_d) == pytest.approx(expected_matrix)


def test_three_fresh_evaluations_match_independent_feedback_and_keep_state_binding():
    args, states, starts = _fixture(), [], []
    scale = SolverScale(omega0=1.7, hbar=.6)
    config = _config(sigma_atol=1e-30, sigma_rtol=1e-14)
    result = feedback.run_channel_scba(*args, g=.017, scale=scale, config=config,
                                       on_evaluation=states.append,
                                       on_start=lambda index, sigma: starts.append((index, sigma)))
    assert len(states) == len(result.history) == 3
    assert not result.converged and result.stop_reason == "max_evaluations"
    assert result.failure_reason is None and result.state is states[-1]
    assert starts[0] == (1, None) and states[0].sigma_input is None
    candidate = None
    for index, state in enumerate(states):
        assert state.index == index + 1
        assert starts[index][1] is state.sigma_input
        green, mapped, currents, _ = _reference(*args, candidate, scale, .017)
        for name in CHANNELS:
            np.testing.assert_allclose(getattr(state.evaluation.green, name), green[name], rtol=3e-12, atol=2e-14)
            np.testing.assert_allclose(getattr(state.evaluation.map_sigma, name), mapped[name], rtol=4e-12, atol=2e-15)
        for name, expected in zip(("left", "right", "collision_power"), currents, strict=True):
            assert getattr(state.evaluation.input_currents, name) == pytest.approx(expected, rel=1e-9, abs=2e-15)
        if index < 2:
            candidate = feedback.ChannelSpectrum(*[
                (1-config.mixing) * (0 if candidate is None else getattr(candidate, name))
                + config.mixing * mapped[name] for name in CHANNELS
            ])
            for name in CHANNELS:
                np.testing.assert_allclose(getattr(states[index+1].sigma_input, name), getattr(candidate, name), rtol=5e-12, atol=2e-15)
                assert not getattr(states[index+1].sigma_input, name).flags.writeable
    assert states[-1].evaluation.input_currents is result.state.evaluation.input_currents


@pytest.mark.parametrize("component", ["retarded", "lesser", "greater"])
def test_original_three_unmixed_sigma_screens_are_each_required(monkeypatch, component):
    mapped = _constant(0)
    if component == "retarded":
        mapped.retarded[0, 0, 0, 0] = 2
    else:
        mapped.statistical[0, 0, 0, 0] = 1
        mapped.spectral[0, 0, 0, 0] = -2 if component == "lesser" else 2
    _fake(monkeypatch, mapped=mapped)
    result = feedback.run_channel_scba(*_fixture(), config=_config(max_evaluations=1))
    record = result.state.record
    assert not result.converged and not record.sigma_passed and record.current_passed
    assert record.sigma_residuals[component].error == 2
    assert record.sigma_residuals[component].ratio > 1


@pytest.mark.parametrize("failed,currents", [
    ("contact_balance", (1.2, 0, -.6)),
    ("collision_power", (.6, 0, -1.2)),
    ("full_balance", (.6, 0, .6)),
])
def test_all_three_original_current_screens_are_required(monkeypatch, failed, currents):
    _fake(monkeypatch, mapped=_constant(0), currents=TransverseCurrents(*currents))
    result = feedback.run_channel_scba(*_fixture(), config=_config(
        max_evaluations=1, current_rtol=0, current_atol_w=1))
    record = result.state.record
    assert record.sigma_passed and not record.current_passed and not result.converged
    assert record.current_residuals[failed].ratio == pytest.approx(1.2)
    assert all(value.ratio <= 1 for name, value in record.current_residuals.items() if name != failed)


def test_huge_c_fixed_mixing_preserves_tiny_d_without_contour_roundtrip():
    args = _fixture()
    first, second = _sigma(*args[:2], huge=True), _sigma(*args[:2])
    before = [getattr(value, name).copy() for value in (first, second) for name in CHANNELS]
    mixed = feedback.mix_channel_spectra(first, second, .1)
    for name in CHANNELS:
        np.testing.assert_allclose(getattr(mixed, name), .9 * getattr(first, name) + .1 * getattr(second, name), rtol=2e-16)
        assert not getattr(mixed, name).flags.writeable
    assert mixed.spectral[0, 2, 0, 1].real == pytest.approx(2e-6, rel=2e-16)
    for actual, original in zip((getattr(value, name) for value in (first, second)
                                  for name in CHANNELS), before, strict=True):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("fail_at", [1, 2])
def test_failed_candidate_is_not_checkpointed_or_promoted(monkeypatch, fail_at):
    calls = _fake(monkeypatch, fail_at=fail_at)
    states, starts = [], []
    result = feedback.run_channel_scba(*_fixture(), config=_config(),
                                       on_evaluation=states.append,
                                       on_start=lambda index, sigma: starts.append((index, sigma)))
    assert len(calls) == len(starts) == fail_at
    assert len(states) == len(result.history) == fail_at - 1
    assert result.stop_reason == "evaluation_error" and not result.converged
    assert "injected fresh evaluation failure" in result.failure_reason
    assert result.state is (None if fail_at == 1 else states[0])
    if result.state:
        assert result.state.sigma_input is None and result.state.index == 1


@pytest.mark.parametrize("hook", ["on_start", "on_evaluation"])
def test_callback_failure_propagates_without_retry(monkeypatch, hook):
    calls = _fake(monkeypatch)
    def fail(*args):
        raise RuntimeError("intentional checkpoint callback failure")
    with pytest.raises(RuntimeError, match="checkpoint callback"):
        feedback.run_channel_scba(*_fixture(), config=_config(), **{hook: fail})
    assert len(calls) == (0 if hook == "on_start" else 1)


def test_time_bound_includes_callback_cost_before_next_candidate(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(feedback, "monotonic", lambda: now[0])
    calls = _fake(monkeypatch, advance=lambda: now.__setitem__(0, now[0] + 8))
    result = feedback.run_channel_scba(*_fixture(), config=_config(max_seconds=20, time_reserve_seconds=3),
                                       on_evaluation=lambda state: now.__setitem__(0, now[0] + 2))
    assert len(calls) == 1 and result.state.index == 1
    assert result.stop_reason == "max_seconds" and result.failure_reason is None
    assert result.history[0].map_seconds == 8


@pytest.mark.parametrize("resource", ["max_bytes", "max_contraction_work", "max_dyson_evaluations"])
def test_resource_gate_precedes_inverse_and_born(monkeypatch, resource):
    def forbidden(*args, **kwargs):
        raise AssertionError("resource rejection must precede numerical work")
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(feedback, "cubic_born_channels_packed_fft", forbidden)
    with pytest.raises(ValueError, match=resource):
        _evaluate(_fixture(), **{resource: 1})
    with pytest.raises(ValueError, match=resource):
        feedback.run_channel_scba(*_fixture(), config=_config(**{resource: 1}))


def test_nonfinite_new_input_rejected_before_inverse(monkeypatch):
    args = _fixture()
    sigma = _sigma(*args[:2])
    sigma.statistical[0, 0, 0, 0] = np.nan
    def forbidden(*args, **kwargs):
        raise AssertionError("inverse must not run on nonfinite channels")
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    with pytest.raises(ValueError, match="finite"):
        _evaluate(args, sigma_input=sigma)


def test_fresh_defaults_require_explicit_continuation_entrypoint(monkeypatch):
    calls = _fake(monkeypatch, mapped=_constant(0))
    fresh = feedback.run_channel_scba(*_fixture(), config=_config(max_evaluations=1))
    assert fresh.state.index == 1 and calls == [None]
    with pytest.raises(ValueError, match="first_index"):
        feedback.run_channel_scba(*_fixture(), sigma_initial=_constant(0), config=_config())
    resumed = feedback.run_channel_scba(*_fixture(), sigma_initial=_constant(0),
                                      first_index=4, config=_config(max_evaluations=1))
    assert resumed.state.index == 4 and resumed.state.sigma_input is not None
