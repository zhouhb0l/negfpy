"""Small equation-level checks; no material data or SCBA runs needed."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from negfpy.anharmonic.harmonic_pointwise import PointwiseHarmonic, surface_equation_metrics
from negfpy.anharmonic._harmonic import (
    assemble_device_dynamical_matrix, build_retarded_contacts,
    solve_keldysh_green, solve_retarded,
)
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.types import (
    CubicSCBAProblem, DenseCubicVertex, HarmonicReservoir, SCBAConfig,
)
from negfpy.core.surface_gf import surface_gf
from negfpy.core.types import Device1D, LeadBlocks


def _problem(acoustic=False, temperatures=(0.7, 0.7)):
    if acoustic:
        lead = LeadBlocks(d00=np.array([[2.0]]), d01=np.array([[-1.0]]))
        device = Device1D(
            onsite_blocks=[np.array([[2.0]]), np.array([[2.1]]), np.array([[2.0]])],
            coupling_blocks=[np.array([[-1.0]]), np.array([[-1.0]])],
        )
        left = HarmonicReservoir(lead, temperatures[0])
        right = HarmonicReservoir(lead, temperatures[1])
    else:
        # Nonsymmetric principal-layer coupling distinguishes left/right
        # surface orientation. Complex explicit contacts detect a mistaken
        # transpose in either self-energy or advanced Green function.
        d00 = np.array([[4.0, 0.2], [0.2, 3.5]], dtype=complex)
        d01 = np.array([[-0.8, 0.35], [-0.1, -0.6]], dtype=complex)
        lead = LeadBlocks(d00=d00, d01=d01, d10=d01.conj().T)
        device = Device1D(
            onsite_blocks=[d00, d00],
            coupling_blocks=[d01],
        )
        left = HarmonicReservoir(
            lead, temperatures[0],
            device_to_lead=np.array([[-0.7, 0.12j], [0.1, -0.65]]),
            device_indices=[1, 0],
        )
        right = HarmonicReservoir(
            lead, temperatures[1],
            device_to_lead=np.array([[-0.72, 0.11j], [0.15, -0.6]]),
            device_indices=[3, 2],
        )
    ndof = device.n_layers * device.dof_per_layer
    return CubicSCBAProblem(
        device=device, left=left, right=right,
        vertex=DenseCubicVertex(np.zeros((ndof, ndof, ndof))),
        grid=SignedFrequencyGrid(0.025, 60),
    )


@pytest.mark.parametrize("acoustic", [False, True])
@pytest.mark.parametrize("temperatures", [(0.7, 0.7), (0.2, 1.1), (0.0, 0.0), (0.0, 0.7)])
def test_matches_uniform_signed_production_harmonic(acoustic, temperatures):
    problem = _problem(acoustic, temperatures)
    config = SCBAConfig(eta_lead=0.025)
    left, right = build_contact_spectra(problem, config.eta_lead)
    from negfpy.anharmonic.solver import _solve_green
    zeros = np.zeros_like(left.retarded)
    production, _ = _solve_green(problem, left, right, zeros, zeros, config)
    retarded = production.retarded
    evaluator = PointwiseHarmonic(problem, config, selected_entries=[(0, 0), (0, 1)])
    for offset in [-56, -55, -54, 54, 55, 56]:
        index = problem.grid.zero_index + offset
        result = evaluator.evaluate(problem.grid.values[index], diagnostic=True)
        record = result.diagnostics
        for name in ("lesser", "greater", "retarded"):
            np.testing.assert_allclose(getattr(result, name), getattr(production, name)[index], rtol=3e-13, atol=3e-13)
        for field in (result.lesser, result.greater):
            np.testing.assert_allclose(field, -field.conj().T, rtol=3e-13, atol=3e-13)
        for saved in record["selected_entries"]:
            row, column = saved["row"], saved["column"]
            np.testing.assert_allclose(
                complex(*saved["retarded"]), retarded[index, row, column],
                rtol=3e-13, atol=3e-13,
            )
        assert record["dyson_backward_error"] < 1e-14
        for side in ("left", "right"):
            assert record["surface_equations"][side]["equation_residual"] < 1e-12
        json.dumps(record, allow_nan=False)


def test_operator_singular_values_and_dyson_error_use_zero_device_eta():
    problem = _problem()
    config = SCBAConfig(eta_lead=0.025)
    evaluator = PointwiseHarmonic(problem, config)
    index = problem.grid.zero_index + 55
    omega = problem.grid.values[index]
    left, right = build_retarded_contacts(problem, config.eta_lead)
    operator = omega ** 2 * np.eye(problem.ndof) - assemble_device_dynamical_matrix(problem.device)
    operator -= left[index] + right[index]
    reference = np.linalg.svd(operator, compute_uv=False)
    record = evaluator.evaluate(omega, diagnostic=True).diagnostics
    assert record["device_operator_singular_min"] == pytest.approx(reference[-1], rel=1e-14)
    assert record["device_operator_singular_max"] == pytest.approx(reference[0], rel=1e-14)
    assert record["eta_device"] == 0.0
    assert record["eta_lead"] == 0.025


def test_surface_equation_detects_reversed_orientation():
    problem = _problem()
    lead = problem.left.lead
    omega, eta = 1.8, 0.025
    left = surface_gf(omega, lead.d00, lead.d10, lead.d01, eta)
    correct = surface_equation_metrics(omega, lead.d00, lead.d10, lead.d01, eta, left)
    wrong = surface_equation_metrics(omega, lead.d00, lead.d01, lead.d10, eta, left)
    assert correct["equation_residual"] < 1e-13
    assert wrong["equation_residual"] > 1e-2
    corrupted = left.copy()
    corrupted[0, 0] += 0.01j
    assert surface_equation_metrics(
        omega, lead.d00, lead.d10, lead.d01, eta, corrupted
    )["equation_residual"] > 1e-3


def test_nonuniform_frequency_and_signed_thermal_factor():
    problem = _problem(acoustic=True)
    evaluator = PointwiseHarmonic(problem, SCBAConfig(eta_lead=0.025))
    omega = 1.3749137
    positive = evaluator.evaluate(omega)
    negative = evaluator.evaluate(-omega)
    # For a real equilibrium system G<(-w)=exp(w/T)*G<(w)^T.
    np.testing.assert_allclose(
        negative.lesser, np.exp(omega / problem.left.temperature) * positive.lesser.T,
        rtol=3e-13, atol=3e-13,
    )


def test_concurrent_calls_do_not_change_inputs_or_results():
    problem = _problem()
    evaluator = PointwiseHarmonic(problem, SCBAConfig(eta_lead=0.025))
    dynamical_before = evaluator.dynamical.copy()
    coupling_before = problem.left.device_to_lead.copy()
    frequencies = [-1.377, -1.371, 1.371, 1.377]
    serial = [evaluator.evaluate(value) for value in frequencies]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(evaluator.evaluate, frequencies))
    for reference, observed in zip(serial, parallel):
        for name in ("lesser", "greater", "retarded"):
            np.testing.assert_array_equal(getattr(reference, name), getattr(observed, name))
        assert reference.diagnostics == observed.diagnostics
        assert "dyson_backward_error" not in observed.diagnostics
    np.testing.assert_array_equal(evaluator.dynamical, dynamical_before)
    np.testing.assert_array_equal(problem.left.device_to_lead, coupling_before)


@pytest.mark.parametrize("omega", [0.0, np.nan, np.inf, -np.inf])
def test_rejects_zero_and_nonfinite_frequency(omega):
    evaluator = PointwiseHarmonic(_problem(), SCBAConfig(eta_lead=0.025))
    with pytest.raises(ValueError, match="finite and nonzero"):
        evaluator.evaluate(omega)


@pytest.mark.parametrize("entry", [(0, 4), (-1, 0), (True, 0), (0.5, 1), (0,)])
def test_rejects_invalid_selected_entry(entry):
    with pytest.raises(ValueError, match="valid integer indices"):
        PointwiseHarmonic(_problem(), SCBAConfig(eta_lead=0.025), selected_entries=[entry])


@pytest.mark.parametrize("temperatures", [(0.7, 0.7), (0.2, 1.1)])
def test_signed_fields_and_independent_greater(temperatures):
    evaluator = PointwiseHarmonic(_problem(acoustic=True, temperatures=temperatures), SCBAConfig(eta_lead=0.025))
    positive = evaluator.evaluate(1.3749137)
    negative = evaluator.evaluate(-1.3749137)
    np.testing.assert_allclose(negative.lesser, positive.greater.T, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(negative.greater, positive.lesser.T, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(positive.greater - positive.lesser, positive.retarded - positive.retarded.conj().T, rtol=3e-13, atol=3e-13)
    if temperatures[0] != temperatures[1]:
        # Neither reservoir temperature can be used as a global FDT shortcut.
        for temperature in temperatures:
            assert not np.allclose(positive.greater, np.exp(1.3749137 / temperature) * positive.lesser, rtol=1e-3, atol=1e-9)


def test_callback_returns_independent_fields_and_internal_arrays_are_readonly():
    problem = _problem()
    evaluator = PointwiseHarmonic(problem, SCBAConfig(eta_lead=0.025))
    result = evaluator.evaluate(-1.37491)
    lesser, greater = evaluator(-1.37491)
    np.testing.assert_array_equal(lesser, result.lesser)
    np.testing.assert_array_equal(greater, result.greater)
    assert not evaluator.dynamical.flags.writeable
    assert not evaluator.contacts[0].coupling.flags.writeable
    assert not np.shares_memory(evaluator.contacts[0].coupling, problem.left.device_to_lead)
    # Modifying a returned field cannot affect subsequent calls.
    lesser[:] = 0
    np.testing.assert_array_equal(evaluator(-1.37491)[0], result.lesser)


def test_rejects_unsupported_device_eta_and_surface_method():
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="eta_device=0"):
        PointwiseHarmonic(_problem(), SimpleNamespace(eta_device=1e-3))
    with pytest.raises(ValueError, match="explicit surface_gf"):
        PointwiseHarmonic(_problem(), SCBAConfig(eta_lead=0.025, surface_gf_method="analytic_diagonal"))
