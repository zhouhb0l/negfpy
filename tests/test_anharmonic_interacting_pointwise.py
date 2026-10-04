"""Independent equation/interpolation checks for pointwise interacting fields."""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from negfpy.anharmonic.acoustic_contacts import _stable_bose_broadening_products
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.harmonic_pointwise import PointwiseHarmonic
from negfpy.anharmonic.interacting_pointwise import GridSelfEnergyInterpolator, PointwiseInteracting
from negfpy.anharmonic.retarded import retarded_from_keldysh
from negfpy.anharmonic.solver import _solve_green
from negfpy.anharmonic.types import KeldyshSpectrum, SCBAConfig
from test_anharmonic_harmonic_pointwise import _problem


def _sigma(grid, ndof, temperature=0.7, *, complex_matrix=True):
    rng = np.random.default_rng(512)
    matrix = rng.normal(size=(ndof, ndof))
    if complex_matrix:
        matrix = matrix + 1j * rng.normal(size=(ndof, ndof))
    matrix = 0.001 * (matrix @ matrix.conj().T)
    lesser, greater = [], []
    for omega in grid.values:
        if omega == 0:
            n_gamma = np1_gamma = temperature * matrix
        else:
            gamma = omega * (1 + 0.3 * omega ** 2) * matrix
            n_gamma, np1_gamma = _stable_bose_broadening_products(omega, temperature, gamma)
        lesser.append(-1j * n_gamma)
        greater.append(-1j * np1_gamma)
    lesser, greater = np.array(lesser), np.array(greater)
    return KeldyshSpectrum(retarded_from_keldysh(lesser, greater, grid), lesser, greater)


@pytest.mark.parametrize("policy", ["linear-keldysh", "equilibrium-broadening"])
def test_exact_nodes_preserve_original_arrays_and_cutoffs(policy):
    grid = SignedFrequencyGrid(0.025, 60)
    sigma = _sigma(grid, 3)
    interp = GridSelfEnergyInterpolator(grid, sigma, policy=policy, temperature=0.7)
    for index in [0, 1, 3, 59, 61, 66, 119, 120]:
        result = interp.evaluate(grid.values[index])
        for name in ("retarded", "lesser", "greater"):
            np.testing.assert_array_equal(getattr(result, name), getattr(sigma, name)[index])
            assert not np.shares_memory(getattr(result, name), getattr(interp, name))
        assert result.diagnostics["exact_grid_node"]
        result.lesser[:] = 0
        np.testing.assert_array_equal(interp(grid.values[index]).lesser, sigma.lesser[index])
    for omega in [np.nextafter(-grid.cutoff, -np.inf), np.nextafter(grid.cutoff, np.inf)]:
        with pytest.raises(ValueError, match="outside"):
            interp(omega)


@pytest.mark.parametrize("omega", [0, np.nan, np.inf, -np.inf])
def test_rejects_invalid_frequency(omega):
    grid = SignedFrequencyGrid(0.1, 3)
    interp = GridSelfEnergyInterpolator(grid, _sigma(grid, 2))
    with pytest.raises(ValueError, match="finite and nonzero"):
        interp(omega)


def test_linear_convex_interpolation_independent_reference_and_fdt_negative_control():
    grid = SignedFrequencyGrid(0.5, 5)
    sigma = _sigma(grid, 3)
    interp = GridSelfEnergyInterpolator(grid, sigma)
    omega = 0.7
    result = interp(omega)
    left, right, weight = 6, 7, 0.4
    expected_l = (1-weight) * sigma.lesser[left] + weight * sigma.lesser[right]
    expected_g = (1-weight) * sigma.greater[left] + weight * sigma.greater[right]
    shift = 0.5 * (sigma.retarded + sigma.retarded.swapaxes(1, 2).conj())
    expected_r = (1-weight) * shift[left] + weight * shift[right] + 0.5 * (expected_g - expected_l)
    for actual, expected in [(result.lesser, expected_l), (result.greater, expected_g), (result.retarded, expected_r)]:
        np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=2e-17)
    assert not np.allclose(result.greater, np.exp(omega/0.7) * result.lesser, rtol=1e-3, atol=1e-12)
    np.testing.assert_allclose(result.retarded-result.retarded.conj().T, result.greater-result.lesser, atol=1e-17)
    for field in [result.lesser, result.greater]:
        assert np.linalg.eigvalsh(1j * field).min() >= -1e-16


def test_equilibrium_policy_uses_query_bose_products_and_fdt():
    grid = SignedFrequencyGrid(0.5, 5)
    sigma = _sigma(grid, 3)
    interp = GridSelfEnergyInterpolator(grid, sigma, policy="equilibrium-broadening", temperature=0.7)
    for omega in [0.7, -0.7, 0.013, -0.013, 1e-10, -1e-10]:
        result = interp(omega)
        gamma = 1j * (result.retarded-result.retarded.conj().T)
        n, np1 = _stable_bose_broadening_products(omega, 0.7, gamma)
        np.testing.assert_allclose(result.lesser, -1j*n, rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(result.greater, -1j*np1, rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(result.greater, np.exp(omega/0.7)*result.lesser, rtol=2e-14, atol=1e-16)
        for field in [result.lesser, result.greater]:
            assert np.linalg.eigvalsh(1j * field).min() > -1e-16


@pytest.mark.parametrize("temperatures,policy", [((0.7,0.7),"linear-keldysh"), ((0.7,0.7),"equilibrium-broadening"), ((0.2,1.1),"linear-keldysh")])
def test_nonzero_interacting_fields_match_production_shared_grid(temperatures, policy):
    problem = _problem(temperatures=temperatures)
    config = SCBAConfig(eta_lead=0.025)
    sigma = _sigma(problem.grid, problem.ndof)
    left, right = build_contact_spectra(problem, config.eta_lead)
    production, actual_sigma = _solve_green(problem, left, right, sigma.lesser, sigma.greater, config)
    evaluator = PointwiseInteracting(problem, config, actual_sigma, policy=policy, selected_entries=[(0,1)])
    for offset in [-56,-55,-54,54,55,56]:
        index = problem.grid.zero_index+offset
        result = evaluator.evaluate(problem.grid.values[index], diagnostic=True)
        for name in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(getattr(result,name), getattr(production,name)[index], rtol=4e-13, atol=4e-13)
        assert result.diagnostics["dyson_backward_error"] < 1e-14
        assert result.diagnostics["green_keldysh_identity_relative"] < 1e-13
        json.dumps(result.diagnostics, allow_nan=False)
        for side in ["left", "right"]:
            assert result.diagnostics["surface_equations"][side]["equation_residual"] < 1e-12


@pytest.mark.parametrize("policy", ["linear-keldysh", "equilibrium-broadening"])
def test_zero_sigma_reproduces_offgrid_harmonic_and_callback(policy):
    problem = _problem(acoustic=True)
    config = SCBAConfig(eta_lead=0.025)
    sigma = KeldyshSpectrum.zeros(n_frequency=problem.grid.n_points, ndof=problem.ndof)
    evaluator = PointwiseInteracting(problem, config, sigma, policy=policy)
    harmonic = PointwiseHarmonic(problem, config)
    for omega in [-1.37291, 1.37291]:
        result = evaluator.evaluate(omega)
        reference = harmonic.evaluate(omega)
        for name in ["retarded", "lesser", "greater"]:
            np.testing.assert_array_equal(getattr(result,name), getattr(reference,name))
        for actual,expected in zip(evaluator(omega), (result.lesser,result.greater)):
            np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize("policy", ["linear-keldysh", "equilibrium-broadening"])
def test_signed_reflection_keldysh_and_independent_greater(policy):
    problem = _problem(acoustic=True)
    sigma = _sigma(problem.grid, problem.ndof, complex_matrix=False)
    evaluator = PointwiseInteracting(problem, SCBAConfig(eta_lead=0.025), sigma, policy=policy)
    positive = evaluator.evaluate(1.37291)
    negative = evaluator.evaluate(-1.37291)
    np.testing.assert_allclose(negative.retarded, positive.retarded.conj(), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(negative.lesser, positive.greater.T, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(negative.greater, positive.lesser.T, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(positive.greater-positive.lesser, positive.retarded-positive.retarded.conj().T, rtol=1e-13, atol=1e-13)
    if policy == "equilibrium-broadening":
        np.testing.assert_allclose(positive.greater, np.exp(1.37291/0.7)*positive.lesser, rtol=1e-13, atol=1e-13)


def test_nonequilibrium_greater_is_not_global_fdt():
    problem = _problem(acoustic=True, temperatures=(0.2,1.1))
    evaluator = PointwiseInteracting(problem, SCBAConfig(eta_lead=0.025), _sigma(problem.grid,problem.ndof))
    result = evaluator.evaluate(1.37291)
    for temperature in [0.2,0.7,1.1]:
        assert not np.allclose(result.greater, np.exp(1.37291/temperature)*result.lesser, rtol=1e-3, atol=1e-9)


def test_concurrent_calls_do_not_mutate_inputs():
    problem = _problem()
    sigma = _sigma(problem.grid,problem.ndof)
    before = sigma.retarded.copy()
    evaluator = PointwiseInteracting(problem, SCBAConfig(eta_lead=0.025), sigma, policy="equilibrium-broadening")
    frequencies = [-1.37291,-0.019,0.019,1.37291]
    serial = [evaluator.evaluate(omega) for omega in frequencies]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(evaluator.evaluate,frequencies))
    for first,second in zip(serial,parallel):
        for name in ["retarded","lesser","greater"]:
            np.testing.assert_array_equal(getattr(first,name),getattr(second,name))
    np.testing.assert_array_equal(sigma.retarded,before)
    for name in ["retarded","lesser","greater","gamma","shift"]:
        assert not getattr(evaluator.interpolator,name).flags.writeable


@pytest.mark.parametrize("field,change,error", [("lesser",0.1,"antihermiticity"), ("greater",0.1,"antihermiticity"), ("retarded",0.1j,"keldysh_identity")])
def test_rejects_invalid_input_algebra(field, change, error):
    grid=SignedFrequencyGrid(0.1,3)
    sigma=_sigma(grid,2)
    arrays={name:getattr(sigma,name).copy() for name in ["retarded","lesser","greater"]}
    arrays[field][1,0,0]+=change
    with pytest.raises(ValueError,match=error):
        GridSelfEnergyInterpolator(grid,KeldyshSpectrum(**arrays))


def test_equilibrium_rejects_nonthermal_input_even_if_keldysh_consistent():
    grid=SignedFrequencyGrid(0.1,3)
    sigma=_sigma(grid,2)
    with pytest.raises(ValueError,match="FDT"):
        GridSelfEnergyInterpolator(grid,sigma,policy="equilibrium-broadening",temperature=0.3)


@pytest.mark.parametrize("temperatures", [(0.2,1.1),(0,0)])
def test_equilibrium_rejects_inapplicable_reservoirs(temperatures):
    problem=_problem(temperatures=temperatures)
    with pytest.raises(ValueError,match="equal strictly positive"):
        PointwiseInteracting(problem,SCBAConfig(eta_lead=0.025),_sigma(problem.grid,problem.ndof),policy="equilibrium-broadening")


def test_near_zero_does_not_amplify_roundoff_broadening():
    grid=SignedFrequencyGrid(0.1,3)
    sigma=_sigma(grid,2)
    lesser,greater=sigma.lesser.copy(),sigma.greater.copy()
    greater[grid.zero_index,0,0]+=1e-14j
    modified=KeldyshSpectrum(retarded_from_keldysh(lesser,greater,grid),lesser,greater)
    interp=GridSelfEnergyInterpolator(grid,modified,policy="equilibrium-broadening",temperature=0.7)
    assert interp.validation_metrics["zero_gamma_roundoff_removed_max"] > 0
    np.testing.assert_array_equal(interp.gamma[grid.zero_index],np.zeros((2,2)))
    assert np.max(np.abs(interp(1e-14).lesser)) < 0.1
    # The source spectrum and all original nonzero nodes are untouched.
    np.testing.assert_array_equal(interp(grid.values[4]).greater,modified.greater[4])
    np.testing.assert_array_equal(interp.greater[grid.zero_index],modified.greater[grid.zero_index])


def test_no_global_scale_can_hide_one_bad_tiny_node():
    grid=SignedFrequencyGrid(0.1,3)
    sigma=_sigma(grid,2)
    arrays={name:getattr(sigma,name).copy() for name in ["retarded","lesser","greater"]}
    for array in arrays.values():
        array[0]*=1e20
    arrays["lesser"][1,0,0]+=1e-7
    with pytest.raises(ValueError,match="antihermiticity"):
        GridSelfEnergyInterpolator(grid,KeldyshSpectrum(**arrays))


def test_rejects_shape_policy_and_tolerance():
    grid=SignedFrequencyGrid(0.1,3)
    sigma=_sigma(grid,2)
    with pytest.raises(ValueError,match="frequency count"):
        GridSelfEnergyInterpolator(SignedFrequencyGrid(0.1,4),sigma)
    with pytest.raises(ValueError,match="Unknown"):
        GridSelfEnergyInterpolator(grid,sigma,policy="cubic")
    with pytest.raises(ValueError,match="tolerances"):
        GridSelfEnergyInterpolator(grid,sigma,rtol=np.nan)
    with pytest.raises(ValueError,match="strictly positive"):
        GridSelfEnergyInterpolator(grid,sigma,policy="equilibrium-broadening",temperature=0)


def test_signed_extremely_near_zero_preserves_broadening_slope():
    grid = SignedFrequencyGrid(0.1, 3)
    sigma = _sigma(grid, 2, complex_matrix=False)
    interp = GridSelfEnergyInterpolator(grid, sigma, policy="equilibrium-broadening", temperature=0.7)
    positive = interp(1e-20)
    negative = interp(-1e-20)
    # The leading Bose product is T times the one-sided Gamma slope. Compute
    # it without subtracting two nearly equal weights or Keldysh fields.
    expected = -1j * 0.7 * interp.gamma[grid.zero_index+1] / grid.step
    for result in [positive, negative]:
        np.testing.assert_allclose(result.lesser, expected, rtol=2e-14, atol=1e-18)
        np.testing.assert_allclose(result.greater, expected, rtol=2e-14, atol=1e-18)
        assert result.diagnostics["left_weight"] > 0
        assert result.diagnostics["right_weight"] > 0
    np.testing.assert_allclose(negative.lesser, positive.greater.T, rtol=2e-14, atol=1e-18)
