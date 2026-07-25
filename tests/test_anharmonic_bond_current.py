import numpy as np
import pytest

from negfpy.anharmonic._harmonic import solve_keldysh_green, solve_retarded
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.current import (
    _evaluate_current_report,
    bond_heat_current_spectra,
    bond_heat_currents,
    contact_heat_current,
    projected_collision_power_spectra,
    projected_eom_exchange_spectra,
    self_energy_nonlocal_ratio,
)
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.solver import solve_one_shot_born
from negfpy.anharmonic.toy import (
    fpu_alpha_chain,
    pinned_scalar_chain,
    pinned_three_polarization_chain,
)
from negfpy.anharmonic.types import (
    AnharmonicResult,
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    KeldyshSpectrum,
    SCBAConfig,
)


def _problem(*, fpu: bool = False) -> CubicSCBAProblem:
    if fpu:
        device, lead, vertex = fpu_alpha_chain(
            4,
            pinning=1.0,
            alpha=0.02,
        )
    else:
        device, lead, vertex = pinned_scalar_chain(
            4,
            pinning=1.0,
            cubic=0.0,
        )
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, temperature=0.8),
        right=HarmonicReservoir(lead, temperature=0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.025, n_positive=104),
    )


@pytest.fixture(scope="module")
def nonlocal_born_case() -> tuple[CubicSCBAProblem, AnharmonicResult]:
    device, lead, vertex = pinned_three_polarization_chain(
        3,
        mass=(1.0, 1.0, 1.0),
        spring=(0.1, 0.1, 0.1),
        pinning=(2.05, 0.61, 0.16),
        g=0.05,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, temperature=0.8),
        right=HarmonicReservoir(lead, temperature=0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=35),
    )
    result = solve_one_shot_born(
        problem,
        SCBAConfig(eta_lead=0.01, current_rtol=1.0e-3),
    )
    return problem, result


def test_harmonic_bond_currents_equal_contact_current() -> None:
    problem = _problem()
    left, right = build_contact_spectra(problem, eta_lead=1.0e-7)
    retarded = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )

    contact = contact_heat_current(problem.grid, left, green, problem.scale)
    spectra = bond_heat_current_spectra(problem, green)
    bonds = bond_heat_currents(problem, green)

    assert spectra.shape == (problem.device.n_layers - 1, problem.grid.n_positive)
    assert bonds.shape == (problem.device.n_layers - 1,)
    assert np.allclose(bonds, contact, rtol=1e-11, atol=1e-12)


def test_harmonic_spatial_report_has_zero_interaction_correction() -> None:
    problem = _problem()
    left, right = build_contact_spectra(problem, eta_lead=1.0e-7)
    retarded = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )
    sigma_pp = KeldyshSpectrum.zeros(
        n_frequency=problem.grid.n_points,
        ndof=problem.ndof,
    )
    currents = _evaluate_current_report(
        problem,
        left,
        right,
        sigma_pp,
        green,
    )

    assert currents.spatial is not None
    assert currents.spatial.sigma_pp_nonlocal_ratio == 0.0
    assert currents.spatial.sigma_pp_is_block_local
    np.testing.assert_array_equal(currents.spatial.eom_layer_exchange, 0.0)
    np.testing.assert_array_equal(currents.spatial.layer_collision_power, 0.0)
    np.testing.assert_array_equal(
        currents.spatial.nonlocal_self_energy_eom_cut_correction,
        0.0,
    )
    np.testing.assert_allclose(currents.spatial.corrected_cut_currents, currents.left)
    np.testing.assert_allclose(currents.spatial.continuity_residual, 0.0, atol=1e-15)


def test_bond_cubic_model_rejects_incomplete_harmonic_current_operator() -> None:
    problem = _problem(fpu=True)
    left, right = build_contact_spectra(problem, eta_lead=1.0e-7)
    retarded = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )

    with pytest.raises(NotImplementedError, match="crosses a layer boundary"):
        bond_heat_currents(problem, green)


def test_cross_cut_vertex_has_no_complete_spatial_report() -> None:
    problem = _problem(fpu=True)
    left, right = build_contact_spectra(problem, eta_lead=1.0e-7)
    retarded = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )
    sigma_pp = KeldyshSpectrum.zeros(
        n_frequency=problem.grid.n_points,
        ndof=problem.ndof,
    )

    report = _evaluate_current_report(problem, left, right, sigma_pp, green)
    assert report.spatial is None


def test_metadata_cannot_enable_bond_operator_for_cross_layer_vertex() -> None:
    original = _problem(fpu=True)
    problem = CubicSCBAProblem(
        device=original.device,
        left=original.left,
        right=original.right,
        vertex=DenseCubicVertex(
            original.vertex.values,
            metadata={"harmonic_bond_current_supported": True},
        ),
        grid=original.grid,
        scale=original.scale,
    )
    left, right = build_contact_spectra(problem, eta_lead=1.0e-7)
    retarded = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )

    with pytest.raises(NotImplementedError, match="crosses a layer boundary"):
        bond_heat_currents(problem, green)


def test_projected_spatial_spectra_match_literal_layer_index_loops(
    nonlocal_born_case: tuple[CubicSCBAProblem, AnharmonicResult],
) -> None:
    problem, result = nonlocal_born_case
    sigma = result.sigma_pp
    green = result.green
    eom = projected_eom_exchange_spectra(problem, sigma, green)
    collision = projected_collision_power_spectra(problem, sigma, green)
    expected_eom = np.zeros_like(eom)
    expected_collision = np.zeros_like(collision)
    positive = problem.grid.positive_indices
    green_a = np.swapaxes(green.retarded.conj(), -1, -2)
    sigma_a = np.swapaxes(sigma.retarded.conj(), -1, -2)
    npl = problem.device.dof_per_layer

    for layer in range(problem.device.n_layers):
        for column, iw in enumerate(positive):
            eom_trace = 0.0j
            collision_trace = 0.0j
            for first in range(layer * npl, (layer + 1) * npl):
                for second in range(problem.ndof):
                    eom_trace += (
                        sigma.retarded[iw, first, second]
                        * green.lesser[iw, second, first]
                        + sigma.lesser[iw, first, second]
                        * green_a[iw, second, first]
                        - green.retarded[iw, first, second]
                        * sigma.lesser[iw, second, first]
                        - green.lesser[iw, first, second]
                        * sigma_a[iw, second, first]
                    )
                    collision_trace += (
                        sigma.greater[iw, first, second]
                        * green.lesser[iw, second, first]
                        - sigma.lesser[iw, first, second]
                        * green.greater[iw, second, first]
                    )
            factor = (
                problem.scale.current_scale * problem.grid.values[iw]
            )
            expected_eom[layer, column] = factor * np.real(eom_trace)
            expected_collision[layer, column] = factor * np.real(
                collision_trace
            )

    np.testing.assert_allclose(eom, expected_eom, rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        collision,
        expected_collision,
        rtol=1e-13,
        atol=1e-14,
    )


def test_nonlocal_scba_self_energy_uses_corrected_cut_closure(
    nonlocal_born_case: tuple[CubicSCBAProblem, AnharmonicResult],
) -> None:
    problem, result = nonlocal_born_case
    currents = result.currents
    spatial = currents.spatial
    assert spatial is not None
    bonds = np.asarray(currents.bond_currents)
    assert spatial.sigma_pp_nonlocal_ratio == pytest.approx(
        self_energy_nonlocal_ratio(problem, result.sigma_pp)
    )
    assert spatial.sigma_pp_nonlocal_ratio > 0.1
    assert not spatial.sigma_pp_is_block_local

    block_mask = np.zeros((problem.ndof, problem.ndof), dtype=bool)
    npl = problem.device.dof_per_layer
    for layer in range(problem.device.n_layers):
        layer_slice = slice(layer * npl, (layer + 1) * npl)
        block_mask[layer_slice, layer_slice] = True
    assert np.max(np.abs(result.sigma_pp.lesser[:, ~block_mask])) > 1.0e-8
    assert np.max(np.abs(bonds - currents.left)) > 1.0e-6

    oriented = np.concatenate(([currents.left], bonds, [-currents.right]))
    np.testing.assert_allclose(
        np.diff(oriented),
        spatial.eom_layer_exchange,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        spatial.corrected_cut_currents,
        currents.left,
        rtol=0.0,
        atol=2.0e-14,
    )
    assert abs(np.sum(spatial.eom_layer_exchange) - currents.collision_power) < 2.0e-14
    assert abs(np.sum(spatial.layer_collision_power) - currents.collision_power) < 2.0e-14
    assert np.max(np.abs(spatial.continuity_residual)) < 2.0e-14
