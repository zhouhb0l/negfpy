"""Coupled-q SCBA compared with the independent dense supercell solver.

These are discrete toy-model equivalence tests, not frequency-grid or physical
transport convergence tests. The four-cell case has complex harmonic q blocks
and intercell cubic terms; it cannot reduce to four independent scalar solves.
"""

from __future__ import annotations

import numpy as np
import pytest

# These helpers assemble a full real-space tensor and perform explicit unitary
# transforms. They do not use production q-routing or contraction helpers.
from test_anharmonic_transverse_q_reference import (
    _fourier_vertex,
    _real_vertex,
    _to_q,
    _unitary_basis,
)

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.solver import solve_scba
from negfpy.anharmonic.transverse_q import TransverseCubicVertex
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    SCBAConfig,
)
from negfpy.core.types import Device1D, LeadBlocks


def _supercell_problem(nq: int) -> tuple[CubicSCBAProblem, np.ndarray]:
    onsite = np.kron(np.eye(nq), np.array([[1.4, -0.1], [-0.1, 1.9]]))
    for cell in range(nq):
        neighbor = (cell + 1) % nq
        # Directed intercell coordinate coupling gives imaginary q offdiagonals.
        onsite[2 * cell, 2 * neighbor + 1] -= 0.12
        onsite[2 * neighbor + 1, 2 * cell] -= 0.12
        for coordinate in range(2):
            first = 2 * cell + coordinate
            second = 2 * neighbor + coordinate
            onsite[first, second] -= 0.05
            onsite[second, first] -= 0.05
    lead = LeadBlocks(onsite.copy(), -0.3 * np.eye(2 * nq))
    problem = CubicSCBAProblem(
        device=Device1D([onsite], []),
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=DenseCubicVertex(0.02 * _real_vertex((nq, 1, 1))),
        grid=SignedFrequencyGrid(step=0.4, n_positive=8),
    )
    return problem, _unitary_basis((nq, 1, 1), ndof=2)


@pytest.mark.parametrize("nq", [1, 4])
def test_converged_coupled_q_fields_and_currents_match_dense_supercell(nq: int) -> None:
    from negfpy.anharmonic.transverse_solver import (
        TransverseSCBAConfig,
        TransverseSCBAProblem,
        TransverseSpectrum,
        solve_transverse_scba,
    )

    real_problem, basis = _supercell_problem(nq)
    dense_config = SCBAConfig(
        eta_lead=0.05,
        max_iterations=40,
        min_iterations=3,
        consecutive_passes=2,
        mixing=0.8,
        sigma_rtol=1e-8,
        sigma_atol=1e-11,
        green_rtol=1e-8,
        green_atol=1e-11,
        identity_rtol=1e-8,
        current_rtol=1e-5,
        current_atol=1e-10,
        on_nonconvergence="return",
    )
    reference = solve_scba(real_problem, dense_config)
    assert reference.converged
    assert np.max(np.abs(reference.sigma_pp.lesser)) > 1e-7

    def q_spectrum(spectrum):
        return TransverseSpectrum(
            **{
                component: _to_q(getattr(spectrum, component), basis, nq)
                for component in ("retarded", "lesser", "greater")
            }
        )

    dynamical = _to_q(real_problem.device.onsite_blocks[0][None], basis, nq)[:, 0]
    coefficients = _fourier_vertex(
        (nq, 1, 1),
        real_problem.vertex.values,
        basis,
    )
    if nq == 4:
        assert np.max(np.abs(dynamical.imag)) > 0.1
        assert np.max(np.abs(coefficients.imag)) > 1e-3
    coupled_problem = TransverseSCBAProblem(
        grid=real_problem.grid,
        vertex=TransverseCubicVertex(UniformBZMesh((nq, 1, 1)), coefficients),
        dynamical=dynamical,
        left=q_spectrum(reference.sigma_left),
        right=q_spectrum(reference.sigma_right),
        scale=real_problem.scale,
    )
    coupled = solve_transverse_scba(
        coupled_problem,
        TransverseSCBAConfig(
            max_iterations=40,
            mixing=0.8,
            sigma_rtol=1e-8,
            sigma_atol=1e-11,
            current_rtol=1e-5,
            current_atol=1e-10,
            identity_rtol=1e-8,
            identity_atol=1e-11,
        ),
    )
    assert coupled.converged, coupled.reason
    assert coupled.history[-1].sigma_residual <= 1.0
    for component in ("retarded", "lesser", "greater"):
        np.testing.assert_allclose(
            getattr(coupled.sigma_pp, component),
            _to_q(getattr(reference.sigma_pp, component), basis, nq),
            rtol=2e-6,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            getattr(coupled.green, component),
            _to_q(getattr(reference.green, component), basis, nq),
            rtol=2e-8,
            atol=2e-9,
        )
    # The real-space result covers the entire transverse supercell; the new
    # solver reports the uniform q average, hence current per transverse cell.
    for name in ("left", "right", "collision_power"):
        assert getattr(coupled.currents, name) == pytest.approx(
            getattr(reference.currents, name) / nq,
            rel=2e-8,
            abs=2e-11,
        )
