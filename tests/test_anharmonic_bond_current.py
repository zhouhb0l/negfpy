import numpy as np
import pytest

from negfpy.anharmonic._harmonic import solve_keldysh_green, solve_retarded
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.current import (
    bond_heat_current_spectra,
    bond_heat_currents,
    contact_heat_current,
)
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.toy import fpu_alpha_chain, pinned_scalar_chain
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
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
