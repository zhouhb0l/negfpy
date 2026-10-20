"""Explicit supercell DFT references for finite-device transverse mapping."""

from __future__ import annotations

import numpy as np
import pytest
from test_anharmonic_transverse_q_reference import (
    _cells,
    _fourier_vertex,
    _real_vertex,
    _to_q,
    _unitary_basis,
)

from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.solver import solve_scba
from negfpy.anharmonic.transverse_q import TransverseCubicVertex
from negfpy.anharmonic.transverse_solver import (
    TransverseSCBAConfig,
    solve_transverse_scba,
)
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    SCBAConfig,
)
from negfpy.core.types import Device1D, DeviceKSpace, LeadBlocks


def _full_model(
    shape: tuple[int, int, int],
) -> tuple[CubicSCBAProblem, np.ndarray]:
    """Create the real-space model before performing any reciprocal mapping."""

    cells = _cells(shape)
    lookup = {cell: index for index, cell in enumerate(cells)}
    nq = len(cells)
    onsite = np.kron(np.eye(nq), np.array([[1.4, -0.1], [-0.1, 1.9]]))
    left_coupling = -0.3 * np.eye(2 * nq)
    right_coupling = -0.28 * np.eye(2 * nq)
    for cell_index, cell in enumerate(cells):
        neighbor = list(cell)
        neighbor[0] = (neighbor[0] + 1) % shape[0]
        next_index = lookup[tuple(neighbor)]
        first, second = 2 * cell_index, 2 * next_index + 1
        onsite[first, second] -= 0.12
        onsite[second, first] -= 0.12
        for coordinate in range(2):
            first = 2 * cell_index + coordinate
            second = 2 * next_index + coordinate
            onsite[first, second] -= 0.05
            onsite[second, first] -= 0.05
        # Distinct directed device-to-lead translations expose the phase and
        # adjoint convention at both contacts; neither coupling is Hermitian.
        left_coupling[2 * cell_index, 2 * next_index + 1] -= 0.025
        right_coupling[2 * next_index + 1, 2 * cell_index] -= 0.035
        if shape[1] > 1:
            y_neighbor = list(cell)
            y_neighbor[1] = (y_neighbor[1] + 1) % shape[1]
            y_index = lookup[tuple(y_neighbor)]
            onsite[2 * cell_index, 2 * y_index + 1] -= 0.07
            onsite[2 * y_index + 1, 2 * cell_index] -= 0.07
            left_coupling[2 * cell_index, 2 * y_index] -= 0.015
    lead = LeadBlocks(onsite.copy(), -0.3 * np.eye(2 * nq))
    real_problem = CubicSCBAProblem(
        device=Device1D([onsite], []),
        left=HarmonicReservoir(lead, 0.8, device_to_lead=left_coupling),
        right=HarmonicReservoir(lead, 0.6, device_to_lead=right_coupling),
        vertex=DenseCubicVertex(0.02 * _real_vertex(shape)),
        grid=SignedFrequencyGrid(step=0.4, n_positive=8),
    )
    return real_problem, _unitary_basis(shape, ndof=2)


def _anchor_matrix_blocks(
    values: np.ndarray,
    shape: tuple[int, int, int],
) -> list[tuple[tuple[int, int, int], np.ndarray]]:
    """Extract real-space source terms, not Fourier coefficients."""

    return [
        (cell, values[:2, 2 * index : 2 * index + 2].copy())
        for index, cell in enumerate(_cells(shape))
    ]


def _anchor_cubic_blocks(
    values: np.ndarray,
    shape: tuple[int, int, int],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int], np.ndarray]]:
    cells = _cells(shape)
    return [
        (
            first,
            second,
            values[
                :2,
                2 * first_index : 2 * first_index + 2,
                2 * second_index : 2 * second_index + 2,
            ].copy(),
        )
        for first_index, first in enumerate(cells)
        for second_index, second in enumerate(cells)
    ]


def _map_model(shape: tuple[int, int, int]):
    from negfpy.anharmonic.transverse_mapping import (
        TransverseCubicTerm,
        build_transverse_problem,
        fourier_transverse_cubic,
        gapped_contact_spectrum,
    )

    real_problem, basis = _full_model(shape)
    mesh = UniformBZMesh(shape)
    terms = []
    for first, second, block in _anchor_cubic_blocks(real_problem.vertex.values, shape):
        for indices in np.ndindex(block.shape):
            value = float(block[indices])
            if value == 0.0:
                continue
            # Split each source entry into weighted equivalent images. This
            # tests accumulation and applying source weights exactly once.
            terms.append(
                TransverseCubicTerm(
                    indices=indices,
                    second_cell=first[:2],
                    third_cell=second[:2],
                    coefficient=value,
                    weight=0.25,
                )
            )
            terms.append(
                TransverseCubicTerm(
                    indices=indices,
                    second_cell=(first[0] + shape[0], first[1]),
                    third_cell=(second[0], second[1] - shape[1]),
                    coefficient=value,
                    weight=0.75,
                )
            )
    vertex = fourier_transverse_cubic(mesh, 2, terms)

    def provider(values):
        blocks = _anchor_matrix_blocks(values, shape)

        def at_q(kpar):
            return sum(
                block * np.exp(1j * (kpar[0] * cell[0] + kpar[1] * cell[1]))
                for cell, block in blocks
            )

        return at_q

    dynamical = provider(real_problem.device.onsite_blocks[0])
    lead_onsite = provider(real_problem.left.lead.d00)
    lead_forward = provider(real_problem.left.lead.d01)
    left_coupling = provider(real_problem.left.device_to_lead)
    right_coupling = provider(real_problem.right.device_to_lead)

    def contact(kpar, side):
        coupling = left_coupling(kpar) if side == "left" else right_coupling(kpar)
        temperature = 0.8 if side == "left" else 0.6
        reservoir = HarmonicReservoir(
            LeadBlocks(lead_onsite(kpar), lead_forward(kpar)),
            temperature,
            device_to_lead=coupling,
        )
        return gapped_contact_spectrum(
            real_problem.grid, reservoir, 2, 0.05, side=side
        )

    mapped = build_transverse_problem(
        grid=real_problem.grid,
        vertex=vertex,
        device=DeviceKSpace(lambda kpar: Device1D([dynamical(kpar)], [])),
        left=lambda kpar: contact(kpar, "left"),
        right=lambda kpar: contact(kpar, "right"),
        input_units="solver",
        scale=real_problem.scale,
    )
    return real_problem, basis, mapped


@pytest.mark.parametrize("shape", [(1, 1, 1), (4, 1, 1), (2, 2, 1)])
def test_mapped_inputs_match_independent_full_supercell_dft(shape) -> None:
    real_problem, basis, mapped = _map_model(shape)
    nq = len(_cells(shape))
    np.testing.assert_allclose(
        mapped.dynamical,
        _to_q(real_problem.device.onsite_blocks[0][None], basis, nq)[:, 0],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        mapped.vertex.coefficients,
        _fourier_vertex(shape, real_problem.vertex.values, basis),
        rtol=2e-13,
        atol=2e-13,
    )
    real_contacts = build_contact_spectra(real_problem, eta_lead=0.05)
    for actual, reference in zip((mapped.left, mapped.right), real_contacts, strict=True):
        for component in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(
                getattr(actual, component),
                _to_q(getattr(reference, component), basis, nq),
                rtol=2e-11,
                atol=2e-12,
            )
    if shape == (4, 1, 1):
        assert np.max(np.abs(mapped.dynamical.imag)) > 0.1
        assert np.max(np.abs(mapped.vertex.coefficients.imag)) > 1e-3
        # Both explicit contact couplings carry a nontrivial Fourier phase.
        for reservoir in (real_problem.left, real_problem.right):
            coupling = _to_q(reservoir.device_to_lead[None], basis, nq)
            assert np.max(np.abs(coupling.imag)) > 0.02


def test_mapped_complex_four_q_scba_matches_dense_supercell_fields_and_currents() -> None:
    real_problem, basis, mapped = _map_model((4, 1, 1))
    dense = solve_scba(
        real_problem,
        SCBAConfig(
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
        ),
    )
    result = solve_transverse_scba(
        mapped,
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
    assert dense.converged
    assert result.converged, result.reason
    assert np.max(np.abs(result.sigma_pp.lesser)) > 1e-7
    for component in ("retarded", "lesser", "greater"):
        np.testing.assert_allclose(
            getattr(result.sigma_pp, component),
            _to_q(getattr(dense.sigma_pp, component), basis, 4),
            rtol=2e-6,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            getattr(result.green, component),
            _to_q(getattr(dense.green, component), basis, 4),
            rtol=2e-8,
            atol=2e-9,
        )
    for name in ("left", "right", "collision_power"):
        assert getattr(result.currents, name) == pytest.approx(
            getattr(dense.currents, name) / 4,
            rel=2e-8,
            abs=2e-11,
        )


def test_transverse_sampling_preserves_open_transport_endpoints() -> None:
    from negfpy.anharmonic.transverse_mapping import (
        build_transverse_problem,
        gapped_contact_spectrum,
    )

    mesh = UniformBZMesh((4, 1, 1))
    grid = SignedFrequencyGrid(0.4, 2)
    vertex = TransverseCubicVertex(mesh, np.zeros((4, 4, 3, 3, 3)))
    device = DeviceKSpace(
        lambda kpar: Device1D(
            [np.array([[2.0]]), np.array([[2.1]]), np.array([[2.2]])],
            [np.array([[-0.2]]), np.array([[-0.3]])],
        )
    )
    lead = LeadBlocks(np.array([[2.0]]), np.array([[-0.3]]))

    def contact(side):
        reservoir = HarmonicReservoir(
            lead,
            0.8 if side == "left" else 0.6,
            device_to_lead=np.array([[-0.3]]),
            device_indices=(0,) if side == "left" else (2,),
        )
        return gapped_contact_spectrum(grid, reservoir, 3, 0.05, side=side)

    mapped = build_transverse_problem(
        grid=grid,
        vertex=vertex,
        device=device,
        left=lambda kpar: contact("left"),
        right=lambda kpar: contact("right"),
        input_units="solver",
    )
    expected = np.array([[2.0, -0.2, 0.0], [-0.2, 2.1, -0.3], [0.0, -0.3, 2.2]])
    np.testing.assert_array_equal(mapped.dynamical, np.repeat(expected[None], 4, axis=0))
    np.testing.assert_array_equal(mapped.dynamical[:, 0, 2], 0.0)
    np.testing.assert_array_equal(mapped.left.retarded[:, :, 1:, :], 0.0)
    np.testing.assert_array_equal(mapped.right.retarded[:, :, :2, :], 0.0)
