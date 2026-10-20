"""Fixed-eta transverse contacts versus a full real-space spring supercell."""

from __future__ import annotations

import numpy as np
import pytest
from test_anharmonic_transverse_q_reference import _cells, _to_q, _unitary_basis

from negfpy.anharmonic._harmonic import retarded_contact_self_energy
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
)
from negfpy.core.types import Device1D, LeadBlocks


def _spring_supercell(shape: tuple[int, int, int]):
    """Build positive spring energies before any transverse transformation."""

    cells = _cells(shape)
    lookup = {cell: index for index, cell in enumerate(cells)}
    nq = len(cells)
    d00 = np.zeros((2 * nq, 2 * nq))
    d01 = np.zeros_like(d00)
    coupling = np.zeros_like(d00)

    def neighbor(cell, direction):
        return lookup[
            tuple((cell[axis] + direction[axis]) % shape[axis] for axis in range(3))
        ]

    def within_layer(first, second, stiffness):
        d00[first, first] += stiffness
        d00[second, second] += stiffness
        d00[first, second] -= stiffness
        d00[second, first] -= stiffness

    def between_layers(first, second, stiffness):
        # A spring from this layer to the next contributes once to each
        # endpoint's onsite entry, including the translated incoming bond.
        d00[first, first] += stiffness
        d00[second, second] += stiffness
        d01[first, second] -= stiffness

    for number, cell in enumerate(cells):
        next_x = neighbor(cell, (1, 0, 0))
        within_layer(2 * number, 2 * number + 1, 0.4)
        within_layer(2 * number, 2 * next_x + 1, 0.3)
        between_layers(2 * number, 2 * number, 0.8)
        between_layers(2 * number + 1, 2 * number + 1, 0.6)
        between_layers(2 * number, 2 * next_x + 1, 0.17)
        coupling[2 * number, 2 * number] = -0.8
        coupling[2 * number + 1, 2 * number + 1] = -0.6
        coupling[2 * number, 2 * next_x + 1] -= 0.13
        if shape[1] > 1:
            next_y = neighbor(cell, (0, 1, 0))
            within_layer(2 * number, 2 * next_y, 0.21)
            within_layer(2 * number + 1, 2 * next_y + 1, 0.12)

    # One uncontacted coordinate per device cell tests explicit embedding.
    indices = tuple(3 * number + a for number in range(nq) for a in (0, 2))
    reservoir = HarmonicReservoir(
        LeadBlocks(d00, d01),
        temperature=0.8,
        device_to_lead=coupling,
        device_indices=indices,
    )
    problem = CubicSCBAProblem(
        device=Device1D([np.eye(3 * nq)], []),
        left=reservoir,
        right=reservoir,
        vertex=DenseCubicVertex(np.zeros((3 * nq,) * 3)),
        grid=SignedFrequencyGrid(step=0.3, n_positive=7),
    )

    def transform_source(values, kpar):
        # Read only first-cell source blocks and sum explicit translations.
        # The reference below instead transforms full solved supercell arrays.
        return sum(
            values[:2, 2 * number : 2 * number + 2]
            * np.exp(1j * (kpar[0] * cell[0] + kpar[1] * cell[1]))
            for number, cell in enumerate(cells)
        )

    def reservoirs(kpar):
        return HarmonicReservoir(
            LeadBlocks(
                transform_source(d00, kpar),
                transform_source(d01, kpar),
            ),
            temperature=reservoir.temperature,
            device_to_lead=transform_source(coupling, kpar),
            device_indices=(0, 2),
        )

    return problem, reservoirs, _unitary_basis(shape, ndof=3)


@pytest.mark.parametrize("shape", [(4, 1, 1), (2, 2, 1)])
def test_paired_q_contacts_match_independent_acoustic_supercell(shape) -> None:
    from negfpy.anharmonic.transverse_contacts import build_transverse_contact

    problem, reservoirs, basis = _spring_supercell(shape)
    mesh = UniformBZMesh(shape)
    nq = len(_cells(shape))
    eta = 0.05
    zero = problem.grid.n_positive
    # Full real-space lead has one acoustic mode and additional optical modes.
    # Its established acoustic path includes every channel in the zero limit.
    dense_contacts = build_contact_spectra(problem, eta_lead=eta)
    actual_contacts = []
    for side, reference in zip(("left", "right"), dense_contacts, strict=True):
        actual = build_transverse_contact(
            mesh, problem.grid, reservoirs, 3, eta, side=side
        )
        actual_contacts.append(actual)
        assert [report["classification"] for report in actual.q_reports] == [
            "acoustic",
            *["gapped"] * (nq - 1),
        ]
        for component in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(
                getattr(actual.spectrum, component),
                _to_q(getattr(reference, component), basis, nq),
                rtol=3e-10,
                atol=3e-12,
            )
        expected_limits = _to_q(
            (1j * reference.lesser[zero] / problem.left.temperature)[None],
            basis,
            nq,
        )[:, 0]
        np.testing.assert_allclose(
            actual.zero_limits, expected_limits, rtol=3e-10, atol=3e-12
        )
        np.testing.assert_allclose(
            actual.spectrum.lesser[:, zero],
            -1j * problem.left.temperature * actual.zero_limits,
            rtol=0.0,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            actual.spectrum.greater[:, zero],
            actual.spectrum.lesser[:, zero],
            rtol=0.0,
            atol=1e-14,
        )
        assert np.linalg.norm(actual.zero_limits[1]) > 1e-3
        assert np.count_nonzero(actual.zero_limits[:, 1, :]) == 0
        assert np.count_nonzero(actual.zero_limits[:, :, 1]) == 0

        for number, address in enumerate(mesh.addresses):
            kpar = tuple(2.0 * np.pi * address[axis] / shape[axis] for axis in range(2))
            lookup = actual.at_q(kpar)
            for component in ("retarded", "lesser", "greater"):
                np.testing.assert_array_equal(
                    getattr(lookup, component),
                    getattr(actual.spectrum, component)[number],
                )

        if shape == (4, 1, 1):
            # q=+/-pi/2 carry phases. A same-q real-symmetry projection or
            # same-q frequency reflection would destroy this exact limit.
            assert np.max(np.abs(actual.zero_limits[1].imag)) > 1e-5
            np.testing.assert_allclose(
                actual.zero_limits[3],
                actual.zero_limits[1].T,
                rtol=3e-10,
                atol=3e-12,
            )
            assert (
                np.max(np.abs(actual.zero_limits[1] - actual.zero_limits[1].T)) > 1e-5
            )

            # Independent finite difference of full real-space surface solves
            # checks the zero-limit derivative without its linearized equation.
            h = 1e-5
            sigma_plus = retarded_contact_self_energy(
                h, problem.left, problem.ndof, eta, side=side
            )
            sigma_minus = retarded_contact_self_energy(
                -h, problem.left, problem.ndof, eta, side=side
            )
            derivative = (sigma_plus - sigma_minus) / (2.0 * h)
            finite_limit = 1j * (derivative - derivative.conj().T)
            np.testing.assert_allclose(
                actual.zero_limits,
                _to_q(finite_limit[None], basis, nq)[:, 0],
                rtol=2e-7,
                atol=2e-9,
            )

    # The same explicit coupling and temperature isolate outward orientation.
    assert (
        np.max(
            np.abs(
                actual_contacts[0].spectrum.retarded
                - actual_contacts[1].spectrum.retarded
            )
        )
        > 1e-3
    )
