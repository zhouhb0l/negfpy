from __future__ import annotations

import numpy as np

from negfpy.anharmonic import SignedFrequencyGrid
from negfpy.anharmonic.toy import wang_2006_three_atom_fixture


def test_wang_2006_harmonic_blocks_match_primary_sources() -> None:
    fixture = wang_2006_three_atom_fixture()

    np.testing.assert_allclose(fixture.left_lead.d00, [[3.13]])
    np.testing.assert_allclose(fixture.left_lead.d01, [[-1.56]])
    np.testing.assert_allclose(fixture.right_lead.d00, [[2.90]])
    np.testing.assert_allclose(fixture.right_lead.d01, [[-1.44]])
    np.testing.assert_allclose(
        [block[0, 0] for block in fixture.device.onsite_blocks],
        [2.94, 2.76, 2.82],
    )
    np.testing.assert_allclose(
        [block[0, 0] for block in fixture.device.coupling_blocks],
        [-1.38, -1.38],
    )


def test_wang_2006_vertex_maps_exactly_to_paper_hamiltonian() -> None:
    fixture = wang_2006_three_atom_fixture()
    factor = fixture.vertex.metadata["paper_vertex_to_solver_factor"]
    paper_vertex = fixture.vertex.values / factor
    displacement = np.array([0.12, -0.07, 0.03])

    h3_from_vertex = np.einsum(
        "abc,a,b,c->",
        paper_vertex,
        displacement,
        displacement,
        displacement,
    ) / 6.0
    h3_from_paper = (1.8 / 3.0) * (
        (displacement[0] - displacement[1]) ** 3
        + (displacement[1] - displacement[2]) ** 3
    )
    np.testing.assert_allclose(h3_from_vertex, h3_from_paper, rtol=1e-14)


def test_wang_2006_problem_converts_kelvin_and_contact_couplings() -> None:
    fixture = wang_2006_three_atom_fixture()
    problem = fixture.problem(
        SignedFrequencyGrid(step=0.05, n_positive=80),
        left_temperature_kelvin=300.0,
        right_temperature_kelvin=280.0,
    )

    expected_left_temperature = (
        300.0 * 1.380649e-23 / (fixture.scale.hbar * fixture.scale.omega0)
    )
    expected_right_temperature = (
        280.0 * 1.380649e-23 / (fixture.scale.hbar * fixture.scale.omega0)
    )
    assert problem.left.temperature == expected_left_temperature
    assert problem.right.temperature == expected_right_temperature
    np.testing.assert_allclose(problem.left.device_to_lead, [[-1.56]])
    np.testing.assert_allclose(problem.right.device_to_lead, [[-1.44]])
    np.testing.assert_array_equal(problem.left.device_indices, [0])
    np.testing.assert_array_equal(problem.right.device_indices, [2])
