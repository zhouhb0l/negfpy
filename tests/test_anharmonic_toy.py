from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from negfpy.anharmonic.toy import (
    fpu_alpha_chain,
    pinned_scalar_chain,
    pinned_three_polarization_chain,
)
from negfpy.anharmonic.types import DenseCubicVertex
from negfpy.core.types import Device1D, LeadBlocks


def _cubic_energy(vertex: DenseCubicVertex, displacement: np.ndarray) -> float:
    return float(
        np.einsum(
            "abc,a,b,c->",
            vertex.values,
            displacement,
            displacement,
            displacement,
        )
        / 6.0
    )


def test_pinned_scalar_chain_builds_mass_weighted_onsite_vertex() -> None:
    device, lead, vertex = pinned_scalar_chain(
        3,
        mass=4.0,
        spring=2.0,
        pinning=1.0,
        cubic=8.0,
    )

    assert isinstance(device, Device1D)
    assert isinstance(lead, LeadBlocks)
    assert isinstance(vertex, DenseCubicVertex)
    assert device.n_layers == 3
    np.testing.assert_allclose(lead.d00, [[1.25]])
    np.testing.assert_allclose(lead.d01, [[-0.5]])
    np.testing.assert_allclose(device.onsite_blocks[0], lead.d00)
    np.testing.assert_allclose(device.coupling_blocks[0], lead.d01)

    expected = np.zeros((3, 3, 3))
    expected[np.arange(3), np.arange(3), np.arange(3)] = 1.0
    np.testing.assert_allclose(vertex.values, expected)
    assert vertex.metadata["interaction_topology"] == "onsite"
    assert vertex.metadata["harmonic_bond_current_supported"] is True
    assert vertex.metadata["mass_weighted_cubic"] == 1.0


def test_pinned_scalar_cubic_energy_matches_coordinate_potential() -> None:
    _, _, vertex = pinned_scalar_chain(2, mass=4.0, cubic=12.0)
    u = np.array([0.3, -0.7])
    x = u / np.sqrt(4.0)

    expected = 12.0 * np.sum(x**3) / 6.0
    assert _cubic_energy(vertex, u) == pytest.approx(expected)


def test_three_polarization_chain_accepts_component_parameters() -> None:
    masses = np.array([1.0, 2.0, 4.0])
    springs = np.array([1.0, 4.0, 8.0])
    pinnings = np.array([0.5, 1.0, 2.0])
    device, lead, vertex = pinned_three_polarization_chain(
        2,
        mass=masses,
        spring=springs,
        pinning=pinnings,
        g=0.3,
    )

    np.testing.assert_allclose(
        np.diag(lead.d00),
        (2.0 * springs + pinnings) / masses,
    )
    np.testing.assert_allclose(np.diag(lead.d01), -springs / masses)
    assert device.dof_per_layer == 3
    assert vertex.ndof == 6
    assert np.count_nonzero(vertex.values) == 12
    for offset in (0, 3):
        for local in permutations((0, 1, 2)):
            indices = tuple(offset + index for index in local)
            assert vertex.values[indices] == pytest.approx(0.3)
    assert vertex.metadata["dof_order"] == ("x", "y", "z")
    assert vertex.metadata["interaction_topology"] == "intralayer"
    assert vertex.metadata["harmonic_bond_current_supported"] is True


def test_three_polarization_vertex_is_exact_g_ux_uy_uz() -> None:
    _, _, vertex = pinned_three_polarization_chain(1, g=-0.7)
    u = np.array([0.2, -0.4, 0.8])

    assert _cubic_energy(vertex, u) == pytest.approx(-0.7 * np.prod(u))


def test_fpu_alpha_expansion_matches_coordinate_bond_potential() -> None:
    alpha = 8.0
    mass = 4.0
    _, _, vertex = fpu_alpha_chain(
        2,
        mass=mass,
        alpha=alpha,
    )

    # alpha / mass**(3/2) == 1 for this fixture.
    assert vertex.values[0, 0, 0] == pytest.approx(-1.0)
    assert vertex.values[1, 1, 1] == pytest.approx(1.0)
    for indices in set(permutations((0, 0, 1))):
        assert vertex.values[indices] == pytest.approx(1.0)
    for indices in set(permutations((0, 1, 1))):
        assert vertex.values[indices] == pytest.approx(-1.0)

    u = np.array([0.25, -0.6])
    x = u / np.sqrt(mass)
    expected = alpha * (x[1] - x[0]) ** 3 / 6.0
    assert _cubic_energy(vertex, u) == pytest.approx(expected)
    assert vertex.metadata["interaction_bonds"] == ((0, 1),)
    assert vertex.metadata["interaction_topology"] == "bond"
    assert vertex.metadata["harmonic_bond_current_supported"] is False


def test_fpu_alpha_selected_overlapping_bonds_accumulate() -> None:
    alpha = 0.4
    _, _, vertex = fpu_alpha_chain(
        4,
        alpha=alpha,
        interaction_bonds=[(0, 1), (1, 2)],
    )
    u = np.array([0.1, 0.3, -0.2, 2.0])

    expected = alpha * ((u[1] - u[0]) ** 3 + (u[2] - u[1]) ** 3) / 6.0
    assert _cubic_energy(vertex, u) == pytest.approx(expected)
    assert not np.any(vertex.values[3])
    assert vertex.metadata["interaction_bonds"] == ((0, 1), (1, 2))


def test_fpu_alpha_empty_selection_is_valid_zero_vertex() -> None:
    _, _, vertex = fpu_alpha_chain(2, alpha=1.0, interaction_bonds=[])
    assert not np.any(vertex.values)
    assert vertex.metadata["interaction_bonds"] == ()


@pytest.mark.parametrize(
    ("builder", "kwargs", "match"),
    [
        (pinned_scalar_chain, {"n_layers": 0}, "at least one"),
        (pinned_scalar_chain, {"n_layers": 2, "mass": 0.0}, "strictly positive"),
        (pinned_scalar_chain, {"n_layers": 2, "spring": -1.0}, "strictly positive"),
        (pinned_scalar_chain, {"n_layers": 2, "pinning": 0.0}, "strictly positive"),
        (
            pinned_three_polarization_chain,
            {"n_layers": 2, "mass": [1.0, 2.0]},
            "scalar or a length-three",
        ),
        (
            fpu_alpha_chain,
            {"n_layers": 1},
            "at least two",
        ),
        (
            fpu_alpha_chain,
            {"n_layers": 3, "interaction_bonds": [(0, 2)]},
            "nearest-neighbor",
        ),
        (
            fpu_alpha_chain,
            {"n_layers": 3, "interaction_bonds": [(0, 1), (0, 1)]},
            "duplicate",
        ),
        (
            fpu_alpha_chain,
            {"n_layers": 3, "interaction_bonds": [(-1, 0)]},
            "outside",
        ),
    ],
)
def test_toy_builders_reject_invalid_parameters(
    builder: object,
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        builder(**kwargs)  # type: ignore[operator]
