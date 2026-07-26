from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np

from negfpy.io import read_ifc
from negfpy.io.readers.phonopy import _shortest_lattice_translations


def _adaptive_shell_oracle(
    cell_delta: np.ndarray,
    basis_delta: np.ndarray,
    lattice: np.ndarray,
    repeats: np.ndarray,
    *,
    tolerance: float = 1.0e-10,
) -> tuple[tuple[int, int, int], ...]:
    """Independent converged shell oracle for a three-dimensional lattice.

    This deliberately does not use the reader's reciprocal-coordinate box.
    A singular-value lower bound proves when every unvisited shell is farther
    away than the current incumbent.
    """

    cell_delta = np.asarray(cell_delta, dtype=np.int64)
    basis_delta = np.asarray(basis_delta, dtype=np.float64)
    lattice = np.asarray(lattice, dtype=np.float64)
    repeats = np.asarray(repeats, dtype=np.int64)
    offset = cell_delta.astype(np.float64) + basis_delta
    repeated_lattice = np.diag(repeats.astype(np.float64)) @ lattice
    # Seed the independent shell proof with Babai nearest-plane rounding.
    # The production reader instead bounds coordinates using inv(lattice).
    q_matrix, r_matrix = np.linalg.qr(repeated_lattice.T)
    transformed_target = q_matrix.T @ (-(offset @ lattice))
    center_float = np.zeros(3, dtype=np.float64)
    for axis in range(2, -1, -1):
        remainder = transformed_target[axis] - np.dot(
            r_matrix[axis, axis + 1 :],
            center_float[axis + 1 :],
        )
        center_float[axis] = np.rint(remainder / r_matrix[axis, axis])
    center = center_float.astype(np.int64)
    residual = (offset + repeats * center) @ lattice
    smallest_singular = float(
        np.linalg.svd(repeated_lattice, compute_uv=False)[-1]
    )
    assert smallest_singular > 0.0

    best = np.inf
    winners: list[tuple[int, int, int]] = []
    for radius in range(1000):
        for shell_offset in product(range(-radius, radius + 1), repeat=3):
            if radius and max(abs(value) for value in shell_offset) != radius:
                continue
            image = center + np.asarray(shell_offset, dtype=np.int64)
            translation = cell_delta + repeats * image
            cartesian = (
                translation.astype(np.float64) + basis_delta
            ) @ lattice
            distance2 = float(np.dot(cartesian, cartesian))
            key = tuple(int(value) for value in translation)
            if distance2 < best - tolerance:
                best = distance2
                winners = [key]
            elif abs(distance2 - best) <= tolerance:
                winners.append(key)

        # Every unvisited offset m has ||m||_2 >= radius + 1. The reverse
        # triangle inequality and sigma_min bound its Cartesian distance.
        lower_distance = max(
            0.0,
            smallest_singular * float(radius + 1) - np.linalg.norm(residual),
        )
        if lower_distance * lower_distance > best + tolerance:
            return tuple(sorted(set(winners)))
    raise AssertionError("Independent shortest-vector oracle did not converge.")


def _write_pair_fixture(
    tmp_path: Path,
    *,
    lattice: np.ndarray,
    positions: np.ndarray,
    pair_value: float,
) -> tuple[Path, Path]:
    poscar = tmp_path / "SPOSCAR"
    force_constants = tmp_path / "FORCE_CONSTANTS"
    poscar.write_text(
        "\n".join(
            [
                "shortest-vector fixture",
                "1.0",
                *[
                    " ".join(f"{value:.17g}" for value in row)
                    for row in lattice
                ],
                "Si",
                "2",
                "Direct",
                *[
                    " ".join(f"{value:.17g}" for value in row)
                    for row in positions
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    force = np.zeros((2, 2, 3, 3), dtype=np.float64)
    force[0, 1, 0, 0] = pair_value
    lines = ["2 2"]
    for atom_i in range(2):
        for atom_j in range(2):
            lines.append(f"{atom_i + 1} {atom_j + 1}")
            lines.extend(
                " ".join(f"{value:.17g}" for value in row)
                for row in force[atom_i, atom_j]
            )
    force_constants.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return poscar, force_constants


def test_skew_cell_shortest_image_outside_fixed_radius_two(
    tmp_path: Path,
) -> None:
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [6.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.49, 0.0]])
    expected = _adaptive_shell_oracle(
        np.zeros(3, dtype=np.int64),
        positions[1] - positions[0],
        lattice,
        np.ones(3, dtype=np.int64),
    )
    assert expected == ((-3, 0, 0),)
    assert _shortest_lattice_translations(
        np.zeros(3, dtype=np.int64),
        positions[1] - positions[0],
        lattice,
        np.ones(3, dtype=np.int64),
    ) == expected
    assert _shortest_lattice_translations(
        np.zeros(3, dtype=np.int64),
        positions[1] - positions[0],
        lattice,
        np.ones(3, dtype=np.int64),
        search_radius=2,
    ) == expected

    poscar, force_constants = _write_pair_fixture(
        tmp_path,
        lattice=lattice,
        positions=positions,
        pair_value=6.0,
    )
    ifc = read_ifc(
        {
            "force_constants_path": force_constants,
            "poscar_path": poscar,
            "supercell": (1, 1, 1),
        },
        reader="phonopy",
    )
    translations = {(term.dx, term.dy, term.dz) for term in ifc.terms}
    assert (-3, 0, 0) in translations
    assert ifc.metadata["shortest_vector_search"] == (
        "reciprocal_bound_exact_v1"
    )


def test_exact_search_matches_independent_oracle_for_random_skew_cells() -> None:
    rng = np.random.default_rng(20260726)
    for _ in range(20):
        lattice = np.array(
            [
                [rng.uniform(0.6, 1.8), 0.0, 0.0],
                [rng.uniform(-2.0, 2.0), rng.uniform(0.6, 1.8), 0.0],
                [
                    rng.uniform(-2.0, 2.0),
                    rng.uniform(-2.0, 2.0),
                    rng.uniform(0.6, 1.8),
                ],
            ]
        )
        cell_delta = rng.integers(-2, 3, size=3, dtype=np.int64)
        basis_delta = rng.uniform(-0.95, 0.95, size=3)
        repeats = rng.integers(1, 4, size=3, dtype=np.int64)
        repeats[0] = 1  # Exercise a collapsed axis in every realization.
        expected = _adaptive_shell_oracle(
            cell_delta,
            basis_delta,
            lattice,
            repeats,
        )
        actual = _shortest_lattice_translations(
            cell_delta,
            basis_delta,
            lattice,
            repeats,
            distance_tolerance=1.0e-10,
        )
        assert actual == expected


def test_reader_distributes_degenerate_shortest_vector_weight_exactly(
    tmp_path: Path,
) -> None:
    lattice = np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    poscar, force_constants = _write_pair_fixture(
        tmp_path,
        lattice=lattice,
        positions=positions,
        pair_value=8.0,
    )
    ifc = read_ifc(
        {
            "force_constants_path": force_constants,
            "poscar_path": poscar,
            "supercell": (1, 1, 1),
        },
        reader="phonopy",
    )
    terms = {
        (term.dx, term.dy, term.dz): np.asarray(term.block)
        for term in ifc.terms
    }
    scale = float(ifc.metadata["force_constant_scale_to_qe"])
    left = float(terms[(-1, 0, 0)][0, 3].real)
    home = float(terms[(0, 0, 0)][0, 3].real)

    # The pair lies exactly halfway between two periodic images. Each image
    # must receive one half of the source block, and the complete source
    # weight must be conserved across the two generated IFC terms.
    np.testing.assert_allclose(left, 4.0 * scale, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(home, 4.0 * scale, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        left + home,
        8.0 * scale,
        rtol=0.0,
        atol=1.0e-15,
    )
    multiplicities = ifc.metadata["shortest_vector_multiplicity_counts"]
    assert int(multiplicities[2]) >= 1
