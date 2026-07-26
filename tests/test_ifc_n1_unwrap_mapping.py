import numpy as np

from negfpy.modeling import IFCData, IFCTerm
from negfpy.workflows.ifc_bulk import _unwrap_transport_n1_minimum_image


def _ifc(
    *,
    repeats: tuple[int, int, int],
    positions: np.ndarray,
    translation: tuple[int, int, int],
    block: np.ndarray,
    lattice_vectors: np.ndarray | None = None,
) -> IFCData:
    nat = int(np.asarray(positions).shape[0])
    return IFCData(
        masses=np.ones(nat),
        dof_per_atom=1,
        terms=(
            IFCTerm(
                dx=translation[0],
                dy=translation[1],
                dz=translation[2],
                block=np.asarray(block, dtype=float),
            ),
        ),
        metadata={"nr": repeats},
        lattice_vectors=(
            np.eye(3)
            if lattice_vectors is None
            else np.asarray(lattice_vectors, dtype=float)
        ),
        atom_positions=np.asarray(positions, dtype=float),
    )


def _terms(ifc: IFCData) -> dict[tuple[int, int, int], np.ndarray]:
    return {
        (int(term.dx), int(term.dy), int(term.dz)): np.asarray(term.block)
        for term in ifc.terms
    }


def test_nonzero_transport_term_still_maps_repeat_one_transverse_axis() -> None:
    source = _ifc(
        repeats=(2, 1, 2),
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 0.8, 0.0]]),
        translation=(1, 0, 0),
        block=np.array([[0.0, 4.0], [0.0, 0.0]]),
    )

    mapped, info = _unwrap_transport_n1_minimum_image(
        source, mode="minimum_image"
    )
    terms = _terms(mapped)

    assert info["applied"] is True
    assert (1, -1, 0) in terms
    np.testing.assert_array_equal(
        terms[(1, -1, 0)], np.array([[0.0, 4.0], [0.0, 0.0]])
    )
    assert (1, 0, 0) not in terms


def test_noncollapsed_arbitrary_integer_components_are_preserved() -> None:
    source = _ifc(
        repeats=(1, 2, 2),
        positions=np.array([[0.0, 0.0, 0.0]]),
        translation=(0, 2, -3),
        block=np.array([[5.0]]),
    )

    mapped, _ = _unwrap_transport_n1_minimum_image(
        source, mode="minimum_image"
    )

    np.testing.assert_array_equal(
        _terms(mapped)[(0, 2, -3)], np.array([[5.0]])
    )


def test_degenerate_shortest_images_split_weight_without_symmetrizing() -> None:
    source = _ifc(
        repeats=(1, 2, 2),
        positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        translation=(0, 0, 0),
        block=np.array([[0.0, 8.0], [0.0, 0.0]]),
    )

    mapped, info = _unwrap_transport_n1_minimum_image(
        source, mode="minimum_image"
    )
    terms = _terms(mapped)

    assert info["ambiguous_pair_count"] == 1
    assert info["degenerate_images_weight_split"] is True
    assert set(terms) == {(-1, 0, 0), (0, 0, 0)}
    for key in ((-1, 0, 0), (0, 0, 0)):
        np.testing.assert_array_equal(
            terms[key], np.array([[0.0, 4.0], [0.0, 0.0]])
        )
    np.testing.assert_array_equal(
        terms[(-1, 0, 0)] + terms[(0, 0, 0)],
        np.array([[0.0, 8.0], [0.0, 0.0]]),
    )


def _adaptive_shell_oracle(
    *,
    fixed_translation: tuple[int, int, int],
    collapsed_axis: int,
    lattice_vectors: np.ndarray,
    maximum_shell: int = 12,
) -> tuple[int, float, int]:
    """Independent direct-shell oracle for the one-axis skew fixture.

    Squared distance is a convex quadratic in this sole integer coordinate.
    """

    best_shift = 0
    best_distance = float("inf")
    stable_shells = 0
    for shell in range(maximum_shell + 1):
        shell_improved = False
        values = (0,) if shell == 0 else (-shell, shell)
        for value in values:
            shift = np.asarray(fixed_translation, dtype=float)
            shift[collapsed_axis] = float(value)
            distance = float(np.linalg.norm(shift @ lattice_vectors))
            if distance < best_distance - 1e-14:
                best_shift = int(value)
                best_distance = distance
                shell_improved = True
        if shell_improved:
            stable_shells = 0
        else:
            stable_shells += 1
        if shell > abs(best_shift) and stable_shells >= 2:
            return best_shift, best_distance, shell
    raise AssertionError("adaptive-shell oracle did not stabilize")


def test_skew_repeat_one_search_finds_image_outside_fixed_shell() -> None:
    lattice_vectors = np.array(
        [
            [3.2, 0.1, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    oracle_shift, oracle_distance, terminal_shell = _adaptive_shell_oracle(
        fixed_translation=(1, 0, 0),
        collapsed_axis=1,
        lattice_vectors=lattice_vectors,
    )
    assert oracle_shift == -3
    assert abs(oracle_shift) > 1
    assert terminal_shell >= 5

    source = _ifc(
        repeats=(2, 1, 2),
        positions=np.array([[0.0, 0.0, 0.0]]),
        translation=(1, 0, 0),
        block=np.array([[7.0]]),
        lattice_vectors=lattice_vectors,
    )
    mapped, info = _unwrap_transport_n1_minimum_image(
        source, mode="minimum_image"
    )
    terms = _terms(mapped)

    assert info["minimum_image_search"] == "collapsed_subspace_dual_bound_v1"
    assert (
        mapped.metadata["n1_transport_unwrap"]["minimum_image_search"]
        == "collapsed_subspace_dual_bound_v1"
    )
    assert set(terms) == {(1, oracle_shift, 0)}
    np.testing.assert_array_equal(
        terms[(1, oracle_shift, 0)], np.array([[7.0]])
    )
    distance = np.linalg.norm(
        np.asarray((1, oracle_shift, 0), dtype=float) @ lattice_vectors
    )
    np.testing.assert_allclose(distance, oracle_distance, rtol=0.0, atol=1e-15)
