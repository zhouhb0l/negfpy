import numpy as np

from negfpy.modeling import IFCData, IFCTerm
from negfpy.workflows.ifc_bulk import _unwrap_transport_n1_minimum_image


def _ifc(
    *,
    repeats: tuple[int, int, int],
    positions: np.ndarray,
    translation: tuple[int, int, int],
    block: np.ndarray,
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
        lattice_vectors=np.eye(3),
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
