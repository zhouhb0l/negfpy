from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import pytest

from negfpy.anharmonic.momentum import (
    GridAddress,
    MomentumTriplet,
    UniformBZMesh,
    enumerate_target_triplets,
    minimum_bz_reciprocal_vector,
    niggli_reduced_reciprocal_lattice,
    shortest_bz_address_images,
    triplet_is_normal_on_bz_surface,
)

_HARD_SKEW_LATTICE = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.92, 0.31, 0.0],
        [-0.44, 0.27, 0.73],
    ]
)
# Frozen from phonopy 4.4.0's spglib Niggli reduction at eps=1e-5.
_HARD_SKEW_REDUCED_RECIPROCAL = np.array(
    [
        [-1.0, 1.0, 1.0],
        [-0.25806451612903203, 0.25806451612903203, -2.9677419354838714],
        [-0.5072912063632347, -0.8625718073353954, 0.33053468846663736],
    ]
)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (2,),
        (2, 3),
        (2, 3, 4, 5),
        (0, 2, 3),
        (-1, 2, 3),
        (True, 2, 3),
        (2.0, 2, 3),
    ],
)
def test_mesh_rejects_nonpositive_or_noninteger_three_vectors(shape) -> None:
    with pytest.raises((TypeError, ValueError)):
        UniformBZMesh(shape)


@pytest.mark.parametrize(
    "components",
    [(), (1,), (1, 2), (1, 2, 3, 4), (True, 0, 0), (1.0, 0, 0)],
)
def test_grid_address_requires_exactly_three_integers(components) -> None:
    with pytest.raises((TypeError, ValueError)):
        GridAddress(components)


def test_mesh_and_address_are_detached_immutable_value_types() -> None:
    shape = np.array([2, 3, 4], dtype=np.int64)
    components = np.array([1, -1, 2], dtype=np.int64)
    mesh = UniformBZMesh(shape)
    address = GridAddress(components)
    shape[:] = 9
    components[:] = 9

    assert mesh.shape == (2, 3, 4)
    assert mesh.mesh == (2, 3, 4)
    assert address.components == (1, -1, 2)
    assert tuple(address) == (1, -1, 2)
    with pytest.raises(FrozenInstanceError):
        mesh.shape = (1, 1, 1)
    with pytest.raises(FrozenInstanceError):
        address.components = (0, 0, 0)


def test_nq_one_mesh_has_one_normal_self_triplet() -> None:
    mesh = UniformBZMesh((1, 1, 1))

    assert mesh.n_qpoints == 1
    assert mesh.size == 1
    assert mesh.address_from_index(0) == GridAddress((0, 0, 0))
    assert mesh.index_from_address((19, -7, 3)) == 0
    assert mesh.negative_index(0) == 0
    assert mesh.normalization == Fraction(1, 1)

    triplets = mesh.triplets_at_q(0)
    assert len(triplets) == 1
    triplet = triplets[0]
    assert triplet.indices == (0, 0, 0)
    assert triplet.addresses == (
        GridAddress((0, 0, 0)),
        GridAddress((0, 0, 0)),
        GridAddress((0, 0, 0)),
    )
    assert triplet.G == (0, 0, 0)
    assert triplet.process == "normal"
    assert triplet.is_normal
    assert not triplet.is_umklapp
    assert triplet.weight == Fraction(1, 1)


def test_canonical_addresses_use_x_fast_phono3py_index_order() -> None:
    mesh = UniformBZMesh((2, 3, 4))

    expected_prefix = (
        (0, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (-1, 1, 0),
        (0, -1, 0),
        (-1, -1, 0),
        (0, 0, 1),
        (-1, 0, 1),
    )
    assert tuple(
        mesh.address_from_index(index).components
        for index in range(len(expected_prefix))
    ) == expected_prefix
    assert mesh.address_from_index(12) == GridAddress((0, 0, -2))
    assert mesh.address_from_index(18) == GridAddress((0, 0, -1))


def test_odd_and_even_meshes_lock_centered_boundary_convention() -> None:
    odd = UniformBZMesh((5, 1, 1))
    even = UniformBZMesh((4, 1, 1))

    assert tuple(address.x for address in odd.addresses) == (0, 1, 2, -2, -1)
    assert tuple(address.x for address in even.addresses) == (0, 1, -2, -1)
    assert odd.canonical_address((7, 0, 0)) == GridAddress((2, 0, 0))
    assert even.canonical_address((2, 0, 0)) == GridAddress((-2, 0, 0))
    assert even.canonical_address((-2, 0, 0)) == GridAddress((-2, 0, 0))


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 1), (2, 3, 4), (3, 5, 2), (4, 4, 4)],
)
def test_address_index_mapping_is_a_full_mesh_bijection(shape) -> None:
    mesh = UniformBZMesh(shape)
    addresses = mesh.addresses

    assert len(addresses) == mesh.n_qpoints
    assert len(set(addresses)) == mesh.n_qpoints
    assert tuple(mesh.index_from_address(address) for address in addresses) == tuple(
        range(mesh.n_qpoints)
    )
    assert tuple(
        mesh.address_from_index(mesh.index_from_address(address))
        for address in addresses
    ) == addresses
    assert tuple(mesh.iter_addresses()) == addresses


def test_index_mapping_canonicalizes_translationally_equivalent_addresses() -> None:
    mesh = UniformBZMesh((3, 4, 5))
    canonical = GridAddress((1, -2, 2))

    assert mesh.is_canonical_address(canonical)
    assert mesh.index_from_address(canonical) == mesh.index_from_address(
        (1 + 6, -2 - 8, 2 + 15)
    )
    assert mesh.canonical_address((7, -10, 17)) == canonical
    with pytest.raises(ValueError, match="canonical"):
        mesh.require_canonical_address((7, -10, 17))


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 5, 7), (4, 1, 6)])
def test_negative_partner_is_exact_and_involutive(shape) -> None:
    mesh = UniformBZMesh(shape)

    for index in range(mesh.n_qpoints):
        address = mesh.address_from_index(index)
        partner_index = mesh.negative_index(index)
        partner = mesh.address_from_index(partner_index)
        assert partner == mesh.negative_address(address)
        assert mesh.negative_index(partner_index) == index
        for component, negative, period in zip(
            address,
            partner,
            mesh.shape,
            strict=True,
        ):
            assert (component + negative) % period == 0


def test_fractional_q_coordinates_are_exact_rationals() -> None:
    mesh = UniformBZMesh((3, 4, 5))

    assert mesh.fractional_q((1, -2, 2)) == (
        Fraction(1, 3),
        Fraction(-1, 2),
        Fraction(2, 5),
    )


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 3, 1), (2, 3, 4)])
def test_every_target_enumerates_one_triplet_per_full_mesh_q1(shape) -> None:
    mesh = UniformBZMesh(shape)

    for target_index in range(mesh.n_qpoints):
        triplets = enumerate_target_triplets(mesh, target_index)
        assert len(triplets) == mesh.n_qpoints
        assert tuple(triplet.q0_index for triplet in triplets) == (
            target_index,
        ) * mesh.n_qpoints
        assert tuple(triplet.q1_index for triplet in triplets) == tuple(
            range(mesh.n_qpoints)
        )
        assert len({triplet.q2_index for triplet in triplets}) == mesh.n_qpoints
        assert all(triplet.multiplicity == 1 for triplet in triplets)


@pytest.mark.parametrize("shape", [(3, 3, 1), (2, 3, 4), (4, 4, 2)])
def test_triplet_conservation_and_classification_use_integer_arithmetic(shape) -> None:
    mesh = UniformBZMesh(shape)

    for target_index in range(mesh.n_qpoints):
        for triplet in mesh.triplets_at_q(target_index):
            total = tuple(
                sum(address[axis] for address in triplet.addresses)
                for axis in range(3)
            )
            assert total == tuple(
                triplet.G[axis] * mesh.shape[axis] for axis in range(3)
            )
            assert triplet.q2_index == mesh.index_from_address(
                tuple(
                    -triplet.q0_address[axis] - triplet.q1_address[axis]
                    for axis in range(3)
                )
            )
            assert triplet.is_normal == (triplet.G == (0, 0, 0))
            assert triplet.is_umklapp == (triplet.G != (0, 0, 0))
            assert triplet.process == (
                "normal" if triplet.G == (0, 0, 0) else "umklapp"
            )


def test_known_odd_and_even_normal_umklapp_routes() -> None:
    odd = UniformBZMesh((3, 1, 1))
    odd_triplets = odd.triplets_at_q(1)
    assert tuple(triplet.G for triplet in odd_triplets) == (
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 0),
    )
    assert tuple(triplet.process for triplet in odd_triplets) == (
        "normal",
        "umklapp",
        "normal",
    )

    even = UniformBZMesh((4, 1, 1))
    boundary_route = even.triplets_at_q(2)[2]
    assert boundary_route.addresses == (
        GridAddress((-2, 0, 0)),
        GridAddress((-2, 0, 0)),
        GridAddress((0, 0, 0)),
    )
    assert boundary_route.G == (-1, 0, 0)
    assert boundary_route.process == "umklapp"


def test_triplet_order_and_results_are_deterministic() -> None:
    mesh = UniformBZMesh((3, 4, 2))

    first = enumerate_target_triplets(mesh, 7)
    second = enumerate_target_triplets(mesh, 7)

    assert first == second
    assert isinstance(first, tuple)
    assert tuple(triplet.q1_index for triplet in first) == tuple(
        range(mesh.n_qpoints)
    )


def test_target_can_be_given_as_a_canonical_address() -> None:
    mesh = UniformBZMesh((3, 4, 2))
    target = mesh.address_from_index(7)

    assert mesh.triplets_at_q(target) == mesh.triplets_at_q(7)


def test_full_mesh_normalization_is_exact_and_sums_to_one() -> None:
    mesh = UniformBZMesh((2, 3, 4))
    triplets = mesh.triplets_at_q(9)

    assert mesh.normalization == Fraction(1, 24)
    assert mesh.normalization_float == pytest.approx(1.0 / 24.0)
    assert all(triplet.weight == Fraction(1, 24) for triplet in triplets)
    assert sum((triplet.weight for triplet in triplets), Fraction()) == 1


def test_momentum_triplet_rejects_inconsistent_reciprocal_vector() -> None:
    mesh = UniformBZMesh((3, 1, 1))

    with pytest.raises(ValueError, match="reciprocal_vector"):
        MomentumTriplet(
            mesh=mesh,
            q0_index=1,
            q1_index=1,
            q2_index=1,
            reciprocal_vector=(0, 0, 0),
        )


def test_bz_surface_images_retain_even_mesh_boundary_duplicates() -> None:
    mesh = UniformBZMesh((2, 1, 1))
    images = shortest_bz_address_images(
        mesh,
        np.eye(3),
        reduced_reciprocal_lattice=np.eye(3),
    )

    assert images[0] == (GridAddress((0, 0, 0)),)
    assert images[1] == (
        GridAddress((-1, 0, 0)),
        GridAddress((1, 0, 0)),
    )
    boundary_route = mesh.triplets_at_q(0)[1]
    assert not boundary_route.is_normal
    assert triplet_is_normal_on_bz_surface(boundary_route, images)
    assert minimum_bz_reciprocal_vector(
        boundary_route,
        images,
        np.eye(3),
        reduced_reciprocal_lattice=np.eye(3),
    ) == (0, 0, 0)


def test_bz_surface_normal_classification_is_residue_order_independent() -> None:
    mesh = UniformBZMesh((2, 2, 1))
    lattice = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.45, 0.9, 0.0],
            [0.1, 0.2, 1.1],
        ]
    )
    images = shortest_bz_address_images(
        mesh,
        lattice,
        reduced_reciprocal_lattice=np.linalg.inv(lattice),
    )

    for target in range(mesh.n_qpoints):
        for triplet in mesh.triplets_at_q(target):
            expected = any(
                all(
                    first[axis] + second[axis] + third[axis] == 0
                    for axis in range(3)
                )
                for first in images[triplet.q0_index]
                for second in images[triplet.q1_index]
                for third in images[triplet.q2_index]
            )
            assert triplet_is_normal_on_bz_surface(triplet, images) == expected


def test_shortest_bz_images_match_phono3py_on_odd_hard_skew_mesh() -> None:
    mesh = UniformBZMesh((3, 3, 3))
    lattice = _HARD_SKEW_LATTICE

    images = shortest_bz_address_images(
        mesh,
        lattice,
        reduced_reciprocal_lattice=_HARD_SKEW_REDUCED_RECIPROCAL,
    )

    # Frozen from phono3py 4.4.0 BZGrid with store_dense_gp_map=True.
    assert images[5] == (GridAddress((-1, -2, 0)),)
    assert images[13] == (GridAddress((-2, -2, 1)),)
    assert images[26] == (GridAddress((2, 2, -1)),)


def test_phono3py_bz_tolerance_controls_even_hard_skew_normal_route() -> None:
    mesh = UniformBZMesh((4, 3, 2))
    lattice = _HARD_SKEW_LATTICE

    images = shortest_bz_address_images(
        mesh,
        lattice,
        reduced_reciprocal_lattice=_HARD_SKEW_REDUCED_RECIPROCAL,
    )

    # Phono3py 4.4.0 keeps both images: their squared reciprocal-space
    # distances differ by 0.0042433996352531, which is inside its
    # mesh-scaled GRID_TOLERANCE_FACTOR=0.01 boundary tolerance.
    assert images[4] == (GridAddress((0, 1, 0)),)
    assert images[17] == (
        GridAddress((-3, -2, 1)),
        GridAddress((1, 1, -1)),
    )
    assert images[19] == (GridAddress((3, 1, -1)),)

    triplet = mesh.triplets_at_q(4)[17]
    assert triplet.indices == (4, 17, 19)
    assert minimum_bz_reciprocal_vector(
        triplet,
        images,
        lattice,
        reduced_reciprocal_lattice=_HARD_SKEW_REDUCED_RECIPROCAL,
    ) == (0, 0, 0)
    assert triplet_is_normal_on_bz_surface(triplet, images)


def test_phono3py_bz_tolerance_uses_reduced_reciprocal_basis() -> None:
    mesh = UniformBZMesh((2, 2, 1))
    lattice = _HARD_SKEW_LATTICE

    images = shortest_bz_address_images(
        mesh,
        lattice,
        reduced_reciprocal_lattice=_HARD_SKEW_REDUCED_RECIPROCAL,
    )

    # Frozen from phono3py 4.4.0 BZGrid. Its squared-distance tolerance is
    # evaluated in the reduced reciprocal basis (0.09916745375909292 here),
    # not from the unreduced primitive basis (which would give
    # 0.03174711135289619 and incorrectly omit the two axial images).
    assert images[2] == (
        GridAddress((-2, -1, 1)),
        GridAddress((0, -1, 0)),
        GridAddress((0, 1, 0)),
        GridAddress((2, 1, -1)),
    )


def test_lazy_niggli_reduction_is_warning_clean_when_spglib_is_available() -> None:
    pytest.importorskip("spglib")

    reduced = niggli_reduced_reciprocal_lattice(np.eye(3))

    np.testing.assert_allclose(
        reduced.T @ reduced,
        np.eye(3),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_lazy_niggli_reduction_matches_frozen_phono3py_basis() -> None:
    pytest.importorskip("spglib")

    reduced = niggli_reduced_reciprocal_lattice(_HARD_SKEW_LATTICE)

    np.testing.assert_allclose(
        reduced,
        _HARD_SKEW_REDUCED_RECIPROCAL,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert not reduced.flags.writeable
    assert shortest_bz_address_images(
        UniformBZMesh((2, 2, 1)),
        _HARD_SKEW_LATTICE,
    )[2] == (
        GridAddress((-2, -1, 1)),
        GridAddress((0, -1, 0)),
        GridAddress((0, 1, 0)),
        GridAddress((2, 1, -1)),
    )


def test_explicit_reciprocal_basis_must_be_integer_equivalent() -> None:
    with pytest.raises(ValueError, match="integer basis transformation"):
        shortest_bz_address_images(
            UniformBZMesh((2, 2, 1)),
            np.eye(3),
            reduced_reciprocal_lattice=np.diag([1.1, 1.0, 1.0]),
        )


def test_explicit_reciprocal_basis_must_be_unimodular() -> None:
    with pytest.raises(ValueError, match="unimodular"):
        shortest_bz_address_images(
            UniformBZMesh((2, 2, 1)),
            np.eye(3),
            reduced_reciprocal_lattice=0.5 * np.eye(3),
        )


def test_bz_surface_image_validation_rejects_wrong_residue_table() -> None:
    mesh = UniformBZMesh((2, 1, 1))
    triplet = mesh.triplets_at_q(0)[1]

    with pytest.raises(ValueError, match="wrong residue"):
        triplet_is_normal_on_bz_surface(
            triplet,
            (
                (GridAddress((1, 0, 0)),),
                (GridAddress((0, 0, 0)),),
            ),
        )


@pytest.mark.parametrize("bad_index", [-1, 24, True, 1.0])
def test_mesh_rejects_invalid_grid_indices(bad_index) -> None:
    mesh = UniformBZMesh((2, 3, 4))

    with pytest.raises((TypeError, IndexError)):
        mesh.address_from_index(bad_index)
