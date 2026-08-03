from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from negfpy.anharmonic._canonical import (
    _IntegerCosetTable,
    _enumerate_integer_cosets,
    build_row_coset_table,
    canonical_encode,
    crystal_mapping_content_sha256,
    digest_bytes,
    domain_digest,
    enumerate_direct_reciprocal_cosets,
    fc2_content_sha256,
    fc2_provenance_sha256,
    normalize_positive_zero,
    primitive_crystal_content_sha256,
    row_cosets_sha256,
    validate_sha256,
)
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data


def _metadata_preimage(value: object) -> bytes:
    return b"NEGFPY-METADATA\x00" + struct.pack("<H", 1) + canonical_encode(value)


def _one_atom_mapping(
    *,
    purpose: str = "fc2",
    primitive_position: tuple[float, float, float] = (0.0, 0.25, 0.5),
) -> CrystalMapping:
    primitive = CrystalCell(
        lattice=np.diag([4.0, 5.0, 6.0]),
        scaled_positions=np.array([primitive_position], dtype=np.float64),
        masses=np.array([28.0855], dtype=np.float64),
        symbols=("Si",),
    )
    supercell = CrystalCell(
        lattice=np.diag([4.0, 5.0, 6.0]),
        scaled_positions=np.array([[0.0, 0.25, 0.5]], dtype=np.float64),
        masses=np.array([28.0855], dtype=np.float64),
        symbols=("Si",),
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3, dtype=np.float64),
        supercell_matrix=np.eye(3, dtype=np.int64),
        primitive=primitive,
        supercell=supercell,
        p2s=np.array([0], dtype=np.int64),
        s2p=np.array([0], dtype=np.int64),
        p2p=np.array([0], dtype=np.int64),
        shortest_vectors=np.zeros((1, 3), dtype=np.float64),
        multiplicity=np.array([[[1, 0]]], dtype=np.int64),
        atomic_permutations=np.array([[0]], dtype=np.int64),
        translation_permutations=np.array([[0]], dtype=np.int64),
        purpose=purpose,
    )


def _fc2_provenance(*, source_path: str = "/machine/a/fc2.hdf5") -> dict[str, object]:
    source_hash = hashlib.sha256(b"phase-a-fc2").hexdigest()
    attrs = {"payload": "raw"}
    return {
        "format": "phono3py_fc2_hdf5",
        "source_path": source_path,
        "source_sha256": source_hash,
        "sha256": source_hash,
        "source_dataset": "force_constants",
        "source_dataset_shape": (1, 1, 3, 3),
        "source_dataset_dtype": "float64",
        "source_dataset_attrs": attrs,
        "dataset_attrs": dict(attrs),
        "source_file_attrs": {"creator": "fixture"},
        "source_auxiliary_dataset_attrs": {
            "p2s_map": {},
            "s2p_map": {},
            "physical_unit": {},
            "version": {},
            "cutoff": {},
        },
        "source_dataset_inventory": {
            "force_constants": {"shape": (1, 1, 3, 3), "dtype": "float64"}
        },
        "source_physical_unit": "eV/angstrom^2",
        "source_version": "4.4.0",
        "source_cutoff": None,
        "units_source": "dataset:physical_unit",
        "version_source": "dataset:version",
        "cutoff_source": "absent",
        "layout_source": "shape_equal_full",
        "p2s_map_source": "absent",
        "s2p_map_source": "absent",
    }


def _full_fc2(
    *,
    values: np.ndarray | None = None,
    provenance: dict[str, object] | None = None,
) -> FC2Data:
    if values is None:
        values = np.arange(9, dtype=np.float64).reshape(1, 1, 3, 3)
    return FC2Data(
        values=values,
        masses=np.array([28.0855], dtype=np.float64),
        units="eV/angstrom^2",
        mass_units="amu",
        compact=False,
        version="4.4.0",
        provenance={} if provenance is None else provenance,
    )


def test_metadata_golden_preimages_and_sha256_are_exact() -> None:
    empty = _metadata_preimage({})
    signed_zero = _metadata_preimage({"a": -0.0})

    assert empty.hex() == (
        "4e45474650592d4d45544144415441000100090000000000000000"
    )
    assert hashlib.sha256(empty).hexdigest() == (
        "eea5ca5f44eac686ea9a53f6503b9c10f4cd2f2a8e270bb4fba0728961ca3a9b"
    )
    assert signed_zero.hex() == (
        "4e45474650592d4d455441444154410001000901000000000000000601000000"
        "0000000061040000000000000000"
    )
    assert hashlib.sha256(signed_zero).hexdigest() == (
        "2d798412ac3a652232f7b86f0df636c7db086006c8b288277835c5eba984f88c"
    )


def test_canonical_encode_covers_declared_tags_and_orders_nfc_keys() -> None:
    values = (
        None,
        False,
        True,
        -7,
        1.25,
        complex(-0.0, -2.0),
        "e\N{COMBINING ACUTE ACCENT}",
        b"raw",
        [1, "two"],
        {"z": 1, "a": 2},
        np.array([[1, -2]], dtype=">i8"),
    )
    tags = tuple(canonical_encode(value)[0] for value in values)

    assert tags == tuple(range(0x00, 0x0B))
    assert canonical_encode("e\N{COMBINING ACUTE ACCENT}") == canonical_encode("é")
    assert canonical_encode({"z": 1, "a": 2}) == canonical_encode({"a": 2, "z": 1})
    assert canonical_encode(np.array([1, -2], dtype=">i8")) == canonical_encode(
        np.array([1, -2], dtype="<i8")
    )
    assert canonical_encode(np.array([-0.0, 0.0], dtype=">f8")) == canonical_encode(
        np.array([0.0, 0.0], dtype="<f8")
    )
    assert canonical_encode(
        np.array([complex(-0.0, -0.0)], dtype=">c16")
    ) == canonical_encode(np.array([0.0j], dtype="<c16"))


@pytest.mark.parametrize(
    "value",
    [
        {1: "non-string"},
        {"e\N{COMBINING ACUTE ACCENT}": 1, "é": 2},
        {1, 2},
        bytearray(b"mutable"),
        np.int64(1),
        np.float64(1.0),
        np.array([1], dtype=np.int32),
        np.array([1.0], dtype=np.float32),
        np.array([object()], dtype=object),
        (1 << 63),
        float("nan"),
        float("inf"),
        np.array([float("nan")], dtype=np.float64),
        np.array([complex(1.0, float("inf"))], dtype=np.complex128),
    ],
)
def test_canonical_encode_rejects_every_forbidden_representation(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        canonical_encode(value)


def test_domain_and_named_digest_validation_fail_closed() -> None:
    digest = domain_digest("NEGFPY-TEST", {"digest": bytes(32)})

    assert validate_sha256(digest) == digest
    assert digest_bytes(digest) == bytes.fromhex(digest)
    assert domain_digest("NEGFPY-TEST", {"digest": bytes(32)}) == digest
    assert domain_digest("NEGFPY-TEST", {"digest": "00" * 32}) != digest
    with pytest.raises(ValueError, match="lowercase"):
        validate_sha256(digest.upper())
    with pytest.raises(ValueError, match="64-character"):
        validate_sha256("0" * 63)
    with pytest.raises(ValueError, match="NUL"):
        domain_digest(b"BAD\x00DOMAIN", {})
    with pytest.raises(ValueError, match="ASCII"):
        domain_digest("NÉGF", {})


def test_positive_zero_normalization_is_componentwise_and_read_only() -> None:
    real = normalize_positive_zero(np.array([-0.0, 2.0], dtype=">f8"))
    complex_values = normalize_positive_zero(
        np.array([complex(-0.0, -0.0), complex(1.0, -0.0)], dtype=">c16")
    )

    assert real.dtype == np.dtype("<f8")
    assert not np.signbit(real[0])
    assert not np.signbit(complex_values.real[0])
    assert not np.signbit(complex_values.imag[0])
    assert not np.signbit(complex_values.imag[1])
    assert not real.flags.writeable


def test_silicon_cosets_have_exact_cardinality_origin_and_repeatable_hashes() -> None:
    matrix = np.array(
        [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        dtype=np.int64,
    )
    observed = [enumerate_direct_reciprocal_cosets(matrix) for _ in range(10)]
    direct, reciprocal = observed[0]

    assert isinstance(direct, _IntegerCosetTable)
    assert direct.signed_determinant == 4
    assert reciprocal.signed_determinant == 4
    assert direct.keys.shape == (4, 3)
    assert reciprocal.keys.shape == (4, 3)
    np.testing.assert_array_equal(direct.keys[0], [0, 0, 0])
    np.testing.assert_array_equal(direct.representatives[0], [0, 0, 0])
    assert all(pair[0].sha256 == direct.sha256 for pair in observed)
    assert all(pair[1].sha256 == reciprocal.sha256 for pair in observed)


def test_nonsymmetric_transform_distinguishes_direct_and_reciprocal_quotients() -> None:
    matrix = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 1]], dtype=np.int64)
    direct, reciprocal = enumerate_direct_reciprocal_cosets(matrix)

    assert direct.signed_determinant == 4
    assert reciprocal.signed_determinant == 4
    assert not np.array_equal(direct.keys, reciprocal.keys)
    assert not np.array_equal(direct.representatives, reciprocal.representatives)
    assert direct.sha256 != reciprocal.sha256
    np.testing.assert_array_equal(
        direct.keys,
        [[0, 0, 0], [0, 2, 2], [2, 1, 3], [2, 3, 1]],
    )
    np.testing.assert_array_equal(
        reciprocal.keys,
        [[0, 0, 0], [1, 2, 0], [2, 0, 0], [3, 2, 0]],
    )


def test_coset_record_hash_canonicalizes_presentation_and_rejects_corruption() -> None:
    matrix = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 1]], dtype=np.int64)
    keys, representatives = _enumerate_integer_cosets(matrix)
    permutation = np.array([2, 0, 3, 1])
    expected = row_cosets_sha256(matrix, keys, representatives, role="D")

    assert row_cosets_sha256(
        matrix,
        keys[permutation],
        representatives[permutation],
        role="D",
    ) == expected
    assert build_row_coset_table(matrix, role="D").sha256 == expected
    corrupted = keys.copy()
    corrupted[0, 0] = 1
    with pytest.raises(ValueError, match="does not reproduce"):
        row_cosets_sha256(matrix, corrupted, representatives, role="D")
    with pytest.raises(ValueError, match="nonsingular"):
        _enumerate_integer_cosets(np.zeros((3, 3), dtype=np.int64))
    with pytest.raises(TypeError, match="integers"):
        _enumerate_integer_cosets(np.eye(3, dtype=np.float64))


def test_primitive_identity_normalizes_periodic_coordinates_but_keeps_order() -> None:
    reference = _one_atom_mapping(primitive_position=(0.0, 0.25, 0.5))
    periodic = _one_atom_mapping(
        primitive_position=(1.0 - 5.0e-11, 1.25, -0.5)
    )

    assert primitive_crystal_content_sha256(reference) == (
        primitive_crystal_content_sha256(periodic)
    )


def test_fc2_and_fc3_mapping_identities_join_only_through_primitive_content() -> None:
    fc2_mapping = _one_atom_mapping(purpose="fc2")
    fc3_mapping = _one_atom_mapping(purpose="fc3")

    assert primitive_crystal_content_sha256(fc2_mapping) == (
        primitive_crystal_content_sha256(fc3_mapping)
    )
    assert crystal_mapping_content_sha256(fc2_mapping) != (
        crystal_mapping_content_sha256(fc3_mapping)
    )


def test_fc2_identity_is_endian_and_signed_zero_stable_but_sensitive() -> None:
    little = np.arange(9, dtype="<f8").reshape(1, 1, 3, 3)
    big = little.astype(">f8")
    little[0, 0, 0, 0] = -0.0
    big[0, 0, 0, 0] = 0.0
    left = _full_fc2(values=little)
    right = _full_fc2(values=big)
    changed_values = np.array(little, copy=True)
    changed_values[0, 0, 0, 1] += 1.0

    assert fc2_content_sha256(left) == fc2_content_sha256(right)
    assert fc2_content_sha256(left) != fc2_content_sha256(
        _full_fc2(values=changed_values)
    )
    changed_units = FC2Data(
        values=little,
        masses=[28.0855],
        units="J/m^2",
        mass_units="amu",
        compact=False,
        version="4.4.0",
    )
    assert fc2_content_sha256(left) != fc2_content_sha256(changed_units)


def test_fc2_content_rejects_a_numeric_conversion_that_is_not_exact() -> None:
    values = np.zeros((1, 1, 3, 3), dtype=np.int64)
    values[0, 0, 0, 0] = (1 << 53) + 1

    with pytest.raises(ValueError, match="changes a numerical value"):
        fc2_content_sha256(_full_fc2(values=values))


def test_fc2_provenance_identity_excludes_paths_and_checks_duplicate_fields() -> None:
    left = _full_fc2(provenance=_fc2_provenance(source_path="/host-a/fc2.hdf5"))
    right_provenance = _fc2_provenance(source_path="/host-b/fc2.hdf5")
    right_provenance["descriptive_extra"] = "excluded"
    right = _full_fc2(provenance=right_provenance)

    assert fc2_provenance_sha256(left) == fc2_provenance_sha256(right)

    bad_hash = _fc2_provenance()
    bad_hash["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not equal"):
        fc2_provenance_sha256(_full_fc2(provenance=bad_hash))

    bad_attrs = _fc2_provenance()
    bad_attrs["dataset_attrs"] = {"payload": "changed"}
    with pytest.raises(ValueError, match="dataset_attrs"):
        fc2_provenance_sha256(_full_fc2(provenance=bad_attrs))
