from __future__ import annotations

import importlib
from types import MappingProxyType

import numpy as np
import pytest

from negfpy.anharmonic.fc3 import FC3Data, validate_fc3
from negfpy.anharmonic.io.phono3py import read_phono3py_fc3


def _write_fc3(
    path,
    values: np.ndarray,
    *,
    p2s_map: np.ndarray | None = None,
    s2p_map: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    cutoff_name: str | None = None,
    cutoff: float | None = None,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fc3", data=values)
        if p2s_map is not None:
            handle.create_dataset("p2s_map", data=p2s_map)
        if s2p_map is not None:
            handle.create_dataset("s2p_map", data=s2p_map)
        if mask is not None:
            handle.create_dataset("fc3_nonzero_indices", data=mask)
        if cutoff_name is not None:
            handle.create_dataset(cutoff_name, data=cutoff)


def test_square_fc3_shape_is_full_even_when_p2s_has_same_size(tmp_path) -> None:
    path = tmp_path / "square-with-equal-p2s.hdf5"
    values = np.ones((2, 2, 2, 3, 3, 3))
    _write_fc3(
        path,
        values,
        p2s_map=np.array([0, 1], dtype=np.int64),
        s2p_map=np.array([0, 1], dtype=np.int64),
    )

    data = read_phono3py_fc3(path, masses=[10.0, 11.0])

    assert not data.compact
    assert data.n_supercell_atoms == 2
    assert data.n_primitive_atoms == 2
    assert data.provenance["compact_source"] == "shape_full_equivalent"


def test_square_fc3_shape_overrides_advisory_compact_attribute(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "square-with-compact-hint.hdf5"
    values = np.ones((2, 2, 2, 3, 3, 3))
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("fc3", data=values)
        dataset.attrs["compact"] = True
        handle.create_dataset("p2s_map", data=np.array([0, 1], dtype=np.int64))

    data = read_phono3py_fc3(path, masses=[10.0, 11.0])

    assert not data.compact
    assert data.provenance["hdf5_compact_hint"] is True
    assert data.provenance["compact_source"] == "shape_overrode_hdf5_attribute"


def test_explicit_compact_argument_must_agree_with_shape(tmp_path) -> None:
    path = tmp_path / "full.hdf5"
    _write_fc3(path, np.ones((2, 2, 2, 3, 3, 3)))

    with pytest.raises(ValueError, match="compact.*shape"):
        read_phono3py_fc3(path, masses=[10.0, 11.0], compact=True)


def test_full_fc3_primitive_count_comes_from_p2s_not_tensor_size(tmp_path) -> None:
    path = tmp_path / "full-supercell.hdf5"
    values = np.ones((4, 4, 4, 3, 3, 3))
    _write_fc3(
        path,
        values,
        p2s_map=np.array([0, 2], dtype=np.int64),
        s2p_map=np.array([0, 0, 2, 2], dtype=np.int64),
    )

    data = read_phono3py_fc3(path, masses=np.ones(4))

    assert not data.compact
    assert data.n_supercell_atoms == 4
    assert data.n_primitive_atoms == 2
    assert validate_fc3(data).metrics["n_primitive_atoms"] == 2.0


@pytest.mark.parametrize(
    ("argument_name", "argument_value", "message"),
    [
        ("p2s_map", np.array([1], dtype=np.int64), "p2s_map.*conflicts"),
        ("s2p_map", np.array([0, 1], dtype=np.int64), "s2p_map.*conflicts"),
    ],
)
def test_explicit_mapping_conflict_with_hdf5_is_rejected(
    tmp_path,
    argument_name: str,
    argument_value: np.ndarray,
    message: str,
) -> None:
    path = tmp_path / f"{argument_name}-conflict.hdf5"
    _write_fc3(
        path,
        np.ones((1, 2, 2, 3, 3, 3)),
        p2s_map=np.array([0], dtype=np.int64),
        s2p_map=np.array([0, 0], dtype=np.int64),
    )

    kwargs = {argument_name: argument_value}
    with pytest.raises(ValueError, match=message):
        read_phono3py_fc3(path, masses=[10.0, 11.0], **kwargs)


def test_equal_explicit_and_hdf5_mappings_are_accepted_without_coercion(
    tmp_path,
) -> None:
    path = tmp_path / "equal-maps.hdf5"
    p2s = np.array([0], dtype=np.int64)
    s2p = np.array([0, 0], dtype=np.int64)
    _write_fc3(
        path,
        np.ones((1, 2, 2, 3, 3, 3)),
        p2s_map=p2s,
        s2p_map=s2p,
    )

    data = read_phono3py_fc3(
        path,
        masses=[10.0, 11.0],
        p2s_map=p2s.copy(),
        s2p_map=s2p.copy(),
    )

    np.testing.assert_array_equal(data.p2s_map, p2s)
    np.testing.assert_array_equal(data.s2p_map, s2p)
    assert data.provenance["p2s_map_source"] == "argument_verified_against_dataset"
    assert data.provenance["s2p_map_source"] == "argument_verified_against_dataset"


def test_reader_preserves_auxiliary_datasets_and_all_source_attributes(
    tmp_path,
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "metadata.hdf5"
    values = np.arange(2 * 2 * 2 * 27, dtype=np.float64).reshape(
        2, 2, 2, 3, 3, 3
    )
    mask = np.ones(values.shape[:3], dtype=np.byte)
    with h5py.File(path, "w") as handle:
        handle.attrs["creator"] = "contract-test"
        dataset = handle.create_dataset("fc3", data=values)
        dataset.attrs["physical_unit"] = "eV/angstrom^3"
        dataset.attrs["vector_attr"] = np.array([1, 2], dtype=np.int16)
        version = handle.create_dataset("version", data=np.bytes_("4.4.0"))
        version.attrs["kind"] = "phono3py"
        nonzero = handle.create_dataset("fc3_nonzero_indices", data=mask)
        nonzero.attrs["meaning"] = "retained triplets"
        cutoff = handle.create_dataset("fc3_cutoff", data=4.25)
        cutoff.attrs["physical_unit"] = "angstrom"
        handle.create_dataset("p2s_map", data=np.array([0, 1], dtype=np.int64))

    data = read_phono3py_fc3(path, masses=[10.0, 11.0])

    assert data.version == "4.4.0"
    assert data.fc3_cutoff == pytest.approx(4.25)
    assert data.cutoff == pytest.approx(4.25)
    np.testing.assert_array_equal(data.fc3_nonzero_indices, mask)
    assert not data.fc3_nonzero_indices.flags.writeable
    assert isinstance(data.source_attributes, MappingProxyType)
    assert data.source_attributes["file"]["creator"] == "contract-test"
    assert (
        data.source_attributes["datasets"]["fc3_nonzero_indices"]["meaning"]
        == "retained triplets"
    )
    np.testing.assert_array_equal(
        data.source_attributes["dataset"]["vector_attr"],
        [1, 2],
    )
    assert not data.source_attributes["dataset"]["vector_attr"].flags.writeable
    assert data.provenance["version"] == "4.4.0"
    assert data.provenance["fc3_cutoff"] == pytest.approx(4.25)
    assert data.provenance["fc3_cutoff_source"] == "dataset:fc3_cutoff"
    assert data.provenance["fc3_nonzero_indices_present"]
    assert data.provenance["fc3_nonzero_indices_shape"] == (2, 2, 2)
    with pytest.raises(ValueError):
        data.fc3_nonzero_indices[0, 0, 0] = 0
    with pytest.raises(TypeError):
        data.source_attributes["file"]["new"] = "value"


def test_legacy_cutoff_dataset_is_preserved_as_fc3_cutoff(tmp_path) -> None:
    path = tmp_path / "legacy-cutoff.hdf5"
    _write_fc3(
        path,
        np.ones((1, 1, 1, 3, 3, 3)),
        cutoff_name="cutoff",
        cutoff=3.5,
    )

    data = read_phono3py_fc3(path, masses=[10.0])

    assert data.fc3_cutoff == pytest.approx(3.5)
    assert data.provenance["fc3_cutoff_source"] == "dataset:cutoff"


def test_conflicting_cutoff_aliases_are_rejected(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "conflicting-cutoffs.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fc3", data=np.ones((1, 1, 1, 3, 3, 3)))
        handle.create_dataset("fc3_cutoff", data=3.5)
        handle.create_dataset("cutoff", data=4.5)

    with pytest.raises(ValueError, match="fc3_cutoff.*cutoff.*conflict"):
        read_phono3py_fc3(path, masses=[10.0])


@pytest.mark.parametrize("dtype", [np.bool_, np.int64, np.float64])
def test_nonzero_mask_requires_official_byte_dtype(tmp_path, dtype) -> None:
    path = tmp_path / f"mask-{np.dtype(dtype).name}.hdf5"
    _write_fc3(
        path,
        np.ones((1, 2, 2, 3, 3, 3)),
        p2s_map=np.array([0], dtype=np.int64),
        mask=np.ones((1, 2, 2), dtype=dtype),
    )

    with pytest.raises(TypeError, match="fc3_nonzero_indices.*byte"):
        read_phono3py_fc3(path, masses=[10.0, 11.0])


def test_nonzero_mask_must_match_fc3_atom_dimensions(tmp_path) -> None:
    path = tmp_path / "mask-shape.hdf5"
    _write_fc3(
        path,
        np.ones((1, 2, 2, 3, 3, 3)),
        p2s_map=np.array([0], dtype=np.int64),
        mask=np.ones((2, 2, 2), dtype=np.byte),
    )

    with pytest.raises(ValueError, match="fc3_nonzero_indices.*shape"):
        read_phono3py_fc3(path, masses=[10.0, 11.0])


def test_reader_does_not_apply_mask_or_modify_raw_fc3(tmp_path) -> None:
    path = tmp_path / "raw-no-correction.hdf5"
    values = np.arange(27, dtype=np.float64).reshape(1, 1, 1, 3, 3, 3)
    mask = np.zeros((1, 1, 1), dtype=np.byte)
    _write_fc3(path, values, mask=mask)

    data = read_phono3py_fc3(path, masses=[10.0])

    np.testing.assert_array_equal(data.values, values)
    np.testing.assert_array_equal(data.fc3_nonzero_indices, mask)
    assert not validate_fc3(data).permutation_valid


@pytest.mark.parametrize("link_kind", ["soft", "external"])
def test_reader_rejects_non_hard_primary_dataset_links(tmp_path, link_kind) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"fc3-primary-{link_kind}.hdf5"
    external = tmp_path / "fc3-primary-external.hdf5"
    values = np.ones((1, 1, 1, 3, 3, 3))
    if link_kind == "external":
        with h5py.File(external, "w") as handle:
            handle.create_dataset("payload", data=values)
    with h5py.File(path, "w") as handle:
        if link_kind == "soft":
            handle.create_dataset("payload", data=values)
            handle["fc3"] = h5py.SoftLink("/payload")
        else:
            handle["fc3"] = h5py.ExternalLink(str(external), "/payload")

    with pytest.raises(ValueError, match="internal hard-link chain"):
        read_phono3py_fc3(path, masses=[1.0])


@pytest.mark.parametrize(
    ("name", "link_kind"),
    [("p2s_map", "soft"), ("version", "external")],
)
def test_reader_rejects_non_hard_optional_dataset_links(
    tmp_path,
    name: str,
    link_kind: str,
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"fc3-optional-{name}.hdf5"
    external = tmp_path / f"fc3-optional-{name}-external.hdf5"
    payload = np.array([0], dtype=np.int64) if name == "p2s_map" else np.bytes_("4.4.0")
    if link_kind == "external":
        with h5py.File(external, "w") as handle:
            handle.create_dataset("payload", data=payload)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fc3", data=np.ones((1, 1, 1, 3, 3, 3)))
        if link_kind == "soft":
            handle.create_dataset("payload", data=payload)
            handle[name] = h5py.SoftLink("/payload")
        else:
            handle[name] = h5py.ExternalLink(str(external), "/payload")

    with pytest.raises(ValueError, match="internal hard-link chain"):
        read_phono3py_fc3(path, masses=[1.0])


def test_reader_validates_source_attribute_object_links(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc3-source-attrs-external-link.hdf5"
    external = tmp_path / "fc3-source-attrs-external-payload.hdf5"
    with h5py.File(external, "w") as handle:
        handle.create_dataset("payload", data=np.ones(1))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fc3", data=np.ones((1, 1, 1, 3, 3, 3)))
        handle["attribute_alias"] = h5py.ExternalLink(str(external), "/payload")

    with pytest.raises(ValueError, match="attribute_alias.*hard-link chain"):
        read_phono3py_fc3(path, masses=[1.0])


def test_reader_accepts_nested_internal_hard_link_chain(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc3-nested-hard-link.hdf5"
    values = np.arange(27.0).reshape(1, 1, 1, 3, 3, 3)
    with h5py.File(path, "w") as handle:
        payload = handle.create_group("payload")
        payload.create_dataset("fc3", data=values)
        handle["hard_alias"] = payload

    data = read_phono3py_fc3(
        path,
        masses=[1.0],
        dataset="hard_alias/fc3",
    )

    np.testing.assert_array_equal(data.values, values)


def test_reader_rejects_external_parent_in_nested_dataset_path(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc3-nested-external-link.hdf5"
    external = tmp_path / "fc3-nested-external-payload.hdf5"
    with h5py.File(external, "w") as handle:
        group = handle.create_group("payload")
        group.create_dataset("fc3", data=np.ones((1, 1, 1, 3, 3, 3)))
    with h5py.File(path, "w") as handle:
        handle["external_alias"] = h5py.ExternalLink(str(external), "/payload")

    with pytest.raises(ValueError, match="external_alias.*hard-link chain"):
        read_phono3py_fc3(
            path,
            masses=[1.0],
            dataset="external_alias/fc3",
        )


def test_reader_rejects_container_change_during_read(tmp_path, monkeypatch) -> None:
    path = tmp_path / "fc3-changing-container.hdf5"
    _write_fc3(path, np.ones((1, 1, 1, 3, 3, 3)))
    reader = importlib.import_module("negfpy.anharmonic.io.phono3py")
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(reader, "_sha256_file", lambda _path: next(digests))

    with pytest.raises(RuntimeError, match="container changed while it was read"):
        read_phono3py_fc3(path, masses=[1.0])


def test_fc3_data_detaches_new_first_class_metadata_arrays() -> None:
    values = np.ones((1, 1, 1, 3, 3, 3))
    mask = np.ones((1, 1, 1), dtype=np.byte)
    attr = np.array([1, 2], dtype=np.int64)
    data = FC3Data(
        values=values,
        masses=[10.0],
        version="4.4.0",
        fc3_nonzero_indices=mask,
        fc3_cutoff=4.0,
        source_attributes={"dataset": {"array": attr}},
    )

    values[...] = 2.0
    mask[...] = 0
    attr[...] = 9

    assert np.all(data.values == 1.0)
    assert np.all(data.fc3_nonzero_indices == 1)
    np.testing.assert_array_equal(
        data.source_attributes["dataset"]["array"],
        [1, 2],
    )
    assert not data.fc3_nonzero_indices.flags.writeable
    assert not data.source_attributes["dataset"]["array"].flags.writeable


@pytest.mark.parametrize(
    "mask",
    [
        np.ones((1, 1, 1), dtype=np.int64),
        np.ones((2, 1, 1), dtype=np.byte),
    ],
)
def test_validate_fc3_reports_invalid_nonzero_mask_contract(
    mask: np.ndarray,
) -> None:
    data = FC3Data(
        values=np.ones((1, 1, 1, 3, 3, 3)),
        masses=[10.0],
        fc3_nonzero_indices=mask,
    )

    report = validate_fc3(data)

    assert not report.dimension_valid
    assert not report.passed
    assert any("fc3_nonzero_indices" in error for error in report.errors)
