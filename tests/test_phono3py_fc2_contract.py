from __future__ import annotations

import builtins
import hashlib
import importlib
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from negfpy.anharmonic.fc2 import FC2Data, validate_fc2
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2


def _write_fc2(
    path,
    values: np.ndarray,
    *,
    p2s_map: np.ndarray | None = None,
    s2p_map: np.ndarray | None = None,
    units: str | None = None,
    version: str | None = None,
    cutoff: float | None = None,
) -> None:
    h5py = pytest.importorskip("h5py")
    with h5py.File(path, "w") as handle:
        handle.attrs["creator"] = "synthetic-fc2-contract"
        dataset = handle.create_dataset("force_constants", data=values)
        dataset.attrs["payload"] = "raw"
        if p2s_map is not None:
            mapping = handle.create_dataset("p2s_map", data=p2s_map)
            mapping.attrs["meaning"] = "primitive-to-supercell"
        if s2p_map is not None:
            handle.create_dataset("s2p_map", data=s2p_map)
        if units is not None:
            handle.create_dataset("physical_unit", data=np.bytes_(units))
        if version is not None:
            handle.create_dataset("version", data=np.bytes_(version))
        if cutoff is not None:
            handle.create_dataset("cutoff", data=cutoff)


def test_fc2_data_detaches_and_freezes_every_input() -> None:
    values = np.arange(36.0).reshape(2, 2, 3, 3)
    masses = np.array([28.0, 29.0])
    p2s = np.array([1], dtype=np.int64)
    s2p = np.array([1, 1], dtype=np.int64)
    nested = np.array([1.0, 2.0])

    data = FC2Data(
        values=values,
        masses=masses,
        compact=False,
        p2s_map=p2s,
        s2p_map=s2p,
        version="4.4.0",
        cutoff=4.5,
        provenance={"reader": {"numbers": nested}},
    )

    values[...] = -1.0
    masses[...] = -1.0
    p2s[...] = -1
    s2p[...] = -1
    nested[...] = -1.0

    np.testing.assert_array_equal(data.values, np.arange(36.0).reshape(2, 2, 3, 3))
    np.testing.assert_array_equal(data.masses, [28.0, 29.0])
    np.testing.assert_array_equal(data.p2s_map, [1])
    np.testing.assert_array_equal(data.s2p_map, [1, 1])
    np.testing.assert_array_equal(
        data.provenance["reader"]["numbers"],
        [1.0, 2.0],
    )
    assert isinstance(data.provenance, MappingProxyType)
    assert isinstance(data.provenance["reader"], MappingProxyType)
    assert data.fc2 is data.values
    assert not data.values.flags.writeable
    assert not data.masses.flags.writeable
    with pytest.raises(ValueError):
        data.values[0, 0, 0, 0] = 0.0
    with pytest.raises(TypeError):
        data.provenance["new"] = "forbidden"
    with pytest.raises(FrozenInstanceError):
        data.version = "changed"  # type: ignore[misc]


def test_validate_fc2_is_non_mutating_and_reports_schema_failures() -> None:
    values = np.ones((1, 2, 3, 3))
    values[0, 0, 0, 0] = np.nan
    data = FC2Data(
        values=values,
        masses=[28.0, -1.0],
        compact=True,
        p2s_map=np.array([3], dtype=np.int64),
        s2p_map=np.array([0, 0], dtype=np.int64),
        units="Ry/bohr^2",
    )
    before = data.values.copy()

    report = validate_fc2(data)

    assert report.dimension_valid
    assert not report.finite_valid
    assert not report.masses_valid
    assert not report.units_valid
    assert not report.mapping_valid
    assert not report.passed
    assert len(report.errors) >= 4
    np.testing.assert_array_equal(data.values, before)
    assert isinstance(report.metrics, MappingProxyType)


def test_validate_fc2_accepts_full_and_compact_official_layouts() -> None:
    full = FC2Data(
        values=np.ones((3, 3, 3, 3)),
        masses=[1.0, 2.0, 3.0],
        compact=False,
    )
    compact = FC2Data(
        values=np.ones((1, 3, 3, 3)),
        masses=[1.0, 2.0, 3.0],
        compact=True,
        p2s_map=np.array([2], dtype=np.int64),
        s2p_map=np.array([2, 2, 2], dtype=np.int64),
    )

    assert validate_fc2(full).passed
    assert validate_fc2(compact).passed
    assert full.n_supercell_atoms == 3
    assert full.n_primitive_atoms is None
    assert compact.n_supercell_atoms == 3
    assert compact.n_primitive_atoms == 1


def test_validate_compact_fc2_requires_p2s_map() -> None:
    data = FC2Data(
        values=np.ones((1, 2, 3, 3)),
        masses=[1.0, 2.0],
        compact=True,
    )

    report = validate_fc2(data)

    assert not report.mapping_valid
    assert not report.passed
    assert any("p2s_map is required" in message for message in report.errors)


def test_read_full_fc2_preserves_official_metadata_and_raw_values(tmp_path) -> None:
    path = tmp_path / "fc2-full.hdf5"
    values = np.arange(36, dtype=np.float32).reshape(2, 2, 3, 3)
    _write_fc2(
        path,
        values,
        p2s_map=np.array([1], dtype=np.int64),
        s2p_map=np.array([1, 1], dtype=np.int64),
        units="eV/angstrom^2",
        version="4.4.0",
        cutoff=5.25,
    )

    data = read_phono3py_fc2(path, masses=[28.0855, 28.0855])

    assert not data.compact
    assert data.values.dtype == np.float32
    np.testing.assert_array_equal(data.values, values)
    np.testing.assert_array_equal(data.p2s_map, [1])
    np.testing.assert_array_equal(data.s2p_map, [1, 1])
    assert data.units == "eV/angstrom^2"
    assert data.version == "4.4.0"
    assert data.cutoff == pytest.approx(5.25)
    assert data.provenance["layout_source"] == "shape_equal_full"
    assert data.provenance["source_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert data.provenance["source_dataset"] == "force_constants"
    assert data.provenance["source_dataset_attrs"]["payload"] == "raw"
    assert data.provenance["source_file_attrs"]["creator"] == (
        "synthetic-fc2-contract"
    )
    assert data.provenance["source_auxiliary_dataset_attrs"]["p2s_map"][
        "meaning"
    ] == "primitive-to-supercell"
    assert validate_fc2(data).passed


def test_read_compact_fc2_preserves_maps_and_documented_unit_default(tmp_path) -> None:
    path = tmp_path / "fc2-compact.hdf5"
    values = np.arange(27.0).reshape(1, 3, 3, 3)
    _write_fc2(
        path,
        values,
        p2s_map=np.array([2], dtype=np.int64),
        s2p_map=np.array([2, 2, 2], dtype=np.int64),
    )

    data = read_phono3py_fc2(path, masses=[4.0, 9.0, 16.0])

    assert data.compact
    np.testing.assert_array_equal(data.values, values)
    np.testing.assert_array_equal(data.p2s_map, [2])
    np.testing.assert_array_equal(data.s2p_map, [2, 2, 2])
    assert data.units == "eV/angstrom^2"
    assert data.version is None
    assert data.cutoff is None
    assert data.provenance["units_source"] == "phono3py_documented_default"
    assert data.provenance["layout_source"] == "shape_unequal_compact"
    assert validate_fc2(data).passed


def test_equal_atom_dimensions_are_always_classified_as_full(tmp_path) -> None:
    path = tmp_path / "fc2-equivalent.hdf5"
    _write_fc2(
        path,
        np.ones((1, 1, 3, 3)),
        p2s_map=np.array([0], dtype=np.int64),
        s2p_map=np.array([0], dtype=np.int64),
    )

    data = read_phono3py_fc2(path, masses=[28.0])

    assert not data.compact
    assert data.provenance["layout_source"] == "shape_equal_full"


@pytest.mark.parametrize(
    ("map_name", "file_map", "explicit_map"),
    [
        ("p2s_map", np.array([2]), np.array([1])),
        ("s2p_map", np.array([2, 2, 2]), np.array([1, 1, 1])),
    ],
)
def test_reader_rejects_explicit_map_conflicts(
    tmp_path,
    map_name: str,
    file_map: np.ndarray,
    explicit_map: np.ndarray,
) -> None:
    path = tmp_path / f"fc2-conflict-{map_name}.hdf5"
    source_maps = {
        "p2s_map": np.array([2]),
        "s2p_map": np.array([2, 2, 2]),
    }
    source_maps[map_name] = file_map
    _write_fc2(
        path,
        np.ones((1, 3, 3, 3)),
        p2s_map=source_maps["p2s_map"],
        s2p_map=source_maps["s2p_map"],
    )

    kwargs = {map_name: explicit_map}
    with pytest.raises(ValueError, match=rf"{map_name}.*conflicts"):
        read_phono3py_fc2(
            path,
            masses=[1.0, 1.0, 1.0],
            **kwargs,
        )


def test_matching_explicit_maps_are_verified_not_used_to_remap(tmp_path) -> None:
    path = tmp_path / "fc2-matching-map.hdf5"
    values = np.arange(27.0).reshape(1, 3, 3, 3)
    _write_fc2(
        path,
        values,
        p2s_map=np.array([2], dtype=np.int64),
        s2p_map=np.array([2, 2, 2], dtype=np.int64),
    )

    data = read_phono3py_fc2(
        path,
        masses=[1.0, 1.0, 1.0],
        p2s_map=np.array([2], dtype=np.int64),
        s2p_map=np.array([2, 2, 2], dtype=np.int64),
    )

    np.testing.assert_array_equal(data.values, values)
    assert data.provenance["p2s_map_source"] == (
        "argument_verified_against_dataset:p2s_map"
    )
    assert data.provenance["s2p_map_source"] == (
        "argument_verified_against_dataset:s2p_map"
    )


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2, 3),
        (2, 2, 2, 3),
        (2, 2, 3, 2),
        (3, 2, 3, 3),
        (0, 2, 3, 3),
    ],
)
def test_reader_rejects_nonofficial_fc2_shapes(tmp_path, shape) -> None:
    path = tmp_path / "fc2-invalid-shape.hdf5"
    _write_fc2(path, np.empty(shape))

    with pytest.raises(ValueError, match="FC2.*shape"):
        read_phono3py_fc2(path, masses=np.ones(max(shape[1], 1)))


def test_reader_requires_force_constants_dataset(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc2-missing.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("not_force_constants", data=np.ones(1))

    with pytest.raises(KeyError, match="force_constants"):
        read_phono3py_fc2(path, masses=[1.0])


@pytest.mark.parametrize("link_kind", ["soft", "external"])
def test_reader_rejects_non_hard_primary_dataset_links(tmp_path, link_kind) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / f"fc2-primary-{link_kind}.hdf5"
    external = tmp_path / "fc2-primary-external.hdf5"
    values = np.ones((1, 1, 3, 3))
    if link_kind == "external":
        with h5py.File(external, "w") as handle:
            handle.create_dataset("payload", data=values)
    with h5py.File(path, "w") as handle:
        if link_kind == "soft":
            handle.create_dataset("payload", data=values)
            handle["force_constants"] = h5py.SoftLink("/payload")
        else:
            handle["force_constants"] = h5py.ExternalLink(
                str(external),
                "/payload",
            )

    with pytest.raises(ValueError, match="internal hard-link chain"):
        read_phono3py_fc2(path, masses=[1.0])


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
    path = tmp_path / f"fc2-optional-{name}.hdf5"
    external = tmp_path / f"fc2-optional-{name}-external.hdf5"
    payload = np.array([0], dtype=np.int64) if name == "p2s_map" else np.bytes_("4.4.0")
    if link_kind == "external":
        with h5py.File(external, "w") as handle:
            handle.create_dataset("payload", data=payload)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("force_constants", data=np.ones((1, 1, 3, 3)))
        if link_kind == "soft":
            handle.create_dataset("payload", data=payload)
            handle[name] = h5py.SoftLink("/payload")
        else:
            handle[name] = h5py.ExternalLink(str(external), "/payload")

    with pytest.raises(ValueError, match="internal hard-link chain"):
        read_phono3py_fc2(path, masses=[1.0])


def test_reader_validates_inventory_links_even_when_not_scientific_inputs(
    tmp_path,
) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc2-inventory-soft-link.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("force_constants", data=np.ones((1, 1, 3, 3)))
        handle.create_dataset("inventory_payload", data=np.ones(1))
        handle["inventory_alias"] = h5py.SoftLink("/inventory_payload")

    with pytest.raises(ValueError, match="inventory_alias.*hard-link chain"):
        read_phono3py_fc2(path, masses=[1.0])


def test_reader_accepts_nested_internal_hard_link_chain(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc2-nested-hard-link.hdf5"
    values = np.arange(9.0).reshape(1, 1, 3, 3)
    with h5py.File(path, "w") as handle:
        payload = handle.create_group("payload")
        payload.create_dataset("force_constants", data=values)
        handle["hard_alias"] = payload

    data = read_phono3py_fc2(
        path,
        masses=[1.0],
        dataset="hard_alias/force_constants",
    )

    np.testing.assert_array_equal(data.values, values)


def test_reader_rejects_soft_parent_in_nested_dataset_path(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc2-nested-soft-link.hdf5"
    with h5py.File(path, "w") as handle:
        payload = handle.create_group("payload")
        payload.create_dataset("force_constants", data=np.ones((1, 1, 3, 3)))
        handle["soft_alias"] = h5py.SoftLink("/payload")

    with pytest.raises(ValueError, match="soft_alias.*hard-link chain"):
        read_phono3py_fc2(
            path,
            masses=[1.0],
            dataset="soft_alias/force_constants",
        )


def test_reader_rejects_container_change_during_read(tmp_path, monkeypatch) -> None:
    path = tmp_path / "fc2-changing-container.hdf5"
    _write_fc2(path, np.ones((1, 1, 3, 3)))
    reader = importlib.import_module("negfpy.anharmonic.io.phono3py_fc2")
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(reader, "_sha256_file", lambda _path: next(digests))

    with pytest.raises(RuntimeError, match="container changed while it was read"):
        read_phono3py_fc2(path, masses=[1.0])


def test_reader_has_actionable_optional_h5py_error(tmp_path, monkeypatch) -> None:
    real_import = builtins.__import__

    def block_h5py(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("blocked for contract test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_h5py)

    with pytest.raises(ImportError, match="optional 'h5py'.*fc2\\.hdf5"):
        read_phono3py_fc2(tmp_path / "unreadable.hdf5", masses=[1.0])
