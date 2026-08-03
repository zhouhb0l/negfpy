from __future__ import annotations

import hashlib
from types import MappingProxyType

import numpy as np
import pytest

from negfpy.anharmonic.fc3 import (
    ATOMIC_MASS_UNIT_KG,
    EV_ANGSTROM3_TO_J_M3,
    FC3Data,
    MassNormalizedFC3,
    convert_fc3_values_to_si,
    mass_normalize_fc3,
    validate_fc3,
)
from negfpy.anharmonic.io import read_phono3py_fc3


def _symmetric_full_fc3(n_atoms: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    values = rng.normal(size=(n_atoms, n_atoms, n_atoms, 3, 3, 3))
    terms = []
    for axes in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        terms.append(
            np.transpose(values, axes + tuple(3 + axis for axis in axes))
        )
    return sum(terms) / 6.0


def test_fc3_data_detaches_and_freezes_all_inputs() -> None:
    values = np.ones((1, 1, 1, 3, 3, 3))
    masses = np.array([28.0])
    p2s = np.array([0], dtype=np.int64)
    nested = np.array([1.0, 2.0])
    data = FC3Data(
        values=values,
        masses=masses,
        compact=True,
        p2s_map=p2s,
        provenance={"reader": {"version": nested}},
    )

    values[...] = 9.0
    masses[...] = 9.0
    p2s[...] = 9
    nested[...] = 9.0

    assert np.all(data.values == 1.0)
    assert np.array_equal(data.masses, [28.0])
    assert np.array_equal(data.p2s_map, [0])
    assert np.array_equal(data.provenance["reader"]["version"], [1.0, 2.0])
    assert isinstance(data.provenance, MappingProxyType)
    assert isinstance(data.provenance["reader"], MappingProxyType)
    with pytest.raises(ValueError):
        data.values[0, 0, 0, 0, 0, 0] = 2.0
    with pytest.raises(TypeError):
        data.provenance["new"] = "value"


def test_validate_full_fc3_and_detect_permutation_failure_without_mutation() -> None:
    values = _symmetric_full_fc3(2)
    data = FC3Data(values=values, masses=[4.0, 9.0])

    report = validate_fc3(data)
    assert report.passed
    assert report.permutation_complete
    assert report.source_fc3_content_sha256 == data.provenance[
        "source_fc3_content_sha256"
    ]
    assert report.metrics["permutation_max_abs"] < 1.0e-12

    broken = values.copy()
    broken[0, 1, 1, 0, 1, 2] += 1.0
    broken_data = FC3Data(values=broken, masses=[4.0, 9.0])
    original = broken_data.values.copy()
    broken_report = validate_fc3(broken_data)
    assert not broken_report.permutation_valid
    np.testing.assert_array_equal(broken_data.values, original)


def test_declared_fc3_content_identity_cannot_spoof_another_tensor() -> None:
    values = np.ones((1, 1, 1, 3, 3, 3))

    with pytest.raises(ValueError, match="conflicts with the canonical FC3"):
        FC3Data(
            values=values,
            masses=[1.0],
            provenance={"source_fc3_content_sha256": "0" * 64},
        )


def test_validate_reports_finite_mass_and_mapping_errors() -> None:
    values = np.ones((1, 2, 2, 3, 3, 3))
    values[0, 0, 0, 0, 0, 0] = np.nan
    data = FC3Data(
        values=values,
        masses=[4.0, -1.0],
        compact=True,
        p2s_map=np.array([2], dtype=np.int64),
        s2p_map=np.array([0, 0], dtype=np.int64),
    )
    report = validate_fc3(data)

    assert report.dimension_valid
    assert not report.finite_valid
    assert not report.masses_valid
    assert not report.mapping_valid
    assert not report.permutation_valid
    assert not report.passed
    assert len(report.errors) >= 3


def test_validate_rejects_unknown_force_constant_or_mass_units() -> None:
    force_units = FC3Data(
        values=np.ones((1, 1, 1, 3, 3, 3)),
        masses=[1.0],
        units="Ry/bohr^3",
    )
    mass_units = FC3Data(
        values=np.ones((1, 1, 1, 3, 3, 3)),
        masses=[1.0],
        mass_units="electron_mass",
    )

    assert not validate_fc3(force_units).units_valid
    assert not validate_fc3(mass_units).units_valid
    with pytest.raises(ValueError, match="Unsupported FC3 units"):
        mass_normalize_fc3(force_units)


def test_compact_validation_is_explicitly_partial() -> None:
    data = FC3Data(
        values=np.ones((1, 3, 3, 3, 3, 3)),
        masses=[1.0, 2.0, 3.0],
        compact=True,
        p2s_map=np.array([2], dtype=np.int64),
        s2p_map=np.array([2, 2, 2], dtype=np.int64),
    )
    report = validate_fc3(data)

    assert report.passed
    assert report.permutation_valid
    assert not report.permutation_complete
    assert 0.0 < report.metrics["permutation_coverage"] < 1.0
    assert any("partial" in warning.lower() for warning in report.warnings)


def test_mass_normalize_full_fc3_in_si() -> None:
    data = FC3Data(
        values=np.ones((2, 2, 2, 3, 3, 3)),
        masses=[4.0, 9.0],
        units="eV/Angstrom^3",
        mass_units="amu",
    )
    normalized = mass_normalize_fc3(data)

    assert isinstance(normalized, MassNormalizedFC3)
    assert normalized is not data
    assert normalized.units == "J/m^3/kg^(3/2)"
    expected = EV_ANGSTROM3_TO_J_M3 / np.sqrt(
        (4.0 * ATOMIC_MASS_UNIT_KG)
        * (9.0 * ATOMIC_MASS_UNIT_KG)
        * (4.0 * ATOMIC_MASS_UNIT_KG)
    )
    assert normalized.values[0, 1, 0, 0, 0, 0] == pytest.approx(expected)
    np.testing.assert_allclose(
        normalized.masses,
        np.array([4.0, 9.0]) * ATOMIC_MASS_UNIT_KG,
    )
    assert np.all(data.values == 1.0)
    assert data.units == "eV/Angstrom^3"


def test_mass_normalize_compact_uses_p2s_for_first_atom() -> None:
    data = FC3Data(
        values=np.ones((1, 3, 3, 3, 3, 3)),
        masses=[4.0, 9.0, 16.0],
        compact=True,
        p2s_map=np.array([2], dtype=np.int64),
        s2p_map=np.array([2, 2, 2], dtype=np.int64),
    )
    normalized = mass_normalize_fc3(data, output_units="source")

    expected = 1.0 / np.sqrt(16.0 * 4.0 * 9.0)
    assert normalized.values[0, 0, 1, 0, 0, 0] == pytest.approx(expected)
    assert normalized.units == "(eV/angstrom^3)/(amu)^(3/2)"


def test_explicit_ev_angstrom_cubed_to_si_conversion() -> None:
    converted = convert_fc3_values_to_si([1.0, -2.0], "eV/Å^3")
    np.testing.assert_allclose(
        converted,
        EV_ANGSTROM3_TO_J_M3 * np.array([1.0, -2.0]),
        rtol=0.0,
        atol=0.0,
    )


def test_read_phono3py_full_hdf5_records_source_provenance(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc3-full.hdf5"
    values = _symmetric_full_fc3(2)
    with h5py.File(path, "w") as handle:
        handle.attrs["creator"] = "synthetic-test"
        dataset = handle.create_dataset("fc3", data=values)
        dataset.attrs["physical_unit"] = "eV/angstrom^3"
        dataset.attrs["version"] = 4

    data = read_phono3py_fc3(path, masses=[28.0, 29.0])

    assert not data.compact
    np.testing.assert_allclose(data.values, values)
    assert data.units == "eV/angstrom^3"
    assert data.provenance["source_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    assert data.provenance["source_dataset"] == "fc3"
    assert data.provenance["source_dataset_attrs"]["version"] == 4
    assert data.provenance["source_file_attrs"]["creator"] == "synthetic-test"
    assert validate_fc3(data).passed


def test_read_phono3py_compact_hdf5_preserves_maps(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc3-compact.hdf5"
    values = np.ones((1, 3, 3, 3, 3, 3))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fc3", data=values)
        handle.create_dataset("p2s_map", data=np.array([2], dtype=np.int64))
        handle.create_dataset("s2p_map", data=np.array([2, 2, 2], dtype=np.int64))

    data = read_phono3py_fc3(path, masses=[4.0, 9.0, 16.0])

    assert data.compact
    assert data.units == "eV/angstrom^3"
    assert data.provenance["units_source"] == "phono3py_documented_default"
    np.testing.assert_array_equal(data.p2s_map, [2])
    np.testing.assert_array_equal(data.s2p_map, [2, 2, 2])
    assert validate_fc3(data).passed
    normalized = mass_normalize_fc3(data, output_units="source")
    assert normalized.values[0, 0, 1, 0, 0, 0] == pytest.approx(
        1.0 / np.sqrt(16.0 * 4.0 * 9.0)
    )


def test_full_fc3_with_smaller_p2s_map_is_not_misclassified(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "fc3-full-with-maps.hdf5"
    values = np.ones((2, 2, 2, 3, 3, 3))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("fc3", data=values)
        handle.create_dataset("p2s_map", data=np.array([0], dtype=np.int64))
        handle.create_dataset("s2p_map", data=np.array([0, 0], dtype=np.int64))

    data = read_phono3py_fc3(path, masses=[4.0, 4.0])

    assert not data.compact
    assert data.provenance["compact_source"] == "shape_full_despite_p2s_map"
