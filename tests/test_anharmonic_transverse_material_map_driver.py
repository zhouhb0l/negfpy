"""Material-map driver guards and binary preservation without material reads."""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def driver():
    return runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_transverse_material_map.py"))


def _profile(tmp_path):
    return {
        "map": {
            "g": 0.03,
            "eta_lead": 0.0015625,
            "eta_device": 0,
            "n_device_layers": 9,
            "interaction_layers": [2, 3, 4, 5, 6],
            "mesh": [4, 1, 1],
            "grid_step": 0.025,
            "n_positive": 300,
            "temperatures_kelvin": [305, 295],
            "make_r0_average": True,
        },
        "runtime_inputs": [
            {"name": name, "path": str(tmp_path / ("never-read-" + name))}
            for name in ("crystal", "fc2", "fc3")
        ],
        "source_commit": "synthetic-driver-guard-fixture",
        "source_tree": "synthetic-driver-guard-fixture",
    }


@pytest.mark.parametrize("key,value,message", [
    ("g", 0.031, "coupling and broadening"),
    ("eta_lead", 0.00078125, "coupling and broadening"),
    ("eta_device", 1e-8, "coupling and broadening"),
    ("n_device_layers", 10, "interaction crop"),
    ("interaction_layers", [1, 2, 3, 4, 5], "interaction crop"),
    ("interaction_layers", [6, 5, 4, 3, 2], "interaction crop"),
    ("mesh", [1, 1, 1], "complex opposite-q pair"),
    ("mesh", [2, 2, 1], "complex opposite-q pair"),
    ("grid_step", 0.0125, "601-node"),
    ("n_positive", 299, "601-node"),
    ("temperatures_kelvin", [295, 305], "thermal bias"),
    ("temperatures_kelvin", [300, 300], "thermal bias"),
    ("make_r0_average", False, "IFC normalization"),
    ("make_r0_average", 1, "IFC normalization"),
])
def test_changed_frozen_configuration_rejects_before_any_material_read(
    driver, tmp_path, monkeypatch, key, value, message,
):
    run = driver["run"]
    for name in (
        "read_phono3py_crystal_mapping", "read_phono3py_fc2", "read_phono3py_fc3",
        "build_compact_transverse_contact", "fourier_material_cubic_packed",
        "evaluate_packed_born_map",
    ):
        monkeypatch.setitem(run.__globals__, name,
                            lambda *a, **k: pytest.fail("guard reached material/numerical work"))
    profile = _profile(tmp_path)
    profile["map"][key] = value
    result, events = {}, []
    directory = tmp_path / "arrays"
    with pytest.raises(ValueError, match=message):
        run(profile, result, lambda *row: events.append(row), directory)
    assert result == {}
    assert events == []
    assert not directory.exists()
    assert not any(tmp_path.iterdir())


def test_valid_frozen_configuration_stops_at_first_read_sentinel(driver, tmp_path, monkeypatch):
    class ReadSentinel(Exception):
        pass

    calls = []

    def first_read(path, *, purpose):
        calls.append((path, purpose))
        raise ReadSentinel

    run = driver["run"]
    monkeypatch.setitem(run.__globals__, "read_phono3py_crystal_mapping", first_read)
    monkeypatch.setitem(run.__globals__, "evaluate_packed_born_map",
                        lambda *a, **k: pytest.fail("guard fixture reached numerical map"))
    profile = _profile(tmp_path)
    result, events = {}, []
    with pytest.raises(ReadSentinel):
        run(profile, result, lambda *row: events.append(row), tmp_path / "arrays")
    assert calls == [(tmp_path / "never-read-crystal", "fc2")]
    assert events == [("read_fc2_start",)]
    assert result["settings"] == profile["map"]
    assert result["source_commit"] == profile["source_commit"]
    assert result["source_tree"] == profile["source_tree"]
    assert result["scope"]["contact_frequency_count"] == 601
    assert result["scope"]["input_sigma"] == "exact-zero"
    assert result["scope"]["scba"] is False
    assert result["scope"]["mapped_sigma_dyson_evaluated"] is False
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_array_artifact_preserves_complex_content_order_hash_and_nonoverwrite(
    driver, tmp_path, dtype, layout,
):
    real = np.arange(24).reshape(4, 6)
    values = (real + 1j * (3 - 2 * real)).astype(dtype)
    if layout == "F":
        values = np.asfortranarray(values)
    elif layout == "strided":
        values = values[::-2, ::2]
        assert not values.flags.c_contiguous and not values.flags.f_contiguous
    original = values.copy()
    record = driver["save_array"](tmp_path, "small_complex", values)
    path = tmp_path / "small_complex.npy"
    saved_bytes = path.read_bytes()
    loaded = np.load(path, allow_pickle=False)
    np.testing.assert_array_equal(loaded, original)
    np.testing.assert_array_equal(values, original)
    assert loaded.dtype == values.dtype
    assert loaded.flags.f_contiguous if layout == "F" else loaded.flags.c_contiguous
    assert record == {
        "relative_path": "arrays/small_complex.npy",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "array_bytes": values.nbytes,
        "file_bytes": len(saved_bytes),
        "file_sha256": hashlib.sha256(saved_bytes).hexdigest(),
    }
    with pytest.raises(FileExistsError):
        driver["save_array"](tmp_path, "small_complex", values + 100)
    assert path.read_bytes() == saved_bytes
    assert sorted(item.name for item in tmp_path.iterdir()) == ["small_complex.npy"]
