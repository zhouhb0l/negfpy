"""Reader for phono3py ``fc3.hdf5`` files.

``h5py`` is intentionally an optional dependency. Importing
``negfpy.anharmonic.io`` does not require it; an actionable error is raised
only when this reader is called without h5py installed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..fc3 import FC3Data


Array = np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _attribute_value(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return tuple(
                bytes(item).decode("utf-8", errors="replace")
                for item in value.reshape(-1)
            )
        return np.array(value, copy=True)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs(obj: Any) -> dict[str, Any]:
    return {str(key): _attribute_value(value) for key, value in obj.attrs.items()}


def _text_value(value: Any) -> str | None:
    converted = _attribute_value(value)
    if isinstance(converted, str) and converted.strip():
        return converted.strip()
    if isinstance(converted, tuple) and len(converted) == 1:
        candidate = converted[0]
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _first_text(
    dataset_attrs: dict[str, Any],
    file_attrs: dict[str, Any],
) -> tuple[str | None, str | None]:
    for name in ("physical_unit", "units", "unit"):
        if name in dataset_attrs:
            text = _text_value(dataset_attrs[name])
            if text is not None:
                return text, f"dataset_attr:{name}"
    for name in ("physical_unit", "units", "unit"):
        if name in file_attrs:
            text = _text_value(file_attrs[name])
            if text is not None:
                return text, f"file_attr:{name}"
    return None, None


def _bool_hint(*mappings: dict[str, Any]) -> bool | None:
    for mapping in mappings:
        for name in ("compact", "is_compact", "is_compact_fc"):
            if name not in mapping:
                continue
            value = _attribute_value(mapping[name])
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
                return bool(value)
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "yes", "1"}:
                    return True
                if normalized in {"false", "no", "0"}:
                    return False
    return None


def _optional_dataset(handle: Any, name: str) -> Array | None:
    if name not in handle:
        return None
    obj = handle[name]
    if not hasattr(obj, "shape"):
        raise ValueError(f"HDF5 object {name!r} is not a dataset.")
    return np.asarray(obj[...])


def read_phono3py_fc3(
    path: str | Path,
    masses: Any,
    *,
    p2s_map: Any | None = None,
    s2p_map: Any | None = None,
    units: str | None = None,
    mass_units: str = "amu",
    compact: bool | None = None,
    dataset: str = "fc3",
) -> FC3Data:
    """Read a raw full or compact phono3py FC3 tensor.

    Explicit ``p2s_map``, ``s2p_map``, ``units``, and ``compact`` arguments
    take precedence over HDF5 metadata. Phono3py's documented
    eV/Angstrom^3 unit is used when the file carries no unit attribute. The
    result is not validated, symmetrized, converted, or mass normalized.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ImportError(
            "read_phono3py_fc3 requires the optional 'h5py' package. "
            "Install h5py to read fc3.hdf5 files."
        ) from exc

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Phono3py FC3 file does not exist: {source_path}")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty HDF5 dataset name.")

    with h5py.File(source_path, "r") as handle:
        if dataset not in handle:
            raise KeyError(
                f"HDF5 dataset {dataset!r} was not found in {source_path}."
            )
        fc3_dataset = handle[dataset]
        if not isinstance(fc3_dataset, h5py.Dataset):
            raise ValueError(f"HDF5 object {dataset!r} is not a dataset.")
        values = np.asarray(fc3_dataset[...])
        dataset_attrs = _attrs(fc3_dataset)
        file_attrs = _attrs(handle)

        file_p2s = _optional_dataset(handle, "p2s_map")
        file_s2p = _optional_dataset(handle, "s2p_map")

    selected_p2s = file_p2s if p2s_map is None else np.asarray(p2s_map)
    selected_s2p = file_s2p if s2p_map is None else np.asarray(s2p_map)

    attribute_units, attribute_units_source = _first_text(
        dataset_attrs,
        file_attrs,
    )
    if units is not None:
        selected_units = units
        units_source = "argument"
    elif attribute_units is not None:
        selected_units = attribute_units
        units_source = attribute_units_source
    else:
        selected_units = "eV/angstrom^3"
        units_source = "phono3py_documented_default"

    attribute_compact = _bool_hint(dataset_attrs, file_attrs)
    if compact is not None:
        selected_compact = compact
        compact_source = "argument"
    elif attribute_compact is not None:
        selected_compact = attribute_compact
        compact_source = "hdf5_attribute"
    elif values.ndim >= 3:
        if values.shape[0] != values.shape[1]:
            selected_compact = True
            compact_source = "shape"
        elif (
            selected_p2s is not None
            and np.asarray(selected_p2s).ndim == 1
            and np.asarray(selected_p2s).size == values.shape[0]
        ):
            # For a primitive cell equal to the supercell, compact and full
            # shapes coincide. Phono3py documents p2s_map as the compact-file
            # discriminator, and both layouts are numerically equivalent in
            # this special case.
            selected_compact = True
            compact_source = "p2s_map_equal_size_shape"
        else:
            selected_compact = False
            compact_source = (
                "shape_full_despite_p2s_map"
                if selected_p2s is not None
                else "shape"
            )
    else:
        selected_compact = False
        compact_source = "shape_invalid_default_full"

    source_sha256 = _sha256_file(source_path)
    provenance = {
        "format": "phono3py_fc3_hdf5",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "sha256": source_sha256,
        "source_dataset": dataset,
        "source_dataset_shape": tuple(int(value) for value in values.shape),
        "source_dataset_dtype": str(values.dtype),
        "source_dataset_attrs": dataset_attrs,
        "dataset_attrs": dataset_attrs,
        "source_file_attrs": file_attrs,
        "units_source": units_source,
        "compact_source": compact_source,
        "p2s_map_source": (
            "argument"
            if p2s_map is not None
            else ("dataset:p2s_map" if file_p2s is not None else "absent")
        ),
        "s2p_map_source": (
            "argument"
            if s2p_map is not None
            else ("dataset:s2p_map" if file_s2p is not None else "absent")
        ),
    }
    return FC3Data(
        values=values,
        masses=masses,
        units=selected_units,
        compact=selected_compact,
        p2s_map=selected_p2s,
        s2p_map=selected_s2p,
        provenance=provenance,
        mass_units=mass_units,
    )


__all__ = ["read_phono3py_fc3"]
