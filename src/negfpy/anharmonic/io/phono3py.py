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


def _optional_scalar_dataset(handle: Any, name: str) -> Any | None:
    values = _optional_dataset(handle, name)
    if values is None:
        return None
    if values.size != 1:
        raise ValueError(f"HDF5 dataset {name!r} must contain exactly one value.")
    return _attribute_value(values.reshape(-1)[0])


def _select_mapping(
    *,
    name: str,
    explicit: Any | None,
    in_file: Array | None,
) -> tuple[Array | None, str]:
    if explicit is None:
        return (
            in_file,
            f"dataset:{name}" if in_file is not None else "absent",
        )

    selected = np.asarray(explicit)
    if in_file is None:
        return selected, "argument"
    if not np.array_equal(selected, in_file):
        raise ValueError(
            f"Explicit {name} conflicts with the mapping stored in the HDF5 file."
        )
    return selected, "argument_verified_against_dataset"


def _validate_nonzero_indices(values: Array, mask: Array | None) -> None:
    if mask is None:
        return
    if mask.dtype != np.dtype(np.byte):
        raise TypeError(
            "HDF5 dataset 'fc3_nonzero_indices' must use NumPy byte "
            "(int8) dtype."
        )
    expected_shape = values.shape[:3] if values.ndim >= 3 else None
    if expected_shape is None or mask.shape != expected_shape:
        raise ValueError(
            "HDF5 dataset 'fc3_nonzero_indices' shape must match the "
            f"three FC3 atom dimensions {expected_shape}; got {mask.shape}."
        )


def _cutoff_value(value: Any, *, dataset: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"HDF5 dataset {dataset!r} must contain a real cutoff scalar."
        )
    try:
        cutoff = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"HDF5 dataset {dataset!r} must contain a real cutoff scalar."
        ) from exc
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError(
            f"HDF5 dataset {dataset!r} must contain a finite non-negative cutoff."
        )
    return cutoff


def _version_text(value: Any, *, source: str) -> str:
    converted = _attribute_value(value)
    if isinstance(converted, str) and converted.strip():
        return converted.strip()
    if isinstance(converted, tuple) and len(converted) == 1:
        return _version_text(converted[0], source=source)
    if isinstance(converted, np.ndarray) and converted.size == 1:
        return _version_text(converted.reshape(-1)[0], source=source)
    if (
        isinstance(converted, (int, float, np.integer, np.floating))
        and not isinstance(converted, (bool, np.bool_))
        and np.isfinite(converted)
    ):
        return str(converted)
    raise ValueError(f"{source} must contain a non-empty text version.")


def _shape_classification(
    values: Array,
    p2s_map: Array | None,
) -> tuple[bool, str]:
    """Classify phono3py FC3 layout strictly from its official shape rule."""

    if values.ndim < 3:
        return False, "shape_invalid_default_full"
    if values.shape[0] != values.shape[1]:
        return True, "shape"
    if p2s_map is None:
        return False, "shape"
    mapping = np.asarray(p2s_map)
    if mapping.ndim == 1 and mapping.size == values.shape[0]:
        # Full and compact storage are numerically shape-equivalent for a
        # primitive cell equal to its supercell. Phono3py's reader resolves
        # this as full/equivalent from shape, not from the presence of p2s.
        return False, "shape_full_equivalent"
    return False, "shape_full_despite_p2s_map"


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

    Explicit maps are accepted only when they agree with mappings stored in
    the file. Layout is classified by phono3py's official shape rule:
    ``shape[0] == shape[1]`` is full (or shape-equivalent to full), otherwise
    the tensor is compact. The optional ``compact`` argument is therefore a
    consistency assertion, not a layout override.

    Phono3py's documented eV/Angstrom^3 unit is used when the file carries no
    unit attribute. The raw tensor and its non-zero mask are detached exactly
    as stored. This reader does not correct, symmetrize, truncate, apply the
    mask, remap, convert, or mass normalize the tensor.
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
        object_attrs = {
            str(name): _attrs(obj)
            for name, obj in handle.items()
        }

        file_p2s = _optional_dataset(handle, "p2s_map")
        file_s2p = _optional_dataset(handle, "s2p_map")
        file_nonzero_indices = _optional_dataset(
            handle,
            "fc3_nonzero_indices",
        )
        version_dataset = _optional_scalar_dataset(handle, "version")
        fc3_cutoff_dataset = _optional_scalar_dataset(handle, "fc3_cutoff")
        cutoff_dataset = _optional_scalar_dataset(handle, "cutoff")

    selected_p2s, p2s_source = _select_mapping(
        name="p2s_map",
        explicit=p2s_map,
        in_file=file_p2s,
    )
    selected_s2p, s2p_source = _select_mapping(
        name="s2p_map",
        explicit=s2p_map,
        in_file=file_s2p,
    )
    _validate_nonzero_indices(values, file_nonzero_indices)

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

    selected_compact, compact_source = _shape_classification(
        values,
        selected_p2s,
    )
    attribute_compact = _bool_hint(dataset_attrs, file_attrs)
    if compact is not None:
        if not isinstance(compact, (bool, np.bool_)):
            raise TypeError("compact must be a boolean when supplied.")
        if bool(compact) != selected_compact:
            raise ValueError(
                "The compact argument conflicts with the official FC3 shape "
                "classification."
            )
        compact_source = "shape_verified_by_argument"
    elif (
        attribute_compact is not None
        and attribute_compact != selected_compact
    ):
        compact_source = "shape_overrode_hdf5_attribute"
    elif attribute_compact is not None:
        compact_source = "shape_verified_by_hdf5_attribute"

    version: str | None = None
    version_source = "absent"
    if version_dataset is not None:
        version = _version_text(version_dataset, source="HDF5 dataset 'version'")
        version_source = "dataset:version"
    elif "version" in dataset_attrs:
        version = _version_text(
            dataset_attrs["version"],
            source=f"HDF5 dataset {dataset!r} attribute 'version'",
        )
        version_source = f"dataset_attr:{dataset}:version"
    elif "version" in file_attrs:
        version = _version_text(
            file_attrs["version"],
            source="HDF5 file attribute 'version'",
        )
        version_source = "file_attr:version"

    selected_cutoff: float | None = None
    cutoff_source = "absent"
    if fc3_cutoff_dataset is not None:
        selected_cutoff = _cutoff_value(
            fc3_cutoff_dataset,
            dataset="fc3_cutoff",
        )
        cutoff_source = "dataset:fc3_cutoff"
    if cutoff_dataset is not None:
        alias_cutoff = _cutoff_value(cutoff_dataset, dataset="cutoff")
        if (
            selected_cutoff is not None
            and alias_cutoff != selected_cutoff
        ):
            raise ValueError(
                "HDF5 datasets 'fc3_cutoff' and 'cutoff' conflict."
            )
        if selected_cutoff is None:
            selected_cutoff = alias_cutoff
            cutoff_source = "dataset:cutoff"
        else:
            cutoff_source = "dataset:fc3_cutoff+cutoff"

    source_attributes = {
        "file": file_attrs,
        "dataset": dataset_attrs,
        "datasets": object_attrs,
    }

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
        "source_attributes": source_attributes,
        "units_source": units_source,
        "compact_source": compact_source,
        "hdf5_compact_hint": attribute_compact,
        "p2s_map_source": p2s_source,
        "s2p_map_source": s2p_source,
        "version": version,
        "version_source": version_source,
        "fc3_nonzero_indices_present": file_nonzero_indices is not None,
        "fc3_nonzero_indices_shape": (
            None
            if file_nonzero_indices is None
            else tuple(int(value) for value in file_nonzero_indices.shape)
        ),
        "fc3_nonzero_indices_dtype": (
            None
            if file_nonzero_indices is None
            else str(file_nonzero_indices.dtype)
        ),
        "fc3_nonzero_indices_source": (
            "dataset:fc3_nonzero_indices"
            if file_nonzero_indices is not None
            else "absent"
        ),
        "fc3_cutoff": selected_cutoff,
        "fc3_cutoff_source": cutoff_source,
        "reader_transforms": (),
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
        version=version,
        fc3_nonzero_indices=file_nonzero_indices,
        fc3_cutoff=selected_cutoff,
        source_attributes=source_attributes,
    )


__all__ = ["read_phono3py_fc3"]
