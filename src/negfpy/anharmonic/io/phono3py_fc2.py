"""Exact reader for phono3py ``fc2.hdf5`` files.

``h5py`` remains optional.  This module imports it only when the reader is
called, so ordinary NEGFPy imports do not acquire an HDF5 dependency.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ..fc2 import FC2Data


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
        if value.ndim == 0:
            return _attribute_value(value.item())
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


def _object_dataset(handle: Any, name: str, h5py: Any) -> Any | None:
    if name not in handle:
        return None
    obj = handle[name]
    if not isinstance(obj, h5py.Dataset):
        raise ValueError(f"HDF5 object {name!r} is not a dataset.")
    return obj


def _optional_array(
    handle: Any,
    name: str,
    h5py: Any,
) -> tuple[Array | None, dict[str, Any]]:
    dataset = _object_dataset(handle, name, h5py)
    if dataset is None:
        return None, {}
    return np.asarray(dataset[...]), _attrs(dataset)


def _scalar_value(dataset: Any, *, name: str) -> Any:
    value = np.asarray(dataset[...])
    if value.size != 1:
        raise ValueError(f"HDF5 metadata dataset {name!r} must contain one value.")
    return _attribute_value(value.reshape(-1)[0])


def _text_value(value: Any) -> str | None:
    converted = _attribute_value(value)
    if isinstance(converted, str) and converted.strip():
        return converted.strip()
    if isinstance(converted, tuple) and len(converted) == 1:
        candidate = converted[0]
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _attribute_text(
    dataset_attrs: dict[str, Any],
    file_attrs: dict[str, Any],
    names: tuple[str, ...],
) -> tuple[str | None, str | None]:
    for name in names:
        if name in dataset_attrs:
            text = _text_value(dataset_attrs[name])
            if text is not None:
                return text, f"dataset_attr:{name}"
    for name in names:
        if name in file_attrs:
            text = _text_value(file_attrs[name])
            if text is not None:
                return text, f"file_attr:{name}"
    return None, None


def _official_layout(shape: tuple[int, ...]) -> tuple[bool, str]:
    if len(shape) != 4:
        raise ValueError(
            "Phono3py FC2 dataset must have official shape "
            "(Ns, Ns, 3, 3) or (Np, Ns, 3, 3)."
        )
    n_first, n_supercell, n_cartesian_1, n_cartesian_2 = shape
    if (
        n_first < 1
        or n_supercell < 1
        or (n_cartesian_1, n_cartesian_2) != (3, 3)
        or n_first > n_supercell
    ):
        raise ValueError(
            "Phono3py FC2 dataset must have official shape "
            "(Ns, Ns, 3, 3) or (Np, Ns, 3, 3) with 0 < Np < Ns."
        )
    if n_first == n_supercell:
        return False, "shape_equal_full"
    return True, "shape_unequal_compact"


def _select_mapping(
    *,
    name: str,
    explicit: Any | None,
    stored: Array | None,
) -> tuple[Array | None, str]:
    if explicit is None:
        if stored is None:
            return None, "absent"
        return stored, f"dataset:{name}"

    explicit_array = np.asarray(explicit)
    if stored is not None:
        if not np.array_equal(explicit_array, stored):
            raise ValueError(
                f"Explicit {name} conflicts with HDF5 dataset {name}; "
                "source mappings cannot be overridden."
            )
        return stored, f"argument_verified_against_dataset:{name}"
    return explicit_array, "argument"


def _metadata_dataset(
    handle: Any,
    *,
    name: str,
    h5py: Any,
) -> tuple[Any | None, dict[str, Any]]:
    dataset = _object_dataset(handle, name, h5py)
    if dataset is None:
        return None, {}
    return _scalar_value(dataset, name=name), _attrs(dataset)


def read_phono3py_fc2(
    path: str | Path,
    masses: Any,
    *,
    p2s_map: Any | None = None,
    s2p_map: Any | None = None,
    units: str | None = None,
    mass_units: str = "amu",
    compact: bool | None = None,
    dataset: str = "force_constants",
) -> FC2Data:
    """Read a raw full or compact phono3py FC2 tensor.

    Layout classification is shape-authoritative: equal first and second atom
    dimensions are the full/equivalent representation, regardless of maps or
    HDF5 hints.  Explicit maps are accepted only when the file omits them or
    when they exactly match the stored maps.  Values are never converted,
    symmetrized, corrected, or remapped.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - tested with import blocking
        raise ImportError(
            "read_phono3py_fc2 requires the optional 'h5py' package to "
            "read fc2.hdf5 files."
        ) from exc

    if h5py is None or not hasattr(h5py, "File"):  # defensive for mocked imports
        raise ImportError(
            "read_phono3py_fc2 requires the optional 'h5py' package to "
            "read fc2.hdf5 files."
        )

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Phono3py FC2 file does not exist: {source_path}")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty HDF5 dataset name.")

    with h5py.File(source_path, "r") as handle:
        if dataset not in handle:
            raise KeyError(
                f"HDF5 dataset {dataset!r} was not found in {source_path}."
            )
        fc2_dataset = handle[dataset]
        if not isinstance(fc2_dataset, h5py.Dataset):
            raise ValueError(f"HDF5 object {dataset!r} is not a dataset.")

        values = np.asarray(fc2_dataset[...])
        dataset_attrs = _attrs(fc2_dataset)
        file_attrs = _attrs(handle)
        selected_compact, layout_source = _official_layout(
            tuple(int(value) for value in values.shape)
        )

        file_p2s, p2s_attrs = _optional_array(handle, "p2s_map", h5py)
        file_s2p, s2p_attrs = _optional_array(handle, "s2p_map", h5py)
        physical_unit, physical_unit_attrs = _metadata_dataset(
            handle,
            name="physical_unit",
            h5py=h5py,
        )
        source_version, version_attrs = _metadata_dataset(
            handle,
            name="version",
            h5py=h5py,
        )
        source_cutoff, cutoff_attrs = _metadata_dataset(
            handle,
            name="cutoff",
            h5py=h5py,
        )

        dataset_inventory = {}
        for name, obj in handle.items():
            if isinstance(obj, h5py.Dataset):
                dataset_inventory[str(name)] = {
                    "shape": tuple(int(value) for value in obj.shape),
                    "dtype": str(obj.dtype),
                }

    if compact is not None:
        if not isinstance(compact, (bool, np.bool_)):
            raise TypeError("compact must be a boolean or None.")
        if bool(compact) != selected_compact:
            raise ValueError(
                "Explicit compact classification conflicts with the official "
                "FC2 shape classification."
            )

    selected_p2s, p2s_source = _select_mapping(
        name="p2s_map",
        explicit=p2s_map,
        stored=file_p2s,
    )
    selected_s2p, s2p_source = _select_mapping(
        name="s2p_map",
        explicit=s2p_map,
        stored=file_s2p,
    )

    source_unit_text = _text_value(physical_unit)
    source_unit_source = (
        "dataset:physical_unit" if source_unit_text is not None else None
    )
    if source_unit_text is None:
        source_unit_text, source_unit_source = _attribute_text(
            dataset_attrs,
            file_attrs,
            ("physical_unit", "units", "unit"),
        )
    if units is not None:
        selected_units = units
        units_source = "argument"
    elif source_unit_text is not None:
        selected_units = source_unit_text
        units_source = source_unit_source
    else:
        selected_units = "eV/angstrom^2"
        units_source = "phono3py_documented_default"

    version_text = _text_value(source_version)
    version_source: str | None = None
    if source_version is not None:
        if version_text is None:
            version_text = str(source_version)
        version_source = "dataset:version"
    if version_text is None:
        version_text, version_source = _attribute_text(
            dataset_attrs,
            file_attrs,
            ("version",),
        )

    cutoff: float | None = None
    if source_cutoff is not None:
        if isinstance(source_cutoff, (bool, np.bool_)):
            raise ValueError("HDF5 cutoff metadata must be a finite real scalar.")
        try:
            cutoff = float(source_cutoff)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HDF5 cutoff metadata must be a finite real scalar."
            ) from exc
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError(
                "HDF5 cutoff metadata must be a non-negative finite real scalar."
            )

    source_sha256 = _sha256_file(source_path)
    auxiliary_attrs = {
        "p2s_map": p2s_attrs,
        "s2p_map": s2p_attrs,
        "physical_unit": physical_unit_attrs,
        "version": version_attrs,
        "cutoff": cutoff_attrs,
    }
    provenance = {
        "format": "phono3py_fc2_hdf5",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "sha256": source_sha256,
        "source_dataset": dataset,
        "source_dataset_shape": tuple(int(value) for value in values.shape),
        "source_dataset_dtype": str(values.dtype),
        "source_dataset_attrs": dataset_attrs,
        "dataset_attrs": dataset_attrs,
        "source_file_attrs": file_attrs,
        "source_auxiliary_dataset_attrs": auxiliary_attrs,
        "source_dataset_inventory": dataset_inventory,
        "source_physical_unit": source_unit_text,
        "source_version": version_text,
        "source_cutoff": cutoff,
        "units_source": units_source,
        "version_source": version_source or "absent",
        "cutoff_source": "dataset:cutoff" if cutoff is not None else "absent",
        "layout_source": layout_source,
        "p2s_map_source": p2s_source,
        "s2p_map_source": s2p_source,
    }
    return FC2Data(
        values=values,
        masses=masses,
        units=selected_units,
        compact=selected_compact,
        p2s_map=selected_p2s,
        s2p_map=selected_s2p,
        version=version_text,
        cutoff=cutoff,
        provenance=provenance,
        mass_units=mass_units,
    )


__all__ = ["read_phono3py_fc2"]
