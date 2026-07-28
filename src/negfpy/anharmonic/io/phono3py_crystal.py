"""Read detached crystal mappings from phono3py YAML inputs.

The optional phono3py dependency is imported only when the default loader is
used.  The resulting :class:`~negfpy.anharmonic.crystal.CrystalMapping`
contains copied arrays and immutable metadata; no live phonopy or phono3py
objects are retained.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..crystal import CrystalCell, CrystalMapping


Loader = Callable[..., Any]
Purpose = Literal["fc2", "fc3"]
_STRUCTURE_ATOL = 1.0e-10


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_attribute(obj: Any, name: str, *, context: str) -> Any:
    try:
        value = getattr(obj, name)
    except AttributeError as exc:
        raise ValueError(f"{context} is missing required attribute {name!r}.") from exc
    if value is None:
        raise ValueError(f"{context} has no usable {name}.")
    return value


def _cell_from_phonopy(value: Any, *, name: str) -> CrystalCell:
    lattice = _required_attribute(value, "cell", context=name)
    scaled_positions = _required_attribute(
        value,
        "scaled_positions",
        context=name,
    )
    cartesian_positions = _required_attribute(
        value,
        "positions",
        context=name,
    )
    masses = _required_attribute(value, "masses", context=name)
    symbols = _required_attribute(value, "symbols", context=name)
    try:
        return CrystalCell(
            lattice=lattice,
            scaled_positions=scaled_positions,
            cartesian_positions=cartesian_positions,
            masses=masses,
            symbols=symbols,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is inconsistent: {exc}") from exc


def _cells_equivalent(left: CrystalCell, right: CrystalCell) -> bool:
    if left.n_atoms != right.n_atoms:
        return False
    if left.symbols != right.symbols:
        return False
    if not np.allclose(
        left.lattice,
        right.lattice,
        rtol=0.0,
        atol=_STRUCTURE_ATOL,
    ):
        return False
    position_delta = left.scaled_positions - right.scaled_positions
    position_delta -= np.rint(position_delta)
    if not np.allclose(
        position_delta,
        0.0,
        rtol=0.0,
        atol=_STRUCTURE_ATOL,
    ):
        return False
    return bool(
        np.allclose(
            left.masses,
            right.masses,
            rtol=0.0,
            atol=_STRUCTURE_ATOL,
        )
    )


def _integer_array(values: Any, *, name: str, ndim: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers.")
    array = np.array(raw, dtype=np.int64, order="C", copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional.")
    return array


def _dense_p2p(
    primitive: Any,
    *,
    p2s: np.ndarray,
    n_supercell_atoms: int,
    name: str,
) -> np.ndarray:
    sparse = _required_attribute(primitive, "p2p_map", context=name)
    if not isinstance(sparse, Mapping):
        raise ValueError(f"{name}.p2p_map must be a mapping.")

    normalized: dict[int, int] = {}
    for raw_key, raw_value in sparse.items():
        if isinstance(raw_key, (bool, np.bool_)) or not isinstance(
            raw_key,
            (int, np.integer),
        ):
            raise ValueError(f"{name}.p2p_map keys must be integer indices.")
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value,
            (int, np.integer),
        ):
            raise ValueError(f"{name}.p2p_map values must be primitive ordinals.")
        normalized[int(raw_key)] = int(raw_value)

    expected_keys = set(int(value) for value in p2s)
    if set(normalized) != expected_keys:
        raise ValueError(
            f"{name}.p2p_map keys must exactly equal the representative "
            "supercell indices in p2s; missing or additional keys are not "
            "inferred or repaired."
        )
    dense = np.full(n_supercell_atoms, -1, dtype=np.int64)
    for primitive_index, representative in enumerate(p2s):
        representative_index = int(representative)
        if representative_index < 0 or representative_index >= n_supercell_atoms:
            raise ValueError(
                f"{name}.p2s_map contains an out-of-range supercell index."
            )
        if normalized[representative_index] != primitive_index:
            raise ValueError(
                f"{name}.p2p_map conflicts with p2s_map primitive ordering."
            )
        dense[representative_index] = primitive_index
    return dense


def _shortest_vectors(primitive: Any, *, name: str) -> tuple[Any, Any]:
    method = _required_attribute(
        primitive,
        "get_smallest_vectors",
        context=name,
    )
    if not callable(method):
        raise ValueError(f"{name}.get_smallest_vectors must be callable.")
    values = method()
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(
            f"{name}.get_smallest_vectors must return "
            "(shortest_vectors, multiplicity)."
        )
    return values[0], values[1]


def _representatives_match(
    *,
    primitive: CrystalCell,
    supercell: CrystalCell,
    p2s: np.ndarray,
    name: str,
) -> None:
    if p2s.shape != (primitive.n_atoms,):
        raise ValueError(
            f"{name}.p2s_map must contain one representative per primitive atom."
        )
    if np.any((p2s < 0) | (p2s >= supercell.n_atoms)):
        raise ValueError(f"{name}.p2s_map contains an out-of-range index.")
    representative_masses = supercell.masses[p2s]
    if not np.allclose(
        primitive.masses,
        representative_masses,
        rtol=0.0,
        atol=_STRUCTURE_ATOL,
    ):
        raise ValueError(
            f"{name} primitive and supercell representative masses conflict."
        )
    representative_symbols = tuple(supercell.symbols[int(index)] for index in p2s)
    if primitive.symbols != representative_symbols:
        raise ValueError(
            f"{name} primitive and supercell representative symbols conflict."
        )


def _symmetry_permutations(value: Any, *, name: str) -> Any:
    return _required_attribute(value, "atomic_permutations", context=name)


def _loader_identity(loader: Loader) -> tuple[str, str]:
    module = getattr(loader, "__module__", loader.__class__.__module__)
    qualname = getattr(loader, "__qualname__", loader.__class__.__qualname__)
    return str(module), str(qualname)


def _resolve_loader(loader: Loader | None) -> tuple[Loader, str | None]:
    if loader is not None:
        if not callable(loader):
            raise TypeError("loader must be callable.")
        return loader, None
    try:
        import phono3py
    except ImportError as exc:  # pragma: no cover - minimal optional environment
        raise ImportError(
            "read_phono3py_crystal_mapping requires phono3py when no loader "
            "is supplied. Install the pinned material-oracle environment."
        ) from exc
    return phono3py.load, str(phono3py.__version__)


def _instance_version(instance: Any, fallback: str | None) -> str:
    try:
        version = getattr(instance, "version")
    except AttributeError:
        version = None
    if isinstance(version, str) and version.strip():
        return version.strip()
    if fallback is not None and fallback.strip():
        return fallback.strip()
    return "unreported"


def _verified_fc2_matrix_fallback(
    instance: Any,
    *,
    phonon_primitive: CrystalCell,
    phonon_supercell: CrystalCell,
    phonon_permutations: np.ndarray,
) -> Any:
    ordinary_primitive_obj = _required_attribute(
        instance,
        "primitive",
        context="phono3py object",
    )
    ordinary_supercell_obj = _required_attribute(
        instance,
        "supercell",
        context="phono3py object",
    )
    ordinary_symmetry = _required_attribute(
        instance,
        "symmetry",
        context="phono3py object",
    )
    ordinary_primitive = _cell_from_phonopy(
        ordinary_primitive_obj,
        name="primitive",
    )
    ordinary_supercell = _cell_from_phonopy(
        ordinary_supercell_obj,
        name="supercell",
    )
    ordinary_permutations = _integer_array(
        _symmetry_permutations(ordinary_symmetry, name="symmetry"),
        name="symmetry.atomic_permutations",
        ndim=2,
    )
    if (
        not _cells_equivalent(phonon_primitive, ordinary_primitive)
        or not _cells_equivalent(phonon_supercell, ordinary_supercell)
        or not np.array_equal(phonon_permutations, ordinary_permutations)
    ):
        raise ValueError(
            "phonon_supercell_matrix is None, but the phonon structures or "
            "phonon supercell symmetry do not match the ordinary FC3 "
            "structures; the FC2 matrix cannot be inferred or repaired."
        )
    return _required_attribute(
        instance,
        "supercell_matrix",
        context="phono3py object",
    )


def read_phono3py_crystal_mapping(
    path: str | Path,
    *,
    purpose: Purpose = "fc3",
    is_nac: bool = False,
    loader: Loader | None = None,
) -> CrystalMapping:
    """Load and detach a phono3py primitive/supercell mapping.

    Parameters
    ----------
    path
        A phono3py YAML-like input containing the crystal structures and
        matrices.
    purpose
        ``"fc3"`` selects the ordinary primitive, supercell and supercell
        symmetry.  ``"fc2"`` selects the corresponding ``phonon_*`` objects.
    is_nac
        NAC loading is disabled by default.  Set this explicitly to ``True``
        only when the mapping provenance must include NAC-loaded input.
    loader
        Optional callable with the :func:`phono3py.load` interface, intended
        for isolated tests.  No object returned by it is retained.

    Notes
    -----
    Force constants are never produced or loaded by this operation:
    ``produce_fc=False`` is mandatory.
    """

    if purpose not in {"fc2", "fc3"}:
        raise ValueError("purpose must be either 'fc2' or 'fc3'.")
    if not isinstance(is_nac, (bool, np.bool_)):
        raise TypeError("is_nac must be a boolean.")
    is_nac = bool(is_nac)

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Phono3py crystal mapping source does not exist: {source_path}"
        )

    selected_loader, imported_version = _resolve_loader(loader)
    instance = selected_loader(
        phono3py_yaml=source_path,
        produce_fc=False,
        is_nac=is_nac,
        log_level=0,
    )
    if instance is None:
        raise ValueError("phono3py loader returned no object.")

    primitive_matrix = _required_attribute(
        instance,
        "primitive_matrix",
        context="phono3py object",
    )

    if purpose == "fc3":
        primitive_obj = _required_attribute(
            instance,
            "primitive",
            context="phono3py object",
        )
        supercell_obj = _required_attribute(
            instance,
            "supercell",
            context="phono3py object",
        )
        symmetry_obj = _required_attribute(
            instance,
            "symmetry",
            context="phono3py object",
        )
        supercell_matrix = _required_attribute(
            instance,
            "supercell_matrix",
            context="phono3py object",
        )
        structure_source = "ordinary"
        supercell_matrix_source = "supercell_matrix"
    else:
        primitive_obj = _required_attribute(
            instance,
            "phonon_primitive",
            context="phono3py object",
        )
        supercell_obj = _required_attribute(
            instance,
            "phonon_supercell",
            context="phono3py object",
        )
        symmetry_obj = _required_attribute(
            instance,
            "phonon_supercell_symmetry",
            context="phono3py object",
        )
        try:
            supercell_matrix = getattr(instance, "phonon_supercell_matrix")
        except AttributeError as exc:
            raise ValueError(
                "phono3py object is missing required attribute "
                "'phonon_supercell_matrix'."
            ) from exc
        structure_source = "phonon"
        supercell_matrix_source = "phonon_supercell_matrix"

    primitive = _cell_from_phonopy(
        primitive_obj,
        name=f"{structure_source}_primitive",
    )
    supercell = _cell_from_phonopy(
        supercell_obj,
        name=f"{structure_source}_supercell",
    )
    permutations = _integer_array(
        _symmetry_permutations(
            symmetry_obj,
            name=(
                "symmetry"
                if purpose == "fc3"
                else "phonon_supercell_symmetry"
            ),
        ),
        name="atomic_permutations",
        ndim=2,
    )

    if purpose == "fc2" and supercell_matrix is None:
        supercell_matrix = _verified_fc2_matrix_fallback(
            instance,
            phonon_primitive=primitive,
            phonon_supercell=supercell,
            phonon_permutations=permutations,
        )
        supercell_matrix_source = "verified_fc3_supercell_matrix_fallback"

    p2s = _integer_array(
        _required_attribute(
            primitive_obj,
            "p2s_map",
            context=f"{structure_source}_primitive",
        ),
        name="p2s_map",
        ndim=1,
    )
    s2p = _integer_array(
        _required_attribute(
            primitive_obj,
            "s2p_map",
            context=f"{structure_source}_primitive",
        ),
        name="s2p_map",
        ndim=1,
    )
    p2p = _dense_p2p(
        primitive_obj,
        p2s=p2s,
        n_supercell_atoms=supercell.n_atoms,
        name=f"{structure_source}_primitive",
    )
    _representatives_match(
        primitive=primitive,
        supercell=supercell,
        p2s=p2s,
        name=structure_source,
    )
    shortest_vectors, multiplicity = _shortest_vectors(
        primitive_obj,
        name=f"{structure_source}_primitive",
    )
    mixture_weights = getattr(primitive_obj, "mixture_weights", None)
    if mixture_weights is not None:
        raise NotImplementedError(
            "Phonopy site-mixture mass normalization is not represented by "
            "CrystalMapping. Such mappings are excluded until separately "
            "implemented and validated."
        )
    translation_permutations = _integer_array(
        _required_attribute(
            primitive_obj,
            "atomic_permutations",
            context=f"{structure_source}_primitive",
        ),
        name="translation_permutations",
        ndim=2,
    )

    loader_module, loader_name = _loader_identity(selected_loader)
    provenance = {
        "reader": (
            "negfpy.anharmonic.io.phono3py_crystal."
            "read_phono3py_crystal_mapping"
        ),
        "source_path": str(source_path),
        "source_sha256": _sha256_file(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "purpose": purpose,
        "structure_source": structure_source,
        "supercell_matrix_source": supercell_matrix_source,
        "produce_fc": False,
        "is_nac": is_nac,
        "loader_module": loader_module,
        "loader_name": loader_name,
        "phono3py_version": _instance_version(instance, imported_version),
        "mass_units": "amu",
        "site_mixture_present": False,
        "translation_permutation_count": int(
            translation_permutations.shape[0]
        ),
    }
    return CrystalMapping(
        primitive_matrix=primitive_matrix,
        supercell_matrix=supercell_matrix,
        primitive=primitive,
        supercell=supercell,
        p2s=p2s,
        s2p=s2p,
        p2p=p2p,
        shortest_vectors=shortest_vectors,
        multiplicity=multiplicity,
        atomic_permutations=permutations,
        purpose=purpose,
        translation_permutations=translation_permutations,
        provenance=provenance,
    )


__all__ = ["read_phono3py_crystal_mapping"]
