"""Deterministic, provenance-bound sampling of nonzero silicon ``|V3|^2``.

The production silicon oracle compares gauge-invariant sums over complete
degenerate mode subspaces.  This module chooses those blocks *from the
independent phono3py reference*, before a NEGFPy interaction tensor exists.
Consequently, a candidate calculation cannot influence which coordinates are
tested.

Selection is deliberately stricter than random scalar sampling:

* exact promoted blocks are deduplicated before selection;
* a robust nonzero floor is derived from the reference itself;
* every populated target-region/branch/process stratum contributes at least
  two blocks;
* Normal and Umklapp low/median/high strength quantiles and the global maximum
  are forced into the sample;
* every archived target q point and target band is represented; and
* all empty strata and all consumed-input identities are recorded.

The JSON schema and strict reader live in :mod:`silicon_oracle_validation`.
Keeping one reader avoids two subtly incompatible release-provenance formats.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phono3py_reference import (
    Phono3pyReferenceSchemaError,
    Phono3pySiliconReference,
)
from .momentum import (
    UniformBZMesh,
    minimum_bz_reciprocal_vector,
    niggli_reduced_reciprocal_lattice,
    shortest_bz_address_images,
)
from .silicon_oracle_validation import (
    V3_SAMPLE_MANIFEST_SCHEMA,
    V3_SAMPLE_SELECTION_ALGORITHM,
    V3_SAMPLE_SELECTION_ALGORITHM_VERSION,
    V3InteractionSampleManifest,
    array_sha256,
    crystal_mapping_sha256,
    read_interaction_sample_manifest,
    write_json_atomic_exclusive,
)


_DEGENERACY_TOLERANCE_THZ = 1.0e-4
_ABSOLUTE_NONZERO_FLOOR_EV2 = 1.0e-30
_RELATIVE_NONZERO_FLOOR = 1.0e-12
_DEFAULT_MINIMUM_PROMOTED_BLOCKS = 64
_SAMPLES_PER_POPULATED_STRATUM = 2
_QUANTILES = (0.1, 0.5, 0.9)
_Q_REGIONS = ("gamma", "near", "generic")
_TARGET_CHARACTERS = ("acoustic", "optical")
_INTERNAL_CHARACTERS = ("aa", "ao", "oo")
_PROCESSES = ("normal", "umklapp")
_SHA256_HEX = frozenset("0123456789abcdef")


class V3SampleCoverageError(ValueError):
    """Raised when a reference is too small for the pinned sample contract."""


@dataclass(frozen=True, slots=True)
class _PromotedBlock:
    q0: int
    q1: int
    q2: int
    target_subspace: tuple[int, ...]
    band1_subspace: tuple[int, ...]
    band2_subspace: tuple[int, ...]
    seed_b0: int
    seed_b1: int
    seed_b2: int
    process: str
    strength_ev2: float
    stratum: str
    coordinate_sha256: str

    @property
    def key(
        self,
    ) -> tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (
            self.q0,
            self.q1,
            self.target_subspace,
            self.band1_subspace,
            self.band2_subspace,
        )

    def with_target_seed(self, band: int) -> _PromotedBlock:
        if band not in self.target_subspace:
            raise ValueError("Target seed must belong to the promoted subspace.")
        return replace(self, seed_b0=int(band))

    def json_record(self) -> dict[str, Any]:
        return {
            "q0_grg": self.q0,
            "q1_grg": self.q1,
            "expected_q2_grg": self.q2,
            "b0": self.seed_b0,
            "b1": self.seed_b1,
            "b2": self.seed_b2,
            "target_subspace": list(self.target_subspace),
            "band1_subspace": list(self.band1_subspace),
            "band2_subspace": list(self.band2_subspace),
            "process": self.process,
            "reference_invariant_block_sum_ev2": self.strength_ev2,
            "stratum": self.stratum,
        }


def _regular_file(path: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link.")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file.")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_reference(reference: Any) -> Phono3pySiliconReference:
    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    if _file_sha256(reference.path) != reference.sha256:
        raise Phono3pyReferenceSchemaError(
            "Reference HDF5 changed after it was verified."
        )
    return reference


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite scalar.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return result


def _degenerate_groups(
    frequencies: np.ndarray,
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    """Use the exact connected grouping rule used by the strict comparator."""

    groups: list[tuple[int, ...]] = []
    used: set[int] = set()
    for first in range(int(frequencies.size)):
        if first in used:
            continue
        group = [first]
        used.add(first)
        for candidate in range(first + 1, int(frequencies.size)):
            if any(
                abs(float(frequencies[index] - frequencies[candidate]))
                < tolerance
                for index in group
            ):
                group.append(candidate)
                used.add(candidate)
        groups.append(tuple(group))
    return tuple(groups)


def _branch_character(
    subspace: Sequence[int],
    *,
    n_bands: int,
) -> str:
    # A crystal has exactly three translational acoustic branches.  This
    # subsystem is intentionally silicon-specific, but retaining min(3, nb)
    # gives a clear failure mode for tiny algebra fixtures.
    acoustic_count = min(3, n_bands)
    if all(band < acoustic_count for band in subspace):
        return "acoustic"
    if all(band >= acoustic_count for band in subspace):
        return "optical"
    raise V3SampleCoverageError(
        "A degenerate subspace crosses the acoustic/optical branch boundary; "
        "the pinned silicon sampling strata are not well-defined."
    )


def _internal_character(
    first: Sequence[int],
    second: Sequence[int],
    *,
    n_bands: int,
) -> str:
    characters = {
        _branch_character(first, n_bands=n_bands),
        _branch_character(second, n_bands=n_bands),
    }
    if characters == {"acoustic"}:
        return "aa"
    if characters == {"optical"}:
        return "oo"
    return "ao"


def _q_regions(
    reference: Phono3pySiliconReference,
    *,
    primitive_lattice: Any,
    address_images: Any,
) -> Mapping[int, str]:
    """Classify q points by minimum Cartesian reciprocal-space distance."""

    periods = np.asarray(reference.d_diag, dtype=np.float64)
    lattice = np.asarray(primitive_lattice, dtype=np.float64)
    if (
        lattice.shape != (3, 3)
        or not np.all(np.isfinite(lattice))
        or np.linalg.matrix_rank(lattice) != 3
    ):
        raise ValueError(
            "fc3_mapping.primitive.lattice must be a finite nonsingular "
            "3-by-3 matrix."
        )
    reciprocal_rows = np.linalg.inv(lattice).T
    reciprocal_metric = reciprocal_rows @ reciprocal_rows.T
    if len(address_images) != reference.n_qpoints:
        raise ValueError(
            "address_images must contain one entry per reference q point."
        )
    norm_squared = np.empty(reference.n_qpoints, dtype=np.float64)
    for qpoint, images in enumerate(address_images):
        if not images:
            raise ValueError("Every q point must have at least one BZ image.")
        fractional = np.asarray(
            [image.components for image in images],
            dtype=np.float64,
        ) / periods[None, :]
        values = np.einsum(
            "qi,ij,qj->q",
            fractional,
            reciprocal_metric,
            fractional,
        )
        norm_squared[qpoint] = float(np.min(values))
    positive = norm_squared[norm_squared > 0.0]
    first_shell = math.inf if positive.size == 0 else float(np.min(positive))
    result: dict[int, str] = {}
    for q0 in (int(value) for value in reference.grid_points_grg):
        value = float(norm_squared[q0])
        if value == 0.0:
            result[q0] = "gamma"
        elif value <= first_shell * (1.0 + 1.0e-12) + 1.0e-15:
            result[q0] = "near"
        else:
            result[q0] = "generic"
    return result


def _coordinate_sha256(
    q0: int,
    q1: int,
    q2: int,
    target_subspace: tuple[int, ...],
    band1_subspace: tuple[int, ...],
    band2_subspace: tuple[int, ...],
) -> str:
    coordinate = {
        "band1_subspace": list(band1_subspace),
        "band2_subspace": list(band2_subspace),
        "q0_grg": q0,
        "q1_grg": q1,
        "q2_grg": q2,
        "target_subspace": list(target_subspace),
    }
    payload = json.dumps(
        coordinate,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _production_process(
    triplet: Any,
    *,
    address_images: Any,
    primitive_lattice: Any,
    reduced_reciprocal_lattice: Any,
) -> str:
    """Classify N/U with the exact production BZ-surface convention."""

    reciprocal_vector = minimum_bz_reciprocal_vector(
        triplet,
        address_images,
        primitive_lattice,
        reduced_reciprocal_lattice=reduced_reciprocal_lattice,
    )
    return (
        "normal"
        if reciprocal_vector == (0, 0, 0)
        else "umklapp"
    )


def _all_strata() -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{region}.{target}.{internal}.{process}"
            for region in _Q_REGIONS
            for target in _TARGET_CHARACTERS
            for internal in _INTERNAL_CHARACTERS
            for process in _PROCESSES
        )
    )


def _enumerate_promoted_blocks(
    reference: Phono3pySiliconReference,
    *,
    degeneracy_tolerance_thz: float,
    primitive_lattice: Any,
) -> tuple[_PromotedBlock, ...]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "V3 sample selection requires the optional h5py package."
        ) from exc

    frequencies = reference.frequencies_in_grg_order()
    groups_by_q = tuple(
        _degenerate_groups(row, degeneracy_tolerance_thz)
        for row in frequencies
    )
    target_band_positions = {
        int(band): position
        for position, band in enumerate(reference.band_indices)
    }
    selected_target_bands = set(target_band_positions)
    periods = np.asarray(reference.d_diag, dtype=np.int64)
    mesh = UniformBZMesh(reference.mesh)
    reduced_reciprocal_lattice = niggli_reduced_reciprocal_lattice(
        primitive_lattice,
    )
    address_images = shortest_bz_address_images(
        mesh,
        primitive_lattice,
        reduced_reciprocal_lattice=reduced_reciprocal_lattice,
    )
    regions = _q_regions(
        reference,
        primitive_lattice=primitive_lattice,
        address_images=address_images,
    )
    result: list[_PromotedBlock] = []
    seen: set[
        tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = set()

    with h5py.File(reference.path, "r") as h5:
        for q0_value in reference.grid_points_grg:
            q0 = int(q0_value)
            group = h5[reference.target_groups[q0]]
            triplets_bzg = np.asarray(group["triplets_bzg"], dtype=np.int64)
            triplets_grg = reference.bzg2grg[triplets_bzg]
            triplet_addresses = np.asarray(
                group["triplet_addresses"],
                dtype=np.int64,
            )
            strengths = np.asarray(
                group["interaction_strength"],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(strengths)) or np.any(strengths < 0.0):
                raise Phono3pyReferenceSchemaError(
                    f"{group.name}/interaction_strength must be finite and "
                    "nonnegative."
                )
            if triplet_addresses.shape != (reference.n_qpoints, 3, 3):
                raise Phono3pyReferenceSchemaError(
                    f"{group.name}/triplet_addresses has an invalid shape."
                )
            route_for_q1 = {
                int(route[1]): position
                for position, route in enumerate(triplets_grg)
            }
            if len(route_for_q1) != reference.n_qpoints:
                raise Phono3pyReferenceSchemaError(
                    f"{group.name} has duplicate or missing q1 routes."
                )
            production_routes = mesh.triplets_at_q(q0)

            for q1 in range(reference.n_qpoints):
                route_position = route_for_q1[q1]
                q2 = int(triplets_grg[route_position, 2])
                address_sum = np.sum(
                    triplet_addresses[route_position],
                    axis=0,
                    dtype=np.int64,
                )
                if np.any(np.mod(address_sum, periods) != 0):
                    raise Phono3pyReferenceSchemaError(
                        f"{group.name} contains a triplet that violates "
                        "mesh momentum conservation."
                    )
                production_triplet = production_routes[q1]
                if production_triplet.q2_index != q2:
                    raise Phono3pyReferenceSchemaError(
                        f"{group.name} differs from the production q2 route."
                    )
                process = _production_process(
                    production_triplet,
                    address_images=address_images,
                    primitive_lattice=primitive_lattice,
                    reduced_reciprocal_lattice=(
                        reduced_reciprocal_lattice
                    ),
                )
                values = strengths[route_position]
                for target_subspace in groups_by_q[q0]:
                    if not set(target_subspace).issubset(selected_target_bands):
                        # An incomplete degenerate target block is gauge
                        # dependent and therefore cannot enter the manifest.
                        continue
                    target_axis = tuple(
                        target_band_positions[band]
                        for band in target_subspace
                    )
                    target_character = _branch_character(
                        target_subspace,
                        n_bands=reference.n_bands,
                    )
                    for band1_subspace in groups_by_q[q1]:
                        for band2_subspace in groups_by_q[q2]:
                            block_values = values[
                                np.ix_(
                                    target_axis,
                                    band1_subspace,
                                    band2_subspace,
                                )
                            ]
                            strength = math.fsum(
                                float(value)
                                for value in block_values.ravel(order="C")
                            )
                            internal = _internal_character(
                                band1_subspace,
                                band2_subspace,
                                n_bands=reference.n_bands,
                            )
                            stratum = (
                                f"{regions[q0]}.{target_character}."
                                f"{internal}.{process}"
                            )
                            key = (
                                q0,
                                q1,
                                target_subspace,
                                band1_subspace,
                                band2_subspace,
                            )
                            if key in seen:
                                raise Phono3pyReferenceSchemaError(
                                    "Reference routes produced a duplicate "
                                    "gauge-invariant promoted block."
                                )
                            seen.add(key)
                            result.append(
                                _PromotedBlock(
                                    q0=q0,
                                    q1=q1,
                                    q2=q2,
                                    target_subspace=target_subspace,
                                    band1_subspace=band1_subspace,
                                    band2_subspace=band2_subspace,
                                    seed_b0=target_subspace[0],
                                    seed_b1=band1_subspace[0],
                                    seed_b2=band2_subspace[0],
                                    process=process,
                                    strength_ev2=strength,
                                    stratum=stratum,
                                    coordinate_sha256=_coordinate_sha256(
                                        q0,
                                        q1,
                                        q2,
                                        target_subspace,
                                        band1_subspace,
                                        band2_subspace,
                                    ),
                                )
                            )
    if _file_sha256(reference.path) != reference.sha256:
        raise Phono3pyReferenceSchemaError(
            "Reference HDF5 changed while V3 samples were selected."
        )
    return tuple(result)


def _quantile_member(
    candidates: Sequence[_PromotedBlock],
    quantile: float,
) -> _PromotedBlock:
    ordered = sorted(
        candidates,
        key=lambda item: (
            item.strength_ev2,
            item.coordinate_sha256,
            item.key,
        ),
    )
    index = int(math.floor(quantile * (len(ordered) - 1) + 0.5))
    return ordered[index]


def _select_blocks(
    candidates: Sequence[_PromotedBlock],
    *,
    reference: Phono3pySiliconReference,
    minimum_promoted_blocks: int,
) -> tuple[tuple[_PromotedBlock, ...], tuple[str, ...], tuple[str, ...]]:
    if not candidates:
        raise V3SampleCoverageError(
            "The reference contains no promoted interaction blocks."
        )
    global_max = max(item.strength_ev2 for item in candidates)
    floor = max(
        _ABSOLUTE_NONZERO_FLOOR_EV2,
        _RELATIVE_NONZERO_FLOOR * global_max,
    )
    viable = tuple(item for item in candidates if item.strength_ev2 >= floor)
    if len(viable) < minimum_promoted_blocks:
        raise V3SampleCoverageError(
            "Reference has only "
            f"{len(viable)} promoted blocks at or above the robust nonzero "
            f"floor {floor:.17g} eV^2; at least "
            f"{minimum_promoted_blocks} are required."
        )

    by_stratum: dict[str, list[_PromotedBlock]] = {}
    by_process: dict[str, list[_PromotedBlock]] = {}
    for candidate in viable:
        by_stratum.setdefault(candidate.stratum, []).append(candidate)
        by_process.setdefault(candidate.process, []).append(candidate)
    missing_processes = [
        process for process in _PROCESSES if process not in by_process
    ]
    if missing_processes:
        raise V3SampleCoverageError(
            "Reference does not contain robustly nonzero blocks for required "
            f"processes: {missing_processes}."
        )
    sparse_strata = {
        name: len(values)
        for name, values in by_stratum.items()
        if len(values) < _SAMPLES_PER_POPULATED_STRATUM
    }
    if sparse_strata:
        raise V3SampleCoverageError(
            "Every populated stratum requires two distinct promoted blocks; "
            f"insufficient strata: {sparse_strata}."
        )
    for process, values in by_process.items():
        quantile_keys = {
            _quantile_member(values, quantile).key for quantile in _QUANTILES
        }
        if len(quantile_keys) != len(_QUANTILES):
            raise V3SampleCoverageError(
                f"{process} has insufficient distinct blocks to represent "
                "low, median, and high strength quantiles."
            )

    selected: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        _PromotedBlock,
    ] = {}

    def add(candidate: _PromotedBlock) -> bool:
        if candidate.key in selected:
            return False
        selected[candidate.key] = candidate
        return True

    by_coordinate = sorted(
        viable,
        key=lambda item: (item.coordinate_sha256, item.key),
    )

    # Exactly the first two canonical-hash-ranked coordinates are forced for
    # every populated stratum. Later coverage requirements may add more.
    for name in sorted(by_stratum):
        ordered = sorted(
            by_stratum[name],
            key=lambda item: (item.coordinate_sha256, item.key),
        )
        for candidate in ordered[:_SAMPLES_PER_POPULATED_STRATUM]:
            add(candidate)

    # Force the global maximum.  Hash ordering resolves an exact-strength tie.
    maximum = min(
        (
            item
            for item in viable
            if item.strength_ev2 == global_max
        ),
        key=lambda item: (item.coordinate_sha256, item.key),
    )
    add(maximum)

    # Force low, median, and high robust-strength quantiles for N and U.
    for process in _PROCESSES:
        for quantile in _QUANTILES:
            add(_quantile_member(by_process[process], quantile))

    # Coverage seeds are assigned only after the hash-stratified and quantile
    # blocks are frozen. A degenerate target subspace can therefore use
    # different scalar members on additional unique blocks without changing
    # which two blocks represented its stratum.
    observed_targets = {item.q0 for item in selected.values()}
    for q0 in (int(value) for value in reference.grid_points_grg):
        if q0 in observed_targets:
            continue
        candidate = next(
            (
                item
                for item in by_coordinate
                if item.q0 == q0 and item.key not in selected
            ),
            None,
        )
        if candidate is None:
            raise V3SampleCoverageError(
                f"No distinct robustly nonzero block covers target q={q0}."
            )
        add(candidate)
        observed_targets.add(q0)
    observed_bands = {item.seed_b0 for item in selected.values()}
    for band_value in reference.band_indices:
        band = int(band_value)
        if band in observed_bands:
            continue
        candidate = next(
            (
                item
                for item in by_coordinate
                if band in item.target_subspace and item.key not in selected
            ),
            None,
        )
        if candidate is None:
            raise V3SampleCoverageError(
                "No distinct robustly nonzero promoted block can cover "
                f"archived target band {band}."
            )
        add(candidate.with_target_seed(band))
        observed_bands.add(band)

    for candidate in by_coordinate:
        if len(selected) >= minimum_promoted_blocks:
            break
        add(candidate)
    if len(selected) < minimum_promoted_blocks:
        raise V3SampleCoverageError(
            "Deterministic selection could not meet the minimum unique "
            "promoted-block coverage."
        )

    selected_blocks = tuple(
        sorted(
            selected.values(),
            key=lambda item: (item.coordinate_sha256, item.key),
        )
    )
    if len({item.key for item in selected_blocks}) != len(selected_blocks):
        raise AssertionError("Selector emitted a duplicate promoted block.")
    required_strata = tuple(sorted(by_stratum))
    empty_strata = tuple(
        name for name in _all_strata() if name not in by_stratum
    )
    return selected_blocks, required_strata, empty_strata


def _input_hashes(
    *,
    reference: Phono3pySiliconReference,
    fc2_compact_hdf5_path: str | os.PathLike[str],
    fc2_full_hdf5_path: str | os.PathLike[str],
    fc3_compact_hdf5_path: str | os.PathLike[str],
    fc3_full_hdf5_path: str | os.PathLike[str],
    phono3py_yaml_path: str | os.PathLike[str],
    input_manifest_path: str | os.PathLike[str],
) -> dict[str, str]:
    paths = {
        "fc2_compact_hdf5_sha256": _regular_file(
            fc2_compact_hdf5_path,
            name="fc2_compact_hdf5_path",
        ),
        "fc2_full_hdf5_sha256": _regular_file(
            fc2_full_hdf5_path,
            name="fc2_full_hdf5_path",
        ),
        "fc3_compact_hdf5_sha256": _regular_file(
            fc3_compact_hdf5_path,
            name="fc3_compact_hdf5_path",
        ),
        "fc3_full_hdf5_sha256": _regular_file(
            fc3_full_hdf5_path,
            name="fc3_full_hdf5_path",
        ),
        "phono3py_yaml_sha256": _regular_file(
            phono3py_yaml_path,
            name="phono3py_yaml_path",
        ),
        "input_manifest_sha256": _regular_file(
            input_manifest_path,
            name="input_manifest_path",
        ),
    }
    return {
        "reference_hdf5_sha256": reference.sha256,
        **{name: _file_sha256(path) for name, path in paths.items()},
    }


def build_v3_sample_manifest(
    reference: Phono3pySiliconReference,
    *,
    fc2_compact_hdf5_path: str | os.PathLike[str],
    fc2_full_hdf5_path: str | os.PathLike[str],
    fc3_compact_hdf5_path: str | os.PathLike[str],
    fc3_full_hdf5_path: str | os.PathLike[str],
    phono3py_yaml_path: str | os.PathLike[str],
    input_manifest_path: str | os.PathLike[str],
    fc2_mapping: Any,
    fc3_mapping: Any,
    minimum_promoted_blocks: int = _DEFAULT_MINIMUM_PROMOTED_BLOCKS,
    degeneracy_tolerance_thz: float = _DEGENERACY_TOLERANCE_THZ,
) -> dict[str, Any]:
    """Build the canonical strict-JSON payload without writing it.

    ``reference`` must already have passed
    :func:`read_phono3py_silicon_reference`.  Its bytes are checked before and
    after selection.  All sample coordinates and invariant strengths are read
    from that artifact; no NEGFPy candidate interaction is accepted here.
    """

    reference = _verified_reference(reference)
    minimum = _positive_int(
        minimum_promoted_blocks,
        name="minimum_promoted_blocks",
    )
    if minimum < _DEFAULT_MINIMUM_PROMOTED_BLOCKS:
        raise ValueError(
            "minimum_promoted_blocks is pinned to at least 64."
        )
    degeneracy = _positive_float(
        degeneracy_tolerance_thz,
        name="degeneracy_tolerance_thz",
    )
    if degeneracy != _DEGENERACY_TOLERANCE_THZ:
        raise ValueError(
            "The silicon oracle pins degeneracy_tolerance_thz to 1e-4."
        )
    candidates = _enumerate_promoted_blocks(
        reference,
        degeneracy_tolerance_thz=degeneracy,
        primitive_lattice=fc3_mapping.primitive.lattice,
    )
    global_max = max(
        (item.strength_ev2 for item in candidates),
        default=0.0,
    )
    floor = max(
        _ABSOLUTE_NONZERO_FLOOR_EV2,
        _RELATIVE_NONZERO_FLOOR * global_max,
    )
    selected, required_strata, empty_strata = _select_blocks(
        candidates,
        reference=reference,
        minimum_promoted_blocks=minimum,
    )
    payload = {
        "schema": V3_SAMPLE_MANIFEST_SCHEMA,
        "schema_version": 1,
        "inputs": _input_hashes(
            reference=reference,
            fc2_compact_hdf5_path=fc2_compact_hdf5_path,
            fc2_full_hdf5_path=fc2_full_hdf5_path,
            fc3_compact_hdf5_path=fc3_compact_hdf5_path,
            fc3_full_hdf5_path=fc3_full_hdf5_path,
            phono3py_yaml_path=phono3py_yaml_path,
            input_manifest_path=input_manifest_path,
        ),
        "mapping_sha256": {
            "fc2": crystal_mapping_sha256(fc2_mapping),
            "fc3": crystal_mapping_sha256(fc3_mapping),
        },
        "mesh": list(reference.mesh),
        "grg2bzg_sha256": array_sha256(reference.grg2bzg),
        "bzg2grg_sha256": array_sha256(reference.bzg2grg),
        "degeneracy_tolerance_thz": degeneracy,
        "selection_algorithm": V3_SAMPLE_SELECTION_ALGORITHM,
        "selection_algorithm_version": (
            V3_SAMPLE_SELECTION_ALGORITHM_VERSION
        ),
        "robust_nonzero_floor_ev2": floor,
        "coverage": {
            "cover_all_target_grid_points": True,
            "cover_all_target_bands": True,
            "required_processes": list(_PROCESSES),
            "required_strata": list(required_strata),
            "empty_strata": list(empty_strata),
            "minimum_promoted_blocks": minimum,
        },
        "samples": [item.json_record() for item in selected],
    }
    return payload


def read_v3_sample_manifest(
    path: str | os.PathLike[str],
    *,
    expected_payload_sha256: str | None = None,
) -> V3InteractionSampleManifest:
    """Read the shared strict schema and optionally pin its canonical bytes."""

    manifest = read_interaction_sample_manifest(path)
    if expected_payload_sha256 is not None:
        if (
            not isinstance(expected_payload_sha256, str)
            or len(expected_payload_sha256) != 64
            or any(
                character not in _SHA256_HEX
                for character in expected_payload_sha256.lower()
            )
        ):
            raise ValueError(
                "expected_payload_sha256 must be a 64-hex SHA256."
            )
        if manifest.sha256 != expected_payload_sha256.lower():
            raise ValueError(
                "V3 sample manifest does not match expected_payload_sha256."
            )
    return manifest


def write_v3_sample_manifest(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> V3InteractionSampleManifest:
    """Atomically publish one immutable canonical manifest and read it back."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping.")
    payload_sha256 = write_json_atomic_exclusive(path, payload)
    return read_v3_sample_manifest(
        path,
        expected_payload_sha256=payload_sha256,
    )


def verify_v3_sample_manifest_identity(
    manifest: V3InteractionSampleManifest,
    reference: Phono3pySiliconReference,
    *,
    fc2_compact_hdf5_path: str | os.PathLike[str],
    fc2_full_hdf5_path: str | os.PathLike[str],
    fc3_compact_hdf5_path: str | os.PathLike[str],
    fc3_full_hdf5_path: str | os.PathLike[str],
    phono3py_yaml_path: str | os.PathLike[str],
    input_manifest_path: str | os.PathLike[str],
    fc2_mapping: Any,
    fc3_mapping: Any,
) -> None:
    """Reject any input, mapping, mesh, or GRG/BZG identity mismatch."""

    if not isinstance(manifest, V3InteractionSampleManifest):
        raise TypeError("manifest must be a V3InteractionSampleManifest.")
    reference = _verified_reference(reference)
    expected_inputs = _input_hashes(
        reference=reference,
        fc2_compact_hdf5_path=fc2_compact_hdf5_path,
        fc2_full_hdf5_path=fc2_full_hdf5_path,
        fc3_compact_hdf5_path=fc3_compact_hdf5_path,
        fc3_full_hdf5_path=fc3_full_hdf5_path,
        phono3py_yaml_path=phono3py_yaml_path,
        input_manifest_path=input_manifest_path,
    )
    if dict(manifest.inputs) != expected_inputs:
        raise ValueError(
            "V3 sample manifest input hashes do not identify these files."
        )
    expected_mapping = {
        "fc2": crystal_mapping_sha256(fc2_mapping),
        "fc3": crystal_mapping_sha256(fc3_mapping),
    }
    if dict(manifest.mapping_sha256) != expected_mapping:
        raise ValueError(
            "V3 sample manifest mapping hashes do not identify these mappings."
        )
    if manifest.mesh != tuple(reference.mesh):
        raise ValueError("V3 sample manifest mesh differs from the reference.")
    if manifest.grg2bzg_sha256 != array_sha256(reference.grg2bzg):
        raise ValueError("V3 sample manifest GRG-to-BZG hash differs.")
    if manifest.bzg2grg_sha256 != array_sha256(reference.bzg2grg):
        raise ValueError("V3 sample manifest BZG-to-GRG hash differs.")


def create_v3_sample_manifest(
    reference: Phono3pySiliconReference,
    *,
    output_path: str | os.PathLike[str],
    fc2_compact_hdf5_path: str | os.PathLike[str],
    fc2_full_hdf5_path: str | os.PathLike[str],
    fc3_compact_hdf5_path: str | os.PathLike[str],
    fc3_full_hdf5_path: str | os.PathLike[str],
    phono3py_yaml_path: str | os.PathLike[str],
    input_manifest_path: str | os.PathLike[str],
    fc2_mapping: Any,
    fc3_mapping: Any,
    minimum_promoted_blocks: int = _DEFAULT_MINIMUM_PROMOTED_BLOCKS,
    degeneracy_tolerance_thz: float = _DEGENERACY_TOLERANCE_THZ,
) -> V3InteractionSampleManifest:
    """Select from the reference and atomically write the immutable manifest."""

    payload = build_v3_sample_manifest(
        reference,
        fc2_compact_hdf5_path=fc2_compact_hdf5_path,
        fc2_full_hdf5_path=fc2_full_hdf5_path,
        fc3_compact_hdf5_path=fc3_compact_hdf5_path,
        fc3_full_hdf5_path=fc3_full_hdf5_path,
        phono3py_yaml_path=phono3py_yaml_path,
        input_manifest_path=input_manifest_path,
        fc2_mapping=fc2_mapping,
        fc3_mapping=fc3_mapping,
        minimum_promoted_blocks=minimum_promoted_blocks,
        degeneracy_tolerance_thz=degeneracy_tolerance_thz,
    )
    return write_v3_sample_manifest(output_path, payload)


__all__ = [
    "V3SampleCoverageError",
    "build_v3_sample_manifest",
    "create_v3_sample_manifest",
    "read_v3_sample_manifest",
    "verify_v3_sample_manifest_identity",
    "write_v3_sample_manifest",
]
