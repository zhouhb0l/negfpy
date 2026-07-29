"""Deterministic, provenance-bound sampling of nonzero silicon ``|V3|^2``.

The production silicon oracle compares gauge-invariant sums over complete
degenerate mode subspaces.  This module chooses those blocks *from the
independent phono3py reference*, before a NEGFPy interaction tensor exists.
Consequently, a candidate calculation cannot influence which coordinates are
tested.

The archived v1 selection is deliberately stricter than random scalar
sampling:

* exact promoted blocks are deduplicated before selection;
* a robust nonzero floor is derived from the reference itself;
* every populated target-region/branch/process stratum contributes at least
  two blocks;
* Normal and Umklapp low/median/high strength quantiles and the global maximum
  are forced into the sample;
* every archived target q point and target band is represented; and
* all empty strata and all consumed-input identities are recorded.

For meshes whose target-q count alone can exceed the 512-record reader limit,
the explicitly selected v2 algorithm makes two streaming passes, retains only
fixed-size deterministic reservoirs, preserves the physical-stratum,
strength-range, global-maximum, and target-band gates, and deliberately does
not claim coverage of every target q point.

The JSON schema and strict reader live in :mod:`silicon_oracle_validation`.
Keeping one reader avoids two subtly incompatible release-provenance formats.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

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
    V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM,
    V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION,
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
_MAXIMUM_PROMOTED_BLOCKS = 512
_SAMPLES_PER_POPULATED_STRATUM = 2
_STRENGTH_BINS_PER_PROCESS = 8
_SAMPLES_PER_STRENGTH_BIN = 2
_TARGET_BAND_RESERVOIR_SIZE = 16
_QUANTILES = (0.1, 0.5, 0.9)
_Q_REGIONS = ("gamma", "near", "generic")
_TARGET_CHARACTERS = ("acoustic", "optical")
_INTERNAL_CHARACTERS = ("aa", "ao", "oo")
_PROCESSES = ("normal", "umklapp")
_SHA256_HEX = frozenset("0123456789abcdef")

LEGACY_FULL_COVERAGE_SELECTION = "legacy_full_coverage_v1"
LARGE_MESH_BOUNDED_SELECTION = "large_mesh_streaming_bounded_v2"


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


@dataclass(frozen=True, slots=True)
class _PromotedScanContext:
    """Read-only mesh and degeneracy data shared by promoted-block scans."""

    reference_sha256: str
    degeneracy_tolerance_thz: float
    primitive_lattice_sha256: str
    frequencies: np.ndarray
    groups_by_q: tuple[tuple[tuple[int, ...], ...], ...]
    target_band_positions: Mapping[int, int]
    selected_target_bands: frozenset[int]
    periods: np.ndarray
    mesh: UniformBZMesh
    reduced_reciprocal_lattice: np.ndarray
    address_images: tuple[tuple[Any, ...], ...]
    regions: Mapping[int, str]


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


def _array_identity_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _prepare_promoted_scan(
    reference: Phono3pySiliconReference,
    *,
    degeneracy_tolerance_thz: float,
    primitive_lattice: Any,
) -> _PromotedScanContext:
    frequencies = reference.frequencies_in_grg_order()
    groups_by_q = tuple(
        _degenerate_groups(row, degeneracy_tolerance_thz)
        for row in frequencies
    )
    target_band_positions = {
        int(band): position
        for position, band in enumerate(reference.band_indices)
    }
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
    return _PromotedScanContext(
        reference_sha256=reference.sha256,
        degeneracy_tolerance_thz=degeneracy_tolerance_thz,
        primitive_lattice_sha256=_array_identity_sha256(primitive_lattice),
        frequencies=frequencies,
        groups_by_q=groups_by_q,
        target_band_positions=target_band_positions,
        selected_target_bands=frozenset(target_band_positions),
        periods=periods,
        mesh=mesh,
        reduced_reciprocal_lattice=reduced_reciprocal_lattice,
        address_images=address_images,
        regions=regions,
    )


def _validated_scan_context(
    reference: Phono3pySiliconReference,
    *,
    degeneracy_tolerance_thz: float,
    primitive_lattice: Any,
    scan_context: _PromotedScanContext | None,
) -> _PromotedScanContext:
    if scan_context is None:
        return _prepare_promoted_scan(
            reference,
            degeneracy_tolerance_thz=degeneracy_tolerance_thz,
            primitive_lattice=primitive_lattice,
        )
    if not isinstance(scan_context, _PromotedScanContext):
        raise TypeError("scan_context must be a _PromotedScanContext.")
    if (
        scan_context.reference_sha256 != reference.sha256
        or scan_context.degeneracy_tolerance_thz
        != degeneracy_tolerance_thz
        or scan_context.primitive_lattice_sha256
        != _array_identity_sha256(primitive_lattice)
    ):
        raise ValueError("scan_context does not identify this exact scan.")
    return scan_context


def _selected_target_grid_points(
    reference: Phono3pySiliconReference,
    target_grid_points: Sequence[int] | None,
) -> tuple[int, ...]:
    archived = tuple(int(value) for value in reference.grid_points_grg)
    if target_grid_points is None:
        return archived
    if isinstance(target_grid_points, (str, bytes)):
        raise TypeError("target_grid_points must be an integer sequence.")
    selected = tuple(int(value) for value in target_grid_points)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("target_grid_points must be nonempty and unique.")
    if any(value not in reference.target_groups for value in selected):
        raise ValueError(
            "target_grid_points contains a q point absent from the reference."
        )
    return selected


def _iter_promoted_blocks(
    reference: Phono3pySiliconReference,
    *,
    degeneracy_tolerance_thz: float,
    primitive_lattice: Any,
    track_duplicates: bool,
    target_grid_points: Sequence[int] | None = None,
    scan_context: _PromotedScanContext | None = None,
    verify_reference_hash: bool = True,
) -> Iterator[_PromotedBlock]:
    """Yield promoted blocks in the legacy canonical traversal order.

    ``track_duplicates`` is retained for the legacy materializing selector,
    whose exact defensive behavior is part of the archived v1 contract.  The
    bounded selector disables it: the already-validated one-to-one q1 route
    map and disjoint degeneracy partitions make duplicate keys impossible,
    and retaining every key would defeat streaming memory bounds.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "V3 sample selection requires the optional h5py package."
        ) from exc

    context = _validated_scan_context(
        reference,
        degeneracy_tolerance_thz=degeneracy_tolerance_thz,
        primitive_lattice=primitive_lattice,
        scan_context=scan_context,
    )
    selected_q0 = _selected_target_grid_points(
        reference,
        target_grid_points,
    )
    seen: (
        set[
            tuple[
                int,
                int,
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
            ]
        ]
        | None
    ) = set() if track_duplicates else None

    with h5py.File(reference.path, "r") as h5:
        for q0 in selected_q0:
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
            production_routes = context.mesh.triplets_at_q(q0)

            for q1 in range(reference.n_qpoints):
                route_position = route_for_q1[q1]
                q2 = int(triplets_grg[route_position, 2])
                address_sum = np.sum(
                    triplet_addresses[route_position],
                    axis=0,
                    dtype=np.int64,
                )
                if np.any(np.mod(address_sum, context.periods) != 0):
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
                    address_images=context.address_images,
                    primitive_lattice=primitive_lattice,
                    reduced_reciprocal_lattice=(
                        context.reduced_reciprocal_lattice
                    ),
                )
                values = strengths[route_position]
                for target_subspace in context.groups_by_q[q0]:
                    if not set(target_subspace).issubset(
                        context.selected_target_bands
                    ):
                        # An incomplete degenerate target block is gauge
                        # dependent and therefore cannot enter the manifest.
                        continue
                    target_axis = tuple(
                        context.target_band_positions[band]
                        for band in target_subspace
                    )
                    target_character = _branch_character(
                        target_subspace,
                        n_bands=reference.n_bands,
                    )
                    for band1_subspace in context.groups_by_q[q1]:
                        for band2_subspace in context.groups_by_q[q2]:
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
                                f"{context.regions[q0]}.{target_character}."
                                f"{internal}.{process}"
                            )
                            key = (
                                q0,
                                q1,
                                target_subspace,
                                band1_subspace,
                                band2_subspace,
                            )
                            if seen is not None and key in seen:
                                raise Phono3pyReferenceSchemaError(
                                    "Reference routes produced a duplicate "
                                    "gauge-invariant promoted block."
                                )
                            if seen is not None:
                                seen.add(key)
                            yield _PromotedBlock(
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
    if (
        verify_reference_hash
        and _file_sha256(reference.path) != reference.sha256
    ):
        raise Phono3pyReferenceSchemaError(
            "Reference HDF5 changed while V3 samples were selected."
        )


def _vectorized_maximum_promoted_block(
    reference: Phono3pySiliconReference,
    *,
    degeneracy_tolerance_thz: float,
    primitive_lattice: Any,
    target_grid_points: Sequence[int] | None = None,
    scan_context: _PromotedScanContext | None = None,
    verify_reference_hash: bool = True,
) -> _PromotedBlock:
    """Find the exact promoted-block maximum with vectorized q-route slabs.

    NumPy is used only to shortlist candidates.  The selected maxima are
    recomputed with the same ``math.fsum`` and SHA256 tie rule as the full
    promoted-block iterator, so this optimization does not alter v2 bytes.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "V3 sample selection requires the optional h5py package."
        ) from exc

    context = _validated_scan_context(
        reference,
        degeneracy_tolerance_thz=degeneracy_tolerance_thz,
        primitive_lattice=primitive_lattice,
        scan_context=scan_context,
    )
    selected_q0 = _selected_target_grid_points(
        reference,
        target_grid_points,
    )
    maximum: _PromotedBlock | None = None
    epsilon = np.finfo(np.float64).eps

    with h5py.File(reference.path, "r") as h5:
        for q0 in selected_q0:
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
            production_routes = context.mesh.triplets_at_q(q0)
            by_partition: dict[
                tuple[
                    tuple[tuple[int, ...], ...],
                    tuple[tuple[int, ...], ...],
                ],
                list[tuple[int, int, int]],
            ] = {}
            for q1 in range(reference.n_qpoints):
                route_position = route_for_q1[q1]
                q2 = int(triplets_grg[route_position, 2])
                address_sum = np.sum(
                    triplet_addresses[route_position],
                    axis=0,
                    dtype=np.int64,
                )
                if np.any(np.mod(address_sum, context.periods) != 0):
                    raise Phono3pyReferenceSchemaError(
                        f"{group.name} contains a triplet that violates "
                        "mesh momentum conservation."
                    )
                if production_routes[q1].q2_index != q2:
                    raise Phono3pyReferenceSchemaError(
                        f"{group.name} differs from the production q2 route."
                    )
                signature = (
                    context.groups_by_q[q1],
                    context.groups_by_q[q2],
                )
                by_partition.setdefault(signature, []).append(
                    (q1, route_position, q2)
                )

            for signature in sorted(by_partition):
                band1_groups, band2_groups = signature
                routes = by_partition[signature]
                route_positions = np.asarray(
                    [value[1] for value in routes],
                    dtype=np.int64,
                )
                for target_subspace in context.groups_by_q[q0]:
                    if not set(target_subspace).issubset(
                        context.selected_target_bands
                    ):
                        continue
                    target_axis = tuple(
                        context.target_band_positions[band]
                        for band in target_subspace
                    )
                    target_character = _branch_character(
                        target_subspace,
                        n_bands=reference.n_bands,
                    )
                    for band1_subspace in band1_groups:
                        for band2_subspace in band2_groups:
                            slab = strengths[
                                np.ix_(
                                    route_positions,
                                    target_axis,
                                    band1_subspace,
                                    band2_subspace,
                                )
                            ]
                            approximate = np.sum(
                                slab,
                                axis=(1, 2, 3),
                                dtype=np.float64,
                            )
                            approximate_maximum = float(np.max(approximate))
                            term_count = (
                                len(target_axis)
                                * len(band1_subspace)
                                * len(band2_subspace)
                            )
                            gamma = (
                                term_count * epsilon
                                / (1.0 - term_count * epsilon)
                            )
                            threshold = np.nextafter(
                                approximate_maximum
                                * (1.0 - gamma)
                                / (1.0 + gamma),
                                -math.inf,
                            )
                            shortlist = np.flatnonzero(
                                approximate >= threshold
                            )
                            internal = _internal_character(
                                band1_subspace,
                                band2_subspace,
                                n_bands=reference.n_bands,
                            )
                            for local_position in shortlist:
                                strength = math.fsum(
                                    float(value)
                                    for value in slab[
                                        int(local_position)
                                    ].ravel(order="C")
                                )
                                q1, _, q2 = routes[int(local_position)]
                                production_triplet = production_routes[q1]
                                process = _production_process(
                                    production_triplet,
                                    address_images=context.address_images,
                                    primitive_lattice=primitive_lattice,
                                    reduced_reciprocal_lattice=(
                                        context.reduced_reciprocal_lattice
                                    ),
                                )
                                candidate = _PromotedBlock(
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
                                    stratum=(
                                        f"{context.regions[q0]}."
                                        f"{target_character}.{internal}."
                                        f"{process}"
                                    ),
                                    coordinate_sha256=_coordinate_sha256(
                                        q0,
                                        q1,
                                        q2,
                                        target_subspace,
                                        band1_subspace,
                                        band2_subspace,
                                    ),
                                )
                                maximum = _canonical_extreme(
                                    maximum,
                                    candidate,
                                    maximum=True,
                                )
    if (
        verify_reference_hash
        and _file_sha256(reference.path) != reference.sha256
    ):
        raise Phono3pyReferenceSchemaError(
            "Reference HDF5 changed while its V3 maximum was selected."
        )
    if maximum is None:
        raise V3SampleCoverageError(
            "The reference contains no promoted interaction blocks."
        )
    return maximum


def _enumerate_promoted_blocks(
    reference: Phono3pySiliconReference,
    *,
    degeneracy_tolerance_thz: float,
    primitive_lattice: Any,
) -> tuple[_PromotedBlock, ...]:
    """Materialize the exact archived v1 candidate population."""

    return tuple(
        _iter_promoted_blocks(
            reference,
            degeneracy_tolerance_thz=degeneracy_tolerance_thz,
            primitive_lattice=primitive_lattice,
            track_duplicates=True,
        )
    )


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


class _SmallestHashReservoir:
    """Retain a fixed number of canonical-hash-smallest blocks."""

    __slots__ = ("capacity", "_heap", "_keys")

    def __init__(self, capacity: int) -> None:
        self.capacity = _positive_int(capacity, name="capacity")
        self._heap: list[
            tuple[
                int,
                tuple[
                    int,
                    int,
                    tuple[int, ...],
                    tuple[int, ...],
                    tuple[int, ...],
                ],
                _PromotedBlock,
            ]
        ] = []
        self._keys: set[
            tuple[
                int,
                int,
                tuple[int, ...],
                tuple[int, ...],
                tuple[int, ...],
            ]
        ] = set()

    def __len__(self) -> int:
        return len(self._heap)

    def add(self, candidate: _PromotedBlock) -> None:
        if candidate.key in self._keys:
            return
        priority = int(candidate.coordinate_sha256, 16)
        entry = (-priority, candidate.key, candidate)
        if len(self._heap) < self.capacity:
            heapq.heappush(self._heap, entry)
            self._keys.add(candidate.key)
            return
        worst_priority = -self._heap[0][0]
        if priority > worst_priority:
            return
        if priority == worst_priority:
            # A SHA256 collision is not a normal execution path, but resolving
            # it explicitly keeps selection deterministic without retaining a
            # population-sized collision table.
            worst = max(
                self._heap,
                key=lambda item: (-item[0], item[1]),
            )
            if (priority, candidate.key) >= (-worst[0], worst[1]):
                return
            self._heap.remove(worst)
            heapq.heapify(self._heap)
            self._keys.remove(worst[2].key)
            heapq.heappush(self._heap, entry)
            self._keys.add(candidate.key)
            return
        removed = heapq.heapreplace(self._heap, entry)
        self._keys.remove(removed[2].key)
        self._keys.add(candidate.key)

    def values(self) -> tuple[_PromotedBlock, ...]:
        return tuple(
            sorted(
                (entry[2] for entry in self._heap),
                key=lambda item: (item.coordinate_sha256, item.key),
            )
        )


def _canonical_extreme(
    current: _PromotedBlock | None,
    candidate: _PromotedBlock,
    *,
    maximum: bool,
) -> _PromotedBlock:
    if current is None:
        return candidate
    if (
        candidate.strength_ev2 > current.strength_ev2
        if maximum
        else candidate.strength_ev2 < current.strength_ev2
    ):
        return candidate
    if (
        candidate.strength_ev2 == current.strength_ev2
        and (candidate.coordinate_sha256, candidate.key)
        < (current.coordinate_sha256, current.key)
    ):
        return candidate
    return current


def _strength_bin(
    strength: float,
    *,
    floor: float,
    global_maximum: float,
) -> int:
    """Return one of eight fixed logarithmic positions in the robust range."""

    if global_maximum <= floor:
        return 0
    fraction = math.log(strength / floor) / math.log(global_maximum / floor)
    return min(
        _STRENGTH_BINS_PER_PROCESS - 1,
        max(0, int(math.floor(fraction * _STRENGTH_BINS_PER_PROCESS))),
    )


class _BoundedSelectionAccumulator:
    """Fixed-memory v2 selector state for the second reference-data pass."""

    __slots__ = (
        "floor",
        "global_maximum",
        "maximum_promoted_blocks",
        "target_bands",
        "viable_count",
        "stratum_counts",
        "global_reservoir",
        "stratum_reservoirs",
        "strength_reservoirs",
        "target_band_reservoirs",
        "process_minima",
        "process_maxima",
        "global_maximum_block",
    )

    def __init__(
        self,
        *,
        floor: float,
        global_maximum: float,
        maximum_promoted_blocks: int,
        target_bands: Sequence[int],
    ) -> None:
        self.floor = floor
        self.global_maximum = global_maximum
        self.maximum_promoted_blocks = maximum_promoted_blocks
        self.target_bands = tuple(int(value) for value in target_bands)
        self.viable_count = 0
        self.stratum_counts: dict[str, int] = {}
        self.global_reservoir = _SmallestHashReservoir(
            maximum_promoted_blocks
        )
        self.stratum_reservoirs: dict[str, _SmallestHashReservoir] = {}
        self.strength_reservoirs: dict[
            tuple[str, int], _SmallestHashReservoir
        ] = {}
        self.target_band_reservoirs = {
            band: _SmallestHashReservoir(_TARGET_BAND_RESERVOIR_SIZE)
            for band in self.target_bands
        }
        self.process_minima: dict[str, _PromotedBlock] = {}
        self.process_maxima: dict[str, _PromotedBlock] = {}
        self.global_maximum_block: _PromotedBlock | None = None

    @property
    def retention_bound(self) -> int:
        return (
            self.maximum_promoted_blocks
            + len(_all_strata()) * _SAMPLES_PER_POPULATED_STRATUM
            + len(_PROCESSES)
            * _STRENGTH_BINS_PER_PROCESS
            * _SAMPLES_PER_STRENGTH_BIN
            + len(self.target_bands) * _TARGET_BAND_RESERVOIR_SIZE
            + 2 * len(_PROCESSES)
            + 1
        )

    @property
    def retained_candidate_slots(self) -> int:
        return (
            len(self.global_reservoir)
            + sum(len(value) for value in self.stratum_reservoirs.values())
            + sum(len(value) for value in self.strength_reservoirs.values())
            + sum(
                len(value) for value in self.target_band_reservoirs.values()
            )
            + len(self.process_minima)
            + len(self.process_maxima)
            + int(self.global_maximum_block is not None)
        )

    def add(self, candidate: _PromotedBlock) -> None:
        if candidate.strength_ev2 < self.floor:
            return
        self.viable_count += 1
        self.stratum_counts[candidate.stratum] = (
            self.stratum_counts.get(candidate.stratum, 0) + 1
        )
        self.global_reservoir.add(candidate)
        stratum_reservoir = self.stratum_reservoirs.get(candidate.stratum)
        if stratum_reservoir is None:
            stratum_reservoir = _SmallestHashReservoir(
                _SAMPLES_PER_POPULATED_STRATUM
            )
            self.stratum_reservoirs[candidate.stratum] = stratum_reservoir
        stratum_reservoir.add(candidate)
        bin_index = _strength_bin(
            candidate.strength_ev2,
            floor=self.floor,
            global_maximum=self.global_maximum,
        )
        strength_key = (candidate.process, bin_index)
        strength_reservoir = self.strength_reservoirs.get(strength_key)
        if strength_reservoir is None:
            strength_reservoir = _SmallestHashReservoir(
                _SAMPLES_PER_STRENGTH_BIN
            )
            self.strength_reservoirs[strength_key] = strength_reservoir
        strength_reservoir.add(candidate)
        for band in candidate.target_subspace:
            reservoir = self.target_band_reservoirs.get(band)
            if reservoir is not None:
                reservoir.add(candidate)
        self.process_minima[candidate.process] = _canonical_extreme(
            self.process_minima.get(candidate.process),
            candidate,
            maximum=False,
        )
        self.process_maxima[candidate.process] = _canonical_extreme(
            self.process_maxima.get(candidate.process),
            candidate,
            maximum=True,
        )
        if candidate.strength_ev2 == self.global_maximum:
            self.global_maximum_block = _canonical_extreme(
                self.global_maximum_block,
                candidate,
                maximum=True,
            )
        if self.retained_candidate_slots > self.retention_bound:
            raise AssertionError("Bounded V3 selection exceeded its memory bound.")


def _select_bounded_blocks(
    accumulator: _BoundedSelectionAccumulator,
    *,
    minimum_promoted_blocks: int,
) -> tuple[tuple[_PromotedBlock, ...], tuple[str, ...], tuple[str, ...]]:
    if accumulator.viable_count < minimum_promoted_blocks:
        raise V3SampleCoverageError(
            "Reference has only "
            f"{accumulator.viable_count} promoted blocks at or above the "
            f"robust nonzero floor {accumulator.floor:.17g} eV^2; at least "
            f"{minimum_promoted_blocks} are required."
        )
    missing_processes = [
        process
        for process in _PROCESSES
        if process not in accumulator.process_minima
    ]
    if missing_processes:
        raise V3SampleCoverageError(
            "Reference does not contain robustly nonzero blocks for required "
            f"processes: {missing_processes}."
        )
    sparse_strata = {
        name: count
        for name, count in accumulator.stratum_counts.items()
        if count < _SAMPLES_PER_POPULATED_STRATUM
    }
    if sparse_strata:
        raise V3SampleCoverageError(
            "Every populated stratum requires two distinct promoted blocks; "
            f"insufficient strata: {sparse_strata}."
        )
    if accumulator.global_maximum_block is None:
        raise AssertionError("The second pass did not recover the global maximum.")

    selected: dict[
        tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        _PromotedBlock,
    ] = {}

    def add(candidate: _PromotedBlock) -> bool:
        if candidate.key in selected:
            return False
        selected[candidate.key] = candidate
        return True

    for name in sorted(accumulator.stratum_reservoirs):
        for candidate in accumulator.stratum_reservoirs[name].values():
            add(candidate)
    add(accumulator.global_maximum_block)
    for process in _PROCESSES:
        add(accumulator.process_minima[process])
        add(accumulator.process_maxima[process])
    for key in sorted(accumulator.strength_reservoirs):
        for candidate in accumulator.strength_reservoirs[key].values():
            add(candidate)

    observed_bands = {candidate.seed_b0 for candidate in selected.values()}
    for band in accumulator.target_bands:
        if band in observed_bands:
            continue
        candidate = next(
            (
                value
                for value in accumulator.target_band_reservoirs[band].values()
                if value.key not in selected
            ),
            None,
        )
        if candidate is None:
            raise V3SampleCoverageError(
                "No distinct robustly nonzero promoted block can cover "
                f"archived target band {band} within the bounded reservoir."
            )
        add(candidate.with_target_seed(band))
        observed_bands.add(band)

    if len(selected) > accumulator.maximum_promoted_blocks:
        raise V3SampleCoverageError(
            "Required bounded V3 coverage exceeds maximum_promoted_blocks."
        )
    for candidate in accumulator.global_reservoir.values():
        if len(selected) >= accumulator.maximum_promoted_blocks:
            break
        add(candidate)
    if len(selected) < minimum_promoted_blocks:
        raise V3SampleCoverageError(
            "Bounded deterministic selection could not meet the minimum "
            "unique promoted-block coverage."
        )

    selected_blocks = tuple(
        sorted(
            selected.values(),
            key=lambda item: (item.coordinate_sha256, item.key),
        )
    )
    if len(selected_blocks) > _MAXIMUM_PROMOTED_BLOCKS:
        raise AssertionError("Bounded selector emitted more than 512 samples.")
    required_strata = tuple(sorted(accumulator.stratum_counts))
    empty_strata = tuple(
        name for name in _all_strata() if name not in accumulator.stratum_counts
    )
    return selected_blocks, required_strata, empty_strata


def _build_bounded_selection(
    reference: Phono3pySiliconReference,
    *,
    degeneracy_tolerance_thz: float,
    primitive_lattice: Any,
    minimum_promoted_blocks: int,
    maximum_promoted_blocks: int,
) -> tuple[
    tuple[_PromotedBlock, ...],
    tuple[str, ...],
    tuple[str, ...],
    float,
]:
    """Execute the pinned two-pass bounded v2 reference scan."""

    scan_context = _prepare_promoted_scan(
        reference,
        degeneracy_tolerance_thz=degeneracy_tolerance_thz,
        primitive_lattice=primitive_lattice,
    )
    maximum_block = _vectorized_maximum_promoted_block(
        reference,
        degeneracy_tolerance_thz=degeneracy_tolerance_thz,
        primitive_lattice=primitive_lattice,
        scan_context=scan_context,
    )
    floor = max(
        _ABSOLUTE_NONZERO_FLOOR_EV2,
        _RELATIVE_NONZERO_FLOOR * maximum_block.strength_ev2,
    )
    accumulator = _BoundedSelectionAccumulator(
        floor=floor,
        global_maximum=maximum_block.strength_ev2,
        maximum_promoted_blocks=maximum_promoted_blocks,
        target_bands=reference.band_indices,
    )
    for candidate in _iter_promoted_blocks(
        reference,
        degeneracy_tolerance_thz=degeneracy_tolerance_thz,
        primitive_lattice=primitive_lattice,
        track_duplicates=False,
        scan_context=scan_context,
    ):
        accumulator.add(candidate)
    selected, required_strata, empty_strata = _select_bounded_blocks(
        accumulator,
        minimum_promoted_blocks=minimum_promoted_blocks,
    )
    return selected, required_strata, empty_strata, floor


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
    selection_mode: str = LEGACY_FULL_COVERAGE_SELECTION,
    maximum_promoted_blocks: int = _MAXIMUM_PROMOTED_BLOCKS,
) -> dict[str, Any]:
    """Build the canonical strict-JSON payload without writing it.

    ``reference`` must already have passed
    :func:`read_phono3py_silicon_reference`.  Its bytes are checked before and
    after selection.  All sample coordinates and invariant strengths are read
    from that artifact; no NEGFPy candidate interaction is accepted here.

    The default ``legacy_full_coverage_v1`` path intentionally retains its
    archived materializing algorithm and exact payload.  Select
    ``large_mesh_streaming_bounded_v2`` explicitly for large target-q sets;
    it scans the reference twice while retaining a mesh-independent bounded
    candidate set and emits at most ``maximum_promoted_blocks`` records.
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
    if selection_mode not in {
        LEGACY_FULL_COVERAGE_SELECTION,
        LARGE_MESH_BOUNDED_SELECTION,
    }:
        raise ValueError(
            "selection_mode must be exactly "
            f"{LEGACY_FULL_COVERAGE_SELECTION!r} or "
            f"{LARGE_MESH_BOUNDED_SELECTION!r}."
        )
    maximum = _positive_int(
        maximum_promoted_blocks,
        name="maximum_promoted_blocks",
    )
    if maximum > _MAXIMUM_PROMOTED_BLOCKS:
        raise ValueError("maximum_promoted_blocks may not exceed 512.")
    if selection_mode == LARGE_MESH_BOUNDED_SELECTION and maximum < minimum:
        raise ValueError(
            "maximum_promoted_blocks must be at least "
            "minimum_promoted_blocks in bounded mode."
        )

    if selection_mode == LEGACY_FULL_COVERAGE_SELECTION:
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
        algorithm = V3_SAMPLE_SELECTION_ALGORITHM
        algorithm_version = V3_SAMPLE_SELECTION_ALGORITHM_VERSION
        cover_all_target_grid_points = True
    else:
        selected, required_strata, empty_strata, floor = (
            _build_bounded_selection(
                reference,
                degeneracy_tolerance_thz=degeneracy,
                primitive_lattice=fc3_mapping.primitive.lattice,
                minimum_promoted_blocks=minimum,
                maximum_promoted_blocks=maximum,
            )
        )
        algorithm = V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM
        algorithm_version = V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
        cover_all_target_grid_points = False
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
        "selection_algorithm": algorithm,
        "selection_algorithm_version": algorithm_version,
        "robust_nonzero_floor_ev2": floor,
        "coverage": {
            "cover_all_target_grid_points": cover_all_target_grid_points,
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
    selection_mode: str = LEGACY_FULL_COVERAGE_SELECTION,
    maximum_promoted_blocks: int = _MAXIMUM_PROMOTED_BLOCKS,
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
        selection_mode=selection_mode,
        maximum_promoted_blocks=maximum_promoted_blocks,
    )
    return write_v3_sample_manifest(output_path, payload)


__all__ = [
    "LARGE_MESH_BOUNDED_SELECTION",
    "LEGACY_FULL_COVERAGE_SELECTION",
    "V3SampleCoverageError",
    "build_v3_sample_manifest",
    "create_v3_sample_manifest",
    "read_v3_sample_manifest",
    "verify_v3_sample_manifest_identity",
    "write_v3_sample_manifest",
]
