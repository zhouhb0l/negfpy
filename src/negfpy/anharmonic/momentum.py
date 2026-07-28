"""Exact momentum routing on an unreduced uniform three-dimensional mesh.

The convention is intentionally small and explicit. It follows the
all-incoming triplet convention documented and implemented by phono3py
v4.4.0::

    q0 + q1 + q2 = G

where ``G`` is an integer reciprocal-lattice vector. Phono3py's generalized
regular-grid index is X-fast,

``index = rx + nx * (ry + ny * rz)``,

for residue addresses ``r = address mod mesh``. This module uses the same
index rule and enumerates every q1 index in ascending order. No point-group,
time-reversal, or q1/q2 exchange reduction is performed, so a fixed q0 always
has exactly ``Nq = nx * ny * nz`` unit-multiplicity triplets and weight
``1/Nq``.

The isolated routing core stores exactly one componentwise-centered
representative per residue class.  For an odd period ``m`` the representatives are
``0, ..., floor(m/2), -floor(m/2), ..., -1``. For an even period, the
Nyquist representative is fixed to ``-m/2``. Brillouin-zone surface
duplicates are absent from this canonical representation.

For comparisons with phono3py, :func:`shortest_bz_address_images` constructs
the metric-relocated surface images using the pinned phono3py 4.4.0 squared-
distance tolerance.  Normal/Umklapp classification can then use every retained
surface image instead of the arbitrary canonical representative.  The
underlying modular routing and conservation decisions remain exact integer
arithmetic.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Literal

import numpy as np


IntegerTriple = tuple[int, int, int]
MomentumProcess = Literal["normal", "umklapp"]

PHONO3PY_TRIPLET_CONVENTION = "q0+q1+q2=G"
GRID_INDEX_ORDER = "x-fast"
PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR = 0.01
PHONO3PY_4_4_NIGGLI_TOLERANCE = 1.0e-5

# This is the exact axis order of phono3py 4.4.0's 5^3 BZ relocation search.
# The public helper returns a sorted immutable set, so this order only affects
# phono3py's tolerance-biased reference-distance selection.
_PHONO3PY_BZ_SEARCH_COMPONENTS = (0, 1, 2, -2, -1)


def _exact_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    try:
        integer = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    return int(integer)


def _integer_triple(values: object, *, name: str) -> IntegerTriple:
    if isinstance(values, GridAddress):
        return values.components
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of three integers.")
    try:
        items = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of three integers.") from exc
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three integers.")
    return tuple(
        _exact_integer(value, name=f"{name}[{axis}]")
        for axis, value in enumerate(items)
    )  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GridAddress:
    """Immutable integer address in a reciprocal-grid basis.

    An address alone is not required to be canonical because its periods live
    on :class:`UniformBZMesh`. Mesh methods canonicalize or validate it.
    """

    components: IntegerTriple

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "components",
            _integer_triple(self.components, name="components"),
        )

    @property
    def x(self) -> int:
        return self.components[0]

    @property
    def y(self) -> int:
        return self.components[1]

    @property
    def z(self) -> int:
        return self.components[2]

    def __iter__(self) -> Iterator[int]:
        return iter(self.components)

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> int:
        return self.components[index]


@dataclass(frozen=True, slots=True)
class UniformBZMesh:
    """Immutable full uniform mesh with exact canonical address routing."""

    shape: IntegerTriple

    def __post_init__(self) -> None:
        shape = _integer_triple(self.shape, name="shape")
        if any(period <= 0 for period in shape):
            raise ValueError("Every mesh period must be a positive integer.")
        object.__setattr__(self, "shape", shape)

    @property
    def mesh(self) -> IntegerTriple:
        """Alias using phono3py's mesh terminology."""

        return self.shape

    @property
    def n_qpoints(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def size(self) -> int:
        return self.n_qpoints

    @property
    def normalization(self) -> Fraction:
        """Exact full-mesh weight for one q1 term."""

        return Fraction(1, self.n_qpoints)

    @property
    def normalization_float(self) -> float:
        return 1.0 / self.n_qpoints

    @property
    def addresses(self) -> tuple[GridAddress, ...]:
        return tuple(self.iter_addresses())

    def _checked_index(self, index: object) -> int:
        checked = _exact_integer(index, name="grid index")
        if checked < 0 or checked >= self.n_qpoints:
            raise IndexError(
                f"grid index {checked} is outside [0, {self.n_qpoints})."
            )
        return checked

    @staticmethod
    def _centered_component(value: int, period: int) -> int:
        residue = value % period
        if 2 * residue >= period:
            residue -= period
        return residue

    def canonical_address(
        self,
        address: GridAddress | Sequence[int],
    ) -> GridAddress:
        components = _integer_triple(address, name="address")
        return GridAddress(
            tuple(
                self._centered_component(component, period)
                for component, period in zip(
                    components,
                    self.shape,
                    strict=True,
                )
            )
        )

    def is_canonical_address(
        self,
        address: GridAddress | Sequence[int],
    ) -> bool:
        components = _integer_triple(address, name="address")
        return self.canonical_address(components).components == components

    def require_canonical_address(
        self,
        address: GridAddress | Sequence[int],
    ) -> GridAddress:
        components = _integer_triple(address, name="address")
        canonical = self.canonical_address(components)
        if canonical.components != components:
            raise ValueError(
                f"address {components} is not canonical for mesh {self.shape}; "
                f"its canonical representative is {canonical.components}."
            )
        return canonical

    def index_from_address(
        self,
        address: GridAddress | Sequence[int],
    ) -> int:
        """Return the X-fast index of any translationally equivalent address."""

        x, y, z = _integer_triple(address, name="address")
        nx, ny, nz = self.shape
        rx, ry, rz = x % nx, y % ny, z % nz
        return rx + nx * (ry + ny * rz)

    def address_from_index(self, index: int) -> GridAddress:
        """Return the unique centered representative of an X-fast index."""

        checked = self._checked_index(index)
        nx, ny, _ = self.shape
        rx = checked % nx
        quotient = checked // nx
        ry = quotient % ny
        rz = quotient // ny
        return self.canonical_address((rx, ry, rz))

    def iter_addresses(self) -> Iterator[GridAddress]:
        for index in range(self.n_qpoints):
            yield self.address_from_index(index)

    def negative_address(
        self,
        address: GridAddress | Sequence[int],
    ) -> GridAddress:
        canonical = self.canonical_address(address)
        return self.canonical_address(tuple(-value for value in canonical))

    def negative_index(self, index: int) -> int:
        return self.index_from_address(
            self.negative_address(self.address_from_index(index))
        )

    def fractional_q(
        self,
        address: GridAddress | Sequence[int],
    ) -> tuple[Fraction, Fraction, Fraction]:
        canonical = self.canonical_address(address)
        return tuple(
            Fraction(component, period)
            for component, period in zip(
                canonical,
                self.shape,
                strict=True,
            )
        )  # type: ignore[return-value]

    def triplets_at_q(
        self,
        target: int | GridAddress,
    ) -> tuple[MomentumTriplet, ...]:
        return enumerate_target_triplets(self, target)


@dataclass(frozen=True, slots=True)
class MomentumTriplet:
    """One exact full-mesh route satisfying ``q0 + q1 + q2 = G``."""

    mesh: UniformBZMesh
    q0_index: int
    q1_index: int
    q2_index: int
    reciprocal_vector: IntegerTriple

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, UniformBZMesh):
            raise TypeError("mesh must be a UniformBZMesh.")
        q0_index = self.mesh._checked_index(self.q0_index)
        q1_index = self.mesh._checked_index(self.q1_index)
        q2_index = self.mesh._checked_index(self.q2_index)
        reciprocal_vector = _integer_triple(
            self.reciprocal_vector,
            name="reciprocal_vector",
        )

        addresses = tuple(
            self.mesh.address_from_index(index)
            for index in (q0_index, q1_index, q2_index)
        )
        total = tuple(
            sum(address[axis] for address in addresses)
            for axis in range(3)
        )
        if any(
            component % period
            for component, period in zip(total, self.mesh.shape, strict=True)
        ):
            raise ValueError(
                "Triplet indices do not satisfy modular momentum conservation."
            )
        expected = tuple(
            component // period
            for component, period in zip(total, self.mesh.shape, strict=True)
        )
        if reciprocal_vector != expected:
            raise ValueError(
                "reciprocal_vector is inconsistent with the canonical "
                f"triplet addresses: expected {expected}, got "
                f"{reciprocal_vector}."
            )

        object.__setattr__(self, "q0_index", q0_index)
        object.__setattr__(self, "q1_index", q1_index)
        object.__setattr__(self, "q2_index", q2_index)
        object.__setattr__(self, "reciprocal_vector", reciprocal_vector)

    @property
    def indices(self) -> tuple[int, int, int]:
        return self.q0_index, self.q1_index, self.q2_index

    @property
    def q0_address(self) -> GridAddress:
        return self.mesh.address_from_index(self.q0_index)

    @property
    def q1_address(self) -> GridAddress:
        return self.mesh.address_from_index(self.q1_index)

    @property
    def q2_address(self) -> GridAddress:
        return self.mesh.address_from_index(self.q2_index)

    @property
    def addresses(self) -> tuple[GridAddress, GridAddress, GridAddress]:
        return self.q0_address, self.q1_address, self.q2_address

    @property
    def G(self) -> IntegerTriple:
        return self.reciprocal_vector

    @property
    def process(self) -> MomentumProcess:
        return (
            "normal"
            if self.reciprocal_vector == (0, 0, 0)
            else "umklapp"
        )

    @property
    def classification(self) -> MomentumProcess:
        return self.process

    @property
    def is_normal(self) -> bool:
        return self.reciprocal_vector == (0, 0, 0)

    @property
    def is_umklapp(self) -> bool:
        return not self.is_normal

    @property
    def multiplicity(self) -> int:
        """Unit multiplicity because this core never symmetry-reduces q1."""

        return 1

    @property
    def weight(self) -> Fraction:
        return self.mesh.normalization

    @property
    def normalization(self) -> Fraction:
        return self.weight


def enumerate_target_triplets(
    mesh: UniformBZMesh,
    target: int | GridAddress,
) -> tuple[MomentumTriplet, ...]:
    """Enumerate the full q1 mesh for a fixed q0 in canonical index order.

    ``q2`` is the unique mesh point congruent to ``-q0-q1``. The returned
    tuple is ordered by ascending q1 index and contains no symmetry reduction.
    """

    if not isinstance(mesh, UniformBZMesh):
        raise TypeError("mesh must be a UniformBZMesh.")
    if isinstance(target, GridAddress):
        q0_index = mesh.index_from_address(target)
    else:
        q0_index = mesh._checked_index(target)
    q0_address = mesh.address_from_index(q0_index)

    triplets: list[MomentumTriplet] = []
    for q1_index in range(mesh.n_qpoints):
        q1_address = mesh.address_from_index(q1_index)
        q2_address = mesh.canonical_address(
            tuple(
                -q0_address[axis] - q1_address[axis]
                for axis in range(3)
            )
        )
        q2_index = mesh.index_from_address(q2_address)
        total = tuple(
            q0_address[axis] + q1_address[axis] + q2_address[axis]
            for axis in range(3)
        )
        reciprocal_vector = tuple(
            component // period
            for component, period in zip(total, mesh.shape, strict=True)
        )
        triplets.append(
            MomentumTriplet(
                mesh=mesh,
                q0_index=q0_index,
                q1_index=q1_index,
                q2_index=q2_index,
                reciprocal_vector=reciprocal_vector,
            )
        )

    return tuple(triplets)


def _reciprocal_metric(primitive_lattice: object) -> np.ndarray:
    raw_lattice = np.asarray(primitive_lattice)
    if raw_lattice.dtype.kind == "c":
        raise TypeError("primitive_lattice must be real-valued.")
    try:
        lattice = np.array(
            raw_lattice,
            dtype=np.float64,
            order="C",
            copy=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "primitive_lattice must be a real numerical matrix."
        ) from exc
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(
            "primitive_lattice must be a finite matrix with shape (3, 3)."
        )
    if np.linalg.matrix_rank(lattice) != 3:
        raise ValueError("primitive_lattice must be nonsingular.")
    reciprocal_rows = np.linalg.inv(lattice).T
    metric = reciprocal_rows @ reciprocal_rows.T
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(metric)))
    if not np.isfinite(minimum_eigenvalue) or minimum_eigenvalue <= 0.0:
        raise ValueError("primitive_lattice produced an invalid reciprocal metric.")
    return metric


def _validated_lattice(
    values: object,
    *,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype.kind == "c":
        raise TypeError(f"{name} must be real-valued.")
    try:
        lattice = np.array(
            raw,
            dtype=np.float64,
            order="C",
            copy=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numerical matrix.") from exc
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise ValueError(f"{name} must be a finite matrix with shape (3, 3).")
    if np.linalg.matrix_rank(lattice) != 3:
        raise ValueError(f"{name} must be nonsingular.")
    return lattice


def _distance_tolerance_factor(value: object) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            "tolerance_factor must be a nonnegative finite scalar."
        )
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "tolerance_factor must be a nonnegative finite scalar."
        )
    return tolerance


def _positive_tolerance(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite scalar.")
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return tolerance


def niggli_reduced_reciprocal_lattice(
    primitive_lattice: object,
    *,
    tolerance: float = PHONO3PY_4_4_NIGGLI_TOLERANCE,
) -> np.ndarray:
    """Return phono3py-4.4's Niggli-reduced reciprocal column basis.

    This operation imports the optional pinned ``spglib`` dependency lazily.
    Supplying a precomputed basis to the BZ helpers keeps their algebra tests
    dependency-free, while certified material calculations derive it with the
    same ``spglib.niggli_reduce`` operation used by phonopy 4.4.0.
    """

    lattice = _validated_lattice(
        primitive_lattice,
        name="primitive_lattice",
    )
    tolerance = _positive_tolerance(tolerance, name="tolerance")
    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - optional minimal install
        raise ImportError(
            "Niggli BZ relocation requires spglib. Install the "
            "'material-oracle' extra or supply reduced_reciprocal_lattice."
        ) from exc

    reciprocal_columns = np.linalg.inv(lattice)
    # spglib 2.7.0 defaults to its deprecated non-raising error mode and
    # emits this warning even on success.  Suppress only that exact upstream
    # transition warning; a failed reduction still returns None and is
    # rejected immediately below.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Set OLD_ERROR_HANDLING to false and catch the errors "
                "directly\\."
            ),
            category=DeprecationWarning,
            module=r"spglib(?:\\..*)?",
        )
        reduced_rows = spglib.niggli_reduce(
            np.array(
                reciprocal_columns.T,
                dtype=np.float64,
                order="C",
                copy=True,
            ),
            eps=tolerance,
        )
    if reduced_rows is None:
        raise RuntimeError("spglib failed to Niggli-reduce the reciprocal cell.")
    reduced = _validated_lattice(
        np.asarray(reduced_rows).T,
        name="Niggli-reduced reciprocal lattice",
    )
    reduced.setflags(write=False)
    return reduced


def _validated_reduced_reciprocal_lattice(
    primitive_lattice: object,
    reduced_reciprocal_lattice: object | None,
    *,
    niggli_tolerance: float,
) -> np.ndarray:
    direct = _validated_lattice(
        primitive_lattice,
        name="primitive_lattice",
    )
    original = np.linalg.inv(direct)
    if reduced_reciprocal_lattice is None:
        return niggli_reduced_reciprocal_lattice(
            direct,
            tolerance=niggli_tolerance,
        )

    reduced = _validated_lattice(
        reduced_reciprocal_lattice,
        name="reduced_reciprocal_lattice",
    )
    transformation = np.linalg.inv(reduced) @ original
    integer_transformation = np.rint(transformation)
    if not np.allclose(
        transformation,
        integer_transformation,
        rtol=0.0,
        atol=1.0e-8,
    ):
        raise ValueError(
            "reduced_reciprocal_lattice is not related to the primitive "
            "reciprocal lattice by an integer basis transformation."
        )
    determinant = int(round(float(np.linalg.det(integer_transformation))))
    if abs(determinant) != 1:
        raise ValueError(
            "reduced_reciprocal_lattice transformation must be unimodular."
        )
    reduced.setflags(write=False)
    return reduced


def _phono3py_bz_distance_tolerance(
    mesh: UniformBZMesh,
    reduced_reciprocal_lattice: np.ndarray,
    tolerance_factor: float,
) -> float:
    periods = np.asarray(mesh.shape, dtype=np.float64)
    return float(
        tolerance_factor
        * np.max(
            np.sum(np.square(reduced_reciprocal_lattice), axis=0)
            / np.square(periods)
        )
    )


def shortest_bz_address_images(
    mesh: UniformBZMesh,
    primitive_lattice: object,
    *,
    tolerance_factor: float = PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR,
    reduced_reciprocal_lattice: object | None = None,
    niggli_tolerance: float = PHONO3PY_4_4_NIGGLI_TOLERANCE,
) -> tuple[tuple[GridAddress, ...], ...]:
    """Return phono3py-4.4-compatible BZ images of each mesh residue.

    ``primitive_lattice`` contains row-wise direct-lattice vectors.  Images
    are selected by the Cartesian norm of the corresponding reciprocal-space
    vector. Boundary duplicates are retained using phono3py 4.4.0's exact
    mesh-scaled squared-distance tolerance,

    ``tolerance_factor * max_i(|b_i^reduced|^2 / mesh_i^2)``.

    The default factor ``0.01`` and the 5-by-5-by-5 candidate order are pinned
    to that release. The reciprocal basis is Niggli-reduced with the pinned
    tolerance ``1e-5``. A caller may supply that reduced column-basis matrix
    explicitly; it is checked to be unimodularly equivalent to the primitive
    reciprocal lattice. This tolerance is part of the external-oracle
    convention: on a strongly skew cell it can intentionally retain a
    near-boundary image that is not the mathematically unique nearest image.
    """

    if not isinstance(mesh, UniformBZMesh):
        raise TypeError("mesh must be a UniformBZMesh.")
    metric = _reciprocal_metric(primitive_lattice)
    tolerance_factor = _distance_tolerance_factor(tolerance_factor)
    niggli_tolerance = _positive_tolerance(
        niggli_tolerance,
        name="niggli_tolerance",
    )
    reduced_reciprocal = _validated_reduced_reciprocal_lattice(
        primitive_lattice,
        reduced_reciprocal_lattice,
        niggli_tolerance=niggli_tolerance,
    )
    distance_tolerance = _phono3py_bz_distance_tolerance(
        mesh,
        reduced_reciprocal,
        tolerance_factor,
    )

    periods = np.array(mesh.shape, dtype=np.int64)
    images: list[tuple[GridAddress, ...]] = []
    for canonical in mesh.addresses:
        address = np.array(canonical.components, dtype=np.int64)
        fractional = address / periods
        candidates: list[tuple[float, tuple[int, int, int]]] = []
        for shift in product(
            _PHONO3PY_BZ_SEARCH_COMPONENTS,
            repeat=3,
        ):
            shift_array = np.asarray(shift, dtype=np.int64)
            shifted_fractional = fractional + shift_array
            norm_squared = float(
                shifted_fractional @ metric @ shifted_fractional
            )
            shifted_address = tuple(
                int(value)
                for value in address + periods * shift_array
            )
            candidates.append((norm_squared, shifted_address))

        # phono3py deliberately does not take the raw floating-point minimum.
        # A candidate replaces the reference only when it is lower by more
        # than the pinned tolerance, stabilizing surface-image selection.
        reference_distance = candidates[0][0]
        for norm_squared, _ in candidates[1:]:
            if norm_squared < reference_distance - distance_tolerance:
                reference_distance = norm_squared
        selected = [
            shifted_address
            for norm_squared, shifted_address in candidates
            if norm_squared < reference_distance + distance_tolerance
        ]

        unique = tuple(
            GridAddress(value) for value in sorted(set(selected))
        )
        if not unique:
            raise RuntimeError("Brillouin-zone image search returned no image.")
        images.append(unique)
    return tuple(images)


def _validated_address_images(
    mesh: UniformBZMesh,
    address_images: Sequence[Sequence[GridAddress]],
) -> tuple[tuple[GridAddress, ...], ...]:
    try:
        images = tuple(tuple(group) for group in address_images)
    except TypeError as exc:
        raise TypeError(
            "address_images must contain one image sequence per mesh point."
        ) from exc
    if len(images) != mesh.n_qpoints:
        raise ValueError(
            "address_images must contain one entry per mesh grid point."
        )
    for group in images:
        if not group or any(not isinstance(item, GridAddress) for item in group):
            raise TypeError(
                "Every address_images entry must contain GridAddress values."
            )
    for index, group in enumerate(images):
        if any(mesh.index_from_address(item) != index for item in group):
            raise ValueError(
                "address_images contains an image in the wrong residue class."
            )
    return images


def triplet_is_normal_on_bz_surface(
    triplet: MomentumTriplet,
    address_images: Sequence[Sequence[GridAddress]],
) -> bool:
    """Return whether shortest BZ images can satisfy ``q0+q1+q2=0``."""

    if not isinstance(triplet, MomentumTriplet):
        raise TypeError("triplet must be a MomentumTriplet.")
    images = _validated_address_images(triplet.mesh, address_images)

    # Phono3py's BZ-triplet search may select a translationally equivalent
    # surface image for any of the three legs while minimizing |G|.
    selected = tuple(images[index] for index in triplet.indices)
    for first, second, third in product(*selected):
        if all(
            first[axis] + second[axis] + third[axis] == 0
            for axis in range(3)
        ):
            return True
    return False


def minimum_bz_reciprocal_vector(
    triplet: MomentumTriplet,
    address_images: Sequence[Sequence[GridAddress]],
    primitive_lattice: object,
    *,
    tolerance_factor: float = PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR,
    reduced_reciprocal_lattice: object | None = None,
    niggli_tolerance: float = PHONO3PY_4_4_NIGGLI_TOLERANCE,
) -> IntegerTriple:
    """Return a deterministic minimum-norm ``G`` over BZ surface images.

    A zero vector is always preferred, reproducing phono3py's Normal-process
    decision. Nonzero ties within its mesh-scaled BZ tolerance are broken
    lexicographically so results do not depend on container iteration order.
    """

    if not isinstance(triplet, MomentumTriplet):
        raise TypeError("triplet must be a MomentumTriplet.")
    images = _validated_address_images(triplet.mesh, address_images)
    metric = _reciprocal_metric(primitive_lattice)
    tolerance_factor = _distance_tolerance_factor(tolerance_factor)
    niggli_tolerance = _positive_tolerance(
        niggli_tolerance,
        name="niggli_tolerance",
    )
    reduced_reciprocal = _validated_reduced_reciprocal_lattice(
        primitive_lattice,
        reduced_reciprocal_lattice,
        niggli_tolerance=niggli_tolerance,
    )
    distance_tolerance = _phono3py_bz_distance_tolerance(
        triplet.mesh,
        reduced_reciprocal,
        tolerance_factor,
    )
    selected = tuple(images[index] for index in triplet.indices)
    periods = triplet.mesh.shape
    candidates: list[
        tuple[
            float,
            IntegerTriple,
            tuple[IntegerTriple, IntegerTriple, IntegerTriple],
        ]
    ] = []
    for first, second, third in product(*selected):
        total = tuple(
            first[axis] + second[axis] + third[axis]
            for axis in range(3)
        )
        if any(
            component % period
            for component, period in zip(total, periods, strict=True)
        ):
            raise ValueError(
                "A BZ surface image changed the triplet residue class."
            )
        reciprocal = tuple(
            component // period
            for component, period in zip(total, periods, strict=True)
        )
        reciprocal_array = np.array(reciprocal, dtype=np.float64)
        norm_squared = float(reciprocal_array @ metric @ reciprocal_array)
        candidates.append(
            (
                norm_squared,
                reciprocal,
                (
                    first.components,
                    second.components,
                    third.components,
                ),
            )
        )
    minimum = min(value[0] for value in candidates)
    if any(value[1] == (0, 0, 0) for value in candidates):
        return (0, 0, 0)
    tied = [
        value
        for value in candidates
        if value[0] < minimum + distance_tolerance
    ]
    return min(tied, key=lambda value: (value[1], value[2]))[1]


triplets_at_q = enumerate_target_triplets


__all__ = [
    "GRID_INDEX_ORDER",
    "PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR",
    "PHONO3PY_4_4_NIGGLI_TOLERANCE",
    "PHONO3PY_TRIPLET_CONVENTION",
    "GridAddress",
    "IntegerTriple",
    "MomentumProcess",
    "MomentumTriplet",
    "UniformBZMesh",
    "enumerate_target_triplets",
    "minimum_bz_reciprocal_vector",
    "niggli_reduced_reciprocal_lattice",
    "shortest_bz_address_images",
    "triplet_is_normal_on_bz_surface",
    "triplets_at_q",
]
