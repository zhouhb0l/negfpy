"""Strict comparison with the pinned phono3py silicon reference artifact.

This module is a validation layer, not another three-phonon solver.  It reads
the HDF5 artifact produced by ``generate_phono3py_si_reference.py`` and
compares independently computed NEGFPy fields without changing either
calculation.

The layer deliberately refuses three tempting but invalid shortcuts:

* phonon frequencies and eigenvectors on a mesh are not relabelled as
  independently generated random-q dynamical matrices;
* individual entries of a three-mode interaction tensor are not compared
  inside degenerate subspaces--the complete subspace ``|V3|^2`` sum is used;
* phase-space arrays are compared only when their Gaussian, cutoff, channel,
  and normalization definitions are explicitly identical.

The reference stores grid-point inputs in generalized-regular-grid (GRG)
X-fast order, while phono3py APIs and triplets use Brillouin-zone-grid (BZG)
indices.  Every route comparison therefore uses the archived ``grg2bzg`` and
``bzg2grg`` maps; identity mapping is never assumed.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np

from .bulk_harmonic import (
    PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ,
    BulkPhononSpectrum,
)
from .bulk_three_phonon import (
    BulkThreePhononInteractions,
    BulkThreePhononLinewidths,
)


Array = np.ndarray
IssueKind = Literal["schema_gap", "definition_mismatch"]

_REFERENCE_SCHEMA_NAME = "negfpy_phono3py_si_reference"
_REFERENCE_SCHEMA_VERSION = 1
_PINNED_VERSION = "4.4.0"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_FC2_QPOINT_CONVENTION = "primitive_reciprocal_fractional"
_FC2_MATRIX_UNITS = "eV/(angstrom^2*amu)"
_FC2_GAUGE = "phonopy_shortest_vector"
_FC2_POINT_SELECTION = "committed_literal_non_mesh_qpoints"
_FC2_RANDOM_QPOINTS = np.asarray(
    (
        (0.125, 0.25, -0.375),
        (0.3141592653589793, -0.2718281828459045, 0.1618033988749895),
        (-0.4375, 0.09375, 0.28125),
        (0.03125, -0.21875, 0.40625),
        (-0.1924500897298753, 1.0 / 3.0, -0.1178511301977579),
        (0.24609375, 0.37109375, -0.46484375),
        (-0.35546875, -0.14453125, 0.21484375),
        (0.419921875, -0.337890625, -0.072265625),
    ),
    dtype=np.float64,
)
_FC2_RANDOM_QPOINTS.setflags(write=False)

_PHASE_DEFINITION = (
    "sum_triplet,band1,band2("
    "triplet_weight*integration_weight)/Nq"
)
_PHASE_CUTOFF_RULE = "strict_all_three_frequencies_above_cutoff"
_PHASE_CHANNEL_0 = "Gaussian delta(omega-f1-f2): decay/sum channel"
_PHASE_CHANNEL_1 = (
    "Gaussian delta(omega+f1-f2) minus "
    "delta(omega-f1+f2): signed difference channel"
)
_PHASE_CANDIDATE_METADATA = {
    "phase_space_units": "1/cyclic_THz",
    "phase_space_definition": _PHASE_DEFINITION,
    "phase_space_cutoff_rule": _PHASE_CUTOFF_RULE,
    "phase_space_channel_0": _PHASE_CHANNEL_0,
    "phase_space_channel_1": _PHASE_CHANNEL_1,
    "phase_space_normalization": "complete_full_mesh_1_over_Nq",
    "sigma_cutoff": None,
}


class Phono3pyReferenceSchemaError(ValueError):
    """Raised when a purported reference artifact violates its base schema."""


def _readonly(value: Any, *, dtype: Any) -> Array:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_text(value: Any, *, name: str) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise Phono3pyReferenceSchemaError(
                f"{name} is not valid UTF-8."
            ) from exc
    if not isinstance(value, str) or not value:
        raise Phono3pyReferenceSchemaError(
            f"{name} must be a non-empty string."
        )
    return value


def _as_bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise Phono3pyReferenceSchemaError(f"{name} must be boolean.")
    return bool(value)


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative scalar.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar.")
    return result


def _finite_positive(value: Any, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "Reading a phono3py reference requires the optional h5py package."
        ) from exc
    return h5py


def _dataset(group: Any, name: str) -> Any:
    if name not in group:
        raise Phono3pyReferenceSchemaError(
            f"Missing HDF5 dataset or group {group.name}/{name}."
        )
    return group[name]


def _attribute(container: Any, name: str) -> Any:
    if name not in container.attrs:
        raise Phono3pyReferenceSchemaError(
            f"Missing HDF5 attribute {container.name}@{name}."
        )
    return container.attrs[name]


def _require_units(dataset: Any, expected: str) -> None:
    actual = _as_text(
        _attribute(dataset, "units"),
        name=f"{dataset.name}@units",
    )
    if actual != expected:
        raise Phono3pyReferenceSchemaError(
            f"{dataset.name} units must be {expected!r}, got {actual!r}."
        )


def _checksum_from_sidecar(path: Path) -> str:
    sidecar = path.parent / "SHA256SUMS"
    if not sidecar.is_file() or sidecar.is_symlink():
        raise Phono3pyReferenceSchemaError(
            f"Required checksum sidecar is missing: {sidecar}."
        )
    matches: list[str] = []
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise Phono3pyReferenceSchemaError(
            "SHA256SUMS must be readable UTF-8 text."
        ) from exc
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise Phono3pyReferenceSchemaError(
                "SHA256SUMS contains a malformed line."
            )
        digest, name = line[:64], line[66:]
        if _SHA256_RE.fullmatch(digest) is None:
            raise Phono3pyReferenceSchemaError(
                "SHA256SUMS contains a malformed digest."
            )
        if name == path.name:
            matches.append(digest.lower())
    if len(matches) != 1:
        raise Phono3pyReferenceSchemaError(
            "SHA256SUMS must contain exactly one entry for the reference HDF5."
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class ReferenceIssue:
    """One explicit reason that a comparison is unavailable."""

    kind: IssueKind
    code: str
    message: str
    hdf5_path: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("schema_gap", "definition_mismatch"):
            raise ValueError("Unsupported reference issue kind.")
        for name in ("code", "message"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string.")
        if self.hdf5_path is not None and (
            not isinstance(self.hdf5_path, str) or not self.hdf5_path
        ):
            raise ValueError("hdf5_path must be None or a non-empty string.")


@dataclass(frozen=True, slots=True, eq=False)
class Phono3pySiliconReference:
    """Detached light-weight index and scalar data from one reference HDF5."""

    path: Path
    sha256: str
    mesh: tuple[int, int, int]
    d_diag: tuple[int, int, int]
    grid_points_grg: Array
    grid_points_bzg: Array
    band_indices: Array
    temperatures_k: Array
    sigmas_thz: Array
    grg2bzg: Array
    bzg2grg: Array
    bz_addresses: Array
    frequencies_thz: Array
    gamma_thz_hwhm: Array
    target_groups: Mapping[int, str]
    make_r0_average: bool
    cutoff_frequency_thz: float
    fc2_random_qpoints: Array | None
    fc2_dynamical_matrices: Array | None
    fc2_hermitianized: bool | None
    schema_issues: tuple[ReferenceIssue, ...]

    def __post_init__(self) -> None:
        unresolved_path = Path(self.path)
        if unresolved_path.is_symlink():
            raise ValueError("path must not be a symbolic link.")
        path = unresolved_path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("path must be a regular non-symlink file.")
        digest = self.sha256.lower()
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("sha256 must be a 64-hex digest.")
        mesh = tuple(int(value) for value in self.mesh)
        d_diag = tuple(int(value) for value in self.d_diag)
        if len(mesh) != 3 or len(d_diag) != 3:
            raise ValueError("mesh and d_diag must contain three integers.")

        arrays = {
            "grid_points_grg": _readonly(
                self.grid_points_grg, dtype=np.int64
            ),
            "grid_points_bzg": _readonly(
                self.grid_points_bzg, dtype=np.int64
            ),
            "band_indices": _readonly(self.band_indices, dtype=np.int64),
            "temperatures_k": _readonly(
                self.temperatures_k, dtype=np.float64
            ),
            "sigmas_thz": _readonly(self.sigmas_thz, dtype=np.float64),
            "grg2bzg": _readonly(self.grg2bzg, dtype=np.int64),
            "bzg2grg": _readonly(self.bzg2grg, dtype=np.int64),
            "bz_addresses": _readonly(self.bz_addresses, dtype=np.int64),
            "frequencies_thz": _readonly(
                self.frequencies_thz, dtype=np.float64
            ),
            "gamma_thz_hwhm": _readonly(
                self.gamma_thz_hwhm, dtype=np.float64
            ),
        }
        if self.fc2_random_qpoints is None:
            fc2_qpoints = None
            fc2_matrices = None
        else:
            fc2_qpoints = _readonly(
                self.fc2_random_qpoints,
                dtype=np.float64,
            )
            if self.fc2_dynamical_matrices is None:
                raise ValueError(
                    "fc2_dynamical_matrices is required with random qpoints."
                )
            fc2_matrices = _readonly(
                self.fc2_dynamical_matrices,
                dtype=np.complex128,
            )
        target_groups = MappingProxyType(
            {int(key): str(value) for key, value in self.target_groups.items()}
        )

        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "d_diag", d_diag)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "target_groups", target_groups)
        object.__setattr__(self, "fc2_random_qpoints", fc2_qpoints)
        object.__setattr__(self, "fc2_dynamical_matrices", fc2_matrices)
        object.__setattr__(
            self,
            "schema_issues",
            tuple(self.schema_issues),
        )

    @property
    def n_qpoints(self) -> int:
        return int(self.grg2bzg.size)

    @property
    def n_bands(self) -> int:
        return int(self.frequencies_thz.shape[1])

    def frequencies_in_grg_order(self) -> Array:
        """Return detached frequencies indexed by NEGFPy's GRG indices."""

        return _readonly(
            self.frequencies_thz[self.grg2bzg],
            dtype=np.float64,
        )


def _validate_target_groups(
    h5: Any,
    *,
    grid_points_grg: Array,
    grid_points_bzg: Array,
    band_indices: Array,
    n_qpoints: int,
    n_bands: int,
    bzg2grg: Array,
) -> Mapping[int, str]:
    targets = _dataset(h5, "target_q")
    found: dict[int, str] = {}
    for name in targets:
        group = targets[name]
        grg = int(_attribute(group, "grid_point_grg"))
        bzg = int(_attribute(group, "grid_point_bzg"))
        if grg in found:
            raise Phono3pyReferenceSchemaError(
                f"Duplicate target GRG index {grg}."
            )
        if grg < 0 or grg >= n_qpoints:
            raise Phono3pyReferenceSchemaError(
                f"Target GRG index {grg} is outside the mesh."
            )
        if bzg < 0 or bzg >= bzg2grg.size or int(bzg2grg[bzg]) != grg:
            raise Phono3pyReferenceSchemaError(
                f"Target {grg} has inconsistent GRG/BZG mapping."
            )
        triplets = _dataset(group, "triplets_bzg")
        weights = np.asarray(_dataset(group, "triplet_weights"))
        strength = _dataset(group, "interaction_strength")
        integration = _dataset(group, "integration_weights")
        if triplets.shape != (n_qpoints, 3):
            raise Phono3pyReferenceSchemaError(
                f"{triplets.name} must have shape {(n_qpoints, 3)}."
            )
        if weights.shape != (n_qpoints,) or int(np.sum(weights)) != n_qpoints:
            raise Phono3pyReferenceSchemaError(
                f"{group.name} does not contain a complete full-mesh weight."
            )
        if not np.array_equal(weights, np.ones(n_qpoints, dtype=weights.dtype)):
            raise Phono3pyReferenceSchemaError(
                f"{group.name} must use unit weights with mesh symmetry disabled."
            )
        expected_strength_shape = (
            n_qpoints,
            band_indices.size,
            n_bands,
            n_bands,
        )
        if strength.shape != expected_strength_shape:
            raise Phono3pyReferenceSchemaError(
                f"{strength.name} must have shape {expected_strength_shape}."
            )
        _require_units(strength, "eV^2")
        axis_order = _as_text(
            _attribute(strength, "axis_order"),
            name=f"{strength.name}@axis_order",
        )
        if axis_order != "triplet,target_band,band1,band2":
            raise Phono3pyReferenceSchemaError(
                f"Unsupported interaction axis order {axis_order!r}."
            )
        expected_integration_shape = (
            2,
            n_qpoints,
            band_indices.size,
            n_bands,
            n_bands,
        )
        if integration.shape != expected_integration_shape:
            raise Phono3pyReferenceSchemaError(
                f"{integration.name} must have shape "
                f"{expected_integration_shape}."
            )
        _require_units(integration, "1/cyclic_THz")
        integration_axis = _as_text(
            _attribute(integration, "axis_order"),
            name=f"{integration.name}@axis_order",
        )
        if integration_axis != (
            "channel,triplet,target_frequency,band1,band2"
        ):
            raise Phono3pyReferenceSchemaError(
                f"Unsupported integration-weight axis order "
                f"{integration_axis!r}."
            )
        for attribute_name, expected_channel in (
            ("channel_0", _PHASE_CHANNEL_0),
            ("channel_1", _PHASE_CHANNEL_1),
        ):
            channel = _as_text(
                _attribute(integration, attribute_name),
                name=f"{integration.name}@{attribute_name}",
            )
            if channel != expected_channel:
                raise Phono3pyReferenceSchemaError(
                    f"{integration.name}@{attribute_name} has an unsupported "
                    "channel definition."
                )
        triplet_values = np.asarray(triplets, dtype=np.int64)
        if np.any(
            (triplet_values < 0) | (triplet_values >= bzg2grg.size)
        ):
            raise Phono3pyReferenceSchemaError(
                f"{triplets.name} contains an invalid BZG index."
            )
        triplets_grg = bzg2grg[triplet_values]
        if not np.all(triplets_grg[:, 0] == grg):
            raise Phono3pyReferenceSchemaError(
                f"{triplets.name} has an inconsistent target leg."
            )
        if np.unique(triplets_grg[:, 1]).size != n_qpoints:
            raise Phono3pyReferenceSchemaError(
                f"{triplets.name} does not enumerate every q1 GRG index once."
            )
        found[grg] = group.name

    expected = {
        int(grg): int(bzg)
        for grg, bzg in zip(
            grid_points_grg,
            grid_points_bzg,
            strict=True,
        )
    }
    if set(found) != set(expected):
        raise Phono3pyReferenceSchemaError(
            "target_q groups do not match settings/grid_points_grg."
        )
    for grg, bzg in expected.items():
        group = h5[found[grg]]
        if int(_attribute(group, "grid_point_bzg")) != bzg:
            raise Phono3pyReferenceSchemaError(
                f"Target GRG {grg} conflicts with settings/grid_points_bzg."
            )
    return MappingProxyType(found)


def _read_optional_fc2_random_q(
    h5: Any,
    *,
    n_bands: int,
) -> tuple[Array | None, Array | None, bool | None, tuple[ReferenceIssue, ...]]:
    if "fc2_random_q" not in h5:
        return (
            None,
            None,
            None,
            (
                ReferenceIssue(
                    kind="schema_gap",
                    code="missing_fc2_random_q",
                    message=(
                        "The artifact contains mesh phonons but no independently "
                        "generated random-q dynamical matrices. Frequencies and "
                        "eigenvectors cannot serve as that independent oracle."
                    ),
                    hdf5_path="/fc2_random_q",
                ),
            ),
        )
    group = h5["fc2_random_q"]
    qpoints = np.asarray(_dataset(group, "qpoints"), dtype=np.float64)
    matrices = np.asarray(
        _dataset(group, "dynamical_matrices"),
        dtype=np.complex128,
    )
    if (
        qpoints.ndim != 2
        or qpoints.shape[0] == 0
        or qpoints.shape[1:] != (3,)
    ):
        raise Phono3pyReferenceSchemaError(
            "/fc2_random_q/qpoints must have nonempty shape (nq, 3)."
        )
    if matrices.shape != (qpoints.shape[0], n_bands, n_bands):
        raise Phono3pyReferenceSchemaError(
            "/fc2_random_q/dynamical_matrices has an inconsistent shape."
        )
    _require_units(
        _dataset(group, "qpoints"),
        "primitive_reciprocal_fractional",
    )
    _require_units(
        _dataset(group, "dynamical_matrices"),
        _FC2_MATRIX_UNITS,
    )
    if not (
        np.all(np.isfinite(qpoints))
        and np.all(np.isfinite(matrices.real))
        and np.all(np.isfinite(matrices.imag))
    ):
        raise Phono3pyReferenceSchemaError(
            "Random-q FC2 reference arrays must be finite."
        )
    if np.unique(qpoints, axis=0).shape[0] != qpoints.shape[0]:
        raise Phono3pyReferenceSchemaError(
            "Random-q FC2 reference contains duplicate qpoints."
        )
    convention = _as_text(
        _attribute(group, "qpoint_coordinates"),
        name="/fc2_random_q@qpoint_coordinates",
    )
    units = _as_text(
        _attribute(group, "dynamical_matrix_units"),
        name="/fc2_random_q@dynamical_matrix_units",
    )
    gauge = _as_text(
        _attribute(group, "gauge"),
        name="/fc2_random_q@gauge",
    )
    point_selection = _as_text(
        _attribute(group, "point_selection"),
        name="/fc2_random_q@point_selection",
    )
    if convention != _FC2_QPOINT_CONVENTION:
        raise Phono3pyReferenceSchemaError(
            "Unsupported random-q coordinate convention."
        )
    if units != _FC2_MATRIX_UNITS:
        raise Phono3pyReferenceSchemaError(
            "Unsupported random-q dynamical-matrix units."
        )
    if gauge != _FC2_GAUGE:
        raise Phono3pyReferenceSchemaError(
            "Unsupported random-q dynamical-matrix gauge."
        )
    if point_selection != _FC2_POINT_SELECTION:
        raise Phono3pyReferenceSchemaError(
            "Unsupported random-q point-selection provenance."
        )
    if not np.array_equal(qpoints, _FC2_RANDOM_QPOINTS):
        raise Phono3pyReferenceSchemaError(
            "Random-q points differ from the committed independent FC2 "
            "validation selection."
        )
    hermitianized = _as_bool(
        _attribute(group, "hermitianized"),
        name="/fc2_random_q@hermitianized",
    )
    if hermitianized and not np.allclose(
        matrices,
        matrices.conj().transpose(0, 2, 1),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise Phono3pyReferenceSchemaError(
            "Hermitianized random-q reference matrices are not Hermitian."
        )
    return qpoints, matrices, hermitianized, ()


def read_phono3py_silicon_reference(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    require_checksum_sidecar: bool = True,
) -> Phono3pySiliconReference:
    """Read and validate the light-weight index of a generated reference."""

    unresolved_path = Path(path)
    if unresolved_path.is_symlink():
        raise ValueError("Reference path must not be a symbolic link.")
    reference_path = unresolved_path.resolve(strict=True)
    if not reference_path.is_file():
        raise ValueError("Reference path must be a regular non-symlink file.")
    actual_sha256 = _sha256(reference_path)
    if expected_sha256 is not None:
        if (
            not isinstance(expected_sha256, str)
            or _SHA256_RE.fullmatch(expected_sha256) is None
        ):
            raise ValueError("expected_sha256 must be a 64-hex digest.")
        if actual_sha256 != expected_sha256.lower():
            raise Phono3pyReferenceSchemaError(
                "Reference HDF5 does not match expected_sha256."
            )
    if require_checksum_sidecar:
        sidecar_sha256 = _checksum_from_sidecar(reference_path)
        if actual_sha256 != sidecar_sha256:
            raise Phono3pyReferenceSchemaError(
                "Reference HDF5 does not match SHA256SUMS."
            )

    h5py = _require_h5py()
    with h5py.File(reference_path, "r") as h5:
        schema_name = _as_text(
            _attribute(h5, "schema_name"),
            name="/@schema_name",
        )
        if schema_name != _REFERENCE_SCHEMA_NAME:
            raise Phono3pyReferenceSchemaError(
                f"Unsupported reference schema {schema_name!r}."
            )
        if int(_attribute(h5, "schema_version")) != _REFERENCE_SCHEMA_VERSION:
            raise Phono3pyReferenceSchemaError(
                "Unsupported reference schema version."
            )
        for name in ("phono3py_version", "phonopy_version"):
            version = _as_text(_attribute(h5, name), name=f"/@{name}")
            if version != _PINNED_VERSION:
                raise Phono3pyReferenceSchemaError(
                    f"{name} must be exactly {_PINNED_VERSION}, got {version}."
                )
        expected_root_strings = {
            "frequency_units": "cyclic_THz",
            "gamma_units": "cyclic_THz_HWHM",
            "interaction_strength_units": "eV^2",
        }
        for name, expected in expected_root_strings.items():
            actual = _as_text(_attribute(h5, name), name=f"/@{name}")
            if actual != expected:
                raise Phono3pyReferenceSchemaError(
                    f"/@{name} must equal {expected!r}."
                )
        if _as_bool(
            _attribute(h5, "is_mesh_symmetry"),
            name="/@is_mesh_symmetry",
        ):
            raise Phono3pyReferenceSchemaError(
                "Comparison requires is_mesh_symmetry=False."
            )
        if _as_bool(_attribute(h5, "is_nac"), name="/@is_nac"):
            raise Phono3pyReferenceSchemaError(
                "The pinned silicon comparison requires is_nac=False."
            )
        make_r0_average = _as_bool(
            _attribute(h5, "make_r0_average"),
            name="/@make_r0_average",
        )
        cutoff = _finite_nonnegative(
            _attribute(h5, "cutoff_frequency_THz"),
            name="/@cutoff_frequency_THz",
        )
        frequency_factor = float(
            _attribute(h5, "frequency_unit_conversion_factor")
        )
        if frequency_factor != PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ:
            raise Phono3pyReferenceSchemaError(
                "The reference frequency conversion factor is not the pinned "
                "phono3py 4.4.0 value."
            )
        if _as_bool(
            _attribute(h5, "symmetrize_fc3q"),
            name="/@symmetrize_fc3q",
        ):
            raise Phono3pyReferenceSchemaError(
                "The independent NEGFPy comparison requires "
                "symmetrize_fc3q=False."
            )

        settings = _dataset(h5, "settings")
        mesh_array = np.asarray(_dataset(settings, "mesh"))
        if (
            mesh_array.shape != (3,)
            or mesh_array.dtype.kind not in "iu"
            or np.any(mesh_array <= 0)
        ):
            raise Phono3pyReferenceSchemaError(
                "/settings/mesh must contain three positive integers."
            )
        mesh = tuple(int(value) for value in mesh_array)
        grid_points_grg = np.asarray(
            _dataset(settings, "grid_points_grg"),
            dtype=np.int64,
        )
        grid_points_bzg = np.asarray(
            _dataset(settings, "grid_points_bzg"),
            dtype=np.int64,
        )
        band_indices = np.asarray(
            _dataset(settings, "band_indices"),
            dtype=np.int64,
        )
        temperatures = np.asarray(
            _dataset(settings, "temperatures"),
            dtype=np.float64,
        )
        sigmas = np.asarray(
            _dataset(settings, "sigmas"),
            dtype=np.float64,
        )
        sigma_cutoff = _attribute(settings, "sigma_cutoff")
        if isinstance(sigma_cutoff, bytes):
            sigma_cutoff = sigma_cutoff.decode("utf-8")
        if sigma_cutoff != "None":
            raise Phono3pyReferenceSchemaError(
                "NEGFPy's current Gaussian oracle has no finite sigma cutoff; "
                "the reference must declare settings@sigma_cutoff='None'."
            )
        if (
            grid_points_grg.ndim != 1
            or grid_points_grg.size == 0
            or grid_points_bzg.shape != grid_points_grg.shape
            or np.unique(grid_points_grg).size != grid_points_grg.size
        ):
            raise Phono3pyReferenceSchemaError(
                "Reference target grid-point selections are invalid."
            )
        if (
            band_indices.ndim != 1
            or band_indices.size == 0
            or np.unique(band_indices).size != band_indices.size
        ):
            raise Phono3pyReferenceSchemaError(
                "Reference band selection is invalid."
            )
        if (
            temperatures.ndim != 1
            or temperatures.size == 0
            or not np.all(np.isfinite(temperatures))
            or np.any(temperatures < 0.0)
        ):
            raise Phono3pyReferenceSchemaError(
                "Reference temperatures are invalid."
            )
        if (
            sigmas.shape != (1,)
            or not np.isfinite(sigmas[0])
            or sigmas[0] <= 0.0
        ):
            raise Phono3pyReferenceSchemaError(
                "The comparison layer requires one positive Gaussian sigma."
            )

        grid = _dataset(h5, "grid")
        expected_grid_attributes = {
            "GRG_index_order": "x_fast",
            "target_grid_point_input_space": "GRG",
            "triplet_grid_point_space": "BZG",
        }
        for name, expected in expected_grid_attributes.items():
            actual = _as_text(
                _attribute(grid, name),
                name=f"/grid@{name}",
            )
            if actual != expected:
                raise Phono3pyReferenceSchemaError(
                    f"/grid@{name} must equal {expected!r}."
                )
        d_diag_array = np.asarray(_dataset(grid, "D_diag"))
        if (
            d_diag_array.shape != (3,)
            or d_diag_array.dtype.kind not in "iu"
            or np.any(d_diag_array <= 0)
        ):
            raise Phono3pyReferenceSchemaError(
                "/grid/D_diag must contain three positive integers."
            )
        d_diag = tuple(int(value) for value in d_diag_array)
        n_qpoints = int(np.prod(d_diag_array))
        if d_diag != mesh:
            raise Phono3pyReferenceSchemaError(
                "NEGFPy's uniform-mesh comparison requires D_diag to equal "
                "the requested mesh without an SNF-axis transformation."
            )
        q_transform = np.asarray(_dataset(grid, "Q"), dtype=np.int64)
        grid_shift = np.asarray(_dataset(grid, "PS"), dtype=np.int64)
        if q_transform.shape != (3, 3) or not np.array_equal(
            q_transform,
            np.eye(3, dtype=np.int64),
        ):
            raise Phono3pyReferenceSchemaError(
                "NEGFPy's current UniformBZMesh comparison requires grid/Q=I."
            )
        if grid_shift.shape != (3,) or np.any(grid_shift != 0):
            raise Phono3pyReferenceSchemaError(
                "NEGFPy's current UniformBZMesh comparison requires an "
                "unshifted grid (grid/PS=0)."
            )
        grg2bzg = np.asarray(_dataset(grid, "grg2bzg"), dtype=np.int64)
        bzg2grg = np.asarray(_dataset(grid, "bzg2grg"), dtype=np.int64)
        addresses = np.asarray(_dataset(grid, "addresses"), dtype=np.int64)
        if grg2bzg.shape != (n_qpoints,):
            raise Phono3pyReferenceSchemaError(
                "/grid/grg2bzg has an inconsistent shape."
            )
        if (
            bzg2grg.ndim != 1
            or addresses.shape != (bzg2grg.size, 3)
            or np.any((grg2bzg < 0) | (grg2bzg >= bzg2grg.size))
            or np.any((bzg2grg < 0) | (bzg2grg >= n_qpoints))
        ):
            raise Phono3pyReferenceSchemaError(
                "Reference BZG/GRG maps or addresses are invalid."
            )
        if not np.array_equal(
            bzg2grg[grg2bzg],
            np.arange(n_qpoints, dtype=np.int64),
        ):
            raise Phono3pyReferenceSchemaError(
                "grg2bzg and bzg2grg are not inverse on primary BZG points."
            )
        if np.any(
            (grid_points_grg < 0) | (grid_points_grg >= n_qpoints)
        ):
            raise Phono3pyReferenceSchemaError(
                "A target GRG index is outside the reference mesh."
            )
        if not np.array_equal(
            grg2bzg[grid_points_grg],
            grid_points_bzg,
        ):
            raise Phono3pyReferenceSchemaError(
                "settings grid_points_bzg is not grg2bzg[grid_points_grg]."
            )

        phonons = _dataset(h5, "phonons")
        frequencies = np.asarray(
            _dataset(phonons, "frequencies"),
            dtype=np.float64,
        )
        if (
            frequencies.ndim != 2
            or frequencies.shape[0] != bzg2grg.size
            or not np.all(np.isfinite(frequencies))
        ):
            raise Phono3pyReferenceSchemaError(
                "/phonons/frequencies has an invalid shape or values."
            )
        n_bands = int(frequencies.shape[1])
        if np.any((band_indices < 0) | (band_indices >= n_bands)):
            raise Phono3pyReferenceSchemaError(
                "Reference band selection is outside the phonon bands."
            )
        eigenvectors = _dataset(phonons, "eigenvectors")
        if eigenvectors.shape != (bzg2grg.size, n_bands, n_bands):
            raise Phono3pyReferenceSchemaError(
                "/phonons/eigenvectors has an inconsistent shape."
            )
        phonon_addresses = np.asarray(
            _dataset(phonons, "bz_grid_addresses"),
            dtype=np.int64,
        )
        if not np.array_equal(phonon_addresses, addresses):
            raise Phono3pyReferenceSchemaError(
                "Phonon and grid BZG addresses differ."
            )
        target_band_frequencies = np.asarray(
            _dataset(phonons, "target_band_frequencies"),
            dtype=np.float64,
        )
        expected_target_frequencies = frequencies[
            grid_points_bzg[:, None],
            band_indices[None, :],
        ]
        if not np.array_equal(
            target_band_frequencies,
            expected_target_frequencies,
        ):
            raise Phono3pyReferenceSchemaError(
                "Target-band frequencies do not match the archived BZG targets."
            )

        gamma_dataset = _dataset(_dataset(h5, "gamma"), "values")
        _require_units(gamma_dataset, "cyclic_THz_HWHM")
        gamma_axis = _as_text(
            _attribute(gamma_dataset, "axis_order"),
            name="/gamma/values@axis_order",
        )
        if gamma_axis != "sigma,temperature,target_grid_point,target_band":
            raise Phono3pyReferenceSchemaError(
                "Unsupported gamma axis order."
            )
        gamma_raw = np.asarray(gamma_dataset, dtype=np.float64)
        expected_gamma = (
            1,
            temperatures.size,
            grid_points_grg.size,
            band_indices.size,
        )
        if gamma_raw.shape != expected_gamma or not np.all(
            np.isfinite(gamma_raw)
        ):
            raise Phono3pyReferenceSchemaError(
                f"/gamma/values must have finite shape {expected_gamma}."
            )
        if np.any(gamma_raw < -1.0e-14):
            raise Phono3pyReferenceSchemaError(
                "/gamma/values contains a materially negative linewidth."
            )

        target_groups = _validate_target_groups(
            h5,
            grid_points_grg=grid_points_grg,
            grid_points_bzg=grid_points_bzg,
            band_indices=band_indices,
            n_qpoints=n_qpoints,
            n_bands=n_bands,
            bzg2grg=bzg2grg,
        )
        fc2_q, fc2_d, fc2_hermitianized, fc2_issues = (
            _read_optional_fc2_random_q(h5, n_bands=n_bands)
        )

        phase_issues: list[ReferenceIssue] = []
        for grg, group_name in target_groups.items():
            phase = _dataset(
                h5[group_name],
                "derived_phase_space_channel_sums",
            )
            _require_units(phase, "1/cyclic_THz")
            if np.asarray(phase).shape != (2, band_indices.size) or not np.all(
                np.isfinite(np.asarray(phase))
            ):
                raise Phono3pyReferenceSchemaError(
                    f"{phase.name} has an invalid shape or values."
                )
            phase_axis = _as_text(
                _attribute(phase, "axis_order"),
                name=f"{phase.name}@axis_order",
            )
            if phase_axis != "channel,target_frequency":
                raise Phono3pyReferenceSchemaError(
                    f"{phase.name} has an unsupported axis order."
                )
            if "cutoff_frequency_THz" not in phase.attrs:
                phase_issues.append(
                    ReferenceIssue(
                        kind="schema_gap",
                        code="phase_space_cutoff_semantics_missing",
                        message=(
                            "The archived phase-space sum does not declare "
                            "whether the phonon cutoff mask was applied, so it "
                            "cannot be compared with NEGFPy's masked diagnostic."
                        ),
                        hdf5_path=phase.name,
                    )
                )
            if "cutoff_rule" not in phase.attrs:
                phase_issues.append(
                    ReferenceIssue(
                        kind="schema_gap",
                        code="phase_space_cutoff_rule_missing",
                        message=(
                            "The archived phase-space sum lacks an explicit "
                            "three-leg cutoff rule."
                        ),
                        hdf5_path=phase.name,
                    )
                )

    return Phono3pySiliconReference(
        path=reference_path,
        sha256=actual_sha256,
        mesh=mesh,
        d_diag=d_diag,
        grid_points_grg=grid_points_grg,
        grid_points_bzg=grid_points_bzg,
        band_indices=band_indices,
        temperatures_k=temperatures,
        sigmas_thz=sigmas,
        grg2bzg=grg2bzg,
        bzg2grg=bzg2grg,
        bz_addresses=addresses,
        frequencies_thz=frequencies,
        gamma_thz_hwhm=gamma_raw[0],
        target_groups=target_groups,
        make_r0_average=make_r0_average,
        cutoff_frequency_thz=cutoff,
        fc2_random_qpoints=fc2_q,
        fc2_dynamical_matrices=fc2_d,
        fc2_hermitianized=fc2_hermitianized,
        schema_issues=tuple(fc2_issues) + tuple(phase_issues),
    )


def _ensure_unchanged(reference: Phono3pySiliconReference) -> None:
    if _sha256(reference.path) != reference.sha256:
        raise Phono3pyReferenceSchemaError(
            "Reference HDF5 changed after it was loaded."
        )


@contextmanager
def _open_reference(
    reference: Phono3pySiliconReference,
    *,
    verify_hash: bool = True,
) -> Iterator[Any]:
    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    if verify_hash:
        _ensure_unchanged(reference)
    h5py = _require_h5py()
    with h5py.File(reference.path, "r") as h5:
        yield h5


@dataclass(frozen=True, slots=True)
class FC2RandomQComparison:
    reference_sha256: str
    available: bool
    passed: bool
    tolerance: float
    max_absolute_error: float | None
    max_scaled_error: float | None
    per_q_absolute_error: tuple[float, ...]
    issues: tuple[ReferenceIssue, ...]


def compare_fc2_random_q(
    reference: Phono3pySiliconReference,
    candidate: BulkPhononSpectrum | None,
    *,
    tolerance: float = 1.0e-10,
    verify_reference_hash: bool = True,
) -> FC2RandomQComparison:
    """Compare random-q matrices with the certified absolute-error gate."""

    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    if not isinstance(verify_reference_hash, (bool, np.bool_)):
        raise TypeError("verify_reference_hash must be boolean.")
    if verify_reference_hash:
        _ensure_unchanged(reference)
    tolerance = _finite_nonnegative(tolerance, name="tolerance")
    fc2_issues = tuple(
        issue
        for issue in reference.schema_issues
        if issue.code.startswith("missing_fc2_")
    )
    if (
        reference.fc2_random_qpoints is None
        or reference.fc2_dynamical_matrices is None
    ):
        return FC2RandomQComparison(
            reference_sha256=reference.sha256,
            available=False,
            passed=False,
            tolerance=tolerance,
            max_absolute_error=None,
            max_scaled_error=None,
            per_q_absolute_error=(),
            issues=fc2_issues,
        )
    if not isinstance(candidate, BulkPhononSpectrum):
        raise TypeError(
            "candidate must be a BulkPhononSpectrum when FC2 data are available."
        )
    if candidate.hermitianized != reference.fc2_hermitianized:
        raise ValueError(
            "Candidate and reference random-q matrices use different "
            "Hermitianization choices."
        )
    if candidate.qpoints.shape != reference.fc2_random_qpoints.shape or not (
        np.allclose(
            candidate.qpoints,
            reference.fc2_random_qpoints,
            rtol=0.0,
            atol=1.0e-14,
        )
    ):
        raise ValueError(
            "Candidate qpoints do not match the archived random-q points."
        )
    if (
        candidate.dynamical_matrices.shape
        != reference.fc2_dynamical_matrices.shape
    ):
        raise ValueError(
            "Candidate random-q dynamical matrices have the wrong shape."
        )
    difference = np.abs(
        candidate.dynamical_matrices
        - reference.fc2_dynamical_matrices
    )
    per_q = np.max(difference, axis=(1, 2))
    reference_scale = np.max(
        np.abs(reference.fc2_dynamical_matrices),
        axis=(1, 2),
    )
    scaled = per_q / np.maximum(1.0, reference_scale)
    return FC2RandomQComparison(
        reference_sha256=reference.sha256,
        available=True,
        passed=bool(np.all(per_q <= tolerance)),
        tolerance=tolerance,
        max_absolute_error=float(np.max(per_q)),
        max_scaled_error=float(np.max(scaled)),
        per_q_absolute_error=tuple(float(value) for value in per_q),
        issues=(),
    )


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    return int(value)


@dataclass(frozen=True, slots=True)
class InteractionStrengthSample:
    """One requested scalar coordinate, promoted to its degenerate block."""

    target_grid_point: int
    q1_grid_point: int
    target_band: int
    band1: int
    band2: int

    def __post_init__(self) -> None:
        for name in (
            "target_grid_point",
            "q1_grid_point",
            "target_band",
            "band1",
            "band2",
        ):
            value = _integer(getattr(self, name), name=name)
            if value < 0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)


def _canonical_samples(
    samples: Sequence[InteractionStrengthSample | Sequence[int]],
) -> tuple[InteractionStrengthSample, ...]:
    if isinstance(samples, (str, bytes)):
        raise TypeError("samples must be a sequence.")
    result: list[InteractionStrengthSample] = []
    for item in samples:
        if isinstance(item, InteractionStrengthSample):
            sample = item
        else:
            try:
                values = tuple(item)
            except TypeError as exc:
                raise TypeError(
                    "Each interaction sample must contain five integers."
                ) from exc
            if len(values) != 5:
                raise ValueError(
                    "Each interaction sample must contain five integers."
                )
            sample = InteractionStrengthSample(*values)
        result.append(sample)
    if not result:
        raise ValueError("samples must not be empty.")
    return tuple(result)


def _degenerate_groups(frequencies: Array, tolerance: float) -> tuple[Array, ...]:
    groups: list[Array] = []
    used: set[int] = set()
    for first in range(frequencies.size):
        if first in used:
            continue
        group = [first]
        used.add(first)
        for candidate in range(first + 1, frequencies.size):
            if any(
                abs(frequencies[index] - frequencies[candidate]) < tolerance
                for index in group
            ):
                group.append(candidate)
                used.add(candidate)
        groups.append(np.asarray(group, dtype=np.int64))
    return tuple(groups)


def _group_for_band(
    frequencies: Array,
    band: int,
    tolerance: float,
) -> Array:
    if band < 0 or band >= frequencies.size:
        raise IndexError(f"Band {band} is outside [0, {frequencies.size}).")
    for group in _degenerate_groups(frequencies, tolerance):
        if band in group:
            return group
    raise AssertionError("Every band must belong to one degeneracy group.")


def _frequencies_match_with_cutoff(
    candidate_frequencies: Array,
    reference_frequencies: Array,
    *,
    cutoff_frequency_thz: float,
    frequency_atol_thz: float,
) -> bool:
    """Compare frequencies after canonicalizing only cutoff-inactive modes."""

    candidate_values = np.asarray(candidate_frequencies)
    reference_values = np.asarray(reference_frequencies)
    candidate_comparison = np.where(
        np.abs(candidate_values) <= cutoff_frequency_thz,
        0.0,
        candidate_values,
    )
    reference_comparison = np.where(
        np.abs(reference_values) <= cutoff_frequency_thz,
        0.0,
        reference_values,
    )
    return bool(
        np.allclose(
            candidate_comparison,
            reference_comparison,
            rtol=0.0,
            atol=frequency_atol_thz,
        )
    )


def _matching_group(
    *,
    candidate_frequencies: Array,
    reference_frequencies: Array,
    band: int,
    tolerance: float,
    cutoff_frequency_thz: float,
    frequency_atol: float,
    label: str,
) -> Array:
    if not _frequencies_match_with_cutoff(
        candidate_frequencies,
        reference_frequencies,
        cutoff_frequency_thz=cutoff_frequency_thz,
        frequency_atol_thz=frequency_atol,
    ):
        raise ValueError(
            f"Candidate/reference phonon frequencies differ at {label}; "
            "interaction gauge blocks cannot be matched."
        )
    candidate_group = _group_for_band(
        candidate_frequencies,
        band,
        tolerance,
    )
    reference_group = _group_for_band(
        reference_frequencies,
        band,
        tolerance,
    )
    if not np.array_equal(candidate_group, reference_group):
        raise ValueError(
            f"Candidate/reference degenerate subspaces differ at {label}."
        )
    return candidate_group


@dataclass(frozen=True, slots=True)
class InteractionBlockComparison:
    target_grid_point: int
    q1_grid_point: int
    q2_grid_point: int
    target_bands: tuple[int, ...]
    bands1: tuple[int, ...]
    bands2: tuple[int, ...]
    reference_sum_ev2: float
    candidate_sum_ev2: float
    absolute_error_ev2: float
    relative_error: float
    passed: bool


@dataclass(frozen=True, slots=True)
class InteractionStrengthComparison:
    reference_sha256: str
    available: bool
    complete: bool
    passed: bool
    relative_tolerance: float
    absolute_tolerance_ev2: float
    blocks: tuple[InteractionBlockComparison, ...]
    max_relative_error: float | None
    max_absolute_error_ev2: float | None
    issues: tuple[ReferenceIssue, ...]


def _position_map(values: Array) -> dict[int, int]:
    return {int(value): index for index, value in enumerate(values)}


def _route_map(q1_values: Array) -> dict[int, int]:
    mapping = {int(value): index for index, value in enumerate(q1_values)}
    if len(mapping) != q1_values.size:
        raise ValueError("A candidate/reference route contains duplicate q1.")
    return mapping


def compare_interaction_strengths(
    reference: Phono3pySiliconReference,
    candidate: BulkThreePhononInteractions,
    samples: Sequence[InteractionStrengthSample | Sequence[int]],
    *,
    relative_tolerance: float = 1.0e-8,
    absolute_tolerance_ev2: float = 1.0e-14,
    degenerate_tolerance_thz: float = 1.0e-4,
    frequency_atol_thz: float = 1.0e-8,
    verify_reference_hash: bool = True,
) -> InteractionStrengthComparison:
    """Compare sampled gauge-invariant ``|V3|^2`` subspace sums."""

    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    if not isinstance(verify_reference_hash, (bool, np.bool_)):
        raise TypeError("verify_reference_hash must be boolean.")
    if verify_reference_hash:
        _ensure_unchanged(reference)
    if not isinstance(candidate, BulkThreePhononInteractions):
        raise TypeError("candidate must be BulkThreePhononInteractions.")
    requested = _canonical_samples(samples)
    rtol = _finite_nonnegative(
        relative_tolerance,
        name="relative_tolerance",
    )
    atol = _finite_nonnegative(
        absolute_tolerance_ev2,
        name="absolute_tolerance_ev2",
    )
    degeneracy = _finite_positive(
        degenerate_tolerance_thz,
        name="degenerate_tolerance_thz",
    )
    frequency_atol = _finite_nonnegative(
        frequency_atol_thz,
        name="frequency_atol_thz",
    )
    if "mesh" not in candidate.metadata:
        raise ValueError("Candidate interaction metadata is missing mesh.")
    if tuple(candidate.metadata["mesh"]) != reference.mesh:
        raise ValueError("Candidate interaction mesh differs from the reference.")
    if candidate.q1_indices.shape[1] != reference.n_qpoints:
        raise ValueError("Candidate does not contain the full reference mesh.")
    if "make_r0_average" not in candidate.metadata:
        raise ValueError(
            "Candidate interaction metadata is missing make_r0_average."
        )
    if bool(candidate.metadata["make_r0_average"]) != (
        reference.make_r0_average
    ):
        raise ValueError(
            "Candidate and reference use different make_r0_average settings."
        )
    expected_interaction_metadata = {
        "interaction_strength_units": "eV^2",
        "factorial_divisor": 36.0,
        "mode_normalization_divisor": 8.0,
        "full_mesh_normalization": 1.0 / reference.n_qpoints,
    }
    for key, expected in expected_interaction_metadata.items():
        if key not in candidate.metadata or candidate.metadata[key] != expected:
            raise ValueError(
                f"Candidate interaction metadata {key!r} does not match the "
                "pinned phono3py convention."
            )
    candidate_frequencies = candidate.phonons.frequencies
    reference_frequencies = reference.frequencies_in_grg_order()
    if candidate_frequencies.shape != reference_frequencies.shape:
        raise ValueError(
            "Candidate/reference phonon spectra have different shapes."
        )

    candidate_target_positions = _position_map(candidate.grid_points)
    candidate_band_positions = _position_map(candidate.band_indices)
    reference_band_positions = _position_map(reference.band_indices)
    blocks: list[InteractionBlockComparison] = []
    issues: list[ReferenceIssue] = []
    seen_blocks: set[
        tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = set()

    with _open_reference(reference, verify_hash=False) as h5:
        for sample in requested:
            q0 = sample.target_grid_point
            q1 = sample.q1_grid_point
            if q0 not in reference.target_groups:
                raise IndexError(f"Reference does not contain target GRG {q0}.")
            if q0 not in candidate_target_positions:
                raise IndexError(f"Candidate does not contain target GRG {q0}.")
            if q1 < 0 or q1 >= reference.n_qpoints:
                raise IndexError(f"q1 GRG {q1} is outside the mesh.")
            if sample.target_band not in candidate_band_positions:
                raise IndexError(
                    f"Candidate does not contain target band {sample.target_band}."
                )
            if sample.target_band not in reference_band_positions:
                raise IndexError(
                    f"Reference does not contain target band {sample.target_band}."
                )

            target_position = candidate_target_positions[q0]
            candidate_routes = _route_map(
                candidate.q1_indices[target_position]
            )
            if q1 not in candidate_routes:
                raise ValueError("Candidate is missing a requested q1 route.")
            candidate_route = candidate_routes[q1]
            q2 = int(candidate.q2_indices[target_position, candidate_route])

            group = h5[reference.target_groups[q0]]
            triplets_bzg = np.asarray(
                _dataset(group, "triplets_bzg"),
                dtype=np.int64,
            )
            triplets_grg = reference.bzg2grg[triplets_bzg]
            reference_routes = _route_map(triplets_grg[:, 1])
            reference_route = reference_routes[q1]
            reference_q2 = int(triplets_grg[reference_route, 2])
            if q2 != reference_q2:
                raise ValueError(
                    "Candidate/reference q2 routing differs after GRG/BZG "
                    "conversion."
                )

            groups = (
                _matching_group(
                    candidate_frequencies=candidate_frequencies[q0],
                    reference_frequencies=reference_frequencies[q0],
                    band=sample.target_band,
                    tolerance=degeneracy,
                    cutoff_frequency_thz=reference.cutoff_frequency_thz,
                    frequency_atol=frequency_atol,
                    label=f"q0={q0}",
                ),
                _matching_group(
                    candidate_frequencies=candidate_frequencies[q1],
                    reference_frequencies=reference_frequencies[q1],
                    band=sample.band1,
                    tolerance=degeneracy,
                    cutoff_frequency_thz=reference.cutoff_frequency_thz,
                    frequency_atol=frequency_atol,
                    label=f"q1={q1}",
                ),
                _matching_group(
                    candidate_frequencies=candidate_frequencies[q2],
                    reference_frequencies=reference_frequencies[q2],
                    band=sample.band2,
                    tolerance=degeneracy,
                    cutoff_frequency_thz=reference.cutoff_frequency_thz,
                    frequency_atol=frequency_atol,
                    label=f"q2={q2}",
                ),
            )
            target_group, group1, group2 = groups
            target_tuple = tuple(int(value) for value in target_group)
            group1_tuple = tuple(int(value) for value in group1)
            group2_tuple = tuple(int(value) for value in group2)
            key = (q0, q1, target_tuple, group1_tuple, group2_tuple)
            if key in seen_blocks:
                continue
            seen_blocks.add(key)

            missing_reference = [
                band
                for band in target_tuple
                if band not in reference_band_positions
            ]
            if missing_reference:
                issues.append(
                    ReferenceIssue(
                        kind="schema_gap",
                        code="incomplete_reference_target_subspace",
                        message=(
                            f"Reference target q={q0} omits degenerate target "
                            f"bands {missing_reference}; a gauge-invariant sum "
                            "cannot be formed."
                        ),
                        hdf5_path=(
                            reference.target_groups[q0]
                            + "/interaction_strength"
                        ),
                    )
                )
                continue
            missing_candidate = [
                band
                for band in target_tuple
                if band not in candidate_band_positions
            ]
            if missing_candidate:
                raise ValueError(
                    "Candidate target selection omits members of a degenerate "
                    f"subspace: {missing_candidate}."
                )

            candidate_target_axis = [
                candidate_band_positions[band] for band in target_tuple
            ]
            reference_target_axis = [
                reference_band_positions[band] for band in target_tuple
            ]
            candidate_values = candidate.interaction_strengths[
                target_position,
                candidate_route,
            ]
            reference_values = np.asarray(
                _dataset(group, "interaction_strength")[reference_route],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(reference_values)) or np.any(
                reference_values < 0.0
            ):
                raise Phono3pyReferenceSchemaError(
                    f"{group.name}/interaction_strength contains invalid values."
                )
            candidate_sum = float(
                np.sum(
                    candidate_values[
                        np.ix_(
                            candidate_target_axis,
                            group1,
                            group2,
                        )
                    ]
                )
            )
            reference_sum = float(
                np.sum(
                    reference_values[
                        np.ix_(
                            reference_target_axis,
                            group1,
                            group2,
                        )
                    ]
                )
            )
            absolute_error = abs(candidate_sum - reference_sum)
            if reference_sum == 0.0:
                relative_error = 0.0 if absolute_error == 0.0 else math.inf
                passed = absolute_error <= atol
            else:
                relative_error = absolute_error / abs(reference_sum)
                passed = relative_error < rtol
            blocks.append(
                InteractionBlockComparison(
                    target_grid_point=q0,
                    q1_grid_point=q1,
                    q2_grid_point=q2,
                    target_bands=target_tuple,
                    bands1=group1_tuple,
                    bands2=group2_tuple,
                    reference_sum_ev2=reference_sum,
                    candidate_sum_ev2=candidate_sum,
                    absolute_error_ev2=absolute_error,
                    relative_error=relative_error,
                    passed=passed,
                )
            )

    complete = len(blocks) == len(seen_blocks) and not issues
    # Duplicate scalar requests intentionally collapse to one invariant block.
    available = bool(blocks)
    passed = available and complete and all(block.passed for block in blocks)
    return InteractionStrengthComparison(
        reference_sha256=reference.sha256,
        available=available,
        complete=complete,
        passed=passed,
        relative_tolerance=rtol,
        absolute_tolerance_ev2=atol,
        blocks=tuple(blocks),
        max_relative_error=(
            None
            if not blocks
            else max(block.relative_error for block in blocks)
        ),
        max_absolute_error_ev2=(
            None
            if not blocks
            else max(block.absolute_error_ev2 for block in blocks)
        ),
        issues=tuple(issues),
    )


@dataclass(frozen=True, slots=True)
class PhaseSpaceComparison:
    reference_sha256: str
    available: bool
    passed: bool
    tolerance: float
    max_absolute_error: float | None
    max_scaled_error: float | None
    issues: tuple[ReferenceIssue, ...]


def _phase_issues(reference: Phono3pySiliconReference) -> tuple[ReferenceIssue, ...]:
    return tuple(
        issue
        for issue in reference.schema_issues
        if issue.code.startswith("phase_space_")
    )


def compare_phase_space(
    reference: Phono3pySiliconReference,
    candidate: BulkThreePhononLinewidths,
    *,
    tolerance: float = 1.0e-10,
    verify_reference_hash: bool = True,
) -> PhaseSpaceComparison:
    """Evaluate the phase-space gate only under an identical definition."""

    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    if not isinstance(verify_reference_hash, (bool, np.bool_)):
        raise TypeError("verify_reference_hash must be boolean.")
    if verify_reference_hash:
        _ensure_unchanged(reference)
    if not isinstance(candidate, BulkThreePhononLinewidths):
        raise TypeError("candidate must be BulkThreePhononLinewidths.")
    tolerance = _finite_nonnegative(tolerance, name="tolerance")
    existing_issues = list(_phase_issues(reference))
    if existing_issues:
        return PhaseSpaceComparison(
            reference_sha256=reference.sha256,
            available=False,
            passed=False,
            tolerance=tolerance,
            max_absolute_error=None,
            max_scaled_error=None,
            issues=tuple(existing_issues),
        )
    for name, expected in _PHASE_CANDIDATE_METADATA.items():
        if name not in candidate.metadata or candidate.metadata[name] != expected:
            raise ValueError(
                f"Candidate phase-space metadata {name!r} does not match the "
                "certified full-mesh Gaussian definition."
            )
    if not np.array_equal(candidate.grid_points, reference.grid_points_grg):
        raise ValueError("Candidate/reference phase-space targets differ.")
    if not np.array_equal(candidate.band_indices, reference.band_indices):
        raise ValueError("Candidate/reference phase-space bands differ.")
    if float(candidate.metadata["sigma_thz"]) != float(
        reference.sigmas_thz[0]
    ):
        raise ValueError("Candidate/reference Gaussian sigma differs.")
    if float(candidate.metadata["cutoff_frequency_thz"]) != (
        reference.cutoff_frequency_thz
    ):
        raise ValueError("Candidate/reference cutoff frequency differs.")

    reference_values: list[Array] = []
    definition_issues: list[ReferenceIssue] = []
    with _open_reference(reference, verify_hash=False) as h5:
        for grg in reference.grid_points_grg:
            group = h5[reference.target_groups[int(grg)]]
            phase = _dataset(group, "derived_phase_space_channel_sums")
            definition = _as_text(
                _attribute(phase, "definition"),
                name=f"{phase.name}@definition",
            )
            cutoff_rule = _as_text(
                _attribute(phase, "cutoff_rule"),
                name=f"{phase.name}@cutoff_rule",
            )
            cutoff = float(_attribute(phase, "cutoff_frequency_THz"))
            integration = _dataset(group, "integration_weights")
            channel0 = _as_text(
                _attribute(integration, "channel_0"),
                name=f"{integration.name}@channel_0",
            )
            channel1 = _as_text(
                _attribute(integration, "channel_1"),
                name=f"{integration.name}@channel_1",
            )
            if (
                definition != _PHASE_DEFINITION
                or cutoff_rule != _PHASE_CUTOFF_RULE
                or cutoff != reference.cutoff_frequency_thz
                or channel0 != _PHASE_CHANNEL_0
                or channel1 != _PHASE_CHANNEL_1
            ):
                definition_issues.append(
                    ReferenceIssue(
                        kind="definition_mismatch",
                        code="phase_space_definition_mismatch",
                        message=(
                            "Reference and NEGFPy phase-space definitions are "
                            "not identical."
                        ),
                        hdf5_path=phase.name,
                    )
                )
                continue
            values = np.asarray(phase, dtype=np.float64)
            if values.shape != (2, reference.band_indices.size):
                raise Phono3pyReferenceSchemaError(
                    f"{phase.name} has an invalid shape."
                )
            reference_values.append(values)
    if definition_issues:
        return PhaseSpaceComparison(
            reference_sha256=reference.sha256,
            available=False,
            passed=False,
            tolerance=tolerance,
            max_absolute_error=None,
            max_scaled_error=None,
            issues=tuple(definition_issues),
        )
    reference_array = np.stack(reference_values, axis=0)
    candidate_array = np.stack(
        (
            candidate.decay_phase_space,
            candidate.difference_phase_space,
        ),
        axis=1,
    )
    difference = np.abs(candidate_array - reference_array)
    scale = np.maximum(1.0, np.abs(reference_array))
    max_absolute = float(np.max(difference))
    max_scaled = float(np.max(difference / scale))
    return PhaseSpaceComparison(
        reference_sha256=reference.sha256,
        available=True,
        passed=bool(np.all(difference <= tolerance * scale)),
        tolerance=tolerance,
        max_absolute_error=max_absolute,
        max_scaled_error=max_scaled,
        issues=(),
    )


@dataclass(frozen=True, slots=True)
class LinewidthSubspaceComparison:
    temperature_k: float
    target_grid_point: int
    bands: tuple[int, ...]
    reference_sum_thz_hwhm: float
    candidate_sum_thz_hwhm: float
    reference_mean_thz_hwhm: float
    candidate_mean_thz_hwhm: float
    absolute_error_thz: float
    mean_absolute_error_thz: float
    relative_error: float | None
    used_relative_gate: bool
    passed: bool


@dataclass(frozen=True, slots=True)
class LinewidthComparison:
    reference_sha256: str
    available: bool
    complete: bool
    passed: bool
    relative_threshold_thz: float
    median_relative_limit: float
    maximum_relative_limit: float
    absolute_tolerance_thz: float
    median_relative_error: float | None
    maximum_relative_error: float | None
    maximum_small_absolute_error_thz: float | None
    subspaces: tuple[LinewidthSubspaceComparison, ...]
    issues: tuple[ReferenceIssue, ...]


def compare_linewidths(
    reference: Phono3pySiliconReference,
    candidate: BulkThreePhononLinewidths,
    *,
    degenerate_tolerance_thz: float = 1.0e-4,
    frequency_atol_thz: float = 1.0e-8,
    relative_threshold_thz: float = 1.0e-6,
    median_relative_limit: float = 0.01,
    maximum_relative_limit: float = 0.05,
    absolute_tolerance_thz: float = 1.0e-6,
    verify_reference_hash: bool = True,
) -> LinewidthComparison:
    """Apply the pinned silicon cyclic-THz HWHM linewidth gates."""

    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    if not isinstance(verify_reference_hash, (bool, np.bool_)):
        raise TypeError("verify_reference_hash must be boolean.")
    if verify_reference_hash:
        _ensure_unchanged(reference)
    if not isinstance(candidate, BulkThreePhononLinewidths):
        raise TypeError("candidate must be BulkThreePhononLinewidths.")
    degeneracy = _finite_positive(
        degenerate_tolerance_thz,
        name="degenerate_tolerance_thz",
    )
    frequency_atol = _finite_nonnegative(
        frequency_atol_thz,
        name="frequency_atol_thz",
    )
    threshold = _finite_nonnegative(
        relative_threshold_thz,
        name="relative_threshold_thz",
    )
    median_limit = _finite_positive(
        median_relative_limit,
        name="median_relative_limit",
    )
    maximum_limit = _finite_positive(
        maximum_relative_limit,
        name="maximum_relative_limit",
    )
    absolute_tolerance = _finite_nonnegative(
        absolute_tolerance_thz,
        name="absolute_tolerance_thz",
    )
    if not np.array_equal(candidate.grid_points, reference.grid_points_grg):
        raise ValueError("Candidate/reference linewidth targets differ.")
    if not np.array_equal(candidate.band_indices, reference.band_indices):
        raise ValueError("Candidate/reference linewidth bands differ.")
    if candidate.temperatures.shape != reference.temperatures_k.shape or not (
        np.allclose(
            candidate.temperatures,
            reference.temperatures_k,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise ValueError("Candidate/reference temperatures differ.")
    if float(candidate.metadata["sigma_thz"]) != float(
        reference.sigmas_thz[0]
    ):
        raise ValueError("Candidate/reference Gaussian sigma differs.")
    if float(candidate.metadata["cutoff_frequency_thz"]) != (
        reference.cutoff_frequency_thz
    ):
        raise ValueError("Candidate/reference cutoff frequency differs.")
    if candidate.metadata.get("gamma_thz_semantics") != (
        "cyclic-frequency HWHM"
    ):
        raise ValueError(
            "Candidate linewidth metadata must declare cyclic-frequency HWHM."
        )
    if candidate.metadata.get("gamma_angular_fwhm_conversion") != (
        "4*pi*gamma_thz_hwhm"
    ):
        raise ValueError(
            "Candidate linewidth metadata must declare the angular-FWHM "
            "conversion."
        )
    if (
        "degenerate_tolerance_thz" not in candidate.metadata
        or float(candidate.metadata["degenerate_tolerance_thz"]) != degeneracy
    ):
        raise ValueError(
            "Candidate linewidth degeneracy tolerance differs from the "
            "comparison convention."
        )
    if candidate.gamma_thz_hwhm.shape != reference.gamma_thz_hwhm.shape:
        raise ValueError("Candidate/reference gamma arrays have different shapes.")
    if np.any(candidate.gamma_thz_hwhm < -1.0e-14):
        raise ValueError("Candidate contains a materially negative linewidth.")
    if not np.allclose(
        candidate.gamma_thz_hwhm,
        candidate.gamma_normal_thz_hwhm
        + candidate.gamma_umklapp_thz_hwhm,
        rtol=1.0e-13,
        atol=1.0e-15,
    ):
        raise ValueError(
            "Candidate total linewidth is inconsistent with its normal and "
            "Umklapp components."
        )
    if not np.allclose(
        candidate.gamma_angular_fwhm,
        4.0 * np.pi * candidate.gamma_thz_hwhm,
        rtol=1.0e-13,
        atol=1.0e-15,
    ):
        raise ValueError(
            "Candidate angular FWHM is not 4*pi times its cyclic-frequency "
            "HWHM."
        )
    interaction_metadata = candidate.interactions.metadata
    if "make_r0_average" not in interaction_metadata or bool(
        interaction_metadata["make_r0_average"]
    ) != reference.make_r0_average:
        raise ValueError(
            "Candidate/reference make_r0_average settings differ."
        )

    candidate_frequencies = candidate.interactions.phonons.frequencies
    reference_frequencies = reference.frequencies_in_grg_order()
    selected = {int(value) for value in reference.band_indices}
    band_positions = _position_map(reference.band_indices)
    comparisons: list[LinewidthSubspaceComparison] = []
    issues: list[ReferenceIssue] = []

    for target_position, target in enumerate(reference.grid_points_grg):
        target = int(target)
        if not _frequencies_match_with_cutoff(
            candidate_frequencies[target],
            reference_frequencies[target],
            cutoff_frequency_thz=reference.cutoff_frequency_thz,
            frequency_atol_thz=frequency_atol,
        ):
            raise ValueError(
                f"Candidate/reference frequencies differ at target GRG {target}."
            )
        candidate_groups = _degenerate_groups(
            candidate_frequencies[target],
            degeneracy,
        )
        reference_groups = _degenerate_groups(
            reference_frequencies[target],
            degeneracy,
        )
        if len(candidate_groups) != len(reference_groups) or any(
            not np.array_equal(left, right)
            for left, right in zip(
                candidate_groups,
                reference_groups,
                strict=True,
            )
        ):
            raise ValueError(
                f"Candidate/reference degeneracy partitions differ at {target}."
            )
        for group in candidate_groups:
            group_tuple = tuple(int(value) for value in group)
            selected_members = [band for band in group_tuple if band in selected]
            if not selected_members:
                continue
            if len(selected_members) != len(group_tuple):
                issues.append(
                    ReferenceIssue(
                        kind="schema_gap",
                        code="incomplete_linewidth_target_subspace",
                        message=(
                            f"Reference target q={target} contains only part of "
                            f"degenerate bands {group_tuple}; an invariant "
                            "linewidth trace cannot be formed."
                        ),
                        hdf5_path="/gamma/values",
                    )
                )
                continue
            positions = [band_positions[band] for band in group_tuple]
            for temperature_position, temperature in enumerate(
                reference.temperatures_k
            ):
                reference_sum = float(
                    np.sum(
                        reference.gamma_thz_hwhm[
                            temperature_position,
                            target_position,
                            positions,
                        ]
                    )
                )
                candidate_sum = float(
                    np.sum(
                        candidate.gamma_thz_hwhm[
                            temperature_position,
                            target_position,
                            positions,
                        ]
                    )
                )
                dimension = len(group_tuple)
                reference_mean = reference_sum / dimension
                candidate_mean = candidate_sum / dimension
                absolute_error = abs(candidate_sum - reference_sum)
                mean_absolute_error = abs(candidate_mean - reference_mean)
                use_relative = abs(reference_mean) > threshold
                relative_error = (
                    absolute_error / abs(reference_sum)
                    if use_relative
                    else None
                )
                # Aggregate relative gates are evaluated below. The local
                # relative criterion uses the maximum limit so each row has a
                # meaningful pass flag.
                local_passed = (
                    relative_error < maximum_limit
                    if use_relative
                    else mean_absolute_error <= absolute_tolerance
                )
                comparisons.append(
                    LinewidthSubspaceComparison(
                        temperature_k=float(temperature),
                        target_grid_point=target,
                        bands=group_tuple,
                        reference_sum_thz_hwhm=reference_sum,
                        candidate_sum_thz_hwhm=candidate_sum,
                        reference_mean_thz_hwhm=reference_mean,
                        candidate_mean_thz_hwhm=candidate_mean,
                        absolute_error_thz=absolute_error,
                        mean_absolute_error_thz=mean_absolute_error,
                        relative_error=relative_error,
                        used_relative_gate=use_relative,
                        passed=local_passed,
                    )
                )

    relative_errors = np.asarray(
        [
            item.relative_error
            for item in comparisons
            if item.relative_error is not None
        ],
        dtype=np.float64,
    )
    small_errors = np.asarray(
        [
            item.mean_absolute_error_thz
            for item in comparisons
            if item.relative_error is None
        ],
        dtype=np.float64,
    )
    median_relative = (
        None if relative_errors.size == 0 else float(np.median(relative_errors))
    )
    maximum_relative = (
        None if relative_errors.size == 0 else float(np.max(relative_errors))
    )
    maximum_small_absolute = (
        None if small_errors.size == 0 else float(np.max(small_errors))
    )
    relative_passed = (
        True
        if relative_errors.size == 0
        else (
            median_relative is not None
            and maximum_relative is not None
            and median_relative < median_limit
            and maximum_relative < maximum_limit
        )
    )
    absolute_passed = (
        True
        if small_errors.size == 0
        else bool(np.all(small_errors <= absolute_tolerance))
    )
    complete = not issues and bool(comparisons)
    passed = complete and relative_passed and absolute_passed
    return LinewidthComparison(
        reference_sha256=reference.sha256,
        available=bool(comparisons),
        complete=complete,
        passed=passed,
        relative_threshold_thz=threshold,
        median_relative_limit=median_limit,
        maximum_relative_limit=maximum_limit,
        absolute_tolerance_thz=absolute_tolerance,
        median_relative_error=median_relative,
        maximum_relative_error=maximum_relative,
        maximum_small_absolute_error_thz=maximum_small_absolute,
        subspaces=tuple(comparisons),
        issues=tuple(issues),
    )


@dataclass(frozen=True, slots=True)
class SiliconReferenceComparison:
    """Combined silicon-oracle release decision."""

    fc2: FC2RandomQComparison
    interactions: InteractionStrengthComparison
    phase_space: PhaseSpaceComparison
    linewidths: LinewidthComparison

    @property
    def passed(self) -> bool:
        return bool(
            self.fc2.passed
            and self.interactions.passed
            and self.phase_space.passed
            and self.linewidths.passed
        )


def compare_silicon_reference(
    reference: Phono3pySiliconReference,
    *,
    fc2_spectrum: BulkPhononSpectrum | None,
    interactions: BulkThreePhononInteractions,
    linewidths: BulkThreePhononLinewidths,
    interaction_samples: Sequence[
        InteractionStrengthSample | Sequence[int]
    ],
) -> SiliconReferenceComparison:
    """Run all strict mandatory silicon-oracle gates."""

    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    _ensure_unchanged(reference)
    return SiliconReferenceComparison(
        fc2=compare_fc2_random_q(
            reference,
            fc2_spectrum,
            verify_reference_hash=False,
        ),
        interactions=compare_interaction_strengths(
            reference,
            interactions,
            interaction_samples,
            verify_reference_hash=False,
        ),
        phase_space=compare_phase_space(
            reference,
            linewidths,
            verify_reference_hash=False,
        ),
        linewidths=compare_linewidths(
            reference,
            linewidths,
            verify_reference_hash=False,
        ),
    )


__all__ = [
    "FC2RandomQComparison",
    "InteractionBlockComparison",
    "InteractionStrengthComparison",
    "InteractionStrengthSample",
    "LinewidthComparison",
    "LinewidthSubspaceComparison",
    "PhaseSpaceComparison",
    "Phono3pyReferenceSchemaError",
    "Phono3pySiliconReference",
    "ReferenceIssue",
    "SiliconReferenceComparison",
    "compare_fc2_random_q",
    "compare_interaction_strengths",
    "compare_linewidths",
    "compare_phase_space",
    "compare_silicon_reference",
    "read_phono3py_silicon_reference",
]
