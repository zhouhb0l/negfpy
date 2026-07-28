"""Deterministic harmonic bulk oracle built from phono3py-style FC2 data.

This module is deliberately independent of phonopy and phono3py runtime
objects.  It implements the same shortest-vector Fourier reconstruction used
by phonopy's non-NAC dynamical matrix, while retaining the un-Hermitianized
matrix as an auditable quantity.  Hermitianization is always an explicit
choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .crystal import CrystalMapping, validate_crystal_mapping
from .fc2 import (
    ANGSTROM_M,
    ATOMIC_MASS_UNIT_KG,
    ELECTRON_VOLT_J,
    FC2Data,
    fc2_force_constant_to_si_factor,
    mass_to_si_factor,
    validate_fc2,
)


Array = np.ndarray

CODATA_FREQUENCY_FACTOR_TO_THZ = (
    np.sqrt(ELECTRON_VOLT_J / ATOMIC_MASS_UNIT_KG)
    / ANGSTROM_M
    / (2.0 * np.pi)
    / 1.0e12
)
PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ = 15.633302300230191
# The bulk oracle is explicitly pinned to phono3py 4.4.0. A modern-CODATA
# factor remains available by name, but is not a compatibility default.
DEFAULT_FREQUENCY_FACTOR_TO_THZ = PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ


def _readonly(values: Any, *, dtype: Any | None = None) -> Array:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _validate_qpoint(qpoint: Any) -> Array:
    raw = np.asarray(qpoint)
    if raw.dtype.kind == "c":
        raise TypeError("qpoint must be real-valued.")
    try:
        q = np.array(qpoint, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("qpoint must be a real numerical vector.") from exc
    if q.shape != (3,):
        raise ValueError("qpoint must have shape (3,).")
    if not np.all(np.isfinite(q)):
        raise ValueError("qpoint must contain only finite values.")
    return q


def _assert_fc2_mapping_compatibility(
    fc2: FC2Data,
    mapping: CrystalMapping,
) -> None:
    report = validate_fc2(fc2)
    required = (
        report.dimension_valid,
        report.finite_valid,
        report.masses_valid,
        report.units_valid,
        report.mapping_valid,
    )
    if not all(required):
        raise ValueError(
            "FC2 schema must pass dimensions, finiteness, masses, units, "
            "and mapping checks before dynamical-matrix reconstruction: "
            + " ".join(report.errors)
        )
    mapping_report = validate_crystal_mapping(mapping)
    if not mapping_report.passed:
        raise ValueError(
            "Crystal mapping must pass validation before FC2 reconstruction: "
            + " ".join(mapping_report.errors)
        )
    if mapping.purpose != "fc2":
        raise ValueError("The crystal mapping purpose must be 'fc2'.")
    if not np.isclose(
        fc2_force_constant_to_si_factor(fc2.units),
        ELECTRON_VOLT_J / ANGSTROM_M**2,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "The phono3py bulk oracle requires FC2 values in "
            "eV/angstrom^2; convert SI input explicitly before use."
        )
    if not np.isclose(
        mass_to_si_factor(fc2.mass_units),
        ATOMIC_MASS_UNIT_KG,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(
            "The phono3py bulk oracle requires masses in amu; convert mass "
            "input explicitly before use."
        )
    if bool(mapping.provenance.get("site_mixture_present", False)):
        raise NotImplementedError(
            "Phonopy site-mixture normalization is not represented by "
            "CrystalMapping and is excluded from the bulk oracle."
        )

    n_supercell = mapping.n_supercell_atoms
    if fc2.n_supercell_atoms != n_supercell:
        raise ValueError(
            "FC2 and crystal mapping have different supercell atom counts."
        )
    if not np.array_equal(
        np.asarray(fc2.masses, dtype=np.float64),
        mapping.supercell.masses,
    ):
        raise ValueError(
            "FC2 masses must exactly match the ordered mapping supercell masses."
        )
    if fc2.p2s_map is not None and not np.array_equal(
        fc2.p2s_map,
        mapping.p2s,
    ):
        raise ValueError("FC2 p2s_map conflicts with the crystal mapping.")
    if fc2.s2p_map is not None and not np.array_equal(
        fc2.s2p_map,
        mapping.s2p,
    ):
        raise ValueError("FC2 s2p_map conflicts with the crystal mapping.")
    if fc2.compact and fc2.values.shape[0] != mapping.n_primitive_atoms:
        raise ValueError(
            "Compact FC2 first-axis length must equal the primitive atom count."
        )


def dynamical_matrix_from_fc2(
    fc2: FC2Data,
    mapping: CrystalMapping,
    qpoint: Any,
    *,
    hermitianize: bool = False,
) -> Array:
    r"""Reconstruct one mass-weighted dynamical matrix.

    ``qpoint`` is expressed in reciprocal primitive-lattice coordinates.  For
    each supercell image, tied shortest vectors are averaged with equal
    multiplicity weight:

    .. math::

       D_{i\alpha,j\beta}(q)=
       (M_iM_j)^{-1/2}\sum_{s\mapsto j}\Phi_{i\alpha,s\beta}
       m_{si}^{-1}\sum_{\ell=1}^{m_{si}}
       e^{2\pi i q\cdot r_{si\ell}}.

    The returned array is a detached complex128 value.  No correction,
    symmetrization, acoustic sum rule, or stability repair is performed.
    """

    if not isinstance(fc2, FC2Data):
        raise TypeError("fc2 must be an FC2Data instance.")
    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping instance.")
    if not isinstance(hermitianize, (bool, np.bool_)):
        raise TypeError("hermitianize must be a boolean.")
    q = _validate_qpoint(qpoint)
    _assert_fc2_mapping_compatibility(fc2, mapping)

    values = np.asarray(fc2.values)
    n_primitive = mapping.n_primitive_atoms
    matrix = np.zeros(
        (3 * n_primitive, 3 * n_primitive),
        dtype=np.complex128,
        order="C",
    )
    masses = mapping.primitive.masses

    for primitive_i, representative_i in enumerate(mapping.p2s):
        first_axis = primitive_i if fc2.compact else int(representative_i)
        for primitive_j, representative_j in enumerate(mapping.p2s):
            block = np.zeros((3, 3), dtype=np.complex128)
            mass_factor = np.sqrt(masses[primitive_i] * masses[primitive_j])
            for supercell_j in range(mapping.n_supercell_atoms):
                if mapping.s2p[supercell_j] != representative_j:
                    continue
                count, offset = mapping.multiplicity[
                    supercell_j,
                    primitive_i,
                ]
                vectors = mapping.shortest_vectors[
                    int(offset) : int(offset + count)
                ]
                phase = np.exp(2j * np.pi * (vectors @ q)).sum() / int(count)
                block += values[first_axis, supercell_j] * phase / mass_factor
            row = slice(3 * primitive_i, 3 * primitive_i + 3)
            column = slice(3 * primitive_j, 3 * primitive_j + 3)
            matrix[row, column] = block

    if hermitianize:
        matrix = (matrix + matrix.conj().T) / 2.0
    return np.array(matrix, dtype=np.complex128, order="C", copy=True)


def _canonicalize_eigenvectors(eigenvectors: Array) -> Array:
    """Fix the arbitrary U(1) phase of each nonzero eigenvector.

    Degenerate-subspace rotations are intentionally untouched; comparisons
    involving degeneracies must use subspace-invariant sums.
    """

    result = np.array(eigenvectors, dtype=np.complex128, order="C", copy=True)
    for band in range(result.shape[1]):
        column = result[:, band]
        pivot = int(np.argmax(np.abs(column)))
        value = column[pivot]
        if abs(value) == 0.0:
            continue
        result[:, band] *= np.exp(-1j * np.angle(value))
        if result[pivot, band].real < 0.0:
            result[:, band] *= -1.0
        result[pivot, band] = complex(abs(result[pivot, band]), 0.0)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class BulkPhononSpectrum:
    """Immutable phonon eigenproblem results on an ordered q-point list."""

    qpoints: Array
    dynamical_matrices: Array
    frequencies: Array
    eigenvectors: Array
    frequency_factor_to_thz: float
    hermitianized: bool

    def __post_init__(self) -> None:
        qpoints = _readonly(self.qpoints, dtype=np.float64)
        dynamical = _readonly(self.dynamical_matrices, dtype=np.complex128)
        frequencies = _readonly(self.frequencies, dtype=np.float64)
        eigenvectors = _readonly(self.eigenvectors, dtype=np.complex128)

        if qpoints.ndim != 2 or qpoints.shape[1:] != (3,):
            raise ValueError("qpoints must have shape (n_qpoints, 3).")
        if not np.all(np.isfinite(qpoints)):
            raise ValueError("qpoints must be finite.")
        if dynamical.ndim != 3 or dynamical.shape[0] != qpoints.shape[0]:
            raise ValueError(
                "dynamical_matrices must have shape (n_qpoints, n_band, n_band)."
            )
        if dynamical.shape[1] != dynamical.shape[2]:
            raise ValueError("Every dynamical matrix must be square.")
        n_qpoints, n_band, _ = dynamical.shape
        if frequencies.shape != (n_qpoints, n_band):
            raise ValueError(
                "frequencies must have shape (n_qpoints, n_band)."
            )
        if eigenvectors.shape != (n_qpoints, n_band, n_band):
            raise ValueError(
                "eigenvectors must have shape (n_qpoints, n_band, n_band)."
            )
        if not (
            np.all(np.isfinite(dynamical))
            and np.all(np.isfinite(frequencies))
            and np.all(np.isfinite(eigenvectors))
        ):
            raise ValueError("Phonon spectrum arrays must be finite.")
        factor = float(self.frequency_factor_to_thz)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError(
                "frequency_factor_to_thz must be a finite positive scalar."
            )
        if not isinstance(self.hermitianized, (bool, np.bool_)):
            raise TypeError("hermitianized must be a boolean.")

        object.__setattr__(self, "qpoints", qpoints)
        object.__setattr__(self, "dynamical_matrices", dynamical)
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "eigenvectors", eigenvectors)
        object.__setattr__(self, "frequency_factor_to_thz", factor)
        object.__setattr__(self, "hermitianized", bool(self.hermitianized))

    @property
    def n_qpoints(self) -> int:
        return int(self.qpoints.shape[0])

    @property
    def n_bands(self) -> int:
        return int(self.frequencies.shape[1])


def compute_bulk_phonons(
    fc2: FC2Data,
    mapping: CrystalMapping,
    qpoints: Any,
    *,
    frequency_factor_to_thz: float = DEFAULT_FREQUENCY_FACTOR_TO_THZ,
    hermitianize: bool = True,
) -> BulkPhononSpectrum:
    """Solve the ordered non-NAC harmonic eigenproblems at ``qpoints``."""

    raw_qpoints = np.asarray(qpoints)
    if raw_qpoints.dtype.kind == "c":
        raise TypeError("qpoints must be real-valued.")
    try:
        qpoint_array = np.array(
            qpoints,
            dtype=np.float64,
            order="C",
            copy=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("qpoints must be a real numerical array.") from exc
    if qpoint_array.ndim != 2 or qpoint_array.shape[1:] != (3,):
        raise ValueError("qpoints must have shape (n_qpoints, 3).")
    if qpoint_array.shape[0] < 1 or not np.all(np.isfinite(qpoint_array)):
        raise ValueError("qpoints must be nonempty and finite.")
    if isinstance(frequency_factor_to_thz, (bool, np.bool_)):
        raise TypeError(
            "frequency_factor_to_thz must be a finite positive scalar."
        )
    factor = float(frequency_factor_to_thz)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            "frequency_factor_to_thz must be a finite positive scalar."
        )
    if not isinstance(hermitianize, (bool, np.bool_)):
        raise TypeError("hermitianize must be a boolean.")

    n_band = 3 * mapping.n_primitive_atoms
    matrices = np.empty(
        (qpoint_array.shape[0], n_band, n_band),
        dtype=np.complex128,
    )
    frequencies = np.empty(
        (qpoint_array.shape[0], n_band),
        dtype=np.float64,
    )
    eigenvectors = np.empty_like(matrices)

    for index, qpoint in enumerate(qpoint_array):
        matrix = dynamical_matrix_from_fc2(
            fc2,
            mapping,
            qpoint,
            hermitianize=bool(hermitianize),
        )
        if not hermitianize and not np.allclose(
            matrix,
            matrix.conj().T,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "A non-Hermitian raw dynamical matrix cannot be diagonalized "
                "without explicit hermitianize=True."
            )
        eigenvalues, vectors = np.linalg.eigh(matrix)
        matrices[index] = matrix
        frequencies[index] = (
            np.sqrt(np.abs(eigenvalues))
            * np.where(eigenvalues < 0.0, -1.0, 1.0)
            * factor
        )
        eigenvectors[index] = _canonicalize_eigenvectors(vectors)

    return BulkPhononSpectrum(
        qpoints=qpoint_array,
        dynamical_matrices=matrices,
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        frequency_factor_to_thz=factor,
        hermitianized=bool(hermitianize),
    )


__all__ = [
    "BulkPhononSpectrum",
    "CODATA_FREQUENCY_FACTOR_TO_THZ",
    "DEFAULT_FREQUENCY_FACTOR_TO_THZ",
    "PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ",
    "compute_bulk_phonons",
    "dynamical_matrix_from_fc2",
]
