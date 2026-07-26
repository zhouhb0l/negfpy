r"""Independent source-reference checks for harmonic dynamical matrices.

This module intentionally does not call :func:`audit_ifc_data`.  It provides
the release preflight with a second route from an :class:`IFCData` object to
``D(q)`` and compares that result with matrices generated independently by a
source reader, source program, or analytic fixture.

The periodic gauge used by ``IFCData`` is

.. math::

   D^P(q) = M^{-1/2}\sum_R \Phi(R)e^{i q\cdot R}M^{-1/2}.

For source matrices that include basis-atom phases, this module uses the
explicit unitary

.. math::

   U_{i\alpha,j\beta}(q)=\delta_{ij}\delta_{\alpha\beta}
   e^{i q\cdot\tau_i},\qquad D^A(q)=U^\dagger D^P(q)U.

Here ``q`` contains dimensionless reciprocal-lattice phase coordinates in
radians and ``tau`` contains fractional basis positions.  No IFC block or
dynamical matrix is symmetrized or Hermitized by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from negfpy.modeling.schema import IFCData


Array = np.ndarray
Gauge = Literal["periodic", "atom_phase"]
QPoint = tuple[float, float, float]
Translation = tuple[int, int, int]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GAUGES = frozenset(("periodic", "atom_phase"))


def _readonly_array(value: Any, *, dtype: np.dtype[Any]) -> Array:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _normalize_json(value: Any, *, path: str = "provenance") -> Any:
    """Return a deterministic, JSON-compatible representation."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} cannot contain non-finite numbers.")
        return value
    if isinstance(value, np.ndarray):
        return _normalize_json(value.tolist(), path=path)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise TypeError(f"{path} keys must be non-empty strings.")
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            normalized[key] = _normalize_json(value[key], path=f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported value of type {type(value).__name__}."
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256.")
    return value.lower()


def _validate_gauge(value: Any) -> Gauge:
    if not isinstance(value, str) or value not in _GAUGES:
        choices = ", ".join(sorted(_GAUGES))
        raise ValueError(f"gauge must be one of: {choices}.")
    return value  # type: ignore[return-value]


def _validate_dof(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("dof_per_atom must be a positive integer.")
    dof = int(value)
    if dof <= 0:
        raise ValueError("dof_per_atom must be a positive integer.")
    return dof


def _canonical_qpoints(
    qpoints: Any,
    *,
    matrices: Array | None = None,
) -> tuple[Array, Array | None]:
    raw_qpoints = np.asarray(qpoints)
    if np.iscomplexobj(raw_qpoints):
        raise TypeError("qpoints must be real-valued.")
    try:
        q_array = np.asarray(raw_qpoints, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("qpoints must be a finite real array with shape (nq, 3).") from exc
    if q_array.ndim != 2 or q_array.shape[1:] != (3,) or q_array.shape[0] == 0:
        raise ValueError("qpoints must have shape (nq, 3) with nq > 0.")
    if not np.all(np.isfinite(q_array)):
        raise ValueError("qpoints must contain only finite values.")

    # Normalize signed zero so exact duplicate detection is platform stable.
    q_array = np.array(q_array, dtype=np.float64, order="C", copy=True)
    q_array[q_array == 0.0] = 0.0
    keys = [tuple(float(component) for component in row) for row in q_array]
    if len(set(keys)) != len(keys):
        raise ValueError("qpoints contains an exact duplicate q point.")

    order = np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.intp)
    ordered_q = _readonly_array(q_array[order], dtype=np.dtype(np.float64))
    if matrices is None:
        return ordered_q, None
    if matrices.shape[0] != q_array.shape[0]:
        raise ValueError("qpoints and matrices must contain the same number of samples.")
    return ordered_q, _readonly_array(matrices[order], dtype=np.dtype(np.complex128))


def _basis_positions(
    value: Any,
    *,
    n_atoms: int,
    required: bool,
) -> Array | None:
    if value is None:
        if required:
            raise ValueError("atom_phase gauge requires explicit basis_positions.")
        return None
    raw_positions = np.asarray(value)
    if np.iscomplexobj(raw_positions):
        raise TypeError("basis_positions must be real-valued.")
    try:
        positions = np.asarray(raw_positions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "basis_positions must be a finite real array with shape (n_atoms, 3)."
        ) from exc
    if positions.shape != (n_atoms, 3):
        raise ValueError(
            "basis_positions must have shape "
            f"({n_atoms}, 3), not {positions.shape}."
        )
    if not np.all(np.isfinite(positions)):
        raise ValueError("basis_positions must contain only finite values.")
    return _readonly_array(positions, dtype=np.dtype(np.float64))


def basis_phase_unitary(
    qpoint: Sequence[float],
    basis_positions: Any,
    dof_per_atom: int,
) -> Array:
    """Return the explicit periodic-to-atom-phase basis unitary.

    The returned matrix ``U`` satisfies ``D_atom = U.conj().T @ D_periodic @ U``.
    ``qpoint`` is in radians and ``basis_positions`` are fractional.
    """

    dof = _validate_dof(dof_per_atom)
    raw_q = np.asarray(qpoint)
    if np.iscomplexobj(raw_q):
        raise TypeError("qpoint must be real-valued.")
    try:
        q = np.asarray(raw_q, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("qpoint must contain exactly three finite real values.") from exc
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("qpoint must contain exactly three finite real values.")
    raw_positions = np.asarray(basis_positions)
    if np.iscomplexobj(raw_positions):
        raise TypeError("basis_positions must be real-valued.")
    try:
        positions = np.asarray(raw_positions, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("basis_positions must have finite shape (n_atoms, 3).") from exc
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or positions.shape[0] == 0
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError("basis_positions must have finite shape (n_atoms, 3).")

    phases = np.exp(1j * (positions @ q))
    diagonal = np.repeat(phases, dof)
    unitary = np.diag(diagonal).astype(np.complex128, copy=False)
    unitary.setflags(write=False)
    return unitary


def _reference_content_hash(
    *,
    qpoints: Array,
    matrices: Array,
    source: str,
    source_sha256: str,
    gauge: Gauge,
    dof_per_atom: int,
    basis_positions: Array | None,
    provenance_json: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    header = {
        "schema": "negfpy.ifc_dynamical_reference.v1",
        "source": source,
        "source_sha256": source_sha256,
        "gauge": gauge,
        "dof_per_atom": dof_per_atom,
        "provenance": provenance_json,
        "qpoint_shape": list(qpoints.shape),
        "matrix_shape": list(matrices.shape),
        "has_basis_positions": basis_positions is not None,
    }
    digest.update(
        json.dumps(
            header,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(np.asarray(qpoints, dtype="<f8", order="C").tobytes(order="C"))
    digest.update(np.asarray(matrices, dtype="<c16", order="C").tobytes(order="C"))
    if basis_positions is not None:
        digest.update(
            np.asarray(basis_positions, dtype="<f8", order="C").tobytes(order="C")
        )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class IFCDynamicalReference:
    """Immutable source-generated ``D(q)`` reference in a declared gauge."""

    qpoints: Array
    matrices: Array
    source: str
    source_sha256: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    gauge: Gauge = "periodic"
    dof_per_atom: int = 3
    basis_positions: Array | None = None
    content_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string.")
        source = self.source.strip()
        source_sha256 = _validate_sha256(self.source_sha256, name="source_sha256")
        gauge = _validate_gauge(self.gauge)
        dof = _validate_dof(self.dof_per_atom)

        try:
            matrices = np.asarray(self.matrices, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "matrices must be a finite complex array with shape (nq, ndof, ndof)."
            ) from exc
        if (
            matrices.ndim != 3
            or matrices.shape[0] == 0
            or matrices.shape[1] == 0
            or matrices.shape[1] != matrices.shape[2]
        ):
            raise ValueError("matrices must have shape (nq, ndof, ndof).")
        if not np.all(np.isfinite(matrices.real)) or not np.all(np.isfinite(matrices.imag)):
            raise ValueError("matrices must contain only finite values.")
        if matrices.shape[1] % dof:
            raise ValueError("matrix dimension must be divisible by dof_per_atom.")

        qpoints, ordered_matrices = _canonical_qpoints(qpoints=self.qpoints, matrices=matrices)
        assert ordered_matrices is not None
        if ordered_matrices.shape[0] != qpoints.shape[0]:
            raise ValueError("qpoints and matrices must contain the same number of samples.")
        n_atoms = ordered_matrices.shape[1] // dof
        positions = _basis_positions(
            self.basis_positions,
            n_atoms=n_atoms,
            required=(gauge == "atom_phase"),
        )

        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping.")
        provenance_json = _normalize_json(self.provenance)
        assert isinstance(provenance_json, dict)
        frozen_provenance = _freeze_json(provenance_json)
        content_sha256 = _reference_content_hash(
            qpoints=qpoints,
            matrices=ordered_matrices,
            source=source,
            source_sha256=source_sha256,
            gauge=gauge,
            dof_per_atom=dof,
            basis_positions=positions,
            provenance_json=provenance_json,
        )

        object.__setattr__(self, "qpoints", qpoints)
        object.__setattr__(self, "matrices", ordered_matrices)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "provenance", frozen_provenance)
        object.__setattr__(self, "gauge", gauge)
        object.__setattr__(self, "dof_per_atom", dof)
        object.__setattr__(self, "basis_positions", positions)
        object.__setattr__(self, "content_sha256", content_sha256)


@dataclass(frozen=True, slots=True)
class IFCDynamicalComparison:
    """Immutable outcome of an IFC-to-source dynamical-matrix comparison."""

    passed: bool
    tolerance: float
    allowed_max_abs_error: float
    max_abs_error: float
    max_relative_error: float
    reference_scale: float
    per_q_max_abs_error: tuple[float, ...]
    per_q_relative_error: tuple[float, ...]
    qpoints: tuple[QPoint, ...]
    worst_qpoint: QPoint
    q_units: Literal["radian"]
    gauge: Gauge
    source: str
    provenance: Mapping[str, Any]
    source_sha256: str
    reference_sha256: str
    ifc_sha256: str
    input_term_count: int
    merged_translation_count: int
    duplicate_translations: tuple[Translation, ...]


def _validated_ifc_terms(
    ifc: IFCData,
) -> tuple[
    Array,
    int,
    list[tuple[Translation, Array]],
    dict[Translation, Array],
    tuple[Translation, ...],
]:
    if not isinstance(ifc, IFCData):
        raise TypeError("ifc must be an IFCData instance.")
    raw_masses = np.asarray(ifc.masses)
    if raw_masses.dtype.kind == "b":
        raise TypeError("IFCData.masses must be real numeric values, not booleans.")
    if np.iscomplexobj(raw_masses):
        raise ValueError("IFCData.masses must be real-valued.")
    try:
        masses = np.asarray(raw_masses, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("IFCData.masses must be a finite positive 1D array.") from exc
    if (
        masses.ndim != 1
        or masses.size == 0
        or not np.all(np.isfinite(masses))
        or np.any(masses <= 0.0)
    ):
        raise ValueError("IFCData.masses must be a finite positive 1D array.")
    dof = _validate_dof(ifc.dof_per_atom)
    ndof = int(masses.size) * dof
    if not isinstance(ifc.terms, tuple) or not ifc.terms:
        raise ValueError("IFCData.terms must be a non-empty tuple.")

    raw_terms: list[tuple[Translation, Array]] = []
    merged: dict[Translation, Array] = {}
    counts: dict[Translation, int] = {}
    for index, term in enumerate(ifc.terms):
        components = (term.dx, term.dy, term.dz)
        if any(
            isinstance(component, (bool, np.bool_))
            or not isinstance(component, (int, np.integer))
            for component in components
        ):
            raise TypeError(f"IFCData.terms[{index}] translation must contain integers.")
        translation = tuple(int(component) for component in components)
        try:
            block = np.asarray(term.block, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"IFCData.terms[{index}].block must be numeric.") from exc
        if block.shape != (ndof, ndof):
            raise ValueError(
                f"IFCData.terms[{index}].block has shape {block.shape}; "
                f"expected {(ndof, ndof)}."
            )
        if not np.all(np.isfinite(block.real)) or not np.all(np.isfinite(block.imag)):
            raise ValueError(f"IFCData.terms[{index}].block contains non-finite values.")
        block_copy = np.array(block, dtype=np.complex128, order="C", copy=True)
        raw_terms.append((translation, block_copy))
        counts[translation] = counts.get(translation, 0) + 1
        if translation not in merged:
            merged[translation] = np.zeros((ndof, ndof), dtype=np.complex128)
        # This is the single and only merge of repeated translations.
        merged[translation] += block_copy

    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    return np.array(masses, copy=True), dof, raw_terms, merged, duplicates


def _ifc_hash(
    ifc: IFCData,
    *,
    masses: Array,
    dof: int,
    raw_terms: list[tuple[Translation, Array]],
) -> str:
    digest = hashlib.sha256()
    header = {
        "schema": "negfpy.ifc_dynamical_input.v1",
        "dof_per_atom": dof,
        "term_count": len(raw_terms),
        "units": str(ifc.units),
        "index_convention": str(ifc.index_convention),
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    digest.update(np.asarray(masses, dtype="<f8", order="C").tobytes(order="C"))
    for translation, block in raw_terms:
        digest.update(
            np.asarray(translation, dtype="<i8", order="C").tobytes(order="C")
        )
        digest.update(np.asarray(block, dtype="<c16", order="C").tobytes(order="C"))
    return digest.hexdigest()


def _reconstruct_periodic(
    *,
    masses: Array,
    dof: int,
    merged: Mapping[Translation, Array],
    qpoints: Array,
) -> Array:
    ndof = int(masses.size) * dof
    inverse_sqrt_mass = 1.0 / np.sqrt(np.repeat(masses, dof))
    output = np.empty((qpoints.shape[0], ndof, ndof), dtype=np.complex128)
    ordered_terms = tuple((key, merged[key]) for key in sorted(merged))
    for q_index, qpoint in enumerate(qpoints):
        phi_q = np.zeros((ndof, ndof), dtype=np.complex128)
        for translation, block in ordered_terms:
            phase = np.exp(1j * np.dot(qpoint, translation))
            phi_q += block * phase
        # Mass normalization is performed once, after the translation sum.
        output[q_index] = (
            inverse_sqrt_mass[:, None]
            * phi_q
            * inverse_sqrt_mass[None, :]
        )
    return output


def compare_ifc_dynamical_matrices(
    ifc: IFCData,
    reference: IFCDynamicalReference,
    *,
    tolerance: float = 1.0e-10,
    basis_position_atol: float = 1.0e-12,
) -> IFCDynamicalComparison:
    """Compare a production ``IFCData`` reconstruction with source matrices.

    At every q point, the pass condition is ``max_abs_error(q) <=
    tolerance * max(1, max(abs(reference(q))))``.

    Duplicate IFC translations are summed exactly once.  Duplicate reference
    q points are rejected while constructing :class:`IFCDynamicalReference`.
    """

    if not isinstance(reference, IFCDynamicalReference):
        raise TypeError("reference must be an IFCDynamicalReference.")
    if isinstance(tolerance, (bool, np.bool_)):
        raise TypeError("tolerance must be a finite non-negative real number.")
    tolerance = float(tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be a finite non-negative real number.")
    if isinstance(basis_position_atol, (bool, np.bool_)):
        raise TypeError("basis_position_atol must be a finite non-negative real number.")
    basis_position_atol = float(basis_position_atol)
    if not np.isfinite(basis_position_atol) or basis_position_atol < 0.0:
        raise ValueError(
            "basis_position_atol must be a finite non-negative real number."
        )

    masses, dof, raw_terms, merged, duplicates = _validated_ifc_terms(ifc)
    ndof = int(masses.size) * dof
    if reference.dof_per_atom != dof:
        raise ValueError(
            "Reference dof_per_atom does not match IFCData "
            f"({reference.dof_per_atom} != {dof})."
        )
    if reference.matrices.shape[1:] != (ndof, ndof):
        raise ValueError(
            "Reference matrix dimension does not match IFCData "
            f"({reference.matrices.shape[1:]} != {(ndof, ndof)})."
        )

    calculated = _reconstruct_periodic(
        masses=masses,
        dof=dof,
        merged=merged,
        qpoints=reference.qpoints,
    )
    if reference.gauge == "atom_phase":
        assert reference.basis_positions is not None
        if ifc.atom_positions is None:
            raise ValueError(
                "atom_phase comparison requires IFCData.atom_positions for "
                "an explicit basis-map check."
            )
        raw_ifc_positions = np.asarray(ifc.atom_positions)
        if np.iscomplexobj(raw_ifc_positions):
            raise TypeError("IFCData.atom_positions must be real-valued.")
        try:
            ifc_positions = np.asarray(raw_ifc_positions, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("IFCData.atom_positions must be a finite real array.") from exc
        if ifc_positions.shape != reference.basis_positions.shape:
            raise ValueError(
                "IFCData.atom_positions shape does not match reference "
                "basis_positions."
            )
        if (
            not np.all(np.isfinite(ifc_positions))
            or not np.allclose(
                ifc_positions,
                reference.basis_positions,
                rtol=0.0,
                atol=basis_position_atol,
            )
        ):
            raise ValueError(
                "IFCData.atom_positions do not match the explicit reference "
                "basis_positions."
            )
        for q_index, qpoint in enumerate(reference.qpoints):
            unitary = basis_phase_unitary(
                qpoint,
                reference.basis_positions,
                dof,
            )
            calculated[q_index] = (
                unitary.conj().T @ calculated[q_index] @ unitary
            )

    difference = calculated - reference.matrices
    per_q_abs = np.max(np.abs(difference), axis=(1, 2))
    per_q_scale = np.max(np.abs(reference.matrices), axis=(1, 2))
    per_q_relative = np.zeros_like(per_q_abs)
    nonzero = per_q_scale > 0.0
    per_q_relative[nonzero] = per_q_abs[nonzero] / per_q_scale[nonzero]
    per_q_relative[~nonzero & (per_q_abs > 0.0)] = np.inf

    per_q_allowed = tolerance * np.maximum(1.0, per_q_scale)
    normalized_gate_error = np.divide(
        per_q_abs,
        per_q_allowed,
        out=np.zeros_like(per_q_abs),
        where=per_q_allowed > 0.0,
    )
    if tolerance == 0.0:
        normalized_gate_error[per_q_abs > 0.0] = np.inf
    worst_index = int(np.argmax(normalized_gate_error))
    max_abs = float(np.max(per_q_abs))
    reference_scale = float(np.max(per_q_scale))
    max_relative = float(np.max(per_q_relative))
    allowed = tolerance * max(1.0, reference_scale)
    qpoints_tuple = tuple(
        tuple(float(component) for component in qpoint)
        for qpoint in reference.qpoints
    )
    return IFCDynamicalComparison(
        passed=bool(np.all(per_q_abs <= per_q_allowed)),
        tolerance=tolerance,
        allowed_max_abs_error=allowed,
        max_abs_error=max_abs,
        max_relative_error=max_relative,
        reference_scale=reference_scale,
        per_q_max_abs_error=tuple(float(value) for value in per_q_abs),
        per_q_relative_error=tuple(float(value) for value in per_q_relative),
        qpoints=qpoints_tuple,
        worst_qpoint=qpoints_tuple[worst_index],
        q_units="radian",
        gauge=reference.gauge,
        source=reference.source,
        provenance=reference.provenance,
        source_sha256=reference.source_sha256,
        reference_sha256=reference.content_sha256,
        ifc_sha256=_ifc_hash(
            ifc,
            masses=masses,
            dof=dof,
            raw_terms=raw_terms,
        ),
        input_term_count=len(raw_terms),
        merged_translation_count=len(merged),
        duplicate_translations=duplicates,
    )


__all__ = [
    "IFCDynamicalComparison",
    "IFCDynamicalReference",
    "basis_phase_unitary",
    "compare_ifc_dynamical_matrices",
]
