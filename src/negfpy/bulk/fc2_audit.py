"""Mapped physical audit for phono3py-style harmonic force constants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from negfpy.anharmonic.bulk_harmonic import dynamical_matrix_from_fc2
from negfpy.anharmonic.crystal import CrystalMapping, validate_crystal_mapping
from negfpy.anharmonic.fc2 import FC2Data, validate_fc2

Array = np.ndarray


def _readonly(values: Any, *, dtype: Any) -> Array:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class MappedFC2AuditReport:
    """Reciprocity, ASR, rotational, Hermiticity, and stability diagnostics."""

    schema_valid: bool
    mapping_valid: bool
    translation_redundancy_valid: bool
    reciprocal_valid: bool
    asr_valid: bool
    rotational_valid: bool
    hermiticity_valid: bool
    stable: bool
    qpoints: Array
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        qpoints = _readonly(self.qpoints, dtype=np.float64)
        if qpoints.ndim != 2 or qpoints.shape[1:] != (3,):
            raise ValueError("qpoints must have shape (n_qpoints, 3).")
        if not np.all(np.isfinite(qpoints)):
            raise ValueError("qpoints must be finite.")
        metrics = {str(key): float(value) for key, value in self.metrics.items()}
        if any(not np.isfinite(value) for value in metrics.values()):
            raise ValueError("FC2 audit metrics must be finite.")
        object.__setattr__(self, "qpoints", qpoints)
        object.__setattr__(self, "errors", tuple(str(value) for value in self.errors))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def passed(self) -> bool:
        return bool(
            self.schema_valid
            and self.mapping_valid
            and self.translation_redundancy_valid
            and self.reciprocal_valid
            and self.asr_valid
            and self.rotational_valid
            and self.hermiticity_valid
            and self.stable
            and not self.errors
        )


def audit_mapped_fc2(
    fc2: FC2Data,
    mapping: CrystalMapping,
    *,
    absolute_tolerance: float = 1.0e-10,
    relative_tolerance: float = 1.0e-8,
    stability_tolerance: float = 1.0e-10,
    random_q_count: int = 8,
    random_seed: int = 1729,
) -> MappedFC2AuditReport:
    """Audit mapped FC2 without symmetrizing, correcting, or dropping terms."""

    if not isinstance(fc2, FC2Data):
        raise TypeError("fc2 must be an FC2Data instance.")
    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping instance.")
    absolute = float(absolute_tolerance)
    relative = float(relative_tolerance)
    stability = float(stability_tolerance)
    if any(not np.isfinite(value) or value < 0.0 for value in (absolute, relative, stability)):
        raise ValueError("FC2 audit tolerances must be finite and nonnegative.")
    if isinstance(random_q_count, (bool, np.bool_)) or not isinstance(
        random_q_count, (int, np.integer)
    ):
        raise TypeError("random_q_count must be an integer.")
    if int(random_q_count) < 0:
        raise ValueError("random_q_count must be nonnegative.")

    schema_report = validate_fc2(fc2)
    mapping_report = validate_crystal_mapping(mapping)
    schema_valid = schema_report.passed
    mapping_valid = mapping_report.passed and mapping.purpose == "fc2"
    errors = [*schema_report.errors, *mapping_report.errors]
    warnings = [*schema_report.warnings, *mapping_report.warnings]
    if mapping.purpose != "fc2":
        errors.append("CrystalMapping purpose must be 'fc2'.")
    if not schema_valid or not mapping_valid:
        return MappedFC2AuditReport(
            schema_valid=schema_valid,
            mapping_valid=mapping_valid,
            translation_redundancy_valid=False,
            reciprocal_valid=False,
            asr_valid=False,
            rotational_valid=False,
            hermiticity_valid=False,
            stable=False,
            qpoints=np.zeros((0, 3)),
            errors=tuple(errors),
            warnings=tuple(warnings),
        )

    translation_redundancy_valid = True
    redundancy_max = 0.0
    redundancy_relative = 0.0
    redundancy_checked_rows = 0
    if not fc2.compact:
        representatives = {int(value) for value in mapping.p2s}
        redundant_rows = [
            source
            for source in range(mapping.n_supercell_atoms)
            if source not in representatives
        ]
        translations = mapping.translation_permutations
        if redundant_rows and translations is None:
            translation_redundancy_valid = False
            errors.append(
                "Full FC2 redundancy cannot be audited without explicit pure "
                "translation permutations."
            )
        else:
            assert translations is not None or not redundant_rows
            values = np.asarray(fc2.values)
            for source_first in redundant_rows:
                representative = int(mapping.s2p[source_first])
                candidates = [
                    row
                    for row in np.asarray(translations, dtype=np.int64)
                    if int(row[source_first]) == representative
                ]
                unique = {
                    tuple(int(value) for value in row): row
                    for row in candidates
                }
                if len(unique) != 1:
                    translation_redundancy_valid = False
                    errors.append(
                        "Full FC2 has no unique pure translation from "
                        f"first-axis atom {source_first} to representative "
                        f"{representative}."
                    )
                    continue
                translation = next(iter(unique.values()))
                expected = values[representative][translation]
                actual = values[source_first]
                residual = float(np.max(np.abs(actual - expected)))
                scale = max(
                    float(np.max(np.abs(actual))),
                    float(np.max(np.abs(expected))),
                    np.finfo(np.float64).tiny,
                )
                relative_residual = residual / scale
                redundancy_max = max(redundancy_max, residual)
                redundancy_relative = max(
                    redundancy_relative,
                    relative_residual,
                )
                redundancy_checked_rows += 1
                threshold = absolute + relative * scale
                if residual > threshold:
                    translation_redundancy_valid = False
                    errors.append(
                        "Full FC2 redundant translation audit failed for "
                        f"first-axis atom {source_first} against representative "
                        f"{representative}: residual {residual:.6e}, allowed "
                        f"{threshold:.6e}."
                    )

    rng = np.random.default_rng(int(random_seed))
    fixed = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
            [0.173, 0.297, 0.419],
        ],
        dtype=np.float64,
    )
    random = rng.uniform(-0.5, 0.5, size=(int(random_q_count), 3))
    qpoints = np.concatenate((fixed, random), axis=0)
    hermiticity_max = 0.0
    time_reversal_max = 0.0
    dynamical_scale = 0.0
    minimum_eigenvalue = np.inf
    for qpoint in qpoints:
        positive = dynamical_matrix_from_fc2(
            fc2,
            mapping,
            qpoint,
            hermitianize=False,
        )
        negative = dynamical_matrix_from_fc2(
            fc2,
            mapping,
            -qpoint,
            hermitianize=False,
        )
        dynamical_scale = max(dynamical_scale, float(np.max(np.abs(positive))))
        hermiticity_max = max(
            hermiticity_max,
            float(np.max(np.abs(positive - positive.conj().T))),
        )
        time_reversal_max = max(
            time_reversal_max,
            float(np.max(np.abs(negative - positive.conj()))),
        )
        minimum_eigenvalue = min(
            minimum_eigenvalue,
            float(np.linalg.eigvalsh((positive + positive.conj().T) / 2.0)[0]),
        )
    reciprocal_threshold = absolute + relative * dynamical_scale
    hermiticity_valid = hermiticity_max <= reciprocal_threshold
    reciprocal_valid = time_reversal_max <= reciprocal_threshold
    stable = minimum_eigenvalue >= -stability
    if not hermiticity_valid:
        errors.append("Sampled FC2 dynamical matrices are not Hermitian.")
    if not reciprocal_valid:
        errors.append("FC2 time-reversal/reciprocity relation failed.")
    if not stable:
        errors.append("FC2 has an unstable mode on the deterministic q sample.")

    gamma = dynamical_matrix_from_fc2(
        fc2,
        mapping,
        np.zeros(3),
        hermitianize=False,
    )
    translations = np.zeros((3 * mapping.n_primitive_atoms, 3))
    for atom, mass in enumerate(mapping.primitive.masses):
        translations[3 * atom : 3 * atom + 3] = np.sqrt(mass) * np.eye(3)
    asr_residual = gamma @ translations
    asr_scale = max(
        float(np.max(np.abs(gamma))) * float(np.max(np.abs(translations))),
        np.finfo(np.float64).tiny,
    )
    asr_max = float(np.max(np.abs(asr_residual)))
    asr_threshold = absolute + relative * asr_scale
    asr_valid = asr_max <= asr_threshold
    if not asr_valid:
        errors.append("FC2 translational acoustic sum rule failed.")

    values = np.asarray(fc2.values)
    moment = np.zeros((mapping.n_primitive_atoms, 3, 3, 3), dtype=np.float64)
    for primitive_i, representative_i in enumerate(mapping.p2s):
        first_axis = primitive_i if fc2.compact else int(representative_i)
        for supercell_j in range(mapping.n_supercell_atoms):
            count, offset = mapping.multiplicity[supercell_j, primitive_i]
            vectors = mapping.shortest_vectors[int(offset) : int(offset + count)]
            average_cartesian = np.mean(
                vectors @ mapping.primitive.lattice,
                axis=0,
            )
            moment[primitive_i] += (
                values[first_axis, supercell_j, :, :, None]
                * average_cartesian[None, None, :]
            )
    rotational_residual = moment - moment.swapaxes(2, 3)
    rotational_scale = max(float(np.max(np.abs(moment))), np.finfo(np.float64).tiny)
    rotational_max = float(np.max(np.abs(rotational_residual)))
    rotational_threshold = absolute + relative * rotational_scale
    rotational_valid = rotational_max <= rotational_threshold
    if not rotational_valid:
        errors.append("FC2 harmonic rotational sum rule failed.")

    metrics = {
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
        "stability_tolerance": stability,
        "dynamical_scale": dynamical_scale,
        "hermiticity_max_abs": hermiticity_max,
        "time_reversal_max_abs": time_reversal_max,
        "reciprocal_threshold": reciprocal_threshold,
        "minimum_sampled_eigenvalue": minimum_eigenvalue,
        "asr_max_abs": asr_max,
        "asr_relative": asr_max / asr_scale,
        "asr_threshold": asr_threshold,
        "rotational_max_abs": rotational_max,
        "rotational_relative": rotational_max / rotational_scale,
        "rotational_threshold": rotational_threshold,
        "translation_redundancy_checked_rows": float(
            redundancy_checked_rows
        ),
        "translation_redundancy_max_abs": redundancy_max,
        "translation_redundancy_max_relative": redundancy_relative,
    }
    return MappedFC2AuditReport(
        schema_valid=True,
        mapping_valid=True,
        translation_redundancy_valid=translation_redundancy_valid,
        reciprocal_valid=reciprocal_valid,
        asr_valid=asr_valid,
        rotational_valid=rotational_valid,
        hermiticity_valid=hermiticity_valid,
        stable=stable,
        qpoints=qpoints,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
    )


__all__ = ["MappedFC2AuditReport", "audit_mapped_fc2"]
