"""Exact compact-input loader for silicon bulk-oracle calculations.

The target-q runner and final release aggregator must construct the identical
``BulkThreePhononProblem`` without importing a repository-local script.  This
private helper centralizes that construction while leaving all public solver
signatures unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .bulk_three_phonon import BulkThreePhononProblem
from .io.phono3py import read_phono3py_fc3
from .io.phono3py_crystal import read_phono3py_crystal_mapping
from .io.phono3py_fc2 import read_phono3py_fc2
from .momentum import UniformBZMesh
from .translational_fc3 import convert_fc3_to_translational
from .translational_fc3_validation import validate_translational_fc3


class SiliconBulkProblemError(RuntimeError):
    """Raised when authenticated compact silicon inputs fail preflight."""


def _mesh(values: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError("mesh must contain exactly three integers.")
    result = tuple(int(value) for value in values)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in values
    ) or any(value <= 0 for value in result):
        raise ValueError("mesh must contain exactly three positive integers.")
    return result


def _finite(value: float, *, name: str, positive: bool) -> float:
    result = float(value)
    if not math.isfinite(result) or (
        result <= 0.0 if positive else result < 0.0
    ):
        relation = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {relation}.")
    return result


def load_silicon_bulk_three_phonon_problem(
    *,
    fc2_compact_path: str,
    fc3_compact_path: str,
    phono3py_yaml_path: str,
    mesh: Sequence[int],
    sigma_thz: float,
    cutoff_frequency_thz: float,
    frequency_factor_to_thz: float,
    degenerate_tolerance_thz: float,
    make_r0_average: bool,
    symprec: float,
    translation_atol: float,
    translation_rtol: float,
    asr_atol: float,
    asr_rtol: float,
    metadata: Mapping[str, Any],
) -> BulkThreePhononProblem:
    """Load and validate the exact compact FC2/FC3 bulk problem."""

    selected_mesh = _mesh(mesh)
    sigma = _finite(sigma_thz, name="sigma_thz", positive=True)
    cutoff = _finite(
        cutoff_frequency_thz,
        name="cutoff_frequency_thz",
        positive=False,
    )
    frequency_factor = _finite(
        frequency_factor_to_thz,
        name="frequency_factor_to_thz",
        positive=True,
    )
    degeneracy = _finite(
        degenerate_tolerance_thz,
        name="degenerate_tolerance_thz",
        positive=True,
    )
    symmetry_tolerance = _finite(
        symprec, name="symprec", positive=True
    )
    translation_absolute = _finite(
        translation_atol, name="translation_atol", positive=False
    )
    translation_relative = _finite(
        translation_rtol, name="translation_rtol", positive=False
    )
    asr_absolute = _finite(asr_atol, name="asr_atol", positive=False)
    asr_relative = _finite(asr_rtol, name="asr_rtol", positive=False)
    if not isinstance(make_r0_average, (bool, np.bool_)):
        raise TypeError("make_r0_average must be boolean.")
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping.")

    fc2_mapping = read_phono3py_crystal_mapping(
        phono3py_yaml_path,
        purpose="fc2",
        is_nac=False,
    )
    fc3_mapping = read_phono3py_crystal_mapping(
        phono3py_yaml_path,
        purpose="fc3",
        is_nac=False,
    )
    fc2 = read_phono3py_fc2(
        fc2_compact_path,
        fc2_mapping.supercell.masses,
        p2s_map=fc2_mapping.p2s,
        s2p_map=fc2_mapping.s2p,
        compact=True,
    )
    raw_fc3 = read_phono3py_fc3(
        fc3_compact_path,
        fc3_mapping.supercell.masses,
        p2s_map=fc3_mapping.p2s,
        s2p_map=fc3_mapping.s2p,
        compact=True,
    )
    fc3 = convert_fc3_to_translational(
        raw_fc3,
        fc3_mapping,
        make_r0_average=bool(make_r0_average),
        symprec=symmetry_tolerance,
        translation_atol=translation_absolute,
        translation_rtol=translation_relative,
    )
    validation = validate_translational_fc3(
        fc3,
        fc3_mapping,
        asr_atol=asr_absolute,
        asr_rtol=asr_relative,
    )
    if not validation.passed:
        raise SiliconBulkProblemError(
            "Compact translational FC3 failed the exact ASR/support "
            "preflight: "
            + " ".join(validation.errors)
        )
    return BulkThreePhononProblem(
        fc2=fc2,
        fc2_mapping=fc2_mapping,
        fc3=fc3,
        fc3_mapping=fc3_mapping,
        mesh=UniformBZMesh(selected_mesh),
        sigma_thz=sigma,
        bz_reduced_reciprocal_lattice=None,
        cutoff_frequency_thz=cutoff,
        frequency_factor_to_thz=frequency_factor,
        degenerate_tolerance_thz=degeneracy,
        metadata=dict(metadata),
    )


__all__ = [
    "SiliconBulkProblemError",
    "load_silicon_bulk_three_phonon_problem",
]
