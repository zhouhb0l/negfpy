"""Frozen observation modes and pole diagnostics for material ``N_q=1`` SCBA."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ._harmonic import assemble_device_dynamical_matrix
from .types import CubicSCBAProblem


Array = np.ndarray


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    descriptor = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(b"negfpy.material-nq1-observation-array.v1\0")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class ObservationSubspace:
    """One nondegenerate mode or complete degenerate harmonic subspace."""

    ordinal: int
    frequency: float
    eigenvalue_indices: tuple[int, ...]
    basis: Array
    projector_sha256: str

    def __post_init__(self) -> None:
        basis = np.array(self.basis, dtype=np.complex128, order="C", copy=True)
        indices = tuple(int(value) for value in self.eigenvalue_indices)
        if self.ordinal < 0 or not math.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("observation subspace frequency must be positive")
        if not indices or basis.ndim != 2 or basis.shape[1] != len(indices):
            raise ValueError("observation subspace basis dimensions are invalid")
        identity = np.eye(basis.shape[1], dtype=np.complex128)
        if not np.allclose(basis.conj().T @ basis, identity, rtol=1e-12, atol=1e-12):
            raise ValueError("observation subspace basis is not orthonormal")
        projector = basis @ basis.conj().T
        if _array_sha256(projector) != self.projector_sha256:
            raise ValueError("observation subspace projector hash does not reproduce")
        basis.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "eigenvalue_indices", indices)

    @property
    def multiplicity(self) -> int:
        return len(self.eigenvalue_indices)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "frequency": self.frequency,
            "eigenvalue_indices": list(self.eigenvalue_indices),
            "multiplicity": self.multiplicity,
            "projector_sha256": self.projector_sha256,
        }


@dataclass(frozen=True, slots=True, eq=False)
class ObservationModeSet:
    observation_layers: tuple[int, ...]
    dof_indices: tuple[int, ...]
    resolution_threshold: float
    degeneracy_tolerance: float
    harmonic_eigenvalues: Array
    harmonic_block_sha256: str
    subspaces: tuple[ObservationSubspace, ...]

    def __post_init__(self) -> None:
        layers = tuple(int(value) for value in self.observation_layers)
        indices = tuple(int(value) for value in self.dof_indices)
        eigenvalues = np.array(
            self.harmonic_eigenvalues, dtype=np.float64, order="C", copy=True
        )
        if not layers or not indices or eigenvalues.shape != (len(indices),):
            raise ValueError("observation mode-set dimensions are invalid")
        if not self.subspaces:
            raise ValueError("resolved observation mode set must be nonempty")
        eigenvalues.setflags(write=False)
        object.__setattr__(self, "observation_layers", layers)
        object.__setattr__(self, "dof_indices", indices)
        object.__setattr__(self, "harmonic_eigenvalues", eigenvalues)
        object.__setattr__(self, "subspaces", tuple(self.subspaces))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "negfpy.material-nq1-observation-modes.v1",
            "observation_layers": list(self.observation_layers),
            "dof_indices": list(self.dof_indices),
            "resolution_threshold": self.resolution_threshold,
            "degeneracy_tolerance": self.degeneracy_tolerance,
            "harmonic_eigenvalues": self.harmonic_eigenvalues.tolist(),
            "harmonic_eigenvalues_sha256": _array_sha256(
                self.harmonic_eigenvalues
            ),
            "harmonic_block_sha256": self.harmonic_block_sha256,
            "subspaces": [item.as_dict() for item in self.subspaces],
        }
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("ascii")
        return payload | {"mode_set_sha256": hashlib.sha256(encoded).hexdigest()}


def build_observation_mode_set(
    problem: CubicSCBAProblem,
    *,
    observation_layers: Sequence[int],
    grid_step: float,
    eta_lead: float,
    degeneracy_rtol: float = 1.0e-10,
) -> ObservationModeSet:
    """Freeze all resolved harmonic subspaces before inspecting SCBA output."""

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem")
    layers = tuple(int(value) for value in observation_layers)
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("observation_layers must be nonempty and unique")
    if min(layers) < 0 or max(layers) >= problem.device.n_layers:
        raise ValueError("observation layer is outside the device")
    step = float(grid_step)
    eta = float(eta_lead)
    if not math.isfinite(step) or not math.isfinite(eta) or step <= 0 or eta <= 0:
        raise ValueError("grid_step and eta_lead must be finite and positive")
    rtol = float(degeneracy_rtol)
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("degeneracy_rtol must be finite and positive")

    npl = problem.device.dof_per_layer
    indices = tuple(
        layer * npl + offset for layer in layers for offset in range(npl)
    )
    dynamical = assemble_device_dynamical_matrix(problem.device)
    block = np.asarray(dynamical[np.ix_(indices, indices)], dtype=np.complex128)
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    stability_tolerance = 1.0e-12 * scale
    if float(np.min(eigenvalues)) < -stability_tolerance:
        raise ValueError("observation harmonic block is unstable")
    frequencies = np.sqrt(np.maximum(eigenvalues, 0.0))
    threshold = max(10.0 * step, 10.0 * eta)
    degeneracy_tolerance = rtol * scale

    groups: list[tuple[int, ...]] = []
    start = 0
    while start < eigenvalues.size:
        stop = start + 1
        while stop < eigenvalues.size and abs(
            float(eigenvalues[stop] - eigenvalues[start])
        ) <= degeneracy_tolerance:
            stop += 1
        groups.append(tuple(range(start, stop)))
        start = stop

    subspaces: list[ObservationSubspace] = []
    for group in groups:
        group_frequencies = frequencies[np.asarray(group)]
        frequency = float(np.mean(group_frequencies))
        if frequency < threshold:
            continue
        basis = np.asarray(eigenvectors[:, group], dtype=np.complex128)
        projector_sha = _array_sha256(basis @ basis.conj().T)
        subspaces.append(ObservationSubspace(
            ordinal=len(subspaces),
            frequency=frequency,
            eigenvalue_indices=group,
            basis=basis,
            projector_sha256=projector_sha,
        ))
    if not subspaces:
        raise ValueError(
            "certifying grid resolves no observation modes; reduce grid spacing"
        )
    return ObservationModeSet(
        observation_layers=layers,
        dof_indices=indices,
        resolution_threshold=threshold,
        degeneracy_tolerance=degeneracy_tolerance,
        harmonic_eigenvalues=eigenvalues,
        harmonic_block_sha256=_array_sha256(block),
        subspaces=tuple(subspaces),
    )


@dataclass(frozen=True, slots=True)
class PoleSubspaceMeasurement:
    ordinal: int
    frequency: float
    linewidths: tuple[float, ...]
    shifts: tuple[float, ...]

    def as_dict(self) -> Mapping[str, Any]:
        return MappingProxyType({
            "ordinal": self.ordinal,
            "frequency": self.frequency,
            "linewidths": list(self.linewidths),
            "shifts": list(self.shifts),
        })


def measure_observation_mode_poles(
    omega: Any,
    sigma_retarded: Any,
    mode_set: ObservationModeSet,
) -> tuple[PoleSubspaceMeasurement, ...]:
    """Project interpolated retarded self-energy into frozen subspaces."""

    grid = np.asarray(omega, dtype=np.float64)
    sigma = np.asarray(sigma_retarded, dtype=np.complex128)
    if grid.ndim != 1 or sigma.ndim != 3 or sigma.shape[0] != grid.size:
        raise ValueError("frequency/self-energy shapes are invalid")
    selected = np.asarray(mode_set.dof_indices, dtype=np.int64)
    if sigma.shape[1:] != (sigma.shape[1], sigma.shape[1]) or selected[-1] >= sigma.shape[1]:
        raise ValueError("self-energy device dimensions are invalid")
    if np.any(np.diff(grid) <= 0) or not np.all(np.isfinite(sigma)):
        raise ValueError("frequency/self-energy data are invalid")

    results: list[PoleSubspaceMeasurement] = []
    for subspace in mode_set.subspaces:
        frequency = subspace.frequency
        if frequency < grid[0] or frequency > grid[-1]:
            raise ValueError("observation frequency lies outside the grid")
        upper = int(np.searchsorted(grid, frequency, side="left"))
        if upper == 0:
            interpolated = sigma[0]
        elif upper == grid.size:
            interpolated = sigma[-1]
        elif grid[upper] == frequency:
            interpolated = sigma[upper]
        else:
            lower = upper - 1
            weight = (frequency - grid[lower]) / (grid[upper] - grid[lower])
            interpolated = (1.0 - weight) * sigma[lower] + weight * sigma[upper]
        local = interpolated[np.ix_(selected, selected)]
        projected = subspace.basis.conj().T @ local @ subspace.basis
        hermitian = 0.5 * (projected + projected.conj().T)
        antihermitian = (projected - projected.conj().T) / (2.0j)
        shifts = np.linalg.eigvalsh(hermitian).real / (2.0 * frequency)
        linewidths = -np.linalg.eigvalsh(antihermitian).real / frequency
        results.append(PoleSubspaceMeasurement(
            ordinal=subspace.ordinal,
            frequency=frequency,
            linewidths=tuple(float(value) for value in linewidths),
            shifts=tuple(float(value) for value in shifts),
        ))
    return tuple(results)


__all__ = [
    "ObservationModeSet",
    "ObservationSubspace",
    "PoleSubspaceMeasurement",
    "build_observation_mode_set",
    "measure_observation_mode_poles",
]
