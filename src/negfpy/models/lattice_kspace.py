"""k-space-ready toy lattices for 2D/3D phonon NEGF benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.types import Device1D, DeviceKSpace, LeadKSpace


@dataclass(frozen=True)
class SquareLatticeParams:
    mass: float
    spring_x: float
    spring_y: float


@dataclass(frozen=True)
class CubicLatticeParams:
    mass: float
    spring_x: float
    spring_y: float
    spring_z: float
    onsite_pinning: float = 0.0


def _get_ky(kpar: tuple[float, ...] | None) -> float:
    if kpar is None or len(kpar) == 0:
        return 0.0
    if len(kpar) > 1:
        raise ValueError("Square-lattice kpar must contain at most one transverse component: (ky,).")
    return float(kpar[0])


def _get_ky_kz(kpar: tuple[float, ...] | None) -> tuple[float, float]:
    if kpar is None:
        return 0.0, 0.0
    if len(kpar) > 2:
        raise ValueError("Cubic-lattice kpar must contain at most two transverse components: (ky,) or (ky, kz).")
    if len(kpar) == 1:
        return float(kpar[0]), 0.0
    return float(kpar[0]), float(kpar[1])


def square_lattice_lead(params: SquareLatticeParams) -> LeadKSpace:
    """Return lead blocks for x-transport with periodic y via ky."""

    def _builder(kpar: tuple[float, ...] | None) -> tuple[np.ndarray, np.ndarray]:
        ky = _get_ky(kpar)
        d00_val = (
            2.0 * params.spring_x
            + 2.0 * params.spring_y
            - 2.0 * params.spring_y * np.cos(ky)
        ) / params.mass
        d01_val = -params.spring_x / params.mass
        d00 = np.array([[d00_val]], dtype=np.complex128)
        d01 = np.array([[d01_val]], dtype=np.complex128)
        return d00, d01

    return LeadKSpace(blocks_builder=_builder)


def square_lattice_device(
    n_layers: int,
    params: SquareLatticeParams,
    defect_index: int | None = None,
    defect_mass: float | None = None,
) -> DeviceKSpace:
    """Return finite device for x-transport with periodic y via ky."""

    if defect_index is not None and (defect_index < 0 or defect_index >= n_layers):
        raise ValueError("defect_index out of range")

    def _builder(kpar: tuple[float, ...] | None) -> Device1D:
        ky = _get_ky(kpar)
        onsite_blocks: list[np.ndarray] = []
        for i in range(n_layers):
            mass = (
                defect_mass
                if defect_index is not None and defect_mass is not None and i == defect_index
                else params.mass
            )
            d00_val = (
                2.0 * params.spring_x
                + 2.0 * params.spring_y
                - 2.0 * params.spring_y * np.cos(ky)
            ) / mass
            onsite_blocks.append(np.array([[d00_val]], dtype=np.complex128))

        d01_val = -params.spring_x / params.mass
        coupling_blocks = [
            np.array([[d01_val]], dtype=np.complex128) for _ in range(n_layers - 1)
        ]
        return Device1D(onsite_blocks=onsite_blocks, coupling_blocks=coupling_blocks)

    return DeviceKSpace(device_builder=_builder)


def cubic_lattice_lead(params: CubicLatticeParams) -> LeadKSpace:
    """Return lead blocks for x-transport with periodic y,z via (ky,kz)."""

    def _builder(kpar: tuple[float, ...] | None) -> tuple[np.ndarray, np.ndarray]:
        ky, kz = _get_ky_kz(kpar)
        d00_val = (
            2.0 * params.spring_x
            + 2.0 * params.spring_y
            + 2.0 * params.spring_z
            + params.onsite_pinning
            - 2.0 * params.spring_y * np.cos(ky)
            - 2.0 * params.spring_z * np.cos(kz)
        ) / params.mass
        d01_val = -params.spring_x / params.mass
        d00 = np.array([[d00_val]], dtype=np.complex128)
        d01 = np.array([[d01_val]], dtype=np.complex128)
        return d00, d01

    return LeadKSpace(blocks_builder=_builder)


def cubic_lattice_device(
    n_layers: int,
    params: CubicLatticeParams,
    defect_index: int | None = None,
    defect_mass: float | None = None,
) -> DeviceKSpace:
    """Return finite device for x-transport with periodic y,z via (ky,kz)."""

    if defect_index is not None and (defect_index < 0 or defect_index >= n_layers):
        raise ValueError("defect_index out of range")

    def _builder(kpar: tuple[float, ...] | None) -> Device1D:
        ky, kz = _get_ky_kz(kpar)
        onsite_blocks: list[np.ndarray] = []
        for i in range(n_layers):
            mass = (
                defect_mass
                if defect_index is not None and defect_mass is not None and i == defect_index
                else params.mass
            )
            d00_val = (
                2.0 * params.spring_x
                + 2.0 * params.spring_y
                + 2.0 * params.spring_z
                + params.onsite_pinning
                - 2.0 * params.spring_y * np.cos(ky)
                - 2.0 * params.spring_z * np.cos(kz)
            ) / mass
            onsite_blocks.append(np.array([[d00_val]], dtype=np.complex128))

        d01_val = -params.spring_x / params.mass
        coupling_blocks = [
            np.array([[d01_val]], dtype=np.complex128) for _ in range(n_layers - 1)
        ]
        return Device1D(onsite_blocks=onsite_blocks, coupling_blocks=coupling_blocks)

    return DeviceKSpace(device_builder=_builder)
