"""Strict rank binding and scaling selection for remote validation.

This module contains orchestration policy only.  It never partitions one
SCBA solve and never changes scientific controls.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import pathlib
from collections.abc import Sequence
from typing import Any


RANK_LADDER = (1, 2, 4, 8, 16)


class MPIScalingError(RuntimeError):
    """Raised when rank-scaling evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class ScalingMeasurement:
    rank_count: int
    wall_seconds: float
    completed: bool = True
    numerically_equal: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.rank_count, bool) or self.rank_count not in RANK_LADDER:
            raise ValueError(f"rank_count must be one of {RANK_LADDER}.")
        if not math.isfinite(self.wall_seconds) or self.wall_seconds <= 0.0:
            raise ValueError("wall_seconds must be finite and positive.")
        object.__setattr__(self, "completed", bool(self.completed))
        object.__setattr__(self, "numerically_equal", bool(self.numerically_equal))

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank_count": self.rank_count,
            "wall_seconds": self.wall_seconds,
            "completed": self.completed,
            "numerically_equal": self.numerically_equal,
        }


@dataclass(frozen=True)
class ScalingSelection:
    measurements: tuple[ScalingMeasurement, ...]
    fastest_rank_count: int
    fastest_wall_seconds: float
    smallest_near_fastest_rank_count: int
    candidate_speedup: float
    toy_rank_count: int
    silicon_rank_count: int
    toy_mpi_enabled: bool
    within_fastest_fraction: float
    minimum_toy_speedup: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "negfpy.remote_mpi_scaling.selection.v1",
            "rank_ladder": [item.rank_count for item in self.measurements],
            "measurements": [item.as_dict() for item in self.measurements],
            "fastest_rank_count": self.fastest_rank_count,
            "fastest_wall_seconds": self.fastest_wall_seconds,
            "smallest_near_fastest_rank_count": (
                self.smallest_near_fastest_rank_count
            ),
            "candidate_speedup": self.candidate_speedup,
            "toy_rank_count": self.toy_rank_count,
            "silicon_rank_count": self.silicon_rank_count,
            "toy_mpi_enabled": self.toy_mpi_enabled,
            "within_fastest_fraction": self.within_fastest_fraction,
            "minimum_toy_speedup": self.minimum_toy_speedup,
        }


def required_rank_ladder(physical_core_count: int) -> tuple[int, ...]:
    """Return the preregistered ladder that fits without oversubscription."""

    if isinstance(physical_core_count, bool) or physical_core_count <= 0:
        raise ValueError("physical_core_count must be positive.")
    return tuple(rank for rank in RANK_LADDER if rank <= physical_core_count)


def openmpi_binding_arguments(
    rank_count: int,
    *,
    physical_core_count: int,
    socket_count: int,
) -> tuple[str, ...]:
    """Return one-core-per-rank Open MPI binding arguments.

    Even rank counts are distributed equally across sockets when possible.
    ``--nooversubscribe`` makes a topology mismatch a hard infrastructure
    failure rather than silently sharing a core.
    """

    if isinstance(rank_count, bool) or rank_count <= 0:
        raise ValueError("rank_count must be positive.")
    if isinstance(physical_core_count, bool) or physical_core_count <= 0:
        raise ValueError("physical_core_count must be positive.")
    if isinstance(socket_count, bool) or socket_count <= 0:
        raise ValueError("socket_count must be positive.")
    if rank_count > physical_core_count:
        raise ValueError("rank_count exceeds the physical-core count.")

    arguments = ["--nooversubscribe", "--bind-to", "core"]
    if rank_count >= socket_count and rank_count % socket_count == 0:
        per_socket = rank_count // socket_count
        arguments.extend(("--map-by", f"ppr:{per_socket}:socket:PE=1"))
    else:
        arguments.extend(("--map-by", "core:PE=1"))
    arguments.append("--report-bindings")
    return tuple(arguments)


def environment_mpi_argv(
    environment_prefix: str | pathlib.Path,
    *,
    rank_count: int,
    physical_core_count: int,
    socket_count: int,
    program: Sequence[str],
) -> tuple[str, ...]:
    """Construct a launch using only ``<prefix>/bin/mpiexec``."""

    prefix = pathlib.Path(environment_prefix)
    if not program:
        raise ValueError("program must be non-empty.")
    return (
        str(prefix / "bin" / "mpiexec"),
        *openmpi_binding_arguments(
            rank_count,
            physical_core_count=physical_core_count,
            socket_count=socket_count,
        ),
        "-n",
        str(rank_count),
        *(str(item) for item in program),
    )


def select_rank_count(
    measurements: Sequence[ScalingMeasurement],
    *,
    physical_core_count: int,
    within_fastest_fraction: float = 0.05,
    minimum_toy_speedup: float = 2.0,
) -> ScalingSelection:
    """Apply the locked toy/silicon rank-selection policy.

    All preregistered measurements must be present, complete, and numerically
    equal.  The silicon choice is the smallest rank within five percent of the
    fastest time.  The toy gate uses that choice only when it is at least twice
    as fast as one rank; otherwise toy certification remains serial/FFT.
    """

    if not math.isfinite(within_fastest_fraction) or not 0.0 <= within_fastest_fraction < 1.0:
        raise ValueError("within_fastest_fraction must be in [0, 1).")
    if not math.isfinite(minimum_toy_speedup) or minimum_toy_speedup <= 1.0:
        raise ValueError("minimum_toy_speedup must be greater than one.")
    expected = required_rank_ladder(physical_core_count)
    ordered = tuple(sorted(tuple(measurements), key=lambda item: item.rank_count))
    observed = tuple(item.rank_count for item in ordered)
    if observed != expected:
        raise MPIScalingError(
            f"rank ladder is incomplete or duplicated: {observed} != {expected}"
        )
    invalid = [
        item.rank_count
        for item in ordered
        if not item.completed or not item.numerically_equal
    ]
    if invalid:
        raise MPIScalingError(
            "rank measurements failed completion/equality: "
            + ", ".join(str(item) for item in invalid)
        )

    baseline = ordered[0]
    fastest = min(ordered, key=lambda item: (item.wall_seconds, item.rank_count))
    limit = fastest.wall_seconds * (1.0 + within_fastest_fraction)
    candidate = min(
        (item for item in ordered if item.wall_seconds <= limit),
        key=lambda item: item.rank_count,
    )
    speedup = baseline.wall_seconds / candidate.wall_seconds
    toy_mpi_enabled = bool(
        candidate.rank_count > 1 and speedup >= minimum_toy_speedup
    )
    return ScalingSelection(
        measurements=ordered,
        fastest_rank_count=fastest.rank_count,
        fastest_wall_seconds=fastest.wall_seconds,
        smallest_near_fastest_rank_count=candidate.rank_count,
        candidate_speedup=speedup,
        toy_rank_count=candidate.rank_count if toy_mpi_enabled else 1,
        silicon_rank_count=candidate.rank_count,
        toy_mpi_enabled=toy_mpi_enabled,
        within_fastest_fraction=within_fastest_fraction,
        minimum_toy_speedup=minimum_toy_speedup,
    )


__all__ = [
    "MPIScalingError",
    "RANK_LADDER",
    "ScalingMeasurement",
    "ScalingSelection",
    "environment_mpi_argv",
    "openmpi_binding_arguments",
    "required_rank_ladder",
    "select_rank_count",
]
