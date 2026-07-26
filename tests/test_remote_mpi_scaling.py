from __future__ import annotations

import importlib
import pathlib
import sys

import pytest


SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
scaling = importlib.import_module("remote_mpi_scaling")


def _measurements(times: tuple[float, ...]):
    return tuple(
        scaling.ScalingMeasurement(rank_count=rank, wall_seconds=wall)
        for rank, wall in zip(scaling.RANK_LADDER, times)
    )


def test_binding_uses_one_core_and_distributes_even_ranks_across_sockets() -> None:
    arguments = scaling.openmpi_binding_arguments(
        8,
        physical_core_count=76,
        socket_count=2,
    )
    assert arguments == (
        "--nooversubscribe",
        "--bind-to",
        "core",
        "--map-by",
        "ppr:4:socket:PE=1",
        "--report-bindings",
    )
    with pytest.raises(ValueError, match="physical-core"):
        scaling.openmpi_binding_arguments(
            80,
            physical_core_count=76,
            socket_count=2,
        )


def test_launch_uses_only_environment_local_mpiexec() -> None:
    argv = scaling.environment_mpi_argv(
        "/validation/environment",
        rank_count=2,
        physical_core_count=76,
        socket_count=2,
        program=("/validation/environment/bin/python", "worker.py"),
    )
    assert argv[0] == "/validation/environment/bin/mpiexec"
    assert argv[-2:] == (
        "/validation/environment/bin/python",
        "worker.py",
    )
    assert ("-n", "2") == argv[-4:-2]


def test_selection_uses_smallest_rank_within_five_percent_of_fastest() -> None:
    result = scaling.select_rank_count(
        _measurements((100.0, 60.0, 31.0, 30.0, 29.8)),
        physical_core_count=76,
    )
    assert result.fastest_rank_count == 16
    assert result.smallest_near_fastest_rank_count == 4
    assert result.candidate_speedup == pytest.approx(100.0 / 31.0)
    assert result.toy_rank_count == 4
    assert result.silicon_rank_count == 4
    assert result.toy_mpi_enabled


def test_subtwofold_candidate_keeps_toy_serial_but_retains_silicon_mpi() -> None:
    result = scaling.select_rank_count(
        _measurements((100.0, 80.0, 60.0, 55.0, 54.0)),
        physical_core_count=76,
    )
    assert result.smallest_near_fastest_rank_count == 8
    assert result.candidate_speedup == pytest.approx(100.0 / 55.0)
    assert result.toy_rank_count == 1
    assert result.silicon_rank_count == 8
    assert not result.toy_mpi_enabled


def test_scaling_rejects_incomplete_duplicate_or_unequal_evidence() -> None:
    complete = _measurements((100.0, 60.0, 40.0, 30.0, 25.0))
    with pytest.raises(scaling.MPIScalingError, match="incomplete"):
        scaling.select_rank_count(complete[:-1], physical_core_count=76)
    duplicate = complete[:-1] + (complete[-2],)
    with pytest.raises(scaling.MPIScalingError, match="duplicated"):
        scaling.select_rank_count(duplicate, physical_core_count=76)
    unequal = list(complete)
    unequal[2] = scaling.ScalingMeasurement(
        rank_count=4,
        wall_seconds=40.0,
        numerically_equal=False,
    )
    with pytest.raises(scaling.MPIScalingError, match="equality"):
        scaling.select_rank_count(tuple(unequal), physical_core_count=76)


def test_rank_ladder_truncates_without_oversubscription() -> None:
    assert scaling.required_rank_ladder(6) == (1, 2, 4)
    with pytest.raises(ValueError, match="positive"):
        scaling.required_rank_ladder(0)
