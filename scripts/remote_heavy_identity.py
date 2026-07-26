"""Immutable deploy-time versus attempt-time environment verification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


IMMUTABLE_FIELDS = (
    "schema",
    "commit",
    "source_tree_sha256",
    "explicit_lock_sha256",
    "tree",
    "hostname",
    "platform",
    "machine",
    "affinity_logical_cpu_count",
    "physical_core_count",
    "socket_count",
    "python_executable",
    "python_version",
    "distributions",
    "mpi",
    "package_source",
    "thread_limits",
)


def compare_environment_identity(
    stored: Mapping[str, Any],
    current: Mapping[str, Any],
) -> None:
    mismatches = [
        field
        for field in IMMUTABLE_FIELDS
        if stored.get(field) != current.get(field)
    ]
    if mismatches:
        raise ValueError(
            "attempt environment differs from immutable deployment: "
            + ", ".join(mismatches)
        )
