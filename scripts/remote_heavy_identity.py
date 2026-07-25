"""Immutable deploy-time versus attempt-time environment verification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


IMMUTABLE_FIELDS = (
    "schema",
    "commit",
    "explicit_lock_sha256",
    "tree",
    "hostname",
    "platform",
    "machine",
    "python_executable",
    "python_version",
    "distributions",
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
