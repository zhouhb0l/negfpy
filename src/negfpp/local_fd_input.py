"""Strict inputs for the existing frozen-kernel local-FD extension."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .input import _number, _object, sha256_file


SECTIONS = {
    "frozen_source": {"manifest", "sha256"},
    "geometry": {"center_layers"},
    "reservoirs": {"left_temperature_k", "right_temperature_k", "equilibrium_temperature_k"},
    "resources": {"workers", "frequency_tile", "max_cache_bytes", "numeric_cap_bytes",
                  "parent_reserve_bytes", "runtime_reserve_bytes", "available_margin_bytes",
                  "max_seconds", "exit_reserve_seconds", "harmonic_max_bytes", "contact_max_bytes"},
    "solver": {"max_newton_steps", "max_krylov_iterations", "krylov_restart",
               "max_line_search_steps", "max_current_calls", "max_jacobian_calls",
               "max_seconds", "preconditioner_seconds"},
    "run": {"name", "output_dir"},
}


@dataclass(frozen=True)
class LocalFDInput:
    source: Path
    document: dict
    manifest: Path
    donor: object
    output: Path


def load_local_fd_input(source, doc):
    """Authenticate the compact donor and controls without long allocation."""
    _object(doc, set(SECTIONS) | {"schema"}, "input")
    for name, keys in SECTIONS.items():
        _object(doc[name], keys, name)
    _number(doc["geometry"]["center_layers"], "center_layers", strict=True, integer=True)
    for key, value in doc["reservoirs"].items():
        _number(value, key, strict=True)
    for section in ("resources", "solver"):
        for key, value in doc[section].items():
            integer = not key.endswith("seconds")
            _number(value, section + "." + key, strict=True, integer=integer)
    resource = doc["resources"]
    if resource["exit_reserve_seconds"] >= resource["max_seconds"]:
        raise ValueError("exit_reserve_seconds must be less than max_seconds")
    if resource["workers"] > 16:
        raise ValueError("The local-FD interface permits at most 16 CPU workers")
    from .local_fd.temperature_solver import SolverOptions
    controls = {k:v for k,v in doc["solver"].items() if k != "preconditioner_seconds"}
    SolverOptions(**controls)  # Retain the original physical tolerances.
    spec = doc["frozen_source"]
    if not isinstance(spec["manifest"], str) or not spec["manifest"]:
        raise ValueError("frozen_source.manifest must name a compact source manifest")
    if not isinstance(spec["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", spec["sha256"]):
        raise ValueError("frozen_source.sha256 must be a SHA-256 value")
    manifest = (source.parent / spec["manifest"]).resolve(strict=True)
    if not manifest.is_file() or sha256_file(manifest) != spec["sha256"]:
        raise ValueError("Frozen-source manifest hash mismatch")
    from .local_fd.source import load_frozen_source
    donor = load_frozen_source(manifest, expected_sha256=spec["sha256"])
    name = doc["run"]["name"]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise ValueError("run.name must be a short filename-safe identifier")
    output = doc["run"]["output_dir"]
    if not isinstance(output, str) or not output:
        raise ValueError("run.output_dir must be a nonempty path")
    return LocalFDInput(source, doc, manifest, donor, (source.parent / output).resolve())
