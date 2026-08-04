#!/usr/bin/env python3
"""Create a non-mutating convergence diagnostic from an SCBA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _json_dataset(handle: h5py.File, name: str) -> dict[str, Any]:
    value = handle[name][()]
    if isinstance(value, bytes):
        return json.loads(value.decode("ascii"))
    return json.loads(bytes(np.asarray(value, dtype=np.uint8)).decode("ascii"))


def analyze(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        status = str(handle.attrs["status"])
        history = np.asarray(handle["iteration_history"][...])
        identity = _json_dataset(handle, "identity_json")
        state = _json_dataset(handle, "solver_state_json")
    if history.size == 0:
        raise ValueError("checkpoint has no iteration history")
    sigma = np.asarray(history["sigma_residual"], dtype=float)
    green = np.asarray(history["green_residual"], dtype=float)
    identity_residual = np.asarray(history["identity_residual"], dtype=float)
    mixing = np.asarray(history["mixing"], dtype=float)
    floor = float(np.min(mixing))
    floor_iteration = int(history["iteration"][np.flatnonzero(mixing == floor)[0]])
    return {
        "schema": "negfpy.sparse-nq1-scba-history-analysis.v1",
        "checkpoint": str(path.resolve()),
        "checkpoint_status": status,
        "identity_sha256": identity.get("identity_sha256", identity["problem_sha256"]),
        "configuration_sha256": identity["configuration_sha256"],
        "problem_hashes": identity["problem_hashes"],
        "iterations": int(history.size),
        "last": {
            "sigma_residual": float(sigma[-1]),
            "green_residual": float(green[-1]),
            "identity_residual": float(identity_residual[-1]),
            "mixing": float(mixing[-1]),
            "consecutive_passes": int(history["consecutive_passes"][-1]),
        },
        "ranges": {
            "sigma_residual_min": float(np.nanmin(sigma)),
            "sigma_residual_max": float(np.nanmax(sigma)),
            "green_residual_min": float(np.nanmin(green)),
            "green_residual_max": float(np.nanmax(green)),
            "identity_residual_max": float(np.nanmax(identity_residual)),
        },
        "mixing_floor": floor,
        "mixing_floor_first_iteration": floor_iteration,
        "adaptive_mixing_used": bool(np.any(np.diff(mixing) < 0.0)),
        "continuation_state": state,
        "next_action": (
            "Do not rerun this terminal checkpoint; use a converged lower-coupling "
            "parent and a refined frequency grid before increasing coupling."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = analyze(args.checkpoint)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
