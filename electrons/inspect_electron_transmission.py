"""Inspect Nanodcal electron transmission data from CalculatedResults.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _resolve_input(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_dir():
        candidate = path / "CalculatedResults.json"
        if candidate.is_file():
            return candidate
    return path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object at the top level.")
    return data


def _resolve_payload(data: dict[str, Any], source: Path) -> tuple[dict[str, Any], Path]:
    if "energyPoints" in data or "averagedTransChannelNumbers" in data:
        return data, source
    candidate = source.with_name("TransmissionChannel.json")
    if candidate.is_file():
        payload = _load_json(candidate)
        if "energyPoints" in payload or "averagedTransChannelNumbers" in payload:
            return payload, candidate
    return data, source


def _shape_2d_or_3d(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    if not isinstance(values[0], list):
        return (len(values),)
    first = values[0]
    if not first:
        return (len(values), 0)
    if not isinstance(first[0], list):
        return (len(values), len(first))
    return (len(values), len(first), len(first[0]))


def build_summary(data: dict[str, Any], source: Path) -> str:
    energy_points = data.get("energyPoints", [])
    averaged = data.get("averagedTransChannelNumbers", [])
    trans = data.get("transChannelNumbers", [])
    spin_index = data.get("spinIndex", [])
    k_grid = data.get("kSpaceGridNumber", [])
    direction = data.get("transmissionDirection")

    lines = [
        f"source: {source}",
        f"transmission_direction: {direction}",
        f"n_energy_points: {len(energy_points) if isinstance(energy_points, list) else 'invalid'}",
        f"energy_min: {energy_points[0] if isinstance(energy_points, list) and energy_points else 'n/a'}",
        f"energy_max: {energy_points[-1] if isinstance(energy_points, list) and energy_points else 'n/a'}",
        f"spin_channels: {len(spin_index) if isinstance(spin_index, list) else 'invalid'}",
        f"spin_index: {spin_index}",
        f"k_grid: {k_grid}",
        f"averaged_shape: {_shape_2d_or_3d(averaged)}",
        f"full_transmission_shape: {_shape_2d_or_3d(trans)}",
    ]

    if isinstance(averaged, list) and averaged:
        lines.append(f"first_averaged_value: {averaged[0]}")
        lines.append(f"last_averaged_value: {averaged[-1]}")

    description = data.get("description")
    if isinstance(description, str):
        first_line = description.strip().splitlines()[0] if description.strip() else ""
        lines.append(f"description: {first_line}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect electron transmission data stored in Nanodcal CalculatedResults.json."
    )
    parser.add_argument(
        "input_path",
        help="Path to CalculatedResults.json or to an ElectronTransChannel_direction_* folder.",
    )
    args = parser.parse_args()

    source = _resolve_input(args.input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Could not find JSON file: {source}")

    data = _load_json(source)
    payload, payload_source = _resolve_payload(data, source)
    print(build_summary(payload, payload_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
