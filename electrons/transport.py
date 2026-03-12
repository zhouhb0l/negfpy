"""Electron-only utilities for Nanodcal transmission-channel data."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import constants


ANGSTROM_TO_M = 1.0e-10
EV_TO_J = constants.e
CONDUCTANCE_QUANTUM = constants.e**2 / constants.h


@dataclass(frozen=True)
class ElectronTransmissionData:
    source_path: Path
    transmission_direction: int
    energy_ev: np.ndarray
    averaged_transmission: np.ndarray
    full_transmission: np.ndarray | None
    spin_index: list[str]
    k_grid: tuple[int, ...]
    central_cell_vectors_angstrom: np.ndarray | None


def default_analysis_output_dir(source_path: Path) -> Path:
    workspace_root = Path(__file__).resolve().parents[3]
    analysis_root = workspace_root / "14. TaN" / "electronic properties" / "electrons"

    case_name = source_path.parents[2].name if len(source_path.parents) >= 3 else "unknown_case"
    direction_name = source_path.parent.name
    return analysis_root / case_name / direction_name


def resolve_json_input(path_str: str) -> Path:
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
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _resolve_payload(source: Path) -> tuple[dict[str, Any], Path]:
    data = _load_json(source)
    if "energyPoints" in data and "averagedTransChannelNumbers" in data:
        return data, source
    candidate = source.with_name("TransmissionChannel.json")
    if candidate.is_file():
        payload = _load_json(candidate)
        if "energyPoints" in payload and "averagedTransChannelNumbers" in payload:
            return payload, candidate
    raise ValueError(f"Could not find transmission payload beside {source}")


def load_electron_transmission(path_str: str) -> ElectronTransmissionData:
    source = resolve_json_input(path_str)
    if not source.is_file():
        raise FileNotFoundError(f"Could not find JSON file: {source}")

    payload, payload_source = _resolve_payload(source)
    energy = np.asarray(payload["energyPoints"], dtype=float)
    averaged = np.asarray(payload["averagedTransChannelNumbers"], dtype=float)
    full = payload.get("transChannelNumbers")
    full_array = np.asarray(full, dtype=float) if isinstance(full, list) else None
    vectors = payload.get("centralCellVectors")
    vector_array = np.asarray(vectors, dtype=float) if isinstance(vectors, list) else None

    return ElectronTransmissionData(
        source_path=payload_source,
        transmission_direction=int(payload["transmissionDirection"]),
        energy_ev=energy,
        averaged_transmission=averaged,
        full_transmission=full_array,
        spin_index=[str(item) for item in payload.get("spinIndex", [])],
        k_grid=tuple(int(x) for x in payload.get("kSpaceGridNumber", [])),
        central_cell_vectors_angstrom=vector_array,
    )


def parse_scf_central_cell_vectors(path_str: str) -> np.ndarray:
    path = Path(path_str).expanduser()
    text = path.read_text(encoding="utf-8")
    line_match = re.search(r"system\.centralCellVectors\s*=\s*(.+)", text)
    if not line_match:
        raise ValueError(f"Could not find system.centralCellVectors in {path}")

    line = line_match.group(1)
    groups = re.findall(r"\[([^\[\]]+)\]'", line)
    if len(groups) != 3:
        groups = re.findall(r"\[([^\[\]]+)\]", line)
    if len(groups) < 3:
        raise ValueError(f"Could not parse three cell vectors from {path}")

    vectors = []
    for group in groups[:3]:
        cleaned = group.replace("'", " ").replace("[", " ").replace("]", " ")
        numbers = [float(x) for x in cleaned.split()]
        if len(numbers) != 3:
            raise ValueError(f"Invalid cell vector in {path}: {group}")
        vectors.append(numbers)
    return np.asarray(vectors, dtype=float)


def transmission_area_from_vectors(vectors_angstrom: np.ndarray, direction: int) -> float:
    if vectors_angstrom.shape != (3, 3):
        raise ValueError("Expected central cell vectors with shape (3, 3)")
    if direction not in (1, 2, 3):
        raise ValueError(f"Unsupported transmission direction: {direction}")

    i = direction - 1
    j, k = [idx for idx in range(3) if idx != i]
    cross = np.cross(vectors_angstrom[j], vectors_angstrom[k])
    return float(np.linalg.norm(cross))


def transmission_area_m2(vectors_angstrom: np.ndarray, direction: int) -> float:
    return transmission_area_from_vectors(vectors_angstrom, direction) * (ANGSTROM_TO_M**2)


def transmission_per_area_m2(data: ElectronTransmissionData, vectors_angstrom: np.ndarray) -> np.ndarray:
    area = transmission_area_m2(vectors_angstrom, data.transmission_direction)
    return data.averaged_transmission / area


def infer_plot_energy_window(energy_ev: np.ndarray, transmission: np.ndarray, threshold: float = 1.0e-3) -> tuple[float, float]:
    mask = np.asarray(transmission) > threshold
    if not np.any(mask):
        return float(energy_ev[0]), float(energy_ev[-1])
    active = energy_ev[mask]
    lo = float(active[0])
    hi = float(active[-1])
    pad = max(0.02, 0.1 * max(abs(lo), abs(hi), hi - lo))
    return max(float(energy_ev[0]), lo - pad), min(float(energy_ev[-1]), hi + pad)


def spin_degeneracy_factor(spin_index: list[str]) -> float:
    normalized = [item.lower() for item in spin_index]
    if normalized == ["nospin"]:
        return 2.0
    return float(max(1, len(normalized)))


def _minus_dfde_joule(energy_j: np.ndarray, mu_j: float, temperature_k: float) -> np.ndarray:
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive.")
    x = (energy_j - mu_j) / (2.0 * constants.k * temperature_k)
    return 0.25 / (constants.k * temperature_k) / np.cosh(x) ** 2


def transport_moments(
    energy_ev: np.ndarray,
    transmission: np.ndarray,
    temperature_k: float,
    mu_ev: float = 0.0,
) -> tuple[float, float, float]:
    energy_j = np.asarray(energy_ev, dtype=float) * EV_TO_J
    mu_j = mu_ev * EV_TO_J
    trans = np.asarray(transmission, dtype=float)
    minus_dfde = _minus_dfde_joule(energy_j, mu_j, temperature_k)
    centered = energy_j - mu_j
    l0 = float(np.trapezoid(trans * minus_dfde, energy_j))
    l1 = float(np.trapezoid(trans * centered * minus_dfde, energy_j))
    l2 = float(np.trapezoid(trans * centered**2 * minus_dfde, energy_j))
    return l0, l1, l2


def electrical_conductance_si(
    energy_ev: np.ndarray,
    transmission: np.ndarray,
    temperature_k: float,
    spin_index: list[str],
    mu_ev: float = 0.0,
) -> float:
    l0, _, _ = transport_moments(energy_ev, transmission, temperature_k, mu_ev=mu_ev)
    return spin_degeneracy_factor(spin_index) * CONDUCTANCE_QUANTUM * l0


def electronic_thermal_conductance_si(
    energy_ev: np.ndarray,
    transmission: np.ndarray,
    temperature_k: float,
    spin_index: list[str],
    mu_ev: float = 0.0,
) -> float:
    l0, l1, l2 = transport_moments(energy_ev, transmission, temperature_k, mu_ev=mu_ev)
    if math.isclose(l0, 0.0, abs_tol=1.0e-30):
        return 0.0
    prefactor = spin_degeneracy_factor(spin_index) / (constants.h * temperature_k)
    return prefactor * (l2 - (l1 * l1 / l0))


def electrical_conductance_per_area_si(
    energy_ev: np.ndarray,
    transmission: np.ndarray,
    temperature_k: float,
    spin_index: list[str],
    area_m2: float,
    mu_ev: float = 0.0,
) -> float:
    if area_m2 <= 0.0:
        raise ValueError("Area must be positive.")
    return electrical_conductance_si(
        energy_ev,
        transmission,
        temperature_k=temperature_k,
        spin_index=spin_index,
        mu_ev=mu_ev,
    ) / area_m2


def electronic_thermal_conductance_per_area_si(
    energy_ev: np.ndarray,
    transmission: np.ndarray,
    temperature_k: float,
    spin_index: list[str],
    area_m2: float,
    mu_ev: float = 0.0,
) -> float:
    if area_m2 <= 0.0:
        raise ValueError("Area must be positive.")
    return electronic_thermal_conductance_si(
        energy_ev,
        transmission,
        temperature_k=temperature_k,
        spin_index=spin_index,
        mu_ev=mu_ev,
    ) / area_m2


def temperature_grid(t_min: float, t_max: float, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("Temperature grid requires at least 2 points.")
    if t_min <= 0.0 or t_max <= 0.0 or t_max <= t_min:
        raise ValueError("Require 0 < t_min < t_max.")
    return np.linspace(t_min, t_max, count, dtype=float)
