"""Create cross-material electron comparison plots for TaN analysis."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from transport import (
    electrical_conductance_per_area_si,
    electronic_thermal_conductance_per_area_si,
    infer_plot_energy_window,
    load_electron_transmission,
    parse_scf_central_cell_vectors,
    temperature_grid,
    transmission_area_m2,
    transmission_per_area_m2,
)


CASE_ORDER = [
    ("TaN", "Pristine TaN"),
    ("4Mg25", "Mg 25%"),
    ("5Mg12.5", "Mg 12.5%"),
    ("6Mg3.7", "Mg 3.7%"),
    ("7N25", "N 25%"),
    ("8N12.5", "N 12.5%"),
    ("9N3.7", "N 3.7%"),
]
DIR_LABELS = {1: "x", 2: "y", 3: "z"}
CASE_STYLES = {
    "TaN": {"color": "black", "linestyle": "-", "linewidth": 2.4},
    "4Mg25": {"color": "red", "linestyle": "-", "linewidth": 2.0},
    "5Mg12.5": {"color": "red", "linestyle": "--", "linewidth": 2.0},
    "6Mg3.7": {"color": "red", "linestyle": ":", "linewidth": 2.4},
    "7N25": {"color": "blue", "linestyle": "-", "linewidth": 2.0},
    "8N12.5": {"color": "blue", "linestyle": "--", "linewidth": 2.0},
    "9N3.7": {"color": "blue", "linestyle": ":", "linewidth": 2.4},
}


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _analysis_dir() -> Path:
    return _workspace_root() / "14. TaN" / "electronic properties" / "analysis"


def _data_root() -> Path:
    return _workspace_root() / "NSCCscratch" / "DeviceStudioProject" / "TaN"


def _load_case_direction(case_name: str, direction: int):
    base = _data_root() / case_name / "Nanodcal-Crystal"
    transmission_dir = base / f"ElectronTransChannel_direction_{direction}"
    data = load_electron_transmission(str(transmission_dir))
    vectors = parse_scf_central_cell_vectors(str(base / "scf.input"))
    per_area = transmission_per_area_m2(data, vectors)
    area_m2 = transmission_area_m2(vectors, data.transmission_direction)
    return data, per_area, area_m2


def _plot_transmission_per_area(direction: int, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x_limits: list[tuple[float, float]] = []
    for case_name, label in CASE_ORDER:
        data, per_area, _ = _load_case_direction(case_name, direction)
        style = CASE_STYLES[case_name]
        ax.plot(
            data.energy_ev,
            per_area,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        x_limits.append(infer_plot_energy_window(data.energy_ev, per_area))
    ax.set_xlabel("Energy relative to Fermi level (eV)")
    ax.set_ylabel(r"Transmission / area (m$^{-2}$)")
    ax.set_title(f"Electron transmission per area, direction {DIR_LABELS[direction]}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(min(x[0] for x in x_limits), max(x[1] for x in x_limits))
    fig.tight_layout()
    fig.savefig(output_dir / f"all_transmission_per_area_{DIR_LABELS[direction]}.png", dpi=220)
    plt.close(fig)


def _plot_conductance_family(direction: int, output_dir: Path) -> None:
    temperatures = temperature_grid(50.0, 400.0, 200)

    fig_g, ax_g = plt.subplots(figsize=(8.0, 5.0))
    fig_k, ax_k = plt.subplots(figsize=(8.0, 5.0))

    for case_name, label in CASE_ORDER:
        data, _, area_m2 = _load_case_direction(case_name, direction)
        style = CASE_STYLES[case_name]
        electrical = np.asarray(
            [
                electrical_conductance_per_area_si(
                    data.energy_ev,
                    data.averaged_transmission,
                    temperature_k=temp,
                    spin_index=data.spin_index,
                    area_m2=area_m2,
                )
                for temp in temperatures
            ],
            dtype=float,
        )
        thermal = np.asarray(
            [
                electronic_thermal_conductance_per_area_si(
                    data.energy_ev,
                    data.averaged_transmission,
                    temperature_k=temp,
                    spin_index=data.spin_index,
                    area_m2=area_m2,
                )
                for temp in temperatures
            ],
            dtype=float,
        )
        ax_g.plot(
            temperatures,
            electrical,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )
        ax_k.plot(
            temperatures,
            thermal,
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
        )

    ax_g.set_xlabel("Temperature (K)")
    ax_g.set_ylabel(r"Electrical conductance / area (S m$^{-2}$)")
    ax_g.set_title(f"Electron electrical conductance per area, direction {DIR_LABELS[direction]}")
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(fontsize=8)
    fig_g.tight_layout()
    fig_g.savefig(output_dir / f"all_electrical_conductance_per_area_{DIR_LABELS[direction]}.png", dpi=220)
    plt.close(fig_g)

    ax_k.set_xlabel("Temperature (K)")
    ax_k.set_ylabel(r"Electronic thermal conductance / area (W K$^{-1}$ m$^{-2}$)")
    ax_k.set_title(f"Electron thermal conductance per area, direction {DIR_LABELS[direction]}")
    ax_k.grid(True, alpha=0.3)
    ax_k.legend(fontsize=8)
    fig_k.tight_layout()
    fig_k.savefig(output_dir / f"all_thermal_conductance_per_area_{DIR_LABELS[direction]}.png", dpi=220)
    plt.close(fig_k)


def main() -> int:
    output_dir = _analysis_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    for direction in (1, 2, 3):
        _plot_transmission_per_area(direction, output_dir)
        _plot_conductance_family(direction, output_dir)

    print(f"Saved analysis plots in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
