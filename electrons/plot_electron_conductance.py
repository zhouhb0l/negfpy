"""Plot electron electrical and thermal conductance versus temperature."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from transport import (
    default_analysis_output_dir,
    electrical_conductance_per_area_si,
    electronic_thermal_conductance_per_area_si,
    load_electron_transmission,
    parse_scf_central_cell_vectors,
    temperature_grid,
    transmission_area_from_vectors,
    transmission_area_m2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot electron conductance versus temperature.")
    parser.add_argument("input_path", help="Path to CalculatedResults.json or ElectronTransChannel_direction_* folder.")
    parser.add_argument("--mu-ev", type=float, default=0.0, help="Chemical potential relative to the energy grid zero (eV).")
    parser.add_argument("--t-min", type=float, default=50.0, help="Minimum temperature in K.")
    parser.add_argument("--t-max", type=float, default=400.0, help="Maximum temperature in K.")
    parser.add_argument("--n-temp", type=int, default=200, help="Number of temperature points.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output figures. Defaults to 14. TaN/electronic properties/electrons/<case>/<direction>.",
    )
    args = parser.parse_args()

    data = load_electron_transmission(args.input_path)
    direction_dir = data.source_path.parent
    scf_path = direction_dir.parent / "scf.input"
    vectors = parse_scf_central_cell_vectors(str(scf_path))
    area_a2 = transmission_area_from_vectors(vectors, data.transmission_direction)
    area_m2 = transmission_area_m2(vectors, data.transmission_direction)
    temperatures = temperature_grid(args.t_min, args.t_max, args.n_temp)
    electrical = np.asarray(
        [
            electrical_conductance_per_area_si(
                data.energy_ev,
                data.averaged_transmission,
                temperature_k=t,
                spin_index=data.spin_index,
                area_m2=area_m2,
                mu_ev=args.mu_ev,
            )
            for t in temperatures
        ],
        dtype=float,
    )
    thermal = np.asarray(
        [
            electronic_thermal_conductance_per_area_si(
                data.energy_ev,
                data.averaged_transmission,
                temperature_k=t,
                spin_index=data.spin_index,
                area_m2=area_m2,
                mu_ev=args.mu_ev,
            )
            for t in temperatures
        ],
        dtype=float,
    )

    output_dir = Path(args.output_dir) if args.output_dir else default_analysis_output_dir(data.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(temperatures, electrical, lw=2.0)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Electrical conductance / area (S m$^{-2}$)")
    ax.set_title(
        f"Electron electrical conductance per area, direction {data.transmission_direction}\n"
        f"Area = {area_a2:.6f} A^2"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_electrical_conductance_per_area_vs_temperature.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(temperatures, thermal, lw=2.0)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel(r"Electronic thermal conductance / area (W K$^{-1}$ m$^{-2}$)")
    ax.set_title(
        f"Electron thermal conductance per area, direction {data.transmission_direction}\n"
        f"Area = {area_a2:.6f} A^2"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_thermal_conductance_per_area_vs_temperature.png", dpi=200)
    plt.close(fig)

    print(f"Saved: {output_dir / 'electron_electrical_conductance_per_area_vs_temperature.png'}")
    print(f"Saved: {output_dir / 'electron_thermal_conductance_per_area_vs_temperature.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
