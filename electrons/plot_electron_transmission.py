"""Plot electron transmission and transmission-per-area from Nanodcal outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from transport import (
    default_analysis_output_dir,
    infer_plot_energy_window,
    load_electron_transmission,
    parse_scf_central_cell_vectors,
    transmission_area_from_vectors,
    transmission_per_area_m2,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot electron transmission and transmission per area.")
    parser.add_argument("input_path", help="Path to CalculatedResults.json or ElectronTransChannel_direction_* folder.")
    parser.add_argument(
        "--scf",
        dest="scf_path",
        help="Path to scf.input. If omitted, use ../scf.input relative to the direction folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output figures. Defaults to 14. TaN/electronic properties/electrons/<case>/<direction>.",
    )
    args = parser.parse_args()

    data = load_electron_transmission(args.input_path)
    direction_dir = data.source_path.parent
    scf_path = Path(args.scf_path) if args.scf_path else direction_dir.parent / "scf.input"
    vectors = parse_scf_central_cell_vectors(str(scf_path))
    area_a2 = transmission_area_from_vectors(vectors, data.transmission_direction)
    transmission_area = transmission_per_area_m2(data, vectors)
    xlim = infer_plot_energy_window(data.energy_ev, data.averaged_transmission)

    output_dir = Path(args.output_dir) if args.output_dir else default_analysis_output_dir(data.source_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(data.energy_ev, data.averaged_transmission, lw=2.0)
    ax.set_xlabel("Energy relative to Fermi level (eV)")
    ax.set_ylabel("Transmission")
    ax.set_title(f"Electron transmission, direction {data.transmission_direction}")
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_transmission.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(data.energy_ev, transmission_area, lw=2.0)
    ax.set_xlabel("Energy relative to Fermi level (eV)")
    ax.set_ylabel(r"Transmission / area (m$^{-2}$)")
    ax.set_title(
        f"Electron transmission per area, direction {data.transmission_direction}\n"
        f"Area = {area_a2:.6f} A^2"
    )
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "electron_transmission_per_area.png", dpi=200)
    plt.close(fig)

    print(f"Saved: {output_dir / 'electron_transmission.png'}")
    print(f"Saved: {output_dir / 'electron_transmission_per_area.png'}")
    print(f"Direction: {data.transmission_direction}")
    print(f"Area (A^2): {area_a2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
