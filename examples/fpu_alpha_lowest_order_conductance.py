"""Plot review-style thermal conductance vs temperature for the FPU-alpha toy model.

This is a clean benchmark driver for the current cubic lowest-order inelastic
path. It follows the review-style symmetric temperature bias convention
T_L = 1.25 T_avg and T_R = 0.75 T_avg by default, while keeping all generated
artifacts inside outputs/inelastic/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.inelastic import FPUAlphaParams, fpu_alpha_lowest_order_conductance_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--spring", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--temp-min", type=float, default=50.0)
    parser.add_argument("--temp-max", type=float, default=500.0)
    parser.add_argument("--n-temp", type=int, default=16)
    parser.add_argument("--n-omega", type=int, default=200)
    parser.add_argument("--omega-max-factor", type=float, default=1.05)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--broadening", type=float, default=0.03)
    parser.add_argument("--bias-fraction", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inelastic/fpu_alpha_lowest_order_review_style"),
    )
    args = parser.parse_args()

    temps = np.linspace(float(args.temp_min), float(args.temp_max), int(args.n_temp))
    params = FPUAlphaParams(
        mass=float(args.mass),
        spring=float(args.spring),
        alpha=float(args.alpha),
    )
    result = fpu_alpha_lowest_order_conductance_sweep(
        temperatures=temps,
        n_layers=int(args.n_layers),
        params=params,
        n_omega=int(args.n_omega),
        omega_max_factor=float(args.omega_max_factor),
        eta=float(args.eta),
        broadening=float(args.broadening),
        bias_fraction=float(args.bias_fraction),
    )

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    table_path = outdir / "conductance_vs_temperature.tsv"
    with table_path.open("w", encoding="utf-8") as fh:
        fh.write("# T_avg_K\tT_left_K\tT_right_K\theat_current_W\tconductance_W_per_K\n")
        for i, temp in enumerate(result["temperature_avg"]):
            fh.write(
                f"{float(temp):.8f}\t"
                f"{float(result['temperature_left'][i]):.8f}\t"
                f"{float(result['temperature_right'][i]):.8f}\t"
                f"{float(result['heat_current'][i]):.12e}\t"
                f"{float(result['conductance'][i]):.12e}\n"
            )

    spectra_path = outdir / "transmission_vs_temperature.tsv"
    with spectra_path.open("w", encoding="utf-8") as fh:
        header = "\t".join(f"T_{float(t):.3f}K" for t in result["temperature_avg"])
        fh.write(f"# omega_rad_s\t{header}\n")
        for iw, omega in enumerate(result["omegas"]):
            cols = "\t".join(f"{float(result['transmission_map'][it, iw]):.12e}" for it in range(len(result["temperature_avg"])))
            fh.write(f"{float(omega):.12e}\t{cols}\n")

    meta = {
        "model": "FPU-alpha lowest-order cubic self-energy",
        "benchmark_style": "review-style symmetric temperature bias",
        "temperature_protocol": {
            "T_left": f"(1 + {float(args.bias_fraction):.6f}) * T_avg",
            "T_right": f"(1 - {float(args.bias_fraction):.6f}) * T_avg",
        },
        "parameters": {
            "n_layers": int(args.n_layers),
            "mass": float(args.mass),
            "spring": float(args.spring),
            "alpha": float(args.alpha),
            "eta": float(args.eta),
            "broadening": float(args.broadening),
            "omega_max_factor": float(args.omega_max_factor),
            "n_omega": int(args.n_omega),
            "temp_min": float(args.temp_min),
            "temp_max": float(args.temp_max),
            "n_temp": int(args.n_temp),
        },
        "literature_note": (
            "This script uses the review-style temperature-bias protocol as a clean benchmark "
            "driver for the current cubic lowest-order implementation. A direct reproduction "
            "of Wang's quartic SCMF figure requires the dedicated fourth-order mean-field path."
        ),
    }
    meta_path = outdir / "benchmark_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(result["temperature_avg"], result["conductance"], color="tab:blue", lw=2.0)
    ax.set_xlabel("Average Temperature (K)")
    ax.set_ylabel("Thermal Conductance (W/K)")
    ax.set_title("FPU-alpha Lowest-Order Inelastic Benchmark")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "conductance_vs_temperature.png", dpi=220)

    fig2, ax2 = plt.subplots(figsize=(7.2, 4.8))
    for idx in [0, len(result["temperature_avg"]) // 2, len(result["temperature_avg"]) - 1]:
        ax2.plot(
            result["omegas"],
            result["transmission_map"][idx],
            lw=1.6,
            label=f"{float(result['temperature_avg'][idx]):.0f} K",
        )
    ax2.set_xlabel(r"$\omega$ (rad/s)")
    ax2.set_ylabel(r"$T(\omega)$")
    ax2.set_title("Representative Inelastic Transmission Spectra")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / "transmission_spectra.png", dpi=220)

    print(f"Saved benchmark outputs to {outdir}")


if __name__ == "__main__":
    main()
