"""Compare quartic toy-model conductance across LO, MF, and SCBA approximations.

This benchmark keeps all artifacts inside ``outputs/inelastic/`` and follows the
Wang-style symmetric temperature protocol

    T_L = (1 + bias_fraction) T_avg
    T_R = (1 - bias_fraction) T_avg

for clean comparison between quartic lowest-order, mean-field, and SCBA-like
closures inside the current codebase.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.inelastic import Phi4Params, phi4_compare_conductance_sweeps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--spring", type=float, default=1.0)
    parser.add_argument("--lambda4", type=float, default=0.08)
    parser.add_argument("--temp-min", type=float, default=50.0)
    parser.add_argument("--temp-max", type=float, default=500.0)
    parser.add_argument("--n-temp", type=int, default=16)
    parser.add_argument("--n-omega", type=int, default=200)
    parser.add_argument("--omega-max-factor", type=float, default=1.05)
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument("--bias-fraction", type=float, default=0.25)
    parser.add_argument("--broadening", type=float, default=0.03)
    parser.add_argument("--frequency-floor", type=float, default=1e-8)
    parser.add_argument("--mf-max-iter", type=int, default=100)
    parser.add_argument("--mf-mixing", type=float, default=0.5)
    parser.add_argument("--mf-tol", type=float, default=1e-8)
    parser.add_argument("--scba-max-iter", type=int, default=20)
    parser.add_argument("--scba-mixing", type=float, default=0.5)
    parser.add_argument("--scba-tol", type=float, default=1e-5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inelastic/phi4_approximations_review_style"),
    )
    args = parser.parse_args()

    temps = np.linspace(float(args.temp_min), float(args.temp_max), int(args.n_temp))
    params = Phi4Params(
        mass=float(args.mass),
        spring=float(args.spring),
        lambda4=float(args.lambda4),
    )
    results = phi4_compare_conductance_sweeps(
        temperatures=temps,
        n_layers=int(args.n_layers),
        params=params,
        n_omega=int(args.n_omega),
        omega_max_factor=float(args.omega_max_factor),
        eta=float(args.eta),
        bias_fraction=float(args.bias_fraction),
        broadening=float(args.broadening),
        frequency_floor=float(args.frequency_floor),
        mean_field_max_iter=int(args.mf_max_iter),
        mean_field_mixing=float(args.mf_mixing),
        mean_field_tol=float(args.mf_tol),
        transport_max_iter=int(args.scba_max_iter),
        transport_mixing=float(args.scba_mixing),
        transport_tol=float(args.scba_tol),
    )

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    combined_path = outdir / "conductance_vs_temperature.tsv"
    approx_order = ("lowest_order", "mean_field", "scba")
    with combined_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# T_avg_K\tT_left_K\tT_right_K\t"
            "G_lo_W_per_K\tG_mf_W_per_K\tG_scba_W_per_K\t"
            "J_lo_W\tJ_mf_W\tJ_scba_W\t"
            "conv_lo\tconv_mf\tconv_scba\n"
        )
        ref = results[approx_order[0]]
        for i, temp in enumerate(ref["temperature_avg"]):
            fh.write(
                f"{float(temp):.8f}\t"
                f"{float(ref['temperature_left'][i]):.8f}\t"
                f"{float(ref['temperature_right'][i]):.8f}\t"
                f"{float(results['lowest_order']['conductance'][i]):.12e}\t"
                f"{float(results['mean_field']['conductance'][i]):.12e}\t"
                f"{float(results['scba']['conductance'][i]):.12e}\t"
                f"{float(results['lowest_order']['heat_current'][i]):.12e}\t"
                f"{float(results['mean_field']['heat_current'][i]):.12e}\t"
                f"{float(results['scba']['heat_current'][i]):.12e}\t"
                f"{int(results['lowest_order']['converged_all'][i])}\t"
                f"{int(results['mean_field']['converged_all'][i])}\t"
                f"{int(results['scba']['converged_all'][i])}\n"
            )

    rep_idx = len(temps) // 2
    spectra_path = outdir / "representative_transmission.tsv"
    with spectra_path.open("w", encoding="utf-8") as fh:
        fh.write("# omega_rad_s\tT_lowest_order\tT_mean_field\tT_scba\n")
        for iw, omega in enumerate(results["lowest_order"]["omegas"]):
            fh.write(
                f"{float(omega):.12e}\t"
                f"{float(results['lowest_order']['transmission_map'][rep_idx, iw]):.12e}\t"
                f"{float(results['mean_field']['transmission_map'][rep_idx, iw]):.12e}\t"
                f"{float(results['scba']['transmission_map'][rep_idx, iw]):.12e}\n"
            )

    meta = {
        "model": "quartic onsite phi^4 chain",
        "approximations": list(approx_order),
        "benchmark_style": "review-style symmetric temperature bias",
        "temperature_protocol": {
            "T_left": f"(1 + {float(args.bias_fraction):.6f}) * T_avg",
            "T_right": f"(1 - {float(args.bias_fraction):.6f}) * T_avg",
        },
        "parameters": {
            "n_layers": int(args.n_layers),
            "mass": float(args.mass),
            "spring": float(args.spring),
            "lambda4": float(args.lambda4),
            "eta": float(args.eta),
            "broadening": float(args.broadening),
            "frequency_floor": float(args.frequency_floor),
            "omega_max_factor": float(args.omega_max_factor),
            "n_omega": int(args.n_omega),
            "temp_min": float(args.temp_min),
            "temp_max": float(args.temp_max),
            "n_temp": int(args.n_temp),
            "mf_max_iter": int(args.mf_max_iter),
            "mf_mixing": float(args.mf_mixing),
            "mf_tol": float(args.mf_tol),
            "scba_max_iter": int(args.scba_max_iter),
            "scba_mixing": float(args.scba_mixing),
            "scba_tol": float(args.scba_tol),
        },
        "literature_note": (
            "This benchmark follows the Wang-style symmetric temperature protocol and compares "
            "quartic LO, MF, and SCBA-like closures within the current negfpy chain-lead setup. "
            "It is a clean internal benchmark, not yet an exact reproduction of the Lorentz-Drude "
            "bath figures in Wang's review and 2007 transport paper."
        ),
        "literature_files": [
            "literature/1303.7317v1.pdf",
            "literature/0701164v1.pdf",
        ],
        "representative_temperature_K": float(temps[rep_idx]),
    }
    (outdir / "benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    colors = {
        "lowest_order": "#1b5e20",
        "mean_field": "#1565c0",
        "scba": "#b71c1c",
    }
    labels = {
        "lowest_order": "LO",
        "mean_field": "MF",
        "scba": "SCBA",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for name in approx_order:
        ax.plot(
            results[name]["temperature_avg"],
            results[name]["conductance"],
            lw=2.0,
            color=colors[name],
            label=labels[name],
        )
    ax.set_xlabel("Average Temperature (K)")
    ax.set_ylabel("Thermal Conductance (W/K)")
    ax.set_title(r"Quartic $\phi^4$ Conductance Comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "conductance_vs_temperature.png", dpi=220)

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.8))
    for name in approx_order:
        ax2.plot(
            results[name]["omegas"],
            results[name]["transmission_map"][rep_idx],
            lw=1.8,
            color=colors[name],
            label=labels[name],
        )
    ax2.set_xlabel(r"$\omega$ (rad/s)")
    ax2.set_ylabel(r"$T(\omega)$")
    ax2.set_title(f"Representative Transmission at {float(temps[rep_idx]):.0f} K")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / "representative_transmission.png", dpi=220)

    print(f"Saved quartic approximation benchmark outputs to {outdir}")


if __name__ == "__main__":
    main()
