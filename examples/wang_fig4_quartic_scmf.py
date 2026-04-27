"""Run Wang-style quartic SCMF benchmark sweeps for one- and two-particle models.

This script targets the literature structure of Wang review Fig. 4:
- quartic one- and two-particle models
- Lorentz-Drude baths
- symmetric temperature protocol T_L = 1.25 T, T_R = 0.75 T

The current implementation uses the published coefficients as raw model-unit
inputs for the paper-native path. The heuristic SI path is useful for checking
temperature/current scales, but it is still under audit for exact unit mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.inelastic import (
    quartic_scmf_current_vs_temperature,
    wang_review_fig4_audit,
    wang_review_fig4_one_particle_cases,
    wang_review_fig4_one_particle_cases_si,
    wang_review_fig4_two_particle_cases,
    wang_review_fig4_two_particle_cases_si,
    wang_review_hbar_effective_si,
    wang_review_kb_effective_si,
)


def _next_run_index(output_dir: Path) -> int:
    used: set[int] = set()
    for path in output_dir.glob("*"):
        stem = path.stem
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            used.add(int(prefix))
    idx = 1
    while idx in used:
        idx += 1
    return idx


def _write_case_table(path: Path, result: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        has_si = "current_watts" in result
        if has_si:
            fh.write("# T_avg_K\tT_left_K\tT_right_K\tcurrent_W\tcurrent_nW\tconverged\titerations\tresidual\n")
        else:
            fh.write("# T_avg\tT_left\tT_right\tcurrent_internal\tconverged\titerations\tresidual\n")
        for i, temp in enumerate(result["temperature_avg"]):
            if has_si:
                fh.write(
                    f"{float(temp):.8f}\t"
                    f"{float(result['temperature_left'][i]):.8f}\t"
                    f"{float(result['temperature_right'][i]):.8f}\t"
                    f"{float(result['current_watts'][i]):.12e}\t"
                    f"{float(result['current_nw'][i]):.12e}\t"
                    f"{int(result['converged'][i])}\t"
                    f"{int(result['iterations'][i])}\t"
                    f"{float(result['residual'][i]):.12e}\n"
                )
            else:
                fh.write(
                    f"{float(temp):.8f}\t"
                    f"{float(result['temperature_left'][i]):.8f}\t"
                    f"{float(result['temperature_right'][i]):.8f}\t"
                    f"{float(result['current_internal'][i]):.12e}\t"
                    f"{int(result['converged'][i])}\t"
                    f"{int(result['iterations'][i])}\t"
                    f"{float(result['residual'][i]):.12e}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-min", type=float, default=50.0)
    parser.add_argument("--temp-max", type=float, default=1200.0)
    parser.add_argument("--n-temp", type=int, default=41)
    parser.add_argument("--omega-max", type=float, default=2.0e13)
    parser.add_argument("--n-omega", type=int, default=1201)
    parser.add_argument("--n-omega-cov", type=int, default=1201)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--kb-effective", type=float, default=0.0)
    parser.add_argument("--hbar-effective", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--unit-system", choices=("si", "raw"), default="si")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/inelastic/Wang7317"),
    )
    args = parser.parse_args()

    temps = np.linspace(float(args.temp_min), float(args.temp_max), int(args.n_temp))
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    run_idx = _next_run_index(outdir)
    prefix = f"{run_idx:03d}"

    use_si = str(args.unit_system) == "si"
    family_builders = (
        {
            "one_particle": wang_review_fig4_one_particle_cases_si,
            "two_particle": wang_review_fig4_two_particle_cases_si,
        }
        if use_si
        else {
            "one_particle": wang_review_fig4_one_particle_cases,
            "two_particle": wang_review_fig4_two_particle_cases,
        }
    )
    kb_effective = float(args.kb_effective) if float(args.kb_effective) > 0.0 else (
        wang_review_kb_effective_si() if use_si else 1.0
    )
    hbar_effective = float(args.hbar_effective) if float(args.hbar_effective) > 0.0 else (
        wang_review_hbar_effective_si() if use_si else 1.0
    )
    colors = {"black": "#111111", "red": "#c62828", "blue": "#1565c0"}
    all_meta: dict[str, object] = {
        "benchmark": "Wang Fig. 4 quartic SCMF benchmark sweep",
        "temperature_protocol": {"T_left": "1.25 * T_avg", "T_right": "0.75 * T_avg"},
        "unit_system": str(args.unit_system),
        "raw_unit_note": (
            "For unit_system=raw, the published coefficients are used in the paper's native units. "
            "For unit_system=si, the coefficients are converted through a heuristic SI mapping that remains under audit."
        ),
        "audit": wang_review_fig4_audit(),
        "parameters": {
            "temp_min": float(args.temp_min),
            "temp_max": float(args.temp_max),
            "n_temp": int(args.n_temp),
            "omega_max": float(args.omega_max),
            "n_omega": int(args.n_omega),
            "n_omega_cov": int(args.n_omega_cov),
            "eta": float(args.eta),
            "kb_effective": float(kb_effective),
            "hbar_effective": float(hbar_effective),
            "max_iter": int(args.max_iter),
            "mixing": float(args.mixing),
            "tol": float(args.tol),
        },
        "families": {},
    }

    for family_name, builder in family_builders.items():
        cases = builder()
        family_meta: dict[str, object] = {}

        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        for color_name in ("black", "red", "blue"):
            case = cases[color_name]
            result = quartic_scmf_current_vs_temperature(
                case,
                temps,
                omega_max=float(args.omega_max),
                n_omega=int(args.n_omega),
                n_omega_cov=int(args.n_omega_cov),
                eta=float(args.eta),
                kb_effective=float(kb_effective),
                hbar_effective=float(hbar_effective),
                max_iter=int(args.max_iter),
                mixing=float(args.mixing),
                tol=float(args.tol),
            )
            if use_si:
                result["current_watts"] = np.asarray(result["current_internal"], dtype=float)
                result["current_nw"] = 1.0e9 * np.asarray(result["current_watts"], dtype=float)
            table_name = f"{prefix}_{family_name}_{color_name}_current_vs_temperature.tsv"
            _write_case_table(outdir / table_name, result)

            ax.plot(
                result["temperature_avg"],
                result["current_nw"] if use_si else result["current_internal"],
                color=colors[color_name],
                lw=2.0,
                label=color_name,
            )

            family_meta[color_name] = {
                "label": str(result["label"]),
                "table": table_name,
                "literature_note": str(result["literature_note"]),
                "converged_fraction": float(np.mean(result["converged"])),
                "iterations_max": int(np.max(result["iterations"])),
                "residual_max": float(np.max(result["residual"])),
            }

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Heat Current (nW)" if use_si else "Heat Current (internal units)")
        ax.set_title(f"Wang Fig. 4 Target: {family_name.replace('_', ' ').title()}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"{prefix}_{family_name}_current_vs_temperature.png", dpi=220)
        all_meta["families"][family_name] = family_meta

    all_meta["run_prefix"] = prefix
    (outdir / f"{prefix}_benchmark_meta.json").write_text(json.dumps(all_meta, indent=2), encoding="utf-8")
    print(f"Saved Wang-target quartic SCMF benchmark outputs to {outdir}")


if __name__ == "__main__":
    main()
