"""Run the Wang 0701164 Fig. 5 cubic-onsite LO benchmark.

This example keeps all outputs in a single folder with ordered prefixes, so we
can compare runs cleanly while iterating toward the published benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.inelastic import (
    wang0701164_compare_sweep_to_digitized,
    wang0701164_compare_sweep_to_vector_pdf,
    wang0701164_extract_fig5_vector_curves,
    wang0701164_fig5_lowest_order_conductance_sweep,
    wang0701164_fig5_spec,
    wang0701164_load_digitized_fig5_curves,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-min", type=float, default=50.0)
    parser.add_argument("--temp-max", type=float, default=2000.0)
    parser.add_argument("--n-temp", type=int, default=21)
    parser.add_argument("--n-omega", type=int, default=121)
    parser.add_argument("--omega-max", type=float, default=2.3)
    parser.add_argument("--internal-oversample", type=int, default=1)
    parser.add_argument("--delta-t", type=float, default=0.02)
    parser.add_argument("--eta", type=float, default=2e-5)
    parser.add_argument("--unit-system", choices=("paper", "si"), default="paper")
    parser.add_argument(
        "--conductance-mode",
        choices=("effective_transmission", "current_over_delta_t", "paper_derivative", "paper_derivative_numeric"),
        default="paper_derivative_numeric",
    )
    parser.add_argument("--include-second-graph", action="store_true")
    parser.add_argument(
        "--compare-reference",
        choices=("none", "digitized", "vector"),
        default="vector",
    )
    parser.add_argument(
        "--digitized-path",
        type=Path,
        default=Path("outputs/inelastic/Wang0701164/002_fig5_digitized_curves.tsv"),
    )
    parser.add_argument("--reference-pdf", type=Path, default=Path("literature/0701164v1.pdf"))
    parser.add_argument("--reference-page", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inelastic/Wang0701164"))
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_next_run_index(outdir):03d}"
    temps = np.linspace(float(args.temp_min), float(args.temp_max), int(args.n_temp))
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        temps,
        unit_system=str(args.unit_system),
        n_omega=int(args.n_omega),
        omega_max=None if args.omega_max is None else float(args.omega_max),
        internal_oversample=int(args.internal_oversample),
        delta_t=float(args.delta_t),
        eta=float(args.eta),
        conductance_mode=str(args.conductance_mode),
        include_second_graph=bool(args.include_second_graph),
    )

    table_path = outdir / f"{prefix}_fig5_conductance_vs_temperature.tsv"
    with table_path.open("w", encoding="utf-8") as fh:
        fh.write("# T")
        for label in result["curves"]:
            fh.write(f"\tkappa_t_{label}")
        fh.write("\n")
        for i, temp in enumerate(result["temperatures"]):
            fh.write(f"{float(temp):.8f}")
            for label in result["curves"]:
                fh.write(f"\t{float(result['curves'][label]['conductance'][i]):.12e}")
            fh.write("\n")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {
        "0.0": "#111111",
        "0.2": "#455a64",
        "0.5": "#1565c0",
        "0.7": "#2e7d32",
        "1.0": "#ef6c00",
        "2.0": "#b71c1c",
    }
    for label, curve in result["curves"].items():
        y = 1.0e9 * np.asarray(curve["conductance"], dtype=float)
        ax.plot(result["temperatures"], y, lw=2.0, color=colors.get(label, None), label=f"t={label}")

    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Conductance (10^-9 W/K)")
    ax.set_title("Wang 0701164 Fig. 5 Cubic Onsite LO")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_fig5_conductance_vs_temperature.png", dpi=220)

    if str(args.compare_reference) != "none":
        if str(args.compare_reference) == "digitized":
            metrics = wang0701164_compare_sweep_to_digitized(result, args.digitized_path)
            reference = wang0701164_load_digitized_fig5_curves(args.digitized_path)
            ref_suffix = "digitized"
        else:
            metrics = wang0701164_compare_sweep_to_vector_pdf(
                result,
                args.reference_pdf,
                page=int(args.reference_page),
            )
            reference = wang0701164_extract_fig5_vector_curves(
                args.reference_pdf,
                page=int(args.reference_page),
            )
            ref_suffix = "vector"

        ref_curves = reference["curves"]
        ref_curve_temperatures = reference.get("curve_temperatures")
        overlay_fig, overlay_ax = plt.subplots(figsize=(7.4, 4.8))
        resid_fig, resid_ax = plt.subplots(figsize=(7.4, 4.8))
        for label, curve in result["curves"].items():
            if isinstance(ref_curve_temperatures, dict) and label in ref_curve_temperatures:
                t_ref = np.asarray(ref_curve_temperatures[label], dtype=float)
            else:
                t_ref = np.asarray(reference["temperatures"], dtype=float)
            y_model = 1.0e9 * np.asarray(curve["conductance"], dtype=float)
            y_interp = 1.0e9 * np.interp(t_ref, result["temperatures"], np.asarray(curve["conductance"], dtype=float))
            y_ref = 1.0e9 * np.asarray(ref_curves[label], dtype=float)
            color = colors.get(label, None)
            overlay_ax.plot(t_ref, y_ref, "--", color=color, alpha=0.65)
            overlay_ax.plot(result["temperatures"], y_model, "-", lw=2.0, color=color, label=f"t={label}")
            resid_ax.plot(t_ref, y_interp - y_ref, "-", lw=2.0, color=color, label=f"t={label}")

        overlay_ax.set_xlabel("Temperature (K)")
        overlay_ax.set_ylabel("Conductance (10^-9 W/K)")
        overlay_ax.set_title(f"{prefix} Model vs {ref_suffix.title()} Fig. 5")
        overlay_ax.grid(alpha=0.3)
        overlay_ax.legend(ncol=2)
        overlay_fig.tight_layout()
        overlay_fig.savefig(outdir / f"{prefix}_fig5_model_vs_{ref_suffix}_overlay.png", dpi=220)

        resid_ax.axhline(0.0, color="black", lw=1)
        resid_ax.set_xlabel("Temperature (K)")
        resid_ax.set_ylabel("Residual (10^-9 W/K)")
        resid_ax.set_title(f"{prefix} Residuals vs {ref_suffix.title()} Fig. 5")
        resid_ax.grid(alpha=0.3)
        resid_ax.legend(ncol=2)
        resid_fig.tight_layout()
        resid_fig.savefig(outdir / f"{prefix}_fig5_model_vs_{ref_suffix}_residuals.png", dpi=220)

        (outdir / f"{prefix}_fig5_model_vs_{ref_suffix}_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

    meta = {
        "spec": wang0701164_fig5_spec(unit_system=str(args.unit_system)),
        "parameters": {
            "temp_min": float(args.temp_min),
            "temp_max": float(args.temp_max),
            "n_temp": int(args.n_temp),
            "n_omega": int(args.n_omega),
            "omega_max": None if args.omega_max is None else float(args.omega_max),
            "internal_oversample": int(args.internal_oversample),
            "delta_t": float(args.delta_t),
            "eta": float(args.eta),
            "unit_system": str(args.unit_system),
            "conductance_mode": str(args.conductance_mode),
            "include_second_graph": bool(args.include_second_graph),
            "compare_reference": str(args.compare_reference),
            "digitized_path": str(args.digitized_path),
            "reference_pdf": str(args.reference_pdf),
            "reference_page": int(args.reference_page),
        },
        "notes": (
            "This benchmark uses the published cubic-onsite toy model. "
            "The default Fig. 5 path evaluates the first LO graph exactly in the time domain and, by default, "
            "removes the constant tadpole-like second graph because that choice reproduces the published conductance "
            "family far more faithfully. Tiny negative residual conductances from finite frequency grids are projected to zero. "
            "The frequency window used for the time transform may extend beyond the physical band edge, but the conductance "
            "integral itself is restricted to the harmonic band."
        ),
    }
    (outdir / f"{prefix}_benchmark_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
