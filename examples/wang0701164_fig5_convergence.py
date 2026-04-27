"""Run a convergence scan for Wang 0701164 Fig. 5 against digitized curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.inelastic import (
    wang0701164_compare_sweep_to_digitized,
    wang0701164_fig5_lowest_order_conductance_sweep,
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


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp-min", type=float, default=50.0)
    parser.add_argument("--temp-max", type=float, default=2000.0)
    parser.add_argument("--n-temp-values", type=str, default="21,41,81")
    parser.add_argument("--n-omega-values", type=str, default="61,121")
    parser.add_argument("--omega-max-values", type=str, default="2.1,2.3")
    parser.add_argument("--eta-values", type=str, default="2e-5,5e-5,1e-4")
    parser.add_argument("--delta-t", type=float, default=0.02)
    parser.add_argument("--conductance-mode", type=str, default="paper_derivative_numeric")
    parser.add_argument("--include-second-graph", action="store_true")
    parser.add_argument(
        "--digitized-path",
        type=Path,
        default=Path("outputs/inelastic/Wang0701164/002_fig5_digitized_curves.tsv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inelastic/Wang0701164"))
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_next_run_index(outdir):03d}"

    temps_by_count = {
        n_temp: np.linspace(float(args.temp_min), float(args.temp_max), int(n_temp))
        for n_temp in _parse_int_list(args.n_temp_values)
    }
    n_omega_values = _parse_int_list(args.n_omega_values)
    omega_max_values = _parse_float_list(args.omega_max_values)
    eta_values = _parse_float_list(args.eta_values)

    rows: list[dict[str, float | int | str]] = []
    best: dict[str, object] | None = None
    best_result: dict[str, object] | None = None
    for n_temp, temps in temps_by_count.items():
        for n_omega in n_omega_values:
            for omega_max in omega_max_values:
                for eta in eta_values:
                    result = wang0701164_fig5_lowest_order_conductance_sweep(
                        temps,
                        unit_system="paper",
                        n_omega=int(n_omega),
                        omega_max=float(omega_max),
                        delta_t=float(args.delta_t),
                        eta=float(eta),
                        conductance_mode=str(args.conductance_mode),
                        include_second_graph=bool(args.include_second_graph),
                    )
                    metrics = wang0701164_compare_sweep_to_digitized(result, args.digitized_path)
                    row = {
                        "n_temp": int(n_temp),
                        "n_omega": int(n_omega),
                        "omega_max": float(omega_max),
                        "eta": float(eta),
                        "score": float(metrics["score"]),
                    }
                    rows.append(row)
                    if best is None or float(row["score"]) < float(best["score"]):
                        best = row
                        best_result = result

    table_path = outdir / f"{prefix}_fig5_convergence_scan.tsv"
    with table_path.open("w", encoding="utf-8") as fh:
        fh.write("n_temp\tn_omega\tomega_max\teta\tscore\n")
        for row in sorted(rows, key=lambda item: float(item["score"])):
            fh.write(
                f"{int(row['n_temp'])}\t{int(row['n_omega'])}\t{float(row['omega_max']):.6f}\t"
                f"{float(row['eta']):.6e}\t{float(row['score']):.12e}\n"
            )

    summary = {
        "digitized_path": str(args.digitized_path),
        "conductance_mode": str(args.conductance_mode),
        "include_second_graph": bool(args.include_second_graph),
        "delta_t": float(args.delta_t),
        "best": best,
        "rows": rows,
    }
    (outdir / f"{prefix}_fig5_convergence_scan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(rows))
    order = np.argsort([float(row["score"]) for row in rows])
    ordered_rows = [rows[i] for i in order]
    ordered_scores = [float(row["score"]) for row in ordered_rows]
    labels = [
        f"Nt={int(row['n_temp'])}, No={int(row['n_omega'])}, "
        f"om={float(row['omega_max']):.2f}, eta={float(row['eta']):.0e}"
        for row in ordered_rows
    ]
    ax.plot(x, ordered_scores, "o-", lw=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.set_ylabel("Digitized Comparison Score")
    ax.set_title("Wang 0701164 Fig. 5 Convergence Scan")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_fig5_convergence_scan.png", dpi=220)

    if best_result is not None:
        best_table_path = outdir / f"{prefix}_fig5_convergence_best.tsv"
        with best_table_path.open("w", encoding="utf-8") as fh:
            fh.write("# T")
            for label in best_result["curves"]:
                fh.write(f"\tkappa_t_{label}")
            fh.write("\n")
            for i, temp in enumerate(best_result["temperatures"]):
                fh.write(f"{float(temp):.8f}")
                for label in best_result["curves"]:
                    fh.write(f"\t{float(best_result['curves'][label]['conductance'][i]):.12e}")
                fh.write("\n")


if __name__ == "__main__":
    main()
