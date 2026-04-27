"""Extract Wang 0701164 Fig. 5 reference curves directly from the PDF vector plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.inelastic import wang0701164_extract_fig5_vector_curves


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
    parser.add_argument("--pdf-path", type=Path, default=Path("literature/0701164v1.pdf"))
    parser.add_argument("--page", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inelastic/Wang0701164"))
    args = parser.parse_args()

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{_next_run_index(outdir):03d}"
    reference = wang0701164_extract_fig5_vector_curves(args.pdf_path, page=int(args.page))

    table_path = outdir / f"{prefix}_fig5_vector_reference.tsv"
    labels = ["0.0", "0.2", "0.5", "0.7", "1.0", "2.0"]
    curve_temperatures = {
        label: np.asarray(reference["curve_temperatures"][label], dtype=float) for label in labels
    }
    ref_t = np.unique(np.concatenate([curve_temperatures[label] for label in labels]))
    with table_path.open("w", encoding="utf-8") as fh:
        fh.write("temperature_K")
        for label in labels:
            key = label.replace(".", "p")
            fh.write(f"\tkappa_t{key}_1e-9_W_per_K")
        fh.write("\n")
        for i, temp in enumerate(ref_t):
            fh.write(f"{float(temp):.8f}")
            for label in labels:
                temps_label = curve_temperatures[label]
                values_label = np.asarray(reference["curves"][label], dtype=float)
                interp_val = np.interp(float(temp), temps_label, values_label)
                fh.write(f"\t{1.0e9 * float(interp_val):.8f}")
            fh.write("\n")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = {
        "0.0": "#111111",
        "0.2": "#d32f2f",
        "0.5": "#2e7d32",
        "0.7": "#1565c0",
        "1.0": "#8e24aa",
        "2.0": "#5d4037",
    }
    for label in labels:
        ax.plot(
            curve_temperatures[label],
            1.0e9 * np.asarray(reference["curves"][label], dtype=float),
            lw=2.0,
            color=colors[label],
            label=f"t={label}",
        )
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Conductance (10^-9 W/K)")
    ax.set_title("Wang 0701164 Fig. 5 Vector Reference")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"{prefix}_fig5_vector_reference.png", dpi=220)

    meta = {
        "pdf_path": str(args.pdf_path),
        "page": int(args.page),
        "notes": (
            "These reference curves are extracted from the vector paths in the PDF page SVG, "
            "not from raster digitization. The six left-panel plot paths are mapped to the "
            "published t = 0, 0.2, 0.5, 0.7, 1.0, 2.0 family by peak-height ordering. "
            "The TSV is written on the union of the curve vertex temperatures, so non-native "
            "curve values in that table are interpolation summaries."
        ),
        "colors": reference.get("colors", {}),
        "axis_mapping": reference.get("axis_mapping", {}),
    }
    (outdir / f"{prefix}_fig5_vector_reference_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
