"""Analyze a completed selected-current package with its pinned frozen assessor.

This executes the assessor from a trusted local report directory after checking
its profile binding. It never contacts a scheduler or evaluates Green functions.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from negfpy.anharmonic import diagnose_current_memory


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--window-start", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    root = args.report.resolve()
    if digest(root / "profile.json") != args.profile_sha256:
        raise ValueError("Input profile changed.")
    profile = json.loads((root / "profile.json").read_text())
    for name, expected in profile["files_sha256"].items():
        if Path(name).name != name or (root / name).is_symlink() or digest(root / name) != expected:
            raise ValueError(f"Frozen operational file differs: {name}")
    path = root / "assess_terminal.py"
    spec = importlib.util.spec_from_file_location("frozen_memory_input_assessor", path)
    assessor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assessor)
    terminal = root / "terminal"
    assessment = assessor.assess(terminal, args.job_id, args.profile_sha256)
    if assessment != json.loads((root / "assessment.json").read_text()):
        raise ValueError("Frozen assessment replay differs from the accepted record.")
    summary = json.loads((terminal / "output/summary.json").read_text())
    reports, curves = [], {}
    with assessor.verified_archives(terminal / "output", summary["artifacts"]) as streams:
        for result in assessment["results"]:
            step, cut = result["evaluation_grid_step"], result["cut"]
            artifact = f"cut-{cut}-step-{step}.npz"
            with np.load(streams[artifact], allow_pickle=False) as archive:
                lag, integrand = archive["lag_times"], archive["connected_integrand_solver"]
            diagnosed = diagnose_current_memory(
                lag, integrand, current_scale=summary["current_scale"],
                harmonic_current=result["harmonic"],
                reference_current=assessment["refined_left_contact_reference"],
                window_start=args.window_start, frequency_step=step)
            endpoint = result["memory_scan"][-1]["connected_cubic"]
            if not np.isclose(diagnosed["connected_current"][-1], endpoint,
                              rtol=1e-10, atol=1e-22):
                raise ValueError("Cumulative integral disagrees with frozen endpoint.")
            selected = [row for row in result["memory_scan"] if row["lag_end"] >= args.window_start]
            sampled_values = [row["connected_cubic"] for row in selected]
            sampled_range = max(sampled_values) - min(sampled_values)
            full_range = diagnosed["summary"]["peak_to_peak_current"]
            reports.append(dict(cut=cut, frequency_step=step,
                                sparse_endpoint_range=sampled_range,
                                sparse_endpoints=[r["lag_end"] for r in selected],
                                full_to_sparse_range_ratio=full_range / sampled_range if sampled_range else None,
                                **diagnosed["summary"]))
            key = f"cut{cut}_step{step}"
            curves[key + "_lag"] = diagnosed["lag_times"]
            curves[key + "_connected"] = diagnosed["connected_current"]
            curves[key + "_total"] = diagnosed["total_current"]
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": "negfpy.research.full-memory-window.v1", "job_id": args.job_id,
                    "source_report": str(root), "profile_sha256": args.profile_sha256,
                    "source_assessment_sha256": digest(root / "assessment.json"),
                    "analyzer_sha256": digest(Path(__file__)),
                    "diagnostics_sha256": digest(Path(__file__).parents[1] / "src/negfpy/anharmonic/memory_diagnostics.py"),
                    "frozen_assessment_reproduced": True,
                    "all_declared_archives_verified_before_load": True,
                    "results": reports, "new_green_evaluations": 0, "new_scba_maps": 0,
                    "convergence_certified": False, "changes_solver_acceptance": False}
    with (args.output / "analysis.json").open("x") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with (args.output / "curves.npz").open("xb") as stream:
        np.savez_compressed(stream, **curves)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cuts = sorted({r["cut"] for r in reports})
        fig, axes = plt.subplots(1, len(cuts), figsize=(11, 4), squeeze=False)
        for ax, cut in zip(axes[0], cuts):
            for row in (r for r in reports if r["cut"] == cut):
                key = f"cut{cut}_step{row['frequency_step']}"
                lag, current = curves[key + "_lag"], curves[key + "_connected"]
                use = lag >= args.window_start
                line, = ax.plot(lag[use], 1e9 * current[use], label=f"frequency step {row['frequency_step']}")
                positions = [int(np.flatnonzero(lag == end)[0]) for end in row["sparse_endpoints"]]
                ax.scatter(lag[positions], 1e9 * current[positions], color=line.get_color(), s=25)
            ax.axhline(0, color="0.4", lw=.7)
            ax.set(title=f"Cut {cut}", xlabel="Memory cutoff (solver units)",
                   ylabel="Connected cubic current (nW)")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle("Saved current curves: oscillations between reported endpoints")
        fig.tight_layout()
        fig.savefig(args.output / "memory-curves.png", dpi=180)
        fig.savefig(args.output / "memory-curves.svg")
        plt.close(fig)
    print(json.dumps({"output": str(args.output), "results": reports}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
