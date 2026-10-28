"""Render compact, portable slide figures from the curated scalar evidence.

Default usage requires only Python, matplotlib and the included JSON::

    python generate_figures.py

Maintainers can refresh the curated JSON from the original, local workspace
completion summaries with ``--workspace /path/to/workspace``. This optional
mode reads six small JSON files only: no remote access, logs or arrays.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
SIGMA = ("retarded", "lesser", "greater")
CURRENTS = ("collision_power", "contact_balance", "full_balance")
PACKAGES = (
    ("original", "Original continuation", "multiq-site-energy-convergence-20261026", [4, 1], 601, 2),
    ("reproduction", "Fresh-zero reproduction", "multiq-site-energy-reproduction-retry-20261027", [4, 1], 601, None),
    ("f901", "Frequency 901", "multiq-site-energy-mesh-study-retry-20261027/f901", [4, 1], 901, None),
    ("f1201", "Frequency 1201 continuation", "multiq-site-energy-f1201-continuation-20261027", [4, 1], 1201, 17),
    ("q12", "Second-axis 1 x 2", "multiq-q-direction-study-20261028/q12", [1, 2], 601, None),
    ("q14", "Equal-size axis control 1 x 4", "multiq-q-direction-study-20261028/q14", [1, 4], 601, None),
)


def scalar(value):
    """Reject booleans and non-finite values instead of silently plotting them."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Expected a finite recorded scalar")
    if not math.isfinite(value):
        raise ValueError("Non-finite recorded scalar")
    return value


def refresh_from_workspace(workspace: Path):
    report_root = Path("reports/phonon-phonon-interaction")
    summaries = {}
    for key, _, package, *_ in PACKAGES:
        rel = report_root / package / "completion-summary.json"
        summaries[key] = (rel.as_posix(), json.loads((workspace / rel).read_text()))

    baseline = scalar(summaries["reproduction"][1]["result"]["selected_anchor"]["currents"]["left"])
    records = []
    trace = []
    for key, label, _, mesh, nodes, seed in PACKAGES:
        source, summary = summaries[key]
        result = summary["result"]
        if not (summary.get("complete") is True
                and summary.get("actual_exit_status") == 0
                and result.get("passed") is True
                and result.get("converged") is True
                and result.get("stop_reason") == "fixed_point"
                and result.get("failure_reason") is None
                and summary.get("wrapper", {}).get("passed") is True):
            raise ValueError(f"Not an eligible completed convergence result: {key}")

        if key == "original":
            selected = next(s for s in result["states"] if s["index"] == result["selected_anchor_index"])
            sigma_ratios = {c: scalar(selected["sigma_residuals"][c]["ratio"]) for c in SIGMA}
            current_ratios = {c: scalar(selected["current_screen_ratios"][c]) for c in CURRENTS}
            comparison = summaries["reproduction"][1]["result"]["current_comparison"]
            currents = {c: scalar(comparison["contacts"][c]["reference_w"]) for c in ("left", "right")}
            current_source = {
                "summary": summaries["reproduction"][0],
                "json_field": "result.current_comparison.contacts.{left,right}.reference_w",
                "note": "Recorded prior-current context from the successful reproduction; original completion summary omits currents.",
            }
            for state in result["states"]:
                trace.append({
                    "job_id": summary["job_id"],
                    "native_state_index": state["index"],
                    "completed_map_in_this_job": len(trace) + 1,
                    "selected_as_anchor": state["selected_as_anchor"],
                    "sigma_passed": state["sigma_passed"],
                    "current_passed": state["current_passed"],
                    **{f"sigma_{c}_ratio": scalar(state["sigma_residuals"][c]["ratio"]) for c in SIGMA},
                    **{f"sigma_{c}_error": scalar(state["sigma_residuals"][c]["error"]) for c in SIGMA},
                    **{f"sigma_{c}_error_frobenius": scalar(state["sigma_residuals"][c]["error_frobenius"]) for c in SIGMA},
                    **{f"current_{c}_ratio": scalar(state["current_screen_ratios"][c]) for c in CURRENTS},
                    "gamma_zero_mapped_C_frobenius_central": scalar(state["Gamma_zero_mapped_C_frobenius"]["central"]),
                    "gamma_zero_mapped_C_frobenius_outer": scalar(state["Gamma_zero_mapped_C_frobenius"]["outer"]),
                })
        else:
            selected = result["selected_anchor"]
            sigma_ratios = {c: scalar(selected["sigma_screen_ratios"][c]) for c in SIGMA}
            current_ratios = {c: scalar(selected["current_screen_ratios"][c]) for c in CURRENTS}
            currents = {c: scalar(selected["currents"][c]) for c in ("left", "right", "collision_power")}
            current_source = {"summary": source, "json_field": "result.selected_anchor.currents"}
        if not (selected["sigma_passed"] is True and selected["current_passed"] is True
                and all(0 <= v <= 1 for v in (*sigma_ratios.values(), *current_ratios.values()))):
            raise ValueError(f"Original six screens are not all passing: {key}")
        if result["selected_anchor_index"] != result["last_evaluated_trial_index"]:
            raise ValueError(f"Selected/latest roles need separate treatment: {key}")
        left = currents["left"]
        records.append({
            "id": key,
            "label": label,
            "job_id": summary["job_id"],
            "source_completion_summary": source,
            "actual_exit_status": summary["actual_exit_status"],
            "passed": result["passed"],
            "converged": result["converged"],
            "stop_reason": result["stop_reason"],
            "failure_reason": result["failure_reason"],
            "q_grid_transverse": mesh,
            "q_points": math.prod(mesh),
            "frequency_nodes": nodes,
            "continuation_seed_native_index": seed,
            "completed_maps_this_job": result["completed_evaluations"],
            "selected_maps_this_job": result["selected_anchor_count"],
            "rejected_maps_this_job": result["rejected_trial_count"],
            "selected_native_state_index": result["selected_anchor_index"],
            "latest_native_state_index": result["last_evaluated_trial_index"],
            "original_six_scba_screens_passed": True,
            "sigma_screen_ratios": sigma_ratios,
            "current_screen_ratios": current_ratios,
            "currents_w": currents,
            "current_provenance": current_source,
            "left_current_difference_from_4x1_601_w": left - baseline,
            "left_current_relative_difference_from_4x1_601": (left - baseline) / abs(baseline),
            "left_current_signed_difference_from_4x1_601_percent": 100 * (left - baseline) / abs(baseline),
            "within_2_percent_diagnostic_target": abs(left - baseline) <= 0.02 * abs(baseline) + 1e-14,
        })
        if key == "q14":
            comparison = result["current_comparison"]
            records[-1]["q_sample_role"] = "Equal-Nq axis control against 4x1; not a two-dimensional mesh refinement."
            records[-1]["recorded_current_comparison"] = {
                "source_completion_summary": source,
                "json_field": "result.current_comparison.contacts",
                "diagnostic_comparison_only": comparison["diagnostic_comparison_only"],
                "scba_gate_unchanged": comparison["scba_gate_unchanged"],
                "rtol": comparison["rtol"],
                "atol_w": comparison["atol_w"],
                "contacts": {
                    contact: {field: values[field] for field in (
                        "actual_w", "reference_w", "difference_w", "relative_difference", "matches"
                    )}
                    for contact, values in comparison["contacts"].items()
                },
            }

    payload = {
        "schema": "negfpy.curated-converged-scalars.v1",
        "curated_date": "2026-10-28",
        "model": {
            "kind": "preserved fitted complete-center-energy research surrogate",
            "reported_bulk_fc3_relative_frobenius_fit_error": 0.021414695219655902,
            "reported_bulk_fc3_relative_frobenius_fit_error_percent": 2.1414695219655902,
            "same_saved_coefficients_without_refit": True,
            "actual_ifc_files_modified": False,
            "scba_vertex_role": "Declared fitted energy-model vertex; distinct from a direct raw-FC3 reference calculation.",
        },
        "common_physics": {
            "g": 0.03,
            "eta_lead": 0.0015625,
            "eta_device": 0,
            "temperatures_k": [305, 295],
            "full_green_dof": 216,
            "active_vertex_dof": 132,
            "device_layers": 9,
            "interaction_layers": [2, 6],
            "frequency_range_solver_units": [-7.5, 7.5],
            "original_guards_and_thresholds_unchanged": True,
            "sigma_rtol": 1e-4,
            "sigma_atol": 1e-10,
            "current_rtol": 1e-3,
            "current_atol_w": 1e-14,
        },
        "interpretation": {
            "included": "Only exit-zero, passed, fixed-point converged jobs with original three Sigma and three current screens passing.",
            "ratio_definition": "Recorded original screen error divided by that screen's original allowance; pass is ratio <= 1.",
            "convergence_scope": "Numerical SCBA convergence for each declared fitted model and sampled grid.",
            "open_questions": ["full two-dimensional q-mesh convergence", "unique physical-interface model", "production transport accuracy"],
            "frequency_comparison": "601/901/1201 comparisons keep the 4x1 transverse grid and frequency range fixed; the 1201 result is a continuation from native17.",
            "q_comparison": "Equal-Nq 1x4 and 4x1 axes agree in recorded left current to about 1.39e-14 relative. The 1x2 coarse second-direction sample differs by about 4.20%; neither is a two-dimensional q-mesh refinement, and full q-mesh convergence remains open.",
            "diagnostic_current_target": {"rtol": 0.02, "atol_w": 1e-14, "is_scba_gate": False},
            "trace_scope": "Original successful continuation only, native3..13 (11 new maps from seed native2). Other full traces are not reconstructed from logs.",
            "no_numerical_replay": True,
        },
        "records": records,
        "original_continuation_trace": trace,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "converged-results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def write_csvs(payload):
    rows = []
    for r in payload["records"]:
        rows.append({
            "result_id": r["id"], "job_id": r["job_id"],
            "q_grid": "x".join(map(str, r["q_grid_transverse"])),
            "frequency_nodes": r["frequency_nodes"],
            "completed_maps_this_job": r["completed_maps_this_job"],
            "continuation_seed_native_index": r["continuation_seed_native_index"],
            "selected_native_state_index": r["selected_native_state_index"],
            "latest_native_state_index": r["latest_native_state_index"],
            "rejected_maps_this_job": r["rejected_maps_this_job"],
            "actual_exit_status": r["actual_exit_status"],
            "converged": r["converged"],
            "original_six_scba_screens_passed": r["original_six_scba_screens_passed"],
            "left_current_w": r["currents_w"]["left"],
            "right_current_w": r["currents_w"]["right"],
            "left_current_nw": r["currents_w"]["left"] * 1e9,
            "relative_difference_from_4x1_601": r["left_current_relative_difference_from_4x1_601"],
            "signed_difference_from_4x1_601_percent": r["left_current_signed_difference_from_4x1_601_percent"],
            **{f"sigma_{c}_ratio": r["sigma_screen_ratios"][c] for c in SIGMA},
            **{f"current_{c}_ratio": r["current_screen_ratios"][c] for c in CURRENTS},
            "source_completion_summary": r["source_completion_summary"],
            "current_source_summary": r["current_provenance"]["summary"],
            "current_source_json_field": r["current_provenance"]["json_field"],
        })
    for name, content in (("converged-currents.csv", rows), ("original-continuation-trace.csv", payload["original_continuation_trace"])):
        with (DATA / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(content[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(content)


def render_figures(payload):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 13, "axes.labelsize": 11,
        "svg.fonttype": "none", "svg.hashsalt": "negfpy-converged-20261028",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    footer = "Fitted research model (2.14147% bulk FC3 fit error). Model/grid SCBA convergence; interface and full q-mesh validity remain open."

    def save(fig, name):
        fig.savefig(FIGURES / f"{name}.png", dpi=220, facecolor="white")
        fig.savefig(FIGURES / f"{name}.svg", facecolor="white", metadata={"Date": None})
        svg = FIGURES / f"{name}.svg"
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        plt.close(fig)

    records = payload["records"]
    labels = ["Original: 4 x 1 / 601*", "Reproduction: 4 x 1 / 601", "Frequency: 4 x 1 / 901", "Frequency: 4 x 1 / 1201", "Coarse direction: 1 x 2 / 601", "Equal-size axis: 1 x 4 / 601"]
    values = [r["currents_w"]["left"] * 1e9 for r in records]
    colors = ["#557a9c", "#1464a0", "#258f83", "#438a45", "#b76527", "#7759a4"]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.28, right=0.88, top=0.84, bottom=0.24)
    ax.scatter(values, range(len(values)), c=colors, s=90, zorder=3)
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    ax.set_xlim(3.84, 4.065)
    ax.set_xlabel("Selected evaluated-input left heat current (nW)")
    ax.axvline(values[1], color="#737373", linewidth=1, linestyle=":")
    ax.grid(axis="x", alpha=0.22)
    ax.set_title("Successful SCBA results: all original six screens pass", pad=15)
    for y, value in enumerate(values):
        ax.annotate(f"{value:.6f}", (value, y), xytext=(8, 7), textcoords="offset points", color=colors[y], fontsize=10)
    fig.text(0.025, 0.13, "* Original current is the recorded prior-current reference in the successful reproduction summary.", fontsize=9)
    fig.text(0.025, 0.075, "The 1 x 2 current is +4.1985% from the 4 x 1 baseline; this is a different q sample, not q-mesh convergence.", fontsize=9)
    fig.text(0.025, 0.025, footer, fontsize=8)
    save(fig, "successful-currents")

    by_id = {record["id"]: record for record in records}
    direction = [by_id[key] for key in ("reproduction", "q12", "q14")]
    direction_current = [record["currents_w"]["left"] * 1e9 for record in direction]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.30, top=0.75)
    ax.scatter(range(3), direction_current, c=["#1464a0", "#b76527", "#7759a4"], s=90, zorder=3)
    ax.set_xticks(range(3), ["4 x 1 baseline\n4 q points", "1 x 2 coarse sample\n2 q points", "1 x 4 axis control\n4 q points"])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(3.82, 4.10)
    ax.set_ylabel("Selected left heat current (nW)")
    ax.axhline(direction_current[0], linestyle=":", linewidth=1, color="#737373")
    ax.grid(axis="y", alpha=0.20)
    for x_pos, value in enumerate(direction_current):
        ax.annotate(f"{value:.6f}", (x_pos, value), xytext=(0, 12), ha="center", textcoords="offset points")
    fig.suptitle("Directional q controls: equal-size axes agree", y=0.96, fontsize=14)
    fig.text(0.11, 0.855, "601 frequency nodes; same fitted model; all three runs pass the original six SCBA screens.", fontsize=10)
    fig.text(0.025, 0.16, "1 x 4 vs 4 x 1: recorded left-current difference = -5.38e-23 W (-1.39e-14 relative).", fontsize=9)
    fig.text(0.025, 0.105, "1 x 2 vs 4 x 1: +4.1985%. These axis samples do not establish two-dimensional q-mesh convergence.", fontsize=9)
    fig.text(0.025, 0.045, footer, fontsize=8)
    save(fig, "q-direction-comparison")

    freq = [r for r in records if r["id"] in ("reproduction", "f901", "f1201")]
    x = [r["frequency_nodes"] for r in freq]
    currents = [r["currents_w"]["left"] * 1e9 for r in freq]
    changes = [r["left_current_signed_difference_from_4x1_601_percent"] for r in freq]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.78, wspace=0.35)
    axes[0].plot(x, currents, "o-", color="#1464a0", markersize=7)
    axes[0].set_ylabel("Left heat current (nW)")
    axes[0].set_ylim(3.855, 3.900)
    axes[1].axhspan(-2, 2, color="#258f83", alpha=0.12, label="2% diagnostic target")
    axes[1].plot(x, changes, "o-", color="#1464a0", markersize=7)
    axes[1].axhline(0, color="#737373", linewidth=0.8)
    axes[1].set_ylabel("Signed current change from 601 (%)")
    axes[1].set_ylim(-2.4, 2.6)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xlabel("Frequency nodes")
        ax.set_xticks(x)
        ax.set_xlim(550, 1250)
        ax.grid(alpha=0.20)
    for xx, value in zip(x, changes):
        axes[1].annotate(f"{value:+.3f}%", (xx, value), xytext=(0, 10), ha="center", textcoords="offset points", fontsize=9)
    fig.suptitle("Frequency sensitivity at fixed 4 x 1 q sampling and range", y=0.96, fontsize=14)
    fig.text(0.09, 0.865, "All three runs converge the original six screens. 1201 uses a native17 continuation.", fontsize=10)
    fig.text(0.025, 0.13, "1201 vs 901: +0.6851%; 1201 vs 601: +0.5863%. The 2% target is diagnostic, not an SCBA gate.", fontsize=9)
    fig.text(0.025, 0.075, "This supports the sampled frequency sensitivity only; it does not establish full q-mesh or physical-interface validity.", fontsize=9)
    fig.text(0.025, 0.025, footer, fontsize=8)
    save(fig, "frequency-sensitivity")

    trace = payload["original_continuation_trace"]
    x = [s["native_state_index"] for s in trace]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.24, top=0.78, wspace=0.34)
    for channel, color, style in zip(SIGMA, ("#1464a0", "#c25c23", "#258f83"), ("-", "--", ":")):
        axes[0].semilogy(x, [s[f"sigma_{channel}_ratio"] for s in trace], style, marker="o", color=color, markersize=3, label=channel.capitalize())
    axes[0].set_ylabel("Original unmixed Sigma screen ratio")
    axes[0].legend(frameon=False, fontsize=8)
    for channel, color, style in zip(CURRENTS, ("#1464a0", "#c25c23", "#258f83"), ("-", "--", ":")):
        axes[1].semilogy(x, [s[f"current_{channel}_ratio"] for s in trace], style, marker="o", color=color, markersize=3, label=channel.replace("_", " ").capitalize())
    axes[1].set_ylabel("Original current screen ratio")
    axes[1].set_ylim(8e-13, 5)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, fontsize=8, loc="upper right")
    for ax in axes:
        ax.axhline(1, color="#555555", linewidth=1, linestyle="--")
        ax.set_xticks([3, 5, 7, 9, 11, 13])
        ax.set_xlabel("Native evaluated state index")
        ax.grid(alpha=0.18)
    fig.suptitle("Original successful continuation: 11 new full maps, native 3–13", y=0.96, fontsize=14)
    fig.text(0.09, 0.865, "Seed is native2. The final native13 passes all three Sigma and all three current screens.", fontsize=10)
    fig.text(0.025, 0.13, "Ratio = recorded error / original allowance; values at or below 1 pass. Close curves can overlap.", fontsize=9)
    fig.text(0.025, 0.075, "Recorded scalar trace only; no field replay, reconstructed missing states, or altered convergence thresholds.", fontsize=9)
    fig.text(0.025, 0.025, footer, fontsize=8)
    save(fig, "original-convergence-trace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, help="Optional original workspace root, for refreshing only the five small completion summaries")
    args = parser.parse_args()
    payload = refresh_from_workspace(args.workspace) if args.workspace else json.loads((DATA / "converged-results.json").read_text())
    write_csvs(payload)
    render_figures(payload)


if __name__ == "__main__":
    main()
