from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import pytest

if any(importlib.util.find_spec(name) is None for name in ("matplotlib", "PIL")):
    pytest.skip(
        "install negfpy[report] for report rendering tests",
        allow_module_level=True,
    )
from PIL import Image  # noqa: E402

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "render_production_toy_gate_figures.py"
)
SPEC = importlib.util.spec_from_file_location(
    "negfpy_render_production_toy_gate_figures",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
RENDERER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDERER
SPEC.loader.exec_module(RENDERER)


def _csv_bytes(headers: tuple[str, ...], rows: list[dict[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(headers), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return stream.getvalue().encode("utf-8")


def _open_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    coordinates = [
        (0.0, -0.02),
        (0.0, -0.01),
        (0.0, 0.0),
        (0.0, 0.01),
        (0.0, 0.02),
        (0.7, -0.02),
        (0.7, -0.01),
        (0.7, 0.0),
        (0.7, 0.01),
        (0.7, 0.02),
    ]
    controls = [(0.1, 101, 5.0, 0.01)] * 10 + [
        (0.05, 201, 5.0, 0.01),
        (0.1, 121, 6.0, 0.01),
        (0.1, 101, 5.0, 0.005),
    ]
    coordinates.extend([(0.7, 0.02)] * 3)
    for index, ((temperature, g), control) in enumerate(zip(coordinates, controls), start=1):
        magnitude = g * g * (1.0 + temperature)
        oracle_linewidth = magnitude
        born_linewidth = magnitude * 1.001
        oracle_shift = -0.25 * magnitude
        born_shift = oracle_shift * 1.001
        rows.append(
            {
                "alias": f"O{index:02d}",
                "name": ("base" if index <= 10 else "refine") + f"_synthetic_{index:02d}",
                "temperature": temperature,
                "g": g,
                "grid_step": control[0],
                "grid_n_points": control[1],
                "cutoff": control[2],
                "eta_lead": control[3],
                "oracle_parent_linewidth": oracle_linewidth,
                "production_parent_linewidth": born_linewidth,
                "linewidth_gate_ratio": 0.2,
                "oracle_parent_shift": oracle_shift,
                "production_parent_shift": born_shift,
                "shift_gate_ratio": 0.3,
                "agreement_passed": "true",
                "weak_coupling_passed": "true",
                "passivity_passed": "true",
                "oracle_wall_seconds": 0.01 * index,
                "born_wall_seconds": 0.005 * index,
            }
        )
    return rows


def _open_checks() -> list[dict[str, object]]:
    return [
        {
            "kind": "sign",
            "check": "sign_01",
            "source": "production_born",
            "observable": "linewidth",
            "temperature": 0.7,
            "g_or_control": 0.02,
            "base_run": "base_negative",
            "candidate_run": "base_positive",
            "gate_ratio": 0.2,
            "passed": "true",
            "certification_critical": "true",
        },
        {
            "kind": "scaling",
            "check": "scaling_01",
            "source": "oracle",
            "observable": "shift",
            "temperature": 0.7,
            "g_or_control": 0.5,
            "base_run": "base_small",
            "candidate_run": "base_large",
            "gate_ratio": 0.4,
            "passed": "true",
            "certification_critical": "true",
        },
        {
            "kind": "refinement",
            "check": "cutoff",
            "source": "production_born",
            "observable": "linewidth",
            "temperature": 0.7,
            "g_or_control": "cutoff",
            "base_run": "base",
            "candidate_run": "refined",
            "gate_ratio": 0.8,
            "passed": "true",
            "certification_critical": "true",
        },
        {
            "kind": "spontaneous_decay",
            "check": "spontaneous_01",
            "source": "production",
            "observable": "linewidth",
            "temperature": 0.0,
            "g_or_control": 0.02,
            "base_run": "base_zero_temperature",
            "candidate_run": "",
            "gate_ratio": 0.1,
            "passed": "true",
            "certification_critical": "true",
        },
    ]


def _outcome_rows() -> list[dict[str, object]]:
    coordinates = [
        ("direct_synthetic", 0.05, 0.2, "zero", "", (0.01, 5.0, 0.01, 0.2)),
        ("path_synthetic_step00", 0.0, 0.0, "zero", "", (0.01, 5.0, 0.01, 0.2)),
        ("path_synthetic_step01", 0.025, 0.0, "continuation", "path_synthetic_step00", (0.01, 5.0, 0.01, 0.2)),
        ("path_synthetic_step02", 0.05, 0.0, "continuation", "path_synthetic_step01", (0.01, 5.0, 0.01, 0.2)),
        ("path_synthetic_step03", 0.05, 0.1, "continuation", "path_synthetic_step02", (0.01, 5.0, 0.01, 0.2)),
        ("path_synthetic_step04", 0.05, 0.2, "continuation", "path_synthetic_step03", (0.01, 5.0, 0.01, 0.2)),
        ("born_synthetic", 0.05, 0.2, "born_seed", "", (0.01, 5.0, 0.01, 0.2)),
        ("refinement_coarse", 0.05, 0.2, "refinement_zero", "", (0.02, 5.0, 0.01, 0.2)),
        ("refinement_half_spacing", 0.05, 0.2, "refinement_zero", "", (0.005, 5.0, 0.01, 0.2)),
        ("refinement_cutoff", 0.05, 0.2, "refinement_zero", "", (0.01, 6.0, 0.01, 0.2)),
        ("refinement_doubled_eta", 0.05, 0.2, "refinement_zero", "", (0.01, 5.0, 0.02, 0.2)),
        ("refinement_half_eta", 0.05, 0.2, "refinement_zero", "", (0.01, 5.0, 0.005, 0.2)),
        ("refinement_mixing", 0.05, 0.2, "refinement_zero", "", (0.01, 5.0, 0.01, 0.1)),
    ]
    diagnostic = {7, 10}
    rows: list[dict[str, object]] = []
    for index, (run_id, g, bias, initialization, parent, control) in enumerate(coordinates):
        rows.append(
            {
                "alias": f"M{index + 1:02d}",
                "run_id": run_id,
                "certification_role": "diagnostic_only" if index in diagnostic else "required",
                "g": g,
                "bias": bias,
                "initialization": initialization,
                "continuation_parent_id": parent,
                "backend": "fft",
                "spacing": control[0],
                "cutoff": control[1],
                "eta_lead": control[2],
                "mixing": control[3],
                "n_frequency": 101,
                "status": "converged",
                "iterations": 3,
                "final_sigma_residual": 1.0e-9,
                "final_green_residual": 2.0e-9,
                "final_identity_residual": 3.0e-9,
                "left_current": bias * (1.0 + g),
                "right_current": -bias * (1.0 + g),
                "contact_balance": 1.0e-12,
                "collision_power": 2.0e-12,
                "full_balance": 3.0e-12,
                "corrected_cut_currents": "[0.0,0.0]",
                "continuity_residual": "[0.0,0.0]",
                "sigma_pp_nonlocal_ratio": 0.0,
                "sigma_tail_ratio": 1.0e-8,
                "result_release_certified": "false" if index in diagnostic else "true",
                "elapsed_seconds": 1.0 + index,
                "seed_elapsed_seconds": 0.1,
                "checkpoint_resumed": "false",
                "checkpoint_reconstruction_seconds": 0.0,
                "checkpoint_file_sha256": "b" * 64,
                "checkpoint_identity_sha256": "c" * 64,
            }
        )
    return rows


def _history_rows(outcomes: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        for iteration, residual in ((1, "Infinity"), (2, 1.0e-4), (3, 1.0e-9)):
            rows.append(
                {
                    "alias": outcome["alias"],
                    "run_id": outcome["run_id"],
                    "iteration": iteration,
                    "sigma_residual": residual,
                    "green_residual": residual,
                    "identity_residual": residual,
                    "current_abs_error": residual,
                    "current_rel_error": residual,
                    "collision_power": 0.0,
                    "mixing": 0.2,
                    "passed": "true" if iteration == 3 else "false",
                    "consecutive_passes": 3 if iteration == 3 else 0,
                }
            )
    return rows


def _comparison_rows(outcomes: list[dict[str, object]]) -> list[dict[str, object]]:
    definitions = [
        ("zero_vs_born_endpoint", "zero_vs_born", 0, 6, True, True, True),
        ("direct_vs_continuation_endpoint", "direct_vs_continuation", 0, 5, True, True, True),
        ("numerical_refinement_spacing_0p0015625", "numerical_refinement", 0, 7, False, False, False),
        ("numerical_refinement_spacing_0p000390625", "numerical_refinement", 0, 8, True, False, False),
        ("numerical_refinement_cutoff_6", "numerical_refinement", 0, 9, True, False, False),
        ("numerical_refinement_eta_lead_0p003125", "numerical_refinement", 0, 10, False, True, False),
        ("numerical_refinement_eta_lead_0p00078125", "numerical_refinement", 0, 11, True, True, False),
        ("numerical_refinement_mixing_0p1", "numerical_refinement", 0, 12, True, True, True),
    ]
    rows = []
    for name, kind, left, right, critical, compare_fields, gate_fields in definitions:
        route = kind != "numerical_refinement"
        rows.append(
            {
                "name": name,
                "kind": kind,
                "left_alias": outcomes[left]["alias"],
                "right_alias": outcomes[right]["alias"],
                "left_run_id": outcomes[left]["run_id"],
                "right_run_id": outcomes[right]["run_id"],
                "current_scope": "symmetrized_transport",
                "field_comparison_applicable": str(compare_fields).lower(),
                "field_gate_applicable": str(gate_fields).lower(),
                "current_gate_applicable": "true",
                "field_rtol": 1.0e-8,
                "field_atol": 1.0e-12,
                "current_rtol": 1.0e-8 if route else 5.0e-3,
                "current_atol": 1.0e-12,
                "max_field_gate_ratio": 0.2 if gate_fields else "",
                "current_max_absolute_error": 1.0e-12,
                "current_allowed_absolute_error": 4.0e-12,
                "current_gate_ratio": 0.25,
                "current_max_relative_error": 1.0e-9,
                "sigma_tail_left": 1.0e-8,
                "sigma_tail_right": 1.0e-8,
                "linewidth_gate_ratio": "" if route else 0.4,
                "shift_gate_ratio": "" if route else 0.5,
                "certification_critical": str(critical).lower(),
                "passed": "true",
                "failures": "[]",
            }
        )
    return rows


def _conservation_rows(outcomes: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "name": "interacting_equilibrium",
            "alias": outcomes[3]["alias"],
            "run_id": outcomes[3]["run_id"],
            "equilibrium": "true",
            "left_current": 0.0,
            "right_current": 0.0,
            "allowed_error": 1.0e-10,
            "contact_balance": 1.0e-11,
            "contact_gate_ratio": 0.1,
            "collision_power": 1.0e-11,
            "collision_gate_ratio": 0.1,
            "full_balance": 2.0e-11,
            "corrected_cut_current_error": 1.0e-11,
            "continuity_residual_error": 1.0e-11,
            "sigma_pp_nonlocal_ratio": 0.0,
            "spatial_current_gate": "raw_harmonic_bond_current_block_local_sigma",
            "passed": "true",
        },
        {
            "name": "interacting_nonequilibrium",
            "alias": outcomes[0]["alias"],
            "run_id": outcomes[0]["run_id"],
            "equilibrium": "false",
            "left_current": 0.2,
            "right_current": -0.2,
            "allowed_error": 1.0e-5,
            "contact_balance": 2.0e-6,
            "contact_gate_ratio": 0.2,
            "collision_power": 3.0e-6,
            "collision_gate_ratio": 0.3,
            "full_balance": 5.0e-6,
            "corrected_cut_current_error": 4.0e-6,
            "continuity_residual_error": 2.0e-6,
            "sigma_pp_nonlocal_ratio": 0.2,
            "spatial_current_gate": "corrected_eom_cut_current_nonlocal_sigma",
            "passed": "true",
        },
    ]


def _passivity_rows(outcomes: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    spectra = ("green", "sigma_pp", "sigma_total", "sigma_left", "sigma_right")
    for outcome in outcomes:
        interacting_applicable = outcome["alias"] != "M01"
        for spectrum in spectra:
            retarded_applicable = bool(
                interacting_applicable
                or spectrum in {"sigma_left", "sigma_right"}
            )
            rows.append(
                {
                    "alias": outcome["alias"],
                    "run_id": outcome["run_id"],
                    "spectrum": spectrum,
                    "interacting_retarded_passivity_applicable": 1.0 if interacting_applicable else 0.0,
                    "retarded_passivity_applicable": 1.0 if retarded_applicable else 0.0,
                    "retarded_passivity_violation": 1.0e-12 if retarded_applicable else 1.0e-3,
                    "lesser_correlation_positivity_applicable": 1.0,
                    "lesser_correlation_positivity_violation": 2.0e-12,
                    "greater_correlation_positivity_applicable": 1.0,
                    "greater_correlation_positivity_violation": 3.0e-12,
                }
            )
    return rows


def _backend_rows() -> list[dict[str, object]]:
    rows = []
    for index, (name, hilbert) in enumerate(
        (("serial", "direct"), ("threaded", "direct"), ("fft", "fft"))
    ):
        rows.append(
            {
                "record_type": "backend",
                "name": name,
                "reference_backend": "",
                "candidate_backend": "",
                "effective_backend": name,
                "hilbert_backend": hilbert,
                "iterations": 3,
                "elapsed_seconds": 1.0 + index,
                "max_field_gate_ratio": "",
                "current_gate_ratio": "",
                "passed": "true",
            }
        )
    for candidate in ("threaded", "fft"):
        rows.append(
            {
                "record_type": "comparison",
                "name": f"serial_vs_{candidate}",
                "reference_backend": "serial",
                "candidate_backend": candidate,
                "effective_backend": "",
                "hilbert_backend": "",
                "iterations": "",
                "elapsed_seconds": "",
                "max_field_gate_ratio": 0.2,
                "current_gate_ratio": 0.3,
                "passed": "true",
            }
        )
    return rows


def _timing_rows(open_rows: list[dict[str, object]], outcomes: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in open_rows:
        rows.extend(
            (
                {"category": "open_oracle", "alias": row["alias"], "name": row["name"], "wall_seconds": row["oracle_wall_seconds"], "authoritative_total": "false"},
                {"category": "open_born", "alias": row["alias"], "name": row["name"], "wall_seconds": row["born_wall_seconds"], "authoritative_total": "false"},
            )
        )
    for row in outcomes:
        rows.append(
            {"category": "scba_outcome", "alias": row["alias"], "name": row["run_id"], "wall_seconds": row["elapsed_seconds"], "authoritative_total": "false"}
        )
    rows.append(
        {"category": "multidimensional_total", "alias": "", "name": "total", "wall_seconds": 100.0, "authoritative_total": "true"}
    )
    return rows


def _synthetic_report_bytes(
    csv_payloads: dict[str, bytes],
    tex_payloads: dict[str, bytes],
) -> bytes:
    nested_core = {
        "schema": RENDERER._MULTIDIMENSIONAL_SCHEMA,
        "schema_version": 4,
        "protocol": RENDERER._PROTOCOL,
        "protocol_version": 2,
        "release_certified": True,
    }
    nested = {
        **nested_core,
        "report_sha256": hashlib.sha256(
            json.dumps(
                nested_core,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    outer_core = {
        "schema": RENDERER._REPORT_SCHEMA,
        "schema_version": 4,
        "summary": {
            "validation_tier": "release",
            "release_certified": True,
            "failures": [],
        },
        "multidimensional_scba": nested,
        "synthetic_csv_payloads": {
            relative: base64.b64encode(payload).decode("ascii")
            for relative, payload in sorted(csv_payloads.items())
        },
        "synthetic_tex_payloads": {
            relative: base64.b64encode(payload).decode("ascii")
            for relative, payload in sorted(tex_payloads.items())
        },
    }
    document = {
        **outer_core,
        "report_sha256": hashlib.sha256(
            json.dumps(
                outer_core,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


@pytest.fixture(autouse=True)
def _synthetic_report_deriver(monkeypatch: pytest.MonkeyPatch) -> None:
    def decode(
        document: dict[str, object],
        field: str,
    ) -> tuple[dict[str, bytes], str]:
        encoded = document[field]
        assert isinstance(encoded, dict)
        return (
            {
                str(relative): base64.b64decode(str(payload), validate=True)
                for relative, payload in encoded.items()
            },
            "9" * 64,
        )

    monkeypatch.setattr(
        RENDERER,
        "_derive_csv_payloads_from_report",
        lambda document: decode(document, "synthetic_csv_payloads"),
    )
    monkeypatch.setattr(
        RENDERER,
        "_derive_manuscript_tex_payloads_from_report",
        lambda document: decode(document, "synthetic_tex_payloads"),
    )

def _write_staging(root: Path) -> None:
    open_rows = _open_rows()
    outcomes = _outcome_rows()
    tables = {
        "open_runs": open_rows,
        "open_checks": _open_checks(),
        "scba_outcomes": outcomes,
        "scba_history": _history_rows(outcomes),
        "scba_comparisons": _comparison_rows(outcomes),
        "conservation": _conservation_rows(outcomes),
        "passivity": _passivity_rows(outcomes),
        "backend_oracle": _backend_rows(),
        "timings": _timing_rows(open_rows, outcomes),
    }
    files: dict[str, bytes] = {}
    for name, rows in tables.items():
        files[f"csv/{name}.csv"] = _csv_bytes(RENDERER._CSV_SCHEMAS[name], rows)
    files["csv/harmonic_preflight.csv"] = b"record_type,name\npoint,synthetic\n"
    files["csv/provenance.csv"] = b"commit,environment_sha256\nsynthetic,synthetic\n"
    for relative in RENDERER._EXPECTED_STAGED_ARTIFACTS:
        if relative.startswith("tex/"):
            files[relative] = b"% synthetic fixture only\n"
        elif relative.startswith("source/"):
            files[relative] = b"synthetic source evidence\n"
    files["production_toy_gate.json"] = _synthetic_report_bytes(
        {
            relative: payload
            for relative, payload in files.items()
            if relative.startswith("csv/")
        },
        {
            relative: files[relative]
            for relative in RENDERER._MANUSCRIPT_TEX_ARTIFACTS
        },
    )
    assert set(files) == set(RENDERER._EXPECTED_STAGED_ARTIFACTS)

    report_sha = hashlib.sha256(files["production_toy_gate.json"]).hexdigest()
    report = json.loads(files["production_toy_gate.json"])
    core = {
        "schema": RENDERER._RECEIPT_SCHEMA,
        "schema_version": 2,
        "implementation": {
            "name": RENDERER._IMPORTER_PATH.name,
            "version": RENDERER._IMPORTER_VERSION,
            "source_sha256": "9" * 64,
        },
        "source": {
            "commit": "a" * 40,
            "attempt": 1,
            "report_relative_path": "production_toy_gate.json",
            "remote_report_relative_path": (
                "attempts/attempt-001/production_toy_gate.json"
            ),
            "report_file_sha256": report_sha,
            "report_body_sha256": report["report_sha256"],
            "nested_report_body_sha256": report["multidimensional_scba"]["report_sha256"],
            "outer_sha256sums_file_sha256": "5" * 64,
            "fetch_receipt_sha256": "4" * 64,
            "sha256sums_sha256": "5" * 64,
            "transport_receipt_schema": (
                RENDERER._REMOTE_TRANSPORT_RECEIPT_SCHEMA
            ),
            "transport_receipt_created_utc": "synthetic-fixture",
            "fetched_utc": "synthetic-fixture",
            "git_tree": "1" * 40,
            "source_tree_sha256": "2" * 64,
            "environment_sha256": "3" * 64,
        },
        "trusted_inputs": {
            "expected_commit": "a" * 40,
            "expected_environment_lock_sha256": "3" * 64,
            "expected_fetch_receipt_sha256": "4" * 64,
            "expected_sha256sums_sha256": "5" * 64,
        },
        "release_contract": {
            "validation_tier": "release",
            "release_certified": True,
            "protocol": RENDERER._PROTOCOL,
            "protocol_version": 2,
            "schema_version": 4,
            "open_system_run_count": 13,
            "multidimensional_outcome_count": 13,
            "checkpoint_count": 13,
        },
        "checkpoint_validation": {
            "mode": "identity_and_integrity_load_without_scba_restore",
            "count": 13,
            "records": [{"run_id": f"synthetic-{index:02d}"} for index in range(13)],
        },
        "staged_artifacts": {
            relative: {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
            for relative, payload in sorted(files.items())
        },
    }
    receipt = {
        **core,
        "receipt_sha256": hashlib.sha256(
            json.dumps(core, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    files["import_receipt.json"] = (json.dumps(receipt, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    root.mkdir()
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "SHA256SUMS").write_text(
        "\n".join(f"{hashlib.sha256(payload).hexdigest()}  {relative}" for relative, payload in sorted(files.items())) + "\n",
        encoding="ascii",
    )


def _rewrite_sha256sums(root: Path) -> None:
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and path != root / "SHA256SUMS"
    }
    (root / "SHA256SUMS").write_text(
        "\n".join(f"{hashlib.sha256(payload).hexdigest()}  {relative}" for relative, payload in sorted(files.items())) + "\n",
        encoding="ascii",
    )


def _receipt_file_sha256(root: Path) -> str:
    return hashlib.sha256((root / "import_receipt.json").read_bytes()).hexdigest()


def _render(root: Path, output: Path):
    return RENDERER.render_figures(
        root,
        output,
        expected_import_receipt_sha256=_receipt_file_sha256(root),
    )


def _reseal_receipt_only(root: Path) -> None:
    report_path = root / "production_toy_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    receipt_path = root / "import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source"]["report_file_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    receipt["source"]["report_body_sha256"] = report["report_sha256"]
    receipt["source"]["nested_report_body_sha256"] = report[
        "multidimensional_scba"
    ]["report_sha256"]
    for relative, record in receipt["staged_artifacts"].items():
        payload = (root / relative).read_bytes()
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["bytes"] = len(payload)
    core = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            core,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rewrite_sha256sums(root)


def _reseal_staging(root: Path) -> None:
    report_path = root / "production_toy_gate.json"
    report_path.write_bytes(
        _synthetic_report_bytes(
            {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in (root / "csv").glob("*.csv")
            },
            {
                relative: (root / relative).read_bytes()
                for relative in RENDERER._MANUSCRIPT_TEX_ARTIFACTS
            },
        )
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    receipt_path = root / "import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source"]["report_file_sha256"] = hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    receipt["source"]["report_body_sha256"] = report["report_sha256"]
    receipt["source"]["nested_report_body_sha256"] = report[
        "multidimensional_scba"
    ]["report_sha256"]
    for relative, record in receipt["staged_artifacts"].items():
        payload = (root / relative).read_bytes()
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["bytes"] = len(payload)
    core = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            core,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    receipt_path.write_text(
        json.dumps(
            receipt,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _rewrite_sha256sums(root)


def _rewrite_csv_cell(
    root: Path,
    table: str,
    *,
    row_index: int,
    column: str,
    value: object,
) -> None:
    path = root / "csv" / f"{table}.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        headers = tuple(reader.fieldnames)
        rows = list(reader)
    rows[row_index][column] = value
    path.write_bytes(_csv_bytes(headers, rows))
    _reseal_staging(root)


def test_renderer_creates_three_native_figure_pairs_atomically_and_idempotently(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "figures"
    _write_staging(staging)
    result = _render(staging, output)
    assert result["status"] == "created"
    assert result["certification_decision_performed"] is False

    expected = {
        "SHA256SUMS",
        "render_manifest.json",
        *(f"{name}.{extension}" for name in RENDERER._FIGURE_SPECS for extension in ("pdf", "png")),
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "render_manifest.json").read_text(encoding="utf-8"))
    assert manifest["scope"]["physics_solver_executed"] is False
    assert manifest["renderer_version"] == RENDERER._RENDERER_VERSION
    assert manifest["scope"]["certification_decision_performed"] is False
    assert manifest["source"]["caller_expected_import_receipt_sha256"] == (
        _receipt_file_sha256(staging)
    )
    assert manifest["source"]["attempt"] == 1
    assert manifest["source"]["report_relative_path"] == (
        "production_toy_gate.json"
    )
    assert manifest["source"]["remote_report_relative_path"] == (
        "attempts/attempt-001/production_toy_gate.json"
    )
    assert manifest["source"]["transport_receipt_schema"] == (
        RENDERER._REMOTE_TRANSPORT_RECEIPT_SCHEMA
    )
    binding = manifest["source"]["raw_report_binding"]
    assert binding["report_file_sha256"] == hashlib.sha256(
        (staging / "production_toy_gate.json").read_bytes()
    ).hexdigest()
    assert set(binding["plotted_csv_sha256"]) == set(
        RENDERER._PLOTTED_CSV_TABLES
    )
    assert set(binding["manuscript_tex_sha256"]) == set(
        RENDERER._MANUSCRIPT_TEX_ARTIFACTS
    )
    for relative in RENDERER._MANUSCRIPT_TEX_ARTIFACTS:
        assert binding["manuscript_tex_sha256"][relative] == hashlib.sha256(
            (staging / relative).read_bytes()
        ).hexdigest()
    assert manifest["style"]["png_dpi"] == 450
    assert manifest["style"]["png_color_mode"] == "RGB"
    assert manifest["style"]["pdf_artwork"] == (
        "vector_no_image_or_inline_image_records"
    )
    assert manifest["style"]["maximum_figure_bytes"] == 10 * 1024 * 1024
    assert set(manifest["style"]["pdf_structural_validation"]) == {
        "pdfimages",
        "pdffonts",
    }
    assert manifest["style"]["palette_standard"] == (
        "Okabe_Ito_colorblind_accessible"
    )
    assert manifest["style"]["metric_shape_redundancy"] is True
    assert manifest["style"]["body_font_points"] == [5.0, 7.0]
    assert manifest["style"]["panel_labels"] == "lowercase_bold"
    assert manifest["style"]["gridlines"] is False
    assert manifest["style"]["not_applicable_rendering"] == "gray_NA"
    assert manifest["style"]["critical_rendering"] == "filled_circle"
    assert manifest["style"]["diagnostic_rendering"] == "open_diamond"
    for name in RENDERER._FIGURE_SPECS:
        pdf = (output / f"{name}.pdf").read_bytes()
        RENDERER._validate_pdf_bytes(pdf, name=f"test {name} PDF")
        pdf_record = manifest["figures"][name]["pdf"]
        assert pdf_record["vector"] is True
        assert pdf_record["raster_image_xobjects"] is False
        assert pdf_record["embedded_truetype_fonts"] is True
        assert len(manifest["figures"][name]["panels"]) == 4
        with Image.open(output / f"{name}.png") as image:
            image.load()
            assert image.mode == "RGB"
            assert image.size == (3150, 2250)
            dpi = image.info["dpi"]
            assert dpi[0] == pytest.approx(450, abs=0.5)
            assert dpi[1] == pytest.approx(450, abs=0.5)
        assert manifest["figures"][name]["png"][
            "fully_verified_and_loaded"
        ] is True
    hashes_before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}
    second = _render(staging, output)
    assert second["status"] == "unchanged"
    assert hashes_before == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir()}


def test_panel_labels_are_lowercase_bold_eight_point_sans_serif() -> None:
    font = RENDERER._resolve_font(None, None)
    figure, axis = RENDERER.plt.subplots()
    try:
        for index in range(4):
            RENDERER._panel_label(axis, index, str(font["family"]))
        assert [item.get_text() for item in axis.texts] == ["a", "b", "c", "d"]
        for item in axis.texts:
            assert item.get_fontweight() == "bold"
            assert item.get_fontsize() == pytest.approx(8.0)
            assert item.get_fontfamily() == [font["family"]]
    finally:
        RENDERER.plt.close(figure)


def test_unrelated_font_family_is_refused() -> None:
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    assert regular.is_file() and bold.is_file()
    with pytest.raises(RENDERER.FigureRenderError, match="Arial/Helvetica"):
        RENDERER._resolve_font(regular, bold)


def test_numeric_zero_applicability_is_masked_even_with_large_raw_value(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write_staging(staging)
    data = RENDERER._load_data(staging)
    aliases = [item["alias"] for item in data["outcomes"]]
    matrix, spectra = RENDERER._passivity_matrix(data["passivity"], aliases, kind="retarded")
    assert spectra[0] == "green"
    assert bool(matrix.mask[0, 0]) is True
    assert bool(matrix.mask[0, 1]) is True
    assert bool(matrix.mask[0, 2]) is True
    assert bool(matrix.mask[0, 3]) is False
    assert bool(matrix.mask[0, 4]) is False
    assert float(matrix.data[0, 0]) == pytest.approx(1.0e-3)
    correlation, _ = RENDERER._passivity_matrix(
        data["passivity"], aliases, kind="correlation"
    )
    assert not bool(correlation.mask.any())


def test_comparison_metadata_preserves_gate_applicability(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write_staging(staging)
    data = RENDERER._load_data(staging)
    comparisons = {item["name"]: item for item in data["comparisons"]}

    assert set(comparisons) == set(RENDERER._EXPECTED_COMPARISON_CONTRACT)
    for name, policy in RENDERER._EXPECTED_COMPARISON_CONTRACT.items():
        item = comparisons[name]
        assert item["kind"] == policy["kind"]
        assert item["critical"] is policy["critical"]
        assert item["current_scope"] == policy["current_scope"]
        assert item["field_comparison_applicable"] is policy[
            "field_comparison_applicable"
        ]
        assert item["field_gate_applicable"] is policy[
            "field_gate_applicable"
        ]
        assert item["current_gate_applicable"] is True
        assert item["field_rtol"] == policy["field_rtol"]
        assert item["field_atol"] == policy["field_atol"]
        assert item["current_rtol"] == policy["current_rtol"]
        assert item["current_atol"] == policy["current_atol"]

    coarse = comparisons["numerical_refinement_spacing_0p0015625"]
    assert coarse["field_comparison_applicable"] is False
    assert coarse["field_gate_applicable"] is False
    assert coarse["field_ratio"] is None

    eta = comparisons["numerical_refinement_eta_lead_0p00078125"]
    assert eta["field_comparison_applicable"] is True
    assert eta["field_gate_applicable"] is False
    assert eta["field_ratio"] is None

    mixing = comparisons["numerical_refinement_mixing_0p1"]
    assert mixing["field_comparison_applicable"] is True
    assert mixing["field_gate_applicable"] is True
    assert mixing["current_scope"] == "symmetrized_transport"
    assert mixing["current_rtol"] == pytest.approx(5.0e-3)
    assert mixing["current_ratio"] == pytest.approx(0.25)


def test_resealed_wrong_refinement_applicability_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    _rewrite_csv_cell(
        staging,
        "scba_comparisons",
        row_index=6,
        column="field_gate_applicable",
        value="true",
    )
    _rewrite_csv_cell(
        staging,
        "scba_comparisons",
        row_index=6,
        column="max_field_gate_ratio",
        value=0.2,
    )
    with pytest.raises(RENDERER.FigureRenderError, match="exact independent policy"):
        _render(staging, output)
    assert not output.exists()


def test_existing_output_cannot_bypass_csv_semantic_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "existing-output"
    _write_staging(staging)
    output.mkdir()
    _rewrite_csv_cell(
        staging,
        "scba_comparisons",
        row_index=6,
        column="field_gate_applicable",
        value="true",
    )
    monkeypatch.setattr(
        RENDERER,
        "_verify_output",
        lambda *args, **kwargs: pytest.fail(
            "existing output was examined before staged CSV semantics"
        ),
    )
    with pytest.raises(RENDERER.FigureRenderError, match="field gate metric"):
        _render(staging, output)


def test_resealed_inconsistent_passivity_scope_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    _rewrite_csv_cell(
        staging,
        "passivity",
        row_index=3,
        column="retarded_passivity_applicable",
        value=0.0,
    )
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="retarded-passivity applicability is inconsistent",
    ):
        _render(staging, output)
    assert not output.exists()


def test_resealed_inconsistent_current_ratio_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    _rewrite_csv_cell(
        staging,
        "scba_comparisons",
        row_index=0,
        column="current_gate_ratio",
        value=0.5,
    )
    with pytest.raises(RENDERER.FigureRenderError, match="current gate ratio is inconsistent"):
        _render(staging, output)
    assert not output.exists()


def test_tampered_csv_is_refused_before_output(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    with (staging / "csv" / "open_runs.csv").open("ab") as stream:
        stream.write(b"tamper\n")
    with pytest.raises(RENDERER.FigureRenderError, match="SHA256 mismatch"):
        _render(staging, output)
    assert not output.exists()


def test_receipt_body_tamper_is_refused_even_when_outer_sha_is_updated(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    receipt_path = staging / "import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source"]["attempt"] = 2
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _rewrite_sha256sums(staging)
    with pytest.raises(RENDERER.FigureRenderError, match="canonical body SHA256"):
        _render(staging, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("attempt", "transport_schema", "report_relative_path"),
    (
        (
            0,
            RENDERER._REMOTE_TRANSPORT_RECEIPT_SCHEMA,
            "attempts/attempt-000/production_toy_gate.json",
        ),
        (
            1,
            RENDERER._REMOTE_TRANSPORT_RECEIPT_SCHEMA,
            "production_toy_gate.json",
        ),
        (
            2,
            RENDERER._LOCAL_TRANSPORT_RECEIPT_SCHEMA,
            "attempts/attempt-002/production_toy_gate.json",
        ),
        (2, "unknown.transport.receipt", "production_toy_gate.json"),
    ),
)
def test_resealed_receipt_rejects_invalid_attempt_or_transport_specific_report_path(
    tmp_path: Path,
    attempt: int,
    transport_schema: str,
    report_relative_path: str,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    receipt_path = staging / "import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source"]["attempt"] = attempt
    receipt["source"]["transport_receipt_schema"] = transport_schema
    receipt["source"]["remote_report_relative_path"] = report_relative_path
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _reseal_receipt_only(staging)
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="attempt|transport schema|report path",
    ):
        _render(staging, output)
    assert not output.exists()


def test_resealed_local_attempt_two_source_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    _write_staging(staging)
    receipt_path = staging / "import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source"]["attempt"] = 2
    receipt["source"]["transport_receipt_schema"] = (
        RENDERER._LOCAL_TRANSPORT_RECEIPT_SCHEMA
    )
    receipt["source"]["report_relative_path"] = "production_toy_gate.json"
    receipt["source"]["fetched_utc"] = None
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _reseal_receipt_only(staging)

    with pytest.raises(
        RENDERER.FigureRenderError,
        match="canonical remote transport schema",
    ):
        RENDERER._validate_staging(
            staging,
            expected_import_receipt_sha256=_receipt_file_sha256(staging),
        )


def test_supplied_staging_root_symlink_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    link = tmp_path / "staging-link"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    link.symlink_to(staging, target_is_directory=True)
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="staging root is a symlink",
    ):
        _render(link, output)
    assert not output.exists()


def test_supplied_output_symlink_is_refused_without_modifying_target(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "existing-target"
    link = tmp_path / "output-link"
    _write_staging(staging)
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="figure output is a symlink",
    ):
        _render(staging, link)
    assert link.is_symlink()
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("entry_kind", ("empty_directory", "fifo"))
def test_staging_tree_rejects_unmanifested_directory_or_special_entry(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    unexpected = staging / (
        "unexpected-empty" if entry_kind == "empty_directory" else "unexpected-fifo"
    )
    if entry_kind == "empty_directory":
        unexpected.mkdir()
    else:
        os.mkfifo(unexpected)
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="special entry|directory set",
    ):
        _render(staging, output)
    assert not output.exists()


def test_receipt_pin_is_mandatory_and_wrong_pin_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    with pytest.raises(TypeError):
        RENDERER.render_figures(staging, output)
    with pytest.raises(RENDERER.FigureRenderError, match="caller-pinned"):
        RENDERER.render_figures(
            staging,
            output,
            expected_import_receipt_sha256="0" * 64,
        )
    assert not output.exists()


def test_fourth_trust_anchor_must_match_fetched_sha256sums(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    receipt_path = staging / "import_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["trusted_inputs"]["expected_sha256sums_sha256"] = "6" * 64
    receipt["source"]["sha256sums_sha256"] = "6" * 64
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _reseal_receipt_only(staging)
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="trusted inputs do not bind",
    ):
        _render(staging, output)
    assert not output.exists()


def test_stale_outer_report_body_digest_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    report_path = staging / "production_toy_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["tamper"] = True
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _reseal_receipt_only(staging)
    with pytest.raises(RENDERER.FigureRenderError, match="outer canonical"):
        _render(staging, output)
    assert not output.exists()


def test_stale_nested_digest_is_refused_after_valid_outer_reseal(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    report_path = staging / "production_toy_gate.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["multidimensional_scba"]["tamper"] = True
    outer_core = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            outer_core,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _reseal_receipt_only(staging)
    with pytest.raises(RENDERER.FigureRenderError, match="nested.*canonical"):
        _render(staging, output)
    assert not output.exists()


def test_resealed_csv_divergence_from_raw_report_is_refused(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    path = staging / "csv" / "open_runs.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    _reseal_receipt_only(staging)
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="canonical raw-report derivative",
    ):
        _render(staging, output)
    assert not output.exists()



@pytest.mark.parametrize("relative", RENDERER._MANUSCRIPT_TEX_ARTIFACTS)
def test_resealed_manuscript_tex_divergence_from_raw_report_is_refused(
    tmp_path: Path,
    relative: str,
) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "must-not-exist"
    _write_staging(staging)
    path = staging / relative
    path.write_bytes(path.read_bytes() + b"% adversarial mutation\n")
    _reseal_receipt_only(staging)
    with pytest.raises(
        RENDERER.FigureRenderError,
        match="manuscript TeX .*canonical raw-report derivative",
    ):
        _render(staging, output)
    assert not output.exists()
def test_structural_pdf_validator_rejects_raster_image_records() -> None:
    figure, axis = RENDERER.plt.subplots()
    try:
        axis.imshow([[0.0, 1.0], [1.0, 0.0]])
        stream = io.BytesIO()
        figure.savefig(stream, format="pdf")
    finally:
        RENDERER.plt.close(figure)
    with pytest.raises(RENDERER.FigureRenderError, match="image or inline-image"):
        RENDERER._validate_pdf_bytes(stream.getvalue(), name="raster fixture")


def test_structural_pdf_validator_rejects_unembedded_type1_font() -> None:
    with RENDERER.plt.rc_context(
        {
            "pdf.use14corefonts": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    ):
        figure, axis = RENDERER.plt.subplots()
        try:
            axis.plot([0.0, 1.0], [0.0, 1.0])
            axis.set_xlabel("x (1)")
            stream = io.BytesIO()
            figure.savefig(stream, format="pdf")
        finally:
            RENDERER.plt.close(figure)
    with pytest.raises(RENDERER.FigureRenderError, match="embedded TrueType"):
        RENDERER._validate_pdf_bytes(stream.getvalue(), name="Type-1 fixture")


def test_png_validator_fully_rejects_truncated_payload() -> None:
    stream = io.BytesIO()
    image = Image.new("RGB", RENDERER._EXPECTED_PIXELS, (255, 255, 255))
    image.save(stream, format="PNG", dpi=(450, 450))
    payload = stream.getvalue()
    with pytest.raises(RENDERER.FigureRenderError, match="verification/load"):
        RENDERER._validate_png_bytes(
            payload[: len(payload) // 2],
            name="truncated PNG fixture",
        )


def test_atomic_directory_publish_never_replaces_racing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ready"
    destination = tmp_path / "winner"
    source.mkdir()
    destination.mkdir()
    (source / "payload").write_text("new", encoding="utf-8")
    (destination / "payload").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        RENDERER._rename_directory_noreplace(source, destination)
    assert (destination / "payload").read_text(encoding="utf-8") == "existing"
    assert (source / "payload").read_text(encoding="utf-8") == "new"


def test_canonical_importer_exposes_hashed_derivation_hooks() -> None:
    importer, sha256 = RENDERER._load_import_deriver()
    assert callable(importer._derive_csv_payloads)
    assert callable(importer._derive_manuscript_tex_payloads)
    assert sha256 == hashlib.sha256(
        RENDERER._IMPORTER_PATH.read_bytes()
    ).hexdigest()
