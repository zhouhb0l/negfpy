#!/usr/bin/env python3
"""Render publication figures from a verified toy-gate staging directory.

This is a read-only consumer of ``import_production_toy_gate.py`` output.  It
does not read remote checkpoints, run Born/SCBA calculations, edit the LaTeX
report, or decide certification.  Before reading any CSV it verifies the
staging ``SHA256SUMS`` file, the strict import receipt, and every receipt-level
artifact identity.  It then writes three assembled figures to a fresh output
directory using one atomic directory rename.

An existing output directory is accepted only when its manifest, source
identity, renderer identity, and every output hash are intact.  It is never
overwritten or repaired in place.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import ctypes
import csv
import errno
import hashlib
import hmac
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import string
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import colors as mpl_colors  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RECEIPT_SCHEMA = "negfpy.anharmonic.production_toy_gate_import_receipt"
_REMOTE_TRANSPORT_RECEIPT_SCHEMA = "negfpy.remote_validation.fetch_receipt.v1"
_LOCAL_TRANSPORT_RECEIPT_SCHEMA = (
    "negfpy.local_validation.publication_receipt.v1"
)
_MANIFEST_SCHEMA = "negfpy.anharmonic.production_toy_gate_figure_render"
_PROTOCOL = "multidimensional_toy_gate_v2"
_DPI = 450
_RENDERER_VERSION = "2"
_IMPORTER_VERSION = "3"
_FIGURE_SIZE = (7.0, 5.0)
_EXPECTED_PIXELS = (3150, 2250)
_MAX_FIGURE_BYTES = 10 * 1024 * 1024
_IMPORTER_PATH = Path(__file__).with_name("import_production_toy_gate.py")
_PDFIMAGES_PATH = Path("/usr/bin/pdfimages")
_PDFFONTS_PATH = Path("/usr/bin/pdffonts")
_REPORT_SCHEMA = "negfpy.anharmonic.production_toy_gate"
_MULTIDIMENSIONAL_SCHEMA = (
    "negfpy.anharmonic.multidimensional_dense_toy_release"
)
_PLOTTED_CSV_TABLES = (
    "open_runs",
    "open_checks",
    "scba_outcomes",
    "scba_history",
    "scba_comparisons",
    "conservation",
    "passivity",
    "backend_oracle",
    "timings",
)
_MANUSCRIPT_TEX_ARTIFACTS = (
    "tex/harmonic_preflight_rows.tex",
    "tex/cost_accuracy_rows.tex",
    "tex/weak_coupling_summary.tex",
    "tex/multidimensional_weak_coupling_summary.tex",
)
_EXPECTED_COMPARISON_CONTRACT = MappingProxyType(
    {
        name: MappingProxyType(
            {
                "kind": (
                    "zero_vs_born"
                    if name == "zero_vs_born_endpoint"
                    else "direct_vs_continuation"
                    if name == "direct_vs_continuation_endpoint"
                    else "numerical_refinement"
                ),
                "critical": name
                in {
                    "zero_vs_born_endpoint",
                    "direct_vs_continuation_endpoint",
                    "numerical_refinement_spacing_0p000390625",
                    "numerical_refinement_cutoff_6",
                    "numerical_refinement_eta_lead_0p00078125",
                    "numerical_refinement_mixing_0p1",
                },
                "current_scope": "symmetrized_transport",
                "field_comparison_applicable": name
                in {
                    "zero_vs_born_endpoint",
                    "direct_vs_continuation_endpoint",
                    "numerical_refinement_eta_lead_0p003125",
                    "numerical_refinement_eta_lead_0p00078125",
                    "numerical_refinement_mixing_0p1",
                },
                "field_gate_applicable": name
                in {
                    "zero_vs_born_endpoint",
                    "direct_vs_continuation_endpoint",
                    "numerical_refinement_mixing_0p1",
                },
                "current_gate_applicable": True,
                "field_rtol": 1.0e-8,
                "field_atol": 1.0e-12,
                "current_rtol": (
                    1.0e-8
                    if name
                    in {
                        "zero_vs_born_endpoint",
                        "direct_vs_continuation_endpoint",
                    }
                    else 5.0e-3
                ),
                "current_atol": 1.0e-12,
            }
        )
        for name in (
            "zero_vs_born_endpoint",
            "direct_vs_continuation_endpoint",
            "numerical_refinement_spacing_0p0015625",
            "numerical_refinement_spacing_0p000390625",
            "numerical_refinement_cutoff_6",
            "numerical_refinement_eta_lead_0p003125",
            "numerical_refinement_eta_lead_0p00078125",
            "numerical_refinement_mixing_0p1",
        )
    }
)

_PALETTE = MappingProxyType(
    {
        "blue": "#0072B2",
        "orange": "#E69F00",
        "green": "#009E73",
        "vermillion": "#D55E00",
        "purple": "#CC79A7",
        "sky": "#56B4E9",
        "black": "#000000",
        "gray": "#9E9E9E",
        "na": "#D0D0D0",
    }
)

_CSV_SCHEMAS = MappingProxyType(
    {
        "open_runs": (
            "alias",
            "name",
            "temperature",
            "g",
            "grid_step",
            "grid_n_points",
            "cutoff",
            "eta_lead",
            "oracle_parent_linewidth",
            "production_parent_linewidth",
            "linewidth_gate_ratio",
            "oracle_parent_shift",
            "production_parent_shift",
            "shift_gate_ratio",
            "agreement_passed",
            "weak_coupling_passed",
            "passivity_passed",
            "oracle_wall_seconds",
            "born_wall_seconds",
        ),
        "open_checks": (
            "kind",
            "check",
            "source",
            "observable",
            "temperature",
            "g_or_control",
            "base_run",
            "candidate_run",
            "gate_ratio",
            "passed",
            "certification_critical",
        ),
        "scba_outcomes": (
            "alias",
            "run_id",
            "certification_role",
            "g",
            "bias",
            "initialization",
            "continuation_parent_id",
            "backend",
            "spacing",
            "cutoff",
            "eta_lead",
            "mixing",
            "n_frequency",
            "status",
            "iterations",
            "final_sigma_residual",
            "final_green_residual",
            "final_identity_residual",
            "left_current",
            "right_current",
            "contact_balance",
            "collision_power",
            "full_balance",
            "corrected_cut_currents",
            "continuity_residual",
            "sigma_pp_nonlocal_ratio",
            "sigma_tail_ratio",
            "result_release_certified",
            "elapsed_seconds",
            "seed_elapsed_seconds",
            "checkpoint_resumed",
            "checkpoint_reconstruction_seconds",
            "checkpoint_file_sha256",
            "checkpoint_identity_sha256",
        ),
        "scba_history": (
            "alias",
            "run_id",
            "iteration",
            "sigma_residual",
            "green_residual",
            "identity_residual",
            "current_abs_error",
            "current_rel_error",
            "collision_power",
            "mixing",
            "passed",
            "consecutive_passes",
        ),
        "scba_comparisons": (
            "name",
            "kind",
            "left_alias",
            "right_alias",
            "left_run_id",
            "right_run_id",
            "current_scope",
            "field_comparison_applicable",
            "field_gate_applicable",
            "current_gate_applicable",
            "field_rtol",
            "field_atol",
            "current_rtol",
            "current_atol",
            "max_field_gate_ratio",
            "current_max_absolute_error",
            "current_allowed_absolute_error",
            "current_gate_ratio",
            "current_max_relative_error",
            "sigma_tail_left",
            "sigma_tail_right",
            "linewidth_gate_ratio",
            "shift_gate_ratio",
            "certification_critical",
            "passed",
            "failures",
        ),
        "conservation": (
            "name",
            "alias",
            "run_id",
            "equilibrium",
            "left_current",
            "right_current",
            "allowed_error",
            "contact_balance",
            "contact_gate_ratio",
            "collision_power",
            "collision_gate_ratio",
            "full_balance",
            "corrected_cut_current_error",
            "continuity_residual_error",
            "sigma_pp_nonlocal_ratio",
            "spatial_current_gate",
            "passed",
        ),
        "passivity": (
            "alias",
            "run_id",
            "spectrum",
            "interacting_retarded_passivity_applicable",
            "retarded_passivity_applicable",
            "retarded_passivity_violation",
            "lesser_correlation_positivity_applicable",
            "lesser_correlation_positivity_violation",
            "greater_correlation_positivity_applicable",
            "greater_correlation_positivity_violation",
        ),
        "backend_oracle": (
            "record_type",
            "name",
            "reference_backend",
            "candidate_backend",
            "effective_backend",
            "hilbert_backend",
            "iterations",
            "elapsed_seconds",
            "max_field_gate_ratio",
            "current_gate_ratio",
            "passed",
        ),
        "timings": (
            "category",
            "alias",
            "name",
            "wall_seconds",
            "authoritative_total",
        ),
    }
)

_FIGURE_SPECS = MappingProxyType(
    {
        "figure_open_oracle_born": (
            "linewidth versus signed coupling",
            "shift versus signed coupling",
            "oracle/Born agreement",
            "open-system validation ratios",
        ),
        "figure_scba_validation": (
            "L-shaped transport path",
            "endpoint fixed-point convergence",
            "conservation gates",
            "numerical refinements",
        ),
        "figure_passivity_backend_timing": (
            "retarded passivity applicability",
            "correlation positivity applicability",
            "bounded backend oracle",
            "wall-time distribution",
        ),
    }
)

_EXPECTED_STAGED_ARTIFACTS = frozenset(
    {
        "production_toy_gate.json",
        "source/SHA256SUMS",
        "source/fetch-receipt.json",
        "source/current-attempt.json",
        "source/attempt-publication.json",
        "source/fetched-artifact-index.json",
        *(f"csv/{name}.csv" for name in (
            "open_runs",
            "open_checks",
            "scba_outcomes",
            "scba_history",
            "scba_comparisons",
            "conservation",
            "passivity",
            "harmonic_preflight",
            "backend_oracle",
            "timings",
            "provenance",
        )),
        *(f"tex/{name}.tex" for name in (
            "release_summary",
            "open_run_rows",
            "scba_outcome_rows",
            "comparison_rows",
            "conservation_rows",
            "passivity_rows",
            "provenance_rows",
            "run_id_rows",
        )),
        *_MANUSCRIPT_TEX_ARTIFACTS,
    }
)


class FigureRenderError(RuntimeError):
    """Raised when staged evidence or rendered output is unsafe/invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FigureRenderError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _reject_constant(token: str) -> None:
    raise FigureRenderError(f"strict JSON contains non-finite token {token!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FigureRenderError(
                f"strict JSON contains duplicate object key {key!r}"
            )
        value[key] = item
    return value


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise FigureRenderError(f"{name} is not UTF-8") from error
    try:
        value = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise FigureRenderError(f"{name} is not valid strict JSON") from error
    if not isinstance(value, dict):
        raise FigureRenderError(f"{name} must contain one JSON object")
    return value


def _strict_json_file(path: Path, *, name: str) -> dict[str, Any]:
    return _strict_json_bytes(path.read_bytes(), name=name)


def _canonical_json_bytes(
    value: Mapping[str, Any],
    *,
    ensure_ascii: bool = False,
) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FigureRenderError("cannot encode canonical strict JSON") from error


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FigureRenderError("cannot encode strict JSON manifest") from error


def _body_digest(
    document: Mapping[str, Any],
    key: str,
    *,
    ensure_ascii: bool = False,
) -> str:
    core = {name: value for name, value in document.items() if name != key}
    return _sha256_bytes(
        _canonical_json_bytes(core, ensure_ascii=ensure_ascii)
    )


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FigureRenderError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
    ):
        raise FigureRenderError(f"{name} must be a JSON array")
    return value


def _safe_file(root: Path, relative_text: Any, *, name: str) -> Path:
    _require(
        isinstance(relative_text, str) and bool(relative_text),
        f"{name} must be a non-empty relative path",
    )
    relative = PurePosixPath(relative_text)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "." not in relative.parts,
        f"unsafe relative path for {name}: {relative_text!r}",
    )
    candidate = root.joinpath(*relative.parts)
    _require(not candidate.is_symlink(), f"{name} is a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FigureRenderError(f"{name} is missing") from error
    _require(
        resolved.is_relative_to(root)
        and resolved.is_file()
        and not resolved.is_symlink(),
        f"{name} is not a regular in-root file",
    )
    return resolved


def _parse_sha256sums(
    root: Path,
    *,
    expected_directories: frozenset[str] = frozenset(),
) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    _require(
        manifest.is_file() and not manifest.is_symlink(),
        "staging directory lacks a regular SHA256SUMS file",
    )
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        _require(match is not None, f"invalid SHA256SUMS line {line_number}")
        assert match is not None
        expected, relative = match.groups()
        _require(relative not in entries, f"duplicate SHA entry {relative!r}")
        path = _safe_file(root, relative, name=f"SHA entry {line_number}")
        _require(
            hmac.compare_digest(_sha256_file(path), expected),
            f"SHA256 mismatch for {relative}",
        )
        entries[relative] = expected
    _require(entries, "SHA256SUMS is empty")
    all_entries = list(root.rglob("*"))
    symlinks = [path for path in all_entries if path.is_symlink()]
    _require(not symlinks, "staging directory contains symlinks")
    special = [
        path.relative_to(root).as_posix()
        for path in all_entries
        if not path.is_symlink() and not path.is_file() and not path.is_dir()
    ]
    _require(not special, "artifact tree contains a non-regular special entry")
    directories = {
        path.relative_to(root).as_posix()
        for path in all_entries
        if path.is_dir() and not path.is_symlink()
    }
    _require(
        directories == set(expected_directories),
        "artifact directory set differs from the exact expected tree",
    )
    regular = {
        path.relative_to(root).as_posix()
        for path in all_entries
        if path.is_file() and not path.is_symlink() and path != manifest
    }
    _require(
        set(entries) == regular,
        "SHA256SUMS does not cover exactly the staged regular files",
    )
    return entries


def _validate_staging(
    root: Path,
    *,
    expected_import_receipt_sha256: str,
) -> Mapping[str, Any]:
    _require(
        _is_sha256(expected_import_receipt_sha256),
        "expected import-receipt SHA256 must be 64 lowercase hex digits",
    )
    entries = _parse_sha256sums(
        root,
        expected_directories=frozenset({"csv", "tex", "source"}),
    )
    receipt_path = _safe_file(root, "import_receipt.json", name="import receipt")
    receipt_file_sha256 = _sha256_file(receipt_path)
    _require(
        hmac.compare_digest(
            receipt_file_sha256, expected_import_receipt_sha256
        ),
        "import receipt differs from the caller-pinned SHA256",
    )
    receipt = _strict_json_file(receipt_path, name="import receipt")
    receipt_sha = receipt.get("receipt_sha256")
    _require(
        receipt.get("schema") == _RECEIPT_SCHEMA
        and receipt.get("schema_version") == 2
        and _is_sha256(receipt_sha)
        and receipt_sha == _body_digest(receipt, "receipt_sha256"),
        "import receipt schema or canonical body SHA256 is invalid",
    )
    implementation = _mapping(
        receipt.get("implementation"), name="receipt.implementation"
    )
    _require(
        implementation.get("name") == _IMPORTER_PATH.name
        and implementation.get("version") == _IMPORTER_VERSION
        and _is_sha256(implementation.get("source_sha256")),
        "import receipt importer identity is invalid",
    )
    source = _mapping(receipt.get("source"), name="receipt.source")
    trusted_inputs = _mapping(
        receipt.get("trusted_inputs"),
        name="receipt.trusted_inputs",
    )
    contract = _mapping(
        receipt.get("release_contract"),
        name="receipt.release_contract",
    )
    checkpoints = _mapping(
        receipt.get("checkpoint_validation"),
        name="receipt.checkpoint_validation",
    )
    checkpoint_records = _sequence(
        checkpoints.get("records"),
        name="receipt.checkpoint_validation.records",
    )
    attempt = source.get("attempt")
    transport_receipt_schema = source.get("transport_receipt_schema")
    _require(
        isinstance(attempt, int)
        and not isinstance(attempt, bool)
        and attempt > 0,
        "import receipt source attempt must be a positive integer",
    )
    _require(
        transport_receipt_schema == _REMOTE_TRANSPORT_RECEIPT_SCHEMA,
        "import receipt must use the canonical remote transport schema",
    )
    expected_report_relative_path = (
        f"attempts/attempt-{attempt:03d}/production_toy_gate.json"
    )
    _require(
        source.get("report_relative_path") == "production_toy_gate.json"
        and source.get("remote_report_relative_path")
        == expected_report_relative_path,
        "import receipt staged/remote report paths disagree with the current attempt",
    )
    _require(
        isinstance(source.get("commit"), str)
        and _COMMIT_RE.fullmatch(source["commit"]) is not None
        and _is_sha256(source.get("report_file_sha256"))
        and _is_sha256(source.get("report_body_sha256"))
        and _is_sha256(source.get("nested_report_body_sha256"))
        and _is_sha256(source.get("source_tree_sha256"))
        and _is_sha256(source.get("environment_sha256"))
        and _is_sha256(source.get("outer_sha256sums_file_sha256"))
        and _is_sha256(source.get("fetch_receipt_sha256"))
        and _is_sha256(source.get("sha256sums_sha256")),
        "import receipt source identity is invalid",
    )
    _require(
        set(trusted_inputs)
        == {
            "expected_commit",
            "expected_environment_lock_sha256",
            "expected_fetch_receipt_sha256",
            "expected_sha256sums_sha256",
        }
        and trusted_inputs.get("expected_commit") == source.get("commit")
        and trusted_inputs.get("expected_environment_lock_sha256")
        == source.get("environment_sha256")
        and trusted_inputs.get("expected_fetch_receipt_sha256")
        == source.get("fetch_receipt_sha256")
        and trusted_inputs.get("expected_sha256sums_sha256")
        == source.get("sha256sums_sha256")
        and source.get("outer_sha256sums_file_sha256")
        == source.get("sha256sums_sha256"),
        "import receipt trusted inputs do not bind its source identity",
    )
    _require(
        contract.get("validation_tier") == "release"
        and contract.get("release_certified") is True
        and contract.get("protocol") == _PROTOCOL
        and contract.get("protocol_version") == 2
        and contract.get("schema_version") == 4
        and contract.get("open_system_run_count") == 13
        and contract.get("multidimensional_outcome_count") == 13
        and contract.get("checkpoint_count") == 13,
        "import receipt does not describe the fixed protocol-v2 staging contract",
    )
    _require(
        checkpoints.get("mode")
        == "identity_and_integrity_load_without_scba_restore"
        and checkpoints.get("count") == 13
        and len(checkpoint_records) == 13,
        "import receipt lacks all thirteen integrity-loaded checkpoints",
    )
    staged = _mapping(
        receipt.get("staged_artifacts"),
        name="receipt.staged_artifacts",
    )
    _require(
        set(staged) == _EXPECTED_STAGED_ARTIFACTS
        and set(entries) == _EXPECTED_STAGED_ARTIFACTS | {"import_receipt.json"},
        "receipt staged-artifact set differs from SHA256SUMS",
    )
    for relative, raw_record in staged.items():
        record = _mapping(raw_record, name=f"staged artifact {relative}")
        path = _safe_file(root, relative, name=f"staged artifact {relative}")
        _require(
            record.get("sha256") == entries.get(relative)
            and record.get("sha256") == _sha256_file(path)
            and record.get("bytes") == path.stat().st_size,
            f"receipt identity differs for staged artifact {relative}",
        )
    report_path = _safe_file(
        root,
        "production_toy_gate.json",
        name="staged production report",
    )
    _require(
        _sha256_file(report_path) == source.get("report_file_sha256"),
        "staged production report differs from the import receipt",
    )
    for table in _CSV_SCHEMAS:
        relative = f"csv/{table}.csv"
        _require(relative in staged, f"required staged table is absent: {relative}")
    return MappingProxyType(
        {
            "receipt": receipt,
            "receipt_file_sha256": receipt_file_sha256,
            "caller_expected_import_receipt_sha256": (
                expected_import_receipt_sha256
            ),
            "receipt_body_sha256": receipt_sha,
            "staging_sha256sums_file_sha256": _sha256_file(
                root / "SHA256SUMS"
            ),
            "source": source,
            "importer_source_sha256": implementation["source_sha256"],
        }
    )


def _load_import_deriver() -> tuple[Any, str]:
    _require(
        _IMPORTER_PATH.is_file() and not _IMPORTER_PATH.is_symlink(),
        "canonical importer validator is missing or unsafe",
    )
    path = _IMPORTER_PATH.resolve(strict=True)
    _require(
        path.is_file(),
        "canonical importer validator is missing or unsafe",
    )
    sha256_before = _sha256_file(path)
    module_name = f"_negfpy_toy_gate_importer_{sha256_before[:16]}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    _require(
        specification is not None and specification.loader is not None,
        "cannot load the canonical importer validator",
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception as error:
        raise FigureRenderError(
            "cannot execute the canonical importer validator"
        ) from error
    finally:
        sys.modules.pop(module_name, None)
    _require(
        sha256_before == _sha256_file(path)
        and callable(getattr(module, "_derive_csv_payloads", None))
        and callable(getattr(module, "_derive_manuscript_tex_payloads", None)),
        "canonical importer validator changed or lacks its derivation hooks",
    )
    return module, sha256_before


def _derive_csv_payloads_from_report(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, bytes], str]:
    importer, importer_sha256 = _load_import_deriver()
    try:
        raw_payloads = importer._derive_csv_payloads(document)
    except Exception as error:
        raise FigureRenderError(
            "canonical importer could not re-derive staged CSV evidence"
        ) from error
    _require(
        isinstance(raw_payloads, Mapping),
        "canonical importer CSV hook did not return a mapping",
    )
    payloads: dict[str, bytes] = {}
    for name, payload in raw_payloads.items():
        _require(
            isinstance(name, str) and isinstance(payload, bytes),
            "canonical importer CSV hook returned invalid keys or values",
        )
        payloads[name] = payload
    _require(
        {f"csv/{name}.csv" for name in _PLOTTED_CSV_TABLES}.issubset(
            payloads
        ),
        "canonical importer CSV hook lacks a plotted evidence table",
    )
    return MappingProxyType(payloads), importer_sha256


def _derive_manuscript_tex_payloads_from_report(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, bytes], str]:
    importer, importer_sha256 = _load_import_deriver()
    try:
        raw_payloads = importer._derive_manuscript_tex_payloads(document)
    except Exception as error:
        raise FigureRenderError(
            "canonical importer could not re-derive staged manuscript TeX evidence"
        ) from error
    _require(
        isinstance(raw_payloads, Mapping),
        "canonical importer manuscript TeX hook did not return a mapping",
    )
    payloads: dict[str, bytes] = {}
    for name, payload in raw_payloads.items():
        _require(
            isinstance(name, str) and isinstance(payload, bytes),
            "canonical importer manuscript TeX hook returned invalid keys or values",
        )
        payloads[name] = payload
    _require(
        set(payloads) == set(_MANUSCRIPT_TEX_ARTIFACTS),
        "canonical importer manuscript TeX hook returned an unexpected file set",
    )
    return MappingProxyType(payloads), importer_sha256


def _validate_raw_report_binding(
    root: Path,
    staging: Mapping[str, Any],
) -> Mapping[str, Any]:
    report_path = _safe_file(
        root,
        "production_toy_gate.json",
        name="staged production report",
    )
    report_bytes = report_path.read_bytes()
    document = _strict_json_bytes(
        report_bytes,
        name="staged production report",
    )
    outer_sha256 = document.get("report_sha256")
    _require(
        document.get("schema") == _REPORT_SCHEMA
        and document.get("schema_version") == 4
        and _is_sha256(outer_sha256)
        and outer_sha256 == _body_digest(document, "report_sha256"),
        "staged production report has an invalid outer canonical body SHA256",
    )
    summary = _mapping(document.get("summary"), name="report.summary")
    _require(
        summary.get("validation_tier") == "release"
        and summary.get("release_certified") is True
        and summary.get("failures") == [],
        "staged production report summary is not release-certified",
    )
    multidimensional = _mapping(
        document.get("multidimensional_scba"),
        name="report.multidimensional_scba",
    )
    nested_sha256 = multidimensional.get("report_sha256")
    _require(
        multidimensional.get("schema") == _MULTIDIMENSIONAL_SCHEMA
        and multidimensional.get("schema_version") == 4
        and multidimensional.get("protocol") == _PROTOCOL
        and multidimensional.get("protocol_version") == 2
        and multidimensional.get("release_certified") is True
        and _is_sha256(nested_sha256)
        and nested_sha256
        == _body_digest(
            multidimensional,
            "report_sha256",
            ensure_ascii=True,
        ),
        "nested multidimensional report has an invalid canonical body SHA256",
    )
    source = _mapping(staging.get("source"), name="staging.source")
    _require(
        source.get("report_file_sha256") == _sha256_bytes(report_bytes)
        and source.get("report_body_sha256") == outer_sha256
        and source.get("nested_report_body_sha256") == nested_sha256,
        "raw report digests differ from the caller-pinned import receipt",
    )
    expected_payloads, importer_sha256 = _derive_csv_payloads_from_report(
        document
    )
    expected_tex_payloads, tex_importer_sha256 = (
        _derive_manuscript_tex_payloads_from_report(document)
    )
    _require(
        tex_importer_sha256 == importer_sha256,
        "canonical importer changed between CSV and manuscript TeX derivation",
    )
    _require(
        staging.get("importer_source_sha256") == importer_sha256,
        "canonical importer source differs from the receipt-bound implementation",
    )
    plotted_csv_sha256: dict[str, str] = {}
    for table in _PLOTTED_CSV_TABLES:
        observed_path = _safe_file(
            root,
            f"csv/{table}.csv",
            name=f"plotted {table} CSV",
        )
        observed = observed_path.read_bytes()
        expected = expected_payloads[f"csv/{table}.csv"]
        _require(
            hmac.compare_digest(
                _sha256_bytes(observed), _sha256_bytes(expected)
            )
            and observed == expected,
            f"plotted {table} CSV is not the canonical raw-report derivative",
        )
        plotted_csv_sha256[table] = _sha256_bytes(observed)

    manuscript_tex_sha256: dict[str, str] = {}
    for relative in _MANUSCRIPT_TEX_ARTIFACTS:
        observed_path = _safe_file(
            root,
            relative,
            name=f"manuscript TeX evidence {relative}",
        )
        observed = observed_path.read_bytes()
        expected = expected_tex_payloads[relative]
        _require(
            hmac.compare_digest(
                _sha256_bytes(observed), _sha256_bytes(expected)
            )
            and observed == expected,
            f"manuscript TeX {relative} is not the canonical raw-report derivative",
        )
        manuscript_tex_sha256[relative] = _sha256_bytes(observed)
    return MappingProxyType(
        {
            "report_file_sha256": _sha256_bytes(report_bytes),
            "report_body_sha256": outer_sha256,
            "nested_report_body_sha256": nested_sha256,
            "canonical_importer_sha256": importer_sha256,
            "plotted_csv_sha256": plotted_csv_sha256,
            "manuscript_tex_sha256": manuscript_tex_sha256,
        }
    )


def _read_csv(root: Path, name: str) -> list[dict[str, str]]:
    path = _safe_file(root, f"csv/{name}.csv", name=f"{name} CSV")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise FigureRenderError(f"{name} CSV is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    _require(
        reader.fieldnames is not None
        and tuple(reader.fieldnames) == _CSV_SCHEMAS[name]
        and len(set(reader.fieldnames)) == len(reader.fieldnames),
        f"{name} CSV header differs from the importer schema",
    )
    rows: list[dict[str, str]] = []
    try:
        for number, row in enumerate(reader, start=2):
            _require(
                None not in row and all(value is not None for value in row.values()),
                f"{name} CSV row {number} has missing/extra fields",
            )
            rows.append(dict(row))
    except csv.Error as error:
        raise FigureRenderError(f"{name} CSV is malformed: {error}") from error
    _require(rows, f"{name} CSV is empty")
    return rows


def _float(
    value: str,
    *,
    name: str,
    allow_empty: bool = False,
    nonnegative: bool = False,
    positive: bool = False,
) -> float | None:
    if value == "" and allow_empty:
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise FigureRenderError(f"{name} is not numeric: {value!r}") from error
    _require(math.isfinite(number), f"{name} is non-finite")
    if nonnegative:
        _require(number >= 0.0, f"{name} must be nonnegative")
    if positive:
        _require(number > 0.0, f"{name} must be positive")
    return number


def _history_float(value: str, *, name: str) -> float | None:
    if value == "Infinity":
        return None
    number = _float(value, name=name, nonnegative=True)
    assert number is not None
    return number


def _integer(value: str, *, name: str, positive: bool = False) -> int:
    _require(re.fullmatch(r"-?[0-9]+", value) is not None, f"{name} is not integer")
    number = int(value)
    if positive:
        _require(number > 0, f"{name} must be positive")
    return number


def _boolean(value: str, *, name: str) -> bool:
    _require(value in {"true", "false"}, f"{name} must be true or false")
    return value == "true"


def _applicability_flag(value: str, *, name: str) -> bool:
    """Parse numeric diagnostic applicability without generic truthiness."""

    if value in {"true", "1", "1.0"}:
        return True
    if value in {"false", "0", "0.0"}:
        return False
    raise FigureRenderError(f"{name} must be an explicit zero/one flag")


def _load_data(root: Path) -> Mapping[str, Any]:
    raw = {name: _read_csv(root, name) for name in _CSV_SCHEMAS}

    open_runs: list[dict[str, Any]] = []
    open_aliases: set[str] = set()
    for number, row in enumerate(raw["open_runs"]):
        alias = row["alias"]
        _require(alias and alias not in open_aliases, "duplicate open-run alias")
        open_aliases.add(alias)
        open_runs.append(
            {
                "alias": alias,
                "name": row["name"],
                "temperature": _float(row["temperature"], name="temperature"),
                "g": _float(row["g"], name="g"),
                "control": (
                    _float(row["grid_step"], name="grid_step", positive=True),
                    _integer(row["grid_n_points"], name="grid_n_points", positive=True),
                    _float(row["cutoff"], name="cutoff", positive=True),
                    _float(row["eta_lead"], name="eta_lead", positive=True),
                ),
                "oracle_linewidth": _float(
                    row["oracle_parent_linewidth"],
                    name=f"open run {number} oracle linewidth",
                    nonnegative=True,
                ),
                "born_linewidth": _float(
                    row["production_parent_linewidth"],
                    name=f"open run {number} Born linewidth",
                    nonnegative=True,
                ),
                "linewidth_ratio": _float(
                    row["linewidth_gate_ratio"],
                    name="linewidth gate ratio",
                    allow_empty=True,
                    nonnegative=True,
                ),
                "oracle_shift": _float(
                    row["oracle_parent_shift"], name="oracle shift"
                ),
                "born_shift": _float(
                    row["production_parent_shift"], name="Born shift"
                ),
                "shift_ratio": _float(
                    row["shift_gate_ratio"],
                    name="shift gate ratio",
                    allow_empty=True,
                    nonnegative=True,
                ),
                "agreement_passed": _boolean(
                    row["agreement_passed"], name="agreement_passed"
                ),
            }
        )
    _require(len(open_runs) == 13, "open_runs CSV must contain 13 rows")

    open_checks: list[dict[str, Any]] = []
    for row in raw["open_checks"]:
        _require(
            row["source"] in {"oracle", "production_born", "production"}
            and row["observable"] in {"linewidth", "shift"},
            "open check has an unknown source or observable",
        )
        open_checks.append(
            {
                **row,
                "gate_ratio": _float(
                    row["gate_ratio"],
                    name="open-check gate ratio",
                    allow_empty=True,
                    nonnegative=True,
                ),
                "passed": _boolean(row["passed"], name="open check passed"),
                "critical": _boolean(
                    row["certification_critical"], name="open check critical"
                ),
            }
        )

    outcomes: list[dict[str, Any]] = []
    outcome_aliases: set[str] = set()
    outcome_ids: set[str] = set()
    for row in raw["scba_outcomes"]:
        alias = row["alias"]
        run_id = row["run_id"]
        role = row["certification_role"]
        _require(
            alias and alias not in outcome_aliases and run_id and run_id not in outcome_ids,
            "SCBA outcome alias/run_id is empty or duplicated",
        )
        _require(
            role in {"required", "diagnostic_only"}
            and row["status"] == "converged",
            "SCBA outcome role/status is invalid",
        )
        outcome_aliases.add(alias)
        outcome_ids.add(run_id)
        outcomes.append(
            {
                "alias": alias,
                "run_id": run_id,
                "role": role,
                "g": _float(row["g"], name="SCBA g", nonnegative=True),
                "bias": _float(row["bias"], name="SCBA bias", nonnegative=True),
                "initialization": row["initialization"],
                "parent": row["continuation_parent_id"],
                "backend": row["backend"],
                "control": (
                    _float(row["spacing"], name="spacing", positive=True),
                    _float(row["cutoff"], name="cutoff", positive=True),
                    _float(row["eta_lead"], name="eta_lead", positive=True),
                    _float(row["mixing"], name="mixing", positive=True),
                ),
                "iterations": _integer(
                    row["iterations"], name="iterations", positive=True
                ),
                "left_current": _float(row["left_current"], name="left current"),
                "elapsed": _float(
                    row["elapsed_seconds"], name="elapsed seconds", positive=True
                ),
                "release_flag": _boolean(
                    row["result_release_certified"], name="result release flag"
                ),
            }
        )
    _require(
        len(outcomes) == 13
        and sum(item["role"] == "required" for item in outcomes) == 11
        and sum(item["role"] == "diagnostic_only" for item in outcomes) == 2,
        "SCBA outcomes must contain 11 required and two diagnostic rows",
    )

    history: list[dict[str, Any]] = []
    for row in raw["scba_history"]:
        _require(
            row["alias"] in outcome_aliases and row["run_id"] in outcome_ids,
            "SCBA history references an unknown outcome",
        )
        history.append(
            {
                "alias": row["alias"],
                "run_id": row["run_id"],
                "iteration": _integer(
                    row["iteration"], name="history iteration", positive=True
                ),
                "sigma": _history_float(
                    row["sigma_residual"], name="sigma residual"
                ),
                "green": _history_float(
                    row["green_residual"], name="green residual"
                ),
                "identity": _history_float(
                    row["identity_residual"], name="identity residual"
                ),
            }
        )
    _require(
        {item["alias"] for item in history} == outcome_aliases,
        "SCBA history does not cover every outcome",
    )

    comparisons: list[dict[str, Any]] = []
    aliases_by_run_id = {
        str(item["run_id"]): str(item["alias"])
        for item in outcomes
    }
    for row in raw["scba_comparisons"]:
        _require(
            row["left_run_id"] in aliases_by_run_id
            and row["right_run_id"] in aliases_by_run_id
            and aliases_by_run_id[row["left_run_id"]] == row["left_alias"]
            and aliases_by_run_id[row["right_run_id"]] == row["right_alias"],
            "SCBA comparison references an unknown or mismatched outcome",
        )
        current_scope = row["current_scope"]
        _require(
            current_scope
            in {"symmetrized_transport", "full_conservation_vector"},
            "SCBA comparison has an unknown current scope",
        )
        field_comparison_applicable = _boolean(
            row["field_comparison_applicable"],
            name="field comparison applicable",
        )
        field_gate_applicable = _boolean(
            row["field_gate_applicable"],
            name="field gate applicable",
        )
        current_gate_applicable = _boolean(
            row["current_gate_applicable"],
            name="current gate applicable",
        )
        _require(
            not field_gate_applicable or field_comparison_applicable,
            "an SCBA field gate requires an applicable field comparison",
        )
        field_rtol = _float(
            row["field_rtol"], name="field rtol", positive=True
        )
        field_atol = _float(
            row["field_atol"], name="field atol", positive=True
        )
        current_rtol = _float(
            row["current_rtol"], name="current rtol", positive=True
        )
        current_atol = _float(
            row["current_atol"], name="current atol", positive=True
        )
        field_ratio = _float(
            row["max_field_gate_ratio"],
            name="field gate ratio",
            allow_empty=True,
            nonnegative=True,
        )
        _require(
            (field_ratio is not None) == field_gate_applicable,
            "field-gate applicability disagrees with the field gate metric",
        )
        current_absolute = _float(
            row["current_max_absolute_error"],
            name="current maximum absolute error",
            nonnegative=True,
        )
        current_allowed = _float(
            row["current_allowed_absolute_error"],
            name="current allowed absolute error",
            positive=True,
        )
        current_ratio = _float(
            row["current_gate_ratio"],
            name="current gate ratio",
            nonnegative=True,
        )
        current_relative = _float(
            row["current_max_relative_error"],
            name="current maximum relative error",
            nonnegative=True,
        )
        assert current_absolute is not None
        assert current_allowed is not None
        assert current_ratio is not None
        assert current_relative is not None
        _require(
            math.isclose(
                current_ratio,
                current_absolute / current_allowed,
                rel_tol=5.0e-13,
                abs_tol=5.0e-15,
            ),
            "current gate ratio is inconsistent with its absolute error and allowance",
        )
        critical = _boolean(
            row["certification_critical"], name="comparison critical"
        )
        passed = _boolean(row["passed"], name="comparison passed")
        _require(not critical or passed, "a critical SCBA comparison failed")
        try:
            failures = json.loads(
                row["failures"],
                parse_constant=_reject_constant,
                object_pairs_hook=_unique_object,
            )
        except json.JSONDecodeError as error:
            raise FigureRenderError(
                "SCBA comparison failures are not strict JSON"
            ) from error
        _require(
            isinstance(failures, list)
            and all(isinstance(item, str) for item in failures)
            and (not passed or not failures),
            "SCBA comparison pass/failure metadata is inconsistent",
        )
        if passed:
            _require(
                not field_gate_applicable
                or (field_ratio is not None and field_ratio <= 1.0 + 1.0e-12),
                "a passed SCBA comparison exceeds its applicable field gate",
            )
            _require(
                not current_gate_applicable
                or current_ratio <= 1.0 + 1.0e-12,
                "a passed SCBA comparison exceeds its applicable current gate",
            )
        linewidth_ratio = _float(
            row["linewidth_gate_ratio"],
            name="linewidth gate ratio",
            allow_empty=True,
            nonnegative=True,
        )
        shift_ratio = _float(
            row["shift_gate_ratio"],
            name="shift gate ratio",
            allow_empty=True,
            nonnegative=True,
        )
        if passed:
            _require(
                linewidth_ratio is None
                or linewidth_ratio <= 1.0 + 1.0e-12,
                "a passed SCBA comparison exceeds its linewidth gate",
            )
            _require(
                shift_ratio is None or shift_ratio <= 1.0 + 1.0e-12,
                "a passed SCBA comparison exceeds its shift gate",
            )
        comparisons.append(
            {
                "name": row["name"],
                "kind": row["kind"],
                "critical": critical,
                "passed": passed,
                "current_scope": current_scope,
                "field_comparison_applicable": field_comparison_applicable,
                "field_gate_applicable": field_gate_applicable,
                "current_gate_applicable": current_gate_applicable,
                "field_rtol": field_rtol,
                "field_atol": field_atol,
                "current_rtol": current_rtol,
                "current_atol": current_atol,
                "field_ratio": field_ratio,
                "current_absolute": current_absolute,
                "current_allowed": current_allowed,
                "current_ratio": current_ratio,
                "current_relative": current_relative,
                "linewidth_ratio": linewidth_ratio,
                "shift_ratio": shift_ratio,
                "failures": tuple(failures),
            }
        )
    comparisons_by_name = {
        str(item["name"]): item for item in comparisons
    }
    _require(
        len(comparisons_by_name) == len(comparisons)
        and set(comparisons_by_name) == set(_EXPECTED_COMPARISON_CONTRACT),
        "SCBA comparison names differ from the exact protocol-v2 contract",
    )
    for name, policy in _EXPECTED_COMPARISON_CONTRACT.items():
        item = comparisons_by_name[name]
        _require(
            item["kind"] == policy["kind"]
            and item["critical"] is policy["critical"]
            and item["current_scope"] == policy["current_scope"]
            and item["field_comparison_applicable"]
            is policy["field_comparison_applicable"]
            and item["field_gate_applicable"]
            is policy["field_gate_applicable"]
            and item["current_gate_applicable"]
            is policy["current_gate_applicable"]
            and item["field_rtol"] == policy["field_rtol"]
            and item["field_atol"] == policy["field_atol"]
            and item["current_rtol"] == policy["current_rtol"]
            and item["current_atol"] == policy["current_atol"],
            f"comparison {name} differs from its exact independent policy",
        )
        line_shape_expected = policy["kind"] == "numerical_refinement"
        _require(
            (item["linewidth_ratio"] is not None) == line_shape_expected
            and (item["shift_ratio"] is not None) == line_shape_expected,
            f"comparison {name} has invalid line-shape applicability",
        )

    conservation: list[dict[str, Any]] = []
    for row in raw["conservation"]:
        allowed = _float(row["allowed_error"], name="allowed error", positive=True)
        assert allowed is not None
        passed = _boolean(row["passed"], name="conservation passed")
        _require(passed, "a conservation gate failed")
        full = _float(row["full_balance"], name="full balance")
        assert full is not None
        spatial_gate = row["spatial_current_gate"]
        spatial_available = spatial_gate != "unavailable"
        corrected_cut_is_gate = spatial_gate.startswith("corrected_eom_cut_current")
        conservation.append(
            {
                "name": row["name"],
                "alias": row["alias"],
                "contact": _float(
                    row["contact_gate_ratio"],
                    name="contact gate ratio",
                    nonnegative=True,
                ),
                "collision": _float(
                    row["collision_gate_ratio"],
                    name="collision gate ratio",
                    nonnegative=True,
                ),
                "full": abs(full) / allowed,
                "cut": (
                    None
                    if not corrected_cut_is_gate
                    or row["corrected_cut_current_error"] == ""
                    else abs(
                        _float(
                            row["corrected_cut_current_error"],
                            name="cut-current error",
                        )
                    )
                    / allowed
                ),
                "continuity": (
                    None
                    if not spatial_available
                    or row["continuity_residual_error"] == ""
                    else abs(
                        _float(
                            row["continuity_residual_error"],
                            name="continuity error",
                        )
                    )
                    / allowed
                ),
                "spatial_gate": spatial_gate,
            }
        )
    _require(len(conservation) == 2, "conservation CSV must contain two rows")

    passivity: list[dict[str, Any]] = []
    spectra = {"green", "sigma_pp", "sigma_total", "sigma_left", "sigma_right"}
    interacting_applicability_by_alias: dict[str, set[bool]] = {}
    for row in raw["passivity"]:
        _require(
            row["run_id"] in aliases_by_run_id
            and aliases_by_run_id[row["run_id"]] == row["alias"]
            and row["spectrum"] in spectra,
            "passivity row references an unknown or mismatched outcome/spectrum",
        )
        alias = row["alias"]
        spectrum = row["spectrum"]
        interacting_applicable = _applicability_flag(
            row["interacting_retarded_passivity_applicable"],
            name="interacting retarded applicability",
        )
        retarded_applicable = _applicability_flag(
            row["retarded_passivity_applicable"],
            name="retarded applicability",
        )
        lesser_applicable = _applicability_flag(
            row["lesser_correlation_positivity_applicable"],
            name="lesser applicability",
        )
        greater_applicable = _applicability_flag(
            row["greater_correlation_positivity_applicable"],
            name="greater applicability",
        )
        expected_retarded = bool(
            spectrum in {"sigma_left", "sigma_right"}
            or interacting_applicable
        )
        _require(
            retarded_applicable == expected_retarded,
            "retarded-passivity applicability is inconsistent with its scope",
        )
        _require(
            lesser_applicable and greater_applicable,
            "correlation positivity must remain applicable for every spectrum",
        )
        interacting_applicability_by_alias.setdefault(alias, set()).add(
            interacting_applicable
        )
        passivity.append(
            {
                "alias": alias,
                "spectrum": spectrum,
                "interacting_retarded_applicable": interacting_applicable,
                "retarded_applicable": retarded_applicable,
                "retarded": _float(
                    row["retarded_passivity_violation"],
                    name="retarded violation",
                    nonnegative=True,
                ),
                "lesser_applicable": lesser_applicable,
                "lesser": _float(
                    row["lesser_correlation_positivity_violation"],
                    name="lesser violation",
                    nonnegative=True,
                ),
                "greater_applicable": greater_applicable,
                "greater": _float(
                    row["greater_correlation_positivity_violation"],
                    name="greater violation",
                    nonnegative=True,
                ),
            }
        )
    observed_pairs = {(item["alias"], item["spectrum"]) for item in passivity}
    _require(
        len(passivity) == 65
        and observed_pairs == {(alias, spectrum) for alias in outcome_aliases for spectrum in spectra},
        "passivity CSV must contain one row for each outcome/spectrum pair",
    )
    _require(
        set(interacting_applicability_by_alias) == outcome_aliases
        and all(
            len(values) == 1
            for values in interacting_applicability_by_alias.values()
        ),
        "interacting retarded-passivity scope must be constant per outcome",
    )

    backends: list[dict[str, Any]] = []
    backend_names: set[str] = set()
    for row in raw["backend_oracle"]:
        passed = _boolean(row["passed"], name="backend oracle passed")
        _require(passed, "a bounded-backend oracle record failed")
        if row["record_type"] == "backend":
            backend_names.add(row["name"])
            backends.append(
                {
                    "type": "backend",
                    "name": row["name"],
                    "elapsed": _float(
                        row["elapsed_seconds"],
                        name="backend elapsed",
                        positive=True,
                    ),
                }
            )
        elif row["record_type"] == "comparison":
            backends.append(
                {
                    "type": "comparison",
                    "name": row["name"],
                    "reference": row["reference_backend"],
                    "candidate": row["candidate_backend"],
                    "field_ratio": _float(
                        row["max_field_gate_ratio"],
                        name="backend field ratio",
                        nonnegative=True,
                    ),
                    "current_ratio": _float(
                        row["current_gate_ratio"],
                        name="backend current ratio",
                        nonnegative=True,
                    ),
                }
            )
        else:
            raise FigureRenderError("backend_oracle has unknown record_type")
    _require(
        backend_names == {"serial", "threaded", "fft"}
        and any(item["type"] == "comparison" for item in backends),
        "bounded backend table is incomplete",
    )

    timings: list[dict[str, Any]] = []
    for row in raw["timings"]:
        category = row["category"]
        _require(
            category
            in {"open_oracle", "open_born", "scba_outcome", "multidimensional_total"},
            "timing row has an unknown category",
        )
        timings.append(
            {
                "category": category,
                "alias": row["alias"],
                "wall": _float(
                    row["wall_seconds"], name="wall seconds", positive=True
                ),
                "total": _boolean(
                    row["authoritative_total"], name="authoritative total"
                ),
            }
        )
    _require(
        sum(item["total"] for item in timings) == 1,
        "timing table must contain one authoritative total",
    )

    return MappingProxyType(
        {
            "open_runs": tuple(open_runs),
            "open_checks": tuple(open_checks),
            "outcomes": tuple(outcomes),
            "history": tuple(history),
            "comparisons": tuple(comparisons),
            "conservation": tuple(conservation),
            "passivity": tuple(passivity),
            "backends": tuple(backends),
            "timings": tuple(timings),
        }
    )


def _resolve_font(
    regular_path: Path | None,
    bold_path: Path | None,
) -> Mapping[str, Any]:
    if (regular_path is None) != (bold_path is None):
        raise FigureRenderError("font regular and bold paths must be supplied together")
    if regular_path is not None and bold_path is not None:
        regular_source = regular_path.expanduser()
        bold_source = bold_path.expanduser()
        _require(
            not regular_source.is_symlink() and not bold_source.is_symlink(),
            "font paths cannot be symlinks",
        )
        regular = regular_source.resolve(strict=True)
        bold = bold_source.resolve(strict=True)
    else:
        exact: dict[str, Path] = {}
        for entry in font_manager.fontManager.ttflist:
            if entry.name in {"Arial", "Helvetica"}:
                key = "bold" if entry.weight in {"bold", 700} else "regular"
                exact.setdefault(key, Path(entry.fname))
        if {"regular", "bold"}.issubset(exact):
            regular, bold = exact["regular"], exact["bold"]
        else:
            regular = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
            bold = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
    _require(
        regular.is_file()
        and bold.is_file()
        and not regular.is_symlink()
        and not bold.is_symlink()
        and regular.suffix.lower() == ".ttf"
        and bold.suffix.lower() == ".ttf",
        "font paths must be regular TrueType files",
    )
    font_manager.fontManager.addfont(str(regular))
    font_manager.fontManager.addfont(str(bold))
    family = font_manager.FontProperties(fname=str(regular)).get_name()
    bold_family = font_manager.FontProperties(fname=str(bold)).get_name()
    exact_families = {"Arial", "Helvetica"}
    metric_families = {"Liberation Sans", "Nimbus Sans"}
    _require(
        family == bold_family and family in exact_families | metric_families,
        "fonts must be matching Arial/Helvetica families or recognized metric-compatible substitutes",
    )
    substitute = family not in exact_families
    return MappingProxyType(
        {
            "family": family,
            "regular_path": str(regular),
            "bold_path": str(bold),
            "regular_sha256": _sha256_file(regular),
            "bold_sha256": _sha256_file(bold),
            "arial_helvetica_metric_substitute": substitute,
            "font_contract": (
                "Arial_or_Helvetica"
                if not substitute
                else "declared_Arial_Helvetica_metric_substitute"
            ),
        }
    )


def _nature_rc(font: Mapping[str, Any]) -> dict[str, Any]:
    family = str(font["family"])
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [family],
        "font.size": 6.0,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7.0,
        "axes.titleweight": "bold",
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.2,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def _panel_label(ax: Any, index: int, family: str) -> None:
    ax.text(
        -0.14,
        1.08,
        string.ascii_lowercase[index],
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        fontfamily=family,
        ha="left",
        va="top",
        clip_on=False,
        color=_PALETTE["black"],
    )


def _finish_axis(ax: Any, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _mode(values: Iterable[tuple[Any, ...]]) -> tuple[Any, ...]:
    counts = Counter(values)
    maximum = max(counts.values())
    return sorted(value for value, count in counts.items() if count == maximum)[0]


def _open_check_category(item: Mapping[str, Any]) -> str:
    if item["kind"] == "refinement":
        return str(item["check"]).replace("eta_lead", "eta")
    if item["kind"] == "spontaneous_decay":
        return "spont."
    return str(item["kind"])


def _ordered_continuation_path(
    outcomes: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Follow explicit parent identities; aliases/document order are cosmetic."""

    path = [item for item in outcomes if str(item["run_id"]).startswith("path_")]
    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    starts: list[Mapping[str, Any]] = []
    for item in path:
        parent = str(item["parent"])
        if not parent:
            starts.append(item)
        else:
            by_parent.setdefault(parent, []).append(item)
    _require(len(starts) == 1, "continuation path must have one parent-free start")
    ordered = [starts[0]]
    seen = {str(starts[0]["run_id"])}
    while True:
        children = by_parent.get(str(ordered[-1]["run_id"]), [])
        if not children:
            break
        _require(len(children) == 1, "continuation path branches ambiguously")
        child = children[0]
        _require(str(child["run_id"]) not in seen, "continuation path contains a cycle")
        ordered.append(child)
        seen.add(str(child["run_id"]))
    _require(len(ordered) == len(path), "continuation path is disconnected")
    return ordered


def _render_open(data: Mapping[str, Any], font: Mapping[str, Any]) -> Any:
    runs = list(data["open_runs"])
    checks = list(data["open_checks"])
    baseline_control = _mode(item["control"] for item in runs)
    baseline = [item for item in runs if item["control"] == baseline_control]
    temperatures = sorted({float(item["temperature"]) for item in baseline})
    temperature_colors = {
        value: (_PALETTE["blue"], _PALETTE["orange"], _PALETTE["green"])[
            index % 3
        ]
        for index, value in enumerate(temperatures)
    }
    fig, axes = plt.subplots(2, 2, figsize=_FIGURE_SIZE)
    axes = axes.ravel()
    for panel, observable, ylabel in (
        (0, "linewidth", "Parent linewidth (ω₀)"),
        (1, "shift", "Parent shift (ω₀)"),
    ):
        ax = axes[panel]
        for temperature in temperatures:
            selected = [
                item for item in baseline if float(item["temperature"]) == temperature
            ]
            selected.sort(key=lambda item: float(item["g"]))
            for source, marker, linestyle in (
                ("oracle", "o", "-"),
                ("born", "s", "--"),
            ):
                ax.plot(
                    [float(item["g"]) for item in selected],
                    [float(item[f"{source}_{observable}"]) for item in selected],
                    marker=marker,
                    linestyle=linestyle,
                    color=temperature_colors[temperature],
                    markerfacecolor=(
                        temperature_colors[temperature] if source == "oracle" else "white"
                    ),
                    markeredgewidth=0.7,
                    label=f"{source}, T={temperature:g}",
                )
        _finish_axis(
            ax,
            title=(
                "Three-mode linewidth" if observable == "linewidth" else "Three-mode shift"
            ),
            xlabel="Signed cubic coupling g (solver units)",
            ylabel=ylabel,
        )
        ax.legend(frameon=False, ncol=2, handlelength=1.5, columnspacing=0.8)

    ax = axes[2]
    linewidth_values = [
        (float(item["oracle_linewidth"]), float(item["born_linewidth"]))
        for item in runs
    ]
    shift_values = [
        (float(item["oracle_shift"]), float(item["born_shift"])) for item in runs
    ]
    for values, label, color, marker in (
        (linewidth_values, "linewidth", _PALETTE["blue"], "o"),
        (shift_values, "shift", _PALETTE["vermillion"], "s"),
    ):
        ax.scatter(
            [value[0] for value in values],
            [value[1] for value in values],
            s=13,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.3,
            label=label,
        )
    all_values = [value for pair in linewidth_values + shift_values for value in pair]
    lower, upper = min(all_values), max(all_values)
    if lower == upper:
        lower, upper = lower - 0.5, upper + 0.5
    ax.plot([lower, upper], [lower, upper], color=_PALETTE["black"], linestyle=":")
    _finish_axis(
        ax,
        title="Independent oracle agreement",
        xlabel="Oracle result (ω₀)",
        ylabel="One-shot Born result (ω₀)",
    )
    ax.legend(frameon=False)

    ax = axes[3]
    categories = []
    for item in checks:
        category = _open_check_category(item)
        if category not in categories:
            categories.append(category)
    category_index = {name: index for index, name in enumerate(categories)}
    styles = {
        ("oracle", "linewidth"): (-0.18, "o", _PALETTE["blue"]),
        ("oracle", "shift"): (-0.06, "s", _PALETTE["orange"]),
        ("production_born", "linewidth"): (0.06, "^", _PALETTE["green"]),
        ("production_born", "shift"): (0.18, "D", _PALETTE["purple"]),
        ("production", "linewidth"): (0.06, "^", _PALETTE["green"]),
    }
    for item in checks:
        ratio = item["gate_ratio"]
        if ratio is None:
            continue
        offset, marker, color = styles[(item["source"], item["observable"])]
        critical = bool(item["critical"])
        ax.scatter(
            category_index[_open_check_category(item)] + offset,
            ratio,
            s=13,
            marker=marker if critical else "D",
            facecolor=color if critical else "white",
            edgecolor=color,
            linewidth=0.7,
        )
    ax.axhline(1.0, color=_PALETTE["black"], linestyle=":", linewidth=0.8)
    ax.set_xticks(range(len(categories)), categories, rotation=25, ha="right")
    ax.set_ylim(bottom=0.0)
    _finish_axis(
        ax,
        title="Preregistered open-system checks",
        xlabel="Validation check (category)",
        ylabel="Gate ratio (1)",
    )
    handles = [
        Line2D([], [], marker=marker, linestyle="None", color=color, label=label)
        for label, marker, color in (
            ("oracle Γ", "o", _PALETTE["blue"]),
            ("oracle Δω", "s", _PALETTE["orange"]),
            ("Born Γ", "^", _PALETTE["green"]),
            ("Born Δω", "D", _PALETTE["purple"]),
        )
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, columnspacing=0.8)
    for index, ax in enumerate(axes):
        _panel_label(ax, index, str(font["family"]))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.96, wspace=0.32, hspace=0.43)
    return fig


def _render_scba(data: Mapping[str, Any], font: Mapping[str, Any]) -> Any:
    outcomes = list(data["outcomes"])
    history = list(data["history"])
    comparisons = list(data["comparisons"])
    conservation = list(data["conservation"])
    fig, axes = plt.subplots(2, 2, figsize=_FIGURE_SIZE)
    axes = axes.ravel()

    ax = axes[0]
    path = _ordered_continuation_path(outcomes)
    _require(len(path) >= 2, "SCBA outcome table lacks the L-shaped path")
    x = np.arange(len(path))
    ax.plot(
        x,
        [item["left_current"] for item in path],
        color=_PALETTE["blue"],
        marker="o",
        markerfacecolor=_PALETTE["blue"],
    )
    labels = [f"g={item['g']:g}\nΔT={item['bias']:g}" for item in path]
    ax.set_xticks(x, labels)
    _finish_axis(
        ax,
        title="L-shaped continuation path",
        xlabel="Continuation state (solver units)",
        ylabel="Left-contact current (J₀)",
    )

    ax = axes[1]
    max_g = max(float(item["g"]) for item in outcomes)
    max_bias = max(float(item["bias"]) for item in outcomes)
    base_control = _mode(item["control"] for item in outcomes)
    endpoints = [
        item
        for item in outcomes
        if item["g"] == max_g
        and item["bias"] == max_bias
        and item["control"] == base_control
        and item["initialization"] in {"zero", "born_seed", "continuation"}
    ]
    init_style = {
        "zero": (_PALETTE["blue"], "o"),
        "born_seed": (_PALETTE["orange"], "s"),
        "continuation": (_PALETTE["green"], "^"),
    }
    for endpoint in endpoints:
        records = sorted(
            (item for item in history if item["alias"] == endpoint["alias"]),
            key=lambda item: item["iteration"],
        )
        finite = []
        for item in records:
            residuals = [
                value
                for value in (item["sigma"], item["green"], item["identity"])
                if value is not None
            ]
            if residuals and max(residuals) > 0.0:
                finite.append({**item, "maximum_residual": max(residuals)})
        color, marker = init_style[endpoint["initialization"]]
        ax.semilogy(
            [item["iteration"] for item in finite],
            [item["maximum_residual"] for item in finite],
            color=color,
            marker=marker,
            markevery=max(1, len(finite) // 8),
            label=endpoint["initialization"].replace("_", " "),
        )
    ax.axhline(1.0e-8, color=_PALETTE["black"], linestyle=":", linewidth=0.8)
    _finish_axis(
        ax,
        title="Endpoint fixed-point convergence",
        xlabel="SCBA iteration (index)",
        ylabel="Max fixed-point residual (1)",
    )
    ax.legend(frameon=False)

    ax = axes[2]
    metrics = (
        ("contact", "contact", _PALETTE["blue"], "o"),
        ("collision", "collision", _PALETTE["orange"], "s"),
        ("full", "full", _PALETTE["green"], "^"),
        ("cut", "cut", _PALETTE["vermillion"], "D"),
        ("continuity", "continuity", _PALETTE["purple"], "v"),
    )
    x = np.arange(len(metrics))
    offsets = np.linspace(-0.11, 0.11, len(conservation))
    for offset, check in zip(offsets, conservation):
        for index, (_, key, color, marker) in enumerate(metrics):
            value = check[key]
            if value is None:
                ax.text(index + offset, 0.03, "NA", color=_PALETTE["gray"], ha="center", va="bottom", fontsize=5)
            else:
                ax.scatter(index + offset, value, s=15, color=color, marker=marker, edgecolor="white", linewidth=0.3)
        ax.plot([], [], color=_PALETTE["black"], marker="o", linestyle="None", label=str(check["name"]).replace("interacting_", ""))
    ax.axhline(1.0, color=_PALETTE["black"], linestyle=":", linewidth=0.8)
    ax.set_xticks(x, [item[0] for item in metrics], rotation=20, ha="right")
    ax.set_ylim(bottom=0.0)
    _finish_axis(
        ax,
        title="Contact, collision and spatial balance",
        xlabel="Conservation metric (normalized)",
        ylabel="Absolute error / tolerance (1)",
    )
    ax.legend(frameon=False)

    ax = axes[3]
    refinements = [item for item in comparisons if item["kind"] == "numerical_refinement"]
    _require(len(refinements) == 6, "SCBA comparison table lacks six refinements")
    metric_styles = (
        ("field_ratio", "fields", _PALETTE["blue"], "o"),
        ("current_ratio", "current", _PALETTE["orange"], "s"),
        ("linewidth_ratio", "linewidth", _PALETTE["green"], "^"),
        ("shift_ratio", "shift", _PALETTE["purple"], "D"),
    )
    offsets = np.linspace(-0.18, 0.18, len(metric_styles))
    for index, item in enumerate(refinements):
        for offset, (key, _, color, marker) in zip(offsets, metric_styles):
            value = item[key]
            if key == "field_ratio":
                if not item["field_comparison_applicable"]:
                    ax.text(
                        index + offset,
                        0.03,
                        "NA",
                        color=_PALETTE["gray"],
                        ha="center",
                        va="bottom",
                        fontsize=4.8,
                    )
                    continue
                if not item["field_gate_applicable"]:
                    ax.text(
                        index + offset,
                        0.03,
                        "desc.",
                        color=color,
                        ha="center",
                        va="bottom",
                        fontsize=4.8,
                    )
                    continue
            if key == "current_ratio" and not item["current_gate_applicable"]:
                ax.text(
                    index + offset,
                    0.03,
                    "desc.",
                    color=color,
                    ha="center",
                    va="bottom",
                    fontsize=4.8,
                )
                continue
            if value is None:
                continue
            ax.scatter(
                index + offset,
                value,
                s=15,
                marker=marker,
                facecolor=color if item["critical"] else "white",
                edgecolor=color,
                linewidth=0.7,
            )
    ax.axhline(1.0, color=_PALETTE["black"], linestyle=":", linewidth=0.8)
    labels = [
        item["name"].replace("numerical_refinement_", "").replace("_", " ")
        + ("" if item["critical"] else "\ndiag.")
        for item in refinements
    ]
    ax.set_xticks(range(len(refinements)), labels, rotation=30, ha="right")
    ax.set_ylim(bottom=0.0)
    _finish_axis(
        ax,
        title="Independent numerical refinements",
        xlabel="Control variation (category)",
        ylabel="Applicable gate ratio (1)",
    )
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="None",
            color=color,
            label=label,
        )
        for _, label, color, marker in metric_styles
    ] + [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color=_PALETTE["black"],
            label="critical row",
        ),
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            linestyle="None",
            color=_PALETTE["black"],
            label="diagnostic row",
        ),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, columnspacing=0.7)
    for index, ax in enumerate(axes):
        _panel_label(ax, index, str(font["family"]))
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.96, wspace=0.34, hspace=0.48)
    return fig


def _passivity_matrix(
    records: Sequence[Mapping[str, Any]],
    aliases: Sequence[str],
    *,
    kind: str,
) -> tuple[np.ma.MaskedArray, tuple[str, ...]]:
    spectra = ("green", "sigma_pp", "sigma_total", "sigma_left", "sigma_right")
    by_pair = {(item["alias"], item["spectrum"]): item for item in records}
    values = np.zeros((len(aliases), len(spectra)), dtype=float)
    mask = np.zeros_like(values, dtype=bool)
    for row_number, alias in enumerate(aliases):
        for column, spectrum in enumerate(spectra):
            item = by_pair[(alias, spectrum)]
            if kind == "retarded":
                applicable = bool(item["retarded_applicable"])
                value = float(item["retarded"])
            elif kind == "correlation":
                applicable = bool(item["lesser_applicable"] and item["greater_applicable"])
                value = max(float(item["lesser"]), float(item["greater"]))
            else:
                raise FigureRenderError(f"unknown passivity matrix kind {kind!r}")
            values[row_number, column] = value
            mask[row_number, column] = not applicable
    return np.ma.array(values, mask=mask), spectra


def _draw_passivity_matrix(
    fig: Any,
    ax: Any,
    matrix: np.ma.MaskedArray,
    aliases: Sequence[str],
    spectra: Sequence[str],
    *,
    title: str,
) -> None:
    cmap = plt.get_cmap("viridis").with_extremes(bad=_PALETTE["na"])
    compressed = matrix.compressed()
    maximum = float(compressed.max()) if compressed.size else 1.0
    norm = mpl_colors.Normalize(vmin=0.0, vmax=maximum if maximum > 0.0 else 1.0)
    image = ax.pcolormesh(
        np.arange(len(spectra) + 1) - 0.5,
        np.arange(len(aliases) + 1) - 0.5,
        matrix,
        shading="flat",
        cmap=cmap,
        norm=norm,
        rasterized=False,
    )
    ax.set_xticks(range(len(spectra)), [value.replace("sigma_", "Σ ") for value in spectra], rotation=25, ha="right")
    ax.set_yticks(range(len(aliases)), aliases)
    ax.set_xlim(-0.5, len(spectra) - 0.5)
    ax.set_ylim(len(aliases) - 0.5, -0.5)
    for row, column in zip(*np.where(np.ma.getmaskarray(matrix))):
        ax.text(column, row, "NA", color="#555555", ha="center", va="center", fontsize=4.8)
    ax.set_title(title, loc="left", pad=3, fontweight="bold")
    ax.set_xlabel("Spectrum (component)")
    ax.set_ylabel("SCBA outcome (alias)")
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    colorbar.set_label("Violation metric (1)", fontsize=6)
    colorbar.ax.tick_params(labelsize=5, width=0.5, length=2)


def _render_passivity(data: Mapping[str, Any], font: Mapping[str, Any]) -> Any:
    outcomes = list(data["outcomes"])
    aliases = [item["alias"] for item in outcomes]
    passivity = list(data["passivity"])
    backends = list(data["backends"])
    timings = list(data["timings"])
    fig, axes = plt.subplots(2, 2, figsize=_FIGURE_SIZE)
    axes = axes.ravel()
    retarded, spectra = _passivity_matrix(passivity, aliases, kind="retarded")
    correlation, _ = _passivity_matrix(passivity, aliases, kind="correlation")
    _draw_passivity_matrix(
        fig,
        axes[0],
        retarded,
        aliases,
        spectra,
        title="Retarded passivity (gray = not applicable)",
    )
    _draw_passivity_matrix(
        fig,
        axes[1],
        correlation,
        aliases,
        spectra,
        title="Correlation positivity (gray = not applicable)",
    )

    ax = axes[2]
    comparisons = [item for item in backends if item["type"] == "comparison"]
    x = np.arange(len(comparisons))
    width = 0.28
    ax.scatter(
        x - width / 2,
        [item["field_ratio"] for item in comparisons],
        marker="o",
        color=_PALETTE["blue"],
        label="all fields",
    )
    ax.scatter(
        x + width / 2,
        [item["current_ratio"] for item in comparisons],
        marker="s",
        facecolor="white",
        edgecolor=_PALETTE["orange"],
        linewidth=0.8,
        label="current",
    )
    ax.axhline(1.0, color=_PALETTE["black"], linestyle=":", linewidth=0.8)
    ax.set_xticks(x, [item["name"].replace("_vs_", " vs ") for item in comparisons], rotation=20, ha="right")
    ax.set_ylim(bottom=0.0)
    _finish_axis(
        ax,
        title="Bounded converged-SCBA backend oracle",
        xlabel="Backend comparison (pair)",
        ylabel="Gate ratio (1)",
    )
    ax.legend(frameon=False)

    ax = axes[3]
    categories = ("open_oracle", "open_born", "scba_outcome", "multidimensional_total")
    labels = ("open oracle", "open Born", "SCBA outcomes", "SCBA total")
    colors = (_PALETTE["blue"], _PALETTE["orange"], _PALETTE["green"], _PALETTE["black"])
    for index, (category, color) in enumerate(zip(categories, colors)):
        selected = [item for item in timings if item["category"] == category]
        for item in selected:
            ax.scatter(
                index,
                item["wall"],
                s=22 if item["total"] else 12,
                marker="*" if item["total"] else "o",
                facecolor=color if item["total"] else "white",
                edgecolor=color,
                linewidth=0.7,
            )
    ax.set_yscale("log")
    ax.set_xticks(range(len(categories)), labels, rotation=20, ha="right")
    _finish_axis(
        ax,
        title="Measured wall time (no extrapolation)",
        xlabel="Calculation tier (category)",
        ylabel="Wall time (s)",
    )
    for index, ax in enumerate(axes):
        _panel_label(ax, index, str(font["family"]))
    fig.subplots_adjust(left=0.105, right=0.97, bottom=0.12, top=0.96, wspace=0.40, hspace=0.50)
    return fig


def _run_pdf_tool(
    executable: Path,
    arguments: Sequence[str],
    *,
    name: str,
) -> str:
    _require(
        executable.is_file() and not executable.is_symlink(),
        f"required structural PDF tool is unavailable: {executable}",
    )
    try:
        completed = subprocess.run(
            [str(executable), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30.0,
            env={
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": "/usr/bin:/bin",
            },
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as error:
        raise FigureRenderError(f"cannot execute {name}") from error
    _require(
        completed.returncode == 0 and not completed.stderr.strip(),
        f"{name} rejected the rendered PDF structure",
    )
    return completed.stdout


def _poppler_table_rows(output: str, *, name: str) -> list[str]:
    lines = output.splitlines()
    separators = [
        index
        for index, line in enumerate(lines)
        if re.fullmatch(r"[- ]+", line.strip()) is not None
    ]
    _require(
        len(separators) == 1,
        f"{name} output has an unexpected table structure",
    )
    return [line.strip() for line in lines[separators[0] + 1 :] if line.strip()]


def _validate_pdf_bytes(pdf: bytes, *, name: str) -> None:
    _require(len(pdf) <= _MAX_FIGURE_BYTES, f"{name} exceeds 10 MiB")
    with tempfile.TemporaryDirectory(prefix="negfpy-pdf-validate-") as raw:
        path = Path(raw) / "figure.pdf"
        with path.open("xb") as stream:
            stream.write(pdf)
            stream.flush()
            os.fsync(stream.fileno())
        image_output = _run_pdf_tool(
            _PDFIMAGES_PATH,
            ("-list", str(path)),
            name="pdfimages",
        )
        image_rows = _poppler_table_rows(
            image_output,
            name="pdfimages",
        )
        _require(
            not image_rows,
            f"{name} contains image or inline-image records",
        )
        font_output = _run_pdf_tool(
            _PDFFONTS_PATH,
            (str(path),),
            name="pdffonts",
        )
        font_rows = _poppler_table_rows(font_output, name="pdffonts")
    _require(font_rows, f"{name} contains no structurally reported fonts")
    font_pattern = re.compile(
        r"^(?P<name>\S+)\s+"
        r"(?P<type>CID TrueType|TrueType|CID Type 0C|CID Type 0|"
        r"Type 1C|Type 1|Type 3)\s+"
        r"(?P<encoding>\S+)\s+(?P<embedded>yes|no)\s+"
        r"(?P<subset>yes|no)\s+(?P<unicode>yes|no)\s+"
        r"(?P<object>[0-9]+)\s+(?P<generation>[0-9]+)$"
    )
    for row in font_rows:
        match = font_pattern.fullmatch(row)
        _require(match is not None, f"pdffonts returned an unparsed row for {name}")
        assert match is not None
        _require(
            match.group("type") in {"TrueType", "CID TrueType"}
            and match.group("embedded") == "yes",
            f"{name} uses a font that is not embedded TrueType",
        )


def _validate_png_bytes(png: bytes, *, name: str) -> Mapping[str, Any]:
    _require(len(png) <= _MAX_FIGURE_BYTES, f"{name} exceeds 10 MiB")
    try:
        with Image.open(io.BytesIO(png)) as image:
            image.verify()
        with Image.open(io.BytesIO(png)) as image:
            image.load()
            dpi = image.info.get("dpi", (0.0, 0.0))
            _require(image.format == "PNG", f"{name} is not a PNG")
            _require(image.mode == "RGB", f"{name} is not RGB")
            _require(image.size == _EXPECTED_PIXELS, f"{name} dimensions changed")
            _require(
                abs(float(dpi[0]) - _DPI) < 0.5
                and abs(float(dpi[1]) - _DPI) < 0.5,
                f"{name} does not carry native 450-dpi metadata",
            )
    except (OSError, SyntaxError, ValueError) as error:
        raise FigureRenderError(f"{name} failed full PNG verification/load") from error
    return MappingProxyType(
        {
            "width_pixels": _EXPECTED_PIXELS[0],
            "height_pixels": _EXPECTED_PIXELS[1],
            "dpi": _DPI,
            "color_mode": "RGB",
            "fully_verified_and_loaded": True,
        }
    )


def _figure_bytes(fig: Any, *, title: str) -> tuple[bytes, bytes, Mapping[str, Any]]:
    pdf_stream = io.BytesIO()
    fig.savefig(
        pdf_stream,
        format="pdf",
        metadata={
            "Title": title,
            "Author": "NEGFPy",
            "Subject": "Protocol-v2 toy-gate visualization",
            "Creator": "NEGFPy production toy-gate figure renderer",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    pdf = pdf_stream.getvalue()
    _validate_pdf_bytes(pdf, name=f"rendered {title} PDF")

    raw_png = io.BytesIO()
    fig.savefig(raw_png, format="png", dpi=_DPI, facecolor="white")
    raw_png.seek(0)
    with Image.open(raw_png) as image:
        _require(
            image.size == _EXPECTED_PIXELS,
            f"native raster size is {image.size}, expected {_EXPECTED_PIXELS}",
        )
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    png_stream = io.BytesIO()
    rgb.save(
        png_stream,
        format="PNG",
        dpi=(_DPI, _DPI),
        optimize=False,
        compress_level=9,
    )
    png = png_stream.getvalue()
    png_metadata = _validate_png_bytes(png, name=f"rendered {title} PNG")
    return pdf, png, png_metadata


def _render_all(
    data: Mapping[str, Any],
    font: Mapping[str, Any],
) -> tuple[dict[str, bytes], dict[str, Any]]:
    builders = (
        ("figure_open_oracle_born", _render_open),
        ("figure_scba_validation", _render_scba),
        ("figure_passivity_backend_timing", _render_passivity),
    )
    files: dict[str, bytes] = {}
    metadata: dict[str, Any] = {}
    with plt.rc_context(_nature_rc(font)):
        for name, builder in builders:
            fig = builder(data, font)
            try:
                pdf, png, png_metadata = _figure_bytes(fig, title=name)
            finally:
                plt.close(fig)
            files[f"{name}.pdf"] = pdf
            files[f"{name}.png"] = png
            metadata[name] = {
                "panels": list(_FIGURE_SPECS[name]),
                "pdf": {
                    "path": f"{name}.pdf",
                    "sha256": _sha256_bytes(pdf),
                    "bytes": len(pdf),
                    "vector": True,
                    "raster_image_xobjects": False,
                    "embedded_truetype_fonts": True,
                    "type3_fonts": False,
                },
                "png": {
                    "path": f"{name}.png",
                    "sha256": _sha256_bytes(png),
                    "bytes": len(png),
                    **dict(png_metadata),
                },
            }
    return files, metadata


def _renderer_sha256() -> str:
    return _sha256_file(Path(__file__).resolve())


def _pdf_tool_identity() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for name, path in (
        ("pdfimages", _PDFIMAGES_PATH),
        ("pdffonts", _PDFFONTS_PATH),
    ):
        _require(
            path.is_file() and not path.is_symlink(),
            f"required structural PDF tool is unavailable: {path}",
        )
        result[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
    return result


def _manifest_core(
    staging: Mapping[str, Any],
    font: Mapping[str, Any],
    figures: Mapping[str, Any],
) -> dict[str, Any]:
    source = staging["source"]
    return {
        "schema": _MANIFEST_SCHEMA,
        "schema_version": 1,
        "renderer_sha256": _renderer_sha256(),
        "renderer_version": _RENDERER_VERSION,
        "source": {
            "commit": source["commit"],
            "attempt": source["attempt"],
            "report_relative_path": source["report_relative_path"],
            "remote_report_relative_path": source["remote_report_relative_path"],
            "transport_receipt_schema": source["transport_receipt_schema"],
            "report_file_sha256": source["report_file_sha256"],
            "import_receipt_file_sha256": staging["receipt_file_sha256"],
            "caller_expected_import_receipt_sha256": staging[
                "caller_expected_import_receipt_sha256"
            ],
            "import_receipt_body_sha256": staging["receipt_body_sha256"],
            "raw_report_binding": dict(staging["report_binding"]),
            "staging_sha256sums_file_sha256": staging[
                "staging_sha256sums_file_sha256"
            ],
        },
        "scope": {
            "input_contract_verified": True,
            "physics_solver_executed": False,
            "certification_decision_performed": False,
            "report_or_existing_figures_modified": False,
            "plotted_values": "staged CSV values and declared algebraic derivatives only",
        },
        "style": {
            "figure_size_inches": list(_FIGURE_SIZE),
            "png_dpi": _DPI,
            "png_color_mode": "RGB",
            "maximum_figure_bytes": _MAX_FIGURE_BYTES,
            "pdf_artwork": "vector_no_image_or_inline_image_records",
            "pdf_structural_validation": _pdf_tool_identity(),
            "body_font_points": [5.0, 7.0],
            "panel_label_points": 8.0,
            "panel_labels": "lowercase_bold",
            "gridlines": False,
            "pdf_fonttype": 42,
            "font": dict(font),
            "palette": dict(_PALETTE),
            "palette_standard": "Okabe_Ito_colorblind_accessible",
            "metric_shape_redundancy": True,
            "not_applicable_rendering": "gray_NA",
            "descriptive_not_gated_rendering": "colored_desc_text",
            "critical_rendering": "filled_circle",
            "diagnostic_rendering": "open_diamond",
        },
        "figures": dict(figures),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_output(
    output: Path,
    *,
    staging: Mapping[str, Any],
    font: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require(
        output.is_dir() and not output.is_symlink(),
        "existing output is not a regular directory",
    )
    entries = _parse_sha256sums(output)
    manifest = _strict_json_file(
        _safe_file(output, "render_manifest.json", name="render manifest"),
        name="render manifest",
    )
    _require(
        manifest.get("schema") == _MANIFEST_SCHEMA
        and manifest.get("schema_version") == 1
        and _is_sha256(manifest.get("manifest_sha256"))
        and manifest.get("manifest_sha256")
        == _body_digest(manifest, "manifest_sha256"),
        "existing render manifest schema/body SHA256 is invalid",
    )
    figures = _mapping(manifest.get("figures"), name="render_manifest.figures")
    expected_core = _manifest_core(staging, font, figures)
    observed_core = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    _require(observed_core == expected_core, "existing render source/style identity differs")
    expected_files = {"render_manifest.json"}
    for name in _FIGURE_SPECS:
        figure = _mapping(figures.get(name), name=f"manifest figure {name}")
        _require(
            figure.get("panels") == list(_FIGURE_SPECS[name]),
            f"existing figure panel specification differs for {name}",
        )
        for kind in ("pdf", "png"):
            record = _mapping(figure.get(kind), name=f"figure {name}.{kind}")
            relative = record.get("path")
            _require(
                relative == f"{name}.{kind}" and _is_sha256(record.get("sha256")),
                f"existing figure record is invalid for {name}.{kind}",
            )
            path = _safe_file(output, relative, name=f"existing {name}.{kind}")
            _require(
                record.get("sha256") == entries.get(relative)
                and record.get("sha256") == _sha256_file(path)
                and record.get("bytes") == path.stat().st_size,
                f"existing figure hash/size differs for {name}.{kind}",
            )
            payload = path.read_bytes()
            if kind == "pdf":
                _require(
                    record.get("vector") is True
                    and record.get("raster_image_xobjects") is False
                    and record.get("embedded_truetype_fonts") is True
                    and record.get("type3_fonts") is False,
                    f"existing PDF style record differs for {name}",
                )
                _validate_pdf_bytes(payload, name=f"existing {name} PDF")
            else:
                png_metadata = _validate_png_bytes(
                    payload, name=f"existing {name} PNG"
                )
                _require(
                    all(
                        record.get(key) == value
                        for key, value in png_metadata.items()
                    ),
                    f"existing PNG style record differs for {name}",
                )
            expected_files.add(relative)
    _require(set(entries) == expected_files, "existing output file set is unexpected")
    return manifest


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    _require(
        renameat2 is not None,
        "atomic no-overwrite directory publication requires renameat2",
    )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            str(destination),
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        str(destination),
    )


def _publish_atomic(output: Path, files: Mapping[str, bytes]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for relative_text, payload in sorted(files.items()):
            relative = PurePosixPath(relative_text)
            _require(
                not relative.is_absolute() and ".." not in relative.parts,
                f"unsafe output path {relative_text!r}",
            )
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(temporary)
        _rename_directory_noreplace(temporary, output)
        _fsync_directory(output.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def render_figures(
    staging_root: Path,
    output_directory: Path,
    *,
    expected_import_receipt_sha256: str,
    font_regular: Path | None = None,
    font_bold: Path | None = None,
) -> Mapping[str, Any]:
    """Validate staged evidence and atomically render three figure pairs."""

    supplied_root = staging_root.expanduser()
    _require(
        not supplied_root.is_symlink(),
        "supplied staging root is a symlink",
    )
    root = supplied_root.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), "staging root is unsafe")
    supplied_output = output_directory.expanduser()
    _require(
        not supplied_output.is_symlink(),
        "supplied figure output is a symlink",
    )
    output = supplied_output.resolve(strict=False)
    _require(
        output != root and not output.is_relative_to(root),
        "figure output cannot be inside the staged evidence directory",
    )
    staging_base = _validate_staging(
        root,
        expected_import_receipt_sha256=expected_import_receipt_sha256,
    )
    report_binding = _validate_raw_report_binding(root, staging_base)
    staging = MappingProxyType(
        {**dict(staging_base), "report_binding": report_binding}
    )
    data = _load_data(root)
    font = _resolve_font(font_regular, font_bold)
    if output.exists():
        _verify_output(output, staging=staging, font=font)
        return MappingProxyType(
            {
                "status": "unchanged",
                "output": str(output),
                "figure_count": 3,
                "input_contract_verified": True,
                "certification_decision_performed": False,
            }
        )

    files, figure_metadata = _render_all(data, font)
    core = _manifest_core(staging, font, figure_metadata)
    manifest = {
        **core,
        "manifest_sha256": _sha256_bytes(_canonical_json_bytes(core)),
    }
    files["render_manifest.json"] = _pretty_json_bytes(manifest)
    checksum_lines = [
        f"{_sha256_bytes(payload)}  {relative}"
        for relative, payload in sorted(files.items())
    ]
    files["SHA256SUMS"] = ("\n".join(checksum_lines) + "\n").encode("ascii")
    try:
        _publish_atomic(output, files)
    except FileExistsError:
        _verify_output(output, staging=staging, font=font)
        status = "unchanged"
    else:
        status = "created"
    return MappingProxyType(
        {
            "status": status,
            "output": str(output),
            "figure_count": 3,
            "input_contract_verified": True,
            "certification_decision_performed": False,
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render three Nature-style figure composites from a verified "
            "production toy-gate import staging directory."
        )
    )
    parser.add_argument("staging_root", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--expected-import-receipt-sha256",
        required=True,
        help="caller-pinned SHA256 of staging/import_receipt.json",
    )
    parser.add_argument(
        "--font-regular",
        type=Path,
        help="optional Arial/Helvetica-compatible regular TTF/OTF",
    )
    parser.add_argument(
        "--font-bold",
        type=Path,
        help="optional matching bold TTF/OTF",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = render_figures(
            arguments.staging_root,
            arguments.output_directory,
            expected_import_receipt_sha256=(
                arguments.expected_import_receipt_sha256
            ),
            font_regular=arguments.font_regular,
            font_bold=arguments.font_bold,
        )
    except (FigureRenderError, OSError, ValueError, TypeError) as error:
        print(f"production figure render refused: {error}", file=sys.stderr)
        return 2
    print(json.dumps(dict(result), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
