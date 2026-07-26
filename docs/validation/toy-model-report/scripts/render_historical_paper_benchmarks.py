#!/usr/bin/env python3
"""Render the isolated, NONCERTIFYING Wang/Mingo historical comparison.

The renderer consumes only five fixed CSV inputs from the report ``data``
directory.  Their identities must be present in a caller-pinned raw
``benchmark_manifest.json``.  The shared data directory may contain unrelated
production inputs; they are outside this renderer's fixed input set.

This lane is deliberately separate from production toy-gate evidence.  It
does not run NEGF calculations, decide certification, edit the report, or
render a Mingo Fig. 8 interacting curve. It emits a separate checksum-covered
manifest binding the recomputed historical metrics, selected data hashes, and
exact renderer implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import csv
import ctypes
import datetime as dt
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib import font_manager  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


_SCHEMA = "negfpy.anharmonic.historical_paper_benchmark_figure_render.v1"
_METRICS_SCHEMA = "negfpy.anharmonic.historical_paper_benchmark_metrics.v1"
_STATUS = "NONCERTIFYING_HISTORICAL_SENSITIVITY"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DPI = 450
_RENDERER_VERSION = "2"
_FIGURE_SIZE_IN = (7.0, 5.0)
_PNG_PIXELS = (3150, 2250)
_MAX_FIGURE_BYTES = 10 * 1024 * 1024

_INPUT_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "wang2006_harmonic.csv": (
        "temperature_K",
        "published_vector_nW_per_K",
        "negfpy_calculated_nW_per_K",
        "calculated_minus_published_nW_per_K",
    ),
    "wang2006_published_reference_curves.csv": (
        "approximation",
        "temperature_K",
        "published_vector_nW_per_K",
    ),
    "wang2006_anharmonic_checkpoints.csv": (
        "approximation",
        "temperature_K",
        "published_vector_nW_per_K",
        "negfpy_calculated_nW_per_K",
        "relative_error_percent",
        "iterations",
        "converged",
        "release_certified",
    ),
    "mingo2006_figure4.csv": (
        "temperature_K",
        "published_vector_pW_per_K",
        "negfpy_printed_normalized_pW_per_K",
        "negfpy_half_weight_sensitivity_pW_per_K",
        "normalized_minus_published_pW_per_K",
        "half_weight_minus_published_pW_per_K",
    ),
    "mingo2006_figure4_smooth_calculations.csv": (
        "temperature_K",
        "negfpy_printed_normalized_pW_per_K",
        "negfpy_half_weight_sensitivity_pW_per_K",
    ),
}
_EXPECTED_ROWS = {
    "wang2006_harmonic.csv": 51,
    "wang2006_anharmonic_checkpoints.csv": 5,
    "mingo2006_figure4.csv": 18,
    "mingo2006_figure4_smooth_calculations.csv": 601,
}
_WANG_APPROXIMATIONS = {
    "harmonic",
    "wang_born_t2",
    "wang_born_t4",
    "wang_selective_mean_field",
}
_CHECKPOINTS = {
    ("wang_born_t2", 100.0),
    ("wang_born_t2", 300.0),
    ("wang_born_t2", 500.0),
    ("wang_selective_mean_field", 300.0),
    ("wang_selective_mean_field", 400.0),
}
_PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#000000",
    "gray": "#7A7A7A",
}

_CAPTION = (
    "Historical, explicitly noncertifying comparison. Published paths and "
    "points were extracted from paper figures and are not author-provided raw "
    "data. Wang harmonic values are shown at all extracted temperatures; the "
    "O(T^2) and selective mean-field calculations are selected checkpoints "
    "conditional on the inferred onsite convention. Mingo Fig. 4 compares "
    "the printed-normalization Eq. 47 harmonic calculation with a separately "
    "named half-weight sensitivity. The half-weight curve is not a fitted "
    "production method. No Mingo Fig. 8 interacting curve was computed or "
    "shown. Nothing in this figure is production validation evidence."
)


class HistoricalRenderError(RuntimeError):
    """Fail-closed historical renderer error."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_without_duplicates(raw: bytes, *, label: str) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise HistoricalRenderError(
                    f"{label} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HistoricalRenderError(f"{label} is not strict UTF-8 JSON") from exc


def _reject_symlink_chain(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        if stat.S_ISLNK(mode):
            raise HistoricalRenderError(
                f"symlink path component is forbidden: {current}"
            )


def _regular_file(path: Path, *, label: str) -> Path:
    _reject_symlink_chain(path)
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise HistoricalRenderError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise HistoricalRenderError(f"{label} must be a regular non-symlink file")
    return path


def _data_root(path: Path) -> Path:
    _reject_symlink_chain(path)
    try:
        status = path.lstat()
    except FileNotFoundError as exc:
        raise HistoricalRenderError(f"missing data directory: {path}") from exc
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISDIR(status.st_mode):
        raise HistoricalRenderError(
            "data directory must be a real, non-symlink directory"
        )
    return path.resolve()


def _strict_csv(path: Path, expected_header: tuple[str, ...]) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if b"\x00" in raw:
        raise HistoricalRenderError(f"{path.name} contains a NUL byte")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HistoricalRenderError(f"{path.name} is not UTF-8") from exc
    rows_raw = list(csv.reader(text.splitlines(), strict=True))
    if not rows_raw:
        raise HistoricalRenderError(f"{path.name} is empty")
    if tuple(rows_raw[0]) != expected_header:
        raise HistoricalRenderError(
            f"{path.name} schema mismatch: expected {expected_header}, "
            f"got {tuple(rows_raw[0])}"
        )
    if len(set(rows_raw[0])) != len(rows_raw[0]):
        raise HistoricalRenderError(f"{path.name} has duplicate columns")
    result: list[dict[str, str]] = []
    for line_number, values in enumerate(rows_raw[1:], start=2):
        if len(values) != len(expected_header):
            raise HistoricalRenderError(
                f"{path.name}:{line_number} has {len(values)} fields; "
                f"expected {len(expected_header)}"
            )
        if any(value != value.strip() for value in values):
            raise HistoricalRenderError(
                f"{path.name}:{line_number} has padded field data"
            )
        result.append(dict(zip(expected_header, values, strict=True)))
    return result


def _finite(row: Mapping[str, str], name: str, *, label: str) -> float:
    raw = row[name]
    try:
        value = float(raw)
    except ValueError as exc:
        raise HistoricalRenderError(f"{label}.{name} is not numeric") from exc
    if not math.isfinite(value):
        raise HistoricalRenderError(f"{label}.{name} must be finite")
    return value


def _positive_int(row: Mapping[str, str], name: str, *, label: str) -> int:
    raw = row[name]
    if not re.fullmatch(r"[1-9][0-9]*", raw):
        raise HistoricalRenderError(f"{label}.{name} must be a positive integer")
    return int(raw)


def _strict_bool(row: Mapping[str, str], name: str, *, label: str) -> bool:
    raw = row[name]
    if raw not in {"True", "False"}:
        raise HistoricalRenderError(f"{label}.{name} must be True or False")
    return raw == "True"


def _assert_close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=2.0e-13, abs_tol=2.0e-15):
        raise HistoricalRenderError(
            f"{label} is inconsistent: {actual!r} != {expected!r}"
        )


def _strictly_increasing(values: Sequence[float], *, label: str) -> None:
    if any(right <= left for left, right in zip(values, values[1:])):
        raise HistoricalRenderError(f"{label} must be strictly increasing")


def _error_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise HistoricalRenderError("cannot summarize an empty error series")
    return {
        "count": len(values),
        "rms": math.sqrt(math.fsum(value * value for value in values) / len(values)),
        "max_abs": max(abs(value) for value in values),
    }


def _validate_semantics(
    tables: Mapping[str, list[dict[str, str]]],
    parent_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    for name, expected in _EXPECTED_ROWS.items():
        if len(tables[name]) != expected:
            raise HistoricalRenderError(
                f"{name} row count is {len(tables[name])}; expected {expected}"
            )

    harmonic = tables["wang2006_harmonic.csv"]
    h_temp: list[float] = []
    h_pub: list[float] = []
    h_calc: list[float] = []
    h_error: list[float] = []
    for index, row in enumerate(harmonic):
        label = f"wang harmonic row {index}"
        temperature = _finite(row, "temperature_K", label=label)
        published = _finite(row, "published_vector_nW_per_K", label=label)
        calculated = _finite(row, "negfpy_calculated_nW_per_K", label=label)
        residual = _finite(
            row, "calculated_minus_published_nW_per_K", label=label
        )
        _assert_close(residual, calculated - published, label=f"{label} residual")
        h_temp.append(temperature)
        h_pub.append(published)
        h_calc.append(calculated)
        h_error.append(residual)
    _strictly_increasing(h_temp, label="Wang harmonic temperatures")

    reference = tables["wang2006_published_reference_curves.csv"]
    reference_by_kind: dict[str, list[tuple[float, float]]] = {}
    seen_reference: set[tuple[str, float]] = set()
    for index, row in enumerate(reference):
        label = f"Wang reference row {index}"
        kind = row["approximation"]
        if kind not in _WANG_APPROXIMATIONS:
            raise HistoricalRenderError(f"{label} has unknown approximation {kind!r}")
        temperature = _finite(row, "temperature_K", label=label)
        published = _finite(row, "published_vector_nW_per_K", label=label)
        key = (kind, temperature)
        if key in seen_reference:
            raise HistoricalRenderError(f"duplicate Wang reference point {key}")
        seen_reference.add(key)
        reference_by_kind.setdefault(kind, []).append((temperature, published))
    if set(reference_by_kind) != _WANG_APPROXIMATIONS:
        raise HistoricalRenderError("Wang reference approximation set is incomplete")
    for kind, points in reference_by_kind.items():
        _strictly_increasing(
            [point[0] for point in points], label=f"Wang {kind} temperatures"
        )
    if reference_by_kind["harmonic"] != list(zip(h_temp, h_pub, strict=True)):
        raise HistoricalRenderError(
            "Wang harmonic table and extracted harmonic reference path differ"
        )

    checkpoints = tables["wang2006_anharmonic_checkpoints.csv"]
    checkpoint_keys: set[tuple[str, float]] = set()
    checkpoint_rows: dict[str, list[dict[str, float | int]]] = {
        "wang_born_t2": [],
        "wang_selective_mean_field": [],
    }
    for index, row in enumerate(checkpoints):
        label = f"Wang checkpoint row {index}"
        kind = row["approximation"]
        if kind not in checkpoint_rows:
            raise HistoricalRenderError(f"{label} has unsupported checkpoint {kind!r}")
        temperature = _finite(row, "temperature_K", label=label)
        published = _finite(row, "published_vector_nW_per_K", label=label)
        calculated = _finite(row, "negfpy_calculated_nW_per_K", label=label)
        relative = _finite(row, "relative_error_percent", label=label)
        iterations = _positive_int(row, "iterations", label=label)
        if not _strict_bool(row, "converged", label=label):
            raise HistoricalRenderError(f"{label} is not converged")
        if _strict_bool(row, "release_certified", label=label):
            raise HistoricalRenderError(
                f"{label} may not be release-certified in the historical lane"
            )
        _assert_close(
            relative,
            100.0 * (calculated - published) / published,
            label=f"{label} relative error",
        )
        key = (kind, temperature)
        if key in checkpoint_keys:
            raise HistoricalRenderError(f"duplicate checkpoint {key}")
        checkpoint_keys.add(key)
        checkpoint_rows[kind].append(
            {
                "temperature": temperature,
                "published": published,
                "calculated": calculated,
                "absolute_error": calculated - published,
                "relative_error_percent": relative,
                "iterations": iterations,
            }
        )
    if checkpoint_keys != _CHECKPOINTS:
        raise HistoricalRenderError(
            f"Wang checkpoint set mismatch: {checkpoint_keys!r}"
        )
    for rows in checkpoint_rows.values():
        rows.sort(key=lambda item: float(item["temperature"]))

    mingo = tables["mingo2006_figure4.csv"]
    m_temp: list[float] = []
    m_pub: list[float] = []
    m_normalized: list[float] = []
    m_half: list[float] = []
    m_normalized_error: list[float] = []
    m_half_error: list[float] = []
    for index, row in enumerate(mingo):
        label = f"Mingo Fig. 4 row {index}"
        temperature = _finite(row, "temperature_K", label=label)
        published = _finite(row, "published_vector_pW_per_K", label=label)
        normalized = _finite(
            row, "negfpy_printed_normalized_pW_per_K", label=label
        )
        half = _finite(
            row, "negfpy_half_weight_sensitivity_pW_per_K", label=label
        )
        normalized_error = _finite(
            row, "normalized_minus_published_pW_per_K", label=label
        )
        half_error = _finite(
            row, "half_weight_minus_published_pW_per_K", label=label
        )
        _assert_close(
            normalized_error,
            normalized - published,
            label=f"{label} normalized residual",
        )
        _assert_close(
            half_error, half - published, label=f"{label} half-weight residual"
        )
        m_temp.append(temperature)
        m_pub.append(published)
        m_normalized.append(normalized)
        m_half.append(half)
        m_normalized_error.append(normalized_error)
        m_half_error.append(half_error)
    _strictly_increasing(m_temp, label="Mingo Fig. 4 temperatures")

    smooth = tables["mingo2006_figure4_smooth_calculations.csv"]
    smooth_temp: list[float] = []
    smooth_normalized: list[float] = []
    smooth_half: list[float] = []
    for index, row in enumerate(smooth):
        label = f"Mingo smooth row {index}"
        temperature = _finite(row, "temperature_K", label=label)
        normalized = _finite(
            row, "negfpy_printed_normalized_pW_per_K", label=label
        )
        half = _finite(
            row, "negfpy_half_weight_sensitivity_pW_per_K", label=label
        )
        if temperature != float(index):
            raise HistoricalRenderError(
                "Mingo smooth temperatures must be exactly 0, 1, ..., 600 K"
            )
        smooth_temp.append(temperature)
        smooth_normalized.append(normalized)
        smooth_half.append(half)
    _strictly_increasing(smooth_temp, label="Mingo smooth temperatures")

    metrics = {
        "wang_harmonic_error_nW_per_K": _error_summary(h_error),
        "wang_born_t2_checkpoint_absolute_error_nW_per_K": _error_summary(
            [
                float(row["absolute_error"])
                for row in checkpoint_rows["wang_born_t2"]
            ]
        ),
        "wang_born_t2_checkpoint_relative_error_percent": _error_summary(
            [
                float(row["relative_error_percent"])
                for row in checkpoint_rows["wang_born_t2"]
            ]
        ),
        "wang_selective_mean_field_checkpoint_absolute_error_nW_per_K": (
            _error_summary(
                [
                    float(row["absolute_error"])
                    for row in checkpoint_rows["wang_selective_mean_field"]
                ]
            )
        ),
        "wang_selective_mean_field_checkpoint_relative_error_percent": (
            _error_summary(
                [
                    float(row["relative_error_percent"])
                    for row in checkpoint_rows["wang_selective_mean_field"]
                ]
            )
        ),
        "mingo_fig4_printed_normalization_error_pW_per_K": _error_summary(
            m_normalized_error
        ),
        "mingo_fig4_half_weight_sensitivity_error_pW_per_K": _error_summary(
            m_half_error
        ),
    }

    try:
        wang_stage = parent_manifest["stage_results"]["wang2006"]
        mingo_stage = parent_manifest["stage_results"]["mingo2006"]
    except (KeyError, TypeError) as exc:
        raise HistoricalRenderError(
            "benchmark manifest lacks Wang/Mingo stage summaries"
        ) from exc
    if not isinstance(wang_stage, Mapping) or not isinstance(mingo_stage, Mapping):
        raise HistoricalRenderError("benchmark stage summaries must be objects")
    _assert_close(
        float(metrics["wang_harmonic_error_nW_per_K"]["rms"]),
        float(wang_stage["harmonic_rms_error_nW_per_K"]),
        label="manifest Wang harmonic RMS",
    )
    _assert_close(
        float(metrics["wang_harmonic_error_nW_per_K"]["max_abs"]),
        float(wang_stage["harmonic_max_abs_error_nW_per_K"]),
        label="manifest Wang harmonic maximum error",
    )
    _assert_close(
        float(metrics["mingo_fig4_printed_normalization_error_pW_per_K"]["rms"]),
        float(mingo_stage["normalized_rms_error_pW_per_K"]),
        label="manifest Mingo normalized RMS",
    )
    _assert_close(
        float(
            metrics["mingo_fig4_printed_normalization_error_pW_per_K"]["max_abs"]
        ),
        float(mingo_stage["normalized_max_abs_error_pW_per_K"]),
        label="manifest Mingo normalized maximum error",
    )
    _assert_close(
        float(metrics["mingo_fig4_half_weight_sensitivity_error_pW_per_K"]["rms"]),
        float(mingo_stage["half_weight_rms_error_pW_per_K"]),
        label="manifest Mingo half-weight RMS",
    )
    _assert_close(
        float(
            metrics["mingo_fig4_half_weight_sensitivity_error_pW_per_K"][
                "max_abs"
            ]
        ),
        float(mingo_stage["half_weight_max_abs_error_pW_per_K"]),
        label="manifest Mingo half-weight maximum error",
    )

    return {
        "harmonic": {
            "temperature": h_temp,
            "published": h_pub,
            "calculated": h_calc,
        },
        "reference_by_kind": reference_by_kind,
        "checkpoints": checkpoint_rows,
        "mingo": {
            "temperature": m_temp,
            "published": m_pub,
            "normalized": m_normalized,
            "half": m_half,
            "normalized_error": m_normalized_error,
            "half_error": m_half_error,
        },
        "mingo_smooth": {
            "temperature": smooth_temp,
            "normalized": smooth_normalized,
            "half": smooth_half,
        },
        "metrics": metrics,
    }


def load_historical_inputs(
    *,
    data_dir: Path,
    benchmark_manifest: Path,
    expected_benchmark_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    """Verify identities/schemas and return validated plot data."""

    if not _SHA256_RE.fullmatch(expected_benchmark_manifest_sha256):
        raise HistoricalRenderError(
            "expected benchmark manifest SHA256 must be 64 lowercase hex digits"
        )
    root = _data_root(data_dir)
    manifest_path = _regular_file(
        benchmark_manifest, label="benchmark manifest"
    ).resolve()
    if manifest_path.parent != root:
        raise HistoricalRenderError(
            "benchmark manifest must be directly inside the data directory"
        )
    manifest_raw = manifest_path.read_bytes()
    actual_manifest_sha256 = _sha256_bytes(manifest_raw)
    if not hmac.compare_digest(
        actual_manifest_sha256, expected_benchmark_manifest_sha256
    ):
        raise HistoricalRenderError(
            "benchmark manifest does not match caller-pinned raw SHA256"
        )
    parsed = _json_without_duplicates(manifest_raw, label="benchmark manifest")
    if not isinstance(parsed, Mapping):
        raise HistoricalRenderError("benchmark manifest root must be an object")
    if parsed.get("generator") != "scripts/generate_benchmarks.py":
        raise HistoricalRenderError("unexpected benchmark manifest generator")
    hashes = parsed.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise HistoricalRenderError("benchmark manifest lacks artifact_sha256")

    tables: dict[str, list[dict[str, str]]] = {}
    input_hashes: dict[str, str] = {}
    for name, schema in _INPUT_SCHEMAS.items():
        path = _regular_file(root / name, label=name)
        if path.resolve().parent != root:
            raise HistoricalRenderError(f"{name} escapes the data directory")
        key = f"data/{name}"
        expected = hashes.get(key)
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise HistoricalRenderError(
                f"benchmark manifest lacks a valid hash for {key}"
            )
        actual = _sha256_file(path)
        if not hmac.compare_digest(actual, expected):
            raise HistoricalRenderError(f"{name} hash mismatch")
        input_hashes[key] = actual
        tables[name] = _strict_csv(path, schema)

    validated = _validate_semantics(tables, parsed)
    source = {
        "benchmark_manifest_basename": manifest_path.name,
        "benchmark_manifest_raw_sha256": actual_manifest_sha256,
        "selected_file_set": sorted(input_hashes),
        "selected_file_sha256": dict(sorted(input_hashes.items())),
        "shared_data_directory_policy": (
            "Only the exact selected_file_set is consumed; unrelated colocated "
            "report data are out of scope."
        ),
    }
    return validated, input_hashes, source


def _font(path: Path, *, role: str) -> tuple[font_manager.FontProperties, dict[str, str]]:
    path = _regular_file(path, label=f"{role} font")
    if path.suffix.lower() != ".ttf":
        raise HistoricalRenderError(f"{role} font must be a .ttf file")
    signature = path.read_bytes()[:4]
    if signature != b"\x00\x01\x00\x00":
        raise HistoricalRenderError(f"{role} font is not a standalone TrueType font")
    face = FT2Font(str(path))
    family = str(face.family_name)
    style = str(face.style_name)
    lowered = family.lower()
    if "sans" not in lowered or "serif" in lowered:
        raise HistoricalRenderError(
            f"{role} font family must be an explicit sans family, got {family!r}"
        )
    if role == "regular" and "bold" in style.lower():
        raise HistoricalRenderError("regular font unexpectedly has a bold style")
    if role == "bold" and "bold" not in style.lower():
        raise HistoricalRenderError("bold font style must identify itself as bold")
    font_manager.fontManager.addfont(str(path))
    properties = font_manager.FontProperties(fname=str(path))
    return properties, {
        "basename": path.name,
        "sha256": _sha256_file(path),
        "family": family,
        "style": style,
    }


def _panel_label(ax: Any, label: str, bold_font: font_manager.FontProperties) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
        fontproperties=bold_font,
        color=_PALETTE["black"],
    )


def _render_figure(
    data: Mapping[str, Any],
    pdf_path: Path,
    png_path: Path,
    *,
    regular_font: font_manager.FontProperties,
    bold_font: font_manager.FontProperties,
    font_family: str,
) -> None:
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": [font_family],
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    with plt.rc_context(rc):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=_FIGURE_SIZE_IN,
            constrained_layout=False,
        )
        figure.subplots_adjust(
            left=0.09, right=0.985, top=0.94, bottom=0.12, wspace=0.31, hspace=0.36
        )

        harmonic = data["harmonic"]
        ax = axes[0, 0]
        ax.plot(
            harmonic["temperature"],
            harmonic["calculated"],
            color=_PALETTE["blue"],
            label="NEGFPy harmonic",
            zorder=2,
        )
        ax.scatter(
            harmonic["temperature"],
            harmonic["published"],
            s=9,
            facecolors="white",
            edgecolors=_PALETTE["orange"],
            linewidths=0.7,
            label="Wang Fig. path (extracted)",
            zorder=3,
        )
        ax.set(
            xlabel="temperature (K)",
            ylabel="conductance (nW K^-1)",
            title="Wang harmonic - all extracted temperatures",
        )
        ax.set_xlim(0.0, 1020.0)
        ax.set_ylim(bottom=-0.01)
        ax.legend(frameon=False, loc="lower right")
        _panel_label(ax, "a", bold_font)

        ax = axes[0, 1]
        reference = data["reference_by_kind"]
        for kind, color, label in (
            ("wang_born_t2", _PALETTE["purple"], "Wang O(T^2) path (extracted)"),
            (
                "wang_selective_mean_field",
                _PALETTE["green"],
                "Wang selective MF path (extracted)",
            ),
        ):
            points = reference[kind]
            ax.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=color,
                alpha=0.78,
                label=label,
            )
        marker_contract = (
            ("wang_born_t2", "s", _PALETTE["purple"], "NEGFPy O(T^2) checkpoints"),
            (
                "wang_selective_mean_field",
                "D",
                _PALETTE["green"],
                "NEGFPy selective MF checkpoints",
            ),
        )
        for kind, marker, color, label in marker_contract:
            rows = data["checkpoints"][kind]
            ax.scatter(
                [row["temperature"] for row in rows],
                [row["published"] for row in rows],
                s=19,
                marker=marker,
                facecolors="white",
                edgecolors=color,
                linewidths=0.8,
                zorder=3,
            )
            ax.scatter(
                [row["temperature"] for row in rows],
                [row["calculated"] for row in rows],
                s=13,
                marker=marker,
                facecolors=color,
                edgecolors=_PALETTE["black"],
                linewidths=0.35,
                label=label,
                zorder=4,
            )
        ax.set(
            xlabel="temperature (K)",
            ylabel="conductance (nW K^-1)",
            title="Wang selected anharmonic checkpoints",
        )
        ax.set_xlim(0.0, 600.0)
        ax.set_ylim(bottom=-0.005)
        ax.legend(frameon=False, loc="lower right")
        ax.text(
            0.03,
            0.96,
            "conditional: inferred onsite convention",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            fontproperties=regular_font,
        )
        _panel_label(ax, "b", bold_font)

        mingo = data["mingo"]
        smooth = data["mingo_smooth"]
        ax = axes[1, 0]
        ax.plot(
            smooth["temperature"],
            smooth["normalized"],
            color=_PALETTE["vermillion"],
            label="NEGFPy Eq. 47 printed normalization",
        )
        ax.plot(
            smooth["temperature"],
            smooth["half"],
            color=_PALETTE["blue"],
            linestyle="--",
            label="half-weight sensitivity (noncert.)",
        )
        ax.scatter(
            mingo["temperature"],
            mingo["published"],
            s=12,
            facecolors="white",
            edgecolors=_PALETTE["black"],
            linewidths=0.7,
            label="Mingo Fig. 4 points (extracted)",
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set(
            xlabel="temperature (K)",
            ylabel="conductance (pW K^-1)",
            title="Mingo Fig. 4 harmonic comparison",
        )
        ax.set_xlim(7.5, 650.0)
        ax.set_ylim(bottom=-0.4)
        ax.legend(frameon=False, loc="lower right")
        _panel_label(ax, "c", bold_font)

        ax = axes[1, 1]
        ax.axhline(0.0, color=_PALETTE["black"], linewidth=0.65)
        ax.plot(
            mingo["temperature"],
            mingo["normalized_error"],
            marker="o",
            markersize=2.5,
            color=_PALETTE["vermillion"],
            label="printed norm. minus extracted",
        )
        ax.plot(
            mingo["temperature"],
            mingo["half_error"],
            marker="s",
            markersize=2.3,
            linestyle="--",
            color=_PALETTE["blue"],
            label="half-weight minus extracted",
        )
        ax.set_xscale("log")
        ax.set(
            xlabel="temperature (K)",
            ylabel="calculated - extracted (pW K^-1)",
            title="Mingo discrepancy and sensitivity",
        )
        ax.set_xlim(7.5, 650.0)
        ax.legend(frameon=False, loc="upper left")
        norm = data["metrics"][
            "mingo_fig4_printed_normalization_error_pW_per_K"
        ]
        half = data["metrics"][
            "mingo_fig4_half_weight_sensitivity_error_pW_per_K"
        ]
        ax.text(
            0.98,
            0.04,
            (
                f"RMS/max (pW K^-1)\n"
                f"printed: {norm['rms']:.3g}/{norm['max_abs']:.3g}\n"
                f"half: {half['rms']:.3g}/{half['max_abs']:.3g}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.5,
            fontproperties=regular_font,
        )
        _panel_label(ax, "d", bold_font)

        for ax in axes.flat:
            ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(direction="out", length=2.5, width=0.65)

        figure.text(
            0.5,
            0.025,
            (
                "NONCERTIFYING historical comparison | figure-extracted, not "
                "author raw data | no computed Mingo Fig. 8 interacting curve"
            ),
            ha="center",
            va="bottom",
            fontsize=5.5,
            fontproperties=bold_font,
            color=_PALETTE["black"],
        )

        source_date = dt.datetime(
            2026, 7, 24, 14, 45, 9, tzinfo=dt.timezone.utc
        )
        metadata = {
            "Title": "NONCERTIFYING Wang and Mingo historical comparison",
            "Author": "NEGFPy validation tooling",
            "Subject": (
                "Figure-extracted historical sensitivity; not production "
                "evidence; no computed Mingo Fig. 8 interacting curve"
            ),
            "Keywords": (
                "NONCERTIFYING; figure-extracted; Wang 2006; Mingo 2006; "
                "historical sensitivity"
            ),
            "Creator": "render_historical_paper_benchmarks.py",
            "CreationDate": source_date,
            "ModDate": source_date,
        }
        figure.savefig(pdf_path, format="pdf", metadata=metadata)
        raw_png = png_path.with_name(f".{png_path.name}.rgba.png")
        figure.savefig(raw_png, format="png", dpi=_DPI, transparent=False)
        plt.close(figure)

    with Image.open(raw_png) as image:
        rgb = image.convert("RGB")
        rgb.save(png_path, format="PNG", dpi=(_DPI, _DPI), optimize=True)
    raw_png.unlink()


def _run_poppler(arguments: Sequence[str]) -> str:
    executable = Path(arguments[0])
    _regular_file(executable, label=f"Poppler executable {executable.name}")
    try:
        completed = subprocess.run(
            list(arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=45,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HistoricalRenderError(
            f"Poppler validation failed: {' '.join(arguments)}"
        ) from exc
    return completed.stdout


def _validate_pdf(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_FIGURE_BYTES:
        raise HistoricalRenderError("historical PDF exceeds 10 MiB")
    info = _run_poppler(("/usr/bin/pdfinfo", str(path)))
    if not re.search(r"^Pages:\s+1\s*$", info, flags=re.MULTILINE):
        raise HistoricalRenderError("historical PDF must contain exactly one page")
    size_match = re.search(
        r"^Page size:\s+([0-9.]+) x ([0-9.]+) pts", info, flags=re.MULTILINE
    )
    if size_match is None:
        raise HistoricalRenderError("Poppler did not report a PDF page size")
    width, height = map(float, size_match.groups())
    if not (
        math.isclose(width, 504.0, abs_tol=0.2)
        and math.isclose(height, 360.0, abs_tol=0.2)
    ):
        raise HistoricalRenderError(f"unexpected PDF page size {width} x {height}")
    if not re.search(
        r"^Subject:.*not production evidence.*no computed Mingo Fig\. 8",
        info,
        flags=re.MULTILINE,
    ):
        raise HistoricalRenderError("PDF noncertifying subject metadata is missing")

    images = _run_poppler(("/usr/bin/pdfimages", "-list", str(path)))
    image_rows = [
        line
        for line in images.splitlines()
        if re.match(r"^\s*[0-9]+\s+[0-9]+\s+", line)
    ]
    if image_rows:
        raise HistoricalRenderError("vector PDF unexpectedly contains raster images")

    fonts = _run_poppler(("/usr/bin/pdffonts", str(path)))
    font_rows = [
        line
        for line in fonts.splitlines()
        if line.strip()
        and not line.startswith("name ")
        and re.fullmatch(r"-+(?:\s+-+)+", line.strip()) is None
    ]
    if not font_rows:
        raise HistoricalRenderError("PDF contains no inspectable fonts")
    for line in font_rows:
        if "Type 3" in line or "TrueType" not in line:
            raise HistoricalRenderError(f"PDF font is not TrueType: {line}")
        if not re.search(r"\s+yes\s+yes\s+(?:yes|no)\s+\d+\s+\d+\s*$", line):
            raise HistoricalRenderError(
                f"PDF font is not embedded and subset as required: {line}"
            )

    with tempfile.TemporaryDirectory(prefix="historical-pdf-render-") as temp:
        prefix = Path(temp) / "page"
        _run_poppler(
            (
                "/usr/bin/pdftoppm",
                "-r",
                "150",
                "-singlefile",
                "-png",
                str(path),
                str(prefix),
            )
        )
        rendered = prefix.with_suffix(".png")
        with Image.open(rendered) as image:
            image.verify()
        with Image.open(rendered) as image:
            image.load()
            if image.size != (1050, 750):
                raise HistoricalRenderError(
                    f"unexpected Poppler render size {image.size}"
                )
    return {
        "pages": 1,
        "page_size_points": [width, height],
        "raster_image_count": 0,
        "font_rows": font_rows,
        "poppler_render_pixels_at_150_dpi": [1050, 750],
    }


def _validate_png(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_FIGURE_BYTES:
        raise HistoricalRenderError("historical PNG exceeds 10 MiB")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG":
            raise HistoricalRenderError("raster companion is not PNG")
        if image.mode != "RGB":
            raise HistoricalRenderError(f"PNG mode must be RGB, got {image.mode}")
        if image.size != _PNG_PIXELS:
            raise HistoricalRenderError(
                f"PNG dimensions must be {_PNG_PIXELS}, got {image.size}"
            )
        dpi = image.info.get("dpi")
        if not isinstance(dpi, tuple) or len(dpi) != 2:
            raise HistoricalRenderError("PNG lacks DPI metadata")
        if not (
            math.isclose(float(dpi[0]), _DPI, abs_tol=0.1)
            and math.isclose(float(dpi[1]), _DPI, abs_tol=0.1)
        ):
            raise HistoricalRenderError(f"PNG DPI must be 450, got {dpi}")
    return {
        "mode": "RGB",
        "pixels": list(_PNG_PIXELS),
        "dpi": [_DPI, _DPI],
    }


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise HistoricalRenderError(
            "atomic no-replace directory publication requires Linux renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
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
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise HistoricalRenderError(f"output directory already exists: {destination}")
    raise HistoricalRenderError(
        f"atomic output publication failed: {os.strerror(error)}"
    )


def _verify_output_tree(path: Path) -> None:
    expected = {
        "historical_paper_benchmarks.pdf",
        "historical_paper_benchmarks.png",
        "historical_paper_benchmarks_render_manifest.json",
        "historical_paper_benchmarks_metrics.json",
        "SHA256SUMS",
    }
    actual: set[str] = set()
    for entry in path.iterdir():
        status = entry.lstat()
        if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
            raise HistoricalRenderError(
                f"output contains a special file or symlink: {entry.name}"
            )
        actual.add(entry.name)
    if actual != expected:
        raise HistoricalRenderError(
            f"output file set mismatch: expected {expected}, got {actual}"
        )
    lines = (path / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    expected_covered = expected - {"SHA256SUMS"}
    covered: set[str] = set()
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None:
            raise HistoricalRenderError("malformed SHA256SUMS line")
        digest, name = match.groups()
        if name in covered or name not in expected_covered:
            raise HistoricalRenderError("SHA256SUMS has duplicate/unexpected coverage")
        if not hmac.compare_digest(digest, _sha256_file(path / name)):
            raise HistoricalRenderError(f"SHA256SUMS mismatch for {name}")
        covered.add(name)
    if covered != expected_covered:
        raise HistoricalRenderError("SHA256SUMS coverage is incomplete")


def render_historical_benchmarks(
    *,
    data_dir: Path,
    benchmark_manifest: Path,
    expected_benchmark_manifest_sha256: str,
    regular_font_path: Path,
    bold_font_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Verify inputs, render, structurally validate, and publish once."""

    output_dir = output_dir.absolute()
    _reject_symlink_chain(output_dir.parent)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_chain(output_dir.parent)
    if not stat.S_ISDIR(output_dir.parent.lstat().st_mode):
        raise HistoricalRenderError("output parent is not a directory")
    if output_dir.exists() or output_dir.is_symlink():
        raise HistoricalRenderError(f"output directory already exists: {output_dir}")

    data, input_hashes, source = load_historical_inputs(
        data_dir=data_dir,
        benchmark_manifest=benchmark_manifest,
        expected_benchmark_manifest_sha256=expected_benchmark_manifest_sha256,
    )
    regular_font, regular_identity = _font(regular_font_path, role="regular")
    bold_font, bold_identity = _font(bold_font_path, role="bold")
    if regular_identity["family"] != bold_identity["family"]:
        raise HistoricalRenderError("regular and bold font families must match")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    published = False
    try:
        pdf_name = "historical_paper_benchmarks.pdf"
        png_name = "historical_paper_benchmarks.png"
        pdf_path = temporary / pdf_name
        png_path = temporary / png_name
        _render_figure(
            data,
            pdf_path,
            png_path,
            regular_font=regular_font,
            bold_font=bold_font,
            font_family=regular_identity["family"],
        )
        pdf_validation = _validate_pdf(pdf_path)
        png_validation = _validate_png(png_path)
        figure_hashes = {
            pdf_name: _sha256_file(pdf_path),
            png_name: _sha256_file(png_path),
        }
        renderer_path = _regular_file(Path(__file__), label="renderer source")
        renderer_identity = {
            "basename": renderer_path.name,
            "version": _RENDERER_VERSION,
            "sha256": _sha256_file(renderer_path),
        }
        metrics_core = {
            "schema": _METRICS_SCHEMA,
            "status": _STATUS,
            "release_certified": False,
            "production_evidence": False,
            "source": source,
            "input_sha256": dict(sorted(input_hashes.items())),
            "metrics_recomputed_from_csv": data["metrics"],
            "renderer": renderer_identity,
        }
        metrics_document = {
            **metrics_core,
            "manifest_sha256": _sha256_bytes(_canonical_json(metrics_core).rstrip(b"\n")),
        }
        metrics_name = "historical_paper_benchmarks_metrics.json"
        (temporary / metrics_name).write_bytes(_canonical_json(metrics_document))
        manifest = {
            "schema": _SCHEMA,
            "status": _STATUS,
            "release_certified": False,
            "production_evidence": False,
            "caption_metadata": {
                "caption": _CAPTION,
                "published_data_provenance": (
                    "figure-extracted paths/points; not author raw data"
                ),
                "wang_convention_scope": "conditional_inferred_onsite_convention",
                "mingo_half_weight_scope": (
                    "separately_named_noncertifying_sensitivity"
                ),
                "mingo_fig8_interacting_curve": "not_computed_not_shown",
            },
            "source": source,
            "input_sha256": dict(sorted(input_hashes.items())),
            "metrics_recomputed_from_csv": data["metrics"],
            "figure_sha256": figure_hashes,
            "figure_validation": {
                "pdf": pdf_validation,
                "png": png_validation,
            },
            "fonts": {
                "regular": regular_identity,
                "bold": bold_identity,
                "policy": "caller-selected TrueType sans fonts; embedded as Type 42",
            },
            "renderer": renderer_identity,
            "metrics_manifest": {
                "path": metrics_name,
                "sha256": _sha256_file(temporary / metrics_name),
            },
            "output_contract": {
                "atomic_noreplace": True,
                "single_multipanel_figure": True,
                "vector_pdf": True,
                "rgb_png_450_dpi": True,
                "sha256sums_exact_coverage": True,
            },
        }
        manifest_name = "historical_paper_benchmarks_render_manifest.json"
        (temporary / manifest_name).write_bytes(_canonical_json(manifest))
        checksum_names = [pdf_name, png_name, manifest_name, metrics_name]
        checksum_text = "".join(
            f"{_sha256_file(temporary / name)}  {name}\n"
            for name in sorted(checksum_names)
        )
        (temporary / "SHA256SUMS").write_text(checksum_text, encoding="ascii")
        _verify_output_tree(temporary)
        _rename_noreplace(temporary, output_dir)
        published = True
        _verify_output_tree(output_dir)
        return manifest
    finally:
        if not published and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render an explicitly NONCERTIFYING Wang/Mingo historical figure "
            "from caller-pinned report data."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-benchmark-manifest-sha256",
        required=True,
        help="Caller-trusted SHA256 of raw benchmark_manifest.json bytes.",
    )
    parser.add_argument("--regular-font-path", type=Path, required=True)
    parser.add_argument("--bold-font-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        manifest = render_historical_benchmarks(
            data_dir=arguments.data_dir,
            benchmark_manifest=arguments.benchmark_manifest,
            expected_benchmark_manifest_sha256=(
                arguments.expected_benchmark_manifest_sha256
            ),
            regular_font_path=arguments.regular_font_path,
            bold_font_path=arguments.bold_font_path,
            output_dir=arguments.output_dir,
        )
    except HistoricalRenderError as exc:
        print(f"historical render refused: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "release_certified": manifest["release_certified"],
                "production_evidence": manifest["production_evidence"],
                "output_dir": str(arguments.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
