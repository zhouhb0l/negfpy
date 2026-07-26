from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any

import pytest

if any(importlib.util.find_spec(name) is None for name in ("matplotlib", "PIL")):
    pytest.skip(
        "install negfpy[report] for report rendering tests",
        allow_module_level=True,
    )
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "render_historical_paper_benchmarks.py"
SOURCE_DATA = ROOT / "historical-data"
SOURCE_MANIFEST = SOURCE_DATA / "benchmark_manifest.json"
REGULAR_FONT = Path(
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
)
BOLD_FONT = Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")
SELECTED = (
    "wang2006_harmonic.csv",
    "wang2006_published_reference_curves.csv",
    "wang2006_anharmonic_checkpoints.csv",
    "mingo2006_figure4.csv",
    "mingo2006_figure4_smooth_calculations.csv",
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("historical_renderer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(tmp_path: Path) -> tuple[Path, Path, str]:
    data = tmp_path / "data"
    data.mkdir()
    for name in SELECTED:
        shutil.copyfile(SOURCE_DATA / name, data / name)
    manifest = data / "benchmark_manifest.json"
    shutil.copyfile(SOURCE_MANIFEST, manifest)
    return data, manifest, _sha(manifest)


def _rewrite_manifest_hash(manifest: Path, name: str, path: Path) -> str:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["artifact_sha256"][f"data/{name}"] = _sha(path)
    manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return _sha(manifest)


def _arguments(data: Path, manifest: Path, anchor: str, output: Path) -> dict[str, Any]:
    return {
        "data_dir": data,
        "benchmark_manifest": manifest,
        "expected_benchmark_manifest_sha256": anchor,
        "regular_font_path": REGULAR_FONT,
        "bold_font_path": BOLD_FONT,
        "output_dir": output,
    }


def test_validated_metrics_are_recomputed_exactly_from_csv(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, anchor = _stage(tmp_path)
    validated, hashes, source = module.load_historical_inputs(
        data_dir=data,
        benchmark_manifest=manifest,
        expected_benchmark_manifest_sha256=anchor,
    )
    with (data / "wang2006_harmonic.csv").open(newline="", encoding="utf-8") as f:
        harmonic = list(csv.DictReader(f))
    errors = [
        float(row["calculated_minus_published_nW_per_K"]) for row in harmonic
    ]
    expected_rms = math.sqrt(math.fsum(value * value for value in errors) / len(errors))
    observed = validated["metrics"]["wang_harmonic_error_nW_per_K"]
    assert observed["count"] == 51
    assert observed["rms"] == expected_rms
    assert observed["max_abs"] == max(abs(value) for value in errors)
    assert set(hashes) == {f"data/{name}" for name in SELECTED}
    assert source["benchmark_manifest_raw_sha256"] == anchor


def test_wrong_caller_manifest_anchor_is_refused(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, _ = _stage(tmp_path)
    with pytest.raises(module.HistoricalRenderError, match="caller-pinned"):
        module.load_historical_inputs(
            data_dir=data,
            benchmark_manifest=manifest,
            expected_benchmark_manifest_sha256="0" * 64,
        )


def test_tampered_csv_is_refused_by_manifest_hash(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, anchor = _stage(tmp_path)
    target = data / "wang2006_harmonic.csv"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(module.HistoricalRenderError, match="hash mismatch"):
        module.load_historical_inputs(
            data_dir=data,
            benchmark_manifest=manifest,
            expected_benchmark_manifest_sha256=anchor,
        )


def test_exact_csv_schema_is_independent_of_hash_anchor(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, _ = _stage(tmp_path)
    target = data / "wang2006_harmonic.csv"
    lines = target.read_text(encoding="utf-8").splitlines()
    columns = lines[0].split(",")
    columns[0], columns[1] = columns[1], columns[0]
    lines[0] = ",".join(columns)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    anchor = _rewrite_manifest_hash(manifest, target.name, target)
    with pytest.raises(module.HistoricalRenderError, match="schema mismatch"):
        module.load_historical_inputs(
            data_dir=data,
            benchmark_manifest=manifest,
            expected_benchmark_manifest_sha256=anchor,
        )


def test_inconsistent_derived_error_is_refused_after_rehash(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, _ = _stage(tmp_path)
    target = data / "mingo2006_figure4.csv"
    rows = list(csv.reader(target.read_text(encoding="utf-8").splitlines()))
    rows[3][-2] = str(float(rows[3][-2]) + 0.1)
    target.write_text(
        "\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8"
    )
    anchor = _rewrite_manifest_hash(manifest, target.name, target)
    with pytest.raises(module.HistoricalRenderError, match="residual"):
        module.load_historical_inputs(
            data_dir=data,
            benchmark_manifest=manifest,
            expected_benchmark_manifest_sha256=anchor,
        )


def test_symlink_and_special_input_files_are_refused(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, anchor = _stage(tmp_path)
    target = data / "wang2006_harmonic.csv"
    outside = tmp_path / "outside.csv"
    target.rename(outside)
    target.symlink_to(outside)
    with pytest.raises(module.HistoricalRenderError, match="symlink"):
        module.load_historical_inputs(
            data_dir=data,
            benchmark_manifest=manifest,
            expected_benchmark_manifest_sha256=anchor,
        )

    target.unlink()
    os.mkfifo(target)
    with pytest.raises(module.HistoricalRenderError, match="regular non-symlink"):
        module.load_historical_inputs(
            data_dir=data,
            benchmark_manifest=manifest,
            expected_benchmark_manifest_sha256=anchor,
        )


def test_manifest_must_be_directly_inside_data_directory(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, anchor = _stage(tmp_path)
    nested = data / "nested"
    nested.mkdir()
    moved = nested / manifest.name
    manifest.rename(moved)
    with pytest.raises(module.HistoricalRenderError, match="directly inside"):
        module.load_historical_inputs(
            data_dir=data,
            benchmark_manifest=moved,
            expected_benchmark_manifest_sha256=anchor,
        )


@pytest.mark.skipif(
    not all(
        path.exists()
        for path in (
            REGULAR_FONT,
            BOLD_FONT,
            Path("/usr/bin/pdfinfo"),
            Path("/usr/bin/pdffonts"),
            Path("/usr/bin/pdfimages"),
            Path("/usr/bin/pdftoppm"),
        )
    ),
    reason="rendering dependencies are unavailable",
)
def test_full_render_is_noncertifying_vector_rgb_and_hash_covered(
    tmp_path: Path,
) -> None:
    module = _load_module()
    data, source_manifest, anchor = _stage(tmp_path)
    output = tmp_path / "rendered"
    manifest = module.render_historical_benchmarks(
        **_arguments(data, source_manifest, anchor, output)
    )
    assert manifest["status"] == "NONCERTIFYING_HISTORICAL_SENSITIVITY"
    assert manifest["release_certified"] is False
    assert manifest["production_evidence"] is False
    assert manifest["renderer"]["version"] == module._RENDERER_VERSION
    metrics_path = output / "historical_paper_benchmarks_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="ascii"))
    assert metrics["schema"] == module._METRICS_SCHEMA
    assert metrics["metrics_recomputed_from_csv"] == manifest["metrics_recomputed_from_csv"]
    assert manifest["metrics_manifest"]["sha256"] == _sha(metrics_path)
    metrics_core = {
        key: value
        for key, value in metrics.items()
        if key != "manifest_sha256"
    }
    assert metrics["manifest_sha256"] == hashlib.sha256(
        json.dumps(
            metrics_core,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    caption = manifest["caption_metadata"]
    assert caption["published_data_provenance"] == (
        "figure-extracted paths/points; not author raw data"
    )
    assert caption["wang_convention_scope"] == (
        "conditional_inferred_onsite_convention"
    )
    assert caption["mingo_fig8_interacting_curve"] == "not_computed_not_shown"
    assert set(entry.name for entry in output.iterdir()) == {
        "historical_paper_benchmarks.pdf",
        "historical_paper_benchmarks.png",
        "historical_paper_benchmarks_render_manifest.json",
        "historical_paper_benchmarks_metrics.json",
        "SHA256SUMS",
    }
    with Image.open(output / "historical_paper_benchmarks.png") as image:
        image.load()
        assert image.mode == "RGB"
        assert image.size == (3150, 2250)
        assert image.info["dpi"] == pytest.approx((450, 450), abs=0.1)
    pdfimages = subprocess.run(
        ["/usr/bin/pdfimages", "-list", output / "historical_paper_benchmarks.pdf"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert not any(
        line.lstrip()[:1].isdigit() for line in pdfimages.splitlines()
    )
    pdffonts = subprocess.run(
        ["/usr/bin/pdffonts", output / "historical_paper_benchmarks.pdf"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "TrueType" in pdffonts
    assert "Type 3" not in pdffonts
    assert " no " not in "\n".join(pdffonts.splitlines()[2:])
    for line in (output / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        assert digest == _sha(output / name)


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    module = _load_module()
    data, manifest, anchor = _stage(tmp_path)
    output = tmp_path / "rendered"
    output.mkdir()
    sentinel = output / "user-file"
    sentinel.write_text("preserve me", encoding="utf-8")
    with pytest.raises(module.HistoricalRenderError, match="already exists"):
        module.render_historical_benchmarks(
            **_arguments(data, manifest, anchor, output)
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve me"


def test_atomic_publish_race_preserves_competing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    source = tmp_path / "temporary"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "competitor"
    sentinel.write_text("winner", encoding="utf-8")
    with pytest.raises(module.HistoricalRenderError, match="already exists"):
        module._rename_noreplace(source, destination)
    assert source.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "winner"


def test_cli_requires_all_trust_and_font_anchors() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--expected-benchmark-manifest-sha256" in completed.stderr
    assert "--regular-font-path" in completed.stderr
    assert "--bold-font-path" in completed.stderr
