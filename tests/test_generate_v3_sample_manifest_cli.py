from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from negfpy.anharmonic.v3_sample_manifest import (
    LARGE_MESH_BOUNDED_SELECTION,
    LEGACY_FULL_COVERAGE_SELECTION,
)


def _load_driver() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_v3_sample_manifest.py"
    )
    spec = importlib.util.spec_from_file_location(
        "negfpy_test_generate_v3_sample_manifest",
        path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load V3 sample-manifest driver.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_arguments() -> list[str]:
    return [
        "--reference",
        "reference.hdf5",
        "--fc2-compact",
        "fc2-compact.hdf5",
        "--fc2-full",
        "fc2-full.hdf5",
        "--fc3-compact",
        "fc3-compact.hdf5",
        "--fc3-full",
        "fc3-full.hdf5",
        "--phono3py-yaml",
        "phono3py.yaml",
        "--input-manifest",
        "input-manifest.json",
        "--output",
        "samples.json",
    ]


def test_manifest_cli_preserves_v1_default_and_parses_explicit_v2() -> None:
    driver = _load_driver()

    default = driver._parser().parse_args(_required_arguments())
    assert default.selection_mode == LEGACY_FULL_COVERAGE_SELECTION
    assert default.maximum_promoted_blocks == 512

    bounded = driver._parser().parse_args(
        [
            *_required_arguments(),
            "--selection-mode",
            LARGE_MESH_BOUNDED_SELECTION,
            "--maximum-promoted-blocks",
            "256",
        ]
    )
    assert bounded.selection_mode == LARGE_MESH_BOUNDED_SELECTION
    assert bounded.maximum_promoted_blocks == 256


def test_manifest_cli_forwards_explicit_bounded_contract(
    monkeypatch,
    capsys,
) -> None:
    driver = _load_driver()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        driver,
        "read_phono3py_silicon_reference",
        lambda path: ("reference", path),
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_crystal_mapping",
        lambda path, *, purpose: (purpose, path),
    )

    def fake_create(reference, **kwargs):
        captured["reference"] = reference
        captured.update(kwargs)
        return SimpleNamespace(
            path=Path("samples.json"),
            sha256="a" * 64,
            samples=tuple(range(256)),
            selection_algorithm=(
                "deterministic_bounded_streaming_reference_invariant_blocks"
            ),
            selection_algorithm_version=2,
        )

    monkeypatch.setattr(driver, "create_v3_sample_manifest", fake_create)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_v3_sample_manifest.py",
            *_required_arguments(),
            "--selection-mode",
            LARGE_MESH_BOUNDED_SELECTION,
            "--maximum-promoted-blocks",
            "256",
        ],
    )

    assert driver.main() == 0
    assert captured["selection_mode"] == LARGE_MESH_BOUNDED_SELECTION
    assert captured["maximum_promoted_blocks"] == 256
    output = json.loads(capsys.readouterr().out)
    assert output["promoted_block_count"] == 256
    assert output["selection_algorithm_version"] == 2
