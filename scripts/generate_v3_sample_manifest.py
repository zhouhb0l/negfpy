#!/usr/bin/env python3
"""Create the immutable reference-selected silicon ``|V3|^2`` manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from negfpy.anharmonic.io.phono3py_crystal import (
    read_phono3py_crystal_mapping,
)
from negfpy.anharmonic.phono3py_reference import (
    read_phono3py_silicon_reference,
)
from negfpy.anharmonic.v3_sample_manifest import (
    LARGE_MESH_BOUNDED_SELECTION,
    LEGACY_FULL_COVERAGE_SELECTION,
    create_v3_sample_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select at least 64 deterministic gauge-invariant nonzero V3 "
            "blocks from a verified phono3py silicon reference."
        )
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fc2-compact", required=True, type=Path)
    parser.add_argument("--fc2-full", required=True, type=Path)
    parser.add_argument("--fc3-compact", required=True, type=Path)
    parser.add_argument("--fc3-full", required=True, type=Path)
    parser.add_argument("--phono3py-yaml", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--selection-mode",
        choices=(
            LEGACY_FULL_COVERAGE_SELECTION,
            LARGE_MESH_BOUNDED_SELECTION,
        ),
        default=LEGACY_FULL_COVERAGE_SELECTION,
        help=(
            "Pinned sampling contract. The default preserves archived v1; "
            "large meshes must explicitly request bounded streaming v2."
        ),
    )
    parser.add_argument(
        "--maximum-promoted-blocks",
        type=int,
        default=512,
        help="Hard v2 output cap (64--512; ignored by legacy selection).",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    reference = read_phono3py_silicon_reference(arguments.reference)
    fc2_mapping = read_phono3py_crystal_mapping(
        arguments.phono3py_yaml,
        purpose="fc2",
    )
    fc3_mapping = read_phono3py_crystal_mapping(
        arguments.phono3py_yaml,
        purpose="fc3",
    )
    manifest = create_v3_sample_manifest(
        reference,
        output_path=arguments.output,
        fc2_compact_hdf5_path=arguments.fc2_compact,
        fc2_full_hdf5_path=arguments.fc2_full,
        fc3_compact_hdf5_path=arguments.fc3_compact,
        fc3_full_hdf5_path=arguments.fc3_full,
        phono3py_yaml_path=arguments.phono3py_yaml,
        input_manifest_path=arguments.input_manifest,
        fc2_mapping=fc2_mapping,
        fc3_mapping=fc3_mapping,
        selection_mode=arguments.selection_mode,
        maximum_promoted_blocks=arguments.maximum_promoted_blocks,
    )
    print(
        json.dumps(
            {
                "path": str(manifest.path),
                "payload_sha256": manifest.sha256,
                "promoted_block_count": len(manifest.samples),
                "selection_algorithm": manifest.selection_algorithm,
                "selection_algorithm_version": (
                    manifest.selection_algorithm_version
                ),
                "status": "complete",
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
