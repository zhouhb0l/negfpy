#!/usr/bin/env python3
"""Prepare or verify the authenticated sparse-Nq=1 Phase-A silicon fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from negfpy.anharmonic.phase_a_fixture import (
    prepare_phase_a_fixture,
    verify_phase_a_fixture,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate the sealed 0.2.0 silicon archive and atomically "
            "publish its deterministic Phase-A projection."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Sealed attempt-005 artifact root (required for preparation).",
    )
    parser.add_argument(
        "--yaml-object",
        type=Path,
        help=(
            "Content-addressed phono3py YAML object declared by the sealed "
            "external-alias manifest (required for preparation)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New fixture directory, or existing fixture with --verify-only.",
    )
    parser.add_argument(
        "--portable-manifest",
        type=Path,
        help=(
            "Authenticated portable-manifest.json; defaults to "
            "<artifact-root>/../handoff/portable-manifest.json."
        ),
    )
    parser.add_argument(
        "--handoff-sha256sums",
        type=Path,
        help=(
            "Authenticated handoff SHA256SUMS; defaults to "
            "<artifact-root>/../handoff/SHA256SUMS."
        ),
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify --output without consulting the source archive.",
    )
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.verify_only:
        if any(
            value is not None
            for value in (
                arguments.artifact_root,
                arguments.yaml_object,
                arguments.portable_manifest,
                arguments.handoff_sha256sums,
            )
        ):
            parser.error(
                "--verify-only accepts --output alone; source arguments are forbidden."
            )
        result = verify_phase_a_fixture(arguments.output)
    else:
        if arguments.artifact_root is None or arguments.yaml_object is None:
            parser.error(
                "preparation requires --artifact-root, --yaml-object, and --output."
            )
        result = prepare_phase_a_fixture(
            artifact_root=arguments.artifact_root,
            yaml_object=arguments.yaml_object,
            output=arguments.output,
            portable_manifest=arguments.portable_manifest,
            handoff_sha256sums=arguments.handoff_sha256sums,
        )
    print(
        json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
