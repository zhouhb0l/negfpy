#!/usr/bin/env python3
"""Seal or verify relocatable NEGFPy silicon release evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from negfpy.anharmonic._silicon_release_evidence import (
    finalize_silicon_release_evidence,
    seal_silicon_release_evidence,
    stage_external_silicon_release_evidence,
    verify_silicon_release_evidence,
    verify_silicon_release_handoff,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage external records, create, or verify a canonical relocatable "
            "archive for a completed NEGFPy 0.2.0 silicon artifact tree."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    stage = commands.add_parser(
        "stage",
        help=(
            "Explicitly copy external path+SHA256 evidence into a "
            "content-addressed subtree."
        ),
    )
    stage.add_argument("--artifact-root", required=True, type=Path)

    seal = commands.add_parser(
        "seal",
        help="Seal a completed artifact tree into an immutable manifest.",
    )
    seal.add_argument("--artifact-root", required=True, type=Path)
    seal.add_argument("--release-report", required=True, type=Path)
    seal.add_argument("--output", required=True, type=Path)
    seal.add_argument(
        "--external-alias-manifest",
        type=Path,
        help=(
            "Immutable alias manifest emitted by the explicit stage action."
        ),
    )

    finalize = commands.add_parser(
        "finalize",
        help=(
            "Stage, seal, self-verify, and atomically publish a trusted "
            "SHA256SUMS-style archival handoff."
        ),
    )
    finalize.add_argument("--artifact-root", required=True, type=Path)
    finalize.add_argument("--release-report", required=True, type=Path)
    finalize.add_argument(
        "--handoff-directory",
        required=True,
        type=Path,
    )

    verify = commands.add_parser(
        "verify",
        help="Verify the same evidence bytes at an original or relocated root.",
    )
    verify.add_argument("--artifact-root", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument(
        "--expected-manifest-sha256",
        help="Trusted remote manifest digest from the transfer checksum.",
    )
    verify.add_argument(
        "--expected-release-report-sha256",
        help="Trusted final-report digest from the transfer checksum.",
    )

    verify_handoff = commands.add_parser(
        "verify-handoff",
        help=(
            "Verify a relocated artifact tree using an authenticated "
            "SHA256SUMS handoff."
        ),
    )
    verify_handoff.add_argument(
        "--artifact-root",
        required=True,
        type=Path,
    )
    verify_handoff.add_argument(
        "--trusted-sha256sums",
        required=True,
        type=Path,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "stage":
            evidence = stage_external_silicon_release_evidence(
                arguments.artifact_root,
            )
            action = "staged"
        elif arguments.command == "seal":
            evidence = seal_silicon_release_evidence(
                arguments.artifact_root,
                arguments.release_report,
                arguments.output,
                external_alias_manifest_path=(
                    arguments.external_alias_manifest
                ),
            )
            action = "sealed"
        elif arguments.command == "finalize":
            evidence = finalize_silicon_release_evidence(
                arguments.artifact_root,
                arguments.release_report,
                arguments.handoff_directory,
            )
            action = "finalized"
        elif arguments.command == "verify":
            evidence = verify_silicon_release_evidence(
                arguments.artifact_root,
                arguments.manifest,
                expected_manifest_sha256=(
                    arguments.expected_manifest_sha256
                ),
                expected_release_report_sha256=(
                    arguments.expected_release_report_sha256
                ),
            )
            action = "verified"
        else:
            evidence = verify_silicon_release_handoff(
                arguments.artifact_root,
                arguments.trusted_sha256sums,
            )
            action = "retrieval_verified"
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "action": arguments.command,
                    "error": {
                        "type": (
                            f"{type(exc).__module__}." f"{type(exc).__qualname__}"
                        ),
                        "message": str(exc),
                    },
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "status": "passed",
                "action": action,
                **evidence.as_dict(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
