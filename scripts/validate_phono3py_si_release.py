#!/usr/bin/env python3
"""Aggregate strict silicon evidence into the NEGFPy 0.2.0 decision."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from negfpy.anharmonic.silicon_oracle_validation import (
    write_json_atomic_exclusive,
)
from negfpy.anharmonic.silicon_release_validation import (
    SILICON_RELEASE_REPORT_SCHEMA,
    SiliconReleaseValidationConfig,
    run_silicon_release_validation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate the passed 11x11x11 external silicon oracle and "
            "fresh full-selection 3x3x3/5x5x5 serial-MPI equality evidence, "
            "then emit the immutable NEGFPy 0.2.0 release decision."
        )
    )
    parser.add_argument("--external-oracle-report", required=True, type=Path)
    parser.add_argument("--equality-3x3-summary", required=True, type=Path)
    parser.add_argument("--equality-5x5-summary", required=True, type=Path)
    parser.add_argument("--v3-3x3-manifest", required=True, type=Path)
    parser.add_argument("--v3-3x3-orchestration", required=True, type=Path)
    parser.add_argument("--v3-3x3-run-directory", required=True, type=Path)
    parser.add_argument("--v3-5x5-manifest", required=True, type=Path)
    parser.add_argument("--v3-5x5-orchestration", required=True, type=Path)
    parser.add_argument("--v3-5x5-run-directory", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = SiliconReleaseValidationConfig(
            external_oracle_report_path=arguments.external_oracle_report,
            equality_3x3_summary_path=arguments.equality_3x3_summary,
            equality_5x5_summary_path=arguments.equality_5x5_summary,
            v3_3x3_manifest_path=arguments.v3_3x3_manifest,
            v3_3x3_orchestration_path=arguments.v3_3x3_orchestration,
            v3_3x3_run_directory_path=arguments.v3_3x3_run_directory,
            v3_5x5_manifest_path=arguments.v3_5x5_manifest,
            v3_5x5_orchestration_path=arguments.v3_5x5_orchestration,
            v3_5x5_run_directory_path=arguments.v3_5x5_run_directory,
            repository_path=arguments.repository,
            output_path=arguments.output,
        )
        report = run_silicon_release_validation(config)
    except Exception as exc:  # noqa: BLE001 - publish every validation failure.
        error_report = {
            "schema": SILICON_RELEASE_REPORT_SCHEMA,
            "schema_version": 1,
            "status": "error",
            "validation_passed": False,
            "certification_eligible": False,
            "release_certified": False,
            "error": {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            },
            "requested_inputs": {
                "external_oracle_report": str(
                    arguments.external_oracle_report
                ),
                "equality_3x3_summary": str(arguments.equality_3x3_summary),
                "equality_5x5_summary": str(arguments.equality_5x5_summary),
                "v3_3x3_manifest": str(arguments.v3_3x3_manifest),
                "v3_3x3_orchestration": str(arguments.v3_3x3_orchestration),
                "v3_3x3_run_directory": str(arguments.v3_3x3_run_directory),
                "v3_5x5_manifest": str(arguments.v3_5x5_manifest),
                "v3_5x5_orchestration": str(arguments.v3_5x5_orchestration),
                "v3_5x5_run_directory": str(arguments.v3_5x5_run_directory),
                "repository": str(arguments.repository),
            },
        }
        try:
            digest = write_json_atomic_exclusive(
                arguments.output, error_report
            )
        except Exception as publication_error:  # noqa: BLE001
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(exc),
                        "report_publication_error": (str(publication_error)),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        print(
            json.dumps(
                {
                    "status": "error",
                    "release_certified": False,
                    "output": str(Path(arguments.output).resolve()),
                    "sha256": digest,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2

    artifact = report["report_artifact"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "validation_passed": report["validation_passed"],
                "certification_eligible": (report["certification_eligible"]),
                "release_certified": report["release_certified"],
                "release_version": report["release_version"],
                "output": artifact["path"],
                "sha256": artifact["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
