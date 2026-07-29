#!/usr/bin/env python3
"""Certify a merged target-q silicon campaign against phono3py."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from negfpy.anharmonic.silicon_mpi_oracle_validation import (
    SILICON_MPI_ORACLE_REPORT_SCHEMA,
    SiliconMPIOracleValidationConfig,
    run_silicon_mpi_oracle_validation,
)
from negfpy.anharmonic.silicon_oracle_validation import (
    write_json_atomic_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate a complete target-q MPI campaign and compare its "
            "merged fields directly with the independent phono3py reference. "
            "No monolithic NEGFPy linewidth calculation is performed."
        )
    )
    parser.add_argument("--qualification-report", required=True, type=Path)
    parser.add_argument("--campaign-summary", required=True, type=Path)
    parser.add_argument("--merge-report", required=True, type=Path)
    parser.add_argument("--merged-result", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fc2-compact", required=True, type=Path)
    parser.add_argument("--fc2-full", required=True, type=Path)
    parser.add_argument("--fc3-compact", required=True, type=Path)
    parser.add_argument("--fc3-full", required=True, type=Path)
    parser.add_argument("--phono3py-yaml", required=True, type=Path)
    parser.add_argument(
        "--input-manifest",
        required=True,
        type=Path,
        help="Generated input-manifest.normalized.json from the reference bundle.",
    )
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--interaction-samples", required=True, type=Path)
    parser.add_argument(
        "--interaction-samples-orchestration",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--interaction-samples-run-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Permit a development diagnostic from a dirty tree. The output "
            "cannot become certification eligible."
        ),
    )
    return parser


def _requested_inputs(arguments: argparse.Namespace) -> dict[str, str]:
    return {
        name: str(getattr(arguments, name))
        for name in (
            "qualification_report",
            "campaign_summary",
            "merge_report",
            "merged_result",
            "reference",
            "fc2_compact",
            "fc2_full",
            "fc3_compact",
            "fc3_full",
            "phono3py_yaml",
            "input_manifest",
            "checksums",
            "environment_lock",
            "interaction_samples",
            "interaction_samples_orchestration",
            "interaction_samples_run_directory",
            "repository",
        )
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = SiliconMPIOracleValidationConfig(
            qualification_report_path=arguments.qualification_report,
            campaign_summary_path=arguments.campaign_summary,
            merge_report_path=arguments.merge_report,
            merged_result_path=arguments.merged_result,
            reference_path=arguments.reference,
            fc2_compact_path=arguments.fc2_compact,
            fc2_full_path=arguments.fc2_full,
            fc3_compact_path=arguments.fc3_compact,
            fc3_full_path=arguments.fc3_full,
            phono3py_yaml_path=arguments.phono3py_yaml,
            input_manifest_path=arguments.input_manifest,
            checksums_path=arguments.checksums,
            environment_lock_path=arguments.environment_lock,
            interaction_samples_path=arguments.interaction_samples,
            interaction_samples_orchestration_path=(
                arguments.interaction_samples_orchestration
            ),
            interaction_samples_run_directory_path=(
                arguments.interaction_samples_run_directory
            ),
            repository_path=arguments.repository,
            output_path=arguments.output,
            require_clean_source=not arguments.allow_dirty_source,
        )
        report = run_silicon_mpi_oracle_validation(config)
    except Exception as exc:
        error_report = {
            "schema": SILICON_MPI_ORACLE_REPORT_SCHEMA,
            "schema_version": 1,
            "status": "error",
            "validation_passed": False,
            "certification_eligible": False,
            "release_certified": False,
            "error": {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            },
            "requested_inputs": _requested_inputs(arguments),
            "execution": {
                "monolithic_negfpy_recomputation": False,
            },
        }
        try:
            digest = write_json_atomic_exclusive(
                arguments.output, error_report
            )
        except Exception as publication_error:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(exc),
                        "report_publication_error": (
                            str(publication_error)
                        ),
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
                "certification_eligible": (
                    report["certification_eligible"]
                ),
                "release_certified": report["release_certified"],
                "output": artifact["path"],
                "sha256": artifact["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
