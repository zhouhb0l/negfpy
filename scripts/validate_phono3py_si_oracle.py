#!/usr/bin/env python3
"""Run one bounded, serial NEGFPy/phono3py silicon-oracle comparison."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from negfpy.anharmonic.silicon_oracle_validation import (
    SILICON_ORACLE_REPORT_SCHEMA,
    SiliconOracleValidationConfig,
    run_silicon_oracle_validation,
    write_json_atomic_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one complete bounded set of silicon target q-points "
            "against the pinned phono3py 4.4.0 reference."
        )
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fc2-compact", required=True, type=Path)
    parser.add_argument("--fc2-full", required=True, type=Path)
    parser.add_argument("--fc3-compact", required=True, type=Path)
    parser.add_argument("--fc3-full", required=True, type=Path)
    parser.add_argument("--phono3py-yaml", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--checksums", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument(
        "--interaction-samples",
        required=True,
        type=Path,
        help=(
            "Canonical checksummed V3 sample manifest tied to the exact "
            "reference, IFCs, mappings, and mesh maps."
        ),
    )
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--asr-atol", type=float, default=1.0e-10)
    parser.add_argument("--asr-rtol", type=float, default=1.0e-8)
    parser.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Permit a development run from a dirty tree. The report will be "
            "marked ineligible for certification."
        ),
    )
    return parser


def _error_report(arguments: argparse.Namespace, exc: Exception) -> dict:
    return {
        "schema": SILICON_ORACLE_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "error",
        "validation_passed": False,
        "certification_eligible": False,
        "error": {
            "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
        },
        "requested_inputs": {
            "reference": str(arguments.reference),
            "fc2_compact": str(arguments.fc2_compact),
            "fc2_full": str(arguments.fc2_full),
            "fc3_compact": str(arguments.fc3_compact),
            "fc3_full": str(arguments.fc3_full),
            "phono3py_yaml": str(arguments.phono3py_yaml),
            "input_manifest": str(arguments.input_manifest),
            "checksums": str(arguments.checksums),
            "environment_lock": str(arguments.environment_lock),
            "interaction_samples": str(arguments.interaction_samples),
            "repository": str(arguments.repository),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = SiliconOracleValidationConfig(
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
            repository_path=arguments.repository,
            output_path=arguments.output,
            require_clean_source=not arguments.allow_dirty_source,
            asr_atol=arguments.asr_atol,
            asr_rtol=arguments.asr_rtol,
        )
        report = run_silicon_oracle_validation(config)
    except Exception as exc:
        error_report = _error_report(arguments, exc)
        try:
            digest = write_json_atomic_exclusive(
                arguments.output,
                error_report,
            )
        except Exception as publication_error:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error": str(exc),
                        "report_publication_error": str(publication_error),
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
                "certification_eligible": report["certification_eligible"],
                "output": artifact["path"],
                "sha256": artifact["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
