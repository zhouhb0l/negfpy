#!/usr/bin/env python3
"""Generate the bounded-v2 silicon V3 sample manifest with task-level MPI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from negfpy.anharmonic._v3_sample_manifest_mpi import (
    V3SampleMPICampaign,
    _reference_descriptor,
    _reference_from_descriptor,
    _resolve_nonsymlink_path,
    current_v3_sampling_source_identity,
    run_v3_sample_manifest_mpi,
)
from negfpy.anharmonic.io.phono3py_crystal import (
    read_phono3py_crystal_mapping,
)
from negfpy.anharmonic.phono3py_reference import (
    read_phono3py_silicon_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the bounded-v2 V3 sample manifest with one complete "
            "target-q reference slab per task and an explicitly recorded "
            "inter-phase global-maximum broadcast."
        )
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--fc2-compact", required=True, type=Path)
    parser.add_argument("--fc2-full", required=True, type=Path)
    parser.add_argument("--fc3-compact", required=True, type=Path)
    parser.add_argument("--fc3-full", required=True, type=Path)
    parser.add_argument("--phono3py-yaml", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--maximum-promoted-blocks",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--require-mpi",
        action="store_true",
        help="Fail unless mpi4py supplies the communicator.",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _communicator(require_mpi: bool):
    try:
        from mpi4py import MPI
    except ImportError as exc:
        if require_mpi:
            raise RuntimeError("--require-mpi needs mpi4py.") from exc
        return None
    return MPI.COMM_WORLD


def main() -> int:
    arguments = _parser().parse_args()
    communicator = None
    rank = 0
    try:
        communicator = _communicator(arguments.require_mpi)
        rank = (
            0
            if communicator is None
            else int(communicator.Get_rank())
        )
        for attribute in (
            "reference",
            "fc2_compact",
            "fc2_full",
            "fc3_compact",
            "fc3_full",
            "phono3py_yaml",
            "input_manifest",
            "environment_lock",
        ):
            setattr(
                arguments,
                attribute,
                _resolve_nonsymlink_path(
                    getattr(arguments, attribute),
                    name=attribute.replace("_", " "),
                    must_exist=True,
                    expected_kind="file",
                ),
            )
        arguments.repository = _resolve_nonsymlink_path(
            arguments.repository,
            name="repository",
            must_exist=True,
            expected_kind="directory",
        )
        arguments.run_directory = _resolve_nonsymlink_path(
            arguments.run_directory,
            name="run directory",
            must_exist=False,
            expected_kind="directory",
        )
        arguments.output = _resolve_nonsymlink_path(
            arguments.output,
            name="output",
            must_exist=False,
            expected_kind="file",
        )
        if communicator is not None and int(communicator.Get_size()) > 1:
            reference_packet = None
            if rank == 0:
                try:
                    loaded = read_phono3py_silicon_reference(
                        arguments.reference
                    )
                    reference_packet = {
                        "ok": True,
                        "descriptor": _reference_descriptor(loaded),
                    }
                except Exception as exc:
                    reference_packet = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            reference_packet = communicator.bcast(
                reference_packet, root=0
            )
            if not reference_packet["ok"]:
                raise RuntimeError(
                    "rank-zero reference load failed: "
                    f"{reference_packet['error_type']}: "
                    f"{reference_packet['error']}"
                )
            reference = _reference_from_descriptor(
                reference_packet["descriptor"]
            )
        else:
            reference = read_phono3py_silicon_reference(
                arguments.reference
            )
        if communicator is not None and int(communicator.Get_size()) > 1:
            source_packet = None
            if rank == 0:
                try:
                    source_packet = {
                        "ok": True,
                        "identity": current_v3_sampling_source_identity(
                            arguments.repository,
                            require_clean=True,
                        ),
                    }
                except Exception as exc:
                    source_packet = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            source_packet = communicator.bcast(source_packet, root=0)
            if not source_packet["ok"]:
                raise RuntimeError(
                    "rank-zero source authentication failed: "
                    f"{source_packet['error_type']}: "
                    f"{source_packet['error']}"
                )
            source_identity = source_packet["identity"]
        else:
            source_identity = current_v3_sampling_source_identity(
                arguments.repository,
                require_clean=True,
            )
        fc2_mapping = read_phono3py_crystal_mapping(
            arguments.phono3py_yaml,
            purpose="fc2",
        )
        fc3_mapping = read_phono3py_crystal_mapping(
            arguments.phono3py_yaml,
            purpose="fc3",
        )
        campaign = V3SampleMPICampaign(
            reference=reference,
            fc2_compact_hdf5_path=arguments.fc2_compact,
            fc2_full_hdf5_path=arguments.fc2_full,
            fc3_compact_hdf5_path=arguments.fc3_compact,
            fc3_full_hdf5_path=arguments.fc3_full,
            phono3py_yaml_path=arguments.phono3py_yaml,
            input_manifest_path=arguments.input_manifest,
            fc2_mapping=fc2_mapping,
            fc3_mapping=fc3_mapping,
            source_revision=source_identity["head_commit"],
            source_tree_sha256=source_identity[
                "source_tree_sha256"
            ],
            environment_sha256=_sha256(arguments.environment_lock),
            source_identity=source_identity,
            require_authenticated_source=True,
            maximum_promoted_blocks=arguments.maximum_promoted_blocks,
        )
        result = run_v3_sample_manifest_mpi(
            campaign,
            run_directory=arguments.run_directory,
            output_path=arguments.output,
            comm=communicator,
            require_mpi=arguments.require_mpi,
        )
    except Exception as exc:
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        return 2
    if rank == 0:
        assert result is not None
        print(
            json.dumps(
                {
                    "status": "complete",
                    "path": str(result.manifest.path),
                    "payload_sha256": result.manifest.sha256,
                    "promoted_block_count": len(result.manifest.samples),
                    "campaign_sha256": result.campaign_sha256,
                    "maximum_phase_execution_id": (
                        result.phase_maximum_execution.execution_id
                    ),
                    "reservoir_phase_execution_id": (
                        result.phase_reservoir_execution.execution_id
                    ),
                    "orchestration_path": str(
                        result.orchestration_path
                    ),
                    "orchestration_file_sha256": (
                        result.orchestration_file_sha256
                    ),
                    "orchestration_payload_sha256": (
                        result.orchestration_payload_sha256
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
