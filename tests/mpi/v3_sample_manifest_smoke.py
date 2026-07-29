"""Real-MPI smoke for bounded-v2 V3 sampling and rank-count restart."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from mpi4py import MPI


REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))
sys.path.insert(0, str(REPOSITORY / "tests"))

from negfpy.anharmonic._v3_sample_manifest_mpi import (  # noqa: E402
    V3SampleMPICampaign,
    current_v3_sampling_source_identity,
    run_v3_sample_manifest_mpi,
    validate_v3_mpi_orchestration_evidence,
)
from negfpy.anharmonic.phono3py_reference import (  # noqa: E402
    read_phono3py_silicon_reference,
)
from negfpy.anharmonic.v3_sample_manifest import (  # noqa: E402
    LARGE_MESH_BOUNDED_SELECTION,
    create_v3_sample_manifest,
)
from test_v3_sample_manifest import (  # noqa: E402
    _inputs,
    _mapping,
    _write_reference,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--phase", choices=("initial", "restart"), required=True
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    comm = MPI.COMM_WORLD
    rank = int(comm.Get_rank())
    root = arguments.root.resolve()
    fixture = root / "fixture"
    if rank == 0 and arguments.phase == "initial":
        fixture.mkdir(parents=True, exist_ok=False)
        reference_path = _write_reference(fixture)
        inputs = _inputs(fixture)
        reference = read_phono3py_silicon_reference(reference_path)
        create_v3_sample_manifest(
            reference,
            output_path=root / "serial.json",
            selection_mode=LARGE_MESH_BOUNDED_SELECTION,
            maximum_promoted_blocks=96,
            **inputs,
        )
    comm.Barrier()

    reference = read_phono3py_silicon_reference(
        fixture / "reference.hdf5"
    )
    source = current_v3_sampling_source_identity(
        REPOSITORY,
        require_clean=False,
    )
    campaign = V3SampleMPICampaign(
        reference=reference,
        fc2_compact_hdf5_path=fixture / "fc2-compact.hdf5",
        fc2_full_hdf5_path=fixture / "fc2-full.hdf5",
        fc3_compact_hdf5_path=fixture / "fc3-compact.hdf5",
        fc3_full_hdf5_path=fixture / "fc3-full.hdf5",
        phono3py_yaml_path=fixture / "phono3py.yaml",
        input_manifest_path=fixture / "input-manifest.json",
        fc2_mapping=_mapping(purpose="fc2"),
        fc3_mapping=_mapping(purpose="fc3"),
        source_revision=source["head_commit"],
        source_tree_sha256=source["source_tree_sha256"],
        environment_sha256="b" * 64,
        source_identity=source,
        require_authenticated_source=False,
        maximum_promoted_blocks=96,
    )
    result = run_v3_sample_manifest_mpi(
        campaign,
        run_directory=root / "run",
        output_path=root / "mpi.json",
        comm=comm,
        require_mpi=True,
    )
    if rank == 0:
        assert result is not None
        assert (root / "mpi.json").read_bytes() == (
            root / "serial.json"
        ).read_bytes()
        expected_resumed = arguments.phase == "restart"
        assert all(
            value.resumed is expected_resumed
            for value in result.phase_maximum_execution.assignments
        )
        assert all(
            value.resumed is expected_resumed
            for value in result.phase_reservoir_execution.assignments
        )
        orchestration = validate_v3_mpi_orchestration_evidence(
            root / "mpi.json",
            result.orchestration_path,
            root / "run",
            require_authenticated_source=False,
            require_mpi=True,
        )
        assert orchestration.selected_modes == ("mpi", "mpi")
        assert orchestration.restart_resume_recorded is expected_resumed
        print(
            f"v3-sample-mpi-{arguments.phase}-passed "
            f"ranks={comm.Get_size()} "
            f"sha256={result.manifest.sha256} "
            f"orchestration_sha256="
            f"{result.orchestration_file_sha256}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(variable) != "1":
            raise RuntimeError(f"{variable} must equal 1.")
    raise SystemExit(main())
