"""Real-MPI scheduler/restart smoke for the phono3py reference wrapper.

The HDF5 payload is deliberately synthetic and bounded; the production
phono3py numerical runtime is exercised separately.  This test isolates the
MPI protocol, immutable artifacts, changed-worker-count restart, canonical
merge, and atomic standard-bundle publication.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mpi4py import MPI


_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from test_generate_phono3py_si_reference_mpi import (  # noqa: E402
    _ENVIRONMENT_IDENTITY,
    _SOURCE_VERIFICATION,
    _config,
    _force_stage_payload,
    _identity,
    _validation_controls,
    _write_one_q,
)
from scripts import generate_phono3py_si_reference as serial  # noqa: E402
from scripts import generate_phono3py_si_reference_mpi as driver  # noqa: E402
from negfpy.anharmonic._reference_mpi_provenance import (  # noqa: E402
    validate_reference_mpi_evidence,
)


_CAMPAIGN_IDENTITY = _identity()


class _FakePhono3pyRuntime:
    phono3py_version = "4.4.0"
    phonopy_version = "4.4.0"

    def verify_versions(self, expected):
        actual = {"phono3py": "4.4.0", "phonopy": "4.4.0"}
        if dict(expected) != actual:
            raise RuntimeError("unexpected pinned versions")
        return actual

    def generate_reference(self, *, output_dir, source_files, config):
        point = int(config.grid_points[0])
        _write_one_q(
            output_dir / serial.REFERENCE_FILENAME,
            identity=_CAMPAIGN_IDENTITY,
            config=config,
            point=point,
        )
        from test_generate_phono3py_si_reference_mpi import _runtime_metadata

        return _runtime_metadata(point, point, config)


def _prepare(root: Path, *, phase: str) -> driver.PreparedCampaign:
    run_dir = root / "run"
    force_dir = root / "force"
    output_dir = root / f"output-{phase}"
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        driver._initialize_run_directory(
            run_dir,
            identity=_CAMPAIGN_IDENTITY,
        )
        if not force_dir.exists():
            force_dir.mkdir()
            for name in serial._FORCE_STAGE_FILENAMES:
                (force_dir / name).write_bytes(_force_stage_payload(name))
    MPI.COMM_WORLD.Barrier()
    return driver.PreparedCampaign(
        identity=_CAMPAIGN_IDENTITY,
        config=_config(),
        source_files={},
        source_verification=_SOURCE_VERIFICATION,
        environment_identity=_ENVIRONMENT_IDENTITY,
        force_stage_dir=force_dir,
        run_dir=run_dir,
        output_dir=output_dir,
    )


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[2] not in {"initial", "restart"}:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("usage: SCRIPT RUN_DIR initial|restart", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    phase = sys.argv[2]
    campaign = _prepare(root, phase=phase)
    output = driver.execute_campaign(
        campaign,
        comm=MPI.COMM_WORLD,
        runtime_factory=_FakePhono3pyRuntime,
        execution_id=f"smoke-{phase}",
        require_mpi=True,
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert output is not None
        execution = json.loads(
            (
                campaign.run_dir
                / "executions"
                / f"smoke-{phase}.json"
            ).read_text()
        )
        resumed = all(
            item["status"] == "resumed"
            for item in execution["assignments"]
        )
        orchestration = json.loads(
            (
                output / driver.MPI_ORCHESTRATION_FILENAME
            ).read_text(encoding="utf-8")
        )
        validated = validate_reference_mpi_evidence(
            output,
            orchestration,
            **_validation_controls(),
        )
        print(
            json.dumps(
                {
                    "communicator_size": MPI.COMM_WORLD.Get_size(),
                    "complete": execution["complete"],
                    "all_resumed": resumed,
                    "output_exists": (
                        output / serial.REFERENCE_FILENAME
                    ).is_file(),
                    "physical_core_binding_available": any(
                        evidence["available"]
                        for evidence in validated[
                            "worker_bindings_by_execution"
                        ].values()
                    ),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
