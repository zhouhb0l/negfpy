"""Manual two-rank 1x1x1 smoke with the pinned phono3py runtime.

Usage:

    mpiexec -n 2 python SCRIPT FORCE_STAGE PHONO3PY_YAML RUN_ROOT [MESH]

This bypasses release preflight only so a dirty development worktree can test
the actual phono3py per-q numerical path.  Its output is smoke evidence, never
release-certifying generation provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mpi4py import MPI


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts import generate_phono3py_si_reference as serial  # noqa: E402
from scripts import generate_phono3py_si_reference_mpi as driver  # noqa: E402


def main() -> int:
    if len(sys.argv) not in (4, 5):
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(
                "usage: SCRIPT FORCE_STAGE PHONO3PY_YAML RUN_ROOT [MESH]",
                file=sys.stderr,
            )
        return 2
    force_stage = Path(sys.argv[1]).resolve()
    yaml_path = Path(sys.argv[2]).resolve()
    root = Path(sys.argv[3]).resolve()
    mesh_axis = 1 if len(sys.argv) == 4 else int(sys.argv[4])
    if mesh_axis not in (1, 3):
        raise ValueError("manual real smoke supports only 1 or 3")
    config = serial.GeneratorConfig(
        mesh=(mesh_axis, mesh_axis, mesh_axis),
        grid_points=(0,) if mesh_axis == 1 else (0, 1),
        temperatures=(300.0,),
        sigma=0.1,
        backend="Rust",
    )
    identity = driver.CampaignIdentity.from_payload(
        {
            "schema_version": 1,
            "purpose": "noncertifying_real_phono3py_smoke",
            "serial_generator": driver._orchestration_code_identity(),
            "mpi_driver": driver._orchestration_code_identity(),
            "source_manifest": {
                "versions": {"phono3py": "4.4.0", "phonopy": "4.4.0"}
            },
        }
    )
    run_dir = root / "run"
    if MPI.COMM_WORLD.Get_rank() == 0:
        driver._initialize_run_directory(run_dir, identity=identity)
    MPI.COMM_WORLD.Barrier()
    campaign = driver.PreparedCampaign(
        identity=identity,
        config=config,
        source_files={"phono3py_yaml": yaml_path},
        source_verification={},
        environment_identity={},
        force_stage_dir=force_stage,
        run_dir=run_dir,
        output_dir=root / "output",
    )
    output = driver.execute_campaign(
        campaign,
        comm=MPI.COMM_WORLD,
        execution_id=f"real-phono3py-{mesh_axis}x{mesh_axis}x{mesh_axis}",
        require_mpi=True,
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert output is not None
        print(
            json.dumps(
                {
                    "status": "complete",
                    "reference_sha256": driver._sha256_file(
                        output / serial.REFERENCE_FILENAME
                    ),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
