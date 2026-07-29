"""Two-rank full-field smoke and exact-restart test for target-q MPI.

Run from the optional MPI environment:

    mpiexec -n 2 python tests/mpi/bulk_target_q_smoke.py RUN_DIRECTORY
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mpi4py import MPI


# Reuse the exact bounded scientific fixture and equality assertion exercised
# by the serial contract tests rather than maintaining a second toy model.
_TEST_ROOT = Path(__file__).resolve().parents[1]
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from test_anharmonic_bulk_target_q_mpi import (  # noqa: E402
    _assert_all_scientific_fields_equal,
    _campaign,
    _problem,
)

from negfpy.anharmonic._bulk_target_q_mpi import (  # noqa: E402
    compare_bulk_target_q_with_serial,
    execute_bulk_target_q_campaign,
    load_bulk_target_q_result,
    make_bulk_target_q_artifact_validator,
    make_bulk_target_q_worker,
    merge_bulk_target_q_campaign,
)
from negfpy.anharmonic._mpi_tasks import (  # noqa: E402
    TaskExecutionError,
    execute_task_manifest,
    serial_task_communicator,
)
from negfpy.anharmonic.bulk_three_phonon import (  # noqa: E402
    compute_three_phonon_linewidths,
)


def main() -> int:
    rank = MPI.COMM_WORLD.Get_rank()
    if len(sys.argv) not in (2, 3):
        if rank == 0:
            print(
                "usage: bulk_target_q_smoke.py RUN_DIRECTORY "
                "[initial|restart]",
                file=sys.stderr,
            )
        return 2

    run_directory = Path(sys.argv[1]).expanduser().resolve()
    phase = "combined" if len(sys.argv) == 2 else sys.argv[2]
    if phase not in {"combined", "initial", "restart", "orphan"}:
        if rank == 0:
            print(f"unsupported phase: {phase}", file=sys.stderr)
        return 2
    campaign = _campaign()
    all_fields_equal = False

    if phase == "orphan":
        if rank == 0:
            worker = make_bulk_target_q_worker(campaign, _problem)

            def publish_then_crash(task, context):
                result = worker(task, context)
                if task == campaign.manifest.tasks[0]:
                    raise RuntimeError(
                        "simulated crash after immutable artifact publish"
                    )
                return result

            try:
                execute_task_manifest(
                    campaign.manifest,
                    publish_then_crash,
                    run_directory,
                    comm=serial_task_communicator(),
                    thread_limit=1,
                    artifact_validator=(
                        make_bulk_target_q_artifact_validator(campaign)
                    ),
                    execution_id="target-q-orphan-crash",
                )
            except TaskExecutionError:
                pass
            else:
                raise AssertionError("orphan setup did not interrupt")
        MPI.COMM_WORLD.Barrier()
        adopted = execute_bulk_target_q_campaign(
            campaign,
            _problem,
            run_directory,
            comm=MPI.COMM_WORLD,
            require_mpi=True,
            thread_limit=1,
            execution_id="target-q-orphan-adoption",
        )
        MPI.COMM_WORLD.Barrier()
        if rank == 0:
            if adopted is None or not adopted.complete:
                raise AssertionError("orphan recovery did not complete")
            adopted_ids = [
                item.task_id
                for item in adopted.assignments
                if item.orphan_artifact_adopted
            ]
            if adopted_ids != [campaign.manifest.tasks[0].task_id]:
                raise AssertionError(
                    "orphan adoption was not explicitly identified"
                )
            if not adopted.assignments[0].resumed:
                raise AssertionError(
                    "adopted preexisting artifact was marked fresh"
                )
            print(
                json.dumps(
                    {
                        "adopted_task_ids": adopted_ids,
                        "adopted_marked_resumed": True,
                        "communicator_size": MPI.COMM_WORLD.Get_size(),
                    },
                    sort_keys=True,
                )
            )
        return 0

    if phase in {"combined", "initial"}:
        first = execute_bulk_target_q_campaign(
            campaign,
            _problem,
            run_directory,
            comm=MPI.COMM_WORLD,
            require_mpi=True,
            thread_limit=1,
            execution_id="target-q-initial",
        )
        MPI.COMM_WORLD.Barrier()

        if rank == 0:
            if first is None or not first.complete:
                raise AssertionError(
                    "rank zero did not receive a complete run"
                )
            if first.mode != "mpi":
                raise AssertionError("target-q smoke did not use MPI")
            if any(item.rank == 0 for item in first.assignments):
                raise AssertionError(
                    "rank zero executed a numerical target-q task"
                )

            merged_report = merge_bulk_target_q_campaign(
                campaign,
                run_directory,
            )
            merged = load_bulk_target_q_result(
                merged_report.merged_artifact_path,
                campaign=campaign,
            )
            serial = compute_three_phonon_linewidths(
                _problem(),
                temperatures=campaign.controls.temperatures,
                grid_points=campaign.controls.grid_points,
                band_indices=campaign.controls.band_indices,
            )
            _assert_all_scientific_fields_equal(merged, serial)
            comparison = compare_bulk_target_q_with_serial(serial, merged)
            if not comparison.passed:
                raise AssertionError(
                    "merged target-q fields differ from serial"
                )
            all_fields_equal = True
        MPI.COMM_WORLD.Barrier()
        if phase == "initial":
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "all_fields_equal": all_fields_equal,
                            "communicator_size": MPI.COMM_WORLD.Get_size(),
                            "manifest_sha256": (
                                campaign.manifest.manifest_sha256
                            ),
                        },
                        sort_keys=True,
                    )
                )
            return 0

    resumed = execute_bulk_target_q_campaign(
        campaign,
        _problem,
        run_directory,
        comm=MPI.COMM_WORLD,
        require_mpi=True,
        thread_limit=1,
        execution_id="target-q-restart",
    )
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        if resumed is None or not resumed.complete:
            raise AssertionError("restart did not produce a complete report")
        all_resumed = all(item.resumed for item in resumed.assignments)
        if not all_resumed:
            raise AssertionError(
                "restart recomputed a terminal target-q task"
            )
        print(
            json.dumps(
                {
                    "all_fields_equal": all_fields_equal,
                    "all_resumed": all_resumed,
                    "communicator_size": MPI.COMM_WORLD.Get_size(),
                    "manifest_sha256": campaign.manifest.manifest_sha256,
                    "task_count": len(resumed.assignments),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
