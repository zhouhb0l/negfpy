"""Run a safe silicon bulk-phonon example with optional heavy SMRTA.

The default path reads and audits only FC2, then computes bands, a full-mesh
DOS, and group velocities.  Cubic FC3 contraction is never started unless
both ``--run-fc3`` and ``--allow-heavy`` are supplied explicitly.

Install the pinned material-reader environment and run from the repository
root::

    python -m pip install -e '.[material-oracle]'
    python examples/bulk_rta_silicon.py \
        --crystal /path/to/phono3py_disp.yaml \
        --fc2 /path/to/fc2.hdf5

The default 3x3x3 mesh is an API exercise, not a converged silicon result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from negfpy import bulk

# Fractional coordinates in the reciprocal basis of the pinned fcc-silicon
# primitive cell. Other primitive-cell conventions need their own path.
SILICON_FCC_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],       # Gamma
        [0.5, 0.0, 0.5],       # X
        [0.5, 0.25, 0.75],     # W
        [0.375, 0.375, 0.75],  # K
        [0.0, 0.0, 0.0],       # Gamma
        [0.5, 0.5, 0.5],       # L
        [0.625, 0.25, 0.625],  # U
        [0.5, 0.25, 0.75],     # W
        [0.5, 0.5, 0.5],       # L
        [0.375, 0.375, 0.75],  # K
    ],
    dtype=np.float64,
)
SILICON_FCC_LABELS = (
    "Gamma",
    "X",
    "W",
    "K",
    "Gamma",
    "L",
    "U",
    "W",
    "L",
    "K",
)
CUTOFF_FREQUENCY_THZ = 1.0e-4
DEGENERATE_TOLERANCE_THZ = 1.0e-4


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute audited silicon bands/DOS/group velocities and, only "
            "with two explicit opt-ins, three-phonon SMRTA conductivity."
        )
    )
    parser.add_argument("--crystal", required=True, type=Path)
    parser.add_argument("--fc2", required=True, type=Path)
    parser.add_argument(
        "--fc3",
        type=Path,
        help="FC3 HDF5 input; read only when the two heavy flags are set.",
    )
    parser.add_argument(
        "--mesh",
        nargs=3,
        type=int,
        default=(3, 3, 3),
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--sigma-thz", type=float, default=0.1)
    parser.add_argument("--dos-sigma-thz", type=float, default=0.1)
    parser.add_argument("--band-points", type=int, default=21)
    parser.add_argument(
        "--run-fc3",
        action="store_true",
        help="Request the cubic linewidth and SMRTA calculation.",
    )
    parser.add_argument(
        "--allow-heavy",
        action="store_true",
        help="Confirm that a potentially long FC3 calculation is intended.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("silicon-smrta-example.h5"),
        help="Schema-checked HDF5 destination used only by the heavy path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the requested HDF5 result.",
    )
    return parser


def _validate_controls(args: argparse.Namespace) -> tuple[int, int, int]:
    mesh = tuple(int(value) for value in args.mesh)
    if any(value <= 0 for value in mesh):
        raise SystemExit("Every mesh dimension must be positive.")
    n_qpoints = int(np.prod(mesh))
    if n_qpoints > 125 and not args.allow_heavy:
        raise SystemExit(
            f"The requested mesh has {n_qpoints} q points. This example "
            "requires --allow-heavy above 125 points."
        )
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        raise SystemExit("--temperature must be finite and nonnegative.")
    if not np.isfinite(args.sigma_thz) or args.sigma_thz <= 0.0:
        raise SystemExit("--sigma-thz must be finite and positive.")
    if not np.isfinite(args.dos_sigma_thz) or args.dos_sigma_thz <= 0.0:
        raise SystemExit("--dos-sigma-thz must be finite and positive.")
    if args.band_points < 2:
        raise SystemExit("--band-points must be at least two.")
    if args.run_fc3 and not args.allow_heavy:
        raise SystemExit("--run-fc3 also requires explicit --allow-heavy.")
    if args.run_fc3 and args.fc3 is None:
        raise SystemExit("--run-fc3 requires an explicit --fc3 path.")
    return mesh


def main() -> None:
    args = _parser().parse_args()
    mesh = _validate_controls(args)

    # The normal/default execution path cannot read or contract FC3.
    harmonic = bulk.BulkHarmonicProblem.from_phono3py(
        args.crystal,
        args.fc2,
    )
    bands = bulk.compute_phonon_band_structure(
        harmonic.fc2,
        harmonic.mapping,
        SILICON_FCC_VERTICES,
        points_per_segment=args.band_points,
        labels=SILICON_FCC_LABELS,
        frequency_factor_to_thz=harmonic.frequency_factor_to_thz,
    )
    spectrum = harmonic.compute_uniform_mesh_phonons(mesh)
    dos = bulk.compute_phonon_dos(
        spectrum,
        sigma_thz=args.dos_sigma_thz,
    )
    velocities = bulk.compute_bulk_group_velocities(
        harmonic.fc2,
        harmonic.mapping,
        spectrum.qpoints,
        frequency_factor_to_thz=harmonic.frequency_factor_to_thz,
        cutoff_frequency_thz=CUTOFF_FREQUENCY_THZ,
        degenerate_tolerance_thz=DEGENERATE_TOLERANCE_THZ,
        symmetrize=True,
    )

    print(f"mesh:                 {mesh} ({spectrum.n_qpoints} q points)")
    print(f"FC2 physical audit:   {harmonic.fc2_physical_audit.passed}")
    print(f"band-path points:     {bands.spectrum.n_qpoints}")
    print(f"maximum frequency:   {np.max(spectrum.frequencies):.8f} THz")
    maximum_speed = np.max(
        np.linalg.norm(velocities.values_m_per_s, axis=2)
    )
    print(f"maximum speed:       {maximum_speed:.8f} m/s")
    print(
        "DOS state count:      "
        f"{dos.integral_states_per_cell:.10f} / "
        f"{dos.expected_states_per_cell}"
    )
    print(f"DOS normalized:       {dos.normalization_certified}")

    if not args.run_fc3:
        print(
            "FC3 was not read. To request SMRTA, provide --fc3 and both "
            "--run-fc3 --allow-heavy."
        )
        return

    # This path is reachable only after both explicit heavy-work confirmations.
    transport = bulk.BulkTransportProblem.from_phono3py(
        args.crystal,
        args.fc2,
        args.fc3,
        mesh=mesh,
        sigma_thz=args.sigma_thz,
        cutoff_frequency_thz=CUTOFF_FREQUENCY_THZ,
        degenerate_tolerance_thz=DEGENERATE_TOLERANCE_THZ,
        make_r0_average=True,
        metadata={"example": "bulk_rta_silicon.py"},
    )
    rta = bulk.compute_rta_thermal_conductivity(
        transport,
        temperatures=[args.temperature],
    )
    output = bulk.write_rta_result_hdf5(
        args.output,
        rta,
        overwrite=args.overwrite,
    )

    print(f"temperature:          {args.temperature:g} K")
    print("SMRTA kappa [W/(m K)]:")
    print(rta.kappa_w_mk[0])
    print(f"RTA archive:          {output}")
    print(
        "CAUTION: an implementation-consistent coarse-mesh value is not a "
        "mesh/smearing-converged silicon prediction."
    )


if __name__ == "__main__":
    main()
