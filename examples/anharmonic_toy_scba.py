"""Run a small pinned-chain cubic SCBA calculation in reduced units.

This is a fast Python-API smoke example, not a grid-refined release
benchmark. Run it from the repository root after installing NEGFPy:

    python examples/anharmonic_toy_scba.py
"""

from __future__ import annotations

from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
    SolverScale,
    solve_scba,
)
from negfpy.anharmonic.toy import pinned_scalar_chain


def main() -> None:
    device, lead, vertex = pinned_scalar_chain(
        1,
        spring=1.0,
        pinning=1.0,
        cubic=0.004,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead=lead, temperature=0.8),
        right=HarmonicReservoir(lead=lead, temperature=0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.12, n_positive=25),
        scale=SolverScale(
            omega0=1.0,
            hbar=1.0,
            metadata={"units": "reduced", "k_B": 1.0},
        ),
    )
    config = SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=80,
        min_iterations=3,
        consecutive_passes=2,
        mixing=0.3,
        sigma_rtol=3.0e-7,
        green_rtol=3.0e-7,
        identity_rtol=1.0e-7,
        current_rtol=1.0e-4,
        current_atol=1.0e-9,
    )

    result = solve_scba(problem, config)
    final = result.history[-1]

    print(f"method:                 {result.method}")
    print(f"converged:              {result.converged}")
    print(f"iterations:             {len(result.history)}")
    print(f"last sigma residual:    {final.sigma_residual:.3e}")
    print(f"last Green residual:    {final.green_residual:.3e}")
    print(f"left current / J0:      {result.currents.left:.9e}")
    print(f"right current / J0:     {result.currents.right:.9e}")
    print(f"contact balance / J0:   {result.currents.contact_balance:.3e}")
    print(f"collision power / J0:   {result.currents.collision_power:.3e}")
    print(f"essential checks pass:  {result.diagnostics.passed}")
    print(f"per-run certified:      {result.release_certified}")

    if result.diagnostics.failures:
        print("diagnostic notes:")
        for failure in result.diagnostics.failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
