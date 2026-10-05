# Full-domain spectral moments for research runs

`ResearchMomentDiagnostic` makes the bounded full-domain GL8/16 check reusable
at the retained input after a fresh final research map. It is opt-in; neither
the production defaults nor SCBA acceptance criteria change.

```python
from negfpy.anharmonic import solve_scba
from negfpy.anharmonic.research_moment import ResearchMomentDiagnostic

moment = ResearchMomentDiagnostic(
    max_points=40000, max_seconds=1800, max_leaves=1024, max_workers=8,
    rtol=.002, atol=1e-8, moment_rtol=.005,
    interpolation_policy="equilibrium-broadening",
)
result = solve_scba(
    problem, config, initial_sigma_pp=initial_guess,
    research_map=bounded_map, research_moment=moment,
)
print(result.converged)  # Existing solver decision, unchanged by this check.
print(result.diagnostics.metrics["green_spectral_moment_residual"])
print(result.metadata["research_moment"])
# moment.arrays retains every final leaf for evidence archiving.
```

An explicit research map is required. The diagnostic runs once, after its
fresh final map completes, on that evaluation's input self-energy. Rejected
Anderson proposals never become diagnostic inputs; the retained reference is
used. If the fresh final map cannot finish, the report is explicitly skipped
with `evaluated=False` and `reason="fresh_final_map_incomplete"`. A complete
final map can also occur after iteration exhaustion; a passing moment does
not convert such a run into SCBA convergence.

The old `green_spectral_moment_residual`, diagnostic failures, history,
converged status and release status are retained. New data are stored under
`metadata["research_moment"]`. The report contains:

- `coarse_spectral_moment_residual`: the unchanged historical metric.
- `quadrature`: full coverage, sampled maximum/Frobenius error estimates,
  tolerances, point count, leaf count, stop reason and matrix moment metrics.
- `moment_screen`: the integrated matrix's identity residual and its separate
  threshold. This screen can pass even when quadrature accuracy fails.
- `passed`: both sampled integration checks and the moment screen passed.
- Hashes of all three input fields and an unchanged-field check.

All original signed node-to-node intervals start with fresh GL8/16 panels;
there is no low-order unselected remainder and no imported adaptive state.
Zero is a partition boundary and is never queried. Both signs are evaluated.
An atomic split replaces its parent only after both children complete.
Submitted work that finishes after a budget is counted but discarded.
Partial initial coverage cannot publish a full-domain moment or pass.

`moment.arrays` contains `grid_values`, `leaf_bounds`, `leaf_parent_indices`,
`leaf_low`, `leaf_high`, `leaf_error_abs`, initial aggregate moments only after
initial coverage completes, and `full_domain_moment` only for complete coverage.
These arrays let a caller reconstruct the final matrix, both summed error
norms and domain coverage. The initial panels replaced during refinement are
not retained individually. For the 216-degree-of-freedom material case, archive
at most 64 leaves per shard to avoid very large files. A separate instance is
needed per concurrent run; sequential reuse clears prior evidence at startup.

The helper uses eight workers by default and limits BLAS to one thread during
integration. The integration time cap is cooperative: callers must additionally
bound the process for stalled linear algebra or evidence output. Budget outcomes
are returned with failed/partial status. Invalid inputs and numerical exceptions
propagate rather than being converted into successful evidence.

The default policy is `equilibrium-broadening`, requiring equal positive
reservoir temperatures and zero device eta. `linear-keldysh` is also available
explicitly, as in the pointwise module; it is not a transport validation claim.
The integral covers only the archived frequency domain with the declared
self-energy interpolation. Sampled GL8/16 agreement is not a rigorous error
bound, and the small identity residual does not measure actual integration
error. Neither unknown tails nor biased-transport conservation are established.

## Material basis

The numerical definitions were extracted without algorithm changes from
full-domain diagnostic 25688099.pbs101, profile
7afbd4c55ed868d95fb2a6b7c44bd0e4590be81816e1636282680d96682a56d1.
At the accepted g=.0175 field, all 600 intervals gave 732 leaves / 20736 calls,
passing both sampled estimates and yielding a moment residual 6.26697e-6.
Fourteen closed archives were independently verified and reconstructed.
This extraction reuses that evidence; it does not rerun or extend that consumed
calculation. The wrapper and final-map routing are covered separately by toy
and solver regression tests. Nq=1 remains a research implementation.
