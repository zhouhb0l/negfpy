# Inspecting saved current memory curves

`diagnose_current_memory` postprocesses an existing connected-current integrand.
It integrates from zero at every saved lag, then inspects an explicitly selected
late-memory window. Equal currents at a few endpoints can hide oscillations;
the sampled minimum, maximum and peak-to-peak range expose those variations.

```python
from negfpy.anharmonic import diagnose_current_memory

diagnostic = diagnose_current_memory(
    lag_times, connected_integrand_solver,
    current_scale=current_scale,
    harmonic_current=harmonic_current_w,
    reference_current=contact_current_w,
    window_start=120.0,
    frequency_step=0.00625,
)
```

The two current inputs and returned curves share physical units; the integrand
is in solver units. Nonuniform lag coordinates are supported. The window start
must be a stored node. Passing frequency spacing enforces the existing
half-recurrence limit. Results contain read-only cumulative curves and a
JSON-compatible summary; caller inputs remain writable and unchanged.

The range is computed before adding the harmonic offset, so a large harmonic
current cannot erase a small connected-current variation through subtraction.
A zero reference yields an undefined relative sensitivity (`None`). A flat
finite window never sets convergence or solver-acceptance flags. No omitted
tail, between-node accuracy, static mean or full spatial conservation is inferred.
No averaging, damping or extrapolated steady current is returned.

`scripts/analyze_selected_current_memory.py` applies this diagnostic to a
trusted completed selected-current report. It authenticates the supplied
profile and frozen operations, reruns that report's pinned terminal assessor,
requires the original assessment to reproduce, and verifies every declared
archive before reading the scalar integrands. It writes a separate output
directory containing JSON, cumulative curves and optional PNG/SVG plots.
It never submits jobs, evaluates Green functions or modifies accepted results.

Example (use the actual job identity and pinned hash):

```sh
python scripts/analyze_selected_current_memory.py \
  --report /path/to/completed-report \
  --job-id JOB_ID --profile-sha256 PROFILE_SHA256 \
  --window-start 120 --output /path/to/new-analysis --plot
```

Compare frequency grids at the same memory window. Report variation relative
to the total/reference current separately from variation relative to the small
cubic contribution. Neither is an error bound on the infinite-memory result.
