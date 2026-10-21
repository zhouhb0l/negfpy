# Bounded packed material SCBA

`transverse_material_scba.py` iterates the previously validated
`evaluate_packed_born_map` without changing that evaluator, the interaction
kernel, contacts, broadening, or coupling convention. It retains all harmonic
coordinates and uses the exact active support only for interaction fields.

The initial material trajectory uses the authenticated candidate
`0.1 * F(0)` from the completed zero-input Born map. This is an unevaluated
candidate. It must pass a new Dyson/Born evaluation before being reported as
an evaluated state. The original map and its harmonic currents are preserved.

## Iteration and stopping

`run_material_scba(grid, vertex, dynamical, left, right, sigma_initial=...,
config=..., on_start=..., on_evaluation=...)` uses fixed convex mixing:

```
next_input = (1 - mixing) * evaluated_input + mixing * raw_Born_image
```

The default mixing is 0.1 and default evaluation budget is 12. Each completed
state contains its input Sigma, evaluated Green and currents, and a separately
named raw Born image. The raw image has not itself been used in Dyson.
The solver copies supplied initial fields and never mutates caller fields.

Convergence requires the **unmixed** map-minus-input maximum-entry defect of
each retarded, lesser, and greater component to be at most
`sigma_atol + sigma_rtol * max(maxabs(input), maxabs(image))`. It also records
each absolute maximum and Frobenius defect, the corresponding field scales,
and the relative Frobenius defect. Thus a small mixing coefficient cannot
create a false convergence flag. The Frobenius records are diagnostics, not
an additional acceptance criterion.

Current tests use the same evaluated input. Contact balance, interaction
collision power, and full balance must each be at most
`current_atol_w + current_rtol * max(abs(left), abs(right))`. Full Dyson balance
alone is insufficient: contact and collision terms can cancel before SCBA
converges. The default tolerances are `sigma_rtol=1e-4`, `sigma_atol=1e-10`,
`current_rtol=1e-3`, and `current_atol_w=1e-14 W`. Identity checks remain those
of the unchanged evaluator. A numerical fixed point on this grid does not
establish q/frequency convergence or broadening independence.

`on_start(index, sigma_input)` can record candidate hashes before a map.
`on_evaluation(state)` runs synchronously after each successful evaluation,
including the last budgeted or converged evaluation. The scalar-only record
can be serialized with `dataclasses.asdict(state.record)`. Checkpoints store
the evaluated input and raw image as separate three-component spectra,
together with currents, residuals, and provenance. They need not duplicate
Green arrays: a continued calculation evaluates its next input again.
Callbacks must not retain historical arrays; checkpoint errors propagate.

Numerical failure returns `stop_reason="evaluation_error"` and a nonempty
`failure_reason`, retaining the last completed state (or `None` if the first
evaluation failed). There is no automatic retry or parameter change.
Ordinary evaluation/time budget stops have `failure_reason=None`, with
`converged=False`. Execution completion and convergence are separate claims.

Before another map the remaining budget must exceed a 300-second reserve
plus 1.25 times the larger of the initial 1200-second estimate and all
completed map durations. The default total budget is 79200 seconds. Callback
time is included in elapsed time before the next decision. This is a runtime
estimate, not an interruption guarantee for a single map; an outer job limit
and checkpointed completed states remain necessary.

## Memory and work

`plan_material_scba` is dimension-only. The conservative numeric allocation
bound includes the map evaluator's owned peak, one next-input spectrum, the
previous evaluated input/Green/image, and bounded matrix-block scratch. The
previous complete state remains available if the next map fails. Initial
caller fields, material vertex/contacts/dynamical arrays, Python metadata,
library scratch, and any callback-retained arrays are separate. History
contains only scalar records. The full 4-q/601-node/120-active-coordinate
trajectory therefore has a higher bound than a single Born map; no prototype
cap is merely raised. The configured memory/work gates precede evaluation.

The planned maximum trajectory work is the evaluation count times the
per-map contraction and full-device Dyson counts; it is an upper bound, not
a timing estimate. The default 16-GiB solver-owned allowance covers recovery
state retention as well as the next map. Saved material files must be
freshly authenticated by the loading/driver layer before reuse.
