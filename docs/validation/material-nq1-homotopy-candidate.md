# Material Nq=1 staged-coupling candidate

Status: development draft only. This file is not a qualification record,
pilot authorization, scheduler submission, or release certificate.

## Immutable failure boundary

The preceding anchor-fix pilot under commit `4fed83a6a4f5b7bef6f0fc4176b1ac83ac5193eb`
is preserved at its remote root. Its `coupling-g0025` task reached scheduler
`F`/`Exit_status=3` after one attempt and exhausted exactly 800 iterations.
The terminal checkpoint was `scientific_failure` with final residuals
sigma `0.806377408268185`, Green `0.014480573782857562`, identity
`0.6345615954827826`, and current-relative error `1.8373639076335317`.
The task receipt is evidence of scientific failure, not a release receipt.
No checkpoint, outcome, or task receipt from that campaign may be used as a
continuation parent for this candidate.

## Observed numerical behavior

The failed g=.025 continuation began with the authenticated coupling-square
predictor, reduced mixing from 0.2 to the 0.001 floor, and reached its lowest
sigma residual near iteration 238. After the floor was reached, sigma and the
identity residual rose steadily through iteration 800. This is an unstable
continuation trajectory, not an infrastructure timeout and not evidence that
the fixed-point tolerances should be relaxed.

## Single proposed hypothesis

Keep the SCBA equations, diagnostics, tolerances, FFT/Hilbert backends, and
fixed mixing rule unchanged. Replace only the large g=.01 -> .025 jump with
an explicit staged coupling path:

```text
.001 -> .010 -> .015 -> .020 -> .025
```

The 19-task graph retains the later coupling stages `.05 -> .1 -> .5 -> 1.0`
and the existing bias, equality, and refinement tasks. This keeps the release
task count at 19 while replacing the unstable large jump with three explicit
intermediate stages.

Each stage is a complete, independently checkpointed SCBA task. The next
stage may start only after the preceding stage has a terminal scheduler F/0,
a self-validating completed checkpoint, passed scientific diagnostics, and no
infrastructure failure. The existing coupling-square predictor is applied at
each edge with the exact squared coupling ratio. Every stage, parent edge,
factor, transformed initial-state hash, and consumed checkpoint identity must
be serialized and included in the canonical task/campaign identity.

This is a new continuation contract. It must not be implemented as an
unrecorded loop inside the g=.025 task, and it must not reuse the failed
g=.025 checkpoint.

## Required candidate work before any heavy submission

1. Decide how the staged path maps to the required 19-task release graph; the
   mapping must remain explicit and hash-bound, not inferred by the runner.
2. Update task, MPI, receipt, checkpoint, strict-validation, and scheduler
   schemas together, with negative tests for skipped stages, wrong parent,
   wrong factor, relabelled coupling, and transformed-state tampering.
3. Add a fresh pilot covering every new edge through g=.025, plus authenticated
   anchor comparisons and independent replay of each continuation transform.
4. Create a new exact commit/tree, environment/input identity, qualification
   report, remote root, execution ID, and fresh scheduler names. Qualification
   must pass both Python suites, MPI smokes, and all provenance tests before a
   pilot job is submitted.
5. Re-estimate memory and walltime for the additional stages. A failed parent
   or any infrastructure event stops the candidate and preserves its tree.

Until these controls are implemented and qualified, this document authorizes
no Aspire2A submission. The previous candidate remains immutable evidence.
