# Material Nq=1 predictor-candidate change record

Status: development draft. This document is neither launch authorization nor
a release certificate.

## Immutable evidence boundary

Aspire2A campaign `material-nq1-release-018` ran exact source commit
`0d1f5bb298b7c267f58b5f444be1e7bedeca01cc`. Its scheduler job
`15185679.pbs101` finished with exit status 2. Only `coupling-g0001` and
`coupling-g001` completed; `coupling-g0025` exhausted the fixed 800-iteration
budget and published a scientific failure, leaving the other sixteen tasks
dependency-blocked. Strict validation, release sealing, retrieval, and final
audit were therefore ineligible and were not run.

That campaign and all of its checkpoints are immutable evidence. This
candidate does not resume, repair, reinterpret, or overwrite them. In
particular, the failed `coupling-g0025` checkpoint is never an initialization
source. Every task in any new pilot must generate its own parent state under
one new exact commit, tree, environment, manifest, and artifact root.

## Single frozen hypothesis

The cubic force-constant vertex is multiplied linearly by `coupling_scale`,
and the cubic bubble contains two vertex factors. For a fixed Green function,
the interaction self-energy therefore scales quadratically with the coupling.
The candidate tests the following coupling-continuation predictor:

```text
Sigma_initial_child^{</>} =
    (g_child / g_parent)^2 * Sigma_parent^{</>}
```

The predictor changes only the initial lesser and greater interaction
self-energy supplied to a fresh child calculation. The retarded component is
rebuilt in the normal solver path. It does not change the SCBA equations,
iteration update, adaptive-mixing rule, convergence diagnostics, or acceptance
thresholds.

The predictor must be explicit in the scientific task identity. Parent and
child couplings, the derived positive real factor, the authenticated
untransformed parent-state hash, the transformed consumed-state hash, and a
canonical initialization-record hash must be bound into manifests,
checkpoints, receipts, and resume validation. Temperature-bias continuation
remains the identity transform. A predictor is never applied while resuming an
existing child checkpoint.

No other scientific control changes in this candidate:

- base frequency grid: step `0.1`, 75 positive points, cutoff `7.5`;
- lead broadening `0.025` and device broadening zero;
- initial mixing `0.2`, mixing floor `0.001`;
- maximum 800 iterations and three consecutive passes;
- existing fixed-point, conservation, field-equality, refinement, and tail
  tolerances;
- existing sparse zero-padded FFT and Hilbert-transform backends.

Grid refinement, a finer coupling ladder, lower mixing, relaxed tolerances,
and Anderson/DIIS acceleration are explicitly outside this candidate.

## Required non-release pilot

Qualification alone must not authorize a full release campaign. The exact
qualified commit must first run a create-only three-task pilot, one scheduler
job at a time:

1. `coupling-g0001`: zero start at `g=0.001`;
2. `coupling-g001`: quadratic continuation from the pilot's completed
   `coupling-g0001` state, factor `100`;
3. `coupling-g0025`: quadratic continuation from the pilot's completed
   `coupling-g001` state, factor `6.25`.

Each child may start only after its new pilot parent has a terminal scheduler
status of success, a completed self-validating checkpoint, passing scientific
diagnostics, and no infrastructure failure. A scientific failure, scheduler
failure, watchdog event, anchor mismatch, or evidence-integrity error stops
the pilot and preserves everything already written.

Every pilot task and any later full-release task is one immutable initial
attempt in a virgin, non-symlink artifact root. Neither workflow infers resume
permission from a checkpoint or infrastructure-failure record. Infrastructure
interruption stops that execution; recovery requires a separately identified
and qualified execution while preserving the interrupted tree.

The two completed release-018 results at `g=0.001` and `g=0.01` are comparison
anchors only. They may be authenticated and compared after the corresponding
new task completes, but they must never be supplied as continuation parents.
All six Green-function and interaction-self-energy fields must agree at the
existing field tolerance (`rtol=1e-8`, `atol=1e-12`), together with the
registered current and conservation quantities.

A pilot pass is still not a release. It may only authorize consideration of a
separately identified full 19-task campaign from the identical source commit
and tree. Any source or control change after observing the pilot requires a
new candidate and a new qualification.

## Qualification boundary

Before the pilot, the new candidate must have a clean exact commit and tree,
an authenticated environment and material input set, unchanged core SCBA
solver bytes, complete Python 3.10 and 3.12 suites, two- and four-rank MPI
smokes, predictor scaling/symmetry/provenance/restart/tamper tests, harmonic
and mode-set preflights, and explicit memory/disk/thread guards. The
qualification report must bind both the complete prospective 19-task control
set and its exact three-task pilot subset, state that release submission is
ineligible, and name the pilot as the next gate.

The full-release launcher and strict validator accept only the exact v3
qualification together with the recursively authenticated, mode-0444 pilot
manifest, report, and audit from the identical commit, Git tree, package tree,
environment, and material inputs. Legacy v2 qualification and qualification
without a passing pilot audit are not launch paths.
