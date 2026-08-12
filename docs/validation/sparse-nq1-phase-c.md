# Sparse silicon (N_q=1) SCBA validation (`0.3.0`)

This document defines the release gate for the quasi-one-dimensional material
SCBA stage. It is deliberately a gate specification, not a claim that the
release has passed.

## Scientific object

The calculation uses the authenticated phono3py Si-PBEsol FC3 mapping from
Phase B, a nine-principal-layer [100] device, complete FC3 support in the
five-layer interaction crop, harmonic lead buffers padded by the certified
FC3 range, and the production cubic-bubble SCBA. The (N_q=1) case is an
algebraic material validation problem; it is not a converged bulk-film
prediction.

The production state stores only the mixed lesser/greater interaction
self-energy in its atomic checkpoint. Retarded self-energy, Green functions,
currents, and diagnostics are recomputed after restart. A coupling
continuation parent is accepted only when it is a terminal `completed`
checkpoint and its source revision, environment, grid, solver configuration,
device, contacts, scale, and authenticated material hashes all match. The
cubic vertex is the only permitted problem-hash difference between parent and
child.

## Required evidence

The release archive must contain, in canonical order:

1. dense/sparse full-field equality on bounded silicon-like cells;
2. authenticated material sparse mapping and complete-support hashes;
3. zero-start and completed-checkpoint continuation results;
4. contact-current balance and internal collision power with `eta_device=0`;
5. half-spacing, larger-cutoff, half-eta, and changed-mixing refinements;
6. invariance under extending the lead-like interacting buffer;
7. the exact source-tree, environment, input, checkpoint, and output hashes.

Every failed scientific case remains archived with its fixed controls. A
failed endpoint is not converted into a release by changing mixing, eta,
cutoff, or grid after the fact.

## Current status

The original `7b0c169` release campaign is terminal and incomplete: two
coupling tasks passed and the `g=0.025` task ended as a scientific failure at
iteration 200.  It did not regress from a completed state.  The later
Aspire2A recovery campaign, run `material-nq1-release-011` at exact commit
`ad4430c`, was still running at the last locally authenticated scheduler
observation on 2026-08-10.  Its terminal MPI report and strict validation
report have not yet been retrieved, so this document makes no completion or
certification claim for that run.

Post-evidence development is isolated from the exact `ad4430c` worktree and
is described in [material-nq1-development-candidate.md](material-nq1-development-candidate.md).
Any such source change defines a new candidate and cannot be used to
retroactively validate run 011.
