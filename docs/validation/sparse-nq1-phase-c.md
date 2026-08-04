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

The coarse low-coupling screening case (`g=0.01`, signed grid with 17 points)
is running remotely under an immutable campaign manifest. It is diagnostic
and cannot certify `0.3.0`; the production gate remains open until the
required refined cases and buffer-invariance check have passed.

