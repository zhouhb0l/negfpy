# Lead-buffer development for Nq=1

`add_harmonic_lead_buffers` prepares a larger device for a separate buffer
study. It adds explicit layers of each periodic harmonic lead, retaining the
original device, interaction crop, coupling strengths, grid, scale and reservoir
temperatures. The accepted calculation is never modified.

```python
from negfpy.anharmonic import add_harmonic_lead_buffers

embedding = add_harmonic_lead_buffers(problem, left_layers=1, right_layers=1)
padded_problem = embedding.problem
padded_sigma = embedding.embed_self_energy(
    sigma, max_output_bytes=4 * 2**30,
)
# Compare the same physical coordinates/cuts using these mappings:
original_coordinates = embedding.original_indices
original_cuts_in_padded_device = embedding.original_cuts
```

All three supplied self-energy components are copied exactly into the shifted
original-device block; new rows and columns are zero. Retarded fields are not
reconstructed. This represents a fixed-field diagnostic or an explicit input
to a fresh solve, not a resumed checkpoint or an accepted padded SCBA solution.
No iteration history or convergence pass transfers.

Dense and sparse cubic vertices are supported. Sparse vertices remain sparse:
all ordered indices are shifted, all coefficients are retained exactly, and no
new interaction term is introduced. Dense expansion and full self-energy
expansion have explicit byte limits. These limits bound array payloads, not
peak RSS; the validated spectrum constructor also copies its input arrays.

The old contact coupling becomes the old-device/new-buffer junction. Its
orientation and any explicit boundary-coordinate ordering are preserved. The
new outer contact is a periodic lead contact. Leads can differ, counts can be
unequal, and only one side need be extended. On a padded side the lead must
have the device layer dimension and the contact must cover one full boundary
layer; other geometries are rejected instead of being silently rearranged.

## What a buffer comparison can establish

This is the implementation prerequisite for comparing a fixed interaction crop
under different harmonic reservoir boundaries. It does not enlarge the FC3
crop. A separate study that extends the *interacting* lead-like region still
needs authenticated additional FC3 terms and fresh SCBA solutions; this helper
must not be described as completing that release requirement.

Production calculations have zero device eta and finite lead eta. Added
explicit layers use the device prescription, so the location of the regulator
changes when the reservoir boundary moves. At finite lead eta, exact equality
of original and padded Green fields is therefore not guaranteed. Buffer length
and lead-eta sensitivity must be reported separately. Do not introduce device
eta to manufacture invariance.

Small-system tests verify the shifted cubic pair contraction, mapped current
operators, unchanged three-component fields, complex/reordered contacts and
an independent elimination of the added layers, including lesser/greater
fields with unequal temperatures. A finite-eta test demonstrates that the
buffer difference decreases as eta is reduced. None of those tests establishes
material buffer convergence.

## Current practical roadmap (2026-10-10)

1. **Completed:** centered g=.03 equilibrium and 305/295 K SCBA convergence;
   full-domain moment diagnostics; all eight internal current cuts compared.
   The eight-cut spread is 0.011399% of the refined contact current at the
   tested grid/memory. The small cubic term remains memory-sensitive.
2. **Implemented here:** harmonic lead-buffer construction and exact fixed-field
   transfer, with small-system correctness checks. Material coordinate/support
   preparation is recorded separately from numerical current evaluation.
3. **Next material diagnostic:** compare the existing nine-layer device with
   one added harmonic layer on each side, first at the same accepted fields
   and lead eta. Report projected Green/current changes; interpret a changed
   boundary regulator explicitly. This requires a distinct approved job.
4. **Remaining Nq=1 work:** fresh field/grid, cutoff and lead-eta refinements;
   interacting-buffer/crop checks and full transport assessment. Preserve the
   separate failed coarse-contact gate. Do not rerun converged SCBA solely to
   improve a displayed residual.
5. **Later:** coupled transverse-momentum SCBA and material/interface benchmarks.
   Physical static means and feedback are a separate model, as described in
   `research-nq1-mean-policy.md`, not an implicit change to centered SCBA.

The frozen accepted/consumed job packages remain authoritative. No release or
convergence flag is promoted by constructing the padded problem.
