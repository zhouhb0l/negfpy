# Historical TaN phonopy transport debugging

This note preserves the material-specific debugging record formerly stored in
the repository-root `PROJECT_NOTES.md`. It describes work performed for the TaN
phonopy-based harmonic transport study. It is historical context, not a
release-certification record or a declaration of current production defaults.
Current release calculations remain subject to the IFC audits, complete
Brillouin-zone weighting, eta policy, and validation gates documented by the
platform.

## Repeat-one supercell directions

For phonopy IFCs, a supercell direction with repeat one collapses the
real-space indexing along that direction. Interactions crossing that periodic
boundary must therefore be reassigned with the minimum-image rule.

The earlier workflow applied this correction only in the transport direction.
This could misassign transverse boundary-crossing couplings as same-cell
couplings. The search was generalized to all repeat-one axes in
[`ifc_bulk.py`](../../src/negfpy/workflows/ifc_bulk.py).

## Principal-layer dependence

After unwrapping, the builder could misclassify a phonopy IFC and fall back to
an approximate reconstruction. This produced an unphysical dependence on
whether the same system used principal-layer size one, two, or four.

The builder's transport-axis completeness logic and transport-axis Nyquist
handling were revised in
[`ifc_to_terms.py`](../../src/negfpy/modeling/builders/ifc_to_terms.py).

## Transverse Nyquist handling

Incomplete transverse Nyquist handling could distort the real-space
representation used for continuous transverse wavevector integration.
Explicit positive/negative Nyquist partners were introduced before the final
transport construction in
[`ifc_bulk.py`](../../src/negfpy/workflows/ifc_bulk.py).

## Historical hybrid-mesh observation

The TaN study observed a numerical tradeoff:

- a shifted mesh reduced some high-frequency spikes but could suppress the
  low-frequency onset by omitting Gamma;
- a centered mesh restored the low-frequency onset but could emphasize sharp
  resonances.

The workflow consequently gained `hybrid_centered_shifted` and
`hybrid_shifted_centered` modes, with separate `kmesh.mode_low` and
`kmesh.mode_high` controls. These options remain implemented in
[`ifc_bulk.py`](../../src/negfpy/workflows/ifc_bulk.py), but the historical
observation is not evidence that a hybrid mesh is universally preferred.
Release calculations must demonstrate the prescribed Brillouin-zone
convergence without dropping wavevectors.

## Preserved study payload

The material inputs and historical outputs are retained outside the source
repository under the
[TaN material study](../../../studies/materials/TaN/). Older manifests may
contain original-machine absolute paths; they are preserved as historical
provenance and are not portable launch configurations.
