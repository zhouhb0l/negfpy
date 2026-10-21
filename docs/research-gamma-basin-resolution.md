# Resolving acoustic minima near Gamma

The research contact classifier now judges a soft minimum against the frequency
resolution and measured local dispersion. It no longer treats an optimizer's
absolute position error of more than `1e-7` radians as sufficient evidence of a
non-Gamma minimum. Lead blocks, broadening, force constants and eigenvalues are
not modified. Raw optimizer positions remain in the report.

This responds to the preserved silicon diagnostic `25894277.pbs101`. Its Gamma
optimizer returned `-1.344539962e-7`, while the eigenvalue differences near Gamma
were around `1e-15`, below the existing `4.133644871e-12` eigenvalue tolerance.
A separate Nq1 assembly differed in lead blocks only around `1e-16` and passed
the old location guard. Those completed calculations and their original failed
classification remain unchanged. This code change has not yet been validated
on the actual material inputs.

## Numerical criterion

The original spectral scan, Gamma nullspace, acoustic sum-rule residual,
projected-curvature and negative-eigenvalue gates remain. The existing
scale-dependent eigenvalue tolerance is denoted by `tau`. For a candidate
acoustic lead, let `h=2*pi/1024` and `m` be the Gamma nullity.

1. Evaluate the actual lowest `m` eigenvalues at `-h`, `-h/2`, `h/2`, and `h`.
   Subtract their sorted Gamma eigenvalues and divide each rise by `q**2`.
   Every rise must exceed `32*tau`. The coefficients must be positive and
   agree across signs and scales within 0.5% plus propagated eigenvalue
   uncertainty. The allowance is `0.005*max(kappa) + 2*max(2*tau/q**2)`.
2. Retain the original projected-curvature screen as well. The new actual
   spectral rise matters because projection alone omits optical relaxation:
   even a flat or quartic branch can have positive projected curvature.
3. Compute the first-order operator in the Gamma nullspace,
   `N.conj().T @ (1j*(d01-d10)) @ N`. Its operator norm must not exceed
   `max(128*eps*full_scale, 4*tau/h)`. A resolved slope can identify a
   displaced basin even when the Gamma energy itself lies below tolerance.
4. Set `delta_energy=4*tau` and `kappa_lower=0.5*min(kappa)` over all probes
   and acoustic branches. The candidate position resolution is
   `sqrt(delta_energy/kappa_lower)` and must lie below `h/4`. This keeps the
   unresolved core inside the measured local basin.
5. Each soft minimum must come from an optimizer bracket containing an
   equivalent Gamma point, belong to a Gamma-null branch, lie within that
   position resolution, and have an eigenvalue within `delta_energy` of its
   Gamma branch value. All candidates must satisfy this test.

A passing minimum is compatible with Gamma at this numerical resolution. Its
raw position is not snapped to zero, and no physically displaced minimum is
claimed to have moved. A rejected or unresolved basin reports zero for
`gamma_q_resolution`; the measured coefficients and slope remain available.

These are conservative numerical screens, not rigorous interval error bounds
or universal dispersion certification. Very soft modes, nearby optical
crossings and strong higher-order dispersion can remain unresolved. Sorted
branch indices are used for a conservative spectral comparison, not mode
tracking across crossings. No threshold is relaxed in the existing stability
or ASR checks.

All negative evidence found by either shifted-grid or centered-grid refinement
and by the new local probes is retained in the final stability minimum. This
also corrects a pre-existing omission in which the shifted-grid minimum could
be lost before the final unstable classification.

## Evidence and compatibility

The private spectral report includes raw minimum positions, branch indices,
values and brackets; per-minimum Gamma compatibility; the measured basin
coefficients; position and energy resolutions; and the projected slope norm
and tolerance. Arrays are immutable. The public material audit forwards the
core basin evidence and identifies policy `gamma-basin-resolution-v1`.
Its canonical classification hash uses domain
`negfpy-harmonic-lead-classification-v2` and includes this evidence. Lead physics
identities are unchanged. Existing frozen material records keep their original
source and report identities.

Fresh synthetic controls exercise rounded optimizer locations under scaling,
small assembly perturbations, degenerate acoustic modes, optical relaxation,
resolved phase shifts, separated soft minima, negative modes, flat and quartic
branches, evidence forwarding, and finite-eta paired-q contact construction.
The separate reports directory records the actual test execution and any
failures. Historical material jobs and test campaigns are not repeated.

The accepted material source and fields stay fixed. Material settings remain
`g=.03`, `eta_lead=.0015625`, and `eta_device=0`; the half-eta proposal stays
cancelled. A new actual-material validation is still required before claiming
that the earlier material-input rejection is resolved in practice.
