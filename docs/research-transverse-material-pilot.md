# Bounded material-input pilot for coupled transverse q

The next milestone is an input-validation pilot using actual material IFCs
with the implemented crystal mapping, transverse vertex, and thermal
contact interfaces. Preparation is in progress; this document records
the intended scope, not a completed numerical result or submission.

The pilot retains the declared nine-layer, 216-DOF device, interaction
layers 2 through 6 inclusive in zero-based indexing, \(g=0.03\),
\(\eta_{\mathrm{lead}}=0.0015625\), and \(\eta_{\mathrm{device}}=0\).
Reservoir temperatures remain 305 K and 295 K, converted with the same
physical `SolverScale`. The broadening is not reduced.

It evaluates inputs on a four-by-one transverse mesh and constructs the
full reduced cubic vertex in solver units before the coupling-strength
multiplier. It reports both that vertex's norms and \(g\) times those
norms, with `coupling_applied=false`; \(g=0.03\) is retained for a future
Born calculation. It does not run a Born map, SCBA iteration, device
Dyson/Keldysh solve, heat-current integration, or convergence study.
Contacts still require surface Green-function solves. Accepted Nq=1
fields and all historical workflows remain unchanged.

## Why this pilot is the next bounded step

The preceding milestones establish synthetic comparisons for the
[crystal-image mapper](research-transverse-material-mapping.md) and
[paired-q thermal contacts](research-transverse-contacts.md). This pilot
tests whether the authenticated material source can pass those interfaces
with its actual basis, shortest-image weights, interaction crop, and
harmonic lead spectrum.

A `(4, 1, 1)` mesh includes Gamma, a self-opposite Nyquist point, and the
distinct pair \(q=\pm\pi/2\). It therefore exercises complex phases and
opposite-q relations. A `(2, 2, 1)` mesh alone would not exercise distinct
opposite momenta because every sampled point is its own negative.
The selected mesh is a small input probe, not evidence of transverse
sampling convergence.

Transport is kept finite. Its matrix, principal-layer width, basis order,
and crop must come from the authenticated material geometry. They are not
reduced to satisfy the dense toy solver's allocation limits.

## Source-to-vertex checks

The source path is:

1. Authenticate the declared FC2, full FC3, crystal mappings, and source
   audits; retain their source and primitive-identity bindings.
2. Reuse the audited full-source image expansion and SI mass normalization.
3. Map every retained image to the finite transport crop and integer
   transverse offsets using the shared geometry.
4. Stream each complete weighted 3×3×3 solver-unit image block directly
   into the full Fourier vertex. Apply image, anchor, and optional `r0`
   weights once, without constructing scalar Python term objects.
5. Apply the existing reality and two generating leg-exchange checks to
   every block of \(V_{acd}(q,p)\), then report its unscaled norm and the
   corresponding norm at \(g=0.03\).

Preparation identified 22,104,360 retained scalar contributions, making an
explicit scalar-object schedule wasteful. The streaming implementation
`fourier_material_cubic` first counts the retained geometry and nonzero
components, then performs a second geometry traversal to accumulate
complete Cartesian blocks. Execution must confirm the final counts.

The planned bounds are two million image/coset evaluations across both
geometry traversals and 512 million scalar-contribution/momentum-pair
evaluations. The phase-work accounting remains
\(N_q^2N_{\mathrm{scalar}}\), even though the implementation reuses a
phase for every Cartesian component and longitudinal copy of an image.
These explicit pilot bounds do not change the default toy limits or the
existing scalar mapper. Work guards and source checks precede allocation
of the dense vertex.

All contributions to the declared crop must be included for the full
vertex check. A term-count limit is a stopping condition, not permission
to retain an arbitrary prefix. No missing permutation is synthesized,
small coefficient clipped, or sum-rule repair silently applied to obtain
a passing vertex.

## Harmonic and contact checks

The harmonic mapper regenerates transverse FC2 image phases from the same
source geometry. Each sampled device matrix and lead block must retain
the prescribed basis and satisfy the relevant Hermiticity, reverse-block,
and opposite-q conjugacy checks.

Both contacts use the new continuous fixed-\(\eta\) construction on only
three signed nodes:

\[
\omega\in\{-0.025,\;0,\;0.025\}.
\]

This retains the existing frequency spacing in a deliberately small local
window. It is not the 601-point transport grid and cannot support a
frequency-integrated current or a global spectral-accuracy claim.

The contact checks include the per-q longitudinal classification,
zero-frequency surface equation and derivative residuals, exact-zero
Hermiticity, zero-limit positivity, and full signed q/−q identities.
The finite-\(\eta\) derivative is retained in both acoustic and gapped
sectors. The older gapped helper's zero assignment remains separate.

The longitudinal classifier is a numerical spectral check. It does not
replace the material audit of mass-weighted rigid translations. A
classification failure or unresolved soft sector must be reported as
such; no pinning, additional device damping, or broadening change is
introduced by this pilot.

## Memory and work boundaries

For \(N_q\) transverse points, \(D\) finite-device DOFs, and \(N_\omega\)
signed frequencies, the complex128 payloads are

\[
B_V=16N_q^2D^3,\qquad
B_{\mathrm{spectrum}}=48N_qN_\omega D^2,
\]

\[
B_{\mathrm{contact+limit}}=
16N_q(3N_\omega+1)D^2
\quad\text{bytes}.
\]

At \(N_q=4,D=216\), the full vertex contains 2,579,890,176 bytes,
approximately 2.40 GiB. The pilot therefore declares a 4 GiB vertex
payload budget. Construction and immutable validation create additional
copies; the payload budget is not a peak-memory estimate.
Each three-node contact plus its zero limits contains 29,859,840 bytes,
approximately 28.48 MiB, within the contact builder's existing 64 MiB guard.

For a lead principal-layer dimension \(d_{\mathrm{lead}}\), the descriptor
constructs a dense complex operator of size
\(d_{\mathrm{lead}}^2\times d_{\mathrm{lead}}^2\), requiring
\(16d_{\mathrm{lead}}^4\) bytes before solver workspace.
For the declared 24-DOF lead layer this operator is about 5.06 MiB.
Its dense linear solve scales as \(d_{\mathrm{lead}}^6\), so enlarging
principal layers is a separate computational concern.

The intended execution environment is direct Aspire2A with q4 resources:
128 CPUs, 440 GB, and 24 hours. This is a resource plan; the eventual
submission and runtime records determine whether and when a job runs.
Requested resources do not imply the present implementation uses every
CPU or that a runtime estimate has been measured.

The dense SCBA guards are unchanged. At the same four q points and
601 signed frequencies, a single three-component 216-DOF spectrum would
require over 5 GiB. The direct Born kernel would evaluate

\[
N_q^2(3M^2+3M+1)=4{,}334{,}416
\]

frequency/momentum pairs per component for \(M=300\), above its default
100,000-pair bound. Each pair uses three dense \(O(D^4)\) contractions
per component. Allocating the vertex for this input pilot therefore
does not make the existing dense material SCBA calculation practical.

## Intended outcome and remaining work

The pilot should produce source/geometry bindings, complete image/scalar
counts and vertex dimensions, declared coupling and unit factors, full vertex
symmetry results, harmonic/contact diagnostics, and measured execution
resources. Any failed or incomplete stage remains explicit.

A pass would show that these actual inputs reach the coupled-q interfaces
under the declared model and finite mesh. It would not establish
self-consistency, current conservation, material transport convergence,
or a release. A larger calculation still needs an efficient representation
of cubic support and spectra, followed by a bounded material solve and
appropriate physical convergence checks.

The streaming implementation has passed its first focused run:
eight checks in 2.34 seconds, with stable source/test hashes and no fixes
or rerun. Four comparisons cover the existing scalar pipeline at Gamma,
complex four-q points, sheared determinant-two geometry with duplicated
images and `r0` expansion, and a two-by-two mesh; the fixtures exercise
all Cartesian block components. Three resource-limit checks and one
changed-source check reject before dense vertex allocation. These are
synthetic implementation checks, separate from the pending material job.
