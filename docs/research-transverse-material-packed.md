# Direct material mapping into packed cubic coordinates

`fourier_material_cubic_packed` in `transverse_material_packed.py` maps the audited,
mass-normalized physical-SI source images directly into
`PackedTransverseCubicVertex`. Its arguments match the existing material Fourier
mapper, with an additional `max_workspace_bytes` budget. The returned
`FourierPackedMaterialCubic` contains `vertex` and immutable `metadata`.

The vertex remains in solver units before coupling. The unchanged conversion
`sqrt(hbar)/omega0**2.5` and each multiplicity, anchor and r0 weight apply once.
The existing integer transport geometry, source identity checks, finite open
transport crop on all three legs and transverse relative-image phases apply.
The full harmonic dimension survives in `vertex.full_ndof`; active coordinates
are sorted full-device addresses. No boundary repair or symmetrization occurs.

The first geometry traversal counts images, scalar contributions and momentum
phase work while discovering the union of exactly nonzero Cartesian coordinates.
The second traversal accumulates each small source block directly into the
candidate packed array. A final matrix-sized support scan removes coordinates
that cancel to exactly zero after summation. There is no magnitude threshold,
full-device cubic array or materialized scalar term schedule. Empty support is
valid. Both conversion underflow and accumulation overflow are rejected.

`max_bytes` bounds the candidate packed coefficient/address payload, even when
later cancellation makes the final support smaller. `max_geometry_evaluations`
counts both nonzero-image/coset traversals and bounds r0 source expansion before
it occurs. `max_phase_evaluations` counts nonzero scalar contributions times
`Nq**2`, before cubic allocation.

`max_workspace_bytes` bounds the conservative mapping/construction numeric plan.
For candidate active dimension A and q count Q, it includes two packed coefficient
arrays, a Q²A³ boolean finite-check mask, 128A³ bytes for constructor validation
and support/compaction scratch, address storage and bounded phase/image arrays.
The two arrays cover either candidate plus compacted output or unowned values
plus the validated constructor copy. Source-image position storage is included;
caller inputs, Python object overhead, source identity hashing and geometry-audit
internals are excluded. The plan is not a process RSS promise. Returned metadata
records the plan, candidate/final payloads and exact cancellation count.

Eighteen distinct new synthetic checks passed: seventeen in 1.86 seconds,
then one new partial-compaction check in 1.04 seconds with no prior case replay. Independent finite bond
polynomial Fourier references cover Gamma, complex four-q, sheared determinant-two
cosets, duplicate images, r0 averaging and a 2×2 transverse grid. A mixed Cartesian
fixture matches the previous dense mapper exactly, with full-device cubic
allocations guarded. Other checks exercise input immutability, small coefficients,
exact cancellation, empty support, identity mismatch and pre-allocation limits.
The initial test invocation stopped at collection because an overflow fixture
constructed an already-invalid `SolverScale`; only the fixture was corrected,
and no numerical test was repeated. Production hashes were unchanged throughout.
Evidence is in `multiq-material-packed-development-20261021` under workspace
reports. Actual silicon mapping with this path remains to be run on Aspire2A.
