# Thermal contacts on a paired transverse-q mesh

[`transverse_contacts.py`](../src/negfpy/anharmonic/transverse_contacts.py)
builds signed thermal contact spectra for the bounded coupled-q solver.
It extends the existing acoustic surface-derivative construction to complex
opposite-q pairs and uses the continuous fixed-broadening zero-frequency
limit in every stable sector, including gapped sectors. Focused synthetic
validation passes all 18 new checks.

This is a separate contact construction. Historical Nq=1 material fields,
broadening choices, and results are unchanged. The older
`gapped_contact_spectrum` helper retains its existing zero-node assignment.

## API and connection to the mapped device

```python
from negfpy.core.types import LeadBlocks
from negfpy.anharmonic.types import HarmonicReservoir
from negfpy.anharmonic.transverse_contacts import build_transverse_contact
from negfpy.anharmonic.transverse_mapping import build_transverse_problem

def left_reservoir(kpar):
    return HarmonicReservoir(
        LeadBlocks(*lead.blocks(kpar)), temperature=left_temperature,
    )

def right_reservoir(kpar):
    return HarmonicReservoir(
        LeadBlocks(*lead.blocks(kpar)), temperature=right_temperature,
    )

left = build_transverse_contact(
    mesh, grid, left_reservoir, ndof, eta_lead, side="left",
)
right = build_transverse_contact(
    mesh, grid, right_reservoir, ndof, eta_lead, side="right",
)
problem = build_transverse_problem(
    grid=grid,
    vertex=vertex,
    device=device,
    left=left.at_q,
    right=right.at_q,
    input_units="solver",
    scale=scale,
)
```

Each reservoir callback receives two transverse **radian** coordinates and
returns a `HarmonicReservoir`. Its harmonic blocks, coupling, temperature,
and positive `eta_lead` already use dimensionless solver units. The mesh
has shape `(ny, nz, 1)`, and the finite transport direction stays inside
the device and lead blocks.

The example uses the matched periodic coupling. A different boundary
requires an explicit `device_to_lead` matrix and matching `device_indices`.
The existing left/right surface orientation is retained. One physical
reservoir has one temperature and one device embedding across its entire
q mesh; the left and right reservoirs may have different temperatures.

The returned `TransverseContact` contains:

- `spectrum`: immutable retarded, lesser, and greater arrays with shape
  `(Nq, Nomega, ndof, ndof)`.
- `zero_limits`: the embedded matrices \(L_\eta(q)\), with shape
  `(Nq, ndof, ndof)`.
- `q_reports`: sector classifications, source and descriptor identities,
  surface/derivative residuals, conditioning, and zero-limit diagnostics.
- `checks`: joint identity and correlation-positivity checks.
- The mesh, grid, side, temperature, and lead broadening.

`at_q(kpar)` serves a precomputed `KeldyshSpectrum` at a resolved mesh point.
Equivalent negative or reciprocal-period-shifted coordinates are accepted.
It neither interpolates between q points nor recomputes a contact. The
complete `spectrum` can also go directly into a `TransverseSCBAProblem`.

## Fixed-broadening zero-frequency limit

At fixed transverse q, let \(A,B=A^\dagger\) be the outward lead couplings,
\(\tau\) the device-to-lead coupling, and

\[
z=(\omega+i\eta)^2,\qquad
g(z)=\left[zI-D_{00}-A g(z)B\right]^{-1},\qquad
\Sigma^r=\tau g\tau^\dagger .
\]

At zero frequency, \(z_0=-\eta^2\) lies below a stable lead's spectrum.
The surface \(g_0=g(z_0)\) and \(\Sigma^r(q,0)\) are Hermitian, but can have
complex off-diagonal entries. Their exact zero-frequency broadening vanishes.
Differentiating the surface equation with respect to \(z\) gives

\[
g_z-g_0 A g_z B g_0=-g_0^2.
\]

The existing descriptor solves this linear matrix equation and checks its
residual and conditioning. Since \(dz/d\omega=2i\eta\) at zero,

\[
\Sigma'_\omega(0)=2i\eta\,\tau g_z\tau^\dagger,\qquad
L_\eta(q)=\lim_{\omega\to0}\frac{\Gamma(q,\omega)}{\omega}
=i\left[\Sigma'_\omega(0)-\Sigma'_\omega(0)^\dagger\right],
\]

where \(\Gamma=i(\Sigma^r-\Sigma^{r\dagger})\).
For the semi-infinite Hermitian lead operator \(H(q)\) and surface
projection \(P\), the same limit is

\[
L_\eta(q)=
4\eta\,\tau P[H(q)+\eta^2 I]^{-2}P^\dagger\tau^\dagger
\succeq0.
\]

This holds for both acoustic and gapped stable sectors. A gapped sector
generally has a nonzero derivative at finite \(\eta\), although its
zero-frequency broadening itself is zero.

The nonzero-frequency thermal fields are

\[
\Sigma^<=-i n_B(\omega,T)\Gamma,\qquad
\Sigma^>=-i[1+n_B(\omega,T)]\Gamma.
\]

Their continuous zero-frequency limits at fixed \(\eta>0\) are

\[
\Sigma^<(q,0)=\Sigma^>(q,0)=-iT L_\eta(q).
\]

At zero temperature both limits vanish. The implementation reuses the
stable Bose-times-broadening evaluation away from zero and the descriptor
limit at zero, avoiding a divergent Bose factor times a numerical zero.

Every q sector uses this limit. A real-space supercell containing acoustic
and optical modes includes all of them in its surface derivative.
Assigning zero to gapped sectors after Fourier decomposition would omit
their finite-\(\eta\) contribution and change that contact representation.

The older gapped helper assigns zero lesser/greater values at its exact
zero node. That convention remains unchanged and is not interchangeable
with the continuous fixed-\(\eta\) convention here. This is an explicit
contact-discretization choice; it does not alter the user-selected
broadening or accepted material results.

## Opposite-q checks and signed-frequency construction

Input onsite, forward/reverse hopping, and resolved device-to-lead
couplings must satisfy

\[
D_{ij}(-q)=D_{ij}(q)^*,\qquad \tau(-q)=\tau(q)^*.
\]

Lead dimension, embedding indices, and temperature stay constant across
the mesh. These checks assume the same declared cell-periodic basis as the
harmonic and cubic mappers. Source blocks are copied; inconsistent pairs
are rejected.

The descriptor supplies the exact-zero retarded matrix. Every nonzero
signed-frequency node is computed directly with the existing surface
solver. No same-q elementwise-conjugation rule creates negative frequencies.
Final joint checks include

\[
\Sigma^r(q,-\omega)=\Sigma^r(-q,\omega)^*,\qquad
\Sigma^<(q,-\omega)=\Sigma^>(-q,\omega)^T,
\]

\[
L_\eta(-q)=L_\eta(q)^*,\qquad
\Sigma^>-\Sigma^<=\Sigma^r-\Sigma^{r\dagger}.
\]

Lesser/greater anti-Hermiticity and positivity of \(i\Sigma^{</>}\) are also
checked. Zero limits are Hermitian and positive semidefinite without
requiring each complex q block to be real or symmetric. At a momentum
equal to its own negative, the joint source condition reduces to the
corresponding real-basis constraint.

No clipping, Hermitian projection, symmetrization, source adjustment, or
device damping is applied to make these checks pass.

## Sector classification and implementation reuse

At each transverse q, the existing classifier samples and refines the
**longitudinal** Bloch spectrum

\[
D(k_\parallel;q)=D_{00}(q)
+D_{01}(q)e^{ik_\parallel}+D_{10}(q)e^{-ik_\parallel}.
\]

Transverse q and this longitudinal scan are distinct coordinates.
A nongamma transverse sector can be gapped even when the full material
has acoustic modes. It receives the same fixed-\(\eta\) construction.

Accepted classifications are `gapped` and `acoustic`. Unstable and
unresolved `invalid-soft` sectors are rejected. The inherited acoustic
classifier requires resolved soft minima at longitudinal Gamma and
positive, consistent local curvature. It does not accept every shifted
soft minimum or flat soft branch. Numerical sector classification does not
replace the material audit of the mass-weighted rigid-translation subspace.

The shared derivative is extracted into
`_build_fixed_eta_contact_descriptor` in
[`acoustic_contacts.py`](../src/negfpy/anharmonic/acoustic_contacts.py).
The existing `_build_acoustic_contact_descriptor` remains acoustic-only;
its established equations and acoustic descriptor identity are retained.
A cached classification must match the current lead's physics identity.
Existing gapped contact construction does not call the generic helper.

The default surface solver is `sancho_rubio`. Nonzero nodes use methods
supported by core `surface_gf`; the separate historical `analytic_diagonal`
contact path is not a full-grid method of this builder.

## Bounds and validation status

`q_samples` controls the longitudinal classification scan at each fixed
transverse q; it does not change the transverse mesh. Its default is 257
points. The default base-work
guard is 100,000 q/frequency and q/scan evaluations. Adaptive basin
refinement and linear algebra are additional work, so this is not a
strict wall-time or total-iteration limit.

The 64 MiB default guards bound the combined returned spectrum/zero-limit
payload, accumulated input lead storage, and each dense surface-derivative
operator separately. They do not bound total process memory.
The coupled-q solver's own bounds still apply.

Both new focused suites passed on their first runs without numerical fixes:
16 unit tests in 2.95 seconds and two independent reference cases in
1.90 seconds, for 18 distinct passing checks. Ruff passed. A blank-line
cleanup in the shared descriptor module after the reference run was
verified to leave its Python AST unchanged; the unit run used the cleaned
file. No numerical replay was required.

The independent reference in
[`test_anharmonic_transverse_contacts_reference.py`](../tests/test_anharmonic_transverse_contacts_reference.py)
constructs a real spring supercell, then compares its complete signed
contact spectra against the Fourier-transformed q construction on four-by-one
and two-by-two transverse meshes. Both outward orientations are tested.
The model has an acoustic Gamma sector, gapped remaining sectors, and
directional hopping and coupling. The four-q case has complex Hermitian
zero-limit matrices that are not individually real symmetric.
An independent finite-frequency difference of full real-space surface
solutions also checks the zero-limit derivative at fixed \(\eta=0.05\).

The unit tests in
[`test_anharmonic_transverse_contacts.py`](../tests/test_anharmonic_transverse_contacts.py)
cover:

- Exact scalar acoustic and gapped zero limits and near-zero continuity,
  in both orientations, at fixed \(\eta=0.0015625\).
- The unchanged historical gapped zero assignment and exact array equality
  with the existing Nq=1 acoustic contact path.
- Zero-temperature limits, immutable outputs, cached mesh lookup, and
  integration with the harmonic toy solver without recomputing contacts.
- Rejection of inconsistent opposite-q sources, changing temperatures or
  embeddings, unstable/unresolved soft sectors, exceeded bounds, and
  mismatched cached classifications.

These are bounded synthetic contact and interface checks. The chosen
broadening values are test inputs, not an eta-refinement study.

Actual material integration, material-specific acoustic audits, frequency
and transverse-mesh convergence, strong-coupling robustness, and scalable
contraction remain later stages. This contact implementation alone adds
no material SCBA or transport-convergence claim.
