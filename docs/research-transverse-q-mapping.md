# Explicit real-space inputs for the coupled-q prototype

This bounded adapter connects explicit transverse real-space cubic terms,
harmonic `DeviceKSpace`/`LeadKSpace` providers, and signed contact spectra
to the [coupled-q toy solver](research-transverse-q-solver.md).
The implementation is in
[`transverse_mapping.py`](../src/negfpy/anharmonic/transverse_mapping.py).
All 26 focused mapping tests pass, including independent real-space DFT
and self-consistent comparisons. No actual material dataset has been
mapped or solved by this milestone.

This module accepts an explicit finite-device interaction schedule.
The subsequent [crystal-image adapter](research-transverse-material-mapping.md)
now constructs that schedule from audited source IFCs and passes 23
focused synthetic checks. Neither path recovers transverse phases from old
Nq=1 fields or certifies material transport. Accepted Nq=1 inputs and
historical results remain separate.

## API

```python
from negfpy.anharmonic.transverse_mapping import (
    TransverseCubicTerm,
    mass_normalize_cubic_terms,
    fourier_transverse_cubic,
    material_harmonic_providers,
    gapped_contact_spectrum,
    build_transverse_problem,
)

# raw_terms is a complete ordered finite-device schedule, with explicit
# cell offsets. Each displacement coordinate has an entry in dof_masses.
solver_terms = mass_normalize_cubic_terms(
    raw_terms,
    dof_masses,
    force_constant_units="eV/angstrom^3",
    mass_units="amu",
    scale=scale,
)
vertex = fourier_transverse_cubic(mesh, ndof, solver_terms)

device, lead = material_harmonic_providers(
    raw_harmonic_params,
    n_layers=n_layers,
    force_constant_units="eV/angstrom^2",
    mass_units="amu",
    scale=scale,
)
problem = build_transverse_problem(
    grid=grid,
    vertex=vertex,
    device=device,
    left=left_contact_callback,
    right=right_contact_callback,
    input_units="solver",
    scale=scale,
)
```

Each contact callback accepts the two radian coordinates `kpar` and returns
a `KeldyshSpectrum` with shape `(Nomega, ndof, ndof)`. To use the gapped
helper, construct a `HarmonicReservoir` from `LeadBlocks(*lead.blocks(kpar))`
and call
`gapped_contact_spectrum(grid, reservoir, ndof, eta_lead, side=side)`.
The reservoir temperature and `eta_lead` are in solver units.
An explicit `device_to_lead` matrix and `device_indices` can describe a
nonmatched boundary. The default periodic coupling applies only when that
matching assumption is appropriate.

## Finite-device terms and Fourier convention

`TransverseCubicTerm(indices, second_cell, third_cell, coefficient, weight)`
represents one real ordered contribution

\[
\Psi_{0a,sc,td},
\qquad
\texttt{indices}=(a,c,d),\quad
\texttt{second\_cell}=s,\quad
\texttt{third\_cell}=t.
\]

The three DOF indices refer to the complete finite device in one transverse
cell. They therefore include transport-layer, atom, and Cartesian-coordinate
ordering. The two cell offsets are integer transverse translations.
Transport-layer separations belong to the DOF indices and are never
Fourier transformed.

`fourier_transverse_cubic` constructs the reduced vertex

\[
V_{acd}(q,p)=
\sum_{\text{terms with }(a,c,d)}
w\,c\,e^{i[p\cdot s+(q-p)\cdot t]}.
\]

Here \(c\) is the stored coefficient and \(w\) the stored term weight.
The Fourier convention is cell-periodic:
\(u_{Ra}=N_q^{-1/2}\sum_qe^{iq\cdot R}u_{qa}\).
No atomic-position phase is implicit. The returned vertex carries no
\(1/N_q\), \(1/\sqrt{N_q}\), or additional \(1/3!\) factor. The existing
Born kernel inserts its \(1/N_q\) momentum weight and cubic bubble
prefactor separately.

Weights can represent already-declared image or anchor multiplicities;
the mapper applies each stored weight once. Distinct contributions to the
same output entry add. Distinct translations that coincide modulo the
sampled mesh also add; they are not overwritten or treated as duplicates.
Such folding is the finite-mesh representation and does not establish
convergence with transverse sampling.

The input is a complete ordered schedule, rather than a list of unordered
triples that the mapper expands automatically. Internal-leg exchange sends
\((a,c,d;s,t)\) to \((a,d,c;t,s)\); exchanging the first two legs sends it
to \((c,a,d;-s,t-s)\). The resulting `TransverseCubicVertex` checks the
required reduced reality and permutation identities without projecting,
averaging, or repairing the values.

## Units and mass normalization

`mass_normalize_cubic_terms` converts declared raw force constants and
masses to dimensionless solver coefficients. The physical conversion is

\[
c_{\rm solver}=
\frac{\sqrt{\hbar}}{\omega_0^{5/2}}
\frac{\Phi^{(3)}_{\rm SI}}{\sqrt{m_a m_c m_d}}.
\]

The masses in this expression are kilograms and refer to the DOF indices
of the finite-device schedule. A physical atom mass is repeated for each
of its Cartesian coordinates. Source FC3 and mass units must be supplied
explicitly. The existing FC3/SI conversion helpers and `SolverScale` define
the conversion. Stored image/anchor weights remain distinct from the
coefficient conversion so they are not applied twice.

Already mass-normalized, solver-scaled coefficients go directly to the
Fourier mapper. Applying the raw-force-constant conversion again would
change the Hamiltonian. Conversely, providing raw eV/angstrom-cubed values
as solver coefficients omits both mass and unit normalization.

For harmonic force constants the corresponding conversion is

\[
D_{ab,\rm solver}=
\frac{\Phi^{(2)}_{ab,\rm SI}}
{\omega_0^2\sqrt{m_a m_b}}.
\]

`MaterialKspaceParams` stores raw harmonic force-constant terms and atomic
masses. Its ordinary providers perform mass division but do not themselves
establish SI units or divide by a physical \(\omega_0^2\).
`material_harmonic_providers` supplies this explicit unit boundary and
returns converted device/lead providers. Its outputs already use solver
frequency-squared units and must not be mass-normalized again.

The harmonic adapter rejects inconsistent opposite-image blocks:

\[
\Phi_{00}(s)=\Phi_{00}(-s)^T,\qquad
\Phi_{01}(s)=\Phi_{10}(-s)^T.
\]

The second check applies when `fc10_terms` is supplied. Missing or extra
nonzero reverse support fails the check. When `fc10_terms` is omitted,
the existing provider's declared conjugate-reverse convention applies.
These checks matter because `Device1D` reconstructs reverse transport
couplings from the forward block; accepting an inconsistent supplied FC10
would silently change the finite device. Comparison tolerances scale with
the block magnitude, so small coefficients do not receive a fixed
absolute-error exemption. No inconsistent block is projected or replaced.

Optional `layer_masses` uses the declared source mass units and is converted
along with the lead masses. Optional onsite pinning uses the declared raw
FC2 units. If the finite device has different boundary masses from the lead,
the caller must provide the corresponding device-to-lead coupling rather
than assume the default matched contact.

All harmonic, cubic, temperature, contact, and current conversions must use
the same `SolverScale`. `build_transverse_problem` requires the explicit
`input_units="solver"` declaration for supplied provider outputs; it does
not infer their units from numerical magnitudes.

## Harmonic sampling and contacts

The prototype mesh has shape `(ny, nz, 1)`. Its first two addresses label
the physical transverse directions used by the harmonic providers.
`DeviceKSpace.device(kpar)` and `LeadKSpace.blocks(kpar)` receive

\[
\texttt{kpar}=(2\pi q_y,\;2\pi q_z),
\]

where \(q_y,q_z\) are fractional reciprocal coordinates. The provider API
uses radians, not fractional coordinates. Its harmonic convention
\(\sum_s\Phi_{0s}e^{iq\cdot s}\) matches the cell-periodic cubic convention.
Every sampled `Device1D` retains the same finite transport-layer ordering.
Its block-tridiagonal matrices are assembled into the complete \(D(q)\)
accepted by `TransverseSCBAProblem`.

`build_transverse_problem` accepts left and right contact callbacks that
return complete signed `KeldyshSpectrum` objects at the sampled q. It
assembles those into `TransverseSpectrum` fields and invokes the existing
problem validation. Contact callbacks must use the same device basis,
normalization, and solver scale as the harmonic and cubic inputs.

For explicitly gapped harmonic leads, `gapped_contact_spectrum` reuses
the existing retarded surface-contact calculation across the full signed
frequency grid and the declared gapped zero-frequency Keldysh convention.
The existing left/right surface orientation and device-to-lead coupling
conventions apply. Complex q blocks must not be constructed by reflecting
positive frequencies within the same q; physical reflection pairs q
with \(-q\).

The gapped helper does not supply an acoustic or Ohmic Bose-times-broadening
limit. The existing audited Nq=1 acoustic descriptors in `contacts.py` and
`acoustic_contacts.py` provide a basis for later complex paired-q
extension and verification. Such reservoirs require a separately declared
callback implementing their zero-frequency prescription. The general
callback path checks the supplied joint identities and correlation
positivity; passing those checks alone does not certify an arbitrary
callback as a passive, thermal, or causal reservoir model.

The default Fourier guards allow 100,000 input cubic contributions,
1,000,000 phase evaluations, and a 64 MiB dense reduced-vertex payload.
The joined problem and gapped contact helper also check their spectrum
payloads before construction. These limits define a bounded prototype;
they do not estimate the entire process memory or establish production
scaling.

## Boundary with real material IFCs

The existing `TranslationalFC3` uses phono3py's bulk, atomic-position
convention. Its phase contains
\((q_0+q_1+q_2)\cdot\tau_a\) and the two shortest-vector offsets.
These offsets include basis positions and are not simply the integer
transverse shifts required here. Its full three-dimensional Fourier
transform also sums transport offsets. Setting the longitudinal momentum
to zero would still collapse those offsets; it would not create a
finite-device interaction.

The separate implementation in `transverse_material.py` now retains the
explicit audited source terms, maps each leg to a finite transport-layer
DOF and an integer transverse cell, preserves image and optional
anchor-average weights, and enforces a declared all-leg interaction crop.
It joins the FC2 and FC3 primitive identities and unit conventions, and
regenerates the FC2 image phases from the source geometry. Its 19 unit
tests and four independent raw-image reference cases pass; see the
[crystal-to-device mapping guide](research-transverse-material-mapping.md).
The old Nq=1 mapped harmonic and cubic objects have already collapsed
transverse winding and cannot reconstruct that information.

Extension and verification of the existing acoustic-contact path for
complex paired q, realistic dataset validation, boundary-model convergence,
and material transport remain explicit later work. Neither adapter
silently infers missing terms or adds symmetrization or acoustic-sum-rule
corrections.

## Validation

The final combined run passed **26 tests in 11.11 seconds**:
21 unit tests and five independent DFT/SCBA reference cases, with no
failures, errors, or skipped tests. Ruff also passed.

The independent comparisons in
[`test_anharmonic_transverse_mapping_reference.py`](../tests/test_anharmonic_transverse_mapping_reference.py)
cover:

- One-by-one, four-by-one, and two-by-two transverse meshes. Mapped cubic
  vertices, harmonic matrices, and all three signed contact components
  agree with direct Fourier transforms of the complete real-space model.
- Genuinely complex four-q harmonic, cubic, and contact-coupling phases.
- A converged four-q SCBA result agreeing with the dense-supercell fields,
  Green functions, and current per cell.
- Open transport endpoints retained explicitly while transverse
  directions are sampled periodically.

The unit tests in
[`test_anharmonic_transverse_mapping.py`](../tests/test_anharmonic_transverse_mapping.py)
check raw FC3/SI/mass/solver conversion with weights applied once; harmonic
unit and layer-mass conversion; detachment from source storage; rejection
of inconsistent explicit FC10 at ordinary and tiny coefficient scales;
missing nonzero FC00 reverse support; integer folding of large translations
and accumulation of aliased images; incomplete cubic permutations; work and
storage limits; acoustic-contact rejection; fixed geometry across q; and
explicit solver-unit declarations.

These are bounded synthetic mapping and solver checks. They establish no
material dataset validation or convergence claim.
