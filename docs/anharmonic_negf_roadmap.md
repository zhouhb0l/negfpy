# Anharmonic phonon NEGF: research, architecture, and validation plan

Status: locked release roadmap with a remotely certified dense
`v0.1.0-rc1`, 2026-07-29. The silicon bulk three-phonon oracle is the active
`0.2.0` gate. Exact conventions and current validation gaps are normative in
`anharmonic_theory.md` and `anharmonic_validation.md`.

## Research execution update — 2026-10-20

A bounded dense coupled-q Born reference is now implemented in
`transverse_q.py`. It retains the full q-prime sum, exact momentum folding,
complex cubic-vertex phases and the explicit `1/Nq` weight. Independent finite
supercell tests cover 1x1, 2x1, 4x1 and 2x2 meshes, including Nq=1 reduction,
cell-origin covariance and paired q/frequency bosonic identities. See
[the prototype conventions and API](research-transverse-q-prototype.md).

This is a synthetic contraction milestone. Coupled-q Dyson/Keldysh iteration,
retarded reconstruction, material FC3 Fourier mapping and material transport
are not implemented by it. The next bounded step is a coupled-q toy solver
with joint opposite-q field validation. Existing Nq=1 fields and scientific
acceptance remain separate; the prototype does not grant an Nq=1 or multi-q
material release.

## Historical research execution update — 2026-10-10

Centered g=.03 equilibrium and biased SCBA are accepted for the declared
research model. The completed eight-cut fixed-field diagnostic has a 0.011399%
current spread relative to the contact; it does not certify a release.
Lead-buffer construction and exact fixed-field mapping are now implemented;
material buffer and interacting-crop invariance remain open. See
[the current practical sequence](research-lead-buffer.md#current-practical-roadmap-2026-10-10).
The historical release roadmap and its required gates follow below.

## 1. Decision

Develop anharmonic transport as a subpackage:

```text
src/negfpy/
  core/                      # existing harmonic implementation
  anharmonic/
    types.py                 # solver scale and immutable public state
    grid.py                  # exact signed frequency grids
    fc3.py                   # raw/normalized full or compact FC3 data
    contacts.py              # contact r, <, > self-energies
    cubic.py                 # cubic Born/SCBA self-energy
    retarded.py              # spectral and Hilbert-transform reconstruction
    solver.py                # one-shot Born and SCBA drivers
    current.py               # contact and bond heat currents
    diagnostics.py           # identities, causality, KMS, and conservation
    io/
      phono3py.py
    toy/
      cubic_chain.py
      three_mode.py
      fgr.py
      open_three_mode.py          # independent eta-free continuum oracle
      multidimensional_release.py # three-layer/nine-DOF SCBA gate
      production_gate_report.py   # combined strict-JSON certification
      refinement.py
      release_report.py
```

Do not put the implementation in one large file parallel to `negf.py`.
Anharmonic NEGF adds several independent concepts and will outgrow one file
quickly. A subpackage isolates the new work while allowing reuse of the tested
harmonic types and surface Green-function solvers.

No anharmonic symbol should be re-exported from top-level `negfpy` until the
first validation gates in section 8 pass.

## 2. Scope of the first physically complete method

The reference implementation will treat:

- harmonic semi-infinite contacts;
- a finite interacting device;
- mass-normalized second- and third-order IFCs;
- cubic (three-phonon) scattering in full nonlocal self-consistent Born
  approximation (SCBA), retaining every output block and internal
  contraction permitted by the supplied cubic vertex;
- retarded, advanced, lesser, and greater Green functions;
- a symmetric signed-frequency grid;
- heat current from the interacting Meir-Wingreen-like expression;
- current conservation as a convergence condition.

The dense `0.1.x` solver accepts `Device1D` and has no transverse momentum
dimension. Transverse periodicity is a later solver because cubic scattering
couples momenta and cannot reuse independent harmonic q-point calculations.

The first version will not silently include:

- fourth-order IFCs or four-phonon scattering;
- thermal expansion or a cubic tadpole correction;
- anharmonic contacts;
- a local-diagonal self-energy approximation;
- a quasiparticle or relaxation-time approximation;
- a phenomenological Büttiker probe.

Those are separate, named models. They must never be selected by an ambiguous
flag such as `fast=True`.

Paper-specific slab/block-local and atom-diagonal SCBA are also separate,
named approximation levels. They may reproduce a historical calculation or
provide a diagnostic, but neither is silently substituted for the full
nonlocal reference.

## 3. Conventions that must be frozen before coding

### 3.1 Coordinates and IFC normalization

Use mass-weighted displacement

\[
u_{i\alpha}=\sqrt{M_i}\,\Delta r_{i\alpha}.
\]

The harmonic dynamical matrix and normalized cubic vertex are

\[
D_{i\alpha,j\beta}
=\frac{\Phi^{(2)}_{i\alpha,j\beta}}{\sqrt{M_iM_j}},
\qquad
\Psi_{i\alpha,j\beta,k\gamma}
=\frac{\Phi^{(3)}_{i\alpha,j\beta,k\gamma}}
{\sqrt{M_iM_jM_k}}.
\]

`FC3Data` must say whether its values are raw \(\Phi^{(3)}\) or normalized
\(\Psi\). Internal kernels accept only normalized values. Unit conversion is
performed once at the input boundary and recorded in metadata.

The cubic Hamiltonian convention is

\[
H_3=\frac{1}{3!}\sum_{abc}\Psi_{abc}u_au_bu_c.
\]

The factor \(1/3!\) is part of the public convention. This matters because
published self-energy prefactors differ when the vertex definition differs.

### 3.2 Green functions

Use one documented displacement Green-function convention throughout. In the
Guo et al. convention used by the two attached papers,

\[
G^r(\omega)=
\left[(\omega+i0^+)^2I-D-\Sigma_L^r-\Sigma_R^r-\Sigma_{pp}^r\right]^{-1}.
\]

The code must test, rather than assume, the identities

\[
G^a=(G^r)^\dagger,\qquad
G^r-G^a=G^>-G^<,
\]

with the corresponding self-energy identity. Lesser and greater signs depend
on the selected Green-function definition, so formulas must be derived from
that definition and encoded in one place.

### 3.3 Signed frequency grid

SCBA needs a grid on \([-\omega_{\max},+\omega_{\max}]\), not the positive-only
grid used for harmonic Landauer integration. The cubic self-energy contains

\[
\int_{-\infty}^{+\infty}\frac{d\omega'}{2\pi}
G^{\gtrless}(\omega')G^{\gtrless}(\omega-\omega').
\]

The grid must:

- contain zero exactly;
- have uniform spacing in the reference implementation;
- map every index to its negative-frequency partner exactly;
- define zero padding rather than circular wraparound for direct/FFT
  convolution;
- include an explicit tail/cutoff convergence test;
- evaluate the Bose function at negative frequency through exact bosonic
  identities, without numerically unstable subtraction.

### 3.4 Cubic scattering self-energy

For the \(1/3!\) vertex convention, use Eq. (8) of Guo et al.,
Phys. Rev. B 102, 195412 (2020), as the initial reference:

\[
\Sigma_{pp,ab}^{\gtrless}(\omega)
=\frac{i\hbar}{2}
\sum_{cdef}\Psi_{acd}\Psi_{bef}
\int\frac{d\omega'}{2\pi}
G_{ce}^{\gtrless}(\omega')
G_{df}^{\gtrless}(\omega-\omega').
\]

Indices above are schematic; the implementation must retain the exact atom,
Cartesian, layer, and transverse-cell mappings.

This equation defines NEGFPy's **full nonlocal SCBA** reference. A restriction
of the output self-energy to diagonal layer/slab blocks, or to diagonal
\(3\times3\) atom blocks, changes the approximation and must be recorded in
the method name and result provenance.

That distinction is required for a faithful reading of the attached papers.
The production silicon calculation in
[Guo et al., Phys. Rev. B 102, 195412
(2020)](https://doi.org/10.1103/PhysRevB.102.195412) retained diagonal slab
blocks of the anharmonic self-energy and selected dominant terms within each
retained block. It separately tested the more restrictive atom-diagonal
approximation, which overestimated low-frequency scattering and lowered
conductance. A paper reproduction must therefore reproduce the slab/block
and retained-FC3 choices by name before comparing that approximation with the
full nonlocal reference.

[Guo et al., Phys. Rev. B 103, 174306
(2021)](https://doi.org/10.1103/PhysRevB.103.174306) explicitly decomposed
spectral exchange into local diagonal and nonlocal off-diagonal scattering-
matrix contributions and found both important at low frequencies. This
supports retaining off-layer blocks and diagnosing their exchange. It does
not derive the bare Hamiltonian interaction-current operator required when an
FPU-\(\alpha\) cubic vertex itself crosses a device cut.

The retarded self-energy is reconstructed as

\[
\Sigma_{pp}^r(\omega)
=\frac{1}{2}\left[\Sigma_{pp}^>(\omega)-\Sigma_{pp}^<(\omega)\right]
+i\,\mathcal P\int\frac{d\omega'}{2\pi}
\frac{\Sigma_{pp}^>(\omega')-\Sigma_{pp}^<(\omega')}
{\omega-\omega'}.
\]

Two explicit strategies are allowed:

- `spectral_only`: omit the principal-value term and therefore omit the
  anharmonic frequency shift;
- `causal_hilbert`: evaluate the principal-value term with a verified discrete
  Hilbert transform.

The selected strategy must be written to every result file. `spectral_only`
cannot be described as a calculation of frequency renormalization.

### 3.5 Transverse periodicity

For 2D and 3D materials, retain real space along transport and Fourier
transform only the transverse directions. The cubic self-energy couples
\((\omega,\mathbf q_\perp)\) to
\((\omega',\mathbf q'_\perp)\) and
\((\omega-\omega',\mathbf q_\perp-\mathbf q'_\perp)\).

Required tests:

- exact folding of \(\mathbf q_\perp-\mathbf q'_\perp\) to the sampled grid;
- normal and Umklapp momentum conservation;
- invariance under a transverse-cell origin shift;
- equality of real-space and reciprocal-space contractions on a small cell;
- correct \(1/N_q\) normalization.

## 4. Solver algorithms

### 4.1 Dense correctness solver

Implement this first, even though it will be slow.

1. Build harmonic contact \(\Sigma_\alpha^r\) on the signed frequency grid.
2. Build equilibrium contact \(\Sigma_\alpha^{<,>}\) at \(T_\alpha\).
3. Initialize \(\Sigma_{pp}^{r,<,>}=0\).
4. Solve Dyson for \(G^r\).
5. Solve Keldysh:
   \[
   G^{<,>}=G^r\Sigma^{<,>}G^a.
   \]
6. Contract the cubic vertices and \(G^{<,>}\) to update
   \(\Sigma_{pp}^{<,>}\).
7. Reconstruct \(\Sigma_{pp}^r\).
8. Apply controlled linear mixing to the lesser and greater components
   together, then reconstruct the retarded component.
9. Repeat until both algebraic residuals and heat-current conservation pass.

Store a full iteration history. A small change in self-energy norm is not a
sufficient convergence condition.

### 4.2 SCBA convergence criteria

All of the following should pass:

- relative change of \(\Sigma_{pp}^{r,<,>}\);
- relative change of \(G^{r,<,>}\);
- \(G^a=(G^r)^\dagger\);
- Keldysh spectral identity;
- positive semidefiniteness of every Hermitian-projected
  \(iC^{</>}\) correlation matrix for Green functions and self-energies in
  every equilibrium and nonequilibrium calculation;
- passivity of each harmonic contact broadening in every calculation;
- Green, total-self-energy, and interaction-self-energy spectral passivity
  only at global equilibrium or zero cubic interaction. In an interacting
  nonequilibrium steady state these quantities are descriptive, not gates:
  bosonic spectral positivity relies on monotonically ordered occupations,
  for which there is no general NESS theorem
  ([Uhrig et al., Phys. Rev. Lett. 122, 130604
  (2019)](https://doi.org/10.1103/PhysRevLett.122.130604);
  [arXiv:1808.10199](https://arxiv.org/abs/1808.10199));
- causality of \(G^r\) and stability of the total embedded retarded problem;
- \(J_L+J_R=0\);
- raw harmonic bond current is constant only when the converged
  \(\Sigma_{pp}\) is layer-block-local at the fixed structural threshold
  \(r_{\rm nl}\le10^{-12}\); full-nonlocal \(\Sigma_{pp}\) instead gates the
  projected EOM continuity and corrected cut currents;
- at equilibrium, \(J_L=J_R=0\);
- the integrated collision power is zero:
  \[
  \int_0^\infty \frac{d\omega}{2\pi}\hbar\omega
  \operatorname{Tr}\!\left[
  \Sigma_{pp}^>G^<-\Sigma_{pp}^<G^>
  \right]=0.
  \]

Use both an absolute current tolerance and a relative tolerance. Relative
error alone is ill-defined near equilibrium.

The projected EOM-exchange sum and the independently layer-projected
greater/lesser collision-power sum must each recover global \(P_{pp}\).
Individual layer contributions may be nonzero and sign-changing because
anharmonic scattering redistributes energy spatially. Only their global sum
and the continuity identities are gated; independent layerwise zeros are not
required.

### 4.3 Current

Caroli transmission is not the general current formula once
\(\Sigma_{pp}^{<,>}\ne0\). Compute contact current as

\[
J_\alpha=
\int_0^\infty\frac{d\omega}{2\pi}\hbar\omega\,
\operatorname{Tr}\left[
\Sigma_\alpha^>G^<-\Sigma_\alpha^<G^>
\right],
\]

with a sign convention fixed by a harmonic-limit test. Also implement the
harmonic bond-current expression using the first off-diagonal lesser
Green-function blocks. For an onsite or intralayer bare vertex, the raw
harmonic bond current is a certification gate only if the converged
self-energy is layer-block-local. Full nonlocal SCBA can generate off-layer
self-energy blocks even from such a bare vertex. In that case the spatial
gate adds the explicitly named
`nonlocal_self_energy_eom_cut_correction`, obtained from the layer-projected
equation of motion, to the raw harmonic current and compares the corrected
cut currents with the contacts.

This correction accounts for exchange carried by a nonlocal self-energy; it
is not a derived bare anharmonic current operator. FPU-\(\alpha\) bond
interactions have a cubic vertex that itself crosses the cut and still
require a separate Hamiltonian interaction-current operator. Until that
operator is derived, those fixtures are validated with contact currents and
global collision power only and have no spatial-current certification gate.

## 5. Approximation hierarchy

| Method | Role | Advantages | Main risk |
|---|---|---|---|
| Full dense cubic SCBA | Reference implementation | Conserving when implemented consistently; handles nonequilibrium broadening | Highest cost |
| Block-sparse SCBA from FC3 support | Production target | Preserves nonlocal terms that actually exist | More complex data movement |
| Guo-2020 slab/block-local SCBA | Named paper-reproduction level | Reproduces the production truncation when its retained blocks and FC3 support are matched | Omits off-slab self-energy exchange; cannot certify the full nonlocal method |
| Atom-diagonal SCBA | Diagnostic paper approximation | Lowest memory among SCBA variants | Can overestimate low-frequency scattering and lower conductance |
| One-shot Born | Debugging and weak-coupling baseline | Simple; self-energy scales as \(\Psi^2\) | Generally not current conserving |
| Lowest-order approximation / Padé rescaling | Later acceleration | Published speedup relative to SCBA | Must be calibrated against SCBA |
| Quasiparticle/on-shell self-energy | Bulk validation and large systems | Direct comparison to BTE/phono3py | Loses device coherence and strong nonequilibrium |
| Self-consistent phonon/effective harmonic model | Frequency renormalization | Useful for strong quartic anharmonicity | Does not by itself describe inelastic current |
| Büttiker probes | Optional engineering model | Very cheap | Phenomenological unless independently calibrated |

The production sequence should be full dense SCBA, then block sparsity, then
optional acceleration. The slab/block-local and atom-diagonal branches remain
explicit comparison methods; neither becomes a silent performance switch.

## 6. Third-order IFC data model

### 6.1 Staged representations

The `0.2.0` ingestion boundary first preserves phono3py's raw full or compact
array exactly:

```text
FC3Data
  values[Ns,Ns,Ns,3,3,3]       # full
  values[Np,Ns,Ns,3,3,3]       # compact
  masses[Ns]
  p2s_map, s2p_map
  units, mass_units
  provenance
```

Reading, validation, physical-unit conversion, and mass normalization are
separate non-mutating operations. Compact input is not expanded merely to
run an ingestion audit.

The later sparse device compiler uses a translationally reduced
representation:

```text
FC3Term / SparseCubicVertex
  masses[n_atom]
  lattice[3,3]
  positions[n_atom,3]
  triplets[n_term]:
    atom_i, atom_j, atom_k
    R_j[3], R_k[3]
    tensor[3,3,3]
  units
  mass_normalized
  metadata
```

The first atom is in the home cell. This sparse object is not silently
substituted for raw `FC3Data`. Internal SCBA kernels accept only
mass-normalized solver-unit vertices, and the complete FC3-implied support is
retained.

### 6.2 Readers

Implement readers in this order:

1. `phono3py fc3.hdf5`, whose documented full array shape is
   `(n_satom, n_satom, n_satom, 3, 3, 3)` and whose default unit is
   eV/Angstrom^3;
2. ShengBTE `FORCE_CONSTANTS_3RD`, a sparse home-cell representation;
3. a small JSON/NPZ format used only by unit tests.

### 6.3 Mandatory IFC3 validation

- finite values and positive masses;
- permutation symmetry of the three displacement derivatives;
- translational sum rules for each fixed pair of remaining indices;
- compatible primitive/supercell maps;
- no double counting of periodic images;
- cutoff report by neighbor shell and by norm;
- unit metadata and one round-trip unit test;
- optional rotational-invariance report;
- comparison of a sampled three-phonon matrix element with phono3py.

Sum-rule correction must be separate from validation. Never silently modify
input IFCs.

## 7. Efficiency plan

The naive cost contains frequency and transverse-momentum convolutions plus
FC3 tensor contractions. Optimize only after the dense solver is validated.

1. Exploit the sparse support of FC3.
2. Restrict \(\Sigma_{pp}\) to the union of blocks implied by that support,
   not to atomic diagonal blocks.
3. Use recursive Green functions to retain only diagonal and first
   off-diagonal blocks needed by current and local observables.
4. Batch BLAS contractions over frequency.
5. The optional zero-padded FFT cubic and finite-grid Hilbert backends are
   now activated after direct/FFT equality, no-wrap, and \(>2\times\)
   performance gates passed. Direct remains the default correctness oracle.
6. Parallelize independent external \((\omega,\mathbf q_\perp)\) points while
   explicitly accounting for the data exchange required by internal
   \((\omega',\mathbf q'_\perp)\) sums.
7. Use the private atomic HDF5 checkpoint path after each completed campaign
   outcome. Store the mixed \(\Sigma_{pp}^{</>}\), histories, timings, and
   hashes, then recompute \(G\), currents, and diagnostics after resume. Reject
   source, environment, grid, or input mismatches.
8. Measure memory and FLOPs from actual FC3 sparsity before selecting CPU,
   GPU, or distributed layouts.

Mixed precision is not an initial optimization because small current
nonconservation errors can hide a wrong self-energy.

## 8. Validation ladder and release gates

The gate order is production-first:

1. harmonic and algebraic gates H/A;
2. the continuum/open-system linewidth-and-shift oracle;
3. the full three-layer, three-polarization conserving SCBA gate;
4. the pinned phono3py silicon bulk oracle;
5. sparse, transverse, and attached-paper material-device benchmarks.

Historical paper reconstructions are retained as named regression evidence,
but they do not displace or block this sequence. In particular, agreement
obtained by fitting a historical normalization or diagrammatic prefactor is
not a validation result.

### Gate H: harmonic baseline

- analytic surface Green function of a semi-infinite monoatomic chain;
- exact unit transmission for a matched chain;
- analytic single-mass-defect transmission;
- channel count for independent multi-DOF chains;
- zero transmission outside the propagating band;
- IFC dispersion against the source phonopy/QE calculation on random
  wavevectors;
- fixed/global eta convergence with no discarded k-point weight.

### Gate A: algebra and equilibrium

- direct tensor contraction against a scalar/brute-force oracle;
- convolution of known analytic arrays;
- self-energy scales as the square of the cubic coupling;
- permutation of identical FC3 legs leaves the answer unchanged;
- equilibrium KMS/fluctuation-dissipation identities;
- positive semidefiniteness of Hermitian-projected \(iC^{</>}\) for the
  lesser/greater correlation matrices;
- harmonic-contact passivity in every calculation, plus Green, total, and
  interaction spectral passivity only at equilibrium or zero interaction.
  All three latter quantities are descriptive in interacting NESS and are
  never clipped;
- exactly zero equilibrium current within the integration tolerance;
- harmonic limit recovered as \(\Psi\to0\).

### Gate B: nonequilibrium toy junction

- one-site and short-chain interacting devices;
- a continuum or explicitly open-system mode-space oracle that compares both
  \(-\operatorname{Im}\Sigma^r(\Omega)/\Omega\) and
  \(\operatorname{Re}\Sigma^r(\Omega)/(2\Omega)\) with independent
  FGR/principal-value integrals while refining physical bath width,
  `eta_lead`, spacing, and cutoff separately;
- a release-grade three-layer, three-polarization intralayer cubic problem,
  with genuinely mixed polarizations and identical zero-start and
  Born-seeded SCBA fixed points;
- current from both contacts agrees;
- raw harmonic bond current is spatially constant when \(\Sigma_{pp}\) is
  block-local; full-nonlocal onsite/intralayer cases instead require the
  projected EOM continuity and
  `nonlocal_self_energy_eom_cut_correction` gates;
- convergence under frequency spacing, cutoff, eta, mixing, and device size;
- weak-coupling comparison to an independently implemented master-equation
  or exact-diagonalization calculation where their assumptions overlap.

The resolved protocol-v2 base point is exactly

\[
\Delta\omega=0.00078125,\quad \omega_{\rm cut}=5,\quad
\eta_{\rm lead}=0.0015625,\quad m=0.2.
\]

Its locked ladders are

\[
\Delta\omega=(0.0015625,0.00078125,0.000390625),
\qquad
\eta_{\rm lead}=(0.003125,0.0015625,0.00078125),
\]

with cutoffs \((5,6)\) and mixings \((0.2,0.1)\). The seven-point harmonic
preflight is the base plus six one-control-at-a-time points: coarse spacing,
half spacing, larger cutoff, doubled eta, half eta, and changed mixing. All
six are executed and archived. The four certification directions are half
spacing \(0.000390625\), cutoff 6, half eta \(0.00078125\), and mixing 0.1;
coarse spacing \(0.0015625\) and doubled eta \(0.003125\) are diagnostic only.

After preflight, the production graph has exactly 13 outcomes:

1. one independently launched zero-start hardest endpoint
   \((g,\Delta T)=(0.05,0.2)\);
2. the five-point L-shaped continuation
   \((0,0)\rightarrow(0.025,0)\rightarrow(0.05,0)
   \rightarrow(0.05,0.1)\rightarrow(0.05,0.2)\);
3. one independently Born-seeded endpoint; and
4. six independent zero-start refinement outcomes.

Every full production outcome uses exact zero-padded FFT convolution, the FFT
implementation of the same finite-grid principal-value Hilbert sum, and
analytic diagonal contacts. Zero/Born starts and continuation endpoints use
`rtol=1e-8`, `atol=1e-12` for every
\(G^{r,<,>}\) and \(\Sigma_{pp}^{r,<,>}\) fixed-point field. Their
route-independent transport observable is
\(J_{\rm tr}=(J_L-J_R)/2\), not a vector containing \(P_{pp}\).
Collision power, contact balance, and full balance are evaluated separately
for every run; two accepted residuals are not cross-compared as transport
current. Backend equivalence is a separate, bounded, converged SCBA oracle
comparing serial/direct, threaded/direct, and FFT/FFT full fields and the
full current/conservation vector at `rtol=1e-10`, `atol=1e-12`. The protocol
makes no claim that direct or threaded SCBA was run on every resolved
production grid.

Refinement field applicability follows the quantity being changed.
\(\eta_{\rm lead}\) changes contact broadening, so its pointwise fields are
descriptive and non-gating; the eta direction gates \(J_{\rm tr}\), the
selected linewidth and shift, and the self-energy tail. Frequency-spacing
and cutoff changes use different grids and also have no pointwise field
gate. Mixing changes only the fixed-point iteration path, so the
changed-mixing direction additionally must reproduce all six fixed-point
fields at `rtol=1e-8`, `atol=1e-12`. These applicability rules do not alter
the preregistered physical controls or the \(0.5\%\) directional observable
threshold.

One-shot Born may screen these cases and seed continuation, but it is not a
fixed point and is generally not conserving. Only converged SCBA can certify
a nonzero-cubic interacting device.

### Gate C: nonblocking historical chain benchmarks

Retain, in increasing difficulty, the following historical comparisons:

1. the small-junction trends in Mingo, Phys. Rev. B 74, 125402 (2006);
2. the chain/junction examples in Wang et al., Phys. Rev. B 74, 033408
   (2006), and Phys. Rev. E 75, 061128 (2007);
3. the equilibrium FPU-beta and phi4 tests of Xu et al., Phys. Rev. B 78,
   224303 (2008).

The FPU-beta and phi4 models contain quartic interactions. They validate a
quartic mean-field/quasiparticle branch, not the cubic SCBA kernel by
themselves.

Mingo's printed surface formulas and cubic-input convention do not uniquely
determine its numerical calculation. Further source discovery is capped at
four developer-hours. If that search does not produce unique primary
evidence for both ambiguities, the benchmark remains frozen as
`historical_unresolved`; half spectral weight, a factor-six vertex, or any
other fitted multiplier is not an admissible closure.

Wang's harmonic, \(O(T^2)\), and stable-range selective-mean-field
checkpoints remain useful source-limited regressions. Porting and testing all
six \(O(T^4)\) diagrams is a separately named historical extension, not a
`0.1.0` production-release blocker.

### Gate D: bulk real-material oracle

Before a device calculation, take the same FC2/FC3 and compare the
weak-coupling equilibrium, on-shell linewidths and frequency shifts against
phono3py or ShengBTE on identical q meshes, smearing, isotope settings, and
sum-rule treatment. This is the most useful bridge between a toy model and
real-material SCBA.

Start with silicon. Add graphene only after the flexural-mode sum rules and
long-wavelength convergence are under control.

### Gate E: attached-paper device benchmarks

Use [Guo et al., Phys. Rev. B 102, 195412
(2020)](https://doi.org/10.1103/PhysRevB.102.195412):

- cross-plane silicon film;
- 300 K;
- thickness trend through approximately 10-13 nm;
- ballistic-to-anharmonic reduction;
- local temperature and spatial current conservation;
- sensitivity to the retained FC3 neighbor shells and self-energy blocks.

The paper reports about a 20% conductivity reduction near 10 nm and, for its
specific first-neighbor FC3 input, 7.4489 and 8.4025 W/m K for 10.8 and
13 nm films. These numbers are regression targets only when the DFT IFCs,
geometry, meshes, and approximations are matched.

The first reproduction uses the paper's named production level: retained
diagonal slab self-energy blocks and its selected dominant within-block/FC3
terms. The more restrictive atom-diagonal test is reported separately. Only
after that paper-specific benchmark is reproduced is full FC3 support and
full nonlocal SCBA compared. A match obtained at one approximation level is
not evidence that the other levels are equivalent.

Then use [Guo et al., Phys. Rev. B 103, 174306
(2021)](https://doi.org/10.1103/PhysRevB.103.174306):

- idealized Si/Ge mass-mismatch interface;
- scattering only in the interface device region;
- spectral generation/annihilation integrates to zero;
- local diagonal and nonlocal off-diagonal spectral exchange are archived
  separately, and their sum reproduces the full result;
- both contributions are retained when testing low-frequency redistribution;
- interfacial mode decay and inelastic spectral redistribution;
- finite-device and contact-resistance sensitivity.

## 9. Findings from the current harmonic audit

### Verified now

- The maintained test suite passes after excluding two duplicate
  “conflicted copy” test files that pytest otherwise collects.
- The retarded device equation has the expected
  \((\omega+i\eta)^2I-D-\Sigma_L-\Sigma_R\) form.
- Contact broadening and Caroli transmission use
  \(\Gamma=i(\Sigma-\Sigma^\dagger)\).
- The general multi-atom builders use
  \(D=M^{-1/2}\Phi M^{-1/2}\).
- The Landauer heat-current implementation uses angular frequency and the
  required \(1/(2\pi)\) factor.
- A mass-defect toy-builder normalization error was corrected and is now
  checked against the exact 1D scattering formula.
- `eta_device=None` resolves to zero independently of the finite lead eta, so
  the surface-GF regulator is not applied as an unaccounted bath throughout
  the device.
- The phonopy and QE reader metadata label the internal frequency unit
  consistently as `sqrt(Ry/(2*me*bohr^2))`.
- The IFC audit reports reciprocal-pair, translational and rotational
  sum-rule, sampled q-space Hermiticity, mapping, and stability checks
  without mutating input.

### Remaining harmonic risks before publication-quality IFC claims

1. Supercell aliasing and comparisons against source dynamical matrices
   remain open publication gates beyond the current IFC audit.
2. The legacy per-k-point adaptive eta scheme may use a different eta for
   each k point and average only successful points. This can bias a Brillouin
   zone average. Production runs should use a fixed eta or one global eta per
   frequency, require effectively complete k-point weight, and report all
   failures.
3. Real IFC readers still need random-q dynamical-matrix comparisons against
   the source code and reference data. Passing synthetic tests alone is not a
   complete validation of a material result.
4. Endpoint mass changes require explicit mass-normalized device-to-lead
   coupling; the automatic matched-contact default is only correct for
   matched endpoint masses.

## 10. Locked release sequence and current state

### Protocol-v2 checkpoint and first-attempt evidence

Protocol v2 encodes the resolved seven-point harmonic preflight, bounded
converged backend oracle, 13-outcome FFT graph, state-conditional spectral
passivity, full-nonlocal spatial-current bookkeeping, weak-coupling gates,
clean common provenance, and private atomic HDF5 outcome resume. The complete
graph subsequently passed on the remote workstation at exact commit
`ad2558961f42e58c15b54a7b2ebdf7219dd6d363`.

The first remote attempt instead used commit
`d05b4404af98e650e712b4ed7afd68705c1014f2`, the superseded protocol/schema,
and a serial 401-point grid with \(\Delta\omega=0.025\), cutoff 5, and
\(\eta_{\rm lead}=0.0125\). On 2026-07-25 it reached 200 iterations in
2481.593 s. The terminal self-energy and Green residuals were
\(4.616316\times10^{-16}\) and \(1.44047\times10^{-17}\); the contact
imbalance was \(3.46945\times10^{-18}\), and
\(P_{pp}=7.05099\times10^{-19}\). Its currents were
\(J_L=0.01757079082964086\) and \(J_R=-0.017570790829640855\).

That result is a provisional failure. The raw harmonic bond currents
\([0.017558902896426334,0.017559378234576487]\) had spread
\(4.75338\times10^{-7}\) and maximum contact mismatch
\(1.188793\times10^{-5}\), exceeding the \(1.757079\times10^{-7}\)
tolerance. Its Green spectral-moment error 0.0704865 also genuinely exceeded
0.005. The old report additionally rejected a nonequilibrium
\(\Gamma_{pp}\) passivity metric of \(5.992684\times10^{-6}\) against
\(10^{-10}\). That state-conditional gate was invalid. Protocol v2 retains
unconditional correlation positivity and contact passivity, and gates Green,
total, and interaction spectral passivity only at equilibrium or zero
interaction.

A postmortem found \(r_{\rm nl}\simeq0.46\), and the projected
EOM exchange closes the archived raw bond/contact mismatch exactly. This
supports the diagnosis, but cannot substitute for a clean rerun. All other
old-graph outcomes were archived as blocked, so there is no old-protocol
continuation, backend, or refinement result to promote.

A later clean protocol-v2 endpoint attempt used immutable commit
`ab467f6abdf9ca9b7d7d13f67a2c966c95f788a0` and the 12,801-point base
grid. It converged in 93 iterations, with terminal self-energy and Green
residuals \(5.626106908840702\times10^{-9}\) and
\(1.883264937980352\times10^{-10}\). The exact archived currents were
\(J_L=0.018310314275714244\) and
\(J_R=-0.018310314266458405\); the contact imbalance was
\(9.25583915067918\times10^{-12}\),
\(P_{pp}=-9.255838633342108\times10^{-12}\), and the full balance was
\(5.173370710020005\times10^{-19}\). The interaction-self-energy tail was
\(5.493998998582065\times10^{-10}\). At the selected parent mode the
linewidth and shift were \(0.012015829368999151\) and
\(0.0014191692116250685\), with weak-coupling ratios
\(0.008010552912666101\) and \(0.0009461128077500456\), respectively.

That endpoint is still a **provisional failed diagnostic**, not a certified
result under its frozen revision. Its report incorrectly gated Green and
total spectral passivity in interacting NESS and therefore stopped the
remaining graph. A no-clipping postmortem located every worst descriptive
value at the first nonzero bin, \(|\omega|=0.00078125\), deep in the pinned
gap: Green scale \(280.5143445104\), minimum eigenvalue
\(-2.362492787\times10^{-6}\), normalized violation
\(8.422003485\times10^{-9}\); total-self-energy scale
\(0.250440241545\), minimum eigenvalue
\(-2.933080306\times10^{-6}\), normalized violation
\(1.171169732\times10^{-5}\); and interaction-self-energy scale
\(0.118031155016\), minimum eigenvalue
\(-2.939018084\times10^{-6}\), normalized violation
\(2.490035858\times10^{-5}\). All lesser/greater correlation-positivity
violations were zero. These postmortem numbers are descriptive evidence,
not extra fields embedded retrospectively in the raw report, and no spectrum
was clipped or projected. A new immutable candidate must rerun the complete
graph under the corrected contract; the old endpoint cannot be
retroactively certified.

The later exact campaign supersedes those provisional failures. Its strict
report SHA256 is
`793ad0fa014f9322c73e1ebb84cb3327400e925a14228f3dbe0a0af075e38972`
and declares `release_certified=true`. It contains 28 completed outcomes,
the complete 13-case multidimensional graph, all 13 open-system cases, 13
scientific checkpoints, and no scientific failure. Python 3.10 and 3.12
wheel qualifications, an independent post-fetch audit, and the regenerated
PDF are archived under
`reports/phonon-phonon-interaction/validated/ad255896.../attempt-001/`.
The annotated tag `v0.1.0-rc1` identifies this passed toy release.

| Release | Required outcome | Current state |
|---|---|---|
| `0.1.0` | Python-only dense cubic SCBA for validated toy models | **Passed.** Exact commit `ad255896...` is tagged `v0.1.0-rc1`; the strict remote report declares `release_certified=true`. Mingo and Wang \(O(T^4)\) remain nonblocking historical extensions and do not calibrate the production prefactor. |
| `0.2.0` | phono3py FC3 import and bulk three-phonon linewidth validation | In progress. Full/compact FC2/FC3 ingestion, immutable crystal mapping, translational FC3, deterministic mode-space interactions and linewidths, exact 3x3x3 scientific agreement, task-level target-\(q\) MPI, independent phono3py MPI reference generation, bounded \(V^{(3)}\) sampling, and strict evidence aggregation are implemented. Release remains blocked on exact-new-commit 3x3x3 and 5x5x5 serial/MPI equality plus the complete 11x11x11 external campaign. |
| `0.3.0` | Sparse real-space SCBA for quasi-1D material devices | Development implementation exists: complete-support `SparseCubicVertex`, authenticated silicon (N_q=1) mapping, atomic SCBA checkpoints, and bounded dense/sparse equality tests. Release remains blocked on the refined material fixed-point, conservation, refinement, and lead-buffer-invariance archive. See `docs/validation/sparse-nq1-phase-c.md`. |
| `0.4.0` | Transverse-momentum SCBA for 2D/3D geometries | Bounded coupled-q Born reference implemented and checked against small real-space supercells, including complex vertices, exact folding and \(1/N_q\) normalization. The coupled-q self-consistent solver and material mapping remain unimplemented; see `research-transverse-q-prototype.md`. |
| `1.0.0` | Silicon/interface paper benchmarks, restart, convergence archive, production CPU performance | The private atomic HDF5 mechanism is implemented, but release restart equivalence and the material/performance benchmarks are not. The attached Guo papers define later targets; they are not evidence for the dense toy candidate. |

Optional LOA/Padé, local self-energy approximations, quartic
renormalization, four-phonon scattering, thermal expansion, and anharmonic
reservoirs are post-reference branches. None may become a silent default.

## 11. Core literature

- G. S. Uhrig, M. H. Kalthoff, and J. K. Freericks, “Positivity of the
  Spectral Densities of Retarded Floquet Green Functions,”
  [Phys. Rev. Lett. 122, 130604
  (2019)](https://doi.org/10.1103/PhysRevLett.122.130604)
  ([arXiv:1808.10199](https://arxiv.org/abs/1808.10199)).
- N. Mingo, “Anharmonic phonon flow through molecular-sized junctions,”
  [Phys. Rev. B 74, 125402 (2006)](https://doi.org/10.1103/PhysRevB.74.125402).
- J.-S. Wang, J. Wang, and N. Zeng, “Nonequilibrium Green’s function
  approach to mesoscopic thermal transport,”
  [Phys. Rev. B 74, 033408 (2006)](https://doi.org/10.1103/PhysRevB.74.033408).
- J.-S. Wang et al., “Nonequilibrium Green’s function method for thermal
  transport in junctions,”
  [Phys. Rev. E 75, 061128 (2007)](https://doi.org/10.1103/PhysRevE.75.061128).
- Y. Xu et al., “Nonequilibrium Green’s function method for phonon-phonon
  interactions and ballistic-diffusive thermal transport,”
  [Phys. Rev. B 78, 224303 (2008)](https://doi.org/10.1103/PhysRevB.78.224303).
- M. Luisier, “Atomistic modeling of anharmonic phonon-phonon scattering in
  nanowires,”
  [Phys. Rev. B 86, 245407 (2012)](https://doi.org/10.1103/PhysRevB.86.245407).
- J.-S. Wang et al., “Nonequilibrium Green’s function method for quantum
  thermal transport,”
  [Front. Phys. 9, 673 (2014)](https://doi.org/10.1007/s11467-013-0340-x).
- Y. Lee et al., “Anharmonic phonon-phonon scattering modeling of
  three-dimensional atomistic transport: An efficient quantum treatment,”
  [Phys. Rev. B 97, 205447 (2018)](https://doi.org/10.1103/PhysRevB.97.205447).
- J. Dai and Z. Tian, “Rigorous formalism of anharmonic atomistic Green’s
  function for three-dimensional interfaces,”
  [Phys. Rev. B 101, 041301 (2020)](https://doi.org/10.1103/PhysRevB.101.041301).
- Y. Guo et al., “Quantum mechanical modeling of anharmonic phonon-phonon
  scattering in nanostructures,”
  [Phys. Rev. B 102, 195412 (2020)](https://doi.org/10.1103/PhysRevB.102.195412).
- Y. Guo et al., “Anharmonic phonon-phonon scattering at the interface
  between two solids by nonequilibrium Green’s function formalism,”
  [Phys. Rev. B 103, 174306 (2021)](https://doi.org/10.1103/PhysRevB.103.174306).
- Phono3py,
  [`fc3.hdf5` input/output documentation](https://phonopy.github.io/phono3py/input-output-files.html#fc3-hdf5).
