# Anharmonic phonon NEGF: research, architecture, and validation plan

Status: locked release roadmap with an executable dense `0.1.x` candidate,
2026-07-24. Exact conventions and current validation gaps are normative in
`anharmonic_theory.md` and `anharmonic_validation.md`.

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
- cubic (three-phonon) scattering in the self-consistent Born approximation
  (SCBA);
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
- nonnegative contact broadenings at positive frequency within tolerance;
- causality of \(G^r\) and \(\Sigma^r\);
- \(J_L+J_R=0\);
- spatial harmonic bond current is constant when the cubic vertices are
  onsite or strictly intralayer;
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
Green-function blocks. It must agree with the contact current after
convergence only for onsite or strictly intralayer cubic vertices.
FPU-alpha bond interactions require an additional interaction-current
operator; until that operator is derived, validate those fixtures with
contact currents and collision power.

## 5. Approximation hierarchy

| Method | Role | Advantages | Main risk |
|---|---|---|---|
| Full dense cubic SCBA | Reference implementation | Conserving when implemented consistently; handles nonequilibrium broadening | Highest cost |
| Block-sparse SCBA from FC3 support | Production target | Preserves nonlocal terms that actually exist | More complex data movement |
| One-shot Born | Debugging and weak-coupling baseline | Simple; self-energy scales as \(\Psi^2\) | Generally not current conserving |
| Lowest-order approximation / Padé rescaling | Later acceleration | Published speedup relative to SCBA | Must be calibrated against SCBA |
| Quasiparticle/on-shell self-energy | Bulk validation and large systems | Direct comparison to BTE/phono3py | Loses device coherence and strong nonequilibrium |
| Self-consistent phonon/effective harmonic model | Frequency renormalization | Useful for strong quartic anharmonicity | Does not by itself describe inelastic current |
| Büttiker probes | Optional engineering model | Very cheap | Phenomenological unless independently calibrated |
| Local or atomic-diagonal SCBA | Diagnostic only | Reduced memory | The attached 2020 paper shows appreciable over-scattering |

The production sequence should be full dense SCBA, then block sparsity, then
optional acceleration. It should not begin with a local-diagonal
approximation.

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
7. Add checkpoint/restart at each converged SCBA iteration.
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
- harmonic bond current is spatially constant for onsite or strictly
  intralayer cubic vertices;
- convergence under frequency spacing, cutoff, eta, mixing, and device size;
- weak-coupling comparison to an independently implemented master-equation
  or exact-diagonalization calculation where their assumptions overlap.

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

Use Guo et al., Phys. Rev. B 102, 195412 (2020):

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

Then use Guo et al., Phys. Rev. B 103, 174306 (2021):

- idealized Si/Ge mass-mismatch interface;
- scattering only in the interface device region;
- spectral generation/annihilation integrates to zero;
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

| Release | Required outcome | Current state |
|---|---|---|
| `0.1.0` | Python-only dense cubic SCBA for validated toy models | The eta-free open three-mode oracle, three-layer/nine-DOF SCBA harness, continuation graph, full preregistered numerical ladders, and combined strict-JSON report are implemented. The initial under-resolved open calculation failed, but the actual production API now passes the eta-free pole gate on a 143,143-frequency base grid and its independent half-spacing, half-eta, and larger-cutoff refinements; the largest observable refinement change is 0.162%. Exact padded FFT convolution/Hilbert paths and analytic diagonal contacts are cross-tested against the direct/Sancho references. The bounded nine-DOF PR graph passes but is non-certifying; the full nine-DOF ladder and final combined archive remain before release. Mingo and Wang \(O(T^4)\) are nonblocking historical extensions. |
| `0.2.0` | phono3py FC3 import and bulk three-phonon linewidth validation | Full/compact reading, schema/permutation checks, unit conversion, mass normalization, hashes, and provenance are implemented. Translational device mapping and the pinned phono3py matrix-element/linewidth oracle remain. |
| `0.3.0` | Sparse real-space SCBA for quasi-1D material devices | Not implemented. It must first reproduce the dense contractions exactly and retain complete FC3-implied support. |
| `0.4.0` | Transverse-momentum SCBA for 2D/3D geometries | Not implemented. It requires explicit coupled \(q'\) sums, momentum folding, and validated \(1/N_q\) normalization. |
| `1.0.0` | Silicon/interface paper benchmarks, restart, convergence archive, production CPU performance | Not implemented. The attached Guo papers define later benchmark targets; they are not evidence for the current dense toy candidate. |

Optional LOA/Padé, local self-energy approximations, quartic
renormalization, four-phonon scattering, thermal expansion, and anharmonic
reservoirs are post-reference branches. None may become a silent default.

## 11. Core literature

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
