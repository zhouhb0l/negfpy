# Anharmonic NEGF validation and release checklist

Status: executable `0.1.0.dev0` validation contract and implementation snapshot,
2026-07-24.

This document distinguishes three things that should not be conflated:

- a solver call that reached a numerically self-consistent state;
- a result whose built-in diagnostics mark it `release_certified`;
- a NEGFPy release for which all independent physics, refinement, and
  provenance gates have been run and archived.

Only the third supports a publication-quality platform claim.

## 1. Running the current checks

From the repository root, in an environment with NEGFPy installed in editable
mode:

```bash
python -m pytest -q
python examples/anharmonic_toy_scba.py
```

Focused dense-anharmonic checks can be run with:

```bash
python -m pytest -q \
  tests/test_anharmonic_types_grid.py \
  tests/test_anharmonic_contacts_current.py \
  tests/test_anharmonic_cubic_retarded.py \
  tests/test_anharmonic_diagnostics.py \
  tests/test_anharmonic_solver.py \
  tests/test_anharmonic_threaded.py \
  tests/test_anharmonic_toy.py \
  tests/test_anharmonic_bond_current.py \
  tests/test_anharmonic_three_mode.py \
  tests/test_anharmonic_fgr.py \
  tests/test_anharmonic_historical_toy.py \
  tests/test_anharmonic_toy_refinement.py \
  tests/test_anharmonic_toy_release_report.py \
  tests/test_anharmonic_release_matrix.py \
  tests/test_anharmonic_open_three_mode.py \
  tests/test_anharmonic_multidimensional_release.py \
  tests/test_anharmonic_production_gate_report.py \
  tests/test_anharmonic_production_gate_runner.py \
  tests/test_anharmonic_fc3.py \
  tests/test_ifc_audit.py
```

The example is intentionally a fast smoke calculation. Its grid is too
coarse to establish cutoff, first-moment, and high-frequency convergence, so
it must not be cited as a release benchmark.

Hosted workflows never enable or execute the full release calculation and do
not connect to private workstations. The bounded nine-DOF smoke graph remains
in pull-request and scheduled CI. The complete gate runs from a clean bundled
commit on the designated workstation using the key-only, checksum-verified
procedure in [remote heavy-validation operations](remote_heavy_validation.md).
Ordinary pytest discovery continues to skip the expensive release runner.

## 2. Test tiers

### Production-first ordering

Release certification follows the production Guo-convention solver, not the
ease with which a historical curve can be matched. The next scientific gates
are, in order:

1. a continuum or explicitly open-system mode-space linewidth and frequency
   shift oracle, with physical bath broadening refined independently of
   `eta_lead` and the frequency grid;
2. a release-grade three-layer, three-polarization intralayer SCBA problem
   that reaches the same fixed point from zero and Born-seeded initial states
   and passes the separate contact-balance and collision-power tests;
3. the pinned phono3py silicon matrix-element, phase-space, and bulk-linewidth
   oracle described in section 4.

The phono3py work may proceed in parallel with the second toy gate, but no
material-device SCBA result is certifiable until both the production toy
gates and the bulk oracle pass.

One-shot Born is a screening calculation. It is appropriate for testing
channel locations, Bose factors, \(\Psi^2\) scaling, and for constructing an
SCBA initial state. Because it is neither a nonlinear fixed point nor
generally current conserving, a nonzero-cubic one-shot result can never
satisfy interacting release certification. Certification requires converged
SCBA plus all applicable algebraic, spectral, refinement, and conservation
gates.

### Pull-request tier

Run on every change:

- scalar surface Green function, matched chain, mass defect, Caroli
  transmission, and Landauer prefactor;
- signed-grid indexing and exact negative-frequency partners;
- contact Keldysh construction, sign convention, and harmonic-limit current;
- one-to-three-DOF direct cubic contraction oracle;
- fixed \(i/2\) prefactor, \(\Psi^2\) scaling, and zero-extended
  convolution;
- direct principal-value reconstruction;
- an independently enumerated exact-bin mode-space Born/FGR oracle, including
  the dynamic bubble shift and the spectral HWHM/FWHM convention;
- algebraic symmetry, passivity, FDT, high-frequency, and moment fixtures;
- zero-vertex, weak-cubic, and alternate-initial-state SCBA;
- serial/threaded equality;
- contact balance, collision power, and supported bond-current checks;
- independent recomputation of contacts, Dyson/Keldysh fields, the unmixed
  cubic fixed point, currents, and terminal convergence history before a
  result can be certified;
- complete partial result on nonconvergence;
- dense memory preflight.

### Nightly tier

Run on pinned hardware and environments:

- dense \(D=8\) and \(D=16\) timing and memory benchmarks;
- \(\Delta\omega/2\), larger-cutoff, and smaller-`eta_lead` refinements;
- continuation in cubic coupling and temperature bias;
- convergence from zero and weak nonzero initial self-energies;
- the eta-free continuum/open-system linewidth and shift oracle against the
  contact-dressed production Born calculation;
- the full preregistered three-layer, three-polarization SCBA release case,
  including zero/Born starts, continuation, serial/threaded equality,
  conservation, and all four independent refinement controls;
- three-mode decay and combination spectra, including spontaneous decay at
  zero temperature;
- mode-space Fermi-golden-rule linewidth and frequency-shift comparisons;
- dense/optimized-backend equality whenever an optimized backend exists;
- phono3py external-oracle checks once FC3 ingestion is released.

### Release-validation tier

Archive machine-readable inputs, outputs, diagnostics, timings, code revision,
dependency versions, and input hashes for:

- the full public toy-model refinement matrix;
- exact historical model parameters and convention maps for the Mingo and
  Wang chain studies;
- the pinned phono3py Si-PBEsol dataset and bulk linewidth oracle;
- silicon-film and Si/Ge interface benchmarks once those solvers exist;
- interrupted/restarted and uninterrupted production calculations;
- representative serial/threaded, dense/sparse, and direct/FFT comparisons
  as those backends are added.

Wall-time regressions above 15% and peak-memory regressions above 10% on the
pinned benchmark host are flagged. Performance changes may not change
physical support, convergence thresholds, or conservation tolerances.

## 3. Numeric acceptance gates

| Group | Required cases | Acceptance |
|---|---|---|
| Harmonic | Scalar surface GF, uniform chain, mass defect, random-\(q\) FC2 reconstruction, left/right current, eta/BZ refinement | Analytic/source error \(10^{-10}\) or better; complete BZ weight |
| Cubic kernel | Literal nested-loop oracle, leg permutations, sign reversal, \(\Psi\mapsto c\Psi\), frequency/time-domain comparison | Relative error below \(10^{-11}\); exact \(c^2\) scaling |
| One-shot Born screen | Contact-dressed \(G_0\), one bubble, one updated solve | Diagnostic agreement with independent weak-coupling algebra; never certifies a nonzero-cubic interacting result |
| Three-mode physics | Decay and combination peaks, occupation factors, zero-temperature decay | Correct peak positions and independent FGR weights |
| Retarded reconstruction | Analytic Hilbert pair, spectral identity, Kramers--Kronig real part | Direct principal-value fixture below \(10^{-11}\) |
| Equilibrium | Gapped contact-only and interacting chains, at zero and finite temperature | KMS/FDT/Keldysh residuals below \(10^{-8}\); \(|J_\alpha|<10^{-10}J_0\) |
| SCBA fixed point | One-site and three-to-five-layer chains, two initial states | Unmixed \(\Sigma^{r,<,>}\) and \(G\) residuals below \(10^{-8}\) for three consecutive iterations; same fixed point |
| Conservation | Local/intralayer and bond-cubic fixtures | \(|J_L+J_R|\), \(|P_{pp}|\le\max(10^{-10}J_0,10^{-5}\max(|J_L|,|J_R|))\) |
| Spectral quality | Passivity, signed-frequency symmetry, high-frequency limit, first moment | Eigenvalue floor no worse than \(-10^{-10}\) of the spectral scale; moment error below \(5\times10^{-3}\) |
| Refinement | Independent half-spacing, larger-cutoff, eta-at-most-half, and changed-mixing runs | Current and internally derived selected-peak changes below \(5\times10^{-3}\); outer interaction-self-energy tail below \(10^{-6}\) |
| Weak coupling | Every released cubic toy point | Linewidth and shift each below 10% of the associated mode frequency |
| Optimized backends | Serial/threaded and later dense/sparse, direct/FFT, dense/selected GF | Field relative error below \(10^{-10}\), scale-aware absolute error below \(10^{-12}\); unchanged conservation gate |
| Restart | Continuous/interrupted runs and backend/worker change | Near-bitwise serial result and cross-backend error below \(10^{-10}\) |

For FPU-\(\alpha\) bond interactions, the conservation gate uses contact
currents and collision power. Spatial constancy of the harmonic bond-current
formula is required only for onsite or strictly intralayer cubic vertices,
until an interaction-current operator is derived.

## 4. Real-material gates

The canonical public input is the
[phono3py v4.4.0 Si-PBEsol
example](https://github.com/phonopy/phono3py/tree/v4.4.0/example/Si-PBEsol),
pinned to commit
`302ad205452ee36e57887b10bbf4a0289dd037bb`.
The archive manifest must contain SHA256 hashes for every consumed file and
must record whether `fc3.hdf5` is full or compact.

Before material-device SCBA is attempted, the same FC2/FC3, cell mappings,
q mesh, smearing, isotope settings, and sum-rule policy must be used to
compare:

1. FC2 dynamical matrices at deterministic random q points;
2. sampled three-phonon matrix elements;
3. three-phonon phase space;
4. on-shell equilibrium linewidths.

The matrix-element squared relative error must be below \(10^{-8}\). For
nonzero-mode linewidths, the median relative error must be below 1% and the
maximum below 5%, with a separately stated absolute tolerance for acoustic
modes.

Later quasi-1D material calculations must demonstrate dense/sparse equality,
conservation, grid refinement, and invariance when the lead-like interacting
buffer is extended beyond the FC3 range. A transverse calculation must first
show that \(N_q=1\) reduces to the real-space result and that real-space and
q-space cubic contractions agree on a small cell.

## 5. Reproducibility record

Every archived result must identify:

- method (`born` or `scba`) and retarded strategy;
- all Green-function, Fourier, vertex, current-sign, and frequency
  conventions;
- frequency grid, cutoff, rectangle weight, temperatures, and `eta_lead`;
- explicit zero `eta_device`;
- \(\omega_0\), \(\hbar\), source units, conversion factors, and current
  scale;
- dtype, backend, worker count, memory estimate, and host information;
- source revision and dependency versions;
- complete hashes for IFCs, geometry, mappings, and configuration;
- mixed interaction self-energy used for continuation/restart;
- complete convergence history and named diagnostic metrics;
- whether the result and the enclosing release passed their distinct gates.

Changing \(\omega_0\), mixing, or worker count must not change the converted
physical fixed point within the stated tolerances.

## 6. Honest implementation snapshot

Implemented and covered by automated tests:

- immutable signed grids, solver scaling, dense symmetric cubic vertices,
  reservoir/problem/configuration/result types, and memory estimation;
- explicit reduced/physical scale provenance, conversion factors, physical
  current tolerances, and invariance under a change of reference
  \(\omega_0\);
- harmonic-contact retarded/lesser/greater construction on a signed grid;
- consistent left/right principal-layer orientation and explicit rejection
  of acoustic or soft leads in the pinned/gapped `0.1.x` path;
- direct zero-extended dense cubic contraction with serial and threaded
  execution;
- direct principal-value and diagnostic `spectral_only` retarded
  reconstruction;
- explicitly selected zero-padded FFT implementations of the exact linear
  cubic convolution and finite-grid principal-value sum. At 513
  frequencies and three DOF, the cubic primitive was 142 times faster than
  direct with maximum field error \(7.8\times10^{-16}\); the Hilbert
  primitive was 148 times faster. Edge-impulse tests prohibit circular
  wraparound, and the direct paths remain the defaults and reference oracles;
- an exact vectorized analytic-diagonal contact method, cross-tested against
  Sancho--Rubio for the open three-mode fixture;
- one-shot Born and dense self-consistent Born solvers with unmixed
  residuals, adaptive downward mixing, continuation, convergence history,
  and partial-result exceptions;
- zero and Born-seeded SCBA starts for both one- and three-layer interacting
  chains, agreeing in every \(\Sigma_{pp}^{r,<,>}\) and \(G^{r,<,>}\) field
  at \(10^{-8}\)-level tolerances after three passing iterations;
- contact current, collision power, and harmonic bond current where its
  topology is valid;
- pinned scalar, three-polarization intralayer, and FPU-\(\alpha\) toy
  builders;
- exact-bin three-mode decay/combination fixtures with hand-derived Bose
  weights and zero-temperature spontaneous decay;
- an independent ordered-channel mode-space oracle for the same exact-bin
  fixture, with
  \(\Delta\Omega=\operatorname{Re}\Sigma^r(\Omega)/(2\Omega)\),
  linewidth/FWHM \(=-\operatorname{Im}\Sigma^r(\Omega)/\Omega\), a fixed 10%
  release acceptance check, and real normal-mode projection;
- an independent eta-free open-system three-mode continuum oracle. It uses
  analytic scalar surface spectra, explicit Bose-channel weights, adaptive
  convolution, and adaptive principal-value quadrature, and does not call
  the production cubic or retarded-reconstruction kernels;
- a three-layer, nine-DOF \(x,y,z\) intralayer benchmark with immutable
  coupling, bias, grid, cutoff, lead-eta, and mixing ladders. Its bounded PR
  profile exercises zero, Born-seeded, continuation, and threaded solves;
  all six coarse solves, all three field comparisons, and both equilibrium
  and nonequilibrium conservation checks pass. That profile is explicitly
  non-certifying;
- a combined strict-JSON production-toy-gate report that derives observables
  and hashes from solver objects, records adaptive-oracle error estimates,
  rejects incomplete temperature/sign/scaling/refinement coverage, and
  cannot promote the PR profile to a release result;
- a bounded spacing/cutoff/eta/mixing refinement harness that retains
  nonconverged partial results without promoting coarse runs to certified
  status;
- deterministic strict-JSON refinement reports with report/input/source
  hashes, Git and package-version checks, internally derived spectral-peak
  measurements, directional coverage of all four required controls, and
  separate numerical-refinement and release-certification outcomes;
- a nightly nonzero-cubic four-control matrix with independent fixed-eta
  spacing refinement and fixed-grid eta refinement. On its 513-point
  reference grid, \(\eta_{\rm lead}=0.015\to0.0075\) changes the interacting
  current by \(2.86\times10^{-3}\), while
  \(\Delta\omega=0.03125\to0.015625\) changes it by
  \(3.12\times10^{-4}\). Every run passes its spectral moment and per-result
  gates, and the fitted interaction-broadening peak frequency changes by
  less than \(5\times10^{-3}\);
- algebraic, symmetry, passivity, FDT, spectral-moment, high-frequency,
  conservation, memory, and deterministic-backend tests;
- a harmonic IFC audit for reciprocity, acoustic and rotational sum-rule
  residuals, q-space Hermiticity, mapping consistency, and sampled stability;
- full and compact phono3py `fc3.hdf5` ingestion foundations, including
  hashes, map validation, unit conversion, and mass normalization;
- a refined harmonic-limit SCBA run that passes all per-result certification
  gates end to end;
- a primary-source Mingo/Wang convention map; separate Wang
  caption-literal and explicitly inferred author-legacy Figure 2 fixtures;
  all four vector EPS series with affine-extraction provenance; conditional
  on the non-unique inferred onsite-unit convention, full 51-point ballistic
  RMS/max errors of \(0.000124/0.000277\) nW/K, historical \(O(T^2)\)
  bubble-plus-tadpole agreement within 1% at 100/300/500 K, and selective
  mean-field agreement within 1% at 300/400 K;

Still required before declaring the complete `0.1.0` research release
validated:

- archive the already passing fine open-system ladder in one combined report.
  The initial \(\Delta\omega=0.015,\eta_{\rm lead}=0.001\) diagnostic failed
  by 42.6% in shift and 10.5% in linewidth because the regulator was much
  narrower than the grid. With the prefactor unchanged, the production API
  at \(\Delta\omega=7.0\times10^{-5}\),
  \(\eta_{\rm lead}=4.5\times10^{-5}\), and cutoff \(5.00997\) agrees with
  the eta-free oracle at \(T=0.7\) to 0.0776% in shift and 0.0278% in
  linewidth. Half spacing, half eta, and cutoff 6 were varied independently;
  the largest observable change was 0.162%, below the 0.5% gate. At \(T=0\),
  the base errors are 0.318% and 0.0269%, and spontaneous decay remains
  nonzero. Every executed fine point passed the production diagnostics;
- execute the remaining signed-coupling matrix in the strict combined
  archive. Sign invariance and exact \(g^2\) scaling already pass both direct
  and FFT primitive tests, but the release report deliberately requires the
  full run-object coverage;
- execute and archive the expensive full nine-DOF SCBA profile. Its complete
  continuation and refinement graph is implemented, but only the
  non-certifying PR profile has been run in this snapshot;
- archive the combined strict-JSON release report on a pinned environment.

Historical extensions are deliberately absent from this blocker list.
Mingo is frozen as an unresolved historical reproduction after at most four
developer-hours of additional primary-source/code searching. It may be
reopened only if unique primary evidence resolves the surface normalization
and cubic-input convention; no spectral weight, vertex multiplier, or
diagrammatic coefficient may be chosen by fitting Figs. 4 or 8. Wang's six
\(O(T^4)\) diagrams are a useful, separately named historical extension, but
they are not part of the production bubble-SCBA method and do not block
`0.1.0`. The existing Wang harmonic, \(O(T^2)\), and stable-range selective
mean-field checkpoints remain source-limited regressions.

Not yet claimed by this snapshot:

- complete phono3py device mapping or validated bulk matrix elements and
  linewidths (`0.2.0`);
- sparse quasi-1D real-material SCBA (`0.3.0`);
- transverse-momentum-coupled SCBA (`0.4.0`);
- silicon-film or Si/Ge paper reproduction, HDF5 restart, or production
  performance certification (`1.0.0`).

Passing the current unit suite is therefore evidence for the dense numerical
foundation, not evidence that the later real-material milestones are
complete.
