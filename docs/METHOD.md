# Numerical method

The numerical engine is selected from the existing NEGFPy implementation; the package adds a portable JSON interface and standard result export. Source selection and adaptations are recorded in `SOURCE_PROVENANCE.json`.

The device uses mass-weighted displacements, `D_ab = Phi2_ab / sqrt(M_a M_b)`, with cubic Hamiltonian `H3 = sum(Psi_abc u_a u_b u_c) / 3!`. The existing cubic bubble has prefactor `i hbar / 2`, full permitted nonlocal contractions and zero-extended frequency convolutions. Retarded self-energy is reconstructed with the existing causal transform. Harmonic lead contacts supply paired retarded/thermal fields and the continuous fixed-eta zero-frequency limit. Device eta is zero.

The fitted site-energy representation selects complete center energies, retaining every endpoint derivative. It defines an explicit finite-boundary model. A bulk IFC fit residual does not bound heat-current error or define a unique physical interface. The example records its measured residual without changing the saved coefficients.

Independent retarded/statistical/spectral channels are retained throughout. No contour subtraction repair, physical projection, or tolerance relaxation is applied to a completed full map. Proposal damping is only an iteration control; physical and convergence guards are unchanged.

Three unmixed Sigma screens and three current screens gate convergence. SCBA self-consistency on one supplied model/grid is distinct from frequency/q/broadening convergence, interface validity and the physical accuracy of the Born approximation.

## Reference/increment SCBA

Set `scba.method` to `"reference-seeded"` to use the existing reference/increment
method with bounded analytic feedback. The other methods, `anderson1` and
`linear`, remain available, and version 1.0.0 JSON inputs remain valid.

By default an invocation starts at exact zero scattering self-energy. Version
1.1.0 optionally reads an authenticated evaluated self-energy as an initial
guess through [initial_self_energy](RESTART.md). References are selected
evaluated states constructed during the new invocation, with the same
model, geometry, frequency/q grids, contacts, temperatures and coupling. No
saved solver history or reference Green function is reused. A loaded guess
undergoes a fresh full evaluation before it can become a selected reference.

After up to two fixed-mixing warm-up maps, the method evaluates subsequent
maps using exact Dyson/Keldysh increments about a selected reference. The cubic
Born increment is assembled through its bilinear cross terms. This is an
algebraic reorganization of the existing full map; it keeps the full device
and all permitted nonlocal interaction blocks. Reference subtraction alone
does not change the SCBA fixed point or establish faster convergence.

The existing analytic-feedback controller can evaluate a signed tangent in
the residual direction and propose an update using one bounded real
coefficient and a common trust limit. Tangents are arithmetic directions,
not physical scattering fields or completed nonlinear maps. Each proposed
state must undergo a fresh full-map evaluation with the original physical
guards and all six convergence screens. Strict selected-anchor acceptance,
PSD proposal backtracking and the existing linear fallback remain in force.

This method admits at most 18 nonlinear map attempts and six signed tangent
directions. Rejected full-map trials consume the map budget; tangent attempts
have their own count and resource/time admission. A failure stops the run
without automatic retry. These ceilings bound the supplied method and do not
replace its physical or convergence tests. Memory/work plans include the
reference/increment and tangent arithmetic; library scratch and caller-owned
storage remain outside some estimates.

Version 1.0 contains finite-device cubic interactions and harmonic leads. Version 1.0.2 additionally packages the existing [Method 1 frozen-retarded/local-FD long-device approximation](METHOD1.md). Neither interface includes quartic/four-phonon diagrams, anharmonic leads, implicit device eta baths, extended interaction range or multi-node job orchestration.
