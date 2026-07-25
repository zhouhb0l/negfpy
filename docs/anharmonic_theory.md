# Dense cubic phonon NEGF: conventions and numerical contract

Status: frozen convention specification for the dense `0.1.x` reference
implementation, 2026-07-24.

This document is normative for `negfpy.anharmonic`. It fixes the signs,
normalizations, units, quadrature, and approximation boundaries used by the
dense solver. A later optimized backend is correct only if it reproduces this
reference on overlapping problems.

## 1. Coordinates, Hamiltonian, and Green functions

The internal coordinate is the mass-weighted displacement

\[
u_a=\sqrt{M_a}\,\Delta r_a,
\]

where a composite index \(a\) identifies an atom and Cartesian direction.
The harmonic dynamical matrix and mass-normalized cubic IFC are

\[
D_{ab}=\frac{\Phi^{(2)}_{ab}}{\sqrt{M_aM_b}},
\qquad
\Psi_{abc}=\frac{\Phi^{(3)}_{abc}}{\sqrt{M_aM_bM_c}}.
\]

Here \(M_a\) denotes the mass of the atom associated with coordinate \(a\).
The cubic Hamiltonian convention is

\[
H_3=\frac{1}{3!}\sum_{abc}\Psi_{abc}u_au_bu_c .
\]

\(\Psi\) is real for the dense real-space solver and fully symmetric under
all permutations of its three indices. Input validation rejects an
asymmetric tensor; the solver never silently symmetrizes it.

The displacement Green functions are

\[
G^<_{ab}(t,t')=-\frac{i}{\hbar}
\langle u_b(t')u_a(t)\rangle ,
\]

\[
G^>_{ab}(t,t')=-\frac{i}{\hbar}
\langle u_a(t)u_b(t')\rangle ,
\qquad
G^r=\theta(t-t')\left(G^>-G^<\right).
\]

The Fourier pair is

\[
F(\omega)=\int_{-\infty}^{\infty}dt\,e^{i\omega t}F(t),
\qquad
F(t)=\int_{-\infty}^{\infty}\frac{d\omega}{2\pi}
e^{-i\omega t}F(\omega).
\]

All frequencies in the solver are angular frequencies. An input or output
interface expressed in THz must state whether it means cycles/s or rad/s and
must apply the factor \(2\pi\) exactly once.

## 2. Dyson and Keldysh equations

For a finite interacting device connected to two harmonic reservoirs,

\[
G^r(\omega)=
\left[
(\omega+i0^+)^2I-D-\Sigma_L^r-\Sigma_R^r-\Sigma_{pp}^r
\right]^{-1},
\]

\[
G^{</>}(\omega)=G^r(\omega)
\left(\Sigma_L^{</>}+\Sigma_R^{</>}+\Sigma_{pp}^{</>}\right)
G^a(\omega),
\qquad
G^a=G^{r\dagger}.
\]

The certified dense calculation uses zero device regulator. The numerical
lead broadening `eta_lead` belongs only to the surface Green-function
calculation. A nonzero device regulator would be an unaccounted dissipative
bath and would invalidate the ordinary Keldysh identity and two-contact
energy balance. A future finite regulator must therefore be represented as
an explicit reservoir with mutually consistent retarded, lesser, and greater
components.

The surface solver evaluates \(z=(\omega+i\eta_{\rm lead})^2\), whose
positive-frequency imaginary part is \(2\omega\eta_{\rm lead}\). Thus its
frequency-scale smoothing is of order \(\eta_{\rm lead}\). A rectangle grid
that does not resolve that smoothing can make a smaller eta look less
accurate, especially in the first spectral moment near a one-dimensional
band-edge singularity. Release refinement therefore uses two independent
controls:

1. reduce \(\eta_{\rm lead}\) by a factor of two at fixed
   \(\Delta\omega\), with both calculations passing the moment and other
   per-result gates;
2. reduce \(\Delta\omega\) by a factor of two at fixed
   \(\eta_{\rm lead}\).

If the eta-reduced calculation fails spectral resolution, the grid is first
refined and the eta-only comparison is repeated. Eta and spacing are never
reduced together and presented as an eta test. A single global eta is used
for every frequency and contact in one calculation. The dense-v0.1 release
does not extrapolate eta to zero; any future extrapolation must be a
separately named approximation with its fitted order and raw sequence
archived.

For reservoir \(\alpha\),

\[
\Gamma_\alpha=i\left(\Sigma_\alpha^r-\Sigma_\alpha^a\right),
\]

\[
\Sigma_\alpha^<=-i n_B(\omega,T_\alpha)\Gamma_\alpha,
\qquad
\Sigma_\alpha^>=-i[1+n_B(\omega,T_\alpha)]\Gamma_\alpha .
\]

The signed-frequency Bose continuation is used. The implementation does not
construct a finite generic value for \(n_B(0)\). Version `0.1.x` is restricted
to pinned or gapped leads, for which the zero-frequency contact spectral
weight vanishes. The current dense path verifies a positive bulk lead gap
over the one-dimensional Brillouin zone before constructing contacts. This
sampled-plus-local-minimization check is a guard for the released pinned toy
leads, not a mathematical global-gap certificate for arbitrary future
multidimensional leads. Acoustic support must instead evaluate the analytic
\(n_B\Gamma\) limit using `expm1`, a small-argument series, and
\(\Gamma/\omega\).

## 3. Cubic self-energy

The production cubic bubble follows Eq. (8) of
[Guo et al., Phys. Rev. B 102, 195412
(2020)](https://doi.org/10.1103/PhysRevB.102.195412):

\[
\Sigma_{pp,ab}^{</>}(\omega)=
\frac{i\hbar}{2}\sum_{cdef}\Psi_{acd}\Psi_{bef}
\int_{-\infty}^{\infty}\frac{d\omega'}{2\pi}
G_{ce}^{</>}(\omega')G_{df}^{</>}(\omega-\omega').
\]

The prefactor \(i\hbar/2\) follows from the \(1/3!\) Hamiltonian convention
above. It is fixed, not fitted. Reversing the sign of every cubic IFC leaves
the bubble unchanged, while \(\Psi\mapsto c\Psi\) gives
\(\Sigma_{pp}^{</>}\mapsto c^2\Sigma_{pp}^{</>}\).

Let

\[
S(\omega)=\Sigma_{pp}^>(\omega)-\Sigma_{pp}^<(\omega).
\]

The causal retarded component is reconstructed as

\[
\Sigma_{pp}^r(\omega)=
\frac{1}{2}S(\omega)+
i\,\mathcal P\int_{-\infty}^{\infty}
\frac{d\omega'}{2\pi}\frac{S(\omega')}{\omega-\omega'}.
\]

`causal_hilbert` evaluates the principal-value sum directly and is the
validated `0.1.x` path. `spectral_only` retains only \(S/2\); it is explicitly
a damping-only, Kramers--Kronig-incomplete diagnostic and cannot be
release-certified.

## 4. Dimensionless solver units

Given a reference angular frequency \(\omega_0\), the solver uses

\[
\widetilde\omega=\omega/\omega_0,\qquad
\widetilde D=D/\omega_0^2,
\]

\[
q=\frac{u}{\sqrt{\hbar/\omega_0}},
\qquad
\widetilde\Psi=\Psi\frac{\sqrt{\hbar}}{\omega_0^{5/2}},
\qquad
\widetilde T=\frac{k_BT}{\hbar\omega_0}.
\]

The physical current scale is

\[
J_0=\hbar\omega_0^2 .
\]

Toy builders use reduced units with
\(\hbar=k_B=\omega_0=1\). Material readers must convert raw IFC and mass units
to a stated physical system before applying these solver scalings. A result
must retain \(\omega_0\), \(\hbar\), source units, conversion metadata, and
\(J_0\). `SolverScale` records the four conversion factors explicitly and
marks non-reduced source units as undeclared unless the caller supplies them;
such a result cannot be release-certified. Re-solving the same physical
problem with a different \(\omega_0\) is a mandatory scale-invariance test.

## 5. Signed frequency grid and quadrature

`SignedFrequencyGrid(step, n_positive)` constructs exactly

\[
\omega_n=n\Delta\omega,\qquad n=-N,\ldots,N,
\]

with \(N=\texttt{n_positive}\) and
\(\Delta\omega=\texttt{step}\). The grid is odd, uniform, contains zero
exactly, and has an exact negative-frequency partner for every index.

Every integral uses the rectangle weight

\[
w_\omega=\frac{\Delta\omega}{2\pi}.
\]

Frequency convolution is performed with integer index arithmetic. Terms for
which \(\omega-\omega'\) lies outside the grid are zero; circular wraparound
is forbidden. Initial toy studies should use
\(\omega_{\rm cut}\ge 3\omega_{\rm harmonic,max}\) and must then refine both
cutoff and spacing. The outer lesser/greater interaction-self-energy
amplitude is required to be small compared with its peak.

The reference Hilbert transform is the direct \(O(N_\omega^2)\)
principal-value sum. Its omitted diagonal implements the discrete principal
value. Any later FFT implementation must use sufficient zero padding and
reproduce this oracle before it can be selected.

## 6. Heat-current convention

Energy current is positive into the device:

\[
J_\alpha=\operatorname{Re}\int_0^\infty\frac{d\omega}{2\pi}
\hbar\omega\,\operatorname{Tr}
\left[\Sigma_\alpha^>G^<-\Sigma_\alpha^<G^>\right].
\]

Thus \(T_L>T_R\) must produce \(J_L>0\) and \(J_R<0\). At
\(\Psi=0\), this expression must reproduce the harmonic Landauer current.

The interaction collision power is

\[
P_{pp}=\operatorname{Re}\int_0^\infty\frac{d\omega}{2\pi}
\hbar\omega\,\operatorname{Tr}
\left[\Sigma_{pp}^>G^<-\Sigma_{pp}^<G^>\right].
\]

A conserving SCBA fixed point requires \(J_L+J_R=0\) and \(P_{pp}=0\)
independently, as well as full balance \(J_L+J_R+P_{pp}=0\). The
frequency-resolved collision spectrum may be nonzero and sign-changing.
Three-phonon scattering need not conserve phonon number.

The harmonic bond-current expression is a spatial conservation check only
for onsite or strictly intralayer cubic vertices. An FPU-\(\alpha\)
bond-cubic vertex transfers energy through an interaction-current operator
that has not yet been derived in this package. Such calculations are checked
with contact currents and collision power, and the harmonic bond-current
field is deliberately omitted.

## 7. One-shot Born and SCBA algorithms

One-shot Born has fixed diagnostic semantics:

1. solve the contact-only harmonic \(G_0^{r,<,>}\);
2. evaluate one cubic bubble from \(G_0^{</>}\);
3. reconstruct \(\Sigma_{pp}^r\);
4. perform one updated Dyson/Keldysh solve.

It is not generally conserving and is never release-certified.

The dense SCBA iteration is:

1. compute signed-frequency contact
   \(\Sigma_\alpha^{r,<,>}\);
2. initialize \(\Sigma_{pp}^{<,>}=0\), or accept an explicit continuation
   state;
3. reconstruct \(\Sigma_{pp}^r\) from the current mixed lesser/greater pair;
4. solve Dyson and Keldysh equations at every frequency;
5. evaluate the raw cubic map \(F(\Sigma)\);
6. measure the **unmixed** residual \(F(\Sigma)-\Sigma\), including the
   reconstructed retarded component;
7. linearly mix the lesser and greater components together, then reconstruct
   the retarded component on the next solve;
8. start from mixing \(0.2\); after three successive residual increases,
   halve it down to a floor of \(0.01\), and never increase it automatically;
9. require all fixed-point, identity, and conservation conditions for three
   consecutive iterations;
10. stop after 200 iterations by default. Nonconvergence raises
    `SCBAConvergenceError` carrying the last complete result and iteration
    history, unless an explicit diagnostic-return policy was requested.

The dense implementation uses `complex128`. The serial backend is the
deterministic oracle. The threaded backend partitions output frequencies and
must reproduce the serial fields without relaxing scientific tolerances.
The stored-state baseline is

\[
M_{\rm peak}\simeq
288N_\omega N_qD^2\ {\rm bytes}.
\]

The implemented preflight adds the actual worker-local tiled cubic
intermediates and row-wise Hilbert scratch to that baseline. It also records
the effective kernel, backend, and worker count (the scalar oracle uses a
serial exact convolution even if a threaded backend was requested).
Predicted use above 70% of available memory is rejected unless the caller
explicitly overrides the guard.

## 8. Required identities and certification meaning

The diagnostic layer checks, as applicable,

\[
G^a=G^{r\dagger},\qquad
G^>-G^<=G^r-G^a,
\]

\[
G^{</>\dagger}=-G^{</>},\qquad
[G^>(\omega)]^T=G^<(-\omega),
\]

\[
G^r(-\omega)=G^r(\omega)^*
\]

for real real-space IFCs. It also checks passivity of
\(\operatorname{sgn}(\omega)A(\omega)\) and
\(\operatorname{sgn}(\omega)\Gamma(\omega)\), the high-frequency limit

\[
\omega^2G^r(\omega)\longrightarrow I,
\]

using independent quadratic fits of \(\omega^2G^r\) versus
\(1/\omega^2\) on the positive and negative outer tails. The fitted
intercepts, rather than the finite-cutoff value, are compared with \(I\);
the latter contains the expected \(D/\omega_{\rm cut}^2\) correction and is
retained as a raw diagnostic. Certification fails closed when the tail is
under-sampled, non-monotone, ill-conditioned, or inconsistent with the fit.

It separately checks the first spectral moment

\[
\int_{-\infty}^{\infty}\frac{d\omega}{2\pi}
\omega A(\omega)=I,\qquad
A=i(G^r-G^a).
\]

Equilibrium calculations additionally check KMS/fluctuation--dissipation
relations without directly evaluating unstable large exponentials.

`DiagnosticReport.passed` means the essential algebraic and conservation
checks for that calculated state passed. `release_certified` is stricter: it
also requires a converged SCBA result, `causal_hilbert`, signed-frequency
symmetries, passivity, high-frequency and moment quality, and a sufficiently
small interaction-self-energy tail. Certification refuses solver tolerances
looser than the locked dense-v0.1 ceilings, fewer than three minimum or
consecutive iterations, undeclared physical source units, or a mismatch
between the installed and source-declared package versions. Provenance hashes
the exact lesser/greater continuation state actually consumed (the retarded
component is rebuilt), method, problem, configuration, and source tree.
Per-run certification does not replace
the independent refinement and benchmark suite described in
[anharmonic_validation.md](anharmonic_validation.md).

## 9. Deliberate `0.1.x` limits

The dense cubic reference does not include:

- cubic tadpole or static equilibrium shifts;
- quartic Hartree/loop diagrams or four-phonon scattering;
- thermal expansion or anharmonic reservoirs;
- local-diagonal or arbitrary block truncation;
- Büttiker probes or LOA/Padé approximations;
- acoustic zero-frequency reservoirs;
- mixed precision, GPU, or distributed execution;
- a production interaction-current operator for bond-cubic vertices.

A globally cubic potential is unbounded. Released toy points must remain in a
weak-coupling regime, with self-consistent linewidths and shifts below 10% of
their associated mode frequencies. The package makes no claim of
high-temperature stability for a pure cubic Hamiltonian.
