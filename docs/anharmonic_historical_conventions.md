# Historical anharmonic toy models: convention and benchmark map

Status: primary-source audit, 2026-07-24.

This note maps two early phonon-NEGF papers onto the frozen NEGFPy
conventions:

- N. Mingo, "Anharmonic phonon flow through molecular-sized junctions,"
  [Phys. Rev. B 74, 125402
  (2006)](https://doi.org/10.1103/PhysRevB.74.125402).
- J.-S. Wang, J. Wang, and N. Zeng, "Nonequilibrium Green's function
  approach to mesoscopic thermal transport," [Phys. Rev. B 74, 033408
  (2006)](https://doi.org/10.1103/PhysRevB.74.033408);
  [author preprint](https://arxiv.org/abs/cond-mat/0605028).

The purpose is reproducibility, not prefactor fitting. The production kernel
remains Eq. (8) of [Guo et al., Phys. Rev. B 102, 195412
(2020)](https://doi.org/10.1103/PhysRevB.102.195412), with

\[
H_3=\frac{1}{3!}\sum_{abc}\Psi_{abc}u_au_bu_c
\]

and

\[
\Sigma_{pp,ab}^{</>}(\omega)=
\frac{i\hbar}{2}\sum_{cdef}\Psi_{acd}\Psi_{bef}
\int\frac{d\omega'}{2\pi}
G_{ce}^{</>}(\omega')G_{df}^{</>}(\omega-\omega').
\]

Published curves are not acceptance oracles unless all model, unit, grid,
regulator, bias, and convergence details are available. A trend comparison
is explicitly weaker than a same-input numerical reproduction.

These historical checks are subordinate to the production validation
sequence. The immediate release gates are the open-system linewidth/shift
oracle and the release-grade three-layer, three-polarization conserving SCBA
case, followed by the pinned phono3py silicon bulk oracle. Historical
extensions do not block those gates.

## 1. Common ground

Both 2006 papers use displacement Green functions with the same lesser and
greater signs used by NEGFPy:

\[
G^<_{ab}(t,t')=-\frac{i}{\hbar}
\langle u_b(t')u_a(t)\rangle,\qquad
G^>_{ab}(t,t')=-\frac{i}{\hbar}
\langle u_a(t)u_b(t')\rangle .
\]

Their frequency-domain formulas are consistent with the Fourier pair

\[
F(\omega)=\int dt\,e^{i\omega t}F(t),\qquad
F(t)=\int\frac{d\omega}{2\pi}e^{-i\omega t}F(\omega).
\]

Both partition the system into harmonic semi-infinite contacts and a finite
interacting region. Both use full matrix self-energies, solve Dyson and
Keldysh equations, and iterate a cubic self-energy. Neither paper justifies
an arbitrary local-diagonal truncation.

The notation differs most importantly in the cubic Hamiltonian. A symbol
called a "third-order force constant" cannot be passed to the production
kernel until its Hamiltonian coefficient has been identified.

## 2. Mingo 2006

### 2.1 Convention map

Mingo begins with physical displacements \(\phi_i\) and defines

\[
V^{(3)}_{ijk}=\frac{1}{3!}
\frac{\partial^3E}{\partial r_i\partial r_j\partial r_k},
\qquad
H_3=\sum_{ijk}V^{(3)}_{ijk}\phi_i\phi_j\phi_k .
\]

After the paper's mass transformation
\(\widetilde\phi_i=\sqrt{M_i/M}\phi_i\), choose the arbitrary reference mass
\(M=1\) in a consistent unit system. The corresponding NEGFPy vertex is

\[
\Psi_{ijk}=
\frac{\partial^3E/\partial r_i\partial r_j\partial r_k}
{\sqrt{M_iM_jM_k}}
=6\,\widetilde V^{(3)}_{ijk}.
\]

This Hamiltonian conversion does **not** make Mingo's printed bubble
identical to the production bubble. Mingo's Eq. (27) uses
\(V^{(3)}\), an integral written as \(d\omega'\) rather than
\(d\omega'/(2\pi)\), and a different coefficient. Guo et al. compare the
expressions in their Table I and report a coefficient magnitude differing
by \(\pi/9\) from their validated Eq. (8). The direction is now encoded
unambiguously:

\[
\Sigma_{\rm Mingo\ legacy}^{</>}
=\frac{\pi}{9}\Sigma_{\rm Guo\ production}^{</>}.
\]

`mingo_2006_legacy_lesser_greater` implements only that named,
non-certifying historical experiment. NEGFPy does not offer a production
"Mingo prefactor" switch.

Mingo reconstructs the retarded self-energy from
\(\Sigma^>-\Sigma^<\): its imaginary part comes from the spectral
difference and its real part from a Hilbert transform. That corresponds to
NEGFPy's `causal_hilbert` strategy, not `spectral_only`.

The calculation is self-consistent in the full cubic bubble. It initializes
the many-body self-energy to zero and iterates \(G^{r,<,>}\) and
\(\Sigma_M^{r,<,>}\). The quartic static diagram is discussed but excluded.
NEGFPy likewise excludes quartic Hartree terms in `0.1.x`.

The paper's Eq. (42) uses

\[
J_{h(c)}^{\rm Mingo}(\omega)=
+(-)\frac{\hbar\omega}{2\pi}
\operatorname{Tr}\!\left[
\Sigma_{h(c)}^<D^>-\Sigma_{h(c)}^>D^<
\right].
\]

NEGFPy reverses the order inside the bracket and defines each contact
current as positive **into** the device. Therefore
\(J_h^{\rm Mingo}=-J_h^{\rm NEGFPy}\) and
\(J_c^{\rm Mingo}=J_c^{\rm NEGFPy}\). Mingo's hot and cold through-current
spectra have the same sign at conservation, whereas NEGFPy's contact
currents obey \(J_L=-J_R\), with \(J_L>0\) when the left contact is hot.

### 2.2 Recoverable molecular-junction inputs

The model has two Si-mass atoms, one displacement per atom normal to the
surfaces, a harmonic molecule-surface spring on each side, and an
anharmonic central bond:

| Quantity | Primary-source value |
|---|---:|
| Molecular degrees of freedom | 2 |
| Atomic mass | Si mass for every atom |
| Surface couplings | \(k_l=k_r\) |
| Central coefficient | \(k_m=\frac12 V''(x_0)\), as printed |
| Cubic derivative | \(V'''=\left|d^3V/dx^3\right|_{x_0}\) |
| Surface cutoff | \(\omega_D=100\), labeled THz |
| Bias | \(T_h=1.5T_c\) |
| Full signed range | \(-300\) to \(+300\), labeled THz |
| Green-function spacing | 0.5, labeled THz |
| Cubic-convolution spacing | 2.5, labeled THz, then interpolated |
| Initial mixing | \(f=0.5\) |
| Convergence criterion | \(s<10^{-5}\) |

The surface is not an explicit periodic lead. Its phenomenological spectral
density is

\[
\rho(\omega^2)=\frac{3}{2}\frac{\omega}{\omega_D^3}
\Theta(\omega_D-\omega),
\qquad
\int_0^\infty \rho(\omega^2)\,d\omega^2=1.
\]

The real part of the bare surface resolvent is obtained by a Hilbert
transform. The molecule changes the surface diagonal, giving

\[
G_l=G_{\rm bare}(1+k_lG_{\rm bare})^{-1},\qquad
\Sigma_l^r=k_l^2G_l,
\]

and similarly on the right.

The printed surface equations are not mutually consistent. Equation (14)
uses the causal resolvent \([(\omega+i0^+)^2-K]^{-1}\), whose imaginary part
is negative for positive frequency. Equation (16) prints the opposite sign,
and the prose immediately before Eq. (47) additionally prints a factor of
two different from Eq. (16). Equation (49) also uses force/Green signs
opposite to a standard positive-Hessian embedding. These are not harmless
typographical details in a pointwise reproduction.

For the normalized density above, the causal bare resolvent can instead be
integrated analytically:

\[
g_0^r(\omega)=\frac{3}{\omega_D^3}
\left[-\omega_D+\frac{\omega}{2}
\log\frac{\omega+\omega_D}{\omega-\omega_D}\right],
\]

where the logarithm takes the retarded branch. NEGFPy evaluates this
expression directly, including its negative-frequency conjugate, so this
historical reservoir requires no numerical lead `eta`. After mapping to a
positive-Hessian force convention, the attached surface is

\[
g_l^r=\frac{g_0^r}{1-k_lg_0^r},
\qquad \Sigma_l^r=k_l^2g_l^r .
\]

The logarithmic bare singularity at \(\omega_D\) has the finite dressed
limit \(g_l^r(\omega_D)=-1/k_l\).

For the paper's temperature curve (Figs. 4 and 8),

\[
k_m=0.316\ {\rm eV/\mathring A^2},\qquad
k_l=k_r=3.16\ {\rm eV/\mathring A^2}.
\]

The two illustrative potentials have

\[
\beta\equiv\frac{V'''}{V''}
\simeq 8.94\ {\rm \mathring A^{-1}}\quad{\rm (Lennard\!-\!Jones)}
\]

and

\[
\beta\simeq2.46\ {\rm \mathring A^{-1}}
\quad{\rm (two\!-\!body\ Stillinger\!-\!Weber)}.
\]

The paper uses \(\xi=2.0951\) angstrom in the Lennard-Jones potential,
placing its minimum at \(x_0=2.35\) angstrom. For the two-body
Stillinger-Weber form it gives \(B=0.6022245584\) and \(a=1.80\).

The label "THz" is ambiguous by modern interface standards. The paper uses
the symbol \(\omega\) and \(\hbar\omega\); moreover, a numerical cutoff near
100 is consistent with the angular-frequency extent of Si
(\(\sim2\pi\times15\) cycles/ps), not a 100-cycles/ps Si phonon band. This is
an inference, not an explicit unit declaration in the paper. A future
reproduction must record whether each number means \(10^{12}\ {\rm rad/s}\)
or \(10^{12}\) cycles/s and must run a \(2\pi\) sensitivity check.

### 2.3 Numerical status and approximation levels

The APS PDF stores Figs. 4 and 8 as vector paths. Their plotted coordinates
are now recorded as immutable reference curves, explicitly labeled as
figure extraction rather than author-supplied raw data.

The eta-free harmonic implementation retains the printed
\(V''=2k_m\), 28 amu Si masses, the 100-\(10^{12}\) rad/s cutoff, and the
paper's 0.5 grid spacing. With the normalized Eq. (47) surface measure, the
Fig. 4 curve differs by 3.2301 pW/K RMS (5.0423 pW/K maximum), so this is
**not** a same-input harmonic reproduction.

A separately named
`figure4_inferred_half_weight_sensitivity` case assigns half of the surface
spectral measure to the calculation. Equation (47) does not derive or
justify this factor: it is an inferred sensitivity observation because it
matches all 18 thick-solid Fig. 4 points with

- RMS difference \(0.1264\) pW/K;
- maximum difference \(0.2295\) pW/K;
- no lead regulator.

The half-weight result is non-normalized, is not derivable from the printed
equations, and is non-certifying. It must not be reported as exact
validation or as a same-input reproduction. Its purpose is to expose the
size and direction of the unresolved historical normalization ambiguity.

The approximation levels are kept distinct:

| Name | Cubic bubble after Hamiltonian mapping | Status |
|---|---:|---|
| `guo_2020_conserving_cubic_scba` | \(1\times\Sigma_{\rm Guo}\) | production |
| `mingo_2006_eq27_guo_audited_pi_over_9` | \((\pi/9)\times\Sigma_{\rm Guo}\) | historical, non-certifying |
| `mingo_2006_unverified_v3_input_x6_sensitivity` | \(4\pi\times\Sigma_{\rm Guo}\) | hypothesis only, not implemented |
| one-shot Born | one contact-only bubble and one updated solve | diagnostic |

The \(4\pi\) row records a specific unresolved input-normalization
hypothesis; it is not another accepted formalism. If historical code passed
the full derivative \(V'''\) into a numerical slot derived for the
Hamiltonian coefficient \(V^{(3)}=V'''/6\), its vertex would be six times
too large and its self-energy would be
\(6^2(\pi/9)\Sigma_{\rm Guo}=4\pi\Sigma_{\rm Guo}\). No primary code or
derivation supports that substitution. The descriptor is intentionally
marked `not_implemented_hypothesis_only`, and the factor must not be selected
to fit Fig. 8.

The vector values for the two quantum Fig. 8 curves are available through
`mingo_2006_figure8_curve`. They provide exact published checkpoints: the
displayed LJ reduction rises from about \(0.085\) at the lowest plotted
temperature to \(0.325\) at 542 K, while the displayed
SW\(\times13.207\) curve rises to \(0.297\). They do **not** by themselves
constitute an anharmonic solver reproduction.

No interacting Mingo SCBA solver or computed Fig. 8 curve is implemented.
The Mingo anharmonic reproduction is frozen as `historical_unresolved`. The
paper provides no code or raw numerical data, and the Fig. 4 transmission
alone does not uniquely determine the local Green functions entering the
cubic bubble.

One final primary-source search is capped at four developer-hours. It must
look specifically for original code, raw arrays, or an author statement that
uniquely resolves both the surface spectral normalization and whether the
numerical \(V'''\) input was converted to \(V^{(3)}=V'''/6\). If that
evidence is not found within the time box, no further candidate-factor sweep
is permitted. The normalized and half-weight surfaces, the factor-six vertex
hypothesis, and the production and legacy kernels must never be selected by
their agreement with Figs. 4 or 8.

If unique primary evidence is later obtained, a reopened closure must use the
source-fixed convention, the printed frequency grids, current conservation,
and grid/interpolation refinement, with no fitted spectral weight, vertex
multiplier, or diagrammatic coefficient. Until then, production Guo-SCBA
results must not be described as a reproduction of Mingo's legacy curve.

### 2.4 Qualitative historical trends

The following are appropriate qualitative regression targets:

- harmonic conductance scales as \(T^3\) at low temperature for the chosen
  surface spectrum;
- at \(k_m=0.316\) and \(k_l=3.16\), cubic anharmonicity reduces conductance,
  with the Lennard-Jones reduction much larger than the Stillinger-Weber
  reduction;
- for weak anharmonicity,
  \(\Delta\sigma/\sigma_h\) is quadratic in \(\beta\), and its
  high-temperature magnitude is approximately linear in temperature;
- sufficiently weak central bonds can instead show an anharmonic
  **increase** in conductance by opening an inelastic channel;
- the paper's cubic-only Lennard-Jones case ceases to converge above about
  600 K, consistent with the instability of a globally cubic potential.

## 3. Wang, Wang, and Zeng 2006

### 3.1 Convention map

Wang et al. set \(\hbar=k_B=1\), use mass-weighted coordinates
\(u_j=\sqrt{m_j}\,x_j\), and write

\[
H_n=\frac13\sum_{ijk}T_{ijk}u_i u_j u_k .
\]

For a fully symmetric tensor, the NEGFPy Hamiltonian convention gives

\[
\Psi_{ijk}=2T_{ijk}.
\]

The first term of Wang et al. Eq. (13) is

\[
\Sigma_{n,jk}^{</>}(\omega)=
2i\hbar\sum_{lmrs}T_{jlm}T_{rsk}
\int\frac{d\omega'}{2\pi}
G_{lr}^{</>}(\omega')G_{ms}^{</>}(\omega-\omega').
\]

Substitution of \(\Psi=2T\) makes this exactly the production
\(i\hbar\Psi\Psi/2\) bubble. Guo et al. independently state this
equivalence after accounting for the factor of two in the vertex.

Wang et al. Eq. (13) also contains a separate static/tadpole contribution.
Their stated mean-field calculation replaces \(G_0\) by the full \(G\) only
in the bubble diagram. NEGFPy `0.1.x` is therefore a conserving,
bubble-only SCBA benchmark; it is not a bit-for-bit implementation of every
diagram in that historical "mean-field" label. Tadpoles and thermal
expansion remain explicitly excluded from the production solver.

The historical validation layer names the non-equivalent choices separately:

| Name | Propagators and diagrams | Figure 2 |
|---|---|---|
| `wang_born_t2` | \(G_0\) bubble plus \(G_0\) static tadpole | \(O(T^2)\) |
| `wang_born_t4` | both \(O(T^2)\) graphs plus all six printed \(O(T^4)\) self-energy graphs | \(O(T^4)\) |
| `wang_selective_mean_field` | dressed bubble, but \(G_0\) tadpole | mean-field |
| `guo_bubble_one_shot` | \(G_0\) bubble only | not a Wang curve |
| `guo_bubble_scba` | dressed bubble only, conserving | not a Wang curve |

`wang_born_t2` and `wang_selective_mean_field` are numerically implemented in
the historical layer. Both reconstruct the dynamic retarded self-energy as
\(\Sigma^{++}-\Sigma^{+-}\) and add the real \(G_0\) static term.
`wang_selective_mean_field` iterates only the dynamic contour bubble, with an
unmixed fixed-point residual, common mixing of its contour components, and a
failure exception carrying the partial result. Neither path changes the Guo
production semantics or qualifies for production release certification.

The paper defines
\(I_L=-\langle\dot H_L\rangle\), positive from the left lead into the
device, matching NEGFPy's left-current sign. Its signed-frequency current
formula and NEGFPy's positive-frequency lesser/greater formula are
equivalent when the Keldysh identities hold.

### 3.2 Caption-literal and inferred author-legacy three-atom models

Figure 2 of the paper gives:

| Quantity | Value |
|---|---:|
| Central atoms | 3 |
| \(k_L\) | 1.56 eV/(angstrom\(^2\) amu) |
| \(k_R\) | 1.44 eV/(angstrom\(^2\) amu) |
| \(k_C\) | 1.38 eV/(angstrom\(^2\) amu) |
| \(k_L^{\rm onsite}\) | 0.01 eV/(angstrom\(^2\) amu) |
| \(k_R^{\rm onsite}\) | 0.02 eV/(angstrom\(^2\) amu) |
| Paper cubic \(t\) | 1.8 eV/(angstrom\(^3\) amu\(^{3/2}\)) |

The authors' [official NEGF code
page](https://phyweb.physics.nus.edu.sg/~phywjs/NEGF/negf.html) and archived
[`noneq.nb`](https://phyweb.physics.nus.edu.sg/~phywjs/NEGF/noneq.nb)
resolve the contact topology: the left/right lead-device spring is,
respectively, \(k_L/k_R\); internal device springs are \(k_C\). Therefore

\[
D_L^{00}=2k_L+k_L^{\rm onsite},\quad D_L^{01}=-k_L,
\]

\[
D_R^{00}=2k_R+k_R^{\rm onsite},\quad D_R^{01}=-k_R,
\]

and

\[
D_C=
\begin{pmatrix}
k_L+k_C & -k_C & 0\\
-k_C & 2k_C & -k_C\\
0 & -k_C & k_C+k_R
\end{pmatrix}.
\]

The author TeX retains source-level legacy values for the three main springs
beside the converted caption values:

\[
(k_L,k_R,k_C)=(2.5,2.3,2.2)
\quad[10^4\ {\rm dyn/(cm\,amu)}].
\]

It does not explicitly archive the onsite values in the legacy unit. Since
\(10^4\ {\rm dyn/cm}=10\ {\rm N/m}\), the exact conversion factor is

\[
1\ [10^4{\rm dyn/cm}]
=0.6241509074460763\ {\rm eV/\mathring A^2}.
\]

The vector EPS strongly favors the *hypothesis* that its onsite numbers
\(0.01,0.02\) were also expressed in the legacy unit: using the final caption
literally misses the 105.8 K ballistic path value by about 8.6%, whereas the
inferred onsite conversion gives close curve-level agreement. The EPS is an
output curve, however, so this inverse assignment is non-unique and must not
be described as exact recovery of the source inputs. NEGFPy therefore exposes
two non-interchangeable fixtures:

- `wang_2006_caption_literal_fixture()`;
- `wang_2006_inferred_author_legacy_figure2_fixture()`.

`wang_2006_author_legacy_figure2_fixture()` remains a compatibility alias for
the explicitly inferred fixture. Its old name is not evidence that the onsite
unit was archived.

The older `wang_2006_three_atom_fixture()` name remains an explicit alias for
the caption-literal interpretation; it never silently selects the inferred
model.

The paper writes

\[
H_n=\frac{t_{\rm paper}}{3}
\sum_{i=1}^{2}(u_i-u_{i+1})^3.
\]

The archived notebook instead uses
\(H_n=t_{\rm code}\sum_i(u_i-u_{i+1})^3\) and tensor entries
\(T=\pm3t_{\rm code}\). Its carbon-chain example has
\(t_{\rm code}\simeq0.6\), resolving

\[
t_{\rm paper}=3t_{\rm code}\simeq1.8.
\]

For an oriented bond \((i,i+1)\), the production tensor is consequently

\[
\Psi_{abc}=2t_{\rm paper}s_as_bs_c,\qquad
s_i=+1,\quad s_{i+1}=-1,
\]

with all ordered triples accumulated. If the generic NEGFPy
`fpu_alpha_chain` orientation \((u_{i+1}-u_i)^3/3!\) is used instead, its
argument must be `alpha=-2*t_paper`. The sign does not affect a pure cubic
bubble, but the model fixture retains it exactly.

Both fixtures implement these matrices, convert the SI vertex to solver
units, and supply a helper that converts kelvin to
\(k_BT/(\hbar\omega_0)\). Their solver reference scale is

\[
\omega_0=\sqrt{
\frac{1\ {\rm eV}}
{1\ {\rm angstrom}^2\,1\ {\rm amu}}
}.
\]

The fixtures are covered by exact block, contact,
Hamiltonian-polynomial, temperature-scaling, and no-silent-substitution tests.

### 3.3 Published Figure 2 source-limited reproduction status

The primary-source `kappa1D.eps` is a Grace vector plot with SHA256
`0e26c34ca70f3b132af8c6b3446d792c7a1ffd8b12a486f04df0c9db7185432b`.
Its plot rectangle maps \(x=0.15\ldots1.15\) to \(0\ldots1000\) K and
\(y=0.15\ldots0.85\) to \(0\ldots0.5\) nW/K. NEGFPy stores all 51
ballistic path points, 49 \(O(T^2)\) path points, 15 \(O(T^4)\) markers, and
56 mean-field path points after this affine conversion. These are vector
coordinates, not raster digitization.

The following quantitative checks pass, conditional on the explicitly
inferred onsite-unit convention:

- independent harmonic Landauer conductance over all 51 vector-path
  temperatures has RMS/max absolute errors of
  \(0.000124/0.000277\) nW/K;
- the complete Wang \(O(T^2)\) bubble-plus-tadpole calculation is within
  \(0.49\%\), \(0.27\%\), and \(0.30\%\) of the interpolated EPS path at
  100, 300, and 500 K, respectively, on the documented odd-grid counterpart
  of the archived calculation;
- the selective mean-field solver agrees at 300 and 400 K within
  \(0.28\%\) and \(0.75\%\), respectively. The 400 K check is the boundary of
  the moderate-temperature range for which the paper reports good behavior.

The notebook's separate \(0.05\ldots3.20\) output quadrature and fine-grid
contact-current integration bracket the published \(O(T^2)\) value within
about 0.5%. The automated historical checkpoint therefore uses a
source-limited 1% tolerance. This is not a fitted interaction prefactor.

The archived notebook uses \(\eta=10^{-4}\) in its frequency unit. The
historical interacting paths reproduce that choice for both contacts and the
device. A finite device regulator is an unaccounted dissipative bath, so every
historical result records it as such and remains non-certifying; production
Guo calculations continue to require `eta_device=0`.

The paper and archived notebook still do not identify one bitwise immutable
input for every curve. In particular:

- the 2007 notebook exposes \(\eta=10^{-4}\), a
  `makeGlistSymmetrize[20.5,1000]` grid, a 295/305 K bias example, and a
  separate \(0.05\)-spaced output integral, but does not prove that every
  setting is identical to the April 2006 plotted run;
- the notebook grid has an even point count and Mathematica rounding, whereas
  the production numerical contract requires an odd grid with exact zero;
- raw pre-plot conductance tables and numerical error estimates are absent;
- the six explicit \(O(T^4)\) diagrams have not yet been ported and
  independently tested. Their implementation is a nonblocking historical
  extension, not a production SCBA release requirement.

A fresh-start 500 K selective mean-field run did not converge in 200
iterations: adaptive mixing reached 0.01 with an unmixed self-energy residual
of \(4.6\times10^{-2}\). This is not counted as a reproduced point. It lies in
the regime where the paper explicitly says the cubic mean-field treatment
breaks down because the cubic potential is only metastable. The solver
therefore exposes deterministic nonconvergence with its partial result instead
of silently accepting the plotted high-temperature branch.

The remaining valid historical trend checks are:

- the cubic correction suppresses conductance increasingly above roughly
  200 K for the quoted parameter set;
- one-shot \(O(T^2)\), \(O(T^4)\), and the historical mean-field curves
  separate as temperature rises;
- the historical mean-field result agrees well with the \(O(T^4)\) result
  only up to roughly 400 K;
- the cubic-only approximation is not claimed to be stable at arbitrarily
  high temperature.

## 4. Release use

Historical checks are divided into three levels:

1. **Exact convention/fixture checks.** Hamiltonians, mass normalization,
   contact topology, unit conversion, tensor symmetry, and the
   \(\Psi=2T\) Wang prefactor map are deterministic release tests.
2. **Vector-curve checkpoints.** Ballistic, Wang \(O(T^2)\), and stable-range
   selective mean-field checks use the archived EPS hash, explicit affine
   extraction, the named inferred onsite convention, and a source-limited
   tolerance.
3. **Trend and breakdown checks.** The unimplemented \(O(T^4)\) curve remains
   a contextual comparison. Selective mean-field results above 400 K are
   treated as documented breakdown comparisons, not solver certification.

One-shot historical Born calculations are screening/regression tools. They
can test convention maps and weak-coupling trends, but they cannot certify a
nonzero-cubic interacting solver. Production certification requires a
converged Guo-convention SCBA fixed point and its independent conservation
and refinement diagnostics.

A future paper-reproduction artifact must archive its source files and
hashes, state whether quoted THz values are angular or cyclic frequencies,
use its predeclared method-specific prefactor without adjustment, report
regulator/grid/cutoff refinement, and publish machine-readable currents and
conservation diagnostics. Only the Guo prefactor can support production
certification; a source-fixed historical prefactor supports only its named
historical regression.

For this audit, the author-posted Wang arXiv v1 PDF and source archive and
the official `noneq.nb` were retrieved on 2026-07-24. Their local audit
SHA256 values were respectively
`e238818b720bc368e45298d9ab11e04c78564353e59d3f56677021f2574f345b`,
`5b9485665edb217fd67a4c9ef7c7ffacdd36a7ce497827698af56a7568da0ae0`,
and
`001aed1ef40d55dcc034244f054d21603a7603a95fb8d7113bdd1456b595f688`.
The Figure 2 EPS hash is
`0e26c34ca70f3b132af8c6b3446d792c7a1ffd8b12a486f04df0c9db7185432b`.
These downloaded audit files are not package runtime inputs.
