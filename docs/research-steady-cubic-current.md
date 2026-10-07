# Steady-state checks of the cubic current approximation

The one-vertex moments now pass a biased quantum steady-state toy test and
an independent classical perturbative reference. Both require the cubic
operator to match internal current to reservoir energy input. This verifies
the specified small centered models; it does not yet validate material
spatial current or release Nq=1.

## Quantum centered bubble reference

`scripts/validate_cubic_steady_current.py` uses three scalar coordinates,
unit solver scales and

\[
D=\begin{pmatrix}1.4&-0.25&0\\-0.25&1.2&-0.22\\0&-0.22&1.3\end{pmatrix},
\qquad H_3=\frac{\alpha}{6}\big[(u_1-u_0)^3+(u_2-u_1)^3\big],
\quad\alpha=0.1.
\]

The end coordinates couple to separate Drude reservoirs, with temperatures
1.0 and 0.5, damping `gamma=0.25` and cutoff `Omega=3`:

\[
\Sigma^R(\omega)=-\frac{i\gamma\omega\Omega}{\Omega-i\omega},\qquad
\Sigma^<(\omega)=-2i\gamma\frac{\Omega^2}{\Omega^2+\omega^2}\omega n_B(\omega,T),
\qquad\Sigma^>-\Sigma^<=2i\operatorname{Im}\Sigma^R.
\]

At zero frequency `omega*n_B` is evaluated as its limit `T`.
The static bath renormalization is included: `SigmaR(0)=0`, so `D` is the
declared static stiffness. Its high-frequency constant `gamma*Omega` is
intentional. Contact spectra are analytic and are not reconstructed by an
unsubtracted Hilbert transform. This convention follows the reservoir
self-energy with renormalization in
[He et al., equations 21–26](https://arxiv.org/pdf/1607.04382).
Only the bath convention is used from that paper, not its anharmonic closure.

The small reference iterates the existing dynamic bubble and its causal
retarded reconstruction, starting from zero interaction self-energy. Mixing
is 0.5; at most 64 maps are allowed. A freshly computed raw map must differ
from its input by less than `1e-12` in maximum absolute norm. The stored Green
functions are those evaluated from that input, not a subsequent untested
update. A cap or nonfinite result raises. This simple toy driver does not
replace the production research controller or its acceptance criteria.

The mean policy is explicit. A fixed local linear potential

\[
H_{\rm lin}=\sum_i h_i u_i,\qquad
h_i=-\tfrac12\sum_{jk}\Psi_{ijk}F_{jk},
\]

is prescribed from the converged centered covariance. It cancels the cubic
mean force in this centered approximation, with `mu=nu=0`. No physical
thermal displacement is inferred. Assigning `h_i*u_i` to coordinate i makes
its internal cut-current contribution cancel between kinetic and potential
energy derivatives; there is no omitted cross-cut linear operator. The
coefficients are fixed for each reference state. Reversing bias or changing
temperature reconstructs this counterterm, so those are separate centered
references rather than unchanged-Hamiltonian bias reversals.

Dressed greater Green functions feed the existing time-domain one-vertex
moment integral. Explicit zero means enter `combine_cubic_moments`; the full
harmonic-plus-cubic operator then gives two independent cuts. Contact currents
come from the reservoir collision spectra. No projected EOM cut correction
is substituted for the cubic Hamiltonian operator.

For the refined quantum case (`Delta omega=.005`, frequency endpoints `+/-12`,
memory 150, lag step .025), the reservoir current is
`0.015728379371879854`. The full cut currents are
`0.01572835859561749` and `0.015728333787699174`; the maximum relative mismatch
is `2.899e-6` (0.000290%). The cubic contributions are
`-0.00019023843916090032` and `-0.0001675378949613292`.
Omitting them leaves approximately 1.21% and 1.06% mismatches.

Separate checks extend memory 75 to 150, halve lag step .05 to .025,
halve frequency step .01 to .005, and extend frequency endpoints 8 to 12.
Each changes total current by less than `1e-4` of contact current and cubic
current by less than `1e-3` of the larger cubic component. The full contact/cut
screen is `1e-3`; the harmonic-only control fails it. Harmonic, equilibrium and
reversed-bias controls also pass. These are toy criteria, separate from all
frozen material screens, which are untouched.

The finite-grid canonical spectral-moment defect is reported separately:
it is approximately `8.91e-4` at the baseline and `2.70e-4` after frequency
domain refinement. It is larger than the current continuity residual.
Neither the small continuity residual nor the refinements are rigorous
infinite-domain or memory-tail error bounds.

## Independent classical stationarity hierarchy

`tests/test_anharmonic_steady_classical_reference.py` builds a different
three-site model and never uses Dyson equations or the Born self-energy to
generate its stationary moments. Let `z=(u,p)` follow a stable Gaussian
Langevin drift with matrix `A`, covariance `S0` and noise `B`:

\[
A S_0+S_0 A^T+B=0.
\]

Add the centered cubic force
`b_p_a=-(g/2) Psi_aij (u_i*u_j-S0_ij)`. The subtraction is a fixed local
counterforce, cancelling the leading tadpole. The third-moment coefficient
`T1` solves the Kronecker-sum stationarity equation `K3*T1=-F`, where

\[
F_{abc}=-\delta_{a,p_\ell}\Psi_{\ell ij}(S_0)_{ib}(S_0)_{jc}
       +\text{the other two external-leg placements}.
\]

The covariance correction `S2` solves

\[
A S_2+S_2 A^T=-Q,\qquad
Q_{ab}=-\tfrac12\delta_{a,p_\ell}\Psi_{\ell ij}(T_1)_{ijb}+(a\leftrightarrow b).
\]

Contract the cut operator using covariance `S0+g^2*S2` and third moment
`g*T1`. The two full cuts equal the independently calculated classical
reservoir powers `gamma_i*(T_i-<p_i^2>)` within `1.25e-16`, through order
`g^2`. Both cubic components are nonzero; omitting them breaks continuity.
The classical reservoir-power formula is not applied to the quantum model.

An independent Gaussian propagator supplies time correlations for the public
three-point API. Its classical limit is evaluated with formal
`C=F+i*epsilon*R/2`, followed by division of the moment by `epsilon`.
Here `epsilon=1e-4` is a formal Planck constant, not damping and not a model
of a quantum thermal bath. The residual quantum term is `O(epsilon^2)`;
halving epsilon estimates its contribution at `4.07e-11`. Time quadrature is
the larger error: the moment differs from the exact stationary hierarchy by
`2.925e-7`, and halving the timestep improves it fourfold. Extending memory
20 to 120 removes the resolved memory transient.

This is a formal classical perturbative expansion around a stable Gaussian
state, not a global stationary distribution for an unrestricted cubic
potential. Higher-order mean and feedback effects remain outside this
reference. Likewise, the quantum test is a numerical centered-bubble
consistency check rather than a general proof of conserving SCBA closure.

## Reproduction and next material step

Run the quantum campaign with `src` on `PYTHONPATH` and one BLAS thread:

```bash
python scripts/validate_cubic_steady_current.py --output quantum-steady-reference.json
```

The classical test module exposes `build_reference()` returning JSON-compatible
evidence. Run both new test modules along with the existing current and
three-point regressions. The development report contains both saved reference
outputs and their source hashes.

The next material step is to prepare a bounded fixed-field selected-cut
diagnostic, estimating its memory/work from the actual sparse topology and
checking whether the available frequency representation resolves the memory
window. The material mean policy must also be stated; the centered toy does
not retroactively establish a counterterm in the accepted material Hamiltonian.
Any missing displacement treatment remains an explicit limitation. Accepted
SCBA convergence and existing transport failures remain unchanged. No material
job, repeated SCBA calculation or automatic follow-on is launched here.
