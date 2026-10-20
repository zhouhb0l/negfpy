# Bounded coupled-q SCBA toy solver

This solver extends the
[coupled-q Born prototype](research-transverse-q-prototype.md) to a small
Dyson/Keldysh fixed-point calculation. It accepts a full transverse mesh,
the reduced cubic vertex, a harmonic dynamical matrix at every q, and
precomputed contact self-energies on the complete signed frequency grid.
It does not build material contacts or Fourier transform material force
constants. Independent one- and four-q self-consistent comparisons against
the existing dense real-space solver pass, together with focused physical,
iteration-control, and input-validation tests.

The existing Nq=1 material fields, fixed-broadening settings, and historical
acceptance records remain separate. These successful toy fixed points
establish results on their declared finite meshes and frequency grids.
They do not establish material transport convergence or grant an Nq=1 or
multi-q material release.

## Supplied problem

The API is in
[`transverse_solver.py`](../src/negfpy/anharmonic/transverse_solver.py):

```python
from negfpy.anharmonic.transverse_solver import (
    TransverseSpectrum,
    TransverseSCBAProblem,
    TransverseSCBAConfig,
    solve_transverse_scba,
)

left = TransverseSpectrum(left_r, left_lesser, left_greater)
right = TransverseSpectrum(right_r, right_lesser, right_greater)
problem = TransverseSCBAProblem(
    grid=grid,
    vertex=vertex,
    dynamical=dynamical_q,
    left=left,
    right=right,
)
result = solve_transverse_scba(
    problem,
    TransverseSCBAConfig(max_iterations=80, mixing=0.5),
)
```

Here `vertex` is the existing `TransverseCubicVertex`; the arrays are
supplied by the caller. An optional `initial_sigma=TransverseSpectrum(...)`
must satisfy the joint field checks and have a retarded component consistent
with the prescribed causal reconstruction. Otherwise iteration starts from
zero interaction self-energy. `TransverseSpectrum` owns read-only copies of
its three components.

`reconstruct_transverse_spectrum(grid, lesser, greater)` exposes the per-q
Hilbert reconstruction. `validate_transverse_spectrum(mesh, grid, spectrum)`
exposes the identities and correlation-positivity checks.
`transverse_heat_current(grid, contact, green, scale)` exposes the current
per transverse cell. The default `SolverScale` is the reduced-unit
convention \(\hbar=\omega_0=1\).

The momentum convention is unchanged:

\[
u_{Ra}=N_q^{-1/2}\sum_q e^{iq\cdot R}u_{qa}.
\]

The two transverse mesh periods are unrestricted positive integers within
the computational budget; the third period is one. The harmonic matrices
have shape `(Nq, D, D)`. Every retarded, lesser, and greater spectrum has
shape `(Nq, Nomega, D, D)`, on one common signed uniform frequency grid.
The reduced vertex has shape `(Nq, Nq, D, D, D)`.

The harmonic model must satisfy

\[
D(q)=D(q)^\dagger,\qquad D(-q)=D(q)^*.
\]

The input contacts supply all three components at every q and frequency.
The solver should not infer their negative-frequency half from a same-q
Nq=1 rule. Temperatures and any finite reservoir broadening belong to the
supplied contact model; the device Dyson equation introduces no additional
regulator.

## Fixed-point equations

For an evaluated input interaction self-energy \(\Sigma_{pp}\),

\[
G^r(q,\omega)=
\left[\omega^2I-D(q)-\Sigma_L^r(q,\omega)
-\Sigma_R^r(q,\omega)-\Sigma_{pp}^r(q,\omega)\right]^{-1},
\]

\[
G^{</>}(q,\omega)=G^r(q,\omega)
\left[\Sigma_L^{</>}+\Sigma_R^{</>}+\Sigma_{pp}^{</>}\right](q,\omega)
G^r(q,\omega)^\dagger.
\]

The existing coupled-q Born contraction computes a fresh lesser/greater
pair from this Green function. It retains the complete internal-q sum,
one factor \(1/N_q\), the second-vertex complex conjugate, and zero extension
in frequency.

Retarded reconstruction is performed independently for each q using the
existing finite-grid principal-value transform:

\[
\Sigma_{pp}^r(q,\omega)
=\tfrac12\left[\Sigma_{pp}^>-\Sigma_{pp}^<\right](q,\omega)
+i\,\mathcal P\int\frac{d\omega'}{2\pi}
\frac{\left[\Sigma_{pp}^>-\Sigma_{pp}^<\right](q,\omega')}
{\omega-\omega'}.
\]

The transform acts only on frequency and is linear in each matrix entry.
It therefore commutes with the transverse Fourier transform. No
same-q frequency-reflection assumption is needed to reuse
`retarded_from_keldysh`.

The iteration measures the fresh, unmixed map against the actual input
field. If another step is needed, the lesser and greater fields use the
same damping coefficient:

\[
\Sigma_{n+1}^{</>}=(1-\alpha)\Sigma_n^{</>}
+\alpha F^{</>}[\Sigma_n],\qquad 0<\alpha\leq1.
\]

The retarded field is then reconstructed from the mixed pair. This
preserves the required linear identities and preserves correlation
positivity when the input and fresh map satisfy it. It does not project
fields, clip eigenvalues, or force a proposed step to pass.

The returned state must be an evaluated state: its Green function,
interaction self-energy, currents, and raw-map residual all refer to the
same input field. A newly mixed but unevaluated field must not be returned
with the preceding Green function. Reaching an iteration or work limit is
reported separately from reaching a fixed point.

`result` contains `converged`, `reason`, `green`, `sigma_pp`, `currents`,
and immutable iteration `history`. A successful stop has reason
`"fixed_point"`; an exhausted iteration limit has reason `"max_iterations"`
and `converged=False`. Each history entry reports the iteration number and
`sigma_residual`, `current_residual`, and `identity_residual`.

These residuals are **tolerance-normalized errors**, not raw relative
changes. For each retarded/lesser/greater component, the fixed-point check
uses

\[
\rho_\Sigma =
\frac{\|F[\Sigma]-\Sigma\|_\infty}
{\texttt{sigma\_atol}+\texttt{sigma\_rtol}
\max(\|F[\Sigma]\|_\infty,\|\Sigma\|_\infty)}.
\]

The reported value is the maximum over all three components. It must not
exceed one. The current residual is the maximum magnitude of contact
balance, interaction collision power, and full balance, divided by
`current_atol + current_rtol * max(abs(J_left), abs(J_right))`. Both
fixed-point and current residuals must pass, and all field identity/PSD
checks must pass. `current_atol` is expressed in the returned current units
after `SolverScale`; its default is intended for reduced-unit toys.

The default iteration limit is 80, with mixing 0.5. A complete Born map
must fit the default budget of 100,000
`(q, qprime, omega, omega_prime)` tuples per contour component before any
Dyson or principal-value work begins. Exceeding that map budget rejects the
call. Each stored three-component spectrum is limited to 64 MiB; this is an
array-payload guard, not a bound on the entire process.

## Joint momentum and frequency checks

Lesser and greater fields are anti-Hermitian within each q block:

\[
C^{</>}(q,\omega)^\dagger=-C^{</>}(q,\omega).
\]

Their Hermitian correlation matrices \(iC^{</>}(q,\omega)\) must be
positive semidefinite. Checking the Hermitian part for numerical
eigenvalue noise does not replace the independent anti-Hermiticity check.
The Keldysh identity is also checked within each block:

\[
C^>(q,\omega)-C^<(q,\omega)
=C^r(q,\omega)-C^r(q,\omega)^\dagger.
\]

The signed-frequency constraints couple opposite momenta:

\[
C^<(q,-\omega)=C^>(-q,\omega)^T,\qquad
C^r(q,-\omega)=C^r(-q,\omega)^*.
\]

These relations apply to Green functions and self-energies. At a q point
that is its own negative they reduce to the existing Nq=1 relations. At
other q points they must be checked jointly. In particular, there is no
general reason to force a single q block to be real at zero frequency.

Correlation positivity remains a gate in nonequilibrium. Spectral
passivity of the interaction, total self-energy, or interacting Green
function is not a general nonequilibrium requirement and is not imposed.
The generic precomputed contact fields receive the same algebraic and PSD
checks, without a separate spectral-passivity or thermal-FDT gate. Thus
passing input validation does not certify that supplied contacts describe
passive thermal harmonic reservoirs; their model and causal validity are
the caller's responsibility.

## Current per transverse cell

For contact \(\alpha\), the sign is positive when energy enters the device.
With the existing solver current scale \(J_0=\hbar\omega_0^2\),

\[
J_\alpha=\frac{J_0}{N_q}
\sum_q\int_0^\infty\frac{d\omega}{2\pi}\,
\omega\,\operatorname{Re}\operatorname{Tr}
\left[\Sigma_\alpha^>G^<-\Sigma_\alpha^<G^>\right](q,\omega).
\]

The implementation uses strictly positive grid nodes and the existing
rectangle weight \(\Delta\omega/(2\pi)\). There is no additional factor
\(1/2\). The finite real-space supercell containing \(N_q\) transverse
cells has total current \(N_qJ_\alpha\). Comparing a supercell current
directly with the q average would therefore introduce a normalization
error.

Interaction collision power uses the same q average and frequency
integration, with \(\Sigma_{pp}\) in place of a contact. Scattering exchanges
energy among momentum channels, so individual q channels need not have
zero interaction collision power. A conservation screen concerns the sum
over the full mesh.

Report \(J_L+J_R\), \(P_{pp}\), and \(J_L+J_R+P_{pp}\) separately. The last
quantity checks accounting for the supplied fields; a small value by itself
does not establish a conserving fixed point. Currents use the input
self-energy whose Dyson/Keldysh solve produced the returned Green
function, not a fresh raw self-energy combined with an older Green.

## Contact-model and zero-frequency caveat

Precomputed contacts must use a declared model and a consistent zero-
frequency prescription. For example, an idealized Ohmic toy bath with
Hermitian positive-semidefinite \(\Lambda_\alpha(q)\) has

\[
\Sigma_\alpha^r=-i\omega\Lambda_\alpha,\qquad
\Sigma_\alpha^<=-2i\omega n_B(\omega,T_\alpha)\Lambda_\alpha,\qquad
\Sigma_\alpha^>=-2i\omega[1+n_B(\omega,T_\alpha)]\Lambda_\alpha.
\]

Both zero-frequency Keldysh limits are
\(-2iT_\alpha\Lambda_\alpha\) in dimensionless temperature units.
The existing gapped-contact builder's zero assignment does not supply
this Ohmic limit. Such a bath is a synthetic reservoir model and does not
represent the accepted material leads.

## Validation and remaining work

The final focused run passed all **20 tests in 11.50 seconds**, with no
warnings, failures, or skipped tests. Ruff also passed. These results cover
the final tested solver and test-file revisions.

The independent reference tests in
[`test_anharmonic_transverse_solver_reference.py`](../tests/test_anharmonic_transverse_solver_reference.py)
pass for one and four q points. They construct a small real-space supercell
with the same Hamiltonian and contact fields, iterate the existing dense
Born solver, and Fourier transform its converged fields and Green functions.
The coupled-q solver converges and agrees with those fields, Green
functions, and current per cell. The four-q case has genuinely complex
harmonic matrices and reduced cubic vertices.

Focused tests in
[`test_anharmonic_transverse_solver.py`](../tests/test_anharmonic_transverse_solver.py)
also pass. They check:

- Four-q harmonic Landauer current and current-unit scaling at zero cubic
  coupling.
- An equilibrium interacting fixed point with nonzero interaction fields
  and zero heat flow within the declared tolerances.
- Joint q/frequency reflection with a complex zero-frequency q block;
  incorrect opposite-q pairing and negative correlation eigenvalues are
  rejected without repairing the fields.
- Unmixed residuals that prevent false convergence at tiny mixing, and
  iteration-limit returns containing the evaluated input and its own Green
  function.
- A relative-only current tolerance when both currents vanish.
- Initial retarded fields inconsistent with causal reconstruction, work
  limits, harmonic reality, and invalid iteration configurations.

These checks complement the earlier Born-kernel tests, including
cell-origin covariance. They do not certify arbitrary supplied contact
fields as thermal, passive, or causal reservoirs.

The subsequent
[explicit real-space input adapter](research-transverse-q-mapping.md)
is implemented and passes 26 focused mapping/reference tests. It maps
finite-device cubic terms with transverse offsets, supplies declared FC2/FC3 unit conversions,
and joins harmonic providers and signed contact callbacks to this solver.
It includes a gapped harmonic-contact helper.

No test on a single chosen frequency grid establishes its cutoff or
quadrature accuracy. Raw material IFC-to-finite-device mapping, acoustic
contact handling, robustness at stronger coupling, scalable contraction,
restart support, and q-mesh convergence remain separate work. The adapter
has not yet mapped or solved an actual material dataset.
