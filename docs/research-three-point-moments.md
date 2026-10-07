# One-vertex cubic moments and explicit displacement means

The opt-in `three_point` module supplies the connected three-point moment
approximation needed by `build_cubic_cut_current_operator`. It can use Gaussian
harmonic lines or supplied dressed SCBA lines. Independent finite quantum
references verify the leading-vertex formula. A self-consistent material
spatial-current calculation and its stationary limit remain unverified.
The existing solver, accepted archives, cross-cut guards and acceptance gates
are unchanged.

## Connected response

Use the real canonical solver coordinates, units and equal-coordinate-leg
energy partition of [the cubic operator](research-cubic-cut-current.md). Let
`x=u-mu` and `delta p=p-nu`. Define centered, unsymmetrized correlations

\[
C_{ia}(\tau)=\langle x_i(\tau)x_a(0)\rangle=iG^>_{ia}(\tau),\qquad
P_{ia}(\tau)=\langle\delta p_i(\tau)x_a(0)\rangle=\partial_\tau C_{ia}(\tau).
\]

For a stationary Gaussian reference, first-order interaction-picture response
to a cubic vertex turned on at time zero is
`-i integral_0^T <[O(T),H3(s)]> ds`. Keep the Wick pairings connecting all
three external legs to the vertex, to obtain the third cumulant

\[
M^c_{ijk}(T)=2\operatorname{Im}\sum_{abc}\Psi_{abc}
 \int_0^T d\tau\,P_{ia}(\tau)C_{jb}(\tau)C_{kc}(\tau).
\]

This is the moment of `W(delta p_i*x_j*x_k)`, with the momentum symmetrized
against the coordinate product. The six cross pairings cancel the potential's
`1/6`. The code preserves every ordered input coefficient even if permutation
entries differ within input tolerance: it collects their commuting potential
monomial, then evaluates all six assignments, retaining repeated-leg factors.
No dense cubic tensor is formed for sparse inputs.

Writing `C=F+iR/2`, where `F` is the symmetrized unequal-time covariance and
`R=G^R` at positive lag, gives the kernel

\[
\Psi\left(\dot RFF+\dot FRF+\dot FFR-\tfrac14\dot RRR\right).
\]

The final term is quantum. Discarding it fails the vacuum oscillator reference.
This kernel is derived here. Green-function conventions and the distinction
between two-point scattering and one-point thermal displacement are supported
by [Wang et al., equations 25–29 and 116–117](https://arxiv.org/pdf/1303.7317).

Using dressed lines in this one-vertex expression defines a skeleton
approximation: the moment is first order in the vertex at fixed lines, so
its contribution to cubic current is second order. This matches the dynamic
bubble's interaction order; it does not by itself prove a conserving dressed
closure, solve vertex corrections, or recover omitted tadpoles.

## Means, tadpoles and harmonic feedback

The exact decomposition of the raw third Weyl moment is

\[
\langle W(p_i u_j u_k)\rangle=M^c_{ijk}
 +\nu_i F_{jk}(0)+\mu_j Q_{ik}+\mu_k Q_{ij}+\nu_i\mu_j\mu_k,
\]

where `Q_ij=<W(delta p_i*x_j)>`. `combine_cubic_moments` requires all means
explicitly. At stationarity `nu=0`, but the two `mu*Q` terms can remain.
Finite-history connected moments need means from that same history; adding
a static mean to a finite quench is not a consistent transient calculation.
Harmonic raw `p*u` moments also need `nu_i*mu_j` when velocity means are nonzero.

The leading static mean force is `f_i=-Psi_ijk F_jk(0)/2` for an exactly
symmetric tensor. `static_cubic_shift` differentiates every polynomial leg
and solves `K_static*mu=f` without changing any solver state. It requires an
explicit real positive-definite stiffness. Relaxed harmonic contacts require
their static Schur complement, `K_static=D+Sigma_contacts^R(0)`; device `D`
alone instead describes clamped contact displacements. The helper neither
infers the boundary condition from a finite-regulator Green matrix nor
pseudoinverts acoustic zero modes. Constraints must be provided explicitly.

The mean is `O(Psi)` and induces `delta D_ij=Psi_ijk*mu_k=O(Psi^2)`.
The helper returns this `induced_stiffness` for diagnosis but does not apply it.
Both mean-dependent cubic current and the change in harmonic current occur
at the dynamic bubble's order. Thus appending the inferred mean to existing
bubble-only fields is not a full tadpole-consistent Hamiltonian calculation.

Production SCBA explicitly omits cubic tadpoles/static equilibrium shifts
([theory scope](anharmonic_theory.md)). Explicit zero means can describe its
centered approximation; they do not establish physical force balance. A
centered physical model generally needs a compensating linear force. Such a
model, including its energy partition, must be stated before conservation is
claimed. This module does not silently impose a counterterm or revise the model.

## API and numerical limits

```python
from negfpy.anharmonic import (
    build_cubic_cut_current_operator, connected_cubic_moments,
    correlations_from_green, combine_cubic_moments,
)

operator = build_cubic_cut_current_operator(problem, cut=1)
lines = correlations_from_green(problem.grid, green, lag_times)
connected = connected_cubic_moments(
    problem.vertex, operator.cubic_indices, lag_times,
    lines['displacement_correlation'],
    lines['velocity_displacement_correlation'],
)
# These means/covariances must describe the same state/history as the lines.
moments = combine_cubic_moments(
    operator.cubic_indices, connected['connected_moments'],
    displacement_mean=mu, velocity_mean=nu,
    displacement_covariance=F0,
    velocity_displacement_covariance=Q,
)
current = operator.evaluate_moments(harmonic_raw_moments, moments['total_moments'])
```

The Fourier helper evaluates signed-grid rectangle sums of `i G>` and
`omega G>`. It expects centered Green functions without a mean delta at zero.
No FFT wraparound, artificial damping or exterior frequency extension is added.
Discrete frequency sums recur at `2*pi/grid.step`; the helper rejects lag
windows reaching half that period. Passing this guard does not establish
decay, adequate frequency resolution or canonical commutators.

The connected integrator uses the supplied finite lag interval and trapezoids.
It returns the last integrand value for inspection but gives no tail estimate
or accuracy pass. For a steady state, memory-window and time-step convergence
must be checked alongside frequency range/resolution and mean consistency.
For fully converged memory, the cyclic sum `M_ijk+M_jik+M_kij` vanishes; at
finite memory it contains an endpoint term, which the tests reproduce. A small
endpoint by itself is not a tail bound.

The implementation is a small-reference and selected-moment tool. It costs
`O(N_lag*N_requested_moments*N_vertex_monomials)` operations and stores two
`O(N_lag*ndof^2)` line arrays. Contraction temporaries are tiled over 256
monomials. Full material evaluation needs a resource estimate and potentially
pair selection or streamed contraction. All moments remain in solver units;
only the cut operator converts the final current to physical units.

## Independent checks and next use

The quantum reference uses a rotated two-mode harmonic vacuum with six basis
states per mode. It calculates finite-time matrix commutators directly and
subtracts means obtained from independent one-point commutators. It also
integrates matrix phases analytically and differentiates exact finite-coupling
Hamiltonian quenches. These are nonzero transient tests, not equilibrium zero
current checks. Finite oscillator truncation is harmless for the degree-six
vacuum matrix products; finite-coupling results are used only in the weak
perturbation comparison.

Separate tests cover Fourier signs/normalization, a finite-memory derivative
identity, time refinement, sparse storage, repeated indices, explicit
disconnected means and the static force/feedback. The next practical step is
a small open-system steady-state test of the dressed approximation, with a
declared mean policy, memory convergence and contact/cut continuity, before
using the accepted material fields. No SCBA rerun or cluster job is launched
by these APIs. Spatial validation and Nq=1 release remain pending.
