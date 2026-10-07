# Cubic Hamiltonian heat current across a layer cut

`build_cubic_cut_current_operator` implements the internal harmonic plus cubic
heat-current operator for real canonical device coordinates. It fixes a local
energy partition, preserves repeated cubic-index multiplicities, and evaluates
explicitly supplied symmetrized moments. The sparse vertex path uses sparse
coordinate monomials without constructing a dense cubic tensor.

This implements the operator missing from the layer EOM diagnostic. Computing
its expectation in the accepted material SCBA state remains separate: a
consistent three-point moment calculation is not yet implemented. Existing
SCBA acceptance, cross-cut guards and spatial-validation flags are unchanged.

## Partition and derivation

Use dimensionless mass-normalized coordinates with `[u_i,p_j]=i delta_ij` and

\[
H=\tfrac12\sum_i p_i^2+\tfrac12\sum_{ij}D_{ij}u_i u_j
+\tfrac16\sum_{ijk}\Psi_{ijk}u_i u_j u_k.
\]

Let `s_i=1` for coordinates in the layers to the left of the cut, and zero
otherwise. Assign each potential monomial equally to its coordinate legs,
counting repeated indices. The left energy is

\[
H_A=\tfrac12\sum_i s_i p_i^2
+\tfrac14\sum_{ij}(s_i+s_j)D_{ij}u_i u_j
+\tfrac1{18}\sum_{ijk}(s_i+s_j+s_k)\Psi_{ijk}u_i u_j u_k.
\]

Together with its complement, this recovers the full device Hamiltonian.
The outward internal current is `J_A=-dH_A/dtau=-i[H,H_A]`. Because all
potential terms commute, only kinetic-potential commutators contribute:

\[
J_A^{(2)}=\tfrac12\sum_{ij}(s_i-s_j)D_{ij}W(p_i u_j),
\qquad
J_A^{(3)}=\tfrac16\sum_{ijk}(2s_i-s_j-s_k)\Psi_{ijk}W(p_i u_j u_k).
\]

Here `W(p_i u_j u_k)={p_i,u_j u_k}/2` equals full Weyl ordering, including
repeated indices. Both pieces are Hermitian. Terms wholly on either side
give zero cut current, and complementing the region reverses the sign.

The local-energy/commutator construction and symmetrized harmonic limit are
supported by [Wu and Segal, sections II and IV](https://arxiv.org/html/0804.3371v1).
The cubic coefficients above are derived here for the stated equal-leg
partition; they are not a quoted formula from that source.

The compiler includes every ordered input coefficient as `Psi_abc/6` in its
commuting potential monomial. It then differentiates each coordinate leg,
including duplicates. Thus accepted last-bit differences between permutation
entries enter their actual polynomial sum; no input entry is replaced by an
averaged or selected representative.

For a repeated triple `(i,i,j)` with only `i` on the left, the contribution is

\[
\tfrac{\Psi_{iij}}3\left[W(p_i u_i u_j)-W(p_j u_i^2)\right].
\]

For the two-site FPU term `alpha*(u_1-u_0)^3/6`, this partition gives the
classical cubic current

\[
J_A^{(3)}=\tfrac{\alpha}{3}(u_1-u_0)(u_0p_1-u_1p_0).
\]

It differs instantaneously from assigning half the entire bond energy to each
site. A change of local-energy partition changes the current by a time
derivative. Equal steady-state averages require the corresponding partition
energy to be stationary; the implementation does not assume this for arbitrary
supplied states.

## Use and units

```python
from negfpy.anharmonic import build_cubic_cut_current_operator

operator = build_cubic_cut_current_operator(problem, cut=1)
two_point = operator.harmonic_moments_from_green(problem.grid, green)

# Evaluate each required <{p_i, u_j*u_k}/2> in the same state.
# Columns in cubic_indices are momentum i, displacement j, displacement k.
three_point = moment_provider(operator.cubic_indices)
currents = operator.evaluate_moments(two_point, three_point)
```

`cut` counts left layers, so valid values are 1 through `n_layers-1`.
Positive current leaves those layers toward the right. The returned `harmonic`,
`cubic`, and `total` currents include `problem.scale.current_scale =
hbar*omega0**2` exactly once. The corresponding `_solver` fields are
dimensionless. Displacements, velocities and supplied moments must be in
solver units. For a classical instantaneous state use
`operator.evaluate_phase_space(displacement, velocity)` with
`velocity=du/dtau=p`; this is not a quantum transport evaluation.

The harmonic moment helper follows the repository convention
`G<_ij(t,t')=-i <u_j(t')u_i(t)>`:

\[
\langle W(p_i u_j)\rangle=\mathrm{Re}\int\frac{d\omega}{2\pi}
\omega G^<_{ij}(\omega).
\]

It uses the supplied signed-grid rectangle quadrature and does not establish
frequency accuracy. Quantum and classical evaluation require moments for
every compiled term. Missing, incorrectly shaped, complex or nonfinite moment
arrays are rejected; cubic moments are never silently filled with zeros.

Only real-space canonical coordinates are supported. A complex Bloch matrix
requires a treatment of paired q and -q coordinates and is explicitly rejected.
The operator covers internal device transport. Reservoir energy injection
must be included separately when checking open-system layer continuity.

## Reference checks and remaining integration

Independent tests assemble the full and partitioned Hamiltonians before
differentiating or commuting them. Classical tests use complex-step potential
gradients. Quantum tests use three oscillator modes with seven basis states
per mode and compare the matrix block with occupations at most two. This
keeps the reference clear of the truncated oscillator commutator at its top
state. Removing the cubic piece produces a measurable disagreement.

Further checks cover a three-layer vertex, repeated indices, the analytic
FPU result, current direction, physical scaling, empty/intralayer support,
sparse/dense agreement, and the existing harmonic Green-function current.

Two-point SCBA fields alone do not supply independent three-point expectations.
A centered Gaussian substitution would set odd moments to zero and omit the
cubic current at the same interaction order as Born self-energy effects.
The next step is therefore a stated three-point approximation, consistent with
the solver's treatment of static shifts and disconnected mean terms, followed
by a small reference comparison. The current module always records
`scba_three_point_closure_implemented=False`, `state_moments_verified=False`,
`spatial_validation_passed=False` and `changes_solver_acceptance=False`.
`full_operator_included=True` describes which terms were contracted; it is
not a material-conservation or frequency-convergence result.
