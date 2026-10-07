# Research diagnostic for layer energy balance

`evaluate_layer_energy_balance` evaluates the layer-projected Dyson equation
using existing Green functions and self-energies. It accepts cubic vertices
that cross layer boundaries and reports where harmonic flow and interaction
exchange balance. It does not provide the missing full cubic Hamiltonian
current operator and does not certify spatial transport or change SCBA
acceptance.

The API is opt-in and returns a JSON-compatible report plus numerical arrays:

```python
from negfpy.anharmonic.research_layer_balance import evaluate_layer_energy_balance

report, arrays = evaluate_layer_energy_balance(
    problem, sigma_left, sigma_right, sigma_pp, green,
    eta_device=0.0, rtol=1e-5,
)
```

All inputs must describe the same problem and retained state. The function
does not rerun SCBA, refine its input frequency grid, or update any field.
Its arrays contain strictly positive frequencies. Spectra are in physical
current units per dimensionless angular-frequency unit, and integrals use
the original grid's rectangle weight `step / (2*pi)`.

## Equation and signs

For each self-energy source \(a\in\{L,R,pp\}\), define

\[
X_a=\Sigma_a^rG^<+\Sigma_a^<G^a-G^r\Sigma_a^<-G^<\Sigma_a^a,
\qquad C_{a,I}(\omega)=J_0\omega\operatorname{Re}\operatorname{Tr}(P_I X_a).
\]

Here \(P_I\) is the layer projector and \(J_0\) is the solver's physical
current scale. A contact current is positive into the device. For a cut
between layers \(I\) and \(I+1\), the harmonic contribution is oriented
left to right:

\[
J^h_I(\omega)=-J_0\omega\operatorname{Re}\operatorname{Tr}
\left[D_{I,I+1}G^<_{I+1,I}-G^<_{I,I+1}D_{I+1,I}\right].
\]

This harmonic contribution is defined even when the cubic vertex crosses a
cut. It then omits a Hamiltonian interaction-current contribution. The new
`harmonic_cut_current_contribution_spectra` name makes that scope explicit.
The existing `bond_heat_current_spectra` API retains its cross-cut rejection.

For zero device regulator, subtracting the left and right lesser Dyson
equations gives \(-[D,G^<]=X_L+X_R+X_{pp}\). Thus, with the two exterior
harmonic boundaries set to zero, the diagnostic residual is

\[
R_I(\omega)=\operatorname{diff}
\left[0,J^h_1(\omega),\ldots,J^h_{N-1}(\omega),0\right]_I
-C_{L,I}(\omega)-C_{R,I}(\omega)-C_{pp,I}(\omega).
\]

Projecting both contact sources at their actual device indices permits
contacts on interior layers or across several layers. Substituting contact
totals at the outer boundaries would generally be wrong for those cases.
Nonzero `eta_device` is rejected because its extra regulator term is not
represented by these three sources.

The report retains the distinct layer collision spectrum

\[
P_{pp,I}(\omega)=J_0\omega\operatorname{Re}\operatorname{Tr}
\left[P_I(\Sigma_{pp}^>G^<-\Sigma_{pp}^<G^>)\right].
\]

Its layer sum recovers the global collision spectrum. Given the Keldysh
spectral identities, the summed EOM projection does too. Individual layer
EOM and collision terms can differ for nonlocal self-energies: cyclic trace
rearrangement does not apply across a layer projector. Neither is required
to vanish independently in every layer. Layer-projected collision exchange
is also used as a descriptive scattering diagnostic in Guo et al., equations
(9)–(13), [arXiv:2103.10087](https://arxiv.org/pdf/2103.10087).

## Acceptance and its limits

`spectral_balance_passed` and `integrated_balance_passed` compare the largest
absolute EOM residual with `atol + rtol * scale`. Each scale is the largest
absolute harmonic-divergence, individual-contact-EOM, or interaction-EOM
term in that comparison. Spectral and integrated comparisons use their own
scales. The default absolute tolerance is `1e-10 * problem.scale.current_scale`.
`eom_balance_passed` combines these two equation checks only.

The report always identifies `equation_balance_only=True` and preserves
`full_spatial_current_available=False`, `spatial_validation_passed=False`,
`changes_solver_acceptance=False`, and
`global_frequency_accuracy_established=False`. It supplies no generic
`passed` flag. No cumulative EOM correction is presented as a measured or
independently derived cubic spatial current.

A one-shot Born state can pass this EOM identity because its Green function
was solved from the same self-energy. Passing therefore establishes internal
equation consistency, not a self-consistent fixed point, a conserving
interaction approximation, frequency convergence, or an independently
validated spatial heat flux. Establishing the complete cubic cut current
still requires a stated local-energy partition and the corresponding
interaction-current expectation values. Cubic forces generate terms
involving one velocity and two displacements. General heat-current operators
contain higher-order terms beyond their harmonic contribution; see the
Hamiltonian and current discussion in section II of
[Phys. Rev. B 97, 104308](https://link.aps.org/accepted/10.1103/PhysRevB.97.104308).

The focused tests compare harmonic divergence with an independent full dense
commutator, check the harmonic Landauer/contact limit and a nonzero cross-cut
Born state, place contacts on interior layers, recover global collision power
from layer projections, and deliberately perturb a Green function so the
equation residual fails. They also verify that cross-cut spatial certification
remains unavailable and that an unrepresented device regulator is rejected.

## Diagnose an archived current comparison

`compare_current_quadrature` attributes an existing refined-versus-rectangle
current difference to the original frequency intervals. It requires only
stored current spectra and the final adaptive current leaves; it evaluates
no Green function or SCBA map.

```python
from negfpy.anharmonic import compare_current_quadrature

comparison = compare_current_quadrature(
    positive_frequencies=positive_frequencies,
    integration_weight=problem.grid.integration_weight,
    contact_spectra=currents.contact_spectra,
    collision_spectrum=currents.collision_spectrum,
    current_leaves=diagnostic.arrays,
)
```

Authenticate closed archives before loading them. The helper checks complete
coverage of every original interval, reconstructs the right-endpoint rectangle
rule and GL16 sums, and reports signed differences, sampled GL8/16 errors and
interval rankings. The rectangle spectra exclude `1/(2*pi)` while the leaf
integrals already include it; the helper applies that factor exactly once.
An incomplete or overlapping partition raises an error. All prior convergence
and acceptance flags remain unchanged.
