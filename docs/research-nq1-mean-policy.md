# Nq=1 centered-mean research convention

The current Nq=1 material calculation uses the declared policy
`centered-fixed-field-diagnostic`. The displacement Green functions describe
fluctuations about the supplied reference structure, and the one-point
displacement is prescribed as zero. The accepted equilibrium and 305/295 K
SCBA fields therefore define a centered bubble-SCBA research model.

This convention is suitable for continuing the reduced-coupling Nq=1 study.
It does not assert that the reference structure satisfies the physical static
force-balance equation of the cubic Hamiltonian. No static displacement,
linear counterforce, or interaction-induced stiffness was included in the
accepted material solves.

## Current calculation

For the recorded material diagnostics, report

\[
J_{\mathrm{centered}}=J_{\mathrm{harmonic}}+J_{\mathrm{cubic,connected}},
\qquad \langle u\rangle=\langle p\rangle=0.
\]

The cubic term is the connected one-vertex moment evaluated with the supplied
centered two-point fields. Disconnected mean terms are zero by the declared
model convention. The selected-current API requires the exact policy string
and records that the zero mean was prescribed, physical force balance was not
claimed, and a post-hoc static mean is not allowed.

This declaration does not retrofit a counterterm into archived SCBA fields.
It gives those fields a precise research interpretation while preserving all
existing convergence and validation flags.

## Why a static mean cannot be appended afterward

For

\[
H=\tfrac12 p^Tp+\tfrac12u^TKu+\tfrac16\Psi_{ijk}u_i u_j u_k,
\]

the leading Gaussian static force is

\[
f_i=-\tfrac12\Psi_{ijk}F_{jk}(0).
\]

Solving `K_static*mu=f` gives a displacement `mu=O(Psi)`. Expanding around
that displacement also produces an induced harmonic matrix

\[
\delta K_{ij}=\Psi_{ijk}\mu_k=O(\Psi^2).
\]

The disconnected current terms involving `mu` and the change of the harmonic
current caused by `delta K` are therefore at the same interaction order as
the bubble self-energy and connected cubic current. Adding only `mu` to the
accepted fields would mix approximation orders and would not represent either
the centered model or a self-consistent physical-mean model.

## Separate future physical-mean model

A physical-mean material calculation must be a new model and a new solve. It
must state the static boundary condition, constrain acoustic translations,
solve the one-point force balance, apply the induced stiffness to the device
and contacts consistently, and then recompute the two-point SCBA fields and
the complete current. The existing `static_cubic_shift` helper may diagnose
the leading force, mean, and induced stiffness, but its output must not be
inserted into accepted fields by itself.

This future calculation is needed before claiming relaxed-structure material
transport. It is not required to retain the present Nq=1 result as a clearly
labelled centered research approximation.
