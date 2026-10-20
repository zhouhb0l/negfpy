# Coupled transverse-q Born prototype

This research prototype evaluates a cubic lesser/greater self-energy for a
supplied set of Green functions on a complete transverse momentum mesh. It
implements the coupled momentum sum required by
[the roadmap, transverse periodicity](anharmonic_negf_roadmap.md#35-transverse-periodicity).
It does not solve Dyson or self-consistent Born equations, construct a
material device, or establish convergence of material transport.

The accepted Nq=1 source and saved X14 benchmark are separate from this
prototype. No saved material fields need to be loaded, transformed, or
recomputed for its synthetic tests. The prototype does not change the chosen
material broadening or the interpretation of any earlier result.

## Data and public API

```python
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_q import (
    TransverseCubicVertex,
    cubic_born_self_energy_q,
    difference_q_indices,
)

mesh = UniformBZMesh((4, 1, 1))
vertex = TransverseCubicVertex(mesh, coefficients)
qminus = difference_q_indices(mesh)
sigma_lesser, sigma_greater = cubic_born_self_energy_q(
    grid, vertex, g_lesser, g_greater
)
```

`coefficients` has shape `(Nq, Nq, D, D, D)`, with indices
`[qout, qprime, a, c, d]`. Each Green function and returned self-energy has
shape `(Nq, Nomega, D, D)`. `qminus` has shape `(Nq, Nq)` and gives the
integer index of `qout - qprime`, folded to the complete mesh.

The first two mesh axes are transverse. The third mesh period must be one;
the code does not Fourier transform the transport direction. The mesh uses
the X-fast index ordering of
[`momentum.py`](../src/negfpy/anharmonic/momentum.py). No irreducible-mesh
weights or point-group reductions are included. The common frequency grid is
the signed, uniform grid used by
[`cubic.py`](../src/negfpy/anharmonic/cubic.py).

## Fourier convention

Let \(R,S,T\) label cells of a finite periodic transverse lattice, with
\(N=N_q\) cells and \(D\) internal real displacement coordinates per cell.
Write \(q\cdot R=2\pi\sum_{\mu=1}^2 k_\mu R_\mu/n_\mu\), where the mesh
address \(k_\mu\) may be any integer representative of its residue class.
The cell-periodic convention has no additional atomic-position phase:

\[
u_{Ra}=\frac{1}{\sqrt N}\sum_q e^{iq\cdot R}u_{qa},
\qquad
u_{qa}=\frac{1}{\sqrt N}\sum_R e^{-iq\cdot R}u_{Ra}.
\]

For a translation-invariant Green function,

\[
G_{Ra,Sb}(\omega)=\frac1N\sum_q
e^{iq\cdot(R-S)}G_{ab}(q,\omega).
\]

The same transform applies to the self-energy. With
\(U_{qR}=e^{-iq\cdot R}/\sqrt N\), this is the block diagonalization
\(\widetilde G=(U\otimes I_D)G(U^\dagger\otimes I_D)\).

The real-space Hamiltonian convention is the existing one:

\[
H_3=\frac1{3!}\sum_{Ra,Sc,Td}
\Psi_{Ra,Sc,Td}u_{Ra}u_{Sc}u_{Td}.
\]

For real, translation-invariant force constants define
\(\phi_{acd}(s,t)=\Psi_{0a,sc,td}\). The reduced vertex is

\[
V_{acd}(q,p)=\sum_{s,t}\phi_{acd}(s,t)
e^{i[p\cdot s+(q-p)\cdot t]}.
\]

It contains no factor \(1/\sqrt N\) or \(1/N\). Substitution into the
Hamiltonian gives one factor \(1/\sqrt N\) for each momentum-conserving
triplet. The vertex's all-incoming momenta are
\((-q,p,q-p)\). When using the existing all-incoming router, its target must
be `mesh.negative_index(qout)`, rather than `qout`. The prototype's direct
integer difference table gives the same second internal momentum.

The formula uses integer cell coordinates. Reciprocal shifts of a momentum
representative therefore leave its phase unchanged. The canonical
normal/Umklapp label records whether the three selected representative
addresses sum to zero or a nonzero reciprocal vector. Both classes are
included with the same full-mesh weight.

## Coupled Born contraction

For either contour component \(x=<\) or \(x=>\),

\[
\Sigma^x_{ab}(q,\omega)=\frac{i}{2N}
\sum_{p,cdef}V_{acd}(q,p)V_{bef}(q,p)^*
\int\frac{d\omega'}{2\pi}
G^x_{ce}(p,\omega')G^x_{df}(q-p,\omega-\omega').
\]

Only the second vertex is conjugated. Neither Green function is conjugated
or transposed in this expression. This follows directly by substituting the
inverse transforms into the existing real-space Born contraction. At the
first vertex the phases are \(e^{i(p\cdot s+(q-p)\cdot t)}\); at the second
they are the negatives. A real real-space vertex makes that second
transform the complex conjugate of the first.

The two inverse Green transforms supply \(1/N^2\), while the transform over
the relative output cell supplies \(N\) and selects \(q=p+(q-p)\) modulo the
mesh. The resulting \(1/N\) is inserted exactly once. The \(1/2\) is the
existing cubic bubble factor; all ordered internal momenta are retained.
There is no additional factor for exchanging the two internal lines.

The frequency quadrature is independently normalized by
`grid.integration_weight`, namely \(\Delta\omega/(2\pi)\). For output index
\(i\), internal index \(j\), and zero-frequency index \(z\), the second
frequency index is \(i-j+z\). Terms outside the supplied frequency grid are
zero. Momentum wraps periodically; frequency does not. This distinction is
essential: a circular frequency convolution is not this Born integral.

At \(N=1\), the reduced real vertex is the single-cell dense cubic vertex,
the momentum sum has one term, and the expression is exactly the existing
Nq=1 contraction, including its prefactor and zero extension.

## Vertex and Green-function constraints

For a reduced vertex derived from real force constants,

\[
V_{acd}(-q,-p)=V_{acd}(q,p)^*.
\]

Exchange of the two internal legs requires

\[
V_{acd}(q,p)=V_{adc}(q,q-p).
\]

Full permutation symmetry of the real Hamiltonian also relates different
choices of the output leg; one useful generator is

\[
V_{acd}(q,p)=V_{cad}(-p,-q).
\]

These relations use folded mesh indices. `TransverseCubicVertex` checks
reality and both generating leg exchanges with relative tolerance
\(10^{-10}\) and absolute tolerance \(10^{-12}\), without projecting or
averaging the supplied coefficients. A freely chosen complex tensor
does not automatically represent a real symmetric cubic Hamiltonian.
Synthetic physical vertices can instead be constructed from a real,
translation-invariant, fully permutation-symmetric real-space tensor.

For physical lesser and greater Green functions, anti-Hermiticity is
within each momentum block:
\(G^x(q,\omega)^\dagger=-G^x(q,\omega)\).
The bosonic frequency-reflection relation couples opposite momenta:

\[
G^<(q,-\omega)=G^>(-q,\omega)^T.
\]

The self-energies obey the corresponding relation. Applying a same-q
Nq=1 frequency-reflection rule independently at each q is generally
incorrect. This becomes visible on a four-point mesh containing opposite
quarter-period momenta. The subsequent
[bounded coupled-q toy solver](research-transverse-q-solver.md) implements
field validation, reconstruction, and iteration with these paired
identities.

The algebraic Born kernel can also be tested with arbitrary finite complex
Green arrays. Agreement in such a test checks the contraction and Fourier
conventions; it does not label the arrays as physical Green functions.

## Cell-origin changes

A common shift of the transverse cell origin leaves the reduced vertex and
the result unchanged. The associated phase is unity because
\(-q+p+(q-p)\) is a reciprocal vector.

A stronger test independently relabels the cell origin of each internal
coordinate: \(u'_{Ra}=u_{R+s_a,a}\), for integer cell vectors \(s_a\).
With \(D_{aa}(q)=e^{iq\cdot s_a}\),

\[
G'(q)=D(q)G(q)D(q)^\dagger,
\]

\[
V'_{acd}(q,p)=
e^{i[q\cdot s_a-p\cdot s_c-(q-p)\cdot s_d]}V_{acd}(q,p).
\]

The contraction must then satisfy
\(\Sigma'(q)=D(q)\Sigma(q)D(q)^\dagger\).
This is covariance under a coordinate relabeling, not a change of the
physical interaction.

## Independent small-cell reference

An independent reference constructs the complete \(ND\)-coordinate real
tensor \(\Psi\) and dense Green matrix from the stated Fourier convention,
then computes the real-space bubble with explicit cell, coordinate, and
zero-extended frequency sums. It Fourier transforms the resulting
self-energy and compares each diagonal q block with the coupled kernel.
The off-diagonal q blocks should vanish for translation-invariant inputs.
The reference should not call the new reduced contraction or use its
momentum-routing table.

Useful complementary checks are:

- A two-point mesh checks exact folding and reciprocal-vector routes.
- A four-point mesh of shape `(4, 1, 1)` with nonlocal real-space
  interactions produces genuinely complex reduced vertices. A `(2, 2, 1)`
  mesh alone has only real phase factors and cannot test conjugation.
- Complex, non-diagonal Green matrices and at least two internal
  coordinates distinguish coordinate ordering and conjugation errors.
- An onsite vertex and identical Green arrays at every q give the same
  result as the Nq=1 kernel at every q. This checks the \(1/N\) factor.
- Relabeling basis-cell origins checks the covariance above.
- Frequency support at the endpoints checks that out-of-grid sums do not
  wrap, even while momentum indices do wrap.
- Physical synthetic lesser/greater inputs check anti-Hermiticity,
  positivity of the correlation matrices, and joint q/frequency reflection
  separately from the raw contraction comparison.

Passing these checks establishes a bounded coupled-q Born implementation.
It does not establish a self-consistent solution, converged q sampling,
finite-frequency-cutoff accuracy, or a material transport prediction.

The default work guard allows at most 100,000
`(q, qprime, omega, omega_prime)` tuples per contour component; lesser and
greater use the same supported tuples. The stored vertex and combined
output-array payload each have a default 64 MiB limit. These are array and
work guards, not a process-memory guarantee. The reference implementation
uses deterministic direct sums and is intended for small tests.

The next milestone is implemented separately in
[`transverse_solver.py`](../src/negfpy/anharmonic/transverse_solver.py):
a bounded toy SCBA iteration with independent one- and four-q dense
real-space comparisons. Its
[solver guide](research-transverse-q-solver.md) records the API, passing
checks, and remaining scope. The Born kernel documented here remains a
contraction of supplied Green functions; it does not itself solve SCBA.
