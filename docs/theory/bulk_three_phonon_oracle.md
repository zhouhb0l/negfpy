# Bulk three-phonon oracle convention

Status: implementation contract for NEGFPy `0.2.0`.

Reference implementation: phono3py 4.4.0, Si-PBEsol example at phono3py
commit `302ad205452ee36e57887b10bbf4a0289dd037bb`.

Validation evidence and release interpretation are recorded in the
[silicon bulk oracle validation record](../validation/silicon_bulk_oracle.md).

## Scope

The bulk oracle validates FC2 reconstruction, FC3 real-to-reciprocal
transformation, normal-mode matrix elements, three-phonon phase space, and
on-shell linewidths. It is independent of the open-device SCBA solver. No
material-device calculation is certified by this oracle.

Raw, corrected, mass-normalized, translational, and device-mapped force
constants are distinct immutable objects. Readers never symmetrize, enforce
sum rules, truncate, expand compact tensors, or apply
`fc3_nonzero_indices`. Any correction is an explicit operation returning a
new object and a before/after report.

## Crystal and mapping convention

- Lattice vectors are rows.
- Scaled and Cartesian positions obey
  \(r_{\mathrm{cart}}=r_{\mathrm{scaled}}A\).
- `p2s[p]` is the representative supercell index of primitive atom `p`.
- `s2p[s]` is a representative supercell index and therefore takes values
  from `p2s`; it is not a primitive ordinal.
- Dense `p2p[s]` equals the primitive ordinal at representative indices and
  `-1` elsewhere.
- Flattened shortest vectors and `(count, offset)` multiplicities must form
  an exact partition without gaps or overlaps.
- FC2 may use the `phonon_*` supercell while FC3 uses the ordinary
  supercell. Their ordered primitive cells must agree even when the two
  supercells differ.

## Harmonic reconstruction

For fractional reciprocal coordinate \(q\), the raw non-NAC dynamical matrix
is

\[
D_{i\alpha,j\beta}(q)=
\frac{1}{\sqrt{M_iM_j}}
\sum_{s:\,s2p(s)=p2s(j)}
\Phi_{i\alpha,s\beta}
\frac{1}{m_{si}}\sum_{\ell=1}^{m_{si}}
e^{2\pi i q\cdot r_{si\ell}}.
\]

For compact FC2, the first tensor axis is the primitive ordinal `i`; for full
FC2 it is `p2s[i]`. Raw Hermiticity is audited before any optional explicit
\((D+D^\dagger)/2\) operation. Frequencies are cyclic THz:

\[
\nu_\lambda(q)=
\operatorname{sgn}(d_\lambda)
\sqrt{|d_\lambda|}\,C_{\mathrm{FC2}\rightarrow\mathrm{THz}}.
\]

The pinned reference conversion is recorded in every result. Negative
eigenvalues remain signed and are never clipped to zero.

## Full-mesh momentum and Brillouin-zone convention

The routing mesh is an unshifted diagonal full mesh
\(D=(D_0,D_1,D_2)\). Residue indices use phono3py's X-fast ordering. NEGFPy
stores one componentwise-centered integer address \(a\) per residue; an even
Nyquist component is represented by \(-D_i/2\). For each fixed \(q_0\), every
\(q_1\) residue occurs exactly once and

\[
a_2\equiv-a_0-a_1\pmod D.
\]

This modular routing is exact integer arithmetic and is independent of
Brillouin-zone relocation. The centered addresses define a canonical routing
vector

\[
G_{\mathrm{canonical},i}
=\frac{a_{0i}+a_{1i}+a_{2i}}{D_i}\in\mathbb Z.
\]

On an even mesh, however, a Brillouin-zone surface residue has multiple
equivalent shortest images. Therefore \(G_{\mathrm{canonical}}\) is not used
for the certified phono3py Normal/Umklapp split.

Let \(A\) contain row-wise primitive direct-lattice vectors and let
\(B=A^{-1}\) be the reciprocal column basis, without an irrelevant common
\(2\pi\) factor. The phono3py-compatible surface-image construction first
computes

\[
B_{\mathrm{red}}=
\left[
\operatorname{niggli\_reduce}
\left(B^T,\epsilon_{\mathrm{Niggli}}\right)
\right]^T,\qquad
\epsilon_{\mathrm{Niggli}}=10^{-5}.
\]

The certified path uses `spglib.niggli_reduce` and records the resulting
basis. An explicitly supplied reduced basis is accepted only after it is
shown to be related to \(B\) by an integer unimodular transformation.

Phono3py 4.4.0 uses a tolerance in **squared reciprocal distance**:

\[
\epsilon_{\mathrm{BZ}}
=0.01\max_i
\frac{\sum_j(B_{\mathrm{red}})_{ji}^2}{D_i^2}.
\]

The factor \(0.01\), the Niggli tolerance, and the reduced basis are all part
of the external-oracle convention. They are not generic numerical
tolerances.

For each centered address, candidate reciprocal translations are visited in
the exact Cartesian-product order

```text
(0, 1, 2, -2, -1)^3.
```

Starting from the zero-shift candidate, a candidate becomes the reference
only if its squared distance is strictly smaller than
\(d_{\mathrm{ref}}^2-\epsilon_{\mathrm{BZ}}\). Every candidate satisfying
\(d^2<d_{\mathrm{ref}}^2+\epsilon_{\mathrm{BZ}}\) is retained. Thus, on a
skewed cell, the pinned algorithm can deliberately retain a near-boundary
image that is not the unique mathematical nearest image.

For one triplet, all retained surface images of all three legs are
considered. It is Normal exactly when at least one image combination obeys

\[
\widetilde a_0+\widetilde a_1+\widetilde a_2=0.
\]

Otherwise it is Umklapp. Equivalently, the selected surface-image reciprocal
vector \(G_{\mathrm{BZ}}\) is zero for Normal and nonzero for Umklapp. A zero
candidate is always preferred. For nonzero candidates, NEGFPy stores the
minimum-Cartesian-norm \(G_{\mathrm{BZ}}\); tolerance ties are broken
lexicographically to make the result independent of container ordering.

`MomentumTriplet.G` is the centered-address routing diagnostic
\(G_{\mathrm{canonical}}\). The bulk-oracle `reciprocal_vectors` and
`normal_mask` fields instead contain \(G_{\mathrm{BZ}}\) and
\(G_{\mathrm{BZ}}=0\), respectively. These can differ at a Brillouin-zone
surface. No irreducible-mesh, time-reversal, or \(q_1/q_2\)-exchange
reduction is used by the serial oracle.

## FC3 reciprocal convention

The reference phase is \(e^{+2\pi i q\cdot r}\). For the single-anchor
(`make_r0_average=False`) path,

\[
\Phi^{(3)}_{ijk}(q_0,q_1,q_2)
=e^{2\pi i(q_0+q_1+q_2)\cdot\tau_i}
\sum_{\substack{s\mapsto j\\t\mapsto k}}
\Phi^{(3)}_{p2s(i),s,t}
\left\langle e^{2\pi i q_1\cdot r_{si}}\right\rangle
\left\langle e^{2\pi i q_2\cdot r_{ti}}\right\rangle .
\]

Momentum conservation is exact integer arithmetic on the full mesh:

\[
q_0+q_1+q_2=G.
\]

Here \(G\) follows the Brillouin-zone distinction above: modular routing uses
\(G_{\mathrm{canonical}}\), while archived bulk interactions use the
surface-image \(G_{\mathrm{BZ}}\) for Normal/Umklapp metadata.

Phono3py 4.4.0 defaults to the separate three-anchor
`make_r0_average=True` convention. NEGFPy implements and names both paths;
the selected convention is part of the problem hash. Results from one path
must never be compared to the other without regenerating the reference.

## Read-only translational FC3 audit

`validate_translational_fc3` audits a `TranslationalFC3` together with its
`CrystalMapping`. It never edits, completes, symmetrizes, thresholds, or
truncates the vertex.

First, it verifies that every raw atom group contains exactly one consistent
expanded term for each product of shortest-vector images. It then constructs
the reciprocal tensor at
\((q_0,q_1,q_2)=(0,0,0)\), using the vertex's selected single-anchor or
three-anchor convention, and retains all three atom-axis residual tensors:

\[
R^{(0)}_{jk,\alpha\beta\gamma}
=\sum_i\Phi^{(3)}_{ijk,\alpha\beta\gamma}(\Gamma,\Gamma,\Gamma),
\]

\[
R^{(1)}_{ik,\alpha\beta\gamma}
=\sum_j\Phi^{(3)}_{ijk,\alpha\beta\gamma}(\Gamma,\Gamma,\Gamma),
\qquad
R^{(2)}_{ij,\alpha\beta\gamma}
=\sum_k\Phi^{(3)}_{ijk,\alpha\beta\gamma}(\Gamma,\Gamma,\Gamma).
\]

Each full residual tensor is stored as an immutable array. With

\[
\Phi_\Gamma^{\max}
=\max_{ijk\alpha\beta\gamma}
\left|\Phi^{(3)}_{ijk,\alpha\beta\gamma}
(\Gamma,\Gamma,\Gamma)\right|,
\]

the common gate is

\[
\max|R^{(a)}|
\le \epsilon_{\mathrm{abs}}
+\epsilon_{\mathrm{rel}}\Phi_\Gamma^{\max},
\quad a=0,1,2,
\]

with defaults \(\epsilon_{\mathrm{abs}}=10^{-10}\) in the FC3 source units
and \(\epsilon_{\mathrm{rel}}=10^{-8}\). All three axes must pass
independently.

Spatial support is evaluated over every expanded term whose raw Cartesian
FC3 block has at least one component numerically unequal to zero. No support
threshold is used. For shortest-vector offsets \(r_1,r_2\) relative to the
anchor and the row-wise primitive lattice \(A\), the audit measures all three
pair distances

\[
d_{01}=\|r_1A\|,\qquad
d_{02}=\|r_2A\|,\qquad
d_{12}=\|(r_2-r_1)A\|.
\]

It reports each pair's minimum and maximum, the full cluster diameter
\(\max(d_{01},d_{02},d_{12})\), and exact-zero/nonzero counts for both raw
blocks and expanded image terms. Distances inherit the primitive lattice's
length unit; for the pinned phono3py input this is angstrom. In particular,
the second--third distance may exceed both anchor radii and must not be
omitted when choosing an interacting-device buffer.

## Normal-mode contraction

Eigenvectors are stored by columns. The phono3py-compatible contraction does
not conjugate the three eigenvector factors:

\[
\mathcal V_{\lambda_0\lambda_1\lambda_2}
=
\frac{
\sum_{ijk,\alpha\beta\gamma}
e_{i\alpha,\lambda_0}(q_0)
e_{j\beta,\lambda_1}(q_1)
e_{k\gamma,\lambda_2}(q_2)
\Phi^{(3)}_{ijk,\alpha\beta\gamma}
}{
\sqrt{M_iM_jM_k\,
\nu_{\lambda_0}\nu_{\lambda_1}\nu_{\lambda_2}}
}.
\]

Modes at or below the declared cutoff are excluded. Individual matrix
elements inside a degenerate subspace are gauge dependent; validation
compares sums of \(|\mathcal V|^2\) over matched degenerate subspaces.

The phono3py-compatible interaction-strength conversion contains the fixed
factors \(1/3!^2=1/36\), \(1/2^3=1/8\), and \(1/N_q\). These are named
factors in result metadata, not an adjustable prefactor.

## Gaussian on-shell linewidth

The first certified linewidth comparison uses the same positive Gaussian
smearing width \(\sigma\), full mesh, mode cutoff, and temperature in both
codes. With cyclic frequencies,

\[
\delta_\sigma(x)=
\frac{\exp[-x^2/(2\sigma^2)]}{\sqrt{2\pi}\sigma}.
\]

The two energy channels follow the phono3py decomposition

\[
g_1=\delta_\sigma(\nu_0-\nu_1-\nu_2),
\]

\[
g_2-g_3=
\delta_\sigma(\nu_0+\nu_1-\nu_2)
-\delta_\sigma(\nu_0-\nu_1+\nu_2).
\]

The occupation factor is

\[
(n_1+n_2+1)g_1+(n_1-n_2)(g_2-g_3).
\]

The archived phono3py `gamma` is cyclic-frequency HWHM in THz. Conversion is
therefore:

\[
\gamma_{\mathrm{angular,HWHM}}=2\pi\gamma_{\mathrm{phono3py}},
\qquad
\Gamma_{\mathrm{angular,FWHM}}=4\pi\gamma_{\mathrm{phono3py}}.
\]

NEGFPy's open-system linewidth
\(-\operatorname{Im}\Sigma^r/\omega\) is angular FWHM, so comparisons use the
second conversion exactly once.

## Certification

- Raw FC2 dynamical matrices: maximum absolute error \(10^{-10}\).
- Full/compact FC3 ingestion: every physical real-space block, every
  redundant full-tensor first-axis translation row, and every expanded
  translational term are compared. The canonical schedules use
  `rtol=1e-10`, `atol=1e-12`; the additional reciprocal-space comparison is
  explicitly a deterministic sampled corroboration, not an exhaustive mesh
  claim.
- Sampled gauge-invariant \(|\mathcal V|^2\): relative error \(10^{-8}\).
- Modes with phono3py \(\gamma>10^{-6}\) THz: median relative error below
  1% and maximum below 5%.
- Smaller linewidths: absolute error below \(10^{-6}\) THz.
- Serial/MPI subsets are merged in canonical target-grid-point order and
  must agree without floating-point MPI reductions.

Passing this oracle certifies only bulk FC2/FC3 algebra and units. Sparse
device SCBA remains blocked until the separate dense/sparse and conservation
gates pass.

## Remaining boundaries and ambiguities

- The three Gamma ASR tests are reciprocal-space aggregate checks. They do
  not yet replace a complete fixed-real-space-pair sum-rule audit, nor do
  they test rotational invariance. For `make_r0_average=True`, they audit the
  selected three-anchor representation; the raw FC3 validation report must
  still be retained.
- Exact-nonzero support is deliberately conservative: floating-point noise
  in an otherwise negligible FC3 component can extend the reported range.
  Any numerical cutoff or neighbor-shell truncation must be a separately
  named approximation with a before/after report.
- At a shortest-vector multiplicity boundary, the explicit phase expansion
  contains every retained image product. Its reported cluster diameter is
  therefore the exact support of this representation, but need not equal a
  unique chemical-neighbor cutoff. A more compact real-space remapping would
  require a separate algebraic-equality proof.
- Normal/Umklapp labels at a Brillouin-zone surface depend on the pinned
  surface-image and tolerance convention. The total scattering result is
  the primary invariant; a Normal/Umklapp decomposition must quote this
  convention.
- The lexicographically selected nonzero \(G_{\mathrm{BZ}}\) is a
  deterministic provenance value, not an additional physical observable.
- The present oracle covers unshifted diagonal full meshes. Generalized
  regular grids, mesh shifts, and symmetry-reduced meshes require their own
  exact routing comparison before certification.
- Niggli reduction can have more than one equivalent basis at exact
  degeneracies. Therefore the pinned spglib version, reduction tolerance,
  and resulting reduced-basis matrix must remain in archived provenance.
