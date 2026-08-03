# Silicon sparse-NEGF Phase A

Phase A validates only the harmonic crystal-to-layer mapping needed by the
future silicon \(N_q=1\) anharmonic device.  Passing it does **not** certify an
anharmonic junction calculation or a NEGFPy 2.0 release.

The frozen silicon fixture is derived from the authenticated phono3py 4.4.0
Si-PBEsol evidence.  Its exact raw force-constant hashes are:

- compact FC3: `c7d0902205d9ae48e657de53400c5c6acd0cdd0822804ac8c8dc154b29c15ffd`
- full FC3: `e4e297963b4c2a2a98953b140217baa1a7461b18cb05f12126ae545fe572cd73`

The historical reader-level identities remain separately recorded as FC2
`e62d4eed227281c2a4a4bf94f7b80db61a9e20a655f3ee7c35083c764b424fa0`
and FC3
`9f2d489638de7f38aa57b6730772316410d4aaf3db00f0b268ea843c5ffd6ba8`.
They must not be substituted for raw-file hashes.

## Frozen comparison policy

Geometry and principal-layer inference use a representation-independent FC2
image envelope: every shortest image and all nine Cartesian components are
included, without a numerical threshold.  Exact numerical support is recorded
separately.  Consequently, compact and full representations may have different
bitwise nonzero support while still constructing the same reciprocal dynamical
matrix within the declared operator tolerance.

Near the acoustic origin, comparing frequencies is ill-conditioned.  Below

\[
\nu_c=\sqrt{\mathrm{operator\_atol}}\,\omega_0/(2\pi\,10^{12})
=10^{-5}\ \mathrm{THz},
\]

the gate compares squared frequencies under the operator tolerance.  Above
that crossover it uses the frequency-in-THz gate.  Reciprocal dynamical
matrices are also compared directly under the combined absolute and relative
operator gate.

The development release line is `2.0.0.dev0`.  Older labels such as “0.3.0
Phase A” are retained only as internal historical milestone names and are not
package releases.

## Remote campaign

`scripts/run_sparse_nq1_phase_a_campaign.py` runs five immutable cases:
exact cosets, compact FC2, full FC2, full/compact equivalence, and solver-scale
invariance.  Every completed outcome is terminal.  An infrastructure
interruption with no valid outcome may resume only against the identical
campaign manifest.  The aggregate report always sets
`release_certified=false`; it can set only `phase_a_certified=true`.
