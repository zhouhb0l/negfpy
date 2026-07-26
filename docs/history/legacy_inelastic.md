# Legacy inelastic implementation

The experimental `negfpy.inelastic` implementation introduced at commit
`5e8aaecad4e4e0d43dcaa11598a376951a7345e1` is intentionally excluded from
the active package.

That code mixed third- and fourth-order approximations, historical
paper-specific drivers, and production-looking interfaces before their
normalization, conservation, and material-IFC contracts had passed the
current validation gates. It must not be used as evidence for the certified
anharmonic solver.

The complete source, tests, examples, derivations, and literature audit remain
recoverable through:

- archival ref `refs/archive/legacy-inelastic-5e8aaec`;
- the checksummed pre-integration Git bundle and dirty-checkout snapshot in the
  workspace archive.

Validated cubic Born and conserving SCBA development lives in
`negfpy.anharmonic`. Historical Wang and Mingo comparisons in that namespace
must remain explicitly labelled as regression or unresolved reproduction
work; they do not alter the production cubic prefactor.
