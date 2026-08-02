# Changelog

## 1.0.0 — 2026-08-02

This release establishes two validated, explicitly separated workflows:

- the existing harmonic open-device NEGF solver; and
- periodic-crystal intrinsic three-phonon thermal transport in the
  single-mode relaxation-time approximation (SMRTA).

The new stable `negfpy.bulk` namespace provides audited phono3py FC2/FC3
ingestion, phonon bands, total and atom/Cartesian-projected DOS, analytic
group velocities, mode heat capacities, three-phonon linewidths and
population lifetimes, mean free paths, mode- and tensor-resolved SMRTA
conductivity, cumulative conductivity, and schema-checked HDF5 archives.

The silicon qualification is an exact-input implementation comparison on
3x3x3, 5x5x5, and 11x11x11 meshes. It does not certify physical mesh or
smearing convergence for a quoted silicon conductivity.

Not included in 1.0 are iterative LBTE, isotope/boundary/four-phonon
scattering, polar non-analytical corrections, or real-material anharmonic
device NEGF. The latter is the principal 2.0 development target.
