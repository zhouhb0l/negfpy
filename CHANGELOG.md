# Changelog

## Unreleased — 2.0.0.dev0 preparation

- Allow explicitly requested contact allocation budgets up to 4 GiB for finer
  q/frequency grids, retaining the 64 MiB default and all contact equations and
  physical checks.
- Expose the existing fitted complete-energy, native R/C/D SCBA, bounded
  Anderson, proposal-feasibility and checkpoint implementations through
  `negfpy.anharmonic.complete_energy`.
- Add a portable caller-input example and document model, boundary and
  convergence assumptions. The fitted model remains a research surrogate.
- Remove 23 dated experimental launch scripts and their 24 driver-only tests
  from the active checkout. Their exact sources remain in Git history and
  external recovery snapshots. Reusable solver and model modules are retained.
- Align the Python minimum with the checkpoint code's Python 3.11
  `hashlib.file_digest` API. Default test discovery is limited to `tests/`;
  historical report checks can still be invoked explicitly.

This is source consolidation for a future release. It changes no solver
physics, physical guards or convergence thresholds, and does not publish or
certify a new software release or physical transport result.

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
