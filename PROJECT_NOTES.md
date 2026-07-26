# NEGFPy platform guide

This repository contains the reusable NEGFPy implementation. Material inputs,
generated calculations, validation artifacts, external programs, and compiled
reports live in the sibling workspace directories described in the
[workspace layout](docs/development/workspace_layout.md).

## Active code responsibilities

| Area | Location | Responsibility |
| --- | --- | --- |
| Harmonic NEGF | [core package](src/negfpy/core/) | Surface Green functions, device Green functions, transmission, currents, and harmonic observables |
| Models and IFC mapping | [modeling package](src/negfpy/modeling/), [I/O package](src/negfpy/io/), and [workflows](src/negfpy/workflows/) | Unit-aware IFC ingestion, mapping, audits, and construction of harmonic transport problems |
| Cubic anharmonic NEGF | [anharmonic package](src/negfpy/anharmonic/) | Signed grids, cubic contractions, Born and conserving SCBA solvers, currents, diagnostics, and restart state |
| Task-level MPI | [private MPI scheduler](src/negfpy/anharmonic/_mpi_tasks.py) | Parallel execution of complete validation cases; it does not partition an SCBA solve |
| Tests | [test suite](tests/) | Scientific identities, analytic fixtures, release gates, MPI equivalence, restart integrity, and remote-lifecycle checks |

The harmonic public interfaces remain the baseline. Anharmonic functionality
is isolated under `negfpy.anharmonic`; historical inelastic implementations do
not define production conventions.

## Theory and validation

- [Anharmonic theory contract](docs/anharmonic_theory.md)
- [Anharmonic validation contract](docs/anharmonic_validation.md)
- [Historical convention mapping](docs/anharmonic_historical_conventions.md)
- [Remote heavy-validation protocol](docs/remote_heavy_validation.md)
- [Toy-model report source](docs/validation/toy-model-report/)
- [Pinned remote MPI environment lock](validation/locks/negf-toy-gate-linux-64.lock)

The production hierarchy is harmonic NEGF, one-shot cubic Born for screening,
and conserving cubic SCBA for interacting conductance. Historical Wang and
Mingo implementations are regression or literature-reproduction work and
cannot calibrate the production cubic prefactor.

## Workspace responsibilities

- Active studies: [harmonic](../studies/harmonic/),
  [anharmonic](../studies/anharmonic/), and
  [materials](../studies/materials/).
- Public or externally supplied IFC data: [datasets](../datasets/).
- Immutable run evidence: [validation artifacts](../artifacts/validation/).
- Human-readable deliverables: [reports](../reports/).
- Environment manifests: [environments](../environments/).
- Recovery snapshots and superseded implementations: [archive](../archive/).

Generated results must not be copied into the source package. A release result
must identify one clean source commit and one explicit environment lock, pass
the applicable scientific and orchestration gates, and be mirrored with
checksums before a report cites it.

## Historical notes

- [Legacy inelastic implementation](docs/history/legacy_inelastic.md)
- [TaN phonopy transport debugging](docs/history/tan_phonopy_transport_debugging.md)

The TaN note was moved from this root guide because it is material-specific. It
is preserved for context and does not establish a current release default.
