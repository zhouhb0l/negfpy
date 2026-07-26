# Workspace layout

The NEGFPy Git repository is deliberately limited to reusable source code,
tests, generic examples, and documentation. Material inputs, generated data,
validation evidence, and external programs live beside the repository so that
package history remains reviewable and reproducible.

| Workspace path | Duty |
| --- | --- |
| `negfpy/` | Canonical source repository |
| `studies/` | Material-specific configurations, drivers, IFC inputs, analyses, and run outputs |
| `datasets/` | Pinned public or externally supplied datasets with provenance manifests |
| `artifacts/validation/` | Immutable validation logs, checkpoints, reports, and checksums grouped by gate and commit |
| `reports/` | Human-readable report sources, figures, and compiled deliverables mirrored from certified artifacts |
| `environments/` | Explicit environment locks and machine manifests |
| `worktrees/` | Temporary clean Git worktrees used for integration and certification |
| `external/` | External scientific programs and their local support files |
| `archive/` | Read-only recovery snapshots and superseded implementations |
| `tmp/` | Disposable scratch space; never a provenance source |

Paths stored in study configurations should be relative to the launch
directory. From the repository root, workspace studies are reached through
`../studies/`. Do not commit material datasets, generated runs, binary
executables, or validation artifacts to the source repository.

Every release result must identify the exact source commit and environment lock
and must be copied into `artifacts/validation/` with a checksum manifest before
it is cited by a report.
