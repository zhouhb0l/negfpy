# Changes

## 1.2.0 — 2026-11-20

- Save an authenticated rolling evaluated-input self-energy checkpoint after each
  completed physical map, atomically replacing the previous checkpoint. Keep
  rejected trials and selected anchors distinct; never import solver history.
- Add saved, qualified complete-site IFC2/IFC3 model inputs for NEP-derived
  finite-device SCBA using the selected complete endpoint and contact algorithms.
- Import authenticated historical evaluated INPUT fields with explicit same-model
  q/temperature initial-guess migration. Preserve geometry, ordered endpoint basis,
  frequencies, units and coupling, and require fresh physical and six-screen checks.
- Preserve all earlier JSON methods, exact-result restart and frozen releases.
  Fix clean-source installation in the version-tag publication workflow.

## 1.1.0 — 2026-11-20

- Add optional `initial_self_energy` to SCBA JSON inputs. Authenticate the prior
  input, converged result report and evaluated R/C/D fields by SHA-256, including
  their cross-file binding. Older result bundles use their original checksum
  manifest.
- Require the same physical inputs, geometry, ordered active DOFs, q/frequency
  grids, coupling, reservoirs and units; retain all original physical guards.
- Start with fresh solver history and a new full-map evaluation, including for
  an already converged guess. Never reuse saved Green fields or a Born image.
- Support linear, Anderson and reference/increment SCBA while preserving the
  original default zero start and all six fresh convergence screens.

## 1.0.2 — 2026-11-15

- Add Method 1 transport using the existing frozen retarded-kernel and
  conserving local thermal-probe implementation through the direct JSON command.
- Authenticate a converged, evaluated one-center short-device donor and preserve
  its complete endpoint support, normalization and physical guards.
- Reconstruct repeated FC2 blocks and contacts without new cubic construction,
  fitting or long-device SCBA maps; retain separate donor and probe qualification.
- Include a bounded silicon input near 1 µm on the donor's coarse grid, with
  explicit model and resource limits. Preserve 1.0.0 and 1.0.1 releases and inputs.

## 1.0.1 — 2026-11-15

- Expose the existing reference/increment SCBA method through
  `scba.method: "reference-seeded"` and the same direct JSON command.
- Construct references freshly within each calculation and retain the existing
  complete cubic model, independent channel fields and six convergence screens.
- Include bounded analytic-feedback proposals, with separate accounting for
  nonlinear maps, rejected trials and signed tangent evaluations.
- Add a five-layer silicon input while preserving version 1.0.0 inputs, releases
  and the original zero-start methods.

## 1.0.0 — 2026-11-15

- Package the existing complete-center cubic SCBA implementation as `negfpp`.
- Add direct JSON execution, authenticated external inputs and standard result files.
- Include the compact 24-atom silicon example with exact saved coefficients.
- Keep finite-device support and original numerical/physical checks; omit long-range extensions from the first release.
