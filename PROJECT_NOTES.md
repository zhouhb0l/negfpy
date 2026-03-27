# Project Notes

## Scope

This file records the debugging work done in `negfpy` for the TaN phonopy-based phonon transport calculations.

## Main bugs fixed

### 1. Repeat-1 supercell direction was not handled generally enough

Problem:

- For phonopy IFCs, if one supercell direction has size `1`, the real-space indexing along that direction is collapsed.
- In that case, interactions must be reassigned using the minimum-image rule.
- The old code only applied this correction in the transport direction.

Consequence:

- If a transverse direction had repeat `1`, some couplings crossing the periodic boundary were misassigned as same-cell couplings.

Fix:

- Generalized the minimum-image reassignment so it applies to any axis with repeat `1`, not only the transport axis.

Main file:

- [src/negfpy/workflows/ifc_bulk.py](/home/zhouhb/Desktop/workspace/negfpy/src/negfpy/workflows/ifc_bulk.py)

### 2. Spurious principal-layer dependence in phonopy IFC reconstruction

Problem:

- After unwrap, the builder could misclassify the phonopy IFC and fall back to an approximate reconstruction path.
- This produced strong dependence on manually chosen principal-layer size.

Consequence:

- Transmission changed significantly when principal-layer size changed from `1` to `2` to `4`, which is not physically acceptable for the same underlying system.

Fix:

- Improved the builder logic so it checks transport-axis completeness correctly.
- This avoids the wrong fallback path and treats transport-axis Nyquist terms more consistently.

Main file:

- [src/negfpy/modeling/builders/ifc_to_terms.py](/home/zhouhb/Desktop/workspace/negfpy/src/negfpy/modeling/builders/ifc_to_terms.py)

### 3. Transverse Nyquist handling was incomplete

Problem:

- The transverse Nyquist handling could distort the real-space representation used for continuous transverse `k`.

Consequence:

- Even after the earlier fixes, some spectra still showed suspicious behavior because the transverse Fourier representation was not fully consistent.

Fix:

- Added explicit transverse Nyquist splitting into `+/-` partners before the final transport construction.

Main file:

- [src/negfpy/workflows/ifc_bulk.py](/home/zhouhb/Desktop/workspace/negfpy/src/negfpy/workflows/ifc_bulk.py)

## Hybrid k-mesh mode added

We found a numerical tradeoff:

- `shifted` mesh:
  - reduces high-frequency spikes
  - but can artificially suppress low-frequency onset because Gamma is missed
- `centered` mesh:
  - restores low-frequency onset
  - but can overemphasize sharp resonances

To address this, hybrid modes were added:

- `hybrid_centered_shifted`
  - low frequency: centered
  - high frequency: shifted
- `hybrid_shifted_centered`
  - low frequency: shifted
  - high frequency: centered

Also supported:

- `kmesh.mode_low`
- `kmesh.mode_high`

These options are implemented in:

- [src/negfpy/workflows/ifc_bulk.py](/home/zhouhb/Desktop/workspace/negfpy/src/negfpy/workflows/ifc_bulk.py)

## Files changed during this debugging cycle

Main modified source files:

- [src/negfpy/workflows/ifc_bulk.py](/home/zhouhb/Desktop/workspace/negfpy/src/negfpy/workflows/ifc_bulk.py)
- [src/negfpy/modeling/builders/ifc_to_terms.py](/home/zhouhb/Desktop/workspace/negfpy/src/negfpy/modeling/builders/ifc_to_terms.py)

## Final useful mental model

For phonopy IFC transport:

- If any supercell direction has repeat `1`, minimum-image reassignment is necessary.
- If principal-layer dependence appears, suspect IFC reconstruction logic.
- If low-frequency gap vs high-frequency spikes depends on `shifted` vs `centered`, suspect transverse `k` integration rather than raw IFC corruption.
- The hybrid k-mesh mode is the practical recommended option for this TaN workflow.

## Related project

The corresponding project-level notes and final result layout are documented in:

- [../TaN/PROJECT_NOTES.md](/home/zhouhb/Desktop/workspace/TaN/PROJECT_NOTES.md)
