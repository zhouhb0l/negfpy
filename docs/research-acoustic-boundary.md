# Experimental finite-range acoustic boundary completion

This is an explicit, opt-in research Hamiltonian. It does **not** replace IFC
ingestion, `map_fc3_to_device`, the accepted material inputs, or their default
Cartesian crop. It does not recover a unique physical interface from the cropped
tensor. The original raw IFC and mapped vertex remain immutable.

## Definition

For the mass-normalized cubic energy

\[
 E_3(Q)=\frac{1}{6}\sum_{ijk}\Psi_{ijk}Q_iQ_jQ_k,
\]

a rigid Cartesian displacement has components
\(t^{(a)}_{i}=\sqrt{m_{\operatorname{atom}(i)}}\delta_{a,\operatorname{axis}(i)}\).
Translation invariance requires the contraction with each of these three vectors
to vanish on every tensor leg. For a fully permutation-symmetric tensor it is
sufficient to impose this on one leg. It then follows directly by expanding the
cubic polynomial that \(E_3(Q+t)=E_3(Q)\): terms with one or more occurrences of
the rigid translation vanish. Equivalently this energy can be written entirely
in relative displacements, without choosing a privileged reference atom.

The model `finite-range-minimum-change-acoustic-completion-v1` chooses

\[
 \Psi^*=\mathop{\rm argmin}_{X}\|X-\Psi\|_F^2
\]

subject to exact permutation symmetry, the Cartesian translation constraints,
and fixed **atom-triplet** support. An atom triplet is allowed if it already has
at least one nonzero Cartesian entry in the source vertex. All Cartesian entries
within such a block are allowed in the correction. No new atom triplets, atoms,
interaction layers, or longer-range interactions are introduced. Exact-zero
Cartesian entries can become nonzero; this is reported and is not thresholded
away afterward.

One scalar variable represents each unordered DOF triple. Weighting it by the
square root of its number of ordered permutations makes the least-squares norm
the full ordered-tensor Frobenius norm, including repeated indices correctly.
The sparse constraint matrix includes one equation per unordered free DOF pair
and contracted Cartesian direction. LSMR from zero obtains the minimum-norm
correction. Its `atol=btol=1e-13` and iteration cap are explicit. A final direct
all-leg contraction must satisfy the declared relative ASR tolerance, default
`1e-10`, or construction fails. The relative error is
`max(abs(contraction))/max(absolute-sum bound)`, not a pointwise ratio.

The source's permitted last-bit permutation differences are averaged explicitly
when defining the symmetric variables. The reported correction includes these
differences. An already symmetric, translation-consistent source is preserved
up to numerical roundoff. Empty source blocks are not invented to repair a
globally inconsistent or insufficient support pattern; a tensor can shrink
substantially if its permitted support cannot represent a useful invariant
interaction, which is why distortion must be inspected.

## Physical meaning and limits

This is a local-support, minimum-change boundary **completion**, expressed as a
constrained IFC correction. It is not a silent global center-of-mass projection:
in particular it cannot connect distant atom triplets to cancel a surface
residual. The optimization coordinates are globally solved within that fixed
local support, so a correction can propagate into interior entries. Boundary
and interior Frobenius changes are reported separately. The `boundary_layers`
argument defines those report categories only, not an immutable interior.

Translation invariance is necessary for the intended unpinned lattice, but it
does not establish the correct microscopic anharmonic interface. The cropped
FC3 does not specify a unique decomposition into complete local interatomic
energy terms. Additional rotational/point-group constraints and the correct
surface chemistry are not fitted by this method. A potential-derived local
bond or cluster construction could be a better later model. This completion
therefore needs explicit harmonic/Born comparison and a stated physical choice
before any new SCBA study. Numerical improvement alone is not validation.

The first research screen requires all of:

- Total ordered-tensor Frobenius change at most 25%.
- Interior-only Frobenius change at most 5%.
- Completed/source tensor norm between 0.75 and 1.25.
- At most twice as many nonzero Cartesian entries.
- At most four times the pair-work proxy (sum of squared support counts per
  external DOF).

These are **heuristic engineering perturbation and cost limits**, not scientific
acceptance tolerances. They prevent silently using a heavily changed or costly
tensor in the next bounded trial. A diagnostic may report an out-of-screen model,
but should not launch an SCBA trial with it. Passing the screen still requires
inspection of measured Born/grid behavior and does not certify a boundary.

## Use and identity

```python
from dataclasses import replace
import numpy as np
from negfpy.anharmonic.research_boundary import complete_acoustic_boundary

completion = complete_acoustic_boundary(
    problem.vertex,
    atom_masses=np.ones(problem.ndof // 3),  # Relative equal masses for Si ONLY
    dof_per_layer=problem.device.dof_per_layer,
    boundary_layers=(2, 6),
)
experimental_problem = replace(problem, vertex=completion.vertex)
report = completion.report
```

For nonuniform masses supply the actual relative atom masses in exactly the
device's Cartesian ordering. Common mass scaling leaves the model unchanged.
The correction returns a new `SparseCubicVertex` with a source physics/storage
hash, new model identity, new physics hash, and research-only metadata. It does
not inherit the original mapping's uncorrected/certified provenance. Its report
is JSON serializable and includes support growth, changes and ASR contractions.

An old checkpoint represents a different Hamiltonian. It cannot be resumed or
relabeled as this model. Use a fresh identity and initialization for any later
bounded SCBA trial. No SCBA run is performed by this module.

## Verification

`tests/test_research_acoustic_boundary.py` checks independently constructed
relative-displacement cubic bond energies, unequal-mass translations, scalar
energy invariance, exact permutation symmetry, finite atom-triplet support,
source immutability, the analytically known two-atom minimum-norm solution,
reported boundary/interior corrections, explicit distortion screening, model
identity, malformed inputs, and zero vertices. These algebraic checks do not
replace the requested real-Si Born/grid comparison.
