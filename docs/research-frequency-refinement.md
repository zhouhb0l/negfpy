# Local frequency refinement for the cubic Born bubble

`negfpy.anharmonic.frequency_refinement` supplies an explicit window integral
of the full lesser and greater self-energy matrices. It is an opt-in research
API. The existing uniform-grid FFT kernel and SCBA defaults are unchanged.

```python
from negfpy.anharmonic.harmonic_pointwise import PointwiseHarmonic
from negfpy.anharmonic.frequency_refinement import (
    GaussLegendreWindowPlan, cubic_window_lesser_greater,
)

fields = PointwiseHarmonic(problem, config)
plan = GaussLegendreWindowPlan(checked_breakpoints, order=16)
local = cubic_window_lesser_greater(
    problem.vertex, fields, -2.75, plan, cutoff=7.5,
    max_workers=8, tracked_ranges={"core": (-1.3875, -1.3625)},
    max_seconds=1000,
)
if not local.completed:
    raise RuntimeError("The returned matrices contain only a partial integral")
```

The callback returns independently evaluated `(G_lesser, G_greater)` at each
arbitrary signed frequency. The implementation preserves all matrix entries,
the two cubic vertices and the `.5j/(2*pi)` normalization; it does not multiply
an old self-energy by a scalar correction. It compiles sparse vertex slices
once and streams a bounded number of frequency pairs in deterministic order.
Outside the declared frequency cutoff the fields are zero; quadrature panels
are clipped to the intersection of the two arguments' supports.

The plan is a fixed positive-weight composite Gauss-Legendre rule, not a
self-certifying adaptive solver. Compare separately chosen/refined partitions,
including full matrix errors, before trusting an integral. A plan selected
using one matrix entry need not resolve every other entry.

`PointwiseHarmonic` follows the existing contact and Dyson equations at nonzero
frequency, including separate reservoir Bose products for lesser and greater.
It requires zero device eta and retains the supplied lead eta and temperatures.
Exact zero needs the existing acoustic-limit machinery and is rejected here.
This harmonic callback is not an interacting off-grid SCBA callback.

A coarse cell may be replaced by subtracting its original matrix contribution
and adding a checked local integral. The experiment uses exactly one complete
midpoint cell, so its boundary weights are unambiguous. General partially
covered cells need explicit overlap weights; overlapping corrections must not
be applied twice. A local correction does not establish total frequency
accuracy, retarded self-energy consistency or a converged SCBA solution.
