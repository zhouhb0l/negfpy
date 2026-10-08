# Selected fixed-field cut-current diagnostic

`FixedFieldGreenEvaluator(problem, config, sigma)` evaluates Dyson and Keldysh
equations with the supplied interaction self-energy. It performs no SCBA map.
Nonzero nodes use `PointwiseInteracting` with `linear-keldysh` interpolation,
including separate reservoir temperatures. At zero it uses the stored
interaction node and the existing acoustic contact derivative limit, or the
existing gapped-contact zero convention. It does not set the Green function
to zero, average neighboring nodes, add device broadening or use a pseudoinverse.

`collect_selected_cut_spectra(plans, grid, evaluator.evaluate,
max_evaluations=..., max_seconds=...)` visits every signed node, including zero,
once for all requested cuts. It keeps a union of the greater-G rows and active
columns needed by the contracted cubic kernel, plus scalar harmonic-current
spectra. Full device matrices exist at one frequency at a time. The fixed
self-energy and the evaluator's interpolation arrays still occupy full-grid
memory. The output-byte limit does not bound those arrays or process RSS.
Evaluation-count and byte limits fail before allocation; the time limit is
checked between calls. An enclosing process timeout is needed for stalled calls.

For each cut, `selected.integrate(cut, lags,
mean_policy=CENTERED_FIXED_FIELD_MEAN_POLICY)` reports harmonic current,
connected one-vertex cubic current, and their centered sum. The harmonic term
uses Re integral omega G<, while the connected kernel uses C=i integral G>
and P=integral omega G>. The two Keldysh fields are never interchanged or inferred
from a single equilibrium temperature. Each physical current scale is applied
once. This path reuses the prior equal-coordinate-leg cut operator unchanged.

The exported constant has the value `centered-fixed-field-diagnostic`. Its
full research interpretation is fixed in the
[Nq=1 mean-policy contract](research-nq1-mean-policy.md).

The centering policy is an explicit diagnostic assumption. It does not solve
static means, add a physical linear counterterm, or supply their induced
stiffness feedback. In particular, appending inferred disconnected means to
bubble-only fields would not establish consistency through order Psi squared.
The result therefore cannot by itself validate material spatial transport or
change existing solver acceptance.

## Frequency and memory scope

The stored rectangle grid gives one finite-frequency approximation. A refined
evaluation grid can sample the same fixed input sigma through linear Keldysh
interpolation: it changes the Green-function quadrature, not the accepted
SCBA solution or its interaction-field frequency accuracy. The prescribed
exact-zero sample can have substantial rectangle weight. Near-zero point checks
and a frequency-step comparison are required to interpret that weight.

Finite-lag integrals must be compared across memory endpoints and time steps.
For step .025, the Fourier helper requires lag < pi/.025 = 125.663706; suggested
diagnostic endpoints are 75, 100 and 120. Halving the evaluation step permits
longer recurrence time, but does not establish a converged memory tail. The
common cutoff remains 7.5; no exterior-tail accuracy is inferred. An unchanged
SCBA acceptance, a resolved local equation, or a passing toy reference is not
a frequency-accuracy or material continuity result.

The research driver and authenticated material preparation live in
`reports/phonon-phonon-interaction/nq1-selected-current-development-20261007`
in the workspace. It targets two internal cuts (3 and 5) using accepted
25703621 input fields only. Its separate result must retain failed or
inconclusive comparisons; it must not relax the old 1% current screen or
replace accepted archives.
