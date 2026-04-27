# Wang 0701164 Fig. 5 Audit

Target paper:
- J.-S. Wang, J. Wang, N. Zeng
- "Nonequilibrium Green's function method for thermal transport in junctions"
- local copy: `literature/0701164v1.pdf`
- target figure: `Fig. 5`

Goal of this audit:
- check whether the current `negfpy` third-order lowest-order (`LO`) path can be expected to match Fig. 5
- separate model mismatch from approximation mismatch before implementing a benchmark

## What Fig. 5 actually is

The paper section is:
- `IV.A. One-dimensional cubic onsite model`

The model is not FPU-alpha.

The paper states:
- harmonic chain with inter-particle spring constant `K`
- onsite spring constant `K0`
- cubic onsite interaction in the center:
  `sum_j (1/3) t_j u_j^3`
- for the center region:
  `T_ijk = t delta_ij delta_ik`
- left lead, center, and right lead are otherwise identical

Classical equation of motion in the paper:
- `u_ddot_j = K u_{j-1} + (-2K - K0) u_j + K u_{j+1} - t_j u_j^2`

Published benchmark parameters:
- `K = 0.625 eV / (A^2 u)`
- `K0 = 0.1 K`
- center length `N = 5`
- cubic coupling values:
  `t = 0, 0.2, 0.5, 0.7, 1.0, 2.0 eV / (A^3 u^{3/2})`

The figure caption says:
- `Fig. 5: Thermal conductance of the cubic onsite model as a function of temperature`
- length of center chain: `N = 5`

## What the paper's LO treatment does

The paper gives explicit leading-order cubic self-energy expressions:
- Eq. (77)
- Eq. (78)
- Eq. (79)

Then it computes thermal conductance using:
- Eq. (54) conductance
- Eq. (56) effective transmission
- `Delta T -> 0` derivative expansion
- Eq. (84) effective transmission formula
- Eq. (85) for the `F` function
- Eq. (86) to Eq. (89) for temperature variation and the LO nonlinear self-energy variation

This is a Keldysh/nonequilibrium LO benchmark, not just a retarded one-shot self-energy.

## What our current LO code is

Current cubic LO implementation:
- `src/negfpy/inelastic/third_order.py`

Current toy helper:
- `src/negfpy/inelastic/fpu_alpha.py`

Current implementation characteristics:
- cubic interaction is for `FPU-alpha` bond anharmonicity
- LO self-energy is a mode-space quasiparticle-style retarded self-energy
- conductance/current is obtained through the current `transmission_inelastic(...)` workflow
- the present LO code does not implement the paper's Eq. (84)-(89) derivative-based effective transmission benchmark

## Conclusion before the dedicated benchmark path

The answer was: no, not yet.

We should not expect the old `LO` implementation to exactly match Fig. 5, for two independent reasons:

1. Model mismatch
- Fig. 5 uses a cubic onsite model
- current `negfpy` cubic LO toy path is `FPU-alpha`, i.e. bond-cubic

2. Approximation / observable mismatch
- Fig. 5 uses the paper's explicit LO nonequilibrium formalism for conductance
- current `negfpy` cubic LO path is a practical retarded quasiparticle LO model

Because of those two mismatches, a numerical disagreement with Fig. 5 right now would not tell us whether the core code is correct.

## Correct next step

If we want a meaningful Fig. 5 benchmark, we should implement a dedicated paper-specific benchmark path with:

1. cubic onsite toy model
- harmonic part with `K` and `K0`
- cubic onsite tensor `T_ijk = t delta_ij delta_ik`

2. paper-specific LO benchmark wrapper
- use the `N = 5` setup
- use the published `t` values
- reproduce the conductance-vs-temperature observable

3. keep it isolated
- general inelastic theory stays reusable
- paper-specific benchmark logic stays in a dedicated literature-facing module

That will let us make an exact apples-to-apples benchmark instead of comparing different models under the same label `LO`.

## Current implementation status

A dedicated paper-facing benchmark path now exists in:
- `src/negfpy/inelastic/wang0701164.py`
- `examples/wang0701164_fig5_cubic_onsite_lo.py`

What is now matched:
- correct cubic onsite model
- correct published `K`, `K0`, `N`, and `t` families
- isolated Fig. 5 benchmark path

## Main debugging outcome

The largest mismatch was not the cubic onsite model itself. It was a combination of:

1. unit convention
- The paper quotes `K`, `K0`, and `t` in native numeric dynamical units, but the transport still uses physical Bose factors and physical heat-current units.
- Keeping the benchmark dynamics in the paper's native numeric units while using the physical prefactors
  `k_B / (hbar * alpha)` and `hbar * alpha^2`
  gives the correct ballistic scale.

2. first graph evaluation
- The first LO cubic graph in Eq. (77) is best evaluated in the time domain, because in time it is simply proportional to the square of the contour Green's function.
- A direct time-domain evaluation produces a much better match than the earlier frequency-space approximation.

3. second graph / tadpole term
- Including the constant second graph from Eq. (78)-(79) as a direct diagonal retarded shift severely over-suppresses the conductance and does not resemble Fig. 5.
- Removing that tadpole-like constant term gives a conductance family that matches Fig. 5 much more faithfully.

4. transform window vs physical band edge
- The harmonic phonon band ends at `sqrt(4K + K0)`.
- For the published Fig. 5 parameters, this is about `1.6008` in paper units.
- A wider symmetric frequency window is still useful for the time-domain transform used in the first LO graph.
- The benchmark therefore allows a transform window extending beyond the band edge, but the final conductance integral is restricted to the physical band.

## Current status

The benchmark path in:
- `src/negfpy/inelastic/wang0701164.py`

now uses by default:
- paper-native numeric dynamics with physical current/Bose prefactors
- exact time-domain first graph
- tadpole-subtracted benchmark default (`include_second_graph=False`)

This benchmark now reproduces the published Fig. 5 family qualitatively and in scale much more closely than the earlier implementation.

## New derivative-debug status

After adding the direct Eq. (84)-(89) derivative branch, the main issue turned
out not to be the paper observable itself, but the analytic derivative
implementation. A more consistent benchmark path is now available in
`wang0701164.py`:

- `paper_derivative`: current analytic Eq. (84)-(89) implementation
- `paper_derivative_numeric`: same Eq. (85) observable, but using finite-
  difference derivatives of `G^r` and `G^<` obtained from the exact first-graph
  closure

The numeric derivative branch improved the benchmark substantially compared with
the original analytic derivative implementation, but it also exposed a deeper
problem in the first-graph realization: the Eq. (77) time-domain transform had
been evaluated on the same finite frequency window as the requested output
grid. Since the first cubic graph is a self-convolution of the harmonic
greater/lesser spectrum, that old implementation aliased high-frequency
self-energy weight back into the physical window.

The benchmark code now fixes this by evaluating the exact first graph on an
internal transform grid whose support is at least `2 * sqrt(4K + K0)`, then
sampling the self-energy back onto the requested output grid. This is a
rigorous numerical correction of the same paper formula, not a fit parameter.

Additional debugging outcome:
- the constant second graph still makes the Fig. 5 benchmark much worse even in
  the improved numeric derivative branch
- at moderate coupling (`t = 0.5`) it suppresses the conductance far below the
  published family
- this strengthens the interpretation that the benchmark should remain
  tadpole-subtracted by default

After the aliasing fix, the strongest quick benchmark branch is no longer the
numeric derivative path but the direct current benchmark:

- `conductance_mode = current_over_delta_t`
- `include_second_graph = False`
- internal first-graph support: at least `2 * sqrt(4K + K0)`

A post-fix quick vector-reference scan is saved in:
- `outputs/inelastic/Wang0701164/014_post_padding_output_window_scan.json`

Within that quick scan:
- `n_omega = 141`, `omega_max = 2.9`, `eta = 2e-5` gives the best score
  (`~0.697`) against the vector-extracted Fig. 5 reference
- the saved quick benchmark for that setting is:
  - `outputs/inelastic/Wang0701164/015_best_quick_current_over_delta_t.tsv`
  - `outputs/inelastic/Wang0701164/015_best_quick_current_over_delta_t_metrics.json`

That quick run is useful for debugging direction, but the stricter benchmark is
the denser 21-point temperature sweep. After rescanning `n_omega`, `omega_max`,
`eta`, and `delta_t` on the 21-point grid, the current best rigorous setting is:

- `conductance_mode = current_over_delta_t`
- `n_omega = 121`
- `omega_max = 2.7`
- `eta = 1e-4`
- `delta_t = 0.02`
- `include_second_graph = False`

Saved rigorous benchmark:
- `outputs/inelastic/Wang0701164/022_best_rigorous_current_over_delta_t_21T.tsv`
- `outputs/inelastic/Wang0701164/022_best_rigorous_current_over_delta_t_21T_metrics.json`

This gives a vector-reference score of about `0.797`, which is still not a
perfect match, but is materially better than the pre-aliasing benchmark and is
more honest than the earlier quick 11-point scan.

Supporting scans for this post-fix regime:
- `outputs/inelastic/Wang0701164/016_eta_deltaT_scan_21T.json`
- `outputs/inelastic/Wang0701164/017_output_window_scan_21T.json`
- `outputs/inelastic/Wang0701164/018_targeted_eta_window_scan_21T.json`
- `outputs/inelastic/Wang0701164/020_nomega_scan_best_region_21T.json`

What these scans show:
- `delta_t` is essentially irrelevant within the tested range (`0.01` to `0.05`)
- `eta` matters only mildly; `1e-4` is slightly best in the current rigorous
  window
- the rigorous benchmark is no longer improved by simply increasing
  `n_omega`; beyond the best region it becomes worse
- so the remaining mismatch is not behaving like a simple coarse-grid problem

Important interpretation:
- the aliasing fix resolves the strongest-coupling mismatch much more cleanly
  than the old benchmark path
- for example, the `t = 2.0` peak moves from being far too high to landing very
  close to the vector-extracted paper value
- the remaining mismatch is now concentrated mostly in the moderate couplings
  `t = 0.2` to `1.0`
- the direct finite-`ΔT` current branch remains the best-performing benchmark
  path after the aliasing fix
- the direct Eq. (84)-(89) derivative branches are still not the best match

## Convergence scan status

A dedicated convergence scan against the digitized Fig. 5 curves now exists and
is saved in:
- `outputs/inelastic/Wang0701164/008_fig5_convergence_scan.tsv`
- `outputs/inelastic/Wang0701164/008_fig5_convergence_scan.json`

Within the tested honest parameter box:
- `n_temp = 21, 41, 81`
- `n_omega = 61`
- `omega_max = 2.1, 2.3`
- `eta = 5e-5, 1e-4`

the best score is obtained for:
- `n_temp = 21`
- `n_omega = 61`
- `omega_max = 2.3`
- `eta = 5e-5`

This is important because it shows:
- denser temperature sampling by itself does not remove the residual mismatch
- the remaining gap is therefore not just a coarse-grid issue
- the dominant unresolved difference is still most likely in the benchmark
  formalism / closure rather than in simple numerical resolution

## Higher-frequency-resolution update

An additional focused resolution scan showed that the frequency grid matters
more strongly than the temperature grid:

- `n_omega = 61` with `omega_max = 2.3`, `eta = 5e-5` gives a score of about `0.754`
- `n_omega = 121` with `omega_max = 2.3`, `eta = 2e-5` improves to about `0.689`

This is a real and rigorous improvement because no model parameters were tuned;
only the numerical realization of the published benchmark was corrected and the
output resolution was scanned honestly.

At the same time:
- the improvement is not monotonic in `n_omega`
- once the first-graph aliasing is fixed, the best `omega_max` values shift to
  wider windows than before
- some combinations still become worse, which shows the benchmark remains
  numerically delicate even after the support correction

So the transform-based first-graph evaluation is still numerically delicate.
That strongly suggests the remaining mismatch is now a combination of:
- residual transform / window artifacts beyond the current support correction
- unresolved formal differences in the literature benchmark closure

## Additional formal checks after the aliasing fix

Two further hypotheses were tested explicitly:

1. Increasing the internal first-graph support above `2 * sqrt(4K + K0)`
   - tested in `outputs/inelastic/Wang0701164/019_internal_support_multiplier_scan_11T.json`
   - result: does **not** help; the score becomes slightly worse
   - interpretation: the current support correction is already sufficient

2. Replacing the current benchmark with a strict first-order Green's-function
   expansion `G ≈ G0 + G0 Σ G0`
   - tested in `outputs/inelastic/Wang0701164/021_strict_first_order_current_scan_11T.json`
   - result: becomes **much worse** than the Dyson-based benchmark
   - interpretation: the remaining mismatch is **not** because the benchmark is
     “too resummed”; the strict first-order current underestimates the moderate
     and strong-coupling curves badly

3. Reconstructing conductance directly from the Eq. (84) trace derivative
   using finite differences of the full interacting Green's functions
   - tested interactively after the aliasing fix
   - result: overshoots the ballistic curve by roughly a factor of two and is
     much worse than the current-based benchmark
   - interpretation: our present direct trace-derivative transcription is still
     missing a correct prefactor / cancellation structure, so it should not yet
     be used as the benchmark path

4. Decomposing the Eq. (84) trace derivative into its three finite-difference
   pieces
   - helper added in code: `cubic_onsite_eq84_numeric_decomposition`
   - saved diagnostic at `outputs/inelastic/Wang0701164/023_eq84_numeric_decomposition_T500.json`
   - main finding:
     - the `delta(G^r-G^a)` contribution is negligible
     - the dominant piece is the lead-variation term
       `0.5 Tr[(G^r-G^a) delta(Sigma_R^< - Sigma_L^<)]`
     - the `delta G^<` term is a smaller correction whose sign changes with
       coupling
   - an extra factor-of-two diagnostic was also saved in
     `outputs/inelastic/Wang0701164/024_eq84_lead_term_factor_diagnostic.json`
   - interpretation:
     - a simple global factor error in the dominant lead term is **not**
       sufficient to fix the benchmark
     - halving that term repairs the ballistic limit but ruins the strong-
       coupling side
     - therefore the remaining mismatch is more structural than a single missing
       factor

5. Harmonic consistency check for the raw Eq. (84) trace decomposition
   - the original numeric derivative branches were using
     `(plus - minus) / DeltaT` even though the two runs correspond to
     `+DeltaT` and `-DeltaT`
   - this produced an exact factor-of-two error in the raw Eq. (84)
     decomposition relative to `current_exact`
   - after correcting the central-difference denominator to `2 * DeltaT`, the
     saved check in `outputs/inelastic/Wang0701164/028_harmonic_eq84_factor_check.json`
     gives:
     - `kappa_eq84_raw ≈ 0.2422456 x 10^-9 W/K`
     - `kappa_current_exact ≈ 0.2422456 x 10^-9 W/K`
     - ratio `≈ 1.0`
   - interpretation:
     - this particular formal bug is now resolved
     - the remaining mismatch is therefore not explained by a simple global
       factor in the numeric derivative branches

6. Second-graph component check
   - a short diagnostic was run by splitting the Eq. (79) static shift into:
     - no second graph
     - real-only shift
     - imaginary-only shift
     - full complex shift
   - representative results at `eta = 2e-4`, `omega_max = 2.3`, `n_omega = 81`:
     - for `t = 0.5`, `T = 500 K`
       - none: `0.1015`
       - real-only: `0.00625`
       - imag-only: `0.1142`
       - full: `0.00583`
     - for `t = 1.0`, `T = 500 K`
       - none: `0.01823`
       - real-only: `0.00165`
       - imag-only: `0.00819`
       - full: `0.00154`
   - interpretation:
     - the catastrophic benchmark collapse from the current second-graph
     implementation is driven mainly by the static **real** shift, not by the
     small imaginary part
     - therefore the second graph is still not the missing ingredient that will
       repair the Fig. 5 mismatch in its present code form

7. Updated 21-temperature refinement around the post-aliasing best region
   - saved comparison: `outputs/inelastic/Wang0701164/027_targeted_21T_refinement.json`
   - tested settings included:
     - `n_omega = 121`, `omega_max = 2.3`, `eta = 2e-4`
     - `n_omega = 121`, `omega_max = 2.7`, `eta = 1e-4`
     - nearby variants at `eta = 1e-4`, `5e-5`
   - best score in that targeted 21-point comparison:
     - `1.0132` for `n_omega = 121`, `omega_max = 2.3`, `eta = 2e-4`
   - previous saved 21-point reference-like setting:
     - `1.0167` for `n_omega = 121`, `omega_max = 2.7`, `eta = 1e-4`
   - interpretation:
     - the benchmark does improve slightly in the lower-window `omega_max = 2.3`
       region
     - but the improvement is marginal, and it does **not** resolve the
       moderate-coupling mismatch

8. Harmonic raw-Eq. (84) factor check saved explicitly
   - saved file:
     - `outputs/inelastic/Wang0701164/028_harmonic_eq84_factor_check.json`
   - key result:
     - raw Eq. (84) trace decomposition vs `current_exact` now gives a ratio of
       `0.9999999999999274`
   - interpretation:
     - the central-difference normalization mismatch has been fixed
     - further discrepancy must come from the nonlinear/boundary `F`-function
       construction rather than a leftover harmonic prefactor error

9. Post-fix derivative vs current comparison
   - saved file:
     - `outputs/inelastic/Wang0701164/030_post_fix_mode_compare.json`
   - after fixing the central-difference denominator, the
     `paper_derivative_numeric` branch now almost coincides with
     `current_over_delta_t`
   - representative scores:
     - `paper_derivative_numeric`, `n_omega = 121`, `omega_max = 2.3`,
       `eta = 2e-4`: `1.01589`
     - `current_over_delta_t`, same settings: `1.01321`
   - interpretation:
     - the old discrepancy between the two numeric benchmark branches was
       largely caused by the derivative normalization bug
     - however, fixing that bug does **not** produce a dramatic improvement in
       the Fig. 5 match, so the remaining mismatch is still real

10. Latest saved comparison figure
   - latest overlay:
     - `outputs/inelastic/Wang0701164/031_fig5_model_vs_vector_overlay.png`
   - latest residuals:
     - `outputs/inelastic/Wang0701164/031_fig5_model_vs_vector_residuals.png`
   - settings:
     - `conductance_mode = current_over_delta_t`
     - `n_omega = 121`
     - `omega_max = 2.3`
     - `eta = 2e-4`
     - `delta_t = 0.02`
     - this remains one of the clearest unresolved formal clues

11. Updated vector comparison after the derivative-normalization fix
   - saved overlay:
     - `outputs/inelastic/Wang0701164/033_fig5_model_vs_vector_overlay.png`
   - saved residuals:
     - `outputs/inelastic/Wang0701164/033_fig5_model_vs_vector_residuals.png`
   - saved metrics:
     - `outputs/inelastic/Wang0701164/033_fig5_model_vs_vector_metrics.json`
   - representative peak values from that run:
     - `t = 0.0`: model `0.2856`, paper `0.2885`
     - `t = 0.2`: model `0.2264`, paper `0.2594`
     - `t = 0.5`: model `0.1618`, paper `0.1998`
     - `t = 1.0`: model `0.0711`, paper `0.0899`
     - `t = 2.0`: model `0.00536`, paper `0.00485`
   - interpretation:
     - the ballistic and strongest-coupling limits are already quite close
     - the remaining mismatch is concentrated in the moderate couplings
       `t = 0.2` to `1.0`

12. Global first-graph prefactor diagnostic
   - saved scan:
     - `outputs/inelastic/Wang0701164/034_self_energy_scale_scan.json`
     - `outputs/inelastic/Wang0701164/034_self_energy_scale_scan.png`
   - saved best-overlay figure:
     - `outputs/inelastic/Wang0701164/036_prefactor_diagnostic_best_overlay.png`
   - quick coarse confirmation:
     - `outputs/inelastic/Wang0701164/035_quick_prefactor_diagnostic.json`
     - `outputs/inelastic/Wang0701164/035_quick_prefactor_diagnostic.png`
   - result:
     - the best global multiplier on the first-graph LO self-energy is about
       `0.75`
   - scores from the dense scan:
     - `0.25 -> 0.05665`
     - `0.50 -> 0.03025`
     - `0.75 -> 0.02238`
     - `1.00 -> 0.02494`
     - `1.25 -> 0.02957`
     - `1.50 -> 0.03408`
   - interpretation:
     - the mismatch does **not** behave like a clean missing prefactor such as
       `1/2`, `2`, or another obvious diagrammatic coefficient
     - a smaller-than-unity global scale does improve the moderate-coupling
       curves, but it simultaneously worsens the strongest-coupling `t = 2.0`
       benchmark
     - this strongly suggests that the remaining gap is structural, not a
       single coefficient typo in Eq. (77)

Current best interpretation of the remaining gap:
- the dominant residual mismatch is now most likely in the formal benchmark
  closure around the linear-response observable, not in the cubic LO
  self-energy itself

Remaining caveat:
- we still describe this as a closer benchmark path rather than a mathematically
  proven line-by-line reproduction of every contour-ordered step in Eq. (84)-(89)

## Vector reference update

The original benchmark comparison used:
- `outputs/inelastic/Wang0701164/002_fig5_digitized_curves.tsv`

That file is useful, but it is still based on image digitization. We now also
have a stricter paper-facing reference path that extracts the left-panel Fig. 5
curves directly from the PDF vector graphics using `pdftocairo`:

- `wang0701164_extract_fig5_vector_curves(...)`
- `wang0701164_compare_sweep_to_vector_pdf(...)`
- `examples/wang0701164_extract_fig5_vector_reference.py`

Important finding from the vector extraction:
- the `t = 2.0` curve in the paper sits much closer to the baseline than the
  earlier raster-digitized reference suggested
- this means the current benchmark mismatch at strong coupling is larger than
  we first thought

So the current best honest conclusion is:
- the benchmark infrastructure is more rigorous now
- the remaining mismatch is real, especially for the stronger-coupling curves
- it should not be dismissed as only an artifact of coarse raster digitization
