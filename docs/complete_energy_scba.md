# Fitted complete-energy SCBA

The import surface is `negfpy.anharmonic.complete_energy`. It exposes the
existing implementations directly; it does not introduce a second solver.
The [portable example](../examples/complete_energy_scba.py) accepts prepared
Python objects and has no scheduler, material paths or calculation at import.

## Inputs and model construction

Supply compatible crystal/device mappings, units and mass conventions,
harmonic dynamical matrices, signed-frequency and q grids, paired retarded and
thermal contacts, and an explicit cubic model. Record contact broadening and
temperatures with the result. No material or fitted coefficients are bundled.

`fit_site_energy_model` constructs a declared research surrogate and reports
its residual. `restore_site_energy_model` restores saved coefficients without
refitting. Keep the measured error and omitted remainder explicit. The
preserved study fitted 728 coefficients and reported a relative Cartesian
bulk FC3 Frobenius error of 0.021414695219655902. That number is not a bound on
heat-current error.

`build_site_energy_vertex(..., crop_derivatives=False)` selects complete
center energies and retains every required endpoint derivative. Its device
must contain all endpoint support. In the preserved study, center layers 2..6
had exact active support of 132 coordinates inside a 216-coordinate harmonic
device. This geometry is a study configuration, not a universal default.

FC3 already specifies a cubic Taylor energy. Fitting is optional: it supplied
a compact, translation-invariant finite-boundary research model when only
bulk IFC files were available. It is not a repair of the original IFCs and
does not determine a unique physical interface. Future potential-based input
can construct complete boundary derivatives directly, subject to its own
training and environment validity.

## Bounded solve and continuation

Use `ChannelSCBAConfig` to set the fresh full-map and time/resource budget.
`run_channel_scba` provides fixed mixing; `run_channel_accel` provides bounded
depth-one Anderson with explicit selection and proposal-feasibility policies.
The example offers exact-zero initialization and continuation functions.

For continuation, authenticate a checkpoint with the expected model/grid/
contact/source binding. Keep **evaluated input R/C/D** separate from its
**unevaluated mapped R/C/D image**. The next warm-up mixes the pair once.
Rejected completed trials consume the map budget and remain evidence;
`result.state` is the selected anchor, while `last_evaluated_state` may be a
later rejected trial. Neither role alone means scientific acceptance.

The example uses the existing strict original-component and actual-merit
nonincrease policy, finite-PSD-only proposal backtracking, one common bounded
real Anderson coefficient and one common trust factor. Proposal damping is
not a physical projection and does not relax a full-map guard. A hard
numerical, source, Green or causality failure stops the solve.

## Convergence and limits

The three original **unmixed** retarded/lesser/greater Sigma screens and three
original current screens together gate SCBA convergence. Mixing-step size,
selected-anchor count, auxiliary merit and a saved checkpoint do not replace
those six checks. Preserve the configured physical and convergence tolerances.
Inspect `converged`, `stop_reason` and `failure_reason` after any bounded run.

The preserved study converged at native state 13 after 11 fresh maps from
state 2. It used g=0.03, lead eta=0.0015625, device eta=0, 305/295 K and a
4-q/601-frequency grid, with the original physical guards and thresholds.
This establishes numerical self-consistency for the fitted model on that
configuration. Physical boundary accuracy, IFC approximation, frequency/q
mesh convergence and accepted transport require separate assessment.

Raw payload hashes and whole-file hashes describe different byte domains;
checkpoint code retains their recorded roles. Do not relabel fields from a
previous Hamiltonian, contact broadening or grid as compatible restart data.
