# Method 1: frozen retarded kernel and local FD

Run the supplied portable example directly:

```bash
negfpp --check-input examples/silicon1um/input.json
negfpp examples/silicon1um/input.json
```

## Model and numerical method

Local FD denotes the fluctuation–dissipation/Bose closure, not finite-difference force constants. Method 1 freezes the **evaluated input** retarded self-energy of the accepted three-layer, one-complete-center silicon reference. Its real and imaginary parts are retained. For each translated complete star, `Gamma = i(SigmaR - SigmaA)`, `Sigma< = -i nB(omega,T) Gamma`, and `Sigma> = -i (1+nB) Gamma`. One temperature per star is shared by all q/frequency points and adjusted until its integrated heat flux vanishes. The full device lesser field need not be locally Bose.

All overlapping 36-coordinate star blocks remain intact. There is no coverage normalization, endpoint truncation, lifetime fit, PSD clipping or acoustic projection. The source kernel must pass its original passivity, signed q/frequency reality and rigid-translation checks. Rebuilt FC2, geometry, units and harmonic contacts must match the authenticated source. Its grid and coupling cannot be overridden in the target JSON. The supported source is the existing equal-mass silicon three-layer 72/36-DOF complete one-center layout; arbitrary materials and five-layer/84-DOF source transfer are not supported by this version.

The existing grouped recursive engine uses at most 48-coordinate blocks. Exact coupling-stencil reuse and current actions avoid the global dense Green inverse and all-pairs transmission table. Startup qualifies the spectral/optical identities and uniform-temperature currents. Each later current/Jacobian action has linear work and storage in target layer count. Total runtime also depends on bounded thermal-solver iterations; no unconditional linear-time claim is made. Workers own disjoint q/frequency tiles, retain every quadrature point and apply the global q mean once. BLAS threads are limited to one.

An approximate chain preconditioner accelerates the solver. All final residuals use the complete recursive transport action. The unchanged physical flux allowance is `1e-25 W + 1e-10 * contact_current_scale`; the inner linear relative tolerance is `1e-12`. Both finite-bias zero-probe-flux closure and independent linear response must succeed, including equilibrium, uniform-temperature Ward, contact and total balance checks. No new Born contractions, cubic fitting, cubic-vertex construction or SCBA maps occur.

This is a frozen short-device approximation that omits cross-center Born vertex pairs. A qualified donor does not establish full long-device SCBA self-consistency. Results are conditional on its supplied model, coupling, grid, eta and reference state.

## One JSON input

Use `examples/silicon1um/input.json` as the complete `negfpp.local-fd.v1` template. Unknown/duplicate keys, nonfinite numbers and invalid limits are rejected. Paths are relative to the input JSON. Output directories are exclusive and never overwritten.

| Section | Required contents |
| --- | --- |
| `frozen_source` | Compact `manifest` filename and its whole-file `sha256` |
| `geometry` | Positive integer `center_layers`; two harmonic contact cells are added |
| `reservoirs` | Left, right and linear-response equilibrium temperatures in kelvin |
| `resources` | CPU `workers` (at most 16), `frequency_tile`, cache/numeric/preparation byte limits, parent/runtime reserves, available-memory margin, shared `max_seconds` and `exit_reserve_seconds` |
| `solver` | Newton/Krylov/restart/line-search/current/Jacobian caps, per-solve `max_seconds` and `preconditioner_seconds`; restart is at most 20 |
| `run` | Filename-safe `name` and fresh `output_dir` |

The source manifest authenticates the original input/report/exit receipt, compact HDF5 and four harmonic/model assets before reading numerical arrays. The loader requires the six original SCBA screens, successful exit and selected evaluated-input identity. Whole original-HDF5, compact-file and selected-array hashes are distinct domains. The compact reference stores only the evaluated `sigma_input_evaluated/retarded`, frequency/q coordinates and complete active ordering. It never substitutes the unevaluated Born image. Source arrays are loaded without pickle and are immutable in memory. `--check-input` verifies these portable assets without constructing contacts or the long device.

Memory admission evaluates retained factors, coupling families, metadata, simultaneous worker startup workspace and parent input reserves for the exact declared dimensions. Preparation caps and separate process/library reserves must also fit the numeric cap and currently available memory. Actual retained numeric payload is checked in each worker. These are estimates and explicit reserves, not an OS-enforced RSS limit. Shared deadlines and callback/iteration limits bound work admission; an in-progress fixed-size tile/action can run until the next guard. Scheduler wall limits may be used externally for production runs.

## Supplied silicon check

The compact reference is the unchanged accepted version-1.0.0 24-atom result at `g=0.10`, Gamma only and 101 signed frequency nodes (50 strictly positive). It has 300.5/299.5 K contacts, `eta_lead=0.0015625` and zero device eta. The harmonic repeat is approximately 0.543356 nm. The target has 1,840 probe layers: center slab approximately 999.775 nm; 1,842 full device cells; 14,736 atoms; 44,208 DOF. This discretization is intended to check software operation at the requested length. It is not a frequency/q/eta-converged silicon conductivity calculation.

The JSON specifies four workers, frequency tile one, a 4 GiB retained-cache cap, a 20 GiB numeric cap, 2 GiB parent reserve, 4 GiB runtime reserve and 4 GiB available-memory margin. The numeric estimate is approximately 13.4 GiB, with a 17.4 GiB total reservation before the available-memory margin. It does not replay a historical high-resolution campaign. Previous finite-device examples remain compatible and retain their own original inputs.

## Standard outputs

| File | Contents |
| --- | --- |
| `report.json` | Donor qualification, target geometry, approximation, nonlinear/linear gates, currents, conductance, solver caps, source hashes, versions and elapsed time |
| `input.json` | Exact target JSON |
| `donor-qualification.json` | Verified source state, six-screen result and portable identity receipts |
| `resource-plan.json` | Exact dimension-based allocation estimates and admission controls |
| `temperatures.csv` | Left/right/probe temperatures, cell origins and finite-bias currents |
| `linear-response.csv` | Temperature shifts and currents per kelvin bias, with fixed contact shifts +0.5/-0.5 |
| `engine.json` | Cache/optical/stencil audits, q/frequency coverage, work counts and worker peak RSS |
| `progress.json`, `progress.jsonl` | Latest phase and append-only phase/callback history |
| `worker-failure.json` | Worker diagnostics when a worker guard fails |
| `SHA256SUMS` | Checksums of available output files |

Contacts are terminals 0 and 1; probes begin at terminal 2. All currents are positive into the device. Coordinates in profiles are cell origins; center-slab length and full cell-box length are reported separately. Currents are W per transverse cell and linear conductance is W/K per transverse cell. Worker peak RSS values are individual lifetime peaks and must not be summed as simultaneous measured RSS. The parent peak excludes workers.

Exit zero requires both nonlinear and linear-response gates plus qualified source and physical guards. Failed/bounded runs return nonzero and retain available diagnostics. The report never describes target probe closure as new SCBA convergence. Dense long-device fields are not exported.
