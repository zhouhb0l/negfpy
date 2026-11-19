# negfpp 1.2.0

`negfpp` calculates heat flow through a finite harmonic device with cubic phonon–phonon interactions and harmonic reservoirs. One JSON file specifies the model, geometry, frequency/q grids, temperatures, SCBA controls and outputs. Run the installed command directly:

```bash
negfpp examples/silicon24/input.json
```

## Install

Use Python 3.11 or newer in a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
negfpp --version
negfpp --check-input examples/silicon24/input.json
negfpp examples/silicon24/input.json
```

A release wheel can be installed with `python -m pip install dist/1.2.0/negfpp-1.2.0-py3-none-any.whl`. The source archive includes all three silicon examples and the compact frozen reference. `python -m negfpp INPUT.json` and `negfpp --input INPUT.json` are equivalent entry points.

## Start from a saved self-energy

Add optional `initial_self_energy` to the SCBA input to use a previous converged
run's evaluated self-energy as the initial guess. Specify its `fields.h5`,
`input.json` and `report.json`, with each file's SHA-256. The same direct command
then performs a fresh evaluation using the original physical and convergence
guards. The model, geometry, grids, coupling and reservoirs must match; solver
history is reset. Omitting the option preserves the original zero start.
See [saved self-energy inputs](docs/RESTART.md) for the JSON fields and checks.

## Rolling self-energy checkpoints and saved complete-energy models

Set `"checkpoint": {"file": "self-energy-latest.h5"}` to atomically replace one
self-energy file after every completed physical map, including a rejected trial.
It records the evaluated input R/C/D, its map ordinal and the separate selected
anchor index. It contains no Green functions, mapped image or solver history.
A bounded or rejected state remains an initial guess and is never accepted
transport. See [checkpoint and initial-guess inputs](docs/CHECKPOINTS.md).

Version 1.2.0 also reads authenticated, previously qualified complete site-energy
IFC2/IFC3 archives through `model.type: "saved_complete_energy"`. This supports
saved NEP-derived inputs without running a potential evaluator or fitting.
The command rebuilds the complete device support and paired contacts with the
original physical guards. See [saved complete-energy inputs](docs/SAVED_COMPLETE_ENERGY.md).

The numerical engine retains its existing internal `negfpy` namespace. Install this product in a separate environment from `negfpy-harmonic` or the full NEGFPy package.

## Capabilities

- Finite layered harmonic devices with exact complete-center cubic endpoint support.
- Self-consistent cubic Born approximation (SCBA), using the existing statistical/spectral channel solver and full nonlocal interaction blocks.
- Fixed mixing, strict depth-one Anderson acceleration, or reference-seeded SCBA using exact same-run reference/increment maps and bounded analytic feedback. Physical guards and all six convergence screens remain unchanged.
- Uniform transverse momentum meshes, paired-q thermal contacts and signed angular-frequency grids.
- Explicit CPU worker, memory, contraction-work, map-count and time-admission budgets.
- HDF5 harmonic FC2 and crystal mappings read through the existing phono3py readers; cubic energy models restored from saved coefficients and their authenticated geometry/fit report.
- Method 1 long-device extension: a qualified short-device retarded kernel, complete overlapping endpoint support, and integrated local fluctuation–dissipation/Bose probe closure. Exact recursive current actions use bounded CPU workers and fixed-size blocks.

Version 1.0 uses **saved finite-range site-energy models**. Its silicon coefficients are a fitted representation of bulk FC3, with a recorded 2.14147% Cartesian FC3 residual. They are not raw FC3 or a uniquely determined interface potential. The command restores them exactly without fitting. Arbitrary raw-FC3-to-device construction and extensions of the interaction range are outside this interface.

## Silicon check

The bundled example has 24 silicon atoms, three harmonic layers, eight cubic energy centers and 12 active endpoint atoms. It uses a coarse Gamma-only, 101-frequency grid, 300.5/299.5 K reservoirs and cubic coupling `g=0.10`. It starts from zero scattering self-energy. The example is intended as a short installation and calculation check; it does not establish frequency/q/eta convergence or interface accuracy.

## Reference-seeded SCBA and five-layer silicon

```bash
negfpp examples/silicon5layer/input.json
```

Select `scba.method: "reference-seeded"`. This reuses the existing reference/increment method: the first fresh evaluation starts from exact zero by default, or from the authenticated optional guess, and establishes a new reference for the same device, grid and contacts; subsequent maps use selected reference states, exact Green increments and all quadratic Born terms. Two warm-up maps precede strict Anderson updates and at most six analytic tangent attempts. Every nonlinear trial, including rejection, counts toward the 18-map limit. References are created within this run; no three-layer fields are padded or reused in the five-layer device.

The new example has **five full device layers, 40 Si atoms and three interacting center layers**. It retains the coarse Gamma/101-frequency grid, original saved model, `g=0.10` and 300.5/299.5 K contacts. The JSON sets a 900-second time-admission budget and four CPU workers. This is a bounded calculation check. Older version-1.0.0 JSON files continue to use their original methods and settings.

## Method 1 with local FD: silicon 1 µm

```bash
negfpp --check-input examples/silicon1um/input.json
negfpp examples/silicon1um/input.json
```

Local FD means **local fluctuation–dissipation**, using a Bose temperature for each complete translated scattering star. The supplied example extends the accepted 24-atom reference to 1,840 probe layers: **999.775 nm of center slab, 14,736 full-device atoms**. Its portable compact reference contains only the previously evaluated retarded self-energy, with the original report, input, actual successful-exit qualification and explicit hashes. The command rebuilds the same FC2/contacts, preserves the full overlapping 36-coordinate support, and solves one integrated zero-flux condition per probe. It performs no new Born contractions or SCBA maps.

The example uses four CPU workers, the reference's coarse Gamma/101-frequency grid, `g=0.10`, and 300.5/299.5 K reservoirs. It checks both nonlinear probe closure and zero-bias linear response with the original strict flux, optical, passivity and conservation guards. The 2,400-second admission budget includes preparation and a 60-second exit reserve. Numeric-array and process-memory admission is explicit in the JSON. No wrapper, private campaign directory or scheduler is required.

Method 1 is a **frozen short-device approximation**. It omits cross-center Born vertex pairs and does not establish full long-device SCBA self-consistency. This coarse installation check does not establish converged silicon conductivity. See [Method 1 inputs, approximation and outputs](docs/METHOD1.md).

## Results

The bundled input writes to `generated/silicon24` in the source directory (`../../generated/silicon24` relative to its JSON file). The command creates a new directory and refuses to overwrite an existing run. Change `run.output_dir` for a later calculation.

| File | Contents |
| --- | --- |
| `report.json` | Status, currents in W per transverse cell, six convergence screens, physical checks, model identity, versions, hashes and resource accounting |
| `input.json` | Exact run settings |
| `iterations.jsonl` | Each completed map, including rejected trials and selection decisions |
| `convergence.csv` | Three unmixed self-energy and three current-balance ratios for every completed map |
| `currents.csv` | Evaluated-state contact currents, collision power and balances |
| `fields.h5` | Selected evaluated Green/self-energy fields and the distinct unevaluated Born image, with frequency/q coordinates |
| `reference.json` | Reference-map arithmetic, tangent attempts, selected/rejected trials and proposal screens for reference-seeded SCBA |
| `SHA256SUMS` | Checksums of the saved outputs |

For finite-device SCBA, exit status is zero only after all six original screens pass with no physical/numerical failure. For Method 1, zero requires a qualified donor and successful nonlinear and linear-response probe gates. Bounded or failed runs return nonzero and preserve available diagnostics. A current from a partial run is not accepted interacting transport. Both contact currents are positive **into** the device; a conserving state has `J_left + J_right ≈ 0`.

See [input and model formats](docs/INPUT.md), [numerical conventions](docs/METHOD.md), [output definitions](docs/OUTPUTS.md) and [source provenance](SOURCE_PROVENANCE.json).
