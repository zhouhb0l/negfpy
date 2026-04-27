# Benchmark vs General

This note separates:

1. the **general inelastic NEGF ideas** we want to keep for future material calculations
2. the **paper-specific assumptions** used to reproduce toy-model benchmark figures

This is especially important for the Wang `0701164v1` Fig. 5 benchmark path in:
- `src/negfpy/inelastic/wang0701164.py`

and the Wang `1303.7317v1` Fig. 4 benchmark path in:
- `src/negfpy/inelastic/wang.py`

## Why this distinction matters

For a benchmark, we want the code path to match the paper as closely as possible.
For future material calculations, we want the formalism to be:
- reusable
- physically systematic
- not tied to one toy model, one unit convention, or one numerical trick

So the benchmark code is allowed to be more specialized than the long-term theory layer.

## General inelastic formalism

These are the parts we want to keep general.

### 1. Interaction-order separation

We keep separate tracks for:
- third-order
- fourth-order

and for each:
- lowest order
- mean field
- SCBA

This structure lives under:
- `src/negfpy/inelastic/approximations/`

### 2. Device / interaction separation

The long-term theory should work with:
- a general harmonic device
- a general third-order tensor
- a general fourth-order tensor

That is the right structure for future material calculations where the tensors come from IFCs or derived couplings, not from toy-model formulas.

### 3. Solver / benchmark separation

The reusable solver should not assume:
- one-dimensionality
- onsite-only anharmonicity
- a specific chain
- a specific bath model from one paper

Those assumptions belong in benchmark wrappers, not in the general solver.

### 4. Clear approximation labeling

We should keep the meanings of:
- `LO`
- `MF`
- `SCBA`

consistent across toy models and materials.

## Wang 0701164 Fig. 5 benchmark specifics

These are **not** part of the general formalism. They are benchmark-specific choices used to reproduce Fig. 5.

### 1. Specific Hamiltonian

The benchmark uses:
- 1D monoatomic chain
- nearest-neighbor harmonic spring `K`
- onsite harmonic spring `K0`
- cubic **onsite** nonlinearity
- finite center length `N = 5`

This is implemented in:
- `CubicOnsiteChainParams`
- `wang0701164_fig5_params(...)`

### 2. Published parameter family

The benchmark hard-codes the paper’s values:
- `K = 0.625 eV/(A^2 u)`
- `K0 = 0.1 K = 0.0625 eV/(A^2 u)`
- `N = 5`
- `t = 0, 0.2, 0.5, 0.7, 1.0, 2.0 eV/(A^3 u^(3/2))`

This is correct for the benchmark, but it is obviously not general.

### 3. Analytic harmonic Green's functions

The benchmark uses the analytic `lambda`-root formulas for:
- `G0^r`
- `G0^<`
- lead self-energies

This is specific to the uniform cubic-onsite chain model.

It is useful for exact benchmarking, but future material calculations should not rely on it.

### 4. Paper-native numeric dynamics

The Fig. 5 benchmark is not treated as a fully naive SI rewrite.

Instead:
- the dynamics are kept in the paper’s native numeric units
- the physical Bose/current prefactors are restored separately

This is encoded in:
- `wang0701164_paper_unit_factors(...)`

This was a benchmark-matching choice, not a general theory principle.

### 5. Exact time-domain first LO graph

The first cubic LO diagram is evaluated in the time domain, because that reproduces Fig. 5 much better than the earlier approximate frequency-space implementation.

This is implemented in:
- `cubic_onsite_lowest_order_self_energies_exact(...)`

This is still legitimate LO physics, but it is benchmark-specialized.

### 6. Tadpole subtraction default

For the Fig. 5 benchmark path, the constant second graph is disabled by default:
- `include_second_graph = False`

Reason:
- including it as a direct retarded diagonal shift over-suppressed the conductance family
- the benchmark matched Fig. 5 much more faithfully after removing that term

This is one of the most important paper-specific choices.

It should **not** automatically become the default assumption for the general third-order material-ready code.

### 7. Benchmark-tuned frequency grid

The Fig. 5 path uses defaults chosen because they reproduce the paper much better:
- `omega_max ≈ 2.1` in paper units
- `n_omega = 61` on the nonnegative branch
- symmetric uniform grid around zero for the time transform

These are good benchmark defaults, not universal production defaults.

### 8. Small finite-ΔT conductance extraction

The current benchmark uses a small finite temperature difference:
- `delta_t = 0.1`

and then constructs the conductance numerically from the effective transmission.

This is close to the paper’s derivative construction, but it is still a benchmark choice.

### 9. Low-temperature positivity cleanup

Very small negative conductances from finite-grid noise are projected to zero in the benchmark output.

This is a numerical cleanup choice for the benchmark figure, not a formal theory statement.

## Wang 1303.7317 Fig. 4 benchmark specifics

The Fig. 4 benchmark in `wang.py` is also specialized.

### Specific benchmark assumptions there include:
- one-particle and two-particle quartic toy models
- Lorentz-Drude baths
- quartic SCMF approximation
- paper-specific parameter presets
- paper-facing unit mapping under audit

These also should remain isolated from the general inelastic theory layer.

## What should carry forward into the material-ready path

These lessons from the benchmark should influence the general implementation:

### 1. Benchmark-first validation is worth it

The benchmark work showed that a code path can look reasonable yet still be wrong because of:
- the wrong toy model
- the wrong observable
- the wrong unit convention
- a hidden diagram term that dominates numerically

That is a strong argument for validating each approximation track against a known toy benchmark before using it for materials.

### 2. Unit conventions must be explicit

Future material calculations should not rely on hidden assumptions about:
- mass normalization
- frequency units
- current prefactors
- interaction tensor units

Those need to be carried explicitly in the API or documented very carefully.

### 3. Diagram choices must be auditable

If we keep or remove specific diagram classes:
- tadpole terms
- static shifts
- self-consistent dressing

we should make that choice explicit in the code and metadata.

### 4. Benchmark wrappers should stay separate

The benchmark modules should remain isolated:
- `wang.py`
- `wang0701164.py`

The general theory should not quietly inherit:
- benchmark cutoffs
- benchmark-specific grid sizes
- benchmark-specific subtractions
- benchmark-only unit conventions

## Practical rule for this project

When adding a new inelastic feature, ask:

1. Is this part of the reusable theory?
2. Or is this only needed to reproduce one benchmark figure?

If it is only for one benchmark figure, it should stay in a paper-specific wrapper or literature-facing module.
