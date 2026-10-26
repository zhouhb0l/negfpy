# NEGFPy

NEGFPy provides harmonic open-device phonon NEGF, periodic-crystal phonons and
intrinsic three-phonon SMRTA transport, and research cubic anharmonic NEGF.
The current version is **2.0.0.dev0**, being prepared for a future release.

The source checkout contains reusable library code, tests, documentation and
portable examples. Material files, run outputs and historical experiments live
outside the package. Raw IFCs are not silently corrected or replaced.

## Installation

Python 3.11 or newer is required. Local preparation checks use Python 3.12.

```bash
python -m pip install .
```

The optional historical silicon oracle environment pins phono3py 4.4.0:

```bash
python -m pip install '.[material-oracle]'
```

## Workflows

- **Harmonic open-device NEGF:** device and contact Green functions,
  transmission and heat currents through the established modeling/workflow APIs.
- **Periodic bulk phonons and SMRTA:** FC2 bands, DOS and group velocities;
  explicit FC3 rates, lifetimes and intrinsic three-phonon conductivity through
  `negfpy.bulk`. See the [bulk guide](docs/bulk_rta.md) and
  [silicon example](examples/bulk_rta_silicon.py).
- **Research anharmonic device NEGF:** independent native retarded,
  statistical and spectral channels, four-term Born maps, bounded SCBA and
  checkpoint continuation. The fitted complete-energy path has a consolidated
  [Python API](src/negfpy/anharmonic/complete_energy.py),
  [portable example](examples/complete_energy_scba.py) and
  [usage guide](docs/complete_energy_scba.md).

A safe FC2-only bulk starting point is:

```python
import negfpy.bulk as bulk

problem = bulk.BulkHarmonicProblem.from_phono3py(
    "phono3py_disp.yaml", "fc2.hdf5",
)
spectrum = problem.compute_uniform_mesh_phonons((11, 11, 11))
dos = bulk.compute_phonon_dos(spectrum, sigma_thz=0.1)
```

Heavy FC3 operations and full SCBA require explicit inputs and resource bounds;
importing the library or portable example does not launch a calculation.

## Model and result scope

FC3 already defines a cubic Taylor energy; SCBA does not require fitting.
The complete-center-energy fit is an explicit compact boundary-model choice
for the available bulk IFCs. Selecting whole energies before differentiation
retains their required endpoint derivatives. It does not uniquely reconstruct
an interface from bulk IFCs.

The preserved fitted-model study reached the original six SCBA convergence
screens on its 4-q/601-frequency grid with unchanged physical guards. Its bulk
IFC fit error is about **2.14147%** in the reported Cartesian Frobenius metric.
This is evidence for that model and grid, not validated physical-interface,
production-IFC, transport or mesh accuracy. Actual IFC reference calculations
and fitted-model results remain distinct. See the guide for definitions.

Bulk SMRTA and open-device interacting transport remain separate methods.
Bulk lifetimes are not substituted for a conserving open-device self-energy.

## Development and release preparation

```bash
python -m pip install '.[test]'
python -m pytest
```

The test suite includes expensive optional cases; read their resource markers
before selecting broad scientific runs. Maintainer utilities remain in
`scripts/`; dated multi-q HPC launchers were removed from the active tree and
are preserved in Git history and workspace recovery archives. The installed
package does not depend on those launchers.

The [platform guide](PROJECT_NOTES.md),
[workspace layout](docs/development/workspace_layout.md),
[release preparation note](docs/development/release_preparation.md) and
[changelog](CHANGELOG.md) describe the boundaries and remaining release work.
No new software release is published by this consolidation.
