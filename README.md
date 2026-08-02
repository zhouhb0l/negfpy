# NEGFPy 1.0

NEGFPy 1.0 is the validated bulk-phonon and harmonic-NEGF release line. It
provides two deliberately separate workflows:

- open-system harmonic phonon transport with the established NEGF API; and
- periodic-crystal phonons plus intrinsic three-phonon thermal conductivity
  in the single-mode relaxation-time approximation (SMRTA).

The bulk API includes audited phono3py FC2/FC3 ingestion, phonon band
structures, total and projected DOS, analytical group velocities, on-shell
three-phonon rates, population lifetimes, mean free paths, mode-resolved and
tensor thermal conductivity, cumulative conductivity, and checked HDF5
archives. Raw IFCs are not silently corrected, symmetrized, truncated, or
remapped.

## Installation

```bash
python -m pip install .
```

The pinned silicon validation environment uses phono3py 4.4.0:

```bash
python -m pip install '.[material-oracle]'
```

## Safe starting point

FC2-only observables do not require cubic force constants:

```python
import negfpy.bulk as bulk

problem = bulk.BulkHarmonicProblem.from_phono3py(
    "phono3py_disp.yaml",
    "fc2.hdf5",
)
spectrum = problem.compute_uniform_mesh_phonons((11, 11, 11))
dos = bulk.compute_phonon_dos(spectrum, sigma_thz=0.1)
```

The full example computes bands, DOS, and group velocities by default. FC3 is
unreachable unless both heavy-work flags are supplied:

```bash
python examples/bulk_rta_silicon.py \
  --crystal phono3py_disp.yaml --fc2 fc2.hdf5
```

See [the bulk SMRTA guide](docs/bulk_rta.md) for the cubic workflow, linewidth
and lifetime conventions, archive formats, and convergence requirements.

## Scientific boundary

The 1.0 conductivity is intrinsic three-phonon SMRTA. It is not iterative
LBTE, and it excludes isotope, boundary, four-phonon, Wigner/interband, and
electron--phonon scattering unless a separately named model is added later.
Coarse-mesh silicon values in validation records establish implementation
agreement; they are not presented as converged material predictions.

Real-material anharmonic device NEGF is the planned 2.0 line. Bulk relaxation
times are an independent oracle and screening observable, not a substitute for
a conserving open-device phonon--phonon self-energy.

## Validation status

The repository preserves the earlier dense-SCBA toy release and silicon bulk
three-phonon oracle. A 1.0 tag is created only after the exact clean wheel
passes the strict remote band/DOS/group-velocity/rate/SMRTA gate and its
machine-readable artifacts and PDF report are retrieved and verified.

Project structure and historical boundaries are summarized in
[PROJECT_NOTES.md](PROJECT_NOTES.md). Release contents and exclusions are
listed in [CHANGELOG.md](CHANGELOG.md).
