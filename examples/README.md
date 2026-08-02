# Examples

The repository contains API examples but does not bundle material IFC files:

- `anharmonic_toy_scba.py` runs the reduced-unit anharmonic toy model.
- `bulk_rta_silicon.py` safely runs audited FC2-only bands, DOS, and group
  velocities by default; FC3/SMRTA requires two explicit heavy-work flags.
- `run_ifc_bulk.py` is the generic entry point for harmonic calculations from an
  IFC configuration.
- `configs/ifc_bulk_template.json` is the generic IFC configuration template.

Run the toy example with:

```bash
python examples/anharmonic_toy_scba.py
```

Run the safe FC2-only silicon material example with:

```bash
python examples/bulk_rta_silicon.py \
  --crystal /path/to/phono3py_disp.yaml \
  --fc2 /path/to/fc2.hdf5
```

The cubic contraction is intentionally unreachable by default. Request it
only in a heavy-work environment and archive the result outside the source
tree:

```bash
python examples/bulk_rta_silicon.py \
  --crystal /path/to/phono3py_disp.yaml \
  --fc2 /path/to/fc2.hdf5 \
  --fc3 /path/to/fc3.hdf5 \
  --run-fc3 --allow-heavy \
  --output ../artifacts/silicon-rta.h5
```

The HDF5 output uses the checked `negfpy.bulk` RTA archive schema. The default
3x3x3 mesh is an API exercise, not a converged material result. See
[`docs/bulk_rta.md`](../docs/bulk_rta.md) for conventions and convergence
requirements.

Generate and edit an IFC configuration before running a material calculation:

```bash
python examples/run_ifc_bulk.py \
  --write-template examples/configs/ifc_bulk_template.json
python examples/run_ifc_bulk.py --input path/to/config.json
```

See `examples/configs/README_ifc_bulk.md` for the configuration reference.

Material-specific drivers, IFC inputs, and generated results belong in the
workspace-level `../studies/` tree rather than in the source repository. Legacy
implementations and superseded examples are retained in the workspace archive.
