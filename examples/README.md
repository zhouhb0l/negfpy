# Examples

The repository contains only generic or self-contained examples:

- `anharmonic_toy_scba.py` runs the reduced-unit anharmonic toy model.
- `run_ifc_bulk.py` is the generic entry point for harmonic calculations from an
  IFC configuration.
- `configs/ifc_bulk_template.json` is the generic IFC configuration template.

Run the toy example with:

```bash
python examples/anharmonic_toy_scba.py
```

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
