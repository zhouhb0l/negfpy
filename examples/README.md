# Examples Overview

## Primary Entry Point

Use the bulk IFC runner for day-to-day material calculations from `.fc` files:

```bash
python examples/run_ifc_bulk.py --input examples/configs/ifc_bulk_template.json
```

Bulk runner docs and config templates:
- `examples/configs/README_ifc_bulk.md`
- `examples/configs/ifc_bulk_template.json`

## Validation / Benchmark Scripts

These are kept at top-level because they are useful for solver and regression checks:
- `examples/graphene_ifc_transmission_benchmark.py`
- `examples/graphene_kmatrix_legacy_benchmark.py`
- `examples/silicon_ifc_dispersion_compare.py`
- `examples/chain_inelastic_validation.py`
- `examples/fpu_alpha_lowest_order_conductance.py`
- `examples/phi4_approximations_conductance.py`
- `examples/wang_fig4_quartic_scmf.py`
- `examples/wang0701164_fig5_cubic_onsite_lo.py`
- `examples/wang0701164_fig5_convergence.py`
- `examples/wang0701164_extract_fig5_vector_reference.py`

## Legacy Archive

Older one-off examples were moved to:
- `examples/legacy/examples_ifc/`
- `examples/legacy/toy_models/`

These scripts are kept for reference but are not the recommended interface for new runs.
