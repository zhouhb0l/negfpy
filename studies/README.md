# Studies Workspace

Use this folder for application-scale calculations.

Recommended pattern:
- one folder per study campaign
- configs under `inputs/configs/`
- IFC inputs under `inputs/ifc/` (recommended copy for reproducibility)
- run outputs auto-written to `runs/<timestamp>_<run_name>/` in study mode
- publication products under `analysis/`

Run with:

```bash
python examples/run_ifc_bulk.py --input studies/<study>/inputs/configs/<config>.json
```
