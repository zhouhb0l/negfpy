# IFC Bulk Runner (QE `.fc` Quick Guide)

This workflow runs bulk phonon transport calculations directly from IFC files
without writing a material-specific Python script.

Supported calculations:
- `transmission`
- `dos`
- `dispersion`

Entry point:

```bash
python examples/run_ifc_bulk.py --input <config.json>
```

## Output Modes

`run.output_mode` controls where results are written:
- `flat` (default): writes to `run.output_dir` (legacy behavior)
- `study`: writes to `studies/<study>/<material>/runs/<timestamp>_<run_name>/`

Study mode also writes:
- `input.json` snapshot inside each run folder
- `manifest.jsonl` (append-only run index) in the study root
- provenance in `*_report.json` (config hash, IFC hash, git head, git dirty flag, hostname)

## Minimal Workflow

1. Put your `.fc` file in the project root (or use a relative/absolute path).
2. Generate a template:

```bash
python examples/run_ifc_bulk.py --write-template examples/configs/ifc_bulk_template.json
```

3. Edit the template (mainly `ifc.path`, `run.calculation`, mesh/range settings).
4. Run:

```bash
python examples/run_ifc_bulk.py --input examples/configs/ifc_bulk_template.json
```

5. Check outputs in `run.output_dir`:
- `<run_name>_<calc>.tsv`
- `<run_name>_<calc>.png`
- `<run_name>_<calc>_report.json`
- `<run_name>_<calc>_eta_diag.json` (for transmission/DOS)

## Fast Start Config (Transmission)

Use these defaults when starting with a new QE `.fc`:
- `run.calculation`: `transmission`
- `ifc.reader`: `qe`
- `ifc.enforce_asr`: `false` (set `true` only if you need explicit ASR correction)
- `solver.surface_gf_method`: `generalized_eigen_svd`
- `solver.eta_scheme`: `adaptive`
- `solver.eta_values`: `[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]`
- `kmesh.mode`: `shifted` (or `auto`)

Typical mesh choices:
- 2D transverse averaging: `kmesh.dimension = 2`, start with `nk = 16~20`
- 1D transverse averaging: `kmesh.dimension = 1`, start with `nk = 24`
- low-frequency denser mesh: set `nk_low > nk` and `low_cm1` (example: `120.0`)

Frequency range:
- set `omega.fmax_cm1` if known from your material
- otherwise use `omega.auto_max = true`

## Config Sections

- `run`: job name, calculation type, output path, plot/data/report switches
- `ifc`: IFC path(s), reader, ASR, optional `dy_cutoff` and `dz_cutoff`
- `model`: principal-layer settings, `n_layers`, assembly options
- `kmesh`: dimensionality, sampling mode, high/low mesh controls
- `omega`: min/max and number of frequency points
- `solver`: SGF method, eta strategy, quality filters
- `dos`: DOS-specific options
- `dispersion`: dispersion path options

Important `run` fields for application-scale organization:
- `output_mode`: `flat` or `study`
- `studies_root`: root folder for study-mode runs
- `study`: study name (folder)
- `material`: material name (folder)
- `timestamped_run_dir`: include UTC timestamp prefix in run folder
- `write_input_snapshot`: copy full input JSON into each run folder
- `manifest_jsonl`: run index filename in study root

## Mesh Modes

`kmesh.mode` supports:
- `auto`: shifted for even `nk`, centered for odd `nk`
- `shifted`: midpoint grid in `[-pi, pi)`
- `centered`: uniform grid with spacing `2pi/nk`
- `legacy_endpoint`: endpoint-inclusive `linspace(-pi, pi, nk)` (for strict old benchmark matching)

## Eta Behavior

- `solver.eta_scheme = fixed`: single eta for all k-points
- `solver.eta_scheme = adaptive`: eta chosen per k-point
- `solver.eta_scheme = adaptive-global`: one eta chosen per omega for all k-points
- `solver.eta_device`: explicit device broadening override; if `null`, follows lead eta used by the current scheme

## Reproducibility Tips

To match a previous benchmark, make these identical:
- IFC file and reader
- `ifc.enforce_asr`
- `model.n_layers` and principal-layer options
- full `kmesh` definition (`dimension`, `mode`, `nk`, `nk_low`, `low_cm1`)
- omega grid (`min`, `max` or `fmax_cm1`, `n_points`)
- solver settings (`surface_gf_method`, eta scheme, eta ladder, `eta_device`)

The run report JSON stores all resolved settings and is the best record for reruns.

## Common Issues

`ValueError: shifted k-mesh requires even nk`:
- use even `nk`, or switch to `kmesh.mode = centered`.

Very long runtime:
- reduce `omega.n_points`
- reduce `nk` / `nk_low`
- avoid excessive `n_layers`

Unexpected transmission differences between runs:
- compare the two `*_report.json` files field-by-field
- especially check `n_layers`, `kmesh.mode`, and eta settings

## Example Configs

- `examples/configs/ifc_bulk_graphene_transmission.json`
- `examples/configs/ifc_bulk_silicon_transmission.json`
- `examples/configs/ifc_bulk_graphene_benchmark_match.json`
