# JSON input

This page describes finite-device `negfpp.run.v1` SCBA inputs. Version 1.0.2 also accepts `negfpp.local-fd.v1` for the existing frozen-kernel Method 1 extension; its complete template and controls are described in [METHOD1.md](METHOD1.md).

Use `examples/silicon24/input.json` as the complete template. All fields in that template are required. The additional top-level `reference` object is optional and applies only to `scba.method: "reference-seeded"`. Unknown or duplicate keys, nonfinite values and mismatched input hashes are rejected. Relative input and output paths resolve against the JSON file's directory, independently of the shell working directory.

| Section | Meaning |
| --- | --- |
| `schema` | `negfpp.run.v1` |
| `run` | Filename-safe `name` and new `output_dir` |
| `inputs_sha256` | Whole-file SHA-256 values for exactly the four referenced files |
| `harmonic` | Full FC2 HDF5 and phono3py crystal YAML, including primitive/supercell mappings and masses |
| `model` | `type: saved_site_energy`, coefficient NPY and model-metadata JSON |
| `geometry` | Integer transport-supercell matrix, harmonic device layers and zero-based cubic center-layer indices |
| `grid` | Transverse `mesh: [n1,n2,1]`; angular-frequency `step` and `n_positive` |
| `scale` | Physical reference angular frequency `omega0_rad_s` and `hbar_j_s` |
| `reservoirs` | Kelvin temperatures, positive lead broadening `eta_lead`, and `eta_device: 0` |
| `scba` | Cubic coupling `g`, method, original tolerances and bounded solver resources |
| `resources` | Separate input/model/vertex/harmonic/contact preparation bounds |

`SignedFrequencyGrid` uses `step * [-n_positive,...,0,...,n_positive]`; frequencies are dimensionless angular frequencies, and physical angular frequency is this value times `omega0_rad_s`. The silicon scale is `2π × 10¹³ rad/s`. Lead broadening uses the same solver frequency units. Temperatures are converted using `k_B T / (hbar omega0)`.

## Cubic model

`model.json` uses `negfpp.saved-site-energy-model.v1` and contains:

- `model_parameters`: whole coefficient-file hash/size, dtype/shape, coefficient and energy identities, and the fit-basis configuration;
- `fit_report`: the original measured fit residual and converged fit record;
- `fit_geometry`: complete periodic star geometry;
- `material`: FC2/crystal/transport identities and explicit cubic/mass units.

The existing restoration and geometry builders verify these identities. Coefficient files are loaded with `allow_pickle=False`. Coefficients, FC2 and mappings must describe the same model. The command never refits, repairs, symmetrizes or crops endpoint derivatives. A complete energy star that leaves the harmonic device is rejected. Use enough harmonic buffer layers to contain every required endpoint.

`g` scales the cubic vertex: the Born self-energy scales as `g²`. The silicon example's `g=0.10` is a weak-coupling check, not the unscaled `g=1` calculation.

## Solver controls

`method: anderson1` uses two fixed-mixing maps from zero, then the existing bounded depth-one Anderson solver. Strict component/merit selection and PSD proposal backtracking remain in force. `method: linear` uses fixed mixing throughout.

`method: reference-seeded` selects the existing exact reference/increment SCBA controller. The first map is fresh zero. Later maps reuse a selected evaluated reference from this same calculation; this option does not import a saved external state. It retains exact resolvent/product increments and every quadratic Born term. The existing two warm-up maps, strict Anderson/analytic selection and PSD proposal backtracking remain in place. Nonlinear attempts are capped at 18, including rejected and failed trials. Analytic tangents have a separate cap of six and do not count as converged physical states.

The five-layer example supplies the optional controls:

```json
"reference": {
  "warmup_evaluations": 2,
  "tangent_interval": 3,
  "max_tangent_directions": 6,
  "initial_tangent_estimate_seconds": 10.0
}
```

Warm-up count and tangent interval are fixed at two and three. The tangent cap may be reduced to zero through six. If the object is omitted, the existing controller defaults apply, including a tangent-time estimate twice the map estimate. Set `scba.max_evaluations` to at most 18 when selecting this method. Physical/convergence tolerances are unchanged. An increment map accounts for two Born-equivalent contractions and two device solves per frequency/q point; a tangent uses two Born equivalents and one device solve. Tangent admission shares the original run deadline. Original `anderson1`/`linear` inputs and their map limits remain compatible. Rejected and failed map attempts consume `max_evaluations`; no automatic retries or continuation are performed.

Convergence uses `sigma_atol + sigma_rtol * scale` for the original unmixed retarded/lesser/greater self-energy residuals, and `current_atol_w + current_rtol * current_scale` for contact balance, collision power and full balance. All six ratios must be at most one. The example retains the original thresholds.

`workers` controls CPU contraction workers on the local machine; BLAS threads are limited to one within the workflow. MPI or multi-node scheduling is not part of this command.

`max_seconds` includes input construction and controls admission of the next map. `time_reserve_seconds`, `initial_map_estimate_seconds` and `next_map_safety_factor` determine that admission. An in-progress map is not interrupted. Byte/work limits are allocation/work estimates; Python objects, caller arrays and library scratch are outside some estimates. Choose an external scheduler wall limit for production work.

Update the explicit SHA-256 entries when deliberately replacing an input file; `sha256sum FILE` prints its whole-file hash. `negfpp --check-input INPUT.json` checks JSON/native solver settings and file hashes without constructing contacts or evaluating SCBA. Geometry and physical consistency are checked during preparation.
