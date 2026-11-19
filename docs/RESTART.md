# Saved self-energy inputs

Version 1.1.0 adds optional `initial_self_energy` to `negfpp.run.v1` inputs.
It accepts the evaluated self-energy from a previous converged `negfpp` result
with `negfpp.fields.v1` fields and a `negfpp.result.v1` report, including results
written by versions 1.0.0–1.0.2. This option supports all three SCBA methods.

Add this top-level object to an otherwise ordinary input:

```json
"initial_self_energy": {
  "fields": "../previous-result/fields.h5",
  "fields_sha256": "<SHA-256 of fields.h5>",
  "input": "../previous-result/input.json",
  "input_sha256": "<SHA-256 of input.json>",
  "report": "../previous-result/report.json",
  "report_sha256": "<SHA-256 of report.json>"
}
```

Replace each placeholder with its 64-character lowercase hexadecimal digest.
Existing result folders contain these digests in `SHA256SUMS`; they can also be
computed with `sha256sum`. Paths resolve relative to the new input JSON file.
Choose a new `run.output_dir`, then run `negfpp INPUT.json` normally.
`--check-input` authenticates the referenced files without running SCBA.

New version 1.1.0 reports bind `fields.h5` by its digest and byte count. For a
version 1.0.0–1.0.2 result, also include the original checksum manifest and its
digest in the same `initial_self_energy` object:

```json
"checksums": "../previous-result/SHA256SUMS",
"checksums_sha256": "<SHA-256 of SHA256SUMS>"
```

The manifest must bind the selected fields, input and report together. Loading
three individually hashed files from different results is rejected.

The original and target runs must have identical physical input-file hashes,
device geometry, complete interaction support, ordered active DOFs, q/frequency
grids, coupling, reservoirs, broadening and units. Controls such as output path,
worker count, iteration limit and SCBA method can change within their normal
constraints. This option does not rescale a state or interpolate between grids.

Only `sigma_input_evaluated/retarded`, `statistical` and `spectral` are loaded.
They retain their native `complex128` bytes, including signed zeros, and are
copied into read-only arrays. Saved Green functions, unevaluated Born images,
acceleration vectors and convergence history are not imported. Source files
remain unchanged. Historical native campaign checkpoint formats require their
own qualified importers; an arbitrary HDF5 file is not a `negfpp` result bundle.

The first new evaluated state is numbered 2; its ordinal in the new run is 1.
The source state index is recorded separately in the initialization receipt.
Every new map consumes the new run's budget. The previous convergence result
does not count toward acceptance: a fresh Dyson/Born map and all original three
self-energy and three current screens must pass, with unchanged physical guards.
An initial fixed point can therefore finish after one new map, but its
convergence is established again by that map. Preparation, file authentication
and loading also consume the new run's time budget.

This option applies to SCBA. Method 1 retains its separate `frozen_source`
interface and frozen-kernel/local-FD approximation.
