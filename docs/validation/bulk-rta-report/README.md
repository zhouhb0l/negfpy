# NEGFPy 1.0 bulk-phonon/SMRTA validation report

This directory contains the report source and a deterministic renderer for
the NEGFPy 1.0 bulk-observable release track. The renderer is deliberately a
post-processing tool: it reads immutable JSON/HDF5 results and never calls the
phonon or transport solvers.

The checked-in PDF, when present, is provisional unless its filename omits
`DRAFT_UNCERTIFIED` and the normalized report state says
`"report_certified": true` and `"authoritative_release_certified": true`.
Only `release-completion-receipt.json` can supply that authoritative decision.

## Render a development artifact

From the workspace root:

```bash
/home/zhouhb/micromamba/envs/phonon-negf/bin/python \
  reports/bulk-rta-1.0.0/scripts/generate_report.py \
  --validation-json tmp/bulk-rta-dev-full-20260802/bulk-rta-validation.json \
  --results-hdf5 tmp/bulk-rta-dev-full-20260802/bulk-rta-results.hdf5 \
  --compile \
  --render-pages
```

This command produces
`output/pdf/NEGFPy_bulk_RTA_1.0.0_DRAFT_UNCERTIFIED.pdf`. Rendered pages are
placed in `tmp/pdfs/` for visual review. Legacy development inputs can only
produce draft wording. The script verifies their adjacent checksums, required
mesh shapes, finite arrays, and the sum of every per-mode conductivity tensor
before it writes report output.

## Final release rendering

The frozen validator provides a provisional JSON/HDF5 pair, a provisional
acyclic evidence sidecar, an exact-tree checksum manifest, and a commit-last
completion receipt for the `3x3x3`, `5x5x5`, and `11x11x11` implementation
matrix.
The locked HDF5 schema is `negfpy.bulk_rta_release_results.hdf5.v1`, with
meshes under `meshes/mesh-NxNxN`. Every mesh carries its complete `rta/`
result and `harmonic/{band,dos,mesh}` arrays. The direct metrics in JSON
`mesh_evidence` are byte-for-byte repeated as canonical `metrics_json` HDF5
attributes.

The provisional sidecar `release-evidence.json` uses schema
`negfpy.bulk_rta_release_evidence.v1`. Its SHA256 is embedded in both the
report and HDF5; the report also records the whole HDF5 SHA256. This avoids a
circular file-hash dependency while binding the campaign, inputs, numerical
metrics, candidate decision, and arrays. Both the JSON report and evidence
sidecar must retain `release_certified=false`.

`release-completion-receipt.json` uses schema
`negfpy.bulk_rta_release_completion_receipt.v1` and is the sole certification
authority. It must have exact fields and boolean types; reproduce the
candidate/scientific decisions; preserve
`provisional_release_certified=false`; and bind the whole report, evidence,
HDF5, and `SHA256SUMS.json` hashes. The manifest itself must match the complete
artifact tree exactly. Only a receipt with both
`candidate_release_gates_passed=true` and `release_certified=true` permits
`--release-mode` wording.

The provisional report must also record three passing mesh decisions, clean
exact provenance, and `physical_mesh_convergence_certified=false`. If the
receipt is absent, negative, malformed, or hash-mismatched, or if any reporting
field is absent, release mode fails rather than inferring missing evidence.

For certified wording, `runtime_before` and `runtime_after` must be identical,
clean, and bound to the exact imported NEGFPy 1.0.0 payload and pinned
phono3py 4.4.0 environment. The renderer reads those authenticated runtime
records and input hashes directly; it does not require a second set of
duplicated provenance booleans.

The reproducible source-only mirror in the candidate repository is
`docs/validation/bulk-rta-report/`. It intentionally contains only this README,
`report.tex`, the generator, and its tests; figures, generated macros, PDFs,
caches, monitoring state, and scientific artifacts are not mirrored.

## Directory contract

- `report.tex`: maintained scientific narrative and layout.
- `scripts/generate_report.py`: strict evidence loader, figure generator, and
  PDF compiler.
- `generated/`: evidence macros, tables, and normalized report state.
- `figures/`: deterministic PDF and PNG figures.
- `output/pdf/`: compiled report, build log, and output checksums.
- `tmp/pdfs/`: rendered pages for visual inspection; not scientific evidence.

The `3^3`, `5^3`, and `11^3` conductivities are implementation-validation
points at fixed settings. They are intentionally labeled
`physical_mesh_convergence_certified=false` throughout the report.
