# Saved qualified complete site energies

Version 1.2.0 adds a second finite-device model input to the same installed
`negfpp INPUT.json` interface. It reads saved independent complete site-energy
derivatives, including NEP-derived datasets. It does not call a potential,
calculate forces, refit a model or substitute a bulk FC3 tensor for the complete
site-energy Hamiltonian.

The conditional input sections are:

```json
"harmonic": {"ifc2": "inputs/ifc2.hdf5"},
"model": {
  "type": "saved_complete_energy",
  "archive": "inputs/site-ifc3.hdf5",
  "metadata": "inputs/model.json"
}
```

`inputs_sha256` must cover exactly these three referenced files. All other
finite-device sections retain their original meaning. The saved fitted
`model.type: "saved_site_energy"` input and its four file roles are unchanged.

The harmonic archive uses `negfpy.material-test-ifc2.v1`, and the cubic archive
uses `negfpy.materialTest.complete-site-ifc3.v1`. They retain their geometry,
image maps and independent endpoint derivatives. A portable
`negfpp.saved-complete-energy-model.v1` JSON file authenticates both archives
and records material/potential identity, physical cell, atomic order, masses,
periodic directions and harmonic principal-layer grouping. It embeds exact
original backend, IFC2, IFC3 and independent IFC3-consistency qualification JSON
texts with their SHA-256 digests and prerequisite bindings. Paths contained in
those original receipts are provenance strings and are never opened by the
release. Qualifying a new potential/dataset remains a separate scientific step.

Preparation checks the original qualifications and complete support, constructs
the full device and paired thermal contacts, and preserves original signs,
units, normalization, passivity, reality, acoustic, optical and final integrated
flux guards. The selected historical contact arithmetic uses extended precision and a decimal
interval certificate for derivative acceptance. This replaces the binary64-only
derivative screen; its original result is reported separately and may fail.
The fixed-eta equations and passivity, reality, acoustic, optical and integrated
flux guards retain their original tolerances. No physical fields are projected.
Geometry, mesh, model inputs, support and qualification identities are exported
in the result report. Both restart and per-map checkpoint options apply.

## Original complete-site second derivatives

An optional `model.source_archive` adds a fourth independently hashed file.
It is used only with an explicitly authenticated lossless harmonic-source
transformation in the model metadata. The release reconstructs image IFC2 from
the original saved complete-site Hessians using the recorded geometry and
permutations, then requires exact array equality with the supplied image HDF5.
It also validates the complete cubic coordinate permutation and retains the
original mass-conversion arithmetic for that source model. No forces or new
finite differences are evaluated. Original qualification receipts remain
labelled with their original file hashes; a derived HDF5 file is never described
as the original qualified file. The original archive participates in input,
restart and checkpoint hashes.
