# Bulk SMRTA 1.0 release operation

`scripts/validate_bulk_rta_1_0.py` is the sole authority that may set
`release_certified=true` for the bulk 1.0 line. Flexible runs require
`--development` and can never certify a release.

The strict campaign is immutable: silicon meshes 3x3x3, 5x5x5, and 11x11x11;
300 K; Gaussian width 0.1 THz; full q-point and band selections; and
phono3py end-to-end comparisons on 3x3x3 and 5x5x5. The 11x11x11 interaction
tensor is not recomputed. Its rates are read from the SHA256-authenticated
0.2.0 archive.

Before creating the output directory, the driver verifies the sealed 0.2.0
manifest, FC2/FC3/crystal hashes, all archive schemas and identities, the
explicit environment lock, package/import paths, exact runtime versions,
Open MPI, free disk space, and output writability. This preflight is intended
to catch infrastructure errors before scientific work begins.

Run from a clean exact checkout using the Python and `mpiexec` from the pinned
environment:

```bash
NEGFPY_RELEASE_PREFIX=/absolute/path/to/phono3py-4.4.0-si-mpi-env
export PATH="${NEGFPY_RELEASE_PREFIX}/bin:/usr/bin:/bin"
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export BLIS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

"${NEGFPY_RELEASE_PREFIX}/bin/python" scripts/validate_bulk_rta_1_0.py \
  --artifact-root /verified/v0.2.0/artifact-root \
  --crystal-yaml /verified/phono3py_disp.yaml \
  --v0-2-release-manifest /verified/NEGFPy_silicon_bulk_oracle_0.2.0_release_manifest.json \
  --environment-lock validation/locks/phono3py-4.4.0-si-pbesol-openmpi-linux-64.lock \
  --environment-lock-sha256 6d8e017b6f1ead8be3ec9fb8abb1c8c1d02fe0de852aad9cbfc69d08ce341bfc \
  --output-dir /artifacts/bulk-rta-1.0.0/<exact-commit>/<run-id>
```

The release imports the installed NEGFPy wheel from this prefix; repository
`src/` injection is prohibited. Every runtime package import must be owned by
the conda inventory, while NEGFPy files must be listed by the wheel RECORD.
Build the candidate wheel outside this prefix, then install that wheel with
`pip install --no-deps`; build helpers or other unowned Python files in the
release prefix intentionally fail preflight.
The pinned explicit lock itself is authenticated by SHA256. Its current URLs
do not carry artifact SHA256 fragments, so package records use their recorded
SHA256 where available and an explicitly labelled MD5 identity otherwise;
stable installed payloads are still verified with `sha256_in_prefix`. A future
lock repin should add SHA256 URL fragments for every artifact.

Each mesh has separate atomic core and phono3py checkpoints. After an
infrastructure interruption, rerun the identical command with `--resume`.
The immutable campaign manifest and checkpoint hashes reject changed inputs
or controls. Before any resumed scientific task starts, the driver audits the
complete partial artifact tree and every existing mesh checkpoint, so a bad
later-mesh checkpoint cannot waste earlier work. A scientific failure is
retained and is not automatically retuned.

Successful output contains:

- `bulk-rta-validation.json`;
- `release-evidence.json`;
- `bulk-rta-results.hdf5`, including bands, DOS, mesh phonons, group
  velocities, rates, lifetimes, mean free paths, and RTA conductivity;
- per-mesh checkpoints; and
- `SHA256SUMS.json`; and
- `release-completion-receipt.json`.

The report JSON and evidence sidecar are deliberately provisional and always
store `release_certified=false`. The directory-aware checksum manifest seals
their exact tree together with the HDF5 result. Only after rederiving all
scientific, runtime, campaign, HDF5, and checksum gates does the driver
atomically write `release-completion-receipt.json`; that commit-last receipt is
the sole file permitted to contain `release_certified=true`. Nothing in the
sealed output is modified afterward. A valid receipt resume performs no
scientific work. Exit status zero in strict mode means the authoritative
receipt certifies the release; any other decision is nonzero. The release
report must additionally preserve the explicit statement
`physical_mesh_convergence_certified=false`.
