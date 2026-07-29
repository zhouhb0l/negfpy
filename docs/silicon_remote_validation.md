# Silicon `0.2.0` remote validation operations

This is the operational contract for the release-scale silicon bulk oracle.
It complements the scientific contract in
[`theory/bulk_three_phonon_oracle.md`](theory/bulk_three_phonon_oracle.md)
and does not reuse the toy-only `remote_heavy_cli.py` deployment.

## Fixed security and execution boundary

- Connect only through the `negf-heavy` SSH alias and require
  `BatchMode=yes`. The pinned ED25519 host fingerprint and key-only options
  are defined in [`remote_heavy_validation.md`](remote_heavy_validation.md).
- The remote account is `hangbo` and the validation root is
  `/home/hangbo/negfpy-validation/<commit>/`.
- Use the explicit material lock
  `validation/locks/phono3py-4.4.0-si-pbesol-openmpi-linux-64.lock`, SHA256
  `6d8e017b6f1ead8be3ec9fb8abb1c8c1d02fe0de852aad9cbfc69d08ce341bfc`.
- Use only the corresponding environment's `python3.12` and `mpiexec`.
  Install the exact candidate wheel with `--no-deps --target` into the
  candidate's private `candidate-site/`; never mutate the shared locked
  environment.
- Set `PYTHONNOUSERSITE=1` and set `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`,
  `BLIS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to `1`.
- Run MPI with one rank per physical core. Rank zero is a coordinator:
  `mpiexec --nooversubscribe --bind-to core
  --map-by ppr:38:socket:PE=1 --report-bindings -n 76`.
- Heavy commands run in a detached, uniquely named `tmux` session. Hosted CI
  never connects to this workstation.

## Seal and deploy one exact candidate

The local repository must have an empty `git status --porcelain=v1
--untracked-files=all`. Record the full commit, Git tree, material-lock hash,
wheel hash, bundle hash, and every transfer file in a canonical
`SHA256SUMS`. Build the wheel directly from that clean checkout so its
embedded package provenance identifies the same commit and declared
`0.2.0` version.

Create a Git bundle containing the exact commit and verify it locally. Transfer
the bundle, wheel, explicit lock, input manifest, and transfer checksum file
to a new remote `incoming/` directory. Remote setup must:

1. verify every transferred SHA256 before use;
2. restore a detached `worktree/` from the bundle;
3. verify the exact commit, tree, clean status, and bundle;
4. install the wheel into an initially absent `candidate-site/`;
5. import NEGFPy only from that site and require clean embedded code
   provenance, version `0.2.0`, and the exact commit; and
6. archive machine, environment, MPI vendor, CPU binding, RAM, disk, source,
   wheel, bundle, and imported-code identities under `artifacts/identity/`.

The pinned phono3py checkout is read-only input:

```text
/home/hangbo/negfpy-validation/datasets/
  phono3py-4.4.0-Si-PBEsol-302ad205/upstream-repository
```

Before every campaign, require its clean commit to be
`302ad205452ee36e57887b10bbf4a0289dd037bb` and re-hash the consumed
`phono3py_disp.yaml` and `FORCES_FC3`.

## Immutable campaign graph

All paths below are children of
`/home/hangbo/negfpy-validation/<commit>/artifacts/`. Every output directory
is new. A scientifically failed task is archived; controls are not changed
and the task is not silently dropped.

1. Run `generate_phono3py_si_reference.py --force-constants-only` once to
   create the authenticated full and compact FC2/FC3 stage.
2. For each of the \(3^3\), \(5^3\), and \(11^3\) meshes, run
   `generate_phono3py_si_reference_mpi.py --all-grid-points`. One worker owns
   one complete target \(q\); rank zero performs a canonical filesystem merge
   with no floating-point MPI reduction.
3. For each mesh, run `generate_v3_sample_manifest_mpi.py` to create the
   bounded-v2 scientific sample JSON and its separately checksummed MPI
   orchestration/binding evidence.
4. At \(3^3\) and \(5^3\), run `validate_phono3py_si_oracle.py` to create the
   strict serial qualification, then run
   `run_phono3py_si_target_q_mpi.py --mode mpi --compare-monolithic`.
   These executions must be fresh, full-\(q\), full-band and non-resumed.
5. At \(11^3\), run `run_phono3py_si_target_q_mpi.py --mode mpi` without a
   monolithic recomputation. Exact checkpoint resume is allowed only for an
   infrastructure interruption with the unchanged hashed task.
6. Run `validate_phono3py_si_mpi_oracle.py` on the complete \(11^3\) merge.
   It rechecks the independent reference, sampled
   \(\lvert V^{(3)}\rvert^2\), phase space, linewidths, task artifacts,
   generation records, code identity, and input hashes.
7. Run `validate_phono3py_si_release.py` with the \(11^3\) external report
   and the fresh \(3^3\)/\(5^3\) equality summaries. This is the only command
   permitted to emit `release_certified=true`. It independently re-merges
   every target-\(q\) task, rejects external/VDS/non-hard HDF5 storage,
   reconstructs the authenticated \(3^3\) and \(5^3\) problems, reruns the
   serial linewidth kernel, and repeats both full-field comparisons.
8. Stage external file records, seal the complete artifact tree, and verify
   the portable manifest as described in
   [`validation/silicon-portable-evidence.md`](validation/silicon-portable-evidence.md).
   Retrieval verification must supply both trusted digests from the
   authenticated remote checksum record; a self-hash is integrity-only.

Before launch, archive a checksummed, immutable command manifest and launch
script alongside the candidate transfer. That outer runner writes separate
stdout, stderr, argv, start/end timestamps and return-code records for every
command. The final publication barrier is a checksum manifest covering every
regular file under the artifact tree except the manifest itself. A failed
checksum finalization is a failed campaign. The launch script is itself
release evidence; changing it creates a new run identity.

## Retrieval and independent verification

After the remote session is terminal, copy the complete regular-file artifact
projection to:

```text
artifacts/validation/si-bulk-oracle/<commit>/<run-id>/
```

Verify the remote and local checksum manifests independently. Re-open the
strict JSON, all HDF5 checkpoints/results, both reference and V3 MPI
orchestration records, the three release inputs, and the final aggregate
report. Run portable verification with the trusted remote manifest and
release-report SHA256 values. The release can advance only if the aggregate
report is immutable, its exact bytes are checksum-covered, the portable
verification scope is `trusted_digest_authenticated_integrity`, and the
report declares `release_certified=true`.

Tagging and fast-forwarding `main` happen only after this retrieval audit.
There is no external push or publication without separate authorization, and
any tracked source change requires the entire exact-commit gate again.
