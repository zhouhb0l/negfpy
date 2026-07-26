# Remote heavy-validation operations

Release-scale toy and material calculations run on the designated workstation.
Hosted GitHub workflows run bounded tests only and must never connect to that
workstation.

## Security contract

The orchestration accepts only the SSH alias `negf-heavy`. It has no hostname,
username, identity-file, or credential options. Every `ssh` and `scp`
invocation forces:

```text
BatchMode=yes
StrictHostKeyChecking=yes
IdentitiesOnly=yes
IdentityFile=~/.ssh/id_ed25519_aspire2a
UserKnownHostsFile=~/.ssh/known_hosts_negf_heavy
GlobalKnownHostsFile=/dev/null
HostKeyAlgorithms=ssh-ed25519
PreferredAuthentications=publickey
PasswordAuthentication=no
KbdInteractiveAuthentication=no
GSSAPIAuthentication=no
HostbasedAuthentication=no
```

The command line also fixes `HostName=172.20.86.67` and `User=hangbo`; a
change to the alias between preflight and launch therefore cannot redirect a
job. The effective alias must select only the dedicated known-hosts file. That
file must be owned by the current user, must not be group/other writable, and
must contain exactly one entry for the workstation using the pinned ED25519
key. System-wide and ordinary user known-hosts files are not consulted.

Configure the alias outside the repository:

```sshconfig
Host negf-heavy
    HostName 172.20.86.67
    UserKnownHostsFile ~/.ssh/known_hosts_negf_heavy
    User hangbo
    IdentityFile ~/.ssh/id_ed25519_aspire2a
    IdentitiesOnly yes
    BatchMode yes
    PreferredAuthentications publickey
    PasswordAuthentication no
    KbdInteractiveAuthentication no
    StrictHostKeyChecking yes
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

Before adding the ED25519 host key to `known_hosts`, verify its fingerprint
through a trusted channel. The pinned value is:

```text
SHA256:sJide2AQhTe8oH2XWtayraDdUsde/LV2HOfUeMeWd70
```

Interactive authentication may be used once by the operator to authorize the
public key. No credential may be placed in a command, environment variable,
script, manifest, log, shell history, or repository file. All automated
operations deliberately fail if key-only login is unavailable.

## Deployment and run

Run commands from the repository root. First check the effective alias,
key-only reconnect, machine resources, and the fixed remote environments:

```bash
python scripts/remote_heavy_cli.py preflight \
  --output validation-artifacts/remote-preflight.json
```

The workstation uses:

- `/home/hangbo/.local/bin/micromamba` for environment creation;
- `/home/hangbo/micromamba/envs/negf-tools/bin` for Git, `tmux`, and `rsync`.

`prepare` refuses a dirty worktree. Commit the isolated release candidate,
then bundle the committed explicit lock at
`validation/locks/negf-toy-gate-linux-64.lock`, source/environment manifests,
and `SHA256SUMS`:

```bash
python scripts/remote_heavy_cli.py prepare
```

The command prints a staging directory ending in the full commit. Assign that
commit once for the remaining examples:

```bash
commit=0123456789abcdef0123456789abcdef01234567
python scripts/remote_heavy_cli.py deploy \
  "validation-artifacts/deployments/$commit"
python scripts/remote_heavy_cli.py launch "$commit"
```

Deployment verifies checksums remotely, checks out exactly the detached clean
commit in `~/negfpy-validation/<commit>/worktree`, verifies explicit-lock SHA
`aca1a3f7eb7675adb5982d571005cc5ae65ed442b4972ad835f2d4fdf6fa33c1`, and
creates `~/negfpy-validation/<commit>/environment` from that lock. The worker
passes the same hash as `NEGFPY_VALIDATION_ENV_SHA256`; no shared scientific
environment is accepted. Deploy-time and attempt-time identity records must
agree on the full commit, Git tree, package-source SHA256, lock SHA256,
interpreter, installed distributions, platform, and thread limits.

The explicit Linux lock pins Python 3.10.19, Open MPI 5.0.10, mpi4py 4.1.1,
NumPy 2.2.6, SciPy 1.15.2, h5py 3.16.0, threadpoolctl 3.6.0, and pytest 9.0.3.
Every MPI launch must use the deployed environment's own
`environment/bin/mpiexec`; a system launcher or launcher from another conda
environment is forbidden. Deployment records the mpi4py vendor/library
identity, launcher version, resolved launcher target, and launcher SHA256, and
attempt-time provenance must match it exactly.

`launch` creates `negf-toy-<12-character-commit>` in detached `tmux`. It fixes
OpenMP, OpenBLAS, MKL, BLIS, NumExpr, and vecLib thread counts to one and runs
the established nightly pytest release entry point.

The release worker does not accept pytest exit status alone as success. It
strictly reloads the current-attempt report and requires:

- combined schema v4 with a valid canonical body digest and
  `release_certified=true`;
- embedded multidimensional protocol v2/schema v4 evidence;
- the exact 13-outcome FFT/FFT SCBA graph with analytic diagonal contacts;
- the passing seven-point harmonic preflight (base plus six preregistered
  numerical points);
- the passing bounded converged serial/threaded/FFT full-field oracle;
- the complete 13-run open-system matrix;
- 13 valid checkpoint files and one environment-lock identity; and
- full commit and package-source identity equal to the deployed candidate.

The protocol-v2 numerical ladder is
`dw=(0.0015625, 0.00078125, 0.000390625)` and
`eta=(0.003125, 0.0015625, 0.00078125)`. The production graph is FFT-only;
serial/threaded/FFT equality is established on the bounded oracle, not by an
additional full-grid threaded outcome. The previous d05b result is provisional
failed evidence, and `0.1.0` remains unvalidated until a clean workstation
rerun passes.

For interpretation of the archived conservation diagnostics, the locked
block-local threshold is `SIGMA_PP_BLOCK_LOCAL_RTOL=1e-12`. Raw harmonic bond
currents gate only block-local self-energies. Full nonlocal SCBA uses the
`nonlocal_self_energy_eom_cut_correction`; it is not labeled as a bare FPU
interaction-current operator. Nonequilibrium `Sigma_pp` broadening may be
indefinite, so no unsupported NESS positive-semidefiniteness claim is added to
remote certification.

## Resume, status, and retrieval

The worker always supplies the persistent checkpoint directory. The release
runner's existing default is resume-enabled, and scientific checkpoint
validation rejects any source, environment, grid, input, or initial-state
hash mismatch. The obsolete, unconsumed resume environment variable is not
exported.

Each execution writes a new `artifacts/attempts/attempt-NNN/` directory with
its own report, log, status, execution identity, report facts, and relative
checkpoint index. Compatible HDF5 checkpoints remain in
`artifacts/checkpoints/`. A root `current-attempt.json` points to the sole
current terminal attempt. A runner failure sidecar is moved into that attempt;
it cannot survive at the artifact root and masquerade as current evidence.

Before a retry, the old `SHA256SUMS` and current-attempt pointer are archived,
so a stale successful report cannot satisfy a later failed attempt. The final
checksum manifest is the publication barrier and the job lock is removed only
after that manifest is durably written. A checksum/finalization error leaves
the lock in place and records `finalization_failed`; it is never labeled
completed. `status` reports a lock with no live tmux session as `stale_lock`.

`artifacts/identity/` contains verified copies of the deployment manifest,
environment manifest, explicit lock, incoming checksum manifest, remote
environment record, and a lock/source identity manifest. The scientific JSON
is preserved byte-for-byte; because it records absolute runtime checkpoint
paths, the attempt's `checkpoint-index.json` supplies checksum-verified paths
relative to the fetched artifact root.

Inspect state without changing the job:

```bash
python scripts/remote_heavy_cli.py status "$commit"
```

After completion or failure, retrieve the complete artifact tree and verify
every advertised SHA256:

```bash
python scripts/remote_heavy_cli.py fetch "$commit"
```

Fetched data are written to `validation-artifacts/remote/<commit>/`, followed
by a local `fetch-receipt.json`. Existing nonempty destinations are rejected.
Failed and partial calculations are retained rather than silently dropped.

Every public operation supports `--dry-run`. It prints the exact commands and
does not connect, create deployment directories, launch jobs, or fetch data:

```bash
python scripts/remote_heavy_cli.py launch "$commit" --dry-run
python scripts/remote_heavy_cli.py status "$commit" --dry-run
python scripts/remote_heavy_cli.py fetch "$commit" --dry-run
```

The full toy gate remains a workstation release operation. GitHub pull-request
and scheduled workflows retain only bounded tests and smoke calculations.
