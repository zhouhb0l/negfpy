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
creates `~/negfpy-validation/<commit>/environment` from that lock. Every
scientific manifest receives that exact lock SHA directly; no shared scientific
environment or environment-variable override is accepted. Identity records must
agree on the full commit, Git tree, package-source SHA256, lock SHA256,
interpreter, installed distributions, platform, and thread limits.

Wheel installs additionally follow the fail-closed build identity contract in
[`development/build_provenance.md`](development/build_provenance.md); embedded
metadata supplements but never replaces the deployment bundle and worktree
receipts. Remote identity comparison treats this SHA256 as an opaque value from
`code_provenance()`; it does not implement a second hash algorithm.

The explicit Linux lock pins Python 3.10.19, Open MPI 5.0.10, mpi4py 4.1.1,
NumPy 2.2.6, SciPy 1.15.2, h5py 3.16.0, threadpoolctl 3.6.0, and pytest 9.0.3.
Every MPI launch must use the deployed environment's own
`environment/bin/mpiexec`; a system launcher or launcher from another conda
environment is forbidden. Deployment records the mpi4py vendor/library
identity, launcher version, resolved launcher target, and launcher SHA256, and
attempt-time provenance must match it exactly.

`launch` creates `negf-toy-<12-character-commit>` in detached `tmux`. It fixes
OpenMP, OpenBLAS, MKL, BLIS, NumExpr, and vecLib thread counts to one and runs
the hardened task-level MPI lifecycle. The worker first runs five fresh,
isolated 1/2/4/8/16-rank representative campaigns, independently rebuilds the
immutable scaling decision, then launches the locked release through that
deployment's absolute `environment/bin/mpiexec` path and binding policy. There
is no pytest wrapper and no scientific command-line override.

Every release subprocess writes separate `mpi-release.stdout.log` and
`mpi-release.stderr.log` files and then atomically publishes the immutable
hashed envelope `mpi-release-launch.json` with schema
`negfpy.remote_validation.release_launch.v1`. The receipt records the exact
environment-local launcher and Python paths, their resolved paths and file
SHA256 values, the complete argv and Open MPI binding arguments, selected rank
count, one-thread environment, source/tree/environment identities, execution
identifier, timestamps, return code, and the relative log paths and hashes.
The worker independently reconstructs the launcher argv and binding policy and
remeasures both executables and logs before it accepts a successful run. The
validated receipt facts are embedded in `mpi_execution_facts`; the attempt
publication names and hashes the receipt and both logs, and `SHA256SUMS` covers
all three. A nonzero launch still retains its receipt and logs as infrastructure
evidence eligible only for an identical hashed retry.

The release worker does not accept process exit status alone as success. It
strictly reloads the scaling, campaign, MPI execution, and scientific reports
and requires:

- combined schema v4 with a valid canonical body digest and
  `release_certified=true`;
- embedded multidimensional protocol v2/schema v4 evidence;
- the exact 13-outcome FFT/FFT SCBA graph with analytic diagonal contacts;
- the passing seven-point harmonic preflight (base plus six preregistered
  numerical points);
- the passing bounded converged serial/threaded/FFT full-field oracle;
- the complete 13-run open-system matrix;
- identical canonical physics digests across every scaling rank;
- a policy-selected toy rank (or the one-rank serial/FFT fallback when speedup
  is below 2x), with the silicon MPI selection retained separately;
- a complete 28-task campaign/execution manifest whose SHA chain is attached
  to the strict report;
- 13 valid scientific checkpoint files and one environment-lock identity; and
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

The release campaign lives at `artifacts/toy-release/` and is resume-enabled.
Its task outcomes and `scientific-checkpoints/` are reused only after source,
environment, grid, input, initial-state, campaign, and task hashes match
exactly. Each attempt uses a new execution identifier; no resume environment
variable or altered control is accepted.

Each execution writes a new `artifacts/attempts/attempt-NNN/` directory with
its own report, log, status, execution identity, scaling reference, report
facts, MPI runner result, MPI execution manifest, and relative checkpoint
index. Compatible HDF5 checkpoints remain in
`artifacts/toy-release/scientific-checkpoints/`.

The first attempt runs a fresh scaling ladder. An exact infrastructure retry
may reuse it only after reopening every rank/worker artifact and rebuilding the
same source-, tree-, environment-, topology-, and physics-locked decision.
A root `current-attempt.json` points to the sole current terminal attempt. A
runner failure sidecar is moved into that attempt;
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

## Task-level MPI scaling and certification runner

The task scheduler stays above the SCBA solver. A worker rank receives only a
case identifier, executes that complete case with the unchanged FFT/SCBA
implementation, and publishes its own atomic HDF5 outcome. Green functions and
self-energies are never sent through MPI or combined by a floating-point MPI
reduction.

`remote_heavy_cli.py launch` performs the ladder and release automatically.
The following lower-level command is for controlled diagnostics only; it must
use the exact deployed environment and source identities:

```bash
environment/bin/python scripts/remote_toy_mpi_runner.py scale \
  --environment-prefix "$PWD/environment" \
  --run-directory "$PWD/artifacts/mpi-scaling/scale-001" \
  --source-revision "$commit" \
  --source-tree-sha256 "$source_tree_sha256" \
  --environment-sha256 "$environment_lock_sha256" \
  --physical-core-count 76 \
  --socket-count 2
```

`scale` itself launches fresh, isolated repetitions with 1, 2, 4, 8, and 16
ranks through `environment/bin/mpiexec`. Each rank is bound to one physical
core, ranks are distributed across both sockets, and all BLAS/OpenMP thread
limits are one. No repetition resumes or shares checkpoints with another rank
count. The outer process measures the complete launcher wall time.

The representative campaign is explicitly noncertifying, but it retains the
full physical and orchestration topology: two preflights, a hardest direct-zero
endpoint, the five-point coupling/bias continuation, one Born-start and six independently varied
nine-DOF outcomes, and the 13 signed-coupling/temperature open-system cases.
The multidimensional controls are:

```text
base:       dw=0.0125, cutoff=5, eta_lead=0.025, mixing=0.2
spacing:    0.025, 0.0125, 0.00625
cutoff:     5, 6
eta_lead:   0.05, 0.025, 0.0125
mixing:     0.2, 0.1
solver:     FFT convolution/Hilbert, complex128, max_iterations=200
thresholds: production fixed-point and conservation tolerances unchanged
```

The open-system base grid is `dw=0.00125`, cutoff 5, and
`eta_lead=0.0008`. It uses the existing ten-case Cartesian matrix of
`T=(0,0.7)` and signed `g=(-0.02,-0.01,0,0.01,0.02)`, plus independent
half-spacing, cutoff-6, and half-eta cases.

Every repetition writes an immutable hashed worker result and an outer
`rank-NNN-manifest.json`. `scaling-selection.json` is published only after all
28 outcomes complete for every rank count and every repetition has the same
canonical `production_toy_physics_sha256`. The selected count is the smallest
within 5% of the fastest wall time. Toy MPI is enabled only for at least a
2x end-to-end speedup; otherwise toy certification remains serial/FFT while
the selected task-MPI count is retained for the silicon target-q workload.

The lower-level certification invocation exposes no grid, eta, mixing,
backend, or solver-tolerance options. The hardened lifecycle constructs it
from the validated selection with the same environment-local MPI executable.
For an eight-rank diagnostic example on the confirmed two-socket workstation:

```bash
environment/bin/mpiexec \
  --nooversubscribe --bind-to core \
  --map-by ppr:4:socket:PE=1 --report-bindings -n 8 \
  environment/bin/python scripts/remote_toy_mpi_runner.py release \
  --run-directory "$PWD/artifacts/toy-release/gate-001" \
  --source-revision "$commit" \
  --source-tree-sha256 "$source_tree_sha256" \
  --environment-sha256 "$environment_lock_sha256" \
  --execution-id gate-001 \
  --expected-ranks 8
```

The release directory must be fresh or contain an exact campaign manifest for
checkpoint resume. A nonempty unrelated directory is rejected. A new
execution identifier is required for each exact-resume attempt, while the
scientific task identities and controls remain fixed. Rank zero writes
`runner-results/<execution-id>.json`, references the strict production report
and MPI execution manifest by relative path and SHA256, records the canonical
physics digest, and returns success only when the locked report has
`release_certified=true`.
