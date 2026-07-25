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
PasswordAuthentication=no
KbdInteractiveAuthentication=no
```

Configure the alias outside the repository:

```sshconfig
Host negf-heavy
    HostName 172.20.86.67
    UserKnownHostsFile ~/.ssh/known_hosts_negf_heavy
    User hangbo
    IdentityFile ~/.ssh/id_ed25519_aspire2a
    IdentitiesOnly yes
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
`b2ce515e488757f61781a559985d401575aec9dd73507c0c3b71b68746022839`, and
creates `~/negfpy-validation/<commit>/environment` from that lock. The worker
passes the same hash as `NEGFPY_VALIDATION_ENV_SHA256`; no shared scientific

`launch` creates `negf-toy-<12-character-commit>` in detached `tmux`. It fixes
OpenMP, OpenBLAS, MKL, BLIS, NumExpr, and vecLib thread counts to one and runs
the established nightly pytest release entry point. The report, log,
environment manifest, job status, and HDF5 checkpoints live below
`~/negfpy-validation/<commit>/artifacts/`.

## Resume, status, and retrieval

The worker always supplies the checkpoint directory and resume flag to the
release runner. A later launch of the same committed deployment therefore
resumes only compatible checkpoint state; scientific checkpoint validation is
responsible for rejecting any hash mismatch.

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
