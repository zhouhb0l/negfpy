# Material Nq=1 next-candidate change record

Status: development draft; not a launch authorization or release certificate.

## Evidence boundary

Aspire2A run `material-nq1-release-011` was launched from exact source commit
`ad4430c811bb3d1b8339b3bbf8d46c3eccd7af1d`.  That worktree remains unchanged
for terminal inspection, strict validation, or an exact checkpoint recovery.
The changes described here are post-evidence work.  They produce different
source-tree, task, and campaign hashes and therefore cannot certify, repair,
or reinterpret run 011.

## Candidate-control corrections

Run 011 already executed numerically with `mixing_floor=0.001`; its exact
`ad4430c` driver supplied that value outside the v1 scientific task payload.
The next candidate does not change that run-011 number.  It makes the adaptive
SCBA mixing floor an explicit member of every `MaterialSCBATask` and serializes
it into the task, MPI payload, qualification control record, receipt
authentication, and campaign hash.  Execution, Born initialization,
checkpoint resume, and strict validation must obtain the value from that
serialized task rather than from an unrecorded default.

This value was selected after the earlier `7b0c169` scientific failure.  It is
therefore a new control contract, not evidence under the earlier frozen
`mixing_floor=0.01` design.  Before a new scientific launch, the complete
candidate controls and acceptance gates must be frozen in a new qualification
record and given a new artifact root and execution identifier.

The watchdog contracts are also separated:

- the bounded one-shot Born gate retains its 2-hour, 4-GiB limits;
- material SCBA uses explicit 12-hour, 8-GiB limits.

The SCBA wrapper passes both limits directly to the supervisor.  It no longer
mutates limits in the imported Born module, preventing import order from
changing either campaign identity.

## Required disposition before launch

1. Retrieve and classify run 011 as running, infrastructure-interrupted,
   scientifically failed, execution-complete but uncertified, or certified.
2. Preserve all run-011 artifacts and validate them only with exact `ad4430c`
   source.
3. Review and freeze this next-candidate control change before qualification.
4. Run the full non-nightly suite, MPI qualification, material input
   preflight, and exact-source qualification on the new commit.
5. Use a new immutable campaign root and execution identifier; never overwrite
   or relabel run 011.
