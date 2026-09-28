# Nq=1 launch-environment candidate

This candidate addresses qualification and five-task pilot provenance. It does
not change the SCBA solver, cubic operator, checkpoint format, scientific
thresholds or iteration budget. It does not certify convergence or a release.
Historical candidate worktrees and campaign artifacts remain unchanged.

## Exact launch policy

`_launch_environment.build_launch_environment` constructs all child environment
variables from explicit repository, executable, home, temporary directory and
Git directory paths. No ambient variables are copied. Python imports use only
the candidate source/scripts, user site packages are disabled, thread limits are
one, and Python hashing, locale and Git behavior are fixed. Open MPI uses each
runtime's own OPAL prefix with pml=ob1, btl=self,sm and oversubscription enabled.
These are single-node qualification/pilot controls, not evidence of general
multi-node transport reliability.

Runtime identity v2 hashes the complete launch map along with the executable
and package inventory. Qualification subprocesses add the resulting runtime
hash as their marker. Check identity/receipt v3 record the complete environment
passed to subprocess.run; any map change prevents reuse. Context v2 is mandatory.
Version and package probes also run with the sanitized runtime map.

The outer base qualification v2 and predictor wrapper v3 remain pilot-only:
the new mandatory inner schemas reject legacy projected-environment receipts.
The existing full release driver is intentionally not enabled for runtime v2;
a separately designed and freshly qualified full campaign remains required.

## Pilot execution

The pilot validates the current environment against the qualified Python 3.12
map before claiming a task. Canonical PBS uses env -i with every authenticated
key/value. Each task publishes an immutable launch-environment record; task
receipt v2 authenticates that record. The SCBA supervisor receives the same
explicit map and does not rewrite it. Generic non-pilot campaign behavior is
unchanged when no explicit launch environment is supplied.

The candidate uses a stable dedicated node-local temporary path. Qualification
and pilot operations create that path exclusively and clean it on exit. Its path
is part of identity; validation of archived evidence does not require the path
to remain present after job cleanup. No scheduler variables or credentials enter
the scientific subprocess environment.

Tests cover actual child environment equality, environment insertion/removal
preventing reuse, rehashed receipt forgery, legacy schema refusal, pilot rejection
before task-root creation, and preservation through the SCBA supervisor.

## PMIx relocation follow-up

The first environment candidate (33e785d) passed all six MPI micro allgathers
and both ten-test fault suites on Aspire2A, but its probe correctly failed on
nonempty micro stderr. PMIx searched the old compiled installation prefix for
its compression component/help data. That attempt remains immutable failure
evidence and does not authorize qualification.

This follow-up adds PMIX_PREFIX equal to each runtime prefix to the exact
sanitized map and uses fresh pilot scheduler names. It does not silence PMIx
warnings. The existing zlib component should be discovered at the relocated
prefix; fresh bounded diagnostics must establish that before qualification.
See the official PMIx [relocation documentation](https://docs.openpmix.org/en/latest/installing-pmix/installation-location.html)
and [v5.0.8 installdirs environment implementation](https://github.com/openpmix/openpmix/blob/v5.0.8/src/mca/pinstalldirs/env/pmix_pinstalldirs_env.c).

## Dependency startup environment follow-up

The PMIx candidate (81903bb) passed qualification, but its first pilot job
25547851.pbs101 failed before creating a scientific task root. Its submitted
environment exactly matched the qualified map. A local fresh-process
reproduction found that importing the pilot adds `KMP_DUPLICATE_LIB_OK=True`:
the pilot imports the qualification/release drivers, which import
`_material_nq1_mpi`, `_mpi_tasks`, and then `threadpoolctl`.
The pinned `threadpoolctl` 3.6.0 source calls
`os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")` at import time.
Preparation had imported those modules before restoring the qualified map,
whereas a direct pilot launch performs those imports after receiving that map.

Read-only compute diagnostic 25548204.pbs101 reached F/0 on
`x1003c4s2b1n0` and confirmed the same single added environment entry. The
recursive qualification validator returned the exact report, while the
original runtime check returned false in 0.337 seconds before invoking pip.
In the diagnostic's isolated reset case, that runtime check passed in
10.96 seconds, with `pip freeze --all` reproducing package SHA-256
`5bbc800922a2572e5ceff88e4f8407c2ccff417b6780c6f0d4e8b462468ac73b`.
Preseeding the exact KMP value preserved the complete environment across fresh
imports; the old candidate's gate still correctly rejected that unqualified
extra key. Pinned inputs were unchanged. These observations establish the
startup mismatch, not a passing scientific task or authority to retry the
failed pilot.

This new candidate binds the dependency's exact existing default before Python startup.
The complete environment remains part of runtime, receipt, and pilot identity;
imports do not receive a post-startup exemption. Fresh-process tests cover both
normal import and the driver's `__main__` help path, checking every environment
entry before and after the real dependency imports. Changed or absent KMP
controls fail validation and cannot reuse a qualification check receipt.

Fresh scheduler names are `nq1kmp925-g1`, `nq1kmp925-g01`, `nq1kmp925-g15`,
`nq1kmp925-g2`, and `nq1kmp925-g25`. The failed PMIx pilot remains immutable.
The source and operations changes passed independent review. The reviewed
source is frozen as a new candidate for fresh qualification and a separately
identified five-task pilot. The affected local suites passed 200 tests on
Python 3.12 and 199 with one existing Python 3.12-only skip on Python 3.10.
This candidate has not yet passed fresh qualification or science; no convergence
or release claim follows from the startup fix.


## Durable scheduler evidence development — 2026-09-28

This new candidate reserves the fresh pilot scheduler names `nq1ds928-g1`,
`nq1ds928-g01`, `nq1ds928-g15`, `nq1ds928-g2`, and `nq1ds928-g25`.
The source scientific algorithms, thresholds, task order, complete launch
environment, and scheduler acceptance requirements are unchanged.

The earlier `material-nq1-kmp-pilot-001` execution is preserved: its second
scientific task has saved convergence, but its exact terminal PBS evidence was
not captured before history became unavailable. That execution is not accepted
as a complete pass, is not resumed, and supplies no authorization for this one.

A separately reviewed operations profile must capture terminal scheduler
evidence durably without depending on a conversational monitor being awake.
It must retain exact scheduler F/0, science, and applicable anchor gates and
allow no consumed-job retry. This candidate requires its own fresh probe,
qualification, virgin execution identity, and five-task pilot before any pilot
pass may be claimed. These source names alone do not implement a watcher or
authorize a scheduler submission. Nq=1 remains unreleased.
