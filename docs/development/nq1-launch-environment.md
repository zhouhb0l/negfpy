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
