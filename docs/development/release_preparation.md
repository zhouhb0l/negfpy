# Release preparation — 2026-10-26

The workspace has one canonical checkout, `negfpy/`. All historical linked
worktrees were snapshotted outside the repository before consolidation,
including tracked changes, untracked files, ignored artifacts and symlinks.
Original Git branches and archive refs retain every original HEAD. Recovery
instructions and a Git bundle are stored in the workspace under
`archives/release-prep-20261026/`. Reports and frozen remote runs are retained.

The active tree removes 23 dated HPC study launchers and 24 tests specific to
those launchers. The library physics, guards and thresholds are unchanged.
The complete-energy import module and portable example replace reliance on a
chain of dated launch scripts. Generic bulk, sparse and maintainer utilities
remain available. Historical implementation experiments remain in Git rather
than being presented as additional active source checkouts.

Version remains `2.0.0.dev0`. Local import, synthetic tests and wheel checks
assess this consolidation; they are not a new material calculation or release
certification. No package publication, production IFC adoption or physical
transport acceptance occurs here.

Before public release, decide the supported API and license/distribution terms,
review third-party/material redistribution rights, specify tested environments,
and qualify the advertised workflows at an appropriate scientific scope.
Research-model approximation and convergence limits must remain visible.
