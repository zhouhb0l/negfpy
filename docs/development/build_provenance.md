# Wheel build provenance

NEGFPy wheels carry a generated, non-Python file
`negfpy/_build_provenance.json`. The file is produced by the narrowly scoped
setuptools `build_py` hook after package sources have been copied into the
wheel staging tree. It is not tracked in `src/` and is not used by ordinary
source or editable imports.

For a wheel built directly from a clean, exact Git checkout, the record stores
the canonical 40- or 64-hex commit object ID, `git_dirty=false`, the declared
package version, and a SHA256 over the complete installed package payload.
The hash is domain-separated and length-prefixes the file count, every UTF-8
relative path, and every content byte string. It excludes only the generated
provenance JSON and runtime `__pycache__`/`.pyc`/`.pyo` files; symlinks and
special files fail closed. The JSON is canonical and is included in wheel
`RECORD`. The Git root is resolved from the build-hook module location, never
from the invoking working directory.

Before copying, the hook claims a build root with a regular, non-symlink
ownership marker in `build_lib.parent`, deliberately outside the wheel staging
root. Its filename contains a SHA256 binding to the exact resolved build-root
path, while the strict JSON records its schema plus the exact resolved project
and build roots. It is created only when `build_lib/negfpy` is absent. A
pre-existing package target without that exact valid marker is rejected and
preserved byte-for-byte; copied, malformed, or symlinked markers do not
authorize removal. A valid marker from the former in-staging location is
migrated outside `build_lib` before any package removal, and an invalid legacy
marker fails closed. Only a previously claimed root may have its exact
generated `build_lib/negfpy` target removed for a repeated build. The hook also
rejects the project root, source ancestors, source descendants, symlinks, and
every target overlapping canonical `src/negfpy`. The wheel regression rejects
ownership markers and absolute project or build paths anywhere in the archive.

After copying, it compares every package path and byte against both
`src/negfpy` and regular HEAD blobs. It independently compares the current
regular `pyproject.toml` bytes with the exact regular HEAD blob and requires
the setuptools package version to equal the static version declared by both.
Thus a clean record binds package payload, project declaration, and version to
one commit even when assume-unchanged or skip-worktree flags hide mutations.
Ignored stale output can never be attributed to a clean commit.

At runtime, live Git is permitted only when the resolved import root is
exactly `<project>/src/negfpy` beside that project's `.git` and
`pyproject.toml`. That authenticated source layout compares the complete
current package payload directly with regular HEAD blobs, so ignored files and
index-hidden mutations remain dirty. The live-source path also compares the
regular current `pyproject.toml` bytes and declared version directly with the
regular HEAD blob, so `git_dirty=false` implies that `package_version` is the
clean HEAD declaration. It ignores any forged embedded record and records it
as worktree dirt. Every other import is treated as an
installed package: its embedded record is used when valid, and an absent or
invalid record remains unknown without consulting ancestor repositories. This
prevents a wheel under a repository-local `.venv` from being attributed to the
enclosing checkout. The record is used
only when it is a regular non-symlink file with the exact schema and keys,
strict JSON (no duplicates or nonfinite values), the installed version, a
canonical Git object ID, and a package-source hash matching the installed
package payload bytes. A present but invalid record fails closed to unknown
Git state; it never falls through to an ancestor repository.

Dirty direct builds record `git_dirty=true`. The only accepted unknown pair is
`git_revision="unknown", git_dirty="unknown"`. Builds without an exact Git
root, including wheels rebuilt from an sdist, record that pair. Both are
noncertifying. All release-report paths additionally require
`git_dirty is False`; a revision and source hash alone are insufficient.

The embedded JSON is an integrity binding, not a digital signature. Release
authenticity still comes from the external checksummed Git bundle, clean
worktree verification, environment manifest, launch receipt, and exact-commit
remote certification. Altering installed package bytes invalidates the embedded
record unless the metadata is also replaced, which remains detectable against
those external release receipts.

A direct PEP 517 wheel can be checked with:

```bash
python -m pip wheel --no-deps --no-build-isolation .
python -m pip install --no-deps --target <empty-target> <wheel>
```

Inspect the wheel archive and its `RECORD`, then import from the empty target
and confirm `code_provenance()` reports the build commit, `git_dirty=false`,
and the same package-source SHA256 stored in the JSON.
