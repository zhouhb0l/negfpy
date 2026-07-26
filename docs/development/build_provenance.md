# Wheel build provenance

NEGFPy wheels carry a generated, non-Python file
`negfpy/_build_provenance.json`. The file is produced by the narrowly scoped
setuptools `build_py` hook after package sources have been copied into the
wheel staging tree. It is not tracked in `src/` and is not used by ordinary
source or editable imports.

For a wheel built directly from a clean, exact Git checkout, the record stores
the canonical 40- or 64-hex commit object ID, `git_dirty=false`, the declared
package version, and a SHA256 over every installed `.py` relative path followed
by its exact bytes. The JSON is canonical and is included in wheel `RECORD`.
The Git root is resolved from the build-hook module location, never from the
invoking working directory.

At runtime, an installed metadata file takes precedence over Git repositories
in ancestor directories. This prevents a wheel under a repository-local
`.venv` from being attributed to the enclosing checkout. The record is used
only when it is a regular non-symlink file with the exact schema and keys,
strict JSON (no duplicates or nonfinite values), the installed version, a
canonical Git object ID, and a package-source hash matching the installed
Python bytes. A present but invalid record fails closed to unknown Git state;
it never falls through to an ancestor repository.

Dirty direct builds record `git_dirty=true`. Builds without an exact Git root,
including wheels rebuilt from an sdist, record unknown Git state. Both are
noncertifying. All release-report paths additionally require
`git_dirty is False`; a revision and source hash alone are insufficient.

The embedded JSON is an integrity binding, not a digital signature. Release
authenticity still comes from the external checksummed Git bundle, clean
worktree verification, environment manifest, launch receipt, and exact-commit
remote certification. Altering installed `.py` bytes invalidates the embedded
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
