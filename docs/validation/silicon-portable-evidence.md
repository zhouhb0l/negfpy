# Portable silicon evidence with external file records

## Operational release handoff

Use `finalize` for the release archival path. It refuses an uncertified final
report, a pre-existing staging tree, any symlink, or an existing output
directory. It stages every external `path` plus `sha256` record, creates and
self-verifies the portable manifest, copies the exact certified report, and
publishes the complete handoff directory with an atomic Linux no-replace
rename:

```bash
python scripts/seal_phono3py_si_release_evidence.py finalize \
  --artifact-root /path/to/artifacts \
  --release-report /path/to/artifacts/release/silicon-release.json \
  --handoff-directory /path/to/release-handoff
```

The destination parent must already exist and must be outside the artifact
root. The destination itself must not exist. Publication contains exactly:

```text
release-handoff/
├── SHA256SUMS
├── portable-manifest.json
└── silicon-release-report.json
```

`SHA256SUMS` uses the standard two-space format and covers both other files.
The report copy is byte-identical to the report inside the sealed artifact
tree. Preparation happens in a private sibling directory; a failed
pre-publication operation removes both that temporary directory and any
external staging tree created by the operation. Publication never replaces an
existing path.

Transfer the artifact tree and handoff directory separately. Obtain
`SHA256SUMS` through the authenticated transfer/checksum channel, then verify
the relocated material with:

```bash
python scripts/seal_phono3py_si_release_evidence.py verify-handoff \
  --artifact-root /path/to/retrieved/artifacts \
  --trusted-sha256sums /path/to/retrieved-handoff/SHA256SUMS
```

Retrieval requires the handoff directory to contain exactly the three files
above, validates both trusted digests, requires the copied and sealed reports
to remain `release_certified=true`, and performs trusted-digest portable
verification against the relocated artifact root.

Both commands report
`scientific_recomputation_performed=false`. This workflow preserves and
authenticates an existing scientific decision; it never reruns the silicon
oracle or upgrades an uncertified report.

## Manual staging and sealing

The portable silicon seal normally requires every strict JSON `path` plus
`sha256` record to identify a regular file inside the artifact root. Release
reports may legitimately bind immutable inputs or subordinate reports stored
outside that root. Those files must be staged explicitly before sealing:

```bash
python scripts/seal_phono3py_si_release_evidence.py stage \
  --artifact-root /path/to/artifacts
```

Staging recursively discovers external records in strict JSON evidence,
rejects ambiguous paths, traversal, symbolic links, hard links, special or
missing files, and verifies each declared SHA256 and optional size. It copies
each unique object into
`.negfpy-external-evidence/<alias-payload-sha256>/objects/` and emits the
canonical immutable `aliases.json` beside those objects. The command prints
the exact alias-manifest path and hashes.

Bind that manifest explicitly while sealing:

```bash
python scripts/seal_phono3py_si_release_evidence.py seal \
  --artifact-root /path/to/artifacts \
  --release-report /path/to/artifacts/release/silicon-release.json \
  --external-alias-manifest /path/to/artifacts/.negfpy-external-evidence/<sha256>/aliases.json \
  --output /path/outside/artifacts/portable-manifest.json
```

The version-2 portable manifest binds the complete alias identity. Subsequent
`verify` calls need only the relocated artifact root and portable manifest;
the original external paths need not exist. Omitting the alias manifest,
adding unused aliases, changing an alias or staged object, or relocating an
incomplete staging subtree fails verification. Evidence with no external
records continues to use the unchanged version-1 portable format.

## Verification scope

The portable manifest is not a second scientific certification authority.
With no trusted digest, `verify` reports
`verification_scope=self_authenticated_integrity_and_relocation_only`: this
detects damage and incomplete retrieval, but a self-hash alone cannot
authenticate who created the report. For release retrieval, take both digests
from the checksum record obtained through the authenticated remote-transfer
channel and require them explicitly:

```bash
python scripts/seal_phono3py_si_release_evidence.py verify \
  --artifact-root /path/to/retrieved/artifacts \
  --manifest /path/to/retrieved/portable-manifest.json \
  --expected-manifest-sha256 <trusted-manifest-sha256> \
  --expected-release-report-sha256 <trusted-report-sha256>
```

Only this form reports
`verification_scope=trusted_digest_authenticated_integrity`. It authenticates
the retrieved bytes against those trusted digests; it does not rerun the
scientific release validator.
