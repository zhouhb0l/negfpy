"""Dedicated known-hosts fingerprint verification."""

from __future__ import annotations

import base64
import hashlib
import pathlib
import shlex


def validate_known_hosts_fingerprint(
    ssh_config: str,
    expected_host: str,
    *,
    expected: str,
) -> pathlib.Path:
    values: dict[str, list[str]] = {}
    for raw_line in ssh_config.splitlines():
        line = raw_line.strip()
        if line and " " in line:
            key, value = line.split(None, 1)
            values.setdefault(key.lower(), []).append(value.strip())
    paths = [
        pathlib.Path(token).expanduser()
        for raw in values.get("userknownhostsfile", [])
        for token in shlex.split(raw)
        if token.lower() != "none"
    ]
    dedicated = [path for path in paths if path.name == "known_hosts_negf_heavy"]
    if len(dedicated) != 1:
        raise ValueError(
            "negf-heavy must select one dedicated known_hosts_negf_heavy file"
        )
    path = dedicated[0]
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"dedicated known-hosts path is not a regular file: {path}"
        )
    fingerprints: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        hosts = fields[0].split(",")
        if expected_host not in hosts and f"[{expected_host}]:22" not in hosts:
            continue
        try:
            index = fields.index("ssh-ed25519")
            blob = base64.b64decode(fields[index + 1], validate=True)
        except (ValueError, IndexError, base64.binascii.Error):
            continue
        digest = hashlib.sha256(blob).digest()
        fingerprints.add(
            "SHA256:"
            + base64.b64encode(digest).decode("ascii").rstrip("=")
        )
    if fingerprints != {expected}:
        raise ValueError(
            f"host ED25519 fingerprint set must equal {{{expected}}}: {path}"
        )
    return path
