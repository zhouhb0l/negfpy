"""Dedicated known-hosts fingerprint verification."""

from __future__ import annotations

import base64
import hashlib
import os
import pathlib
import shlex
import stat


def validate_known_hosts_fingerprint(
    ssh_config: str,
    expected_host: str,
    *,
    expected: str,
    expected_path: pathlib.Path | None = None,
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
    dedicated = [
        path.resolve(strict=False)
        for path in paths
        if path.name == "known_hosts_negf_heavy"
    ]
    if len(dedicated) != 1:
        raise ValueError(
            "negf-heavy must select one dedicated known_hosts_negf_heavy file"
        )
    path = dedicated[0]
    if (
        expected_path is not None
        and path != expected_path.expanduser().resolve(strict=False)
    ):
        raise ValueError(
            "negf-heavy dedicated known-hosts path does not match the "
            f"locked path: {path}"
        )
    if not path.is_file() or path.is_symlink():
        raise ValueError(
            f"dedicated known-hosts path is not a regular file: {path}"
        )
    metadata = path.stat()
    if metadata.st_uid != os.getuid():
        raise ValueError(
            f"dedicated known-hosts file is not owned by the current user: {path}"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise ValueError(
            "dedicated known-hosts file must not be group/other writable: "
            f"{path}"
        )
    fingerprints: set[str] = set()
    matching_entries = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        fields = stripped.split()
        if len(fields) < 3 or fields[0].startswith("@"):
            raise ValueError(
                f"invalid entry in dedicated known-hosts file: {path}"
            )
        hosts = fields[0].split(",")
        if expected_host not in hosts and f"[{expected_host}]:22" not in hosts:
            raise ValueError(
                "dedicated known-hosts file contains an unexpected host: "
                f"{path}"
            )
        matching_entries += 1
        if fields[1] != "ssh-ed25519":
            raise ValueError(
                "dedicated host entry must use ssh-ed25519 exclusively: "
                f"{path}"
            )
        try:
            blob = base64.b64decode(fields[2], validate=True)
        except (IndexError, base64.binascii.Error) as error:
            raise ValueError(
                f"invalid ssh-ed25519 key in dedicated known-hosts file: {path}"
            ) from error
        digest = hashlib.sha256(blob).digest()
        fingerprints.add(
            "SHA256:"
            + base64.b64encode(digest).decode("ascii").rstrip("=")
        )
    if matching_entries != 1 or fingerprints != {expected}:
        raise ValueError(
            "host ED25519 file must contain exactly one pinned entry with "
            f"fingerprint {{{expected}}}: {path}"
        )
    return path
