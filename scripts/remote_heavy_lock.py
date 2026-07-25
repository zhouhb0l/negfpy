"""Validation for a pinned micromamba explicit lock."""

from __future__ import annotations

import hashlib
import pathlib


def validate_explicit_lock(path: pathlib.Path, *, expected_sha256: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing regular explicit lock: {path}")
    first_directive = next(
        (
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if first_directive != "@EXPLICIT":
        raise ValueError(
            f"first noncomment micromamba lock line must be @EXPLICIT: {path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"explicit lock SHA256 mismatch: {digest} != {expected_sha256}"
        )
    return digest
