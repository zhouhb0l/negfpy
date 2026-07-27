"""Read-only status and checksum-verified retrieval helpers."""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import remote_heavy_validation as core


def _default_session(commit: str) -> str:
    return f"negf-toy-{core._validate_commit(commit)[:12]}"


def status(
    *,
    commit: str,
    session: str | None,
    runner: core.CommandRunner,
) -> dict[str, Any]:
    commit = core._validate_commit(commit)
    core.check_ssh_alias(runner)
    selected = core._validate_session(session or _default_session(commit))
    remote = core._remote_shell_dir(commit)
    command = (
        f"if {core.REMOTE_TMUX} has-session -t {selected} 2>/dev/null; "
        "then printf '{\"state\":\"running\"}\\n'; "
        f"elif test -d {remote}/artifacts/job.lock; "
        "then printf '{\"state\":\"stale_lock\"}\\n'; "
        f"elif test -f {remote}/artifacts/job-status.json && "
        f"test -f {remote}/artifacts/SHA256SUMS; "
        f"then cat {remote}/artifacts/job-status.json; "
        f"elif test -f {remote}/artifacts/job-status.json; "
        "then printf '{\"state\":\"finalizing\"}\\n'; "
        "else printf '{\"state\":\"missing\"}\\n'; fi"
    )
    result = runner.run(core.ssh_argv(command))
    if runner.dry_run:
        return {"state": "dry-run", "commit": commit, "session": selected}
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise core.RemoteValidationError(
            "remote status was not valid JSON"
        ) from error
    value.setdefault("commit", commit)
    value.setdefault("session", selected)
    return value


def _fresh_fetch_destination(
    *,
    destination_root: pathlib.Path,
    commit: str,
    dry_run: bool,
) -> pathlib.Path:
    """Create a fresh lexical destination without following any symlink."""

    root = pathlib.Path(os.path.abspath(os.fspath(destination_root.expanduser())))
    current = pathlib.Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink():
                raise core.RemoteValidationError(
                    f"fetch destination root traverses a symlink: {current}"
                )
            if not current.is_dir():
                raise core.RemoteValidationError(
                    f"fetch destination root is not a directory: {current}"
                )
        elif not dry_run:
            current.mkdir()
    destination = root / commit
    if os.path.lexists(destination):
        raise core.RemoteValidationError(
            f"fetch destination already exists: {destination}"
        )
    if not dry_run:
        destination.mkdir()
        if destination.is_symlink() or not destination.is_dir():
            raise core.RemoteValidationError(
                f"fetch destination is not a fresh regular directory: {destination}"
            )
    return destination


def _prune_empty_directories(root: pathlib.Path) -> None:
    """Canonicalize a fresh fetch to its checksum-covered file projection.

    ``scp -r`` preserves runtime scratch directories even when they contain no
    files. Directory entries are not representable in ``SHA256SUMS`` and the
    publication importer deliberately rejects such unreferenced topology.
    Remove empty directories bottom-up; ``rmdir`` cannot remove a directory
    containing a file, symlink, special entry, or nonempty child.
    """

    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.relative_to(root).parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def _validate_checksum_file_projection(
    root: pathlib.Path,
    *,
    expected_regular_files: set[str],
) -> None:
    """Require an exact regular-file tree after empty-directory pruning."""

    symlinks = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    }
    special_entries = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if not path.is_symlink()
        and not path.is_file()
        and not path.is_dir()
    }
    regular_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    required_directories: set[str] = set()
    for relative in regular_files:
        parent = pathlib.PurePosixPath(relative).parent
        while parent != pathlib.PurePosixPath("."):
            required_directories.add(parent.as_posix())
            parent = parent.parent
    observed_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    }
    if symlinks or special_entries or regular_files != expected_regular_files:
        raise core.RemoteValidationError(
            "fetched artifact file projection changed during validation"
        )
    if observed_directories != required_directories:
        raise core.RemoteValidationError(
            "fetched artifacts retain an empty or unreferenced directory"
        )


def fetch(
    *,
    commit: str,
    destination_root: pathlib.Path,
    runner: core.CommandRunner,
) -> pathlib.Path:
    commit = core._validate_commit(commit)
    job = status(commit=commit, session=None, runner=runner)
    if not runner.dry_run:
        if job.get("state") not in {"completed", "failed"}:
            raise core.RemoteValidationError(
                f"remote job is not terminal: {job.get('state')!r}"
            )
        if job.get("commit") != commit:
            raise core.RemoteValidationError(
                "remote terminal status commit mismatch"
            )
        if not isinstance(job.get("returncode"), int):
            raise core.RemoteValidationError(
                "remote terminal status lacks an integer returncode"
            )

    destination = _fresh_fetch_destination(
        destination_root=destination_root,
        commit=commit,
        dry_run=runner.dry_run,
    )
    runner.run(
        core.scp_argv(
            "-r",
            f"{core.SSH_ALIAS}:"
            f"{core._remote_relative_dir(commit)}/artifacts/.",
            str(destination),
        )
    )
    if not runner.dry_run:
        if (destination / "job.lock").exists():
            raise core.RemoteValidationError(
                "fetched artifacts contain an active job.lock"
            )
        symlinks = [
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_symlink()
        ]
        if symlinks:
            raise core.RemoteValidationError(
                "fetched artifacts contain symlinks: " + ", ".join(symlinks)
            )
        nested_manifests = sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("SHA256SUMS")
            if path.relative_to(destination).as_posix() != "SHA256SUMS"
        )
        if nested_manifests:
            raise core.RemoteValidationError(
                "nested SHA256SUMS files are forbidden: "
                + ", ".join(nested_manifests)
            )
        special_entries = [
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if not path.is_symlink()
            and not path.is_file()
            and not path.is_dir()
        ]
        if special_entries:
            raise core.RemoteValidationError(
                "fetched artifacts contain special filesystem entries: "
                + ", ".join(special_entries)
            )
        verified = core.verify_sha256sums(destination)
        regular = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.relative_to(destination).as_posix() != "SHA256SUMS"
        }
        if set(verified) != regular:
            missing = sorted(regular - set(verified))
            stale = sorted(set(verified) - regular)
            raise core.RemoteValidationError(
                "SHA256SUMS does not cover exactly all fetched regular files; "
                f"unlisted={missing}, nonexistent={stale}"
            )
        _prune_empty_directories(destination)
        _validate_checksum_file_projection(
            destination,
            expected_regular_files=regular | {"SHA256SUMS"},
        )
        core._atomic_write_json(
            destination / "fetch-receipt.json",
            {
                "schema": "negfpy.remote_validation.fetch_receipt.v1",
                "fetched_utc": core._utc_now(),
                "commit": commit,
                "remote": (
                    f"~/{core._remote_relative_dir(commit)}/artifacts"
                ),
                "verified_files": list(verified),
            },
        )
    return destination
