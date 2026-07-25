"""Read-only status and checksum-verified retrieval helpers."""

from __future__ import annotations

import json
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
        f"if test -d {remote}/artifacts/job.lock || "
        f"{core.REMOTE_TMUX} has-session -t {selected} 2>/dev/null; "
        "then printf '{\"state\":\"running\"}\\n'; "
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

    destination = destination_root.resolve() / commit
    if destination.exists() and any(destination.iterdir()):
        raise core.RemoteValidationError(
            f"fetch destination is not empty: {destination}"
        )
    if not runner.dry_run:
        destination.mkdir(parents=True, exist_ok=True)
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
        verified = core.verify_sha256sums(destination)
        regular = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and path.name != "SHA256SUMS"
        }
        if set(verified) != regular:
            missing = sorted(regular - set(verified))
            stale = sorted(set(verified) - regular)
            raise core.RemoteValidationError(
                "SHA256SUMS does not cover exactly all fetched regular files; "
                f"unlisted={missing}, nonexistent={stale}"
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
