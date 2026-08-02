#!/usr/bin/env python3
"""One-shot hourly monitor for the hash-bound bulk-RTA release campaign.

The monitor is deliberately conservative: it records reachability and remote
publication state, but it never changes scientific controls or resumes a task
unless a future, explicitly hash-bound recovery authority is supplied.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REMOTE_ROOT_PREFIX = "/home/hangbo/negfpy-validation/"
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:/=-]+$")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _append_report(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        timestamp = entry["timestamp"]
        lines = [f"\n### {timestamp}\n", "\n"]
        for key, value in entry.items():
            if key == "timestamp":
                continue
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, sort_keys=True)
            else:
                rendered = str(value)
            lines.append(f"- {key}: {rendered}\n")
        stream.write("".join(lines))
        stream.flush()
        os.fsync(stream.fileno())
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _validated_remote_state(state: dict[str, Any]) -> tuple[str, str, str] | None:
    phase = state.get("phase")
    if phase in {"prelaunch_hardening", "complete", "blocked"}:
        return None
    if phase != "running":
        raise ValueError("state.phase must be a recognized release phase")
    commit = state.get("commit")
    session = state.get("remote_session")
    root = state.get("remote_root")
    if (
        not isinstance(commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
        or not isinstance(session, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", session)
        or not isinstance(root, str)
        or not root.startswith(REMOTE_ROOT_PREFIX)
        or not SAFE_TOKEN.fullmatch(root)
    ):
        raise ValueError("running state has invalid hash, session, or remote root")
    return commit, session, root


def _probe_remote(session: str, root: str) -> dict[str, Any]:
    command = (
        "set -eu; "
        "printf 'hostname='; hostname; "
        f"if tmux has-session -t {session!s} 2>/dev/null; then printf 'session=alive\\n'; "
        "else printf 'session=absent\\n'; fi; "
        f"if test -f {root}/release-completion-receipt.json; then printf 'receipt=present\\n'; "
        "else printf 'receipt=absent\\n'; fi; "
        f"if test -f {root}/launcher-exit-code.json; then printf 'exit=present\\n'; "
        "else printf 'exit=absent\\n'; fi"
    )
    result = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "IdentitiesOnly=yes",
            "negf-heavy",
            "bash",
            "-lc",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        return {
            "remote_reachable": False,
            "ssh_returncode": result.returncode,
            "stderr_sha256": _sha256_bytes(result.stderr.encode()),
        }
    facts: dict[str, Any] = {"remote_reachable": True}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            facts[key] = value
    return facts


def run_once(state_path: Path, report_path: Path) -> int:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        state = _read_json(state_path)
        phase = state.get("phase")
        running = _validated_remote_state(state)
        entry: dict[str, Any] = {
            "timestamp": timestamp,
            "phase": phase,
            "action": "observe_only",
            "scientific_restart": "not_permitted_without_hash_bound_recovery_authority",
        }
        if running is None:
            entry["status"] = "no_remote_campaign_to_probe"
            _append_report(report_path, entry)
            return 0
        _commit, session, root = running
        entry.update(_probe_remote(session, root))
        if entry.get("remote_reachable") is False:
            entry["status"] = "remote_unreachable; retry_next_hour"
        elif entry.get("receipt") == "present":
            entry["status"] = "completion_receipt_present; independent_retrieval_required"
        elif entry.get("session") == "alive":
            entry["status"] = "scientific_campaign_running"
        else:
            entry["status"] = "session_absent_without_receipt; preserve_checkpoints_and_diagnose"
        _append_report(report_path, entry)
        return 0
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        # Monitor failure is visible and retried next hour.
        _append_report(
            report_path,
            {
                "timestamp": timestamp,
                "phase": "monitor_error",
                "status": "monitor_error; no_scientific_action_taken",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    return run_once(args.state.expanduser(), args.report.expanduser())


if __name__ == "__main__":
    raise SystemExit(main())
