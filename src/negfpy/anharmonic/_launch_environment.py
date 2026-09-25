"""Exact, sanitized launch environments for material candidate qualification.

No ambient variables are inherited. Job-specific state must be supplied as an
explicit path and is therefore part of runtime and check identity.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

RUNTIME_MARKER = "NEGFPY_QUALIFICATION_RUNTIME_SHA256"
THREAD_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _absolute(path: Any) -> str:
    value = os.fspath(path)
    if (not isinstance(value, str) or not value or "\x00" in value
            or not Path(value).is_absolute() or ".." in Path(value).parts
            or os.pathsep in value or value != os.path.abspath(value)):
        raise ValueError("Launch environment paths must be exact absolute paths")
    return value


def build_launch_environment(
    *, repo: Path, executable: Path, home: Path, tmpdir: Path, git_bin: Path,
) -> dict[str, str]:
    repo = Path(_absolute(repo))
    executable = Path(_absolute(executable))
    prefix = executable.parent.parent
    values = {
        "HOME": _absolute(home), "TMPDIR": _absolute(tmpdir),
        "PATH": os.pathsep.join((str(executable.parent), _absolute(git_bin), "/usr/bin", "/bin")),
        "PYTHONPATH": os.pathsep.join((str(repo / "src"), str(repo / "scripts"))),
        "LANG": "C", "LC_ALL": "C", "TZ": "UTC",
        "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONCOERCECLOCALE": "0",
        "PYTHONUTF8": "1",
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0",
        "OPAL_PREFIX": str(prefix), "PMIX_PREFIX": str(prefix),
        "OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,sm",
        "OMPI_MCA_rmaps_base_oversubscribe": "1",
        # threadpoolctl 3.6.0 sets this exact default during the pilot's imports.
        # Bind it before startup so imports preserve the qualified environment.
        "KMP_DUPLICATE_LIB_OK": "True",
        **{name: "1" for name in THREAD_KEYS},
    }
    return dict(sorted(values.items()))


def validate_launch_environment(
    environment: Mapping[str, str], *, repo: Path, executable: Path,
) -> dict[str, str]:
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str)
        or "\x00" in key or "\x00" in value for key, value in environment.items()
    ):
        raise ValueError("Invalid launch environment mapping")
    try:
        paths = environment["PATH"].split(os.pathsep)
        if len(paths) != 4:
            raise ValueError("Invalid launch tool path")
        expected = build_launch_environment(
            repo=repo, executable=executable, home=Path(environment["HOME"]),
            tmpdir=Path(environment["TMPDIR"]), git_bin=Path(paths[1]),
        )
    except (KeyError, TypeError, OSError) as error:
        raise ValueError("Invalid launch environment fields") from error
    if dict(environment) != expected:
        raise ValueError("Launch environment differs from the sanitized policy")
    return expected


def qualification_environment(runtime: Mapping[str, Any], *, repo: Path) -> dict[str, str]:
    try:
        environment = validate_launch_environment(
            runtime["launch_environment"], repo=repo, executable=Path(runtime["executable"]),
        )
        digest = runtime["runtime_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Runtime lacks its qualified launch environment") from error
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid qualified runtime identity")
    return environment | {RUNTIME_MARKER: digest}
