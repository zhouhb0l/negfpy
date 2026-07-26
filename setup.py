"""Setuptools command registration for wheel build provenance."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from setuptools import setup


_hook_path = Path(__file__).resolve().with_name("_negfpy_build.py")
_spec = spec_from_file_location("_negfpy_build", _hook_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load build hook from {_hook_path}")
_hook = module_from_spec(_spec)
_spec.loader.exec_module(_hook)

setup(cmdclass={"build_py": _hook.ProvenanceBuildPy})
