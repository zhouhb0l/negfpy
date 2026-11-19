"""Run a bounded phonon–phonon calculation from one JSON file."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from . import __version__


def main(argv=None):
    parser = argparse.ArgumentParser(prog="negfpp", description=__doc__)
    parser.add_argument("input_file", type=Path, nargs="?", metavar="INPUT.json")
    parser.add_argument("--input", dest="input_option", type=Path, metavar="INPUT.json")
    parser.add_argument("--check-input", action="store_true", help="Validate JSON and input hashes without calculating.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    if args.input_file and args.input_option:
        parser.error("provide the input file once")
    path = args.input_file or args.input_option
    if path is None:
        parser.error("provide INPUT.json")
    try:
        from .input import load_input
        config = load_input(path)
        if args.check_input:
            print(f"Input and hashes verified: {path}")
            return 0
        from .workflow import run
        report, output = run(config)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, TimeoutError) as exc:
        print(f"negfpp: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Status: {report['status']}")
    if report.get("calculation_type") == "method1-local-fd":
        print(f"Local FD: {report['stop_reason']}; no new SCBA maps")
        if report.get("linear_response"):
            print(f"Conductance: {report['linear_response']['conductance_w_per_kelvin']:.12e} W/K per transverse cell")
    else:
        print(f"SCBA: {report['stop_reason']}; {report['counts']['completed']} completed maps")
    if report.get("currents_w"):
        print(f"Left current: {report['currents_w']['left']:.12e} W per transverse cell")
    if report.get("failure_reason"):
        print(f"Failure: {report['failure_reason']}", file=sys.stderr)
    print(f"Results: {output / 'report.json'}")
    return 0 if report["converged"] else 1
