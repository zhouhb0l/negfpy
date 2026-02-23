"""Quantum ESPRESSO q2r.x IFC adapter to the unified IFC schema."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.modeling.validators import validate_ifc_data


_SPECIES_RE = re.compile(r"^\s*(\d+)\s+'([^']+)'\s+([Ee0-9\+\-\.]+)\s*$")


def _signed_fft_index(m: int, n: int) -> int:
    idx = int(m) - 1
    # For even grids, map the Nyquist index to the negative side
    # so shifts are in a symmetric signed range.
    if n > 1 and idx >= n // 2:
        idx -= n
    return idx


def _next_nonempty(lines: list[str], start: int) -> tuple[int, str]:
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if line:
            return i, line
        i += 1
    raise ValueError("Unexpected end of file while parsing QE IFC file.")


def read_qe_q2r_ifc(source: Any) -> IFCData:
    """Parse QE q2r.x text IFC file (``*.fc``) into ``IFCData``."""

    path = Path(source)
    lines = [ln.rstrip("\n") for ln in path.open("r", encoding="utf-8")]

    i, line = _next_nonempty(lines, 0)
    first = line.split()
    if len(first) < 3:
        raise ValueError("Invalid QE IFC header line.")
    ntyp = int(first[0])
    nat = int(first[1])
    ibrav = int(first[2])
    i += 1

    species_masses: dict[int, float] = {}
    species_labels: dict[int, str] = {}
    for _ in range(ntyp):
        i, line = _next_nonempty(lines, i)
        m = _SPECIES_RE.match(line)
        if m is None:
            raise ValueError(f"Invalid species line in QE IFC file: '{line}'")
        sid = int(m.group(1))
        species_labels[sid] = m.group(2).strip()
        species_masses[sid] = float(m.group(3))
        i += 1

    atom_species: list[int] = []
    atom_positions: list[list[float]] = []
    for _ in range(nat):
        i, line = _next_nonempty(lines, i)
        toks = line.split()
        if len(toks) < 5:
            raise ValueError(f"Invalid atomic-position line in QE IFC file: '{line}'")
        atom_species.append(int(toks[1]))
        atom_positions.append([float(toks[2]), float(toks[3]), float(toks[4])])
        i += 1

    i, line = _next_nonempty(lines, i)
    lattice_vectors = None
    born_charges = None
    if line in {"T", "F"}:
        has_long_range = (line == "T")
        i += 1
        if has_long_range:
            eps = []
            for _ in range(3):
                i, row = _next_nonempty(lines, i)
                eps.append([float(x) for x in row.split()[:3]])
                i += 1
            born = np.zeros((nat, 3, 3), dtype=float)
            for ia in range(nat):
                i, _ = _next_nonempty(lines, i)  # atom index line
                i += 1
                for a in range(3):
                    i, row = _next_nonempty(lines, i)
                    born[ia, a, :] = [float(x) for x in row.split()[:3]]
                    i += 1
            lattice_vectors = np.asarray(eps, dtype=float)
            born_charges = born

    i, line = _next_nonempty(lines, i)
    nr_toks = line.split()
    if len(nr_toks) < 3:
        raise ValueError("Missing real-space IFC grid dimensions in QE IFC file.")
    nr1, nr2, nr3 = int(nr_toks[0]), int(nr_toks[1]), int(nr_toks[2])
    ngrid = nr1 * nr2 * nr3
    i += 1

    ndof = 3 * nat
    blocks: dict[tuple[int, int, int], np.ndarray] = {}
    n_ifc_blocks = nat * nat * 3 * 3

    for _ in range(n_ifc_blocks):
        i, line = _next_nonempty(lines, i)
        head = line.split()
        if len(head) < 4:
            raise ValueError(f"Invalid IFC block header in QE IFC file: '{line}'")
        a = int(head[0]) - 1
        b = int(head[1]) - 1
        ia = int(head[2]) - 1
        ib = int(head[3]) - 1
        if not (0 <= ia < nat and 0 <= ib < nat and 0 <= a < 3 and 0 <= b < 3):
            raise ValueError(f"Out-of-range IFC block indices in line: '{line}'")
        i += 1

        for _ in range(ngrid):
            i, row = _next_nonempty(lines, i)
            toks = row.split()
            if len(toks) < 4:
                raise ValueError(f"Invalid IFC value line in QE IFC file: '{row}'")
            m1, m2, m3 = int(toks[0]), int(toks[1]), int(toks[2])
            val = float(toks[3].replace("D", "E"))
            dx = _signed_fft_index(m1, nr1)
            dy = _signed_fft_index(m2, nr2)
            dz = _signed_fft_index(m3, nr3)
            key = (dx, dy, dz)
            if key not in blocks:
                blocks[key] = np.zeros((ndof, ndof), dtype=np.complex128)
            row_idx = 3 * ia + a
            col_idx = 3 * ib + b
            blocks[key][row_idx, col_idx] = val
            i += 1

    terms = tuple(IFCTerm(dx=k[0], dy=k[1], dz=k[2], block=v) for k, v in sorted(blocks.items()))
    masses = np.asarray([species_masses[sid] for sid in atom_species], dtype=float)
    atom_symbols = tuple(species_labels[sid] for sid in atom_species)
    ifc = IFCData(
        masses=masses,
        dof_per_atom=3,
        terms=terms,
        units="sqrt(Ry/(me*bohr^2))",
        metadata={
            "source_format": "qe_q2r_fc",
            "path": str(path),
            "ntyp": ntyp,
            "nat": nat,
            "ibrav": ibrav,
            "nr": (nr1, nr2, nr3),
            "force_constant_unit": "Ry/bohr^2",
            "mass_unit": "2*electron_mass (QE Ry mass unit)",
            "omega_internal_unit": "sqrt(Ry/(2*me*bohr^2))",
            "born_charges": born_charges,
        },
        lattice_vectors=lattice_vectors,
        atom_positions=np.asarray(atom_positions, dtype=float),
        atom_symbols=atom_symbols,
        index_convention="qe_q2r_fft",
    )
    validate_ifc_data(ifc)
    return ifc
