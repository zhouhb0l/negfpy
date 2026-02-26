"""Phonopy IFC adapters to the unified IFC schema.

Supported sources:
- Normalized dict/json payload with ``masses``, ``dof_per_atom``, ``terms``.
- Phonopy text ``FORCE_CONSTANTS`` file path (with sibling ``SPOSCAR``/``POSCAR``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from negfpy.modeling.atomic_masses import get_atomic_mass_amu
from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.modeling.units import BOHR_M, EV_J, QE_RY_MASS_UNIT_KG, RYDBERG_J
from negfpy.modeling.validators import validate_ifc_data

AMU_KG = 1.66053906660e-27


def _load_source_payload(source: Any) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_term(item: dict[str, Any]) -> IFCTerm:
    if "translation" in item:
        dx, dy, dz = item["translation"]
    else:
        dx, dy, dz = item["dx"], item["dy"], item["dz"]
    block = np.asarray(item["block"], dtype=np.complex128)
    return IFCTerm(dx=int(dx), dy=int(dy), dz=int(dz), block=block)


def _signed_fft_index(raw_idx: int, n: int) -> int:
    if n > 1 and raw_idx >= (n // 2):
        return raw_idx - n
    return raw_idx


def _phonopy_fc_to_qe_factor() -> float:
    """Scale eV/Angstrom^2 -> Ry/bohr^2."""

    ev_to_ry = float(EV_J / RYDBERG_J)
    angstrom_to_bohr = float(1.0e-10 / BOHR_M)
    # 1/Angstrom^2 = 1/(Angstrom_to_bohr^2) * 1/bohr^2
    return ev_to_ry / (angstrom_to_bohr * angstrom_to_bohr)


def _amu_to_qe_mass_factor() -> float:
    """Scale amu -> QE Ry-mass units (2*electron_mass)."""

    return float(AMU_KG / QE_RY_MASS_UNIT_KG)


def _next_nonempty(lines: list[str], start: int) -> tuple[int, str]:
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if line:
            return i, line
        i += 1
    raise ValueError("Unexpected end of file while parsing phonopy FORCE_CONSTANTS.")


def _parse_poscar(path: Path) -> dict[str, Any]:
    lines = [ln.rstrip("\n") for ln in path.open("r", encoding="utf-8")]
    if len(lines) < 8:
        raise ValueError(f"POSCAR seems too short: '{path}'")

    scale = float(lines[1].split()[0])
    lv = np.asarray([[float(x) for x in lines[2 + i].split()[:3]] for i in range(3)], dtype=float) * scale

    # VASP5 format has species names line before counts.
    line5 = lines[5].split()
    has_symbols = not all(tok.lstrip("+-").isdigit() for tok in line5)
    if has_symbols:
        symbols = line5
        counts = [int(x) for x in lines[6].split()]
        i = 7
    else:
        counts = [int(x) for x in lines[5].split()]
        symbols = [f"X{j + 1}" for j in range(len(counts))]
        i = 6

    if i < len(lines) and lines[i].strip().lower().startswith("s"):
        i += 1
    if i >= len(lines):
        raise ValueError(f"POSCAR missing coordinate mode line: '{path}'")
    coord_mode = lines[i].strip().lower()
    i += 1

    nat = sum(counts)
    if i + nat > len(lines):
        raise ValueError(f"POSCAR atom coordinate section is incomplete: '{path}'")

    coords = []
    for j in range(nat):
        toks = lines[i + j].split()
        if len(toks) < 3:
            raise ValueError(f"Invalid POSCAR coordinate line: '{lines[i + j]}'")
        coords.append([float(toks[0]), float(toks[1]), float(toks[2])])
    coords_arr = np.asarray(coords, dtype=float)

    if coord_mode.startswith("c") or coord_mode.startswith("k"):
        inv_lv = np.linalg.inv(lv)
        coords_arr = coords_arr @ inv_lv
    elif not coord_mode.startswith("d"):
        raise ValueError(f"Unsupported POSCAR coordinate mode '{lines[i - 1].strip()}'.")
    coords_arr = np.mod(coords_arr, 1.0)

    atom_symbols: list[str] = []
    for sym, n in zip(symbols, counts):
        atom_symbols.extend([sym] * int(n))

    return {
        "lattice_vectors": lv,
        "atom_positions": coords_arr,
        "atom_symbols": tuple(atom_symbols),
        "species": tuple(symbols),
        "counts": tuple(int(v) for v in counts),
    }


def _parse_force_constants(path: Path) -> np.ndarray:
    lines = [ln.rstrip("\n") for ln in path.open("r", encoding="utf-8")]
    i, first = _next_nonempty(lines, 0)
    head = first.split()
    if len(head) < 1:
        raise ValueError("Invalid FORCE_CONSTANTS header.")
    nat = int(head[0])
    if len(head) >= 2 and int(head[1]) != nat:
        raise ValueError("FORCE_CONSTANTS first line must have matching nat values.")
    i += 1

    fc = np.zeros((nat, nat, 3, 3), dtype=float)
    for _ in range(nat * nat):
        i, pair = _next_nonempty(lines, i)
        toks = pair.split()
        if len(toks) < 2:
            raise ValueError(f"Invalid FORCE_CONSTANTS pair line: '{pair}'")
        ia = int(toks[0]) - 1
        ja = int(toks[1]) - 1
        i += 1
        if not (0 <= ia < nat and 0 <= ja < nat):
            raise ValueError(f"FORCE_CONSTANTS pair indices out of range: '{pair}'")
        for a in range(3):
            i, row = _next_nonempty(lines, i)
            vals = row.split()
            if len(vals) < 3:
                raise ValueError(f"Invalid FORCE_CONSTANTS matrix row: '{row}'")
            fc[ia, ja, a, :] = [float(vals[0]), float(vals[1]), float(vals[2])]
            i += 1
    return fc


def _round_frac(v: np.ndarray, tol: float = 1e-6) -> tuple[int, int, int]:
    return tuple(int(np.round(float(x) / tol)) for x in np.mod(v, 1.0))


def _is_translation_symmetry(
    frac: np.ndarray,
    symbols: tuple[str, ...],
    axis: int,
    n: int,
    tol: float = 1e-6,
) -> bool:
    if n <= 1:
        return True
    delta = np.zeros(3, dtype=float)
    delta[axis] = 1.0 / float(n)
    by_symbol: dict[str, set[tuple[int, int, int]]] = {}
    for p, s in zip(frac, symbols):
        by_symbol.setdefault(s, set()).add(_round_frac(np.asarray(p, dtype=float), tol=tol))
    for s, pts in by_symbol.items():
        shifted = {_round_frac(np.asarray(p, dtype=float) + delta, tol=tol) for p in frac[np.array(symbols) == s]}
        if shifted != pts:
            return False
    return True


def _infer_supercell_repeats(
    frac: np.ndarray,
    symbols: tuple[str, ...],
    nat_super: int,
) -> tuple[int, int, int]:
    max_n = max(1, nat_super)
    out = [1, 1, 1]
    for axis in range(3):
        best = 1
        for n in range(2, max_n + 1):
            if nat_super % n != 0:
                continue
            if _is_translation_symmetry(frac, symbols, axis=axis, n=n):
                best = n
        out[axis] = best
    n1, n2, n3 = int(out[0]), int(out[1]), int(out[2])
    if (nat_super % (n1 * n2 * n3)) != 0:
        raise ValueError("Failed to infer a valid supercell repeat vector from POSCAR.")
    return n1, n2, n3


def _infer_poscar_path(fc_path: Path) -> Path:
    candidates = [
        fc_path.parent / "SPOSCAR",
        fc_path.parent / "POSCAR",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Could not find POSCAR/SPOSCAR next to '{fc_path}'. "
        "Provide a POSCAR in the same folder as FORCE_CONSTANTS."
    )


def _build_ifc_from_phonopy_force_constants(
    force_constants_path: Path,
    poscar_path: Path | None = None,
    *,
    supercell: tuple[int, int, int] | None = None,
) -> IFCData:
    fc_path = force_constants_path.resolve()
    pos_path = (_infer_poscar_path(fc_path) if poscar_path is None else Path(poscar_path).resolve())

    pos = _parse_poscar(pos_path)
    frac = np.asarray(pos["atom_positions"], dtype=float)
    symbols_super = tuple(pos["atom_symbols"])
    nat_super = int(len(symbols_super))
    fc = _parse_force_constants(fc_path)
    if fc.shape[0] != nat_super:
        raise ValueError(
            f"Mismatch between POSCAR atoms ({nat_super}) and FORCE_CONSTANTS nat ({fc.shape[0]})."
        )

    if supercell is None:
        n1, n2, n3 = _infer_supercell_repeats(frac=frac, symbols=symbols_super, nat_super=nat_super)
    else:
        n1, n2, n3 = int(supercell[0]), int(supercell[1]), int(supercell[2])
        if n1 <= 0 or n2 <= 0 or n3 <= 0:
            raise ValueError("supercell repeats must be positive integers.")
    nmult = n1 * n2 * n3
    if nat_super % nmult != 0:
        raise ValueError(
            f"nat_super={nat_super} is not divisible by inferred/provided supercell size {n1}x{n2}x{n3}."
        )
    nat_prim = nat_super // nmult

    nvec = np.asarray([n1, n2, n3], dtype=float)
    q = frac * nvec[None, :]
    t = np.floor(q + 1e-8).astype(int)
    r = np.mod(q - t.astype(float), 1.0)
    t[:, 0] %= n1
    t[:, 1] %= n2
    t[:, 2] %= n3

    home_mask = (t[:, 0] == 0) & (t[:, 1] == 0) & (t[:, 2] == 0)
    home_idx = np.flatnonzero(home_mask)
    if int(home_idx.size) != nat_prim:
        raise ValueError(
            "Could not identify primitive home cell atoms from supercell mapping. "
            f"Expected {nat_prim}, found {int(home_idx.size)}."
        )

    # Deterministic basis order: symbol then fractional coordinate.
    home_sorted = sorted(
        [int(v) for v in home_idx.tolist()],
        key=lambda ii: (str(symbols_super[ii]), float(r[ii, 0]), float(r[ii, 1]), float(r[ii, 2])),
    )
    basis_frac = np.asarray([r[ii, :] for ii in home_sorted], dtype=float)
    basis_symbols = [str(symbols_super[ii]) for ii in home_sorted]

    atom_basis = np.full(nat_super, -1, dtype=int)
    for i in range(nat_super):
        sym = str(symbols_super[i])
        ri = r[i, :]
        best_j = None
        best_d = None
        for j in range(nat_prim):
            if basis_symbols[j] != sym:
                continue
            dr = np.mod(ri - basis_frac[j, :] + 0.5, 1.0) - 0.5
            d = float(np.linalg.norm(dr))
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is None or best_d is None or best_d > 5e-5:
            raise ValueError("Failed to map supercell atom to primitive basis with geometric tolerance.")
        atom_basis[i] = int(best_j)

    masses_amu = np.asarray([get_atomic_mass_amu(sym) for sym in basis_symbols], dtype=float)
    mass_scale = _amu_to_qe_mass_factor()
    masses_arr = masses_amu * mass_scale

    ndof = 3 * nat_prim
    block_sum: dict[tuple[int, int, int], np.ndarray] = {}
    block_count: dict[tuple[int, int, int], np.ndarray] = {}

    for i in range(nat_super):
        bi = int(atom_basis[i])
        ti = t[i]
        for j in range(nat_super):
            bj = int(atom_basis[j])
            tj = t[j]
            raw = np.array(
                [
                    (int(tj[0]) - int(ti[0])) % n1,
                    (int(tj[1]) - int(ti[1])) % n2,
                    (int(tj[2]) - int(ti[2])) % n3,
                ],
                dtype=int,
            )
            key = (
                _signed_fft_index(int(raw[0]), n1),
                _signed_fft_index(int(raw[1]), n2),
                _signed_fft_index(int(raw[2]), n3),
            )
            if key not in block_sum:
                block_sum[key] = np.zeros((ndof, ndof), dtype=np.complex128)
                block_count[key] = np.zeros((ndof, ndof), dtype=float)
            rs = 3 * bi
            cs = 3 * bj
            block_sum[key][rs : rs + 3, cs : cs + 3] += np.asarray(fc[i, j], dtype=np.complex128)
            block_count[key][rs : rs + 3, cs : cs + 3] += 1.0

    terms: list[IFCTerm] = []
    for key in sorted(block_sum):
        cnt = block_count[key]
        val = np.zeros_like(block_sum[key])
        np.divide(block_sum[key], cnt, out=val, where=cnt > 0.0)
        terms.append(IFCTerm(dx=int(key[0]), dy=int(key[1]), dz=int(key[2]), block=val))

    fc_scale = _phonopy_fc_to_qe_factor()
    ifc = IFCData(
        masses=masses_arr,
        dof_per_atom=3,
        terms=tuple(
            IFCTerm(dx=int(t.dx), dy=int(t.dy), dz=int(t.dz), block=np.asarray(t.block, dtype=np.complex128) * fc_scale)
            for t in terms
        ),
        units="sqrt(Ry/(me*bohr^2))",
        metadata={
            "source_format": "phonopy_force_constants",
            "force_constants_path": str(fc_path),
            "poscar_path": str(pos_path),
            "nat_supercell": nat_super,
            "nat_primitive_inferred": nat_prim,
            "nr": (n1, n2, n3),
            "supercell_repeats": (n1, n2, n3),
            "force_constant_unit_source": "eV/Angstrom^2 (phonopy default)",
            "mass_unit_source": "amu",
            "force_constant_unit_converted": "Ry/bohr^2",
            "mass_unit_converted": "2*electron_mass (QE Ry mass unit)",
            "force_constant_scale_to_qe": fc_scale,
            "mass_scale_to_qe": mass_scale,
            "omega_internal_unit": "sqrt(Ry/(2*me*bohr^2))",
        },
        lattice_vectors=np.asarray(pos["lattice_vectors"], dtype=float) / nvec[:, None],
        atom_positions=basis_frac,
        atom_symbols=tuple(basis_symbols),
        index_convention="phonopy_fft_signed",
    )
    validate_ifc_data(ifc)
    return ifc


def read_phonopy_ifc(source: Any) -> IFCData:
    """Parse phonopy IFC input into IFCData.

    Supported input:
    - ``dict`` with keys ``masses``, ``dof_per_atom``, ``terms``.
    - JSON file path containing the same structure.
    - ``FORCE_CONSTANTS`` text file path with sibling ``SPOSCAR`` or ``POSCAR``.
    - ``dict`` with keys:
      - ``force_constants_path`` (required for raw mode),
      - ``poscar_path`` (optional),
      - ``supercell`` (optional, 3-int repeats).
    """

    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file() and p.suffix.lower() == ".json":
            payload = _load_source_payload(p)
            if not ("terms" in payload and "masses" in payload):
                if "force_constants_path" in payload or "path" in payload:
                    fc_raw = payload.get("force_constants_path", payload.get("path"))
                    poscar_raw = payload.get("poscar_path", None)
                    if fc_raw is None:
                        raise ValueError("Phonopy JSON source missing force_constants_path/path.")
                    fc_path = Path(fc_raw)
                    if not fc_path.is_absolute():
                        fc_path = (p.parent / fc_path).resolve()
                    poscar_path = None
                    if poscar_raw is not None:
                        poscar_path = Path(poscar_raw)
                        if not poscar_path.is_absolute():
                            poscar_path = (p.parent / poscar_path).resolve()
                    supercell_raw = payload.get("supercell", None)
                    supercell = None
                    if supercell_raw is not None:
                        if not isinstance(supercell_raw, (list, tuple)) or len(supercell_raw) != 3:
                            raise ValueError("source['supercell'] must be a 3-element sequence when provided.")
                        supercell = (int(supercell_raw[0]), int(supercell_raw[1]), int(supercell_raw[2]))
                    return _build_ifc_from_phonopy_force_constants(
                        force_constants_path=fc_path,
                        poscar_path=poscar_path,
                        supercell=supercell,
                    )
                raise ValueError(
                    "Unsupported phonopy JSON payload. Expected normalized IFC "
                    "keys ('masses','terms') or force_constants_path/path."
                )
        else:
            return _build_ifc_from_phonopy_force_constants(force_constants_path=p)
    elif isinstance(source, dict):
        if "terms" in source and "masses" in source:
            payload = source
        elif "force_constants_path" in source or "path" in source:
            fc_path = Path(source.get("force_constants_path", source.get("path")))
            poscar_path = source.get("poscar_path", None)
            supercell_raw = source.get("supercell", None)
            supercell = None
            if supercell_raw is not None:
                if not isinstance(supercell_raw, (list, tuple)) or len(supercell_raw) != 3:
                    raise ValueError("source['supercell'] must be a 3-element sequence when provided.")
                supercell = (int(supercell_raw[0]), int(supercell_raw[1]), int(supercell_raw[2]))
            return _build_ifc_from_phonopy_force_constants(
                force_constants_path=fc_path,
                poscar_path=(None if poscar_path is None else Path(poscar_path)),
                supercell=supercell,
            )
        else:
            raise ValueError(
                "Unsupported phonopy source dict. Expected normalized IFC payload "
                "or keys 'force_constants_path'/'path'."
            )
    else:
        raise TypeError("Unsupported source type for phonopy reader.")

    terms = tuple(_parse_term(item) for item in payload["terms"])
    ifc = IFCData(
        masses=np.asarray(payload["masses"], dtype=float),
        dof_per_atom=int(payload.get("dof_per_atom", 3)),
        terms=terms,
        units=str(payload.get("units", "unknown")),
        metadata=dict(payload.get("metadata", {})),
        lattice_vectors=(
            np.asarray(payload["lattice_vectors"], dtype=float) if payload.get("lattice_vectors") is not None else None
        ),
        atom_positions=(
            np.asarray(payload["atom_positions"], dtype=float) if payload.get("atom_positions") is not None else None
        ),
        atom_symbols=tuple(payload["atom_symbols"]) if payload.get("atom_symbols") is not None else None,
        index_convention=str(payload.get("index_convention", "layer-major")),
    )
    validate_ifc_data(ifc)
    return ifc
