"""Quantum ESPRESSO q2r.x IFC adapter to the unified IFC schema."""

from __future__ import annotations

from hashlib import sha256
from itertools import product
import re
from pathlib import Path
from typing import Any

import numpy as np

from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.modeling.validators import validate_ifc_data


_SPECIES_RE = re.compile(r"^\s*(\d+)\s+'([^']+)'\s+([DdEe0-9\+\-\.]+)\s*$")
_LATGEN_COMMIT = "4387dbefb7f4795711752b953f97f95137327b94"
_LATGEN_PATH = "Modules/latgen.f90"
_LATGEN_URL = (
    "https://gitlab.com/QEF/q-e/-/blob/"
    f"{_LATGEN_COMMIT}/{_LATGEN_PATH}"
)
_SQRT2_QE = 1.414213562373
_SQRT3_QE = 1.732050807569


def _signed_fft_index(m: int, n: int) -> int:
    idx = int(m) - 1
    # Wrap FFT-grid indices into a signed range centered at zero:
    # - odd n:  [-(n//2), ..., 0, ..., +(n//2)] (e.g. n=3 -> [-1,0,1])
    # - even n: [-(n//2), ..., -1, 0, ..., +(n//2-1)] (Nyquist on negative side)
    if n > 1 and ((n % 2 == 0 and idx >= n // 2) or (n % 2 == 1 and idx > n // 2)):
        idx -= n
    return idx


def _canonical_signed_index(value: int, repeat: int) -> int:
    """Return QE's signed representative of an integer modulo ``repeat``."""

    if repeat <= 0:
        raise ValueError("FFT repeat must be positive.")
    return _signed_fft_index((int(value) % int(repeat)) + 1, int(repeat))


def _qe_latgen_lattice(
    ibrav: int,
    celldm: tuple[float, float, float, float, float, float],
    *,
    explicit_lattice: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    """Port the crystallographic-vector part of pinned QE ``latgen``.

    Primitive vectors are returned as rows in bohr.  The formulas and the
    truncated square-root constants intentionally match QE commit
    ``4387dbef``; no crystallographic reorientation is applied.
    """

    cell = np.asarray(celldm, dtype=np.float64).copy()
    if cell.shape != (6,) or not np.all(np.isfinite(cell)):
        raise ValueError("QE celldm must contain six finite values.")
    lattice = np.zeros((3, 3), dtype=np.float64)

    if int(ibrav) == 0:
        if explicit_lattice is None:
            raise ValueError("QE ibrav=0 requires three explicit lattice vectors.")
        candidate = np.asarray(explicit_lattice, dtype=np.float64)
        if candidate.shape != (3, 3) or not np.all(np.isfinite(candidate)):
            raise ValueError("QE explicit lattice must be a finite 3x3 array.")
        if np.any(np.linalg.norm(candidate, axis=1) == 0.0):
            raise ValueError("QE ibrav=0 lattice vectors must be nonzero.")
        lattice = candidate.copy()
        if cell[0] != 0.0:
            lattice *= cell[0]
        else:
            cell[0] = np.linalg.norm(lattice[0])
    if cell[0] <= 0.0:
        raise ValueError("QE celldm(1) must be positive.")
    if int(ibrav) == 0:
        return lattice, tuple(float(value) for value in cell)

    a = float(cell[0])
    if int(ibrav) == 1:
        lattice = np.diag([a, a, a])
    elif int(ibrav) == 2:
        term = a / 2.0
        lattice = np.asarray(
            [[-term, 0.0, term], [0.0, term, term], [-term, term, 0.0]]
        )
    elif abs(int(ibrav)) == 3:
        term = a / 2.0
        lattice.fill(term)
        if int(ibrav) < 0:
            lattice[0, 0] *= -1.0
            lattice[1, 1] *= -1.0
            lattice[2, 2] *= -1.0
        else:
            lattice[1, 0] *= -1.0
            lattice[2, 0] *= -1.0
            lattice[2, 1] *= -1.0
    elif int(ibrav) == 4:
        if cell[2] <= 0.0:
            raise ValueError("QE ibrav=4 requires celldm(3)>0.")
        lattice[0, 0] = a
        lattice[1, 0] = -a / 2.0
        lattice[1, 1] = a * _SQRT3_QE / 2.0
        lattice[2, 2] = a * cell[2]
    elif abs(int(ibrav)) == 5:
        if cell[3] <= -0.5 or cell[3] >= 1.0:
            raise ValueError("QE |ibrav|=5 requires -0.5<celldm(4)<1.")
        term1 = np.sqrt(1.0 + 2.0 * cell[3])
        term2 = np.sqrt(1.0 - cell[3])
        if int(ibrav) == 5:
            lattice[1, 1] = _SQRT2_QE * a * term2 / _SQRT3_QE
            lattice[1, 2] = a * term1 / _SQRT3_QE
            lattice[0, 0] = a * term2 / _SQRT2_QE
            lattice[0, 1] = -lattice[0, 0] / _SQRT3_QE
            lattice[0, 2] = lattice[1, 2]
            lattice[2, 0] = -lattice[0, 0]
            lattice[2, 1] = lattice[0, 1]
            lattice[2, 2] = lattice[1, 2]
        else:
            lattice[0, 0] = a * (term1 - 2.0 * term2) / 3.0
            lattice[0, 1] = a * (term1 + term2) / 3.0
            lattice[0, 2] = lattice[0, 1]
            lattice[1] = lattice[0, [2, 0, 1]]
            lattice[2] = lattice[0, [1, 2, 0]]
    elif int(ibrav) == 6:
        if cell[2] <= 0.0:
            raise ValueError("QE ibrav=6 requires celldm(3)>0.")
        lattice = np.diag([a, a, a * cell[2]])
    elif int(ibrav) == 7:
        if cell[2] <= 0.0:
            raise ValueError("QE ibrav=7 requires celldm(3)>0.")
        half = a / 2.0
        height = cell[2] * half
        lattice = np.asarray(
            [[half, -half, height], [half, half, height], [-half, -half, height]]
        )
    elif int(ibrav) == 8:
        if cell[1] <= 0.0 or cell[2] <= 0.0:
            raise ValueError("QE ibrav=8 requires celldm(2:3)>0.")
        lattice = np.diag([a, a * cell[1], a * cell[2]])
    elif abs(int(ibrav)) == 9:
        if cell[1] <= 0.0 or cell[2] <= 0.0:
            raise ValueError("QE |ibrav|=9 requires celldm(2:3)>0.")
        half = a / 2.0
        if int(ibrav) == 9:
            lattice[0, :2] = [half, half * cell[1]]
            lattice[1, :2] = [-half, half * cell[1]]
        else:
            lattice[0, :2] = [half, -half * cell[1]]
            lattice[1, :2] = [half, half * cell[1]]
        lattice[2, 2] = a * cell[2]
    elif int(ibrav) == 91:
        if cell[1] <= 0.0 or cell[2] <= 0.0:
            raise ValueError("QE ibrav=91 requires celldm(2:3)>0.")
        lattice[0, 0] = a
        lattice[1, 1] = a * cell[1] / 2.0
        lattice[1, 2] = -a * cell[2] / 2.0
        lattice[2, 1] = lattice[1, 1]
        lattice[2, 2] = -lattice[1, 2]
    elif int(ibrav) == 10:
        if cell[1] <= 0.0 or cell[2] <= 0.0:
            raise ValueError("QE ibrav=10 requires celldm(2:3)>0.")
        half = a / 2.0
        lattice[1, :2] = [half, half * cell[1]]
        lattice[0, [0, 2]] = [half, half * cell[2]]
        lattice[2, 1:] = [half * cell[1], half * cell[2]]
    elif int(ibrav) == 11:
        if cell[1] <= 0.0 or cell[2] <= 0.0:
            raise ValueError("QE ibrav=11 requires celldm(2:3)>0.")
        half = a / 2.0
        lattice[0] = [half, half * cell[1], half * cell[2]]
        lattice[1] = [-half, half * cell[1], half * cell[2]]
        lattice[2] = [-half, -half * cell[1], half * cell[2]]
    elif int(ibrav) == 12:
        if cell[1] <= 0.0 or cell[2] <= 0.0 or abs(cell[3]) >= 1.0:
            raise ValueError("QE ibrav=12 has invalid celldm(2:4).")
        sine = np.sqrt(1.0 - cell[3] ** 2)
        lattice[0, 0] = a
        lattice[1, :2] = [a * cell[1] * cell[3], a * cell[1] * sine]
        lattice[2, 2] = a * cell[2]
    elif int(ibrav) == -12:
        if cell[1] <= 0.0 or cell[2] <= 0.0 or abs(cell[4]) >= 1.0:
            raise ValueError("QE ibrav=-12 has invalid celldm(2:3,5).")
        sine = np.sqrt(1.0 - cell[4] ** 2)
        lattice[0, 0] = a
        lattice[1, 1] = a * cell[1]
        lattice[2, [0, 2]] = [a * cell[2] * cell[4], a * cell[2] * sine]
    elif int(ibrav) == 13:
        if cell[1] <= 0.0 or cell[2] <= 0.0 or abs(cell[3]) >= 1.0:
            raise ValueError("QE ibrav=13 has invalid celldm(2:4).")
        sine = np.sqrt(1.0 - cell[3] ** 2)
        half = a / 2.0
        lattice[0, [0, 2]] = [half, -half * cell[2]]
        lattice[1, :2] = [a * cell[1] * cell[3], a * cell[1] * sine]
        lattice[2, [0, 2]] = [half, half * cell[2]]
    elif int(ibrav) == -13:
        if cell[1] <= 0.0 or cell[2] <= 0.0 or abs(cell[4]) >= 1.0:
            raise ValueError("QE ibrav=-13 has invalid celldm(2:3,5).")
        sine = np.sqrt(1.0 - cell[4] ** 2)
        half = a / 2.0
        lattice[0, :2] = [half, -half * cell[1]]
        lattice[1, :2] = [half, half * cell[1]]
        lattice[2, [0, 2]] = [a * cell[2] * cell[4], a * cell[2] * sine]
    elif int(ibrav) == 14:
        if (
            cell[1] <= 0.0
            or cell[2] <= 0.0
            or np.any(np.abs(cell[3:]) >= 1.0)
        ):
            raise ValueError("QE ibrav=14 has invalid celldm(2:6).")
        sine_gamma = np.sqrt(1.0 - cell[5] ** 2)
        determinant = (
            1.0
            + 2.0 * cell[3] * cell[4] * cell[5]
            - cell[3] ** 2
            - cell[4] ** 2
            - cell[5] ** 2
        )
        if determinant < 0.0:
            raise ValueError("QE ibrav=14 celldm values do not define a lattice.")
        third = np.sqrt(determinant / (1.0 - cell[5] ** 2))
        lattice[0, 0] = a
        lattice[1, :2] = [a * cell[1] * cell[5], a * cell[1] * sine_gamma]
        lattice[2] = [
            a * cell[2] * cell[4],
            a * cell[2] * (cell[3] - cell[4] * cell[5]) / sine_gamma,
            a * cell[2] * third,
        ]
    else:
        raise ValueError(f"Unsupported QE Bravais lattice ibrav={ibrav}.")

    return lattice, tuple(float(value) for value in cell)


def _next_nonempty(lines: list[str], start: int) -> tuple[int, str]:
    i = start
    while i < len(lines):
        line = lines[i].strip()
        if line:
            return i, line
        i += 1
    raise ValueError("Unexpected end of file while parsing QE IFC file.")


def _qe_float(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


def _parse_three_floats(line: str) -> tuple[float, float, float] | None:
    toks = line.split()
    if len(toks) < 3:
        return None
    try:
        return _qe_float(toks[0]), _qe_float(toks[1]), _qe_float(toks[2])
    except ValueError:
        return None


def _translation_block_sha256(
    blocks: dict[tuple[int, int, int], np.ndarray],
) -> str:
    digest = sha256()
    for translation in sorted(blocks):
        digest.update(np.asarray(translation, dtype="<i8").tobytes())
        values = np.ascontiguousarray(blocks[translation], dtype="<c16")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _validate_raw_modulo_reciprocity(
    blocks: dict[tuple[int, int, int], np.ndarray],
    repeats: tuple[int, int, int],
    *,
    atol: float = 1.0e-10,
    rtol: float = 0.0,
) -> tuple[float, float]:
    """Reject q2r data that violate reciprocity on their periodic FFT grid."""

    block_scale = max(float(np.max(np.abs(block))) for block in blocks.values())
    threshold = float(atol + rtol * block_scale)
    maximum = 0.0
    worst: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
    for translation in sorted(blocks):
        partner = tuple(
            _canonical_signed_index(-translation[axis], repeats[axis])
            for axis in range(3)
        )
        if partner not in blocks:
            raise ValueError(
                "QE q2r raw grid is incomplete: translation "
                f"{translation} lacks modulo-reciprocal partner {partner}."
            )
        residual = float(
            np.max(np.abs(blocks[translation] - blocks[partner].conj().T))
        )
        if residual > maximum:
            maximum = residual
            worst = (translation, partner)
    if maximum > threshold:
        assert worst is not None
        raise ValueError(
            "QE q2r raw modulo-grid reciprocity failed before Nyquist "
            f"expansion: max residual {maximum:.6e}, allowed {threshold:.6e}, "
            f"at translations {worst[0]} and {worst[1]}."
        )
    return maximum, threshold


def _nyquist_sign_variants(
    translation: tuple[int, int, int],
    repeats: tuple[int, int, int],
) -> tuple[tuple[int, int, int], ...]:
    choices: list[tuple[int, ...]] = []
    for value, repeat in zip(translation, repeats, strict=True):
        if repeat > 1 and repeat % 2 == 0 and value == -(repeat // 2):
            choices.append((-repeat // 2, repeat // 2))
        else:
            choices.append((int(value),))
    return tuple(tuple(int(value) for value in item) for item in product(*choices))


def _fourier_blocks(
    blocks: dict[tuple[int, int, int], np.ndarray],
    qpoint: np.ndarray,
) -> np.ndarray:
    shape = next(iter(blocks.values())).shape
    output = np.zeros(shape, dtype=np.complex128)
    for translation, block in blocks.items():
        output += block * np.exp(1j * np.dot(qpoint, translation))
    return output


def _expand_q2r_nyquist(
    raw_blocks: dict[tuple[int, int, int], np.ndarray],
    repeats: tuple[int, int, int],
) -> tuple[dict[tuple[int, int, int], np.ndarray], dict[str, Any]]:
    """Expand each collapsed even-grid Nyquist image without changing data.

    A raw term touching ``k`` Nyquist axes is copied to all ``2**k`` sign
    variants with weight ``1/2**k``.  This is the unique equal-image extension
    that preserves the source FFT mesh while making ordinary real-space
    reciprocal partners explicit.  Blocks are never averaged, transposed, or
    Hermitized here.
    """

    expanded: dict[tuple[int, int, int], np.ndarray] = {}
    variant_histogram: dict[int, int] = {}
    for translation in sorted(raw_blocks):
        variants = _nyquist_sign_variants(translation, repeats)
        nvariants = len(variants)
        variant_histogram[nvariants] = variant_histogram.get(nvariants, 0) + 1
        weighted = np.asarray(raw_blocks[translation], dtype=np.complex128) / nvariants
        for variant in variants:
            if variant in expanded:
                raise ValueError(
                    "QE Nyquist expansion produced duplicate explicit "
                    f"translation {variant}; raw support is not canonical."
                )
            expanded[variant] = weighted.copy()

    mesh_max = 0.0
    mesh_scale = 0.0
    for indices in product(*(range(repeat) for repeat in repeats)):
        qpoint = 2.0 * np.pi * np.asarray(indices, dtype=np.float64) / np.asarray(
            repeats, dtype=np.float64
        )
        raw_value = _fourier_blocks(raw_blocks, qpoint)
        expanded_value = _fourier_blocks(expanded, qpoint)
        mesh_max = max(mesh_max, float(np.max(np.abs(raw_value - expanded_value))))
        mesh_scale = max(mesh_scale, float(np.max(np.abs(raw_value))))
    mesh_threshold = 1.0e-10
    if mesh_max > mesh_threshold:
        raise ValueError(
            "QE Nyquist expansion failed its source-mesh equivalence check: "
            f"max residual {mesh_max:.6e}, allowed {mesh_threshold:.6e}."
        )

    even_axes = tuple(
        axis for axis, repeat in enumerate(repeats) if repeat > 1 and repeat % 2 == 0
    )
    metadata: dict[str, Any] = {
        "schema": "negfpy.qe_q2r_nyquist_expansion.v1",
        "strategy": "all_sign_variants_equal_weight",
        "raw_support_count": len(raw_blocks),
        "expanded_support_count": len(expanded),
        "raw_support": tuple(sorted(raw_blocks)),
        "expanded_support": tuple(sorted(expanded)),
        "raw_sha256": _translation_block_sha256(raw_blocks),
        "expanded_sha256": _translation_block_sha256(expanded),
        "even_axes": even_axes,
        "variant_histogram": tuple(sorted(variant_histogram.items())),
        "source_mesh_max_abs": mesh_max,
        "source_mesh_scale": mesh_scale,
        "source_mesh_threshold": mesh_threshold,
        "source_mesh_equivalent": True,
        "complete_reciprocal_grid": True,
        "blocks_modified_by_symmetrization": False,
    }
    return expanded, metadata


def read_qe_q2r_ifc(source: Any) -> IFCData:
    """Parse QE q2r.x text IFC file (``*.fc``) into ``IFCData``."""

    path = Path(source)
    source_bytes = path.read_bytes()
    lines = source_bytes.decode("utf-8").splitlines()

    i, line = _next_nonempty(lines, 0)
    first = line.split()
    if len(first) < 9:
        raise ValueError("QE IFC header must contain ntyp, nat, ibrav, and celldm(1:6).")
    ntyp = int(first[0])
    nat = int(first[1])
    ibrav = int(first[2])
    celldm_input = tuple(_qe_float(value) for value in first[3:9])
    i += 1

    explicit_lattice = None
    if ibrav == 0:
        vectors: list[tuple[float, float, float]] = []
        for _ in range(3):
            i, row = _next_nonempty(lines, i)
            values = _parse_three_floats(row)
            if values is None:
                raise ValueError("Invalid explicit lattice-vector block in QE IFC file.")
            vectors.append(values)
            i += 1
        explicit_lattice = np.asarray(vectors, dtype=np.float64)

    lattice_vectors, celldm = _qe_latgen_lattice(
        ibrav,
        celldm_input,
        explicit_lattice=explicit_lattice,
    )
    if abs(float(np.linalg.det(lattice_vectors))) <= 1.0e-14:
        raise ValueError("QE lattice reconstructed by latgen is singular.")

    species_masses: dict[int, float] = {}
    species_labels: dict[int, str] = {}
    for _ in range(ntyp):
        i, line = _next_nonempty(lines, i)
        m = _SPECIES_RE.match(line)
        if m is None:
            raise ValueError(f"Invalid species line in QE IFC file: '{line}'")
        sid = int(m.group(1))
        species_labels[sid] = m.group(2).strip()
        species_masses[sid] = _qe_float(m.group(3))
        i += 1

    atom_species: list[int] = []
    atom_positions: list[list[float]] = []
    for _ in range(nat):
        i, line = _next_nonempty(lines, i)
        toks = line.split()
        if len(toks) < 5:
            raise ValueError(f"Invalid atomic-position line in QE IFC file: '{line}'")
        atom_species.append(int(toks[1]))
        atom_positions.append([_qe_float(toks[2]), _qe_float(toks[3]), _qe_float(toks[4])])
        i += 1

    # q2r writes tau in Cartesian coordinates measured in alat=celldm(1).
    # IFCData stores fractional row-lattice coordinates.  Do not wrap them:
    # an out-of-cell Cartesian source coordinate is part of the source gauge.
    atom_positions_alat = np.asarray(atom_positions, dtype=np.float64)
    atom_positions_cartesian_bohr = atom_positions_alat * celldm[0]
    atom_positions_fractional = atom_positions_cartesian_bohr @ np.linalg.inv(
        lattice_vectors
    )

    i, line = _next_nonempty(lines, i)
    dielectric_tensor = None
    born_charges = None
    if line in {"T", "F"}:
        has_long_range = (line == "T")
        i += 1
        if has_long_range:
            eps = []
            for _ in range(3):
                i, row = _next_nonempty(lines, i)
                eps.append([_qe_float(x) for x in row.split()[:3]])
                i += 1
            dielectric_tensor = np.asarray(eps, dtype=np.float64)
            born = np.zeros((nat, 3, 3), dtype=float)
            for ia in range(nat):
                i, _ = _next_nonempty(lines, i)  # atom index line
                i += 1
                for a in range(3):
                    i, row = _next_nonempty(lines, i)
                    born[ia, a, :] = [_qe_float(x) for x in row.split()[:3]]
                    i += 1
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
            val = _qe_float(toks[3])
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

    repeats = (nr1, nr2, nr3)
    expected_raw_support = nr1 * nr2 * nr3
    if len(blocks) != expected_raw_support:
        raise ValueError(
            "QE q2r raw translation support is incomplete: "
            f"got {len(blocks)}, expected {expected_raw_support}."
        )
    raw_reciprocal_max, raw_reciprocal_threshold = (
        _validate_raw_modulo_reciprocity(blocks, repeats)
    )
    expanded_blocks, expansion_metadata = _expand_q2r_nyquist(blocks, repeats)
    terms = tuple(
        IFCTerm(dx=key[0], dy=key[1], dz=key[2], block=value)
        for key, value in sorted(expanded_blocks.items())
    )
    masses = np.asarray([species_masses[sid] for sid in atom_species], dtype=float)
    atom_symbols = tuple(species_labels[sid] for sid in atom_species)
    ifc = IFCData(
        masses=masses,
        dof_per_atom=3,
        terms=terms,
        units="sqrt(Ry/(2*me*bohr^2))",
        metadata={
            "source_format": "qe_q2r_fc",
            "path": str(path),
            "ntyp": ntyp,
            "nat": nat,
            "ibrav": ibrav,
            "celldm_input": celldm_input,
            "celldm_effective": celldm,
            "nr": (nr1, nr2, nr3),
            "force_constant_unit": "Ry/bohr^2",
            "mass_unit": "2*electron_mass (QE Ry mass unit)",
            "omega_internal_unit": "sqrt(Ry/(2*me*bohr^2))",
            "lattice_vector_unit": "bohr",
            "lattice_vector_storage": "primitive vectors as rows",
            "lattice_convention": "QE latgen without reorientation",
            "atom_position_input_unit": "alat Cartesian",
            "atom_position_output_unit": "fractional row-lattice coordinates",
            "atom_position_wrapping_applied": False,
            "atom_positions_cartesian_alat": tuple(
                tuple(float(value) for value in row) for row in atom_positions_alat
            ),
            "dielectric_tensor": dielectric_tensor,
            "dielectric_tensor_role": "macroscopic dielectric tensor; never lattice vectors",
            "born_charges": born_charges,
            "raw_modulo_reciprocity_max_abs": raw_reciprocal_max,
            "raw_modulo_reciprocity_threshold": raw_reciprocal_threshold,
            "raw_modulo_reciprocity_valid": True,
            "nyquist_expansion": expansion_metadata,
            "reciprocal_grid_complete": True,
            "explicit_nyquist_signs": True,
            "term_generation": "qe_q2r_explicit_nyquist",
            "source_sha256": sha256(source_bytes).hexdigest(),
            "geometry_provenance": {
                "implementation": "Quantum ESPRESSO latgen",
                "commit": _LATGEN_COMMIT,
                "path": _LATGEN_PATH,
                "url": _LATGEN_URL,
            },
        },
        lattice_vectors=lattice_vectors,
        atom_positions=atom_positions_fractional,
        atom_symbols=atom_symbols,
        index_convention="qe_q2r_periodic_gauge_explicit_nyquist",
    )
    validate_ifc_data(ifc)
    return ifc
