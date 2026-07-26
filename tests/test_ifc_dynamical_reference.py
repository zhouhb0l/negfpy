from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import negfpy
from negfpy.io import read_ifc
from negfpy.modeling import (
    IFCDynamicalReference,
    basis_phase_unitary,
    compare_ifc_dynamical_matrices,
)
from negfpy.modeling.schema import IFCData, IFCTerm


def _source_hash(*payloads: bytes) -> str:
    digest = sha256()
    for payload in payloads:
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)
    return digest.hexdigest()


def _reference(
    qpoints: np.ndarray,
    matrices: np.ndarray,
    *,
    source_sha256: str | None = None,
    gauge: str = "periodic",
    dof_per_atom: int = 1,
    basis_positions: np.ndarray | None = None,
) -> IFCDynamicalReference:
    return IFCDynamicalReference(
        qpoints=qpoints,
        matrices=matrices,
        source="independent synthetic source",
        source_sha256=source_sha256 or sha256(b"independent-source").hexdigest(),
        provenance={"generator": "test-side direct formula", "seed": 417},
        gauge=gauge,
        dof_per_atom=dof_per_atom,
        basis_positions=basis_positions,
    )


def test_tests_import_the_candidate_source_tree() -> None:
    expected = Path(__file__).resolve().parents[1] / "src" / "negfpy"
    imported = Path(negfpy.__file__).resolve().parent
    assert imported == expected.resolve()


def test_direct_analytic_reference_is_canonical_and_merges_each_translation_once() -> None:
    rng = np.random.default_rng(417)
    qpoints = np.vstack(
        [
            rng.uniform(-np.pi, np.pi, size=(5, 3)),
            np.array([[0.0, 0.0, 0.0], [np.pi / 3.0, 0.0, 0.0]]),
        ]
    )
    mass = 2.0
    matrices = np.empty((qpoints.shape[0], 1, 1), dtype=np.complex128)
    for index, qpoint in enumerate(qpoints):
        # This source formula is independent of the production reconstruction.
        matrices[index, 0, 0] = (
            4.0 - np.exp(1j * qpoint[0]) - np.exp(-1j * qpoint[0])
        ) / mass

    # The two R=0 contributions must be merged once into 4.0, not selected,
    # normalized independently, or counted a second time.
    ifc = IFCData(
        masses=np.array([mass]),
        dof_per_atom=1,
        terms=(
            IFCTerm(0, 0, 0, np.array([[1.25]])),
            IFCTerm(1, 0, 0, np.array([[-1.0]])),
            IFCTerm(0, 0, 0, np.array([[2.75]])),
            IFCTerm(-1, 0, 0, np.array([[-1.0]])),
        ),
        index_convention="analytic_periodic",
    )
    reference = _reference(qpoints, matrices)
    reversed_reference = _reference(qpoints[::-1], matrices[::-1])

    result = compare_ifc_dynamical_matrices(ifc, reference)

    assert result.passed
    assert result.tolerance == 1.0e-10
    assert result.max_abs_error < 1.0e-15
    assert result.max_relative_error < 5.0e-16
    assert result.q_units == "radian"
    assert result.provenance["seed"] == 417
    assert result.input_term_count == 4
    assert result.merged_translation_count == 3
    assert result.duplicate_translations == ((0, 0, 0),)
    assert result.qpoints == tuple(sorted(result.qpoints))
    assert reference.content_sha256 == reversed_reference.content_sha256
    assert not reference.qpoints.flags.writeable
    assert not reference.matrices.flags.writeable
    with pytest.raises((FrozenInstanceError, AttributeError)):
        reference.source = "mutated"  # type: ignore[misc]
    with pytest.raises(TypeError):
        reference.provenance["seed"] = 1  # type: ignore[index]


def test_reference_strictly_rejects_bad_shapes_nonfinite_values_and_duplicate_q() -> None:
    digest = sha256(b"strict-source").hexdigest()
    matrices = np.zeros((2, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="duplicate"):
        _reference(
            np.array([[0.0, 0.0, 0.0], [-0.0, 0.0, 0.0]]),
            matrices,
            source_sha256=digest,
        )
    with pytest.raises(ValueError, match="same number"):
        _reference(
            np.array([[0.0, 0.0, 0.0]]),
            matrices,
            source_sha256=digest,
        )
    with pytest.raises(ValueError, match="finite"):
        _reference(
            np.array([[np.nan, 0.0, 0.0]]),
            np.zeros((1, 1, 1)),
            source_sha256=digest,
        )
    with pytest.raises(ValueError, match="finite"):
        _reference(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[[np.inf + 0.0j]]]),
            source_sha256=digest,
        )
    with pytest.raises(TypeError, match="real-valued"):
        _reference(
            np.array([[0.0 + 1.0j, 0.0, 0.0]]),
            np.zeros((1, 1, 1)),
            source_sha256=digest,
        )
    with pytest.raises(ValueError, match="SHA256"):
        _reference(
            np.array([[0.0, 0.0, 0.0]]),
            np.zeros((1, 1, 1)),
            source_sha256="not-a-hash",
        )


def test_comparison_does_not_silently_hermitize_nonreciprocal_input() -> None:
    block = np.array(
        [[1.0 + 0.0j, 2.0 + 0.5j], [3.0 - 0.25j, 4.0 + 0.0j]],
        dtype=np.complex128,
    )
    masses = np.array([2.0, 3.0])
    inverse = 1.0 / np.sqrt(masses)
    expected = inverse[:, None] * block * inverse[None, :]
    qpoints = np.array([[0.0, 0.0, 0.0], [0.27, -0.41, 0.19]])
    matrices = np.repeat(expected[None, :, :], qpoints.shape[0], axis=0)
    ifc = IFCData(
        masses=masses,
        dof_per_atom=1,
        terms=(IFCTerm(0, 0, 0, block),),
    )

    result = compare_ifc_dynamical_matrices(ifc, _reference(qpoints, matrices))

    assert result.passed
    assert not np.allclose(expected, expected.conj().T)


def _signed_fft_index_independent(index_zero_based: int, repeat: int) -> int:
    """Test-side FFT centering formula, independent of either IFC reader."""

    return (index_zero_based + repeat // 2) % repeat - repeat // 2


def _qe_full_grid_source() -> tuple[str, np.ndarray, tuple[int, int, int], float]:
    repeats = (3, 4, 1)
    raw = np.zeros((*repeats, 3, 3), dtype=np.float64)
    onsite = np.array(
        [[6.0, 0.2, 0.0], [0.2, 7.0, 0.1], [0.0, 0.1, 8.0]]
    )
    positive_x = np.array(
        [[-0.7, 0.05, 0.0], [-0.02, -0.5, 0.03], [0.0, -0.01, -0.4]]
    )
    positive_y = np.array(
        [[-0.3, 0.02, 0.01], [0.0, -0.2, 0.04], [-0.03, 0.0, -0.25]]
    )
    raw[0, 0, 0] = onsite
    raw[1, 0, 0] = positive_x
    raw[2, 0, 0] = positive_x.T
    raw[0, 1, 0] = positive_y
    raw[0, 3, 0] = positive_y.T
    mass = 5.0

    lines = [
        "1 1 1 1.0 0.0 0.0 0.0 0.0 0.0",
        f"1 'X ' {mass:.16g}",
        "1 1 0.0 0.0 0.0",
        f"{repeats[0]} {repeats[1]} {repeats[2]}",
    ]
    for alpha in range(3):
        for beta in range(3):
            lines.append(f"{alpha + 1} {beta + 1} 1 1")
            for ix, iy, iz in product(
                range(repeats[0]),
                range(repeats[1]),
                range(repeats[2]),
            ):
                lines.append(
                    f"{ix + 1} {iy + 1} {iz + 1} "
                    f"{raw[ix, iy, iz, alpha, beta]:.17g}"
                )
    return "\n".join(lines) + "\n", raw, repeats, mass


def _qe_source_matrices(
    raw: np.ndarray,
    repeats: tuple[int, int, int],
    mass: float,
    qpoints: np.ndarray,
) -> np.ndarray:
    matrices = np.zeros((qpoints.shape[0], 3, 3), dtype=np.complex128)
    for q_index, qpoint in enumerate(qpoints):
        for ix, iy, iz in product(
            range(repeats[0]),
            range(repeats[1]),
            range(repeats[2]),
        ):
            translation = np.array(
                [
                    _signed_fft_index_independent(ix, repeats[0]),
                    _signed_fft_index_independent(iy, repeats[1]),
                    _signed_fft_index_independent(iz, repeats[2]),
                ]
            )
            matrices[q_index] += (
                raw[ix, iy, iz]
                * np.exp(1j * np.dot(qpoint, translation))
                / mass
            )
    return matrices


def test_qe_full_grid_source_oracle_covers_odd_and_even_fft_indices(
    tmp_path: Path,
) -> None:
    source_text, raw, repeats, mass = _qe_full_grid_source()
    source_path = tmp_path / "odd-even.fc"
    source_path.write_text(source_text, encoding="utf-8")
    ifc = read_ifc(source_path, reader="qe")

    rng = np.random.default_rng(90210)
    qpoints = np.vstack(
        [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [2.0 * np.pi / 3.0, np.pi / 2.0, 0.0],
                    [-2.0 * np.pi / 3.0, -np.pi, 0.0],
                ]
            ),
            rng.uniform(-np.pi, np.pi, size=(7, 3)),
        ]
    )
    source_matrices = _qe_source_matrices(raw, repeats, mass, qpoints)
    reference = IFCDynamicalReference(
        qpoints=qpoints,
        matrices=source_matrices,
        source="QE q2r synthetic 3x4x1 full grid",
        source_sha256=_source_hash(source_text.encode("utf-8")),
        provenance={
            "format": "qe_q2r_fc",
            "fft_repeats": repeats,
            "generic_q_seed": 90210,
            "reference_route": "raw full-grid direct Fourier sum",
        },
        gauge="periodic",
        dof_per_atom=3,
    )

    result = compare_ifc_dynamical_matrices(ifc, reference)

    assert sorted({term.dx for term in ifc.terms}) == [-1, 0, 1]
    assert sorted({term.dy for term in ifc.terms}) == [-2, -1, 0, 1]
    assert sorted({term.dz for term in ifc.terms}) == [0]
    assert result.passed
    assert result.max_abs_error < 2.0e-15
    assert result.max_relative_error < 2.0e-15
    assert len(result.source_sha256) == 64
    assert len(result.reference_sha256) == 64
    assert len(result.ifc_sha256) == 64


def _force_constants_text(force_constants: np.ndarray) -> str:
    lines = [f"{force_constants.shape[0]} {force_constants.shape[1]}"]
    for atom_i in range(force_constants.shape[0]):
        for atom_j in range(force_constants.shape[1]):
            lines.append(f"{atom_i + 1} {atom_j + 1}")
            for row in force_constants[atom_i, atom_j]:
                lines.append(" ".join(f"{value:.17g}" for value in row))
    return "\n".join(lines) + "\n"


def _shortest_vectors_independent(
    basis_i: np.ndarray,
    basis_j: np.ndarray,
    lattice: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Brute-force raw-source shortest vectors, independent of the reader."""

    candidates: list[tuple[float, tuple[int, int, int]]] = []
    for image in product(range(-2, 3), repeat=3):
        translation = np.asarray(image, dtype=np.int64)
        displacement = translation + basis_j - basis_i
        cartesian = displacement @ lattice
        candidates.append((float(np.dot(cartesian, cartesian)), image))
    minimum = min(value for value, _ in candidates)
    return tuple(
        np.asarray(image, dtype=np.float64)
        for value, image in candidates
        if abs(value - minimum) <= 1.0e-12
    )


def _phonopy_source_atom_phase_matrices(
    force_constants: np.ndarray,
    basis_positions: np.ndarray,
    lattice: np.ndarray,
    qpoints: np.ndarray,
) -> np.ndarray:
    # Independent CODATA-2018 conversion written directly from source units.
    ev_j = 1.602176634e-19
    rydberg_j = 2.1798723611035e-18
    bohr_m = 5.29177210903e-11
    electron_mass_kg = 9.1093837015e-31
    amu_kg = 1.66053906660e-27
    fc_to_qe = (ev_j / rydberg_j) / ((1.0e-10 / bohr_m) ** 2)
    silicon_mass_qe = 28.085 * amu_kg / (2.0 * electron_mass_kg)
    masses = np.repeat(np.array([silicon_mass_qe, silicon_mass_qe]), 3)
    inverse_sqrt_mass = 1.0 / np.sqrt(masses)

    output = np.zeros((qpoints.shape[0], 6, 6), dtype=np.complex128)
    for q_index, qpoint in enumerate(qpoints):
        phi = np.zeros((6, 6), dtype=np.complex128)
        for atom_i in range(2):
            for atom_j in range(2):
                vectors = _shortest_vectors_independent(
                    basis_positions[atom_i],
                    basis_positions[atom_j],
                    lattice,
                )
                for translation in vectors:
                    displacement = (
                        translation
                        + basis_positions[atom_j]
                        - basis_positions[atom_i]
                    )
                    phi[
                        3 * atom_i : 3 * atom_i + 3,
                        3 * atom_j : 3 * atom_j + 3,
                    ] += (
                        force_constants[atom_i, atom_j]
                        * fc_to_qe
                        * np.exp(1j * np.dot(qpoint, displacement))
                        / len(vectors)
                    )
        output[q_index] = (
            inverse_sqrt_mass[:, None] * phi * inverse_sqrt_mass[None, :]
        )
    return output


def test_phonopy_shortest_vector_source_oracle_handles_repeat_one_and_atom_gauge(
    tmp_path: Path,
) -> None:
    lattice = np.array(
        [
            [3.0, 0.7, 0.2],
            [0.0, 2.5, 0.4],
            [0.0, 0.0, 4.0],
        ]
    )
    basis_positions = np.array(
        [
            [0.10, 0.05, 0.00],
            [0.90, 0.05, 0.00],
        ]
    )
    stiffness = np.array(
        [[2.0, 0.1, 0.0], [0.1, 1.5, 0.2], [0.0, 0.2, 1.0]]
    )
    force_constants = np.empty((2, 2, 3, 3), dtype=np.float64)
    force_constants[0, 0] = stiffness
    force_constants[1, 1] = stiffness
    force_constants[0, 1] = -stiffness
    force_constants[1, 0] = -stiffness

    poscar_text = "\n".join(
        [
            "Skewed repeat-one shortest-vector fixture",
            "1.0",
            *[" ".join(f"{value:.17g}" for value in row) for row in lattice],
            "Si",
            "2",
            "Direct",
            *[
                " ".join(f"{value:.17g}" for value in row)
                for row in basis_positions
            ],
        ]
    ) + "\n"
    force_text = _force_constants_text(force_constants)
    poscar_path = tmp_path / "SPOSCAR"
    force_path = tmp_path / "FORCE_CONSTANTS"
    poscar_path.write_text(poscar_text, encoding="utf-8")
    force_path.write_text(force_text, encoding="utf-8")
    ifc = read_ifc(
        {
            "force_constants_path": force_path,
            "poscar_path": poscar_path,
            "supercell": (1, 1, 1),
        },
        reader="phonopy",
    )

    rng = np.random.default_rng(7441)
    qpoints = np.vstack(
        [
            np.array([[0.0, 0.0, 0.0], [0.73, -0.41, 0.29]]),
            rng.uniform(-np.pi, np.pi, size=(6, 3)),
        ]
    )
    source_matrices = _phonopy_source_atom_phase_matrices(
        force_constants,
        basis_positions,
        lattice,
        qpoints,
    )
    reference = IFCDynamicalReference(
        qpoints=qpoints,
        matrices=source_matrices,
        source="phonopy FORCE_CONSTANTS shortest-vector fixture",
        source_sha256=_source_hash(
            force_text.encode("utf-8"),
            poscar_text.encode("utf-8"),
        ),
        provenance={
            "format": "phonopy_FORCE_CONSTANTS",
            "supercell": [1, 1, 1],
            "generic_q_seed": 7441,
            "reference_route": "raw FC pair and Cartesian shortest-vector sum",
        },
        gauge="atom_phase",
        dof_per_atom=3,
        basis_positions=basis_positions,
    )

    result = compare_ifc_dynamical_matrices(ifc, reference)
    unitary = basis_phase_unitary(
        np.array([0.73, -0.41, 0.29]),
        basis_positions,
        3,
    )

    translations = {(term.dx, term.dy, term.dz) for term in ifc.terms}
    assert ifc.metadata["term_generation"] == "phonopy_shortest_vectors"
    assert (-1, 0, 0) in translations
    assert (1, 0, 0) in translations
    np.testing.assert_allclose(ifc.lattice_vectors, lattice, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        unitary.conj().T @ unitary,
        np.eye(6),
        rtol=0.0,
        atol=3.0e-16,
    )
    assert result.passed
    assert result.max_abs_error < 3.0e-15
    assert result.max_relative_error < 3.0e-15

    # The same atom-phase matrices deliberately mislabeled as periodic must
    # fail.  This guards against calibrating away a reader gauge mismatch.
    wrong_gauge = IFCDynamicalReference(
        qpoints=qpoints,
        matrices=source_matrices,
        source=reference.source,
        source_sha256=reference.source_sha256,
        provenance={"deliberate_test": "wrong gauge"},
        gauge="periodic",
        dof_per_atom=3,
    )
    wrong_result = compare_ifc_dynamical_matrices(ifc, wrong_gauge)
    assert not wrong_result.passed
    assert wrong_result.max_abs_error > wrong_result.allowed_max_abs_error

    corrupted_matrices = np.array(source_matrices, copy=True)
    corrupted_matrices[0, 0, 0] += 1.0e-5
    corrupted = IFCDynamicalReference(
        qpoints=qpoints,
        matrices=corrupted_matrices,
        source=reference.source,
        source_sha256=reference.source_sha256,
        provenance={"deliberate_test": "matrix corruption"},
        gauge="atom_phase",
        dof_per_atom=3,
        basis_positions=basis_positions,
    )
    corrupted_result = compare_ifc_dynamical_matrices(ifc, corrupted)
    assert not corrupted_result.passed
    assert corrupted_result.max_abs_error >= 9.9e-6

    shifted_basis = np.array(basis_positions, copy=True)
    shifted_basis[1, 0] += 0.01
    mismatched_basis = IFCDynamicalReference(
        qpoints=qpoints,
        matrices=source_matrices,
        source=reference.source,
        source_sha256=reference.source_sha256,
        provenance={"deliberate_test": "basis mismatch"},
        gauge="atom_phase",
        dof_per_atom=3,
        basis_positions=shifted_basis,
    )
    with pytest.raises(ValueError, match="atom_positions"):
        compare_ifc_dynamical_matrices(ifc, mismatched_basis)
