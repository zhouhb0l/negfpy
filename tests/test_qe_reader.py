from pathlib import Path
from itertools import product

import numpy as np
import pytest

from negfpy.io import list_readers, read_ifc
from negfpy.io.readers.qe import _signed_fft_index
from negfpy.modeling import qe_omega_to_cm1, qe_omega_to_thz
from negfpy.modeling.builders import build_fc_terms
from negfpy.modeling.schema import BuildConfig
from negfpy.modeling.validators import audit_ifc_data


def _q2r_fixture_text(
    *,
    ibrav: int,
    celldm: tuple[float, float, float, float, float, float],
    atom_position_alat: tuple[float, float, float],
    repeats: tuple[int, int, int],
    raw_blocks: np.ndarray,
    explicit_lattice: np.ndarray | None = None,
    dielectric: np.ndarray | None = None,
) -> str:
    lines = [
        "1 1 " + str(ibrav) + " " + " ".join(f"{value:.17g}" for value in celldm)
    ]
    if ibrav == 0:
        assert explicit_lattice is not None
        lines.extend(
            " ".join(f"{value:.17g}" for value in row) for row in explicit_lattice
        )
    lines.extend(
        [
            "1 'X ' 10.0",
            "1 1 " + " ".join(f"{value:.17g}" for value in atom_position_alat),
        ]
    )
    if dielectric is None:
        lines.append("F")
    else:
        lines.append("T")
        lines.extend(
            " ".join(f"{value:.17g}" for value in row) for row in dielectric
        )
        lines.append("1")
        lines.extend(("1 0 0", "0 1 0", "0 0 1"))
    lines.append(" ".join(str(value) for value in repeats))
    for alpha in range(3):
        for beta in range(3):
            lines.append(f"{alpha + 1} {beta + 1} 1 1")
            for ix, iy, iz in product(*(range(value) for value in repeats)):
                lines.append(
                    f"{ix + 1} {iy + 1} {iz + 1} "
                    f"{raw_blocks[ix, iy, iz, alpha, beta]:.17g}"
                )
    return "\n".join(lines) + "\n"


def test_qe_reader_registered() -> None:
    assert "qe" in list_readers()


def test_qe_reader_parses_minimal_fc(tmp_path: Path) -> None:
    fc_text = "\n".join(
        [
            " 1 1 1 1.0 0.0 0.0 0.0 0.0 0.0",
            " 1 'X ' 10.0",
            " 1 1 0.0 0.0 0.0",
            " 1 1 1",
            " 1 1 1 1",
            " 1 1 1 1.0",
            " 1 2 1 1",
            " 1 1 1 0.0",
            " 1 3 1 1",
            " 1 1 1 0.0",
            " 2 1 1 1",
            " 1 1 1 0.0",
            " 2 2 1 1",
            " 1 1 1 2.0",
            " 2 3 1 1",
            " 1 1 1 0.0",
            " 3 1 1 1",
            " 1 1 1 0.0",
            " 3 2 1 1",
            " 1 1 1 0.0",
            " 3 3 1 1",
            " 1 1 1 3.0",
        ]
    )
    fc_path = tmp_path / "mini.fc"
    fc_path.write_text(fc_text, encoding="utf-8")

    ifc = read_ifc(fc_path, reader="qe")
    assert ifc.dof_per_atom == 3
    assert ifc.masses.shape == (1,)
    assert len(ifc.terms) == 1
    assert ifc.units == "sqrt(Ry/(2*me*bohr^2))"
    assert ifc.units == ifc.metadata["omega_internal_unit"]
    term = ifc.terms[0]
    assert (term.dx, term.dy, term.dz) == (0, 0, 0)
    assert term.block.shape == (3, 3)
    assert np.allclose(np.diag(term.block).real, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(ifc.lattice_vectors, np.eye(3), rtol=0.0, atol=0.0)
    assert ifc.metadata["celldm_input"] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert ifc.metadata["dielectric_tensor"] is None


_GEOMETRY_CASES = (
    (
        0,
        (2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        np.array([[1.0, 0.2, 0.0], [0.0, 1.5, 0.1], [0.0, 0.0, 0.8]]),
        np.array([[2.0, 0.4, 0.0], [0.0, 3.0, 0.2], [0.0, 0.0, 1.6]]),
    ),
    (
        2,
        (4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        None,
        np.array([[-2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [-2.0, 2.0, 0.0]]),
    ),
    (
        4,
        (3.0, 0.0, 1.7, 0.0, 0.0, 0.0),
        None,
        np.array(
            [[3.0, 0.0, 0.0], [-1.5, 3.0 * 1.732050807569 / 2.0, 0.0], [0.0, 0.0, 5.1]]
        ),
    ),
    (
        8,
        (2.0, 1.4, 2.1, 0.0, 0.0, 0.0),
        None,
        np.diag([2.0, 2.8, 4.2]),
    ),
    (
        14,
        (2.0, 1.3, 1.7, 0.2, -0.1, 0.3),
        None,
        np.array(
            [
                [2.0, 0.0, 0.0],
                [2.0 * 1.3 * 0.3, 2.0 * 1.3 * np.sqrt(1.0 - 0.3**2), 0.0],
                [
                    2.0 * 1.7 * -0.1,
                    2.0 * 1.7 * (0.2 - (-0.1 * 0.3)) / np.sqrt(1.0 - 0.3**2),
                    2.0
                    * 1.7
                    * np.sqrt(
                        (1.0 + 2.0 * 0.2 * -0.1 * 0.3 - 0.2**2 - (-0.1) ** 2 - 0.3**2)
                        / (1.0 - 0.3**2)
                    ),
                ],
            ]
        ),
    ),
)


@pytest.mark.parametrize("ibrav,celldm,explicit_lattice,expected_lattice", _GEOMETRY_CASES)
def test_qe_latgen_geometry_and_cartesian_alat_positions(
    tmp_path: Path,
    ibrav: int,
    celldm: tuple[float, float, float, float, float, float],
    explicit_lattice: np.ndarray | None,
    expected_lattice: np.ndarray,
) -> None:
    raw = np.zeros((1, 1, 1, 3, 3))
    raw[0, 0, 0] = np.diag([2.0, 3.0, 4.0])
    source_position = np.array([1.2, -0.1, 0.5])
    source = _q2r_fixture_text(
        ibrav=ibrav,
        celldm=celldm,
        atom_position_alat=tuple(source_position),
        repeats=(1, 1, 1),
        raw_blocks=raw,
        explicit_lattice=explicit_lattice,
    )
    path = tmp_path / f"ibrav-{ibrav}.fc"
    path.write_text(source, encoding="utf-8")

    ifc = read_ifc(path, reader="qe")

    np.testing.assert_allclose(ifc.lattice_vectors, expected_lattice, rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        ifc.atom_positions,
        ((source_position * celldm[0]) @ np.linalg.inv(expected_lattice))[None, :],
        rtol=0.0,
        atol=2.0e-13,
    )
    assert ifc.metadata["atom_position_input_unit"] == "alat Cartesian"
    assert ifc.metadata["atom_position_wrapping_applied"] is False
    assert ifc.metadata["geometry_provenance"]["commit"].startswith("4387dbef")


def test_qe_dielectric_tensor_is_not_used_as_lattice(tmp_path: Path) -> None:
    raw = np.zeros((1, 1, 1, 3, 3))
    raw[0, 0, 0] = np.eye(3)
    dielectric = np.diag([11.0, 12.0, 13.0])
    source = _q2r_fixture_text(
        ibrav=2,
        celldm=(4.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        atom_position_alat=(0.0, 0.0, 0.0),
        repeats=(1, 1, 1),
        raw_blocks=raw,
        dielectric=dielectric,
    )
    path = tmp_path / "dielectric.fc"
    path.write_text(source, encoding="utf-8")

    ifc = read_ifc(path, reader="qe")

    np.testing.assert_allclose(
        ifc.lattice_vectors,
        np.array([[-2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [-2.0, 2.0, 0.0]]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(ifc.metadata["dielectric_tensor"], dielectric)
    np.testing.assert_allclose(ifc.metadata["born_charges"], np.eye(3)[None, :, :])


def test_qe_raw_modulo_reciprocity_violation_fails(tmp_path: Path) -> None:
    raw = np.zeros((2, 1, 1, 3, 3))
    raw[0, 0, 0] = np.eye(3)
    raw[1, 0, 0] = np.array([[1.0, 0.3, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    source = _q2r_fixture_text(
        ibrav=1,
        celldm=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        atom_position_alat=(0.0, 0.0, 0.0),
        repeats=(2, 1, 1),
        raw_blocks=raw,
    )
    path = tmp_path / "nonreciprocal.fc"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(ValueError, match="raw modulo-grid reciprocity failed"):
        read_ifc(path, reader="qe")


def test_explicit_transport_nyquist_halves_are_not_split_twice(tmp_path: Path) -> None:
    stiffness = np.diag([2.0, 3.0, 4.0])
    raw = np.zeros((2, 1, 1, 3, 3))
    raw[0, 0, 0] = stiffness
    raw[1, 0, 0] = -stiffness
    source = _q2r_fixture_text(
        ibrav=1,
        celldm=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        atom_position_alat=(0.0, 0.0, 0.0),
        repeats=(2, 1, 1),
        raw_blocks=raw,
    )
    path = tmp_path / "explicit-nyquist.fc"
    path.write_text(source, encoding="utf-8")
    ifc = read_ifc(path, reader="qe")

    fc00, fc01, fc10 = build_fc_terms(
        ifc,
        BuildConfig(
            principal_layer_size=1,
            nyquist_split_half=True,
            enforce_hermitian_fc00=False,
        ),
    )

    np.testing.assert_allclose(fc00[(0, 0)], stiffness, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(fc01[(0, 0)], -0.5 * stiffness, rtol=0.0, atol=0.0)
    assert fc10 is not None
    np.testing.assert_allclose(fc10[(0, 0)], -0.5 * stiffness, rtol=0.0, atol=0.0)
    report = audit_ifc_data(ifc, rotational_atol=10.0)
    assert report.metrics["explicit_nyquist_grid_declared"] == 1.0
    assert report.metrics["explicit_nyquist_grid_complete"] == 1.0


def test_qe_reader_parses_si444_if_present() -> None:
    candidates = (
        Path("si444.fc"),
        Path("../studies/materials/silicon_bulk_2026q1/inputs/ifc/si444.fc"),
    )
    fc_path = next((value for value in candidates if value.exists()), candidates[0])
    if not fc_path.exists():
        return
    ifc = read_ifc(fc_path, reader="qe")
    assert ifc.dof_per_atom == 3
    assert ifc.masses.shape == (2,)
    assert len(ifc.terms) > 1
    max_abs_dx = max(abs(t.dx) for t in ifc.terms)
    max_abs_dy = max(abs(t.dy) for t in ifc.terms)
    max_abs_dz = max(abs(t.dz) for t in ifc.terms)
    assert max_abs_dx == 2
    assert max_abs_dy == 2
    assert max_abs_dz == 2
    assert len(ifc.terms) == 125
    expected_lattice = np.array(
        [[-5.1, 0.0, 5.1], [0.0, 5.1, 5.1], [-5.1, 5.1, 0.0]]
    )
    np.testing.assert_allclose(ifc.lattice_vectors, expected_lattice, rtol=0.0, atol=0.0)
    raw_positions = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    np.testing.assert_allclose(
        ifc.atom_positions,
        raw_positions * 10.2 @ np.linalg.inv(expected_lattice),
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_allclose(
        ifc.metadata["dielectric_tensor"],
        np.eye(3) * 13.734417186872,
        rtol=0.0,
        atol=0.0,
    )
    expansion = ifc.metadata["nyquist_expansion"]
    assert expansion["raw_support_count"] == 64
    assert expansion["expanded_support_count"] == 125
    assert expansion["source_mesh_max_abs"] < 1.0e-12

    blocks = {
        (term.dx, term.dy, term.dz): np.asarray(term.block) for term in ifc.terms
    }
    inverse_mass = np.repeat(1.0 / np.sqrt(ifc.masses), 3)
    for qpoint in (np.array([0.31, -0.27, 0.19]), np.array([1.13, 0.73, -0.91])):
        matrix = sum(
            block * np.exp(1j * np.dot(qpoint, translation))
            for translation, block in blocks.items()
        )
        matrix = inverse_mass[:, None] * matrix * inverse_mass[None, :]
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=0.0, atol=2.0e-17)


def test_qe_frequency_conversion_factors() -> None:
    thz = float(qe_omega_to_thz(1.0))
    cm1 = float(qe_omega_to_cm1(1.0))
    assert np.isclose(thz, 3289.8419602568943, rtol=0.0, atol=1e-9)
    assert np.isclose(cm1, 109737.31568180058, rtol=0.0, atol=1e-6)


def test_signed_fft_index_uses_symmetric_center_for_odd_n() -> None:
    mapped = [_signed_fft_index(m, 3) for m in (1, 2, 3)]
    assert mapped == [0, 1, -1]
    assert sorted(mapped) == [-1, 0, 1]


def test_signed_fft_index_keeps_negative_nyquist_for_even_n() -> None:
    mapped = [_signed_fft_index(m, 4) for m in (1, 2, 3, 4)]
    assert mapped == [0, 1, -2, -1]
    assert sorted(mapped) == [-2, -1, 0, 1]
