from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic.bulk_harmonic import (
    DEFAULT_FREQUENCY_FACTOR_TO_THZ,
    PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ,
    BulkGroupVelocities,
    compute_bulk_group_velocities,
    compute_bulk_phonons,
    dynamical_matrix_derivatives_from_fc2,
    dynamical_matrix_from_fc2,
)
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.bulk.fc2_audit import audit_mapped_fc2


def _chain_mapping() -> CrystalMapping:
    primitive = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0]],
        masses=[2.0],
        symbols=["X"],
    )
    supercell = CrystalCell(
        lattice=np.diag([3.0, 1.0, 1.0]),
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [2.0 / 3.0, 0.0, 0.0],
        ],
        masses=[2.0, 2.0, 2.0],
        symbols=["X", "X", "X"],
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.diag([3, 1, 1]),
        primitive=primitive,
        supercell=supercell,
        p2s=[1],
        s2p=[1, 1, 1],
        p2p=[-1, 0, -1],
        shortest_vectors=[
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        multiplicity=[
            [[1, 0]],
            [[1, 1]],
            [[1, 2]],
        ],
        atomic_permutations=[[0, 1, 2]],
        purpose="fc2",
    )


def _compact_fc2() -> FC2Data:
    values = np.zeros((1, 3, 3, 3))
    values[0, 0] = -2.0 * np.eye(3)
    values[0, 1] = 4.0 * np.eye(3)
    values[0, 2] = -2.0 * np.eye(3)
    return FC2Data(
        values=values,
        masses=[2.0, 2.0, 2.0],
        compact=True,
        p2s_map=[1],
        s2p_map=[1, 1, 1],
    )


def _skew_chain_mapping() -> CrystalMapping:
    lattice = np.array(
        [
            [2.0, 0.5, 0.0],
            [0.0, 1.5, 0.2],
            [0.0, 0.0, 1.0],
        ]
    )
    primitive = CrystalCell(
        lattice=lattice,
        scaled_positions=[[0.0, 0.0, 0.0]],
        masses=[2.0],
        symbols=["X"],
    )
    supercell_matrix = np.diag([3, 1, 1])
    supercell = CrystalCell(
        lattice=supercell_matrix @ lattice,
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [2.0 / 3.0, 0.0, 0.0],
        ],
        masses=[2.0, 2.0, 2.0],
        symbols=["X", "X", "X"],
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=supercell_matrix,
        primitive=primitive,
        supercell=supercell,
        p2s=[1],
        s2p=[1, 1, 1],
        p2p=[-1, 0, -1],
        shortest_vectors=[
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        multiplicity=[[[1, 0]], [[1, 1]], [[1, 2]]],
        atomic_permutations=[[0, 1, 2]],
        purpose="fc2",
    )


def _full_fc2() -> FC2Data:
    values = np.zeros((3, 3, 3, 3))
    values[1] = _compact_fc2().values[0]
    return FC2Data(
        values=values,
        masses=[2.0, 2.0, 2.0],
        compact=False,
        p2s_map=[1],
        s2p_map=[1, 1, 1],
    )


def _translation_complete_chain_mapping() -> CrystalMapping:
    mapping = _chain_mapping()
    return CrystalMapping(
        primitive_matrix=mapping.primitive_matrix,
        supercell_matrix=mapping.supercell_matrix,
        primitive=mapping.primitive,
        supercell=mapping.supercell,
        p2s=mapping.p2s,
        s2p=mapping.s2p,
        p2p=mapping.p2p,
        shortest_vectors=mapping.shortest_vectors,
        multiplicity=mapping.multiplicity,
        atomic_permutations=mapping.atomic_permutations,
        translation_permutations=[
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        ],
        purpose="fc2",
    )


def _translation_complete_full_fc2() -> FC2Data:
    representative = _compact_fc2().values[0]
    values = np.empty((3, 3, 3, 3))
    values[1] = representative
    values[0] = representative[[1, 2, 0]]
    values[2] = representative[[2, 0, 1]]
    return FC2Data(
        values=values,
        masses=[2.0, 2.0, 2.0],
        compact=False,
        p2s_map=[1],
        s2p_map=[1, 1, 1],
    )


def _asymmetric_two_species_mapping() -> CrystalMapping:
    """Return a deterministic two-species shortest-vector fixture.

    The flattened shortest vectors are ordered by
    ``(supercell_atom, primitive_atom)``.  At q=(1/4, 0, 0), their phases
    are therefore exactly 1, +i, or -i.
    """

    primitive = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ],
        masses=[2.0, 8.0],
        symbols=["A", "B"],
    )
    supercell = CrystalCell(
        lattice=np.diag([2.0, 1.0, 1.0]),
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.75, 0.0, 0.0],
        ],
        masses=[2.0, 2.0, 8.0, 8.0],
        symbols=["A", "A", "B", "B"],
    )
    multiplicity = np.empty((4, 2, 2), dtype=np.int64)
    multiplicity[..., 0] = 1
    multiplicity[..., 1] = np.arange(8, dtype=np.int64).reshape(4, 2)
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.diag([2, 1, 1]),
        primitive=primitive,
        supercell=supercell,
        p2s=[0, 2],
        s2p=[0, 0, 2, 2],
        p2p=[0, -1, 1, -1],
        shortest_vectors=[
            [0.0, 0.0, 0.0],  # supercell 0, primitive 0: 1
            [1.0, 0.0, 0.0],  # supercell 0, primitive 1: +i
            [1.0, 0.0, 0.0],  # supercell 1, primitive 0: +i
            [0.0, 0.0, 0.0],  # supercell 1, primitive 1: 1
            [0.0, 0.0, 0.0],  # supercell 2, primitive 0: 1
            [-1.0, 0.0, 0.0],  # supercell 2, primitive 1: -i
            [-1.0, 0.0, 0.0],  # supercell 3, primitive 0: -i
            [0.0, 0.0, 0.0],  # supercell 3, primitive 1: 1
        ],
        multiplicity=multiplicity,
        atomic_permutations=[[0, 1, 2, 3]],
        purpose="fc2",
    )


def _asymmetric_compact_fc2() -> FC2Data:
    """FC2 whose q=1/4 dynamical matrix has a closed-form complex value."""

    onsite_a = np.diag([5.0, 9.0, 15.0])
    onsite_b = np.diag([7.0, 12.0, 20.0])
    upper_coupling = np.diag([1.0, 2.0, 3.0])
    lower_coupling = np.diag([0.5, 1.5, 2.5])

    values = np.zeros((2, 4, 3, 3))
    # sqrt(M_A M_A)=2 and this image has phase 1.
    values[0, 0] = 2.0 * onsite_a
    # sqrt(M_A M_B)=4 and this image has phase -i.
    values[0, 3] = -4.0 * upper_coupling
    # sqrt(M_B M_A)=4 and this image has phase +i.
    values[1, 0] = -4.0 * lower_coupling
    # sqrt(M_B M_B)=8 and this image has phase 1.
    values[1, 3] = 8.0 * onsite_b
    return FC2Data(
        values=values,
        masses=[2.0, 2.0, 8.0, 8.0],
        compact=True,
        p2s_map=[0, 2],
        s2p_map=[0, 0, 2, 2],
    )


def _asymmetric_full_fc2() -> FC2Data:
    compact = _asymmetric_compact_fc2()
    values = np.zeros((4, 4, 3, 3))
    values[0] = compact.values[0]
    values[2] = compact.values[1]
    # Full FC2 reconstruction must use only representative rows 0 and 2.
    values[1] = 123.0
    values[3] = -456.0
    return FC2Data(
        values=values,
        masses=compact.masses,
        compact=False,
        p2s_map=compact.p2s_map,
        s2p_map=compact.s2p_map,
    )


def _asymmetric_expected_raw(*, q_sign: float = 1.0) -> np.ndarray:
    """Analytical q=(+/-1/4,0,0) raw matrix for the fixture."""

    matrix = np.zeros((6, 6), dtype=np.complex128)
    matrix[:3, :3] = np.diag([5.0, 9.0, 15.0])
    matrix[3:, 3:] = np.diag([7.0, 12.0, 20.0])
    matrix[:3, 3:] = q_sign * 1j * np.diag([1.0, 2.0, 3.0])
    matrix[3:, :3] = -q_sign * 1j * np.diag([0.5, 1.5, 2.5])
    return matrix


def _asymmetric_analytical_eigenprojectors() -> tuple[np.ndarray, np.ndarray]:
    """Return closed-form eigenvalues and rank-one projectors.

    Hermitianization replaces each pair of unequal directed couplings by its
    arithmetic mean.  Each Cartesian polarization is then an independent
    two-level Hermitian problem.
    """

    onsite_a = np.array([5.0, 9.0, 15.0])
    onsite_b = np.array([7.0, 12.0, 20.0])
    coupling = np.array([0.75, 1.75, 2.75])
    entries: list[tuple[float, np.ndarray]] = []
    for axis, (a, d, g) in enumerate(
        zip(onsite_a, onsite_b, coupling, strict=True)
    ):
        discriminant = np.sqrt((a - d) ** 2 + 4.0 * g**2)
        for branch in (-1.0, 1.0):
            eigenvalue = (a + d + branch * discriminant) / 2.0
            local = np.array([1j * g, eigenvalue - a])
            local /= np.linalg.norm(local)
            vector = np.zeros(6, dtype=np.complex128)
            vector[axis] = local[0]
            vector[axis + 3] = local[1]
            projector = np.outer(vector, vector.conj())
            entries.append((float(eigenvalue), projector))
    entries.sort(key=lambda item: item[0])
    return (
        np.array([item[0] for item in entries]),
        np.array([item[1] for item in entries]),
    )


@pytest.mark.parametrize("q", [0.0, 0.125, 0.25, 0.5])
def test_shortest_vector_fc2_reconstruction_matches_chain_formula(q: float) -> None:
    matrix = dynamical_matrix_from_fc2(
        _compact_fc2(),
        _chain_mapping(),
        [q, 0.0, 0.0],
    )
    expected = 2.0 * (1.0 - np.cos(2.0 * np.pi * q))
    np.testing.assert_allclose(matrix, expected * np.eye(3), atol=1.0e-14)


def test_full_and_compact_reconstructions_are_exactly_equal() -> None:
    mapping = _chain_mapping()
    for q in (
        [0.13, 0.0, 0.0],
        [-0.27, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ):
        full = dynamical_matrix_from_fc2(_full_fc2(), mapping, q)
        compact = dynamical_matrix_from_fc2(_compact_fc2(), mapping, q)
        np.testing.assert_array_equal(full, compact)


@pytest.mark.parametrize("q", [0.125, 0.25, 0.375])
def test_chain_analytical_dynamical_derivative(q: float) -> None:
    derivatives = dynamical_matrix_derivatives_from_fc2(
        _compact_fc2(),
        _chain_mapping(),
        [q, 0.0, 0.0],
    )
    expected = 4.0 * np.pi * np.sin(2.0 * np.pi * q)
    np.testing.assert_allclose(
        derivatives[0],
        expected * np.eye(3),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(derivatives[1:], 0.0, atol=1.0e-15)


def test_chain_group_velocity_matches_closed_form_and_si_conversion() -> None:
    qpoints = np.array([[0.125, 0.0, 0.0], [0.25, 0.0, 0.0]])
    result = compute_bulk_group_velocities(
        _compact_fc2(),
        _chain_mapping(),
        qpoints,
        frequency_factor_to_thz=1.0,
        cutoff_frequency_thz=1.0e-8,
    )

    assert isinstance(result, BulkGroupVelocities)
    expected_x = 2.0 * np.pi * np.cos(np.pi * qpoints[:, 0])
    np.testing.assert_allclose(
        result.values_thz_angstrom[:, :, 0],
        np.repeat(expected_x[:, None], 3, axis=1),
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(result.values_thz_angstrom[:, :, 1:], 0.0)
    np.testing.assert_allclose(
        result.values_m_per_s,
        100.0 * result.values_thz_angstrom,
    )
    assert not result.values_thz_angstrom.flags.writeable


def test_skew_cell_group_velocity_uses_cartesian_row_lattice_and_time_reversal() -> None:
    mapping = _skew_chain_mapping()
    q = 0.21
    result = compute_bulk_group_velocities(
        _compact_fc2(),
        mapping,
        [[q, 0.0, 0.0], [-q, 0.0, 0.0]],
        frequency_factor_to_thz=1.0,
        cutoff_frequency_thz=1.0e-8,
    )
    expected = 2.0 * np.pi * np.cos(np.pi * q) * mapping.primitive.lattice[0]
    np.testing.assert_allclose(
        result.values_thz_angstrom[0],
        np.repeat(expected[None, :], 3, axis=0),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.values_thz_angstrom[1],
        -result.values_thz_angstrom[0],
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_mapped_fc2_physical_audit_accepts_chain_and_rejects_nonreciprocity() -> None:
    accepted = audit_mapped_fc2(_compact_fc2(), _chain_mapping())
    assert accepted.passed
    assert accepted.asr_valid
    assert accepted.rotational_valid
    assert accepted.hermiticity_valid
    assert accepted.stable

    rejected = audit_mapped_fc2(
        _asymmetric_compact_fc2(),
        _asymmetric_two_species_mapping(),
    )
    assert not rejected.passed
    assert not rejected.hermiticity_valid


def test_full_fc2_audit_checks_every_redundant_translation_row() -> None:
    mapping = _translation_complete_chain_mapping()
    valid = _translation_complete_full_fc2()
    accepted = audit_mapped_fc2(valid, mapping)
    assert accepted.passed
    assert accepted.translation_redundancy_valid
    assert accepted.metrics["translation_redundancy_checked_rows"] == 2.0

    corrupted_values = valid.values.copy()
    corrupted_values[0, 0, 0, 0] += 12345.0
    rejected = audit_mapped_fc2(
        FC2Data(
            values=corrupted_values,
            masses=valid.masses,
            compact=False,
            p2s_map=valid.p2s_map,
            s2p_map=valid.s2p_map,
        ),
        mapping,
    )
    assert not rejected.passed
    assert not rejected.translation_redundancy_valid
    assert any("redundant translation audit failed" in error for error in rejected.errors)


def test_negative_q_is_complex_conjugate_before_hermitianization() -> None:
    mapping = _chain_mapping()
    positive = dynamical_matrix_from_fc2(
        _compact_fc2(), mapping, [0.17, 0.0, 0.0]
    )
    negative = dynamical_matrix_from_fc2(
        _compact_fc2(), mapping, [-0.17, 0.0, 0.0]
    )
    np.testing.assert_allclose(negative, positive.conj(), atol=1.0e-14)


def test_bulk_phonons_use_signed_frequencies_and_fixed_eigenvector_gauge() -> None:
    spectrum = compute_bulk_phonons(
        _compact_fc2(),
        _chain_mapping(),
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        frequency_factor_to_thz=1.0,
    )

    assert spectrum.n_qpoints == 2
    assert spectrum.n_bands == 3
    np.testing.assert_allclose(spectrum.frequencies[0], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(spectrum.frequencies[1], np.sqrt(2.0))
    np.testing.assert_allclose(
        spectrum.eigenvectors.conj().transpose(0, 2, 1)
        @ spectrum.eigenvectors,
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=1.0e-14,
    )
    assert not spectrum.qpoints.flags.writeable
    assert not spectrum.dynamical_matrices.flags.writeable
    assert not spectrum.frequencies.flags.writeable
    assert not spectrum.eigenvectors.flags.writeable


def test_tied_shortest_vectors_receive_equal_average_weight() -> None:
    mapping = _chain_mapping()
    multiplicity = mapping.multiplicity.copy()
    multiplicity[0, 0] = [2, 0]
    multiplicity[1, 0] = [1, 2]
    multiplicity[2, 0] = [1, 3]
    tied = CrystalMapping(
        primitive_matrix=mapping.primitive_matrix,
        supercell_matrix=mapping.supercell_matrix,
        primitive=mapping.primitive,
        supercell=mapping.supercell,
        p2s=mapping.p2s,
        s2p=mapping.s2p,
        p2p=mapping.p2p,
        shortest_vectors=[
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        multiplicity=multiplicity,
        atomic_permutations=mapping.atomic_permutations,
        purpose="fc2",
    )
    matrix = dynamical_matrix_from_fc2(
        _compact_fc2(),
        tied,
        [0.25, 0.0, 0.0],
    )
    # First neighbor contributes the mean of exp(-i*pi/2) and exp(+i*pi/2).
    expected = 2.0 - 1.0j
    np.testing.assert_allclose(matrix, expected * np.eye(3), atol=1.0e-14)


def test_fc2_mapping_conflicts_are_rejected_without_remapping() -> None:
    data = _compact_fc2()
    conflicting = FC2Data(
        values=data.values,
        masses=data.masses,
        compact=True,
        p2s_map=[0],
        s2p_map=[0, 0, 0],
    )
    with pytest.raises(ValueError, match="p2s_map conflicts"):
        dynamical_matrix_from_fc2(
            conflicting,
            _chain_mapping(),
            [0.0, 0.0, 0.0],
        )


def test_mass_order_mismatch_is_rejected() -> None:
    data = _compact_fc2()
    mismatched = FC2Data(
        values=data.values,
        masses=[2.0, 3.0, 2.0],
        compact=True,
        p2s_map=data.p2s_map,
        s2p_map=data.s2p_map,
    )
    with pytest.raises(ValueError, match="masses must exactly match"):
        dynamical_matrix_from_fc2(
            mismatched,
            _chain_mapping(),
            [0.0, 0.0, 0.0],
        )


@pytest.mark.parametrize(
    "data_factory",
    [_asymmetric_compact_fc2, _asymmetric_full_fc2],
)
def test_asymmetric_two_species_raw_and_hermitianized_matrices_are_analytical(
    data_factory,
) -> None:
    mapping = _asymmetric_two_species_mapping()
    data = data_factory()
    positive_q = np.array([0.25, 0.0, 0.0])
    negative_q = -positive_q

    raw = dynamical_matrix_from_fc2(
        data,
        mapping,
        positive_q,
        hermitianize=False,
    )
    negative_raw = dynamical_matrix_from_fc2(
        data,
        mapping,
        negative_q,
        hermitianize=False,
    )
    hermitianized = dynamical_matrix_from_fc2(
        data,
        mapping,
        positive_q,
        hermitianize=True,
    )

    expected_raw = _asymmetric_expected_raw(q_sign=1.0)
    expected_negative = _asymmetric_expected_raw(q_sign=-1.0)
    expected_hermitianized = (
        expected_raw + expected_raw.conj().T
    ) / 2.0
    np.testing.assert_allclose(raw, expected_raw, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        negative_raw,
        expected_negative,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        negative_raw,
        raw.conj(),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        hermitianized,
        expected_hermitianized,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert np.max(np.abs(raw - raw.conj().T)) == pytest.approx(0.5)
    assert np.max(np.abs(raw.imag)) == pytest.approx(3.0)
    # Reversing the production phase sign would return the negative-q matrix.
    assert np.max(np.abs(raw - expected_negative)) == pytest.approx(6.0)


def test_asymmetric_full_and_compact_fields_are_exactly_equal() -> None:
    mapping = _asymmetric_two_species_mapping()
    qpoint = [0.25, 0.0, 0.0]

    for hermitianize in (False, True):
        full = dynamical_matrix_from_fc2(
            _asymmetric_full_fc2(),
            mapping,
            qpoint,
            hermitianize=hermitianize,
        )
        compact = dynamical_matrix_from_fc2(
            _asymmetric_compact_fc2(),
            mapping,
            qpoint,
            hermitianize=hermitianize,
        )
        np.testing.assert_array_equal(full, compact)


def test_nondegenerate_bulk_eigenvectors_match_analytical_projectors() -> None:
    expected_eigenvalues, expected_projectors = (
        _asymmetric_analytical_eigenprojectors()
    )
    assert np.min(np.diff(expected_eigenvalues)) > 0.4

    spectrum = compute_bulk_phonons(
        _asymmetric_compact_fc2(),
        _asymmetric_two_species_mapping(),
        [[0.25, 0.0, 0.0]],
        frequency_factor_to_thz=1.0,
        hermitianize=True,
    )

    np.testing.assert_allclose(
        spectrum.frequencies[0],
        np.sqrt(expected_eigenvalues),
        rtol=0.0,
        atol=1.0e-14,
    )
    for band in range(spectrum.n_bands):
        vector = spectrum.eigenvectors[0, :, band]
        projector = np.outer(vector, vector.conj())
        np.testing.assert_allclose(
            projector,
            expected_projectors[band],
            rtol=0.0,
            atol=2.0e-14,
        )
        pivot = int(np.argmax(np.abs(vector)))
        assert vector[pivot].imag == pytest.approx(0.0, abs=1.0e-15)
        assert vector[pivot].real > 0.0


def test_default_frequency_factor_is_pinned_to_phono3py_4_4() -> None:
    assert DEFAULT_FREQUENCY_FACTOR_TO_THZ == (
        PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ
    )
    assert DEFAULT_FREQUENCY_FACTOR_TO_THZ == 15.633302300230191


@pytest.mark.parametrize(
    ("units", "mass_units", "message"),
    [
        ("J/m^2", "amu", "requires FC2 values"),
        ("eV/angstrom^2", "kg", "requires masses"),
    ],
)
def test_bulk_oracle_rejects_noncanonical_phono3py_units(
    units: str,
    mass_units: str,
    message: str,
) -> None:
    source = _compact_fc2()
    data = FC2Data(
        values=source.values,
        masses=source.masses,
        compact=True,
        p2s_map=source.p2s_map,
        s2p_map=source.s2p_map,
        units=units,
        mass_units=mass_units,
    )
    with pytest.raises(ValueError, match=message):
        dynamical_matrix_from_fc2(
            data,
            _chain_mapping(),
            [0.0, 0.0, 0.0],
        )
