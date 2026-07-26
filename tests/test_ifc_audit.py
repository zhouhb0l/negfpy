from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.modeling.validators import (
    IFCValidationReport,
    audit_ifc_data,
)


def _scalar_chain_ifc(
    *,
    onsite: float = 2.0,
    positive: float = -1.0,
    negative: float = -1.0,
) -> IFCData:
    return IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(0, 0, 0, np.array([[onsite]])),
            IFCTerm(1, 0, 0, np.array([[positive]])),
            IFCTerm(-1, 0, 0, np.array([[negative]])),
        ),
        lattice_vectors=np.eye(3),
        atom_positions=np.array([[0.0, 0.0, 0.0]]),
        atom_symbols=("X",),
        index_convention="layer-major",
    )


def _two_atom_cartesian_ifc(
    stiffness: np.ndarray,
    *,
    include_geometry: bool = True,
) -> IFCData:
    """Return an ASR-satisfying two-atom Cartesian spring fixture."""

    coupling = np.asarray(stiffness, dtype=float)
    block = np.block(
        [
            [coupling, -coupling],
            [-coupling, coupling],
        ]
    )
    # This skewed row-vector lattice makes fractional @ lattice observably
    # different from lattice @ fractional.
    lattice = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 4.0],
        ]
    )
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25, 0.50, 0.0],
        ]
    )
    return IFCData(
        masses=np.array([1.0, 2.0]),
        dof_per_atom=3,
        terms=(IFCTerm(0, 0, 0, block),),
        lattice_vectors=lattice if include_geometry else None,
        atom_positions=positions if include_geometry else None,
        atom_symbols=("A", "B") if include_geometry else None,
        index_convention="atom-major Cartesian components",
    )


def _translated_two_atom_cartesian_ifc(stiffness: np.ndarray) -> IFCData:
    """Return one central spring crossing a translated-cell boundary."""

    coupling = np.asarray(stiffness, dtype=float)
    zero = np.zeros((6, 6), dtype=float)
    onsite = zero.copy()
    onsite[:3, :3] = coupling
    onsite[3:, 3:] = coupling
    positive = zero.copy()
    positive[:3, 3:] = -coupling
    negative = positive.T
    return IFCData(
        masses=np.array([1.0, 2.0]),
        dof_per_atom=3,
        terms=(
            IFCTerm(0, 0, 0, onsite),
            IFCTerm(1, -1, 0, positive),
            IFCTerm(-1, 1, 0, negative),
        ),
        lattice_vectors=np.array(
            [
                [2.0, 1.0, 0.0],
                [0.0, 3.0, 1.0],
                [0.0, 0.0, 4.0],
            ]
        ),
        atom_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.50, 0.0],
            ]
        ),
        atom_symbols=("A", "B"),
        index_convention="atom-major Cartesian components",
    )


def test_valid_chain_passes_reciprocal_asr_hermiticity_and_stability() -> None:
    report = audit_ifc_data(
        _scalar_chain_ifc(),
        qpoints=[(0.17, -0.31, 0.29)],
        random_q_count=3,
        random_seed=81,
    )

    assert isinstance(report, IFCValidationReport)
    assert report.passed
    assert report.schema_valid
    assert report.mapping_valid
    assert report.reciprocal_valid
    assert report.asr_valid
    assert report.hermiticity_valid
    assert report.stable
    assert report.metrics["reciprocal_max_abs"] == pytest.approx(0.0)
    assert report.metrics["asr_max_abs"] == pytest.approx(0.0)
    assert report.metrics["minimum_eigenvalue"] >= -1.0e-14
    assert (0.17, -0.31, 0.29) in report.qpoints


def test_rotational_invariance_uses_fractional_positions_and_row_lattice() -> None:
    lattice = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 1.0],
            [0.0, 0.0, 4.0],
        ]
    )
    fractional_bond = np.array([0.25, 0.50, 0.0])
    translation = np.array([1.0, -1.0, 0.0])
    cartesian_bond = fractional_bond @ lattice + translation @ lattice
    direction = cartesian_bond / np.linalg.norm(cartesian_bond)
    central_stiffness = np.outer(direction, direction)
    ifc = _translated_two_atom_cartesian_ifc(central_stiffness)
    original_lattice = np.array(ifc.lattice_vectors, copy=True)
    original_positions = np.array(ifc.atom_positions, copy=True)

    report = audit_ifc_data(
        ifc,
        rotational_atol=1.0e-12,
        rotational_rtol=1.0e-12,
        random_q_count=0,
    )

    assert report.passed
    assert report.rotational_geometry_complete
    assert report.rotational_applicable
    assert report.rotational_valid
    assert report.metrics["rotational_geometry_complete"] == 1.0
    assert report.metrics["rotational_applicable"] == 1.0
    assert report.metrics["rotational_max_abs"] < 1.0e-15
    assert report.metrics["rotational_relative"] < 1.0e-15
    np.testing.assert_array_equal(ifc.lattice_vectors, original_lattice)
    np.testing.assert_array_equal(ifc.atom_positions, original_positions)


def test_rotational_violation_fails_only_rotational_gate() -> None:
    # A Cartesian-x force constant for the skewed bond is positive
    # semidefinite and ASR-compliant, but it exerts a nonzero torque.
    stiffness = np.diag([1.0, 0.0, 0.0])
    report = audit_ifc_data(
        _two_atom_cartesian_ifc(stiffness),
        rotational_atol=0.0,
        rotational_rtol=0.0,
        random_q_count=0,
    )

    assert report.schema_valid
    assert report.mapping_valid
    assert report.reciprocal_valid
    assert report.asr_valid
    assert report.hermiticity_valid
    assert report.stable
    assert report.rotational_applicable
    assert not report.rotational_valid
    assert not report.passed
    assert report.metrics["rotational_max_abs"] > 0.0
    assert report.metrics["rotational_relative"] > 0.0
    assert report.metrics["rotational_threshold"] == 0.0
    assert any("rotational invariance failed" in error for error in report.errors)


def test_rotational_check_is_not_applicable_without_geometry() -> None:
    direction = np.array([1.0, 0.0, 0.0])
    report = audit_ifc_data(
        _two_atom_cartesian_ifc(
            np.outer(direction, direction),
            include_geometry=False,
        ),
        random_q_count=0,
    )

    assert report.passed
    assert not report.rotational_geometry_complete
    assert not report.rotational_applicable
    assert report.rotational_valid
    assert report.metrics["rotational_geometry_complete"] == 0.0
    assert report.metrics["rotational_applicable"] == 0.0
    assert report.metrics["rotational_max_abs"] == 0.0
    assert report.metrics["rotational_relative"] == 0.0
    assert any(
        "Rotational invariance check is not applicable" in warning
        for warning in report.warnings
    )


def test_rotational_check_is_not_applicable_to_scalar_toy() -> None:
    report = audit_ifc_data(_scalar_chain_ifc(), random_q_count=0)

    assert report.passed
    assert report.rotational_geometry_complete
    assert not report.rotational_applicable
    assert report.rotational_valid
    assert any(
        "dof_per_atom is not 3 Cartesian components" in warning
        for warning in report.warnings
    )


def test_duplicate_translations_are_merged_and_reported_without_mutation() -> None:
    terms = (
        IFCTerm(0, 0, 0, np.array([[0.75]])),
        IFCTerm(0, 0, 0, np.array([[1.25]])),
        IFCTerm(1, 0, 0, np.array([[-1.0]])),
        IFCTerm(-1, 0, 0, np.array([[-1.0]])),
    )
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=terms,
    )

    report = audit_ifc_data(ifc, random_q_count=0)

    assert report.passed
    assert report.duplicate_translations == ((0, 0, 0),)
    assert report.metrics["input_term_count"] == 4
    assert report.metrics["merged_translation_count"] == 3
    assert report.metrics["duplicate_translation_count"] == 1
    assert any("duplicate" in warning for warning in report.warnings)
    assert len(ifc.terms) == 4
    np.testing.assert_array_equal(ifc.terms[0].block, [[0.75]])


def test_missing_reciprocal_partner_and_nonhermitian_dq_are_reported() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(0, 0, 0, np.array([[2.0]])),
            IFCTerm(1, 0, 0, np.array([[-1.0]])),
        ),
    )

    report = audit_ifc_data(ifc, qpoints=[(0.37, 0.0, 0.0)], random_q_count=0)

    assert not report.passed
    assert not report.reciprocal_valid
    assert not report.hermiticity_valid
    assert report.missing_reciprocal_translations == ((1, 0, 0),)
    assert report.metrics["reciprocal_max_abs"] == pytest.approx(1.0)
    assert report.metrics["dynamical_hermiticity_max_abs"] > 0.0
    assert any("reciprocal relation" in error for error in report.errors)


def test_asr_failure_is_independent_of_reciprocity_and_stability() -> None:
    report = audit_ifc_data(
        _scalar_chain_ifc(onsite=3.0),
        random_q_count=2,
    )

    assert report.reciprocal_valid
    assert report.hermiticity_valid
    assert report.stable
    assert not report.asr_valid
    assert report.metrics["asr_max_abs"] == pytest.approx(1.0)
    assert any("acoustic sum rule" in error for error in report.errors)


def test_translational_asr_sums_displaced_atoms_not_the_full_matrix() -> None:
    # A two-atom cell can have a nonzero Gamma force-constant matrix while
    # every row sums to zero under rigid translation.
    block = np.array([[1.0, -1.0], [-1.0, 1.0]])
    ifc = IFCData(
        masses=np.array([1.0, 2.0]),
        dof_per_atom=1,
        terms=(IFCTerm(0, 0, 0, block),),
        lattice_vectors=np.eye(3),
        atom_positions=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        atom_symbols=("A", "B"),
    )

    report = audit_ifc_data(ifc, random_q_count=0)

    assert report.passed
    assert report.asr_valid
    assert report.metrics["asr_max_abs"] == pytest.approx(0.0)
    assert report.metrics["phi_sum_matrix_max_abs"] == pytest.approx(1.0)


def test_negative_dynamical_eigenvalues_fail_stability_only() -> None:
    report = audit_ifc_data(
        _scalar_chain_ifc(onsite=-2.0, positive=1.0, negative=1.0),
        random_q_count=2,
    )

    assert report.schema_valid
    assert report.reciprocal_valid
    assert report.asr_valid
    assert report.hermiticity_valid
    assert not report.stable
    assert report.metrics["minimum_eigenvalue"] < -1.0
    assert report.metrics["unstable_qpoint_count"] > 0
    assert report.minimum_eigenvalue_qpoint is not None


def test_schema_errors_are_returned_instead_of_calling_legacy_validator() -> None:
    ifc = IFCData(
        masses=np.array([1.0, np.nan]),
        dof_per_atom=1,
        terms=(IFCTerm(0, 0, 0, np.ones((1, 1))),),
    )

    report = audit_ifc_data(ifc)

    assert not report.schema_valid
    assert not report.passed
    assert not report.reciprocal_valid
    assert any("masses" in error for error in report.errors)
    assert any("shape" in error for error in report.errors)
    assert any("skipped" in error for error in report.errors)


def test_geometry_atom_mapping_mismatch_is_a_reported_failure() -> None:
    identity = np.eye(2)
    ifc = IFCData(
        masses=np.array([1.0, 1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(0, 0, 0, 2.0 * identity),
            IFCTerm(1, 0, 0, -identity),
            IFCTerm(-1, 0, 0, -identity),
        ),
        lattice_vectors=np.eye(3),
        atom_positions=np.array([[0.0, 0.0, 0.0]]),
        atom_symbols=("A", "B"),
    )

    report = audit_ifc_data(ifc, random_q_count=0)

    assert report.schema_valid
    assert not report.mapping_valid
    assert not report.passed
    assert any("atom_positions atom count" in error for error in report.errors)


def test_absent_geometry_is_explicit_warning_but_not_a_toy_model_failure() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=_scalar_chain_ifc().terms,
    )

    report = audit_ifc_data(ifc, random_q_count=0)

    assert report.passed
    assert report.mapping_valid
    assert report.metrics["geometry_metadata_complete"] == 0.0
    assert any("could not be independently checked" in warning for warning in report.warnings)


def test_audit_is_deterministic_and_report_is_deeply_immutable() -> None:
    kwargs = {"random_q_count": 5, "random_seed": 2026}
    first = audit_ifc_data(_scalar_chain_ifc(), **kwargs)
    second = audit_ifc_data(_scalar_chain_ifc(), **kwargs)

    assert first.qpoints == second.qpoints
    assert first.metrics == second.metrics
    with pytest.raises(FrozenInstanceError):
        first.stable = False  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.metrics["new"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"reciprocal_atol": -1.0}, ValueError),
        ({"rotational_atol": -1.0}, ValueError),
        ({"rotational_rtol": np.inf}, ValueError),
        ({"random_q_count": -1}, ValueError),
        ({"random_q_count": 1.5}, TypeError),
        ({"qpoints": [(0.0, 1.0)]}, ValueError),
        ({"qpoints": [(0.0, np.nan, 1.0)]}, ValueError),
    ],
)
def test_invalid_audit_controls_are_rejected(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        audit_ifc_data(_scalar_chain_ifc(), **kwargs)
