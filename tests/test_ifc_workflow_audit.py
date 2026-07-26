from __future__ import annotations

import json

import numpy as np
import pytest

from negfpy.modeling import enforce_translational_asr_on_self_term
from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.modeling.validators import IFCValidationReport, audit_ifc_data
from negfpy.workflows._ifc_audit import (
    ifc_audit_payload,
    ifc_validation_report_to_dict,
)
from negfpy.workflows.ifc_bulk import (
    _harmonic_release_preflight,
    _reorient_ifc_for_transport_direction,
)


def _scalar_chain(*, onsite: float) -> IFCData:
    return IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(0, 0, 0, np.array([[onsite]])),
            IFCTerm(1, 0, 0, np.array([[-1.0]])),
            IFCTerm(-1, 0, 0, np.array([[-1.0]])),
        ),
        lattice_vectors=np.eye(3),
        atom_positions=np.zeros((1, 3)),
        atom_symbols=("X",),
    )


def _release_config(*, requested: bool) -> dict[str, object]:
    return {
        "run": {"release_preflight": requested},
        "solver": {
            "eta_scheme": "fixed",
            "min_success_fraction": 1.0,
            "eta_device": None,
        },
    }


def _unique_mesh() -> dict[str, object]:
    section = {"kpoints": [(0.0,), (np.pi,)]}
    return {"low": section, "high": section}


def test_ifc_report_adapter_emits_strict_json_primitives() -> None:
    report = IFCValidationReport(
        schema_valid=True,
        mapping_valid=True,
        reciprocal_valid=True,
        asr_valid=True,
        hermiticity_valid=True,
        stable=True,
        metrics={"finite": 1.25, "unbounded": float("inf")},
    )

    serialized = ifc_validation_report_to_dict(report)

    assert serialized["metrics"]["unbounded"] == "Infinity"
    json.dumps(serialized, sort_keys=True, allow_nan=False)
    with pytest.raises(TypeError, match="IFCValidationReport"):
        ifc_validation_report_to_dict(object())  # type: ignore[arg-type]


def test_raw_and_corrected_ifc_audits_are_both_preserved() -> None:
    raw_ifc = _scalar_chain(onsite=3.0)
    corrected_ifc, residual = enforce_translational_asr_on_self_term(raw_ifc)
    raw = audit_ifc_data(raw_ifc, random_q_count=0)
    final = audit_ifc_data(corrected_ifc, random_q_count=0)

    payload = ifc_audit_payload(
        raw=raw,
        final=final,
        release_preflight=True,
    )

    assert residual == pytest.approx(1.0)
    assert raw.passed is False
    assert final.passed is True
    assert payload["raw"]["passed"] is False
    assert payload["final"]["passed"] is True
    assert payload["release_preflight_passed"] is True
    assert payload["certification_eligible"] is True
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_final_ifc_audit_is_gated_only_when_release_preflight_is_requested() -> None:
    failed = audit_ifc_data(_scalar_chain(onsite=3.0), random_q_count=0)
    payload = ifc_audit_payload(
        raw=failed,
        final=failed,
        release_preflight=True,
    )

    exploratory = _harmonic_release_preflight(
        _release_config(requested=False),
        calculation="transmission",
        kmesh=_unique_mesh(),
        ifc_audit=payload,
    )
    assert exploratory["checks"]["final_ifc_audit"] is False
    assert exploratory["release_certified"] is False

    with pytest.raises(ValueError, match="final IFC dataset failed"):
        _harmonic_release_preflight(
            _release_config(requested=True),
            calculation="transmission",
            kmesh=_unique_mesh(),
            ifc_audit=payload,
        )


@pytest.mark.parametrize(
    ("direction", "permutation"),
    [(2, (1, 0, 2)), (3, (2, 0, 1))],
)
def test_transport_reorientation_preserves_skew_cell_cartesian_geometry(
    direction: int,
    permutation: tuple[int, int, int],
) -> None:
    lattice = np.array(
        [[2.0, 0.3, 0.1], [0.2, 3.0, 0.4], [0.5, 0.1, 4.0]]
    )
    positions = np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.9]])
    ifc = IFCData(
        masses=np.array([1.0, 2.0]),
        dof_per_atom=1,
        terms=(IFCTerm(1, 2, 3, np.eye(2)),),
        metadata={"nr": (4, 6, 8)},
        lattice_vectors=lattice.copy(),
        atom_positions=positions.copy(),
        atom_symbols=("A", "B"),
    )
    cartesian_before = positions @ lattice

    oriented = _reorient_ifc_for_transport_direction(ifc, direction=direction)
    index = np.asarray(permutation, dtype=int)

    np.testing.assert_array_equal(oriented.lattice_vectors, lattice[index, :])
    np.testing.assert_array_equal(oriented.atom_positions, positions[:, index])
    np.testing.assert_allclose(
        oriented.atom_positions @ oriented.lattice_vectors,
        cartesian_before,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert (
        oriented.terms[0].dx,
        oriented.terms[0].dy,
        oriented.terms[0].dz,
    ) == tuple((1, 2, 3)[axis] for axis in permutation)
    assert tuple(oriented.metadata["nr"]) == tuple(
        (4, 6, 8)[axis] for axis in permutation
    )
    np.testing.assert_array_equal(ifc.lattice_vectors, lattice)
    np.testing.assert_array_equal(ifc.atom_positions, positions)
