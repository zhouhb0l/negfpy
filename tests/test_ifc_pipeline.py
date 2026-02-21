import numpy as np

from negfpy.io import list_readers, read_ifc
from negfpy.modeling import (
    BuildConfig,
    IFCData,
    IFCTerm,
    build_fc_terms,
    build_material_kspace_params,
    infer_principal_layer_size,
)


def _toy_ifc_payload() -> dict:
    return {
        "masses": [1.0],
        "dof_per_atom": 1,
        "terms": [
            {"translation": [0, 0, 0], "block": [[2.0]]},
            {"translation": [0, 1, 0], "block": [[-0.2]]},
            {"translation": [0, -1, 0], "block": [[-0.2]]},
            {"translation": [1, 0, 0], "block": [[-0.7]]},
        ],
    }


def test_phonopy_reader_is_registered() -> None:
    assert "phonopy" in list_readers()


def test_read_ifc_and_build_material_params() -> None:
    ifc = read_ifc(_toy_ifc_payload(), reader="phonopy")
    params = build_material_kspace_params(ifc, config=BuildConfig(onsite_pinning=1e-6))
    assert params.dof_per_atom == 1
    assert np.isclose(params.fc00_terms[(0, 0)][0, 0].real, 2.0)
    assert np.isclose(params.fc01_terms[(0, 0)][0, 0].real, -0.7)


def test_negative_dx_can_map_into_fc01() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[2.0]])),
            IFCTerm(dx=-1, dy=0, dz=0, block=np.array([[-0.5]])),
        ),
    )
    _, fc01, _ = build_fc_terms(ifc, config=BuildConfig(infer_fc01_from_negative_dx=True))
    assert np.isclose(fc01[(0, 0)][0, 0].real, -0.5)


def test_negative_dx_inference_does_not_double_count_existing_positive_dx() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[2.0]])),
            IFCTerm(dx=1, dy=0, dz=0, block=np.array([[-0.7]])),
            IFCTerm(dx=-1, dy=0, dz=0, block=np.array([[-0.7]])),
        ),
    )
    _, fc01, _ = build_fc_terms(ifc, config=BuildConfig(infer_fc01_from_negative_dx=True))
    assert np.isclose(fc01[(0, 0)][0, 0].real, -0.7)


def test_auto_principal_layer_enlargement_from_max_dx() -> None:
    ifc = IFCData(
        masses=np.array([12.0, 12.0]),
        dof_per_atom=3,
        terms=(
            IFCTerm(dx=0, dy=0, dz=0, block=2.0 * np.eye(6)),
            IFCTerm(dx=2, dy=0, dz=0, block=-0.1 * np.eye(6)),
        ),
    )
    cfg = BuildConfig()
    assert infer_principal_layer_size(ifc, cfg) == 2
    params = build_material_kspace_params(ifc, cfg)
    assert params.masses.shape == (4,)
    assert params.ndof == 12
    assert params.fc00_terms[(0, 0)].shape == (12, 12)
    assert params.fc01_terms[(0, 0)].shape == (12, 12)


def test_full_grid_metadata_disables_negative_dx_inference() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[2.0]])),
            IFCTerm(dx=1, dy=0, dz=0, block=np.array([[-0.7]])),
            IFCTerm(dx=0, dy=1, dz=0, block=np.array([[-0.2]])),
            IFCTerm(dx=0, dy=0, dz=1, block=np.array([[-0.3]])),
        ),
        metadata={"nr": (2, 2, 1)},
    )
    _, fc01_infer, _ = build_fc_terms(ifc, config=BuildConfig(infer_fc01_from_negative_dx=True))
    _, fc01_noinfer, _ = build_fc_terms(ifc, config=BuildConfig(infer_fc01_from_negative_dx=False))
    assert fc01_infer.keys() == fc01_noinfer.keys()
    for k in fc01_infer:
        assert np.allclose(fc01_infer[k], fc01_noinfer[k])


def test_full_grid_metadata_disables_fc00_termwise_hermitization() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[0.0 + 1.0j]])),
            IFCTerm(dx=1, dy=0, dz=0, block=np.array([[-0.5]])),
            IFCTerm(dx=0, dy=1, dz=0, block=np.array([[-0.2]])),
            IFCTerm(dx=0, dy=0, dz=1, block=np.array([[-0.3]])),
        ),
        metadata={"nr": (2, 2, 1)},
    )
    fc00_h, _, _ = build_fc_terms(ifc, config=BuildConfig(enforce_hermitian_fc00=True))
    fc00_nh, _, _ = build_fc_terms(ifc, config=BuildConfig(enforce_hermitian_fc00=False))
    assert np.allclose(fc00_h[(0, 0)], fc00_nh[(0, 0)])
