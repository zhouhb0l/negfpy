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
from negfpy.workflows.ifc_bulk import (
    _drop_nyquist_transverse_terms,
    _enforce_transverse_pm_symmetry,
    _reorient_ifc_for_transport_direction,
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


def _write_phonopy_fc_and_poscar(tmp_path) -> tuple:
    poscar = tmp_path / "POSCAR"
    force_constants = tmp_path / "FORCE_CONSTANTS"

    poscar.write_text(
        "\n".join(
            [
                "test supercell",
                "1.0",
                "4.0 0.0 0.0",
                "0.0 1.0 0.0",
                "0.0 0.0 1.0",
                "Si",
                "4",
                "Direct",
                "0.0 0.0 0.0",
                "0.25 0.0 0.0",
                "0.5 0.0 0.0",
                "0.75 0.0 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # 4x1x1 supercell of a 1-atom primitive cell.
    # nearest-neighbor chain IFC: onsite 2I, +-1 neighbor -1I.
    lines = ["4 4"]
    for i in range(4):
        for j in range(4):
            lines.append(f"{i + 1} {j + 1}")
            raw = (j - i) % 4
            if raw == 0:
                m = np.eye(3) * 2.0
            elif raw in {1, 3}:
                m = -np.eye(3)
            else:
                m = np.zeros((3, 3))
            for a in range(3):
                lines.append(f"{m[a,0]: .16f} {m[a,1]: .16f} {m[a,2]: .16f}")
    force_constants.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return poscar, force_constants


def test_read_phonopy_force_constants_from_path(tmp_path) -> None:
    _poscar, fc = _write_phonopy_fc_and_poscar(tmp_path)
    ifc = read_ifc(fc, reader="phonopy")

    assert ifc.dof_per_atom == 3
    assert ifc.masses.shape == (1,)
    assert ifc.atom_symbols == ("Si",)
    assert tuple(ifc.metadata.get("nr", ())) == (4, 1, 1)
    assert ifc.metadata.get("source_format") == "phonopy_force_constants"
    assert ifc.metadata.get("mass_unit_converted") == "2*electron_mass (QE Ry mass unit)"
    assert ifc.metadata.get("force_constant_unit_converted") == "Ry/bohr^2"

    terms = {(int(t.dx), int(t.dy), int(t.dz)): np.asarray(t.block) for t in ifc.terms}
    fc_scale = float(ifc.metadata["force_constant_scale_to_qe"])
    mass_scale = float(ifc.metadata["mass_scale_to_qe"])
    assert (0, 0, 0) in terms
    assert (1, 0, 0) in terms
    assert (-1, 0, 0) in terms
    assert np.allclose(terms[(0, 0, 0)], (2.0 * fc_scale) * np.eye(3))
    assert np.allclose(terms[(1, 0, 0)], (-1.0 * fc_scale) * np.eye(3))
    assert np.allclose(terms[(-1, 0, 0)], (-1.0 * fc_scale) * np.eye(3))
    assert np.isclose(float(ifc.masses[0]), 28.085 * mass_scale)


def test_read_phonopy_force_constants_from_dict_source(tmp_path) -> None:
    poscar, fc = _write_phonopy_fc_and_poscar(tmp_path)
    ifc = read_ifc(
        {
            "force_constants_path": str(fc),
            "poscar_path": str(poscar),
            "supercell": [4, 1, 1],
        },
        reader="phonopy",
    )
    assert tuple(ifc.metadata.get("nr", ())) == (4, 1, 1)
    assert ifc.atom_symbols == ("Si",)


def test_read_phonopy_force_constants_from_json_source(tmp_path) -> None:
    poscar, fc = _write_phonopy_fc_and_poscar(tmp_path)
    cfg = tmp_path / "phonopy_reader.json"
    cfg.write_text(
        (
            "{\n"
            '  "force_constants_path": "FORCE_CONSTANTS",\n'
            '  "poscar_path": "POSCAR",\n'
            '  "supercell": [4, 1, 1]\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    ifc = read_ifc(cfg, reader="phonopy")
    assert tuple(ifc.metadata.get("nr", ())) == (4, 1, 1)
    assert ifc.atom_symbols == ("Si",)
    assert poscar.exists() and fc.exists()


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


def test_full_grid_even_nr1_rejects_oversized_principal_layer() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=-2, dy=0, dz=0, block=np.array([[0.1]])),
            IFCTerm(dx=-1, dy=0, dz=0, block=np.array([[0.2]])),
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[1.0]])),
            IFCTerm(dx=1, dy=0, dz=0, block=np.array([[0.2]])),
        ),
        metadata={"nr": (4, 1, 1)},
    )
    try:
        infer_principal_layer_size(ifc, BuildConfig(principal_layer_size=4))
        raise AssertionError("Expected oversized principal_layer_size to be rejected.")
    except ValueError as exc:
        assert "principal_layer_size is too large" in str(exc)


def test_full_grid_even_nr1_accepts_principal_layer_up_to_max_abs_dx() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=-2, dy=0, dz=0, block=np.array([[0.1]])),
            IFCTerm(dx=-1, dy=0, dz=0, block=np.array([[0.2]])),
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[1.0]])),
            IFCTerm(dx=1, dy=0, dz=0, block=np.array([[0.2]])),
        ),
        metadata={"nr": (4, 1, 1)},
    )
    assert infer_principal_layer_size(ifc, BuildConfig(principal_layer_size=2)) == 2


def test_full_grid_even_nr1_allows_oversized_principal_layer_with_nyquist_half_split() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=-2, dy=0, dz=0, block=np.array([[0.4]])),
            IFCTerm(dx=-1, dy=0, dz=0, block=np.array([[0.3]])),
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[1.0]])),
            IFCTerm(dx=1, dy=0, dz=0, block=np.array([[0.3]])),
        ),
        metadata={"nr": (4, 1, 1)},
    )
    cfg = BuildConfig(principal_layer_size=4, nyquist_split_half=True)
    assert infer_principal_layer_size(ifc, cfg) == 4
    fc00, fc01, fc10 = build_fc_terms(ifc, config=cfg)
    assert (0, 0) in fc00
    assert (0, 0) in fc01
    assert (0, 0) in fc10
    assert fc01[(0, 0)].shape == (4, 4)
    assert fc10[(0, 0)].shape == (4, 4)


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


def test_drop_nyquist_transverse_terms_filters_even_grid_negative_half_index() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[5.0]])),
            IFCTerm(dx=0, dy=-2, dz=0, block=np.array([[1.0]])),
            IFCTerm(dx=0, dy=-1, dz=0, block=np.array([[2.0]])),
            IFCTerm(dx=1, dy=0, dz=0, block=np.array([[6.0]])),
            IFCTerm(dx=1, dy=-2, dz=0, block=np.array([[3.0]])),
            IFCTerm(dx=1, dy=-1, dz=0, block=np.array([[4.0]])),
            IFCTerm(dx=-1, dy=0, dz=0, block=np.array([[6.0]])),
            IFCTerm(dx=-1, dy=-2, dz=0, block=np.array([[3.0]])),
            IFCTerm(dx=-1, dy=-1, dz=0, block=np.array([[4.0]])),
        ),
        metadata={"nr": (1, 4, 1)},
    )
    params = build_material_kspace_params(ifc, config=BuildConfig(principal_layer_size=1))
    filtered, info = _drop_nyquist_transverse_terms(params, ifc_metadata=ifc.metadata)
    assert info["dropped_dy"] == -2
    assert info["n_dropped_fc00"] == 1
    assert info["n_dropped_fc01"] == 1
    assert (-2, 0) not in filtered.fc00_terms
    assert (-2, 0) not in filtered.fc01_terms
    assert (-1, 0) in filtered.fc00_terms
    assert (-1, 0) in filtered.fc01_terms


def test_enforce_transverse_pm_symmetry_completes_fc00_and_fc01_fc10_pairs_for_nyquist_only() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(
            IFCTerm(dx=0, dy=0, dz=0, block=np.array([[2.0]])),
            IFCTerm(dx=0, dy=2, dz=0, block=np.array([[1.0 + 2.0j]])),
            IFCTerm(dx=1, dy=2, dz=0, block=np.array([[3.0 + 1.0j]])),
            IFCTerm(dx=-1, dy=0, dz=0, block=np.array([[4.0]])),
        ),
        metadata={"nr": (1, 4, 1)},
    )
    params = build_material_kspace_params(ifc, config=BuildConfig(principal_layer_size=1))
    sym, info = _enforce_transverse_pm_symmetry(params, ifc_metadata=ifc.metadata)

    assert info["enabled"] is True
    assert (-2, 0) in sym.fc00_terms
    assert np.allclose(sym.fc00_terms[(-2, 0)], sym.fc00_terms[(2, 0)].conj().T)
    assert (-2, 0) in sym.fc10_terms
    assert np.allclose(sym.fc10_terms[(-2, 0)], sym.fc01_terms[(2, 0)].conj().T)
    # Non-Nyquist term should remain untouched by Nyquist-only enforcement.
    assert (-1, 0) not in sym.fc00_terms


def test_reorient_ifc_for_transport_direction_keeps_identity_for_direction_1() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(IFCTerm(dx=1, dy=2, dz=3, block=np.array([[1.0]])),),
        metadata={"nr": (4, 6, 8)},
    )
    out = _reorient_ifc_for_transport_direction(ifc, direction=1)
    assert out.terms[0].dx == 1 and out.terms[0].dy == 2 and out.terms[0].dz == 3
    assert tuple(out.metadata["nr"]) == (4, 6, 8)


def test_reorient_ifc_for_transport_direction_swaps_indices_for_direction_2() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(IFCTerm(dx=1, dy=2, dz=3, block=np.array([[1.0]])),),
        metadata={"nr": (4, 6, 8)},
    )
    out = _reorient_ifc_for_transport_direction(ifc, direction=2)
    assert out.terms[0].dx == 2 and out.terms[0].dy == 1 and out.terms[0].dz == 3
    assert tuple(out.metadata["nr"]) == (6, 4, 8)


def test_reorient_ifc_for_transport_direction_cycles_indices_for_direction_3() -> None:
    ifc = IFCData(
        masses=np.array([1.0]),
        dof_per_atom=1,
        terms=(IFCTerm(dx=1, dy=2, dz=3, block=np.array([[1.0]])),),
        metadata={"nr": (4, 6, 8)},
    )
    out = _reorient_ifc_for_transport_direction(ifc, direction=3)
    assert out.terms[0].dx == 3 and out.terms[0].dy == 1 and out.terms[0].dz == 2
    assert tuple(out.metadata["nr"]) == (8, 4, 6)
