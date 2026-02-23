import numpy as np
import pytest

from negfpy.core import transmission
from negfpy.models import (
    CubicLatticeParams,
    MaterialKspaceParams,
    SquareLatticeParams,
    cubic_lattice_device,
    cubic_lattice_lead,
    material_kspace_device,
    material_kspace_lead,
    square_lattice_device,
    square_lattice_lead,
)


def _as_1x1_terms(
    onsite_const: float,
    onsite_y: float = 0.0,
    onsite_z: float = 0.0,
    couple_x: float = 0.0,
) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]]:
    fc00_terms: dict[tuple[int, int], np.ndarray] = {(0, 0): np.array([[onsite_const]], dtype=float)}
    if onsite_y != 0.0:
        fc00_terms[(1, 0)] = np.array([[onsite_y]], dtype=float)
        fc00_terms[(-1, 0)] = np.array([[onsite_y]], dtype=float)
    if onsite_z != 0.0:
        fc00_terms[(0, 1)] = np.array([[onsite_z]], dtype=float)
        fc00_terms[(0, -1)] = np.array([[onsite_z]], dtype=float)

    fc01_terms = {(0, 0): np.array([[couple_x]], dtype=float)}
    return fc00_terms, fc01_terms


def test_material_kspace_reduces_to_square_scalar_model() -> None:
    m = 1.0
    kx = 1.0
    ky = 0.8
    omega = 1.1
    kpar = (np.pi / 3.0,)

    toy_params = SquareLatticeParams(mass=m, spring_x=kx, spring_y=ky)
    toy_lead = square_lattice_lead(toy_params)
    toy_device = square_lattice_device(n_layers=30, params=toy_params)

    fc00_terms, fc01_terms = _as_1x1_terms(
        onsite_const=2.0 * kx + 2.0 * ky,
        onsite_y=-ky,
        couple_x=-kx,
    )
    mat_params = MaterialKspaceParams(
        masses=np.array([m]),
        fc00_terms=fc00_terms,
        fc01_terms=fc01_terms,
        dof_per_atom=1,
    )
    mat_lead = material_kspace_lead(mat_params)
    mat_device = material_kspace_device(n_layers=30, params=mat_params)

    t_toy = transmission(omega, toy_device, toy_lead, toy_lead, kpar=kpar, eta=1e-8)
    t_mat = transmission(omega, mat_device, mat_lead, mat_lead, kpar=kpar, eta=1e-8)
    assert abs(t_toy - t_mat) < 1e-10


def test_material_kspace_reduces_to_cubic_scalar_model() -> None:
    m = 1.0
    kx = 1.0
    ky = 0.6
    kz = 0.4
    omega = 1.2
    kpar = (np.pi / 4.0, np.pi / 5.0)

    toy_params = CubicLatticeParams(mass=m, spring_x=kx, spring_y=ky, spring_z=kz)
    toy_lead = cubic_lattice_lead(toy_params)
    toy_device = cubic_lattice_device(n_layers=35, params=toy_params)

    fc00_terms, fc01_terms = _as_1x1_terms(
        onsite_const=2.0 * kx + 2.0 * ky + 2.0 * kz,
        onsite_y=-ky,
        onsite_z=-kz,
        couple_x=-kx,
    )
    mat_params = MaterialKspaceParams(
        masses=np.array([m]),
        fc00_terms=fc00_terms,
        fc01_terms=fc01_terms,
        dof_per_atom=1,
    )
    mat_lead = material_kspace_lead(mat_params)
    mat_device = material_kspace_device(n_layers=35, params=mat_params)

    t_toy = transmission(omega, toy_device, toy_lead, toy_lead, kpar=kpar, eta=1e-8)
    t_mat = transmission(omega, mat_device, mat_lead, mat_lead, kpar=kpar, eta=1e-8)
    assert abs(t_toy - t_mat) < 1e-10


def test_material_kspace_multidof_runs() -> None:
    n_atoms = 2
    dof = 3
    ndof = n_atoms * dof
    masses = np.array([1.0, 1.4])

    # Simple stable diagonal model for smoke validation.
    fc00_terms = {(0, 0): 2.0 * np.eye(ndof), (1, 0): -0.2 * np.eye(ndof), (-1, 0): -0.2 * np.eye(ndof)}
    fc01_terms = {(0, 0): -0.7 * np.eye(ndof)}
    params = MaterialKspaceParams(
        masses=masses,
        fc00_terms=fc00_terms,
        fc01_terms=fc01_terms,
        dof_per_atom=dof,
        onsite_pinning=1e-6,
    )
    lead = material_kspace_lead(params)
    device = material_kspace_device(n_layers=20, params=params)
    tval = transmission(omega=0.7, device=device, lead_left=lead, lead_right=lead, kpar=(0.2, -0.3), eta=1e-8)
    assert np.isfinite(tval)
    assert tval > -1e-8


def test_material_kspace_generalized_eigen_matches_sancho_scalar() -> None:
    m = 1.0
    kx = 1.0
    ky = 0.8
    omega = 1.1
    kpar = (np.pi / 3.0,)

    fc00_terms, fc01_terms = _as_1x1_terms(
        onsite_const=2.0 * kx + 2.0 * ky,
        onsite_y=-ky,
        couple_x=-kx,
    )
    mat_params = MaterialKspaceParams(
        masses=np.array([m]),
        fc00_terms=fc00_terms,
        fc01_terms=fc01_terms,
        dof_per_atom=1,
    )
    mat_lead = material_kspace_lead(mat_params)
    mat_device = material_kspace_device(n_layers=30, params=mat_params)

    t_sr = transmission(
        omega,
        mat_device,
        mat_lead,
        mat_lead,
        kpar=kpar,
        eta=1e-8,
        surface_gf_method="sancho_rubio",
    )
    t_ge = transmission(
        omega,
        mat_device,
        mat_lead,
        mat_lead,
        kpar=kpar,
        eta=1e-8,
        surface_gf_method="generalized_eigen",
    )
    t_svd = transmission(
        omega,
        mat_device,
        mat_lead,
        mat_lead,
        kpar=kpar,
        eta=1e-8,
        surface_gf_method="generalized_eigen_svd",
    )
    assert abs(t_sr - t_ge) < 1e-10
    assert abs(t_sr - t_svd) < 1e-10


def test_material_kspace_multidof_generalized_eigen_runs() -> None:
    n_atoms = 2
    dof = 3
    ndof = n_atoms * dof
    masses = np.array([1.0, 1.4])

    fc00_terms = {(0, 0): 2.0 * np.eye(ndof), (1, 0): -0.2 * np.eye(ndof), (-1, 0): -0.2 * np.eye(ndof)}
    fc01_terms = {(0, 0): -0.7 * np.eye(ndof)}
    params = MaterialKspaceParams(
        masses=masses,
        fc00_terms=fc00_terms,
        fc01_terms=fc01_terms,
        dof_per_atom=dof,
        onsite_pinning=1e-6,
    )
    lead = material_kspace_lead(params)
    device = material_kspace_device(n_layers=20, params=params)
    tval = transmission(
        omega=0.7,
        device=device,
        lead_left=lead,
        lead_right=lead,
        kpar=(0.2, -0.3),
        eta=1e-8,
        surface_gf_method="generalized_eigen",
    )
    assert np.isfinite(tval)
    assert tval > -1e-8


def test_material_kspace_rejects_kpar_with_more_than_two_components() -> None:
    fc00_terms = {(0, 0): np.array([[2.0]], dtype=float)}
    fc01_terms = {(0, 0): np.array([[-1.0]], dtype=float)}
    params = MaterialKspaceParams(
        masses=np.array([1.0]),
        fc00_terms=fc00_terms,
        fc01_terms=fc01_terms,
        dof_per_atom=1,
    )
    lead = material_kspace_lead(params)

    with pytest.raises(ValueError, match="at most two transverse components"):
        lead.blocks((0.1, 0.2, 0.3))


def test_toy_lattice_models_reject_invalid_kpar_dimensions() -> None:
    sq = SquareLatticeParams(mass=1.0, spring_x=1.0, spring_y=0.8)
    square_lead = square_lattice_lead(sq)
    with pytest.raises(ValueError, match="at most one transverse component"):
        square_lead.blocks((0.1, 0.2))

    cb = CubicLatticeParams(mass=1.0, spring_x=1.0, spring_y=0.8, spring_z=0.6)
    cubic_lead = cubic_lattice_lead(cb)
    with pytest.raises(ValueError, match="at most two transverse components"):
        cubic_lead.blocks((0.1, 0.2, 0.3))
