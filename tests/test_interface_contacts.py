import numpy as np

from negfpy.core import transmission
from negfpy.core.types import Device1D, LeadBlocks
from negfpy.models import ChainParams, device_perfect_chain, lead_blocks


def test_backward_compatible_default_contact_coupling() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=18, params=params)

    t_default = transmission(omega=1.0, device=device, lead_left=lead, lead_right=lead, eta=1e-8)
    t_explicit = transmission(
        omega=1.0,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        device_to_lead_left=lead.d01.conj().T,
        device_to_lead_right=lead.d01,
        contact_left_indices=[0],
        contact_right_indices=[17],
    )
    assert abs(t_default - t_explicit) < 1e-12


def test_rectangular_contact_coupling_with_dof_mismatch() -> None:
    # Lead has 1 DOF per layer; device has 2 DOF per layer.
    lead = LeadBlocks(
        d00=np.array([[2.0]], dtype=np.complex128),
        d01=np.array([[-1.0]], dtype=np.complex128),
    )

    n_layers = 4
    onsite = [np.diag([2.0, 100.0]).astype(np.complex128) for _ in range(n_layers)]
    coupling = [np.diag([-1.0, 0.0]).astype(np.complex128) for _ in range(n_layers - 1)]
    device = Device1D(onsite_blocks=onsite, coupling_blocks=coupling)

    # Device-to-lead couplings (rectangular): n_contact(=1) x n_lead(=1).
    vdl_left = np.array([[-1.0]], dtype=np.complex128)
    vdl_right = np.array([[-1.0]], dtype=np.complex128)

    # Contact only the first DOF of first/last device layers.
    left_idx = [0]
    right_idx = [6]
    tval = transmission(
        omega=1.0,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        device_to_lead_left=vdl_left,
        device_to_lead_right=vdl_right,
        contact_left_indices=left_idx,
        contact_right_indices=right_idx,
    )

    assert np.isfinite(tval)
    assert 0.95 < tval < 1.05


def test_kdependent_device_to_lead_callable_matches_explicit_array() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=10, params=params)

    vdl = np.array([[-1.0]], dtype=np.complex128)
    t_explicit = transmission(
        omega=1.0,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        kpar=(0.1,),
        device_to_lead_left=vdl,
        device_to_lead_right=vdl,
        contact_left_indices=[0],
        contact_right_indices=[9],
    )
    t_callable = transmission(
        omega=1.0,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        kpar=(0.1,),
        device_to_lead_left=lambda _k: vdl,
        device_to_lead_right=lambda _k: vdl,
        contact_left_indices=[0],
        contact_right_indices=[9],
    )
    assert abs(t_explicit - t_callable) < 1e-12
