"""Generated precise-contact guards, FD limits and isolated input-path tests."""
from dataclasses import replace
import hashlib
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.core.types import LeadBlocks
from negfpy.anharmonic.complete_energy import UniformBZMesh,SignedFrequencyGrid
from negfpy.anharmonic.types import HarmonicReservoir
from negfpy.anharmonic import contact_precision as precise


def caps(n=2,actions=8):
    return precise.PreciseContactCaps(max_refinement_bytes=16*1024**2,
        max_certificate_work_per_descriptor=precise.certificate.planned_work(n),
        max_total_certificate_work=actions*precise.certificate.planned_work(n),
        max_descriptor_actions=actions,max_finite_frequency_surfaces=3*actions,descriptor_seconds=20.)


def reservoir(side="left"):
    d00=np.array([[3.,.02],[.02,2.5]],complex)
    d01=np.array([[-.25,.05],[0.,-.2]],complex)
    lead=LeadBlocks(d00,d01,d01.conj().T)
    return HarmonicReservoir(lead,.4,device_to_lead=np.array([[-.2,.03],[.01,-.15]],complex),device_indices=(0,2))


@pytest.mark.parametrize("side",["left","right"])
def test_generated_descriptor_new_contract_fd_limit_and_unchanged_physical_audits(side):
    raw=reservoir(side);deadline=monotonic()+20;budget=precise.PreciseContactBudget(caps(),deadline);records=[]
    original=precise.acoustic._build_fixed_eta_contact_descriptor
    desc=precise.build_precise_descriptor(raw,side=side,ndof=3,eta_lead=.1,
        surface_gf_method="sancho_rubio",budget=budget,records=records)
    row=records[0]
    assert row["NEW_arithmetic_contract"] and row["machine_residual_certificate"]["passed"]
    assert row["finite_frequency_limit"]["passed"] and row["all_original_contact_physical_audits_passed"]
    assert row["Gamma0_cast_relative_frobenius"]<1e-14
    assert desc.surface_derivative.dtype==np.dtype(np.clongdouble)
    assert desc.contact_limit.dtype==np.dtype(np.complex128)
    assert not desc.surface_derivative.flags.writeable
    assert desc.embedded_limit.shape==(3,3)
    assert budget.counts["descriptor_completed"]==1
    assert budget.counts["finite_frequency_surface_completed"]==3
    assert row["surface_zero_completed"]==row["condition_svd_completed"]==1
    assert precise.acoustic._build_fixed_eta_contact_descriptor is original


def test_both_sides_all_q_full_signed_contact_guards_and_global_counts():
    mesh=UniformBZMesh((2,1,1));grid=SignedFrequencyGrid(.25,1)
    deadline=monotonic()+30;pc=caps(actions=4);budget=precise.PreciseContactBudget(pc,deadline)
    for side in ("left","right"):
        result=precise.build_precise_compact_contact(mesh,grid,lambda _:reservoir(side),3,.1,
            side=side,precision_caps=pc,deadline_monotonic=deadline,budget=budget,
            q_samples=9,max_bytes=16*1024**2,max_base_evaluations=1000)
        assert len(result.contact.q_reports)==2
        assert result.device_indices==(0,2)
        assert result.spectrum.retarded.shape==(2,3,2,2)
        assert np.isfinite(result.zero_limits).all()
        assert all(q["precise_contact"]["finite_frequency_limit"]["passed"] for q in result.contact.q_reports)
    assert budget.counts["descriptor_completed"]==4
    assert budget.counts["certificate_completed"]==4
    assert budget.counts["finite_frequency_surface_completed"]==12
    assert budget.counts["signed_nonzero_retarded_completed"]==8
    assert budget.counts["certificate_work_completed"]==4*precise.certificate.planned_work(2)


def test_precise_caps_reject_before_original_operator_allocation_and_fail_closed_fd(monkeypatch):
    raw=reservoir();deadline=monotonic()+10;records=[]
    pc=replace(caps(),max_refinement_bytes=1)
    budget=precise.PreciseContactBudget(pc,deadline)
    with pytest.raises(MemoryError):
        precise.build_precise_descriptor(raw,side="left",ndof=3,eta_lead=.1,surface_gf_method="sancho_rubio",budget=budget,records=records)
    assert records[0]["surface_zero_attempted"]==0
    pc=caps();budget=precise.PreciseContactBudget(pc,deadline);records=[]
    monkeypatch.setattr(precise,"FD_RELATIVE_TOLERANCE",0.)
    with pytest.raises(ValueError,match="fixed-frequency"):
        precise.build_precise_descriptor(raw,side="left",ndof=3,eta_lead=.1,surface_gf_method="sancho_rubio",budget=budget,records=records)
    assert records[0]["machine_residual_certificate"]["passed"]
    assert not records[0]["completed"]


def test_action_expiry_and_total_budget_are_mandatory():
    with pytest.raises(TimeoutError):precise.PreciseContactBudget(caps(),monotonic()-1)
    deadline=monotonic()+10;budget=precise.PreciseContactBudget(caps(actions=1),deadline)
    budget.begin(2)
    with pytest.raises(ValueError,match="action cap"):budget.begin(2)
    with pytest.raises(ValueError):replace(caps(),descriptor_seconds=float("inf"))


def test_genuinely_complex_opposite_q_contacts_keep_all_canonical_pair_guards():
    mesh=UniformBZMesh((4,1,1));grid=SignedFrequencyGrid(.25,1)
    deadline=monotonic()+30;pc=caps(actions=8);budget=precise.PreciseContactBudget(pc,deadline)
    def at_q(point):
        s=np.sin(point[0])
        d00=np.array([[3.,.02+.03j*s],[.02-.03j*s,2.5]],complex)
        d01=np.array([[-.25+.01j*s,.05],[.02j*s,-.2]],complex)
        coupling=np.array([[-.2,.03+.01j*s],[.01-.02j*s,-.15]],complex)
        return HarmonicReservoir(LeadBlocks(d00,d01,d01.conj().T),.4,
            device_to_lead=coupling,device_indices=(0,2))
    for side in ("left","right"):
        result=precise.build_precise_compact_contact(mesh,grid,at_q,3,.1,
            side=side,precision_caps=pc,deadline_monotonic=deadline,budget=budget,
            q_samples=9,max_bytes=16*1024**2,max_base_evaluations=1000)
        assert all(report["precise_contact"]["finite_frequency_limit"]["passed"] for report in result.contact.q_reports)
        assert np.max(np.abs(result.zero_limits[1].imag))>1e-8
        np.testing.assert_allclose(result.zero_limits[3],result.zero_limits[1].conj(),rtol=1e-8,atol=1e-11)
    assert budget.counts["descriptor_completed"]==8
    assert budget.counts["finite_frequency_surface_completed"]==24
