"""NEW derivative arithmetic contract, unchanged fixed-eta contact physics.

Extended X is interval-screened, coupling is contracted in extended precision,
and only final Gamma/omega is cast to complex128 before the canonical physical
audits.  The separately recorded binary64 derivative guard may FAIL.  Canonical
module namespaces, frozen helpers, eta and physical matrices are never changed.
"""
from dataclasses import dataclass,asdict,replace
from decimal import Decimal
import hashlib,json,math
from pathlib import Path
from time import monotonic
from types import MappingProxyType

import numpy as np

from functools import partial

from . import acoustic_contacts as acoustic
from . import transverse_contacts as transverse
from . import transverse_compact_contacts as compact
from . import contact_extended_refinement as refinement
from . import contact_machine_residual_certificate as certificate

ALGORITHM_VERSION="extended-derivative-machine-interval-final-Gamma128-v1"
FD_FRACTIONS=(.02,.01,.005)
FD_RELATIVE_TOLERANCE=1e-3
CAST_RELATIVE_TOLERANCE=1e-14


@dataclass(frozen=True)
class PreciseContactCaps:
    max_refinement_bytes:int
    max_certificate_work_per_descriptor:int
    max_total_certificate_work:int
    max_descriptor_actions:int
    max_finite_frequency_surfaces:int
    descriptor_seconds:float

    def __post_init__(self):
        for name,value in asdict(self).items():
            if name!="descriptor_seconds" and (type(value) is not int or value<1):
                raise ValueError(name+" must be an explicit positive integer")
        if isinstance(self.descriptor_seconds,bool) or not math.isfinite(self.descriptor_seconds) or self.descriptor_seconds<=0:
            raise ValueError("explicit finite positive descriptor_seconds required")




def algorithm_manifest():
    return {"schema":"negfpy.materialTest.precise-contact-arithmetic.v1",
        "algorithm_version":ALGORITHM_VERSION,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_sources_sha256":{Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (refinement,certificate,acoustic,transverse,compact)},"NEW_arithmetic_contract":True,
        "replaced_screen":"binary64 associated derivative residual -> decimal60 interval upper residual of exact machine X",
        "thresholds":["1e-10","estimated condition*residual <=1e-8"],
        "original_binary64_derivative_guard":"reported separately on cast X; may fail and is never relabeled PASS",
        "physical_eta_matrices_and_guards_changed":False,"canonical_namespaces_changed":False,
        "Gamma0_arithmetic":"extended derivative and coupling contraction, one final complex128 cast before physical audits",
        "finite_frequency_fractions_of_eta":FD_FRACTIONS,
        "finite_frequency_relative_tolerance":FD_RELATIVE_TOLERANCE,
        "cast_relative_tolerance":CAST_RELATIVE_TOLERANCE,
        "condition_is_certified_bound":False,"true_forward_error_certified":False}


def _check(deadline):
    if isinstance(deadline,(bool,np.bool_)) or not isinstance(deadline,(int,float,np.floating)) or not np.isfinite(deadline):
        raise ValueError("finite explicit monotonic deadline required")
    if monotonic()>=deadline:raise TimeoutError("precise contact reached its shared/action deadline")


class PreciseContactBudget:
    def __init__(self,caps,deadline):
        if not isinstance(caps,PreciseContactCaps):raise TypeError("explicit PreciseContactCaps required")
        _check(deadline);self.caps=caps;self.deadline=deadline
        self.counts={key:0 for key in ("descriptor_attempted","descriptor_completed",
            "certificate_attempted","certificate_completed","certificate_work_reserved",
            "certificate_work_completed","finite_frequency_surface_attempted",
            "finite_frequency_surface_completed","signed_nonzero_retarded_attempted",
            "signed_nonzero_retarded_completed","classification_attempted","classification_completed")}

    def dimension_plan(self,n):
        work=certificate.planned_work(n);numeric=refinement.planned_numeric_bytes(n)
        if work>self.caps.max_certificate_work_per_descriptor or numeric>self.caps.max_refinement_bytes:
            raise MemoryError("precise derivative per-action work/numeric admission failed")
        return work,numeric

    def begin(self,n):
        _check(self.deadline);work,numeric=self.dimension_plan(n)
        if self.counts["descriptor_attempted"]+1>self.caps.max_descriptor_actions:
            raise ValueError("precise descriptor action cap exceeded")
        if self.counts["certificate_work_reserved"]+work>self.caps.max_total_certificate_work:
            raise ValueError("precise total certificate work cap exceeded")
        if self.counts["finite_frequency_surface_attempted"]+3>self.caps.max_finite_frequency_surfaces:
            raise ValueError("precise finite-frequency surface cap exceeded")
        self.counts["descriptor_attempted"]+=1;self.counts["certificate_work_reserved"]+=work
        return min(self.deadline,monotonic()+self.caps.descriptor_seconds),work,numeric


@dataclass(frozen=True)
class PreciseDescriptor:
    """Duck-typed canonical consumer payload, retaining extended X read-only."""
    data:object
    def __getattr__(self,name):
        try:return self.data[name]
        except KeyError as error:raise AttributeError(name) from error


def _norm_extended(value):
    return np.sqrt(np.sum(value.real*value.real+value.imag*value.imag,dtype=np.longdouble))


def _relative(error,scale):
    if scale==0:return 0. if error==0 else None
    return float(error/scale)


def _machine_hash(array,deadline):
    h=hashlib.sha256()
    for row in array:
        _check(deadline)
        for value in row:
            for part in (value.real,value.imag):
                n,d=part.as_integer_ratio();h.update((format(n,"x")+"/"+format(d,"x")+";").encode("ascii"))
    return h.hexdigest()


def build_precise_descriptor(reservoir,*,side,ndof,eta_lead,surface_gf_method,
        spectral_classification=None,budget,records):
    """Explicit same-physics descriptor with the declared new derivative screen."""
    ac=acoustic;record={"side":side,"q_index":len(records),
        "NEW_arithmetic_contract":True,"physical_guards_retained":True,
        "original_binary64_derivative_guard_is_acceptance_authority":False,
        "refinement_records":[],"completed":False,"surface_zero_attempted":0,"surface_zero_completed":0,
        "condition_svd_attempted":0,"condition_svd_completed":0}
    records.append(record)
    try:
        n=int(reservoir.lead.d00.shape[0]);deadline,work,numeric=budget.begin(n)
        record.update(certificate_work_reserved=work,refinement_numeric_plan_bytes=numeric,
                      action_seconds_cap=budget.caps.descriptor_seconds)
        if not np.isfinite(eta_lead) or eta_lead<=0:raise ValueError("finite positive eta required")
        spectral=spectral_classification or ac._classify_lead_spectrum(reservoir.lead)
        if not spectral.passed or spectral.classification not in {"acoustic","gapped"}:
            raise ValueError("original lead spectral classification failed")
        if spectral.lead_physics_sha256!=ac.harmonic_lead_physics_sha256(reservoir.lead):
            raise ValueError("cached original lead classification differs")
        method=surface_gf_method.strip().lower();a,b=ac._surface_orientation(reservoir.lead,side)
        def surface(omega):
            _check(deadline)
            value=(ac._analytic_diagonal_surface_zero(reservoir.lead,eta_lead)
                if omega==0 and method in {"analytic_diagonal","analytic-diagonal"} else
                ac.surface_gf(omega=omega,d00=np.asarray(reservoir.lead.d00,dtype=np.complex128),
                    d01=a,d10=b,eta=float(eta_lead),method=surface_gf_method))
            _check(deadline)
            if not np.isfinite(value).all():raise ValueError("nonfinite original surface Green function")
            return value
        record["surface_zero_attempted"]+=1;g0=surface(0.);record["surface_zero_completed"]+=1
        sop=-float(eta_lead)**2*np.eye(n,dtype=np.complex128)-reservoir.lead.d00-a@g0@b
        se=sop@g0-np.eye(n,dtype=np.complex128)
        sr=float(np.linalg.norm(se,"fro"))/(float(np.linalg.norm(sop@g0,"fro"))+float(np.sqrt(n)))
        if sr>1e-10:raise ValueError("original surface fixed-point guard failed")
        _check(deadline)
        operator=np.eye(n*n,dtype=np.complex128)-np.kron((b@g0).T,g0@a)
        rhs=(-g0@g0).reshape(-1,order="F")
        record["condition_svd_attempted"]+=1;condition=float(np.linalg.cond(operator));record["condition_svd_completed"]+=1
        if not np.isfinite(condition) or condition>=1/ac._EPS:raise ValueError("original finite condition screen failed")
        x=refinement.refine_extended_iterate(operator,rhs,g0,a,b,condition,
            max_bytes=budget.caps.max_refinement_bytes,deadline_monotonic=deadline,
            records=record["refinement_records"],max_corrections=8)
        budget.counts["certificate_attempted"]+=1
        cert=certificate.certify_machine_contact_residual(g0,a,b,x,condition_estimate=condition,
            max_work=work,deadline_monotonic=deadline,max_dimension=96)
        budget.counts["certificate_completed"]+=1;budget.counts["certificate_work_completed"]+=cert.work_completed
        record["machine_residual_certificate"]=cert.as_dict()
        record["original_binary64_derivative_guard_on_cast"]=record["refinement_records"][-1]["final_metrics"]["original_binary64_on_cast"]
        if not cert.passed:raise ValueError("NEW exact-machine interval derivative screen failed")
        coupling=ac._device_to_lead_coupling(reservoir,side)
        cl=np.asarray(coupling,dtype=np.clongdouble)
        prime=cl@(np.clongdouble(2j)*np.longdouble(eta_lead)*x)@cl.conj().T
        extended_limit=np.clongdouble(1j)*(prime-prime.conj().T)
        contact_limit=np.asarray(extended_limit,dtype=np.complex128)  # sole final Gamma0 cast
        if not np.isfinite(extended_limit).all() or not np.isfinite(contact_limit).all():
            raise ValueError("nonfinite extended/final-cast contact limit")
        cast_error=_norm_extended(extended_limit-np.asarray(contact_limit,dtype=np.clongdouble))
        cast_scale=_norm_extended(extended_limit);cast_relative=_relative(cast_error,cast_scale)
        record["Gamma0_cast_relative_frobenius"]=cast_relative
        if cast_relative is None or cast_relative>CAST_RELATIVE_TOLERANCE:
            raise ValueError("final Gamma0 complex128 cast failed its explicit precision check")
        # Additional independent limit qualification, for every q AND side.
        values=[];fd=[];scale=float(np.linalg.norm(contact_limit,"fro"))
        for fraction in FD_FRACTIONS:
            _check(deadline);budget.counts["finite_frequency_surface_attempted"]+=1
            gw=surface(float(eta_lead)*fraction)
            budget.counts["finite_frequency_surface_completed"]+=1
            sigma=coupling@gw@coupling.conj().T
            value=1j*(sigma-sigma.conj().T)/(float(eta_lead)*fraction)
            if not np.isfinite(value).all():raise ValueError("nonfinite independent finite-frequency contact limit")
            values.append(value);fd.append({"omega_solver":float(eta_lead)*fraction,
                "relative_frobenius":_relative(float(np.linalg.norm(value-contact_limit,"fro")),scale)})
        coarse=(4*values[1]-values[0])/3;fine=(4*values[2]-values[1])/3
        err=float(np.linalg.norm(fine-contact_limit,"fro"));agreement=float(np.linalg.norm(fine-coarse,"fro"))
        relative=_relative(err,scale);agreement_relative=_relative(agreement,scale)
        passed=(relative is not None and agreement_relative is not None
                and relative<=FD_RELATIVE_TOLERANCE and agreement_relative<=FD_RELATIVE_TOLERANCE)
        record["finite_frequency_limit"]={"checks":fd,"relative_tolerance":FD_RELATIVE_TOLERANCE,
            "fine_Richardson_relative_error":relative,"two_Richardson_agreement_relative":agreement_relative,
            "passed":bool(passed),"all_q_and_both_sides_required":True}
        if not passed:raise ValueError("independent fixed-frequency Gamma0 limit check failed")
        # The following physical audits preserve the canonical equations and
        # tolerances exactly, applied to the once-cast observable Gamma0.
        indices=ac.normalize_contact_indices(reservoir,ndof,side)
        embedded=ac.embed_contact_block(contact_limit,indices,ndof)
        limit_scale=max(float(np.linalg.norm(contact_limit,ord=2)),1.)
        tolerance=1e-10*limit_scale+1e-12
        herm=float(np.max(np.abs(contact_limit-contact_limit.conj().T)))
        real=max(float(np.max(np.abs(contact_limit.imag))),float(np.max(np.abs(contact_limit-contact_limit.T))))
        mineig=float(np.min(np.linalg.eigvalsh((contact_limit+contact_limit.conj().T)/2)))
        if herm>tolerance:raise ValueError("original Gamma0 Hermiticity audit failed")
        lead_real=all(float(np.max(np.abs(np.asarray(v).imag)))<=100*ac._EPS*max(1.,spectral.full_scale)
            for v in (reservoir.lead.d00,reservoir.lead.d01,
                      reservoir.lead.d10 if reservoir.lead.d10 is not None else reservoir.lead.d01.conj().T,coupling))
        if lead_real and real>tolerance:raise ValueError("original real-contact Gamma0 symmetry audit failed")
        if mineig < -tolerance:raise ValueError("original Gamma0 passivity audit failed")
        zero_sigma=coupling@g0@coupling.conj().T
        embedded_zero=ac.embed_contact_block(zero_sigma,indices,ndof)
        zero_res=float(np.linalg.norm(1j*(zero_sigma-zero_sigma.conj().T),ord=2))/max(float(np.linalg.norm(zero_sigma,ord=2)),1.)
        if zero_res>1e-10:raise ValueError("original exact-zero retarded Hermiticity audit failed")
        rfloat=float(cert.rho_upper)
        if Decimal.from_float(rfloat)<cert.rho_upper:rfloat=float(np.nextafter(rfloat,np.inf))
        coupling_sha=ac.domain_digest("negfpy-contact-coupling-v1",np.array(coupling,dtype=np.complex128,order="C"))
        indices_sha=ac.domain_digest("negfpy-contact-indices-v1",np.array(indices,dtype=np.int64,order="C"))
        record["surface_derivative_machine_ratio_sha256"]=_machine_hash(x,deadline)
        certificate_identity={key:value for key,value in cert.as_dict().items() if key!="elapsed_seconds"}
        digest_payload={"arithmetic_version":ALGORITHM_VERSION,"side":side,"eta_lead":float(eta_lead),
            "lead_sha256":spectral.lead_physics_sha256,"coupling_sha256":coupling_sha,
            "indices_sha256":indices_sha,"machine_X_sha256":record["surface_derivative_machine_ratio_sha256"],
            "interval_certificate":json.dumps(certificate_identity,sort_keys=True),"Gamma0":contact_limit,"g0":g0}
        descriptor_sha=ac.domain_digest("negfpy-materialTest-precise-contact-descriptor-v1",digest_payload)
        data=dict(side=side,orientation="outward-left" if side=="left" else "increasing-right",
            eta_lead=float(eta_lead),surface_gf_method=method,lead_physics_sha256=spectral.lead_physics_sha256,
            coupling_sha256=coupling_sha,device_indices_sha256=indices_sha,spectral_classification=spectral,
            g0=g0,surface_derivative=x,embedded_zero_retarded=embedded_zero,contact_limit=contact_limit,
            embedded_limit=embedded,surface_equation_residual=sr,derivative_residual=rfloat,
            derivative_condition=condition,hermiticity_residual=herm,real_symmetry_residual=real,
            passivity_minimum_eigenvalue=mineig,analytic_zero_residual=zero_res,descriptor_sha256=descriptor_sha)
        for key in ("g0","surface_derivative","embedded_zero_retarded","contact_limit","embedded_limit"):
            data[key]=np.array(data[key],copy=True);data[key].setflags(write=False)
        _check(deadline);record["all_original_contact_physical_audits_passed"]=True
        record["completed"]=True;budget.counts["descriptor_completed"]+=1
        return PreciseDescriptor(MappingProxyType(data))
    except BaseException as error:
        record["failure"]={"type":type(error).__name__,"message":str(error)};raise


def build_precise_compact_contact(mesh,grid,reservoirs,ndof,eta_lead,*,side,
        precision_caps,deadline_monotonic,budget=None,surface_gf_method="sancho_rubio",
        q_samples=257,max_bytes=64*1024**2,max_base_evaluations=100000):
    _check(deadline_monotonic)
    budget=PreciseContactBudget(precision_caps,deadline_monotonic) if budget is None else budget
    if budget.caps!=precision_caps or budget.deadline!=deadline_monotonic:
        raise ValueError("shared precise-contact budget binding differs")
    records=[]
    def descriptor(*args,**kwargs):return build_precise_descriptor(*args,**kwargs,budget=budget,records=records)
    def classify(lead,**kwargs):
        _check(deadline_monotonic);budget.dimension_plan(int(lead.d00.shape[0]))
        budget.counts["classification_attempted"]+=1
        result=acoustic._classify_lead_spectrum(lead,**kwargs)
        budget.counts["classification_completed"]+=1;_check(deadline_monotonic);return result
    original_retarded=transverse.retarded_contact_self_energy
    def retarded(*args,**kwargs):
        _check(deadline_monotonic);budget.counts["signed_nonzero_retarded_attempted"]+=1
        result=original_retarded(*args,**kwargs)
        budget.counts["signed_nonzero_retarded_completed"]+=1;_check(deadline_monotonic);return result
    contact_builder=partial(transverse.build_transverse_contact,
        _descriptor_builder=descriptor,_spectrum_classifier=classify,
        _retarded_builder=retarded)
    try:
        result=compact.build_compact_transverse_contact(mesh,grid,reservoirs,ndof,eta_lead,
            _contact_builder=contact_builder,side=side,surface_gf_method=surface_gf_method,
            q_samples=q_samples,max_bytes=max_bytes,max_base_evaluations=max_base_evaluations)
        _check(deadline_monotonic)
    except BaseException as error:
        error.refinement_records=records;error.refinement_algorithm=algorithm_manifest()
        error.precise_contact_counts=dict(budget.counts);raise
    reports=tuple({**dict(report),"NEW_derivative_arithmetic_contract":True,"precise_contact":row}
        for report,row in zip(result.contact.q_reports,records,strict=True))
    return replace(result,contact=replace(result.contact,q_reports=reports))
