"""Feasibility-only same-equation refinement retaining extended iterate bits.

The original binary64 operator/LU and an initial binary64 solve are reused.
Each residual and iterate update is evaluated as clongdouble.  Only correction
RHS values are rounded to complex128 for the unchanged LU solve; the iterate is
NEVER quantized back after an update.  Fixed candidates [current,1,.5,1.5] are
ranked by their extended matrix residual.  The caller must independently enclose the final machine residual and qualify
the final contact limit; this is not original-binary64 acceptance.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import monotonic

import numpy as np
import scipy.linalg as la

from . import _contact_metric as baseline


ALGORITHM_VERSION = "extended-iterate-matrix-rhs-binary64-lu-fixed-damping-v1"
BASELINE_SHA256 = "a89d0f68733c6e316a4a35d8bea6cfb0816bc1bc199eeda7af5670221f257586"
MAX_DIMENSION = 96
MAX_CORRECTIONS = 8
DAMPING_FACTORS = (1.,.5,1.5)


def algorithm_manifest():
    return {"schema":"negfpy.materialTest.contact-refinement-algorithm.v1",
        "algorithm_version":ALGORITHM_VERSION,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selected_metric_sha256":hashlib.sha256(Path(baseline.__file__).read_bytes()).hexdigest(),
        "maximum_corrections":MAX_CORRECTIONS,"damping_factors":DAMPING_FACTORS,
        "current_iterate_included_in_minimum":True,
        "iterate_and_residual_dtype":"numpy.clongdouble; no per-update complex128 quantization",
        "correction_solver":"unchanged binary64 operator; one cached SciPy LU",
        "ranking":"extended associated matrix residual; current retained on ties",
        "stopping_rule":"fixed correction cap or exactly zero computed extended residual",
        "NEW_arithmetic_contract":True,"original_binary64_acceptance_replaced_in_production":False,
        "original_binary64_guard":"recorded separately on cast complex128 candidate; never overridden",
        "future_Gamma0_complex128_cast":"unqualified; needs independent finite-frequency limit check before any production use",
        "production_descriptor_or_adapter":False,"physical_input_or_eta_or_projection_changed":False}


def _check(deadline):
    if isinstance(deadline,(bool,np.bool_)) or not isinstance(deadline,(int,float,np.floating)) or not np.isfinite(deadline):
        raise ValueError("finite explicit monotonic deadline required")
    if monotonic()>=deadline:
        raise TimeoutError("Extended-iterate refinement reached its shared deadline")


def planned_numeric_bytes(n):
    if type(n) is not int or not 1<=n<=MAX_DIMENSION:
        raise ValueError("extended-iterate dimension must lie in 1..96")
    # Includes supplied dense operator plus cached LU and matrix scratch;
    # excludes Python/runtime/library-internal workspace, not an RSS promise.
    return 32*n**4+1024*n*n+4096


def _extended_error(x,g,a,b):
    product = g@a@x@b@g
    square = g@g
    error = x-product+square
    def norm(value):
        return np.sqrt(np.sum(value.real*value.real+value.imag*value.imag,dtype=np.longdouble))
    denominator = norm(x)+norm(product)+norm(square)
    if not np.isfinite(error).all() or not np.isfinite(denominator) or denominator<=0:
        raise ValueError("finite positive extended residual normalization required")
    rho = norm(error)/denominator
    if not np.isfinite(rho):
        raise ValueError("nonfinite extended-iterate residual")
    return error,rho


def _metric(x,g,a,b,g128,a128,b128,condition):
    error,rho = _extended_error(x,g,a,b)
    cast = np.asarray(x,dtype=np.complex128)
    if not np.isfinite(cast).all():
        raise ValueError("extended candidate cannot be represented as finite complex128 for the separate old screen")
    _,old = baseline._equation_metrics(cast,g128,a128,b128,condition)
    if not np.isfinite(old["condition_times_residual"]):
        raise ValueError("separate cast binary64 screen is nonfinite")
    product = np.longdouble(condition)*rho
    return error,rho,{"extended_matrix_residual":str(rho),
        "estimated_condition_times_extended_residual":str(product),
        "extended_approximate_screen_passed":bool(rho<=np.longdouble("1e-10") and product<=np.longdouble("1e-8")),
        "extended_metric_is_interval_certificate":False,
        "original_binary64_on_cast":old}


def refine_extended_iterate(operator,rhs,g0,a,b,condition,*,max_bytes,
        deadline_monotonic,records=None,max_corrections=MAX_CORRECTIONS):
    """Return the bounded extended candidate, with separate honest old metrics.

    Return does not assert either screen passed: the caller must independently
    enclose the returned exact machine components using the NEW certificate.
    Only binary64 input operators are admitted; all operator bindings/condition
    remain the caller's original ones.  No material arrays are loaded here.
    """
    _check(deadline_monotonic)
    if type(max_corrections) is not int or not 0<=max_corrections<=MAX_CORRECTIONS:
        raise ValueError("correction cap must be a Python integer in 0..8")
    if np.finfo(np.longdouble).nmant<=np.finfo(np.float64).nmant:
        raise RuntimeError("extended-iterate feasibility requires precision beyond binary64")
    if isinstance(condition,(bool,np.bool_)) or not np.isfinite(condition) or not 1<=condition<1/baseline._acoustic._EPS:
        raise ValueError("original finite condition screen failed")
    if not hasattr(g0,"shape") or len(g0.shape)!=2 or g0.shape[0]!=g0.shape[1]:
        raise ValueError("g0 must be a square matrix")
    n=int(g0.shape[0]);plan=planned_numeric_bytes(n)
    if type(max_bytes) is not int or max_bytes<=0:
        raise ValueError("positive explicit max_bytes required")
    if plan>max_bytes:
        raise MemoryError("extended-iterate numeric plan exceeds cap before allocation")
    def original_array(value,shape,name):
        array=np.asarray(value)
        if (array.shape!=shape or array.dtype.kind not in "fciu"
                or array.dtype.itemsize>(16 if array.dtype.kind=="c" else 8)
                or not np.isfinite(array).all()):
            raise ValueError(name+" must be finite binary64 original data of the declared shape")
        return np.asarray(array,dtype=np.complex128)
    g128,a128,b128=(original_array(value,(n,n),name) for value,name in ((g0,"g0"),(a,"a"),(b,"b")))
    op=original_array(operator,(n*n,n*n),"operator")
    rr=original_array(rhs,(n*n,),"rhs")
    record={"algorithm_version":ALGORITHM_VERSION,"NEW_arithmetic_contract":True,
        "production_qualification":False,"physical_inputs_or_eta_changed":False,
        "lead_dimension":n,"condition_estimate":float(condition),
        "numeric_plan_bytes":plan,"numeric_plan_excludes":"Python/runtime/library-internal workspace",
        "maximum_corrections":max_corrections,"longdouble_fraction_mantissa_bits":int(np.finfo(np.longdouble).nmant),
        "initial_solve_attempted":0,"initial_solve_completed":0,
        "factorizations_attempted":0,"factorizations_completed":0,
        "correction_attempted":0,"correction_completed":0,
        "candidate_trials_attempted":0,"candidate_trials_completed":0,
        "residual_trials":[],"correction_trials":[],
        "original_binary64_guard_passed":False,"completed":False}
    if records is not None:records.append(record)
    try:
        g,aa,bb=(np.asarray(value,dtype=np.clongdouble) for value in (g128,a128,b128))
        _check(deadline_monotonic)
        record["initial_solve_attempted"]+=1
        x=np.asarray(np.linalg.solve(op,rr).reshape((n,n),order="F"),dtype=np.clongdouble)
        record["initial_solve_completed"]+=1
        error,rho,metric=_metric(x,g,aa,bb,g128,a128,b128,condition)
        record["residual_trials"].append({"corrections_completed":0,**metric})
        factor=None
        for correction in range(1,max_corrections+1):
            _check(deadline_monotonic)
            if rho==0:break
            record["correction_attempted"]+=1
            correction_rhs=-np.asarray(error,dtype=np.complex128).reshape(-1,order="F")
            if not np.isfinite(correction_rhs).all():
                raise ValueError("nonfinite rounded correction RHS")
            if factor is None:
                record["factorizations_attempted"]+=1
                factor=la.lu_factor(op,check_finite=True)
                record["factorizations_completed"]+=1
            delta=np.asarray(la.lu_solve(factor,correction_rhs,check_finite=True).reshape((n,n),order="F"),dtype=np.clongdouble)
            if not np.isfinite(delta).all():
                raise ValueError("nonfinite binary64-LU correction")
            record["correction_completed"]+=1
            choices=[(rho,x,error,metric,0.)];trials=[]
            for damping in DAMPING_FACTORS:
                _check(deadline_monotonic)
                record["candidate_trials_attempted"]+=1
                candidate=x+np.longdouble(damping)*delta
                candidate_error,candidate_rho,candidate_metric=_metric(candidate,g,aa,bb,g128,a128,b128,condition)
                record["candidate_trials_completed"]+=1
                choices.append((candidate_rho,candidate,candidate_error,candidate_metric,damping))
                trials.append({"damping_factor":damping,**candidate_metric})
            rho,x,error,metric,damping=min(choices,key=lambda row:row[0])
            record["correction_trials"].append({"correction":correction,
                "selected_damping_factor":damping,"candidates":trials})
            record["residual_trials"].append({"corrections_completed":correction,**metric})
        _check(deadline_monotonic)
        record["final_metrics"]=metric
        record["original_binary64_guard_passed"]=metric["original_binary64_on_cast"]["original_guard_passed"]
        record["extended_approximate_screen_passed"]=metric["extended_approximate_screen_passed"]
        record["iterate_dtype"]=str(x.dtype)
        record["iterate_retains_components_beyond_binary64"]=bool(np.any(x!=np.asarray(np.asarray(x,dtype=np.complex128),dtype=np.clongdouble)))
        record["completed"]=True
        return x
    except BaseException as error:
        record["failure"]={"type":type(error).__name__,"message":str(error)}
        raise
