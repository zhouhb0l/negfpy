"""Actual material IFC3 terms crossing a finite interaction crop.

Only external q=0 and external crop coordinates are stored. Both internal
coordinates keep the full harmonic-device domain. No IFC completion, projection,
thresholding, or Green-function boundary change is made here.
"""
from __future__ import annotations

import json
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

from ._canonical import domain_digest
from .layered import LayeredDeviceMapping
from .mapped_fc3 import (
    MassNormalizedTranslationalFC3,
    _effective_terms,
    _layer_tuple,
    _normalized_payload,
    _positive_integer,
    _validate_mapping_identity,
)
from .momentum import UniformBZMesh
from .transverse_boundary_cancellation import _comparison, _norms
from .transverse_mapping import _kpoints
from .transverse_material import _geometry, _position
from .transverse_packed import PackedTransverseCubicVertex, _finite, _integer
from .transverse_q import difference_q_indices
from .types import SolverScale


@dataclass(frozen=True, slots=True, eq=False)
class CrossingSupportVertex:
    """q0-only coefficients, external addresses and scalar mapping evidence."""

    coefficients: np.ndarray
    external_dofs: np.ndarray
    full_ndof: int
    metadata: dict

    def __post_init__(self):
        m = _integer(self.full_ndof, "full_ndof")
        value = self.coefficients
        addresses = self.external_dofs
        if (not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.complex128)
                or value.ndim != 4 or value.shape[0] < 1
                or value.shape[2:] != (m, m)):
            raise ValueError("Crossing coefficients must have shape(nq,A,M,M)/complex128.")
        if (not isinstance(addresses, np.ndarray) or addresses.dtype != np.dtype(np.int64)
                or addresses.shape != (value.shape[1],) or len(addresses) < 1
                or len(np.unique(addresses)) != len(addresses)
                or np.any((addresses < 0) | (addresses >= m))):
            raise ValueError("External addresses must be unique full-device int64 coordinates.")
        _finite(value, "crossing coefficients")
        value.flags.writeable = False
        addresses = addresses.copy()
        addresses.flags.writeable = False
        object.__setattr__(self, "external_dofs", addresses)


def build_crossing_support_vertex(
    mesh, fc3, device_mapping, *, n_device_layers, interaction_layers, scale,
    reference_vertex=None, max_bytes=2*1024**3, max_workspace_bytes=4*1024**3,
    max_geometry_evaluations=2_000_000, max_phase_evaluations=1_000_000_000,
    deadline_monotonic=None,
):
    """Retain every actual source term with its external leg in the crop.

    The prepass rejects required internal coordinates outside the harmonic
    device instead of clipping or wrapping them. The two internal axes are the
    full contiguous device domain. The optional independent production crop
    fixes the external coordinate order and checks its retained internal slice.
    """
    started = time.monotonic()

    def check_time():
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            raise TimeoutError("Crossing-support mapping deadline reached.")

    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite.")
    check_time()
    if not isinstance(mesh, UniformBZMesh) or mesh.shape[2] != 1:
        raise TypeError("A transverse UniformBZMesh is required.")
    if not isinstance(fc3, MassNormalizedTranslationalFC3):
        raise TypeError("A mass-normalized material FC3 is required.")
    if not isinstance(device_mapping, LayeredDeviceMapping) or not isinstance(scale, SolverScale):
        raise TypeError("A layered device mapping and solver scale are required.")
    layers = _positive_integer(n_device_layers, name="n_device_layers")
    cap = _positive_integer(max_bytes, name="max_bytes")
    workspace = _positive_integer(max_workspace_bytes, name="max_workspace_bytes")
    geometry_cap = _positive_integer(max_geometry_evaluations, name="max_geometry_evaluations")
    phase_cap = _positive_integer(max_phase_evaluations, name="max_phase_evaluations")
    crop = _layer_tuple(interaction_layers, name="interaction_layers", n_device_layers=layers)
    mapping = device_mapping
    _geometry(mapping)
    _validate_mapping_identity(fc3, mapping)
    if fc3.output_units != "J m^-3 kg^-3/2":
        raise ValueError("Material FC3 must use mass-normalized physical SI units.")
    if domain_digest("NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3", _normalized_payload(fc3)) != fc3.content_sha256:
        raise ValueError("Normalized FC3 content no longer matches its identity.")
    p, nq = mapping.principal_cells, mesh.n_qpoints
    m = layers*mapping.n_principal_layer_dofs
    minimum, maximum = crop[0]*p, (crop[-1]+1)*p-1
    start, end = crop[0]*mapping.n_principal_layer_dofs, (crop[-1]+1)*mapping.n_principal_layer_dofs
    if reference_vertex is not None:
        if (not isinstance(reference_vertex, PackedTransverseCubicVertex)
                or reference_vertex.mesh.shape != mesh.shape or reference_vertex.full_ndof != m):
            raise ValueError("Production reference mesh/domain differs.")
        active = np.asarray(reference_vertex.active_dofs)
        if np.any((active < start) | (active >= end)):
            raise ValueError("Production external coordinates lie outside the crop.")
    else:
        active = np.arange(start, end, dtype=np.int64)
    a = len(active)
    if not a:
        raise ValueError("A nonempty external crop is required.")
    coefficient_bytes = 16*nq*a*m*m
    auxiliary = 48*len(fc3.terms)+8192*nq+128*a+65536
    # Coefficients plus comparison scratch at most one q block at a time.
    numeric_plan = coefficient_bytes+128*a*m*m+auxiliary
    if coefficient_bytes+8*a > cap:
        raise ValueError("Crossing coefficients exceed max_bytes.")
    if numeric_plan > workspace:
        raise ValueError("Crossing numeric workspace plan exceeds max_workspace_bytes.")
    if len(fc3.terms)*(3 if fc3.make_r0_average else 1) > geometry_cap:
        raise ValueError("Source expansion exceeds max_geometry_evaluations.")
    effective = _effective_terms(fc3)
    nonzero = sum(bool(np.any(term.values != 0)) for term in effective)
    evaluations = 2*nonzero*len(mapping.direct_coset_representatives)
    if evaluations > geometry_cap:
        raise ValueError("Crossing source geometry work exceeds max_geometry_evaluations.")
    factor = float(np.sqrt(scale.hbar)/np.float64(scale.omega0)**2.5)
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("Material solver conversion is not representable.")
    lookup = {int(x): i for i, x in enumerate(active)}

    def bases_at(mapped, anchor):
        result = []
        for cell, basis in mapped:
            layer, subcell = divmod(int(cell[0])+anchor, p)
            result.append(layer*mapping.n_principal_layer_dofs + 3*(subcell*mapping.n_transport_basis_atoms+basis))
        return result

    def images():
        for i, term in enumerate(effective):
            if i % 256 == 0:
                check_time()
            nz = term.values != 0
            components = int(np.count_nonzero(nz))
            if not components:
                continue
            weighted = term.values*factor*(term.multiplicity_weight*term.anchor_weight*term.r0_leg_factor)
            if not np.isfinite(weighted).all() or np.any(nz & (weighted == 0)):
                raise ValueError("Weighted FC3 block is not finitely representable.")
            cartesian = [np.flatnonzero(np.any(nz, axis=tuple(j for j in range(3) if j != leg))) for leg in range(3)]
            for representative in mapping.direct_coset_representatives:
                mapped = tuple(_position(atom, position+representative, mapping)
                    for atom, position in zip(term.primitive_indices, term.positions, strict=True))
                cell0 = int(mapped[0][0][0])
                yield mapped, minimum-cell0, maximum-cell0, cartesian, components, weighted

    anchors = cropped = scalar_terms = 0
    outside = [0, 0]
    internal_min, internal_max = None, None
    # Count all actual required images before allocating a new field.
    for mapped, low, high, cartesian, components, _ in images():
        for anchor in range(low, high+1):
            bases = bases_at(mapped, anchor)
            if not any(bases[0]+int(c) in lookup for c in cartesian[0]):
                continue
            anchors += 1
            scalar_terms += components
            cells = [int(position[0][0])+anchor for position in mapped]
            if all(minimum <= cell <= maximum for cell in cells):
                cropped += 1
            for leg in (1, 2):
                internal_min = cells[leg] if internal_min is None else min(internal_min, cells[leg])
                internal_max = cells[leg] if internal_max is None else max(internal_max, cells[leg])
                outside[leg-1] += int(not 0 <= cells[leg] < layers*p)
    support = {"required_anchor_images": anchors, "all_leg_cropped_anchor_images": cropped,
        "crossing_anchor_images": anchors-cropped, "outside_device_counts_by_internal_leg": outside,
        "required_internal_longitudinal_cell_min": internal_min,
        "required_internal_longitudinal_cell_max": internal_max,
        "available_longitudinal_cells": [0, layers*p-1], "complete_internal_support": not any(outside)}
    if any(outside):
        raise ValueError("Crossing support exceeds the harmonic device: "+json.dumps(support, sort_keys=True))
    phase_work = nq*scalar_terms
    if phase_work > phase_cap:
        raise ValueError("Crossing Fourier work exceeds max_phase_evaluations.")
    check_time()
    values = np.zeros((nq, a, m, m), dtype=np.complex128)
    routes = difference_q_indices(mesh, max_bytes=cap)
    negative = routes[0]
    kpoints = _kpoints(mesh)
    qs = np.arange(nq)
    for mapped, low, high, cartesian, _, weighted in images():
        second = tuple((int(b)-int(c)) % period for c,b,period in zip(mapped[0][0][1:],mapped[1][0][1:],mesh.shape[:2],strict=True))
        third = tuple((int(b)-int(c)) % period for c,b,period in zip(mapped[0][0][1:],mapped[2][0][1:],mesh.shape[:2],strict=True))
        phases = np.exp(1j*(kpoints@second))*np.exp(1j*(kpoints@third))[negative]
        for anchor in range(low, high+1):
            bases = bases_at(mapped, anchor)
            ext_components = np.asarray([int(c) for c in cartesian[0] if bases[0]+int(c) in lookup], dtype=int)
            if not len(ext_components):
                continue
            external = np.asarray([lookup[bases[0]+int(c)] for c in ext_components])
            internal = [bases[leg]+cartesian[leg] for leg in (1,2)]
            block = weighted[np.ix_(ext_components, cartesian[1], cartesian[2])]
            values[np.ix_(qs, external, *internal)] += phases[:,None,None,None]*block[None]
    _finite(values, "mapped crossing coefficients")
    reality, exchange, crop_checks = [], [], []
    for q in range(nq):
        check_time()
        reality.append(_comparison(values[q], values[int(negative[q])].conj()))
        exchange.append(_comparison(values[q], values[int(negative[q])].transpose(0,2,1)))
        if reference_vertex is not None:
            crop_checks.append(_comparison(values[q][:,active][:,:,active], reference_vertex.coefficients[0,q]))
    if not all(row["passed"] for row in reality+exchange+crop_checks):
        raise ValueError("Crossing reality, internal-leg exchange or production crop identity failed.")
    masses = np.asarray(mapping._crystal.primitive.masses)[mapping.basis_primitive_atoms]
    full_masses = np.tile(masses, layers*p)
    translations = np.zeros((m,3))
    translations[np.arange(m), np.arange(m)%3] = np.repeat(np.sqrt(full_masses/np.sum(full_masses)),3)
    acoustic = []
    for leg, expression in ((1,"acd,c->ad"),(2,"acd,d->ac")):
        for direction in range(3):
            acoustic.append({"internal_leg":leg,"direction":direction,
                **_norms(np.einsum(expression,values[0],translations[:,direction],optimize=False))})
    metadata = {"schema":"negfpy.crossing-support-vertex.v1","external_q_index":0,
        "mesh_shape":list(mesh.shape),"external_ndof":a,"internal_ndof":m,
        "interaction_layers":list(crop),"n_device_layers":layers,"support":support,
        "source_fc3_sha256":fc3.source_fc3_content_sha256,"normalized_fc3_sha256":fc3.content_sha256,
        "geometry_sha256":mapping.layered_geometry_sha256,"solver_factor":factor,
        "multiplicity_anchor_r0_weights_applied_once":True,"coupling_applied":False,
        "projection":False,"ifc_correction":False,"longitudinal_wrapping":False,
        "geometry_evaluations":evaluations,"phase_component_visits":phase_work,
        "coefficient_bytes":coefficient_bytes,"numeric_workspace_plan_bytes":numeric_plan,
        "caller_reference_payload_excluded":True,"reality":reality,"internal_leg_exchange":exchange,
        "production_crop_comparison":crop_checks,"identity_rtol":1e-8,"identity_atol":1e-11,
        "internal_gamma_translation_diagnostic_only":acoustic,
        "external_internal_full_permutation_tested":False,"elapsed_seconds":time.monotonic()-started}
    return CrossingSupportVertex(values,active,m,metadata)


def contract_rectangular_bins(
    first_vertex, green_left, green_right, *, external_tile=16, workers=8, bin_tile=1,
    max_bytes=2*1024**3, max_contraction_work=1_000_000_000_000,
):
    """Unweighted V[A,M,M] X[B,M,M] Y[B,M,M] conjugate(V), with bounded scratch."""
    tile = _integer(external_tile,"external_tile")
    workers = _integer(workers,"workers")
    bins = _integer(bin_tile,"bin_tile")
    cap = _integer(max_bytes,"max_bytes")
    work_cap = _integer(max_contraction_work,"max_contraction_work")
    if (not isinstance(first_vertex,np.ndarray) or first_vertex.ndim != 3
            or first_vertex.dtype != np.dtype(np.complex128)):
        raise ValueError("first_vertex must be complex128 with shape(A,M,M).")
    a,m,last = first_vertex.shape
    if a < 1 or m < 1 or last != m:
        raise ValueError("Rectangular vertex dimensions must be nonempty and internally square.")
    if (not isinstance(green_left,np.ndarray) or green_left.ndim != 3
            or green_left.shape[0] < 1):
        raise ValueError("green_left must have shape(B,M,M), B>=1.")
    b = green_left.shape[0]
    for value,name in ((green_left,"green_left"),(green_right,"green_right")):
        if not isinstance(value,np.ndarray) or value.shape != (b,m,m) or value.dtype != np.dtype(np.complex128):
            raise ValueError(name+" must be complex128 with shape(B,M,M).")
    effective = min(workers,(b+bins-1)//bins)
    t = min(tile,a)
    owned = 16*(b*a*a+a*m*m+effective*(2*t*m*m+t*a))+m*m
    work = b*(2*a*m**3+a*a*m*m)
    if owned > cap:
        raise ValueError("Rectangular output/workspace exceeds max_bytes.")
    if work > work_cap:
        raise ValueError("Rectangular contraction exceeds max_contraction_work.")
    for value,name in ((first_vertex,"first_vertex"),(green_left,"green_left"),(green_right,"green_right")):
        _finite(value,name)
    conjugate_payload = np.empty((a,m,m),dtype=np.complex128)
    np.conjugate(first_vertex,out=conjugate_payload)
    conjugate = conjugate_payload.reshape(a,m*m).T
    output = np.zeros((b,a,a),dtype=np.complex128)

    def chunk(start,stop):
        left = np.empty((t,m,m),dtype=np.complex128)
        both = np.empty_like(left)
        final = np.empty((t,a),dtype=np.complex128)
        for index in range(start,stop):
            for begin in range(0,a,t):
                end = min(begin+t,a)
                width = end-begin
                np.matmul(green_left[index].T,first_vertex[begin:end],out=left[:width])
                np.matmul(left[:width],green_right[index],out=both[:width])
                np.matmul(both[:width].reshape(width,m*m),conjugate,out=final[:width])
                output[index,begin:end] = final[:width]

    with threadpool_limits(limits=1,user_api="blas"):
        if effective == 1:
            for start in range(0,b,bins):
                chunk(start,min(start+bins,b))
        else:
            with ThreadPoolExecutor(max_workers=effective) as executor:
                pending = deque()
                for start in range(0,b,bins):
                    if len(pending) == effective:
                        pending.popleft().result()
                    pending.append(executor.submit(chunk,start,min(start+bins,b)))
                for future in pending:
                    future.result()
    _finite(output,"rectangular output")
    return output
