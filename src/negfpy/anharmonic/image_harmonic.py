"""Exact finite-range, mixed-mass harmonic interfaces for saved complete-energy data.

IFC image ``R`` means Phi[atom in cell 0, atom in cell R], in eV/Angstrom^2.
The first lattice row is the transport cell; transverse q components are
dimensionless radians. No minimum-image folding, ASR, PSD repair or range
cropping is performed here. Imports do not read material inputs or run NEGF.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np

EV_J = 1.602176634e-19
ANGSTROM_M = 1e-10
AMU_KG = 1.66053906660e-27
HBAR_J_S = 1.054571817e-34
KB_J_K = 1.380649e-23
OMEGA0_RAD_S = 2 * np.pi * 1e13
IFC2_SCHEMA = "negfpy.material-test-ifc2.v1"


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name, *, positive=False):
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _adjoint(value):
    return value.conj().T


@dataclass(frozen=True)
class IFC2Data:
    images: np.ndarray
    blocks_ev_A2: np.ndarray
    cell_rows_angstrom: np.ndarray
    fractional_positions: np.ndarray
    symbols: tuple[str, ...]
    masses_amu: np.ndarray
    metadata: dict
    source_path: str
    source_sha256: str

    @property
    def image_blocks(self):
        return {tuple(int(x) for x in image): block for image, block in zip(self.images, self.blocks_ev_A2, strict=True)}


def load_ifc2(path, *, max_bytes=512 * 1024**2) -> IFC2Data:
    """Load a complete image-resolved IFC2 artifact after bounded shape checks.

    The returned arrays retain raw force constants. Reciprocity is measured
    separately by ImageHarmonics; any requested averaging is explicit there.
    """
    import h5py
    source = Path(path).expanduser().resolve()
    cap = _integer(max_bytes, "max_bytes")
    if source.stat().st_size > cap:
        raise ValueError("IFC2 file exceeds max_bytes before array loading")
    with h5py.File(source, "r") as handle:
        from .io._saved_hdf import require_internal_storage
        require_internal_storage(handle)
        schema = handle.attrs.get("schema")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        published = handle.attrs.get("complete", False)
        if schema != IFC2_SCHEMA or not isinstance(published, (bool,np.bool_)) or not bool(published):
            raise ValueError("IFC2 schema or complete publication flag is invalid")
        required = ("images", "blocks_ev_A2", "cell_rows_angstrom", "fractional_positions", "symbols", "masses_amu")
        if any(name not in handle for name in required):
            raise ValueError("IFC2 artifact is missing a required dataset")
        images, blocks, masses = handle["images"], handle["blocks_ev_A2"], handle["masses_amu"]
        if len(images.shape) != 2 or images.shape[1] != 3 or images.dtype.kind not in "iu":
            raise ValueError("IFC2 images must have integer shape (N,3)")
        if len(masses.shape) != 1 or not masses.shape[0]:
            raise ValueError("IFC2 masses must be a nonempty vector")
        nb = masses.shape[0]
        if blocks.shape != (images.shape[0], 3*nb, 3*nb) or blocks.dtype.kind not in "fi":
            raise ValueError("IFC2 block dimensions or real dtype differ from the mass basis")
        if handle["cell_rows_angstrom"].shape != (3, 3) or handle["fractional_positions"].shape != (nb, 3) or handle["symbols"].shape != (nb,):
            raise ValueError("IFC2 crystal dataset dimensions differ")
        payload = sum(int(handle[name].size * handle[name].dtype.itemsize) for name in required)
        if payload > cap:
            raise ValueError("IFC2 declared numeric payload exceeds max_bytes")
        raw_meta = handle.attrs.get("metadata_json", handle.attrs.get("metadata", "{}"))
        if isinstance(raw_meta, bytes):
            raw_meta = raw_meta.decode("utf-8")
        metadata = json.loads(str(raw_meta))
        if not isinstance(metadata, dict):
            raise ValueError("IFC2 metadata must be a JSON object")
        data = dict(images=_readonly(images[:], np.int64), blocks_ev_A2=_readonly(blocks[:], float),
                    cell_rows_angstrom=_readonly(handle["cell_rows_angstrom"][:], float),
                    fractional_positions=_readonly(handle["fractional_positions"][:], float),
                    symbols=tuple(str(x) for x in handle["symbols"].asstr()[:]),
                    masses_amu=_readonly(masses[:], float))
    for name in ("blocks_ev_A2", "cell_rows_angstrom", "fractional_positions", "masses_amu"):
        if not np.isfinite(data[name]).all():
            raise ValueError("IFC2 contains nonfinite " + name)
    if np.any(data["masses_amu"] <= 0) or abs(np.linalg.det(data["cell_rows_angstrom"])) < 1e-12:
        raise ValueError("IFC2 masses or crystal cell are invalid")
    if any(not symbol or not symbol.isalpha() for symbol in data["symbols"]):
        raise ValueError("IFC2 symbols must be nonempty element labels")
    if len(set(map(tuple, data["images"].tolist()))) != len(data["images"]):
        raise ValueError("IFC2 image labels must be unique")
    sha = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            sha.update(chunk)
    return IFC2Data(**data, metadata=metadata, source_path=str(source), source_sha256=sha.hexdigest())


@dataclass(frozen=True)
class HarmonicAtQ:
    dynamical: np.ndarray
    device: object
    lead: object
    device_to_lead_left: np.ndarray
    device_to_lead_right: np.ndarray
    contact_left_indices: np.ndarray
    contact_right_indices: np.ndarray
    q_radians: tuple[float, float]
    n_device_cells: int
    principal_cells: int
    omega0_rad_s: float



class ImageHarmonics:
    """Same-potential finite device and exact semi-infinite homogeneous leads.

    ``images`` is a mapping (Rx,Ry,Rz) -> real IFC2 block in eV/Angstrom².
    Principal-layer thickness is the largest *actual nonzero* |Rx|. All
    longer images and contact couplings are retained. The finite Device1D is
    represented as one dense block so an arbitrary original-cell count,
    including overlapping left/right contact rows, is supported exactly.
    ``unit_conversion`` defaults to the existing material-test arithmetic;
    ``native-diagonal-mass`` preserves the archived-Hessian producer's explicit
    kg-mass square roots, multiplication/division and sorted-image order.
    """
    def __init__(self, images: Mapping, masses, cell, *, omega0_rad_s=OMEGA0_RAD_S,
                 reciprocity="reject", reciprocity_rtol=1e-9, reciprocity_atol=1e-10,
                 max_numeric_bytes=512*1024**2, unit_conversion="material-test"):
        if (not isinstance(unit_conversion, str)
                or unit_conversion not in {"material-test", "native-diagonal-mass"}):
            raise ValueError("unit_conversion must be material-test or native-diagonal-mass")
        if not isinstance(images, Mapping) or not images:
            raise ValueError("images must be a nonempty image -> IFC2-block mapping")
        masses = np.asarray(masses, dtype=float)
        cell = np.asarray(cell, dtype=float)
        if masses.ndim != 1 or not len(masses) or not np.isfinite(masses).all() or np.any(masses <= 0):
            raise ValueError("One finite positive atomic mass is required per basis atom")
        if cell.shape != (3,3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError("cell must contain three nonsingular lattice rows in Angstrom")
        self.omega0_rad_s = _finite(omega0_rad_s, "omega0_rad_s", positive=True)
        self.masses_amu = _readonly(masses)
        self.cell_rows_angstrom = _readonly(cell)
        self.cell_ndof = 3*len(masses)
        self.max_numeric_bytes = _integer(max_numeric_bytes, "max_numeric_bytes")
        if 2*len(images)*self.cell_ndof**2*8 > self.max_numeric_bytes:
            raise ValueError("IFC2-owned raw and dynamical copies exceed max_numeric_bytes")
        rtol = _finite(reciprocity_rtol, "reciprocity_rtol")
        atol = _finite(reciprocity_atol, "reciprocity_atol")
        if rtol < 0 or atol < 0 or reciprocity not in ("reject", "average"):
            raise ValueError("Supply nonnegative reciprocity tolerances and explicit reject/average policy")
        raw = {}
        for label, block in images.items():
            key = tuple(label)
            if len(key) != 3 or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) for i in key):
                raise ValueError("Each image label must contain three integers")
            value = np.asarray(block)
            if value.shape != (self.cell_ndof,self.cell_ndof) or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError("Every IFC2 block must be a finite real 3Nb-square matrix")
            raw[tuple(int(i) for i in key)] = np.array(value, dtype=float, copy=True)
        if (0,0,0) not in raw:
            raise ValueError("The onsite IFC2 image must be explicitly supplied")
        for image, block in tuple(raw.items()):
            opposite = tuple(-i for i in image)
            if opposite not in raw:
                if np.any(block != 0):
                    raise ValueError("A nonzero IFC2 image is missing its reciprocal partner: " + str(image))
                raw[opposite] = np.zeros_like(block)
        scale = max(float(np.max(np.abs(v))) for v in raw.values())
        error = max(float(np.max(np.abs(v-raw[tuple(-i for i in k)].T))) for k,v in raw.items())
        squared = sum(float(np.sum((v-raw[tuple(-i for i in k)].T)**2)) for k,v in raw.items())
        norm2 = sum(float(np.sum(v*v)) for v in raw.values())
        allowance = atol+rtol*scale
        if reciprocity == "reject" and error > allowance:
            raise ValueError(f"Raw IFC2 reciprocity defect {error:.6g} exceeds {allowance:.6g}; explicit averaging or new derivatives are required")
        selected = {k: .5*(v+raw[tuple(-i for i in k)].T) for k,v in raw.items()} if reciprocity == "average" else raw
        correction = max(float(np.max(np.abs(selected[k]-v))) for k,v in raw.items())
        self.reciprocity = dict(policy=reciprocity,raw_max_abs_ev_A2=error,
            raw_relative_frobenius=float(np.sqrt(squared/norm2)) if norm2 else 0.,
            allowance_ev_A2=allowance,raw_within_tolerance=bool(error<=allowance),
            averaging_applied=reciprocity=="average",maximum_applied_correction_ev_A2=correction,
            asr_repair=False,psd_repair=False,range_cropped=False)
        self.unit_conversion = unit_conversion
        if unit_conversion == "native-diagonal-mass":
            # Preserve the operation order of the original archived-Hessian
            # producer: convert each mass to kg before square roots, multiply
            # by omega0**2, then multiply Phi by its unit and divide once.
            # Sorting also preserves its image Fourier accumulation order.
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                dof_masses = np.repeat(np.asarray(masses, dtype=float) * AMU_KG, 3)
                denominator = (np.sqrt(dof_masses[:, None]) * np.sqrt(dof_masses[None, :])
                               * np.float64(self.omega0_rad_s)**2)
            if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                raise ValueError("Mass/frequency conversion is not representable")
            with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
                dynamic = {k: np.asarray(selected[k], dtype=np.float64) * (EV_J/ANGSTROM_M**2) / denominator
                           for k in sorted(selected)}
        else:
            denominator = np.sqrt(np.repeat(masses,3)[:,None]*np.repeat(masses,3)[None,:])
            conversion = EV_J/ANGSTROM_M**2/AMU_KG/self.omega0_rad_s**2
            dynamic = {k: v/denominator*conversion for k,v in selected.items()}
        for k,v in dynamic.items():
            if not np.isfinite(v).all() or np.any((selected[k]!=0)&(v==0)):
                raise ValueError("IFC2 unit conversion overflowed or underflowed a nonzero entry")
        self.image_dynamical = {k:_readonly(v) for k,v in dynamic.items()}
        nonzero = [k for k,v in selected.items() if np.any(v!=0)]
        self.principal_cells = max((abs(k[0]) for k in nonzero), default=0)
        if self.principal_cells < 1:
            raise ValueError("No nonzero longitudinal coupling: a two-terminal lead cannot be built")
        self.longitudinal_image_range = (-self.principal_cells,self.principal_cells)

    @classmethod
    def from_data(cls, data: IFC2Data, **kwargs):
        return cls(data.image_blocks, data.masses_amu, data.cell_rows_angstrom, **kwargs)

    def transverse_blocks(self, q_y, q_z):
        q = (_finite(q_y,"q_y"),_finite(q_z,"q_z"))
        blocks = {}
        for (x,y,z), value in self.image_dynamical.items():
            if x not in blocks:
                blocks[x] = np.zeros((self.cell_ndof,self.cell_ndof),complex)
            blocks[x] += value*np.exp(1j*(q[0]*y+q[1]*z))
        return blocks

    def bulk_dynamical(self, k_x, q_y=0., q_z=0.):
        """D(k)=sum_R D_R exp(i k.R), in units omega0²."""
        phase = np.asarray([_finite(k_x,"k_x"),_finite(q_y,"q_y"),_finite(q_z,"q_z")])
        matrix = np.zeros((self.cell_ndof,self.cell_ndof),complex)
        for image,value in self.image_dynamical.items():
            matrix += value*np.exp(1j*np.dot(phase,image))
        return matrix

    # The spelling makes the physical cell Bloch matrix easy to discover.
    bulkD = bulk_dynamical

    def at_q(self, q_y, q_z, n_device_cells) -> HarmonicAtQ:
        from negfpy.core.types import Device1D,LeadBlocks
        n = _integer(n_device_cells,"n_device_cells")
        r,b = self.principal_cells,self.cell_ndof
        if n < r:
            raise ValueError("Device shorter than the IFC2 longitudinal range creates a direct left-right lead bypass; n_device_cells must be >= principal_cells")
        # D, canonical device copy, lead/coupling blocks and construction scratch.
        if 16*(3*(n*b)**2+8*(r*b)**2) > self.max_numeric_bytes:
            raise ValueError("Finite harmonic device/interface plan exceeds max_numeric_bytes")
        blocks = self.transverse_blocks(q_y,q_z)
        zero = np.zeros((b,b),complex)
        def assemble(rows,cols):
            return np.block([[blocks.get(j-i,zero) for j in cols] for i in rows])
        d = assemble(range(n),range(n))
        lead00 = assemble(range(r),range(r))
        lead01 = assemble(range(r),range(r,2*r))
        lead10 = assemble(range(r,2*r),range(r))
        # Independent source reverse and exact mixed-mass reciprocity checks.
        scale = max(float(np.max(np.abs(d))),float(np.max(np.abs(lead00))),1.)
        for defect in (np.max(np.abs(d-_adjoint(d))),np.max(np.abs(lead00-_adjoint(lead00))),np.max(np.abs(lead10-_adjoint(lead01)))):
            if defect > 1e-10+1e-8*scale:
                raise ValueError("Fourier harmonic/interface Hermiticity exceeds admission tolerance")
        left_indices = np.arange(r*b,dtype=int)
        right_indices = np.arange((n-r)*b,n*b,dtype=int)
        left = assemble(range(r),range(-r,0))
        right = assemble(range(n-r,n),range(n,n+r))
        # Closest external cells are -1 and n; n>=r proves no omitted bypass.
        lead = LeadBlocks(_readonly(lead00),_readonly(lead01),_readonly(lead10))
        device = Device1D([_readonly(d)],[])
        return HarmonicAtQ(device.onsite_blocks[0],device,lead,_readonly(left),_readonly(right),
            _readonly(left_indices,int),_readonly(right_indices,int),(float(q_y),float(q_z)),n,r,self.omega0_rad_s)

    def stability_audit(self, qpoints, *, longitudinal_points=65,
                        negative_tolerance_thz2=1e-4, deadline=None):
        """Fail on sampled unstable modes; never alter or clip eigenvalues.

        This is a declared finite Bloch sample, not a global positivity proof.
        The explicit tolerance is in squared cyclic THz. Accepted negative
        values remain present in the minimum and count diagnostics.
        """
        nk = _integer(longitudinal_points,"longitudinal_points",3)
        tolerance = _finite(negative_tolerance_thz2,"negative_tolerance_thz2")
        if tolerance < 0:
            raise ValueError("negative_tolerance_thz2 must be nonnegative")
        minimum = np.inf;where = None;negative_count = 0;completed = 0
        conversion = (self.omega0_rad_s/(2*np.pi*1e12))**2
        for qy,qz in qpoints:
            for kx in np.linspace(-np.pi,np.pi,nk,endpoint=True):
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("Harmonic stability audit reached the shared deadline")
                d = self.bulk_dynamical(kx,qy,qz)
                defect = float(np.max(np.abs(d-_adjoint(d))))
                if defect > 1e-10+1e-8*max(float(np.max(np.abs(d))),1.):
                    raise ValueError("Sampled bulk D is not Hermitian")
                values = np.linalg.eigvalsh(d)*conversion
                completed += 1;negative_count += int(np.sum(values<0))
                if float(values[0]) < minimum:
                    minimum=float(values[0]);where=[float(kx),float(qy),float(qz)]
        if not completed:
            raise ValueError("Stability audit needs at least one transverse q point")
        result=dict(passed=minimum>=-tolerance,minimum_squared_frequency_thz2=minimum,
            minimum_phase_radians=where,negative_eigenvalue_count=negative_count,
            negative_tolerance_thz2=tolerance,bloch_points_completed=completed,
            eigenvalues_clipped=False,finite_sample_only=True)
        if not result["passed"]:
            raise HarmonicInstability(result)
        return result


class HarmonicInstability(ValueError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__("Harmonic instability on the declared Bloch sample: " + json.dumps(audit,sort_keys=True))
