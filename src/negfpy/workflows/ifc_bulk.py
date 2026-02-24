
"""Unified IFC-driven bulk calculations (transmission, DOS, dispersion)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from negfpy.core import (
    lead_phonon_dispersion_3d,
    lead_surface_dos,
    lead_surface_dos_kavg,
    lead_surface_dos_kavg_adaptive,
    transmission_kavg_adaptive,
    transmission_kavg_adaptive_global_eta,
)
from negfpy.core.types import LeadKSpace
from negfpy.io import read_ifc
from negfpy.modeling import (
    BuildConfig,
    IFCData,
    enforce_translational_asr_on_self_term,
    qe_ev_to_omega,
    qe_omega_to_cm1,
    qe_omega_to_thz,
)
from negfpy.modeling.builders import build_material_kspace_params
from negfpy.models import material_kspace_device, material_kspace_lead


DEFAULT_ETA_VALUES: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return token if token else "unnamed"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(base_dir: Path, path_like: str | Path) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (base_dir / p)


def _git_head_sha(cwd: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception:
        return None
    return out if out else None


def _git_dirty(cwd: Path) -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(cwd), text=True)
    except Exception:
        return None
    return bool(out.strip())


def _append_manifest_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_to_builtin(payload), sort_keys=True))
        fh.write("\n")


def _qe_cm1_to_omega(freq_cm1: float) -> float:
    if freq_cm1 <= 0.0:
        raise ValueError("cm^-1 frequency must be positive.")
    return float(freq_cm1 / float(np.asarray(qe_omega_to_cm1(1.0), dtype=float)))


def _qe_thz_to_omega(freq_thz: float) -> float:
    if freq_thz <= 0.0:
        raise ValueError("THz frequency must be positive.")
    return float(freq_thz / float(np.asarray(qe_omega_to_thz(1.0), dtype=float)))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _load_json_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as fh:
        cfg = json.load(fh)
    if not isinstance(cfg, dict):
        raise ValueError("Input config must be a JSON object.")
    return cfg


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(_to_builtin(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")


def _resolve_kline(nk: int, mode: str) -> tuple[np.ndarray, str]:
    if nk <= 0:
        raise ValueError("nk must be positive.")
    if nk == 1:
        return np.array([0.0], dtype=float), "gamma"
    mode_eff = mode
    if mode_eff == "auto":
        mode_eff = "shifted" if (nk % 2 == 0) else "centered"
    if mode_eff == "shifted":
        if nk % 2 != 0:
            raise ValueError("shifted k-mesh requires even nk to avoid sampling Gamma exactly.")
        return -np.pi + (np.arange(nk, dtype=float) + 0.5) * (2.0 * np.pi / nk), mode_eff
    if mode_eff == "centered":
        step = 2.0 * np.pi / nk
        idx = np.arange(nk, dtype=float) - (nk // 2)
        return idx * step, mode_eff
    if mode_eff in {"legacy_endpoint", "legacy"}:
        # Legacy benchmark grid: includes both -pi and +pi endpoints.
        return np.linspace(-np.pi, np.pi, nk, endpoint=True), "legacy_endpoint"
    raise ValueError(f"Unknown k-mesh mode '{mode}'.")


def _build_kpoints(dim: int, nky: int, nkz: int, mode: str) -> tuple[list[tuple[float, ...]], dict[str, Any]]:
    if dim == 0:
        return [tuple()], {"dimension": 0, "mode_y": "gamma", "mode_z": "gamma", "nky": 1, "nkz": 1}
    if dim == 1:
        ky, mode_y = _resolve_kline(nky, mode)
        return [(float(v),) for v in ky], {"dimension": 1, "mode_y": mode_y, "mode_z": "none", "nky": nky, "nkz": 1}
    if dim == 2:
        ky, mode_y = _resolve_kline(nky, mode)
        kz, mode_z = _resolve_kline(nkz, mode)
        return [(float(a), float(b)) for a in ky for b in kz], {
            "dimension": 2,
            "mode_y": mode_y,
            "mode_z": mode_z,
            "nky": nky,
            "nkz": nkz,
        }
    raise ValueError("kmesh.dimension must be 0, 1, or 2.")


def _prepare_kmesh(cfg: dict[str, Any]) -> dict[str, Any]:
    kcfg = dict(cfg.get("kmesh", {}))
    dim = int(kcfg.get("dimension", 2))
    mode = str(kcfg.get("mode", "auto"))

    nk = int(kcfg.get("nk", 1))
    nky = int(kcfg.get("nky", nk))
    nkz = int(kcfg.get("nkz", nk))
    nk_low = int(kcfg.get("nk_low", max(nky, nkz)))
    nky_low = int(kcfg.get("nky_low", max(nky, nk_low)))
    nkz_low = int(kcfg.get("nkz_low", max(nkz, nk_low)))
    low_cm1 = float(kcfg.get("low_cm1", 0.0))

    kpts_hi, info_hi = _build_kpoints(dim, nky=nky, nkz=nkz, mode=mode)
    kpts_lo, info_lo = _build_kpoints(dim, nky=nky_low, nkz=nkz_low, mode=mode)
    return {
        "dim": dim,
        "mode": mode,
        "low_cm1": low_cm1,
        "high": {"kpoints": kpts_hi, **info_hi},
        "low": {"kpoints": kpts_lo, **info_lo},
    }


def _default_kgrid(nk: int, centered: bool) -> np.ndarray:
    if nk <= 0:
        raise ValueError("nk must be positive.")
    if centered:
        step = 2.0 * np.pi / nk
        idx = np.arange(nk, dtype=float) - (nk // 2)
        return idx * step
    return -np.pi + (np.arange(nk, dtype=float) + 0.5) * (2.0 * np.pi / nk)


def _estimate_omega_max_from_ifc(
    ifc: IFCData,
    *,
    nkx: int,
    nky: int,
    nkz: int,
    centered_grid: bool,
    safety_factor: float = 1.02,
) -> float:
    kx = _default_kgrid(nkx, centered=centered_grid)
    ky = _default_kgrid(nky, centered=centered_grid)
    kz = _default_kgrid(nkz, centered=centered_grid)
    ndof = len(ifc.masses) * ifc.dof_per_atom
    mrep = np.repeat(np.asarray(ifc.masses, dtype=float), ifc.dof_per_atom)
    minv = np.diag(1.0 / np.sqrt(mrep))
    terms = [(int(t.dx), int(t.dy), int(t.dz), np.asarray(t.block, dtype=np.complex128)) for t in ifc.terms]

    wmax = 0.0
    for kxv in kx:
        for kyv in ky:
            for kzv in kz:
                phi = np.zeros((ndof, ndof), dtype=np.complex128)
                for dx, dy, dz, block in terms:
                    phi += block * np.exp(1j * (kxv * dx + kyv * dy + kzv * dz))
                dmat = minv @ phi @ minv
                dmat = 0.5 * (dmat + dmat.conj().T)
                vals = np.linalg.eigvalsh(dmat)
                wmax = max(wmax, float(np.sqrt(max(float(np.max(vals.real)), 0.0))))
    return float(max(wmax * safety_factor, 1e-8))


def _apply_transverse_cutoff(
    ifc: IFCData,
    *,
    dy_cutoff: int | None,
    dz_cutoff: int | None,
) -> tuple[IFCData, int]:
    if dy_cutoff is None and dz_cutoff is None:
        return ifc, 0

    keep_terms = []
    removed = 0
    for term in ifc.terms:
        if dy_cutoff is not None and abs(int(term.dy)) > dy_cutoff:
            removed += 1
            continue
        if dz_cutoff is not None and abs(int(term.dz)) > dz_cutoff:
            removed += 1
            continue
        keep_terms.append(term)

    if len(keep_terms) == 0:
        raise ValueError("All IFC terms were removed by dy/dz cutoffs.")
    if not any(t.dx == 0 and t.dy == 0 and t.dz == 0 for t in keep_terms):
        raise ValueError("dy/dz cutoff removed IFC self-term (0,0,0), invalid model.")

    metadata = dict(ifc.metadata)
    metadata["dy_cutoff"] = dy_cutoff
    metadata["dz_cutoff"] = dz_cutoff
    metadata["n_terms_removed_by_cutoff"] = int(removed)
    out = IFCData(
        masses=np.asarray(ifc.masses, dtype=float),
        dof_per_atom=ifc.dof_per_atom,
        terms=tuple(keep_terms),
        units=ifc.units,
        metadata=metadata,
        lattice_vectors=ifc.lattice_vectors,
        atom_positions=ifc.atom_positions,
        atom_symbols=ifc.atom_symbols,
        index_convention=ifc.index_convention,
    )
    return out, removed


def _build_omega_grid(cfg: dict[str, Any], ifc: IFCData) -> tuple[np.ndarray, dict[str, Any]]:
    ocfg = dict(cfg.get("omega", {}))
    omega_min = float(ocfg.get("min", 0.0))
    n_points = int(ocfg.get("n_points", ocfg.get("nw", 160)))
    if n_points <= 1:
        raise ValueError("omega.n_points must be > 1.")

    omega_max_raw = ocfg.get("max", None)
    fmax_cm1 = ocfg.get("fmax_cm1", None)
    fmax_thz = ocfg.get("fmax_thz", None)
    auto_max = bool(ocfg.get("auto_max", True))

    if fmax_cm1 is not None:
        omega_max = _qe_cm1_to_omega(float(fmax_cm1))
        source = "fmax_cm1"
    elif fmax_thz is not None:
        omega_max = _qe_thz_to_omega(float(fmax_thz))
        source = "fmax_thz"
    elif omega_max_raw is not None:
        omega_max = float(omega_max_raw)
        source = "omega.max"
    elif auto_max:
        omega_max = _estimate_omega_max_from_ifc(
            ifc,
            nkx=int(ocfg.get("auto_sample_nk", 12)),
            nky=int(ocfg.get("auto_sample_nk", 12)),
            nkz=int(ocfg.get("auto_sample_nk", 12)),
            centered_grid=bool(ocfg.get("auto_centered_grid", False)),
            safety_factor=float(ocfg.get("safety_factor", 1.02)),
        )
        source = "auto_ifc_estimate"
    else:
        raise ValueError("No omega maximum specified. Set omega.max or omega.auto_max=true.")

    if omega_max <= omega_min:
        raise ValueError("Resolved omega max must be greater than omega min.")
    omegas = np.linspace(omega_min, omega_max, n_points)
    info = {
        "omega_min_qe": float(omega_min),
        "omega_max_qe": float(omega_max),
        "n_points": int(n_points),
        "omega_max_source": source,
        "omega_max_thz": float(np.asarray(qe_omega_to_thz(omega_max), dtype=float)),
        "omega_max_cm1": float(np.asarray(qe_omega_to_cm1(omega_max), dtype=float)),
    }
    return omegas, info

def _resolve_eta_values_for_omega(
    omega: float,
    omega_cm1: float,
    *,
    eta_values: tuple[float, ...],
    max_eta_over_omega: float | None,
    eta_ratio_cm1_min: float | None,
) -> tuple[float, ...]:
    if max_eta_over_omega is None:
        return eta_values
    apply_cap = eta_ratio_cm1_min is None or omega_cm1 >= eta_ratio_cm1_min
    if not apply_cap:
        return eta_values
    filtered = tuple(e for e in eta_values if (e / omega) <= max_eta_over_omega)
    return filtered if len(filtered) > 0 else (min(eta_values),)


def _parse_ky_kz(kpar: tuple[float, ...] | None) -> tuple[float, float]:
    if kpar is None or len(kpar) == 0:
        return 0.0, 0.0
    if len(kpar) == 1:
        return float(kpar[0]), 0.0
    if len(kpar) == 2:
        return float(kpar[0]), float(kpar[1])
    raise ValueError("kpar must contain at most two transverse components: (ky,) or (ky, kz).")


def _sum_k_terms(terms: dict[tuple[int, int], np.ndarray], ky: float, kz: float, ndof: int) -> np.ndarray:
    out = np.zeros((ndof, ndof), dtype=np.complex128)
    for (dy, dz), block in terms.items():
        out += np.asarray(block, dtype=np.complex128) * np.exp(1j * (ky * dy + kz * dz))
    return out


def _build_interface_contacts(params):
    """Build left/right lead and interface couplings for IFC bulk transport."""

    fc00_terms = {k: np.asarray(v, dtype=np.complex128) for k, v in params.fc00_terms.items()}
    if params.fc10_terms is None:
        fc10_terms = {(-dy, -dz): np.asarray(v, dtype=np.complex128).conj().T for (dy, dz), v in params.fc01_terms.items()}
    else:
        fc10_terms = {k: np.asarray(v, dtype=np.complex128) for k, v in params.fc10_terms.items()}

    # Mapping from IFC block convention to transport interface convention:
    # right interface coupling term in transport basis:
    # KR01(dy,dz) == FC10(-dy,-dz)^T
    fc_r01_terms = {(-dy, -dz): np.asarray(v, dtype=np.complex128).T for (dy, dz), v in fc10_terms.items()}
    # left lead-side nearest-neighbor coupling:
    # KL01(dy,dz) == KR01(-dy,-dz)^T == FC10(dy,dz)
    fc_l01_terms = {k: np.asarray(v, dtype=np.complex128) for k, v in fc10_terms.items()}

    masses = np.repeat(np.asarray(params.masses, dtype=float), int(params.dof_per_atom))
    mhalf = np.diag(1.0 / np.sqrt(masses))
    ndof = int(params.ndof)

    def _d00(ky: float, kz: float) -> np.ndarray:
        phi00 = _sum_k_terms(fc00_terms, ky=ky, kz=kz, ndof=ndof)
        if float(params.onsite_pinning) != 0.0:
            phi00 = phi00 + float(params.onsite_pinning) * np.eye(ndof, dtype=np.complex128)
        return mhalf @ phi00 @ mhalf

    def _d01_right(ky: float, kz: float) -> np.ndarray:
        return mhalf @ _sum_k_terms(fc_r01_terms, ky=ky, kz=kz, ndof=ndof) @ mhalf

    def _d01_left(ky: float, kz: float) -> np.ndarray:
        return mhalf @ _sum_k_terms(fc_l01_terms, ky=ky, kz=kz, ndof=ndof) @ mhalf

    def _right_builder(kpar):
        ky, kz = _parse_ky_kz(kpar)
        d00 = _d00(ky, kz)
        d01 = _d01_right(ky, kz)
        return d00, d01, d01.conj().T

    def _left_builder(kpar):
        ky, kz = _parse_ky_kz(kpar)
        d00 = _d00(ky, kz)
        d01 = _d01_left(ky, kz)
        return d00, d01, d01.conj().T

    def _vdl_right(kpar):
        ky, kz = _parse_ky_kz(kpar)
        return _d01_right(ky, kz)

    def _vdl_left(kpar):
        return _vdl_right(kpar).conj().T

    return LeadKSpace(blocks_builder=_left_builder), LeadKSpace(blocks_builder=_right_builder), _vdl_left, _vdl_right


def _drop_nyquist_transverse_terms(params, ifc_metadata: dict[str, Any]):
    nr = ifc_metadata.get("nr") if isinstance(ifc_metadata, dict) else None
    if nr is None or len(nr) != 3:
        return params, {"dropped_dy": None, "dropped_dz": None, "n_dropped_fc00": 0, "n_dropped_fc01": 0, "n_dropped_fc10": 0}
    try:
        nr2 = int(nr[1])
        nr3 = int(nr[2])
    except Exception:
        return params, {"dropped_dy": None, "dropped_dz": None, "n_dropped_fc00": 0, "n_dropped_fc01": 0, "n_dropped_fc10": 0}

    drop_dy = (-nr2 // 2) if (nr2 > 1 and nr2 % 2 == 0) else None
    drop_dz = (-nr3 // 2) if (nr3 > 1 and nr3 % 2 == 0) else None
    if drop_dy is None and drop_dz is None:
        return params, {"dropped_dy": None, "dropped_dz": None, "n_dropped_fc00": 0, "n_dropped_fc01": 0, "n_dropped_fc10": 0}

    def _keep(key: tuple[int, int]) -> bool:
        dy, dz = int(key[0]), int(key[1])
        if drop_dy is not None and dy == drop_dy:
            return False
        if drop_dz is not None and dz == drop_dz:
            return False
        return True

    fc00 = {k: v for k, v in params.fc00_terms.items() if _keep(k)}
    fc01 = {k: v for k, v in params.fc01_terms.items() if _keep(k)}
    fc10 = None if params.fc10_terms is None else {k: v for k, v in params.fc10_terms.items() if _keep(k)}
    dropped = {
        "dropped_dy": drop_dy,
        "dropped_dz": drop_dz,
        "n_dropped_fc00": int(len(params.fc00_terms) - len(fc00)),
        "n_dropped_fc01": int(len(params.fc01_terms) - len(fc01)),
        "n_dropped_fc10": int(0 if params.fc10_terms is None else (len(params.fc10_terms) - len(fc10))),
    }
    return replace(params, fc00_terms=fc00, fc01_terms=fc01, fc10_terms=fc10), dropped


def _apply_mass_mode(params, mass_mode_raw: str):
    """Apply model.mass_mode to MaterialKspaceParams."""
    mass_mode = str(mass_mode_raw).lower().replace("-", "_")
    if mass_mode == "ifc":
        return params, mass_mode
    if mass_mode in {"unit", "ones", "legacy_unit"}:
        return replace(params, masses=np.ones_like(params.masses, dtype=float)), "unit"
    raise ValueError("model.mass_mode must be one of: ifc, unit.")


def _apply_transverse_stability_controls(params, *, ifc_metadata: dict[str, Any], drop_nyquist: bool):
    """Apply optional Nyquist drop and required Nyquist pair-symmetry stabilization."""
    nyquist_info = {"dropped_dy": None, "dropped_dz": None, "n_dropped_fc00": 0, "n_dropped_fc01": 0, "n_dropped_fc10": 0}
    out = params
    if drop_nyquist:
        out, nyquist_info = _drop_nyquist_transverse_terms(out, ifc_metadata=ifc_metadata)
    # Always enforce Nyquist +/- pair symmetry for longest-range transverse
    # terms to keep numerical behavior stable across interfaces.
    out, pm_sym_info = _enforce_transverse_pm_symmetry(out, ifc_metadata=ifc_metadata)
    return out, nyquist_info, pm_sym_info


def _enforce_transverse_pm_symmetry(params, ifc_metadata: dict[str, Any] | None = None):
    """Complete missing +/- transverse terms for Nyquist shifts only.

    - Onsite terms:   fc00(dy,dz) = fc00(-dy,-dz)^†
    - Interface terms: fc10(dy,dz) = fc01(-dy,-dz)^†

    Only terms touching Nyquist transverse indices are modified. Non-Nyquist
    force-constant terms are left unchanged.
    """

    nr = ifc_metadata.get("nr") if isinstance(ifc_metadata, dict) else None
    if nr is None or len(nr) != 3:
        return params, {"enabled": False, "reason": "missing_nr_metadata"}
    try:
        nr2 = int(nr[1])
        nr3 = int(nr[2])
    except Exception:
        return params, {"enabled": False, "reason": "invalid_nr_metadata"}
    nyq_abs_dy = (nr2 // 2) if (nr2 > 1 and nr2 % 2 == 0) else None
    nyq_abs_dz = (nr3 // 2) if (nr3 > 1 and nr3 % 2 == 0) else None
    if nyq_abs_dy is None and nyq_abs_dz is None:
        return params, {"enabled": False, "reason": "no_even_transverse_grid"}

    def _is_nyquist_key(key: tuple[int, int]) -> bool:
        dy, dz = int(key[0]), int(key[1])
        if nyq_abs_dy is not None and abs(dy) == nyq_abs_dy:
            return True
        if nyq_abs_dz is not None and abs(dz) == nyq_abs_dz:
            return True
        return False

    def _complete_self_conjugate_pairs(terms: dict[tuple[int, int], np.ndarray] | None):
        if terms is None:
            return None, {"n_terms_in": 0, "n_terms_out": 0, "n_pairs_added": 0, "n_self_hermitianized": 0}
        out = {k: np.asarray(v, dtype=np.complex128).copy() for k, v in terms.items()}
        n_added = 0
        n_self = 0
        for k in list(out.keys()):
            if not _is_nyquist_key(k):
                continue
            kp = (-int(k[0]), -int(k[1]))
            if k == kp:
                out[k] = 0.5 * (out[k] + out[k].conj().T)
                n_self += 1
                continue
            if kp not in out:
                out[kp] = out[k].conj().T
                n_added += 1
        return out, {
            "n_terms_in": int(len(terms)),
            "n_terms_out": int(len(out)),
            "n_pairs_added": int(n_added),
            "n_self_hermitianized": int(n_self),
        }

    def _complete_cross_conjugate_pairs(
        terms_01: dict[tuple[int, int], np.ndarray],
        terms_10: dict[tuple[int, int], np.ndarray] | None,
    ):
        out01 = {k: np.asarray(v, dtype=np.complex128).copy() for k, v in terms_01.items()}
        out10 = None if terms_10 is None else {k: np.asarray(v, dtype=np.complex128).copy() for k, v in terms_10.items()}
        if out10 is None:
            return out01, None, {"n_fc01_in": int(len(terms_01)), "n_fc01_out": int(len(out01)), "n_fc10_in": 0, "n_fc10_out": 0, "n_pairs_added_to_fc01": 0, "n_pairs_added_to_fc10": 0}

        add01 = 0
        add10 = 0

        for k, b01 in list(out01.items()):
            if not _is_nyquist_key(k):
                continue
            kp = (-int(k[0]), -int(k[1]))
            if kp not in out10:
                out10[kp] = b01.conj().T
                add10 += 1

        for k, b10 in list(out10.items()):
            if not _is_nyquist_key(k):
                continue
            kp = (-int(k[0]), -int(k[1]))
            if kp not in out01:
                out01[kp] = b10.conj().T
                add01 += 1

        info = {
            "n_fc01_in": int(len(terms_01)),
            "n_fc01_out": int(len(out01)),
            "n_fc10_in": int(len(terms_10)),
            "n_fc10_out": int(len(out10)),
            "n_pairs_added_to_fc01": int(add01),
            "n_pairs_added_to_fc10": int(add10),
        }
        return out01, out10, info

    fc00, info00 = _complete_self_conjugate_pairs(params.fc00_terms)
    fc01, fc10, info01_10 = _complete_cross_conjugate_pairs(params.fc01_terms, params.fc10_terms)
    info = {
        "enabled": True,
        "nyquist_abs_dy": nyq_abs_dy,
        "nyquist_abs_dz": nyq_abs_dz,
        "fc00": info00,
        "fc01_fc10": info01_10,
    }
    return replace(params, fc00_terms=fc00, fc01_terms=fc01, fc10_terms=fc10), info


def _lead_surface_dos_kavg_adaptive_global_eta(
    omega: float,
    lead,
    kpoints: list[tuple[float, ...]],
    eta_values: tuple[float, ...],
    min_success_fraction: float,
    *,
    normalize_per_mode: bool,
    surface_gf_method: str,
    negative_tolerance: float,
) -> tuple[float, dict[str, Any]]:
    n_total = len(kpoints)
    for eta in eta_values:
        vals: list[float] = []
        for kpar in kpoints:
            try:
                dval = lead_surface_dos(
                    omega=omega,
                    lead=lead,
                    eta=eta,
                    kpar=kpar if len(kpar) > 0 else None,
                    normalize_per_mode=normalize_per_mode,
                    surface_gf_method=surface_gf_method,
                )
            except Exception:
                continue
            if not np.isfinite(dval) or dval < -negative_tolerance:
                continue
            vals.append(float(max(dval, 0.0)))
        n_success = len(vals)
        success_fraction = float(n_success / n_total)
        if n_success > 0 and success_fraction >= min_success_fraction:
            return float(np.mean(vals)), {
                "n_total": int(n_total),
                "n_success": int(n_success),
                "n_failed": int(n_total - n_success),
                "success_fraction": success_fraction,
                "eta_selected": float(eta),
                "eta_histogram": {float(e): (int(n_success) if float(e) == float(eta) else 0) for e in eta_values},
            }
    raise RuntimeError("No eta value satisfied global adaptive DOS quality threshold.")


def _run_transmission(
    cfg: dict[str, Any],
    *,
    device,
    lead_left,
    lead_right,
    device_to_lead_left=None,
    device_to_lead_right=None,
    omegas: np.ndarray,
    kmesh: dict[str, Any],
) -> dict[str, Any]:
    scfg = dict(cfg.get("solver", {}))
    sgf_method = str(scfg.get("surface_gf_method", "generalized_eigen_svd"))
    eta_scheme = str(scfg.get("eta_scheme", "adaptive")).lower().replace("-", "_")
    eta_fixed = float(scfg.get("eta_fixed", 1e-4))
    eta_values_base = tuple(float(v) for v in scfg.get("eta_values", DEFAULT_ETA_VALUES))
    eta_device_raw = scfg.get("eta_device", None)
    eta_device = None if eta_device_raw is None else float(eta_device_raw)
    min_success_fraction = float(scfg.get("min_success_fraction", 0.7))
    nonnegative_tolerance = float(scfg.get("nonnegative_tolerance", 1e-8))
    max_channel_factor_raw = scfg.get("max_channel_factor", 1.5)
    max_channel_factor = None if max_channel_factor_raw is None else float(max_channel_factor_raw)
    collect_rejected = bool(scfg.get("collect_rejected", False))
    max_eta_over_omega_raw = scfg.get("max_eta_over_omega", None)
    max_eta_over_omega = None if max_eta_over_omega_raw is None else float(max_eta_over_omega_raw)
    eta_ratio_cm1_min_raw = scfg.get("eta_ratio_cm1_min", None)
    eta_ratio_cm1_min = None if eta_ratio_cm1_min_raw is None else float(eta_ratio_cm1_min_raw)

    omega_scale_mode = str(scfg.get("omega_scale_mode", "none")).lower()
    if omega_scale_mode == "ev":
        omega_scale = float(np.asarray(qe_ev_to_omega(1.0), dtype=float))
    elif omega_scale_mode == "none":
        omega_scale = None
    else:
        raise ValueError("solver.omega_scale_mode must be 'none' or 'ev'.")

    vals = np.zeros_like(omegas, dtype=float)
    omega_failures: list[dict[str, Any]] = []
    eta_records: list[dict[str, Any]] = []
    rejected_total = 0

    for i, w in enumerate(omegas):
        if w <= 0.0:
            vals[i] = 0.0
            continue
        w_cm1 = float(np.asarray(qe_omega_to_cm1(w), dtype=float))
        kpts = kmesh["low"]["kpoints"] if w_cm1 <= float(kmesh["low_cm1"]) else kmesh["high"]["kpoints"]
        eta_values = _resolve_eta_values_for_omega(
            float(w),
            w_cm1,
            eta_values=eta_values_base,
            max_eta_over_omega=max_eta_over_omega,
            eta_ratio_cm1_min=eta_ratio_cm1_min,
        )

        common = dict(
            omega=float(w),
            device=device,
            lead_left=lead_left,
            lead_right=lead_right,
            kpoints=kpts,
            eta_device=eta_device,
            min_success_fraction=min_success_fraction,
            device_to_lead_left=device_to_lead_left,
            device_to_lead_right=device_to_lead_right,
            surface_gf_method=sgf_method,
            omega_scale=omega_scale,
            nonnegative_tolerance=nonnegative_tolerance,
            max_channel_factor=max_channel_factor,
            collect_rejected=collect_rejected,
        )
        try:
            if eta_scheme == "fixed":
                tavg, info = transmission_kavg_adaptive_global_eta(**common, eta_values=(eta_fixed,))
            elif eta_scheme == "adaptive_global":
                tavg, info = transmission_kavg_adaptive_global_eta(**common, eta_values=eta_values)
            else:
                tavg, info = transmission_kavg_adaptive(**common, eta_values=eta_values)
            vals[i] = float(tavg)
        except Exception as exc:
            vals[i] = np.nan
            omega_failures.append({"omega_qe": float(w), "omega_cm1": w_cm1, "error": str(exc)})
            continue

        hist = dict(info.get("eta_histogram", {}))
        used = [float(e) for e, c in hist.items() if int(c) > 0]
        max_eta_used = max(used) if used else np.nan
        eta_records.append(
            {
                "omega_qe": float(w),
                "omega_cm1": w_cm1,
                "n_failed_kpoints": int(info.get("n_failed", 0)),
                "success_fraction": float(info.get("success_fraction", 0.0)),
                "eta_histogram": hist,
                "eta_selected": info.get("eta_selected", None),
                "max_eta_used": None if not np.isfinite(max_eta_used) else float(max_eta_used),
                "max_eta_over_omega": None if not np.isfinite(max_eta_used) else float(max_eta_used / float(w)),
            }
        )
        if collect_rejected:
            rejected_total += int(info.get("n_rejected", 0))

    if np.all(~np.isfinite(vals)):
        raise RuntimeError("Transmission calculation failed for all omega points.")

    vals = np.where(np.isfinite(vals), vals, np.nan)
    vals = np.where(vals < 0.0, np.maximum(vals, -1e-12), vals)
    vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), np.nan)
    max_ratio = max((r["max_eta_over_omega"] for r in eta_records if r["max_eta_over_omega"] is not None), default=None)
    return {
        "values": vals,
        "eta_records": eta_records,
        "omega_failures": omega_failures,
        "n_rejected_total": int(rejected_total),
        "summary": {
            "surface_gf_method": sgf_method,
            "eta_scheme": eta_scheme,
            "eta_fixed": eta_fixed,
            "eta_values_base": eta_values_base,
            "eta_device": eta_device,
            "max_eta_over_omega": max_ratio,
            "n_omega_failures": len(omega_failures),
        },
    }

def _run_dos(
    cfg: dict[str, Any],
    *,
    lead,
    omegas: np.ndarray,
    kmesh: dict[str, Any],
) -> dict[str, Any]:
    scfg = dict(cfg.get("solver", {}))
    dcfg = dict(cfg.get("dos", {}))

    sgf_method = str(scfg.get("surface_gf_method", "generalized_eigen_svd"))
    eta_scheme = str(scfg.get("eta_scheme", "adaptive")).lower().replace("-", "_")
    eta_fixed = float(scfg.get("eta_fixed", 1e-4))
    eta_values_base = tuple(float(v) for v in scfg.get("eta_values", DEFAULT_ETA_VALUES))
    min_success_fraction = float(scfg.get("min_success_fraction", 0.7))
    max_eta_over_omega_raw = scfg.get("max_eta_over_omega", None)
    max_eta_over_omega = None if max_eta_over_omega_raw is None else float(max_eta_over_omega_raw)
    eta_ratio_cm1_min_raw = scfg.get("eta_ratio_cm1_min", None)
    eta_ratio_cm1_min = None if eta_ratio_cm1_min_raw is None else float(eta_ratio_cm1_min_raw)

    normalize_per_mode = bool(dcfg.get("normalize_per_mode", False))
    negative_tolerance = float(dcfg.get("negative_tolerance", 1e-12))

    vals = np.zeros_like(omegas, dtype=float)
    omega_failures: list[dict[str, Any]] = []
    eta_records: list[dict[str, Any]] = []

    for i, w in enumerate(omegas):
        if w < 0.0:
            vals[i] = np.nan
            continue
        w_cm1 = float(np.asarray(qe_omega_to_cm1(w), dtype=float))
        kpts = kmesh["low"]["kpoints"] if w_cm1 <= float(kmesh["low_cm1"]) else kmesh["high"]["kpoints"]
        eta_values = _resolve_eta_values_for_omega(
            float(max(w, 1e-30)),
            w_cm1,
            eta_values=eta_values_base,
            max_eta_over_omega=max_eta_over_omega,
            eta_ratio_cm1_min=eta_ratio_cm1_min,
        )
        try:
            if eta_scheme == "fixed":
                dval = lead_surface_dos_kavg(
                    omega=float(w),
                    lead=lead,
                    kpoints=kpts,
                    eta=eta_fixed,
                    normalize_per_mode=normalize_per_mode,
                    surface_gf_method=sgf_method,
                )
                info: dict[str, Any] = {
                    "n_total": int(len(kpts)),
                    "n_success": int(len(kpts)),
                    "n_failed": 0,
                    "success_fraction": 1.0,
                    "eta_selected": float(eta_fixed),
                    "eta_histogram": {float(eta_fixed): int(len(kpts))},
                }
            elif eta_scheme == "adaptive_global":
                dval, info = _lead_surface_dos_kavg_adaptive_global_eta(
                    omega=float(w),
                    lead=lead,
                    kpoints=kpts,
                    eta_values=eta_values,
                    min_success_fraction=min_success_fraction,
                    normalize_per_mode=normalize_per_mode,
                    surface_gf_method=sgf_method,
                    negative_tolerance=negative_tolerance,
                )
            else:
                dval, info = lead_surface_dos_kavg_adaptive(
                    omega=float(w),
                    lead=lead,
                    kpoints=kpts,
                    eta_values=eta_values,
                    min_success_fraction=min_success_fraction,
                    normalize_per_mode=normalize_per_mode,
                    surface_gf_method=sgf_method,
                )
            vals[i] = float(max(dval, 0.0))
        except Exception as exc:
            vals[i] = np.nan
            omega_failures.append({"omega_qe": float(w), "omega_cm1": w_cm1, "error": str(exc)})
            continue

        hist = dict(info.get("eta_histogram", {}))
        used = [float(e) for e, c in hist.items() if int(c) > 0]
        max_eta_used = max(used) if used else np.nan
        eta_records.append(
            {
                "omega_qe": float(w),
                "omega_cm1": w_cm1,
                "n_failed_kpoints": int(info.get("n_failed", 0)),
                "success_fraction": float(info.get("success_fraction", 0.0)),
                "eta_histogram": hist,
                "eta_selected": info.get("eta_selected", None),
                "max_eta_used": None if not np.isfinite(max_eta_used) else float(max_eta_used),
                "max_eta_over_omega": None
                if not np.isfinite(max_eta_used) or w <= 0.0
                else float(max_eta_used / float(w)),
            }
        )

    if np.all(~np.isfinite(vals)):
        raise RuntimeError("DOS calculation failed for all omega points.")
    max_ratio = max((r["max_eta_over_omega"] for r in eta_records if r["max_eta_over_omega"] is not None), default=None)
    return {
        "values": vals,
        "eta_records": eta_records,
        "omega_failures": omega_failures,
        "summary": {
            "surface_gf_method": sgf_method,
            "eta_scheme": eta_scheme,
            "eta_fixed": eta_fixed,
            "eta_values_base": eta_values_base,
            "normalize_per_mode": normalize_per_mode,
            "max_eta_over_omega": max_ratio,
            "n_omega_failures": len(omega_failures),
        },
    }


def _run_dispersion(cfg: dict[str, Any], *, lead) -> dict[str, Any]:
    dcfg = dict(cfg.get("dispersion", {}))
    kx_min = float(dcfg.get("kx_min", 0.0))
    kx_max = float(dcfg.get("kx_max", np.pi))
    n_kx = int(dcfg.get("n_kx", 240))
    if n_kx <= 1:
        raise ValueError("dispersion.n_kx must be > 1.")
    kpar_raw = dcfg.get("kpar", [0.0, 0.0])
    if not isinstance(kpar_raw, list) or len(kpar_raw) > 2:
        raise ValueError("dispersion.kpar must be a list with 0-2 values.")
    if len(kpar_raw) == 0:
        ky, kz = 0.0, 0.0
    elif len(kpar_raw) == 1:
        ky, kz = float(kpar_raw[0]), 0.0
    else:
        ky, kz = float(kpar_raw[0]), float(kpar_raw[1])
    negative_tolerance = float(dcfg.get("negative_tolerance", 1e-8))
    allow_unstable = bool(dcfg.get("allow_unstable", False))

    kx = np.linspace(kx_min, kx_max, n_kx)
    omega4 = lead_phonon_dispersion_3d(
        lead=lead,
        kx_points=kx,
        ky_points=[ky],
        kz_points=[kz],
        negative_tolerance=negative_tolerance,
        allow_unstable=allow_unstable,
    )
    bands = omega4[:, 0, 0, :]
    return {
        "kx": kx,
        "bands_qe": bands,
        "bands_cm1": np.asarray(qe_omega_to_cm1(bands), dtype=float),
        "summary": {
            "kpar": [ky, kz],
            "n_kx": int(n_kx),
            "n_modes": int(bands.shape[1]),
            "negative_tolerance": negative_tolerance,
            "allow_unstable": allow_unstable,
        },
    }


def _save_spectrum_data(path: Path, omegas: np.ndarray, values: np.ndarray, value_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_thz = np.asarray(qe_omega_to_thz(omegas), dtype=float)
    x_cm1 = np.asarray(qe_omega_to_cm1(omegas), dtype=float)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# omega_qe\tfreq_thz\tfreq_cm^-1\t{value_name}\n")
        for w, thz, cm1, v in zip(omegas, x_thz, x_cm1, values):
            v_str = "nan" if not np.isfinite(v) else f"{float(v):.12e}"
            fh.write(f"{float(w):.10e}\t{float(thz):.8f}\t{float(cm1):.8f}\t{v_str}\n")


def _save_dispersion_data(path: Path, kx: np.ndarray, bands_qe: np.ndarray, bands_cm1: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        cols = "\t".join(f"mode{i}" for i in range(bands_qe.shape[1]))
        cols_cm1 = "\t".join(f"mode{i}_cm1" for i in range(bands_qe.shape[1]))
        fh.write(f"# kx\tkx_over_pi\t{cols}\t{cols_cm1}\n")
        for i in range(kx.size):
            qe_vals = "\t".join(f"{float(v):.12e}" for v in bands_qe[i])
            cm_vals = "\t".join(f"{float(v):.8f}" for v in bands_cm1[i])
            fh.write(f"{float(kx[i]):.10e}\t{float(kx[i]/np.pi):.10e}\t{qe_vals}\t{cm_vals}\n")


def _plot_spectrum(path: Path, x: np.ndarray, y: np.ndarray, *, xlabel: str, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, y, color="tab:blue", lw=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)


def _plot_dispersion(path: Path, kx: np.ndarray, bands_cm1: np.ndarray, *, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for m in range(bands_cm1.shape[1]):
        ax.plot(kx / np.pi, bands_cm1[:, m], color="tab:blue", lw=1.0, alpha=0.65)
    ax.set_xlabel(r"$k_x / \pi$")
    ax.set_ylabel(r"Frequency (cm$^{-1}$)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)

def _default_template() -> dict[str, Any]:
    return {
        "run": {
            "name": "bulk_transport_run",
            "calculation": "transmission",
            "output_dir": "outputs/bulk_ifc_runs",
            "output_mode": "flat",
            "studies_root": "studies",
            "study": "bulk_study",
            "material": "material",
            "timestamped_run_dir": True,
            "write_input_snapshot": True,
            "manifest_jsonl": "manifest.jsonl",
            "write_plot": True,
            "write_data": True,
            "write_report": True,
            "x_unit": "cm-1",
        },
        "ifc": {
            "path": "si444.fc",
            "left_path": "si444.fc",
            "center_path": "si444.fc",
            "right_path": "si444.fc",
            "reader": "qe",
            "enforce_asr": False,
            "dy_cutoff": None,
            "dz_cutoff": None,
        },
        "model": {
            "n_layers": 30,
            "principal_layer_size": 2,
            "auto_principal_layer_enlargement": True,
            "infer_fc01_from_negative_dx": True,
            "enforce_hermitian_fc00": True,
            "mass_mode": "ifc",
            "drop_nyquist_transverse": False,
            "onsite_pinning": 0.0,
        },
        "kmesh": {
            "dimension": 2,
            "mode": "auto",
            "nk": 18,
            "nk_low": 24,
            "low_cm1": 120.0,
        },
        "omega": {
            "min": 0.0,
            "max": None,
            "n_points": 160,
            "auto_max": True,
            "auto_sample_nk": 12,
            "safety_factor": 1.02,
            "fmax_thz": None,
            "fmax_cm1": None,
        },
        "solver": {
            "surface_gf_method": "generalized_eigen_svd",
            "eta_scheme": "adaptive",
            "eta_fixed": 1e-4,
            "eta_values": list(DEFAULT_ETA_VALUES),
            "eta_device": None,
            "min_success_fraction": 0.7,
            "max_channel_factor": 1.5,
            "nonnegative_tolerance": 1e-8,
            "collect_rejected": False,
            "max_eta_over_omega": None,
            "eta_ratio_cm1_min": None,
            "omega_scale_mode": "none",
        },
        "dos": {
            "normalize_per_mode": False,
            "negative_tolerance": 1e-12,
        },
        "dispersion": {
            "kpar": [0.0, 0.0],
            "kx_min": 0.0,
            "kx_max": float(np.pi),
            "n_kx": 240,
            "negative_tolerance": 1e-8,
            "allow_unstable": False,
        },
    }


def write_input_template(path: str | Path) -> Path:
    out = Path(path)
    _save_json(out, _default_template())
    return out


def run_ifc_bulk(config_path: str | Path) -> dict[str, Any]:
    cfg_path = Path(config_path)
    cfg_dir = cfg_path.parent if cfg_path.parent != Path("") else Path(".")
    cfg = _load_json_config(cfg_path)
    run_cfg = dict(cfg.get("run", {}))
    calc_type = str(run_cfg.get("calculation", "transmission")).lower()
    if calc_type not in {"transmission", "dos", "dispersion"}:
        raise ValueError("run.calculation must be one of: transmission, dos, dispersion.")

    run_name = str(run_cfg.get("name", f"{calc_type}_{cfg_path.stem}"))
    run_name_safe = _sanitize_token(run_name)
    output_mode = str(run_cfg.get("output_mode", "flat")).lower()
    output_dir_flat = Path(run_cfg.get("output_dir", "outputs/bulk_ifc_runs"))
    studies_root = Path(run_cfg.get("studies_root", "studies"))
    study_name = _sanitize_token(str(run_cfg.get("study", "bulk_study")))
    material_name = _sanitize_token(str(run_cfg.get("material", "material")))
    timestamped_run_dir = bool(run_cfg.get("timestamped_run_dir", True))
    write_input_snapshot = bool(run_cfg.get("write_input_snapshot", True))
    manifest_jsonl_name = str(run_cfg.get("manifest_jsonl", "manifest.jsonl"))
    if output_mode == "flat":
        output_dir = output_dir_flat
        study_root = None
        run_dir_name = None
        manifest_path = None
    elif output_mode == "study":
        run_dir_name = f"{_utc_stamp()}_{run_name_safe}" if timestamped_run_dir else run_name_safe
        study_root = studies_root / study_name / material_name
        output_dir = study_root / "runs" / run_dir_name
        manifest_path = study_root / manifest_jsonl_name
        (study_root / "inputs" / "configs").mkdir(parents=True, exist_ok=True)
        (study_root / "inputs" / "ifc").mkdir(parents=True, exist_ok=True)
        (study_root / "analysis" / "figures").mkdir(parents=True, exist_ok=True)
        (study_root / "analysis" / "tables").mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("run.output_mode must be one of: flat, study.")
    write_plot = bool(run_cfg.get("write_plot", True))
    write_data = bool(run_cfg.get("write_data", True))
    write_report = bool(run_cfg.get("write_report", True))
    x_unit = str(run_cfg.get("x_unit", "cm-1")).lower()
    if x_unit not in {"cm-1", "thz", "qe"}:
        raise ValueError("run.x_unit must be one of: cm-1, thz, qe.")

    ifc_cfg = dict(cfg.get("ifc", {}))
    ifc_path_raw = ifc_cfg.get("path", "si444.fc")
    ifc_path = _resolve_path(cfg_dir, ifc_path_raw)
    left_path = _resolve_path(cfg_dir, ifc_cfg.get("left_path", ifc_path_raw))
    center_path = _resolve_path(cfg_dir, ifc_cfg.get("center_path", ifc_path_raw))
    right_path = _resolve_path(cfg_dir, ifc_cfg.get("right_path", ifc_path_raw))
    if left_path != center_path or right_path != center_path:
        raise NotImplementedError(
            "Different left/center/right IFC files are not implemented in this bulk runner yet. "
            "Set left_path=center_path=right_path or use ifc.path for identical leads/device."
        )
    ifc_path = center_path
    reader = str(ifc_cfg.get("reader", "qe"))
    enforce_asr = bool(ifc_cfg.get("enforce_asr", False))
    dy_cutoff = ifc_cfg.get("dy_cutoff", None)
    dz_cutoff = ifc_cfg.get("dz_cutoff", None)
    if dy_cutoff is not None:
        dy_cutoff = int(dy_cutoff)
    if dz_cutoff is not None:
        dz_cutoff = int(dz_cutoff)

    t0 = time.perf_counter()
    started = _utc_now_iso()
    ifc_path = ifc_path.resolve()
    cfg_path_abs = cfg_path.resolve()
    workspace_root = Path.cwd()
    cfg_sha256 = _sha256_file(cfg_path_abs)
    ifc_sha256 = _sha256_file(ifc_path)
    git_head = _git_head_sha(workspace_root)
    git_is_dirty = _git_dirty(workspace_root)

    ifc_raw = read_ifc(ifc_path, reader=reader)
    n_terms_raw = len(ifc_raw.terms)
    asr_residual_max = None
    if enforce_asr:
        ifc_work, asr_residual_max = enforce_translational_asr_on_self_term(ifc_raw)
    else:
        ifc_work = ifc_raw
    ifc_filtered, n_terms_removed = _apply_transverse_cutoff(ifc_work, dy_cutoff=dy_cutoff, dz_cutoff=dz_cutoff)

    mcfg = dict(cfg.get("model", {}))
    n_layers = int(mcfg.get("n_layers", 30))
    build_cfg = BuildConfig(
        onsite_pinning=float(mcfg.get("onsite_pinning", 0.0)),
        principal_layer_size=(None if mcfg.get("principal_layer_size", None) is None else int(mcfg["principal_layer_size"])),
        auto_principal_layer_enlargement=bool(mcfg.get("auto_principal_layer_enlargement", True)),
        infer_fc01_from_negative_dx=bool(mcfg.get("infer_fc01_from_negative_dx", True)),
        enforce_hermitian_fc00=bool(mcfg.get("enforce_hermitian_fc00", True)),
        dtype=str(mcfg.get("dtype", "complex128")),
    )
    params = build_material_kspace_params(ifc=ifc_filtered, config=build_cfg)
    params, mass_mode = _apply_mass_mode(params, mcfg.get("mass_mode", "ifc"))
    drop_nyquist = bool(mcfg.get("drop_nyquist_transverse", False))
    params, nyquist_info, pm_sym_info = _apply_transverse_stability_controls(
        params,
        ifc_metadata=dict(ifc_filtered.metadata),
        drop_nyquist=drop_nyquist,
    )

    lead_left, lead_right, device_to_lead_left, device_to_lead_right = _build_interface_contacts(params)
    device = material_kspace_device(n_layers=n_layers, params=params)

    kmesh_info = _prepare_kmesh(cfg)
    results: dict[str, Any]
    omega_info: dict[str, Any] | None = None
    omegas: np.ndarray | None = None

    if calc_type in {"transmission", "dos"}:
        omegas, omega_info = _build_omega_grid(cfg, ifc_filtered)
        if calc_type == "transmission":
            results = _run_transmission(
                cfg,
                device=device,
                lead_left=lead_left,
                lead_right=lead_right,
                device_to_lead_left=device_to_lead_left,
                device_to_lead_right=device_to_lead_right,
                omegas=omegas,
                kmesh=kmesh_info,
            )
        else:
            results = _run_dos(cfg, lead=lead_right, omegas=omegas, kmesh=kmesh_info)
    else:
        results = _run_dispersion(cfg, lead=lead_right)

    output_dir.mkdir(parents=True, exist_ok=True)
    if write_input_snapshot:
        _save_json(output_dir / "input.json", cfg)
    outputs: dict[str, str] = {}
    is_study_mode = output_mode == "study"

    if calc_type in {"transmission", "dos"} and omegas is not None:
        values = np.asarray(results["values"], dtype=float)
        data_default = f"{calc_type}.tsv" if is_study_mode else f"{run_name}_{calc_type}.tsv"
        data_path = output_dir / run_cfg.get("data_filename", data_default)
        if write_data:
            val_name = "transmission" if calc_type == "transmission" else "surface_dos"
            _save_spectrum_data(data_path, omegas=omegas, values=values, value_name=val_name)
            outputs["data"] = str(data_path)
        if write_plot:
            plot_default = f"{calc_type}.png" if is_study_mode else f"{run_name}_{calc_type}.png"
            plot_path = output_dir / run_cfg.get("plot_filename", plot_default)
            if x_unit == "thz":
                x = np.asarray(qe_omega_to_thz(omegas), dtype=float)
                xlabel = "Frequency (THz)"
            elif x_unit == "qe":
                x = np.asarray(omegas, dtype=float)
                xlabel = "Frequency (QE internal omega unit)"
            else:
                x = np.asarray(qe_omega_to_cm1(omegas), dtype=float)
                xlabel = r"Frequency (cm$^{-1}$)"
            ylabel = r"$T(\omega)$" if calc_type == "transmission" else "Surface DOS"
            _plot_spectrum(plot_path, x=x, y=values, xlabel=xlabel, ylabel=ylabel, title=f"{calc_type.capitalize()} ({run_name})")
            outputs["plot"] = str(plot_path)

        eta_default = f"{calc_type}_eta_diag.json" if is_study_mode else f"{run_name}_{calc_type}_eta_diag.json"
        eta_diag_path = output_dir / run_cfg.get("eta_diag_filename", eta_default)
        _save_json(
            eta_diag_path,
            {
                "eta_records": results.get("eta_records", []),
                "omega_failures": results.get("omega_failures", []),
                "n_rejected_total": results.get("n_rejected_total", 0),
            },
        )
        outputs["eta_diagnostics"] = str(eta_diag_path)
    else:
        kx = np.asarray(results["kx"], dtype=float)
        bands_qe = np.asarray(results["bands_qe"], dtype=float)
        bands_cm1 = np.asarray(results["bands_cm1"], dtype=float)
        data_default = f"{calc_type}.tsv" if is_study_mode else f"{run_name}_{calc_type}.tsv"
        data_path = output_dir / run_cfg.get("data_filename", data_default)
        if write_data:
            _save_dispersion_data(data_path, kx=kx, bands_qe=bands_qe, bands_cm1=bands_cm1)
            outputs["data"] = str(data_path)
        if write_plot:
            plot_default = f"{calc_type}.png" if is_study_mode else f"{run_name}_{calc_type}.png"
            plot_path = output_dir / run_cfg.get("plot_filename", plot_default)
            _plot_dispersion(plot_path, kx=kx, bands_cm1=bands_cm1, title=f"Dispersion ({run_name})")
            outputs["plot"] = str(plot_path)

    runtime = time.perf_counter() - t0
    finished = _utc_now_iso()

    report = {
        "run": {
            "name": run_name,
            "calculation": calc_type,
            "output_mode": output_mode,
            "input_config": str(cfg_path_abs),
            "started_utc": started,
            "finished_utc": finished,
            "runtime_seconds": float(runtime),
            "run_dir": str(output_dir.resolve()),
        },
        "ifc": {
            "path": str(ifc_path),
            "sha256": ifc_sha256,
            "reader": reader,
            "enforce_asr": enforce_asr,
            "asr_residual_max": asr_residual_max,
            "n_terms_raw": int(n_terms_raw),
            "n_terms_after_filter": int(len(ifc_filtered.terms)),
            "n_terms_removed_by_cutoff": int(n_terms_removed),
            "dy_cutoff": dy_cutoff,
            "dz_cutoff": dz_cutoff,
            "n_atoms": int(len(ifc_filtered.masses)),
            "dof_per_atom": int(ifc_filtered.dof_per_atom),
            "atom_symbols": None if ifc_filtered.atom_symbols is None else list(ifc_filtered.atom_symbols),
            "lattice_vectors": None if ifc_filtered.lattice_vectors is None else np.asarray(ifc_filtered.lattice_vectors).tolist(),
            "metadata": dict(ifc_filtered.metadata),
        },
        "model": {
            "n_layers": int(n_layers),
            "principal_layer_size": build_cfg.principal_layer_size,
            "auto_principal_layer_enlargement": bool(build_cfg.auto_principal_layer_enlargement),
            "infer_fc01_from_negative_dx": bool(build_cfg.infer_fc01_from_negative_dx),
            "enforce_hermitian_fc00": bool(build_cfg.enforce_hermitian_fc00),
            "mass_mode": mass_mode,
            "drop_nyquist_transverse": bool(drop_nyquist),
            "nyquist_filter": nyquist_info,
            "enforce_transverse_pm_symmetry": True,
            "pm_symmetry": pm_sym_info,
            "onsite_pinning": float(build_cfg.onsite_pinning),
            "ndof": int(params.ndof),
            "n_atoms_per_principal_layer": int(params.n_atoms),
        },
        "kmesh": {
            "dimension": int(kmesh_info["dim"]),
            "mode": str(kmesh_info["mode"]),
            "low_cm1": float(kmesh_info["low_cm1"]),
            "high": {k: v for k, v in kmesh_info["high"].items() if k != "kpoints"},
            "low": {k: v for k, v in kmesh_info["low"].items() if k != "kpoints"},
        },
        "omega": omega_info,
        "solver_summary": results.get("summary", {}),
        "provenance": {
            "config_sha256": cfg_sha256,
            "workspace": str(workspace_root.resolve()),
            "git_head": git_head,
            "git_dirty": git_is_dirty,
            "hostname": socket.gethostname(),
        },
        "outputs": outputs,
    }
    if output_mode == "study":
        report["study"] = {
            "root": str(study_root.resolve()) if study_root is not None else None,
            "study": study_name,
            "material": material_name,
            "run_dir_name": run_dir_name,
            "manifest_jsonl": str(manifest_path.resolve()) if manifest_path is not None else None,
        }

    if write_report:
        report_default = f"{calc_type}_report.json" if is_study_mode else f"{run_name}_{calc_type}_report.json"
        report_path = output_dir / run_cfg.get("report_filename", report_default)
        _save_json(report_path, report)
        report["outputs"]["report"] = str(report_path)
    if output_mode == "study" and manifest_path is not None:
        manifest_row = {
            "timestamp_utc": finished,
            "run_name": run_name,
            "calculation": calc_type,
            "run_dir": str(output_dir.resolve()),
            "report": report["outputs"].get("report", None),
            "ifc_path": str(ifc_path),
            "ifc_sha256": ifc_sha256,
            "git_head": git_head,
            "git_dirty": git_is_dirty,
        }
        _append_manifest_jsonl(manifest_path, manifest_row)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Path to JSON run configuration.")
    parser.add_argument("--write-template", type=Path, default=None, help="Write template config and exit.")
    args = parser.parse_args()

    if args.write_template is not None:
        out = write_input_template(args.write_template)
        print(f"Wrote template: {out}")
        return
    if args.input is None:
        raise ValueError("Provide --input <config.json> or --write-template <path>.")

    report = run_ifc_bulk(args.input)
    print(f"Run complete: {report['run']['name']}")
    print(f"calculation={report['run']['calculation']}")
    print(f"runtime_seconds={report['run']['runtime_seconds']:.3f}")
    print(f"outputs={report['outputs']}")
