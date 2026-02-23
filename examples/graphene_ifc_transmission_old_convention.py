"""Temporary benchmark script to mimic old Fortran transmission conventions.

Conventions replicated from oldfortrancode:
- fixed eta (no adaptive eta ladder)
- 1D reduced-k loop over kx in [-0.5, 0.5] with step dk
- accumulation Tw(omega) = sum_k T(omega, kx) * dk
- frequency conversion: omega_cm^-1 = omega_internal * 3634.872 / sqrt(ma_ref)

This script is for validation/benchmarking only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import transmission
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig
from negfpy.modeling.builders import build_material_kspace_params
from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.models import material_kspace_device, material_kspace_lead


def _read_ifc(path: Path, reader: str):
    if reader == "qe":
        return read_qe_q2r_ifc(path)
    if reader == "phonopy":
        return read_phonopy_ifc(path)
    return read_ifc(path, reader=reader)


def _enforce_asr_on_self_terms(ifc: IFCData) -> tuple[IFCData, float]:
    ndof = len(ifc.masses) * ifc.dof_per_atom
    terms = list(ifc.terms)
    phi_sum = np.zeros((ndof, ndof), dtype=np.complex128)
    r0_idx = None
    for idx, term in enumerate(terms):
        block = np.asarray(term.block, dtype=np.complex128)
        phi_sum += block
        if term.dx == 0 and term.dy == 0 and term.dz == 0:
            r0_idx = idx
    if r0_idx is None:
        raise ValueError("Cannot enforce ASR: missing IFC term at (0,0,0).")

    residual = np.zeros((ndof, ifc.dof_per_atom), dtype=np.complex128)
    for beta in range(ifc.dof_per_atom):
        cols = np.arange(beta, ndof, ifc.dof_per_atom)
        residual[:, beta] = np.sum(phi_sum[:, cols], axis=1)
    residual_max = float(np.max(np.abs(residual)))

    corrected = np.asarray(terms[r0_idx].block, dtype=np.complex128).copy()
    for row in range(ndof):
        atom_i = row // ifc.dof_per_atom
        for beta in range(ifc.dof_per_atom):
            col = atom_i * ifc.dof_per_atom + beta
            corrected[row, col] -= residual[row, beta]
    corrected = 0.5 * (corrected + corrected.conj().T)
    terms[r0_idx] = IFCTerm(dx=0, dy=0, dz=0, block=corrected)

    ifc_corr = IFCData(
        masses=np.asarray(ifc.masses, dtype=float),
        dof_per_atom=ifc.dof_per_atom,
        terms=tuple(terms),
        units=ifc.units,
        metadata=dict(ifc.metadata),
        lattice_vectors=ifc.lattice_vectors,
        atom_positions=ifc.atom_positions,
        atom_symbols=ifc.atom_symbols,
        index_convention=ifc.index_convention,
    )
    return ifc_corr, residual_max


def _reduced_k_grid(kmin: float, kmax: float, dk: float) -> np.ndarray:
    if dk <= 0.0:
        raise ValueError("dk must be positive.")
    n = int(np.floor((kmax - kmin) / dk + 1.0e-12)) + 1
    return kmin + dk * np.arange(n, dtype=float)


def _w_grid(wmin: float, wmax: float, dw: float) -> np.ndarray:
    if dw <= 0.0:
        raise ValueError("dw must be positive.")
    n = int(np.floor((wmax - wmin) / dw + 1.0e-12)) + 1
    return wmin + dw * np.arange(n, dtype=float)


def _old_cm1_from_internal_w(omega: np.ndarray, ma_ref: float) -> np.ndarray:
    if ma_ref <= 0.0:
        raise ValueError("ma_ref must be positive.")
    return np.asarray(omega, dtype=float) * (3634.872 / np.sqrt(float(ma_ref)))


def _load_old_tw(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    dat = np.loadtxt(path)
    if dat.ndim != 2 or dat.shape[1] < 3:
        return None
    return np.asarray(dat[:, 1], dtype=float), np.asarray(dat[:, 2], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("graphene_1L_PBE_van.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--principal-layer-size", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--enforce-asr", action="store_true")
    parser.add_argument("--w-min", type=float, default=0.001)
    parser.add_argument("--w-max", type=float, default=1.62)
    parser.add_argument("--dw", type=float, default=0.02)
    parser.add_argument("--k-min", type=float, default=-0.5, help="Reduced k-space (old convention).")
    parser.add_argument("--k-max", type=float, default=0.5, help="Reduced k-space (old convention).")
    parser.add_argument("--dk", type=float, default=0.01, help="Reduced k-space step (old convention).")
    parser.add_argument("--eta", type=float, default=1e-9)
    parser.add_argument("--eta-device", type=float, default=1e-9)
    parser.add_argument(
        "--surface-gf-method",
        type=str,
        default="generalized_eigen_svd",
        choices=["sancho_rubio", "generalized_eigen", "generalized_eigen_svd"],
    )
    parser.add_argument("--ma-ref", type=float, default=118.71, help="Reference mass used in old cm^-1 mapping.")
    parser.add_argument("--old-tw", type=Path, default=Path("oldfortrancode/Tw.txt"))
    parser.add_argument("--save", type=Path, default=Path("outputs/graphene_transmission_old_convention_compare.png"))
    parser.add_argument(
        "--save-data",
        type=Path,
        default=Path("outputs/graphene_transmission_old_convention_data.txt"),
    )
    args = parser.parse_args()

    ifc = _read_ifc(args.ifc, reader=args.reader)
    if args.enforce_asr:
        ifc, residual = _enforce_asr_on_self_terms(ifc)
        print(f"ASR enforced: yes (max pre-correction residual={residual:.6e})")
    else:
        print("ASR enforced: no")

    params = build_material_kspace_params(
        ifc=ifc,
        config=BuildConfig(
            principal_layer_size=args.principal_layer_size,
            infer_fc01_from_negative_dx=True,
        ),
    )
    lead = material_kspace_lead(params)
    device = material_kspace_device(n_layers=args.n_layers, params=params)

    omegas = _w_grid(args.w_min, args.w_max, args.dw)
    k_red = _reduced_k_grid(args.k_min, args.k_max, args.dk)

    tw = np.zeros_like(omegas, dtype=float)
    tavg = np.zeros_like(omegas, dtype=float)
    n_k = len(k_red)
    for iw, w in enumerate(omegas):
        tsum = 0.0
        count = 0
        for kr in k_red:
            # Old Fortran reduced-k convention: kk in [-0.5,0.5], phase uses 2*pi*kk.
            kpar = (float(2.0 * np.pi * kr),)
            t = transmission(
                omega=float(w),
                device=device,
                lead_left=lead,
                lead_right=lead,
                kpar=kpar,
                eta=float(args.eta),
                eta_device=float(args.eta_device),
                surface_gf_method=args.surface_gf_method,
            )
            if np.isfinite(t):
                t_pos = max(float(t), 0.0)
                tsum += t_pos
                count += 1
        tw[iw] = tsum * float(args.dk)
        tavg[iw] = tsum / float(count) if count > 0 else np.nan

    omega_cm_old = _old_cm1_from_internal_w(omegas, ma_ref=args.ma_ref)
    old = _load_old_tw(args.old_tw)

    args.save_data.parent.mkdir(parents=True, exist_ok=True)
    with args.save_data.open("w", encoding="utf-8") as fh:
        fh.write("# w_internal\tfreq_cm_oldconv\tTw_sumTdk\tTavg_k\n")
        for w, x, y1, y2 in zip(omegas, omega_cm_old, tw, tavg):
            fh.write(f"{w:.10f}\t{x:.8f}\t{y1:.12e}\t{y2:.12e}\n")
    print(f"Saved data: {args.save_data}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(omega_cm_old, tw, color="tab:blue", lw=1.6, label="Python (old convention: sum T * dk)")
    if old is not None:
        ox, oy = old
        ax.plot(ox, oy, color="tab:orange", lw=1.3, alpha=0.9, label="Old Fortran Tw.txt")
    ax.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax.set_ylabel("Transmission-like integral")
    ax.set_title("Graphene Transmission Check (Old Convention)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)
    print(f"Saved figure: {args.save}")
    print(
        "Run summary: "
        f"n_omega={len(omegas)}, n_k={n_k}, eta={args.eta:.1e}, eta_device={args.eta_device:.1e}, "
        f"method={args.surface_gf_method}, ma_ref={args.ma_ref:.6g}"
    )


if __name__ == "__main__":
    main()
