"""Benchmark-only transmission comparison for graphene IFC.

This script does not modify core algorithms. It compares:
1) Different surface Green's-function solvers on the same setup.
2) Old Fortran Tw.txt (if available).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import transmission_kavg, transmission_kavg_adaptive
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, qe_ev_to_omega, qe_omega_to_cm1
from negfpy.modeling.builders import build_material_kspace_params
from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.models import material_kspace_device, material_kspace_lead


METHOD_LABELS = {
    "sancho_rubio": "Sancho-Rubio",
    "generalized_eigen": "Generalized Eigen",
    "generalized_eigen_svd": "Generalized Eigen (SVD)",
}


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


def _load_old_tw(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    dat = np.loadtxt(path)
    if dat.ndim != 2 or dat.shape[1] < 3:
        return None
    return np.asarray(dat[:, 1], float), np.asarray(dat[:, 2], float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("graphene.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--principal-layer-size", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1, help="Benchmark with N_C=1 by default.")
    parser.add_argument("--enforce-asr", action="store_true")
    parser.add_argument("--cm1-max", type=float, default=1700.0)
    parser.add_argument("--n-omega", type=int, default=220)
    parser.add_argument("--nk", type=int, default=24, help="ky points for each solver profile.")
    parser.add_argument(
        "--profile",
        type=str,
        default="adaptive",
        choices=["adaptive", "fixed"],
        help="Comparison profile: adaptive eta ladder or fixed eta.",
    )
    parser.add_argument("--eta-fixed", type=float, default=1e-9, help="Lead eta used for --profile fixed.")
    parser.add_argument("--eta-device", type=float, default=1e-8, help="Device eta for all methods.")
    parser.add_argument(
        "--omega-scale-mode",
        type=str,
        default="none",
        choices=["none", "ev"],
        help="Internal numerical scaling. 'ev' solves in E=ħω (eV) units.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["sancho_rubio", "generalized_eigen", "generalized_eigen_svd"],
        choices=["sancho_rubio", "generalized_eigen", "generalized_eigen_svd", "legacy_eigen_svd"],
        help="Surface GF solvers to compare.",
    )
    parser.add_argument("--old-tw", type=Path, default=Path("oldfortrancode/Tw.txt"))
    parser.add_argument(
        "--save",
        type=Path,
        default=Path("outputs/graphene_transmission_benchmark_sgf_compare_cm1_1700.png"),
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=Path("outputs/graphene_transmission_benchmark_sgf_compare_data.txt"),
    )
    args = parser.parse_args()
    omega_scale = None
    if args.omega_scale_mode == "ev":
        omega_scale = float(np.asarray(qe_ev_to_omega(1.0), dtype=float))
        print(f"Omega scaling enabled: mode=ev, omega_scale={omega_scale:.6e} (internal omega per 1 eV)")

    ifc = _read_ifc(args.ifc, reader=args.reader)
    if args.enforce_asr:
        ifc, residual = _enforce_asr_on_self_terms(ifc)
        print(f"ASR enforced: yes (max pre-correction residual={residual:.6e})")
    else:
        print("ASR enforced: no")

    params = build_material_kspace_params(
        ifc=ifc,
        config=BuildConfig(principal_layer_size=args.principal_layer_size, infer_fc01_from_negative_dx=True),
    )
    lead = material_kspace_lead(params)
    device = material_kspace_device(n_layers=args.n_layers, params=params)

    omegas_cm = np.linspace(0.0, float(args.cm1_max), int(args.n_omega))
    cm_per_qe = float(np.asarray(qe_omega_to_cm1(1.0), dtype=float))
    omegas = omegas_cm / cm_per_qe

    k_red = np.linspace(-0.5, 0.5, int(args.nk), endpoint=True)
    kpts = [(float(2.0 * np.pi * kr),) for kr in k_red]
    method_list = list(dict.fromkeys(args.methods))
    results: dict[str, np.ndarray] = {}
    success_stats: dict[str, float] = {}

    for method in method_list:
        tvals = np.zeros_like(omegas, dtype=float)
        n_ok = 0
        for i, w in enumerate(omegas):
            if w <= 0.0:
                continue
            try:
                if args.profile == "fixed":
                    t = transmission_kavg(
                        omega=float(w),
                        device=device,
                        lead_left=lead,
                        lead_right=lead,
                        kpoints=kpts,
                        eta=float(args.eta_fixed),
                        eta_device=float(args.eta_device),
                        surface_gf_method=method,
                        omega_scale=omega_scale,
                    )
                    tvals[i] = max(float(t), 0.0)
                else:
                    tavg, _ = transmission_kavg_adaptive(
                        omega=float(w),
                        device=device,
                        lead_left=lead,
                        lead_right=lead,
                        kpoints=kpts,
                        eta_values=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                        eta_device=float(args.eta_device),
                        min_success_fraction=0.7,
                        surface_gf_method=method,
                        omega_scale=omega_scale,
                        nonnegative_tolerance=1e-8,
                        max_channel_factor=1.5,
                        collect_rejected=False,
                    )
                    tvals[i] = max(float(tavg), 0.0)
                n_ok += 1
            except Exception:
                tvals[i] = np.nan
        results[method] = tvals
        success_stats[method] = float(n_ok / max(int(np.count_nonzero(omegas > 0.0)), 1))

    old = _load_old_tw(args.old_tw)

    # Save comparison data.
    args.save_data.parent.mkdir(parents=True, exist_ok=True)
    with args.save_data.open("w", encoding="utf-8") as fh:
        col_names = "\t".join(f"T_{m}" for m in method_list)
        fh.write(f"# omega_cm^-1\t{col_names}\n")
        for i, x in enumerate(omegas_cm):
            cols = "\t".join(f"{float(results[m][i]):.12e}" for m in method_list)
            fh.write(f"{x:.8f}\t{cols}\n")
    print(f"Saved data: {args.save_data}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    style_cycle = [
        ("tab:blue", "-"),
        ("tab:green", "--"),
        ("tab:red", "-."),
    ]
    for i, method in enumerate(method_list):
        color, ls = style_cycle[i % len(style_cycle)]
        label = METHOD_LABELS.get(method, method)
        ax.plot(omegas_cm, results[method], color=color, lw=1.5, ls=ls, label=label)
    if old is not None:
        ox, oy = old
        m = ox <= float(args.cm1_max)
        ax.plot(ox[m], oy[m], color="tab:orange", lw=1.3, alpha=0.9, label="Old Fortran Tw.txt")
    ax.set_xlim(0.0, float(args.cm1_max))
    ax.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax.set_ylabel("Transmission")
    profile_desc = "adaptive eta ladder" if args.profile == "adaptive" else f"fixed eta={args.eta_fixed:.1e}"
    ax.set_title(f"Graphene Transmission: SGF Solver Comparison ({profile_desc})")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)
    print(f"Saved figure: {args.save}")
    for method in method_list:
        print(f"{method}: success_fraction={success_stats[method]:.3f}")


if __name__ == "__main__":
    main()
