"""Silicon lead dispersion check from IFC files (QE q2r format by default)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import lead_dynamical_matrix, lead_phonon_dispersion_3d
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1, qe_omega_to_thz
from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.models import material_kspace_lead


def _enforce_asr_on_self_terms(ifc: IFCData) -> IFCData:
    """Project translational ASR by correcting only the R=(0,0,0) self block."""

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
        raise ValueError("Cannot enforce ASR: missing IFC term at translation (0,0,0).")

    # Residual for each row DOF and Cartesian direction beta.
    residual = np.zeros((ndof, ifc.dof_per_atom), dtype=np.complex128)
    for beta in range(ifc.dof_per_atom):
        cols = np.arange(beta, ndof, ifc.dof_per_atom)
        residual[:, beta] = np.sum(phi_sum[:, cols], axis=1)

    corrected = np.asarray(terms[r0_idx].block, dtype=np.complex128).copy()
    for row in range(ndof):
        atom_i = row // ifc.dof_per_atom
        for beta in range(ifc.dof_per_atom):
            col = atom_i * ifc.dof_per_atom + beta
            corrected[row, col] -= residual[row, beta]
    corrected = 0.5 * (corrected + corrected.conj().T)
    terms[r0_idx] = IFCTerm(dx=0, dy=0, dz=0, block=corrected)

    md = dict(ifc.metadata)
    md["asr_enforced"] = True
    return IFCData(
        masses=np.asarray(ifc.masses, dtype=float),
        dof_per_atom=ifc.dof_per_atom,
        terms=tuple(terms),
        units=ifc.units,
        metadata=md,
        lattice_vectors=ifc.lattice_vectors,
        atom_positions=ifc.atom_positions,
        atom_symbols=ifc.atom_symbols,
        index_convention=ifc.index_convention,
    )


def _load_lead(
    ifc_path: Path,
    reader: str,
    principal_layer_size: int | None,
    onsite_pinning: float,
    enforce_asr: bool,
) -> tuple[object, int]:
    ifc = _read_ifc(ifc_path=ifc_path, reader=reader, enforce_asr=enforce_asr)
    config = BuildConfig(
        onsite_pinning=float(onsite_pinning),
        principal_layer_size=principal_layer_size,
        auto_principal_layer_enlargement=(principal_layer_size is None),
    )
    params = build_material_kspace_params(ifc=ifc, config=config)
    lead = material_kspace_lead(params)
    return lead, params.ndof


def _read_ifc(ifc_path: Path, reader: str, enforce_asr: bool) -> IFCData:
    if reader == "qe":
        ifc = read_qe_q2r_ifc(ifc_path)
    elif reader == "phonopy":
        ifc = read_phonopy_ifc(ifc_path)
    else:
        ifc = read_ifc(ifc_path, reader=reader)
    if enforce_asr:
        ifc = _enforce_asr_on_self_terms(ifc)
    return ifc


def _direct_ifc_bands_vs_kx(
    ifc: IFCData,
    kx: np.ndarray,
    ky: float,
    kz: float,
    negative_tolerance: float,
) -> np.ndarray:
    ndof = len(ifc.masses) * ifc.dof_per_atom
    mvec = np.repeat(np.asarray(ifc.masses, dtype=float), ifc.dof_per_atom)
    minv = np.diag(1.0 / np.sqrt(mvec))
    out = np.zeros((kx.size, ndof), dtype=float)
    min_w2 = np.inf
    for ix, kxv in enumerate(kx):
        phi = np.zeros((ndof, ndof), dtype=np.complex128)
        for term in ifc.terms:
            phase = np.exp(1j * (kxv * term.dx + ky * term.dy + kz * term.dz))
            phi += np.asarray(term.block, dtype=np.complex128) * phase
        dmat = minv @ phi @ minv
        dmat = 0.5 * (dmat + dmat.conj().T)
        vals = np.linalg.eigvalsh(dmat)
        min_w2 = min(min_w2, float(np.min(vals)))
        vals = np.where((vals < 0.0) & (vals >= -negative_tolerance), 0.0, vals)
        out[ix, :] = np.sqrt(np.clip(vals.real, 0.0, None))
    if min_w2 < -negative_tolerance:
        raise ValueError(
            "direct_ifc mode found unstable omega^2 below tolerance: "
            f"min omega^2={min_w2:.6e}, tolerance={negative_tolerance:.6e}."
        )
    return out


def _convert_units(omega: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    if unit == "internal":
        return omega, "omega (internal)"
    if unit == "thz":
        return np.asarray(qe_omega_to_thz(omega), dtype=float), "f (THz)"
    if unit == "cm-1":
        return np.asarray(qe_omega_to_cm1(omega), dtype=float), r"$\tilde{\nu}$ (cm$^{-1}$)"
    raise ValueError(f"Unsupported unit: {unit}")


def _bands_vs_kx(
    lead: object,
    kx: np.ndarray,
    ky: float,
    kz: float,
    allow_unstable: bool,
    negative_tolerance: float,
) -> np.ndarray:
    disp = lead_phonon_dispersion_3d(
        lead=lead,
        kx_points=kx,
        ky_points=[ky],
        kz_points=[kz],
        allow_unstable=allow_unstable,
        negative_tolerance=negative_tolerance,
    )
    return disp[:, 0, 0, :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-ifc", type=Path, default=Path("studies/silicon_bulk_2026q1/inputs/ifc/si444.fc"))
    parser.add_argument("--right-ifc", type=Path, default=None)
    parser.add_argument("--reader", type=str, default="qe", help="IFC reader key (default: qe)")
    parser.add_argument("--principal-layer-size", type=int, default=None)
    parser.add_argument("--onsite-pinning", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        type=str,
        default="direct_ifc",
        choices=["direct_ifc", "lead_nn"],
        help="Use direct IFC Fourier sum or nearest-neighbor lead mapping",
    )
    parser.add_argument("--ky", type=float, default=0.0, help="Fixed ky cut for dispersion")
    parser.add_argument("--kz", type=float, default=0.0, help="Fixed kz cut for dispersion")
    parser.add_argument("--nkx", type=int, default=241, help="Number of kx points in [-pi, pi]")
    parser.add_argument(
        "--unit",
        type=str,
        default="cm-1",
        choices=["internal", "thz", "cm-1"],
        help="Y-axis unit for plotting",
    )
    parser.add_argument("--allow-unstable", action="store_true")
    parser.add_argument("--enforce-asr", action="store_true", help="Apply translational ASR correction to IFC")
    parser.add_argument(
        "--negative-tolerance",
        type=float,
        default=1e-8,
        help="Tolerance for small negative omega^2 before raising instability errors",
    )
    parser.add_argument("--save", type=Path, default=None, help="Output image path; if omitted, show interactively")
    args = parser.parse_args()

    right_ifc = args.right_ifc if args.right_ifc is not None else args.left_ifc
    if not args.left_ifc.exists():
        raise FileNotFoundError(f"Left IFC file not found: {args.left_ifc}")
    if not right_ifc.exists():
        raise FileNotFoundError(f"Right IFC file not found: {right_ifc}")

    kx = np.linspace(-np.pi, np.pi, args.nkx)
    if args.mode == "direct_ifc":
        left_ifc = _read_ifc(args.left_ifc, reader=args.reader, enforce_asr=args.enforce_asr)
        right_ifc_data = _read_ifc(right_ifc, reader=args.reader, enforce_asr=args.enforce_asr)
        left_bands = _direct_ifc_bands_vs_kx(
            left_ifc,
            kx,
            ky=args.ky,
            kz=args.kz,
            negative_tolerance=args.negative_tolerance,
        )
        right_bands = _direct_ifc_bands_vs_kx(
            right_ifc_data,
            kx,
            ky=args.ky,
            kz=args.kz,
            negative_tolerance=args.negative_tolerance,
        )
        gamma_eval = np.linalg.eigvalsh(
            0.5
            * (
                np.diag(1.0 / np.sqrt(np.repeat(np.asarray(left_ifc.masses, dtype=float), left_ifc.dof_per_atom)))
                @ sum(
                    np.asarray(t.block, dtype=np.complex128)
                    * np.exp(1j * (args.ky * t.dy + args.kz * t.dz))
                    for t in left_ifc.terms
                )
                @ np.diag(1.0 / np.sqrt(np.repeat(np.asarray(left_ifc.masses, dtype=float), left_ifc.dof_per_atom)))
                + (
                    np.diag(1.0 / np.sqrt(np.repeat(np.asarray(left_ifc.masses, dtype=float), left_ifc.dof_per_atom)))
                    @ sum(
                        np.asarray(t.block, dtype=np.complex128)
                        * np.exp(1j * (args.ky * t.dy + args.kz * t.dz))
                        for t in left_ifc.terms
                    )
                    @ np.diag(1.0 / np.sqrt(np.repeat(np.asarray(left_ifc.masses, dtype=float), left_ifc.dof_per_atom)))
                ).conj().T
            )
        )
    else:
        left_lead, left_ndof = _load_lead(
            ifc_path=args.left_ifc,
            reader=args.reader,
            principal_layer_size=args.principal_layer_size,
            onsite_pinning=args.onsite_pinning,
            enforce_asr=args.enforce_asr,
        )
        right_lead, right_ndof = _load_lead(
            ifc_path=right_ifc,
            reader=args.reader,
            principal_layer_size=args.principal_layer_size,
            onsite_pinning=args.onsite_pinning,
            enforce_asr=args.enforce_asr,
        )
        if left_ndof != right_ndof:
            raise ValueError(f"Left/right mode count mismatch: {left_ndof} vs {right_ndof}")
        left_bands = _bands_vs_kx(
            left_lead,
            kx,
            ky=args.ky,
            kz=args.kz,
            allow_unstable=args.allow_unstable,
            negative_tolerance=args.negative_tolerance,
        )
        right_bands = _bands_vs_kx(
            right_lead,
            kx,
            ky=args.ky,
            kz=args.kz,
            allow_unstable=args.allow_unstable,
            negative_tolerance=args.negative_tolerance,
        )
        gamma_eval = np.linalg.eigvalsh(
            lead_dynamical_matrix(left_lead, kx=0.0, kpar=(args.ky, args.kz))
        )

    left_plot, ylabel = _convert_units(left_bands, args.unit)
    right_plot, _ = _convert_units(right_bands, args.unit)
    max_abs_diff = float(np.max(np.abs(left_plot - right_plot)))
    gamma_omega = np.sqrt(np.clip(gamma_eval, 0.0, None))
    gamma_plot, _ = _convert_units(gamma_omega, args.unit)

    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for m in range(left_plot.shape[1]):
        ax[0].plot(kx, left_plot[:, m], color="tab:blue", alpha=0.50, lw=1.0)
        ax[1].plot(kx, right_plot[:, m], color="tab:orange", alpha=0.50, lw=1.0)
    ax[0].set_title("Left Lead Dispersion")
    ax[1].set_title("Right Lead Dispersion")
    for a in ax:
        a.set_xlabel(r"$k_x$ (rad)")
        a.grid(alpha=0.25)
    ax[0].set_ylabel(ylabel)
    fig.suptitle(
        f"Silicon IFC lead check: ky={args.ky:.4f}, kz={args.kz:.4f}, modes={left_plot.shape[1]}, "
        f"max |L-R|={max_abs_diff:.3e}"
    )
    fig.tight_layout()

    print(f"Left IFC : {args.left_ifc}")
    print(f"Right IFC: {right_ifc}")
    print(f"Reader   : {args.reader}")
    print(f"Mode     : {args.mode}")
    print(f"Modes    : {left_plot.shape[1]}")
    print(f"ky, kz   : ({args.ky}, {args.kz})")
    print(f"ASR      : {'on' if args.enforce_asr else 'off'}")
    print(f"neg_tol  : {args.negative_tolerance}")
    print(f"Gamma lowest 6 ({args.unit}): {np.sort(gamma_plot)[:6]}")
    print(f"Max |left-right| in plotted unit: {max_abs_diff:.6e}")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=200)
        print(f"Saved figure to: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
