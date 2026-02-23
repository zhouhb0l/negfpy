"""Temporary script to reproduce old Fortran Tw convention from Kmatrix.dat.

Outputs a Tw-like text file with columns:
    index   omega_cm^-1   Tw
where Tw = sum_k T(omega, k) * dk using reduced-k sampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import transmission
from negfpy.core.types import Device1D, LeadBlocks


def _parse_kmatrix(path: Path) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], int]:
    mode: str | None = None
    rows11: list[tuple[int, int, int, int, float]] = []
    rows01: list[tuple[int, int, int, int, float]] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.split()
        if not s:
            continue
        if s[0] == "kR11":
            mode = "11"
            continue
        if s[0] == "kR01":
            mode = "01"
            continue
        if len(s) < 5 or mode is None:
            continue
        item = (int(s[0]), int(s[1]), int(s[2]), int(s[3]), float(s[4]))
        if mode == "11":
            rows11.append(item)
        else:
            rows01.append(item)
    if len(rows11) == 0 or len(rows01) == 0:
        raise ValueError("Invalid Kmatrix.dat: missing kR11/kR01 data.")

    ndof = max(max(r[2], r[3]) for r in rows11)
    keys = sorted({(r[0], r[1]) for r in rows11})
    kr11: dict[tuple[int, int], np.ndarray] = {(l, m): np.zeros((ndof, ndof), dtype=float) for (l, m) in keys}
    kr01: dict[tuple[int, int], np.ndarray] = {(l, m): np.zeros((ndof, ndof), dtype=float) for (l, m) in keys}

    for l, m, i, j, v in rows11:
        kr11[(l, m)][i - 1, j - 1] = v
    for l, m, i, j, v in rows01:
        if (l, m) in kr01:
            kr01[(l, m)][i - 1, j - 1] = v
    return kr11, kr01, ndof


def _r_to_k(rblocks: dict[tuple[int, int], np.ndarray], kx_reduced: float, ky_reduced: float = 0.0) -> np.ndarray:
    out = np.zeros_like(next(iter(rblocks.values())), dtype=np.complex128)
    for (l, m), blk in rblocks.items():
        out += blk * np.exp(-1j * 2.0 * np.pi * (float(l) * kx_reduced + float(m) * ky_reduced))
    return out


def _grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("step must be positive.")
    n = int(np.floor((stop - start) / step + 1.0e-12)) + 1
    return start + step * np.arange(n, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmatrix", type=Path, default=Path("oldfortrancode/force/Kmatrix.dat"))
    parser.add_argument("--w-min", type=float, default=0.001)
    parser.add_argument("--w-max", type=float, default=1.62)
    parser.add_argument("--dw", type=float, default=0.02)
    parser.add_argument("--k-min", type=float, default=-0.5)
    parser.add_argument("--k-max", type=float, default=0.5)
    parser.add_argument("--dk", type=float, default=0.01)
    parser.add_argument("--eta", type=float, default=1e-9)
    parser.add_argument("--eta-device", type=float, default=1e-9)
    parser.add_argument(
        "--surface-gf-method",
        type=str,
        default="generalized_eigen_svd",
        choices=["sancho_rubio", "generalized_eigen", "generalized_eigen_svd", "legacy_eigen_svd"],
    )
    parser.add_argument("--ma-ref", type=float, default=12.0)
    parser.add_argument(
        "--ifc-file",
        type=Path,
        default=Path("graphene_1L_PBE_van.fc"),
        help="Used to infer lattice constant a from QE .fc header when --a-nm is not set.",
    )
    parser.add_argument("--a-nm", type=float, default=None, help="In-plane graphene lattice constant in nm.")
    parser.add_argument(
        "--thickness-nm",
        type=float,
        default=0.335,
        help="Graphene effective thickness in nm (literature value ~0.335 nm).",
    )
    parser.add_argument("--old-tw", type=Path, default=Path("oldfortrancode/Tw.txt"))
    parser.add_argument("--save-tw", type=Path, default=Path("outputs/Tw_reproduced_oldconvention.txt"))
    parser.add_argument("--save-fig", type=Path, default=Path("outputs/Tw_reproduced_oldconvention_compare.png"))
    parser.add_argument(
        "--save-area-fig",
        type=Path,
        default=Path("outputs/Tw_reproduced_oldconvention_per_area_nm2.png"),
    )
    args = parser.parse_args()

    kr11, kr01, ndof = _parse_kmatrix(args.kmatrix)
    ws = _grid(args.w_min, args.w_max, args.dw)
    ks = _grid(args.k_min, args.k_max, args.dk)
    tw = np.zeros_like(ws, dtype=float)

    for iw, w in enumerate(ws):
        tsum = 0.0
        for kr in ks:
            d00 = _r_to_k(kr11, float(kr), 0.0)
            d01 = _r_to_k(kr01, float(kr), 0.0)
            lead = LeadBlocks(d00=d00, d01=d01, d10=d01.conj().T)
            device = Device1D(onsite_blocks=[d00], coupling_blocks=[])
            t = transmission(
                omega=float(w),
                device=device,
                lead_left=lead,
                lead_right=lead,
                eta=float(args.eta),
                eta_device=float(args.eta_device),
                surface_gf_method=args.surface_gf_method,
            )
            if np.isfinite(t):
                tsum += max(float(t), 0.0)
        tw[iw] = tsum * float(args.dk)

    omega_cm = ws * (3634.872 / np.sqrt(float(args.ma_ref)))
    idx = np.arange(1, len(ws) + 1, dtype=int)

    bohr_to_nm = 5.29177210903e-2
    if args.a_nm is not None:
        a_nm = float(args.a_nm)
    else:
        first = args.ifc_file.read_text(encoding="utf-8").splitlines()[0].split()
        if len(first) < 4:
            raise ValueError("Failed to parse lattice constant from IFC header.")
        celldm1_bohr = float(first[3])
        a_nm = celldm1_bohr * bohr_to_nm
    if a_nm <= 0.0:
        raise ValueError("a-nm must be positive.")
    if args.thickness_nm <= 0.0:
        raise ValueError("thickness-nm must be positive.")

    # For hexagonal graphene with transport along a1:
    # transverse width per unit cell = a * sin(60) = sqrt(3)/2 * a.
    width_nm = 0.5 * np.sqrt(3.0) * a_nm
    area_cross_nm2 = width_nm * float(args.thickness_nm)
    tw_per_area_nm2 = tw / area_cross_nm2

    args.save_tw.parent.mkdir(parents=True, exist_ok=True)
    with args.save_tw.open("w", encoding="utf-8") as fh:
        fh.write("# index omega_cm^-1 Tw_sumTdk Tw_per_area_nm^-2\n")
        for i, x, y, ya in zip(idx, omega_cm, tw, tw_per_area_nm2):
            fh.write(f"{i:12d} {x:20.12f} {y:20.12e} {ya:20.12e}\n")
    print(f"Saved Tw-like file: {args.save_tw}")

    old = None
    if args.old_tw.exists():
        old = np.loadtxt(args.old_tw)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(omega_cm, tw, color="tab:blue", lw=1.6, label="Reproduced (Python old convention)")
    if old is not None and old.ndim == 2 and old.shape[1] >= 3:
        ox, oy = old[:, 1], old[:, 2]
        ax.plot(ox, oy, color="tab:orange", lw=1.3, alpha=0.9, label="Old Tw.txt")
        yi = np.interp(ox, omega_cm, tw, left=np.nan, right=np.nan)
        m = np.isfinite(yi) & np.isfinite(oy) & (yi > 1e-15)
        if np.any(m):
            corr = float(np.corrcoef(yi[m], oy[m])[0, 1])
            scale = float(np.median(oy[m] / yi[m]))
            print(f"Comparison to old Tw: corr={corr:.6f}, median(old/reproduced)={scale:.6g}, n={int(np.sum(m))}")
    ax.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax.set_ylabel("Tw")
    ax.set_title("Old Convention Tw Reproduction")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    args.save_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save_fig, dpi=220)
    print(f"Saved figure: {args.save_fig}")
    fig2, ax2 = plt.subplots(figsize=(8.4, 5.0))
    ax2.plot(omega_cm, tw_per_area_nm2, color="tab:blue", lw=1.6)
    ax2.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax2.set_ylabel(r"$T/A$ (nm$^{-2}$)")
    ax2.set_title("Old Convention Reproduction: Transmission per Area")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    args.save_area_fig.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(args.save_area_fig, dpi=220)
    print(f"Saved per-area figure: {args.save_area_fig}")
    print(
        f"Run summary: ndof={ndof}, n_omega={len(ws)}, n_k={len(ks)}, "
        f"eta={args.eta:.1e}, eta_device={args.eta_device:.1e}, method={args.surface_gf_method}"
    )
    print(
        "Area normalization: "
        f"a={a_nm:.6f} nm, thickness={args.thickness_nm:.6f} nm, "
        f"width_per_uc={width_nm:.6f} nm, area_per_uc={area_cross_nm2:.6f} nm^2"
    )


if __name__ == "__main__":
    main()
