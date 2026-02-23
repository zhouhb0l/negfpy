"""Temporary one-to-one benchmark using old Kmatrix.dat conventions.

This script uses ``oldfortrancode/force/Kmatrix.dat`` directly and mirrors the
old transmission accumulation:
- reduced-k loop in [-0.5, 0.5] with step dk
- Tw(omega) = sum_k T(omega, k) * dk
- cm^-1 mapping: omega_cm^-1 = omega_internal * 3634.872 / sqrt(ma_ref)

Use for cross-check only; do not use as the default workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import transmission
from negfpy.core.types import Device1D, LeadBlocks


def _parse_kmatrix(path: Path) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    mode: str | None = None
    rows11: list[tuple[int, int, int, int, float]] = []
    rows01: list[tuple[int, int, int, int, float]] = []
    for line in lines:
        s = line.split()
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
        l = int(s[0])
        m = int(s[1])
        i = int(s[2])
        j = int(s[3])
        v = float(s[4])
        if mode == "11":
            rows11.append((l, m, i, j, v))
        else:
            rows01.append((l, m, i, j, v))

    if len(rows11) == 0 or len(rows01) == 0:
        raise ValueError("Failed to parse Kmatrix.dat (missing kR11/kR01 rows).")

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


def _transform_r_to_k(
    rblocks: dict[tuple[int, int], np.ndarray],
    kx_reduced: float,
    ky_reduced: float = 0.0,
) -> np.ndarray:
    # old 2D convention in transform.f90:
    # phase = exp(-i * 2*pi * (l*kx + m*ky))
    out = np.zeros_like(next(iter(rblocks.values())), dtype=np.complex128)
    for (l, m), blk in rblocks.items():
        phase = np.exp(-1j * 2.0 * np.pi * (float(l) * kx_reduced + float(m) * ky_reduced))
        out += blk * phase
    return out


def _load_old_tw(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    dat = np.loadtxt(path)
    if dat.ndim != 2 or dat.shape[1] < 3:
        return None
    return np.asarray(dat[:, 1], dtype=float), np.asarray(dat[:, 2], dtype=float)


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
        "--python-scale",
        type=float,
        default=1.0,
        help="Multiply Python Tw by this factor for visual comparison.",
    )
    parser.add_argument(
        "--auto-scale-to-old",
        action="store_true",
        help="Auto-fit a scalar factor to old Tw.txt over overlap range (median old/python).",
    )
    parser.add_argument("--old-tw", type=Path, default=Path("oldfortrancode/Tw.txt"))
    parser.add_argument(
        "--report-prefactor",
        action="store_true",
        help="Print normalization diagnostics: sum(T)dk vs k-average vs BZ-normalized integral.",
    )
    parser.add_argument("--save", type=Path, default=Path("outputs/graphene_kmatrix_legacy_benchmark.png"))
    parser.add_argument(
        "--save-data",
        type=Path,
        default=Path("outputs/graphene_kmatrix_legacy_benchmark_data.txt"),
    )
    args = parser.parse_args()

    kr11, kr01, ndof = _parse_kmatrix(args.kmatrix)
    ws = _grid(args.w_min, args.w_max, args.dw)
    ks = _grid(args.k_min, args.k_max, args.dk)

    tw = np.zeros_like(ws, dtype=float)
    tavg = np.zeros_like(ws, dtype=float)
    tbz = np.zeros_like(ws, dtype=float)
    n_ok_arr = np.zeros_like(ws, dtype=int)
    for iw, w in enumerate(ws):
        tsum = 0.0
        n_ok = 0
        for kx in ks:
            d00 = _transform_r_to_k(kr11, kx_reduced=float(kx), ky_reduced=0.0)
            d01 = _transform_r_to_k(kr01, kx_reduced=float(kx), ky_reduced=0.0)
            lead = LeadBlocks(d00=d00, d01=d01, d10=d01.conj().T)
            # old center uses one layer and same onsite block for this graphene setup
            device = Device1D(onsite_blocks=[d00], coupling_blocks=[])
            t = transmission(
                omega=float(w),
                device=device,
                lead_left=lead,
                lead_right=lead,
                kpar=None,
                eta=float(args.eta),
                eta_device=float(args.eta_device),
                surface_gf_method=args.surface_gf_method,
            )
            if np.isfinite(t):
                tsum += max(float(t), 0.0)
                n_ok += 1
        tw[iw] = tsum * float(args.dk)
        tavg[iw] = tsum / float(n_ok) if n_ok > 0 else np.nan
        tbz[iw] = tw[iw] / float(args.k_max - args.k_min) if args.k_max > args.k_min else np.nan
        n_ok_arr[iw] = int(n_ok)

    x_cm = ws * (3634.872 / np.sqrt(float(args.ma_ref)))
    old = _load_old_tw(args.old_tw)
    scale = float(args.python_scale)
    if args.auto_scale_to_old and old is not None:
        ox, oy = old
        yi = np.interp(ox, x_cm, tw, left=np.nan, right=np.nan)
        mask = np.isfinite(yi) & np.isfinite(oy) & (yi > 1e-15)
        if np.any(mask):
            scale = float(np.median(oy[mask] / yi[mask]))
            print(f"Auto-fit python scale factor: {scale:.6g}")
        else:
            print("Auto-fit skipped: insufficient overlap with old Tw.")
    tw_scaled = tw * scale

    args.save_data.parent.mkdir(parents=True, exist_ok=True)
    with args.save_data.open("w", encoding="utf-8") as fh:
        fh.write("# w_internal\tfreq_cm_oldconv\tTw_sumTdk\tTw_sumTdk_scaled\tTavg_k\tTw_bz_norm\tn_k_success\n")
        for w, x, y1, ys, y2, y3, nkok in zip(ws, x_cm, tw, tw_scaled, tavg, tbz, n_ok_arr):
            fh.write(f"{w:.10f}\t{x:.8f}\t{y1:.12e}\t{ys:.12e}\t{y2:.12e}\t{y3:.12e}\t{nkok:d}\n")
    print(f"Saved data: {args.save_data}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    if abs(scale - 1.0) > 1e-12:
        ax.plot(
            x_cm,
            tw_scaled,
            color="tab:blue",
            lw=1.6,
            label=f"Python from Kmatrix (sum T*dk) x {scale:.3g}",
        )
        ax.plot(x_cm, tw, color="tab:blue", lw=1.0, ls="--", alpha=0.55, label="Python raw")
    else:
        ax.plot(x_cm, tw, color="tab:blue", lw=1.6, label="Python from Kmatrix (sum T*dk)")
    if old is not None:
        ox, oy = old
        ax.plot(ox, oy, color="tab:orange", lw=1.3, alpha=0.9, label="Old Fortran Tw.txt")
        ax.set_xlim(0.0, max(float(np.max(x_cm)), float(np.max(ox))))
    else:
        ax.set_xlim(0.0, float(np.max(x_cm)))
    ax.set_xlabel(r"Frequency (cm$^{-1}$)")
    ax.set_ylabel("Transmission-like integral")
    ax.set_title("Legacy One-to-One Benchmark (Kmatrix)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)
    print(f"Saved figure: {args.save}")
    print(
        "Run summary: "
        f"ndof={ndof}, n_omega={len(ws)}, n_k={len(ks)}, eta={args.eta:.1e}, "
        f"eta_device={args.eta_device:.1e}, method={args.surface_gf_method}, ma_ref={args.ma_ref:.6g}"
    )
    if args.report_prefactor:
        k_span = float(args.k_max - args.k_min)
        print(
            "Normalization diagnostics: "
            f"n_k={len(ks)}, dk={args.dk:.6g}, n_k*dk={len(ks)*args.dk:.6g}, k_span={k_span:.6g}"
        )
        mask = np.isfinite(tavg) & np.isfinite(tw) & (tavg > 1e-20)
        if np.any(mask):
            ratio = tw[mask] / tavg[mask]
            print(
                "Tw_sumTdk / Tavg_k: "
                f"median={float(np.median(ratio)):.6g}, min={float(np.min(ratio)):.6g}, "
                f"max={float(np.max(ratio)):.6g}"
            )
        if old is not None:
            ox, oy = old
            y_old_on_x = np.interp(ox, x_cm, tw, left=np.nan, right=np.nan)
            y_avg_on_x = np.interp(ox, x_cm, tavg, left=np.nan, right=np.nan)
            y_bz_on_x = np.interp(ox, x_cm, tbz, left=np.nan, right=np.nan)
            for name, yy in [
                ("sumTdk", y_old_on_x),
                ("kavg", y_avg_on_x),
                ("bz_norm", y_bz_on_x),
            ]:
                m = np.isfinite(yy) & np.isfinite(oy) & (yy > 1e-15)
                if np.any(m):
                    sf = float(np.median(oy[m] / yy[m]))
                    corr = float(np.corrcoef(oy[m], yy[m])[0, 1])
                    print(f"Old/Python ({name}): median_scale={sf:.6g}, corr={corr:.6f}, n={int(np.sum(m))}")


if __name__ == "__main__":
    main()
