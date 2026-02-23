"""Plot graphene phonon dispersion along Gamma-K-M-Gamma from IFC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.io import read_ifc
from negfpy.modeling import enforce_translational_asr_on_self_term, qe_omega_to_cm1, qe_omega_to_thz


def _direct_dynamical(ifc, kx: float, ky: float, kz: float) -> np.ndarray:
    ndof = len(ifc.masses) * ifc.dof_per_atom
    m = np.repeat(np.asarray(ifc.masses, dtype=float), ifc.dof_per_atom)
    minv = np.diag(1.0 / np.sqrt(m))
    phi = np.zeros((ndof, ndof), dtype=np.complex128)
    for t in ifc.terms:
        phi += np.asarray(t.block, dtype=np.complex128) * np.exp(1j * (kx * t.dx + ky * t.dy + kz * t.dz))
    dmat = minv @ phi @ minv
    return 0.5 * (dmat + dmat.conj().T)


def _segment(a: tuple[float, float, float], b: tuple[float, float, float], n: int) -> np.ndarray:
    ta = np.asarray(a, dtype=float)
    tb = np.asarray(b, dtype=float)
    if n <= 1:
        return np.asarray([ta], dtype=float)
    s = np.linspace(0.0, 1.0, n, endpoint=False)
    return ta[None, :] * (1.0 - s[:, None]) + tb[None, :] * s[:, None]


def _convert_omega(omega: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
    if unit == "thz":
        return np.asarray(qe_omega_to_thz(omega), dtype=float), "Frequency (THz)"
    if unit == "cm-1":
        return np.asarray(qe_omega_to_cm1(omega), dtype=float), "Frequency (cm^-1)"
    return np.asarray(omega, dtype=float), "Frequency (QE internal omega unit)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("graphene.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--nseg", type=int, default=120, help="Points per high-symmetry segment")
    parser.add_argument("--unit", type=str, default="thz", choices=["thz", "cm-1", "internal"])
    parser.add_argument("--ymax", type=float, default=None, help="Optional upper y-limit in selected unit")
    parser.add_argument("--negative-tolerance", type=float, default=1e-8)
    parser.add_argument("--enforce-asr", action="store_true", help="Enforce translational ASR on IFC self term")
    parser.add_argument("--save", type=Path, default=Path("outputs/graphene_dispersion_gkmg_thz.png"))
    args = parser.parse_args()

    ifc = read_ifc(args.ifc, reader=args.reader)
    asr_residual_max = None
    if args.enforce_asr:
        ifc, asr_residual_max = enforce_translational_asr_on_self_term(ifc)

    # Reduced coordinates in reciprocal-basis units.
    gamma = (0.0, 0.0, 0.0)
    kpt = (1.0 / 3.0, 1.0 / 3.0, 0.0)
    mpt = (0.5, 0.0, 0.0)
    nodes = [gamma, kpt, mpt, gamma]
    labels = [r"$\Gamma$", "K", "M", r"$\Gamma$"]

    path_red = np.vstack(
        [
            _segment(nodes[0], nodes[1], args.nseg),
            _segment(nodes[1], nodes[2], args.nseg),
            _segment(nodes[2], nodes[3], args.nseg),
            np.asarray([nodes[3]], dtype=float),
        ]
    )
    path = (2.0 * np.pi) * path_red

    # Geometric path coordinate for plotting.
    x = np.zeros(path.shape[0], dtype=float)
    if path.shape[0] > 1:
        x[1:] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))

    tick_idx = [0, args.nseg, 2 * args.nseg, 3 * args.nseg]
    ticks = [x[i] for i in tick_idx]

    bands = []
    min_omega2 = np.inf
    for kx, ky, kz in path:
        vals = np.linalg.eigvalsh(_direct_dynamical(ifc, float(kx), float(ky), float(kz)))
        min_omega2 = min(min_omega2, float(np.min(vals)))
        vals = np.where((vals < 0.0) & (vals >= -args.negative_tolerance), 0.0, vals)
        bands.append(np.sqrt(np.clip(vals.real, 0.0, None)))
    bands = np.asarray(bands, dtype=float)
    y, ylabel = _convert_omega(bands, unit=args.unit)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for b in range(y.shape[1]):
        ax.plot(x, y[:, b], color="tab:blue", lw=1.0, alpha=0.70)
    for t in ticks:
        ax.axvline(t, color="k", lw=0.7, alpha=0.35)
    ax.set_xticks(ticks, labels)
    ax.set_xlabel("k-path distance")
    ax.set_ylabel(ylabel)
    ax.set_title("Graphene Dispersion (Direct IFC): Gamma-K-M-Gamma")
    if args.ymax is not None:
        ax.set_ylim(0.0, float(args.ymax))
    ax.grid(alpha=0.2)
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=240)

    print(f"Saved: {args.save}")
    print(f"Modes: {y.shape[1]}")
    print(f"Minimum omega^2 on path: {min_omega2:.6e}")
    if asr_residual_max is not None:
        print(f"ASR enforced: yes (max pre-correction residual={asr_residual_max:.6e})")
    else:
        print("ASR enforced: no")
    gamma_sorted = np.sort(y[0])
    print(f"Gamma frequencies ({args.unit}, sorted): {np.round(gamma_sorted, 6)}")


if __name__ == "__main__":
    main()
