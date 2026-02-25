"""Plot silicon phonon dispersion along a user-defined reduced-coordinate path from direct IFC."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import qe_omega_to_cm1


def _read_ifc(path: Path, reader: str):
    if reader == "qe":
        return read_qe_q2r_ifc(path)
    if reader == "phonopy":
        return read_phonopy_ifc(path)
    return read_ifc(path, reader=reader)


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
    s = np.linspace(0.0, 1.0, n, endpoint=False)
    return ta[None, :] * (1.0 - s[:, None]) + tb[None, :] * s[:, None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("studies/silicon_bulk_2026q1/inputs/ifc/si444.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--save", type=Path, default=Path("outputs/silicon_dispersion_gxgl.png"))
    args = parser.parse_args()

    ifc = _read_ifc(args.ifc, reader=args.reader)

    # Exact path provided by user in reduced coordinates (multiples of 2*pi).
    k_red = np.asarray(
        [
            (0.0, 0.0, 0.0),  # Gamma
            (0.1, 0.1, 0.0),
            (0.2, 0.2, 0.0),
            (0.3, 0.3, 0.0),
            (0.4, 0.4, 0.0),
            (0.5, 0.5, 0.0),
            (0.6, 0.6, 0.0),
            (0.7, 0.7, 0.0),  # K
            (0.8, 0.8, 0.0),
            (0.9, 0.9, 0.0),
            (1.0, 1.0, 0.0),  # X
            (1.0, 1.0, 0.1),
            (1.0, 1.0, 0.2),
            (1.0, 1.0, 0.3),
            (1.0, 1.0, 0.4),
            (1.0, 1.0, 0.5),
            (1.0, 1.0, 0.6),
            (1.0, 1.0, 0.7),
            (1.0, 1.0, 0.8),
            (1.0, 1.0, 0.9),
            (1.0, 1.0, 1.0),  # Gamma (equivalent)
            (0.9, 0.9, 0.9),
            (0.8, 0.8, 0.8),
            (0.7, 0.7, 0.7),
            (0.6, 0.6, 0.6),
            (0.5, 0.5, 0.5),  # L
        ],
        dtype=float,
    )
    kpath = (2.0 * np.pi) * k_red

    bands = []
    for kx, ky, kz in kpath:
        w2 = np.linalg.eigvalsh(_direct_dynamical(ifc, float(kx), float(ky), float(kz)))
        bands.append(np.sqrt(np.clip(w2.real, 0.0, None)))
    bands = np.asarray(bands, dtype=float)
    y = np.asarray(qe_omega_to_cm1(bands), dtype=float)

    # Use cumulative |dk| so segment lengths are proportional to geometric distance in k-space.
    x = np.zeros(kpath.shape[0], dtype=float)
    if kpath.shape[0] > 1:
        dk = np.linalg.norm(np.diff(kpath, axis=0), axis=1)
        x[1:] = np.cumsum(dk)
    tick0 = x[0]
    tick1 = x[7]
    tick2 = x[10]
    tick3 = x[20]
    tick4 = x[25]

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    for b in range(y.shape[1]):
        ax.plot(x, y[:, b], color="tab:blue", lw=1.0, alpha=0.65)
    for t in [tick0, tick1, tick2, tick3, tick4]:
        ax.axvline(t, color="k", lw=0.7, alpha=0.35)
    ax.set_xticks([tick0, tick1, tick2, tick3, tick4], [r"$\Gamma$", "K", "X", r"$\Gamma$", "L"])
    ax.set_ylabel("Frequency (cm^-1)")
    ax.set_xlabel("k-path distance")
    ax.set_title("Silicon Dispersion (Direct IFC): Gamma-K-X-Gamma-L")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=240)
    print(f"Saved: {args.save}")
    x_idx = 10
    x_sorted = np.sort(y[x_idx])
    print(f"X-point frequencies (cm^-1): {np.round(x_sorted, 6)}")


if __name__ == "__main__":
    main()
