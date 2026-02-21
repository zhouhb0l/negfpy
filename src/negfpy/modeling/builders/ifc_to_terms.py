"""Convert IFC schema terms into MaterialKspace term dictionaries."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from negfpy.modeling.schema import BuildConfig, IFCData
from negfpy.modeling.validators import validate_ifc_data, validate_transport_connectivity


Array = np.ndarray
Shift2D = tuple[int, int]


def infer_principal_layer_size(ifc: IFCData, config: BuildConfig | None = None) -> int:
    config = config or BuildConfig()
    if config.principal_layer_size is not None:
        if config.principal_layer_size <= 0:
            raise ValueError("principal_layer_size must be positive when provided.")
        return int(config.principal_layer_size)
    if not config.auto_principal_layer_enlargement:
        return 1
    max_abs_dx = max(abs(int(term.dx)) for term in ifc.terms)
    return max(1, max_abs_dx)


def _add_subblock(target: Array, row_idx: int, col_idx: int, block: Array, block_size: int) -> None:
    rs = row_idx * block_size
    cs = col_idx * block_size
    target[rs : rs + block_size, cs : cs + block_size] += block


def _ifc_has_full_realspace_grid(ifc: IFCData) -> bool:
    nr = ifc.metadata.get("nr") if isinstance(ifc.metadata, dict) else None
    if nr is None or len(nr) != 3:
        return False
    try:
        nr1, nr2, nr3 = int(nr[0]), int(nr[1]), int(nr[2])
    except Exception:
        return False
    if nr1 <= 0 or nr2 <= 0 or nr3 <= 0:
        return False
    unique_shifts = {(int(t.dx), int(t.dy), int(t.dz)) for t in ifc.terms}
    return len(unique_shifts) == nr1 * nr2 * nr3


def build_fc_terms(
    ifc: IFCData,
    config: BuildConfig | None = None,
) -> tuple[dict[Shift2D, Array], dict[Shift2D, Array], dict[Shift2D, Array] | None]:
    config = config or BuildConfig()
    validate_ifc_data(ifc)
    validate_transport_connectivity(ifc)

    fc00_acc: defaultdict[Shift2D, Array] = defaultdict(lambda: np.zeros((0, 0), dtype=config.dtype))
    fc01_acc: defaultdict[Shift2D, Array] = defaultdict(lambda: np.zeros((0, 0), dtype=config.dtype))
    fc10_acc: defaultdict[Shift2D, Array] = defaultdict(lambda: np.zeros((0, 0), dtype=config.dtype))
    block_size = np.asarray(ifc.masses).size * ifc.dof_per_atom
    pl_size = infer_principal_layer_size(ifc, config=config)
    ndof = pl_size * block_size
    present_shifts = {(int(term.dx), int(term.dy), int(term.dz)) for term in ifc.terms}
    is_full_grid = _ifc_has_full_realspace_grid(ifc)
    enable_negative_dx_inference = config.infer_fc01_from_negative_dx and not is_full_grid

    def _ensure_shape(target: defaultdict[Shift2D, Array], key: Shift2D) -> Array:
        if target[key].shape != (ndof, ndof):
            target[key] = np.zeros((ndof, ndof), dtype=config.dtype)
        return target[key]

    for term in ifc.terms:
        block = np.asarray(term.block, dtype=config.dtype)
        for row_cell in range(pl_size):
            col_abs = row_cell + int(term.dx)
            if 0 <= col_abs < pl_size:
                key = (term.dy, term.dz)
                _ensure_shape(fc00_acc, key)
                _add_subblock(fc00_acc[key], row_cell, col_abs, block, block_size)
                continue
            if pl_size <= col_abs < 2 * pl_size:
                key = (term.dy, term.dz)
                _ensure_shape(fc01_acc, key)
                _add_subblock(fc01_acc[key], row_cell, col_abs - pl_size, block, block_size)
                continue
            if -pl_size <= col_abs < 0:
                key10 = (term.dy, term.dz)
                _ensure_shape(fc10_acc, key10)
                _add_subblock(fc10_acc[key10], row_cell, col_abs + pl_size, block, block_size)
                if enable_negative_dx_inference:
                    # Backward-compatible optional inference for inputs that
                    # do not provide explicit +dx terms.
                    if (-int(term.dx), -int(term.dy), -int(term.dz)) not in present_shifts:
                        key01 = (-term.dy, -term.dz)
                        _ensure_shape(fc01_acc, key01)
                        _add_subblock(fc01_acc[key01], col_abs + pl_size, row_cell, block.conj().T, block_size)
                continue

    if len(fc00_acc) == 0:
        raise ValueError("No dx=0 IFC terms were mapped to fc00_terms.")
    if len(fc01_acc) == 0:
        raise ValueError("No x-coupling IFC terms were mapped to fc01_terms.")

    fc00 = {k: np.asarray(v, dtype=config.dtype) for k, v in fc00_acc.items()}
    fc01 = {k: np.asarray(v, dtype=config.dtype) for k, v in fc01_acc.items()}
    fc10 = {k: np.asarray(v, dtype=config.dtype) for k, v in fc10_acc.items()}

    # For full real-space grids (e.g., QE q2r full nr1*nr2*nr3 terms), forcing
    # Hermitian term-by-term on fc00 alters exact supercell/primitive
    # equivalence. Keep raw assembled fc00 in that case.
    if config.enforce_hermitian_fc00 and not _ifc_has_full_realspace_grid(ifc):
        fc00 = {k: 0.5 * (b + b.conj().T) for k, b in fc00.items()}

    # For non-full-grid inputs that do not provide explicit n=-1 couplings,
    # keep backward compatibility by leaving fc10 unspecified.
    if not is_full_grid and len(fc10) == 0:
        fc10_out: dict[Shift2D, Array] | None = None
    else:
        fc10_out = fc10
    return fc00, fc01, fc10_out
