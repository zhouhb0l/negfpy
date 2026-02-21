"""Build MaterialKspaceParams and transport objects from IFC schema."""

from __future__ import annotations

from typing import Any

import numpy as np

from negfpy.modeling.builders.ifc_to_terms import build_fc_terms, infer_principal_layer_size
from negfpy.modeling.schema import BuildConfig, IFCData
from negfpy.modeling.validators import validate_ifc_data, validate_material_kspace_params
from negfpy.models import MaterialKspaceParams, material_kspace_device, material_kspace_lead


def build_material_kspace_params(ifc: IFCData, config: BuildConfig | None = None) -> MaterialKspaceParams:
    config = config or BuildConfig()
    validate_ifc_data(ifc)
    pl_size = infer_principal_layer_size(ifc=ifc, config=config)
    fc00_terms, fc01_terms, fc10_terms = build_fc_terms(ifc, config=config)
    params = MaterialKspaceParams(
        masses=np.tile(np.asarray(ifc.masses, dtype=float), pl_size),
        fc00_terms=fc00_terms,
        fc01_terms=fc01_terms,
        fc10_terms=fc10_terms,
        dof_per_atom=ifc.dof_per_atom,
        onsite_pinning=float(config.onsite_pinning),
    )
    validate_material_kspace_params(params)
    return params


def build_transport_components(
    ifc: IFCData,
    n_layers: int,
    config: BuildConfig | None = None,
    layer_masses: np.ndarray | None = None,
) -> dict[str, Any]:
    params = build_material_kspace_params(ifc=ifc, config=config)
    return {
        "params": params,
        "lead": material_kspace_lead(params),
        "device": material_kspace_device(n_layers=n_layers, params=params, layer_masses=layer_masses),
    }
