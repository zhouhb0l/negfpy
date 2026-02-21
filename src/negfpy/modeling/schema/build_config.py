"""Builder configuration for IFC-to-model conversion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildConfig:
    """Config knobs for model assembly from IFC data."""

    onsite_pinning: float = 0.0
    principal_layer_size: int | None = None
    auto_principal_layer_enlargement: bool = True
    infer_fc01_from_negative_dx: bool = True
    enforce_hermitian_fc00: bool = True
    dtype: str = "complex128"
