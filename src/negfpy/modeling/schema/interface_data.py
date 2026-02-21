"""Schema for interface/contact coupling data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class InterfaceData:
    """Coupling matrices and mapping metadata for interfaces."""

    left_contact: Array | None = None
    right_contact: Array | None = None
    index_map_left: tuple[int, ...] | None = None
    index_map_right: tuple[int, ...] | None = None
    convention: str = "direct"
    metadata: dict[str, Any] = field(default_factory=dict)

