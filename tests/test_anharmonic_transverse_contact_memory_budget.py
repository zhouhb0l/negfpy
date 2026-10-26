"""Explicit refined-mesh allocation budgets retain default contact physics."""

import numpy as np
import pytest

from negfpy.anharmonic import transverse_contacts as module
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_compact_contacts import (
    build_compact_transverse_contact,
)
from negfpy.anharmonic.transverse_contacts import build_transverse_contact
from negfpy.anharmonic.types import HarmonicReservoir
from negfpy.core.types import LeadBlocks


@pytest.mark.parametrize("builder", [build_transverse_contact, build_compact_transverse_contact])
def test_refined_payload_requires_explicit_budget_before_spectral_work(builder, monkeypatch):
    # 8 q x 601 frequencies x 24 contact rows exceeds 64 MiB. Stop at the first
    # spectral operation: this tests real dimension admission without allocating
    # the large contact fields or solving a material problem.
    mesh, grid = UniformBZMesh((4, 2, 1)), SignedFrequencyGrid(.025, 300)
    source = HarmonicReservoir(LeadBlocks(3*np.eye(24), -np.eye(24)), .7)
    with pytest.raises(ValueError, match="spectra"):
        builder(mesh, grid, lambda q: source, 24, .01, side="left")

    def admitted(*args, **kwargs):
        raise RuntimeError("spectral-work-admitted")

    monkeypatch.setattr(module, "_classify_lead_spectrum", admitted)
    with pytest.raises(RuntimeError, match="spectral-work-admitted"):
        builder(mesh, grid, lambda q: source, 24, .01, side="left", max_bytes=256*1024**2)
    with pytest.raises(ValueError, match="4 GiB"):
        builder(mesh, grid, lambda q: pytest.fail("oversized request reached callback"),
                24, .01, side="left", max_bytes=4*1024**3+1)


@pytest.mark.parametrize("builder", [build_transverse_contact, build_compact_transverse_contact])
def test_explicit_larger_budget_leaves_the_same_contact_fields(builder):
    mesh, grid = UniformBZMesh((1, 1, 1)), SignedFrequencyGrid(.2, 2)
    source = HarmonicReservoir(LeadBlocks(np.array([[2.6]]), np.array([[-1.3]])), .7)
    default = builder(mesh, grid, lambda q: source, 1, .02, side="left")
    admitted = builder(mesh, grid, lambda q: source, 1, .02, side="left", max_bytes=256*1024**2)
    for name in ("retarded", "lesser", "greater"):
        np.testing.assert_array_equal(getattr(default.spectrum, name), getattr(admitted.spectrum, name))
    np.testing.assert_array_equal(default.zero_limits, admitted.zero_limits)
