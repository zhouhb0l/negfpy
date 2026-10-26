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
from negfpy.anharmonic.transverse_solver import TransverseSpectrum
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
    assert default.spectrum.max_bytes == 64*1024**2
    assert admitted.spectrum.max_bytes == 256*1024**2
    for name in ("retarded", "lesser", "greater"):
        np.testing.assert_array_equal(getattr(default.spectrum, name), getattr(admitted.spectrum, name))
    np.testing.assert_array_equal(default.zero_limits, admitted.zero_limits)


def test_spectrum_really_copies_refined_mesh_payload_above_default_budget():
    # Actual refined q/frequency/contact-row dimensions, not a mocked
    # preallocation exit. Broadcast inputs avoid a second large input payload;
    # construction must allocate and detach all three finite complex128 fields.
    shape = (8, 601, 24, 24)
    source = np.broadcast_to(np.zeros((), dtype=np.complex128), shape)
    payload = 3*source.size*source.itemsize
    assert payload > 64*1024**2
    with pytest.raises(ValueError, match="max_bytes"):
        TransverseSpectrum(source, source, source)
    with pytest.raises(ValueError, match="max_bytes"):
        TransverseSpectrum(source, source, source, max_bytes=payload-1)
    spectrum = TransverseSpectrum(source, source, source, max_bytes=payload)
    assert sum(getattr(spectrum, name).nbytes for name in
               ("retarded", "lesser", "greater")) == payload
    for name in ("retarded", "lesser", "greater"):
        values = getattr(spectrum, name)
        assert values.shape == shape
        assert values.dtype == np.complex128
        assert not values.flags.writeable
        assert not np.shares_memory(values, source)
        assert not np.any(values)


@pytest.mark.parametrize("max_bytes, error", [
    (True, TypeError), (1.5, TypeError), (0, ValueError),
    (4*1024**3+1, ValueError),
])
def test_spectrum_rejects_invalid_explicit_budget(max_bytes, error):
    values = np.zeros((1, 1, 1, 1), dtype=np.complex128)
    with pytest.raises(error, match="max_bytes"):
        TransverseSpectrum(values, values, values, max_bytes=max_bytes)
