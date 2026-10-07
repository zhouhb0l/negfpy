"""Contact physics, bounded iteration and steady cubic-current regressions."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

from negfpy.anharmonic import SignedFrequencyGrid


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
reference = importlib.import_module("validate_cubic_steady_current")


def test_drude_bath_keldysh_fdt_zero_limit_and_static_stiffness():
    grid = SignedFrequencyGrid(step=0.1, n_positive=50)
    temperature = 0.7
    sigma = reference.drude_contact(grid, 0, temperature)
    np.testing.assert_allclose(sigma.greater - sigma.lesser,
                               sigma.retarded - sigma.retarded.conj().transpose(0, 2, 1),
                               atol=1e-15)
    positive = grid.positive_indices
    np.testing.assert_allclose(sigma.lesser[positive, 0, 0],
                               sigma.greater[positive, 0, 0]
                               * np.exp(-grid.values[positive] / temperature), atol=1e-15)
    assert np.all(sigma.retarded[positive, 0, 0].imag < 0)
    assert sigma.retarded[grid.zero_index, 0, 0] == 0
    assert sigma.lesser[grid.zero_index, 0, 0] == pytest.approx(-0.5j * temperature)
    np.testing.assert_allclose(sigma.retarded[::-1, 0, 0], sigma.retarded[:, 0, 0].conj())


def test_map_budget_failure_cannot_produce_an_accepted_current_state():
    with pytest.raises(RuntimeError, match="budget exhausted"):
        reference.solve_reference(max_maps=1)


def test_biased_dressed_current_requires_cubic_operator():
    state = reference.solve_reference()
    current = reference.measure_currents(state)
    left, right = state["metadata"]["contact_currents"]
    assert left > 0.01 and abs(left + right) < 1e-10
    for cut in current["cuts"]:
        assert abs(cut["contact_difference"]) / left < 1e-3
        assert abs(cut["harmonic_only_contact_difference"]) / left > 1e-3
        assert abs(cut["cubic"]) > 1e-4
    assert state["metadata"]["raw_map_residual_history"][-1] < 1e-12
    assert state["metadata"]["means_prescribed_not_solved"]
    assert not state["metadata"]["material_spatial_validation_passed"]
