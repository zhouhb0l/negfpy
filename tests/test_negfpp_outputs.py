"""Field exports retain physical coordinates and distinct SCBA state roles."""

from types import SimpleNamespace

import h5py
import numpy as np

from negfpp.workflow import save_fields
from negfpy.anharmonic.complete_energy import (
    ChannelSpectrum,
    SignedFrequencyGrid,
    SolverScale,
    UniformBZMesh,
)


def spectrum(value):
    shape = (2, 3, 2, 2)
    return ChannelSpectrum(
        np.full(shape, value + 1j, dtype=np.complex128),
        np.full(shape, value + 2j, dtype=np.complex128),
        np.full(shape, value + 3j, dtype=np.complex128),
    )


def fixture_state(sigma):
    grid = SignedFrequencyGrid(0.15, 1)
    mesh = UniformBZMesh((2, 1, 1))
    scale = SolverScale(omega0=4.0, hbar=2.0, metadata={"source_units": "SI"})
    vertex = SimpleNamespace(active_dofs=np.array([2, 5]), full_ndof=6)
    state = SimpleNamespace(
        index=7,
        sigma_input=sigma,
        evaluation=SimpleNamespace(green=spectrum(20), map_sigma=spectrum(30)),
    )
    return state, grid, mesh, vertex, scale


def test_export_physical_coordinates_and_evaluated_state_roles(tmp_path):
    state, grid, mesh, vertex, scale = fixture_state(spectrum(10))
    destination = tmp_path / "fields.h5"
    save_fields(destination, state, grid, mesh, vertex, scale=scale)

    with h5py.File(destination, "r") as result:
        np.testing.assert_allclose(result["omega_solver"][:], [-0.15, 0.0, 0.15])
        physical = result["omega_rad_s"][:]
        np.testing.assert_allclose(physical, [-0.6, 0.0, 0.6])
        np.testing.assert_array_equal(physical[grid.negative_partners], -physical)
        np.testing.assert_allclose(result["q_reduced"][:], [[0.0, 0.0], [-0.5, 0.0]])
        np.testing.assert_allclose(result["q_phase_rad"][:], [[0.0, 0.0], [-np.pi, 0.0]])
        assert result["omega_rad_s"].attrs["units"] == "rad/s"
        assert result["q_reduced"].attrs["units"] == "reciprocal lattice fractions"
        assert result["q_phase_rad"].attrs["units"] == "rad"
        assert result.attrs["omega0_rad_s"] == scale.omega0
        assert result.attrs["hbar_j_s"] == scale.hbar
        np.testing.assert_array_equal(result["active_device_indices"][:], [2, 5])
        assert result.attrs["evaluated_state_index"] == 7
        assert not result.attrs["mapped_image_dyson_evaluated"]

        roles = {
            "green_evaluated": state.evaluation.green,
            "sigma_input_evaluated": state.sigma_input,
            "sigma_born_image_unevaluated": state.evaluation.map_sigma,
        }
        for name, source in roles.items():
            assert not result[name].attrs["exact_zero"]
            for component in ("retarded", "statistical", "spectral"):
                np.testing.assert_array_equal(result[name][component][:], getattr(source, component))


def test_exact_zero_input_is_distinct_from_nonzero_born_image(tmp_path):
    state, grid, mesh, vertex, scale = fixture_state(None)
    destination = tmp_path / "fields.h5"
    save_fields(destination, state, grid, mesh, vertex, scale=scale)

    with h5py.File(destination, "r") as result:
        zero = result["sigma_input_evaluated"]
        image = result["sigma_born_image_unevaluated"]
        assert zero.attrs["exact_zero"]
        assert list(zero) == []
        assert not image.attrs["exact_zero"]
        np.testing.assert_array_equal(image["retarded"][:], state.evaluation.map_sigma.retarded)
        assert np.any(image["retarded"][:] != 0)
        assert not result.attrs["mapped_image_dyson_evaluated"]
