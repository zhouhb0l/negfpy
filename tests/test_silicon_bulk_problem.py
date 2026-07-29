from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import _silicon_bulk_problem as loader
from test_anharmonic_bulk_target_q_mpi import _problem


def test_loader_constructs_one_exact_validated_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    template = _problem()
    mappings = {
        "fc2": template.fc2_mapping,
        "fc3": template.fc3_mapping,
    }
    monkeypatch.setattr(
        loader,
        "read_phono3py_crystal_mapping",
        lambda path, *, purpose, is_nac: mappings[purpose],
    )
    monkeypatch.setattr(loader, "read_phono3py_fc2", lambda *args, **kwargs: template.fc2)
    raw_fc3 = object()
    monkeypatch.setattr(
        loader, "read_phono3py_fc3", lambda *args, **kwargs: raw_fc3
    )
    monkeypatch.setattr(
        loader,
        "convert_fc3_to_translational",
        lambda values, mapping, **kwargs: template.fc3,
    )
    monkeypatch.setattr(
        loader,
        "validate_translational_fc3",
        lambda *args, **kwargs: SimpleNamespace(passed=True, errors=()),
    )

    result = loader.load_silicon_bulk_three_phonon_problem(
        fc2_compact_path="fc2.hdf5",
        fc3_compact_path="fc3.hdf5",
        phono3py_yaml_path="phono3py.yaml",
        mesh=(3, 1, 1),
        sigma_thz=0.05,
        cutoff_frequency_thz=1.0e-8,
        frequency_factor_to_thz=1.0,
        degenerate_tolerance_thz=1.0e-8,
        make_r0_average=True,
        symprec=1.0e-5,
        translation_atol=1.0e-12,
        translation_rtol=1.0e-10,
        asr_atol=1.0e-10,
        asr_rtol=1.0e-8,
        metadata={"fixture": "shared-loader"},
    )

    assert result.fc2 is template.fc2
    assert result.fc3 is template.fc3
    assert tuple(result.mesh.mesh) == (3, 1, 1)
    assert np.isclose(result.sigma_thz, 0.05)
    assert result.metadata["fixture"] == "shared-loader"


def test_loader_rejects_invalid_controls_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        loader,
        "read_phono3py_crystal_mapping",
        lambda *args, **kwargs: pytest.fail("reader was called"),
    )

    with pytest.raises(ValueError, match="mesh"):
        loader.load_silicon_bulk_three_phonon_problem(
            fc2_compact_path="fc2.hdf5",
            fc3_compact_path="fc3.hdf5",
            phono3py_yaml_path="phono3py.yaml",
            mesh=(3, 0, 1),
            sigma_thz=0.05,
            cutoff_frequency_thz=0.0,
            frequency_factor_to_thz=1.0,
            degenerate_tolerance_thz=1.0e-8,
            make_r0_average=True,
            symprec=1.0e-5,
            translation_atol=1.0e-12,
            translation_rtol=1.0e-10,
            asr_atol=1.0e-10,
            asr_rtol=1.0e-8,
            metadata={},
        )
