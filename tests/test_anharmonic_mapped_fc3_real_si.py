from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from negfpy.anharmonic.fc3 import validate_fc3
from negfpy.anharmonic.io.phono3py import read_phono3py_fc3
from negfpy.anharmonic.io.phono3py_crystal import (
    read_phono3py_crystal_mapping,
)
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.mapped_fc3 import (
    map_fc3_to_device,
    mass_normalize_translational_fc3,
)
from negfpy.anharmonic.translational_fc3 import (
    convert_fc3_to_translational,
)


_FULL_FC3_ENV = "NEGFPY_PHASE_A_FULL_FC3"
_SOURCE_FC3_CONTENT_SHA256 = (
    "6368368a4db9dcd324321fbc4d2d65519c2167e702641a098509afaf3de6c32e"
)
_PRIMITIVE_CONTENT_SHA256 = (
    "14efe6eff8478b04873f8c9a38c01adbfc83d0a6a10d6ffee59d51d073587ec9"
)
_FC3_MAPPING_CONTENT_SHA256 = (
    "49079c8d4b1f8de394088160fdbdf5adec9cf563694fc489a6815878c3e038d5"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_a_fixture() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "validation/materials/si-pbesol-phono3py-4.4.0/phase-a-fixture"
    )


def _authenticated_full_fc3(fixture: Path) -> Path:
    supplied = os.environ.get(_FULL_FC3_ENV)
    if supplied is None:
        pytest.skip(
            f"set {_FULL_FC3_ENV} to the sealed Si fc3.full.hdf5 artifact"
        )
    path = Path(supplied).expanduser().resolve()
    if not path.is_file():
        pytest.fail(f"{_FULL_FC3_ENV} is not a file: {path}")
    expected = {
        record.split(maxsplit=1)[1]: record.split(maxsplit=1)[0]
        for record in (
            fixture / "force-constants-SHA256SUMS"
        ).read_text(encoding="ascii").splitlines()
    }["fc3.full.hdf5"]
    assert _sha256_file(path) == expected
    return path


def test_real_si_r0_full_fc3_maps_on_the_minimum_nine_layer_crop() -> None:
    """Exercise the range-complete nine-layer real-Si Phase-B bridge."""

    pytest.importorskip("phono3py")
    fixture = _phase_a_fixture()
    full_fc3_path = _authenticated_full_fc3(fixture)
    yaml_path = fixture / "phono3py_disp.yaml"

    fc3_mapping = read_phono3py_crystal_mapping(yaml_path, purpose="fc3")
    raw_fc3 = read_phono3py_fc3(
        full_fc3_path,
        fc3_mapping.supercell.masses,
        p2s_map=fc3_mapping.p2s,
        s2p_map=fc3_mapping.s2p,
        compact=False,
    )
    source_certification = validate_fc3(raw_fc3)
    assert source_certification.passed
    assert source_certification.permutation_complete
    assert (
        source_certification.source_fc3_content_sha256
        == _SOURCE_FC3_CONTENT_SHA256
    )

    translated = convert_fc3_to_translational(
        raw_fc3,
        fc3_mapping,
        make_r0_average=True,
    )
    assert translated.make_r0_average is True
    assert translated.provenance["source_fc3_content_sha256"] == (
        _SOURCE_FC3_CONTENT_SHA256
    )
    assert translated.provenance["primitive_crystal_content_sha256"] == (
        _PRIMITIVE_CONTENT_SHA256
    )
    assert translated.provenance[
        "fc3_crystal_mapping_content_sha256"
    ] == _FC3_MAPPING_CONTENT_SHA256
    normalized = mass_normalize_translational_fc3(
        translated,
        fc3_mapping,
        source_certification=source_certification,
    )

    fc2_mapping = read_phono3py_crystal_mapping(yaml_path, purpose="fc2")
    fc2 = read_phono3py_fc2(
        fixture / "fc2.full.hdf5",
        fc2_mapping.supercell.masses,
        p2s_map=fc2_mapping.p2s,
        s2p_map=fc2_mapping.s2p,
        compact=False,
    )
    device_mapping = build_layered_device_mapping(
        fc2_mapping,
        fc2,
        transport_supercell_matrix=np.array(
            [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
            dtype=np.int64,
        ),
    )
    assert normalized.primitive_crystal_content_sha256 == (
        device_mapping.primitive_crystal_content_sha256
    )

    mapped = map_fc3_to_device(
        normalized,
        device_mapping,
        n_device_layers=9,
        interaction_layers=(2, 3, 4, 5, 6),
        observation_layers=(4,),
    )
    assert mapped.ndof == 9 * device_mapping.n_principal_layer_dofs == 216
    assert mapped.nnz > 0
    assert mapped.report.final_nonzero_count == mapped.nnz
    assert mapped.report.required_left_halo_layers == 2
    assert mapped.report.required_right_halo_layers == 2
    np.testing.assert_array_equal(
        mapped.report.pairwise_max_principal_layers,
        np.full(3, 2, dtype=np.int64),
    )
    assert mapped.report.contact_layers == (0, 8)
    assert mapped.report.harmonic_buffer_layers == ((0, 1), (7, 8))
    assert mapped.report.permutation_tolerance_passed
    assert mapped.report.permutation_missing_count == 0
    assert mapped.report.permutation_max_forward_error_ratio <= 1.0
    # Bitwise equality is diagnostic only: the r0 reference-leg paths have
    # different weighted decompositions whose arithmetic and authenticated
    # raw-translation uncertainties are bounded without modifying values.
    assert isinstance(mapped.report.exact_permutation_equality, bool)
    assert mapped.report.certification_passed
    assert mapped.report.failure_codes == ()
    preflight = mapped.metadata["compact_accumulation_preflight"]
    assert preflight["geometry_pass_count"] == 2
    assert preflight["retained_periodic_record_count"] == 0
    assert preflight["retained_python_contribution_record_count"] == 0
    assert preflight["estimated_peak_bytes"] < 2 * 1024**3
    assert mapped.report.source_fc3_content_sha256 == (
        _SOURCE_FC3_CONTENT_SHA256
    )
    assert mapped.report.fc3_crystal_mapping_content_sha256 == (
        _FC3_MAPPING_CONTENT_SHA256
    )
