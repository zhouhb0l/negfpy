"""Fail-closed assembly of a finite material :math:`N_q=1` problem.

This module is intentionally private.  The IFC readers, layered FC2 mapper,
and finite FC3 mapper remain separate scientific stages; this file only joins
their immutable outputs into the standard :class:`CubicSCBAProblem` consumed
by the solver.  In particular, it does not infer a device size or silently
crop a cubic interaction.  The complete-range and padding decision made by
``map_fc3_to_device`` must reproduce and pass here.

The lead convention is increasing layer number.  Consequently the finite
device couples to the left lead with ``d10`` and to the right lead with
``d01``.  Both couplings are stored explicitly so that no later default can
obscure that orientation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import numpy as np

from negfpy.core.types import Device1D

from ._canonical import digest_bytes, domain_digest
from .grid import SignedFrequencyGrid
from .layered import (
    LayeredDeviceMapping,
    LayeredHarmonicModel,
    _model_payload,
)
from .mapped_fc3 import (
    DeviceMappedFC3,
    _cubic_physics_sha256,
    _report_payload,
    scale_device_fc3,
)
from .sparse_cubic import SparseCubicVertex
from .types import CubicSCBAProblem, HarmonicReservoir, SolverScale


Array = np.ndarray


_PHASE_B_ARCHIVE_SCHEMA = "negfpy.sparse-nq1-phase-b-mapped-fc3.v1"
_PHASE_B_VERTEX_SCHEMA = "negfpy.sparse-cubic-vertex.material.v1"
_PHASE_B_METADATA_ENCODING = "negfpy.lossless-metadata-json.v1"
_CERTIFIED_PHASE_B_ARCHIVE_SHA256 = (
    "06b3294ecc81ddad5399f4ebe0ae226eab406c1c06c4fef277c76420a0865101"
)
_PHASE_B_N_DEVICE_LAYERS = 9
_PHASE_B_N_PRINCIPAL_LAYER_DOFS = 24
_PHASE_B_N_DEVICE_DOFS = 216
_PHASE_B_CONTACT_LAYERS = (0, 8)
_PHASE_B_INTERACTION_LAYERS = (2, 3, 4, 5, 6)
_PHASE_B_OBSERVATION_LAYERS = (4,)
_PHASE_B_REQUIRED_HALO_LAYERS = (2, 2)
_PHASE_B_REQUIRED_CHECKS = frozenset(
    {
        "raw_file_sha256",
        "canonical_fc3_sha256",
        "raw_fc3_complete_validation",
        "r0_average_enabled",
        "nine_device_layers",
        "silicon_principal_layer_dofs",
        "silicon_total_device_dofs",
        "contact_layers",
        "interaction_layers",
        "observation_layers",
        "contact_layers_count_as_harmonic_buffers",
        "range_certification",
        "mapping_certification",
        "permutation_support_complete",
        "permutation_forward_error_bound",
        "no_source_mutation_or_truncation",
        "real_acoustic_lead",
        "sparse_vertex_scaled",
    }
)


class MaterialNQ1ProblemError(ValueError):
    """Raised when material stages cannot be joined without guessing."""


def _finite_scalar(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or np.asarray(value).ndim != 0
        or np.iscomplexobj(value)
    ):
        raise TypeError(f"{name} must be a scalar real number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _exact_array(left: Any, right: Any, *, name: str) -> None:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape or not np.array_equal(
        left_array, right_array
    ):
        raise MaterialNQ1ProblemError(f"{name} identity does not match.")


def _ordered_contiguous(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise MaterialNQ1ProblemError(
            f"{name} is not an integer sequence."
        ) from exc
    if not result or result != tuple(range(result[0], result[-1] + 1)):
        raise MaterialNQ1ProblemError(
            f"{name} must be nonempty, ordered, unique, and contiguous."
        )
    return result


def _mapping(value: Any, *, name: str) -> Any:
    if not isinstance(value, Mapping):
        raise MaterialNQ1ProblemError(f"{name} must be a mapping.")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MaterialNQ1ProblemError(
            f"{name} must be a lowercase SHA256 digest."
        )
    return value


def _array_sha256(values: Any) -> str:
    from ._canonical import canonical_encode

    return hashlib.sha256(canonical_encode(np.asarray(values))).hexdigest()


def _exact_metadata_value(
    metadata: Any,
    key: str,
    expected: Any,
    *,
    name: str,
) -> None:
    if key not in metadata or metadata[key] != expected:
        raise MaterialNQ1ProblemError(
            f"{name} {key!r} does not match the authenticated Phase-B value."
        )


def _validate_harmonic_model(model: LayeredHarmonicModel) -> int:
    if not isinstance(model, LayeredHarmonicModel):
        raise TypeError("harmonic_model must be a LayeredHarmonicModel.")
    if not isinstance(model.scale, SolverScale):
        raise MaterialNQ1ProblemError(
            "Layered harmonic model has no valid SolverScale."
        )
    if not isinstance(model.mapping, LayeredDeviceMapping):
        raise MaterialNQ1ProblemError(
            "Layered harmonic model has no valid device mapping."
        )
    if (
        model.validation is None
        or not bool(model.validation.certification_passed)
        or model.mapping.validation is None
        or not bool(model.mapping.validation.certification_passed)
    ):
        raise MaterialNQ1ProblemError(
            "Layered harmonic model and mapping must carry passing audits."
        )

    identities = (
        ("FC2 content", model.fc2_content_sha256, model.mapping.fc2_content_sha256),
        (
            "primitive crystal",
            model.primitive_crystal_content_sha256,
            model.mapping.primitive_crystal_content_sha256,
        ),
        (
            "FC2 crystal mapping",
            model.fc2_crystal_mapping_content_sha256,
            model.mapping.fc2_crystal_mapping_content_sha256,
        ),
        (
            "layered geometry",
            model.layered_geometry_sha256,
            model.mapping.layered_geometry_sha256,
        ),
    )
    for name, model_value, mapping_value in identities:
        if model_value != mapping_value:
            raise MaterialNQ1ProblemError(
                f"Layered harmonic {name} identity does not match its mapping."
            )

    d00 = np.asarray(model.d00)
    d01 = np.asarray(model.d01)
    d10 = np.asarray(model.d10)
    if d00.ndim != 2 or d00.shape[0] < 1 or d00.shape[0] != d00.shape[1]:
        raise MaterialNQ1ProblemError("Harmonic d00 must be a nonempty square block.")
    if d01.shape != d00.shape or d10.shape != d00.shape:
        raise MaterialNQ1ProblemError(
            "Harmonic d00, d01, and d10 must have identical shapes."
        )
    if not all(np.all(np.isfinite(block)) for block in (d00, d01, d10)):
        raise MaterialNQ1ProblemError("Harmonic blocks must be finite.")
    if not np.array_equal(d00, d00.conj().T):
        raise MaterialNQ1ProblemError("Harmonic d00 must be exactly Hermitian.")
    if not np.array_equal(d10, d01.conj().T):
        raise MaterialNQ1ProblemError(
            "Harmonic orientation requires exact d10=d01^dagger."
        )
    layer_dofs = int(model.mapping.n_principal_layer_dofs)
    if layer_dofs != d00.shape[0]:
        raise MaterialNQ1ProblemError(
            "Harmonic block size does not match the mapped principal layer."
        )
    _exact_array(model.lead.d00, d00, name="lead d00")
    _exact_array(model.lead.d01, d01, name="lead d01")
    lead_d10 = (
        model.lead.d10
        if model.lead.d10 is not None
        else np.asarray(model.lead.d01).conj().T
    )
    _exact_array(lead_d10, d10, name="lead d10")

    expected_model_hash = domain_digest(
        "NEGFPY-LAYERED-HARMONIC-MODEL-PHYSICS",
        _model_payload(
            model.mapping,
            model.scale,
            d00=d00,
            d01=d01,
            d10=d10,
            atom_masses_kg=np.asarray(model.principal_layer_atom_masses_kg),
            rigid_translation_vectors=np.asarray(model.rigid_translation_vectors),
        ),
    )
    if expected_model_hash != model.model_physics_sha256:
        raise MaterialNQ1ProblemError(
            "Layered harmonic model physics identity does not reproduce."
        )
    return layer_dofs


def _authenticate_phase_b_input(
    harmonic_model: LayeredHarmonicModel,
    sparse_vertex: SparseCubicVertex,
    phase_b_archive: Mapping[str, Any],
    *,
    layer_dofs: int,
) -> tuple[int, str, str, str]:
    """Authenticate a script-loaded Phase-B archive without importing HDF5.

    ``phase_b_archive`` is the verified mapping returned by the Phase-B
    script's ``verify_phase_b_hdf5`` function and ``sparse_vertex`` is the
    result of its lossless loader.  Keeping those file operations at the
    script boundary lets ordinary package imports remain independent of
    :mod:`h5py`; this function reproduces every identity needed to bind the
    loaded solver coefficients to one exact layered harmonic model.
    """

    if not isinstance(sparse_vertex, SparseCubicVertex):
        raise TypeError("sparse_vertex must be a SparseCubicVertex.")
    archive = _mapping(phase_b_archive, name="phase_b_archive")
    if archive.get("schema") != _PHASE_B_ARCHIVE_SCHEMA:
        raise MaterialNQ1ProblemError("Phase-B archive schema is unsupported.")
    archive_sha256 = _sha256(
        archive.get("sha256"), name="phase_b_archive.sha256"
    )
    if archive_sha256 != _CERTIFIED_PHASE_B_ARCHIVE_SHA256:
        raise MaterialNQ1ProblemError(
            "Phase-B archive is not the exact certified mapped HDF5."
        )
    if archive.get("phase_b_certified") is not True:
        raise MaterialNQ1ProblemError("Phase-B archive is not scientifically certified.")
    if archive.get("release_certified") is not False:
        raise MaterialNQ1ProblemError(
            "Phase-B mapping evidence must not claim release certification."
        )
    if archive.get("scba_executed") is not False:
        raise MaterialNQ1ProblemError(
            "Phase-B mapping evidence must not claim SCBA execution."
        )

    metadata = _mapping(archive.get("metadata"), name="phase_b_archive.metadata")
    if metadata.get("schema") != _PHASE_B_ARCHIVE_SCHEMA:
        raise MaterialNQ1ProblemError("Phase-B metadata schema is unsupported.")
    if metadata.get("phase_b_certified") is not True:
        raise MaterialNQ1ProblemError("Phase-B metadata is not certified.")
    if metadata.get("release_certified") is not False:
        raise MaterialNQ1ProblemError(
            "Phase-B metadata falsely claims release certification."
        )
    if metadata.get("scba_executed") is not False:
        raise MaterialNQ1ProblemError(
            "Phase-B metadata falsely claims SCBA execution."
        )
    if metadata.get("method") != "authenticated_r0_fc3_finite_mapping_preflight":
        raise MaterialNQ1ProblemError("Phase-B mapping method is unsupported.")
    if metadata.get("sparse_vertex_metadata_encoding") != _PHASE_B_METADATA_ENCODING:
        raise MaterialNQ1ProblemError(
            "Phase-B sparse metadata does not use the lossless encoding."
        )
    campaign_sha256 = _sha256(
        metadata.get("campaign_manifest_sha256"),
        name="phase_b campaign manifest",
    )

    checks = _mapping(metadata.get("checks"), name="phase_b checks")
    missing_checks = sorted(_PHASE_B_REQUIRED_CHECKS.difference(checks))
    if missing_checks:
        raise MaterialNQ1ProblemError(
            f"Phase-B archive lacks required checks: {missing_checks!r}."
        )
    failed_checks = sorted(name for name, passed in checks.items() if passed is not True)
    if failed_checks:
        raise MaterialNQ1ProblemError(
            f"Phase-B archive contains failed checks: {failed_checks!r}."
        )

    configuration = _mapping(
        metadata.get("configuration"), name="phase_b configuration"
    )
    dimensions = _mapping(metadata.get("dimensions"), name="phase_b dimensions")
    identities = _mapping(metadata.get("identities"), name="phase_b identities")
    array_hashes = _mapping(
        metadata.get("array_hashes"), name="phase_b array hashes"
    )
    reports = _mapping(metadata.get("reports"), name="phase_b reports")
    mapping_report = _mapping(
        reports.get("device_mapping"), name="phase_b device mapping report"
    )
    acoustic_report = _mapping(
        reports.get("acoustic_lead_classification"),
        name="phase_b acoustic lead report",
    )

    try:
        n_layers = int(dimensions["n_device_layers"])
        archived_layer_dofs = int(dimensions["n_principal_layer_dofs"])
        archived_ndof = int(dimensions["ndof"])
        archived_nnz = int(dimensions["nnz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MaterialNQ1ProblemError(
            "Phase-B dimensions are incomplete or invalid."
        ) from exc
    if (
        n_layers != _PHASE_B_N_DEVICE_LAYERS
        or archived_layer_dofs != _PHASE_B_N_PRINCIPAL_LAYER_DOFS
        or archived_ndof != _PHASE_B_N_DEVICE_DOFS
        or archived_layer_dofs != layer_dofs
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B principal-layer dimensions do not match the harmonic model."
        )
    if archived_ndof != n_layers * layer_dofs or archived_ndof != sparse_vertex.ndof:
        raise MaterialNQ1ProblemError(
            "Phase-B device dimension does not match its vertex and harmonic model."
        )
    if archived_nnz != sparse_vertex.nnz:
        raise MaterialNQ1ProblemError("Phase-B sparse support count does not match.")
    if archive.get("ndof") != archived_ndof or archive.get("nnz") != archived_nnz:
        raise MaterialNQ1ProblemError(
            "Phase-B outer archive dimensions do not match its metadata."
        )

    try:
        configured_layers = int(configuration["n_device_layers"])
        configured_omega0 = float(configuration["omega0_rad_s"])
        configured_hbar = float(configuration["hbar_j_s"])
        contact_layers = tuple(int(value) for value in configuration["contact_layers"])
        interaction_layers = _ordered_contiguous(
            configuration["interaction_layers"], name="interaction_layers"
        )
        observation_layers = _ordered_contiguous(
            configuration["observation_layers"], name="observation_layers"
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MaterialNQ1ProblemError(
            "Phase-B geometry or scaling configuration is invalid."
        ) from exc
    if (
        configured_layers != n_layers
        or contact_layers != _PHASE_B_CONTACT_LAYERS
        or interaction_layers != _PHASE_B_INTERACTION_LAYERS
        or observation_layers != _PHASE_B_OBSERVATION_LAYERS
    ):
        raise MaterialNQ1ProblemError("Phase-B contact geometry does not reproduce.")
    if configuration.get("make_r0_average") is not True:
        raise MaterialNQ1ProblemError("Phase-B archive did not retain r0 averaging.")
    if (
        configured_omega0 != harmonic_model.scale.omega0
        or configured_hbar != harmonic_model.scale.hbar
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B solver scale does not match the harmonic model."
        )
    if (
        not set(observation_layers).issubset(interaction_layers)
        or interaction_layers[0] <= 0
        or interaction_layers[-1] >= n_layers - 1
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B interaction/observation crop is not interior."
        )

    expected_buffers = (
        tuple(range(interaction_layers[0])),
        tuple(range(interaction_layers[-1] + 1, n_layers)),
    )
    try:
        report_contacts = tuple(int(value) for value in mapping_report["contact_layers"])
        report_interactions = tuple(
            int(value) for value in mapping_report["interaction_layers"]
        )
        report_observations = tuple(
            int(value) for value in mapping_report["observation_layers"]
        )
        report_buffers = tuple(
            tuple(int(value) for value in side)
            for side in mapping_report["harmonic_buffer_layers"]
        )
        left_halo = int(mapping_report["required_left_halo_layers"])
        right_halo = int(mapping_report["required_right_halo_layers"])
        forward_error_ratio = float(
            mapping_report["permutation_max_forward_error_ratio"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MaterialNQ1ProblemError(
            "Phase-B device mapping geometry is incomplete."
        ) from exc
    if (
        int(mapping_report.get("n_device_layers", -1)) != n_layers
        or report_contacts != contact_layers
        or report_interactions != interaction_layers
        or report_observations != observation_layers
        or report_buffers != expected_buffers
        or (left_halo, right_halo) != _PHASE_B_REQUIRED_HALO_LAYERS
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B device mapping report and configuration disagree."
        )
    if (
        left_halo < 0
        or right_halo < 0
        or len(expected_buffers[0]) < left_halo
        or len(expected_buffers[1]) < right_halo
        or observation_layers[0] - interaction_layers[0] < left_halo
        or interaction_layers[-1] - observation_layers[-1] < right_halo
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B device mapping lacks its declared FC3 range padding."
        )
    forbidden_report_flags = (
        "symmetrized",
        "corrected",
        "truncated",
        "thresholded",
    )
    if (
        mapping_report.get("certification_passed") is not True
        or mapping_report.get("failure_codes") not in ([], ())
        or mapping_report.get("permutation_missing_count") != 0
        or mapping_report.get("permutation_tolerance_passed") is not True
        or any(mapping_report.get(name) is not False for name in forbidden_report_flags)
        or not math.isfinite(forward_error_ratio)
        or forward_error_ratio < 0.0
        or forward_error_ratio > 1.0
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B device mapping report does not reproduce its certification."
        )
    if (
        acoustic_report.get("passed") is not True
        or acoustic_report.get("classification") != "acoustic"
        or acoustic_report.get("failure_reasons") not in ([], ())
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B harmonic lead did not pass the acoustic classification."
        )

    required_identities = (
        "raw_fc3_sha256",
        "canonical_fc3_sha256",
        "source_translational_sha256",
        "mass_normalized_fc3_sha256",
        "device_mapped_fc3_sha256",
        "mapping_report_sha256",
        "cubic_physics_sha256_physical",
        "sparse_vertex_storage_sha256",
        "sparse_vertex_metadata_sha256",
        "cubic_physics_sha256_solver",
        "fc2_content_sha256",
        "primitive_crystal_content_sha256",
        "fc2_crystal_mapping_content_sha256",
        "fc3_crystal_mapping_content_sha256",
        "layered_geometry_sha256",
    )
    for name in required_identities:
        _sha256(identities.get(name), name=f"phase_b identities[{name!r}]")
    if mapping_report.get("report_sha256") != identities["mapping_report_sha256"]:
        raise MaterialNQ1ProblemError(
            "Phase-B device mapping report hash does not match its identity."
        )
    if (
        mapping_report.get("cubic_physics_sha256")
        != identities["cubic_physics_sha256_physical"]
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B physical cubic hash does not match its mapping report."
        )
    if identities["sparse_vertex_storage_sha256"] != sparse_vertex.vertex_storage_sha256:
        raise MaterialNQ1ProblemError("Phase-B sparse vertex storage hash mismatch.")
    if identities["sparse_vertex_metadata_sha256"] != sparse_vertex.metadata_sha256:
        raise MaterialNQ1ProblemError("Phase-B sparse vertex metadata hash mismatch.")
    if identities["cubic_physics_sha256_solver"] != sparse_vertex.cubic_physics_sha256:
        raise MaterialNQ1ProblemError("Phase-B sparse vertex physics hash mismatch.")
    if array_hashes.get("indices") != _array_sha256(sparse_vertex.indices):
        raise MaterialNQ1ProblemError("Phase-B sparse index array hash mismatch.")
    if array_hashes.get("solver_coefficients") != _array_sha256(
        sparse_vertex.coefficients
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B sparse coefficient array hash mismatch."
        )
    _sha256(array_hashes.get("values"), name="phase_b physical value array hash")

    harmonic_identities = {
        "fc2_content_sha256": harmonic_model.fc2_content_sha256,
        "primitive_crystal_content_sha256": (
            harmonic_model.primitive_crystal_content_sha256
        ),
        "fc2_crystal_mapping_content_sha256": (
            harmonic_model.fc2_crystal_mapping_content_sha256
        ),
        "layered_geometry_sha256": harmonic_model.layered_geometry_sha256,
    }
    for name, expected in harmonic_identities.items():
        if identities[name] != expected:
            raise MaterialNQ1ProblemError(
                f"Phase-B {name} does not match the harmonic model."
            )

    vertex_metadata = _mapping(
        sparse_vertex.metadata, name="phase_b sparse vertex metadata"
    )
    if vertex_metadata.get("schema_version") != _PHASE_B_VERTEX_SCHEMA:
        raise MaterialNQ1ProblemError(
            "Phase-B sparse vertex is not an uncontinued material vertex."
        )
    if vertex_metadata.get("solver_scaled") is not True:
        raise MaterialNQ1ProblemError(
            "Phase-B sparse vertex is not pre-scaled to solver units."
        )
    if vertex_metadata.get("weights_applied") is not True:
        raise MaterialNQ1ProblemError("Phase-B sparse vertex lacks FC3 weights.")
    if any(vertex_metadata.get(name) is not False for name in forbidden_report_flags):
        raise MaterialNQ1ProblemError(
            "Phase-B sparse vertex metadata records a forbidden correction."
        )
    if any(
        name in vertex_metadata
        for name in (
            "cubic_continuation_factor",
            "cubic_continuation_applied_count",
            "uncontinued_vertex_storage_sha256",
        )
    ):
        raise MaterialNQ1ProblemError(
            "Phase-B sparse vertex has already undergone coupling continuation."
        )

    vertex_identity_links = {
        "source_fc3_content_sha256": "canonical_fc3_sha256",
        "device_mapped_fc3_sha256": "device_mapped_fc3_sha256",
        "mapping_report_sha256": "mapping_report_sha256",
        "cubic_physics_sha256_physical": "cubic_physics_sha256_physical",
        "fc2_content_sha256": "fc2_content_sha256",
        "primitive_crystal_content_sha256": "primitive_crystal_content_sha256",
        "fc2_crystal_mapping_content_sha256": (
            "fc2_crystal_mapping_content_sha256"
        ),
        "fc3_crystal_mapping_content_sha256": (
            "fc3_crystal_mapping_content_sha256"
        ),
        "layered_geometry_sha256": "layered_geometry_sha256",
    }
    for metadata_name, identity_name in vertex_identity_links.items():
        _exact_metadata_value(
            vertex_metadata,
            metadata_name,
            identities[identity_name],
            name="Phase-B sparse vertex metadata",
        )
    _exact_metadata_value(
        vertex_metadata,
        "solver_scale_omega0",
        harmonic_model.scale.omega0,
        name="Phase-B sparse vertex metadata",
    )
    _exact_metadata_value(
        vertex_metadata,
        "solver_scale_hbar",
        harmonic_model.scale.hbar,
        name="Phase-B sparse vertex metadata",
    )
    expected_factor = (
        math.sqrt(harmonic_model.scale.hbar)
        / harmonic_model.scale.omega0**2.5
    )
    _exact_metadata_value(
        vertex_metadata,
        "cubic_vertex_to_solver_factor",
        expected_factor,
        name="Phase-B sparse vertex metadata",
    )

    if sparse_vertex.nnz:
        active_layers = np.asarray(sparse_vertex.indices) // layer_dofs
        if not np.all(np.isin(active_layers, np.asarray(interaction_layers))):
            raise MaterialNQ1ProblemError(
                "Phase-B sparse support escapes the authenticated interaction crop."
            )

    binding_sha256 = domain_digest(
        "NEGFPY-PHASE-B-MATERIAL-NQ1-BINDING",
        {
            "schema_version": "negfpy.phase-b-material-nq1-binding.v1",
            "archive_sha256": digest_bytes(archive_sha256),
            "campaign_manifest_sha256": digest_bytes(campaign_sha256),
            "harmonic_model_physics_sha256": digest_bytes(
                harmonic_model.model_physics_sha256
            ),
            "sparse_vertex_storage_sha256": digest_bytes(
                sparse_vertex.vertex_storage_sha256
            ),
            "device_mapped_fc3_sha256": digest_bytes(
                identities["device_mapped_fc3_sha256"]
            ),
            "mapping_report_sha256": digest_bytes(
                identities["mapping_report_sha256"]
            ),
            "n_device_layers": n_layers,
            "n_principal_layer_dofs": layer_dofs,
            "interaction_layers": interaction_layers,
            "observation_layers": observation_layers,
            "solver_scale": {
                "omega0": harmonic_model.scale.omega0,
                "hbar": harmonic_model.scale.hbar,
            },
        },
    )
    return n_layers, binding_sha256, archive_sha256, campaign_sha256


def _validate_mapped_fc3(
    mapped: DeviceMappedFC3,
    model: LayeredHarmonicModel,
    *,
    layer_dofs: int,
) -> int:
    if not isinstance(mapped, DeviceMappedFC3):
        raise TypeError("mapped_fc3 must be a DeviceMappedFC3.")
    report = mapped.report
    if not report.certification_passed or report.failure_codes:
        raise MaterialNQ1ProblemError(
            "FC3 device mapping report did not pass complete-range preflight: "
            f"{report.failure_codes!r}."
        )
    forbidden = {
        "symmetrized": report.symmetrized,
        "corrected": report.corrected,
        "truncated": report.truncated,
        "thresholded": report.thresholded,
    }
    changed = tuple(name for name, value in forbidden.items() if value)
    if changed:
        raise MaterialNQ1ProblemError(
            "FC3 device mapping used forbidden support-changing operations: "
            f"{changed!r}."
        )
    if (
        report.permutation_missing_count != 0
        or not report.permutation_tolerance_passed
    ):
        raise MaterialNQ1ProblemError(
            "FC3 device mapping lacks complete, roundoff-bounded "
            "permutation certification."
        )

    expected_report_hash = domain_digest(
        "NEGFPY-FC3-DEVICE-MAPPING", _report_payload(report)
    )
    if expected_report_hash != report.report_sha256:
        raise MaterialNQ1ProblemError("FC3 mapping report identity does not reproduce.")
    expected_physics_hash = _cubic_physics_sha256(
        mapped.ndof,
        np.asarray(mapped.indices),
        np.asarray(mapped.values),
    )
    if (
        expected_physics_hash != mapped.cubic_physics_sha256
        or expected_physics_hash != report.cubic_physics_sha256
    ):
        raise MaterialNQ1ProblemError("Mapped FC3 physics identity does not reproduce.")
    expected_content_hash = domain_digest(
        "NEGFPY-DEVICE-MAPPED-FC3",
        {
            "ndof": mapped.ndof,
            "indices": np.asarray(mapped.indices),
            "values": np.asarray(mapped.values),
            "interaction_layers": mapped.interaction_layers,
            "observation_layers": mapped.observation_layers,
            "report_sha256": digest_bytes(report.report_sha256),
        },
    )
    if expected_content_hash != mapped.content_sha256:
        raise MaterialNQ1ProblemError("Mapped FC3 content identity does not reproduce.")

    joined_identities = (
        ("FC2 content", report.fc2_content_sha256, model.fc2_content_sha256),
        (
            "primitive crystal",
            report.primitive_crystal_content_sha256,
            model.primitive_crystal_content_sha256,
        ),
        (
            "FC2 crystal mapping",
            report.fc2_crystal_mapping_content_sha256,
            model.fc2_crystal_mapping_content_sha256,
        ),
        (
            "layered geometry",
            report.layered_geometry_sha256,
            model.layered_geometry_sha256,
        ),
    )
    for name, mapped_value, harmonic_value in joined_identities:
        if mapped_value != harmonic_value:
            raise MaterialNQ1ProblemError(
                f"FC2/FC3 {name} identity mismatch."
            )

    n_layers = int(report.n_device_layers)
    if n_layers < 1 or mapped.ndof != n_layers * layer_dofs:
        raise MaterialNQ1ProblemError(
            "Mapped FC3 ndof does not match the harmonic device geometry."
        )
    if report.contact_layers != (0, n_layers - 1):
        raise MaterialNQ1ProblemError(
            "FC3 report contact layers do not span the finite device."
        )
    interactions = _ordered_contiguous(
        mapped.interaction_layers, name="interaction_layers"
    )
    observations = _ordered_contiguous(
        mapped.observation_layers, name="observation_layers"
    )
    if (
        interactions != report.interaction_layers
        or observations != report.observation_layers
        or not set(observations).issubset(interactions)
        or interactions[0] <= 0
        or interactions[-1] >= n_layers - 1
    ):
        raise MaterialNQ1ProblemError(
            "FC3 interaction/observation layers do not define an interior crop."
        )
    # The finite contact layers are part of the harmonic device and therefore
    # count toward the noninteracting FC3 halo.  They are not reservoir FC3;
    # they are the lead-like device layers between the interaction crop and
    # the harmonic semi-infinite contacts.
    expected_buffers = (
        tuple(range(0, interactions[0])),
        tuple(range(interactions[-1] + 1, n_layers)),
    )
    if report.harmonic_buffer_layers != expected_buffers:
        raise MaterialNQ1ProblemError(
            "FC3 harmonic-buffer geometry does not reproduce."
        )
    if (
        len(expected_buffers[0]) < report.required_left_halo_layers
        or len(expected_buffers[1]) < report.required_right_halo_layers
        or observations[0] - interactions[0] < report.required_left_halo_layers
        or interactions[-1] - observations[-1] < report.required_right_halo_layers
    ):
        raise MaterialNQ1ProblemError(
            "FC3 crop lacks the directional range or observation padding "
            "declared by its report."
        )

    indices = np.asarray(mapped.indices)
    values = np.asarray(mapped.values)
    if indices.ndim != 2 or indices.shape[1:] != (3,):
        raise MaterialNQ1ProblemError("Mapped FC3 indices must have shape (nnz, 3).")
    if values.shape != (indices.shape[0],):
        raise MaterialNQ1ProblemError(
            "Mapped FC3 values do not match its sparse indices."
        )
    if indices.dtype.kind not in "iu":
        raise MaterialNQ1ProblemError("Mapped FC3 indices are not exact integers.")
    if np.any(indices < 0) or np.any(indices >= mapped.ndof):
        raise MaterialNQ1ProblemError("Mapped FC3 index lies outside the device.")
    if len(indices):
        active_layers = indices // layer_dofs
        if not np.all(np.isin(active_layers, np.asarray(interactions))):
            raise MaterialNQ1ProblemError(
                "Mapped FC3 support escapes the certified interaction crop."
            )
    if mapped.nnz != report.final_nonzero_count:
        raise MaterialNQ1ProblemError(
            "Mapped FC3 support count differs from its report."
        )
    metadata = mapped.metadata
    if (
        metadata.get("solver_scaled") is not False
        or metadata.get("weights_applied") is not True
        or metadata.get("mapping_report_sha256") != report.report_sha256
    ):
        raise MaterialNQ1ProblemError(
            "Mapped FC3 metadata does not describe physical-SI, weighted input."
        )
    return n_layers


def _continued_vertex(
    base: SparseCubicVertex,
    *,
    coupling_scale: float,
    assembly_sha256: str,
    source_metadata: Mapping[str, Any] | None = None,
) -> SparseCubicVertex:
    coefficients = np.asarray(base.coefficients) * coupling_scale
    if not np.all(np.isfinite(coefficients)):
        raise MaterialNQ1ProblemError(
            "Cubic continuation scaling produced a non-finite coefficient."
        )
    if coupling_scale == 0.0:
        indices = np.empty((0, 3), dtype=np.int64)
        coefficients = np.empty((0,), dtype=np.float64)
    else:
        if np.any(coefficients == 0.0):
            raise MaterialNQ1ProblemError(
                "Cubic continuation scaling underflowed nonzero FC3 support."
            )
        indices = base.indices
    return SparseCubicVertex(
        ndof=base.ndof,
        indices=indices,
        coefficients=coefficients,
        metadata={
            **dict(base.metadata),
            **({} if source_metadata is None else dict(source_metadata)),
            "schema_version": "negfpy.sparse-cubic-vertex.material-problem.v1",
            "uncontinued_vertex_storage_sha256": base.vertex_storage_sha256,
            "uncontinued_cubic_physics_sha256": base.cubic_physics_sha256,
            "cubic_continuation_factor": coupling_scale,
            "cubic_continuation_applied_count": 1,
            "material_nq1_assembly_sha256": assembly_sha256,
        },
    )


def _build_problem_from_scaled_vertex(
    harmonic_model: LayeredHarmonicModel,
    base_vertex: SparseCubicVertex,
    *,
    n_layers: int,
    layer_dofs: int,
    grid: SignedFrequencyGrid,
    mean_temperature: float,
    temperature_bias: float,
    coupling_scale: float,
    source_identity: Mapping[str, Any],
    source_metadata: Mapping[str, Any] | None = None,
) -> CubicSCBAProblem:
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    mean = _finite_scalar(
        mean_temperature,
        name="mean_temperature",
        nonnegative=True,
    )
    bias = _finite_scalar(temperature_bias, name="temperature_bias")
    coupling = _finite_scalar(
        coupling_scale,
        name="coupling_scale",
        nonnegative=True,
    )
    left_temperature = mean + 0.5 * bias
    right_temperature = mean - 0.5 * bias
    if left_temperature < 0.0 or right_temperature < 0.0:
        raise ValueError(
            "temperature_bias makes at least one reservoir temperature negative."
        )
    if (
        not isinstance(base_vertex, SparseCubicVertex)
        or base_vertex.ndof != n_layers * layer_dofs
    ):
        raise MaterialNQ1ProblemError(
            "Pre-scaled cubic vertex does not match the harmonic device dimension."
        )

    assembly_sha256 = domain_digest(
        "NEGFPY-MATERIAL-NQ1-PROBLEM-ASSEMBLY",
        {
            "schema_version": "negfpy.material-nq1-problem.v2",
            "harmonic_model_physics_sha256": digest_bytes(
                harmonic_model.model_physics_sha256
            ),
            "base_vertex_storage_sha256": digest_bytes(
                base_vertex.vertex_storage_sha256
            ),
            "source_identity": source_identity,
            "solver_scale": {
                "omega0": harmonic_model.scale.omega0,
                "hbar": harmonic_model.scale.hbar,
            },
            "n_device_layers": n_layers,
            "layer_dofs": layer_dofs,
            "left_device_to_lead": np.asarray(harmonic_model.d10),
            "right_device_to_lead": np.asarray(harmonic_model.d01),
            "contact_orientation": "left=d10,right=d01,increasing-layer-lead",
            "grid": {"step": grid.step, "n_positive": grid.n_positive},
            "mean_temperature": mean,
            "temperature_bias": bias,
            "left_temperature": left_temperature,
            "right_temperature": right_temperature,
            "cubic_continuation_factor": coupling,
        },
    )
    vertex = _continued_vertex(
        base_vertex,
        coupling_scale=coupling,
        assembly_sha256=assembly_sha256,
        source_metadata=source_metadata,
    )

    raw_device = harmonic_model.device(n_layers)
    device = Device1D(
        onsite_blocks=[
            np.array(block, dtype=np.complex128, order="C", copy=True)
            for block in raw_device.onsite_blocks
        ],
        coupling_blocks=[
            np.array(block, dtype=np.complex128, order="C", copy=True)
            for block in raw_device.coupling_blocks
        ],
    )
    for block in (*device.onsite_blocks, *device.coupling_blocks):
        block.setflags(write=False)
    if any(
        not np.array_equal(block, harmonic_model.d00)
        for block in device.onsite_blocks
    ) or any(
        not np.array_equal(block, harmonic_model.d01)
        for block in device.coupling_blocks
    ):
        raise MaterialNQ1ProblemError(
            "Layered harmonic device is not the exact repeated lead operator."
        )

    ndof = n_layers * layer_dofs
    left_indices = np.arange(layer_dofs, dtype=np.int64)
    right_indices = np.arange(ndof - layer_dofs, ndof, dtype=np.int64)
    left = HarmonicReservoir(
        lead=harmonic_model.lead,
        temperature=left_temperature,
        device_to_lead=np.asarray(harmonic_model.d10),
        device_indices=left_indices,
    )
    right = HarmonicReservoir(
        lead=harmonic_model.lead,
        temperature=right_temperature,
        device_to_lead=np.asarray(harmonic_model.d01),
        device_indices=right_indices,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=left,
        right=right,
        vertex=vertex,
        grid=grid,
        scale=harmonic_model.scale,
    )
    if problem.ndof != base_vertex.ndof or problem.vertex.ndof != base_vertex.ndof:
        raise MaterialNQ1ProblemError(
            "Final harmonic and cubic problem dimensions do not agree."
        )
    return problem


def build_material_nq1_problem(
    harmonic_model: LayeredHarmonicModel,
    mapped_fc3: DeviceMappedFC3,
    *,
    grid: SignedFrequencyGrid,
    mean_temperature: float,
    temperature_bias: float = 0.0,
    coupling_scale: float = 1.0,
) -> CubicSCBAProblem:
    """Join certified FC2/FC3 stages into one solver-ready material problem.

    Temperatures are in solver energy units.  ``temperature_bias`` is signed
    and is defined as ``T_left - T_right`` around ``mean_temperature``.
    ``coupling_scale`` is the nonnegative cubic continuation factor applied
    once, after the physical mapped FC3 has been converted with the harmonic
    model's :class:`SolverScale`.
    """

    layer_dofs = _validate_harmonic_model(harmonic_model)
    n_layers = _validate_mapped_fc3(
        mapped_fc3,
        harmonic_model,
        layer_dofs=layer_dofs,
    )
    scale = harmonic_model.scale
    base_vertex = scale_device_fc3(mapped_fc3, scale)
    if base_vertex.ndof != mapped_fc3.ndof or not np.array_equal(
        base_vertex.indices, mapped_fc3.indices
    ):
        raise MaterialNQ1ProblemError(
            "Scaled sparse vertex changed mapped FC3 indices or ndof."
        )

    return _build_problem_from_scaled_vertex(
        harmonic_model,
        base_vertex,
        n_layers=n_layers,
        layer_dofs=layer_dofs,
        grid=grid,
        mean_temperature=mean_temperature,
        temperature_bias=temperature_bias,
        coupling_scale=coupling_scale,
        source_identity={
            "kind": "device_mapped_fc3",
            "device_mapped_fc3_sha256": digest_bytes(mapped_fc3.content_sha256),
            "mapping_report_sha256": digest_bytes(
                mapped_fc3.report.report_sha256
            ),
        },
    )


def build_material_nq1_problem_from_phase_b(
    harmonic_model: LayeredHarmonicModel,
    sparse_vertex: SparseCubicVertex,
    phase_b_archive: Mapping[str, Any],
    *,
    grid: SignedFrequencyGrid,
    mean_temperature: float,
    temperature_bias: float = 0.0,
    coupling_scale: float = 1.0,
) -> CubicSCBAProblem:
    """Assemble an exact problem from an authenticated Phase-B HDF5 load.

    The HDF5 reader deliberately remains in the validation-script layer.  A
    caller first uses ``verify_phase_b_hdf5`` and
    ``load_phase_b_sparse_vertex`` from that script, then passes their
    returned mappings here.  This path neither reads raw FC3 nor invokes
    ``map_fc3_to_device`` or ``scale_device_fc3``; the archived coefficients
    are already in solver units and receive only the explicit continuation
    factor requested for this problem.
    """

    layer_dofs = _validate_harmonic_model(harmonic_model)
    n_layers, binding, archive_sha256, campaign_sha256 = (
        _authenticate_phase_b_input(
            harmonic_model,
            sparse_vertex,
            phase_b_archive,
            layer_dofs=layer_dofs,
        )
    )
    return _build_problem_from_scaled_vertex(
        harmonic_model,
        sparse_vertex,
        n_layers=n_layers,
        layer_dofs=layer_dofs,
        grid=grid,
        mean_temperature=mean_temperature,
        temperature_bias=temperature_bias,
        coupling_scale=coupling_scale,
        source_identity={
            "kind": "authenticated_phase_b_archive",
            "phase_b_binding_sha256": digest_bytes(binding),
            "phase_b_archive_sha256": digest_bytes(archive_sha256),
            "phase_b_campaign_manifest_sha256": digest_bytes(campaign_sha256),
        },
        source_metadata={
            "phase_b_archive_schema": _PHASE_B_ARCHIVE_SCHEMA,
            "phase_b_archive_sha256": archive_sha256,
            "phase_b_campaign_manifest_sha256": campaign_sha256,
            "phase_b_binding_sha256": binding,
            "phase_b_input_authenticated": True,
            "phase_b_fc3_remapped": False,
            "phase_b_vertex_rescaled_to_solver_units": False,
        },
    )


__all__ = [
    "MaterialNQ1ProblemError",
    "build_material_nq1_problem",
    "build_material_nq1_problem_from_phase_b",
]
