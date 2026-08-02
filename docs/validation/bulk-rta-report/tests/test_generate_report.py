from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/generate_report.py"
SPEC = importlib.util.spec_from_file_location("bulk_rta_report_generator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mesh_evidence() -> list[dict[str, Any]]:
    records = []
    for mesh, rate_character, harmonic_character in (
        (3, "3", "c"),
        (5, "5", "d"),
        (11, "b", "e"),
    ):
        records.append(
            {
                "mesh": [mesh, mesh, mesh],
                "n_qpoints": mesh**3,
                "passed": True,
                "core_passed": True,
                "end_to_end_required": mesh in (3, 5),
                "end_to_end_passed": True if mesh in (3, 5) else None,
                "archive_sha256": "a" * 64,
                "rate_sha256": rate_character * 64,
                "rta_sha256": rate_character * 64,
                "harmonic_sha256": harmonic_character * 64,
                "harmonic_metrics": {
                    "passed": True,
                    "fc2_schema_audit_passed": True,
                    "fc2_physical_audit_passed": True,
                    "uniform_mesh_frequency_max_abs_error_thz": 1.0e-13,
                    "group_velocity_max_abs_error_thz_angstrom": 1.0e-13,
                    "band_frequency_max_abs_error_thz": 1.0e-13,
                    "dos_max_abs_error_states_per_thz_cell": 1.0e-14,
                    "dos_integral_states_per_cell": 6.0,
                    "dos_expected_states_per_cell": 6,
                    "dos_normalization_error_states_per_cell": 0.0,
                    "dos_normalization_certified": True,
                },
                "maximum_heat_capacity_absolute_error_ev_per_k": 1.0e-20,
                "mode_kappa_equal": True,
                "kappa_diagonal_nonnegative": True,
                "kappa_offdiagonal_within_tolerance": True,
                "negfpy_kappa_voigt_w_mk": [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                ],
                "reference_kappa_voigt_w_mk": [
                    [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
                ],
            }
        )
    return records


def _runtime() -> dict[str, Any]:
    return {
        "clean": True,
        "commit": "1" * 40,
        "tree": "2" * 40,
        "negfpy_version": "1.0.0",
        "package_versions": {"phono3py": "4.4.0"},
        "negfpy_import_tied_to_repository_payload": True,
        "code_provenance": {
            "package_version_consistent": True,
            "installed_package_version": "1.0.0",
            "git_dirty": False,
            "git_revision": "1" * 40,
        },
    }


def _settings() -> dict[str, Any]:
    return {
        "meshes": [3, 5, 11],
        "end_to_end_meshes": [3, 5],
        "temperatures_k": [300.0],
        "sigma_thz": 0.1,
        "cutoff_frequency_thz": 1.0e-4,
    }


def _write_hdf5(
    path: Path,
    *,
    campaign_sha256: str,
    evidence_sha256: str,
    records: list[dict[str, Any]],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["schema"] = GENERATOR.FINAL_HDF5_SCHEMA
        handle.attrs["campaign_sha256"] = campaign_sha256
        handle.attrs["evidence_schema"] = GENERATOR.FINAL_EVIDENCE_SCHEMA
        handle.attrs["evidence_sha256"] = evidence_sha256
        meshes_group = handle.create_group("meshes")
        for record in records:
            mesh = int(record["mesh"][0])
            n_qpoints = mesh**3
            n_bands = 6
            group = meshes_group.create_group(f"mesh-{mesh}x{mesh}x{mesh}")
            group.attrs["n_qpoints"] = n_qpoints
            group.attrs["passed"] = True
            group.attrs["source_rta_sha256"] = record["rta_sha256"]
            group.attrs["source_harmonic_sha256"] = record["harmonic_sha256"]
            group.attrs["metrics_json"] = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
            )
            rta = group.create_group("rta")
            frequencies = np.ones((n_qpoints, n_bands))
            velocities = np.zeros((n_qpoints, n_bands, 3))
            mode = np.zeros((1, n_qpoints, n_bands, 3, 3))
            fields = {
                "frequencies_thz": frequencies,
                "group_velocities_thz_angstrom": velocities,
                "heat_capacities_ev_per_k": np.ones((1, n_qpoints, n_bands)),
                "gamma_thz_hwhm": np.ones((1, n_qpoints, n_bands)),
                "population_lifetime_ps": np.ones((1, n_qpoints, n_bands)),
                "mean_free_path_angstrom": np.ones((1, n_qpoints, n_bands)),
                "mode_kappa_w_mk": mode,
                "kappa_w_mk": np.zeros((1, 3, 3)),
            }
            for field, values in fields.items():
                rta.create_dataset(field, data=values)
            harmonic = group.create_group("harmonic")
            harmonic.create_dataset("band/qpoints", data=np.zeros((4, 3)))
            harmonic.create_dataset(
                "band/distances_inv_angstrom",
                data=np.arange(4.0),
            )
            harmonic.create_dataset("band/vertex_indices", data=np.arange(4))
            harmonic.create_dataset(
                "band/frequencies_thz",
                data=np.ones((4, n_bands)),
            )
            harmonic.create_dataset(
                "dos/frequency_points_thz",
                data=np.linspace(0.0, 2.0, 5),
            )
            harmonic.create_dataset(
                "dos/total_states_per_thz_cell",
                data=np.ones(5),
            )
            harmonic.create_dataset(
                "dos/projected_states_per_thz_cell",
                data=np.ones((2, 3, 5)),
            )
            harmonic.create_dataset("mesh/frequencies_thz", data=frequencies)
            harmonic.create_dataset(
                "mesh/group_velocities_thz_angstrom",
                data=velocities,
            )


def _write_locked_artifact(
    root: Path,
    *,
    candidate: bool = True,
    evidence_release_certified: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    campaign_path = root / "campaign-manifest.json"
    _write_json(campaign_path, {"schema": "synthetic.campaign.v1"})
    campaign_sha256 = _sha256(campaign_path)
    records = _mesh_evidence()
    settings = _settings()
    evidence = {
        "schema": GENERATOR.FINAL_EVIDENCE_SCHEMA,
        "campaign_sha256": campaign_sha256,
        "scientific_passed": True,
        "candidate_release_gates_passed": candidate,
        "release_certified": evidence_release_certified,
        "certification_authority": "release-completion-receipt.json",
        "settings": settings,
        "source_sha256": {"fc2": "3" * 64, "fc3": "4" * 64},
        "archive_sha256": {"3": "5" * 64, "5": "6" * 64, "11": "7" * 64},
        "meshes": records,
    }
    evidence_path = root / "release-evidence.json"
    _write_json(evidence_path, evidence)
    evidence_sha256 = _sha256(evidence_path)
    hdf5_path = root / "bulk-rta-results.hdf5"
    _write_hdf5(
        hdf5_path,
        campaign_sha256=campaign_sha256,
        evidence_sha256=evidence_sha256,
        records=records,
    )
    runtime = _runtime()
    report = {
        "schema": GENERATOR.EXPECTED_SCHEMA,
        "mode": "release" if candidate else "development",
        "scientific_passed": True,
        "candidate_release_gates_passed": candidate,
        "release_certified": False,
        "certification_authority": "release-completion-receipt.json",
        "physical_mesh_convergence_certified": False,
        "physical_mesh_convergence_note": "Implementation validation only.",
        "approximation": "intrinsic_three_phonon_smrta",
        "excluded_physics": ["iterative_lbte"],
        "campaign_manifest": str(campaign_path),
        "campaign_sha256": campaign_sha256,
        "evidence_document": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "results_hdf5": str(hdf5_path),
        "results_hdf5_sha256": _sha256(hdf5_path),
        "runtime_before": runtime,
        "runtime_after": dict(runtime),
        "runtime_unchanged": True,
        "inputs": {
            "source_paths": {},
            "source_sha256": evidence["source_sha256"],
            "archive_paths": {},
            "archive_sha256": evidence["archive_sha256"],
        },
        "settings": settings,
        "mesh_evidence": records,
        "meshes": [],
    }
    report_path = root / "bulk-rta-validation.json"
    _write_json(report_path, report)
    checksums_path = root / "SHA256SUMS.json"
    manifest = {
        path.name: _sha256(path)
        for path in (campaign_path, evidence_path, hdf5_path, report_path)
    }
    _write_json(checksums_path, manifest)
    receipt = {
        "schema": GENERATOR.FINAL_RECEIPT_SCHEMA,
        "certification_authority": "sole",
        "campaign_sha256": campaign_sha256,
        "scientific_passed": True,
        "candidate_release_gates_passed": candidate,
        "provisional_release_certified": False,
        "release_certified": candidate,
        "report_sha256": _sha256(report_path),
        "evidence_sha256": evidence_sha256,
        "results_hdf5_sha256": _sha256(hdf5_path),
        "sha256sums_sha256": _sha256(checksums_path),
        "evidence_scientific_passed": True,
        "evidence_candidate_release_gates_passed": candidate,
    }
    receipt_path = root / "release-completion-receipt.json"
    _write_json(receipt_path, receipt)
    return report_path, hdf5_path, evidence_path, checksums_path, receipt_path


class ReportGeneratorTest(unittest.TestCase):
    def test_locked_receipt_is_the_only_certification_authority(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report_path, hdf5_path, _, checksums_path, receipt_path = (
                _write_locked_artifact(root)
            )
            report = GENERATOR._load_json(report_path)
            records = GENERATOR._validate_json(report)
            evidence, receipt, loaded_checksums = GENERATOR._load_release_evidence(
                report,
                json_path=report_path,
                hdf5_path=hdf5_path,
            )
            self.assertIsNotNone(evidence)
            self.assertIsNotNone(receipt)
            assert evidence is not None and receipt is not None
            self.assertFalse(report["release_certified"])
            self.assertFalse(evidence["release_certified"])
            self.assertTrue(receipt["release_certified"])
            self.assertEqual(loaded_checksums, checksums_path.resolve())
            arrays, optional = GENERATOR._load_scientific_arrays(
                hdf5_path,
                report,
                evidence,
                _sha256(report_path),
                records,
            )
            self.assertEqual(set(arrays), {3, 5, 11})
            self.assertEqual(optional["band"]["labels"], ["Γ", "X", "L", "Γ"])
            harmonic_complete = GENERATOR._harmonic_reporting_complete(
                report,
                records,
                optional,
            )
            self.assertTrue(harmonic_complete)
            self.assertEqual(
                GENERATOR._release_gaps(
                    report,
                    records,
                    harmonic_complete=harmonic_complete,
                    receipt=receipt,
                ),
                [],
            )
            self.assertEqual(
                receipt["sha256sums_sha256"],
                _sha256(checksums_path),
            )
            self.assertTrue(receipt_path.is_file())

    def test_valid_negative_receipt_remains_draft(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report_path, hdf5_path, *_ = _write_locked_artifact(
                root,
                candidate=False,
            )
            report = GENERATOR._load_json(report_path)
            records = GENERATOR._validate_json(report)
            _, receipt, _ = GENERATOR._load_release_evidence(
                report,
                json_path=report_path,
                hdf5_path=hdf5_path,
            )
            self.assertIsNotNone(receipt)
            assert receipt is not None
            self.assertFalse(receipt["release_certified"])
            gaps = GENERATOR._release_gaps(
                report,
                records,
                harmonic_complete=True,
                receipt=receipt,
            )
            self.assertIn(
                "release-completion-receipt.json certifies the candidate",
                gaps,
            )

    def test_missing_receipt_is_rejected_for_frozen_schema(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report_path, hdf5_path, *_, receipt_path = _write_locked_artifact(root)
            receipt_path.unlink()
            report = GENERATOR._load_json(report_path)
            with self.assertRaisesRegex(ValueError, "completion receipt does not exist"):
                GENERATOR._load_release_evidence(
                    report,
                    json_path=report_path,
                    hdf5_path=hdf5_path,
                )

    def test_receipt_hash_decision_and_type_tampering_are_rejected(self) -> None:
        mutations = {
            "report hash": {"report_sha256": "0" * 64},
            "evidence hash": {"evidence_sha256": "0" * 64},
            "HDF5 hash": {"results_hdf5_sha256": "0" * 64},
            "checksum hash": {"sha256sums_sha256": "0" * 64},
            "candidate decision": {
                "evidence_candidate_release_gates_passed": False
            },
            "boolean type": {"release_certified": 1},
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as name:
                root = Path(name)
                report_path, hdf5_path, *_, receipt_path = _write_locked_artifact(
                    root
                )
                receipt = GENERATOR._load_json(receipt_path)
                receipt.update(mutation)
                _write_json(receipt_path, receipt)
                report = GENERATOR._load_json(report_path)
                with self.assertRaisesRegex(ValueError, "Completion receipt binding"):
                    GENERATOR._load_release_evidence(
                        report,
                        json_path=report_path,
                        hdf5_path=hdf5_path,
                    )

    def test_provisional_report_flag_and_physical_convergence_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report_path, *_ = _write_locked_artifact(root)
            report = GENERATOR._load_json(report_path)
            report["release_certified"] = True
            with self.assertRaisesRegex(ValueError, "must remain provisional"):
                GENERATOR._validate_json(report)
            report["release_certified"] = False
            report["physical_mesh_convergence_certified"] = True
            with self.assertRaisesRegex(ValueError, "must not claim physical"):
                GENERATOR._validate_json(report)

    def test_provisional_evidence_flag_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report_path, hdf5_path, *_ = _write_locked_artifact(
                root,
                evidence_release_certified=True,
            )
            report = GENERATOR._load_json(report_path)
            GENERATOR._validate_json(report)
            with self.assertRaisesRegex(
                ValueError,
                "Release JSON/evidence binding failed",
            ):
                GENERATOR._load_release_evidence(
                    report,
                    json_path=report_path,
                    hdf5_path=hdf5_path,
                )

    def test_manifest_tampering_is_rejected_before_receipt_use(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            report_path, hdf5_path, *_ = _write_locked_artifact(root)
            (root / "unexpected.txt").write_text("tamper\n", encoding="utf-8")
            report = GENERATOR._load_json(report_path)
            with self.assertRaisesRegex(ValueError, "exact artifact tree"):
                GENERATOR._load_release_evidence(
                    report,
                    json_path=report_path,
                    hdf5_path=hdf5_path,
                )

    def test_uniform_mesh_dos_has_six_mode_normalization(self) -> None:
        frequencies = np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.2, 2.2, 3.2, 4.2, 5.2, 6.2],
            ]
        )
        grid, values = GENERATOR._gaussian_dos(frequencies, 0.05)
        self.assertAlmostEqual(
            float(np.trapezoid(values, grid)),
            6.0,
            delta=2.0e-6,
        )


if __name__ == "__main__":
    unittest.main()
