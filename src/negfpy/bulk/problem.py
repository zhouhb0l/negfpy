"""User-facing construction of audited bulk FC2/FC3 transport problems."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from negfpy.anharmonic.bulk_harmonic import (
    PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ,
    BulkPhononSpectrum,
    compute_bulk_phonons,
)
from negfpy.anharmonic.bulk_three_phonon import BulkThreePhononProblem
from negfpy.anharmonic.crystal import CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data, FC2ValidationReport, validate_fc2
from negfpy.anharmonic.io import (
    read_phono3py_crystal_mapping,
    read_phono3py_fc2,
    read_phono3py_fc3,
)
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import convert_fc3_to_translational
from negfpy.anharmonic.translational_fc3_validation import (
    TranslationalFC3ValidationReport,
    validate_translational_fc3,
)

from .fc2_audit import MappedFC2AuditReport, audit_mapped_fc2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _source_path(value: str | Path, *, name: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{name} source must not be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} source does not exist.") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} source must be a regular file.")
    return resolved


@dataclass(frozen=True, slots=True, eq=False)
class BulkHarmonicProblem:
    """Audited FC2-only problem for bands, uniform-mesh phonons, and DOS."""

    fc2: FC2Data
    mapping: CrystalMapping
    source_paths: Mapping[str, str]
    source_sha256: Mapping[str, str]
    fc2_audit: FC2ValidationReport
    fc2_physical_audit: MappedFC2AuditReport
    frequency_factor_to_thz: float = PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ
    input_format: str = "phono3py-hdf5"

    def __post_init__(self) -> None:
        if not isinstance(self.fc2, FC2Data):
            raise TypeError("fc2 must be an FC2Data instance.")
        if not isinstance(self.mapping, CrystalMapping) or self.mapping.purpose != "fc2":
            raise TypeError("mapping must be an FC2 CrystalMapping.")
        if not isinstance(self.fc2_audit, FC2ValidationReport):
            raise TypeError("fc2_audit must be an FC2ValidationReport.")
        if not self.fc2_audit.passed:
            raise ValueError("BulkHarmonicProblem requires a passing FC2 schema audit.")
        if not isinstance(self.fc2_physical_audit, MappedFC2AuditReport):
            raise TypeError("fc2_physical_audit must be a MappedFC2AuditReport.")
        if not self.fc2_physical_audit.passed:
            raise ValueError("BulkHarmonicProblem requires a passing physical FC2 audit.")
        paths = {str(key): str(value) for key, value in self.source_paths.items()}
        hashes = {str(key): str(value) for key, value in self.source_sha256.items()}
        expected = {"crystal", "fc2"}
        if set(paths) != expected or set(hashes) != expected:
            raise ValueError("Harmonic provenance must contain crystal and fc2.")
        if any(not _is_sha256(value) for value in hashes.values()):
            raise ValueError("Every source SHA256 must have 64 hexadecimal digits.")
        factor = float(self.frequency_factor_to_thz)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("frequency_factor_to_thz must be positive.")
        object.__setattr__(self, "source_paths", MappingProxyType(paths))
        object.__setattr__(self, "source_sha256", MappingProxyType(hashes))
        object.__setattr__(self, "frequency_factor_to_thz", factor)

    @classmethod
    def from_phono3py(
        cls,
        crystal_yaml: str | Path,
        fc2_hdf5: str | Path,
        *,
        frequency_factor_to_thz: float = PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ,
    ) -> BulkHarmonicProblem:
        paths = {
            "crystal": _source_path(crystal_yaml, name="crystal"),
            "fc2": _source_path(fc2_hdf5, name="fc2"),
        }
        hashes_before = {name: _sha256(path) for name, path in paths.items()}
        mapping = read_phono3py_crystal_mapping(paths["crystal"], purpose="fc2")
        fc2 = read_phono3py_fc2(
            paths["fc2"],
            mapping.supercell.masses,
            p2s_map=mapping.p2s,
            s2p_map=mapping.s2p,
        )
        hashes_after = {name: _sha256(path) for name, path in paths.items()}
        if hashes_after != hashes_before:
            raise RuntimeError("A harmonic source changed while it was being read.")
        audit = validate_fc2(fc2)
        if not audit.passed:
            raise ValueError("FC2 input failed its schema audit: " + " ".join(audit.errors))
        physical_audit = audit_mapped_fc2(fc2, mapping)
        if not physical_audit.passed:
            raise ValueError(
                "FC2 input failed its mapped physical audit: "
                + " ".join(physical_audit.errors)
            )
        return cls(
            fc2=fc2,
            mapping=mapping,
            source_paths={name: str(path) for name, path in paths.items()},
            source_sha256=hashes_after,
            fc2_audit=audit,
            fc2_physical_audit=physical_audit,
            frequency_factor_to_thz=frequency_factor_to_thz,
        )

    @property
    def primitive_volume_angstrom3(self) -> float:
        return abs(float(np.linalg.det(self.mapping.primitive.lattice)))

    def compute_uniform_mesh_phonons(
        self,
        mesh: tuple[int, int, int],
    ) -> BulkPhononSpectrum:
        grid = UniformBZMesh(mesh)
        qpoints = np.asarray(grid.addresses, dtype=np.float64) / np.asarray(
            grid.shape,
            dtype=np.float64,
        )[None, :]
        return compute_bulk_phonons(
            self.fc2,
            self.mapping,
            qpoints,
            frequency_factor_to_thz=self.frequency_factor_to_thz,
            hermitianize=True,
            sampling_kind="uniform_full_mesh",
            mesh_shape=grid.shape,
        )


@dataclass(frozen=True, slots=True, eq=False)
class BulkTransportProblem:
    """Audited phono3py FC2/FC3 problem with immutable input provenance."""

    three_phonon: BulkThreePhononProblem
    source_paths: Mapping[str, str]
    source_sha256: Mapping[str, str]
    fc2_audit: FC2ValidationReport
    fc2_physical_audit: MappedFC2AuditReport
    fc3_audit: TranslationalFC3ValidationReport
    input_format: str = "phono3py-hdf5"

    def __post_init__(self) -> None:
        if not isinstance(self.three_phonon, BulkThreePhononProblem):
            raise TypeError("three_phonon must be a BulkThreePhononProblem.")
        if not isinstance(self.fc2_audit, FC2ValidationReport):
            raise TypeError("fc2_audit must be an FC2ValidationReport.")
        if not isinstance(self.fc3_audit, TranslationalFC3ValidationReport):
            raise TypeError(
                "fc3_audit must be a TranslationalFC3ValidationReport."
            )
        if not isinstance(self.fc2_physical_audit, MappedFC2AuditReport):
            raise TypeError("fc2_physical_audit must be a MappedFC2AuditReport.")
        if not (
            self.fc2_audit.passed
            and self.fc2_physical_audit.passed
            and self.fc3_audit.passed
        ):
            raise ValueError("BulkTransportProblem requires passing IFC audits.")
        paths = {str(key): str(value) for key, value in self.source_paths.items()}
        hashes = {str(key): str(value) for key, value in self.source_sha256.items()}
        expected = {"crystal", "fc2", "fc3"}
        if set(paths) != expected or set(hashes) != expected:
            raise ValueError("Source provenance must contain crystal, fc2, and fc3.")
        if any(not _is_sha256(value) for value in hashes.values()):
            raise ValueError("Every source SHA256 must have 64 hexadecimal digits.")
        if self.input_format != "phono3py-hdf5":
            raise ValueError("The 1.0 release supports only phono3py HDF5 input.")
        object.__setattr__(self, "source_paths", MappingProxyType(paths))
        object.__setattr__(self, "source_sha256", MappingProxyType(hashes))

    @classmethod
    def from_phono3py(
        cls,
        crystal_yaml: str | Path,
        fc2_hdf5: str | Path,
        fc3_hdf5: str | Path,
        *,
        mesh: tuple[int, int, int],
        sigma_thz: float,
        cutoff_frequency_thz: float = 1.0e-4,
        degenerate_tolerance_thz: float = 1.0e-4,
        make_r0_average: bool = True,
        frequency_factor_to_thz: float = 15.633302300230191,
        metadata: Mapping[str, Any] | None = None,
    ) -> BulkTransportProblem:
        """Read, map, and audit one phono3py bulk material problem.

        No force constant is symmetrized, corrected, truncated, remapped, or
        mass-normalized silently.  The original source paths and SHA256 values
        remain bound to the returned problem.
        """

        paths = {
            "crystal": _source_path(crystal_yaml, name="crystal"),
            "fc2": _source_path(fc2_hdf5, name="fc2"),
            "fc3": _source_path(fc3_hdf5, name="fc3"),
        }
        hashes_before = {name: _sha256(path) for name, path in paths.items()}
        harmonic = BulkHarmonicProblem.from_phono3py(
            paths["crystal"],
            paths["fc2"],
            frequency_factor_to_thz=frequency_factor_to_thz,
        )
        mapping2 = harmonic.mapping
        mapping3 = read_phono3py_crystal_mapping(paths["crystal"], purpose="fc3")
        fc2 = harmonic.fc2
        fc3_raw = read_phono3py_fc3(
            paths["fc3"],
            mapping3.supercell.masses,
            p2s_map=mapping3.p2s,
            s2p_map=mapping3.s2p,
        )
        fc3 = convert_fc3_to_translational(
            fc3_raw,
            mapping3,
            make_r0_average=make_r0_average,
        )
        hashes_after = {name: _sha256(path) for name, path in paths.items()}
        if hashes_after != hashes_before:
            raise RuntimeError("A bulk transport source changed while it was being read.")
        fc2_audit = harmonic.fc2_audit
        fc3_audit = validate_translational_fc3(fc3, mapping3)
        if not fc2_audit.passed:
            raise ValueError(
                "FC2 input failed the release audit: "
                + " ".join(fc2_audit.errors)
            )
        if not fc3_audit.passed:
            raise ValueError(
                "FC3 input failed the release audit: "
                + " ".join(fc3_audit.errors)
            )
        problem_metadata = {
            "schema": "negfpy.bulk_transport_problem.v1",
            "input_format": "phono3py-hdf5",
            "make_r0_average": bool(make_r0_average),
            "user_metadata": {} if metadata is None else dict(metadata),
        }
        problem = BulkThreePhononProblem(
            fc2=fc2,
            fc2_mapping=mapping2,
            fc3=fc3,
            fc3_mapping=mapping3,
            mesh=UniformBZMesh(mesh),
            sigma_thz=sigma_thz,
            cutoff_frequency_thz=cutoff_frequency_thz,
            frequency_factor_to_thz=frequency_factor_to_thz,
            degenerate_tolerance_thz=degenerate_tolerance_thz,
            metadata=problem_metadata,
        )
        return cls(
            three_phonon=problem,
            source_paths={name: str(path) for name, path in paths.items()},
            source_sha256=hashes_after,
            fc2_audit=fc2_audit,
            fc2_physical_audit=harmonic.fc2_physical_audit,
            fc3_audit=fc3_audit,
        )

    @property
    def mesh(self) -> UniformBZMesh:
        return self.three_phonon.mesh

    @property
    def primitive_volume_angstrom3(self) -> float:
        return abs(
            float(
                np.linalg.det(
                    self.three_phonon.fc2_mapping.primitive.lattice
                )
            )
        )

    def verify_source_and_object_binding(self) -> None:
        """Rebuild from source bytes and verify the in-memory physical object.

        This deliberately costs an additional FC2/FC3 read and audit.  It is
        used at the release-facing RTA boundary so a manually assembled
        wrapper cannot claim provenance merely by supplying hash-shaped text
        and preconstructed passing report objects.
        """

        current_paths = {
            name: _source_path(self.source_paths[name], name=name)
            for name in ("crystal", "fc2", "fc3")
        }
        current_hashes = {
            name: _sha256(path) for name, path in current_paths.items()
        }
        if any(
            not hmac.compare_digest(current_hashes[name], self.source_sha256[name])
            for name in current_hashes
        ):
            raise RuntimeError(
                "A bulk transport source no longer matches its recorded SHA256."
            )
        candidate = type(self).from_phono3py(
            current_paths["crystal"],
            current_paths["fc2"],
            current_paths["fc3"],
            mesh=self.three_phonon.mesh.shape,
            sigma_thz=self.three_phonon.sigma_thz,
            cutoff_frequency_thz=self.three_phonon.cutoff_frequency_thz,
            degenerate_tolerance_thz=(
                self.three_phonon.degenerate_tolerance_thz
            ),
            make_r0_average=self.three_phonon.fc3.make_r0_average,
            frequency_factor_to_thz=(
                self.three_phonon.frequency_factor_to_thz
            ),
        )
        from negfpy.anharmonic._bulk_identity import (
            bulk_three_phonon_physics_sha256,
        )

        if not hmac.compare_digest(
            bulk_three_phonon_physics_sha256(candidate.three_phonon),
            bulk_three_phonon_physics_sha256(self.three_phonon),
        ):
            raise ValueError(
                "BulkTransportProblem in-memory FC2/FC3 or controls do not "
                "match the recorded source files."
            )

    @property
    def harmonic(self) -> BulkHarmonicProblem:
        return BulkHarmonicProblem(
            fc2=self.three_phonon.fc2,
            mapping=self.three_phonon.fc2_mapping,
            source_paths={
                "crystal": self.source_paths["crystal"],
                "fc2": self.source_paths["fc2"],
            },
            source_sha256={
                "crystal": self.source_sha256["crystal"],
                "fc2": self.source_sha256["fc2"],
            },
            fc2_audit=self.fc2_audit,
            fc2_physical_audit=self.fc2_physical_audit,
            frequency_factor_to_thz=self.three_phonon.frequency_factor_to_thz,
        )


__all__ = ["BulkHarmonicProblem", "BulkTransportProblem"]
