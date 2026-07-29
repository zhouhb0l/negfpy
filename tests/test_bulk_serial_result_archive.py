from __future__ import annotations

import os
from pathlib import Path

import h5py
import pytest

from negfpy.anharmonic._bulk_result_archive import (
    BulkSerialResultArchiveError,
    read_bulk_serial_result_archive,
    write_bulk_serial_result_archive,
)
from negfpy.anharmonic.bulk_three_phonon import (
    compute_three_phonon_linewidths,
)
from test_anharmonic_bulk_target_q_mpi import (
    _assert_all_scientific_fields_equal,
    _problem,
)


def _result():
    return compute_three_phonon_linewidths(
        _problem(),
        temperatures=[0.0, 300.0],
        grid_points=[0, 1, 2],
        band_indices=[0, 2],
    )


def test_serial_result_archive_round_trip_and_exclusive_publication(
    tmp_path: Path,
) -> None:
    result = _result()
    target = tmp_path / "serial-result.hdf5"
    written = write_bulk_serial_result_archive(
        target,
        result,
        provenance={
            "campaign_manifest_sha256": "a" * 64,
            "source_revision": "b" * 40,
        },
    )
    loaded = read_bulk_serial_result_archive(target)

    assert written.result_sha256 == loaded.result_sha256
    assert dict(written.dataset_sha256) == dict(loaded.dataset_sha256)
    assert dict(loaded.provenance) == {
        "campaign_manifest_sha256": "a" * 64,
        "source_revision": "b" * 40,
    }
    _assert_all_scientific_fields_equal(loaded.result, result)
    with pytest.raises(FileExistsError):
        write_bulk_serial_result_archive(
            target,
            result,
            provenance={"campaign_manifest_sha256": "a" * 64},
        )


def test_serial_result_archive_rejects_dataset_tamper(
    tmp_path: Path,
) -> None:
    target = tmp_path / "serial-result.hdf5"
    write_bulk_serial_result_archive(
        target,
        _result(),
        provenance={"campaign_manifest_sha256": "a" * 64},
    )
    target.chmod(0o644)
    with h5py.File(target, "r+") as handle:
        handle["linewidths/gamma_thz_hwhm"][0, 0, 0] += 1.0

    with pytest.raises(
        BulkSerialResultArchiveError,
        match="invalid serial bulk result archive",
    ):
        read_bulk_serial_result_archive(target)


def test_serial_result_archive_rejects_symlink_and_nonfinite_provenance(
    tmp_path: Path,
) -> None:
    target = tmp_path / "serial-result.hdf5"
    result = _result()
    write_bulk_serial_result_archive(
        target,
        result,
        provenance={"campaign_manifest_sha256": "a" * 64},
    )
    link = tmp_path / "linked.hdf5"
    link.symlink_to(target)
    with pytest.raises(
        BulkSerialResultArchiveError,
        match="symbolic link",
    ):
        read_bulk_serial_result_archive(link)

    with pytest.raises(ValueError, match="Out of range float values"):
        write_bulk_serial_result_archive(
            tmp_path / "nonfinite.hdf5",
            result,
            provenance={"bad": float("nan")},
        )


def test_serial_result_archive_rejects_multiple_hard_links(
    tmp_path: Path,
) -> None:
    target = tmp_path / "serial-result.hdf5"
    write_bulk_serial_result_archive(
        target,
        _result(),
        provenance={"campaign_manifest_sha256": "a" * 64},
    )
    os.link(target, tmp_path / "duplicate.hdf5")

    with pytest.raises(
        BulkSerialResultArchiveError,
        match="multiple hard links",
    ):
        read_bulk_serial_result_archive(target)


def test_serial_result_archive_rejects_external_hdf5_link(
    tmp_path: Path,
) -> None:
    target = tmp_path / "serial-result.hdf5"
    write_bulk_serial_result_archive(
        target,
        _result(),
        provenance={"campaign_manifest_sha256": "a" * 64},
    )
    external = tmp_path / "external.hdf5"
    with h5py.File(target, "r") as source:
        values = source["linewidths/gamma_thz_hwhm"][...]
    with h5py.File(external, "w") as destination:
        destination.create_dataset("values", data=values)
    target.chmod(0o644)
    with h5py.File(target, "r+") as handle:
        del handle["linewidths/gamma_thz_hwhm"]
        handle["linewidths/gamma_thz_hwhm"] = h5py.ExternalLink(
            external.as_posix(),
            "/values",
        )

    with pytest.raises(
        BulkSerialResultArchiveError,
        match="invalid serial bulk result archive",
    ):
        read_bulk_serial_result_archive(target)
