from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from negfpy.anharmonic.io.phono3py_crystal import (
    read_phono3py_crystal_mapping,
)


class _FakeCell:
    def __init__(
        self,
        lattice: np.ndarray,
        scaled_positions: np.ndarray,
        masses: np.ndarray,
        symbols: list[str],
    ) -> None:
        self.cell = np.array(lattice, dtype=float)
        self.scaled_positions = np.array(scaled_positions, dtype=float)
        self.positions = self.scaled_positions @ self.cell
        self.masses = np.array(masses, dtype=float)
        self.symbols = list(symbols)


class _FakePrimitive(_FakeCell):
    def __init__(
        self,
        *,
        lattice: np.ndarray,
        scaled_positions: np.ndarray,
        masses: np.ndarray,
        symbols: list[str],
        p2s: np.ndarray,
        s2p: np.ndarray,
        p2p: dict[int, int],
        shortest_vectors: np.ndarray,
        multiplicity: np.ndarray,
    ) -> None:
        super().__init__(lattice, scaled_positions, masses, symbols)
        self.p2s_map = np.array(p2s, dtype=np.int64)
        self.s2p_map = np.array(s2p, dtype=np.int64)
        self.p2p_map = dict(p2p)
        n_supercell = len(s2p)
        self.atomic_permutations = np.array(
            [
                np.roll(np.arange(n_supercell, dtype=np.int64), -shift)
                for shift in range(n_supercell)
            ],
            dtype=np.int64,
        )
        self._shortest_vectors = np.array(shortest_vectors, dtype=float)
        self._multiplicity = np.array(multiplicity, dtype=np.int64)

    def get_smallest_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        return self._shortest_vectors, self._multiplicity


class _FakeSymmetry:
    def __init__(self, atomic_permutations: np.ndarray) -> None:
        self.atomic_permutations = np.array(
            atomic_permutations,
            dtype=np.int64,
        )


class _FakePhono3py:
    def __init__(self) -> None:
        primitive_lattice = np.diag([4.0, 5.0, 6.0])
        primitive_positions = np.array([[0.25, 0.5, 0.75]])

        ordinary_multiplicity = np.empty((2, 1, 2), dtype=np.int64)
        ordinary_multiplicity[..., 0] = 1
        ordinary_multiplicity[..., 1] = np.arange(2).reshape(2, 1)
        self.primitive = _FakePrimitive(
            lattice=primitive_lattice,
            scaled_positions=primitive_positions,
            masses=np.array([28.085]),
            symbols=["Si"],
            p2s=np.array([0]),
            s2p=np.array([0, 0]),
            p2p={0: 0},
            shortest_vectors=np.arange(6, dtype=float).reshape(2, 3),
            multiplicity=ordinary_multiplicity,
        )
        self.supercell = _FakeCell(
            np.diag([8.0, 5.0, 6.0]),
            np.array([[0.25, 0.5, 0.75], [0.75, 0.5, 0.75]]),
            np.array([28.085, 28.085]),
            ["Si", "Si"],
        )
        self.symmetry = _FakeSymmetry([[0, 1], [1, 0]])

        phonon_multiplicity = np.empty((3, 1, 2), dtype=np.int64)
        phonon_multiplicity[..., 0] = 1
        phonon_multiplicity[..., 1] = np.arange(3).reshape(3, 1)
        self.phonon_primitive = _FakePrimitive(
            lattice=primitive_lattice,
            scaled_positions=primitive_positions,
            masses=np.array([28.085]),
            symbols=["Si"],
            p2s=np.array([1]),
            s2p=np.array([1, 1, 1]),
            p2p={1: 0},
            shortest_vectors=np.arange(9, dtype=float).reshape(3, 3),
            multiplicity=phonon_multiplicity,
        )
        self.phonon_supercell = _FakeCell(
            np.diag([12.0, 5.0, 6.0]),
            np.array(
                [
                    [0.0, 0.5, 0.75],
                    [0.25, 0.5, 0.75],
                    [0.75, 0.5, 0.75],
                ]
            ),
            np.array([28.085, 28.085, 28.085]),
            ["Si", "Si", "Si"],
        )
        self.phonon_supercell_symmetry = _FakeSymmetry(
            [[0, 1, 2], [2, 0, 1]]
        )

        self.primitive_matrix = np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        )
        self.supercell_matrix = np.diag([2, 1, 1])
        self.phonon_supercell_matrix = np.diag([3, 1, 1])
        self.version = "4.4.0-fake"


class _RecordingLoader:
    def __init__(self, value: Any) -> None:
        self.value = value
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        return self.value


def _yaml_path(tmp_path: Path) -> Path:
    path = tmp_path / "phono3py_disp.yaml"
    path.write_text("synthetic crystal mapping fixture\n", encoding="utf-8")
    return path


def test_reader_selects_fc3_structures_and_detaches_all_values(
    tmp_path: Path,
) -> None:
    path = _yaml_path(tmp_path)
    source = _FakePhono3py()
    loader = _RecordingLoader(source)

    mapping = read_phono3py_crystal_mapping(
        path,
        purpose="fc3",
        loader=loader,
    )

    assert loader.calls == [
        {
            "phono3py_yaml": path.resolve(),
            "produce_fc": False,
            "is_nac": False,
            "log_level": 0,
        }
    ]
    assert mapping.purpose == "fc3"
    assert mapping.n_primitive_atoms == 1
    assert mapping.n_supercell_atoms == 2
    np.testing.assert_array_equal(mapping.supercell_matrix, np.diag([2, 1, 1]))
    np.testing.assert_array_equal(mapping.p2s, [0])
    np.testing.assert_array_equal(mapping.s2p, [0, 0])
    np.testing.assert_array_equal(mapping.p2p, [0, -1])
    np.testing.assert_array_equal(
        mapping.atomic_permutations,
        [[0, 1], [1, 0]],
    )
    np.testing.assert_array_equal(
        mapping.translation_permutations,
        [[0, 1], [1, 0]],
    )
    assert mapping.provenance["structure_source"] == "ordinary"
    assert mapping.provenance["produce_fc"] is False
    assert mapping.provenance["is_nac"] is False
    assert mapping.provenance["phono3py_version"] == "4.4.0-fake"
    assert mapping.provenance["source_sha256"] == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    source.primitive.p2s_map[...] = 1
    source.primitive.s2p_map[...] = 1
    source.primitive._shortest_vectors[...] = -1
    source.supercell.cell[...] = -1
    source.symmetry.atomic_permutations[...] = -1
    np.testing.assert_array_equal(mapping.p2s, [0])
    np.testing.assert_array_equal(mapping.s2p, [0, 0])
    np.testing.assert_array_equal(
        mapping.atomic_permutations,
        [[0, 1], [1, 0]],
    )
    assert np.all(mapping.shortest_vectors >= 0)
    assert np.all(mapping.supercell.lattice >= 0)


def test_reader_selects_fc2_phonon_structures_and_explicit_nac(
    tmp_path: Path,
) -> None:
    path = _yaml_path(tmp_path)
    source = _FakePhono3py()
    loader = _RecordingLoader(source)

    mapping = read_phono3py_crystal_mapping(
        path,
        purpose="fc2",
        is_nac=True,
        loader=loader,
    )

    assert loader.calls[0]["produce_fc"] is False
    assert loader.calls[0]["is_nac"] is True
    assert mapping.purpose == "fc2"
    assert mapping.n_supercell_atoms == 3
    np.testing.assert_array_equal(mapping.supercell_matrix, np.diag([3, 1, 1]))
    np.testing.assert_array_equal(mapping.p2s, [1])
    np.testing.assert_array_equal(mapping.s2p, [1, 1, 1])
    np.testing.assert_array_equal(mapping.p2p, [-1, 0, -1])
    np.testing.assert_array_equal(
        mapping.atomic_permutations,
        [[0, 1, 2], [2, 0, 1]],
    )
    np.testing.assert_array_equal(
        mapping.translation_permutations,
        [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
    )
    assert mapping.provenance["structure_source"] == "phonon"
    assert (
        mapping.provenance["supercell_matrix_source"]
        == "phonon_supercell_matrix"
    )


def test_fc2_none_matrix_uses_verified_same_supercell_contract(
    tmp_path: Path,
) -> None:
    source = _FakePhono3py()
    source.phonon_supercell_matrix = None
    source.phonon_supercell = source.supercell
    source.phonon_primitive = source.primitive
    source.phonon_supercell_symmetry = source.symmetry

    mapping = read_phono3py_crystal_mapping(
        _yaml_path(tmp_path),
        purpose="fc2",
        loader=_RecordingLoader(source),
    )

    np.testing.assert_array_equal(mapping.supercell_matrix, np.diag([2, 1, 1]))
    assert (
        mapping.provenance["supercell_matrix_source"]
        == "verified_fc3_supercell_matrix_fallback"
    )


def test_fc2_none_matrix_rejects_unverified_different_phonon_supercell(
    tmp_path: Path,
) -> None:
    source = _FakePhono3py()
    source.phonon_supercell_matrix = None

    with pytest.raises(ValueError, match="phonon_supercell_matrix is None"):
        read_phono3py_crystal_mapping(
            _yaml_path(tmp_path),
            purpose="fc2",
            loader=_RecordingLoader(source),
        )


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("phonon_primitive", "phonon_primitive"),
        ("phonon_supercell", "phonon_supercell"),
        ("phonon_supercell_symmetry", "phonon_supercell_symmetry"),
    ],
)
def test_reader_rejects_missing_fc2_structures(
    tmp_path: Path,
    attribute: str,
    message: str,
) -> None:
    source = _FakePhono3py()
    setattr(source, attribute, None)

    with pytest.raises(ValueError, match=message):
        read_phono3py_crystal_mapping(
            _yaml_path(tmp_path),
            purpose="fc2",
            loader=_RecordingLoader(source),
        )


def test_reader_rejects_incomplete_sparse_p2p_without_repair(
    tmp_path: Path,
) -> None:
    source = _FakePhono3py()
    source.phonon_primitive.p2p_map = {}

    with pytest.raises(ValueError, match="p2p_map keys"):
        read_phono3py_crystal_mapping(
            _yaml_path(tmp_path),
            purpose="fc2",
            loader=_RecordingLoader(source),
        )


@pytest.mark.parametrize("field", ["masses", "symbols"])
def test_reader_rejects_primitive_representative_conflicts(
    tmp_path: Path,
    field: str,
) -> None:
    source = _FakePhono3py()
    if field == "masses":
        source.supercell.masses[0] = 99.0
    else:
        source.supercell.symbols[0] = "Ge"

    with pytest.raises(ValueError, match=f"representative {field}"):
        read_phono3py_crystal_mapping(
            _yaml_path(tmp_path),
            purpose="fc3",
            loader=_RecordingLoader(source),
        )


def test_reader_rejects_invalid_purpose_before_calling_loader(
    tmp_path: Path,
) -> None:
    loader = _RecordingLoader(_FakePhono3py())

    with pytest.raises(ValueError, match="purpose"):
        read_phono3py_crystal_mapping(
            _yaml_path(tmp_path),
            purpose="fc4",  # type: ignore[arg-type]
            loader=loader,
        )

    assert loader.calls == []


def test_reader_rejects_missing_source_before_calling_loader(
    tmp_path: Path,
) -> None:
    loader = _RecordingLoader(_FakePhono3py())

    with pytest.raises(FileNotFoundError):
        read_phono3py_crystal_mapping(
            tmp_path / "missing.yaml",
            purpose="fc3",
            loader=loader,
        )

    assert loader.calls == []
