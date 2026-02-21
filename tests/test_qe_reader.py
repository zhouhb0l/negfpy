from pathlib import Path

import numpy as np

from negfpy.io import list_readers, read_ifc
from negfpy.modeling import qe_omega_to_cm1, qe_omega_to_thz


def test_qe_reader_registered() -> None:
    assert "qe" in list_readers()


def test_qe_reader_parses_minimal_fc(tmp_path: Path) -> None:
    fc_text = "\n".join(
        [
            " 1 1 1 1.0 0.0 0.0 0.0 0.0 0.0",
            " 1 'X ' 10.0",
            " 1 1 0.0 0.0 0.0",
            " 1 1 1",
            " 1 1 1 1",
            " 1 1 1 1.0",
            " 1 2 1 1",
            " 1 1 1 0.0",
            " 1 3 1 1",
            " 1 1 1 0.0",
            " 2 1 1 1",
            " 1 1 1 0.0",
            " 2 2 1 1",
            " 1 1 1 2.0",
            " 2 3 1 1",
            " 1 1 1 0.0",
            " 3 1 1 1",
            " 1 1 1 0.0",
            " 3 2 1 1",
            " 1 1 1 0.0",
            " 3 3 1 1",
            " 1 1 1 3.0",
        ]
    )
    fc_path = tmp_path / "mini.fc"
    fc_path.write_text(fc_text, encoding="utf-8")

    ifc = read_ifc(fc_path, reader="qe")
    assert ifc.dof_per_atom == 3
    assert ifc.masses.shape == (1,)
    assert len(ifc.terms) == 1
    term = ifc.terms[0]
    assert (term.dx, term.dy, term.dz) == (0, 0, 0)
    assert term.block.shape == (3, 3)
    assert np.allclose(np.diag(term.block).real, [1.0, 2.0, 3.0])


def test_qe_reader_parses_si444_if_present() -> None:
    fc_path = Path("si444.fc")
    if not fc_path.exists():
        return
    ifc = read_ifc(fc_path, reader="qe")
    assert ifc.dof_per_atom == 3
    assert ifc.masses.shape == (2,)
    assert len(ifc.terms) > 1
    max_abs_dx = max(abs(t.dx) for t in ifc.terms)
    max_abs_dy = max(abs(t.dy) for t in ifc.terms)
    max_abs_dz = max(abs(t.dz) for t in ifc.terms)
    assert max_abs_dx == 2
    assert max_abs_dy == 2
    assert max_abs_dz == 2


def test_qe_frequency_conversion_factors() -> None:
    thz = float(qe_omega_to_thz(1.0))
    cm1 = float(qe_omega_to_cm1(1.0))
    assert np.isclose(thz, 3289.8419602568943, rtol=0.0, atol=1e-9)
    assert np.isclose(cm1, 109737.31568180058, rtol=0.0, atol=1e-6)
