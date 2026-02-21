from negfpy.io.readers import read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.io.registry import get_reader, list_readers, read_ifc, register_reader


register_reader("phonopy", read_phonopy_ifc)
register_reader("qe", read_qe_q2r_ifc)

__all__ = ["register_reader", "get_reader", "list_readers", "read_ifc", "read_phonopy_ifc", "read_qe_q2r_ifc"]
