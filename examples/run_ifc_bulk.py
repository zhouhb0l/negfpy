"""Run bulk IFC calculations from a JSON input file.

Usage examples:
  python examples/run_ifc_bulk.py --write-template examples/configs/ifc_bulk_template.json
  python examples/run_ifc_bulk.py --input examples/configs/ifc_bulk_template.json
"""

from negfpy.workflows.ifc_bulk import main


if __name__ == "__main__":
    main()
