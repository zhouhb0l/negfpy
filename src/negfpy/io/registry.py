"""Reader registry for IFC import adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from negfpy.modeling.schema import IFCData


Reader = Callable[[Any], IFCData]
_READERS: dict[str, Reader] = {}


def register_reader(name: str, reader: Reader) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Reader name must be non-empty.")
    _READERS[key] = reader


def get_reader(name: str) -> Reader:
    key = name.strip().lower()
    try:
        return _READERS[key]
    except KeyError as exc:
        available = ", ".join(sorted(_READERS)) or "<none>"
        raise KeyError(f"Unknown IFC reader '{name}'. Available readers: {available}") from exc


def list_readers() -> tuple[str, ...]:
    return tuple(sorted(_READERS.keys()))


def read_ifc(source: Any, reader: str) -> IFCData:
    return get_reader(reader)(source)

