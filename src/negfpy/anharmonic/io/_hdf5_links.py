"""Internal HDF5 link-safety helpers for provenance-bearing readers."""

from __future__ import annotations

from typing import Any


def resolve_hard_link_chain(
    handle: Any,
    name: str,
    h5py: Any,
    *,
    required: bool,
) -> Any | None:
    """Resolve an HDF5 path only when every component is an internal hard link."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("HDF5 object name must be a non-empty string.")
    components = tuple(component for component in name.split("/") if component)
    if not components or any(component in {".", ".."} for component in components):
        raise ValueError(f"HDF5 object path {name!r} is not canonical.")

    current = handle
    traversed: list[str] = []
    for position, component in enumerate(components):
        traversed.append(component)
        link = current.get(component, getlink=True)
        if link is None:
            if required:
                raise KeyError(f"HDF5 object {name!r} was not found.")
            return None
        if not isinstance(link, h5py.HardLink):
            path = "/".join(traversed)
            raise ValueError(  # noqa: TRY004 - malformed HDF5 file content
                f"HDF5 object {name!r} must use an internal hard-link chain; "
                f"component {path!r} is a {type(link).__name__}."
            )
        obj = current[component]
        if position < len(components) - 1:
            if not isinstance(obj, h5py.Group):
                path = "/".join(traversed)
                raise ValueError(
                    f"HDF5 object path {name!r} traverses non-group component "
                    f"{path!r}."
                )
            current = obj
            continue
        return obj
    raise AssertionError("Non-empty HDF5 paths always return from the traversal.")


__all__ = ["resolve_hard_link_chain"]
