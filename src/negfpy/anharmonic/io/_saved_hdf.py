"""Require self-contained, bounded HDF5 storage before payload access."""

def require_internal_storage(handle, max_objects=10000):
    import h5py
    seen = set()
    pending = [handle]
    count = 0
    while pending:
        group = pending.pop()
        token = h5py.h5o.get_info(group.id).addr
        if token in seen:
            continue
        seen.add(token)
        for name in group:
            count += 1
            if count > max_objects:
                raise ValueError("Saved HDF5 object count exceeds its bounded input schema")
            if not isinstance(group.get(name,getlink=True),h5py.HardLink):
                raise ValueError("Saved HDF5 inputs require internal hard links")
            obj = group[name]
            if isinstance(obj,h5py.Group):
                pending.append(obj)
            elif not isinstance(obj,h5py.Dataset) or obj.is_virtual or obj.external:
                raise ValueError("Saved HDF5 inputs require internal non-virtual dataset storage")
