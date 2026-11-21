"""Indexed row movement for contiguous 910B tensors."""


def gather_rows(source, row_map, *, out=None, fill=0):
    """Gather along dimension zero. Out-of-range indices produce fill rows."""
    from ..runtime.native import load_library

    return load_library().gather_rows(source, row_map, out, fill)


def scatter_rows(updates, row_map, out):
    """Skip out-of-range destinations. For a repeated destination, the last row wins."""
    from ..runtime.native import load_library

    return load_library().scatter_rows(updates, row_map, out)
