"""Compact expert metadata from independent CPU prefix arithmetic."""

import numpy as np


def reference(ends, capacity):
    ends = np.asarray(ends, dtype=np.int64)
    result = np.zeros((len(ends) + 1, 4), dtype=np.int32)
    result[0, 3] = 1
    counts = np.diff(np.r_[np.int64(0), ends])
    if np.any(counts < 0) or np.any(ends > capacity):
        result[0, 2] = 1
    else:
        ids = np.flatnonzero(counts)
        result[0, :2] = len(ids), ends[-1]
        result[1 : len(ids) + 1, 0] = ids
        result[1 : len(ids) + 1, 1] = ends[ids] - counts[ids]
        result[1 : len(ids) + 1, 2] = counts[ids]
    return result.ravel()
