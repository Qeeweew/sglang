"""Independent signed nibble / group32 reference (no device packer imports)."""

import numpy as np


def fixture(n, k, *, seed=41):
    rng = np.random.default_rng(seed)
    q = rng.integers(-8, 8, (n, k), dtype=np.int8)
    codes = ((q[:, 0::2] & 15) | ((q[:, 1::2] & 15) << 4)).astype(np.uint8)
    scales = rng.uniform(0.0005, 0.04, (n, k // 32)).astype(np.float16)
    scales[0, :4] = [0, np.float16(2**-24), np.float16(2**-16), 0.03][: min(4, k // 32)]
    return codes, scales


def weight(codes, scales):
    n, half_k = codes.shape
    q = np.empty((n, half_k * 2), dtype=np.int16)
    q[:, 0::2] = codes & 15
    q[:, 1::2] = codes >> 4
    q[q >= 8] -= 16
    # The device rounds each dequantized weight to FP16 before the Cube.
    return (
        q.astype(np.float64) * np.repeat(scales.astype(np.float64), 32, axis=1)
    ).astype(np.float16)


def check(actual, x, w):
    """FP64 GEMM oracle over independently decoded FP16 weights."""
    rows = np.arange(len(x))
    if len(x) * w.size > 100_000_000:
        rows = np.unique(np.linspace(0, len(x) - 1, 17, dtype=int))
    expected = (x[rows].astype(np.float64) @ w.astype(np.float64).T).astype(np.float16)
    actual = np.asarray(actual)[rows]
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=0.002, atol=0.002)
    difference = actual.astype(np.float32) - expected.astype(np.float32)
    return dict(
        checked_rows=len(rows),
        full_output=len(rows) == len(x),
        max_abs_error=float(np.max(np.abs(difference), initial=0)),
        relative_l2=float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(expected.astype(np.float32)), 1e-20)
        ),
    )
