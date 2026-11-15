"""Independent checkpoint E2M1 formula; no rank packing or bit correction."""

import numpy as np


def weight(codes, exponents, k):
    codes, exponents = np.asarray(codes), np.asarray(exponents)
    nibbles = np.empty((codes.shape[0], codes.shape[1] * 2), dtype=np.uint8)
    nibbles[:, 0::2], nibbles[:, 1::2] = codes & 15, codes >> 4
    nibbles = nibbles[:, :k]
    magnitude = nibbles & 7
    exponent, mantissa = magnitude // 2, magnitude % 2
    values = np.where(
        exponent == 0,
        mantissa / 2,
        np.exp2(exponent.astype(np.int32) - 1) * (1 + mantissa / 2),
    )
    values = np.where(nibbles & 8, -values, values)
    scales = np.exp2(exponents.astype(np.int32) - 127).repeat(32, axis=1)[:, :k]
    return (values * scales).astype(np.float16)


def grouped(x, weights, ends):
    result = np.zeros((len(x), weights[0].shape[0]), dtype=np.float16)
    begin = 0
    for w, end in zip(weights, ends):
        result[begin:end] = (
            x[begin:end].astype(np.float64) @ w.astype(np.float64).T
        ).astype(np.float16)
        begin = end
    return result


def fixture(e, n, k, seed=17):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, (e, n, k // 2), dtype=np.uint8)
    scales = rng.integers(123, 127, (e, n, k // 32), dtype=np.uint8)
    # Exercise all sixteen nibbles, both signs and negative zero in every expert.
    codes[:, 0, :8] = np.arange(0, 16, 2, dtype=np.uint8) | (
        np.arange(1, 16, 2, dtype=np.uint8) << 4
    )
    return codes, scales


def checkpoint(root, ids, projection="gate_up", layer=0):
    """Map only the requested raw expert bytes; no model/framework loader."""
    import json
    import struct
    from pathlib import Path

    root = Path(root)
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    headers = {}

    def read(name):
        path = root / index[name]
        if path not in headers:
            with path.open("rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                headers[path] = (size + 8, json.loads(stream.read(size)))
        base, header = headers[path]
        item = header[name]
        assert item["dtype"] in {"I8", "U8", "F8_E8M0"}, item
        return np.memmap(
            path,
            mode="r",
            dtype=np.uint8,
            offset=base + item["data_offsets"][0],
            shape=tuple(item["shape"]),
        )

    codes, scales = [], []
    for expert in ids:
        prefix = f"layers.{layer}.ffn.experts.{expert}"
        if projection == "gate_up":
            codes.append(
                np.concatenate([read(f"{prefix}.{w}.weight") for w in ("w1", "w3")])
            )
            scales.append(
                np.concatenate([read(f"{prefix}.{w}.scale") for w in ("w1", "w3")])
            )
        elif projection == "down":
            codes.append(read(f"{prefix}.w2.weight"))
            scales.append(read(f"{prefix}.w2.scale"))
        else:
            raise ValueError("checkpoint projection must be gate_up or down")
    return codes, scales
