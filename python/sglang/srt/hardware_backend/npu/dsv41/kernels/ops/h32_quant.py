"""Block-Hadamard transform and per-token INT8 quantization on Ascend 910B."""

import torch


def h32_quant(x, interleaved, *, count=None):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )

    rows, width = x.shape
    codes = x.new_empty((rows, width), dtype=torch.int8)
    scale = x.new_empty(rows, dtype=torch.float32)
    load_library().h32_quant(x.contiguous(), codes, scale, interleaved, count)
    return codes, scale
