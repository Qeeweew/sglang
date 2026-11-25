"""Block-Hadamard transform and per-token INT8 quantization on Ascend 910B."""

import torch


def h32_quant(x, interleaved, *, count=None):
    from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import (
        load_library,
    )
    from sglang.srt.runtime_context import get_buffer

    rows, width = x.shape

    def indices():
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("warm H32 before capture")
        offsets = torch.arange(width, dtype=torch.int32, device="cpu") * 4
        return torch.stack(
            (
                offsets.reshape(-1, 8, 32).transpose(1, 2).reshape(-1),
                offsets.reshape(-1, 32, 8).transpose(1, 2).reshape(-1),
            )
        ).to(x.device)

    perm = get_buffer(f"dsv41.h32.{x.device}.{width}", indices)
    codes = x.new_empty((rows, width), dtype=torch.int8)
    scale = x.new_empty(rows, dtype=torch.float32)
    load_library().h32_quant(
        x.contiguous(), perm, codes, scale, interleaved, count
    )
    return codes, scale
