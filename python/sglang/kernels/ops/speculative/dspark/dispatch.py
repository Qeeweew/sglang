from __future__ import annotations

import torch


def inputs_on_cuda(*args, **kwargs) -> bool:
    """Route kernel dispatch by input placement: the first tensor argument
    decides. CUDA inputs take the fused triton kernel; CPU inputs take the
    torch reference implementation (triton is CUDA-only, and CPU-side callers
    such as unit tests exercise the reference path)."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            if value.device.type == "npu":
                from sglang.srt.hardware_backend.npu.dsv41.policy import enabled

                # transfer_to_npu reports is_cuda=True. The native 910B path
                # uses the device Torch implementations for these small helpers;
                # its specialized proposal/schedule ops select their own kernels.
                if enabled():
                    return False
            return value.is_cuda
    raise AssertionError("kernel dispatch requires at least one tensor argument")
