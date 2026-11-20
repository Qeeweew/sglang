"""Common ND tensor, alias and device contracts for native 910B operators."""

from __future__ import annotations

import json
import os
from threading import Lock

import torch
import torch_npu

from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.toolchain import (
    validate_jit_toolchain,
)


def npu_device(device):
    result = torch.device(device)
    if result.type != "npu":
        raise ValueError("requires NPU")
    return (
        result
        if result.index is not None
        else torch.device("npu", torch.npu.current_device())
    )


def check_tensor(value, dtype, shape, device, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.device != device
        or value.dtype != dtype
        or not value.is_contiguous()
        or (shape is not None and tuple(value.shape) != tuple(shape))
    ):
        raise ValueError(f"{name} requires contiguous {dtype} {shape} on {device}")
    if torch_npu.get_npu_format(value) not in (0, 2):
        raise ValueError(f"{name} requires ND storage (format 0 or 2)")


def overlaps(first, second):
    if first is None or second is None or not first.numel() or not second.numel():
        return False
    return max(first.data_ptr(), second.data_ptr()) < min(
        first.data_ptr() + first.numel() * first.element_size(),
        second.data_ptr() + second.numel() * second.element_size(),
    )


def require_disjoint(outputs, inputs=()):
    seen = [tensor for tensor in inputs if tensor is not None]
    for output in outputs:
        if output is not None:
            if any(overlaps(output, other) for other in seen):
                raise ValueError(
                    "outputs/workspace must not overlap inputs or each other"
                )
            seen.append(output)


def record_inputs(owner, args, stream, *, attribute="_last"):
    """Retain capture-time tensors; the extension records eager stream use."""
    if torch.npu.is_current_stream_capturing():
        tensors = tuple(value for value in args if isinstance(value, torch.Tensor))
        setattr(owner, attribute, tensors)
    else:
        setattr(owner, attribute, ())


_loaded_path = None
_load_lock = Lock()


def load_library(path=None):
    """Load one CMake-built dispatcher extension, without a Python FFI.

    Torch registrations are process-global. Publication updates affect the next
    process; reject a second version before it can duplicate registrations.
    """
    global _loaded_path
    with _load_lock:
        if path is None and _loaded_path is not None:
            return torch.ops.dsv41
        path = native_library_path(path)
        if _loaded_path is None:
            publication = json.loads((path.parent / "manifest.json").read_text())
            validate_jit_toolchain(publication)
            torch.ops.load_library(str(path))
            abi = torch.ops.dsv41.abi()
            if abi != 28:
                raise RuntimeError(
                    f"incompatible Torch-NPU extension ABI {abi} (expected 28): {path}; rebuild"
                )
            torch.ops.dsv41.configure_jit(
                publication["kernel_source_root"],
                publication["kernel_signature"],
                os.environ.get(
                    "DSV41_JIT_CACHE_DIR", str(path.parents[2] / "jit-cache")
                ),
            )
            _loaded_path = path
        elif path != _loaded_path:
            raise RuntimeError(
                f"extension already loaded from {_loaded_path}; restart to use {path}"
            )
    return torch.ops.dsv41


class NativeOperator:
    """Device ownership and common validation, independent of any operator family."""

    def __init__(self, device="npu:0"):
        self.device = npu_device(device)
        self.cores = torch.npu.get_device_properties(self.device).vector_core_num

    def _check(self, value, dtype, shape=None, name="tensor"):
        check_tensor(value, dtype, shape, self.device, name)
