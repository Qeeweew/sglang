"""910B Engram: compressed E4M3FN/E8M0 rows, software decode to BF16.

No model/framework registration. Keep tables alive for all outstanding launches
and for the entire lifetime of any graph that captures their addresses.
"""

from __future__ import annotations

from functools import cache, partial

import torch
import torch_npu  # noqa: F401
from sglang.srt.hardware_backend.npu.dsv41.kernels.paths import native_library_path
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library


class HostBuffer:
    """Tensor-owned compressed host storage registered by the extension.

    Allocation is startup work. Views retain the allocation and registration;
    close releases this owner's references after retiring asynchronous readers.
    """

    def __init__(self, shape, *, device=0, allocation="mmap"):
        if allocation not in ("mmap", "acl"):
            raise ValueError("allocation must be 'mmap' or 'acl'")
        self.device, self.shape = int(device), tuple(shape)
        self.allocation = allocation
        load_library()
        self._owner = torch.classes.dsv41.HostBuffer(
            list(shape), self.device, -1, allocation == "acl"
        )
        self.tensor = self._owner.tensor()
        self.nbytes = self.tensor.numel()

    def close(self):
        if self.tensor is not None:
            torch.npu.synchronize(self.device)
            self.tensor = None
            self._owner.close()

    @classmethod
    def from_shared_fd(cls, fd, shape, *, device=0):
        """Register a MAP_SHARED view; native ownership duplicates the descriptor."""
        result = cls.__new__(cls)
        result.device, result.shape = int(device), tuple(shape)
        result.allocation = "shared"
        load_library()
        result._owner = torch.classes.dsv41.HostBuffer.from_shared_fd(
            list(shape), result.device, int(fd)
        )
        result.tensor = result._owner.tensor()
        result.nbytes = result.tensor.numel()
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@cache
def _lookup_tables(device):
    # CPU FP8 decoding is only used to build the tiny 256-entry LUT at startup.
    # Model construction runs under a default NPU device context. These tiny
    # lookup tables are decoded on CPU because 910B has no native FP8 cast.
    fp8 = (
        torch.arange(256, dtype=torch.uint8, device="cpu")
        .view(torch.float8_e4m3fn)
        .float()
    )
    scale = torch.tensor(
        [2.0 ** (i - 127) for i in range(255)] + [float("nan")], device="cpu"
    )
    with torch.npu.device(device):
        lut = torch.cat([fp8, scale]).to(f"npu:{device}")
        # Startup only; permits later use on arbitrary caller streams.
        torch.npu.synchronize()
    return lut


class EngramTable:
    """Byte table ``[vocab,width]`` with one E8M0 exponent per32 codes.

    Construct with HostBuffers for host residency, or raw-byte NPU tensors for
    an HBM control. This operator only gathers and decodes; hashing/gating remain
    separate. Invalid IDs (negative or >= vocab) return an all-zero row.
    """

    def __init__(self, codes, scales, *, library=None, device=0):
        self.codes, self.scales = codes, scales
        self.device = int(device)
        self.library = load_library(native_library_path(library))
        c = codes.tensor if isinstance(codes, HostBuffer) else codes
        s = scales.tensor if isinstance(scales, HostBuffer) else scales
        if c is None or s is None or c.dtype != torch.uint8 or s.dtype != torch.uint8:
            raise ValueError("codes and scales must be live raw-byte buffers")
        if (
            c.ndim != 2
            or c.shape[1] <= 0
            or c.shape[1] % 32
            or s.shape != (c.shape[0], c.shape[1] // 32)
        ):
            raise ValueError("expected codes[vocab,width], scales[vocab,width/32]")
        if not c.is_contiguous() or not s.is_contiguous():
            raise ValueError("codes and scales must be contiguous")
        for value in (codes, scales):
            if isinstance(value, HostBuffer):
                if value.device != self.device:
                    raise ValueError("host mapping belongs to another NPU")
            elif value.device.type != "npu" or value.device.index != self.device:
                raise ValueError(
                    "use HostBuffer for CPU data, or tensors on the selected NPU"
                )
        self.vocab = c.shape[0]
        self.width = c.shape[1]
        self.lut = _lookup_tables(self.device)

    @staticmethod
    def _tensor(value):
        if isinstance(value, HostBuffer):
            if value.tensor is None:
                raise RuntimeError("host buffer has already been closed")
            return value.tensor
        return value

    def bind(self, ids, *, out):
        """Validate and prepare before capture; retain every table allocation."""
        args = (
            self._tensor(self.codes),
            self._tensor(self.scales),
            ids,
            out,
            self.lut,
        )
        self.library.prepare_engram(*args)
        return partial(self.library.engram, *args)

    def gather(self, ids, out=None):
        out = (
            ids.new_empty((*ids.shape, self.width), dtype=torch.bfloat16)
            if out is None
            else out
        )
        self.library.engram(
            self._tensor(self.codes),
            self._tensor(self.scales),
            ids,
            out,
            self.lut,
        )
        return out
