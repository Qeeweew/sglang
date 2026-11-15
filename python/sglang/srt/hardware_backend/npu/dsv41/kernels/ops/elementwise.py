"""DeepJIT Ascend C RMSNorm and FP32-arithmetic clipped SwiGLU."""

import math
from functools import partial

import torch
from sglang.srt.hardware_backend.npu.dsv41.kernels.config import V41
from sglang.srt.hardware_backend.npu.dsv41.kernels.runtime.native import load_library

# arch2201 Vector repeat is 256 bytes; a two-level r64 tree covers one
# panel of VECTOR_ELEMENTS**2 FP32 elements. Native binding rechecks the budget.
_FP32_BYTES = torch.finfo(torch.float32).bits // 8
_VECTOR_ELEMENTS = 256 // _FP32_BYTES
_REDUCTION_PANEL_ELEMENTS = _VECTOR_ELEMENTS**2


class RMSNorm:
    def __init__(self, *, dimension=V41.hidden, epsilon=1e-20):
        if (
            dimension <= 0
            or dimension > 2 * _REDUCTION_PANEL_ELEMENTS
            or dimension % _VECTOR_ELEMENTS
            or (
                dimension > _REDUCTION_PANEL_ELEMENTS
                and dimension % (2 * _VECTOR_ELEMENTS)
            )
        ):
            raise ValueError(
                "RMSNorm dimension requires complete arch22 reduction panels"
            )
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("RMSNorm epsilon must be positive and finite")
        self.d, self.epsilon = dimension, float(epsilon)
        library = load_library()
        self._native, self._prepare = library.rms_norm, library.prepare_rms_norm
        self._prepared_shapes = set()

    def bind(self, x, weight, *, out):
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError("RMSNorm input width differs from dimension")
        key = (x.device, x.shape[0])
        if key not in self._prepared_shapes:
            self._prepare(x, weight, out, self.epsilon)
            self._prepared_shapes.add(key)
        return partial(self._native, x, weight, out, self.epsilon)

    def __call__(self, x, weight, *, out=None):
        out = torch.empty_like(x) if out is None else out
        self.bind(x, weight, out=out)()
        return out


class ClippedSwiGLU:
    def __init__(self, *, dimension=V41.intermediate, limit=10.0, dtype="bf16"):
        if dimension <= 0 or dimension % 32:
            raise ValueError(
                "SwiGLU requires a positive multiple-of-32 dimension fitting the row UB budget"
            )
        if dtype not in ("bf16", "f16") or not math.isfinite(limit) or limit <= 0:
            raise ValueError("SwiGLU requires BF16/FP16 and a positive finite clamp")
        self.d, self.limit, self.dtype = dimension, float(limit), dtype
        library = load_library()
        self._native, self._prepare = library.swiglu, library.prepare_swiglu
        self._prepared_shapes = set()

    def _dtype(self, x):
        expected = torch.bfloat16 if self.dtype == "bf16" else torch.float16
        if x.dtype != expected:
            raise ValueError(f"SwiGLU requires {expected}")

    def bind(self, gate, up, *, out):
        self._dtype(gate)
        if gate.ndim != 2 or gate.shape[1] != self.d:
            raise ValueError("SwiGLU input width differs from dimension")
        key = (gate.device, gate.shape[0])
        if key not in self._prepared_shapes:
            self._prepare(gate, up, None, out, self.limit, False)
            self._prepared_shapes.add(key)
        return partial(self._native, gate, up, None, out, self.limit, False)

    def __call__(self, gate, up, *, out=None):
        out = torch.empty_like(gate) if out is None else out
        self.bind(gate, up, out=out)()
        return out


class PackedClippedSwiGLU(ClippedSwiGLU):
    def __init__(
        self,
        *,
        dimension=V41.intermediate,
        limit=10.0,
        dtype="bf16",
        prefix=False,
    ):
        super().__init__(dimension=dimension, limit=limit, dtype=dtype)
        self.prefix = bool(prefix)

    def bind(self, gate_up, *, out, rows=None):
        self._dtype(gate_up)
        if gate_up.ndim != 2 or gate_up.shape[1] != 2 * self.d:
            raise ValueError("packed SwiGLU input width differs from twice dimension")
        if self.prefix != (rows is not None):
            raise ValueError("device row count is required exactly for prefix SwiGLU")
        key = (gate_up.device, gate_up.shape[0])
        if key not in self._prepared_shapes:
            self._prepare(gate_up, gate_up, rows, out, self.limit, True)
            self._prepared_shapes.add(key)
        return partial(self._native, gate_up, gate_up, rows, out, self.limit, True)

    def __call__(self, gate_up, *, out=None, rows=None):
        out = gate_up.new_empty((gate_up.shape[0], self.d)) if out is None else out
        self.bind(gate_up, out=out, rows=rows)()
        return out
