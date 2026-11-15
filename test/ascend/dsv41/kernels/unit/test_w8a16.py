"""Weight storage and the numerical boundaries exposed by W8A16."""

import torch
import pytest

from sglang.srt.hardware_backend.npu.dsv41.kernels.ops.w8a16 import repack_weight
from dsv41.kernels.references.w8a16 import dequantize


def test_repack_preserves_codes_and_group_column_scales():
    n, k = 128, 768
    weight = torch.arange(n * k).remainder(256).sub(128).to(torch.int8).reshape(n, k)
    scales = torch.arange(1, 3 * n + 1).reshape(3, n).mul(2**-16).half()
    packed, factors = repack_weight(weight, scales)
    assert packed.numel() == weight.numel() and packed.dtype == torch.int8
    for row in (0, 15, 16, 47, 127):
        assert torch.equal(packed[row // 16, :, row % 16], weight[row])
        assert torch.equal(factors[row // 16, :, row % 16], scales[:, row])


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), 512.0])
def test_repack_rejects_invalid_scales(bad):
    with pytest.raises(ValueError, match="scales"):
        repack_weight(
            torch.zeros(128, 256, dtype=torch.int8), torch.full((1, 128), bad).half()
        )


def test_dequantize_retains_fp16_rounding_before_bf16():
    weight = torch.full((128, 256), 3, dtype=torch.int8)
    scales = torch.full((1, 128), 4.178285598754883e-05, dtype=torch.float16)
    expected = 3 * float(scales[0, 0])
    actual = dequantize(weight, scales, torch.bfloat16)
    assert actual[0, 0] == torch.tensor(expected).half().bfloat16()
    assert actual[0, 0] != torch.tensor(expected).bfloat16()
