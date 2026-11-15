"""Matched compressed INT4/MXFP4 bytes and independent decoded references."""

import numpy as np


def matched_weights(n, k):
    from dsv41.kernels.references.quant_packing import pack_experts
    from dsv41.kernels.references.quant_packing import pack_int4_group32
    from dsv41.kernels.references.int4 import weight as int4_weight
    from dsv41.kernels.references.mxfp4_grouped import fixture, weight

    codes, exponents = fixture(1, n, k)
    packed, scales = pack_experts(zip(codes, exponents), k=k, n=n, count=1)
    low, high = codes[0] & 15, codes[0] >> 4
    low = np.where(low & 8, -(low & 7).astype(np.int16), low & 7)
    high = np.where(high & 8, -(high & 7).astype(np.int16), high & 7)
    int4_codes = ((low & 15) | ((high & 15) << 4)).astype(np.uint8)
    int4_scales = np.exp2(exponents[0].astype(np.int32) - 128).astype(np.float16)
    int4_packed, int4_packed_scales = pack_int4_group32(int4_codes, int4_scales)
    np.testing.assert_array_equal(int4_packed, packed[0])
    np.testing.assert_array_equal(int4_packed_scales, scales[0])
    return (
        int4_packed,
        int4_packed_scales,
        dict(
            int4=int4_weight(int4_codes, int4_scales),
            mxfp4=weight(codes[0], exponents[0], k),
        ),
    )
