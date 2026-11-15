#pragma once
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <elementwise/layout.h>

namespace elementwise {
using namespace AscendC;
// This entry owns all basic-API events and uses no TPipe/TQue allocations.
// IDs0/1 are independent raw/packed slots on each named pipe pair. There are
// no SDK implicit credits or event-pool reset in this operator.
struct RowEvents {
    static constexpr TEventID input_ready[kSlots] = {0, 1};
    static constexpr TEventID raw_reusable[kSlots] = {0, 1};
    static constexpr TEventID output_ready[kSlots] = {0, 1};
    static constexpr TEventID output_reusable[kSlots] = {0, 1};
    static constexpr TEventID scalar_ready = 0, load_drained = 0, output_drained = 0;
};
static_assert(kVectorRepeatBytes == ONE_REPEAT_BYTE_SIZE && kDmaBlockBytes == ONE_BLK_SIZE);
// Pure conversions: preserve NONE on import and RNE on output.
template <class Storage>
__aicore__ inline void unpack(__ubuf__ float *dst, __ubuf__ Storage *src, uint32_t elements) {
    if constexpr (__is_same(Storage, half))
        asc_half2float(dst, src, elements);
    else
        asc_bfloat162float(dst, src, elements);
}
template <class Storage>
__aicore__ inline void pack(__ubuf__ Storage *dst, __ubuf__ float *src, uint32_t elements) {
    if constexpr (__is_same(Storage, half))
        asc_float2half_rn(dst, src, elements);
    else
        asc_float2bfloat16_rn(dst, src, elements);
}
// Two-level r64 FP32 addition tree; owns mask changes, no scalar handoff.
__aicore__ inline void reduce_r64(__ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *scratch,
                                  uint32_t elements) {
    asc_set_vector_mask(0, ~uint64_t(0));
    asc_repeat_reduce_sum(scratch, src, uint8_t(elements / kFp32VectorElements), 1, 1,
                          kRepeatBlocks);
    pipe_barrier(PIPE_V);
    const uint32_t count = elements / kFp32VectorElements;
    asc_set_vector_mask(0,
                        count == kFp32VectorElements ? ~uint64_t(0) : (uint64_t(1) << count) - 1);
    asc_repeat_reduce_sum(dst, scratch, uint8_t(1), 1, 1, kRepeatBlocks);
    pipe_barrier(PIPE_V);
}
} // namespace elementwise
