#pragma once
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <position_quant/params.h>

namespace position_quant {
// Private, fixed-capacity UB. Allocation is independent of runtime geometry;
// every user owns its dataflow events and explicitly drains consumer pipes.
// Reserve the last hardware ID for immediate tail drains; dataflow uses ID0.
inline constexpr AscendC::TEventID kTailDrainEvent = 7;
inline constexpr uint32_t kBlockFloats = AscendC::ONE_BLK_SIZE / sizeof(float);
inline constexpr uint32_t kVectorFloats = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(float);
inline constexpr uint32_t kVectorBlocks = AscendC::ONE_REPEAT_BYTE_SIZE / AscendC::ONE_BLK_SIZE;
static_assert(kDmaAlignmentBytes == AscendC::ONE_BLK_SIZE);
struct GatherArena {
    static constexpr uint32_t trig_columns = kRotaryCapacityElements / kRotaryPairElements;
    static constexpr uint32_t trig_bytes = kRotaryPairElements * trig_columns * sizeof(float);
    static constexpr uint32_t cosine = 0;
    static constexpr uint32_t mask = cosine + trig_bytes;
    static constexpr uint32_t mask_bytes = kGatherGroupRows * sizeof(int32_t);
    static constexpr uint32_t slots = mask + mask_bytes;
    static constexpr uint32_t slot_bytes = kGatherGroupRows * sizeof(int64_t);
    static constexpr uint32_t bytes = slots + slot_bytes;
    static_assert(bytes <= arch2201::kUbBytes);
};
struct ScatterArena {
    // Cast emits one full FP16 scale group. Two symmetric conversion slots
    // retain the qualified layout: FP32 source followed by FP16 destination.
    static constexpr uint32_t cast_elements = kScaleGroupRows;
    static constexpr uint32_t source_bytes = cast_elements * sizeof(float);
    static constexpr uint32_t result_bytes = source_bytes;
    static constexpr uint32_t scale_bytes = source_bytes + result_bytes;
};
struct QuantArena {
    static constexpr uint32_t width = kWidthCapacityElements;
    static constexpr uint32_t rotary = kRotaryCapacityElements;
    static constexpr uint32_t bf16_bytes = width * sizeof(bfloat16_t);
    static constexpr uint32_t fp32_bytes = width * sizeof(float);
    static constexpr uint32_t rotary_bytes = rotary * sizeof(float);
    static constexpr uint32_t trig_bytes = kRotaryPairElements * rotary_bytes;
    static constexpr uint32_t offset_bytes = kRotaryPairElements * rotary * sizeof(uint32_t);
    static constexpr uint32_t scale32_bytes = kScaleGroupRows * sizeof(float);
    static constexpr uint32_t scale16_bytes = kScaleGroupRows * sizeof(half);
    static constexpr uint32_t code_bytes = width * sizeof(int8_t);
    static constexpr uint32_t rounded_bytes = width * sizeof(int16_t);
    static constexpr uint32_t input = 0;
    static constexpr uint32_t output = input + bf16_bytes;
    static constexpr uint32_t floats = output + bf16_bytes;
    static constexpr uint32_t work = floats + fp32_bytes;
    static constexpr uint32_t absolute = work + fp32_bytes;
    static constexpr uint32_t reduction = absolute + fp32_bytes;
    static constexpr uint32_t rawtrig = reduction + fp32_bytes;
    static constexpr uint32_t trig = rawtrig + rotary_bytes;
    static constexpr uint32_t swapped = trig + trig_bytes;
    static constexpr uint32_t offsets = swapped + rotary_bytes;
    static constexpr uint32_t signs = offsets + offset_bytes;
    static constexpr uint32_t scale32 = signs + rotary_bytes;
    static constexpr uint32_t scale16 = scale32 + scale32_bytes;
    static constexpr uint32_t codes = scale16 + scale16_bytes;
    static constexpr uint32_t rounded = codes + code_bytes;
    static constexpr uint32_t bytes = rounded + rounded_bytes;
    static_assert(bytes <= arch2201::kUbBytes);
};
// A block8 max tree followed by bounded repeat reductions. These basic APIs
// issue only Vector work; unlike ReduceMax(false), they never allocate hidden
// TPipe events or move a partial result through Scalar.
__aicore__ inline void maximum(__ubuf__ float *result, __ubuf__ float *input,
                               __ubuf__ float *partial, uint32_t width) {
    asc_set_mask_count();
    asc_set_vector_mask(0, width);
    asc_datablock_reduce_max(partial, input, uint8_t(1), 1, 1, kVectorBlocks);
    asc_set_mask_norm();
    pipe_barrier(PIPE_V);
    for (uint32_t offset = 0; offset < width / kBlockFloats; offset += kVectorFloats) {
        const uint32_t count = min(kVectorFloats, width / kBlockFloats - offset);
        asc_set_vector_mask(0, count == kVectorFloats ? ~uint64_t(0) : (uint64_t(1) << count) - 1);
        asc_repeat_reduce_max_only_value(offset ? result + kBlockFloats : result, partial + offset,
                                         uint8_t(1), 1, 1, kVectorBlocks);
        pipe_barrier(PIPE_V);
        if (offset) {
            asc_max(result, result, result + kBlockFloats, uint32_t(1));
            pipe_barrier(PIPE_V);
        }
    }
}
} // namespace position_quant
