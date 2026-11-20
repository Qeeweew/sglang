#pragma once
#include <common/device.h>
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <mhc/geometry.h>
namespace mhc {
using native_device::ub_ptr;
using namespace AscendC;
template <class T> __aicore__ inline __gm__ T *gm(GM_ADDR address) {
    return reinterpret_cast<__gm__ T *>(address);
}
// Serialized coefficient/scalar handoffs use an ID belonging to their exact
// pipe pair. Concurrent raw and packed slots reserve independent IDs below.
struct VectorEvents {
    static constexpr TEventID loaded = 0, raw_free = 0, scalar_read = 0, scalar_write = 0,
                              store_ready = 0, stored = 0;
    static constexpr TEventID raw_ready[kSlots] = {1, 2}, raw_reuse[kSlots] = {1, 2};
    static constexpr TEventID packed_ready[kSlots] = {1, 2}, packed_reuse[kSlots] = {1, 2};
    static constexpr TEventID load_drained = 0, store_drained = 0;
};
// One descriptor gathers branch rows into a padded UB tile. All columns are
// 64-aligned, so both row strides are integral32-byte blocks.
template <class T>
__aicore__ inline void load_rows(__ubuf__ T *dst, __gm__ T *src, uint32_t rows, uint32_t columns,
                                 uint32_t source_stride, uint32_t tile_stride) {
    if (columns == source_stride && columns == tile_stride)
        asc_copy_gm2ub(dst, src, uint32_t((rows * columns) * sizeof(T)));
    else
        asc_copy_gm2ub(dst, src, uint16_t(rows), uint16_t(columns * sizeof(T) / kDataBlockBytes),
                       uint16_t((source_stride - columns) * sizeof(T) / kDataBlockBytes),
                       uint16_t((tile_stride - columns) * sizeof(T) / kDataBlockBytes));
}
__aicore__ inline void reduce_r64(__ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *scratch,
                                  uint32_t n) {
    {
        asc_set_vector_mask(0, UINT64_MAX);
        asc_repeat_reduce_sum(scratch, src, uint8_t(int32_t(n / kRepeatFloats)), 1, 1,
                              kRepeatBlocks);
    };
    pipe_barrier(PIPE_V);
    {
        asc_set_vector_mask(0, (int32_t(n / kRepeatFloats)) >= kRepeatFloats
                                   ? UINT64_MAX
                                   : (uint64_t(1) << (int32_t(n / kRepeatFloats))) - 1);
        asc_repeat_reduce_sum(dst, scratch, uint8_t(1), 1, 1, kRepeatBlocks);
    };
    pipe_barrier(PIPE_V);
}
template <ReduceType type>
__aicore__ inline void reduce_blocks(__ubuf__ float *dst, __ubuf__ float *src, uint32_t count) {
    // The repeat field is eight bits. A VR8 x K2048 panel has256 repeats;
    // split it without changing the per-block FP32 addition tree.
    constexpr uint32_t max_elements = 255 * kRepeatFloats;
    for (uint32_t start = 0; start < count; start += max_elements) {
        const uint32_t n = count - start < max_elements ? count - start : max_elements;
        {
            asc_set_vector_mask(
                0, (int32_t(n < kRepeatFloats ? n : kRepeatFloats)) >= kRepeatFloats
                       ? UINT64_MAX
                       : (uint64_t(1) << (int32_t(n < kRepeatFloats ? n : kRepeatFloats))) - 1);
            asc_datablock_reduce_sum((dst + (start / kBlockFloats)), (src + (start)),
                                     uint8_t(int32_t((n + kRepeatFloats - 1) / kRepeatFloats)), 1,
                                     1, kRepeatBlocks);
        };
    }
}
// Projection's block8 -> block8 -> r32 tree matches the former AF schedule.
__aicore__ inline void panel_sum(__ubuf__ float *dst, __ubuf__ float *src, __ubuf__ float *scratch,
                                 uint32_t r, uint32_t panel) {
    reduce_blocks<ReduceType::SUM>(scratch, src, r * panel);
    pipe_barrier(PIPE_V);
    reduce_blocks<ReduceType::SUM>((scratch + (r * panel / kBlockFloats)), scratch,
                                   r * panel / kBlockFloats);
    pipe_barrier(PIPE_V);
    {
        asc_set_vector_mask(0, (int32_t(panel / 64)) >= kRepeatFloats
                                   ? UINT64_MAX
                                   : (uint64_t(1) << (int32_t(panel / 64))) - 1);
        asc_repeat_reduce_sum(dst, (scratch + (r * panel / kBlockFloats)), uint8_t(int32_t(r)), 1,
                              1, panel / 512);
    };
    pipe_barrier(PIPE_V);
}
__aicore__ inline float inverse_rms(__ubuf__ float *stat, float inverse_width, VectorEvents &ev) {
    const uint32_t scalar_elements = 1;
    const uint64_t scalar_mask =
        scalar_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << scalar_elements) - 1;
    const uint8_t scalar_repeats = (1 + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, scalar_mask);
    asc_mul_scalar(stat, stat, inverse_width, scalar_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_add_scalar(stat, stat, 1e-20f, scalar_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_sqrt(stat, stat, scalar_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    {
        asc_duplicate((stat + (kBlockFloats)), 1.0f, kBlockFloats);
        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    };
    pipe_barrier(PIPE_V);
    asc_set_vector_mask(0, scalar_mask);
    asc_div(stat, (stat + (kBlockFloats)), stat, scalar_repeats, 1, 1, 1, kRepeatBlocks,
            kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    native_device::pipe_fence<PIPE_V, PIPE_S, VectorEvents::scalar_read>();
    return stat[0];
}
#if defined(MHC_COMBINE)
__aicore__ inline void normalize_row(__ubuf__ bfloat16_t *collapsed, __ubuf__ float *value,
                                     __ubuf__ float *work, __ubuf__ float *scratch,
                                     __ubuf__ float *stat, __ubuf__ float *weight, uint32_t width,
                                     float inverse_width, VectorEvents &ev) {
    const uint32_t half = width / 2;
    constexpr bool full_row = kColumnTile >= kWidth;
    {
        if (full_row) {
            {
                asc_bfloat162float(value, collapsed, width);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            pipe_barrier(PIPE_V);
            const uint32_t row_elements = width;
            const uint64_t row_mask =
                row_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << row_elements) - 1;
            const uint8_t row_repeats = (width) / kRepeatFloats;
            asc_set_vector_mask(0, row_mask);
            asc_mul(work, value, value, row_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
                    kRepeatBlocks);
            pipe_barrier(PIPE_V);
            reduce_r64(stat, work, scratch, half);
            reduce_r64((stat + (kBlockFloats)), (work + (half)), scratch, half);
        } else
            for (uint32_t part = 0; part < 2; ++part) {
                {
                    asc_bfloat162float(value, (collapsed + (part * half)), half);
                    asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
                };
                pipe_barrier(PIPE_V);
                const uint32_t half_elements = half;
                const uint64_t half_mask = half_elements >= kRepeatFloats
                                               ? UINT64_MAX
                                               : (uint64_t(1) << half_elements) - 1;
                const uint8_t half_repeats = (half) / kRepeatFloats;
                asc_set_vector_mask(0, half_mask);
                asc_mul(work, value, value, half_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
                        kRepeatBlocks);
                pipe_barrier(PIPE_V);
                reduce_r64((stat + (part * kBlockFloats)), work, scratch, half);
            }
    }
    const uint32_t scalar_elements = 1;
    const uint64_t scalar_mask =
        scalar_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << scalar_elements) - 1;
    const uint8_t scalar_repeats = (1 + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, scalar_mask);
    asc_add(stat, stat, (stat + (kBlockFloats)), scalar_repeats, 1, 1, 1, kRepeatBlocks,
            kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    const float inverse = inverse_rms(stat, inverse_width, ev);
    if (full_row) {
        const uint32_t row_elements = width;
        const uint64_t row_mask =
            row_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << row_elements) - 1;
        const uint8_t row_repeats = (width) / kRepeatFloats;
        asc_set_vector_mask(0, row_mask);
        asc_mul_scalar(work, value, inverse, row_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
        pipe_barrier(PIPE_V);
        asc_mul(work, work, weight, row_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
                kRepeatBlocks);
        pipe_barrier(PIPE_V);
        {
            asc_float2bfloat16_rn(collapsed, work, width);
            asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
        };
        pipe_barrier(PIPE_V);

    } else
        for (uint32_t part = 0; part < 2; ++part) {
            {
                asc_bfloat162float(value, (collapsed + (part * half)), half);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            pipe_barrier(PIPE_V);
            const uint32_t half_elements = half;
            const uint64_t half_mask =
                half_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << half_elements) - 1;
            const uint8_t half_repeats = (half) / kRepeatFloats;
            asc_set_vector_mask(0, half_mask);
            asc_mul_scalar(work, value, inverse, half_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
            pipe_barrier(PIPE_V);
            asc_mul(work, work, (weight + (part * half)), half_repeats, 1, 1, 1, kRepeatBlocks,
                    kRepeatBlocks, kRepeatBlocks);
            pipe_barrier(PIPE_V);
            {
                asc_float2bfloat16_rn((collapsed + (part * half)), work, half);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            pipe_barrier(PIPE_V);
        }
}
#endif
} // namespace mhc
