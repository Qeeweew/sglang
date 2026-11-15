#pragma once
#include <mhc/vector_pipeline.h>
namespace mhc {
// Source-major [branches, rows, eight lanes] makes every Sinkhorn row one
// data block. Runtime B<=8 follows that dataflow, independent of V4.1 B=4.
// Caller selects one eight-lane NORMAL mask. arch2201 Adds accepts
// signed INT32; byte offsets fit its positive range.
__aicore__ inline void byte_offsets(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *lanes,
                                    uint32_t base) {
    asc_add_scalar(reinterpret_cast<__ubuf__ int32_t *>(dst),
                   reinterpret_cast<__ubuf__ int32_t *>(lanes), int32_t(base), 1, 1, 1,
                   kRepeatBlocks, kRepeatBlocks);
}
__aicore__ inline uint64_t lane_bits(uint32_t elements) {
    return elements >= kRepeatFloats ? UINT64_MAX : (UINT64_C(1) << elements) - 1;
}
struct StatisticsConstants {
    float pre_scale, post_scale, matrix_scale;
    bool resident = false;
};
__aicore__ inline void compute_statistics(GM_ADDR projection, GM_ADDR squares, GM_ADDR scale,
                                          GM_ADDR base, GM_ADDR pre, GM_ADDR post, GM_ADDR comb,
                                          const StatsParams &p, uint64_t first, uint32_t r,
                                          bool publish, VectorEvents &ev,
                                          StatisticsConstants &constants, float epsilon, uint64_t rows,
                                              uint64_t padded_rows) {
    const uint32_t b = p.branches, lanes = kBlockFloats;
    constexpr StatsLayout l;
    auto a = ub<float>(l.a), other = ub<float>(l.b);
    auto stat = ub<float>(l.stat), gate = ub<float>(l.gate), bias = ub<float>(l.bias);
    auto matrix = ub<float>(l.matrix), red = ub<float>(l.reduction);
    auto broad = ub<float>(l.broadcast), column = ub<float>(l.column);
    auto ones = ub<float>(l.ones), basecache = ub<float>(l.base);
    auto gate_index = ub<uint32_t>(l.gate_index);
    auto matrix_index = ub<uint32_t>(l.matrix_index);
    auto matrix_bias = ub<float>(l.matrix_bias);
    const uint64_t live_mask = ((uint64_t(1) << b) - 1) * UINT64_C(0x0101010101010101);
    uint64_t mask[2] = {live_mask, 0};
    uint64_t gate_mask[2] = {live_mask & lane_bits(2 * r * lanes), 0};
    uint64_t matrix_gather_mask[2] = {live_mask & lane_bits(b * r * lanes), 0};
    const uint8_t matrix_repeats = (b * r + lanes - 1) / lanes;
    // Cache role masks once. The Sinkhorn loop restores their bitmaps
    // directly after Brcb/reduction instead of rebuilding a mask from len.
    const uint64_t matrix_mask = lane_bits(b * r * lanes);
    const uint64_t branch_mask = lane_bits(r * lanes);
    const uint64_t reduction_mask = lane_bits(b * r);
    auto projected = gm<float>(projection), squared = gm<float>(squares), scales = gm<float>(scale);
    if (!constants.resident) {
        {
            set_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
            wait_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
        }
        if (r != 1) {
            asc_copy_gm2ub_align(basecache, gm<float>(base), 1,
                                 p.projection_columns * sizeof(*(basecache)), 0, 0, 0, 0);
            {
                set_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
                wait_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
            }
        }
        constants.pre_scale = scales[0];
        constants.post_scale = scales[1];
        constants.matrix_scale = scales[2];
        {
            asc_duplicate(ones, 1.0f, 2 * r * lanes);
            asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
        };
        if (r == 1) {
            // DMA lays compact GM coefficients into aligned eight-lane rows.
            // Projection Gather needs only its own resident index packet.
            auto lane_index = reinterpret_cast<__ubuf__ int32_t *>(column);
#pragma unroll
            for (uint32_t lane = 0; lane < kBlockFloats; ++lane)
                lane_index[lane] = int32_t(lane * sizeof(float));
            {
                set_flag(PIPE_S, PIPE_V, ::event_t(VectorEvents::scalar_write));
                wait_flag(PIPE_S, PIPE_V, ::event_t(VectorEvents::scalar_write));
            }
            {
                asc_duplicate(gate_index, uint32_t(0), kRepeatFloats);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                asc_duplicate(matrix_index, uint32_t(0), kRepeatFloats);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                asc_duplicate(bias, 0.0f, 2 * lanes);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                asc_duplicate(matrix_bias, -__builtin_inff(), kRepeatFloats);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                set_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
                wait_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
            }
            asc_copy_gm2ub_align(bias, gm<float>(base), 2, b * uint32_t(sizeof(float)), 0, 0, 0, 0);
            asc_copy_gm2ub_align(matrix_bias, (gm<float>(base) + (2 * b)), uint16_t(b),
                                 b * uint32_t(sizeof(float)), 0, 0, 0, 0);
            {
                set_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
                wait_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
            }
            uint64_t inactive[2] = {~live_mask, 0};
            {
                asc_set_vector_mask(inactive[1], inactive[0]);
                asc_duplicate(matrix_bias, -__builtin_inff(), 1, 1, kRepeatBlocks);
            };
            asc_set_vector_mask(0, (int32_t(lanes)) >= 64 ? UINT64_MAX
                                                          : (uint64_t(1) << (int32_t(lanes))) - 1);
            byte_offsets(gate_index, reinterpret_cast<__ubuf__ uint32_t *>(column), 0);
            byte_offsets((gate_index + (lanes)), reinterpret_cast<__ubuf__ uint32_t *>(column),
                         b * uint32_t(sizeof(float)));
            for (uint32_t source = 0; source < b; ++source)
                byte_offsets((matrix_index + (source * lanes)),
                             reinterpret_cast<__ubuf__ uint32_t *>(column),
                             (2 * b + source * b) * uint32_t(sizeof(float)));
            pipe_barrier(PIPE_V);
        } else {
            // Indices and biases remain resident for every row tile owned by
            // this actor. Vector-generated byte offsets avoid per-element
            // Scalar UB loads/stores while dimensions remain runtime values.
            auto lane_index = reinterpret_cast<__ubuf__ int32_t *>(column);
            auto bias_index = reinterpret_cast<__ubuf__ uint32_t *>(other);
            // CreateVecIndex/ArithProgression call the SDK TPipe allocator on
            // arch2201. This basic-API entry owns the eight Scalar seeds and
            // their explicit S -> V handoff instead.
#pragma unroll
            for (uint32_t lane = 0; lane < kBlockFloats; ++lane)
                lane_index[lane] = int32_t(lane * sizeof(float));
            {
                set_flag(PIPE_S, PIPE_V, ::event_t(VectorEvents::scalar_write));
                wait_flag(PIPE_S, PIPE_V, ::event_t(VectorEvents::scalar_write));
            }
            {
                asc_duplicate(gate_index, uint32_t(0), round_up(2 * r * lanes, kRepeatFloats));
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                asc_duplicate(bias_index, uint32_t(0), round_up(2 * r * lanes, kRepeatFloats));
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                asc_duplicate(bias, 0.0f, 2 * r * lanes);
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            pipe_barrier(PIPE_V);
            asc_set_vector_mask(0, (int32_t(kBlockFloats)) >= 64
                                       ? UINT64_MAX
                                       : (uint64_t(1) << (int32_t(kBlockFloats))) - 1);
            for (uint32_t row = 0; row < r; ++row) {
                byte_offsets((gate_index + (row * lanes)),
                             reinterpret_cast<__ubuf__ uint32_t *>(column),
                             row * p.projection_stride * uint32_t(sizeof(float)));
                byte_offsets((gate_index + ((r + row) * lanes)),
                             reinterpret_cast<__ubuf__ uint32_t *>(column),
                             (row * p.projection_stride + b) * uint32_t(sizeof(float)));
                byte_offsets((bias_index + (row * lanes)),
                             reinterpret_cast<__ubuf__ uint32_t *>(column), uint32_t(0));
                byte_offsets((bias_index + ((r + row) * lanes)),
                             reinterpret_cast<__ubuf__ uint32_t *>(column),
                             b * uint32_t(sizeof(float)));
            }
            pipe_barrier(PIPE_V);
            {
                asc_set_vector_mask(gate_mask[1], gate_mask[0]);
                asc_gather(bias, basecache, bias_index, kRepeatBlocks,
                           uint8_t((2 * r + lanes - 1) / lanes));
            };
            {
                asc_duplicate(matrix_index, uint32_t(0), round_up(b * r * lanes, kRepeatFloats));
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                asc_duplicate(bias_index, uint32_t(0), round_up(b * r * lanes, kRepeatFloats));
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            {
                asc_duplicate(matrix_bias, -__builtin_inff(),
                              round_up(b * r * lanes, kRepeatFloats));
                asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
            };
            pipe_barrier(PIPE_V);
            asc_set_vector_mask(0, (int32_t(kBlockFloats)) >= 64
                                       ? UINT64_MAX
                                       : (uint64_t(1) << (int32_t(kBlockFloats))) - 1);
            for (uint32_t source = 0; source < b; ++source)
                for (uint32_t row = 0; row < r; ++row) {
                    const uint32_t target = (source * r + row) * lanes;
                    const uint32_t base_column = 2 * b + source * b;
                    byte_offsets(
                        (matrix_index + (target)), reinterpret_cast<__ubuf__ uint32_t *>(column),
                        (row * p.projection_stride + base_column) * uint32_t(sizeof(float)));
                    byte_offsets((bias_index + (target)),
                                 reinterpret_cast<__ubuf__ uint32_t *>(column),
                                 base_column * uint32_t(sizeof(float)));
                }
            pipe_barrier(PIPE_V);
            {
                asc_set_vector_mask(matrix_gather_mask[1], matrix_gather_mask[0]);
                asc_gather(matrix_bias, basecache, bias_index, kRepeatBlocks, matrix_repeats);
            };
            pipe_barrier(PIPE_V);
        }
        constants.resident = true;
    }
    {
        set_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
        wait_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
    }
    asc_copy_gm2ub_align(a, (projected + (first * p.projection_stride)), 1,
                         r * p.projection_stride * sizeof(*(a)), 0, 0, 0, 0);
    {
        set_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
        wait_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
    }
    // Fixed storage capacity with constant lane indices keeps the live
    // square sums in Scalar registers. R itself remains a runtime extent.
    float sums[kStatsRowCapacity];
#pragma unroll
    for (uint32_t row = 0; row < kStatsRowCapacity; ++row)
        if (row < r)
            sums[row] = squared[first + row];
    const uint32_t merge_elements = r * p.projection_stride;
    const bool simple_span = merge_elements <= kRepeatFloats || merge_elements % kRepeatFloats == 0;
    const uint32_t merge_span_elements = simple_span ? merge_elements : kRepeatFloats;
    const uint64_t merge_mask = merge_span_elements >= kRepeatFloats
                                    ? UINT64_MAX
                                    : (uint64_t(1) << merge_span_elements) - 1;
    const uint8_t merge_repeats =
        ((simple_span ? merge_elements : kRepeatFloats) + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, merge_mask);
    for (uint32_t split = 1; split < p.splits; ++split) {
        {
            set_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
            wait_flag(PIPE_V, PIPE_MTE2, ::event_t(VectorEvents::raw_free));
        }
        asc_copy_gm2ub_align(
            other, (projected + ((uint64_t(split) * padded_rows + first) * p.projection_stride)),
            1, r * p.projection_stride * sizeof(*(other)), 0, 0, 0, 0);
        {
            set_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
            wait_flag(PIPE_MTE2, PIPE_V, ::event_t(VectorEvents::loaded));
        }
        if (simple_span)
            asc_add(a, a, other, merge_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
                    kRepeatBlocks);
        else {
            asc_add(a, a, other, merge_elements);
            asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
        };
        pipe_barrier(PIPE_V);
#pragma unroll
        for (uint32_t row = 0; row < kStatsRowCapacity; ++row)
            if (row < r)
                sums[row] += squared[uint64_t(split) * padded_rows + first + row];
    }
    {
        set_flag(PIPE_V, PIPE_S, ::event_t(VectorEvents::scalar_read));
        wait_flag(PIPE_V, PIPE_S, ::event_t(VectorEvents::scalar_read));
    }
#pragma unroll
    for (uint32_t row = 0; row < kStatsRowCapacity; ++row)
        if (row < r)
            stat[row] = sums[row] * p.inverse_input_width + epsilon;
    {
        set_flag(PIPE_S, PIPE_V, ::event_t(VectorEvents::scalar_write));
        wait_flag(PIPE_S, PIPE_V, ::event_t(VectorEvents::scalar_write));
    }
    const uint32_t row_elements = r;
    const uint64_t row_mask =
        row_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << row_elements) - 1;
    const uint8_t row_repeats = (r + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, row_mask);
    asc_sqrt(stat, stat, row_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_div(stat, ones, stat, row_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    {
        asc_duplicate(gate, 0.0f, 2 * r * lanes);
        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    };
    {
        asc_duplicate(matrix, 0.0f, round_up(b * r * lanes, kRepeatFloats));
        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    };
    pipe_barrier(PIPE_V);
    const uint32_t branch_elements = r * lanes;
    const uint64_t branch_span_mask =
        branch_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << branch_elements) - 1;
    const uint8_t branch_repeats = (r * lanes + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, branch_mask);
    const uint32_t matrix_span_elements = b * r * lanes;
    const uint64_t matrix_span_mask = matrix_span_elements >= kRepeatFloats
                                          ? UINT64_MAX
                                          : (uint64_t(1) << matrix_span_elements) - 1;
    const uint8_t matrix_span_repeats = (b * r * lanes + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, matrix_span_mask);
    {
        {
            asc_set_vector_mask(gate_mask[1], gate_mask[0]);
            asc_gather(gate, a, gate_index, kRepeatBlocks, uint8_t((2 * r + lanes - 1) / lanes));
        };
        {
            asc_set_vector_mask(matrix_gather_mask[1], matrix_gather_mask[0]);
            asc_gather(matrix, a, matrix_index, kRepeatBlocks, matrix_repeats);
        };
        asc_brcb(reinterpret_cast<__ubuf__ uint32_t *>(broad),
                 reinterpret_cast<__ubuf__ uint32_t *>(stat), 1, kRepeatBlocks, uint8_t(1));
        pipe_barrier(PIPE_V);
        if (r == 1)
            asc_set_vector_mask(0, branch_mask);
        asc_mul(gate, gate, broad, branch_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
                kRepeatBlocks);
        asc_mul((gate + (r * lanes)), (gate + (r * lanes)), broad, branch_repeats, 1, 1, 1,
                kRepeatBlocks, kRepeatBlocks, kRepeatBlocks);
        {
            asc_set_vector_mask(0, (int32_t(r * lanes)) >= 64
                                       ? UINT64_MAX
                                       : (uint64_t(1) << (int32_t(r * lanes))) - 1);
            asc_mul(matrix, matrix, broad, uint8_t(b), 1, 1, 1, uint8_t(r), uint8_t(r), 0);
        };
        pipe_barrier(PIPE_V);
        asc_mul_scalar(gate, gate, constants.pre_scale, branch_repeats, 1, 1, kRepeatBlocks,
                       kRepeatBlocks);
        asc_mul_scalar((gate + (r * lanes)), (gate + (r * lanes)), constants.post_scale,
                       branch_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
        if (r == 1)
            asc_set_vector_mask(0, matrix_span_mask);
        asc_mul_scalar(matrix, matrix, constants.matrix_scale, matrix_span_repeats, 1, 1,
                       kRepeatBlocks, kRepeatBlocks);
        pipe_barrier(PIPE_V);
        asc_add(matrix, matrix, matrix_bias, matrix_span_repeats, 1, 1, 1, kRepeatBlocks,
                kRepeatBlocks, kRepeatBlocks);
        pipe_barrier(PIPE_V);
    }
    const uint32_t gate_span_elements = 2 * r * lanes;
    const uint64_t gate_span_mask =
        gate_span_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << gate_span_elements) - 1;
    const uint8_t gate_span_repeats = (2 * r * lanes + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, gate_span_mask);
    asc_add(gate, gate, bias, gate_span_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
            kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_mul_scalar(gate, gate, -1.0f, gate_span_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_exp(gate, gate, gate_span_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_add_scalar(gate, gate, 1.0f, gate_span_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_div(gate, ones, gate, gate_span_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
            kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_set_vector_mask(uint64_t(0), branch_mask);
    asc_add_scalar(gate, gate, 1e-6f, branch_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    asc_mul_scalar((gate + (r * lanes)), (gate + (r * lanes)), 2.0f, branch_repeats, 1, 1,
                   kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    // The statistics capacity is at most eight repeats. Its one reduction
    // packet reuses the matrix span instead of a general large-span loop.
    asc_set_vector_mask(uint64_t(0), matrix_mask);
    asc_datablock_reduce_max(red, matrix, uint8_t(int32_t(matrix_repeats)), 1, 1, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_brcb(reinterpret_cast<__ubuf__ uint32_t *>(broad),
             reinterpret_cast<__ubuf__ uint32_t *>(red), 1, kRepeatBlocks,
             uint8_t((b * r + lanes - 1) / lanes));
    pipe_barrier(PIPE_V);
    asc_set_vector_mask(uint64_t(0), matrix_mask);
    asc_sub(matrix, matrix, broad, matrix_span_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
            kRepeatBlocks);
    pipe_barrier(PIPE_V);
    asc_exp(matrix, matrix, matrix_span_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
    pipe_barrier(PIPE_V);
    const uint32_t reduction_elements = b * r;
    const uint64_t reduction_span_mask =
        reduction_elements >= kRepeatFloats ? UINT64_MAX : (uint64_t(1) << reduction_elements) - 1;
    const uint8_t reduction_repeats = (b * r + kRepeatFloats - 1) / kRepeatFloats;
    asc_set_vector_mask(0, reduction_mask);
    for (uint32_t iteration = 0; iteration < kSinkhornIterations; ++iteration) {
        asc_set_vector_mask(uint64_t(0), matrix_mask);
        asc_datablock_reduce_sum(red, matrix, uint8_t(int32_t(matrix_repeats)), 1, 1,
                                 kRepeatBlocks);
        pipe_barrier(PIPE_V);
        if (iteration) {
            asc_set_vector_mask(uint64_t(0), reduction_mask);
            asc_add_scalar(red, red, 1e-6f, reduction_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
            pipe_barrier(PIPE_V);
        }
        asc_brcb(reinterpret_cast<__ubuf__ uint32_t *>(broad),
                 reinterpret_cast<__ubuf__ uint32_t *>(red), 1, kRepeatBlocks,
                 uint8_t((b * r + lanes - 1) / lanes));
        pipe_barrier(PIPE_V);
        asc_set_vector_mask(uint64_t(0), matrix_mask);
        asc_div(matrix, matrix, broad, matrix_span_repeats, 1, 1, 1, kRepeatBlocks, kRepeatBlocks,
                kRepeatBlocks);
        pipe_barrier(PIPE_V);
        if (!iteration) {
            {
                asc_set_vector_mask(mask[1], mask[0]);
                asc_add_scalar(matrix, matrix, 1e-6f, uint8_t((b * r + lanes - 1) / lanes), 1, 1,
                               kRepeatBlocks, kRepeatBlocks);
            };
            pipe_barrier(PIPE_V);
        }
        asc_set_vector_mask(uint64_t(0), branch_mask);
        if (b == 1)
            asc_mul_scalar(column, matrix, 1.0f, branch_repeats, 1, 1, kRepeatBlocks,
                           kRepeatBlocks);
        else
            asc_add(column, matrix, (matrix + (r * lanes)), branch_repeats, 1, 1, 1, kRepeatBlocks,
                    kRepeatBlocks, kRepeatBlocks);
        pipe_barrier(PIPE_V);
        uint32_t source = 2;
        // Two complete source additions per loop avoid capacity-wide
        // predication. A final source preserves the original serial order.
        for (; source + 1 < b; source += 2) {
            asc_add(column, column, (matrix + (source * r * lanes)), branch_repeats, 1, 1, 1,
                    kRepeatBlocks, kRepeatBlocks, kRepeatBlocks);
            pipe_barrier(PIPE_V);
            asc_add(column, column, (matrix + ((source + 1) * r * lanes)), branch_repeats, 1, 1, 1,
                    kRepeatBlocks, kRepeatBlocks, kRepeatBlocks);
            pipe_barrier(PIPE_V);
        }
        if (source < b) {
            asc_add(column, column, (matrix + (source * r * lanes)), branch_repeats, 1, 1, 1,
                    kRepeatBlocks, kRepeatBlocks, kRepeatBlocks);
            pipe_barrier(PIPE_V);
        }
        asc_add_scalar(column, column, 1e-6f, branch_repeats, 1, 1, kRepeatBlocks, kRepeatBlocks);
        pipe_barrier(PIPE_V);
        // Source-major rows repeat the same denominator block. One Vector
        // instruction broadcasts it with zero source repeat stride.
        if (r == 1) {
            // All eight source blocks fit one private repeat. Only the live
            // source rows participate in reductions; unused rows stay -Inf.
            // Broadcasting the denominator by block replaces B masked repeats.
            asc_set_vector_mask(0, UINT64_MAX);
            asc_div(matrix, matrix, column, 1, 1, 1, 0, kRepeatBlocks, kRepeatBlocks, 0);
        } else {
            asc_set_vector_mask(0, (int32_t(r * lanes)) >= 64
                                       ? UINT64_MAX
                                       : (uint64_t(1) << (int32_t(r * lanes))) - 1);
            asc_div(matrix, matrix, column, uint8_t(b), 1, 1, 1, uint8_t(r), uint8_t(r), 0);
        };
        pipe_barrier(PIPE_V);
    }
    const uint32_t valid = rows - first < r ? rows - first : r;
    if (publish) {
        {
            set_flag(PIPE_V, PIPE_MTE3, ::event_t(VectorEvents::store_ready));
            wait_flag(PIPE_V, PIPE_MTE3, ::event_t(VectorEvents::store_ready));
        }
        asc_copy_ub2gm_align((gm<float>(pre) + (first * b)), gate, uint16_t(valid),
                             b * uint32_t(sizeof(float)), 0, 0, 0, 0);
        asc_copy_ub2gm_align((gm<float>(post) + (first * b)), (gate + (r * lanes)), uint16_t(valid),
                             b * uint32_t(sizeof(float)), 0, 0, 0, 0);
        for (uint32_t source = 0; source < b; ++source)
            asc_copy_ub2gm_align((gm<float>(comb) + ((first * b + source) * b)),
                                 (matrix + (source * r * lanes)), uint16_t(valid),
                                 b * uint32_t(sizeof(float)), 0, 0, 0,
                                 (b * b - b) * uint32_t(sizeof(float)));
        {
            set_flag(PIPE_MTE3, PIPE_V, ::event_t(VectorEvents::stored));
            wait_flag(PIPE_MTE3, PIPE_V, ::event_t(VectorEvents::stored));
        }
    }
#if defined(MHC_STATS_POST)
    // The fused post stage reads matrix/gate directly from this disjoint
    // resident arena. Publish only the Vector -> Scalar dependency.
    {
        set_flag(PIPE_V, PIPE_S, ::event_t(VectorEvents::scalar_read));
        wait_flag(PIPE_V, PIPE_S, ::event_t(VectorEvents::scalar_read));
    }
#endif
}
} // namespace mhc
