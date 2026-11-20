#pragma once
#include <common/device.h>
#include <int4_g32_a8/layout.h>

namespace int4_g32_a8 {
using native_device::ub_ptr;
__aicore__ inline int valid_elements(int step) {
    return step % blocks == blocks - 1 ? local - (blocks - 1) * elements : elements;
}
__aicore__ inline void load(const Inputs &args, const Tile &tile, int step, int serial, int lane) {
    const int slot = serial & 1, count = valid_elements(step);
    const int64_t column = int64_t(tile.expert) * width + tile.col;
    const int64_t start =
        column * depth + step / blocks * weight_bytes + lane * local + step % blocks * elements;
    auto raw = ub_ptr<uint8_t>(raw_base + slot * elements / 2);
    auto factors = ub_ptr<uint8_t>(factor_base + slot * scale_slot);
    if (serial >= slots)
        wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
    asc_copy_gm2ub(raw, reinterpret_cast<__gm__ uint8_t *>(args.codes) + start / 2,
                   uint32_t(count / 2));
    auto source = reinterpret_cast<__gm__ uint8_t *>(args.factors) + start / 64;
    if constexpr (scale_bytes % 32 == 0)
        asc_copy_gm2ub(factors, source, uint32_t(scale_bytes));
    else
        asc_copy_gm2ub_align(factors, source, 1, count / 64, 0, 0, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, ::event_t(slot));
}
__aicore__ inline void decode(int step, int serial, int known) {
    const int slot = serial & 1, count = valid_elements(step);
    auto raw = ub_ptr<uint8_t>(raw_base + slot * elements / 2);
    auto factors = ub_ptr<uint8_t>(factor_base + slot * scale_slot);
    wait_flag(PIPE_MTE2, PIPE_V, ::event_t(slot));
    auto rank = ub_ptr<half>(rank_base);
    auto small = ub_ptr<half>(small_base);
    auto expanded = ub_ptr<half>(expanded_base);
    auto decoded = ub_ptr<int8_t>(output_base + slot * elements);
    const uint8_t repeats = count / 128;
    pipe_barrier(PIPE_V);
    if constexpr (elements % (128 * group) == 0) {
        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
        asc_int42half(small, reinterpret_cast<__ubuf__ int4b_t *>(factors),
                      uint8_t(count / (128 * group)), 1, 1, 8, 2);
    } else {
        asc_int42half(small, reinterpret_cast<__ubuf__ int4b_t *>(factors),
                      uint32_t(count / group));
        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    }
    asc_int42half(rank, reinterpret_cast<__ubuf__ int4b_t *>(raw), repeats, 1, 1, 8, 2);
    if (known < 0 || serial + slots < known * blocks)
        set_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
    pipe_barrier(PIPE_V);
    asc_set_vector_mask(0, UINT64_MAX);
    for (int offset = 0; offset < 32; offset += 16)
        asc_add_scalar(expanded + offset, small + offset, half(9), uint8_t(count / 1024), 2, 0, 8,
                       2);
    asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    pipe_barrier(PIPE_V);
    if (serial >= slots)
        wait_flag(PIPE_MTE3, PIPE_V, ::event_t(slot));
#pragma unroll
    for (int atom = 0; atom < elements / 1024; ++atom)
        if (atom * 1024 < count)
            asc_mul_cast_half2int8(decoded + atom * 1024, rank + atom * 1024, expanded + atom * 128,
                                   uint8_t(8), 1, 1, 1, 4, 8, 0);
    set_flag(PIPE_V, PIPE_MTE3, ::event_t(slot));
}
__aicore__ inline void publish(int step, int serial, int lane, __gm__ int8_t *bridge, int known) {
    const int slot = serial & 1, generation = serial / blocks, inner = step % blocks;
    if (!inner && generation >= slots)
        CrossCoreWaitFlag<2, PIPE_MTE3>(2 + (generation & 1));
    wait_flag(PIPE_V, PIPE_MTE3, ::event_t(slot));
    asc_copy_ub2gm(bridge + (generation & 1) * weight_bytes + lane * local + inner * elements,
                   ub_ptr<int8_t>(output_base + slot * elements),
                   uint32_t(valid_elements(step)));
    if (inner == blocks - 1)
        CrossCoreSetFlag<2, PIPE_MTE3>(generation & 1);
    if (known < 0 || serial + slots < known * blocks)
        set_flag(PIPE_MTE3, PIPE_V, ::event_t(slot));
}

template <bool row_split>
__aicore__ inline void finish_tile(const Inputs &args, const Tile &tile, int job, int lane,
                                   __gm__ int8_t *bridge, const Tile *ahead) {
    constexpr int channels = row_split ? columns : half_n;
    const int split = (tile.valid + 1) / 2;
    const int offset = row_split ? lane * split : 0;
    const int valid = row_split ? (lane ? tile.valid - split : split) : tile.valid;
    const int first_col = row_split ? 0 : lane * half_n;
    auto dot = ub_ptr<int32_t>(final_base);
    auto value = reinterpret_cast<__ubuf__ float *>(dot);
#if A8_BF16
    using Output = bfloat16_t;
#else
    using Output = half;
#endif
    auto out = ub_ptr<Output>(final_base + final_elements * 4);
    auto channel = ub_ptr<float>(final_base + final_elements * 6);
    auto token = channel + channels;
    auto broadcast = token + rows;
    auto source =
        reinterpret_cast<__gm__ int32_t *>(bridge + result_offset + (job & 1) * result_bytes);
    auto dst = reinterpret_cast<__gm__ Output *>(args.output) + int64_t(tile.row + offset) * width +
               tile.col + first_col;
    if (job)
        wait_flag(PIPE_V, PIPE_MTE2, ::event_t(6));
    asc_copy_gm2ub(channel,
                   reinterpret_cast<__gm__ float *>(args.channel) + int64_t(tile.expert) * width +
                       tile.col + first_col,
                   uint32_t(channels * 4));
    asc_copy_gm2ub_align(token, reinterpret_cast<__gm__ float *>(args.token) + tile.row + offset, 1,
                         valid * 4, 0, 0, 0, 0);
    CrossCoreWaitFlag<2, PIPE_MTE2>(4 + (job & 1));
    if constexpr (row_split)
        asc_copy_gm2ub(dot, source + offset * columns, uint32_t(valid * columns * 4));
    else
        asc_copy_gm2ub(dot, source + first_col, uint16_t(valid), channels / 8, channels / 8, 0);
    set_flag(PIPE_MTE2, PIPE_V, ::event_t(6));
    if constexpr (carry_weights) {
        if (ahead)
            load(args, *ahead, lookahead, (job + slots) * weight_steps + lookahead, lane);
    }
    wait_flag(PIPE_MTE2, PIPE_V, ::event_t(6));
    asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    asc_brcb(reinterpret_cast<__ubuf__ uint32_t *>(broadcast),
             reinterpret_cast<__ubuf__ uint32_t *>(token), 1, 8, uint8_t((valid + 7) / 8));
    asc_int322float_rn(value, dot, uint32_t(valid * channels));
    pipe_barrier(PIPE_V);
    for (int col = 0; col < channels; col += 64) {
        asc_set_vector_mask(0, channels - col >= 64 ? UINT64_MAX : UINT32_MAX);
        asc_mul(value + col, value + col, channel + col, uint8_t(valid), 1, 1, 1, channels / 8,
                channels / 8, 0);
    }
    pipe_barrier(PIPE_V);
    for (int col = 0; col < channels; col += 64) {
        asc_set_vector_mask(0, channels - col >= 64 ? UINT64_MAX : UINT32_MAX);
        asc_mul(value + col, value + col, broadcast, uint8_t(valid), 1, 1, 0, channels / 8,
                channels / 8, 1);
    }
    pipe_barrier(PIPE_V);
    asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
    if (job)
        wait_flag(PIPE_MTE3, PIPE_V, ::event_t(6));
#if A8_BF16
    asc_float2bfloat16_rn(out, value, uint32_t(valid * channels));
#else
    asc_float2half_rn(out, value, uint32_t(valid * channels));
#endif
    set_flag(PIPE_V, PIPE_MTE2, ::event_t(6));
    native_device::pipe_fence<PIPE_V, PIPE_MTE3>(6);
    asc_copy_ub2gm(dst, out, uint16_t(valid), channels / 16, 0, uint16_t((width - channels) / 16));
    set_flag(PIPE_MTE3, PIPE_V, ::event_t(6));
}
__aicore__ inline void finish(const Inputs &args, const Tile &tile, int job, int lane,
                              __gm__ int8_t *bridge, const Tile *ahead = nullptr) {
    if constexpr (rows > fractal_rows) {
        if (tile.valid > fractal_rows) {
            finish_tile<true>(args, tile, job, lane, bridge, ahead);
            return;
        }
    }
    finish_tile<false>(args, tile, job, lane, bridge, ahead);
}
} // namespace int4_g32_a8
