// BF16 tail64 RoPE with compact FP32 trig. Only the rotated span is read or
// written in-place. Strided DMA packs a tile of heads/tokens into the UB.
#pragma once
#include <common/device.h>
#include <kernel_operator.h>
#include <position_quant/params.h>
using namespace AscendC;

namespace position_quant {
// V -> MTE2 retires zero-fill of a partial head tile before the input DMA
// overwrites its real lanes. Event 2 is separate from the input ring's 0/1.
inline constexpr TEventID kPartialHeadPaddingRetired = 2;
// Final V/MTE2/MTE3 -> Scalar drains retire ring-consumer Waits. Event 0 is
// reused only after the live ring credits have been consumed on each pair.
inline constexpr TEventID kRopeConsumerDrain = 0;
// A two-slot MTE2 -> Vector input ring and Vector -> MTE3 output ring.
// Each arena uses its constexpr allocation capacity; request-derived token/head
// tiles select the active elements. Scratch remains private to one Vector actor.
struct RopeArena {
    static constexpr uint32_t input_bytes = kTileRowsCapacity * kRotaryCapacityElements * 6;
    static constexpr uint32_t output_bytes = kTileRowsCapacity * kRotaryCapacityElements * 2;
    static constexpr uint32_t scratch_bytes = kTileRowsCapacity * kRotaryCapacityElements * 4;
    static constexpr uint32_t outputs = 2 * input_bytes;
    static constexpr uint32_t floats = outputs + 2 * output_bytes;
    static constexpr uint32_t swapped = floats + scratch_bytes;
    static constexpr uint32_t trig = swapped + scratch_bytes;
    static constexpr uint32_t swap_ids = trig + scratch_bytes;
    static constexpr uint32_t trig_ids = swap_ids + scratch_bytes;
    static_assert(trig_ids + scratch_bytes == kRopeTileBytes);
};
} // namespace position_quant

__aicore__ inline void rope_copy_in(__ubuf__ bfloat16_t *raw, __gm__ bfloat16_t *x, __gm__ float *c,
                                    __gm__ float *s, int64_t job, int64_t tokens, uint32_t heads,
                                    uint32_t bh, uint32_t bt, int64_t head_groups, TEventID ready,
                                    uint32_t dim, uint32_t rotary_dim, int32_t group_shift) {
    constexpr uint32_t B = position_quant::kTileRowsCapacity;
    const uint32_t R = rotary_dim;
    const int64_t token_group = group_shift >= 0 ? job >> group_shift : job / head_groups;
    int64_t token = token_group * bt;
    uint32_t head = (job - token_group * head_groups) * bh;
    uint32_t nt = tokens - token < bt ? tokens - token : bt,
             nh = heads - head < bh ? heads - head : bh;
    // A non-power-of-two head count can leave holes between token rows.
    if (nh < bh) {
        asc_duplicate(reinterpret_cast<__ubuf__ half *>(raw), half(0), nt * bh * R);
        native_device::pipe_fence<PIPE_V, PIPE_MTE2, position_quant::kPartialHeadPaddingRetired>();
    }
    if (nh == heads) {
        asc_copy_gm2ub_align(
            reinterpret_cast<__ubuf__ uint8_t *>(raw),
            reinterpret_cast<__gm__ uint8_t *>((x + (token * heads * dim + dim - R))),
            static_cast<uint16_t>(nt * nh), R * 2, 0, 0, (dim - R) * 2, 0);
    } else
        for (uint32_t t = 0; t < nt; ++t) {
            asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t *>((raw + (t * bh * R))),
                                 reinterpret_cast<__gm__ uint8_t *>(
                                     (x + (((token + t) * heads + head) * dim + dim - R))),
                                 static_cast<uint16_t>(nh), R * 2, 0, 0, (dim - R) * 2, 0);
        }
    auto rc = (reinterpret_cast<__ubuf__ float *>(raw) + (B * R / 2));
    auto rs = rc + B * (R / 2);
    asc_copy_gm2ub(rc, (c + (token * (R / 2))), uint32_t((nt * (R / 2)) * sizeof(float)));
    asc_copy_gm2ub(rs, (s + (token * (R / 2))), uint32_t((nt * (R / 2)) * sizeof(float)));
    set_flag(PIPE_MTE2, PIPE_V, ::event_t(ready));
}

// Prefix and rotated spans own disjoint 32B-aligned output cache lines.
// Copy only the current rotation tile, as in HEAD. The independent staging
// slot permits overlap with Vector work already issued for that tile.
// Event 6 publishes MTE2 -> MTE3 and retires MTE3 -> MTE2 before each reuse.
__aicore__ inline void copy_unrotated_prefix(__gm__ bfloat16_t *source,
                                             __gm__ bfloat16_t *destination, int64_t token,
                                             uint32_t head, uint32_t nt, uint32_t nh,
                                             uint32_t heads, uint32_t dim, uint32_t rotary_dim) {
    if (POSITION_QUANT_INPLACE || dim == rotary_dim)
        return;
    constexpr TEventID prefix_event = 6;
    auto staging =
        reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(position_quant::kRopeTileBytes));
    constexpr uint32_t tile_rows = position_quant::kPrefixTileRows;
    const uint32_t rows = nh == heads ? nt * nh : nh;
    const uint32_t loops = nh == heads ? 1 : nt;
    for (uint32_t t = 0; t < loops; ++t) {
        for (uint32_t row = 0; row < rows; row += tile_rows) {
            const uint16_t count = min(tile_rows, rows - row);
            const int64_t first = nh == heads ? token * heads + row
                                             : (token + t) * heads + head + row;
            asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ uint8_t *>(staging),
                                 reinterpret_cast<__gm__ uint8_t *>((source + (first * dim))), count,
                                 uint32_t((dim - rotary_dim) * sizeof(bfloat16_t)), 0, 0,
                                 uint32_t(rotary_dim * sizeof(bfloat16_t)), 0);
            native_device::pipe_fence<PIPE_MTE2, PIPE_MTE3, prefix_event>();
            asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t *>((destination + (first * dim))),
                                 reinterpret_cast<__ubuf__ uint8_t *>(staging), count,
                                 uint32_t((dim - rotary_dim) * sizeof(bfloat16_t)), 0, 0, 0,
                                 uint32_t(rotary_dim * sizeof(bfloat16_t)));
            native_device::pipe_fence<PIPE_MTE3, PIPE_MTE2, prefix_event>();
        }
    }
}

__global__ __aicore__ void rope_tail_kernel(GM_ADDR input, GM_ADDR cosine, GM_ADDR sine,
                                            GM_ADDR output, GM_ADDR tables, int64_t rows) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    InitSocState();
    SetSaturationFlag<SaturationMode::FLOAT>(false);
    constexpr uint32_t heads = POSITION_QUANT_HEADS;
    constexpr uint32_t dim = POSITION_QUANT_DIM;
    constexpr uint32_t rotary_dim = POSITION_QUANT_ROTARY_DIM;
    constexpr bool inverse = POSITION_QUANT_INVERSE;
    const int64_t tokens = rows / heads;
    constexpr uint32_t bh = POSITION_QUANT_TILE_HEADS, bt = POSITION_QUANT_TILE_TOKENS;
    constexpr int64_t head_groups = (heads + bh - 1) / bh;
    const int64_t jobs = ((tokens + bt - 1) / bt) * head_groups;
    constexpr int32_t group_shift = position_quant::head_group_shift(head_groups);
    constexpr uint32_t B = position_quant::kTileRowsCapacity;
    const uint32_t R = rotary_dim;
    using Arena = position_quant::RopeArena;
    auto f = reinterpret_cast<__ubuf__ float *>(uint64_t(Arena::floats)),
         sw = reinterpret_cast<__ubuf__ float *>(uint64_t(Arena::swapped)),
         tr = reinterpret_cast<__ubuf__ float *>(uint64_t(Arena::trig));
    auto swap_ids = reinterpret_cast<__ubuf__ int32_t *>(uint64_t(Arena::swap_ids)),
         trig_ids = reinterpret_cast<__ubuf__ int32_t *>(uint64_t(Arena::trig_ids));
    __gm__ bfloat16_t *x, *y;
    __gm__ float *c, *s;
    __gm__ int32_t *lut;
    x = (__gm__ bfloat16_t *)input;
    y = (__gm__ bfloat16_t *)output;
    c = (__gm__ float *)cosine;
    s = (__gm__ float *)sine;
    lut = (__gm__ int32_t *)tables;
    constexpr TEventID loaded = 2;
    asc_copy_gm2ub(swap_ids, lut, uint32_t((R) * sizeof(float)));
    asc_copy_gm2ub(trig_ids, (lut + (R)), uint32_t((R) * sizeof(float)));
    native_device::pipe_fence<PIPE_MTE2, PIPE_V, loaded>();
    uint32_t capacity = bh * bt * R;
    for (uint32_t n = R; n < capacity; n *= 2) {
        asc_add_scalar((swap_ids + (n)), swap_ids, static_cast<int32_t>(n * 4), n);
        pipe_barrier(PIPE_V);
    }
    for (uint32_t n = R; n < bt * R; n *= 2) {
        asc_add_scalar((trig_ids + (n)), trig_ids, static_cast<int32_t>(n / 2 * 4), n);
        pipe_barrier(PIPE_V);
    }
    rope_copy_in(reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(0)), x, c, s, GetBlockIdx(),
                 tokens, heads, bh, bt, head_groups, 0, dim, rotary_dim, group_shift);
    int32_t iteration = 0;
    for (int64_t job = GetBlockIdx(); job < jobs; job += kWorkers, ++iteration) {
        const int32_t slot = iteration % 2, next_slot = 1 - slot;
        auto raw = reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(slot * Arena::input_bytes));
        wait_flag(PIPE_MTE2, PIPE_V, ::event_t(slot));
        if (job + kWorkers < jobs) {
            if (iteration > 0)
                wait_flag(PIPE_V, PIPE_MTE2, ::event_t(next_slot));
            rope_copy_in(
                reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(next_slot * Arena::input_bytes)),
                x, c, s, job + kWorkers, tokens, heads, bh, bt, head_groups, next_slot, dim,
                rotary_dim, group_shift);
        }
        const int64_t token_group = group_shift >= 0 ? job >> group_shift : job / head_groups;
        int64_t token = token_group * bt;
        uint32_t head = (job - token_group * head_groups) * bh;
        uint32_t nt = tokens - token < bt ? tokens - token : bt,
                 nh = heads - head < bh ? heads - head : bh;
        uint32_t n = nt * bh * R;
        auto rc = (reinterpret_cast<__ubuf__ float *>(raw) + (B * R / 2));
        auto rs = rc + B * (R / 2);
        asc_bfloat162float(f, raw, n);
        pipe_barrier(PIPE_V);
        asc_gather(sw, f, reinterpret_cast<__ubuf__ uint32_t *>(swap_ids), n);
        asc_gather(tr, rc, reinterpret_cast<__ubuf__ uint32_t *>(trig_ids), nt * R);
        pipe_barrier(PIPE_V);
        // One expanded trig row per token; vector repeat broadcasts over heads.
        uint32_t loops = bh == 1 ? 1 : nt, repeats = bh == 1 ? nt : bh;
        asc_set_vector_mask(0, R == 64 ? ~uint64_t(0) : (uint64_t(1) << R) - 1);
        for (uint32_t t = 0; t < loops; ++t)
            asc_mul(f + t * bh * R, f + t * bh * R, tr + t * R, uint8_t(repeats), 1, 1, 1,
                    uint8_t(R / 8), uint8_t(R / 8), uint8_t(bh == 1 ? R / 8 : 0));
        pipe_barrier(PIPE_V);
        asc_gather(tr, rs, reinterpret_cast<__ubuf__ uint32_t *>(trig_ids), nt * R);
        asc_set_vector_mask(0, R == 64 ? ~uint64_t(0) : (uint64_t(1) << R) - 1);
        pipe_barrier(PIPE_V);
        for (uint32_t t = 0; t < loops; ++t)
            asc_mul(sw + t * bh * R, sw + t * bh * R, tr + t * R, uint8_t(repeats), 1, 1, 1,
                    uint8_t(R / 8), uint8_t(R / 8), uint8_t(bh == 1 ? R / 8 : 0));
        pipe_barrier(PIPE_V);
        const uint64_t active_mask = R == 64 ? ~uint64_t(0) : (uint64_t(1) << R) - 1;
        asc_set_vector_mask(0, (inverse ? 0xAAAAAAAAAAAAAAAAULL : 0x5555555555555555ULL) &
                                   active_mask);
        asc_mul_scalar(sw, sw, -1.f, uint8_t(nt * bh), 1, 1, uint8_t(R / 8), uint8_t(R / 8));
        pipe_barrier(PIPE_V);
        asc_add(f, f, sw, n);
        pipe_barrier(PIPE_V);
        if (iteration >= 2)
            wait_flag(PIPE_MTE3, PIPE_V, ::event_t(slot));
        auto out = reinterpret_cast<__ubuf__ bfloat16_t *>(
            uint64_t(Arena::outputs + slot * Arena::output_bytes));
        asc_float2bfloat16_rn(out, f, n);
        // Raw BF16 and compact trig may be loaded again only after their
        // final Vector reader retires; output has an independent lifetime.
        set_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
        native_device::pipe_fence<PIPE_V, PIPE_MTE3>(slot);
        if (nh == heads) {
            asc_copy_ub2gm_align(
                reinterpret_cast<__gm__ uint8_t *>((y + (token * heads * dim + dim - R))),
                reinterpret_cast<__ubuf__ uint8_t *>(out), static_cast<uint16_t>(nt * nh), R * 2, 0,
                0, 0, (dim - R) * 2);
        } else
            for (uint32_t t = 0; t < nt; ++t) {
                asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t *>(
                                         (y + (((token + t) * heads + head) * dim + dim - R))),
                                     reinterpret_cast<__ubuf__ uint8_t *>((out + (t * bh * R))),
                                     static_cast<uint16_t>(nh), R * 2, 0, 0, 0, (dim - R) * 2);
            }
        set_flag(PIPE_MTE3, PIPE_V, ::event_t(slot));
        copy_unrotated_prefix(x, y, token, head, nt, nh, heads, dim, R);
    }
    // Drain exactly the live input/output credits; no TPipe destructor or
    // implicit PIPE_ALL participates in the event protocol.
    for (int32_t slot = 0; slot < min(iteration, 2); ++slot) {
        wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
        wait_flag(PIPE_MTE3, PIPE_V, ::event_t(slot));
    }
    native_device::pipe_fence<PIPE_V, PIPE_S, position_quant::kRopeConsumerDrain>();
    native_device::pipe_fence<PIPE_MTE2, PIPE_S, position_quant::kRopeConsumerDrain>();
    native_device::pipe_fence<PIPE_MTE3, PIPE_S, position_quant::kRopeConsumerDrain>();
}
