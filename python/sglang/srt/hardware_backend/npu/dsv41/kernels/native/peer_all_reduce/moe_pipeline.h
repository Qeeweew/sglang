#pragma once
#include <peer_all_reduce/protocol.h>

namespace peer_moe {
namespace p = peer_all_reduce;
namespace s = peer_protocol;
// Attention TP is a finite layout choice; protocol and exported capacity are
// identical for DP4/TP2 and DP8/TP1 (at most 512 global token rows).
#ifndef PEER_ATTN_TP
#define PEER_ATTN_TP 2
#endif
constexpr uint32_t H = 5120, T = p::kMoeTile, TP = PEER_ATTN_TP, DP = 8 / TP;
static_assert(TP == 1 || TP == 2);
struct View {
    GM_ADDR local;
    GM_ADDR table;
    uint32_t rank, rows, shard_rows, owned;
    uint64_t epoch;
    __aicore__ View(GM_ADDR l, GM_ADDR peers, uint32_t r, uint32_t n)
        : local(l), table(peers), rank(r), rows(n), shard_rows((n + TP - 1) / TP),
          owned(n - r % TP * shard_rows < shard_rows ? n - r % TP * shard_rows : shard_rows),
          epoch(s::begin(l, peers, 8)) {}
    __aicore__ uint32_t first_row(uint32_t r) const {
        return r / TP * rows + r % TP * shard_rows;
    }
    __aicore__ uint32_t owned_rows(uint32_t r) const {
        const uint32_t left = rows - r % TP * shard_rows;
        return left < shard_rows ? left : shard_rows;
    }
    __aicore__ uint32_t owner(uint32_t row) const {
        return rank / TP * TP + row / shard_rows;
    }
    __aicore__ GM_ADDR region(uint32_t r) const {
        return reinterpret_cast<GM_ADDR>(reinterpret_cast<__gm__ uint64_t *>(table)[r]);
    }
    __aicore__ __gm__ bfloat16_t *payload(uint32_t r) const {
        return reinterpret_cast<__gm__ bfloat16_t *>(region(r) + p::kControlBytes);
    }
    __aicore__ __gm__ bfloat16_t *returned(uint32_t r) const {
        return reinterpret_cast<__gm__ bfloat16_t *>(region(r) + p::kMoeReturnOffset);
    }
    __aicore__ void ready(uint32_t kind) const {
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        s::publish(s::signal(local, kind, AscendC::GetBlockIdx()), epoch);
    }
    __aicore__ void all_ready(uint32_t kind) const {
        s::aggregate(local, kind, epoch);
    }
    __aicore__ void finish() const {
        s::finish(local, epoch);
    }
};

// Fixed whole-token address ownership survives odd rows and shape changes.
__aicore__ inline void gather(const View &v, GM_ADDR input, GM_ADDR output) {
    auto raw = reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(0));
    const uint32_t channel = AscendC::GetBlockIdx(), stride = p::kWorkers * T;
    auto x = reinterpret_cast<__gm__ bfloat16_t *>(input) + (v.rank % TP) * v.shard_rows * H;
    const uint32_t destination = v.first_row(v.rank) * H;
    for (uint32_t offset = channel * T; offset < v.owned * H; offset += stride) {
        asc_copy_gm2ub(raw, x + offset, T * 2);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        // Push the owned row directly into every destination's compact gather
        // plane. Rotate destinations to avoid synchronized many-to-one bursts.
        for (uint32_t step = 0; step < 8; ++step)
            asc_copy_ub2gm(v.payload((v.rank + step) % 8) + destination + offset, raw, T * 2);
        s::stored();
    }
    v.ready(1);
    for (uint32_t rank = 0; rank < 8; ++rank) {
        s::await(s::signal(v.region(rank), 1, channel), v.epoch);
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        const uint32_t count = v.owned_rows(rank);
        auto out = reinterpret_cast<__gm__ bfloat16_t *>(output) + v.first_row(rank) * H;
        for (uint32_t offset = channel * T; offset < count * H; offset += stride) {
            const uint32_t source = v.first_row(rank) * H;
            asc_copy_gm2ub(raw, v.payload(v.rank) + source + offset, T * 2);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            asc_copy_ub2gm(out + offset, raw, T * 2);
            s::stored();
        }
    }
}

template <bool Finalize>
__aicore__ inline void publish_partial(const View &v, GM_ADDR input, GM_ADDR inverse, GM_ADDR ids,
                                       GM_ADDR weights, uint32_t topk) {
    auto raw = reinterpret_cast<__ubuf__ half *>(uint64_t(0));
    auto packed = reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(2 * T));
    auto fp = reinterpret_cast<__ubuf__ float *>(uint64_t(4 * T));
    auto sum = reinterpret_cast<__ubuf__ float *>(uint64_t(8 * T));
    const uint32_t stride = p::kWorkers * T;
    for (uint32_t offset = AscendC::GetBlockIdx() * T; offset < DP * v.rows * H; offset += stride) {
        if constexpr (Finalize) {
            const uint32_t row = offset / H, col = offset % H;
            asc_duplicate(sum, 0.f, T);
            pipe_barrier(PIPE_V);
            for (uint32_t k = 0; k < topk; ++k) {
                const uint32_t route = row * topk + k;
                if (reinterpret_cast<__gm__ int32_t *>(ids)[route] < 0)
                    continue;
                const int32_t slot = reinterpret_cast<__gm__ int32_t *>(inverse)[route];
                const float weight = reinterpret_cast<__gm__ float *>(weights)[route];
                asc_copy_gm2ub(
                    raw, reinterpret_cast<__gm__ half *>(input) + uint64_t(slot) * H + col, T * 2);
                s::loaded();
                asc_half2float(fp, raw, T);
                pipe_barrier(PIPE_V);
                asc_mul_scalar(fp, fp, weight, T);
                pipe_barrier(PIPE_V);
                asc_add(sum, sum, fp, T);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }
            asc_float2bfloat16_rn(packed, sum, T);
            s::vector_store();
        } else {
            asc_copy_gm2ub(packed, reinterpret_cast<__gm__ bfloat16_t *>(input) + offset, T * 2);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        }
        asc_copy_ub2gm(v.payload(v.rank) + offset, packed, T * 2);
        s::stored();
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    v.ready(1);
}

__aicore__ inline void reduce_owned(const View &v, GM_ADDR shared) {
    auto raw = reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(0));
    auto fp = reinterpret_cast<__ubuf__ float *>(uint64_t((TP == 1 ? 16 : 4) * T));
    auto sum = reinterpret_cast<__ubuf__ float *>(uint64_t((TP == 1 ? 20 : 8) * T));
    const uint32_t channel = AscendC::GetBlockIdx(), stride = p::kWorkers * T;
    const uint32_t first = v.first_row(v.rank) * H;
    const uint32_t source_channel = (first / T + channel) % p::kWorkers;
    for (uint32_t rank = 0; rank < 8; ++rank)
        s::await(s::signal(v.region(rank), 1, source_channel), v.epoch);
    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    const uint32_t owned = v.owned < v.shard_rows ? v.owned : v.shard_rows;
    for (uint32_t offset = channel * T; offset < owned * H; offset += stride) {
        asc_duplicate(sum, 0.f, T);
        pipe_barrier(PIPE_V);
        if constexpr (TP == 1) {
            // Stage all eight packets in disjoint UB slots. Rotate the DMA
            // source order across owners to avoid a rank-0 read hotspot;
            // keep the subsequent FP32 arithmetic strictly rank ordered.
            // 80 KiB BF16 + 20 KiB conversion + 20 KiB accumulator = 120 KiB.
            for (uint32_t step = 0; step < 8; ++step) {
                const uint32_t source = (v.rank + step) % 8;
                asc_copy_gm2ub(raw + source * T, v.payload(source) + first + offset, T * 2);
            }
            s::loaded();
            for (uint32_t rank = 0; rank < 8; ++rank) {
                asc_bfloat162float(fp, raw + rank * T, T);
                pipe_barrier(PIPE_V);
                asc_add(sum, sum, fp, T);
                pipe_barrier(PIPE_V);
            }
        } else {
            for (uint32_t rank = 0; rank < 8; ++rank) {
                asc_copy_gm2ub(raw, v.payload(rank) + first + offset, T * 2);
                s::loaded();
                asc_bfloat162float(fp, raw, T);
                pipe_barrier(PIPE_V);
                asc_add(sum, sum, fp, T);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            }
        }
        // Preserve the BF16 expert-sum boundary before adding shared once.
        asc_float2bfloat16_rn(raw, sum, T);
        pipe_barrier(PIPE_V);
        if (shared) {
            asc_bfloat162float(sum, raw, T);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            asc_copy_gm2ub(raw, reinterpret_cast<__gm__ bfloat16_t *>(shared) + offset, T * 2);
            s::loaded();
            asc_bfloat162float(fp, raw, T);
            pipe_barrier(PIPE_V);
            asc_add(sum, sum, fp, T);
            pipe_barrier(PIPE_V);
            asc_float2bfloat16_rn(raw, sum, T);
            pipe_barrier(PIPE_V);
        }
        s::vector_store();
        asc_copy_ub2gm(v.returned(v.rank) + offset, raw, T * 2);
        s::stored();
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    v.ready(3);
}

__aicore__ inline void restore(const View &v, GM_ADDR output) {
    auto raw = reinterpret_cast<__ubuf__ bfloat16_t *>(uint64_t(0));
    const uint32_t channel = AscendC::GetBlockIdx();
    for (uint32_t rank = v.rank / TP * TP; rank < v.rank / TP * TP + TP; ++rank) {
        s::await(s::signal(v.region(rank), 3, channel), v.epoch);
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        const uint32_t count = v.owned_rows(rank);
        auto out = reinterpret_cast<__gm__ bfloat16_t *>(output) + rank % TP * v.shard_rows * H;
        for (uint32_t offset = channel * T; offset < count * H; offset += p::kWorkers * T) {
            asc_copy_gm2ub(raw, v.returned(rank) + offset, T * 2);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            asc_copy_ub2gm(out + offset, raw, T * 2);
            s::stored();
        }
    }
}
} // namespace peer_moe
