#pragma once
#include <w8a16/layout.h>

namespace w8a16 {
// Complete AIV stage ownership is documented in README's event/memory tables.
class DecodePipeline {
    __gm__ int8_t *weight_;
    __gm__ half *scales_;
    __gm__ Element *workspace_;
    template <class T> __aicore__ inline static __ubuf__ T *ub(int offset_bytes) {
        return reinterpret_cast<__ubuf__ T *>(uint64_t(offset_bytes));
    }
    __aicore__ inline void load_panel(const Tile &tile, int generation, int lane) {
        const int slot = generation % kSlots, start = generation % kPanels * kPanelK;
        const int column = tile.batch * kLogicalN + tile.column + lane * kColumns / kVectorActors;
        if (generation >= kSlots) {
            wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
            wait_flag(PIPE_V, PIPE_MTE2, ::event_t(slot + kSlots));
        }
        asc_copy_gm2ub(ub<int8_t>(slot * kMemory.local_elements),
            weight_ + int64_t(column) * kLogicalK + start * kC0,
            uint16_t(kLocalBlocks), uint16_t(kPanelK * kC0 / 32),
            uint16_t((kLogicalK - kPanelK) * kC0 / 32), 0);
        set_flag(PIPE_MTE2, PIPE_V, ::event_t(slot));
        asc_copy_gm2ub(ub<half>(kMemory.scale_offset + slot * kMemory.scale_elements * 2),
            scales_ + int64_t(column) * (kLogicalK / kGroup) + (start / kGroup) * kC0,
            uint16_t(kLocalBlocks), uint16_t(kPanelGroups),
            uint16_t(kLogicalK / kGroup - kPanelGroups), 0);
        set_flag(PIPE_MTE2, PIPE_V, ::event_t(slot + kSlots));
    }
    __aicore__ inline void decode_chunk(int block, int count) {
        const int generation = block / kSlots, inner = block % kSlots;
        const int slot = generation % kSlots, output_slot = block % kSlots;
        const int first = inner * kMemory.chunk_elements;
        if (block >= kSlots) wait_flag(PIPE_MTE3, PIPE_V, ::event_t(output_slot));
        if (!inner) {
            wait_flag(PIPE_MTE2, PIPE_V, ::event_t(slot));
            wait_flag(PIPE_MTE2, PIPE_V, ::event_t(slot + kSlots));
        }
        auto decoded = ub<half>(kMemory.decoded_offset + output_slot * kMemory.chunk_elements * 2);
        asc_int82half(decoded, ub<int8_t>(slot * kMemory.local_elements + first),
                      uint8_t(kMemory.chunk_elements / kVectorHalfElements), 1, 1, 8, 4);
        if (inner + 1 == kSlots && generation + kSlots < count)
            set_flag(PIPE_V, PIPE_MTE2, ::event_t(slot));
        pipe_barrier(PIPE_V);
        auto scale = ub<half>(kMemory.scale_offset + slot * kMemory.scale_elements * 2) +
                     first / (kScaleDepth * kC0) * kC0;
#pragma unroll
        for (int group = 0; group < kMemory.chunk_elements / (kScaleDepth * kC0); ++group)
            asc_mul(decoded + group * kScaleDepth * kC0, decoded + group * kScaleDepth * kC0,
                    scale + group * kC0, uint8_t(kScaleDepth * kC0 / kVectorHalfElements),
                    1, 1, 0, 8, 8, 0);
        if (inner + 1 == kSlots && generation + kSlots < count)
            set_flag(PIPE_V, PIPE_MTE2, ::event_t(slot + kSlots));
        if constexpr (kBf16) {
            pipe_barrier(PIPE_V);
            auto fp32 = ub<float>(kMemory.cast_offset);
            for (int offset = 0; offset < kMemory.chunk_elements; offset += kCastChunkElements) {
                const int elements = min(kCastChunkElements, kMemory.chunk_elements - offset);
                // Retire the previous read before overwriting FP32 scratch.
                // The last cast needs no V barrier: the output-ready event
                // orders MTE3, and the next decode's mul barrier orders V.
                if (offset) pipe_barrier(PIPE_V);
                asc_half2float(fp32, decoded + offset, uint8_t(elements / kVectorFloatElements), 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                asc_float2bfloat16_rn(reinterpret_cast<__ubuf__ bfloat16_t *>(decoded + offset), fp32,
                                     uint8_t(elements / kVectorFloatElements), 1, 1, 4, 8);
            }
        }
        set_flag(PIPE_V, PIPE_MTE3, ::event_t(output_slot));
    }
    __aicore__ inline void store_chunk(int block, int count, int core, int lane) {
        const int generation = block / kSlots, inner = block % kSlots;
        const int slot = generation % kSlots, output_slot = block % kSlots;
        if (!inner && generation >= kSlots) {
            if (slot == 0) CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE3>(kPanel0Free);
            else CrossCoreWaitFlag<kGroupFlagMode, PIPE_MTE3>(kPanel1Free);
        }
        wait_flag(PIPE_V, PIPE_MTE3, ::event_t(output_slot));
        const int64_t destination = (int64_t(core) * kSlots + slot) * kPanelK * kColumns +
                                     lane * kMemory.local_elements + inner * kMemory.chunk_elements;
        asc_copy_ub2gm(workspace_ + destination,
            ub<Element>(kMemory.decoded_offset + output_slot * kMemory.chunk_elements * 2),
            uint32_t(kMemory.chunk_elements * 2));
        if (inner + 1 == kSlots) {
            if (slot == 0) CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE3>(kPanel0Ready);
            else CrossCoreSetFlag<kGroupFlagMode, PIPE_MTE3>(kPanel1Ready);
        }
        if (block + kSlots < count * kSlots) set_flag(PIPE_MTE3, PIPE_V, ::event_t(output_slot));
    }
  public:
    __aicore__ inline void run(GM_ADDR weight, GM_ADDR scales, GM_ADDR workspace, int64_t rows) {
        weight_ = reinterpret_cast<__gm__ int8_t *>(weight);
        scales_ = reinterpret_cast<__gm__ half *>(scales);
        workspace_ = reinterpret_cast<__gm__ Element *>(workspace);
        const int core = GetBlockIdx() / kVectorActors, lane = GetSubBlockIdx();
        const int64_t jobs = (rows + kRows - 1) / kRows * (kLogicalN / kColumns) * kBatches;
        const int count = int((jobs - core + kWorkers - 1) / kWorkers) * kPanels;
        asc_set_mask_norm();
        asc_set_vector_mask(UINT64_MAX, UINT64_MAX);
        for (int step = 0; step < count * kSlots + kSlots; ++step) {
            const int generation = step / kSlots;
            if (step % kSlots == 0 && generation < count)
                load_panel(coordinates(core + int64_t(generation / kPanels) * kWorkers, rows), generation, lane);
            // Publish a completed chunk promptly. Only reuse of a GM panel
            // can block Scalar: queue the other output slot's V work first
            // in that case, matching the grouped GEMM handoff schedule.
            const bool gm_reuse_wait = step >= (kSlots + 1) * kSlots &&
                                       (step - kSlots) % kSlots == 0;
            if (step >= kSlots && !gm_reuse_wait) store_chunk(step - kSlots, count, core, lane);
            if (step >= 1 && step - 1 < count * kSlots) decode_chunk(step - 1, count);
            if (gm_reuse_wait) store_chunk(step - kSlots, count, core, lane);
        }
        set_flag(PIPE_MTE3, PIPE_S, ::event_t(kDrainEvent));
        wait_flag(PIPE_MTE3, PIPE_S, ::event_t(kDrainEvent));
    }
};
} // namespace w8a16
