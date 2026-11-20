#pragma once
#include <common/device.h>
#include <qli/rank_pipeline.h>
#include <qli/schedule.h>

namespace qli {
class VectorPipeline {
    const Sources &args_;
    Workspace workspace_;
    uint32_t lane_;
    __aicore__ __ubuf__ float *tokens(uint32_t local) const {
        return ub_ptr<float>(UbLayout::tokens) + (local * 2 * kKeep);
    }
    __aicore__ __ubuf__ float *blocks(uint32_t local) const {
        return ub_ptr<float>(UbLayout::mode) + (local * 2 * kKeyTileCapacity);
    }
    __aicore__ __ubuf__ float *candidates(uint32_t local) const {
        return ub_ptr<float>(UbLayout::mode) + (local * kKeyTileCapacity);
    }
    __aicore__ void weights(const Task &task) {
        if (lane_ == 0) {
            auto diagonal = ub_ptr<half>(UbLayout::aux);
            auto weights = (diagonal + (kCubeRows * kMatrixRows));
            auto scales = (weights + (kMatrixRows));
            asc_duplicate(diagonal, half(0), kCubeRows * kMatrixRows + 2 * kMatrixRows);
            pipe_barrier(PIPE_V);
            native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
            asc_copy_gm2ub(weights,
                           (gm_pointer<half>(args_.weights) + (uint64_t(task.row) * kHeads)),
                           uint32_t((task.queries * kHeads) * sizeof(half)));
            asc_copy_gm2ub(scales,
                           (gm_pointer<half>(args_.q_scale) + (uint64_t(task.row) * kHeads)),
                           uint32_t((task.queries * kHeads) * sizeof(half)));
            native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
            asc_mul(weights, weights, scales, kMatrixRows); // explicit FP16 weight*scale rounding
            pipe_barrier(PIPE_V);
            for (uint32_t query = 0; query < task.queries; ++query)
                asc_add_scalar((diagonal + (query * kMatrixRows + query * kHeads)),
                               (weights + (query * kHeads)), half(0), kHeads);
            pipe_barrier(PIPE_V);
            native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
            asc_copy_ub2gm(workspace_.weights(), diagonal,
                           uint32_t((kCubeRows * kMatrixRows) * sizeof(half)));
        }
        CrossCoreSetFlag<2, PIPE_MTE3>(WeightReady);
    }
    __aicore__ void prefetch_weights(TaskCursor &cursor, uint64_t &ordinal) {
        Task future{};
        while (cursor.locate(ordinal, future)) {
            ordinal += kScoreWorkers;
            if (future.key_begin >= future.key_end)
                continue;
            weights(future);
            // WeightReady publishes to the Cube, but current ranking also
            // needs to reuse this UB scratch. Retire its own DMA reader here.
            native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
            return;
        }
    }
    __aicore__ void initialize_candidates(uint32_t row, uint32_t local) {
        auto integers = ub_ptr<int32_t>(UbLayout::aux);
        auto values = ub_ptr<float>(UbLayout::scores);
        auto ids = ub_ptr<int32_t>(UbLayout::indices);
        native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
        asc_copy_gm2ub(integers,
                       (gm_pointer<int32_t>(args_.candidates) + (uint64_t(row) * kKeyTileCapacity)),
                       uint32_t((kKeyTileCapacity) * sizeof(int32_t)));
        native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
        asc_int322float(values, integers, kKeyTileCapacity);
        RankPipeline::sequence(ids, 0, kKeyTileCapacity);
        pipe_barrier(PIPE_V);
        auto sorted = RankPipeline::sort(values, reinterpret_cast<__ubuf__ uint32_t *>(ids),
                                         kKeyTileCapacity);
        RankPipeline::extract_values(candidates(local), sorted, kKeyTileCapacity);
    }
    __aicore__ uint32_t count_ge(__ubuf__ float *sorted, int32_t value) {
        uint32_t begin = 0, end = kKeyTileCapacity;
        while (begin < end) {
            const uint32_t middle = (begin + end) / 2;
            if (sorted[middle] >= float(value))
                begin = middle + 1;
            else
                end = middle;
        }
        return begin;
    }
    __aicore__ void candidate_penalty(uint32_t local, uint32_t base, uint32_t count) {
        auto blocks = ub_ptr<float>(UbLayout::aux);
        auto expanded = ub_ptr<float>(UbLayout::float_scales);
        asc_duplicate(blocks, float(0), kKeyTileCapacity / kCandidateBlockTokens);
        native_device::pipe_fence<PIPE_V, PIPE_S, kImmediateEvent>();
        auto sorted = candidates(local);
        const uint32_t first_block = base / kCandidateBlockTokens;
        const uint32_t first =
            count_ge(sorted, first_block + kKeyTileCapacity / kCandidateBlockTokens);
        const uint32_t last = count_ge(sorted, first_block);
        // 910B has no vector Scatter. Sorted membership bounds visit only the
        // candidates in this tile; scalar writes publish through S->V.
        for (uint32_t index = first; index < last; ++index)
            blocks[uint32_t(int32_t(sorted[index])) - first_block] = float(1);
        native_device::pipe_fence<PIPE_S, PIPE_V, kImmediateEvent>();
        asc_brcb(reinterpret_cast<__ubuf__ uint32_t *>(expanded),
                 reinterpret_cast<__ubuf__ uint32_t *>(blocks), 1, 8,
                 uint8_t(kKeyTileCapacity / kCandidateBlockTokens / 8));
        pipe_barrier(PIPE_V);
        asc_add_scalar(expanded, expanded, float(-1), count);
        pipe_barrier(PIPE_V);
        asc_abs(expanded, expanded, count);
        pipe_barrier(PIPE_V);
        auto invalid = ub_ptr<int32_t>(UbLayout::aux);
        asc_float2int32_rn(invalid, expanded, count);
        pipe_barrier(PIPE_V);
        auto temporary = ub_ptr<int32_t>(uint32_t(UbLayout::aux + kKeyTileCapacity * sizeof(int32_t)));
        auto bits = ub_ptr<int32_t>(UbLayout::scores);
        asc_duplicate(temporary, kNegativePenaltyBits, count);
        pipe_barrier(PIPE_V);
        asc_sub(temporary, temporary, bits, count);
        pipe_barrier(PIPE_V);
        asc_mul(temporary, temporary, invalid, count);
        pipe_barrier(PIPE_V);
        asc_add(bits, bits, temporary, count);
        pipe_barrier(PIPE_V);
    }
    __aicore__ void load_scales(const Task &task, uint32_t begin, uint32_t count) {
        auto scales = ub_ptr<half>(UbLayout::half_scales);
        uint32_t destination = 0;
        while (destination < count) {
            const uint32_t position = begin + destination;
            const uint32_t column = position / kPageRows;
            const uint32_t within = position - column * kPageRows;
            const uint32_t page =
                gm_pointer<int32_t>(args_.pages)[uint64_t(task.batch) * kPageColumns + column];
            const uint32_t rows = min(count - destination, kPageRows - within);
            asc_copy_gm2ub_align(
                (scales + (destination)),
                (gm_pointer<half>(args_.k_scale) + (uint64_t(page) * kScalePageStride + within)), 1,
                rows * uint32_t(sizeof(half)), 0, 0, 0, 0);
            destination += rows;
        }
    }
    __aicore__ void rank_blocks(uint32_t local, uint32_t begin, uint32_t valid, uint32_t length,
                                uint32_t count) {
        auto values = ub_ptr<float>(UbLayout::scores);
        auto block_scores = ub_ptr<float>(UbLayout::half_scales);
        // K scales are consumed before this stage. Their 4-KiB span now owns
        // at most512 block scores and512 block IDs, separate from token IDs.
        auto ids = ub_ptr<int32_t>(
            UbLayout::half_scales +
            RankPipeline::sort_count(kKeyTileCapacity / kCandidateBlockTokens) * sizeof(float));
        const uint32_t block_count = RankPipeline::sort_count((count + 7) / 8);
        asc_duplicate(reinterpret_cast<__ubuf__ int32_t *>(block_scores), kNegativeInfinityBits,
                      block_count);
        pipe_barrier(PIPE_V);
        {
            asc_set_vector_mask(0, ~uint64_t(0));
            asc_datablock_reduce_max(block_scores, values,
                                     uint8_t((count + kVectorWords - 1) / kVectorWords), 1, 1,
                                     kVectorBlocks);
        };
        pipe_barrier(PIPE_V);
        // The latest visible block is forced into the source candidate list.
        const uint32_t latest = valid ? (valid - 1) / 8 : UINT32_MAX;
        const uint32_t first = begin / 8;
        if (latest >= first && latest < first + block_count) {
            native_device::pipe_fence<PIPE_V, PIPE_S, kImmediateEvent>();
            reinterpret_cast<__ubuf__ int32_t *>(block_scores)[latest - first] = 0x7f800000;
            native_device::pipe_fence<PIPE_S, PIPE_V, kImmediateEvent>();
        }
        RankPipeline::mask_invalid(block_scores, ids, first, (min(valid, begin + length) + 7) / 8,
                                   block_count);
        auto sorted = RankPipeline::sort(block_scores, reinterpret_cast<__ubuf__ uint32_t *>(ids),
                                         block_count);
        RankPipeline::merge(blocks(local), kKeyTileCapacity, sorted, block_count);
    }
    __aicore__ void store(uint32_t row, uint32_t split, uint32_t local) {
        if (kKeySplits > 1) {
            native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
            asc_copy_ub2gm(workspace_.ranks(row, split), tokens(local),
                           uint32_t((kKeep * 2) * sizeof(float)));
            if constexpr (kMode == kSourceCandidates)
                asc_copy_ub2gm((workspace_.ranks(row, split) + (kKeep * 2)), blocks(local),
                               uint32_t((kKeyTileCapacity * 2) * sizeof(float)));
        } else {
            auto ids = ub_ptr<int32_t>(UbLayout::indices);
            RankPipeline::extract_ids(ids, tokens(local), kTopk);
            native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
            asc_copy_ub2gm((gm_pointer<int32_t>(args_.indices) + (uint64_t(row) * kTopk)), ids,
                           uint32_t((kTopk) * sizeof(int32_t)));
            native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
            if constexpr (kMode == kSourceCandidates) {
                RankPipeline::extract_ids(ids, blocks(local), kKeyTileCapacity);
                native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
                asc_copy_ub2gm(
                    (gm_pointer<int32_t>(args_.candidate_out) + (uint64_t(row) * kKeyTileCapacity)),
                    ids, uint32_t((kKeyTileCapacity) * sizeof(int32_t)));
            }
        }
        native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
    }

  public:
    __aicore__ VectorPipeline(const Sources &args, uint32_t core, uint32_t lane)
        : args_(args), workspace_{args, core}, lane_(lane) {}
    __aicore__ void run() {
        RankPipeline::initialize_sequence();
        TaskCursor cursor(args_);
        TaskCursor weight_cursor(args_);
        uint64_t weight_ordinal = workspace_.core;
        prefetch_weights(weight_cursor, weight_ordinal);
        Task task{};
        uint64_t consumed = 0;
        for (uint64_t ordinal = workspace_.core; cursor.locate(ordinal, task);
             ordinal += kScoreWorkers) {
            for (uint32_t local = 0, query = lane_; query < task.queries;
                 query += kVectorActors, ++local) {
                RankPipeline::empty(tokens(local), kKeep);
                if constexpr (kMode == kSourceCandidates)
                    RankPipeline::empty(blocks(local), kKeyTileCapacity);
                if constexpr (kMode == kConsumerCandidates)
                    initialize_candidates(task.row + query, local);
            }
            if (task.key_begin < task.key_end) {
                // Import completion returns the one GM weight span. Prepare
                // the following nonempty task before ranking this one's score.
                CrossCoreWaitFlag<2, PIPE_MTE3>(WeightCredit);
                native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
                prefetch_weights(weight_cursor, weight_ordinal);
                for (uint32_t begin = task.key_begin; begin < task.key_end;
                     begin += kKeyTileCapacity, ++consumed) {
                    const uint32_t slot = consumed % kScoreSlots;
                    CrossCoreWaitFlag<2, PIPE_MTE2>(slot ? ScoreReady1 : ScoreReady0);
                    const uint32_t length = min(uint32_t(kKeyTileCapacity), task.key_end - begin);
                    for (uint32_t local = 0, query = lane_; query < task.queries;
                         query += kVectorActors, ++local) {
                        const uint32_t valid = task.valid_keys(query);
                        const uint32_t count = RankPipeline::sort_count(length);
                        auto scores = ub_ptr<float>(UbLayout::scores);
                        auto float_scales = ub_ptr<float>(UbLayout::float_scales);
                        auto ids = ub_ptr<int32_t>(UbLayout::indices);
                        native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
                        asc_copy_gm2ub_align(scores,
                                             (workspace_.scores(slot) + (query * kKeyTileCapacity)),
                                             1, length * uint32_t(sizeof(float)), 0, 0, 0, 0);
                        load_scales(task, begin, length);
                        native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
                        asc_half2float(float_scales, ub_ptr<half>(UbLayout::half_scales),
                                       length);
                        pipe_barrier(PIPE_V);
                        asc_mul(scores, scores, float_scales, length);
                        pipe_barrier(PIPE_V);
                        if constexpr (kMode == kConsumerCandidates)
                            candidate_penalty(local, begin, count);
                        RankPipeline::mask_invalid(scores, ids, begin, min(valid, begin + length),
                                                   count);
                        // Block ranking consumes unsorted token scores; token
                        // sort owns distinct buffers and can follow it safely.
                        if constexpr (kMode == kSourceCandidates)
                            rank_blocks(local, begin, valid, length, count);
                        auto sorted = RankPipeline::sort(
                            scores, reinterpret_cast<__ubuf__ uint32_t *>(ids), count);
                        RankPipeline::merge(tokens(local), kKeep, sorted, count);
                    }
                    CrossCoreSetFlag<2, PIPE_MTE2>(slot ? ScoreCredit1 : ScoreCredit0);
                }
            }
            for (uint32_t local = 0, query = lane_; query < task.queries;
                 query += kVectorActors, ++local)
                store(task.row + query, task.split, local);
        }
        pipe_barrier(PIPE_ALL);
    }
};
} // namespace qli
