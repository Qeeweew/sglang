#pragma once
#include <common/device.h>
#include <qli/rank_pipeline.h>

namespace qli {
// A separate launch is the global publication boundary between K partitions.
// There is no device-wide spin barrier or workspace atomics.
class MergePipeline {
    const Sources &args_;
    Workspace workspace_;

  public:
    __aicore__ explicit MergePipeline(const Sources &args) : args_(args), workspace_{args, 0} {}
    __aicore__ void run() {
        for (uint32_t row = GetBlockIdx(); row < args_.rows; row += kWorkers) {
            auto accumulator = ub_ptr<float>(UbLayout::tokens);
            auto blocks = ub_ptr<float>(UbLayout::mode);
            auto incoming = ub_ptr<float>(UbLayout::sort0);
            RankPipeline::empty(accumulator, kKeep);
            if constexpr (kMode == kSourceCandidates)
                RankPipeline::empty(blocks, kKeyTileCapacity);
            for (uint32_t split = 0; split < kKeySplits; ++split) {
                native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
                asc_copy_gm2ub(incoming, workspace_.ranks(row, split),
                               uint32_t((kKeep * 2) * sizeof(float)));
                native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
                RankPipeline::merge(accumulator, kKeep, incoming, kKeep);
                if constexpr (kMode == kSourceCandidates) {
                    native_device::pipe_fence<PIPE_V, PIPE_MTE2, kImmediateEvent>();
                    asc_copy_gm2ub(incoming, (workspace_.ranks(row, split) + (kKeep * 2)),
                                   uint32_t((kKeyTileCapacity * 2) * sizeof(float)));
                    native_device::pipe_fence<PIPE_MTE2, PIPE_V, kImmediateEvent>();
                    RankPipeline::merge(blocks, kKeyTileCapacity, incoming, kKeyTileCapacity);
                }
            }
            auto ids = ub_ptr<int32_t>(UbLayout::indices);
            RankPipeline::extract_ids(ids, accumulator, kTopk);
            native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
            asc_copy_ub2gm((gm_pointer<int32_t>(args_.indices) + (uint64_t(row) * kTopk)), ids,
                           uint32_t((kTopk) * sizeof(int32_t)));
            native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
            if constexpr (kMode == kSourceCandidates) {
                RankPipeline::extract_ids(ids, blocks, kKeyTileCapacity);
                native_device::pipe_fence<PIPE_V, PIPE_MTE3, kImmediateEvent>();
                asc_copy_ub2gm(
                    (gm_pointer<int32_t>(args_.candidate_out) + (uint64_t(row) * kKeyTileCapacity)),
                    ids, uint32_t((kKeyTileCapacity) * sizeof(int32_t)));
                native_device::pipe_fence<PIPE_MTE3, PIPE_V, kImmediateEvent>();
            }
        }
        pipe_barrier(PIPE_ALL);
    }
};
} // namespace qli
