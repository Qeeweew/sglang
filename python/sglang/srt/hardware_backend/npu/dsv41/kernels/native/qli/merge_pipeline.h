#pragma once
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
            auto accumulator = ub_pointer<float>(UbLayout::tokens);
            auto blocks = ub_pointer<float>(UbLayout::mode);
            auto incoming = ub_pointer<float>(UbLayout::sort0);
            RankPipeline::empty(accumulator, kKeep);
            if constexpr (kMode == kSourceCandidates)
                RankPipeline::empty(blocks, kKeyTileCapacity);
            for (uint32_t split = 0; split < kKeySplits; ++split) {
                {
                    set_flag(PIPE_V, PIPE_MTE2, ::event_t(kImmediateEvent));
                    wait_flag(PIPE_V, PIPE_MTE2, ::event_t(kImmediateEvent));
                }
                asc_copy_gm2ub(incoming, workspace_.ranks(row, split),
                               uint32_t((kKeep * 2) * sizeof(float)));
                {
                    set_flag(PIPE_MTE2, PIPE_V, ::event_t(kImmediateEvent));
                    wait_flag(PIPE_MTE2, PIPE_V, ::event_t(kImmediateEvent));
                }
                RankPipeline::merge(accumulator, kKeep, incoming, kKeep);
                if constexpr (kMode == kSourceCandidates) {
                    {
                        set_flag(PIPE_V, PIPE_MTE2, ::event_t(kImmediateEvent));
                        wait_flag(PIPE_V, PIPE_MTE2, ::event_t(kImmediateEvent));
                    }
                    asc_copy_gm2ub(incoming, (workspace_.ranks(row, split) + (kKeep * 2)),
                                   uint32_t((kKeyTileCapacity * 2) * sizeof(float)));
                    {
                        set_flag(PIPE_MTE2, PIPE_V, ::event_t(kImmediateEvent));
                        wait_flag(PIPE_MTE2, PIPE_V, ::event_t(kImmediateEvent));
                    }
                    RankPipeline::merge(blocks, kKeyTileCapacity, incoming, kKeyTileCapacity);
                }
            }
            auto ids = ub_pointer<int32_t>(UbLayout::indices);
            RankPipeline::extract_ids(ids, accumulator, kTopk);
            {
                set_flag(PIPE_V, PIPE_MTE3, ::event_t(kImmediateEvent));
                wait_flag(PIPE_V, PIPE_MTE3, ::event_t(kImmediateEvent));
            }
            asc_copy_ub2gm((gm_pointer<int32_t>(args_.indices) + (uint64_t(row) * kTopk)), ids,
                           uint32_t((kTopk) * sizeof(int32_t)));
            {
                set_flag(PIPE_MTE3, PIPE_V, ::event_t(kImmediateEvent));
                wait_flag(PIPE_MTE3, PIPE_V, ::event_t(kImmediateEvent));
            }
            if constexpr (kMode == kSourceCandidates) {
                RankPipeline::extract_ids(ids, blocks, kKeyTileCapacity);
                {
                    set_flag(PIPE_V, PIPE_MTE3, ::event_t(kImmediateEvent));
                    wait_flag(PIPE_V, PIPE_MTE3, ::event_t(kImmediateEvent));
                }
                asc_copy_ub2gm(
                    (gm_pointer<int32_t>(args_.candidate_out) + (uint64_t(row) * kKeyTileCapacity)),
                    ids, uint32_t((kKeyTileCapacity) * sizeof(int32_t)));
                {
                    set_flag(PIPE_MTE3, PIPE_V, ::event_t(kImmediateEvent));
                    wait_flag(PIPE_MTE3, PIPE_V, ::event_t(kImmediateEvent));
                }
            }
        }
        pipe_barrier(PIPE_ALL);
    }
};
} // namespace qli
