#pragma once
#include <cstdint>
#include <common/arch2201.h>
#include <qli/params.h>

#if defined(__CCE_AICORE__)
#define QLI_HOST_DEVICE __aicore__
#else
#define QLI_HOST_DEVICE
#endif
namespace qli {
// CANN9.1 arch2201: MMAD m0/n0=16, DMA block=32B, vector repeat=256B.
inline constexpr uint32_t kCubeRows = 16;
inline constexpr uint32_t kDmaBytes = 32;
inline constexpr uint32_t kVectorBytes = 256;
inline constexpr uint32_t kVectorWords = kVectorBytes / sizeof(uint32_t);
inline constexpr uint32_t kVectorBlocks = kVectorBytes / kDmaBytes;
inline constexpr uint32_t kDmaWords = kDmaBytes / sizeof(uint32_t);
inline constexpr uint32_t kSortRun = 32; // SDK Sort32: number of score/index pairs.
inline constexpr uint32_t kSortMergeInputs = 4;
inline constexpr uint32_t kRankMergeInputs = 2;
inline constexpr uint32_t kScoreSlots = 2;
inline constexpr uint32_t kKeySlots = 2;
inline constexpr uint32_t kReluSlots = 2;
inline constexpr uint32_t kC0Partitions = 2;
inline constexpr uint32_t kVectorActors = 2;
inline constexpr uint32_t kRankFields = 2;
inline constexpr uint32_t kCandidateBlockTokens = 8;
inline constexpr uint32_t kKeyTileCapacity = 2048; // public source-candidate/rank-list contract
// Host geometry admission reserves eight key fractals. The actual device N is
// rederived from its simultaneous operand/result budgets in state.h.
inline constexpr uint32_t kAdmissionKeyRows = 8 * kCubeRows;
// QK and weighted FP32 outputs own disjoint half-L0C spans. Derive the legal
// flattened query/head capacity from that allocation, not a model dimension.
inline constexpr uint32_t kQueryHeadCapacity =
    arch2201::kL0CBytes / kC0Partitions / (kAdmissionKeyRows * sizeof(int32_t));
inline constexpr uint32_t kHeadCapacity = kQueryHeadCapacity;
inline constexpr uint32_t kDepthCapacity = arch2201::kL0BBytes / kAdmissionKeyRows / sizeof(int8_t);
inline constexpr uint32_t kInt8DepthAlignment = kDmaBytes / sizeof(int8_t);
inline constexpr uint32_t kQueryTileRows = 4; // two rank rows per AIV in the private arena
inline constexpr uint32_t kDefaultTopk = 512;
inline constexpr uint32_t kSourceCandidates = 1;
inline constexpr uint32_t kConsumerCandidates = 2;
inline constexpr uint32_t kNoCandidates = 3;
QLI_HOST_DEVICE constexpr uint64_t align32(uint64_t value) {
    return (value + kSortRun - 1) / kSortRun * kSortRun;
}

// Preserve the existing two-query-tile target and one-wave key partition policy.
QLI_HOST_DEVICE constexpr uint32_t plan_query_rows(uint64_t rows, uint32_t workers,
                                                   uint32_t heads) {
    constexpr uint32_t kTargetQueryTilesPerWorker = 2;
    uint64_t target = rows / (uint64_t(workers) * kTargetQueryTilesPerWorker);
    uint32_t tile = 1;
    while (tile * 2 <= kQueryTileRows && tile * 2 <= kQueryHeadCapacity / heads &&
           tile * 2 <= target)
        tile *= 2;
    return tile;
}
QLI_HOST_DEVICE constexpr uint32_t plan_key_splits(uint64_t rows, uint32_t workers, uint32_t heads,
                                                   uint64_t key_capacity) {
    const uint32_t queries = plan_query_rows(rows, workers, heads);
    const uint64_t groups = rows ? (rows + queries - 1) / queries : 1;
    const uint64_t chunks =
        key_capacity ? (key_capacity + kKeyTileCapacity - 1) / kKeyTileCapacity : 1;
    const uint64_t wave = workers / groups;
    const uint64_t splits = chunks < wave ? chunks : wave;
    return splits ? splits : 1;
}

// All core-local score rings precede the diagonal head-weight matrices.
// K-split rank lists follow them; each row/split owns a disjoint sorted list.
QLI_HOST_DEVICE constexpr uint64_t score_bytes(const Config &c) {
    return uint64_t(c.cores) * kScoreSlots * c.query_tile_rows * kKeyTileCapacity * sizeof(float);
}
QLI_HOST_DEVICE constexpr uint64_t weight_bytes(const Config &c) {
    return uint64_t(c.cores) * kCubeRows * c.query_tile_rows * c.heads * sizeof(uint16_t);
}
QLI_HOST_DEVICE constexpr uint64_t rank_list_words(const Config &c) {
    return kRankFields *
           (align32(c.topk) + (c.candidate_mode == kSourceCandidates ? kKeyTileCapacity : 0));
}
QLI_HOST_DEVICE constexpr uint64_t workspace_bytes(const Config &c) {
    return score_bytes(c) + weight_bytes(c) +
           (c.key_splits > 1 ? uint64_t(c.rows) * c.key_splits * rank_list_words(c) * sizeof(float)
                             : 0);
}
} // namespace qli
#undef QLI_HOST_DEVICE
