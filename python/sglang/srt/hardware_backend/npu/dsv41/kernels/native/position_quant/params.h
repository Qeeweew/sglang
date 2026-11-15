#pragma once
#include <cstdint>
#include <common/arch2201.h>
namespace position_quant {
// These are allocation capacities. Actual width, rotary span and head count
// are checked at bind time and passed to the same device entries.
inline constexpr uint32_t kWidthCapacityElements = 2048;
inline constexpr uint32_t kRotaryCapacityElements = 64;
inline constexpr uint32_t kTileRowsCapacity = 64;
// HEAD's staging group bounds each strided prefix DMA to eight rows.
inline constexpr uint32_t kPrefixTileRows = 8;
inline constexpr uint32_t kDmaAlignmentBytes = 32;
// Adjacent pairs are the rotary storage format. Scale groups own one full
// FP16 cache line; gather groups own one full INT32 validity cache line.
inline constexpr uint32_t kRotaryPairElements = 2;
inline constexpr uint32_t kScaleGroupRows = kDmaAlignmentBytes / sizeof(uint16_t);
inline constexpr uint32_t kGatherGroupRows = kDmaAlignmentBytes / sizeof(int32_t);
inline constexpr uint64_t kRopeTileBytes =
    2 * (kTileRowsCapacity * kRotaryCapacityElements * sizeof(uint16_t) +
         kTileRowsCapacity * (kRotaryCapacityElements / 2) * sizeof(float) * 2) +
    2 * kTileRowsCapacity * kRotaryCapacityElements * sizeof(uint16_t) +
    5 * kTileRowsCapacity * kRotaryCapacityElements * sizeof(float);
inline constexpr uint64_t kPrefixCapacityBytes =
    kPrefixTileRows * (kWidthCapacityElements - kDmaAlignmentBytes / sizeof(uint16_t)) * sizeof(uint16_t);
static_assert(kRopeTileBytes + kPrefixCapacityBytes <= arch2201::kUbBytes);

// A pure plan shared by host prepare and device constexpr code. Keep the
// HEAD tile and actor mapping; no tensor metadata is read.
#if defined(__NPU_ARCH__)
#define POSITION_QUANT_PLAN_API __aicore__
#else
#define POSITION_QUANT_PLAN_API
#endif
POSITION_QUANT_PLAN_API constexpr int32_t head_group_shift(int64_t groups) {
    int32_t shift = 0;
    while ((int64_t(1) << shift) < groups)
        ++shift;
    return (int64_t(1) << shift) == groups ? shift : -1;
}
struct RopePlan {
    uint32_t heads, tokens;
    int64_t jobs, head_groups;
    int32_t group_shift;
};
POSITION_QUANT_PLAN_API constexpr RopePlan rope_plan(int64_t rows, uint32_t heads,
                                                     uint32_t worker_budget) {
    const int64_t target = (rows + worker_budget - 1) / worker_budget;
    uint32_t tile = 1;
    while (tile < kTileRowsCapacity && tile * 2 <= target)
        tile *= 2;
    uint32_t tile_heads = tile;
    while (tile_heads > heads)
        tile_heads /= 2;
    const uint32_t tile_tokens = tile / tile_heads;
    const int64_t tokens = rows / heads;
    const int64_t head_groups = (heads + tile_heads - 1) / tile_heads;
    const int64_t jobs = ((tokens + tile_tokens - 1) / tile_tokens) * head_groups;
    const int32_t group_shift = head_group_shift(head_groups);
    return {
        tile_heads, tile_tokens,
        jobs,       head_groups, group_shift};
}
#undef POSITION_QUANT_PLAN_API
} // namespace position_quant
