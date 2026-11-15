#pragma once
#include <cstdint>
#include <common/arch2201.h>
namespace router_layout {
inline constexpr uint32_t kMaxTileRows = 8, kPreferredTileElements = 2048;
inline constexpr uint32_t kExpertAlignment = 64, kDataBlockBytes = 32;
inline constexpr uint32_t kMaxTopK = 64;
inline constexpr float kMinimumNormalF32 = 0x1p-126f, kWeightDenominatorEpsilon = 1e-20f;
struct ArenaLayout {
    uint32_t bytes = 0, score = 0, bias = 0, alternate = 0, exponential = 0, one_plus_exp = 0, logarithm = 0;
    uint32_t denominator = 0, ranking = 0, indices = 0, selected = 0, selected_ids = 0, score_offsets = 0;
    uint32_t sums = 0, pair_offsets = 0, packing = 0, output = 0, output_stride = 0;
#if defined(__CCE_AICORE__)
    __aicore__
#endif
    constexpr uint32_t take(uint32_t size) {
        const uint32_t at = bytes;
        bytes += (size + kDataBlockBytes - 1) / kDataBlockBytes * kDataBlockBytes;
        return at;
    }
#if defined(__CCE_AICORE__)
    __aicore__
#endif
    constexpr ArenaLayout(uint32_t rows, uint32_t experts, uint32_t topk) {
        const uint32_t tensor = rows * experts * sizeof(float);
        score = take(tensor); bias = take(tensor); alternate = take(tensor);
        // After score/rank publication these adjacent spans become sort A/B,
        // each holding [value,INT32 label] pairs for the complete row tile.
        exponential = take(tensor); one_plus_exp = take(tensor);
        logarithm = take(tensor); denominator = take(tensor);
        ranking = take(tensor); indices = take(tensor);
        selected = take(rows * kMaxTopK * sizeof(float));
        selected_ids = take(rows * kMaxTopK * sizeof(int32_t));
        score_offsets = take(rows * kMaxTopK * sizeof(uint32_t));
        sums = take(kDataBlockBytes); pair_offsets = take(kMaxTopK * sizeof(uint32_t));
        packing = take(rows * topk * sizeof(uint32_t));
        output_stride = (rows * topk + 7) / 8 * 8;
        output = take(2 * output_stride * sizeof(float));
    }
};
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint64_t arena_bytes(uint32_t rows, uint32_t experts, uint32_t topk) {
    return 2 * uint64_t(rows) * experts * sizeof(float) + ArenaLayout(rows, experts, topk).bytes;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint32_t tile_rows(uint32_t experts, uint32_t topk) {
    uint32_t rows = kMaxTileRows;
    while (rows > 1 && (rows * uint64_t(experts) > kPreferredTileElements ||
                       arena_bytes(rows, experts, topk) > arch2201::kUbBytes)) rows /= 2;
    return rows;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint32_t task_blocks(int64_t rows, uint32_t experts, uint32_t topk, uint32_t workers) {
    const int64_t groups = (rows + tile_rows(experts, topk) - 1) / tile_rows(experts, topk);
    const int64_t per_worker = (groups + workers - 1) / workers;
    return groups ? (groups + per_worker - 1) / per_worker : 0;
}
} // namespace router_layout
