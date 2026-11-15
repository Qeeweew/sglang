#pragma once
#include <cstdint>
#include <common/arch2201.h>

namespace engram_ops {
#if defined(__NPU_ARCH__)
#define ENGRAM_OPS_LAYOUT_API __aicore__
#else
#define ENGRAM_OPS_LAYOUT_API
#endif
inline constexpr uint32_t kDmaBlockBytes = 32;
inline constexpr uint32_t kVectorRepeatBytes = 256;
inline constexpr uint32_t kRepeatBlocks = kVectorRepeatBytes / kDmaBlockBytes;
inline constexpr uint32_t kFp32BlockElements = kDmaBlockBytes / sizeof(float);
inline constexpr uint32_t kFp32VectorElements = kVectorRepeatBytes / sizeof(float);
ENGRAM_OPS_LAYOUT_API inline constexpr uint32_t align_bytes(uint32_t bytes) {
    return (bytes + kDmaBlockBytes - 1) / kDmaBlockBytes * kDmaBlockBytes;
}
// cu/end and slots are signed INT32. Saturating the corresponding bound
// to that index domain preserves every device comparison, including when
// the backing tensor itself has more rows. Batch remains a full INT64 extent.
struct CommitLaunchParams {
    int64_t batch;
    uint32_t rows, history_rows, history_tokens;
};
static_assert(sizeof(CommitLaunchParams) == 24); // Five64-bit pointers +24B =64B.
ENGRAM_OPS_LAYOUT_API inline constexpr bool valid_commit(int32_t slot, int32_t accepted,
                                                         int32_t begin, int32_t end, uint32_t rows,
                                                         uint32_t history_rows) {
    // Reject invalid signs/order before the subtraction. In the remaining
    // domain end-begin and begin+accepted both fit signed INT32 exactly.
    return slot >= 0 && uint32_t(slot) < history_rows && accepted > 0 && begin >= 0 &&
           end >= begin && uint32_t(end) <= rows && accepted <= end - begin;
}
#if !defined(__NPU_ARCH__)
inline CommitLaunchParams commit_dimensions(int64_t rows, int64_t batch, int64_t history_rows,
                                            uint32_t history_tokens) {
    constexpr int64_t last_end = INT32_MAX;
    constexpr int64_t slot_domain = int64_t(UINT32_C(1) << 31);
    return {batch, uint32_t(rows < last_end ? rows : last_end),
            uint32_t(history_rows < slot_domain ? history_rows : slot_domain), history_tokens};
}
#endif
// Only the result row resides in UB; Scalar parameter reads use GM cache.
struct HashLayout {
    uint32_t result, bytes;
    ENGRAM_OPS_LAYOUT_API constexpr explicit HashLayout(uint32_t columns)
        : result(0), bytes(align_bytes(columns * sizeof(int64_t))) {}
};
// Three BF16 vectors and five FP32 vectors, followed by FP32 scalar lanes.
// Width is an input; the resource contract bounds only the allocated arena.
struct GateLayout {
    static constexpr uint32_t kBytesPerColumn = 3 * sizeof(uint16_t) + 5 * sizeof(float);
    static constexpr uint32_t kStatBytes = kFp32VectorElements * sizeof(float);
    uint32_t raw_hidden, raw_key, packed, hidden, key, weight, work, scratch, stat, bytes;
    ENGRAM_OPS_LAYOUT_API constexpr explicit GateLayout(uint32_t width)
        : raw_hidden(0), raw_key(raw_hidden + width * sizeof(uint16_t)),
          packed(raw_key + width * sizeof(uint16_t)), hidden(packed + width * sizeof(uint16_t)),
          key(hidden + width * sizeof(float)), weight(key + width * sizeof(float)),
          work(weight + width * sizeof(float)), scratch(work + width * sizeof(float)),
          stat(scratch + width * sizeof(float)), bytes(stat + kStatBytes) {}
};
#undef ENGRAM_OPS_LAYOUT_API
} // namespace engram_ops
