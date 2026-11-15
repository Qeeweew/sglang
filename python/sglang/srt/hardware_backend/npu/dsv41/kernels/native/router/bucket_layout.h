#pragma once
#include <cstdint>
namespace router_bucket {
inline constexpr uint32_t kChunk = 8192;
inline constexpr uint32_t kDmaBlockBytes = 32;
inline constexpr uint32_t kCacheLineBytes = 64;
inline constexpr uint32_t kVectorRepeatBytes = 256;
inline constexpr uint32_t kHistogramAlignment = kCacheLineBytes / sizeof(int32_t);
inline constexpr uint32_t kHistogramRows = 4;
// A worker receives at least one DMA block; only launch extent changes.
inline constexpr uint32_t kSlotsPerTask = kDmaBlockBytes / sizeof(int32_t);
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint32_t histogram_stride(uint32_t experts) {
    return (experts + kHistogramAlignment - 1) / kHistogramAlignment * kHistogramAlignment;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint32_t aligned(uint32_t bytes) {
    return (bytes + kDmaBlockBytes - 1) / kDmaBlockBytes * kDmaBlockBytes;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint32_t owned_offset(uint32_t experts) {
    const uint64_t vector_pitch = (experts + 63) / 64 * 64;
    const uint64_t tables = 13 * vector_pitch * sizeof(int32_t) +
        aligned((experts + 1) * sizeof(int32_t)) + (experts + 1) * 4ULL * sizeof(int32_t);
    constexpr uint32_t order_end = 16 * kChunk + kChunk / 8;
    return aligned(tables > order_end ? tables : order_end);
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr uint64_t table_bytes(uint32_t experts) {
    return owned_offset(experts) + 4ULL * experts;
}
} // namespace router_bucket
