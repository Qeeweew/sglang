#pragma once
#include <cstdint>
#include <common/arch2201.h>

namespace elementwise {
#if defined(__NPU_ARCH__)
#define ELEMENTWISE_LAYOUT_API __aicore__
#else
#define ELEMENTWISE_LAYOUT_API
#endif
inline constexpr uint32_t kSlots = 2;
#if defined(ELEMENTWISE_SINGLE_TILE)
inline constexpr uint32_t kActiveSlots = 1;
#else
inline constexpr uint32_t kActiveSlots = kSlots;
#endif
// arch2201 public Vector/DMA widths, checked against SDK in row_pipeline.h.
inline constexpr uint32_t kVectorRepeatBytes = 256, kDmaBlockBytes = 32;
inline constexpr uint32_t kRepeatBlocks = kVectorRepeatBytes / kDmaBlockBytes;
inline constexpr uint32_t kOperandPhaseBytes = kDmaBlockBytes;
inline constexpr uint32_t kFp32BlockElements = kDmaBlockBytes / sizeof(float);
inline constexpr uint32_t kFp32VectorElements = kVectorRepeatBytes / sizeof(float);
struct RmsStatistics {
    static constexpr uint32_t inverse = kFp32BlockElements;
};
inline constexpr uint32_t kReductionPanelElements = kFp32VectorElements * kFp32VectorElements;
ELEMENTWISE_LAYOUT_API inline constexpr uint32_t align_elements(uint32_t n) {
    return (n + kFp32VectorElements - 1) / kFp32VectorElements * kFp32VectorElements;
}
struct RmsLayout {
    uint32_t raw, packed, weight_raw, weight, value, work, scratch, stat, bytes;
    ELEMENTWISE_LAYOUT_API constexpr explicit RmsLayout(uint32_t width, uint32_t slots = kSlots)
        : raw(0), packed(raw + slots * width * sizeof(uint16_t)),
          weight_raw(packed + slots * width * sizeof(uint16_t)),
          // Independent FP32 operands occupy different DMA-block phases;
          // the resource arena stays fixed for all runtime row widths.
          weight(weight_raw + width * sizeof(uint16_t) + kOperandPhaseBytes),
          value(weight + width * sizeof(float) + kOperandPhaseBytes),
          work(value + width * sizeof(float) + kOperandPhaseBytes),
          scratch(work + width * sizeof(float) + kOperandPhaseBytes),
          stat(scratch + 2 * kFp32VectorElements * sizeof(float) + kOperandPhaseBytes),
          bytes(stat + 2 * kFp32VectorElements * sizeof(float)) {}
};
struct SwiGluLayout {
    uint32_t gate_raw, up_raw, packed, gate, up, negative, work, ones, bytes;
    ELEMENTWISE_LAYOUT_API constexpr explicit SwiGluLayout(uint32_t tile_elements,
                                                           uint32_t slots = kSlots)
        : gate_raw(0), up_raw(gate_raw + slots * tile_elements * sizeof(uint16_t)),
          packed(up_raw + slots * tile_elements * sizeof(uint16_t)),
          gate(packed + slots * tile_elements * sizeof(uint16_t)),
          up(gate + tile_elements * sizeof(float)), negative(up + tile_elements * sizeof(float)),
          work(negative + tile_elements * sizeof(float)),
          ones(work + tile_elements * sizeof(float)),
          bytes(ones + kFp32VectorElements * sizeof(float)) {}
};
// One bounded arena covers all legal tiles; dimensions select active bytes.
// This is a resource capacity, not a model width or a JIT specialization.
inline constexpr uint32_t kSwiGluBytesPerElement =
    3 * kSlots * sizeof(uint16_t) + 4 * sizeof(float);
inline constexpr uint32_t kSwiGluTileCapacity =
    (arch2201::kUbBytes - kFp32VectorElements * sizeof(float)) / kSwiGluBytesPerElement /
    kFp32VectorElements * kFp32VectorElements;
static_assert(SwiGluLayout(kSwiGluTileCapacity).bytes <= arch2201::kUbBytes);
ELEMENTWISE_LAYOUT_API inline constexpr uint32_t tile_rows(uint64_t capacity, uint32_t width,
                                                           uint32_t cores) {
    return capacity >= uint64_t(kSlots) * cores &&
                   SwiGluLayout(kSlots * align_elements(width)).bytes <= arch2201::kUbBytes
               ? kSlots
               : 1;
}
#undef ELEMENTWISE_LAYOUT_API
} // namespace elementwise
