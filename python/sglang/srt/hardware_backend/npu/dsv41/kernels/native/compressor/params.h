#pragma once
#include <cstdint>
#include <common/arch2201.h>
namespace compressor {
// C2 pooling owns seven width-sized FP32 payloads plus BF16 staging.
// 4096 elements use less than the arch2201 192KiB UB budget.
inline constexpr int64_t kWidthCapacityElements = 4096;
inline constexpr int64_t kDmaAlignmentBytes = 32;
inline constexpr int64_t kWidthAlignmentElements = kDmaAlignmentBytes / sizeof(uint16_t);
inline constexpr uint64_t kPrepareCapacityBytes =
    kWidthCapacityElements * (2 * sizeof(uint16_t) + 7 * sizeof(float)) + 3 * kDmaAlignmentBytes +
    512;
static_assert(kPrepareCapacityBytes <= arch2201::kUbBytes);
} // namespace compressor
