#pragma once
#include <kernel_operator.h>
#include <compressor/params.h>

namespace compressor {
// Private address maps. Payload and metadata lifetimes are disjoint; no allocator or hidden events.
template <int Ratio> struct PrepareArena {
    static_assert(Ratio == 1 || Ratio == 2);
    static constexpr uint32_t kBf16Bytes = kWidthCapacityElements * sizeof(uint16_t);
    static constexpr uint32_t kFloatBytes = kWidthCapacityElements * sizeof(float);
    static constexpr uint32_t kRaw = 0;
    static constexpr uint32_t kPacked = kRaw + kBf16Bytes;
    static constexpr uint32_t kFirst = kPacked + kBf16Bytes;
    static constexpr uint32_t kSecond = kFirst + kFloatBytes;
    static constexpr uint32_t kScore0 = kSecond + (Ratio == 2 ? kFloatBytes : 0);
    static constexpr uint32_t kScore1 = kScore0 + kFloatBytes;
    static constexpr uint32_t kWork = kScore1 + kFloatBytes;
    static constexpr uint32_t kWeight = kWork + (Ratio == 2 ? kFloatBytes : 0);
    static constexpr uint32_t kStats = kWeight + kFloatBytes;
    static constexpr uint32_t kRefinement = kStats + kDmaAlignmentBytes;
    static constexpr uint32_t kReciprocal = kRefinement + 2 * kDmaAlignmentBytes;
    static constexpr uint32_t kMetadata = kReciprocal + (Ratio == 2 ? kFloatBytes : 0);
    static constexpr uint32_t kMetadataBytes = 512; // two 32-entry INT64 position spans
    static_assert(kMetadata + kMetadataBytes <= arch2201::kUbBytes);
};

} // namespace compressor
