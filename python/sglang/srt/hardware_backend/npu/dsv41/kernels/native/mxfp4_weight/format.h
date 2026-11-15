#pragma once
#include <cstdint>
namespace mxfp4_weight {
inline constexpr uint32_t kNibbleBits = 4;
inline constexpr uint32_t kByteBits = 8;
inline constexpr uint32_t kCodesPerByte = kByteBits / kNibbleBits;
inline constexpr uint32_t kNibblesPerHalfword = sizeof(uint16_t) * kByteBits / kNibbleBits;
inline constexpr uint32_t kScaleGroup = 32;  // checkpoint E8M0 group contract
inline constexpr uint32_t kPackColumns = 32 / sizeof(uint16_t); // FP16 NZ C0, not a tile/stripe choice
// The startup decoder's fixed UB budget supports at most 512 K elements.
inline constexpr uint32_t kMaxPanelRows = 512;
inline constexpr uint32_t kCodesPerInt32 = sizeof(int32_t) * kCodesPerByte;
inline constexpr uint32_t kPackedBytesPerScale = kScaleGroup / kCodesPerByte;
inline constexpr uint32_t kAlphabet = 1U << kByteBits;
inline constexpr uint32_t kLutEntries = 2 * kAlphabet; // rank pairs, FP16 scales
inline constexpr uint32_t kLutBytes = kLutEntries * sizeof(uint16_t);
inline constexpr uint32_t kTableIndexOffset = kLutBytes / sizeof(uint32_t);
constexpr uint64_t table_bytes(uint64_t panel_rows) {
    const uint64_t rank_indices = panel_rows / kCodesPerByte * (kPackColumns / kNibblesPerHalfword);
    const uint64_t output_indices = panel_rows * (kPackColumns / kNibblesPerHalfword);
    const uint64_t scale_indices = panel_rows / kScaleGroup * kPackColumns;
    return kLutBytes + (rank_indices + output_indices + scale_indices) * sizeof(uint32_t);
}
inline constexpr uint16_t kNibbleMask = (1U << kNibbleBits) - 1;
// Compact S/2: FP16 high byte. Subnormals are boosted by 2^10 and sign-tagged.
inline constexpr int32_t kE8M0Bias = 127;
inline constexpr int32_t kFp16Bias = 15;
inline constexpr int32_t kFp16FractionBits = 10;
inline constexpr int32_t kScaleHalvingExponent = 1;
inline constexpr int32_t kScaleExponentOffset = kE8M0Bias + kScaleHalvingExponent - kFp16Bias;
inline constexpr int32_t kNormalMinCode = kE8M0Bias + kScaleHalvingExponent + 1 - kFp16Bias;
inline constexpr int32_t kHighByteExponentUnit = 1 << (kFp16FractionBits - kByteBits);
inline constexpr int32_t kCompactSubnormalOffset =
    (1 << (kByteBits - 1)) + kFp16FractionBits * kHighByteExponentUnit;
} // namespace mxfp4_weight
