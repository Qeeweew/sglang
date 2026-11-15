#pragma once
#include <cstdint>

namespace engram {
// E4M3FN and E8M0 are byte encodings; one E8M0 exponent covers32 codes.
// Decode capacity is independent of the embedding row's runtime width.
inline constexpr uint32_t kByteValues = uint32_t(1) << 8;
inline constexpr uint32_t kScaleGroupElements = 32;
inline constexpr uint32_t kDecodeCapacityElements = 256; // vector decode capacity, not row width
inline constexpr uint32_t kCodebookElements = 2 * kByteValues;
inline constexpr uint32_t kDmaBlockBytes = 32;
inline constexpr uint32_t align_bytes(uint32_t value) {
    return (value + kDmaBlockBytes - 1) / kDmaBlockBytes * kDmaBlockBytes;
}
struct DecodeLayout {
    static constexpr uint32_t codes_bytes = kDecodeCapacityElements;
    static constexpr uint32_t exponent_capacity = kDmaBlockBytes;
    static constexpr uint32_t half_bytes = kDecodeCapacityElements * sizeof(uint16_t);
    // Keep the following half/index/FP32 vectors on their original full
    // cast-vector boundary. Tight packing of the byte-scale tail changes
    // the UB bank phase of every Gather operand for a saving of only224B.
    static constexpr uint32_t raw_bytes = half_bytes;
    static_assert(raw_bytes >= codes_bytes + exponent_capacity);
    static constexpr uint32_t vector_bytes = kDecodeCapacityElements * sizeof(float);
    static constexpr uint32_t output_bytes = kDecodeCapacityElements * sizeof(uint16_t);
    static constexpr uint32_t codebook_bytes = kCodebookElements * sizeof(float);
    static constexpr uint32_t group_offsets_bytes = kDecodeCapacityElements * sizeof(uint32_t);
    static constexpr uint32_t raw = 0, half = raw + raw_bytes, offsets = half + half_bytes;
    static constexpr uint32_t value = offsets + vector_bytes, scale = value + vector_bytes;
    static constexpr uint32_t packed = scale + vector_bytes, lookup = packed + output_bytes;
    static constexpr uint32_t groups = lookup + codebook_bytes,
                              total_bytes = groups + group_offsets_bytes;
};
} // namespace engram
