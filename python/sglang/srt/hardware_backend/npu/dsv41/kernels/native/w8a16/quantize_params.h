#pragma once
#include <cstdint>
#include <common/arch2201.h>
#if defined(__NPU_ARCH__)
#define W8A16_QUANT_HOST_DEVICE __aicore__
#else
#define W8A16_QUANT_HOST_DEVICE
#endif
namespace w8a16 {
struct QuantizeSources {
    uintptr_t input, quantized, scales;
    int64_t rows;
};
// At most 8192 arithmetic elements occupy one Vector tile. This capacity is
// independent of M/K/B: the cyclic task loop covers arbitrary input lengths.
inline constexpr uint32_t kQuantCapacityElements = 8 * 1024;
inline constexpr uint32_t kQuantMaxGroupElements = 256;
inline constexpr uint32_t kQuantSlots = 2;
struct QuantizeLayout {
    uint32_t elements, scale_pitch;
    uint32_t values_bytes, maxima_bytes, temporary_bytes, rounded_bytes;
    uint32_t scale_fp32_bytes, quantized_bytes, scale_fp16_bytes;
    uint32_t integer_bytes, partial_bytes, broadcast_bytes, total_bytes;
    W8A16_QUANT_HOST_DEVICE constexpr QuantizeLayout(uint32_t rows, uint32_t group) {
        elements = rows * group;
        scale_pitch = rows < 16 ? 16 : rows;
        values_bytes = 4 * elements;
        maxima_bytes = values_bytes + 4 * elements;
        // Two row banks separate reduction data and subsequent Vector sources.
        temporary_bytes = maxima_bytes + 2 * rows * sizeof(float);
        rounded_bytes = temporary_bytes + elements * sizeof(float);
        scale_fp32_bytes = rounded_bytes + elements * sizeof(uint16_t);
        quantized_bytes = scale_fp32_bytes + rows * sizeof(float);
        scale_fp16_bytes = quantized_bytes + kQuantSlots * elements;
        integer_bytes = scale_fp16_bytes + kQuantSlots * scale_pitch * sizeof(uint16_t);
        // Reserve two INT16 banks to preserve the conversion/broadcast phases.
        partial_bytes = integer_bytes + elements * sizeof(float);
        broadcast_bytes = partial_bytes + elements / 8 * sizeof(float);
        total_bytes = broadcast_bytes + rows * 8 * sizeof(float);
    }
};
inline constexpr uint32_t kQuantUbBytes = QuantizeLayout(32, 256).total_bytes;
static_assert(kQuantUbBytes <= arch2201::kUbBytes);
} // namespace w8a16

#undef W8A16_QUANT_HOST_DEVICE
