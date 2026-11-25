#pragma once
#include <cstdint>

namespace h32_quant {
inline constexpr uint32_t kPanelElements = 256;
inline constexpr uint32_t kMaxWidth = 8192;

constexpr bool valid_width(int64_t width) {
    return width >= kPanelElements && width <= kMaxWidth && width % kPanelElements == 0;
}
void launch(void *stream, uint32_t workers, uint32_t width, bool bf16, bool interleaved,
            uint8_t *input, uint8_t *codes, uint8_t *scales, uint8_t *count, int64_t rows);
} // namespace h32_quant
