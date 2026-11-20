#pragma once
#include <cstdint>

namespace h32_quant {
constexpr bool valid_width(int64_t width) {
    return width >= 256 && width <= 8192 && width % 256 == 0;
}
void launch(void *stream, uint32_t workers, uint32_t width, bool bf16, bool interleaved,
            uint8_t *input, uint8_t *indices, uint8_t *codes, uint8_t *scales, int64_t rows,
            uint8_t *count, int64_t shared_rows);
} // namespace h32_quant
