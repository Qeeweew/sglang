#pragma once
#include <cstdint>

namespace dspark_markov {
constexpr uint32_t kRowsPerTile = 4;
constexpr uint32_t kColumns = 2048;
void launch(void *stream, uint8_t *candidates, uint8_t *out, int64_t ranks, int64_t rows,
            bool packet = false);
void select(void *stream, uint8_t *base, uint8_t *bias, uint8_t *out, int64_t rows,
            int64_t width, int64_t parts, int64_t base_stride, int64_t bias_stride,
            uint32_t start, uint32_t formats);
} // namespace dspark_markov
