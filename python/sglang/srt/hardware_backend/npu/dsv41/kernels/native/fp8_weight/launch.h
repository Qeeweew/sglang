#pragma once
#include <cstdint>
extern "C" void dsv41_fp8_weight_launch(void *stream, uint32_t blocks, uint8_t *codes,
                                        uint8_t *scales, uint8_t *out, uint8_t *lut,
                                        uint8_t *groups, int64_t rows, int64_t cols);
