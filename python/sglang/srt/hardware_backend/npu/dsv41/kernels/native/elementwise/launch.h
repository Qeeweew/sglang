#pragma once
#include <cstdint>
extern "C" void dsv41_rms_norm_launch(void *stream, uint32_t blocks, uint8_t *x, uint8_t *weight,
                                      uint8_t *out, int64_t rows, uint32_t dimension,
                                      float epsilon);
extern "C" void dsv41_swiglu_launch(void *stream, uint32_t blocks, uint8_t *gate, uint8_t *up,
                                    uint8_t *count, uint8_t *out, int64_t rows, uint32_t dimension,
                                    float limit, bool packed, bool fp16);
