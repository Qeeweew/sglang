#pragma once
#include <cstdint>
void split_rms_norm_launch(void *stream, uint32_t budget, uint8_t *input, uint8_t *first_weight,
                           uint8_t *second_weight, uint8_t *first_out, uint8_t *second_out,
                           uint32_t first_width, uint32_t second_width, float first_eps,
                           float second_eps, int64_t rows, int64_t stride);
extern "C" void dsv41_rms_norm_launch(void *stream, uint32_t blocks, uint8_t *x, uint8_t *weight,
                                      uint8_t *out, int64_t rows, uint32_t dimension,
                                      float epsilon);
extern "C" void dsv41_swiglu_launch(void *stream, uint32_t blocks, uint8_t *gate, uint8_t *up,
                                    uint8_t *count, uint8_t *out, int64_t rows, uint32_t dimension,
                                    float limit, bool packed, bool fp16);
