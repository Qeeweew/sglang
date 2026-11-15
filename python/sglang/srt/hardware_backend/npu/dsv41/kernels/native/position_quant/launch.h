#pragma once
#include <cstdint>
extern "C" void dsv41_position_quant_launch(void *stream, uint32_t blocks, uint8_t *x, uint8_t *cos,
                                            uint8_t *sin, uint8_t *y, uint8_t *scales,
                                            uint8_t *tables, int64_t rows, uint32_t heads,
                                            uint32_t dim, uint32_t mode, bool inverse,
                                            uint32_t rotary_dim);
extern "C" void dsv41_cache_scatter_launch(void *stream, uint32_t blocks, uint8_t *values,
                                           uint8_t *scales, uint8_t *slots, uint8_t *cache,
                                           uint8_t *cache_scales, int64_t rows, int64_t capacity,
                                           uint32_t dim, uint32_t dtype);
extern "C" void dsv41_rope_cache_gather_launch(void *stream, uint32_t blocks, uint8_t *cos_cache,
                                               uint8_t *sin_cache, uint8_t *positions,
                                               uint8_t *input_slots, uint8_t *cos_out,
                                               uint8_t *sin_out, uint8_t *valid_out,
                                               uint8_t *slots_out, int64_t rows, int64_t cache_rows,
                                               uint32_t trig_columns);
