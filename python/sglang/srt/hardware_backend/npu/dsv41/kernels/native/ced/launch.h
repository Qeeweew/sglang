#pragma once
#include <cstdint>
extern "C" void dsv41_ced_metadata_launch(void *stream, uint8_t *cu, uint8_t *row_map,
                                          uint8_t *requests, uint8_t *tail_cu, uint8_t *live,
                                          uint32_t batch, uint32_t window, int64_t capacity);
extern "C" void dsv41_ced_copy_launch(void *stream, uint32_t blocks, uint8_t *source,
                                      uint8_t *row_map, uint8_t *output, int64_t map_rows,
                                      int64_t source_rows, int64_t output_rows, int64_t row_bytes,
                                      uint32_t scatter, uint32_t fill_byte);
