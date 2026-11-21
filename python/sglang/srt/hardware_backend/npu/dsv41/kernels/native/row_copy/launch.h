#pragma once
#include <cstdint>

void dsv41_row_copy_launch(void *stream, uint32_t workers, uint8_t *source, uint8_t *row_map,
                           uint8_t *output, int64_t map_rows, int64_t source_rows,
                           int64_t output_rows, int64_t row_bytes, bool scatter,
                           uint32_t fill_byte);
