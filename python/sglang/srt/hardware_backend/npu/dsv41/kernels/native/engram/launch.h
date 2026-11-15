#pragma once
#include <cstdint>
extern "C" void dsv41_engram_launch(void *stream, uint32_t blocks, uint8_t *codes, uint8_t *scales,
                                    uint8_t *ids, uint8_t *out, uint8_t *tables, int64_t rows,
                                    int64_t vocab, uint32_t columns);
