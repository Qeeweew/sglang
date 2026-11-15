#pragma once
#include <cstdint>
extern "C" void dsv41_mxfp4_pack(void *stream, uint32_t blocks, void *codes, void *scales,
                                 void *tables, void *packed, void *scale_output, int64_t k,
                                 int64_t n, int64_t panel_k, bool compact);
