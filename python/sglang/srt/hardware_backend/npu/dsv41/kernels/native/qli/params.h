#pragma once
#include <cstdint>

namespace qli {
// Checked host geometry. Device uses the corresponding actual DeepJIT constants.
struct Config {
    uint32_t rows, batch, heads, depth, page_rows, page_columns;
    uint32_t key_page_stride, scale_page_stride, ratio, topk, candidate_mode;
    uint32_t cores, query_tile_rows, key_splits;
};
static_assert(sizeof(Config) == 56);
struct Sources {
    uintptr_t q, k, weights, q_scale, k_scale, cu, lengths, residual, pages;
    uintptr_t candidates, indices, candidate_out, workspace;
    uint32_t rows, batch;
    uintptr_t positions = 0;
};
} // namespace qli
