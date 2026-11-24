#pragma once
#include <cstdint>
#include <common/arch2201.h>

namespace attention {
enum class Mode : int32_t { Window = 0, Sparse = 1, Draft = 2 };
// Allocation capacities are independent of all actual copy/MMAD extents.
inline constexpr int32_t kHeadCapacity = 64;
inline constexpr int32_t kWidthCapacity = 512;
inline constexpr int32_t kSelectedCapacity = 512;
inline constexpr int32_t kDotRows = 128;
inline constexpr int32_t kFeaturePanelColumns = 256;
inline constexpr int32_t kMatrixFractalRows = arch2201::kMmadFractal;
inline constexpr int32_t kMatrixFractalElements = kMatrixFractalRows * kMatrixFractalRows;
inline constexpr int32_t kSmallQueryTileRows = 2 * kMatrixFractalRows;
inline constexpr int32_t kSparseWindowCapacity = kDotRows;
inline constexpr int32_t kSparseRowCapacity = kSparseWindowCapacity + kSelectedCapacity;
inline constexpr int32_t kSequenceCapacity = 4096;
struct Config {
    int32_t tokens, heads, width, batch, columns, compressed_columns;
    int32_t page_rows, compressed_page_rows, window_rows, selected_rows;
    int32_t capacity, ratio, mode, cores, tile_rows, coordinate_shift, page_shift;
    bool window_ring, coordinate32, coordinate_elements32;
    float scale;
    int32_t verify_ring_capacity = 0;
    bool explicit_positions = false;
};
struct Sources {
    uintptr_t q, kv, compressed, indices, table, compressed_table;
    uintptr_t cu, lengths, compressed_lengths, residual, window, sinks, out, workspace;
    int32_t tokens = 0, batch = 0;
    uintptr_t verify_requests = 0, verify_pending = 0;
    uintptr_t positions = 0;
};
struct Arguments {
    Sources sources;
    Config config;
};
struct DirectSources {
    uintptr_t q, kv, table, cu, lengths, window, sinks, out, workspace;
    int32_t tokens, batch;
    uintptr_t positions = 0;
};
inline int64_t workspace_bytes(const Config &c) {
    const int64_t score_probability_bytes =
        int64_t(c.cores) * 2 * c.heads * c.capacity * (sizeof(float) + sizeof(uint16_t));
    if (c.mode != int32_t(Mode::Sparse))
        return score_probability_bytes;
    return score_probability_bytes +
           int64_t(c.cores) * (2 * c.heads * c.width * sizeof(float) +
                               3 * c.selected_rows * (c.width * sizeof(uint16_t) + sizeof(float)));
}
} // namespace attention
namespace native_host {
attention::Config attention_geometry(int64_t tokens, int64_t heads, int64_t batch, int64_t columns,
                                     int64_t compressed_columns, int64_t capacity, int64_t ratio,
                                     int64_t mode, int64_t cores, bool ring, int64_t width,
                                     int64_t page_rows, int64_t window_rows, int64_t selected_rows);
}
extern "C" void dsv41_attention_launch(void *stream, const attention::Arguments &args);
