#pragma once
#include <cstdint>
#include <common/arch2201.h>

namespace mxfp4_a8 {
inline constexpr int fractal_rows = 16, group = 32, slots = 2;
constexpr int tile_columns(int depth, int width, bool g32) {
    for (int size = g32 && depth < width ? 128 : 256; size > 64; size -= 64)
        if (width % size == 0)
            return size;
    return 64;
}
constexpr int panel_depth(int depth, int columns, bool g32) {
    const int limit = 131072 / columns;
    for (int size = limit / group * group; size >= group; size -= group)
        if (depth % size == 0)
            return size;
    return 0;
}
constexpr int quantum_size(int local) {
    for (int size = 16384; size > 1024; size -= 1024)
        if (local % size == 0)
            return size;
    return 1024;
}
constexpr int dot_depth(int columns) {
    return 32768 / columns / group * group;
}
constexpr int tile_rows(int depth, int width, bool g32) {
    const int columns = tile_columns(depth, width, g32), panel = panel_depth(depth, columns, g32);
    for (int size = g32 ? 128 : fractal_rows; size > fractal_rows; size /= 2)
        if (slots * (columns + size) * panel <= arch2201::kL1Bytes &&
            slots * size * dot_depth(columns) <= arch2201::kL0ABytes &&
            size * columns * sizeof(int32_t) <= arch2201::kL0CBytes)
            return size;
    return fractal_rows;
}
constexpr int64_t scratch_bytes(int depth, int width, int workers, bool g32) {
    const int columns = tile_columns(depth, width, g32);
    return int64_t(workers) * slots * columns *
           (panel_depth(depth, columns, g32) + tile_rows(depth, width, g32) * sizeof(int32_t));
}
struct Inputs {
    uintptr_t activation, codes, factors, channel, token, routing, output, scratch;
    int experts;
};
void launch(void *stream, int depth, int width, int workers, const Inputs &inputs, bool g32, bool bf16);
} // namespace mxfp4_a8
