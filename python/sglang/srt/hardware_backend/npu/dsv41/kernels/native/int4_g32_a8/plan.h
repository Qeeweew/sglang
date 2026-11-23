#pragma once
#include <cstdint>
#include <common/arch2201.h>

namespace int4_g32_a8 {
inline constexpr int fractal_rows = 16, group = 32, slots = 2;
constexpr bool valid_geometry(int64_t depth, int64_t width) {
    return depth >= 256 && depth <= 8192 && depth % group == 0 && width > 0 &&
           width <= UINT16_MAX && width % 64 == 0;
}
constexpr int tile_columns(int depth, int width) {
    for (int size = depth < width ? 128 : 256; size > 64; size -= 64)
        if (width % size == 0)
            return size;
    return 64;
}
constexpr int panel_depth(int depth, int columns) {
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
constexpr int tile_rows(int depth, int width) {
    const int columns = tile_columns(depth, width), panel = panel_depth(depth, columns);
    for (int size = 128; size > fractal_rows; size /= 2)
        if (slots * (columns + size) * panel <= arch2201::kL1Bytes &&
            slots * size * dot_depth(columns) <= arch2201::kL0ABytes &&
            size * columns * sizeof(int32_t) <= arch2201::kL0CBytes)
            return size;
    return fractal_rows;
}
constexpr int64_t scratch_bytes(int depth, int width, int workers) {
    const int columns = tile_columns(depth, width);
    return int64_t(workers) * slots * columns *
           (panel_depth(depth, columns) + tile_rows(depth, width) * sizeof(int32_t));
}
struct Inputs {
    uintptr_t activation, codes, factors, channel, token, routing, output, scratch, bypass_alias;
    int experts;
};
void launch(void *stream, int depth, int width, int workers, const Inputs &inputs, bool bf16);
} // namespace int4_g32_a8
