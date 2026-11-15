#pragma once
#include <cstdint>
#if defined(__CCE_AICORE__)
#include <kernel_operator.h>
#endif

namespace moe_routes {
// Qualified 8/32-KiB FP32 spans select latency/bandwidth allocation placement.
// Both capacities are tiled; neither limits a logical hidden width or batch.
inline constexpr uint32_t kLatencyCapacity = 2048;
inline constexpr uint32_t kBandwidthCapacity = 8192;
// Cyclic row/column strides. Logical row counts may be device metadata; no
// division is needed while walking conversion tasks. The selected allocation
// capacity makes width rounding a shift, without adding a kernel argument.
struct ConversionTiling {
    uint32_t elements, row_step;
    uint64_t tiles, column_step;
};
#if defined(__CCE_AICORE__)
template <uint32_t Capacity>
__aicore__ constexpr ConversionTiling conversion_for_width(uint64_t hidden, uint32_t cores) {
    const uint64_t tiles = (hidden + Capacity - 1) / Capacity;
    if (tiles >= cores)
        return {Capacity, 0, tiles, cores};
    const uint32_t columns = uint32_t(tiles);
    return {Capacity, cores / columns, tiles, cores % columns};
}
struct ConversionCursor {
    uint64_t row, column;
    ConversionTiling tiling;
    __aicore__ explicit ConversionCursor(ConversionTiling value, uint32_t core, uint32_t cores)
        : row(0), column(core), tiling(value) {
        if (tiling.tiles == 1) {
            row = core;
            column = 0;
        } else if (tiling.tiles < cores) {
            // This branch proves that the divisor fits the physical grid.
            const uint32_t columns = uint32_t(tiling.tiles);
            row = core / columns;
            column = core % columns;
        }
    }
    __aicore__ void advance() {
        row += tiling.row_step;
        column += tiling.column_step;
        if (column >= tiling.tiles) {
            column -= tiling.tiles;
            ++row;
        }
    }
};
#endif
} // namespace moe_routes
