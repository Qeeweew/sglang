#pragma once
#include <cstdint>
namespace projection {
// Capacities are chosen from arch2201 L1/L0 budgets; actual extents below are
// runtime values. Two K panels fit L1, two operand pairs fit L0A/L0B.
inline constexpr uint32_t kMaxTileRows = 256;
inline constexpr uint32_t kMaxTileColumns = 256;
inline constexpr uint32_t kMaxPanelColumns = 256;
inline constexpr uint32_t kMaxDotColumns = 64;
inline constexpr uint32_t kCubeFractalElements = 16;
inline constexpr uint32_t kDenseTileColumns = kMaxTileColumns / 2;
inline constexpr uint32_t kGroupedTileRows = kMaxTileRows / 2;
inline constexpr uint32_t kDensePanelColumns = kMaxPanelColumns;
// C0 permits a 256x128 or 128x256 tile, excluding the 256x256 combination.
inline constexpr uint32_t kMaxAccumulatorElements = 128 * 256;
inline constexpr uint32_t kMaxCombinedTileExtent = 256 + 128;
struct Sources {
    uintptr_t x, w, y;
};
struct Arguments {
    uintptr_t x, w, y;
    int64_t m;
    uint32_t k, n, groups, cores;
    uint32_t tile_rows, tile_columns, panel_columns, dot_columns;
};
} // namespace projection
extern "C" void dsv41_projection_launch(void *stream, const projection::Arguments &,
                                        bool grouped, bool fp32);
