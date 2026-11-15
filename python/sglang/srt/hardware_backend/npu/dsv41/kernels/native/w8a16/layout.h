#pragma once
#include <kernel_operator.h>
#include <c_api/asc_simd.h>
#include <w8a16/launch.h>

namespace w8a16 {
using namespace AscendC;
constexpr int kLogicalK = W8A16_K, kLogicalN = W8A16_N, kBatches = W8A16_BATCHES;
constexpr int kRows = W8A16_BM, kColumns = W8A16_BN, kPanelK = W8A16_BK;
constexpr int kWorkers = W8A16_WORKERS, kPanels = kLogicalK / kPanelK;
constexpr bool kBf16 = W8A16_BF16;
#if W8A16_BF16
using Element = bfloat16_t;
#else
using Element = half;
#endif
constexpr Storage kMemory(kRows, kColumns, kPanelK, kBf16);
static_assert(kMemory.fits() && kLogicalK % kPanelK == 0);
constexpr int kLocalBlocks = kColumns / (kVectorActors * kC0);
constexpr int kPanelGroups = (kPanelK + kGroup - 1) / kGroup;
constexpr int kScaleDepth = kPanelK < kGroup ? kPanelK : kGroup;
constexpr int kVectorHalfElements = 128, kVectorFloatElements = 64;
constexpr int kOperandReuseEventBase = 3, kStateEvent = 6, kDrainEvent = 7;
constexpr int kWeightPanelEventBase = kSlots;
constexpr int kAccumulatorEvent = 0, kFootprintRetiredEvent = 5, kGroupFlagMode = 2;
constexpr int kAccumulatorContinue = 2, kAccumulatorPublish = 3;
constexpr int kShortMmadFractals = 10;
enum PanelFlag : uint16_t { kPanel0Ready = 0, kPanel1Ready = 1, kPanel0Free = 2, kPanel1Free = 3 };
struct Tile { int64_t row; int column, batch, valid, matrix_rows; };
__aicore__ inline Tile coordinates(int64_t job, int64_t rows) {
    Tile tile{};
    tile.batch = job % kBatches;
    const int64_t local = job / kBatches;
    constexpr int columns = kLogicalN / kColumns;
    int64_t row_tile = local / columns;
    int column_tile = local % columns;
    if constexpr (kRows >= 128) {
        // Same eight-by-eight locality swizzle as quantized grouped GEMM.
        // Interleave neighboring M/N tiles across persistent workers.
        constexpr int swizzle_tiles = 8;
        const int64_t row_tiles = (rows + kRows - 1) / kRows;
        const int64_t row_base = row_tile / swizzle_tiles * swizzle_tiles;
        const int height = int(min(int64_t(swizzle_tiles), row_tiles - row_base));
        const int64_t relative = local - row_base * columns;
        const int column_base = int(relative / (height * swizzle_tiles)) * swizzle_tiles;
        const int width = min(swizzle_tiles, columns - column_base);
        const int inner = int(relative % (height * swizzle_tiles));
        row_tile = row_base + inner % height;
        column_tile = column_base + (inner / height + inner % height) % width;
    }
    tile.column = column_tile * kColumns;
    tile.row = row_tile * kRows;
    tile.valid = min(int64_t(kRows), rows - tile.row);
    tile.matrix_rows = (tile.valid + kC0 - 1) / kC0 * kC0;
    return tile;
}
} // namespace w8a16
