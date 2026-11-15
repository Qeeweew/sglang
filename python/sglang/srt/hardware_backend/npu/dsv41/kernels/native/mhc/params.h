#pragma once
#include <common/arch2201.h>
#include <cstdint>

namespace mhc {
#if defined(__NPU_ARCH__)
#define MHC_API __aicore__
#else
#define MHC_API
#endif
// Storage capacities follow one 32-byte FP32 block, one 256-byte Vector
// repeat and the arch2201 matrix fractal. None is a model/request dimension.
inline constexpr uint32_t kDataBlockBytes = 32;
inline constexpr uint32_t kVectorRepeatBytes = 256;
inline constexpr uint32_t kFp32Bytes = sizeof(float);
inline constexpr uint32_t kBf16Bytes = sizeof(uint16_t); // BF16 wire storage is sixteen bits.
inline constexpr uint32_t kIndexBytes = sizeof(uint32_t);
inline constexpr uint32_t kRepeatBlocks = kVectorRepeatBytes / kDataBlockBytes;
inline constexpr uint32_t kBlockFloats = kDataBlockBytes / sizeof(float);
inline constexpr uint32_t kRepeatFloats = kVectorRepeatBytes / sizeof(float);
inline constexpr uint32_t kMatrixRows = 16;
inline constexpr uint32_t kMmadK = kRepeatFloats;
inline constexpr uint32_t kSlots = 2;
inline constexpr uint32_t kStatsRowCapacity = kBlockFloats;
inline constexpr uint32_t kBranchCapacity = kBlockFloats;
inline constexpr uint32_t kPanelKCapacity = 2048;
inline constexpr uint32_t kPanelKAlignment = kBlockFloats * kRepeatFloats;
inline constexpr uint32_t kCubeRowCapacity =
    arch2201::kL0ABytes / (kSlots * kMmadK * sizeof(float));
inline constexpr uint32_t kL1TileKCapacity = 4 * kMmadK;
// Two r64 panels are the normalization reduction contract; independent
// runtime column tiling may use more panels when resident storage requires it.
inline constexpr uint32_t kSinkhornIterations = 20;
inline constexpr uint32_t kHiddenWidthAlignment = 2 * kRepeatFloats;
inline constexpr uint32_t kHiddenWidthCapacity = 2 * kRepeatFloats * kRepeatFloats;
inline constexpr uint32_t kBankPhaseBytes = kDataBlockBytes;
MHC_API inline constexpr uint32_t round_up(uint32_t v, uint32_t a) {
    return (v + a - 1) / a * a;
}
MHC_API inline constexpr uint32_t round_down(uint32_t v, uint32_t a) {
    return v / a * a;
}
struct StateParams {
    uint64_t rows;
    uint32_t width, branches, column_tile, column_tiles, column, output_columns;
};
struct StatsParams {
    uint64_t rows, padded_rows;
    uint32_t branches, hidden_width, projection_columns, projection_stride;
    uint32_t splits, tile_rows, post_tile_columns, post_column_tiles;
    float inverse_input_width, epsilon;
};
struct ProjectParams {
    uint64_t rows, padded_rows;
    uint32_t hidden_width, branches, input_width, projection_columns, projection_stride;
    uint32_t tile_rows, panel_k, panel_count, l1_tile_k;
    uint32_t splits, row_groups, cube_groups, vector_groups, row_tiles;
};
// Whole-row collapse/normalization, with a runtime column tile when full-row
// activation storage does not fit. BF16 collapsed state and FP32 norm weight
// remain resident while activations are tiled.
struct CombineLayout {
    uint32_t raw, streams, total, work, rounded, weight, scratch, stat, bytes;
    MHC_API constexpr CombineLayout(uint32_t b, uint32_t h, uint32_t t)
        : raw(0), streams(b * t * kBf16Bytes),
          total(streams + (b * t > h / 2 ? b * t : h / 2) * kFp32Bytes),
          work(total + (t > h / 2 ? t : h / 2) * kFp32Bytes + kBankPhaseBytes),
          rounded(work + (t > h / 2 ? t : h / 2) * kFp32Bytes + kBankPhaseBytes),
          weight(rounded + h * kBf16Bytes), scratch(weight + h * kFp32Bytes + kBankPhaseBytes),
          stat(scratch + 2 * kRepeatFloats * kFp32Bytes),
          bytes(stat + 2 * kRepeatFloats * kFp32Bytes) {}
};
struct PostLayout {
    uint32_t raw, rawx, packed, streams, value, total, work, mixed, rounded;
    uint32_t collapsed, weight, scratch, stat, coefficients, pre, post, bytes;
    MHC_API constexpr PostLayout(uint32_t b, uint32_t h, uint32_t t)
        : raw(0), rawx(b * t * kBf16Bytes), packed(rawx + t * kBf16Bytes),
          streams(packed + t * kBf16Bytes), value(streams + b * t * kFp32Bytes),
          total(value + (t > h / 2 ? t : h / 2) * kFp32Bytes + kBankPhaseBytes),
          work(total + t * kFp32Bytes + kBankPhaseBytes),
          mixed(work + (t > h / 2 ? t : h / 2) * kFp32Bytes + kBankPhaseBytes),
          rounded(mixed + t * kFp32Bytes + kBankPhaseBytes),
          collapsed(rounded + t * kFp32Bytes + kBankPhaseBytes),
          weight(collapsed + h * kBf16Bytes + kBankPhaseBytes),
          scratch(weight + h * kFp32Bytes + kBankPhaseBytes),
          stat(scratch + 2 * kRepeatFloats * kFp32Bytes),
          coefficients(stat + 2 * kRepeatFloats * kFp32Bytes),
          pre(coefficients + round_up(b * b, kBlockFloats) * kFp32Bytes),
          post(pre + kDataBlockBytes), bytes(post + kDataBlockBytes) {}
};
struct MeanLayout {
    uint32_t raw, packed, total, value, bytes;
    MHC_API constexpr MeanLayout(uint32_t b, uint32_t t, uint32_t slots = kSlots)
        : raw(0), packed(slots * b * t * kBf16Bytes), total(packed + slots * t * kBf16Bytes),
          value(total + t * kFp32Bytes + kBankPhaseBytes), bytes(value + t * kFp32Bytes) {}
};
// The statistics arena is small even at its full block capacities. Device
// entries use fixed offsets and runtime active extents, avoiding repeated
// scalar address construction without specializing the input dimensions.
struct StatsLayout {
    uint32_t a, b, stat, gate, bias, matrix, reduction, broadcast, column, ones, base;
    uint32_t gate_index, matrix_index, matrix_bias, bytes;
    MHC_API constexpr StatsLayout(
        uint32_t branches = kBranchCapacity, uint32_t rows = kStatsRowCapacity,
        uint32_t stride = round_up(kBranchCapacity * (kBranchCapacity + 2), kMatrixRows))
        : a(0), b(a + rows * stride * kFp32Bytes), stat(b + rows * stride * kFp32Bytes),
          gate(stat + kRepeatFloats * kFp32Bytes),
          bias(gate + 2 * rows * kBlockFloats * kFp32Bytes),
          matrix(bias + 2 * rows * kBlockFloats * kFp32Bytes),
          reduction(matrix + round_up(branches * rows * kBlockFloats, kRepeatFloats) * kFp32Bytes),
          broadcast(reduction + kRepeatFloats * kFp32Bytes),
          column(broadcast + round_up(branches * rows * kBlockFloats, kRepeatFloats) * kFp32Bytes),
          ones(column + rows * kBlockFloats * kFp32Bytes),
          base(ones + 2 * rows * kBlockFloats * kFp32Bytes),
          gate_index(base + round_up(branches * (branches + 2), kBlockFloats) * kFp32Bytes),
          matrix_index(gate_index + 2 * rows * kBlockFloats * kIndexBytes),
          matrix_bias(matrix_index +
                      round_up(branches * rows * kBlockFloats, kRepeatFloats) * kIndexBytes),
          bytes(matrix_bias +
                round_up(branches * rows * kBlockFloats, kRepeatFloats) * kFp32Bytes) {}
};
// The standalone Vector projection owns this whole arena. Its largest
// power-of-two K panel is selected from UB, independently of Cube L1.
struct SmallProjectionLayout {
    uint32_t weights, value, products, raw, scratch, output, sums, stat, bytes;
    MHC_API constexpr explicit SmallProjectionLayout(uint32_t panel)
        : weights(0), value(kBlockFloats * panel * kFp32Bytes),
          products(value + panel * kFp32Bytes), raw(products + kBlockFloats * panel * kFp32Bytes),
          scratch(raw + panel * kBf16Bytes),
          output(scratch +
                 kBlockFloats * (panel / kBlockFloats + panel / kRepeatFloats) * kFp32Bytes),
          sums(output + kCubeRowCapacity * kBlockFloats * kFp32Bytes),
          stat(sums + kCubeRowCapacity * kFp32Bytes), bytes(stat + kRepeatFloats * kFp32Bytes) {}
};
MHC_API inline constexpr uint32_t small_projection_panel_capacity() {
    uint32_t panel = kRepeatFloats;
    while (SmallProjectionLayout(2 * panel).bytes <= arch2201::kUbBytes)
        panel *= 2;
    return panel;
}
inline constexpr uint32_t kSmallProjectionPanelCapacity = small_projection_panel_capacity();
static_assert(SmallProjectionLayout(kSmallProjectionPanelCapacity).bytes <= arch2201::kUbBytes);
struct ProjectionVectorLayout {
    uint32_t rows, raw, value, squared, scratch, sums, accumulator, product, stat, bytes;
    MHC_API constexpr ProjectionVectorLayout(uint32_t tile_rows, uint32_t panel, uint32_t columns)
        : rows(kStatsRowCapacity), raw(0), value(rows * panel * sizeof(uint16_t)),
          squared(value + rows * panel * sizeof(float)),
          scratch(squared + rows * panel * sizeof(float)),
          sums(scratch + rows * (panel / kBlockFloats + panel / kRepeatFloats) * sizeof(float)),
          accumulator(sums + round_up(tile_rows / 2, kBlockFloats) * sizeof(float)),
          product(accumulator + tile_rows / 2 * columns * sizeof(float)),
          stat(product + tile_rows / 2 * columns * sizeof(float)),
          bytes(stat + kRepeatFloats * sizeof(float)) {}
};
#if !defined(__NPU_ARCH__)
inline uint32_t mean_tile(uint32_t b, uint32_t h) {
    // One equal column panel per raw slot balances the two-slot pipeline;
    // actual branches/width may require a smaller resource-valid extent.
    uint32_t t = round_up((h + kSlots - 1) / kSlots, kRepeatFloats);
    while (MeanLayout(b, t).bytes > arch2201::kUbBytes)
        t -= kRepeatFloats;
    return t;
}
inline uint32_t combine_tile(uint32_t b, uint32_t h) {
    uint32_t t = h;
    while (CombineLayout(b, h, t).bytes > arch2201::kUbBytes)
        t -= kRepeatFloats;
    return t;
}
inline uint32_t post_tile(uint64_t rows, uint32_t b, uint32_t h, uint32_t cores, bool fused) {
    uint32_t t = h;
    // The fused collapse/normalization pipeline has a dedicated two-equal-
    // panel path. Prefer it when whole-row storage does not fit, then shrink
    // by whole Vector repeats only if the runtime resident-weight budget needs it.
    if (PostLayout(b, h, t).bytes > arch2201::kUbBytes)
        t = round_up((h + 1) / 2, kRepeatFloats);
    while (PostLayout(b, h, t).bytes > arch2201::kUbBytes)
        t -= kRepeatFloats;
    if (!fused) {
        // Tiny rows expose one column partition per independent destination
        // stream; larger row batches amortize coefficient loads across columns.
        uint32_t desired = rows <= kSlots ? cores / b : (cores + rows - 1) / rows;
        if (!desired)
            desired = 1;
        if (rows > kSlots && desired > b)
            desired = b;
        const uint32_t parallel = round_up((h + desired - 1) / desired, kRepeatFloats);
        if (parallel < t)
            t = parallel;
    }
    return t;
}
inline uint32_t stats_post_tile(uint64_t rows, uint32_t branches, uint32_t width, uint32_t cores) {
    uint32_t tile = post_tile(rows, branches, width, cores, false);
    const uint32_t minimum = round_up((width + branches - 1) / branches, kRepeatFloats);
    if (tile < minimum)
        tile = minimum;
    constexpr StatsLayout statistics;
    constexpr uint32_t arena = round_up(statistics.bytes, kVectorRepeatBytes);
    while (arena + PostLayout(branches, width, tile).bytes > arch2201::kUbBytes)
        tile -= kRepeatFloats;
    return tile;
}
inline ProjectParams project_plan(uint64_t rows, uint32_t branches, uint32_t width,
                                  uint32_t cube_cores, uint32_t vector_cores) {
    ProjectParams p{};
    p.rows = rows;
    p.hidden_width = width;
    p.branches = branches;
    p.input_width = branches * width;
    p.projection_columns = branches * (branches + 2);
    p.projection_stride = round_up(p.projection_columns, kMatrixRows);
    p.tile_rows =
        rows < kCubeRowCapacity ? round_up(rows ? rows : 1, kMatrixRows) : kCubeRowCapacity;
    if (rows <= kStatsRowCapacity) {
        p.tile_rows = kMatrixRows;
        p.panel_k = kSmallProjectionPanelCapacity;
        p.panel_count = (p.input_width + p.panel_k - 1) / p.panel_k;
        p.splits = p.panel_count;
        p.row_tiles = (rows + p.tile_rows - 1) / p.tile_rows;
        // Compute remains one16-row tile; only the GM split pitch grows
        // to the Cube row capacity. The allocation gap is never read or
        // written, and M/B/H remain runtime dimensions.
        p.padded_rows = round_up(p.row_tiles * p.tile_rows, kCubeRowCapacity);
        p.vector_groups = vector_cores;
        return p;
    }
    const uint64_t a_min = uint64_t(kSlots) * p.tile_rows * kMmadK * sizeof(float);
    const uint32_t max_panel = (arch2201::kL1Bytes - a_min) / (p.projection_stride * sizeof(float));
    p.panel_k =
        round_down(max_panel < kPanelKCapacity ? max_panel : kPanelKCapacity, kPanelKAlignment);
    // Keep eight row sums in one aligned FP32 block. Reduce the K panel,
    // rather than producing misaligned sub-block Vector accumulators.
    while (ProjectionVectorLayout(p.tile_rows, p.panel_k, p.projection_stride).bytes >
           arch2201::kUbBytes)
        p.panel_k -= kPanelKAlignment;
    p.panel_count = (p.input_width + p.panel_k - 1) / p.panel_k;
    const uint64_t fn_bytes = uint64_t(p.projection_stride) * p.panel_k * sizeof(float);
    // Favor the full L1 copy extent. Its row capacity is below the L0A
    // capacity when resident Fn plus two A slots would enter SDK reserved L1.
    const uint32_t copy_row_capacity = round_down(
        (arch2201::kL1Bytes - fn_bytes) / (kSlots * kL1TileKCapacity * sizeof(float)), kMatrixRows);
    if (rows > p.tile_rows && copy_row_capacity >= kMatrixRows && copy_row_capacity < p.tile_rows)
        p.tile_rows = copy_row_capacity;
    const uint32_t max_l1 =
        (arch2201::kL1Bytes - fn_bytes) / (kSlots * p.tile_rows * sizeof(float));
    p.l1_tile_k = round_down(max_l1 < kL1TileKCapacity ? max_l1 : kL1TileKCapacity, kMmadK);
    p.row_tiles = (rows + p.tile_rows - 1) / p.tile_rows;
    const uint32_t live_tiles = p.row_tiles ? p.row_tiles : 1;
    const uint32_t available = cube_cores / live_tiles;
    p.splits = available ? (available < p.panel_count ? available : p.panel_count) : 1;
    if (rows <= kStatsRowCapacity)
        p.splits = p.panel_count;
    p.row_groups = cube_cores / p.splits;
    p.cube_groups = p.splits * p.row_groups;
    p.vector_groups = vector_cores;
    p.padded_rows = uint64_t(p.row_tiles) * p.tile_rows;
    return p;
}
#endif
#undef MHC_API
} // namespace mhc
