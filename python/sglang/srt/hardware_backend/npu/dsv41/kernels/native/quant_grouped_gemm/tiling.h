#pragma once
#include <cstdint>
#include <common/arch2201.h>

namespace quant_gmm {
inline constexpr int kVectorLanesPerCube = 2;
struct Config { int k, n; };
struct Arguments {
    void *stream;
    int cores, experts, tile_rows, tile_columns;
    void *activation, *packed, *scales, *routing, *output, *workspace;
    Config config;
};
enum Field { K, N, BM, BN, STEP_K, CK, LOCAL_ELEMENTS, OUTPUT_BYTES, MICRO_ELEMENTS, Fields };
struct Geometry {
    int64_t values[Fields]{};
    constexpr int64_t &operator[](Field i) { return values[i]; }
    constexpr int64_t operator[](Field i) const { return values[i]; }
};
inline constexpr int tile_columns(int n, int bm) {
    int selected = 64;
    constexpr int column_step = kVectorLanesPerCube * (32 / sizeof(uint16_t));
    for (int v = 64; v <= 256; v += column_step)
        if (n % v == 0 &&
            bm * v * sizeof(float) <= arch2201::kL0CBytes)
            selected = v;
    return selected;
}
// Capacities are element/byte budgets, independent of storage N subdivisions.
// One quantization atom is K32 x hardware C0=16, not a tunable Width.
struct Storage {
    static constexpr int kHalfBytes = sizeof(uint16_t), kCodesPerByte = 2;
    static constexpr int kCodesPerInt32 = sizeof(int32_t) * kCodesPerByte;
    static constexpr int kFractalRows = 16, kC0Elements = 32 / kHalfBytes;
    static constexpr int kDecodeTileRows = 16, kPrefillTileRows = 128, kWidePrefillTileRows = 256;
    static constexpr int kScaleGroup = 32, kSlots = 2, kVectorBytes = 256, kMaxRepeat = 255;
    static constexpr int kAtomElements = kScaleGroup * kC0Elements;
    // No TPipe/KFC/printf/SIMT services: this entry owns the complete UB arena.
    static constexpr int kUbArenaBytes = arch2201::kUbBytes + arch2201::kSdkReservedBytes;
    int step_k, cube_k, local_elements, output_elements, micro_elements;
    static constexpr int fixed_bytes(int elements) {
        // Double compressed input, double worst-case FP16 scale input,
        // One natural N16 FP16 scale row per K32; vector block-stride0
        // broadcasts it directly. Threshold and two bank-phase gaps remain.
        return elements + elements / 8 + elements / 16 + 3 * kVectorBytes;
    }
    static constexpr int output_capacity(int elements) {
        return (kUbArenaBytes - fixed_bytes(elements)) / (3 * kHalfBytes) /
            kAtomElements * kAtomElements;
    }
    constexpr Storage(int k, int bm, int bn)
        : step_k(32), cube_k(16), local_elements(kAtomElements),
          output_elements(0), micro_elements(0) {
        while (fixed_bytes(2 * local_elements) < kUbArenaBytes &&
               2 * local_elements <= 2 * output_capacity(2 * local_elements))
            local_elements *= 2;
        const int whole = k * (bn / kVectorLanesPerCube);
        if (whole < local_elements) local_elements = whole;
        if (fixed_bytes(whole) < kUbArenaBytes && whole <= 2 * output_capacity(whole) &&
            2 * kHalfBytes * k * (bm + bn) <= arch2201::kL1Bytes)
            local_elements = whole;
        for (int depth = kScaleGroup; depth <= 2 * local_elements / bn &&
             2 * kHalfBytes * depth * (bm + bn) <= arch2201::kL1Bytes; depth += kScaleGroup)
            if (k % depth == 0) step_k = depth;
        const int extent = bm > bn ? bm : bn;
        while (cube_k < 256 && 2 * kHalfBytes * (cube_k * 2) * extent <= arch2201::kL0ABytes)
            cube_k *= 2;
        output_elements = output_capacity(local_elements);
        if (output_elements > local_elements) output_elements = local_elements;
        // Balance the two chunks when a raw-capacity block exceeds one
        // output slot. An almost-full first chunk delays panel publication
        // and leaves little V work to overlap its MTE3 drain.
        else output_elements = (local_elements / 2 + kAtomElements - 1) / kAtomElements * kAtomElements;
        micro_elements = output_elements;
        const int repeat_limit = kMaxRepeat * (kVectorBytes / kHalfBytes) / kAtomElements * kAtomElements;
        if (micro_elements > repeat_limit) micro_elements = repeat_limit;
    }
};
inline Geometry geometry(int k, int n, int bm) {
    if (k <= 0 || k > UINT16_MAX || n <= 0 || k % Storage::kScaleGroup || n % 64 ||
        (bm != 16 && bm != 128 && bm != 256)) return {};
    const int bn = tile_columns(n, bm);
    const Storage s(k, bm, bn);
    return {{k, n, bm, bn, s.step_k, s.cube_k, s.local_elements,
             s.output_elements * Storage::kHalfBytes, s.micro_elements}};
}
} // namespace quant_gmm
