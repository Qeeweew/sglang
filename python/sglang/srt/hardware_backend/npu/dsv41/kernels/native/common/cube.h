#pragma once
#include <c_api/asc_simd.h>
#include <common/arch2201.h>

namespace native_device {
enum class MmadColumns { Aligned, Tail };

// The short-MMAD hazard counts 16x16 output fractals. M is padded by the caller.
// QLI passes a live N tail to the instruction but occupies a whole final fractal.
template <MmadColumns Columns = MmadColumns::Aligned, class Acc, class T, class M, class N>
__aicore__ inline void mmad(__cc__ Acc *c, __ca__ T *a, __cb__ T *b, M rows, uint16_t depth,
                           N columns, uint8_t unit, bool initialize) {
    asc_mmad(c, a, b, uint16_t(rows), depth, uint16_t(columns), unit, false, false, initialize);
    constexpr int fractal = arch2201::kMmadFractal;
    const auto n_blocks =
        Columns == MmadColumns::Tail ? (columns + fractal - 1) / fractal : columns / fractal;
    if (rows / fractal * n_blocks < arch2201::kUnfencedMmadFractals)
        pipe_barrier(PIPE_M);
}

// The 16-bit Load3D modes share bit layouts for FP16 and BF16 operands.
template <class T>
__aicore__ inline void load_a16_3d(__ca__ T *dst, __cbuf__ T *src, int rows, int depth) {
    static_assert(sizeof(T) == 2);
    asc_copy_l12l0a(reinterpret_cast<__ca__ half *>(dst), reinterpret_cast<__cbuf__ half *>(src),
                    depth, rows, 0, 0, 1, 1, 1, 1, 1, 1, false, false, false, false, depth);
}

template <class T>
__aicore__ inline void load_b16_transposed(__cb__ T *dst, __cbuf__ T *src, int columns, int depth) {
    static_assert(sizeof(T) == 2);
    asc_copy_l12l0b(reinterpret_cast<__cb__ half *>(dst), reinterpret_cast<__cbuf__ half *>(src),
                    columns, depth, 0, 0, 1, 1, 1, 1, 1, 1, false, false, false, true, columns);
}
} // namespace native_device
