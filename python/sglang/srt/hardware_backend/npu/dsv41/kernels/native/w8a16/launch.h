#pragma once
#include <cstdint>
#include <common/arch2201.h>
#include <w8a16/quantize_params.h>

namespace w8a16 {
inline constexpr int kGroup = 256, kC0 = 16, kSlots = 2, kVectorActors = 2;
// 8192 FP32 values fit beside the two input panels and output chunks, including
// BN256/BK256. 16384 would exceed the arch2201 UB budget by the scale spans.
inline constexpr int kCastChunkElements = 8192, kMaxDotK = 128;
struct Arguments {
    uintptr_t input, weight, scales, output, workspace;
    int64_t rows;
    int k, n, batches, bm, bn, bk, workers;
    bool bf16;
};
inline int64_t jobs(int64_t rows, int64_t n, int64_t batches, int64_t bm, int64_t bn) {
    return ((rows + bm - 1) / bm) * (n / bn) * batches;
}
// Host and device share byte formulas. No allocator or synchronization here.
struct Storage {
    int local_elements, chunk_elements, scale_elements, scale_offset, decoded_offset, cast_offset, ub_bytes;
    int a1_bytes, b1_bytes, dot_k, a0_bytes, b0_bytes, c0_bytes;
    constexpr Storage(int bm, int bn, int bk, bool bf16)
        : local_elements(bk * bn / kVectorActors),
          chunk_elements(local_elements / kSlots),
          scale_elements((bk + kGroup - 1) / kGroup * bn / kVectorActors),
          scale_offset(kSlots * local_elements),
          decoded_offset(scale_offset + kSlots * scale_elements * 2),
          cast_offset(decoded_offset + kSlots * chunk_elements * 2),
          ub_bytes(cast_offset + (bf16 ? kCastChunkElements * 4 : 0)),
          a1_bytes(bm * bk * 2), b1_bytes(bn * bk * 2),
          dot_k((int(arch2201::kL0ABytes) / (kSlots * bm * 2) < kMaxDotK
                     ? int(arch2201::kL0ABytes) / (kSlots * bm * 2) : kMaxDotK)),
          a0_bytes(0), b0_bytes(0), c0_bytes(bm * bn * 4) {
        const int b_limit = arch2201::kL0BBytes / (kSlots * bn * 2);
        if (dot_k > b_limit) dot_k = b_limit;
        if (dot_k > bk) dot_k = bk;
        dot_k = dot_k / kC0 * kC0;
        a0_bytes = bm * dot_k * 2;
        b0_bytes = bn * dot_k * 2;
    }
    constexpr bool fits() const {
        return ub_bytes <= int(arch2201::kUbBytes) &&
               kSlots * (a1_bytes + b1_bytes) <= int(arch2201::kL1Bytes) &&
               kSlots * a0_bytes <= int(arch2201::kL0ABytes) &&
               kSlots * b0_bytes <= int(arch2201::kL0BBytes) &&
               c0_bytes <= int(arch2201::kL0CBytes) && dot_k > 0;
    }
};
} // namespace w8a16
void w8a16_launch(void *stream, const w8a16::Arguments &args);
void w8a16_quantize_launch(void *stream, const w8a16::QuantizeSources &sources,
                          int k, int workers);
