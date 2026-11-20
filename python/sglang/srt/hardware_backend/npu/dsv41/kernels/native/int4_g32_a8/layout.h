#pragma once
#include <c_api/asc_simd.h>
#include <common/grouped_schedule.h>
#include <int4_g32_a8/plan.h>

namespace int4_g32_a8 {
using namespace AscendC;
using grouped_gemm::Tile;
constexpr int depth = A8_DEPTH, width = A8_WIDTH, workers = A8_WORKERS;
constexpr int rows = tile_rows(depth, width);
constexpr int columns = tile_columns(depth, width), half_n = columns / 2;
constexpr int panel = panel_depth(depth, columns), panels = depth / panel;
constexpr int local = half_n * panel, weight_bytes = columns * panel;
constexpr bool keep_activation =
    panels >= slots && rows * depth + slots * weight_bytes <= arch2201::kL1Bytes;
constexpr int activation_slots = keep_activation ? panels : slots;
constexpr int elements = quantum_size(local), blocks = local / elements;
constexpr int weight_steps = panels * blocks;
constexpr bool carry_weights = weight_steps >= slots;
constexpr int lookahead = blocks + 1 < weight_steps ? blocks + 1 : weight_steps - 1;
constexpr int dot_k = dot_depth(columns), chunks = (panel + dot_k - 1) / dot_k;
constexpr int scale_bytes = elements / 64, scale_slot = (scale_bytes + 31) / 32 * 32;
constexpr int result_offset = slots * weight_bytes, result_bytes = rows * columns * 4;
constexpr int worker_bytes = result_offset + slots * result_bytes;
constexpr int raw_base = 0, factor_base = elements, output_base = factor_base + slots * scale_slot;
constexpr int rank_base = output_base + slots * elements;
constexpr int expanded_base = rank_base + elements * 2 + 256;
constexpr int small_base = expanded_base + elements / 4;
constexpr int final_base = small_base + elements / 16 + 256;
constexpr int final_elements = rows * half_n;
static_assert(final_base + final_elements * 6 + (columns + rows * 9) * 4 <= arch2201::kUbBytes);
static_assert(slots * columns * dot_k <= arch2201::kL0BBytes &&
              slots * rows * dot_k <= arch2201::kL0ABytes);
static_assert(slots * weight_bytes + activation_slots * rows * panel <= arch2201::kL1Bytes);
} // namespace int4_g32_a8
