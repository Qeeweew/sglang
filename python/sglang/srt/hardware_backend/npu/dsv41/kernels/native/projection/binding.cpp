#include "torch.h"
#include "launch.h"
#include "jit.h"

namespace native_host {
namespace {
void run_projection(const at::Tensor &x, const at::Tensor &w, const at::Tensor &y, bool grouped) {
    TORCH_CHECK_VALUE(x.dim() == (grouped ? 3 : 2) && w.dim() == 3,
                      "projection requires dense [M,K]/[1,K,N] or grouped [M,G,K]/[G,N,K]");
    const auto m = x.size(0), k = x.size(-1), g = grouped ? x.size(1) : 1;
    const auto n = w.size(grouped ? 1 : 2);
    TORCH_CHECK_VALUE(m >= 0 && k > 0 && n > 0 && g > 0 && k <= INT32_MAX && n <= INT32_MAX &&
                          g <= INT32_MAX && k % projection::kCubeFractalElements == 0 &&
                          n % projection::kCubeFractalElements == 0,
                      "projection requires positive INT32 K/N/G, complete Cube K/N fractals");
    TORCH_CHECK_VALUE(
        g * k <= UINT16_MAX && (grouped || n <= UINT16_MAX) && uint64_t(g) * n <= UINT32_MAX,
        "projection ND2NZ source stride / FIX destination stride exceeds hardware fields");
    auto device = x.device();
    check(x, at::kBFloat16, device);
    check(w, at::kBFloat16, device,
          grouped ? at::IntArrayRef({g, n, k}) : at::IntArrayRef({1, k, n}));
    TORCH_CHECK_VALUE(y.scalar_type() == at::kBFloat16 ||
                          (!grouped && y.scalar_type() == at::kFloat),
                      "dense BF16/FP32 or grouped BF16 output required");
    check(y, y.scalar_type(), device);
    TORCH_CHECK_VALUE(y.sizes() == (grouped ? at::IntArrayRef({m, g, n}) : at::IntArrayRef({m, n})),
                      "projection output shape differs");
    disjoint({y}, {x, w});
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    for (const auto &t : {x, w, y})
        record(t, stream);
    if (m) {
        const auto cores = uint32_t(core_count(device.index(), true));
        uint32_t dense_rows = projection::kCubeFractalElements;
        // PMU identifies operand reads as the dense bottleneck. Rank bounded
        // row capacities by the critical worker's A+B import volume, retaining
        // the runtime actual extent and the same two-slot Cube pipeline.
        const uint64_t column_tiles =
            (n + projection::kDenseTileColumns - 1) / projection::kDenseTileColumns;
        uint64_t best_import_units = UINT64_MAX;
        for (uint32_t rows = projection::kCubeFractalElements; rows <= projection::kMaxTileRows;
             rows *= 2) {
            const uint64_t row_tiles = (m + rows - 1) / rows;
            const uint64_t waves = (row_tiles * column_tiles + cores - 1) / cores;
            const uint64_t import_units = waves * (rows * column_tiles + n);
            if (import_units < best_import_units) {
                best_import_units = import_units;
                dense_rows = rows;
            }
        }
        const uint32_t tile_rows =
            grouped ? (m <= projection::kCubeFractalElements ? projection::kCubeFractalElements
                                                             : projection::kGroupedTileRows)
                    : dense_rows;
        const uint32_t tile_columns = grouped && n % projection::kMaxTileColumns == 0
                                          ? projection::kMaxTileColumns
                                          : projection::kDenseTileColumns;
        const uint32_t panel_columns = std::min<int64_t>(
            grouped ? projection::kMaxPanelColumns : projection::kDensePanelColumns, k);
        projection::Arguments args{reinterpret_cast<uintptr_t>(ptr(x)),
                                   reinterpret_cast<uintptr_t>(ptr(w)),
                                   reinterpret_cast<uintptr_t>(ptr(y)),
                                   m,
                                   uint32_t(k),
                                   uint32_t(n),
                                   uint32_t(g),
                                   cores,
                                   tile_rows,
                                   tile_columns,
                                   panel_columns,
                                   std::min(projection::kMaxDotColumns, panel_columns)};
        dsv41_projection_launch(stream.stream(), args, grouped, y.scalar_type() == at::kFloat);
    }
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("projection(Tensor x, Tensor weight, Tensor(a!) out, bool grouped) -> ()",
          &run_projection);
    m.def("prepare_projection(Tensor x, Tensor weight, Tensor(a!) out, bool grouped) -> ()",
          [](const at::Tensor &x, const at::Tensor &w, const at::Tensor &y, bool grouped) {
              native_jit::PrepareOnlyScope prepare;
              run_projection(x, w, y, grouped);
          });
}
} // namespace native_host
