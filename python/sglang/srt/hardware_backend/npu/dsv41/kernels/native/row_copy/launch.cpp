#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
#include <numeric>

namespace {
int64_t dma_row_group(int64_t row_bytes) {
    // Each worker owns an aligned DMA block, including for sub-block rows.
    constexpr int64_t dma_bytes = 32;
    return dma_bytes / std::gcd(row_bytes, dma_bytes);
}

native_jit::Kernel &copy_entry(int64_t row_bytes, bool scatter, uint32_t fill, uint32_t workers) {
    static thread_local native_jit::ActualEntries<4> entries;
    return entries.get({uint64_t(row_bytes), scatter, fill, workers}, [&] {
        return std::make_unique<native_jit::Kernel>(
            scatter ? "scatter_rows" : "gather_rows", "row_copy/row_copy.asc",
            std::vector<std::string>{"--cce-auto-sync=off",
                                     "-DROW_COPY_ROW_BYTES=" + std::to_string(row_bytes),
                                     "-DROW_COPY_ROW_GROUP=" + std::to_string(dma_row_group(row_bytes)),
                                     "-DROW_COPY_SCATTER=" + std::to_string(scatter),
                                     "-DROW_COPY_FILL=" + std::to_string(fill),
                                     "-DROW_COPY_WORKERS=" + std::to_string(workers)},
            workers);
    });
}
} // namespace

void dsv41_row_copy_launch(void *stream, uint32_t workers, uint8_t *source, uint8_t *row_map,
                           uint8_t *output, int64_t map_rows, int64_t source_rows,
                           int64_t output_rows, int64_t row_bytes, bool scatter,
                           uint32_t fill_byte) {
    const int64_t row_group = dma_row_group(row_bytes);
    const int64_t owner_rows = scatter ? output_rows : map_rows;
    if (map_rows > 0 && owner_rows > 0)
        copy_entry(row_bytes, scatter, fill_byte, workers)
            .launch_persistent(stream, (owner_rows + row_group - 1) / row_group, source, row_map,
                               output, map_rows, source_rows, output_rows);
}

TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_row_copy(int row_bytes, bool scatter=False, int fill=0, int workers=0) -> ()",
          [](int64_t row_bytes, bool scatter, int64_t fill, int64_t workers) {
              TORCH_CHECK(row_bytes > 0 && (fill == 0 || (!scatter && fill == -1)),
                          "invalid row-copy geometry");
              const auto budget = native_host::core_count(c10_npu::current_device());
              TORCH_CHECK(workers >= 0 && workers <= budget, "invalid worker budget");
              copy_entry(row_bytes, scatter, fill == -1 ? UINT8_MAX : 0,
                         workers ? workers : budget).prepare();
          });
}
