#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
#include <algorithm>
#include <numeric>
namespace {
// A complete int32 DMA block selects the existing scalar metadata path.
constexpr uint32_t kScalarMapRows = 32 / sizeof(int32_t);
native_jit::Kernel &metadata_entry(uint64_t window, uint64_t capacity) {
    static thread_local native_jit::ActualEntries<2> entries;
    return entries.get({window, uint64_t(capacity <= kScalarMapRows)}, [&] {
        return std::make_unique<native_jit::Kernel>(
            "ced_metadata", "ced/ced.asc",
            std::vector<std::string>{"--cce-auto-sync=off", "-DCED_METADATA=1",
                                     "-DCED_SMALL_MAP=" + std::to_string(capacity <= kScalarMapRows),
                                     "-DCED_WINDOW=" + std::to_string(window)},
            1);
    });
}
native_jit::Kernel &copy_entry(uint64_t row_bytes, uint64_t scatter, uint64_t fill, uint32_t workers) {
    static thread_local native_jit::ActualEntries<4> entries;
    return entries.get(
        {row_bytes, scatter, fill, workers}, [&] {
            return std::make_unique<native_jit::Kernel>(
                "ced_copy", "ced/ced.asc",
                std::vector<std::string>{"--cce-auto-sync=off",
                                         "-DCED_ROW_BYTES=" + std::to_string(row_bytes),
                                         "-DCED_SCATTER=" + std::to_string(scatter),
                                         "-DCED_FILL=" + std::to_string(fill),
                                         "-DCED_WORKERS=" + std::to_string(workers)}, workers);
        });
}
} // namespace
extern "C" void dsv41_ced_metadata_launch(void *stream, uint8_t *cu, uint8_t *row_map,
                                          uint8_t *requests, uint8_t *tail_cu, uint8_t *live,
                                          uint32_t batch, uint32_t window, int64_t capacity) {
    metadata_entry(window, capacity)
        .launch_persistent(stream, 1, cu, row_map, requests, tail_cu, live, batch, capacity);
}
extern "C" void dsv41_ced_copy_launch(void *stream, uint32_t blocks, uint8_t *source,
                                      uint8_t *row_map, uint8_t *output, int64_t map_rows,
                                      int64_t source_rows, int64_t output_rows, int64_t row_bytes,
                                      uint32_t scatter, uint32_t fill_byte) {
    const int64_t row_group = 32 / std::gcd<int64_t>(row_bytes, 32); // arch2201 DMA block
    const int64_t owner_rows = scatter ? output_rows : map_rows;
    if (map_rows > 0 && (!scatter || output_rows > 0))
        copy_entry(row_bytes, scatter, fill_byte, blocks)
            .launch_persistent(stream, (owner_rows + row_group - 1) / row_group, source, row_map, output, map_rows, source_rows, output_rows);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_ced(str stage, int batch=0, int window=128, int capacity=0, int map_rows=0, "
          "int source_rows=0, int output_rows=0, int row_bytes=0, bool scatter=False, int fill=0, "
          "int workers=0) -> ()",
          [](const std::string &stage, int64_t batch, int64_t window, int64_t capacity,
             int64_t map_rows, int64_t source_rows, int64_t output_rows, int64_t row_bytes,
             bool scatter, int64_t fill, int64_t workers) {
              if (stage == "metadata") {
                  TORCH_CHECK(batch >= 0 && batch <= INT32_MAX && window > 0 &&
                                  window <= INT32_MAX && capacity >= 0 && capacity <= INT32_MAX,
                              "invalid metadata geometry");
                  metadata_entry(window, capacity).prepare();
              } else {
                  TORCH_CHECK(stage == "copy" && map_rows > 0 && source_rows >= 0 &&
                                  output_rows >= 0 && row_bytes > 0 && (fill == 0 || fill == -1),
                              "invalid copy geometry");
                  const auto budget = native_host::core_count(c10_npu::current_device());
                  TORCH_CHECK(workers >= 0 && workers <= budget, "invalid worker budget");
                  copy_entry(row_bytes, scatter, fill == -1 ? UINT8_MAX : 0,
                             workers ? workers : budget).prepare();
              }
          });
}
