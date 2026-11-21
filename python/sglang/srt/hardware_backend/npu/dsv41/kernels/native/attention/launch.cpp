#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
#include <sstream>
#include <cmath>
namespace {
using attention::Config;
using Entries = native_jit::ActualEntries<16>;
native_jit::Kernel &entry(const Config &c) {
    static thread_local Entries entries;
    const bool sparse = c.mode == int32_t(attention::Mode::Sparse);
    const Entries::Key key{
        uint64_t(c.mode),
        uint64_t(c.tile_rows),
        uint64_t(c.heads),
        uint64_t(c.width),
        uint64_t(c.columns),
        uint64_t(sparse ? c.compressed_columns : 0),
        uint64_t(c.page_rows),
        uint64_t(c.window_rows),
        uint64_t(sparse ? c.selected_rows : 0),
        uint64_t(c.mode == int32_t(attention::Mode::Draft) ? c.capacity : 0),
        uint64_t(sparse ? c.ratio : 1),
        uint64_t(c.cores),
        sparse && c.window_ring,
        sparse && c.coordinate32,
        sparse && c.coordinate_elements32,
        uint64_t(c.verify_ring_capacity)};
    return entries.get(key, [&] {
        static constexpr const char *names[]{"MODE",
                                             "TILE_ROWS",
                                             "HEADS",
                                             "WIDTH",
                                             "PAGE_COLUMNS",
                                             "COMPRESSED_COLUMNS",
                                             "PAGE_ROWS",
                                             "WINDOW_ROWS",
                                             "SELECTED_ROWS",
                                             "DRAFT_CAPACITY",
                                             "RATIO",
                                             "WORKER_BUDGET",
                                             "WINDOW_RING",
                                             "COORDINATE32",
                                             "COORDINATE_ELEMENTS32",
                                             "VERIFY_RING_CAPACITY"};
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        for (size_t i = 0; i < key.size(); ++i)
            flags.emplace_back(std::string("-DATTENTION_") + names[i] + "=" +
                               std::to_string(key[i]));
        // Same float sqrt and division as the checked host geometry.
        std::ostringstream scale;
        scale << std::hexfloat << (1.0f / std::sqrt(float(c.width))) << 'f';
        flags.emplace_back("-DATTENTION_SCALE=" + scale.str());
        const char *name = sparse        ? "sparse_attention"
                           : c.mode == 0 ? "window_attention"
                                         : "draft_attention";
        return std::make_unique<native_jit::Kernel>(name, "attention/attention.asc", flags, c.cores);
    });
}
} // namespace
extern "C" void dsv41_attention_launch(void *stream, const attention::Arguments &args) {
    const auto &c = args.config;
    auto s = args.sources;
    s.tokens = c.tokens;
    s.batch = c.batch;
    if (c.mode == int32_t(attention::Mode::Sparse) || c.verify_ring_capacity)
        entry(c).launch_persistent(stream, c.tokens, s);
    else
        entry(c).launch_persistent(
            stream, c.tokens, attention::DirectSources{s.q, s.kv, s.table, s.cu, s.lengths, s.window,
                                                    s.sinks, s.out, s.workspace, c.tokens, c.batch});
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_attention(int tokens, int heads, int width, int batch, int columns, int "
          "page_rows, "
          "int window_rows, int mode=0, int ratio=1, int selected_rows=0, int capacity=0, "
          "int compressed_columns=0, int compressed_pages=0, int workers=0, bool ring=False) -> ()",
          [](int64_t tokens, int64_t heads, int64_t width, int64_t batch, int64_t columns,
             int64_t page_rows, int64_t window_rows, int64_t mode, int64_t ratio,
             int64_t selected_rows, int64_t capacity, int64_t compressed_columns,
             int64_t compressed_pages, int64_t workers, bool ring) {
              const auto available = native_host::core_count(c10_npu::current_device(), true);
              TORCH_CHECK(tokens > 0 && workers >= 0 && workers <= available,
                          "actual nonempty token capacity and worker budget required");
              TORCH_CHECK(mode == 1 || ratio == 1, "direct attention has one KV source ratio");
              if (mode == 0)
                  capacity = (window_rows + attention::kDotRows - 1) /
                             attention::kDotRows * attention::kDotRows;
              else if (mode == 1)
                  capacity = attention::kSparseWindowCapacity + selected_rows;
              auto c =
                  native_host::attention_geometry(tokens, heads, batch, columns, compressed_columns,
                                            capacity, ratio, mode, workers ? workers : available,
                                            ring, width, page_rows, window_rows, selected_rows);
              if (mode == 1) {
                  TORCH_CHECK(compressed_pages > 0, "actual compressed cache pages required");
                  c.coordinate32 = c.coordinate_shift > 0 &&
                                   compressed_pages <= INT32_MAX / c.compressed_page_rows;
                  c.coordinate_elements32 =
                      __int128(compressed_pages) * c.compressed_page_rows * width <= INT32_MAX;
              }
              entry(c).prepare();
          });
}
