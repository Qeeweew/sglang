#include "launch.h"
#include "actual_entry.h"
#include <algorithm>
#include <torch/library.h>
namespace {
using Arguments = projection::Arguments;
using Entries = native_jit::ActualEntries<10>;
native_jit::Kernel &entry(const Arguments &a, bool grouped, bool fp32, uint32_t workers) {
    static thread_local Entries entries;
    const Entries::Key key{grouped ? 2U
                           : fp32  ? 1U
                                   : 0U,
                           uint64_t(a.m < a.tile_rows && a.k % a.panel_columns == 0),
                           a.k,
                           a.n,
                           a.groups,
                           a.tile_rows,
                           a.tile_columns,
                           a.panel_columns,
                           a.dot_columns,
                           workers};
    return entries.get(key, [&] {
        std::string tag = "projection";
        for (auto v : key)
            tag += "_" + std::to_string(v);
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        auto field = [&](const char *n, uint64_t v) {
            flags.push_back(std::string("-DPROJECTION_") + n + "=" + std::to_string(v));
        };
        field("GROUPED", grouped);
        field("FP32", fp32);
        field("SHORT_ROWS", a.m < a.tile_rows && a.k % a.panel_columns == 0);
        field("K", a.k);
        field("N", a.n);
        field("GROUPS", a.groups);
        field("TM", a.tile_rows);
        field("TN", a.tile_columns);
        field("PANEL", a.panel_columns);
        field("DOT", a.dot_columns);
        field("WORKERS", workers);
        return std::make_unique<native_jit::Kernel>(tag.c_str(), "projection/projection.asc",
                                                    std::move(flags), workers);
    });
}
} // namespace
extern "C" void dsv41_projection_launch(void *stream, const Arguments &a, bool grouped, bool fp32) {
    const uint64_t jobs = uint64_t((a.m + a.tile_rows - 1) / a.tile_rows) *
                          ((a.n + a.tile_columns - 1) / a.tile_columns) * a.groups;
    entry(a, grouped, fp32, a.cores).launch_persistent(stream, jobs, a.x, a.w, a.y, a.m);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_projection(int rows, int k, int n, int groups, bool grouped, bool fp32, int "
          "tile_rows, "
          "int tile_columns, int panel_columns, int dot_columns, int workers) -> ()",
          [](int64_t rows, int64_t k, int64_t n, int64_t groups, bool grouped, bool fp32,
             int64_t tm, int64_t tn, int64_t panel, int64_t dot, int64_t workers) {
              TORCH_CHECK(rows > 0 && k > 0 && n > 0 && groups > 0 && k <= INT32_MAX &&
                              n <= INT32_MAX && groups <= INT32_MAX && k % 16 == 0 && n % 16 == 0 &&
                              workers > 0 && workers <= INT32_MAX && (!grouped || !fp32) &&
                              (tm == 16 || tm == 32 || tm == 64 || tm == 128 || tm == 256) &&
                              (tn == 128 || tn == 256) && tm * tn <= 32768 && tm + tn <= 384 &&
                              panel > 0 && panel <= 256 && panel % 16 == 0 && dot > 0 &&
                              dot <= 64 && dot % 16 == 0 && dot <= panel,
                          "invalid actual projection geometry/plan");
              TORCH_CHECK(grouped || groups == 1, "dense projection has one weight matrix");
              TORCH_CHECK(
                  groups * k <= UINT16_MAX && (grouped || n <= UINT16_MAX) &&
                      uint64_t(groups) * n <= UINT32_MAX && panel <= 4 * dot,
                  "projection ND2NZ stride / FIX stride or chunk plan exceeds hardware fields");
              Arguments a{};
              a.m = rows;
              a.k = k;
              a.n = n;
              a.groups = groups;
              a.tile_rows = tm;
              a.tile_columns = tn;
              a.panel_columns = panel;
              a.dot_columns = dot;
              entry(a, grouped, fp32, workers).prepare();
          });
}
