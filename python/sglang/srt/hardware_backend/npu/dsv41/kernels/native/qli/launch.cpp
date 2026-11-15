#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
#include <qli/layout.h>
#include <algorithm>
namespace {
using Entries = native_jit::ActualEntries<13>;
using qli::Config;
native_jit::Kernel &entry(const Config &c, uint32_t score_workers, bool merge) {
    static thread_local Entries entries;
    const auto workers = merge ? 2 * score_workers : score_workers;
    const Entries::Key key{c.heads, c.depth, c.topk, c.candidate_mode, c.page_rows,
        c.page_columns, c.key_page_stride, c.scale_page_stride, c.ratio, score_workers,
        merge, c.query_tile_rows, c.key_splits};
    return entries.get(key, [&] {
        static constexpr const char *names[]{
            "HEADS", "DEPTH", "TOPK", "MODE", "PAGE_ROWS", "PAGE_COLUMNS",
            "KEY_PAGE_STRIDE", "SCALE_PAGE_STRIDE", "RATIO", "SCORE_WORKERS", "MERGE",
            "QUERY_ROWS", "KEY_SPLITS"};
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        for (size_t i = 0; i < key.size(); ++i)
            flags.emplace_back(std::string("-DQLI_") + names[i] + "=" + std::to_string(key[i]));
        flags.emplace_back("-DQLI_WORKERS=" + std::to_string(workers));
        return std::make_unique<native_jit::Kernel>(merge ? "qli_merge" : "qli_score",
                                                    "qli/qli.asc", flags, workers);
    });
}
} // namespace
extern "C" void dsv41_qli_launch(void *stream, uint32_t blocks, const qli::Config *config,
                                 uint8_t *q, uint8_t *k, uint8_t *w, uint8_t *qs, uint8_t *ks,
                                 uint8_t *cu, uint8_t *lengths, uint8_t *residual, uint8_t *pages,
                                 uint8_t *candidates, uint8_t *indices, uint8_t *candidate_out,
                                 uint8_t *workspace) {
    const qli::Sources sources{
        uintptr_t(q),        uintptr_t(k),          uintptr_t(w),       uintptr_t(qs),
        uintptr_t(ks),       uintptr_t(cu),         uintptr_t(lengths), uintptr_t(residual),
        uintptr_t(pages),    uintptr_t(candidates), uintptr_t(indices), uintptr_t(candidate_out),
        uintptr_t(workspace), config->rows, config->batch};
    entry(*config, blocks, false).launch_persistent(stream, uint64_t(config->rows) * config->key_splits, sources);
    if (config->key_splits > 1)
        entry(*config, blocks, true).launch_persistent(stream, config->rows, sources);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_qli(int rows, int batch, int heads, int depth, int topk, int mode, int "
          "page_rows, "
          "int page_columns, int key_page_stride, int scale_page_stride, int ratio, "
          "int score_workers=0, bool merge=False) -> ()",
          [](int64_t rows, int64_t batch, int64_t heads, int64_t depth, int64_t topk, int64_t mode,
             int64_t page_rows, int64_t page_columns, int64_t key_page_stride,
             int64_t scale_page_stride, int64_t ratio, int64_t score_workers, bool merge) {
              using namespace qli;
              const auto available = native_host::core_count(c10_npu::current_device(), true);
              TORCH_CHECK(rows > 0 && rows <= UINT32_MAX && batch > 0 && batch <= UINT32_MAX &&
                              heads > 0 && heads <= kHeadCapacity && heads % kCubeRows == 0 &&
                              depth > 0 && depth <= kDepthCapacity &&
                              depth % kInt8DepthAlignment == 0 && topk > 0 &&
                              topk <= kKeyTileCapacity && topk % kCandidateBlockTokens == 0 &&
                              mode >= kSourceCandidates && mode <= kNoCandidates &&
                              page_rows >= kCubeRows && page_rows <= UINT32_MAX &&
                              page_rows % kCubeRows == 0 && page_columns > 0 &&
                              page_columns <= UINT32_MAX && key_page_stride >= page_rows * depth &&
                              key_page_stride <= UINT32_MAX && scale_page_stride >= page_rows &&
                              scale_page_stride <= UINT32_MAX && (ratio == 1 || ratio == 2) &&
                              score_workers >= 0 && score_workers <= available,
                          "actual QLI layout and valid persistent worker budget required");
              const auto workers = uint32_t(score_workers ? score_workers : available);
              Config c{uint32_t(rows),
                       uint32_t(batch),
                       uint32_t(heads),
                       uint32_t(depth),
                       uint32_t(page_rows),
                       uint32_t(page_columns),
                       uint32_t(key_page_stride),
                       uint32_t(scale_page_stride),
                       uint32_t(ratio),
                       uint32_t(topk),
                       uint32_t(mode),
                       workers,
                       plan_query_rows(rows, workers, heads),
                       plan_key_splits(rows, workers, heads, uint64_t(page_rows) * page_columns)};
              TORCH_CHECK(!merge || c.key_splits > 1,
                          "merge is used only with multiple K partitions");
              entry(c, workers, merge).prepare();
          });
}
