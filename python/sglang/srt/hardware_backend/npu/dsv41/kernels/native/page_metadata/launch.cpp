#include "launch.h"
#include "actual_entry.h"
#include <algorithm>
#include <array>
#include <memory>
#include <torch/library.h>
namespace {
// Key order: request_rows, context stride, mapping_size, page_rows, window_rows, workers.
// Proposal block is stable geometry; request batch and prefixes are runtime.
using MetadataKey = std::array<int64_t, 7>;
native_jit::Kernel &entry(const MetadataKey &key) {
    static thread_local native_jit::KernelCache<MetadataKey> entries;
    return entries.get(key, [&] {
        const std::vector<std::string> flags{
            "--cce-auto-sync=off",
            "-DPAGE_METADATA_REQUEST_ROWS=" + std::to_string(key[0]),
            "-DPAGE_METADATA_CONTEXT=" + std::to_string(key[1]),
            "-DPAGE_METADATA_MAPPING_SIZE=" + std::to_string(key[2]),
            "-DPAGE_METADATA_PAGE_ROWS=" + std::to_string(key[3]),
            "-DPAGE_METADATA_WINDOW_ROWS=" + std::to_string(key[4]),
            "-DPAGE_METADATA_WORKERS=" + std::to_string(key[5]),
            "-DPAGE_METADATA_BLOCK=" + std::to_string(key[6])};
        return std::make_unique<native_jit::Kernel>(
            "draft_page_metadata", "page_metadata/page_metadata.asc", flags, uint32_t(key[5]));
    });
}
} // namespace
extern "C" void
dsv41_draft_page_metadata_launch(void *stream, uint32_t blocks, uint8_t *req_to_token,
                                 uint8_t *full_to_swa, uint8_t *requests, uint8_t *prefix,
                                 uint8_t *full_slots, uint8_t *cu, uint8_t *lengths, uint8_t *table,
                                 uint8_t *slots, uint8_t *routed_rows, int64_t batch, int64_t block,
                                 int64_t request_rows, int64_t context, int64_t /* columns */,
                                 int64_t mapping_size, int64_t page_rows, int64_t window_rows) {
    entry({request_rows, context, mapping_size, page_rows, window_rows, blocks, block})
        .launch_persistent(stream, std::max<int64_t>(1, (batch + page_metadata::kRequestsPerGroup - 1) /
                                       page_metadata::kRequestsPerGroup),
                req_to_token, full_to_swa, requests, prefix, full_slots, cu,
                lengths, table, slots, routed_rows, int32_t(batch));
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def(
        "precompile_page_metadata(int request_rows, int context, int mapping_size, int page_rows, "
        "int window_rows, int workers, int batch, int block) -> ()",
        [](int64_t request_rows, int64_t context, int64_t mapping_size, int64_t page_rows,
           int64_t window_rows, int64_t workers, int64_t batch, int64_t block) {
            TORCH_CHECK(request_rows > 0 && context > 0 && mapping_size > 0 && page_rows > 0 &&
                            page_rows <= INT32_MAX && window_rows >= 0 &&
                            window_rows <= INT32_MAX && workers > 0 && workers <= UINT32_MAX &&
                            batch >= 0 && batch <= INT32_MAX && block >= 0 && block <= INT32_MAX &&
                            batch * block <= INT32_MAX,
                        "actual metadata layout and worker plan required");
            entry({request_rows, context, mapping_size, page_rows, window_rows, workers, block})
                .prepare();
        });
}
