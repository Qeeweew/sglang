#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
#include "format.h"
#include <algorithm>
namespace {
using Entries = native_jit::ActualEntries<5>;
native_jit::Kernel &entry(int64_t k, int64_t n, int64_t panel_k, bool compact,
                          uint32_t workers) {
    static thread_local Entries entries;
    const Entries::Key key{uint64_t(k), uint64_t(n), uint64_t(panel_k), compact, workers};
    return entries.get(key, [&] {
        std::string tag = "mxfp4_weight";
        for (auto value : key)
            tag += "_" + std::to_string(value);
        std::vector<std::string> flags{"--cce-auto-sync=off",
                                       "-DMXFP4_WEIGHT_K=" + std::to_string(k),
                                       "-DMXFP4_WEIGHT_N=" + std::to_string(n),
                                       "-DMXFP4_WEIGHT_PANEL_K=" + std::to_string(panel_k),
                                       "-DMXFP4_WEIGHT_COMPACT=" + std::to_string(compact),
                                       "-DMXFP4_WEIGHT_WORKERS=" + std::to_string(workers)};
        return std::make_unique<native_jit::Kernel>(tag.c_str(), "mxfp4_weight/mxfp4_weight.asc",
                                                    std::move(flags), workers);
    });
}
} // namespace
extern "C" void dsv41_mxfp4_pack(void *stream, uint32_t blocks, void *codes, void *scales,
                                 void *tables, void *packed, void *scale_output, int64_t k,
                                 int64_t n, int64_t panel_k, bool compact) {
    entry(k, n, panel_k, compact, blocks)
        .launch_persistent(stream, uint64_t(k / panel_k) * (n / mxfp4_weight::kPackColumns), codes, scales, tables, packed, scale_output);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def(
        "precompile_mxfp4_weight(int k, int n, int panel_k, int workers, bool "
        "compact=True) -> ()",
        [](int64_t k, int64_t n, int64_t panel_k, int64_t workers, bool compact) {
            TORCH_CHECK_VALUE(k > 0 && n > 0 && k % 32 == 0 && panel_k >= 32 && panel_k <= 512 &&
                                  panel_k % 32 == 0 && k % panel_k == 0 &&
                                  n % 64 == 0 &&
                                  workers > 0 &&
                                  workers <= native_host::core_count(c10_npu::current_device()),
                              "actual MXFP4 packing geometry and worker plan required");
            entry(k, n, panel_k, compact, uint32_t(workers)).prepare();
        });
}
