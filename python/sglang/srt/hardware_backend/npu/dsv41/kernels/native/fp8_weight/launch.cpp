#include "launch.h"
#include "format.h"
#include "actual_entry.h"
#include "torch.h"
namespace {
using Entries = native_jit::ActualEntries<3>;
native_jit::Kernel &entry(int64_t rows, int64_t cols, uint32_t workers) {
    static thread_local Entries entries;
    const Entries::Key key{uint64_t(rows), uint64_t(cols), workers};
    return entries.get(key, [&] {
        std::string tag = "fp8_weight";
        for (auto value : key)
            tag += "_" + std::to_string(value);
        std::vector<std::string> flags{"--cce-auto-sync=off",
                                       "-DFP8_WEIGHT_ROWS=" + std::to_string(rows),
                                       "-DFP8_WEIGHT_COLS=" + std::to_string(cols),
                                       "-DFP8_WEIGHT_WORKERS=" + std::to_string(workers)};
        return std::make_unique<native_jit::Kernel>(tag.c_str(), "fp8_weight/fp8_weight.asc",
                                                    std::move(flags), workers);
    });
}
} // namespace
extern "C" void dsv41_fp8_weight_launch(void *stream, uint32_t blocks, uint8_t *codes,
                                        uint8_t *scales, uint8_t *output, uint8_t *lut,
                                        uint8_t *groups, int64_t rows, int64_t cols) {
    const uint64_t jobs = uint64_t((rows + fp8_weight::kTileRows - 1) / fp8_weight::kTileRows) *
                          ((cols + fp8_weight::kTileColumns - 1) / fp8_weight::kTileColumns);
    entry(rows, cols, blocks).launch_persistent(stream, jobs, codes, scales, output, lut, groups);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_fp8_weight(int rows, int cols, int workers) -> ()",
          [](int64_t rows, int64_t cols, int64_t workers) {
              TORCH_CHECK_VALUE(rows > 0 && cols > 0 && workers > 0 &&
                                    workers <= native_host::core_count(c10_npu::current_device()),
                                "actual decode geometry and valid worker plan required");
              entry(rows, cols, uint32_t(workers)).prepare();
          });
}
