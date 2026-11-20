#include "plan.h"
#include "actual_entry.h"
#include "torch.h"

namespace int4_g32_a8 {
namespace {
native_jit::Kernel &entry(int depth, int width, int workers, bool bf16) {
    static thread_local native_jit::ActualEntries<4> entries;
    return entries.get({uint64_t(depth), uint64_t(width), uint64_t(workers), bf16}, [&] {
        const auto tag = "int4_g32_a8_" + std::to_string(depth) + "_" + std::to_string(width) +
                         "_" + std::to_string(workers);
        return std::make_unique<native_jit::Kernel>(
            tag.c_str(), "int4_g32_a8/gemm.asc",
            std::vector<std::string>{"--cce-auto-sync=off", "-DA8_DEPTH=" + std::to_string(depth),
                                     "-DA8_WIDTH=" + std::to_string(width),
                                     "-DA8_BF16=" + std::to_string(bf16),
                                     "-DA8_WORKERS=" + std::to_string(workers)},
            workers);
    });
}
} // namespace

void launch(void *stream, int depth, int width, int workers, const Inputs &inputs, bool bf16) {
    entry(depth, width, workers, bf16).launch_persistent(stream, workers, inputs);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_int4_g32_a8(int depth, int width, bool bf16=False) -> ()",
          [](int64_t depth, int64_t width, bool bf16) {
              TORCH_CHECK_VALUE(valid_geometry(depth, width), "unsupported INT4 G32 geometry");
              const int workers = native_host::core_count(c10_npu::current_device(), true);
              entry(depth, width, workers, bf16).prepare();
          });
}
} // namespace int4_g32_a8
