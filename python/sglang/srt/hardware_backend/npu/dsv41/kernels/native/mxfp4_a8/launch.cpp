#include "plan.h"
#include "jit.h"
#include <array>
#include <map>

namespace mxfp4_a8 {
void launch(void *stream, int depth, int width, int workers, const Inputs &inputs, bool g32, bool bf16) {
    static thread_local std::map<std::array<int, 5>, std::unique_ptr<native_jit::Kernel>> kernels;
    auto &entry = kernels[{depth, width, workers, g32, bf16}];
    if (!entry) {
        const auto tag = (g32 ? "int4_g32_a8_" : "mxfp4_a8_") + std::to_string(depth) + "_" + std::to_string(width) + "_" +
                         std::to_string(workers);
        entry = std::make_unique<native_jit::Kernel>(
            tag.c_str(), "mxfp4_a8/gemm.asc",
            std::vector<std::string>{"--cce-auto-sync=off", "-DA8_DEPTH=" + std::to_string(depth),
                                     "-DA8_WIDTH=" + std::to_string(width),
                                     "-DA8_G32=" + std::to_string(g32),
                                     "-DA8_BF16=" + std::to_string(bf16),
                                     "-DA8_WORKERS=" + std::to_string(workers)},
            workers);
    }
    entry->launch_persistent(stream, workers, inputs);
}
} // namespace mxfp4_a8
