#include "launch.h"
#include "jit.h"
#include "torch.h"

namespace dspark_markov {
namespace {
native_jit::Kernel &entry(bool packet) {
    static thread_local native_jit::Kernel kernel(
        "markov_merge_candidates", "dspark_markov/merge.asc", {"--cce-auto-sync=off"},
        native_host::core_count(c10_npu::current_device()));
    static thread_local native_jit::Kernel fold(
        "markov_fold_packets", "dspark_markov/merge.asc", {"--cce-auto-sync=off", "-DMARKOV_EMIT_PACKET=1"},
        native_host::core_count(c10_npu::current_device()));
    return packet ? fold : kernel;
}
native_jit::Kernel &local_entry() {
    static thread_local native_jit::Kernel kernel(
        "markov_local_greedy", "dspark_markov/local.asc", {"--cce-auto-sync=off"},
        native_host::core_count(c10_npu::current_device()));
    return kernel;
}
}

void launch(void *stream, uint8_t *candidates, uint8_t *tokens, int64_t ranks, int64_t rows, bool packet) {
    entry(packet).launch_persistent(stream, (rows + kRowsPerTile - 1) / kRowsPerTile, candidates,
                              tokens, ranks, rows);
}
void select(void *stream, uint8_t *base, uint8_t *bias, uint8_t *out, int64_t rows,
            int64_t width, int64_t parts, int64_t base_stride, int64_t bias_stride,
            uint32_t start, uint32_t formats) {
    local_entry().launch_persistent(stream, rows * parts, base, bias, out, rows, width, parts,
                                    base_stride, bias_stride, start, formats);
}

TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_dspark_markov() -> ()", [] {
        entry(false).prepare();
        entry(true).prepare();
        local_entry().prepare();
    });
}
} // namespace dspark_markov
