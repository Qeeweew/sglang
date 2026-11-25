#include "launch.h"
#include "actual_entry.h"
#include "torch.h"

namespace h32_quant {
namespace {
native_jit::Kernel &entry(uint32_t workers) {
    static thread_local native_jit::ActualEntries<1> entries;
    return entries.get({workers}, [workers] {
        return std::make_unique<native_jit::Kernel>(
            "h32_quant", "h32_quant/h32.asc",
            std::vector<std::string>{"--cce-auto-sync=off"}, workers);
    });
}
} // namespace

void launch(void *stream, uint32_t workers, uint32_t width, bool bf16, bool interleaved,
            uint8_t *input, uint8_t *codes, uint8_t *scales, uint8_t *count, int64_t rows) {
    entry(workers).launch_persistent(stream, rows, input, codes, scales, count, rows, width,
                                     uint32_t(bf16), uint32_t(interleaved));
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_h32_quant(int width, bool bf16, bool interleaved) -> ()",
          [](int64_t width, bool, bool) {
              TORCH_CHECK_VALUE(valid_width(width), "invalid H32 width");
              const int workers = native_host::core_count(c10_npu::current_device());
              entry(workers).prepare();
          });
}
} // namespace h32_quant
