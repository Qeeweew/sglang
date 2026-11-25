#include "launch.h"
#include "actual_entry.h"
#include "torch.h"

namespace h32_quant {
namespace {
native_jit::Kernel &entry(uint32_t workers, uint32_t width, bool bf16, bool interleaved) {
    static thread_local native_jit::ActualEntries<4> entries;
    return entries.get({width, bf16, interleaved, workers}, [&] {
        return std::make_unique<native_jit::Kernel>(
            "h32_quant", "h32_quant/h32.asc",
            std::vector<std::string>{"--cce-auto-sync=off", "-DH32_WIDTH=" + std::to_string(width),
                                     "-DH32_BF16=" + std::to_string(bf16),
                                     "-DH32_INTERLEAVED=" + std::to_string(interleaved)},
            workers);
    });
}
} // namespace

void launch(void *stream, uint32_t workers, uint32_t width, bool bf16, bool interleaved,
            uint8_t *input, uint8_t *indices, uint8_t *codes, uint8_t *scales, int64_t rows,
            uint8_t *count) {
    entry(workers, width, bf16, interleaved)
        .launch_persistent(stream, rows, input, indices, codes, scales, rows, count);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_h32_quant(int width, bool bf16, bool interleaved) -> ()",
          [](int64_t width, bool bf16, bool interleaved) {
              TORCH_CHECK_VALUE(valid_width(width), "invalid H32 width");
              const int workers = native_host::core_count(c10_npu::current_device());
              entry(workers, width, bf16, interleaved).prepare();
          });
}
} // namespace h32_quant
