#include "launch.h"
#include "layout.h"
#include "actual_entry.h"
#include "torch.h"
namespace {
enum class Operation : uint64_t { Rms, SwiGlu };
using Entries = native_jit::ActualEntries<8>;
native_jit::Kernel &entry(Operation operation, uint64_t rows, uint32_t width, uint32_t budget,
                          bool packed = false, bool fp16 = false, bool prefix = false) {
    static thread_local Entries entries;
    const uint32_t tile_rows =
        operation == Operation::Rms ? 1 : elementwise::tile_rows(rows, width, budget);
    const uint64_t tiles = (rows + tile_rows - 1) / tile_rows;
    const uint32_t workers = budget;
    const Entries::Key key{
        uint64_t(operation), uint64_t(tiles <= workers), width, tile_rows, packed, fp16, prefix, workers};
    return entries.get(key, [&] {
        std::string tag = operation == Operation::Rms ? "rms_norm" : "swiglu";
        for (auto value : key)
            tag += "_" + std::to_string(value);
        std::vector<std::string> flags{"--cce-auto-sync=off",
                                       "-DELEMENTWISE_WIDTH=" + std::to_string(width),
                                       "-DELEMENTWISE_TILE_ROWS=" + std::to_string(tile_rows),
                                       "-DELEMENTWISE_WORKERS=" + std::to_string(workers),
                                       "-DELEMENTWISE_PACKED=" + std::to_string(packed),
                                       "-DELEMENTWISE_PREFIX=" + std::to_string(prefix)};
        if (operation == Operation::Rms)
            flags.emplace_back("-DELEMENTWISE_RMS=1");
        if (fp16)
            flags.emplace_back("-DELEMENTWISE_STORAGE_FP16=1");
        if (tiles <= workers)
            flags.emplace_back("-DELEMENTWISE_SINGLE_TILE=1");
        return std::make_unique<native_jit::Kernel>(tag.c_str(), "elementwise/elementwise.asc",
                                                    std::move(flags), workers);
    });
}
native_jit::Kernel &split_entry(int64_t rows, uint32_t first_width, uint32_t second_width,
                                uint32_t budget) {
    const uint32_t workers = budget / 2 * 2;
    TORCH_CHECK_VALUE(workers >= 2, "split RMSNorm requires two vector workers");
    const bool single_wave = rows <= workers / 2;
    static thread_local native_jit::ActualEntries<4> entries;
    return entries.get({first_width, second_width, workers, single_wave}, [&] {
        std::vector<std::string> flags{"--cce-auto-sync=off", "-DELEMENTWISE_RMS=1",
            "-DELEMENTWISE_WIDTH=" + std::to_string(first_width),
            "-DELEMENTWISE_SECOND_WIDTH=" + std::to_string(second_width),
            "-DELEMENTWISE_WORKERS=" + std::to_string(workers)};
        if (single_wave) flags.emplace_back("-DELEMENTWISE_SINGLE_TILE=1");
        return std::make_unique<native_jit::Kernel>("split_rms_norm", "elementwise/elementwise.asc",
                                                    std::move(flags), workers);
    });
}
} // namespace
void split_rms_norm_launch(void *stream, uint32_t budget, uint8_t *input, uint8_t *first_weight,
                           uint8_t *second_weight, uint8_t *first_out, uint8_t *second_out,
                           uint32_t first_width, uint32_t second_width, float first_eps,
                           float second_eps, int64_t rows, int64_t stride) {
    split_entry(rows, first_width, second_width, budget)
        .launch_persistent(stream, uint64_t(rows) * 2, input, first_weight, second_weight,
                           first_out, second_out, first_eps, second_eps, rows, stride);
}
extern "C" void dsv41_rms_norm_launch(void *stream, uint32_t blocks, uint8_t *input,
                                      uint8_t *weight, uint8_t *output, int64_t rows,
                                      uint32_t width, float epsilon) {
    entry(Operation::Rms, rows, width, blocks)
        .launch_persistent(stream, rows, input, weight, output, epsilon, rows);
}
extern "C" void dsv41_swiglu_launch(void *stream, uint32_t blocks, uint8_t *gate, uint8_t *up,
                                    uint8_t *count, uint8_t *output, int64_t capacity,
                                    uint32_t width, float limit, bool packed, bool fp16) {
    const uint32_t tile_rows = elementwise::tile_rows(capacity, width, blocks);
    entry(Operation::SwiGlu, capacity, width, blocks, packed, fp16, count != nullptr)
        .launch_persistent(stream,
            (capacity + tile_rows - 1) / tile_rows, gate, up, count, output, limit, capacity);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_elementwise(str operation, int rows, int width, int workers, bool "
          "packed=False, "
          "bool fp16=False, bool prefix=False, int second_width=0) -> ()",
          [](const std::string &operation, int64_t rows, int64_t width, int64_t workers,
             bool packed, bool fp16, bool prefix, int64_t second_width) {
              using namespace elementwise;
              TORCH_CHECK_VALUE(rows > 0 && width > 0 && width <= UINT32_MAX && workers > 0 &&
                                    workers <= native_host::core_count(c10_npu::current_device()),
                                "actual elementwise geometry/worker budget required");
              if (operation == "split_rms_norm") {
                  TORCH_CHECK_VALUE(!packed && !fp16 && !prefix && width <= kReductionPanelElements &&
                                    width % kFp32VectorElements == 0 && second_width > 0 &&
                                    second_width <= kReductionPanelElements &&
                                    second_width % kFp32VectorElements == 0, "invalid split RMSNorm widths");
                  split_entry(rows, uint32_t(width), uint32_t(second_width), uint32_t(workers)).prepare();
                  return;
              }
              TORCH_CHECK_VALUE(second_width == 0, "second_width is only for split RMSNorm");
              if (operation == "rms_norm") {
                  TORCH_CHECK_VALUE(!packed && !fp16 && !prefix &&
                                        width <= 2 * kReductionPanelElements &&
                                        width % kFp32VectorElements == 0 &&
                                        (width <= kReductionPanelElements ||
                                         width % (2 * kFp32VectorElements) == 0),
                                    "invalid RMSNorm geometry/storage");
                  entry(Operation::Rms, rows, uint32_t(width), uint32_t(workers)).prepare();
              } else {
                  TORCH_CHECK_VALUE(operation == "swiglu" && width % 32 == 0 &&
                                        SwiGluLayout(align_elements(uint32_t(width))).bytes <=
                                            arch2201::kUbBytes,
                                    "invalid SwiGLU geometry");
                  entry(Operation::SwiGlu, rows, uint32_t(width), uint32_t(workers), packed, fp16,
                        prefix)
                      .prepare();
              }
          });
}
