#include "launch.h"
#include "actual_entry.h"
#include "torch.h"
namespace {
using Entries = native_jit::ActualEntries<3>;
native_jit::Kernel &entry(int64_t vocab, uint32_t width, uint32_t workers) {
    static thread_local Entries entries;
    const Entries::Key key{uint64_t(vocab), uint64_t(width), workers};
    return entries.get(key, [&] {
        std::string tag = "engram";
        for (auto value : key)
            tag += "_" + std::to_string(value);
        std::vector<std::string> flags{
            "--cce-auto-sync=off",
            "-DENGRAM_VOCAB=" + std::to_string(vocab), "-DENGRAM_WIDTH=" + std::to_string(width),
            "-DENGRAM_WORKERS=" + std::to_string(workers)};
        return std::make_unique<native_jit::Kernel>(tag.c_str(), "engram/engram.asc",
                                                    std::move(flags), workers);
    });
}
} // namespace
extern "C" void dsv41_engram_launch(void *stream, uint32_t blocks, uint8_t *codes, uint8_t *scales,
                                    uint8_t *ids, uint8_t *out, uint8_t *tables, int64_t rows,
                                    int64_t vocab, uint32_t columns) {
    entry(vocab, columns, blocks).launch_persistent(stream, rows, codes, scales,
        ids, out, tables, rows);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_engram(int rows, int vocab, int width, int workers) -> ()",
          [](int64_t rows, int64_t vocab, int64_t width, int64_t workers) {
              TORCH_CHECK_VALUE(rows > 0 && vocab >= 0 && width > 0 && width <= UINT32_MAX &&
                                    width % 32 == 0 && workers > 0 &&
                                    workers <= native_host::core_count(c10_npu::current_device()),
                                "actual decode geometry and valid worker plan required");
              entry(vocab, uint32_t(width), uint32_t(workers)).prepare();
          });
}
