#include "launch.h"
#include "actual_entry.h"
#include "torch.h"

namespace {
native_jit::Kernel &entry(const w8a16::Arguments &a) {
    using Entries = native_jit::ActualEntries<9>;
    static thread_local Entries entries;
    const Entries::Key key{uint64_t(a.k), uint64_t(a.n), uint64_t(a.batches), uint64_t(a.bf16),
                           uint64_t(a.bm), uint64_t(a.bn), uint64_t(a.bk), uint64_t(a.workers), a.output_bf16};
    return entries.get(key, [&] {
        std::string tag = "w8a16";
        for (auto value : key) tag += "_" + std::to_string(value);
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        auto field = [&](const char *name, int value) {
            flags.push_back(std::string("-DW8A16_") + name + "=" + std::to_string(value));
        };
        field("K", a.k); field("N", a.n); field("BATCHES", a.batches); field("BF16", a.bf16);
        field("OUTPUT_BF16", a.output_bf16);
        field("BM", a.bm); field("BN", a.bn); field("BK", a.bk); field("WORKERS", a.workers);
        return std::make_unique<native_jit::Kernel>(tag.c_str(), "w8a16/w8a16.asc",
                                                    std::move(flags), a.workers);
    });
}
} // namespace
void w8a16_launch(void *stream, const w8a16::Arguments &a) {
    const uint64_t jobs = w8a16::jobs(a.rows, a.n, a.batches, a.bm, a.bn);
    entry(a).launch_persistent(stream, jobs, a.input, a.weight, a.scales, a.output, a.workspace, a.rows);
}
void w8a16_quantize_launch(void *stream, const w8a16::QuantizeSources &sources, int k, int workers) {
    const int tile = sources.rows <= 8 ? 8 : sources.rows <= 16 ? 16 : 32;
    using Entries = native_jit::ActualEntries<3>;
    static thread_local Entries entries;
    const Entries::Key key{uint64_t(k), uint64_t(tile), uint64_t(workers)};
    auto &kernel = entries.get(key, [&] {
        const auto k_str = std::to_string(k);
        const auto tile_str = std::to_string(tile);
        const auto worker_str = std::to_string(workers);
        const auto tag = "quantize_g256_" + k_str + "_" + tile_str + "_" + worker_str;
        return std::make_unique<native_jit::Kernel>(
            tag.c_str(), "w8a16/quantize.asc",
            std::vector<std::string>{"--cce-auto-sync=off", "-DW8A16_QUANT_K=" + k_str,
                                     "-DW8A16_QUANT_ROWS=" + tile_str,
                                     "-DW8A16_QUANT_WORKERS=" + worker_str},
            workers);
    });
    const uint64_t jobs = uint64_t((sources.rows + tile - 1) / tile) * (k / w8a16::kGroup);
    kernel.launch_persistent(stream, jobs, sources);
}

TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_w8a16(int k, int n, int batches, bool bf16, int bm, int bn, int bk, int workers=0, "
          "bool? output_bf16=None) -> ()",
          [](int64_t k, int64_t n, int64_t batches, bool bf16, int64_t bm, int64_t bn,
             int64_t bk, int64_t workers, std::optional<bool> output_bf16) {
              TORCH_CHECK_VALUE(k > 0 && k <= UINT16_MAX && k % w8a16::kGroup == 0 &&
                                    n > 0 && n <= INT32_MAX && n % 128 == 0 && batches > 0 &&
                                    batches * k <= UINT16_MAX && uint64_t(batches) * n <= UINT32_MAX / 2 &&
                                    (bm == 16 || bm == 32 || bm == 64 || bm == 128 || bm == 256) &&
                                    (bn == 64 || bn == 128 || bn == 256) && n % bn == 0 &&
                                    (bk == 128 || bk == 256 || bk == 512) && k % bk == 0 &&
                                    w8a16::Storage(bm, bn, bk, bf16).fits(),
                                "invalid W8A16 stable geometry/resource plan");
              const auto available = native_host::core_count(c10_npu::current_device(), true);
              if (!workers) workers = available;
              TORCH_CHECK_VALUE(workers > 0 && workers <= available, "invalid W8A16 worker budget");
              w8a16::Arguments a{};
              a.k = k; a.n = n; a.batches = batches; a.bf16 = bf16;
              a.output_bf16 = output_bf16.value_or(bf16);
              TORCH_CHECK_VALUE(!bf16 || a.output_bf16, "unsupported BF16 to FP16 W8A16");
              a.bm = bm; a.bn = bn; a.bk = bk; a.workers = workers;
              entry(a).prepare();
          });
}
