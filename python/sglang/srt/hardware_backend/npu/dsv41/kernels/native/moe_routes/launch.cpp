#include "launch.h"
#include "actual_entry.h"
#include "conversion.h"
#include "torch.h"
namespace {
enum class Stage : uint64_t { Gather, Combine, GatherA8, CombineBF16 };
native_jit::Kernel &entry(Stage stage, uint64_t rows, uint64_t hidden, uint64_t topk,
                          uint64_t workers) {
    static thread_local native_jit::ActualEntries<5> entries;
    const uint64_t capacity = rows >= 4 * workers
        ? moe_routes::kBandwidthCapacity : moe_routes::kLatencyCapacity;
    return entries.get({uint64_t(stage), capacity, hidden, topk, workers}, [&] {
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        auto field = [&](const char *name, uint64_t value) {
            flags.emplace_back(std::string("-DMOE_ROUTES_") + name + "=" + std::to_string(value));
        };
        field("WORKERS", workers);
        const bool gather = stage == Stage::Gather;
        field(stage == Stage::GatherA8 ? "GATHER_A8" : gather ? "GATHER" : "COMBINE", 1);
        field("BF16", stage == Stage::CombineBF16);
        field("HIDDEN", hidden);
        field("TOPK", topk);
        field("CAPACITY", capacity);
        return std::make_unique<native_jit::Kernel>(
            gather ? "route_gather" : stage == Stage::GatherA8 ? "route_gather_a8" : "route_combine",
            "moe_routes/moe_routes.asc", flags, uint32_t(workers));
    });
}
} // namespace
extern "C" void dsv41_route_gather(void *stream, uint32_t blocks, void *x, void *order, void *meta,
                                   void *out, void *inverse, int64_t rows, int64_t hidden,
                                   int64_t topk, int64_t output_rows) {
    const uint64_t capacity = rows >= 4 * blocks
        ? moe_routes::kBandwidthCapacity : moe_routes::kLatencyCapacity;
    if (!hidden || !output_rows) return;
    entry(Stage::Gather, rows, hidden, topk, blocks).launch_persistent(
        stream, uint64_t(output_rows) * ((hidden + capacity - 1) / capacity),
        x, order, meta, out, inverse, output_rows);
}
void dsv41_route_gather_a8(void *stream, uint32_t blocks, void *x, void *scales, void *order,
                           void *meta, void *out, void *sorted_scales, void *inverse, int64_t capacity,
                           int64_t hidden, int64_t topk) {
    if (capacity) entry(Stage::GatherA8, capacity, hidden, topk, blocks).launch_persistent(
        stream, capacity, x, scales, order, meta, out, sorted_scales, inverse, capacity);
}
extern "C" void dsv41_route_combine(void *stream, uint32_t blocks, void *x, void *inverse,
                                    void *ids, void *weights, void *out, int64_t rows,
                                    int64_t hidden, int64_t topk, bool bf16) {
    const uint64_t capacity = rows >= 4 * blocks
        ? moe_routes::kBandwidthCapacity : moe_routes::kLatencyCapacity;
    if (hidden)
        entry(bf16 ? Stage::CombineBF16 : Stage::Combine, rows, hidden, topk, blocks).launch_persistent(
            stream, uint64_t(rows) * ((hidden + capacity - 1) / capacity),
            x, inverse, ids, weights, out, rows);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_moe_routes(str stage, int workers, int rows, int hidden, int topk) -> ()",
          [](const std::string &stage, int64_t workers, int64_t rows, int64_t hidden, int64_t topk) {
              TORCH_CHECK(workers > 0 && workers <= native_host::core_count(c10_npu::current_device()) &&
                              (stage == "gather" || stage == "combine") &&
                              rows > 0 && hidden > 0 && topk > 0,
                          "invalid conversion geometry");
              entry(stage == "gather" ? Stage::Gather : Stage::Combine,
                    rows, hidden, topk, workers).prepare();
          });
}
