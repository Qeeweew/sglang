#include "launch.h"
#include "layout.h"
#include "bucket_layout.h"
#include "actual_entry.h"
#include "torch.h"
namespace {
native_jit::Kernel &topk_entry(const router::Plan &p) {
    static thread_local native_jit::ActualEntries<5> entries;
    return entries.get({p.experts, p.topk, p.workers, uint64_t(p.dp_stride > 0), uint64_t(p.dp_int64)}, [&] {
        std::vector<std::string> flags{"--cce-auto-sync=off"};
        auto field = [&](const char *name, uint64_t value) {
            flags.emplace_back(std::string("-DROUTER_") + name + "=" + std::to_string(value));
        };
        field("EXPERTS", p.experts);
        field("TOP_K", p.topk);
        field("WORKERS", p.workers);
        field("HAS_DP", p.dp_stride > 0);
        field("DP_INT64", p.dp_int64);
        return std::make_unique<native_jit::Kernel>("router_topk", "router/router.asc", flags, p.workers);
    });
}
native_jit::Kernel &bucket_entry(uint32_t experts, uint32_t workers) {
    static thread_local native_jit::ActualEntries<2> entries;
    return entries.get({experts, workers}, [&] {
        return std::make_unique<native_jit::Kernel>(
            "route_bucket", "router/bucket.asc",
            std::vector<std::string>{"--cce-auto-sync=off",
                "-DROUTER_BUCKET_EXPERTS=" + std::to_string(experts),
                "-DROUTER_BUCKET_WORKERS=" + std::to_string(workers)}, workers);
    });
}
} // namespace
extern "C" void dsv41_router_launch(void *stream, const router::Sources &s, const router::Plan &p,
                                    float factor) {
    topk_entry(p).launch_persistent(stream,
        router_layout::task_blocks(p.rows, p.experts, p.topk, p.workers),
        s.logits, s.bias, s.ids, s.weights, s.live, s.alternate, s.images, s.dp_counts,
        factor, p.rows, p.dp_stride);
}
extern "C" void dsv41_route_bucket(void *stream, uint32_t workers, void *ids, void *local_ids,
                                   void *counts, void *offsets, void *order, void *metadata,
                                   void *histogram, int64_t slots, int32_t first, uint32_t experts) {
    const uint32_t active = std::min<uint64_t>(workers, std::max<int64_t>(1,
        (slots + router_bucket::kSlotsPerTask - 1) / router_bucket::kSlotsPerTask));
    bucket_entry(experts, workers).launch_persistent(stream, active,
        ids, local_ids, counts, offsets, order, metadata, histogram, slots, first, active);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("precompile_router(int rows, int experts, int topk, int workers, int dp_stride=0, "
          "bool dp_int64=False) -> ()",
          [](int64_t rows, int64_t experts, int64_t topk, int64_t workers,
             int64_t dp_stride, bool dp_int64) {
              TORCH_CHECK(rows >= 0 && experts > 0 && experts % 64 == 0 && experts <= UINT32_MAX &&
                              topk > 0 && topk <= experts && topk <= router_layout::kMaxTopK && workers > 0 &&
                              workers <= native_host::core_count(c10_npu::current_device()) &&
                              dp_stride >= 0 && (!dp_stride || rows % dp_stride == 0),
                          "invalid actual router geometry");
              TORCH_CHECK(router_layout::arena_bytes(1, experts, topk) <= arch2201::kUbBytes,
                          "router UB resource budget exceeded");
              topk_entry({rows, dp_stride, uint32_t(experts), uint32_t(topk), uint32_t(workers),
                           dp_stride && dp_int64}).prepare();
          });
    m.def("precompile_route_bucket(int experts, int workers) -> ()",
          [](int64_t experts, int64_t workers) {
              TORCH_CHECK(experts > 0 && experts <= INT32_MAX &&
                              router_bucket::table_bytes(experts) <= arch2201::kUbBytes &&
                              workers > 0 && workers <= experts &&
                              workers <= native_host::core_count(c10_npu::current_device()),
                          "invalid route bucket worker plan");
              bucket_entry(uint32_t(experts), uint32_t(workers)).prepare();
          });
}
