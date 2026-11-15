#pragma once
#include <cstdint>
namespace router {
struct Sources {
    uint8_t *logits, *bias, *ids, *weights, *live, *alternate, *images, *dp_counts;
};
struct Plan {
    int64_t rows, dp_stride;
    uint32_t experts, topk, workers;
    bool dp_int64;
};
} // namespace router
extern "C" void dsv41_router_launch(void *, const router::Sources &, const router::Plan &, float);
extern "C" void dsv41_route_bucket(void *, uint32_t, void *, void *, void *, void *, void *, void *,
                                   void *, int64_t, int32_t, uint32_t);
