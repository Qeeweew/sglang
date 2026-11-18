#include "torch.h"
#include "launch.h"
#include "layout.h"
#include "bucket_layout.h"
#include "grouped_metadata.h"
#include "jit.h"
#include "workspace.h"
#include <limits>
namespace native_host {
namespace {
using namespace router_layout;
namespace route_metadata = grouped_metadata;
int64_t resolve_topk(int64_t experts, int64_t requested) {
    // Preserve serving defaults; arbitrary aligned expert counts pass topk.
    const auto topk = requested ? requested : (experts == 384 ? 6 : experts == 128 ? 3 : 0);
    TORCH_CHECK_VALUE(topk > 0 && topk <= experts && topk <= kMaxTopK, "specify topk in 1..min(experts,64)");
    TORCH_CHECK_VALUE(arena_bytes(1, experts, topk) <= arch2201::kUbBytes,
                      "router single-row UB resource budget exceeded");
    return topk;
}

std::vector<at::Tensor>
route(const at::Tensor &x, const std::optional<at::Tensor> &bias, double factor,
      const std::optional<at::Tensor> &valid_rows, const std::optional<at::Tensor> &dp_counts,
      int64_t dp_stride, const std::optional<at::Tensor> &alt,
      const std::optional<at::Tensor> &images, const std::optional<at::Tensor> &ids_out,
      const std::optional<at::Tensor> &weights_out, int64_t requested_topk) {
    const auto device = x.device();
    check(x, at::kFloat, device);
    TORCH_CHECK_VALUE(x.dim() == 2 && x.size(1) > 0 && x.size(1) % kExpertAlignment == 0,
                      "logits require [rows,experts] with one-repeat expert alignment");
    const int64_t m = x.size(0), e = x.size(1), k = resolve_topk(e, requested_topk),
                  cores = core_count(x.get_device());
    TORCH_CHECK_VALUE(std::isfinite(factor), "factor must be finite");
    if (bias)
        check(*bias, at::kFloat, device, {e});
    if (valid_rows)
        check(*valid_rows, at::kInt, device, {1});
    TORCH_CHECK_VALUE(bool(alt) == bool(images),
                      "alternate_bias and image_mask must be provided together");
    if (alt) {
        check(*alt, at::kFloat, device, {e});
        check(*images, at::kInt, device, {m});
    }
    c10_npu::NPUGuard guard(device);
    auto live = valid_rows;
    if (dp_counts) {
        TORCH_CHECK_VALUE(
            !valid_rows && dp_stride > 0 && dp_counts->dim() == 1 &&
                (dp_counts->scalar_type() == at::kInt || dp_counts->scalar_type() == at::kLong) &&
                dp_counts->numel() * dp_stride == m && m <= INT32_MAX,
            "invalid DP counts/stride geometry");
        check(*dp_counts, dp_counts->scalar_type(), device);
    } else
        TORCH_CHECK_VALUE(dp_stride == 0, "DP stride requires DP counts");
    auto allocate = [&](const std::optional<at::Tensor> &value, at::ScalarType type,
                        at::IntArrayRef shape) {
        auto t = value ? *value : at::empty(shape, x.options().dtype(type));
        check(t, type, device, shape);
        return t;
    };
    auto ids = allocate(ids_out, at::kInt, {m, k}), w = allocate(weights_out, at::kFloat, {m, k});
    disjoint({ids, w}, {x, bias, live, alt, images, dp_counts});
    auto stream = c10_npu::getCurrentNPUStream(x.get_device());
    for (const auto &t : {x, ids, w})
        record(t, stream);
    for (const auto &t : {bias, live, alt, images, dp_counts})
        track(t, stream);
    router::Sources sources{ptr(x),    ptr(bias), ptr(ids),    ptr(w),
                            ptr(live), ptr(alt),  ptr(images), ptr(dp_counts)};
    const router::Plan geometry{m,          dp_stride,
                                u32(e),     u32(k),
                                u32(cores), dp_counts && dp_counts->scalar_type() == at::kLong};
    if (m)
        dsv41_router_launch(stream.stream(), sources, geometry, f32(factor));
    return {ids, w};
}
void bucket(const at::Tensor &ids, const at::Tensor &local_ids, const at::Tensor &counts,
            const at::Tensor &offsets, const at::Tensor &order, const at::Tensor &metadata,
            const std::optional<at::Tensor> &histogram, int64_t first) {
    const auto device = ids.device();
    check(ids, at::kInt, device);
    check(counts, at::kInt, device);
    TORCH_CHECK_VALUE(ids.dim() == 2 && ids.size(1) > 0 && counts.dim() == 1 &&
                          counts.numel() > 0 && first >= 0 &&
                          first <= INT32_MAX - counts.numel() &&
                          router_bucket::table_bytes(counts.numel()) <= arch2201::kUbBytes &&
                          ids.numel() <= std::numeric_limits<int32_t>::max(),
                      "invalid local route bucket geometry");
    check(local_ids, at::kInt, device, ids.sizes());
    check(offsets, at::kInt, device, {counts.numel() + 1});
    check(order, at::kInt, device, {ids.numel()});
    check(metadata, at::kInt, device, {route_metadata::kWords * (counts.numel() + 1)});
    c10_npu::NPUGuard guard(device);
    const uint32_t workers = std::min<int64_t>(core_count(ids.get_device()), counts.numel());
    const int64_t stride = router_bucket::histogram_stride(counts.numel());
    if (histogram)
        check(*histogram, at::kInt, device, {workers, stride});
    disjoint({local_ids, counts, offsets, order, metadata, histogram}, {ids});
    auto stream = c10_npu::getCurrentNPUStream(ids.get_device());
    const auto acl_stream = stream.stream();
    const LaunchWorkspace scratch(histogram, workers * stride * sizeof(int32_t), acl_stream);
    for (const auto &tensor : {ids, local_ids, counts, offsets, order, metadata})
        record(tensor, stream);
    track(histogram, stream);
    dsv41_route_bucket(acl_stream, workers, ids.data_ptr(), local_ids.data_ptr(),
                       counts.data_ptr(), offsets.data_ptr(), order.data_ptr(), metadata.data_ptr(),
                       scratch.data(), ids.numel(), int32_t(first), uint32_t(counts.numel()));
}

} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("prepare_route_bucket(Tensor ids, Tensor local_ids, Tensor counts, Tensor offsets, "
          "Tensor order, Tensor metadata, Tensor? histogram, int first) -> ()",
          [](const at::Tensor &ids, const at::Tensor &local_ids, const at::Tensor &counts,
             const at::Tensor &offsets, const at::Tensor &order, const at::Tensor &metadata,
             const std::optional<at::Tensor> &histogram,
             int64_t first) {
              native_jit::PrepareOnlyScope prepare;
              bucket(ids, local_ids, counts, offsets, order, metadata, histogram, first);
          });
    m.def("route_bucket(Tensor ids, Tensor(a!) local_ids, Tensor(b!) counts, Tensor(c!) offsets, "
          "Tensor(d!) order, Tensor(e!) metadata, Tensor(f!)? histogram, int first) -> ()", &bucket);

    m.def("router(Tensor logits, Tensor? bias, float factor, Tensor? valid_rows, Tensor? dp_counts, "
          "int dp_stride, Tensor? alternate_bias, Tensor? image_mask, Tensor(a!)? ids_out, "
          "Tensor(b!)? weights_out, int topk=0) -> Tensor[]", &route);
    m.def("prepare_router(Tensor logits, Tensor? bias, float factor, Tensor? valid_rows, Tensor? dp_counts, "
          "int dp_stride, Tensor? alternate_bias, Tensor? image_mask, Tensor(a!)? ids_out, "
          "Tensor(b!)? weights_out, int topk=0) -> Tensor[]",
          [](const at::Tensor &x, const std::optional<at::Tensor> &bias, double factor,
             const std::optional<at::Tensor> &live, const std::optional<at::Tensor> &dp,
             int64_t stride, const std::optional<at::Tensor> &alternate,
             const std::optional<at::Tensor> &images, const std::optional<at::Tensor> &ids,
             const std::optional<at::Tensor> &weights, int64_t topk) {
              TORCH_CHECK(ids && weights, "prepare requires preallocated TopK outputs");
              native_jit::PrepareOnlyScope prepare;
              return route(x, bias, factor, live, dp, stride, alternate, images, ids, weights, topk);
          });
}
} // namespace native_host
