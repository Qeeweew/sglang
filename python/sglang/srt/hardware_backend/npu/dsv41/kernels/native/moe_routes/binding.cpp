#include "torch.h"
#include "launch.h"
#include "jit.h"
#include <common/grouped_metadata.h>
namespace native_host {
namespace {
namespace route_metadata = grouped_metadata;
void gather_rows(const at::Tensor &x, const at::Tensor &order, const at::Tensor &metadata,
            const at::Tensor &sorted, const at::Tensor &inverse, int64_t topk,
            const std::optional<at::Tensor> &scales, const std::optional<at::Tensor> &sorted_scales) {
    const auto device = x.device();
    check(x, scales ? at::kChar : at::kBFloat16, device);
    TORCH_CHECK_VALUE(x.dim() == 2 && topk > 0, "invalid route gather shape");
    check(order, at::kInt, device, {x.size(0) * topk});
    check(metadata, at::kInt, device);
    TORCH_CHECK_VALUE(metadata.dim() == 1 && metadata.numel() >= 2 * route_metadata::kWords &&
                          metadata.numel() % route_metadata::kWords == 0,
                      "invalid grouped metadata");
    check(sorted, scales ? at::kChar : at::kHalf, device);
    // metadata[1] is the device-side valid prefix, bounded by this capacity.
    // The caller obtains exact eager EP counts or supplies a safe Graph bound.
    TORCH_CHECK_VALUE(sorted.dim() == 2 && sorted.size(1) == x.size(1) &&
                          sorted.size(0) <= x.size(0) * topk,
                      "invalid compact gather capacity");
    check(inverse, at::kInt, device, {x.size(0) * topk});
    TORCH_CHECK_VALUE(bool(scales) == bool(sorted_scales), "missing scale buffer");
    if (scales) {
        TORCH_CHECK_VALUE(x.size(1) > 0 && x.size(1) <= 8192 && x.size(1) % 32 == 0, "invalid A8 width");
        check(*scales, at::kFloat, device, {x.size(0)});
        check(*sorted_scales, at::kFloat, device, {sorted.size(0)});
    }
    disjoint({sorted, inverse, sorted_scales}, {x, order, metadata, scales});
    if (!x.size(0))
        return;

    const int64_t blocks = core_count(x.get_device());

    c10_npu::NPUGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream(x.get_device());
    for (const auto &t : {x, order, metadata, sorted, inverse})
        record(t, stream);
    if (scales) {
        track(scales, stream);
        track(sorted_scales, stream);
        dsv41_route_gather_a8(stream.stream(), blocks, ptr(x), ptr(scales), ptr(order), ptr(metadata),
                              ptr(sorted), ptr(sorted_scales), ptr(inverse), sorted.size(0), x.size(1), topk);
        return;
    }
    dsv41_route_gather(stream.stream(), blocks, x.data_ptr(), order.data_ptr(), metadata.data_ptr(),
                       sorted.data_ptr(), inverse.data_ptr(), x.size(0), x.size(1), topk,
                       sorted.size(0));
}
void gather(const at::Tensor &x, const at::Tensor &order, const at::Tensor &metadata,
            const at::Tensor &sorted, const at::Tensor &inverse, int64_t topk) {
    gather_rows(x, order, metadata, sorted, inverse, topk, {}, {});
}
void combine(const at::Tensor &values, const at::Tensor &inverse, const at::Tensor &ids,
             const at::Tensor &weights, const at::Tensor &output) {
    const auto device = values.device();
    const bool bf16 = values.scalar_type() == at::kBFloat16;
    check(values, bf16 ? at::kBFloat16 : at::kHalf, device);
    check(ids, at::kInt, device);
    TORCH_CHECK_VALUE(ids.dim() == 2 && values.dim() == 2 && values.size(0) <= ids.numel(),
                      "invalid route combine shape");
    check(inverse, at::kInt, device, {ids.numel()});
    check(weights, at::kFloat, device, ids.sizes());
    check(output, at::kBFloat16, device, {ids.size(0), values.size(1)});
    disjoint({output}, {values, inverse, ids, weights});
    if (!ids.size(0))
        return;

    const int64_t blocks = core_count(values.get_device());

    c10_npu::NPUGuard guard(values.device());
    auto stream = c10_npu::getCurrentNPUStream(values.get_device());
    for (const auto &t : {values, inverse, ids, weights, output})
        record(t, stream);
    dsv41_route_combine(stream.stream(), blocks, values.data_ptr(), inverse.data_ptr(),
                        ids.data_ptr(), weights.data_ptr(), output.data_ptr(), output.size(0),
                        output.size(1), ids.size(1), bf16);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("route_gather_a8(Tensor x, Tensor order, Tensor metadata, Tensor(a!) sorted, Tensor(b!) inverse, "
          "int topk, Tensor scales, Tensor(c!) sorted_scales) -> ()",
          [](const at::Tensor &x, const at::Tensor &order, const at::Tensor &metadata, const at::Tensor &sorted,
             const at::Tensor &inverse, int64_t topk, const at::Tensor &scales, const at::Tensor &sorted_scales) {
              gather_rows(x, order, metadata, sorted, inverse, topk, scales, sorted_scales);
          });
    m.def("prepare_route_gather(Tensor x, Tensor order, Tensor metadata, Tensor sorted, Tensor "
          "inverse, int topk) -> ()",
          [](const at::Tensor &x, const at::Tensor &order, const at::Tensor &metadata,
             const at::Tensor &sorted, const at::Tensor &inverse, int64_t topk) {
              native_jit::PrepareOnlyScope prepare;
              gather(x, order, metadata, sorted, inverse, topk);
          });
    m.def("prepare_route_combine(Tensor values, Tensor inverse, Tensor ids, Tensor weights, Tensor "
          "output) -> ()",
          [](const at::Tensor &values, const at::Tensor &inverse, const at::Tensor &ids,
             const at::Tensor &weights, const at::Tensor &output) {
              native_jit::PrepareOnlyScope prepare;
              combine(values, inverse, ids, weights, output);
          });
    m.def("route_gather(Tensor x, Tensor order, Tensor metadata, Tensor(a!) sorted, Tensor(b!) "
          "inverse, int topk) -> ()",
          &gather);
    m.def("route_combine(Tensor values, Tensor inverse, Tensor ids, Tensor weights, Tensor(a!) "
          "output) -> ()",
          &combine);
}
} // namespace native_host
