#include "torch.h"
#include "launch.h"
#include "jit.h"
#include "params.h"

namespace native_host {
namespace {
void position_quant(const at::Tensor &x, const std::optional<at::Tensor> &cos,
                    const std::optional<at::Tensor> &sin, const at::Tensor &y,
                    const std::optional<at::Tensor> &scales, const at::Tensor &tables, int64_t mode,
                    bool inverse) {
    const auto device = x.device();
    check(x, at::kBFloat16, device);
    TORCH_CHECK_VALUE((x.dim() == 2 || x.dim() == 3) &&
                          (x.size(-1) > 0 && x.size(-1) <= position_quant::kWidthCapacityElements &&
                           x.size(-1) % 16 == 0),
                      "x must have positive DMA-aligned width within UB capacity");
    TORCH_CHECK_VALUE(bool(cos) == bool(sin) && mode >= 0 && mode <= 2, "invalid RoPE/quant mode");
    const auto rotary_dim = tables.numel() / 3;
    TORCH_CHECK_VALUE(tables.numel() % 3 == 0 && rotary_dim > 0 &&
                          rotary_dim <= position_quant::kRotaryCapacityElements && rotary_dim % 16 == 0,
                      "rotary dimension must be a positive multiple of16 within capacity64");
    TORCH_CHECK_VALUE(x.dim() != 3 || x.size(1) > 0, "heads must be positive");
    if (cos) {
        TORCH_CHECK_VALUE(x.size(-1) >= rotary_dim, "rotary span exceeds input width");
        check(*cos, at::kFloat, device, {x.size(0), rotary_dim / 2});
        check(*sin, at::kFloat, device, cos->sizes());
    }
    check(tables, at::kInt, device, {3 * rotary_dim});
    check(y, mode ? at::kChar : at::kBFloat16, device, x.sizes());
    if (!mode) {
        TORCH_CHECK_VALUE(cos && !scales, "RoPE requires trig tables without scales");
        TORCH_CHECK_VALUE(!overlaps(x, y) || x.data_ptr() == y.data_ptr(),
                          "partial in-place alias");
        disjoint({y}, {cos, sin, tables});
    } else {
        const auto h = x.dim() == 3 ? x.size(1) : 1;
        TORCH_CHECK_VALUE(h > 0 && scales, "invalid Indexer Q/K geometry");
        check(*scales, mode == 1 ? at::kHalf : at::kFloat, device, x.sizes().slice(0, x.dim() - 1));
        disjoint({y, scales}, {x, cos, sin, tables});
    }
    if (!x.numel())
        return;

    const int64_t blocks = core_count(x.get_device());
    const int64_t heads = x.dim() == 3 ? x.size(1) : 1;
    const int64_t dim = x.size(-1);
    const int64_t rows = x.numel() / dim;

    c10_npu::NPUGuard guard(x.device());
    auto stream = c10_npu::getCurrentNPUStream(x.get_device());
    track(x, stream);
    track(cos, stream);
    track(sin, stream);
    track(y, stream);
    track(scales, stream);
    track(tables, stream);
    dsv41_position_quant_launch(stream.stream(), u32(blocks), ptr(x), ptr(cos), ptr(sin), ptr(y),
                                ptr(scales), ptr(tables), rows, u32(heads), u32(dim), u32(mode),
                                inverse, u32(rotary_dim));
}
void cache_scatter(const at::Tensor &values, const std::optional<at::Tensor> &scales,
                   const at::Tensor &slots, const at::Tensor &cache,
                   const std::optional<at::Tensor> &cache_scales) {
    const auto device = values.device();
    const auto type = values.scalar_type();
    TORCH_CHECK_VALUE((values.dim() == 2 || (values.dim() == 3 && values.size(1) == 1)) &&
                          values.size(-1) > 0 &&
                          values.size(-1) <= position_quant::kWidthCapacityElements &&
                          ((type == at::kBFloat16 && values.size(-1) % 16 == 0) ||
                           (type == at::kChar && values.size(-1) % 32 == 0)),
                      "invalid cache scatter values");
    check(values, type, device);
    check(slots, at::kLong, device, {values.size(0)});
    check(cache, type, device);
    TORCH_CHECK_VALUE((cache.dim() == 2 || cache.dim() == 3) && cache.size(-1) == values.size(-1),
                      "invalid cache shape");
    TORCH_CHECK_VALUE(bool(scales) == bool(cache_scales), "scale_cache requires scales");
    if (scales) {
        check(*scales, at::kFloat, device);
        TORCH_CHECK_VALUE(scales->numel() == values.size(0), "invalid scale count");
        check(*cache_scales, at::kHalf, device, cache.sizes().slice(0, cache.dim() - 1));
    }
    disjoint({cache, cache_scales}, {values, slots, scales});
    if (!values.size(0))
        return;

    const int64_t blocks = core_count(values.get_device());
    const int64_t rows = values.size(0);
    const int64_t dim = values.size(-1);
    const int64_t capacity = cache.numel() / dim;
    const int64_t dtype = values.scalar_type() == at::kChar ? 0 : 1;

    c10_npu::NPUGuard guard(values.device());
    auto stream = c10_npu::getCurrentNPUStream(values.get_device());
    track(values, stream);
    track(scales, stream);
    track(slots, stream);
    track(cache, stream);
    track(cache_scales, stream);
    dsv41_cache_scatter_launch(stream.stream(), u32(blocks), ptr(values), ptr(scales), ptr(slots),
                               ptr(cache), ptr(cache_scales), rows, capacity, u32(dim), u32(dtype));
}
void rope_cache_gather(const at::Tensor &cos_cache, const at::Tensor &sin_cache,
                       const at::Tensor &positions, const std::optional<at::Tensor> &input_slots,
                       const at::Tensor &cos_out, const at::Tensor &sin_out,
                       const at::Tensor &valid_out, const std::optional<at::Tensor> &slots_out) {
    const auto device = cos_cache.device();
    check(cos_cache, at::kFloat, device);
    TORCH_CHECK_VALUE(cos_cache.dim() == 2 && cos_cache.size(1) > 0 &&
                          cos_cache.size(1) <= position_quant::kRotaryCapacityElements / 2 &&
                          cos_cache.size(1) % 8 == 0,
                      "trig columns must be DMA-aligned within rotary capacity");
    check(sin_cache, at::kFloat, device, cos_cache.sizes());
    check(positions, at::kLong, device);
    TORCH_CHECK_VALUE(positions.dim() == 1 && bool(input_slots) == bool(slots_out),
                      "invalid position/slot axes");
    if (input_slots) {
        check(*input_slots, at::kLong, device, positions.sizes());
        check(*slots_out, at::kLong, device, positions.sizes());
    }
    check(cos_out, at::kFloat, device, {positions.numel(), cos_cache.size(1)});
    check(sin_out, at::kFloat, device, cos_out.sizes());
    check(valid_out, at::kInt, device, positions.sizes());
    disjoint({cos_out, sin_out, valid_out, slots_out},
             {cos_cache, sin_cache, positions, input_slots});
    if (!positions.numel())
        return;

    const int64_t blocks = core_count(cos_cache.get_device());
    const int64_t rows = positions.numel();
    const int64_t cache_rows = cos_cache.size(0);

    c10_npu::NPUGuard guard(cos_cache.device());
    auto stream = c10_npu::getCurrentNPUStream(cos_cache.get_device());
    track(cos_cache, stream);
    track(sin_cache, stream);
    track(positions, stream);
    track(input_slots, stream);
    track(cos_out, stream);
    track(sin_out, stream);
    track(valid_out, stream);
    track(slots_out, stream);
    dsv41_rope_cache_gather_launch(stream.stream(), u32(blocks), ptr(cos_cache), ptr(sin_cache),
                                   ptr(positions), ptr(input_slots), ptr(cos_out), ptr(sin_out),
                                   ptr(valid_out), ptr(slots_out), rows, cache_rows,
                                   u32(cos_cache.size(1)));
}
void prepare_position_quant(const at::Tensor &x, const std::optional<at::Tensor> &cos,
                            const std::optional<at::Tensor> &sin, const at::Tensor &y,
                            const std::optional<at::Tensor> &scales, const at::Tensor &tables,
                            int64_t mode, bool inverse) {
    native_jit::PrepareOnlyScope scope;
    position_quant(x, cos, sin, y, scales, tables, mode, inverse);
}
void prepare_cache_scatter(const at::Tensor &values, const std::optional<at::Tensor> &scales,
                           const at::Tensor &slots, const at::Tensor &cache,
                           const std::optional<at::Tensor> &cache_scales) {
    native_jit::PrepareOnlyScope scope;
    cache_scatter(values, scales, slots, cache, cache_scales);
}
void prepare_rope_cache_gather(const at::Tensor &cos_cache, const at::Tensor &sin_cache,
                               const at::Tensor &positions,
                               const std::optional<at::Tensor> &input_slots,
                               const at::Tensor &cos_out, const at::Tensor &sin_out,
                               const at::Tensor &valid_out,
                               const std::optional<at::Tensor> &slots_out) {
    native_jit::PrepareOnlyScope scope;
    rope_cache_gather(cos_cache, sin_cache, positions, input_slots, cos_out, sin_out, valid_out,
                      slots_out);
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("position_quant(Tensor x, Tensor? cos, Tensor? sin, Tensor(a!) y, Tensor(a!)? scales, "
          "Tensor tables, int mode, bool inverse) -> ()",
          &position_quant);
    m.def("prepare_position_quant(Tensor x, Tensor? cos, Tensor? sin, Tensor(a!) y, Tensor(a!)? "
          "scales, "
          "Tensor tables, int mode, bool inverse) -> ()",
          &prepare_position_quant);
    m.def("cache_scatter(Tensor values, Tensor? scales, Tensor slots, Tensor(a!) cache, "
          "Tensor(a!)? cache_scales) -> ()",
          &cache_scatter);
    m.def("prepare_cache_scatter(Tensor values, Tensor? scales, Tensor slots, Tensor(a!) cache, "
          "Tensor(a!)? cache_scales) -> ()",
          &prepare_cache_scatter);
    m.def("rope_cache_gather(Tensor cos_cache, Tensor sin_cache, Tensor positions, Tensor? "
          "input_slots, Tensor(a!) cos_out, Tensor(a!) sin_out, Tensor(a!) valid_out, Tensor(a!)? "
          "slots_out) -> ()",
          &rope_cache_gather);
    m.def("prepare_rope_cache_gather(Tensor cos_cache, Tensor sin_cache, Tensor positions, Tensor? "
          "input_slots, Tensor(a!) cos_out, Tensor(a!) sin_out, Tensor(a!) valid_out, Tensor(a!)? "
          "slots_out) -> ()",
          &prepare_rope_cache_gather);
}
} // namespace native_host
