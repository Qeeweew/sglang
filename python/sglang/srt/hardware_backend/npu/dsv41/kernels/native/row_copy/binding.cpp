#include "torch.h"
#include "launch.h"
#include "jit.h"
#include <numeric>

namespace native_host {
namespace {
at::Tensor copy(const at::Tensor &source, const at::Tensor &row_map,
                const std::optional<at::Tensor> &output, bool scatter, int64_t fill) {
    const auto device = source.device();
    const auto type = source.scalar_type();
    TORCH_CHECK_VALUE(type == at::kBFloat16 || type == at::kFloat || type == at::kInt ||
                          type == at::kLong,
                      "unsupported row dtype");
    check(source, type, device);
    check(row_map, at::kInt, device);
    TORCH_CHECK_VALUE(source.dim() >= 1 && row_map.dim() == 1, "row axes required");
    const auto rows = row_map.numel();
    TORCH_CHECK_VALUE(!scatter || (source.size(0) == rows && output),
                      "scatter requires map_rows updates and output");
    TORCH_CHECK_VALUE(fill == 0 || (fill == -1 && (type == at::kInt || type == at::kLong)),
                      "fill=-1 is only for integer metadata");
    c10_npu::NPUGuard guard(device);
    at::Tensor out;
    if (output)
        out = *output;
    else {
        auto shape = source.sizes().vec();
        shape[0] = rows;
        out = at::empty(shape, source.options());
    }
    check(out, type, device);
    TORCH_CHECK_VALUE(out.dim() == source.dim() &&
                          out.sizes().slice(1) == source.sizes().slice(1) &&
                          (scatter || out.size(0) == rows),
                      "row shape mismatch");
    disjoint({out}, {source, row_map});
    const auto bytes =
        source.element_size() * std::accumulate(source.sizes().begin() + 1, source.sizes().end(),
                                                int64_t(1), std::multiplies<int64_t>());
    if (rows && bytes) {
        auto stream = c10_npu::getCurrentNPUStream(source.get_device());
        for (const auto &t : {source, row_map, out})
            record(t, stream);
        dsv41_row_copy_launch(stream.stream(), u32(core_count(source.get_device())), ptr(source),
                              ptr(row_map), ptr(out), rows, source.size(0), out.size(0), bytes,
                              scatter,
                              fill == -1 ? UINT8_MAX : 0);
    }
    return out;
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("gather_rows(Tensor source, Tensor row_map, Tensor(a!)? out, int fill) -> Tensor(a!)",
          [](const at::Tensor &source, const at::Tensor &row_map,
             const std::optional<at::Tensor> &out, int64_t fill) {
              return copy(source, row_map, out, false, fill);
          });
    m.def("scatter_rows(Tensor updates, Tensor row_map, Tensor(a!) out) -> Tensor(a!)",
          [](const at::Tensor &updates, const at::Tensor &row_map, const at::Tensor &out) {
              return copy(updates, row_map, out, true, 0);
          });
    m.def("prepare_gather_rows(Tensor source, Tensor row_map, Tensor out, int fill) -> ()",
          [](const at::Tensor &source, const at::Tensor &row_map, const at::Tensor &out,
             int64_t fill) {
              native_jit::PrepareOnlyScope prepare;
              copy(source, row_map, out, false, fill);
          });
    m.def("prepare_scatter_rows(Tensor updates, Tensor row_map, Tensor out) -> ()",
          [](const at::Tensor &updates, const at::Tensor &row_map, const at::Tensor &out) {
              native_jit::PrepareOnlyScope prepare;
              copy(updates, row_map, out, true, 0);
          });
}
} // namespace native_host
