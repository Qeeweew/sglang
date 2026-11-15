#include "torch.h"
#include "launch.h"
#include "jit.h"

namespace native_host {
namespace {
std::vector<at::Tensor> metadata(const at::Tensor &cu, int64_t window, int64_t capacity,
                                 std::vector<at::Tensor> out) {
    check(cu, at::kInt, cu.device());
    TORCH_CHECK_VALUE(cu.dim() == 1 && cu.numel() >= 1, "cu must be [B+1]");
    const auto batch = cu.numel() - 1;
    TORCH_CHECK_VALUE(window >= 1 && window <= INT32_MAX, "invalid window");
    if (capacity == -1)
        capacity = batch * window;
    TORCH_CHECK_VALUE(capacity >= 0 && capacity <= INT32_MAX, "invalid capacity");
    c10_npu::NPUGuard guard(cu.device());
    if (out.empty())
        for (int64_t size : {capacity, batch + 1, capacity, int64_t(1)})
            out.push_back(at::empty({size}, cu.options()));
    TORCH_CHECK_VALUE(out.size() == 4, "metadata needs four output tensors");
    check(out[0], at::kInt, cu.device(), {capacity});
    check(out[1], at::kInt, cu.device(), {batch + 1});
    check(out[2], at::kInt, cu.device(), {capacity});
    check(out[3], at::kInt, cu.device(), {1});
    disjoint({out[0], out[1], out[2], out[3]}, {cu});
    auto stream = c10_npu::getCurrentNPUStream(cu.get_device());
    record(cu, stream);
    for (const auto &t : out)
        record(t, stream);
    dsv41_ced_metadata_launch(stream.stream(), ptr(cu), ptr(out[0]), ptr(out[2]), ptr(out[1]),
                              ptr(out[3]), u32(batch), u32(window), capacity);
    return out;
}
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
        dsv41_ced_copy_launch(stream.stream(), u32(core_count(source.get_device())), ptr(source),
                              ptr(row_map), ptr(out), rows, source.size(0), out.size(0), bytes,
                              scatter,
                              fill == -1 ? UINT8_MAX : 0);
    }
    return out;
}
} // namespace
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("prepare_ced_metadata(Tensor cu, int window, int capacity, Tensor[] out) -> ()",
          [](const at::Tensor &cu, int64_t window, int64_t capacity, std::vector<at::Tensor> out) {
              TORCH_CHECK(out.size() == 4, "prepare requires preallocated outputs");
              native_jit::PrepareOnlyScope prepare;
              metadata(cu, window, capacity, std::move(out));
          });
    m.def(
        "prepare_ced_copy(Tensor source, Tensor row_map, Tensor out, bool scatter, int fill) -> ()",
        [](const at::Tensor &source, const at::Tensor &row_map, const at::Tensor &out, bool scatter,
           int64_t fill) {
            native_jit::PrepareOnlyScope prepare;
            copy(source, row_map, out, scatter, fill);
        });
    m.def("ced_metadata(Tensor cu, int window, int capacity, Tensor(a!)[] out) -> Tensor(a!)[]",
          &metadata);
    m.def("ced_copy(Tensor source, Tensor row_map, Tensor(a!)? out, bool scatter, int fill) -> "
          "Tensor(a!)",
          &copy);
}
} // namespace native_host
