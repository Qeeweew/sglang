#include "launch.h"
#include "torch.h"

namespace h32_quant {
using namespace native_host;
void run(const at::Tensor &input, const at::Tensor &codes, const at::Tensor &scales,
         bool interleaved, const std::optional<at::Tensor> &count) {
    const auto device = input.device();
    TORCH_CHECK_VALUE(input.dim() == 2 && (input.scalar_type() == at::kHalf ||
                                           input.scalar_type() == at::kBFloat16),
                      "expected FP16/BF16 matrix");
    const auto rows = input.size(0), width = input.size(1);
    TORCH_CHECK_VALUE(valid_width(width), "invalid H32 width");
    check(input, input.scalar_type(), device);
    check(codes, at::kChar, device, {rows, width});
    check(scales, at::kFloat, device, {rows});
    if (count)
        check(*count, at::kInt, device, {1});
    disjoint({codes, scales}, {input, count});
    if (!rows)
        return;
    c10_npu::NPUGuard guard(device);
    auto stream = c10_npu::getCurrentNPUStream(device.index());
    for (const auto &tensor : {input, codes, scales})
        record(tensor, stream);
    track(count, stream);
    const uint64_t workers = core_count(device.index());
    const bool bf16 = input.scalar_type() == at::kBFloat16;
    launch(stream.stream(), workers, width, bf16, interleaved, ptr(input), ptr(codes),
           ptr(scales), ptr(count), rows);
}
TORCH_LIBRARY_FRAGMENT(dsv41, m) {
    m.def("h32_quant(Tensor input, Tensor(a!) codes, Tensor(b!) scales, bool "
          "interleaved, Tensor? count=None) -> ()",
          &run);
}
} // namespace h32_quant
